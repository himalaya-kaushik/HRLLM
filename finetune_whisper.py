#!/usr/bin/env python3
"""
Fine-tune openai/whisper-medium on Haryanvi/Bangru speech using LoRA.

Training : ankitdhiman/haryanvi-tts  (2768 clips, speaker A)
Test     : bridgeconn/snow-mountain  haryanvi subset (60 clips, speaker B)
Hardware : Kaggle T4 16GB VRAM / 30GB RAM
"""

# ── env vars must be set before any HF import ────────────────────────────────
import os, sys

os.environ.update({
    "HF_HOME":                f"/kaggle/working/hf_cache",
    "TRANSFORMERS_CACHE":     f"/kaggle/working/hf_cache",
    "HF_DATASETS_CACHE":      f"/kaggle/working/hf_cache/datasets",
    "WANDB_DISABLED":         "true",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
})

import argparse
import csv
import gc
import io
import json
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import torch
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
WORK        = Path("/kaggle/working")
TTS_DIR     = WORK / "data" / "haryanvi-tts"
SNOW_DIR    = WORK / "data" / "snow-mountain-test"
CKPT_DIR    = WORK / "checkpoints"
ADAPTER_DIR = WORK / "adapter"
RESULTS_DIR = WORK / "results"
CACHE_DIR   = WORK / "cache"

# ── constants ─────────────────────────────────────────────────────────────────
REPO_TTS   = "ankitdhiman/haryanvi-tts"
REPO_SNOW  = "bridgeconn/snow-mountain"
LANGUAGE   = "hi"
TARGET_SR  = 16_000
MAX_TEST   = 60

LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.1
LORA_TARGETS = ["q_proj", "v_proj"]

MAX_STEPS    = 1000
LR           = 1e-4
WARMUP_STEPS = 100
EVAL_STEPS   = 200
LOG_STEPS    = 25
GEN_MAX_LEN  = 225


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _resample(arr: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_SR:
        return arr
    try:
        from scipy.signal import resample_poly
        g = gcd(orig_sr, TARGET_SR)
        return resample_poly(arr, TARGET_SR // g, orig_sr // g).astype(np.float32)
    except ImportError:
        n = int(len(arr) * TARGET_SR / orig_sr)
        return np.interp(
            np.linspace(0, len(arr) - 1, n), np.arange(len(arr)), arr
        ).astype(np.float32)


def load_audio(path: str) -> np.ndarray:
    arr, sr = sf.read(path)
    arr = arr.astype(np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return _resample(arr, sr)


# ══════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD — TTS TRAINING SET
# ══════════════════════════════════════════════════════════════════════════════

def download_tts() -> list[dict]:
    from huggingface_hub import snapshot_download

    meta = TTS_DIR / "metadata.csv"
    if not meta.exists():
        TTS_DIR.mkdir(parents=True, exist_ok=True)
        print("  Downloading haryanvi-tts …")
        snapshot_download(repo_id=REPO_TTS, repo_type="dataset",
                          local_dir=str(TTS_DIR))
        print("  Done.")

    rows = []
    with open(meta, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = TTS_DIR / r["file_name"]
            if p.exists() and r.get("text", "").strip():
                rows.append({"audio_path": str(p), "text": r["text"].strip()})
    print(f"  TTS rows loaded: {len(rows)}")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD — SNOW-MOUNTAIN TEST SET (streaming tar.gz)
# ══════════════════════════════════════════════════════════════════════════════

def _proportional_select(rows: list[dict], n: int) -> list[dict]:
    """Largest-remainder proportional selection across Bible books."""
    book_idx: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        book_idx[r["path"].split("/")[-2]].append(i)

    total = len(rows)
    books = sorted(book_idx)
    ideal = {b: len(book_idx[b]) / total * n for b in books}
    alloc = {b: int(ideal[b]) for b in books}
    gap   = n - sum(alloc.values())
    for b in sorted(books, key=lambda b: ideal[b] - alloc[b], reverse=True)[:gap]:
        alloc[b] += 1

    sel: list[int] = []
    for b in books:
        sel.extend(book_idx[b][: alloc[b]])
    sel.sort()
    return [rows[i] for i in sel]


def _stream_book(book: str, needed: set[str], fname_map: dict,
                 audio_dir: Path) -> int:
    """Stream one book's tar.gz from HF; extract only the wav files in needed."""
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    url = hf_hub_url(
        repo_id=REPO_SNOW,
        filename=f"data/cleaned/haryanvi/{book}.tar.gz",
        repo_type="dataset",
    )
    try:
        resp = requests.get(url, stream=True,
                            headers=build_hf_headers(), timeout=180)
        resp.raise_for_status()
        resp.raw.decode_content = True
    except Exception as e:
        tqdm.write(f"  ERROR opening {book}.tar.gz: {e}")
        return 0

    remaining = set(needed)
    extracted = 0
    try:
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
            for member in tar:
                fname = os.path.basename(member.name)
                if fname not in remaining:
                    continue
                f = tar.extractfile(member)
                if f is None:
                    remaining.discard(fname)
                    continue
                try:
                    arr, sr = sf.read(io.BytesIO(f.read()))
                except Exception as e:
                    tqdm.write(f"  WARN {fname}: {e}")
                    remaining.discard(fname)
                    continue
                arr = arr.astype(np.float32)
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
                arr = _resample(arr, sr)
                out = audio_dir / fname_map[fname]["out_fname"]
                sf.write(str(out), arr, TARGET_SR, subtype="PCM_16")
                remaining.discard(fname)
                extracted += 1
                if not remaining:
                    break
    except Exception as e:
        tqdm.write(f"  ERROR streaming {book}.tar.gz: {e}")
    finally:
        resp.close()
    return extracted


def download_snow_mountain() -> list[dict]:
    from huggingface_hub import hf_hub_download

    audio_dir = SNOW_DIR / "audio"
    meta_path = SNOW_DIR / "metadata.csv"

    if meta_path.exists():
        rows = list(csv.DictReader(open(meta_path, encoding="utf-8")))
        print(f"  Snow-mountain test cached: {len(rows)} clips")
        return [{"audio_path": str(SNOW_DIR / r["file_name"]),
                 "text": r["text"]} for r in rows]

    audio_dir.mkdir(parents=True, exist_ok=True)

    print("  Fetching test_common.csv …")
    csv_path = hf_hub_download(
        repo_id=REPO_SNOW,
        filename="data/experiments/haryanvi/test_common.csv",
        repo_type="dataset",
    )
    all_rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    rows = _proportional_select(all_rows, MAX_TEST)
    print(f"  Selected {len(rows)} clips (proportional across books)")

    fname_map: dict[str, dict] = {}
    for i, r in enumerate(rows):
        wb = os.path.basename(r["path"])
        fname_map[wb] = {
            "out_fname": f"haryanvi_{i+1:04d}.wav",
            "text":      r["sentence"].strip(),
            "book":      r["path"].split("/")[-2],
        }

    by_book: dict[str, set[str]] = defaultdict(set)
    for wb, info in fname_map.items():
        by_book[info["book"]].add(wb)

    total = 0
    for book in tqdm(sorted(by_book), desc="  Streaming books", unit="book"):
        total += _stream_book(book, by_book[book], fname_map, audio_dir)
    print(f"  Extracted {total}/{len(rows)} clips")

    meta_rows = []
    for r in rows:
        wb  = os.path.basename(r["path"])
        out = audio_dir / fname_map[wb]["out_fname"]
        if out.exists() and out.stat().st_size > 0:
            meta_rows.append({
                "file_name": f"audio/{fname_map[wb]['out_fname']}",
                "text":      r["sentence"].strip(),
            })
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "text"])
        w.writeheader()
        w.writerows(meta_rows)

    return [{"audio_path": str(SNOW_DIR / r["file_name"]), "text": r["text"]}
            for r in meta_rows]


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (RAM-safe: disk-cached, one example at a time)
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(audio_rows: list[dict], processor, cache_file: str):
    from datasets import Dataset

    def process(ex):
        arr = load_audio(ex["audio_path"])
        feats = processor.feature_extractor(
            arr, sampling_rate=TARGET_SR, return_tensors="np"
        ).input_features[0]
        labels = processor.tokenizer(ex["text"]).input_ids
        return {"input_features": feats, "labels": labels}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if os.path.exists(cache_file):
        os.remove(cache_file)   # always bust cache — prevents mel-channel mismatch after model switch
    ds = Dataset.from_list(audio_rows)
    ds = ds.map(
        process,
        remove_columns=["audio_path", "text"],
        desc="  Features",
        num_proc=1,
        batched=False,
        writer_batch_size=50,
        cache_file_name=cache_file,
    )
    ds.set_format("numpy")
    gc.collect()
    return ds


# ══════════════════════════════════════════════════════════════════════════════
# DATA COLLATOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpeechCollator:
    processor: object

    def __call__(self, features: list) -> dict:
        in_feats = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(in_feats, return_tensors="pt")

        lbl_feats = [{"input_ids": f["labels"]} for f in features]
        lbl_batch = self.processor.tokenizer.pad(lbl_feats, return_tensors="pt")
        labels = lbl_batch["input_ids"].masked_fill(
            lbl_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _jiwer_transforms():
    import jiwer
    return jiwer.transforms.Compose([
        jiwer.transforms.Strip(),
        jiwer.transforms.ReduceToListOfListOfWords(),
    ])


def make_compute_metrics(processor):
    import jiwer
    tfm = _jiwer_transforms()

    def compute_metrics(pred):
        pred_ids = pred.predictions
        lbl_ids  = pred.label_ids.copy()
        lbl_ids[lbl_ids == -100] = processor.tokenizer.pad_token_id
        hyp = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref = processor.tokenizer.batch_decode(lbl_ids,  skip_special_tokens=True)
        wer = jiwer.wer(ref, hyp, reference_transform=tfm, hypothesis_transform=tfm)
        return {"wer": round(wer, 4)}

    return compute_metrics


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, processor, rows: list[dict],
             device: str, batch_size: int = 8) -> tuple[list, float, float]:
    import jiwer
    tfm = _jiwer_transforms()

    # Clear suppress_tokens conflict (transformers 4.47 raises ValueError if they diverge)
    model.config.suppress_tokens            = None
    model.generation_config.suppress_tokens = None
    # Do NOT set language/task on generation_config alongside forced_decoder_ids.
    # WhisperForConditionalGeneration._retrieve_init_tokens() silently discards
    # forced_decoder_ids when task is also set, then re-derives from language/task —
    # this re-derivation is unreliable on GPU fp16 and is the root cause of 97% WER.
    model.generation_config.language = None
    model.generation_config.task      = None
    model.generation_config.max_length = None   # kills the "max_length=20" HF default warning
    if hasattr(model.config, "max_length"):
        model.config.max_length = None
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task="transcribe")
    model.eval()

    preds, refs = [], []
    for i in tqdm(range(0, len(rows), batch_size), desc="  Eval", leave=False):
        batch  = rows[i : i + batch_size]
        arrays = [load_audio(r["audio_path"]) for r in batch]
        inputs = processor.feature_extractor(
            arrays, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
        ).input_features.to(device).half()

        with torch.no_grad():
            ids = model.generate(
                input_features=inputs,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=GEN_MAX_LEN,
            )
        preds.extend(processor.tokenizer.batch_decode(ids, skip_special_tokens=True))
        refs.extend(r["text"] for r in batch)

    wer = round(jiwer.wer(refs, preds, reference_transform=tfm, hypothesis_transform=tfm), 4)
    cer = round(jiwer.cer(refs, preds), 4)
    return preds, wer, cer


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def save_results(test_rows, base_preds, ft_preds,
                 base_wer, base_cer, ft_wer, ft_cer,
                 log_history, model_name):
    import jiwer
    tfm = _jiwer_transforms()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rel = round((base_wer - ft_wer) / base_wer * 100, 1) if base_wer else None

    with open(RESULTS_DIR / "final_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "lora": {"r": LORA_R, "alpha": LORA_ALPHA,
                     "targets": LORA_TARGETS, "dropout": LORA_DROPOUT},
            "max_steps": MAX_STEPS, "lr": LR,
            "test_clips": len(test_rows),
            "baseline":  {"wer": base_wer, "cer": base_cer},
            "finetuned": {"wer": ft_wer,   "cer": ft_cer},
            "wer_relative_improvement_pct": rel,
        }, f, ensure_ascii=False, indent=2)

    clip_rows = []
    with open(RESULTS_DIR / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "gold_text",
                                           "predicted_text", "clip_wer"])
        w.writeheader()
        for row, pred in zip(test_rows, ft_preds):
            cw = round(jiwer.wer([row["text"]], [pred],
                                  reference_transform=tfm,
                                  hypothesis_transform=tfm), 4)
            clip_rows.append((cw, row, pred))
            w.writerow({
                "file_name":      Path(row["audio_path"]).name,
                "gold_text":      row["text"],
                "predicted_text": pred,
                "clip_wer":       cw,
            })

    with open(RESULTS_DIR / "training_log.json", "w") as f:
        json.dump(log_history, f, indent=2)

    worst = sorted(clip_rows, key=lambda x: x[0], reverse=True)[:10]
    best  = sorted(clip_rows, key=lambda x: x[0])[:10]

    def tbl(clips):
        lines = ["| WER | Gold | Prediction |", "|-----|------|------------|"]
        for cw, row, pred in clips:
            lines.append(f"| {cw:.0%} | {row['text']} | {pred} |")
        return "\n".join(lines)

    report = f"""# Haryanvi/Bangru Whisper Fine-tune — Results

## Setup
- **Base model**: `{model_name}`
- **LoRA**: r={LORA_R}, α={LORA_ALPHA}, targets: {", ".join(LORA_TARGETS)}
- **Steps**: {MAX_STEPS} | **LR**: {LR} | **Language**: hi / transcribe
- **Test set**: snow-mountain haryanvi, {len(test_rows)} clips (Bible domain, speaker B)

## Results

| Metric | Baseline | Fine-tuned | Δ |
|--------|----------|------------|---|
| **WER** | {base_wer:.1%} | {ft_wer:.1%} | {rel:+.1f}% |
| **CER** | {base_cer:.1%} | {ft_cer:.1%} | — |

## 10 Worst Clips
{tbl(worst)}

## 10 Best Clips
{tbl(best)}
"""
    (RESULTS_DIR / "report.md").write_text(report, encoding="utf-8")

    print(f"  final_results.json  WER {base_wer:.1%} → {ft_wer:.1%}  ({rel:+.1f}%)")
    print(f"  predictions.csv     {len(clip_rows)} clips")
    print(f"  report.md")
    print(f"  training_log.json   {len(log_history)} entries")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # single T4; DataParallel broadcast OOMs on Kaggle x2

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-medium")
    parser.add_argument("--smoke", action="store_true",
                        help="Baseline eval only: load model, evaluate test set, print WER, exit.")
    args   = parser.parse_args()
    MODEL  = args.model

    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    from peft import LoraConfig, TaskType, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. data ───────────────────────────────────────────────────────────────
    print("\n" + "="*62)
    print("STEP 1 — Download data")
    print("="*62)
    tts_rows  = download_tts()
    snow_rows = download_snow_mountain()
    if not snow_rows:
        print("WARNING: no snow-mountain clips — eval will be skipped")

    # ── 2. processor + features ───────────────────────────────────────────────
    print("\n" + "="*62)
    print("STEP 2 — Processor + feature extraction")
    print("="*62)
    processor = WhisperProcessor.from_pretrained(
        MODEL, language=LANGUAGE, task="transcribe"
    )
    print(f"  Building train dataset ({len(tts_rows)} clips) …")
    train_ds = build_dataset(
        tts_rows, processor,
        str(CACHE_DIR / "train_features.arrow"),
    )
    eval_ds = None
    if snow_rows:
        print(f"  Building eval dataset ({len(snow_rows)} clips) …")
        eval_ds = build_dataset(
            snow_rows, processor,
            str(CACHE_DIR / "eval_features.arrow"),
        )
    print(f"  train={len(train_ds)}  eval={len(eval_ds) if eval_ds else 0}")

    # ── 3. model + LoRA ───────────────────────────────────────────────────────
    print("\n" + "="*62)
    print("STEP 3 — Load model + apply LoRA")
    print("="*62)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, low_cpu_mem_usage=True
    )
    model.config.use_cache          = False
    model.config.suppress_tokens    = None
    model.generation_config.suppress_tokens = None
    # Set forced_decoder_ids directly — do NOT also set language/task, which would
    # cause Whisper to silently discard forced_decoder_ids (_retrieve_init_tokens).
    _forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task="transcribe")
    model.generation_config.forced_decoder_ids = _forced_decoder_ids
    model.generation_config.language           = None   # must stay None
    model.generation_config.task               = None   # must stay None
    model.generation_config.max_new_tokens     = GEN_MAX_LEN
    model.generation_config.max_length         = None

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ── 4. baseline eval ──────────────────────────────────────────────────────
    base_preds, base_wer, base_cer = [], None, None
    if snow_rows:
        print("\n" + "="*62)
        print("STEP 4 — Baseline evaluation (LoRA B=0 ≡ vanilla model)")
        print("="*62)
        model.half().to(device)
        base_preds, base_wer, base_cer = evaluate(
            model, processor, snow_rows, device
        )
        print(f"  Baseline  WER={base_wer:.1%}  CER={base_cer:.1%}")
        print(f"  Sample predictions (first 5):")
        for row, pred in zip(snow_rows[:5], base_preds[:5]):
            print(f"    ref : {row['text']}")
            print(f"    pred: {pred}")
        if args.smoke:
            print("\n  --smoke: baseline eval complete. Exiting.")
            return
        model.float().cpu()
        torch.cuda.empty_cache()
        gc.collect()

    # ── 5. training ───────────────────────────────────────────────────────────
    print("\n" + "="*62)
    print("STEP 5 — Fine-tuning")
    print("="*62)

    collator        = SpeechCollator(processor=processor)
    compute_metrics = make_compute_metrics(processor)

    train_args = Seq2SeqTrainingArguments(
        output_dir=str(CKPT_DIR),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=GEN_MAX_LEN,
        logging_steps=LOG_STEPS,
        report_to="none",
        push_to_hub=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch_fused",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics if eval_ds else None,
        processing_class=processor.feature_extractor,
    )

    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        print("\n  OOM — retrying with batch_size=2, grad_accum=16")
        torch.cuda.empty_cache()
        gc.collect()
        train_args_small = Seq2SeqTrainingArguments(
            **{**train_args.to_dict(),
               "per_device_train_batch_size": 2,
               "gradient_accumulation_steps": 16}
        )
        trainer = Seq2SeqTrainer(
            model=model, args=train_args_small,
            train_dataset=train_ds, eval_dataset=eval_ds,
            data_collator=collator,
            compute_metrics=compute_metrics if eval_ds else None,
            processing_class=processor.feature_extractor,
        )
        trainer.train()

    print(f"  Best checkpoint: {trainer.state.best_model_checkpoint}")

    # ── 6. save adapter ───────────────────────────────────────────────────────
    print("\n" + "="*62)
    print("STEP 6 — Save LoRA adapter")
    print("="*62)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(ADAPTER_DIR))
    processor.save_pretrained(str(ADAPTER_DIR))
    print(f"  Saved to {ADAPTER_DIR}")

    # ── 7. final eval ─────────────────────────────────────────────────────────
    ft_preds, ft_wer, ft_cer = [], None, None
    if snow_rows:
        print("\n" + "="*62)
        print("STEP 7 — Final evaluation (fine-tuned)")
        print("="*62)
        trainer.model.half().to(device)
        ft_preds, ft_wer, ft_cer = evaluate(
            trainer.model, processor, snow_rows, device
        )
        print(f"  Fine-tuned  WER={ft_wer:.1%}  CER={ft_cer:.1%}")
        if base_wer is not None:
            rel = (base_wer - ft_wer) / base_wer * 100
            print(f"  WER change  {base_wer:.1%} → {ft_wer:.1%}  ({rel:+.1f}% relative)")

    # ── 8. save results ───────────────────────────────────────────────────────
    if snow_rows and ft_wer is not None:
        print("\n" + "="*62)
        print("STEP 8 — Save results")
        print("="*62)
        save_results(
            snow_rows, base_preds, ft_preds,
            base_wer, base_cer, ft_wer, ft_cer,
            trainer.state.log_history, MODEL,
        )
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "training_log.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print("  No test set. Training log saved.")

    print("\n" + "="*62)
    print("DONE")
    print(f"  Adapter : {ADAPTER_DIR}")
    print(f"  Results : {RESULTS_DIR}")
    print("="*62)


if __name__ == "__main__":
    main()
