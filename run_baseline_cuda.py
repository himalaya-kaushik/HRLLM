#!/usr/bin/env python3
"""
Baseline ASR evaluation — CUDA version (Kaggle T4 / any NVIDIA GPU).
Uses faster-whisper (CTranslate2) instead of mlx-whisper, which is Apple-only.

pip install:
    pip install faster-whisper jiwer pandas tqdm datasets huggingface_hub soundfile

Usage:
    python run_baseline_cuda.py --sample 20          # sanity check on 20 random clips
    python run_baseline_cuda.py                      # full 2,767-clip run
    python run_baseline_cuda.py --model medium       # faster fallback model
    python run_baseline_cuda.py --data /kaggle/input/haryanvi-tts   # custom data path

Model names (faster-whisper uses plain names, not HF repo IDs):
    large-v3  medium  small  base
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── paths & defaults ──────────────────────────────────────────────────────────
DEFAULT_DATA_DIR  = Path("./data/haryanvi-tts")
OUT_DIR           = Path("./baseline_results")
LANGUAGE          = "hi"
DEFAULT_MODEL     = "large-v3"   # faster-whisper model name
COMPUTE_TYPE      = "float16"    # optimal for T4 / any CUDA fp16-capable GPU
DEVICE            = "cuda"


# ── inference ─────────────────────────────────────────────────────────────────
def run_inference(df: pd.DataFrame, model_name: str, data_dir: Path) -> pd.DataFrame:
    from faster_whisper import WhisperModel

    print(f"Loading faster-whisper model '{model_name}' on {DEVICE} ({COMPUTE_TYPE}) ...")
    # Model is downloaded once to ~/.cache/huggingface/ and reused on subsequent runs.
    model = WhisperModel(model_name, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("Model loaded.\n")

    rows = []
    start_wall = time.perf_counter()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="ASR inference", unit="clip",
                       dynamic_ncols=True):
        audio_path = str(data_dir / row["file_name"])
        t0 = time.perf_counter()

        segments, _ = model.transcribe(
            audio_path,
            language=LANGUAGE,
            beam_size=5,
            vad_filter=True,   # skip silent regions — speeds up short clips
        )
        predicted = " ".join(seg.text for seg in segments).strip()
        elapsed = time.perf_counter() - t0

        rows.append({
            "file_name":      row["file_name"],
            "gold_text":      row["text"],
            "predicted_text": predicted,
            "inference_sec":  round(elapsed, 3),
        })

    total_wall = time.perf_counter() - start_wall
    print(f"\nInference done in {total_wall/60:.1f} min  "
          f"({total_wall/len(rows):.2f}s per clip average)")
    return pd.DataFrame(rows)


# ── metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame):
    import jiwer

    # Minimal transforms: strip whitespace only — no punctuation removal,
    # no lowercasing. Keeps gold Bangru dialect spelling untouched.
    transforms = jiwer.transforms.Compose([
        jiwer.transforms.Strip(),
        jiwer.transforms.ReduceToListOfListOfWords(),
    ])

    corpus_wer = jiwer.wer(
        list(df["gold_text"]), list(df["predicted_text"]),
        reference_transform=transforms, hypothesis_transform=transforms,
    )
    corpus_cer = jiwer.cer(
        list(df["gold_text"]), list(df["predicted_text"])
    )

    per_wer = [
        jiwer.wer(g, p, reference_transform=transforms, hypothesis_transform=transforms)
        for g, p in tqdm(
            zip(df["gold_text"], df["predicted_text"]),
            total=len(df), desc="Computing per-clip WER", unit="clip",
            dynamic_ncols=True,
        )
    ]
    df = df.copy()
    df["clip_wer"] = per_wer
    return df, corpus_wer, corpus_cer


# ── save results ──────────────────────────────────────────────────────────────
def save_results(df: pd.DataFrame, corpus_wer: float, corpus_cer: float,
                 model_name: str, sample_mode: bool) -> dict:
    OUT_DIR.mkdir(exist_ok=True)
    tag = "sample" if sample_mode else "full"

    pred_path = OUT_DIR / f"predictions_{tag}.csv"
    df.to_csv(pred_path, index=False)
    print(f"  Saved predictions : {pred_path}")

    sorted_df = df.sort_values("clip_wer", ascending=False)
    worst10 = sorted_df.head(10)[["file_name", "gold_text", "predicted_text", "clip_wer"]].to_dict("records")
    best10  = sorted_df.tail(10)[["file_name", "gold_text", "predicted_text", "clip_wer"]].to_dict("records")

    avg_sec       = df["inference_sec"].mean()
    projected_min = (avg_sec * 2767) / 60

    summary = {
        "run_type":           tag,
        "model":              model_name,
        "device":             DEVICE,
        "compute_type":       COMPUTE_TYPE,
        "language_forced":    LANGUAGE,
        "clips_evaluated":    len(df),
        "corpus_wer":         round(corpus_wer, 4),
        "corpus_cer":         round(corpus_cer, 4),
        "avg_inference_sec":  round(avg_sec, 3),
        "projected_full_min": round(projected_min, 1) if sample_mode else None,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "worst_10_clips":     worst10,
        "best_10_clips":      best10,
    }
    json_path = OUT_DIR / f"summary_{tag}.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"  Saved summary     : {json_path}")

    # ── markdown report ───────────────────────────────────────────────────────
    eta_block = ""
    if sample_mode:
        eta_block = f"\n**Projected full-run time:** ~{projected_min:.0f} min ({projected_min/60:.1f} hrs)"
        if projected_min > 60:
            eta_block += (
                f"\n\n> ⚠️ Full run will exceed 1 hour with `{model_name}`.\n"
                f"> Consider switching to `medium` for a faster baseline:\n"
                f"> ```\n> python run_baseline_cuda.py --model medium\n> ```"
            )

    def clip_rows(clips):
        return "\n".join(
            f"| {r['clip_wer']:.0%} | {r['gold_text']} | {r['predicted_text']} |"
            for r in clips
        )

    report = f"""# Haryanvi/Bangru Baseline ASR — {tag.upper()} RUN (CUDA)

**Model:** `whisper-{model_name}` (faster-whisper / CTranslate2)
**Device:** `{DEVICE}` ({COMPUTE_TYPE})
**Language forced:** `{LANGUAGE}` (Hindi)
**Clips evaluated:** {len(df)}
**Timestamp:** {summary['timestamp']}

---

## Metrics

| Metric | Value |
|--------|-------|
| **Corpus WER** | **{corpus_wer:.1%}** |
| **Corpus CER** | **{corpus_cer:.1%}** |
| Avg inference / clip | {avg_sec:.2f}s |
{eta_block}

---

## 10 Worst Clips (highest WER — biggest failures)

| WER | Gold (Bangru) | Predicted |
|-----|---------------|-----------|
{clip_rows(worst10)}

---

## 10 Best Clips (lowest WER — closest matches)

| WER | Gold (Bangru) | Predicted |
|-----|---------------|-----------|
{clip_rows(best10)}
"""
    md_path = OUT_DIR / f"report_{tag}.md"
    md_path.write_text(report)
    print(f"  Saved report      : {md_path}")

    return summary


# ── console print ─────────────────────────────────────────────────────────────
def print_console(summary: dict):
    wer  = summary["corpus_wer"]
    cer  = summary["corpus_cer"]
    avg  = summary["avg_inference_sec"]
    proj = summary.get("projected_full_min")

    print(f"\n{'='*62}")
    print(f"  CORPUS WER  : {wer:.1%}")
    print(f"  CORPUS CER  : {cer:.1%}")
    print(f"  Avg / clip  : {avg:.2f}s")
    if proj is not None:
        print(f"  Proj. full  : ~{proj:.0f} min  ({proj/60:.1f} hrs)")
        if proj > 60:
            print(f"  ⚠️  >1 hr — consider --model medium")
    print(f"{'='*62}")

    print("\n── 10 WORST CLIPS (highest WER) ────────────────────────────")
    for r in summary["worst_10_clips"]:
        print(f"  WER {r['clip_wer']:.0%}")
        print(f"    GOLD : {r['gold_text']}")
        print(f"    PRED : {r['predicted_text']}")
        print()

    print("── 10 BEST CLIPS (lowest WER) ──────────────────────────────")
    for r in summary["best_10_clips"]:
        print(f"  WER {r['clip_wer']:.0%}")
        print(f"    GOLD : {r['gold_text']}")
        print(f"    PRED : {r['predicted_text']}")
        print()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Haryanvi/Bangru baseline ASR — CUDA/T4")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="Sanity-check mode: run on N random clips only")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="faster-whisper model name: large-v3, medium, small (default: large-v3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --sample selection")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR,
                        help="Path to haryanvi-tts dataset dir (default: ./data/haryanvi-tts)")
    args = parser.parse_args()

    data_dir = args.data
    meta_csv = data_dir / "metadata.csv"

    df = pd.read_csv(meta_csv)
    sample_mode = args.sample is not None

    if sample_mode:
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
        print(f"\nSANITY CHECK MODE — {args.sample} random clips")
    else:
        print(f"\nFULL EVALUATION — {len(df)} clips")

    print(f"Model    : whisper-{args.model}  (faster-whisper, {DEVICE}, {COMPUTE_TYPE})")
    print(f"Language : {LANGUAGE} (forced)\n")

    results_df = run_inference(df, args.model, data_dir)
    results_df, corpus_wer, corpus_cer = compute_metrics(results_df)

    print("\nSaving results...")
    summary = save_results(results_df, corpus_wer, corpus_cer, args.model, sample_mode)
    print_console(summary)


if __name__ == "__main__":
    main()
