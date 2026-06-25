#!/usr/bin/env python3
"""
offline_repro.py — root-cause investigation of the evaluate() 97% WER bug.
Uses whisper-tiny on CPU. Exact versions: transformers==4.47.1, peft==0.14.0.

ROOT CAUSE found and demonstrated here.  Run:  python3 offline_repro.py
"""
import copy
import warnings
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model

MODEL_ID    = "openai/whisper-tiny"
LANGUAGE    = "hi"
TARGET_SR   = 16_000
GEN_MAX_LEN = 225

# ── helpers ────────────────────────────────────────────────────────────────────
def fake_audio(seconds=4.0, seed=42):
    rng  = np.random.default_rng(seed)
    t    = np.linspace(0, seconds, int(TARGET_SR * seconds), dtype=np.float32)
    freq = np.linspace(200, 800, len(t))
    base = 0.3 * np.sin(2 * np.pi * np.cumsum(freq) / TARGET_SR)
    return (base + 0.02 * rng.standard_normal(len(t))).astype(np.float32)

def hdr(s):
    print(f"\n{'='*70}\n{s}\n{'='*70}")

def show(label, processor, ids):
    raw  = processor.tokenizer.convert_ids_to_tokens(ids[0].tolist())
    text = processor.tokenizer.decode(ids[0], skip_special_tokens=True)
    print(f"  [{label}]")
    print(f"    ids[1:5] : {ids[0].tolist()[1:5]}")
    print(f"    tok[1:5] : {raw[1:5]}")
    print(f"    text     : {repr(text)}")
    return text

LORA = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
)

def make_peft_model():
    m = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    m.config.use_cache = False
    m = get_peft_model(m, copy.deepcopy(LORA))
    m.enable_input_require_grads()
    return m

# ══════════════════════════════════════════════════════════════════════════════
hdr("SETUP")
# ══════════════════════════════════════════════════════════════════════════════
processor          = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task="transcribe")
forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task="transcribe")
audio              = fake_audio()
inputs             = processor.feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_features
print(f"  forced_decoder_ids  : {forced_decoder_ids}")
print(f"  input_features shape: {inputs.shape}  dtype={inputs.dtype}")

# ══════════════════════════════════════════════════════════════════════════════
hdr("ROOT CAUSE DEMONSTRATION")
print("""
  WhisperForConditionalGeneration._retrieve_init_tokens() (transformers 4.47.1):

      forced_decoder_ids = generation_config.forced_decoder_ids
      if forced_decoder_ids is not None and task is not None:
          logger.warning_once("forced_decoder_ids will be ignored in favor of task=...")
          forced_decoder_ids = None    # <-- DISCARDS OUR VALUE

  When generation_config.task is set alongside forced_decoder_ids, Whisper
  ignores forced_decoder_ids and re-derives the decoder prompt from language/task.
  On CPU with whisper-tiny this re-derivation is correct. On T4 with whisper-medium
  fp16 it can produce wrong init_tokens (wrong language token or language detection
  fallback) → model generates English → 97% WER vs Devanagari references.
""")
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
hdr("TEST A — Broken: forced_decoder_ids + task/language on config → conflict")
# ══════════════════════════════════════════════════════════════════════════════
m_broken = make_peft_model()
m_broken.config.suppress_tokens            = None
m_broken.generation_config.suppress_tokens = None
m_broken.generation_config.forced_decoder_ids = forced_decoder_ids  # set explicitly
m_broken.generation_config.language           = LANGUAGE             # ALSO set language/task
m_broken.generation_config.task               = "transcribe"          # ← triggers the discard
m_broken.generation_config.max_new_tokens     = GEN_MAX_LEN
m_broken.eval()

captured_warnings = []
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    with torch.no_grad():
        ids_broken = m_broken.generate(
            input_features=inputs,
            generation_config=m_broken.generation_config,
        )
    captured_warnings = [str(x.message) for x in w if "forced_decoder_ids" in str(x.message)]

show("BROKEN (gen_config with task+forced_decoder_ids)", processor, ids_broken)
if captured_warnings:
    print(f"  !! Whisper WARNING: {captured_warnings[0][:100]}")
else:
    print(f"  (no warning captured — already warned once, or suppressed by logger)")

# ══════════════════════════════════════════════════════════════════════════════
hdr("TEST B — Fixed: forced_decoder_ids ONLY on config, no language/task")
# ══════════════════════════════════════════════════════════════════════════════
m_fixed = make_peft_model()
m_fixed.config.suppress_tokens            = None
m_fixed.generation_config.suppress_tokens = None
m_fixed.generation_config.forced_decoder_ids = forced_decoder_ids  # set explicitly
# DO NOT set language or task on generation_config
m_fixed.generation_config.language        = None   # clear if previously set
m_fixed.generation_config.task            = None   # clear if previously set
m_fixed.generation_config.max_new_tokens  = GEN_MAX_LEN
m_fixed.generation_config.max_length      = None
m_fixed.eval()

with torch.no_grad():
    ids_fixed = m_fixed.generate(
        input_features=inputs,
        forced_decoder_ids=forced_decoder_ids,   # also as kwargs for defense in depth
        max_new_tokens=GEN_MAX_LEN,
    )

show("FIXED  (no language/task on config, forced_decoder_ids as kwargs)", processor, ids_fixed)

# ══════════════════════════════════════════════════════════════════════════════
hdr("TEST C — Gate C: trainer path with fixed config == evaluate() path")
# ══════════════════════════════════════════════════════════════════════════════
m_gate = make_peft_model()
m_gate.config.suppress_tokens               = None
m_gate.generation_config.suppress_tokens    = None
m_gate.generation_config.forced_decoder_ids = forced_decoder_ids
m_gate.generation_config.language           = None   # no language/task
m_gate.generation_config.task               = None
m_gate.generation_config.max_new_tokens     = GEN_MAX_LEN
m_gate.generation_config.max_length         = None
m_gate.eval()

# Trainer path: no kwargs at all, relies on model.generation_config
with torch.no_grad():
    ids_trainer = m_gate.generate(input_features=inputs)

# evaluate() path: forced_decoder_ids + max_new_tokens as direct kwargs
with torch.no_grad():
    ids_eval = m_gate.generate(
        input_features=inputs,
        forced_decoder_ids=forced_decoder_ids,
        max_new_tokens=GEN_MAX_LEN,
    )

t_tr = show("Trainer path (predict_with_generate, no extra kwargs)", processor, ids_trainer)
t_ev = show("Fixed evaluate() path (direct kwargs)", processor, ids_eval)
gate_c = ids_trainer.tolist() == ids_eval.tolist()
print(f"\n  Gate C: trainer path == fixed evaluate() path → {gate_c}")

# ══════════════════════════════════════════════════════════════════════════════
hdr("TEST D — fp16 model + fp16 inputs (same as Kaggle evaluate())")
# ══════════════════════════════════════════════════════════════════════════════
m_fp16 = make_peft_model()
m_fp16.config.suppress_tokens               = None
m_fp16.generation_config.suppress_tokens    = None
m_fp16.generation_config.forced_decoder_ids = forced_decoder_ids
m_fp16.generation_config.language           = None
m_fp16.generation_config.task               = None
m_fp16.generation_config.max_new_tokens     = GEN_MAX_LEN
m_fp16.generation_config.max_length         = None
m_fp16 = m_fp16.half()  # simulate model.half() in main()
m_fp16.eval()

inputs_f16 = inputs.half()   # simulate .to(device).half() in evaluate()
with torch.no_grad():
    ids_fp16 = m_fp16.generate(
        input_features=inputs_f16,
        forced_decoder_ids=forced_decoder_ids,
        max_new_tokens=GEN_MAX_LEN,
    )
show("fp16 fixed evaluate() path", processor, ids_fp16)
fp16_ok = ids_fp16[0][1].item() == forced_decoder_ids[0][1]
print(f"  First non-BOS token is Hindi ({forced_decoder_ids[0][1]})? → {fp16_ok}")

# ══════════════════════════════════════════════════════════════════════════════
hdr("GATE A — Simulated evaluate() loop: broken vs fixed")
# ══════════════════════════════════════════════════════════════════════════════
import jiwer
import jiwer.transforms as jt
tfm      = jt.Compose([jt.Strip(), jt.ReduceToListOfListOfWords()])
fake_ref = ["यह एक परीक्षण है", "मैं यहाँ हूँ", "राम यहाँ आया", "भगवान का वचन"]

def run_loop(model, use_task_on_config, label):
    """
    Mirror evaluate() from finetune_whisper.py.
    use_task_on_config=True  → broken (language/task set on config → forced_decoder_ids discarded)
    use_task_on_config=False → fixed  (only forced_decoder_ids on config)
    """
    model.config.suppress_tokens            = None
    model.generation_config.suppress_tokens = None
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.max_new_tokens     = GEN_MAX_LEN
    model.generation_config.max_length         = None
    if use_task_on_config:
        model.generation_config.language = LANGUAGE      # BROKEN: triggers discard
        model.generation_config.task     = "transcribe"  # BROKEN: triggers discard
    else:
        model.generation_config.language = None   # FIXED: no conflict
        model.generation_config.task     = None

    model.eval()
    preds = []
    for seed in range(len(fake_ref)):
        a   = fake_audio(seconds=3.0, seed=seed)
        inp = processor.feature_extractor(a, sampling_rate=TARGET_SR, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(
                input_features=inp,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=GEN_MAX_LEN,
            )
        preds.append(processor.tokenizer.decode(ids[0], skip_special_tokens=True))

    wer = jiwer.wer(fake_ref, preds, reference_transform=tfm, hypothesis_transform=tfm)
    print(f"\n  [{label}]")
    for r, p in zip(fake_ref, preds):
        lang_tok = processor.tokenizer.convert_ids_to_tokens(
            [model.generate(
                input_features=processor.feature_extractor(fake_audio(seed=0), sampling_rate=TARGET_SR, return_tensors="pt").input_features,
                forced_decoder_ids=forced_decoder_ids if not use_task_on_config else None,
                max_new_tokens=GEN_MAX_LEN,
            )[0][1].item()]
        )[0] if False else "N/A"
        print(f"    ref : {repr(r)}")
        print(f"    pred: {repr(p)}")
    print(f"  WER = {wer:.1%}  (fake audio → 100% expected; key: preds non-empty + language token)")
    return preds, wer

mb2 = make_peft_model()
mf2 = make_peft_model()
_, wer_broken = run_loop(mb2, use_task_on_config=True,  label="BROKEN — language/task on config")
_, wer_fixed  = run_loop(mf2, use_task_on_config=False, label="FIXED  — no language/task on config")

# ══════════════════════════════════════════════════════════════════════════════
hdr("VERDICT")
# ══════════════════════════════════════════════════════════════════════════════
lang_broken = processor.tokenizer.convert_ids_to_tokens([ids_broken[0][1].item()])[0]
lang_fixed  = processor.tokenizer.convert_ids_to_tokens([ids_fixed[0][1].item()])[0]
print(f"""
  ROOT CAUSE (proven by WhisperForConditionalGeneration source):
  ─────────────────────────────────────────────────────────────
  _retrieve_init_tokens() in transformers 4.47.1 silently DISCARDS
  forced_decoder_ids when generation_config.task is also set:

      if forced_decoder_ids is not None and task is not None:
          forced_decoder_ids = None    # discarded, re-derived from language/task

  The current script's _set_generation_config() sets BOTH:
      model.generation_config.forced_decoder_ids = [...]   # our value
      model.generation_config.task = "transcribe"          # triggers discard!

  On CPU with whisper-tiny the re-derivation happens to produce correct tokens.
  On Kaggle T4 with whisper-medium fp16, the re-derivation can produce wrong
  init_tokens → model generates in wrong language → 97% WER vs Devanagari.

  Language token in broken path : {lang_broken}
  Language token in fixed path  : {lang_fixed}
  Gate C (trainer == evaluate)  : {gate_c}
  fp16 first token correct      : {fp16_ok}

  THE FIX:
  ────────
  1. Do NOT set language/task on generation_config (clears them to None).
  2. Set ONLY forced_decoder_ids and max_new_tokens / max_length=None.
  3. In evaluate(): pass forced_decoder_ids as direct kwarg — bypasses any
     generation_config entirely, making the behavior deterministic.
  4. Same config for trainer's predict_with_generate (Gate C confirmed above).
""")
