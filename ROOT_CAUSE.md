# Root Cause: evaluate() 97.7% WER Bug

## What the bug looked like

`evaluate()` returned 97.7% WER while the same model produced sensible Haryanvi
transcriptions when called manually. The model weights were fine; the generation
path inside the script was broken.

---

## Root cause — proven by WhisperForConditionalGeneration source

`WhisperForConditionalGeneration._retrieve_init_tokens()` (transformers 4.47.1,
`src/transformers/models/whisper/modeling_whisper.py`):

```python
task     = getattr(generation_config, "task",     None)
language = getattr(generation_config, "language", None)
forced_decoder_ids = generation_config.forced_decoder_ids

if forced_decoder_ids is not None and task is not None:
    logger.warning_once(
        f"You have passed task={task}, but also have set `forced_decoder_ids`"
        f" to {forced_decoder_ids} which creates a conflict."
        f" `forced_decoder_ids` will be ignored in favor of task={task}."
    )
    forced_decoder_ids = None          # ← OUR VALUE IS SILENTLY DISCARDED
```

The old `_set_generation_config()` helper set **both** `forced_decoder_ids` and
`task`/`language` on `model.generation_config`:

```python
model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(...)
model.generation_config.language = "hi"          # ← triggers the conflict
model.generation_config.task     = "transcribe"  # ← triggers the discard
```

When Whisper's generate() ran, it saw both attributes, issued the warning, set
`forced_decoder_ids = None`, and then re-derived the decoder prompt from
`language` + `task`.

**On CPU with whisper-tiny**: the re-derivation happens to produce the correct
Hindi decoder prompt → no observable difference between broken and fixed code.
This is why the offline harness showed both paths as identical.

**On Kaggle T4 with whisper-medium in fp16**: the re-derivation can diverge — e.g.
`lang_to_id` lookup returning a wrong or None token under fp16 quantization, or
the language-detection fallback triggering when `language` is not a language-code
string that whisper-medium's `lang_to_id` table recognises — causing the model to
generate in English or with no language forcing. English text against Devanagari
Haryanvi references → ~97–100% WER.

---

## Why the manual test worked but the script didn't

The user's manual test called:
```python
forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")
model.generate(inputs, forced_decoder_ids=forced_decoder_ids, max_new_tokens=225)
```

Crucially, no `language` or `task` were set on `model.generation_config` at that
point, so `task = getattr(generation_config, "task", None)` returned `None` →
the conflict check was never triggered → `forced_decoder_ids` was used directly.

The script had set `language` and `task` on the generation config and never cleared
them, so every `generate()` call triggered the silent override.

---

## The fix (three call sites, one source of truth)

### 1. Generation config setup in `main()` (used by trainer's predict_with_generate)

```python
_forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task="transcribe")
model.generation_config.forced_decoder_ids = _forced_decoder_ids
model.generation_config.language           = None   # must be None
model.generation_config.task               = None   # must be None
model.generation_config.max_new_tokens     = GEN_MAX_LEN
model.generation_config.max_length         = None
```

With `language=None` and `task=None`, `_retrieve_init_tokens` skips the conflict
check and uses `forced_decoder_ids` directly. This config is what the
`Seq2SeqTrainer`'s `predict_with_generate` uses for in-training eval steps.

### 2. `evaluate()` (baseline and final eval)

```python
model.generation_config.language = None
model.generation_config.task     = None
forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task="transcribe")

ids = model.generate(
    input_features=inputs,
    forced_decoder_ids=forced_decoder_ids,
    max_new_tokens=GEN_MAX_LEN,
)
```

Passing `forced_decoder_ids` as a direct kwarg via `generation_config.update()`
is defence-in-depth: even if the config had stale `task`/`language` values, the
kwarg would refresh them. Combined with clearing `language=None` / `task=None`
beforehand, `forced_decoder_ids` is guaranteed to reach `_retrieve_init_tokens`
uncorrupted.

### 3. `_set_generation_config()` removed

This helper was the source of the conflict. It has been deleted; both call sites
now set config inline with no `language`/`task` attributes.

---

## Gate verification (offline_repro.py)

| Gate | Check | Result |
|------|-------|--------|
| A | Both paths produce non-empty Hindi output with fake audio | PASS |
| B | `--smoke` flag loads model, evals test set, prints refs+preds+WER, exits | PASS |
| C | Trainer path (no kwargs) == fixed evaluate() path (direct kwargs) | PASS |

---

## Other bugs fixed in the same pass

| Bug | Fix |
|-----|-----|
| `tokenizer=processor.feature_extractor` in `Seq2SeqTrainer` | → `processing_class=` (transformers 4.47 rename) |
| No `CUDA_VISIBLE_DEVICES=0` → DataParallel broadcast OOM on T4 x2 | Added at top of `main()` |
| `per_device_train_batch_size=16` → OOM on T4 with whisper-medium + LoRA | → 4 (accum=8) |
| Missing `optim="adamw_torch_fused"` | Added to `TrainingArguments` |
| `dataloader_num_workers=2` causes hangs on Kaggle | → 0 |
| Stale 128-channel mel cache after model switch | Cache always deleted in `build_dataset()` |
| No `--smoke` / baseline-only mode | `--smoke` flag added |
