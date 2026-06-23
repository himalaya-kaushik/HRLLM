#!/usr/bin/env python3
"""
Pre-transcribe video2 clips with faster-whisper and write results into
the whisper_draft column of transcription_sheet.csv.

Only touches rows whose clip_file starts with "video2_".
Adds the whisper_draft column if it doesn't exist yet.
Safe to re-run — skips clips that already have a draft.

Usage:
    python3 pretranscribe_video2.py                 # auto-detect device
    python3 pretranscribe_video2.py --device cpu    # force CPU (M2 Mac)
    python3 pretranscribe_video2.py --device cuda   # force CUDA (Kaggle)
    python3 pretranscribe_video2.py --model medium  # faster model
"""

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

CLIPS_DIR  = Path("./test_clips")
CSV_PATH   = Path("./transcription_sheet.csv")
LANGUAGE   = "hi"
COLUMN_ORDER = ["clip_file", "duration_sec", "whisper_draft", "transcribe_here", "skip_reason"]


# ── CSV helpers ───────────────────────────────────────────────────────────────
def load_csv() -> tuple[list[str], list[dict]]:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows   = list(reader)
    return fields, rows


def save_csv(fields: list[str], rows: list[dict]) -> None:
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ensure_whisper_draft_column(fields: list[str], rows: list[dict]) -> list[str]:
    if "whisper_draft" in fields:
        return fields
    # Insert after duration_sec
    try:
        idx = fields.index("duration_sec") + 1
    except ValueError:
        idx = len(fields)
    fields = fields[:idx] + ["whisper_draft"] + fields[idx:]
    for row in rows:
        row.setdefault("whisper_draft", "")
    return fields


# ── device detection ──────────────────────────────────────────────────────────
def pick_device(requested: str | None) -> tuple[str, str]:
    if requested:
        device = requested
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    return device, compute


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-transcribe video2 clips")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                        help="Inference device (default: auto-detect)")
    parser.add_argument("--model", default="large-v3",
                        help="faster-whisper model name (default: large-v3)")
    args = parser.parse_args()

    if not CSV_PATH.exists():
        sys.exit(f"ERROR: {CSV_PATH} not found — run chop_clips.py first.")

    fields, rows = load_csv()
    fields = ensure_whisper_draft_column(fields, rows)

    # Reorder fields to canonical order (add any unexpected extras at end)
    known   = [f for f in COLUMN_ORDER if f in fields]
    unknown = [f for f in fields if f not in COLUMN_ORDER]
    fields  = known + unknown

    video2_rows = [r for r in rows if r["clip_file"].startswith("video2_")]
    pending     = [r for r in video2_rows if not r.get("whisper_draft", "").strip()]

    print(f"video2 clips total   : {len(video2_rows)}")
    print(f"Already have draft   : {len(video2_rows) - len(pending)}")
    print(f"To pre-transcribe    : {len(pending)}")

    if not pending:
        print("\nAll video2 clips already have a whisper_draft. Nothing to do.")
        save_csv(fields, rows)
        return

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit(
            "ERROR: faster-whisper not installed.\n"
            "Run: pip install faster-whisper"
        )

    device, compute = pick_device(args.device)
    print(f"\nLoading faster-whisper '{args.model}' on {device} ({compute}) ...")
    model = WhisperModel(args.model, device=device, compute_type=compute)
    print("Model loaded.\n")

    count = 0
    for row in tqdm(pending, desc="Pre-transcribing video2", unit="clip", dynamic_ncols=True):
        audio_path = CLIPS_DIR / row["clip_file"]
        if not audio_path.exists():
            tqdm.write(f"  SKIP (file not found): {row['clip_file']}")
            continue
        segments, _ = model.transcribe(
            str(audio_path),
            language=LANGUAGE,
            beam_size=5,
            vad_filter=True,
        )
        row["whisper_draft"] = " ".join(seg.text for seg in segments).strip()
        count += 1

    save_csv(fields, rows)
    print(f"\nPre-transcribed : {count} video2 clips")
    print(f"Updated CSV     : {CSV_PATH}")


if __name__ == "__main__":
    main()
