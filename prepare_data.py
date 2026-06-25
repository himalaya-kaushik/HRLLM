#!/usr/bin/env python3
"""
Download ONLY the test clips listed in test_common.csv from bridgeconn/snow-mountain.

NOTE: Individual wav files are NOT separately accessible in this HF repo.
Audio is stored as per-book tar.gz archives (e.g. data/cleaned/haryanvi/MAT.tar.gz).
Strategy: download each needed book archive, stream-extract ONLY the clips we want,
then delete the archive immediately to recover space.

  1. Download data/experiments/haryanvi/test_common.csv (~125KB, instant)
  2. Optionally limit to N clips spread proportionally across books (--max-clips)
  3. Per book: hf_hub_download the book's tar.gz, stream-extract needed clips, delete
  4. Save to data/snow-mountain-haryanvi/audio/ with metadata.csv

Already-cached archives (MAT 788MB, MRK 490MB) are reused — no re-download.

Usage:
    python3 prepare_data.py                   # all 500 clips
    python3 prepare_data.py --max-clips 60    # 60 clips, proportional spread
    python3 prepare_data.py --dry-run         # show allocation, no download
"""

import argparse
import csv
import io
import os
import sys
import tarfile
from collections import defaultdict
from math import gcd
from pathlib import Path

import requests
import numpy as np
import soundfile as sf
from tqdm import tqdm
from huggingface_hub import hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers

REPO_ID   = "bridgeconn/snow-mountain"
SAVE_DIR  = Path("./data/snow-mountain-haryanvi")
AUDIO_DIR = SAVE_DIR / "audio"
TARGET_SR = 16000
LANG      = "haryanvi"

TEST_CSV_PATH = f"data/experiments/{LANG}/test_common.csv"


# ── resampling ────────────────────────────────────────────────────────────────
def resample(arr: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_SR:
        return arr
    try:
        import librosa
        return librosa.resample(arr.astype(np.float32), orig_sr=orig_sr, target_sr=TARGET_SR)
    except ImportError:
        pass
    try:
        from scipy.signal import resample_poly
        g = gcd(orig_sr, TARGET_SR)
        return resample_poly(arr, TARGET_SR // g, orig_sr // g).astype(np.float32)
    except ImportError:
        pass
    n = int(len(arr) * TARGET_SR / orig_sr)
    return np.interp(np.linspace(0, len(arr) - 1, n), np.arange(len(arr)), arr).astype(np.float32)


# ── read test_common.csv ──────────────────────────────────────────────────────
def load_test_csv() -> tuple[list[dict], dict[str, dict]]:
    """
    Returns:
      rows      — original CSV rows in order (for metadata.csv output)
      fname_map — {wav_basename: {out_fname, text, book}} for fast lookup during extraction
    """
    print(f"Downloading test_common.csv ...")
    local_csv = hf_hub_download(
        repo_id=REPO_ID,
        filename=TEST_CSV_PATH,
        repo_type="dataset",
    )
    with open(local_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"  {len(rows)} clips in test_common.csv")

    fname_map: dict[str, dict] = {}
    for i, row in enumerate(rows):
        hf_path  = row["path"]            # data/cleaned/haryanvi/MAT/MAT_012_037.wav
        wav_base = os.path.basename(hf_path)   # MAT_012_037.wav
        book     = hf_path.split("/")[-2]      # MAT
        fname_map[wav_base] = {
            "out_fname": f"haryanvi_{i+1:04d}.wav",
            "text":      row["sentence"].strip(),
            "book":      book,
        }
    return rows, fname_map


# ── stream one book's tar.gz, extract only needed clips ───────────────────────
def stream_book(book: str, needed: set[str], fname_map: dict, audio_dir: Path) -> int:
    """
    Stream data/cleaned/haryanvi/{book}.tar.gz from HF without writing it to disk.
    Extracts only the wav files in `needed`, saves them resampled to audio_dir.
    Stops streaming as soon as all needed files are found.
    Returns the number of clips successfully extracted.
    """
    url = hf_hub_url(
        repo_id=REPO_ID,
        filename=f"data/cleaned/{LANG}/{book}.tar.gz",
        repo_type="dataset",
    )
    headers = build_hf_headers()

    try:
        resp = requests.get(url, stream=True, headers=headers, timeout=120)
        resp.raise_for_status()
        resp.raw.decode_content = True   # handle Transfer-Encoding: chunked
    except Exception as e:
        tqdm.write(f"  ERROR opening stream for {book}.tar.gz: {e}")
        return 0

    remaining = set(needed)
    extracted = 0

    try:
        # r|gz = streaming gzip — no seeking, reads chunk-by-chunk
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
            for member in tar:
                wav_base = os.path.basename(member.name)
                if wav_base not in remaining:
                    continue

                f = tar.extractfile(member)
                if f is None:
                    remaining.discard(wav_base)
                    continue

                raw_bytes = f.read()
                try:
                    arr, sr = sf.read(io.BytesIO(raw_bytes))
                except Exception as e:
                    tqdm.write(f"  WARN {wav_base}: {e}")
                    remaining.discard(wav_base)
                    continue

                arr = arr.astype(np.float32)
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                arr = resample(arr, sr)

                out_fname = fname_map[wav_base]["out_fname"]
                sf.write(str(audio_dir / out_fname), arr, TARGET_SR, subtype="PCM_16")
                remaining.discard(wav_base)
                extracted += 1

                if not remaining:
                    break   # got everything — connection closes, rest of archive skipped

    except Exception as e:
        tqdm.write(f"  ERROR streaming {book}.tar.gz: {e}")
    finally:
        resp.close()

    return extracted


# ── proportional clip selection ───────────────────────────────────────────────
def select_proportional(rows: list[dict], max_clips: int) -> list[dict]:
    """
    Pick max_clips rows spread proportionally across books.
    Uses largest-remainder method so the total is exactly max_clips.
    Books with very few clips still get at least 1 where the remainder allows it.
    """
    book_indices: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        book = row["path"].split("/")[-2]
        book_indices[book].append(i)

    total  = len(rows)
    books  = sorted(book_indices.keys())
    ideal  = {b: len(book_indices[b]) / total * max_clips for b in books}
    alloc  = {b: int(ideal[b]) for b in books}
    gap    = max_clips - sum(alloc.values())

    # Give the remaining slots to books with the largest fractional remainders
    by_remainder = sorted(books, key=lambda b: ideal[b] - alloc[b], reverse=True)
    for b in by_remainder[:gap]:
        alloc[b] += 1

    # Print allocation so the user can see the spread
    print(f"\n  Clip allocation across {len(books)} books (--max-clips {max_clips}):")
    for b in books:
        n = alloc[b]
        if n:
            bar = "█" * n
            print(f"    {b:>4}  {bar}  {n}")

    selected: list[int] = []
    for b in books:
        selected.extend(book_indices[b][: alloc[b]])
    selected.sort()
    return [rows[i] for i in selected]


# ── stats ─────────────────────────────────────────────────────────────────────
def print_stats(metadata_rows: list[dict], audio_dir: Path) -> None:
    total_sec = 0.0
    for row in tqdm(metadata_rows, desc="Computing durations", unit="clip", dynamic_ncols=True):
        wav_path = audio_dir / Path(row["file_name"]).name
        if wav_path.exists():
            total_sec += sf.info(str(wav_path)).duration

    non_empty = sum(1 for r in metadata_rows if r["text"])

    print(f"\n{'='*62}")
    print("  SNOW-MOUNTAIN HARYANVI — TEST SET STATS")
    print(f"{'='*62}")
    print(f"  Clips saved      : {len(metadata_rows):,}")
    print(f"  Total duration   : {total_sec/60:.1f} min  ({total_sec:.0f}s)")
    print(f"  Avg clip length  : {total_sec/len(metadata_rows):.1f}s")
    print(f"  Clips with text  : {non_empty:,} / {len(metadata_rows):,}")
    print(f"{'='*62}")
    print(f"\n  Sample transcripts (first 8):")
    for row in metadata_rows[:8]:
        preview = row["text"][:88] + ("…" if len(row["text"]) > 88 else "")
        print(f"    {row['file_name']}  →  {preview}")
    print(f"\n  Saved to  : {SAVE_DIR}/")
    print(f"    audio/       — {len(metadata_rows):,} .wav files @ 16kHz mono")
    print(f"    metadata.csv — {len(metadata_rows):,} rows, columns: file_name, text")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Haryanvi test clips from bridgeconn/snow-mountain"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show book/clip breakdown without downloading any audio")
    parser.add_argument("--max-clips", type=int, default=None, metavar="N",
                        help="Limit to N clips, spread proportionally across books")
    args = parser.parse_args()

    if not args.dry_run:
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    rows, fname_map = load_test_csv()

    if args.max_clips is not None:
        if args.max_clips >= len(rows):
            print(f"  --max-clips {args.max_clips} >= total {len(rows)}, using all clips.")
        else:
            rows = select_proportional(rows, args.max_clips)
            # Rebuild fname_map from the selected rows only
            fname_map = {}
            for i, row in enumerate(rows):
                wav_base = os.path.basename(row["path"])
                fname_map[wav_base] = {
                    "out_fname": f"haryanvi_{i+1:04d}.wav",
                    "text":      row["sentence"].strip(),
                    "book":      row["path"].split("/")[-2],
                }

    # Group clips by book, skipping any already saved
    book_to_fnames: dict[str, set[str]] = defaultdict(set)
    already_done = 0
    for wav_base, info in fname_map.items():
        out_path = AUDIO_DIR / info["out_fname"]
        if not args.dry_run and out_path.exists() and out_path.stat().st_size > 0:
            already_done += 1
        else:
            book_to_fnames[info["book"]].add(wav_base)

    books_needed = sorted(book_to_fnames.keys())
    total_pending = sum(len(v) for v in book_to_fnames.values())

    print(f"\nClips to stream  : {total_pending}  (already done: {already_done})")
    print(f"Books to stream  : {len(books_needed)}  ({', '.join(books_needed)})")
    print("  Each archive is streamed — only the needed clips are read,")
    print("  the rest of the tar.gz is discarded without hitting disk.\n")

    if args.dry_run:
        print(f"Dry run complete — {total_pending} clips across "
              f"{len(books_needed)} books would be streamed.")
        return

    total_extracted = 0
    for book in tqdm(books_needed, desc="Books", unit="book", dynamic_ncols=True):
        needed   = book_to_fnames[book]
        n        = stream_book(book, needed, fname_map, AUDIO_DIR)
        total_extracted += n
        missing  = len(needed) - n
        if missing:
            tqdm.write(f"  ⚠  {book}: {missing} clip(s) not found in archive")

    print(f"\nExtracted {total_extracted} / {total_pending} clips.")

    # Build metadata.csv in original CSV order
    metadata_rows = []
    for i, row in enumerate(rows):
        wav_base  = os.path.basename(row["path"])
        out_fname = fname_map[wav_base]["out_fname"]
        out_path  = AUDIO_DIR / out_fname
        if out_path.exists() and out_path.stat().st_size > 0:
            metadata_rows.append({
                "file_name": f"audio/{out_fname}",
                "text":      row["sentence"].strip(),
            })

    csv_path = SAVE_DIR / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"\nSaved metadata.csv  ({len(metadata_rows)} rows)")

    if not metadata_rows:
        print("ERROR: no clips were saved.")
        sys.exit(1)

    print_stats(metadata_rows, AUDIO_DIR)
    print("\nVerify the transcripts look like authentic Haryanvi Bible speech.")
    print("When confirmed, proceed to fine-tuning setup.\n")


if __name__ == "__main__":
    main()
