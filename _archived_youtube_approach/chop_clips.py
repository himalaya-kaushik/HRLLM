#!/usr/bin/env python3
"""
STEP 2-4: Silence-aware clip extraction + transcription sheet generation.

Usage:
    python3 chop_clips.py          # process both videos
    python3 chop_clips.py --dry-run  # show silence stats only, no clips written

Expects:
    ./test_raw/video1.wav
    ./test_raw/video2.wav

Outputs:
    ./test_clips/video1_001.wav ...
    ./test_clips/video2_001.wav ...
    ./transcription_sheet.csv
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
RAW_DIR   = Path("./test_raw")
CLIPS_DIR = Path("./test_clips")
CSV_OUT   = Path("./transcription_sheet.csv")

VIDEOS = [
    ("video1", RAW_DIR / "video1.wav"),
    ("video2", RAW_DIR / "video2.wav"),
]

MIN_CLIP     = 8.0   # seconds — don't cut before this
MAX_CLIP     = 12.0  # seconds — look for silence up to here
FALLBACK_CUT = 10.0  # seconds — hard-cut if no silence in window
TAIL_MERGE   = 3.0   # seconds — merge trailing stub into previous clip

SILENCE_NOISE = "-35dB"   # conservative: won't trigger on breath noise
SILENCE_DUR   = "0.3"     # 300 ms minimum silence


# ── ffprobe duration ──────────────────────────────────────────────────────────
def get_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


# ── silence detection ─────────────────────────────────────────────────────────
def detect_silences(path: Path) -> list[tuple[float, float]]:
    """Return list of (silence_start, silence_end) in seconds."""
    r = subprocess.run(
        ["ffmpeg", "-i", str(path),
         "-af", f"silencedetect=noise={SILENCE_NOISE}:d={SILENCE_DUR}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    stderr = r.stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", stderr)]
    ends   = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", stderr)]
    # silence_end lines also carry the gap duration — strip them
    ends   = ends[: len(starts)]   # guard against off-by-one on last clip
    return list(zip(starts, ends))


# ── cut-point builder ─────────────────────────────────────────────────────────
def build_cut_points(silences: list[tuple[float, float]], total: float) -> list[float]:
    """
    Walk forward from 0 using silence midpoints as natural cut boundaries.
    Falls back to hard-cut at FALLBACK_CUT when no silence found in window.
    Merges a short trailing stub (< TAIL_MERGE s) into the previous clip.
    """
    midpoints = sorted((s + e) / 2 for s, e in silences)

    cuts = [0.0]
    pos = 0.0

    while pos < total:
        remaining = total - pos
        if remaining < MIN_CLIP:
            break  # remainder becomes the tail of the last clip

        win_min = pos + MIN_CLIP
        win_max = min(pos + MAX_CLIP, total)

        candidates = [m for m in midpoints if win_min <= m <= win_max]

        if candidates:
            cut = candidates[0]        # earliest natural pause in window
        else:
            cut = min(pos + FALLBACK_CUT, total)

        # Don't add a cut when we're nearly at the end
        if total - cut < TAIL_MERGE:
            break

        cuts.append(cut)
        pos = cut

    cuts.append(total)
    return cuts


# ── clip extraction ───────────────────────────────────────────────────────────
def extract_clips(
    video_name: str,
    wav_path: Path,
    clips_dir: Path,
    dry_run: bool = False,
) -> list[dict]:
    total = get_duration(wav_path)
    silences = detect_silences(wav_path)

    print(f"\n  {video_name}:")
    print(f"    Total duration : {total/60:.1f} min ({total:.0f}s)")
    print(f"    Silence regions: {len(silences)}")

    cuts = build_cut_points(silences, total)
    segments = list(zip(cuts[:-1], cuts[1:]))

    print(f"    Clips planned  : {len(segments)}")

    if dry_run:
        durs = [e - s for s, e in segments]
        print(f"    Avg clip       : {sum(durs)/len(durs):.1f}s  "
              f"(min {min(durs):.1f}s  max {max(durs):.1f}s)")
        return []

    clips = []
    for i, (start, end) in enumerate(
        tqdm(segments, desc=f"    Extracting clips", unit="clip", dynamic_ncols=True)
    ):
        clip_name = f"{video_name}_{i+1:03d}.wav"
        out_path  = clips_dir / clip_name
        dur       = round(end - start, 2)

        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i",  str(wav_path),
             "-ss", f"{start:.3f}",
             "-to", f"{end:.3f}",
             "-ar", "16000", "-ac", "1",
             str(out_path)],
            check=True,
        )
        clips.append({"clip_file": clip_name, "duration_sec": dur})

    return clips


# ── CSV generation ─────────────────────────────────────────────────────────────
def generate_csv(all_clips: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clip_file", "duration_sec", "transcribe_here", "skip_reason"]
        )
        writer.writeheader()
        for clip in all_clips:
            writer.writerow({**clip, "transcribe_here": "", "skip_reason": ""})


# ── stats ──────────────────────────────────────────────────────────────────────
def print_stats(per_video: dict[str, list[dict]], all_clips: list[dict]) -> None:
    print("\n" + "=" * 62)
    print("  CLIP STATS")
    print("=" * 62)

    for video_name, clips in per_video.items():
        durs = [c["duration_sec"] for c in clips]
        print(f"\n  {video_name}:")
        print(f"    Clips     : {len(clips)}")
        if durs:
            print(f"    Total     : {sum(durs)/60:.1f} min  ({sum(durs):.0f}s)")
            print(f"    Avg clip  : {sum(durs)/len(durs):.1f}s")
            print(f"    Min clip  : {min(durs):.1f}s  |  Max clip : {max(durs):.1f}s")

    if len(per_video) > 1:
        all_durs = [c["duration_sec"] for c in all_clips]
        print(f"\n  COMBINED:")
        print(f"    Clips     : {len(all_clips)}")
        if all_durs:
            print(f"    Total     : {sum(all_durs)/60:.1f} min")
            print(f"    Avg clip  : {sum(all_durs)/len(all_durs):.1f}s")

    print("=" * 62)

    total_clips = len(all_clips)
    if total_clips >= 120:
        print(f"\n  ✓  {total_clips} clips generated — comfortably above 100 target.")
    elif total_clips >= 100:
        print(f"\n  ✓  {total_clips} clips — right at target; expect ~100 usable after skips.")
    else:
        print(f"\n  ⚠  Only {total_clips} clips. "
              f"You may end up with fewer than 100 usable clips after skipping music/Hindi/unclear.")

    print()


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Silence-aware clip extractor")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show silence stats and clip counts without writing any files")
    args = parser.parse_args()

    if not args.dry_run:
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    all_clips: list[dict] = []
    per_video: dict[str, list[dict]] = {}

    for video_name, wav_path in VIDEOS:
        if not wav_path.exists():
            print(f"\n  SKIP {video_name}: {wav_path} not found — run Step 1 first.")
            continue
        clips = extract_clips(video_name, wav_path, CLIPS_DIR, dry_run=args.dry_run)
        per_video[video_name] = clips
        all_clips.extend(clips)

    if not per_video:
        print("\nNo WAV files found in ./test_raw/ — nothing to do.")
        sys.exit(1)

    if not args.dry_run and all_clips:
        generate_csv(all_clips, CSV_OUT)
        print(f"\n  Saved CSV: {CSV_OUT}  ({len(all_clips)} rows)")

    print_stats(per_video, all_clips)

    if not args.dry_run:
        print("DONE.  Review the clip counts above.")
        print("When satisfied, start the transcription helper:")
        print("  python3 transcription_server.py\n")


if __name__ == "__main__":
    main()
