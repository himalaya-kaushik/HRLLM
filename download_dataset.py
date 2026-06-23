"""
Download ankitdhiman/haryanvi-tts dataset to ./data/haryanvi-tts/
Saves raw .wav files under train/ and metadata.csv at top level.
Run from the project root:
    source hrenv/bin/activate
    python download_dataset.py
"""

import os
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "ankitdhiman/haryanvi-tts"
SAVE_DIR = "./data/haryanvi-tts"
TRAIN_DIR = os.path.join(SAVE_DIR, "train")

os.makedirs(TRAIN_DIR, exist_ok=True)

# --- Step 1: download metadata.csv ---
print("Fetching metadata.csv ...")
meta_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="metadata.csv",
    repo_type="dataset",
    local_dir=SAVE_DIR,
)
df = pd.read_csv(meta_path)
print(f"  {len(df)} rows found in metadata.csv")

# --- Step 2: list all wav files in the repo ---
print("Listing audio files in repo ...")
all_repo_files = list(list_repo_files(REPO_ID, repo_type="dataset"))
wav_files = [f for f in all_repo_files if f.startswith("train/") and f.endswith(".wav")]
print(f"  {len(wav_files)} wav files found in repo")

# --- Step 3: download each wav with a progress bar ---
skipped = 0
failed = []

for rel_path in tqdm(wav_files, desc="Downloading audio", unit="clip"):
    fname = os.path.basename(rel_path)
    out_path = os.path.join(TRAIN_DIR, fname)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        skipped += 1
        continue

    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=rel_path,
            repo_type="dataset",
            local_dir=SAVE_DIR,
        )
    except Exception as e:
        failed.append((rel_path, str(e)))

# --- Summary ---
print(f"\nDone.")
print(f"  WAV files dir      : {TRAIN_DIR}/")
print(f"  Metadata           : {meta_path}")
print(f"  Total in repo      : {len(wav_files)}")
print(f"  Already on disk    : {skipped} (skipped)")
print(f"  Newly downloaded   : {len(wav_files) - skipped - len(failed)}")
print(f"  Failed             : {len(failed)}")
print(f"  Rows in CSV        : {len(df)}")
print(f"  Duplicate texts    : {df['text'].duplicated().sum()}")

if failed:
    print("\nFailed files:")
    for path, err in failed[:10]:
        print(f"  {path}: {err}")
