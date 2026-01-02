import os
from pathlib import Path

import polars as pl
from tqdm import tqdm

from cafa6.constants import EMBEDDINGS_PATH, ESM_MODEL, TRAIN_TERMS_EXTENDED_PATH

# Adjust path to your final combined training file

EMBEDDINGS_DIR = EMBEDDINGS_PATH / f"train/{ESM_MODEL}/mean"


def remove_ghost_embeddings():
    print("Loading valid IDs...")
    # Get the set of all IDs that actually have labels
    df = pl.read_csv(TRAIN_TERMS_EXTENDED_PATH)
    valid_ids = set(df["EntryID"])

    print(f"Found {len(valid_ids)} valid labeled proteins.")

    files_in_dir = list(EMBEDDINGS_DIR.glob("*.npy"))
    print(f"Scanning {len(files_in_dir)} files in embedding directory...")

    deleted_count = 0

    for file_path in tqdm(files_in_dir):
        # Extract ID: "path/to/Q9Z.npy" -> "Q9Z"
        pid = file_path.stem

        if pid not in valid_ids:
            # This is a ghost (test protein with no labels)
            os.remove(file_path)
            deleted_count += 1

    print(f"Cleanup complete. Deleted {deleted_count} ghost embeddings.")


if __name__ == "__main__":
    remove_ghost_embeddings()
