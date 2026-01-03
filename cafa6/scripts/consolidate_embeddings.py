"""
Script to consolidate individual ESM embedding .npy files into a single
memory-mapped array for faster data loading.

Usage:
    python -m cafa6.scripts.consolidate_embeddings --split train
    python -m cafa6.scripts.consolidate_embeddings --split test
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cafa6.constants import (
    DATA_BASE_PATH,
    EMBEDDINGS_PATH,
    ESM_MODEL,
)


def consolidate_embeddings(
    split: str = "train",
    esm_model: str = ESM_MODEL,
    strategy: str = "mean",
):
    """Consolidate individual .npy embedding files into a single memory-mapped array."""

    embedding_dir = EMBEDDINGS_PATH / split / esm_model / strategy
    output_path = DATA_BASE_PATH / f"embeddings_{split}_{esm_model}_{strategy}.npy"
    index_path = DATA_BASE_PATH / f"embeddings_{split}_{esm_model}_{strategy}_idx.pkl"

    # Find all embedding files
    embedding_files = sorted(embedding_dir.glob("*.npy"))
    n_files = len(embedding_files)

    if n_files == 0:
        raise FileNotFoundError(f"No .npy files found in {embedding_dir}")

    print(f"Found {n_files} embedding files in {embedding_dir}")

    # Get embedding dimension from first file
    sample = np.load(embedding_files[0])
    emb_dim = sample.shape[-1] if sample.ndim > 1 else sample.shape[0]

    # For "mean" strategy, embeddings are 1D (emb_dim,)
    # For "raw" strategy, embeddings are 2D (seq_len, emb_dim)
    if strategy == "mean":
        shape = (n_files, emb_dim)
    else:
        # For raw, we'd need to handle variable lengths - skip for now
        raise NotImplementedError("Raw strategy requires padding handling")

    print(f"Creating consolidated array with shape {shape}")
    print(f"Output: {output_path}")

    # Create memory-mapped output file
    consolidated = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )

    # Build index mapping: protein_id -> array index
    pid_to_idx = {}

    for i, emb_file in enumerate(tqdm(embedding_files, desc="Consolidating")):
        protein_id = emb_file.stem  # filename without .npy
        pid_to_idx[protein_id] = i
        consolidated[i] = np.load(emb_file).astype(np.float32)

    # Flush to disk
    del consolidated

    # Save index mapping
    with open(index_path, "wb") as f:
        pickle.dump(pid_to_idx, f)

    print(f"Saved index mapping to {index_path}")
    print("Done!")


if __name__ == "__main__":
    consolidate_embeddings("train", ESM_MODEL, "mean")
