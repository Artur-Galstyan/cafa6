import gc
import os
import pickle
import sys

import faiss
import numpy as np
import polars as pl
from beartype.typing import Literal
from Bio import SeqIO
from tqdm import tqdm

from cafa6.constants import (
    EMBEDDINGS_PATH,
    ESM_MODEL,
    IA_PATH,
    MODEL_TO_DIMS,
    PARTIAL_SUBMISSION_FULL_PATH,
    TEST_NEIGHBOR_MATRIX_IDX_MAP_PATH,
    TEST_NEIGHBOR_MATRIX_PATH,
    TEXT_EMBEDDINGS_PATH_TRAIN,
    TEXT_EMBEDDINGS_TEST_NEIGHBOR_IDX_PATH,
    TEXT_EMBEDDINGS_TEST_NEIGHBOR_MATRIX_PATH,
    TEXT_EMBEDDINGS_TRAIN_NEIGHBOR_IDX_PATH,
    TEXT_EMBEDDINGS_TRAIN_NEIGHBOR_MATRIX_PATH,
    TRAIN_FASTA_PATH_SPLIT,
    TRAIN_NEIGHBOR_MATRIX_IDX_MAP_PATH,
    TRAIN_NEIGHBOR_MATRIX_PATH,
    VAL_FASTA_PATH_SPLIT,
    VAL_NEIGHBOR_MATRIX_IDX_MAP_PATH,
    VAL_NEIGHBOR_MATRIX_PATH,
    VAL_TEXT_NEIGHBOR_IDX_MAP_PATH,
    VAL_TEXT_NEIGHBOR_MATRIX_PATH,
)


def generate_neighbor_priors_matrix(target_set: Literal["train", "test"], k: int = 6):
    ia_df = pl.read_csv(IA_PATH, separator="\t")
    terms_to_idx = {row["term"]: i for i, row in enumerate(ia_df.iter_rows(named=True))}
    n_terms = len(terms_to_idx)
    embed_dim = MODEL_TO_DIMS[ESM_MODEL]

    source_path = EMBEDDINGS_PATH / f"train/{ESM_MODEL}/mean"
    target_path = EMBEDDINGS_PATH / f"{target_set}/{ESM_MODEL}/mean"

    source_ids = [pid.replace(".npy", "") for pid in os.listdir(source_path)]
    source_pid_to_idx = {pid: i for i, pid in enumerate(source_ids)}
    n_source = len(source_ids)

    print(f"Loading {n_source} SOURCE embeddings (float32)...")
    source_embeddings = np.stack(
        [np.load(f"{source_path}/{pid}.npy") for pid in tqdm(source_ids)]
    ).astype(np.float32)

    faiss.normalize_L2(source_embeddings)

    print("Building Source Label Matrix (float16)...")
    source_labels = np.zeros((n_source, n_terms), dtype=np.float16)
    all_known_terms_df = pl.read_csv(PARTIAL_SUBMISSION_FULL_PATH)

    source_ids_set = set(source_ids)
    valid_rows = all_known_terms_df.filter(pl.col("EntryID").is_in(source_ids_set))

    for row in tqdm(valid_rows.iter_rows(named=True)):
        pid = row["EntryID"]
        term = row["term"]
        if term in terms_to_idx:
            source_labels[source_pid_to_idx[pid], terms_to_idx[term]] = 1.0

    target_ids = [pid.replace(".npy", "") for pid in os.listdir(target_path)]
    n_target = len(target_ids)

    print(f"Loading {n_target} TARGET embeddings (float16)...")
    target_embeddings = np.stack(
        [np.load(f"{target_path}/{pid}.npy") for pid in tqdm(target_ids)]
    ).astype(np.float16)

    print("Building Index...")
    index = faiss.IndexFlatIP(embed_dim)  # ty:ignore[possibly-missing-attribute]
    index.add(source_embeddings)  # ty:ignore[missing-argument]

    neighbor_priors = np.zeros(shape=(n_target, n_terms), dtype=np.float16)
    batch_size = 1000

    print("Searching Neighbors...")
    for start in tqdm(range(0, n_target, batch_size)):
        end = min(start + batch_size, n_target)

        queries_f16 = target_embeddings[start:end]
        queries_f32 = queries_f16.astype(np.float32)
        faiss.normalize_L2(queries_f32)

        search_k = k + 1 if target_set == "train" else k
        distances, indices = index.search(queries_f32, search_k)  # ty:ignore[missing-argument]

        for i, neighbors in enumerate(indices):
            if target_set == "train":
                neighbors = neighbors[1:]

            neighbor_labels = source_labels[neighbors]
            neighbor_priors[start + i] = neighbor_labels.mean(axis=0).astype(np.float16)

        del queries_f16, queries_f32
        gc.collect()

    save_path = (
        TRAIN_NEIGHBOR_MATRIX_PATH
        if target_set == "train"
        else TEST_NEIGHBOR_MATRIX_PATH
    )
    np.save(save_path, neighbor_priors)

    target_pid_to_idx = {pid: i for i, pid in enumerate(target_ids)}
    idx_map_save_path = (
        TRAIN_NEIGHBOR_MATRIX_IDX_MAP_PATH
        if target_set == "train"
        else TEST_NEIGHBOR_MATRIX_IDX_MAP_PATH
    )
    with open(idx_map_save_path, "wb") as f:
        pickle.dump(target_pid_to_idx, f)


def generate_neighbor_priors_matrix_text_embeddings(
    target_set: Literal["train", "test"], k: int = 6
):
    embed_dim = MODEL_TO_DIMS[ESM_MODEL]
    voyage_embed_dim = 1024  # voyage embed dim

    source_path = EMBEDDINGS_PATH / f"train/{ESM_MODEL}/mean"
    target_path = EMBEDDINGS_PATH / f"{target_set}/{ESM_MODEL}/mean"

    source_ids = [pid.replace(".npy", "") for pid in os.listdir(source_path)]
    source_idx_to_pid = {i: pid for i, pid in enumerate(source_ids)}
    n_source = len(source_ids)

    print(f"Loading {n_source} SOURCE embeddings (float32)...")
    source_embeddings = np.stack(
        [np.load(f"{source_path}/{pid}.npy") for pid in tqdm(source_ids)]
    ).astype(np.float32)

    faiss.normalize_L2(source_embeddings)

    print("Building Source Label Matrix (float16)...")
    target_ids = [pid.replace(".npy", "") for pid in os.listdir(target_path)]
    n_target = len(target_ids)

    print(f"Loading {n_target} TARGET embeddings (float16)...")
    target_embeddings = np.stack(
        [np.load(f"{target_path}/{pid}.npy") for pid in tqdm(target_ids)]
    ).astype(np.float16)

    print("Building Index...")
    index = faiss.IndexFlatIP(embed_dim)  # ty:ignore[possibly-missing-attribute]
    index.add(source_embeddings)  # ty:ignore[missing-argument]

    neighbor_priors = np.zeros(shape=(n_target, voyage_embed_dim), dtype=np.float16)
    batch_size = 5

    print("Searching Neighbors...")
    for start in tqdm(range(0, n_target, batch_size)):
        end = min(start + batch_size, n_target)
        queries_f16 = target_embeddings[start:end]
        queries_f32 = queries_f16.astype(np.float32)
        faiss.normalize_L2(queries_f32)

        search_k = k + 1 if target_set == "train" else k
        distances, indices = index.search(queries_f32, search_k)  # ty:ignore[missing-argument]

        for i, neighbors in enumerate(indices):
            if target_set == "train":
                neighbors = neighbors[1:]

            neighbor_text_embs = []
            for n in neighbors:
                protein_id = source_idx_to_pid[n]
                text_emb_path = TEXT_EMBEDDINGS_PATH_TRAIN / f"{protein_id}.npy"
                if text_emb_path.exists():
                    neighbor_text_embs.append(np.load(text_emb_path))

            if neighbor_text_embs:
                avg_text_emb = np.stack(neighbor_text_embs).mean(axis=0)
            else:
                avg_text_emb = np.zeros(voyage_embed_dim, dtype=np.float16)

            neighbor_priors[start + i] = avg_text_emb.astype(np.float16)

        del queries_f16, queries_f32
        gc.collect()

    save_path = (
        TEXT_EMBEDDINGS_TRAIN_NEIGHBOR_MATRIX_PATH
        if target_set == "train"
        else TEXT_EMBEDDINGS_TEST_NEIGHBOR_MATRIX_PATH
    )
    np.save(save_path, neighbor_priors)

    target_pid_to_idx = {pid: i for i, pid in enumerate(target_ids)}
    idx_map_save_path = (
        TEXT_EMBEDDINGS_TRAIN_NEIGHBOR_IDX_PATH
        if target_set == "train"
        else TEXT_EMBEDDINGS_TEST_NEIGHBOR_IDX_PATH
    )
    with open(idx_map_save_path, "wb") as f:
        pickle.dump(target_pid_to_idx, f)


def get_protein_ids_from_fasta(fasta_path):
    """Extract protein IDs from a FASTA file."""
    protein_ids = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        pid = record.id.split("|")[1] if "|" in record.id else record.id
        protein_ids.append(pid)
    return protein_ids


def generate_split_aware_neighbor_priors(
    target_set: Literal["train", "val", "test"], k: int = 6
):
    """
    Generate neighbor priors where:
    - Train proteins find neighbors among OTHER train proteins (excluding self)
    - Val proteins find neighbors among train proteins ONLY (no val proteins)
    - Test proteins find neighbors among train proteins ONLY

    This ensures validation mimics test-time conditions (no data leakage).
    """
    ia_df = pl.read_csv(IA_PATH, separator="\t")
    terms_to_idx = {row["term"]: i for i, row in enumerate(ia_df.iter_rows(named=True))}
    n_terms = len(terms_to_idx)
    embed_dim = MODEL_TO_DIMS[ESM_MODEL]

    # Source is ALWAYS train proteins only
    train_protein_ids = get_protein_ids_from_fasta(TRAIN_FASTA_PATH_SPLIT)
    train_ids_set = set(train_protein_ids)

    source_path = EMBEDDINGS_PATH / f"train/{ESM_MODEL}/mean"

    # Filter source to only include train split proteins
    source_ids = [
        pid.replace(".npy", "")
        for pid in os.listdir(source_path)
        if pid.replace(".npy", "") in train_ids_set
    ]
    source_pid_to_idx = {pid: i for i, pid in enumerate(source_ids)}
    n_source = len(source_ids)

    print(f"Loading {n_source} SOURCE (train-only) embeddings...")
    source_embeddings = np.stack(
        [np.load(f"{source_path}/{pid}.npy") for pid in tqdm(source_ids)]
    ).astype(np.float32)
    faiss.normalize_L2(source_embeddings)

    # Build source labels from train proteins only
    print("Building Source Label Matrix...")
    source_labels = np.zeros((n_source, n_terms), dtype=np.float16)
    all_known_terms_df = pl.read_csv(PARTIAL_SUBMISSION_FULL_PATH)
    valid_rows = all_known_terms_df.filter(pl.col("EntryID").is_in(train_ids_set))

    for row in tqdm(valid_rows.iter_rows(named=True)):
        pid = row["EntryID"]
        term = row["term"]
        if pid in source_pid_to_idx and term in terms_to_idx:
            source_labels[source_pid_to_idx[pid], terms_to_idx[term]] = 1.0

    # Determine target proteins based on target_set
    if target_set == "train":
        target_ids = train_protein_ids
        target_path = source_path
        exclude_self = True
    elif target_set == "val":
        val_protein_ids = get_protein_ids_from_fasta(VAL_FASTA_PATH_SPLIT)
        target_ids = val_protein_ids
        target_path = (
            EMBEDDINGS_PATH / f"train/{ESM_MODEL}/mean"
        )  # val embeddings are in train folder
        exclude_self = False  # val proteins are not in the index
    else:  # test
        target_path = EMBEDDINGS_PATH / f"test/{ESM_MODEL}/mean"
        target_ids = [pid.replace(".npy", "") for pid in os.listdir(target_path)]
        exclude_self = False

    n_target = len(target_ids)
    print(f"Loading {n_target} TARGET ({target_set}) embeddings...")

    target_embeddings = np.stack(
        [np.load(f"{target_path}/{pid}.npy") for pid in tqdm(target_ids)]
    ).astype(np.float16)

    print("Building FAISS Index (train proteins only)...")
    index = faiss.IndexFlatIP(embed_dim)
    index.add(source_embeddings)

    neighbor_priors = np.zeros((n_target, n_terms), dtype=np.float16)
    batch_size = 1000

    print(f"Searching Neighbors for {target_set}...")
    for start in tqdm(range(0, n_target, batch_size)):
        end = min(start + batch_size, n_target)

        queries_f32 = target_embeddings[start:end].astype(np.float32)
        faiss.normalize_L2(queries_f32)

        search_k = k + 1 if exclude_self else k
        distances, indices = index.search(queries_f32, search_k)

        for i, neighbors in enumerate(indices):
            if exclude_self:
                # For train, skip the first result (self-match)
                neighbors = neighbors[1:]

            neighbor_labels = source_labels[neighbors]
            neighbor_priors[start + i] = neighbor_labels.mean(axis=0).astype(np.float16)

        del queries_f32
        gc.collect()

    # Save outputs
    if target_set == "train":
        save_path = TRAIN_NEIGHBOR_MATRIX_PATH
        idx_map_path = TRAIN_NEIGHBOR_MATRIX_IDX_MAP_PATH
    elif target_set == "val":
        save_path = VAL_NEIGHBOR_MATRIX_PATH
        idx_map_path = VAL_NEIGHBOR_MATRIX_IDX_MAP_PATH
    else:
        save_path = TEST_NEIGHBOR_MATRIX_PATH
        idx_map_path = TEST_NEIGHBOR_MATRIX_IDX_MAP_PATH

    np.save(save_path, neighbor_priors)
    print(f"Saved neighbor priors to {save_path}")

    target_pid_to_idx = {pid: i for i, pid in enumerate(target_ids)}
    with open(idx_map_path, "wb") as f:
        pickle.dump(target_pid_to_idx, f)
    print(f"Saved idx map to {idx_map_path}")


def generate_split_aware_text_neighbor_priors(
    target_set: Literal["train", "val", "test"], k: int = 6
):
    """
    Generate text embedding neighbor priors with proper train/val/test separation.
    """
    embed_dim = MODEL_TO_DIMS[ESM_MODEL]
    voyage_embed_dim = 1024

    # Source is ALWAYS train proteins only
    train_protein_ids = get_protein_ids_from_fasta(TRAIN_FASTA_PATH_SPLIT)
    train_ids_set = set(train_protein_ids)

    source_path = EMBEDDINGS_PATH / f"train/{ESM_MODEL}/mean"

    source_ids = [
        pid.replace(".npy", "")
        for pid in os.listdir(source_path)
        if pid.replace(".npy", "") in train_ids_set
    ]
    source_idx_to_pid = {i: pid for i, pid in enumerate(source_ids)}
    n_source = len(source_ids)

    print(f"Loading {n_source} SOURCE (train-only) embeddings...")
    source_embeddings = np.stack(
        [np.load(f"{source_path}/{pid}.npy") for pid in tqdm(source_ids)]
    ).astype(np.float32)
    faiss.normalize_L2(source_embeddings)

    # Determine target proteins
    if target_set == "train":
        target_ids = train_protein_ids
        target_path = source_path
        exclude_self = True
    elif target_set == "val":
        val_protein_ids = get_protein_ids_from_fasta(VAL_FASTA_PATH_SPLIT)
        target_ids = val_protein_ids
        target_path = EMBEDDINGS_PATH / f"train/{ESM_MODEL}/mean"
        exclude_self = False
    else:  # test
        target_path = EMBEDDINGS_PATH / f"test/{ESM_MODEL}/mean"
        target_ids = [pid.replace(".npy", "") for pid in os.listdir(target_path)]
        exclude_self = False

    n_target = len(target_ids)
    print(f"Loading {n_target} TARGET ({target_set}) embeddings...")

    target_embeddings = np.stack(
        [np.load(f"{target_path}/{pid}.npy") for pid in tqdm(target_ids)]
    ).astype(np.float16)

    print("Building FAISS Index (train proteins only)...")
    index = faiss.IndexFlatIP(embed_dim)
    index.add(source_embeddings)

    neighbor_priors = np.zeros((n_target, voyage_embed_dim), dtype=np.float16)
    batch_size = 5

    print(f"Searching Text Neighbors for {target_set}...")
    for start in tqdm(range(0, n_target, batch_size)):
        end = min(start + batch_size, n_target)

        queries_f32 = target_embeddings[start:end].astype(np.float32)
        faiss.normalize_L2(queries_f32)

        search_k = k + 1 if exclude_self else k
        distances, indices = index.search(queries_f32, search_k)

        for i, neighbors in enumerate(indices):
            if exclude_self:
                neighbors = neighbors[1:]

            neighbor_text_embs = []
            for n in neighbors:
                protein_id = source_idx_to_pid[n]
                text_emb_path = TEXT_EMBEDDINGS_PATH_TRAIN / f"{protein_id}.npy"
                if text_emb_path.exists():
                    neighbor_text_embs.append(np.load(text_emb_path))

            if neighbor_text_embs:
                avg_text_emb = np.stack(neighbor_text_embs).mean(axis=0)
            else:
                avg_text_emb = np.zeros(voyage_embed_dim, dtype=np.float16)

            neighbor_priors[start + i] = avg_text_emb.astype(np.float16)

        del queries_f32
        gc.collect()

    # Save outputs
    if target_set == "train":
        save_path = TEXT_EMBEDDINGS_TRAIN_NEIGHBOR_MATRIX_PATH
        idx_map_path = TEXT_EMBEDDINGS_TRAIN_NEIGHBOR_IDX_PATH
    elif target_set == "val":
        save_path = VAL_TEXT_NEIGHBOR_MATRIX_PATH
        idx_map_path = VAL_TEXT_NEIGHBOR_IDX_MAP_PATH
    else:
        save_path = TEXT_EMBEDDINGS_TEST_NEIGHBOR_MATRIX_PATH
        idx_map_path = TEXT_EMBEDDINGS_TEST_NEIGHBOR_IDX_PATH

    np.save(save_path, neighbor_priors)
    print(f"Saved text neighbor priors to {save_path}")

    target_pid_to_idx = {pid: i for i, pid in enumerate(target_ids)}
    with open(idx_map_path, "wb") as f:
        pickle.dump(target_pid_to_idx, f)
    print(f"Saved idx map to {idx_map_path}")


if __name__ == "__main__":
    # Old functions (don't use these anymore)
    # generate_neighbor_priors_matrix("train")
    # generate_neighbor_priors_matrix("test")
    # generate_neighbor_priors_matrix_text_embeddings("train")
    # generate_neighbor_priors_matrix_text_embeddings("test")

    # New split-aware functions - use these!
    print("=== Generating ESM neighbor priors ===")
    generate_split_aware_neighbor_priors("train", k=6)
    generate_split_aware_neighbor_priors("val", k=6)
    generate_split_aware_neighbor_priors("test", k=6)

    print("\n=== Generating Text neighbor priors ===")
    generate_split_aware_text_neighbor_priors("train", k=6)
    generate_split_aware_text_neighbor_priors("val", k=6)
    generate_split_aware_text_neighbor_priors("test", k=6)
