import gc
import os
import pickle

import faiss
import numpy as np
import polars as pl
from beartype.typing import Literal
from tqdm import tqdm

from cafa6.constants import (
    EMBEDDINGS_PATH,
    ESM_MODEL,
    IA_PATH,
    MODEL_TO_DIMS,
    PARTIAL_SUBMISSION_FULL_PATH,
    TEST_NEIGHBOR_MATRIX_IDX_MAP_PATH,
    TEST_NEIGHBOR_MATRIX_PATH,
    TRAIN_NEIGHBOR_MATRIX_IDX_MAP_PATH,
    TRAIN_NEIGHBOR_MATRIX_PATH,
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


if __name__ == "__main__":
    # generate_neighbor_priors_matrix("train")
    generate_neighbor_priors_matrix("test")
