import pickle
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from Bio import SeqIO
from tqdm import tqdm

from cafa6.constants import (
    DATA_BASE_PATH,
    IA_PATH,
    MATCHA_PATH,
    PARTIAL_SUBMISSION_FULL_PATH,
    TEA_TEST_NEIGHBOR_IDX_MAP_PATH,
    TEA_TEST_NEIGHBOR_MATRIX_PATH,
    TEA_TRAIN_NEIGHBOR_IDX_MAP_PATH,
    TEA_TRAIN_NEIGHBOR_MATRIX_PATH,
    TEST_TEA_PATH,
    TRAIN_FASTA_PATH_SPLIT,
    TRAIN_TEA_PATH,
    VAL_FASTA_PATH_SPLIT,
    VAL_TEA_NEIGHBOR_IDX_MAP_PATH,
    VAL_TEA_NEIGHBOR_MATRIX_PATH,
)


def build_tea_neighbor_priors(
    query_tea_fasta: Path,
    target_tea_fasta: Path,
    results_file: Path,
    k: int = 6,
):
    """Run MMseqs2 with TEA alphabet to find remote homologs."""

    matcha_path = MATCHA_PATH

    cmd = f"""
    mmseqs easy-search {query_tea_fasta} {target_tea_fasta} {results_file} tmp/ \
        --comp-bias-corr 0 \
        --mask 0 \
        --gap-open 18 \
        --gap-extend 3 \
        --sub-mat {matcha_path} \
        --seed-sub-mat {matcha_path} \
        --exact-kmer-matching 1 \
        --max-seqs {k + 1}
    """
    subprocess.run(cmd, shell=True, check=True)

    neighbors = defaultdict(list)
    with open(results_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            query_id = parts[0].split("|")[0]  # handle |H=entropy suffix
            target_id = parts[1].split("|")[0]
            seq_identity = float(parts[2])

            if query_id == target_id:
                continue

            neighbors[query_id].append((target_id, seq_identity))

    for pid in neighbors:
        neighbors[pid] = sorted(neighbors[pid], key=lambda x: -x[1])[:k]

    return dict(neighbors)


def compute_and_save_tea_priors(
    neighbors: dict,
    protein_ids: list[str],
    protein_to_terms: dict[str, list[str]],
    terms_to_idx: dict[str, int],
    n_terms: int,
    output_matrix_path: Path,
    output_idx_map_path: Path,
):
    """Compute GO term priors from TEA neighbors and save as numpy array."""

    n_proteins = len(protein_ids)
    priors = np.zeros((n_proteins, n_terms), dtype=np.float16)
    pid_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

    for i, query_id in enumerate(tqdm(protein_ids, desc="Computing TEA priors")):
        neighbor_list = neighbors.get(query_id, [])

        if neighbor_list:
            for target_id, score in neighbor_list:
                if target_id in protein_to_terms:
                    for term in protein_to_terms[target_id]:
                        if term in terms_to_idx:
                            priors[i, terms_to_idx[term]] += score

            priors[i] /= len(neighbor_list)

    np.save(output_matrix_path, priors)

    with open(output_idx_map_path, "wb") as f:
        pickle.dump(pid_to_idx, f)

    print(f"Saved {n_proteins} priors to {output_matrix_path}")
    print(f"Saved idx map to {output_idx_map_path}")


def generate_tea_priors(target_set: str = "train", k: int = 6):
    """Full pipeline to generate TEA neighbor priors."""

    # Paths
    query_tea_fasta = TRAIN_TEA_PATH if target_set == "train" else TEST_TEA_PATH
    target_tea_fasta = TRAIN_TEA_PATH
    results_file = DATA_BASE_PATH / f"tea_results_{target_set}.m8"

    if target_set == "train":
        output_matrix_path = TEA_TRAIN_NEIGHBOR_MATRIX_PATH
        output_idx_map_path = TEA_TRAIN_NEIGHBOR_IDX_MAP_PATH
    else:
        output_matrix_path = TEA_TEST_NEIGHBOR_MATRIX_PATH
        output_idx_map_path = TEA_TEST_NEIGHBOR_IDX_MAP_PATH

    ia_df = pl.read_csv(IA_PATH, separator="\t")
    terms_to_idx = {row["term"]: i for i, row in enumerate(ia_df.iter_rows(named=True))}
    n_terms = len(terms_to_idx)

    all_known_terms_df = pl.read_csv(PARTIAL_SUBMISSION_FULL_PATH)
    protein_to_terms = defaultdict(list)
    for row in all_known_terms_df.iter_rows(named=True):
        protein_to_terms[row["EntryID"]].append(row["term"])

    protein_ids = []
    with open(query_tea_fasta) as f:
        for line in f:
            if line.startswith(">"):
                pid = line[1:].strip().split("|")[0]
                protein_ids.append(pid)

    print(f"Running TEA search for {target_set}...")
    neighbors = build_tea_neighbor_priors(
        query_tea_fasta, target_tea_fasta, results_file, k=k
    )

    compute_and_save_tea_priors(
        neighbors,
        protein_ids,
        protein_to_terms,
        terms_to_idx,
        n_terms,
        output_matrix_path,
        output_idx_map_path,
    )


def get_protein_ids_from_fasta(fasta_path):
    """Extract protein IDs from a FASTA file."""
    protein_ids = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        pid = record.id.split("|")[1] if "|" in record.id else record.id
        protein_ids.append(pid)
    return set(protein_ids)


def create_split_tea_fasta(source_tea_fasta: Path, split_ids: set, output_path: Path):
    """
    Create a TEA FASTA file containing only proteins in split_ids.
    """
    records_to_write = []
    for record in SeqIO.parse(source_tea_fasta, "fasta"):
        # TEA fasta IDs may have |H=entropy suffix
        pid = record.id.split("|")[0]
        if pid in split_ids:
            records_to_write.append(record)

    with open(output_path, "w") as f:
        SeqIO.write(records_to_write, f, "fasta")

    print(f"Created {output_path} with {len(records_to_write)} sequences")
    return output_path


def generate_split_aware_tea_priors(
    target_set: Literal["train", "val", "test"], k: int = 6
):
    """
    Generate TEA neighbor priors with proper train/val/test separation.

    - Train proteins search against OTHER train proteins (excluding self)
    - Val proteins search against train proteins ONLY
    - Test proteins search against train proteins ONLY

    This ensures validation mimics test-time conditions.
    """
    # Get train/val protein IDs from the split FASTAs
    train_ids = get_protein_ids_from_fasta(TRAIN_FASTA_PATH_SPLIT)
    val_ids = get_protein_ids_from_fasta(VAL_FASTA_PATH_SPLIT)

    # Create train-only TEA FASTA (used as the search target for all queries)
    train_tea_fasta = DATA_BASE_PATH / "train_split_tea.fasta"
    if not train_tea_fasta.exists():
        print("Creating train-only TEA FASTA...")
        create_split_tea_fasta(TRAIN_TEA_PATH, train_ids, train_tea_fasta)

    # Determine query FASTA and output paths
    if target_set == "train":
        query_tea_fasta = train_tea_fasta
        output_matrix_path = TEA_TRAIN_NEIGHBOR_MATRIX_PATH
        output_idx_map_path = TEA_TRAIN_NEIGHBOR_IDX_MAP_PATH
        query_ids = train_ids
    elif target_set == "val":
        # Create val TEA FASTA
        val_tea_fasta = DATA_BASE_PATH / "val_split_tea.fasta"
        if not val_tea_fasta.exists():
            print("Creating val TEA FASTA...")
            create_split_tea_fasta(TRAIN_TEA_PATH, val_ids, val_tea_fasta)
        query_tea_fasta = val_tea_fasta
        output_matrix_path = VAL_TEA_NEIGHBOR_MATRIX_PATH
        output_idx_map_path = VAL_TEA_NEIGHBOR_IDX_MAP_PATH
        query_ids = val_ids
    else:  # test
        query_tea_fasta = TEST_TEA_PATH
        output_matrix_path = TEA_TEST_NEIGHBOR_MATRIX_PATH
        output_idx_map_path = TEA_TEST_NEIGHBOR_IDX_MAP_PATH
        query_ids = None  # will read from file

    results_file = DATA_BASE_PATH / f"tea_results_{target_set}_split.m8"

    # Load term mappings
    ia_df = pl.read_csv(IA_PATH, separator="\t")
    terms_to_idx = {row["term"]: i for i, row in enumerate(ia_df.iter_rows(named=True))}
    n_terms = len(terms_to_idx)

    # Load protein -> terms mapping (only from train proteins for labels!)
    all_known_terms_df = pl.read_csv(PARTIAL_SUBMISSION_FULL_PATH)
    # Filter to only train proteins for the label lookup
    train_terms_df = all_known_terms_df.filter(pl.col("EntryID").is_in(train_ids))
    protein_to_terms = defaultdict(list)
    for row in train_terms_df.iter_rows(named=True):
        protein_to_terms[row["EntryID"]].append(row["term"])

    # Get query protein IDs
    protein_ids = []
    with open(query_tea_fasta) as f:
        for line in f:
            if line.startswith(">"):
                pid = line[1:].strip().split("|")[0]
                protein_ids.append(pid)

    print(
        f"Running TEA search for {target_set} ({len(protein_ids)} queries) against train ({len(train_ids)} targets)..."
    )

    # Run MMseqs2 search: query against train-only TEA
    neighbors = build_tea_neighbor_priors(
        query_tea_fasta, train_tea_fasta, results_file, k=k
    )

    # Compute and save priors
    compute_and_save_tea_priors(
        neighbors,
        protein_ids,
        protein_to_terms,
        terms_to_idx,
        n_terms,
        output_matrix_path,
        output_idx_map_path,
    )


if __name__ == "__main__":
    # Old function (don't use)
    # generate_tea_priors("train", k=6)
    # generate_tea_priors("test", k=6)

    # New split-aware functions
    print("=== Generating split-aware TEA priors ===")
    generate_split_aware_tea_priors("train", k=6)
    generate_split_aware_tea_priors("val", k=6)
    generate_split_aware_tea_priors("test", k=6)
