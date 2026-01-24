import json
import math
import os

import numpy as np
import torch
import torch.multiprocessing as mp
from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from tqdm import tqdm

from cafa6.constants import (
    DATA_BASE_PATH,
    MASTER_INDEX_PATH,
    TEST_FASTA_PATH,
    TRAIN_FASTA_PATH,
    UNIPROT_SPROT_FASTA_PATH,
)


def gpu_worker(
    rank, chunks, sequences, offset_map, memmap_path, total_residues, max_seq_len
):
    ids_chunk = chunks[rank]
    device = f"cuda:{rank}"

    print(f"[GPU {rank}] Initializing model on {device}...")
    client = ESMC.from_pretrained("esmc_600m").to(device)
    client.eval()

    embedding_dim = 1152
    fp = np.memmap(
        memmap_path, dtype="float16", mode="r+", shape=(total_residues, embedding_dim)
    )

    print(f"[GPU {rank}] Processing {len(ids_chunk)} proteins...")

    for pid in tqdm(ids_chunk, position=rank, desc=f"GPU {rank}"):
        if pid not in sequences:
            continue

        seq = sequences[pid]
        trunc_seq = seq[:max_seq_len]

        try:
            protein_obj = ESMProtein(sequence=trunc_seq)
            protein_tensor = client.encode(protein_obj)

            logits_output = client.logits(
                protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )

            emb = logits_output.embeddings.squeeze(0).detach().cpu().numpy()

            if emb.shape[0] != len(trunc_seq):
                if emb.shape[0] == len(trunc_seq) + 2:
                    emb = emb[1:-1]

            start = offset_map[pid]["start"]
            end = start + len(trunc_seq)

            fp[start:end] = emb[: len(trunc_seq)]

        except Exception as e:
            print(f"[GPU {rank}] Error processing {pid}: {e}")

    fp.flush()
    print(f"[GPU {rank}] Done.")


def generate_raw_embeddings_packed(
    train_fasta,
    test_fasta,
    uniprot_fasta_path,
    master_index_path,
    output_prefix,
    max_seq_len=1024,
    num_gpus=2,
):
    raw_index_path = f"{output_prefix}_raw_index.json"
    memmap_path = f"{output_prefix}_raw_embeddings.dat"

    if os.path.exists(raw_index_path) and os.path.exists(memmap_path):
        print(f"Files already exist at {memmap_path}. Skipping generation.")
        return

    print("Loading Master Index to identify targets...")
    with open(master_index_path, "r") as f:
        master_index = json.load(f)

    sorted_ids = sorted(master_index.keys(), key=lambda k: master_index[k])
    target_ids_set = set(sorted_ids)

    sequences = {}
    print("Gathering sequences...")
    for path in [train_fasta, test_fasta, uniprot_fasta_path]:
        for record in SeqIO.parse(path, "fasta"):
            pid = record.id.split("|")[1] if "|" in record.id else record.id
            if pid in target_ids_set:
                sequences[pid] = str(record.seq)

    print("Pass 1: Calculating packed storage layout...")
    offset_map = {}
    total_residues = 0

    for pid in sorted_ids:
        if pid not in sequences:
            continue

        seq_len = min(len(sequences[pid]), max_seq_len)
        offset_map[pid] = {"start": total_residues, "length": seq_len}
        total_residues += seq_len

    print(f"Total Residues to store: {total_residues}")
    print(f"Estimated size: {total_residues * 1152 * 2 / (1024**3):.2f} GB (float16)")

    with open(raw_index_path, "w") as f:
        json.dump(offset_map, f)

    embedding_dim = 1152
    fp = np.memmap(
        memmap_path, dtype="float16", mode="w+", shape=(total_residues, embedding_dim)
    )
    del fp

    print("Pass 2: Distributed Computing...")

    chunk_size = math.ceil(len(sorted_ids) / num_gpus)
    chunks = [
        sorted_ids[i : i + chunk_size] for i in range(0, len(sorted_ids), chunk_size)
    ]

    mp.spawn(
        gpu_worker,
        args=(chunks, sequences, offset_map, memmap_path, total_residues, max_seq_len),
        nprocs=len(chunks),
        join=True,
    )

    print("Done. Raw Embeddings Generated.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    generate_raw_embeddings_packed(
        train_fasta=TRAIN_FASTA_PATH,
        test_fasta=TEST_FASTA_PATH,
        uniprot_fasta_path=UNIPROT_SPROT_FASTA_PATH,
        master_index_path=MASTER_INDEX_PATH,
        output_prefix=DATA_BASE_PATH / "master_600m",
        max_seq_len=1024,
        num_gpus=2,
    )
