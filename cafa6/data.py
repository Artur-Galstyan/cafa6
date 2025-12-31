import random
import re
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from beartype.typing import Literal, SupportsIndex
from Bio import SeqIO
from grain.sources import RandomAccessDataSource
from grain.transforms import Map

from cafa6.constants import EMBEDDINGS_PATH, MODEL_TO_DIMS

_SHARED_PRIORS = None


class DataSource(RandomAccessDataSource):
    def __init__(
        self,
        train_fasta_path: str,
        train_terms_path: str,
        indices: list[int] | None = None,
    ):
        self.sequences = []
        self.protein_ids = []
        self.terms = []
        self.taxa = []

        terms_df = cast(pl.DataFrame, pl.read_csv(train_terms_path, separator="\t"))
        terms_grouped = terms_df.group_by("EntryID").agg(pl.col("term")).to_dict()
        entry_to_terms = dict(zip(terms_grouped["EntryID"], terms_grouped["term"]))

        all_sequences = []
        all_protein_ids = []
        all_terms = []
        all_taxi = []

        for record in SeqIO.parse(train_fasta_path, "fasta"):
            all_sequences.append(str(record.seq))
            protein_id = record.id.split("|")[1]
            description = record.description
            taxon = int(re.search(r"(?<=OX=)\d+", description).group(0))  # ty:ignore[possibly-missing-attribute]

            all_taxi.append(taxon)
            all_protein_ids.append(protein_id)
            all_terms.append(entry_to_terms.get(protein_id, []))

        if indices is None:
            indices = list(range(len(all_sequences)))

        for i in indices:
            self.sequences.append(all_sequences[i])
            self.protein_ids.append(all_protein_ids[i])
            self.terms.append(all_terms[i])
            self.taxa.append(all_taxi[i])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, record_key: SupportsIndex) -> tuple[str, str, list[str], int]:
        return (
            self.protein_ids[record_key],
            self.sequences[record_key],
            self.terms[record_key],
            self.taxa[record_key],
        )


def get_datasources(train_fasta_path, train_terms_path, ratio=0.1):
    n_total = 0
    for record in SeqIO.parse(train_fasta_path, "fasta"):
        n_total += 1

    all_indices = list(range(n_total))
    random.seed(42)
    random.shuffle(all_indices)

    val_size = int(n_total * ratio)
    val_indices = all_indices[:val_size]
    train_indices = all_indices[val_size:]

    train_data_source = DataSource(
        str(train_fasta_path), str(train_terms_path), train_indices
    )
    val_data_source = DataSource(
        str(train_fasta_path), str(train_terms_path), val_indices
    )

    return train_data_source, val_data_source


class MapTermsToArrayAndEmbeddings(Map):
    def __init__(
        self,
        terms_to_idx_weight: dict,
        neighbor_priors_path: str,
        protein_ids_path: str,
        esm_model: str,
        max_protein_seq: int = 1024,
        esm_strategy: Literal["mean", "raw"] = "mean",
        embedding_base_path: str | Path = EMBEDDINGS_PATH,
    ):
        self.terms_to_idx_weight = terms_to_idx_weight

        self.neighbor_priors_path = neighbor_priors_path
        protein_ids = np.load(protein_ids_path)
        self.pid_to_prior_idx = {pid: i for i, pid in enumerate(protein_ids)}
        self.embedding_base_path = Path(embedding_base_path)
        self.esm_model = esm_model
        self.max_protein_seq = max_protein_seq
        self.esm_strategy = esm_strategy

    def map(self, element: tuple[str, str, list[str], int]):
        global _SHARED_PRIORS
        if _SHARED_PRIORS is None:
            _SHARED_PRIORS = np.load(self.neighbor_priors_path, mmap_mode="r")

        protein_id, sequence, terms, taxon = element
        taxon = np.array(taxon)
        indices = np.array([self.terms_to_idx_weight[t][0] for t in terms])

        labels = np.zeros(len(self.terms_to_idx_weight), dtype=np.float32)
        labels[indices] = 1.0

        esm_path = (
            self.embedding_base_path
            / "train"
            / self.esm_model
            / self.esm_strategy
            / f"{protein_id}.npy"
        )

        embeddings_esm = np.zeros(
            shape=(self.max_protein_seq, MODEL_TO_DIMS[self.esm_model])
        )

        mask = np.zeros((self.max_protein_seq,), dtype=np.float32)

        if esm_path.exists():
            embeddings_esm_non_trunc = np.load(str(esm_path))
            if self.esm_strategy == "raw":
                seq_len, _ = embeddings_esm_non_trunc.shape
                lens = min(self.max_protein_seq, seq_len)
                embeddings_esm[:lens] = embeddings_esm_non_trunc[:lens]
                mask[:lens] = 1.0
            elif self.esm_strategy == "mean":
                embeddings_esm = embeddings_esm_non_trunc
        else:
            raise FileNotFoundError(f"{esm_path} not found!")

        prior_idx = self.pid_to_prior_idx.get(protein_id)
        if prior_idx is not None:
            neighbor_prior = _SHARED_PRIORS[prior_idx]
        else:
            neighbor_prior = np.zeros(len(self.terms_to_idx_weight), dtype=np.float32)

        return protein_id, embeddings_esm, neighbor_prior, taxon, mask, labels
