import pickle
import random
import re
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from beartype.typing import Literal, SupportsIndex
from Bio import SeqIO
from grain import DataLoader, ReadOptions
from grain.samplers import IndexSampler
from grain.sharding import ShardOptions
from grain.sources import RandomAccessDataSource
from grain.transforms import Batch, Map

from cafa6.constants import (
    DATA_BASE_PATH,
    EMBEDDINGS_PATH,
    ESM_MODEL,
    IA_PATH,
    MODEL_TO_DIMS,
    TEST_FASTA_PATH,
    TEST_NEIGHBOR_MATRIX_IDX_MAP_PATH,
    TEST_NEIGHBOR_MATRIX_PATH,
    TEST_SUPERSET_TAXON_LOOKUP_PATH,
    TEXT_EMBEDDING_SIZE,
    TEXT_EMBEDDINGS_TEST_NEIGHBOR_IDX_PATH,
    TEXT_EMBEDDINGS_TEST_NEIGHBOR_MATRIX_PATH,
    TEXT_EMBEDDINGS_TRAIN_NEIGHBOR_IDX_PATH,
    TEXT_EMBEDDINGS_TRAIN_NEIGHBOR_MATRIX_PATH,
    TRAIN_FASTA_EXTENDED_CORRECTED_PATH,
    TRAIN_FASTA_PATH,
    TRAIN_NEIGHBOR_MATRIX_IDX_MAP_PATH,
    TRAIN_NEIGHBOR_MATRIX_PATH,
    TRAIN_TERMS_EXTENDED_PATH,
)

_SHARED_PRIORS = None
_SHARED_TEXT_PRIORS = None


class TestDataSource(RandomAccessDataSource):
    def __init__(
        self,
        test_fasta_path: str | Path = TEST_FASTA_PATH,
    ):
        self.sequences = []
        self.protein_ids = []
        self.taxa = []

        all_sequences = []
        all_protein_ids = []
        all_taxi = []

        for record in SeqIO.parse(test_fasta_path, "fasta"):
            all_sequences.append(str(record.seq))
            protein_id = record.id.split(" ")[0]
            taxon = int(record.description.split(" ")[1])

            all_taxi.append(taxon)
            all_protein_ids.append(protein_id)

        indices = list(range(len(all_sequences)))

        for i in indices:
            self.sequences.append(all_sequences[i])
            self.protein_ids.append(all_protein_ids[i])
            self.taxa.append(all_taxi[i])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, record_key: SupportsIndex) -> tuple[str, list[str], int]:
        return (
            self.protein_ids[record_key],
            self.sequences[record_key],
            self.taxa[record_key],
        )


def get_test_loader(batch_size: int = 128, worker_count: int = 4):
    test_data_source = TestDataSource()
    terms_to_idx_weight = {}
    ia_df = pl.read_csv(IA_PATH, separator="\t")
    for i, row in enumerate(ia_df.iter_rows(named=True)):
        terms_to_idx_weight[row["term"]] = (i, row["weight"])

    transformations = [
        TestMapToArrayAndEmbeddings(
            str(TEST_NEIGHBOR_MATRIX_PATH),
            str(TEST_NEIGHBOR_MATRIX_IDX_MAP_PATH),
            str(TEXT_EMBEDDINGS_TEST_NEIGHBOR_MATRIX_PATH),
            str(TEXT_EMBEDDINGS_TEST_NEIGHBOR_IDX_PATH),
            ESM_MODEL,
        ),
        Batch(batch_size=batch_size, drop_remainder=False),
    ]

    sampler = IndexSampler(
        num_records=len(test_data_source),
        num_epochs=1,
        shard_options=ShardOptions(shard_index=0, shard_count=1, drop_remainder=True),
        shuffle=False,
        seed=42,
    )
    return DataLoader(
        data_source=test_data_source,
        operations=transformations,
        sampler=sampler,
        worker_count=worker_count,
    )


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
            protein_id = record.id if "|" not in record.id else record.id.split("|")[1]
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


_SHARED_TEST_PRIORS = None
_SHARED_TEST_TEXT_PRIORS = None


class TestMapToArrayAndEmbeddings(Map):
    def __init__(
        self,
        neighbor_priors_path: str,
        neighbor_priors_idx_map_path: str,
        text_neighbor_priors_path: str,
        text_neighbor_priors_idx_map_path: str,
        esm_model: str,
        max_protein_seq: int = 1024,
        esm_strategy: Literal["mean", "raw"] = "mean",
        embedding_base_path: str | Path = EMBEDDINGS_PATH,
        taxon_lookup_path: str | Path = TEST_SUPERSET_TAXON_LOOKUP_PATH,
    ):
        self.neighbor_priors_path = neighbor_priors_path
        self.neighbor_priors_idx_map_path = neighbor_priors_idx_map_path
        self.text_neighbor_priors_path = text_neighbor_priors_path
        self.text_neighbor_priors_idx_map_path = text_neighbor_priors_idx_map_path
        self.taxon_lookup_path = taxon_lookup_path

        with open(neighbor_priors_idx_map_path, "rb") as f:
            self.pid_to_prior_idx = pickle.load(f)

        with open(text_neighbor_priors_idx_map_path, "rb") as f:
            self.text_pid_to_prior_idx = pickle.load(f)

        self.taxon_to_idx = {}
        taxon_idx_df = pl.read_csv(self.taxon_lookup_path)
        for row in taxon_idx_df.iter_rows(named=True):
            self.taxon_to_idx[row["ID"]] = row["idx"]

        self.embedding_base_path = Path(embedding_base_path)
        self.esm_model = esm_model
        self.max_protein_seq = max_protein_seq
        self.esm_strategy = esm_strategy

    def map(self, element: tuple[str, list[str], int]):
        global _SHARED_TEST_PRIORS, _SHARED_TEST_TEXT_PRIORS

        if _SHARED_TEST_PRIORS is None:
            _SHARED_TEST_PRIORS = np.load(self.neighbor_priors_path, mmap_mode="r")

        if _SHARED_TEST_TEXT_PRIORS is None:
            _SHARED_TEST_TEXT_PRIORS = np.load(
                self.text_neighbor_priors_path, mmap_mode="r"
            )

        protein_id, sequence, taxon = element

        esm_path = (
            self.embedding_base_path
            / "test"
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
            neighbor_prior = _SHARED_TEST_PRIORS[prior_idx]
        else:
            raise KeyError(f"Couldn't find priors for {protein_id}")

        text_prior_idx = self.text_pid_to_prior_idx.get(protein_id)
        if text_prior_idx is not None:
            text_neighbor_prior = _SHARED_TEST_TEXT_PRIORS[text_prior_idx]
        else:
            text_neighbor_prior = np.zeros(TEXT_EMBEDDING_SIZE, dtype=np.float32)

        return (
            protein_id,
            embeddings_esm,
            neighbor_prior,
            text_neighbor_prior,
            self.taxon_to_idx[taxon],
            mask,
        )


def get_datasources(
    train_fasta_path=TRAIN_FASTA_EXTENDED_CORRECTED_PATH,
    train_terms_path=TRAIN_TERMS_EXTENDED_PATH,
    original_fasta_path=TRAIN_FASTA_PATH,
    ratio=0.1,
):
    original_ids = set()
    for record in SeqIO.parse(original_fasta_path, "fasta"):
        pid = record.id.split("|")[1] if "|" in record.id else record.id
        original_ids.add(pid)

    all_proteins = []
    for record in SeqIO.parse(train_fasta_path, "fasta"):
        pid = record.id.split("|")[1] if "|" in record.id else record.id
        all_proteins.append(pid)

    original_indices = [i for i, pid in enumerate(all_proteins) if pid in original_ids]
    new_indices = [i for i, pid in enumerate(all_proteins) if pid not in original_ids]

    random.seed(42)
    random.shuffle(original_indices)
    val_size = int(len(original_indices) * ratio)
    val_indices = original_indices[:val_size]

    train_indices = original_indices[val_size:] + new_indices

    train_data_source = DataSource(
        str(train_fasta_path), str(train_terms_path), train_indices
    )
    val_data_source = DataSource(
        str(train_fasta_path), str(train_terms_path), val_indices
    )

    return train_data_source, val_data_source, len(train_indices)


_SHARED_EMBEDDINGS = None
_SHARED_EMBEDDINGS_IDX = None


class MapTermsToArrayAndEmbeddings(Map):
    def __init__(
        self,
        terms_to_idx_weight: dict,
        neighbor_priors_path: str,
        neighbor_priors_idx_map_path: str,
        text_neighbor_priors_path: str,
        text_neighbor_priors_idx_map_path: str,
        esm_model: str,
        max_protein_seq: int = 1024,
        esm_strategy: Literal["mean", "raw"] = "mean",
        embedding_base_path: str | Path = EMBEDDINGS_PATH,
        taxon_lookup_path: str | Path = TEST_SUPERSET_TAXON_LOOKUP_PATH,
        consolidated_emb_path: str | Path | None = None,
        consolidated_idx_path: str | Path | None = None,
    ):
        self.terms_to_idx_weight = terms_to_idx_weight
        self.neighbor_priors_path = neighbor_priors_path
        self.neighbor_priors_idx_map_path = neighbor_priors_idx_map_path

        self.text_neighbor_priors_path = text_neighbor_priors_path
        self.text_neighbor_priors_idx_map_path = text_neighbor_priors_idx_map_path

        self.taxon_lookup_path = taxon_lookup_path

        self.consolidated_emb_path = consolidated_emb_path
        self.consolidated_idx_path = consolidated_idx_path

        with open(neighbor_priors_idx_map_path, "rb") as f:
            self.pid_to_prior_idx = pickle.load(f)

        with open(text_neighbor_priors_idx_map_path, "rb") as f:
            self.text_pid_to_prior_idx = pickle.load(f)

        self.taxon_to_idx = {}
        taxon_idx_df = pl.read_csv(self.taxon_lookup_path)
        for row in taxon_idx_df.iter_rows(named=True):
            self.taxon_to_idx[row["ID"]] = row["idx"]

        self.embedding_base_path = Path(embedding_base_path)
        self.esm_model = esm_model
        self.max_protein_seq = max_protein_seq
        self.esm_strategy = esm_strategy

    def map(self, element: tuple[str, str, list[str], int]):
        assert self.consolidated_idx_path is not None
        global \
            _SHARED_PRIORS, \
            _SHARED_EMBEDDINGS, \
            _SHARED_EMBEDDINGS_IDX, \
            _SHARED_TEXT_PRIORS

        if _SHARED_PRIORS is None:
            _SHARED_PRIORS = np.load(self.neighbor_priors_path, mmap_mode="r")

        if _SHARED_TEXT_PRIORS is None:
            _SHARED_TEXT_PRIORS = np.load(self.text_neighbor_priors_path, mmap_mode="r")

        if self.consolidated_emb_path and _SHARED_EMBEDDINGS is None:
            _SHARED_EMBEDDINGS = np.load(self.consolidated_emb_path, mmap_mode="r")
            with open(self.consolidated_idx_path, "rb") as f:
                _SHARED_EMBEDDINGS_IDX = pickle.load(f)

        assert _SHARED_EMBEDDINGS_IDX is not None

        protein_id, sequence, terms, taxon = element
        indices = np.array([self.terms_to_idx_weight[t][0] for t in terms])

        labels = np.zeros(len(self.terms_to_idx_weight), dtype=np.float32)
        labels[indices] = 1.0

        if _SHARED_EMBEDDINGS is not None and protein_id in _SHARED_EMBEDDINGS_IDX:
            emb_idx = _SHARED_EMBEDDINGS_IDX[protein_id]
            embeddings_esm = _SHARED_EMBEDDINGS[emb_idx]
            mask = np.zeros((self.max_protein_seq,), dtype=np.float32)
        else:
            # fallback
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

        text_prior_idx = self.text_pid_to_prior_idx.get(protein_id)

        if text_prior_idx is not None:
            text_neighbor_prior = _SHARED_TEXT_PRIORS[text_prior_idx]
        else:
            text_neighbor_prior = np.zeros(TEXT_EMBEDDING_SIZE, dtype=np.float32)

        return (
            protein_id,
            embeddings_esm,
            neighbor_prior,
            text_neighbor_prior,
            self.taxon_to_idx[taxon],
            mask,
            labels,
        )


def get_transformations(batch_size: int = 128, drop_remainder=True):
    terms_to_idx_weight = {}
    ia_df = pl.read_csv(IA_PATH, separator="\t")
    for i, row in enumerate(ia_df.iter_rows(named=True)):
        terms_to_idx_weight[row["term"]] = (i, row["weight"])

    transformations = [
        MapTermsToArrayAndEmbeddings(
            terms_to_idx_weight,
            str(TRAIN_NEIGHBOR_MATRIX_PATH),
            str(TRAIN_NEIGHBOR_MATRIX_IDX_MAP_PATH),
            str(TEXT_EMBEDDINGS_TRAIN_NEIGHBOR_MATRIX_PATH),  # ADD THIS
            str(TEXT_EMBEDDINGS_TRAIN_NEIGHBOR_IDX_PATH),  # ADD THIS
            ESM_MODEL,
            consolidated_emb_path=str(
                DATA_BASE_PATH / f"embeddings_train_{ESM_MODEL}_mean.npy"
            ),
            consolidated_idx_path=str(
                DATA_BASE_PATH / f"embeddings_train_{ESM_MODEL}_mean_idx.pkl"
            ),
        ),
        Batch(batch_size=batch_size, drop_remainder=drop_remainder),
    ]
    return transformations


def create_train_loader(
    train_data_source,
    transformations,
    batch_size: int = 128,
    num_epochs: int = 8,
    worker_count: int = 0,
):
    train_sampler = IndexSampler(
        num_records=len(train_data_source),
        num_epochs=num_epochs,
        shard_options=ShardOptions(shard_index=0, shard_count=1, drop_remainder=True),
        shuffle=True,
        seed=42,
    )
    return DataLoader(
        data_source=train_data_source,
        operations=transformations,
        sampler=train_sampler,
        worker_count=worker_count,
        read_options=ReadOptions(prefetch_buffer_size=16, num_threads=8),
    )


def create_val_loader(val_data_source, transformations):
    val_sampler = IndexSampler(
        num_records=len(val_data_source),
        num_epochs=1,
        shard_options=ShardOptions(shard_index=0, shard_count=1, drop_remainder=True),
        shuffle=False,
        seed=0,
    )
    return DataLoader(
        data_source=val_data_source,
        operations=transformations,
        sampler=val_sampler,
        worker_count=0,
    )
