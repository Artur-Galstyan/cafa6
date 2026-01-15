import json
import random
from pathlib import Path

import numpy as np
import polars as pl
from beartype.typing import SupportsIndex
from Bio import SeqIO
from grain import DataLoader, ReadOptions
from grain.samplers import IndexSampler
from grain.sharding import ShardOptions
from grain.sources import RandomAccessDataSource
from grain.transforms import Batch, Map

from cafa6.constants import (
    IA_PATH,
    MASTER_EMBEDDINGS_PATH,
    MASTER_INDEX_PATH,
    SET6_MAX_SCALE_GOOD_PATH,
    TEST_FASTA_PATH,
)

_SHARED_EMBEDDINGS = None
_SHARED_EMBEDDINGS_IDX = None


def _load_master_embeddings():
    global _SHARED_EMBEDDINGS, _SHARED_EMBEDDINGS_IDX
    if _SHARED_EMBEDDINGS is None:
        with open(MASTER_INDEX_PATH, "r") as f:
            _SHARED_EMBEDDINGS_IDX = json.load(f)
        num_proteins = len(_SHARED_EMBEDDINGS_IDX)
        embedding_dim = 1152
        _SHARED_EMBEDDINGS = np.memmap(
            str(MASTER_EMBEDDINGS_PATH),
            dtype="float32",
            mode="r",
            shape=(num_proteins, embedding_dim),
        )
    return _SHARED_EMBEDDINGS, _SHARED_EMBEDDINGS_IDX


class TrainDataSource(RandomAccessDataSource):
    def __init__(
        self,
        protein_ids: list[str],
        terms: list[list[str]],
        emb_indices: list[int],
    ):
        self.protein_ids = protein_ids
        self.terms = terms
        self.emb_indices = emb_indices

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, record_key: SupportsIndex) -> tuple[str, list[str], int]:
        return (
            self.protein_ids[record_key],
            self.terms[record_key],
            self.emb_indices[record_key],
        )


class TestDataSource(RandomAccessDataSource):
    def __init__(
        self,
        test_fasta_path: str | Path = TEST_FASTA_PATH,
        master_index_path: str | Path = MASTER_INDEX_PATH,
    ):
        with open(master_index_path, "r") as f:
            master_index = json.load(f)

        self.protein_ids = []
        self.taxa = []
        self.emb_indices = []

        for record in SeqIO.parse(test_fasta_path, "fasta"):
            protein_id = record.id.split(" ")[0]
            taxon = int(record.description.split(" ")[1])

            if protein_id in master_index:
                self.protein_ids.append(protein_id)
                self.taxa.append(taxon)
                self.emb_indices.append(master_index[protein_id])

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, record_key: SupportsIndex) -> tuple[str, int, int]:
        return (
            self.protein_ids[record_key],
            self.taxa[record_key],
            self.emb_indices[record_key],
        )


class TrainMapTransform(Map):
    def __init__(self, terms_to_idx: dict[str, int]):
        self.terms_to_idx = terms_to_idx
        self.n_terms = len(terms_to_idx)

    def map(self, element: tuple[str, list[str], int]):
        embeddings, _ = _load_master_embeddings()

        protein_id, terms, emb_idx = element

        esm_embedding = embeddings[emb_idx]

        indices = [self.terms_to_idx[t] for t in terms if t in self.terms_to_idx]
        labels = np.zeros(self.n_terms, dtype=np.float32)
        if indices:
            labels[indices] = 1.0

        return (
            protein_id,
            esm_embedding,
            labels,
        )


class TestMapTransform(Map):
    def map(self, element: tuple[str, int, int]):
        embeddings, _ = _load_master_embeddings()

        protein_id, taxon, emb_idx = element

        esm_embedding = embeddings[emb_idx]

        return (
            protein_id,
            esm_embedding,
            taxon,
        )


def get_train_val_datasources(
    labels_path: str | Path = SET6_MAX_SCALE_GOOD_PATH,
    master_index_path: str | Path = MASTER_INDEX_PATH,
    ratio: float = 0.2,
):
    with open(master_index_path, "r") as f:
        master_index = json.load(f)

    labels_df = pl.read_csv(labels_path, separator="\t")

    all_protein_ids = []
    all_terms = []
    all_emb_indices = []

    for row in labels_df.iter_rows(named=True):
        pid = row["id"]
        if pid in master_index:
            all_protein_ids.append(pid)
            all_terms.append(row["go_term"].split(","))
            all_emb_indices.append(master_index[pid])

    indices = list(range(len(all_protein_ids)))
    random.seed(42)
    random.shuffle(indices)

    val_size = int(len(indices) * ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_pids = [all_protein_ids[i] for i in train_indices]
    train_terms = [all_terms[i] for i in train_indices]
    train_emb_idx = [all_emb_indices[i] for i in train_indices]

    val_pids = [all_protein_ids[i] for i in val_indices]
    val_terms = [all_terms[i] for i in val_indices]
    val_emb_idx = [all_emb_indices[i] for i in val_indices]

    train_ds = TrainDataSource(train_pids, train_terms, train_emb_idx)
    val_ds = TrainDataSource(val_pids, val_terms, val_emb_idx)

    return train_ds, val_ds, len(train_indices)


def get_train_datasource(labels_path: str | Path = SET6_MAX_SCALE_GOOD_PATH):
    with open(MASTER_INDEX_PATH, "r") as f:
        master_index = json.load(f)

    labels_df = pl.read_csv(labels_path, separator="\t")

    protein_ids = []
    terms = []
    emb_indices = []

    for row in labels_df.iter_rows(named=True):
        pid = row["id"]
        if pid in master_index:
            protein_ids.append(pid)
            terms.append(row["go_term"].split(","))
            emb_indices.append(master_index[pid])

    return TrainDataSource(protein_ids, terms, emb_indices)


def get_test_datasource():
    return TestDataSource()


def get_train_transforms(batch_size: int = 128, drop_remainder: bool = True):
    terms_to_idx = {}
    ia_df = pl.read_csv(IA_PATH, separator="\t")
    for i, row in enumerate(ia_df.iter_rows(named=True)):
        terms_to_idx[row["term"]] = i

    return [
        TrainMapTransform(terms_to_idx),
        Batch(batch_size=batch_size, drop_remainder=drop_remainder),
    ]


def get_val_transforms(batch_size: int = 128, drop_remainder: bool = False):
    terms_to_idx = {}
    ia_df = pl.read_csv(IA_PATH, separator="\t")
    for i, row in enumerate(ia_df.iter_rows(named=True)):
        terms_to_idx[row["term"]] = i

    return [
        TrainMapTransform(terms_to_idx),
        Batch(batch_size=batch_size, drop_remainder=drop_remainder),
    ]


def get_test_transforms(batch_size: int = 128, drop_remainder: bool = False):
    return [
        TestMapTransform(),
        Batch(batch_size=batch_size, drop_remainder=drop_remainder),
    ]


def create_train_loader(
    train_data_source: TrainDataSource,
    transformations: list,
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


def create_val_loader(
    val_data_source: TrainDataSource,
    transformations: list,
    worker_count: int = 0,
):
    val_sampler = IndexSampler(
        num_records=len(val_data_source),
        num_epochs=1,
        shard_options=ShardOptions(shard_index=0, shard_count=1, drop_remainder=False),
        shuffle=False,
        seed=42,
    )
    return DataLoader(
        data_source=val_data_source,
        operations=transformations,
        sampler=val_sampler,
        worker_count=worker_count,
    )


def get_test_loader(batch_size: int = 128, worker_count: int = 4):
    test_data_source = TestDataSource()
    transformations = get_test_transforms(batch_size=batch_size)

    sampler = IndexSampler(
        num_records=len(test_data_source),
        num_epochs=1,
        shard_options=ShardOptions(shard_index=0, shard_count=1, drop_remainder=False),
        shuffle=False,
        seed=42,
    )
    return DataLoader(
        data_source=test_data_source,
        operations=transformations,
        sampler=sampler,
        worker_count=worker_count,
    )
