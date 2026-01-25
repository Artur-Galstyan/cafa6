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

from cafa6.config import TrainConfig
from cafa6.constants import (
    DATA_BASE_PATH,
    IA_PATH,
    MASTER_EMBEDDINGS_PATH,
    MASTER_INDEX_PATH,
    MASTER_TAXON_INDEX_PATH,
    SET6_MAX_SCALE_GOOD_PATH,
    TEST_FASTA_PATH,
)

_SHARED_EMBEDDINGS = None
_SHARED_EMBEDDINGS_IDX = None

_SHARED_RAW_EMBEDDINGS = None
_SHARED_RAW_EMBEDDINGS_IDX = None


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


def _load_raw_embeddings(prefix: str = "master_600m"):
    global _SHARED_RAW_EMBEDDINGS, _SHARED_RAW_EMBEDDINGS_IDX
    if _SHARED_RAW_EMBEDDINGS is None:
        idx_path = DATA_BASE_PATH / f"{prefix}_raw_index.json"
        data_path = DATA_BASE_PATH / f"{prefix}_raw_embeddings.dat"

        with open(idx_path, "r") as f:
            _SHARED_RAW_EMBEDDINGS_IDX = json.load(f)

        total_residues = max(
            meta["start"] + meta["length"]
            for meta in _SHARED_RAW_EMBEDDINGS_IDX.values()
        )
        embedding_dim = 1152

        _SHARED_RAW_EMBEDDINGS = np.memmap(
            str(data_path),
            dtype="float16",  # make sure this matches what you wrote
            mode="r",
            shape=(total_residues, embedding_dim),
        )
    return _SHARED_RAW_EMBEDDINGS, _SHARED_RAW_EMBEDDINGS_IDX


class TrainDataSource(RandomAccessDataSource):
    def __init__(
        self,
        protein_ids: list[str],
        terms: list[list[str]],
        emb_indices: list[int],
        taxons: list[int],
    ):
        self.protein_ids = protein_ids
        self.terms = terms
        self.emb_indices = emb_indices
        self.taxons = taxons

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, record_key: SupportsIndex) -> tuple[str, list[str], int, int]:
        return (
            self.protein_ids[record_key],
            self.terms[record_key],
            self.emb_indices[record_key],
            self.taxons[record_key],
        )


class TestDataSource(RandomAccessDataSource):
    def __init__(
        self,
        test_fasta_path: str | Path = TEST_FASTA_PATH,
        master_index_path: str | Path = MASTER_INDEX_PATH,
        master_taxon_path: str | Path = MASTER_TAXON_INDEX_PATH,
    ):
        with open(master_index_path, "r") as f:
            master_index = json.load(f)

        with open(master_taxon_path, "r") as f:
            master_taxon_index = json.load(f)

        self.protein_ids = []
        self.emb_indices = []
        self.taxons = []

        for record in SeqIO.parse(test_fasta_path, "fasta"):
            protein_id = record.id.split(" ")[0]

            if protein_id in master_index:
                self.protein_ids.append(protein_id)
                self.emb_indices.append(master_index[protein_id])
                self.taxons.append(master_taxon_index[protein_id])

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, record_key: SupportsIndex) -> tuple[str, int, int]:
        return (
            self.protein_ids[record_key],
            self.emb_indices[record_key],
            self.taxons[record_key],
        )


class TrainMapTransformMean(Map):
    def __init__(self, terms_to_idx: dict[str, int]):
        self.terms_to_idx = terms_to_idx
        self.n_terms = len(terms_to_idx)

    def map(self, element: tuple[str, list[str], int, int]):
        embeddings, _ = _load_master_embeddings()
        protein_id, terms, emb_idx, taxon = element
        esm_embedding = embeddings[emb_idx]

        indices = [self.terms_to_idx[t] for t in terms if t in self.terms_to_idx]
        labels = np.zeros(self.n_terms, dtype=np.float32)
        if indices:
            labels[indices] = 1.0

        return (protein_id, esm_embedding, int(taxon), labels)


class TrainMapTransformRaw(Map):
    def __init__(self, terms_to_idx: dict[str, int], config: TrainConfig):
        self.terms_to_idx = terms_to_idx
        self.n_terms = len(terms_to_idx)
        self.max_len = config.max_seq_len
        self.embedding_dim = 1152

    def map(self, element: tuple[str, list[str], int, int]):
        raw_data, raw_idx = _load_raw_embeddings()
        protein_id, terms, _, taxon = element

        meta = raw_idx[protein_id]
        start = meta["start"]
        length = meta["length"]

        actual_len = min(length, self.max_len)
        raw_seq = raw_data[start : start + actual_len]

        padded = np.zeros((self.max_len, self.embedding_dim), dtype=np.float32)
        padded[:actual_len] = raw_seq.astype(np.float32)

        indices = [self.terms_to_idx[t] for t in terms if t in self.terms_to_idx]
        labels = np.zeros(self.n_terms, dtype=np.float32)
        if indices:
            labels[indices] = 1.0

        return (protein_id, padded, int(taxon), labels)


class TestMapTransformMean(Map):
    def map(self, element: tuple[str, int, int]):
        embeddings, _ = _load_master_embeddings()
        protein_id, emb_idx, taxon = element
        esm_embedding = embeddings[emb_idx]
        return (protein_id, esm_embedding, int(taxon))


class TestMapTransformRaw(Map):
    def __init__(self, config: TrainConfig):
        self.max_len = config.max_seq_len
        self.embedding_dim = config.esm_embedding_size

    def map(self, element: tuple[str, int, int]):
        raw_data, raw_idx = _load_raw_embeddings()
        protein_id, _, taxon = element

        meta = raw_idx[protein_id]
        start = meta["start"]
        length = meta["length"]

        actual_len = min(length, self.max_len)
        raw_seq = raw_data[start : start + actual_len]

        padded = np.zeros((self.max_len, self.embedding_dim), dtype=np.float32)
        padded[:actual_len] = raw_seq.astype(np.float32)

        return (protein_id, padded, int(taxon))


def get_train_val_datasources(
    labels_path: str | Path = SET6_MAX_SCALE_GOOD_PATH,
    master_index_path: str | Path = MASTER_INDEX_PATH,
    taxon_index_path: str | Path = MASTER_TAXON_INDEX_PATH,
    ratio: float = 0.2,
):
    with open(master_index_path, "r") as f:
        master_index = json.load(f)

    with open(taxon_index_path, "r") as f:
        master_taxons = json.load(f)

    labels_df = pl.read_csv(labels_path, separator="\t")

    all_protein_ids = []
    all_terms = []
    all_emb_indices = []
    all_taxons = []

    for row in labels_df.iter_rows(named=True):
        pid = row["id"]
        if pid in master_index:
            all_protein_ids.append(pid)
            all_terms.append(row["go_term"].split(","))
            all_emb_indices.append(master_index[pid])
            all_taxons.append(master_taxons[pid])

    indices = list(range(len(all_protein_ids)))
    random.seed(42)
    random.shuffle(indices)

    val_size = int(len(indices) * ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_pids = [all_protein_ids[i] for i in train_indices]
    train_terms = [all_terms[i] for i in train_indices]
    train_emb_idx = [all_emb_indices[i] for i in train_indices]
    train_taxons = [all_taxons[i] for i in train_indices]

    val_pids = [all_protein_ids[i] for i in val_indices]
    val_terms = [all_terms[i] for i in val_indices]
    val_emb_idx = [all_emb_indices[i] for i in val_indices]
    val_taxons = [all_taxons[i] for i in val_indices]

    train_ds = TrainDataSource(train_pids, train_terms, train_emb_idx, train_taxons)
    val_ds = TrainDataSource(val_pids, val_terms, val_emb_idx, val_taxons)

    return train_ds, val_ds, len(train_indices)


def get_train_transforms(
    config: TrainConfig,
    drop_remainder: bool = True,
):
    terms_to_idx = {}
    ia_df = pl.read_csv(IA_PATH, separator="\t")
    for i, row in enumerate(ia_df.iter_rows(named=True)):
        terms_to_idx[row["term"]] = i

    if config.used_dataset == "raw":
        transform = TrainMapTransformRaw(terms_to_idx, config)
    else:
        transform = TrainMapTransformMean(terms_to_idx)

    return [
        transform,
        Batch(batch_size=config.batch_size, drop_remainder=drop_remainder),
    ]


def get_val_transforms(
    config: TrainConfig,
    drop_remainder: bool = True,
):
    terms_to_idx = {}
    ia_df = pl.read_csv(IA_PATH, separator="\t")
    for i, row in enumerate(ia_df.iter_rows(named=True)):
        terms_to_idx[row["term"]] = i

    if config.used_dataset == "raw":
        transform = TrainMapTransformRaw(terms_to_idx, config)
    else:
        transform = TrainMapTransformMean(terms_to_idx)

    return [
        transform,
        Batch(batch_size=config.batch_size, drop_remainder=drop_remainder),
    ]


def get_test_transforms(
    config: TrainConfig,
    drop_remainder: bool = True,
):
    if config.used_dataset == "raw":
        transform = TestMapTransformRaw(config=config)
    else:
        transform = TestMapTransformMean()

    return [
        transform,
        Batch(batch_size=config.batch_size, drop_remainder=drop_remainder),
    ]


def create_train_loader(
    train_data_source: TrainDataSource, transformations: list, config: TrainConfig
):
    train_sampler = IndexSampler(
        num_records=len(train_data_source),
        num_epochs=config.num_epochs,
        shard_options=ShardOptions(shard_index=0, shard_count=1, drop_remainder=True),
        shuffle=True,
        seed=42,
    )
    return DataLoader(
        data_source=train_data_source,
        operations=transformations,
        sampler=train_sampler,
        worker_count=config.worker_count,
        read_options=ReadOptions(prefetch_buffer_size=16, num_threads=8),
    )


def create_val_loader(
    val_data_source: TrainDataSource,
    transformations: list,
    config: TrainConfig,
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
        worker_count=config.worker_count,
    )


def get_test_loader(
    config: TrainConfig,
):
    test_data_source = TestDataSource()
    transformations = get_test_transforms(config=config)

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
        worker_count=config.worker_count,
    )
