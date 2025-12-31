import random
import re
from typing import cast

import polars as pl
from beartype.typing import SupportsIndex
from Bio import SeqIO
from grain.sources import RandomAccessDataSource


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
