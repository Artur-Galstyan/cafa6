import pickle

import numpy as np
import polars as pl
from Bio import Seq, SeqIO, SeqRecord

from cafa6.constants import (
    IA_PATH,
    TRAIN_FASTA_EXTENDED_CORRECTED_PATH,
    TRAIN_FASTA_PATH_SPLIT,
    TRAIN_NEIGHBOR_MATRIX_IDX_MAP_PATH,
    TRAIN_NEIGHBOR_MATRIX_PATH,
    TRAIN_TERMS_EXTENDED_PATH,
    VAL_FASTA_PATH_SPLIT,
)


def split_by_difficulty(
    scored_df, go_df, hard_threshold=0.8, hard_train_ratio=0.8, easy_train_ratio=0.8
):
    threshold_score = scored_df["hybrid_score"].quantile(hard_threshold)

    hard_df = scored_df.filter(pl.col("hybrid_score") >= threshold_score)
    easy_df = scored_df.filter(pl.col("hybrid_score") < threshold_score)

    train_hard = hard_df.sample(fraction=hard_train_ratio, seed=42)
    val_hard = hard_df.join(train_hard, on="EntryID", how="anti")

    train_easy = easy_df.sample(fraction=easy_train_ratio, seed=42)
    val_easy = easy_df.join(train_easy, on="EntryID", how="anti")

    train_initial = pl.concat([train_hard, train_easy])
    val_initial = pl.concat([val_hard, val_easy])

    train_terms = (
        train_initial.join(go_df, on="EntryID", how="left").select("term").unique()
    )
    val_terms = (
        val_initial.join(go_df, on="EntryID", how="left").select("term").unique()
    )

    missing_terms = val_terms.join(train_terms, on="term", how="anti")

    if missing_terms.height > 0:
        proteins_to_rescue = (
            go_df.join(missing_terms, on="term", how="inner").select("EntryID").unique()
        )

        rows_to_move = val_initial.join(proteins_to_rescue, on="EntryID", how="inner")

        train_final = pl.concat([train_initial, rows_to_move]).unique(
            subset=["EntryID"]
        )
        val_final = val_initial.join(proteins_to_rescue, on="EntryID", how="anti")

        print(
            f"Repaired Split: Moved {rows_to_move.height} proteins from Val to Train to cover missing terms."
        )
        return train_final, val_final

    return train_initial, val_initial


def get_difficult_predictions():
    nearest_neighbors = np.load(TRAIN_NEIGHBOR_MATRIX_PATH)
    with open(TRAIN_NEIGHBOR_MATRIX_IDX_MAP_PATH, "rb") as f:
        nearest_neighbors_idx = pickle.load(f)

    ids = list(nearest_neighbors_idx.keys())
    # print(nearest_neighbors_idx)
    isolation_scores = np.mean(nearest_neighbors, axis=1)

    go_df = pl.read_csv(TRAIN_TERMS_EXTENDED_PATH, separator="\t")
    ia_weights_df = pl.read_csv(IA_PATH, separator="\t")
    richness_df = (
        go_df.join(ia_weights_df, on="term", how="left")
        .group_by("EntryID")
        .agg(pl.col("weight").sum().fill_null(0.0).alias("richness_raw"))
    )

    scores_df = pl.DataFrame({"EntryID": ids, "isolation_raw": isolation_scores})

    final_df = (
        scores_df.join(richness_df, on="EntryID", how="left")
        .with_columns(pl.col("richness_raw").fill_null(0.0))
        .with_columns(
            [
                (pl.col("isolation_raw").rank() / pl.len()).alias("iso_rank"),
                (pl.col("richness_raw").rank() / pl.len()).alias("rich_rank"),
            ]
        )
        .with_columns(
            ((pl.col("iso_rank") * 0.5) + (pl.col("rich_rank") * 0.5)).alias(
                "hybrid_score"
            )
        )
        .sort("hybrid_score", descending=True)
    )

    return final_df


def check_term_coverage(train_df, val_df, go_df):
    train_terms = train_df.join(go_df, on="EntryID", how="left").select("term").unique()
    val_terms = val_df.join(go_df, on="EntryID", how="left").select("term").unique()

    unseen_terms = val_terms.join(train_terms, on="term", how="anti")

    total_val_terms = val_terms.height
    missing_count = unseen_terms.height

    print(f"Total unique terms in Validation: {total_val_terms}")
    print(f"Terms in Val but NOT in Train: {missing_count}")
    print(
        f"Percentage of impossible predictions: {(missing_count / total_val_terms) * 100:.2f}%"
    )

    return unseen_terms


def analyze_missing_term_weights(unseen_terms_df, ia_weights_df):
    missing_weights = (
        unseen_terms_df.join(ia_weights_df, on="term", how="left")
        .select(["term", "weight"])
        .with_columns(pl.col("weight").fill_null(0.0))
        .sort("weight", descending=True)
    )

    print(missing_weights.describe())

    print("\nTop 5 'Most Expensive' Missing Terms:")
    print(missing_weights.head(5))

    return missing_weights


def create_train_val_split_fastas():
    go_df = pl.read_csv(TRAIN_TERMS_EXTENDED_PATH, separator="\t")

    scored_df = get_difficult_predictions()
    train_df, val_df = split_by_difficulty(scored_df, go_df)
    print(f"{len(train_df)=}")
    print(f"{len(val_df)=}")
    train_dict = train_df.to_dict()
    val_dict = val_df.to_dict()
    train_protein_ids = set(train_dict["EntryID"])
    val_protein_ids = set(val_dict["EntryID"])
    train_records = []
    val_records = []

    for record in SeqIO.parse(TRAIN_FASTA_EXTENDED_CORRECTED_PATH, "fasta"):
        id = record.id
        if id in train_protein_ids:
            train_records.append(record)
        else:
            assert id in val_protein_ids
            val_records.append(record)

    print(f"{len(train_records)=}")
    print(f"{len(val_records)=}")

    with open(TRAIN_FASTA_PATH_SPLIT, "w") as f:
        SeqIO.write(train_records, f, "fasta")

    with open(VAL_FASTA_PATH_SPLIT, "w") as f:
        SeqIO.write(val_records, f, "fasta")


if __name__ == "__main__":
    create_train_val_split_fastas()
