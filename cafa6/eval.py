import pickle
from pathlib import Path

import polars as pl
from cafaeval.evaluation import cafa_score_dfs

from cafa6.constants import (
    IA_PATH,
    IA_PATH_NO_HEAD,
    OBO_PATH,
    SET6_MAX_SCALE_GOOD_PATH,
    TRAIN_TERMS_PATH,
)
from cafa6.scripts.create_submission import create_submission


def build_term_to_aspect(
    terms_path: str | Path = TRAIN_TERMS_PATH,
    save_path: str | Path | None = None,
    cache: bool = False,
) -> dict[str, str]:
    if save_path and Path(save_path).exists():
        with open(save_path, "rb") as f:
            return pickle.load(f)
    df = pl.read_csv(terms_path, separator="\t")
    term_aspect = df.select(["term", "aspect"]).unique()
    term_to_aspect = dict(zip(term_aspect["term"], term_aspect["aspect"]))

    if cache and save_path:
        with open(save_path, "wb") as f:
            pickle.dump(term_to_aspect, f)

    return term_to_aspect


def get_terms_to_idx() -> dict[str, int]:
    terms_to_idx = {}
    ia_df = pl.read_csv(IA_PATH, separator="\t")
    for i, row in enumerate(ia_df.iter_rows(named=True)):
        terms_to_idx[row["term"]] = i
    return terms_to_idx


def evaluate(
    model,
    val_data_loader,
    ground_truth_path: str | Path = SET6_MAX_SCALE_GOOD_PATH,
):
    submission = create_submission(
        model=model,
        data_loader=val_data_loader,
        worker_count=0,
        preds_per_term=50,
        include_partial=False,
        is_val=True,
    )
    submission = submission.rename(
        {"EntryID": "protein_id", "term": "term_id", "value": "score"}
    )

    val_protein_ids = submission["protein_id"].unique().to_list()

    gt_df = pl.read_csv(ground_truth_path, separator="\t")
    gt_df = gt_df.select(
        [
            pl.col("id").alias("protein_id"),
            pl.col("go_term").alias("term_id"),
        ]
    )
    gt_df = gt_df.with_columns(pl.col("term_id").str.split(",")).explode("term_id")
    gt_df = gt_df.filter(pl.col("protein_id").is_in(val_protein_ids))

    score = cafa_score_dfs(
        OBO_PATH,
        submission,
        gt_df,
        IA_PATH_NO_HEAD,
    )

    return score
