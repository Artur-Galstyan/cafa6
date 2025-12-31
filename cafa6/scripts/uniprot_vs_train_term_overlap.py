import polars as pl
from loguru import logger

from cafa6.constants import (
    PARTIAL_SUBMISSION_FULL_PATH,
    PARTIAL_SUBMISSION_PATH,
    TRAIN_TERMS_PATH,
)


def compute_uniprot_train_terms_overlap():
    partial_submission = pl.read_csv(PARTIAL_SUBMISSION_PATH)

    train_terms = pl.read_csv(TRAIN_TERMS_PATH, separator="\t")
    train_terms = train_terms.with_columns(pl.lit(1.0).alias("value"))
    train_terms = train_terms.select(["EntryID", "term", "value"])

    anti_join = train_terms.join(partial_submission, on=["EntryID", "term"], how="anti")

    logger.info(f"Adding {len(anti_join)} new terms from training set")

    partial_submission = pl.concat([partial_submission, anti_join])
    partial_submission.write_csv(PARTIAL_SUBMISSION_FULL_PATH)


if __name__ == "__main__":
    compute_uniprot_train_terms_overlap()
