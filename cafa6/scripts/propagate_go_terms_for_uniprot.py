import polars as pl

from cafa6.constants import TRAIN_TERMS_EXTENDED_PATH, TRAIN_TERMS_PATH


def propagate():
    original = pl.read_csv(TRAIN_TERMS_PATH, separator="\t")
    extended = pl.read_csv(TRAIN_TERMS_EXTENDED_PATH, separator="\t")

    original_ids = set(original["EntryID"].unique())

    # Split extended into original vs new proteins
    extended_original = extended.filter(pl.col("EntryID").is_in(original_ids))
    extended_new = extended.filter(~pl.col("EntryID").is_in(original_ids))

    print(
        f"Original proteins: {extended_original.group_by('EntryID').len()['len'].mean():.1f} terms/protein"
    )
    print(
        f"New (test) proteins: {extended_new.group_by('EntryID').len()['len'].mean():.1f} terms/protein"
    )
    print(f"New protein count: {extended_new['EntryID'].n_unique()}")


if __name__ == "__main__":
    propagate()
