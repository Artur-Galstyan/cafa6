from pathlib import Path

import polars as pl
from fire import Fire

from cafa6.constants import IA_PATH, TERM_TO_IDX_LOOKUP_PATH


def generate_term_to_idx_lookup(
    ia_path: str | Path = IA_PATH,
    save_path: str | Path | None = TERM_TO_IDX_LOOKUP_PATH,
    cache: bool = True,
) -> pl.DataFrame:
    """
    This function generates a DataFrame which serves as a lookup table to map
    an index to a term and vice versa. If a save path is provided, it will load the DF from
    there (and save it there later if cached is true).
    Returns:
        polars dataframe
    """
    if save_path and Path(save_path).exists():
        return pl.read_csv(save_path)

    ia_df = pl.read_csv(ia_path, separator="\t")
    terms_to_idx_weight = []
    for i, row in enumerate(ia_df.iter_rows(named=True)):
        terms_to_idx_weight.append(
            {"term": row["term"], "index": i, "weight": row["weight"]}
        )
    df = pl.DataFrame(terms_to_idx_weight)
    if cache:
        if not save_path:
            raise ValueError("Need save path to cache")
        df.write_csv(save_path)

    return df


if __name__ == "__main__":
    Fire(generate_term_to_idx_lookup)
