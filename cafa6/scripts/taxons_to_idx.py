from pathlib import Path

import polars as pl

from cafa6.constants import TEST_SUPERSET_TAXON_LOOKUP_PATH, TEST_SUPERSET_TAXON_PATH


def taxons_to_idx(
    taxons_path: str | Path = TEST_SUPERSET_TAXON_PATH,
    test_superset_taxon_lookup_path: str
    | Path
    | None = TEST_SUPERSET_TAXON_LOOKUP_PATH,
    cache: bool = True,
) -> pl.DataFrame:
    """
    Generates a dataframe which maps a taxon id (from a fasta file)
    to a unique integer. This can be used to construct one-hot encoded
    vectors for taxons.

    E.g. like this
    ```python
        filtered_frame = taxon_to_idx_df.filter(pl.col("ID").is_in([35659, 45621]))
        # these are the idx to make one-hot encoded vectors, i.e. set those to 1
    ```
    """

    if (
        test_superset_taxon_lookup_path
        and Path(test_superset_taxon_lookup_path).exists()
    ):
        return pl.read_csv(test_superset_taxon_lookup_path)

    testset_taxons = pl.read_csv(taxons_path, separator="\t")
    unique_ids = testset_taxons.select(pl.col("ID")).unique()
    taxon_to_idx = []
    for i, idx in enumerate(unique_ids.iter_rows(named=True)):
        taxon_to_idx.append({"ID": idx["ID"], "idx": i})

    taxon_to_idx_df = pl.DataFrame(taxon_to_idx)
    if cache:
        if not test_superset_taxon_lookup_path:
            raise ValueError("Need location for cache")
        taxon_to_idx_df.write_csv(test_superset_taxon_lookup_path)
    return taxon_to_idx_df


if __name__ == "__main__":
    taxons_to_idx()
