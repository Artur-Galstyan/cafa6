import gc
import gzip
from pathlib import Path

import polars as pl
from tqdm import tqdm

from cafa6.constants import (
    DATA_BASE_PATH,
    GOA_UNIPROT_ALL_GAF_PATH,
    GOA_UNIPROT_TEST_PREDS_PATH,
    TERM_TO_IDX_LOOKUP_PATH,
)


def extract_go_terms_from_uniprot(
    goa_uniprot_all_gaf_path: str | Path = GOA_UNIPROT_ALL_GAF_PATH,
    goa_uniprot_test_preds_path: str | Path | None = GOA_UNIPROT_TEST_PREDS_PATH,
    cache: bool = True,
) -> pl.DataFrame:
    """
    **This function has side-effects**

    This function iterates over the goa_uniprot_all.gaf file and extracts
    all the go terms in can find for each protein in there.

    This will be later used as a lookup library. Basically, when making our submission,
    we shouldn't trust the GO term predictions from our model, because we already have
    experimentally verified GO terms from uniprot. Thus, we will treat this as the
    ground truth.

    This function returns the result as a dataframe but also caches it. If a file in the given
    preds path already exists, it will load it instead.


    Returns:
        the polars dataframe
    """

    if goa_uniprot_test_preds_path and Path(goa_uniprot_test_preds_path).exists():
        return pl.read_csv(goa_uniprot_test_preds_path)

    data_buffer = []
    counter = 0
    chunk_counter = 0

    go_terms_df = pl.read_csv(TERM_TO_IDX_LOOKUP_PATH)
    go_terms = go_terms_df.get_column("term").unique()
    go_terms = set(go_terms)

    with gzip.open(goa_uniprot_all_gaf_path, "rt") as f:
        for line in tqdm(f, mininterval=1.0, desc="Scanning GAF"):
            if line.startswith("!"):
                continue

            splits = line.split("\t")

            if len(splits) < 5:
                continue

            counter += 1
            protein_id = splits[1]
            qualifier = splits[3]
            go_term = splits[4]

            if "NOT" in qualifier:
                continue

            if go_term not in go_terms:
                continue

            data_buffer.append({"EntryID": protein_id, "term": go_term, "value": 1.0})
            if counter % 50_000_000 == 0:
                df = pl.DataFrame(data_buffer)
                df.write_csv(Path(DATA_BASE_PATH) / f"{chunk_counter}.csv")
                chunk_counter += 1
                del df
                data_buffer = []
                gc.collect()

    if data_buffer:
        df = pl.DataFrame(data_buffer)
        df.write_csv(Path(DATA_BASE_PATH) / f"{chunk_counter}.csv")
        chunk_counter += 1
    chunks = [
        pl.read_csv(Path(DATA_BASE_PATH) / f"{i}.csv") for i in range(chunk_counter)
    ]
    df = pl.concat(chunks)
    df = df.unique()

    if cache:
        if not goa_uniprot_test_preds_path:
            raise ValueError("Path expected if supposed to cache!")
        df.write_csv(
            goa_uniprot_test_preds_path
        )  # you gotta have enough RAM for this (which I don't so it crased xd)
    return df


if __name__ == "__main__":
    extract_go_terms_from_uniprot()
