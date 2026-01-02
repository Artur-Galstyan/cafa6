import gc
import gzip
import re
import sys
from pathlib import Path

import polars as pl
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from loguru import logger
from tqdm import tqdm

from cafa6.constants import (
    DATA_BASE_PATH,
    GOA_UNIPROT_ALL_GAF_PATH,
    GOA_UNIPROT_TEST_PREDS_PATH,
    PARTIAL_SUBMISSION_PATH,
    TERM_TO_IDX_LOOKUP_PATH,
    TEST_FASTA_PATH,
    TRAIN_FASTA_EXTENDED_PATH,
    TRAIN_FASTA_PATH,
    TRAIN_FASTA_UNIPROT_PATH,
    TRAIN_TERMS_EXTENDED_PATH,
    TRAIN_TERMS_PATH,
    TRAIN_TERMS_UNIPROT_PATH,
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
                df.write_csv(
                    Path(DATA_BASE_PATH) / "gaf_chunks" / f"{chunk_counter}.csv"
                )
                chunk_counter += 1
                del df
                data_buffer = []
                gc.collect()

    if data_buffer:
        df = pl.DataFrame(data_buffer)
        df.write_csv(Path(DATA_BASE_PATH) / f"{chunk_counter}.csv")
        chunk_counter += 1
    chunks = [
        pl.read_csv(Path(DATA_BASE_PATH) / "gaf_chunks" / f"{i}.csv")
        for i in range(chunk_counter)
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


def get_existing_go_terms_for_testset(
    test_fasta_path: str | Path = TEST_FASTA_PATH,
    partial_submission_path: str | Path = PARTIAL_SUBMISSION_PATH,
):
    protein_ids_set = set()
    for record in tqdm(SeqIO.parse(test_fasta_path, "fasta")):
        protein_id = record.id
        protein_ids_set.add(protein_id)

    chunk_paths = []
    total_chunks = 23  # you get this number if you ran the above function
    for chunk in range(total_chunks):
        chunk_paths.append(DATA_BASE_PATH / "gaf_chunks" / f"{chunk}.csv")

    lazy_df = pl.scan_csv(chunk_paths)
    protein_ids_list = list(protein_ids_set)
    filtered_lazy = lazy_df.filter(pl.col("EntryID").is_in(protein_ids_list))

    filtered_lazy.sink_csv(partial_submission_path)


def create_extended_train_terms_from_uniprot(
    goa_uniprot_all_gaf_path: str | Path = GOA_UNIPROT_ALL_GAF_PATH,
    train_terms_uniprot_path: str | Path = TRAIN_TERMS_UNIPROT_PATH,
    extended_fasta_path: str | Path = TRAIN_FASTA_UNIPROT_PATH,
    cache: bool = True,
):
    if train_terms_uniprot_path and Path(train_terms_uniprot_path).exists():
        return pl.read_csv(train_terms_uniprot_path)

    test_protein_sequences = {}
    for record in tqdm(SeqIO.parse(TEST_FASTA_PATH, "fasta"), desc="Loading Test Seqs"):
        test_protein_sequences[record.id] = str(record.seq)

    csv_buffer = []
    found_proteins_map = {}
    chunk_counter = 0

    go_terms_df = pl.read_csv(TERM_TO_IDX_LOOKUP_PATH)
    valid_go_terms = set(go_terms_df["term"])

    gaf_chunks_dir = Path(DATA_BASE_PATH) / "gaf_chunks"
    gaf_chunks_dir.mkdir(exist_ok=True, parents=True)

    with gzip.open(goa_uniprot_all_gaf_path, "rt") as f:
        for line in tqdm(f, mininterval=1.0, desc="Scanning GAF"):
            if line.startswith("!"):
                continue

            splits = line.split("\t")
            if len(splits) < 15:
                continue

            protein_id = splits[1]
            if protein_id not in test_protein_sequences:
                continue

            qualifier = splits[3]
            if "NOT" in qualifier:
                continue

            go_term = splits[4]
            if go_term not in valid_go_terms:
                continue

            aspect = splits[8]
            taxon_raw = splits[12]
            taxon = taxon_raw.split(":")[1] if ":" in taxon_raw else taxon_raw

            csv_buffer.append(
                {"EntryID": protein_id, "term": go_term, "aspect": aspect}
            )

            found_proteins_map[protein_id] = taxon

            if len(csv_buffer) >= 1_000_000:
                df = pl.DataFrame(csv_buffer)
                df.write_csv(gaf_chunks_dir / f"{chunk_counter}-train-terms.csv")
                chunk_counter += 1
                csv_buffer = []

    if csv_buffer:
        df = pl.DataFrame(csv_buffer)
        df.write_csv(gaf_chunks_dir / f"{chunk_counter}-train-terms.csv")
        chunk_counter += 1

    print(f"Creating extended FASTA with {len(found_proteins_map)} new proteins...")

    with open(extended_fasta_path, "w") as f_out:
        original_records = SeqIO.parse(TRAIN_FASTA_PATH, "fasta")
        SeqIO.write(original_records, f_out, "fasta")

        new_records = []
        for protein_id, taxon in found_proteins_map.items():
            new_records.append(
                SeqRecord(
                    Seq(test_protein_sequences[protein_id]),
                    id=protein_id,
                    description=f"OX={taxon}",
                )
            )
        SeqIO.write(new_records, f_out, "fasta")

    chunks = [pl.read_csv(p) for p in gaf_chunks_dir.glob("*-train-terms.csv")]

    if not chunks:
        raise ValueError("No matching data found in GAF file!")

    final_df = pl.concat(chunks)
    final_df = final_df.unique()

    if cache:
        final_df.write_csv(train_terms_uniprot_path)

    return final_df


def combine_uniprot_and_train_sets():
    combined_records = []
    protein_ids_set = set()
    for record in SeqIO.parse(TRAIN_FASTA_UNIPROT_PATH, "fasta"):
        protein_id = record.id if "|" not in record.id else record.id.split("|")[1]
        if protein_id in protein_ids_set:
            continue
        protein_ids_set.add(protein_id)
        description = record.description
        match = re.search(r"(?<=OX=)\d+", description)
        taxon = int(match.group(0)) if match else 9606

        combined_records.append(
            SeqRecord(
                Seq(record.seq),
                id=protein_id,
                description=f"OX={taxon}",
            )
        )
    with open(TRAIN_FASTA_EXTENDED_PATH, "w") as f_out:
        SeqIO.write(combined_records, f_out, "fasta")

    train_terms = pl.read_csv(TRAIN_TERMS_PATH, separator="\t")
    train_terms_extended = pl.read_csv(TRAIN_TERMS_UNIPROT_PATH)

    train_terms_diff = train_terms_extended.join(
        train_terms, on=["EntryID", "term"], how="anti"
    )

    combined_df = pl.concat([train_terms, train_terms_diff])
    combined_df.write_csv(TRAIN_TERMS_EXTENDED_PATH, separator="\t")


if __name__ == "__main__":
    # create_extended_train_terms_from_uniprot()
    # extract_go_terms_from_uniprot()
    # get_existing_go_terms_for_testset()
    combine_uniprot_and_train_sets()
