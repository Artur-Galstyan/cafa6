import gzip
import time
from pathlib import Path

import numpy as np
import polars as pl
import requests
import voyageai
from beartype.typing import Literal
from Bio import SeqIO
from tqdm import tqdm

from cafa6.constants import (
    DATA_BASE_PATH,
    GOA_UNIPROT_ALL_GAF_PATH,
    TEXT_EMBEDDINGS_PATH_TRAIN,
    TRAIN_FASTA_EXTENDED_CORRECTED_PATH,
    TRAIN_PROTEIN_RAW_DESCRIPTIONS,
    TRAIN_PROTEIN_RAW_DESCRIPTIONS_RICH,
)


def get_protein_descriptions(target: Literal["train", "test"]):
    protein_ids = set()

    for record in SeqIO.parse(TRAIN_FASTA_EXTENDED_CORRECTED_PATH, "fasta"):
        protein_id = record.id.split(" ")[0]
        protein_ids.add(protein_id)

    csv_buffer = []
    chunk_counter = 0

    gaf_chunks_dir = Path(DATA_BASE_PATH) / "gaf_chunks"
    gaf_chunks_dir.mkdir(exist_ok=True, parents=True)

    seen_prots = set()

    with gzip.open(GOA_UNIPROT_ALL_GAF_PATH, "rt") as f:
        for line in tqdm(f, mininterval=1.0, desc="Scanning GAF"):
            if line.startswith("!"):
                continue

            splits = line.split("\t")
            if len(splits) < 15:
                continue

            protein_id = splits[1]
            if protein_id not in protein_ids or protein_id in seen_prots:
                continue

            qualifier = splits[3]
            if "NOT" in qualifier:
                continue

            description = splits[9]
            csv_buffer.append({"EntryID": protein_id, "description": description})
            seen_prots.add(protein_id)
            if len(csv_buffer) >= 1_000_000:
                df = pl.DataFrame(csv_buffer)
                df.write_csv(gaf_chunks_dir / f"{chunk_counter}-train-descriptions.csv")
                chunk_counter += 1
                csv_buffer = []

    if csv_buffer:
        df = pl.DataFrame(csv_buffer)
        df.write_csv(gaf_chunks_dir / f"{chunk_counter}-train-descriptions.csv")
        chunk_counter += 1

    chunks = [pl.read_csv(p) for p in gaf_chunks_dir.glob("*-train-descriptions.csv")]

    if not chunks:
        raise ValueError("No matching data found in GAF file!")

    final_df = pl.concat(chunks)
    final_df = final_df.unique()

    final_df.write_csv(TRAIN_PROTEIN_RAW_DESCRIPTIONS)

    return final_df


def generate_embeddings():
    vo = voyageai.Client()

    voyage_model = "voyage-3-large"
    df = pl.read_csv(TRAIN_PROTEIN_RAW_DESCRIPTIONS_RICH)
    batch_size = 400
    for frame in tqdm(df.iter_slices(n_rows=batch_size)):
        text_batch = frame["description"].to_list()
        protein_ids = frame["EntryID"].to_list()

        embeddings = vo.embed(text_batch, model=voyage_model).embeddings
        for protein_id, embedding in zip(protein_ids, embeddings):
            embedding_array = np.array(embedding)
            np.save(TEXT_EMBEDDINGS_PATH_TRAIN / f"{protein_id}.npy", embedding_array)


def get_uniprot_text(protein_id: str) -> str | None:
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}"
    headers = {"Accept": "application/json"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None

        data = response.json()
        text_parts = []

        # 1. Function description (most valuable)
        for comment in data.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                for text in comment.get("texts", []):
                    if "value" in text:
                        text_parts.append(text["value"])

        # 2. Subcellular location
        for comment in data.get("comments", []):
            if comment.get("commentType") == "SUBCELLULAR LOCATION":
                for loc in comment.get("subcellularLocations", []):
                    if "location" in loc and "value" in loc["location"]:
                        text_parts.append(loc["location"]["value"])

        # 3. Keywords (always include as fallback)
        keywords = [kw["name"] for kw in data.get("keywords", [])]
        if keywords:
            text_parts.append("Keywords: " + ", ".join(keywords))

        # 4. Protein name as last resort
        if not text_parts:
            protein_desc = data.get("proteinDescription", {})
            rec_name = protein_desc.get("recommendedName", {})
            if "fullName" in rec_name:
                text_parts.append(rec_name["fullName"].get("value", ""))

        return " ".join(text_parts) if text_parts else None

    except Exception as e:
        print(f"Error fetching {protein_id}: {e}")
        return None


def fetch_batch(protein_ids: list[str]) -> dict:
    """Fetch up to 500 proteins at once"""
    ids_str = ",".join(protein_ids)
    url = "https://rest.uniprot.org/uniprotkb/accessions"

    params = {
        "accessions": ids_str,
        "fields": "accession,cc_function,cc_subcellular_location,keyword",
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return {}

    results = {}
    for entry in response.json().get("results", []):
        pid = entry.get("primaryAccession")
        text_parts = []

        for comment in entry.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                for t in comment.get("texts", []):
                    text_parts.append(t.get("value", ""))
            elif comment.get("commentType") == "SUBCELLULAR LOCATION":
                for loc in comment.get("subcellularLocations", []):
                    if "location" in loc:
                        text_parts.append(loc["location"].get("value", ""))

        keywords = [kw["name"] for kw in entry.get("keywords", [])]
        if keywords:
            text_parts.append("Keywords: " + ", ".join(keywords))

        if text_parts:
            results[pid] = " ".join(text_parts)

    return results


def generate_embeddings_rich():
    protein_ids = set()

    for record in SeqIO.parse(TRAIN_FASTA_EXTENDED_CORRECTED_PATH, "fasta"):
        protein_id = record.id.split(" ")[0]
        protein_ids.add(protein_id)

    buffer = []
    protein_ids = list(protein_ids)
    batch_size = 500
    for i in tqdm(range(0, len(protein_ids), batch_size)):
        batch = protein_ids[i : i + batch_size]
        results = fetch_batch(batch)
        for p, t in results.items():
            buffer.append({"EntryID": p, "description": t})
        time.sleep(0.5)

    df = pl.DataFrame(buffer)
    df.write_csv(TRAIN_PROTEIN_RAW_DESCRIPTIONS_RICH)


if __name__ == "__main__":
    # get_protein_descriptions("train")
    # generate_embeddings_rich()
    generate_embeddings()
