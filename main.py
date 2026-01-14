import gzip
import io
import os
import random

import matplotlib.pyplot as plt
import polars as pl
import psycopg2
from Bio import SeqIO
from tqdm import tqdm

n_go_terms = 40121

test_protein_ids = set()

for record in SeqIO.parse("data/testsuperset.fasta", "fasta"):
    test_protein_ids.add(record.id)

train_protein_ids = set()
for record in SeqIO.parse("data/train_sequences.fasta", "fasta"):
    train_protein_ids.add(record.id.split("|")[1])

print(f"{len(test_protein_ids)=}")
print(f"{len(train_protein_ids)=}")

print(f"{len(test_protein_ids | train_protein_ids)=}")
print(f"{len(test_protein_ids & train_protein_ids)=}")

random.seed(44)

train_terms = pl.read_csv("data/train_terms.tsv", separator="\t")

print(f"Total annotations: {len(train_terms)}")
print(f"Unique proteins: {train_terms['EntryID'].n_unique()}")
print(f"Unique GO terms: {train_terms['term'].n_unique()}")
print("\nAspect distribution:")
print(train_terms.group_by("aspect").len().sort("len", descending=True))

goa_uniprot_all_gaf_path = "data/goa_uniprot_all.gaf.gz"
batch_size = 100000

valid_evidence_codes = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC"}


conn = psycopg2.connect(
    host="postgres.arturgalstyan.dev",
    dbname="postgres",
    user="postgres",
    password=os.environ.get("PGPASSWORD"),
)
cur = conn.cursor()


def upload():
    cur.execute("DROP TABLE IF EXISTS goa")
    cur.execute("""
        CREATE TABLE goa (
            db VARCHAR(20),
            protein_id VARCHAR(50),
            symbol VARCHAR(255),
            qualifier VARCHAR(100),
            go_id VARCHAR(20),
            reference TEXT,
            evidence_code VARCHAR(10),
            with_from TEXT,
            aspect CHAR(1),
            name TEXT,
            synonym TEXT,
            type VARCHAR(50),
            taxon VARCHAR(100),
            date_modified VARCHAR(10),
            assigned_by VARCHAR(100),
            extension TEXT,
            isoform VARCHAR(100)
        )
    """)
    conn.commit()

    batch_buffer = io.StringIO()
    batch_count = 0

    with gzip.open(goa_uniprot_all_gaf_path, "rt") as f:
        for line in tqdm(f, desc="Loading High-Quality GOA"):
            if line.startswith("!"):
                continue

            fields = line.strip().split("\t")

            if len(fields) < 15:
                continue

            if fields[6] not in valid_evidence_codes:
                continue

            while len(fields) < 17:
                fields.append("")

            cleaned_fields = [
                val.replace("\\", "\\\\").replace("\0", "") for val in fields[:17]
            ]

            batch_buffer.write("\t".join(cleaned_fields) + "\n")
            batch_count += 1

            if batch_count >= batch_size:
                batch_buffer.seek(0)
                cur.copy_from(
                    batch_buffer,
                    "goa",
                    sep="\t",
                    null="",
                    columns=[
                        "db",
                        "protein_id",
                        "symbol",
                        "qualifier",
                        "go_id",
                        "reference",
                        "evidence_code",
                        "with_from",
                        "aspect",
                        "name",
                        "synonym",
                        "type",
                        "taxon",
                        "date_modified",
                        "assigned_by",
                        "extension",
                        "isoform",
                    ],
                )
                conn.commit()
                batch_buffer.seek(0)
                batch_buffer.truncate(0)
                batch_count = 0

    if batch_count > 0:
        batch_buffer.seek(0)
        cur.copy_from(
            batch_buffer,
            "goa",
            sep="\t",
            null="",
            columns=[
                "db",
                "protein_id",
                "symbol",
                "qualifier",
                "go_id",
                "reference",
                "evidence_code",
                "with_from",
                "aspect",
                "name",
                "synonym",
                "type",
                "taxon",
                "date_modified",
                "assigned_by",
                "extension",
                "isoform",
            ],
        )
        conn.commit()

    cur.execute("CREATE INDEX idx_protein ON goa(protein_id)")
    cur.execute("CREATE INDEX idx_go ON goa(go_id)")
    cur.execute("CREATE INDEX idx_evidence ON goa(evidence_code)")
    conn.commit()

    cur.close()
    conn.close()


def check_good_vs_back_evidence_codes():
    good_codes = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC"}

    train_good = 0
    train_bad = 0
    test_good = 0
    test_bad = 0

    with gzip.open(goa_uniprot_all_gaf_path, "rt") as f:
        for line in tqdm(f, desc="Analyzing evidence codes"):
            if line.startswith("!"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 7:
                continue

            protein_id = fields[1]
            evidence_code = fields[6]
            is_good = evidence_code in good_codes

            if protein_id in train_protein_ids:
                if is_good:
                    train_good += 1
                else:
                    train_bad += 1

            if protein_id in test_protein_ids:
                if is_good:
                    test_good += 1
                else:
                    test_bad += 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].bar(["Good", "Bad"], [train_good, train_bad], color=["green", "red"])
    axes[0].set_title(f"Train ({len(train_protein_ids)} proteins)")
    axes[0].set_ylabel("Annotation count")

    axes[1].bar(["Good", "Bad"], [test_good, test_bad], color=["green", "red"])
    axes[1].set_title(f"Test ({len(test_protein_ids)} proteins)")
    axes[1].set_ylabel("Annotation count")

    plt.tight_layout()
    plt.savefig("evidence_code_distribution.png", dpi=150)
    plt.show()

    print(
        f"Train - Good: {train_good}, Bad: {train_bad}, Ratio: {train_good / (train_good + train_bad):.2%}"
    )
    print(
        f"Test - Good: {test_good}, Bad: {test_bad}, Ratio: {test_good / (test_good + test_bad):.2%}"
    )


check_good_vs_back_evidence_codes()
