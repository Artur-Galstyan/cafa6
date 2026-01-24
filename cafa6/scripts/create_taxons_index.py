import json
import os
import re

import psycopg2
from Bio import SeqIO
from dotenv import load_dotenv
from tqdm import tqdm

from cafa6.constants import MASTER_INDEX_PATH, MASTER_TAXON_INDEX_PATH, TEST_FASTA_PATH

load_dotenv()

with open(MASTER_INDEX_PATH, "rb") as f:
    master_prots = json.load(f)
print(len(master_prots.keys()))


conn = psycopg2.connect(
    host="postgres.arturgalstyan.dev",
    dbname="postgres",
    user="postgres",
    password=os.environ.get("PGPASSWORD"),
)


cur = conn.cursor()

taxon_indices = {}

# Fetch all data in a single query using WHERE IN
protein_ids = list(master_prots.keys())
print(f"Fetching taxon data for {len(protein_ids)} proteins...")

# Process in batches to avoid query size limits
batch_size = 10000
for i in tqdm(range(0, len(protein_ids), batch_size)):
    batch = protein_ids[i : i + batch_size]

    # Use parameterized query to prevent SQL injection
    placeholders = ",".join(["%s"] * len(batch))
    query = f"SELECT DISTINCT protein_id, taxon FROM goa WHERE protein_id IN ({placeholders})"

    cur.execute(query, batch)
    results = cur.fetchall()

    for protein_id, taxon in results:
        taxon_number_match = re.search(r"\d+", taxon)
        if taxon_number_match:
            taxon_number = taxon_number_match.group(0)
            taxon_indices[protein_id] = taxon_number

print(f"Found taxon data for {len(taxon_indices)} proteins")

counter = 0
for record in SeqIO.parse(TEST_FASTA_PATH, "fasta"):
    protein_id = record.id
    taxon = record.description.split(" ")[-1]
    if taxon not in taxon_indices:
        taxon_indices[protein_id] = taxon
        counter += 1

print(f"Found additional taxon data: {counter}")

with open(MASTER_TAXON_INDEX_PATH, "w") as f:
    json.dump(taxon_indices, f)
