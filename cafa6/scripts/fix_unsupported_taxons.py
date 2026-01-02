import re

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cafa6.constants import (
    TEST_FASTA_PATH,
    TRAIN_FASTA_EXTENDED_CORRECTED_PATH,
    TRAIN_FASTA_EXTENDED_PATH,
)


def fix_unsupported_taxons():
    new_records = []

    test_set = {}

    counter = 0

    for record in SeqIO.parse(TEST_FASTA_PATH, "fasta"):
        protein_id = record.id.split(" ")[0]
        taxon = int(record.description.split(" ")[-1])
        test_set[protein_id] = taxon

    for record in SeqIO.parse(TRAIN_FASTA_EXTENDED_PATH, "fasta"):
        protein_id = record.id
        description = record.description
        taxon = int(re.search(r"(?<=OX=)\d+", description).group(0))  # ty:ignore[possibly-missing-attribute]

        if protein_id in test_set:
            if taxon != test_set[protein_id]:
                counter += 1
            taxon = test_set[protein_id]

        else:
            print(f"{protein_id=} not found in test set")

        new_records.append(
            SeqRecord(Seq(record.seq), id=protein_id, description=f"OX={taxon}")
        )

    print(f"Found {counter} proteins with different taxons than expected.")
    print(f"New training length is {len(new_records)}")
    with open(TRAIN_FASTA_EXTENDED_CORRECTED_PATH, "w") as f_out:
        SeqIO.write(new_records, f_out, "fasta")


if __name__ == "__main__":
    fix_unsupported_taxons()
