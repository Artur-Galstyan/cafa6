from Bio import SeqIO

from cafa6.constants import TEST_FASTA_PATH, TRAIN_FASTA_PATH

lengths = []
for path in [TRAIN_FASTA_PATH, TEST_FASTA_PATH]:
    for record in SeqIO.parse(path, "fasta"):
        lengths.append(len(record.seq))

import numpy as np

lengths = np.array(lengths)

print(f"Median: {np.median(lengths):.0f}")
print(f"Mean: {np.mean(lengths):.0f}")
print(f"<= 256: {(lengths <= 256).mean() * 100:.1f}%")
print(f"<= 512: {(lengths <= 512).mean() * 100:.1f}%")
print(f"<= 1024: {(lengths <= 1024).mean() * 100:.1f}%")
