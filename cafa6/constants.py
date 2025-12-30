import pathlib

MODEL_TO_DIMS: dict[str, int] = {
    "esmc_600m": 1152,
}

DATA_BASE_PATH = pathlib.Path(__file__).parent.parent / "data"

TRAIN_FASTA_PATH = DATA_BASE_PATH / "train/train_sequences.fasta"
TRAIN_TERMS_PATH = DATA_BASE_PATH / "train/train_terms.tsv"
OBO_PATH = DATA_BASE_PATH / "train/go-basic.obo"
TEST_FASTA_PATH = DATA_BASE_PATH / "test/testsuperset.fasta"
IA_PATH = DATA_BASE_PATH / "IA.tsv"
IA_PATH_NO_HEAD = DATA_BASE_PATH / "IA_NO_HEAD.tsv"


OBO_GRAPH_CHILDREN_PATH = DATA_BASE_PATH / "go_terms_children.npy"
OBO_GRAPH_PARENTS_PATH = DATA_BASE_PATH / "go_terms_parent.npy"

GOA_UNIPROT_ALL_GAF_PATH = DATA_BASE_PATH / "goa_uniprot_all.gaf.gz"
GOA_UNIPROT_TEST_PREDS_PATH = DATA_BASE_PATH / "goa_uniprot_all_test_preds.tsv"

TERM_TO_IDX_LOOKUP_PATH = DATA_BASE_PATH / "term_to_idx_lookup.tsv"
