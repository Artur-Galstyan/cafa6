import pathlib

MODEL_TO_DIMS: dict[str, int] = {
    "esmc_600m": 1152,
}

ESM_MODEL = "esmc_600m"

DATA_BASE_PATH = pathlib.Path(__file__).parent.parent / "data"
EMBEDDINGS_PATH = DATA_BASE_PATH / "embeddings"

TRAIN_FASTA_PATH = DATA_BASE_PATH / "train/train_sequences.fasta"
TRAIN_TERMS_PATH = DATA_BASE_PATH / "train/train_terms.tsv"
OBO_PATH = DATA_BASE_PATH / "train/go-basic.obo"
TEST_FASTA_PATH = DATA_BASE_PATH / "test/testsuperset.fasta"
IA_PATH = DATA_BASE_PATH / "IA.tsv"
IA_PATH_NO_HEAD = DATA_BASE_PATH / "IA_NO_HEAD.tsv"

TERM_TO_ASPECT_PATH = DATA_BASE_PATH / "term_to_aspect.pkl"

OBO_GRAPH_CHILDREN_PATH = DATA_BASE_PATH / "go_terms_children.npy"
OBO_GRAPH_PARENTS_PATH = DATA_BASE_PATH / "go_terms_parent.npy"

GOA_UNIPROT_ALL_GAF_PATH = DATA_BASE_PATH / "goa_uniprot_all.gaf.gz"
GOA_UNIPROT_TEST_PREDS_PATH = DATA_BASE_PATH / "goa_uniprot_all_test_preds.tsv"

TERM_TO_IDX_LOOKUP_PATH = DATA_BASE_PATH / "term_to_idx_lookup.tsv"

TEST_SUPERSET_TAXON_PATH = DATA_BASE_PATH / "test/testsuperset-taxon-list.tsv"
TEST_SUPERSET_TAXON_LOOKUP_PATH = DATA_BASE_PATH / "test/testsuperset-taxon-lookup.tsv"
PARTIAL_SUBMISSION_PATH = DATA_BASE_PATH / "partial_submission.csv"
PARTIAL_SUBMISSION_FULL_PATH = DATA_BASE_PATH / "partial_submission_full.csv"

TRAIN_NEIGHBOR_MATRIX_PATH = DATA_BASE_PATH / "neighbor_matrix_path_train.npy"
TEST_NEIGHBOR_MATRIX_PATH = DATA_BASE_PATH / "neighbor_matrix_path_test.npy"

TRAIN_NEIGHBOR_MATRIX_IDX_MAP_PATH = (
    DATA_BASE_PATH / "neighbor_matrix_path_train_idx_map.pkl"
)
TEST_NEIGHBOR_MATRIX_IDX_MAP_PATH = (
    DATA_BASE_PATH / "neighbor_matrix_path_test_idx_map.pkl"
)
