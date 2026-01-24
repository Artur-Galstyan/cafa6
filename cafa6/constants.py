import pathlib

MODEL_TO_DIMS: dict[str, int] = {
    "esmc_600m": 1152,
}

ESM_MODEL = "esmc_600m"

DATA_BASE_PATH = pathlib.Path(__file__).parent.parent / "data"
WEIGHTS_BASE_PATH = pathlib.Path(__file__).parent.parent / "weights"

TRAIN_FASTA_PATH = DATA_BASE_PATH / "train/train_sequences.fasta"
TRAIN_TERMS_PATH = DATA_BASE_PATH / "train/train_terms.tsv"
OBO_PATH = DATA_BASE_PATH / "train/go-basic.obo"

TEST_FASTA_PATH = DATA_BASE_PATH / "test/testsuperset.fasta"
TEST_SUPERSET_TAXON_PATH = DATA_BASE_PATH / "test/testsuperset-taxon-list.tsv"
TEST_SUPERSET_TAXON_LOOKUP_PATH = DATA_BASE_PATH / "test/testsuperset-taxon-lookup.tsv"

IA_PATH = DATA_BASE_PATH / "IA.tsv"
IA_PATH_NO_HEAD = DATA_BASE_PATH / "IA_NO_HEAD.tsv"

GOA_UNIPROT_ALL_GAF_PATH = DATA_BASE_PATH / "goa_uniprot_all.gaf.gz"
UNIPROT_SPROT_FASTA_PATH = DATA_BASE_PATH / "uniprot_sprot.fasta"

MASTER_EMBEDDINGS_PATH = DATA_BASE_PATH / "master_esm_c_600m_embeddings.dat"
MASTER_INDEX_PATH = DATA_BASE_PATH / "master_esm_c_600m_index.json"
MASTER_TAXON_INDEX_PATH = DATA_BASE_PATH / "master_taxon_index.json"

SET1_TRAIN_TEST_GOOD_PATH = DATA_BASE_PATH / "set1_train_test_good.tsv"
SET2_TRAIN_ALL_PATH = DATA_BASE_PATH / "set2_train_all.tsv"
SET3_TRAIN_GOOD_PATH = DATA_BASE_PATH / "set3_train_good.tsv"
SET4_TRAIN_TEST_ALL_PATH = DATA_BASE_PATH / "set4_train_test_all.tsv"
SET5_TRAIN_ALL_TEST_GOOD_PATH = DATA_BASE_PATH / "set5_train_all_test_good.tsv"
SET6_MAX_SCALE_GOOD_PATH = DATA_BASE_PATH / "set6_max_scale_good.tsv"

PARTIAL_SUBMISSION_PATH = DATA_BASE_PATH / "partial_submission.csv"
PARTIAL_SUBMISSION_FULL_PATH = DATA_BASE_PATH / "partial_submission_full.csv"

MATCHA_PATH = DATA_BASE_PATH / "matcha.out"
