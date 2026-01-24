from pydantic import BaseModel


class TrainingSets:
    SET_1_TRAIN_TEST_GOOD: str = "set1_train_test_good"
    SET_2_TRAIN_ALL: str = "set2_train_all"
    SET_3_TRAIN_GOOD: str = "set3_train_good"
    SET_4_TRAIN_TEST_ALL: str = "set4_train_test_all"
    SET_5_TRAIN_ALL_TEST_GOOD: str = "set5_train_all_test_good"
    SET_6_MAX_SCALE_GOOD: str = "set6_max_scale_good"


class TrainConfig(BaseModel):
    n_terms: int = 40_122
    n_deepgo_models: int = 1
    n_taxons: int = 8453
    batch_size: int = 4096
    accumulation_steps: int = 4
    # batch_size: int = 16384 * 2
    taxon_embedding_size: int = 1024
    learning_rate: float = 0.0009
    num_epochs: int = 80
    worker_count: int = 2
    patience: int = 15

    ratio: float = 0.15
    training_set: str = TrainingSets.SET_6_MAX_SCALE_GOOD

    esm_model: str = "esmc_600m"

    deepgo_se_embedding_size: int = 1024

    esm_proj_width_size: int = 1024
    esm_proj_depth: int = 3

    esm_model_width_size: int = 1024
    esm_model_depth: int = 3

    warmup_steps: int = 500

    esm_embedding_size: int = 1152

    gate_mlp_width_size: int = 2048
    gate_mlp_depth: int = 3
