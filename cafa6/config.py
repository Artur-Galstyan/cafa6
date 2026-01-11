from beartype.typing import Literal
from pydantic import BaseModel


class TrainConfig(BaseModel):
    n_terms: int = 40_122
    batch_size: int = 256
    learning_rate: float = 0.0008
    num_epochs: int = 50
    worker_count: int = 4
    patience: int = 10

    knn: int = 6
    max_protein_seq: int = 1024

    esm_model: str = "esmc_600m"
    esm_strategy: Literal["raw", "mean"] = "mean"

    deepgo_se_embedding_size: int = 1024

    esm_proj_width_size: int = 1024
    esm_proj_depth: int = 2

    taxa_embedding_size: int = 1024
    taxa_vocab_size: int = 8500
    taxa_mlp_depth: int = 2
    taxa_mlp_width: int = 1024

    esm_model_width_size: int = 1024
    esm_model_depth: int = 2

    warmup_steps: int = 500
    ratio: float = 0.2

    text_embedding_size: int = 1024  # voyage embs, don't change!
    text_embedding_hidden_size: int = 512
    text_embedding_mlp_width: int = 1024
    text_embedding_mlp_depth: int = 2

    esm_embedding_size: int = 1152

    tea_mlp_depth: int = 2
    tea_mlp_width_size: int = 1024

    gate_mlp_width_size: int = 2048
    gate_mlp_depth: int = 3
