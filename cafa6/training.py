import os

import mlflow
from beartype.typing import Literal
from pydantic import BaseModel


class TrainConfig(BaseModel):
    batch_size: int = 128
    learning_rate: float = 0.0005
    num_epochs: int = 100
    worker_count: int = 4
    patience: int = 10

    knn: int = 6
    max_protein_seq: int = 1024

    esm_model: str = "esmc_600m"
    esm_strategy: Literal["raw", "mean"] = "mean"

    pos_weight: float = 20.0

    deepgo_se_embedding_size: int = 1024

    esm_proj_width_size: int = 1024
    esm_proj_depth: int = 4

    taxa_embedding_size: int = 512
    taxa_vocab_size: int = 8500
    taxa_mlp_depth: int = 3
    taxa_mlp_width: int = 512

    esm_model_width_size: int = 1024
    esm_model_depth: int = 3


def _setup_mlflow():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    assert tracking_uri is not None

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Cafa6")


def train(train_config: TrainConfig):
    _setup_mlflow()
