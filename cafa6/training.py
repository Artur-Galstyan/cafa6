import os

import equinox as eqx
import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import optax
import polars as pl
from beartype.typing import Literal
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from pydantic import BaseModel
from tqdm import tqdm

from cafa6.constants import (
    ESM_MODEL,
    IA_PATH,
    MODEL_TO_DIMS,
    TERM_TO_IDX_LOOKUP_PATH,
    WEIGHTS_BASE_PATH,
)
from cafa6.data import get_dataloaders
from cafa6.eval import evaluate
from cafa6.model import Model
from cafa6.utils import get_graph_edges


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

    warmup_steps: int = 500
    ratio: float = 0.15


def _setup_mlflow():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    assert tracking_uri is not None

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Cafa6")


def axiom_loss(
    terms_emb: Float[Array, "n_terms emb_size"],
    terms_radii: Float[Array, "n_terms"],
    ontology_graph: Int[Array, "n_edges 2"],
    margin: float = 0.05,
):
    child_idx = ontology_graph[:, 0]
    parent_idx = ontology_graph[:, 1]

    c_B, r_B = terms_emb[child_idx], terms_radii[child_idx]
    r_B = r_B.reshape(-1, 1)
    c_A, r_A = terms_emb[parent_idx], terms_radii[parent_idx]
    r_A = r_A.reshape(-1, 1)

    center_distance = jnp.linalg.norm(c_B - c_A, axis=1, keepdims=True)
    axiom_loss = center_distance + r_B - r_A + margin

    return jnp.mean(jax.nn.relu(axiom_loss))


def loss_fn(
    model: PyTree,
    X: tuple[Array, ...],
    labels: Int[Array, "batch_size n_terms"],
    sampled_weight: float,
    go_term_weight: Array,
    ontology_graph: Int[Array, "n_edges 2"],
    key: PRNGKeyArray,
):
    esm_emb, neighbor_prior, taxa, mask = X
    keys = jax.random.split(key, len(labels))
    logits = eqx.filter_vmap(model)(*X, sampled_weight, keys)

    smoothing = 0.05
    soft_labels = labels * (1.0 - smoothing) + 0.5 * smoothing
    per_term_loss = optax.sigmoid_binary_cross_entropy(logits, soft_labels)

    # per_term_loss = optax.sigmoid_binary_cross_entropy(logits, labels)

    weight_multiplier = 1.0 + (labels * (sampled_weight - 1.0))
    weighted_loss = per_term_loss * weight_multiplier * go_term_weight

    base_loss = jnp.mean(weighted_loss)

    axiom_loss_value = axiom_loss(
        model.deepgo.term_centers.weight,
        model.deepgo.term_radii,
        ontology_graph,
    )

    return base_loss + axiom_loss_value


@eqx.filter_jit
def step_fn(
    model: PyTree,
    X: tuple[Array, ...],
    y: Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    ontology_graph: Int[Array, "n_terms 2"],
    go_term_weights: Array,
    key: PRNGKeyArray,
):
    key, cond_key = jax.random.split(key)
    batch_size, *_ = y.shape
    sampled_weight = jax.random.uniform(
        cond_key, shape=(batch_size, 1), minval=5.0, maxval=50.0
    )
    value, grads = eqx.filter_value_and_grad(loss_fn)(
        model, X, y, sampled_weight, go_term_weights, ontology_graph, key
    )
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )

    model = eqx.apply_updates(model, updates)
    return model, opt_state, value


def train(train_config: TrainConfig = TrainConfig()):
    _setup_mlflow()
    n_terms = len(pl.read_csv(TERM_TO_IDX_LOOKUP_PATH))
    terms_to_idx_weight: dict[str, tuple[int, float]] = {}
    idx_to_terms_weight: dict[int, tuple[str, float]] = {}
    ia_df = pl.read_csv(IA_PATH, separator="\t")
    for i, row in enumerate(ia_df.iter_rows(named=True)):
        terms_to_idx_weight[row["term"]] = (i, row["weight"])
        idx_to_terms_weight[i] = (row["term"], row["weight"])

    terms_to_idx = {t: idx for t, (idx, _) in terms_to_idx_weight.items()}

    go_term_weights = np.array(
        [w for _, (_, w) in sorted(terms_to_idx_weight.items(), key=lambda x: x[1][0])]
    )

    model = Model(
        n_terms=n_terms,
        embedding_size=train_config.deepgo_se_embedding_size,
        esm_embedding_size=MODEL_TO_DIMS[ESM_MODEL],
        esm_proj_width_size=train_config.esm_proj_width_size,
        esm_proj_depth=train_config.esm_proj_depth,
        taxa_embedding_size=train_config.taxa_embedding_size,
        taxa_vocab_size=train_config.taxa_vocab_size,
        taxa_mlp_depth=train_config.taxa_mlp_depth,
        taxa_mlp_width=train_config.taxa_mlp_width,
        esm_model_width_size=train_config.esm_model_width_size,
        esm_model_depth=train_config.esm_model_depth,
        key=jax.random.key(0),
    )

    train_data_loader, validation_data_loader, n_total = get_dataloaders(
        train_config.batch_size, train_config.num_epochs, train_config.worker_count
    )
    best_model = model
    children, parents = get_graph_edges(terms_to_idx)
    ontology_graph = jnp.vstack((children, parents)).T

    total_steps = (n_total // train_config.batch_size) * train_config.num_epochs

    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=train_config.learning_rate,
        warmup_steps=train_config.warmup_steps,
        decay_steps=total_steps,
        end_value=1e-6,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=scheduler),
    )

    # optimizer = optax.MultiSteps(optimizer, every_k_schedule=16)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    key = jax.random.key(1)

    with mlflow.start_run():
        mlflow.log_params(train_config.model_dump())

        best_score = 0.0
        no_improve = 0
        epoch = 0
        epoch_loss = 0.0

        steps_per_epoch = (
            int(n_total * (1 - train_config.ratio)) // train_config.batch_size
        )
        total_steps = steps_per_epoch * train_config.num_epochs

        for step, batch in tqdm(
            enumerate(train_data_loader),
            total=(n_total * (1 - train_config.ratio)) // train_config.batch_size,
        ):
            key, subkey = jax.random.split(key)
            idx, esm_emb, neighbor_prior, taxon, mask, y = batch
            esm_emb, neighbor_prior, taxon, mask, y = (
                jnp.array(esm_emb),
                jnp.array(neighbor_prior),
                jnp.array(taxon),
                jnp.array(mask),
                jnp.array(y),
            )

            model, opt_state, loss = step_fn(
                model,
                (esm_emb, neighbor_prior, taxon, mask),
                y,
                optimizer,
                opt_state,
                ontology_graph,
                go_term_weights,
                subkey,
            )
            curr_loss = float(loss)
            mlflow.log_metric("train_loss", curr_loss, step=step)
            epoch_loss += curr_loss

            if (step + 1) % steps_per_epoch == 0:
                epoch += 1
                avg_loss = epoch_loss / steps_per_epoch
                epoch_loss = 0.0

                val_iterator = iter(validation_data_loader)
                score = evaluate(model, val_iterator)

                mlflow.log_metric("validation_score", score, step=step)
                mlflow.log_metric("epoch_avg_loss", avg_loss, step=step)

                print(
                    f"Epoch {epoch} | Val score: {score:.4f} | Avg Loss: {avg_loss:.4f}"
                )

                if score > best_score:
                    best_score = score
                    no_improve = 0
                    best_model = model
                else:
                    no_improve += 1
                    if no_improve >= train_config.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

    eqx.tree_serialise_leaves(WEIGHTS_BASE_PATH / "best-model.eqx", best_model)

    return model


if __name__ == "__main__":
    train()
