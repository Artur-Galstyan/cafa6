import os

from beartype.typing import Literal

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import json

import coolname
import equinox as eqx
import jax
import jax.numpy as jnp
import mlflow
import optax
import polars as pl
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from tqdm import tqdm

from cafa6.config import TrainConfig
from cafa6.constants import (
    DATA_BASE_PATH,
    IA_PATH,
    WEIGHTS_BASE_PATH,
)
from cafa6.data import (
    create_train_loader,
    create_val_loader,
    get_train_transforms,
    get_train_val_datasources,
    get_val_transforms,
)
from cafa6.eval import evaluate
from cafa6.model import Model
from cafa6.utils import get_graph_edges


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


def hierarchy_consistency_loss(probs, ontology_graph):
    child_idx = ontology_graph[:, 0]
    parent_idx = ontology_graph[:, 1]

    child_probs = probs[:, child_idx]
    parent_probs = probs[:, parent_idx]

    violation = jax.nn.relu(child_probs - parent_probs)
    return violation.mean()


def focal_loss(
    logits: Float[Array, "batch n_terms"],
    labels: Float[Array, "batch n_terms"],
    gamma: float = 2.0,
):
    probs = jax.nn.sigmoid(logits)
    probs = jnp.clip(probs, 1e-6, 1 - 1e-6)
    p_t = labels * probs + (1 - labels) * (1 - probs)
    focal_weight = (1 - p_t) ** gamma
    bce = optax.sigmoid_binary_cross_entropy(logits, labels)
    return focal_weight * bce


def loss_fn(
    model: PyTree,
    X: tuple[Array, ...],
    labels: Int[Array, "batch_size n_terms"],
    sampled_weight: float,
    go_term_weight: Array,
    ontology_graph: Int[Array, "n_edges 2"],
    key: PRNGKeyArray,
):
    keys = jax.random.split(key, len(labels))
    logits = eqx.filter_vmap(model)(*X, sampled_weight, keys)
    probs = jax.nn.sigmoid(logits)

    smoothing = 0.05
    soft_labels = labels * (1.0 - smoothing) + 0.5 * smoothing

    per_term_loss = focal_loss(logits, soft_labels, gamma=2.0)

    weight_multiplier = 1.0 + (labels * (sampled_weight - 1.0))
    weighted_loss = per_term_loss * weight_multiplier * go_term_weight

    base_loss = jnp.mean(weighted_loss)

    axiom_loss_value = axiom_loss(
        model.deepgo.term_centers.weight,
        model.deepgo.term_radii,
        ontology_graph,
    )

    hierarchy_loss_value = hierarchy_consistency_loss(probs, ontology_graph)

    return base_loss + axiom_loss_value + 0.3 * hierarchy_loss_value


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


def train(train_config: TrainConfig = TrainConfig(), model_name: str | None = None):
    _setup_mlflow()
    if model_name is None:
        model_name = coolname.generate_slug(3)
    assert model_name is not None
    print("Training model with name:")
    print(model_name)

    terms_to_idx: dict[str, int] = {}
    ia_df = pl.read_csv(IA_PATH, separator="\t")
    for i, row in enumerate(ia_df.iter_rows(named=True)):
        terms_to_idx[row["term"]] = i

    go_term_weights = ia_df["weight"].to_numpy()

    model = Model(
        train_config,
        key=jax.random.key(0),
    )

    labels_path = DATA_BASE_PATH / f"{train_config.training_set}.tsv"
    train_ds, val_ds, n_train = get_train_val_datasources(
        labels_path=labels_path,
        ratio=train_config.ratio,
    )

    train_transforms = get_train_transforms(
        dataset_type=train_config.used_dataset, batch_size=train_config.batch_size
    )
    val_transforms = get_val_transforms(
        dataset_type=train_config.used_dataset, batch_size=train_config.batch_size
    )

    train_data_loader = create_train_loader(
        train_ds,
        train_transforms,
        train_config.batch_size,
        train_config.num_epochs,
        train_config.worker_count,
    )

    best_model = model
    children, parents = get_graph_edges(terms_to_idx)
    ontology_graph = jnp.vstack((children, parents)).T

    total_steps = (n_train // train_config.batch_size) * train_config.num_epochs

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

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    key = jax.random.key(1)

    with mlflow.start_run(description=model_name, run_name=model_name):
        mlflow.log_params(train_config.model_dump())

        best_score = 0.0
        no_improve = 0
        epoch = 0
        epoch_loss = 0.0

        steps_per_epoch = n_train // train_config.batch_size

        for step, batch in tqdm(
            enumerate(train_data_loader),
            total=total_steps,
        ):
            key, subkey = jax.random.split(key)
            idx, *model_inputs, y = batch
            model_inputs = tuple(jnp.array(m) for m in model_inputs)
            y = jnp.array(y)

            model, opt_state, loss = step_fn(
                model,
                model_inputs,
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

            if (step + 1) % (steps_per_epoch * 5) == 0:
                epoch += 1
                avg_loss = epoch_loss / steps_per_epoch
                epoch_loss = 0.0

                val_loader = create_val_loader(val_ds, val_transforms)
                score = evaluate(model, val_loader)

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

    eqx.tree_serialise_leaves(WEIGHTS_BASE_PATH / f"{model_name}.eqx", best_model)

    with open(WEIGHTS_BASE_PATH / f"{model_name}-config.json", "w") as f:
        json.dump(train_config.model_dump(), f)

    return model


def main(gpu: Literal[0, 1] = 0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    train()
