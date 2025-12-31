import pickle
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import polars as pl
from jaxtyping import Array

from cafa6.constants import (
    TERM_TO_ASPECT_PATH,
    TERM_TO_IDX_LOOKUP_PATH,
    TRAIN_TERMS_PATH,
)
from cafa6.utils import get_go_term_weights


def build_term_to_aspect(
    terms_path: str | Path = TRAIN_TERMS_PATH,
    save_path: str | Path | None = TERM_TO_ASPECT_PATH,
    cache: bool = True,
) -> dict[str, str]:
    if save_path and Path(save_path).exists():
        with open(save_path, "rb") as f:
            return pickle.load(f)
    df = pl.read_csv(terms_path, separator="\t")
    term_aspect = df.select(["term", "aspect"]).unique()
    term_to_aspect = dict(zip(term_aspect["term"], term_aspect["aspect"]))

    if cache:
        if not save_path:
            raise ValueError("Need save path to cache")
        with open(save_path, "wb") as f:
            pickle.dump(term_to_aspect, f)

    return term_to_aspect


def build_aspect_masks(
    terms_to_idx: dict[str, int], term_to_aspect: dict[str, str]
) -> dict[str, Array]:
    n_terms = len(terms_to_idx)
    masks = {"F": jnp.zeros(n_terms), "P": jnp.zeros(n_terms), "C": jnp.zeros(n_terms)}

    for term, idx in terms_to_idx.items():
        aspect = term_to_aspect.get(term)
        if aspect in masks:
            masks[aspect] = masks[aspect].at[idx].set(1.0)

    return masks


def weighted_f1_masked(
    preds: Array, labels: Array, weights: Array, mask: Array, threshold: float
) -> Array:
    binary_preds = (preds > threshold).astype(float)

    binary_preds = binary_preds * mask
    labels = labels * mask

    tp = binary_preds * labels * weights
    p = binary_preds * weights
    t = labels * weights

    precision = tp.sum(axis=1) / (p.sum(axis=1) + 1e-8)
    recall = tp.sum(axis=1) / (t.sum(axis=1) + 1e-8)

    avg_precision = precision.mean()
    avg_recall = recall.mean()

    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-8)
    return jnp.array(f1)


def max_f1_masked(preds: Array, labels: Array, weights: Array, mask: Array) -> Array:
    thresholds = jnp.linspace(0.01, 0.99, 99)

    def _body(carry, t):
        return 1, weighted_f1_masked(preds, labels, weights, mask, t)

    _, f1s = jax.lax.scan(_body, 1, xs=thresholds)
    return jnp.max(f1s)


def _evaluate(preds: Array, labels: Array) -> Array:
    weights = jnp.array(get_go_term_weights())
    term_to_aspect = build_term_to_aspect()
    if not TERM_TO_IDX_LOOKUP_PATH.exists():
        raise FileNotFoundError(
            f"{TERM_TO_IDX_LOOKUP_PATH} not found. Run scripts/cafa6_terms_to_idx.py"
        )
    term_to_idx_weight = pl.read_csv(TERM_TO_IDX_LOOKUP_PATH)
    term_to_idx = dict(zip(term_to_idx_weight["term"], term_to_idx_weight["index"]))
    aspect_masks = build_aspect_masks(term_to_idx, term_to_aspect)

    @eqx.filter_jit
    def _eval():
        f1_mf = max_f1_masked(preds, labels, weights, aspect_masks["F"])
        f1_bp = max_f1_masked(preds, labels, weights, aspect_masks["P"])
        f1_cc = max_f1_masked(preds, labels, weights, aspect_masks["C"])

        return (f1_mf + f1_bp + f1_cc) / 3

    return _eval()


def evaluate(model, val_data_loader, condition_val: Array = jnp.array(30.0)):
    all_preds = []
    all_labels = []

    inference_model = eqx.nn.inference_mode(model)

    cond_array = jnp.array([condition_val])

    @eqx.filter_jit
    def _get_preds(_model, _e, _n, _t, _m, _c):
        logits = eqx.filter_vmap(_model, in_axes=(0, 0, 0, 0, None, None))(
            _e, _n, _t, _m, _c, None
        )
        preds = jax.nn.sigmoid(logits)
        return preds

    for batch in val_data_loader:
        idx, esm_emb, neighbor_prior, taxon, mask, y = batch
        esm_emb, neighbor_prior, taxon, mask, y = (
            jnp.array(esm_emb),
            jnp.array(neighbor_prior),
            jnp.array(taxon),
            jnp.array(mask),
            jnp.array(y),
        )

        preds = _get_preds(
            inference_model, esm_emb, neighbor_prior, taxon, mask, cond_array
        )

        all_preds.append(preds)
        all_labels.append(y)

    all_preds = jnp.concatenate(all_preds, axis=0)
    all_labels = jnp.concatenate(all_labels, axis=0)

    return _evaluate(
        all_preds,
        all_labels,
    )
