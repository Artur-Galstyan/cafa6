import json
from pathlib import Path
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from tqdm import tqdm

from cafa6.constants import (
    PARTIAL_SUBMISSION_FULL_PATH,
    TERM_TO_IDX_LOOKUP_PATH,
    WEIGHTS_BASE_PATH,
)
from cafa6.data import get_test_loader
from cafa6.model import Model


@eqx.filter_jit
def _get_top_preds(
    model, embeddings_esm, neighbor_priors, taxons, masks, condition, preds_per_term
):
    preds = eqx.filter_vmap(model, in_axes=(0, 0, 0, 0, None, None))(
        embeddings_esm, neighbor_priors, taxons, masks, condition, None
    )
    preds = jax.nn.sigmoid(preds)
    top_values, top_indices = jax.lax.top_k(preds, preds_per_term)
    return top_indices, top_values


def create_submission(
    model_path: str | Path,
    model_config_path: str | Path,
    preds_per_term: int = 50,
    batch_size: int = 128,
    worker_count: int = 4,
):
    terms_to_idx_df = pl.read_csv(TERM_TO_IDX_LOOKUP_PATH)

    with open(WEIGHTS_BASE_PATH / model_config_path, "r") as f:
        model_config = json.load(f)

    model = Model(
        **model_config,
        key=jax.random.key(0),
    )

    model = eqx.tree_deserialise_leaves(model_path, model)
    model = eqx.nn.inference_mode(model)
    model = eqx.filter_jit(model)

    test_dataloader = get_test_loader(batch_size, worker_count)
    condition = jnp.array([25.0])

    buffer_ids = []
    buffer_indices = []
    buffer_values = []

    for batch in tqdm(test_dataloader):
        protein_ids, embeddings_esm, neighbor_priors, taxons, masks = batch

        top_indices, top_values = _get_top_preds(
            model,
            embeddings_esm,
            neighbor_priors,
            taxons,
            masks,
            condition,
            preds_per_term,
        )

        top_indices = np.array(top_indices).flatten()
        top_values = np.array(top_values).flatten()
        protein_ids_repeated = np.repeat(protein_ids, preds_per_term)

        buffer_ids.extend(protein_ids_repeated)
        buffer_indices.extend(top_indices)
        buffer_values.extend(top_values)

    raw_submission = pl.DataFrame(
        {"EntryID": buffer_ids, "index": buffer_indices, "value": buffer_values}
    )

    raw_submission = raw_submission.cast({"index": pl.Int64})

    current_preds = raw_submission.join(terms_to_idx_df, on="index", how="left")
    current_preds = current_preds.select(["EntryID", "term", "value"])

    partial_submission = pl.read_csv(PARTIAL_SUBMISSION_FULL_PATH)
    partial_submission = partial_submission.unique(subset=["EntryID", "term"])

    new_preds = current_preds.join(
        partial_submission, on=["EntryID", "term"], how="anti"
    )

    final_submission = new_preds.vstack(partial_submission)
    final_submission = cast(pl.DataFrame, final_submission)

    final_submission.write_csv("submission.tsv", include_header=False, separator="\t")


if __name__ == "__main__":
    model_name = "ingenious-gleaming-chamois"
    model_path = WEIGHTS_BASE_PATH / f"{model_name}.eqx"
    model_config_path = WEIGHTS_BASE_PATH / f"{model_name}-config.json"
    create_submission(
        model_path,
        model_config_path,
    )
