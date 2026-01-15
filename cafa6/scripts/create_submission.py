import json
from pathlib import Path
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from fire import Fire
from grain import DataLoader
from tqdm import tqdm

from cafa6.config import TrainConfig
from cafa6.constants import (
    IA_PATH,
    PARTIAL_SUBMISSION_FULL_PATH,
    WEIGHTS_BASE_PATH,
)
from cafa6.data import get_test_loader
from cafa6.model import Model


@eqx.filter_jit
def _get_preds(model, esm_emb, condition):
    preds = eqx.filter_vmap(model, in_axes=(0, None, None))(
        esm_emb,
        condition,
        None,
    )
    preds = jax.nn.sigmoid(preds)
    return preds


def create_submission(
    model_name: str | None = None,
    model: Model | None = None,
    data_loader: DataLoader | None = None,
    preds_per_term: int = 50,
    batch_size: int = 128,
    worker_count: int = 4,
    tta_conditions: list[float] | None = None,
    save: bool = False,
    include_partial: bool = True,
    is_val: bool = False,
) -> pl.DataFrame:
    if tta_conditions is None:
        tta_conditions = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]

    if not model:
        assert model_name is not None, (
            "when no model is provided, need model name for loading"
        )

        model_path = WEIGHTS_BASE_PATH / f"{model_name}.eqx"
        model_config_path = WEIGHTS_BASE_PATH / f"{model_name}-config.json"

        with open(WEIGHTS_BASE_PATH / model_config_path, "r") as f:
            model_config = json.load(f)

        model_config = TrainConfig.model_validate(model_config)
        model = Model(
            model_config,
            key=jax.random.key(0),
        )

        model = eqx.tree_deserialise_leaves(model_path, model)

    model = eqx.nn.inference_mode(model)

    ia_df = pl.read_csv(IA_PATH, separator="\t")
    idx_to_term = {i: row["term"] for i, row in enumerate(ia_df.iter_rows(named=True))}

    if not data_loader:
        data_loader = get_test_loader(batch_size, worker_count)

    buffer_ids = []
    buffer_indices = []
    buffer_values = []

    for batch in tqdm(data_loader, desc="Inference"):
        if is_val:
            idx, esm_emb, _ = batch
        else:
            idx, esm_emb, taxon = batch
        esm_emb = jnp.array(esm_emb)

        all_preds = []
        for cond_val in tta_conditions:
            condition = jnp.array([cond_val])
            preds = _get_preds(model, esm_emb, condition)
            all_preds.append(preds)

        avg_preds = jnp.stack(all_preds).mean(axis=0)

        top_values, top_indices = jax.lax.top_k(avg_preds, preds_per_term)

        top_indices = np.array(top_indices).flatten()
        top_values = np.array(top_values).flatten()
        protein_ids_repeated = np.repeat(idx, preds_per_term)

        buffer_ids.extend(protein_ids_repeated)
        buffer_indices.extend(top_indices)
        buffer_values.extend(top_values)

    raw_submission = pl.DataFrame(
        {"EntryID": buffer_ids, "index": buffer_indices, "value": buffer_values}
    )

    current_preds = raw_submission.with_columns(
        pl.col("index").replace_strict(idx_to_term).alias("term")
    ).select(["EntryID", "term", "value"])

    if include_partial:
        partial_submission = pl.read_csv(PARTIAL_SUBMISSION_FULL_PATH)
        partial_submission = partial_submission.unique(subset=["EntryID", "term"])
        new_preds = current_preds.join(
            partial_submission, on=["EntryID", "term"], how="anti"
        )
        new_preds = new_preds.cast({"value": pl.Float64})
        final_submission = new_preds.vstack(partial_submission)
    else:
        final_submission = current_preds
    final_submission = cast(pl.DataFrame, final_submission)

    output_path = f"{model_name}-submission.tsv"

    final_submission = final_submission.sort(
        ["EntryID", "value"], descending=[False, True]
    )
    if save:
        final_submission.write_csv(output_path, include_header=False, separator="\t")
        model_py_path = Path(__file__).parent.parent / "model.py"
        architecture_output_path = WEIGHTS_BASE_PATH / f"{model_name}-architecture.txt"
        with open(model_py_path, "r") as f:
            model_code = f.read()
        with open(architecture_output_path, "w") as f:
            f.write(model_code)
        print(f"Saved model architecture to {architecture_output_path}")

        print(f"Saved {len(final_submission)} rows to {output_path}")
    return final_submission


def main():
    Fire(create_submission)


if __name__ == "__main__":
    main()
