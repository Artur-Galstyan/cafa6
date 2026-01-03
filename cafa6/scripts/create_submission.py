import json
from collections import defaultdict
from pathlib import Path
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from fire import Fire
from tqdm import tqdm

from cafa6.constants import (
    OBO_PATH,
    PARTIAL_SUBMISSION_FULL_PATH,
    TERM_TO_IDX_LOOKUP_PATH,
    WEIGHTS_BASE_PATH,
)
from cafa6.data import get_test_loader
from cafa6.model import Model


def parse_obo_parents(go_obo_path: Path | str) -> tuple[dict[str, set[str]], set[str]]:
    """Parse OBO file to get parent relationships."""
    term_parents: dict[str, set[str]] = defaultdict(set)
    roots = {"GO:0003674", "GO:0008150", "GO:0005575"}  # MF, BP, CC

    with open(go_obo_path, "r") as f:
        cur_id = None
        for line in f:
            line = line.strip()
            if line == "[Term]":
                cur_id = None
            elif line.startswith("id: "):
                cur_id = line.split("id: ")[1].strip()
            elif line.startswith("is_a: "):
                pid = line.split()[1].strip()
                if cur_id:
                    term_parents[cur_id].add(pid)
            elif line.startswith("relationship: part_of "):
                parts = line.split()
                if len(parts) >= 3:
                    pid = parts[2].strip()
                    if cur_id:
                        term_parents[cur_id].add(pid)

    return dict(term_parents), roots


def build_ancestors_map(term_parents: dict[str, set[str]]) -> dict[str, set[str]]:
    """Build map of term -> all ancestors."""
    ancestors: dict[str, set[str]] = {}

    def get_all_ancestors(term: str) -> set[str]:
        if term in ancestors:
            return ancestors[term]
        parents = term_parents.get(term, set())
        all_anc = set(parents)
        for p in parents:
            all_anc |= get_all_ancestors(p)
        ancestors[term] = all_anc
        return all_anc

    for term in term_parents.keys():
        get_all_ancestors(term)

    return ancestors


def postprocess_submission(
    df: pl.DataFrame,
    ancestors_map: dict[str, set[str]],
    roots: set[str],
    output_path: str,
) -> int:
    """Apply propagation + normalization, write directly to file."""

    # Group by protein
    data_map: dict[str, dict[str, float]] = defaultdict(dict)

    for row in tqdm(df.iter_rows(named=True), desc="Grouping", total=len(df)):
        pid = row["EntryID"]
        term = row["term"]
        score = float(row["value"])
        data_map[pid][term] = score

    row_count = 0

    with open(output_path, "w") as f:
        for pid, terms_dict in tqdm(data_map.items(), desc="Propagating"):
            final_scores = terms_dict.copy()

            # A. Propagate to ancestors
            for term, score in terms_dict.items():
                if term in ancestors_map:
                    for anc in ancestors_map[term]:
                        final_scores[anc] = max(final_scores.get(anc, 0.0), score)

            # B. Force roots = 1.0
            if len(final_scores) > 0:
                for r in roots:
                    final_scores[r] = 1.0

            # C. Rank normalization
            max_val = 0.0
            for t, s in final_scores.items():
                if t not in roots:
                    max_val = max(max_val, s)

            if max_val > 0 and max_val < 0.95:
                scale_factor = 0.95 / max_val
                for t in final_scores:
                    if t not in roots:
                        final_scores[t] = min(1.0, final_scores[t] * scale_factor)

            # Write directly to file
            for go_term, score in final_scores.items():
                if score >= 0.001:
                    f.write(f"{pid}\t{go_term}\t{score}\n")
                    row_count += 1

    return row_count


@eqx.filter_jit
def _get_preds(
    model,
    embeddings_esm,
    neighbor_priors,
    text_neighbor_priors,
    taxons,
    masks,
    condition,
):
    preds = eqx.filter_vmap(model, in_axes=(0, 0, 0, 0, 0, None, None))(
        embeddings_esm,
        neighbor_priors,
        text_neighbor_priors,
        taxons,
        masks,
        condition,
        None,
    )
    preds = jax.nn.sigmoid(preds)
    return preds


def create_submission(
    model_name: str,
    preds_per_term: int = 100,  # increased to capture more terms before propagation
    batch_size: int = 128,
    worker_count: int = 4,
    tta_conditions: list[float] | None = None,
    apply_postprocess: bool = True,
):
    if tta_conditions is None:
        tta_conditions = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]

    model_path = WEIGHTS_BASE_PATH / f"{model_name}.eqx"
    model_config_path = WEIGHTS_BASE_PATH / f"{model_name}-config.json"

    terms_to_idx_df = pl.read_csv(TERM_TO_IDX_LOOKUP_PATH)

    with open(WEIGHTS_BASE_PATH / model_config_path, "r") as f:
        model_config = json.load(f)

    model = Model(
        **model_config,
        key=jax.random.key(0),
    )

    model = eqx.tree_deserialise_leaves(model_path, model)
    model = eqx.nn.inference_mode(model)

    test_dataloader = get_test_loader(batch_size, worker_count)

    buffer_ids = []
    buffer_indices = []
    buffer_values = []

    for batch in tqdm(test_dataloader, desc="Inference"):
        (
            protein_ids,
            embeddings_esm,
            neighbor_priors,
            text_neighbor_priors,
            taxons,
            masks,
        ) = batch

        all_preds = []
        for cond_val in tta_conditions:
            condition = jnp.array([cond_val])
            preds = _get_preds(
                model,
                embeddings_esm,
                neighbor_priors,
                text_neighbor_priors,
                taxons,
                masks,
                condition,
            )
            all_preds.append(preds)

        avg_preds = jnp.stack(all_preds).mean(axis=0)

        top_values, top_indices = jax.lax.top_k(avg_preds, preds_per_term)

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
    new_preds = new_preds.cast({"value": pl.Float64})

    final_submission = new_preds.vstack(partial_submission)
    final_submission = cast(pl.DataFrame, final_submission)

    output_path = f"{model_name}-submission.tsv"

    if apply_postprocess:
        print("Applying post-processing (propagation + normalization)...")
        term_parents, roots = parse_obo_parents(OBO_PATH)
        ancestors_map = build_ancestors_map(term_parents)
        row_count = postprocess_submission(
            final_submission, ancestors_map, roots, output_path
        )
        print(f"Saved {row_count} rows to {output_path}")
    else:
        final_submission = final_submission.sort(
            ["EntryID", "value"], descending=[False, True]
        )
        final_submission.write_csv(output_path, include_header=False, separator="\t")
        print(f"Saved {len(final_submission)} rows to {output_path}")


if __name__ == "__main__":
    Fire(create_submission)
