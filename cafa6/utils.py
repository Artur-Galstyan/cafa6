from pathlib import Path

import numpy as np
import obonet
import polars as pl

from cafa6.constants import (
    OBO_GRAPH_CHILDREN_PATH,
    OBO_GRAPH_PARENTS_PATH,
    OBO_PATH,
    TERM_TO_IDX_LOOKUP_PATH,
)


def get_graph_edges(terms_to_idx: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    This function parses the obo file to find out which go term is a child of which
    other go term.

    The returned arrays have the same length and each row of each array corresponds to each other.

    E.g.

    The GO term in row 1 of the children array is the child of the GO term in row 1 of the parent array.


    Returns a tuple:
        children: np.array
        parents: np.array
    """
    if (OBO_GRAPH_CHILDREN_PATH).exists() and (OBO_GRAPH_PARENTS_PATH).exists():
        return np.load(OBO_GRAPH_CHILDREN_PATH), np.load(OBO_GRAPH_PARENTS_PATH)

    graph = obonet.read_obo(OBO_PATH)

    children = []
    parents = []

    for child, parent, key in graph.edges(keys=True):
        if key == "is_a":
            if child in terms_to_idx and parent in terms_to_idx:
                child_idx = terms_to_idx[child]
                parent_idx = terms_to_idx[parent]

                children.append(child_idx)
                parents.append(parent_idx)

    children, parents = np.array(children), np.array(parents)
    np.save(OBO_GRAPH_CHILDREN_PATH, children)
    np.save(OBO_GRAPH_PARENTS_PATH, parents)
    return children, parents


def get_go_term_weights(
    term_to_idx_lookup_path: str | Path = TERM_TO_IDX_LOOKUP_PATH,
) -> np.ndarray:
    if not Path(term_to_idx_lookup_path).exists():
        raise FileNotFoundError(
            "Couldn't find the lookup dir! Run scripts/cafa6_terms_to_idx.py!"
        )
    df = pl.read_csv(term_to_idx_lookup_path)
    col = df.select(pl.col("weight"))
    col = np.array(col).reshape(-1)
    return col
