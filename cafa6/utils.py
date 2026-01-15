import numpy as np
import obonet
import polars as pl

from cafa6.constants import (
    IA_PATH,
    OBO_PATH,
)


def get_graph_edges(terms_to_idx: dict) -> tuple[np.ndarray, np.ndarray]:
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
    return children, parents


def get_go_term_weights() -> np.ndarray:
    df = pl.read_csv(IA_PATH, separator="\t")
    weights = df["weight"].to_numpy()
    return weights
