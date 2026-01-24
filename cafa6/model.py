import equinox as eqx
import jax
import jax.numpy as jnp
from jaxonlayers.layers import TransformerEncoderLayer
from jaxtyping import Array, Float, PRNGKeyArray

from cafa6.config import TrainConfig


class DeepGOModel(eqx.Module):
    term_centers: eqx.nn.Embedding
    term_radii: Array

    has_function: Array
    mlp: eqx.nn.MLP

    def __init__(
        self,
        config: TrainConfig,
        *,
        key: PRNGKeyArray,
    ):
        key, mlp_key, radii_key, fn_key = jax.random.split(key, 4)
        self.term_centers = eqx.nn.Embedding(
            config.n_terms, config.deepgo_se_embedding_size, key=key
        )
        self.term_radii = jax.random.normal(key=radii_key, shape=(config.n_terms, 1))
        self.has_function = jax.random.normal(
            key=fn_key, shape=(config.deepgo_se_embedding_size,)
        )

        self.mlp = eqx.nn.MLP(
            in_size=config.esm_embedding_size,
            out_size=config.deepgo_se_embedding_size,
            width_size=config.esm_proj_width_size,
            depth=config.esm_proj_depth,
            activation=jax.nn.silu,
            key=mlp_key,
        )

    def __call__(
        self,
        esm_emb: Float[Array, "esm_embs"],
        key: PRNGKeyArray | None = None,
    ):
        protein_proj = self.mlp(esm_emb).reshape(1, -1)
        term_centers = self.term_centers.weight
        shifted_term_centers = (term_centers + self.has_function).T
        similarity_vector = protein_proj @ shifted_term_centers

        return (similarity_vector + self.term_radii.T).squeeze()


class Model(eqx.Module):
    deepgo: DeepGOModel

    taxon_embeddings: eqx.nn.Embedding
    taxon_linear: eqx.nn.Linear

    condition_mlp: eqx.nn.MLP

    dropout: eqx.nn.Dropout
    inference: bool

    def __init__(
        self,
        config: TrainConfig,
        *,
        key: PRNGKeyArray,
    ):
        key, deepgo_key, taxon_key, taxon_lin_key, cond_key = jax.random.split(key, 5)

        self.deepgo = DeepGOModel(config, key=deepgo_key)

        self.taxon_embeddings = eqx.nn.Embedding(
            config.n_taxons, config.taxon_embedding_size, key=taxon_key
        )
        self.taxon_linear = eqx.nn.Linear(
            in_features=config.taxon_embedding_size,
            out_features=config.n_terms,
            key=taxon_lin_key,
        )

        self.condition_mlp = eqx.nn.MLP(
            in_size=1,
            out_size=config.esm_embedding_size,
            width_size=64,
            depth=2,
            activation=jax.nn.silu,
            key=cond_key,
        )

        self.dropout = eqx.nn.Dropout()
        self.inference = False

    def __call__(
        self,
        esm_emb: Array,
        taxon: Array,
        condition: Float[Array, "1"],
        key: PRNGKeyArray | None = None,
    ):
        esm_emb = self.dropout(esm_emb, key=key)
        esm_emb += self.condition_mlp(condition)
        deepgo_logits = self.deepgo(esm_emb)
        taxon_logits = self.taxon_linear(self.taxon_embeddings(taxon))

        return deepgo_logits + taxon_logits
