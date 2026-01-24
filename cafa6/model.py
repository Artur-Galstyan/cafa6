import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from cafa6.config import TrainConfig


class DeepGOModel(eqx.Module):
    term_centers: eqx.nn.Embedding
    term_radii: Array
    has_function: Array
    cross_attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP

    def __init__(
        self,
        config: TrainConfig,
        *,
        key: PRNGKeyArray,
    ):
        key, mlp_key, attn_key, radii_key, fn_key = jax.random.split(key, 5)

        self.term_centers = eqx.nn.Embedding(
            config.n_terms, config.deepgo_se_embedding_size, key=key
        )
        self.term_radii = jax.random.normal(key=radii_key, shape=(config.n_terms, 1))
        self.has_function = jax.random.normal(
            key=fn_key, shape=(config.deepgo_se_embedding_size,)
        )

        self.cross_attn = eqx.nn.MultiheadAttention(
            num_heads=8,
            query_size=config.deepgo_se_embedding_size,
            key_size=config.esm_embedding_size,
            value_size=config.esm_embedding_size,
            output_size=config.deepgo_se_embedding_size,
            key=attn_key,
        )

        self.mlp = eqx.nn.MLP(
            in_size=config.deepgo_se_embedding_size,
            out_size=config.deepgo_se_embedding_size,
            width_size=config.esm_proj_width_size,
            depth=config.esm_proj_depth,
            activation=jax.nn.silu,
            key=mlp_key,
        )

    def __call__(
        self,
        esm_emb: Float[Array, "seq_len esm_dim"],
        key: PRNGKeyArray | None = None,
    ):
        term_emb = self.term_centers.weight

        attended = self.cross_attn(
            query=term_emb,
            key_=esm_emb,
            value=esm_emb,
            inference=True,
        )

        protein_aware_terms = eqx.filter_vmap(self.mlp)(attended)
        shifted = protein_aware_terms + self.has_function

        scores = jnp.sum(shifted, axis=-1) + self.term_radii.squeeze()

        return scores


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
