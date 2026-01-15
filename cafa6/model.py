import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from cafa6.config import TrainConfig


class ESMModel(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(
        self,
        config: TrainConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.mlp = eqx.nn.MLP(
            in_size=config.esm_embedding_size,
            out_size=config.n_terms,
            depth=config.esm_model_depth,
            width_size=config.esm_model_width_size,
            activation=jax.nn.silu,
            key=key,
        )

    def __call__(self, esm_emb: Array):
        return self.mlp(esm_emb)


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


class PerTermGating(eqx.Module):
    gate_mlp: eqx.nn.MLP

    n_terms: int

    def __init__(self, config: TrainConfig, *, key: PRNGKeyArray):
        self.gate_mlp = eqx.nn.MLP(
            in_size=config.esm_embedding_size,
            out_size=config.n_terms * 2,
            width_size=config.gate_mlp_width_size,
            depth=config.gate_mlp_depth,
            key=key,
        )
        self.n_terms = config.n_terms

    def __call__(
        self,
        protein_emb,
        deepgo_logits,
        esm_logits,
    ):
        stacked = jnp.stack([deepgo_logits, esm_logits], axis=-1)

        gate_flat = self.gate_mlp(protein_emb)
        gates = gate_flat.reshape(self.n_terms, 2)
        gates = jax.nn.softmax(gates, axis=-1)

        return (stacked * gates).sum(axis=-1)


class Model(eqx.Module):
    gate: PerTermGating
    deepgo: DeepGOModel

    esm_model: ESMModel
    esm_norm: eqx.nn.LayerNorm

    condition_mlp: eqx.nn.MLP

    dropout: eqx.nn.Dropout
    inference: bool

    def __init__(
        self,
        config: TrainConfig,
        *,
        key: PRNGKeyArray,
    ):
        key, deepgo_key, esm_key, cond_key, gate_key = jax.random.split(key, 5)

        self.gate = PerTermGating(config, key=gate_key)
        self.deepgo = DeepGOModel(config, key=deepgo_key)

        self.esm_norm = eqx.nn.LayerNorm(config.esm_embedding_size)
        self.esm_model = ESMModel(config, key=esm_key)

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
        condition: Float[Array, "1"],
        key: PRNGKeyArray | None = None,
    ):
        orig_esm_emb = esm_emb
        esm_emb = self.dropout(esm_emb, key=key)
        cond_emb = self.condition_mlp(condition)
        esm_emb = esm_emb + cond_emb

        deepgo_logits = self.deepgo(orig_esm_emb)

        esm_emb = self.esm_norm(esm_emb)
        esm_logits = self.esm_model(esm_emb)

        return self.gate(orig_esm_emb, deepgo_logits, esm_logits)
