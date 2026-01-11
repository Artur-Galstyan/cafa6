import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from cafa6.config import TrainConfig


class AttentionPooling(eqx.Module):
    """
    Unused; Mean-Pooling works good enough
    """

    layer: eqx.nn.Linear

    def __init__(self, in_size: int, key: PRNGKeyArray):
        self.layer = eqx.nn.Linear(in_size, 1, key=key)

    def __call__(
        self,
        x: Float[Array, "max_seq_len embed_dim"],
        mask: Float[Array, "max_seq_len"],
    ) -> Array:
        scores = jnp.squeeze(eqx.filter_vmap(self.layer)(x))
        scores = jnp.where(mask == 1, scores, -1e9)
        weights = jax.nn.softmax(scores)
        weighted_embeddings = x * weights[:, None]
        pooled = jnp.sum(weighted_embeddings, axis=0)
        return pooled


class TextEmbeddingModel(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(
        self,
        config: TrainConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.mlp = eqx.nn.MLP(
            in_size=config.text_embedding_size,
            out_size=config.n_terms,
            depth=config.text_embedding_mlp_depth,
            width_size=config.text_embedding_mlp_width,
            activation=jax.nn.silu,
            key=key,
        )

    def __call__(self, text_neighbor_emb: Array):
        return self.mlp(text_neighbor_emb)


# class TextEmbeddingModel(eqx.Module):
#     esm_mlp: eqx.nn.MLP
#     text_mlp: eqx.nn.MLP

#     final: eqx.nn.Linear

#     def __init__(
#         self,
#         text_embedding_size: int,
#         esm_in_size: int,
#         hidden_size: int,
#         out_size: int,
#         depth: int,
#         width_size: int,
#         *,
#         key: PRNGKeyArray,
#     ):
#         k1, k2, k3 = jax.random.split(key, 3)

#         self.esm_mlp = eqx.nn.MLP(
#             in_size=esm_in_size,
#             out_size=hidden_size,
#             depth=depth,
#             width_size=width_size,
#             activation=jax.nn.silu,
#             key=k1,
#         )
#         self.text_mlp = eqx.nn.MLP(
#             in_size=text_embedding_size,
#             out_size=hidden_size,
#             depth=depth,
#             width_size=width_size,
#             activation=jax.nn.silu,
#             key=k2,
#         )
#         self.final = eqx.nn.Linear(
#             in_features=hidden_size * 2, out_features=out_size, key=k3
#         )

#     def __call__(
#         self, esm_emb: Array, neighbor_prior: Float[Array, "text_embedding_size"]
#     ):
#         esm_logits = self.esm_mlp(esm_emb)
#         text_logits = self.text_mlp(neighbor_prior)
#         logits = self.final(jnp.concatenate([esm_logits, text_logits]))

#         return logits


class ESMModel(eqx.Module):
    mlp: eqx.nn.MLP
    prior_scale: Array

    def __init__(
        self,
        config: TrainConfig,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2 = jax.random.split(key, 2)

        self.mlp = eqx.nn.MLP(
            in_size=config.esm_embedding_size,
            out_size=config.n_terms,
            depth=config.esm_model_depth,
            width_size=config.esm_model_width_size,
            activation=jax.nn.silu,
            key=k2,
        )

        self.prior_scale = jnp.array(0.1)

    def __call__(self, esm_emb: Array, neighbor_prior: Float[Array, "n_terms"]):
        logits = self.mlp(esm_emb)
        clamped_prior = jnp.clip(neighbor_prior, 0.01, 0.99)
        prior_logits = jnp.log(clamped_prior) - jnp.log(1 - clamped_prior)
        logits = logits + self.prior_scale * prior_logits

        return logits


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
        protein_proj = self.mlp(esm_emb).reshape(1, -1)  # 1 x E
        term_centers = self.term_centers.weight
        shifted_term_centers = (term_centers + self.has_function).T  # E x N
        similarity_vector = (
            protein_proj @ shifted_term_centers
        )  # 1 x E times E x N = 1 x N

        return (similarity_vector + self.term_radii.T).squeeze()


class PerTermGating(eqx.Module):
    gate_mlp: eqx.nn.MLP

    n_terms: int

    def __init__(self, config: TrainConfig, *, key: PRNGKeyArray):
        self.gate_mlp = eqx.nn.MLP(
            in_size=config.esm_embedding_size,
            out_size=config.n_terms * 5,
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
        taxa_logits,
        text_logits,
        tea_logits,
    ):
        stacked = jnp.stack(
            [deepgo_logits, esm_logits, taxa_logits, text_logits, tea_logits], axis=-1
        )  # (n_terms, 3)

        gate_flat = self.gate_mlp(protein_emb)
        gates = gate_flat.reshape(self.n_terms, 5)
        gates = jax.nn.softmax(gates, axis=-1)

        return (stacked * gates).sum(axis=-1)


class Model(eqx.Module):
    # attn_pool: AttentionPooling

    gate: PerTermGating

    deepgo: DeepGOModel

    taxa_embeddings: eqx.nn.Embedding
    taxa_norm: eqx.nn.LayerNorm
    taxa_mlp: eqx.nn.MLP

    esm_model: ESMModel
    esm_norm: eqx.nn.LayerNorm

    condition_mlp: eqx.nn.MLP

    dropout: eqx.nn.Dropout
    inference: bool

    text_embedding_model: TextEmbeddingModel
    tea_mlp: eqx.nn.MLP

    def __init__(
        self,
        config: TrainConfig,
        *,
        key: PRNGKeyArray,
    ):
        (
            key,
            deepgo_key,
            taxa_key,
            esm_key,
            cond_key,
            attn_key,
            gate_key,
            text_key,
            tea_key,
        ) = jax.random.split(key, 9)
        self.gate = PerTermGating(config, key=gate_key)
        self.deepgo = DeepGOModel(
            config,
            key=deepgo_key,
        )

        self.taxa_embeddings = eqx.nn.Embedding(
            config.taxa_vocab_size,
            config.taxa_embedding_size,
            key=key,
        )
        self.taxa_mlp = eqx.nn.MLP(
            config.taxa_embedding_size,
            config.n_terms,
            depth=config.taxa_mlp_depth,
            width_size=config.taxa_mlp_width,
            activation=jax.nn.silu,
            key=taxa_key,
        )

        self.taxa_norm = eqx.nn.LayerNorm(shape=(config.taxa_embedding_size))

        self.esm_norm = eqx.nn.LayerNorm(config.esm_embedding_size)
        self.esm_model = ESMModel(
            config,
            key=esm_key,
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
        # self.attn_pool = AttentionPooling(esm_embedding_size, key=attn_key)

        # self.text_embedding_model = TextEmbeddingModel(
        #     text_embedding_size,
        #     esm_embedding_size,
        #     text_embedding_hidden_size,
        #     n_terms,
        #     text_embedding_mlp_depth,
        #     text_embedding_mlp_width,
        #     key=text_key,
        # )
        self.text_embedding_model = TextEmbeddingModel(
            config,
            key=text_key,
        )

        self.tea_mlp = eqx.nn.MLP(
            in_size=config.n_terms,
            out_size=config.n_terms,
            width_size=config.tea_mlp_width_size,
            depth=config.tea_mlp_depth,
            key=tea_key,
        )

    def __call__(
        self,
        esm_emb: Array,
        neighbor_prior: Float[Array, "n_terms"],
        text_neighbor_prior: Float[Array, "text_embedding_size"],
        tea_neighbor_prior: Float[Array, "n_terms"],
        taxa: Int[Array, ""],
        mask: Float[Array, "max_seq_len"],
        condition: Float[Array, "1"],
        key: PRNGKeyArray | None = None,
    ):
        orig_esm_emb = esm_emb
        esm_emb = self.dropout(esm_emb, key=key)
        cond_emb = self.condition_mlp(condition)
        esm_emb += cond_emb
        taxa_embeddings = self.taxa_embeddings(taxa)
        taxa_embeddings = self.taxa_norm(taxa_embeddings)
        taxa_logits = self.taxa_mlp(taxa_embeddings)

        deepgo_logits = self.deepgo(orig_esm_emb)

        esm_emb = self.esm_norm(esm_emb)
        esm_logits = self.esm_model(esm_emb, neighbor_prior)

        # text_logits = self.text_embedding_model(orig_esm_emb, text_neighbor_prior)
        text_logits = self.text_embedding_model(text_neighbor_prior)
        tea_logits = self.tea_mlp(tea_neighbor_prior)
        return self.gate(
            orig_esm_emb,
            deepgo_logits,
            esm_logits,
            taxa_logits,
            text_logits,
            tea_logits,
        )
