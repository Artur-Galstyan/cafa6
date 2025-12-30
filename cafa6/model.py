import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray


class AttentionPooling(eqx.Module):
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


class ESMModel(eqx.Module):
    mlp: eqx.nn.MLP
    prior_scale: Array

    def __init__(
        self,
        in_size: int,
        out_size: int,
        depth: int,
        width_size: int,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2 = jax.random.split(key, 2)

        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            depth=depth,
            width_size=width_size,
            activation=jax.nn.gelu,
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
        n_terms: int,
        embedding_size: int,
        esm_embedding_size: int,
        esm_proj_width_size: int,
        esm_proj_depth: int,
        *,
        key: PRNGKeyArray,
    ):
        key, mlp_key, radii_key, fn_key = jax.random.split(key, 4)
        self.term_centers = eqx.nn.Embedding(n_terms, embedding_size, key=key)
        self.term_radii = jax.random.normal(key=radii_key, shape=(n_terms, 1))
        self.has_function = jax.random.normal(key=fn_key, shape=(embedding_size,))
        self.mlp = eqx.nn.MLP(
            in_size=esm_embedding_size,
            out_size=embedding_size,
            width_size=esm_proj_width_size,
            depth=esm_proj_depth,
            activation=jax.nn.gelu,
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


class Model(eqx.Module):
    # attn_pool: AttentionPooling

    deepgo: DeepGOModel

    taxa_embeddings: eqx.nn.Embedding
    taxa_norm: eqx.nn.LayerNorm
    taxa_mlp: eqx.nn.MLP

    esm_model: ESMModel
    esm_norm: eqx.nn.LayerNorm

    taxa_weight: Array
    esm_weight: Array
    deepgo_weight: Array

    condition_mlp: eqx.nn.MLP

    dropout: eqx.nn.Dropout
    inference: bool

    def __init__(
        self,
        n_terms: int,
        embedding_size: int,
        esm_embedding_size: int,
        esm_proj_width_size: int,
        esm_proj_depth: int,
        taxa_vocab_size: int,
        taxa_embedding_size: int,
        taxa_mlp_depth: int,
        taxa_mlp_width: int,
        esm_model_width_size: int,
        esm_model_depth: int,
        *,
        key: PRNGKeyArray,
    ):
        key, deepgo_key, taxa_key, esm_key, cond_key, attn_key = jax.random.split(
            key, 6
        )
        self.deepgo = DeepGOModel(
            n_terms,
            embedding_size,
            esm_embedding_size,
            esm_proj_width_size,
            esm_proj_depth,
            key=deepgo_key,
        )

        self.taxa_embeddings = eqx.nn.Embedding(
            taxa_vocab_size,
            taxa_embedding_size,
            key=key,
        )
        self.taxa_mlp = eqx.nn.MLP(
            taxa_embedding_size,
            n_terms,
            depth=taxa_mlp_depth,
            width_size=taxa_mlp_width,
            activation=jax.nn.gelu,
            key=taxa_key,
        )

        self.taxa_norm = eqx.nn.LayerNorm(shape=(taxa_embedding_size))

        self.esm_norm = eqx.nn.LayerNorm(esm_embedding_size)
        self.esm_model = ESMModel(
            esm_embedding_size,
            n_terms,
            esm_model_width_size,
            esm_model_depth,
            key=esm_key,
        )
        self.deepgo_weight = jnp.array(1.0)
        self.esm_weight = jnp.array(1.0)
        self.taxa_weight = jnp.array(1.0)

        self.condition_mlp = eqx.nn.MLP(
            in_size=1,
            out_size=esm_embedding_size,
            width_size=64,
            depth=2,
            activation=jax.nn.gelu,
            key=cond_key,
        )

        self.dropout = eqx.nn.Dropout()
        self.inference = False
        # self.attn_pool = AttentionPooling(esm_embedding_size, key=attn_key)

    def __call__(
        self,
        esm_emb: Array,
        neighbor_prior: Float[Array, "n_terms"],
        taxa: Int[Array, ""],
        mask: Float[Array, "max_seq_len"],
        condition: Float[Array, "1"],
        key: PRNGKeyArray | None = None,
    ):
        print("JIT MODEL")
        esm_emb = self.dropout(esm_emb, key=key)
        # esm_emb = self.attn_pool(esm_emb, mask)
        cond_emb = self.condition_mlp(condition)
        esm_emb += cond_emb
        taxa_embeddings = self.taxa_embeddings(taxa)
        taxa_embeddings = self.taxa_norm(taxa_embeddings)
        taxa_logits = self.taxa_mlp(taxa_embeddings)

        deepgo_logits = self.deepgo(esm_emb)

        esm_emb = self.esm_norm(esm_emb)
        esm_logits = self.esm_model(esm_emb, neighbor_prior)

        return (
            self.deepgo_weight * deepgo_logits
            + self.esm_weight * esm_logits
            + self.taxa_weight * taxa_logits
        )
