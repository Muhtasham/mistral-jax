import math
import jax
from jax import Array
import jax.numpy as jnp
import torch

import einops as op
from einshard import einshard
from transformers import MistralForCausalLM
from torch.nn import Embedding as TorchEmbedding
from transformers.models.mistral.modeling_mistral import MistralRMSNorm, MistralAttention, MistralMLP, MistralDecoderLayer
from typing import NamedTuple
from .utils import jax2pt, pt2jax

KVCache = Array | None

_k = 128
d_model = 4096
n_rep_kv = 4
n_heads_kv = 8
d_k = d_v = 128
rms_norm_eps = 1e-5

AttentionParams = tuple[Array, Array, Array, Array]

RMSNormParams = Array
EmbeddingParams = Array

MLPLayerParams = tuple[Array, Array, Array]

DecoderBlockParams = tuple[RMSNormParams, AttentionParams, MLPLayerParams, RMSNormParams]
DecoderParams = list[DecoderBlockParams]

class RotaryValues(NamedTuple):
    sin_val: Array
    cos_val: Array

def convert_embedding_params(embedding: TorchEmbedding) -> EmbeddingParams:
    """
    Converts PyTorch embedding parameters to a EmbeddingParams compatible with JAX.

    Args:
        embedding (TorchEmbedding): The PyTorch embedding layer from which to extract the weights.

    Returns:
        EmbeddingParams: The embedding parameters extracted from the PyTorch layer and formatted for compatibility with JAX operations.
    """
    return pt2jax(embedding.weight.data)

def convert_back_embedding_params():
    pass

def shard_embedding_params(params: EmbeddingParams) -> EmbeddingParams:
    """
    Shard the EmbeddingParams params for distributed computing.

    Args:
        params (EmbeddingParams): The EmbeddingParams parameters.

    Returns:
        EmbeddingParams: The decoder embedding parameters replica for distributed computation across multiple devices.
    """
    return einshard(params, '... -> * ...')

def forward_embedding(params: EmbeddingParams, input_ids: Array) -> Array:
    """
    Get the embedding with input IDS.

    Args:
        params (EmbeddingParams): The embedding parameters.
        input_ids (Array): An array of input IDS to look up the embedding.

    Returns:
        Array: The embedding Array of input IDS.
    """
    return params[input_ids]

def convert_mlp_layer_params(mlp_layer: MistralMLP) -> MLPLayerParams:
    """
    Converts PyTorch MLP layer parameters to a MLPLayerParams compatible with JAX.

    Args:
        mlp_layer (MistralMLP): The PyTorch MLP layer from which to extract the weights.

    Returns:
        MLPLayerParams: The embedding parameters extracted from the PyTorch layer and formatted for compatibility with JAX operations.
    """
    gate_proj = pt2jax(mlp_layer.gate_proj.weight.data.T)
    up_proj = pt2jax(mlp_layer.up_proj.weight.data.T)
    down_proj = pt2jax(mlp_layer.down_proj.weight.data.T)
    return gate_proj, up_proj, down_proj

def convert_back_mlp_layer_params(mlp_layer: MLPLayerParams) -> MistralMLP:
    # mlp_layer_pt = MistralMLP(config_pt)  # TODO: handle config
    pass

def shard_mlp_layer_params(params: MLPLayerParams) -> MLPLayerParams:
    """
    Shard the MLPLayerParams params for distributed computing.

    Args:
        params (MLPLayerParams): The MLPLayerParams parameters.

    Returns:
        MLPLayerParams: The decoder embedding parameters replica for distributed computation across multiple devices.
    """
    gate_proj, up_proj, down_proj = params
    gate_proj = einshard(gate_proj, 'm f -> m f*')
    up_proj = einshard(up_proj, 'm f -> m f*')
    down_proj = einshard(down_proj, 'f m -> f* m')
    return gate_proj, up_proj, down_proj

def forward_mlp_layer(params: MLPLayerParams, seq: Array) -> Array:
    """
    Executes the forward pass of MLP.

    Args:
        params (MLPLayerParams): The MLP layer parameters.
        seq (Array): The input sequences.

    Returns:
        Array: The output after MLP layer.
    """
    gate_proj, up_proj, down_proj = params

    ff = jax.nn.silu(seq @ gate_proj) * (seq @ up_proj)
    seq = ff @ down_proj
    return seq

def convert_rms_norm_params(rms_norm: MistralRMSNorm) -> RMSNormParams:
    """
    Converts PyTorch rms norm parameters to a RMSNormParams compatible with JAX.

    Args:
        rms_norm (MistralRMSNorm): The PyTorch rms norm from which to extract the weights.

    Returns:
        RMSNormParams: The rms norm parameters extracted from the PyTorch layer and formatted for compatibility with JAX operations.
    """
    return pt2jax(rms_norm.weight)

def convert_back_rms_norm_params(rms_norm: RMSNormParams) -> MistralRMSNorm:
    rms_norm_pt = MistralRMSNorm(rms_norm.shape[0], rms_norm_eps)
    rms_norm_pt.weight = torch.nn.Parameter(jax2pt(rms_norm))
    return rms_norm_pt

def shard_rms_norm_params(params: RMSNormParams) -> RMSNormParams:
    """
    Shard the RMSNormParams params for distributed computing.

    Args:
        params (RMSNormParams): The RMSNormParams parameters.

    Returns:
        RMSNormParams: The rms norm parameters replica for distributed computation across multiple devices.
    """
    return einshard(params, '... -> * ...')

# Taken from https://github.com/ayaka14732/llama-2-jax/blob/main/lib/llama/rms_norm.py
def forward_rms_norm(params: RMSNormParams, x: Array) -> Array:
    """
    Executes the forward pass of MLP.

    Args:
        params (RMSNormParams): The rms norm parameters.
        x (Array): The input array.

    Returns:
        Array: The output after rms norm.
    """
    x_rms = jnp.sqrt((x * x).mean(axis=-1, keepdims=True) + rms_norm_eps)
    y = x / x_rms * params
    return y

def convert_decoder_block_params(decoder_block: MistralDecoderLayer) -> DecoderBlockParams:
    """
    Converts decoder block parameters from MistralDecoderLayer(PyTorch tensor) to DecoderBlockParams(JAX Array).

    Args:
        decoder_block (MistralDecoderLayer): The decoder block's MistralDecoderLayer.

    Returns:
        DecoderBlockParams: The converted decoder block parameters.
    """

    input_layernorm = convert_rms_norm_params(decoder_block.input_layernorm)
    self_attn = convert_attention_params(decoder_block.self_attn)
    mlp = convert_mlp_layer_params(decoder_block.mlp)
    post_attention_layernorm = convert_rms_norm_params(decoder_block.post_attention_layernorm)
    return input_layernorm, self_attn, mlp, post_attention_layernorm

def convert_back_decoder_block_params():
    pass

def shard_decoder_block_params(params: DecoderBlockParams) -> DecoderBlockParams:
    """
    Shard the DecoderBlockParams params for distributed computing.

    Args:
        params (DecoderBlockParams): The decoder block parameters.

    Returns:
        DecoderBlockParams: The decoder block parameters modified with tensor parallelism, allowing for distributed computation across multiple devices.
    """
    input_layernorm, self_attn, mlp, post_attention_layernorm = params
    input_layernorm = shard_rms_norm_params(input_layernorm)
    self_attn = shard_attention_params(self_attn)
    mlp = shard_mlp_layer_params(mlp)
    post_attention_layernorm = shard_rms_norm_params(post_attention_layernorm)
    return input_layernorm, self_attn, mlp, post_attention_layernorm

def forward_decoder_block(params: DecoderBlockParams, seq: Array, qk_mask: Array, rotary_values: RotaryValues ,kv_cache_cur: KVCache, kv_cache_pre: KVCache) -> tuple[Array, KVCache, KVCache]:
    """
    Executes the forward pass of a decoder block using the specified parameters and input sequence.

    Args:
        params (DecoderBlockParams): The decoder block parameters.
        seq (Array): The input sequences to the decoder block.
        qk_mask (Array): The qk mask for the attention mechanism, determining which parts of the sequence are allowed to attend to each other.
        rotary_values (RotaryValues): Rotary positional embeddings values.
        kv_cache_cur (KVCache): The current KVCache.
        kv_cache_pre (KVCache): The previous KVCache.

    Returns:
        tuple[Array, KVCache, KVCache]: A tuple containing the output sequence after decoder block, and the updated current and previous KVCache.
    """
    input_layernorm, self_attn, mlp, post_attention_layernorm = params

    # residual connection
    seq_ = seq
    seq = forward_rms_norm(input_layernorm, seq)
    seq, kv_cache_cur, kv_cache_pre = forward_attention(self_attn, seq, qk_mask, rotary_values, kv_cache_cur, kv_cache_pre)
    seq += seq_

    seq_ = seq
    seq = forward_rms_norm(post_attention_layernorm, seq)
    seq = forward_mlp_layer(mlp, seq)
    seq += seq_
    return seq, kv_cache_cur, kv_cache_pre

def _make_weights(seq_len: int, d_k: int) -> tuple[Array, Array]:
    inv_freq = 1. / (1000000 ** (jnp.arange(0, d_k, 2) / d_k))
    sinusoid_inp = op.einsum(jnp.arange(seq_len), inv_freq, 'L, j -> L j')
    sin_val = jnp.sin(sinusoid_inp)
    cos_val = jnp.cos(sinusoid_inp)
    sin_val = op.repeat(sin_val, 'L K -> L (i K)', i=2)
    cos_val = op.repeat(cos_val, 'L K -> L (i K)', i=2)
    return sin_val, cos_val

def _rotate_half(x: Array) -> Array:
    x = op.rearrange(x, '... (i x) -> ... i x', i=2)  # split the last dimension: (..., n) -> (..., 2, n // 2)
    x = x[..., ::-1, :]  # reverse dimension -2
    x = x.at[..., 0, :].multiply(-1)  # negate the first half of dimension -2
    x = op.rearrange(x, '... i x -> ... (i x)')  # merge the last two dimensions: (..., 2, n // 2) -> (..., n)
    return x


def forward_rotary_embedding(m: Array, *, rotary_values: RotaryValues) -> Array:
    sin_val, cos_val = rotary_values
    assert sin_val.dtype == jnp.float32
    assert cos_val.dtype == jnp.float32
    n = _rotate_half(m)
    a = op.einsum(m, cos_val, 'B ... L K, B L K -> B ... L K').astype(m.dtype)
    b = op.einsum(n, sin_val, 'B ... L K, B L K -> B ... L K').astype(m.dtype)
    return a + b

def make_rotary_values(batch_size: int, seq_len: int) -> RotaryValues:
    """
    Generates sine and cosine values for rotary positional embeddings based on sequence length.

    Args:
        batch_size (int): The number of sequences in a batch.
        seq_len (int): The length of every sequences in a batch.

    Returns:
        RotaryValues: Rotary embedding values with sine values, and cosine values.
    """
    sin_val, cos_val = _make_weights(seq_len, d_k)

    sin_val = jnp.repeat(sin_val[None], batch_size, axis=0)
    cos_val = jnp.repeat(cos_val[None], batch_size, axis=0)
    return RotaryValues(sin_val, cos_val)

def get_rotary_values_at_position(rotary_values: RotaryValues, position: Array) -> RotaryValues:
    """
    Extracts the rotary positional embedding values for a specific position across all sequences in a batch.

    Args:
        rotary_values (RotaryValues): The rotary values from which to extract the positional embeddings.
        position (Array): The position for which to extract the rotary values.

    Returns:
        RotaryValues: Rotary embedding values for the specified position.
    """
    sin_val, cos_val = rotary_values
    sin_val = sin_val[:, position][:, None]
    cos_val = cos_val[:, position][:, None]

    rotary_values = RotaryValues(sin_val, cos_val)
    return rotary_values

def convert_attention_params(self_attn: MistralAttention) -> AttentionParams:
    """
    Converts the attention parameters from MistralAttention HuggingFace format with PyTorch `tensor` to `jax.Array`.

    Args:
        self_attn (MistralAttention): The attention parameters in MistralAttention.

    Returns:
        AttentionParams: The attention parameters converted into the AttentionParams with JAX.
    """
    q_proj = self_attn.q_proj.weight.data
    k_proj = self_attn.k_proj.weight.data
    v_proj = self_attn.v_proj.weight.data
    o_proj = self_attn.o_proj.weight.data

    q_proj_jax = pt2jax(q_proj.T).reshape(d_model, n_heads_kv, n_rep_kv, d_k).transpose(0, 2, 1, 3)
    k_proj_jax = pt2jax(k_proj.T).reshape(d_model, n_heads_kv, d_k)
    v_proj_jax = pt2jax(v_proj.T).reshape(d_model, n_heads_kv, d_v)
    o_proj_jax = pt2jax(o_proj.T).reshape(n_heads_kv, n_rep_kv, d_v, d_model).transpose(1, 0, 2, 3)

    return q_proj_jax, k_proj_jax, v_proj_jax, o_proj_jax

def convert_back_attention_params():
    pass

def shard_attention_params(params: AttentionParams) -> AttentionParams:
    """
    Shard the attention parameters for distributed computing.

    Args:
        params (AttentionParams): The attention parameters.

    Returns:
        AttentionParams: The attention parameters modified with tensor parallelism, allowing for distributed computation across multiple devices.
    """
    q_proj, k_proj, v_proj, o_proj = params
    # q_proj = einshard(q_proj, 'm r h k -> m r h* k')
    # k_proj = einshard(k_proj, 'm h k -> m h* k')
    # v_proj = einshard(v_proj, 'm h v -> m h* v')
    # o_proj = einshard(o_proj, 'r h v m -> r h* v m')
    q_proj = einshard(q_proj, 'm r h k -> m r h k*')
    k_proj = einshard(k_proj, 'm h k -> m h k*')
    v_proj = einshard(v_proj, 'm h v -> m h v*')
    o_proj = einshard(o_proj, 'r h v m -> r h v* m')
    return q_proj, k_proj, v_proj, o_proj

def forward_attention(params: AttentionParams, seq: Array, qk_mask: Array, rotary_values: RotaryValues, kv_cache_cur: KVCache, kv_cache_pre: KVCache) -> tuple[Array, KVCache, KVCache]:
    """
    Performs the forward pass of the attention mechanism using.

    This function executes the attention mechanism on the input sequence `seq` using the provided attention parameters. 

    Args:
        params (AttentionParams): The attention parameters.
        seq (Array): The input sequences on which attention is to be applied.
        qk_mask (Array): The qk mask for the attention mechanism, determining which parts of the sequence are allowed to attend to each other.
        rotary_values (RotaryValues): Rotary positional embeddings values.
        kv_cache_cur (KVCache): The current KVCache.
        kv_cache_pre (KVCache): The previous KVCache.

    Returns:
        tuple[Array, KVCache, KVCache]: A tuple containing the output sequence after applying attention, and the updated current and previous KVCache.
    """
    q_proj_jax, k_proj_jax, v_proj_jax, o_proj_jax = params

    # for q, the seq is src_seq, 
    # for k and v, the seq is des_seq,
    # in self_atten the src_ and des_seq are the same

    # q.shape: (1 batch_size, 4 n_rep_kv, 8 n_head, 6 seq_len ?, 128 k_dimension)
    # k.shape: (1 batch_size, 8 n_head, 6 seq_len, 128 k_dimension)
    # v.shape: (1 batch_size, 8 n_head, 6 seq_len, 128 v_dimension)

    # einsum can use to apply matrix multiplication
    q = op.einsum(seq, q_proj_jax, 'b s m, m r h k -> b r h s k')
    k = op.einsum(seq, k_proj_jax, 'b d m, m h k -> b h d k')
    v = op.einsum(seq, v_proj_jax, 'b d m, m h v -> b h d v')

    # before self attention, add position information
    # q.shape: (1 batch_size, 4, 8, 6 seq_len, 128)
    q = forward_rotary_embedding(q, rotary_values=rotary_values)
    k = forward_rotary_embedding(k, rotary_values=rotary_values)

    # KVCache to optimize generation
    if kv_cache_pre is not None:
        layer_n = 0 if kv_cache_cur is None else kv_cache_cur.shape[1]
        k = jnp.concatenate((kv_cache_pre[0, layer_n, ...], k), axis=-2)
        v = jnp.concatenate((kv_cache_pre[1, layer_n, ...], v), axis=-2)

    k_cache_cur = k[None, ...] if kv_cache_cur is None else jnp.concatenate((kv_cache_cur[0, ...], k[None, ...]), axis=0)
    v_cache_cur = v[None, ...] if kv_cache_cur is None else jnp.concatenate((kv_cache_cur[1, ...], v[None, ...]), axis=0)
    kv_cache_cur = jnp.concatenate((k_cache_cur[None, ...], v_cache_cur[None, ...]), axis=0)
    # self-attention
    # (1 batch_size, 4 repetition, 8 head number, 6 seq_len, 6 seq_len)
    # Scaled Dot-Product Attention as 3.2.1 equation(1) in orginal Transformer paper
    qk = jnp.einsum('brhsk,bhdk->brhsd', q, k) / math.sqrt(d_k)

    qk = jax.nn.softmax(qk, where=qk_mask, initial=0.)
    qkv = jnp.einsum('brhsd,bhdv->brhsv', qk, v)
    out = jnp.einsum('brhsv,rhvm->bsm', qkv, o_proj_jax)
    return out, kv_cache_cur, kv_cache_pre
