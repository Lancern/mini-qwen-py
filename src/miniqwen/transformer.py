from typing import Optional, TYPE_CHECKING

from safetensors import safe_open
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .cache import Cache, LayerCache


class SelfAttention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        layernorm_eps: float,
        model_tensors: safe_open,
        cache: Optional["LayerCache"] = None,
    ):
        super().__init__()

        self._head_dim = head_dim
        self._num_attention_heads = num_attention_heads
        self._num_kv_heads = num_kv_heads
        self._num_kv_groups = self._num_attention_heads // self._num_kv_heads

        self._cache = cache

        self._q_proj = nn.Linear(
            hidden_size, num_attention_heads * head_dim, bias=False
        )
        self._k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self._v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self._o_proj = nn.Linear(
            num_attention_heads * head_dim, hidden_size, bias=False
        )
        self._q_norm = LayerNorm(
            f"model.layers.{layer_idx}.self_attn.q_norm.weight",
            head_dim,
            layernorm_eps,
            model_tensors,
        )
        self._k_norm = LayerNorm(
            f"model.layers.{layer_idx}.self_attn.k_norm.weight",
            head_dim,
            layernorm_eps,
            model_tensors,
        )

        # Load linear weights from model_tensors
        self._q_proj.load_state_dict(
            {
                "weight": model_tensors.get_tensor(
                    f"model.layers.{layer_idx}.self_attn.q_proj.weight"
                )
            }
        )
        self._k_proj.load_state_dict(
            {
                "weight": model_tensors.get_tensor(
                    f"model.layers.{layer_idx}.self_attn.k_proj.weight"
                )
            }
        )
        self._v_proj.load_state_dict(
            {
                "weight": model_tensors.get_tensor(
                    f"model.layers.{layer_idx}.self_attn.v_proj.weight"
                )
            }
        )
        self._o_proj.load_state_dict(
            {
                "weight": model_tensors.get_tensor(
                    f"model.layers.{layer_idx}.self_attn.o_proj.weight"
                )
            }
        )

    def forward(
        self, x: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ):
        # x :: (batch_size, seq_len, hidden_size)

        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self._head_dim)
        # hidden_shape = (batch_size, seq_len, -1, self._head_dim)

        q_states = self._q_norm(self._q_proj(x).view(hidden_shape)).transpose(1, 2)
        # q_states :: (batch_size, num_attention_heads, seq_len, head_dim)
        k_states = self._k_norm(self._k_proj(x).view(hidden_shape)).transpose(1, 2)
        # k_states :: (batch_size, num_kv_heads, seq_len, head_dim)
        v_states = self._v_proj(x).view(hidden_shape).transpose(1, 2)
        # v_states :: (batch_size, num_kv_heads, seq_len, head_dim)

        cos, sin = position_embeddings
        q_states, k_states = self._rotary_pos_embed(q_states, k_states, cos, sin)

        attn_output = self._attend(q_states, k_states, v_states)
        # attn_output :: (batch_size, seq_len, num_attention_heads, head_dim)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # attn_output :: (batch_size, seq_len, num_attention_heads * head_dim)
        attn_output = self._o_proj(attn_output)
        # attn_output :: (batch_size, seq_len, hidden_size)

        return attn_output

    def _rotary_pos_embed(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q :: (batch_size, num_attention_heads, seq_len, head_dim)
        # k :: (batch_size, num_kv_heads, seq_len, head_dim)
        # cos, sin :: (batch_size, seq_len, head_dim)

        def rotate_half(x):
            # x :: (..., d)
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            # x1, x2 :: (..., d / 2)
            return torch.cat((-x2, x1), dim=-1)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        # cos, sin :: (batch_size, 1, seq_len, head_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def _attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        # q :: (batch_size, num_attention_heads, seq_len, head_dim)
        # k :: (batch_size, num_kv_heads, kv_seq_len, head_dim)
        # v :: (batch_size, num_kv_heads, kv_seq_len, head_dim)

        k = self._expand_kv(k)
        v = self._expand_kv(v)
        # k :: (batch_size, num_attention_heads, kv_seq_len, head_dim)
        # v :: (batch_size, num_attention_heads, kv_seq_len, head_dim)

        if self._cache is not None:
            k, v = self._cache.update_and_concat(k, v)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        scaling = self._head_dim**-0.5
        is_causal = q.shape[2] > 1
        attn_output = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=is_causal, scale=scaling
        )
        # attn_output :: (batch_size, num_attention_heads, seq_len, head_dim)

        # ret :: (batch_size, seq_len, num_attention_heads, head_dim)
        return attn_output.transpose(1, 2).contiguous()

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        # x :: (batch_size, num_kv_heads, seq_len, head_dim)
        # ret :: (batch_size, num_kv_heads * num_kv_groups, seq_len, head_dim)

        batch_size, num_kv_heads, seq_len, head_dim = x.shape

        if self._num_kv_groups <= 1:
            return x

        x = x[:, :, None, :, :].expand(
            batch_size, num_kv_heads, self._num_kv_groups, seq_len, head_dim
        )
        return x.reshape(batch_size, -1, seq_len, head_dim)


class MLP(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        intermediate_size: int,
        model_tensors: safe_open,
    ):
        super().__init__()

        self._gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self._up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self._down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self._act_fn = nn.SiLU()

        # Load weights from model_tensors
        self._gate_proj.load_state_dict(
            {
                "weight": model_tensors.get_tensor(
                    f"model.layers.{layer_idx}.mlp.gate_proj.weight"
                )
            }
        )
        self._up_proj.load_state_dict(
            {
                "weight": model_tensors.get_tensor(
                    f"model.layers.{layer_idx}.mlp.up_proj.weight"
                )
            }
        )
        self._down_proj.load_state_dict(
            {
                "weight": model_tensors.get_tensor(
                    f"model.layers.{layer_idx}.mlp.down_proj.weight"
                )
            }
        )

    def forward(self, x: torch.Tensor):
        # x :: (batch_size, seq_len, hidden_size)
        # ret :: (batch_size, seq_len, hidden_size)
        return self._down_proj(self._act_fn(self._gate_proj(x)) * self._up_proj(x))


class LayerNorm(nn.Module):
    def __init__(
        self,
        name: str,
        hidden_size: int,
        eps: float,
        model_tensors: safe_open,
    ):
        super().__init__()

        self._hidden_size = hidden_size
        self._eps = eps

        self._weight = model_tensors.get_tensor(name)

    def forward(self, x: torch.Tensor):
        # x :: (batch_size, seq_len, hidden_size)
        # ret :: (batch_size, seq_len, hidden_size)
        input_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self._eps)
        return self._weight * x.to(input_dtype)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        layernorm_eps: float,
        model_tensors: safe_open,
        cache: Optional["Cache"] = None,
    ):
        super().__init__()

        self._layer_idx = layer_idx
        self._hidden_size = hidden_size

        if cache is not None:
            self._cache = cache[layer_idx]
        else:
            self._cache = None

        self._self_attn = SelfAttention(
            layer_idx,
            hidden_size,
            head_dim,
            num_attention_heads,
            num_kv_heads,
            layernorm_eps,
            model_tensors,
            cache=self._cache,
        )
        self._mlp = MLP(layer_idx, hidden_size, intermediate_size, model_tensors)
        self._input_layernorm = LayerNorm(
            f"model.layers.{layer_idx}.input_layernorm.weight",
            hidden_size,
            layernorm_eps,
            model_tensors,
        )
        self._post_att_layernorm = LayerNorm(
            f"model.layers.{layer_idx}.post_attention_layernorm.weight",
            hidden_size,
            layernorm_eps,
            model_tensors,
        )

    def forward(
        self, x: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ):
        # x :: (batch_size, seq_len, hidden_size)
        # ret :: (batch_size, seq_len, hidden_size)
        residual = x

        x = self._input_layernorm(x)
        # x :: (batch_size, seq_len, hidden_size)

        x = self._self_attn(x, position_embeddings)
        x = x + residual
        # x :: (batch_size, seq_len, hidden_size)

        residual = x
        x = self._mlp(self._post_att_layernorm(x))
        x = x + residual
        # x :: (batch_size, seq_len, hidden_size)

        return x
