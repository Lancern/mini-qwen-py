import os
import json
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import Qwen2Tokenizer

from .transformer import DecoderLayer, LayerNorm

if TYPE_CHECKING:
    from os import PathLike


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, model_tensors: safe_open):
        super().__init__()

        self._embed = nn.Embedding(vocab_size, hidden_size)
        self._embed.load_state_dict(
            {"weight": model_tensors.get_tensor("model.embed_tokens.weight")}
        )

    def forward(self, x):
        # x :: (batch_size, seq_len)
        # ret :: (batch_size, seq_len, hidden_size)
        return self._embed(x)


class RoPE(nn.Module):
    def __init__(self, theta: int | float, head_dim: int):
        assert head_dim % 2 == 0
        super().__init__()

        self._theta = theta

        # inv_freq :: (head_dim / 2)
        # inv_freq[i] = theta ** (-2i / head_dim)
        self._inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x :: (batch_size, seq_len, head_dim)
        # position_ids :: (batch_size, num_pos)
        # ret :: tuple
        #   => ret[0] :: (batch_size, num_pos, head_dim)
        #   => ret[1] :: (batch_size, num_pos, head_dim)

        batch_size = x.shape[0]

        inv_freq_expanded = (
            self._inv_freq[None, :, None].float().expand(batch_size, -1, 1)
        )
        # inv_freq_expanded :: (batch_size, head_dim / 2, 1)

        position_ids_expanded = position_ids[:, None, :].float()
        # position_ids_expanded :: (batch_size, 1, num_pos)

        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            # freqs :: (batch_size, num_pos, head_dim / 2)

            emb = torch.cat((freqs, freqs), dim=-1)
            # emb :: (batch_size, num_pos, head_dim)

            cos = emb.cos()
            sin = emb.sin()
            # cos, sin :: (batch_size, num_pos, head_dim)

        return cos.to(x.dtype), sin.to(x.dtype)


class Model(nn.Module):
    def __init__(self, model_dir: "PathLike"):
        super().__init__()

        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            self._config = json.load(f)

        self._vocab_size: int = self._config["vocab_size"]
        self._rope_theta: float = self._config["rope_theta"]
        self._hidden_size: int = self._config["hidden_size"]
        self._head_dim: int = self._config["head_dim"]
        self._num_attention_heads: int = self._config["num_attention_heads"]
        self._num_key_value_heads: int = self._config["num_key_value_heads"]
        self._intermediate_size: int = self._config["intermediate_size"]
        self._rms_norm_eps: float = self._config["rms_norm_eps"]
        self._num_hidden_layers: int = self._config["num_hidden_layers"]

        self._tokenizer = Qwen2Tokenizer.from_pretrained(model_dir)
        self._model_tensors = safe_open(
            os.path.join(model_dir, "model.safetensors"), framework="pt"
        )

        self._embedding = Embedding(
            self._vocab_size, self._hidden_size, self._model_tensors
        )
        self._rope = RoPE(self._rope_theta, self._head_dim)
        self._decoder_layers = [
            DecoderLayer(
                idx,
                self._hidden_size,
                self._head_dim,
                self._num_attention_heads,
                self._num_key_value_heads,
                self._intermediate_size,
                self._rms_norm_eps,
                self._model_tensors,
            )
            for idx in range(self._num_hidden_layers)
        ]
        self._norm = LayerNorm(
            "model.norm.weight",
            self._hidden_size,
            self._rms_norm_eps,
            self._model_tensors,
        )

        self._lm_head = nn.Linear(self._hidden_size, self._vocab_size, bias=False)
        self._lm_head.load_state_dict(
            {"weight": self._model_tensors.get_tensor("lm_head.weight")}
        )

    def generate(self, prompt: str) -> str:
        input_ids = self._tokenizer(
            self._apply_chat_template(prompt), return_tensors="pt"
        )["input_ids"]
        # input_ids :: (1, seq_len)

        logits = self(input_ids)
        # logits :: (1, seq_len, vocab_size)
        print("input_ids shape:", input_ids.shape)
        print("logits shape:", logits.shape)

        pass

    def forward(self, input_ids: torch.Tensor):
        # input_ids :: (batch_size, seq_len)
        # ret :: (batch_size, seq_len, vocab_size)

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        hidden_state = self._embedding(input_ids)
        # input_embedded :: (batch_size, seq_len, hidden_size)
        assert hidden_state.shape == (
            batch_size,
            seq_len,
            self._config["hidden_size"],
        )

        position_ids = torch.arange(0, seq_len, device=hidden_state.device).unsqueeze(0)
        # position_ids :: (1, seq_len)
        position_embeddings = self._rope(hidden_state, position_ids)
        # position_embeddings :: tuple
        #   => position_embeddings[0] :: (batch_size, seq_len, hidden_size)
        #   => position_embeddings[1] :: (batch_size, seq_len, hidden_size)

        for decoder_layer in self._decoder_layers:
            hidden_state = decoder_layer(hidden_state, position_embeddings)

        hidden_state = self._norm(hidden_state)
        # hidden_state :: (batch_size, seq_len, hidden_size)

        return self._lm_head(hidden_state)

    def _apply_chat_template(self, prompt: str) -> str:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
