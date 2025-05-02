import os
import json
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import Qwen2Tokenizer

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

    def forward(self, x, position_ids):
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

        with torch.autocast(enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # freqs :: (batch_size, num_pos, head_dim / 2)

            emb = torch.cat((freqs, freqs), dim=-1)
            # emb :: (batch_size, num_pos, head_dim)

            cos = emb.cos()
            sin = emb.sin()
            # cos, sin :: (batch_size, num_pos, head_dim)

        return cos.to(x.dtype), sin.to(x.dtype)


class Model(nn.Module):
    def __init__(self, model_dir: "PathLike"):
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            self._config = json.load(f)

        self._tokenizer = Qwen2Tokenizer.from_pretrained(model_dir)
        self._model_tensors = safe_open(
            os.path.join(model_dir, "model.safetensors"), framework="pt"
        )

        # Load embedding weights
        self._embedding = Embedding(
            self._config["vocab_size"], self._config["hidden_size"], self._model_tensors
        )
        self._rope = RoPE(self._config["rope_theta"], self._config["head_dim"])

    def generate(self, prompt: str) -> str:
        input_ids = self._tokenizer(self._apply_chat_template(prompt), return_tensors="pt")[
            "input_ids"
        ]
        # input_ids :: (1, seq_len)

        logits = self(input_ids)
        # logits :: (1, 1, vocab_size)

        pass

    def forward(self, input_ids):
        # input_ids :: (batch_size, seq_len)

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        input_embedded = self._embedding(input_ids)
        # input_embedded :: (batch_size, seq_len, hidden_size)
        assert input_embedded.shape == (batch_size, seq_len, self._config["hidden_size"])

        position_ids = torch.arange(0, seq_len, device=input_embedded.device).unsqueeze(0)
        # position_ids :: (1, seq_len)
        position_embed = self._rope(input_embedded, position_ids)
        # position_embed :: (batch_size, seq_len, hidden_size)

        pass

    def _apply_chat_template(self, prompt: str) -> str:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
