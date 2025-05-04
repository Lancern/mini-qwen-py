import math
import os
import json
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from safetensors import safe_open

from .cache import Cache
from .tokenizer import Tokenizer
from .transformer import DecoderLayer, LayerNorm

if TYPE_CHECKING:
    from collections.abc import Generator
    from os import PathLike


class RoPE(nn.Module):
    def __init__(self, theta: int | float, head_dim: int):
        super().__init__()
        assert head_dim % 2 == 0

        self._theta = theta

        # inv_freq :: (head_dim / 2)
        # inv_freq[i] = theta ** (-2i / head_dim)
        self.inv_freq = nn.Buffer(
            1.0 / (theta ** (torch.arange(0, head_dim, 2) / head_dim)),
            persistent=False,
        )

    def __call__(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x :: (batch_size, seq_len, head_dim)
        # position_ids :: (batch_size, num_pos)
        # ret :: tuple
        #   => ret[0] :: (batch_size, num_pos, head_dim)
        #   => ret[1] :: (batch_size, num_pos, head_dim)

        batch_size = x.shape[0]

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(batch_size, -1, 1)
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
    def __init__(
        self,
        vocab_size: int,
        rope_theta: float,
        hidden_size: int,
        num_hidden_layers: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
        use_cache: bool = True,
    ):
        super().__init__()

        self._use_cache = use_cache
        if use_cache:
            self._cache = Cache(num_hidden_layers)
        else:
            self._cache = None

        self._rope = RoPE(rope_theta, head_dim)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    idx,
                    hidden_size,
                    head_dim,
                    num_attention_heads,
                    num_key_value_heads,
                    intermediate_size,
                    rms_norm_eps,
                    cache=self._cache,
                )
                for idx in range(num_hidden_layers)
            ]
        )
        self.norm = LayerNorm(hidden_size, rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids :: (batch_size, seq_len)
        # ret :: (batch_size, seq_len, vocab_size)

        hidden_state = self.embed_tokens(input_ids)
        # hidden_state :: (batch_size, seq_len, hidden_size)

        if self._use_cache:
            generated_seq_len = self._cache[0].cached_seq_len
        else:
            generated_seq_len = 0

        seq_len = input_ids.shape[1]
        position_ids = torch.arange(
            generated_seq_len, generated_seq_len + seq_len, device=hidden_state.device
        ).unsqueeze(0)
        # position_ids :: (1, seq_len)
        position_embeddings = self._rope(hidden_state, position_ids)
        # position_embeddings :: tuple
        #   => position_embeddings[0] :: (batch_size, seq_len, hidden_size)
        #   => position_embeddings[1] :: (batch_size, seq_len, hidden_size)

        for decoder_layer in self.layers:
            hidden_state = decoder_layer(hidden_state, position_embeddings)
            # hidden_state :: (batch_size, seq_len, hidden_size)

        return self.norm(hidden_state)


class MiniQwen(nn.Module):
    @staticmethod
    def from_pretrained(model_dir: "PathLike") -> "MiniQwen":
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        with open(os.path.join(model_dir, "generation_config.json"), "r", encoding="utf-8") as f:
            generation_config = json.load(f)

        tokenizer = Tokenizer(model_dir)
        module = MiniQwen(
            tokenizer=tokenizer,
            vocab_size=config["vocab_size"],
            rope_theta=config["rope_theta"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            head_dim=config["head_dim"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            intermediate_size=config["intermediate_size"],
            rms_norm_eps=config["rms_norm_eps"],
            temperature=generation_config["temperature"],
            top_k=generation_config["top_k"],
            top_p=generation_config["top_p"],
        )

        with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
            sdict = {key: f.get_tensor(key) for key in f.keys()}
            module.load_state_dict(sdict)

        return module

    def __init__(
        self,
        tokenizer: Tokenizer,
        vocab_size: int,
        rope_theta: float,
        hidden_size: int,
        num_hidden_layers: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
        temperature: float,
        top_k: int,
        top_p: float,
        use_cache: bool = True,
    ):
        super().__init__()

        self._tokenizer = tokenizer
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._use_cache = use_cache

        self.model = Model(
            vocab_size,
            rope_theta,
            hidden_size,
            num_hidden_layers,
            head_dim,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            rms_norm_eps,
            use_cache,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def generate(self, prompt: str, max_generate_len: int = 1000) -> "Generator[str]":
        device = self.lm_head.weight.device

        input_ids = self._tokenizer.tokenize_for_chat(prompt).to(device)
        # input_ids :: (1, seq_len)

        for _ in range(max_generate_len):
            output_id = int(self.generate_once(input_ids).squeeze())
            if output_id == self._tokenizer.eos:
                break

            output_token = self._tokenizer.decode(output_id)
            yield output_token

            if self._use_cache:
                input_ids = torch.tensor([[output_id]], device=device)
            else:
                input_ids = torch.cat(
                    (input_ids, torch.tensor([[output_id]], device=device)), dim=1
                )

    def generate_once(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids :: (batch_size, seq_len)
        # ret :: (batch_size, 1)

        logits = self(input_ids)
        # logits :: (batch_size, seq_len, vocab_size)

        logits = logits[:, -1, :].squeeze(dim=1) / self._temperature
        # logits :: (batch_size, vocab_size)

        probs = self._apply_top_p(self._apply_top_k(logits)).softmax(dim=-1)
        # probs :: (batch_size, vocab_size)

        return torch.multinomial(probs, num_samples=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids :: (batch_size, seq_len)
        return self.lm_head(self.model(input_ids))

    def _apply_top_k(self, logits: torch.Tensor) -> torch.Tensor:
        # logits :: (batch_size, vocab_size)

        min_top_k_prob = torch.topk(logits, self._top_k)[0][..., -1, None]
        # min_top_k_prob :: (batch_size, 1)
        mask = logits < min_top_k_prob
        # mask :: (batch_size, vocab_size)

        return logits.masked_fill(mask, -math.inf)

    def _apply_top_p(self, logits: torch.Tensor) -> torch.Tensor:
        # logits :: (batch_size, vocab_size)

        sorted_logits, sorted_indecies = torch.sort(logits, descending=False)
        # sorted_logits :: (batch_size, vocab_size)
        # sorted_indecies :: (batch_size, vocab_size)

        cum_prob = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # cum_prob :: (batch_size, vocab_size)

        reset_mask_sorted = cum_prob <= (1 - self._top_p)
        # Make sure the logit giving the highest probability is always kept
        reset_mask_sorted[..., -1:] = 0
        # reset_mask :: (batch_size, vocab_size)

        reset_mask = reset_mask_sorted.scatter(1, sorted_indecies, reset_mask_sorted)
        return logits.masked_fill(reset_mask, -math.inf)
