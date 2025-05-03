from typing import Optional

import torch


class LayerCache:
    def __init__(self):
        self._cached_k = None
        self._cached_v = None
        # cached_k :: (batch_size, num_attention_heads, kv_seq_len, head_dim)
        # cached_v :: (batch_size, num_attention_heads, kv_seq_len, head_dim)

    @property
    def cached_seq_len(self) -> int:
        if self._cached_k is None:
            return 0
        return self._cached_k.shape[2]

    def update_and_concat(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._cached_k = LayerCache._concat_with_cached(self._cached_k, k)
        self._cached_v = LayerCache._concat_with_cached(self._cached_v, v)
        return self._cached_k, self._cached_v

    @staticmethod
    def _concat_with_cached(
        cached: Optional[torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        # cached :: (batch_size, num_attention_heads, kv_seq_len, head_dim)
        # x :: (batch_size, num_attention_heads, seq_len, head_dim)
        # ret :: (batch_size, num_attention_heads, kv_seq_len+seq_len, head_dim)
        if cached is None:
            return x
        return torch.cat((cached, x), dim=2)


class Cache:
    def __init__(self, num_layers: int):
        self._layer_caches = [LayerCache() for _ in range(num_layers)]

    def __len__(self) -> int:
        return len(self._layer_caches)

    def __getitem__(self, index: int) -> LayerCache:
        return self._layer_caches[index]
