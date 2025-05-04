from dataclasses import dataclass
import itertools
import json
import os
import regex
from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from os import PathLike


@dataclass
class TokenInfo:
    content: str
    id: int
    is_special: bool


class Vocab:
    def __init__(self, tokens: list[TokenInfo]):
        self._tokens = tokens
        self._tokens.sort(key=lambda ti: ti.id)

        # Make sure we don't have missing token ids in the middle
        for i, ti in enumerate(self._tokens):
            if i != ti.id:
                raise RuntimeError(f"Missing token id {i}")

        self._content_to_id = {ti.content: ti.id for ti in self._tokens}

        # Identify all special tokens
        self._special_tokens = [ti for ti in self._tokens if ti.is_special]

    @property
    def special_tokens(self) -> list[TokenInfo]:
        return self._special_tokens

    def get_token_id(self, content: str) -> Optional[int]:
        return self._content_to_id.get(content)

    def get_token_info(self, token_id: int) -> Optional[TokenInfo]:
        if token_id >= len(self._tokens):
            return None
        return self._tokens[token_id]


class Tokenizer:
    def __init__(self, model_dir: "PathLike"):
        with open(
            os.path.join(model_dir, "tokenizer.json"), "r", encoding="utf-8"
        ) as f:
            tokenizer_config = json.load(f)

        raw_vocab: dict[str, int] = tokenizer_config["model"]["vocab"]
        tokens = [
            TokenInfo(content=k, id=v, is_special=False) for k, v in raw_vocab.items()
        ]

        for add_token in tokenizer_config["added_tokens"]:
            tokens.append(
                TokenInfo(
                    content=add_token["content"],
                    id=add_token["id"],
                    is_special=True,
                )
            )

        self._vocab = Vocab(tokens)
        self._eos = self._vocab.get_token_id("<|im_end|>")

        self._byte_codec = ByteCodec()

        merge_pairs: list[tuple[str, str]] = list(
            map(tuple, tokenizer_config["model"]["merges"])
        )
        self._merge_ranks = dict(zip(merge_pairs, range(len(merge_pairs))))
        self._max_merge_rank = len(merge_pairs)

        pretokenize_regex = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        self._pretokenize_re = regex.compile(pretokenize_regex)

    @property
    def eos(self) -> int:
        return self._eos

    def tokenize(
        self, text: str, identify_special=True, return_tensor=True
    ) -> list[int] | torch.Tensor:
        fragments: list[str | int] = [text]
        if identify_special:
            self._identify_specials(fragments)

        i = 0
        while i < len(fragments):
            frag = fragments[i]
            if not isinstance(frag, str):
                i += 1
                continue

            tokens: list[str] = []
            for word in regex.findall(self._pretokenize_re, frag):
                tokens += self._bpe_tokenize(self._byte_codec.encode(word))

            fragments[i : i + 1] = [self._vocab.get_token_id(t) for t in tokens]
            i += len(tokens)

        if return_tensor:
            return torch.tensor([fragments])
        return fragments

    def tokenize_for_chat(self, prompt: str) -> torch.Tensor:
        template = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return self.tokenize(template)

    def decode(self, token_id: int, skip_special: bool = True) -> str:
        ti = self._vocab.get_token_info(token_id)
        if ti is None:
            return ""
        if skip_special and ti.is_special:
            return ""

        return self._byte_codec.decode(ti.content)

    def _identify_specials(self, fragments: list[str | int]):
        for special_ti in self._vocab.special_tokens:
            i = 0
            while i < len(fragments):
                frag = fragments[i]
                if not isinstance(frag, str):
                    i += 1
                    continue
                subfrag = self._identify_special(frag, special_ti)
                fragments[i : i + 1] = subfrag
                i += len(subfrag)

    def _identify_special(self, text: str, special_ti: TokenInfo) -> list[str | int]:
        result = []
        i = 0
        while i < len(text):
            j = text.find(special_ti.content, i)
            if j == -1:
                result.append(text[i:])
                break
            if j > i:
                result.append(text[i:j])
            result.append(special_ti.id)
            i = j + len(special_ti.content)
        return result

    def _bpe_tokenize(self, word: str) -> tuple[str, ...]:
        chunks = tuple(word)
        chunk_pairs = set(itertools.pairwise(chunks))

        while len(chunk_pairs) > 0:
            # Choose the chunk pair with the lowest merge rank and merge it
            bigram = min(
                chunk_pairs,
                key=lambda p: self._merge_ranks.get(p, self._max_merge_rank),
            )
            if bigram not in self._merge_ranks:
                break

            chunks = self._bpe_update_chunks(chunks, bigram)
            chunk_pairs = set(itertools.pairwise(chunks))

        return chunks

    def _bpe_update_chunks(
        self, chunks: tuple[str, ...], bigram: tuple[str, str]
    ) -> tuple[str, ...]:
        first, second = bigram
        new_chunks = []
        i = 0
        while i < len(chunks):
            try:
                j = chunks.index(first, i)
            except ValueError:
                new_chunks.extend(chunks[i:])
                break
            else:
                new_chunks.extend(chunks[i:j])

            if j + 1 < len(chunks) and chunks[j + 1] == second:
                new_chunks.append(first + second)
                i = j + 2
            else:
                new_chunks.append(first)
                i = j + 1

        return tuple(new_chunks)


class ByteCodec:
    def __init__(self):
        # Find all printable characters in 0x00~0xFF
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = list(map(chr, bs))

        # For a non-printable character c in 0x00~0xFF, map it to 0x100+c, which is a valid Unicode
        # printable character.
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(chr(2**8 + n))
                n += 1

        self._b2c: dict[int, str] = dict(zip(bs, cs))
        self._c2b: dict[str, int] = {v: k for k, v in self._b2c.items()}

    def encode(self, raw: str) -> str:
        return "".join((self._b2c[b] for b in raw.encode("utf-8")))

    def decode(self, encoded: str) -> str:
        return bytes((self._c2b[c] for c in encoded)).decode("utf-8")
