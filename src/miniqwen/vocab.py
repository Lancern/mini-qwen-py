from dataclasses import dataclass
import json
import logging
from typing import Dict, Optional, List, overload, TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike


@dataclass
class TokenInfo:
    content: str
    value: int
    lstrip: bool = False
    normalized: bool = False
    rstrip: bool = False
    single_word: bool = False
    special: bool = False

    def __eq__(self, other):
        return isinstance(other, TokenInfo) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return self.content

    def __int__(self):
        return self.value


class Vocabulary:
    def __init__(self, vocab_size: int):
        self._tokens: List[TokenInfo] = [None] * vocab_size
        self._txt2tk: Dict[str, int] = {}

    def load_vocab_file(self, file: "PathLike"):
        with open(file, "r", encoding="utf-8") as f:
            raw_txt2tk: Dict[str, int] = json.load(f)

        for text, tk in raw_txt2tk.items():
            info = TokenInfo(text, tk)
            self.add(info)

        logging.info("%d tokens loaded from the vocabulary file", len(raw_txt2tk))

    def add(self, info: TokenInfo):
        self._tokens[info.value] = info
        self._txt2tk[info.content] = info.value

    @overload
    def get_token(self, text: str) -> Optional[int]: ...
    @overload
    def get_token(self, text: str, default: int) -> int: ...

    def get_token(self, text, default=None):
        """
        Get the token corresponding to the specified text.
        """
        return self._txt2tk.get(text, default=default)

    @overload
    def get_text(self, token: int) -> Optional[str]: ...
    @overload
    def get_text(self, token: int, default: str) -> str: ...

    def get_text(self, token, default=None):
        """
        Get the text corresponding to the specified token.
        """

        info = self.get_info(token)
        if info is None:
            return default
        return info.content

    def get_info(self, token: int) -> Optional[TokenInfo]:
        """
        Get information about the given token.
        """

        if token < 0 or token >= len(self._tokens):
            return None
        return self._tokens[token]
