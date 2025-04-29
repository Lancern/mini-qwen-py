import os
from typing import TYPE_CHECKING

from transformers import Qwen2Tokenizer

if TYPE_CHECKING:
    from os import PathLike


class Model:
    def __init__(self, model_dir: "PathLike"):
        self._tokenizer = Qwen2Tokenizer(
            vocab_file=os.path.join(model_dir, "vocab.json"),
            merges_file=os.path.join(model_dir, "merges.txt"),
        )

    def generate(self, prompt: str) -> str:
        pass
