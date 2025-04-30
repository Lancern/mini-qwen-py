from typing import TYPE_CHECKING

from transformers import Qwen2Tokenizer

if TYPE_CHECKING:
    from os import PathLike


class Model:
    def __init__(self, model_dir: "PathLike"):
        self._tokenizer = Qwen2Tokenizer.from_pretrained(model_dir)

    def generate(self, prompt: str) -> str:
        input = self._tokenizer(self._apply_chat_template(prompt))["input_ids"]
        pass

    def _apply_chat_template(self, prompt: str) -> str:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
