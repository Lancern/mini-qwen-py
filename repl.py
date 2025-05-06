from argparse import ArgumentParser
from time import time
import os

import torch

from miniqwen.model import MiniQwen


parser = ArgumentParser(description="MiniQwen REPL")
parser.add_argument("-d", "--device", default="cpu", help="Device to use")
parser.add_argument("model_dir", help="Path to the model directory")

args = parser.parse_args()
model_dir = args.model_dir or os.getcwd()
device = torch.device(args.device)

m = MiniQwen.from_pretrained(model_dir).to(device)

while True:
    prompt = input("> ").strip()
    if prompt == ".exit" or prompt == ".quit":
        break

    num_tokens_generated = 0
    start_time = time()

    for token in m.generate(prompt, max_generate_len=1000):
        print(token, end="", flush=True)
        num_tokens_generated += 1

    end_time = time()
    print()

    elapsed_sec = end_time - start_time
    tokens_per_sec = num_tokens_generated / elapsed_sec
    print(f"--- {num_tokens_generated} tokens generated in {elapsed_sec:.2f} seconds")
    print(f"--- {tokens_per_sec:.2f} tokens/sec")
