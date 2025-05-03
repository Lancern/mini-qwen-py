from time import time

import torch

from miniqwen.model import Model


device = torch.device("cpu")
m = Model("/mnt/d/LLM/Qwen3-0.6B").to(device)

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
