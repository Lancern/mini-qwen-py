import torch

from miniqwen.model import Model


device = torch.device("cpu")
m = Model("/mnt/d/LLM/Qwen3-0.6B", use_cache=True).to(device)

while True:
    prompt = input("> ").strip()
    if prompt == ".exit" or prompt == ".quit":
        break

    for token in m.generate(prompt, max_generate_len=1000):
        print(token, end="", flush=True)

    print()
