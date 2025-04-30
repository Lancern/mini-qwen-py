import logging

from transformers import Qwen2Tokenizer, Qwen3ForCausalLM


logging.basicConfig(level=logging.INFO)
model_dir = "/mnt/d/LLM/Qwen3-0.6B"

logging.info("Loading tokenizer ...")
tokenizer = Qwen2Tokenizer.from_pretrained(model_dir)
logging.info("Loading model ...")
model = Qwen3ForCausalLM.from_pretrained(model_dir)

while True:
    print("> ", end="", flush=True)
    prompt = input().strip()

    if prompt == ".exit" or prompt == ".quit":
        break

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    logging.info("========== Chat template ==========")
    logging.info(text)

    model_inputs = tokenizer([text], return_tensors="pt")
    logging.info("========== Model input ==========")
    logging.info(model_inputs)

    generated_ids = model.generate(**model_inputs)
    logging.info("========== Model output ==========")
    logging.info(generated_ids)

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    logging.info("========== Decoded model output ==========")
    logging.info(content)
