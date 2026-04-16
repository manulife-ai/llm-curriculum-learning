import torch


def generate_predictions(model, tokenizer, prompt_texts: list[str], generation_config: dict, device: str) -> list[str]:
    predictions = []
    for prompt_text in prompt_texts:
        encoded = tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=generation_config["max_new_tokens"],
                num_beams=generation_config["num_beams"],
                do_sample=generation_config["do_sample"],
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        output_tokens = generated[0][prompt_len:]
        predictions.append(tokenizer.decode(output_tokens, skip_special_tokens=True).strip())
    return predictions
