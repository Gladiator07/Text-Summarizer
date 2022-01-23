import torch
from transformers import T5Tokenizer


def abstractive_summarizer(tokenizer, model, text):
    device = torch.device("cpu")
    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_text = "summarize: " + preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_text, return_tensors="pt").to(device)

    # summmarize
    summary_ids = model.generate(
        tokenized_text,
        num_beams=4,
        no_repeat_ngram_size=2,
        min_length=30,
        max_length=300,
        early_stopping=True,
    )
    abs_summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return abs_summarized_text
