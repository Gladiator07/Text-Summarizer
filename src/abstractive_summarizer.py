import torch
from nltk.tokenize import sent_tokenize
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


def preprocess_text_for_abstractive_summarization(tokenizer, text):
    sentences = sent_tokenize(text)

    # initialize
    length = 0
    chunk = ""
    chunks = []
    count = -1
    for sentence in sentences:
        count += 1
        combined_length = (
            len(tokenizer.tokenize(sentence)) + length
        )  # add the no. of sentence tokens to the length counter

        if combined_length <= tokenizer.max_len_single_sentence:  # if it doesn't exceed
            chunk += sentence + " "  # add the sentence to the chunk
            length = combined_length  # update the length counter

            # if it is the last sentence
            if count == len(sentences) - 1:
                chunks.append(chunk.strip())  # save the chunk

        else:
            chunks.append(chunk.strip())  # save the chunk

            # reset
            length = 0
            chunk = ""

            # take care of the overflow sentence
            chunk += sentence + " "
            length = len(tokenizer.tokenize(sentence))

    return chunks
