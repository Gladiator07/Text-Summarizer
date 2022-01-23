import torch
from nltk.tokenize import sent_tokenize
from transformers import T5Tokenizer


def abstractive_summarizer(tokenizer, model, text):
    # inputs to the model
    inputs = [
        tokenizer.encode(f"summarize: {chunk}", return_tensors="pt") for chunk in text
    ]
    abs_summarized_text = []
    for input in inputs:
        output = model.generate(**input)
        tmp_sum = tokenizer.decode(*output, skip_special_tokens=True)
        abs_summarized_text.append(tmp_sum)

    abs_summarized_text = " ".join([summ for summ in abs_summarized_text])
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
