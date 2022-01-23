import torch
import nltk
import validators
import streamlit as st
from nltk.tokenize import sent_tokenize
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration

# local modules
from extractive_summarizer.model_processors import Summarizer
from src.utils import clean_text, fetch_article_text
from src.abstractive_summarizer import (
    abstractive_summarizer,
    preprocess_text_for_abstractive_summarization,
)

# abstractive summarizer model
@st.cache()
def load_abs_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    return tokenizer, model


if __name__ == "__main__":
    # ---------------------------------
    # Main Application
    # ---------------------------------
    st.title("Text Summarizer 📝")
    summarize_type = st.sidebar.selectbox(
        "Summarization type", options=["Extractive", "Abstractive"]
    )
    nltk.download("punkt")

    inp_text = st.text_input("Enter text or a url here")

    is_url = validators.url(inp_text)
    if is_url:
        # complete text, chunks to summarize (list of sentences for long docs)
        text, clean_txt = fetch_article_text(url=inp_text)
    else:
        clean_txt = clean_text(inp_text)

    # view summarized text (expander)
    with st.expander("View input text"):
        if is_url:
            st.write(clean_txt[0])
        else:
            st.write(clean_txt)
    summarize = st.button("Summarize")

    # called on toggle button [summarize]
    if summarize:
        if summarize_type == "Extractive":
            if is_url:
                text_to_summarize = " ".join([txt for txt in clean_txt])
            # extractive summarizer

            with st.spinner(
                text="Creating extractive summary. This might take a few seconds ..."
            ):
                ext_model = Summarizer()
                summarized_text = ext_model(text_to_summarize, num_sentences=6)

        elif summarize_type == "Abstractive":
            with st.spinner(
                text="Creating abstractive summary. This might take a few seconds ..."
            ):
                text_to_summarize = clean_txt
                abs_tokenizer, abs_model = load_abs_model()
                if not is_url:
                    text_to_summarize = preprocess_text_for_abstractive_summarization(
                        tokenizer=abs_tokenizer, text=clean_txt
                    )
                summarized_text = abstractive_summarizer(
                    abs_tokenizer, abs_model, text_to_summarize
                )

        #         abs_tokenizer, abs_model = load_abs_model()
        #         summarized_text = abstractive_summarizer(
        #             abs_tokenizer, abs_model, text_to_summarize
        #         )
        # elif summarize_type == "Abstractive" and is_url:
        #     abs_url_summarizer = pipeline("summarization")
        #     tmp_sum = abs_url_summarizer(
        #         text_to_summarize, max_length=120, min_length=30, do_sample=False
        #     )
        #     summarized_text = " ".join([summ["summary_text"] for summ in tmp_sum])

        # final summarized output
        st.subheader("Summarized text")
        st.info(summarized_text)
