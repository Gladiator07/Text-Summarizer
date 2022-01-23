import torch
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# local modules
from extractive_summarizer.model_processors import Summarizer
from src.utils import clean_text
from src.abstractive_summarizer import abstractive_summarizer


# abstractive summarizer model
@st.cache()
def load_abs_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
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

    inp_text = st.text_input("Enter the text here")

    inp_text = clean_text(inp_text)

    # view summarized text (expander)
    with st.expander("View input text"):
        st.write(inp_text)

    summarize = st.button("Summarize")

    # called on toggle button [summarize]
    if summarize:
        if summarize_type == "Extractive":
            # extractive summarizer

            with st.spinner(
                text="Creating extractive summary. This might take a few seconds ..."
            ):
                ext_model = Summarizer()
                summarized_text = ext_model(inp_text, num_sentences=6)

        elif summarize_type == "Abstractive":
            with st.spinner(
                text="Creating abstractive summary. This might take a few seconds ..."
            ):
                abs_tokenizer, abs_model = load_abs_model()
                summarized_text = abstractive_summarizer(
                    abs_tokenizer, abs_model, inp_text
                )

        # final summarized output
        st.subheader("Summarized text")
        st.info(summarized_text)
