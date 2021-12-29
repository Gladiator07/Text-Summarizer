import torch
import streamlit as st
from extractive_summarizer.model_processors import Summarizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

def abstractive_summarizer(text : str):

    model = T5ForConditionalGeneration.from_pretrained('t5-large')
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    device = torch.device('cpu')

    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_text = "summarize: " + preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_text, return_tensors="pt").to(device)

    # summmarize 
    summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=100,
                                        early_stopping=True)
    abs_summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return abs_summarized_text

if __name__ == "__main__":

    st.title("Text Summarizer 📝")
    summarize_type = st.sidebar.selectbox("Summarization type", options=["Extractive", "Abstractive"])

    inp_text = st.text_input("Enter the text here")

    # view summarized text (expander)
    with st.expander("View input text"):
        st.write(inp_text)

    summarize = st.button("Summarize")

    # called on toggle button [summarize]
    if summarize:
        if summarize_type == "Extractive":
            # extractive summarizer
            # init model
            model = Summarizer()
            summarized_text = model(inp_text, num_sentences=5)
      
        elif summarize_type == "Abstractive":
            summarized_text = abstractive_summarizer(inp_text)

        # final summarized output    
        st.subheader("Summarized text")
        st.info(summarized_text)
