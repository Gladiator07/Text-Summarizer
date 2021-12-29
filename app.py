import streamlit as st


if __name__ == "__main__":
    # adding modules to path
    import sys
    sys.path.append("../extractive_summarizer")
    st.title("Text Summarizer 📝")
    summarize_type = st.sidebar.selectbox("Summarization type", options=["Extractive", "Abstractive"])

    inp_text = st.text_input("Enter the text here")

    if summarize_type == "Extractive":
        from extractive_summarizer.model_processors import Summarizer

        # init model
        model = Summarizer()
        summarize = st.button("Summarize")
        if summarize:
            with st.expander("View input text"):
                st.write(inp_text)
            st.subheader("Summarized text")
            summarized_text = model(inp_text, num_sentences=5)
            st.info(summarized_text)
