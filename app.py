import streamlit as st


if __name__ == "__main__":
    # adding modules to path
    import sys
    sys.path.append("../extractive_summarizer")
    st.title("Text Summarizer 📝")
    st.write("This is a test statement")

    inp_text = st.text_input("Enter the text here")
    summarize_type = st.sidebar.selectbox("Summarization type", options=["Extractive", "Abstractive"])

    if summarize_type == "Extractive":
        from model_processors import Summarizer

        # init model
        model = Summarizer()
        summarized_text = model(inp_text, num_sentences=5)

        st.subheader("Summarized text")
        st.markdown(summarized_text)