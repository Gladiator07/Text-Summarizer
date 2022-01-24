import docx2txt
import streamlit as st
from io import StringIO
from PyPDF2 import PdfFileReader


def read_pdf(file):
    pdfReader = PdfFileReader(file)
    count = pdfReader.numPages
    all_page_text = ""
    for i in range(count):
        page = pdfReader.getPage(i)
        all_page_text += page.extractText()

    return all_page_text


if __name__ == "__main__":
    st.header("Testing file uploads")

    uploaded_file = st.file_uploader("Upload a file here")
    st.markdown(
        "<h3 style='text-align: center; color: red;'>OR</h3>",
        unsafe_allow_html=True,
    )
