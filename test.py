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

    st.write(uploaded_file.type)
    docx_text = docx2txt.process(uploaded_file)

    st.write(docx_text)
