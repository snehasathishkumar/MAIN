import streamlit as st
from converter import converter
from extractive_summarizer import summarizer
st.set_page_config(layout='wide')
 
st.title('Extractive Summarizer')

if "uploaded_file_sum" not in st.session_state:
    st.session_state["uploaded_file_sum"] = None

if not st.session_state["uploaded_file_sum"]:
    st.session_state["uploaded_file_sum"] = st.file_uploader("Upload your file here")

if st.session_state["uploaded_file_sum"]:
    extracted_text = converter(st.session_state["uploaded_file_sum"])
    # print(summarizer(extracted_text))
    st.write(summarizer(extracted_text))



