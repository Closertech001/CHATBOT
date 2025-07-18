import streamlit as st
from rag_engine import get_chat_response

st.set_page_config(page_title="University Chatbot", layout="wide")
st.title("🎓 University Chatbot")

query = st.text_input("Ask a question about the university")

if query:
    with st.spinner("Thinking..."):
        response = get_chat_response(query)
        st.write(response)
