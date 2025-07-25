# university_chatbot_app.py

import streamlit as st
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
import openai
from textblob import TextBlob
from symspellpy import SymSpell
from pathlib import Path
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load SentenceTransformer model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load knowledge base
with open("qa_dataset.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Precompute embeddings
questions = [item["question"] for item in qa_data]
question_embeddings = embed_model.encode(questions, convert_to_tensor=True).cpu().numpy()

# Build FAISS index
index = faiss.IndexFlatL2(question_embeddings.shape[1])
index.add(question_embeddings)

# Load SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = Path("frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, 0, 1)

# Abbreviations, Synonyms, Plurals
ABBREVIATIONS = {
    "siwes": "student industrial work experience scheme",
    "dept": "department",
    "dept.": "department",
    "cuab": "crescent university",
    "ict": "information and communication technology",
    "cohes": "college of health sciences",
    "coes": "college of environmental sciences",
    "conas": "college of natural and applied sciences",
    "casmas": "college of arts social and management sciences",
    "cicot": "college of information and communication technology",
    "bacolaw": "bola ajibola college of law"
}

SYNONYMS = {
    "school fees": "tuition",
    "fees": "tuition",
    "accommodation": "hostel",
    "freshers": "new students",
    "returning students": "old students",
    "lecturers": "academic staff",
    "teachers": "lecturers",
    "professors": "academic staff",
    "registration": "enrollment",
    "course list": "courses",
    "head of department": "hod",
    "contact": "phone number",
    "cost": "price",
    "amount": "fee",
    "procedure": "process"
}

PLURAL_REPLACEMENTS = {
    "students": "student",
    "lectures": "lecture",
    "departments": "department",
    "fees": "fee",
    "courses": "course",
    "requirements": "requirement",
    "projects": "project",
    "contacts": "contact",
    "exams": "exam",
    "subjects": "subject"
}

def normalize_input(text):
    text = text.lower()
    # Abbreviation expansion
    for abbr, full in ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", full, text)
    # Synonym replacement
    for syn, base in SYNONYMS.items():
        text = re.sub(rf"\b{re.escape(syn)}\b", base, text)
    # Plural to singular
    for plural, singular in PLURAL_REPLACEMENTS.items():
        text = re.sub(rf"\b{plural}\b", singular, text)
    return text

def correct_spelling(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

def detect_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    return "neutral"

def search_answer(user_input):
    query = normalize_input(correct_spelling(user_input))
    query_embedding = embed_model.encode([query], convert_to_tensor=True).cpu().numpy()
    D, I = index.search(query_embedding, k=1)
    top_score = D[0][0]
    top_match = qa_data[I[0][0]]
    return top_match if top_score < 0.5 else None

def fallback_gpt(user_input, history=[]):
    prompt = f"""
You are CrescentBot, a friendly and helpful assistant for a university. Be conversational, empathetic, and informative.
User: {user_input}
Assistant:
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful university assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return response.choices[0].message["content"].strip()

# Streamlit UI
st.set_page_config(page_title="🎓 CrescentBot - University Assistant", layout="wide")
st.title("🎓 CrescentBot - University Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me anything about Crescent University...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    sentiment = detect_sentiment(user_input)
    response = search_answer(user_input)

    if response:
        answer = response["answer"]
    else:
        answer = fallback_gpt(user_input, st.session_state.chat_history)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
