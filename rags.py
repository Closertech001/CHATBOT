import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import random
import re
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources
import json
import openai
import os
import hashlib
import uuid  # ✅ Added for unique key generation

# --- API Key Setup ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- SymSpell Setup ---
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# --- Abbreviations and Department Mapping ---
abbreviations = {
    "u": "you", "r": "are", "ur": "your", "ow": "how", "pls": "please", "plz": "please",
    "tmrw": "tomorrow", "cn": "can", "wat": "what", "cud": "could", "shud": "should",
    "wud": "would", "abt": "about", "bcz": "because", "btw": "between", "asap": "as soon as possible",
    "idk": "i don't know", "imo": "in my opinion", "msg": "message", "doc": "document", "d": "the",
    "yr": "year", "sem": "semester", "dept": "department", "admsn": "admission",
    "cresnt": "crescent", "uni": "university", "clg": "college", "sch": "school",
    "info": "information", "l": "level", "CSC": "Computer Science", "ECO": "Economics with Operations Research",
    "PHY": "Physics", "STAT": "Statistics", "1st": "First", "2nd": "Second"
}

department_map = {
    "GST": "General Studies", "MTH": "Mathematics", "PHY": "Physics", "STA": "Statistics",
    "COS": "Computer Science", "CUAB-CSC": "Computer Science", "CSC": "Computer Science",
    "IFT": "Computer Science", "SEN": "Software Engineering", "ENT": "Entrepreneurship",
    "CYB": "Cybersecurity", "ICT": "Information and Communication Technology",
    "DTS": "Data Science", "CUAB-CPS": "Computer Science", "CUAB-ECO": "Economics with Operations Research",
    "ECO": "Economics with Operations Research", "SSC": "Social Sciences", "CUAB-BCO": "Economics with Operations Research",
    "LIB": "Library Studies", "LAW": "Law (BACOLAW)", "GNS": "General Studies", "ENG": "English",
    "SOS": "Sociology", "PIS": "Political Science", "CPS": "Computer Science",
    "LPI": "Law (BACOLAW)", "ICL": "Law (BACOLAW)", "LPB": "Law (BACOLAW)", "TPT": "Law (BACOLAW)",
    "FAC": "Agricultural Sciences", "ANA": "Anatomy", "BIO": "Biological Sciences",
    "CHM": "Chemical Sciences", "CUAB-BCH": "Biochemistry", "CUAB": "Crescent University - General"
}

# --- Text Preprocessing ---
def normalize_text(text):
    text = re.sub(r'([^a-zA-Z0-9\s])', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text

def preprocess_text(text):
    text = normalize_text(text)
    words = text.split()
    expanded = [abbreviations.get(word.lower(), word) for word in words]
    corrected = []
    for word in expanded:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected.append(suggestions[0].term if suggestions else word)
    return ' '.join(corrected)

def extract_prefix(code):
    match = re.match(r"([A-Z\-]+)", code)
    return match.group(1) if match else None

# --- Model & Data Load ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    with open("qa_dataset.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    rag_data = []
    for entry in raw_data:
        question = entry.get("question", "").strip()
        answer = entry.get("answer", "").strip()
        department = entry.get("department", "").strip()
        level = entry.get("level", "").strip()
        semester = entry.get("semester", "").strip()
        faculty = entry.get("faculty", "").strip()

        if question and answer:
            rag_data.append({
                "text": f"Q: {question}\nA: {answer}",
                "question": question,
                "answer": answer,
                "department": department,
                "level": level,
                "semester": semester,
                "faculty": faculty
            })

    return pd.DataFrame(rag_data)

@st.cache_data
def compute_question_embeddings(questions: list):
    model = load_model()
    return model.encode(questions, convert_to_tensor=True)

# --- OpenAI fallback ---
def fallback_openai(user_input, context_qa=None):
    system_prompt = (
        "You are a helpful assistant specialized in Crescent University information. "
        "If you don't know an answer, politely say so and refer to university resources."
    )
    messages = [{"role": "system", "content": system_prompt}]

    if context_qa:
        context_text = f"Here is some relevant university information:\nQ: {context_qa['question']}\nA: {context_qa['answer']}\n\n"
        user_message = context_text + "Answer this question: " + user_input
    else:
        user_message = user_input

    messages.append({"role": "user", "content": user_message})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message["content"].strip()
    except Exception:
        return "Sorry, I couldn't reach the server. Try again later."

# --- Response Finder ---
def find_response(user_input, dataset, embeddings, threshold=0.4):
    model = load_model()
    user_input_clean = preprocess_text(user_input)

    if embeddings is None or len(dataset) == 0:
        return "No matching data found for your filters.", None, 0.0, []

    greetings = ["hi", "hello", "hey", "hi there", "greetings", "how are you",
                 "how are you doing", "how's it going", "can we talk?",
                 "can we have a conversation?", "okay", "i'm fine", "i am fine"]
    if user_input_clean.lower() in greetings:
        return random.choice(["Hello!", "Hi there!", "Hey!", "Greetings!","I'm doing well, thank you!", 
                              "Sure pal", "I'm fine, thank you", "Hi! How can I help you?", 
                              "Hello! Ask me anything about Crescent University."]), None, 1.0, []

    user_embedding = model.encode(user_input_clean, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_embedding, embeddings)[0]
    top_scores, top_indices = torch.topk(cos_scores, k=5)

    top_score = top_scores[0].item()
    top_index = top_indices[0].item()

    if top_score < threshold:
        context_qa = {
            "question": dataset.iloc[top_index]["question"],
            "answer": dataset.iloc[top_index]["answer"]
        }
        gpt_reply = fallback_openai(user_input, context_qa)
        return gpt_reply, None, top_score, []

    response = dataset.iloc[top_index]["answer"]
    question = dataset.iloc[top_index]["question"]
    related_questions = [dataset.iloc[i.item()]["question"] for i in top_indices[1:]]

    match = re.search(r"\b([A-Z]{2,}-?\d{3,})\b", question)
    department = None
    if match:
        code = match.group(1)
        prefix = extract_prefix(code)
        department = department_map.get(prefix, "Unknown")

    if random.random() < 0.2:
        response = random.choice(["I think ", "Maybe: ", "Possibly: ", "Here's what I found: "]) + response

    return response, department, top_score, related_questions

# --- Streamlit UI ---
st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")

model = load_model()
dataset = load_data()
question_list = dataset['question'].tolist()
question_embeddings = compute_question_embeddings(question_list)

# --- Apply filters ---
def apply_filters(df, faculty, department, level, semester):
    filtered_df = df.copy()
    if faculty:
        filtered_df = filtered_df[filtered_df['faculty'].isin(faculty)]
    if department:
        filtered_df = filtered_df[filtered_df['department'].isin(department)]
    if level:
        filtered_df = filtered_df[filtered_df['level'].isin(level)]
    if semester:
        filtered_df = filtered_df[filtered_df['semester'].isin(semester)]
    return filtered_df

# --- Sidebar Filters ---
with st.sidebar:
    st.header("Filter Questions")
    faculty_options = sorted(dataset['faculty'].dropna().unique())
    department_options = sorted(dataset['department'].dropna().unique())
    level_options = sorted(dataset['level'].dropna().unique())
    semester_options = sorted(dataset['semester'].dropna().unique())

    selected_faculty = st.multiselect("Faculty", faculty_options)
    selected_department = st.multiselect("Department", department_options)
    selected_level = st.multiselect("Level", level_options)
    selected_semester = st.multiselect("Semester", semester_options)

filtered_dataset = apply_filters(dataset, selected_faculty, selected_department, selected_level, selected_semester)

if filtered_dataset.empty:
    st.warning("No questions found for the selected filters. Please adjust your filter selection.")
    question_embeddings = None
else:
    question_list = filtered_dataset['question'].tolist()
    question_embeddings = compute_question_embeddings(question_list)

# --- Sidebar ---
with st.sidebar:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.related_questions = []
        st.session_state.last_department = None
        st.rerun()

# --- Title and Styles ---
st.markdown("""
<style>
    html, body, .stApp { font-family: 'Open Sans', sans-serif; }
    h1, h2, h3, h4, h5 { font-family: 'Merriweather', serif; color: #004080; }
    .chat-message-user {
        background-color: #d6eaff;
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 10px;
        margin-left: auto;
        max-width: 75%;
        font-weight: 550;
        color: #000;
    }
    .chat-message-assistant {
        background-color: #f5f5f5;
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 10px;
        margin-right: auto;
        max-width: 75%;
        font-weight: 600;
        color: #000;
    }
    .related-question {
        background-color: #e6f2ff;
        padding: 8px 12px;
        margin: 6px 6px 6px 0;
        display: inline-block;
        border-radius: 10px;
        font-size: 0.9rem;
        cursor: pointer;
    }
    .department-label {
        font-family: 'Merriweather', serif;
        font-size: 0.85rem;
        color: #004080;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 Crescent University Chatbot")

# --- Chat Render ---
for message in st.session_state.chat_history:
    role_class = "chat-message-user" if message["role"] == "user" else "chat-message-assistant"
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="{role_class}">{message["content"]}</div>', unsafe_allow_html=True)
        if message["role"] == "assistant" and st.session_state.last_department:
            st.markdown(f'<div class="department-label">Department: {st.session_state.last_department}</div>', unsafe_allow_html=True)

# --- Input ---
prompt = st.chat_input("Ask me anything about Crescent University...")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    matched_row = filtered_dataset[filtered_dataset['question'].str.lower() == prompt.lower()]
    if not matched_row.empty:
        answer = matched_row.iloc[0]['answer']
        department = None
        related = []
    else:
        answer, department, score, related = find_response(prompt, filtered_dataset, question_embeddings)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.related_questions = related
    st.session_state.last_department = department
    st.rerun()

# --- Related Suggestions ---
if st.session_state.related_questions:
    st.markdown("#### 💡 You might also ask:")
    for q in st.session_state.related_questions:
        unique_key = f"{uuid.uuid4().hex}"
        if st.button(q, key=f"related_{unique_key}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            answer, department, score, related = find_response(q, filtered_dataset, question_embeddings)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.related_questions = related
            st.session_state.last_department = department
            st.rerun()
