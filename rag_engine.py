import os
import faiss
import openai
from dotenv import load_dotenv
from utils.chunker import load_json_chunks

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Embedding function
def get_embedding(text, model="text-embedding-3-small"):
    from openai import OpenAI
    client = OpenAI()  # Automatically picks up API key from env
    text = text.replace("\n", " ")  # Clean newlines
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Load and chunk the JSON data
def get_chunks():
    return load_json_chunks("data/university_data.json")

# Build or load FAISS index
def build_or_load_faiss_index(chunks, dim=1536):
    index_file = "faiss_index/index.faiss"

    if os.path.exists(index_file):
        try:
            index = faiss.read_index(index_file)
            return index
        except Exception as e:
            print(f"[WARN] Failed to load existing index: {e}. Rebuilding...")

    print("[INFO] Building new FAISS index...")
    index = faiss.IndexFlatL2(dim)
    vectors = [get_embedding(c) for c in chunks]
    index.add(vectors)

    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, index_file)
    return index

# Search relevant chunks for the query
def search_chunks(query, chunks, index, top_k=3):
    query_vec = get_embedding(query)
    D, I = index.search([query_vec], top_k)
    return [chunks[i] for i in I[0] if i < len(chunks)]

# Main chatbot function
def get_chat_response(query):
    chunks = get_chunks()
    index = build_or_load_faiss_index(chunks)
    relevant_chunks = search_chunks(query, chunks, index)

    context = "\n".join(relevant_chunks)
    prompt = f"""You are a helpful university assistant.
Use the following context to answer the user's question.\n
Context:\n{context}\n
Question: {query}
Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()
