import faiss
import os
import openai
import tiktoken
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv
from utils.chunker import load_json_chunks

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_chunks():
    return load_json_chunks("data/university_data.json")

def build_or_load_faiss_index(chunks, dim=1536):
    index_file = "faiss_index/index.faiss"
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
    else:
        index = faiss.IndexFlatL2(dim)
        vectors = [get_embedding(c, engine="text-embedding-3-small") for c in chunks]
        index.add(vectors)
        faiss.write_index(index, index_file)
    return index

def search_chunks(query, chunks, index, top_k=3):
    query_vec = get_embedding(query, engine="text-embedding-3-small")
    D, I = index.search([query_vec], top_k)
    return [chunks[i] for i in I[0]]

def get_chat_response(query):
    chunks = get_chunks()
    index = build_or_load_faiss_index(chunks)
    context = "\n".join(search_chunks(query, chunks, index))

    prompt = f"""Answer the question using the context below.\n
Context:\n{context}\n
Question: {query}
Answer:"""

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content.strip()
