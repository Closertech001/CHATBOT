import json

def load_json_chunks(path, max_chars=1000):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = []
    for entry in data:
        q = entry.get("question", "").strip()
        a = entry.get("answer", "").strip()
        if q and a:
            chunk = f"Q: {q}\nA: {a}"
            chunks.append(chunk[:max_chars])  # optional truncation

    return chunks
