import json

def load_json_chunks(path, max_chars=500):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = []

    def flatten_json(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                flatten_json(v, prefix + k + ": ")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                flatten_json(item, prefix + f"{i}: ")
        else:
            chunks.append(prefix + str(obj))

    flatten_json(data)
    
    return [c.strip()[:max_chars] for c in chunks if c.strip()]
