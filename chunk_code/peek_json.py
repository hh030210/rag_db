import json
with open("output_baseline_v1/all_chunks_chunks.json", encoding="utf-8") as f:
    data = json.load(f)
print(type(data))
if isinstance(data, list):
    print(f"List of {len(data)} items")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))
elif isinstance(data, dict):
    print(f"Dict with keys: {list(data.keys())[:5]}")
