import json
import os
from tqdm import tqdm

# 你的 chunks 文件
input_json = "output_chunks/all_chunks_chunks.json"

# 输出目录：每个 chunk 一个 .txt 文件
output_dir = "data/your_chunks"
os.makedirs(output_dir, exist_ok=True)

with open(input_json, "r", encoding="utf-8") as f:
    chunks = json.load(f)

for i, chunk in enumerate(tqdm(chunks, desc="Converting chunks")):
    filepath = os.path.join(output_dir, f"chunk_{i:06d}.txt")
    with open(filepath, "w", encoding="utf-8") as out:
        out.write(chunk["chunk_text"])

print(f"Done! {len(chunks)} chunks saved to {output_dir}/")
