import json, time, sys, os
sys.path.insert(0, ".")
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from src.embeddings.base import HuggingfaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from llama_index.data_structs import Node

JSON_PATH = r"C:\Users\胡铭强\Desktop\chunk_code\output_enhanced_v2\all_chunks_chunks.json"

print("Loading JSON...")
t0 = time.time()
with open(JSON_PATH, encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks in {time.time()-t0:.1f}s")

print("Creating nodes...")
t0 = time.time()
nodes = [Node(text=c["chunk_text"]) for c in chunks]
print(f"Created {len(nodes)} nodes in {time.time()-t0:.1f}s")

print("Testing embedding batch...")
lc = LangchainEmbedding(HuggingfaceEmbeddings(model_name='BAAI/bge-base-zh-v1.5'))
texts = [n.get_text() for n in nodes[:100]]
t0 = time.time()
embs = lc.get_text_embedding_batch(texts)
print(f"Batch 100 texts in {time.time()-t0:.1f}s")
print(f"Speed: {100/(time.time()-t0):.1f}/s")
print(f"Estimated 8034 chunks: {8034*100/(time.time()-t0)/len(embs[0]):.0f}s embedding time")
