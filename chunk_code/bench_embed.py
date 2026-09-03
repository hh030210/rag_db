import time, sys, os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, "Meta-Chunking/eval/CRUD")
from llama_index.embeddings import LangchainEmbedding
from src.embeddings.base import HuggingfaceEmbeddings
from concurrent.futures import ThreadPoolExecutor

embed = HuggingfaceEmbeddings(model_name='BAAI/bge-base-zh-v1.5')
lc_embed = LangchainEmbedding(embed)
texts = ["测试文本" + str(i) for i in range(512)]

# Serial
t0 = time.time()
for t in texts:
    lc_embed.get_text_embedding(t)
serial = time.time() - t0
print(f"Serial 512 texts: {serial:.1f}s ({512/serial:.1f}/s)")

# Batch
t0 = time.time()
lc_embed.get_text_embedding_batch(texts)
batch = time.time() - t0
print(f"Batch 512 texts: {batch:.1f}s ({512/batch:.1f}/s)")

# Parallel 8 workers
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = [ex.submit(lc_embed.get_text_embedding, t) for t in texts]
    [f.result() for f in futures]
parallel = time.time() - t0
print(f"Parallel-8 512 texts: {parallel:.1f}s ({512/parallel:.1f}/s)")
