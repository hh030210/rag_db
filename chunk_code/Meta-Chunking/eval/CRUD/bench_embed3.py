import time, sys, os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, ".")
from src.embeddings.base import HuggingfaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from concurrent.futures import ThreadPoolExecutor

embed = HuggingfaceEmbeddings(model_name='BAAI/bge-base-zh-v1.5')
lc = LangchainEmbedding(embed)
texts = ["test text " + str(i) for i in range(256)]

# Warmup
lc.get_text_embedding("warmup")

# Serial
t0 = time.time()
for t in texts:
    lc.get_text_embedding(t)
serial = time.time() - t0
print(f"Serial 256: {serial:.1f}s ({256/serial:.1f}/s)")

# Batch
t0 = time.time()
lc.get_text_embedding_batch(texts)
batch = time.time() - t0
print(f"Batch 256: {batch:.1f}s ({256/batch:.1f}/s)")

# Parallel 8w
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = [ex.submit(lc.get_text_embedding, t) for t in texts]
    [f.result() for f in futures]
parallel = time.time() - t0
print(f"Parallel-8 256: {parallel:.1f}s ({256/parallel:.1f}/s)")
