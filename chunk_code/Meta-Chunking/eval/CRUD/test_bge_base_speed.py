import sys, os, json, time
sys.path.insert(0, ".")
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from src.embeddings.base import HuggingfaceEmbeddings
from llama_index.embeddings import LangchainEmbedding

embed = HuggingfaceEmbeddings(model_name='BAAI/bge-base-zh-v1.5')
lc = LangchainEmbedding(embed)

# Warm up
lc.get_text_embedding("test")

# Test 100 texts
texts = [f"这是第{i}个测试文本内容包含中文和标点符号用于测试嵌入向量的生成速度" for i in range(100)]
t0 = time.time()
embs = lc.get_text_embedding_batch(texts)
t1 = time.time()
print(f"bge-base: 100 texts in {t1-t0:.1f}s ({100/(t1-t0):.1f}/s)")

# Test 256 texts
texts2 = [f"这是第{i}个测试文本内容包含中文和标点符号用于测试嵌入向量的生成速度" for i in range(256)]
t0 = time.time()
embs2 = lc.get_text_embedding_batch(texts2)
t1 = time.time()
print(f"bge-base: 256 texts in {t1-t0:.1f}s ({256/(t1-t0):.1f}/s)")
