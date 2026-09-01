import sys, os
sys.path.insert(0, ".")
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from src.embeddings.base import HuggingfaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from llama_index.data_structs import Node

embed = HuggingfaceEmbeddings(model_name='BAAI/bge-base-zh-v1.5')
lc = LangchainEmbedding(embed)

# Test batch
texts = ["test " + str(i) for i in range(4)]
emb = lc.get_text_embedding_batch(texts)
print(f"Batch type: {type(emb)}, len: {len(emb)}, first[0]: {type(emb[0])}")

# Test node injection
n = Node()
n.text = "hello"
n.embedding = emb[0]
print(f"Node embedding type: {type(n.embedding)}, shape-like: {len(n.embedding)}")
print("Node injection OK")
