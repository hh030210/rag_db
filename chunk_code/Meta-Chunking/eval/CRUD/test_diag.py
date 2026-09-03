import sys, os
sys.path.insert(0, ".")
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from src.embeddings.base import HuggingfaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from llama_index.data_structs import Node
from llama_index import GPTVectorStoreIndex, ServiceContext
from llama_index.vector_stores import MilvusVectorStore
from llama_index import StorageContext
from pymilvus import connections, utility

connections.connect(host="localhost", port="19530")
utility.drop_collection("eval_baseline_v1_top8")

embed = HuggingfaceEmbeddings(model_name='BAAI/bge-base-zh-v1.5')
lc = LangchainEmbedding(embed)

# Read first 100 files from docs
docs_path = r"C:\Users\胡铭强\Desktop\chunk_code\crud_data\baseline_v1\docs"
import glob
files = sorted(glob.glob(os.path.join(docs_path, "*.txt")))[:100]
nodes = []
for fpath in files:
    with open(fpath, encoding="utf-8") as f:
        content = f.read().strip()
    if len(content) >= 10:
        nodes.append(Node(text=content))
print(f"Read {len(nodes)} nodes")

# Inject embeddings
texts = [n.get_text() for n in nodes]
embs = lc.get_text_embedding_batch(texts)
for n, e in zip(nodes, embs):
    n.embedding = e

# Write to Milvus
vector_store = MilvusVectorStore(
    uri="http://localhost:19530", token="",
    dim=768, overwrite=True,
    collection_name="eval_baseline_v1_top8"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=lc, llm=None)

idx = GPTVectorStoreIndex(nodes, service_context=service_context, storage_context=storage_context, show_progress=True)
print(f"Index created")

from pymilvus import Collection
c = Collection("eval_baseline_v1_top8")
c.load()
print(f"Collection has {c.num_entities} entities")
