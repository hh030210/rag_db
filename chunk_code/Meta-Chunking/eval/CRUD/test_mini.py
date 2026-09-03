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
from pymilvus import connections, utility, Collection

connections.connect(host="localhost", port="19530")
utility.drop_collection("test_mini")

embed = HuggingfaceEmbeddings(model_name='BAAI/bge-base-zh-v1.5')
lc = LangchainEmbedding(embed)

# Generate 500 synthetic nodes (no file I/O)
nodes = [Node(text=f"test document number {i} with some content here and there " * 5) for i in range(500)]
print(f"Created {len(nodes)} nodes")

# Batch embed
texts = [n.get_text() for n in nodes]
print("Embedding...")
embs = lc.get_text_embedding_batch(texts)
print(f"Got {len(embs)} embeddings")

for n, e in zip(nodes, embs):
    n.embedding = e
print("Injected into nodes")

# Write to Milvus
print("Writing to Milvus...")
vector_store = MilvusVectorStore(
    uri="http://localhost:19530", token="",
    dim=768, overwrite=True,
    collection_name="test_mini"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=lc, llm=None)

idx = GPTVectorStoreIndex(
    nodes, service_context=service_context,
    storage_context=storage_context, show_progress=True
)
print("Index done")

c = Collection("test_mini")
c.load()
print(f"Collection test_mini: {c.num_entities} entities")
