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
from pymilvus import connections, Collection

connections.connect(host="localhost", port="19530")

embed = HuggingfaceEmbeddings(model_name='BAAI/bge-base-zh-v1.5')
lc = LangchainEmbedding(embed)

# Test with 5 nodes
texts = ["test text " + str(i) for i in range(5)]
nodes = [Node(text=t) for t in texts]

# Inject embeddings
embs = lc.get_text_embedding_batch(texts)
for n, e in zip(nodes, embs):
    n.embedding = e

print(f"Node[0] embedding type: {type(nodes[0].embedding)}, len={len(nodes[0].embedding)}")

# Write to test collection
vector_store = MilvusVectorStore(
    uri="http://localhost:19530", token="",
    dim=768, overwrite=True,
    collection_name="embed_test"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=lc, llm=None)

idx = GPTVectorStoreIndex(nodes, service_context=service_context, storage_context=storage_context)
print("Index created")

# Check Milvus
c = Collection("embed_test")
c.load()
print(f"Collection has {c.num_entities} entities")
