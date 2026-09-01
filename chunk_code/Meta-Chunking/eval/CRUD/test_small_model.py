import sys, os, json, time
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
utility.drop_collection("eval_test_small")

embed = HuggingfaceEmbeddings(model_name='BAAI/bge-small-zh-v1.5')
lc = LangchainEmbedding(embed)

print("Loading 200 sample chunks...")
with open(r"C:\Users\胡铭强\Desktop\chunk_code\output_enhanced_v2\all_chunks_chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)
chunks = chunks[:200]
print(f"Testing {len(chunks)} chunks")

t0 = time.time()
embs = lc.get_text_embedding_batch([c["chunk_text"] for c in chunks])
emb_time = time.time() - t0
print(f"Embedding {len(chunks)} texts: {emb_time:.1f}s ({len(chunks)/emb_time:.1f}/s)")

nodes = []
for c, e in zip(chunks, embs):
    n = Node(text=c["chunk_text"])
    n.embedding = e
    nodes.append(n)

vector_store = MilvusVectorStore(uri="http://localhost:19530", token="", dim=512, overwrite=True, collection_name="eval_test_small")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=lc, llm=None)

t0 = time.time()
idx = GPTVectorStoreIndex(nodes, service_context=service_context, storage_context=storage_context, show_progress=True)
print(f"Index write: {time.time()-t0:.1f}s")

c = Collection("eval_test_small")
c.load()
print(f"Collection: {c.num_entities} entities, dim=512")
print(f"Total time for 200 chunks: {emb_time + time.time()-t0:.1f}s")
print(f"Estimated 8034 chunks: {(emb_time + time.time()-t0)/200*8034:.0f}s = {(emb_time + time.time()-t0)/200*8034/60:.1f}min")
