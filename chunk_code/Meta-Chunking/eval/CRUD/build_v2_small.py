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
utility.drop_collection("eval_enhanced_v2_top8")

embed = HuggingfaceEmbeddings(model_name='BAAI/bge-small-zh-v1.5')
lc = LangchainEmbedding(embed)
EMBED_DIM = 512

print("Loading JSON...")
t0 = time.time()
with open(r"C:\Users\胡铭强\Desktop\chunk_code\output_enhanced_v2\all_chunks_chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks in {time.time()-t0:.1f}s")

print("Pre-computing embeddings...")
t0 = time.time()
all_texts = [c["chunk_text"] for c in chunks]
del chunks

all_embs = []
BATCH = 512
for i in range(0, len(all_texts), BATCH):
    batch = all_texts[i:i+BATCH]
    embs = lc.get_text_embedding_batch(batch)
    all_embs.extend(embs)
    pct = min(100, (i + BATCH) * 100 // len(all_texts))
    elapsed = time.time() - t0
    eta = elapsed * (len(all_texts) - i - BATCH) / max(1, i + BATCH)
    print(f"  {pct}% ({min(i+BATCH, len(all_texts))}/{len(all_texts)}), elapsed {elapsed:.0f}s, ETA {eta:.0f}s")
del all_texts
embed_time = time.time() - t0
print(f"All embeddings done in {embed_time:.1f}s")

print("Writing to Milvus...")
t0 = time.time()
vector_store = MilvusVectorStore(uri="http://localhost:19530", token="", dim=EMBED_DIM, overwrite=True, collection_name="eval_enhanced_v2_top8")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=lc, llm=None)

BATCH_NODES = 8000
for spilt_ids in range(0, len(all_embs), BATCH_NODES):
    end_ids = min(spilt_ids + BATCH_NODES, len(all_embs))
    batch_nodes = []
    for idx, emb in enumerate(all_embs[spilt_ids:end_ids]):
        n = Node(text=f"chunk_{spilt_ids + idx}")
        n.embedding = emb
        batch_nodes.append(n)

    print(f"Writing {spilt_ids}-{end_ids} ({len(batch_nodes)} nodes)...")
    idx_obj = GPTVectorStoreIndex(
        batch_nodes, service_context=service_context,
        storage_context=storage_context, show_progress=False
    )
    del batch_nodes

    vector_store = MilvusVectorStore(uri="http://localhost:19530", token="", dim=EMBED_DIM, overwrite=False, collection_name="eval_enhanced_v2_top8")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

write_time = time.time() - t0
c = Collection("eval_enhanced_v2_top8")
c.load()
print(f"DONE! {c.num_entities} entities")
print(f"  Embed: {embed_time:.1f}s, Write: {write_time:.1f}s, Total: {embed_time + write_time:.1f}s")
