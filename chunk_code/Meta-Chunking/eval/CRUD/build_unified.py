import sys, os, json, time
sys.path.insert(0, ".")
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import builtins
_orig_print = builtins.print
def flush_print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    sys.stdout.flush()
builtins.print = flush_print

from src.embeddings.base import HuggingfaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from llama_index.data_structs import Node
from llama_index import GPTVectorStoreIndex, ServiceContext
from llama_index.vector_stores import MilvusVectorStore
from llama_index import StorageContext
from pymilvus import connections, utility, Collection

COLLECTION = sys.argv[1] if len(sys.argv) > 1 else "eval_enhanced_v2_top8"
JSON_PATH = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\胡铭强\Desktop\chunk_code\output_enhanced_v2\all_chunks_chunks.json"
EMBED_MODEL = sys.argv[3] if len(sys.argv) > 3 else 'BAAI/bge-base-zh-v1.5'
EMBED_DIM = int(sys.argv[4]) if len(sys.argv) > 4 else 768
MILVUS_URI = os.environ.get("DENOISE_MILVUS_URI", "http://localhost:19530")

# 关键修复:llama_index 的 MilvusVectorStore 会建临时 alias (cm-XXXX),
# 但 ORM Collection() 默认从 'default' alias 查——需要在调用前 patch 一下
# 让 _fetch_handler 在任意 alias 上找不到时兜底到 'default'。
def _patch_milvus_default_alias():
    from pymilvus.orm import connections as _conn_mod
    _orig_fetch = _conn_mod.Connections._fetch_handler
    def _patched_fetch(self, using):
        try:
            return _orig_fetch(self, using)
        except Exception:
            return _orig_fetch(self, "default")
    _conn_mod.Connections._fetch_handler = _patched_fetch

_patch_milvus_default_alias()
if not connections.has_connection("default"):
    connections.connect(alias="default", uri=MILVUS_URI)
try:
    utility.drop_collection(COLLECTION)
except:
    pass

print(f"Using model: {EMBED_MODEL}, dim={EMBED_DIM}")
embed = HuggingfaceEmbeddings(model_name=EMBED_MODEL)
lc = LangchainEmbedding(embed)

print("Loading JSON...")
t0 = time.time()
with open(JSON_PATH, encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks in {time.time()-t0:.1f}s")

print("Pre-computing embeddings...")
t0 = time.time()
all_texts = [c["chunk_text"] for c in chunks]  # 保留文本,稍后写入 Node.text
chunk_ids = list(range(len(chunks)))
del chunks

all_embs = []
BATCH = 256
for i in range(0, len(all_texts), BATCH):
    batch = all_texts[i:i+BATCH]
    embs = lc.get_text_embedding_batch(batch)
    all_embs.extend(embs)
    pct = min(100, (i + BATCH) * 100 // len(all_texts))
    elapsed = time.time() - t0
    eta = elapsed * (len(all_texts) - i - BATCH) / max(1, i + BATCH)
    print(f"  {pct}% ({min(i+BATCH, len(all_texts))}/{len(all_texts)}), elapsed {elapsed:.0f}s, ETA {eta:.0f}s", flush=True)
embed_time = time.time() - t0
print(f"All embeddings done in {embed_time:.1f}s")

print("Writing to Milvus...")
t0 = time.time()
vector_store = MilvusVectorStore(uri=MILVUS_URI, token="", dim=EMBED_DIM, overwrite=True, collection_name=COLLECTION)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=lc, llm=None)

BATCH_NODES = 4000
for spilt_ids in range(0, len(all_embs), BATCH_NODES):
    end_ids = min(spilt_ids + BATCH_NODES, len(all_embs))
    batch_nodes = []
    for idx, emb in enumerate(all_embs[spilt_ids:end_ids]):
        # 关键修复:Node.text 写入真实 chunk 文本,这样 search 才能拿到原文
        # 而非"chunk_xxxxx"的文件名占位符
        n = Node(text=all_texts[spilt_ids + idx])
        n.embedding = emb
        batch_nodes.append(n)

    print(f"Writing {spilt_ids}-{end_ids} ({len(batch_nodes)} nodes)...", flush=True)
    idx_obj = GPTVectorStoreIndex(
        batch_nodes, service_context=service_context,
        storage_context=storage_context, show_progress=False
    )
    del batch_nodes

    vector_store = MilvusVectorStore(uri=MILVUS_URI, token="", dim=EMBED_DIM, overwrite=False, collection_name=COLLECTION)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

write_time = time.time() - t0
c = Collection(COLLECTION)
c.load()
print(f"DONE! {c.num_entities} entities")
print(f"  Embed: {embed_time:.1f}s, Write: {write_time:.1f}s, Total: {embed_time + write_time:.1f}s", flush=True)
