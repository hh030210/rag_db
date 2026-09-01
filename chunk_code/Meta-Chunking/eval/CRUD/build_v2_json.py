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

COLLECTION = "eval_enhanced_v2_top8"
JSON_PATH = r"C:\Users\胡铭强\Desktop\chunk_code\output_enhanced_v2\all_chunks_chunks.json"

utility.drop_collection(COLLECTION)

embed = HuggingfaceEmbeddings(model_name='BAAI/bge-base-zh-v1.5')
lc = LangchainEmbedding(embed)

print("Loading JSON...")
with open(JSON_PATH, encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks")

nodes = [Node(text=c["chunk_text"]) for c in chunks]
del chunks

vector_store = MilvusVectorStore(
    uri="http://localhost:19530", token="",
    dim=768, overwrite=True,
    collection_name=COLLECTION
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=lc, llm=None)

t0 = time.time()
BATCH = 8000
for spilt_ids in range(0, len(nodes), BATCH):
    end_ids = min(spilt_ids + BATCH, len(nodes))
    batch_nodes = nodes[spilt_ids:end_ids]
    texts = [n.get_text() for n in batch_nodes]

    print(f"Embedding {spilt_ids}-{end_ids}...")
    embs = lc.get_text_embedding_batch(texts)
    for n, e in zip(batch_nodes, embs):
        n.embedding = e

    print(f"Writing to Milvus...")
    idx = GPTVectorStoreIndex(
        batch_nodes, service_context=service_context,
        storage_context=storage_context, show_progress=True
    )
    print(f"Part {spilt_ids} done!")

    vector_store = MilvusVectorStore(
        uri="http://localhost:19530", token="",
        dim=768, overwrite=False,
        collection_name=COLLECTION
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

elapsed = time.time() - t0
c = Collection(COLLECTION)
c.load()
print(f"DONE! {c.num_entities} entities in {elapsed:.1f}s")
