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
utility.drop_collection("eval_baseline_v1_top8")

embed = HuggingfaceEmbeddings(model_name='BAAI/bge-base-zh-v1.5')
lc = LangchainEmbedding(embed)

# Read all docs
docs_path = r"C:\Users\胡铭强\Desktop\chunk_code\crud_data\baseline_v1\docs"
import glob
files = sorted(glob.glob(os.path.join(docs_path, "*.txt")))
nodes = []
for fpath in files:
    with open(fpath, encoding="utf-8") as f:
        content = f.read().strip()
    if len(content) >= 10:
        nodes.append(Node(text=content))
print(f"Read {len(nodes)} nodes")

# Write to Milvus in 8000-chunk batches with pre-computed embeddings
vector_store = MilvusVectorStore(
    uri="http://localhost:19530", token="",
    dim=768, overwrite=True,
    collection_name="eval_baseline_v1_top8"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=lc, llm=None)

BATCH = 8000
for spilt_ids in range(0, len(nodes), BATCH):
    end_ids = min(spilt_ids + BATCH, len(nodes))
    batch_nodes = nodes[spilt_ids:end_ids]
    texts_batch = [n.get_text() for n in batch_nodes]

    print(f"Embedding batch {spilt_ids}-{end_ids} ({len(texts_batch)} texts)...")
    embs = lc.get_text_embedding_batch(texts_batch)
    for n, e in zip(batch_nodes, embs):
        n.embedding = e

    print(f"Writing to Milvus...")
    idx = GPTVectorStoreIndex(
        batch_nodes, service_context=service_context,
        storage_context=storage_context, show_progress=True
    )
    print(f"Part {spilt_ids} finished!")

    # Update storage context for next batch
    vector_store = MilvusVectorStore(
        uri="http://localhost:19530", token="",
        dim=768, overwrite=False,
        collection_name="eval_baseline_v1_top8"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

c = Collection("eval_baseline_v1_top8")
c.load()
print(f"DONE! Collection has {c.num_entities} entities")
