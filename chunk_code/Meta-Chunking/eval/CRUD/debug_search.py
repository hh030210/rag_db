"""Debug: 看 query_engine 的实际 response 格式"""
import sys
import os
sys.path.insert(0, "C:/Users/胡铭强/Desktop/chunk_code/Meta-Chunking/eval/CRUD")
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from src.embeddings.base import HuggingfaceEmbeddings
from src.retrievers.base import BaseRetriever

embed_model = HuggingfaceEmbeddings(model_name='BAAI/bge-base-zh-v1.5')
retriever = BaseRetriever(
    docs_directory="C:/Users/胡铭强/Desktop/chunk_code/data/chunks_txt_integrated",
    embed_model=embed_model, embed_dim=768,
    chunk_size=128, chunk_overlap=0,
    construct_index=False, add_index=False,
    collection_name="eval_integrated_top8",
    similarity_top_k=4,
)

q = "2023年7月28日 启明行动 防控 儿童 近视"
response = retriever.query_engine.query(q)
print("=== response type ===", type(response))
print("=== response.response ===")
print(repr(response.response[:2000]))
print("=== source_nodes count ===", len(response.source_nodes))
for i, sn in enumerate(response.source_nodes[:3]):
    print(f"--- node {i} ---")
    print("  score:", sn.score)
    print("  text head:", sn.node.text[:300] if hasattr(sn, 'node') and sn.node else 'NO NODE')
    print("  metadata:", sn.node.metadata if hasattr(sn, 'node') and sn.node else 'NO')
