"""
Smoke test: 验证 retriever + LLM 端到端链路是否正常
- 用 5 条 questanswer_1doc 数据
- 跑 retriever.search_docs(question) → 拿到 context
- 把 context + question 喂给 LLM, 生成答案
- 看是否出现 IndexError (llama-index 0.9+ response 格式变化) 等问题
"""
import json
import os
import sys
import time

sys.path.insert(0, "C:/Users/胡铭强/Desktop/chunk_code/Meta-Chunking/eval/CRUD")
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from src.embeddings.base import HuggingfaceEmbeddings
from src.retrievers.base import BaseRetriever
from src.llms import Qwen_API_Chat

print("[1/3] Loading embedder (BGE-base-zh-v1.5)...")
embed_model = HuggingfaceEmbeddings(model_name='BAAI/bge-base-zh-v1.5')

print("[2/3] Loading retriever (eval_integrated_top8)...")
retriever = BaseRetriever(
    docs_directory="C:/Users/胡铭强/Desktop/chunk_code/data/chunks_txt_integrated",
    embed_model=embed_model, embed_dim=768,
    chunk_size=128, chunk_overlap=0,
    construct_index=False, add_index=False,
    collection_name="eval_integrated_top8",
    similarity_top_k=4,  # smoke 用 4 即可
)
print("  retriever ready")

print("[3/3] Loading LLM (Qwen via SiliconFlow)...")
llm = Qwen_API_Chat(model_name='qwen_api', temperature=0.1, max_new_tokens=512)

# 5 条样本
print("Loading 5 sample questions from 1doc_QA.json...")
with open("C:/Users/胡铭强/Desktop/chunk_code/data/1doc_QA.json", encoding='utf-8') as f:
    qa_list = json.load(f)
samples = qa_list[:5]

for i, qa in enumerate(samples):
    print(f"\n=== Sample {i+1}/5 (id={qa['id']}) ===")
    print(f"Q: {qa['questions'][:120]}")
    print(f"GT: {qa['answers'][:120]}")

    # 检索
    t0 = time.time()
    try:
        ctx = retriever.search_docs(qa['questions'])
    except Exception as e:
        print(f"  [SEARCH ERROR] {type(e).__name__}: {e}")
        continue
    print(f"  retrieve time: {time.time()-t0:.2f}s, context len: {len(ctx)}")
    if not ctx.strip():
        print("  [WARN] empty context!")
        continue
    print(f"  ctx head: {ctx[:200]}")

    # 生成
    template = open(
        "C:/Users/胡铭强/Desktop/chunk_code/Meta-Chunking/eval/CRUD/src/prompts/quest_answer.txt",
        encoding='utf-8'
    ).read()
    query = template.format(question=qa['questions'], search_documents=ctx)
    t0 = time.time()
    try:
        res = llm.safe_request(query)
    except Exception as e:
        print(f"  [LLM ERROR] {type(e).__name__}: {e}")
        continue
    real = res.split('<response>')[-1].split('</response>')[0].strip()
    print(f"  llm time: {time.time()-t0:.2f}s")
    print(f"  ANSWER: {real[:200]}")

print("\n=== SMOKE TEST DONE ===")
