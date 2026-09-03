"""
将用户的分片数据和QA数据集适配到 CRUD eval 实验框架。

适配流程：
1. chunks_json → data/chunks_txt/  (每个chunk一个.txt文件，供Retriever读取)
2. QA_json     → data/split_merged.json  (统一格式，供get_task_datasets加载)

实验 Pipeline:
  chunks_txt/ (docs_directory)
       ↓ Milvus 向量索引
       ↓
  split_merged.json (questions/answers)
       ↓ Retriever.search_docs(query) → Top-K chunks
       ↓ LLM 生成答案
       ↓ Evaluator 评测
"""

import json
import os
from tqdm import tqdm

BASE_DIR = r"c:\Users\胡铭强\Desktop\chunk_code"

# ── 配置 ──────────────────────────────────────────────────────────────────
CHUNKS_JSON   = os.path.join(BASE_DIR, "output_chunks/all_chunks_chunks.json")
QA_1DOC       = os.path.join(BASE_DIR, "data/1doc_QA.json")
QA_2DOCS      = os.path.join(BASE_DIR, "data/2docs_QA.json")
QA_3DOCS      = os.path.join(BASE_DIR, "data/3docs_QA.json")

CHUNKS_TXT_DIR = os.path.join(BASE_DIR, "data/chunks_txt")
SPLIT_JSON     = os.path.join(BASE_DIR, "data/split_merged.json")


def step1_chunks_to_txt():
    """把 chunks JSON 转成每个 chunk 一个 .txt 文件。"""
    os.makedirs(CHUNKS_TXT_DIR, exist_ok=True)

    print(f"[Step 1] 读取 {CHUNKS_JSON} ...")
    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[Step 1] 写入 {CHUNKS_TXT_DIR}/ (共 {len(chunks)} 个 chunk) ...")
    for i, chunk in enumerate(tqdm(chunks, desc="Converting chunks")):
        fname = f"chunk_{i:06d}.txt"
        fpath = os.path.join(CHUNKS_TXT_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as out:
            out.write(chunk["chunk_text"])

    print(f"[Step 1] 完成! {len(chunks)} 个 txt 文件已保存到 {CHUNKS_TXT_DIR}/")
    return len(chunks)


def step2_qa_to_split_merged():
    """
    将 3 个 QA JSON 合并为 split_merged.json，
    字段映射: id → ID, questions → questions, answers → answers
    """
    print(f"[Step 2] 读取 QA 文件并合并 ...")

    def load_qa(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    qa_1doc  = load_qa(QA_1DOC)
    qa_2docs = load_qa(QA_2DOCS)
    qa_3docs = load_qa(QA_3DOCS)

    def adapt(qa_list, doc_count):
        adapted = []
        for item in tqdm(qa_list, desc=f"Adapting {doc_count}doc"):
            adapted.append({
                "ID":       item["id"],        # 字符串 id → 整型兼容
                "questions": item["questions"],
                "answers":   item["answers"],
            })
        return adapted

    result = {
        "questanswer_1doc":  adapt(qa_1doc,  1),
        "questanswer_2docs": adapt(qa_2docs, 2),
        "questanswer_3docs": adapt(qa_3docs, 3),
    }

    print(f"[Step 2] 写入 {SPLIT_JSON} ...")
    with open(SPLIT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    counts = {k: len(v) for k, v in result.items()}
    print(f"[Step 2] 完成! 各子集样本数: {counts}")
    return counts


if __name__ == "__main__":
    n_chunks = step1_chunks_to_txt()
    qa_counts = step2_qa_to_split_merged()

    print("\n" + "=" * 60)
    print("适配完成! 接下来运行实验:")
    print()
    print(f"  文档目录 (docs_path):  {CHUNKS_TXT_DIR}")
    print(f"  数据集文件 (data_path): {SPLIT_JSON}")
    print()
    print("  # 首次运行（构建向量索引）:")
    print("  python quick_start.py \\")
    print("      --model_name 'qwen7b' \\")
    print(f"      --data_path '{SPLIT_JSON}' \\")
    print(f"      --docs_path '{CHUNKS_TXT_DIR}' \\")
    print("      --docs_type 'txt' \\")
    print("      --chunk_size 128 \\")
    print("      --chunk_overlap 0 \\")
    print("      --retriever_name 'base' \\")
    print("      --collection_name 'meta_chunks_v1' \\")
    print("      --retrieve_top_k 8 \\")
    print("      --task 'quest_answer' \\")
    print("      --bert_score_eval \\")
    print("      --construct_index")
    print()
    print("  # 后续运行（跳过索引构建，直接评测）:")
    print("  python quick_start.py \\")
    print("      --model_name 'qwen7b' \\")
    print(f"      --data_path '{SPLIT_JSON}' \\")
    print(f"      --docs_path '{CHUNKS_TXT_DIR}' \\")
    print("      --retriever_name 'base' \\")
    print("      --collection_name 'meta_chunks_v1' \\")
    print("      --retrieve_top_k 8 \\")
    print("      --task 'quest_answer' \\")
    print("      --bert_score_eval")
    print("=" * 60)
