"""
quick_dimension.py

快速维度抽取：跳过迭代优化，直接从候选维度生成标签。

用法：
  python quick_dimension.py --chunks output_no_ingest/chunks/all_chunks_chunks.json --dataset nankong_v1
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

# 环境变量已在父进程设置，此处不再重复设置

print("[环境]")
print(f"  LLM_OPENAI_COMPAT={os.getenv('LLM_OPENAI_COMPAT', '')}")
print(f"  LLM_BASE_URL={os.getenv('LLM_BASE_URL', '')}")
print(f"  LLM_MODEL={os.getenv('LLM_MODEL', '')}")
print(f"  LLM_API_KEY 前6位={os.getenv('LLM_API_KEY', '')[:6]}...")

from llm_service import DimensionMiningWithQwen
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np


def load_chunks_from_file(chunks_file: str):
    """加载 chunks JSON"""
    path = Path(chunks_file)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 统一格式
    docs = []
    for item in data:
        doc_id = item.get("id") or item.get("doc_id") or item.get("chunk_id")
        text = item.get("text") or item.get("doc_text") or item.get("chunk_text") or ""
        if doc_id and text:
            docs.append({"id": str(doc_id), "text": str(text)})

    print(f"加载 {len(docs)} 条文档")
    return docs


def cluster_sample(docs, n_clusters=20, n_per_cluster=5):
    """TF-IDF 聚类采样"""
    texts = [d["text"] for d in docs]

    if len(texts) < n_clusters:
        return texts[:10]

    try:
        vectorizer = TfidfVectorizer(max_features=1024, max_df=0.8, min_df=2)
        vectors = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    except Exception:
        return texts[:10]

    k = min(n_clusters, max(1, len(vectors) - 1))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)

    sampled = []
    for i in range(k):
        indices = np.where(labels == i)[0]
        if len(indices) == 0:
            continue
        # 取靠近中心的 n_per_cluster 个
        cluster_vecs = vectors[indices]
        center = kmeans.cluster_centers_[i].reshape(1, -1)
        dists = np.linalg.norm(cluster_vecs - center, axis=1)
        top_indices = np.argsort(dists)[:n_per_cluster]
        for idx in top_indices:
            sampled.append(texts[indices[idx]])

    print(f"聚类采样: {len(sampled)} 条（原始 {len(texts)} 条）")
    return sampled


def generate_candidates(sampled_docs, miner: DimensionMiningWithQwen):
    """生成候选维度（单次 LLM 调用）"""
    # 限制：前5篇，每篇最多300字
    snippets = [d[:300] if len(d) > 300 else d for d in sampled_docs[:5]]
    docs_text = "\n\n---\n\n".join(snippets)

    prompt = f"""你是一个专业的领域知识结构化专家。

请从以下文档集合中，归纳出能够全面描述该领域知识的所有关键维度（Dimension）。
每个维度应是一个简洁的名词短语（如"适宜人群"、"疾病类别"、"治疗方案"等）。

要求：
1. 维度应具有领域代表性，涵盖该领域的主要信息轴
2. 每个维度的值应当是离散的、可枚举的
3. 优先识别高频共性维度，兼顾领域特殊性
4. 维度之间应尽量独立，避免重复
5. 【禁止】不要生成"其他"、"其他信息"、"备注"、"杂项"等兜底性质的维度

请直接输出维度列表，用中文逗号分隔，不要包含任何解释：
维度1, 维度2, 维度3, ..."""

    from llm_service import PROMPT_GENERATE_CANDIDATES
    formatted_prompt = PROMPT_GENERATE_CANDIDATES.format(
        n=len(snippets),
        docs_snippet=docs_text
    )

    print(f"调用 LLM 生成候选维度（输入 {len(docs_text)} 字）...")
    raw = miner._call_llm(formatted_prompt, temperature=0.7, timeout=120)

    dims = [d.strip() for d in raw.split("，") if d.strip()]
    if len(dims) <= 1:
        dims = [d.strip() for d in raw.split(",") if d.strip()]

    print(f"LLM 返回 {len(dims)} 个维度: {dims}")
    return dims


def extract_tags(docs, dims, miner, output_path):
    """批量抽取标签"""
    results = {}
    total = len(docs)
    for i, doc in enumerate(docs):
        doc_id = doc["id"]
        text = doc["text"]
        if len(text) < 10:
            results[doc_id] = {}
            continue

        try:
            tags = miner.extract_batch_dimensions(text, dims)
            if tags:
                results[doc_id] = tags
            else:
                keywords = miner.extract_keywords_fallback(text)
                results[doc_id] = {"关键词": keywords} if keywords else {}
        except Exception as e:
            print(f"  [Error] {doc_id}: {e}")
            results[doc_id] = {}

        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{total} ...")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="快速维度抽取")
    parser.add_argument("--chunks", required=True, help="chunks JSON 文件路径")
    parser.add_argument("--dataset", default="default", help="数据集标识")
    parser.add_argument("--output", default="", help="输出文件路径（默认 experiment_data/）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少条文档（0=全部）")
    args = parser.parse_args()

    # 1. 加载 chunks
    print("=" * 60)
    print("Step 1: 加载文档")
    docs = load_chunks_from_file(args.chunks)
    if args.limit > 0:
        docs = docs[:args.limit]
        print(f"  限制为前 {args.limit} 条")

    # 2. 聚类采样
    print("=" * 60)
    print("Step 2: TF-IDF 聚类采样")
    sampled = cluster_sample(docs)

    # 3. 生成候选维度
    print("=" * 60)
    print("Step 3: LLM 生成候选维度")
    miner = DimensionMiningWithQwen()
    candidates = generate_candidates(sampled, miner)
    core_dims = candidates  # 直接作为核心维度

    print(f"\n核心维度（共 {len(core_dims)} 个）:")
    for d in core_dims:
        print(f"  - {d}")

    # 4. 批量抽取标签
    print("=" * 60)
    print(f"Step 4: 批量抽取标签（{len(docs)} 条文档 × {len(core_dims)} 个维度）")

    exp_dir = PROJECT_ROOT / "experiment_data"
    exp_dir.mkdir(exist_ok=True)
    v_cand_path = exp_dir / f"V_cand_{args.dataset}.json"
    v_core_path = exp_dir / f"V_core_{args.dataset}.json"
    tags_path = exp_dir / f"tags_output_{args.dataset}.json"

    with open(v_cand_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False)
    with open(v_core_path, "w", encoding="utf-8") as f:
        json.dump(core_dims, f, ensure_ascii=False)
    print(f"  候选维度: {v_cand_path}")
    print(f"  核心维度: {v_core_path}")

    results = extract_tags(docs, core_dims, miner, tags_path)
    print(f"  标签结果: {tags_path}")

    # 5. 统计
    print("=" * 60)
    print("维度统计:")
    dim_doc_counts = {d: 0 for d in core_dims}
    for doc_tags in results.values():
        for dim, vals in doc_tags.items():
            if dim in dim_doc_counts and vals:
                dim_doc_counts[dim] += 1

    for dim, cnt in sorted(dim_doc_counts.items(), key=lambda x: -x[1]):
        coverage = cnt / len(results) * 100 if results else 0
        print(f"  {dim}: {cnt}/{len(results)} 篇 ({coverage:.0f}%)")

    print("=" * 60)
    print("完成！")
    print(f"  核心维度: {v_core_path}")
    print(f"  标签结果: {tags_path}")
    print(f"  处理文档: {len(results)} 篇")

    return 0


if __name__ == "__main__":
    sys.exit(main())
