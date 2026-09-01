"""
Boundary Clarity 评测 - 适配 Integrated Chunks
=================================================
输入: data/chunks_txt_integrated/ 目录下的所有 .txt 文件
输出:
  - no_semantic_similarity: 跨 chunk 边界的语义差异 (越高越好)
  - no_transition_naturalness: 跨 chunk 边界的过渡不自然度 (越高越好)

依赖:
  - SiliconFlow API (Qwen2.5-7B-Instruct / Qwen2.5-14B-Instruct, 跨 chunk 过渡自然度评估)
  - sentence-transformers/all-MiniLM-L6-v2 (语义相似度, 本地缓存)

运行示例:
  python eval_boundary_clarity.py --max_pairs 100
  python eval_boundary_clarity.py --models qwen2.5-7b qwen2.5-14b
  python eval_boundary_clarity.py --chunks_dir ../../../data/chunks_txt_integrated --output results.json
"""

import os
import json
import time
import glob
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# SiliconFlow API 配置：通过环境变量注入，避免把密钥写入源码
SILICONFLOW_API_KEY = os.environ.get('SILICONFLOW_API_KEY', '')
SILICONFLOW_API_BASE = 'https://api.siliconflow.cn/v1'

# 可用模型映射
MODEL_MAP = {
    'qwen2.5-7b':   'Qwen/Qwen2.5-7B-Instruct',
    'qwen2.5-14b': 'Qwen/Qwen2.5-14B-Instruct',
}

SIM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"


def load_sim_model():
    """从本地缓存加载 Sentence Transformer，避免网络问题"""
    from pathlib import Path
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache = cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2"

    # snapshot 才是实际模型文件所在
    snapshots = model_cache / "snapshots"
    if snapshots.exists():
        for snap in snapshots.iterdir():
            if snap.is_dir():
                model_path = str(snap)
                print(f"  [使用本地缓存] {model_path}")
                return SentenceTransformer(model_path, device=DEVICE)

    print(f"  [使用远程加载] {SIM_MODEL_NAME}")
    return SentenceTransformer(SIM_MODEL_NAME, device=DEVICE)


# ============================================================
# SiliconFlow API Client (过渡自然度评估)
# ============================================================
class SiliconFlowClient:
    def __init__(self, api_key, api_base, model_name):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name

    def get_transition_score(self, prefix, target):
        """
        评估从 prefix 过渡到 target 的自然程度 (越大越自然)。
        直接衡量跨 chunk 边界的语义连贯性。
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a text coherence evaluator. Output ONLY a single float number between 0 and 1."},
                    {"role": "user", "content": f"How natural is it for the second text to follow the first? Output only a single number 0-1:\n\nFirst: {prefix}\n\nSecond: {target}"}
                ],
                max_tokens=8,
                temperature=0,
                timeout=60,
            )
            raw = resp.choices[0].message.content.strip()
            return max(0.01, min(1.0, float(raw)))
        except Exception as e:
            print(f"  [API 错误] {e}")
            return None


# ============================================================
# 语义相似度计算
# ============================================================
def get_sim_score(model, text1, text2):
    """计算两个文本的余弦相似度"""
    emb1 = model.encode([text1], normalize_embeddings=True)
    emb2 = model.encode([text2], normalize_embeddings=True)
    return (emb1 @ emb2.T).item()


# ============================================================
# 主评测逻辑
# ============================================================
def load_chunks_from_dir(chunk_dir):
    """从目录加载所有 .txt chunk 文件"""
    txt_files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.txt")))
    chunks = []
    for fpath in txt_files:
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if len(text) >= 5:  # 过滤太短的 chunk
            chunks.append(text)
    return chunks


def evaluate_boundary_clarity(chunks, api_client, sim_model, model_display_name, max_pairs=None, batch_size=50):
    """
    计算 Boundary Clarity 指标

    对每对相邻 chunk (chunks[i], chunks[i+1]):
      - no_semantic_similarity = 1 - cosine_sim(chunks[i+1], chunks[i])
      - no_transition_naturalness = 1 - transition_score(chunks[i], chunks[i+1])
    """
    n = len(chunks)
    sim_scores = []
    transition_scores = []

    print(f"\n  共 {n} 个 chunks, 将评估 {n-1} 个跨 chunk 边界...")
    if max_pairs:
        print(f"  (限制最多评估 {max_pairs} 对)")

    start_time = time.time()

    for i in tqdm(range(n - 1), desc=f"[{model_display_name}] 评估跨块边界"):
        if max_pairs and i >= max_pairs:
            break

        text1 = chunks[i]
        text2 = chunks[i + 1]

        try:
            # 语义相似度
            sim = get_sim_score(sim_model, text2, text1)
            no_sim = 1.0 - sim
            sim_scores.append(no_sim)

            # 跨 chunk 边界的过渡分数 (越大 = 过渡越自然 = 边界越不清晰)
            transition = api_client.get_transition_score(text1, text2)
            if transition is not None:
                no_transition = 1.0 - transition
                transition_scores.append(no_transition)
            else:
                # API 失败时跳过
                if sim_scores:
                    sim_scores.pop()

        except Exception as e:
            print(f"\n  [警告] chunk {i} 评估失败: {e}")
            continue

        if (i + 1) % batch_size == 0:
            elapsed = time.time() - start_time
            avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0
            avg_trans = sum(transition_scores) / len(transition_scores) if transition_scores else 0
            print(f"\n  [{i+1}/{n-1}] 当前均值 - no_semantic_similarity: {avg_sim:.4f}, "
                  f"no_transition_naturalness: {avg_trans:.4f} (耗时 {elapsed:.0f}s)")

    elapsed = time.time() - start_time

    # 汇总
    avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0
    avg_trans = sum(transition_scores) / len(transition_scores) if transition_scores else 0.0
    std_sim = (sum((x - avg_sim) ** 2 for x in sim_scores) / len(sim_scores)) ** 0.5 if sim_scores else 0.0
    std_trans = (sum((x - avg_trans) ** 2 for x in transition_scores) / len(transition_scores)) ** 0.5 if transition_scores else 0.0

    results = {
        "model": model_display_name,
        "api_model": api_client.model_name,
        "n_pairs_evaluated": len(sim_scores),
        "no_semantic_similarity": {
            "mean": avg_sim,
            "std": std_sim,
            "min": min(sim_scores) if sim_scores else 0,
            "max": max(sim_scores) if sim_scores else 0,
        },
        "no_transition_naturalness": {
            "mean": avg_trans,
            "std": std_trans,
            "min": min(transition_scores) if transition_scores else 0,
            "max": max(transition_scores) if transition_scores else 0,
        },
        "elapsed_seconds": elapsed,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Boundary Clarity 评测 (SiliconFlow)")
    parser.add_argument("--chunks_dir", type=str,
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 3), 'data', 'chunks_txt_integrated')),
                        help="chunks .txt 文件目录")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出路径 (默认: data/boundary_clarity_{model}.json)")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="最多评估多少对相邻 chunk (默认全部)")
    parser.add_argument("--batch_log", type=int, default=50,
                        help="每多少对输出一次日志")
    parser.add_argument("--models", type=str, nargs='+',
                        choices=list(MODEL_MAP.keys()),
                        default=['qwen2.5-7b'],
                        help="选择要评测的模型 (默认: qwen2.5-7b)")
    args = parser.parse_args()

    print("=" * 60)
    print("Boundary Clarity 评测 (SiliconFlow API)")
    print("=" * 60)
    print(f"  Chunks 目录: {args.chunks_dir}")
    print(f"  评测模型: {args.models}")
    print(f"  Sim 模型: {SIM_MODEL_NAME}")
    print(f"  设备: {DEVICE}")

    # 加载 chunks
    print("\n[1/3] 加载 chunks...")
    chunks = load_chunks_from_dir(args.chunks_dir)
    print(f"  加载了 {len(chunks)} 个 chunks")

    if len(chunks) < 2:
        print("  [错误] chunks 数量不足，无法评估边界！")
        return

    # 加载 Sentence Transformer
    print(f"\n[2/3] 加载 Sentence Transformer: {SIM_MODEL_NAME}")
    sim_model = load_sim_model()
    print(f"  Sentence Transformer 加载完成 ({DEVICE})")

    # 加载 SiliconFlow API Client
    print("\n[3/3] 初始化 SiliconFlow API client...")

    all_results = {}

    for model_key in args.models:
        model_full_name = MODEL_MAP[model_key]
        print(f"\n{'='*60}")
        print(f"开始评测模型: {model_key} ({model_full_name})")
        print(f"{'='*60}")

        # 初始化 client
        api_client = SiliconFlowClient(
            api_key=SILICONFLOW_API_KEY,
            api_base=SILICONFLOW_API_BASE,
            model_name=model_full_name,
        )
        print(f"  API client 就绪 (模型: {model_full_name})")

        # 评测
        results = evaluate_boundary_clarity(
            chunks, api_client, sim_model,
            model_display_name=model_key,
            max_pairs=args.max_pairs,
            batch_size=args.batch_log
        )

        all_results[model_key] = results

        # 保存单个模型结果
        output_path = args.output or f"../../../data/boundary_clarity_{model_key}.json"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  结果已保存到: {output_path}")

        # 打印摘要
        print(f"\n  [{model_key}] 结果摘要:")
        print(f"    no_semantic_similarity:     {results['no_semantic_similarity']['mean']:.4f} "
              f"± {results['no_semantic_similarity']['std']:.4f}")
        print(f"    no_transition_naturalness:  {results['no_transition_naturalness']['mean']:.4f} "
              f"± {results['no_transition_naturalness']['std']:.4f}")
        print(f"    耗时: {results['elapsed_seconds']:.0f}s")

        # 避免 API 限流
        if model_key != args.models[-1]:
            print("\n  等待 3 秒后切换下一个模型...")
            time.sleep(3)

    # 打印汇总
    print("\n" + "=" * 60)
    print("全部模型评测完成 - 汇总对比")
    print("=" * 60)
    print(f"{'模型':<15} {'no_sem_sim':<15} {'no_trans_nat':<15} {'耗时(s)':<10}")
    print("-" * 60)
    for model_key, r in all_results.items():
        print(f"{model_key:<15} {r['no_semantic_similarity']['mean']:.4f} ± {r['no_semantic_similarity']['std']:.4f}   "
              f"{r['no_transition_naturalness']['mean']:.4f} ± {r['no_transition_naturalness']['std']:.4f}   "
              f"{r['elapsed_seconds']:.0f}s")
    print("=" * 60)

    # 保存汇总结果
    summary_path = args.output or "../../../data/boundary_clarity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n汇总结果已保存到: {summary_path}")


if __name__ == "__main__":
    main()
