"""
Chunk Stickiness 评测 - 适配 Integrated Chunks
===============================================
输入: data/chunks_txt_integrated/ 目录下的所有 .txt 文件
输出:
  - structural_entropy: 结构熵 (越低越好 = chunk 内部黏连度高)

原理:
  将每个 chunk 内部的 sentences 建成语义连通图，过滤弱边(权重<0.8)后，
  计算图的节点度分布结构熵。熵越低表示节点度分布越不均匀（少数句子高度连通）
  = 块内语义抱团，Stickiness 高。

依赖:
  - Qwen/Qwen2-1.5B-Instruct (本地缓存, PPL 计算)

运行:
  $env:HF_HUB_OFFLINE='1'
  python eval_chunk_stickiness.py --max_chunks 100
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
import json
import time
import math
import glob
import torch
import argparse
import heapq
import copy
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 模型配置 - 通过命令行参数指定
# 默认使用本地缓存的 Qwen2-1.5B-Instruct（需设置 HF_HUB_OFFLINE=1）
# 也可通过 --llm_model 传入 SiliconFlow API 地址
# ============================================================
LLM_MODEL_PATH = "Qwen/Qwen2-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# PPL 计算 (来自 perplexity_tools.py)
# ============================================================
class Chunking:
    """复用原始 perplexity_chunking.py 的 PPL 计算逻辑"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_ppl_for_next(self, first_sentence, next_sentence):
        tokenized_1 = self.tokenizer(first_sentence, return_tensors="pt", add_special_tokens=False)
        tokenized_2 = self.tokenizer(next_sentence, return_tensors="pt", add_special_tokens=False)
        input_ids = torch.cat([
            tokenized_1["input_ids"].to(self.model.device),
            tokenized_2["input_ids"].to(self.model.device)
        ], dim=-1)
        attention_mask = torch.cat([
            tokenized_1["attention_mask"].to(self.model.device),
            tokenized_2["attention_mask"].to(self.model.device)
        ], dim=-1)

        with torch.no_grad():
            response = self.model(
                input_ids, attention_mask=attention_mask,
                past_key_values=None, use_cache=True,
            )
            past_key_values = response.past_key_values

        past_length = tokenized_1["input_ids"].to(self.model.device).shape[1]
        shift_logits = response.logits[..., past_length - 1:-1, :].contiguous()
        shift_labels = input_ids[..., past_length:].contiguous()
        active = (attention_mask[:, past_length:] == 1).view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)
        res = loss.sum().item()
        return (res, past_key_values, shift_labels.shape[1])


# ============================================================
# 图构建 (来自 perplexity_tools.py)
# ============================================================
def create_graph_1(sentences, ch):
    """构建 PPL 全连通图"""
    ppl_list = []
    for sentence in sentences:
        sum_ppl, _, ppl_len = ch.get_ppl_for_next(" ", sentence)
        ppl_list.append(sum_ppl)

    n = len(sentences)
    graph = {i: {} for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i == j:
                graph[i][i] = ppl_list[i]
            else:
                sum_ppl, _, _ = ch.get_ppl_for_next(sentences[i], sentences[j])
                graph[i][j] = sum_ppl
    return graph


def create_graph_3(sentences, graph_1, delta, sentence_token_nums):
    """归一化边权重图"""
    n = len(sentences)
    graph = {i: {} for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i == j:
                graph[i][i] = 1
            else:
                exp_self = math.exp(graph_1[j][j] / sentence_token_nums[j])
                exp_cross = math.exp(graph_1[i][j] / sentence_token_nums[j])
                weight_temp = (exp_self - exp_cross) / exp_self
                weight = -weight_temp + 1 + delta * abs(i - j) / (n - 1)
                graph[i][j] = weight
    return graph


def create_graph_2(sentences, graph):
    """取一半边（全连接图的三角部分）"""
    n = len(sentences)
    g = {i: {} for i in range(n)}
    for i in range(n):
        for j in range(i, n):
            if i == j:
                g[i][i] = graph[i][i]
            else:
                g[i][j] = graph[i][j]
    return g


# ============================================================
# 结构熵计算
# ============================================================
def build_graph_from_edges(edges):
    """从边列表构建节点度分布"""
    degree = defaultdict(int)
    for edge in edges:
        degree[edge["row"]] += 1
        degree[edge["column"]] += 1
    return dict(degree)


def calculate_structural_entropy(graph):
    """计算结构熵"""
    total_degree = sum(graph.values())
    if total_degree == 0:
        return 0.0
    entropy = 0.0
    for degree in graph.values():
        if degree > 0:
            p = degree / total_degree
            entropy -= p * math.log(p, 2)
    return entropy


def find_edges_above_threshold(graph, threshold=0.8):
    """筛选权重 > threshold 的边"""
    edges = []
    for i, row in graph.items():
        for j, weight in row.items():
            if weight > threshold and i != j:
                edges.append({"row": i, "column": j, "value": weight})
    return edges


# ============================================================
# Dijkstra (来自 perplexity_tools.py)
# ============================================================
def dijkstra_3(graph, start):
    """带路径追踪的 Dijkstra"""
    distances = {node: float("infinity") for node in graph}
    distances[start] = 0
    numbers = {node: 0 for node in graph}

    pq = [(0, start)]
    path = {node: None for node in graph}
    path_temp = copy.deepcopy(path)

    while pq:
        current_dist, current_node = heapq.heappop(pq)
        if current_dist > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            if neighbor != current_node:
                new_dist = (current_dist * numbers[current_node] + weight) / (numbers[current_node] + 1)
                if new_dist < distances[neighbor]:
                    path_temp[neighbor] = current_node
                    if not _has_cycle(path_temp):
                        distances[neighbor] = new_dist
                        numbers[neighbor] = numbers[current_node] + 1
                        path[neighbor] = current_node
                        heapq.heappush(pq, (new_dist, neighbor))
                    else:
                        path_temp = copy.deepcopy(path)

    result = {}
    for znode, zdist in distances.items():
        if znode == start:
            continue
        shortest_path = []
        end = znode
        while end is not None:
            shortest_path.append(end)
            end = path[end]
        shortest_path.reverse()
        result[znode] = {"distance": zdist, "shortest_path": shortest_path}

    return result


def _has_cycle(path_dict):
    """检查路径是否有环"""
    visited = set()
    recursion_stack = set()

    def dfs(node):
        if node in recursion_stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        recursion_stack.add(node)
        neighbor = path_dict.get(node)
        if neighbor is not None and dfs(neighbor):
            return True
        recursion_stack.remove(node)
        return False

    for node in path_dict.keys():
        if dfs(node):
            return True
    return False


def remove_duplicate_edges(edges_dict):
    """去除重复的最短路径边"""
    seen = set()
    unique_edges = []
    for row_key, row_value in edges_dict.items():
        for col_key, value in row_value.items():
            for a, b in zip(value["shortest_path"], value["shortest_path"][1:]):
                key = (a, b)
                if key not in seen:
                    unique_edges.append({"row": a, "column": b})
                    seen.add(key)
    return unique_edges


# ============================================================
# 句子切分
# ============================================================
def split_into_sentences(text):
    """简单按句子切分（中文/英文）"""
    import re
    # 按常见句子结束符切分
    sents = re.split(r'([。！？\n]|\.(?=\s|$)|[.!?](?=\s|$))', text)
    # 合并句子和结束符
    merged = []
    i = 0
    while i < len(sents):
        if i + 1 < len(sents) and sents[i + 1] and sents[i + 1][0] in '。！？.!?':
            merged.append(sents[i] + sents[i + 1])
            i += 2
        elif sents[i].strip():
            merged.append(sents[i].strip())
            i += 1
        else:
            i += 1
    return [s for s in merged if len(s.strip()) >= 5]


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
        if len(text) >= 10:
            chunks.append(text)
    return chunks


def evaluate_chunk_stickiness(chunks, llm_model, llm_tokenizer, delta=0.0,
                              edge_threshold=0.8, max_chunks=None):
    """
    计算 Chunk Stickiness (结构熵)
    
    对每个 chunk:
      1. 切分成 sentences
      2. 构建 PPL 全连通图 (Graph_1)
      3. 归一化权重 (Graph_3)
      4. 筛选强边 (weight > threshold)
      5. 计算图的节点度结构熵
    
    返回: 平均结构熵 (越低越好)
    """
    ch = Chunking(llm_model, llm_tokenizer)
    entropies_g1 = []
    entropies_g3 = []
    skipped = 0

    chunks_to_eval = chunks if max_chunks is None else chunks[:max_chunks]
    print(f"\n将评估 {len(chunks_to_eval)} 个 chunks 的 Stickiness...")

    start_time = time.time()

    for ci, chunk_text in enumerate(tqdm(chunks_to_eval, desc="[Chunk Stickiness] 评估每个 chunk")):
        sentences = split_into_sentences(chunk_text)

        # 至少需要 3 个句子才值得建图
        if len(sentences) < 3:
            skipped += 1
            continue

        try:
            # Graph_1: PPL 全连通图
            token_nums = [
                llm_tokenizer.encode(s, return_tensors="pt").shape[1]
                for s in sentences
            ]

            graph_1 = create_graph_1(sentences, ch)

            # Graph_3: 归一化权重
            graph_3 = create_graph_3(sentences, graph_1, delta, token_nums)

            # 筛选强边并计算结构熵
            edges_g3 = find_edges_above_threshold(graph_3, edge_threshold)
            degree_g3 = build_graph_from_edges(edges_g3)
            entropy_g3 = calculate_structural_entropy(degree_g3)
            entropies_g3.append(entropy_g3)

            # Graph_2 版本 (全连接三角)
            graph_2 = create_graph_2(sentences, graph_1)
            edges_g2 = find_edges_above_threshold(graph_2, edge_threshold)
            degree_g2 = build_graph_from_edges(edges_g2)
            entropy_g2 = calculate_structural_entropy(degree_g2)
            entropies_g1.append(entropy_g2)

        except Exception as e:
            print(f"\n  [警告] chunk {ci} 评估失败: {e}")
            skipped += 1
            continue

        if (ci + 1) % 50 == 0:
            elapsed = time.time() - start_time
            avg_e = sum(entropies_g3) / len(entropies_g3) if entropies_g3 else 0
            print(f"\n  [{ci+1}/{len(chunks_to_eval)}] 当前 G3 平均结构熵: {avg_e:.4f} (耗时 {elapsed:.0f}s)")

    elapsed = time.time() - start_time

    # 汇总
    avg_g1 = sum(entropies_g1) / len(entropies_g1) if entropies_g1 else 0.0
    std_g1 = (sum((x - avg_g1) ** 2 for x in entropies_g1) / len(entropies_g1)) ** 0.5 if entropies_g1 else 0.0
    avg_g3 = sum(entropies_g3) / len(entropies_g3) if entropies_g3 else 0.0
    std_g3 = (sum((x - avg_g3) ** 2 for x in entropies_g3) / len(entropies_g3)) ** 0.5 if entropies_g3 else 0.0

    results = {
        "n_chunks_evaluated": len(entropies_g3),
        "n_chunks_skipped": skipped,
        "edge_threshold": edge_threshold,
        "delta": delta,
        "structural_entropy_G1": {
            "mean": avg_g1,
            "std": std_g1,
            "min": min(entropies_g1) if entropies_g1 else 0,
            "max": max(entropies_g1) if entropies_g1 else 0,
        },
        "structural_entropy_G3": {
            "mean": avg_g3,
            "std": std_g3,
            "min": min(entropies_g3) if entropies_g3 else 0,
            "max": max(entropies_g3) if entropies_g3 else 0,
        },
        "elapsed_seconds": elapsed,
    }

    return results, entropies_g3


def main():
    parser = argparse.ArgumentParser(description="Chunk Stickiness 评测")
    parser.add_argument("--chunks_dir", type=str,
                        default=r"../../../data/chunks_txt_integrated",
                        help="chunks .txt 文件目录")
    parser.add_argument("--output", type=str,
                        default="../../../data/chunk_stickiness_results.json",
                        help="结果输出路径")
    parser.add_argument("--max_chunks", type=int, default=None,
                        help="最多评估多少个 chunks (默认全部)")
    parser.add_argument("--edge_threshold", type=float, default=0.8,
                        help="边筛选阈值 (默认 0.8)")
    parser.add_argument("--delta", type=float, default=0.0,
                        help="Graph_3 距离惩罚参数 (默认 0.0)")
    parser.add_argument("--llm_model", type=str, default=LLM_MODEL_PATH,
                        help="LLM 模型路径")
    args = parser.parse_args()

    print("=" * 60)
    print("Chunk Stickiness 评测")
    print("=" * 60)
    print(f"  Chunks 目录: {args.chunks_dir}")
    print(f"  LLM 模型: {args.llm_model}")
    print(f"  设备: {DEVICE}")
    print(f"  边阈值: {args.edge_threshold}")
    print(f"  Delta: {args.delta}")

    # 加载 chunks
    print("\n[1/3] 加载 chunks...")
    chunks = load_chunks_from_dir(args.chunks_dir)
    print(f"  加载了 {len(chunks)} 个 chunks")

    # 加载模型
    print("\n[2/3] 加载 LLM...")
    print(f"  加载: {args.llm_model}")
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_model, trust_remote_code=True, device_map="auto"
    )
    llm_model.eval()
    print(f"  LLM 加载完成 ({DEVICE})")

    # 评测
    print("\n[3/3] 开始 Chunk Stickiness 评测...")
    results, entropies = evaluate_chunk_stickiness(
        chunks, llm_model, llm_tokenizer,
        delta=args.delta,
        edge_threshold=args.edge_threshold,
        max_chunks=args.max_chunks,
    )

    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存到: {args.output}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("Chunk Stickiness 评测结果摘要")
    print("=" * 60)
    print(f"  评估 chunks 数: {results['n_chunks_evaluated']}")
    print(f"  跳过 chunks 数: {results['n_chunks_skipped']}")
    print(f"  结构熵 G1 (全连接三角图): {results['structural_entropy_G1']['mean']:.4f} "
          f"± {results['structural_entropy_G1']['std']:.4f}")
    print(f"  结构熵 G3 (归一化权重图): {results['structural_entropy_G3']['mean']:.4f} "
          f"± {results['structural_entropy_G3']['std']:.4f}")
    print(f"  总耗时: {results['elapsed_seconds']:.0f}s")
    print("\n  [解读] 结构熵越低 → chunk 内部句子越抱团 → Stickiness 越高")
    print("=" * 60)


if __name__ == "__main__":
    main()
