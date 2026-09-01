# -*- coding: utf-8 -*-
"""
fusion.py
=========
顶层融合：

    1. 静态融合：
        - rrf   : RRF(dim, sem, k=60) alpha 控制 dim 路权重
        - score : alpha*dim + (1-alpha)*sem (归一化后再加权)

    2. 自适应融合（compute_adaptive_weights）：
        - 综合结构化置信度 P_q
        - 标签证据置信度 T_q
        - 单路集中度 C_dim / C_sem
        - 效用 U = P + T + C (or 1-P + 1-T + C)，softmax 后作为最终 α
"""

import math
from typing import Any, Dict, List, Optional, Tuple


# ===================== 自适应权重组件 =====================

def compute_structural_confidence(constraints: Dict[str, Any]) -> float:
    """P_q = 2*n_b / (n_d + n_b + 1)
    n_d: 有效维度数
    n_b: 至少绑定了一个值的维度数
    """
    if not constraints:
        return 0.0
    real = {k: v for k, v in constraints.items() if not k.startswith("_")}
    n_d = len(real)
    if n_d == 0:
        return 0.0
    n_b = sum(1 for v in real.values() if v and len(v) > 0)
    return (2 * n_b) / (n_d + n_b + 1)


def compute_label_evidence_confidence(
    constraints: Dict[str, Any],
    dim_results: List[Dict[str, Any]],
    tag_vector_scores: Optional[Dict[str, float]] = None,
) -> float:
    """T_q = 2*E_T / (|M_q| + E_T + 1)"""
    if not constraints:
        return 0.0
    real = {k: v for k, v in constraints.items() if not k.startswith("_")}
    M_q = list(real.keys())
    if not M_q:
        return 0.0

    dim_hit_tags: Dict[str, set] = {}
    for r in dim_results:
        dn = r.get("dim_name") or ""
        tn = r.get("tag_name") or ""
        if dn and tn:
            dim_hit_tags.setdefault(dn, set()).add(tn)

    E_T = 0.0
    for m in M_q:
        query_tags = set(real.get(m, []))
        hit_tags = dim_hit_tags.get(m, set())
        if not query_tags or not hit_tags:
            continue
        if tag_vector_scores is not None:
            s_m = max(
                (tag_vector_scores.get(t, 1.0) for t in hit_tags if t in query_tags),
                default=0.0,
            )
        else:
            overlap = query_tags & hit_tags
            if overlap:
                s_m = max(
                    (r.get("score", 0.0) for r in dim_results
                     if r.get("dim_name") == m and r.get("tag_name") in overlap),
                    default=1.0,
                )
            else:
                s_m = 0.0
        E_T += s_m

    return (2 * E_T) / (len(M_q) + E_T + 1)


def compute_concentration(
    dim_results: List[Dict[str, Any]],
    sem_results: List[Dict[str, Any]],
) -> Tuple[float, float]:
    """C_r = 1 - H(p)/log|K|，归一化熵"""
    def _cr(results: List[Dict[str, Any]]) -> float:
        if not results:
            return 0.0
        scores = [r.get("score", 0.0) for r in results]
        total = sum(scores)
        if total <= 0 or len(scores) <= 1:
            return 1.0 if results else 0.0
        n = len(scores)
        h = 0.0
        for s in scores:
            if s > 0:
                p = s / total
                h -= p * math.log(p)
        h_norm = h / math.log(n) if n > 1 else 0.0
        return max(0.0, 1.0 - h_norm)
    return _cr(dim_results), _cr(sem_results)


def compute_adaptive_weights(
    dim_results: List[Dict[str, Any]],
    sem_results: List[Dict[str, Any]],
    constraints: Dict[str, Any],
    top_k: int = 5,
) -> Tuple[float, float]:
    """返回 (alpha_dim, alpha_sem)"""
    P_q = compute_structural_confidence(constraints)
    T_q = compute_label_evidence_confidence(constraints, dim_results)
    C_dim, C_sem = compute_concentration(dim_results[:top_k], sem_results[:top_k])

    U_dim = P_q + T_q + C_dim
    U_sem = (1 - P_q) + (1 - T_q) + C_sem

    eps = 1e-6
    a_dim = U_dim / (U_dim + U_sem + eps)
    return a_dim, 1.0 - a_dim


# ===================== 顶层融合 =====================

def _normalize_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """把每个 chunk 的 score 在 0~1 之间归一化"""
    scores = [r.get("score", 0.0) for r in results]
    if not scores:
        return {}
    s_min, s_max = min(scores), max(scores)
    if s_max == s_min:
        return {r["chunk_id"]: 1.0 for r in results if r.get("chunk_id")}
    out = {}
    for r in results:
        cid = r.get("chunk_id")
        if cid:
            out[cid] = (r.get("score", 0.0) - s_min) / (s_max - s_min)
    return out


def fuse_rrf(
    dim_results: List[Dict[str, Any]],
    sem_results: List[Dict[str, Any]],
    top_k: int = 8,
    alpha: float = 0.5,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """RRF 融合：score = alpha/(k+rank_dim) + (1-alpha)/(k+rank_sem)"""
    scores: Dict[str, float] = {}
    records: Dict[str, Dict[str, Any]] = {}

    for i, r in enumerate(dim_results or []):
        cid = r.get("chunk_id")
        if not cid:
            continue
        scores[cid] = scores.get(cid, 0.0) + alpha / (rrf_k + i + 1)
        records[cid] = r
        records[cid]["dim_rank"] = i + 1

    for i, r in enumerate(sem_results or []):
        cid = r.get("chunk_id")
        if not cid:
            continue
        scores[cid] = scores.get(cid, 0.0) + (1 - alpha) / (rrf_k + i + 1)
        if cid not in records:
            records[cid] = r
        records[cid]["sem_rank"] = i + 1

    sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    out = []
    for rank, cid in enumerate(sorted_cids[:top_k], 1):
        merged = dict(records[cid])
        merged["final_score"] = float(scores[cid])
        merged["fusion_rank"] = rank
        if "dim_rank" in merged and "sem_rank" in merged:
            merged["source"] = "dim+sem"
        elif "dim_rank" in merged:
            merged["source"] = "dim"
        else:
            merged["source"] = "sem"
        out.append(merged)
    return out


def fuse_score(
    dim_results: List[Dict[str, Any]],
    sem_results: List[Dict[str, Any]],
    top_k: int = 8,
    alpha: float = 0.5,
) -> List[Dict[str, Any]]:
    """线性加权：score = alpha*norm(dim) + (1-alpha)*norm(sem)"""
    dim_norm = _normalize_scores(dim_results or [])
    sem_norm = _normalize_scores(sem_results or [])

    scores: Dict[str, float] = {}
    records: Dict[str, Dict[str, Any]] = {}

    for i, r in enumerate(dim_results or []):
        cid = r.get("chunk_id")
        if not cid:
            continue
        scores[cid] = scores.get(cid, 0.0) + alpha * dim_norm.get(cid, 0.0)
        records[cid] = r
        records[cid]["dim_rank"] = i + 1

    for i, r in enumerate(sem_results or []):
        cid = r.get("chunk_id")
        if not cid:
            continue
        scores[cid] = scores.get(cid, 0.0) + (1 - alpha) * sem_norm.get(cid, 0.0)
        if cid not in records:
            records[cid] = r
        records[cid]["sem_rank"] = i + 1

    sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    out = []
    for rank, cid in enumerate(sorted_cids[:top_k], 1):
        merged = dict(records[cid])
        merged["final_score"] = float(scores[cid])
        merged["fusion_rank"] = rank
        if "dim_rank" in merged and "sem_rank" in merged:
            merged["source"] = "dim+sem"
        elif "dim_rank" in merged:
            merged["source"] = "dim"
        else:
            merged["source"] = "sem"
        out.append(merged)
    return out


def fuse_two(
    dim_results: List[Dict[str, Any]],
    sem_results: List[Dict[str, Any]],
    alpha: float = 0.5,
    top_k: int = 8,
    strategy: str = "rrf",
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    if strategy == "score":
        return fuse_score(dim_results, sem_results, top_k=top_k, alpha=alpha)
    return fuse_rrf(dim_results, sem_results, top_k=top_k, alpha=alpha, rrf_k=rrf_k)
