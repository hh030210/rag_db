# -*- coding: utf-8 -*-
"""
dim_rerank.py
=============
维度检索的"精排"阶段，两种方法：

    1. tag_sim    S(q,D) = (1/|A_D|) * sum_{(m,t) in A_D} sim(v_q, v_{m,t})
    2. tag_match  S(q,D) = sum_{m in M_q} max_{t_q,t_D} sim(v_{t_q}, v_{t_D})

内部融合：
    - rrf   : 粗排 RRF + 精排归一化得分
    - score : 精排得分直接覆盖粗排得分
"""

import math
import threading
from typing import Any, Dict, List, Optional


# -------- 精排方法一：tag_sim --------

def _rerank_by_tag_sim(
    query_vec: List[float],
    raw_candidates: List[dict],
    recalled_tag_hits: List[dict],
    tag_vectors: Dict[tuple, Any],
) -> Dict[str, float]:
    """S(q,D) = (1/|A_D|) * sum_{(m,t) in A_D} sim(v_q, v_{m,t})"""
    if not raw_candidates:
        return {}

    import numpy as np
    q_arr = np.array(query_vec)

    rerank_scores: Dict[str, float] = {}
    for c in raw_candidates:
        cid = c.get("chunk_id", "")
        if not cid:
            continue
        tag_hits = c.get("tag_hits", [])
        if not tag_hits:
            rerank_scores[cid] = c.get("recall_score", 0.0)
            continue

        tag_sim_sum = 0.0
        tag_count = 0
        for th in tag_hits:
            dm = th.get("dim", "")
            tn = th.get("tag", "")
            score = th.get("score", 0.0)
            key = (dm, tn)
            tag_vec = tag_vectors.get(key)

            if tag_vec is not None and isinstance(tag_vec, np.ndarray):
                denom = (np.linalg.norm(q_arr) * np.linalg.norm(tag_vec) + 1e-8)
                tag_sim = float(np.dot(q_arr, tag_vec) / denom)
            else:
                tag_sim = score

            tag_sim_sum += tag_sim
            tag_count += 1

        rerank_scores[cid] = tag_sim_sum / tag_count if tag_count > 0 else c.get("recall_score", 0.0)

    return rerank_scores


# -------- 精排方法二：tag_match --------

def _rerank_by_tag_match(
    query_text: str,
    raw_candidates: List[dict],
    tag_vectors: Dict[tuple, Any],
    parse_constraints_fn,
) -> Dict[str, float]:
    """S(q,D) = sum_{m in M_q} max_{t_q,t_D} sim(v_{t_q}, v_{t_D})"""
    if not raw_candidates:
        return {}

    constraints = parse_constraints_fn(query_text) if parse_constraints_fn else {}
    real_dims = {k: v for k, v in (constraints or {}).items() if not k.startswith("_")}
    if not real_dims:
        return {c["chunk_id"]: c.get("recall_score", 0.0) for c in raw_candidates if c.get("chunk_id")}

    import numpy as np

    rerank_scores: Dict[str, float] = {}
    for c in raw_candidates:
        cid = c.get("chunk_id", "")
        if not cid:
            continue

        tag_hits = c.get("tag_hits", [])
        doc_tags_by_dim: Dict[str, set] = {}
        for th in tag_hits:
            dm = th.get("dim", "")
            tn = th.get("tag", "")
            if dm:
                doc_tags_by_dim.setdefault(dm, set()).add(tn)

        dim_score_sum = 0.0
        for dim_name, query_tags in real_dims.items():
            if not query_tags:
                continue
            doc_tags = doc_tags_by_dim.get(dim_name, set())
            if not doc_tags:
                continue

            best_sim = 0.0
            for t_q in query_tags:
                for t_D in doc_tags:
                    if tag_vectors:
                        key_q = (dim_name, t_q)
                        key_D = (dim_name, t_D)
                        if key_q in tag_vectors and key_D in tag_vectors:
                            v_q = np.array(tag_vectors[key_q])
                            v_D = np.array(tag_vectors[key_D])
                            denom = (np.linalg.norm(v_q) * np.linalg.norm(v_D) + 1e-8)
                            sim = float(np.dot(v_q, v_D) / denom)
                        else:
                            sim = 1.0 if t_q == t_D else 0.0
                    else:
                        sim = 1.0 if t_q == t_D else 0.0
                    if sim > best_sim:
                        best_sim = sim
            dim_score_sum += best_sim

        rerank_scores[cid] = dim_score_sum

    return rerank_scores


# -------- 内部融合：rrf --------

def _internal_rrf(
    raw_candidates: List[dict],
    rerank_scores: Dict[str, float],
    top_k: int,
    rrf_k: int = 60,
) -> List[dict]:
    combined: Dict[str, float] = {}
    for i, c in enumerate(raw_candidates):
        cid = c.get("chunk_id", "")
        if not cid:
            continue
        combined[cid] = combined.get(cid, 0) + 1.0 / (rrf_k + i + 1)

    max_rerank = max(rerank_scores.values()) if rerank_scores else 1.0
    for cid, rs in rerank_scores.items():
        combined[cid] = combined.get(cid, 0) + (rs / max_rerank)

    sorted_cids = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)
    by_cid = {c["chunk_id"]: c for c in raw_candidates if c.get("chunk_id")}
    out = []
    for rank, cid in enumerate(sorted_cids[:top_k], 1):
        if cid in by_cid:
            r = dict(by_cid[cid])
            r["score"] = combined[cid]
            r["dim_rank"] = rank
            r["recall_score"] = r.get("recall_score", 0.0)
            r["rerank_score"] = rerank_scores.get(cid, 0.0)
            r["internal_fusion"] = "rrf"
            out.append(r)
    return out


# -------- 内部融合：score --------

def _replace_with_rerank(
    raw_candidates: List[dict],
    rerank_scores: Dict[str, float],
    top_k: int,
) -> List[dict]:
    by_cid = {}
    for c in raw_candidates:
        if c.get("chunk_id"):
            by_cid[c["chunk_id"]] = dict(c)
            by_cid[c["chunk_id"]]["rerank_score"] = rerank_scores.get(c["chunk_id"], c.get("recall_score", 0.0))
            by_cid[c["chunk_id"]]["score"] = by_cid[c["chunk_id"]]["rerank_score"]
            by_cid[c["chunk_id"]]["internal_fusion"] = "score"
    out = sorted(by_cid.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    for i, r in enumerate(out):
        r["dim_rank"] = i + 1
    return out


# -------- 公开入口 --------

def rerank_dim(
    query_vec: List[float],
    query_text: str,
    raw_candidates: List[dict],
    recalled_tag_hits: List[dict],
    top_k: int,
    rerank_method: str = "tag_sim",
    internal_fusion: str = "score",
    tag_vectors: Optional[Dict[tuple, Any]] = None,
    parse_constraints_fn=None,
    rrf_k: int = 60,
) -> List[dict]:
    """执行精排 + 内部融合，返回最终 dim 候选。"""
    if not raw_candidates:
        return []

    tag_vectors = tag_vectors or {}
    if rerank_method == "tag_sim":
        rerank_scores = _rerank_by_tag_sim(query_vec, raw_candidates, recalled_tag_hits, tag_vectors)
    elif rerank_method == "tag_match":
        rerank_scores = _rerank_by_tag_match(query_text, raw_candidates, tag_vectors, parse_constraints_fn)
    else:
        rerank_scores = {c["chunk_id"]: c.get("recall_score", 0.0) for c in raw_candidates if c.get("chunk_id")}

    if internal_fusion == "rrf":
        return _internal_rrf(raw_candidates, rerank_scores, top_k, rrf_k=rrf_k)
    return _replace_with_rerank(raw_candidates, rerank_scores, top_k)
