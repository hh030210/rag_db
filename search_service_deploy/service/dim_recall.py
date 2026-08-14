# -*- coding: utf-8 -*-
"""
dim_recall.py
=============
维度检索的"粗召回"阶段，三种方法：

    1. vec         query 向量 ⇨ 文档向量  （最弱语义）
    2. constraint  query 维度 ⇨ D 维度     （需 LLM 解析，本服务不依赖）
    3. tag         query 向量 ⇨ 维度标签向量 （默认）

返回 (candidates, recalled_tag_hits)
"""

from typing import Any, Dict, List, Tuple


def _normalize(text: str) -> str:
    if not text:
        return ""
    return str(text).strip().replace("\u3000", " ").replace("  ", " ")


# ──────────────── 共用：基于 tag_hit 累加 candidate ────────────────

def _merge_candidate(hit: dict, candidates: Dict[str, dict]) -> None:
    payload = hit.get("payload") or {}
    chunk_ids = payload.get("chunk_ids") or []
    if not isinstance(chunk_ids, list):
        return
    tag_score = float(hit.get("score", 0.0))
    for cid in chunk_ids:
        if not cid:
            continue
        cid_norm = _normalize(cid)
        if cid_norm not in candidates:
            candidates[cid_norm] = {
                "chunk_id": cid,
                "dim_score": tag_score,
                "tag_name": payload.get("tag_name", ""),
                "dim_name": payload.get("dim_name", ""),
                "tag_hits": [],
            }
        candidates[cid_norm]["tag_hits"].append({
            "tag": payload.get("tag_name", ""),
            "dim": payload.get("dim_name", ""),
            "score": tag_score,
        })
        if tag_score > candidates[cid_norm]["dim_score"]:
            candidates[cid_norm]["dim_score"] = tag_score


def _build_candidates(
    payloads: List[dict],
    candidates: Dict[str, dict],
    recall_method: str,
) -> List[Dict[str, Any]]:
    """合并 payload + candidate meta，组装粗召回候选"""
    out = []
    for pt in payloads:
        pl = pt.get("payload") or {}
        cid = pl.get("chunk_id")
        meta = candidates.get(_normalize(cid), {}) if cid else {}
        if not meta or not cid:
            continue
        out.append({
            "chunk_id": cid,
            "chunk_text": pl.get("chunk_text", ""),
            "chunk_text_full": pl.get("chunk_text", ""),
            "doc_title": pl.get("doc_title", ""),
            "chunk_gen_title": pl.get("chunk_gen_title", ""),
            "recall_score": float(meta.get("dim_score", 0.0)),
            "score": float(meta.get("dim_score", 0.0)),
            "dim_rank": 0,
            "source": "dim",
            "recall_method": recall_method,
            "tag_name": meta.get("tag_name", ""),
            "dim_name": meta.get("dim_name", ""),
            "tag_hits": meta.get("tag_hits", []),
            "evidence": [f"{h['dim']}:{h['tag']}" for h in meta.get("tag_hits", [])][:3],
        })

    out.sort(key=lambda x: x["recall_score"], reverse=True)
    for i, r in enumerate(out):
        r["dim_rank"] = i + 1
    return out


# ──────────────── 方法一：vec ────────────────

def recall_by_vec(client, collection: str, query_vec: List[float], top_k: int) -> Tuple[list, list]:
    """方法一：query 向量 → 文档向量（仅用 chunk collection）"""
    hits = client.search(
        collection=collection,
        query_vector=query_vec,
        vector_name="chunk_text_vec",
        top_k=top_k,
    )
    candidates = []
    for i, hit in enumerate(hits):
        payload = hit.get("payload") or {}
        cid = payload.get("chunk_id") or hit.get("id")
        candidates.append({
            "chunk_id": cid,
            "chunk_text": payload.get("chunk_text", ""),
            "chunk_text_full": payload.get("chunk_text", ""),
            "doc_title": payload.get("doc_title", ""),
            "chunk_gen_title": payload.get("chunk_gen_title", ""),
            "recall_score": float(hit.get("score", 0.0)),
            "dim_rank": i + 1,
            "source": "dim",
            "recall_method": "vec",
            "tag_name": "",
            "dim_name": "",
            "tag_hits": [],
            "evidence": [f"vec_score={hit.get('score', 0):.4f}"],
        })
    return candidates, []


# ──────────────── 方法二：constraint (默认 no-op) ────────────────

def recall_by_constraint(_client, _collection: str, _query_text: str, _top_k: int) -> Tuple[list, list]:
    """方法二：需要 LLM 解析维度约束；本服务默认禁用，返回空集合。"""
    return [], []


# ──────────────── 方法三：tag (默认) ────────────────

def recall_by_tag(
    client,
    dim_tags_collection: str,
    chunk_collection: str,
    query_vec: List[float],
    top_k: int,
) -> Tuple[list, list]:
    """方法三：query 向量 → 维度标签向量 → 反查 chunk（推荐）"""
    tag_hits = client.search(
        collection=dim_tags_collection,
        query_vector=query_vec,
        vector_name="chunk_text_vec",
        top_k=top_k * 2,
    )
    if not tag_hits:
        return [], []

    all_candidates: Dict[str, dict] = {}
    for hit in tag_hits:
        _merge_candidate(hit, all_candidates)
    if not all_candidates:
        return [], []

    payloads = client.retrieve(chunk_collection, list(all_candidates.keys()))
    return _build_candidates(payloads, all_candidates, "tag"), tag_hits
