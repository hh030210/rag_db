# -*- coding: utf-8 -*-
"""
qdrant_client.py
================
极简 Qdrant HTTP 客户端（兼容 Qdrant 1.18+）。

实现：
    - search(): /points/query (named vector 友好) → 回退 /points/search
    - scroll(): /points/scroll
    - retrieve(): 通过 filter + scroll 实现（避开字符串 point id）

所有外部依赖：httpx。
"""

from typing import List, Optional

import httpx


class QdrantHTTPClient:
    """轻量 Qdrant HTTP 客户端，只实现 search + scroll + retrieve"""

    def __init__(self, host: str, port: int):
        self.base = f"http://{host}:{port}"

    # ──────────────── POST helper ────────────────

    def _post(self, path: str, body: dict, timeout: float = 30.0) -> dict:
        r = httpx.post(self.base + path, json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()

    # ──────────────── search ────────────────

    def search(
        self,
        collection: str,
        query_vector: List[float],
        vector_name: str = "chunk_text_vec",
        top_k: int = 20,
        score_threshold: Optional[float] = None,
        with_payload: bool = True,
    ) -> List[dict]:
        """向量检索（named vector 友好版）

        优先使用 /points/query，回退到 /points/search。
        返回 list[dict]，每个点含 id / score / payload 字段。
        """
        body = {
            "query": query_vector,
            "limit": top_k,
            "with_payload": with_payload,
            "with_vector": False,
        }
        if vector_name:
            body["using"] = vector_name
        if score_threshold is not None:
            body["score_threshold"] = score_threshold

        try:
            data = self._post(f"/collections/{collection}/points/query", body)
            return data.get("result", {}).get("points", [])
        except Exception:
            fallback = {
                "vector": query_vector,
                "limit": top_k,
                "with_payload": with_payload,
                "with_vector": False,
            }
            if vector_name:
                fallback["vector_name"] = vector_name
            try:
                data = self._post(f"/collections/{collection}/points/search", fallback)
                return data.get("result", [])
            except Exception:
                return []

    # ──────────────── scroll ────────────────

    def scroll(
        self,
        collection: str,
        limit: int = 100,
        with_payload: bool = True,
        body: Optional[dict] = None,
    ) -> List[dict]:
        """滚动取出 collection 内点；body 用于带 filter 检索（按 dim/tag 精确匹配）"""
        if body is None:
            body = {"limit": limit, "with_payload": with_payload, "with_vector": False}
        data = self._post(f"/collections/{collection}/points/scroll", body)
        return data.get("result", {}).get("points", [])

    # ──────────────── retrieve ────────────────

    def retrieve(self, collection: str, ids: List[str], with_payload: bool = True) -> List[dict]:
        """通过 filter + scroll 按 chunk_id 字段取值（Qdrant 1.18 不支持字符串 point id）"""
        if not ids:
            return []
        BATCH = 64
        out = []
        for i in range(0, len(ids), BATCH):
            chunk = ids[i : i + BATCH]
            body = {
                "filter": {"must": [{"key": "chunk_id", "match": {"any": chunk}}]},
                "limit": len(chunk) * 2,
                "with_payload": with_payload,
                "with_vector": False,
            }
            data = self._post(f"/collections/{collection}/points/scroll", body)
            out.extend(data.get("result", {}).get("points", []))
        return out

    # ──────────────── ensure ────────────────

    def health(self) -> bool:
        try:
            r = httpx.get(self.base + "/healthz", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False
