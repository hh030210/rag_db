# -*- coding: utf-8 -*-
"""
semantic_searcher.py
====================
纯语义检索器：直接对 chunk_text_vec 做向量 ANN。

不依赖任何维度 / 标签层，纯向量召回。
"""

from typing import Any, Dict, List, Optional

from qdrant_client import QdrantHTTPClient


class SemanticSearcher:
    """纯语义检索（BGE-M3 → chunk_text_vec）"""

    def __init__(self, client: QdrantHTTPClient, collection_name: str, encoder):
        self.client = client
        self.collection_name = collection_name
        self.encoder = encoder

    def search(self, query_text: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """返回 list[dict]，每条含 chunk_id / score / sem_rank / source='sem' / text 字段"""
        from bge_encoder import encode_query

        qv = encode_query(self.encoder, query_text)
        if qv is None:
            return []

        hits = self.client.search(
            collection=self.collection_name,
            query_vector=qv,
            vector_name="chunk_text_vec",
            top_k=top_k,
        )

        out = []
        for i, hit in enumerate(hits):
            payload = hit.get("payload") or {}
            cid = payload.get("chunk_id") or hit.get("id")
            out.append({
                "chunk_id": cid,
                "chunk_text": payload.get("chunk_text", ""),
                "chunk_text_full": payload.get("chunk_text", ""),
                "doc_title": payload.get("doc_title", ""),
                "chunk_gen_title": payload.get("chunk_gen_title", ""),
                "score": float(hit.get("score", 0.0)),
                "sem_rank": i + 1,
                "source": "sem",
            })
        return out
