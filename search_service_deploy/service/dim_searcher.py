# -*- coding: utf-8 -*-
"""
dim_searcher.py
===============
维度检索编排器。两阶段：

    粗召回（3 种方法）→ 截断 top DIM_RERANK_TOP_K → 精排（2 种方法）→ 内部融合（2 种）

返回 list[dict]：dim candidates，每条含 score / dim_rank / recall_score / rerank_score / tag_hits / evidence。
"""

from typing import Any, Dict, List, Optional, Tuple

from dim_recall import recall_by_vec, recall_by_constraint, recall_by_tag
from dim_rerank import rerank_dim


class DimSearcher:
    def __init__(
        self,
        client,
        chunk_collection: str,
        dim_tags_collection: str,
        encoder,
        recall_method: str = "tag",
        rerank_method: str = "tag_sim",
        internal_fusion: str = "score",
        dim_top_k: int = 100,
        dim_rerank_top_k: int = 50,
        rrf_k: int = 60,
        tag_vectors: Optional[Dict[tuple, Any]] = None,
        parse_constraints_fn=None,
    ):
        self.client = client
        self.chunk_collection = chunk_collection
        self.dim_tags_collection = dim_tags_collection
        self.encoder = encoder
        self.recall_method = recall_method
        self.rerank_method = rerank_method
        self.internal_fusion = internal_fusion
        self.dim_top_k = dim_top_k
        self.dim_rerank_top_k = dim_rerank_top_k
        self.rrf_k = rrf_k
        self.tag_vectors = tag_vectors or {}
        self.parse_constraints_fn = parse_constraints_fn

    def search(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """返回 (dim_results, constraints)"""
        from bge_encoder import encode_query

        qv = encode_query(self.encoder, query_text)
        if qv is None:
            return [], {}

        # ── 第一阶段：粗召回 ──
        if self.recall_method == "vec":
            raw, hits = recall_by_vec(self.client, self.chunk_collection, qv, self.dim_top_k)
        elif self.recall_method == "constraint":
            raw, hits = recall_by_constraint(self.client, self.dim_tags_collection, query_text, self.dim_top_k)
        elif self.recall_method == "tag":
            raw, hits = recall_by_tag(self.client, self.dim_tags_collection, self.chunk_collection, qv, self.dim_top_k)
        else:
            raw, hits = recall_by_tag(self.client, self.dim_tags_collection, self.chunk_collection, qv, self.dim_top_k)

        if not raw:
            return [], {}

        # 粗排截断
        truncated = sorted(raw, key=lambda x: x.get("recall_score", 0.0), reverse=True)[:self.dim_rerank_top_k]

        # ── 第二阶段：精排 ──
        reranked = rerank_dim(
            query_vec=qv,
            query_text=query_text,
            raw_candidates=truncated,
            recalled_tag_hits=hits,
            top_k=top_k,
            rerank_method=self.rerank_method,
            internal_fusion=self.internal_fusion,
            tag_vectors=self.tag_vectors,
            parse_constraints_fn=self.parse_constraints_fn,
            rrf_k=self.rrf_k,
        )

        # ── 收集 constraints ──
        constraints: Dict[str, set] = {}
        for r in raw:
            dm = r.get("dim_name") or ""
            tn = r.get("tag_name") or ""
            if dm:
                constraints.setdefault(dm, set()).add(tn)
        constraints_out = {k: sorted([x for x in v if x])[:8] for k, v in constraints.items()}

        return reranked, constraints_out
