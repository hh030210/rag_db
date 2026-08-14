"""
检索融合引擎

融合 RAG_DB_slim 的维度感知检索和语义检索能力：
- 语义检索：基于向量相似度
- 维度检索：基于维度标签过滤
- RRF 融合：将维度结果和语义结果按排名融合

同时支持直接 Qdrant 检索作为降级方案。
"""

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 尝试加载 experiment_data 中的索引文件（RAG_DB_slim 的维度索引）
EXPERIMENT_DATA = _project_root / "experiment_data"
PATH_INVERTED_INDEX = EXPERIMENT_DATA / "inverted_index.json"
PATH_DIM_META = EXPERIMENT_DATA / "dimension_metadata.json"
PATH_TAG_VECTORS = EXPERIMENT_DATA / "tag_vectors.pkl"


class RetrievalConfig:
    """检索配置"""
    VECTOR_TOP_K = 200
    RRF_K = 60
    DEFAULT_ALPHA = 0.2       # 0=纯语义, 1=纯维度
    SOFT_MATCH_THRESHOLD = 0.65
    SEM_TOP_K = 20
    DIM_TOP_K = 100
    EXCLUDED_DIMS: Set[str] = set()


def rrf_fuse(
    dim_results: List[Dict],
    sem_results: List[Dict],
    k: int = 60,
    alpha: float = 0.5,
) -> List[Dict]:
    """
    RRF（Reciprocal Rank Fusion）融合

    Args:
        dim_results: 维度检索结果列表，每项含 chunk_id 和 score
        sem_results: 语义检索结果列表，每项含 chunk_id 和 score
        k: RRF 参数
        alpha: 权重因子（alpha * dim + (1-alpha) * sem）
    """
    scores: Dict[str, float] = {}

    for i, r in enumerate(dim_results):
        cid = r.get("chunk_id") or r.get("id", "")
        scores[cid] = scores.get(cid, 0) + alpha * (1.0 / (k + i + 1))

    for i, r in enumerate(sem_results):
        cid = r.get("chunk_id") or r.get("id", "")
        scores[cid] = scores.get(cid, 0) + (1 - alpha) * (1.0 / (k + i + 1))

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [{"chunk_id": cid, "fused_score": scores[cid]} for cid in sorted_ids]


class RetrievalEngine:
    """
    检索融合引擎

    支持三种检索模式：
    - "semantic": 纯语义向量检索
    - "dimension": 维度过滤检索（需要索引文件）
    - "fusion": RRF 融合检索（默认）
    """

    def __init__(self, qdrant_client, embedding_service):
        """
        Args:
            qdrant_client: QdrantClient 实例
            embedding_service: EmbeddingService 实例
        """
        self.qdrant = qdrant_client
        self.embedding = embedding_service
        self.config = RetrievalConfig()

        # 加载维度索引（RAG_DB_slim 的维度感知能力）
        self.inverted_index: Dict[str, List[str]] = {}
        self.dim_meta: Dict[str, Dict] = {}
        self.tag_vectors: Dict[str, List[float]] = {}

        self._load_indexes()

    def _load_indexes(self):
        """加载维度索引文件"""
        if PATH_INVERTED_INDEX.exists():
            with open(PATH_INVERTED_INDEX, "r", encoding="utf-8") as f:
                self.inverted_index = json.load(f)
            print(f"[检索] 倒排索引加载: {len(self.inverted_index)} 个维度")

        if PATH_DIM_META.exists():
            with open(PATH_DIM_META, "r", encoding="utf-8") as f:
                self.dim_meta = json.load(f)
            print(f"[检索] 维度元数据加载: {len(self.dim_meta)} 个维度")

        if PATH_TAG_VECTORS.exists():
            with open(PATH_TAG_VECTORS, "rb") as f:
                self.tag_vectors = pickle.load(f)
            print(f"[检索] 标签向量加载: {len(self.tag_vectors)} 个维度")

    # ==================== 维度解析 ====================

    def _parse_dimensions(self, query: str) -> Dict[str, Any]:
        """
        从查询中解析维度约束
        这是简化版本，实际可接入 code/dimension_search.py 的更复杂逻辑
        """
        dims = {}
        query_lower = query.lower()

        # 简单的关键词匹配（实际应复用 RAG_DB_slim 的维度解析逻辑）
        for dim_name, dim_info in self.dim_meta.items():
            keywords = dim_info.get("keywords", [])
            if not keywords:
                keywords = [dim_name]

            for kw in keywords:
                if kw.lower() in query_lower:
                    dims[dim_name] = {"match": kw, "score": 1.0}
                    break

        return dims

    # ==================== 语义检索 ====================

    def search_semantic(
        self,
        query: str,
        top_k: int = 20,
        collection_name: str = None,
        vector_field: str = "chunk_text_vec",
    ) -> List[Dict]:
        """纯语义向量检索"""
        if not self.qdrant or not self.qdrant.is_connected():
            return []

        try:
            query_vec = self.embedding.encode_query(query, model="bgem3")

            results = self.qdrant.search(
                query_vector=query_vec,
                collection_name=collection_name,
                limit=top_k,
                using=vector_field,
                with_payload=True,
            )

            output = []
            for hit in results:
                item = {
                    "chunk_id": hit["id"],
                    "score": hit["score"],
                    "payload": hit.get("payload", {}),
                    "chunk_text": hit.get("payload", {}).get("chunk_text", ""),
                    "doc_title": hit.get("payload", {}).get("doc_title", ""),
                    "chunk_gen_title": hit.get("payload", {}).get("chunk_gen_title", ""),
                }
                output.append(item)

            return output

        except Exception as e:
            print(f"[检索] 语义检索失败: {e}")
            return []

    # ==================== 维度检索 ====================

    def search_dimension(
        self,
        query: str,
        top_k: int = 100,
        collection_name: str = None,
    ) -> List[Dict]:
        """维度过滤检索"""
        if not self.qdrant or not self.qdrant.is_connected():
            return []

        if not self.inverted_index:
            return []

        dims = self._parse_dimensions(query)
        if not dims:
            return []

        try:
            matched_chunk_ids: Set[str] = set()
            for dim_name, dim_info in dims.items():
                if dim_name in self.inverted_index:
                    chunk_ids = self.inverted_index[dim_name]
                    if not matched_chunk_ids:
                        matched_chunk_ids = set(chunk_ids)
                    else:
                        matched_chunk_ids &= set(chunk_ids)

            if not matched_chunk_ids:
                return []

            # 批量获取这些 chunk 的向量
            # Qdrant scroll 获取所有点，筛选
            matched_ids = list(matched_chunk_ids)[:top_k]
            return [{"chunk_id": cid, "dim_score": 1.0} for cid in matched_ids]

        except Exception as e:
            print(f"[检索] 维度检索失败: {e}")
            return []

    # ==================== 融合检索（主入口）====================

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "fusion",
        alpha: float = None,
        collection_name: str = None,
    ) -> Dict[str, Any]:
        """
        融合检索主入口

        Args:
            query: 用户查询
            top_k: 返回数量
            mode: 检索模式
                - "semantic": 纯语义
                - "dimension": 纯维度
                - "fusion": RRF 融合（默认）
            alpha: 融合权重（0=全语义, 1=全维度）
            collection_name: Collection 名称
        """
        if alpha is None:
            alpha = self.config.DEFAULT_ALPHA

        start_time = time.time()

        sem_results = []
        dim_results = []

        if mode in ("semantic", "fusion"):
            sem_results = self.search_semantic(
                query, top_k=self.config.SEM_TOP_K, collection_name=collection_name
            )

        if mode in ("dimension", "fusion"):
            dim_results = self.search_dimension(
                query, top_k=self.config.DIM_TOP_K, collection_name=collection_name
            )

        if mode == "semantic":
            fused = [{"chunk_id": r["chunk_id"], "score": r["score"]} for r in sem_results[:top_k]]
            final = self._enrich_results(fused, collection_name)
        elif mode == "dimension":
            fused = dim_results[:top_k]
            final = self._enrich_results(fused, collection_name)
        else:  # fusion
            # 提取 ID 和分数
            sem_fused = [{"chunk_id": r["chunk_id"], "sem_score": r["score"]} for r in sem_results]
            dim_fused = [{"chunk_id": r["chunk_id"], "dim_score": r["dim_score"]} for r in dim_results]

            # 构建 chunk_id -> score 映射
            sem_map = {r["chunk_id"]: r["sem_score"] for r in sem_fused}
            dim_map = {r["chunk_id"]: r["dim_score"] for r in dim_fused}
            all_ids = list(set(list(sem_map.keys()) + list(dim_map.keys())))

            # 计算加权分数
            scored = []
            for cid in all_ids:
                sem_s = sem_map.get(cid, 0)
                dim_s = dim_map.get(cid, 0)
                fused_score = (1 - alpha) * sem_s + alpha * dim_s
                scored.append({"chunk_id": cid, "score": fused_score})

            scored.sort(key=lambda x: x["score"], reverse=True)
            final = self._enrich_results(scored[:top_k], collection_name)

        elapsed = time.time() - start_time
        return {
            "query": query,
            "mode": mode,
            "alpha": alpha,
            "total": len(final),
            "results": final,
            "timing_ms": round(elapsed * 1000, 2),
        }

    def _enrich_results(
        self,
        fused: List[Dict],
        collection_name: str = None,
    ) -> List[Dict]:
        """根据 chunk_id 从 Qdrant 获取完整信息"""
        if not fused or not self.qdrant:
            return fused

        chunk_ids = [r["chunk_id"] for r in fused]
        score_map = {r["chunk_id"]: r.get("score", 0) for r in fused}

        try:
            # 通过 scroll 获取完整数据
            all_points = []
            offset = None
            while True:
                page = self.qdrant.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                )
                all_points.extend(page.get("points", []))
                offset = page.get("next_page_offset")
                if not offset:
                    break

            # 筛选匹配 chunk_id 的
            id_set = set(chunk_ids)
            enriched = []
            for pt in all_points:
                cid = str(pt["id"])
                if cid in id_set:
                    payload = pt.get("payload", {})
                    enriched.append({
                        "chunk_id": cid,
                        "score": score_map.get(cid, 0),
                        "chunk_text": payload.get("chunk_text", ""),
                        "doc_title": payload.get("doc_title", ""),
                        "chunk_gen_title": payload.get("chunk_gen_title", ""),
                        "doc_id": payload.get("doc_id", ""),
                        "profile_json": payload.get("profile_json", {}),
                    })

            # 按分数排序
            enriched.sort(key=lambda x: x["score"], reverse=True)
            return enriched

        except Exception as e:
            print(f"[检索] enrich_results 失败: {e}")
            return [{"chunk_id": r["chunk_id"], "score": r.get("score", 0)} for r in fused]


# ==================== 简化版检索器（直接 Qdrant）====================

class SimpleSearcher:
    """简化检索器，直接使用 Qdrant 向量检索，无需索引文件"""

    def __init__(self, qdrant_client, embedding_service):
        self.qdrant = qdrant_client
        self.embedding = embedding_service

    def search(
        self,
        query: str,
        top_k: int = 10,
        collection_name: str = None,
        vector_field: str = "chunk_text_vec",
    ) -> List[Dict]:
        """纯向量检索"""
        if not self.qdrant or not self.qdrant.is_connected():
            return []

        try:
            query_vec = self.embedding.encode_query(query, model="bgem3")

            results = self.qdrant.search(
                query_vector=query_vec,
                collection_name=collection_name,
                limit=top_k,
                using=vector_field,
                with_payload=True,
            )

            output = []
            for hit in results:
                payload = hit.get("payload", {})
                output.append({
                    "chunk_id": hit["id"],
                    "score": hit["score"],
                    "chunk_text": payload.get("chunk_text", ""),
                    "doc_title": payload.get("doc_title", ""),
                    "chunk_gen_title": payload.get("chunk_gen_title", ""),
                    "doc_id": payload.get("doc_id", ""),
                })

            return output

        except Exception as e:
            print(f"[SimpleSearcher] 检索失败: {e}")
            return []
