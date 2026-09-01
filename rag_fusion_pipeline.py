# -*- coding: utf-8 -*-
"""
rag_fusion_pipeline.py - 维度向量检索 + 语义检索融合 RAG 流水线
=============================================================================

核心设计：维度存储从文字匹配改为向量检索。

维度向量集合 (Qdrant collection: dimension_tags):
  - 每个维度标签（如 "景区介绍"、"长陵"、"万历皇帝"）存储为一个向量点
  - Payload 包含: dim_name, tag_name, chunk_ids（该标签关联的所有 chunk ID 列表）
  - 通过 query 向量搜索找到最相关的维度标签

检索流程（基于 Qdrant payload dim_* 字段）:
  1. query 向量搜索 dimension_tags collection → 找到匹配的维度标签
  2. 对语义检索结果，提取其 payload 中的 dim_* 字段
  3. 计算每个 chunk 的维度命中得分（与 query 命中维度的交集 × 均分）
  4. 按维度得分降序，作为维度检索结果
  5. 与语义检索结果 RRF 融合

使用方式:
  # 构建维度向量集合（首次或维度数据更新后）
  python rag_fusion_pipeline.py --build-dim-index

  # 交互式检索
  python rag_fusion_pipeline.py --query "明十三陵的历史背景"
"""

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# ── 路径配置 ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "fusion_app"))

# ── 数据路径 ─────────────────────────────────────────────────────────────────
EXP_DIR = PROJECT_ROOT / "experiment_data"
PATH_INVERTED_INDEX = EXP_DIR / "inverted_index.json"       # 原始维度标签数据
PATH_DIM_META = EXP_DIR / "dimension_metadata.json"           # 维度元数据
PATH_TAG_VECTORS = EXP_DIR / "tag_vectors.pkl"               # 预计算的标签向量（可选）
DIM_COLLECTION_NAME = "dimension_tags"                        # Qdrant 中的维度向量 collection

# ── 检索配置 ─────────────────────────────────────────────────────────────────
DEFAULT_ALPHA = 0.5        # RRF 融合权重：alpha=维度权重，1-alpha=语义权重
SEM_TOP_K = 50             # 语义检索返回数量
DIM_TAG_TOP_K = 20         # 维度标签向量搜索返回数量
DIM_CHUNK_TOP_K = 200      # 每个标签关联的 chunk 数量上限
FINAL_TOP_K = 10           # 最终返回数量
RRF_K = 60                 # RRF 参数
DIM_SCORE_THRESHOLD = 0.3  # 维度标签搜索的最低分数阈值


# ════════════════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════════════════

def _load_qdrant_client():
    """加载 Qdrant 客户端"""
    try:
        from fusion_app.app.core.qdrant_client import QdrantClient
        from fusion_app.config import get_fusion_config
        cfg = get_fusion_config()
        return QdrantClient(cfg.vecdb_qdrant)
    except Exception as e:
        print(f"[Qdrant] 加载失败: {e}")
        return None


def _normalize_cid(cid: str) -> str:
    """统一 chunk_id 格式：去掉 .txt 后缀和末尾下划线"""
    cid = str(cid)
    cid = cid.rstrip(".txt")
    cid = cid.rstrip("_")
    return cid


def _encode_bgem3(texts: List[str], normalize: bool = True) -> Optional[List[List[float]]]:
    """用 BGE-M3 编码文本列表"""
    try:
        from fusion_app.app.core.embedding_service import get_embedding_service
        svc = get_embedding_service()
        if not svc.load_bgem3():
            print("[Embedding] BGE-M3 加载失败")
            return None
        return svc.encode_bgem3(texts, normalize=normalize)
    except Exception as e:
        print(f"[Embedding] BGE-M3 编码失败: {e}")
        return None


def _encode_query_bgem3(query: str) -> Optional[List[float]]:
    """用 BGE-M3 编码查询"""
    try:
        from fusion_app.app.core.embedding_service import get_embedding_service
        svc = get_embedding_service()
        if not svc.load_bgem3():
            return None
        return svc.encode_query_bgem3(query)
    except Exception:
        return None


def _chunk_cache_from_qdrant(collection_name: str, qdrant_client) -> Dict[str, Dict]:
    """从 Qdrant 加载全部 chunk 到内存缓存"""
    cache: Dict[str, Dict] = {}
    try:
        offset = None
        while True:
            page = qdrant_client.scroll(
                collection_name=collection_name,
                limit=500,
                offset=offset,
                with_payload=True,
            )
            if hasattr(page, "points"):
                points = page.points
                offset = page.next_page_offset
            else:
                points = page.get("points", [])
                offset = page.get("next_page_offset")
            if not points:
                break
            for pt in points:
                cid = str(pt.id) if hasattr(pt, "id") else str(pt["id"])
                payload = pt.payload if hasattr(pt, "payload") else pt.get("payload", {})
                cache[cid] = {
                    "chunk_text": payload.get("chunk_text", ""),
                    "doc_title": payload.get("doc_title", ""),
                    "chunk_gen_title": payload.get("chunk_gen_title", ""),
                    "doc_id": payload.get("doc_id", ""),
                    "profile_json": payload.get("profile_json", {}),
                }
            if not offset:
                break
        print(f"  Chunk 缓存: {len(cache)} 个")
    except Exception as e:
        print(f"  Chunk 缓存加载失败: {e}")
    return cache


# ════════════════════════════════════════════════════════════════════════════
# 维度向量集合构建器
# ════════════════════════════════════════════════════════════════════════════

class DimensionIndexBuilder:
    """
    将维度标签（维度的 tag）编码为向量，存入 Qdrant 的 dimension_tags collection。

    每个点的结构:
      - id: str，格式 "dim_name::tag_name"
      - vector: BGE-M3 向量（1024 维）
      - payload: {
          "dim_name": "景区介绍",
          "tag_name": "长陵",
          "chunk_ids": ["c1", "c2", ...],   # 该标签关联的 chunk ID 列表
          "dim_score": 1.0                   # 原始维度权重（用于最终排序）
        }
    """

    def __init__(self, qdrant_client):
        self.qdrant = qdrant_client

    def build(self, top_chunk_ids_per_tag: int = DIM_CHUNK_TOP_K) -> int:
        """
        从 inverted_index.json 构建维度向量集合。

        Args:
            top_chunk_ids_per_tag: 每个 tag 最多关联多少个 chunk ID（避免 payload 过大）

        Returns:
            写入的向量点数量
        """
        if not self.qdrant or not self.qdrant.is_connected():
            print("[构建] Qdrant 未连接")
            return 0

        # 1. 加载原始维度数据
        if not PATH_INVERTED_INDEX.exists():
            print(f"[构建] 找不到倒排索引文件: {PATH_INVERTED_INDEX}")
            return 0

        with open(PATH_INVERTED_INDEX, "r", encoding="utf-8") as f:
            inverted_index = json.load(f)

        print(f"[构建] 加载倒排索引: {len(inverted_index)} 个维度")

        # 2. 收集所有 (dim_name, tag_name) 对
        tag_entries = []
        for dim_name, tags_dict in inverted_index.items():
            if not isinstance(tags_dict, dict):
                continue
            for tag_name, chunk_ids in tags_dict.items():
                if not isinstance(chunk_ids, list) or not chunk_ids:
                    continue
                chunk_ids = [cid.rstrip(".txt") for cid in chunk_ids[:top_chunk_ids_per_tag]]
                tag_entries.append({
                    "dim_name": dim_name,
                    "tag_name": tag_name,
                    "chunk_ids": chunk_ids,
                })

        print(f"[构建] 共 {len(tag_entries)} 个维度标签")

        # 3. 编码向量
        tag_texts = [e["tag_name"] for e in tag_entries]
        vectors = _encode_bgem3(tag_texts)
        if not vectors:
            print("[构建] 向量编码失败")
            return 0

        vec_dim = len(vectors[0])
        print(f"[构建] 向量维度: {vec_dim}，标签数量: {len(vectors)}")

        # 4. 创建 collection
        self.qdrant.create_collection(
            collection_name=DIM_COLLECTION_NAME,
            vector_dim=vec_dim,
            distance="Cosine",
            force=True,
        )

        # 5. 构建 points
        points = []
        for i, entry in enumerate(tag_entries):
            raw_id = f"{entry['dim_name']}::{entry['tag_name']}"
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))
            points.append({
                "id": point_uuid,
                "vector": vectors[i],
                "payload": {
                    "dim_name": entry["dim_name"],
                    "tag_name": entry["tag_name"],
                    "chunk_ids": entry["chunk_ids"],
                    "dim_score": 1.0,
                },
            })

        # 6. 写入
        result = self.qdrant.upsert(points=points, collection_name=DIM_COLLECTION_NAME)
        count = result.get("count", len(points))
        print(f"[构建] 写入完成: {count} 个维度标签向量")
        return count


# ════════════════════════════════════════════════════════════════════════════
# 核心检索器
# ════════════════════════════════════════════════════════════════════════════

class FusionRetriever:
    """
    融合检索器：语义向量检索 + 维度向量检索 + RRF 融合。

    维度检索（基于 Qdrant payload dim_* 字段）:
      - 语义检索 top-N chunks（带回 payload）
      - 从 payload 中提取 dim_* 字段（预提取的维度标签）
      - 通过 dimension_tags 向量搜索得到 query 对应的目标维度
      - 计算每个 chunk 的维度命中得分
      - 与语义结果 RRF 融合
    """

    def __init__(
        self,
        collection_name: str = "rag_chunks",
        alpha: float = DEFAULT_ALPHA,
    ):
        self.collection_name = collection_name
        self.alpha = alpha

        self.qdrant = _load_qdrant_client()
        self._chunk_cache: Dict[str, Dict] = {}

    # ── 初始化 ────────────────────────────────────────────────────────────────

    def load_chunk_cache(self):
        """预加载 chunk 缓存（加速最终结果丰富化）"""
        if not self.qdrant:
            return
        if self._chunk_cache:
            return
        self._chunk_cache = _chunk_cache_from_qdrant(self.collection_name, self.qdrant)

    # ── 语义检索 ─────────────────────────────────────────────────────────────

    def _search_semantic(self, query_vec: List[float], top_k: int = SEM_TOP_K) -> List[Dict]:
        """纯语义向量检索"""
        if not self.qdrant:
            return []
        try:
            results = self.qdrant.search(
                query_vector=query_vec,
                collection_name=self.collection_name,
                limit=top_k,
                with_payload=True,
                using="chunk_text_vec",
            )
            out = []
            for hit in results:
                cid = str(hit.id) if hasattr(hit, "id") else str(hit["id"])
                cid = _normalize_cid(cid)
                score = float(hit.score) if hasattr(hit, "score") else float(hit.get("score", 0))
                payload = hit.payload if hasattr(hit, "payload") else hit.get("payload", {})
                out.append({
                    "chunk_id": cid,
                    "sem_score": score,
                    "payload": payload,
                })
            return out
        except Exception as e:
            print(f"[语义检索] 失败: {e}")
            return []

    # ── 维度标签向量搜索 ───────────────────────────────────────────────────────

    def _search_dim_tags(
        self,
        query_vec: List[float],
        top_k: int = DIM_TAG_TOP_K,
    ) -> List[Dict]:
        """
        在 dimension_tags collection 中搜索最匹配的维度标签。
        返回格式: [{"dim_name": "...", "tag_name": "...", "score": 0.x, "chunk_ids": [...]}]
        """
        if not self.qdrant or not self.qdrant.is_connected():
            return []
        try:
            results = self.qdrant.search(
                query_vector=query_vec,
                collection_name=DIM_COLLECTION_NAME,
                limit=top_k,
                with_payload=True,
                using="chunk_text_vec",
            )
            out = []
            for hit in results:
                payload = hit.payload if hasattr(hit, "payload") else hit.get("payload", {})
                score = float(hit.score) if hasattr(hit, "score") else float(hit.get("score", 0))
                out.append({
                    "dim_name": payload.get("dim_name", ""),
                    "tag_name": payload.get("tag_name", ""),
                    "score": score,
                    "chunk_ids": payload.get("chunk_ids", []),
                })
            return out
        except Exception as e:
            print(f"[维度标签搜索] 失败: {e}")
            return []

    # ── 维度检索（基于 Qdrant payload dim_* 字段）────────────────────────────

    def _search_dimension(
        self,
        query_vec: List[float],
        top_k: int = DIM_CHUNK_TOP_K,
    ) -> List[Dict]:
        """
        维度感知检索：利用 Qdrant chunk payload 中的 dim_* 字段做维度匹配。

        流程：
        1. 语义检索 top-N chunks（带回 payload 中的 dim_* 字段）
        2. 从 payload 中解析出 chunk 所属的维度标签集合
        3. 通过 dimension_tags 向量搜索得到 query 对应的目标维度
        4. 计算每个 chunk 的维度命中得分（与目标维度交集大小 × 均分）
        5. 按维度得分降序排序
        """
        # 1. 语义检索（带回 payload）
        sem_results = self._search_semantic(query_vec, top_k=SEM_TOP_K)
        if not sem_results:
            print("[维度检索] 警告: 语义检索无结果")
            return []

        # 2. 从语义结果 payload 中提取 dim_* 字段，构建 chunk -> 维度信息映射
        chunk_dim_map: Dict[str, Dict[str, str]] = {}
        for item in sem_results:
            cid = item["chunk_id"]
            payload = item.get("payload", {})
            dim_info: Dict[str, str] = {}
            for k, v in payload.items():
                if k.startswith("dim_"):
                    dim_info[k[4:]] = str(v)  # "dim_建筑名称" -> "建筑名称"
            chunk_dim_map[cid] = dim_info

        # 3. 通过 dimension_tags 向量搜索找到 query 的目标维度
        raw_tags = self._search_dim_tags(query_vec, top_k=DIM_TAG_TOP_K)
        if not raw_tags:
            print("[维度检索] 警告: query 未匹配到维度标签")
            return []

        # 按维度名去重，每维度保留得分最高的标签
        best_per_dim: Dict[str, Dict] = {}
        for item in raw_tags:
            dname = item["dim_name"]
            if dname not in best_per_dim or item["score"] > best_per_dim[dname]["score"]:
                best_per_dim[dname] = item

        matched_dim_names = list(best_per_dim.keys())
        print(f"[维度检索] 匹配到 {len(matched_dim_names)} 个维度: {matched_dim_names}")

        if not matched_dim_names:
            return []

        # 4. 对语义检索结果按维度命中评分
        # 命中 query 的维度越多 + 标签匹配度越高 -> 得分越高
        scored_chunks: Dict[str, float] = {}
        for item in sem_results:
            cid = item["chunk_id"]
            chunk_dims = chunk_dim_map.get(cid, {})

            hit_count = 0
            total_score = 0.0
            for dname in matched_dim_names:
                target_tag = best_per_dim[dname]["tag_name"]
                target_score = best_per_dim[dname]["score"]
                chunk_dim_val = chunk_dims.get(dname, "")

                if not chunk_dim_val:
                    continue

                dim_labels = [l.strip() for l in chunk_dim_val.split(";") if l.strip()]
                matched = False
                for label in dim_labels:
                    if label and (label in target_tag or target_tag in label):
                        hit_count += 1
                        total_score += target_score
                        matched = True
                        break

                if not matched and dim_labels:
                    hit_count += 0.5
                    total_score += target_score * 0.3

            if hit_count > 0:
                avg_score = total_score / max(hit_count, 1)
                dim_score = hit_count * avg_score
            else:
                dim_score = 0.0

            scored_chunks[cid] = dim_score

        # 5. 按维度得分排序
        sorted_chunks = sorted(scored_chunks.items(), key=lambda x: x[1], reverse=True)

        result = []
        for cid, score in sorted_chunks[:top_k]:
            chunk_dims = chunk_dim_map.get(cid, {})
            result.append({
                "chunk_id": cid,
                "dim_score": score,
                "dim_hit_count": len(matched_dim_names),
                "dim_names": matched_dim_names,
                "dim_tags": {d: best_per_dim[d]["tag_name"] for d in matched_dim_names},
                "payload": sem_results[next((i for i, r in enumerate(sem_results) if r["chunk_id"] == cid), -1)].get("payload", {}),
            })

        if not result:
            print("[维度检索] 警告: 无 chunk 命中维度过滤")
        else:
            print(f"[维度检索] 返回 {len(result)} 个 chunks")

        return result

    # ── RRF 融合 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_fuse(
        sem_chunks: List[Dict],
        dim_chunks: List[Dict],
        alpha: float,
        top_k: int = FINAL_TOP_K,
    ) -> List[Dict]:
        """
        RRF 融合语义和维度检索结果。

        综合分 = alpha * dim_normalized + (1-alpha) * sem_normalized
        其中 dim/sem_normalized = 1 / (RRF_K + rank)
        """
        scores: Dict[str, Dict] = {}

        for rank, chunk in enumerate(sem_chunks):
            cid = chunk["chunk_id"]
            rrf = 1.0 / (RRF_K + rank + 1)
            if cid not in scores:
                scores[cid] = {"sem_score": 0, "dim_score": 0, "chunk_id": cid}
            scores[cid]["sem_score"] = chunk.get("sem_score", 0)
            scores[cid]["_sem_rrf"] = rrf

        for rank, chunk in enumerate(dim_chunks):
            cid = chunk["chunk_id"]
            rrf = 1.0 / (RRF_K + rank + 1)
            if cid not in scores:
                scores[cid] = {"sem_score": 0, "dim_score": 0, "chunk_id": cid}
            scores[cid]["dim_score"] = chunk.get("dim_score", 0)
            scores[cid]["_dim_rrf"] = rrf

        # 计算综合分
        max_sem = max((s["sem_score"] for s in scores.values() if s["sem_score"] > 0), default=1.0)
        max_dim = max((s["dim_score"] for s in scores.values() if s["dim_score"] > 0), default=1.0)

        final = []
        for cid, s in scores.items():
            sem_norm = s["sem_score"] / max_sem if s["sem_score"] > 0 else 0
            dim_norm = s["dim_score"] / max_dim if s["dim_score"] > 0 else 0
            sem_rrf = s.get("_sem_rrf", 0)
            dim_rrf = s.get("_dim_rrf", 0)
            fused = (1 - alpha) * sem_rrf + alpha * dim_rrf
            fused = (1 - alpha) * (sem_norm * 0.1 + sem_rrf) + alpha * (dim_norm * 0.1 + dim_rrf)
            dim_hit = 0
            for dc in dim_chunks:
                if dc["chunk_id"] == cid:
                    dim_hit = dc.get("dim_hit_count", 0)
                    break
            final.append({
                "chunk_id": cid,
                "fused_score": fused,
                "sem_score": s["sem_score"],
                "dim_score": s["dim_score"],
                "dim_hit_count": dim_hit,
            })

        final.sort(key=lambda x: x["fused_score"], reverse=True)
        return final[:top_k]

    # ── 主检索入口 ────────────────────────────────────────────────────────────

    def search(
        self,
        question: str,
        top_k: int = FINAL_TOP_K,
        mode: str = "fusion",   # "semantic" | "dimension" | "fusion"
        alpha: float = None,
    ) -> Dict[str, Any]:
        """
        融合检索主入口。

        Args:
            question: 用户问题
            top_k: 返回数量
            mode: 检索模式
                - "semantic": 纯语义
                - "dimension": 纯维度
                - "fusion": RRF 融合（默认）
            alpha: 融合权重，默认 self.alpha
        """
        alpha = alpha if alpha is not None else self.alpha

        t0 = time.time()
        sem_chunks = []
        dim_chunks = []

        # 编码 query
        query_vec = _encode_query_bgem3(question)
        if query_vec is None:
            return {"error": "query 编码失败", "question": question}

        # 语义检索
        if mode in ("semantic", "fusion"):
            sem_chunks = self._search_semantic(query_vec, top_k=SEM_TOP_K)
            print(f"[语义检索] 返回 {len(sem_chunks)} 个结果")

        # 维度检索
        if mode in ("dimension", "fusion"):
            dim_chunks = self._search_dimension(query_vec, top_k=DIM_CHUNK_TOP_K)
            print(f"[维度检索] 返回 {len(dim_chunks)} 个结果")

        # 融合
        if mode == "semantic":
            fused = [{"chunk_id": c["chunk_id"], "sem_score": c["sem_score"], "dim_score": 0.0}
                     for c in sem_chunks[:top_k]]
        elif mode == "dimension":
            fused = [{"chunk_id": c["chunk_id"], "dim_score": c["dim_score"], "sem_score": 0.0,
                      "dim_hit_count": c.get("dim_hit_count", 0)}
                     for c in dim_chunks[:top_k]]
        else:
            fused = self._rrf_fuse(sem_chunks, dim_chunks, alpha=alpha, top_k=top_k)

        # 丰富结果：补充 chunk 文本等信息
        self.load_chunk_cache()
        final_results = []
        for item in fused:
            cid = item["chunk_id"]
            payload = self._chunk_cache.get(cid, {})
            final_results.append({
                "chunk_id": cid,
                "chunk_text": payload.get("chunk_text", ""),
                "doc_title": payload.get("doc_title", ""),
                "chunk_gen_title": payload.get("chunk_gen_title", ""),
                "final_score": item.get("fused_score", item.get("sem_score", 0) + item.get("dim_score", 0)),
                "sem_score": item.get("sem_score", 0),
                "dim_score": item.get("dim_score", 0),
                "dim_hit_count": item.get("dim_hit_count", 0),
            })

        elapsed = (time.time() - t0) * 1000
        return {
            "question": question,
            "mode": mode,
            "alpha": alpha,
            "total": len(final_results),
            "results": final_results,
            "timing_ms": round(elapsed, 2),
        }


# ════════════════════════════════════════════════════════════════════════════
# 命令行入口
# ════════════════════════════════════════════════════════════════════════════

def _print_results(result: Dict):
    """格式化打印检索结果"""
    print(f"\n◆ 检索结果  [{result.get('mode', '?')} mode, alpha={result.get('alpha', 0):.1f}, {result.get('timing_ms', 0):.0f}ms]")
    print("=" * 70)
    for i, r in enumerate(result.get("results", []), 1):
        sem = r.get("sem_score", 0)
        dim = r.get("dim_score", 0)
        total = r.get("final_score", 0)
        hit = r.get("dim_hit_count", 0)
        title = r.get("chunk_gen_title") or r.get("doc_title", "")
        text = r.get("chunk_text", "")[:80].replace("\n", " ")
        print(f"  {i}. [{title}] 总分={total:.4f} (语义={sem:.4f} 维度={dim:.4f} 命中={hit}维)")
        if text:
            print(f"     {text}...")


def cmd_build_dim_index():
    """构建维度向量集合"""
    print("=" * 60)
    print("Step: 构建维度向量集合 (Qdrant dimension_tags collection)")
    print("=" * 60)

    qdrant = _load_qdrant_client()
    if not qdrant:
        return

    builder = DimensionIndexBuilder(qdrant)
    count = builder.build()
    print(f"\n构建完成: {count} 个维度标签向量写入 Qdrant")


def cmd_query(question: str, mode: str, alpha: float, top_k: int):
    """执行检索"""
    print("=" * 60)
    print(f"Question: {question}")
    print(f"Mode: {mode}, alpha={alpha}, top_k={top_k}")
    print("=" * 60)

    retriever = FusionRetriever()
    result = retriever.search(
        question=question,
        top_k=top_k,
        mode=mode,
        alpha=alpha,
    )
    if "error" in result:
        print(f"错误: {result['error']}")
        return
    _print_results(result)


def cmd_interactive():
    """交互式检索循环"""
    print("=" * 60)
    print("交互式检索（输入 q 退出）")
    print(f"默认 mode=fusion, alpha={DEFAULT_ALPHA}")
    print("=" * 60)

    retriever = FusionRetriever()

    while True:
        try:
            q = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("q", "quit", "exit"):
            break
        if q.lower().startswith("mode:"):
            parts = q.split(maxsplit=1)
            if len(parts) == 2:
                mode = parts[1].strip()
                print(f"Mode 切换为: {mode}")
                continue
        result = retriever.search(q, mode="fusion", alpha=DEFAULT_ALPHA)
        if "error" in result:
            print(f"错误: {result['error']}")
            continue
        _print_results(result)


def main():
    parser = argparse.ArgumentParser(description="RAG Fusion Pipeline")
    parser.add_argument("--build-dim-index", action="store_true", help="构建维度向量索引")
    parser.add_argument("--query", type=str, help="执行单次检索")
    parser.add_argument("--interactive", action="store_true", help="交互式检索")
    parser.add_argument("--mode", type=str, default="fusion",
                        choices=["semantic", "dimension", "fusion"],
                        help="检索模式")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help=f"融合权重 (0=全语义, 1=全维度), 默认 {DEFAULT_ALPHA}")
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K,
                        help=f"返回数量, 默认 {FINAL_TOP_K}")
    args = parser.parse_args()

    if args.build_dim_index:
        cmd_build_dim_index()
    elif args.query:
        cmd_query(args.query, args.mode, args.alpha, args.top_k)
    elif args.interactive:
        cmd_interactive()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
