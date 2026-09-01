"""
retrieval_fusion_eval.py

兼容层：为 interactive_qa.py 提供检索功能。
实现 DimensionSearcher 和 SemanticSearcher，直接走 Qdrant HTTP API。
保留原 signature: search() -> {"results": [...], "constraints": {...}, "query_text": "..."}

使用方式：
    from retrieval_fusion_eval import (
        DimensionSearcher,
        SemanticSearcher,
        rrf_fuse_all,
        SearchConfig,
        _load_bge_encoder,
        _RateLimiter,
    )
"""

import sys
import os
import json
import pickle
import threading
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
EXPERIMENT_DATA = PROJECT_ROOT / "experiment_data"

# 把 code/ 目录本身加入 sys.path，让 llm_service / query_parser 可作为顶层模块导入。
# 注意：不能直接 import code 包（会与标准库同名），但可以把 code/ 目录作为模块路径。
_CODE_DIR = PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))


# ===================== 加载 db_config =====================

def _load_db_config():
    """从 db_config.yaml 加载配置（Qdrant 版本）"""
    config_path = PROJECT_ROOT / "db_config.yaml"
    if not config_path.exists():
        return {
            "host": "127.0.0.1",
            "port": "6333",
            "collection_name": "rag_chunks",
            "dim_tags_collection": "dimension_tags",
            "vector_dim": 1024,
            "distance": "Cosine",
        }

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # 兼容两种顶层 key：vecdb_qdrant（新）和 vecdb（旧/Milvus）
        vecdb = cfg.get("vecdb_qdrant") or cfg.get("vecdb", {})
        return {
            "host": vecdb.get("host", "127.0.0.1"),
            "port": str(vecdb.get("port", "6333")),
            "collection_name": vecdb.get("collection_name", "rag_chunks"),
            "dim_tags_collection": cfg.get("dim_tags_collection", "dimension_tags"),
            "vector_dim": vecdb.get("vector_dim", 1024),
            "distance": vecdb.get("distance", "Cosine"),
        }
    except Exception:
        return {
            "host": "127.0.0.1",
            "port": "6333",
            "collection_name": "rag_chunks",
            "dim_tags_collection": "dimension_tags",
            "vector_dim": 1024,
            "distance": "Cosine",
        }


_qdrant_cfg = _load_db_config()


# ===================== SearchConfig =====================

class SearchConfig:
    VECTOR_TOP_K = 200
    RRF_K = 60
    DEFAULT_ALPHA = 0.2
    SOFT_MATCH_THRESHOLD = 0.65
    SEM_TOP_K = 20
    DIM_TOP_K = 100
    EXCLUDED_DIMS: Set[str] = set()
    # 维度检索粗排方法（对应 semantic_dimension_fusion_strategy.md 2.1 节）
    # "vec":       方法一：query 向量 - D 向量（直接向量召回）
    # "constraint": 方法二：query 维度 - D 维度（解析维度约束过滤）
    # "tag":        方法三：query 向量 - 维度标签向量（tag 向量召回）
    DIM_RECALL_METHOD = "tag"
    # 粗排截断数量：精排前保留 top_k 个候选（粗排得分降序截断）
    DIM_RERANK_TOP_K = 50
    # 维度检索精排方法（对应 semantic_dimension_fusion_strategy.md 2.2 节）
    # "tag_sim":  方法一：query 向量 vs 维度-标签向量均值（精排得分覆盖粗排得分）
    # "tag_match": 方法二：逐维标签匹配得分（query 标签 vs 文档标签向量相似度）
    DIM_RERANK_METHOD = "tag_sim"
    # 维度检索两阶段内部融合策略（对应 ！！！！！注释 ！！！！！！）
    # "rrf":  粗排得分 + 精排得分先做一次 RRF 融合
    # "score": 直接使用精排得分（覆盖粗排得分）
    DIM_INTERNAL_FUSION = "score"


# ===================== Qdrant HTTP 客户端（轻量） =====================

class _QdrantClient:
    """极简 Qdrant HTTP 客户端（只实现 search + scroll + retrieve）"""

    def __init__(self, host: str, port: str):
        self.base = f"http://{host}:{port}"

    def _post(self, path: str, body: dict, timeout: float = 30.0) -> dict:
        import httpx
        r = httpx.post(self.base + path, json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def search(self, collection: str, query_vector: List[float],
               vector_name: str = "chunk_text_vec",
               top_k: int = 20, score_threshold: Optional[float] = None,
               with_payload: bool = True) -> List[dict]:
        # 优先用 Qdrant 的 /points/query API，对 named vector 是最友好的
        body = {
            "query": query_vector,   # dense 模式
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
            # 回退到 /points/search（旧的，无 named vector 友好）
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


    def _legacy_search(self, collection: str, query_vector: List[float],
               vector_name: str = "chunk_text_vec",
               top_k: int = 20, score_threshold: Optional[float] = None,
               with_payload: bool = True) -> List[dict]:
        # legacy /points/search using nested named vector form (Qdrant 1.18 schema)
        nested = {"name": vector_name, "vector": query_vector}
        body = {
            "vector": nested,
            "limit": top_k,
            "with_payload": with_payload,
            "with_vector": False,
        }
        if score_threshold is not None:
            body["score_threshold"] = score_threshold
        data = self._post(f"/collections/{collection}/points/search", body)
        return data.get("result", [])

    def scroll(self, collection: str, limit: int = 100,
               with_payload: bool = True) -> List[dict]:
        body = {"limit": limit, "with_payload": with_payload, "with_vector": False}
        data = self._post(f"/collections/{collection}/points/scroll", body)
        return data.get("result", {}).get("points", [])

    def retrieve(self, collection: str, ids: List[str], with_payload: bool = True) -> List[dict]:
        """通过 filter 在指定字段（默认 chunk_id）上查询

        ids: chunk_id 列表（Qdrant 自带 point id 不能用字符串）
        """
        if not ids:
            return []
        # 用 filter + scroll 替代 retrieve，因为 Qdrant 1.18 不支持字符串 point id
        # 分批处理：每次最多 256 个 any
        BATCH = 64
        out = []
        for i in range(0, len(ids), BATCH):
            chunk = ids[i:i+BATCH]
            body = {
                "filter": {"must": [{"key": "chunk_id", "match": {"any": chunk}}]},
                "limit": len(chunk) * 2,
                "with_payload": with_payload,
                "with_vector": False,
            }
            try:
                data = self._post(f"/collections/{collection}/points/scroll", body)
                out.extend(data.get("result", {}).get("points", []))
            except Exception as e:
                print(f"[警告] scroll + filter 失败 ({collection}): {e}")
        return out


# ===================== BGE 编码器 =====================

def _load_bge_encoder():
    """加载 BGE-M3 编码器（优先 FlagEmbedding，失败回退 SentenceTransformer）"""
    model_path = str(PROJECT_ROOT / "model" / "bge-m3")
    if not os.path.exists(model_path):
        for base in [PROJECT_ROOT.parent, PROJECT_ROOT.parent.parent]:
            alt = base / "model" / "bge-m3"
            if alt.exists():
                model_path = str(alt)
                break

    if not os.path.exists(model_path):
        print("[警告] 未找到 BGE-M3 模型目录")
        return None

    # 优先 FlagEmbedding（与原 retriever 一致）
    try:
        from FlagEmbedding import BGEM3FlagModel
        return BGEM3FlagModel(model_path, use_fp16=False, device='cpu')
    except ImportError:
        pass
    except Exception as e:
        print(f"[警告] FlagEmbedding 加载失败 ({e})，回退到 SentenceTransformer")

        # 回退 SentenceTransformer（与 pipeline_qdrant.py 一致）
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path, local_files_only=True)
        # FlagEmbedding 优先标志：返回 _FlagProxy
        return _FlagProxy(model)
    except Exception as e:
        print(f"[警告] SentenceTransformer 加载失败: {e}")
        return None


class _FlagProxy:
    """让 SentenceTransformer 暴露与 BGEM3FlagModel 兼容的 .encode 接口"""

    def __init__(self, model):
        self._model = model

    def encode(self, texts, return_dense=False, **kwargs):
        # 移除 sentence-transformers 不识别的 kwarg
        kwargs.pop("normalize_embeddings", None)
        kwargs.pop("show_progress_bar", None)
        emb = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if return_dense:
            return {"dense_vecs": emb}
        return emb


def _encode_query(encoder, text: str) -> Optional[List[float]]:
    """编码单条查询文本，返回 1024 维 list[float]，失败返回 None"""
    if encoder is None:
        return None
    cls_name = encoder.__class__.__name__
    try:
        if cls_name == "_FlagProxy":
            # 使用 proxy 的统一接口
            emb = encoder.encode([text], return_dense=True)
            vec = emb["dense_vecs"][0]
        elif cls_name in ("BGEM3FlagModel", "M3Embedder"):
            # FlagEmbedding 原生：encode 不接受 normalize_embeddings
            emb = encoder.encode([text], return_dense=True)
            vec = emb["dense_vecs"][0]
        else:
            # 通用 sentence-transformers
            emb = encoder.encode(
                [text],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vec = emb[0]
        if hasattr(vec, "tolist"):
            return vec.tolist()
        return list(vec)
    except Exception as e:
        print(f"[警告] 向量编码失败 ({cls_name}): {e}")
        return None


# ===================== Rate Limiter =====================

class _RateLimiter:
    """API 调用频率限制器"""
    def __init__(self, min_interval_seconds: float):
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._lock = threading.Lock()
        self._next_allowed_time = 0.0

    def wait(self):
        if self.min_interval_seconds <= 0:
            return
        with self._lock:
            now = _time.time()
            if now < self._next_allowed_time:
                _time.sleep(self._next_allowed_time - now)
            self._next_allowed_time = now + self.min_interval_seconds


# ===================== RRF 融合 =====================

def rrf_fuse_all(dim_results: list, sem_results: list, k: int = 60, alpha: float = 0.5) -> list:
    """
    RRF 融合：Reciprocal Rank Fusion
    """
    def rrf_score(rank, weight=1.0):
        return weight / (k + rank)

    scores = {}
    for i, r in enumerate(dim_results):
        cid = r.get("chunk_id") or r.get("id", "")
        scores[cid] = scores.get(cid, 0) + alpha * rrf_score(i)

    for i, r in enumerate(sem_results):
        cid = r.get("chunk_id") or r.get("id", "")
        scores[cid] = scores.get(cid, 0) + (1 - alpha) * rrf_score(i)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [{"chunk_id": cid, "score": scores[cid]} for cid in sorted_ids]


# ===================== 维度缓存 =====================

def _normalize(text: str) -> str:
    """文本归一化（用于 chunk_id 模糊匹配）"""
    if not text:
        return ""
    return str(text).strip().replace("\u3000", " ").replace("  ", " ")


# ===================== 核心检索类 =====================

class DimensionAwareSearch:
    """
    维度感知检索器（Qdrant 实现）。

    sem：直接在 rag_chunks 上做 chunk_text_vec 向量检索
    dim：在 dimension_tags 上做向量检索，取 tag_name/dim_name，
         再展开 chunk_ids 查 rag_chunks 取出全文，最后 RRF 融合
    """

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or _qdrant_cfg["collection_name"]
        self.dim_tags_collection = _qdrant_cfg.get("dim_tags_collection", "dimension_tags")
        print(f">>> 初始化检索器 (Qdrant)...")
        print(f"    语料: {self.collection_name}  维度标签: {self.dim_tags_collection}")

        self.client = _QdrantClient(_qdrant_cfg["host"], _qdrant_cfg["port"])

        # BGE-M3 编码器
        self.encoder = _load_bge_encoder()
        if self.encoder is None:
            print("[警告] 编码器未加载，将无法进行向量检索")

        # 维度检索 metadata（dim_name → 是否启用），保留兼容
        self.inverted_index: Dict = {}
        self.dim_meta: Dict = {}
        self.tag_vectors: Dict = {}
        self._load_indexes()

    def _load_indexes(self):
        """加载本地倒排索引/维度元数据/标签向量（若存在则用于补充）"""
        path_inverted = EXPERIMENT_DATA / "inverted_index.json"
        path_dim_meta = EXPERIMENT_DATA / "dimension_metadata.json"
        path_tag_vec = EXPERIMENT_DATA / "tag_vectors.pkl"

        if path_inverted.exists():
            try:
                with open(path_inverted, 'r', encoding='utf-8') as f:
                    self.inverted_index = json.load(f)
                print(f"    倒排索引: {len(self.inverted_index)} 个维度")
            except Exception:
                self.inverted_index = {}

        if path_dim_meta.exists():
            try:
                with open(path_dim_meta, 'r', encoding='utf-8') as f:
                    self.dim_meta = json.load(f)
                print(f"    维度元数据: {len(self.dim_meta)} 个维度")
            except Exception:
                self.dim_meta = {}

        if path_tag_vec.exists():
            try:
                with open(path_tag_vec, 'rb') as f:
                    self.tag_vectors = pickle.load(f)
                print(f"    标签向量: {len(self.tag_vectors)} 个条目")
            except Exception:
                self.tag_vectors = {}

    # ---------- 检索主入口 ----------
    def search(
        self,
        query_text: str,
        top_k: int = 10,
        fusion: str = "rrf",
        alpha: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行检索。
        fusion: "rrf" | "score" | "dim_only" | "sem_only"
        alpha: 标签权重（0=纯语义，1=纯标签）
        """
        if alpha is None:
            alpha = SearchConfig.DEFAULT_ALPHA

        results = {"results": [], "constraints": {}, "query_text": query_text}

        query_vec = _encode_query(self.encoder, query_text)
        if query_vec is None:
            return results

        # ---- 语义检索 ----
        sem_results = []
        if fusion != "dim_only":
            try:
                sem_results = self._sem_search_via_qdrant(query_vec, top_k)
                results["constraints"]["_sem_count"] = len(sem_results)
            except Exception as e:
                print(f"[警告] 语义检索失败: {e}")

        # ---- 维度检索 ----
        dim_results = []
        recall_method = SearchConfig.DIM_RECALL_METHOD
        if fusion != "sem_only":
            try:
                dim_results = self._dim_search_via_qdrant(
                    query_vec, query_text, top_k, recall_method=recall_method
                )
                # 从 dim 收集中提炼 constraints
                constraints: Dict[str, set] = {}
                for r in dim_results:
                    dm = r.get("dim_name") or ""
                    tn = r.get("tag_name") or ""
                    if dm:
                        constraints.setdefault(dm, set()).add(tn)
                for k in constraints:
                    constraints[k] = sorted([x for x in constraints[k] if x])[:8]
                results["constraints"].update(constraints)
                results["recall_method"] = recall_method
            except Exception as e:
                print(f"[警告] 维度检索失败: {e}")

        results["results"] = self._fuse(dim_results, sem_results, alpha, top_k, fusion)
        results["dim_results"] = dim_results
        return results

    # ---------- 语义子检索（rag_chunks / chunk_text_vec） ----------
    def _sem_search_via_qdrant(self, query_vec: list, top_k: int) -> list:
        hits = self.client.search(
            collection=self.collection_name,
            query_vector=query_vec,
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
                "doc_title": payload.get("doc_title", ""),
                "chunk_gen_title": payload.get("chunk_gen_title", ""),
                "chunk_text_full": payload.get("chunk_text", ""),
                "score": hit.get("score", 0.0),
                "sem_rank": i + 1,
                "source": "sem",
            })
        return out

    # ---------- 维度检索主方法（两阶段：召回 + 精排） ----------
    def _dim_search_via_qdrant(
        self,
        query_vec: list,
        query_text: str,
        top_k: int,
        recall_method: str = None,
    ) -> list:
        """
        维度检索两阶段：
          1. 粗召回（三种方法可选）→ 全部候选
          2. 粗排截断 → 取粗排得分 top DIM_RERANK_TOP_K 个
          3. 精排 → 仅在截断后的候选上重打分

        对应 semantic_dimension_fusion_strategy.md 2.1 + 2.2 节。
        """
        if recall_method is None:
            recall_method = SearchConfig.DIM_RECALL_METHOD

        # ---- 第一阶段：粗召回（全部候选）----
        if recall_method == "vec":
            raw_candidates, recalled_tag_hits = self._dim_recall_by_vec(query_vec, query_text, top_k)
        elif recall_method == "constraint":
            raw_candidates, recalled_tag_hits = self._dim_recall_by_constraint(query_text, top_k)
        elif recall_method == "tag":
            raw_candidates, recalled_tag_hits = self._dim_recall_by_tag(query_vec, query_text, top_k)
        else:
            raw_candidates, recalled_tag_hits = self._dim_recall_by_tag(query_vec, query_text, top_k)

        if not raw_candidates:
            return []

        # ---- 粗排截断：按 recall_score 降序，只保留 top_k 个进入精排 ----
        rerank_top_k = SearchConfig.DIM_RERANK_TOP_K
        truncated = sorted(raw_candidates, key=lambda x: x.get("recall_score", 0), reverse=True)[:rerank_top_k]

        # ---- 第二阶段：精排（仅在截断候选上重打分）----
        reranked = self._dim_rerank(
            query_vec=query_vec,
            query_text=query_text,
            raw_candidates=truncated,
            recalled_tag_hits=recalled_tag_hits,
            top_k=top_k,
        )
        return reranked

    # ---------- 方法一：query 向量 - D 向量 ----------
    def _dim_recall_by_vec(self, query_vec: list, query_text: str, top_k: int) -> tuple[list, list]:
        """
        方法一：query 向量 - D 向量。
        D_cand = TopK_D( sim(v_q, v_D) )
        返回 (候选列表, 空 tag_hits列表)
        """
        hits = self.client.search(
            collection=self.collection_name,
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
                "recall_score": hit.get("score", 0.0),
                "dim_rank": i + 1,
                "source": "dim",
                "recall_method": "vec",
                "tag_name": "",
                "dim_name": "",
                "tag_hits": [],
                "evidence": [f"vec_score={hit.get('score', 0):.4f}"],
            })
        return candidates, []

    # ---------- 方法二：query 维度 - D 维度 ----------
    def _dim_recall_by_constraint(self, query_text: str, top_k: int) -> tuple[list, list]:
        """
        方法二：query 维度 - D 维度。
        解析 query 维度约束，筛选文档。
        返回 (候选列表, 空 tag_hits列表)
        """
        constraints = self._parse_query_constraints(query_text)
        if not constraints:
            print("[提示] 维度约束解析失败，constraint 方法无结果")
            return [], []

        real_dims = {k: v for k, v in constraints.items() if not k.startswith("_")}
        if not real_dims:
            return [], []

        all_candidates: Dict[str, dict] = {}
        tag_hits_out: list = []

        for dim_name, tag_list in real_dims.items():
            for tag_name in tag_list:
                hits = self._search_tag_by_name(dim_name, tag_name, top_k=10)
                for hit in hits:
                    self._merge_candidate(hit, all_candidates)
                    tag_hits_out.append(hit)

        if not all_candidates:
            return [], []

        payloads = self.client.retrieve(self.collection_name, list(all_candidates.keys()))
        candidates = self._build_candidates(payloads, all_candidates, "constraint")
        return candidates, tag_hits_out

    # ---------- 方法三：query 向量 - 维度标签向量 ----------
    def _dim_recall_by_tag(self, query_vec: list, query_text: str, top_k: int) -> tuple[list, list]:
        """
        方法三：query 向量 - 维度标签向量。
        L_q = TopK_{(m,t)}( sim(v_q, v_{m,t}) )
        D_cand = {D | T_D ∩ L_q != empty}
        返回 (候选列表, 原始 tag_hits)
        """
        tag_hits = self.client.search(
            collection=self.dim_tags_collection,
            query_vector=query_vec,
            vector_name="chunk_text_vec",
            top_k=top_k * 2,
        )
        if not tag_hits:
            return [], []

        all_candidates: Dict[str, dict] = {}
        for hit in tag_hits:
            self._merge_candidate(hit, all_candidates)

        if not all_candidates:
            return [], []

        payloads = self.client.retrieve(self.collection_name, list(all_candidates.keys()))
        candidates = self._build_candidates(payloads, all_candidates, "tag")
        return candidates, tag_hits

    # ---------- 辅助方法：按 (dim_name, tag_name) 精确查找 ----------
    def _search_tag_by_name(self, dim_name: str, tag_name: str, top_k: int = 10) -> list:
        """
        在 dimension_tags collection 中按 dim_name + tag_name 精确查找。
        返回命中的 tag 记录（含 chunk_ids）。
        """
        body = {
            "filter": {
                "must": [
                    {"key": "dim_name", "match": {"value": dim_name}},
                    {"key": "tag_name", "match": {"value": tag_name}},
                ]
            },
            "limit": top_k,
            "with_payload": True,
            "with_vector": False,
        }
        try:
            data = self._post_json(f"/collections/{self.dim_tags_collection}/points/scroll", body)
            return data.get("result", {}).get("points", [])
        except Exception:
            return []

    def _merge_candidate(self, hit: dict, candidates: dict):
        """将一个 tag_hit 合并到候选集合中。"""
        payload = hit.get("payload") or {}
        chunk_ids = payload.get("chunk_ids") or []
        if not isinstance(chunk_ids, list):
            return

        tag_score = hit.get("score", 0.0)
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
        self,
        payloads: list,
        candidates: dict,
        recall_method: str,
        top_k: int = None,
    ) -> list:
        """
        将 retrieve 到的 chunk payload 与候选元信息合并，组装为粗召回候选。
        按 recall_score 降序排列，最多返回 top_k 条。
        候选中携带 recall_score 和 tag_hits，供精排阶段使用。
        """
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
                "recall_score": meta.get("dim_score", 0.0),
                "score": meta.get("dim_score", 0.0),
                "dim_rank": 0,
                "source": "dim",
                "recall_method": recall_method,
                "tag_name": meta.get("tag_name", ""),
                "dim_name": meta.get("dim_name", ""),
                "tag_hits": meta.get("tag_hits", []),
                "evidence": [
                    f"{h['dim']}:{h['tag']}"
                    for h in meta.get("tag_hits", [])
                ][:3],
            })

        # 按 recall_score 降序排列
        out.sort(key=lambda x: x["recall_score"], reverse=True)

        # 重排 dim_rank（粗召回排名）
        for i, r in enumerate(out):
            r["dim_rank"] = i + 1

        if top_k:
            out = out[:top_k]
        return out

    # ---------- 第二阶段精排 ----------
    def _dim_rerank(
        self,
        query_vec: list,
        query_text: str,
        raw_candidates: list,
        recalled_tag_hits: list,
        top_k: int,
    ) -> list:
        """
        精排阶段：在粗召回候选集合内，用更细粒度的得分函数重打分。

        支持两种精排方法（SearchConfig.DIM_RERANK_METHOD 选择）：
          1. "tag_sim":  query 向量 vs 维度-标签向量均值
                          S = (1/|A_D|) * sum( sim(v_q, v_{m,t}) )
          2. "tag_match": 逐维标签匹配得分
                          S = sum_m max_{t_q in Q_m, t_D in T_D[m]} sim(v_{t_q}, v_{t_D})

        两阶段内部融合（SearchConfig.DIM_INTERNAL_FUSION）：
          "rrf":  粗排 recall_score + 精排 rerank_score 做一次 RRF
          "score": 直接用精排 rerank_score 覆盖粗排得分
        """
        rerank_method = SearchConfig.DIM_RERANK_METHOD
        internal_fusion = SearchConfig.DIM_INTERNAL_FUSION

        rerank_scores: Dict[str, float] = {}
        if rerank_method == "tag_sim":
            rerank_scores = self._rerank_by_tag_sim(query_vec, raw_candidates, recalled_tag_hits)
        elif rerank_method == "tag_match":
            rerank_scores = self._rerank_by_tag_match(query_text, raw_candidates)
        else:
            rerank_scores = {c["chunk_id"]: c.get("recall_score", 0.0) for c in raw_candidates}

        if internal_fusion == "rrf":
            results = self._dim_internal_rrf(raw_candidates, rerank_scores, top_k)
        else:
            results = self._dim_replace_with_rerank(raw_candidates, rerank_scores, top_k)
        return results

    def _rerank_by_tag_sim(
        self,
        query_vec: list,
        raw_candidates: list,
        recalled_tag_hits: list,
    ) -> Dict[str, float]:
        """
        精排方法一：query 向量与维度-标签向量均值。

        S(q,D) = (1/|A_D|) * sum_{(m,t) in A_D} sim(v_q, v_{m,t})

        其中 A_D^q = 候选 chunk D 在粗召回中命中的 (dim, tag) 集合。
        若有 tag_vectors，用向量点积计算；否则用粗召回 tag_hit 的 score 作为近似。
        """
        if not raw_candidates:
            return {}

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

                if self.tag_vectors:
                    key = (dm, tn)
                    if key in self.tag_vectors:
                        import numpy as np
                        tag_vec = self.tag_vectors[key]
                        if isinstance(tag_vec, np.ndarray):
                            q_arr = np.array(query_vec)
                            tag_sim = float(np.dot(q_arr, tag_vec) / (np.linalg.norm(q_arr) * np.linalg.norm(tag_vec) + 1e-8))
                        else:
                            tag_sim = score
                    else:
                        tag_sim = score
                else:
                    tag_sim = score

                tag_sim_sum += tag_sim
                tag_count += 1

            rerank_scores[cid] = (tag_sim_sum / tag_count) if tag_count > 0 else c.get("recall_score", 0.0)

        return rerank_scores

    def _rerank_by_tag_match(
        self,
        query_text: str,
        raw_candidates: list,
    ) -> Dict[str, float]:
        """
        精排方法二：标签匹配得分。

        S(q,D) = sum_{m in M_q} max_{t_q in Q_m, t_D in T_D[m]} sim(v_{t_q}, v_{t_D})

        即：逐维计算 query 标签值和文档标签值的最大相似度并求和。
        若无 tag_vectors，则精确匹配计 1 分，否则计 0 分。
        """
        if not raw_candidates:
            return {}

        constraints = self._parse_query_constraints(query_text)
        if not constraints:
            return {c["chunk_id"]: c.get("recall_score", 0.0) for c in raw_candidates}

        real_dims = {k: v for k, v in constraints.items() if not k.startswith("_")}
        if not real_dims:
            return {c["chunk_id"]: c.get("recall_score", 0.0) for c in raw_candidates}

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
                        if self.tag_vectors:
                            key_q = (dim_name, t_q)
                            key_D = (dim_name, t_D)
                            if key_q in self.tag_vectors and key_D in self.tag_vectors:
                                import numpy as np
                                v_q = np.array(self.tag_vectors[key_q])
                                v_D = np.array(self.tag_vectors[key_D])
                                sim = float(np.dot(v_q, v_D) / (np.linalg.norm(v_q) * np.linalg.norm(v_D) + 1e-8))
                            else:
                                sim = 1.0 if t_q == t_D else 0.0
                        else:
                            sim = 1.0 if t_q == t_D else 0.0
                        if sim > best_sim:
                            best_sim = sim
                dim_score_sum += best_sim

            rerank_scores[cid] = dim_score_sum

        return rerank_scores

    def _dim_internal_rrf(
        self,
        raw_candidates: list,
        rerank_scores: dict,
        top_k: int,
    ) -> list:
        """
        两阶段内部 RRF 融合：粗排 RRF 得分 + 精排归一化得分加权求和。
        """
        k_rrf = SearchConfig.RRF_K
        combined: Dict[str, float] = {}

        for i, c in enumerate(raw_candidates):
            cid = c.get("chunk_id", "")
            if not cid:
                continue
            combined[cid] = combined.get(cid, 0) + 1.0 / (k_rrf + i + 1)

        max_rerank = max(rerank_scores.values()) if rerank_scores else 1.0
        for cid, rs in rerank_scores.items():
            combined[cid] = combined.get(cid, 0) + (rs / max_rerank)

        sorted_cids = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)
        out = []
        by_cid = {c["chunk_id"]: c for c in raw_candidates if c.get("chunk_id")}
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

    def _dim_replace_with_rerank(
        self,
        raw_candidates: list,
        rerank_scores: dict,
        top_k: int,
    ) -> list:
        """
        直接用精排得分覆盖粗排得分。
        """
        for c in raw_candidates:
            cid = c.get("chunk_id", "")
            c["recall_score"] = c.get("recall_score", 0.0)
            c["rerank_score"] = rerank_scores.get(cid, 0.0)
            c["score"] = rerank_scores.get(cid, c.get("recall_score", 0.0))
            c["internal_fusion"] = "score"

        out = sorted(raw_candidates, key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(out):
            r["dim_rank"] = i + 1
        return out[:top_k]

    def _parse_query_constraints(self, query_text: str) -> dict:
        """
        解析 query_text 为维度-标签约束 C_q = {dim_name: [tag_name, ...]}。

        策略（按优先级）：
          1. LLM 抽取（对齐 code/query_parser.py + llm_service.py）：
             调用 DimensionMiningWithQwen.parse_query_intent()，
             传入所有维度名 + 每个维度的候选枚举值列表，
             让 LLM 理解自然语言 query 中隐含的维度约束。
             结果缓存到 _llm_constraint_cache（文件持久化）。

          2. 启发式兜底（原有逻辑）：
             - 优先用 inverted_index 做字符串子串匹配
             - 无索引时扫描 dimension_tags collection 缓存

        返回 {"dim_name": ["tag1", "tag2", ...], ...}
        """
        if not query_text:
            return {}

        # ---- 策略 1: LLM 抽取（带缓存）----
        llm_result = self._parse_by_llm(query_text)
        if llm_result:
            return llm_result

        # ---- 策略 2: 启发式匹配 ----
        if self.inverted_index:
            return self._parse_constraints_by_inverted_index(query_text)
        else:
            return self._parse_constraints_by_scan(query_text)

    def _parse_by_llm(self, query_text: str) -> Optional[dict]:
        """
        调用 LLM 抽取维度约束，对齐 code/llm_service.py 的 parse_query_intent。
        缓存到 _llm_constraint_cache（内存）+ _llm_cache_path（磁盘）。
        """
        try:
            import hashlib
            q_hash = hashlib.md5(query_text.encode("utf-8")).hexdigest()
        except Exception:
            q_hash = str(hash(query_text))

        # 读缓存（内存优先，磁盘兜底）
        if hasattr(self, "_llm_constraint_cache") and q_hash in self._llm_constraint_cache:
            return self._llm_constraint_cache[q_hash]
        disk_cache = self._load_llm_cache_from_disk()
        if q_hash in disk_cache:
            if not hasattr(self, "_llm_constraint_cache"):
                self._llm_constraint_cache = {}
            self._llm_constraint_cache[q_hash] = disk_cache[q_hash]
            return disk_cache[q_hash]

        # 初始化 LLM 服务
        llm = self._get_llm_parser()
        if llm is None:
            return None

        # 构建枚举值映射（从 dim_meta）
        enum_values_map = {}
        all_dims = []
        if self.dim_meta:
            for dim_name, meta in self.dim_meta.items():
                if not isinstance(meta, dict):
                    continue
                vals = meta.get("values") or []
                if vals:
                    enum_values_map[dim_name] = vals[:50]   # 限制候选数量避免 token 爆炸
                else:
                    all_dims.append(dim_name)
        else:
            all_dims = list(self.inverted_index.keys()) if self.inverted_index else []

        dims = list(enum_values_map.keys()) + all_dims
        if not dims:
            return None

        try:
            result = llm.parse_query_intent(
                query_text=query_text,
                dims=dims,
                enum_values_map=enum_values_map,
                schema_dim_fields=None,
            )
        except Exception as e:
            print(f"[LLM 约束抽取失败] {e}")
            return None

        if not isinstance(result, dict) or not result:
            return None

        # 清理：只保留与候选 tag 精确匹配的值
        cleaned: Dict[str, list] = {}
        if self.dim_meta:
            for dim, vals in result.items():
                if dim not in self.dim_meta:
                    continue
                candidates = set(self.dim_meta[dim].get("values") or [])
                if not candidates:
                    continue
                matched = []
                for v in (vals if isinstance(vals, list) else [vals]):
                    v = str(v).strip()
                    if v in candidates:
                        matched.append(v)
                if matched:
                    cleaned[dim] = matched
        else:
            cleaned = {k: v for k, v in result.items() if v}

        if not cleaned:
            return None

        # 写缓存
        if not hasattr(self, "_llm_constraint_cache"):
            self._llm_constraint_cache = {}
        self._llm_constraint_cache[q_hash] = cleaned
        self._save_llm_cache_to_disk(q_hash, cleaned)

        return cleaned

    def _get_llm_parser(self):
        """懒加载 LLM parser（线程安全）。"""
        if hasattr(self, "_llm_parser") and self._llm_parser is not None:
            return self._llm_parser
        try:
            from llm_service import DimensionMiningWithQwen
            self._llm_parser = DimensionMiningWithQwen()
            return self._llm_parser
        except ImportError as e:
            print(f"[LLM 导入失败] {e}，constraint 粗排回退到启发式")
            self._llm_parser = None
            return None
        except Exception as e:
            print(f"[LLM 初始化失败] {e}")
            self._llm_parser = None
            return None

    @property
    def _llm_cache_path(self) -> Path:
        return EXPERIMENT_DATA / "llm_query_constraints_cache.json"

    def _load_llm_cache_from_disk(self) -> dict:
        p = self._llm_cache_path
        if not p.exists():
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_llm_cache_to_disk(self, q_hash: str, constraints: dict):
        p = self._llm_cache_path
        try:
            cache = {}
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            cache[q_hash] = constraints
            with open(p, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
        except Exception:
            pass

    def _parse_constraints_by_inverted_index(self, query_text: str) -> dict:
        """启发式匹配：从 inverted_index 做字符串子串匹配。"""
        constraints: Dict[str, set] = {}
        q_lower = query_text.lower()
        for dim_name, tags in self.inverted_index.items():
            if not isinstance(tags, dict):
                continue
            for tag_name in tags:
                if tag_name and tag_name.lower() in q_lower:
                    constraints.setdefault(dim_name, set()).add(tag_name)
        return {k: sorted(v) for k, v in constraints.items() if v}

    def _parse_constraints_by_scan(self, query_text: str) -> dict:
        """
        当本地无倒排索引时，从 dimension_tags collection 中扫描匹配。
        仅在首次调用时缓存全量 tag。
        """
        if not hasattr(self, "_cached_tags") or self._cached_tags is None:
            try:
                body = {"limit": 10000, "with_payload": True, "with_vector": False}
                data = self._post_json(
                    f"/collections/{self.dim_tags_collection}/points/scroll", body
                )
                points = data.get("result", {}).get("points", [])
                tag_index: Dict[str, set] = {}
                for pt in points:
                    pl = pt.get("payload") or {}
                    dm = pl.get("dim_name", "")
                    tn = pl.get("tag_name", "")
                    if dm and tn:
                        tag_index.setdefault(dm, set()).add(tn)
                self._cached_tags = tag_index
            except Exception:
                self._cached_tags = {}
                return {}

        q_lower = query_text.lower()
        constraints: Dict[str, set] = {}
        for dim_name, tags in self._cached_tags.items():
            for tag_name in tags:
                if tag_name and tag_name.lower() in q_lower:
                    constraints.setdefault(dim_name, set()).add(tag_name)

        return {k: sorted(v) for k, v in constraints.items() if v}

    def _post_json(self, path: str, body: dict, timeout: float = 30.0) -> dict:
        """发送 POST 请求到 Qdrant。"""
        import httpx
        base = self.client.base
        r = httpx.post(base + path, json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()

    # ---------- 双路融合自适应权重（对应 semantic_dimension_fusion_strategy.md 3.2 节） ----------
    def compute_adaptive_alpha(
        self,
        dim_results: list,
        sem_results: list,
        constraints: dict,
    ) -> tuple[float, float]:
        """
        根据 3.2 节的规则型自适应权重方案，
        综合结构化置信度、标签证据置信度、集中度，动态计算 alpha_dim 和 alpha_sem。

        返回 (alpha_dim, alpha_sem)
        """
        # Step 1: 结构化置信度 P_q
        P_q = self._compute_structural_confidence(constraints)

        # Step 2: 标签证据置信度 T_q
        T_q = self._compute_label_evidence_confidence(dim_results, constraints)

        # Step 3: 集中度
        C_dim, C_sem = self._compute_concentration(dim_results, sem_results)

        # Step 4: 效用
        U_dim = P_q + T_q + C_dim
        U_sem = (1 - P_q) + (1 - T_q) + C_sem

        # Step 5: 归一化
        eps = 1e-6
        alpha_dim = U_dim / (U_dim + U_sem + eps)
        alpha_sem = 1.0 - alpha_dim

        return alpha_dim, alpha_sem

    def _compute_structural_confidence(self, constraints: dict) -> float:
        """
        结构化置信度 P_q。

        P_q = 2*n_b / (n_d + n_b + 1)  (当 n_d > 0)
            = 0                         (当 n_d = 0)

        n_d: query 解析出的有效维度数量（去重）
        n_b: 至少绑定了一个维度值的维度数量
        """
        real_dims = {k: v for k, v in constraints.items() if not k.startswith("_")}
        n_d = len(real_dims)
        if n_d == 0:
            return 0.0

        n_b = sum(1 for tags in real_dims.values() if tags)
        return (2.0 * n_b) / (n_d + n_b + 1.0)

    def _compute_label_evidence_confidence(
        self,
        dim_results: list,
        constraints: dict,
    ) -> float:
        """
        标签证据置信度 T_q。

        T_q = 2*E_T / (|M_q| + E_T + 1)  (当 |M_q| > 0)
            = 0                          (当 |M_q| = 0)

        E_T = sum_{m in M_q} s_m
        s_m = max_{D in K_dim, t_q in Q_m, t_D in T_D[m]} sim(v_{t_q}, v_{t_D})
              (当存在 D in K_dim 使得 T_D[m] != empty)
              0 otherwise
        """
        real_dims = {k: v for k, v in constraints.items() if not k.startswith("_")}
        M_q = list(real_dims.keys())
        if not M_q:
            return 0.0

        # 构建候选 chunk 的标签集合 T_D[m]
        doc_tags_by_dim: Dict[str, set] = {}
        for r in dim_results:
            for th in r.get("tag_hits", []):
                dm = th.get("dim", "")
                tn = th.get("tag", "")
                if dm:
                    doc_tags_by_dim.setdefault(dm, set()).add(tn)

        # 逐维计算 s_m
        E_T = 0.0
        for m in M_q:
            Q_m = real_dims.get(m, [])
            if not Q_m:
                continue
            T_D = doc_tags_by_dim.get(m, set())
            if not T_D:
                continue

            best_sim = 0.0
            for t_q in Q_m:
                for t_D in T_D:
                    if self.tag_vectors:
                        key_q = (m, t_q)
                        key_D = (m, t_D)
                        if key_q in self.tag_vectors and key_D in self.tag_vectors:
                            import numpy as np
                            v_q = np.array(self.tag_vectors[key_q])
                            v_D = np.array(self.tag_vectors[key_D])
                            sim = float(np.dot(v_q, v_D) / (np.linalg.norm(v_q) * np.linalg.norm(v_D) + 1e-8))
                        else:
                            sim = 1.0 if t_q == t_D else 0.0
                    else:
                        sim = 1.0 if t_q == t_D else 0.0
                    if sim > best_sim:
                        best_sim = sim
            E_T += best_sim

        return (2.0 * E_T) / (len(M_q) + E_T + 1.0)

    def _compute_concentration(
        self,
        dim_results: list,
        sem_results: list,
    ) -> tuple[float, float]:
        """
        单路 Top 结果集集中度 C_r。

        对一路 r，取 top_k 个结果，计算归一化熵：

        C_r = 1 - H_normalized
            = 1 - (-sum(p_i * log(p_i))) / log(|K_r|)

        极端情况：
          |K_r| == 1: C_r = 1
          |K_r| == 0 或总分 == 0: C_r = 0
        """
        import math
        C_dim = self._concentration_single(dim_results)
        C_sem = self._concentration_single(sem_results)
        return C_dim, C_sem

    def _concentration_single(self, results: list) -> float:
        """计算单路集中度。"""
        import math
        if not results:
            return 0.0

        total = sum(r.get("score", 0.0) for r in results)
        if total <= 0:
            return 0.0

        n = len(results)
        if n == 1:
            return 1.0

        # 归一化得分
        probs = [r.get("score", 0.0) / total for r in results]
        # 熵
        H = 0.0
        for p in probs:
            if p > 0:
                H -= p * math.log(p)

        # 归一化熵
        H_max = math.log(n)
        C = 1.0 - (H / H_max)
        return C

    # ---------- 融合 ----------
    def _fuse(self, dim_results: list, sem_results: list, alpha: float,
              top_k: int, fusion: str) -> list:
        if fusion == "sem_only":
            return sem_results[:top_k]
        if fusion == "dim_only":
            return dim_results[:top_k]

        # 双路得分归一化函数
        def normalize_scores(results: list) -> Dict[str, float]:
            """将原始得分归一化到 [0,1] 区间。"""
            if not results:
                return {}
            scores = [r.get("score", 0.0) for r in results]
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return {r.get("chunk_id", ""): 1.0 for r in results}
            out = {}
            for r in results:
                cid = r.get("chunk_id", "")
                out[cid] = (r.get("score", 0.0) - min_s) / (max_s - min_s)
            return out

        if fusion == "adaptive":
            # 自适应权重融合（3.2 规则型方案）
            constraints = {}
            for r in dim_results:
                dm = r.get("dim_name") or ""
                tn = r.get("tag_name") or ""
                if dm:
                    constraints.setdefault(dm, set()).add(tn)
            for k in constraints:
                constraints[k] = sorted([x for x in constraints[k] if x])[:8]

            alpha_dim, alpha_sem = self.compute_adaptive_alpha(dim_results, sem_results, constraints)
            alpha_sem_weight = alpha_sem
            alpha_dim_weight = alpha_dim
        elif fusion == "score":
            # 得分直接加权融合（3.1 方案）：R_sem 和 R_dim 先归一化，再加权
            dim_norm = normalize_scores(dim_results)
            sem_norm = normalize_scores(sem_results)
            scores = {}
            by_id: Dict[str, dict] = {}
            for r in dim_results + sem_results:
                cid = r.get("chunk_id", "")
                if cid not in by_id:
                    by_id[cid] = r
                scores[cid] = scores.get(cid, 0.0) + alpha * dim_norm.get(cid, 0.0) + \
                              (1.0 - alpha) * sem_norm.get(cid, 0.0)
            sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            result = []
            for cid in sorted_ids[:top_k]:
                src = by_id.get(cid, {})
                merged = dict(src)
                merged["score"] = scores[cid]
                merged["final_score"] = scores[cid]
                merged["source"] = ("dim" if cid in dim_norm else "") + \
                                  ("+sem" if cid in sem_norm else "")
                merged["alpha_dim"] = alpha
                merged["alpha_sem"] = 1.0 - alpha
                result.append(merged)
            return result
        else:
            # 默认 RRF 融合
            fused = rrf_fuse_all(dim_results, sem_results, k=SearchConfig.RRF_K, alpha=alpha)
            alpha_sem_weight = 1.0 - alpha
            alpha_dim_weight = alpha
            by_id: Dict[str, dict] = {}
            for r in dim_results + sem_results:
                cid = r.get("chunk_id", "")
                if cid and cid not in by_id:
                    by_id[cid] = r
            result = []
            for i, fr in enumerate(fused):
                src = by_id.get(fr["chunk_id"], {})
                merged = dict(src)
                merged["score"] = fr["score"]
                merged["final_score"] = fr["score"]
                merged["source"] = ("dim" if any(r.get("chunk_id") == fr["chunk_id"] for r in dim_results) else "") + \
                                  ("+sem" if any(r.get("chunk_id") == fr["chunk_id"] for r in sem_results) else "")
                result.append(merged)
                if len(result) >= top_k:
                    break
            result[0]["alpha_dim"] = alpha_dim_weight
            result[0]["alpha_sem"] = alpha_sem_weight
            return result

        # adaptive / rrf 分支的共同逻辑
        if fusion == "adaptive":
            fused = rrf_fuse_all(dim_results, sem_results, k=SearchConfig.RRF_K, alpha=alpha_sem_weight)

        by_id: Dict[str, dict] = {}
        for r in dim_results + sem_results:
            cid = r.get("chunk_id", "")
            if cid and cid not in by_id:
                by_id[cid] = r

        result = []
        for i, fr in enumerate(fused):
            src = by_id.get(fr["chunk_id"], {})
            merged = dict(src)
            merged["score"] = fr["score"]
            merged["final_score"] = fr["score"]
            merged["source"] = ("dim" if any(r.get("chunk_id") == fr["chunk_id"] for r in dim_results) else "") + \
                              ("+sem" if any(r.get("chunk_id") == fr["chunk_id"] for r in sem_results) else "")
            merged["alpha_dim"] = alpha_dim_weight
            merged["alpha_sem"] = alpha_sem_weight
            result.append(merged)
            if len(result) >= top_k:
                break
        return result


class SemanticSearcher:
    """语义检索器"""
    def __init__(self, collection_name: str = None):
        self._dim_searcher = DimensionAwareSearch(collection_name=collection_name)

    def search(self, query: str, top_k: int = None, **kwargs):
        return self._dim_searcher.search(
            query,
            top_k=top_k or SearchConfig.SEM_TOP_K,
            fusion="sem_only",
            alpha=0.0,
            **kwargs
        )


# DimensionSearcher 保留别名，与原版一致
DimensionSearcher = DimensionAwareSearch


__all__ = [
    "DimensionSearcher",
    "SemanticSearcher",
    "DimensionAwareSearch",
    "rrf_fuse_all",
    "SearchConfig",
    "_load_bge_encoder",
    "_RateLimiter",
]
