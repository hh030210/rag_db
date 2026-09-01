"""
Qdrant 向量数据库客户端

统一管理 Qdrant 连接，支持：
- Collection 创建/删除/列表
- 向量 upsert / delete / search
- payload 查询
"""

import sys
import uuid
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# 路径兼容：支持从 fusion_app/ 或 RAG_DB_slim/ 引用
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from config import get_fusion_config
    _cfg = get_fusion_config()
    _qdrant_cfg = _cfg.vecdb_qdrant
except Exception:
    _qdrant_cfg = None


class QdrantClient:
    """Qdrant 客户端封装"""

    _instance: Optional["QdrantClient"] = None

    def __init__(self, config=None):
        cfg = config or _qdrant_cfg
        if cfg is None:
            raise RuntimeError("Qdrant 配置未初始化，请先调用 init() 或传入 config")

        self.url = f"http://{cfg.host}:{cfg.port}"
        self.grpc_port = getattr(cfg, "grpc_port", 6334)
        self.collection_name = getattr(cfg, "collection_name", "rag_chunks")
        self.vector_dim = getattr(cfg, "vector_dim", 1024)
        self.distance = getattr(cfg, "distance", "Cosine")
        self.hnsw_ef_construct = getattr(cfg, "hnsw_ef_construct", 512)
        self.hnsw_m = getattr(cfg, "hnsw_m", 16)
        self.timeout = 60

        self._client = None
        self._connect()

    def _connect(self):
        try:
            from qdrant_client import QdrantClient as _QC
            self._client = _QC(
                url=self.url,
                timeout=self.timeout,
                prefer_grpc=True,
            )
            print(f"[Qdrant] 连接成功: {self.url}")
        except ImportError:
            print("[Qdrant] qdrant-client 未安装")
            self._client = None
        except Exception as e:
            print(f"[Qdrant] 连接失败: {e}")
            self._client = None

    @property
    def client(self):
        return self._client

    def is_connected(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    # ==================== Collection 管理 ====================

    def list_collections(self) -> List[str]:
        """列出所有 Collection"""
        if not self._client:
            return []
        try:
            result = self._client.get_collections()
            return [c.name for c in result.collections]
        except Exception as e:
            print(f"[Qdrant] list_collections 失败: {e}")
            return []

    def collection_exists(self, name: str = None) -> bool:
        name = name or self.collection_name
        return name in self.list_collections()

    def create_collection(
        self,
        name: str = None,
        collection_name: str = None,
        vector_dim: int = None,
        distance: str = None,
        force: bool = False,
        vector_fields: Dict[str, int] = None,
    ) -> Dict[str, Any]:
        """
        创建 Collection

        Args:
            name: Collection 名称（默认用配置中的）
            collection_name: Collection 名称（name 的别名）
            vector_dim: 向量维度（默认从模型获取）
            distance: 距离度量（Cosine/Euclid/Dot）
            force: 是否强制重建
            vector_fields: 多向量字段配置，如 {"chunk_text_vec": 1024}
        """
        if not self._client:
            return {"status": "error", "message": "Qdrant 未连接"}

        name = collection_name or name or self.collection_name
        vector_dim = vector_dim or self.vector_dim
        distance = distance or self.distance

        try:
            if self.collection_exists(name):
                if force:
                    self._client.delete_collection(collection_name=name)
                    print(f"[Qdrant] 已删除旧 Collection: {name}")
                else:
                    return {"status": "exists", "message": f"Collection 已存在: {name}"}

            dist_map = {"Cosine": "Cosine", "Euclid": "Euclid", "Dot": "Dot"}
            qdrant_dist = dist_map.get(distance, "Cosine")

            from qdrant_client.models import VectorParams, Distance as QdrantDistance

            if vector_fields:
                vectors_config = {}
                for field, dim in vector_fields.items():
                    vectors_config[field] = VectorParams(size=dim, distance=QdrantDistance[qdrant_dist.upper()])
            else:
                vectors_config = {
                    "chunk_text_vec": VectorParams(size=vector_dim, distance=QdrantDistance[qdrant_dist.upper()])
                }

            self._client.create_collection(
                collection_name=name,
                vectors_config=vectors_config,
                hnsw_config={
                    "m": self.hnsw_m,
                    "ef_construct": self.hnsw_ef_construct,
                },
                optimizers_config={"indexing_threshold": 20000},
            )
            print(f"[Qdrant] 创建 Collection: {name}, 维度={vector_dim}, 距离={qdrant_dist}")
            return {"status": "created", "message": f"Collection {name} 创建成功"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def delete_collection(self, name: str = None, collection_name: str = None) -> Dict[str, Any]:
        """删除 Collection"""
        if not self._client:
            return {"status": "error", "message": "Qdrant 未连接"}
        name = collection_name or name or self.collection_name
        try:
            if not self.collection_exists(name):
                return {"status": "not_found", "message": f"Collection 不存在: {name}"}
            self._client.delete_collection(collection_name=name)
            print(f"[Qdrant] 删除 Collection: {name}")
            return {"status": "deleted", "message": f"Collection {name} 已删除"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_collection_info(self, name: str = None) -> Dict[str, Any]:
        """获取 Collection 信息"""
        if not self._client:
            return {}
        name = name or self.collection_name
        try:
            info = self._client.get_collection(collection_name=name)
            return {
                "name": name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception:
            return {"name": name, "vectors_count": 0, "points_count": 0, "status": "not_found"}

    # ==================== 向量操作 ====================

    def upsert(
        self,
        points: List[Dict[str, Any]],
        collection_name: str = None,
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """
        批量写入向量数据

        Args:
            points: 数据点列表，每项包含:
                - id: str, 向量 ID
                - vector: dict, 向量字典（如 {"chunk_text_vec": [0.1, ...]}）
                - payload: dict, 元数据（doc_id, chunk_text, doc_title 等）
            collection_name: Collection 名称
            batch_size: 每批写入数量
        """
        if not self._client:
            return {"status": "error", "message": "Qdrant 未连接"}

        name = collection_name or self.collection_name

        try:
            from qdrant_client.models import PointStruct, VectorParams, Distance

            all_points = []
            for pt in points:
                raw_pid = pt.get("id") or str(uuid.uuid4())
                point_id = uuid.UUID(raw_pid) if isinstance(raw_pid, str) else raw_pid
                vector = pt.get("vector", {})
                payload = pt.get("payload", {})

                if isinstance(vector, list):
                    vector = {"chunk_text_vec": vector}

                all_points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            total = len(all_points)
            for i in range(0, total, batch_size):
                batch = all_points[i : i + batch_size]
                self._client.upsert(collection_name=name, points=batch)

            print(f"[Qdrant] 写入 {total} 条向量到 {name}")
            return {"status": "success", "count": total}

        except Exception as e:
            print(f"[Qdrant] upsert 失败: {e}")
            return {"status": "error", "message": str(e)}

    def search(
        self,
        query_vector: Union[List[float], Dict[str, List[float]]],
        collection_name: str = None,
        limit: int = 10,
        score_threshold: float = None,
        query_filter: Dict = None,
        with_payload: bool = True,
        using: str = None,
    ) -> List[Dict[str, Any]]:
        """
        向量相似度检索

        Args:
            query_vector: 查询向量（可以是单向量 dict 或多向量 dict）
            collection_name: Collection 名称
            limit: 返回数量
            score_threshold: 最低分数阈值
            query_filter: Qdrant filter 条件
            with_payload: 是否返回 payload
            using: 使用的向量字段（如 "chunk_text_vec"）
        """
        if not self._client:
            return []

        name = collection_name or self.collection_name

        try:
            search_params = {
                "collection_name": name,
                "query": query_vector,
                "limit": limit,
            }

            if using:
                search_params["using"] = using
            if with_payload:
                search_params["with_payload"] = True

            resp = self._client.query_points(**search_params)

            output = []
            for hit in resp.points:
                item = {
                    "id": str(hit.id),
                    "score": hit.score,
                    "payload": hit.payload if with_payload else {},
                }
                output.append(item)

            return output

        except Exception as e:
            print(f"[Qdrant] search 失败: {e}")
            return []

    def search_batch(
        self,
        queries: List[Dict[str, Any]],
        collection_name: str = None,
        limit: int = 10,
    ) -> List[List[Dict[str, Any]]]:
        """
        批量检索

        Args:
            queries: 查询列表，每项包含:
                - query_vector: List[float] 或 Dict
                - filter: 可选的 filter 条件
        """
        if not self._client:
            return [[] for _ in queries]

        name = collection_name or self.collection_name

        try:
            from qdrant_client.models import SearchRequest

            requests = []
            for q in queries:
                req = SearchRequest(
                    vector=q.get("query_vector"),
                    limit=limit,
                    with_payload=True,
                )
                if "filter" in q:
                    req.filter = q["filter"]
                if "using" in q:
                    req.using = q["using"]
                requests.append(req)

            results = self._client.search_batch(
                collection_name=name, requests=requests
            )

            output = []
            for batch in results:
                batch_results = []
                for hit in batch:
                    batch_results.append({
                        "id": str(hit.id),
                        "score": hit.score,
                        "payload": hit.payload,
                    })
                output.append(batch_results)

            return output

        except Exception as e:
            print(f"[Qdrant] search_batch 失败: {e}")
            return [[] for _ in queries]

    def delete_points(self, point_ids: List[str], collection_name: str = None) -> Dict[str, Any]:
        """删除指定 ID 的向量"""
        if not self._client:
            return {"status": "error", "message": "Qdrant 未连接"}
        name = collection_name or self.collection_name
        try:
            self._client.delete(
                collection_name=name,
                points_selector={"points": point_ids},
            )
            return {"status": "success", "deleted": len(point_ids)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def scroll(
        self,
        collection_name: str = None,
        limit: int = 100,
        offset: str = None,
        with_payload: bool = True,
    ) -> Dict[str, Any]:
        """滚动获取所有点（用于全量导出/迁移）"""
        if not self._client:
            return {"points": [], "next_page_offset": None}
        name = collection_name or self.collection_name
        try:
            results, next_offset = self._client.scroll(
                collection_name=name,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
            )
            points = []
            for hit in results:
                points.append({
                    "id": str(hit.id),
                    "vector": hit.vector,
                    "payload": hit.payload,
                })
            return {"points": points, "next_page_offset": next_offset}
        except Exception as e:
            return {"points": [], "next_page_offset": None, "error": str(e)}


# ==================== 单例管理器 ====================

_qdrant_instance: Optional[QdrantClient] = None


def init_qdrant(config=None) -> QdrantClient:
    """初始化 Qdrant 客户端（单例）"""
    global _qdrant_instance
    _qdrant_instance = QdrantClient(config)
    return _qdrant_instance


def get_qdrant() -> Optional[QdrantClient]:
    """获取 Qdrant 客户端实例"""
    global _qdrant_instance
    if _qdrant_instance is None:
        try:
            _qdrant_instance = init_qdrant()
        except Exception as e:
            print(f"[Qdrant] 初始化失败: {e}")
    return _qdrant_instance
