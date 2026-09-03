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
import math
import re
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
    # 向量兜底只在同一维度内生效；精确/别名命中不受该阈值限制。
    SOFT_MATCH_THRESHOLD = 0.58
    DIM_TAG_SIM_THRESHOLD = 0.58
    DIM_TAG_MARGIN = 0.04
    DIM_TAG_TOP_K_PER_DIM = 5
    DIM_MAX_ACTIVE_DIMS = 3
    DIM_ENTITY_BONUS = 0.12
    DIM_COVERAGE_BONUS = 0.08
    SPOT_FILTER_POOL_MULTIPLIER = 5
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

    def collection_exists(self, collection: str) -> bool:
        """检查 collection 是否存在，避免每条 query 重复触发 404 回退。"""
        import httpx
        try:
            response = httpx.get(self.base + f"/collections/{collection}", timeout=5.0)
            return response.is_success
        except Exception:
            return False

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

    sem：直接在语料 collection 上做 chunk_text_vec 向量检索
    dim：优先读取语料 point 的 dim_* payload 字段，按查询激活维度做
         精确/别名匹配和同维度向量兜底，再展开 chunk_ids 取出全文。
         外层问答流程负责一次归一化融合。
    """

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or _qdrant_cfg["collection_name"]
        self.dim_tags_collection = _qdrant_cfg.get("dim_tags_collection", "dimension_tags")
        print(f">>> 初始化检索器 (Qdrant)...")
        print(f"    语料: {self.collection_name}  维度标签: {self.dim_tags_collection}")

        self.client = _QdrantClient(_qdrant_cfg["host"], _qdrant_cfg["port"])
        self._dim_tags_collection_available = self.client.collection_exists(
            self.dim_tags_collection
        )
        if not self._dim_tags_collection_available:
            print("    独立维度标签 collection 不可用，将直接使用语料 payload 维度字段")

        # BGE-M3 编码器
        self.encoder = _load_bge_encoder()
        if self.encoder is None:
            print("[警告] 编码器未加载，将无法进行向量检索")

        # 维度检索 metadata（dim_name → 是否启用），保留兼容
        self.inverted_index: Dict = {}
        self.dim_meta: Dict = {}
        self.tag_vectors: Dict = {}
        # 新版 Qdrant 将维度标签直接写入 unified_corpus 的 dim_* payload
        # 字段，不再强制依赖独立的 dimension_tags collection。
        self._payload_dim_tags: Dict[str, Set[str]] = {}
        self._payload_tag_points: Dict[tuple, Set[str]] = {}
        self._payload_tags_by_chunk: Dict[str, list] = {}
        self._payload_tag_vectors: Dict[tuple, list] = {}
        self._payload_spot_by_chunk: Dict[str, str] = {}
        self._payload_spot_chunks: Dict[str, Set[str]] = {}
        self._payload_spot_entity_index: Dict[str, Set[str]] = {}
        self._payload_spot_aliases: Dict[str, Set[str]] = {}
        self._payload_index_loaded = False
        self._payload_tag_vectors_loaded = False
        self._load_indexes()
        self._load_payload_dimension_index()

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

    @staticmethod
    def _split_payload_tags(value: Any) -> list[str]:
        """解析 Qdrant dim_* 字段中的单值、多值和分号连接值。"""
        if value is None:
            return []
        if isinstance(value, dict):
            value = list(value.values())
        if isinstance(value, (list, tuple, set)):
            raw_values = value
        else:
            raw_values = re.split(r"[;；]", str(value))
        out = []
        for item in raw_values:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
        return out

    # 查询分析使用的词典只负责“识别”，不负责直接决定答案。这样可以把
    # 查询解析与召回结果解耦，避免先召回错误 chunk 再反向制造 constraints。
    _DIMENSION_CUES = {
        "历史沿革": "历史 始建 建于 营建 修建 发展 起源 年代 朝代 恢复 历时 过程 原因 为何 为什么 生前".split(),
        "历史事件": "历史 始建 建于 营建 修建 发展 起源 年代 朝代 恢复 历时 过程 原因 为何 为什么".split(),
        "建筑形制": "建筑 殿 门 塔 墙 桥 布局 结构 形制 规模 面积 米 石像生 神道 顺序 外观".split(),
        "建筑功能": "建筑 殿 门 塔 墙 桥 布局 结构 用途 功能 祭祀 规模 面积".split(),
        "功能用途": "用途 作用 功能 用于 干什么 做什么".split(),
        "文化象征": "寓意 象征 意义 称号 为什么叫 名称由来 代表".split(),
        "人物谱系": "谁 人物 皇帝 皇后 祖先 供奉 合葬 身份 哪位 相关人物".split(),
        "涉及皇帝": "谁 皇帝 皇后 祖先 供奉 合葬 哪位".split(),
        "相关人物": "谁 人物 皇帝 皇后 祖先 供奉 合葬 身份 哪位".split(),
        "地理位置": "位于 哪里 位置 路线 交通 距离 附近 从哪里 地址 方位 地处".split(),
        "行政区域": "位于 哪里 位置 行政 区县 城市 省市 地处".split(),
        "景观特征": "景观 看点 景色 风景 特色 好看 景致 景物".split(),
        "艺术类型": "艺术 雕刻 雕塑 书法 绘画 石刻 造像 艺术类型".split(),
        "宗教内涵": "祭祀 祭孔 宗教 佛 道 供奉 典礼 儒学 礼制".split(),
        "文献典籍": "典籍 论语 文献 书 记载 碑 文献记载".split(),
        "游览服务": "门票 票价 票 预约 费用 收费 开放 服务 讲解 厕所 母婴 餐厅 吃饭 盖章 活动 电话".split(),
        "交通路径类型": "路线 交通 道路 公交 地铁 自驾 怎么去 到达".split(),
        "交通服务类型": "停车场 接驳 公交 观光车 交通 服务".split(),
        "观光交通设施": "停车场 公交车 观光车 索道 缆车 游船 交通设施".split(),
    }
    _SPOT_ALIASES = {
        "明十三陵": {"十三陵", "明十三陵", "十三陵景区"},
        "南孔庙": {"南孔", "南孔庙", "孔氏南宗家庙", "南宗家庙", "衢州南孔庙", "衢州孔庙", "孔庙"},
        "西湖": {"西湖", "杭州西湖"},
        "丽江古城": {"丽江", "丽江古城"},
        "西双版纳热带植物园": {"西双版纳", "西双版纳热带植物园", "热带植物园"},
        "张家界": {"张家界", "张家界国家森林公园"},
    }
    _ENTITY_STOPWORDS = {
        "为什么", "怎么", "如何", "哪个", "哪些", "什么", "是否", "可以", "景区", "景点",
        "历史", "建筑", "功能", "用途", "位置", "地理", "文化", "艺术", "宗教", "典籍",
        "服务", "交通", "相关", "分别", "介绍", "情况", "特点", "信息", "问题", "请问",
    }

    @staticmethod
    def _normalize_match_text(text: Any) -> str:
        """统一中文标签/别名匹配的大小写、空白和标点。"""
        value = str(text or "").lower()
        return re.sub(r"[\s\u3000，。！？；：、（）()【】\[\]{}‘’“”\"'·_-]+", "", value)

    @classmethod
    def _spot_from_payload(cls, payload: dict) -> str:
        """优先读取显式景区字段，否则从文档标题稳定推导景区名。"""
        for key in ("spot_name", "景区名称", "spot", "景区"):
            value = payload.get(key)
            if isinstance(value, (list, tuple, set)):
                value = next(iter(value), "")
            if value and str(value).strip():
                return str(value).strip()
        title = str(payload.get("doc_title") or payload.get("chunk_gen_title") or "").strip()
        if not title:
            return ""
        first = re.split(r"[-—_|｜/:：]", title, maxsplit=1)[0].strip()
        return re.sub(r"(?:景区|知识库|文档)$", "", first).strip()

    @classmethod
    def _canonical_spot(cls, spot: str) -> str:
        normalized = cls._normalize_match_text(spot)
        if not normalized:
            return ""
        for canonical, aliases in cls._SPOT_ALIASES.items():
            if normalized == cls._normalize_match_text(canonical):
                return canonical
            if any(normalized == cls._normalize_match_text(alias) for alias in aliases):
                return canonical
        return str(spot).strip()

    @classmethod
    def _aliases_for_spot(cls, spot: str) -> set[str]:
        canonical = cls._canonical_spot(spot)
        aliases = set(cls._SPOT_ALIASES.get(canonical, set()))
        if canonical:
            aliases.add(canonical)
            aliases.add(re.sub(r"(?:景区|国家森林公园)$", "", canonical))
        return {item for item in aliases if item}

    @classmethod
    def _dimension_cue_score(cls, dim_name: str, query_text: str) -> int:
        q = cls._normalize_match_text(query_text)
        if not q:
            return 0
        cues = cls._DIMENSION_CUES.get(dim_name, [])
        # 未收录的维度允许用维度名本身作弱线索，但不会凭空激活全部维度。
        if not cues:
            cues = [dim_name]
        return sum(1 for cue in cues if cls._normalize_match_text(cue) in q)

    def _register_entity_terms(self, text: str, spot: str):
        """从标题、标签和短正文建立轻量实体->景区索引。"""
        if not spot or not text:
            return
        runs = re.findall(r"[\u4e00-\u9fff]{2,}", str(text))
        for run in runs:
            limit = min(len(run), 8)
            for size in (2, 3, 4):
                if size > limit:
                    continue
                for start in range(0, len(run) - size + 1):
                    term = run[start:start + size]
                    if term in self._ENTITY_STOPWORDS:
                        continue
                    self._payload_spot_entity_index.setdefault(term, set()).add(spot)

    def _load_payload_dimension_index(self):
        """从语料 collection 的 dim_* payload 字段建立轻量维度倒排索引。

        该索引兼容 Step 6 的新版入库方式：每个 chunk point 自带维度列，
        因而维度检索可以直接由这些字段恢复 (dimension, tag) -> chunk_id。
        """
        if self._payload_index_loaded:
            return
        self._payload_index_loaded = True
        try:
            points = self.client.scroll(self.collection_name, limit=10000, with_payload=True)
        except Exception as exc:
            print(f"    [提示] 未能读取 Qdrant dim_* 字段: {exc}")
            return

        for point in points:
            payload = point.get("payload") or {}
            chunk_id = payload.get("chunk_id") or point.get("id")
            if chunk_id is None:
                continue
            chunk_id = str(chunk_id)
            spot = self._canonical_spot(self._spot_from_payload(payload))
            if spot:
                self._payload_spot_by_chunk[chunk_id] = spot
                self._payload_spot_chunks.setdefault(spot, set()).add(chunk_id)
                for alias in self._aliases_for_spot(spot):
                    self._payload_spot_aliases.setdefault(
                        self._normalize_match_text(alias), set()
                    ).add(spot)
                self._register_entity_terms(
                    " ".join(
                        str(payload.get(key) or "")
                        for key in ("doc_title", "chunk_gen_title", "chunk_text")
                    )[:1800],
                    spot,
                )
            for field, value in payload.items():
                if not isinstance(field, str) or not field.startswith("dim_"):
                    continue
                dim_name = field[4:]
                if not dim_name:
                    continue
                tags = self._split_payload_tags(value)
                for tag_name in tags:
                    key = (dim_name, tag_name)
                    self._payload_dim_tags.setdefault(dim_name, set()).add(tag_name)
                    self._payload_tag_points.setdefault(key, set()).add(chunk_id)
                    self._payload_tags_by_chunk.setdefault(chunk_id, []).append(
                        {"dim": dim_name, "tag": tag_name}
                    )

        # 兼容旧版独立 dimension_tags collection。将其平铺到同一套索引后，
        # 后续仍然执行“查询分析 -> 同维度匹配 -> 景区过滤”，不再走全局标签混排。
        if not self._payload_tag_points and self._dim_tags_collection_available:
            try:
                tag_points = self.client.scroll(
                    self.dim_tags_collection, limit=10000, with_payload=True
                )
                for point in tag_points:
                    tag_payload = point.get("payload") or {}
                    dim_name = str(tag_payload.get("dim_name") or "").strip()
                    tag_name = str(tag_payload.get("tag_name") or "").strip()
                    chunk_ids = tag_payload.get("chunk_ids") or []
                    if not dim_name or not tag_name or not isinstance(chunk_ids, list):
                        continue
                    self._payload_dim_tags.setdefault(dim_name, set()).add(tag_name)
                    self._payload_tag_points.setdefault((dim_name, tag_name), set()).update(
                        str(cid) for cid in chunk_ids if cid
                    )
            except Exception as exc:
                print(f"    [提示] 未能读取独立维度标签索引: {exc}")

        if self._payload_dim_tags:
            tag_count = sum(len(values) for values in self._payload_dim_tags.values())
            chunk_count = len(self._payload_tags_by_chunk)
            print(
                f"    Qdrant payload 维度索引: {len(self._payload_dim_tags)} 个维度, "
                f"{tag_count} 个标签, {chunk_count} 个带标签 chunk"
            )
        else:
            print("    [提示] Qdrant 语料中没有非空 dim_* 字段")
        if self._payload_spot_chunks:
            print(f"    景区索引: {len(self._payload_spot_chunks)} 个景区")

    @staticmethod
    def _cosine(left: list, right: list) -> float:
        if not left or not right:
            return 0.0
        size = min(len(left), len(right))
        dot = sum(float(left[i]) * float(right[i]) for i in range(size))
        norm_left = math.sqrt(sum(float(left[i]) ** 2 for i in range(size)))
        norm_right = math.sqrt(sum(float(right[i]) ** 2 for i in range(size)))
        return dot / (norm_left * norm_right + 1e-8)

    def _encode_payload_tags(self):
        """为 payload 标签建立一次性向量缓存；失败时仍可用字符串匹配。"""
        if self._payload_tag_vectors_loaded:
            return
        self._payload_tag_vectors_loaded = True
        if not self.encoder or not self._payload_tag_points:
            return

        labels = [f"{dim}: {tag}" for dim, tag in self._payload_tag_points]
        try:
            cls_name = self.encoder.__class__.__name__
            if cls_name in ("_FlagProxy", "BGEM3FlagModel", "M3Embedder"):
                encoded = self.encoder.encode(labels, return_dense=True)
                vectors = encoded["dense_vecs"]
            else:
                vectors = self.encoder.encode(
                    labels, normalize_embeddings=True, show_progress_bar=False
                )
            for key, vector in zip(self._payload_tag_points, vectors):
                self._payload_tag_vectors[key] = (
                    vector.tolist() if hasattr(vector, "tolist") else list(vector)
                )
            print(f"    payload 标签向量: {len(self._payload_tag_vectors)} 个")
        except Exception as exc:
            # 标签向量只是增强项，不能影响基于 payload 的精确匹配。
            print(f"    [提示] payload 标签向量生成失败，将使用文本匹配: {exc}")

    def _tag_match(self, query_text: str, dim_name: str, tag_name: str) -> tuple[float, str]:
        """返回 (匹配分数, 来源)，来源为 exact、alias 或 vector。"""
        query_norm = self._normalize_match_text(query_text)
        tag_norm = self._normalize_match_text(tag_name)
        if not query_norm or not tag_norm or len(tag_norm) < 2:
            return 0.0, "none"
        if tag_norm in query_norm:
            return 1.0 + min(len(tag_norm) / max(len(query_norm), 1), 0.25), "exact"

        # 维度标签别名只在同一维度内生效，避免“交通/服务”等短词跨维度串线。
        aliases = {
            "门票": {"票价", "票", "收费", "费用"},
            "票价": {"门票", "票", "收费", "费用"},
            "开放时间": {"开放", "营业时间", "开门时间"},
            "地理位置": {"位置", "位于", "在哪里", "地处"},
            "人物谱系": {"谁", "哪位", "人物"},
            "涉及皇帝": {"谁", "哪位", "皇帝"},
            "相关人物": {"谁", "哪位", "人物"},
        }
        for alias in aliases.get(str(tag_name).strip(), set()):
            if self._normalize_match_text(alias) in query_norm:
                return 0.96, "alias"
        return 0.0, "none"

    def _payload_tag_score(
        self,
        query_text: str,
        dim_name: str,
        tag_name: str,
        query_vec: list,
    ) -> tuple[float, str]:
        """先做同维度精确/别名判断，失败后才计算标签向量相似度。"""
        score, source = self._tag_match(query_text, dim_name, tag_name)
        if source != "none":
            return score, source

        tag_vec = self._payload_tag_vectors.get((dim_name, tag_name))
        if tag_vec:
            return self._cosine(query_vec, tag_vec), "vector"

        # 编码器不可用时只返回保守的向量兜底近似值；调用方仍会经过阈值和 margin。
        q_chars = set(self._normalize_match_text(query_text))
        t_chars = set(self._normalize_match_text(tag_name))
        if len(t_chars) < 2:
            return 0.0, "none"
        return len(q_chars & t_chars) / math.sqrt(len(q_chars) * len(t_chars) + 1e-8), "vector"

    def _analyze_query(self, query_text: str) -> dict:
        """按固定顺序独立分析景区、实体和 1～3 个查询维度。"""
        query = str(query_text or "").strip()
        normalized = self._normalize_match_text(query)
        spot_scores: Dict[str, float] = {}
        entity_terms: list[str] = []

        # 1) 显式景区与别名识别。
        for canonical, aliases in self._SPOT_ALIASES.items():
            for alias in aliases | {canonical}:
                alias_norm = self._normalize_match_text(alias)
                if alias_norm and alias_norm in normalized:
                    spot_scores[canonical] = max(
                        spot_scores.get(canonical, 0.0),
                        1.0 + len(alias_norm) / 100.0,
                    )
        for alias_norm, spots in self._payload_spot_aliases.items():
            if alias_norm and alias_norm in normalized:
                for spot in spots:
                    spot_scores[spot] = max(spot_scores.get(spot, 0.0), 1.0 + len(alias_norm) / 100.0)

        # 2) 通过已观察到的实体词反推景区；只接受唯一且明显领先的景区。
        query_runs = re.findall(r"[\u4e00-\u9fff]{2,}", query)
        entity_candidates: Dict[str, Set[str]] = {}
        for run in query_runs:
            for size in (4, 3, 2):
                for start in range(0, max(0, len(run) - size + 1)):
                    term = run[start:start + size]
                    if term in self._ENTITY_STOPWORDS or term in {"问题", "内容", "方面"}:
                        continue
                    spots = self._payload_spot_entity_index.get(term, set())
                    if spots:
                        entity_candidates[term] = set(spots)
        counts: Dict[str, int] = {}
        explicit_spots = set(spot_scores)
        for term, spots in entity_candidates.items():
            if len(spots) != 1:
                continue
            spot = next(iter(spots))
            # 显式景区存在时，只保留属于该景区的实体；否则用于反推景区。
            if explicit_spots and spot not in explicit_spots:
                continue
            entity_terms.append(term)
            counts[spot] = counts.get(spot, 0) + min(len(term), 4)
        if not spot_scores and counts:
            ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            if len(ordered) == 1 or ordered[0][1] >= ordered[1][1] * 1.5:
                spot_scores[ordered[0][0]] = min(0.88, 0.45 + ordered[0][1] / 20.0)
        entity_terms = sorted(set(entity_terms), key=len, reverse=True)[:8]

        spot_names = [spot for spot, _ in sorted(spot_scores.items(), key=lambda item: item[1], reverse=True)]
        spot_confidence = min(1.0, max(spot_scores.values(), default=0.0))

        # 3) 先收集标签精确/别名值，再结合问题词确定最多三个相关维度。
        dimension_values: Dict[str, list[str]] = {}
        dimension_sources: Dict[str, list[str]] = {}
        exact_dim_scores: Dict[str, int] = {}
        for dim_name, tags in self._payload_dim_tags.items():
            for tag_name in tags:
                score, source = self._tag_match(query, dim_name, tag_name)
                if source == "none":
                    continue
                dimension_values.setdefault(dim_name, []).append(tag_name)
                dimension_sources.setdefault(dim_name, []).append(source)
                exact_dim_scores[dim_name] = exact_dim_scores.get(dim_name, 0) + 1

        dim_scores: Dict[str, int] = {}
        for dim_name in self._payload_dim_tags:
            cue_score = self._dimension_cue_score(dim_name, query)
            # 标签显式命中权重高于问题措辞线索。
            dim_scores[dim_name] = cue_score + 3 * exact_dim_scores.get(dim_name, 0)

        ranked_dims = sorted(dim_scores, key=lambda dim: (dim_scores[dim], exact_dim_scores.get(dim, 0)), reverse=True)
        active_dimensions = [
            dim for dim in ranked_dims
            if dim_scores.get(dim, 0) > 0 and dim not in SearchConfig.EXCLUDED_DIMS
        ][:SearchConfig.DIM_MAX_ACTIVE_DIMS]
        # 没有维度线索时，严格返回空集合，由语义检索负责，而不是全库标签向量召回。
        active_set = set(active_dimensions)
        dimension_values = {dim: sorted(set(values)) for dim, values in dimension_values.items() if dim in active_set}
        dimension_sources = {dim: sorted(set(values)) for dim, values in dimension_sources.items() if dim in active_set}
        cue_total = sum(1 for dim in active_dimensions if self._dimension_cue_score(dim, query) > 0)
        exact_total = sum(exact_dim_scores.get(dim, 0) for dim in active_dimensions)
        confidence = min(1.0, 0.45 * spot_confidence + 0.15 * min(cue_total, 3) + 0.15 * min(exact_total, 3))
        if active_dimensions and confidence == 0.0:
            confidence = 0.25

        return {
            "spot_names": spot_names[:3],
            "spot_confidence": round(spot_confidence, 4),
            "entity_terms": entity_terms,
            "active_dimensions": active_dimensions,
            "dimension_values": dimension_values,
            "dimension_sources": dimension_sources,
            "confidence": round(confidence, 4),
            "required_slots": [],
        }

    def _entity_score(self, item: dict, query_analysis: dict) -> float:
        """计算候选是否包含查询识别出的景点/实体，范围固定在 [0, 1]。"""
        terms = query_analysis.get("entity_terms") or []
        if not terms:
            return 1.0 if query_analysis.get("spot_names") and item.get("spot_name") in set(query_analysis.get("spot_names")) else 0.0
        text = self._normalize_match_text(
            " ".join(
                str(item.get(key) or "")
                for key in ("chunk_gen_title", "doc_title", "chunk_text_full", "chunk_text")
            )
        )
        matched = [term for term in terms if self._normalize_match_text(term) in text]
        return min(1.0, max((len(term) for term in matched), default=0) / max(len(terms[0]), 1))

    @staticmethod
    def _dimension_coverage(item: dict, query_analysis: dict) -> float:
        active = set(query_analysis.get("active_dimensions") or [])
        if not active:
            return 0.0
        matched = set(item.get("matched_dimensions") or [])
        if not matched:
            matched = {
                hit.get("dim")
                for hit in item.get("tag_hits", [])
                if hit.get("dim")
            }
        return len(active & matched) / len(active)

    def _dim_recall_by_payload(
        self,
        query_vec: list,
        query_text: str,
        top_k: int,
        query_analysis: Optional[dict] = None,
    ) -> tuple[list, list]:
        """景区硬过滤后，在激活维度内精确/别名优先、向量兜底。"""
        if not self._payload_tag_points:
            return [], []
        self._encode_payload_tags()
        analysis = query_analysis or self._analyze_query(query_text)
        active_dimensions = [
            dim for dim in analysis.get("active_dimensions", [])
            if dim in self._payload_dim_tags and dim not in SearchConfig.EXCLUDED_DIMS
        ]
        if not active_dimensions:
            return [], []

        selected: list[tuple[float, str, str, str]] = []
        allowed_spots = set(analysis.get("spot_names") or [])
        for dim_name in active_dimensions:
            exact_hits = []
            vector_hits = []
            for tag_name in self._payload_dim_tags.get(dim_name, set()):
                score, source = self._payload_tag_score(query_text, dim_name, tag_name, query_vec)
                tag_chunk_ids = self._payload_tag_points.get((dim_name, tag_name), set())
                if allowed_spots:
                    if not any(
                        self._payload_spot_by_chunk.get(str(cid), "") in allowed_spots
                        for cid in tag_chunk_ids
                    ):
                        continue
                if source in {"exact", "alias"}:
                    exact_hits.append((score, dim_name, tag_name, source))
                elif source == "vector":
                    vector_hits.append((score, dim_name, tag_name, source))

            # 一个维度已经有明确值时，不让相邻语义标签污染该维度。
            if exact_hits:
                selected.extend(sorted(exact_hits, reverse=True)[:SearchConfig.DIM_TAG_TOP_K_PER_DIM])
                continue

            vector_hits.sort(reverse=True)
            if not vector_hits:
                continue
            best = vector_hits[0][0]
            second = vector_hits[1][0] if len(vector_hits) > 1 else 0.0
            if best < SearchConfig.DIM_TAG_SIM_THRESHOLD:
                continue
            if len(vector_hits) > 1 and best - second < SearchConfig.DIM_TAG_MARGIN:
                continue
            selected.extend(
                hit for hit in vector_hits[:SearchConfig.DIM_TAG_TOP_K_PER_DIM]
                if hit[0] >= SearchConfig.DIM_TAG_SIM_THRESHOLD
            )

        if not selected:
            return [], []

        all_candidates: Dict[str, dict] = {}
        tag_hits = []
        for tag_score, dim_name, tag_name, match_source in selected:
            chunk_ids = self._payload_tag_points.get((dim_name, tag_name), set())
            if allowed_spots:
                chunk_ids = {
                    cid for cid in chunk_ids
                    if self._payload_spot_by_chunk.get(str(cid), "") in allowed_spots
                }
            if not chunk_ids:
                continue
            hit = {
                "id": f"payload:{dim_name}:{tag_name}",
                "score": tag_score,
                "payload": {
                    "dim_name": dim_name,
                    "tag_name": tag_name,
                    "chunk_ids": sorted(chunk_ids),
                    "match_source": match_source,
                    "raw_score": tag_score,
                },
            }
            tag_hits.append(hit)
            self._merge_candidate(hit, all_candidates)

        if not all_candidates:
            return [], tag_hits
        payloads = self.client.retrieve(self.collection_name, list(all_candidates.keys()))
        return self._build_candidates(payloads, all_candidates, "payload"), tag_hits

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

        query_analysis = kwargs.get("query_analysis") or self._analyze_query(query_text)
        independent_constraints = {
            dim: values
            for dim, values in (query_analysis.get("dimension_values") or {}).items()
            if values
        }
        independent_constraints.update({
            "_active_dimensions": query_analysis.get("active_dimensions", []),
            "_spot_names": query_analysis.get("spot_names", []),
            "_query_confidence": query_analysis.get("confidence", 0.0),
        })
        results = {
            "results": [],
            "constraints": independent_constraints,
            "query_analysis": query_analysis,
            "query_text": query_text,
        }

        query_vec = _encode_query(self.encoder, query_text)
        if query_vec is None:
            return results

        # ---- 语义检索 ----
        sem_results = []
        if fusion != "dim_only":
            try:
                sem_results = self._sem_search_via_qdrant(query_vec, top_k, query_analysis)
                results["constraints"]["_sem_count"] = len(sem_results)
            except Exception as e:
                print(f"[警告] 语义检索失败: {e}")

        # ---- 维度检索 ----
        dim_results = []
        recall_method = SearchConfig.DIM_RECALL_METHOD
        if fusion != "sem_only":
            try:
                dim_results = self._dim_search_via_qdrant(
                    query_vec, query_text, top_k, recall_method=recall_method,
                    query_analysis=query_analysis,
                )
                results["recall_method"] = recall_method
            except Exception as e:
                print(f"[警告] 维度检索失败: {e}")

        results["results"] = self._fuse(dim_results, sem_results, alpha, top_k, fusion)
        results["dim_results"] = dim_results
        return results

    # ---------- 语义子检索（rag_chunks / chunk_text_vec） ----------
    def _sem_search_via_qdrant(
        self,
        query_vec: list,
        top_k: int,
        query_analysis: Optional[dict] = None,
    ) -> list:
        analysis = query_analysis or {}
        allowed_spots = set(analysis.get("spot_names") or [])
        hard_filter = bool(allowed_spots and self._payload_spot_by_chunk)
        search_k = top_k * SearchConfig.SPOT_FILTER_POOL_MULTIPLIER if hard_filter else top_k
        hits = self.client.search(
            collection=self.collection_name,
            query_vector=query_vec,
            vector_name="chunk_text_vec",
            top_k=search_k,
        )
        out = []
        for i, hit in enumerate(hits):
            payload = hit.get("payload") or {}
            cid = payload.get("chunk_id") or hit.get("id")
            spot_name = self._payload_spot_by_chunk.get(str(cid), "") or self._canonical_spot(
                self._spot_from_payload(payload)
            )
            if hard_filter and spot_name not in allowed_spots:
                continue
            item = {
                "chunk_id": cid,
                "chunk_text": payload.get("chunk_text", ""),
                "doc_title": payload.get("doc_title", ""),
                "chunk_gen_title": payload.get("chunk_gen_title", ""),
                "chunk_text_full": payload.get("chunk_text", ""),
                "score": hit.get("score", 0.0),
                "sem_rank": i + 1,
                "source": "sem",
                "spot_name": spot_name,
                "entity_score": self._entity_score(payload, analysis),
            }
            out.append(item)
            if len(out) >= top_k:
                break
        return out

    # ---------- 维度检索主方法（两阶段：召回 + 精排） ----------
    def _dim_search_via_qdrant(
        self,
        query_vec: list,
        query_text: str,
        top_k: int,
        recall_method: str = None,
        query_analysis: Optional[dict] = None,
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
            raw_candidates, recalled_tag_hits = self._dim_recall_by_vec(
                query_vec, query_text, top_k, query_analysis
            )
        elif recall_method == "constraint":
            raw_candidates, recalled_tag_hits = self._dim_recall_by_constraint(
                query_text, top_k, query_analysis
            )
        elif recall_method == "tag":
            raw_candidates, recalled_tag_hits = self._dim_recall_by_tag(
                query_vec, query_text, top_k, query_analysis
            )
        else:
            raw_candidates, recalled_tag_hits = self._dim_recall_by_tag(
                query_vec, query_text, top_k, query_analysis
            )

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
        for item in reranked:
            item.setdefault("entity_score", self._entity_score(item, query_analysis or {}))
            item.setdefault("dimension_coverage", self._dimension_coverage(item, query_analysis or {}))
        return reranked

    # ---------- 方法一：query 向量 - D 向量 ----------
    def _dim_recall_by_vec(
        self,
        query_vec: list,
        query_text: str,
        top_k: int,
        query_analysis: Optional[dict] = None,
    ) -> tuple[list, list]:
        """
        方法一：query 向量 - D 向量。
        D_cand = TopK_D( sim(v_q, v_D) )
        返回 (候选列表, 空 tag_hits列表)
        """
        analysis = query_analysis or self._analyze_query(query_text)
        allowed_spots = set(analysis.get("spot_names") or [])
        hard_filter = bool(allowed_spots and self._payload_spot_by_chunk)
        hits = self.client.search(
            collection=self.collection_name,
            query_vector=query_vec,
            vector_name="chunk_text_vec",
            top_k=top_k * SearchConfig.SPOT_FILTER_POOL_MULTIPLIER if hard_filter else top_k,
        )
        candidates = []
        for i, hit in enumerate(hits):
            payload = hit.get("payload") or {}
            cid = payload.get("chunk_id") or hit.get("id")
            spot_name = self._payload_spot_by_chunk.get(str(cid), "") or self._canonical_spot(
                self._spot_from_payload(payload)
            )
            if hard_filter and spot_name not in allowed_spots:
                continue
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
                "matched_dimensions": [],
                "match_sources": [],
                "match_source": "vector",
                "spot_name": spot_name,
                "entity_score": self._entity_score(payload, analysis),
                "evidence": [f"vec_score={hit.get('score', 0):.4f}"],
            })
        return candidates, []

    # ---------- 方法二：query 维度 - D 维度 ----------
    def _dim_recall_by_constraint(
        self,
        query_text: str,
        top_k: int,
        query_analysis: Optional[dict] = None,
    ) -> tuple[list, list]:
        """
        方法二：query 维度 - D 维度。
        解析 query 维度约束，筛选文档。
        返回 (候选列表, 空 tag_hits列表)
        """
        analysis = query_analysis or self._analyze_query(query_text)
        constraints = analysis.get("dimension_values") or {}
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

        allowed_spots = set(analysis.get("spot_names") or [])
        if allowed_spots and self._payload_spot_by_chunk:
            all_candidates = {
                key: value for key, value in all_candidates.items()
                if self._payload_spot_by_chunk.get(key, "") in allowed_spots
            }

        if not all_candidates:
            return [], []

        payloads = self.client.retrieve(self.collection_name, list(all_candidates.keys()))
        candidates = self._build_candidates(payloads, all_candidates, "constraint")
        return candidates, tag_hits_out

    # ---------- 方法三：query 向量 - 维度标签向量 ----------
    def _dim_recall_by_tag(
        self,
        query_vec: list,
        query_text: str,
        top_k: int,
        query_analysis: Optional[dict] = None,
    ) -> tuple[list, list]:
        """
        方法三：query 向量 - 维度标签向量。
        L_q = TopK_{(m,t)}( sim(v_q, v_{m,t}) )
        D_cand = {D | T_D ∩ L_q != empty}
        返回 (候选列表, 原始 tag_hits)
        """
        analysis = query_analysis or self._analyze_query(query_text)
        # 新 payload 通路包含景区、维度和值三类信息，优先使用它实现严格的
        # 景区硬过滤和同维度阈值/margin；否则旧 collection 的全局标签召回会
        # 把不同景区的同名标签混在一起。
        if self._payload_tag_points:
            payload_candidates, payload_tag_hits = self._dim_recall_by_payload(
                query_vec, query_text, top_k, analysis
            )
            if payload_candidates or analysis.get("active_dimensions"):
                return payload_candidates, payload_tag_hits

        tag_hits = []
        if self._dim_tags_collection_available:
            tag_hits = self.client.search(
                collection=self.dim_tags_collection,
                query_vector=query_vec,
                vector_name="chunk_text_vec",
                top_k=top_k * 2,
            )
        if not tag_hits:
            # 新版入库将标签写在 unified_corpus 的 dim_* payload 中。
            # 当旧的 dimension_tags collection 不存在或为空时，直接走该通路。
            payload_candidates, payload_tag_hits = self._dim_recall_by_payload(
                query_vec, query_text, top_k, analysis
            )
            if payload_candidates:
                return payload_candidates, payload_tag_hits
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
        if self._dim_tags_collection_available:
            try:
                data = self._post_json(f"/collections/{self.dim_tags_collection}/points/scroll", body)
                points = data.get("result", {}).get("points", [])
                if points:
                    return points
            except Exception:
                pass

        # dimension_tags 不存在时，从 Qdrant 语料 point 的动态维度列恢复命中。
        chunk_ids = sorted(self._payload_tag_points.get((dim_name, tag_name), set()))
        if not chunk_ids:
            return []
        return [{
            "id": f"payload:{dim_name}:{tag_name}",
            "score": 1.0,
            "payload": {
                "dim_name": dim_name,
                "tag_name": tag_name,
                "chunk_ids": chunk_ids,
            },
        }]

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
                    "matched_dimensions": set(),
                    "match_sources": set(),
                }
            candidates[cid_norm]["tag_hits"].append({
                "tag": payload.get("tag_name", ""),
                "dim": payload.get("dim_name", ""),
                "score": tag_score,
                "raw_score": payload.get("raw_score", tag_score),
                "match_source": payload.get("match_source", "vector"),
            })
            if payload.get("dim_name"):
                candidates[cid_norm]["matched_dimensions"].add(payload.get("dim_name"))
            if payload.get("match_source"):
                candidates[cid_norm]["match_sources"].add(payload.get("match_source"))
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
                "matched_dimensions": sorted(meta.get("matched_dimensions", set())),
                "match_sources": sorted(meta.get("match_sources", set())),
                "match_source": (
                    "exact" if "exact" in meta.get("match_sources", set()) else
                    "alias" if "alias" in meta.get("match_sources", set()) else
                    "vector" if "vector" in meta.get("match_sources", set()) else ""
                ),
                "spot_name": self._payload_spot_by_chunk.get(str(cid), ""),
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

                # 精确/别名命中是离散证据，不能被标签向量均值稀释。
                if th.get("match_source") == "exact":
                    tag_sim_sum += 1.0
                    tag_count += 1
                    continue
                if th.get("match_source") == "alias":
                    tag_sim_sum += 0.96
                    tag_count += 1
                    continue

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
                if not self._dim_tags_collection_available:
                    raise RuntimeError("dimension_tags collection unavailable")
                data = self._post_json(f"/collections/{self.dim_tags_collection}/points/scroll", body)
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

            # 新版 Qdrant 直接保存 dim_* payload，独立标签 collection 可能不存在。
            for dim_name, tags in self._payload_dim_tags.items():
                self._cached_tags.setdefault(dim_name, set()).update(tags)

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
