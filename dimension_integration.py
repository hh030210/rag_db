"""
dimension_integration.py

RAG_DB 维度 Pipeline 集成脚本（本地化版本，无外部 API 依赖）
================================================================

功能：整合 llm_service + TF-IDF 聚类 + RDB 写入，实现：
1. 从 MySQL MainIndex.doc_text 加载文档进行 TF-IDF 聚类采样
2. LLM 归纳候选维度 + 迭代优化（覆盖率/辨识度/冗余检测）
3. 将挖掘出的维度添加到 MySQL MainIndex 表
4. 为 MySQL MainIndex 的文档生成维度标签
5. 将标签写入 MySQL MainIndex 的维度列
6. 查询时使用维度约束

使用方式：
    python dimension_integration.py --step 1        # 维度挖掘
    python dimension_integration.py --step 2        # 添加维度列到 MySQL
    python dimension_integration.py --step 3        # 生成文档标签
    python dimension_integration.py --step 4        # 写入标签到 MySQL
    python dimension_integration.py --step 5 -q "儿童发烧"   # 查询解析
    python dimension_integration.py --all           # 完整流程
    python dimension_integration.py --step 3 --doc_ids doc001 doc002  # 局部处理
    python dimension_integration.py --step 3 --force  # 强制重跑（跳过断点）
    python dimension_integration.py --step 1 --validate_all  # 全量复核 schema（不写标签）
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from tqdm import tqdm

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from collections import Counter

# 添加项目根目录和 code 目录到 path（code 是普通目录，不是包）
project_root = Path(__file__).parent
CODE_DIR = project_root / "code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(project_root))

from db_config import get_config

# 读取配置中的 DashScope API Key。通过 db_config.py 统一展开
# `${DASHSCOPE_API_KEY:-}`，避免把占位符原样当成真实 key。
_dashscope_key = ""
try:
    _dashscope_key = str(getattr(get_config().qgen, "api_key", "") or "")
    if _dashscope_key:
        os.environ["DASHSCOPE_API_KEY"] = _dashscope_key
        print(f"[配置] DashScope API Key 已加载 (前6位: {_dashscope_key[:6]}...)")
except Exception as e:
    print(f"[配置] 读取 qgen 配置失败: {e}")
    _dashscope_key = ""

if not _dashscope_key:
    _dashscope_key = os.getenv("DASHSCOPE_API_KEY", "")
    if _dashscope_key:
        print(f"[配置] DashScope API Key 从环境变量 DASHSCOPE_API_KEY 读取")

# === LLM 配置提示 ===
_llm_mode = "OpenAI 兼容" if os.getenv("LLM_OPENAI_COMPAT", "").strip() in ("1", "true", "TRUE", "yes", "YES") else "DashScope"
_llm_api = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
_llm_url = os.getenv("LLM_BASE_URL", "")
_llm_model = os.getenv("LLM_MODEL", "")
if _llm_api:
    _preview_key = _llm_api[:6]
    _url_short = _llm_url if _llm_url else "(默认)"
    _model_short = _llm_model if _llm_model else "(默认)"
    print(f"[配置] LLM 模式={_llm_mode} | Key前6位={_preview_key}... | URL={_url_short} | Model={_model_short}")

from llm_service import DimensionMiningWithQwen

# ===================== 配置 =====================

DATA_DIR = project_root / "experiment_data"
DATA_DIR.mkdir(exist_ok=True)

# 输出路径
PATH_V_CAND = DATA_DIR / "V_cand.json"
PATH_V_CORE = DATA_DIR / "V_core.json"
PATH_TAGS = DATA_DIR / "tags_output.json"
PATH_DIM_META = DATA_DIR / "dimension_metadata.json"
PATH_DIM_DIAGNOSTICS = DATA_DIR / "dimension_diagnostics.json"

# 维度挖掘参数
K_CLUSTERS = 50
N_CORE_SAMPLES = 5
N_BOUND_SAMPLES = 5
TH_COV = 0.20   # 覆盖率阈值（降低到 20%，让融合逻辑有机会合并维度）
TH_DIS = 0.30   # 归一化熵阈值；不再使用受取值数量影响的原始熵
TH_DIFF = 0.3
MAX_ITER = 2
SAMPLE_LIMIT = 10000  # 聚类采样时最多加载的文档数（用于 Milvus 向量聚类）


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    """读取整数环境变量，非法或过小值回退到默认值。"""
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    """读取布尔环境变量。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# ===================== 工具函数 =====================

def normalize_attr_key(dim_name: str) -> str:
    """将维度名称转换为 MySQL 列名格式：dim_xxx"""
    key = dim_name.strip().lower()
    key = key.replace(" ", "_").replace("-", "_")
    # 移除非法字符
    key = "".join(c for c in key if c.isalnum() or c == "_")
    if not key.startswith("dim_"):
        key = "dim_" + key
    return key[:64]  # MySQL 列名最长 64


def _calc_jaccard(list_a: list, list_b: list) -> float:
    """计算 Jaccard 系数"""
    set_a = {x for x in list_a if x and x != "NULL"}
    set_b = {x for x in list_b if x and x != "NULL"}
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def _normalized_entropy(values: list) -> float:
    """计算标签值分布的归一化熵，结果范围为 [0, 1]。"""
    cleaned = [str(value).strip() for value in values if value and str(value).strip() not in {"NULL", "NONE"}]
    if len(cleaned) <= 1:
        return 0.0
    counts = Counter(cleaned)
    if len(counts) <= 1:
        return 0.0
    raw_entropy = entropy([count / len(cleaned) for count in counts.values()])
    max_entropy = np.log(len(counts))
    return float(raw_entropy / max_entropy) if max_entropy > 0 else 0.0


def _spread_items(items: list, limit: int) -> list:
    """从列表中均匀抽取若干项，避免只使用列表前缀。"""
    if not items:
        return []
    limit = max(1, min(limit, len(items)))
    if limit == len(items):
        return list(items)
    if limit == 1:
        return [items[0]]
    indices = [round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)]
    return [items[i] for i in indices]


def _parse_chunks_to_docs(data) -> List[Dict[str, str]]:
    """把已加载的 JSON 数据解析为统一格式 [{"id": ..., "text": ...}, ...]

    支持：
    1. list[{"id": ..., "text": ...}]
    2. list[{"doc_id": ..., "doc_text": ...}]
    3. list[{"doc_id": ..., "text": ...}]
    4. {"chunks": [...]} / {"documents": [...]} / {"data": [...]} 包装
    5. integrated_chunker 输出：list[{"doc_id": ..., "file_name": ..., "chunks": [{"chunk_id": ..., "chunk_text": ...}, ...]}]
    """
    if isinstance(data, dict):
        for key in ("chunks", "documents", "docs", "data", "items"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise ValueError("无法识别 JSON 结构")

    if not isinstance(data, list):
        raise ValueError(f"chunks 数据不是 list: {type(data).__name__}")

    docs: List[Dict[str, str]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        doc_id = item.get("id") or item.get("doc_id") or item.get("docid")
        doc_text = (
            item.get("text")
            or item.get("doc_text")
            or item.get("content")
            or item.get("chunk_text")
        )

        # integrated_chunker 格式：嵌套 chunks 列表，拼接所有子 chunk
        if doc_text is None and "chunks" in item and isinstance(item["chunks"], list):
            sub_texts = []
            for sc in item["chunks"]:
                if isinstance(sc, dict):
                    t = sc.get("chunk_text") or sc.get("text") or sc.get("content")
                    if t:
                        sub_texts.append(str(t))
            doc_text = "\n\n".join(sub_texts) if sub_texts else None

        if doc_id is None:
            doc_id = f"chunk_{i:06d}"
        if doc_text is None:
            continue
        docs.append({"id": str(doc_id), "text": str(doc_text)})

    return docs


# ===================== Milvus 连接 =====================

def get_milvus_connection():
    """连接到 RAG_DB 的 Milvus"""
    config = get_config()
    try:
        from pymilvus import connections
        connections.connect(
            "default",
            host=config.vecdb.host,
            port=str(config.vecdb.port)
        )
        print(f"已连接到 Milvus: {config.vecdb.host}:{config.vecdb.port}")
    except Exception as e:
        print(f"[错误] Milvus 连接失败: {e}")
        raise


def get_collection(collection_name: str = None):
    """获取 Milvus Collection"""
    from pymilvus import Collection, utility
    config = get_config()
    name = collection_name or config.vecdb.collection_name

    if not utility.has_collection(name):
        raise RuntimeError(f"Collection 不存在: {name}")

    collection = Collection(name)
    collection.load()
    return collection


def load_docs_from_milvus(collection, limit: int = 10000, fields: List[str] = None):
    """
    使用迭代器从 Milvus 加载文档
    返回: [{"id": ..., "text": ..., "doc_id": ...}, ...]
    """
    if fields is None:
        fields = ["chunk_id", "doc_id_link", "chunk_text"]

    print(f"正在加载文档 (limit={limit})...")
    docs_map = {}  # doc_id -> {"id": ..., "text": ...}

    try:
        iterator = collection.query_iterator(
            collection_name=collection.name,
            filter="",
            output_fields=fields,
            batch_size=100
        )

        with tqdm(desc="加载文档") as pbar:
            while True:
                batch = iterator.next()
                if not batch:
                    break

                for item in batch:
                    doc_id = item.get("doc_id_link", "")
                    chunk_text = item.get("chunk_text", "")

                    # 聚合：一个 doc 的多个 chunk 拼接 text
                    if doc_id not in docs_map:
                        docs_map[doc_id] = {
                            "id": doc_id,
                            "text": chunk_text,
                        }
                    else:
                        # 追加到现有文本
                        docs_map[doc_id]["text"] += "\n" + chunk_text

                    pbar.update(1)

                    if len(docs_map) >= limit:
                        break

                if len(docs_map) >= limit:
                    break

    except Exception as e:
        print(f"[错误] 加载文档失败: {e}")
        raise

    result = list(docs_map.values())[:limit]
    print(f"加载完成，共 {len(result)} 篇文档")
    return result


def load_vectors_only_from_rdb(limit: int = 10000):
    """从 MySQL 读取文档并做 TF-IDF 向量化（用于聚类）"""
    print(f"正在从 MySQL 加载文档进行 TF-IDF 向量化 (limit={limit})...")

    docs = load_docs_from_rdb(doc_ids=None)
    if not docs:
        return [], np.array([]), None

    texts = [d.get("text", "") for d in docs]
    ids = [d["id"] for d in docs]

    if len(texts) > limit:
        texts = texts[:limit]
        ids = ids[:limit]

    try:
        vectorizer = TfidfVectorizer(max_features=1024, max_df=0.8, min_df=2)
        vectors = vectorizer.fit_transform(texts)
        vectors_np = vectors.toarray().astype(np.float32)
        print(f"TF-IDF 向量化完成: {vectors_np.shape}")
        return ids, vectors_np, vectorizer
    except Exception as e:
        print(f"[错误] TF-IDF 向量化失败: {e}")
        return ids, np.zeros((len(texts), 512), dtype=np.float32), None


# ===================== RDB 操作 =====================

def connect_rdb():
    """连接到 RAG_DB 的 MySQL"""
    import mysql.connector
    config = get_config()
    conn = mysql.connector.connect(
        host=config.rdb.host,
        port=config.rdb.port,
        user=config.rdb.user,
        password=config.rdb.password,
        database=config.rdb.database,
        charset="utf8mb4",
        collation="utf8mb4_unicode_ci",
    )
    return conn


def ensure_rdb_db():
    """确保数据库存在"""
    import mysql.connector
    config = get_config()
    conn = mysql.connector.connect(
        host=config.rdb.host,
        port=config.rdb.port,
        user=config.rdb.user,
        password=config.rdb.password,
        charset="utf8mb4",
    )
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{config.rdb.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.commit()
    cur.close()
    conn.close()
    print(f"数据库 {config.rdb.database} 就绪")


def fetch_rdb_columns(table: str) -> List[str]:
    """获取 RDB 表的列名列表"""
    config = get_config()
    conn = connect_rdb()
    cur = conn.cursor()
    cur.execute(
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s",
        (config.rdb.database, table)
    )
    cols = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return cols


def load_docs_from_rdb(doc_ids: List[str] = None) -> List[Dict[str, str]]:
    """
    从 MySQL MainIndex 表读取文档文本（全量）。

    Args:
        doc_ids: 若指定，则只读取这些 doc_id；None = 全量

    Returns:
        [{"id": doc_id, "text": doc_text}, ...]
    """
    config = get_config()
    table = config.rdb.table
    doc_id_col = config.rdb.doc_id_column

    print(f"正在从 MySQL 读取文档 (table={table})...")

    conn = connect_rdb()
    cur = conn.cursor()

    if doc_ids:
        placeholders = ", ".join(["%s"] * len(doc_ids))
        sql = f"SELECT `{doc_id_col}`, `doc_text` FROM `{table}` WHERE `{doc_id_col}` IN ({placeholders})"
        cur.execute(sql, doc_ids)
    else:
        sql = f"SELECT `{doc_id_col}`, `doc_text` FROM `{table}`"
        cur.execute(sql)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    docs = []
    for row in rows:
        row_doc_id = str(row[0])
        row_doc_text = row[1] or ""
        if row_doc_text and str(row_doc_text).strip():
            docs.append({"id": row_doc_id, "text": str(row_doc_text)})

    print(f"  -> 读取完成，共 {len(docs)} 条有效文档")
    return docs


# ===================== Step 1: 维度挖掘 =====================

class DimensionMiner:
    """维度挖掘系统"""

    def __init__(self, docs_source: str = "rdb"):
        """
        Args:
            docs_source: 数据来源
              - "rdb" : 从 MySQL 读取
              - 文件路径（.json）: 直接加载 chunks 文件
              - 目录路径: 加载目录下所有 *.json
        """
        self.miner = DimensionMiningWithQwen()
        self.vectorizer = None
        self.docs_source = docs_source
        self._in_memory_docs: List[Dict[str, str]] = []
        self.dimension_diagnostics: Dict[str, Dict[str, Any]] = {}

    def _load_docs_from_source(self) -> List[Dict[str, str]]:
        """统一加载文档"""
        src = self.docs_source
        if isinstance(src, str) and src == "rdb":
            return load_docs_from_rdb(doc_ids=None)

        path = Path(src)
        if not path.exists():
            raise FileNotFoundError(f"docs_source 路径不存在: {path}")

        if path.is_dir():
            files = sorted(path.glob("*.json"))
            if not files:
                raise FileNotFoundError(f"目录下未找到 .json: {path}")
            merged: List[Dict[str, str]] = []
            for jf in files:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                merged.extend(_parse_chunks_to_docs(data))
            print(f"  [docs_source] 从目录 {path}/ 加载 {len(merged)} 条文档")
            return merged

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = _parse_chunks_to_docs(data)
        print(f"  [docs_source] 从 {path.name} 加载 {len(docs)} 条文档")
        return docs

    def step1_clustering_sampling(self):
        """聚类采样获取代表性文档（支持 MySQL 或 chunks 文件）"""
        print("\n>>> [Step 1] 语义聚类采样（来源: {}）...".format(
            "MySQL" if self.docs_source == "rdb" else str(self.docs_source)
        ))

        if self.docs_source != "rdb":
            # 直接从 chunks 文件加载，跳过 MySQL
            docs = self._load_docs_from_source()
            if not docs:
                print("[Warning] 未加载到文档，返回空列表")
                self.sampled_texts = []
                return self.sampled_texts

            texts = [d.get("text", "") for d in docs]
            ids = [d["id"] for d in docs]

            # TF-IDF 向量化（如果文本足够多）
            if len(texts) >= 5:
                try:
                    self.vectorizer = TfidfVectorizer(max_features=1024, max_df=0.8, min_df=2)
                    vectors = self.vectorizer.fit_transform(texts).toarray().astype(np.float32)
                except Exception:
                    self.vectorizer = None
                    vectors = np.zeros((len(texts), 512), dtype=np.float32)

                k = min(K_CLUSTERS, max(1, len(vectors) - 1))
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(vectors)

                selected_indices = set()
                for i in range(k):
                    cluster_indices = np.where(labels == i)[0]
                    if len(cluster_indices) == 0:
                        continue
                    cluster_vectors = vectors[cluster_indices]
                    center = kmeans.cluster_centers_[i].reshape(1, -1)
                    dists = pairwise_distances_argmin_min(cluster_vectors, center, metric='euclidean')[1]
                    core_local = np.argsort(dists)[:N_CORE_SAMPLES]
                    bound_local = np.argsort(dists)[-N_BOUND_SAMPLES:]
                    for idx in np.concatenate([core_local, bound_local]):
                        if idx < len(cluster_indices):
                            selected_indices.add(int(cluster_indices[idx]))

                sampled_ids = [ids[i] for i in selected_indices if i < len(texts)]
                id_to_doc = {d["id"]: d for d in docs}
                self.sampled_texts = [
                    id_to_doc.get(doc_id, {}).get("text", "")
                    for doc_id in sampled_ids
                ]
                result = [t for t in self.sampled_texts if t]
                print(f"  采样文本数量: {len(result)}（原始 {len(self.sampled_texts)}）")
                return result
            else:
                self.sampled_texts = texts
                print(f"  数据量过少，跳过聚类，返回全部 {len(texts)} 条")
                return texts

        # 原 MySQL 逻辑
        all_ids, vectors, self.vectorizer = load_vectors_only_from_rdb(limit=SAMPLE_LIMIT)

        if len(vectors) < K_CLUSTERS:
            print("数据量过少，直接返回全部文档")
            docs = load_docs_from_rdb(doc_ids=None)
            self.sampled_texts = [d.get("text", "") for d in docs]
            return self.sampled_texts

        # KMeans 聚类
        print(f"正在聚类 {len(vectors)} 条向量...")
        kmeans = KMeans(n_clusters=min(K_CLUSTERS, len(vectors) - 1), random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)

        selected_indices = set()
        for i in range(min(K_CLUSTERS, len(vectors))):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_vectors = vectors[cluster_indices]
            center = kmeans.cluster_centers_[i].reshape(1, -1)
            dists = pairwise_distances_argmin_min(cluster_vectors, center, metric='euclidean')[1]

            core_local = np.argsort(dists)[:N_CORE_SAMPLES]
            bound_local = np.argsort(dists)[-N_BOUND_SAMPLES:]

            for idx in np.concatenate([core_local, bound_local]):
                selected_indices.add(cluster_indices[idx])

        sampled_ids = [all_ids[idx] for idx in selected_indices]
        print(f"采样完成，锁定 {len(sampled_ids)} 个文档")

        # 从 MySQL 加载采样文档
        docs = load_docs_from_rdb(doc_ids=sampled_ids)
        id_to_doc = {d["id"]: d for d in docs}

        self.sampled_texts = [id_to_doc.get(doc_id, {}).get("text", "") for doc_id in sampled_ids]
        result = [t for t in self.sampled_texts if t]
        print(f"  采样文本数量: {len(result)}（原始 {len(self.sampled_texts)}）")
        return result

    def step2_generate_candidates(self, sampled_docs):
        """LLM 归纳候选维度"""
        if PATH_V_CAND.exists() and not _env_flag("DIM_FORCE_REDISCOVER"):
            print("检测到已有候选维度，跳过生成")
            with open(PATH_V_CAND, "r", encoding="utf-8") as f:
                return json.load(f)

        print("\n>>> [Step 2] LLM 归纳候选维度...")
        print(f"  [DEBUG] sampled_docs 数量={len(sampled_docs)}, 首条长度={len(sampled_docs[0]) if sampled_docs else 0}")
        try:
            V_cand = self.miner.generate_candidate_dimensions(sampled_docs)
            print(f"  [DEBUG] LLM 调用成功，返回 {len(V_cand)} 个维度")
        except Exception as e:
            print(f"  [ERROR] LLM 调用失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        # 仅做可控上限，不再固定截断到 15 个；维度数量由后续全量验证
        # 和质量诊断共同决定。LLM 输出已经在 llm_service 中去重和清理。
        MAX_CANDIDATES = _env_int("MAX_DIM_CANDIDATES", 30, minimum=1)
        if len(V_cand) > MAX_CANDIDATES:
            print(f"  LLM 返回 {len(V_cand)} 个候选维度，截断到前 {MAX_CANDIDATES} 个")
            V_cand = V_cand[:MAX_CANDIDATES]
        print(f"  LLM 返回候选维度 {len(V_cand)} 个: {V_cand[:5]}...")

        with open(PATH_V_CAND, "w", encoding="utf-8") as f:
            json.dump(V_cand, f, ensure_ascii=False, indent=2)
        print(f"生成候选维度 {len(V_cand)} 个: {V_cand}")
        return V_cand

    def step3_iterative_optimization(self, initial_dims):
        """迭代优化（使用与聚类相同的文档集和向量器）"""
        if PATH_V_CORE.exists() and not _env_flag("DIM_FORCE_REDISCOVER"):
            print("检测到已有核心维度，跳过优化")
            with open(PATH_V_CORE, "r", encoding="utf-8") as f:
                return json.load(f)

        print("\n>>> [Step 3] 迭代优化...")
        current_dims = list(initial_dims)

        # 复用 step1 的 vectorizer 和文档（self 在 run_step1 中被赋值）
        if not hasattr(self, 'vectorizer') or self.vectorizer is None:
            try:
                self.vectorizer = TfidfVectorizer(max_features=1024, max_df=0.8, min_df=2)
            except:
                pass
        if not hasattr(self, 'sampled_texts') or not self.sampled_texts:
            # fallback: 重新从 MySQL 加载少量文档（快速模式用 100 条）
            docs = load_docs_from_rdb(doc_ids=None)
            texts = [d["text"] for d in docs[:100] if d.get("text")]
        else:
            texts = self.sampled_texts

        self.validation_texts = texts  # 保存给 Step 3 使用
        verified_dims: Set[str] = set()
        validation_max_chars = _env_int("DIM_VALIDATION_CHARS", 4000, minimum=0)

        for iteration in range(MAX_ITER):
            print(f"\n--- Iteration {iteration + 1} / {MAX_ITER} ---")

            active = [d for d in current_dims if d not in verified_dims]
            if not active:
                break

            # 抽取
            print(f"  开始抽取（{len(active)} 个维度 x {len(texts)} 条文本）...")
            extraction_results = {dim: [] for dim in active}
            extraction_doc_hits = {dim: 0 for dim in active}
            validation_batch_size = _env_int("DIM_SCHEMA_BATCH_SIZE", 8, minimum=1)
            for start in tqdm(
                range(0, len(texts), validation_batch_size),
                desc="Schema 验证进度",
            ):
                batch_texts = texts[start:start + validation_batch_size]
                records = [
                    {"doc_id": f"sample_{start + offset}", "doc_text": text}
                    for offset, text in enumerate(batch_texts)
                ]
                batch_result = self.miner.validate_dimension_schema_batch(
                    records, active, max_text_chars=validation_max_chars
                )
                for record in records:
                    dim_values = batch_result.get(record["doc_id"], {})
                    for dim in active:
                        values = dim_values.get(dim, [])
                        if values:
                            # 覆盖率按“命中的 chunk 数”计算；辨识度按全部值统计。
                            extraction_doc_hits[dim] += 1
                            extraction_results[dim].extend(values)
            print(
                "  抽取完成，各维度命中 chunk 数: "
                f"{[(d, extraction_doc_hits[d]) for d in active[:5]]}..."
            )

            # 诊断
            dims_to_remove = set()
            new_dims_added = []
            candidates = []
            failing_dims = []  # (dim, cov, dis, issue_type, msg, vals)

            # Part A: 覆盖率 & 辨识度诊断（先全部收集，不调用 LLM）
            for dim in active:
                vals = extraction_results[dim]
                covered_chunks = extraction_doc_hits[dim]
                cov = covered_chunks / len(texts) if texts else 0
                dis = _normalized_entropy(vals)

                self.dimension_diagnostics[dim] = {
                    "covered_chunks": covered_chunks,
                    "total_chunks": len(texts),
                    "coverage": round(cov, 4),
                    "value_count": len(vals),
                    "unique_values": len(set(vals)),
                    "normalized_entropy": round(dis, 4),
                    "validation_chars": validation_max_chars,
                }

                issue_type = None
                msg = ""
                if cov < TH_COV:
                    issue_type = "低覆盖率"
                    msg = f"覆盖率仅 {cov:.2%}"
                elif dis < TH_DIS:
                    issue_type = "低辨识度"
                    msg = f"信息熵 {dis:.2f}"

                if not issue_type:
                    candidates.append(dim)
                else:
                    failing_dims.append((dim, cov, dis, issue_type, msg, vals))

            # Part A2: merge 语义预筛选——候选目标超过 10 个时，
            # 用 TF-IDF 余弦相似度快速找到 top-3 最相关的目标传给 LLM，
            # 避免 98 候选导致 merge 调用量爆炸（98*100=9800次/轮）
            MERGE_TOP_N = 3
            approved = [d for d in candidates if d not in dims_to_remove]
            prefilter_map = {}  # failing_dim -> [top3_targets]
            if len(approved) > 10 and self.vectorizer is not None:
                try:
                    cov_map = {
                        d: extraction_doc_hits.get(d, 0) / len(texts)
                        for d in approved
                    }
                    pool = sorted(approved, key=lambda d: cov_map[d], reverse=True)[:20]
                    pool_vecs = self.vectorizer.transform(pool).toarray().astype(np.float32)
                    for dim, _, _, _, _, _ in failing_dims:
                        dim_vec = self.vectorizer.transform([dim]).toarray().astype(np.float32)
                        sims = cosine_similarity(dim_vec, pool_vecs)[0]
                        top_idx = np.argsort(sims)[::-1][:MERGE_TOP_N]
                        prefilter_map[dim] = [pool[i] for i in top_idx]
                    print(f"  [预筛选] merge 候选从 {len(approved)} 降至 top-{MERGE_TOP_N}（池 {len(pool)} 个）")
                except Exception as e:
                    print(f"  [预筛选] merge 跳过: {e}")
                    prefilter_map = {}

            # Part A3: 对不合格维度调用 LLM 决策（传入预筛选后的 merge 目标）
            print(f"  诊断: 合格 {len(candidates)} 个, 不合格 {len(failing_dims)} 个")
            for dim, cov, dis, issue_type, msg, vals in failing_dims:
                merge_targets = prefilter_map.get(dim, approved)  # 有预筛选用预筛选，否则用全部 approved
                decision = self.miner.merge_with_targets(
                    dim, issue_type, msg, vals[:5], merge_targets
                )
                self._handle_decision(dim, decision, dims_to_remove, new_dims_added, verified_dims)

            # Part B: 冗余检测
            check = [d for d in candidates if d not in dims_to_remove]
            if len(check) > 1 and self.vectorizer is not None:
                # 语义预筛选：维度太多时只取与任意维度相似度较高的组成候选对
                PRE_FILTER_ENABLED = True
                if PRE_FILTER_ENABLED and len(check) > 10:
                    try:
                        vectors = self.vectorizer.transform(check).toarray().astype(np.float32)
                        sim_mat = cosine_similarity(vectors)
                        # 找出每行 top-3 最相似的列（排除自身）
                        keep_indices = set()
                        for i in range(len(check)):
                            top_j = np.argsort(sim_mat[i])[::-1][1:4]  # 排除自身
                            keep_indices.update([i] + list(top_j))
                        check = [check[i] for i in sorted(keep_indices)]
                        print(f"  [预筛选] 冗余检测候选从 {len(candidates)} 降至 {len(check)}")
                    except Exception as e:
                        print(f"  [预筛选] 跳过: {e}")

                if len(check) > 1:
                    print(f"正在进行差异性评估 ({len(check)} 个候选)...")
                    try:
                        dim_vectors = self.vectorizer.transform(check).toarray().astype(np.float32)
                        sim_mat = cosine_similarity(dim_vectors)

                        skip = set()
                        for i in range(len(check)):
                            if i in skip:
                                continue
                            for j in range(i + 1, len(check)):
                                if j in skip:
                                    continue

                                sim_def = sim_mat[i][j]
                                vals_a, vals_b = extraction_results.get(check[i], []), extraction_results.get(check[j], [])
                                sim_data = _calc_jaccard(vals_a, vals_b)
                                redundancy = min(sim_def, sim_data)
                                r_diff = 1.0 - redundancy

                                if r_diff < TH_DIFF:
                                    print(f"  [冗余] {check[i]} vs {check[j]} ({redundancy:.2f})")
                                    decision = self.miner.optimize_dimension(
                                        check[i], "语义/数据冗余",
                                        f"与 {check[j]} 冗余度 {redundancy:.2f}",
                                        (vals_a[:3] + vals_b[:3])
                                    )
                                    self._handle_decision(check[i], decision, dims_to_remove, new_dims_added, verified_dims)
                                    if decision and decision.get("action", "").upper() == "MERGE":
                                        dims_to_remove.add(check[j])
                                        skip.add(j)

                    except Exception as e:
                        print(f"  差异性评估跳过: {e}")

            # 更新
            for dim in check:
                if dim not in dims_to_remove:
                    verified_dims.add(dim)

            # --- 新维度质量验证（防止 LLM merge 出"其他"等无意义维度）---
            validated_new_dims = []
            for nd in new_dims_added:
                nd = nd.strip()
                if not nd or nd in current_dims or nd in verified_dims:
                    continue

                sample_texts = _spread_items(texts, 20)
                nd_vals = []
                nd_doc_hits = 0
                for txt in sample_texts:
                    values = self.miner.extract_dimension_values(
                        txt, nd, max_chars=validation_max_chars
                    )
                    if values:
                        nd_doc_hits += 1
                        nd_vals.extend(values)

                nd_cov = nd_doc_hits / len(sample_texts) if sample_texts else 0
                nd_dis = _normalized_entropy(nd_vals)
                self.dimension_diagnostics[nd] = {
                    "covered_chunks": nd_doc_hits,
                    "total_chunks": len(sample_texts),
                    "coverage": round(nd_cov, 4),
                    "value_count": len(nd_vals),
                    "unique_values": len(set(nd_vals)),
                    "normalized_entropy": round(nd_dis, 4),
                    "validation_chars": validation_max_chars,
                    "is_new_dimension": True,
                }

                if nd_cov >= TH_COV and nd_dis >= TH_DIS:
                    validated_new_dims.append(nd)
                    print(f"  [新维度验证通过] {nd} (cov={nd_cov:.2%}, dis={nd_dis:.2f})")
                else:
                    print(f"  [新维度验证淘汰] {nd} (cov={nd_cov:.2%}, dis={nd_dis:.2f}) — 已被丢弃")

            # 仅将通过验证的新维度加入 current_dims
            current_dims = [d for d in current_dims if d not in dims_to_remove]
            for nd in validated_new_dims:
                current_dims.append(nd)

            print(f"本轮结束 -> 移除: {len(dims_to_remove)}, 新增(验证后): {len(validated_new_dims)}")
            if len(dims_to_remove) == 0 and len(validated_new_dims) == 0:
                print("系统收敛")
                break

        with open(PATH_V_CORE, "w", encoding="utf-8") as f:
            json.dump(current_dims, f, ensure_ascii=False, indent=2)

        diagnostics = {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "docs_source": str(self.docs_source),
            "sampled_chunks": len(texts),
            "thresholds": {
                "coverage": TH_COV,
                "normalized_entropy": TH_DIS,
                "difference": TH_DIFF,
            },
            "dimensions": self.dimension_diagnostics,
            "core_dimensions": current_dims,
            "note": (
                "本文件只记录候选维度名称的发现/验证诊断，不是 chunk 标签结果；"
                "具体维度值的打标签由后续独立流程完成。"
            ),
        }
        with open(PATH_DIM_DIAGNOSTICS, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, ensure_ascii=False, indent=2)
        print(f"维度发现诊断已保存至: {PATH_DIM_DIAGNOSTICS}")

        return current_dims

    def validate_schema_on_all_chunks(
        self,
        dimensions: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """在全量 chunk 上复核维度 schema，但不生成、不写入标签。

        这一步只回答“候选维度在全量 chunk 中是否有明确证据、取值是否具有区分度”，
        用于验证维度名称是否值得进入后续打标签阶段。它不会修改 Qdrant/MySQL，
        也不会产出 `dim_*` 字段。
        """
        docs = self._load_docs_from_source()
        dimensions = [str(dim).strip() for dim in (dimensions or []) if str(dim).strip()]
        if not dimensions:
            return {}

        texts = [str(doc.get("text", "")) for doc in docs if str(doc.get("text", "")).strip()]
        if not texts:
            return {}

        max_chars = _env_int("DIM_VALIDATION_CHARS", 4000, minimum=0)
        # 维度发现的全量复核默认抽取一小批维度，避免把它误当成正式标签生成。
        batch_size = _env_int("DIM_SCHEMA_VALIDATE_BATCH_SIZE", 8, minimum=1)
        covered = {dim: 0 for dim in dimensions}
        values_by_dim = {dim: [] for dim in dimensions}
        failed_batches = 0

        print(
            f"\n>>> [Schema Validation] 全量复核 {len(texts)} 个 chunk、"
            f"{len(dimensions)} 个候选维度（不写标签）..."
        )
        for start in tqdm(range(0, len(texts), batch_size), desc="全量 schema 复核"):
            records = [
                {"doc_id": str(start + offset), "doc_text": text}
                for offset, text in enumerate(texts[start:start + batch_size])
            ]
            try:
                # 使用 Schema 专用探针，仅作为“是否存在维度证据”的判断，结果不落库。
                result = self.miner.validate_dimension_schema_batch(
                    records, dimensions, max_text_chars=max_chars
                )
            except Exception as exc:
                failed_batches += 1
                print(f"  [Warning] schema 复核批次失败 start={start}: {exc}")
                continue

            for record in records:
                tag_map = result.get(record["doc_id"], {}) if isinstance(result, dict) else {}
                for dim in dimensions:
                    raw_values = tag_map.get(dim, []) if isinstance(tag_map, dict) else []
                    if isinstance(raw_values, str):
                        raw_values = [raw_values]
                    values = [str(value).strip() for value in (raw_values or []) if str(value).strip()]
                    if values:
                        covered[dim] += 1
                        values_by_dim[dim].extend(values)

        diagnostics = {}
        for dim in dimensions:
            values = values_by_dim[dim]
            diagnostics[dim] = {
                "covered_chunks": covered[dim],
                "total_chunks": len(texts),
                "coverage": round(covered[dim] / len(texts), 4),
                "value_count": len(values),
                "unique_values": len(set(values)),
                "normalized_entropy": round(_normalized_entropy(values), 4),
                "validation_chars": max_chars,
                "validation_scope": "all_chunks",
            }

        self.dimension_diagnostics.update(diagnostics)
        output = {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "docs_source": str(self.docs_source),
            "total_chunks": len(texts),
            "failed_batches": failed_batches,
            "thresholds": {
                "coverage": TH_COV,
                "normalized_entropy": TH_DIS,
                "difference": TH_DIFF,
            },
            "dimensions": diagnostics,
            "core_dimensions": dimensions,
            "note": (
                "全量 schema 复核只用于判断维度名称是否有证据和区分度；"
                "不生成、不写入任何 chunk 标签。"
            ),
        }
        with open(PATH_DIM_DIAGNOSTICS, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"全量 schema 复核结果已保存至: {PATH_DIM_DIAGNOSTICS}")
        return diagnostics

    def _handle_decision(self, dim, decision, dims_to_remove, new_dims_added, verified_dims,
                          candidates_for_merge=None):
        """
        Args:
            candidates_for_merge: 当前轮次通过 Part A 的合格维度候选（用于语义预筛选 merge 目标）
        """
        if not decision:
            return
        action = decision.get("action", "").upper()
        new_dims = decision.get("new_dimensions", [])
        if not isinstance(new_dims, list):
            new_dims = [str(new_dims)] if new_dims else []

        print(f"  决策: {action} | {dim}")
        if action == "DELETE":
            dims_to_remove.add(dim)
        elif action == "KEEP":
            verified_dims.add(dim)
        elif action == "NOT_MERGE":
            # “没有合适的合并目标”不等于“维度没有价值”。保留该维度，
            # 避免稀疏但有检索价值的维度因合并失败被误删。
            verified_dims.add(dim)
        elif action in ("RENAME", "SPLIT", "MERGE"):
            dims_to_remove.add(dim)
            new_dims_added.extend(new_dims)


def run_step1(docs_source: str = "rdb", validate_all: bool = False):
    """执行 Step 1: 维度挖掘（支持 MySQL 或 chunks 文件）

    Args:
        docs_source: "rdb" 或 chunks JSON 文件/目录路径
        validate_all: 是否在核心维度确定后，对全量 chunk 做 schema 复核；只记录诊断，不写标签
    """
    print("=" * 60)
    src_label = "MySQL" if docs_source == "rdb" else str(docs_source)
    print(f"Step 1: 维度挖掘（数据源: {src_label}）")
    print("=" * 60)

    miner = DimensionMiner(docs_source=docs_source)

    # Step 1: 聚类采样
    sampled_docs = miner.step1_clustering_sampling()

    # Step 2: 生成候选维度
    V_cand = miner.step2_generate_candidates(sampled_docs)

    # Step 3: 迭代优化
    V_core = miner.step3_iterative_optimization(V_cand)

    if validate_all:
        miner.validate_schema_on_all_chunks(V_core)

    print("\n" + "=" * 60)
    print("维度挖掘完成!")
    print(f"候选维度: {len(V_cand)} 个")
    print(f"核心维度: {len(V_core)} 个")
    print(f"维度列表: {V_core}")
    print("=" * 60)

    return V_core


# ===================== Step 2: 添加维度列到 RDB =====================

def run_step2():
    """执行 Step 2: 将维度添加到 RDB"""
    print("=" * 60)
    print("Step 2: 添加维度列到 RDB")
    print("=" * 60)

    if not PATH_V_CORE.exists():
        print("[错误] 请先执行 Step 1 挖掘维度")
        return []

    with open(PATH_V_CORE, "r", encoding="utf-8") as f:
        dims = json.load(f)

    print(f"读取到 {len(dims)} 个维度: {dims}")

    # 确保数据库存在
    ensure_rdb_db()

    # 获取现有列
    config = get_config()
    table = config.rdb.table
    existing_cols = fetch_rdb_columns(table)
    print(f"现有列: {existing_cols}")

    # 添加缺失的维度列
    conn = connect_rdb()
    cur = conn.cursor()

    added = []
    skipped = []

    for dim in dims:
        col_name = normalize_attr_key(dim)
        if col_name in existing_cols:
            skipped.append(col_name)
            continue

        try:
            sql = f"ALTER TABLE `{table}` ADD COLUMN `{col_name}` TEXT NULL"
            cur.execute(sql)
            conn.commit()
            added.append(col_name)
            print(f"  + 添加列: {col_name}")
        except Exception as e:
            print(f"  [错误] 添加 {col_name} 失败: {e}")

    cur.close()
    conn.close()

    print("\n" + "=" * 60)
    print(f"添加完成: 新增 {len(added)} 列, 跳过 {len(skipped)} 列")
    print(f"新增列: {added}")
    print("=" * 60)

    return dims


# ===================== Step 3: 生成文档标签 =====================

class TagGenerator:
    """文档标签生成器（支持 MySQL 或 chunks 文件）"""

    def __init__(self, dims: List[str], docs_source: str = "rdb"):
        self.dims = dims
        self.miner = DimensionMiningWithQwen()
        self.results: Dict[str, Dict[str, List[str]]] = {}
        self.processed_ids: Set[str] = set()
        self.docs_source = docs_source
        self._load_checkpoint()

    def _load_checkpoint(self):
        """加载断点续传"""
        if PATH_TAGS.exists():
            with open(PATH_TAGS, "r", encoding="utf-8") as f:
                self.results = json.load(f)
            self.processed_ids = set(self.results.keys())
            print(f"加载断点: 已处理 {len(self.processed_ids)} 篇")

    def _load_docs(self) -> List[Dict[str, str]]:
        """根据 docs_source 加载文档"""
        if isinstance(self.docs_source, str) and self.docs_source == "rdb":
            return load_docs_from_rdb(doc_ids=None)

        path = Path(self.docs_source)
        if path.is_dir():
            files = sorted(path.glob("*.json"))
            merged = []
            for jf in files:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                merged.extend(_parse_chunks_to_docs(data))
            print(f"  [TagGenerator] 从目录 {path}/ 加载 {len(merged)} 条文档")
            return merged

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = _parse_chunks_to_docs(data)
        print(f"  [TagGenerator] 从 {path.name} 加载 {len(docs)} 条文档")
        return docs

    def run(self, doc_ids: List[str] = None):
        """
        运行标签生成（支持 MySQL 或 chunks 文件）。

        Args:
            doc_ids: 若指定，则只处理这些 doc_id；None = 全量
        """
        # 根据 docs_source 加载
        docs = self._load_docs()

        if doc_ids:
            doc_ids_set = set(doc_ids)
            docs = [d for d in docs if d["id"] in doc_ids_set]

        remaining = [d for d in docs if d["id"] not in self.processed_ids]
        print(f"需要处理: {len(remaining)} 篇 (已处理 {len(self.processed_ids)} 篇)")

        if not remaining:
            print("所有文档已处理完毕")
            return self.results

        save_interval = 50
        count = 0

        for doc in tqdm(remaining, desc="标签抽取"):
            doc_id = doc["id"]
            text = doc.get("text", "")

            if len(text) < 10:
                self.results[doc_id] = {}
                self.processed_ids.add(doc_id)
                continue

            try:
                extracted = self.miner.extract_batch_dimensions(text, self.dims)

                has_valid = False
                doc_tags = {}
                if extracted:
                    for dim, val_list in extracted.items():
                        if val_list and len(val_list) > 0:
                            doc_tags[dim] = val_list
                            has_valid = True

                if not has_valid:
                    keywords = self.miner.extract_keywords_fallback(text)
                    if keywords:
                        doc_tags["关键词"] = keywords

                self.results[doc_id] = doc_tags

            except Exception as e:
                print(f"\n[错误] {doc_id}: {e}")
                continue

            self.processed_ids.add(doc_id)
            count += 1

            if count % save_interval == 0:
                self._save_checkpoint()

        self._save_checkpoint()
        print(f"\n完成: 本次处理 {count} 篇, 总计 {len(self.processed_ids)} 篇")
        return self.results

    def _save_checkpoint(self):
        """保存断点"""
        with open(PATH_TAGS, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)


def run_step3(doc_ids: List[str] = None, docs_source: str = "rdb"):
    """执行 Step 3: 生成文档标签（支持 MySQL 或 chunks 文件）

    Args:
        doc_ids: 仅处理这些 doc_id
        docs_source: "rdb" 或 chunks JSON 文件/目录路径
    """
    print("=" * 60)
    print("Step 3: 生成文档标签")
    print("=" * 60)

    # 读取维度
    if not PATH_V_CORE.exists():
        print("[错误] 请先执行 Step 1 挖掘维度")
        return

    with open(PATH_V_CORE, "r", encoding="utf-8") as f:
        dims = json.load(f)

    print(f"使用维度: {dims}")
    src_label = "MySQL MainIndex.doc_text" if docs_source == "rdb" else str(docs_source)
    print(f"数据源: {src_label}")
    if doc_ids:
        print(f"指定处理: {len(doc_ids)} 个 doc_id")

    # 生成标签（支持 MySQL 或 chunks 文件）
    generator = TagGenerator(dims, docs_source=docs_source)
    tags = generator.run(doc_ids=doc_ids)

    print(f"\n标签生成完成: {len(tags)} 篇文档")
    return tags


# ===================== Step 4: 写入标签到 RDB =====================

def run_step4():
    """执行 Step 4: 将标签写入 RDB"""
    print("=" * 60)
    print("Step 4: 写入标签到 RDB")
    print("=" * 60)

    if not PATH_TAGS.exists():
        print("[错误] 请先执行 Step 3 生成标签")
        return

    with open(PATH_TAGS, "r", encoding="utf-8") as f:
        tags = json.load(f)

    print(f"读取标签: {len(tags)} 篇文档")

    # 连接 RDB
    ensure_rdb_db()
    config = get_config()
    table = config.rdb.table
    existing_cols = fetch_rdb_columns(table)

    # 读取维度
    if PATH_V_CORE.exists():
        with open(PATH_V_CORE, "r", encoding="utf-8") as f:
            dims = json.load(f)
    else:
        dims = []

    # 构建列名映射
    dim_cols = {dim: normalize_attr_key(dim) for dim in dims}

    # 检查需要的列是否存在
    missing_cols = [col for col in dim_cols.values() if col not in existing_cols]
    if missing_cols:
        print(f"[警告] 以下列不存在: {missing_cols}")
        print("请先执行 Step 2 添加维度列")

    # 写入
    conn = connect_rdb()
    cur = conn.cursor()

    updated = 0
    failed = 0
    batch_size = 100
    items = list(tags.items())

    for i in tqdm(range(0, len(items), batch_size), desc="写入进度"):
        batch = items[i:i + batch_size]

        for doc_id, doc_tags in batch:
            try:
                # 过滤掉关键词等非维度标签
                label_cols = []
                label_vals = []

                for dim, col in dim_cols.items():
                    if col not in existing_cols:
                        continue
                    val = doc_tags.get(dim)
                    if val is not None:
                        if isinstance(val, list):
                            val = ",".join(str(v) for v in val)
                        elif not isinstance(val, str):
                            val = str(val)
                        label_cols.append(col)
                        label_vals.append(val)

                if not label_cols:
                    continue

                sql = f"UPDATE `{table}` SET "
                sql += ", ".join(f"`{col}` = %s" for col in label_cols)
                sql += f", `{config.rdb.updated_at_column}` = NOW()"
                sql += f" WHERE `{config.rdb.doc_id_column}` = %s"

                cur.execute(sql, tuple(label_vals + [doc_id]))
                conn.commit()

                if cur.rowcount > 0:
                    updated += 1
                else:
                    failed += 1

            except Exception as e:
                print(f"\n[错误] {doc_id}: {e}")
                failed += 1

    cur.close()
    conn.close()

    print("\n" + "=" * 60)
    print(f"写入完成: 成功 {updated} 篇, 失败 {failed} 篇")
    print("=" * 60)


# ===================== Step 5: 查询解析 =====================

def run_step5(query: str = None, qid: str = "q_001"):
    """执行 Step 5: 查询解析"""
    print("=" * 60)
    print("Step 5: 查询解析")
    print("=" * 60)

    if not PATH_V_CORE.exists():
        print("[错误] 请先执行 Step 1 挖掘维度")
        return

    with open(PATH_V_CORE, "r", encoding="utf-8") as f:
        dims = json.load(f)

    # 从已写入的标签中提取枚举值
    enum_map: Dict[str, Set[str]] = {}

    if PATH_TAGS.exists():
        with open(PATH_TAGS, "r", encoding="utf-8") as f:
            tags = json.load(f)

        for doc_tags in tags.values():
            for dim, vals in doc_tags.items():
                if dim == "关键词":
                    continue
                if dim not in enum_map:
                    enum_map[dim] = set()
                if isinstance(vals, list):
                    for v in vals:
                        enum_map[dim].add(str(v))
                else:
                    enum_map[dim].add(str(vals))

    # 构建枚举映射
    enum_values_map: Dict[str, List[str]] = {}
    for dim in dims:
        if dim in enum_map and enum_map[dim]:
            vals = sorted(enum_map[dim])[:50]  # 限制最多 50 个值
            enum_values_map[dim] = vals

    # 交互式或指定查询
    if not query:
        print("\n请输入查询（输入 quit 退出）:")
        while True:
            query = input("\n查询: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            _parse_query(query, qid, dims, enum_values_map)
            qid = f"{qid.split('_')[0]}_{int(qid.split('_')[1]) + 1 if '_' in qid else 1}"
    else:
        _parse_query(query, qid, dims, enum_values_map)


def _parse_query(query: str, qid: str, dims: List[str], enum_values_map: Dict[str, List[str]]):
    """解析单个查询"""
    print(f"\n查询: {query}")

    miner = DimensionMiningWithQwen()

    try:
        constraints = miner.parse_query_intent(query, dims, enum_values_map)

        print(f"维度约束:")
        for dim, vals in constraints.items():
            print(f"  {dim}: {vals}")

        # 构建 Milvus 过滤表达式（示例）
        print("\n可用的 Milvus 过滤表达式:")
        for dim, vals in constraints.items():
            col = normalize_attr_key(dim)
            if isinstance(vals, list) and vals:
                vals_str = ", ".join(f'"{v}"' for v in vals)
                print(f'  {col} in [{vals_str}]')

        return constraints

    except Exception as e:
        print(f"[错误] 查询解析失败: {e}")
        return {}


# ===================== 主入口 =====================

def main():
    parser = argparse.ArgumentParser(description="RAG_DB 维度 Pipeline 集成")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5],
                        help="执行指定步骤")
    parser.add_argument("--all", action="store_true",
                        help="执行完整流程 (Step 1-4)")
    parser.add_argument("-q", "--query", type=str,
                        help="查询文本 (配合 --step 5 使用)")
    parser.add_argument("--qid", type=str, default="q_001",
                        help="查询 ID")
    parser.add_argument("--force", action="store_true",
                        help="强制重新执行（跳过断点）")
    parser.add_argument("--doc_ids", nargs="*",
                        help="Step 3 专用：只处理指定的 doc_id 列表（默认全量）")
    parser.add_argument("--docs_source", default="rdb",
                        help="Step 1/3 的数据源：'rdb'（MySQL）或 chunks JSON 文件/目录路径")
    parser.add_argument("--validate_all", action="store_true",
                        help="Step 1 完成后对全量 chunk 复核维度 schema（不生成、不写入标签）")

    args = parser.parse_args()

    # 强制模式：删除断点文件
    if args.force:
        for f in [PATH_V_CAND, PATH_V_CORE, PATH_TAGS, PATH_DIM_DIAGNOSTICS]:
            if f.exists():
                f.unlink()
                print(f"已删除: {f}")

    docs_source = getattr(args, 'docs_source', 'rdb') or 'rdb'

    if args.step:
        if args.step == 1:
            run_step1(docs_source=docs_source, validate_all=args.validate_all)
        elif args.step == 2:
            run_step2()
        elif args.step == 3:
            run_step3(doc_ids=args.doc_ids, docs_source=docs_source)
        elif args.step == 4:
            run_step4()
        elif args.step == 5:
            run_step5(query=args.query, qid=args.qid)
    elif args.all:
        print("\n" + "=" * 60)
        print("开始执行完整流程: Step 1 -> 2 -> 3 -> 4")
        print("=" * 60)

        dims = run_step1(docs_source=docs_source, validate_all=args.validate_all)
        run_step2()
        run_step3(docs_source=docs_source)
        run_step4()

        print("\n" + "=" * 60)
        print("完整流程执行完毕!")
        print("=" * 60)
    else:
        parser.print_help()
        print("\n示例:")
        print("  python dimension_integration.py --step 1")
        print("  python dimension_integration.py --step 2")
        print("  python dimension_integration.py --step 3")
        print("  python dimension_integration.py --step 4")
        print("  python dimension_integration.py --step 5 -q '儿童发烧怎么办'")
        print("  python dimension_integration.py --all")


if __name__ == "__main__":
    main()
