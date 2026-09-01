# -*- coding: utf-8 -*-
"""
config.py
=========
服务端配置加载：
    - 内置默认（容器内 /models/bge-m3 等）
    - 环境变量覆盖（HOST / PORT / QDRANT_HOST / QDRANT_PORT / ...）
    - 可选 yaml 文件覆盖（默认 db_config.yaml）
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ServerConfig:
    # 服务端
    host: str = "0.0.0.0"
    port: int = 8100

    # Qdrant
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333
    chunk_collection: str = "unified_corpus"
    dim_tags_collection: str = "dimension_tags"
    vector_dim: int = 1024

    # Embedding
    bge_model_path: Optional[str] = None
    bge_device: str = "cpu"
    bge_use_fp16: bool = False

    # 检索默认
    recall_method: str = "tag"
    rerank_method: str = "tag_sim"
    internal_fusion: str = "score"
    sem_top_k: int = 20
    dim_top_k: int = 100
    dim_rerank_top_k: int = 50
    default_top_k: int = 8
    fusion_strategy: str = "rrf"

    # 距量文件
    tag_vectors_path: Optional[str] = None
    inverted_index_path: Optional[str] = None
    dimension_metadata_path: Optional[str] = None


def _try_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_config(config_path: Optional[str] = None) -> ServerConfig:
    cfg = ServerConfig()

    # ── 1. env 覆盖 ──
    cfg.host = os.getenv("HOST", cfg.host)
    cfg.port = int(os.getenv("PORT", str(cfg.port)))
    cfg.qdrant_host = os.getenv("QDRANT_HOST", cfg.qdrant_host)
    cfg.qdrant_port = int(os.getenv("QDRANT_PORT", str(cfg.qdrant_port)))
    cfg.chunk_collection = os.getenv("CHUNK_COLLECTION", cfg.chunk_collection)
    cfg.dim_tags_collection = os.getenv("DIM_TAGS_COLLECTION", cfg.dim_tags_collection)
    cfg.vector_dim = int(os.getenv("VECTOR_DIM", str(cfg.vector_dim)))
    cfg.bge_model_path = os.getenv("BGE_MODEL_PATH", None)
    cfg.bge_device = os.getenv("BGE_DEVICE", cfg.bge_device)
    cfg.bge_use_fp16 = os.getenv("BGE_USE_FP16", "0") not in ("0", "false", "False")
    cfg.recall_method = os.getenv("DIM_RECALL_METHOD", cfg.recall_method)
    cfg.rerank_method = os.getenv("DIM_RERANK_METHOD", cfg.rerank_method)
    cfg.internal_fusion = os.getenv("DIM_INTERNAL_FUSION", cfg.internal_fusion)
    cfg.sem_top_k = int(os.getenv("SEM_TOP_K", str(cfg.sem_top_k)))
    cfg.dim_top_k = int(os.getenv("DIM_TOP_K", str(cfg.dim_top_k)))
    cfg.dim_rerank_top_k = int(os.getenv("DIM_RERANK_TOP_K", str(cfg.dim_rerank_top_k)))
    cfg.default_top_k = int(os.getenv("DEFAULT_TOP_K", str(cfg.default_top_k)))
    cfg.fusion_strategy = os.getenv("FUSION_STRATEGY", cfg.fusion_strategy)
    cfg.tag_vectors_path = os.getenv("TAG_VECTORS_PATH", None)
    cfg.inverted_index_path = os.getenv("INVERTED_INDEX_PATH", None)
    cfg.dimension_metadata_path = os.getenv("DIM_META_PATH", None)

    # ── 2. yaml 覆盖（env 优先）──
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", str(Path(__file__).parent / "db_config.yaml"))
    yaml_cfg = _try_yaml(Path(config_path))
    if yaml_cfg:
        # 服务端
        api_cfg = yaml_cfg.get("api", {})
        embed_cfg = yaml_cfg.get("embedding", {})
        vecdb_cfg = yaml_cfg.get("vecdb_qdrant") or yaml_cfg.get("vecdb", {})

        cfg.qdrant_host = os.getenv("QDRANT_HOST", vecdb_cfg.get("host", cfg.qdrant_host))
        cfg.qdrant_port = int(os.getenv("QDRANT_PORT", str(vecdb_cfg.get("port", cfg.qdrant_port))))
        cfg.chunk_collection = os.getenv("CHUNK_COLLECTION", vecdb_cfg.get("collection_name", cfg.chunk_collection))
        cfg.dim_tags_collection = os.getenv("DIM_TAGS_COLLECTION", yaml_cfg.get("dim_tags_collection", vecdb_cfg.get("dim_tags_collection", cfg.dim_tags_collection)))
        cfg.vector_dim = int(os.getenv("VECTOR_DIM", str(vecdb_cfg.get("vector_dim", cfg.vector_dim))))

        cfg.bge_model_path = os.getenv("BGE_MODEL_PATH", embed_cfg.get("model_path", cfg.bge_model_path))
        # model_path 可能是 ${VAR:-/models}/bge-m3 形式，做一次简单展开
        if cfg.bge_model_path and "${" in cfg.bge_model_path:
            import re
            def _exp(m):
                var = m.group(1)
                default = m.group(2) if m.group(2) is not None else ""
                return os.environ.get(var, default)
            cfg.bge_model_path = re.sub(r"\$\{([^}:]+)(?::-([^}]*))?\}", _exp, cfg.bge_model_path)

    return cfg
