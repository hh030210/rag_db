# -*- coding: utf-8 -*-
"""
service_launcher.py
===================
启动入口：根据 ServerConfig 加载模型 / 索引，构建 pipeline，包装为 FastAPI 启动器。

可以这样用：

    from service_launcher import build_app
    app = build_app("/path/to/db_config.yaml")
    uvicorn.run(app, host="0.0.0.0", port=8100)
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from config import load_config, ServerConfig
from qdrant_client import QdrantHTTPClient
from bge_encoder import load_bge_encoder
from semantic_searcher import SemanticSearcher
from dim_searcher import DimSearcher
from pipeline import SearchPipeline


logger = logging.getLogger("search_service")


def _load_tag_vectors(path: Optional[str]) -> dict:
    if not path or not Path(path).exists():
        return {}
    try:
        with open(path, "rb") as f:
            tv = pickle.load(f)
        logger.info("已加载标签向量: %d 条 (%s)", len(tv), path)
        return tv
    except Exception as e:
        logger.warning("tag_vectors.pkl 加载失败: %s", e)
        return {}


def build_app(config_path: Optional[str] = None) -> FastAPI:
    """构造 FastAPI app（pipeline + 全局状态）"""
    from search_api_server import register_routes
    cfg = load_config(config_path)

    logger.info("启动检索服务...")
    logger.info("Qdrant: %s:%d", cfg.qdrant_host, cfg.qdrant_port)
    logger.info("Collection: %s / dim_tags: %s", cfg.chunk_collection, cfg.dim_tags_collection)

    client = QdrantHTTPClient(cfg.qdrant_host, cfg.qdrant_port)
    healthy = client.health()
    logger.info("Qdrant healthz: %s", "OK" if healthy else "FAIL")

    encoder = load_bge_encoder()
    if encoder is None:
        logger.warning("BGE-M3 编码器加载失败：搜索结果可能为空")

    tag_vectors = _load_tag_vectors(cfg.tag_vectors_path or "experiment_data/tag_vectors.pkl")

    sem = SemanticSearcher(
        client=client,
        collection_name=cfg.chunk_collection,
        encoder=encoder,
    )
    dim = DimSearcher(
        client=client,
        chunk_collection=cfg.chunk_collection,
        dim_tags_collection=cfg.dim_tags_collection,
        encoder=encoder,
        recall_method=cfg.recall_method,
        rerank_method=cfg.rerank_method,
        internal_fusion=cfg.internal_fusion,
        dim_top_k=cfg.dim_top_k,
        dim_rerank_top_k=cfg.dim_rerank_top_k,
        tag_vectors=tag_vectors,
        parse_constraints_fn=None,
    )
    pipeline = SearchPipeline(
        sem_searcher=sem,
        dim_searcher=dim,
        sem_top_k=cfg.sem_top_k,
        dim_top_k=cfg.dim_top_k,
    )

    app = FastAPI(
        title="RAG 检索服务",
        version="1.0.0",
        description="独立检索服务（语义 / 维度 / 融合 / 自适应权重），top8 返回",
    )
    register_routes(app, pipeline, cfg)
    return app
