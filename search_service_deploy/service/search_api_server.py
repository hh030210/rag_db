# -*- coding: utf-8 -*-
"""
search_api_server.py
====================
FastAPI 入口。暴露三个接口：

    GET  /healthz          健康检查
    GET  /config           当前配置（便于排障）
    POST /search           检索

    POST /search 请求体：
        {
            "query":             "原始查询 1~500 字符",
            "mode":              "sem" | "dim" | "fusion"，默认 "sem"
            "top_k":             8,                           # 默认 8
            "alpha_dim":         null,                        # 覆盖维度权重
            "alpha_sem":         null,                        # 覆盖语义权重
            "fusion_strategy":   "adaptive" | "rrf" | "score"，默认 adaptive
        }

    行为：
        - alpha_* 都给 → 用给的固定权重（不再权重计算）
        - alpha_* 给一个 → 用给的（另一个 = 1-此）
        - 都不给 → 自适应权重（P/T/C）计算权重
        - 返回 top 8：
            * mode=sem → sem_results (top 8)
            * mode=dim → dim_results (top 8)
            * mode=fusion → dim_results + sem_results + fusion_results（各 top 8）

    v1.1 变更（2026-07-16）：
        - 删除 data.elapsed
        - 删除 data.alpha_dim / data.alpha_sem / data.alpha_source
        - 删除 data.weight_breakdown
        - 删除 data.constraints
        - 删除顶层 data.mode / data.top_k / data.query 回显
        - ChunkHit 删除：chunk_gen_title / recall_score / rerank_score /
                        internal_fusion / recall_method / chunk_text_full

启动：
    uvicorn search_api_server:app --host 0.0.0.0 --port 8100 --workers 1
"""

import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field


# ===================== Pydantic 模型 =====================

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="原始查询")
    mode: str = Field("sem", description="sem | dim | fusion")
    top_k: int = Field(8, ge=1, le=50, description="返回条数，默认 8")

    alpha_dim: Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="维度权重（覆盖自适应）；与 alpha_sem 至少给一个",
    )
    alpha_sem: Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="语义权重（覆盖自适应）；与 alpha_dim 至少给一个",
    )
    fusion_strategy: Optional[str] = Field(
        "adaptive",
        description="adaptive | rrf | score（adaptive 模式下 alpha 覆盖仍生效）",
    )


class ChunkHit(BaseModel):
    """单个检索命中。
    保留：chunk_id / doc_title / score / final_score / dim_rank / sem_rank /
          fusion_rank / source / dim_name / tag_name / tag_hits / evidence / chunk_text
    删除（v1.1）：chunk_gen_title / recall_score / rerank_score /
                  internal_fusion / recall_method / chunk_text_full
    """
    chunk_id: Optional[str]
    doc_title: Optional[str] = None
    score: Optional[float] = None
    final_score: Optional[float] = None
    dim_rank: Optional[int] = None
    sem_rank: Optional[int] = None
    fusion_rank: Optional[int] = None
    source: Optional[str] = None
    dim_name: Optional[str] = None
    tag_name: Optional[str] = None
    tag_hits: Optional[List[dict]] = None
    evidence: Optional[List[str]] = None
    chunk_text: Optional[str] = None


class SearchData(BaseModel):
    """v1.1：仅保留三组结果，删掉所有权重/约束/耗时/回显字段。"""
    sem_results: List[ChunkHit] = Field(default_factory=list)
    dim_results: List[ChunkHit] = Field(default_factory=list)
    fusion_results: List[ChunkHit] = Field(default_factory=list)


class SearchResponse(BaseModel):
    ok: bool
    code: int
    data: Optional[SearchData] = None
    error: Optional[str] = None


# ===================== 路由注册 =====================

def register_routes(app: FastAPI, pipeline, config) -> None:
    """为 FastAPI app 绑定 /healthz、/config 和 /search"""

    @app.get("/healthz")
    def healthz():
        return {
            "ok": True,
            "service": "search-service",
            "mode": "ready",
            "qdrant": f"{config.qdrant_host}:{config.qdrant_port}",
            "collection": config.chunk_collection,
        }

    @app.get("/config")
    def get_config():
        return {
            "chunk_collection": config.chunk_collection,
            "dim_tags_collection": config.dim_tags_collection,
            "recall_method": config.recall_method,
            "rerank_method": config.rerank_method,
            "internal_fusion": config.internal_fusion,
            "sem_top_k": config.sem_top_k,
            "dim_top_k": config.dim_top_k,
            "dim_rerank_top_k": config.dim_rerank_top_k,
            "default_top_k": config.default_top_k,
            "fusion_strategy": config.fusion_strategy,
        }

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest):
        try:
            if req.mode not in ("sem", "dim", "fusion"):
                return SearchResponse(
                    ok=False,
                    code=400,
                    error=f"invalid mode: {req.mode}（应为 sem / dim / fusion）",
                )

            strat = req.fusion_strategy or "adaptive"
            if strat not in ("adaptive", "rrf", "score"):
                return SearchResponse(
                    ok=False,
                    code=400,
                    error=f"invalid fusion_strategy: {strat}（应为 adaptive / rrf / score）",
                )

            out = pipeline.search(
                query_text=req.query,
                mode=req.mode,
                top_k=req.top_k,
                alpha_dim=req.alpha_dim,
                alpha_sem=req.alpha_sem,
                fusion_strategy=strat,
            )

            # v1.1：data 只保留三组结果，过滤 ChunkHit 上已被删的字段
            allowed = {
                "chunk_id", "doc_title", "score", "final_score",
                "dim_rank", "sem_rank", "fusion_rank", "source",
                "dim_name", "tag_name", "tag_hits", "evidence",
                "chunk_text",
            }

            def _clean(rows):
                cleaned = []
                for r in (rows or []):
                    cleaned.append({k: v for k, v in r.items() if k in allowed})
                return cleaned

            data = SearchData(
                sem_results=[ChunkHit(**r) for r in _clean(out.get("sem_results", []))],
                dim_results=[ChunkHit(**r) for r in _clean(out.get("dim_results", []))],
                fusion_results=[ChunkHit(**r) for r in _clean(out.get("fusion_results", []))],
            )
            return SearchResponse(ok=True, code=200, data=data)
        except Exception as e:
            return SearchResponse(ok=False, code=500, error=str(e))


# ===================== 直接启动 =====================

def _make_app():
    """给 uvicorn 用的工厂"""
    try:
        from service_launcher import build_app
        return build_app()
    except Exception as e:
        # 若服务装配失败（Qdrant 不可达 / 模型加载失败），仍暴露健康检查
        import logging
        logging.exception("build_app failed: %s", e)
        app = FastAPI(title="RAG Search (fallback)")
        @app.get("/healthz")
        def _h():
            return {"ok": False, "service": "search-service", "error": str(e)[:300]}
        return app


# uvicorn search_api_server:app 默认查找 'app' 变量
app = _make_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8100")), workers=1)