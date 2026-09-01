# -*- coding: utf-8 -*-
"""
api_server.py
=============
FastAPI HTTP 入口，两个独立接口：

    POST /optimize       场景级：输入 query → 子查询 + 实体 + 聚类 PromptModule
    POST /optimize_case  案例级：输入 query + prompt + direction
                         可选传入场景级结果，在场景上下文基础上做 prompt 定向改进
    GET  /healthz        健康检查

启动：
    pip install -r requirements_api.txt
    python api_server.py
    # 或生产：
    # uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1

环境变量：
    HOST           监听地址（默认 0.0.0.0）
    PORT           监听端口（默认 8000）

DashScope API Key 传递方式（**两种任选其一，推荐 header**）：
    1. HTTP Header（推荐）：
           Authorization: Bearer sk-xxxxxxxxxxxxxxxxxxxxxxxx
       自动从 ``Authorization: Bearer <token>`` 解析。
    2. 请求体 body（兼容老调用）：
           {"api_key": "sk-xxx", "query": "..."}

    优先级：header > body。
    服务端不做任何 Key 缓存或落盘，避免上传服务器后泄露。
"""

import os
from typing import Optional

import requests
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field
import prompt_iteration_optimizer as _pio

from prompt_iteration_service import (
    optimize_for_query,
    passthrough,
)
from case_level_service import refine_prompt, infer_direction_and_refine

# 案例级迭代优化（专利方案落实）：metrics / initial_prompt / iterate / cluster / prototypes
from case_level_optimizer_service import (
    build_evaluation_metrics,
    build_initial_prompt,
    judge_answer,
    score_answer,
    refine_with_eval,
    iterate_prompt_until_converged,
    cluster_case_samples,
    extract_scene_prototypes,
    archive_optimization,
    write_archive_to_file,
    _safe_json_load,
    _normalize_weights,
)


app = FastAPI(
    title="Prompt Iteration Optimizer API",
    version="1.0.0",
    description="景区 QA 场景下的 prompt 迭代优化服务",
)


def _resolve_api_key(authorization: Optional[str], body_api_key: Optional[str]) -> str:
    """
    从 header 或 body 解析 DashScope API Key。

    Header 优先；body 作为回退，保证老调用方不需要改代码。
    """
    if authorization and authorization.strip():
        auth = authorization.strip()
        # 标准 OAuth2 风格: "Bearer <token>"
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        # 兼容裸 token: "sk-xxx"
        return auth
    if body_api_key and body_api_key.strip():
        return body_api_key.strip()
    raise ValueError(
        "DashScope API Key 未提供：请通过 Header `Authorization: Bearer <key>` "
        "或在请求 body `api_key` 字段传入。"
    )


# ────────────────────────────────────────────
# /optimize Pydantic 模型
# ────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    api_key: Optional[str] = Field(
        None,
        description="DashScope API Key（可选；推荐改用 Header `Authorization: Bearer <key>`）。"
    )
    query: str = Field(..., min_length=1, max_length=500)
    use_llm_subqueries: Optional[bool] = True
    use_entity_extraction: Optional[bool] = True
    use_cluster_prompt: Optional[bool] = True
    cluster_top_n: Optional[int] = Field(2, ge=1, le=5)
    base_prompt: Optional[str] = Field(None, description="兜底 system prompt，聚类未命中且 LLM 失败时使用")
    return_optimized_prompt: Optional[bool] = Field(True, description="是否额外 LLM 生成优化后的 system prompt")


class OptimizeData(BaseModel):
    original_query: str
    fusion_query: str
    sub_queries: list
    entity_terms: list
    cluster_id: Optional[int]
    cluster_sim: float
    prompt_module: Optional[dict]
    top_clusters: list
    top_prompts: list
    optimized_prompt: Optional[str] = None
    optimize_time: float
    total_time: float


class OptimizeResponse(BaseModel):
    ok: bool
    code: int
    data: Optional[OptimizeData] = None
    error: Optional[str] = None


# ────────────────────────────────────────────
# /passthrough Pydantic 模型
# ────────────────────────────────────────────

class PassthroughRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    prompt: str = Field(..., min_length=0)
    direction: Optional[str] = ""


class PassthroughData(BaseModel):
    query: str
    prompt: str
    direction: str
    processed: bool


class PassthroughResponse(BaseModel):
    ok: bool
    code: int
    data: Optional[PassthroughData] = None
    error: Optional[str] = None


# ────────────────────────────────────────────
# /optimize_case Pydantic 模型（放在路由前，/passthrough 也要用）
# ────────────────────────────────────────────

class OptimizeCaseRequest(BaseModel):
    api_key: Optional[str] = Field(
        None,
        description="DashScope API Key（可选；推荐用 Header `Authorization: Bearer <key>`）"
    )
    query: str = Field(..., min_length=1, max_length=500)
    prompt: str = Field(..., min_length=1, description="当前使用的 system prompt")
    direction: str = Field(
        ...,
        description="迭代方向：简洁 / 详细 / 专业化 / 友好 / 精确 / 全面（也支持自定义描述）"
    )
    scenario_result: Optional[dict] = Field(
        None,
        description="场景级优化结果（来自 /optimize 接口）。案例级基于场景上下文做定向改进"
    )


class OptimizeCaseData(BaseModel):
    original_query: str
    refined_prompt: str
    direction: str
    scenario_used: bool
    iterations: int
    llm_time: float
    total_time: float


class OptimizeCaseResponse(BaseModel):
    ok: bool
    code: int
    data: Optional[OptimizeCaseData] = None
    error: Optional[str] = None


# ────────────────────────────────────────────
# /refine_with_gt Pydantic 模型
# ────────────────────────────────────────────

class RefineWithGtRequest(BaseModel):
    api_key: Optional[str] = Field(
        None,
        description="DashScope API Key（可选；推荐用 Header `Authorization: Bearer <key>`）"
    )
    query: str = Field(..., min_length=1, max_length=500)
    prompt: str = Field(..., min_length=1, description="当前使用的 system prompt")
    ground_truth: str = Field(..., min_length=1, description="标准答案（用户提供的参考答案）")
    search_url: str = Field(
        "http://81.70.191.196:80/search",
        description="Search 服务地址"
    )
    max_iterations: Optional[int] = Field(3, ge=1, le=10, description="最大迭代次数，默认 3")


class RefineWithGtData(BaseModel):
    original_query: str
    inferred_direction: str
    refined_prompt: str
    iterations: int
    llm_time: float
    total_time: float


class RefineWithGtResponse(BaseModel):
    ok: bool
    code: int
    data: Optional[RefineWithGtData] = None
    error: Optional[str] = None


# ────────────────────────────────────────────
# 路由
# ────────────────────────────────────────────

@app.post("/optimize", response_model=OptimizeResponse)
def optimize(
    req: OptimizeRequest,
    authorization: Optional[str] = Header(
        None,
        alias="Authorization",
        description="Bearer DashScope API Key（推荐方式）",
    ),
):
    """场景级 prompt 迭代优化：query → 子查询 + 实体 + 聚类 PromptModule。"""
    try:
        api_key = _resolve_api_key(authorization, req.api_key)
        data = optimize_for_query(
            query=req.query,
            api_key=api_key,
            use_llm_subqueries=req.use_llm_subqueries,
            use_entity_extraction=req.use_entity_extraction,
            use_cluster_prompt=req.use_cluster_prompt,
            cluster_top_n=req.cluster_top_n,
            base_prompt=req.base_prompt,
            return_optimized_prompt=req.return_optimized_prompt,
        )
        return OptimizeResponse(ok=True, code=200, data=data)
    except ValueError as ve:
        return OptimizeResponse(ok=False, code=400, error=str(ve))
    except Exception as e:
        return OptimizeResponse(ok=False, code=500, error=str(e))


@app.post("/passthrough", response_model=OptimizeCaseResponse)
def passthrough_endpoint(
    req: PassthroughRequest,
    authorization: Optional[str] = Header(
        None,
        alias="Authorization",
        description="Bearer DashScope API Key（推荐方式）",
    ),
):
    """
    透传接口（内部调用案例级 refine_prompt）：
    把 query / prompt / direction 透传给 LLM 做案例级 prompt 改进。
    """
    try:
        api_key = _resolve_api_key(authorization, req.api_key if hasattr(req, "api_key") else None)
        # 从 body 字段解析（兼容请求体透传方式）
        body_data = {}
        if req.query and req.prompt and req.direction is not None:
            body_data = {"query": req.query, "prompt": req.prompt, "direction": req.direction}
        result = refine_prompt(
            query=body_data.get("query", ""),
            prompt=body_data.get("prompt", ""),
            direction=body_data.get("direction", ""),
            api_key=api_key,
            scenario_result=None,
        )
        data = OptimizeCaseData(
            original_query=body_data.get("query", req.query or ""),
            refined_prompt=result["refined_prompt"],
            direction=result["direction"],
            scenario_used=result["scenario_used"],
            iterations=result.get("iterations", 1),
            llm_time=result["llm_time"],
            total_time=result["total_time"],
        )
        return OptimizeCaseResponse(ok=True, code=200, data=data)
    except ValueError as ve:
        return OptimizeCaseResponse(ok=False, code=400, error=str(ve))
    except Exception as e:
        return OptimizeCaseResponse(ok=False, code=500, error=str(e))


@app.get("/healthz")
def health():
    return {"ok": True, "service": "prompt-iteration-optimizer"}


@app.post("/refine_with_gt", response_model=RefineWithGtResponse)
def refine_with_gt_endpoint(
    req: RefineWithGtRequest,
    authorization: Optional[str] = Header(
        None,
        alias="Authorization",
        description="Bearer DashScope API Key（推荐方式）",
    ),
):
    """
    基于标准答案自动推断迭代方向并改进 prompt。

    流程：
    1. 调用 search 接口检索上下文
    2. 将参考答案 + 检索上下文 + 当前 prompt 传给 LLM 推断迭代方向
    3. 用推断出的方向调用 refine_prompt 进行多轮优化
    """
    try:
        api_key = _resolve_api_key(authorization, req.api_key)

        # ── Step 1: 调用 search 检索上下文 ──
        try:
            search_resp = requests.post(
                req.search_url,
                json={"query": req.query, "mode": "fusion", "top_k": 3},
                timeout=15,
            )
            search_data = search_resp.json()
        except Exception as e:
            return RefineWithGtResponse(
                ok=False, code=500,
                error=f"Search 服务调用失败: {str(e)}"
            )

        if not search_data.get("ok"):
            return RefineWithGtResponse(
                ok=False, code=500,
                error=f"Search 服务返回错误: {search_data.get('error')}"
            )

        # 提取检索结果文本
        fusion_results = search_data.get("data", {}).get("fusion_results", [])
        if not fusion_results:
            return RefineWithGtResponse(
                ok=False, code=400,
                error="Search 返回空结果，无法进行方向推断"
            )

        retrieved_context = "\n".join([
            h.get("doc_text", "") or h.get("text", "") or ""
            for h in fusion_results[:3]
            if h.get("doc_text") or h.get("text")
        ])

        if not retrieved_context.strip():
            return RefineWithGtResponse(
                ok=False, code=400,
                error="Search 返回的文档内容为空"
            )

        # ── Step 2+3: 推断方向 + 改进 prompt ──
        result = infer_direction_and_refine(
            query=req.query,
            prompt=req.prompt,
            ground_truth=req.ground_truth,
            retrieved_context=retrieved_context,
            api_key=api_key,
            max_iterations=req.max_iterations,
        )

        data = RefineWithGtData(
            original_query=req.query,
            inferred_direction=result["inferred_direction"],
            refined_prompt=result["refined_prompt"],
            iterations=result["iterations"],
            llm_time=result["llm_time"],
            total_time=result["total_time"],
        )
        return RefineWithGtResponse(ok=True, code=200, data=data)

    except ValueError as ve:
        return RefineWithGtResponse(ok=False, code=400, error=str(ve))
    except Exception as e:
        return RefineWithGtResponse(ok=False, code=500, error=str(e))


@app.post("/optimize_case", response_model=OptimizeCaseResponse)
def optimize_case_endpoint(
    req: OptimizeCaseRequest,
    authorization: Optional[str] = Header(
        None,
        alias="Authorization",
        description="Bearer DashScope API Key（推荐方式）",
    ),
):
    """
    案例级 prompt 迭代优化：独立接口，与场景级 /optimize 完全分离。

    输入：
        - query:     原始问题
        - prompt:    当前使用的 system prompt
        - direction: 迭代方向（简洁/详细/专业化/友好/精确/全面）

    输出：
        - refined_prompt:  改进后的 system prompt
    """
    try:
        api_key = _resolve_api_key(authorization, req.api_key)
        result = refine_prompt(
            query=req.query,
            prompt=req.prompt,
            direction=req.direction,
            api_key=api_key,
            scenario_result=req.scenario_result,
        )
        data = OptimizeCaseData(
            original_query=req.query,
            refined_prompt=result["refined_prompt"],
            direction=result["direction"],
            scenario_used=result["scenario_used"],
            iterations=result.get("iterations", 1),
            llm_time=result["llm_time"],
            total_time=result["total_time"],
        )
        return OptimizeCaseResponse(ok=True, code=200, data=data)
    except ValueError as ve:
        return OptimizeCaseResponse(ok=False, code=400, error=str(ve))
    except Exception as e:
        return OptimizeCaseResponse(ok=False, code=500, error=str(e))


# ────────────────────────────────────────────
# 案例级迭代优化（专利方案）Pydantic 模型 + 路由
# 目录：/case_iter/{metrics,initial_prompt,iterate,cluster,prototypes,archive}
# ────────────────────────────────────────────

class CaseIterMetricsRequest(BaseModel):
    api_key: Optional[str] = Field(None, description="DashScope API Key（推荐 Header）")
    field: str = Field(..., min_length=1, max_length=64, description="业务领域，如'旅游景区'")
    scenario_result: Optional[dict] = Field(None, description="场景级结果（来自 /optimize），可选")


class CaseIterMetricsData(BaseModel):
    field: str
    metrics: list
    parse_ok: bool
    used_fallback: bool
    llm_time: float


class CaseIterMetricsResponse(BaseModel):
    ok: bool
    code: int
    data: Optional[CaseIterMetricsData] = None
    error: Optional[str] = None


@app.post("/case_iter/metrics", response_model=CaseIterMetricsResponse)
def case_iter_metrics(
    req: CaseIterMetricsRequest,
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """根据领域 + 可选场景级结果，调用 LLM 生成多维评估指标（含权重）。"""
    try:
        api_key = _resolve_api_key(authorization, req.api_key)
        result = build_evaluation_metrics(
            field=req.field,
            expand_result=req.scenario_result,
            api_key=api_key,
        )
        return CaseIterMetricsResponse(
            ok=True, code=200,
            data=CaseIterMetricsData(
                field=result["field"],
                metrics=result["metrics"],
                parse_ok=result["parse_ok"],
                used_fallback=result["used_fallback"],
                llm_time=result["llm_time"],
            ),
        )
    except ValueError as ve:
        return CaseIterMetricsResponse(ok=False, code=400, error=str(ve))
    except Exception as e:
        return CaseIterMetricsResponse(ok=False, code=500, error=str(e))


class CaseIterInitialPromptRequest(BaseModel):
    api_key: Optional[str] = Field(None)
    field: str = Field(..., min_length=1, max_length=64)
    query: str = Field(..., min_length=1, max_length=500)
    scenario_result: Optional[dict] = None
    output_format: Optional[str] = Field("Markdown 段落 + 必要时使用列表", description="输出格式描述")
    base_prompt: Optional[str] = Field(None, description="兜底 system prompt，LLM 失败时回退")


class CaseIterInitialPromptData(BaseModel):
    field: str
    query: str
    initial_prompt: str
    used_base: bool
    llm_time: float


class CaseIterInitialPromptResponse(BaseModel):
    ok: bool
    code: int
    data: Optional[CaseIterInitialPromptData] = None
    error: Optional[str] = None


@app.post("/case_iter/initial_prompt", response_model=CaseIterInitialPromptResponse)
def case_iter_initial_prompt(
    req: CaseIterInitialPromptRequest,
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """生成结构化初始 system prompt（专利 Step 2）。"""
    try:
        api_key = _resolve_api_key(authorization, req.api_key)
        result = build_initial_prompt(
            field=req.field,
            query=req.query,
            expand_result=req.scenario_result,
            api_key=api_key,
            output_format=req.output_format,
            base_prompt=req.base_prompt,
        )
        return CaseIterInitialPromptResponse(
            ok=True, code=200,
            data=CaseIterInitialPromptData(
                field=result["field"],
                query=req.query,
                initial_prompt=result["initial_prompt"],
                used_base=result["used_base"],
                llm_time=result["llm_time"],
            ),
        )
    except ValueError as ve:
        return CaseIterInitialPromptResponse(ok=False, code=400, error=str(ve))
    except Exception as e:
        return CaseIterInitialPromptResponse(ok=False, code=500, error=str(e))


class CaseIterIterateRequest(BaseModel):
    api_key: Optional[str] = Field(None)
    query: str = Field(..., min_length=1, max_length=500)
    prompt: str = Field(..., min_length=1, description="首版 prompt，可来自 initial_prompt 接口")
    ground_truth: str = Field(..., min_length=1, description="参考答案")
    search_url: str = Field(
        "http://81.70.191.196:80/search",
        description="Search 服务地址（接口内部自动调用）"
    )
    metrics: list = Field(..., description="评估指标列表（来自 /case_iter/metrics）")
    max_iterations: Optional[int] = Field(5, ge=1, le=20)
    score_threshold: Optional[float] = Field(85.0, ge=0, le=100)
    score_window: Optional[int] = Field(2, ge=1, le=10)
    min_delta: Optional[float] = Field(0.5, ge=0, le=10)
    monotonic_max_retries: Optional[int] = Field(3, ge=1, le=10, description="每次改写时单调性验证的最大重试次数，默认 3")


class CaseIterIterateData(BaseModel):
    best_prompt: str
    best_score: float
    iterations: int
    stop_reason: str
    history: list
    total_llm_time: float
    total_time: float


class CaseIterIterateResponse(BaseModel):
    ok: bool
    code: int
    data: Optional[CaseIterIterateData] = None
    error: Optional[str] = None


@app.post("/case_iter/iterate", response_model=CaseIterIterateResponse)
def case_iter_iterate(
    req: CaseIterIterateRequest,
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    案例级多轮迭代优化（Algorithm 1）。

    注意：本接口自身不调外部回答模型——把首版 prompt 当作"当前最佳 prompt"，
    每次迭代的"候选答案"由本服务内部用 prompt_iteration_optimizer._call_llm
    充当 generate_answer_fn（即承担下游回答模型的角色）。
    真正用于生产时建议调用方在外部跑下游模型，把结果 ground-truth-aware 微调。
    """
    try:
        api_key = _resolve_api_key(authorization, req.api_key)

        # ── Step 1: 调用 search 检索上下文 ──
        try:
            search_resp = requests.post(
                req.search_url,
                json={"query": req.query, "mode": "fusion", "top_k": 3},
                timeout=15,
            )
            search_data = search_resp.json()
        except Exception as e:
            return CaseIterIterateResponse(
                ok=False, code=500,
                error=f"Search 服务调用失败: {str(e)}"
            )

        if not search_data.get("ok"):
            return CaseIterIterateResponse(
                ok=False, code=500,
                error=f"Search 返回错误: {search_data.get('error')}"
            )

        # 提取检索结果文本
        fusion_results = search_data.get("data", {}).get("fusion_results", [])
        if not fusion_results:
            return CaseIterIterateResponse(
                ok=False, code=400,
                error="Search 返回空结果，无法进行迭代优化"
            )

        retrieved_context = "\n".join([
            r.get("content", r.get("text", "")) for r in fusion_results
        ])
        if not retrieved_context.strip():
            return CaseIterIterateResponse(
                ok=False, code=400,
                error="Search 返回的文档内容为空"
            )

        # ── Step 2: 归一化 metrics ──
        metrics = _normalize_weights(req.metrics or [])

        def _answer_with_internal_llm(prompt_text: str) -> str:
            """充当 generate_answer_fn：用 prompt + 私域上下文回答用户问题。"""
            user_msg = (
                f"【私域上下文】\n{retrieved_context[:3000]}\n\n"
                f"【用户问题】\n{req.query}\n\n"
                "请用中文基于上述私域上下文回答用户问题。"
                "若上下文不足请明确说明，不要编造。"
            )
            ans = _pio._call_llm(
                prompt=user_msg,
                system=prompt_text,
                temperature=0.3,
                max_tokens=512,
                api_key=api_key,
                timeout=30,
            )
            return ans.strip()

        result = iterate_prompt_until_converged(
            query=req.query,
            initial_prompt=req.prompt,
            ground_truth=req.ground_truth,
            retrieved_context=retrieved_context,
            metrics=metrics,
            generate_answer_fn=_answer_with_internal_llm,
            api_key=api_key,
            max_iterations=req.max_iterations,
            score_threshold=req.score_threshold,
            score_window=req.score_window,
            min_delta=req.min_delta,
            monotonic_max_retries=req.monotonic_max_retries,
        )
        return CaseIterIterateResponse(
            ok=True, code=200,
            data=CaseIterIterateData(
                best_prompt=result["best_prompt"],
                best_score=result["best_score"],
                iterations=result["iterations"],
                stop_reason=result["stop_reason"],
                history=result["history"],
                total_llm_time=result["total_llm_time"],
                total_time=result["total_time"],
            ),
        )
    except ValueError as ve:
        return CaseIterIterateResponse(ok=False, code=400, error=str(ve))
    except Exception as e:
        return CaseIterIterateResponse(ok=False, code=500, error=str(e))


class CaseIterClusterRequest(BaseModel):
    samples: list = Field(..., description="案例列表，每条含 question / optimized_prompt")
    n_clusters: Optional[int] = Field(None, ge=2, le=20, description="聚类数；为空时启发式取值")


class CaseIterClusterData(BaseModel):
    n_samples: int
    n_clusters: int
    clusters: list
    encoders_used: bool


class CaseIterClusterResponse(BaseModel):
    ok: bool
    code: int
    data: Optional[CaseIterClusterData] = None
    error: Optional[str] = None


@app.post("/case_iter/cluster", response_model=CaseIterClusterResponse)
def case_iter_cluster(req: CaseIterClusterRequest):
    """对问答样例 + 优化后 prompt 进行聚类（复用 BGE 编码器）。"""
    try:
        result = cluster_case_samples(
            samples=req.samples,
            n_clusters=req.n_clusters,
        )
        return CaseIterClusterResponse(
            ok=True, code=200,
            data=CaseIterClusterData(
                n_samples=result["n_samples"],
                n_clusters=result["n_clusters"],
                clusters=result["clusters"],
                encoders_used=result["encoders_used"],
            ),
        )
    except Exception as e:
        return CaseIterClusterResponse(ok=False, code=500, error=str(e))


class CaseIterPrototypesRequest(BaseModel):
    api_key: Optional[str] = Field(None)
    clusters: list = Field(..., description="聚类结果（来自 /case_iter/cluster）")
    max_cases_per_cluster: Optional[int] = Field(5, ge=1, le=20)


class CaseIterPrototypesData(BaseModel):
    prototypes: list


class CaseIterPrototypesResponse(BaseModel):
    ok: bool
    code: int
    data: Optional[CaseIterPrototypesData] = None
    error: Optional[str] = None


@app.post("/case_iter/prototypes", response_model=CaseIterPrototypesResponse)
def case_iter_prototypes(
    req: CaseIterPrototypesRequest,
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """对每个聚类抽取通用原型 prompt。"""
    try:
        api_key = _resolve_api_key(authorization, req.api_key)
        protos = extract_scene_prototypes(
            clusters=req.clusters,
            api_key=api_key,
            max_cases_per_cluster=req.max_cases_per_cluster,
        )
        return CaseIterPrototypesResponse(
            ok=True, code=200,
            data=CaseIterPrototypesData(prototypes=protos),
        )
    except Exception as e:
        return CaseIterPrototypesResponse(ok=False, code=500, error=str(e))


class CaseIterArchiveRequest(BaseModel):
    case_id: Optional[str] = Field(None)
    field: str
    query: str
    metrics: list
    initial_prompt: str
    best_prompt: str
    best_score: float
    iterations: int
    stop_reason: str
    extra: Optional[dict] = None
    write_to: Optional[str] = Field(None, description="可选落盘路径，例如 /data/archives/case1.json")


class CaseIterArchiveData(BaseModel):
    archive: dict
    written_to: Optional[str] = None


class CaseIterArchiveResponse(BaseModel):
    ok: bool
    code: int
    data: Optional[CaseIterArchiveData] = None
    error: Optional[str] = None


@app.post("/case_iter/archive", response_model=CaseIterArchiveResponse)
def case_iter_archive(req: CaseIterArchiveRequest):
    """把单次优化结果封装为可读档案，可选落盘。"""
    try:
        archive = archive_optimization(
            query=req.query,
            metrics=req.metrics,
            initial_prompt=req.initial_prompt,
            best_prompt=req.best_prompt,
            best_score=req.best_score,
            iterations=req.iterations,
            stop_reason=req.stop_reason,
            field=req.field,
            case_id=req.case_id,
            extra=req.extra,
        )
        written = None
        if req.write_to:
            written = write_archive_to_file(archive, req.write_to)
        return CaseIterArchiveResponse(
            ok=True, code=200,
            data=CaseIterArchiveData(archive=archive, written_to=written),
        )
    except Exception as e:
        return CaseIterArchiveResponse(ok=False, code=500, error=str(e))


# ────────────────────────────────────────────
# 启动入口
# ────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
