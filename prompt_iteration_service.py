# -*- coding: utf-8 -*-
"""
prompt_iteration_service.py
===========================
Service 层：对 ``prompt_iteration_optimizer.py`` core 做零 IO 封装，
供 FastAPI (``api_server.py``) 与本地脚本共同调用。

要点：
  - 维持单个全局 ``PromptIterationOptimizer`` 单例，避免每次请求重载 BGE 编码器
    与 ``chapter3_backup`` 景区聚类数据。
  - 提供 ``optimize_for_query`` 与 ``passthrough`` 两个无副作用的纯函数。
  - 不依赖 FastAPI/Flask 等任何 Web 框架，可单独 import 测试。
  - DashScope API Key **不在本模块硬编码**，要求调用方（HTTP 接口 /
    本地脚本）显式传入，避免上传到服务器后泄露本地 Key。
"""

import time
from typing import Any, Dict, Optional

from prompt_iteration_optimizer import (
    PromptIterationOptimizer,
    build_fusion_query,
    build_qa_system_prompt,
    _call_llm,
    DEFAULT_LLM_MODEL,
)


# ────────────────────────────────────────────
# 优化 prompt 生成（场景级）
# ────────────────────────────────────────────

# 用于"基于子查询/实体/聚类 PromptModule 生成最终优化后 prompt"的 LLM 提示词
_OPTIMIZE_PROMPT_SYSTEM = """你是一个资深的 RAG Prompt 工程专家。你的任务是基于用户原始问题、拆解出的子查询、提取出的实体术语，以及（可选的）命中的场景 PromptModule 模板，重写并生成一个更适合下游 RAG 检索-回答链路的"优化后 system prompt"。"""

_OPTIMIZE_PROMPT_TEMPLATE = """请根据以下信息，生成一个"优化后的 system prompt"，用于下游 RAG 问答。

【原始问题】
{query}

【拆解出的子查询】
{sub_queries}

【提取出的实体术语】
{entity_terms}

【命中聚类 PromptModule（若有）】
{module_info}

【输出要求】
1. 输出必须是一个完整的、可直接粘贴进 system prompt 的中文文本；
2. 应当显式地引导模型：聚焦上述实体、子查询的语义边界；
3. 若命中了 PromptModule 模板，请以它的角色/任务风格为骨架；
4. 不要解释、不要列标题、不要用代码块包裹，直接输出最终 prompt 正文。
"""


def build_optimized_prompt(
    expand_result: Dict[str, Any],
    api_key: str,
    base_prompt: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    timeout: float = 30.0,
) -> str:
    """
    场景级 prompt 迭代优化：基于 expand_result 中的子查询/实体/PromptModule，
    调 LLM 生成最终的"优化后 system prompt"。

    Params:
        expand_result: PromptIterationOptimizer.expand() 的输出
        api_key: DashScope API Key
        base_prompt: 兜底 system prompt（聚类未命中时使用）
        model: LLM 模型名（默认 qwen-plus）
        timeout: LLM 调用超时秒数

    Returns:
        优化后的 system prompt 文本（失败时回退到 base_prompt 或空串）
    """
    if not api_key or not api_key.strip():
        raise ValueError("api_key 必须由调用方显式传入，不允许空值或服务端兜底。")

    query = expand_result.get("original_query", "")
    sub_queries = expand_result.get("sub_queries", []) or []
    entity_terms = expand_result.get("entity_terms", []) or []
    pm = expand_result.get("prompt_module")

    # 命中聚类时把模板信息拼出来；未命中则提示 LLM 用 base_prompt 作为骨架
    if pm:
        module_info = (
            f"id={pm.get('id')}, name={pm.get('name')}, "
            f"template={pm.get('template', '')}"
        )
    else:
        module_info = (
            "（未命中聚类 PromptModule，请以 base_prompt 风格为基础）\n"
            f"base_prompt={base_prompt or ''}"
        )

    user_msg = _OPTIMIZE_PROMPT_TEMPLATE.format(
        query=query,
        sub_queries="\n".join(f"- {s}" for s in sub_queries) or "（无）",
        entity_terms="\n".join(f"- {e}" for e in entity_terms) or "（无）",
        module_info=module_info,
    )

    optimized = _call_llm(
        api_key=api_key,
        prompt=user_msg,
        system=_OPTIMIZE_PROMPT_SYSTEM,
        model=model,
        temperature=0.4,
        max_tokens=512,
        timeout=timeout,
    )

    # LLM 失败/返回空 → 退回到 PromptModule 模板或 base_prompt
    if not optimized.strip():
        if pm and pm.get("template"):
            return str(pm["template"])
        return base_prompt or ""
    return optimized.strip()


# ────────────────────────────────────────────
# 全局单例
# ────────────────────────────────────────────

_optimizer: Optional[PromptIterationOptimizer] = None


def get_optimizer() -> PromptIterationOptimizer:
    """
    全局 ``PromptIterationOptimizer`` 单例。
    首次访问时构造（懒加载 BGE 编码器与聚类数据）；
    后续访问直接返回，避免每个 HTTP 请求都重头初始化。
    """
    global _optimizer
    if _optimizer is None:
        _optimizer = PromptIterationOptimizer(
            use_llm_subqueries=True,
            use_entity_extraction=True,
            use_cluster_prompt=True,
            cluster_top_n=2,
        )
    return _optimizer


def reset_optimizer() -> None:
    """重置单例，主要用于测试。"""
    global _optimizer
    _optimizer = None


# ────────────────────────────────────────────
# 业务函数
# ────────────────────────────────────────────

def optimize_for_query(
    query: str,
    api_key: str,
    use_llm_subqueries: bool = True,
    use_entity_extraction: bool = True,
    use_cluster_prompt: bool = True,
    cluster_top_n: int = 2,
    base_prompt: Optional[str] = None,
    return_optimized_prompt: bool = True,
) -> Dict[str, Any]:
    """
    场景级 prompt 迭代优化。

    Params:
        query:               原始用户问题
        api_key:             DashScope API Key，**必传**，由调用方注入
        use_llm_subqueries:  是否调用 LLM 生成子查询
        use_entity_extraction: 是否调用 LLM 提取实体术语
        use_cluster_prompt:  是否做聚类 + PromptModule 选择
        cluster_top_n:       聚类匹配的 Top-N
        base_prompt:         兜底 system prompt（聚类未命中且 LLM 失败时使用）
        return_optimized_prompt: 是否额外用 LLM 生成"优化后的 system prompt"

    Returns:
        dict（含 ``fusion_query`` / ``optimized_prompt`` / ``total_time`` 等
              service 层追加字段，其余字段与 ``PromptIterationOptimizer.expand()``
              完全一致）
    """
    if not api_key or not api_key.strip():
        raise ValueError("api_key 必须由调用方显式传入，不允许空值或服务端兜底。")

    t0 = time.time()
    optimizer = get_optimizer()
    result = optimizer.expand(
        query,
        api_key=api_key,
    )
    result["fusion_query"] = build_fusion_query(result)
    if return_optimized_prompt:
        result["optimized_prompt"] = build_optimized_prompt(
            result,
            api_key=api_key,
            base_prompt=base_prompt,
        )
    else:
        result["optimized_prompt"] = None
    result["total_time"] = round(time.time() - t0, 2)
    return result


def passthrough(query: str, prompt: str, direction: str = "") -> Dict[str, Any]:
    """
    占位接口：原样透传 ``query`` / ``prompt`` / ``direction``，不做任何处理。

    协议：
        输入参数 → 输出参数原样回传，附加 ``processed=False`` 标识当前未做处理。

    预留扩展：将来若需要按 ``direction`` 调用 LLM 改写 prompt，
    在此处加分支即可，对外协议不变。
    """
    return {
        "query": query,
        "prompt": prompt,
        "direction": direction,
        "processed": False,
    }

