# -*- coding: utf-8 -*-
"""
case_level_service.py
=====================
案例级 prompt 迭代优化：
对用户提供的原始 prompt 进行定向改进，输出改进后的版本。

与场景级接口（/optimize）完全独立：
  - 场景级：输入 query → 分析 query 语义结构（子查询/实体/聚类）
  - 案例级：输入 query + 当前 prompt + 方向 → 定向改进 prompt

输入：
  - query:    原始问题
  - prompt:   当前使用的 system prompt
  - direction: 迭代方向（简洁/详细/专业化/友好/精确/全面）

输出：
  - refined_prompt:  改进后的 system prompt
  - direction:       实际使用的迭代方向

不依赖场景级结果，不调用 PromptIterationOptimizer。
"""

import time
from typing import Any, Dict, Optional

from prompt_iteration_optimizer import (
    _call_llm,
    DEFAULT_LLM_MODEL,
)


# ════════════════════════════════════════════════════════════════
# Prompt 模板
# ════════════════════════════════════════════════════════════════

_CASE_LEVEL_SYSTEM = (
    "你是一个专业的 RAG Prompt 工程师，擅长根据具体问题对 system prompt 做定向改进。"
)

_CASE_LEVEL_TEMPLATE_NO_SCENARIO = """你是一个 RAG Prompt 工程师，请对下面的 system prompt 进行定向改进。

【用户原始问题】
{query}

【当前使用的 system prompt】
---
{prompt}
---

【迭代方向】
{direction}

【输出要求】
1. 直接输出一段完整的、改进后的 system prompt 正文；
2. 严格按照迭代方向进行改进；
3. 不要解释、不要加标题、不要用代码块包裹；
4. 保持输出的 prompt 结构清晰、指令明确。
"""

_CASE_LEVEL_TEMPLATE_WITH_SCENARIO = """你是一个 RAG Prompt 工程师，请对下面的 system prompt 进行定向改进。

【用户原始问题】
{query}

【场景级分析结果】
- 子查询：{sub_queries}
- 实体术语：{entity_terms}
- 聚类命中：{cluster_name}
- PromptModule 模板：{module_info}

【当前使用的 system prompt】
---
{prompt}
---

【迭代方向】
{direction}

【输出要求】
1. 直接输出一段完整的、改进后的 system prompt 正文；
2. 严格遵循场景级分析的结构约束（子查询维度、实体边界、PromptModule 的角色风格）；
3. 严格按照迭代方向进行改进；
4. 不要解释、不要加标题、不要用代码块包裹；
5. 保持输出的 prompt 结构清晰、指令明确。
"""

# ════════════════════════════════════════════════════════════════
# 推理迭代方向 + Prompt 改进（基于标准答案）
# ════════════════════════════════════════════════════════════════

_INFER_DIRECTION_SYSTEM = (
    "你是一个 RAG Prompt 分析专家，擅长通过对比参考答案与检索上下文，找出当前 prompt 的不足方向。"
)

_INFER_DIRECTION_TEMPLATE = """你是一个 RAG Prompt 分析专家，请对比下面的【参考答案】与【当前 prompt 生成的回答】，判断当前 prompt 存在什么问题，应该朝哪个方向改进。

【用户问题】
{query}

【当前 prompt】
---
{prompt}
---

【参考答案】
---
{ground_truth}
---

【检索到的上下文（用于生成回答）】
---
{retrieved_context}
---

【输出要求】
1. 直接输出一个迭代方向关键词，不要解释、不要加标题、不要用代码块包裹；
2. 从以下六个方向中选择最合适的一个（简洁 / 详细 / 专业化 / 友好 / 精确 / 全面）；
3. 如果当前 prompt 已经很好，输出"精确"作为最小改动方向。
"""

_DIRECTION_GUIDANCE = {
    "简洁": "减少冗余描述，只保留核心指令，使 prompt 更短更精炼",
    "详细": "增加背景说明、约束条件和输出格式要求，使 prompt 更完整",
    "专业化": "使用更精确的领域术语，提升回答的专业性和准确性",
    "友好": "增加亲和力引导，使回答风格更亲切、更易理解",
    "精确": "增加具体约束，减少歧义，提升回答的精准度",
    "全面": "补充遗漏维度，覆盖更多相关知识点，避免信息缺失",
}


def infer_direction_and_refine(
    query: str,
    prompt: str,
    ground_truth: str,
    retrieved_context: str,
    api_key: str,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.3,
    timeout: float = 30.0,
    max_iterations: int = 3,
) -> Dict[str, Any]:
    """
    基于标准答案自动推断迭代方向并改进 prompt。

    流程：
    1. 将参考答案 + 检索上下文 + 当前 prompt 传给 LLM，让 LLM 对比推断出迭代方向
    2. 用推断出的方向 + query + prompt 调用 refine_prompt 进行多轮优化

    Params:
        query:              原始问题
        prompt:             当前使用的 system prompt
        ground_truth:       标准答案（用户提供的参考答案）
        retrieved_context:  检索到的上下文（来自 search 接口）
        api_key:            DashScope API Key，**必传**
        model:              LLM 模型名
        temperature:        采样温度
        timeout:            超时秒数
        max_iterations:     最大迭代次数，默认 3

    Returns:
        {
            "inferred_direction": str,   # LLM 推断出的迭代方向
            "refined_prompt":    str,    # 改进后的 prompt
            "iterations":        int,    # 实际迭代轮数
            "llm_time":          float,  # LLM 调用总耗时（秒）
            "total_time":        float,  # 总耗时（秒）
        }
    """
    if not api_key or not api_key.strip():
        raise ValueError("api_key 必须由调用方显式传入，不允许空值。")

    if not prompt or not prompt.strip():
        raise ValueError("current prompt 不能为空。")

    if not ground_truth or not ground_truth.strip():
        raise ValueError("ground_truth 不能为空。")

    t0 = time.time()

    # ── Step 1: 推断迭代方向 ──
    direction_t0 = time.time()
    direction_prompt = _INFER_DIRECTION_TEMPLATE.format(
        query=query,
        prompt=prompt,
        ground_truth=ground_truth,
        retrieved_context=retrieved_context,
    )
    inferred = _call_llm(
        api_key=api_key,
        prompt=direction_prompt,
        system=_INFER_DIRECTION_SYSTEM,
        model=model,
        temperature=temperature,
        max_tokens=32,
        timeout=int(timeout),
    )
    direction_elapsed = round(time.time() - direction_t0, 2)

    # 解析方向：取第一行非空，匹配六选一
    direction_raw = inferred.strip().split("\n")[0].strip()
    valid_directions = list(_DIRECTION_GUIDANCE.keys())
    inferred_direction = next(
        (d for d in valid_directions if d in direction_raw), "精确"
    )

    # ── Step 2: 用推断方向改进 prompt ──
    refine_result = refine_prompt(
        query=query,
        prompt=prompt,
        direction=inferred_direction,
        api_key=api_key,
        scenario_result=None,
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_iterations=max_iterations,
    )

    total_elapsed = round(time.time() - t0, 2)
    total_llm_time = round(direction_elapsed + refine_result["llm_time"], 2)

    return {
        "inferred_direction": inferred_direction,
        "refined_prompt": refine_result["refined_prompt"],
        "iterations": refine_result["iterations"],
        "llm_time": total_llm_time,
        "total_time": total_elapsed,
    }


def refine_prompt(
    query: str,
    prompt: str,
    direction: str,
    api_key: str,
    scenario_result: Optional[Dict[str, Any]] = None,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.3,
    timeout: float = 30.0,
    max_iterations: int = 3,
) -> Dict[str, Any]:
    """
    案例级 prompt 定向改进（支持多轮迭代）。

    基于场景级分析结果（子查询/实体/聚类/PromptModule），对用户提供的 prompt
    做定向改进。与场景级 /optimize 完全独立——调用方先调 /optimize 拿到
    scenario_result，再传入本接口做改进。

    Params:
        query:           原始问题
        prompt:          当前使用的 system prompt
        direction:       迭代方向（简洁/详细/专业化/友好/精确/全面），
                          也支持自定义描述文字
        api_key:         DashScope API Key，**必传**
        scenario_result:  场景级结果（来自 /optimize），可选
        model:           LLM 模型名
        temperature:     采样温度
        timeout:         超时秒数
        max_iterations:  最大迭代次数，默认 3

    Returns:
        {
            "refined_prompt": str,   # 改进后的 prompt
            "direction":      str,   # 实际使用的方向描述
            "scenario_used":  bool,  # 是否使用了场景级上下文
            "iterations":     int,   # 实际迭代轮数
            "llm_time":       float, # LLM 调用总耗时（秒）
            "total_time":     float, # 总耗时（秒）
        }
    """
    if not api_key or not api_key.strip():
        raise ValueError("api_key 必须由调用方显式传入，不允许空值。")

    if not prompt or not prompt.strip():
        raise ValueError("current prompt 不能为空。")

    t0 = time.time()
    direction_text = _DIRECTION_GUIDANCE.get(direction, direction)

    # ── 预计算场景上下文（不变部分） ──
    if scenario_result:
        cluster_info = scenario_result.get("prompt_module", {})
        cluster_name = ""
        if cluster_info:
            cluster_name = cluster_info.get("name", cluster_info.get("cluster_name", "未命名聚类"))
        else:
            cid = scenario_result.get("cluster_id")
            cluster_name = f"聚类 {cid}" if cid is not None else "未命中聚类"

        sub_queries = scenario_result.get("sub_queries", [])
        sub_queries_str = "、".join(sub_queries) if sub_queries else "（无）"

        entity_terms = scenario_result.get("entity_terms", [])
        entity_terms_str = "、".join(entity_terms) if entity_terms else "（无）"

        module_name = cluster_info.get("template", "") if cluster_info else "（无）"

        scenario_context = {
            "query": query,
            "sub_queries": sub_queries_str,
            "entity_terms": entity_terms_str,
            "cluster_name": cluster_name,
            "module_info": module_name,
        }
        base_template = _CASE_LEVEL_TEMPLATE_WITH_SCENARIO
    else:
        scenario_context = {"query": query}
        base_template = _CASE_LEVEL_TEMPLATE_NO_SCENARIO

    # ── 多轮迭代 ──
    current_prompt = prompt
    llm_t0 = time.time()
    for i in range(max_iterations):
        if scenario_result:
            user_prompt = base_template.format(
                **scenario_context,
                prompt=current_prompt,
                direction=f"[{direction}] {direction_text}",
            )
        else:
            user_prompt = base_template.format(
                prompt=current_prompt,
                direction=f"[{direction}] {direction_text}",
            )

        refined = _call_llm(
            api_key=api_key,
            prompt=user_prompt,
            system=_CASE_LEVEL_SYSTEM,
            model=model,
            temperature=temperature,
            max_tokens=1024,
            timeout=int(timeout),
        )
        if refined.strip():
            current_prompt = refined.strip()
        else:
            break

    llm_elapsed = round(time.time() - llm_t0, 2)
    total_elapsed = round(time.time() - t0, 2)

    return {
        "refined_prompt": current_prompt,
        "direction": direction,
        "scenario_used": scenario_result is not None,
        "iterations": max_iterations,
        "llm_time": llm_elapsed,
        "total_time": total_elapsed,
    }
