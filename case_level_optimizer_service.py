# -*- coding: utf-8 -*-
"""
case_level_optimizer_service.py
================================
案例级 Prompt 自动迭代优化（专利方案落实）。

依据专利文档《面向私域场景知识问答的大模型提示自动迭代优化方法及装置》，
把"案例级优化"由"LLM 单轮改写 prompt"扩展为完整的
"初始 prompt 生成 → 多维评估 → 加权打分 → 迭代收敛 → 原型 Prompt 抽取"五步闭环。

模块结构：

    ┌─────────────────────────────────────────────────────────────────────┐
    │  1. build_evaluation_metrics                                       │
    │        让 LLM 根据领域 + 场景样例生成 4~6 个评估指标（含权重）      │
    │        输出: List[{metric_name, description, weight}]              │
    ├─────────────────────────────────────────────────────────────────────┤
    │  2. build_initial_prompt                                           │
    │        用结构化模板（角色 / 内容 / 回答要求 / 不确定性表述）生成      │
    │        首版 system prompt                                           │
    ├─────────────────────────────────────────────────────────────────────┤
    │  3. judge_answer      LLM-as-a-Judge 多维打分                       │
    │      score_answer      加权综合得分 + 收敛判断                       │
    │      refine_with_eval  基于综合得分变化迭代优化 prompt               │
    ├─────────────────────────────────────────────────────────────────────┤
    │  4. cluster_case_samples   案例级样例聚类（复用 BGE-M3 + numpy）    │
    │      extract_scene_prototypes  抽取场景级原型 Prompt                │
    ├─────────────────────────────────────────────────────────────────────┤
    │  5. archive_optimization    提示优化档案（结构化回写）              │
    └─────────────────────────────────────────────────────────────────────┘

复用说明：
    - 直接 import prompt_iteration_optimizer 内的 ``_call_llm`` / ``DEFAULT_LLM_MODEL``
      / ``_encoder`` / ``_load_cluster_data``，避免重新实现 DashScope HTTP 调用与 BGE 加载。
    - 沿用 prompt_iteration_service.get_optimizer / build_optimized_prompt / build_fusion_query
      可选搭接场景级分析结果（scenario_result）。
    - 沿用 case_level_service._DIRECTION_GUIDANCE 与 _CASE_LEVEL_TEMPLATE_* 模板骨架，
      但本次落地以"客观综合得分"驱动收敛，方向描述仅作为 refine_with_eval 的可选参考。

设计要点：
    - 所有 LLM 输入/输出都用 JSON 规范，由 ``_safe_json_load`` 负责兜底解析。
    - 得分归一化到 0~100，所有指标 weight 之和为 1（鲁棒性：求和后归一）。
    - 收敛条件：综合得分 ≥ score_threshold 或 连续 score_window 轮改进 < min_delta。
    - 不引入新的全局 state，所有重资源（BGE、聚类）继续走 prompt_iteration_optimizer._encoder
      / ClusterPromptSelector 的懒加载机制。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ────────────────────────────────────────────
# 复用 prompt_iteration_optimizer 中的核心能力
# ────────────────────────────────────────────
from prompt_iteration_optimizer import (
    _call_llm,
    DEFAULT_LLM_MODEL,
    _encoder,                       # BGE 单例（懒加载）
    _load_cluster_data,             # 聚类中心 + PromptModule 加载
    _encode_query,                  # BGE 编码
    cosine_sim,                     # 余弦相似度
)
import prompt_iteration_optimizer as _pio  # 用作 `_pio._call_llm(...)` 的显式模块引用，规避闭包 globals 查找


# ════════════════════════════════════════════════════════════════
# 通用：JSON 解析兜底
# ════════════════════════════════════════════════════════════════

def _safe_json_load(text: str) -> Optional[Any]:
    """
    鲁棒的 JSON 解析：处理 LLM 返回中夹带的 ```json 代码块、中文标点、未闭合括号。
    返回解析结果；解析失败返回 None（不抛异常）。
    """
    if not text:
        return None
    text = text.strip()

    # 1) 直解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) 去掉 markdown 围栏
    fenced = text.strip()
    if fenced.startswith("```"):
        lines = fenced.split("\n")
        # 去掉首行 ```json / ``` 与末行 ```
        if len(lines) >= 2:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        fenced = "\n".join(lines).strip()
    try:
        return json.loads(fenced)
    except Exception:
        pass

    # 3) 截取最大平衡子串
    start = fenced.find("{")
    end = fenced.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(fenced[start:end + 1])
        except Exception:
            pass

    start = fenced.find("[")
    end = fenced.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(fenced[start:end + 1])
        except Exception:
            pass

    return None


def _normalize_weights(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    鲁棒归一化：保证所有 weight ∈ [0, 1]，合计为 1。
    若 LLM 给出的 weight 缺失 / 越界，自动归一化。
    """
    cleaned: List[Dict[str, Any]] = []
    for m in metrics or []:
        if not isinstance(m, dict):
            continue
        name = m.get("metric_name") or m.get("name") or m.get("metric") or ""
        if not str(name).strip():
            continue
        desc = m.get("description") or m.get("定义") or ""
        try:
            w = float(m.get("weight", 0.0))
        except Exception:
            w = 0.0
        w = max(0.0, min(1.0, w))
        cleaned.append({"metric_name": str(name).strip(), "description": str(desc).strip(), "weight": w})

    total = sum(c["weight"] for c in cleaned)
    if total <= 0:
        # 等分
        if cleaned:
            eq = 1.0 / len(cleaned)
            for c in cleaned:
                c["weight"] = round(eq, 4)
        return cleaned
    # 等比归一
    for c in cleaned:
        c["weight"] = round(c["weight"] / total, 4)
    # 修正尾部舍入
    diff = round(1.0 - sum(c["weight"] for c in cleaned), 4)
    if cleaned:
        cleaned[-1]["weight"] = round(cleaned[-1]["weight"] + diff, 4)
    return cleaned


# ════════════════════════════════════════════════════════════════
# Step 1：评估指标生成（领域自适应 + 权重分配）
# ════════════════════════════════════════════════════════════════

_METRICS_SYSTEM = (
    "你是一名资深的 RAG 评估指标专家，擅长为不同业务场景设计"
    "可量化、可执行的 prompt 优化评估指标，并合理分配权重。"
)

_METRICS_TEMPLATE = """你是一名 RAG 评估指标专家，请根据下面的领域与场景样例，定义一套用于
"评估当前 prompt 在该领域问答效果"的评估指标集合。

【领域】
{field}

【场景样例】
- 子查询列表：
{sub_queries}
- 实体术语：
{entity_terms}
- 命中聚类 PromptModule（若有）：
{module_info}

【输出要求】
1. 覆盖 4 个核心维度：回答准确性、回答完整性、回答相关性、不确定性表述。
   在此基础上可结合领域补充 1~2 个特色指标（如"时效性"、"游客友好度"、"用语专业度"等）。
2. 每个指标给出：
   - metric_name：      简短中文指标名（例如"回答准确性"）
   - description：      该指标评价什么（一句话）
   - weight：           权重，0~1，所有指标 weight 之和必须等于 1。
                        越重要的指标 weight 越大；最低不低于 0.05。
3. **不要输出指标以外的内容**，**不要解释、不要列标题**。
4. 必须用如下 JSON 格式输出：

```json
{{
  "evaluation_metrics": [
    {{"metric_name": "回答准确性", "description": "...", "weight": 0.30}},
    {{"metric_name": "回答完整性", "description": "...", "weight": 0.25}}
  ]
}}
```
"""


def build_evaluation_metrics(
    field: str,
    expand_result: Optional[Dict[str, Any]] = None,
    api_key: str = ...,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.3,
    timeout: int = 60,
    fallback_metrics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    调用 LLM 评估指标专家，根据领域生成多维评估指标 + 权重分配。

    Params:
        field:           业务领域（如"旅游景区"、"医疗问答"、"法律咨询"）
        expand_result:   场景级分析结果（来自 PromptIterationOptimizer.expand），
                          可选，提供后会让 LLM 更好地对齐子查询/实体/聚类 PromptModule 风格
        api_key:         DashScope API Key（必传，复用 _call_llm）
        model:           LLM 模型
        temperature:     采样温度
        timeout:         LLM 调用超时秒数
        fallback_metrics: LLM 解析失败时使用的兜底指标集

    Returns:
        {
            "field": str,
            "metrics": [{"metric_name","description","weight"}, ...] (weight 之和 = 1),
            "raw":    str,           # LLM 原始输出，便于排查
            "parse_ok": bool,        # JSON 是否解析成功
            "used_fallback": bool,   # 是否使用兜底指标
            "llm_time": float,
        }
    """
    # — 准备子查询 / 实体 / PromptModule 摘要 —
    if expand_result:
        sub_queries = expand_result.get("sub_queries") or []
        entity_terms = expand_result.get("entity_terms") or []
        pm = expand_result.get("prompt_module")
        if pm:
            module_info = (
                f"id={pm.get('id')}, name={pm.get('name')}, "
                f"template={(pm.get('template') or '')[:240]}"
            )
        else:
            module_info = "（未命中聚类 PromptModule）"
    else:
        sub_queries, entity_terms, module_info = [], [], "（无场景上下文）"

    user_prompt = _METRICS_TEMPLATE.format(
        field=field,
        sub_queries="\n".join(f"- {s}" for s in sub_queries) or "（无）",
        entity_terms="\n".join(f"- {e}" for e in entity_terms) or "（无）",
        module_info=module_info,
    )

    t0 = time.time()
    raw = _pio._call_llm(
        prompt=user_prompt,
        system=_METRICS_SYSTEM,
        temperature=temperature,
        max_tokens=512,
        model=model,
        api_key=api_key,
        timeout=timeout,
    )
    llm_time = round(time.time() - t0, 3)

    parsed = _safe_json_load(raw)
    parse_ok = False
    metrics_raw: List[Dict[str, Any]] = []
    if isinstance(parsed, dict):
        m_list = parsed.get("evaluation_metrics")
        if isinstance(m_list, list):
            metrics_raw = m_list
            parse_ok = True
    elif isinstance(parsed, list):
        metrics_raw = parsed
        parse_ok = True

    metrics = _normalize_weights(metrics_raw)
    if not metrics:
        # 兜底：使用传入的或通用默认值
        metrics = _normalize_weights(fallback_metrics or [
            {"metric_name": "回答准确性", "description": "回答的事实与参考答案是否一致", "weight": 0.30},
            {"metric_name": "回答完整性", "description": "回答是否覆盖了参考答案的所有关键信息", "weight": 0.25},
            {"metric_name": "回答相关性", "description": "回答是否紧扣用户问题", "weight": 0.20},
            {"metric_name": "不确定性表述", "description": "证据不足时是否明确说明而不是凭空捏造", "weight": 0.15},
            {"metric_name": "格式规范性", "description": "输出格式、风格是否符合系统角色定义", "weight": 0.10},
        ])
        used_fallback = True
    else:
        used_fallback = False

    return {
        "field": field,
        "metrics": metrics,
        "raw": raw,
        "parse_ok": parse_ok,
        "used_fallback": used_fallback,
        "llm_time": llm_time,
    }


# ════════════════════════════════════════════════════════════════
# Step 2：初始 Prompt 生成（结构化模板）
# ════════════════════════════════════════════════════════════════

_INITIAL_PROMPT_SYSTEM = (
    "你是一名资深的 Prompt Engineer，擅长根据行业领域与初始输入生成"
    "可直接用于 RAG 问答系统的、结构清晰、约束明确、容错完备的 system prompt。"
)

_INITIAL_PROMPT_TEMPLATE = """请你作为资深 Prompt Engineer，根据下面的领域与场景样例，生成一份"初始 system prompt"。

【领域】
{field}

【场景样例】
- 子查询列表：
{sub_queries}
- 实体术语：
{entity_terms}
- 命中聚类 PromptModule（若有）：
{module_info}

【当前用户输入（用于对齐意图）】
{query}

【输出要求】
1. 直接输出一段完整的 system prompt 正文，**不要解释、不要标题、不要代码块包裹**。
2. 必须显式包含以下模块：

   （1）角色定位：明确规定角色是哪一领域的专业问答专家（如"资深景区导游"、"专业医生助理"等）。
   （2）回答要求（必须覆盖以下 5 点）：
         - 内容忠实：严格依据<私有上下文>回答，严禁编造私域不存在的事实、票种、时间、路线、口碑、规范等。
         - 答案范围：紧扣用户问题，剔除无关信息。
         - 证据使用：多片段时综合取舍，剔除无关 / 重复片段。
         - 格式：按 {output_format} 输出，结构清晰。
         - 不确定性表述：信息缺失或相互冲突时，**必须明确说明"依据当前信息无法确定"**，给出可执行建议而非凭空编造。
   （3）可选附加：补充与领域相关的禁止项、注意点。
3. prompt 总体长度 250~600 字；语气专业克制；中文输出。
"""


def build_initial_prompt(
    field: str,
    query: str,
    expand_result: Optional[Dict[str, Any]] = None,
    api_key: str = ...,
    output_format: str = "Markdown 段落 + 必要时使用列表",
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.3,
    timeout: int = 60,
    base_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    生成结构化初始 prompt。

    Params: （同上）
        field / query: 领域与用户问题
        expand_result: 场景级分析结果（可选，强烈建议传入）
        output_format: 输出格式描述，默认 Markdown
        base_prompt:   兜底 prompt；当 LLM 解析失败时直接返回 base_prompt

    Returns:
        {
            "initial_prompt": str,
            "raw": str,
            "used_base": bool,
            "field": str,
            "llm_time": float,
        }
    """
    if expand_result:
        sub_queries = expand_result.get("sub_queries") or []
        entity_terms = expand_result.get("entity_terms") or []
        pm = expand_result.get("prompt_module")
        if pm:
            module_info = (
                f"id={pm.get('id')}, name={pm.get('name')}, "
                f"template={(pm.get('template') or '')[:240]}"
            )
        else:
            module_info = "（未命中聚类 PromptModule）"
    else:
        sub_queries, entity_terms, module_info = [], [], "（无场景上下文）"

    user_prompt = _INITIAL_PROMPT_TEMPLATE.format(
        field=field,
        query=query,
        sub_queries="\n".join(f"- {s}" for s in sub_queries) or "（无）",
        entity_terms="\n".join(f"- {e}" for e in entity_terms) or "（无）",
        module_info=module_info,
        output_format=output_format,
    )

    t0 = time.time()
    raw = _pio._call_llm(
        prompt=user_prompt,
        system=_INITIAL_PROMPT_SYSTEM,
        temperature=temperature,
        max_tokens=768,
        model=model,
        api_key=api_key,
        timeout=timeout,
    )
    llm_time = round(time.time() - t0, 3)

    initial = raw.strip() if raw else ""
    used_base = False
    if not initial:
        initial = base_prompt or ""
        used_base = True

    return {
        "initial_prompt": initial,
        "raw": raw,
        "used_base": used_base,
        "field": field,
        "llm_time": llm_time,
    }


# ════════════════════════════════════════════════════════════════
# Step 3：LLM-as-a-Judge 多维打分（基于参考答案与私域上下文）
# ════════════════════════════════════════════════════════════════

_JUDGE_SYSTEM = (
    "你是一名严格、客观的 RAG 质量评估专家。给定一份评估指标集，请严格按维度对"
    "模型生成的答案打分，并简洁地说明理由。"
)

_JUDGE_TEMPLATE = """请根据下面的评估指标集，对【模型生成的答案】进行多维打分。

【用户问题】
{query}

【私域检索到的上下文（用于回答）】
---
{retrieved_context}
---

【参考答案】
---
{ground_truth}
---

【模型生成的答案（待评分）】
---
{candidate_answer}
---

【评估指标集（每项 0~100 分）】
{metrics_block}

【输出要求】
1. 严格按上表逐个指标打分，分值 0~100，整数。
2. 每个指标必须给出简短理由（不超过 30 字）。
3. 最终在末尾输出一行 JSON：

```json
{{
  "scores": [
    {{"metric_name": "指标1", "score": 87, "reason": "覆盖关键事实，无编造"}},
    {{"metric_name": "指标2", "score": 75, "reason": "..."}}
  ]
}}
```

4. 只输出 JSON，不要额外文字；JSON 必须是合法格式。
"""


def judge_answer(
    query: str,
    candidate_answer: str,
    ground_truth: str,
    retrieved_context: str,
    metrics: List[Dict[str, Any]],
    api_key: str,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.2,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    LLM-as-a-Judge：对单条候选答案做多维打分。

    Params:
        query, candidate_answer, ground_truth, retrieved_context: 标准四元组
        metrics:   build_evaluation_metrics() 的输出（list[dict]）
        api_key / model / temperature / timeout: LLM 调用参数

    Returns:
        {
            "scores": [{"metric_name","score","reason"}, ...],
            "metrics": metrics,            # 透传
            "raw":     str,
            "parse_ok":bool,
            "llm_time":float,
        }
    """
    if not metrics:
        raise ValueError("metrics 不能为空，请先调用 build_evaluation_metrics。")

    metrics_block_lines = []
    for i, m in enumerate(metrics, 1):
        metrics_block_lines.append(
            f"{i}. {m['metric_name']}（weight={m['weight']:.2f}）: {m['description']}"
        )
    metrics_block = "\n".join(metrics_block_lines)

    user_prompt = _JUDGE_TEMPLATE.format(
        query=query,
        candidate_answer=candidate_answer,
        ground_truth=ground_truth,
        retrieved_context=retrieved_context,
        metrics_block=metrics_block,
    )

    t0 = time.time()
    raw = _pio._call_llm(
        prompt=user_prompt,
        system=_JUDGE_SYSTEM,
        temperature=temperature,
        max_tokens=768,
        model=model,
        api_key=api_key,
        timeout=timeout,
    )
    llm_time = round(time.time() - t0, 3)

    parsed = _safe_json_load(raw)
    scores_out: List[Dict[str, Any]] = []
    parse_ok = False
    if isinstance(parsed, dict):
        sl = parsed.get("scores")
        if isinstance(sl, list):
            for s in sl:
                if not isinstance(s, dict):
                    continue
                try:
                    sv = float(s.get("score", 0))
                except Exception:
                    sv = 0.0
                sv = max(0.0, min(100.0, sv))
                scores_out.append({
                    "metric_name": str(s.get("metric_name", "")).strip(),
                    "score": round(sv, 2),
                    "reason": str(s.get("reason", "")).strip(),
                })
            parse_ok = bool(scores_out)

    # 兜底：若按名字没匹配上的指标补 0 分（不阻断后续加权）
    if not scores_out:
        for m in metrics:
            scores_out.append({"metric_name": m["metric_name"], "score": 0.0, "reason": "judge 解析失败"})

    return {
        "scores": scores_out,
        "metrics": metrics,
        "raw": raw,
        "parse_ok": parse_ok,
        "llm_time": llm_time,
    }


def score_answer(judge_result: Dict[str, Any], metrics: Optional[List[Dict[str, Any]]] = None) -> float:
    """
    根据 judge 的各维分数与 weight 计算加权综合得分（0~100）。

    Params:
        judge_result: judge_answer() 的输出
        metrics:      评估指标定义（若不传则取 judge_result["metrics"]）

    Returns:
        综合得分（float, 0~100）
    """
    scores = judge_result.get("scores") or []
    metrics = metrics or judge_result.get("metrics") or []

    # 名称 → weight 映射
    weight_map = {m["metric_name"]: float(m["weight"]) for m in metrics}
    # 名称 → score（如果一个指标被打多次，取平均）
    score_map: Dict[str, List[float]] = {}
    for s in scores:
        name = s.get("metric_name")
        if name is None:
            continue
        score_map.setdefault(name, []).append(float(s.get("score", 0.0)))

    total = 0.0
    weight_sum_used = 0.0
    used_names: List[str] = []
    for name, w in weight_map.items():
        sc_list = score_map.get(name)
        if not sc_list:
            continue
        avg = sum(sc_list) / len(sc_list)
        total += avg * w
        weight_sum_used += w
        used_names.append(name)

    # 若部分维度缺分，按已打分维度归一化（避免总分被稀释）
    if weight_sum_used > 0:
        # 直接使用加权和（不二次放大，保守且直观）
        return round(total / max(weight_sum_used, 1e-9), 2)

    # 所有维度都没打分 → 退到简单均值
    flat = [s.get("score", 0.0) for s in scores]
    if flat:
        return round(sum(flat) / len(flat), 2)
    return 0.0


# ════════════════════════════════════════════════════════════════
# Step 4：用 scorer 反馈迭代优化 prompt（Algorithm 1）
# ════════════════════════════════════════════════════════════════

_REFINE_SYSTEM = (
    "你是 RAG Prompt 工程专家。给定当前 prompt 的多维评估反馈，请改写出"
    "在低分维度上更合规、在高分维度上保持水准的新 prompt。"
)

_REFINE_TEMPLATE = """请根据【当前 prompt 的多维评估反馈】，对 prompt 进行一次定向改写。

【用户问题】
{query}

【私域上下文（节选）】
---
{retrieved_context}
---

【参考答案】
---
{ground_truth}
---

【当前 Prompt】
---
{current_prompt}
---

【当前 Prompt 的多维评估反馈】
{feedback_block}

【改写要求】
1. 必须保证所有指标不低于上一轮得分：即改写后的 prompt 在每一个评估维度上，
   都必须达到或超过【当前 Prompt 的多维评估反馈】中的得分（不得以牺牲任一维度为代价换取综合分提升）。
   具体来说：
   - 若某一维度得分 < 85，**必须**在新 prompt 中显式补强相应约束；
   - 若某一维度得分 ≥ 90，保持其约束不变；
   - **严禁**为提升某一维度而引入新内容导致其他维度回退。
2. 严禁新增与当前评估问题无关的冗余内容。
3. 输出**新 prompt 正文**，保持中文，结构化（角色 / 内容 / 回答要求 / 不确定性）。
4. 不要解释、不要代码块、不要标题。
"""


def _format_feedback(metrics: List[Dict[str, Any]], judge: Dict[str, Any]) -> str:
    """把 judge 的反馈格式化为 prompt 友好的 Markdown 文本。"""
    score_map = {s.get("metric_name"): s for s in judge.get("scores", [])}
    lines = []
    for i, m in enumerate(metrics, 1):
        s = score_map.get(m["metric_name"]) or {}
        score = s.get("score", 0)
        reason = s.get("reason", "无评语")
        lines.append(
            f"{i}. **{m['metric_name']}**（weight={m['weight']:.2f}）: "
            f"得分={score:.1f} — {reason}"
        )
    return "\n".join(lines)


def refine_with_eval(
    query: str,
    current_prompt: str,
    candidate_answer: str,
    ground_truth: str,
    retrieved_context: str,
    metrics: List[Dict[str, Any]],
    judge_result: Dict[str, Any],
    api_key: str,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.4,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    基于综合评估反馈，对 prompt 做一次性定向改写（Algorithm 1 第 9~10 步）。

    Returns:
        {
            "new_prompt":  str,
            "raw":         str,
            "llm_time":    float,
            "used_fallback": bool,
        }
    """
    feedback_block = _format_feedback(metrics, judge_result)

    user_prompt = _REFINE_TEMPLATE.format(
        query=query,
        retrieved_context=retrieved_context[:2400],   # 节选避免超长
        ground_truth=ground_truth[:1200],
        current_prompt=current_prompt,
        feedback_block=feedback_block,
    )

    t0 = time.time()
    raw = _pio._call_llm(
        prompt=user_prompt,
        system=_REFINE_SYSTEM,
        temperature=temperature,
        max_tokens=1024,
        model=model,
        api_key=api_key,
        timeout=timeout,
    )
    llm_time = round(time.time() - t0, 3)

    new_prompt = raw.strip() if raw else ""
    used_fallback = False
    if not new_prompt:
        new_prompt = current_prompt
        used_fallback = True

    return {
        "new_prompt": new_prompt,
        "raw": raw,
        "llm_time": llm_time,
        "used_fallback": used_fallback,
    }


# ════════════════════════════════════════════════════════════════
# Step 4 Orchestration：Algorithm 1 闭环迭代
# ════════════════════════════════════════════════════════════════

def iterate_prompt_until_converged(
    query: str,
    initial_prompt: str,
    ground_truth: str,
    retrieved_context: str,
    metrics: List[Dict[str, Any]],
    generate_answer_fn,
    api_key: str,
    model: str = DEFAULT_LLM_MODEL,
    judge_temperature: float = 0.2,
    refine_temperature: float = 0.4,
    timeout: int = 60,
    max_iterations: int = 5,
    score_threshold: float = 85.0,
    score_window: int = 2,
    min_delta: float = 0.5,
    monotonic_max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Algorithm 1: 迭代优化 prompt 直到收敛（含每个指标的严格单调性约束）。

    **单调性约束**：每次改写后，候选 prompt 的每一项指标得分都必须不低于上一轮
    对应指标的得分；任何一项指标回退都会触发自动重试（最多 monotonic_max_retries 次），
    仍无法满足则放弃该次改写，保留当前 prompt。

    Params:
        query / initial_prompt / ground_truth / retrieved_context: 标准四元组
        metrics:           评估指标定义（必须由 build_evaluation_metrics 提供）
        generate_answer_fn:可调用 ``fn(prompt:str) -> str``；返回模型对当前 prompt 的答复
        api_key / model:   DashScope 配置
        judge_temperature: judge 打分温度
        refine_temperature:改写 prompt 温度
        max_iterations:    最大迭代轮数（防止长尾）
        score_threshold:   综合得分达标线，达到即停
        score_window:      连续若干轮改进 < min_delta 即视作收敛
        min_delta:         收敛灵敏阈值
        monotonic_max_retries: 每次改写时单调性验证的最大重试次数
        timeout:           单次 LLM 调用超时秒数

    Returns:
        {
            "best_prompt":    str,
            "best_score":     float,
            "iterations":     int,
            "history":        [
                {"iter":0,"prompt":..,"answer":..,"judge":..,"score":..,
                 "per_metric_scores":{...},"delta":..,"stop_reason":..}, ...
            ],
            "stop_reason":    str,
            "total_llm_time": float,
            "total_time":     float,
        }
    """
    if not metrics:
        raise ValueError("metrics 不能为空，请先调用 build_evaluation_metrics。")
    if not callable(generate_answer_fn):
        raise ValueError("generate_answer_fn 必须是可调用对象。")
    if max_iterations < 1:
        raise ValueError("max_iterations 必须 >= 1。")

    # 指标名字集合（用于快速查找）
    metric_names = [m["metric_name"] for m in metrics]

    # 上一轮各指标得分（初始为 None，表示"无上一轮"）
    prev_metric_scores: Optional[Dict[str, float]] = None

    t_begin = time.time()
    current_prompt = initial_prompt
    history: List[Dict[str, Any]] = []
    best_prompt = initial_prompt
    best_score = -1.0
    stop_reason = "unknown"
    consecutive_no_gain = 0
    total_llm_time = 0.0

    for i in range(max_iterations):
        # 1) 让模型用当前 prompt 生成答案（外部模型 / 内部 LLM）
        ans_t0 = time.time()
        try:
            candidate = generate_answer_fn(current_prompt)
        except Exception as e:
            candidate = ""
            history.append({
                "iter": i,
                "prompt": current_prompt,
                "answer": "",
                "error": f"generate_answer_fn 异常: {e}",
                "score": best_score,
                "per_metric_scores": {},
                "delta": 0.0,
                "stop_reason": "generate_error",
            })
            stop_reason = "generate_error"
            break
        ans_time = time.time() - ans_t0

        # 2) 多维打分
        judge = judge_answer(
            query=query,
            candidate_answer=candidate,
            ground_truth=ground_truth,
            retrieved_context=retrieved_context,
            metrics=metrics,
            api_key=api_key,
            model=model,
            temperature=judge_temperature,
            timeout=timeout,
        )
        total_llm_time += judge["llm_time"]
        score = score_answer(judge, metrics)

        # 提取当前各指标分数（key: metric_name, value: float）
        current_metric_scores = {}
        for s in (judge.get("scores") or []):
            name = s.get("metric_name", "")
            if name in metric_names:
                current_metric_scores[name] = float(s.get("score", 0.0))

        delta = score - best_score if best_score >= 0 else 0.0

        history.append({
            "iter": i,
            "prompt": current_prompt,
            "answer": candidate,
            "judge": judge,
            "score": score,
            "per_metric_scores": current_metric_scores,
            "delta": delta,
            "answer_time": round(ans_time, 3),
            "judge_time": judge["llm_time"],
        })

        # 更新 best
        if score > best_score:
            best_score = score
            best_prompt = current_prompt

        # 3) 收敛条件
        if score >= score_threshold:
            stop_reason = f"reached_threshold({score:.2f}>={score_threshold})"
            break
        if delta < min_delta:
            consecutive_no_gain += 1
        else:
            consecutive_no_gain = 0
        if consecutive_no_gain >= score_window:
            stop_reason = f"converged(no_gain_for_{score_window}_iters)"
            break

        # 4) ── 单调性约束改写 prompt ──────────────────────────────────────────
        #
        #   规则：候选 prompt 的每一项指标得分都必须 >= 上一轮对应指标的得分。
        #         任一指标回退 → 触发 refine 重试（最多 monotonic_max_retries 次）。
        #         仍无法满足 → 放弃本次改写，current_prompt 保持不变，迭代终止。
        #
        prev_for_check = prev_metric_scores  # None 表示第一轮，无需单调性检查

        accepted_prompt = None
        accepted_metric_scores = None
        accepted_llm_time = 0.0

        for retry_round in range(monotonic_max_retries):
            refine = refine_with_eval(
                query=query,
                current_prompt=current_prompt,
                candidate_answer=candidate,
                ground_truth=ground_truth,
                retrieved_context=retrieved_context,
                metrics=metrics,
                judge_result=judge,
                api_key=api_key,
                model=model,
                temperature=refine_temperature,
                timeout=timeout,
            )
            total_llm_time += refine["llm_time"]
            retry_llm_time = refine["llm_time"]

            new_prompt = refine["new_prompt"]
            if not new_prompt or new_prompt.strip() == current_prompt.strip():
                # LLM 未产生新内容，直接停止
                stop_reason = "refine_no_change"
                break

            # 用新 prompt 重新生成答案并打分
            try:
                candidate_retry = generate_answer_fn(new_prompt)
            except Exception:
                candidate_retry = ""

            judge_retry = judge_answer(
                query=query,
                candidate_answer=candidate_retry,
                ground_truth=ground_truth,
                retrieved_context=retrieved_context,
                metrics=metrics,
                api_key=api_key,
                model=model,
                temperature=judge_temperature,
                timeout=timeout,
            )
            total_llm_time += judge_retry["llm_time"]
            retry_llm_time += judge_retry["llm_time"]

            retry_metric_scores: Dict[str, float] = {}
            for s in (judge_retry.get("scores") or []):
                name = s.get("metric_name", "")
                if name in metric_names:
                    retry_metric_scores[name] = float(s.get("score", 0.0))

            # ── 单调性验证 ────────────────────────────────────────────────────
            monotonic_ok = True
            degraded_metrics: List[str] = []
            if prev_for_check is not None:
                for mname in metric_names:
                    prev_s = prev_for_check.get(mname, -1.0)
                    curr_s = retry_metric_scores.get(mname, 0.0)
                    if curr_s < prev_s - 0.01:   # 允许 ±0.01 的浮点容差
                        monotonic_ok = False
                        degraded_metrics.append(
                            f"{mname}(上轮{prev_s:.1f}→本轮{curr_s:.1f})"
                        )

            if monotonic_ok:
                accepted_prompt = new_prompt
                accepted_metric_scores = retry_metric_scores
                accepted_llm_time = retry_llm_time
                break
            else:
                # 有指标回退：记录警告，继续重试
                history.append({
                    "iter": i,
                    "prompt": new_prompt,
                    "answer": candidate_retry,
                    "judge": judge_retry,
                    "score": score_answer(judge_retry, metrics),
                    "per_metric_scores": retry_metric_scores,
                    "delta": 0.0,
                    "answer_time": 0.0,
                    "judge_time": judge_retry["llm_time"],
                    "retry_round": retry_round + 1,
                    "monotonic_rejected": True,
                    "degraded_metrics": degraded_metrics,
                    "stop_reason": f"metric_degraded({','.join(degraded_metrics)})",
                })
        # ── end monotonic retry loop ────────────────────────────────────────

        if accepted_prompt is not None:
            current_prompt = accepted_prompt
            prev_metric_scores = accepted_metric_scores
            total_llm_time += accepted_llm_time
        else:
            # 所有重试都未满足单调性，放弃改写，迭代终止
            stop_reason = f"monotonic_constraint_failed(all_retries_degraded)"
            break

    else:
        stop_reason = f"max_iterations({max_iterations})"

    return {
        "best_prompt": best_prompt,
        "best_score": round(best_score if best_score >= 0 else 0.0, 2),
        "iterations": len(history),
        "history": history,
        "stop_reason": stop_reason,
        "total_llm_time": round(total_llm_time, 3),
        "total_time": round(time.time() - t_begin, 3),
    }


# ════════════════════════════════════════════════════════════════
# Step 5：样例聚类 + 原型 Prompt 抽取（复用 BGE）
# ════════════════════════════════════════════════════════════════

_CLUSTER_SYSTEM = (
    "你是一名 RAG Prompt 专家。给定聚类后的问答样例与各案例迭代得到的优化 prompt，"
    "请提炼出该聚类下通用的「原型 prompt」，供下游 RAG 直接复用。"
)

_CLUSTER_PROTOTYPE_TEMPLATE = """请根据聚类下的问答样例与各案例迭代得到的优化 prompt，提炼出一个"通用原型 prompt"。

【聚类 ID】{cluster_id}
【聚类代表词】{cluster_terms}

【聚类下的问答样例（最多 N 条）】
{cases_block}

【对应的各案例优化后的 prompt（最多 N 条）】
{prompts_block}

【提炼要求（请严格保证所有要素同时具备）】
1. 写作目标（answered_goal）：    提炼该场景常见的回答目标。
2. 证据选取规则（evidence_rules）：总结私域上下文里应该用 / 不应该用的内容。
3. 答案组织结构（answer_layout）：总结合适的输出结构（如结论先行 / 分段 / 列表 / 反问 / 建议等）。
4. 格式约束（format_rules）：      总结输出格式、长度、表述方式。
5. 禁止项（prohibited_items）：   总结必须规避的行为（编造 / 越界 / 答非所问）。

【输出要求】
1. 直接输出一段**通用原型 prompt 正文**，可粘贴到 system prompt 槽位使用。
2. 必须显式覆盖：角色定义 / 回答目标 / 证据使用规则 / 输出格式 / 不确定性表述。
3. **不要解释、不要标题、不要代码块**，总计 250~600 字，中文输出。
"""


def cluster_case_samples(
    samples: List[Dict[str, Any]],
    encoder=None,
    n_clusters: Optional[int] = None,
) -> Dict[str, Any]:
    """
    对案例集合（每条含 question/optimized_prompt 等）做 K-means 聚类。

    Params:
        samples: List[{
                    "question": str,
                    "optimized_prompt": str,
                    "answer": Optional[str],   # 参考答案（可选）
                }, ...]
        encoder: BGE 编码器（默认用 _encoder 单例，复用 prompt_iteration_optimizer）
        n_clusters: 聚类数；为 None 时按 sqrt(N/2) 启发式取值（最小 2，最大 8）。

    Returns:
        {
            "n_samples": int,
            "n_clusters": int,
            "clusters": [
                {
                    "cluster_id": int,
                    "size": int,
                    "sample_indices": [...],   # 原 samples 索引
                    "representative_terms": [str, ...],
                    "representative_question": str,
                    "questions": [str, ...],
                    "prompts":    [str, ...],
                }, ...
            ],
            "encoders_used": bool,   # BGE 是否真的可用（不可用时用 TF-IDF fallback）
        }
    """
    if not samples:
        return {"n_samples": 0, "n_clusters": 0, "clusters": [], "encoders_used": False}

    # 优先复用 BGE；不可用时退化到 TF-IDF + 简单 k-means
    use_real_encoder = False
    enc = encoder or _encoder
    if enc is None:
        from prompt_iteration_optimizer import _load_bge_encoder  # 触发懒加载
        enc = _load_bge_encoder()
    if enc is not None:
        use_real_encoder = True

    # 编码
    if use_real_encoder:
        embs = []
        for s in samples:
            q = s.get("question") or s.get("query") or ""
            if not q.strip():
                embs.append(None)
                continue
            v = _encode_query(enc, q)
            embs.append(np.array(v, dtype=np.float32) if v is not None else None)
        # 丢掉空缺的
        valid_idx = [i for i, e in enumerate(embs) if e is not None]
        if not valid_idx:
            use_real_encoder = False
        else:
            X = np.stack([embs[i] for i in valid_idx], axis=0)
    if not use_real_encoder:
        # TF-IDF 回退（零额外依赖，使用 sklearn）
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(max_features=4096)
            X = vec.fit_transform([(s.get("question") or "") for s in samples]).toarray()
            valid_idx = list(range(len(samples)))
        except Exception:
            # sklearn 缺失时把所有样本归到 1 类
            return {
                "n_samples": len(samples),
                "n_clusters": 1,
                "clusters": [{
                    "cluster_id": 0,
                    "size": len(samples),
                    "sample_indices": list(range(len(samples))),
                    "representative_terms": [],
                    "representative_question": samples[0].get("question", ""),
                    "questions": [s.get("question", "") for s in samples],
                    "prompts":  [s.get("optimized_prompt", "") for s in samples],
                }],
                "encoders_used": False,
                "fallback": "no_sklearn",
            }

    # 决定聚类数
    n = len(valid_idx)
    if n_clusters is None or n_clusters < 1:
        n_clusters = max(2, min(8, int(round(np.sqrt(n / 2.0)) or 2)))
    n_clusters = min(n_clusters, n)

    # KMeans（优先 sklearn；缺失时用 numpy 手写）
    centroids: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        km.fit(X)
        labels = km.labels_
        centroids = km.cluster_centers_
    except Exception:
        # 简易随机+最近邻迭代实现
        rng = np.random.default_rng(42)
        # 余弦归一
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms
        init_idx = rng.choice(n, size=n_clusters, replace=False)
        centroids = Xn[init_idx].copy()
        for _ in range(20):
            sims = Xn @ centroids.T           # (n, k)
            labels = np.argmax(sims, axis=1)
            for k in range(n_clusters):
                mask = (labels == k)
                if mask.sum() == 0:
                    continue
                centroids[k] = Xn[mask].mean(axis=0)
                nrm = np.linalg.norm(centroids[k])
                if nrm > 0:
                    centroids[k] = centroids[k] / nrm

    # 聚合结果
    clusters: List[Dict[str, Any]] = []
    for k in range(n_clusters):
        idxs = [valid_idx[j] for j, lab in enumerate(labels) if lab == k]
        if not idxs:
            continue
        cluster_qs = [samples[i].get("question", "") for i in idxs]
        cluster_pms = [samples[i].get("optimized_prompt", "") for i in idxs]

        # 选样例中心最近的样本作为代表 question
        sub_X = X[[valid_idx.index(i) for i in idxs]]
        if centroids is not None:
            ck = centroids[k]
            sims = sub_X @ ck / (
                np.linalg.norm(sub_X, axis=1, keepdims=True) * (np.linalg.norm(ck) + 1e-9)
            )
            rep_local = int(np.argmax(sims))
        else:
            rep_local = 0
        representative_question = cluster_qs[rep_local]

        # 简单统计代表词：取聚类内出现频次较高的非停用词
        rep_terms = _top_terms(sub_X, cluster_qs, top_k=8)

        clusters.append({
            "cluster_id": k,
            "size": len(idxs),
            "sample_indices": idxs,
            "representative_terms": rep_terms,
            "representative_question": representative_question,
            "questions": cluster_qs,
            "prompts":   cluster_pms,
        })

    clusters.sort(key=lambda c: c["size"], reverse=True)
    # 重新编号 cluster_id 保持连续
    for new_id, c in enumerate(clusters):
        c["cluster_id"] = new_id

    return {
        "n_samples": len(samples),
        "n_clusters": len(clusters),
        "clusters": clusters,
        "encoders_used": use_real_encoder,
    }


# 简易停用词（含中英文常见停用词）
_STOPWORDS = set("""
的 了 和 是 在 我 你 他 她 它 们 这 那 与 及 或 也 但 而 就 都 不 没 没有 还 要 会 可以 能 请 问
a an the and or but of in on at to for with by as is are was were be been being it its this that these those
""".split())


def _top_terms(X, questions, top_k: int = 8) -> List[str]:
    """基于词频的简易代表词抽取（不依赖外部库）。"""
    from collections import Counter
    counter: Counter = Counter()
    for q in questions:
        for tok in (q or "").replace("\n", " ").split():
            t = tok.strip(" ,。、?!？!,.;:：；“”\"'()【】[]《》<> ")
            if not t or t.lower() in _STOPWORDS or len(t) < 2:
                continue
            if all('\u4e00' <= ch <= '\u9fff' for ch in t) and len(t) == 1:
                continue
            counter[t] += 1
    return [w for w, _ in counter.most_common(top_k)]


def extract_scene_prototypes(
    clusters: List[Dict[str, Any]],
    api_key: str,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.3,
    timeout: int = 60,
    max_cases_per_cluster: int = 5,
) -> List[Dict[str, Any]]:
    """
    对每个聚类，先取代表样本，再用 LLM 提炼"通用原型 prompt"。

    Params:
        clusters: cluster_case_samples() 的输出 clusters 字段
        api_key / model / temperature / timeout: LLM 参数
        max_cases_per_cluster: 每个聚类最多喂给 LLM 的样例条数

    Returns:
        [
            {
                "cluster_id": int,
                "representative_terms": [...],
                "representative_question": "...",
                "prototype_prompt": "...",
                "raw": "...",
                "llm_time": float,
                "used_fallback": bool,
            }, ...
        ]
    """
    out: List[Dict[str, Any]] = []
    for c in clusters:
        cid = c["cluster_id"]
        rep_terms = c.get("representative_terms") or []
        # 截取最多 N 条喂给 LLM
        n = min(max_cases_per_cluster, c["size"])
        idxs = c["sample_indices"][:n]
        questions = c["questions"][:n]
        prompts = c["prompts"][:n]

        cases_block = "\n".join(
            f"{i+1}. Q: {q}\n   A: (省略)" for i, q in enumerate(questions)
        )
        prompts_block = "\n".join(
            f"{i+1}. {p[:600]}" for i, p in enumerate(prompts)
        )

        user_prompt = _CLUSTER_PROTOTYPE_TEMPLATE.format(
            cluster_id=cid,
            cluster_terms=", ".join(rep_terms) or "（无）",
            cases_block=cases_block or "（无）",
            prompts_block=prompts_block or "（无）",
        )

        t0 = time.time()
        raw = _pio._call_llm(
            prompt=user_prompt,
            system=_CLUSTER_SYSTEM,
            temperature=temperature,
            max_tokens=900,
            model=model,
            api_key=api_key,
            timeout=timeout,
        )
        llm_time = round(time.time() - t0, 3)

        proto = raw.strip() if raw else ""
        used_fallback = False
        if not proto:
            # 退化：把当前聚类最长 prompt 当原型
            proto = max(prompts, key=len, default="")
            used_fallback = True

        out.append({
            "cluster_id": cid,
            "representative_terms": rep_terms,
            "representative_question": c.get("representative_question", ""),
            "prototype_prompt": proto,
            "raw": raw,
            "llm_time": llm_time,
            "used_fallback": used_fallback,
        })

    return out


# ════════════════════════════════════════════════════════════════
# Step 6：提示优化档案（结构化、可写盘）
# ════════════════════════════════════════════════════════════════

def archive_optimization(
    query: str,
    metrics: List[Dict[str, Any]],
    initial_prompt: str,
    best_prompt: str,
    best_score: float,
    iterations: int,
    stop_reason: str,
    field: str,
    case_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    把单次优化结果封装成结构化档案（可供前端展示 / 数据库回写）。

    Params:
        case_id: 可选业务方传入的案例编号
        extra:   任意附加字段（如原型 prompt 标签 / 用户 id / 时间戳）

    Returns:
        包含 prompt_v1 / prompt_vN / score / iter_history 的归档字典
    """
    archive = {
        "case_id": case_id,
        "field": field,
        "query": query,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": metrics,
        "initial_prompt": initial_prompt,
        "best_prompt": best_prompt,
        "best_score": best_score,
        "iterations": iterations,
        "stop_reason": stop_reason,
        "extra": extra or {},
    }
    return archive


def write_archive_to_file(archive: Dict[str, Any], path: str) -> str:
    """把档案写到 JSON 文件。返回绝对路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(archive, f, ensure_ascii=False, indent=2)
    return str(target.resolve())
