# -*- coding: utf-8 -*-
"""
test_case_iter.py
=================
本地快速验证 case_level_optimizer_service 的所有公开函数
& api_server 的 /case_iter/* 路由的输入输出格式。

策略：
  - LLM 调用会被 monkey-patch，避免消耗真实 token；
  - 真实加载 BGE 等大对象时跳过（cluster_case_samples 走 BGE 时单独测）；
  - 只验：函数能跑通 + 返回值字段结构符合 FastAPI 模型。
"""

import json
import sys
import types
from typing import Any, Dict, List

sys.path.insert(0, r"D:\RAG_DB_slim")

import case_level_optimizer_service as cls_module
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


# ─── LLM 替身：按入参不同假装返回合理 JSON ───

def _fake_call_llm(prompt: str, system: str = "", temperature: float = 0.3, max_tokens: int = 512,
                    model: str = "qwen-plus", api_key: str = None, timeout: int = 60) -> str:
    """Monkey-patch 后的 _call_llm，根据入参特征返回测试数据。"""
    p = prompt or ""
    s = system or ""

    # 1) 评估指标专家：系统提示含有 "评估指标专家" 且 prompt 含 metric_name 字段提示
    if "评估指标专家" in s and "metric_name" in p:
        return json.dumps({
            "evaluation_metrics": [
                {"metric_name": "回答准确性", "description": "覆盖参考答案事实", "weight": 0.30},
                {"metric_name": "回答完整性", "description": "覆盖关键字段",     "weight": 0.25},
                {"metric_name": "回答相关性", "description": "紧扣用户问题",     "weight": 0.20},
                {"metric_name": "不确定性表述", "description": "数据不足时声明", "weight": 0.15},
                {"metric_name": "格式规范",    "description": "输出风格稳定",     "weight": 0.10},
            ]
        }, ensure_ascii=False)

    # 2) Judge：系统提示含 "RAG 质量评估专家"
    if "RAG 质量评估专家" in s:
        names = ["回答准确性", "回答完整性", "回答相关性", "不确定性表述", "格式规范"]
        scores = {"回答准确性": 78, "回答完整性": 65, "回答相关性": 85,
                  "不确定性表述": 50, "格式规范": 70}
        return json.dumps({
            "scores": [{"metric_name": n, "score": scores.get(n, 70), "reason": "测试 reason"}
                       for n in names]
        }, ensure_ascii=False)

    # 3) Refine：系统提示含 "RAG Prompt 工程专家"
    if "RAG Prompt 工程专家" in s:
        return "（新版 prompt）我是新版景区导游，要求语气专业、回答准确。"

    # 4) initial prompt：系统提示含 "资深 Prompt Engineer"
    if "资深 Prompt Engineer" in s:
        return "（初始 prompt）我是景区导游。基于私域上下文回答，禁止编造。"

    # 5) cluster prototype 模板：系统提示含 "RAG Prompt 专家"
    if "RAG Prompt 专家" in s:
        return "（原型 prompt）通用景区问答模板，要求覆盖完整性、准确性、相关性。"

    # 6) iterate 里充当生成答案：用户消息含 "私域上下文" + "用户问题"
    return "（默认候选答案）南孔庙每日 8:30-17:00 开放，节假日不闭馆。"


# Monkey-patch：把 mock 同步替换到 prompt_iteration_optimizer 的符号
# 因为 judge_answer / refine_with_eval 在 import 时已经把 _call_llm 拷到了自己模块的 globals
import prompt_iteration_optimizer as pio
pio._call_llm = _fake_call_llm
cls_module._call_llm = _fake_call_llm

# 但 case_level_optimizer_service 模块内的 _call_llm 是从 pio 重新导入，
# 所以两边的引用都已经 mock 进真实函数体内（因为函数体是在 import 时已经求值 _call_llm
# 这个名字，但每个函数都写的是 `_call_llm(...)`，是 globals 查找）。
# case_level_optimizer_service 的 globals 仍指向 pio._call_llm 的旧绑定。
# 因此最安全的做法：直接把 mock 写回 prompt_iteration_optimizer.__dict__：
# （已经做了）→ 同时把 cls_module 的 globals 表里的 _call_llm 也覆盖：
cls_module.__dict__["_call_llm"] = _fake_call_llm
# 进一步：把 judge_answer / refine_with_eval 内部的 _call_llm 名称重绑定到 mock。
# 由于它们写的是 `_call_llm(...)`，会在 case_level_optimizer_service 的 globals
# 中查找；我们已经直接修改 cls_module.__dict__，所以会查到 mock。
print("[mock] _call_llm patched in both pio and cls_module.__dict__")


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def assert_keys(d: Dict[str, Any], keys: List[str], name: str) -> None:
    missing = [k for k in keys if k not in d]
    assert not missing, f"{name} 缺少字段: {missing}"
    ok(f"{name} 字段齐全: {keys}")


# ────────────────────────────────────────────
# 1) _safe_json_load
# ────────────────────────────────────────────
print("\n[1] _safe_json_load")
assert _safe_json_load("""{"a": 1}""") == {"a": 1}
assert _safe_json_load('```json\n{"a":2}\n```') == {"a": 2}
assert _safe_json_load("garbage text without json") is None
ok("纯 JSON / markdown 围栏 / 乱码 → 三种都正确处理")

# ────────────────────────────────────────────
# 2) _normalize_weights
# ────────────────────────────────────────────
print("\n[2] _normalize_weights")
m = _normalize_weights([
    {"metric_name": "A", "description": "x", "weight": 0.5},
    {"metric_name": "B", "description": "x", "weight": 0.5},
])
assert abs(sum(c["weight"] for c in m) - 1.0) < 1e-6, "权重之和必须为 1"
ok(f"权重之和 = {sum(c['weight'] for c in m):.4f}")

m2 = _normalize_weights([{"metric_name": "A", "weight": 2.0}, {"metric_name": "B", "weight": 1.5}])
assert all(0 <= c["weight"] <= 1 for c in m2), "权重必须归一化到 [0,1]"
ok(f"超界权重成功归一化: {[c['weight'] for c in m2]}")

# ────────────────────────────────────────────
# 3) build_evaluation_metrics (mock LLM)
# ────────────────────────────────────────────
print("\n[3] build_evaluation_metrics (mock)")
m_res = build_evaluation_metrics(
    field="旅游景区",
    expand_result={"sub_queries": ["南孔庙开放时间"], "entity_terms": ["南孔庙"], "prompt_module": None},
    api_key="sk-fake",
)
assert_keys(m_res, ["field", "metrics", "raw", "parse_ok", "used_fallback", "llm_time"], "build_evaluation_metrics")
assert sum(c["weight"] for c in m_res["metrics"]) - 1.0 < 1e-6
ok(f"返回 {len(m_res['metrics'])} 个评估指标，权重之和 = {sum(c['weight'] for c in m_res['metrics']):.4f}")

# ────────────────────────────────────────────
# 4) build_initial_prompt (mock LLM)
# ────────────────────────────────────────────
print("\n[4] build_initial_prompt (mock)")
i_res = build_initial_prompt(
    field="旅游景区",
    query="南孔庙的开放时间是什么时候？",
    api_key="sk-fake",
    base_prompt="（兜底 prompt）我是景区助手。",
)
assert_keys(i_res, ["initial_prompt", "raw", "used_base", "field", "llm_time"], "build_initial_prompt")
assert i_res["initial_prompt"], "initial_prompt 不能为空"
ok(f"初始 prompt 长度 = {len(i_res['initial_prompt'])} 字符")

# ────────────────────────────────────────────
# 5) judge_answer + score_answer (mock LLM)
# ────────────────────────────────────────────
print("\n[5] judge_answer + score_answer (mock)")
judge = judge_answer(
    query="南孔庙的开放时间？",
    candidate_answer="南孔庙每天 8:30-17:00 开放。",
    ground_truth="南孔庙开放时间为每日 8:30-17:00。",
    retrieved_context="南孔庙，每天 8:30 开门，17:00 关门。",
    metrics=m_res["metrics"],
    api_key="sk-fake",
)
assert_keys(judge, ["scores", "metrics", "raw", "parse_ok", "llm_time"], "judge_answer")
score = score_answer(judge, m_res["metrics"])
assert 0 <= score <= 100, "综合得分必须 0~100"
ok(f"综合得分 = {score}")

# ────────────────────────────────────────────
# 6) refine_with_eval + iterate_prompt_until_converged (mock)
# ────────────────────────────────────────────
print("\n[6] iterate_prompt_until_converged (mock)")

def _gen(p: str) -> str:
    return "（mock 答案）南孔庙每日 8:30-17:00 开放，节假日不闭馆。"

it = iterate_prompt_until_converged(
    query="南孔庙的开放时间？",
    initial_prompt="我是助手。",
    ground_truth="南孔庙开放时间为每日 8:30-17:00。",
    retrieved_context="南孔庙，每天 8:30 开门，17:00 关门。",
    metrics=m_res["metrics"],
    generate_answer_fn=_gen,
    api_key="sk-fake",
    max_iterations=3,
    score_threshold=99.0,    # 强制走满 3 轮，验证迭代能跑
    score_window=2,
    min_delta=0.5,
)
assert_keys(it, ["best_prompt", "best_score", "iterations", "history", "stop_reason",
                 "total_llm_time", "total_time"], "iterate_prompt_until_converged")
assert it["iterations"] >= 1
assert isinstance(it["history"], list)
ok(f"完成 {it['iterations']} 轮迭代，stop_reason={it['stop_reason']}, best_score={it['best_score']}")

# ────────────────────────────────────────────
# 7) cluster_case_samples (TF-IDF 回退路径，不依赖 BGE)
# ────────────────────────────────────────────
print("\n[7] cluster_case_samples (TF-IDF 回退)")
samples = [
    {"question": "南孔庙的开放时间？", "optimized_prompt": "p1"},
    {"question": "南孔庙门票多少钱？", "optimized_prompt": "p2"},
    {"question": "江郎山的海拔是多少？", "optimized_prompt": "p3"},
    {"question": "江郎山在哪个省市？", "optimized_prompt": "p4"},
    {"question": "廿八都的特色小吃？", "optimized_prompt": "p5"},
]
clu = cluster_case_samples(samples, encoder=None, n_clusters=3)
assert_keys(clu, ["n_samples", "n_clusters", "clusters", "encoders_used"], "cluster_case_samples")
assert clu["n_clusters"] >= 1
ok(f"聚类：{clu['n_samples']} 条 → {clu['n_clusters']} 类；BGE={clu['encoders_used']}")

# ────────────────────────────────────────────
# 8) extract_scene_prototypes (mock LLM)
# ────────────────────────────────────────────
print("\n[8] extract_scene_prototypes (mock)")
protos = extract_scene_prototypes(clu["clusters"], api_key="sk-fake")
assert isinstance(protos, list)
assert len(protos) == clu["n_clusters"]
for p in protos:
    assert_keys(p, ["cluster_id", "representative_terms", "representative_question",
                     "prototype_prompt", "raw", "llm_time", "used_fallback"], "prototype")
ok(f"成功抽出 {len(protos)} 个场景原型 prompt")

# ────────────────────────────────────────────
# 9) archive_optimization + write_archive_to_file
# ────────────────────────────────────────────
print("\n[9] archive_optimization + 落盘")
arc = archive_optimization(
    query="南孔庙开放时间？",
    metrics=m_res["metrics"],
    initial_prompt=i_res["initial_prompt"],
    best_prompt=it["best_prompt"],
    best_score=it["best_score"],
    iterations=it["iterations"],
    stop_reason=it["stop_reason"],
    field="旅游景区",
    case_id="case-001",
    extra={"tag": "smoke"},
)
assert_keys(arc, ["case_id", "field", "query", "created_at", "metrics",
                  "initial_prompt", "best_prompt", "best_score", "iterations",
                  "stop_reason", "extra"], "archive_optimization")
path = write_archive_to_file(arc, "_test_archive_case001.json")
ok(f"档案已落盘：{path}")
import os
os.remove(path)

# ────────────────────────────────────────────
# 10) FastAPI 路由 Schema 校验（不实际启服务）
# ────────────────────────────────────────────
print("\n[10] FastAPI 路由 Pydantic 模型实例化")
# 直接 import api_server 会启动 BGE 加载，这里仅检查 Pydantic 模型
from pydantic import ValidationError
import importlib.util
spec = importlib.util.spec_from_file_location("api_server_stub", r"D:\RAG_DB_slim\api_server.py")
# 不真执行模块，只读模块文本通过 spec 触发 Pydantic 模型定义需要 FastAPI
# 改为只用 Pydantic 模型：
from api_server import (
    CaseIterMetricsRequest, CaseIterMetricsData,
    CaseIterInitialPromptRequest,
    CaseIterIterateRequest,
    CaseIterClusterRequest,
    CaseIterPrototypesRequest,
    CaseIterArchiveRequest,
)

# 正常构造
CaseIterMetricsRequest(field="旅游景区")
CaseIterInitialPromptRequest(field="旅游景区", query="x")
CaseIterIterateRequest(query="x", prompt="p", ground_truth="g",
                       metrics=[{"metric_name":"M","description":"d","weight":1.0}])
CaseIterClusterRequest(samples=[{"question":"q","optimized_prompt":"p"}])
CaseIterPrototypesRequest(clusters=[{"cluster_id":0,"size":1,"sample_indices":[0],
                                      "representative_terms":[],"representative_question":"q",
                                      "questions":["q"],"prompts":["p"]}])
CaseIterArchiveRequest(field="a", query="q", metrics=[], initial_prompt="p",
                       best_prompt="p", best_score=10.0, iterations=1, stop_reason="ok")
ok("全部 Pydantic 模型能正常构造")

# 字段必填校验
try:
    CaseIterMetricsRequest()  # no field
    raise AssertionError("应该校验 field 必填")
except ValidationError:
    ok("CaseIterMetricsRequest 缺失 field 必填字段已拦截")

try:
    CaseIterIterateRequest(query="q")  # 缺 prompt/gt/ctx/metrics
    raise AssertionError("应该校验必填字段")
except ValidationError:
    ok("CaseIterIterateRequest 缺失必填字段已拦截")

print("\n========== 所有 case_level_optimizer_service 内部单元通过 ==========")
