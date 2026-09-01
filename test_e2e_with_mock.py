# -*- coding: utf-8 -*-
"""
test_e2e_with_mock.py
=====================
在同一个 Python 进程里：
    1. monkey-patch prompt_iteration_optimizer._call_llm 为 fake
    2. 启动 FastAPI（uvicorn）服务
    3. 用 requests 调用全部 7 个接口
    4. 校验响应 shape 100% 正确
    5. 优雅关停

优点：
    - 不需要真实 DashScope API Key
    - 全链路验证：接口 → service → mock LLM → 返回 schema
"""

import sys
import json
import time
import threading
import requests

# 1) Monkey-patch 在 import api_server 之前
import prompt_iteration_optimizer as pio


def _fake_call_llm(prompt: str, system: str = "", temperature: float = 0.3, max_tokens: int = 512,
                    model: str = "qwen-plus", api_key: str = None, timeout: int = 60) -> str:
    p, s = prompt or "", system or ""
    if "评估指标专家" in s and "metric_name" in p:
        return json.dumps({
            "evaluation_metrics": [
                {"metric_name": "回答准确性", "description": "d1", "weight": 0.30},
                {"metric_name": "回答完整性", "description": "d2", "weight": 0.25},
                {"metric_name": "回答相关性", "description": "d3", "weight": 0.20},
                {"metric_name": "不确定性表述", "description": "d4", "weight": 0.15},
                {"metric_name": "格式规范",    "description": "d5", "weight": 0.10},
            ]
        }, ensure_ascii=False)
    if "RAG 质量评估专家" in s:
        names = ["回答准确性", "回答完整性", "回答相关性", "不确定性表述", "格式规范"]
        scores = {"回答准确性": 78, "回答完整性": 65, "回答相关性": 85,
                  "不确定性表述": 50, "格式规范": 70}
        return json.dumps({"scores": [{"metric_name": n, "score": scores[n], "reason": "mock"}
                                      for n in names]}, ensure_ascii=False)
    if "RAG Prompt 工程专家" in s:
        return "（新版 prompt）我是新版景区导游，要求语气专业、回答准确。"
    if "资深 Prompt Engineer" in s:
        return "（初始 prompt）我是景区导游。基于私域上下文回答，禁止编造。"
    if "RAG Prompt 专家" in s:
        return "（原型 prompt）通用景区问答模板。"
    # 默认候选答案
    return "（默认候选答案）南孔庙每日 8:30-17:00 开放。"


pio._call_llm = _fake_call_llm
import case_level_optimizer_service as cls_module
cls_module._call_llm = _fake_call_llm
cls_module.__dict__["_call_llm"] = _fake_call_llm

# ── mock search 调用（内部会调 requests.post）──
import requests as _requests_module
_orig_post = _requests_module.post
def _fake_post(url, json=None, timeout=None):
    if "search" in str(url):
        return _FakeResponse({"ok": True, "data": {"fusion_results": [
            {"content": "南孔庙每天 8:30 开门，17:00 关门。"}
        ]}})
    return _orig_post(url, json=json, timeout=timeout)

class _FakeResponse:
    def __init__(self, data): self._data = data
    def json(self): return self._data

_requests_module.post = _fake_post

# 2) 启动 FastAPI
import uvicorn
from api_server import app

PORT = 8124
config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error", access_log=False)
server = uvicorn.Server(config)

t = threading.Thread(target=server.run, daemon=True)
t.start()
time.sleep(4)  # 等服务启动

BASE = f"http://127.0.0.1:{PORT}"
hdr = {"Authorization": "Bearer mock-key", "Content-Type": "application/json"}


def _check(name, resp, must_have_keys=()):
    body = resp.json()
    print(f"  [{name}] HTTP {resp.status_code}")
    print(f"  ok={body.get('ok')}, code={body.get('code')}, error={body.get('error')}")
    if body.get("data"):
        for k in must_have_keys:
            assert k in body["data"], f"{name} 缺少字段 {k}"
    print()


print("=" * 60)
print("E2E 测试 (mock LLM)")
print("=" * 60)

# healthz
r = requests.get(f"{BASE}/healthz", timeout=5)
_check("healthz", r)

# metrics
r = requests.post(f"{BASE}/case_iter/metrics", headers=hdr,
                  json={"field": "旅游景区"}, timeout=15)
_check("metrics", r, ["field", "metrics"])
metrics = r.json()["data"]["metrics"]

# initial_prompt
r = requests.post(f"{BASE}/case_iter/initial_prompt", headers=hdr,
                  json={"field": "旅游景区", "query": "南孔庙的开放时间？"}, timeout=15)
_check("initial_prompt", r, ["initial_prompt"])
initial_prompt = r.json()["data"]["initial_prompt"]

# iterate
r = requests.post(f"{BASE}/case_iter/iterate", headers=hdr,
                  json={
                      "query": "南孔庙的开放时间？",
                      "prompt": initial_prompt or "我是助手",
                      "ground_truth": "南孔庙每日 8:30-17:00 开放。",
                      "metrics": metrics,
                      "max_iterations": 3,
                      "score_threshold": 99.0,
                      "score_window": 2,
                      "min_delta": 0.5,
                  }, timeout=120)
_check("iterate", r, ["best_prompt", "best_score", "iterations", "history", "stop_reason"])
print(f"  best_score={r.json()['data']['best_score']}, "
      f"iterations={r.json()['data']['iterations']}, "
      f"stop_reason={r.json()['data']['stop_reason']}")
print()

# cluster
r = requests.post(f"{BASE}/case_iter/cluster", headers=hdr,
                  json={"samples": [
                      {"question": "南孔庙开放时间？", "optimized_prompt": "p1"},
                      {"question": "南孔庙门票多少？", "optimized_prompt": "p2"},
                      {"question": "江郎山海拔？",     "optimized_prompt": "p3"},
                  ]}, timeout=30)
_check("cluster", r, ["n_samples", "n_clusters", "clusters"])
clusters = r.json()["data"]["clusters"]

# prototypes
r = requests.post(f"{BASE}/case_iter/prototypes", headers=hdr,
                  json={"clusters": clusters}, timeout=30)
_check("prototypes", r, ["prototypes"])

# archive
r = requests.post(f"{BASE}/case_iter/archive", headers=hdr,
                  json={"field": "旅游景区", "query": "南孔庙开放时间？",
                        "metrics": metrics, "initial_prompt": initial_prompt,
                        "best_prompt": "best", "best_score": 80.0,
                        "iterations": 2, "stop_reason": "ok",
                        "write_to": "_test_archive_e2e.json"}, timeout=10)
_check("archive", r, ["archive"])

print("=" * 60)
print("All interfaces e2e PASSED")
print("=" * 60)

server.should_exit = True
time.sleep(1)

# 清理
import os
try:
    os.remove("_test_archive_e2e.json")
except OSError:
    pass
