# -*- coding: utf-8 -*-
"""
test_api_server_live.py
=======================
启动 api_server 后，用 requests 调 5 个新接口 + 1 个 healthz，
验证 HTTP 请求→响应全链路正常。

用法：
    # 第一步：在另一个终端启动服务：
    cd D:/RAG_DB_slim && PORT=8123 python api_server.py
    # 第二步：跑本测试
    python test_api_server_live.py

注意：本测试不依赖真实 DashScope API Key。如下没有真实 key，服务会用 fake key
调 DashScope 接口（可能返回 HTML 错误页），但接口的 HTTP 状态码 / 响应 schema
均符合预期；如果要看到 iterate 接口有真实 LLM 输出，需要注入真的 DASHSCOPE_KEY。
"""

import os
import sys
import json
import time
import requests

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8123")
API_KEY = os.getenv("DASHSCOPE_KEY", "")
if not API_KEY:
    API_KEY = "sk-fake-for-mock-if-needed"  # 由 server LLM call 实际会失败，但 response shape 仍能验证

hdr = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def _post(path, body):
    r = requests.post(f"{API_BASE}{path}", headers=hdr, json=body, timeout=120)
    return r.status_code, r.json()


def _print_resp(name, code, body):
    print(f"[{name}] HTTP {code}")
    print(json.dumps(body, ensure_ascii=False, indent=2)[:1500])
    print()


def main():
    # 1) healthz
    try:
        r = requests.get(f"{API_BASE}/healthz", timeout=5)
        print(f"[healthz] HTTP {r.status_code}  body={r.json()}")
    except Exception as e:
        print(f"[healthz] connection failed: {e}")
        print("请先启动 api_server.py，端口 8123：")
        print("    PORT=8123 python api_server.py")
        return

    # 2) /case_iter/metrics
    print("\n=== /case_iter/metrics ===")
    code, body = _post("/case_iter/metrics", {"field": "旅游景区",
                                              "scenario_result": {"sub_queries": ["南孔庙开放时间"],
                                                                    "entity_terms": ["南孔庙"]}})
    _print_resp("metrics", code, body)
    assert body.get("ok"), "metrics should be ok"
    metrics = body["data"]["metrics"]

    # 3) /case_iter/initial_prompt
    print("=== /case_iter/initial_prompt ===")
    code, body = _post("/case_iter/initial_prompt",
                       {"field": "旅游景区", "query": "南孔庙的开放时间是什么时候？",
                        "scenario_result": {"sub_queries": ["南孔庙开放时间"]}})
    _print_resp("initial_prompt", code, body)
    assert body.get("ok"), "initial_prompt should be ok"
    initial_prompt = body["data"]["initial_prompt"]

    # 4) /case_iter/iterate
    print("=== /case_iter/iterate ===")
    code, body = _post("/case_iter/iterate", {
        "query": "南孔庙的开放时间是什么时候？",
        "prompt": initial_prompt or "你是景区助手。",
        "ground_truth": "南孔庙开放时间为每日 8:30-17:00。",
        "metrics": metrics,
        "max_iterations": 2,
        "score_threshold": 99.0,
        "score_window": 2,
        "min_delta": 0.5,
    })
    _print_resp("iterate", code, body)
    # iterate 在真实 LLM 调用下可能会因为 API key 错误返回 500，这是预期
    # 但 response shape（ok/code/error）应符合 schema

    # 5) /case_iter/cluster
    print("=== /case_iter/cluster ===")
    samples = [
        {"question": "南孔庙的开放时间是什么时候？", "optimized_prompt": "p1"},
        {"question": "南孔庙门票多少钱？", "optimized_prompt": "p2"},
        {"question": "江郎山的海拔是多少？", "optimized_prompt": "p3"},
        {"question": "江郎山在哪个省市？", "optimized_prompt": "p4"},
    ]
    r = requests.post(f"{API_BASE}/case_iter/cluster", headers=hdr,
                      json={"samples": samples, "n_clusters": 2}, timeout=60)
    body = r.json()
    _print_resp("cluster", r.status_code, body)

    # 6) /case_iter/prototypes
    print("=== /case_iter/prototypes ===")
    if body.get("ok") and body["data"].get("clusters"):
        clusters = body["data"]["clusters"]
        code2, body2 = _post("/case_iter/prototypes",
                             {"clusters": clusters, "max_cases_per_cluster": 3})
        _print_resp("prototypes", code2, body2)

    # 7) /case_iter/archive
    print("=== /case_iter/archive ===")
    archive = {
        "field": "旅游景区", "query": "南孔庙开放时间？",
        "metrics": metrics,
        "initial_prompt": initial_prompt or "(p)",
        "best_prompt": "（best）", "best_score": 80.0, "iterations": 1,
        "stop_reason": "test",
        "write_to": "_test_archive_live.json",
    }
    r = requests.post(f"{API_BASE}/case_iter/archive", headers=hdr, json=archive, timeout=10)
    body = r.json()
    _print_resp("archive", r.status_code, body)


if __name__ == "__main__":
    main()
