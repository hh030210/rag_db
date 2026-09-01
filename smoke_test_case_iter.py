#!/usr/bin/env python3
"""Server-side smoke test for new /case_iter/* endpoints."""
import sys
import json
import requests

BASE = "http://127.0.0.1:8000"
HDR = {"Authorization": "Bearer test-key", "Content-Type": "application/json"}

def check(name, resp, expect_ok=True):
    body = resp.json()
    ok = body.get("ok", False)
    code = resp.status_code
    print(f"  [{name}] HTTP {code}  ok={ok}")
    if expect_ok:
        assert ok, f"{name} failed: {body.get('error')}"
        assert code == 200, f"{name} status {code}"
    return body

print("=== Server smoke test ===")

# 1) healthz
r = requests.get(f"{BASE}/healthz", timeout=5)
check("healthz", r)

# 2) metrics
r = requests.post(f"{BASE}/case_iter/metrics", headers=HDR,
                   json={"field": "旅游景区"}, timeout=30)
data = check("metrics", r)
print(f"    metrics count: {len(data['data']['metrics'])}")

# 3) initial_prompt
r = requests.post(f"{BASE}/case_iter/initial_prompt", headers=HDR,
                   json={"field": "旅游景区", "query": "南孔庙开放时间？"}, timeout=30)
check("initial_prompt", r)

# 4) cluster
r = requests.post(f"{BASE}/case_iter/cluster", headers=HDR,
                   json={"samples": [{"question": "南孔庙开放时间？", "optimized_prompt": "p1"},
                                    {"question": "南孔庙门票？", "optimized_prompt": "p2"},
                                    {"question": "江郎山海拔？", "optimized_prompt": "p3"}],
                           "n_clusters": 2}, timeout=30)
data = check("cluster", r)
print(f"    n_clusters: {data['data']['n_clusters']}")

# 5) archive
r = requests.post(f"{BASE}/case_iter/archive", headers=HDR,
                   json={"field": "旅游景区", "query": "test",
                         "metrics": [], "initial_prompt": "ip", "best_prompt": "bp",
                         "best_score": 80.0, "iterations": 2, "stop_reason": "smoke_test"},
                   timeout=10)
check("archive", r)

print("\n=== All server tests PASSED ===")
