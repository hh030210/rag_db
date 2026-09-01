#!/usr/bin/env python3
"""Quick sanity check for the monotonic constraint changes."""
import sys
sys.path.insert(0, 'd:/RAG_DB_slim')

from case_level_optimizer_service import (
    iterate_prompt_until_converged,
    refine_with_eval,
    judge_answer,
    score_answer,
    _REFINE_TEMPLATE,
)
from api_server import CaseIterIterateRequest

# 1. Check function signature
import inspect
params = list(inspect.signature(iterate_prompt_until_converged).parameters.keys())
assert 'monotonic_max_retries' in params, "missing monotonic_max_retries"
print("[OK] iterate_prompt_until_converged has monotonic_max_retries param")

# 2. Check API model
fields = CaseIterIterateRequest.model_fields
assert 'monotonic_max_retries' in fields, "missing monotonic_max_retries in request model"
print(f"[OK] monotonic_max_retries field default={fields['monotonic_max_retries'].default}")

# 3. Check refine template mentions monotonicity
assert '不低于上一轮得分' in _REFINE_TEMPLATE, "refine template missing monotonicity requirement"
assert '严禁' in _REFINE_TEMPLATE, "refine template missing prohibition"
print("[OK] refine template contains monotonicity constraints")

# 4. Smoke test iterate_prompt_until_converged with mock functions
def mock_generate(prompt):
    return "mock answer about 南孔庙开放时间 08:00-17:30"

mock_metrics = [
    {"metric_name": "回答准确性", "description": "事实正确", "weight": 0.5},
    {"metric_name": "格式规范性", "description": "格式好", "weight": 0.5},
]

result = iterate_prompt_until_converged(
    query="南孔庙几点开门？",
    initial_prompt="你是一个景区导游。",
    ground_truth="南孔庙开放时间为08:00-17:30。",
    retrieved_context="南孔庙开放时间：08:00-17:30。",
    metrics=mock_metrics,
    generate_answer_fn=mock_generate,
    api_key="mock-key",
    max_iterations=1,
    monotonic_max_retries=2,
)
assert "best_prompt" in result, "missing best_prompt in result"
assert "history" in result, "missing history"
assert len(result["history"]) >= 1, "history should have at least 1 entry"
# Check per_metric_scores is present in history
h = result["history"][0]
assert "per_metric_scores" in h, f"missing per_metric_scores in history entry: {list(h.keys())}"
print(f"[OK] iterate_prompt_until_converged smoke test: iterations={result['iterations']}, stop_reason={result['stop_reason']}")
print(f"     per_metric_scores keys: {list(result['history'][0]['per_metric_scores'].keys())}")

print("\n=== All checks PASSED ===")
