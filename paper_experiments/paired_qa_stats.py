#!/usr/bin/env python3
"""Compute paired, non-human uncertainty statistics for QA outputs."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


TASK_FILES = {
    "1-doc": "QuestAnswer1Doc_qwen_api.json",
    "2-doc": "QuestAnswer2Docs_qwen_api.json",
    "3-doc": "QuestAnswer3Docs_qwen_api.json",
}
METRICS = ("bleu-avg", "rouge-L", "bertScore")


def read_task(root: Path, filename: str) -> dict[int, dict[str, float]]:
    matches = list(root.rglob(filename))
    if not matches:
        return {}
    data = json.loads(matches[0].read_text(encoding="utf-8"))
    rows: dict[int, dict[str, float]] = {}
    for row in data.get("results", []):
        if not row.get("valid", False):
            continue
        try:
            row_id = int(row["id"])
        except (KeyError, TypeError, ValueError):
            continue
        metrics = row.get("metrics", {})
        rows[row_id] = {name: float(metrics.get(name, 0.0) or 0.0) for name in METRICS}
    return rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def paired_bootstrap(diffs: list[float], draws: int, seed: int) -> tuple[float, float, float]:
    if not diffs:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(diffs)
    means = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws)]
    mean = sum(diffs) / n
    return mean, percentile(means, 0.025), percentile(means, 0.975)


def sign_test_pvalue(diffs: list[float]) -> float:
    signs = [1 if value > 0 else -1 for value in diffs if value != 0]
    n = len(signs)
    if n == 0:
        return 1.0
    smaller = min(sum(x > 0 for x in signs), sum(x < 0 for x in signs))
    tail = sum(math.comb(n, i) for i in range(smaller + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()

    qa_root = Path(args.qa_root).resolve()
    methods = ["mechanical_200", "mechanical_300", "mechanical_400", "three_stage"]
    loaded = {
        method: {
            task: read_task(qa_root / method, filename)
            for task, filename in TASK_FILES.items()
        }
        for method in methods
    }
    if not any(loaded["three_stage"].values()):
        raise RuntimeError("未找到三阶段问答结果，不能进行配对统计")

    result = {"comparison": "each mechanical method minus three_stage", "tasks": {}}
    for task in TASK_FILES:
        result["tasks"][task] = {}
        baseline = loaded["three_stage"][task]
        for method in methods[:-1]:
            current = loaded[method][task]
            ids = sorted(set(current) & set(baseline))
            result["tasks"][task][method] = {"n_paired": len(ids), "metrics": {}}
            for metric_index, metric in enumerate(METRICS):
                diffs = [current[row_id][metric] - baseline[row_id][metric] for row_id in ids]
                mean, low, high = paired_bootstrap(diffs, args.draws, 1000 + metric_index)
                result["tasks"][task][method]["metrics"][metric] = {
                    "mean_diff": mean,
                    "bootstrap_ci95": [low, high],
                    "sign_test_p": sign_test_pvalue(diffs),
                }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
