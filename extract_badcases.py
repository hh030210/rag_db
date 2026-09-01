#!/usr/bin/env python3
"""从批量问答 JSONL 中提取自动候选 badcase。"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="提取 auto_candidate_badcase 记录")
    parser.add_argument("--input", required=True, help="批量问答 JSONL")
    parser.add_argument("--output", required=True, help="提取后的 badcase JSONL")
    parser.add_argument("--summary", required=True, help="提取结果汇总 JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    total = 0
    valid = 0
    badcases = 0
    labels = collections.Counter()
    severities = collections.Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if row.get("input_valid", True):
                valid += 1
            classification = row.get("classification") or {}
            if not classification.get("auto_candidate_badcase", False):
                continue
            badcases += 1
            labels.update(classification.get("labels") or [])
            severities.update([classification.get("severity", "unknown")])
            target.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    summary = {
        "input": str(input_path),
        "total_records": total,
        "valid_inputs": valid,
        "invalid_inputs": total - valid,
        "badcase_records": badcases,
        "badcase_rate": badcases / total if total else None,
        "label_counts": dict(labels),
        "severity_counts": dict(severities),
        "definition": "classification.auto_candidate_badcase == true",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
