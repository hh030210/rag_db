#!/usr/bin/env python3
"""从二次复核 JSONL 中提取综合判定为 badcase 的记录。"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="提取 overall_judgement=badcase 记录")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--valid-only", action="store_true", help="排除 input_valid=false 的记录")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    total = 0
    selected = []

    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        review = row.get("review") or {}
        if review.get("overall_judgement") != "badcase":
            continue
        if args.valid_only and not row.get("input_valid", True):
            continue
        selected.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in selected),
        encoding="utf-8",
    )
    error_types = collections.Counter((row.get("review") or {}).get("error_type", "unknown") for row in selected)
    answer_judgements = collections.Counter((row.get("review") or {}).get("answer_judgement", "unknown") for row in selected)
    retrieval_judgements = collections.Counter((row.get("review") or {}).get("retrieval_judgement", "unknown") for row in selected)
    summary = {
        "input": str(input_path),
        "total_records": total,
        "extracted_records": len(selected),
        "valid_only": args.valid_only,
        "error_type_counts": dict(error_types),
        "answer_judgement_counts": dict(answer_judgements),
        "retrieval_judgement_counts": dict(retrieval_judgements),
        "definition": "review.overall_judgement == badcase",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
