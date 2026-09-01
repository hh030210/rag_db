#!/usr/bin/env python3
"""将二次复核 badcase JSONL 整理为按错误类型分组的可读 JSON。"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path


ERROR_NAMES = {
    "incomplete": "答案不完整",
    "retrieval": "检索未命中",
    "wrong": "事实或内容错误",
    "refusal": "应答但拒答",
}


def clip(value, limit=400):
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def simplify(row, row_number):
    review = row.get("review") or {}
    classification = row.get("classification") or {}
    qa = row.get("qa") or {}
    evidence = []
    for chunk in (row.get("retrieval") or {}).get("top_chunks", [])[:5]:
        evidence.append(
            OrderedDict(
                [
                    ("排名", chunk.get("rank")),
                    ("标题", chunk.get("title", "")),
                    ("文本摘要", clip(chunk.get("text", ""))),
                ]
            )
        )
    return OrderedDict(
        [
            ("原始行号", row_number),
            ("ID", row.get("id")),
            ("问题", row.get("question", "")),
            ("标准答案", row.get("reference_answer", "")),
            ("系统答案", qa.get("answer", "")),
            ("答案判定", review.get("answer_judgement")),
            ("检索判定", review.get("retrieval_judgement")),
            ("综合判定", review.get("overall_judgement")),
            ("主错误类型", ERROR_NAMES.get(review.get("error_type"), review.get("error_type"))),
            ("复核理由", review.get("reason", "")),
            ("自动候选标签", classification.get("labels", [])),
            ("Top-K检索证据摘要", evidence),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    groups = OrderedDict((name, []) for name in ["答案不完整", "检索未命中", "事实或内容错误", "应答但拒答", "其他"])
    rows = []
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    for row_number, row in enumerate(rows, 1):
        error_type = (row.get("review") or {}).get("error_type")
        group = ERROR_NAMES.get(error_type, "其他")
        groups[group].append(simplify(row, row_number))

    output = OrderedDict(
        [
            ("说明", "融合检索实验二次复核 badcase 的人工可读整理版；原始 JSONL 未修改。"),
            ("总记录数", len(rows)),
            ("分类统计", OrderedDict((name, len(items)) for name, items in groups.items())),
            ("记录", groups),
        ]
    )
    target = Path(args.output)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成 {target}，共 {len(rows)} 条记录")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
