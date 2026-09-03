#!/usr/bin/env python3
"""将 badcase JSONL 整理为便于人工阅读的分组 JSON。"""

import json
from collections import OrderedDict
from pathlib import Path


SOURCE = Path("results/server0/badcase_reviewed.jsonl")
TARGET = Path("results/server0/badcase_reviewed_readable.json")

ERROR_NAMES = {
    "incomplete": "答案不完整",
    "retrieval": "检索未命中",
    "wrong": "事实或内容错误",
    "refusal": "应答但拒答",
}


def clip(value, limit=500):
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def make_record(row_number, row, include_evidence=False):
    classification = row.get("classification") or {}
    review = row.get("review") or {}
    qa = row.get("qa") or {}
    result = OrderedDict(
        [
            ("原始行号", row_number),
            ("ID", row.get("id")),
            ("问题", row.get("question", "")),
            ("标准答案", row.get("reference_answer", "")),
            ("系统答案", qa.get("answer", "")),
            (
                "自动筛选",
                OrderedDict(
                    [
                        ("候选badcase", classification.get("auto_candidate_badcase", False)),
                        ("标签", classification.get("labels", [])),
                        ("严重等级", classification.get("severity")),
                        ("检索命中", classification.get("retrieval_hit_at_k")),
                    ]
                ),
            ),
            (
                "二次复核",
                OrderedDict(
                    [
                        ("答案判定", review.get("answer_judgement")),
                        ("检索判定", review.get("retrieval_judgement")),
                        ("综合判定", review.get("overall_judgement")),
                        ("主错误类型", ERROR_NAMES.get(review.get("error_type"), review.get("error_type"))),
                        ("复核理由", review.get("reason", "")),
                    ]
                ),
            ),
        ]
    )
    if include_evidence:
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
        result["Top-K检索证据摘要"] = evidence
    return result


def main():
    groups = OrderedDict(
        [
            ("答案不完整", []),
            ("检索未命中", []),
            ("事实或内容错误", []),
            ("应答但拒答", []),
            ("不确定", []),
            ("可接受", []),
            ("未进入二次复核", []),
            ("输入无效", []),
        ]
    )

    for row_number, line in enumerate(SOURCE.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        review = row.get("review") or {}
        source = review.get("review_source")
        overall = review.get("overall_judgement")
        if source == "qwen-plus" and overall == "badcase":
            group = ERROR_NAMES.get(review.get("error_type"), "不确定")
        elif source == "qwen-plus" and overall == "uncertain":
            group = "不确定"
        elif source == "qwen-plus" and overall == "acceptable":
            group = "可接受"
        elif not row.get("input_valid", True):
            group = "输入无效"
        else:
            group = "未进入二次复核"
        groups[group].append(make_record(row_number, row, group in ERROR_NAMES.values()))

    output = OrderedDict(
        [
            ("说明", "这是 badcase_reviewed.jsonl 的人工可读整理版；原始 JSONL 未修改。"),
            (
                "统计",
                OrderedDict(
                    [
                        ("总记录数", sum(len(items) for items in groups.values())),
                        ("模型二次复核候选数", sum(len(groups[name]) for name in ["答案不完整", "检索未命中", "事实或内容错误", "应答但拒答", "不确定", "可接受"])),
                        ("模型综合badcase数", sum(len(groups[name]) for name in ["答案不完整", "检索未命中", "事实或内容错误", "应答但拒答"])),
                        ("无效输入数", len(groups["输入无效"])),
                    ]
                ),
            ),
            (
                "分类统计",
                OrderedDict((name, len(items)) for name, items in groups.items()),
            ),
            ("记录", groups),
        ]
    )
    TARGET.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成 {TARGET}，共 {sum(len(items) for items in groups.values())} 条记录")


if __name__ == "__main__":
    main()
