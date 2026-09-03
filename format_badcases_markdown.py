#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把最终 badcase JSONL 整理为按错误类型分组的 Markdown。"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path


ERROR_NAMES = OrderedDict(
    [
        ("incomplete", "答案不完整"),
        ("requirement", "未满足问题要求"),
        ("wrong", "事实或内容错误"),
        ("retrieval", "检索未命中"),
        ("unsupported", "证据不足/无依据"),
        ("uncertain", "不确定"),
    ]
)


def clip(value, limit=800):
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def load_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        review = row.get("review") or {}
        if review.get("overall_judgement") == "badcase" and row.get("input_valid", True):
            rows.append(row)
    return rows


def build_markdown(rows, input_path: Path) -> str:
    groups = OrderedDict((key, []) for key in ERROR_NAMES)
    for row in rows:
        error_type = (row.get("review") or {}).get("error_type", "uncertain")
        groups.setdefault(error_type, []).append(row)

    config = (rows[0].get("config") or {}) if rows else {}
    lines = [
        "# 融合检索问答实验全部最终 Badcase",
        "",
        f"本文件整理本轮复核后全部 `{len(rows)}` 条有效最终 badcase。自动候选共 280 条，只有经过二次复核且 `overall_judgement=badcase` 的记录才纳入本文件。",
        "",
        "## 实验条件",
        "",
        f"- 检索模式：{config.get('mode', 'fusion')}",
        f"- Top-K：{config.get('top_k', '未知')}",
        f"- 问答模型：{config.get('model', '未知')}",
        f"- 每个 chunk 提供给模型的字符数：{config.get('context_chars', '未知')}",
        f"- Query 扩展：{config.get('expand_enabled', config.get('expand', '未知'))}",
        "- 说明：二次复核由模型完成，不等同于人工金标准；不确定样本仍需人工确认。",
        "",
        "## 分类统计",
        "",
        "| 错误类型 | 数量 |",
        "|---|---:|",
    ]
    for key, name in ERROR_NAMES.items():
        if groups.get(key):
            lines.append(f"| {name} | {len(groups[key])} |")
    for key, group in groups.items():
        if not group:
            continue
        name = ERROR_NAMES.get(key, key)
        lines.extend(["", f"## {name}（{len(group)} 条）", ""])
        for index, row in enumerate(group, 1):
            review = row.get("review") or {}
            classification = row.get("classification") or {}
            retrieval = row.get("retrieval") or {}
            qa = row.get("qa") or {}
            row_number = review.get("record_index")
            row_label = f"原始行号 {row_number + 1}" if isinstance(row_number, int) else ""
            lines.extend(
                [
                    f"### {index}. ID {row.get('id', '')}（{row_label}）",
                    "",
                    f"- 问题：{clip(row.get('question', ''), 1000)}",
                    f"- 标准答案：{clip(row.get('reference_answer', ''), 1600)}",
                    f"- 系统答案：{clip(qa.get('answer', ''), 1600)}",
                    f"- 答案判定：{review.get('answer_judgement', '未知')}",
                    f"- 检索判定：{review.get('retrieval_judgement', '未知')}",
                    f"- 复核理由：{clip(review.get('reason', ''), 500)}",
                    f"- 自动候选标签：{', '.join(classification.get('labels') or []) or '无'}",
                    "",
                    "**Top-K 检索证据：**",
                    "",
                ]
            )
            top_chunks = retrieval.get("top_chunks") or []
            if not top_chunks:
                lines.append("- 无检索证据。")
            else:
                for chunk in top_chunks[:5]:
                    lines.extend(
                        [
                            f"- Top-{chunk.get('rank', '?')}｜{chunk.get('chunk_id', '')}｜{chunk.get('title', '')}",
                            f"  - 来源：{chunk.get('source', '') or '未知'}；维度召回方式：{chunk.get('recall_method', '') or '无'}",
                            f"  - 证据摘要：{clip(chunk.get('text', ''), 600)}",
                        ]
                    )
            lines.extend(["", "---", ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    rows = load_rows(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown(rows, input_path), encoding="utf-8")
    print(f"已生成 {output_path}，共 {len(rows)} 条最终 badcase")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
