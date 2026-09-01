#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 LLM 对 badcase 自动候选做二次复核。

输出保留原始逐题记录，并增加：
    review.answer_judgement    答案是否构成错误/不完整
    review.retrieval_judgement Top-K 是否提供了回答所需证据
    review.overall_judgement   综合判断
    review.reason              简短依据

该脚本只复核原始自动候选，其他记录保留为 not_candidate；结果仍需人工抽查。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import time
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """你是严格的 RAG 问答错误分析员。请根据用户问题、标准答案、系统答案和 Top-K 检索上下文，判断自动候选是否是真正的错误。

判定原则：
1. 只评价问题实际要求的内容；标准答案中的额外信息没有被问到时，系统答案遗漏它不自动算错。
2. 事实错误、与标准答案矛盾、应当回答却拒答、没有满足“至少/列举/分别”等明确要求，判定答案 badcase。
3. 语义等价、只是措辞不同，判定答案 acceptable。
4. 仅当 Top-K 上下文没有提供回答问题所需的支持性信息时，才判定检索 miss；标准证据文本较长而 Top-K 命中其中相关片段时，判定 retrieval hit。
5. 信息不足以确定时使用 uncertain，不要猜测。

只输出 JSON，不要 Markdown：
{"answer_judgement":"badcase|acceptable|uncertain","retrieval_judgement":"miss|hit|uncertain","overall_judgement":"badcase|acceptable|uncertain","error_type":"wrong|incomplete|refusal|retrieval|unsupported|requirement|none|uncertain","reason":"不超过80字的中文理由"}"""


def _clip(text: Any, limit: int = 1200) -> str:
    return str(text or "")[:limit]


def _prompt(record: dict[str, Any]) -> str:
    chunks = []
    for item in (record.get("retrieval") or {}).get("top_chunks", [])[:5]:
        chunks.append(
            f"[Top-{item.get('rank', '?')}] {item.get('title', '')}\n"
            f"{_clip(item.get('text', ''), 1200)}"
        )
    labels = ",".join((record.get("classification") or {}).get("labels", []))
    return (
        f"用户问题：{record.get('question', '')}\n\n"
        f"标准答案：{_clip(record.get('reference_answer', ''), 1600)}\n\n"
        f"系统答案：{_clip((record.get('qa') or {}).get('answer', ''), 1600)}\n\n"
        f"自动候选标签：{labels}\n\n"
        f"Top-K 检索上下文：\n{chr(10).join(chunks)}"
    )


def _parse_json(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def _normalize_review(value: dict[str, Any]) -> dict[str, Any]:
    allowed_answer = {"badcase", "acceptable", "uncertain"}
    allowed_retrieval = {"miss", "hit", "uncertain"}
    answer = value.get("answer_judgement", "uncertain")
    retrieval = value.get("retrieval_judgement", "uncertain")
    overall = value.get("overall_judgement", "uncertain")
    if answer not in allowed_answer:
        answer = "uncertain"
    if retrieval not in allowed_retrieval:
        retrieval = "uncertain"
    if overall not in allowed_answer:
        overall = "uncertain"
    error_type = str(value.get("error_type", "uncertain"))
    if error_type not in {"wrong", "incomplete", "refusal", "retrieval", "unsupported", "requirement", "none", "uncertain"}:
        error_type = "uncertain"
    return {
        "answer_judgement": answer,
        "retrieval_judgement": retrieval,
        "overall_judgement": overall,
        "error_type": error_type,
        "reason": _clip(value.get("reason", ""), 300),
        "review_source": "qwen-plus",
    }


def _call_llm(api_key: str, prompt: str) -> dict[str, Any]:
    import dashscope
    from dashscope import Generation

    result = Generation.call(
        "qwen-plus",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        result_format="message",
        temperature=0,
        max_tokens=256,
        api_key=api_key,
    )
    if getattr(result, "status_code", None) != 200:
        raise RuntimeError(f"DashScope error: {getattr(result, 'code', '')} {getattr(result, 'message', '')}")
    content = result.output.choices[0].message.content
    parsed = _parse_json(content)
    if not parsed:
        raise ValueError("LLM 未返回合法 JSON")
    return _normalize_review(parsed)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--interval", type=float, default=2.0, help="两次 API 请求的最小间隔秒数")
    parser.add_argument("--max-items", type=int, default=0, help="只复核前多少条候选，0 表示全部")
    return parser.parse_args()


def main() -> int:
    args = _args()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("请通过 DASHSCOPE_API_KEY 环境变量提供 API Key")

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output_path.with_name(output_path.stem + "_summary.json")
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = [
        (row_index, row) for row_index, row in enumerate(rows)
        if row.get("input_valid", True) and (row.get("classification") or {}).get("auto_candidate_badcase")
    ]
    if args.max_items > 0:
        candidates = candidates[:args.max_items]
    candidate_indexes = {row_index for row_index, _row in candidates}

    existing: dict[int, dict[str, Any]] = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                review = row.get("review", {})
                if review.get("review_source") == "qwen-plus" and isinstance(review.get("record_index"), int):
                    existing[review["record_index"]] = review
            except Exception:
                continue

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reviewed: dict[str, dict[str, Any]] = {}
    last_call = 0.0
    for index, (record_index, row) in enumerate(candidates, 1):
        if record_index in existing:
            reviewed[record_index] = existing[record_index]
            continue
        wait = args.interval - (time.monotonic() - last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            review = _call_llm(api_key, _prompt(row))
            last_call = time.monotonic()
        except Exception as exc:
            review = {
                "answer_judgement": "uncertain",
                "retrieval_judgement": "uncertain",
                "overall_judgement": "uncertain",
                "error_type": "uncertain",
                "reason": f"复核请求失败：{type(exc).__name__}: {exc}",
                "review_source": "qwen-plus",
            }
            last_call = time.monotonic()
        review["record_index"] = record_index
        reviewed[record_index] = review
        print(f"[{index}/{len(candidates)}] row={record_index + 1} id={row.get('id')} overall={review['overall_judgement']} answer={review['answer_judgement']} retrieval={review['retrieval_judgement']}", flush=True)

    output_rows = []
    for record_index, row in enumerate(rows):
        if record_index in reviewed:
            row["review"] = reviewed[record_index]
        elif not row.get("input_valid", True):
            row["review"] = {
                "answer_judgement": "badcase",
                "retrieval_judgement": "uncertain",
                "overall_judgement": "badcase",
                "error_type": "requirement",
                "reason": "问题为空，输入无效",
                "review_source": "rule",
            }
        else:
            row["review"] = {"overall_judgement": "not_candidate", "review_source": "rule"}
        output_rows.append(row)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    reviewed_rows = [
        row for row in output_rows
        if row["review"].get("review_source") == "qwen-plus"
        and row["review"].get("record_index") in candidate_indexes
    ]
    summary = {
        "input": str(input_path),
        "total_records": len(output_rows),
        "candidate_records_reviewed": len(reviewed_rows),
        "candidate_records_expected": len(candidate_indexes),
        "overall_counts": dict(collections.Counter(row["review"].get("overall_judgement", "unknown") for row in reviewed_rows)),
        "answer_counts": dict(collections.Counter(row["review"].get("answer_judgement", "unknown") for row in reviewed_rows)),
        "retrieval_counts": dict(collections.Counter(row["review"].get("retrieval_judgement", "unknown") for row in reviewed_rows)),
        "error_type_counts": dict(collections.Counter(row["review"].get("error_type", "unknown") for row in reviewed_rows)),
        "limitations": [
            "这是 Qwen 二次复核，不是人工金标准；uncertain 需要人工确认。",
            "复核依据是已保存的 Top-K 文本截断内容，不能替代完整知识库审计。",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"records: {output_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
