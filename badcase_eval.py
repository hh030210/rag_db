#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量统计 interactive_qa.py 的 badcase 候选。

该脚本复用 interactive_qa.py 的检索、Query 扩展和问答逻辑，输出：

1. 每条问题一行的 JSONL 轨迹文件；
2. 汇总指标 JSON 文件。

输入支持 JSON、JSONL/NDJSON 和纯文本问题列表。若输入没有标准答案和
标准证据，只能统计系统/检索运行错误，不能可靠判断答案是否正确。

示例：
    DASHSCOPE_API_KEY=... python badcase_eval.py \
      --input qa_cases.jsonl \
      --output results/badcase_records.jsonl

    python badcase_eval.py --input qa_cases.jsonl --no-expand --mode sem --top-k 5
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import interactive_qa as qa


_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+")
_NUMBER_RE = re.compile(r"\d+(?:[.:：/\-]\d+)*%?")
_REFUSAL_HINTS = (
    "无法回答",
    "没有提供相应信息",
    "文中未说明",
    "知识库中没有",
    "无法确定",
    "不知道",
)


def _first(mapping: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "否", "不可回答"}
    return bool(value)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """读取常见 QA 数据格式，并统一为内部字段。"""
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        raw_rows = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                raw_rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行不是合法 JSON: {exc}") from exc
    elif path.suffix.lower() == ".txt":
        raw_rows = [
            {"question": line.strip()}
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            raw_rows = data
        elif isinstance(data, dict):
            raw_rows = None
            for key in ("data", "items", "questions", "samples", "records"):
                if isinstance(data.get(key), list):
                    raw_rows = data[key]
                    break
            if raw_rows is None:
                raw_rows = [data]
        else:
            raise ValueError("JSON 顶层必须是对象或数组")

    rows = []
    for index, raw in enumerate(raw_rows, 1):
        if isinstance(raw, str):
            raw = {"question": raw}
        if not isinstance(raw, dict):
            raise ValueError(f"第 {index} 条数据不是对象")

        question = _first(raw, ("question", "query", "instruction", "问题"), "")
        question = str(question).strip()

        reference = _first(
            raw,
            ("reference_answer", "ground_truth", "reference", "gold_answer", "标准答案", "answer"),
            "",
        )
        evidence_ids = _first(
            raw,
            ("evidence_chunk_ids", "gold_chunk_ids", "evidence_ids", "标准证据ID"),
            [],
        )
        evidence_texts = _first(
            raw,
            ("evidence_texts", "gold_evidence", "evidence", "标准证据", "source"),
            [],
        )

        rows.append(
            {
                "id": str(_first(raw, ("id", "question_id", "qid", "编号"), index)),
                "question": question,
                "input_valid": bool(question),
                "reference_answer": str(reference or ""),
                "evidence_chunk_ids": _as_list(evidence_ids),
                "evidence_texts": _as_list(evidence_texts),
                "answerable": _as_bool(_first(raw, ("answerable", "可回答"), True)),
                "question_type": str(_first(raw, ("question_type", "type", "问题类型"), "unknown")),
                "spot": str(_first(raw, ("spot", "景区", "entity"), "")),
                "source": raw.get("source", ""),
            }
        )
    return rows


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _token_f1(prediction: str, reference: str) -> float | None:
    pred = collections.Counter(_tokens(prediction))
    ref = collections.Counter(_tokens(reference))
    if not pred or not ref:
        return None
    overlap = sum((pred & ref).values())
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _evidence_text_hit(top_chunks: list[dict[str, Any]], evidence_texts: list[str]) -> bool:
    """用标准证据文本近似判断 Top-K 是否命中，作为没有 chunk_id 时的替代。"""
    retrieved = [
        _normalize_text(item.get("chunk_text_full") or item.get("chunk_text") or "")
        for item in top_chunks
    ]
    retrieved = [item for item in retrieved if item]
    for evidence in evidence_texts:
        gold = _normalize_text(evidence)
        if len(gold) < 8:
            continue
        for chunk in retrieved:
            if gold in chunk or (len(chunk) >= 8 and chunk in gold):
                return True
            if _token_f1(chunk, gold) is not None and _token_f1(chunk, gold) >= 0.60:
                return True
    return False


def _safe_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_result(result: dict[str, Any], rank: int, trace_chars: int) -> dict[str, Any]:
    text = result.get("chunk_text_full") or result.get("chunk_text") or ""
    if trace_chars > 0:
        text = str(text)[:trace_chars]
    compact = {
        "rank": rank,
        "chunk_id": str(result.get("chunk_id", "")),
        "title": str(result.get("chunk_gen_title") or result.get("doc_title") or ""),
        "text": text,
        "source": str(result.get("source", "")),
    }
    for key in ("score", "final_score", "dim_score", "sem_score"):
        if key in result:
            number = _safe_number(result.get(key))
            compact[key] = number if number is not None else result.get(key)
    return compact


def _trace_results(results: list[dict[str, Any]], trace_chars: int) -> list[dict[str, Any]]:
    return [_compact_result(item, index, trace_chars) for index, item in enumerate(results, 1)]


def _config_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "top_k": args.top_k,
        "dim_alpha": args.dim_alpha,
        "sem_alpha": args.sem_alpha,
        "rerank_mode": args.rerank_mode,
        "fusion_strategy": args.fusion_strategy,
        "expand_enabled": args.expand,
        "model": qa.DS_MODEL,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "context_chars": args.context_chars,
        "api_interval": args.api_interval,
    }


def _classify(
    case: dict[str, Any],
    search_result: dict[str, Any],
    qa_result: dict[str, Any],
    low_overlap_threshold: float,
) -> dict[str, Any]:
    labels: list[str] = []
    answer = str(qa_result.get("answer") or "").strip()
    top_chunks = search_result.get("top_chunks") or []
    top_ids = {str(item.get("chunk_id", "")) for item in top_chunks if item.get("chunk_id")}
    gold_ids = set(case.get("evidence_chunk_ids") or [])
    gold_texts = case.get("evidence_texts") or []

    retrieval_hit = None
    retrieval_hit_basis = None
    if gold_ids:
        retrieval_hit_basis = "evidence_chunk_ids"
        retrieval_hit = bool(gold_ids & top_ids)
        if not retrieval_hit:
            labels.append("retrieval_miss_at_k")
    elif gold_texts:
        retrieval_hit_basis = "evidence_texts_approximate"
        retrieval_hit = _evidence_text_hit(top_chunks, gold_texts)
        if not retrieval_hit:
            labels.append("retrieval_miss_at_k_candidate")

    search_error = str(search_result.get("error") or "").strip()
    qa_error = str(qa_result.get("error") or "").strip()
    generation_errors = qa_result.get("generation_errors") or []
    if search_error or qa_error or generation_errors:
        labels.append("system_error")
    if not top_chunks:
        labels.append("empty_retrieval")

    if case.get("answerable", True):
        if not answer:
            labels.append("empty_answer")

        reference = str(case.get("reference_answer") or "")
        f1 = _token_f1(answer, reference) if reference else None
        reference_numbers = _numbers(reference)
        answer_numbers = _numbers(answer)
        missing_numbers = [number for number in reference_numbers if number not in answer_numbers]
        if missing_numbers:
            labels.append("reference_number_missing_candidate")
        if f1 is not None and f1 < low_overlap_threshold:
            labels.append("low_reference_overlap_candidate")
    else:
        f1 = None
        missing_numbers = []
        if answer and not any(hint in answer for hint in _REFUSAL_HINTS):
            labels.append("out_of_scope_answer_candidate")

    critical = {"system_error", "empty_retrieval", "empty_answer"}
    major = {"retrieval_miss_at_k", "reference_number_missing_candidate", "out_of_scope_answer_candidate"}
    if any(label in critical for label in labels):
        severity = "critical"
    elif any(label in major for label in labels):
        severity = "major"
    elif labels:
        severity = "minor_review"
    else:
        severity = "none"

    return {
        "auto_candidate_badcase": bool(labels),
        "labels": labels,
        "severity": severity,
        "retrieval_hit_at_k": retrieval_hit,
        "retrieval_hit_basis": retrieval_hit_basis,
        "reference_token_f1": f1,
        "reference_numbers": _numbers(str(case.get("reference_answer") or "")),
        "answer_numbers": _numbers(answer),
        "missing_reference_numbers": missing_numbers,
        "needs_human_review": any("candidate" in label for label in labels),
    }


def _run_one(case: dict[str, Any], args: argparse.Namespace, expander: Any = None) -> dict[str, Any]:
    started = time.time()
    raw_query = case["question"]
    retrieval_query = raw_query
    expansion = {"enabled": args.expand, "error": ""}
    expansion_result = None

    if args.expand:
        try:
            if expander is None:
                expander = qa._get_expander()
            expansion_result = expander.expand(raw_query)
            retrieval_query = qa.build_fusion_query(expansion_result)
            expansion.update(
                {
                    "sub_queries": expansion_result.get("sub_queries", []),
                    "entity_terms": expansion_result.get("entity_terms", []),
                    "cluster_id": expansion_result.get("cluster_id"),
                    "cluster_sim": expansion_result.get("cluster_sim"),
                    "expand_time": expansion_result.get("expand_time"),
                }
            )
        except Exception as exc:
            expansion["error"] = f"{type(exc).__name__}: {exc}"
            # 只让当前问题回退到原始 query，不改变后续问题的扩展开关。

    coverage = {"in_range": True, "uncovered_spots": []}
    try:
        in_range, uncovered = qa.check_coverage(raw_query)
        coverage = {"in_range": in_range, "uncovered_spots": uncovered}
    except Exception as exc:
        coverage["error"] = f"{type(exc).__name__}: {exc}"

    search_result: dict[str, Any]
    search_started = time.time()
    try:
        search_result = qa.do_search(
            query=retrieval_query,
            mode=args.mode,
            top_k=args.top_k,
            dim_alpha=args.dim_alpha,
            sem_alpha=args.sem_alpha,
            rerank_mode=args.rerank_mode,
        )
    except Exception as exc:
        search_result = {
            "mode": args.mode,
            "elapsed": time.time() - search_started,
            "dim_results": [],
            "sem_results": [],
            "fusion_results": [],
            "top_chunks": [],
            "constraints": {},
            "error": f"{type(exc).__name__}: {exc}",
        }

    qa_result: dict[str, Any]
    qa_started = time.time()
    try:
        # 交互函数仍会打印结果；批量模式默认吞掉控制台输出，避免污染 JSONL 统计。
        output = io.StringIO()
        stream = sys.stdout if args.verbose else output
        with contextlib.redirect_stdout(stream):
            result = qa.do_qa(
                query=raw_query,
                chunks=search_result.get("top_chunks", []),
                top_k=args.top_k,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                expand_result=expansion_result,
                context_chars=args.context_chars,
            )
        qa_result = result or {
            "answer": "",
            "elapsed": time.time() - qa_started,
            "prompt_count": 0,
            "successful_prompt_count": 0,
            "generation_errors": [],
            "error": "qa_returned_no_result",
        }
    except Exception as exc:
        qa_result = {
            "answer": "",
            "elapsed": time.time() - qa_started,
            "prompt_count": 0,
            "successful_prompt_count": 0,
            "generation_errors": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    record = {
        "id": case["id"],
        "question": raw_query,
        "input_valid": case.get("input_valid", True),
        "retrieval_query": retrieval_query,
        "reference_answer": case.get("reference_answer", ""),
        "evidence_chunk_ids": case.get("evidence_chunk_ids", []),
        "evidence_texts": case.get("evidence_texts", []),
        "answerable": case.get("answerable", True),
        "question_type": case.get("question_type", "unknown"),
        "spot": case.get("spot", ""),
        "coverage": coverage,
        "expansion": expansion,
        "config": _config_snapshot(args),
        "retrieval": {
            "elapsed": search_result.get("elapsed"),
            "error": search_result.get("error", ""),
            "constraints": search_result.get("constraints", {}),
            "adaptive_dim_alpha": search_result.get("adaptive_dim_alpha"),
            "adaptive_sem_alpha": search_result.get("adaptive_sem_alpha"),
            "semantic_results": _trace_results(search_result.get("sem_results", []), args.trace_chars),
            "dimension_results": _trace_results(search_result.get("dim_results", []), args.trace_chars),
            "fusion_results": _trace_results(search_result.get("fusion_results", []), args.trace_chars),
            "top_chunks": _trace_results(search_result.get("top_chunks", []), args.trace_chars),
        },
        "qa": {
            "answer": qa_result.get("answer", ""),
            "elapsed": qa_result.get("elapsed", time.time() - qa_started),
            "prompt_count": qa_result.get("prompt_count", 0),
            "successful_prompt_count": qa_result.get("successful_prompt_count", 0),
            "generation_errors": qa_result.get("generation_errors", []),
            "error": qa_result.get("error", ""),
        },
        "timing": {"total_elapsed": time.time() - started},
    }
    record["classification"] = _classify(
        case,
        search_result,
        qa_result,
        low_overlap_threshold=args.low_overlap_threshold,
    )
    return record


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _build_summary(records: list[dict[str, Any]], args: argparse.Namespace, input_path: Path) -> dict[str, Any]:
    total = len(records)
    invalid_inputs = [record for record in records if not record.get("input_valid", True)]
    valid_records = [record for record in records if record.get("input_valid", True)]
    answerable = [record for record in valid_records if record.get("answerable", True)]
    candidates = [record for record in valid_records if record["classification"]["auto_candidate_badcase"]]
    system_errors = [record for record in valid_records if "system_error" in record["classification"]["labels"]]
    empty_retrieval = [record for record in valid_records if "empty_retrieval" in record["classification"]["labels"]]
    empty_answer = [record for record in valid_records if "empty_answer" in record["classification"]["labels"]]
    with_gold_evidence = [
        record for record in answerable
        if record.get("evidence_chunk_ids") or record.get("evidence_texts")
    ]
    retrieval_hits = [record for record in with_gold_evidence if record["classification"]["retrieval_hit_at_k"]]
    retrieval_misses = [record for record in with_gold_evidence if not record["classification"]["retrieval_hit_at_k"]]

    category_counts = collections.Counter()
    for record in records:
        category_counts.update(record["classification"]["labels"])

    by_type: dict[str, dict[str, Any]] = {}
    for question_type in sorted({record.get("question_type", "unknown") for record in records}):
        subset = [record for record in records if record.get("question_type", "unknown") == question_type]
        bad = sum(record["classification"]["auto_candidate_badcase"] for record in subset)
        by_type[question_type] = {
            "total": len(subset),
            "candidate_badcases": bad,
            "candidate_badcase_rate": _rate(bad, len(subset)),
        }

    return {
        "input": str(input_path),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": _config_snapshot(args),
        "counts": {
            "total": total,
            "valid_inputs": len(valid_records),
            "invalid_inputs": len(invalid_inputs),
            "answerable": len(answerable),
            "unanswerable": total - len(answerable),
            "candidate_badcases": len(candidates),
            "system_errors": len(system_errors),
            "empty_retrieval": len(empty_retrieval),
            "empty_answer": len(empty_answer),
            "gold_evidence_available": len(with_gold_evidence),
            "retrieval_hits_at_k": len(retrieval_hits),
            "retrieval_misses_at_k": len(retrieval_misses),
        },
        "rates": {
            "candidate_badcase_rate": _rate(len(candidates), total),
            "answerable_candidate_badcase_rate": _rate(
                sum(record["classification"]["auto_candidate_badcase"] for record in answerable),
                len(answerable),
            ),
            "system_error_rate": _rate(len(system_errors), total),
            "retrieval_miss_rate_given_gold_evidence": _rate(len(retrieval_misses), len(with_gold_evidence)),
            "empty_retrieval_rate": _rate(len(empty_retrieval), total),
            "empty_answer_rate_on_answerable": _rate(
                sum("empty_answer" in record["classification"]["labels"] for record in answerable),
                len(answerable),
            ),
        },
        "category_counts": dict(category_counts),
        "by_question_type": by_type,
        "limitations": [
            "candidate_badcase 是自动筛选结果，不等同于最终人工判定。",
            "没有标准证据 ID 时无法计算可靠的检索 Hit@K。",
            "BLEU/ROUGE 或 token overlap 不能单独判定语义正确性。",
            "幻觉、事实矛盾和上下文支持关系需要规则增强或人工/LLM 复核。",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量统计 interactive_qa.py 的 badcase 候选")
    parser.add_argument("--input", required=True, help="JSON/JSONL/NDJSON/TXT 问题集")
    parser.add_argument("--output", default="badcase_records.jsonl", help="逐问题 JSONL 输出路径")
    parser.add_argument("--max-items", type=int, default=0, help="最多运行多少条，0 表示全部")
    parser.add_argument("--mode", choices=("fusion", "dim", "sem"), default=qa.RETRIEVAL_MODE)
    parser.add_argument("--top-k", type=int, default=qa.RETRIEVAL_TOP_K)
    parser.add_argument("--dim-alpha", type=float, default=qa.DIM_ALPHA)
    parser.add_argument("--sem-alpha", type=float, default=qa.SEM_ALPHA)
    parser.add_argument("--rerank-mode", choices=("score", "interleaved"), default=qa.RERANK_MODE)
    parser.add_argument("--fusion-strategy", choices=("adaptive", "fixed"), default=qa.FUSION_STRATEGY)
    parser.add_argument("--temperature", type=float, default=qa.LLM_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=qa.LLM_MAX_TOKENS)
    parser.add_argument("--trace-chars", type=int, default=2000, help="每个检索 chunk 保存的字符数，0 表示不截断")
    parser.add_argument("--context-chars", type=int, default=500, help="每个检索 chunk 传给问答模型的字符数，0 表示不截断")
    parser.add_argument("--api-interval", type=float, default=2.0, help="两次问答 API 请求的最小间隔秒数")
    parser.add_argument("--low-overlap-threshold", type=float, default=0.15)
    parser.add_argument("--verbose", action="store_true", help="显示交互问答过程")
    parser.add_argument("--expand", dest="expand", action="store_true", default=qa.EXPAND_ENABLED)
    parser.add_argument("--no-expand", dest="expand", action="store_false")
    args = parser.parse_args()
    if args.top_k < 1 or args.max_tokens < 1 or args.trace_chars < 0 or args.context_chars < 0 or args.api_interval < 0:
        parser.error("top-k 和 max-tokens 必须为正数，trace-chars、context-chars 和 api-interval 不能为负数")
    return args


def main() -> int:
    args = _parse_args()
    qa._DS_RATE_LIMITER.min_interval_seconds = args.api_interval
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    cases = _load_rows(input_path)
    if args.max_items > 0:
        cases = cases[: args.max_items]
    if not cases:
        raise ValueError("输入数据为空")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    expander = None
    for index, case in enumerate(cases, 1):
        if not case.get("input_valid", True):
            record = {
                "id": case["id"],
                "question": case.get("question", ""),
                "input_valid": False,
                "reference_answer": case.get("reference_answer", ""),
                "evidence_chunk_ids": case.get("evidence_chunk_ids", []),
                "evidence_texts": case.get("evidence_texts", []),
                "answerable": case.get("answerable", True),
                "question_type": case.get("question_type", "unknown"),
                "spot": case.get("spot", ""),
                "coverage": {},
                "expansion": {"enabled": args.expand, "error": "invalid_empty_question"},
                "config": _config_snapshot(args),
                "retrieval": {"top_chunks": [], "error": "invalid_empty_question"},
                "qa": {"answer": "", "error": "invalid_empty_question"},
                "timing": {"total_elapsed": 0.0},
                "classification": {
                    "auto_candidate_badcase": True,
                    "labels": ["invalid_input"],
                    "severity": "critical",
                    "retrieval_hit_at_k": None,
                    "retrieval_hit_basis": None,
                    "reference_token_f1": None,
                    "reference_numbers": _numbers(case.get("reference_answer", "")),
                    "answer_numbers": [],
                    "missing_reference_numbers": [],
                    "needs_human_review": False,
                },
            }
        else:
            record = _run_one(case, args, expander=expander)
        records.append(record)
        print(
            f"[{index}/{len(cases)}] {case['id']} "
            f"badcase={record['classification']['auto_candidate_badcase']} "
            f"labels={','.join(record['classification']['labels']) or '-'}",
            file=sys.stderr,
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    summary = _build_summary(records, args, input_path)
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"records: {output_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(summary["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
