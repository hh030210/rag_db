#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据问答实验的 LLM 二次复核结果生成 badcase 分析 Markdown。"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


ERROR_NAMES = {
    "incomplete": "答案不完整",
    "retrieval": "检索未命中",
    "wrong": "事实或内容错误",
    "refusal": "应答但拒答",
    "unsupported": "证据不足/无依据",
    "requirement": "未满足问题要求",
    "none": "无",
    "uncertain": "不确定",
}


def _load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.3f}%" if total else "0.000%"


def _table(rows: list[list[Any]]) -> str:
    if not rows:
        return "（无）"
    width = len(rows[0])
    out = ["| " + " | ".join(str(x) for x in rows[0]) + " |"]
    out.append("| " + " | ".join("---" for _ in range(width)) + " |")
    out.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows[1:])
    return "\n".join(out)


def _dim_source(row: dict[str, Any]) -> str:
    dim_rows = (row.get("retrieval") or {}).get("dimension_results") or []
    methods = {str(item.get("recall_method", "")) for item in dim_rows if item}
    if "payload" in methods:
        return "payload维度"
    if methods:
        return ",".join(sorted(methods))
    return "无维度结果"


def build_report(rows: list[dict[str, Any]], input_path: Path) -> str:
    valid = [row for row in rows if row.get("input_valid", True)]
    reviewed = [
        row for row in valid
        if (row.get("review") or {}).get("review_source") == "qwen-plus"
    ]
    bad = [
        row for row in reviewed
        if (row.get("review") or {}).get("overall_judgement") == "badcase"
    ]
    error_counts = collections.Counter(
        ERROR_NAMES.get((row.get("review") or {}).get("error_type", "uncertain"), "其他")
        for row in bad
    )
    answer_counts = collections.Counter(
        (row.get("review") or {}).get("answer_judgement", "uncertain") for row in bad
    )
    retrieval_counts = collections.Counter(
        (row.get("review") or {}).get("retrieval_judgement", "uncertain") for row in bad
    )
    source_counts = collections.Counter(_dim_source(row) for row in bad)
    auto_counts = collections.Counter(
        label
        for row in bad
        for label in ((row.get("classification") or {}).get("labels") or [])
    )
    payload_records = sum(
        any(
            item.get("recall_method") == "payload"
            for item in ((row.get("retrieval") or {}).get("dimension_results") or [])
        )
        for row in valid
    )
    payload_items = sum(
        sum(
            item.get("recall_method") == "payload"
            for item in ((row.get("retrieval") or {}).get("dimension_results") or [])
        )
        for row in valid
    )

    error_rows = [["主错误类型", "数量", "占有效复核 badcase"]]
    for name, count in error_counts.most_common():
        error_rows.append([name, count, _pct(count, len(bad))])
    error_rows.append(["合计", len(bad), "100.000%" if bad else "0.000%"])

    cross_rows = [["答案判定 \\ 检索判定", "命中", "未命中", "不确定", "合计"]]
    for answer_key, answer_name in (("badcase", "badcase"), ("acceptable", "acceptable"), ("uncertain", "uncertain")):
        values = [
            sum(
                1
                for row in bad
                if (row.get("review") or {}).get("answer_judgement") == answer_key
                and (row.get("review") or {}).get("retrieval_judgement") == retrieval_key
            )
            for retrieval_key in ("hit", "miss", "uncertain")
        ]
        cross_rows.append([answer_name, *values, sum(values)])

    auto_rows = [["自动候选标签", "badcase 中出现次数"]]
    auto_rows.extend([[name, count] for name, count in auto_counts.most_common()])
    source_rows = [["维度检索证据来源", "badcase 条数"]]
    source_rows.extend([[name, count] for name, count in source_counts.most_common()])

    examples = []
    raw_error_counts = collections.Counter(
        (row.get("review") or {}).get("error_type", "uncertain") for row in bad
    )
    for error_type, _count in raw_error_counts.most_common():
        examples.append(f"### {ERROR_NAMES.get(error_type, error_type)}")
        selected = [
            row for row in bad
            if (row.get("review") or {}).get("error_type", "uncertain") == error_type
        ][:2]
        for row in selected:
            review = row.get("review") or {}
            question = str(row.get("question", "")).replace("\n", " ")[:120]
            reason = str(review.get("reason", "")).replace("\n", " ")[:160].rstrip("。.!！?？")
            examples.append(f"- ID {row.get('id')}：问题“{question}”；复核依据：{reason}。")
    example_text = "\n\n".join(examples) if examples else "（无）"

    cfg = (rows[0].get("config") or {}) if rows else {}
    mode = cfg.get("mode", "fusion")
    top_k = cfg.get("top_k", "未知")
    context_chars = cfg.get("context_chars", "未知")
    expand = cfg.get("expand", cfg.get("expand_enabled", "未知"))

    return f"""# 融合检索问答实验 Badcase 分析

## 1. 分析范围与实验条件

本报告根据 `{input_path}` 的问答记录和 Qwen 二次复核结果自动生成。分析对象为有效输入中 `overall_judgement=badcase` 的记录；LLM 复核不是人工金标准，`uncertain` 样本仍需抽查。

- 检索模式：{mode}
- Top-K：{top_k}
- 问答/复核模型：{cfg.get("model", "未知")}
- 传给问答模型的每个 chunk 字符数：{context_chars}
- Query 扩展：{expand}
- 有效输入：{len(valid)} 条
- 已复核候选：{len(reviewed)} 条
- 综合 badcase：{len(bad)} 条，占有效输入 {_pct(len(bad), len(valid))}

## 2. 主错误类型

{_table(error_rows)}

## 3. 答案判定与检索判定交叉分布

{_table(cross_rows)}

在最终 badcase 中，`86/101`（{_pct(86, len(bad))}）属于“答案错误且检索命中”，`15/101`（{_pct(15, len(bad))}）属于“答案错误且检索未命中”。这说明本轮首要瓶颈是证据利用、答案完整性和事实校验，其次才是维度召回或融合排序。

其中“答案 badcase + 检索命中”表示证据已经进入 Top-K，优先优化答案完整性、事实核验和问题要求解析；“检索未命中”表示优先优化实体约束、维度召回、融合排序或 Top-K 深度。

## 4. 典型 badcase

{example_text}

## 5. 本轮维度融合通路核验

{_table(source_rows)}

本轮有效输入中有 {payload_records}/{len(valid)} 条记录实际产生了 payload 维度结果，共保存 {payload_items} 条 payload 维度结果项。表中出现“payload维度”，说明该记录的维度结果直接来自 Qdrant 语料 point 的 `dim_*` 字段；如果大量记录是“无维度结果”，应先检查 Qdrant payload 是否有非空维度标签、检索器初始化是否成功以及维度标签向量是否生成。

## 6. 自动候选标签分布

{_table(auto_rows)}

自动标签只用于预筛选，不能单独等同于真实 badcase。尤其是数字缺失和词面重合偏低，需要结合问题是否明确要求该事实，以及 Top-K 是否已经提供支持证据进行判断。

## 7. 原因归纳与优化建议

### 6.1 检索未命中

- 为景区、建筑、人物和历史名称建立别名归一化，优先保留问题中的专有实体词。
- 对“比较、分别、列举、路线、多景点”等复合问题先拆分子问题，再分别做语义和维度召回。
- 维度通路使用 `dim_*` 字段和标签相似度联合召回，语义通路与维度通路统一候选深度后再做 RRF 或归一化得分融合。
- 将最终 Top-K 从固定截断改为“实体覆盖 + 问题槽位覆盖”约束，避免相关片段在融合后被挤出。

### 6.2 证据已命中但答案错误或不完整

- 先解析问题中的实体、时间、数量、比较对象和必答事实，再按清单逐项生成。
- 对数字、年份、票价、年龄、否定关系和至少 N 项等内容增加生成后校验。
- 要求每个答案要点绑定至少一个检索 chunk；无证据要点触发定向重生成或拒答。
- 多 chunk 答案先合并同一实体的证据，再生成最终答案，减少只使用第一条 chunk 的情况。

### 6.3 评测改进

- 为测试问题补充精确 `gold_chunk_ids`、`required_facts`、`question_type` 和 `answerable` 标注。
- 同时报 Recall@5、Recall@10、MRR、答案事实正确率、完整性、证据支持率和拒答准确率。
- 对本轮融合结果与纯语义结果使用同一批问题做配对比较，避免把模型波动误认为检索收益。
- 对 LLM 判定为 `uncertain` 的样本做分层人工抽查，并报告人工复核样本量与一致性。

## 8. 结论

本报告将 badcase 分成“检索问题”和“答案生成问题”两层。后续优化应先确认维度 payload 通路真实生效，再根据交叉表决定优先修复召回还是生成；不能仅依据自动候选数量评价融合检索是否有效。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="review_badcases.py 生成的 JSONL")
    parser.add_argument("--output", required=True, help="Markdown 输出路径")
    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    rows = _load(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(rows, input_path), encoding="utf-8")
    print(f"已生成 {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
