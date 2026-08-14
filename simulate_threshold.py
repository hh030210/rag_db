# -*- coding: utf-8 -*-
import sys, io, json
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

experiment_data_dir = Path("D:/RAG_DB_slim/experiment_data")

# 加载所有 tags_output 文件
tags_files = list(experiment_data_dir.glob("tags_output*.json"))
tags_files = [f for f in tags_files if f.is_file()]

merged_doc_data = {}
dim_value_sets = defaultdict(set)
dim_dataset_doc_counts = defaultdict(lambda: defaultdict(int))
dataset_doc_counts = defaultdict(int)

for f in tags_files:
    with open(f, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    name = f.stem.replace("tags_output", "").lstrip("_") or "default"
    dataset_doc_counts[name] = len(data)
    merged_doc_data.update(data)

    for doc_id, tags in data.items():
        for dim, vals in tags.items():
            if not vals:
                continue
            if isinstance(vals, str):
                vals = [vals]
            for v in vals:
                dim_value_sets[dim].add(v)
                dim_dataset_doc_counts[dim][name] += 1

total_docs = len(merged_doc_data)
print(f"合并后共 {total_docs} 条文档（来自 {len(dataset_doc_counts)} 个数据集）\n")

# 测试不同阈值的效果
for threshold in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
    dims_pass = []
    dims_fail = []
    for dim, val_set in dim_value_sets.items():
        per_dataset_coverage = {
            ds: dim_dataset_doc_counts[dim][ds] / dataset_doc_counts[ds]
            if dataset_doc_counts[ds] > 0 else 0.0
            for ds in dataset_doc_counts
        }
        max_cov = max(per_dataset_coverage.values()) if per_dataset_coverage else 0.0
        best_ds = max(per_dataset_coverage, key=per_dataset_coverage.get) if per_dataset_coverage else ""
        if max_cov >= threshold:
            dims_pass.append((dim, len(val_set), max_cov, best_ds))
        else:
            dims_fail.append((dim, len(val_set), max_cov, best_ds))

    print(f"{'='*60}")
    print(f"阈值 {threshold:.0%} -> 保留 {len(dims_pass)} 个维度, 过滤 {len(dims_fail)} 个")
    print(f"{'='*60}")
    if dims_fail:
        for dim, n, cov, best_ds in sorted(dims_fail, key=lambda x: x[2], reverse=True):
            print(f"  过滤: [{dim}] {n} 值, 最高覆盖率 {cov:.1%} ({best_ds})")
    print()
