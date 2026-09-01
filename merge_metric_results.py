#!/usr/bin/env python3
import json
import time
from pathlib import Path

root = Path('/home/humq/chunk_code/metric_results')
expected = [root / f'fast_stickiness_{mode}_shard{i}.json' for mode in ('off', 'on') for i in range(4)]
while not all(p.exists() for p in expected):
    time.sleep(60)

def summary(values):
    n = len(values)
    mean = sum(values) / n if n else 0.0
    std = (sum((x - mean) ** 2 for x in values) / n) ** 0.5 if n else 0.0
    return {'n': n, 'mean': mean, 'std': std, 'min': min(values) if n else 0.0, 'max': max(values) if n else 0.0}

report = {'boundary': {}}
for mode in ('off', 'on'):
    rows = [json.loads((root / f'fast_stickiness_{mode}_shard{i}.json').read_text())['stickiness'] for i in range(4)]
    g1 = [x for row in rows for x in row.get('values_G1', [])]
    g3 = [x for row in rows for x in row.get('values_G3', [])]
    report[f'stickiness_{mode}'] = {
        'metric': 'Chunk Stickiness (PPL graph structural entropy)',
        'structural_entropy_G1': summary(g1),
        'structural_entropy_G3': summary(g3),
        'n_chunks_input': sum(row['n_chunks_input'] for row in rows),
        'n_chunks_evaluated': sum(row['n_chunks_evaluated'] for row in rows),
        'n_chunks_skipped': sum(row['n_chunks_skipped'] for row in rows),
    }
    boundary_file = root / f'boundary_semantic_{mode}_v2.json'
    if boundary_file.exists():
        report['boundary'][mode] = json.loads(boundary_file.read_text())['boundary']

for mode in ('off', 'on'):
    relation_file = root / f'fast_relation_{mode}_full_v6.json'
    if relation_file.exists():
        report[f'relation_{mode}'] = json.loads(relation_file.read_text()).get('relation', {})

(root / 'paper_metrics_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2))
