import json, statistics, sys
from pathlib import Path

results = {}

for name, path in [
    ("v1_baseline",  "output_baseline_v1/all_chunks_chunks.json"),
    ("v2_nodedup",   "output_enhanced_v2_nodedup/all_chunks_chunks.json"),
    ("v2_dedup",     "output_enhanced_v2/all_chunks_chunks.json"),
]:
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    lengths = [c["chunk_len"] for c in chunks]
    s = sorted(lengths)
    results[name] = {
        "count": len(chunks),
        "mean": round(statistics.mean(lengths), 1),
        "median": round(statistics.median(lengths), 1),
        "stdev": round(statistics.stdev(lengths), 1),
        "min": min(lengths),
        "max": max(lengths),
        "p25": s[len(s)//4],
        "p75": s[3*len(s)//4],
    }

for name, r in results.items():
    print(f"\n=== {name} ({r['count']} chunks) ===")
    print(f"  mean={r['mean']}  median={r['median']}  stdev={r['stdev']}")
    print(f"  range=[{r['min']}, {r['max']}]")
    print(f"  P25={r['p25']}  P75={r['p75']}")
