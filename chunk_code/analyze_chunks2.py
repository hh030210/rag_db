import json, statistics

for name, path in [
    ("v1_baseline",  "output_baseline_v1/all_chunks_chunks.json"),
    ("v2_dedup",     "output_enhanced_v2/all_chunks_chunks.json"),
]:
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    lengths = [c["chunk_len"] for c in chunks]
    s = sorted(lengths)
    n = len(lengths)
    print(f"\n=== {name} ({n} chunks) ===")
    print(f"  mean={statistics.mean(lengths):.1f}  median={statistics.median(lengths):.1f}  stdev={statistics.stdev(lengths):.1f}")
    print(f"  range=[{min(lengths)}, {max(lengths)}]")
    print(f"  P25={s[n//4]}  P50={s[n//2]}  P75={s[3*n//4]}  P90={s[9*n//10]}")
    # Length distribution
    bins = {"<100": 0, "100-300": 0, "300-500": 0, "500-800": 0, ">800": 0}
    for l in lengths:
        if l < 100: bins["<100"] += 1
        elif l < 300: bins["100-300"] += 1
        elif l < 500: bins["300-500"] += 1
        elif l <= 800: bins["500-800"] += 1
        else: bins[">800"] += 1
    for k, v in bins.items():
        print(f"  {k}: {v} ({100*v/n:.1f}%)")
