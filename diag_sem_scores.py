# -*- coding: utf-8 -*-
"""诊断语义检索的得分分布"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import interactive_qa as iqa

queries = [
    "明十三陵有哪十三个",
    "龙门石窟的门票价格",
    "颐和园怎么游玩",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"查询: {q}")
    print(f"{'='*60}")

    # 只跑语义检索，看完整 Top-20 得分
    try:
        raw = iqa.sem_searcher.search(q, top_k=20)
        results = raw.get("results", [])
        print(f"共返回 {len(results)} 条，结果如下：")
        for i, r in enumerate(results):
            score = r.get("score", 0)
            cid = r.get("chunk_id", "?")
            text = (r.get("chunk_text_full") or r.get("chunk_text") or "")[:80].replace("\n", " ")
            marker = " ← 相关!" if any(kw in cid for kw in ["明十三陵", "南孔庙", "龙门石窟", "颐和园"]) else ""
            print(f"  #{i+1:2d}  score={score:.4f}  {cid[:40]}  {text}...{marker}")
    except Exception as e:
        print(f"  [错误] {e}")
