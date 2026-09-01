# -*- coding: utf-8 -*-
"""诊断知识库实际覆盖了哪些景区"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import httpx

def qdrant_scroll(collection, limit=1000, offset=None):
    body = {"limit": limit, "with_payload": True, "with_vector": False}
    if offset:
        body["offset"] = offset
    r = httpx.post(f"http://127.0.0.1:6333/collections/{collection}/points/scroll",
        json=body, timeout=15)
    data = r.json()
    return data.get("result", {}).get("points", []), data.get("result", {}).get("next_page_offset")

# 扫描 rag_chunks，提取所有景区前缀
print("=== 扫描 rag_chunks，提取景区分布 ===")
all_pts = []
offset = None
while True:
    pts, offset = qdrant_scroll("rag_chunks", limit=1000, offset=offset)
    all_pts.extend(pts)
    if not offset:
        break

from collections import Counter
prefixes = Counter()
for p in all_pts:
    cid = (p.get("payload") or {}).get("chunk_id", "")
    # 格式通常是 "景区名-文件类型_编号_sub_编号"
    parts = cid.split("_")
    if parts:
        # 取第一段，可能是"景区名-文件类型"
        prefix = parts[0]
        # 提取景区名（去掉最后一段文件类型）
        import re
        match = re.match(r"(.+?)-\w+$", prefix)
        if match:
            spot = match.group(1)
        else:
            spot = prefix
        prefixes[spot] += 1

print(f"总 Chunk 数: {len(all_pts)}")
print(f"独立景区数: {len(prefixes)}")
print("\n各景区 Chunk 数量分布:")
for spot, cnt in sorted(prefixes.items(), key=lambda x: -x[1]):
    print(f"  {spot}: {cnt} chunks")
