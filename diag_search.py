# -*- coding: utf-8 -*-
"""诊断：明十三陵的chunk是否在rag_chunks中，以及维度检索为何没命中"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import httpx

client_base = "http://127.0.0.1:6333"

def qdrant_scroll(collection, limit=2000, offset=None):
    body = {"limit": limit, "with_payload": True, "with_vector": False}
    if offset:
        body["offset"] = offset
    r = httpx.post(f"{client_base}/collections/{collection}/points/scroll", json=body, timeout=15)
    data = r.json()
    return data.get("result", {}).get("points", []), data.get("result", {}).get("next_page_offset")

# 1. 检查 rag_chunks 中有多少明十三陵相关chunk
print("=== 1. 检查 rag_chunks 中的明十三陵 Chunk ===")
all_pts = []
offset = None
while True:
    pts, offset = qdrant_scroll("rag_chunks", limit=1000, offset=offset)
    all_pts.extend(pts)
    if not offset:
        break

msl_chunks = []
for p in all_pts:
    cid = (p.get("payload") or {}).get("chunk_id", "")
    if "明十三陵" in cid:
        msl_chunks.append(cid)

print(f"总 chunk 数: {len(all_pts)}")
print(f"明十三陵 相关 chunk 数: {len(msl_chunks)}")
if msl_chunks:
    print(f"  示例: {msl_chunks[:10]}")

# 2. 检查 dimension_tags 中有多少明十三陵相关标签
print("\n=== 2. 检查 dimension_tags 中明十三陵相关标签 ===")
all_dt = []
offset = None
while True:
    pts, offset = qdrant_scroll("dimension_tags", limit=1000, offset=offset)
    all_dt.extend(pts)
    if not offset:
        break

msl_tags = []
for p in all_dt:
    pl = p.get("payload") or {}
    tn = pl.get("tag_name", "")
    dn = pl.get("dim_name", "")
    cids = pl.get("chunk_ids") or []
    if "明十三陵" in tn or "明十三陵" in " ".join(cids):
        msl_tags.append((dn, tn, len(cids)))

print(f"dimension_tags 总点数: {len(all_dt)}")
print(f"明十三陵 相关标签数: {len(msl_tags)}")
if msl_tags:
    print(f"  示例: {msl_tags[:10]}")

# 3. 直接在 rag_chunks 上向量检索"明十三陵有哪十三个"，看Top-20
print("\n=== 3. 直接在 rag_chunks 上向量检索 Top-20 ===")
import interactive_qa as iqa
iqa._init_searchers()

# encode query
from retrieval_fusion_eval import _load_bge_encoder
encoder = _load_bge_encoder()
qvec = encoder.encode(["明十三陵有哪十三个"])[0]

r = httpx.post(f"{client_base}/collections/rag_chunks/points/query", json={
    "query": qvec,
    "limit": 20,
    "with_payload": True,
    "with_vector": False,
    "using": "chunk_text_vec"
}, timeout=15)
hits = r.json().get("result", {}).get("points", [])
for i, h in enumerate(hits):
    pl = h.get("payload") or {}
    cid = pl.get("chunk_id", "?")
    score = h.get("score", 0)
    text = (pl.get("chunk_text") or "")[:60].replace("\n", " ")
    marker = " ← 明十三陵!" if "明十三陵" in cid else ""
    print(f"  #{i+1:2d} score={score:.4f} {cid[:45]}{marker}")
    print(f"         {text}...")
