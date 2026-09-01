# -*- coding: utf-8 -*-
"""扫描 rag_chunks，建立景区名索引"""
import sys, re
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

# 扫描
all_pts = []
offset = None
while True:
    pts, offset = qdrant_scroll("rag_chunks", limit=1000, offset=offset)
    all_pts.extend(pts)
    if not offset:
        break

print(f"总 Chunk 数: {len(all_pts)}")

# 格式: "景区名_编号_sub_编号" 或 "景区名(作者)_编号_sub_编号"
# 景区名 = 第一个 "_" 之前的内容（但要排除文件名中的下划线）
# 策略：找最后一个 "_sub_"，前面那个 "_" 之前的就是景区名
# 实际上格式固定为：spot_name_fileid_sub_subid
# spot_name 是第一个 "_" 之前，但spot_name本身可能含下划线？

# 实际格式是: spotName_fileNum_sub_subNum
# spotName 可以包含中文、括号，但不含下划线？
# 最可靠：从末尾数，固定 _sub_ 是分隔符

def extract_spot(cid):
    """从 chunk_id 提取景区名"""
    if not cid:
        return None
    # 找最后一个 "_sub_"
    idx = cid.rfind("_sub_")
    if idx > 0:
        # 往前找到再前一个 "_"
        before = cid[:idx]
        last_underscore = before.rfind("_")
        if last_underscore > 0:
            return before[:last_underscore]
        return before
    return None

spot_to_chunks = {}
spot_prefixes = set()

for p in all_pts:
    cid = (p.get("payload") or {}).get("chunk_id", "")
    if not cid:
        continue
    spot = extract_spot(cid)
    if not spot:
        continue
    if spot not in spot_to_chunks:
        spot_to_chunks[spot] = []
    spot_to_chunks[spot].append(cid)
    spot_prefixes.add(spot)

print(f"\n独立景区数: {len(spot_prefixes)}")
print("\n景区列表:")
for spot, chunks in sorted(spot_to_chunks.items(), key=lambda x: -len(x[1])):
    print(f"  [{len(chunks):4d}] {spot}")

# 验证：打印几个 chunk_id 样本
print("\n样本 chunk_id:")
for cid in [all_pts[0].get("payload", {}).get("chunk_id", "")] + \
           [(p.get("payload") or {}).get("chunk_id", "") for p in all_pts[100:105]]:
    print(f"  {cid}  →  {extract_spot(cid)}")

# 保存
import pickle
output = {
    "spot_to_chunks": spot_to_chunks,
    "spot_prefixes": list(spot_prefixes),
}
with open("experiment_data/spot_index.pkl", "wb") as f:
    pickle.dump(output, f)

print(f"\n索引已保存: {len(spot_prefixes)} 个景区, {len(all_pts)} 个 Chunk")
