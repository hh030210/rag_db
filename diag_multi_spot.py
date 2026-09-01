# -*- coding: utf-8 -*-
"""扫描多景区知识覆盖了哪些景区"""
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

all_pts = []
offset = None
while True:
    pts, offset = qdrant_scroll("rag_chunks", limit=1000, offset=offset)
    all_pts.extend(pts)
    if not offset:
        break

# 从 chunk_id 提取景区名（多景区知识的文件编号格式）
def extract_spot_v2(cid):
    if not cid:
        return None
    # "多景区知识_001_sub_005" → spot="多景区知识", 子chunk
    # 其实景区名在chunk_text开头
    return None

# 扫描多景区知识的chunk，提取景区名
spot_mentions = {}  # spot_name → count
spot_mention_pattern = re.compile(r"^([^\s，。、：]+(?:山|湖|陵|园|庙|窟|寺|城|瀑|峡|峰|阁|塔|岛|镇|村)]+)")

for p in all_pts:
    cid = (p.get("payload") or {}).get("chunk_id", "")
    text = (p.get("payload") or {}).get("chunk_text", "")[:200]
    if not cid.startswith("多景区知识"):
        continue
    # 从文本中提取景区名
    # 常见景区名：雁荡山、西湖、丽江、普陀山、云和梯田等
    known_spots = [
        "雁荡山", "杭州西湖", "西湖", "丽江古城", "普陀山", "云和梯田",
        "西溪湿地", "龙门石窟", "颐和园", "明十三陵", "天门山",
        "张家界", "乌镇", "南浔", "西塘", "千岛湖", "天目山", "莫干山",
        "楠溪江", "象山影视城", "鲁迅故里", "沈园", "兰亭", "大佛寺",
        "大鹿岛", "洞头", "南麂岛", "江心屿", "泰顺廊桥", "石门洞",
        "南孔庙", "孔氏南宗家庙", "龙游石窟", "烂柯山", "钱江源",
    ]
    found_spots = []
    for spot in known_spots:
        if spot in text:
            found_spots.append(spot)
    for s in found_spots:
        spot_mentions[s] = spot_mentions.get(s, 0) + 1

print("=== 多景区知识 Chunk 中的景区提及 ===")
for spot, cnt in sorted(spot_mentions.items(), key=lambda x: -x[1]):
    print(f"  {cnt:3d}次  {spot}")

# 另一种方法：从doc_title提取
from collections import Counter
doc_titles = Counter()
for p in all_pts:
    cid = (p.get("payload") or {}).get("chunk_id", "")
    title = (p.get("payload") or {}).get("doc_title", "")
    if cid.startswith("多景区知识") and title:
        doc_titles[title] += 1

print("\n=== 多景区知识 DocTitle 分布 ===")
for title, cnt in doc_titles.most_common(20):
    print(f"  [{cnt:3d}] {title}")

# 直接看一批多景区知识的 chunk 内容
print("\n=== 多景区知识 Chunk 样本（前5条文本）===")
count = 0
for p in all_pts:
    cid = (p.get("payload") or {}).get("chunk_id", "")
    if not cid.startswith("多景区知识"):
        continue
    text = (p.get("payload") or {}).get("chunk_text", "")[:200].replace("\n", " ")
    print(f"  [{cid}] {text}")
    count += 1
    if count >= 5:
        break
