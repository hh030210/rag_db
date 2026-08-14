# -*- coding: utf-8 -*-
"""
将 test_data 的 chunks 快速入库到 Qdrant rag_chunks
不依赖 pipeline，直接读取已有的 chunks 文件并写入
"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from pathlib import Path

# ── 配置 ──────────────────────────────────────────────
QDRANT_HOST  = "127.0.0.1"
QDRANT_PORT  = "6333"
COLLECTION   = "rag_chunks"
BATCH_SIZE   = 50
# ──────────────────────────────────────────────────────

import httpx
client = httpx.Client(base_url=f"http://{QDRANT_HOST}:{QDRANT_PORT}", timeout=60)

# 加载 BGE encoder
sys.path.insert(0, str(Path(__file__).parent))
from retrieval_fusion_eval import _load_bge_encoder, _encode_query
encoder = _load_bge_encoder()

test_spots = ["龙门石窟", "少林寺", "颐和园", "张家界", "西湖", "南孔庙", "明十三陵"]

def load_test_chunks():
    """加载所有 test_data 的 chunks"""
    all_chunks = []
    for spot in test_spots:
        for suffix in ["景区介绍", "景点介绍", "运营信息", "运营介绍"]:
            base_name = f"{spot}-{suffix}"
            chunks_f = Path(f"output_chunks/{base_name}_chunks.json")
            if not chunks_f.exists():
                continue
            with open(chunks_f, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                genre = data[0].get('genre', '通用')
                for sub in data:
                    for idx, chunk in enumerate(sub.get('chunks', [])):
                        txt = chunk.get('chunk_text', '').strip()
                        if txt:
                            all_chunks.append({
                                'doc_id': base_name,
                                'chunk_id': f"{base_name}_sub_{idx:03d}",
                                'chunk_text': txt,
                                'genre': genre,
                            })
    return all_chunks

def get_collection_count():
    r = client.get(f"/collections/{COLLECTION}")
    if r.status_code == 200:
        return r.json().get("result", {}).get("points_count", 0)
    return 0

def upsert_batch(points):
    """批量写入 Qdrant"""
    body = {"points": points}
    r = client.put(f"/collections/{COLLECTION}/points", json=body)
    if r.status_code not in (200, 201):
        print(f"  [错误] upsert: {r.status_code} {r.text[:300]}")
        return False
    return True

def main():
    print("=== 加载 test_data chunks ===")
    all_chunks = load_test_chunks()
    print(f"共 {len(all_chunks)} 个 chunks")

    # 按景区统计
    spot_counts = {}
    for c in all_chunks:
        for s in test_spots:
            if s in c['doc_id']:
                spot_counts[s] = spot_counts.get(s, 0) + 1
                break
    for s, n in sorted(spot_counts.items()):
        print(f"  {s}: {n} chunks")

    current_count = get_collection_count()
    print(f"\n当前 rag_chunks 点数: {current_count}")

    start_id = current_count + 1
    total_written = 0
    batch = []

    print(f"\n=== 写入 Qdrant (批量={BATCH_SIZE}) ===")
    for i, chunk in enumerate(all_chunks):
        vec = _encode_query(encoder, chunk['chunk_text'])
        if vec is None:
            print(f"  [跳过] 向量编码失败: {chunk['chunk_id']}")
            continue

        pt = {
            "id": start_id + i,
            "vector": {"chunk_text_vec": vec},
            "payload": {
                "chunk_id": chunk['chunk_id'],
                "doc_id": chunk['doc_id'],
                "doc_title": f"{chunk['doc_id']}.md",
                "genre": chunk['genre'],
                "chunk_text": chunk['chunk_text'],
                "chunk_gen_title": chunk['doc_id'],
                "chunk_text_full": chunk['chunk_text'],
            }
        }
        batch.append(pt)

        if len(batch) >= BATCH_SIZE:
            ok = upsert_batch(batch)
            if ok:
                total_written += len(batch)
                print(f"  写入 {total_written}/{len(all_chunks)} ({total_written*100//len(all_chunks)}%)")
            batch = []

    if batch:
        ok = upsert_batch(batch)
        if ok:
            total_written += len(batch)
            print(f"  写入 {total_written}/{len(all_chunks)} (100%)")

    new_count = get_collection_count()
    print(f"\n{'='*50}")
    print(f"入库完成! 新增 {total_written} chunks")
    print(f"rag_chunks 总点数: {new_count} (之前 {current_count})")

    # 验证
    print(f"\n=== 验证入库 ===")
    offset = None
    verified = {}
    checked = 0
    while checked < new_count:
        body = {"limit": 500, "with_payload": True, "with_vector": False}
        if offset:
            body["offset"] = offset
        r = client.post(f"/collections/{COLLECTION}/points/scroll", json=body)
        pts = r.json().get("result", {}).get("points", [])
        if not pts:
            break
        for p in pts:
            cid = (p.get("payload") or {}).get("chunk_id", "")
            for s in test_spots:
                if s in cid:
                    verified[s] = verified.get(s, 0) + 1
        checked += len(pts)
        offset = r.json().get("result", {}).get("next_page_offset")
        if not offset:
            break

    for s, n in sorted(verified.items(), key=lambda x: -x[1]):
        print(f"  {s}: {n} chunks")

    all_ok = all(s in verified for s in test_spots)
    if all_ok:
        print("\n 所有 test_data 景区已入库成功!")
    else:
        missing = [s for s in test_spots if s not in verified]
        print(f"\n 未入库: {missing}")

if __name__ == "__main__":
    main()
