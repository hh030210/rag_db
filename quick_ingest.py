# -*- coding: utf-8 -*-
"""
快速入库脚本: 将 test_data 文档分块并写入 Qdrant
- 简单按段落/固定长度分块，不依赖 LLM
- 直接写入 Qdrant rag_chunks collection
"""
import sys, re, time, json
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from pathlib import Path

# ── 配置 ──────────────────────────────────────────────
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = "6333"
COLLECTION   = "rag_chunks"
BATCH_SIZE   = 50          # 每批写入 Qdrant 的点数
CHUNK_LEN    = 600         # 每个 chunk 的最大字符数
CHUNK_OVERLAP = 50         # 相邻 chunk 重叠字符数
# ──────────────────────────────────────────────────────

import httpx

client = httpx.Client(base_url=f"http://{QDRANT_HOST}:{QDRANT_PORT}", timeout=60)

# 加载 BGE encoder
sys.path.insert(0, str(Path(__file__).parent))
from retrieval_fusion_eval import _load_bge_encoder, _encode_query
encoder = _load_bge_encoder()

def load_md_files(folder: Path):
    """加载所有 .md 文件"""
    files = []
    for f in sorted(folder.glob("*.md")):
        content = f.read_text(encoding="utf-8")
        files.append({"path": f, "name": f.stem, "content": content})
    return files

def chunk_text(text: str, max_len=CHUNK_LEN, overlap=CHUNK_OVERLAP):
    """简单分块：按段落拆分，合并到 max_len"""
    # 先按段落拆分（保留换行）
    paras = re.split(r'\n+', text)
    paras = [p.strip() for p in paras if p.strip()]
    
    result = []
    current = ""
    for para in paras:
        if len(current) + len(para) + 1 <= max_len:
            current += ("\n" if current else "") + para
        else:
            if current:
                result.append(current)
            # 如果单段落超过 max_len，按固定长度截断
            if len(para) > max_len:
                for i in range(0, len(para), max_len - overlap):
                    result.append(para[i:i + max_len])
                current = ""
            else:
                current = para
    if current:
        result.append(current)
    return result

def get_collection_info():
    r = client.get(f"/collections/{COLLECTION}")
    if r.status_code == 200:
        info = r.json().get("result", {})
        return info.get("points_count", 0)
    return 0

def upsert_points(vectors_data, collection=COLLECTION):
    """批量写入 Qdrant（使用 /points/upsert）"""
    body = {"points": vectors_data}
    r = client.put(f"/collections/{collection}/points/upsert", json=body)
    if r.status_code not in (200, 201):
        print(f"  [错误] upsert 失败: {r.status_code} {r.text[:200]}")
        return False
    return True

def generate_doc_id(doc_name: str, genre: str) -> str:
    """从文件名生成 doc_id"""
    # 例如 "南孔庙-景区介绍" → "南孔庙_景区介绍"
    return doc_name.replace("-", "_")

def main():
    test_data_dir = Path("data_input/test_data")
    if not test_data_dir.exists():
        print(f"[错误] 目录不存在: {test_data_dir}")
        return

    print(f"[Info] 找到 {len(list(test_data_dir.glob('*.md')))} 个文件")
    
    files = load_md_files(test_data_dir)
    all_chunks = []
    
    for fdata in files:
        fname = fdata["name"]  # e.g. "南孔庙-景区介绍"
        content = fdata["content"]
        
        # 判断类型（景区介绍/运营信息/运营介绍）
        if "运营信息" in fname or "运营介绍" in fname:
            genre = "运营信息"
        elif "景区介绍" in fname or "景点介绍" in fname:
            genre = "景区介绍"
        else:
            genre = "通用"
        
        chunks = chunk_text(content)
        doc_id = generate_doc_id(fname, genre)
        
        print(f"  [{fname}] → {len(chunks)} chunks")
        
        for i, txt in enumerate(chunks):
            cid = f"{doc_id}_sub_{i:03d}"
            all_chunks.append({
                "doc_id": doc_id,
                "doc_title": f"{fname}.md",
                "genre": genre,
                "chunk_id": cid,
                "chunk_text": txt,
            })
    
    print(f"\n总计: {len(all_chunks)} chunks 待入库")
    current_count = get_collection_info()
    print(f"当前 rag_chunks 点数: {current_count}")
    
    # 批量写入
    vectors_batch = []
    start_id = current_count + 1  # 从现有数据之后开始编 ID
    total_written = 0
    
    print(f"\n开始写入 Qdrant (批量大小={BATCH_SIZE})...")
    for i, chunk in enumerate(all_chunks):
        # 编码
        vec = _encode_query(encoder, chunk["chunk_text"])
        if vec is None:
            print(f"  [错误] 向量编码失败: {chunk['chunk_id']}")
            continue

        pt = {
            "id": start_id + i,  # 避免与现有 ID 冲突
            "vector": {"chunk_text_vec": vec},  # 命名向量格式
            "payload": {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "doc_title": chunk["doc_title"],
                "genre": chunk["genre"],
                "chunk_text": chunk["chunk_text"],
                "chunk_gen_title": chunk["doc_id"],
                "chunk_text_full": chunk["chunk_text"],
            }
        }
        vectors_batch.append(pt)
        
        if len(vectors_batch) >= BATCH_SIZE:
            ok = upsert_points(vectors_batch)
            if ok:
                total_written += len(vectors_batch)
                print(f"  已写入 {total_written}/{len(all_chunks)} chunks")
            vectors_batch = []
    
    # 最后一批
    if vectors_batch:
        ok = upsert_points(vectors_batch)
        if ok:
            total_written += len(vectors_batch)
            print(f"  已写入 {total_written}/{len(all_chunks)} chunks")
    
    new_count = get_collection_info()
    print(f"\n{'='*50}")
    print(f"入库完成! 新增 {total_written} chunks")
    print(f"rag_chunks 总点数: {new_count} (之前 {current_count})")
    
    # 验证：检查入库的景区
    print(f"\n验证入库情况...")
    offset = None
    new_spots = {}
    checked = 0
    while checked < new_count:
        body = {"limit": 200, "with_payload": True, "with_vector": False, "offset": offset} if offset else {"limit": 200, "with_payload": True, "with_vector": False}
        r = client.post(f"/collections/{COLLECTION}/points/scroll", json=body)
        pts = r.json().get("result", {}).get("points", [])
        if not pts:
            break
        for p in pts:
            cid = (p.get("payload") or {}).get("chunk_id", "")
            for spot in ["龙门石窟", "少林寺", "颐和园", "张家界", "西湖", "南孔庙", "明十三陵"]:
                if spot in cid:
                    new_spots[spot] = new_spots.get(spot, 0) + 1
        checked += len(pts)
        offset = r.json().get("result", {}).get("next_page_offset")
        if not offset:
            break
    
    print(f"\n景区入库统计:")
    for spot, cnt in sorted(new_spots.items(), key=lambda x: -x[1]):
        print(f"  {spot}: {cnt} chunks ✅")
    
    all_spots_in = all(spot in new_spots for spot in ["龙门石窟", "少林寺", "颐和园", "张家界", "西湖", "南孔庙", "明十三陵"])
    if all_spots_in:
        print("\n🎉 所有 test_data 景区已入库!")
    else:
        missing = [s for s in ["龙门石窟", "少林寺", "颐和园", "张家界", "西湖", "南孔庙", "明十三陵"] if s not in new_spots]
        print(f"\n⚠ 未入库: {missing}")


if __name__ == "__main__":
    main()
