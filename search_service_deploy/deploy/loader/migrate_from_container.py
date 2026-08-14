import json, time, hashlib, os, sys

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
BGE_MODEL_PATH = "/opt/search_service/models/bge-m3"
PYTHON = "/mnt/userhome/liangyanjie/anaconda3/bin/python"

BATCH_SIZE = 8

def encode(encoder, texts):
    if encoder is None:
        return None
    emb = encoder.encode(texts, return_dense=True)
    return emb["dense_vecs"].tolist()

def stable_id(text: str) -> int:
    h = hashlib.md5(text.encode()).hexdigest()
    return int(h[:15], 16) % (2**63)

def upsert_batch(collection, points):
    import httpx
    body = {"points": points}
    r = httpx.put(
        f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection}/points?wait=true",
        json=body, timeout=120.0
    )
    r.raise_for_status()
    return r.json()

def load_and_load_chunks():
    """从容器里读 chunks 并灌入 Qdrant"""
    import subprocess

    # docker cp 从容器里拉文件到宿主机
    local_chunks = "/tmp/all_chunks_chunks.json"
    subprocess.run([
        "docker", "cp",
        "mingqiang-rag-processor-run-f8ffb1fb3595:/app/RAG_DB_slim/output_chunks/all_chunks_chunks.json",
        local_chunks
    ], check=True, capture_output=True)

    with open(local_chunks, "r", encoding="utf-8") as f:
        all_docs = json.load(f)

    print(f"[load_chunks] 共 {len(all_docs)} 个文档")

    # 加载 BGE 编码器
    sys.path.insert(0, "/opt/search_service/service")
    from bge_encoder import load_bge_encoder
    encoder = load_bge_encoder()
    print(f"[load_chunks] 编码器: {encoder.__class__.__name__}")

    # 收集所有需要编码的文本
    all_items = []
    for doc in all_docs:
        doc_id = doc["doc_id"]
        file_name = doc["file_name"]
        chunks_list = doc.get("chunks", [])
        for sub in chunks_list:
            chunk_text = sub.get("chunk_text", "").strip()
            if not chunk_text:
                continue
            chunk_id = f"{doc_id}_{chunks_list.index(sub):03d}"
            all_items.append({
                "doc_id": doc_id,
                "file_name": file_name,
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
            })

    print(f"[load_chunks] 共 {len(all_items)} 个子 chunk")

    # 批量编码
    t0 = time.time()
    total = 0
    for i in range(0, len(all_items), BATCH_SIZE):
        batch = all_items[i:i+BATCH_SIZE]
        texts = [it["chunk_text"] for it in batch]
        vecs = encode(encoder, texts)
        if vecs is None:
            print(f"[ERROR] 批次 {i//BATCH_SIZE+1} 编码失败")
            continue
        points = []
        for item, vec in zip(batch, vecs):
            p = {
                "id": stable_id(item["chunk_id"]),
                "vector": vec,
                "payload": {
                    "chunk_id": item["chunk_id"],
                    "doc_id": item["doc_id"],
                    "file_name": item["file_name"],
                    "chunk_text": item["chunk_text"],
                }
            }
            points.append(p)
        upsert_batch("unified_corpus", points)
        total += len(points)
        print(f"  batch {i//BATCH_SIZE+1}: upsert {len(points)} -> total {total}")

    elapsed = time.time() - t0
    print(f"[load_chunks] 完成: {total} 条, {elapsed:.1f}s")


def load_and_load_tags():
    """从容器里读 tags 并灌入 Qdrant"""
    import subprocess

    local_tags = "/tmp/tags_output.json"
    subprocess.run([
        "docker", "cp",
        "mingqiang-rag-processor-run-f8ffb1fb3595:/app/RAG_DB_slim/experiment_data/tags_output.json",
        local_tags
    ], check=True, capture_output=True)

    with open(local_tags, "r", encoding="utf-8") as f:
        tags_dict = json.load(f)

    print(f"[load_tags] 共 {len(tags_dict)} 个文件的标签")

    # 加载 BGE 编码器
    sys.path.insert(0, "/opt/search_service/service")
    from bge_encoder import load_bge_encoder
    encoder = load_bge_encoder()
    print(f"[load_tags] 编码器: {encoder.__class__.__name__}")

    # 构建 tag 列表
    all_tags = []
    for file_name, dim_map in tags_dict.items():
        if not isinstance(dim_map, dict):
            continue
        for dim_name, tag_list in dim_map.items():
            for tag_name in tag_list:
                if not tag_name or not isinstance(tag_name, str):
                    continue
                all_tags.append({
                    "file_name": file_name,
                    "dim_name": dim_name,
                    "tag_name": tag_name,
                    "tag_text": f"{dim_name}:{tag_name}",
                })

    print(f"[load_tags] 共 {len(all_tags)} 个 tag 维度条目")

    # 批量编码
    t0 = time.time()
    total = 0
    for i in range(0, len(all_tags), BATCH_SIZE):
        batch = all_tags[i:i+BATCH_SIZE]
        texts = [it["tag_text"] for it in batch]
        vecs = encode(encoder, texts)
        if vecs is None:
            print(f"[ERROR] 批次 {i//BATCH_SIZE+1} 编码失败")
            continue
        points = []
        for item, vec in zip(batch, vecs):
            key = f"{item['file_name']}::{item['dim_name']}::{item['tag_name']}"
            p = {
                "id": stable_id(key),
                "vector": vec,
                "payload": {
                    "file_name": item["file_name"],
                    "dim_name": item["dim_name"],
                    "tag_name": item["tag_name"],
                    "tag_text": item["tag_text"],
                }
            }
            points.append(p)
        upsert_batch("dimension_tags", points)
        total += len(points)
        print(f"  batch {i//BATCH_SIZE+1}: upsert {len(points)} -> total {total}")

    elapsed = time.time() - t0
    print(f"[load_tags] 完成: {total} 条, {elapsed:.1f}s")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: 灌入 chunks")
    print("=" * 60)
    load_and_load_chunks()

    print()
    print("=" * 60)
    print("STEP 2: 灌入 tags")
    print("=" * 60)
    load_and_load_tags()

    print()
    print("全部完成!")
