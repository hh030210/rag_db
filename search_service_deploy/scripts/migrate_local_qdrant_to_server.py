"""
migrate_local_qdrant_to_server.py

将本地 Qdrant 的 unified_corpus 和 dimension_tags 全量迁移到服务器。
流程：
1. 从本地 Qdrant scroll 读取所有 points（含向量）
2. 上传到服务器 Qdrant
"""

import json, sys, time
from pathlib import Path

LOCAL_QDRANT_HOST = "127.0.0.1"
LOCAL_QDRANT_PORT = 6333
SERVER_QDRANT_HOST = os.getenv("SERVER_QDRANT_HOST", "127.0.0.1")
SERVER_QDRANT_PORT = int(os.getenv("SERVER_QDRANT_PORT", "6333"))

BATCH_SIZE = 100

# ============================================================
# Step 1: 从本地导出
# ============================================================

def export_collection(client, coll_name):
    """scroll 全量数据，返回 [(id, vector, payload), ...]"""
    all_points = []
    offset = None
    total = 0
    while True:
        result = client.scroll(
            collection_name=coll_name,
            offset=offset,
            limit=BATCH_SIZE,
            with_vectors=True,
        )
        pts = result[0]
        next_page_offset = result[1]
        for pt in pts:
            vec_dict = getattr(pt, 'vector', None) or {}
            # 提取 chunk_text_vec 向量
            vec = vec_dict.get('chunk_text_vec')
            all_points.append({
                "id": pt.id,
                "vector": vec,
                "payload": pt.payload or {},
            })
            total += 1
        if next_page_offset is None:
            break
        offset = next_page_offset
        print(f"  {coll_name}: {total} points fetched...", flush=True)
    print(f"  {coll_name}: 导出完成，共 {total} 条")
    return all_points


# ============================================================
# Step 2: 上传到服务器
# ============================================================

def upsert_batch(server_client, coll_name, points):
    from qdrant_client.models import PointStruct
    qdrant_pts = []
    for p in points:
        vec_dict = p["vector"]
        # 如果是 list of floats，保持 dict 格式
        qdrant_pts.append(PointStruct(
            id=p["id"],
            vector=vec_dict,
            payload=p["payload"],
        ))
    operation_info = server_client.upsert(
        collection_name=coll_name,
        points=qdrant_pts,
        wait=True,
    )
    return operation_info


def ensure_collection(server_client, coll_name, vec_size=1024):
    """确保 collection 存在且配置正确"""
    from qdrant_client.models import Distance, VectorParams

    collections = [c.name for c in server_client.get_collections().collections]
    if coll_name in collections:
        # 检查维度
        info = server_client.get_collection(coll_name)
        existing_vec_cfg = info.config.params.vectors
        if existing_vec_cfg and 'chunk_text_vec' in existing_vec_cfg:
            existing_dim = existing_vec_cfg['chunk_text_vec'].size
            if existing_dim != vec_size:
                print(f"[WARN] {coll_name} 已有向量维度 {existing_dim}，期望 {vec_size}，将覆盖")
                server_client.delete_collection(coll_name)
                collections.remove(coll_name)

    if coll_name not in collections:
        print(f"  创建 collection: {coll_name}")
        server_client.create_collection(
            collection_name=coll_name,
            vectors_config={
                "chunk_text_vec": VectorParams(size=vec_size, distance=Distance.COSINE)
            },
        )
        print(f"  创建完成: {coll_name}")
    else:
        print(f"  collection 已存在: {coll_name}")


def main():
    from qdrant_client import QdrantClient

    # 连接本地和服务器
    print("连接本地 Qdrant...")
    local = QdrantClient(host=LOCAL_QDRANT_HOST, port=LOCAL_QDRANT_PORT)

    print("连接服务器 Qdrant...")
    server = QdrantClient(
        host=SERVER_QDRANT_HOST,
        port=SERVER_QDRANT_PORT,
        timeout=60.0,
        check_compatibility=False,
    )

    collections_to_migrate = ["unified_corpus", "dimension_tags"]

    for coll in collections_to_migrate:
        print(f"\n{'='*60}")
        print(f"迁移: {coll}")
        print(f"{'='*60}")

        # Step 1: 导出
        print(f"[1/3] 从本地导出 {coll}...")
        points = export_collection(local, coll)
        if not points:
            print(f"  {coll} 无数据，跳过")
            continue

        # Step 2: 确保 collection 存在
        print(f"[2/3] 确保服务器 collection 存在...")
        ensure_collection(server, coll, vec_size=1024)

        # Step 3: 清空旧数据后灌入
        print(f"[3/3] 清空旧数据 + 灌入新数据...")
        try:
            server.delete_collection(coll)
            print(f"  已删除旧 {coll}")
        except Exception as e:
            print(f"  删除旧 collection 失败(可能不存在): {e}")

        # 重新创建（维度必须匹配）
        from qdrant_client.models import Distance, VectorParams
        server.create_collection(
            collection_name=coll,
            vectors_config={
                "chunk_text_vec": VectorParams(size=1024, distance=Distance.COSINE)
            },
        )
        print(f"  已重建 {coll}")

        # 批量上传
        t0 = time.time()
        total_uploaded = 0
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i+BATCH_SIZE]
            batch_to_upload = []
            for p in batch:
                vec = p["vector"]
                if isinstance(vec, dict) and "chunk_text_vec" in vec:
                    vec = vec["chunk_text_vec"]
                batch_to_upload.append({
                    "id": p["id"],
                    "vector": {"chunk_text_vec": vec},
                    "payload": p["payload"],
                })
            upsert_batch(server, coll, batch_to_upload)
            total_uploaded += len(batch)
            print(f"  batch {i//BATCH_SIZE+1}: {len(batch)} -> total {total_uploaded}/{len(points)}", flush=True)

        elapsed = time.time() - t0
        print(f"  完成: {total_uploaded} 条, {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print("全部迁移完成!")
    print(f"{'='*60}")

    # 验证
    print("\n验证服务器数据:")
    for coll in collections_to_migrate:
        info = server.get_collection(coll)
        print(f"  {coll}: {info.points_count} points")


if __name__ == "__main__":
    main()
