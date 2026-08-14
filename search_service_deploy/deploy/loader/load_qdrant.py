# 灌库工具
# 装包：pip install qdrant-client numpy
# 用法：python load_qdrant.py --input /data/chunks.jsonl ...

import argparse
import json
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models


def embed_texts(texts, model_dir: str):
    """用 BGE-M3 编码文本列表。优先 FlagEmbedding，回退 sentence-transformers。"""
    try:
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel(model_dir, use_fp16=False, device="cpu")
        out = model.encode(texts, return_dense=True, batch_size=8)
        return out["dense_vecs"].tolist()
    except Exception:
        pass

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_dir, local_files_only=True)
        emb = model.encode(
            texts, normalize_embeddings=True,
            batch_size=8, show_progress_bar=True, convert_to_numpy=True,
        )
        return emb.tolist()
    except Exception as e:
        print(f"[ERROR] BGE-M3 加载失败: {e}", file=sys.stderr)
        return None


def ensure_collection(client, name, vector_dim, distance="Cosine"):
    if client.collection_exists(name):
        print(f"  [OK] collection 已存在: {name}")
        return
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=vector_dim,
            distance=models.Distance(distance.upper()),
        ),
        hnsw_config=models.HnswConfigDiff(
            m=16, ef_construct=512,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            default_segment_number=2,
        ),
    )
    print(f"  [OK] 已创建 collection: {name}")


def load_chunks_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def upsert_chunks(client, collection, chunks, vectors):
    points = []
    for c, v in zip(chunks, vectors):
        cid = c.get("chunk_id")
        if not cid:
            continue
        # 浮点 id
        try:
            point_id = abs(hash(cid)) % (2**63)
        except Exception:
            point_id = cid
        payload = {
            "chunk_id": cid,
            "chunk_text": c.get("chunk_text", ""),
            "doc_title": c.get("doc_title", ""),
            "chunk_gen_title": c.get("chunk_gen_title", ""),
        }
        points.append(models.PointStruct(id=point_id, vector=v, payload=payload))

    # 分批 256
    BATCH = 256
    for i in range(0, len(points), BATCH):
        chunk = points[i:i+BATCH]
        client.upsert(collection_name=collection, points=chunk, wait=True)
    print(f"  [OK] 已 upsert {len(points)} 条到 {collection}")


def load_dim_tags(client, collection, dim_tags):
    """dim_tags: list[dict] {dim_name, tag_name, chunk_ids, tag_text...}"""
    points = []
    for t in dim_tags:
        key = f"{t['dim_name']}::{t['tag_name']}"
        point_id = abs(hash(key)) % (2**63)
        payload = {
            "dim_name": t.get("dim_name", ""),
            "tag_name": t.get("tag_name", ""),
            "chunk_ids": t.get("chunk_ids", []),
            "tag_text": t.get("tag_text", ""),
        }
        # 使用 tag_text 编码向量（兼容 rag_chunks 的同维向量名空间）
        text = f"{t['dim_name']}:{t['tag_name']} - {t.get('tag_text','')}"
        # 占位向量，等下方 encode
        points.append({"id": point_id, "payload": payload, "text": text})

    if not points:
        return
    texts = [p["text"] for p in points]
    vectors = embed_texts(texts, os.getenv("BGE_MODEL_PATH", "/models/bge-m3"))
    if not vectors:
        print("[ERROR] dim_tags 编码失败")
        return
    out = [
        models.PointStruct(id=p["id"], vector=v, payload=p["payload"])
        for p, v in zip(points, vectors)
    ]
    client.upsert(collection_name=collection, points=out, wait=True)
    print(f"  [OK] 已 upsert {len(out)} 条 dim_tags 到 {collection}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="chunks.jsonl")
    ap.add_argument("--chunk-collection", default="unified_corpus")
    ap.add_argument("--dim-tags-collection", default="dimension_tags")
    ap.add_argument("--vector-dim", type=int, default=1024)
    ap.add_argument("--bge-model", default="/models/bge-m3")
    ap.add_argument("--host", default=os.getenv("QDRANT_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    args = ap.parse_args()

    print(f"连接 Qdrant: {args.host}:{args.port}")
    client = QdrantClient(host=args.host, port=args.port)

    print(f"读取数据: {args.input}")
    items = load_chunks_jsonl(args.input)
    print(f"  共 {len(items)} 条")

    print(f"[1/3] 创建 collections...")
    ensure_collection(client, args.chunk_collection, args.vector_dim)
    ensure_collection(client, args.dim_tags_collection, args.vector_dim)

    print(f"[2/3] 编码 chunks...")
    texts = [c.get("chunk_text", "") for c in items]
    vectors = embed_texts(texts, args.bge_model)
    if not vectors:
        sys.exit(1)

    print(f"[3/3] upsert chunks...")
    upsert_chunks(client, args.chunk_collection, items, vectors)

    print("\n[OK] 全部完成")


if __name__ == "__main__":
    main()
