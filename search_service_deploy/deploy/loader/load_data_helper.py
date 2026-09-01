# 灌库 helper（一次性的）
# 用法：
#   /mnt/userhome/liangyanjie/anaconda3/bin/python load_data_helper.py chunks.jsonl unified_corpus
#   /mnt/userhome/liangyanjie/anaconda3/bin/python load_data_helper.py dim_tags.jsonl dimension_tags

import sys
import os
import json
import time
import hashlib
import argparse

import httpx
import numpy as np


QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
VECTOR_DIM = 1024
CHUNK_COLLECTION = "unified_corpus"
DIM_TAGS_COLLECTION = "dimension_tags"
BGE_MODEL_PATH = "/opt/search_service/models/bge-m3"
PYTHON_EXE = "/mnt/userhome/liangyanjie/anaconda3/bin/python"

# 在 conda 环境跑，并切到 BGE 模型路径
ENCODER = None


def load_encoder():
    global ENCODER
    if ENCODER is not None:
        return ENCODER
    sys.path.insert(0, "/opt/search_service/service")
    from bge_encoder import load_bge_encoder
    ENCODER = load_bge_encoder()
    return ENCODER


def encode_texts(texts):
    enc = load_encoder()
    if enc is None:
        return None
    if enc.__class__.__name__ == "_FlagProxy":
        emb = enc.encode(texts, return_dense=True)
        return np.array(emb["dense_vecs"]).tolist()
    elif enc.__class__.__name__ in ("BGEM3FlagModel", "M3Embedder"):
        emb = enc.encode(texts, return_dense=True)
        return np.array(emb["dense_vecs"]).tolist()
    else:
        from sentence_transformers import SentenceTransformer
        if not hasattr(enc, "_model"):
            return None
        emb = enc._model.encode(texts, normalize_embeddings=True, batch_size=4, show_progress_bar=True, convert_to_numpy=True)
        return emb.tolist()


def stable_id(text: str) -> int:
    h = hashlib.md5(text.encode()).hexdigest()
    return int(h[:15], 16) % (2**63)


def make_chunk_point(item, vector):
    cid = item.get("chunk_id")
    payload = {
        "chunk_id": cid,
        "chunk_text": item.get("chunk_text", ""),
        "doc_title": item.get("doc_title", ""),
        "chunk_gen_title": item.get("chunk_gen_title", ""),
    }
    return {
        "id": stable_id(cid),
        "vector": vector,
        "payload": payload,
    }


def make_dim_tag_point(item, vector):
    payload = {
        "dim_name": item.get("dim_name", ""),
        "tag_name": item.get("tag_name", ""),
        "tag_text": item.get("tag_text", ""),
        "chunk_ids": item.get("chunk_ids", []),
    }
    key = f"{payload['dim_name']}::{payload['tag_name']}"
    return {"id": stable_id(key), "vector": vector, "payload": payload}


def upsert_batch(collection, points):
    """PUT /collections/{name}/points?wait=true"""
    body = {"points": points}
    r = httpx.put(f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection}/points?wait=true",
                  json=body, timeout=120.0)
    r.raise_for_status()
    return r.json()


def load_jsonl(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="JSONL 文件")
    ap.add_argument("collection", help="目标 collection")
    ap.add_argument("--kind", choices=["chunk", "dim_tag"], default="chunk")
    args = ap.parse_args()

    print(f"[load] input={args.input}, collection={args.collection}, kind={args.kind}")

    items = load_jsonl(args.input)
    print(f"[load] 共 {len(items)} 条")
    if not items:
        return

    print(f"[load] 加载编码器: {BGE_MODEL_PATH}")
    enc = load_encoder()
    print(f"[load] 编码器类: {enc.__class__.__name__ if enc else 'None'}")

    print(f"[load] 编码 {len(items)} 条文本...")
    if args.kind == "chunk":
        texts = [it.get("chunk_text", "") for it in items]
    else:
        texts = [f"{it.get('dim_name','')}:{it.get('tag_name','')} - {it.get('tag_text','')}" for it in items]

    vectors = encode_texts(texts)
    if vectors is None:
        print("[ERROR] 编码失败")
        return
    print(f"[load] 编码完成，dim={len(vectors[0])}")

    BATCH = 16  # 受限于 server httpx timeout + chunk 大小
    total = 0
    t0 = time.time()
    for i in range(0, len(items), BATCH):
        batch = items[i:i+BATCH]
        batch_vec = vectors[i:i+BATCH]
        if args.kind == "chunk":
            points = [make_chunk_point(it, v) for it, v in zip(batch, batch_vec)]
        else:
            points = [make_dim_tag_point(it, v) for it, v in zip(batch, batch_vec)]

        # 验证
        if points and len(points[0]["vector"]) != VECTOR_DIM:
            print(f"[ERROR] 向量维度 {len(points[0]['vector'])} != {VECTOR_DIM}")
            return

        result = upsert_batch(args.collection, points)
        status = result.get("status", "?")
        total += len(points)
        print(f"  [batch {i//BATCH+1}] upsert {len(points)} -> status={status}")

    elapsed = time.time() - t0
    print(f"[load] 完成: {total} 条，{elapsed:.2f}s")


if __name__ == "__main__":
    main()
