"""
rebuild_dimension_tags.py

从 unified_corpus 实际 chunk_id 重建 dimension_tags 集合。

保持 DimensionSearcher 的逻辑不变（无需改 interactive_qa.py）：
- payload 字段：tag_name, dim_name, chunk_ids
- 向量名：chunk_text_vec (1024, Cosine)
"""
import sys, time
import httpx
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 复用项目里现成的 BGE-M3 加载器
from retrieval_fusion_eval import _load_bge_encoder

UNIFIED = "unified_corpus"
DIM_TAGS = "dimension_tags"
VEC_NAME = "chunk_text_vec"
VEC_DIM = 1024

# unified_corpus 里的 13 个 _entities 维度字段
DIM_FIELDS = [
    "person_entities", "dynasty_entities", "era_entities", "place_entities",
    "culture_entities", "event_entities", "building_entities", "exhibit_entities",
    "identity_entities", "org_entities", "ritual_entities", "spot_entities",
    "work_entities",
]

QDRANT = "http://127.0.0.1:6333"


def scroll_all(collection, with_payload_keys=None, batch=1000):
    """分页 scroll 拉全量 points"""
    out, offset = [], None
    while True:
        body = {"limit": batch, "with_vector": False}
        if with_payload_keys:
            body["with_payload"] = with_payload_keys
        if offset:
            body["offset"] = offset
        page = httpx.post(f"{QDRANT}/collections/{collection}/points/scroll",
                          json=body, timeout=60).json().get("result", {})
        out.extend(page.get("points", []))
        offset = page.get("next_page_offset")
        if not offset:
            break
    return out


def recreate_collection():
    """删除并重建 dimension_tags collection"""
    print(f"\n[Step 1] 重建 {DIM_TAGS} collection ...")
    httpx.delete(f"{QDRANT}/collections/{DIM_TAGS}", timeout=30)
    time.sleep(1)
    httpx.put(f"{QDRANT}/collections/{DIM_TAGS}", json={
        "vectors": {VEC_NAME: {"size": VEC_DIM, "distance": "Cosine"}},
        "hnsw_config": {"m": 16, "ef_construct": 512},
    }, timeout=30).raise_for_status()
    print(f"  √ {DIM_TAGS} 已重建")


def build_tag_index():
    """从 unified_corpus 拉所有 chunk → 13 个维度的字符串 tag → 反向索引"""
    print(f"\n[Step 2] 从 {UNIFIED} 拉所有 chunk 的维度字段 ...")
    points = scroll_all(UNIFIED, with_payload_keys=["chunk_id"] + DIM_FIELDS)
    print(f"  √ 拿到 {len(points)} 个 chunk")

    # (dim_name, tag_name) → set(chunk_id)
    tag_to_cids = defaultdict(set)
    cid_count = 0
    for pt in points:
        payload = pt.get("payload") or {}
        cid = payload.get("chunk_id")
        if not cid:
            continue
        cid_count += 1
        for dim in DIM_FIELDS:
            tags = payload.get(dim) or []
            if not isinstance(tags, list):
                continue
            for t in tags:
                if not t or not isinstance(t, str):
                    continue
                t_norm = t.strip()
                if not t_norm:
                    continue
                tag_to_cids[(dim, t_norm)].add(cid)

    print(f"  √ 共 {len(tag_to_cids)} 个唯一 (dim, tag) 组合")
    print(f"  √ 覆盖 {cid_count} 个 chunk")
    return tag_to_cids


def encode_and_upsert(tag_to_cids, encoder):
    print(f"\n[Step 3] BGE-M3 编码所有 tag ...")
    items = list(tag_to_cids.items())
    print(f"  编码 {len(items)} 条 ...")

    texts = [f"{dim}:{tag}" for (dim, tag), _ in items]

    # BGE-M3 批量编码
    # 先用 duck typing 探测是哪一种 encoder
    try:
        probe = encoder.encode(["test"], return_dense=True)
        # FlagEmbedding: 返回 dict 含 'dense_vecs'
        if isinstance(probe, dict) and "dense_vecs" in probe:
            from_flag = True
        else:
            from_flag = False
    except TypeError:
        from_flag = False

    if from_flag:
        out = encoder.encode(texts, batch_size=32, return_dense=True, max_length=64)
        vectors = out["dense_vecs"].tolist()
    else:
        # SentenceTransformer
        embs = encoder.encode(texts, batch_size=32,
                              normalize_embeddings=True,
                              show_progress_bar=False)
        vectors = embs.tolist() if hasattr(embs, "tolist") else list(embs)

    print(f"  √ 编码完成，shape 样本 = {len(vectors)}×{len(vectors[0])}")

    # 分批 upsert
    print(f"\n[Step 4] 批量 upsert 到 {DIM_TAGS} ...")
    BATCH = 200
    for i in range(0, len(items), BATCH):
        batch_items = items[i:i + BATCH]
        batch_vecs = vectors[i:i + BATCH]
        points = []
        for j, ((dim, tag), cids) in enumerate(batch_items):
            points.append({
                "id": i + j + 1,
                "vector": {VEC_NAME: batch_vecs[j]},
                "payload": {
                    "tag_name": tag,
                    "dim_name": dim,
                    "chunk_ids": sorted(cids),
                }
            })
        httpx.put(f"{QDRANT}/collections/{DIM_TAGS}/points",
                  json={"points": points}, timeout=120)
        if (i // BATCH) % 5 == 0:
            print(f"  ... 已 upsert {i + len(batch_items)} / {len(items)}")

    print(f"  √ 全部 {len(items)} 个点已写入")


def verify():
    print(f"\n[Step 5] 验证：从 {DIM_TAGS} 抽 5 个 tag，看 chunk_ids 是否在 {UNIFIED} 命中")
    sample = scroll_all(DIM_TAGS, with_payload_keys=["tag_name", "dim_name", "chunk_ids"])[:5]
    # 拿 unified_corpus 全部 chunk_id
    corpus_cids = set()
    for pt in scroll_all(UNIFIED, with_payload_keys=["chunk_id"]):
        c = (pt.get("payload") or {}).get("chunk_id")
        if c:
            corpus_cids.add(c)

    hits = 0
    for pt in sample:
        p = pt.get("payload") or {}
        tag = p.get("tag_name")
        dim = p.get("dim_name")
        cids = p.get("chunk_ids") or []
        hit = sum(1 for c in cids if c in corpus_cids)
        hits += hit
        print(f"  [{dim}] '{tag}' → {len(cids)} chunk_ids, 命中 {hit}")
    print(f"  √ 样本命中 {hits} 条（应该 > 0）")


if __name__ == "__main__":
    print("=" * 70)
    print("  重建 dimension_tags  ←  unified_corpus 实际 chunk_id")
    print("=" * 70)

    encoder = _load_bge_encoder()
    if encoder is None:
        print("[X] BGE-M3 编码器加载失败")
        sys.exit(1)
    print("  [√] BGE-M3 编码器已加载")

    recreate_collection()
    tag_to_cids = build_tag_index()
    if not tag_to_cids:
        print("[X] 没有任何 tag，写入空集合")
        sys.exit(1)
    encode_and_upsert(tag_to_cids, encoder)
    verify()

    print("\n" + "=" * 70)
    print("  ✅ 重建完成")
    print(f"  现在可以 python interactive_qa.py 测试 dim 路检索")
    print("=" * 70)