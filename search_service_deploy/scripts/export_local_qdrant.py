"""
导出本地 Qdrant 数据为 JSON 文件。
"""
import json, sys
from qdrant_client import QdrantClient
from pathlib import Path

LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 6333
BATCH = 200
OUT_DIR = Path("C:/temp_qdrant_export")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def export(client, coll_name):
    all_pts = []
    offset = None
    total = 0
    while True:
        result = client.scroll(
            collection_name=coll_name,
            offset=offset,
            limit=BATCH,
            with_vectors=True,
        )
        pts = result[0]
        nxt = result[1]
        for pt in pts:
            vec_dict = getattr(pt, 'vector', None) or {}
            vec = vec_dict.get('chunk_text_vec') if isinstance(vec_dict, dict) else None
            all_pts.append({
                "id": str(pt.id),
                "vector": vec,
                "payload": pt.payload or {},
            })
            total += 1
        if nxt is None:
            break
        offset = nxt
        print(f"  fetched {total}...", flush=True)

    out_path = OUT_DIR / f"{coll_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_pts, f, ensure_ascii=False)
    print(f"  {coll_name}: {total} points -> {out_path} ({Path(out_path).stat().st_size/1024/1024:.1f} MB)")
    return total


def main():
    client = QdrantClient(host=LOCAL_HOST, port=LOCAL_PORT)
    for coll in ["unified_corpus", "dimension_tags"]:
        print(f"Exporting {coll}...")
        n = export(client, coll)
        print(f"  done: {n} points")
    print("\nAll exported!")


if __name__ == "__main__":
    main()
