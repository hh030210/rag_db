from pymilvus import connections, Collection
connections.connect(host="localhost", port="19530")
try:
    c = Collection("eval_baseline_v1_top8")
    c.load()
    print(f"Collection 'eval_baseline_v1_top8': {c.num_entities} entities")
except Exception as e:
    print(f"eval_baseline_v1_top8: {e}")
try:
    c2 = Collection("meta_chunks_full_v2")
    c2.load()
    print(f"Collection 'meta_chunks_full_v2': {c2.num_entities} entities")
except Exception as e:
    print(f"meta_chunks_full_v2: {e}")
