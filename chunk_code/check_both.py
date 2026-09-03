from pymilvus import connections, Collection
connections.connect(host="localhost", port="19530")
for name in ["eval_baseline_v1_top8", "eval_enhanced_v2_top8"]:
    try:
        c = Collection(name)
        c.load()
        print(f"{name}: {c.num_entities} entities")
    except Exception as e:
        print(f"{name}: {e}")
