from pymilvus import connections, utility
connections.connect(host="localhost", port="19530")
old = ["eval_baseline_v1_top8", "test_mini", "embed_test"]
for c in old:
    try:
        utility.drop_collection(c)
        print(f"Dropped: {c}")
    except Exception as e:
        print(f"Drop {c}: {e}")
print("Done")
