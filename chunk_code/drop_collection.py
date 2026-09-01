from pymilvus import connections, utility
connections.connect(host="localhost", port="19530")
try:
    utility.drop_collection("eval_baseline_v1_top8")
    print("Dropped eval_baseline_v1_top8")
except Exception as e:
    print(f"Drop: {e}")
