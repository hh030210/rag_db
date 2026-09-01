from pymilvus import connections, Collection
connections.connect(host="localhost", port="19530")
c = Collection("meta_chunks_full_v2")
c.load()
print(f"meta_chunks_full_v2: {c.num_entities} entities")
