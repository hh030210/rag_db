from pymilvus import connections, utility
connections.connect(host="localhost", port="19530")
for c in utility.list_collections():
    print(f"Dropping: {c}")
    utility.drop_collection(c)
print("Cleaned all collections")
