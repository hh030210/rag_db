"""
强制重建ChromaDB索引
"""
import shutil
from pathlib import Path
import chromadb
from chromadb.config import Settings

persist_dir = Path("vector_dbs/nq_test")

print("=" * 60)
print("强制重建ChromaDB索引")
print("=" * 60)

# 1. 找到并删除所有HNSW目录
print("\n1. 清理HNSW索引目录...")
hnsw_dirs = [d for d in persist_dir.iterdir() if d.is_dir()]
for hnsw_dir in hnsw_dirs:
    print(f"  删除: {hnsw_dir.name}")
    shutil.rmtree(hnsw_dir)

# 2. 重新创建客户端（会触发索引重建）
print("\n2. 重新初始化ChromaDB...")
client = chromadb.PersistentClient(
    path=str(persist_dir),
    settings=Settings(anonymized_telemetry=False)
)

# 3. 获取集合并强制加载
print("\n3. 加载集合并重建索引...")
collection = client.get_collection("nq_test_collection")
print(f"  文档数: {collection.count()}")

# 4. 执行查询以触发索引重建
print("\n4. 执行测试查询触发索引重建...")
# 先获取一个文档来触发索引
sample = collection.peek(limit=1)
print(f"  获取样本: {len(sample['ids'])} 个文档")

# 然后执行向量查询
import sys
sys.path.insert(0, '.')
from embedder import embedder

query_embedding = embedder.encode_query("test query")
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)
print("  ✓ 向量查询成功!")

# 5. 验证文件已创建
print("\n5. 验证索引文件...")
hnsw_dirs = [d for d in persist_dir.iterdir() if d.is_dir()]
for hnsw_dir in hnsw_dirs:
    print(f"  索引目录: {hnsw_dir.name}")
    for f in hnsw_dir.iterdir():
        size = f.stat().st_size
        print(f"    - {f.name}: {size:,} bytes")

print("\n" + "=" * 60)
print("重建完成!")
print("=" * 60)
