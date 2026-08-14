"""
调试ChromaDB问题
"""
import chromadb
from chromadb.config import Settings
from pathlib import Path

persist_dir = Path("vector_dbs/nq_test")

print("=" * 60)
print("调试ChromaDB")
print("=" * 60)

# 检查文件
print("\n1. 检查文件结构:")
hnsw_dirs = [d for d in persist_dir.iterdir() if d.is_dir()]
for hnsw_dir in hnsw_dirs:
    print(f"  目录: {hnsw_dir.name}")
    for f in sorted(hnsw_dir.iterdir()):
        size = f.stat().st_size
        print(f"    - {f.name}: {size:,} bytes")

# 直接测试（不通过vector_store）
print("\n2. 直接连接测试:")
client = chromadb.PersistentClient(
    path=str(persist_dir),
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection("nq_test_collection")
print(f"  文档数: {collection.count()}")

# 测试peek
print("\n3. 测试peek:")
sample = collection.peek(limit=1)
print(f"  获取: {len(sample['ids'])} 个文档")

# 测试查询
print("\n4. 测试向量查询:")
import sys
sys.path.insert(0, '.')
from embedder import embedder

query_embedding = embedder.encode_query("test")
print(f"  向量维度: {len(query_embedding)}")

try:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )
    print(f"  ✓ 查询成功! 返回 {len(results['ids'][0])} 个结果")
except Exception as e:
    print(f"  ✗ 查询失败: {e}")

print("\n" + "=" * 60)
print("调试完成")
print("=" * 60)
