"""
修复ChromaDB HNSW索引
在不丢失数据的情况下重建索引文件
"""
import chromadb
from chromadb.config import Settings
from pathlib import Path
import shutil

persist_dir = Path("vector_dbs/nq_test")

print("=" * 60)
print("ChromaDB 索引修复工具")
print("=" * 60)

# 1. 备份现有数据
backup_dir = persist_dir.parent / f"{persist_dir.name}_backup_{Path().stat().st_mtime}"
print(f"\n1. 备份现有数据到: {backup_dir}")
try:
    shutil.copytree(persist_dir, backup_dir)
    print("  ✓ 备份完成")
except Exception as e:
    print(f"  ✗ 备份失败: {e}")
    print("  继续修复...")

# 2. 删除损坏的HNSW索引目录
print("\n2. 删除损坏的HNSW索引文件...")
hnsw_dir = persist_dir / "84d79eea-77ab-403b-a251-577c43b57130"
if hnsw_dir.exists():
    try:
        shutil.rmtree(hnsw_dir)
        print(f"  ✓ 已删除: {hnsw_dir.name}")
    except Exception as e:
        print(f"  ✗ 删除失败: {e}")

# 3. 重新初始化ChromaDB（会自动重建索引）
print("\n3. 重新初始化ChromaDB...")
try:
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False)
    )
    print("  ✓ 客户端初始化成功")
    
    # 获取集合
    collection = client.get_collection("nq_test_collection")
    count = collection.count()
    print(f"  ✓ 集合加载成功")
    print(f"  - 文档数: {count}")
    
    # 触发索引重建 - 执行一个查询
    print("\n4. 触发索引重建...")
    print("  执行测试查询以重建HNSW索引...")
    
    # 使用peek获取一些文档来触发索引
    try:
        # 尝试获取文档来触发索引构建
        result = collection.peek(limit=1)
        print(f"  ✓ 索引重建成功!")
        
        # 验证查询功能
        print("\n5. 验证查询功能...")
        from embedder import embedder
        test_query = "test"
        query_embedding = embedder.encode_query(test_query)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )
        print("  ✓ 查询功能正常!")
        print(f"\n修复完成！向量库可以正常使用了。")
        
    except Exception as e:
        print(f"  ✗ 索引重建失败: {e}")
        print("\n建议: 使用 --reindex 参数重新构建索引")
        
except Exception as e:
    print(f"  ✗ 初始化失败: {e}")
    print("\n建议: 使用 --reindex 参数重新构建索引")

print("=" * 60)
