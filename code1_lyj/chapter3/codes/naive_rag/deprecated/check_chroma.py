"""
检查ChromaDB向量库状态
"""
import pickle
import chromadb
from chromadb.config import Settings
from pathlib import Path

persist_dir = Path("vector_dbs/nq_test")

print("=" * 60)
print("ChromaDB 向量库诊断")
print("=" * 60)

# 1. 检查文件结构
print("\n1. 文件结构:")
for item in persist_dir.rglob("*"):
    rel_path = item.relative_to(persist_dir)
    size = item.stat().st_size if item.is_file() else "<DIR>"
    print(f"  {rel_path}: {size}")

# 2. 检查metadata
print("\n2. 索引元数据:")
metadata_file = persist_dir / "84d79eea-77ab-403b-a251-577c43b57130" / "index_metadata.pickle"
if metadata_file.exists():
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
        print(f"  类型: {type(metadata)}")
        print(f"  内容: {metadata}")
else:
    print("  元数据文件不存在!")

# 3. 尝试连接并获取信息
print("\n3. ChromaDB连接测试:")
try:
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False)
    )
    print("  ✓ 客户端连接成功")
    
    # 列出集合
    collections = client.list_collections()
    print(f"  集合数量: {len(collections)}")
    
    for col in collections:
        print(f"\n  集合: {col.name}")
        print(f"    文档数: {col.count()}")
        
except Exception as e:
    print(f"  ✗ 连接失败: {e}")

# 4. 检查SQLite数据库
print("\n4. SQLite数据库检查:")
sqlite_file = persist_dir / "chroma.sqlite3"
if sqlite_file.exists():
    import sqlite3
    try:
        conn = sqlite3.connect(str(sqlite_file))
        cursor = conn.cursor()
        
        # 检查表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"  表数量: {len(tables)}")
        
        # 检查集合表
        cursor.execute("SELECT id, name FROM collections;")
        collections = cursor.fetchall()
        print(f"  集合: {collections}")
        
        # 检查segment表
        cursor.execute("SELECT id, type, scope FROM segments;")
        segments = cursor.fetchall()
        print(f"  Segments:")
        for seg in segments:
            print(f"    ID: {seg[0]}, Type: {seg[1]}, Scope: {seg[2]}")
        
        conn.close()
    except Exception as e:
        print(f"  数据库检查失败: {e}")
else:
    print("  SQLite文件不存在!")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
