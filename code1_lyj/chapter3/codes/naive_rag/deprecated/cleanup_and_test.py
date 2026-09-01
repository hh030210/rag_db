"""
清理旧数据并测试Milvus
"""
import shutil
from pathlib import Path
import sys

print("=" * 60)
print("清理旧数据并测试Milvus")
print("=" * 60)

# 1. 删除旧的ChromaDB数据
vector_db_root = Path("vector_dbs")
if vector_db_root.exists():
    print("\n1. 删除旧的ChromaDB数据...")
    for dataset_dir in vector_db_root.iterdir():
        if dataset_dir.is_dir():
            # 删除ChromaDB文件，保留目录结构
            chroma_sqlite = dataset_dir / "chroma.sqlite3"
            if chroma_sqlite.exists():
                chroma_sqlite.unlink()
                print(f"  ✓ 删除: {chroma_sqlite}")
            
            # 删除HNSW目录
            for subdir in dataset_dir.iterdir():
                if subdir.is_dir() and len(subdir.name) == 36:  # UUID格式
                    shutil.rmtree(subdir)
                    print(f"  ✓ 删除HNSW目录: {subdir.name}")
    print("✓ 旧数据清理完成")
else:
    vector_db_root.mkdir(parents=True, exist_ok=True)
    print("✓ 创建vector_dbs目录")

# 2. 测试Milvus连接
print("\n2. 测试Milvus连接...")
try:
    sys.path.insert(0, '.')
    from vector_store import get_vector_store
    
    vs = get_vector_store("nq_test")
    print(f"✓ Milvus连接成功")
    print(f"  - 数据集: nq_test")
    print(f"  - 文档数: {vs.collection.num_entities}")
    
except Exception as e:
    print(f"✗ 连接失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("准备就绪！可以使用以下命令测试：")
print(f"  python main.py --dataset nq_test --index \"your_data.json\"")
print("=" * 60)
