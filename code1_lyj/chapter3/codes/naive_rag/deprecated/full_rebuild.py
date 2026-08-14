"""
完全重建ChromaDB（删除所有数据重新索引）
"""
import shutil
from pathlib import Path

persist_dir = Path("vector_dbs/nq_test")

print("=" * 60)
print("完全重建ChromaDB")
print("=" * 60)

# 备份
backup_dir = persist_dir.parent / f"{persist_dir.name}_backup_full"
if backup_dir.exists():
    shutil.rmtree(backup_dir)
shutil.copytree(persist_dir, backup_dir)
print(f"✓ 备份到: {backup_dir}")

# 删除整个向量库
print("\n删除向量库...")
shutil.rmtree(persist_dir)
print("✓ 已删除")

print("\n" + "=" * 60)
print("向量库已清除，请重新运行索引命令:")
print(f"  python main.py --dataset nq_test --index \"your_data.json\"")
print("=" * 60)
