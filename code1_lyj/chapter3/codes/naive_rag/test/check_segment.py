"""
检查ChromaDB segment ID匹配
"""
import sqlite3
from pathlib import Path

persist_dir = Path(__file__).parent.parent / "vector_dbs/nq_test"
sqlite_file = persist_dir / "chroma.sqlite3"

print("=" * 60)
print("检查Segment ID匹配")
print("=" * 60)

# 连接SQLite
conn = sqlite3.connect(str(sqlite_file))
cursor = conn.cursor()

# 获取表结构
cursor.execute("PRAGMA table_info(segments);")
columns = cursor.fetchall()
print("\nSegments表结构:")
for col in columns:
    print(f"  {col[1]}: {col[2]}")

# 获取segments
cursor.execute("SELECT * FROM segments;")
segments = cursor.fetchall()
print(f"\n数据库中的Segments ({len(segments)}个):")
for seg in segments:
    print(f"  {seg}")

# 检查目录
print("\n实际目录:")
for d in persist_dir.iterdir():
    if d.is_dir():
        print(f"  {d.name}")

conn.close()
print("\n" + "=" * 60)
