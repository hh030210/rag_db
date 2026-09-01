"""
检查header.bin文件
"""
from pathlib import Path
import struct

header_file = Path(__file__).parent.parent / "vector_dbs/nq_test/84d79eea-77ab-403b-a251-577c43b57130/header.bin"

print("=" * 60)
print("检查header.bin文件")
print("=" * 60)

if not header_file.exists():
    print(f"✗ 文件不存在: {header_file}")
else:
    size = header_file.stat().st_size
    print(f"文件大小: {size} bytes")
    
    with open(header_file, 'rb') as f:
        data = f.read()
        print(f"实际读取: {len(data)} bytes")
        print(f"原始数据: {data}")
        
        # 尝试解析为整数（HNSW通常存储维度等信息）
        if len(data) >= 4:
            try:
                val = struct.unpack('i', data[:4])[0]
                print(f"前4字节作为int32: {val}")
            except:
                pass

print("\n" + "=" * 60)
