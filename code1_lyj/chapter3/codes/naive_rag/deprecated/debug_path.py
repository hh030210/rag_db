"""
调试路径问题
"""
import os
from pathlib import Path
import config

print("=" * 60)
print("调试路径问题")
print("=" * 60)

# 检查 VECTOR_DB_ROOT
print(f"\n1. VECTOR_DB_ROOT: {config.VECTOR_DB_ROOT}")
print(f"   类型: {type(config.VECTOR_DB_ROOT)}")
print(f"   是否存在: {config.VECTOR_DB_ROOT.exists()}")

# 创建测试目录
test_dir = config.VECTOR_DB_ROOT / "test_faiss" / "faiss"
print(f"\n2. 测试目录: {test_dir}")
print(f"   类型: {type(test_dir)}")

# 使用 pathlib 创建
try:
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"   mkdir 成功")
    print(f"   是否存在: {test_dir.exists()}")
except Exception as e:
    print(f"   mkdir 失败: {e}")

# 使用 os.makedirs
test_dir_str = str(test_dir)
print(f"\n3. 使用 os.makedirs: {test_dir_str}")
try:
    os.makedirs(test_dir_str, exist_ok=True)
    print(f"   makedirs 成功")
    print(f"   是否存在: {os.path.exists(test_dir_str)}")
except Exception as e:
    print(f"   makedirs 失败: {e}")

# 检查父目录
print(f"\n4. 检查父目录:")
parent = test_dir.parent
print(f"   父目录: {parent}")
print(f"   父目录是否存在: {parent.exists()}")

# 尝试创建文件
test_file = test_dir / "test.txt"
print(f"\n5. 尝试创建文件: {test_file}")
try:
    with open(test_file, 'w') as f:
        f.write("test")
    print(f"   文件创建成功")
    print(f"   文件是否存在: {test_file.exists()}")
    # 清理
    test_file.unlink()
    test_dir.rmdir()
    parent.rmdir()
except Exception as e:
    print(f"   文件创建失败: {e}")

print("\n" + "=" * 60)
