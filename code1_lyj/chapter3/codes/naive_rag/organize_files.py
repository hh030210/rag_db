"""
整理naive_rag目录结构
"""
import os
import shutil
from pathlib import Path

base_dir = Path(__file__).parent

print("=" * 70)
print("整理目录结构")
print("=" * 70)

# 定义文件分类
categories = {
    'core': ['chunker.py', 'config.py', 'embedder.py', 'llm_client.py', 
             'main.py', 'naive_rag.py', 'vector_store.py', '__init__.py', 'requirements.txt'],
    'test': ['test_faiss.py', 'test_full_rag.py', 'test_main_fix.py', 
             'test_tourist_search.py', 'check_tourist_stats.py', 'use_RAG.py',
             'quick_test.py', 'check_gpu.py', 'check_header.py', 'check_segment.py'],
    'data_processing': ['index_tourist_data.py', 'stat_tourist_data.py', 
                        'delete_tourist.py', 'run_index_tourist.bat'],
    'deprecated': ['vector_store_fixed.py', 'cleanup_and_test.py', 'check_chroma.py',
                   'debug_chroma.py', 'debug_path.py', 'force_rebuild.py', 
                   'full_rebuild.py', 'repair_chroma.py'],
    'results': ['results_test_20260303_004927.json', 
                'results_nq_test_20260303_084501.json',
                'results_nq_test_20260303_084505.json']
}

# 创建目录
for category in categories.keys():
    dir_path = base_dir / category
    dir_path.mkdir(exist_ok=True)
    print(f"✓ 创建目录: {category}/")

# 移动文件
moved_count = 0
for category, files in categories.items():
    target_dir = base_dir / category
    for filename in files:
        source = base_dir / filename
        if source.exists():
            target = target_dir / filename
            shutil.move(str(source), str(target))
            print(f"  移动: {filename} -> {category}/")
            moved_count += 1

print(f"\n✓ 共移动 {moved_count} 个文件")

# 清理__pycache__
pycache = base_dir / '__pycache__'
if pycache.exists():
    print(f"\n清理缓存目录: __pycache__/")
    shutil.rmtree(pycache)
    print("✓ 已清理")

print("\n" + "=" * 70)
print("整理完成!")
print("=" * 70)
print("\n新的目录结构:")
print("  core/          - 核心模块")
print("  test/          - 测试脚本")
print("  data_processing/ - 数据处理脚本")
print("  deprecated/    - 废弃/旧版本文件")
print("  results/       - 结果文件")
print("  llm_logs/      - LLM日志")
print("  vector_dbs/    - 向量数据库")
