"""
检查tourist向量库统计信息
"""
import sys
from pathlib import Path
# 添加core目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from vector_store import get_vector_store
import pickle

print("=" * 70)
print("Tourist向量库统计")
print("=" * 70)

vs = get_vector_store('tourist')

print(f"\n向量库路径: {vs.persist_dir}")
print(f"文档块总数: {len(vs.metadata)}")

if len(vs.metadata) > 0:
    print("\n" + "-" * 70)
    print("文档块详情:")
    print("-" * 70)
    
    # 统计每个源文件的文档数
    from collections import defaultdict
    source_stats = defaultdict(int)
    chunk_lengths = []
    
    for idx, meta in vs.metadata.items():
        chunk_id = meta.get('chunk_id', '')
        text = meta.get('text', '')
        metadata = meta.get('metadata', {})
        
        # 提取源文件信息
        doc_id = metadata.get('doc_id', 'unknown')
        source_stats[doc_id] += 1
        chunk_lengths.append(len(text))
        
        # 显示前5个文档块的信息
        if int(idx) < 5:
            print(f"\n文档块 {idx}:")
            print(f"  chunk_id: {chunk_id}")
            print(f"  doc_id: {doc_id}")
            print(f"  长度: {len(text)} 字符")
            print(f"  内容预览: {text[:100]}...")
    
    print("\n" + "-" * 70)
    print("按源文件统计:")
    print("-" * 70)
    for source, count in sorted(source_stats.items()):
        print(f"  {source}: {count} 个文档块")
    
    print("\n" + "-" * 70)
    print("文档块长度统计:")
    print("-" * 70)
    print(f"  平均长度: {sum(chunk_lengths)/len(chunk_lengths):.0f} 字符")
    print(f"  最大长度: {max(chunk_lengths)} 字符")
    print(f"  最小长度: {min(chunk_lengths)} 字符")

print("\n" + "=" * 70)
