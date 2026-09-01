"""
测试Faiss向量存储
"""
import sys
from pathlib import Path
# 添加core目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

print("=" * 60)
print("测试Faiss向量存储")
print("=" * 60)

# 测试1: 创建新的向量库
print("\n测试1: 创建新的向量库")
try:
    from vector_store import get_vector_store
    
    vs = get_vector_store("test_faiss")
    print(f"✓ 创建成功")
    print(f"  - 文档数: {len(vs.metadata)}")
    
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试2: 添加一些测试数据
print("\n测试2: 添加测试数据")
try:
    test_chunks = [
        {
            'chunk_id': 'doc1_chunk0',
            'text': 'This is a test document about machine learning.',
            'metadata': {'doc_id': 'doc1', 'chunk_index': 0}
        },
        {
            'chunk_id': 'doc2_chunk0',
            'text': 'Python is a popular programming language.',
            'metadata': {'doc_id': 'doc2', 'chunk_index': 0}
        },
        {
            'chunk_id': 'doc3_chunk0',
            'text': 'Deep learning is a subset of machine learning.',
            'metadata': {'doc_id': 'doc3', 'chunk_index': 0}
        }
    ]
    
    vs.add_documents(test_chunks, batch_size=2)
    print(f"✓ 添加成功")
    print(f"  - 文档数: {len(vs.metadata)}")
    
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试3: 搜索
print("\n测试3: 搜索测试")
try:
    results = vs.search("machine learning", top_k=2)
    print(f"✓ 搜索成功")
    print(f"  - 返回 {len(results)} 个结果")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.4f}] {r['text'][:50]}...")
    
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试4: 重新加载
print("\n测试4: 重新加载向量库")
try:
    vs2 = get_vector_store("test_faiss")
    print(f"✓ 加载成功")
    print(f"  - 文档数: {len(vs2.metadata)}")
    
    results = vs2.search("python programming", top_k=2)
    print(f"  - 搜索返回 {len(results)} 个结果")
    
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()

# 清理
print("\n测试5: 清理测试数据")
try:
    vs.delete_collection()
    print("✓ 清理完成")
except Exception as e:
    print(f"✗ 清理失败: {e}")

print("\n" + "=" * 60)
print("所有测试完成！")
print("=" * 60)
