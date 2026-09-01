"""
测试 main.py 修复后的兼容性
"""
import sys
from pathlib import Path
# 添加core目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

print("=" * 70)
print("测试 main.py 修复后的兼容性")
print("=" * 70)

# 测试导入
print("\n[1/3] 测试模块导入...")
try:
    from naive_rag import NaiveRAG
    from vector_store import get_vector_store
    print("✓ 模块导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 测试向量库初始化
print("\n[2/3] 测试向量库初始化...")
try:
    rag = NaiveRAG(dataset_name='nq_test_faiss', use_llm=False)
    doc_count = len(rag.vector_store.metadata)
    print(f"✓ 向量库初始化成功")
    print(f"  - 文档数: {doc_count}")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试回答功能
print("\n[3/3] 测试回答功能...")
try:
    question = "What is the longest word?"
    answer = rag.answer(question, top_k=2)
    print(f"✓ 回答生成成功")
    print(f"  - 问题: {question}")
    print(f"  - 回答: {answer.answer[:100]}...")
    print(f"  - 检索到的文档数: {len(answer.retrieved_chunks)}")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("所有测试通过！main.py 修复成功。")
print("=" * 70)
