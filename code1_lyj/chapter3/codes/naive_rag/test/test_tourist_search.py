"""
测试tourist数据集检索
"""
import sys
from pathlib import Path
# 添加core目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from naive_rag import NaiveRAG

print("=" * 70)
print("Tourist数据集检索测试")
print("=" * 70)

# 初始化RAG
rag = NaiveRAG(dataset_name='tourist', use_llm=False)

# 检查文档数量
doc_count = len(rag.vector_store.metadata)
print(f"\n当前向量库文档数: {doc_count}")

# 测试查询
test_queries = [
    "南孔庙有哪些景点",
    "少林寺的开放时间",
    "张家界门票价格",
    "颐和园怎么预约",
    "西湖游船多少钱"
]

print("\n" + "=" * 70)
print("检索测试")
print("=" * 70)

for query in test_queries:
    print(f"\n查询: {query}")
    results = rag.vector_store.search(query, top_k=3)
    print(f"返回 {len(results)} 个结果:")
    for i, r in enumerate(results, 1):
        text = r['text'][:80] + "..." if len(r['text']) > 80 else r['text']
        print(f"  {i}. [相似度: {r['score']:.4f}] {text}")

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
