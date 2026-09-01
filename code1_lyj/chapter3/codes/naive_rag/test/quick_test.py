"""快速测试"""
import sys
from pathlib import Path
# 添加core目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
from vector_store import get_vector_store

vs = get_vector_store('nq_test')
print(f'文档数: {len(vs.metadata)}')

# 测试查询
query = 'what is the longest english word'
print(f'查询: {query}')
results = vs.search(query, top_k=3)
print(f'结果数: {len(results)}')
for i, r in enumerate(results):
    text = r["text"][:100]
    print(f'{i+1}. {text}...')
