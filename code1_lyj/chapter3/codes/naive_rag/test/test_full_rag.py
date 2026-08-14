"""
完整RAG流程测试 - 使用Faiss向量库
"""
import sys
from pathlib import Path
# 添加core目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

print("=" * 70)
print("完整RAG流程测试 - Faiss向量库")
print("=" * 70)

# 测试1: 初始化向量库
print("\n[1/5] 初始化向量库...")
try:
    from vector_store import get_vector_store
    vs = get_vector_store('nq_test_faiss')
    print(f"✓ 向量库初始化成功")
    print(f"  - 数据集名称: nq_test_faiss")
    print(f"  - 当前文档数: {len(vs.metadata)}")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试2: 添加一些测试文档
print("\n[2/5] 添加测试文档...")
try:
    test_chunks = [
        {
            'chunk_id': 'wiki_1',
            'text': 'The longest word in English is pneumonoultramicroscopicsilicovolcanoconiosis, '
                    'a word that refers to a lung disease contracted from the inhalation of very fine silicate or quartz dust.',
            'metadata': {'doc_id': 'wiki_1', 'title': 'Longest English Word'}
        },
        {
            'chunk_id': 'wiki_2',
            'text': 'Supercalifragilisticexpialidocious is a song from the 1964 Disney musical film Mary Poppins. '
                    'The song was written by the Sherman Brothers.',
            'metadata': {'doc_id': 'wiki_2', 'title': 'Mary Poppins Song'}
        },
        {
            'chunk_id': 'wiki_3',
            'text': 'Antidisestablishmentarianism is a political position that developed in 19th-century Britain. '
                    'It is opposition to the disestablishment of the Church of England.',
            'metadata': {'doc_id': 'wiki_3', 'title': 'Political Movement'}
        },
        {
            'chunk_id': 'wiki_4',
            'text': 'The Great Wall of China is a series of fortifications that were built across the historical '
                    'northern borders of ancient Chinese states. It is one of the most famous landmarks in the world.',
            'metadata': {'doc_id': 'wiki_4', 'title': 'Great Wall of China'}
        },
        {
            'chunk_id': 'wiki_5',
            'text': 'The Amazon River in South America is the largest river by discharge volume of water in the world. '
                    'It flows through Brazil, Peru, and Colombia.',
            'metadata': {'doc_id': 'wiki_5', 'title': 'Amazon River'}
        }
    ]
    
    vs.add_documents(test_chunks, batch_size=2)
    print(f"✓ 成功添加 {len(test_chunks)} 个文档")
    print(f"  - 当前总文档数: {len(vs.metadata)}")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试3: 检索测试
print("\n[3/5] 检索测试...")
try:
    queries = [
        "What is the longest word in English?",
        "Tell me about the Great Wall",
        "What is the largest river in the world?"
    ]
    
    for query in queries:
        print(f"\n  查询: '{query}'")
        results = vs.search(query, top_k=2)
        print(f"  返回 {len(results)} 个结果:")
        for i, r in enumerate(results, 1):
            text = r['text'][:80] + "..." if len(r['text']) > 80 else r['text']
            print(f"    {i}. [相似度: {r['score']:.4f}] {text}")
    print("✓ 检索测试完成")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试4: 重新加载向量库（验证持久化）
print("\n[4/5] 验证向量库持久化...")
try:
    vs2 = get_vector_store('nq_test_faiss')
    print(f"✓ 重新加载成功")
    print(f"  - 文档数: {len(vs2.metadata)}")
    
    # 验证数据完整性
    results = vs2.search("longest word", top_k=1)
    if results and 'pneumonoultramicroscopicsilicovolcanoconiosis' in results[0]['text']:
        print("✓ 数据完整性验证通过")
    else:
        print("⚠ 数据完整性验证警告")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试5: RAG完整流程（不使用LLM）
print("\n[5/5] RAG完整流程测试（简单模式，不使用LLM）...")
try:
    from naive_rag import NaiveRAG
    
    rag = NaiveRAG(dataset_name='nq_test_faiss', use_llm=False)
    
    question = "What is the longest word in English?"
    print(f"\n  问题: {question}")
    
    answer = rag.answer(question, top_k=2)
    
    print(f"\n  检索到的文档块:")
    for i, chunk in enumerate(answer.retrieved_chunks, 1):
        text = chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
        print(f"    {i}. [{chunk.get('score', 0):.4f}] {text}")
    
    print(f"\n  生成的回答:")
    print(f"    {answer.answer}")
    
    print("✓ RAG流程测试完成")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()

# 清理（可选）
print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)
print("\n是否清理测试数据? (y/n): ", end="")
# 不自动清理，让用户决定
print("跳过自动清理，测试数据保留在 nq_test_faiss 数据集中")
print(f"\n向量库统计:")
print(f"  - 数据集: nq_test_faiss")
print(f"  - 文档数: {len(vs.metadata)}")
print(f"  - 存储路径: {vs.persist_dir}")
