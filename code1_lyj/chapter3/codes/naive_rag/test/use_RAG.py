# 使用示例
from naive_rag import NaiveRAG

# 初始化（不使用LLM）
rag = NaiveRAG(dataset_name='nq_test_faiss', use_llm=True)

# 检查文档数量
doc_count = len(rag.vector_store.metadata)
print(f"文档数: {doc_count}")

# 回答问题
answer = rag.answer("What is the longest word?", top_k=5)
print(answer)
print("提取答案:")
print(answer.answer)