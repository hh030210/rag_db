"""
朴素RAG实现
"""
import json
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import config
from chunker import chunker
from vector_store import get_vector_store
from llm_client import LLMClient

# 导入模型配置
try:
    from model_config import CURRENT_LLM_DISPLAY_NAME
except ImportError:
    CURRENT_LLM_DISPLAY_NAME = "GLM-4.1V-9B-Thinking"


@dataclass
class RAGAnswer:
    """RAG回答结果"""
    question: str
    answer: str
    answer_json: Dict[str, Any]  # JSON格式的回答
    retrieved_chunks: List[Dict]
    context: str


class NaiveRAG:
    """朴素RAG系统"""
    
    def __init__(self, dataset_name: str, use_llm: bool = True):
        """
        初始化RAG系统
        
        Args:
            dataset_name: 数据集名称，用于选择对应的向量库
            use_llm: 是否使用LLM生成回答（默认使用大语言模型）
        """
        self.dataset_name = dataset_name
        self.vector_store = get_vector_store(dataset_name)
        self.results_history = []  # 存储问答历史
        self.use_llm = use_llm
        
        # 初始化LLM客户端
        if use_llm:
            self.llm_client = LLMClient(log_file=f"llm_logs/{dataset_name}_llm.log")
            print(f"✓ RAG系统初始化完成，使用数据集: {dataset_name}，LLM: {CURRENT_LLM_DISPLAY_NAME}")
        else:
            self.llm_client = None
            print(f"✓ RAG系统初始化完成，使用数据集: {dataset_name}，LLM: 禁用（使用简单提取）")
    
    def index_documents(self, json_file_path: str, batch_size: int = None):
        """
        索引文档到向量库（追加模式，流式读取）
        
        Args:
            json_file_path: JSON文件路径
            batch_size: 每批处理的文档数
        """
        print(f"\n{'='*60}")
        print(f"正在索引文件: {json_file_path}")
        print(f"{'='*60}")
        
        # 显示当前向量库状态
        current_count = len(self.vector_store.metadata)
        print(f"当前向量库文档数: {current_count}")
        
        # 使用配置的批次大小
        if batch_size is None:
            batch_size = config.INDEX_BATCH_SIZE
        
        # 流式读取JSON文件
        print(f"\n[1/4] 正在流式读取JSON文件...")
        print(f"  批次大小: {batch_size} 个文档")
        total_docs = 0
        total_chunks = 0
        batch = []
        
        # 使用ijson流式读取
        try:
            import ijson
            print("  使用ijson进行流式解析...")
            
            with open(json_file_path, 'rb') as f:
                for doc in ijson.items(f, 'item'):
                    batch.append(doc)
                    total_docs += 1
                    
                    if len(batch) >= batch_size:
                        # 处理当前批次
                        print(f"  处理批次: 已处理 {total_docs} 个文档...")
                        chunks = chunker.chunk_documents(batch)
                        total_chunks += len(chunks)
                        self.vector_store.add_documents(chunks)
                        batch = []  # 清空批次
                        
                    if total_docs % 10000 == 0:
                        print(f"    进度: {total_docs} 个文档已处理")
            
            # 处理剩余文档
            if batch:
                print(f"  处理最后批次: {len(batch)} 个文档...")
                chunks = chunker.chunk_documents(batch)
                total_chunks += len(chunks)
                self.vector_store.add_documents(chunks)
                
        except ImportError:
            print("  ijson未安装，使用标准json分批读取...")
            print("  提示: pip install ijson 可以获得更好的大文件支持")
            
            # 回退到标准json，但分批处理
            with open(json_file_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            total_docs = len(documents)
            print(f"✓ 加载了 {total_docs} 个文档")
            
            # 分批处理
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                print(f"  处理批次 {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}: {len(batch)} 个文档...")
                chunks = chunker.chunk_documents(batch)
                total_chunks += len(chunks)
                self.vector_store.add_documents(chunks)
        
        print(f"\n[2/4] 文档切片完成")
        print(f"✓ 总共处理: {total_docs} 个文档")
        print(f"✓ 总共生成: {total_chunks} 个文档块")
        
        # 显示完成状态
        new_count = len(self.vector_store.metadata)
        print(f"\n[3/4] 索引完成！")
        print(f"✓ 本次新增: {total_chunks} 个文档块")
        print(f"✓ 向量库总计: {new_count} 个文档块 (原有 {current_count}, 新增 {new_count - current_count})")
        print(f"{'='*60}\n")
    
    def answer(self, question: str, top_k: int = None, gold_answers: List[str] = None) -> RAGAnswer:
        """
        回答问题
        
        Args:
            question: 问题文本
            top_k: 检索的文档块数量
            gold_answers: 标准答案列表（用于评估）
            
        Returns:
            RAGAnswer对象
        """
        # 检索相关文档块
        print(f"\n检索相关问题: {question}")
        retrieved_chunks = self.vector_store.search(question, top_k=top_k or config.TOP_K)
        
        print(f"✓ 检索到 {len(retrieved_chunks)} 个相关文档块")
        for i, chunk in enumerate(retrieved_chunks):
            print(f"  [{i+1}] 分数: {chunk['score']:.4f} | 来源: {chunk['metadata'].get('original_doc_id', 'N/A')}")
        
        # 构建上下文
        context = self._build_context(retrieved_chunks)
        
        # 生成回答（JSON格式）
        answer_json = self._generate_answer_json(question, retrieved_chunks)
        answer = answer_json.get("answer", "")
        
        # 计算是否正确（如果有标准答案）
        is_correct = None
        if gold_answers:
            is_correct = self._check_exact_match(answer, gold_answers)
        
        # 保存结果到历史
        result_record = {
            "question": question,
            "gold_answers": gold_answers if gold_answers else [],
            "model_answer": answer,
            "answer_json": answer_json,
            "is_correct": is_correct,
            "retrieved_chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["metadata"].get("original_doc_id", "N/A"),
                    "text": chunk["text"],
                    "score": chunk["score"]
                }
                for chunk in retrieved_chunks
            ]
        }
        self.results_history.append(result_record)
        
        return RAGAnswer(
            question=question,
            answer=answer,
            answer_json=answer_json,
            retrieved_chunks=retrieved_chunks,
            context=context
        )
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """构建上下文"""
        contexts = []
        for i, chunk in enumerate(chunks):
            contexts.append(f"[文档 {i+1}] {chunk['text']}")
        return "\n\n".join(contexts)
    
    def _generate_answer_json(self, question: str, chunks: List[Dict]) -> Dict[str, Any]:
        """
        生成JSON格式的回答
        
        返回格式:
        {
            "answer": "直接答案",
            "confidence": 0.95,
            "source": "chunk_id"
        }
        """
        if not chunks:
            return {
                "answer": "",
                "confidence": 0.0,
                "source": None
            }
        
        # 使用LLM生成回答
        if self.use_llm and self.llm_client:
            return self._generate_answer_with_llm(question, chunks)
        else:
            # 不使用LLM，返回最相关的文档块
            return self._generate_answer_simple(question, chunks)
    
    def _generate_answer_with_llm(self, question: str, chunks: List[Dict]) -> Dict[str, Any]:
        """
        使用LLM生成回答
        
        Args:
            question: 问题
            chunks: 检索到的文档块
            
        Returns:
            JSON格式的回答
        """
        # 构建上下文
        context = self._build_context(chunks)
        
        # 构建系统提示词（要求简短回答，适合NQ数据集）
        system_prompt = '''你是一个专业的问答助手。请基于提供的参考文档回答问题。
要求：
1. 回答要简短直接，1-5个词最佳
2. 只输出答案本身，不要解释
3. 如果文档中没有答案，输出"unknown"
4. 必须基于文档内容，不要使用外部知识'''
        
        # 构建用户提示词
        prompt = f'''参考文档：
{context}

问题：{question}

请根据参考文档回答问题。回答要简短（1-5个词）。如果文档中没有相关信息，请回答"unknown"。

请以JSON格式输出：
{{
    "answer": "你的简短回答",
    "confidence": 0.95,
    "sources": ["文档1", "文档2"]
}}'''
        
        # 调用LLM
        result = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            response_format="json",
            temperature=0.01,
            max_tokens=500,
            desc="生成RAG回答"
        )
        
        # 处理结果
        if "error" in result:
            print(f"[警告] LLM调用失败: {result['error']}")
            # 失败时回退到简单提取
            return self._generate_answer_simple(question, chunks)
        
        # 确保有answer字段
        if "answer" not in result:
            result["answer"] = result.get("text", "")
        
        # 添加来源信息
        if "sources" not in result:
            result["sources"] = [chunk["chunk_id"] for chunk in chunks[:3]]
        
        if "confidence" not in result:
            result["confidence"] = chunks[0]["score"] if chunks else 0.5
            
        return result
    
    def _generate_answer_simple(self, question: str, chunks: List[Dict]) -> Dict[str, Any]:
        """
        简单方式生成回答（不使用LLM）
        
        Args:
            question: 问题
            chunks: 检索到的文档块
            
        Returns:
            JSON格式的回答
        """
        # 返回最相关的文档块作为答案
        best_chunk = chunks[0]
        
        # 提取最相关的句子作为答案
        answer_text = self._extract_best_sentence(best_chunk["text"], question)
        
        return {
            "answer": answer_text,
            "confidence": best_chunk["score"],
            "source": best_chunk["chunk_id"]
        }
    
    def _extract_best_sentence(self, text: str, question: str) -> str:
        """从文本中提取最相关的句子作为答案"""
        # 简单实现：返回前200个字符
        if len(text) <= 200:
            return text.strip()
        return text[:200].strip() + "..."
    
    def _check_exact_match(self, answer: str, gold_answers: List[str]) -> bool:
        """
        检查精确匹配（EM）
        Args:
            answer: 模型回答
            gold_answers: 标准答案列表
        Returns:
            是否匹配
        """
        if not answer or not gold_answers:
            return False
        
        # 标准化答案（小写、去除多余空格和标点）
        def normalize(text: str) -> str:
            text = text.lower().strip()
            # 去除常见标点
            text = re.sub(r'[.,!?;:"\'\(\)\[\]{}]', '', text)
            # 去除多余空格
            text = re.sub(r'\s+', ' ', text)
            return text
        
        normalized_answer = normalize(answer)
        
        for gold in gold_answers:
            normalized_gold = normalize(gold)
            # 完全匹配或包含关系
            if normalized_answer == normalized_gold:
                return True
            if normalized_gold in normalized_answer or normalized_answer in normalized_gold:
                return True
        
        return False
    
    def batch_answer(self, questions: List[str], top_k: int = None) -> List[RAGAnswer]:
        """
        批量回答问题
        
        Args:
            questions: 问题列表
            top_k: 检索的文档块数量
            
        Returns:
            RAGAnswer列表
        """
        results = []
        for i, question in enumerate(questions):
            print(f"\n{'='*60}")
            print(f"处理问题 {i+1}/{len(questions)}")
            print(f"{'='*60}")
            result = self.answer(question, top_k)
            results.append(result)
        return results
    
    def save_results(self, output_path: str = None):
        """
        保存问答结果到JSON文件
        
        Args:
            output_path: 输出文件路径，默认为 results_{dataset_name}_{timestamp}.json
        """
        if not self.results_history:
            print("没有结果需要保存")
            return
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results_{self.dataset_name}_{timestamp}.json"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 计算统计信息
        total = len(self.results_history)
        correct_count = sum(1 for r in self.results_history if r.get("is_correct") is True)
        has_gold = sum(1 for r in self.results_history if r.get("gold_answers"))
        
        save_data = {
            "metadata": {
                "dataset_name": self.dataset_name,
                "total_questions": total,
                "questions_with_gold": has_gold,
                "correct_count": correct_count,
                "accuracy": correct_count / has_gold if has_gold > 0 else None,
            },
            "results": self.results_history
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 结果已保存到: {output_file}")
        print(f"  - 总问题数: {total}")
        print(f"  - 有标准答案: {has_gold}")
        print(f"  - 正确数: {correct_count}")
        if has_gold > 0:
            print(f"  - 准确率: {correct_count / has_gold:.2%}")
    
    def evaluate_on_dataset(self, json_file_path: str, max_samples: int = None, save_results: bool = True) -> Dict:
        """
        在数据集上评估RAG性能
        
        Args:
            json_file_path: 数据集JSON文件路径
            max_samples: 最大评估样本数
            save_results: 是否保存结果
            
        Returns:
            评估结果统计
        """
        print(f"\n加载评估数据: {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        print(f"评估样本数: {len(data)}")
        
        correct = 0
        total = 0
        
        for i, item in enumerate(data):
            question = item.get('question', '')
            gold_answers = item.get('annotations', {}).get('short_answers', [])
            
            if not question:
                continue
            
            print(f"\n[{i+1}/{len(data)}] 问题: {question}")
            
            # 使用answer方法（会自动保存到history）
            result = self.answer(question, top_k=5, gold_answers=gold_answers)
            
            if gold_answers:
                total += 1
                if self.results_history[-1].get("is_correct"):
                    correct += 1
                print(f"  答案: {result.answer_json.get('answer', '')}")
                print(f"  标准答案: {gold_answers}")
                print(f"  是否正确: {self.results_history[-1].get('is_correct')}")
            
            if total > 0 and total % 10 == 0:
                print(f"\n  >>> 已处理 {total} 个问题，当前EM准确率: {correct/total:.2%}")
        
        accuracy = correct / total if total > 0 else 0
        
        # 保存结果
        if save_results and self.results_history:
            self.save_results()
        
        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy
        }
