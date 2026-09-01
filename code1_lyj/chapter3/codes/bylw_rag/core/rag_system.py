"""
主RAG系统
整合向量检索、Prompt管理、评估和迭代
"""
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .prompt_module import StructuredPrompt
from .prompt_library import PromptLibrary
from .prompt_mutator import PromptMutator
from ..evaluation.judge import LLMJudge, EvaluationResult
from ..vector_stores.chroma_store import VectorStoreManager


@dataclass
class RAGResponse:
    """RAG回答结果"""
    question: str
    answer: str
    answer_json: Dict[str, Any]
    retrieved_contexts: List[Dict]
    prompt_used: StructuredPrompt
    evaluation: Optional[EvaluationResult] = None


class BYLWRAGSystem:
    """
    毕业论文RAG系统
    
    核心功能：
    1. 向量检索
    2. Prompt选择和管理
    3. 回答生成
    4. 自动评估
    5. Prompt迭代优化
    """
    
    def __init__(self, 
                 dataset_name: str,
                 question_type: str = "fact_retrieval",
                 domain: str = "general"):
        """
        初始化RAG系统
        
        Args:
            dataset_name: 数据集名称（选择向量库）
            question_type: 问题类型
            domain: 领域
        """
        self.dataset_name = dataset_name
        self.question_type = question_type
        self.domain = domain
        
        # 初始化组件
        self.vector_store = VectorStoreManager().get_store(dataset_name)
        self.prompt_library = PromptLibrary()
        self.mutator = PromptMutator()
        self.judge = LLMJudge()
        
        # 当前使用的Prompt
        self.current_prompt: Optional[StructuredPrompt] = None
        
        # 迭代历史
        self.iteration_history: List[Dict] = []
        
        print(f"✓ RAG系统初始化完成")
        print(f"  - 数据集: {dataset_name}")
        print(f"  - 问题类型: {question_type}")
        print(f"  - 领域: {domain}")
    
    def initialize_prompts(self):
        """初始化Prompt库（如果不存在）"""
        existing_prompts = self.prompt_library.get_prompts(
            self.question_type, 
            self.domain
        )
        
        if not existing_prompts:
            print(f"\n初始化 {self.question_type}/{self.domain} 的Prompt库...")
            self.prompt_library.initialize_default_prompts(
                self.question_type,
                self.domain
            )
        else:
            print(f"✓ 已存在 {len(existing_prompts)} 个Prompt")
    
    def select_prompt(self, prompt_id: Optional[str] = None) -> StructuredPrompt:
        """
        选择Prompt
        
        Args:
            prompt_id: 指定Prompt ID（为None则选择最佳）
            
        Returns:
            选中的Prompt
        """
        if prompt_id:
            # 查找指定Prompt
            prompts = self.prompt_library.get_prompts(self.question_type, self.domain)
            for p in prompts:
                if p.prompt_id == prompt_id:
                    self.current_prompt = p
                    return p
            raise ValueError(f"未找到Prompt: {prompt_id}")
        else:
            # 选择最佳Prompt
            self.current_prompt = self.prompt_library.get_best_prompt(
                self.question_type,
                self.domain
            )
            
            if not self.current_prompt:
                # 初始化默认Prompt
                self.initialize_prompts()
                self.current_prompt = self.prompt_library.get_best_prompt(
                    self.question_type,
                    self.domain
                )
            
            return self.current_prompt
    
    def answer(self, question: str, context: str = "", 
               use_retrieval: bool = True) -> RAGResponse:
        """
        回答问题
        
        Args:
            question: 问题
            context: 外部提供的上下文（可选）
            use_retrieval: 是否使用向量检索
            
        Returns:
            回答结果
        """
        # 确保有Prompt
        if not self.current_prompt:
            self.select_prompt()
        
        # 检索上下文
        retrieved_contexts = []
        if use_retrieval:
            # 这里应该调用嵌入模型获取query embedding
            # 暂时使用空列表
            retrieved_contexts = []
            context = "\n\n".join([c['text'] for c in retrieved_contexts])
        
        # 编译Prompt
        compiled_prompt = self.current_prompt.compile(
            context=context,
            question=question
        )
        
        # 这里应该调用LLM API生成回答
        # 暂时返回模拟回答
        answer_json = {
            "answer": f"这是基于Prompt [{self.current_prompt.name}] 生成的模拟回答",
            "confidence": 0.85,
            "sources": []
        }
        
        return RAGResponse(
            question=question,
            answer=answer_json['answer'],
            answer_json=answer_json,
            retrieved_contexts=retrieved_contexts,
            prompt_used=self.current_prompt
        )
    
    def evaluate_and_iterate(self, response: RAGResponse, 
                            gold_answer: Optional[str] = None,
                            auto_iterate: bool = True,
                            iteration: int = 0,
                            record_mutation: bool = True) -> RAGResponse:
        """
        评估回答并迭代优化
        
        Args:
            response: 回答结果
            gold_answer: 标准答案
            auto_iterate: 是否自动迭代
            iteration: 当前迭代次数
            record_mutation: 是否记录变异过程
            
        Returns:
            可能已更新的回答结果
        """
        # 评估
        context_text = "\n\n".join([c['text'] for c in response.retrieved_contexts])
        
        evaluation = self.judge.evaluate(
            question=response.question,
            model_answer=response.answer,
            context=context_text,
            gold_answer=gold_answer
        )
        
        evaluation.prompt_id = response.prompt_used.prompt_id
        response.evaluation = evaluation
        
        # 准备评估分数字典
        eval_scores = {
            'total_score': evaluation.scores.total_score if evaluation.scores else 0,
            'context_utilization': evaluation.scores.context_utilization if evaluation.scores else 0,
            'answer_completeness': evaluation.scores.answer_completeness if evaluation.scores else 0,
            'noise_misuse_rate': evaluation.scores.noise_misuse_rate if evaluation.scores else 0,
            'uncertainty_expression': evaluation.scores.uncertainty_expression if evaluation.scores else 0,
            'justification': evaluation.scores.justification if evaluation.scores else ""
        }
        
        # 记录迭代
        iteration_record = {
            'timestamp': datetime.now().isoformat(),
            'prompt_id': response.prompt_used.prompt_id,
            'prompt_name': response.prompt_used.name,
            'question': response.question,
            'answer': response.answer,
            'scores': eval_scores,
            'needs_iteration': evaluation.needs_iteration,
            'suggested_modules': evaluation.suggested_modules
        }
        self.iteration_history.append(iteration_record)
        
        # 更新Prompt评分
        if evaluation.scores:
            self.prompt_library.update_prompt_score(
                response.prompt_used.prompt_id,
                evaluation.scores.total_score
            )
        
        # 自动迭代
        if auto_iterate and evaluation.needs_iteration:
            print(f"\n{'='*60}")
            print("触发Prompt迭代优化")
            print(f"{'='*60}")
            print(f"当前总分: {evaluation.scores.total_score:.2f}")
            print(f"建议改进模块: {evaluation.suggested_modules}")
            
            # 如果是第一次迭代，开始记录变异会话
            if record_mutation and iteration == 0:
                self.mutator.start_mutation_session(
                    prompt=response.prompt_used,
                    question=response.question,
                    evaluation_scores=eval_scores
                )
            
            # 执行变异
            new_prompt = self.mutator.mutate_prompt(
                response.prompt_used,
                evaluation.suggested_modules,
                evaluation.to_dict(),
                iteration=iteration
            )
            
            if new_prompt:
                # 添加到库
                new_prompt_id = self.prompt_library.add_prompt(new_prompt)
                print(f"✓ 生成新Prompt: {new_prompt_id}")
                
                # 切换到新Prompt
                self.current_prompt = new_prompt
                
                # 重新回答
                print("使用新Prompt重新生成回答...")
                new_response = self.answer(response.question, context_text, use_retrieval=False)
                new_response.evaluation = response.evaluation
                
                return new_response
            else:
                # 变异失败，结束记录
                if record_mutation and iteration == 0:
                    self.mutator.finalize_mutation_session(
                        final_prompt=response.prompt_used,
                        final_evaluation=eval_scores
                    )
        else:
            # 不需要迭代，结束记录
            if record_mutation and iteration == 0 and self.mutator.current_record_id:
                self.mutator.finalize_mutation_session(
                    final_prompt=response.prompt_used,
                    final_evaluation=eval_scores
                )
        
        return response
    
    def run_qa_cycle(self, question: str, gold_answer: Optional[str] = None,
                    max_iterations: int = 2) -> RAGResponse:
        """
        运行完整的问答-评估-迭代周期
        
        Args:
            question: 问题
            gold_answer: 标准答案
            max_iterations: 最大迭代次数
            
        Returns:
            最终回答结果
        """
        print(f"\n{'='*60}")
        print(f"问答周期开始")
        print(f"{'='*60}")
        print(f"问题: {question}")
        
        response = None
        
        for i in range(max_iterations):
            print(f"\n--- 迭代 {i+1}/{max_iterations} ---")
            
            # 生成回答
            if i == 0:
                response = self.answer(question)
            else:
                # 使用当前选中的Prompt（可能已更新）
                response = self.answer(question, use_retrieval=False)
            
            print(f"使用Prompt: {response.prompt_used.name}")
            print(f"回答: {response.answer[:100]}...")
            
            # 评估和迭代
            response = self.evaluate_and_iterate(
                response, 
                gold_answer, 
                auto_iterate=True,
                iteration=i,
                record_mutation=True
            )
            
            # 检查是否需要继续迭代
            if not response.evaluation or not response.evaluation.needs_iteration:
                print(f"\n✓ 回答质量达标，停止迭代")
                # 结束变异记录
                if self.mutator.current_record_id and i > 0:
                    final_scores = {
                        'total_score': response.evaluation.scores.total_score if response.evaluation.scores else 0,
                        'context_utilization': response.evaluation.scores.context_utilization if response.evaluation.scores else 0,
                        'answer_completeness': response.evaluation.scores.answer_completeness if response.evaluation.scores else 0,
                        'noise_misuse_rate': response.evaluation.scores.noise_misuse_rate if response.evaluation.scores else 0,
                        'uncertainty_expression': response.evaluation.scores.uncertainty_expression if response.evaluation.scores else 0
                    }
                    self.mutator.finalize_mutation_session(
                        final_prompt=response.prompt_used,
                        final_evaluation=final_scores
                    )
                break
            
            if i == max_iterations - 1:
                print(f"\n! 达到最大迭代次数")
                # 结束变异记录
                if self.mutator.current_record_id:
                    final_scores = {
                        'total_score': response.evaluation.scores.total_score if response.evaluation.scores else 0,
                        'context_utilization': response.evaluation.scores.context_utilization if response.evaluation.scores else 0,
                        'answer_completeness': response.evaluation.scores.answer_completeness if response.evaluation.scores else 0,
                        'noise_misuse_rate': response.evaluation.scores.noise_misuse_rate if response.evaluation.scores else 0,
                        'uncertainty_expression': response.evaluation.scores.uncertainty_expression if response.evaluation.scores else 0
                    }
                    self.mutator.finalize_mutation_session(
                        final_prompt=response.prompt_used,
                        final_evaluation=final_scores
                    )
        
        # 显示最终结果
        print(f"\n{'='*60}")
        print("最终回答")
        print(f"{'='*60}")
        print(f"回答: {response.answer}")
        if response.evaluation and response.evaluation.scores:
            print(f"评分: {response.evaluation.scores.total_score:.2f}")
        
        return response
    
    def save_session(self, filepath: str):
        """保存会话历史"""
        session_data = {
            'dataset_name': self.dataset_name,
            'question_type': self.question_type,
            'domain': self.domain,
            'current_prompt_id': self.current_prompt.prompt_id if self.current_prompt else None,
            'iteration_history': self.iteration_history,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 会话已保存: {filepath}")
    
    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        return {
            'dataset': self.dataset_name,
            'question_type': self.question_type,
            'domain': self.domain,
            'prompt_library': self.prompt_library.get_prompt_stats(),
            'iteration_count': len(self.iteration_history),
            'current_prompt': self.current_prompt.name if self.current_prompt else None
        }
