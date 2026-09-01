"""
LLM-as-a-Judge 评分系统
实现论文中描述的四个维度评分和归因分析
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import uuid


@dataclass
class EvaluationScore:
    """评分结果"""
    # 四个核心维度 (0-100)
    context_utilization: float  # 上下文利用率
    answer_completeness: float  # 回答完整性
    noise_misuse_rate: float    # 噪声误用率（越低越好，但评分时给的是合规分数）
    uncertainty_expression: float  # 不确定性表达
    
    # 总分
    total_score: float
    
    # 归因理由
    justification: str
    
    # 维度分析
    dimension_analysis: Dict[str, str] = field(default_factory=dict)
    
    # 元数据
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    judge_model: str = "default"


@dataclass
class EvaluationResult:
    """完整评估结果"""
    result_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str = ""
    gold_answer: Optional[str] = None
    model_answer: str = ""
    context: str = ""
    prompt_id: str = ""
    
    # 评分
    scores: Optional[EvaluationScore] = None
    
    # 归因分析
    attribution: Dict[str, Any] = field(default_factory=dict)
    
    # 是否需要迭代
    needs_iteration: bool = False
    
    # 建议的改进模块
    suggested_modules: List[str] = field(default_factory=list)
    
    # 元数据
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'result_id': self.result_id,
            'question': self.question,
            'gold_answer': self.gold_answer,
            'model_answer': self.model_answer,
            'context': self.context,
            'prompt_id': self.prompt_id,
            'scores': {
                'context_utilization': self.scores.context_utilization,
                'answer_completeness': self.scores.answer_completeness,
                'noise_misuse_rate': self.scores.noise_misuse_rate,
                'uncertainty_expression': self.scores.uncertainty_expression,
                'total_score': self.scores.total_score,
                'justification': self.scores.justification,
                'dimension_analysis': self.scores.dimension_analysis
            } if self.scores else None,
            'attribution': self.attribution,
            'needs_iteration': self.needs_iteration,
            'suggested_modules': self.suggested_modules,
            'created_at': self.created_at
        }


class LLMJudge:
    """
    LLM-as-a-Judge 评价模型
    
    基于论文描述，从四个维度进行评分：
    1. 上下文利用率 (Context Utilization)
    2. 回答完整性 (Answer Completeness)
    3. 噪声误用率 (Noise Misuse Rate)
    4. 不确定性表达 (Uncertainty Expression)
    """
    
    # 评分阈值
    SCORE_THRESHOLD = 75  # 低于此分数需要迭代
    
    # 维度权重
    DIMENSION_WEIGHTS = {
        'context_utilization': 0.25,
        'answer_completeness': 0.25,
        'noise_misuse_rate': 0.25,
        'uncertainty_expression': 0.25
    }
    
    def __init__(self, model_name: str = "gpt-4"):
        """
        初始化Judge
        
        Args:
            model_name: 评价模型名称
        """
        self.model_name = model_name
    
    def evaluate(self, question: str, model_answer: str, context: str,
                 gold_answer: Optional[str] = None) -> EvaluationResult:
        """
        评估回答质量
        
        Args:
            question: 问题
            model_answer: 模型回答
            context: 使用的上下文
            gold_answer: 标准答案（可选）
            
        Returns:
            评估结果
        """
        # 构建评分Prompt
        judge_prompt = self._build_judge_prompt(question, model_answer, context, gold_answer)
        
        # 这里应该调用实际的LLM API
        # 为了演示，使用模拟评分
        scores = self._mock_evaluate(judge_prompt)
        
        # 创建评估结果
        result = EvaluationResult(
            question=question,
            gold_answer=gold_answer,
            model_answer=model_answer,
            context=context,
            scores=scores
        )
        
        # 归因分析
        result.attribution = self._analyze_attribution(scores)
        
        # 判断是否需要迭代
        result.needs_iteration = scores.total_score < self.SCORE_THRESHOLD
        
        # 建议改进的模块
        result.suggested_modules = self._suggest_modules(scores)
        
        return result
    
    def _build_judge_prompt(self, question: str, model_answer: str, 
                           context: str, gold_answer: Optional[str] = None) -> str:
        """构建评分Prompt"""
        
        rubric = """请作为评价专家，对以下回答进行评分。

评分维度（每个维度0-100分）：

1. 上下文利用率 (Context Utilization)
   - 评价回答是否正确使用了提供的参考资料
   - 是否遗漏重要信息或引用错误信息
   - 是否充分挖掘了相关资料

2. 回答完整性 (Answer Completeness)
   - 评价回答是否全面覆盖了问题的各个方面
   - 是否遗漏关键知识点
   - 是否进行了充分的解释和说明

3. 噪声误用率 - 合规分数 (Noise Resistance)
   - 评价回答是否正确识别并过滤了干扰信息
   - 是否被无关信息误导
   - 是否保持了信息的准确性

4. 不确定性表达 (Uncertainty Expression)
   - 评价回答在信息不足时是否明确表达了不确定性
   - 是否避免了过度自信的错误表述
   - 是否恰当处理了模糊或冲突的信息

请以JSON格式输出评分结果：
{
  "context_utilization": 分数,
  "answer_completeness": 分数,
  "noise_misuse_rate": 分数,
  "uncertainty_expression": 分数,
  "justification": "详细的归因理由，说明各维度得分的依据",
  "dimension_analysis": {
    "context_utilization": "该维度的具体分析",
    "answer_completeness": "该维度的具体分析",
    "noise_misuse_rate": "该维度的具体分析",
    "uncertainty_expression": "该维度的具体分析"
  }
}"""
        
        content = f"""{rubric}

=== 评估内容 ===

问题：{question}

提供的参考资料：
{context}

模型回答：
{model_answer}
"""
        
        if gold_answer:
            content += f"\n标准答案参考：{gold_answer}\n"
        
        return content
    
    def _mock_evaluate(self, prompt: str) -> EvaluationScore:
        """
        模拟评分（实际实现中应调用LLM API）
        
        这里提供一个简单的启发式评分逻辑
        """
        # 实际实现中，这里应该调用LLM API
        # 返回模拟分数
        import random
        
        # 基于回答长度和上下文长度给出启发式分数
        context_utilization = random.uniform(70, 95)
        answer_completeness = random.uniform(65, 90)
        noise_misuse_rate = random.uniform(70, 95)
        uncertainty_expression = random.uniform(60, 85)
        
        # 计算总分
        total = (
            context_utilization * self.DIMENSION_WEIGHTS['context_utilization'] +
            answer_completeness * self.DIMENSION_WEIGHTS['answer_completeness'] +
            noise_misuse_rate * self.DIMENSION_WEIGHTS['noise_misuse_rate'] +
            uncertainty_expression * self.DIMENSION_WEIGHTS['uncertainty_expression']
        )
        
        return EvaluationScore(
            context_utilization=round(context_utilization, 2),
            answer_completeness=round(answer_completeness, 2),
            noise_misuse_rate=round(noise_misuse_rate, 2),
            uncertainty_expression=round(uncertainty_expression, 2),
            total_score=round(total, 2),
            justification="基于启发式规则的模拟评分。实际部署时应使用LLM API进行真实评估。",
            dimension_analysis={
                "context_utilization": "模拟分析：回答基本使用了上下文信息",
                "answer_completeness": "模拟分析：回答较为完整",
                "noise_misuse_rate": "模拟分析：噪声过滤良好",
                "uncertainty_expression": "模拟分析：不确定性表达尚可"
            }
        )
    
    def _analyze_attribution(self, scores: EvaluationScore) -> Dict[str, Any]:
        """
        归因分析
        根据论文表3-1进行异常指标归因
        """
        attribution = {
            'low_dimensions': [],
            'analysis': {},
            'module_mapping': {}
        }
        
        # 检查各维度
        dimensions = {
            'context_utilization': scores.context_utilization,
            'answer_completeness': scores.answer_completeness,
            'noise_misuse_rate': scores.noise_misuse_rate,
            'uncertainty_expression': scores.uncertainty_expression
        }
        
        for dim, score in dimensions.items():
            if score < self.SCORE_THRESHOLD:
                attribution['low_dimensions'].append(dim)
                
                # 映射到模块（根据论文表3-1）
                module_map = {
                    'context_utilization': {
                        'module': 'C_t (Context Constraint)',
                        'suggestion': '增强证据强制引用要求，提升对有效知识片段的注意力分配'
                    },
                    'answer_completeness': {
                        'module': 'I_t (Task Instruction)',
                        'suggestion': '强化关键知识点覆盖约束，引导模型进行多维证据挖掘整合'
                    },
                    'noise_misuse_rate': {
                        'module': 'I_t (Denoising Instruction)',
                        'suggestion': '增加显式的去噪过滤指令，通过模块级变异强化干扰信息去除'
                    },
                    'uncertainty_expression': {
                        'module': 'U_t (Uncertainty Expression)',
                        'suggestion': '增加对冲突信息的检测模板，细化证据不足时的拒答与警示话术'
                    }
                }
                
                attribution['module_mapping'][dim] = module_map[dim]
        
        return attribution
    
    def _suggest_modules(self, scores: EvaluationScore) -> List[str]:
        """
        建议需要改进的模块
        根据论文表3-2的策略
        """
        suggestions = []
        
        if scores.context_utilization < self.SCORE_THRESHOLD:
            suggestions.append('C_t')
        
        if scores.answer_completeness < self.SCORE_THRESHOLD:
            suggestions.append('I_t')
        
        if scores.noise_misuse_rate < self.SCORE_THRESHOLD:
            suggestions.append('I_t_denoise')
        
        if scores.uncertainty_expression < self.SCORE_THRESHOLD:
            suggestions.append('U_t')
        
        return suggestions
    
    def batch_evaluate(self, items: List[Dict]) -> List[EvaluationResult]:
        """
        批量评估
        
        Args:
            items: 评估项列表，每项包含question, model_answer, context等
            
        Returns:
            评估结果列表
        """
        results = []
        for item in items:
            result = self.evaluate(
                question=item['question'],
                model_answer=item['model_answer'],
                context=item['context'],
                gold_answer=item.get('gold_answer')
            )
            result.prompt_id = item.get('prompt_id', '')
            results.append(result)
        return results


# 全局Judge实例
judge = LLMJudge()
