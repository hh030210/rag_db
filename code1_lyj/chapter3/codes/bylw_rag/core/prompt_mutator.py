"""
Prompt变异器
实现Prompt模块的自动变异和迭代
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import difflib

from .prompt_module import PromptModule, StructuredPrompt


class MutationRecorder:
    """
    变异记录器
    记录Prompt变异过程中的评分和修改
    """
    
    def __init__(self, record_dir: str = "mutation_records"):
        """
        初始化记录器
        
        Args:
            record_dir: 记录保存目录
        """
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.current_record = None
        self.current_record_file = None
    
    def start_record(self, prompt: StructuredPrompt, question: str,
                     evaluation_scores: Dict[str, Any]) -> str:
        """
        开始记录一个新的变异过程
        
        Args:
            prompt: 原始Prompt
            question: 问题
            evaluation_scores: 评估分数
            
        Returns:
            记录ID
        """
        record_id = f"{prompt.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_record = {
            "record_id": record_id,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "original_prompt": {
                "prompt_id": prompt.prompt_id,
                "name": prompt.name,
                "question_type": prompt.question_type,
                "domain": prompt.domain,
                "modules": {
                    "P_sys": prompt.P_sys.to_dict(),
                    "I_t": prompt.I_t.to_dict(),
                    "C_t": prompt.C_t.to_dict(),
                    "F_t": prompt.F_t.to_dict(),
                    "U_t": prompt.U_t.to_dict()
                }
            },
            "original_evaluation": evaluation_scores,
            "mutations": [],
            "final_prompt": None,
            "final_evaluation": None
        }
        
        self.current_record_file = self.record_dir / f"{record_id}.json"
        self._save_record()
        
        return record_id
    
    def add_mutation(self, iteration: int, module_type: str,
                     original_module: PromptModule,
                     mutated_module: PromptModule,
                     strategy: Dict, reason: str,
                     evaluation_scores: Optional[Dict] = None):
        """
        添加一次变异记录
        
        Args:
            iteration: 迭代次数
            module_type: 变异的模块类型
            original_module: 原始模块
            mutated_module: 变异后的模块
            strategy: 变异策略
            reason: 变异原因
            evaluation_scores: 变异后的评估分数（可选）
        """
        if self.current_record is None:
            raise ValueError("请先调用start_record开始记录")
        
        # 计算修改差异
        diff = self._compute_diff(original_module.content, mutated_module.content)
        
        mutation_entry = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "module_type": module_type,
            "original": {
                "module_id": original_module.id if hasattr(original_module, 'id') else None,
                "name": original_module.name,
                "content": original_module.content,
                "mutation_count": original_module.mutation_count
            },
            "mutated": {
                "module_id": mutated_module.id if hasattr(mutated_module, 'id') else None,
                "name": mutated_module.name,
                "content": mutated_module.content,
                "mutation_count": mutated_module.mutation_count
            },
            "changes": {
                "strategy_name": strategy.get('name', 'unknown'),
                "strategy_description": strategy.get('description', ''),
                "reason": reason,
                "diff": diff,
                "added_lines": diff['added'],
                "removed_lines": diff['removed']
            },
            "evaluation_after_mutation": evaluation_scores
        }
        
        self.current_record["mutations"].append(mutation_entry)
        self._save_record()
    
    def finalize_record(self, final_prompt: StructuredPrompt,
                       final_evaluation: Dict[str, Any]):
        """
        完成记录
        
        Args:
            final_prompt: 最终Prompt
            final_evaluation: 最终评估分数
        """
        if self.current_record is None:
            return
        
        self.current_record["final_prompt"] = {
            "prompt_id": final_prompt.prompt_id,
            "name": final_prompt.name,
            "modules": {
                "P_sys": final_prompt.P_sys.to_dict(),
                "I_t": final_prompt.I_t.to_dict(),
                "C_t": final_prompt.C_t.to_dict(),
                "F_t": final_prompt.F_t.to_dict(),
                "U_t": final_prompt.U_t.to_dict()
            }
        }
        self.current_record["final_evaluation"] = final_evaluation
        self.current_record["completed_at"] = datetime.now().isoformat()
        
        self._save_record()
        
        print(f"\n✓ 变异记录已保存: {self.current_record_file}")
    
    def _compute_diff(self, original: str, mutated: str) -> Dict:
        """
        计算文本差异
        
        Args:
            original: 原始文本
            mutated: 变异后的文本
            
        Returns:
            差异信息
        """
        original_lines = original.split('\n')
        mutated_lines = mutated.split('\n')
        
        diff = list(difflib.unified_diff(
            original_lines, mutated_lines,
            fromfile='original', tofile='mutated',
            lineterm=''
        ))
        
        added = [line[1:] for line in diff if line.startswith('+') and not line.startswith('+++')]
        removed = [line[1:] for line in diff if line.startswith('-') and not line.startswith('---')]
        
        return {
            "unified_diff": '\n'.join(diff),
            "added": added,
            "removed": removed,
            "num_added": len(added),
            "num_removed": len(removed)
        }
    
    def _save_record(self):
        """保存记录到文件"""
        if self.current_record and self.current_record_file:
            with open(self.current_record_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_record, f, ensure_ascii=False, indent=2)
    
    def get_record(self, record_id: str) -> Optional[Dict]:
        """
        获取指定记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            记录内容
        """
        record_file = self.record_dir / f"{record_id}.json"
        if record_file.exists():
            with open(record_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def list_records(self) -> List[Dict]:
        """
        列出所有记录
        
        Returns:
            记录摘要列表
        """
        records = []
        for record_file in sorted(self.record_dir.glob("*.json"), reverse=True):
            with open(record_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                records.append({
                    "record_id": data.get("record_id"),
                    "timestamp": data.get("timestamp"),
                    "question": data.get("question", "")[:50] + "...",
                    "prompt_name": data.get("original_prompt", {}).get("name"),
                    "num_mutations": len(data.get("mutations", [])),
                    "original_score": data.get("original_evaluation", {}).get("total_score"),
                    "final_score": data.get("final_evaluation", {}).get("total_score")
                })
        return records


class PromptMutator:
    """
    Prompt变异器
    
    根据评估结果，对Prompt模块进行定向变异
    """
    
    def __init__(self, mutations_dir: str = "mutations"):
        """
        初始化变异器
        
        Args:
            mutations_dir: 变异记录保存目录
        """
        self.mutations_dir = Path(mutations_dir)
        self.mutations_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化记录器
        self.recorder = MutationRecorder()
        
        # 加载变异策略
        self.mutation_strategies = self._load_mutation_strategies()
        
        # 当前记录ID
        self.current_record_id = None
    
    def _load_mutation_strategies(self) -> Dict[str, List[Dict]]:
        """加载变异策略库"""
        return {
            'C_t': [  # 上下文约束模块变异策略
                {
                    'name': 'enhance_citation',
                    'description': '增强引用要求',
                    'template': '在原有基础上增加：\n- 每个关键信息必须标注来源文档ID\n- 禁止引用未提供的参考资料\n- 对引用的准确性负责'
                },
                {
                    'name': 'strict_scope',
                    'description': '严格范围限制',
                    'template': '在原有基础上增加：\n- 仅回答参考资料中明确涉及的内容\n- 对于参考资料外的信息，明确标注"超出知识范围"\n- 优先使用相关性最高的文档片段'
                },
                {
                    'name': 'multi_source_verification',
                    'description': '多源验证',
                    'template': '在原有基础上增加：\n- 重要信息需要至少2个来源支持\n- 标注单一来源信息的可靠性等级\n- 对比不同来源的一致性'
                }
            ],
            'I_t': [  # 任务指令模块变异策略
                {
                    'name': 'enhance_coverage',
                    'description': '增强覆盖要求',
                    'template': '在原有基础上增加：\n- 全面覆盖问题的所有子问题\n- 识别并回答隐含的问题要素\n- 提供多角度分析'
                },
                {
                    'name': 'step_by_step',
                    'description': '分步推理',
                    'template': '在原有基础上增加：\n- 按逻辑步骤分解问题\n- 每步明确说明推理依据\n- 综合各步骤得出最终答案'
                },
                {
                    'name': 'evidence_driven',
                    'description': '证据驱动',
                    'template': '在原有基础上增加：\n- 先列出所有相关证据\n- 基于证据进行推理\n- 证据不足时明确说明'
                }
            ],
            'I_t_denoise': [  # 去噪指令变异策略
                {
                    'name': 'explicit_filter',
                    'description': '显式过滤',
                    'template': '在原有基础上增加：\n- 明确识别并排除无关信息\n- 标注被过滤的噪声内容\n- 说明过滤的理由'
                },
                {
                    'name': 'relevance_check',
                    'description': '相关性检查',
                    'template': '在原有基础上增加：\n- 评估每个信息片段与问题的相关性\n- 仅使用高相关度的信息\n- 对边缘相关信息标注不确定性'
                },
                {
                    'name': 'conflict_resolution',
                    'description': '冲突解决',
                    'template': '在原有基础上增加：\n- 识别信息冲突\n- 评估各来源的可信度\n- 选择最可靠的信息或说明冲突'
                }
            ],
            'U_t': [  # 不确定性表达模块变异策略
                {
                    'name': 'confidence_scale',
                    'description': '置信度分级',
                    'template': '在原有基础上增加：\n- 使用明确的置信度等级（高/中/低）\n- 说明置信度的判断依据\n- 低置信度时建议进一步验证'
                },
                {
                    'name': 'uncertainty_templates',
                    'description': '不确定模板',
                    'template': '在原有基础上增加：\n- 信息不足时："根据现有资料，无法确定..."\n- 存在冲突时："不同来源对此有不同说法..."\n- 需要验证时："建议进一步核实..."'
                },
                {
                    'name': 'knowledge_boundary',
                    'description': '知识边界',
                    'template': '在原有基础上增加：\n- 明确标注已知和未知的边界\n- 说明信息的时效性限制\n- 承认知识的局限性'
                }
            ],
            'F_t': [  # 格式约束模块变异策略
                {
                    'name': 'structured_output',
                    'description': '结构化输出',
                    'template': '要求以JSON格式输出，包含：answer, confidence, sources, reasoning字段'
                },
                {
                    'name': 'markdown_format',
                    'description': 'Markdown格式',
                    'template': '使用Markdown格式，包括：标题、列表、引用等结构化元素'
                }
            ]
        }
    
    def start_mutation_session(self, prompt: StructuredPrompt, question: str,
                               evaluation_scores: Dict[str, Any]) -> str:
        """
        开始变异会话
        
        Args:
            prompt: 原始Prompt
            question: 问题
            evaluation_scores: 评估分数
            
        Returns:
            记录ID
        """
        self.current_record_id = self.recorder.start_record(prompt, question, evaluation_scores)
        return self.current_record_id
    
    def finalize_mutation_session(self, final_prompt: StructuredPrompt,
                                  final_evaluation: Dict[str, Any]):
        """
        结束变异会话
        
        Args:
            final_prompt: 最终Prompt
            final_evaluation: 最终评估分数
        """
        self.recorder.finalize_record(final_prompt, final_evaluation)
        self.current_record_id = None
    
    def mutate_module(self, module: PromptModule, strategy_name: Optional[str] = None,
                     reason: str = "", iteration: int = 0,
                     evaluation_scores: Optional[Dict] = None) -> Optional[PromptModule]:
        """
        变异单个模块
        
        Args:
            module: 要变异的模块
            strategy_name: 策略名称（为None则自动选择）
            reason: 变异原因
            iteration: 迭代次数
            evaluation_scores: 变异后的评估分数
            
        Returns:
            变异后的新模块，如果已达最大变异次数则返回None
        """
        # 检查变异次数
        if module.mutation_count >= 2:
            print(f"模块 {module.name} 已达最大变异次数(2次)")
            return None
        
        # 获取变异策略
        strategies = self.mutation_strategies.get(module.module_type, [])
        if not strategies:
            print(f"模块类型 {module.module_type} 没有定义变异策略")
            return None
        
        # 选择策略
        if strategy_name:
            strategy = next((s for s in strategies if s['name'] == strategy_name), strategies[0])
        else:
            # 根据变异次数选择不同策略
            strategy = strategies[min(module.mutation_count, len(strategies) - 1)]
        
        # 生成新内容
        new_content = self._apply_mutation(module.content, strategy)
        
        # 创建变异后的模块
        mutated_module = module.mutate(new_content, reason)
        
        # 记录变异（如果有活跃会话）
        if self.current_record_id:
            self.recorder.add_mutation(
                iteration=iteration,
                module_type=module.module_type,
                original_module=module,
                mutated_module=mutated_module,
                strategy=strategy,
                reason=reason,
                evaluation_scores=evaluation_scores
            )
        
        # 保存到mutations目录
        self._record_mutation(module, mutated_module, strategy, reason)
        
        return mutated_module
    
    def _apply_mutation(self, original_content: str, strategy: Dict) -> str:
        """应用变异策略"""
        # 简单的变异策略：在原有内容后添加策略模板
        template = strategy['template']
        
        # 如果内容已包含类似指令，则替换
        if '在原有基础上增加：' in template:
            # 提取新增部分
            additions = template.split('在原有基础上增加：')[1].strip()
            return f"{original_content}\n\n【增强指令】\n{additions}"
        
        return f"{original_content}\n\n{template}"
    
    def _record_mutation(self, original: PromptModule, mutated: PromptModule,
                        strategy: Dict, reason: str):
        """记录变异历史"""
        record = {
            'mutation_id': str(uuid.uuid4())[:8],
            'timestamp': datetime.now().isoformat(),
            'original_module': original.to_dict(),
            'mutated_module': mutated.to_dict(),
            'strategy': strategy,
            'reason': reason
        }
        
        # 保存到文件
        mutation_file = self.mutations_dir / f"mutation_{record['mutation_id']}.json"
        with open(mutation_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 变异已记录: {mutation_file}")
    
    def mutate_prompt(self, prompt: StructuredPrompt, 
                     target_modules: List[str],
                     evaluation_result: Optional[Dict] = None,
                     iteration: int = 0) -> Optional[StructuredPrompt]:
        """
        变异整个Prompt（替换指定模块）
        
        Args:
            prompt: 原始Prompt
            target_modules: 要变异的模块类型列表，如 ['C_t', 'I_t']
            evaluation_result: 评估结果（用于指导变异）
            iteration: 迭代次数
            
        Returns:
            变异后的新Prompt
        """
        new_prompt = prompt
        
        for module_type in target_modules:
            # 获取当前模块
            current_module = prompt.get_module(module_type)
            
            # 确定变异原因
            reason = ""
            if evaluation_result and 'attribution' in evaluation_result:
                mapping = evaluation_result['attribution'].get('module_mapping', {})
                for dim, info in mapping.items():
                    if module_type in info.get('module', ''):
                        reason = info.get('suggestion', '')
                        break
            
            # 变异模块
            mutated_module = self.mutate_module(
                current_module, 
                reason=reason,
                iteration=iteration
            )
            
            if mutated_module:
                # 创建新Prompt
                new_prompt = new_prompt.replace_module(mutated_module)
                print(f"✓ 模块 {module_type} 已变异: {current_module.name} -> {mutated_module.name}")
            else:
                print(f"✗ 模块 {module_type} 变异失败")
        
        return new_prompt
    
    def get_mutation_history(self, module_id: Optional[str] = None) -> List[Dict]:
        """
        获取变异历史
        
        Args:
            module_id: 模块ID（为None则返回所有历史）
            
        Returns:
            变异记录列表
        """
        history = []
        
        for mutation_file in self.mutations_dir.glob("mutation_*.json"):
            with open(mutation_file, 'r', encoding='utf-8') as f:
                record = json.load(f)
                
                if module_id is None or record['original_module']['id'] == module_id:
                    history.append(record)
        
        # 按时间排序
        history.sort(key=lambda x: x['timestamp'])
        return history
    
    def get_module_lineage(self, module_id: str) -> List[Dict]:
        """
        获取模块的变异谱系
        
        Args:
            module_id: 模块ID
            
        Returns:
            从原始模块到当前模块的变异链
        """
        lineage = []
        current_id = module_id
        
        # 向前追溯
        while current_id:
            history = self.get_mutation_history(current_id)
            if history:
                record = history[-1]  # 最新的变异记录
                lineage.insert(0, record)
                current_id = record['original_module'].get('parent_id')
            else:
                break
        
        return lineage
    
    def get_session_records(self) -> List[Dict]:
        """
        获取所有会话记录摘要
        
        Returns:
            记录摘要列表
        """
        return self.recorder.list_records()


# 全局变异器实例
mutator = PromptMutator()
