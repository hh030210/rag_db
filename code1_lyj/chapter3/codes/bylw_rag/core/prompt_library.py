"""
Prompt库管理器
管理不同领域和问题类型的Prompt集合
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .prompt_module import StructuredPrompt, PromptModule


class PromptLibrary:
    """
    Prompt库管理器
    管理多个领域和问题类型的Prompt集合
    """
    
    # 定义问题类型
    QUESTION_TYPES = [
        "fact_retrieval",        # 事实检索型
        "subjective_opinion",    # 主观观点型
        "exploratory_open",      # 探索开放型
        "short_answer"           # 简短回答型（适合NQ数据集）
    ]
    
    # 定义领域
    DOMAINS = [
        "psychology",           # 心理学
        "computer_science",     # 计算机科学
        "medicine",             # 医学
        "general"               # 通用
    ]
    
    def __init__(self, base_path: str = "prompts"):
        """
        初始化Prompt库
        
        Args:
            base_path: Prompt库根目录
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 内存中的Prompt缓存
        self._prompts: Dict[str, List[StructuredPrompt]] = {
            qt: [] for qt in self.QUESTION_TYPES
        }
        
        # 加载现有Prompt
        self._load_all_prompts()
    
    def _get_prompt_dir(self, question_type: str) -> Path:
        """获取问题类型的Prompt目录"""
        return self.base_path / question_type
    
    def _load_all_prompts(self):
        """加载所有Prompt"""
        for question_type in self.QUESTION_TYPES:
            prompt_dir = self._get_prompt_dir(question_type)
            if not prompt_dir.exists():
                continue
            
            # 加载该类型下的所有Prompt
            for prompt_file in prompt_dir.glob("*.json"):
                try:
                    prompt = StructuredPrompt.load(str(prompt_file))
                    self._prompts[question_type].append(prompt)
                except Exception as e:
                    print(f"加载Prompt失败 {prompt_file}: {e}")
    
    def add_prompt(self, prompt: StructuredPrompt, save: bool = True) -> str:
        """
        添加Prompt到库
        
        Args:
            prompt: Prompt对象
            save: 是否保存到文件
            
        Returns:
            Prompt ID
        """
        question_type = prompt.question_type
        
        if question_type not in self.QUESTION_TYPES:
            raise ValueError(f"未知的问题类型: {question_type}")
        
        self._prompts[question_type].append(prompt)
        
        if save:
            prompt_dir = self._get_prompt_dir(question_type)
            prompt_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{prompt.prompt_id}_v{prompt.version}.json"
            prompt.save(str(prompt_dir / filename))
        
        return prompt.prompt_id
    
    def get_prompts(self, question_type: str, domain: Optional[str] = None) -> List[StructuredPrompt]:
        """
        获取Prompt列表
        
        Args:
            question_type: 问题类型
            domain: 领域（可选，为None则返回所有领域）
            
        Returns:
            Prompt列表
        """
        prompts = self._prompts.get(question_type, [])
        
        if domain:
            prompts = [p for p in prompts if p.domain == domain]
        
        # 按性能评分排序（如果有）
        prompts.sort(key=lambda p: (p.performance_score or 0), reverse=True)
        
        return prompts
    
    def get_best_prompt(self, question_type: str, domain: Optional[str] = None) -> Optional[StructuredPrompt]:
        """
        获取最佳Prompt
        
        Args:
            question_type: 问题类型
            domain: 领域
            
        Returns:
            最佳Prompt，如果没有则返回None
        """
        prompts = self.get_prompts(question_type, domain)
        
        if not prompts:
            return None
        
        # 返回评分最高的激活Prompt
        active_prompts = [p for p in prompts if p.is_active]
        if active_prompts:
            return active_prompts[0]
        
        return prompts[0]
    
    def update_prompt_score(self, prompt_id: str, score: float):
        """
        更新Prompt评分
        
        Args:
            prompt_id: Prompt ID
            score: 新评分
        """
        for question_type, prompts in self._prompts.items():
            for prompt in prompts:
                if prompt.prompt_id == prompt_id:
                    prompt.performance_score = score
                    # 重新保存
                    prompt_dir = self._get_prompt_dir(question_type)
                    filename = f"{prompt.prompt_id}_v{prompt.version}.json"
                    prompt.save(str(prompt_dir / filename))
                    return True
        return False
    
    def deactivate_prompt(self, prompt_id: str):
        """停用Prompt"""
        for prompts in self._prompts.values():
            for prompt in prompts:
                if prompt.prompt_id == prompt_id:
                    prompt.is_active = False
                    return True
        return False
    
    def get_prompt_stats(self) -> Dict:
        """获取Prompt库统计信息"""
        stats = {
            "total_prompts": 0,
            "by_type": {},
            "by_domain": {}
        }
        
        for question_type, prompts in self._prompts.items():
            count = len(prompts)
            stats["total_prompts"] += count
            stats["by_type"][question_type] = {
                "count": count,
                "active": sum(1 for p in prompts if p.is_active),
                "avg_score": sum(p.performance_score or 0 for p in prompts) / count if count > 0 else 0
            }
        
        return stats
    
    def initialize_default_prompts(self, question_type: str, domain: str = "general"):
        """
        初始化默认Prompt集合（每个问题类型5个）
        
        Args:
            question_type: 问题类型
            domain: 领域
        """
        from ..utils.default_prompts import get_default_prompts
        
        prompts = get_default_prompts(question_type, domain)
        
        for prompt in prompts:
            self.add_prompt(prompt, save=True)
        
        print(f"✓ 已为 {question_type}/{domain} 初始化 {len(prompts)} 个Prompt")


# 全局单例
prompt_library = PromptLibrary()
