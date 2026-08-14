"""
模块化Prompt系统
实现论文中描述的 P = <P_sys, I_t, C_t, F_t, U_t> 结构
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json
import uuid
from datetime import datetime


@dataclass
class PromptModule:
    """
    Prompt模块基类
    支持可拆卸、可变异的功能模块
    """
    name: str
    content: str
    module_type: str  # 'P_sys', 'I_t', 'C_t', 'F_t', 'U_t'
    version: int = 1
    mutation_count: int = 0  # 变异次数（最多2次）
    parent_id: Optional[str] = None  # 父模块ID（用于追踪变异来源）
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not hasattr(self, 'id'):
            self.id = str(uuid.uuid4())[:8]
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'id': getattr(self, 'id', str(uuid.uuid4())[:8]),
            'name': self.name,
            'content': self.content,
            'module_type': self.module_type,
            'version': self.version,
            'mutation_count': self.mutation_count,
            'parent_id': self.parent_id,
            'created_at': self.created_at,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PromptModule':
        """从字典创建"""
        module = cls(
            name=data['name'],
            content=data['content'],
            module_type=data['module_type'],
            version=data.get('version', 1),
            mutation_count=data.get('mutation_count', 0),
            parent_id=data.get('parent_id'),
            created_at=data.get('created_at', datetime.now().isoformat()),
            metadata=data.get('metadata', {})
        )
        if 'id' in data:
            module.id = data['id']
        return module
    
    def mutate(self, new_content: str, mutation_reason: str = "") -> 'PromptModule':
        """
        创建变异后的新模块
        
        Args:
            new_content: 变异后的内容
            mutation_reason: 变异原因
            
        Returns:
            新的PromptModule实例
        """
        if self.mutation_count >= 2:
            raise ValueError(f"模块 {self.name} 已达到最大变异次数(2次)")
        
        new_module = PromptModule(
            name=f"{self.name}_v{self.version + 1}",
            content=new_content,
            module_type=self.module_type,
            version=self.version + 1,
            mutation_count=self.mutation_count + 1,
            parent_id=getattr(self, 'id', None),
            metadata={
                **self.metadata,
                'mutation_reason': mutation_reason,
                'parent_content': self.content
            }
        )
        return new_module


@dataclass
class StructuredPrompt:
    """
    结构化Prompt
    P = <P_sys, I_t, C_t, F_t, U_t>
    """
    name: str
    question_type: str  # 问题类型：fact_retrieval, subjective_opinion等
    domain: str  # 领域：psychology, computer_science, medicine等
    
    # 五个核心模块
    P_sys: PromptModule  # System Prompt：系统提示词（不可拆卸）
    I_t: PromptModule    # Task Instruction：任务指令
    C_t: PromptModule    # Context Constraint：上下文约束
    F_t: PromptModule    # Format Constraint：格式约束
    U_t: PromptModule    # Uncertainty Expression：不确定性表达
    
    prompt_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1
    is_active: bool = True  # 是否激活
    performance_score: Optional[float] = None  # 性能评分
    
    def __post_init__(self):
        """验证模块类型"""
        assert self.P_sys.module_type == 'P_sys', "P_sys必须是System Prompt类型"
        assert self.I_t.module_type == 'I_t', "I_t必须是Task Instruction类型"
        assert self.C_t.module_type == 'C_t', "C_t必须是Context Constraint类型"
        assert self.F_t.module_type == 'F_t', "F_t必须是Format Constraint类型"
        assert self.U_t.module_type == 'U_t', "U_t必须是Uncertainty Expression类型"
    
    def compile(self, context: str = "", question: str = "") -> str:
        """
        编译Prompt为字符串
        
        Args:
            context: 检索到的上下文
            question: 用户问题
            
        Returns:
            完整的Prompt字符串
        """
        # 替换模板变量
        i_t_content = self.I_t.content.replace("{{question}}", question)
        c_t_content = self.C_t.content.replace("{{context}}", context)
        
        compiled = f"""{self.P_sys.content}

{i_t_content}

{c_t_content}

{self.F_t.content}

{self.U_t.content}"""
        
        return compiled
    
    def compile_with_variables(self, variables: Dict[str, str]) -> str:
        """
        使用变量字典编译Prompt
        
        Args:
            variables: 变量字典，如 {'question': '...', 'context': '...'}
            
        Returns:
            完整的Prompt字符串
        """
        content = self.compile(
            context=variables.get('context', ''),
            question=variables.get('question', '')
        )
        
        # 替换其他变量
        for key, value in variables.items():
            content = content.replace(f"{{{{{key}}}}}", value)
        
        return content
    
    def get_module(self, module_type: str) -> PromptModule:
        """获取指定类型的模块"""
        module_map = {
            'P_sys': self.P_sys,
            'I_t': self.I_t,
            'C_t': self.C_t,
            'F_t': self.F_t,
            'U_t': self.U_t
        }
        return module_map[module_type]
    
    def replace_module(self, new_module: PromptModule) -> 'StructuredPrompt':
        """
        替换模块（创建新Prompt）
        
        Args:
            new_module: 新模块
            
        Returns:
            新的StructuredPrompt实例
        """
        kwargs = {
            'name': f"{self.name}_modified",
            'question_type': self.question_type,
            'domain': self.domain,
            'P_sys': self.P_sys,
            'I_t': self.I_t,
            'C_t': self.C_t,
            'F_t': self.F_t,
            'U_t': self.U_t,
            'version': self.version + 1
        }
        
        # 替换对应模块
        if new_module.module_type == 'P_sys':
            kwargs['P_sys'] = new_module
        elif new_module.module_type == 'I_t':
            kwargs['I_t'] = new_module
        elif new_module.module_type == 'C_t':
            kwargs['C_t'] = new_module
        elif new_module.module_type == 'F_t':
            kwargs['F_t'] = new_module
        elif new_module.module_type == 'U_t':
            kwargs['U_t'] = new_module
        
        return StructuredPrompt(**kwargs)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'prompt_id': self.prompt_id,
            'name': self.name,
            'question_type': self.question_type,
            'domain': self.domain,
            'version': self.version,
            'created_at': self.created_at,
            'is_active': self.is_active,
            'performance_score': self.performance_score,
            'modules': {
                'P_sys': self.P_sys.to_dict(),
                'I_t': self.I_t.to_dict(),
                'C_t': self.C_t.to_dict(),
                'F_t': self.F_t.to_dict(),
                'U_t': self.U_t.to_dict()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StructuredPrompt':
        """从字典创建"""
        modules = data['modules']
        return cls(
            name=data['name'],
            question_type=data['question_type'],
            domain=data['domain'],
            P_sys=PromptModule.from_dict(modules['P_sys']),
            I_t=PromptModule.from_dict(modules['I_t']),
            C_t=PromptModule.from_dict(modules['C_t']),
            F_t=PromptModule.from_dict(modules['F_t']),
            U_t=PromptModule.from_dict(modules['U_t']),
            prompt_id=data.get('prompt_id', str(uuid.uuid4())[:8]),
            created_at=data.get('created_at', datetime.now().isoformat()),
            version=data.get('version', 1),
            is_active=data.get('is_active', True),
            performance_score=data.get('performance_score')
        )
    
    def save(self, filepath: str):
        """保存到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'StructuredPrompt':
        """从文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
