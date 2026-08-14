"""
大语言模型客户端
使用SiliconFlow API调用大语言模型
与chapter2保持一致
"""
import json
import os
import requests
import re
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# 导入模型配置
try:
    from model_config import (
        SILICONFLOW_API_KEY as API_KEY,
        SILICONFLOW_API_URL as API_URL,
        CURRENT_LLM_MODEL as MODEL_NAME
    )
except ImportError:
    # 默认配置（如果配置文件不存在）
    API_KEY = os.getenv('SILICONFLOW_API_KEY', '')
    API_URL = 'https://api.siliconflow.cn/v1/chat/completions'
    MODEL_NAME = 'THUDM/GLM-4.1V-9B-Thinking'


class LLMClient:
    """
    LLM客户端
    封装对SiliconFlow API的调用
    """
    
    # 使用配置文件中的设置
    API_KEY = API_KEY
    API_URL = API_URL
    MODEL_NAME = MODEL_NAME
    
    def __init__(self, log_file: Optional[str] = None):
        """
        初始化LLM客户端
        
        Args:
            log_file: 日志文件路径
        """
        self.log_file = log_file or "llm_interaction.log"
        self.log_path = Path(self.log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_request_time = 0  # 上次请求时间
        self.min_request_interval = 0.5  # 最小请求间隔（秒）
    
    def generate(self, 
                 prompt: str, 
                 system_prompt: str = None,
                 temperature: float = 0.01,
                 max_tokens: int = 4000,
                 response_format: str = None,
                 desc: str = "Calling LLM") -> Dict[str, Any]:
        """
        调用LLM生成回答
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式 ("json" 或 None)
            desc: 描述信息
            
        Returns:
            包含回答的字典
        """
        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }
        
        # 构建消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # 如果需要JSON格式输出
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        
        # 实现限流：确保请求间隔
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        # 重试机制
        max_retries = 3
        base_wait = 2  # 基础等待时间（秒）
        
        for attempt in range(max_retries):
            try:
                print(f"[LLM] {desc}... (尝试 {attempt + 1}/{max_retries})")
                
                self.last_request_time = time.time()
                response = requests.post(
                    self.API_URL, 
                    headers=headers, 
                    json=payload, 
                    timeout=120
                )
                
                # 处理429错误
                if response.status_code == 429:
                    wait_time = base_wait * (2 ** attempt)  # 指数退避
                    print(f"[警告] 收到429错误，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # 记录交互日志
                self._log_interaction(prompt, content, system_prompt)
                
                # 如果需要JSON格式，尝试解析
                if response_format == "json":
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError as je:
                        print(f"[警告] JSON解析失败，尝试正则恢复...")
                        # 尝试提取JSON
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            try:
                                return json.loads(json_match.group())
                            except:
                                pass
                        # 返回原始内容包装
                        return {"answer": content, "parsed": False}
                
                return {"answer": content}
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = base_wait * (2 ** attempt)
                    print(f"[警告] 请求失败: {e}，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"[错误] API调用失败（已重试{max_retries}次）: {e}")
                    return {"error": str(e), "answer": ""}
            except Exception as e:
                print(f"[错误] 处理失败: {e}")
                return {"error": str(e), "answer": ""}
        
        # 所有重试都失败
        return {"error": "Max retries exceeded", "answer": ""}
    
    def generate_answer(self, 
                       question: str, 
                       context: str,
                       system_prompt: str = None,
                       output_json: bool = True) -> Dict[str, Any]:
        """
        生成问答回答（RAG专用）
        
        Args:
            question: 问题
            context: 上下文
            system_prompt: 系统提示词
            output_json: 是否输出JSON格式
            
        Returns:
            回答字典
        """
        # 构建RAG提示词
        prompt = f'''基于以下参考文档回答问题：

参考文档：
{context}

问题：{question}

请根据参考文档内容回答问题。如果文档中没有相关信息，请明确说明"根据提供的文档无法回答"。'''
        
        if output_json:
            prompt += '''

请以JSON格式输出，格式如下：
{
    "answer": "你的回答",
    "confidence": 0.85,
    "sources": ["文档1", "文档2"]
}'''
        
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt or "你是一个专业的问答助手，基于提供的文档准确回答问题。",
            response_format="json" if output_json else None,
            desc="生成RAG回答"
        )
    
    def _log_interaction(self, prompt: str, response: str, system_prompt: str = None):
        """
        记录交互日志
        
        Args:
            prompt: 提示词
            response: 响应
            system_prompt: 系统提示词
        """
        mode = "w" if not self.log_path.exists() else "a"
        with open(self.log_path, mode, encoding="utf-8") as f:
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n")
            if system_prompt:
                f.write(f"--- SYSTEM PROMPT ---\n{system_prompt}\n")
            f.write(f"--- USER PROMPT ---\n{prompt}\n")
            f.write(f"--- RESPONSE ---\n{response}\n")
            f.write(f"{'='*50}\n\n")


# 全局LLM客户端实例
llm_client = LLMClient()


if __name__ == "__main__":
    # 测试
    client = LLMClient()
    
    # 测试简单问答
    result = client.generate(
        prompt="什么是机器学习？",
        desc="测试简单问答"
    )
    print(f"回答: {result.get('answer', '')}")
    
    # 测试JSON输出
    result = client.generate(
        prompt="列举3个机器学习的应用场景",
        system_prompt="你是一个AI专家",
        response_format="json",
        desc="测试JSON输出"
    )
    print(f"JSON回答: {json.dumps(result, ensure_ascii=False, indent=2)}")
