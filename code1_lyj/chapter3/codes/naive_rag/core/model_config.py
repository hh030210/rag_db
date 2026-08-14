"""
模型配置文件
统一管理所有模型名称和API配置
"""

import os

# ==================== LLM模型配置 ====================
# 当前使用的模型名称（修改此处即可改变所有地方的模型）
# CURRENT_LLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# CURRENT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# CURRENT_LLM_MODEL = "THUDM/GLM-Z1-9B-0414"
# CURRENT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# CURRENT_LLM_MODEL = "THUDM/GLM-4-9B-0414"
# CURRENT_LLM_MODEL = "internlm/internlm2_5-7b-chat"
# CURRENT_LLM_MODEL = "Qwen/Qwen2-7B-Instruct"
CURRENT_LLM_MODEL = "Qwen/Qwen3-8B"

# 模型显示名称（用于日志和报告）
# CURRENT_LLM_DISPLAY_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# CURRENT_LLM_DISPLAY_NAME = "Qwen/Qwen2.5-7B-Instruct"
# CURRENT_LLM_DISPLAY_NAME = "THUDM/GLM-Z1-9B-0414"
# CURRENT_LLM_DISPLAY_NAME = "Qwen/Qwen2.5-7B-Instruct"
# CURRENT_LLM_DISPLAY_NAME = "THUDM/GLM-4-9B-0414"
# CURRENT_LLM_DISPLAY_NAME = "internlm/internlm2_5-7b-chat"
# CURRENT_LLM_DISPLAY_NAME = "Qwen/Qwen2-7B-Instruct"
CURRENT_LLM_DISPLAY_NAME = "Qwen/Qwen3-8B"

# SiliconFlow API配置
SILICONFLOW_API_KEY = os.getenv('SILICONFLOW_API_KEY', '')
SILICONFLOW_API_URL = 'https://api.siliconflow.cn/v1/chat/completions'

# ==================== Embedding模型配置 ====================
EMBEDDING_MODEL_NAME = "bge-large-zh-v1.5"
EMBEDDING_MODEL_PATH = r"i:\bylw_final\Code\models\embedding\bge-large-zh-v1.5"
EMBEDDING_DIM = 1024

# ==================== 其他配置 ====================
# 默认温度参数
DEFAULT_TEMPERATURE = 0.01
# 默认最大token数
DEFAULT_MAX_TOKENS = 4000
