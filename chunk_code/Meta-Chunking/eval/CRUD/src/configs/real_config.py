# ============================================================
# 运行环境设置（建议不要删除，防止联网超时）
# ============================================================
import os
os.environ['HF_HUB_OFFLINE'] = '1'    # 强制使用本地缓存的 HuggingFace 模型
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # transformers 也使用本地缓存

# ============================================================
# 运行时配置 - 请根据你的实际环境修改以下配置
# ============================================================

# ---------- LLM 配置 ----------

# 方式 A: 使用本地模型（从 HuggingFace 缓存加载，需要完整下载）
# 注意：Qwen2.5-7B-Instruct 目前缓存不完整，请使用下方 API 方式
# Qwen_7B_local_path = 'Qwen/Qwen2.5-7B-Instruct'

# 方式 B: 使用 OpenAI-compatible API（推荐！免费额度充足）
#
#   【推荐】硅基流动 SiliconFlow（https://siliconflow.cn）
#   注册后获取 API Key，免费送 28元额度，支持 Qwen2.5-7B-Instruct 等模型
#   API Base: https://api.siliconflow.cn/v1
#   模型名: Qwen/Qwen2.5-7B-Instruct
#
#   【备选】阿里云百炼 DashScope（https://dashscope.aliyuncs.com）
#   新用户有免费额度
#   API Base: https://dashscope.aliyuncs.com/compatible-mode/v1
#   模型名: qwen-plus 或 qwen-turbo
#
Qwen_OpenAI_API_Key = os.environ.get('QWEN_OPENAI_API_KEY', '')
Qwen_OpenAI_API_Base = os.environ.get('QWEN_OPENAI_API_BASE', 'https://api.siliconflow.cn/v1')
Qwen_OpenAI_Model_Name = os.environ.get('QWEN_OPENAI_MODEL_NAME', 'Qwen/Qwen2.5-7B-Instruct')

# ---------- 向量检索配置 ----------
# BGE 模型（用于 Embedding，向量维度 768）- 从本地缓存加载
BGE_model_name = os.environ.get('BGE_MODEL_PATH', 'BAAI/bge-base-zh-v1.5')

# ============================================================
# 以下为默认参数，无需修改
# ============================================================
GPT_api_key = 'sk-no-gpt-key'            # 占位符 - 由于 QuestEval 强制需要 GPT 模块可导入
GPT_api_base = 'https://api.openai.com/v1'  # 占位符，实际不调用
GPT_transit_url = ''
GPT_transit_token = ''
GPT_transit_user = ''
Qwen_7B_local_path = ''
Baichuan2_13b_local_path = ''
ChatGLM3_local_path = ''
Qwen_14B_local_path = ''
