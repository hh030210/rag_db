"""
系统配置文件
"""
import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

# 确保目录存在
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 服务器配置
HOST = "0.0.0.0"
PORT = 8001  # 使用8001避免与chapter4冲突
FRONTEND_PORT = 7910  # 使用7910避免端口冲突

# 文件上传配置
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {
    '.pdf', '.docx', '.doc', '.pptx', '.ppt', 
    '.xlsx', '.xls', '.txt', '.md'
}

# 处理阶段配置
PROCESSING_STAGES = [
    {"id": "raw", "name": "原始文本", "description": "文件转换后的原始文本"},
    {"id": "rule_denoised", "name": "规则去噪", "description": "基于规则的初步去噪"},
    {"id": "llm_denoised", "name": "LLM去噪", "description": "基于LLM的语义去噪"},
    {"id": "organized", "name": "内容重组", "description": "内容重组和归纳后的最终文本"}
]

# API配置（从chapter2复制）
# SiliconFlow API 配置
API_KEY = os.getenv('SILICONFLOW_API_KEY', '')
API_URL = 'https://api.siliconflow.cn/v1/chat/completions'
MODEL_NAME = 'PaddlePaddle/PaddleOCR-VL-1.5'
TEXT_MODEL_NAME = 'Qwen/Qwen3-8B'  # 默认使用 Qwen3-8B 进行文本处理

# 临时图像目录
TEMP_IMAGE_DIR = 'temp_images'

# 并行处理配置
API_MAX_WORKERS = 10  # 可以根据需要修改并行进程数

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"

# ==================== RAG系统配置（第三章） ====================

# Milvus向量数据库配置
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_USER = os.getenv("MILVUS_USER", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")

# Embedding模型配置
# 如果使用本地模型，请设置为模型路径，例如：./models/bge-large-zh-v1.5
# 如果使用HuggingFace模型，请设置为模型名称，例如：BAAI/bge-large-zh-v1.5
# 默认使用本地模型路径
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", r"I:\毕业论文最新版\Code\models\embedding\bge-large-zh-v1.5")
EMBEDDING_DIMENSION = 1024  # BGE-large-zh-v1.5 的向量维度
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # cpu, cuda

# RAG检索配置
RAG_DEFAULT_TOP_K = 5  # 默认检索返回的文档数
RAG_DEFAULT_CHUNK_SIZE = 500  # 默认文本分块大小
RAG_DEFAULT_CHUNK_OVERLAP = 50  # 默认文本分块重叠大小
RAG_MAX_CHUNK_SIZE = 2000  # 最大文本分块大小
RAG_MIN_CHUNK_SIZE = 100  # 最小文本分块大小

# RAG对话配置
RAG_MAX_CHAT_HISTORY = 5  # 最大保留的对话历史轮数
RAG_DEFAULT_TEMPERATURE = 0.7  # LLM生成温度
RAG_MAX_TOKENS = 2000  # LLM最大生成token数
RAG_SYSTEM_PROMPT = """你是一个专业的文档问答助手。基于提供的上下文信息回答用户的问题。
请根据上下文信息提供准确、简洁的回答。如果上下文不包含相关信息，请明确说明。
回答时请引用相关信息来源。"""

# 向量库配置
VECTOR_COLLECTION_PREFIX = "rag_collection_"  # Milvus集合名称前缀
VECTOR_INDEX_TYPE = "IVF_FLAT"  # 向量索引类型
VECTOR_METRIC_TYPE = "COSINE"  # 向量相似度度量方式
