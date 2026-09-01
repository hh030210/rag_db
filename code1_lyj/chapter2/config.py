# 配置文件
import os

# SiliconFlow API 配置
API_KEY = os.getenv('SILICONFLOW_API_KEY', '')
API_URL = 'https://api.siliconflow.cn/v1/chat/completions'
MODEL_NAME = 'PaddlePaddle/PaddleOCR-VL-1.5'
TEXT_MODEL_NAME = 'Qwen/Qwen3-8B' # 默认使用 GLM-4 进行文本处理

# 临时图像目录

TEMP_IMAGE_DIR = 'temp_images'

# 并行处理配置

API_MAX_WORKERS = 10  # 可以根据需要修改并行进程数
