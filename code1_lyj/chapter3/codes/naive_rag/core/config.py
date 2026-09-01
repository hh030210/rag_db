"""
配置文件
优化版本：平衡速度和内存使用
"""
import os
from pathlib import Path

# 项目根目录（指向naive_rag根目录，即core的父目录）
BASE_DIR = Path(__file__).parent.parent

# 模型路径 - 使用相对路径或自动检测
EMBEDDING_MODEL_PATH = r"i:\bylw_final\Code\models\embedding\bge-large-zh-v1.5"

# 向量维度
EMBEDDING_DIM = 1024  # bge-large-zh-v1.5的维度

# 向量数据库根目录
VECTOR_DB_ROOT = BASE_DIR / "vector_dbs"
VECTOR_DB_ROOT.mkdir(parents=True, exist_ok=True)

# 数据集路径
DATASET_PATH = r"i:\bylw_final\Code\chapter3\datasets\natural_questions\validation-00000-of-00007.json"

# 设备配置
DEVICE = "cuda"  # 使用GPU

# 切片配置
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# 检索配置
TOP_K = 5

# ==================== 性能优化配置 ====================
# RTX 2070 SUPER 8GB 显存优化配置

# 索引时每批处理的文档数（增大以提升速度）
# 原：500，建议：2000-5000（取决于文档长度）
INDEX_BATCH_SIZE = 1024

# 编码时的批次大小（关键参数）
# 原：16，建议：64-128（8GB显存可承受）
# 计算公式：batch_size * 512(tokens) * 1024(dim) * 4(bytes) / (1024**3) ≈ 显存占用(GB)
# 64 * 512 * 1024 * 4 / (1024**3) ≈ 0.125 GB 每批次
EMBEDDING_BATCH_SIZE = 32

# 最大安全批次（防止OOM）
MAX_SAFE_BATCH_SIZE = 128

# 并行工作线程数（用于数据加载）
NUM_WORKERS = 4

# 是否使用混合精度（可提速30-50%，节省显存）
USE_AMP = True  # Automatic Mixed Precision

# 显存清理频率（每N批次清理一次）
CLEAR_CACHE_EVERY = 10

# ChromaDB HNSW索引配置（影响检索速度）
HNSW_CONFIG = {
    "M": 16,  # 图连接数，越大越准但越慢
    "efConstruction": 100,  # 构建时的搜索深度
    "ef": 64  # 查询时的搜索深度
}
