"""
Embedding编码器 - 使用BGE-large-zh-v1.5模型
优化版本：支持混合精度和更大批次
"""
import torch
from sentence_transformers import SentenceTransformer
from typing import List
import config


class BGEEmbedder:
    """BGE嵌入模型编码器"""
    
    _instance = None
    
    def __new__(cls):
        """单例模式，确保只有一个模型实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        print(f"正在加载Embedding模型: {config.EMBEDDING_MODEL_PATH}")
        
        # 检查GPU可用性
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU可用: {gpu_name} ({gpu_memory:.1f} GB)")
            device = "cuda"
        else:
            print("✗ GPU不可用，使用CPU")
            device = "cpu"
        
        # 加载模型
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_PATH, device=device)
        self.device = device
        self._initialized = True
        
        # 启用混合精度（如果配置允许且是GPU）
        self.use_amp = config.USE_AMP and device == "cuda"
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print(f"✓ 启用混合精度 (AMP)")
        
        print(f"✓ 模型加载完成，使用设备: {device}")
        print(f"✓ 向量维度: {self.model.get_sentence_embedding_dimension()}")
        print(f"✓ 编码批次大小: {config.EMBEDDING_BATCH_SIZE}")
    
    def encode(self, texts: List[str], batch_size: int = None, show_progress: bool = True) -> List[List[float]]:
        """
        将文本编码为向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小（默认使用config中的配置）
            show_progress: 是否显示进度条
            
        Returns:
            向量列表
        """
        if not texts:
            return []
        
        # 使用配置的批次大小
        if batch_size is None:
            batch_size = config.EMBEDDING_BATCH_SIZE
        
        # 确保不超过最大安全批次
        batch_size = min(batch_size, config.MAX_SAFE_BATCH_SIZE)
        
        # 如果文本数量很大，使用分块处理
        if len(texts) > batch_size * 4:  # 如果数据量很大
            return self._encode_large_batch(texts, batch_size, show_progress)
        
        # 正常编码
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[警告] OOM错误，减小批次大小重试...")
                torch.cuda.empty_cache()
                # 减半批次重试
                return self.encode(texts, batch_size=batch_size//2, show_progress=show_progress)
            raise
    
    def _encode_large_batch(self, texts: List[str], batch_size: int, show_progress: bool) -> List[List[float]]:
        """
        大批量编码（分块处理，定期清理显存）
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度
            
        Returns:
            向量列表
        """
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            # 编码当前批次
            batch_embeddings = self.model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,  # 不显示内部进度
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            all_embeddings.extend(batch_embeddings.tolist())
            
            # 定期清理显存
            if batch_num % config.CLEAR_CACHE_EVERY == 0:
                torch.cuda.empty_cache()
            
            # 显示进度
            if show_progress and batch_num % 10 == 0:
                print(f"  进度: {batch_num}/{total_batches} 批次 ({i+len(batch)}/{len(texts)} 文档)")
        
        return all_embeddings
    
    def encode_queries(self, queries: List[str], batch_size: int = None) -> List[List[float]]:
        """
        批量编码查询（添加查询前缀）
        
        Args:
            queries: 查询列表
            batch_size: 批处理大小
            
        Returns:
            向量列表
        """
        # BGE模型建议为查询添加前缀
        query_texts = [f"为这个句子生成表示以用于检索相关文章：{q}" for q in queries]
        return self.encode(query_texts, batch_size=batch_size, show_progress=False)
    
    def encode_query(self, query: str) -> List[float]:
        """
        编码单个查询
        """
        return self.encode_queries([query])[0]
    
    def get_gpu_memory_info(self) -> dict:
        """
        获取GPU内存信息
        
        Returns:
            内存信息字典
        """
        if self.device != "cuda":
            return {"device": "cpu"}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return {
            "device": torch.cuda.get_device_name(0),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - allocated, 2)
        }


# 全局单例
embedder = BGEEmbedder()
