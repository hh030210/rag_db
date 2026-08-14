"""
向量存储模块 - 使用Faiss，支持多数据集
修复版本：解决中文路径问题 - 使用临时文件
"""
from typing import List, Dict, Optional
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pickle
import os
import tempfile
import shutil

# Faiss导入
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("警告: faiss未安装，请运行: pip install faiss-cpu")

import config
from embedder import embedder


class VectorStore:
    """向量数据库管理器 - 使用Faiss，支持多数据集"""
    
    def __init__(self, dataset_name: str):
        """
        初始化向量存储
        
        Args:
            dataset_name: 数据集名称，每个数据集有独立的向量库
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss未安装，请运行: pip install faiss-cpu")
        
        self.dataset_name = dataset_name
        
        # 为每个数据集创建独立的存储目录
        self.persist_dir = config.VECTOR_DB_ROOT / dataset_name / "faiss"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件路径 - 使用os.path处理中文路径
        self.index_file = str(self.persist_dir / "index.faiss")
        self.metadata_file = str(self.persist_dir / "metadata.pkl")
        
        # 向量维度
        self.dim = config.EMBEDDING_DIM
        
        # 加载或创建索引
        self.index, self.metadata = self._load_or_create_index()
        
        print(f"✓ 向量库初始化完成: {dataset_name}")
        print(f"  - 存储路径: {self.persist_dir}")
        print(f"  - 现有文档数: {len(self.metadata)}")
    
    def _load_or_create_index(self):
        """加载或创建Faiss索引"""
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            # 加载现有索引
            try:
                # 先复制到临时目录再加载（避免中文路径问题）
                with tempfile.NamedTemporaryFile(delete=False, suffix='.faiss') as tmp:
                    tmp_path = tmp.name
                shutil.copy2(self.index_file, tmp_path)
                index = faiss.read_index(tmp_path)
                os.unlink(tmp_path)
                
                with open(self.metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                print(f"  - 已加载现有索引: {len(metadata)} 个文档")
                return index, metadata
            except Exception as e:
                print(f"  - 加载索引失败: {e}，创建新索引")
        
        # 创建新索引（使用内积作为相似度度量，等同于余弦相似度当向量归一化时）
        index = faiss.IndexFlatIP(self.dim)  # IP = Inner Product
        
        # 如果有GPU，使用GPU加速
        if hasattr(faiss, 'StandardGpuResources'):
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print("  - 使用GPU加速")
            except:
                pass
        
        metadata = {}
        return index, metadata
    
    def _save_index(self):
        """保存索引到磁盘"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        
        # 如果是GPU索引，先转回CPU
        index_to_save = self.index
        if hasattr(self.index, 'is_gpu') and self.index.is_gpu:
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        
        # 使用临时文件保存，然后复制到目标位置
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.faiss') as tmp:
                tmp_path = tmp.name
            
            # 保存到临时文件
            faiss.write_index(index_to_save, tmp_path)
            
            # 复制到目标位置
            shutil.copy2(tmp_path, self.index_file)
            os.unlink(tmp_path)
            
            # 保存元数据
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
                
        except Exception as e:
            print(f"  - 警告: 保存索引失败: {e}")
            print(f"  - 索引将只在内存中可用")
    
    def add_documents(self, chunks: List[Dict], batch_size: int = None):
        """
        添加文档块到向量库
        
        Args:
            chunks: 文档块列表
            batch_size: 批处理大小
        """
        if not chunks:
            print("没有文档块需要添加")
            return
        
        # 使用配置的批次大小
        if batch_size is None:
            batch_size = config.EMBEDDING_BATCH_SIZE
        
        total_batches = (len(chunks) - 1) // batch_size + 1
        print(f"开始添加 {len(chunks)} 个文档块到向量库...")
        print(f"  使用设备: {embedder.device}")
        print(f"  编码批次大小: {batch_size}")
        print()
        
        start_id = len(self.metadata)
        
        # 使用tqdm创建进度条
        with tqdm(total=total_batches, desc="向量化进度", 
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            # 分批处理
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # 准备数据
                texts = [chunk['text'] for chunk in batch]
                
                # 生成向量
                embeddings_list = embedder.encode(texts, batch_size=batch_size, show_progress=False)
                
                # 转换为numpy数组
                embeddings = np.array(embeddings_list, dtype=np.float32)
                
                # 归一化向量（使内积等同于余弦相似度）
                faiss.normalize_L2(embeddings)
                
                # 添加到索引
                self.index.add(embeddings)
                
                # 保存元数据
                for j, chunk in enumerate(batch):
                    idx = start_id + i + j
                    self.metadata[idx] = {
                        'chunk_id': chunk['chunk_id'],
                        'text': chunk['text'],
                        'metadata': chunk['metadata']
                    }
                
                # 更新进度条
                pbar.update(1)
                
                # 清理GPU缓存
                if embedder.device == "cuda":
                    import torch
                    torch.cuda.empty_cache()
        
        # 保存索引
        self._save_index()
        
        print()
        print(f"✓ 成功添加 {len(chunks)} 个文档块")
        print(f"  - 当前总文档数: {len(self.metadata)}")
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        搜索最相似的文档块
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            相似文档块列表
        """
        top_k = top_k or config.TOP_K
        
        # 编码查询
        query_embedding_list = embedder.encode_query(query)
        query_embedding = np.array(query_embedding_list, dtype=np.float32).reshape(1, -1)
        
        # 归一化查询向量
        faiss.normalize_L2(query_embedding)
        
        # 搜索
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 格式化结果
        formatted_results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # Faiss返回-1表示没有更多结果
                continue
            
            meta = self.metadata.get(int(idx), {})
            formatted_results.append({
                'chunk_id': meta.get('chunk_id', f'chunk_{idx}'),
                'text': meta.get('text', ''),
                'metadata': meta.get('metadata', {}),
                'distance': float(1 - dist),  # 转换为距离
                'score': float(dist)  # 内积就是相似度
            })
        
        return formatted_results
    
    def delete_collection(self):
        """删除当前集合"""
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)
        print(f"✓ 已删除向量库: {self.dataset_name}")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'dataset': self.dataset_name,
            'document_count': len(self.metadata),
            'persist_dir': str(self.persist_dir)
        }


def get_vector_store(dataset_name: str) -> VectorStore:
    """
    获取向量存储实例
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        VectorStore实例
    """
    return VectorStore(dataset_name)
