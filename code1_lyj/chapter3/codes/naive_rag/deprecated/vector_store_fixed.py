"""
向量存储模块 - 使用ChromaDB，支持多数据集
修复版本：添加 bruteforce 搜索回退
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import config
from embedder import embedder


class VectorStore:
    """向量数据库管理器 - 支持多数据集"""
    
    def __init__(self, dataset_name: str):
        """
        初始化向量存储
        
        Args:
            dataset_name: 数据集名称，每个数据集有独立的向量库
        """
        self.dataset_name = dataset_name
        
        # 为每个数据集创建独立的持久化目录
        self.persist_dir = config.VECTOR_DB_ROOT / dataset_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=f"{dataset_name}_collection",
            metadata={"dataset": dataset_name}
        )
        
        # 检查HNSW索引是否可用
        self.hnsw_available = self._check_hnsw_available()
        
        print(f"✓ 向量库初始化完成: {dataset_name}")
        print(f"  - 存储路径: {self.persist_dir}")
        print(f"  - 现有文档数: {self.collection.count()}")
        print(f"  - HNSW索引: {'可用' if self.hnsw_available else '不可用(将使用暴力搜索)'}")
    
    def _check_hnsw_available(self) -> bool:
        """检查HNSW索引是否可用"""
        try:
            # 尝试执行一个简单的查询
            query_embedding = embedder.encode_query("test")
            self.collection.query(
                query_embeddings=[query_embedding],
                n_results=1
            )
            return True
        except Exception as e:
            if "Cannot open header file" in str(e):
                return False
            raise
    
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
        
        # 使用tqdm创建进度条
        with tqdm(total=total_batches, desc="向量化进度", 
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            # 分批处理
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # 准备数据
                ids = [chunk['chunk_id'] for chunk in batch]
                texts = [chunk['text'] for chunk in batch]
                metadatas = [chunk['metadata'] for chunk in batch]
                
                # 生成向量（使用更小的batch_size避免OOM）
                embeddings = embedder.encode(texts, batch_size=batch_size, show_progress=False)
                
                # 添加到ChromaDB
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
                
                # 更新进度条
                pbar.update(1)
                
                # 清理GPU缓存
                if embedder.device == "cuda":
                    import torch
                    torch.cuda.empty_cache()
        
        print()
        print(f"✓ 成功添加 {len(chunks)} 个文档块")
        print(f"  - 当前总文档数: {self.collection.count()}")
        
        # 重新检查HNSW可用性
        self.hnsw_available = self._check_hnsw_available()
    
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
        query_embedding = embedder.encode_query(query)
        
        # 如果HNSW可用，使用正常查询
        if self.hnsw_available:
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                return self._format_results(results)
            except Exception as e:
                if "Cannot open header file" not in str(e):
                    raise
                # HNSW突然不可用，切换到暴力搜索
                self.hnsw_available = False
                print("[警告] HNSW索引不可用，切换到暴力搜索模式")
        
        # 使用暴力搜索
        return self._brute_force_search(query_embedding, top_k)
    
    def _brute_force_search(self, query_embedding: List[float], top_k: int) -> List[Dict]:
        """
        暴力搜索（当HNSW不可用时使用）
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            
        Returns:
            相似文档块列表
        """
        print(f"[暴力搜索] 检索中...")
        
        # 获取所有文档（分批获取避免内存问题）
        batch_size = 1000
        all_docs = []
        offset = 0
        
        while True:
            batch = self.collection.get(
                limit=batch_size,
                offset=offset,
                include=["embeddings", "documents", "metadatas"]
            )
            
            if not batch['ids']:
                break
            
            for i in range(len(batch['ids'])):
                all_docs.append({
                    'chunk_id': batch['ids'][i],
                    'text': batch['documents'][i],
                    'metadata': batch['metadatas'][i],
                    'embedding': batch['embeddings'][i]
                })
            
            offset += len(batch['ids'])
            if len(batch['ids']) < batch_size:
                break
        
        # 计算余弦相似度
        query_vec = np.array(query_embedding)
        similarities = []
        
        for doc in all_docs:
            doc_vec = np.array(doc['embedding'])
            # 余弦相似度 = dot(a, b) / (norm(a) * norm(b))
            sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarities.append((doc, float(sim)))
        
        # 排序并返回top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc, sim in similarities[:top_k]:
            results.append({
                'chunk_id': doc['chunk_id'],
                'text': doc['text'],
                'metadata': doc['metadata'],
                'distance': 1 - sim,  # 转换为距离
                'score': sim
            })
        
        return results
    
    def _format_results(self, results: Dict) -> List[Dict]:
        """格式化查询结果"""
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'chunk_id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'score': 1 - results['distances'][0][i]
            })
        return formatted_results
    
    def delete_collection(self):
        """删除当前集合"""
        self.client.delete_collection(name=f"{self.dataset_name}_collection")
        print(f"✓ 已删除向量库: {self.dataset_name}")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'dataset': self.dataset_name,
            'document_count': self.collection.count(),
            'persist_dir': str(self.persist_dir),
            'hnsw_available': self.hnsw_available
        }


# 全局向量存储实例
_vector_store_instance = None
_vector_store_dataset = None


def get_vector_store(dataset_name: str) -> VectorStore:
    """
    获取向量存储实例
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        VectorStore实例
    """
    global _vector_store_instance, _vector_store_dataset
    
    # 如果数据集变化或实例不存在，创建新实例
    if _vector_store_instance is None or _vector_store_dataset != dataset_name:
        _vector_store_instance = VectorStore(dataset_name)
        _vector_store_dataset = dataset_name
    
    return _vector_store_instance
