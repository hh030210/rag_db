"""
ChromaDB向量存储封装
支持多数据集和动态选择
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from pathlib import Path


class ChromaVectorStore:
    """ChromaDB向量存储"""
    
    def __init__(self, dataset_name: str, persist_dir: str = None):
        """
        初始化向量存储
        
        Args:
            dataset_name: 数据集名称
            persist_dir: 持久化目录，默认使用 naive_rag 的 vector_dbs
        """
        self.dataset_name = dataset_name
        
        # 默认使用 naive_rag 的 vector_dbs 目录
        if persist_dir is None:
            # 获取当前文件所在目录，然后找到 naive_rag 的 vector_dbs
            current_file = Path(__file__).resolve()
            # bylw_rag/vector_stores/chroma_store.py -> chapter3/codes/ -> naive_rag/vector_dbs
            naive_rag_dir = current_file.parent.parent.parent / "naive_rag"
            persist_dir = naive_rag_dir / "vector_dbs"
        
        self.persist_path = Path(persist_dir) / dataset_name
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化客户端
        self.client = chromadb.PersistentClient(
            path=str(self.persist_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=f"{dataset_name}_collection"
        )
        
        print(f"✓ 向量存储初始化: {dataset_name}")
        print(f"  - 路径: {self.persist_path}")
        print(f"  - 文档数: {self.collection.count()}")
    
    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """
        添加文档
        
        Args:
            documents: 文档列表
            embeddings: 嵌入向量列表
        """
        if not documents:
            return
        
        ids = [doc['id'] for doc in documents]
        texts = [doc['text'] for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        搜索相似文档
        
        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            
        Returns:
            搜索结果列表
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        documents = []
        for i in range(len(results['ids'][0])):
            documents.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': results['distances'][0][i] if 'distances' in results else 0.0
            })
        
        return documents
    
    def delete_collection(self):
        """删除集合"""
        try:
            self.client.delete_collection(f"{self.dataset_name}_collection")
            print(f"✓ 集合已删除: {self.dataset_name}")
        except:
            pass


class VectorStoreManager:
    """向量存储管理器"""
    
    def __init__(self, base_dir: str = "vector_dbs"):
        """
        初始化管理器
        
        Args:
            base_dir: 向量库根目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._stores: Dict[str, ChromaVectorStore] = {}
    
    def get_store(self, dataset_name: str) -> ChromaVectorStore:
        """
        获取向量存储（缓存）
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            向量存储实例
        """
        if dataset_name not in self._stores:
            self._stores[dataset_name] = ChromaVectorStore(
                dataset_name=dataset_name,
                persist_dir=str(self.base_dir)
            )
        return self._stores[dataset_name]
    
    def list_datasets(self) -> List[str]:
        """列出所有数据集"""
        if not self.base_dir.exists():
            return []
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
    
    def delete_dataset(self, dataset_name: str):
        """删除数据集"""
        if dataset_name in self._stores:
            self._stores[dataset_name].delete_collection()
            del self._stores[dataset_name]


# 全局管理器
store_manager = VectorStoreManager()
