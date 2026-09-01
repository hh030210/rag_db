"""
RAG系统 API 路由 - 第三章
集成 ChromaDB + BGE-ZH-1.5 + Qwen3-8B
"""
import os
import sys
import uuid
import json
import asyncio
import httpx
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from functools import lru_cache

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from loguru import logger
import asyncio

# 导入tqdm进度条
from tqdm import tqdm

import config

# 禁用ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# 尝试导入RAG相关库
try:
    import chromadb
    from chromadb.config import Settings
    # 完全禁用telemetry
    import chromadb.telemetry.product.posthog as posthog
    posthog.Posthog.capture = lambda *args, **kwargs: None
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb 未安装，将使用内存存储作为备用")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers 未安装")

# 创建路由
router = APIRouter(prefix="/api/rag", tags=["RAG系统"])

# ==================== 配置 ====================

# ChromaDB配置
CHROMA_DB_PATH = os.path.join(config.BASE_DIR, "chroma_db")

# Embedding模型配置 - 使用config中的配置（本地模型路径）
EMBEDDING_MODEL_PATH = config.EMBEDDING_MODEL_PATH
EMBEDDING_DIMENSION = config.EMBEDDING_DIMENSION  # BGE-large-zh-v1.5 的维度
EMBEDDING_DEVICE = config.EMBEDDING_DEVICE

# LLM配置（使用第二章的Qwen3-8B）
LLM_API_KEY = config.API_KEY
LLM_API_URL = config.API_URL
LLM_MODEL_NAME = config.TEXT_MODEL_NAME  # Qwen/Qwen3-8B

# ==================== 全局变量 ====================

_embedding_model = None
_chroma_client = None
_chroma_db_connected = False

# 内存存储（备用方案）
_memory_collections: Dict[str, Any] = {}
_memory_documents: Dict[str, Any] = {}
_memory_vectors: Dict[str, List[float]] = {}

# 任务状态存储
_task_status: Dict[str, Dict[str, Any]] = {}

# 持久化存储路径
RAG_DATA_DIR = os.path.join(config.BASE_DIR, "rag_data")
COLLECTIONS_FILE = os.path.join(RAG_DATA_DIR, "collections.json")
DOCUMENTS_FILE = os.path.join(RAG_DATA_DIR, "documents.json")

# 确保数据目录存在
os.makedirs(RAG_DATA_DIR, exist_ok=True)


def _save_collections():
    """保存向量库元数据到文件"""
    try:
        # 只保存可序列化的数据
        collections_data = {}
        for col_id, col in _memory_collections.items():
            collections_data[col_id] = {
                "id": col.get("id"),
                "name": col.get("name"),
                "description": col.get("description"),
                "dimension": col.get("dimension"),
                "created_at": col.get("created_at"),
                "updated_at": col.get("updated_at"),
                "documents": col.get("documents", []),
                "chroma_collection": col.get("chroma_collection")
            }
        with open(COLLECTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(collections_data, f, ensure_ascii=False, indent=2)
        logger.debug(f"保存 {len(collections_data)} 个向量库元数据")
    except Exception as e:
        logger.error(f"保存向量库元数据失败: {str(e)}")


def _load_collections():
    """从文件加载向量库元数据"""
    global _memory_collections
    if not os.path.exists(COLLECTIONS_FILE):
        logger.info("向量库元数据文件不存在，初始化为空")
        _memory_collections = {}
        return
    
    try:
        with open(COLLECTIONS_FILE, 'r', encoding='utf-8') as f:
            collections_data = json.load(f)
        
        _memory_collections = collections_data
        logger.info(f"加载 {len(collections_data)} 个向量库元数据")
    except Exception as e:
        logger.error(f"加载向量库元数据失败: {str(e)}")
        _memory_collections = {}


def _save_documents():
    """保存文档元数据到文件"""
    try:
        # 只保存可序列化的数据（不包含embeddings）
        documents_data = {}
        for doc_id, doc in _memory_documents.items():
            documents_data[doc_id] = {
                "id": doc.get("id"),
                "file_id": doc.get("file_id"),
                "file_name": doc.get("file_name"),
                "collection_id": doc.get("collection_id"),
                "chunk_count": doc.get("chunk_count"),
                "added_at": doc.get("added_at"),
                "chunks": doc.get("chunks", [])  # 保存文本块内容
            }
        with open(DOCUMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(documents_data, f, ensure_ascii=False, indent=2)
        logger.debug(f"保存 {len(documents_data)} 个文档元数据")
    except Exception as e:
        logger.error(f"保存文档元数据失败: {str(e)}")


def _load_documents():
    """从文件加载文档元数据"""
    global _memory_documents
    if not os.path.exists(DOCUMENTS_FILE):
        logger.info("文档元数据文件不存在，初始化为空")
        _memory_documents = {}
        return
    
    try:
        with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
            documents_data = json.load(f)
        
        _memory_documents = documents_data
        logger.info(f"加载 {len(documents_data)} 个文档元数据")
    except Exception as e:
        logger.error(f"加载文档元数据失败: {str(e)}")
        _memory_documents = {}


def _sync_from_chromadb():
    """从ChromaDB同步集合信息到内存"""
    client = get_chroma_client()
    if not client:
        return
    
    try:
        collections = client.list_collections()
        logger.info(f"从ChromaDB发现 {len(collections)} 个集合")
        
        for collection in collections:
            col_name = collection.name
            # 检查是否已在内存中
            existing = False
            for col_id, col_data in _memory_collections.items():
                if col_data.get("chroma_collection") == col_name:
                    existing = True
                    break
            
            if not existing:
                # 创建新的向量库记录
                collection_id = f"col_{col_name}_{datetime.now().timestamp()}"
                _memory_collections[collection_id] = {
                    "id": collection_id,
                    "name": col_name,
                    "description": f"从ChromaDB恢复的集合: {col_name}",
                    "dimension": EMBEDDING_DIMENSION,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "documents": [],
                    "chroma_collection": col_name
                }
                logger.info(f"从ChromaDB恢复集合: {col_name}")
        
        if collections:
            _save_collections()
    except Exception as e:
        logger.error(f"从ChromaDB同步失败: {str(e)}")


# 启动时加载数据（在函数定义之后调用）
# _load_collections() 和 _load_documents() 在模块末尾调用

# ==================== 数据模型 ====================

class CollectionCreate(BaseModel):
    """创建向量库请求"""
    name: str
    description: str = ""

class CollectionInfo(BaseModel):
    """向量库信息"""
    id: str
    name: str
    description: str
    document_count: int
    dimension: int
    created_at: str
    updated_at: str

class CollectionListResponse(BaseModel):
    """向量库列表响应"""
    collections: List[CollectionInfo]
    total: int

class DocumentAddRequest(BaseModel):
    """添加文档请求"""
    file_ids: List[str]
    chunk_size: int = 500
    chunk_overlap: int = 50

class DocumentInfo(BaseModel):
    """文档信息"""
    id: str
    file_name: str
    file_id: str
    chunk_count: int
    added_at: str

class DocumentListResponse(BaseModel):
    """文档列表响应"""
    documents: List[DocumentInfo]
    total: int

class SearchRequest(BaseModel):
    """检索请求"""
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    """检索结果"""
    content: str
    score: float
    source: str
    chunk_id: str

class SearchResponse(BaseModel):
    """检索响应"""
    query: str
    results: List[SearchResult]
    total: int

class ChatRequest(BaseModel):
    """对话请求"""
    query: str
    context: str = ""
    chat_history: List[List[str]] = []

class ChatResponse(BaseModel):
    """对话响应"""
    answer: str
    sources: List[str]


# ==================== Embedding模型管理 ====================

def get_embedding_model():
    """获取或初始化Embedding模型"""
    global _embedding_model
    
    if _embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            logger.info(f"正在加载Embedding模型: {EMBEDDING_MODEL_PATH}")
            logger.info(f"使用设备: {EMBEDDING_DEVICE}")
            
            # 检查是否是本地路径
            if os.path.exists(EMBEDDING_MODEL_PATH):
                logger.info("检测到本地模型路径，从本地加载...")
            else:
                logger.info("未找到本地模型，尝试从HuggingFace下载...")
            
            _embedding_model = SentenceTransformer(
                EMBEDDING_MODEL_PATH,
                device=EMBEDDING_DEVICE
            )
            logger.info("Embedding模型加载成功")
        except Exception as e:
            logger.error(f"Embedding模型加载失败: {str(e)}")
            _embedding_model = None
    
    return _embedding_model


def encode_texts(texts: List[str]) -> List[List[float]]:
    """将文本编码为向量"""
    model = get_embedding_model()
    
    if model is None:
        # 备用方案：生成随机向量
        import random
        logger.warning("Embedding模型不可用，使用随机向量")
        return [[random.random() for _ in range(EMBEDDING_DIMENSION)] for _ in texts]
    
    try:
        # BGE模型需要在文本前添加指令
        instruction = "为这个句子生成表示以用于检索相关文章："
        texts_with_instruction = [f"{instruction}{text}" for text in texts]
        embeddings = model.encode(texts_with_instruction, normalize_embeddings=True)
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"文本编码失败: {str(e)}")
        import random
        return [[random.random() for _ in range(EMBEDDING_DIMENSION)] for _ in texts]


def encode_query(query: str) -> List[float]:
    """将查询编码为向量"""
    model = get_embedding_model()
    
    if model is None:
        import random
        return [random.random() for _ in range(EMBEDDING_DIMENSION)]
    
    try:
        # 查询也需要添加指令
        instruction = "为这个句子生成表示以用于检索相关文章："
        query_with_instruction = f"{instruction}{query}"
        embedding = model.encode(query_with_instruction, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"查询编码失败: {str(e)}")
        import random
        return [random.random() for _ in range(EMBEDDING_DIMENSION)]


# ==================== WebSocket进度管理 ====================

class ProgressManager:
    """管理向量化进度"""
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.progress_data: Dict[str, dict] = {}
    
    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[client_id] = websocket
        logger.info(f"WebSocket客户端连接: {client_id}")
    
    def disconnect(self, client_id: str):
        if client_id in self.connections:
            del self.connections[client_id]
            logger.info(f"WebSocket客户端断开: {client_id}")
    
    async def send_progress(self, client_id: str, data: dict):
        """发送进度更新"""
        if client_id in self.connections:
            try:
                await self.connections[client_id].send_json(data)
            except Exception as e:
                logger.error(f"发送进度失败: {str(e)}")
    
    async def broadcast_progress(self, task_id: str, data: dict):
        """广播进度到所有连接的客户端"""
        self.progress_data[task_id] = data
        for client_id, ws in list(self.connections.items()):
            try:
                await ws.send_json({
                    "task_id": task_id,
                    **data
                })
            except Exception as e:
                logger.error(f"广播进度到 {client_id} 失败: {str(e)}")

# 全局进度管理器
progress_manager = ProgressManager()

# ==================== ChromaDB连接管理 ====================

def get_chroma_client():
    """获取或初始化ChromaDB客户端"""
    global _chroma_client, _chroma_db_connected
    
    if not CHROMADB_AVAILABLE:
        return None
    
    if _chroma_client is None:
        try:
            # 确保数据库目录存在
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            
            logger.info(f"正在连接ChromaDB: {CHROMA_DB_PATH}")
            _chroma_client = chromadb.PersistentClient(
                path=CHROMA_DB_PATH,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            _chroma_db_connected = True
            logger.info("ChromaDB连接成功")
        except Exception as e:
            logger.error(f"连接ChromaDB失败: {str(e)}")
            _chroma_client = None
            _chroma_db_connected = False
    
    return _chroma_client


def get_chroma_collection(collection_name: str):
    """获取或创建ChromaDB集合"""
    client = get_chroma_client()
    if client is None:
        return None
    
    try:
        # 获取或创建集合
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        return collection
    except Exception as e:
        logger.error(f"获取ChromaDB集合失败: {str(e)}")
        return None


# ==================== LLM服务（使用Qwen3-8B） ====================

async def call_qwen_api(prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """调用Qwen3-8B API"""
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(LLM_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"调用Qwen API失败: {str(e)}")
        return f"生成回答时出错: {str(e)}"


# ==================== 辅助函数 ====================

def _split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """将文本分割成块"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # 尝试在句子边界处分割
        if end < len(text):
            for sep in [".", "?", "!", "。", "？", "！", "\n"]:
                pos = chunk.rfind(sep)
                if pos > chunk_size * 0.5:
                    chunk = chunk[:pos + 1]
                    end = start + pos + 1
                    break
        
        chunks.append(chunk.strip())
        start = end - chunk_overlap
    
    return chunks


# ==================== API路由 ====================

@router.get("/collections", response_model=CollectionListResponse)
async def get_collections():
    """获取所有向量库列表"""
    collections = []
    
    # 从ChromaDB获取
    client = get_chroma_client()
    if client:
        try:
            chroma_collections = client.list_collections()
            for col_name in chroma_collections:
                try:
                    col = client.get_collection(col_name)
                    count = col.count()
                    
                    # 从内存获取元数据
                    col_id = col_name.replace("rag_", "")
                    col_data = _memory_collections.get(col_id, {})
                    
                    collections.append(CollectionInfo(
                        id=col_id,
                        name=col_data.get("name", col_name),
                        description=col_data.get("description", ""),
                        document_count=count,
                        dimension=EMBEDDING_DIMENSION,
                        created_at=col_data.get("created_at", ""),
                        updated_at=col_data.get("updated_at", "")
                    ))
                except Exception as e:
                    logger.warning(f"获取集合 {col_name} 信息失败: {str(e)}")
        except Exception as e:
            logger.error(f"从ChromaDB获取集合列表失败: {str(e)}")
    
    # 如果ChromaDB不可用，从内存获取
    if not collections:
        for col_id, col_data in _memory_collections.items():
            collections.append(CollectionInfo(
                id=col_id,
                name=col_data["name"],
                description=col_data.get("description", ""),
                document_count=len(col_data.get("documents", [])),
                dimension=EMBEDDING_DIMENSION,
                created_at=col_data.get("created_at", ""),
                updated_at=col_data.get("updated_at", "")
            ))
    
    return CollectionListResponse(collections=collections, total=len(collections))


@router.post("/collections")
async def create_collection(request: CollectionCreate):
    """创建新的向量库"""
    # 检查名称是否已存在
    for col in _memory_collections.values():
        if col["name"] == request.name:
            raise HTTPException(status_code=400, detail="向量库名称已存在")
    
    collection_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    collection_name = f"rag_{collection_id}"
    
    _memory_collections[collection_id] = {
        "id": collection_id,
        "name": request.name,
        "description": request.description,
        "documents": [],
        "dimension": EMBEDDING_DIMENSION,
        "created_at": now,
        "updated_at": now,
        "chroma_collection": collection_name
    }
    
    # 如果ChromaDB可用，创建集合
    if get_chroma_client():
        chroma_col = get_chroma_collection(collection_name)
        if chroma_col:
            logger.info(f"在ChromaDB中创建集合: {collection_name}")
    
    logger.info(f"创建向量库: {collection_id} - {request.name}")
    
    # 保存到文件
    _save_collections()
    
    return {
        "id": collection_id,
        "name": request.name,
        "message": "向量库创建成功"
    }


@router.get("/collections/{collection_id}")
async def get_collection(collection_id: str):
    """获取向量库详细信息"""
    if collection_id not in _memory_collections:
        raise HTTPException(status_code=404, detail="向量库不存在")
    
    col = _memory_collections[collection_id]
    
    # 从ChromaDB获取文档数量
    doc_count = len(col.get("documents", []))
    if get_chroma_client() and "chroma_collection" in col:
        try:
            chroma_col = get_chroma_collection(col["chroma_collection"])
            if chroma_col:
                doc_count = chroma_col.count()
        except:
            pass
    
    return {
        "id": collection_id,
        "name": col["name"],
        "description": col.get("description", ""),
        "document_count": doc_count,
        "dimension": EMBEDDING_DIMENSION,
        "created_at": col.get("created_at", ""),
        "updated_at": col.get("updated_at", "")
    }


@router.delete("/collections/{collection_id}")
async def delete_collection(collection_id: str):
    """删除向量库"""
    if collection_id not in _memory_collections:
        raise HTTPException(status_code=404, detail="向量库不存在")
    
    col = _memory_collections[collection_id]
    
    # 删除ChromaDB中的集合
    if get_chroma_client() and "chroma_collection" in col:
        try:
            client = get_chroma_client()
            client.delete_collection(col["chroma_collection"])
            logger.info(f"删除ChromaDB集合: {col['chroma_collection']}")
        except Exception as e:
            logger.warning(f"删除ChromaDB集合失败: {str(e)}")
    
    # 删除内存中的数据
    for doc_id in col.get("documents", []):
        if doc_id in _memory_documents:
            del _memory_documents[doc_id]
        if doc_id in _memory_vectors:
            del _memory_vectors[doc_id]
    
    del _memory_collections[collection_id]
    
    logger.info(f"删除向量库: {collection_id}")
    
    # 保存到文件
    _save_collections()
    _save_documents()
    
    return {"message": "向量库已删除"}


@router.get("/collections/{collection_id}/documents")
async def get_collection_documents(collection_id: str):
    """获取向量库中的文档列表"""
    if collection_id not in _memory_collections:
        raise HTTPException(status_code=404, detail="向量库不存在")
    
    col = _memory_collections[collection_id]
    documents = []
    
    for doc_id in col.get("documents", []):
        if doc_id in _memory_documents:
            doc = _memory_documents[doc_id]
            documents.append(DocumentInfo(
                id=doc_id,
                file_name=doc.get("file_name", "Unknown"),
                file_id=doc.get("file_id", ""),
                chunk_count=doc.get("chunk_count", 0),
                added_at=doc.get("added_at", "")
            ))
    
    return DocumentListResponse(documents=documents, total=len(documents))


@router.post("/collections/{collection_id}/documents")
async def add_documents(collection_id: str, request: DocumentAddRequest):
    """将文档添加到向量库"""
    if collection_id not in _memory_collections:
        raise HTTPException(status_code=404, detail="向量库不存在")
    
    from core.file_manager import file_manager
    
    col = _memory_collections[collection_id]
    added_count = 0
    total_chunks = 0
    total_files = len(request.file_ids)
    
    print(f"\n{'=' * 70}")
    print(f"📚 向量化任务 - 共 {total_files} 个文件")
    print(f"{'=' * 70}")
    
    # 确保Embedding模型已加载
    print("\n[Step 1/4] 加载Embedding模型...")
    get_embedding_model()
    print("✓ Embedding模型加载完成")
    
    # 获取ChromaDB集合
    chroma_col = None
    if get_chroma_client() and "chroma_collection" in col:
        chroma_col = get_chroma_collection(col["chroma_collection"])
        print(f"✓ 使用ChromaDB集合: {col['chroma_collection']}")
    else:
        print("⚠ 使用内存存储模式")
    
    print(f"\n[Step 2/4] 处理文件 (0/{total_files})")
    print("-" * 70)
    
    # 使用tqdm创建进度条
    pbar = tqdm(
        request.file_ids,
        desc="向量化进度",
        total=total_files,
        unit="文件",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ncols=70
    )
    
    for file_id in pbar:
        idx = pbar.n
        
        # 获取文件信息
        file_info = file_manager.get_file_info(file_id)
        if not file_info:
            pbar.set_postfix_str(f"⚠ 文件不存在")
            continue
        
        # 更新进度条描述
        pbar.set_description(f"处理: {file_info.original_name[:30]}...")
        
        # 获取处理后的内容（优先使用organized，其次是llm_denoised，最后是raw）
        content = file_manager.get_stage_content(file_id, "organized")
        source_type = "organized"
        if not content:
            content = file_manager.get_stage_content(file_id, "llm_denoised")
            source_type = "llm_denoised"
        if not content:
            content = file_manager.get_stage_content(file_id, "raw")
            source_type = "raw"
        
        if not content:
            pbar.set_postfix_str(f"⚠ 无内容")
            continue
        
        # 更新进度状态
        pbar.set_postfix_str(f"📖 {source_type} | {len(content)}字符")
        
        # 文本分块
        chunks = _split_text(content, request.chunk_size, request.chunk_overlap)
        pbar.set_postfix_str(f"✂️ {len(chunks)}块")
        
        # 生成向量
        chunk_embeddings = encode_texts(chunks)
        pbar.set_postfix_str(f"🔢 {len(chunk_embeddings)}向量")
        
        # 创建文档记录
        doc_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        _memory_documents[doc_id] = {
            "id": doc_id,
            "file_id": file_id,
            "file_name": file_info.original_name,
            "collection_id": collection_id,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "added_at": now,
            "embeddings": chunk_embeddings
        }
        
        # 添加到向量库
        col["documents"].append(doc_id)
        
        # 如果ChromaDB可用，插入数据
        if chroma_col:
            try:
                chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
                chroma_col.add(
                    ids=chunk_ids,
                    embeddings=chunk_embeddings,
                    documents=chunks,
                    metadatas=[{
                        "doc_id": doc_id,
                        "file_name": file_info.original_name,
                        "collection_id": collection_id
                    } for _ in chunks]
                )
                pbar.set_postfix_str(f"✅ 已存入ChromaDB")
            except Exception as e:
                pbar.set_postfix_str(f"❌ 存入失败")
        
        added_count += 1
        total_chunks += len(chunks)
    
    # 关闭进度条
    pbar.close()
    
    print("-" * 70)
    print(f"✅ 向量化完成: {added_count}/{total_files} 个文件, {total_chunks} 个文本块")
    print(f"{'=' * 70}\n")
    
    # 更新向量库时间
    col["updated_at"] = datetime.now().isoformat()
    
    # 保存到文件
    _save_collections()
    _save_documents()
    
    return {
        "added_count": added_count,
        "chunk_count": total_chunks,
        "message": f"成功添加 {added_count} 个文档，共 {total_chunks} 个文本块"
    }


@router.delete("/collections/{collection_id}/documents/{document_id}")
async def delete_document(collection_id: str, document_id: str):
    """从向量库中删除文档"""
    if collection_id not in _memory_collections:
        raise HTTPException(status_code=404, detail="向量库不存在")
    
    if document_id not in _memory_documents:
        raise HTTPException(status_code=404, detail="文档不存在")
    
    col = _memory_collections[collection_id]
    
    # 从ChromaDB中删除（如果可用）
    if get_chroma_client() and "chroma_collection" in col:
        try:
            chroma_col = get_chroma_collection(col["chroma_collection"])
            if chroma_col:
                doc = _memory_documents[document_id]
                chunk_ids = [f"{document_id}_{i}" for i in range(doc.get("chunk_count", 0))]
                chroma_col.delete(ids=chunk_ids)
                logger.info(f"从ChromaDB删除文档: {document_id}")
        except Exception as e:
            logger.warning(f"从ChromaDB删除文档失败: {str(e)}")
    
    # 从向量库中移除
    if document_id in col.get("documents", []):
        col["documents"].remove(document_id)
    
    # 删除文档记录
    if document_id in _memory_documents:
        del _memory_documents[document_id]
    
    logger.info(f"从向量库 {collection_id} 删除文档: {document_id}")
    
    # 保存到文件
    _save_collections()
    _save_documents()
    
    return {"message": "文档已删除"}


@router.post("/collections/{collection_id}/search")
async def search_documents(collection_id: str, request: SearchRequest):
    """在向量库中检索相似文档"""
    if collection_id not in _memory_collections:
        raise HTTPException(status_code=404, detail="向量库不存在")
    
    col = _memory_collections[collection_id]
    
    # 编码查询
    query_embedding = encode_query(request.query)
    
    results = []
    
    # 尝试从ChromaDB检索
    if get_chroma_client() and "chroma_collection" in col:
        try:
            chroma_col = get_chroma_collection(col["chroma_collection"])
            if chroma_col:
                search_results = chroma_col.query(
                    query_embeddings=[query_embedding],
                    n_results=request.top_k,
                    include=["documents", "metadatas", "distances"]
                )
                
                if search_results and search_results["ids"]:
                    for i, doc_id in enumerate(search_results["ids"][0]):
                        distance = search_results["distances"][0][i] if search_results["distances"] else 0
                        document = search_results["documents"][0][i] if search_results["documents"] else ""
                        metadata = search_results["metadatas"][0][i] if search_results["metadatas"] else {}
                        
                        # ChromaDB返回的是距离（越小越相似），转换为相似度分数
                        score = 1.0 - distance
                        
                        results.append(SearchResult(
                            content=document[:500] + "..." if len(document) > 500 else document,
                            score=float(score),
                            source=metadata.get("file_name", "Unknown"),
                            chunk_id=doc_id
                        ))
                
                logger.info(f"从ChromaDB检索到 {len(results)} 个结果")
        except Exception as e:
            logger.error(f"ChromaDB检索失败: {str(e)}")
    
    # 如果ChromaDB没有结果，从内存检索
    if not results:
        logger.info("从内存中检索")
        all_chunks = []
        for doc_id in col.get("documents", []):
            if doc_id in _memory_documents:
                doc = _memory_documents[doc_id]
                for i, chunk in enumerate(doc.get("chunks", [])):
                    chunk_embedding = doc.get("embeddings", [])[i] if i < len(doc.get("embeddings", [])) else None
                    all_chunks.append({
                        "content": chunk,
                        "source": doc.get("file_name", "Unknown"),
                        "chunk_id": f"{doc_id}_{i}",
                        "embedding": chunk_embedding
                    })
        
        if all_chunks:
            # 计算相似度
            import numpy as np
            
            def cosine_similarity(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
            scored_chunks = []
            for chunk in all_chunks:
                if chunk["embedding"]:
                    score = cosine_similarity(query_embedding, chunk["embedding"])
                    scored_chunks.append((score, chunk))
            
            # 排序并取前top_k
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            for score, chunk in scored_chunks[:request.top_k]:
                results.append(SearchResult(
                    content=chunk["content"][:500] + "..." if len(chunk["content"]) > 500 else chunk["content"],
                    score=float(score),
                    source=chunk["source"],
                    chunk_id=chunk["chunk_id"]
                ))
    
    return SearchResponse(
        query=request.query,
        results=results,
        total=len(results)
    )


@router.post("/chat")
async def chat_with_rag(request: ChatRequest):
    """RAG对话问答"""
    if not request.query:
        raise HTTPException(status_code=400, detail="查询内容不能为空")
    
    # 构建提示词
    context = request.context if request.context else "暂无相关上下文"
    
    # 构建对话历史
    history_text = ""
    if request.chat_history:
        for i, turn in enumerate(request.chat_history[-3:]):  # 只保留最近3轮
            if len(turn) >= 2:
                history_text += f"用户：{turn[0]}\n助手：{turn[1]}\n\n"
    
    prompt = f"""你是一个专业的文档问答助手。基于以下上下文信息回答用户的问题。

上下文信息：
{context}

{history_text}用户问题：{request.query}

请根据上下文信息提供准确、简洁的回答。如果上下文不包含相关信息，请明确说明。回答时请引用相关信息来源。

回答："""
    
    # 调用Qwen3-8B生成回答
    answer = await call_qwen_api(prompt, temperature=0.7, max_tokens=2000)
    
    # 提取来源（从上下文中提取文件名）
    sources = list(set([line.split("：")[0] for line in context.split("\n") if "：" in line]))[:5]
    if not sources:
        sources = ["相关文档"]
    
    return ChatResponse(answer=answer, sources=sources)


# ==================== WebSocket端点 ====================

@router.websocket("/ws/progress/{client_id}")
async def websocket_progress(websocket: WebSocket, client_id: str):
    """WebSocket端点：实时推送向量化进度"""
    await progress_manager.connect(client_id, websocket)
    try:
        while True:
            # 保持连接，接收客户端心跳
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # 发送心跳保持连接
                try:
                    await websocket.send_text("heartbeat")
                except:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        # 忽略连接重置错误
        if "10054" not in str(e) and "Connection reset" not in str(e):
            logger.error(f"WebSocket错误: {str(e)}")
    finally:
        progress_manager.disconnect(client_id)


# ==================== 系统接口 ====================

@router.get("/health")
async def health_check():
    """RAG系统健康检查"""
    # 检查ChromaDB连接
    client = get_chroma_client()
    chromadb_status = "connected" if client is not None else "disconnected"
    
    # 检查Embedding模型
    model = get_embedding_model()
    embedding_status = "loaded" if model is not None else "not_loaded"
    
    # 获取ChromaDB统计
    collection_count = 0
    if client:
        try:
            collections = client.list_collections()
            collection_count = len(collections)
        except:
            pass
    
    return {
        "status": "healthy",
        "chromadb": chromadb_status,
        "chromadb_path": CHROMA_DB_PATH,
        "collection_count": collection_count,
        "embedding_model": embedding_status,
        "embedding_model_path": EMBEDDING_MODEL_PATH,
        "llm_model": LLM_MODEL_NAME,
        "memory_collections": len(_memory_collections),
        "memory_documents": len(_memory_documents),
        "version": "2.0.0"
    }


@router.post("/init")
async def init_rag_system():
    """初始化RAG系统（加载模型等）"""
    try:
        # 加载Embedding模型
        model = get_embedding_model()
        
        # 连接ChromaDB
        client = get_chroma_client()
        chromadb_connected = client is not None
        
        return {
            "status": "success",
            "embedding_model_loaded": model is not None,
            "chromadb_connected": chromadb_connected,
            "chromadb_path": CHROMA_DB_PATH,
            "message": "RAG系统初始化完成"
        }
    except Exception as e:
        logger.error(f"RAG系统初始化失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"初始化失败: {str(e)}")


# ==================== 启动时加载数据 ====================

# 加载向量库和文档元数据
_load_collections()
_load_documents()

# 从ChromaDB同步集合信息（需要在get_chroma_client定义之后）
_sync_from_chromadb()
