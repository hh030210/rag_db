"""API 数据模型"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum


class RetrievalMode(str, Enum):
    SEMANTIC = "semantic"
    DIMENSION = "dimension"
    FUSION = "fusion"


class RetrievalResult(BaseModel):
    chunk_id: str
    score: float
    chunk_text: str = ""
    doc_title: str = ""
    chunk_gen_title: str = ""
    doc_id: str = ""


class CollectionInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    document_count: int = 0
    vectors_count: int = 0
    dimension: int = 1024
    created_at: str = ""


class CollectionListResponse(BaseModel):
    collections: List[CollectionInfo]
    total: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    mode: RetrievalMode = RetrievalMode.FUSION
    alpha: float = Field(default=0.2, ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    query: str
    mode: str
    alpha: float
    results: List[RetrievalResult]
    total: int
    timing_ms: float


class ChatRequest(BaseModel):
    query: str
    context: str = ""
    chat_history: List[List[str]] = []
    temperature: float = 0.7
    max_tokens: int = 2000


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    context_chunks: List[RetrievalResult] = []


class PipelineRunRequest(BaseModel):
    input_dir: str
    from_step: int = Field(default=1, ge=1, le=6)
    to_step: int = Field(default=6, ge=1, le=6)
    force: bool = False


class PipelineRunResponse(BaseModel):
    status: str
    message: str
    step_results: Dict[str, Any] = {}


class DBInitRequest(BaseModel):
    force: bool = False


class DBInitResponse(BaseModel):
    mysql: Dict[str, Any] = {}
    qdrant: Dict[str, Any] = {}
