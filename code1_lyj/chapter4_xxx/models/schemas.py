"""
Pydantic 数据模型定义
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class FileStatus(str, Enum):
    """文件处理状态"""
    PENDING = "pending"           # 等待处理
    PROCESSING = "processing"     # 处理中
    COMPLETED = "completed"       # 完成
    FAILED = "failed"             # 失败

class ProcessingStage(str, Enum):
    """处理阶段"""
    RAW = "raw"
    RULE_DENOISED = "rule_denoised"
    LLM_DENOISED = "llm_denoised"
    ORGANIZED = "organized"

class FileInfo(BaseModel):
    """文件信息模型"""
    id: str = Field(..., description="文件唯一标识")
    filename: str = Field(..., description="原始文件名")
    original_name: str = Field(..., description="用户上传时的文件名")
    file_type: str = Field(..., description="文件类型")
    file_size: int = Field(..., description="文件大小（字节）")
    upload_time: datetime = Field(..., description="上传时间")
    status: FileStatus = Field(default=FileStatus.PENDING)
    stages: Dict[str, Any] = Field(default_factory=dict, description="各阶段处理结果")
    error_message: Optional[str] = Field(None, description="错误信息")

class StageResult(BaseModel):
    """阶段处理结果"""
    stage: ProcessingStage
    status: FileStatus
    output_file: Optional[str] = None
    preview_content: Optional[str] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None

class ProcessingRequest(BaseModel):
    """处理请求"""
    file_id: Optional[str] = Field(default=None, description="文件ID（可选，也可从URL路径获取）")
    stages: List[ProcessingStage] = Field(
        default=[ProcessingStage.RAW, ProcessingStage.RULE_DENOISED, 
                ProcessingStage.LLM_DENOISED, ProcessingStage.ORGANIZED]
    )

class ProcessingResponse(BaseModel):
    """处理响应"""
    file_id: str
    status: FileStatus
    current_stage: Optional[str] = None
    stages_result: List[StageResult] = []
    message: str

class FileListResponse(BaseModel):
    """文件列表响应"""
    files: List[FileInfo]
    total: int

class PreviewRequest(BaseModel):
    """预览请求"""
    file_id: str
    stage: ProcessingStage
    max_length: int = Field(default=5000, description="最大预览长度")

class PreviewResponse(BaseModel):
    """预览响应"""
    file_id: str
    stage: ProcessingStage
    content: str
    total_length: int
    is_truncated: bool
