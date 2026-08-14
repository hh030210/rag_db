"""
FastAPI 主应用
"""
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import aiofiles
from loguru import logger

import config
from models.schemas import (
    FileInfo, FileListResponse, ProcessingRequest, 
    ProcessingResponse, PreviewRequest, PreviewResponse,
    FileStatus, ProcessingStage
)
from core.file_manager import file_manager
from core.file_processor import file_processor
from app.rag_routes import router as rag_router

# 配置日志
logger.add(
    config.BASE_DIR / "logs" / "app.log",
    rotation="10 MB",
    retention="30 days",
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT
)

# 创建FastAPI应用
app = FastAPI(
    title="智能文档处理与RAG系统",
    description="集成文件转换、去噪、内容重组和RAG检索增强生成的智能文档处理系统",
    version="2.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 确保日志目录存在
(config.BASE_DIR / "logs").mkdir(exist_ok=True)

# 注册RAG路由
app.include_router(rag_router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "智能文档处理与RAG系统 API",
        "version": "2.0.0",
        "docs": "/docs",
        "features": [
            "第二章：文件转换与处理系统",
            "第三章：RAG检索增强生成系统"
        ]
    }


@app.get("/api/stages")
async def get_stages():
    """获取处理阶段列表"""
    return config.PROCESSING_STAGES


@app.post("/api/files/upload", response_model=FileInfo)
async def upload_file(file: UploadFile = File(...)):
    """上传单个文件"""
    # 检查文件类型
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的文件类型: {file_ext}"
        )
    
    # 检查文件大小
    content = await file.read()
    if len(content) > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超过限制: {config.MAX_FILE_SIZE / 1024 / 1024}MB"
        )
    
    # 创建文件记录
    file_info = file_manager.create_file_record(
        original_name=file.filename,
        file_type=file_ext,
        file_size=len(content)
    )
    
    # 保存文件
    file_path = file_manager.get_file_path(file_info.id)
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)
    
    logger.info(f"文件上传成功: {file_info.id} - {file.filename}")
    return file_info


@app.post("/api/files/upload-batch")
async def upload_files(files: List[UploadFile] = File(...)):
    """批量上传文件"""
    uploaded_files = []
    errors = []
    
    for file in files:
        try:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in config.ALLOWED_EXTENSIONS:
                errors.append(f"{file.filename}: 不支持的文件类型")
                continue
            
            content = await file.read()
            if len(content) > config.MAX_FILE_SIZE:
                errors.append(f"{file.filename}: 文件大小超过限制")
                continue
            
            file_info = file_manager.create_file_record(
                original_name=file.filename,
                file_type=file_ext,
                file_size=len(content)
            )
            
            file_path = file_manager.get_file_path(file_info.id)
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            uploaded_files.append(file_info)
            
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")
    
    return {
        "uploaded": uploaded_files,
        "errors": errors,
        "total": len(files),
        "success": len(uploaded_files)
    }


@app.get("/api/files", response_model=FileListResponse)
async def list_files():
    """获取文件列表"""
    files = file_manager.list_files()
    return FileListResponse(files=files, total=len(files))


@app.get("/api/files/{file_id}", response_model=FileInfo)
async def get_file(file_id: str):
    """获取文件信息"""
    file_info = file_manager.get_file_info(file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="文件不存在")
    return file_info


@app.post("/api/files/{file_id}/process", response_model=ProcessingResponse)
async def process_file(
    file_id: str, 
    background_tasks: BackgroundTasks,
    request: ProcessingRequest = None
):
    """处理文件"""
    file_info = file_manager.get_file_info(file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    if file_info.status == FileStatus.PROCESSING:
        raise HTTPException(status_code=400, detail="文件正在处理中")
    
    # 确定处理阶段
    stages = request.stages if request else [
        ProcessingStage.RAW,
        ProcessingStage.RULE_DENOISED,
        ProcessingStage.LLM_DENOISED,
        ProcessingStage.ORGANIZED
    ]
    
    # 在后台处理文件
    background_tasks.add_task(file_processor.process_file, file_id, stages)
    
    return ProcessingResponse(
        file_id=file_id,
        status=FileStatus.PROCESSING,
        current_stage=stages[0],
        message="文件处理已启动"
    )


@app.get("/api/files/{file_id}/preview")
async def preview_file(file_id: str, stage: str = "raw", max_length: int = 5000):
    """预览文件内容"""
    file_info = file_manager.get_file_info(file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    content = file_manager.get_stage_content(file_id, stage)
    if content is None:
        raise HTTPException(status_code=404, detail="阶段文件不存在")
    
    is_truncated = len(content) > max_length
    preview = content[:max_length] if is_truncated else content
    
    return PreviewResponse(
        file_id=file_id,
        stage=stage,
        content=preview,
        total_length=len(content),
        is_truncated=is_truncated
    )


@app.get("/api/files/{file_id}/download/{stage}")
async def download_file(file_id: str, stage: str):
    """下载处理后的文件"""
    file_info = file_manager.get_file_info(file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    output_path = file_manager.get_output_path(file_id, stage)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="阶段文件不存在")
    
    # 生成下载文件名
    original_name = Path(file_info.original_name).stem
    download_name = f"{original_name}_{stage}.txt"
    
    return FileResponse(
        path=output_path,
        filename=download_name,
        media_type="text/plain"
    )


@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str):
    """删除文件"""
    success = file_manager.delete_file(file_id)
    if not success:
        raise HTTPException(status_code=404, detail="文件不存在")
    return {"message": "文件已删除"}


@app.get("/api/files/{file_id}/status")
async def get_file_status(file_id: str):
    """获取文件处理状态"""
    file_info = file_manager.get_file_info(file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return {
        "file_id": file_id,
        "status": file_info.status,
        "stages": file_info.stages
    }


@app.get("/api/progress")
async def get_all_progress():
    """获取所有正在处理的文件的详细进度"""
    from core.progress_tracker import progress_tracker
    
    progress_data = progress_tracker.get_all_progress()
    result = {}
    
    for file_id, progress in progress_data.items():
        result[file_id] = {
            "file_id": progress.file_id,
            "file_name": progress.file_name,
            "current_stage": progress.current_stage,
            "stage_progress": progress.stage_progress,
            "overall_progress": progress.overall_progress,
            "status": progress.status,
            "message": progress.message,
            "elapsed_time": time.time() - progress.start_time
        }
    
    return {"progress": result}


@app.get("/api/progress/{file_id}")
async def get_file_progress(file_id: str):
    """获取单个文件的处理进度"""
    from core.progress_tracker import progress_tracker
    
    progress = progress_tracker.get_progress(file_id)
    if not progress:
        raise HTTPException(status_code=404, detail="未找到该文件的处理进度")
    
    return {
        "file_id": progress.file_id,
        "file_name": progress.file_name,
        "current_stage": progress.current_stage,
        "stage_progress": progress.stage_progress,
        "overall_progress": progress.overall_progress,
        "status": progress.status,
        "message": progress.message,
        "elapsed_time": time.time() - progress.start_time
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        log_level="info"
    )
