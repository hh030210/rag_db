"""
文件管理器 - 负责文件的存储、检索和状态管理
"""
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

import config
from models.schemas import FileInfo, FileStatus, StageResult

class FileManager:
    """文件管理器"""
    
    def __init__(self):
        self.upload_dir = config.UPLOAD_DIR
        self.output_dir = config.OUTPUT_DIR
        self.metadata_file = config.BASE_DIR / "file_metadata.json"
        self._files: Dict[str, FileInfo] = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """加载文件元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for file_id, file_data in data.items():
                        self._files[file_id] = FileInfo(**file_data)
                logger.info(f"已加载 {len(self._files)} 个文件的元数据")
            except Exception as e:
                logger.error(f"加载元数据失败: {e}")
                self._files = {}
    
    def _save_metadata(self):
        """保存文件元数据"""
        try:
            data = {k: v.dict() for k, v in self._files.items()}
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
    
    def create_file_record(self, original_name: str, file_type: str, 
                          file_size: int) -> FileInfo:
        """创建文件记录"""
        file_id = str(uuid.uuid4())
        filename = f"{file_id}{Path(original_name).suffix}"
        
        file_info = FileInfo(
            id=file_id,
            filename=filename,
            original_name=original_name,
            file_type=file_type,
            file_size=file_size,
            upload_time=datetime.now(),
            status=FileStatus.PENDING
        )
        
        self._files[file_id] = file_info
        self._save_metadata()
        
        logger.info(f"创建文件记录: {file_id} - {original_name}")
        return file_info
    
    def get_file_info(self, file_id: str) -> Optional[FileInfo]:
        """获取文件信息"""
        return self._files.get(file_id)
    
    def get_file_path(self, file_id: str) -> Optional[Path]:
        """获取文件路径"""
        file_info = self._files.get(file_id)
        if file_info:
            return self.upload_dir / file_info.filename
        return None
    
    def get_output_path(self, file_id: str, stage: str, suffix: str = ".txt") -> Path:
        """获取输出文件路径"""
        return self.output_dir / f"{file_id}_{stage}{suffix}"
    
    def update_file_status(self, file_id: str, status: FileStatus, 
                          error_message: Optional[str] = None):
        """更新文件状态"""
        if file_id in self._files:
            self._files[file_id].status = status
            if error_message:
                self._files[file_id].error_message = error_message
            self._save_metadata()
            logger.info(f"更新文件状态: {file_id} -> {status}")
    
    def update_stage_result(self, file_id: str, stage: str, 
                           result: StageResult):
        """更新阶段处理结果"""
        if file_id in self._files:
            if file_id not in self._files[file_id].stages:
                self._files[file_id].stages = {}
            self._files[file_id].stages[stage] = result.dict()
            self._save_metadata()
            logger.info(f"更新阶段结果: {file_id} - {stage}")
    
    def list_files(self) -> List[FileInfo]:
        """列出所有文件（只返回实际存在的文件）"""
        existing_files = []
        files_to_remove = []
        
        for file_id, file_info in self._files.items():
            file_path = self.get_file_path(file_id)
            if file_path and file_path.exists():
                existing_files.append(file_info)
            else:
                # 文件不存在，标记为待删除
                files_to_remove.append(file_id)
        
        # 清理不存在的文件记录
        if files_to_remove:
            for file_id in files_to_remove:
                del self._files[file_id]
            self._save_metadata()
            logger.info(f"清理了 {len(files_to_remove)} 个不存在的文件记录")
        
        return existing_files
    
    def delete_file(self, file_id: str) -> bool:
        """删除文件"""
        if file_id not in self._files:
            return False
        
        try:
            # 删除上传的文件
            file_path = self.get_file_path(file_id)
            if file_path and file_path.exists():
                file_path.unlink()
            
            # 删除输出文件
            for stage in config.PROCESSING_STAGES:
                output_path = self.get_output_path(file_id, stage["id"])
                if output_path.exists():
                    output_path.unlink()
            
            # 删除记录
            del self._files[file_id]
            self._save_metadata()
            
            logger.info(f"删除文件: {file_id}")
            return True
        except Exception as e:
            logger.error(f"删除文件失败: {e}")
            return False
    
    def get_stage_content(self, file_id: str, stage: str) -> Optional[str]:
        """获取阶段处理内容"""
        output_path = self.get_output_path(file_id, stage)
        if output_path.exists():
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"读取文件内容失败: {e}")
        return None

# 全局文件管理器实例
file_manager = FileManager()
