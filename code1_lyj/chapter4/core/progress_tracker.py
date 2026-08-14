"""
处理进度跟踪器 - 跟踪文件处理的详细进度
"""
import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ProcessingProgress:
    """处理进度信息"""
    file_id: str
    file_name: str
    current_stage: str
    stage_progress: float = 0.0  # 当前阶段的进度 (0-100)
    overall_progress: float = 0.0  # 总体进度 (0-100)
    status: str = "processing"  # processing, completed, failed
    message: str = ""  # 详细消息
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)


class ProgressTracker:
    """进度跟踪器 - 单例模式"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._progress: Dict[str, ProcessingProgress] = {}
        return cls._instance
    
    def start_processing(self, file_id: str, file_name: str, stages: list):
        """开始处理文件"""
        self._progress[file_id] = ProcessingProgress(
            file_id=file_id,
            file_name=file_name,
            current_stage=stages[0] if stages else "unknown",
            overall_progress=0.0,
            status="processing",
            message=f"开始处理，共 {len(stages)} 个阶段"
        )
    
    def update_stage(self, file_id: str, stage: str, stage_progress: float, message: str = ""):
        """更新当前阶段进度"""
        if file_id in self._progress:
            progress = self._progress[file_id]
            progress.current_stage = stage
            progress.stage_progress = stage_progress
            progress.message = message
            progress.last_update = time.time()
    
    def update_overall(self, file_id: str, overall_progress: float):
        """更新总体进度"""
        if file_id in self._progress:
            self._progress[file_id].overall_progress = overall_progress
            self._progress[file_id].last_update = time.time()
    
    def complete_processing(self, file_id: str):
        """完成处理"""
        if file_id in self._progress:
            self._progress[file_id].status = "completed"
            self._progress[file_id].overall_progress = 100.0
            self._progress[file_id].stage_progress = 100.0
            self._progress[file_id].message = "处理完成"
            self._progress[file_id].last_update = time.time()
    
    def fail_processing(self, file_id: str, error_message: str):
        """处理失败"""
        if file_id in self._progress:
            self._progress[file_id].status = "failed"
            self._progress[file_id].message = f"处理失败: {error_message}"
            self._progress[file_id].last_update = time.time()
    
    def get_progress(self, file_id: str) -> Optional[ProcessingProgress]:
        """获取进度信息"""
        return self._progress.get(file_id)
    
    def get_all_progress(self) -> Dict[str, ProcessingProgress]:
        """获取所有进度信息"""
        # 清理超过10分钟的旧记录
        current_time = time.time()
        expired = [
            fid for fid, p in self._progress.items()
            if current_time - p.last_update > 600 and p.status in ["completed", "failed"]
        ]
        for fid in expired:
            del self._progress[fid]
        
        return self._progress.copy()
    
    def clear_progress(self, file_id: str):
        """清除进度记录"""
        if file_id in self._progress:
            del self._progress[file_id]


# 全局进度跟踪器实例
progress_tracker = ProgressTracker()
