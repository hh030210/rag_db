"""
文件处理器 - 负责文件转换和处理流程
集成 chapter2 的处理逻辑
"""
import asyncio
import sys
import time
import re
from pathlib import Path
from typing import Optional
from loguru import logger

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from core.file_manager import file_manager
from core.progress_tracker import progress_tracker
from models.schemas import FileStatus, StageResult, ProcessingStage

# 导入 chapter2 的处理模块
from utils.editable_parser import parse_editable
from utils.vlm_parser import parse_non_editable
from utils.llm_denoiser import rule_based_clean, LLMDenoiser
from utils.file_identifier import identify_file_type


class FileProcessor:
    """文件处理器"""
    
    def __init__(self):
        self.processing = False
        self.llm_denoiser = None  # 延迟初始化
    
    def _get_llm_denoiser(self):
        """获取或创建LLM去噪器实例"""
        if self.llm_denoiser is None:
            self.llm_denoiser = LLMDenoiser(
                chunk_size=2000,
                log_file=str(config.BASE_DIR / "logs" / "llm_interaction.log"),
                organize_log_file=str(config.BASE_DIR / "logs" / "organize_interaction.log")
            )
        return self.llm_denoiser
    
    async def process_file(self, file_id: str, stages: list):
        """处理文件"""
        file_info = file_manager.get_file_info(file_id)
        file_name = file_info.original_name if file_info else file_id
        
        # 开始跟踪进度
        progress_tracker.start_processing(file_id, file_name, stages)
        file_manager.update_file_status(file_id, FileStatus.PROCESSING)
        
        try:
            total_stages = len(stages)
            for idx, stage in enumerate(stages):
                # 更新总体进度
                overall_progress = (idx / total_stages) * 100
                progress_tracker.update_overall(file_id, overall_progress)
                
                await self._process_stage(file_id, stage)
            
            progress_tracker.complete_processing(file_id)
            file_manager.update_file_status(file_id, FileStatus.COMPLETED)
            logger.info(f"文件处理完成: {file_id}")
            
        except Exception as e:
            logger.error(f"文件处理失败: {file_id}, 错误: {e}")
            progress_tracker.fail_processing(file_id, str(e))
            file_manager.update_file_status(file_id, FileStatus.FAILED, str(e))
    
    async def _process_stage(self, file_id: str, stage: str):
        """处理单个阶段"""
        start_time = time.time()
        file_info = file_manager.get_file_info(file_id)
        
        if not file_info:
            raise ValueError(f"文件不存在: {file_id}")
        
        logger.info(f"开始处理阶段: {file_id} - {stage}")
        
        try:
            if stage == ProcessingStage.RAW:
                await self._extract_raw_text(file_id)
            elif stage == ProcessingStage.RULE_DENOISED:
                await self._rule_based_denoise(file_id)
            elif stage == ProcessingStage.LLM_DENOISED:
                await self._llm_denoise(file_id)
            elif stage == ProcessingStage.ORGANIZED:
                await self._organize_content(file_id)
            
            processing_time = time.time() - start_time
            
            # 读取预览内容
            output_path = file_manager.get_output_path(file_id, stage)
            preview_content = ""
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8') as f:
                    preview_content = f.read()[:1000]  # 前1000字符作为预览
            
            result = StageResult(
                stage=stage,
                status=FileStatus.COMPLETED,
                output_file=str(output_path),
                preview_content=preview_content,
                processing_time=processing_time
            )
            
            file_manager.update_stage_result(file_id, stage, result)
            logger.info(f"阶段处理完成: {file_id} - {stage}, 耗时: {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"阶段处理失败: {file_id} - {stage}, 错误: {e}")
            result = StageResult(
                stage=stage,
                status=FileStatus.FAILED,
                error_message=str(e)
            )
            file_manager.update_stage_result(file_id, stage, result)
            raise
    
    async def _extract_raw_text(self, file_id: str):
        """提取原始文本 - 使用 chapter2 的逻辑"""
        file_path = file_manager.get_file_path(file_id)
        file_info = file_manager.get_file_info(file_id)
        
        if not file_path or not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_id}")
        
        # 使用 chapter2 的文件类型识别
        file_type, is_editable = identify_file_type(str(file_path))
        logger.info(f"文件类型识别: {file_id} - {file_type}, 可编辑: {is_editable}")
        
        if is_editable:
            # 使用 chapter2 的可编辑文件解析器
            text = parse_editable(str(file_path), file_type)
        else:
            # 使用 chapter2 的 VLM 解析器（非可编辑PDF）
            text = parse_non_editable(str(file_path), file_type)
        
        if not text:
            raise ValueError("未能提取到有效内容")
        
        # 清理 VLM 占位符
        text = re.sub(r'<\|LOC_\d+\|>', '', text)
        
        # 保存原始文本
        output_path = file_manager.get_output_path(file_id, ProcessingStage.RAW)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"原始文本已保存: {output_path}")
    
    async def _rule_based_denoise(self, file_id: str):
        """基于规则的去噪 - 使用 chapter2 的 rule_based_clean"""
        # 读取原始文本
        raw_path = file_manager.get_output_path(file_id, ProcessingStage.RAW)
        with open(raw_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 使用 chapter2 的规则去噪函数
        cleaned_text = rule_based_clean(text)
        
        # 保存去噪后的文本
        output_path = file_manager.get_output_path(file_id, ProcessingStage.RULE_DENOISED)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        logger.info(f"规则去噪完成: {output_path}")
    
    async def _llm_denoise(self, file_id: str):
        """基于LLM的去噪 - 使用 chapter2 的 LLMDenoiser"""
        # 更新进度
        progress_tracker.update_stage(file_id, "llm_denoised", 0, "读取规则去噪后的文本...")
        
        # 读取规则去噪后的文本
        rule_path = file_manager.get_output_path(file_id, ProcessingStage.RULE_DENOISED)
        with open(rule_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 获取文件类型
        file_info = file_manager.get_file_info(file_id)
        file_type = Path(file_info.original_name).suffix.lower() if file_info else ".txt"
        
        # 使用 chapter2 的 LLM 去噪器
        denoiser = self._get_llm_denoiser()
        
        logger.info(f"开始LLM去噪: {file_id}")
        progress_tracker.update_stage(file_id, "llm_denoised", 10, "开始LLM去噪，提取噪声特征...")
        
        # 第一步: 提取噪声特征
        noise_info = denoiser.extract_noise_types(text)
        
        progress_tracker.update_stage(file_id, "llm_denoised", 30, "噪声特征提取完成，准备语义去噪...")
        
        if not noise_info:
            logger.warning(f"噪声特征提取失败，使用默认规则: {file_id}")
            progress_tracker.update_stage(file_id, "llm_denoised", 100, "噪声特征提取失败，使用规则去噪结果")
            # 如果提取失败，直接使用规则去噪的结果
            output_path = file_manager.get_output_path(file_id, ProcessingStage.LLM_DENOISED)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            return
        
        # 第二步: 执行语义去噪
        progress_tracker.update_stage(file_id, "llm_denoised", 40, "执行语义去噪...")
        denoise_result = denoiser.denoise_text(text, noise_info, file_type)
        
        progress_tracker.update_stage(file_id, "llm_denoised", 80, "语义去噪完成，保存结果...")
        
        if not denoise_result:
            logger.warning(f"LLM去噪执行失败，使用规则去噪结果: {file_id}")
            output_path = file_manager.get_output_path(file_id, ProcessingStage.LLM_DENOISED)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            return
        
        llm_only_text = denoise_result.get("去噪后文本", text)
        noise_content = denoise_result.get("噪声内容", [])
        
        # 保存去噪后的文本
        output_path = file_manager.get_output_path(file_id, ProcessingStage.LLM_DENOISED)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(llm_only_text)
        
        # 保存噪声日志
        log_path = config.OUTPUT_DIR / f"{file_id}_llm_denoise_log.json"
        import json
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "extracted_features": noise_info,
                "identified_noise_fragments": noise_content
            }, f, ensure_ascii=False, indent=4)
        
        logger.info(f"LLM去噪完成: {output_path}, 噪声片段数: {len(noise_content)}")
    
    async def _organize_content(self, file_id: str):
        """内容重组和归纳 - 使用 chapter2 的 organize_content"""
        # 读取LLM去噪后的文本
        llm_path = file_manager.get_output_path(file_id, ProcessingStage.LLM_DENOISED)
        with open(llm_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 获取文件类型
        file_info = file_manager.get_file_info(file_id)
        file_type = Path(file_info.original_name).suffix.lower() if file_info else ".txt"
        
        # 使用 chapter2 的 LLM 去噪器进行内容重组
        denoiser = self._get_llm_denoiser()
        
        logger.info(f"开始内容重组: {file_id}")
        
        # 执行内容重组
        paragraphs, responses = denoiser.organize_content(text, file_type)
        
        if not paragraphs:
            logger.warning(f"内容重组失败，使用LLM去噪结果: {file_id}")
            output_path = file_manager.get_output_path(file_id, ProcessingStage.ORGANIZED)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            return
        
        # 保存最终重组文本
        output_path = file_manager.get_output_path(file_id, ProcessingStage.ORGANIZED)
        with open(output_path, 'w', encoding='utf-8') as f:
            # 每个段落之间用空行分隔
            f.write('\n'.join(paragraphs))
        
        # 保存内容重组日志
        organize_log_path = config.OUTPUT_DIR / f"{file_id}_organize_log.json"
        import json
        with open(organize_log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "重组后段落": paragraphs,
                "原始响应": responses
            }, f, ensure_ascii=False, indent=4)
        
        logger.info(f"内容重组完成: {output_path}, 段落数: {len(paragraphs)}")


# 全局文件处理器实例
file_processor = FileProcessor()
