"""
Gradio 前端应用 - 集成第二章（文件转换与处理系统）和第三章（RAG系统）
"""
import os
import sys
import requests
import gradio as gr
from gradio.themes import Soft
from pathlib import Path
from typing import List, Tuple, Dict
import json
import time
import threading

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from pages.rag_page import RAGPage
from pages.tourist_rag_page import TouristRAGPage

# API基础URL (使用localhost而不是0.0.0.0)
API_BASE_URL = f"http://127.0.0.1:{config.PORT}"

# 创建紫色主题
purple_theme = Soft(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=["Inter", "system-ui", "sans-serif"],
).set(
    # 主要按钮颜色
    button_primary_background_fill="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #5a6fd6 0%, #6a4190 100%)",
    button_primary_text_color="#ffffff",
    # 次要按钮
    button_secondary_background_fill="#f3f4f6",
    button_secondary_background_fill_hover="#e5e7eb",
    button_secondary_text_color="#374151",
    # 输入框
    input_background_fill="#ffffff",
    input_border_color="#e5e7eb",
    input_border_color_focus="#667eea",
    # 背景
    background_fill_primary="#f9fafb",
    background_fill_secondary="#ffffff",
    # 边框
    border_color_accent="#667eea",
    # 文本
    body_text_color="#1f2937",
    # 标题
    block_title_text_color="#1f2937",
    block_title_text_weight="600",
    # 标签
    block_label_text_color="#6b7280",
)

# 自定义CSS
CUSTOM_CSS = """
.dropdown-scroll .choices__list--dropdown {
    max-height: 300px !important;
    overflow-y: auto !important;
}
.dropdown-scroll .choices__list--dropdown .choices__list {
    max-height: 300px !important;
    overflow-y: auto !important;
}

/* 标题样式 */
.main-title {
    text-align: center !important;
    padding: 20px !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border-radius: 12px !important;
    color: white !important;
    margin-bottom: 20px !important;
}

.main-title h1 {
    color: white !important;
    font-size: 32px !important;
    font-weight: bold !important;
    margin: 0 !important;
}
"""


class FileConversionUI:
    """文件转换UI类 - 第二章"""
    
    def __init__(self):
        self.current_file_id = None
        self.processing_status = {}  # 存储处理状态
    
    def upload_file(self, files: List[str]):
        """上传文件 - 返回gr.update格式"""
        if not files:
            return "请选择文件", "", gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])
        
        uploaded_count = 0
        errors = []
        
        for file_path in files:
            try:
                with open(file_path, 'rb') as f:
                    response = requests.post(
                        f"{API_BASE_URL}/api/files/upload",
                        files={"file": f}
                    )
                    if response.status_code == 200:
                        uploaded_count += 1
                    else:
                        errors.append(f"{Path(file_path).name}: {response.json().get('detail', '未知错误')}")
            except Exception as e:
                errors.append(f"{Path(file_path).name}: {str(e)}")
        
        result_msg = f"成功上传 {uploaded_count} 个文件"
        if errors:
            result_msg += f"\n错误: {', '.join(errors)}"
        
        # 刷新所有列表
        files_info = self.get_files_list()
        pending_files = self.get_pending_files_checkbox()
        completed_files = self.get_completed_files_checkbox()
        all_files = self.get_files_checkbox()
        
        # 上传后清除选中状态
        return result_msg, files_info, gr.update(choices=pending_files, value=[]), gr.update(choices=completed_files, value=[]), gr.update(choices=completed_files, value=[]), gr.update(choices=all_files, value=[])
    
    def get_files_list(self) -> str:
        """获取文件列表（文本格式）"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/files")
            if response.status_code == 200:
                data = response.json()
                files = data.get("files", [])
                
                if not files:
                    return "暂无文件"
                
                result = []
                for f in files:
                    status = f.get("status", "unknown")
                    size = f.get("file_size", 0) / 1024  # KB
                    result.append(
                        f"📄 {f.get('original_name', 'Unknown')} | "
                        f"{size:.1f} KB | "
                        f"状态: {status} | "
                        f"ID: {f.get('id', 'N/A')[:8]}..."
                    )
                return "\n".join(result)
            return f"获取文件列表失败: {response.status_code}"
        except Exception as e:
            return f"获取文件列表错误: {str(e)}"
    
    def get_files_dropdown(self) -> list:
        """获取所有文件列表（下拉框格式）"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/files")
            if response.status_code == 200:
                data = response.json()
                files = data.get("files", [])
                
                if not files:
                    return []
                
                result = []
                for f in files:
                    status = f.get("status", "unknown")
                    size = f.get("file_size", 0) / 1024  # KB
                    file_id = f.get('id', '')
                    original_name = f.get('original_name', 'Unknown')
                    display_text = f"{original_name} ({size:.1f} KB, {status}) | ID: {file_id}"
                    result.append(display_text)
                return result
            return []
        except Exception as e:
            print(f"获取文件下拉列表错误: {str(e)}")
            return []
    
    def get_files_checkbox(self) -> list:
        """获取所有文件列表（CheckboxGroup格式）"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/files")
            if response.status_code == 200:
                data = response.json()
                files = data.get("files", [])
                
                if not files:
                    return []
                
                result = []
                for f in files:
                    status = f.get("status", "unknown")
                    size = f.get("file_size", 0) / 1024  # KB
                    file_id = f.get('id', '')
                    original_name = f.get('original_name', 'Unknown')
                    display_text = f"{original_name} ({size:.1f} KB, {status}) | ID: {file_id}"
                    # CheckboxGroup 格式: [(label, value), ...]
                    result.append((display_text, display_text))
                return result
            return []
        except Exception as e:
            print(f"获取文件列表错误: {str(e)}")
            return []
    
    def get_pending_files_checkbox(self) -> list:
        """获取待处理文件列表（CheckboxGroup格式）- 排除已完成（有organized阶段）的文件"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/files")
            if response.status_code == 200:
                data = response.json()
                files = data.get("files", [])
                
                result = []
                for f in files:
                    status = f.get("status", "unknown")
                    stages = f.get("stages", {})
                    # 排除已完成的文件（有organized阶段输出）
                    has_organized = "organized" in stages
                    if status in ["pending", "failed"] and not has_organized:
                        size = f.get("file_size", 0) / 1024  # KB
                        file_id = f.get('id', '')
                        original_name = f.get('original_name', 'Unknown')
                        display_text = f"{original_name} ({size:.1f} KB, {status}) | ID: {file_id}"
                        # CheckboxGroup 格式: [(label, value), ...]
                        result.append((display_text, display_text))
                return result
            return []
        except Exception as e:
            print(f"获取待处理文件列表错误: {str(e)}")
            return []
    
    def get_completed_files_dropdown(self) -> list:
        """获取已完成文件列表（completed状态）- Dropdown格式"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/files")
            if response.status_code == 200:
                data = response.json()
                files = data.get("files", [])
                
                result = []
                for f in files:
                    status = f.get("status", "unknown")
                    stages = f.get("stages", {})
                    # 检查是否有所有四个阶段的输出
                    has_all_stages = all(stage in stages for stage in ["raw", "rule_denoised", "llm_denoised", "organized"])
                    if status == "completed" or has_all_stages:
                        size = f.get("file_size", 0) / 1024  # KB
                        file_id = f.get('id', '')
                        original_name = f.get('original_name', 'Unknown')
                        display_text = f"{original_name} ({size:.1f} KB, completed) | ID: {file_id}"
                        result.append(display_text)
                return result
            return []
        except Exception as e:
            print(f"获取已完成文件列表错误: {str(e)}")
            return []
    
    def get_completed_files_checkbox(self) -> list:
        """获取已完成文件列表（completed状态）- CheckboxGroup格式"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/files")
            if response.status_code == 200:
                data = response.json()
                files = data.get("files", [])
                
                result = []
                for f in files:
                    status = f.get("status", "unknown")
                    stages = f.get("stages", {})
                    # 检查是否有所有四个阶段的输出
                    has_all_stages = all(stage in stages for stage in ["raw", "rule_denoised", "llm_denoised", "organized"])
                    if status == "completed" or has_all_stages:
                        size = f.get("file_size", 0) / 1024  # KB
                        file_id = f.get('id', '')
                        original_name = f.get('original_name', 'Unknown')
                        display_text = f"{original_name} ({size:.1f} KB, completed) | ID: {file_id}"
                        # CheckboxGroup 格式: [(label, value), ...]
                        result.append((display_text, display_text))
                return result
            return []
        except Exception as e:
            print(f"获取已完成文件列表错误: {str(e)}")
            return []
    
    def extract_file_id(self, file_info: str) -> str:
        """从文件信息中提取文件ID"""
        if not file_info or "ID:" not in file_info:
            return ""
        try:
            file_id_part = file_info.split("ID:")[1].strip()
            return file_id_part
        except:
            return ""
    
    def process_files_batch(self, file_infos: List[str], stages: List[str]) -> str:
        """批量处理文件"""
        if not file_infos:
            return "请至少选择一个文件"
        
        if not stages:
            return "请至少选择一个处理阶段"
        
        results = []
        for file_info in file_infos:
            file_id = self.extract_file_id(file_info)
            if not file_id:
                results.append(f"无法解析文件ID: {file_info}")
                continue
            
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/files/{file_id}/process",
                    json={"stages": stages}
                )
                if response.status_code == 200:
                    results.append(f"✅ {file_info.split('|')[0].strip()}: 处理已启动")
                else:
                    error_msg = response.json().get('detail', '未知错误')
                    results.append(f"❌ {file_info.split('|')[0].strip()}: {error_msg}")
            except Exception as e:
                results.append(f"❌ {file_info.split('|')[0].strip()}: {str(e)}")
        
        return "\n".join(results)
    
    def get_processing_progress(self) -> Tuple[str, float, str]:
        """获取处理进度 - 包含详细进度信息"""
        try:
            # 获取文件列表统计
            response = requests.get(f"{API_BASE_URL}/api/files")
            if response.status_code == 200:
                data = response.json()
                files = data.get("files", [])
                
                # 统计各状态文件数量
                total = len(files)
                pending = sum(1 for f in files if f.get("status") == "pending")
                processing = sum(1 for f in files if f.get("status") == "processing")
                completed = sum(1 for f in files if f.get("status") == "completed")
                failed = sum(1 for f in files if f.get("status") == "failed")
                
                # 计算进度
                if total == 0:
                    progress = 0.0
                    status_text = "暂无文件"
                else:
                    progress = completed / total
                    status_text = f"总文件: {total} | 待处理: {pending} | 处理中: {processing} | 已完成: {completed} | 失败: {failed}"
                
                # 获取详细进度信息
                progress_details = []
                try:
                    progress_response = requests.get(f"{API_BASE_URL}/api/progress")
                    if progress_response.status_code == 200:
                        progress_data = progress_response.json()
                        progress_list = progress_data.get("progress", {})
                        
                        for file_id, p in progress_list.items():
                            file_name = p.get("file_name", "Unknown")
                            stage = p.get("current_stage", "unknown")
                            stage_progress = p.get("stage_progress", 0)
                            message = p.get("message", "")
                            elapsed = p.get("elapsed_time", 0)
                            
                            # 格式化显示
                            elapsed_str = f"{int(elapsed)}s" if elapsed < 60 else f"{int(elapsed/60)}m{int(elapsed%60)}s"
                            progress_bar = "█" * int(stage_progress / 10) + "░" * (10 - int(stage_progress / 10))
                            
                            progress_details.append(
                                f"📄 {file_name}\n"
                                f"   阶段: {stage} | 进度: [{progress_bar}] {stage_progress:.1f}% | 耗时: {elapsed_str}\n"
                                f"   {message}"
                            )
                except Exception as e:
                    progress_details.append(f"获取详细进度失败: {str(e)}")
                
                detail_text = "\n\n".join(progress_details) if progress_details else "无正在处理的文件"
                
                return status_text, progress, detail_text
            return "获取进度失败", 0.0, ""
        except Exception as e:
            return f"获取进度错误: {str(e)}", 0.0, ""
    
    def preview_all_stages(self, file_infos: List[str]) -> Tuple[str, str, str, str]:
        """预览文件的所有四个阶段 - 现在接收CheckboxGroup的列表"""
        if not file_infos or len(file_infos) == 0:
            return "请勾选一个文件查看", "", "", ""
        
        # 只取第一个勾选的文件
        file_info = file_infos[0]
        file_id = self.extract_file_id(file_info)
        if not file_id:
            return "无法解析文件ID", "", "", ""
        
        stages_content = []
        stages = ["raw", "rule_denoised", "llm_denoised", "organized"]
        
        for stage in stages:
            try:
                response = requests.get(
                    f"{API_BASE_URL}/api/files/{file_id}/preview",
                    params={"stage": stage, "max_length": 5000}
                )
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("content", "")
                    stages_content.append(content)
                else:
                    stages_content.append(f"[阶段 {stage} 暂无内容]")
            except Exception as e:
                stages_content.append(f"[获取阶段 {stage} 失败: {str(e)}]")
        
        return tuple(stages_content)
    
    def download_stage(self, file_info: str, stage: str) -> str:
        """下载阶段文件"""
        if not file_info:
            return "请选择文件"
        
        file_id = self.extract_file_id(file_info)
        if not file_id:
            return "无法解析文件ID"
        
        download_url = f"{API_BASE_URL}/api/files/{file_id}/download/{stage}"
        return f"下载链接: {download_url}\n\n请在浏览器中打开此链接下载文件。"
    
    def delete_file(self, file_infos: List[str]):
        """删除文件 - 支持多选，返回gr.update格式"""
        if not file_infos or len(file_infos) == 0:
            return "请至少选择一个文件", "", gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])
        
        results = []
        success_count = 0
        
        for file_info in file_infos:
            file_id = self.extract_file_id(file_info)
            if not file_id:
                results.append(f"❌ 无法解析文件ID: {file_info}")
                continue
            
            try:
                response = requests.delete(f"{API_BASE_URL}/api/files/{file_id}")
                if response.status_code == 200:
                    file_name = file_info.split('|')[0].strip()
                    results.append(f"✅ 已删除: {file_name}")
                    success_count += 1
                else:
                    error_msg = response.json().get('detail', '未知错误')
                    results.append(f"❌ 删除失败: {error_msg}")
            except Exception as e:
                results.append(f"❌ 删除错误: {str(e)}")
        
        # 刷新所有列表
        files_info = self.get_files_list()
        pending_files = self.get_pending_files_checkbox()
        completed_files = self.get_completed_files_checkbox()
        all_files = self.get_files_checkbox()
        
        result_msg = f"删除完成: {success_count}/{len(file_infos)} 个文件\n" + "\n".join(results)
        # 删除后清除所有选中状态
        return result_msg, files_info, gr.update(choices=pending_files, value=[]), gr.update(choices=completed_files, value=[]), gr.update(choices=completed_files, value=[]), gr.update(choices=all_files, value=[])
    
    def refresh_all_lists(self):
        """刷新所有列表 - 返回gr.update格式（用于刷新按钮）"""
        files_info = self.get_files_list()
        pending_files = self.get_pending_files_checkbox()
        completed_files = self.get_completed_files_checkbox()
        all_files = self.get_files_checkbox()
        # 刷新后清除选中状态
        return files_info, gr.update(choices=pending_files, value=[]), gr.update(choices=completed_files, value=[]), gr.update(choices=completed_files, value=[]), gr.update(choices=all_files, value=[])
    
    def init_all_dropdowns(self):
        """初始化所有下拉列表 - 返回gr.update格式"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/files")
            if response.status_code == 200:
                data = response.json()
                files = data.get("files", [])
                
                # 构建待处理文件列表 (CheckboxGroup 格式: [(label, value), ...])
                pending_files = []
                for f in files:
                    status = f.get("status", "unknown")
                    stages = f.get("stages", {})
                    has_organized = "organized" in stages
                    if status in ["pending", "failed"] and not has_organized:
                        size = f.get("file_size", 0) / 1024
                        file_id = f.get('id', '')
                        original_name = f.get('original_name', 'Unknown')
                        display_text = f"{original_name} ({size:.1f} KB, {status}) | ID: {file_id}"
                        pending_files.append((display_text, display_text))
                
                # 构建已完成文件列表 (CheckboxGroup 格式)
                completed_files = []
                for f in files:
                    status = f.get("status", "unknown")
                    stages = f.get("stages", {})
                    has_organized = "organized" in stages
                    if status == "completed" or has_organized:
                        size = f.get("file_size", 0) / 1024
                        file_id = f.get('id', '')
                        original_name = f.get('original_name', 'Unknown')
                        display_text = f"{original_name} ({size:.1f} KB, completed) | ID: {file_id}"
                        completed_files.append((display_text, display_text))
                
                # 构建所有文件列表 (CheckboxGroup 格式)
                all_files = []
                for f in files:
                    size = f.get("file_size", 0) / 1024
                    file_id = f.get('id', '')
                    original_name = f.get('original_name', 'Unknown')
                    status = f.get("status", "unknown")
                    display_text = f"{original_name} ({size:.1f} KB, {status}) | ID: {file_id}"
                    all_files.append((display_text, display_text))
                
                return gr.update(choices=pending_files), gr.update(choices=completed_files), gr.update(choices=completed_files), gr.update(choices=all_files)
            return gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])
        except Exception as e:
            print(f"初始化下拉列表错误: {str(e)}")
            return gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])
    
    def create_ui(self) -> gr.Blocks:
        """创建第二章的Gradio界面（文件转换与处理系统）"""
        with gr.Blocks(title="文件转换与处理系统") as app:
            gr.Markdown("""
            # 📄 文件转换与处理系统
            
            支持 PDF、Word、PPT、Excel、TXT 等格式的文件转换、去噪和内容重组。
            """)
            
            with gr.Tab("📤 文件上传"):
                with gr.Row():
                    with gr.Column(scale=2):
                        file_input = gr.File(
                            label="选择文件",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".txt", ".md"]
                        )
                    with gr.Column(scale=1):
                        upload_btn = gr.Button("📤 上传文件", variant="primary", size="lg")
                        upload_result = gr.Textbox(label="上传结果", lines=3)
            
            with gr.Tab("📋 文件列表"):
                with gr.Row():
                    refresh_btn = gr.Button("🔄 刷新列表", variant="secondary")
                
                files_list = gr.Textbox(
                    label="文件列表",
                    lines=15,
                    max_lines=20,
                    value=self.get_files_list()
                )
            
            with gr.Tab("⚙️ 文件处理"):
                gr.Markdown("### 选择要处理的文件（可多选）")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # 获取初始待处理文件列表
                        initial_pending_files = self.get_pending_files_checkbox()
                        selected_files = gr.CheckboxGroup(
                            label="待处理文件（勾选要处理的文件）",
                            choices=initial_pending_files,
                            interactive=True
                        )
                    with gr.Column(scale=1):
                        refresh_process_btn = gr.Button("🔄 刷新文件列表", variant="secondary")
                        stages_check = gr.CheckboxGroup(
                            label="处理阶段",
                            choices=[
                                ("原始文本提取", "raw"),
                                ("规则去噪", "rule_denoised"),
                                ("LLM去噪", "llm_denoised"),
                                ("内容重组", "organized")
                            ],
                            value=["raw", "rule_denoised", "llm_denoised", "organized"]
                        )
                        process_btn = gr.Button("▶️ 开始批量处理", variant="primary")
                
                with gr.Row():
                    process_result = gr.Textbox(label="处理结果", lines=5)
                
                gr.Markdown("### 处理进度")
                with gr.Row():
                    with gr.Column():
                        progress_status = gr.Textbox(label="总体状态", lines=1, interactive=False)
                        progress_bar = gr.Slider(label="总体进度", minimum=0, maximum=1, value=0, interactive=False)
                        processing_details = gr.Textbox(label="正在处理的文件", lines=5, interactive=False)
                        refresh_progress_btn = gr.Button("🔄 刷新进度", variant="secondary")
                
                # 注意：自动刷新已移除，改为手动刷新
            
            with gr.Tab("👁️ 内容预览"):
                gr.Markdown("### 选择已处理完成的文件查看各阶段结果（单选）")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # 获取初始已完成文件列表 - 使用 CheckboxGroup 单选模式
                        initial_completed_files = self.get_completed_files_checkbox()
                        preview_file_input = gr.CheckboxGroup(
                            label="选择文件（勾选一个查看）",
                            choices=initial_completed_files,
                            interactive=True
                        )
                    with gr.Column(scale=1):
                        refresh_preview_btn = gr.Button("🔄 刷新文件列表", variant="secondary")
                        preview_btn = gr.Button("👁️ 加载所有阶段", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**阶段1: 原始文本**")
                        raw_content = gr.Textbox(label="", lines=10, max_lines=15)
                    with gr.Column():
                        gr.Markdown("**阶段2: 规则去噪**")
                        rule_denoised_content = gr.Textbox(label="", lines=10, max_lines=15)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**阶段3: LLM去噪**")
                        llm_denoised_content = gr.Textbox(label="", lines=10, max_lines=15)
                    with gr.Column():
                        gr.Markdown("**阶段4: 内容重组**")
                        organized_content = gr.Textbox(label="", lines=10, max_lines=15)
            
            with gr.Tab("⬇️ 下载文件"):
                gr.Markdown("### 选择已处理完成的文件下载各阶段结果（可多选）")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # 获取初始已完成文件列表 - 使用 CheckboxGroup 多选模式
                        initial_completed_files_download = self.get_completed_files_checkbox()
                        download_file_input = gr.CheckboxGroup(
                            label="选择文件（勾选要下载的）",
                            choices=initial_completed_files_download,
                            interactive=True
                        )
                    with gr.Column(scale=1):
                        refresh_download_btn = gr.Button("🔄 刷新文件列表", variant="secondary")
                        download_all_btn = gr.Button("⬇️ 获取所有阶段下载链接", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        download_links = gr.Textbox(label="下载链接", lines=10)
            
            with gr.Tab("🗑️ 删除文件"):
                gr.Markdown("### 选择要删除的文件（可多选）")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # 获取初始所有文件列表 - 使用 CheckboxGroup 多选模式
                        initial_all_files = self.get_files_checkbox()
                        delete_file_input = gr.CheckboxGroup(
                            label="选择要删除的文件（勾选要删除的）",
                            choices=initial_all_files,
                            interactive=True
                        )
                    with gr.Column(scale=1):
                        refresh_delete_btn = gr.Button("🔄 刷新文件列表", variant="secondary")
                        delete_btn = gr.Button("🗑️ 删除文件", variant="stop")
                
                delete_result = gr.Textbox(label="删除结果", lines=3)
            
            # 事件绑定
            upload_btn.click(
                fn=self.upload_file,
                inputs=[file_input],
                outputs=[upload_result, files_list, selected_files, preview_file_input, download_file_input, delete_file_input]
            )
            
            refresh_btn.click(
                fn=self.refresh_all_lists,
                outputs=[files_list, selected_files, preview_file_input, download_file_input, delete_file_input]
            )
            
            # 文件处理页面
            def refresh_pending_files():
                """刷新待处理文件列表 - 同时清除选中状态"""
                choices = self.get_pending_files_checkbox()
                return gr.update(choices=choices, value=[])
            
            refresh_process_btn.click(
                fn=refresh_pending_files,
                outputs=[selected_files]
            )
            
            process_btn.click(
                fn=self.process_files_batch,
                inputs=[selected_files, stages_check],
                outputs=[process_result]
            )
            
            # 刷新进度按钮
            refresh_progress_btn.click(
                fn=self.get_processing_progress,
                outputs=[progress_status, progress_bar, processing_details]
            )
            
            # 内容预览页面
            def refresh_preview_files():
                """刷新预览文件列表 - 同时清除选中状态"""
                choices = self.get_completed_files_checkbox()
                return gr.update(choices=choices, value=[])
            
            refresh_preview_btn.click(
                fn=refresh_preview_files,
                outputs=[preview_file_input]
            )
            
            preview_btn.click(
                fn=self.preview_all_stages,
                inputs=[preview_file_input],
                outputs=[raw_content, rule_denoised_content, llm_denoised_content, organized_content]
            )
            
            # 下载文件页面
            def refresh_download_files():
                """刷新下载文件列表 - 同时清除选中状态"""
                choices = self.get_completed_files_checkbox()
                return gr.update(choices=choices, value=[])
            
            refresh_download_btn.click(
                fn=refresh_download_files,
                outputs=[download_file_input]
            )
            
            def get_all_download_links(file_infos: List[str]) -> str:
                """获取所有阶段的下载链接 - 支持多选"""
                if not file_infos or len(file_infos) == 0:
                    return "请至少选择一个文件"
                
                all_links = []
                stages = ["raw", "rule_denoised", "llm_denoised", "organized"]
                
                for file_info in file_infos:
                    file_id = self.extract_file_id(file_info)
                    if not file_id:
                        all_links.append(f"❌ 无法解析文件ID: {file_info}")
                        continue
                    
                    file_name = file_info.split('|')[0].strip()
                    all_links.append(f"📄 {file_name}")
                    
                    for stage in stages:
                        url = f"{API_BASE_URL}/api/files/{file_id}/download/{stage}"
                        all_links.append(f"  - 阶段 {stage}: {url}")
                    
                    all_links.append("")  # 空行分隔
                
                return "\n".join(all_links)
            
            download_all_btn.click(
                fn=get_all_download_links,
                inputs=[download_file_input],
                outputs=[download_links]
            )
            
            # 删除文件页面
            def refresh_delete_files():
                """刷新删除文件列表 - 同时清除选中状态"""
                choices = self.get_files_checkbox()
                return gr.update(choices=choices, value=[])
            
            refresh_delete_btn.click(
                fn=refresh_delete_files,
                outputs=[delete_file_input]
            )
            
            delete_btn.click(
                fn=self.delete_file,
                inputs=[delete_file_input],
                outputs=[delete_result, files_list, selected_files, preview_file_input, download_file_input, delete_file_input]
            )
            
            # 页面加载时初始化所有下拉列表和进度（只执行一次）
            def init_page():
                """初始化页面 - 返回所有需要初始化的组件"""
                # 获取下拉列表更新
                pending_choices = self.get_pending_files_checkbox()
                completed_choices = self.get_completed_files_checkbox()
                all_choices = self.get_files_checkbox()
                
                # 获取进度信息
                status_text, progress, detail_text = self.get_processing_progress()
                
                return (
                    gr.update(choices=pending_choices),  # selected_files
                    gr.update(choices=completed_choices),  # preview_file_input
                    gr.update(choices=completed_choices),  # download_file_input
                    gr.update(choices=all_choices),  # delete_file_input
                    status_text,  # progress_status
                    progress,  # progress_bar
                    detail_text  # processing_details
                )
            
            # 只在页面加载时执行一次初始化
            app.load(
                fn=init_page,
                outputs=[selected_files, preview_file_input, download_file_input, delete_file_input, 
                        progress_status, progress_bar, processing_details]
            )
        
        return app


def create_main_app():
    """创建主应用，包含章节切换功能"""
    
    # 自定义CSS样式
    custom_css = """
    .nav-bar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .nav-title {
        color: white;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .nav-buttons {
        display: flex;
        gap: 10px;
    }
    .nav-btn {
        background: rgba(255, 255, 255, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.3);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    .nav-btn:hover {
        background: rgba(255, 255, 255, 0.3);
        border-color: rgba(255, 255, 255, 0.5);
    }
    .nav-btn.active {
        background: white;
        color: #667eea;
        border-color: white;
    }
    .chapter-container {
        min-height: 600px;
    }
    """
    
    # 创建第二章UI实例
    chapter2_ui = FileConversionUI()
    chapter2_app = chapter2_ui.create_ui()
    
    # 创建第三章UI实例
    rag_page = RAGPage()
    chapter3_app = rag_page.create_ui()
    
    # 创建第四章UI实例（基于提示自动迭代的RAG问答系统）
    tourist_rag_page = TouristRAGPage()
    chapter4_app = tourist_rag_page.create_ui()
    
    # 创建主应用
    with gr.Blocks(title="智能文档处理与RAG系统", theme=purple_theme) as main_app:
        
        # 顶部导航栏
        gr.Markdown("# 📚 智能文档处理与RAG系统")
        
        with gr.Row():
            chapter2_btn = gr.Button(
                "📄 文件格式转换与去噪系统",
                variant="primary",
                size="lg"
            )
            chapter3_btn = gr.Button(
                "🔍 RAG检索增强生成系统",
                variant="secondary",
                size="lg"
            )
            chapter4_btn = gr.Button(
                "🚀 基于提示自动迭代的RAG问答系统",
                variant="secondary",
                size="lg"
            )
        
        # 章节内容容器
        with gr.Column():
            # 第二章内容（默认显示）
            chapter2_container = gr.Column(visible=True)
            with chapter2_container:
                # 直接嵌入第二章的所有组件
                gr.Markdown("""
                # 📄 文件转换与处理系统
                
                支持 PDF、Word、PPT、Excel、TXT 等格式的文件转换、去噪和内容重组。
                """)
                
                with gr.Tab("📤 文件上传"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            file_input = gr.File(
                                label="选择文件",
                                file_count="multiple",
                                file_types=[".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".txt", ".md"]
                            )
                        with gr.Column(scale=1):
                            upload_btn = gr.Button("📤 上传文件", variant="primary", size="lg")
                            upload_result = gr.Textbox(label="上传结果", lines=3)
                
                with gr.Tab("📋 文件列表"):
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 刷新列表", variant="secondary")
                    
                    files_list = gr.Textbox(
                        label="文件列表",
                        lines=15,
                        max_lines=20,
                        value=chapter2_ui.get_files_list()
                    )
                
                with gr.Tab("⚙️ 文件处理"):
                    gr.Markdown("### 选择要处理的文件（可多选）")
                    
                    with gr.Row():
                        with gr.Column(scale=3):
                            initial_pending_files = chapter2_ui.get_pending_files_checkbox()
                            selected_files = gr.CheckboxGroup(
                                label="待处理文件（勾选要处理的文件）",
                                choices=initial_pending_files,
                                interactive=True
                            )
                        with gr.Column(scale=1):
                            refresh_process_btn = gr.Button("🔄 刷新文件列表", variant="secondary")
                            stages_check = gr.CheckboxGroup(
                                label="处理阶段",
                                choices=[
                                    ("原始文本提取", "raw"),
                                    ("规则去噪", "rule_denoised"),
                                    ("LLM去噪", "llm_denoised"),
                                    ("内容重组", "organized")
                                ],
                                value=["raw", "rule_denoised", "llm_denoised", "organized"]
                            )
                            process_btn = gr.Button("▶️ 开始批量处理", variant="primary")
                    
                    with gr.Row():
                        process_result = gr.Textbox(label="处理结果", lines=5)
                    
                    gr.Markdown("### 处理进度")
                    with gr.Row():
                        with gr.Column():
                            progress_status = gr.Textbox(label="总体状态", lines=1, interactive=False)
                            progress_bar = gr.Slider(label="总体进度", minimum=0, maximum=1, value=0, interactive=False)
                            processing_details = gr.Textbox(label="正在处理的文件", lines=5, interactive=False)
                            refresh_progress_btn = gr.Button("🔄 刷新进度", variant="secondary")
                
                with gr.Tab("👁️ 内容预览"):
                    gr.Markdown("### 选择已处理完成的文件查看各阶段结果（单选）")
                    
                    with gr.Row():
                        with gr.Column(scale=3):
                            initial_completed_files = chapter2_ui.get_completed_files_checkbox()
                            preview_file_input = gr.CheckboxGroup(
                                label="选择文件（勾选一个查看）",
                                choices=initial_completed_files,
                                interactive=True
                            )
                        with gr.Column(scale=1):
                            refresh_preview_btn = gr.Button("🔄 刷新文件列表", variant="secondary")
                            preview_btn = gr.Button("👁️ 加载所有阶段", variant="primary")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**阶段1: 原始文本**")
                            raw_content = gr.Textbox(label="", lines=10, max_lines=15)
                        with gr.Column():
                            gr.Markdown("**阶段2: 规则去噪**")
                            rule_denoised_content = gr.Textbox(label="", lines=10, max_lines=15)
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**阶段3: LLM去噪**")
                            llm_denoised_content = gr.Textbox(label="", lines=10, max_lines=15)
                        with gr.Column():
                            gr.Markdown("**阶段4: 内容重组**")
                            organized_content = gr.Textbox(label="", lines=10, max_lines=15)
                
                with gr.Tab("⬇️ 下载文件"):
                    gr.Markdown("### 选择已处理完成的文件下载各阶段结果（可多选）")
                    
                    with gr.Row():
                        with gr.Column(scale=3):
                            initial_completed_files_download = chapter2_ui.get_completed_files_checkbox()
                            download_file_input = gr.CheckboxGroup(
                                label="选择文件（勾选要下载的）",
                                choices=initial_completed_files_download,
                                interactive=True
                            )
                        with gr.Column(scale=1):
                            refresh_download_btn = gr.Button("🔄 刷新文件列表", variant="secondary")
                            download_all_btn = gr.Button("⬇️ 获取所有阶段下载链接", variant="primary")
                    
                    with gr.Row():
                        with gr.Column():
                            download_links = gr.Textbox(label="下载链接", lines=10)
                
                with gr.Tab("🗑️ 删除文件"):
                    gr.Markdown("### 选择要删除的文件（可多选）")
                    
                    with gr.Row():
                        with gr.Column(scale=3):
                            initial_all_files = chapter2_ui.get_files_checkbox()
                            delete_file_input = gr.CheckboxGroup(
                                label="选择要删除的文件（勾选要删除的）",
                                choices=initial_all_files,
                                interactive=True
                            )
                        with gr.Column(scale=1):
                            refresh_delete_btn = gr.Button("🔄 刷新文件列表", variant="secondary")
                            delete_btn = gr.Button("🗑️ 删除文件", variant="stop")
                    
                    delete_result = gr.Textbox(label="删除结果", lines=3)
            
            # 第三章内容（默认隐藏）
            chapter3_container = gr.Column(visible=False)
            with chapter3_container:
                # 使用RAGPage的UI
                gr.Markdown("""
                # 🔍 RAG检索增强生成系统
                
                基于向量数据库的文档检索与智能问答系统。
                """)
                
                with gr.Tab("📚 向量库管理"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 创建新向量库")
                            collection_name = gr.Textbox(
                                label="向量库名称",
                                placeholder="输入名称..."
                            )
                            collection_desc = gr.Textbox(
                                label="描述（可选）",
                                placeholder="输入描述...",
                                lines=2
                            )
                            create_collection_btn = gr.Button("➕ 创建向量库", variant="primary")
                            create_result = gr.Textbox(label="操作结果", lines=2)
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### 管理现有向量库")
                            collections_dropdown = gr.Dropdown(
                                label="选择向量库",
                                choices=rag_page.get_collections(),
                                interactive=True
                            )
                            refresh_collections_btn = gr.Button("🔄 刷新列表")
                            collection_info = gr.Markdown()
                            delete_collection_btn = gr.Button("🗑️ 删除向量库", variant="stop")
                            delete_collection_result = gr.Textbox(label="删除结果")
                
                with gr.Tab("📄 文档向量化"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 选择目标向量库")
                            target_collection = gr.Dropdown(
                                label="目标向量库",
                                choices=rag_page.get_collections(),
                                interactive=True
                            )
                            refresh_target_collection_btn = gr.Button("🔄 刷新向量库列表")
                            
                            gr.Markdown("### 分块参数")
                            chunk_size = gr.Slider(
                                label="分块大小",
                                minimum=100,
                                maximum=2000,
                                value=500,
                                step=50
                            )
                            chunk_overlap = gr.Slider(
                                label="重叠大小",
                                minimum=0,
                                maximum=200,
                                value=50,
                                step=10
                            )
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### 选择要向量化的文档")
                            available_docs = gr.CheckboxGroup(
                                label="可用文档（已处理完成的文件）",
                                choices=rag_page.get_available_documents(),
                                interactive=True
                            )
                            refresh_docs_btn = gr.Button("🔄 刷新文档列表")
                            add_docs_btn = gr.Button("➕ 添加到向量库", variant="primary")
                            add_docs_result = gr.Textbox(label="添加结果", lines=3)
                    
                    gr.Markdown("---")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 向量库中的文档")
                            collection_docs = gr.CheckboxGroup(
                                label="已添加的文档",
                                choices=[],
                                interactive=True
                            )
                            refresh_collection_docs_btn = gr.Button("🔄 刷新文档列表")
                            delete_doc_btn = gr.Button("🗑️ 删除选中文档", variant="stop")
                            delete_doc_result = gr.Textbox(label="删除结果")
                
                with gr.Tab("🔍 相似度检索"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            search_collection = gr.Dropdown(
                                label="选择向量库",
                                choices=rag_page.get_collections(),
                                interactive=True
                            )
                            refresh_search_collection_btn = gr.Button("🔄 刷新向量库列表")
                            search_query = gr.Textbox(
                                label="检索内容",
                                placeholder="输入要检索的内容...",
                                lines=3
                            )
                            top_k = gr.Slider(
                                label="返回结果数量",
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1
                            )
                            search_btn = gr.Button("🔍 开始检索", variant="primary")
                        
                        with gr.Column(scale=2):
                            search_results = gr.Markdown(label="检索结果")
                            search_sources = gr.Textbox(
                                label="参考来源",
                                lines=3,
                                interactive=False
                            )
                
                with gr.Tab("💬 智能问答"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            chat_collection = gr.Dropdown(
                                label="选择向量库",
                                choices=rag_page.get_collections(),
                                interactive=True
                            )
                            refresh_chat_collection_btn = gr.Button("🔄 刷新向量库列表")
                            gr.Markdown("---")
                            gr.Markdown("**使用说明**：\n1. 选择向量库\n2. 在下方输入问题\n3. 系统会自动检索相关文档并生成回答")
                        
                        with gr.Column(scale=2):
                            chatbot = gr.Chatbot(
                                label="对话历史",
                                height=400
                            )
                            chat_input = gr.Textbox(
                                label="输入问题",
                                placeholder="请输入您的问题...",
                                lines=2
                            )
                            with gr.Row():
                                chat_submit_btn = gr.Button("💬 发送", variant="primary")
                                chat_clear_btn = gr.Button("🗑️ 清空对话")
            
            # 第四章内容（默认隐藏）
            chapter4_container = gr.Column(visible=False)
            with chapter4_container:
                # 使用TouristRAGPage的UI
                gr.Markdown("""
                # 🚀 基于提示自动迭代的RAG问答系统
                
                基于群智优化的RAG问答系统
                """)
                
                with gr.Tab("📊 1. 数据集展示"):
                    gr.Markdown("""
                    ### 📊 数据集展示
                    
                    从tourist数据集中加载问题、答案和应该检索的上下文。
                    """)
                    
                    load_dataset_btn = gr.Button("📂 加载数据集", variant="primary")
                    dataset_status = gr.Textbox(label="状态", lines=1)
                    
                    # 全宽表格
                    dataset_table = gr.Dataframe(
                        headers=["序号", "问题ID", "景点", "问题", "参考答案", "上下文预览"],
                        label="数据集列表",
                        interactive=True,
                        row_count=10
                    )
                    
                    # 单选框 + 按钮查看详情
                    with gr.Row():
                        with gr.Column(scale=1):
                            dataset_selected = gr.Number(label="选中行序号", value=-1, visible=False)
                            dataset_select_info = gr.Textbox(label="选择状态", value="请点击表格选择一行", interactive=False)
                            show_dataset_detail_btn = gr.Button("📋 展示问题详细信息", variant="secondary")
                        with gr.Column(scale=3):
                            dataset_detail = gr.Markdown(label="详细信息")
                    
                    dataset_state = gr.State([])
                
                with gr.Tab("📝 2. 提示初始化"):
                    gr.Markdown("""
                    ### 📝 生成初始化Prompt
                    
                    展示每个问题的初始化Prompt，包括完整的Prompt模板和分析信息。
                    """)
                    
                    load_prompts_btn = gr.Button("✨ 生成初始化Prompt", variant="primary")
                    prompts_status = gr.Textbox(label="状态", lines=1)
                    
                    # 全宽表格
                    prompts_table = gr.Dataframe(
                        headers=["序号", "问题ID", "问题", "Prompt长度", "类别"],
                        label="Prompt列表",
                        interactive=True,
                        row_count=10
                    )
                    
                    # 单选框 + 按钮查看详情
                    with gr.Row():
                        with gr.Column(scale=1):
                            prompts_selected = gr.Number(label="选中行序号", value=-1, visible=False)
                            prompts_select_info = gr.Textbox(label="选择状态", value="请点击表格选择一行", interactive=False)
                            show_prompt_detail_btn = gr.Button("📋 展示Prompt详细信息", variant="secondary")
                        with gr.Column(scale=3):
                            prompt_detail = gr.Markdown(label="Prompt详情")
                    
                    prompts_state = gr.State([])
                
                with gr.Tab("🔄 3. 迭代优化"):
                    gr.Markdown("""
                    ### 🔄 单问题迭代优化提示
                    
                    选择数据集加载迭代数据，展示每一步生成的答案、改进建议和迭代后的Prompt。
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### 选择数据集")
                            iteration_dataset_dropdown = gr.Dropdown(
                                label="数据集",
                                choices=["Tourist"],
                                value="Tourist",
                                interactive=True
                            )
                            load_iteration_dataset_btn = gr.Button("📂 加载数据集", variant="primary")
                            iteration_dataset_status = gr.Textbox(label="状态", lines=1)
                    
                    # 全宽表格显示24条数据
                    iteration_table = gr.Dataframe(
                        headers=["序号", "问题ID", "问题预览"],
                        label="迭代数据集（24条）",
                        interactive=True,
                        row_count=10
                    )
                    
                    # 单选框 + 按钮查看详情
                    with gr.Row():
                        with gr.Column(scale=1):
                            iteration_selected = gr.Number(label="选中行序号", value=-1, visible=False)
                            iteration_select_info = gr.Textbox(label="选择状态", value="请点击表格选择一行", interactive=False)
                            show_iteration_detail_btn = gr.Button("📋 展示迭代详情", variant="secondary")
                        with gr.Column(scale=3):
                            iteration_detail = gr.Markdown(label="迭代详情（含三次迭代结果）")
                    
                    iteration_state = gr.State([])
                
                with gr.Tab("🎯 4. 聚类与群智优化"):
                    gr.Markdown("""
                    ### 🎯 聚类与群智优化结果
                    
                    展示聚类后的结果、每一类的群智优化Prompt和基于的问题。
                    """)
                    
                    load_clusters_btn = gr.Button("📂 加载聚类结果", variant="primary")
                    clusters_status = gr.Textbox(label="状态", lines=1)
                    
                    # 全宽表格
                    clusters_table = gr.Dataframe(
                        headers=["Prompt ID", "聚类类别", "基于问题数", "优化原因"],
                        label="聚类列表",
                        interactive=True,
                        row_count=10
                    )
                    
                    # 单选框 + 按钮查看详情
                    with gr.Row():
                        with gr.Column(scale=1):
                            clusters_selected = gr.Number(label="选中行序号", value=-1, visible=False)
                            clusters_select_info = gr.Textbox(label="选择状态", value="请点击表格选择一行", interactive=False)
                            show_cluster_detail_btn = gr.Button("📋 展示聚类详情", variant="secondary")
                        with gr.Column(scale=3):
                            cluster_detail = gr.Markdown(label="聚类详情")
                    
                    clusters_state = gr.State([])
                
                with gr.Tab("🔍 5. Query匹配"):
                    gr.Markdown("""
                    ### 🔍 根据用户Query匹配聚类簇
                    
                    输入查询，从高到低依次展示匹配度最高的聚类簇和对应Prompt。
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=3):
                            match_query_input = gr.Textbox(
                                label="输入您的查询",
                                placeholder="例如：颐和园有哪些景点？",
                                lines=3
                            )
                        with gr.Column(scale=1):
                            match_btn = gr.Button("🔍 开始匹配", variant="primary")
                            match_status = gr.Textbox(label="状态", lines=1)
                    
                    # 全宽表格
                    matches_table = gr.Dataframe(
                        headers=["匹配度", "聚类类别", "基于问题数", "Prompt ID", "相关问题"],
                        label="匹配结果（按匹配度排序）",
                        interactive=True,
                        row_count=5
                    )
                    
                    # 单选框 + 按钮查看详情
                    with gr.Row():
                        with gr.Column(scale=1):
                            matches_selected = gr.Number(label="选中行序号", value=-1, visible=False)
                            matches_select_info = gr.Textbox(label="选择状态", value="请点击表格选择一行", interactive=False)
                            show_match_detail_btn = gr.Button("📋 展示匹配Prompt详情", variant="secondary")
                        with gr.Column(scale=3):
                            match_detail = gr.Markdown(label="Prompt详情")
                    
                    matches_state = gr.State([])
                
                with gr.Tab("💬 6. 答案生成与融合"):
                    gr.Markdown("""
                    ### 💬 答案生成与融合
                    
                    根据匹配到的两个提示和上下文检索生成答案，展示两个独立答案和最终融合答案。
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            fusion_query_input = gr.Textbox(
                                label="输入您的查询",
                                placeholder="输入您想问的问题...",
                                lines=3
                            )
                        with gr.Column(scale=2):
                            # 加载可用Prompt选项
                            fusion_prompts_dropdown = gr.Dropdown(
                                label="选择用于生成的Prompt（多选）",
                                choices=[],
                                multiselect=True,
                                interactive=True
                            )
                            refresh_prompts_btn = gr.Button("🔄 刷新Prompt列表", variant="secondary", size="sm")
                            fusion_prompts_status = gr.Textbox(label="Prompt加载状态", lines=1)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            generate_btn = gr.Button("✨ 生成答案", variant="primary")
                            fusion_status = gr.Textbox(label="生成状态", lines=1)
                    
                    with gr.Row():
                        with gr.Column():
                            answer1_output = gr.Textbox(label="答案1", lines=5)
                        with gr.Column():
                            answer2_output = gr.Textbox(label="答案2", lines=5)
                    
                    fused_answer_output = gr.Textbox(label="最终融合答案", lines=10)
        
        # ==================== 章节切换事件 ====================
        
        def switch_to_chapter2():
            """切换到第二章"""
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(variant="primary"),
                gr.update(variant="secondary"),
                gr.update(variant="secondary")
            )
        
        def switch_to_chapter3():
            """切换到第三章"""
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(variant="secondary"),
                gr.update(variant="primary"),
                gr.update(variant="secondary")
            )
        
        def switch_to_chapter4():
            """切换到第四章"""
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(variant="secondary"),
                gr.update(variant="secondary"),
                gr.update(variant="primary")
            )
        
        chapter2_btn.click(
            fn=switch_to_chapter2,
            outputs=[chapter2_container, chapter3_container, chapter4_container, chapter2_btn, chapter3_btn, chapter4_btn]
        )
        
        chapter3_btn.click(
            fn=switch_to_chapter3,
            outputs=[chapter2_container, chapter3_container, chapter4_container, chapter2_btn, chapter3_btn, chapter4_btn]
        )
        
        chapter4_btn.click(
            fn=switch_to_chapter4,
            outputs=[chapter2_container, chapter3_container, chapter4_container, chapter2_btn, chapter3_btn, chapter4_btn]
        )
        
        # ==================== 第二章事件绑定 ====================
        
        upload_btn.click(
            fn=chapter2_ui.upload_file,
            inputs=[file_input],
            outputs=[upload_result, files_list, selected_files, preview_file_input, download_file_input, delete_file_input]
        )
        
        refresh_btn.click(
            fn=chapter2_ui.refresh_all_lists,
            outputs=[files_list, selected_files, preview_file_input, download_file_input, delete_file_input]
        )
        
        def refresh_pending_files():
            choices = chapter2_ui.get_pending_files_checkbox()
            return gr.update(choices=choices, value=[])
        
        refresh_process_btn.click(
            fn=refresh_pending_files,
            outputs=[selected_files]
        )
        
        process_btn.click(
            fn=chapter2_ui.process_files_batch,
            inputs=[selected_files, stages_check],
            outputs=[process_result]
        )
        
        refresh_progress_btn.click(
            fn=chapter2_ui.get_processing_progress,
            outputs=[progress_status, progress_bar, processing_details]
        )
        
        def refresh_preview_files():
            choices = chapter2_ui.get_completed_files_checkbox()
            return gr.update(choices=choices, value=[])
        
        refresh_preview_btn.click(
            fn=refresh_preview_files,
            outputs=[preview_file_input]
        )
        
        preview_btn.click(
            fn=chapter2_ui.preview_all_stages,
            inputs=[preview_file_input],
            outputs=[raw_content, rule_denoised_content, llm_denoised_content, organized_content]
        )
        
        def refresh_download_files():
            choices = chapter2_ui.get_completed_files_checkbox()
            return gr.update(choices=choices, value=[])
        
        refresh_download_btn.click(
            fn=refresh_download_files,
            outputs=[download_file_input]
        )
        
        def get_all_download_links(file_infos: List[str]) -> str:
            if not file_infos or len(file_infos) == 0:
                return "请至少选择一个文件"
            
            all_links = []
            stages = ["raw", "rule_denoised", "llm_denoised", "organized"]
            
            for file_info in file_infos:
                file_id = chapter2_ui.extract_file_id(file_info)
                if not file_id:
                    all_links.append(f"❌ 无法解析文件ID: {file_info}")
                    continue
                
                file_name = file_info.split('|')[0].strip()
                all_links.append(f"📄 {file_name}")
                
                for stage in stages:
                    url = f"{API_BASE_URL}/api/files/{file_id}/download/{stage}"
                    all_links.append(f"  - 阶段 {stage}: {url}")
                
                all_links.append("")
            
            return "\n".join(all_links)
        
        download_all_btn.click(
            fn=get_all_download_links,
            inputs=[download_file_input],
            outputs=[download_links]
        )
        
        def refresh_delete_files():
            choices = chapter2_ui.get_files_checkbox()
            return gr.update(choices=choices, value=[])
        
        refresh_delete_btn.click(
            fn=refresh_delete_files,
            outputs=[delete_file_input]
        )
        
        delete_btn.click(
            fn=chapter2_ui.delete_file,
            inputs=[delete_file_input],
            outputs=[delete_result, files_list, selected_files, preview_file_input, download_file_input, delete_file_input]
        )
        
        # ==================== 第三章事件绑定 ====================
        
        # 向量库管理事件
        create_collection_btn.click(
            fn=rag_page.create_collection,
            inputs=[collection_name, collection_desc],
            outputs=[create_result]
        ).then(
            fn=lambda: gr.update(choices=rag_page.get_collections()),
            outputs=[collections_dropdown]
        ).then(
            fn=lambda: gr.update(choices=rag_page.get_collections()),
            outputs=[target_collection]
        ).then(
            fn=lambda: gr.update(choices=rag_page.get_collections()),
            outputs=[search_collection]
        ).then(
            fn=lambda: gr.update(choices=rag_page.get_collections()),
            outputs=[chat_collection]
        )
        
        refresh_collections_btn.click(
            fn=lambda: gr.update(choices=rag_page.get_collections()),
            outputs=[collections_dropdown]
        )
        
        collections_dropdown.change(
            fn=rag_page.get_collection_info,
            inputs=[collections_dropdown],
            outputs=[collection_info]
        )
        
        delete_collection_btn.click(
            fn=rag_page.delete_collection,
            inputs=[collections_dropdown],
            outputs=[delete_collection_result]
        ).then(
            fn=lambda: gr.update(choices=rag_page.get_collections()),
            outputs=[collections_dropdown]
        ).then(
            fn=lambda: gr.update(choices=rag_page.get_collections()),
            outputs=[target_collection]
        ).then(
            fn=lambda: gr.update(choices=rag_page.get_collections()),
            outputs=[search_collection]
        ).then(
            fn=lambda: gr.update(choices=rag_page.get_collections()),
            outputs=[chat_collection]
        )
        
        # 文档向量化事件
        refresh_target_collection_btn.click(
            fn=lambda: gr.update(choices=rag_page.get_collections()),
            outputs=[target_collection]
        )
        
        refresh_docs_btn.click(
            fn=lambda: gr.update(choices=rag_page.get_available_documents()),
            outputs=[available_docs]
        )
        
        add_docs_btn.click(
            fn=rag_page.add_documents,
            inputs=[target_collection, available_docs],
            outputs=[add_docs_result]
        )
        
        target_collection.change(
            fn=rag_page.get_collection_documents,
            inputs=[target_collection],
            outputs=[collection_docs]
        )
        
        refresh_collection_docs_btn.click(
            fn=rag_page.get_collection_documents,
            inputs=[target_collection],
            outputs=[collection_docs]
        )
        
        delete_doc_btn.click(
            fn=rag_page.delete_document_from_collection,
            inputs=[target_collection, collection_docs],
            outputs=[delete_doc_result]
        ).then(
            fn=rag_page.get_collection_documents,
            inputs=[target_collection],
            outputs=[collection_docs]
        )
        
        # 相似度检索事件
        refresh_search_collection_btn.click(
            fn=lambda: gr.update(choices=rag_page.get_collections()),
            outputs=[search_collection]
        )
        
        search_btn.click(
            fn=rag_page.search_documents,
            inputs=[search_collection, search_query, top_k],
            outputs=[search_results]
        )
        
        # 智能问答事件
        refresh_chat_collection_btn.click(
            fn=lambda: gr.update(choices=rag_page.get_collections()),
            outputs=[chat_collection]
        )
        
        chat_submit_btn.click(
            fn=rag_page.chat,
            inputs=[chat_collection, chat_input, chatbot],
            outputs=[chatbot, chat_input]
        )
        
        chat_clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, chat_input]
        )
        
        # ==================== 第四章事件绑定 ====================
        
        # Tab 1: 数据集事件
        load_dataset_btn.click(
            fn=tourist_rag_page.load_dataset,
            outputs=[dataset_status, dataset_table, dataset_state]
        )
        
        dataset_table.select(
            fn=tourist_rag_page.on_dataset_select,
            outputs=[dataset_selected, dataset_select_info]
        )
        
        show_dataset_detail_btn.click(
            fn=tourist_rag_page.show_dataset_detail,
            inputs=[dataset_selected, dataset_state],
            outputs=[dataset_detail]
        )
        
        # Tab 2: Prompt事件
        load_prompts_btn.click(
            fn=tourist_rag_page.load_initial_prompts,
            outputs=[prompts_status, prompts_table, prompts_state]
        )
        
        prompts_table.select(
            fn=tourist_rag_page.on_prompt_select,
            outputs=[prompts_selected, prompts_select_info]
        )
        
        show_prompt_detail_btn.click(
            fn=tourist_rag_page.show_prompt_detail,
            inputs=[prompts_selected, prompts_state],
            outputs=[prompt_detail]
        )
        
        # Tab 3: 迭代事件
        load_iteration_dataset_btn.click(
            fn=tourist_rag_page.load_iteration_dataset,
            outputs=[iteration_dataset_status, iteration_table, iteration_state]
        )
        
        iteration_table.select(
            fn=tourist_rag_page.on_iteration_select,
            outputs=[iteration_selected, iteration_select_info]
        )
        
        show_iteration_detail_btn.click(
            fn=tourist_rag_page.show_iteration_question_detail,
            inputs=[iteration_selected, iteration_state],
            outputs=[iteration_detail]
        )
        
        # Tab 4: 聚类事件
        load_clusters_btn.click(
            fn=tourist_rag_page.load_all_cluster_results,
            outputs=[clusters_status, clusters_table, clusters_state]
        )
        
        clusters_table.select(
            fn=tourist_rag_page.on_cluster_select,
            outputs=[clusters_selected, clusters_select_info]
        )
        
        show_cluster_detail_btn.click(
            fn=tourist_rag_page.show_cluster_detail,
            inputs=[clusters_selected, clusters_state],
            outputs=[cluster_detail]
        )
        
        # Tab 5: 匹配事件
        match_btn.click(
            fn=tourist_rag_page.match_query_to_clusters,
            inputs=[match_query_input],
            outputs=[match_status, matches_table, matches_state]
        )
        
        matches_table.select(
            fn=tourist_rag_page.on_match_select,
            outputs=[matches_selected, matches_select_info]
        )
        
        show_match_detail_btn.click(
            fn=tourist_rag_page.show_matched_prompt,
            inputs=[matches_selected, matches_state],
            outputs=[match_detail]
        )
        
        # Tab 6: 融合事件
        refresh_prompts_btn.click(
            fn=tourist_rag_page.refresh_prompts_for_fusion,
            outputs=[fusion_prompts_dropdown, fusion_prompts_status]
        )
        
        generate_btn.click(
            fn=tourist_rag_page.generate_answers_with_fusion,
            inputs=[fusion_query_input, fusion_prompts_dropdown],
            outputs=[fused_answer_output, answer1_output, answer2_output, fusion_status]
        )
        
        # ==================== 页面初始化 ====================
        
        def init_all():
            """初始化所有页面"""
            pending_choices = chapter2_ui.get_pending_files_checkbox()
            completed_choices = chapter2_ui.get_completed_files_checkbox()
            all_choices = chapter2_ui.get_files_checkbox()
            status_text, progress, detail_text = chapter2_ui.get_processing_progress()
            collections = rag_page.get_collections()
            available_documents = rag_page.get_available_documents()
            
            return (
                gr.update(choices=pending_choices),
                gr.update(choices=completed_choices),
                gr.update(choices=completed_choices),
                gr.update(choices=all_choices),
                status_text,
                progress,
                detail_text,
                gr.update(choices=collections),
                gr.update(choices=collections),
                gr.update(choices=collections),
                gr.update(choices=collections),
                gr.update(choices=available_documents)
            )
        
        main_app.load(
            fn=init_all,
            outputs=[selected_files, preview_file_input, download_file_input, delete_file_input,
                    progress_status, progress_bar, processing_details,
                    collections_dropdown, target_collection, search_collection, chat_collection, available_docs]
        )
    
    return main_app


# 创建主应用实例
app = create_main_app()

if __name__ == "__main__":
    import os
    # 禁用Gradio的更新检查和telemetry以提高启动速度
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    
    print("正在启动Gradio前端服务...")
    print(f"服务地址: http://{config.HOST}:{config.FRONTEND_PORT}")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=config.FRONTEND_PORT,
        show_error=True,
        quiet=True,
        inbrowser=False,
        share=False
    )
