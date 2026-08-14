"""
Gradio 前端应用 - 集成第二章（文件转换与处理系统）和第三章（RAG系统）
"""
import os
# 禁用Gradio的analytics以避免SSL连接问题
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TELEMETRY_ENABLED"] = "False"

import sys
import requests
import gradio as gr
from pathlib import Path
from typing import List, Tuple, Dict
import json
import time
import threading

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from pages.rag_page import RAGPage
from pages.tourist_rag_new import TouristRAGNewPage

# API基础URL (使用localhost而不是0.0.0.0)
API_BASE_URL = f"http://127.0.0.1:{config.PORT}"

# 自定义CSS让下拉列表可以滚动
CUSTOM_CSS = """
.dropdown-scroll .choices__list--dropdown {
    max-height: 300px !important;
    overflow-y: auto !important;
}
.dropdown-scroll .choices__list--dropdown .choices__list {
    max-height: 300px !important;
    overflow-y: auto !important;
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
    
    # 创建第四章UI实例（Tourist RAG）
    tourist_rag_page = TouristRAGNewPage()
    chapter4_app = tourist_rag_page.create_ui()
    
    # 创建主应用
    with gr.Blocks(title="智能文档处理与RAG系统") as main_app:
        
        # 当前章节状态（用于后端逻辑）
        current_chapter = gr.State(value=2)
        
        # 顶部导航栏
        with gr.Row(elem_classes="nav-bar"):
            with gr.Column():
                gr.Markdown("<div class='nav-title'>📚 智能文档处理与RAG系统</div>")
                
                with gr.Row():
                    chapter2_btn = gr.Button(
                        "📄 第二章：文件转换与处理系统",
                        variant="primary",
                        size="lg"
                    )
                    chapter3_btn = gr.Button(
                        "🔍 第三章：RAG检索增强生成系统",
                        variant="secondary",
                        size="lg"
                    )
                    chapter4_btn = gr.Button(
                        "🏛️ 第四章：Tourist RAG演示系统",
                        variant="secondary",
                        size="lg"
                    )
        
        # 章节内容容器
        with gr.Column(elem_classes="chapter-container"):
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
            
            # 第四章内容（Tourist RAG演示）
            chapter4_container = gr.Column(visible=False)
            with chapter4_container:
                # 直接嵌入第四章的所有组件
                gr.Markdown("""
                # 🏛️ Tourist RAG 演示系统
                
                基于群智优化的旅游问答RAG系统演示
                """)
                
                with gr.Tabs():
                    # ========== 功能1: 数据集展示 ==========
                    with gr.TabItem("📊 数据集展示"):
                        gr.Markdown("### 查看Tourist数据集中的问题、答案和上下文")
                        
                        with gr.Row():
                            refresh_dataset_btn_ch4 = gr.Button("🔄 刷新数据", variant="primary")
                        
                        dataset_table_ch4 = gr.Dataframe(
                            headers=["问题ID", "景点", "问题", "答案摘要"],
                            label="数据集列表",
                            interactive=False
                        )
                        
                        gr.Markdown("### 问题详情")
                        with gr.Row():
                            question_id_input_ch4 = gr.Textbox(label="问题ID（从上方表格复制）", placeholder="输入问题ID...")
                            load_detail_btn_ch4 = gr.Button("加载详情")
                        
                        with gr.Row():
                            detail_question_ch4 = gr.Textbox(label="问题", lines=2, interactive=False)
                            detail_attraction_ch4 = gr.Textbox(label="景点", interactive=False)
                        
                        detail_answer_ch4 = gr.Textbox(label="答案", lines=5, interactive=False)
                        detail_context_ch4 = gr.Textbox(label="上下文", lines=5, interactive=False)
                    
                    # ========== 功能2: 提示初始化生成 ==========
                    with gr.TabItem("📝 提示初始化"):
                        gr.Markdown("### 查看初始化的Prompt生成结果")
                        
                        with gr.Row():
                            refresh_prompt_init_btn_ch4 = gr.Button("🔄 刷新列表", variant="primary")
                        
                        prompt_init_table_ch4 = gr.Dataframe(
                            headers=["问题ID", "问题", "问题类型", "复杂度"],
                            label="Prompt初始化列表",
                            interactive=False
                        )
                        
                        gr.Markdown("### Prompt详情")
                        with gr.Row():
                            prompt_init_id_input_ch4 = gr.Textbox(label="问题ID", placeholder="输入问题ID...")
                            load_prompt_init_btn_ch4 = gr.Button("加载详情")
                        
                        init_question_ch4 = gr.Textbox(label="问题", lines=2, interactive=False)
                        init_full_prompt_ch4 = gr.Textbox(label="完整Prompt", lines=10, interactive=False)
                        
                        with gr.Row():
                            init_key_aspects_ch4 = gr.Textbox(label="关键方面", lines=3, interactive=False)
                            init_scene_ch4 = gr.Textbox(label="场景分析", lines=3, interactive=False)
                        
                        init_prompt_module_ch4 = gr.Textbox(label="Prompt模块", lines=5, interactive=False)
                    
                    # ========== 功能3: 单问题迭代优化 ==========
                    with gr.TabItem("🔄 迭代优化"):
                        gr.Markdown("### 查看单问题的迭代优化过程")
                        
                        iteration_question_dropdown_ch4 = gr.Dropdown(
                            choices=[],
                            label="选择问题",
                            interactive=True
                        )
                        iteration_num_ch4 = gr.Slider(minimum=0, maximum=4, step=1, value=0, label="迭代轮次")
                        
                        load_iteration_btn_ch4 = gr.Button("加载迭代详情", variant="primary")
                        
                        iteration_info_ch4 = gr.Textbox(label="迭代信息", lines=3, interactive=False)
                        iteration_answer_ch4 = gr.Textbox(label="生成的答案", lines=5, interactive=False)
                        iteration_improvements_ch4 = gr.Textbox(label="改进建议", lines=5, interactive=False)
                        iteration_prompt_ch4 = gr.Textbox(label="当前Prompt", lines=5, interactive=False)
                    
                    # ========== 功能4: 聚类与群智优化 ==========
                    with gr.TabItem("🎯 聚类展示"):
                        gr.Markdown("### 查看聚类与群智优化结果")
                        
                        with gr.Row():
                            refresh_cluster_btn_ch4 = gr.Button("🔄 刷新聚类", variant="primary")
                        
                        cluster_table_ch4 = gr.Dataframe(
                            headers=["聚类ID", "问题数量", "问题示例"],
                            label="聚类列表",
                            interactive=False
                        )
                        
                        gr.Markdown("### 聚类详情")
                        with gr.Row():
                            cluster_id_input_ch4 = gr.Textbox(label="聚类ID（如：簇 0）", placeholder="输入聚类ID...")
                            load_cluster_btn_ch4 = gr.Button("加载详情")
                        
                        cluster_stats_ch4 = gr.Textbox(label="统计信息", lines=3, interactive=False)
                        cluster_questions_ch4 = gr.Textbox(label="簇内问题", lines=10, interactive=False)
                        cluster_prompts_ch4 = gr.Textbox(label="群智优化Prompts", lines=10, interactive=False)
                    
                    # ========== 功能5: Query匹配聚类簇 ==========
                    with gr.TabItem("🔍 Query匹配"):
                        gr.Markdown("### 输入Query匹配最相似的聚类簇")
                        
                        with gr.Row():
                            query_input_ch4 = gr.Textbox(label="输入Query", placeholder="请输入您的问题...", lines=2)
                            match_btn_ch4 = gr.Button("匹配聚类", variant="primary")
                        
                        match_results_ch4 = gr.Dataframe(
                            headers=["聚类ID", "匹配度", "问题数量"],
                            label="匹配结果（按匹配度排序）",
                            interactive=False
                        )
                        
                        gr.Markdown("### 选中簇的Prompts")
                        with gr.Row():
                            selected_cluster_id_ch4 = gr.Textbox(label="选中的聚类ID", placeholder="从上方表格复制...")
                            load_cluster_prompts_btn_ch4 = gr.Button("加载Prompts")
                        
                        cluster_prompts_display_ch4 = gr.Textbox(label="簇Prompts", lines=15, interactive=False)
                    
                    # ========== 功能6: 双提示生成与答案融合 ==========
                    with gr.TabItem("🔀 答案生成与融合"):
                        gr.Markdown("### 使用两个簇的Prompt生成答案并融合")
                        
                        query_for_generation_ch4 = gr.Textbox(label="输入Query", placeholder="请输入您的问题...", lines=2)
                        
                        with gr.Row():
                            cluster1_dropdown_ch4 = gr.Dropdown(
                                choices=[f"簇 {i}" for i in range(8)],
                                label="选择第一个聚类簇"
                            )
                            cluster2_dropdown_ch4 = gr.Dropdown(
                                choices=[f"簇 {i}" for i in range(8)],
                                label="选择第二个聚类簇"
                            )
                        
                        generate_btn_ch4 = gr.Button("生成并融合答案", variant="primary")
                        
                        generation_info_ch4 = gr.Textbox(label="生成信息", interactive=False)
                        
                        with gr.Row():
                            answer1_output_ch4 = gr.Textbox(label="答案1（来自簇1）", lines=8, interactive=False)
                            answer2_output_ch4 = gr.Textbox(label="答案2（来自簇2）", lines=8, interactive=False)
                        
                        fused_answer_output_ch4 = gr.Textbox(label="融合后的最终答案", lines=10, interactive=False)
        
        # ==================== 章节切换事件 ====================
        
        def switch_to_chapter2():
            """切换到第二章"""
            return {
                chapter2_container: gr.update(visible=True),
                chapter3_container: gr.update(visible=False),
                chapter4_container: gr.update(visible=False),
                chapter2_btn: gr.update(variant="primary"),
                chapter3_btn: gr.update(variant="secondary"),
                chapter4_btn: gr.update(variant="secondary"),
                current_chapter: 2
            }
        
        def switch_to_chapter3():
            """切换到第三章"""
            return {
                chapter2_container: gr.update(visible=False),
                chapter3_container: gr.update(visible=True),
                chapter4_container: gr.update(visible=False),
                chapter2_btn: gr.update(variant="secondary"),
                chapter3_btn: gr.update(variant="primary"),
                chapter4_btn: gr.update(variant="secondary"),
                current_chapter: 3
            }
        
        def switch_to_chapter4():
            """切换到第四章"""
            return {
                chapter2_container: gr.update(visible=False),
                chapter3_container: gr.update(visible=False),
                chapter4_container: gr.update(visible=True),
                chapter2_btn: gr.update(variant="secondary"),
                chapter3_btn: gr.update(variant="secondary"),
                chapter4_btn: gr.update(variant="primary"),
                current_chapter: 4
            }
        
        chapter2_btn.click(
            fn=switch_to_chapter2,
            outputs=[chapter2_container, chapter3_container, chapter4_container, chapter2_btn, chapter3_btn, chapter4_btn, current_chapter]
        )
        
        chapter3_btn.click(
            fn=switch_to_chapter3,
            outputs=[chapter2_container, chapter3_container, chapter4_container, chapter2_btn, chapter3_btn, chapter4_btn, current_chapter]
        )
        
        chapter4_btn.click(
            fn=switch_to_chapter4,
            outputs=[chapter2_container, chapter3_container, chapter4_container, chapter2_btn, chapter3_btn, chapter4_btn, current_chapter]
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
        
        # 功能1: 数据集展示
        refresh_dataset_btn_ch4.click(
            fn=tourist_rag_page.get_dataset_list,
            outputs=[dataset_table_ch4]
        )
        
        load_detail_btn_ch4.click(
            fn=tourist_rag_page.get_question_detail,
            inputs=[question_id_input_ch4],
            outputs=[detail_question_ch4, detail_answer_ch4, detail_context_ch4, detail_attraction_ch4]
        )
        
        # 功能2: 提示初始化
        refresh_prompt_init_btn_ch4.click(
            fn=tourist_rag_page.get_prompt_init_list,
            outputs=[prompt_init_table_ch4]
        )
        
        load_prompt_init_btn_ch4.click(
            fn=tourist_rag_page.get_prompt_init_detail,
            inputs=[prompt_init_id_input_ch4],
            outputs=[init_question_ch4, init_full_prompt_ch4, init_key_aspects_ch4, init_scene_ch4, init_prompt_module_ch4]
        )
        
        # 功能3: 迭代优化
        def init_chapter4_questions():
            """初始化第四章问题列表"""
            return gr.update(choices=tourist_rag_page.get_iteration_questions())
        
        load_iteration_btn_ch4.click(
            fn=tourist_rag_page.get_iteration_detail,
            inputs=[iteration_question_dropdown_ch4, iteration_num_ch4],
            outputs=[iteration_info_ch4, iteration_answer_ch4, iteration_improvements_ch4, iteration_prompt_ch4]
        )
        
        # 功能4: 聚类展示
        refresh_cluster_btn_ch4.click(
            fn=tourist_rag_page.get_cluster_list,
            outputs=[cluster_table_ch4]
        )
        
        load_cluster_btn_ch4.click(
            fn=tourist_rag_page.get_cluster_detail,
            inputs=[cluster_id_input_ch4],
            outputs=[cluster_stats_ch4, cluster_questions_ch4, cluster_prompts_ch4]
        )
        
        # 功能5: Query匹配
        match_btn_ch4.click(
            fn=tourist_rag_page.match_query_to_clusters,
            inputs=[query_input_ch4],
            outputs=[match_results_ch4]
        )
        
        load_cluster_prompts_btn_ch4.click(
            fn=tourist_rag_page.get_cluster_prompts_for_query,
            inputs=[selected_cluster_id_ch4],
            outputs=[cluster_prompts_display_ch4]
        )
        
        # 功能6: 答案生成与融合
        generate_btn_ch4.click(
            fn=tourist_rag_page.generate_answers_with_prompts,
            inputs=[query_for_generation_ch4, cluster1_dropdown_ch4, cluster2_dropdown_ch4],
            outputs=[answer1_output_ch4, answer2_output_ch4, fused_answer_output_ch4, generation_info_ch4]
        )
        
        # ==================== 页面初始化 ====================
        
        def init_chapter2():
            """初始化第二章页面"""
            pending_choices = chapter2_ui.get_pending_files_checkbox()
            completed_choices = chapter2_ui.get_completed_files_checkbox()
            all_choices = chapter2_ui.get_files_checkbox()
            status_text, progress, detail_text = chapter2_ui.get_processing_progress()
            
            return (
                gr.update(choices=pending_choices),
                gr.update(choices=completed_choices),
                gr.update(choices=completed_choices),
                gr.update(choices=all_choices),
                status_text,
                progress,
                detail_text
            )
        
        def init_chapter3():
            """初始化第三章页面"""
            collections = rag_page.get_collections()
            available_documents = rag_page.get_available_documents()
            
            return (
                gr.update(choices=collections),
                gr.update(choices=collections),
                gr.update(choices=collections),
                gr.update(choices=collections),
                gr.update(choices=available_documents)
            )
        
        def init_chapter4():
            """初始化第四章页面"""
            iteration_questions = tourist_rag_page.get_iteration_questions()
            return gr.update(choices=iteration_questions)
        
        main_app.load(
            fn=init_chapter2,
            outputs=[selected_files, preview_file_input, download_file_input, delete_file_input,
                    progress_status, progress_bar, processing_details]
        )
        
        main_app.load(
            fn=init_chapter3,
            outputs=[collections_dropdown, target_collection, search_collection, chat_collection, available_docs]
        )
        
        main_app.load(
            fn=init_chapter4,
            outputs=[iteration_question_dropdown_ch4]
        )
    
    return main_app


# 创建主应用实例
app = create_main_app()

if __name__ == "__main__":
    print("正在启动Gradio前端服务...")
    print(f"服务地址: http://{config.HOST}:{config.FRONTEND_PORT}")
    
    app.launch(
        server_name=config.HOST,
        server_port=config.FRONTEND_PORT,
        show_error=True,
        quiet=False,  # 显示日志输出以便调试
        inbrowser=False,  # 不自动打开浏览器
        share=False  # 禁用共享链接
    )
