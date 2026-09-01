"""
第三章：RAG系统页面
包含向量数据库管理、文档向量化、检索问答等功能
"""
import os
import sys
import requests
import gradio as gr
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import time
import asyncio
import threading

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config

# API基础URL
API_BASE_URL = f"http://127.0.0.1:{config.PORT}"
WS_BASE_URL = f"ws://127.0.0.1:{config.PORT}"

# 尝试导入websocket库
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("[警告] 未安装 websocket-client，实时进度功能不可用")


class RAGPage:
    """RAG系统页面类"""
    
    def __init__(self):
        self.current_collection = None
        self.chat_history = []
        self.ws_client = None
        self.progress_callback = None
        self.client_id = f"client_{int(time.time())}"
    
    # ==================== WebSocket进度管理 ====================
    
    def connect_websocket(self, on_progress=None):
        """连接WebSocket接收实时进度"""
        if not WEBSOCKET_AVAILABLE:
            print("[WebSocket] 不可用，跳过连接")
            return False
        
        def on_message(ws, message):
            """收到进度消息"""
            try:
                data = json.loads(message)
                if on_progress:
                    on_progress(data)
            except Exception as e:
                print(f"[WebSocket] 解析消息错误: {e}")
        
        def on_error(ws, error):
            # 忽略连接断开的错误（正常现象）
            if "ConnectionResetError" in str(error) or "10054" in str(error):
                return
            print(f"[WebSocket] 错误: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            # 正常关闭不打印错误
            if close_status_code is None or close_status_code == 1000:
                return
            print(f"[WebSocket] 连接关闭: {close_status_code}")
        
        def on_open(ws):
            print(f"[WebSocket] 连接成功: {self.client_id}")
        
        # 创建WebSocket连接
        ws_url = f"{WS_BASE_URL}/api/rag/ws/progress/{self.client_id}"
        print(f"[WebSocket] 正在连接: {ws_url}")
        
        self.ws_client = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # 在后台线程运行WebSocket
        def run_ws():
            self.ws_client.run_forever()
        
        threading.Thread(target=run_ws, daemon=True).start()
        return True
    
    def disconnect_websocket(self):
        """断开WebSocket连接"""
        if self.ws_client:
            self.ws_client.close()
            self.ws_client = None
            print("[WebSocket] 已断开")
    
    # ==================== 向量库管理 ====================
    
    def get_collections(self) -> List[Tuple[str, str]]:
        """获取所有向量库列表"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/rag/collections")
            if response.status_code == 200:
                data = response.json()
                collections = data.get("collections", [])
                return [(c["name"], c["id"]) for c in collections]
            return []
        except Exception as e:
            print(f"获取向量库列表错误: {str(e)}")
            return []
    
    def create_collection(self, name: str, description: str = "") -> str:
        """创建新的向量库"""
        if not name:
            return "请输入向量库名称"
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/rag/collections",
                json={"name": name, "description": description}
            )
            if response.status_code == 200:
                return f"✅ 向量库 '{name}' 创建成功"
            else:
                error = response.json().get("detail", "未知错误")
                return f"❌ 创建失败: {error}"
        except Exception as e:
            return f"❌ 创建错误: {str(e)}"
    
    def delete_collection(self, collection_id: str) -> str:
        """删除向量库"""
        if not collection_id:
            return "请选择要删除的向量库"
        
        try:
            response = requests.delete(
                f"{API_BASE_URL}/api/rag/collections/{collection_id}"
            )
            if response.status_code == 200:
                return "✅ 向量库已删除"
            else:
                error = response.json().get("detail", "未知错误")
                return f"❌ 删除失败: {error}"
        except Exception as e:
            return f"❌ 删除错误: {str(e)}"
    
    def get_collection_info(self, collection_id: str) -> str:
        """获取向量库详细信息"""
        if not collection_id:
            return "请选择向量库"
        
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/rag/collections/{collection_id}"
            )
            if response.status_code == 200:
                data = response.json()
                info = f"""
**向量库名称**: {data.get('name', 'N/A')}

**描述**: {data.get('description', '无')}

**文档数量**: {data.get('document_count', 0)}

**向量维度**: {data.get('dimension', 'N/A')}

**创建时间**: {data.get('created_at', 'N/A')}

**最后更新**: {data.get('updated_at', 'N/A')}
                """
                return info
            else:
                error = response.json().get("detail", "未知错误")
                return f"获取信息失败: {error}"
        except Exception as e:
            return f"获取信息错误: {str(e)}"
    
    # ==================== 文档向量化 ====================
    
    def get_available_documents(self) -> List[Tuple[str, str]]:
        """获取可用于向量化的文档列表（已处理完成的文件）"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/files")
            if response.status_code == 200:
                data = response.json()
                files = data.get("files", [])
                
                result = []
                for f in files:
                    status = f.get("status", "unknown")
                    stages = f.get("stages", {})
                    # 只显示已完成的文件
                    has_organized = "organized" in stages
                    if status == "completed" or has_organized:
                        file_id = f.get('id', '')
                        original_name = f.get('original_name', 'Unknown')
                        display_text = f"{original_name} | ID: {file_id}"
                        result.append((display_text, file_id))
                return result
            return []
        except Exception as e:
            print(f"获取文档列表错误: {str(e)}")
            return []
    
    def add_documents_to_collection(self, collection_id: str, file_ids: List[str], 
                                     chunk_size: int = 500, chunk_overlap: int = 50):
        """将文档添加到向量库"""
        if not collection_id:
            yield "请选择目标向量库", "等待开始..."
            return
        
        if not file_ids:
            yield "请至少选择一个文档", "等待开始..."
            return
        
        total_files = len(file_ids)
        yield "正在向量化...", f"0/{total_files} - 处理中，请查看后端日志..."
        
        try:
            # 使用长超时时间（无限制）
            response = requests.post(
                f"{API_BASE_URL}/api/rag/collections/{collection_id}/documents",
                json={
                    "file_ids": file_ids,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                },
                timeout=None  # 无超时限制
            )
            
            if response.status_code == 200:
                data = response.json()
                added = data.get('added_count', 0)
                chunks = data.get('chunk_count', 0)
                result_msg = f"✅ 成功添加 {added} 个文档，共 {chunks} 个文本块"
                progress_msg = f"{added}/{total_files} - 完成!"
                yield result_msg, progress_msg
            else:
                error = response.json().get("detail", "未知错误")
                yield f"❌ 添加失败: {error}", f"0/{total_files} - 失败"
        except Exception as e:
            yield f"❌ 添加错误: {str(e)}", f"0/{total_files} - 错误"
    
    def get_collection_documents(self, collection_id: str) -> List[Tuple[str, str]]:
        """获取向量库中的文档列表"""
        if not collection_id:
            return []
        
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/rag/collections/{collection_id}/documents"
            )
            if response.status_code == 200:
                data = response.json()
                documents = data.get("documents", [])
                return [
                    (f"{d.get('file_name', 'Unknown')} ({d.get('chunk_count', 0)} chunks)", d.get('id'))
                    for d in documents
                ]
            return []
        except Exception as e:
            print(f"获取文档列表错误: {str(e)}")
            return []
    
    def delete_document_from_collection(self, collection_id: str, document_id: str) -> str:
        """从向量库中删除文档"""
        if not collection_id or not document_id:
            return "请选择要删除的文档"
        
        try:
            response = requests.delete(
                f"{API_BASE_URL}/api/rag/collections/{collection_id}/documents/{document_id}"
            )
            if response.status_code == 200:
                return "✅ 文档已从向量库中删除"
            else:
                error = response.json().get("detail", "未知错误")
                return f"❌ 删除失败: {error}"
        except Exception as e:
            return f"❌ 删除错误: {str(e)}"
    
    # ==================== 检索问答 ====================
    
    def search_similar(self, collection_id: str, query: str, top_k: int = 5) -> Tuple[str, str]:
        """相似度检索"""
        if not collection_id:
            return "请选择向量库", ""
        
        if not query:
            return "请输入查询内容", ""
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/rag/collections/{collection_id}/search",
                json={"query": query, "top_k": top_k}
            )
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                # 格式化检索结果
                formatted_results = []
                sources = []
                
                for i, result in enumerate(results, 1):
                    content = result.get("content", "")
                    score = result.get("score", 0)
                    source = result.get("source", "Unknown")
                    
                    formatted_results.append(
                        f"**结果 {i}** (相似度: {score:.3f})\n\n{content}\n\n---\n"
                    )
                    sources.append(source)
                
                results_text = "\n".join(formatted_results) if formatted_results else "未找到相关结果"
                sources_text = "\n".join(set(sources)) if sources else ""
                
                return results_text, sources_text
            else:
                error = response.json().get("detail", "未知错误")
                return f"检索失败: {error}", ""
        except Exception as e:
            return f"检索错误: {str(e)}", ""
    
    def chat_with_rag(self, collection_id: str, query: str, chat_history: List) -> Tuple[List, str]:
        """RAG问答对话"""
        if not collection_id:
            return chat_history + [[query, "请先选择向量库"]], ""
        
        if not query:
            return chat_history, ""
        
        try:
            # 先检索相关文档
            search_response = requests.post(
                f"{API_BASE_URL}/api/rag/collections/{collection_id}/search",
                json={"query": query, "top_k": 5}
            )
            
            context = ""
            sources = []
            if search_response.status_code == 200:
                search_data = search_response.json()
                results = search_data.get("results", [])
                context = "\n\n".join([r.get("content", "") for r in results])
                sources = list(set([r.get("source", "Unknown") for r in results]))
            
            # 调用对话API
            response = requests.post(
                f"{API_BASE_URL}/api/rag/chat",
                json={
                    "query": query,
                    "context": context,
                    "chat_history": chat_history[-5:] if len(chat_history) > 5 else chat_history  # 只保留最近5轮
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                
                # 添加来源信息
                if sources:
                    answer += "\n\n**参考来源**:\n" + "\n".join([f"- {s}" for s in sources])
                
                chat_history.append([query, answer])
                return chat_history, ""
            else:
                error = response.json().get("detail", "未知错误")
                chat_history.append([query, f"回答生成失败: {error}"])
                return chat_history, ""
        except Exception as e:
            chat_history.append([query, f"对话错误: {str(e)}"])
            return chat_history, ""
    
    def clear_chat(self) -> Tuple[List, str]:
        """清空对话历史"""
        return [], ""
    
    # ==================== 创建UI ====================
    
    def create_ui(self) -> gr.Blocks:
        """创建RAG系统的Gradio界面"""
        
        with gr.Blocks(title="RAG检索增强生成系统") as rag_app:
            gr.Markdown("""
            # 🔍 RAG检索增强生成系统
            
            基于向量数据库的文档检索与智能问答系统。
            """)
            
            # ==================== 向量库管理标签页 ====================
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
                            choices=self.get_collections(),
                            interactive=True
                        )
                        refresh_collections_btn = gr.Button("🔄 刷新列表")
                        collection_info = gr.Markdown(label="向量库信息")
                        delete_collection_btn = gr.Button("🗑️ 删除向量库", variant="stop")
                        delete_collection_result = gr.Textbox(label="删除结果")
            
            # ==================== 文档向量化标签页 ====================
            with gr.Tab("📄 文档向量化"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 选择目标向量库")
                        target_collection = gr.Dropdown(
                            label="目标向量库",
                            choices=self.get_collections(),
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
                            choices=self.get_available_documents(),
                            interactive=True
                        )
                        refresh_docs_btn = gr.Button("🔄 刷新文档列表")
                        add_docs_btn = gr.Button("➕ 添加到向量库", variant="primary")
                        add_docs_progress = gr.Textbox(
                            label="向量化进度",
                            value="等待开始...",
                            interactive=False
                        )
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
            
            # ==================== 相似度检索标签页 ====================
            with gr.Tab("🔍 相似度检索"):
                with gr.Row():
                    with gr.Column(scale=1):
                        search_collection = gr.Dropdown(
                            label="选择向量库",
                            choices=self.get_collections(),
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
            
            # ==================== 智能问答标签页 ====================
            with gr.Tab("💬 智能问答"):
                with gr.Row():
                    with gr.Column(scale=1):
                        chat_collection = gr.Dropdown(
                            label="选择向量库",
                            choices=self.get_collections(),
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
            
            # ==================== 事件绑定 ====================
            
            # 向量库管理事件
            create_collection_btn.click(
                fn=self.create_collection,
                inputs=[collection_name, collection_desc],
                outputs=[create_result]
            ).then(
                fn=lambda: gr.update(choices=self.get_collections()),
                outputs=[collections_dropdown]
            ).then(
                fn=lambda: gr.update(choices=self.get_collections()),
                outputs=[target_collection]
            ).then(
                fn=lambda: gr.update(choices=self.get_collections()),
                outputs=[search_collection]
            ).then(
                fn=lambda: gr.update(choices=self.get_collections()),
                outputs=[chat_collection]
            )
            
            refresh_collections_btn.click(
                fn=lambda: gr.update(choices=self.get_collections()),
                outputs=[collections_dropdown]
            )
            
            collections_dropdown.change(
                fn=self.get_collection_info,
                inputs=[collections_dropdown],
                outputs=[collection_info]
            )
            
            delete_collection_btn.click(
                fn=self.delete_collection,
                inputs=[collections_dropdown],
                outputs=[delete_collection_result]
            ).then(
                fn=lambda: gr.update(choices=self.get_collections()),
                outputs=[collections_dropdown]
            ).then(
                fn=lambda: gr.update(choices=self.get_collections()),
                outputs=[target_collection]
            ).then(
                fn=lambda: gr.update(choices=self.get_collections()),
                outputs=[search_collection]
            ).then(
                fn=lambda: gr.update(choices=self.get_collections()),
                outputs=[chat_collection]
            )
            
            # 文档向量化事件
            refresh_target_collection_btn.click(
                fn=lambda: gr.update(choices=self.get_collections()),
                outputs=[target_collection]
            )
            
            refresh_docs_btn.click(
                fn=lambda: gr.update(choices=self.get_available_documents()),
                outputs=[available_docs]
            )
            
            add_docs_btn.click(
                fn=self.add_documents_to_collection,
                inputs=[target_collection, available_docs, chunk_size, chunk_overlap],
                outputs=[add_docs_result, add_docs_progress]
            )
            
            target_collection.change(
                fn=self.get_collection_documents,
                inputs=[target_collection],
                outputs=[collection_docs]
            )
            
            refresh_collection_docs_btn.click(
                fn=self.get_collection_documents,
                inputs=[target_collection],
                outputs=[collection_docs]
            )
            
            delete_doc_btn.click(
                fn=self.delete_document_from_collection,
                inputs=[target_collection, collection_docs],
                outputs=[delete_doc_result]
            ).then(
                fn=self.get_collection_documents,
                inputs=[target_collection],
                outputs=[collection_docs]
            )
            
            # 相似度检索事件
            refresh_search_collection_btn.click(
                fn=lambda: gr.update(choices=self.get_collections()),
                outputs=[search_collection]
            )
            
            search_btn.click(
                fn=self.search_similar,
                inputs=[search_collection, search_query, top_k],
                outputs=[search_results, search_sources]
            )
            
            # 智能问答事件
            refresh_chat_collection_btn.click(
                fn=lambda: gr.update(choices=self.get_collections()),
                outputs=[chat_collection]
            )
            
            chat_submit_btn.click(
                fn=self.chat_with_rag,
                inputs=[chat_collection, chat_input, chatbot],
                outputs=[chatbot, chat_input]
            )
            
            chat_clear_btn.click(
                fn=self.clear_chat,
                outputs=[chatbot, chat_input]
            )
            
            # 页面加载时初始化所有下拉列表
            rag_app.load(
                fn=lambda: (
                    gr.update(choices=self.get_collections()),
                    gr.update(choices=self.get_collections()),
                    gr.update(choices=self.get_collections()),
                    gr.update(choices=self.get_collections()),
                    gr.update(choices=self.get_available_documents())
                ),
                outputs=[
                    collections_dropdown, target_collection, 
                    search_collection, chat_collection, available_docs
                ]
            )
        
        return rag_app
