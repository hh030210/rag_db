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
    
    # ==================== API调用方法 ====================
    
    def get_collections(self):
        """获取所有向量库列表"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/rag/collections", timeout=5)
            if response.status_code == 200:
                data = response.json()
                collections = data.get("collections", [])
                return [c["name"] for c in collections]
            return []
        except Exception as e:
            print(f"获取向量库列表错误: {e}")
            return []
    
    def create_collection(self, name, description=""):
        """创建新向量库"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/rag/collections",
                json={"name": name, "description": description},
                timeout=10
            )
            if response.status_code == 200:
                return f"✅ 向量库 '{name}' 创建成功"
            else:
                error = response.json().get("detail", "未知错误")
                return f"❌ 创建失败: {error}"
        except Exception as e:
            return f"❌ 请求错误: {str(e)}"
    
    def delete_collection(self, collection_name):
        """删除向量库"""
        try:
            response = requests.delete(
                f"{API_BASE_URL}/api/rag/collections/{collection_name}",
                timeout=10
            )
            if response.status_code == 200:
                return f"✅ 向量库 '{collection_name}' 删除成功"
            else:
                error = response.json().get("detail", "未知错误")
                return f"❌ 删除失败: {error}"
        except Exception as e:
            return f"❌ 请求错误: {str(e)}"
    
    def get_collection_info(self, collection_name):
        """获取向量库信息"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/rag/collections/{collection_name}",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                info = data.get("collection", {})
                return f"""
**向量库名称**: {info.get('name', 'N/A')}

**描述**: {info.get('description', '无')}

**文档数量**: {info.get('document_count', 0)}

**创建时间**: {info.get('created_at', 'N/A')}
                """
            return "无法获取向量库信息"
        except Exception as e:
            return f"获取信息错误: {str(e)}"
    
    def get_available_documents(self):
        """获取可向量化的文档列表"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/files", timeout=5)
            if response.status_code == 200:
                data = response.json()
                files = data.get("files", [])
                # 只返回已处理完成的文件
                available = []
                for f in files:
                    status = f.get("status", "")
                    stages = f.get("stages", {})
                    if status == "completed" or "organized" in stages:
                        file_id = f.get('id', '')
                        original_name = f.get('original_name', 'Unknown')
                        available.append(f"{original_name} | ID: {file_id}")
                return available
            return []
        except Exception as e:
            print(f"获取文档列表错误: {e}")
            return []
    
    def add_documents(self, collection_name, file_selections):
        """添加文档到向量库"""
        if not collection_name:
            return "❌ 请先选择向量库"
        
        if not file_selections:
            return "❌ 请至少选择一个文档"
        
        try:
            # 解析文件ID
            file_ids = []
            for selection in file_selections:
                if "| ID: " in selection:
                    file_id = selection.split("| ID: ")[1].strip()
                    file_ids.append(file_id)
            
            if not file_ids:
                return "❌ 无法解析文件ID"
            
            # 连接WebSocket接收进度
            progress_messages = []
            
            def on_progress(data):
                msg = f"[{data.get('stage', 'unknown')}] {data.get('message', '')} - {data.get('progress', 0):.1f}%"
                progress_messages.append(msg)
            
            self.connect_websocket(on_progress)
            
            # 发送向量化请求
            response = requests.post(
                f"{API_BASE_URL}/api/rag/collections/{collection_name}/documents",
                json={
                    "file_ids": file_ids,
                    "client_id": self.client_id
                },
                timeout=300
            )
            
            # 断开WebSocket
            time.sleep(1)  # 等待最后一条消息
            self.disconnect_websocket()
            
            if response.status_code == 200:
                result = response.json()
                return f"✅ {result.get('message', '文档添加成功')}\n\n进度:\n" + "\n".join(progress_messages[-5:])
            else:
                error = response.json().get("detail", "未知错误")
                return f"❌ 添加失败: {error}"
                
        except Exception as e:
            self.disconnect_websocket()
            return f"❌ 请求错误: {str(e)}"
    
    def search_documents(self, collection_name, query, top_k=5):
        """搜索文档"""
        if not collection_name:
            return "❌ 请先选择向量库"
        
        if not query:
            return "❌ 请输入搜索内容"
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/rag/collections/{collection_name}/search",
                json={"query": query, "top_k": top_k},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                if not results:
                    return "未找到相关文档"
                
                output = []
                for i, result in enumerate(results, 1):
                    content = result.get("content", "")[:500]
                    score = result.get("score", 0)
                    source = result.get("source", "未知")
                    output.append(f"【结果 {i}】相似度: {score:.4f}\n来源: {source}\n内容: {content}...\n")
                
                return "\n".join(output)
            else:
                error = response.json().get("detail", "未知错误")
                return f"❌ 搜索失败: {error}"
                
        except Exception as e:
            return f"❌ 请求错误: {str(e)}"
    
    def chat(self, collection_name, message, history):
        """对话问答"""
        if not collection_name:
            return history + [[message, "❌ 请先选择向量库"]], ""
        
        if not message:
            return history, ""
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/rag/chat",
                json={
                    "collection_name": collection_name,
                    "query": message,
                    "chat_history": history[-5:] if history else []  # 只保留最近5轮
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "无法生成回答")
                return history + [[message, answer]], ""
            else:
                error = response.json().get("detail", "未知错误")
                return history + [[message, f"❌ 请求失败: {error}"]], ""
                
        except Exception as e:
            return history + [[message, f"❌ 请求错误: {str(e)}"]], ""
    
    def get_collection_documents(self, collection_name):
        """获取向量库中的文档列表"""
        if not collection_name:
            return []
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/rag/collections/{collection_name}/documents",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                docs = data.get("documents", [])
                return [f"{d.get('id', 'unknown')} | {d.get('source', '未知')[:30]}..." for d in docs]
            return []
        except Exception as e:
            print(f"获取文档列表错误: {e}")
            return []
    
    def delete_document_from_collection(self, collection_name, doc_selection):
        """从向量库中删除文档"""
        if not collection_name:
            return "❌ 请先选择向量库"
        if not doc_selection:
            return "❌ 请选择要删除的文档"
        try:
            # 解析文档ID
            doc_id = doc_selection.split(" | ")[0].strip()
            response = requests.delete(
                f"{API_BASE_URL}/api/rag/collections/{collection_name}/documents/{doc_id}",
                timeout=10
            )
            if response.status_code == 200:
                return f"✅ 文档 '{doc_id}' 删除成功"
            else:
                error = response.json().get("detail", "未知错误")
                return f"❌ 删除失败: {error}"
        except Exception as e:
            return f"❌ 请求错误: {str(e)}"
    
    # ==================== Tourist RAG 功能 ====================
    
    def load_tourist_dataset(self):
        """加载Tourist数据集"""
        try:
            eval_data_path = Path("i:/bylw_final/Code/chapter3/codes/bylw_rag/new_experiments/datas/tourist/tourist_eval.json")
            with open(eval_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            dataset_list = []
            for item in data[:50]:
                dataset_list.append({
                    "问题ID": item.get('question_id', '')[:20] + "...",
                    "景点": item.get('attraction', '未知')[:20],
                    "问题": item.get('question', '')[:40] + "...",
                    "参考答案": item.get('answer', '')[:50] + "...",
                })
            return dataset_list
        except Exception as e:
            print(f"加载数据集错误: {e}")
            return []
    
    def load_tourist_prompts(self):
        """加载Tourist Prompts"""
        try:
            prompt_dir = Path("i:/bylw_final/Code/chapter3/codes/bylw_rag/new_experiments/prompt_library/tourist")
            prompts = []
            for f in prompt_dir.glob("*_prompt.json"):
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    prompts.append({
                        "问题ID": f.stem.replace('_prompt', '')[:20] + "...",
                        "问题": data.get('question', '')[:40] + "...",
                        "Prompt长度": len(data.get('full_prompt', ''))
                    })
            return prompts
        except Exception as e:
            print(f"加载Prompts错误: {e}")
            return []
    
    def load_tourist_clusters(self):
        """加载Tourist聚类"""
        try:
            cluster_dir = Path("i:/bylw_final/Code/chapter3/codes/bylw_rag/new_experiments/optimized_prompts/tourist")
            clusters = []
            for f in cluster_dir.glob("cluster_*.json"):
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    clusters.append({
                        "聚类ID": f.stem,
                        "Prompt数量": len(data)
                    })
            return clusters
        except Exception as e:
            print(f"加载聚类错误: {e}")
            return []
    
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
                        refresh_target_collection_btn = gr.Button("🔄 刷新列表")
                        
                        gr.Markdown("### 选择要添加的文档")
                        available_docs = gr.CheckboxGroup(
                            label="可用文档",
                            choices=self.get_available_documents()
                        )
                        refresh_docs_btn = gr.Button("🔄 刷新文档列表")
                        
                        add_docs_btn = gr.Button("➕ 添加到向量库", variant="primary")
                        add_docs_result = gr.Textbox(label="添加结果", lines=10)
            
            # ==================== 检索问答标签页 ====================
            with gr.Tab("🔍 检索问答"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 搜索文档")
                        search_collection = gr.Dropdown(
                            label="选择向量库",
                            choices=self.get_collections()
                        )
                        search_query = gr.Textbox(
                            label="搜索内容",
                            placeholder="输入搜索关键词...",
                            lines=3
                        )
                        search_btn = gr.Button("🔍 搜索", variant="primary")
                        search_results = gr.Textbox(label="搜索结果", lines=20)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 智能问答")
                        chat_collection = gr.Dropdown(
                            label="选择向量库",
                            choices=self.get_collections()
                        )
                        chatbot = gr.Chatbot(label="对话历史", height=400)
                        chat_input = gr.Textbox(
                            label="输入问题",
                            placeholder="请输入您的问题...",
                            lines=2
                        )
                        with gr.Row():
                            chat_submit_btn = gr.Button("💬 发送", variant="primary")
                            chat_clear_btn = gr.Button("🗑️ 清空")
            
            # ==================== Tourist RAG演示标签页 ====================
            with gr.Tab("🏛️ Tourist RAG演示"):
                gr.Markdown("""
                ### 🏛️ Tourist RAG演示系统
                
                基于群智优化的旅游问答RAG系统演示。
                """)
                
                with gr.Tab("📊 数据集"):
                    with gr.Row():
                        dataset_table = gr.Dataframe(
                            headers=["问题ID", "景点", "问题", "参考答案"],
                            label="数据集列表"
                        )
                    refresh_dataset_btn = gr.Button("🔄 加载数据")
                
                with gr.Tab("📝 Prompts"):
                    with gr.Row():
                        prompt_table = gr.Dataframe(
                            headers=["问题ID", "问题", "Prompt长度"],
                            label="Prompt列表"
                        )
                    refresh_prompt_btn = gr.Button("🔄 加载Prompts")
                
                with gr.Tab("🎯 聚类"):
                    with gr.Row():
                        cluster_table = gr.Dataframe(
                            headers=["聚类ID", "Prompt数量"],
                            label="聚类列表"
                        )
                    refresh_cluster_btn = gr.Button("🔄 加载聚类")
            
            # ==================== 事件绑定 ====================
            
            # 向量库管理事件
            create_collection_btn.click(
                fn=self.create_collection,
                inputs=[collection_name, collection_desc],
                outputs=[create_result]
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
                fn=self.add_documents,
                inputs=[target_collection, available_docs],
                outputs=[add_docs_result]
            )
            
            # 检索问答事件
            search_btn.click(
                fn=self.search_documents,
                inputs=[search_collection, search_query],
                outputs=[search_results]
            )
            
            chat_submit_btn.click(
                fn=self.chat,
                inputs=[chat_collection, chat_input, chatbot],
                outputs=[chatbot, chat_input]
            )
            
            chat_clear_btn.click(
                fn=lambda: ([], ""),
                outputs=[chatbot, chat_input]
            )
            
            # Tourist RAG事件
            refresh_dataset_btn.click(
                fn=self.load_tourist_dataset,
                outputs=[dataset_table]
            )
            
            refresh_prompt_btn.click(
                fn=self.load_tourist_prompts,
                outputs=[prompt_table]
            )
            
            refresh_cluster_btn.click(
                fn=self.load_tourist_clusters,
                outputs=[cluster_table]
            )
        
        return rag_app
