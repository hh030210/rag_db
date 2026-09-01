"""
Chapter4 New - Tourist RAG演示系统
集成数据集展示、Prompt管理、聚类展示、答案生成等功能
"""
import os
import sys
import json
import hashlib
import pickle
import numpy as np
import requests
import gradio as gr
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

# 路径配置
PROMPT_LIBRARY_DIR = Path("i:/bylw_final/Code/chapter3/codes/bylw_rag/new_experiments/prompt_library/tourist")
EVAL_DATA_PATH = Path("i:/bylw_final/Code/chapter3/codes/bylw_rag/new_experiments/datas/tourist/tourist_eval.json")
ITERATION_RESULTS_DIR = Path("i:/bylw_final/Code/chapter3/codes/bylw_rag/new_experiments/iteration_results/tourist")
OPTIMIZED_PROMPTS_DIR = Path("i:/bylw_final/Code/chapter3/codes/bylw_rag/new_experiments/optimized_prompts/tourist")
VECTOR_DB_PATH = Path("i:/bylw_final/Code/chapter3/codes/naive_rag/vector_dbs/tourist")

# LLM配置
API_KEY = os.getenv('SILICONFLOW_API_KEY', '')
API_URL = 'https://api.siliconflow.cn/v1/chat/completions'
MODEL_NAME = 'Qwen/Qwen3-8B'

# 全局变量
_embedding_model = None
_faiss_index = None
_faiss_metadata = None

def get_embedding_model():
    """获取Embedding模型"""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        import torch
        model_path = r"i:\bylw_final\Code\models\embedding\bge-large-zh-v1.5"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"加载Embedding模型: {model_path} (设备: {device})")
        _embedding_model = SentenceTransformer(model_path, device=device)
    return _embedding_model

def load_faiss_index():
    """加载Faiss索引"""
    global _faiss_index, _faiss_metadata
    
    if _faiss_index is None:
        import faiss
        
        index_file = VECTOR_DB_PATH / "faiss" / "index.faiss"
        metadata_file = VECTOR_DB_PATH / "faiss" / "metadata.pkl"
        
        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"向量库文件不存在: {index_file} 或 {metadata_file}")
        
        # 使用临时文件加载（避免中文路径问题）
        with tempfile.NamedTemporaryFile(delete=False, suffix='.faiss') as tmp:
            tmp_path = tmp.name
        shutil.copy2(str(index_file), tmp_path)
        _faiss_index = faiss.read_index(tmp_path)
        os.unlink(tmp_path)
        
        with open(metadata_file, 'rb') as f:
            _faiss_metadata = pickle.load(f)
        
        print(f"✓ 已加载Faiss索引: {len(_faiss_metadata)} 个文档")
    
    return _faiss_index, _faiss_metadata

def search_faiss(query: str, top_k: int = 5) -> List[Dict]:
    """使用Faiss搜索"""
    import faiss
    
    index, metadata = load_faiss_index()
    model = get_embedding_model()
    
    # 编码查询
    instruction = "为这个句子生成表示以用于检索相关文章："
    query_with_instruction = f"{instruction}{query}"
    query_embedding = model.encode([query_with_instruction], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    
    # 搜索
    distances, indices = index.search(query_embedding, top_k)
    
    # 格式化结果
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            continue
        
        meta = metadata.get(int(idx), {})
        results.append({
            'chunk_id': meta.get('chunk_id', f'chunk_{idx}'),
            'text': meta.get('text', ''),
            'score': float(dist)
        })
    
    return results

def call_llm(prompt: str, system_prompt: str = None, temperature: float = 0.3, max_tokens: int = 2000) -> str:
    """调用LLM API"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"调用LLM失败: {str(e)}"

# ==================== 数据加载函数 ====================

def load_eval_data() -> List[Dict]:
    """加载评估数据集"""
    with open(EVAL_DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_prompt_library() -> Dict[str, Dict]:
    """加载Prompt库数据"""
    prompts = {}
    if not PROMPT_LIBRARY_DIR.exists():
        return prompts
    
    for question_dir in PROMPT_LIBRARY_DIR.iterdir():
        if question_dir.is_dir():
            question_id = question_dir.name.replace('tourist_question_', '')
            iteration_0_path = question_dir / 'iteration_0.json'
            if iteration_0_path.exists():
                with open(iteration_0_path, 'r', encoding='utf-8') as f:
                    prompts[question_id] = json.load(f)
    return prompts

def load_iteration_results() -> Dict[str, Dict]:
    """加载迭代结果数据"""
    results = {}
    if not ITERATION_RESULTS_DIR.exists():
        return results
    
    for question_dir in ITERATION_RESULTS_DIR.iterdir():
        if question_dir.is_dir():
            question_id = question_dir.name.replace('tourist_question_', '')
            result_file = question_dir / 'iteration_results.json'
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    results[question_id] = json.load(f)
    return results

def load_optimized_prompts() -> Dict[int, List[Dict]]:
    """加载群智优化后的Prompts"""
    cluster_prompts = {}
    if not OPTIMIZED_PROMPTS_DIR.exists():
        return cluster_prompts
    
    for cluster_dir in OPTIMIZED_PROMPTS_DIR.iterdir():
        if cluster_dir.is_dir() and cluster_dir.name.startswith('cluster_'):
            cluster_id = int(cluster_dir.name.replace('cluster_', ''))
            cluster_prompts[cluster_id] = []
            
            for prompt_file in sorted(cluster_dir.glob('optimized_prompt_*.json')):
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    cluster_prompts[cluster_id].append(json.load(f))
    
    return cluster_prompts

def get_cluster_questions(cluster_id: int) -> List[str]:
    """获取某个聚类包含的问题ID列表"""
    cluster_dir = OPTIMIZED_PROMPTS_DIR / f'cluster_{cluster_id}'
    summary_file = cluster_dir / 'optimization_summary.json'
    
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('questions', [])
    return []

# ==================== 页面1: 数据集展示 ====================

def create_dataset_page():
    """创建数据集展示页面"""
    
    def load_dataset():
        """加载并展示数据集"""
        eval_data = load_eval_data()
        prompts = load_prompt_library()
        
        dataset_list = []
        for item in eval_data:
            question_id = item.get('question_id', '')
            question = item.get('question', '')
            answer = item.get('answer', '')
            source = item.get('source', [])
            attraction = item.get('attraction', '')
            
            # 获取对应的prompt信息
            prompt_info = prompts.get(question_id, {})
            has_prompt = "✓" if prompt_info else "✗"
            
            dataset_list.append({
                "问题ID": question_id[:16] + "..." if len(question_id) > 16 else question_id,
                "完整ID": question_id,
                "景点": attraction,
                "问题": question[:50] + "..." if len(question) > 50 else question,
                "完整问题": question,
                "参考答案": answer[:100] + "..." if len(answer) > 100 else answer,
                "完整答案": answer,
                "参考来源数": len(source),
                "完整来源": "\n\n".join(source) if source else "无",
                "已有Prompt": has_prompt
            })
        
        return dataset_list
    
    def on_select(evt: gr.SelectData):
        """当选择某行时展示详情"""
        if evt.index is not None:
            # 从索引获取行数据
            dataset_list = load_dataset()
            if evt.index[0] < len(dataset_list):
                row_data = dataset_list[evt.index[0]]
                return (
                    row_data.get("完整问题", ""),
                    row_data.get("完整答案", ""),
                    row_data.get("完整来源", ""),
                    row_data.get("完整ID", "")
                )
        return "", "", "", ""
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📊 数据集概览")
            dataset_table = gr.Dataframe(
                headers=["问题ID", "景点", "问题", "参考答案", "参考来源数", "已有Prompt"],
                label="数据集列表",
                interactive=False
            )
            refresh_btn = gr.Button("🔄 刷新数据", variant="primary")
        
        with gr.Column(scale=3):
            gr.Markdown("### 📋 详细信息")
            with gr.Tab("问题"):
                detail_question = gr.Textbox(label="问题内容", lines=3, interactive=False)
            with gr.Tab("答案"):
                detail_answer = gr.Textbox(label="参考答案", lines=8, interactive=False)
            with gr.Tab("参考来源"):
                detail_source = gr.Textbox(label="参考来源", lines=10, interactive=False)
            with gr.Tab("问题ID"):
                detail_id = gr.Textbox(label="完整问题ID", lines=2, interactive=False)
    
    # 初始加载
    dataset_table.value = load_dataset()
    
    # 事件绑定
    refresh_btn.click(fn=load_dataset, outputs=dataset_table)
    dataset_table.select(fn=on_select, outputs=[detail_question, detail_answer, detail_source, detail_id])
    
    return dataset_table

# ==================== 页面2: Prompt初始化展示 ====================

def create_prompt_init_page():
    """创建Prompt初始化展示页面"""
    
    def load_prompts():
        """加载所有初始Prompt"""
        prompts = load_prompt_library()
        prompt_list = []
        
        for question_id, data in prompts.items():
            full_prompt = data.get('full_prompt', '')
            key_aspects = data.get('key_aspects', [])
            scene_analysis = data.get('scene_analysis', {})
            
            prompt_list.append({
                "问题ID": question_id[:16] + "...",
                "完整ID": question_id,
                "问题": data.get('question', '')[:50] + "...",
                "完整问题": data.get('question', ''),
                "参考答案": data.get('ground_truth', '')[:100] + "...",
                "完整答案": data.get('ground_truth', ''),
                "关键方面数": len(key_aspects),
                "关键方面": "\n".join([f"- {k}" for k in key_aspects]),
                "场景类型": scene_analysis.get('scene_type', '未知'),
                "复杂度": scene_analysis.get('complexity', '未知'),
                "Prompt长度": len(full_prompt),
                "完整Prompt": full_prompt
            })
        
        return prompt_list
    
    def on_select(evt: gr.SelectData):
        """展示选中Prompt的详情"""
        if evt.index is not None:
            prompt_list = load_prompts()
            if evt.index[0] < len(prompt_list):
                row_data = prompt_list[evt.index[0]]
                return (
                    row_data.get("完整问题", ""),
                    row_data.get("完整答案", ""),
                    row_data.get("关键方面", ""),
                    row_data.get("场景类型", ""),
                    row_data.get("复杂度", ""),
                    row_data.get("完整Prompt", "")
                )
        return "", "", "", "", "", ""
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Prompt初始化列表")
            prompt_table = gr.Dataframe(
                headers=["问题ID", "问题", "关键方面数", "场景类型", "复杂度", "Prompt长度"],
                label="初始Prompt列表",
                interactive=False
            )
            refresh_btn = gr.Button("🔄 刷新数据", variant="primary")
        
        with gr.Column(scale=3):
            gr.Markdown("### 📋 Prompt详情")
            with gr.Tab("问题与答案"):
                prompt_question = gr.Textbox(label="问题", lines=3, interactive=False)
                prompt_answer = gr.Textbox(label="参考答案", lines=5, interactive=False)
            with gr.Tab("关键方面"):
                prompt_aspects = gr.Textbox(label="关键方面", lines=8, interactive=False)
            with gr.Tab("场景分析"):
                prompt_scene = gr.Textbox(label="场景类型", interactive=False)
                prompt_complexity = gr.Textbox(label="复杂度", interactive=False)
            with gr.Tab("完整Prompt"):
                prompt_full = gr.Textbox(label="完整Prompt", lines=20, interactive=False)
    
    prompt_table.value = load_prompts()
    refresh_btn.click(fn=load_prompts, outputs=prompt_table)
    prompt_table.select(fn=on_select, outputs=[prompt_question, prompt_answer, prompt_aspects, prompt_scene, prompt_complexity, prompt_full])
    
    return prompt_table

# ==================== 页面3: 单问题迭代优化 ====================

def create_iteration_page():
    """创建单问题迭代优化页面"""
    
    def load_iteration_data():
        """加载迭代数据"""
        results = load_iteration_results()
        iteration_list = []
        
        for question_id, data in results.items():
            iterations = data.get('iterations', [])
            
            iteration_list.append({
                "问题ID": question_id[:16] + "...",
                "完整ID": question_id,
                "问题": data.get('question', '')[:50] + "...",
                "完整问题": data.get('question', ''),
                "迭代轮数": len(iterations),
                "数据": data
            })
        
        return iteration_list
    
    def on_select(evt: gr.SelectData):
        """展示迭代详情"""
        if evt.index is not None:
            iteration_list = load_iteration_data()
            if evt.index[0] < len(iteration_list):
                row_data = iteration_list[evt.index[0]]
                data = row_data.get("数据", {})
                iterations = data.get('iterations', [])
                
                # 构建迭代详情展示
                iteration_details = []
                for i, iter_data in enumerate(iterations):
                    detail = f"""=== 第 {i+1} 轮迭代 ===
生成答案: {iter_data.get('answer', '')[:200]}...
F1分数: {iter_data.get('f1_score', 0):.4f}
改进建议: {iter_data.get('improvement_suggestions', '无')}
迭代后Prompt: {iter_data.get('iterated_prompt', '')[:300]}...
"""
                    iteration_details.append(detail)
                
                return (
                    row_data.get("完整问题", ""),
                    row_data.get("迭代轮数", 0),
                    "\n\n".join(iteration_details)
                )
        return "", 0, ""
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 🔄 迭代优化列表")
            iteration_table = gr.Dataframe(
                headers=["问题ID", "问题", "迭代轮数"],
                label="迭代优化列表",
                interactive=False
            )
            refresh_btn = gr.Button("🔄 刷新数据", variant="primary")
        
        with gr.Column(scale=3):
            gr.Markdown("### 📋 迭代详情")
            iter_question = gr.Textbox(label="问题", lines=3, interactive=False)
            iter_count = gr.Number(label="迭代轮数", interactive=False)
            iter_details = gr.Textbox(label="迭代过程", lines=25, interactive=False)
    
    iteration_table.value = load_iteration_data()
    refresh_btn.click(fn=load_iteration_data, outputs=iteration_table)
    iteration_table.select(fn=on_select, outputs=[iter_question, iter_count, iter_details])
    
    return iteration_table

# ==================== 页面4: 聚类与群智优化展示 ====================

def create_cluster_page():
    """创建聚类与群智优化展示页面"""
    
    def load_clusters():
        """加载聚类数据"""
        cluster_prompts = load_optimized_prompts()
        cluster_list = []
        
        for cluster_id, prompts in cluster_prompts.items():
            questions = get_cluster_questions(cluster_id)
            
            cluster_list.append({
                "聚类ID": f"Cluster {cluster_id}",
                "问题数量": len(questions),
                "优化Prompt数": len(prompts),
                "问题ID列表": "\n".join([q[:16] + "..." for q in questions[:5]]) + (f"\n...等共{len(questions)}个问题" if len(questions) > 5 else ""),
                "完整问题列表": questions,
                "Prompts": prompts
            })
        
        return cluster_list
    
    def on_select(evt: gr.SelectData):
        """展示聚类详情"""
        if evt.index is not None:
            cluster_list = load_clusters()
            if evt.index[0] < len(cluster_list):
                row_data = cluster_list[evt.index[0]]
                prompts = row_data.get("Prompts", [])
                questions = row_data.get("完整问题列表", [])
                
                # 构建Prompt详情
                prompt_details = []
                for i, prompt_data in enumerate(prompts[:3]):  # 只展示前3个
                    detail = f"""=== 优化Prompt {i+1} ===
基于问题数: {len(prompt_data.get('based_on_questions', []))}
优化策略: {prompt_data.get('optimization_strategy', '未知')}
Prompt内容: {prompt_data.get('full_prompt', '')[:400]}...
"""
                    prompt_details.append(detail)
                
                return (
                    f"共 {len(questions)} 个问题",
                    "\n".join([f"- {q[:30]}..." for q in questions[:10]]) + (f"\n...等共{len(questions)}个问题" if len(questions) > 10 else ""),
                    "\n\n".join(prompt_details)
                )
        return "", "", ""
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 🎯 聚类列表")
            cluster_table = gr.Dataframe(
                headers=["聚类ID", "问题数量", "优化Prompt数"],
                label="聚类列表",
                interactive=False
            )
            refresh_btn = gr.Button("🔄 刷新数据", variant="primary")
        
        with gr.Column(scale=3):
            gr.Markdown("### 📋 聚类详情")
            cluster_info = gr.Textbox(label="聚类信息", interactive=False)
            cluster_questions = gr.Textbox(label="包含的问题", lines=10, interactive=False)
            cluster_prompts_detail = gr.Textbox(label="群智优化Prompt", lines=20, interactive=False)
    
    cluster_table.value = load_clusters()
    refresh_btn.click(fn=load_clusters, outputs=cluster_table)
    cluster_table.select(fn=on_select, outputs=[cluster_info, cluster_questions, cluster_prompts_detail])
    
    return cluster_table

# ==================== 页面5: Query匹配聚类 ====================

def create_query_match_page():
    """创建Query匹配聚类页面"""
    
    def match_clusters(query: str):
        """匹配聚类簇"""
        if not query:
            return "请输入查询内容"
        
        # 加载聚类数据
        cluster_prompts = load_optimized_prompts()
        
        # 获取embedding模型
        model = get_embedding_model()
        query_embedding = model.encode([query], normalize_embeddings=True)[0]
        
        # 计算与每个聚类的相似度
        similarities = []
        for cluster_id, prompts in cluster_prompts.items():
            # 使用聚类中第一个prompt的问题来计算相似度
            if prompts and prompts[0].get('based_on_questions'):
                # 获取该聚类的问题文本
                questions = prompts[0].get('based_on_questions', [])
                if questions:
                    question_text = " ".join(questions[:3])  # 取前3个问题
                    cluster_embedding = model.encode([question_text], normalize_embeddings=True)[0]
                    similarity = np.dot(query_embedding, cluster_embedding)
                    
                    similarities.append({
                        "cluster_id": cluster_id,
                        "similarity": similarity,
                        "prompts": prompts,
                        "questions": questions
                    })
        
        # 按相似度排序
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # 构建结果展示
        results = []
        for i, sim in enumerate(similarities[:5]):  # 展示前5个
            result = f"""=== 匹配结果 {i+1} ===
聚类ID: Cluster {sim['cluster_id']}
相似度: {sim['similarity']:.4f}
基于问题数: {len(sim['questions'])}
优化Prompt示例: {sim['prompts'][0].get('full_prompt', '')[:300]}...
"""
            results.append(result)
        
        return "\n\n".join(results) if results else "未找到匹配的聚类"
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔍 Query匹配聚类")
            query_input = gr.Textbox(label="输入查询", lines=3, placeholder="请输入您想查询的问题...")
            match_btn = gr.Button("🔍 匹配聚类", variant="primary")
            match_result = gr.Textbox(label="匹配结果", lines=30, interactive=False)
    
    match_btn.click(fn=match_clusters, inputs=query_input, outputs=match_result)
    
    return query_input

# ==================== 页面6: 答案生成与融合 ====================

def create_answer_generation_page():
    """创建答案生成与融合页面"""
    
    def generate_answers(query: str):
        """生成答案"""
        if not query:
            return "请输入查询内容", "", ""
        
        # 1. 匹配Top-2聚类
        cluster_prompts = load_optimized_prompts()
        model = get_embedding_model()
        query_embedding = model.encode([query], normalize_embeddings=True)[0]
        
        similarities = []
        for cluster_id, prompts in cluster_prompts.items():
            if prompts and prompts[0].get('based_on_questions'):
                questions = prompts[0].get('based_on_questions', [])
                if questions:
                    question_text = " ".join(questions[:3])
                    cluster_embedding = model.encode([question_text], normalize_embeddings=True)[0]
                    similarity = np.dot(query_embedding, cluster_embedding)
                    similarities.append({
                        "cluster_id": cluster_id,
                        "similarity": similarity,
                        "prompt": prompts[0]
                    })
        
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        if len(similarities) < 2:
            return "未找到足够的匹配聚类", "", ""
        
        # 2. 检索上下文
        try:
            contexts = search_faiss(query, top_k=5)
            context_text = "\n\n".join([ctx['text'] for ctx in contexts])
        except Exception as e:
            context_text = f"检索失败: {str(e)}"
        
        # 3. 使用Top-2 Prompt生成答案
        answers = []
        for i in range(2):
            if i < len(similarities):
                prompt_template = similarities[i]["prompt"].get("full_prompt", "")
                # 替换变量
                filled_prompt = prompt_template.replace("{context}", context_text).replace("{question}", query)
                
                # 调用LLM
                answer = call_llm(filled_prompt)
                answers.append({
                    "cluster_id": similarities[i]["cluster_id"],
                    "similarity": similarities[i]["similarity"],
                    "answer": answer
                })
        
        # 4. 融合答案
        if len(answers) >= 2:
            fusion_prompt = f"""请融合以下两个答案，生成一个更准确、更完整的最终答案：

【问题】
{query}

【答案1】(来自Cluster {answers[0]['cluster_id']}, 相似度: {answers[0]['similarity']:.4f})
{answers[0]['answer']}

【答案2】(来自Cluster {answers[1]['cluster_id']}, 相似度: {answers[1]['similarity']:.4f})
{answers[1]['answer']}

【要求】
1. 综合两个答案的优点
2. 去除重复信息
3. 确保信息准确
4. 生成简洁明了的最终答案

请直接输出最终答案："""
            
            fused_answer = call_llm(fusion_prompt)
        else:
            fused_answer = answers[0]['answer'] if answers else "生成失败"
        
        # 构建展示结果
        answer1_text = f"""【Prompt 1】来自 Cluster {answers[0]['cluster_id']} (相似度: {answers[0]['similarity']:.4f})

{answers[0]['answer']}""" if len(answers) > 0 else "未生成"
        
        answer2_text = f"""【Prompt 2】来自 Cluster {answers[1]['cluster_id']} (相似度: {answers[1]['similarity']:.4f})

{answers[1]['answer']}""" if len(answers) > 1 else "未生成"
        
        return answer1_text, answer2_text, fused_answer
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🤖 答案生成与融合")
            gen_query = gr.Textbox(label="输入查询", lines=3, placeholder="请输入您想查询的问题...")
            gen_btn = gr.Button("🚀 生成答案", variant="primary")
    
    with gr.Row():
        with gr.Column(scale=1):
            answer1_output = gr.Textbox(label="答案1 (Prompt 1)", lines=15, interactive=False)
        with gr.Column(scale=1):
            answer2_output = gr.Textbox(label="答案2 (Prompt 2)", lines=15, interactive=False)
    
    with gr.Row():
        with gr.Column():
            fused_output = gr.Textbox(label="融合后的最终答案", lines=10, interactive=False)
    
    gen_btn.click(fn=generate_answers, inputs=gen_query, outputs=[answer1_output, answer2_output, fused_output])
    
    return gen_query

# ==================== 主应用 ====================

def create_app():
    """创建主应用"""
    
    with gr.Blocks(title="Tourist RAG演示系统") as app:
        gr.Markdown("""
# 🏛️ Tourist RAG演示系统

基于群智优化的旅游问答RAG系统演示
        """)
        
        with gr.Tabs():
            with gr.TabItem("📊 数据集展示"):
                create_dataset_page()
            
            with gr.TabItem("📝 Prompt初始化"):
                create_prompt_init_page()
            
            with gr.TabItem("🔄 迭代优化"):
                create_iteration_page()
            
            with gr.TabItem("🎯 聚类与群智优化"):
                create_cluster_page()
            
            with gr.TabItem("🔍 Query匹配聚类"):
                create_query_match_page()
            
            with gr.TabItem("🤖 答案生成与融合"):
                create_answer_generation_page()
    
    return app

if __name__ == "__main__":
    import torch
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7863, share=False)
