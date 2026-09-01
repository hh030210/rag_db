"""
Tourist RAG 演示系统 - 完整功能实现
包含6个功能页面：
1. 数据集展示
2. 提示初始化生成
3. 单问题迭代优化
4. 聚类与群智优化
5. Query匹配聚类簇
6. 双提示生成与答案融合
"""
import os
import sys
import json
import requests
import gradio as gr
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config

# API基础URL
API_BASE_URL = f"http://127.0.0.1:{config.PORT}"

# 数据路径配置
BASE_DATA_PATH = Path("i:/bylw_final/Code/chapter3/codes/bylw_rag/new_experiments")
TOURIST_EVAL_PATH = BASE_DATA_PATH / "datas/tourist/tourist_eval.json"
PROMPT_LIBRARY_PATH = BASE_DATA_PATH / "prompt_library/tourist"
CLUSTERING_RESULTS_PATH = BASE_DATA_PATH / "clustering_results/tourist"


class TouristRAGNewPage:
    """Tourist RAG 演示页面 - 新功能"""
    
    def __init__(self):
        self.eval_data = None
        self.prompt_data = {}
        self.cluster_data = None
        self._load_all_data()
    
    def _load_all_data(self):
        """预加载所有数据"""
        self._load_eval_data()
        self._load_prompt_data()
        self._load_cluster_data()
    
    def _load_eval_data(self):
        """加载评估数据集"""
        try:
            with open(TOURIST_EVAL_PATH, 'r', encoding='utf-8') as f:
                self.eval_data = json.load(f)
            print(f"[OK] 加载了 {len(self.eval_data)} 条评估数据")
        except Exception as e:
            print(f"[Error] 加载评估数据失败: {e}")
            self.eval_data = []
    
    def _load_prompt_data(self):
        """加载所有prompt数据"""
        try:
            if not PROMPT_LIBRARY_PATH.exists():
                print(f"❌ Prompt库路径不存在: {PROMPT_LIBRARY_PATH}")
                return
            
            for folder in PROMPT_LIBRARY_PATH.iterdir():
                if folder.is_dir():
                    question_id = folder.name.replace('tourist_question_', '')
                    iteration_files = list(folder.glob("iteration_*.json"))
                    if iteration_files:
                        # 加载所有迭代数据
                        iterations = []
                        for f in sorted(iteration_files):
                            with open(f, 'r', encoding='utf-8') as fp:
                                iterations.append(json.load(fp))
                        self.prompt_data[question_id] = iterations
            
            print(f"[OK] 加载了 {len(self.prompt_data)} 个问题的prompt数据")
        except Exception as e:
            print(f"[Error] 加载prompt数据失败: {e}")
    
    def _load_cluster_data(self):
        """加载聚类数据"""
        try:
            cluster_file = CLUSTERING_RESULTS_PATH / "cluster_results.json"
            if cluster_file.exists():
                with open(cluster_file, 'r', encoding='utf-8') as f:
                    self.cluster_data = json.load(f)
                print(f"[OK] 加载了聚类数据: {self.cluster_data.get('n_clusters', 0)} 个簇")
            else:
                print(f"[Error] 聚类结果文件不存在: {cluster_file}")
        except Exception as e:
            print(f"[Error] 加载聚类数据失败: {e}")
            self.cluster_data = None
    
    # ==================== 功能1: 数据集展示 ====================
    def get_dataset_list(self) -> List[Dict]:
        """获取数据集列表"""
        if not self.eval_data:
            return []
        
        result = []
        for item in self.eval_data[:50]:  # 限制显示50条
            result.append({
                "问题ID": item.get('question_id', '')[:20] + "...",
                "景点": item.get('attraction', '未知')[:15],
                "问题": item.get('question', '')[:40] + "..." if len(item.get('question', '')) > 40 else item.get('question', ''),
                "答案摘要": item.get('answer', '')[:50] + "..." if len(item.get('answer', '')) > 50 else item.get('answer', '')
            })
        return result
    
    def get_question_detail(self, question_id_short: str) -> Tuple[str, str, str, str]:
        """获取问题详情"""
        # 从短ID找到完整数据
        question_id = None
        for item in self.eval_data:
            if item.get('question_id', '').startswith(question_id_short.replace('...', '')):
                question_id = item.get('question_id')
                break
        
        if not question_id:
            return "未找到", "未找到", "未找到", "未找到"
        
        for item in self.eval_data:
            if item.get('question_id') == question_id:
                question = item.get('question', '')
                answer = item.get('answer', '')
                context = item.get('document', '')
                attraction = item.get('attraction', '未知')
                return question, answer, context, attraction
        
        return "未找到", "未找到", "未找到", "未找到"
    
    # ==================== 功能2: 提示初始化生成 ====================
    def get_prompt_init_list(self) -> List[Dict]:
        """获取提示初始化列表"""
        result = []
        for qid, iterations in list(self.prompt_data.items())[:30]:
            if iterations:
                init_data = iterations[0]  # 取iteration_0
                result.append({
                    "问题ID": qid[:20] + "...",
                    "问题": init_data.get('question', '')[:40] + "..." if len(init_data.get('question', '')) > 40 else init_data.get('question', ''),
                    "问题类型": init_data.get('question_analysis', {}).get('question_category', '未知'),
                    "复杂度": init_data.get('question_analysis', {}).get('complexity', '未知')
                })
        return result
    
    def get_prompt_init_detail(self, question_id_short: str) -> Tuple[str, str, str, str, str]:
        """获取提示初始化详情"""
        question_id = self._find_full_question_id(question_id_short)
        if not question_id or question_id not in self.prompt_data:
            return "未找到", "", "", "", ""
        
        init_data = self.prompt_data[question_id][0]
        question = init_data.get('question', '')
        full_prompt = init_data.get('full_prompt', '')
        
        # 提取key_aspects
        key_aspects = init_data.get('question_analysis', {}).get('key_aspects', [])
        key_aspects_str = ', '.join(key_aspects) if key_aspects else '无'
        
        # 提取scene_analysis
        scene = init_data.get('scene_analysis', {})
        scene_str = f"领域: {scene.get('domain', '未知')}\n语言: {scene.get('language', '未知')}\n问题类型: {', '.join(scene.get('question_types', []))}"
        
        # 提取prompt模块
        prompt_module = init_data.get('prompt_module', {})
        prompt_module_str = f"系统提示: {prompt_module.get('P_sys', '')[:200]}..."
        
        return question, full_prompt, key_aspects_str, scene_str, prompt_module_str
    
    # ==================== 功能3: 单问题迭代优化 ====================
    def get_iteration_questions(self) -> List[str]:
        """获取有迭代数据的问题列表"""
        return [f"{qid[:20]}... | {self._get_question_text(qid)[:30]}..." 
                for qid in list(self.prompt_data.keys())[:20]]
    
    def _get_question_text(self, question_id: str) -> str:
        """获取问题文本"""
        if question_id in self.prompt_data and self.prompt_data[question_id]:
            return self.prompt_data[question_id][0].get('question', '未知')
        return '未知'
    
    def _find_full_question_id(self, question_id_short: str) -> Optional[str]:
        """从短ID找到完整ID"""
        short = question_id_short.replace('...', '').strip()
        for qid in self.prompt_data.keys():
            if qid.startswith(short):
                return qid
        return None
    
    def get_iteration_detail(self, question_selection: str, iteration_num: int) -> Tuple[str, str, str, str]:
        """获取迭代详情"""
        if not question_selection:
            return "请选择问题", "", "", ""
        
        question_id_short = question_selection.split(" | ")[0].strip()
        question_id = self._find_full_question_id(question_id_short)
        
        if not question_id or question_id not in self.prompt_data:
            return "未找到数据", "", "", ""
        
        iterations = self.prompt_data[question_id]
        if iteration_num >= len(iterations):
            return "迭代不存在", "", "", ""
        
        data = iterations[iteration_num]
        
        # 获取答案
        answer = data.get('answer', '无答案')
        
        # 获取改进建议
        improvements = data.get('improvement_suggestions', [])
        improvements_str = '\n'.join([f"{i+1}. {imp}" for i, imp in enumerate(improvements)]) if improvements else '无改进建议'
        
        # 获取当前prompt
        current_prompt = data.get('full_prompt', '')[:500] + "..." if len(data.get('full_prompt', '')) > 500 else data.get('full_prompt', '')
        
        # 获取迭代信息
        iteration_info = f"当前迭代: {data.get('iteration', iteration_num)}\n问题: {data.get('question', '')}"
        
        return iteration_info, answer, improvements_str, current_prompt
    
    # ==================== 功能4: 聚类与群智优化 ====================
    def get_cluster_list(self) -> List[Dict]:
        """获取聚类列表"""
        if not self.cluster_data:
            return []
        
        result = []
        clusters = self.cluster_data.get('clusters', [])
        for cluster in clusters:
            result.append({
                "聚类ID": f"簇 {cluster.get('cluster_id', 0)}",
                "问题数量": cluster.get('size', 0),
                "问题示例": self._get_cluster_sample_questions(cluster)[:50] + "..."
            })
        return result
    
    def _get_cluster_sample_questions(self, cluster: Dict) -> str:
        """获取聚类中的示例问题"""
        questions = cluster.get('questions', [])
        if not questions:
            return "无问题"
        # 返回前2个问题
        sample = [q.get('question', '')[:20] for q in questions[:2]]
        return ', '.join(sample)
    
    def get_cluster_detail(self, cluster_id_str: str) -> Tuple[str, str, str]:
        """获取聚类详情"""
        if not cluster_id_str or not self.cluster_data:
            return "", "", ""
        
        # 解析cluster_id
        try:
            cluster_id = int(cluster_id_str.replace("簇 ", "").strip())
        except:
            return "无效的簇ID", "", ""
        
        clusters = self.cluster_data.get('clusters', [])
        cluster = None
        for c in clusters:
            if c.get('cluster_id') == cluster_id:
                cluster = c
                break
        
        if not cluster:
            return "未找到簇", "", ""
        
        # 获取该簇的所有问题
        questions = cluster.get('questions', [])
        questions_str = "\n\n".join([
            f"问题{i+1}: {q.get('question', '')}\n答案: {q.get('answer', '')[:100]}..."
            for i, q in enumerate(questions[:5])  # 只显示前5个
        ])
        
        # 获取群智优化后的prompts
        prompts_str = ""
        for q in questions[:3]:  # 显示前3个问题的prompt
            if 'final_prompt' in q and 'full_prompt' in q['final_prompt']:
                prompts_str += f"\n{'='*50}\n问题: {q.get('question', '')[:30]}...\n优化后Prompt:\n{q['final_prompt']['full_prompt'][:300]}...\n"
        
        # 统计信息
        stats = f"簇ID: {cluster_id}\n问题数量: {len(questions)}\n"
        
        return stats, questions_str, prompts_str
    
    # ==================== 功能5: Query匹配聚类簇 ====================
    def match_query_to_clusters(self, query: str) -> List[Dict]:
        """根据Query匹配聚类簇"""
        if not query or not self.cluster_data:
            return []
        
        clusters = self.cluster_data.get('clusters', [])
        results = []
        
        # 简单的关键词匹配计算相似度
        query_keywords = set(query.lower().split())
        
        for cluster in clusters:
            questions = cluster.get('questions', [])
            if not questions:
                continue
            
            # 计算与簇中问题的平均相似度
            total_sim = 0
            for q in questions[:5]:  # 只取前5个问题计算
                q_text = q.get('question', '').lower()
                q_keywords = set(q_text.split())
                
                # Jaccard相似度
                intersection = len(query_keywords & q_keywords)
                union = len(query_keywords | q_keywords)
                sim = intersection / union if union > 0 else 0
                total_sim += sim
            
            avg_sim = total_sim / min(len(questions), 5)
            
            results.append({
                "聚类ID": f"簇 {cluster.get('cluster_id', 0)}",
                "匹配度": f"{avg_sim:.2%}",
                "问题数量": cluster.get('size', 0),
                "相似度分数": avg_sim
            })
        
        # 按相似度排序
        results.sort(key=lambda x: x['相似度分数'], reverse=True)
        
        # 移除用于排序的字段
        for r in results:
            del r['相似度分数']
        
        return results[:5]  # 返回前5个最匹配的
    
    def get_cluster_prompts_for_query(self, cluster_id_str: str) -> str:
        """获取指定簇的prompts用于生成答案"""
        if not cluster_id_str:
            return ""
        
        try:
            cluster_id = int(cluster_id_str.replace("簇 ", "").strip())
        except:
            return "无效的簇ID"
        
        clusters = self.cluster_data.get('clusters', [])
        for cluster in clusters:
            if cluster.get('cluster_id') == cluster_id:
                questions = cluster.get('questions', [])
                prompts = []
                for q in questions[:2]:  # 取前2个问题的final_prompt
                    if 'final_prompt' in q:
                        fp = q['final_prompt']
                        prompts.append({
                            'question': q.get('question', ''),
                            'full_prompt': fp.get('full_prompt', '')
                        })
                
                if prompts:
                    return json.dumps(prompts, ensure_ascii=False, indent=2)
                return "该簇没有可用的prompt"
        
        return "未找到簇"
    
    # ==================== 功能6: 双提示生成与答案融合 ====================
    def generate_answers_with_prompts(self, query: str, cluster1_id: str, cluster2_id: str) -> Tuple[str, str, str, str]:
        """使用两个簇的prompt生成答案并融合"""
        if not query:
            return "请输入Query", "", "", ""
        
        # 获取两个簇的prompts
        prompt1 = self._get_prompt_from_cluster(cluster1_id)
        prompt2 = self._get_prompt_from_cluster(cluster2_id)
        
        if not prompt1 or not prompt2:
            return "无法获取prompt", "", "", ""
        
        # 模拟调用LLM生成答案（实际应该调用真实API）
        answer1 = self._simulate_llm_call(query, prompt1)
        answer2 = self._simulate_llm_call(query, prompt2)
        
        # 融合答案
        fused_answer = self._fuse_answers(answer1, answer2)
        
        return answer1, answer2, fused_answer, f"使用的簇: {cluster1_id}, {cluster2_id}"
    
    def _get_prompt_from_cluster(self, cluster_id_str: str) -> Optional[str]:
        """从簇中获取一个代表性prompt"""
        if not cluster_id_str:
            return None
        
        try:
            cluster_id = int(cluster_id_str.replace("簇 ", "").strip())
        except:
            return None
        
        clusters = self.cluster_data.get('clusters', [])
        for cluster in clusters:
            if cluster.get('cluster_id') == cluster_id:
                questions = cluster.get('questions', [])
                if questions and 'final_prompt' in questions[0]:
                    return questions[0]['final_prompt'].get('full_prompt', '')
        
        return None
    
    def _simulate_llm_call(self, query: str, prompt: str) -> str:
        """模拟LLM调用（实际应该调用真实API）"""
        # 这里应该调用真实的LLM API
        # 现在返回模拟结果
        return f"[模拟答案] 基于prompt生成的答案 for: {query[:30]}...\n\n这里应该调用真实的大模型API生成答案。"
    
    def _fuse_answers(self, answer1: str, answer2: str) -> str:
        """融合两个答案"""
        # 简单的融合策略：取两者的并集信息
        fused = f"【融合答案】\n\n基于两个prompt生成的答案融合:\n\n"
        fused += f"来源1要点:\n{answer1[:200]}...\n\n"
        fused += f"来源2要点:\n{answer2[:200]}...\n\n"
        fused += "综合结论:\n根据两个来源的信息综合得出最终答案。"
        return fused
    
    # ==================== 创建UI ====================
    def create_ui(self):
        """创建Tourist RAG演示UI"""
        with gr.Blocks() as demo:
            gr.Markdown("""
            # 🏛️ Tourist RAG 演示系统
            
            基于群智优化的旅游问答RAG系统演示
            """)
            
            with gr.Tabs():
                # ========== 功能1: 数据集展示 ==========
                with gr.TabItem("📊 数据集展示"):
                    gr.Markdown("### 查看Tourist数据集中的问题、答案和上下文")
                    
                    with gr.Row():
                        refresh_dataset_btn = gr.Button("🔄 刷新数据", variant="primary")
                    
                    dataset_table = gr.Dataframe(
                        headers=["问题ID", "景点", "问题", "答案摘要"],
                        label="数据集列表",
                        interactive=False
                    )
                    
                    gr.Markdown("### 问题详情")
                    with gr.Row():
                        question_id_input = gr.Textbox(label="问题ID（从上方表格复制）", placeholder="输入问题ID...")
                        load_detail_btn = gr.Button("加载详情")
                    
                    with gr.Row():
                        detail_question = gr.Textbox(label="问题", lines=2, interactive=False)
                        detail_attraction = gr.Textbox(label="景点", interactive=False)
                    
                    detail_answer = gr.Textbox(label="答案", lines=5, interactive=False)
                    detail_context = gr.Textbox(label="上下文", lines=5, interactive=False)
                    
                    # 事件绑定
                    refresh_dataset_btn.click(
                        fn=self.get_dataset_list,
                        outputs=[dataset_table]
                    )
                    load_detail_btn.click(
                        fn=self.get_question_detail,
                        inputs=[question_id_input],
                        outputs=[detail_question, detail_answer, detail_context, detail_attraction]
                    )
                
                # ========== 功能2: 提示初始化生成 ==========
                with gr.TabItem("📝 提示初始化"):
                    gr.Markdown("### 查看初始化的Prompt生成结果")
                    
                    with gr.Row():
                        refresh_prompt_init_btn = gr.Button("🔄 刷新列表", variant="primary")
                    
                    prompt_init_table = gr.Dataframe(
                        headers=["问题ID", "问题", "问题类型", "复杂度"],
                        label="Prompt初始化列表",
                        interactive=False
                    )
                    
                    gr.Markdown("### Prompt详情")
                    with gr.Row():
                        prompt_init_id_input = gr.Textbox(label="问题ID", placeholder="输入问题ID...")
                        load_prompt_init_btn = gr.Button("加载详情")
                    
                    init_question = gr.Textbox(label="问题", lines=2, interactive=False)
                    init_full_prompt = gr.Textbox(label="完整Prompt", lines=10, interactive=False)
                    
                    with gr.Row():
                        init_key_aspects = gr.Textbox(label="关键方面", lines=3, interactive=False)
                        init_scene = gr.Textbox(label="场景分析", lines=3, interactive=False)
                    
                    init_prompt_module = gr.Textbox(label="Prompt模块", lines=5, interactive=False)
                    
                    # 事件绑定
                    refresh_prompt_init_btn.click(
                        fn=self.get_prompt_init_list,
                        outputs=[prompt_init_table]
                    )
                    load_prompt_init_btn.click(
                        fn=self.get_prompt_init_detail,
                        inputs=[prompt_init_id_input],
                        outputs=[init_question, init_full_prompt, init_key_aspects, init_scene, init_prompt_module]
                    )
                
                # ========== 功能3: 单问题迭代优化 ==========
                with gr.TabItem("🔄 迭代优化"):
                    gr.Markdown("### 查看单问题的迭代优化过程")
                    
                    with gr.Row():
                        iteration_question_dropdown = gr.Dropdown(
                            choices=self.get_iteration_questions(),
                            label="选择问题",
                            interactive=True
                        )
                        iteration_num = gr.Slider(minimum=0, maximum=4, step=1, value=0, label="迭代轮次")
                    
                    load_iteration_btn = gr.Button("加载迭代详情", variant="primary")
                    
                    iteration_info = gr.Textbox(label="迭代信息", lines=3, interactive=False)
                    iteration_answer = gr.Textbox(label="生成的答案", lines=5, interactive=False)
                    iteration_improvements = gr.Textbox(label="改进建议", lines=5, interactive=False)
                    iteration_prompt = gr.Textbox(label="当前Prompt", lines=5, interactive=False)
                    
                    # 事件绑定
                    load_iteration_btn.click(
                        fn=self.get_iteration_detail,
                        inputs=[iteration_question_dropdown, iteration_num],
                        outputs=[iteration_info, iteration_answer, iteration_improvements, iteration_prompt]
                    )
                
                # ========== 功能4: 聚类与群智优化 ==========
                with gr.TabItem("🎯 聚类展示"):
                    gr.Markdown("### 查看聚类与群智优化结果")
                    
                    with gr.Row():
                        refresh_cluster_btn = gr.Button("🔄 刷新聚类", variant="primary")
                    
                    cluster_table = gr.Dataframe(
                        headers=["聚类ID", "问题数量", "问题示例"],
                        label="聚类列表",
                        interactive=False
                    )
                    
                    gr.Markdown("### 聚类详情")
                    with gr.Row():
                        cluster_id_input = gr.Textbox(label="聚类ID（如：簇 0）", placeholder="输入聚类ID...")
                        load_cluster_btn = gr.Button("加载详情")
                    
                    cluster_stats = gr.Textbox(label="统计信息", lines=3, interactive=False)
                    cluster_questions = gr.Textbox(label="簇内问题", lines=10, interactive=False)
                    cluster_prompts = gr.Textbox(label="群智优化Prompts", lines=10, interactive=False)
                    
                    # 事件绑定
                    refresh_cluster_btn.click(
                        fn=self.get_cluster_list,
                        outputs=[cluster_table]
                    )
                    load_cluster_btn.click(
                        fn=self.get_cluster_detail,
                        inputs=[cluster_id_input],
                        outputs=[cluster_stats, cluster_questions, cluster_prompts]
                    )
                
                # ========== 功能5: Query匹配聚类簇 ==========
                with gr.TabItem("🔍 Query匹配"):
                    gr.Markdown("### 输入Query匹配最相似的聚类簇")
                    
                    with gr.Row():
                        query_input = gr.Textbox(label="输入Query", placeholder="请输入您的问题...", lines=2)
                        match_btn = gr.Button("匹配聚类", variant="primary")
                    
                    match_results = gr.Dataframe(
                        headers=["聚类ID", "匹配度", "问题数量"],
                        label="匹配结果（按匹配度排序）",
                        interactive=False
                    )
                    
                    gr.Markdown("### 选中簇的Prompts")
                    with gr.Row():
                        selected_cluster_id = gr.Textbox(label="选中的聚类ID", placeholder="从上方表格复制...")
                        load_cluster_prompts_btn = gr.Button("加载Prompts")
                    
                    cluster_prompts_display = gr.Textbox(label="簇Prompts", lines=15, interactive=False)
                    
                    # 事件绑定
                    match_btn.click(
                        fn=self.match_query_to_clusters,
                        inputs=[query_input],
                        outputs=[match_results]
                    )
                    load_cluster_prompts_btn.click(
                        fn=self.get_cluster_prompts_for_query,
                        inputs=[selected_cluster_id],
                        outputs=[cluster_prompts_display]
                    )
                
                # ========== 功能6: 双提示生成与答案融合 ==========
                with gr.TabItem("🔀 答案生成与融合"):
                    gr.Markdown("### 使用两个簇的Prompt生成答案并融合")
                    
                    query_for_generation = gr.Textbox(label="输入Query", placeholder="请输入您的问题...", lines=2)
                    
                    with gr.Row():
                        cluster1_dropdown = gr.Dropdown(
                            choices=[f"簇 {i}" for i in range(8)],
                            label="选择第一个聚类簇"
                        )
                        cluster2_dropdown = gr.Dropdown(
                            choices=[f"簇 {i}" for i in range(8)],
                            label="选择第二个聚类簇"
                        )
                    
                    generate_btn = gr.Button("生成并融合答案", variant="primary")
                    
                    generation_info = gr.Textbox(label="生成信息", interactive=False)
                    
                    with gr.Row():
                        answer1_output = gr.Textbox(label="答案1（来自簇1）", lines=8, interactive=False)
                        answer2_output = gr.Textbox(label="答案2（来自簇2）", lines=8, interactive=False)
                    
                    fused_answer_output = gr.Textbox(label="融合后的最终答案", lines=10, interactive=False)
                    
                    # 事件绑定
                    generate_btn.click(
                        fn=self.generate_answers_with_prompts,
                        inputs=[query_for_generation, cluster1_dropdown, cluster2_dropdown],
                        outputs=[answer1_output, answer2_output, fused_answer_output, generation_info]
                    )
        
        return demo


# 创建页面实例
page = TouristRAGNewPage()

if __name__ == "__main__":
    demo = page.create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7890)
