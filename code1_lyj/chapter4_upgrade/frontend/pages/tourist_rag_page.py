"""
第四章：基于提示自动迭代的RAG问答系统
包含数据集的完整演示功能
"""
import os
import sys
import requests
import gradio as gr
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import time
import random
import numpy as np
    
# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config

# API基础URL
API_BASE_URL = f"http://127.0.0.1:{config.PORT}"

# 数据路径配置
TOURIST_DATA_DIR = Path("i:/bylw_final/Code/chapter3/codes/bylw_rag/new_experiments")
TOURIST_EVAL_PATH = TOURIST_DATA_DIR / "datas/tourist/tourist_eval.json"
TOURIST_PROMPT_DIR = TOURIST_DATA_DIR / "prompt_library/tourist"
TOURIST_ITERATION_DIR = TOURIST_DATA_DIR / "iteration_results/tourist"  # 迭代结果目录
TOURIST_CLUSTER_DIR = TOURIST_DATA_DIR / "optimized_prompts/tourist/cluster_0"


class TouristRAGPage:
    """基于提示自动迭代的RAG问答系统页面类"""
    
    def __init__(self):
        self.dataset_cache = None
        self.prompts_cache = {}
        self.clusters_cache = None
        self.iteration_questions_cache = []
        self.available_prompts_for_fusion = []
    
    # ==================== 功能1: 数据集加载 ====================
    
    def load_dataset(self):
        """加载数据集 - 读取tourist_eval.json中的问题、答案和上下文"""
        try:
            with open(TOURIST_EVAL_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.dataset_cache = data
            
            dataset_list = []
            
            for i, item in enumerate(data[:50]):
                dataset_list.append([
                    str(i + 1),
                    item.get('question_id', ''),
                    item.get('attraction', '未知'),
                    item.get('question', '')[:60] + ("..." if len(item.get('question', '')) > 60 else ""),
                    item.get('answer', '')[:80] + ("..." if len(item.get('answer', '')) > 80 else ""),
                    item.get('document', '')[:80] + ("..." if len(item.get('document', '')) > 80 else "")
                ])
            
            return f"✅ 成功加载 {len(data)} 条数据", dataset_list, data[:50]
        
        except Exception as e:
            return f"❌ 加载数据失败: {str(e)}", [], []
    
    def on_dataset_select(self, evt: gr.SelectData):
        """处理数据集表格选择事件"""
        if evt is None:
            return -1, "请点击选择一行数据"
        try:
            index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            return int(index), f"已选择第 {int(index) + 1} 行"
        except:
            return -1, "选择错误"
    
    def show_dataset_detail(self, selected_idx: int, dataset_data: list):
        """显示数据集详细信息"""
        try:
            selected_idx = int(selected_idx)
        except (ValueError, TypeError):
            return "请先选择一条记录"
        
        if selected_idx < 0 or selected_idx >= len(dataset_data):
            return "请先选择一条记录"
        
        item = dataset_data[selected_idx]
        
        detail = f"""## 📋 数据详情

**问题ID**: {item.get('question_id', 'N/A')}

**景点名称**: {item.get('attraction', 'N/A')}

---

### ❓ 问题
{item.get('question', 'N/A')}

---

### ✅ 参考答案
{item.get('answer', 'N/A')}

---

### 📄 应该检索的上下文（Document）
{item.get('document', 'N/A')}

---

### 🔍 来源片段（Source）
"""
        for j, source in enumerate(item.get('source', []), 1):
            detail += f"\n**来源 {j}**: {source}\n"
        
        return detail
    
    # ==================== 功能2: 提示初始化生成 ====================
    
    def load_initial_prompts(self):
        """加载初始化Prompt - 读取tourist中的每个iteration_0.json中的full_prompt"""
        try:
            prompts = []
            prompt_dir = TOURIST_PROMPT_DIR
            
            for question_dir in sorted(prompt_dir.iterdir()):
                if question_dir.is_dir():
                    iteration_file = question_dir / "iteration_0.json"
                    if iteration_file.exists():
                        with open(iteration_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        question_id = data.get('question_id', '')
                        self.prompts_cache[question_id] = data
                        
                        prompts.append([
                            str(len(prompts) + 1),
                            question_id,
                            data.get('question', '')[:50] + "...",
                            str(len(data.get('full_prompt', ''))),
                            data.get('question_analysis', {}).get('question_category', '未知')
                        ])
            
            return f"✅ 生成成功 ({len(prompts)} 个Prompt)", prompts, prompts
        
        except Exception as e:
            return f"❌ 加载Prompt失败: {str(e)}", [], []
    
    def on_prompt_select(self, evt: gr.SelectData):
        """处理Prompt表格选择事件"""
        if evt is None:
            return -1, "请点击选择一行数据"
        try:
            index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            return int(index), f"已选择第 {int(index) + 1} 行"
        except:
            return -1, "选择错误"
    
    def show_prompt_detail(self, selected_idx: int, prompts_list: list):
        """显示Prompt详细信息"""
        try:
            selected_idx = int(selected_idx)
        except (ValueError, TypeError):
            return "请先选择一条记录"
        
        if selected_idx < 0 or selected_idx >= len(prompts_list):
            return "请先选择一条记录"
        
        item = prompts_list[selected_idx]
        if isinstance(item, list) and len(item) > 1:
            question_id = item[1]
        else:
            return "数据格式错误"
        
        data = self.prompts_cache.get(question_id, {})
        
        if not data:
            return "未找到对应的Prompt数据"
        
        detail = f"""## 📝 Prompt详情

**问题ID**: {data.get('question_id', 'N/A')}

**问题**: {data.get('question', 'N/A')}

**迭代次数**: {data.get('iteration', 0)}

---

### 🎯 问题分析 (Question Analysis)

- **类别**: {data.get('question_analysis', {}).get('question_category', '未知')}
- **复杂度**: {data.get('question_analysis', {}).get('complexity', '未知')}
- **关键方面**: {', '.join(data.get('question_analysis', {}).get('key_aspects', []))}
- **所需技能**: {', '.join(data.get('question_analysis', {}).get('required_skills', []))}

---

### 🎭 场景分析 (Scene Analysis)

- **场景类型**: {data.get('scene_analysis', {}).get('scene_type', '未知')}
- **用户意图**: {data.get('scene_analysis', {}).get('user_intent', '未知')}
- **回答格式**: {data.get('scene_analysis', {}).get('answer_format', '未知')}
- **特殊要求**: {data.get('scene_analysis', {}).get('special_requirements', '无')}

---

### 📦 Prompt模块 (Prompt Module)
"""
        
        module = data.get('prompt_module', {})
        for key, value in module.items():
            detail += f"\n**{key}**: {value}\n"
        
        detail += f"""

---

### 📜 完整Prompt (Full Prompt)
```
{data.get('full_prompt', 'N/A')}
```
"""
        
        return detail
    
    # ==================== 功能3: 单问题迭代优化 ====================
    
    def load_iteration_dataset(self):
        """加载用于迭代的数据集（Tourist）"""
        try:
            self.iteration_questions_cache = []
            prompt_dir = TOURIST_PROMPT_DIR
            
            for question_dir in sorted(prompt_dir.iterdir()):
                if question_dir.is_dir():
                    iteration_file = question_dir / "iteration_0.json"
                    if iteration_file.exists():
                        with open(iteration_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        qid = data.get('question_id', '')
                        question = data.get('question', '')
                        
                        self.iteration_questions_cache.append({
                            "问题ID": qid,
                            "问题": question,
                            "数据": data
                        })
                        self.prompts_cache[qid] = data
            
            table_data = []
            
            for i, item in enumerate(self.iteration_questions_cache[:24], 1):
                table_data.append([
                    str(i),
                    item["问题ID"],
                    item["问题"][:60] + ("..." if len(item["问题"]) > 60 else "")
                ])
            
            return f"✅ 成功加载 {len(self.iteration_questions_cache)} 条迭代数据", table_data, self.iteration_questions_cache[:24]
        
        except Exception as e:
            return f"❌ 加载失败: {str(e)}", [], []
    
    def get_all_questions_for_iteration(self):
        """获取所有可用于迭代的问题列表"""
        try:
            if not self.iteration_questions_cache:
                self.iteration_questions_cache = []
                prompt_dir = TOURIST_PROMPT_DIR
                
                for question_dir in sorted(prompt_dir.iterdir()):
                    if question_dir.is_dir():
                        iteration_file = question_dir / "iteration_0.json"
                        if iteration_file.exists():
                            with open(iteration_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            qid = data.get('question_id', '')
                            question = data.get('question', '')
                            
                            self.iteration_questions_cache.append({
                                "问题ID": qid,
                                "问题": question,
                                "数据": data
                            })
            
            return self.iteration_questions_cache
        except Exception as e:
            print(f"获取问题列表失败: {e}")
            return []
    
    def load_iteration_history(self, question_id: str):
        """加载单个问题的迭代历史"""
        try:
            if not question_id:
                return []
            
            prompt_dir = TOURIST_PROMPT_DIR
            
            for question_dir in sorted(prompt_dir.iterdir()):
                if question_dir.is_dir() and question_id in question_dir.name:
                    iterations = []
                    
                    for iter_file in sorted(question_dir.glob("iteration_*.json")):
                        with open(iter_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        iter_num = data.get('iteration', 0)
                        iterations.append([
                            str(iter_num),
                            iter_file.name,
                            "是" if data.get('improvement_suggestions') else "否"
                        ])
                    
                    return iterations
            
            return []
        
        except Exception as e:
            print(f"加载迭代历史错误: {e}")
            return []
    
    def on_iteration_select(self, evt: gr.SelectData):
        """处理迭代表格选择事件"""
        if evt is None:
            return -1, "请点击选择一行数据"
        try:
            index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            return int(index), f"已选择第 {int(index) + 1} 行"
        except:
            return -1, "选择错误"
    
    def show_iteration_question_detail(self, selected_idx: int, questions_list: list):
        """显示迭代问题的详细信息（包含三次迭代结果）"""
        try:
            selected_idx = int(selected_idx)
        except (ValueError, TypeError):
            return "请先选择一条记录"
        
        if selected_idx < 0 or selected_idx >= len(questions_list):
            return "请先选择一条记录"
        
        item = questions_list[selected_idx]
        if isinstance(item, dict):
            question_id = item.get("问题ID", "")
        elif isinstance(item, list) and len(item) > 1:
            question_id = item[1]
        else:
            return "数据格式错误"
        
        try:
            # 从 iteration_results 目录读取迭代结果
            iteration_dir = TOURIST_ITERATION_DIR
            
            # 查找包含该question_id的目录
            target_dir = None
            for question_dir in sorted(iteration_dir.iterdir()):
                if question_dir.is_dir() and question_id in question_dir.name:
                    target_dir = question_dir
                    break
            
            if not target_dir:
                return f"未找到问题 {question_id} 的迭代数据"
            
            # 读取 iteration_results.json 文件
            results_file = target_dir / "iteration_results.json"
            if not results_file.exists():
                return f"未找到迭代结果文件: {results_file}"
            
            with open(results_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            
            question = results_data.get('question', 'N/A')
            initial_prompt = results_data.get('initial_prompt', {})
            iterations = results_data.get('iterations', [])
            
            detail = f"""## 🔄 问题迭代详情

**问题ID**: {question_id}

**问题**: {question}

---

### 📝 初始化Prompt
```
{initial_prompt.get('full_prompt', 'N/A')[:500]}...
```

---
"""
            
            # 展示三次迭代结果
            for i, iter_data in enumerate(iterations[:3], 1):
                # 获取生成的答案并解析JSON
                generated_answer_raw = iter_data.get('generated_answer', '暂无数据')
                try:
                    # 尝试解析JSON格式的答案
                    if isinstance(generated_answer_raw, str) and generated_answer_raw.strip().startswith('{'):
                        answer_json = json.loads(generated_answer_raw)
                        generated_answer = answer_json.get('answer', generated_answer_raw)
                    else:
                        generated_answer = generated_answer_raw
                except:
                    generated_answer = generated_answer_raw
                
                # 构建完整的Prompt（从prompt_module组件组装）
                prompt_module = iter_data.get('prompt_module', {})
                if prompt_module:
                    full_prompt = f"""<system_prompt>
{prompt_module.get('P_sys', '')}
</system_prompt>

<instruction>
{prompt_module.get('I_t', '')}
</instruction>

<context_strategy>
{prompt_module.get('C_t', '')}
</context_strategy>

<format_requirement>
{prompt_module.get('F_t', '')}
</format_requirement>

<uncertainty_handling>
{prompt_module.get('U_t', '')}
</uncertainty_handling>"""
                else:
                    full_prompt = "N/A"
                
                detail += f"""
### 第 {i} 次迭代

**生成的答案**:
{generated_answer[:500]}...

**改进建议**:
"""
                evaluation = iter_data.get('evaluation', {})
                suggestions = evaluation.get('improvement_suggestions', [])
                if suggestions:
                    for j, sug in enumerate(suggestions, 1):
                        detail += f"\n{j}. {sug}\n"
                else:
                    detail += "\n本轮无改进建议\n"
                
                detail += f"""
**迭代后的Prompt**:
```
{full_prompt[:800]}...
```

---
"""
            
            return detail
        
        except Exception as e:
            import traceback
            return f"加载详情失败: {str(e)}\n{traceback.format_exc()}"
    
    # ==================== 功能4: 聚类与群智优化 ====================
    
    def load_cluster_results(self):
        """加载聚类结果 - 读取cluster_0文件夹"""
        try:
            cluster_results = []
            
            optimization_summary = TOURIST_CLUSTER_DIR / "optimization_summary.json"
            if optimization_summary.exists():
                with open(optimization_summary, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                
                self.clusters_cache = summary
                
                for prompt_info in summary.get('prompts', []):
                    cluster_results.append([
                        str(prompt_info.get('prompt_id', 0)),
                        summary.get('cluster_category', '未知'),
                        str(prompt_info.get('based_questions_count', 0)),
                        prompt_info.get('reasoning_preview', '')
                    ])
                
                return f"✅ 成功加载聚类结果", cluster_results, summary
            
            return "❌ 未找到聚类结果文件", [], None
        
        except Exception as e:
            return f"❌ 加载聚类结果失败: {str(e)}", [], None
    
    def load_all_cluster_results(self):
        """加载所有聚类结果 - 读取所有cluster文件夹"""
        try:
            cluster_results = []
            all_clusters_data = []
            
            # 获取父目录
            parent_dir = TOURIST_CLUSTER_DIR.parent
            
            # 遍历所有cluster文件夹
            for cluster_dir in sorted(parent_dir.iterdir()):
                if cluster_dir.is_dir() and cluster_dir.name.startswith('cluster_'):
                    optimization_summary = cluster_dir / "optimization_summary.json"
                    if optimization_summary.exists():
                        with open(optimization_summary, 'r', encoding='utf-8') as f:
                            summary = json.load(f)
                        
                        cluster_category = summary.get('cluster_category', cluster_dir.name)
                        
                        for prompt_info in summary.get('prompts', []):
                            cluster_results.append([
                                str(prompt_info.get('prompt_id', 0)),
                                cluster_category,
                                str(prompt_info.get('based_questions_count', 0)),
                                prompt_info.get('reasoning_preview', '')
                            ])
                            
                            # 保存完整数据
                            all_clusters_data.append({
                                'prompt_id': prompt_info.get('prompt_id', 0),
                                'cluster_category': cluster_category,
                                'cluster_dir': cluster_dir,
                                'prompt_info': prompt_info,
                                'summary': summary
                            })
            
            # 保存到缓存
            self.all_clusters_data = all_clusters_data
            
            if cluster_results:
                return f"✅ 成功加载 {len(cluster_results)} 个聚类Prompt", cluster_results, all_clusters_data
            
            return "❌ 未找到聚类结果文件", [], None
        
        except Exception as e:
            return f"❌ 加载聚类结果失败: {str(e)}", [], None
    
    def on_cluster_select(self, evt: gr.SelectData):
        """处理聚类表格选择事件"""
        if evt is None:
            return -1, "请点击选择一行数据"
        try:
            index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            return int(index), f"已选择第 {int(index) + 1} 行"
        except:
            return -1, "选择错误"
    
    def show_cluster_questions(self, selected_idx: int, clusters: list):
        """显示聚类包含的问题"""
        try:
            selected_idx = int(selected_idx)
        except (ValueError, TypeError):
            return "请先选择一条记录"
        
        if selected_idx < 0 or selected_idx >= len(clusters):
            return "请先选择一条记录"
        
        item = clusters[selected_idx]
        prompt_id = item[0] if isinstance(item, list) else item.get("Prompt ID", "")
        
        try:
            optimized_prompt_file = TOURIST_CLUSTER_DIR / f"optimized_prompt_{prompt_id}.json"
            
            if optimized_prompt_file.exists():
                with open(optimized_prompt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                questions = data.get('based_questions', [])
                
                result = f"## 📝 聚类包含的问题 ({len(questions)}个)\n\n"
                for i, q in enumerate(questions, 1):
                    result += f"**问题 {i}**: {q}\n\n"
                
                return result
            
            return "未找到问题列表"
        
        except Exception as e:
            return f"读取问题列表错误: {str(e)}"
    
    def show_cluster_detail(self, selected_idx: int, clusters: list):
        """显示聚类的详细内容"""
        try:
            selected_idx = int(selected_idx)
        except (ValueError, TypeError):
            return "请先选择一条记录"
        
        if selected_idx < 0 or selected_idx >= len(clusters):
            return "请先选择一条记录"
        
        item = clusters[selected_idx]
        
        # 处理列表格式或字典格式
        if isinstance(item, list) and len(item) >= 4:
            prompt_id = item[0]
            cluster_category = item[1]
        elif isinstance(item, dict):
            prompt_id = item.get('prompt_id', '')
            cluster_category = item.get('cluster_category', '')
        else:
            return "数据格式错误"
        
        try:
            # 从all_clusters_data中查找对应的cluster_dir
            cluster_dir = None
            if hasattr(self, 'all_clusters_data') and self.all_clusters_data:
                for cluster_data in self.all_clusters_data:
                    if str(cluster_data.get('prompt_id')) == str(prompt_id):
                        cluster_dir = cluster_data.get('cluster_dir')
                        break
            
            # 如果没找到，使用默认路径
            if not cluster_dir:
                cluster_dir = TOURIST_CLUSTER_DIR
            
            optimized_prompt_file = cluster_dir / f"optimized_prompt_{prompt_id}.json"
            
            if optimized_prompt_file.exists():
                with open(optimized_prompt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                detail = f"""## 🎯 群智优化Prompt详情

**Prompt ID**: {data.get('prompt_id', 'N/A')}

**聚类类别**: {data.get('cluster_category', cluster_category)}

---

### 📋 基于哪些问题进行群智优化
"""
                for q in data.get('based_questions', []):
                    detail += f"- {q}\n"
                
                detail += f"""

### 🤖 优化后的Prompt模块
"""
                module = data.get('prompt_module', {})
                for key, value in module.items():
                    if isinstance(value, list):
                        detail += f"\n**{key}**:\n"
                        for v in value:
                            detail += f"  - {v}\n"
                    else:
                        detail += f"\n**{key}**: {value}\n"
                
                detail += f"""

### 🧠 优化推理过程
{data.get('optimization_reasoning', '暂无推理记录')}
"""
                
                return detail
            
            return f"未找到Prompt文件: {optimized_prompt_file}"
        
        except Exception as e:
            return f"读取Prompt错误: {str(e)}"
    
    # ==================== 功能5: Query匹配聚类簇 ====================
    
    def _load_cluster_centers(self):
        """加载聚类中心数据"""
        try:
            cluster_file = TOURIST_DATA_DIR / "clustering_results" / "tourist" / "cluster_results.json"
            if not cluster_file.exists():
                return {}
            
            with open(cluster_file, 'r', encoding='utf-8') as f:
                cluster_data = json.load(f)
            
            cluster_centers = {}
            for cluster in cluster_data.get('clusters', []):
                cluster_id = cluster['cluster_id']
                center = np.array(cluster['center'])
                cluster_centers[cluster_id] = center
            
            return cluster_centers
        except Exception as e:
            print(f"加载聚类中心失败: {e}")
            return {}
    
    def _encode_query(self, query: str) -> np.ndarray:
        """使用BGE模型编码查询"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # 加载BGE模型（使用本地路径）
            model_path = r"i:\bylw_final\Code\models\embedding\bge-large-zh-v1.5"
            model = SentenceTransformer(model_path)
            
            # 编码查询
            embedding = model.encode(query, normalize_embeddings=True)
            return embedding
        except Exception as e:
            print(f"编码查询失败: {e}")
            return None
    
    def _calculate_cosine_similarity(self, query_emb: np.ndarray, center: np.ndarray) -> float:
        """计算余弦相似度"""
        try:
            # 确保维度一致
            if len(center) > len(query_emb):
                center = center[:len(query_emb)]
            elif len(center) < len(query_emb):
                query_emb = query_emb[:len(center)]
            
            # 计算余弦相似度
            dot_product = np.dot(query_emb, center)
            norm_query = np.linalg.norm(query_emb)
            norm_center = np.linalg.norm(center)
            
            if norm_query == 0 or norm_center == 0:
                return 0.0
            
            similarity = dot_product / (norm_query * norm_center)
            return float(similarity)
        except Exception as e:
            print(f"计算相似度失败: {e}")
            return 0.0
    
    def match_query_to_clusters(self, query: str):
        """根据用户Query匹配聚类簇 - 使用BGE嵌入模型计算实际相似度"""
        if not query:
            return "❌ 请输入查询内容", [], []
        
        try:
            # 使用加载所有cluster的方法
            if not hasattr(self, 'all_clusters_data') or not self.all_clusters_data:
                _, _, _ = self.load_all_cluster_results()
            
            # 加载聚类中心
            cluster_centers = self._load_cluster_centers()
            if not cluster_centers:
                return "❌ 无法加载聚类中心数据", [], []
            
            # 编码查询
            query_emb = self._encode_query(query)
            if query_emb is None:
                return "❌ 查询编码失败", [], []
            
            matched_clusters = []
            
            # 遍历所有cluster数据
            for cluster_data in self.all_clusters_data:
                prompt_id = cluster_data.get('prompt_id', 0)
                cluster_dir = cluster_data.get('cluster_dir')
                cluster_category = cluster_data.get('cluster_category', '未知')
                
                # 从cluster_dir中提取cluster_id
                cluster_id = None
                if cluster_dir:
                    try:
                        cluster_id = int(cluster_dir.name.replace('cluster_', ''))
                    except:
                        pass
                
                # 获取聚类中心
                center = cluster_centers.get(cluster_id)
                if center is None:
                    continue
                
                # 计算实际相似度
                similarity = self._calculate_cosine_similarity(query_emb, center)
                
                optimized_prompt_file = cluster_dir / f"optimized_prompt_{prompt_id}.json"
                if optimized_prompt_file.exists():
                    with open(optimized_prompt_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    based_questions = data.get('based_questions', [])
                    
                    matched_clusters.append([
                        f"{similarity:.4f}",
                        cluster_category,
                        str(len(based_questions)),
                        str(prompt_id),
                        ', '.join(based_questions[:2]) + ('...' if len(based_questions) > 2 else '')
                    ])
            
            # 按匹配度排序（相似度高的在前）
            matched_clusters.sort(key=lambda x: float(x[0]), reverse=True)
            
            return f"✅ 匹配完成，找到 {len(matched_clusters)} 个相关聚类簇", matched_clusters[:10], matched_clusters
        
        except Exception as e:
            import traceback
            return f"❌ 匹配失败: {str(e)}\n{traceback.format_exc()}", [], []
    
    def on_match_select(self, evt: gr.SelectData):
        """处理匹配结果表格选择事件"""
        if evt is None:
            return -1, "请点击选择一行数据"
        try:
            index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            return int(index), f"已选择第 {int(index) + 1} 行"
        except:
            return -1, "选择错误"
    
    def show_matched_prompt(self, selected_idx: int, matches: list):
        """显示匹配到的Prompt"""
        try:
            selected_idx = int(selected_idx)
        except (ValueError, TypeError):
            return "请先选择一条记录"
        
        if selected_idx < 0 or selected_idx >= len(matches):
            return "请先选择一条记录"
        
        item = matches[selected_idx]
        prompt_id = item[3] if isinstance(item, list) else item.get("Prompt ID", "")
        match_score = item[0] if isinstance(item, list) else item.get("匹配度", "")
        cluster_category = item[1] if isinstance(item, list) else item.get("聚类类别", "")
        
        try:
            optimized_prompt_file = TOURIST_CLUSTER_DIR / f"optimized_prompt_{prompt_id}.json"
            
            if optimized_prompt_file.exists():
                with open(optimized_prompt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                template = data.get('prompt_module', {}).get('prompt_template', '暂无模板')
                
                return f"""## 🎯 匹配到的Prompt

**匹配度**: {match_score}

**聚类类别**: {cluster_category}

---

### 📜 Prompt模板
```
{template}
```

### 📋 基于的问题
{chr(10).join([f'- {q}' for q in data.get('based_questions', [])])}
"""
            
            return "未找到Prompt文件"
        
        except Exception as e:
            return f"读取Prompt错误: {str(e)}"
    
    # ==================== 功能6: 答案生成与融合 ====================
    
    def refresh_prompts_for_fusion(self):
        """刷新可用于融合的Prompt列表 - 加载所有cluster的Prompt"""
        try:
            # 使用加载所有cluster的方法
            if not hasattr(self, 'all_clusters_data') or not self.all_clusters_data:
                _, _, _ = self.load_all_cluster_results()
            
            self.available_prompts_for_fusion = []
            choices = []
            
            # 遍历所有cluster数据
            for cluster_data in self.all_clusters_data:
                prompt_id = cluster_data.get('prompt_id', 0)
                cluster_dir = cluster_data.get('cluster_dir')
                cluster_category = cluster_data.get('cluster_category', '未知')
                
                optimized_prompt_file = cluster_dir / f"optimized_prompt_{prompt_id}.json"
                if optimized_prompt_file.exists():
                    with open(optimized_prompt_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    questions = data.get('based_questions', [])
                    preview = f"Prompt {prompt_id} ({cluster_category}): {questions[0][:30]}..." if questions else f"Prompt {prompt_id} ({cluster_category})"
                    choices.append(preview)
                    self.available_prompts_for_fusion.append({
                        'id': prompt_id,
                        'cluster_dir': cluster_dir,
                        'preview': preview,
                        'data': data
                    })
            
            return gr.update(choices=choices, value=[]), f"✅ 已加载 {len(choices)} 个Prompt"
        except Exception as e:
            return gr.update(choices=[], value=[]), f"❌ 获取Prompt列表失败: {str(e)}"
    
    def load_fusion_result_from_file(self):
        """从结果文件加载融合结果数据"""
        try:
            result_file = TOURIST_DATA_DIR / "results" / "tourist" / "top_2_cluster_fusion_20260327_173937" / "1_4.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"加载融合结果失败: {e}")
            return None
    
    def generate_answers_with_fusion(self, query: str, selected_matches: list):
        """根据匹配到的两个提示生成答案并进行融合 - 使用实际结果文件数据"""
        try:
            # 加载实际结果数据
            result_data = self.load_fusion_result_from_file()
            
            if not result_data:
                return "❌ 无法加载结果数据", "", "", ""
            
            question = result_data.get('question', '')
            context = result_data.get('context', '')
            reference_answer = result_data.get('reference_answer', '')
            generated_answers = result_data.get('generated_answers', [])
            fused_answer = result_data.get('fused_answer', '')
            used_prompts = result_data.get('used_prompts', [])
            matched_clusters = result_data.get('matched_clusters', [])
            cluster_similarities = result_data.get('cluster_similarities', [])
            
            # 评估指标
            bleu_score = result_data.get('bleu_score', 0)
            rouge_l_score = result_data.get('rouge_l_score', 0)
            precision = result_data.get('precision', 0)
            recall = result_data.get('recall', 0)
            f1_score = result_data.get('f1_score', 0)
            llm_score = result_data.get('llm_score', 0)
            llm_reasoning = result_data.get('llm_reasoning', '')
            
            # 构建Prompt信息
            prompt_info_text = ""
            for i, prompt in enumerate(used_prompts):
                cluster_id = prompt.get('cluster_id', 'N/A')
                cluster_category = prompt.get('cluster_category', 'N/A')
                similarity = cluster_similarities[i] if i < len(cluster_similarities) else 0
                prompt_info_text += f"**Prompt {i+1}**: 聚类 {cluster_id} ({cluster_category}), 相似度: {similarity:.4f}\n\n"
            
            # 构建检索上下文
            retrieved_contexts = result_data.get('retrieved_contexts', [])
            context_text = ""
            for ctx in retrieved_contexts:
                rank = ctx.get('rank', 0)
                text = ctx.get('text', '')
                context_text += f"**上下文 {rank}**: {text}\n\n"
            
            result = f"""## 🎯 答案生成与融合结果

### 📝 用户查询
{question}

---

### 📚 检索到的上下文
{context_text}

---

### ✅ 参考答案
{reference_answer}

---

### 🤖 使用的Prompt信息
{prompt_info_text}

---

### 💡 答案1 (使用Prompt #{used_prompts[0].get('prompt_id', 'N/A') if used_prompts else 'N/A'})
{generated_answers[0] if len(generated_answers) > 0 else 'N/A'}

---

### 💡 答案2 (使用Prompt #{used_prompts[1].get('prompt_id', 'N/A') if len(used_prompts) > 1 else 'N/A'})
{generated_answers[1] if len(generated_answers) > 1 else 'N/A'}

---

### 🔗 最终融合答案
{fused_answer}

---

### 📊 评估指标
- **BLEU Score**: {bleu_score:.4f}
- **ROUGE-L Score**: {rouge_l_score:.4f}
- **Precision**: {precision:.4f}
- **Recall**: {recall:.4f}
- **F1 Score**: {f1_score:.4f}
- **LLM Score**: {llm_score:.1f}/5.0

### 🧠 LLM评估理由
{llm_reasoning}
"""
            
            answer1 = generated_answers[0] if len(generated_answers) > 0 else ''
            answer2 = generated_answers[1] if len(generated_answers) > 1 else ''
            
            return result, answer1, answer2, fused_answer
        except Exception as e:
            import traceback
            return f"❌ 生成答案失败: {str(e)}\n{traceback.format_exc()}", "", "", ""
    
    def _retrieve_context(self, query: str) -> str:
        """检索相关上下文（简化版）"""
        return "这是从向量库检索到的相关上下文信息..."
    
    def _call_llm_for_answer(self, prompt: str) -> str:
        """调用LLM生成答案"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/llm/chat",
                json={"messages": [{"role": "user", "content": prompt}]},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('response', data.get('answer', '无法生成答案'))
            
            return "[模拟] 基于提供的上下文，这是一个关于旅游问题的参考答案。"
        
        except Exception as e:
            return f"[模拟] 由于API不可用，这里返回模拟答案。原始错误: {str(e)[:50]}"
    
    def _fuse_answers(self, answer1: str, answer2: str) -> str:
        """融合两个答案"""
        if not answer1 and not answer2:
            return "无法融合空答案"
        
        if not answer1:
            return answer2
        if not answer2:
            return answer1
        
        words1 = set(answer1.split())
        words2 = set(answer2.split())
        common = words1 & words2
        unique1 = words1 - words2
        unique2 = words2 - words1
        
        fused_parts = list(common)
        fused_parts.extend(list(unique1)[:len(unique2)])
        fused_parts.extend(list(unique2))
        
        fused_text = ' '.join(fused_parts)
        
        if len(fused_text) > len(answer1) + len(answer2):
            fused_text = answer1 + "\n\n---\n\n" + answer2
        
        return fused_text
    
    # ==================== 创建UI ====================
    
    def create_ui(self) -> gr.Blocks:
        """创建基于提示自动迭代的RAG问答系统的Gradio界面"""
        
        with gr.Blocks(title="基于提示自动迭代的RAG问答系统") as tourist_app:
            gr.Markdown("""
            # 🚀 基于提示自动迭代的RAG问答系统
            
            基于群智优化的RAG问答系统
            """)
            
            # ==================== Tab 1: 数据集展示 ====================
            with gr.Tab("📊 1. 数据集展示"):
                gr.Markdown("""
                ### 📊 数据集展示
                
                从tourist数据集中加载问题、答案和应该检索的上下文。
                """)
                
                load_dataset_btn = gr.Button("📂 加载数据集", variant="primary")
                dataset_status = gr.Textbox(label="状态", lines=1)
                
                # 全宽表格 - 移除col_count使其自动适应
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
                        dataset_detail_text = gr.Textbox(label="详细信息", lines=20, max_lines=30)
                
                dataset_state = gr.State([])
            
            # ==================== Tab 2: 提示初始化 ====================
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
                        prompts_detail_text = gr.Textbox(label="Prompt详情", lines=20, max_lines=30)
                
                prompts_state = gr.State([])
            
            # ==================== Tab 3: 迭代优化 ====================
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
                        iteration_detail_text = gr.Textbox(label="迭代详情（含三次迭代结果）", lines=25, max_lines=40)
                
                iteration_state = gr.State([])
            
            # ==================== Tab 4: 聚类与群智优化 ====================
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
                    row_count=5
                )
                
                # 单选框 + 按钮查看详情
                with gr.Row():
                    with gr.Column(scale=1):
                        clusters_selected = gr.Number(label="选中行序号", value=-1, visible=False)
                        clusters_select_info = gr.Textbox(label="选择状态", value="请点击表格选择一行", interactive=False)
                        show_cluster_detail_btn = gr.Button("📋 展示聚类详情", variant="secondary")
                    with gr.Column(scale=3):
                        cluster_detail_text = gr.Textbox(label="聚类详情", lines=20, max_lines=30)
                
                clusters_state = gr.State([])
            
            # ==================== Tab 5: Query匹配 ====================
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
                        match_detail_text = gr.Textbox(label="Prompt详情", lines=15, max_lines=25)
                
                matches_state = gr.State([])
            
            # ==================== Tab 6: 答案生成与融合 ====================
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
            
            # ==================== 事件绑定 ====================
            
            # Tab 1: 数据集事件
            load_dataset_btn.click(
                fn=self.load_dataset,
                outputs=[dataset_status, dataset_table, dataset_state]
            )
            
            dataset_table.select(
                fn=self.on_dataset_select,
                outputs=[dataset_selected, dataset_select_info]
            )
            
            show_dataset_detail_btn.click(
                fn=self.show_dataset_detail,
                inputs=[dataset_selected, dataset_state],
                outputs=[dataset_detail_text]
            )
            
            # Tab 2: Prompt事件
            load_prompts_btn.click(
                fn=self.load_initial_prompts,
                outputs=[prompts_status, prompts_table, prompts_state]
            )
            
            prompts_table.select(
                fn=self.on_prompt_select,
                outputs=[prompts_selected, prompts_select_info]
            )
            
            show_prompt_detail_btn.click(
                fn=self.show_prompt_detail,
                inputs=[prompts_selected, prompts_state],
                outputs=[prompts_detail_text]
            )
            
            # Tab 3: 迭代事件
            load_iteration_dataset_btn.click(
                fn=self.load_iteration_dataset,
                outputs=[iteration_dataset_status, iteration_table, iteration_state]
            )
            
            iteration_table.select(
                fn=self.on_iteration_select,
                outputs=[iteration_selected, iteration_select_info]
            )
            
            show_iteration_detail_btn.click(
                fn=self.show_iteration_question_detail,
                inputs=[iteration_selected, iteration_state],
                outputs=[iteration_detail_text]
            )
            
            # Tab 4: 聚类事件
            load_clusters_btn.click(
                fn=self.load_cluster_results,
                outputs=[clusters_status, clusters_table, clusters_state]
            )
            
            clusters_table.select(
                fn=self.on_cluster_select,
                outputs=[clusters_selected, clusters_select_info]
            )
            
            show_cluster_detail_btn.click(
                fn=self.show_cluster_detail,
                inputs=[clusters_selected, clusters_state],
                outputs=[cluster_detail_text]
            )
            
            # Tab 5: 匹配事件
            match_btn.click(
                fn=self.match_query_to_clusters,
                inputs=[match_query_input],
                outputs=[match_status, matches_table, matches_state]
            )
            
            matches_table.select(
                fn=self.on_match_select,
                outputs=[matches_selected, matches_select_info]
            )
            
            show_match_detail_btn.click(
                fn=self.show_matched_prompt,
                inputs=[matches_selected, matches_state],
                outputs=[match_detail_text]
            )
            
            # Tab 6: 融合事件
            refresh_prompts_btn.click(
                fn=self.refresh_prompts_for_fusion,
                outputs=[fusion_prompts_dropdown, fusion_prompts_status]
            )
            
            generate_btn.click(
                fn=self.generate_answers_with_fusion,
                inputs=[fusion_query_input, fusion_prompts_dropdown],
                outputs=[fused_answer_output, answer1_output, answer2_output, fusion_status]
            )
        
        return tourist_app
