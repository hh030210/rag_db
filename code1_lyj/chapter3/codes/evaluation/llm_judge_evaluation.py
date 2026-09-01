"""
LLM as Judge 评估脚本
对多个模型的RAG结果进行4维度评估：准确性、完整性、相关性、上下文利用率

支持两种数据格式：
1. naive_rag格式：detailed_results.json，包含results数组
2. bylw_rag格式：每个问题单独的json文件
"""

import json
import os
import re
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
import numpy as np


# LLM评估Prompt（4维度，无加权，无overall）
EVALUATION_PROMPT_TEMPLATE = """你是一位专业的信息质量评估专家。请对以下生成的答案进行多维度评估。

【问题】
{question}

【上下文信息】
{context}

【标准答案】
{reference_answer}

【生成答案】
{generated_answer}

请从以下4个维度进行评估（0-5分，保留一位小数）：

1. **准确性 (accuracy)**: 
   - 答案中的事实、数据、信息是否正确无误
   - 与标准答案对比，有无错误信息
   - 数值、时间、地点等关键信息是否准确

2. **完整性 (completeness)**: 
   - 是否完整回答了问题的所有方面
   - 与标准答案相比，有无遗漏重要信息
   - 是否覆盖了问题中的所有关键点

3. **相关性 (relevance)**: 
   - 答案是否与问题紧密相关
   - 有无答非所问或包含无关内容
   - 是否直接针对用户的问题给出回答

4. **上下文利用率 (context_utilization)**: 
   - 是否正确使用了提供的上下文信息
   - 是否基于上下文中的事实回答，而非编造或依赖模型内部知识
   - 引用上下文内容是否准确、恰当
   - 当上下文信息不足时，是否合理处理（如说明不确定）

请以JSON格式返回评估结果：
{
  "accuracy": 分数,
  "completeness": 分数,
  "relevance": 分数,
  "context_utilization": 分数,
  "reasoning": "详细说明各维度的评分理由，引用生成答案中的具体内容进行对比分析，特别说明上下文使用情况"
}

评分标准：
- 5分 = 优秀：完全符合要求，无瑕疵
- 4分 = 良好：基本符合，有小问题
- 3分 = 一般：部分符合，有明显不足
- 2分 = 较差：勉强相关，问题较多
- 1分 = 很差：严重偏离要求
- 0分 = 完全错误/未使用上下文

注意：
- 评估时请将生成答案与标准答案进行对比
- 重点关注生成答案是否忠实于上下文，有无幻觉或编造信息
- 如果生成答案包含上下文外的正确信息，不扣分；但如果与上下文矛盾，则准确性扣分"""


class LLMClient:
    """LLM客户端 - 根据你的实际API配置修改"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model
        
    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2000,
                 max_retries: int = 3) -> Optional[str]:
        """调用LLM生成回复 - 使用硅基流动API格式"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model,
            'messages': [
                {"role": "system", "content": "你是一位专业的信息质量评估专家。请严格按照JSON格式返回评估结果。"},
                {"role": "user", "content": prompt}
            ],
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': False
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url + '/chat/completions',
                    headers=headers,
                    json=data,
                    timeout=120
                )

                if response.status_code != 200:
                    print(f"[错误] API返回状态码 {response.status_code}")
                    print(f"[错误] 响应内容: {response.text[:500]}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return None

                result = response.json()
                return result['choices'][0]['message']['content']
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"[错误] LLM调用失败 (重试{max_retries}次后): {e}")
                    return None

        return None


class LLMJudgeEvaluator:
    """LLM作为评判者的评估器"""
    
    def __init__(self, llm_client: LLMClient, num_threads: int = 5):
        self.llm_client = llm_client
        self.num_threads = num_threads
        
    def parse_evaluation_response(self, response: str) -> Optional[Dict]:
        """解析LLM返回的评估结果"""
        if not response:
            return None
        
        try:
            # 尝试直接解析JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # 验证必要字段
                required_fields = ['accuracy', 'completeness', 'relevance', 'context_utilization']
                if all(field in result for field in required_fields):
                    return result
        except json.JSONDecodeError:
            pass
        
        # 如果解析失败，返回默认评分
        print(f"[警告] 无法解析评估结果，使用默认值")
        return {
            'accuracy': 0.0,
            'completeness': 0.0,
            'relevance': 0.0,
            'context_utilization': 0.0,
            'reasoning': f'解析失败，原始响应: {response[:200]}'
        }
    
    def evaluate_single(self, question: str, context: str, reference: str, generated: str) -> Dict:
        """评估单个样本"""
        # 手动替换变量，避免花括号转义问题
        prompt = EVALUATION_PROMPT_TEMPLATE.replace('{question}', question)
        prompt = prompt.replace('{context}', context)
        prompt = prompt.replace('{reference_answer}', reference)
        prompt = prompt.replace('{generated_answer}', generated)
        
        response = self.llm_client.generate(prompt)
        result = self.parse_evaluation_response(response)
        
        if result is None:
            return {
                'accuracy': 0.0,
                'completeness': 0.0,
                'relevance': 0.0,
                'context_utilization': 0.0,
                'reasoning': '评估失败'
            }
        
        return result
    
    def _get_unique_filename(self, details_dir: Path, base_filename: str, used_ids: set) -> str:
        """生成唯一的文件名，处理重复的question_id"""
        if base_filename not in used_ids:
            used_ids.add(base_filename)
            return base_filename
        
        # 如果已存在，添加后缀 _1, _2, ...
        counter = 1
        while f"{base_filename}_{counter}" in used_ids:
            counter += 1
        new_filename = f"{base_filename}_{counter}"
        used_ids.add(new_filename)
        return new_filename
    
    def _find_latest_details_dir(self, output_dir: Path, current_dir: Path = None) -> Optional[Path]:
        """找到最近一次的evaluate_details文件夹（排除当前正在创建的文件夹）"""
        if not output_dir.exists():
            return None
        
        details_dirs = []
        for item in output_dir.iterdir():
            if item.is_dir():
                # 跳过当前正在创建的文件夹
                if current_dir and item == current_dir:
                    continue
                    
                if item.name == 'evaluate_details':
                    # 旧格式文件夹，使用修改时间
                    mtime = item.stat().st_mtime
                    details_dirs.append((mtime, item))
                elif item.name.startswith('evaluate_details_'):
                    # 新格式文件夹，带时间戳
                    try:
                        timestamp_str = item.name.replace('evaluate_details_', '')
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        details_dirs.append((timestamp.timestamp(), item))
                    except ValueError:
                        # 如果解析失败，使用修改时间
                        mtime = item.stat().st_mtime
                        details_dirs.append((mtime, item))
        
        if not details_dirs:
            return None
        
        # 按时间排序，返回最新的
        details_dirs.sort(reverse=True)
        return details_dirs[0][1]
    
    def _load_existing_results(self, details_dir: Path) -> Dict[str, List[Dict]]:
        """从已有的details文件夹加载评估结果，按question_id分组"""
        existing_results = {}
        if not details_dir.exists():
            return existing_results
        
        for json_file in details_dir.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    question_id = data.get('question_id', '')
                    if question_id:
                        if question_id not in existing_results:
                            existing_results[question_id] = []
                        existing_results[question_id].append(data)
            except Exception as e:
                print(f"[警告] 加载已有结果文件失败 {json_file}: {e}")
                continue
        
        return existing_results
    
    def _copy_existing_files(self, src_dir: Path, dst_dir: Path, used_filenames: set) -> int:
        """复制已有文件到新文件夹，返回复制的文件数"""
        if not src_dir.exists():
            return 0
        
        copied = 0
        for json_file in src_dir.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 使用原始文件名（从文件路径获取）
                filename = json_file.stem
                
                # 直接复制，保持文件名不变
                used_filenames.add(filename)
                
                # 更新文件名字段并保存
                data['filename'] = filename
                dst_file = dst_dir / f'{filename}.json'
                with open(dst_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                copied += 1
            except Exception as e:
                print(f"[警告] 复制文件失败 {json_file}: {e}")
                continue
        
        return copied
    
    def evaluate_batch(self, samples: List[Dict], model_name: str = "", model_idx: int = 0, 
                        total_models: int = 0, output_dir: Path = None) -> List[Dict]:
        """批量评估样本 - 支持断点续传，实时保存每个样本的结果"""
        results = []
        total_samples = len(samples)
        
        # 显示模型总体进度
        model_progress = f"[{model_idx}/{total_models}]" if total_models > 0 else ""
        print(f"\n{model_progress} 开始评估模型 '{model_name}' 的 {total_samples} 个样本 (线程数: {self.num_threads})")
        
        # 创建带时间戳的evaluate_details文件夹
        if output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            details_dir = output_dir / f'evaluate_details_{timestamp}'
            details_dir.mkdir(parents=True, exist_ok=True)
            print(f"  详细结果将保存到: {details_dir}")
        else:
            details_dir = None
        
        # 用于跟踪已使用的文件名
        used_filenames = set()
        
        # 查找并复制上次的评估结果（断点续传）
        skipped_count = 0
        if output_dir and details_dir:
            latest_dir = self._find_latest_details_dir(output_dir, details_dir)
            if latest_dir:
                print(f"  发现上次的评估结果: {latest_dir}")
                copied = self._copy_existing_files(latest_dir, details_dir, used_filenames)
                skipped_count = copied
                print(f"  已复制 {copied} 个已有结果，跳过重新评估")
        
        # 加载已有结果用于判断哪些样本需要评估
        existing_results = {}
        if details_dir:
            existing_results = self._load_existing_results(details_dir)
        
        # 跟踪每个question_id已经使用了多少个已有结果
        used_existing_counts = {}
        
        # 过滤出需要评估的样本
        samples_to_evaluate = []
        for idx, sample in enumerate(samples):
            question_id = str(sample.get('question_id', ''))
            
            # 检查是否已有结果（按顺序匹配）
            found_existing = False
            if question_id in existing_results:
                existing_list = existing_results[question_id]
                used_count = used_existing_counts.get(question_id, 0)
                
                if used_count < len(existing_list):
                    # 使用已有结果
                    existing_data = existing_list[used_count]
                    results.append({
                        'question_id': question_id,
                        'question': sample['question'],
                        'evaluation': existing_data.get('evaluation', {})
                    })
                    used_existing_counts[question_id] = used_count + 1
                    found_existing = True
            
            if not found_existing:
                # 需要重新评估，生成新的文件名
                base_filename = re.sub(r'[\\/*?:"<>|]', '_', question_id) if question_id else f"sample_{idx+1}"
                unique_filename = self._get_unique_filename_for_resume(base_filename, used_filenames)
                samples_to_evaluate.append((idx + 1, sample, unique_filename))
        
        remaining_count = len(samples_to_evaluate)
        completed = len(results)
        
        print(f"  总计: {total_samples}, 已有结果: {completed}, 待评估: {remaining_count}")
        
        if remaining_count > 0:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {}
                for idx, sample, unique_filename in samples_to_evaluate:
                    future = executor.submit(
                        self.evaluate_single,
                        sample['question'],
                        sample['context'],
                        sample['reference'],
                        sample['generated']
                    )
                    futures[future] = (idx, sample, unique_filename)
                
                # 使用tqdm显示进度条
                pbar = tqdm(total=remaining_count, desc=f"评估 {model_name}")
                
                for future in as_completed(futures):
                    idx, sample, unique_filename = futures[future]
                    question_id = str(sample.get('question_id', ''))
                    
                    try:
                        eval_result = future.result()
                        result_item = {
                            'question_id': question_id,
                            'question': sample['question'],
                            'evaluation': eval_result
                        }
                        results.append(result_item)
                        completed += 1
                        
                        # 实时保存到文件
                        if details_dir:
                            detail_file = details_dir / f'{unique_filename}.json'
                            detail_data = {
                                'model_name': model_name,
                                'question_id': question_id,
                                'filename': unique_filename,
                                'question': sample.get('question', ''),
                                'context': sample.get('context', ''),
                                'reference_answer': sample.get('reference', ''),
                                'generated_answer': sample.get('generated', ''),
                                'evaluation': eval_result,
                                'timestamp': datetime.now().isoformat()
                            }
                            with open(detail_file, 'w', encoding='utf-8') as f:
                                json.dump(detail_data, f, ensure_ascii=False, indent=2)
                        
                        # 每10个样本打印一次进度
                        if completed % 10 == 0 or completed == total_samples:
                            print(f"\r{model_progress} [{model_name}] 进度: {completed}/{total_samples} ({completed/total_samples*100:.1f}%)", end='', flush=True)
                        
                    except Exception as e:
                        print(f"\n[错误] 评估样本 {question_id} 失败: {e}")
                        error_result = {
                            'accuracy': 0.0,
                            'completeness': 0.0,
                            'relevance': 0.0,
                            'context_utilization': 0.0,
                            'reasoning': f'评估异常: {str(e)}'
                        }
                        result_item = {
                            'question_id': question_id,
                            'question': sample['question'],
                            'evaluation': error_result
                        }
                        results.append(result_item)
                        completed += 1
                        
                        # 实时保存错误结果
                        if details_dir:
                            detail_file = details_dir / f'{unique_filename}.json'
                            detail_data = {
                                'model_name': model_name,
                                'question_id': question_id,
                                'filename': unique_filename,
                                'question': sample.get('question', ''),
                                'context': sample.get('context', ''),
                                'reference_answer': sample.get('reference', ''),
                                'generated_answer': sample.get('generated', ''),
                                'evaluation': error_result,
                                'timestamp': datetime.now().isoformat()
                            }
                            with open(detail_file, 'w', encoding='utf-8') as f:
                                json.dump(detail_data, f, ensure_ascii=False, indent=2)
                    
                    pbar.update(1)
                
                pbar.close()
        
        print(f"\n{model_progress} [{model_name}] 评估完成: {completed}/{total_samples}")
        if details_dir:
            print(f"  详细结果已保存到: {details_dir}")
        return results
    
    def _get_unique_filename_for_resume(self, base_filename: str, used_ids: set) -> str:
        """为断点续传生成唯一的文件名（不修改used_ids集合）"""
        if base_filename not in used_ids:
            return base_filename
        
        counter = 1
        while f"{base_filename}_{counter}" in used_ids:
            counter += 1
        return f"{base_filename}_{counter}"


class ResultLoader:
    """结果加载器 - 支持两种格式"""
    
    @staticmethod
    def load_naive_rag_results(result_path: Path) -> List[Dict]:
        """加载naive_rag格式的detailed_results.json"""
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for result in data.get('results', []):
            # 提取检索的上下文文本
            contexts = []
            for ctx in result.get('retrieved_context', []):
                contexts.append(ctx.get('text', ''))
            
            sample = {
                'question_id': str(result.get('id', '')),
                'question': result.get('query', ''),
                'context': '\n\n'.join(contexts),
                'reference': result.get('ground_truth', ''),
                'generated': result.get('prediction', '')
            }
            samples.append(sample)
        
        return samples
    
    @staticmethod
    def load_bylw_rag_results(result_dir: Path) -> List[Dict]:
        """加载bylw_rag格式的单独json文件"""
        samples = []
        
        for json_file in sorted(result_dir.glob('*.json')):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取检索的上下文文本
                contexts = []
                for ctx in data.get('retrieved_contexts', []):
                    contexts.append(ctx.get('text', ''))
                
                sample = {
                    'question_id': str(data.get('question_id', '')),
                    'question': data.get('question', ''),
                    'context': '\n\n'.join(contexts),
                    'reference': data.get('reference_answer', ''),
                    'generated': data.get('fused_answer', data.get('generated_answers', [''])[0] if data.get('generated_answers') else '')
                }
                samples.append(sample)
            except Exception as e:
                print(f"[警告] 加载文件失败 {json_file}: {e}")
        
        return samples


def calculate_average_scores(evaluations: List[Dict]) -> Dict[str, float]:
    """计算各指标的平均分"""
    if not evaluations:
        return {}
    
    metrics = ['accuracy', 'completeness', 'relevance', 'context_utilization']
    scores = {metric: [] for metric in metrics}
    
    for eval_item in evaluations:
        eval_data = eval_item.get('evaluation', {})
        for metric in metrics:
            score = eval_data.get(metric, 0.0)
            if isinstance(score, (int, float)):
                scores[metric].append(float(score))
    
    averages = {}
    for metric in metrics:
        if scores[metric]:
            averages[metric] = np.mean(scores[metric])
            averages[f'{metric}_std'] = np.std(scores[metric])
        else:
            averages[metric] = 0.0
            averages[f'{metric}_std'] = 0.0
    
    return averages


def save_evaluation_results(output_dir: Path, model_name: str, evaluations: List[Dict], averages: Dict):
    """保存评估结果汇总"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存详细结果（所有评估的汇总JSON）
    detailed_file = output_dir / f'{model_name}_llm_judge_detailed.json'
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': model_name,
            'total_samples': len(evaluations),
            'evaluations': evaluations
        }, f, ensure_ascii=False, indent=2)
    
    # 保存汇总结果
    summary_file = output_dir / f'{model_name}_llm_judge_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': model_name,
            'total_samples': len(evaluations),
            'average_scores': {
                'accuracy': round(averages.get('accuracy', 0), 4),
                'completeness': round(averages.get('completeness', 0), 4),
                'relevance': round(averages.get('relevance', 0), 4),
                'context_utilization': round(averages.get('context_utilization', 0), 4)
            },
            'std_scores': {
                'accuracy': round(averages.get('accuracy_std', 0), 4),
                'completeness': round(averages.get('completeness_std', 0), 4),
                'relevance': round(averages.get('relevance_std', 0), 4),
                'context_utilization': round(averages.get('context_utilization_std', 0), 4)
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_dir}")
    print(f"  - 详细结果: {detailed_file.name}")
    print(f"  - 汇总结果: {summary_file.name}")


def print_summary(model_name: str, averages: Dict, total: int, model_idx: int = 0, total_models: int = 0):
    """打印汇总结果"""
    progress_info = f"[{model_idx}/{total_models}] " if total_models > 0 else ""
    print(f"\n{'='*60}")
    print(f"{progress_info}模型: {model_name}")
    print(f"评估样本数: {total}")
    print(f"{'='*60}")
    print("各指标平均分:")
    print(f"  准确性 (Accuracy):           {averages.get('accuracy', 0):.4f} (±{averages.get('accuracy_std', 0):.4f})")
    print(f"  完整性 (Completeness):       {averages.get('completeness', 0):.4f} (±{averages.get('completeness_std', 0):.4f})")
    print(f"  相关性 (Relevance):          {averages.get('relevance', 0):.4f} (±{averages.get('relevance_std', 0):.4f})")
    print(f"  上下文利用率 (Context Util): {averages.get('context_utilization', 0):.4f} (±{averages.get('context_utilization_std', 0):.4f})")
    print(f"{'='*60}\n")


def parse_arguments():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description='LLM as Judge 评估脚本')
    parser.add_argument('--num_threads', type=int, default=3,
                        help='评估使用的线程数 (默认: 3)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='指定要评估的模型名称，不指定则评估所有模型')
    parser.add_argument('--api_key', type=str, default=None,
                        help='LLM API密钥')
    parser.add_argument('--base_url', type=str, default=None,
                        help='LLM API基础URL')
    parser.add_argument('--model', type=str, default=None,
                        help='评估使用的LLM模型名称 (默认: Qwen/Qwen3-8B)')
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 配置 - 默认使用硅基流动的API配置
    DEFAULT_API_KEY = ''
    DEFAULT_API_URL = 'https://api.siliconflow.cn/v1'
    DEFAULT_MODEL = 'Qwen/Qwen3-8B'
    
    LLM_API_KEY = args.api_key or os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY)
    LLM_BASE_URL = args.base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_API_URL)
    LLM_MODEL = args.model or DEFAULT_MODEL
    NUM_THREADS = args.num_threads
    
    print(f"评估配置:")
    print(f"  - 评估模型: {LLM_MODEL}")
    print(f"  - 线程数: {NUM_THREADS}")
    print(f"  - API地址: {LLM_BASE_URL}")
    print()
    
    # 定义要评估的模型路径
    MODEL_PATHS = {
        # naive_rag格式 (detailed_results.json)
        # 'DeepSeek-R1-Distill-Qwen-7B': {
        #     'path': 'i:/bylw_final/Code/chapter3/codes/naive_rag/experiments/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/20260304-144308-results/detailed_results.json',
        #     'format': 'naive_rag'
        # },
        # 'internlm2_5-7b-chat': {
        #     'path': 'i:/bylw_final/Code/chapter3/codes/naive_rag/experiments/internlm/internlm2_5-7b-chat/20260305-025201-results/detailed_results.json',
        #     'format': 'naive_rag'
        # },
        # 'Qwen2-7B-Instruct': {
        #     'path': 'i:/bylw_final/Code/chapter3/codes/naive_rag/experiments/Qwen/Qwen2-7B-Instruct/20260305-030540-results/detailed_results.json',
        #     'format': 'naive_rag'
        # },
        'Qwen3-8B': {
            'path': 'i:/bylw_final/Code/chapter3/codes/naive_rag/experiments/Qwen3-8B/20260304-041652-results/detailed_results.json',
            'format': 'naive_rag'
        },
        'GLM-4-9B-0414': {
            'path': 'i:/bylw_final/Code/chapter3/codes/naive_rag/experiments/THUDM/GLM-4-9B-0414/20260305-023329-results/detailed_results.json',
            'format': 'naive_rag'
        },
        'GLM-Z1-9B-0414': {
            'path': 'i:/bylw_final/Code/chapter3/codes/naive_rag/experiments/THUDM/GLM-Z1-9B-0414/20260305-013025-results/detailed_results.json',
            'format': 'naive_rag'
        },
        # bylw_rag格式 (单独json文件目录)
        'bylw_rag_top2_fusion': {
            'path': 'i:/bylw_final/Code/chapter3/codes/bylw_rag/new_experiments/results/tourist/top_2_cluster_fusion_20260327_173937',
            'format': 'bylw_rag'
        }
    }
    
    # 输出目录
    OUTPUT_BASE_DIR = Path('i:/bylw_final/Code/chapter3/codes/evaluation/llm_judge_results')
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 初始化LLM客户端和评估器
    llm_client = LLMClient(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, model=LLM_MODEL)
    evaluator = LLMJudgeEvaluator(llm_client, num_threads=NUM_THREADS)
    
    # 存储所有模型的汇总结果
    all_summaries = {}
    
    # 过滤要评估的模型
    if args.models:
        models_to_evaluate = {k: v for k, v in MODEL_PATHS.items() if k in args.models}
        if not models_to_evaluate:
            print(f"[错误] 指定的模型 {args.models} 不在配置中")
            print(f"可用模型: {list(MODEL_PATHS.keys())}")
            return
    else:
        models_to_evaluate = MODEL_PATHS
    
    total_models = len(models_to_evaluate)
    print(f"将要评估 {total_models} 个模型: {list(models_to_evaluate.keys())}")
    print()
    
    # 评估每个模型
    for model_idx, (model_name, config) in enumerate(models_to_evaluate.items(), 1):
        print(f"\n{'='*70}")
        print(f"总体进度: [{model_idx}/{total_models}] 当前模型: {model_name}")
        print(f"{'='*70}")
        
        result_path = Path(config['path'])
        
        # 加载数据
        print(f"\n[1/3] 加载数据...")
        if config['format'] == 'naive_rag':
            samples = ResultLoader.load_naive_rag_results(result_path)
        else:
            samples = ResultLoader.load_bylw_rag_results(result_path)
        
        print(f"加载了 {len(samples)} 个样本")
        
        if not samples:
            print(f"[警告] 没有加载到样本，跳过 {model_name}")
            continue
        
        # 进行评估
        print(f"\n[2/3] 开始LLM评估...")
        model_output_dir = OUTPUT_BASE_DIR / model_name
        evaluations = evaluator.evaluate_batch(samples, model_name=model_name, model_idx=model_idx, 
                                                total_models=total_models, output_dir=model_output_dir)
        
        # 计算平均分
        print(f"\n[3/3] 计算评估指标...")
        averages = calculate_average_scores(evaluations)
        
        # 打印汇总
        print_summary(model_name, averages, len(evaluations), model_idx, total_models)
        
        # 保存结果（汇总文件）
        save_evaluation_results(model_output_dir, model_name, evaluations, averages)
        
        # 记录汇总
        all_summaries[model_name] = {
            'total_samples': len(evaluations),
            'average_scores': {
                'accuracy': round(averages.get('accuracy', 0), 4),
                'completeness': round(averages.get('completeness', 0), 4),
                'relevance': round(averages.get('relevance', 0), 4),
                'context_utilization': round(averages.get('context_utilization', 0), 4)
            }
        }
    
    # 保存所有模型的对比汇总
    comparison_file = OUTPUT_BASE_DIR / 'all_models_comparison.json'
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    
    # 打印对比表格
    print(f"\n{'='*80}")
    print("所有模型对比汇总")
    print(f"{'='*80}")
    print(f"{'模型名称':<30} {'准确性':>10} {'完整性':>10} {'相关性':>10} {'上下文利用':>10}")
    print(f"{'-'*80}")
    for model_name, summary in all_summaries.items():
        scores = summary['average_scores']
        print(f"{model_name:<30} {scores['accuracy']:>10.4f} {scores['completeness']:>10.4f} {scores['relevance']:>10.4f} {scores['context_utilization']:>10.4f}")
    print(f"{'='*80}")
    print(f"\n对比结果已保存到: {comparison_file}")


if __name__ == '__main__':
    main()
