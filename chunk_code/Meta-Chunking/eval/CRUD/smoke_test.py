"""
端到端冒烟测试 - 验证 Milvus 索引、检索、LLM 调用、评分全流程通畅。
仅使用 split_merged.json 中的 5 个样本进行测试。
"""
import os, sys, json
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=r'c:\Users\胡铭强\Desktop\chunk_code\data\split_merged.json')
parser.add_argument('--docs_path', default=r'c:\Users\胡铭强\Desktop\chunk_code\data\chunks_txt')
parser.add_argument('--docs_type', default='txt')
parser.add_argument('--chunk_size', type=int, default=128)
parser.add_argument('--chunk_overlap', type=int, default=0)
parser.add_argument('--construct_index', action='store_true', default=False)
parser.add_argument('--add_index', action='store_true', default=False)
parser.add_argument('--collection_name', default='meta_chunks_smoke_test')
parser.add_argument('--retrieve_top_k', type=int, default=4)
parser.add_argument('--retriever_name', default='base')
parser.add_argument('--embedding_dim', type=int, default=768)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--max_new_tokens', type=int, default=512)
parser.add_argument('--num_threads', type=int, default=2)
parser.add_argument('--task', default='quest_answer')
parser.add_argument('--bert_score_eval', action='store_true', default=True)
parser.add_argument('--limit', type=int, default=3, help='限制每个子集测试样本数')
args = parser.parse_args()

print('='*60)
print('冒烟测试 - 端到端流程验证')
print('='*60)

# 构造一个限制样本的小数据集
print('\n[1/5] 加载数据集（限制每个子集 %d 条样本）...' % args.limit)
with open(args.data_path, 'r', encoding='utf-8') as f:
    full_data = json.load(f)
sampled = {}
for k, v in full_data.items():
    sampled[k] = v[:args.limit]
print('  - questanswer_1doc: %d' % len(sampled.get('questanswer_1doc', [])))
print('  - questanswer_2docs: %d' % len(sampled.get('questanswer_2docs', [])))
print('  - questanswer_3docs: %d' % len(sampled.get('questanswer_3docs', [])))

# 写一个临时小数据集
tmp_data = r'c:\Users\胡铭强\Desktop\chunk_code\data\split_smoke.json'
with open(tmp_data, 'w', encoding='utf-8') as f:
    json.dump(sampled, f, ensure_ascii=False, indent=2)
args.data_path = tmp_data
print('  临时数据集已保存到:', tmp_data)

# 注入参数
sys.argv = [
    'quick_start.py',
    '--model_name', 'qwen_api',
    '--data_path', args.data_path,
    '--docs_path', args.docs_path,
    '--docs_type', args.docs_type,
    '--chunk_size', str(args.chunk_size),
    '--chunk_overlap', str(args.chunk_overlap),
    '--collection_name', args.collection_name,
    '--retrieve_top_k', str(args.retrieve_top_k),
    '--retriever_name', args.retriever_name,
    '--embedding_dim', str(args.embedding_dim),
    '--temperature', str(args.temperature),
    '--max_new_tokens', str(args.max_new_tokens),
    '--num_threads', str(args.num_threads),
    '--task', args.task,
]
if args.construct_index:
    sys.argv.append('--construct_index')
if args.add_index:
    sys.argv.append('--add_index')
if args.bert_score_eval:
    sys.argv.append('--bert_score_eval')

print('\n[2/5] 准备 Retriever & LLM ...')
from loguru import logger
from src.llms import Qwen_API_Chat
from src.embeddings.base import HuggingfaceEmbeddings
from src.retrievers import BaseRetriever
from src.datasets.xinhua import get_task_datasets
from src.tasks.quest_answer import QuestAnswer1Doc, QuestAnswer2Docs, QuestAnswer3Docs
from evaluator import BaseEvaluator
from importlib import import_module
conf = import_module('src.configs.real_config')

llm = Qwen_API_Chat(model_name='qwen_api', temperature=args.temperature, max_new_tokens=args.max_new_tokens)
print('  - LLM 初始化 OK:', conf.Qwen_OpenAI_Model_Name)

embed_model = HuggingfaceEmbeddings(model_name=conf.BGE_model_name)
print('  - Embedding 模型加载 OK:', conf.BGE_model_name)

print('\n[3/5] 构建 Retriever ...')
retriever = BaseRetriever(
    args.docs_path, embed_model=embed_model, embed_dim=args.embedding_dim,
    chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
    construct_index=args.construct_index, add_index=args.add_index,
    collection_name=args.collection_name, similarity_top_k=args.retrieve_top_k,
)
print('  - Retriever 初始化 OK, top_k =', args.retrieve_top_k)

print('\n[4/5] 加载 Task ...')
tasks = [
    QuestAnswer1Doc(use_quest_eval=False, use_bert_score=args.bert_score_eval),
    QuestAnswer2Docs(use_quest_eval=False, use_bert_score=args.bert_score_eval),
    QuestAnswer3Docs(use_quest_eval=False, use_bert_score=args.bert_score_eval),
]
print('  - 任务数:', len(tasks))

print('\n[5/5] 加载数据集 & 运行评测 ...')
datasets = get_task_datasets(args.data_path, 'quest_answer')
print('  - 数据子集:', len(datasets))

print()
print('='*60)
print('开始运行端到端测试...')
print('='*60)

for task, dataset in zip(tasks, datasets):
    print('\n>>> Task:', task.__class__.__name__)
    evaluator = BaseEvaluator(task, llm, retriever, dataset, num_threads=args.num_threads)
    evaluator.run(show_progress_bar=True, contain_original_data=False)
    print('-'*60)

print('\n冒烟测试全部通过！')
