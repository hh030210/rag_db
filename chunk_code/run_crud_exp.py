"""
CRUD 实验：运行你的分片方法
===================================
Step 1: 将你分片好的 chunks 转换为 CRUD 格式（每个 chunk 一个 .txt 文件）
Step 2: 运行 CRUD 评测（建索引 → 召回 → 生成答案 → 评测）

用法:
  Step 1（只需运行一次）:
    python run_crud_exp.py step1 --input ./output_chunks/all_chunks_chunks.json

  Step 2:
    python run_crud_exp.py step2 \
        --docs_path ./crud_data/your_chunks \
        --collection_name your_exp \
        --model_name qwen7b \
        --task quest_answer \
        --construct_index
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "Meta-Chunking" / "eval" / "CRUD"))

import shutil


def step1(args):
    """
    Step 1: 将分片结果转换为 CRUD 格式

    输入: 你的 chunks JSON 文件（每条含 chunk_text）
    输出: docs_directory/，每个 chunk 一行，保存为 chunk_0000.txt, chunk_0001.txt, ...
    """
    output_dir = Path(args.output_dir)
    docs_dir = output_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("[ERROR] chunks 文件为空")
        return

    # 同时生成原始语料（供对比实验）
    raw_dir = output_dir / "docs_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for idx, chunk in enumerate(chunks):
        chunk_text = chunk.get("chunk_text", "").strip()
        if len(chunk_text) < 10:
            continue

        fname = f"chunk_{idx:06d}.txt"
        (docs_dir / fname).write_text(chunk_text, encoding="utf-8")

    # 原始语料：每个原始文档一行（不切分，作为 baseline 对比）
    input_path = Path(args.raw_corpus)
    if input_path.exists():
        raw_lines = [ln.strip() for ln in input_path.read_text(encoding="utf-8").split("\n") if ln.strip() and len(ln.strip()) >= 10]
        for idx, line_text in enumerate(raw_lines):
            fname = f"raw_{idx:06d}.txt"
            (raw_dir / fname).write_text(line_text, encoding="utf-8")
        print(f"[Step1] 原始语料 {len(raw_lines)} 条写入 {raw_dir}")
    else:
        print(f"[Step1] 原始语料文件不存在，跳过: {args.raw_corpus}")

    print(f"[Step1] 完成！分片 {len(list(docs_dir.glob('*.txt')))} 条写入 {docs_dir}")
    print(f"[Step1] 请修改 eval_crud.py 中的 data_path 指向评测数据 JSON（见 README）")


def step2(args):
    """
    Step 2: 运行 CRUD 评测

    核心链路:
      docs_directory（你的 chunks）
        → Milvus 建向量索引
        → 对每个 QA 查询召回 top_k chunks
        → LLM 生成答案
        → BERT-Score / F1 评测

    关键参数说明:
      --docs_path       Step1 生成的 docs 目录路径
      --collection_name Milvus 集合名（不同实验用不同名，避免冲突）
      --construct_index 是否重建索引（首次运行必须加，后续复用可不加）
      --task            任务类型: quest_answer | event_summary | continuing_writing | hallu_modified | all
      --retrieve_top_k  召回 chunks 数量
      --model_name      LLM: qwen7b | gpt-4o | gpt-4o-mini | ...
    """
    print("=" * 60)
    print("CRUD 评测参数:")
    print(f"  docs_path       = {args.docs_path}")
    print(f"  collection_name = {args.collection_name}")
    print(f"  task            = {args.task}")
    print(f"  model_name      = {args.model_name}")
    print(f"  retrieve_top_k  = {args.retrieve_top_k}")
    print(f"  construct_index = {args.construct_index}")
    print("=" * 60)

    # 检查依赖
    try:
        from src.datasets.xinhua import get_task_datasets
        from evaluator import BaseEvaluator
    except ImportError as e:
        print(f"[ERROR] 缺少依赖: {e}")
        print("请确保已安装 CRUD 所需依赖:")
        print("  pip install -r Meta-Chunking/requirements.txt")
        print("  pip install llama-index pymilvus loguru")
        return

    # 参数映射到 quick_start.py 的格式
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=args.model_name)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--max_new_tokens', type=int, default=1280)
    parser.add_argument('--data_path', default=args.data_path)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--embedding_name', default='sentence-transformers/bge-base-zh-v1.5')
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--docs_path', default=args.docs_path)
    parser.add_argument('--docs_type', default="txt")
    parser.add_argument('--chunk_size', type=int, default=128)
    parser.add_argument('--chunk_overlap', type=int, default=0)
    parser.add_argument('--construct_index', action='store_true', default=args.construct_index)
    parser.add_argument('--add_index', action='store_true', default=False)
    parser.add_argument('--collection_name', default=args.collection_name)
    parser.add_argument('--retrieve_top_k', type=int, default=args.retrieve_top_k)
    parser.add_argument('--retriever_name', default="base")
    parser.add_argument('--quest_eval', action='store_true', default=False)
    parser.add_argument('--bert_score_eval', action='store_true', default=True)
    parser.add_argument('--task', default=args.task)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--show_progress_bar', type=lambda x: x.lower() == "true", default=True)
    parser.add_argument('--contain_original_data', action='store_true', default=False)

    from loguru import logger
    from src.llms import GPT, Qwen_7B_Chat
    from src.tasks.summary import Summary
    from src.tasks.continue_writing import ContinueWriting
    from src.tasks.hallucinated_modified import HalluModified
    from src.tasks.quest_answer import QuestAnswer1Doc, QuestAnswer2Docs, QuestAnswer3Docs
    from src.retrievers import BaseRetriever
    from src.embeddings.base import HuggingfaceEmbeddings

    args_cfg = parser.parse_args([])

    # 手动赋值
    class Cfg:
        model_name = args.model_name
        temperature = 0.1
        max_new_tokens = 1280
        data_path = args.data_path
        shuffle = False
        embedding_name = 'sentence-transformers/bge-base-zh-v1.5'
        embedding_dim = 768
        docs_path = args.docs_path
        docs_type = "txt"
        chunk_size = 128
        chunk_overlap = 0
        construct_index = args.construct_index
        add_index = False
        collection_name = args.collection_name
        retrieve_top_k = args.retrieve_top_k
        retriever_name = "base"
        quest_eval = False
        bert_score_eval = True
        task = args.task
        num_threads = 1
        show_progress_bar = True
        contain_original_data = False

    cfg = Cfg()

    logger.info(cfg.__dict__)

    # 初始化 LLM
    if cfg.model_name.startswith("gpt"):
        llm = GPT(model_name=cfg.model_name, temperature=cfg.temperature, max_new_tokens=cfg.max_new_tokens)
    elif cfg.model_name == "qwen7b":
        llm = Qwen_7B_Chat(model_name=cfg.model_name, temperature=cfg.temperature, max_new_tokens=cfg.max_new_tokens)
    else:
        raise ValueError(f"不支持的模型: {cfg.model_name}")

    embed_model = HuggingfaceEmbeddings(model_name=cfg.embedding_name)

    # 初始化 Retriever
    retriever = BaseRetriever(
        cfg.docs_path, embed_model=embed_model, embed_dim=cfg.embedding_dim,
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap,
        construct_index=cfg.construct_index, add_index=cfg.add_index,
        collection_name=cfg.collection_name, similarity_top_k=cfg.retrieve_top_k
    )

    # 任务映射
    task_mapping = {
        'event_summary': [Summary],
        'continuing_writing': [ContinueWriting],
        'hallu_modified': [HalluModified],
        'quest_answer': [QuestAnswer1Doc, QuestAnswer2Docs, QuestAnswer3Docs],
        'all': [Summary, ContinueWriting, HalluModified],
    }

    if cfg.task not in task_mapping:
        raise ValueError(f"Unknown task: {cfg.task}")

    tasks = [task(use_quest_eval=cfg.quest_eval, use_bert_score=cfg.bert_score_eval)
             for task in task_mapping[cfg.task]]

    datasets = get_task_datasets(cfg.data_path, cfg.task)

    for task, dataset in zip(tasks, datasets):
        evaluator = BaseEvaluator(task, llm, retriever, dataset,
                                   num_threads=cfg.num_threads)
        evaluator.run(show_progress_bar=cfg.show_progress_bar,
                     contain_original_data=cfg.contain_original_data)

    print("=" * 60)
    print(f"评测完成！结果查看 Milvus collection: {cfg.collection_name}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="CRUD 实验：运行你的分片方法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

  # Step 1: 将你的 chunks 转换为 CRUD 格式
  python run_crud_exp.py step1 \\
      --input ./output_chunks/all_chunks_chunks.json \\
      --output_dir ./crud_data/your_chunks \\
      --raw_corpus ./data/db_qa.txt

  # Step 2: 运行评测（首次需要 --construct_index）
  python run_crud_exp.py step2 \\
      --docs_path ./crud_data/your_chunks/docs \\
      --docs_raw_path ./crud_data/your_chunks/docs_raw \\
      --data_path ./data/crud_split/split_merged.json \\
      --collection_name your_exp \\
      --model_name qwen7b \\
      --task quest_answer \\
      --construct_index

  # 对比: 用原始语料跑 baseline（不加 --construct_index，复用索引）
  python run_crud_exp.py step2 \\
      --docs_path ./crud_data/your_chunks/docs_raw \\
      --data_path ./data/crud_split/split_merged.json \\
      --collection_name raw_baseline \\
      --model_name qwen7b \\
      --task quest_answer
"""
    )

    subparsers = parser.add_subparsers(dest="step", help="运行步骤")

    # Step1
    p1 = subparsers.add_parser("step1", help="将 chunks 转换为 CRUD 格式")
    p1.add_argument("--input", type=str, required=True,
                    help="你的分片结果 JSON 文件（每项含 chunk_text 字段）")
    p1.add_argument("--output_dir", type=str, required=True,
                    help="输出目录，将创建 <output_dir>/docs/ 和 <output_dir>/docs_raw/")
    p1.add_argument("--raw_corpus", type=str, default="./data/db_qa.txt",
                    help="原始语料文件（每行一个文档，用于 baseline 对比）")

    # Step2
    p2 = subparsers.add_parser("step2", help="运行 CRUD 评测")
    p2.add_argument("--docs_path", type=str, required=True,
                    help="Step1 生成的 docs 目录路径")
    p2.add_argument("--data_path", type=str,
                    default="./Meta-Chunking/eval/CRUD/data/crud_split/split_merged.json",
                    help="CRUD 评测数据 JSON 路径")
    p2.add_argument("--collection_name", type=str, required=True,
                    help="Milvus 集合名（区分不同实验）")
    p2.add_argument("--model_name", type=str, default="qwen7b",
                    help="LLM 模型: qwen7b | gpt-4o | ...")
    p2.add_argument("--task", type=str, default="quest_answer",
                    choices=["quest_answer", "event_summary", "continuing_writing",
                             "hallu_modified", "all"],
                    help="评测任务类型")
    p2.add_argument("--retrieve_top_k", type=int, default=8,
                    help="召回的 chunks 数量")
    p2.add_argument("--construct_index", action="store_true",
                    help="首次运行需加此参数重建索引")

    args = parser.parse_args()

    if args.step == "step1":
        step1(args)
    elif args.step == "step2":
        step2(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
