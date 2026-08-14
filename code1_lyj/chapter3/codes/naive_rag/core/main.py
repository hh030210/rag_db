"""
朴素RAG主程序
使用大语言模型进行问答
"""
import argparse
import json
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from naive_rag import NaiveRAG
import config

# 导入模型配置
try:
    from model_config import CURRENT_LLM_DISPLAY_NAME
except ImportError:
    CURRENT_LLM_DISPLAY_NAME = "GLM-4.1V-9B-Thinking"


def main():
    parser = argparse.ArgumentParser(
        description=f'朴素RAG系统 - 使用{CURRENT_LLM_DISPLAY_NAME}大模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 构建索引
  python main.py --dataset nq_test --index data.json
  
  # 单次查询（使用LLM）
  python main.py --dataset nq_test --query "什么是RAG?"
  
  # 禁用LLM，使用简单提取
  python main.py --dataset nq_test --query "什么是RAG?" --no-llm
  
  # 评估数据集
  python main.py --dataset nq_test --index data.json --evaluate --max_eval 100
        """
    )
    parser.add_argument('--dataset', type=str, default='nq_validation',
                        help='数据集名称（用于选择向量库）')
    parser.add_argument('--index', type=str,
                        default=None,
                        help='JSON文件路径，用于构建索引（如果不提供且向量库为空，会报错）')
    parser.add_argument('--query', type=str,
                        help='单个查询问题')
    parser.add_argument('--top_k', type=int, default=5,
                        help='检索的文档块数量')
    parser.add_argument('--reindex', action='store_true',
                        help='重新构建索引（会删除现有索引）')
    parser.add_argument('--evaluate', action='store_true',
                        help='在数据集上评估性能')
    parser.add_argument('--max_eval', type=int, default=100,
                        help='评估时最大样本数')
    parser.add_argument('--save_results', type=str,
                        help='保存结果到指定JSON文件')
    parser.add_argument('--no_save', action='store_true',
                        help='评估时不自动保存结果')
    parser.add_argument('--no-llm', action='store_true',
                        help='禁用LLM，使用简单文本提取（用于测试或节省API调用）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"朴素RAG系统 - {CURRENT_LLM_DISPLAY_NAME}")
    print("=" * 60)
    
    # 初始化RAG系统
    use_llm = not args.no_llm
    rag = NaiveRAG(dataset_name=args.dataset, use_llm=use_llm)
    
    # 如果需要重新索引
    if args.reindex:
        print("\n重新构建索引...")
        rag.vector_store.delete_collection()
        rag = NaiveRAG(dataset_name=args.dataset, use_llm=use_llm)
    
    # 如果提供了--index参数，则添加数据（无论向量库是否为空）
    if args.index:
        existing_count = len(rag.vector_store.metadata)
        if existing_count > 0:
            print(f"\n向量库已有 {existing_count} 条文档，将继续添加新数据...")
        rag.index_documents(args.index)
    elif len(rag.vector_store.metadata) == 0:
        print("错误：向量库为空，请提供--index参数构建索引")
        return
    
    # 评估模式
    if args.evaluate:
        if not args.index:
            print("错误：评估模式需要提供--index参数")
            return
        
        results = rag.evaluate_on_dataset(
            args.index, 
            max_samples=args.max_eval,
            save_results=not args.no_save
        )
        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)
        print(f"总样本数: {results['total']}")
        print(f"正确数: {results['correct']}")
        print(f"EM准确率: {results['accuracy']:.2%}")
        return
    
    # 交互模式或单次查询
    if args.query:
        # 单次查询
        result = rag.answer(args.query, top_k=args.top_k)
        print("\n" + "=" * 60)
        print("查询结果")
        print("=" * 60)
        print(f"问题: {result.question}")
        print(f"\nJSON格式回答:")
        print(json.dumps(result.answer_json, ensure_ascii=False, indent=2))
        
        # 保存结果
        if args.save_results:
            rag.save_results(args.save_results)
    else:
        # 交互模式
        print("\n进入交互模式（输入'quit'退出，输入'save'保存结果）")
        print("-" * 60)
        
        while True:
            question = input("\n请输入问题: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                # 退出前询问是否保存
                if rag.results_history:
                    save = input("是否保存问答结果? (y/n): ").strip().lower()
                    if save == 'y':
                        rag.save_results(args.save_results)
                print("再见！")
                break
            
            if question.lower() == 'save':
                rag.save_results(args.save_results)
                continue
            
            if not question:
                continue
            
            result = rag.answer(question, top_k=args.top_k)
            
            print("\n" + "-" * 60)
            print(f"JSON格式回答:")
            print(json.dumps(result.answer_json, ensure_ascii=False, indent=2))
            print("-" * 60)


if __name__ == "__main__":
    main()
