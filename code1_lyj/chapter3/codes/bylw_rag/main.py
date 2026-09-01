"""
毕业论文RAG系统主程序
实现第三章描述的动态Prompt迭代RAG系统
"""
import argparse
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core.rag_system import BYLWRAGSystem
from core.prompt_library import PromptLibrary


def main():
    parser = argparse.ArgumentParser(
        description='毕业论文RAG系统 - 动态Prompt迭代优化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 初始化Prompt库
  python main.py --init --question-type fact_retrieval --domain general
  
  # 单次问答
  python main.py --dataset nq_validation --question-type fact_retrieval --query "什么是RAG?"
  
  # 带标准答案的评估
  python main.py --dataset nq_validation --query "什么是RAG?" --gold-answer "检索增强生成"
  
  # 查看统计信息
  python main.py --stats
        """
    )
    
    # 基本参数
    parser.add_argument('--dataset', type=str, default='nq_validation',
                        help='数据集名称（选择向量库）')
    parser.add_argument('--question-type', type=str, 
                        default='fact_retrieval',
                        choices=['fact_retrieval', 'subjective_opinion', 
                                'exploratory_open', 'short_answer'],
                        help='问题类型')
    parser.add_argument('--domain', type=str, default='general',
                        choices=['general', 'psychology', 'computer_science', 'medicine'],
                        help='领域')
    
    # 操作模式
    parser.add_argument('--init', action='store_true',
                        help='初始化Prompt库')
    parser.add_argument('--query', type=str,
                        help='查询问题')
    parser.add_argument('--gold-answer', type=str,
                        help='标准答案（用于评估）')
    parser.add_argument('--max-iterations', type=int, default=2,
                        help='最大迭代次数')
    parser.add_argument('--no-iterate', action='store_true',
                        help='禁用自动迭代')
    
    # 其他选项
    parser.add_argument('--stats', action='store_true',
                        help='显示统计信息')
    parser.add_argument('--list-prompts', action='store_true',
                        help='列出所有Prompt')
    parser.add_argument('--save-session', type=str,
                        help='保存会话到文件')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("毕业论文RAG系统 - 动态Prompt迭代优化")
    print("=" * 70)
    
    # 初始化模式
    if args.init:
        print(f"\n初始化Prompt库...")
        print(f"问题类型: {args.question_type}")
        print(f"领域: {args.domain}")
        
        rag = BYLWRAGSystem(
            dataset_name=args.dataset,
            question_type=args.question_type,
            domain=args.domain
        )
        rag.initialize_prompts()
        
        print("\n✓ Prompt库初始化完成")
        return
    
    # 显示统计信息
    if args.stats:
        library = PromptLibrary()
        stats = library.get_prompt_stats()
        
        print("\nPrompt库统计信息:")
        print(f"  总Prompt数: {stats['total_prompts']}")
        print("\n  按类型分布:")
        for qtype, info in stats['by_type'].items():
            print(f"    {qtype}: {info['count']}个 (激活: {info['active']}, 平均分: {info['avg_score']:.2f})")
        return
    
    # 列出Prompt
    if args.list_prompts:
        library = PromptLibrary()
        
        print("\nPrompt列表:")
        for qtype in library.QUESTION_TYPES:
            prompts = library.get_prompts(qtype)
            if prompts:
                print(f"\n  [{qtype}]:")
                for p in prompts:
                    status = "✓" if p.is_active else "✗"
                    score = f"({p.performance_score:.1f})" if p.performance_score else "(N/A)"
                    print(f"    {status} {p.name} {score} - {p.prompt_id}")
        return
    
    # 问答模式
    if args.query:
        # 初始化RAG系统
        rag = BYLWRAGSystem(
            dataset_name=args.dataset,
            question_type=args.question_type,
            domain=args.domain
        )
        
        # 运行问答周期
        response = rag.run_qa_cycle(
            question=args.query,
            gold_answer=args.gold_answer,
            max_iterations=args.max_iterations
        )
        
        # 保存会话
        if args.save_session:
            rag.save_session(args.save_session)
        
        return
    
    # 交互模式
    print("\n进入交互模式（输入'quit'退出，输入'stats'查看统计）")
    print("-" * 70)
    
    # 初始化RAG系统
    rag = BYLWRAGSystem(
        dataset_name=args.dataset,
        question_type=args.question_type,
        domain=args.domain
    )
    
    while True:
        try:
            question = input("\n请输入问题: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            if question.lower() == 'stats':
                stats = rag.get_stats()
                print("\n系统状态:")
                print(f"  当前Prompt: {stats['current_prompt']}")
                print(f"  迭代次数: {stats['iteration_count']}")
                continue
            
            if not question:
                continue
            
            # 运行问答
            response = rag.run_qa_cycle(
                question=question,
                max_iterations=args.max_iterations
            )
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    main()
