"""
检查tourist向量库 - 适用于整理后的目录结构
"""
import sys
from pathlib import Path
import os

# 切换到脚本所在目录
script_dir = Path(__file__).parent
os.chdir(script_dir)
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / "core"))

print("=" * 70)
print("Tourist向量库检查")
print("=" * 70)
print(f"当前目录: {os.getcwd()}")
print(f"Python: {sys.executable}")

# 检查GPU
print("\n【1/3】检查GPU...")
try:
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠ 使用CPU模式")
except Exception as e:
    print(f"检查GPU出错: {e}")

# 检查向量库
print("\n【2/3】检查向量库...")
try:
    from vector_store import get_vector_store
    
    vs = get_vector_store('tourist')
    print(f"向量库路径: {vs.persist_dir}")
    print(f"文档块总数: {len(vs.metadata)}")
    
    if len(vs.metadata) > 0:
        from collections import defaultdict
        source_stats = defaultdict(int)
        chunk_lengths = []
        
        for idx, meta in vs.metadata.items():
            text = meta.get('text', '')
            metadata = meta.get('metadata', {})
            doc_id = metadata.get('doc_id', 'unknown')
            source_stats[doc_id] += 1
            chunk_lengths.append(len(text))
        
        print("\n按源文件统计:")
        for source, count in sorted(source_stats.items()):
            print(f"  {source}: {count} 个文档块")
        
        print("\n文档块长度统计:")
        print(f"  平均: {sum(chunk_lengths)/len(chunk_lengths):.0f} 字符")
        print(f"  最大: {max(chunk_lengths)} 字符")
        print(f"  最小: {min(chunk_lengths)} 字符")
    else:
        print("⚠ 向量库为空")
        
except Exception as e:
    print(f"检查向量库出错: {e}")
    import traceback
    traceback.print_exc()

# 测试检索
print("\n【3/3】测试检索...")
try:
    if len(vs.metadata) > 0:
        query = "南孔庙有哪些景点"
        print(f"\n查询: {query}")
        results = vs.search(query, top_k=3)
        print(f"返回 {len(results)} 个结果:")
        for i, r in enumerate(results, 1):
            text = r['text'][:60] + "..." if len(r['text']) > 60 else r['text']
            print(f"  {i}. [{r['score']:.4f}] {text}")
    else:
        print("⚠ 向量库为空，跳过检索测试")
except Exception as e:
    print(f"测试检索出错: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("检查完成!")
print("=" * 70)
