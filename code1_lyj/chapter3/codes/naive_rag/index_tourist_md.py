"""
景区MD文件向量化脚本
将tourist_project目录下的所有景区介绍和运营信息.md文件向量化到Faiss向量库
"""
import sys
from pathlib import Path

# 添加core目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'core'))

import os
import re
import pickle
from tqdm import tqdm

from naive_rag import NaiveRAG
from vector_store import get_vector_store
from chunker import chunker
from embedder import embedder
import config


# 数据目录
DATA_DIR = Path(r"i:\bylw_final\Code\chapter3\datasets\tourist_project")
VECTOR_DB_DIR = Path(r"i:\bylw_final\Code\chapter3\codes\naive_rag\vector_dbs\tourist")


def parse_introduction_md(file_path: Path) -> list:
    """
    解析景区介绍markdown文件
    支持两种格式:
    1. "POI_ID": "景点名称" + 分隔线 + 内容 (南孔庙格式)
    2. 标题\n内容 (其他景区格式)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    documents = []
    scenic_area = file_path.parent.name
    
    # 尝试格式1: "id": "名称",\n分隔线(30个-)\n内容\n分隔线(30个-)
    pattern1 = r'"([^"]+)":\s*"([^"]+)",?\s*\n-{30}\n\n?(.*?)(?:\n-{30}\n|\Z)'
    matches = re.findall(pattern1, content, re.DOTALL)
    
    if matches:
        # 格式1: 南孔庙格式
        for poi_id, poi_name, poi_content in matches:
            doc = {
                'chunk_id': f"{scenic_area}_{poi_id}",
                'text': f"【{poi_name}】\n{poi_content.strip()}",
                'metadata': {
                    'doc_id': f"{scenic_area}_{poi_id}",
                    'title': poi_name.strip(),
                    'source_file': file_path.name,
                    'scenic_area': scenic_area,
                    'doc_type': 'introduction'
                }
            }
            documents.append(doc)
    else:
        # 格式2: 其他景区格式，按段落分割
        # 首先尝试按 "景点名称\n内容" 的格式分割
        # 查找类似 "少林寺\n少林始建于..." 或 "少林塔林\n少林塔林位于..." 的模式
        pattern2 = r'^([^\n]{2,20})\n([^\n].*?)(?=\n[^\n]{2,20}\n|\Z)'
        matches2 = re.findall(pattern2, content, re.MULTILINE | re.DOTALL)
        
        if matches2:
            for i, (poi_name, poi_content) in enumerate(matches2):
                poi_name = poi_name.strip()
                poi_content = poi_content.strip()
                if len(poi_content) > 50:  # 过滤掉太短的段落
                    doc = {
                        'chunk_id': f"{scenic_area}_POI_{i:03d}",
                        'text': f"【{poi_name}】\n{poi_content}",
                        'metadata': {
                            'doc_id': f"{scenic_area}_POI_{i:03d}",
                            'title': poi_name,
                            'source_file': file_path.name,
                            'scenic_area': scenic_area,
                            'doc_type': 'introduction'
                        }
                    }
                    documents.append(doc)
        
        # 如果没有匹配到，将整个文件作为一个文档
        if not documents and len(content.strip()) > 100:
            doc = {
                'chunk_id': f"{scenic_area}_intro",
                'text': content.strip(),
                'metadata': {
                    'doc_id': f"{scenic_area}_intro",
                    'title': f"{scenic_area}景区介绍",
                    'source_file': file_path.name,
                    'scenic_area': scenic_area,
                    'doc_type': 'introduction'
                }
            }
            documents.append(doc)
    
    return documents


def parse_operation_md(file_path: Path) -> list:
    """
    解析运营信息markdown文件
    格式: 纯文本段落，包含景区运营相关信息
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    documents = []
    scenic_area = file_path.parent.name
    
    # 将运营信息按句子切分，每个句子作为一个文档
    # 首先提取所有有用信息
    sentences = re.split(r'(?<=。)(?!["」])', content)
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if len(sentence) > 10:  # 过滤掉太短的句子
            doc = {
                'chunk_id': f"{scenic_area}_operation_{i:04d}",
                'text': sentence,
                'metadata': {
                    'doc_id': f"{scenic_area}_operation",
                    'title': f"{scenic_area}运营信息",
                    'source_file': file_path.name,
                    'scenic_area': scenic_area,
                    'doc_type': 'operation'
                }
            }
            documents.append(doc)
    
    return documents


def get_all_md_files(data_dir: Path) -> dict:
    """获取所有景区介绍和运营信息.md文件，按景区分类"""
    md_files = {
        'introduction': [],
        'operation': []
    }
    
    # 遍历所有子目录
    for scenic_dir in data_dir.iterdir():
        if scenic_dir.is_dir():
            for md_file in scenic_dir.glob("*.md"):
                # 分类处理
                if "介绍" in md_file.name:
                    md_files['introduction'].append(md_file)
                elif "运营" in md_file.name:
                    md_files['operation'].append(md_file)
    
    return md_files


def delete_existing_vector_db():
    """删除现有的tourist向量库"""
    if VECTOR_DB_DIR.exists():
        print(f"\n删除现有向量库: {VECTOR_DB_DIR}")
        import shutil
        shutil.rmtree(VECTOR_DB_DIR)
        print("✓ 已删除")


def main():
    print("=" * 70)
    print("景区MD文件向量化")
    print("=" * 70)
    
    # 1. 删除现有向量库
    delete_existing_vector_db()
    
    # 2. 获取所有md文件
    print("\n[1/4] 扫描文件...")
    md_files = get_all_md_files(DATA_DIR)
    print(f"✓ 找到 {len(md_files['introduction'])} 个介绍文件, {len(md_files['operation'])} 个运营文件")
    
    for f in md_files['introduction']:
        print(f"  [介绍] {f.parent.name}/{f.name}")
    for f in md_files['operation']:
        print(f"  [运营] {f.parent.name}/{f.name}")
    
    # 3. 解析所有文件
    print("\n[2/4] 解析文件内容...")
    all_documents = []
    
    # 解析介绍文件
    for md_file in tqdm(md_files['introduction'], desc="解析介绍文件"):
        docs = parse_introduction_md(md_file)
        all_documents.extend(docs)
        print(f"  ✓ {md_file.parent.name}/{md_file.name}: {len(docs)} 个文档")
    
    # 解析运营文件
    for md_file in tqdm(md_files['operation'], desc="解析运营文件"):
        docs = parse_operation_md(md_file)
        all_documents.extend(docs)
        print(f"  ✓ {md_file.parent.name}/{md_file.name}: {len(docs)} 个文档")
    
    print(f"\n✓ 总共解析出 {len(all_documents)} 个文档")
    
    if len(all_documents) == 0:
        print("错误: 没有解析到任何文档")
        return
    
    # 4. 初始化向量库
    print("\n[3/4] 初始化向量库...")
    vector_store = get_vector_store('tourist')
    print(f"✓ 向量库初始化完成")
    
    # 5. 编码并添加到向量库
    print("\n[4/4] 向量化文档...")
    print(f"  文档总数: {len(all_documents)}")
    print(f"  编码批次大小: {config.EMBEDDING_BATCH_SIZE}")
    
    # 分批编码
    batch_size = config.EMBEDDING_BATCH_SIZE
    total_chunks = len(all_documents)
    
    for i in tqdm(range(0, total_chunks, batch_size), desc="编码进度"):
        batch = all_documents[i:i + batch_size]
        
        # 编码文本
        texts = [doc['text'] for doc in batch]
        embeddings = embedder.encode(texts, batch_size=len(batch), show_progress=False)
        
        # 添加到向量库
        vector_store.add_documents(batch)
    
    print(f"\n✓ 向量化完成!")
    print(f"  - 总文档数: {len(vector_store.metadata)}")
    print(f"  - 存储路径: {vector_store.persist_dir}")
    
    # 6. 测试检索
    print("\n" + "=" * 70)
    print("测试检索")
    print("=" * 70)
    
    test_queries = [
        "南孔庙有哪些景点",
        "少林寺的开放时间",
        "张家界门票价格",
        "颐和园怎么预约",
        "西湖游船多少钱"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        results = vector_store.search(query, top_k=3)
        print(f"返回 {len(results)} 个结果:")
        for j, r in enumerate(results, 1):
            text = r['text'][:80] + "..." if len(r['text']) > 80 else r['text']
            print(f"  {j}. [相似度: {r['score']:.4f}] {text}")
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
