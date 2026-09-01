"""
景区数据向量化脚本
将tourist_project目录下的所有景区介绍和运营信息.md文件向量化到Faiss向量库
"""
import sys
from pathlib import Path
# 添加core目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

import os
import re
from tqdm import tqdm

from naive_rag import NaiveRAG
import config

# 数据目录
DATA_DIR = Path(r"i:\bylw_final\Code\chapter3\datasets\tourist_project")

def parse_md_file(file_path: Path) -> list:
    """
    解析markdown文件，提取景点信息
    格式: "POI_ID": "景点名称" + 分隔线 + 内容
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    documents = []
    
    # 使用正则表达式匹配每个景点块
    # 格式: "id": "名称" + 分隔线 + 内容
    pattern = r'"([^"]+)":\s*"([^"]+)"\s*\n[-=]+\n(.*?)\n[-=]+\n'
    matches = re.findall(pattern, content, re.DOTALL)
    
    if matches:
        for poi_id, poi_name, poi_content in matches:
            doc = {
                'id': poi_id.strip(),
                'title': poi_name.strip(),
                'content': poi_content.strip(),
                'source_file': file_path.name,
                'scenic_area': file_path.parent.name
            }
            documents.append(doc)
    else:
        # 如果没有匹配到特定格式，将整个文件作为一个文档
        doc = {
            'id': file_path.stem,
            'title': file_path.stem.replace('-', ' '),
            'content': content,
            'source_file': file_path.name,
            'scenic_area': file_path.parent.name
        }
        documents.append(doc)
    
    return documents

def get_all_md_files(data_dir: Path) -> list:
    """获取所有景区介绍和运营信息.md文件"""
    md_files = []
    
    # 遍历所有子目录
    for scenic_dir in data_dir.iterdir():
        if scenic_dir.is_dir():
            for md_file in scenic_dir.glob("*.md"):
                # 只处理景区介绍和运营信息文件
                if "介绍" in md_file.name or "运营" in md_file.name:
                    md_files.append(md_file)
    
    return sorted(md_files)

def main():
    print("=" * 70)
    print("景区数据向量化")
    print("=" * 70)
    
    # 1. 获取所有md文件
    print("\n[1/4] 扫描文件...")
    md_files = get_all_md_files(DATA_DIR)
    print(f"✓ 找到 {len(md_files)} 个文件:")
    for f in md_files:
        print(f"  - {f.parent.name}/{f.name}")
    
    # 2. 解析所有文件
    print("\n[2/4] 解析文件内容...")
    all_documents = []
    for md_file in tqdm(md_files, desc="解析进度"):
        docs = parse_md_file(md_file)
        all_documents.extend(docs)
        print(f"  ✓ {md_file.parent.name}/{md_file.name}: {len(docs)} 个文档")
    
    print(f"\n✓ 总共解析出 {len(all_documents)} 个文档")
    
    # 3. 初始化RAG系统
    print("\n[3/4] 初始化向量库...")
    rag = NaiveRAG(dataset_name='tourist', use_llm=False)
    
    # 4. 向量化文档
    print("\n[4/4] 向量化文档...")
    
    # 将文档转换为chunker需要的格式
    from chunker import chunker
    
    # 转换文档格式以匹配chunker期望的字段
    formatted_docs = []
    for doc in all_documents:
        formatted_doc = {
            'id': doc['id'],
            'question': doc['title'],  # 使用title作为question
            'document': doc['content'],  # 使用content作为document（这是chunker期望的字段）
            'metadata': {
                'source_file': doc['source_file'],
                'scenic_area': doc['scenic_area'],
                'title': doc['title']
            }
        }
        formatted_docs.append(formatted_doc)
    
    print(f"开始处理 {len(formatted_docs)} 个文档...")
    chunks = chunker.chunk_documents(formatted_docs)
    print(f"✓ 生成 {len(chunks)} 个文档块")
    
    # 添加到向量库
    rag.vector_store.add_documents(chunks)
    
    # 5. 完成统计
    final_count = len(rag.vector_store.metadata)
    print("\n" + "=" * 70)
    print("向量化完成!")
    print("=" * 70)
    print(f"数据集名称: tourist")
    print(f"向量库存储路径: {rag.vector_store.persist_dir}")
    print(f"总文档数: {len(all_documents)}")
    print(f"总文档块数: {final_count}")
    print("=" * 70)

if __name__ == "__main__":
    main()
