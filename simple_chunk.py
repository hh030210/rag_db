"""
simple_chunk.py

轻量级分片脚本：将输入文件按段落和字数切分为 chunks，
输出格式与 integrated_chunker 兼容，直接进入 dimension_integration。

用法：
  python simple_chunk.py --input "data_input/insert_Data/南孔文化(吴锡标).txt" --output output_no_ingest/chunks --chunk_size 800
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict

CHUNK_SIZE = 800  # 默认 chunk 大小（字数）
OVERLAP = 50      # 重叠字数


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[Dict]:
    """按段落切分，每段再按字数限制拆分为 chunks"""
    # 先按段落拆分
    para_pattern = re.compile(r'\n\n+|\r\n\r\n+')
    paragraphs = para_pattern.split(text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    chunk_id = 0
    for para in paragraphs:
        if not para:
            continue
        # 如果段落本身超过 chunk_size，再按句子拆
        if len(para) <= chunk_size:
            chunks.append({
                "chunk_id": f"chunk_{chunk_id:06d}",
                "chunk_text": para
            })
            chunk_id += 1
        else:
            # 按句子拆分
            sent_pattern = re.compile(r'[。！？；!?;]')
            sentences = sent_pattern.split(para)
            sentences = [s.strip() for s in sentences if s.strip()]

            current = ""
            for sent in sentences:
                if not sent:
                    continue
                # 如果当前块加上这句话超过限制
                if len(current) + len(sent) + 1 > chunk_size:
                    if current:
                        chunks.append({
                            "chunk_id": f"chunk_{chunk_id:06d}",
                            "chunk_text": current
                        })
                        chunk_id += 1
                    # 新块：取重叠部分 + 当前句
                    current = current[-overlap:] + sent if overlap > 0 and len(current) > overlap else sent
                else:
                    current = (current + "。" + sent) if current else sent
            # 剩余内容
            if current:
                chunks.append({
                    "chunk_id": f"chunk_{chunk_id:06d}",
                    "chunk_text": current
                })
                chunk_id += 1

    return chunks


def process_file(input_path: Path, output_dir: Path, chunk_size: int) -> Dict:
    """处理单个文件"""
    try:
        text = input_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = input_path.read_text(encoding="gbk")

    # 清理多余空白
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    chunks = split_into_chunks(text, chunk_size=chunk_size)
    base_name = input_path.stem

    result = {
        "doc_id": base_name,
        "file_name": input_path.name,
        "genre": "doc",
        "chunk_count": len(chunks),
        "chunks": chunks
    }

    # 保存单个文件结果
    out_file = output_dir / f"{base_name}_chunks.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


def process_directory(input_dir: Path, output_dir: Path, chunk_size: int) -> List[Dict]:
    """处理目录"""
    files = sorted([
        f for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in {".txt", ".md"}
    ])

    print(f"找到 {len(files)} 个文件")
    all_results = []

    for i, f in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {f.name}")
        result = process_file(f, output_dir, chunk_size)
        all_results.append(result)
        print(f"    -> {result['chunk_count']} chunks")

    # 合并所有结果
    merged_chunks = []
    for r in all_results:
        for c in r["chunks"]:
            merged_chunks.append({
                "doc_id": c["chunk_id"],
                "text": c["chunk_text"],
                "source_file": r["file_name"],
                "source_doc_id": r["doc_id"]
            })

    # 保存汇总
    summary = {
        "total_files": len(all_results),
        "total_chunks": len(merged_chunks),
        "files": [{"file_name": r["file_name"], "chunks": r["chunk_count"]} for r in all_results]
    }
    summary_file = output_dir / "all_chunks_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保存合并的 chunks
    chunks_file = output_dir / "all_chunks_chunks.json"
    chunks_file.write_text(json.dumps(merged_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="轻量级分片脚本")
    parser.add_argument("--input", "-i", required=True, help="输入文件或目录")
    parser.add_argument("--output", "-o", default="output_chunks", help="输出目录")
    parser.add_argument("--chunk_size", type=int, default=800, help="每个 chunk 的目标字数")
    parser.add_argument("--overlap", type=int, default=50, help="chunk 之间的重叠字数")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[Error] 路径不存在: {input_path}")
        return 1

    print(f"输入: {input_path}")
    print(f"输出: {output_dir}")
    print(f"chunk_size: {args.chunk_size}")

    if input_path.is_dir():
        results = process_directory(input_path, output_dir, args.chunk_size)
        total_chunks = sum(r["chunk_count"] for r in results)
        print(f"\n完成！总计 {len(results)} 个文件，{total_chunks} 个 chunks")
    else:
        result = process_file(input_path, output_dir, args.chunk_size)
        print(f"\n完成！{result['chunk_count']} 个 chunks")

    print(f"输出目录: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
