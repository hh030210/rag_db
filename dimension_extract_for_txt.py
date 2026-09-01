"""
dimension_extract_for_txt.py

从纯文本 TXT chunks 中自动抽取维度标签（doc_tags）。

关键设计：
  - 直接使用与 test_collection 一致的 15 个固定维度（Milvus 哈希字段名）
  - 避免 LLM 生成维度名导致的字段名映射问题

doc_tags JSON 格式：
  {
    "chunk_id": {
      "dim___d2e6065c": ["南孔庙", "孔氏家庙"],
      "dim___64b4c1a7": ["孔子", "孔端友"]
    }
  }

使用方式：
    python dimension_extract_for_txt.py --txt chunk_result/merged_chunk_200.txt --collection merged_chunk_200
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

# 代码模块路径
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from llm_service import DimensionMiningWithQwen

EXPERIMENT_DATA = PROJECT_ROOT / "experiment_data"
EXPERIMENT_DATA.mkdir(exist_ok=True)

# ============================================================
# 与 test_collection 一致的 15 个固定维度（Milvus 哈希字段名）
# ============================================================
# 中文维度名 → Milvus 哈希字段名
DIM_CHINESE_TO_MILVUS: Dict[str, str] = {
    "景点名称": "dim___d2e6065c",
    "历史人物": "dim___64b4c1a7",
    "朝代":      "dim___cc065709",
    "建筑类型": "dim___be291f25",
    "地理位置": "dim___da10bece",
    "文化价值": "dim___2d313d2a",
    "展出内容": "dim___308609fe",
    "相关事件": "dim___3c0aea5f",
    "人物身份": "dim___f2a6f001",
    "年代时间": "dim___5d209889",
    "相关地点": "dim___280b28c",
    "作品名称": "dim___8e37afab",
    "相关机构": "dim___54745b3d",
    "祭祀活动": "dim___f6d982db",
    "教育功能": "dim___34ea37de",
}

# 用于调用 LLM 的维度名称列表（原始中文名，LLM 能理解）
DIM_ORIGINAL_NAMES: List[str] = list(DIM_CHINESE_TO_MILVUS.keys())

# 用于 Milvus schema 的哈希字段名
DIM_MILVUS_FIELDS: List[str] = list(DIM_CHINESE_TO_MILVUS.values())


def _sanitize(name: str) -> str:
    """将 collection 名称转换为合法格式（纯字母数字下划线）"""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower() or "chunk"


def load_chunks_from_txt(txt_file: str, collection_name: str) -> List[Dict[str, str]]:
    """
    从 TXT 文件加载 chunks，每行一个段落。

    Returns:
        [{"chunk_id": "...", "chunk_text": "..."}, ...]
    """
    txt_path = Path(txt_file)
    if not txt_path.exists():
        raise FileNotFoundError(f"TXT 文件不存在: {txt_file}")

    safe_col = _sanitize(collection_name)
    chunks = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            chunks.append({
                "chunk_id":    f"{safe_col}_chunk_{i:04d}",
                "chunk_text":   line,
            })
    return chunks


def extract_tags_for_chunks(
    chunks: List[Dict],
    collection_name: str,
) -> Dict[str, Dict[str, List[str]]]:
    """
    对所有 chunks 批量抽取维度标签。

    直接使用固定 15 个中文维度名调用 LLM，
    结果保存到 experiment_data/doc_tags_{collection}.json，
    JSON 的 key 使用 Milvus 哈希字段名（如 dim___d2e6065c）。
    支持断点续跑。
    """
    output_file = EXPERIMENT_DATA / f"doc_tags_{collection_name}.json"

    # 断点续跑
    doc_tags: Dict[str, Dict[str, List[str]]] = {}
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            doc_tags = json.load(f)
        print(f"[Step 2] 检测到断点，已处理 {len(doc_tags)}/{len(chunks)} 条")

    # 用于 LLM 调用的维度名（原始中文名）
    llm_dims = DIM_ORIGINAL_NAMES  # 15 个中文维度名

    miner = DimensionMiningWithQwen()

    for i, chunk in enumerate(chunks):
        cid = chunk["chunk_id"]
        if cid in doc_tags:
            continue

        text = chunk["chunk_text"]
        if not text or len(text) < 10:
            doc_tags[cid] = {}
            continue

        try:
            # 传入 15 个中文维度名（LLM 能理解）
            tags = miner.extract_batch_dimensions(text, llm_dims)

            # 将 LLM 返回的 {中文维度名: [vals]} 转换为 {Milvus哈希名: [vals]}
            normalized_tags: Dict[str, List[str]] = {}
            for chinese, vals in (tags or {}).items():
                if not vals:
                    continue
                hashed = DIM_CHINESE_TO_MILVUS.get(chinese)
                if hashed:
                    normalized_tags[hashed] = vals

            doc_tags[cid] = normalized_tags

        except Exception as e:
            print(f"[WARN] 抽取失败 chunk_id={cid}: {e}")
            doc_tags[cid] = {}

        processed = len(doc_tags)
        if processed % 10 == 0:
            print(f"[Step 2] 进度: {processed}/{len(chunks)} ...")

        if processed % 20 == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(doc_tags, f, ensure_ascii=False, indent=2)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(doc_tags, f, ensure_ascii=False, indent=2)
    print(f"[Step 2] 完成，共抽取 {len(doc_tags)} 条标签，保存���: {output_file}")

    total_tags = sum(len(v) for v in doc_tags.values())
    dims_with_tags = sum(1 for v in doc_tags.values() if v)
    print(f"[Step 2] 有标签文档: {dims_with_tags}/{len(doc_tags)}，总标签数: {total_tags}")

    return doc_tags


def print_stats(doc_tags: Dict, collection_name: str):
    """打印统计信息（显示中文维度名）。"""
    if not doc_tags:
        print("[Stats] 无标签数据")
        return

    # 哈希名 → 中文名 的反向映射
    milvus_to_chinese = {v: k for k, v in DIM_CHINESE_TO_MILVUS.items()}

    dim_coverage: Dict[str, int] = {}
    for tags in doc_tags.values():
        for dim, vals in tags.items():
            if vals:
                label = milvus_to_chinese.get(dim, dim)
                dim_coverage[label] = dim_coverage.get(label, 0) + 1

    print(f"\n[Stats] 维度覆盖率统计（共 {len(doc_tags)} 文档）")
    print(f"{'Milvus 字段名':<26} {'中文名':<14} {'有标签文档数':>12} {'覆盖率':>8}")
    print("-" * 65)
    for label, cnt in sorted(dim_coverage.items(), key=lambda x: -x[1]):
        pct = cnt / len(doc_tags) * 100
        # 找对应的 Milvus 哈希名
        milvus_name = DIM_CHINESE_TO_MILVUS.get(label, "?")
        print(f"{milvus_name:<26} {label:<14} {cnt:>12} {pct:>7.1f}%")


def run_full(
    txt_file: str,
    collection_name: str,
):
    """一键运行全部流程。"""
    print(f"\n{'='*60}")
    print(f"  维度标签抽取  |  TXT: {txt_file}  |  Collection: {collection_name}")
    print(f"  固定维度候选: {DIM_ORIGINAL_NAMES}")
    print(f"{'='*60}")

    chunks = load_chunks_from_txt(txt_file, collection_name)
    print(f"[加载] {len(chunks)} 个 chunks")

    doc_tags = extract_tags_for_chunks(chunks, collection_name)
    print_stats(doc_tags, collection_name)

    print(f"\n{'='*60}")
    print(f"  全部完成！")
    print(f"{'='*60}")
    print(f"  输出: experiment_data/doc_tags_{collection_name}.json")
    print(f"  下一步（回写 Milvus）:")
    print(f"    python migrate_mysql_to_milvus.py --enrich_tags experiment_data/doc_tags_{collection_name}.json --collection {collection_name}")


def main():
    """CLI 主入口函数。"""
    parser = argparse.ArgumentParser(
        description="从 TXT chunks 自动抽取维度标签（使用固定 15 维）"
    )
    parser.add_argument("--txt", type=str, required=True,
                       help="TXT 文件路径（每行一个 chunk）")
    parser.add_argument("--collection", type=str, required=True,
                       help="Milvus Collection 名称（用于命名输出文件）")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="LLM 批量抽取每批文档数（默认 10）")
    parser.add_argument("--step1_only", action="store_true",
                       help="已废弃（维度名已固定，无需生成）")
    parser.add_argument("--step2_only", action="store_true",
                       help="仅运行抽取（需要已有 chunks）")
    parser.add_argument("--step3_only", action="store_true",
                       help="仅运行统计（需要已有 doc_tags 文件）")
    args = parser.parse_args()

    if args.step1_only:
        print("[INFO] --step1_only 已废弃，维度名已固定为 15 个，无需生成候选维度")
        return

    if args.step3_only:
        output_file = EXPERIMENT_DATA / f"doc_tags_{args.collection}.json"
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                doc_tags = json.load(f)
            print_stats(doc_tags, args.collection)
        else:
            print(f"[错误] 文件不存在: {output_file}")
        return

    if args.step2_only:
        chunks = load_chunks_from_txt(args.txt, args.collection)
        doc_tags = extract_tags_for_chunks(chunks, args.collection)
        print_stats(doc_tags, args.collection)
        return

    run_full(txt_file=args.txt, collection_name=args.collection)


if __name__ == "__main__":
    main()

