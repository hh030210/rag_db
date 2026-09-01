"""
run_extract_only.py

使用指定的 OpenAI 兼容 LLM，对指定输入文件执行：
  1) 分片（integrated_chunker）
  2) 维度抽取（dimension_integration，读取 chunks 文件）
  3) 不入库

输出全部写入 output_no_ingest/ 目录，不触碰 MySQL / Qdrant。

用法示例：
  python run_extract_only.py `
    --api_key sk-xxx `
    --base_url https://xxx.compatible-mode/v1 `
    --model qwen3.6-max-preview `
    --input "data_input\\insert_Data\\南孔文化(吴锡标).txt" `
           "data_input\\insert_Data\\多景区知识.md" `
           "data_input\\test_data" `
    --dataset nankong_v2
"""

import argparse
import os
import shutil
import subprocess
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_llm_env(api_key: str, base_url: str, model: str, timeout: int = 120):
    """设置 OpenAI 兼容模式环境变量（在调子进程前调用）"""
    os.environ["LLM_OPENAI_COMPAT"] = "1"
    os.environ["LLM_API_KEY"] = api_key
    os.environ["LLM_BASE_URL"] = base_url
    os.environ["LLM_MODEL"] = model
    # 同时设 DASHSCOPE_API_KEY 以兼容旧代码
    os.environ["DASHSCOPE_API_KEY"] = api_key
    print(f"\n[LLM 环境变量]")
    print(f"  LLM_OPENAI_COMPAT=1")
    print(f"  LLM_API_KEY 前6位={api_key[:6]}...")
    print(f"  LLM_BASE_URL={base_url}")
    print(f"  LLM_MODEL={model}")


def collect_inputs(input_paths, merged_dir: Path) -> list:
    """合并所有输入文件到 merged_dir"""
    merged_dir.mkdir(parents=True, exist_ok=True)
    collected = []
    for raw in input_paths:
        p = Path(raw)
        if not p.exists():
            print(f"  [Warning] 路径不存在: {p}")
            continue
        if p.is_file():
            # 单文件：直接复制
            target = merged_dir / p.name
            if target.resolve() == p.resolve():
                # 已在 merged_dir 内，跳过复制
                collected.append(target)
            else:
                shutil.copy2(p, target)
                collected.append(target)
        elif p.is_dir():
            # 目录：递归复制所有 .md / .txt
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in {".md", ".txt", ".json"}:
                    target = merged_dir / f.name
                    if target.resolve() == f.resolve():
                        collected.append(target)
                        continue
                    # 避免同名冲突
                    if target.exists():
                        target = merged_dir / f"{f.parent.name}_{f.name}"
                    shutil.copy2(f, target)
                    collected.append(target)
    print(f"\n[输入汇总] 共 {len(collected)} 个文件，输出到 {merged_dir}/")
    for f in collected:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name}  ({size_kb:.1f} KB)")
    return collected


def run_chunking(input_dir: Path, output_dir: Path, api_key: str, base_url: str, model: str):
    """调用 simple_chunk.py 进行轻量分片（快速，不依赖 PPL/LLM）"""
    print(f"\n{'=' * 60}")
    print(f"[Step 1/2] 分片处理（simple_chunk）")
    print(f"  输入: {input_dir}")
    print(f"  输出: {output_dir}")
    print(f"{'=' * 60}")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "simple_chunk.py"),
        "--input", str(input_dir),
        "--output", str(output_dir),
        "--chunk_size", "800",
        "--overlap", "50",
    ]

    print(f"\n[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, env=os.environ.copy())
    if result.returncode != 0:
        raise RuntimeError(f"integrated_chunker 失败: returncode={result.returncode}")

    # 列出输出
    summary = output_dir / "all_chunks_summary.json"
    chunks = output_dir / "all_chunks_chunks.json"
    if summary.exists():
        with open(summary, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"\n  ✓ 分片完成: {data.get('total_files', '?')} 个文件, "
              f"{data.get('total_chunks', '?')} 个 chunk")
    print(f"  ✓ chunks 文件: {chunks}")
    return chunks if chunks.exists() else None


def run_dimension_extraction(chunks_file: Path, dataset: str, output_dir: Path, reset_cache: bool):
    """调用 dimension_integration 进行维度抽取（不依赖 MySQL）"""
    print(f"\n{'=' * 60}")
    print(f"[Step 2/2] 维度抽取（dimension_integration）")
    print(f"  chunks: {chunks_file}")
    print(f"  dataset: {dataset}")
    print(f"{'=' * 60}")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "dimension_integration.py"),
        "--all",
        "--docs_source", str(chunks_file),
    ]
    if reset_cache:
        cmd.append("--force")

    print(f"\n[CMD] {' '.join(cmd)}")
    # dimension_integration 的输出写到 experiment_data/
    result = subprocess.run(cmd, env=os.environ.copy())
    if result.returncode != 0:
        raise RuntimeError(f"dimension_integration 失败: returncode={result.returncode}")

    # 收集输出到 output_no_ingest/
    exp_data = PROJECT_ROOT / "experiment_data"
    copied = []
    for fname in [f"V_cand_{dataset}.json", f"V_core_{dataset}.json",
                  f"tags_output_{dataset}.json", f"step5_result_{dataset}.json"]:
        src = exp_data / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            copied.append(fname)

    # 同时把原始默认命名也备份
    for fname in ["V_cand.json", "V_core.json", "tags_output.json", "step5_result.json"]:
        src = exp_data / fname
        if src.exists():
            target = output_dir / fname
            shutil.copy2(src, target)
            copied.append(fname)

    print(f"\n  ✓ 已复制 {len(copied)} 个结果文件到 {output_dir}/")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="使用 OpenAI 兼容 LLM 对指定文件执行分片+维度抽取（不入库）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api_key", required=True, help="LLM API Key")
    parser.add_argument("--base_url", required=True, help="LLM Base URL（OpenAI 兼容）")
    parser.add_argument("--model", required=True, help="LLM 模型名")
    parser.add_argument("--input", "-i", nargs="+", required=True,
                        help="输入文件/目录列表（可多个）")
    parser.add_argument("--dataset", "-d", default="default",
                        help="数据集标识名（用于输出文件隔离）")
    parser.add_argument("--output_root", default="output_no_ingest",
                        help="输出根目录（默认 output_no_ingest）")
    parser.add_argument("--timeout", type=int, default=120, help="LLM 调用超时秒数")
    parser.add_argument("--reset_cache", action="store_true",
                        help="清空 experiment_data 中的 V_cand/V_core/tags_output 等缓存")
    parser.add_argument("--keep_merged_inputs", action="store_true",
                        help="保留合并后的输入副本（默认运行后清理）")

    args = parser.parse_args()

    # 1. 设置 LLM 环境变量
    setup_llm_env(args.api_key, args.base_url, args.model, args.timeout)

    # 2. 准备输出目录
    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    merged_inputs = output_root / "input_merged"
    chunks_out = output_root / "chunks"
    extraction_out = output_root / "extraction"
    chunks_out.mkdir(parents=True, exist_ok=True)
    extraction_out.mkdir(parents=True, exist_ok=True)

    # 3. 收集所有输入文件到 merged_inputs
    print(f"\n[准备] 收集输入文件...")
    collected = collect_inputs(args.input, merged_inputs)
    if not collected:
        print("[Error] 未收集到任何有效输入文件")
        return 1

    # 4. 分片
    try:
        chunks_file = run_chunking(merged_inputs, chunks_out,
                                   args.api_key, args.base_url, args.model)
    finally:
        # 5. 清理合并输入
        if not args.keep_merged_inputs:
            try:
                shutil.rmtree(merged_inputs)
                print(f"\n[清理] 已删除临时目录: {merged_inputs}")
            except Exception as e:
                print(f"\n[清理失败] {e}")

    if not chunks_file or not chunks_file.exists():
        print("[Error] 分片输出未找到")
        return 1

    # 6. 维度抽取（不写入 MySQL）
    extraction_out_dataset = extraction_out  # dataset 已通过 PATH_V_CORE 等隔离
    try:
        run_dimension_extraction(chunks_file, args.dataset,
                                 extraction_out_dataset, args.reset_cache)
    except Exception as e:
        print(f"[Error] 维度抽取失败: {e}")
        return 1

    print(f"\n{'=' * 60}")
    print(f"全部完成！")
    print(f"  分片输出:   {chunks_out}/")
    print(f"  抽取结果:   {extraction_out}/")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())