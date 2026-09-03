#!/usr/bin/env python3
"""Run one fixed-condition end-to-end QA experiment and keep it resumable."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_root", required=True)
    ap.add_argument("--docs_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--collection_name", required=True)
    ap.add_argument("--milvus_uri", required=True)
    ap.add_argument("--embedding_name", default=os.environ.get("BGE_MODEL_PATH", "BAAI/bge-base-zh-v1.5"))
    ap.add_argument("--embedding_dim", type=int, default=768)
    ap.add_argument("--retrieve_top_k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max_new_tokens", type=int, default=1280)
    args = ap.parse_args()

    code_root = Path(args.code_root).resolve()
    eval_dir = code_root / "Meta-Chunking" / "eval" / "CRUD"
    quick_start = eval_dir / "quick_start.py"
    if not quick_start.exists():
        raise FileNotFoundError(quick_start)
    if not Path(args.docs_path).is_dir():
        raise FileNotFoundError(args.docs_path)
    if not Path(args.data_path).is_file():
        raise FileNotFoundError(args.data_path)
    if not os.environ.get("QWEN_OPENAI_API_KEY"):
        raise RuntimeError("QWEN_OPENAI_API_KEY 未设置，已停止 API 问答实验")

    env = os.environ.copy()
    env["DENOISE_MILVUS_URI"] = str(Path(args.milvus_uri).resolve())
    env.setdefault("QWEN_MIN_INTERVAL", "10")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(quick_start),
        "--model_name", "qwen_api",
        "--temperature", str(args.temperature),
        "--max_new_tokens", str(args.max_new_tokens),
        "--data_path", str(Path(args.data_path).resolve()),
        "--embedding_name", args.embedding_name,
        "--embedding_dim", str(args.embedding_dim),
        "--docs_path", str(Path(args.docs_path).resolve()),
        "--docs_type", "txt",
        "--chunk_size", "128",
        "--chunk_overlap", "0",
        "--retriever_name", "base",
        "--collection_name", args.collection_name,
        "--retrieve_top_k", str(args.retrieve_top_k),
        "--task", "quest_answer",
        "--num_threads", "1",
        "--show_progress_bar", "True",
        "--output_dir", str(output),
        "--construct_index",
    ]
    print("[qa] method output:", output, flush=True)
    print("[qa] API interval:", env.get("QWEN_MIN_INTERVAL", "10"), "seconds", flush=True)
    subprocess.run(command, cwd=eval_dir, env=env, check=True)


if __name__ == "__main__":
    main()
