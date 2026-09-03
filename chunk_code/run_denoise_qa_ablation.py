#!/usr/bin/env python3
"""Run the canonical CRUD QuestAnswer evaluation for denoise off/on outputs.

The actual QA entrypoint is Meta-Chunking/eval/CRUD/quick_start.py. This wrapper
keeps the retriever, embedding, generator and QA dataset identical, while using
different Milvus collection names and document directories for the two chunking
outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _redact_command(command: list[str]) -> list[str]:
    # API credentials are read from the environment and should never enter the
    # command manifest; this also protects users who pass a key accidentally.
    redacted = list(command)
    try:
        idx = redacted.index("--llm_api_key")
        if idx + 1 < len(redacted):
            redacted[idx + 1] = "***"
    except ValueError:
        pass
    return redacted


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    default_ablation = repo_root / "results" / "denoise_ablation_db_qa_textsafe"
    default_eval = repo_root / "Meta-Chunking" / "eval" / "CRUD"

    parser = argparse.ArgumentParser(description="去噪开关两组 QuestAnswer 对照实验")
    parser.add_argument("--ablation_root", default=str(default_ablation))
    parser.add_argument("--data_path", default=str(repo_root / "data" / "split_merged.json"))
    parser.add_argument("--eval_dir", default=str(default_eval))
    parser.add_argument("--output_dir", default="", help="QA JSON 输出目录；默认写入 ablation_root/qa_output")
    parser.add_argument("--embedding_name", default="sentence-transformers/bge-base-zh-v1.5")
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--model_name", default="qwen_api", choices=["qwen_api", "qwen7b"])
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=1280)
    parser.add_argument("--retrieve_top_k", type=int, default=8)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--bert_score_eval", action="store_true")
    parser.add_argument("--quest_eval", action="store_true")
    args = parser.parse_args()

    ablation_root = Path(args.ablation_root).expanduser().resolve()
    data_path = Path(args.data_path).expanduser().resolve()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else ablation_root / "qa_output"
    quick_start = eval_dir / "quick_start.py"
    docs = {
        "denoise_off": ablation_root / "denoise_off" / "docs",
        "denoise_on": ablation_root / "denoise_on" / "docs",
    }

    for path in (data_path, quick_start, docs["denoise_off"], docs["denoise_on"]):
        if not path.exists():
            raise FileNotFoundError(f"实验输入不存在: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "quick_start": str(quick_start),
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "same_qa_parameters": True,
        "experiments": [],
        "api_key_source": "QWEN_OPENAI_API_KEY environment variable",
    }

    common = [
        sys.executable,
        str(quick_start),
        "--model_name", args.model_name,
        "--temperature", str(args.temperature),
        "--max_new_tokens", str(args.max_new_tokens),
        "--data_path", str(data_path),
        "--embedding_name", args.embedding_name,
        "--embedding_dim", str(args.embedding_dim),
        "--docs_type", "txt",
        "--chunk_size", "128",
        "--chunk_overlap", "0",
        "--retriever_name", "base",
        "--retrieve_top_k", str(args.retrieve_top_k),
        "--task", "quest_answer",
        "--num_threads", str(args.num_threads),
        "--show_progress_bar", "True",
        "--output_dir", str(output_dir),
        "--construct_index",
    ]
    if args.bert_score_eval:
        common.append("--bert_score_eval")
    if args.quest_eval:
        common.append("--quest_eval")

    for label in ("denoise_off", "denoise_on"):
        collection = f"denoise_{label.split('_')[-1]}_dbqa_top{args.retrieve_top_k}"
        command = common + [
            "--docs_path", str(docs[label]),
            "--collection_name", collection,
        ]
        safe_command = _redact_command(command)
        print("\n[run]", shlex.join(safe_command), flush=True)
        subprocess.run(command, cwd=eval_dir, check=True)
        manifest["experiments"].append({
            "label": label,
            "collection_name": collection,
            "docs_path": str(docs[label]),
            "command": safe_command,
        })

    manifest_path = ablation_root / "qa_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
