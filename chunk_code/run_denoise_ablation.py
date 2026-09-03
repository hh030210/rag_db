#!/usr/bin/env python3
"""Run the same three-stage chunker twice with denoise on and off.

The script keeps all chunker parameters identical and writes a manifest so the
two outputs can be connected to the later Milvus/QuestAnswer experiments.
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
    redacted = list(command)
    try:
        idx = redacted.index("--llm_api_key")
        if idx + 1 < len(redacted):
            redacted[idx + 1] = "***" if redacted[idx + 1] else ""
    except ValueError:
        pass
    return redacted


def main() -> None:
    parser = argparse.ArgumentParser(description="三阶段分片去噪开关对照实验")
    parser.add_argument("--input", required=True, help="输入 txt/md 文件或目录")
    parser.add_argument("--output_root", required=True, help="两组结果的根目录")
    parser.add_argument("--line_mode", action="store_true", help="每行作为独立文档")
    parser.add_argument("--llm_sample_interval", type=int, default=0)
    parser.add_argument("--llm_api_key", default="", help="可选；默认不调用外部 LLM")
    parser.add_argument("--llm_base_url", default=os.environ.get("CHUNK_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--llm_model", default=os.environ.get("CHUNK_LLM_MODEL", "qwen3-8b"))
    parser.add_argument("--ppl_model_name", default="")
    parser.add_argument("--window_w", type=int, default=3)
    parser.add_argument("--beta_small", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=1.1)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    chunker = repo_root / "integrated_chunker.py"
    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    common = [
        sys.executable,
        str(chunker),
        "--input", str(input_path),
        "--llm_sample_interval", str(args.llm_sample_interval),
        "--llm_api_key", args.llm_api_key,
        "--llm_base_url", args.llm_base_url,
        "--llm_model", args.llm_model,
        "--ppl_model_name", args.ppl_model_name,
        "--window_w", str(args.window_w),
        "--beta_small", str(args.beta_small),
        "--beta", str(args.beta),
    ]
    if args.line_mode:
        common.append("--line_mode")

    manifest = {
        "input": str(input_path),
        "chunker": str(chunker),
        "same_parameters": True,
        "llm_api_key_provided": bool(args.llm_api_key),
        "experiments": [],
    }

    for label, enabled in (("denoise_off", False), ("denoise_on", True)):
        output_dir = output_root / label
        command = common + ["--denoise", str(enabled).lower(), "--output", str(output_dir)]
        safe_command = _redact_command(command)
        print("\n[run]", shlex.join(safe_command), flush=True)
        subprocess.run(command, cwd=repo_root, check=True)
        manifest["experiments"].append({
            "label": label,
            "denoise_enabled": enabled,
            "output": str(output_dir),
            "command": safe_command,
        })

    manifest_path = output_root / "ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
