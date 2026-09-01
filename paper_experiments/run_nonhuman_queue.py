#!/usr/bin/env python3
"""Ordered queue for the non-human paper experiments.

Stages are resumable. Human chunk-quality annotation is intentionally absent.
The main QA comparison is serialized and uses the existing API rate limiter.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def marker(root: Path, name: str) -> Path:
    return root / "markers" / f"{name}.done.json"


def run_once(root: Path, name: str, command: list[str], cwd: Path,
             env: dict[str, str] | None = None) -> None:
    done = marker(root, name)
    if done.exists():
        print(f"[skip] {name}", flush=True)
        return
    started = time.time()
    run(command, cwd, env)
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps({
        "name": name,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - started,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--data_path", default="")
    ap.add_argument("--qa_data_path", default="")
    ap.add_argument("--work_root", default="")
    ap.add_argument("--stage", choices=("prepare", "qa", "stats", "ablation", "all"), default="all")
    args = ap.parse_args()

    code_root = Path(args.code_root).expanduser().resolve()
    data_path = Path(args.data_path or code_root / "data" / "db_qa.txt").resolve()
    qa_data = Path(args.qa_data_path or code_root / "data" / "split_merged.json").resolve()
    work = Path(args.work_root or code_root / "results" / "nonhuman_paper_queue_20260825").resolve()
    scripts = Path(__file__).resolve().parent
    mechanical = code_root / "mechanical_chunker.py"
    integrated = code_root / "integrated_chunker.py"
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    for path in (mechanical, integrated):
        if not path.exists():
            raise FileNotFoundError(path)

    chunks = {
        "mechanical_200": work / "chunks" / "mechanical_200",
        "mechanical_300": work / "chunks" / "mechanical_300",
        "mechanical_400": work / "chunks" / "mechanical_400",
        "three_stage": work / "chunks" / "three_stage",
    }
    docs = {name: work / "docs" / name for name in chunks}
    common_env = os.environ.copy()
    common_env.setdefault("PYTHONUNBUFFERED", "1")

    if args.stage in ("prepare", "all"):
        for size in (200, 300, 400):
            name = f"mechanical_{size}"
            run_once(work, f"chunk_{name}", [
                sys.executable, str(mechanical), "--input", str(data_path),
                "--chunk_size", str(size), "--output", str(chunks[name]),
            ], code_root, common_env)
        run_once(work, "chunk_three_stage", [
            sys.executable, str(integrated), "--input", str(data_path),
            "--line_mode", "--llm_sample_interval", "0", "--llm_api_key", "",
            "--ppl_model_name", "", "--denoise", "true",
            "--output", str(chunks["three_stage"]),
        ], code_root, common_env)
        for name in chunks:
            run_once(work, f"docs_{name}", [
                sys.executable, str(scripts / "prepare_docs.py"),
                "--chunks_json", str(chunks[name] / "all_chunks_chunks.json"),
                "--output_dir", str(docs[name]),
            ], code_root, common_env)

    if args.stage in ("qa", "all"):
        if not qa_data.exists():
            raise FileNotFoundError(qa_data)
        qa_root = work / "qa"
        for name in ("mechanical_200", "mechanical_300", "mechanical_400", "three_stage"):
            run_once(work, f"qa_{name}", [
                sys.executable, str(scripts / "run_qa_method.py"),
                "--code_root", str(code_root), "--docs_path", str(docs[name]),
                "--data_path", str(qa_data), "--output_dir", str(qa_root / name),
                "--collection_name", f"nonhuman_{name}_top8",
                "--milvus_uri", str(work / f"milvus_{name}.db"),
            ], code_root, common_env)

    if args.stage in ("stats", "all"):
        run_once(work, "paired_qa_stats", [
            sys.executable, str(scripts / "paired_qa_stats.py"),
            "--qa_root", str(work / "qa"),
            "--output", str(work / "stats" / "paired_qa_stats.json"),
        ], code_root, common_env)

    if args.stage in ("ablation", "all"):
        ablation_specs = {
            "no_structure": ["--disable_structure_split"],
            "no_denoise": ["--denoise", "false"],
            "no_optimization": ["--disable_optimization"],
        }
        for name, extra in ablation_specs.items():
            out = work / "ablation_chunks" / name
            run_once(work, f"ablation_chunk_{name}", [
                sys.executable, str(integrated), "--input", str(data_path),
                "--line_mode", "--llm_sample_interval", "0", "--llm_api_key", "",
                "--ppl_model_name", "", *extra,
                "--output", str(out),
            ], code_root, common_env)
            run_once(work, f"ablation_docs_{name}", [
                sys.executable, str(scripts / "prepare_docs.py"),
                "--chunks_json", str(out / "all_chunks_chunks.json"),
                "--output_dir", str(work / "ablation_docs" / name),
            ], code_root, common_env)

        summary = {
            "experiment": "three_stage_component_ablation",
            "variants": {
                "full": "structure split + denoise + optimization",
                "no_structure": "structure split disabled",
                "no_denoise": "denoise disabled",
                "no_optimization": "optimization disabled",
            },
            "results": {},
        }
        for name in ("three_stage", "no_structure", "no_denoise", "no_optimization"):
            path = (chunks["three_stage"] if name == "three_stage" else work / "ablation_chunks" / name) / "all_chunks_summary.json"
            if path.exists():
                summary["results"][name] = json.loads(path.read_text(encoding="utf-8"))
        output = work / "stats" / "component_ablation_summary.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"work_root": str(work), "stage": args.stage}, ensure_ascii=False))


if __name__ == "__main__":
    main()
