#!/usr/bin/env python3
"""在统一字符预算下重算内容一致性，并保留原始指标。"""

import argparse
import json
from pathlib import Path

from four_dimension_eval import evaluate_content_consistency, load_chunks, split_sentences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_json", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--input_result", required=True)
    parser.add_argument("--output_result", required=True)
    parser.add_argument("--embedding_model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_chars", type=int, default=200)
    args = parser.parse_args()

    source_lines = [
        x.strip()
        for x in Path(args.source).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    chunks = load_chunks(Path(args.chunks_json))
    sentence_groups = [split_sentences(x) or [x] for x in chunks]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.embedding_model, device=args.device)
    content = evaluate_content_consistency(
        source_lines,
        chunks,
        sentence_groups,
        model,
        args.batch_size,
        args.device,
        args.target_chars,
    )

    result = json.loads(Path(args.input_result).read_text(encoding="utf-8"))
    result["metrics"]["content_consistency"] = content
    result["content_consistency_normalization"] = {
        "target_chars": args.target_chars,
        "unit": "fixed character windows with sentence-local internal windows",
    }
    output = Path(args.output_result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[{result['method']}] normalized consistency saved: {output}", flush=True)


if __name__ == "__main__":
    main()
