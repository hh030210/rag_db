#!/usr/bin/env python3
"""按固定字符数对 db_qa.txt 做机械切分。

每个非空原始行视为一个独立文档，在行内按固定字符数顺序切分，
不做去噪、语义判断、重叠或跨行合并。
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--chunk_size", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    source = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    chunks = []
    total_input_chars = 0
    nonempty_lines = 0

    for line_idx, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines()):
        text = raw_line.strip()
        if not text:
            continue
        nonempty_lines += 1
        total_input_chars += len(text)
        for start in range(0, len(text), args.chunk_size):
            piece = text[start:start + args.chunk_size]
            if piece:
                chunks.append({
                    "chunk_text": piece,
                    "chunk_len": len(piece),
                    "source_line": line_idx,
                    "start_char": start,
                    "mechanical_chunk_size": args.chunk_size,
                })

    compact = [
        {"chunk_text": c["chunk_text"], "chunk_len": c["chunk_len"]}
        for c in chunks
    ]
    (output / "all_chunks_chunks.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "all_chunks_chunks.txt").write_text(
        "\n\n---\n\n".join(c["chunk_text"] for c in compact), encoding="utf-8"
    )
    summary = {
        "method": f"mechanical_{args.chunk_size}char",
        "chunk_size_chars": args.chunk_size,
        "source": str(source),
        "nonempty_lines": nonempty_lines,
        "input_chars": total_input_chars,
        "total_chunks": len(compact),
        "output_chars": sum(c["chunk_len"] for c in compact),
        "min_chunk_chars": min((c["chunk_len"] for c in compact), default=0),
        "max_chunk_chars": max((c["chunk_len"] for c in compact), default=0),
        "avg_chunk_chars": (
            sum(c["chunk_len"] for c in compact) / len(compact)
            if compact else 0.0
        ),
    }
    (output / "all_chunks_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
