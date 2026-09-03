#!/usr/bin/env python3
"""仅重算四种分片结果中的语义困惑度，避免重复计算其他指标。"""

import argparse
import json
from pathlib import Path

from four_dimension_eval import evaluate_ppl, load_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_json", required=True)
    parser.add_argument("--result_json", required=True)
    parser.add_argument("--ppl_model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    chunks = load_chunks(Path(args.chunks_json))
    ppl = evaluate_ppl(
        chunks,
        args.ppl_model,
        args.device,
        args.batch_size,
        args.max_length,
    )
    result_path = Path(args.result_json)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["metrics"]["semantic_perplexity"] = ppl
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[{result['method']}] PPL updated: {result_path}")


if __name__ == "__main__":
    main()
