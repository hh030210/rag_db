#!/usr/bin/env python3
"""Automatic answer-evidence retention evaluation for denoise off/on chunks.

This is a retrieval-only experiment.  It builds one independent Milvus Lite
index per condition, retrieves the same QA questions, and measures how much
of each reference answer is covered by the retrieved chunks.  Because the
dataset does not provide explicit gold evidence spans, the answer n-gram
coverage is reported as an automatic evidence proxy, not as human gold labels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path


# The script is stored at chunk_code/ while CRUD's packages live below
# Meta-Chunking/eval/CRUD.  Add that directory explicitly so the same script
# works both from the project root and from the CRUD working directory.
SCRIPT_ROOT = Path(__file__).resolve().parent
CRUD_ROOT = SCRIPT_ROOT / "Meta-Chunking" / "eval" / "CRUD"
if str(CRUD_ROOT) not in sys.path:
    sys.path.insert(0, str(CRUD_ROOT))


def clean_rows(data: dict) -> list[dict]:
    rows = []
    for name in ("questanswer_1doc", "questanswer_2docs", "questanswer_3docs"):
        for row in data.get(name, []):
            q = row.get("questions")
            a = row.get("answers")
            if isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip():
                rows.append({"category": name, "id": row.get("ID"),
                             "question": q.strip(), "answer": a.strip()})
    return rows


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def ngrams(text: str, n: int = 2) -> set[str]:
    text = compact(text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def coverage(answer: str, context: str, n: int = 2) -> float:
    grams = ngrams(answer, n)
    if not grams:
        return 0.0
    return len(grams & ngrams(context, n)) / len(grams)


def evaluate(args: argparse.Namespace) -> None:
    # Imports are delayed so this script can be inspected without the CRUD env.
    from src.embeddings.base import HuggingfaceEmbeddings
    from src.retrievers.base import BaseRetriever

    data = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    rows = clean_rows(data)
    embed = HuggingfaceEmbeddings(model_name=args.embedding_name)
    retriever = BaseRetriever(
        args.docs_path,
        embed_model=embed,
        embed_dim=args.embedding_dim,
        chunk_size=128,
        chunk_overlap=0,
        collection_name=args.collection_name,
        construct_index=args.construct_index,
        similarity_top_k=max(args.top_ks),
    )

    results = []
    started = time.time()
    for i, row in enumerate(rows, 1):
        try:
            source_nodes = retriever.vector_retriever.retrieve(row["question"])
            texts = []
            for source_node in source_nodes:
                node = getattr(source_node, "node", None)
                text = getattr(node, "text", None) if node is not None else None
                if text:
                    texts.append(text)
            per_k = {}
            for k in args.top_ks:
                context = "\n".join(texts[:k])
                cov = coverage(row["answer"], context)
                per_k[str(k)] = {
                    "answer_bigram_coverage": cov,
                    "answer_exact_substring": compact(row["answer"]) in compact(context),
                    "context_chars": len(context),
                    "retrieved_chunks": len(texts[:k]),
                }
            results.append({"id": row["id"], "category": row["category"],
                            "metrics": per_k, "valid": True})
        except Exception as exc:
            results.append({"id": row["id"], "category": row["category"],
                            "error": repr(exc), "valid": False})
        if i % 100 == 0 or i == len(rows):
            print(f"retrieval {i}/{len(rows)} elapsed={time.time() - started:.1f}s", flush=True)

    summary = {"valid": sum(x.get("valid", False) for x in results),
               "invalid": sum(not x.get("valid", False) for x in results),
               "by_category": {}}
    for category in ("questanswer_1doc", "questanswer_2docs", "questanswer_3docs"):
        cat = [x for x in results if x["category"] == category and x.get("valid")]
        summary["by_category"][category] = {"n": len(cat), "top_k": {}}
        for k in args.top_ks:
            vals = [x["metrics"][str(k)]["answer_bigram_coverage"] for x in cat]
            hits = [x["metrics"][str(k)]["answer_exact_substring"] for x in cat]
            summary["by_category"][category]["top_k"][str(k)] = {
                "mean_answer_bigram_coverage": sum(vals) / len(vals) if vals else 0.0,
                "exact_answer_hit_rate": sum(hits) / len(hits) if hits else 0.0,
            }
    output = {"experiment": "retrieval_answer_evidence_proxy",
              "condition": args.condition, "docs_path": args.docs_path,
              "collection_name": args.collection_name,
              "top_ks": args.top_ks, "summary": summary, "results": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {args.output}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--docs_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--db_uri", required=True)
    ap.add_argument("--collection_name", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--embedding_name", required=True)
    ap.add_argument("--embedding_dim", type=int, default=768)
    ap.add_argument("--top_ks", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--construct_index", action="store_true")
    args = ap.parse_args()
    import os
    os.environ["DENOISE_MILVUS_URI"] = args.db_uri
    evaluate(args)


if __name__ == "__main__":
    main()
