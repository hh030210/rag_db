#!/usr/bin/env python3
"""Paper-ready structural metrics for the denoise chunking comparison.

This adapter reconstructs source-document groups because the historical denoise
JSON output kept only chunk_text/chunk_len.  It then computes the embedding part
of Boundary Clarity without creating false boundaries between adjacent news
articles.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_chunks(path: Path) -> list[dict]:
    if path.is_dir():
        files = sorted(path.glob("chunk_*.txt"))
        data = [{"chunk_text": f.read_text(encoding="utf-8")} for f in files]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("chunks", data.get("data", []))
    out = []
    for i, row in enumerate(data):
        text = str(row.get("chunk_text", row.get("text", ""))).strip()
        if text:
            out.append({"index": i, "chunk_text": text, "chunk_len": len(text)})
    return out


def read_source_lines(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def compact(s: str) -> str:
    return re.sub(r"\s+", "", s)


def find_anchor(line: str, text: str, start: int) -> int:
    """Find a reliable substring anchor inside one source line.

    The denoising output can remove a sentence between two chunks, so the next
    chunk's *prefix* is not always present verbatim.  Probe several windows
    (including the suffix) while allowing a small overlap with the previous
    match.  The caller still enforces monotonically increasing source-line
    order, so a repeated short phrase cannot move the match backwards.
    """
    search_start = max(0, min(start, len(line)) - 64)
    if text:
        pos = line.find(text, search_start)
        if pos >= 0:
            return pos
    starts = sorted({0, 5, 10, 20, 40, 80, len(text) // 2,
                     max(0, len(text) - 80)})
    for n in (80, 40, 20, 10):
        for text_start in starts:
            anchor = text[text_start:text_start + n]
            if len(anchor) < min(n, 8):
                continue
            pos = line.find(anchor, search_start)
            if pos >= 0:
                return pos
    compact_line = compact(line[search_start:])
    for text_start in starts:
        compact_text = compact(text[text_start:text_start + 80])
        if len(compact_text) < 8:
            continue
        compact_pos = compact_line.find(compact_text)
        if compact_pos >= 0:
            return search_start + compact_pos
    return -1


def assign_source_groups(chunks: list[dict], source_lines: list[str]) -> tuple[list[dict], dict]:
    """Assign sequential chunks to source lines using source-order anchors."""
    assigned = []
    line_idx = 0
    offset = 0
    unmatched = []
    for row in chunks:
        text = row["chunk_text"]
        found = -1
        found_line = line_idx
        # Search current line first, then later lines.  Never move backwards.
        for j in range(line_idx, len(source_lines)):
            p = find_anchor(source_lines[j], text, offset if j == line_idx else 0)
            # A denoised chunk may overlap the tail of the preceding chunk;
            # if the cursor has moved past the actual end, retry the current
            # source line before considering a later article.
            if p < 0 and j == line_idx and offset > 0:
                p = find_anchor(source_lines[j], text, 0)
            if p >= 0:
                found, found_line = p, j
                break
        if found < 0:
            # Keep the row for length statistics, but do not fabricate a source
            # document assignment: that would create false boundaries between
            # unrelated articles.
            unmatched.append(row["index"])
            row = dict(row)
            row["doc_id"] = None
            row["source_line"] = None
            assigned.append(row)
            continue
        else:
            line_idx = found_line
            # Chunks are emitted in source order.  Keep the cursor bounded; a
            # chunk can slightly overlap or cross the end of a source line.
            offset = min(len(source_lines[line_idx]), found + max(1, len(text)))
        row = dict(row)
        row["doc_id"] = f"source_line_{found_line:05d}"
        row["source_line"] = found_line
        assigned.append(row)
    return assigned, {
        "source_lines": len(source_lines),
        "chunks": len(chunks),
        "unmatched_chunks": len(unmatched),
        "unmatched_indices": unmatched[:100],
    }


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "p90": 0, "p95": 0, "max": 0, "mean": 0, "std": 0}
    x = np.asarray(values, dtype=float)
    return {
        "min": float(x.min()), "p25": float(np.quantile(x, .25)),
        "median": float(np.median(x)), "p75": float(np.quantile(x, .75)),
        "p90": float(np.quantile(x, .90)), "p95": float(np.quantile(x, .95)),
        "max": float(x.max()), "mean": float(x.mean()), "std": float(x.std()),
    }


def boundary_metric(rows: list[dict], model_path: str, device: str, batch_size: int) -> dict:
    from sentence_transformers import SentenceTransformer

    groups = defaultdict(list)
    for row in rows:
        groups[row["doc_id"]].append(row)
    ordered_groups = [groups[k] for k in sorted(groups)]
    texts = [r["chunk_text"] for r in rows]
    print(f"Loading embedding model: {model_path}", flush=True)
    model = SentenceTransformer(model_path, device=device)
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    by_index = {r["index"]: emb[i] for i, r in enumerate(rows)}
    scores = []
    doc_scores = {}
    for doc_id, group in zip(sorted(groups), ordered_groups):
        local = []
        for left, right in zip(group, group[1:]):
            # normalized embeddings make the dot product cosine similarity.
            score = 1.0 - float(np.dot(by_index[left["index"]], by_index[right["index"]]))
            scores.append(score)
            local.append(score)
        if local:
            doc_scores[doc_id] = float(np.mean(local))
    # Bootstrap at document level to avoid treating thousands of chunks from a
    # single document as independent observations.
    rng = np.random.default_rng(20260812)
    vals = np.asarray(list(doc_scores.values()), dtype=float)
    boot = []
    if len(vals):
        for _ in range(2000):
            boot.append(float(rng.choice(vals, size=len(vals), replace=True).mean()))
    q = quantiles(scores)
    q["bootstrap_doc_mean_95ci"] = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))] if boot else [0, 0]
    return {
        "metric": "Boundary Clarity semantic separation",
        "direction": "higher_is_better",
        "formula": "1 - cosine_similarity(chunk_i, chunk_i+1)",
        "n_documents_with_boundaries": len(doc_scores),
        "n_boundaries": len(scores),
        "score": q,
        "n_documents_without_boundary": sum(1 for g in ordered_groups if len(g) < 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_json", required=True, help="JSON file or docs directory")
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()
    chunks = read_chunks(Path(args.chunks_json))
    source = read_source_lines(Path(args.source))
    rows, mapping = assign_source_groups(chunks, source)
    lengths = [r["chunk_len"] for r in rows]
    result = {
        "chunks_json": str(Path(args.chunks_json).resolve()),
        "source": str(Path(args.source).resolve()),
        "mapping": mapping,
        "chunk_count": len(rows),
        "chunk_length": quantiles(lengths),
        "chunks_lt_10": sum(x < 10 for x in lengths),
        "chunks_lt_20": sum(x < 20 for x in lengths),
        "boundary": boundary_metric(rows, args.model_path, args.device, args.batch_size),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
