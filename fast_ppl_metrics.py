#!/usr/bin/env python3
"""Batched Qwen implementation of the paper's PPL graph metrics.

The original scripts call the causal LM once for every sentence pair.  This
adapter keeps their graph formulas but batches those pairs, making a full
comparison practical on the V100 server.  It reports both the per-chunk
Stickiness score and the per-source-document Relation Coherence score.
"""
from __future__ import annotations

import argparse
import copy
import heapq
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def split_into_sentences(text: str) -> list[str]:
    parts = re.split(r"([。！？\n]|\.(?=\s|$)|[.!?](?=\s|$))", text)
    out, i = [], 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i + 1] and parts[i + 1][0] in "。！？.!?":
            value = parts[i] + parts[i + 1]
            i += 2
        else:
            value = parts[i]
            i += 1
        if value.strip() and len(value.strip()) >= 5:
            out.append(value.strip())
    return out


def load_chunks(path: Path) -> list[dict]:
    files = sorted(path.glob("chunk_*.txt"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    return [
        {"index": i, "chunk_text": p.read_text(encoding="utf-8").strip()}
        for i, p in enumerate(files)
        if len(p.read_text(encoding="utf-8").strip()) >= 10
    ]


def load_source_groups(chunks_dir: Path, source_path: Path, adapter_path: Path) -> dict[str, list[dict]]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("paper_metric_adapter", adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    rows = adapter.read_chunks(chunks_dir)
    rows = [r for r in rows if len(r["chunk_text"]) >= 10]
    source = adapter.read_source_lines(source_path)
    assigned, mapping = adapter.assign_source_groups(rows, source)
    if mapping["unmatched_chunks"]:
        raise RuntimeError(f"source mapping has {mapping['unmatched_chunks']} unmatched chunks")
    groups = defaultdict(list)
    for row in assigned:
        groups[row["doc_id"]].append(row)
    return groups


def load_or_build_source_groups(chunks_dir: Path, source_path: Path, adapter_path: Path,
                                cache_path: Path | None) -> dict[str, list[dict]]:
    if cache_path and cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return {k: list(v) for k, v in raw.items()}
    groups = load_source_groups(chunks_dir, source_path, adapter_path)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(groups, ensure_ascii=False), encoding="utf-8")
    return groups


def batch_ppl(pairs: list[tuple[str, str]], model, tokenizer, device: str, batch_size: int) -> list[float]:
    """Return summed NLL for the second string in each pair."""
    values = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        first, second = zip(*batch)
        texts = [a + b for a, b in batch]
        first_tokens = tokenizer(list(first), add_special_tokens=False, padding=False)["input_ids"]
        second_tokens = tokenizer(list(second), add_special_tokens=False, padding=False)["input_ids"]
        sequences = [a + b for a, b in zip(first_tokens, second_tokens)]
        max_len = max(len(x) for x in sequences)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for row, seq in enumerate(sequences):
            ids[row, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[row, :len(seq)] = 1
        ids = ids.to(device)
        mask = mask.to(device)
        # Calling AutoModelForCausalLM normally materializes logits for every
        # padded position and the entire ~150k-token vocabulary.  Relation
        # needs logits only at positions whose labels belong to the second
        # string, so run the transformer backbone and project only those
        # hidden states below.  This avoids an avoidable multi-GB allocation.
        with torch.inference_mode():
            backbone = getattr(model, "model", model)
            hidden = backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            lm_head = getattr(model, "lm_head", None)
            if lm_head is None:
                raise RuntimeError("model does not expose a language-model head")
        target_ranges = []
        selected_hidden = []
        selected_labels = []
        for row, first_ids in enumerate(first_tokens):
            total = int(mask[row].sum().item())
            target_start = min(max(1, len(first_ids)), total)
            target_end = total
            if target_end <= target_start:
                target_ranges.append((len(selected_hidden), len(selected_hidden)))
                continue
            selected_hidden.append(hidden[row, target_start - 1:target_end - 1])
            selected_labels.append(ids[row, target_start:target_end])
            end_index = sum(x.shape[0] for x in selected_hidden)
            target_ranges.append((end_index - selected_hidden[-1].shape[0], end_index))
        if selected_hidden:
            # Flatten all target positions in this batch and project them in
            # larger chunks.  This preserves each request's summed NLL while
            # avoiding thousands of tiny lm_head calls.
            flat_hidden = torch.cat(selected_hidden, dim=0)
            flat_labels = torch.cat(selected_labels, dim=0)
            flat_losses = []
            with torch.no_grad():
                for pos in range(0, flat_hidden.shape[0], 512):
                    end = min(flat_hidden.shape[0], pos + 512)
                    pred = lm_head(flat_hidden[pos:end].clone()).float()
                    flat_losses.append(F.cross_entropy(pred, flat_labels[pos:end], reduction="none").detach().cpu())
            flat_losses = torch.cat(flat_losses)
            for left, right in target_ranges:
                values.append(float(flat_losses[left:right].sum().item()))
        else:
            values.extend([0.0] * len(first_tokens))
        del hidden, ids, mask
    return values


def graph_1(sentences: list[str], model, tokenizer, device: str, batch_size: int) -> tuple[list[list[float]], list[int]]:
    n = len(sentences)
    pairs = [(" ", s) for s in sentences]
    locations = [(i, i) for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                locations.append((i, j))
                pairs.append((sentences[i], sentences[j]))
    vals = batch_ppl(pairs, model, tokenizer, device, batch_size)
    graph = [[0.0] * n for _ in range(n)]
    for (i, j), value in zip(locations, vals):
        graph[i][j] = value
    token_nums = [len(x) for x in tokenizer(sentences, add_special_tokens=False, padding=False)["input_ids"]]
    return graph, token_nums


def graph_3(graph: list[list[float]], token_nums: list[int], delta: float = 0.0) -> list[list[float]]:
    n = len(graph)
    out = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            exponent = (graph[i][j] - graph[j][j]) / max(1, token_nums[j])
            exponent = max(-50.0, min(50.0, exponent))
            out[i][j] = math.exp(exponent) + delta * abs(i - j) / max(1, n - 1)
    return out


def graph_2(graph: list[list[float]]) -> list[list[float]]:
    n = len(graph)
    return [[graph[i][j] if j >= i else 0.0 for j in range(n)] for i in range(n)]


def entropy_from_edges(graph: list[list[float]], threshold: float = 0.8) -> float:
    degree = defaultdict(int)
    for i, row in enumerate(graph):
        for j, value in enumerate(row):
            if i != j and value > threshold:
                degree[i] += 1
                degree[j] += 1
    total = sum(degree.values())
    if not total:
        return 0.0
    return -sum((d / total) * math.log(d / total, 2) for d in degree.values() if d)


def dijkstra(graph: list[list[float]], start: int) -> dict[int, dict]:
    n = len(graph)
    dist = [float("inf")] * n
    count = [0] * n
    parent = [None] * n
    dist[start] = 0.0
    queue = [(0.0, start)]
    while queue:
        cur, node = heapq.heappop(queue)
        if cur > dist[node]:
            continue
        for nxt, weight in enumerate(graph[node]):
            if nxt == node or weight == 0:
                continue
            new = (cur * count[node] + weight) / (count[node] + 1)
            if new < dist[nxt]:
                dist[nxt] = new
                count[nxt] = count[node] + 1
                parent[nxt] = node
                heapq.heappush(queue, (new, nxt))
    result = {}
    for node in range(n):
        if node == start:
            continue
        path, cur = [], node
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        result[node] = {"distance": dist[node], "shortest_path": list(reversed(path))}
    return result


def dijkstra_entropy(graph: list[list[float]]) -> float:
    seen = set()
    degree = defaultdict(int)
    for start in range(len(graph)):
        for value in dijkstra(graph, start).values():
            path = value["shortest_path"]
            for a, b in zip(path, path[1:]):
                if (a, b) not in seen:
                    seen.add((a, b))
                    degree[a] += 1
                    degree[b] += 1
    total = sum(degree.values())
    if not total:
        return 0.0
    return -sum((d / total) * math.log(d / total, 2) for d in degree.values() if d)


def evaluate_stickiness(chunks, model, tokenizer, device, batch_size, max_chunks=None, start_chunk=0, end_chunk=None):
    rows = chunks if max_chunks is None else chunks[:max_chunks]
    rows = rows[start_chunk:end_chunk]
    g2, g3, skipped = [], [], 0
    t0 = time.time()
    for k, row in enumerate(rows, 1):
        sentences = split_into_sentences(row["chunk_text"])
        if len(sentences) < 3:
            skipped += 1
            continue
        graph, tokens = graph_1(sentences, model, tokenizer, device, batch_size)
        g2.append(entropy_from_edges(graph_2(graph)))
        g3.append(entropy_from_edges(graph_3(graph, tokens)))
        if k % 25 == 0:
            print(f"stickiness {k}/{len(rows)} elapsed={time.time()-t0:.1f}s", flush=True)
    def summary(values):
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        mean = sum(values) / len(values)
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        return {"mean": mean, "std": std, "min": min(values), "max": max(values)}
    return {
        "metric": "Chunk Stickiness (PPL graph structural entropy)",
        "n_chunks_input": len(rows), "n_chunks_evaluated": len(g3), "n_chunks_skipped": skipped,
        "structural_entropy_G1": summary(g2), "structural_entropy_G3": summary(g3),
        "values_G1": g2, "values_G3": g3,
        "elapsed_seconds": time.time() - t0,
    }


def evaluate_relation(groups, model, tokenizer, device, batch_size, max_docs=None):
    items = [(k, v) for k, v in sorted(groups.items()) if len(v) >= 2]
    if max_docs is not None:
        items = items[:max_docs]
    raw_full, raw_tri, path_full, path_tri = [], [], [], []
    t0 = time.time()
    # Build all per-document pair requests first, then run one global batched
    # PPL pass.  The original implementation restarted a tiny model batch for
    # every document, which is mathematically equivalent but very inefficient
    # when most documents contain only two or three chunks.
    requests = []
    layouts = []
    for doc_id, rows in items:
        segments = [x["chunk_text"] for x in rows]
        n = len(segments)
        start = len(requests)
        for s in segments:
            requests.append((" ", s))
        for i in range(n):
            for j in range(n):
                if i != j:
                    requests.append((segments[i], segments[j]))
        layouts.append((segments, start, len(requests)))
    print(f"relation pair requests={len(requests)} docs={len(items)}", flush=True)
    values = batch_ppl(requests, model, tokenizer, device, batch_size)
    print(f"relation PPL pass done elapsed={time.time()-t0:.1f}s", flush=True)
    for k, (segments, start, end) in enumerate(layouts, 1):
        n = len(segments)
        graph = [[0.0] * n for _ in range(n)]
        local = values[start:end]
        for i in range(n):
            graph[i][i] = local[i]
        pos = n
        for i in range(n):
            for j in range(n):
                if i != j:
                    graph[i][j] = local[pos]
                    pos += 1
        tokens = [len(x) for x in tokenizer(segments, add_special_tokens=False, padding=False)["input_ids"]]
        g3 = graph_3(graph, tokens)
        raw_full.append(entropy_from_edges(g3))
        raw_tri.append(entropy_from_edges(graph_2(g3)))
        path_full.append(dijkstra_entropy(g3))
        path_tri.append(dijkstra_entropy(graph_2(g3)))
        if k % 25 == 0:
            print(f"relation {k}/{len(items)} elapsed={time.time()-t0:.1f}s", flush=True)
    def mean_std(values):
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        mean = sum(values) / len(values)
        return {"mean": mean, "std": (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5,
                "min": min(values), "max": max(values)}
    return {
        "metric": "Relation Coherence (PPL graph structural entropy)",
        "n_source_documents_input": len(groups), "n_documents_evaluated": len(items),
        "n_documents_single_chunk": sum(len(v) < 2 for v in groups.values()),
        "raw_graph_full": mean_std(raw_full), "raw_graph_triangular": mean_std(raw_tri),
        "dijkstra_full": mean_std(path_full), "dijkstra_triangular": mean_std(path_tri),
        "elapsed_seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_dir", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--groups_cache", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_chunks", type=int, default=None)
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument("--start_chunk", type=int, default=0)
    ap.add_argument("--end_chunk", type=int, default=None)
    ap.add_argument("--metrics", nargs="+", choices=["stickiness", "relation"], default=["stickiness", "relation"])
    args = ap.parse_args()
    print(f"loading model {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float16,
    ).to(args.device)
    model.eval()
    chunks_dir = Path(args.chunks_dir)
    chunks = load_chunks(chunks_dir)
    result = {"model": args.model, "chunks_dir": str(chunks_dir), "batch_size": args.batch_size}
    if "stickiness" in args.metrics:
        result["stickiness"] = evaluate_stickiness(chunks, model, tokenizer, args.device, args.batch_size, args.max_chunks, args.start_chunk, args.end_chunk)
    if "relation" in args.metrics:
        groups = load_or_build_source_groups(
            chunks_dir, Path(args.source), Path(args.adapter),
            Path(args.groups_cache) if args.groups_cache else None,
        )
        result["relation"] = evaluate_relation(groups, model, tokenizer, args.device, args.batch_size, args.max_docs)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
