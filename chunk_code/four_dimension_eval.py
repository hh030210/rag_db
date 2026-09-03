#!/usr/bin/env python3
"""对一种分片结果计算四个维度的统一指标。

1. semantic perplexity: fixed local LM 的 token-weighted chunk PPL；
2. topic distance: shared TF-IDF + LSA topic space 的 chunk 内离散度和边界距离；
3. information difference content: TF-IDF 信息密度与邻域非冗余度；
4. content consistency: 图示的切分前后外部一致性和 chunk 内部一致性。
"""

import argparse
import gc
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])|\n+")


def split_sentences(text: str) -> list[str]:
    return [x.strip() for x in SENTENCE_SPLIT.split(text) if x.strip()]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def weighted_center(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = weights.astype(np.float32)
    total = float(weights.sum())
    if total <= 0:
        return values.mean(axis=0) if len(values) else np.zeros(1, dtype=np.float32)
    return (values * (weights / total)[:, None]).sum(axis=0)


def batched(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def load_chunks(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("chunks", data.get("results", []))
    result = []
    for item in data:
        text = item.get("chunk_text", item.get("text", "")) if isinstance(item, dict) else str(item)
        text = text.strip()
        if text:
            result.append(text)
    return result


def encode_texts(model, texts: list[str], batch_size: int, device: str, desc: str) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device,
    ).astype(np.float32)


def fixed_char_windows(text: str, target_chars: int) -> list[str]:
    target_chars = max(1, int(target_chars))
    return [text[i:i + target_chars] for i in range(0, len(text), target_chars)]


def fixed_sentence_windows(group: list[str], target_chars: int) -> list[list[str]]:
    """将句子按统一字符预算分成局部窗口，避免长 Chunk 天然包含更多句子。"""
    target_chars = max(1, int(target_chars))
    windows = []
    current = []
    current_chars = 0
    for sentence in group:
        sentence_chars = max(1, len(sentence))
        if current and current_chars + sentence_chars > target_chars:
            windows.append(current)
            current = []
            current_chars = 0
        current.append(sentence)
        current_chars += sentence_chars
        if sentence_chars >= target_chars:
            windows.append(current)
            current = []
            current_chars = 0
    if current:
        windows.append(current)
    return windows


def topic_vectors_for_sentences(vectorizer, svd, texts: list[str], batch_size: int) -> np.ndarray:
    outputs = []
    for batch in batched(texts, batch_size):
        x = vectorizer.transform(batch)
        outputs.append(svd.transform(x).astype(np.float32))
    return np.vstack(outputs) if outputs else np.zeros((0, svd.n_components), dtype=np.float32)


def sparse_row_cosines(matrix) -> np.ndarray:
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()).astype(np.float64)
    if matrix.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    dots = np.asarray(matrix[:-1].multiply(matrix[1:]).sum(axis=1)).ravel()
    denom = norms[:-1] * norms[1:]
    return np.divide(dots, denom, out=np.zeros_like(dots, dtype=np.float64), where=denom > 0)


def evaluate_topic_and_information(chunks, sentence_groups, vectorizer, svd, batch_size):
    chunk_matrix = vectorizer.transform(chunks)
    chunk_topic = svd.transform(chunk_matrix).astype(np.float32)

    all_sentences = [s for group in sentence_groups for s in group]
    sentence_topic = topic_vectors_for_sentences(vectorizer, svd, all_sentences, batch_size)

    intra = []
    spans = []
    pos = 0
    for group in sentence_groups:
        end = pos + len(group)
        spans.append((pos, end))
        if len(group) >= 2:
            vals = sentence_topic[pos:end]
            weights = np.asarray([max(1, len(x)) for x in group], dtype=np.float32)
            center = weighted_center(vals, weights)
            distances = [1.0 - cosine(v, center) for v in vals]
            intra.append(float(np.average(distances, weights=weights)))
        pos = end

    boundary = [1.0 - cosine(chunk_topic[i], chunk_topic[i + 1]) for i in range(len(chunk_topic) - 1)]
    in_mean = float(np.mean(intra)) if intra else 0.0
    out_mean = float(np.mean(boundary)) if boundary else 0.0

    neighbor_cos = sparse_row_cosines(chunk_matrix)
    novelty = np.ones(len(chunks), dtype=np.float64)
    if len(chunks) > 1:
        neighbor_sum = np.zeros(len(chunks), dtype=np.float64)
        neighbor_count = np.zeros(len(chunks), dtype=np.float64)
        neighbor_sum[:-1] += neighbor_cos
        neighbor_sum[1:] += neighbor_cos
        neighbor_count[:-1] += 1
        neighbor_count[1:] += 1
        novelty = 1.0 - np.divide(
            neighbor_sum,
            neighbor_count,
            out=np.zeros_like(neighbor_sum),
            where=neighbor_count > 0,
        )
    idf_density = np.asarray(
        [float(row.data.sum()) / max(len(text), 1) for row, text in zip(chunk_matrix, chunks)],
        dtype=np.float64,
    )
    idc = idf_density * novelty
    return {
        "topic_distance": {
            "intra_topic_dispersion_mean": in_mean,
            "boundary_topic_distance_mean": out_mean,
            "topic_contrast_mean": out_mean - in_mean,
            "n_chunks_with_internal_sentences": len(intra),
        },
        "information_difference": {
            "idf_density_mean": float(idf_density.mean()) if len(idf_density) else 0.0,
            "neighbor_novelty_mean": float(novelty.mean()) if len(novelty) else 0.0,
            "information_difference_content_mean": float(idc.mean()) if len(idc) else 0.0,
            "neighbor_redundancy_mean": 1.0 - (float(novelty.mean()) if len(novelty) else 0.0),
        },
    }


def evaluate_content_consistency(
    source_lines,
    chunks,
    sentence_groups,
    embedding_model,
    batch_size,
    device,
    normalization_target_chars=200,
):
    source_emb = encode_texts(embedding_model, source_lines, batch_size, device, "source")
    chunk_emb = encode_texts(embedding_model, chunks, batch_size, device, "chunks")
    all_sentences = [s for group in sentence_groups for s in group]
    sent_emb = encode_texts(embedding_model, all_sentences, batch_size, device, "sentences")

    source_weights = np.asarray([max(1, len(x)) for x in source_lines], dtype=np.float32)
    chunk_weights = np.asarray([max(1, len(x)) for x in chunks], dtype=np.float32)
    before = weighted_center(source_emb, source_weights)
    after = weighted_center(chunk_emb, chunk_weights)

    internal_scores = []
    pos = 0
    for group in sentence_groups:
        end = pos + len(group)
        if len(group) >= 2:
            vals = sent_emb[pos:end]
            weights = np.asarray([max(1, len(x)) for x in group], dtype=np.float32)
            center = weighted_center(vals, weights)
            internal_scores.append(float(np.average([cosine(v, center) for v in vals], weights=weights)))
        pos = end

    internal = float(np.mean(internal_scores)) if internal_scores else 0.0

    # 长度均一化：把原文和 Chunk 都转换为相同字符预算的评价单元，
    # 再计算外部中心向量；内部一致性则在固定预算的局部句子窗口上计算。
    source_units = [
        unit for text in source_lines
        for unit in fixed_char_windows(text, normalization_target_chars)
    ]
    chunk_units = [
        unit for text in chunks
        for unit in fixed_char_windows(text, normalization_target_chars)
    ]
    source_unit_emb = encode_texts(
        embedding_model, source_units, batch_size, device, "normalized_source"
    )
    chunk_unit_emb = encode_texts(
        embedding_model, chunk_units, batch_size, device, "normalized_chunks"
    )
    source_unit_weights = np.asarray(
        [max(1, len(x)) for x in source_units], dtype=np.float32
    )
    chunk_unit_weights = np.asarray(
        [max(1, len(x)) for x in chunk_units], dtype=np.float32
    )
    normalized_external = cosine(
        weighted_center(source_unit_emb, source_unit_weights),
        weighted_center(chunk_unit_emb, chunk_unit_weights),
    )

    normalized_internal_scores = []
    normalized_internal_weights = []
    normalized_window_count = 0
    normalized_eligible_windows = 0
    pos = 0
    for group in sentence_groups:
        end = pos + len(group)
        vals = sent_emb[pos:end]
        local_pos = 0
        for window in fixed_sentence_windows(group, normalization_target_chars):
            n = len(window)
            normalized_window_count += 1
            local_vals = vals[local_pos:local_pos + n]
            local_pos += n
            if n >= 2:
                weights = np.asarray([max(1, len(x)) for x in window], dtype=np.float32)
                center = weighted_center(local_vals, weights)
                score = float(
                    np.average(
                        [cosine(v, center) for v in local_vals],
                        weights=weights,
                    )
                )
                normalized_internal_scores.append(score)
                normalized_internal_weights.append(float(weights.sum()))
                normalized_eligible_windows += 1
        pos = end

    if normalized_internal_scores:
        normalized_internal = float(
            np.average(
                normalized_internal_scores,
                weights=np.asarray(normalized_internal_weights, dtype=np.float32),
            )
        )
    else:
        normalized_internal = 0.0
    normalized_coverage = normalized_eligible_windows / max(normalized_window_count, 1)
    normalized_combined = 0.5 * normalized_external + 0.5 * normalized_internal
    return {
        "external_consistency": cosine(before, after),
        "internal_consistency": internal,
        "internal_coverage": len(internal_scores) / max(len(chunks), 1),
        "combined_consistency": 0.5 * cosine(before, after) + 0.5 * internal,
        "length_normalized_external_consistency": normalized_external,
        "length_normalized_internal_consistency": normalized_internal,
        "length_normalized_internal_coverage": normalized_coverage,
        "length_normalized_combined_consistency": normalized_combined,
        "length_normalization_target_chars": int(normalization_target_chars),
        "length_normalized_window_count": normalized_window_count,
        "length_normalized_eligible_window_count": normalized_eligible_windows,
    }


def evaluate_ppl(chunks, model_path, device, batch_size, max_length):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # PPL 对累积损失的数值稳定性要求较高；半精度 Qwen logits 在长批次上
    # 可能产生非有限交叉熵，导致最终均值为 NaN。这里统一使用 float32。
    dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    truncated = 0
    with torch.inference_mode():
        for batch in batched(chunks, batch_size):
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            if any(len(tokenizer(x, add_special_tokens=True)["input_ids"]) > max_length for x in batch):
                truncated += sum(
                    len(tokenizer(x, add_special_tokens=True)["input_ids"]) > max_length
                    for x in batch
                )
            input_ids = encoded["input_ids"].to(device)
            attention = encoded["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            mask = attention[:, 1:].bool()
            losses = F.cross_entropy(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view_as(shift_labels)
            total_nll += float(losses[mask].sum().item())
            total_tokens += int(mask.sum().item())

    del model, tokenizer
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    mean_log_ppl = total_nll / max(total_tokens, 1)
    return {
        "mean_log_ppl": mean_log_ppl,
        "ppl": math.exp(min(mean_log_ppl, 20.0)),
        "tokens_scored": total_tokens,
        "truncated_chunks": truncated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--chunks_json", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--topic_model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding_model", required=True)
    parser.add_argument("--ppl_model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    parser.add_argument("--topic_batch_size", type=int, default=512)
    parser.add_argument("--ppl_batch_size", type=int, default=4)
    parser.add_argument("--ppl_max_length", type=int, default=1024)
    parser.add_argument("--consistency_target_chars", type=int, default=200)
    args = parser.parse_args()

    source_lines = [x.strip() for x in Path(args.source).read_text(encoding="utf-8").splitlines() if x.strip()]
    chunks = load_chunks(Path(args.chunks_json))
    sentence_groups = [split_sentences(x) or [x] for x in chunks]
    topic_package = joblib.load(args.topic_model)
    vectorizer = topic_package["vectorizer"]
    svd = topic_package["svd"]

    print(f"[{args.method}] chunks={len(chunks)} source_lines={len(source_lines)} device={args.device}", flush=True)
    topic_info = evaluate_topic_and_information(
        chunks, sentence_groups, vectorizer, svd, args.topic_batch_size
    )

    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(args.embedding_model, device=args.device)
    content = evaluate_content_consistency(
        source_lines, chunks, sentence_groups, embedding_model,
        args.embedding_batch_size, args.device,
        args.consistency_target_chars,
    )
    del embedding_model
    gc.collect()

    ppl = evaluate_ppl(
        chunks, args.ppl_model, args.device,
        args.ppl_batch_size, args.ppl_max_length,
    )

    result = {
        "method": args.method,
        "source": args.source,
        "chunks_json": args.chunks_json,
        "n_chunks": len(chunks),
        "input_chars": sum(len(x) for x in source_lines),
        "chunk_chars": sum(len(x) for x in chunks),
        "models": {
            "embedding": args.embedding_model,
            "ppl": args.ppl_model,
            "topic_model": args.topic_model,
        },
        "metrics": {
            "semantic_perplexity": ppl,
            **topic_info,
            "content_consistency": content,
        },
        "direction": {
            "semantic_perplexity_mean_log_ppl": "lower_is_better",
            "topic_intra_dispersion": "lower_is_better",
            "topic_boundary_distance": "higher_is_better",
            "information_difference_content": "higher_is_better",
            "content_consistency": "higher_is_better",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.method}] saved {output}", flush=True)


if __name__ == "__main__":
    main()
