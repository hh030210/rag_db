"""
enhanced_chunker.py
===================

对 integrated_chunker.py 的增强版分片器。

可插拔开关（CLI）：
  --use_batched_lm        启用 BatchedLM PPL scorer（替代字符 n-gram）
  --use_dedup             启用 jieba 4-gram 去重
  --use_embedding_ig      启用 bge-small-zh 替代字符频率的 IG 融合

内部增强（默认开启）：
  - AdaptiveSplitter：阶段二切分点感知 l_min/l_max
  - LengthAwareDenoiser：降噪叠加长度约束

用法示例：
  # 与原版完全兼容
  python enhanced_chunker.py --input data/db_qa.txt --line_mode --output out_v3

  # 启用所有改进
  python enhanced_chunker.py --input data/db_qa.txt --line_mode --output out_v3 \\
      --use_batched_lm --batched_lm_model Qwen/Qwen2.5-0.5B-Instruct \\
      --use_dedup --use_embedding_ig

依赖：
  - jieba (去重)
  - torch + transformers (BatchedLM，可选)
  - sentence-transformers (EmbeddingIG，可选)
  - 缺失时自动 fallback 到原版行为
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Optional

# 复用原 chunker 的工具类
from integrated_chunker import (
    ChunkBlock,
    IGCalculator,
    detect_genre_hint,
    GENRE_RECOMMENDATIONS,
    call_llm_recommend,
    CharNgramPPLScorer,
    LocalHuggingFacePPLScorer,
    split_by_structure,
    split_into_sentences,
    calc_mean_std,
)

# 引入新增强模块
from chunking_enhancements import (
    BatchedLM,
    AdaptiveSplitter,
    LengthAwareDenoiser,
    JiebaFingerprintDedup,
    EmbeddingIGCalculator,
)


@dataclass
class EnhancedConfig:
    """所有增强开关的配置"""
    use_batched_lm: bool = False
    batched_lm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    use_dedup: bool = True
    use_embedding_ig: bool = False
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    target_ratio: float = 0.75
    denoise_min_length_ratio: float = 0.5


class EnhancedChunker:
    """
    三阶段增强分片器。

    与原 IntegratedChunker 的差异：
      1. PPL scorer 可选 BatchedLM (本地 LM 批处理)
      2. 阶段二切分点感知 l_min/l_max (AdaptiveSplitter)
      3. 降噪时叠加长度约束 (LengthAwareDenoiser)
      4. 阶段三 IG 融合可选用 bge-small embedding
      5. 输出阶段可选去重 (JiebaFingerprintDedup)
    """

    def __init__(self, args, config: EnhancedConfig):
        self.window_w = getattr(args, "window_w", 3)
        self.beta_small = getattr(args, "beta_small", 0.8)
        self.beta = getattr(args, "beta", 1.1)
        self.denoise = getattr(args, "denoise", True)
        self.line_mode = getattr(args, "line_mode", False)
        self.llm_api_key = getattr(args, "llm_api_key", "")
        self.llm_base_url = getattr(args, "llm_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.llm_model = getattr(args, "llm_model", "qwen3-8b")
        self.llm_sample_interval = getattr(args, "llm_sample_interval", 300)
        self.ppl_model_name = getattr(args, "ppl_model_name", "")
        self.config = config

        # PPL scorer: 优先级 BatchedLM > 原 LocalHuggingFacePPLScorer > CharNgramPPLScorer
        self._ppl_scorer = None
        self._batched_lm: Optional[BatchedLM] = None
        if config.use_batched_lm:
            self._batched_lm = BatchedLM(config.batched_lm_model)
            if self._batched_lm.available:
                self._ppl_scorer_kind = "batched_lm"
            else:
                self._batched_lm = None
                self._ppl_scorer_kind = self._init_fallback_scorer()
        else:
            self._ppl_scorer_kind = self._init_fallback_scorer()

        # 阶段三 IG calculator
        if config.use_embedding_ig:
            self._emb_ig = EmbeddingIGCalculator(config.embedding_model)
            if self._emb_ig.available:
                self._ig_calc = self._emb_ig
                self._ig_kind = "embedding"
            else:
                self._ig_calc = IGCalculator()
                self._ig_kind = "char_freq"
        else:
            self._ig_calc = IGCalculator()
            self._ig_kind = "char_freq"

        # 去重器
        self._dedup: Optional[JiebaFingerprintDedup] = None
        if config.use_dedup:
            self._dedup = JiebaFingerprintDedup()

        print(f"[EnhancedChunker] PPL={self._ppl_scorer_kind}, IG={self._ig_kind}, dedup={'on' if self._dedup and self._dedup.available else 'off'}")

    def _init_fallback_scorer(self):
        """初始化 PPL scorer 兜底逻辑"""
        if self.ppl_model_name:
            self._ppl_scorer = LocalHuggingFacePPLScorer(self.ppl_model_name)
            return "local_hf"
        else:
            self._ppl_scorer = CharNgramPPLScorer()
            return "char_ngram"

    # ---------- 阶段一：参数推荐（沿用原逻辑） ----------
    def _get_recommendation(self, content: str, subfile_idx: int, line_idx: Optional[int] = None):
        genre = detect_genre_hint(content)
        should_call_llm = (
            self.llm_sample_interval > 0
            and self.llm_api_key
            and (line_idx is None or line_idx % self.llm_sample_interval == 0)
        )
        if should_call_llm:
            rec = call_llm_recommend(content, self.llm_api_key, self.llm_base_url, self.llm_model)
        else:
            rec = None

        if rec:
            genre = rec.get("genre", genre or "doc")
            l_min = rec.get("l_min", GENRE_RECOMMENDATIONS.get(genre, GENRE_RECOMMENDATIONS["doc"])["l_min"])
            l_max = rec.get("l_max", GENRE_RECOMMENDATIONS.get(genre, GENRE_RECOMMENDATIONS["doc"])["l_max"])
        else:
            rec = GENRE_RECOMMENDATIONS.get(genre, GENRE_RECOMMENDATIONS["doc"])
            l_min, l_max = rec["l_min"], rec["l_max"]
        return genre, l_min, l_max

    # ---------- PPL 计算（统一接口） ----------
    def _compute_ppls(self, sentences: list[str], context_text: str = "") -> list[float]:
        if self._ppl_scorer_kind == "batched_lm" and self._batched_lm is not None:
            contexts = []
            for i in range(len(sentences)):
                ctx_start = max(0, i - self.window_w)
                contexts.append("".join(sentences[ctx_start:i]))
            return self._batched_lm.score_batch(contexts, sentences)
        elif self._ppl_scorer_kind == "local_hf" and self._ppl_scorer is not None:
            ppls = []
            for i, sent in enumerate(sentences):
                ctx_start = max(0, i - self.window_w)
                context = "".join(sentences[ctx_start:i])
                try:
                    ppls.append(min(self._ppl_scorer.score(context, sent), 1e6))
                except Exception:
                    ppls.append(100.0)
            return ppls
        else:
            # char ngram fallback
            assert isinstance(self._ppl_scorer, CharNgramPPLScorer)
            self._ppl_scorer.train(context_text)
            ppls = []
            for i, sent in enumerate(sentences):
                ctx_start = max(0, i - self.window_w)
                context = "".join(sentences[ctx_start:i])
                try:
                    ppls.append(min(self._ppl_scorer.score(context, sent), 1e6))
                except Exception:
                    ppls.append(100.0)
            return ppls

    # ---------- 阶段二：PPL 去噪 + 切分（增强版） ----------
    def _round2_enhanced(self, sentences: list[str], l_min: int, l_max: int) -> list[str]:
        if len(sentences) < 2:
            return ["".join(sentences)] if sentences else []

        context_text = "".join(sentences)
        ppls = self._compute_ppls(sentences, context_text)

        # 改进3: 长度感知降噪
        if self.denoise:
            denoiser = LengthAwareDenoiser(l_min=l_min, min_length_ratio=self.config.denoise_min_length_ratio)
            denoised, ppls_denoised = denoiser.denoise(sentences, ppls)
        else:
            denoised, ppls_denoised = sentences, ppls

        if not denoised:
            return ["".join(sentences)]

        # 重新计算去噪后的 PPL
        ppls2 = self._compute_ppls(denoised, "".join(denoised))

        # 改进2: 自适应切分
        splitter = AdaptiveSplitter(l_min=l_min, l_max=l_max, target_ratio=self.config.target_ratio)
        mean2, std2 = calc_mean_std(ppls2)
        base_t = mean2 + std2

        chunks = []
        curr = []
        curr_len = 0
        for s, p in zip(denoised, ppls2):
            s_len = len(s)
            splitter.add(s_len)

            if curr and splitter.should_split(curr_len, p, base_t):
                chunks.append("".join(curr))
                curr = [s]
                curr_len = s_len
                splitter.reset()
            else:
                curr.append(s)
                curr_len += s_len

        if curr:
            chunks.append("".join(curr))

        return chunks

    # ---------- 阶段三：IG 融合（增强版） ----------
    def _round3_enhanced(self, chunks: list[ChunkBlock]) -> list[ChunkBlock]:
        final = []
        for b in chunks:
            if b.length > b.l_max:
                units = [ChunkBlock(s, len(s), b.l_min, b.l_max) for s in split_into_sentences(b.text)]
                units = [u for u in units if u.length > 0]
                if not units:
                    final.append(b)
                    continue

                # 改进5: 用 embedding IG (每次只比较相邻对)
                while len(units) > 1:
                    best_ig, best_i = -1.0, -1
                    for i in range(len(units) - 1):
                        if units[i].length + units[i + 1].length <= b.l_max:
                            if self._ig_kind == "embedding":
                                ig = self._ig_calc.ig(units[i].text, units[i + 1].text)
                            else:
                                ig = self._ig_calc.ig(units[i], units[i + 1])
                            if ig > best_ig:
                                best_ig, best_i = ig, i
                    if best_i == -1:
                        break
                    merged = ChunkBlock(
                        units[best_i].text + units[best_i + 1].text,
                        units[best_i].length + units[best_i + 1].length,
                        b.l_min, b.l_max,
                    )
                    units[best_i] = merged
                    units.pop(best_i + 1)

                for u in units:
                    if u.length <= b.l_max:
                        final.append(u)
                    else:
                        final.append(ChunkBlock(u.text[:b.l_max], b.l_max, b.l_min, b.l_max))
            else:
                final.append(b)

        # 融合过小 chunk
        optimized = []
        for b in final:
            if not optimized or b.length >= b.l_min * self.beta_small:
                optimized.append(b)
            else:
                if optimized[-1].length + b.length <= b.l_max * self.beta:
                    optimized[-1] = ChunkBlock(
                        optimized[-1].text + b.text,
                        optimized[-1].length + b.length,
                        optimized[-1].l_min,
                        optimized[-1].l_max,
                    )
                else:
                    optimized.append(b)
        return optimized

    # ---------- 子文件 / 文档级 ----------
    def chunk_subfile(self, content: str, doc_id: str, subfile_idx: int, line_idx: Optional[int] = None):
        genre, l_min, l_max = self._get_recommendation(content, subfile_idx, line_idx)
        sentences = split_into_sentences(content)
        raw_chunks = self._round2_enhanced(sentences, l_min, l_max)
        chunk_blocks = [ChunkBlock(c, len(c), l_min, l_max) for c in raw_chunks]
        return self._round3_enhanced(chunk_blocks)

    def chunk_document(self, content: str, filename: str, line_idx: Optional[int] = None):
        doc_id = pathlib.Path(filename).stem
        subfiles = split_by_structure(content)
        all_chunks = []
        for idx, sf_content in enumerate(subfiles):
            sf_id = f"{doc_id}_sub_{idx:03d}"
            blocks = self.chunk_subfile(sf_content, doc_id, idx, line_idx)
            for block in blocks:
                all_chunks.append({
                    "doc_id": doc_id,
                    "subfile_id": sf_id,
                    "genre": detect_genre_hint(sf_content),
                    "chunk_text": block.text,
                    "chunk_len": block.length,
                    "l_min": block.l_min,
                    "l_max": block.l_max,
                })
        return all_chunks

    # ---------- 主入口 ----------
    def run(self, input_path: pathlib.Path, output_path: pathlib.Path):
        output_path.mkdir(parents=True, exist_ok=True)
        all_results = []
        summary = {
            "version": "enhanced_v2",
            "ppl_scorer": self._ppl_scorer_kind,
            "ig_kind": self._ig_kind,
            "dedup_enabled": bool(self._dedup and self._dedup.available),
            "total_files": 0,
            "total_chunks_before_dedup": 0,
            "total_chunks": 0,
            "files": [],
        }

        files = [input_path] if input_path.is_file() else list(input_path.glob("*.txt")) + list(input_path.glob("*.md"))
        t0 = time.time()
        for fpath in files:
            content = fpath.read_text(encoding="utf-8")
            if self.line_mode:
                lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
                file_chunks_count = 0
                for line_idx, line_content in enumerate(lines):
                    doc_id = f"{fpath.stem}_line_{line_idx:05d}"
                    chunks = self.chunk_document(line_content, doc_id, line_idx=line_idx)
                    for c in chunks:
                        c["doc_id"] = doc_id
                        c["source_file"] = fpath.name
                    all_results.extend(chunks)
                    file_chunks_count += len(chunks)
                summary["total_files"] += 1
                summary["total_chunks_before_dedup"] += file_chunks_count
                summary["files"].append({"filename": fpath.name, "lines": len(lines), "chunks": file_chunks_count})
                print(f"  {fpath.name}: {len(lines)} 行 -> {file_chunks_count} chunks (累计 {time.time()-t0:.1f}s)")
            else:
                chunks = self.chunk_document(content, fpath.name)
                all_results.extend(chunks)
                summary["total_files"] += 1
                summary["total_chunks_before_dedup"] += len(chunks)
                summary["files"].append({"filename": fpath.name, "chunks": len(chunks)})

        # 改进4: 去重
        if self._dedup and self._dedup.available:
            all_results = self._dedup.dedup(all_results)
        summary["total_chunks"] = len(all_results)
        summary["elapsed_sec"] = round(time.time() - t0, 1)

        # 输出
        chunks_json = [{"chunk_text": c["chunk_text"], "chunk_len": c["chunk_len"]} for c in all_results]
        (output_path / "all_chunks_chunks.json").write_text(
            json.dumps(chunks_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_path / "all_chunks_chunks.txt").write_text(
            "\n\n---\n\n".join(c["chunk_text"] for c in chunks_json), encoding="utf-8"
        )
        (output_path / "all_chunks_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"\n[Done] {summary['total_files']} files, {summary['total_chunks_before_dedup']} -> {summary['total_chunks']} chunks in {summary['elapsed_sec']}s")
        print(f"  保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="增强版三阶段智能分片器")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--line_mode", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--llm_api_key", type=str, default="")
    parser.add_argument("--llm_base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--llm_model", type=str, default="qwen3-8b")
    parser.add_argument("--ppl_model_name", type=str, default="")
    parser.add_argument("--llm_sample_interval", type=int, default=300)
    parser.add_argument("--window_w", type=int, default=3)
    parser.add_argument("--beta_small", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=1.1)
    parser.add_argument("--denoise", type=lambda x: x.lower() != "false", default=True)

    # 增强开关
    parser.add_argument("--use_batched_lm", action="store_true",
                        help="启用 BatchedLM PPL scorer（本地 LM 批处理）")
    parser.add_argument("--batched_lm_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--use_dedup", action="store_true", default=True)
    parser.add_argument("--no_dedup", dest="use_dedup", action="store_false")
    parser.add_argument("--use_embedding_ig", action="store_true",
                        help="用 bge-small-zh 替代字符频率做 IG 融合")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--target_ratio", type=float, default=0.75,
                        help="切分目标长度占 l_min/l_max 均值的比例")
    parser.add_argument("--denoise_min_length_ratio", type=float, default=0.5,
                        help="降噪时叠加的长度保护比例")

    args = parser.parse_args()
    config = EnhancedConfig(
        use_batched_lm=args.use_batched_lm,
        batched_lm_model=args.batched_lm_model,
        use_dedup=args.use_dedup,
        use_embedding_ig=args.use_embedding_ig,
        embedding_model=args.embedding_model,
        target_ratio=args.target_ratio,
        denoise_min_length_ratio=args.denoise_min_length_ratio,
    )
    chunker = EnhancedChunker(args, config)
    chunker.run(pathlib.Path(args.input), pathlib.Path(args.output))


if __name__ == "__main__":
    main()
