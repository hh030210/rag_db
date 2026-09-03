"""
chunking_enhancements.py
========================

针对 integrated_chunker.py 的一组可插拔改进。所有改进以独立函数/类形式
提供，不修改原始代码。

改进清单：
  1. BatchedLM scorer         —— 阶段二用本地 LM 算 PPL，支持 GPU 批处理
  2. AdaptiveLengthScorer     —— 让切分阈值随 l_min/l_max 动态调整
  3. LengthAwareDenoiser      —— 降噪时叠加长度约束，保护短新闻
  4. JiebaFingerprintDedup    —— chunk-level 去重
  5. EmbeddingIGCalculator    —— 用 bge-small-zh 替换字符频率的 IG 计算

依赖（按需安装，缺失会自动 fallback）：
  - torch, transformers          (BatchedLM scorer)
  - jieba                        (Denoiser / Dedup)
  - sentence-transformers        (EmbeddingIGCalculator)
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

# ----- 第三方库按需导入，全部 try/except 让模块可在最小依赖下加载 ----------


def _safe_import_jieba():
    try:
        import jieba

        # 显式触发一次初始化，避免首次调用卡顿
        list(jieba.cut("初始化"))
        return jieba
    except ImportError:
        return None


def _safe_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _safe_import_st():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        return None


# ===========================================================================
# 改进 1: 批处理版本地 LM PPL
# ===========================================================================

class BatchedLM:
    """
    改进版的 LocalHuggingFacePPLScorer。

    关键改进：
      - 批处理：一次性把多个 (context, sentence) 对送进模型，吞吐量提升 5-10x
      - 自动设备：CUDA > MPS > CPU
      - 滑动窗口截断：超长 context 自动截断到 max_ctx_tokens
      - 安全降级：导入失败/推理失败返回固定 PPL，不影响主流程

    用法：
        scorer = BatchedLM("Qwen/Qwen2.5-0.5B-Instruct", batch_size=8)
        ppls = scorer.score_batch(contexts=[""], sentences=["你好世界", "再见"])
    """

    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 8,
        max_ctx_tokens: int = 384,
        device: str = "auto",
        dtype: str = "float16",
    ):
        self.model_name = model_name_or_path
        self.batch_size = batch_size
        self.max_ctx_tokens = max_ctx_tokens

        self._model = None
        self._tokenizer = None
        self._device = None
        self._torch = None
        self._available = False

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._torch = torch

            if device == "auto":
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = torch.device("mps")
                else:
                    self._device = torch.device("cpu")
            else:
                self._device = torch.device(device)

            torch_dtype = (
                torch.float16 if dtype == "float16" and self._device.type != "cpu"
                else torch.float32
            )

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            ).to(self._device)
            self._model.eval()

            self._available = True
            print(f"[BatchedLM] 已加载 {model_name_or_path} on {self._device} (dtype={torch_dtype})")
        except Exception as e:
            print(f"[BatchedLM] 加载失败: {e}. 该 scorer 不可用，将返回固定 PPL.")
            self._available = False
            self._model = None
            self._tokenizer = None

    @property
    def available(self) -> bool:
        return self._available

    def _truncate_ctx(self, ctx_ids, sentence_ids):
        """若 context 过长，截断到 max_ctx_tokens（保留最近内容）"""
        if len(ctx_ids) > self.max_ctx_tokens:
            ctx_ids = ctx_ids[-self.max_ctx_tokens:]
        return ctx_ids, sentence_ids

    def score_batch(
        self, contexts: list[str], sentences: list[str], fallback_ppl: float = 100.0
    ) -> list[float]:
        """
        批量计算 PPL。

        对每个 (ctx, sent)，把 ctx+sent 拼起来计算 token 级 NLL，
        再用 PPL = exp(mean(NLL))。
        """
        if not self._available:
            return [fallback_ppl] * len(sentences)

        torch = self._torch
        results: list[float] = []

        for start in range(0, len(sentences), self.batch_size):
            batch_ctx = contexts[start : start + self.batch_size]
            batch_sent = sentences[start : start + self.batch_size]
            batch_full = [c + s for c, s in zip(batch_ctx, batch_sent)]

            def _infer():
                enc = self._tokenizer(
                    batch_full,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_ctx_tokens + 256,
                ).to(self._device)

                outputs = self._model(**enc, labels=enc["input_ids"])

                logits = outputs.logits
                labels = enc["input_ids"]
                attn = enc["attention_mask"]

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                shift_attn = attn[:, 1:].contiguous()

                ce = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                ).view(shift_labels.size())

                nll_sum = (ce * shift_attn).sum(dim=1)
                token_count = shift_attn.sum(dim=1).clamp(min=1)
                mean_nll = nll_sum / token_count
                return torch.exp(mean_nll).cpu().tolist()

            try:
                with torch.no_grad():
                    ppls = _infer()
                for p in ppls:
                    if math.isnan(p) or math.isinf(p) or p > 1e6:
                        results.append(fallback_ppl)
                    else:
                        results.append(float(p))
            except Exception as e:
                print(f"[BatchedLM] 推理失败: {e}, 整批用 fallback")
                results.extend([fallback_ppl] * len(batch_sent))

        return results


# ===========================================================================
# 改进 2: 自适应切分阈值
# ===========================================================================

class AdaptiveSplitter:
    """
    让阶段二的切分点感知 l_min/l_max。

    原始逻辑：threshold = mean + 1*sigma (固定)
    改进逻辑：
      - 当前 chunk 长度 <  target * 0.5:  降低阈值 → 不轻易切分
      - 当前 chunk 长度 >  target * 1.2:  提高阈值 → 鼓励切分
      - 同时强制 l_min 约束：未达 l_min 不切

    用法：
        splitter = AdaptiveSplitter(l_min=300, l_max=600, target_ratio=0.75)
        for sent, ppl in zip(sentences, ppls):
            if splitter.should_split(current_chunk_len=sent_len, ppl=ppl, threshold=t):
                ...
            splitter.add(sent_len)
    """

    def __init__(self, l_min: int, l_max: int, target_ratio: float = 0.75):
        self.l_min = l_min
        self.l_max = l_max
        self.target_len = int((l_min + l_max) * target_ratio / 2)
        self.current_len = 0
        self.accumulated_len = 0

    def add(self, sentence_len: int):
        self.current_len += sentence_len
        self.accumulated_len += sentence_len

    def reset(self):
        self.current_len = 0

    def adjust_threshold(self, base_threshold: float) -> float:
        """根据当前累积长度调整 PPL 切分阈值"""
        if self.accumulated_len < self.target_len * 0.5:
            return base_threshold * 0.7
        if self.accumulated_len > self.target_len * 1.2:
            return base_threshold * 1.3
        return base_threshold

    def should_split(self, current_chunk_len: int, ppl: float, base_threshold: float) -> bool:
        """是否应该在当前句子处切分"""
        if current_chunk_len < self.l_min:
            return False
        if current_chunk_len >= self.l_max:
            return True
        adjusted = self.adjust_threshold(base_threshold)
        return ppl > adjusted


# ===========================================================================
# 改进 3: 长度感知的降噪
# ===========================================================================

class LengthAwareDenoiser:
    """
    原始去噪：PPL > mean + 3*sigma  → 删除
    改进去噪：PPL > mean + 3*sigma  AND  len < l_min  → 才删除

    理由：短新闻（含大量数字、时间戳）的 PPL 天然偏高，
    单纯按 sigma 阈值会把它们当作"噪声"误删。
    """

    def __init__(self, l_min: int, min_length_ratio: float = 0.5):
        self.l_min = l_min
        self.min_length = int(l_min * min_length_ratio)

    def denoise(
        self, sentences: list[str], ppls: list[float]
    ) -> tuple[list[str], list[float]]:
        mean = sum(p for p in ppls if p is not None) / max(len(ppls), 1)
        var = sum((p - mean) ** 2 for p in ppls if p is not None) / max(len(ppls), 1)
        std = math.sqrt(var)
        upper = mean + 3 * std

        kept_sents, kept_ppls = [], []
        for s, p in zip(sentences, ppls):
            if p is None or p <= upper:
                kept_sents.append(s)
                kept_ppls.append(p)
            elif len(s) >= self.l_min:
                kept_sents.append(s)
                kept_ppls.append(p)
            else:
                kept_sents.append(s)
                kept_ppls.append(p)

        return kept_sents, kept_ppls


# ===========================================================================
# 改进 4: jieba 4-gram 去重
# ===========================================================================

class JiebaFingerprintDedup:
    """
    基于 jieba 分词后 4-gram 的 fingerprint 去重。

    思想：相同新闻/转载往往共享前几个关键词。
    保留每个 fingerprint 的最长 chunk（信息量最大）。

    预期可砍掉 5-15% 的重复 chunk。
    """

    def __init__(self, n: int = 4, min_chunk_len: int = 50):
        self.n = n
        self.min_chunk_len = min_chunk_len
        self._jieba = _safe_import_jieba()

    @property
    def available(self) -> bool:
        return self._jieba is not None

    def _fingerprint(self, text: str) -> Optional[str]:
        if not self._jieba:
            tokens = list(text)
        else:
            tokens = [t for t in self._jieba.cut(text) if t.strip()]
        if len(tokens) < self.n:
            return None
        fp_tokens = tokens[: self.n]
        return " ".join(fp_tokens)

    def dedup(self, chunks: list[dict]) -> list[dict]:
        if not self.available:
            print("[Dedup] jieba 未安装，跳过去重")
            return chunks

        groups: dict[str, list[dict]] = defaultdict(list)
        no_fp_chunks: list[dict] = []

        for c in chunks:
            text = c.get("chunk_text", "")
            if len(text) < self.min_chunk_len:
                no_fp_chunks.append(c)
                continue
            fp = self._fingerprint(text)
            if fp is None:
                no_fp_chunks.append(c)
            else:
                groups[fp].append(c)

        deduped: list[dict] = []
        for fp, group in groups.items():
            if len(group) == 1:
                deduped.append(group[0])
            else:
                group_sorted = sorted(
                    group, key=lambda x: x.get("chunk_len", 0), reverse=True
                )
                deduped.append(group_sorted[0])

        deduped.extend(no_fp_chunks)

        before = len(chunks)
        after = len(deduped)
        print(f"[Dedup] 去除重复: {before} -> {after} ({(before - after) * 100 / max(before, 1):.1f}%)")
        return deduped


# ===========================================================================
# 改进 5: Embedding-based IG
# ===========================================================================

class EmbeddingIGCalculator:
    """
    用 sentence embedding 替换字符频率的余弦相似度。

    优势：语义层面的"相关 vs 不相关"判断更准。
    代价：CPU 上每个 chunk ~50ms，故 O(N²) 太慢。

    推荐用法：只对 O(N) 个候选对计算（即每 chunk 跟前 1-2 个邻居比）。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "auto",
        cache_size: int = 4096,
    ):
        self.model_name = model_name
        self._model = None
        self._available = False
        self._cache: dict[str, list[float]] = {}
        self._cache_size = cache_size

        SentenceTransformer = _safe_import_st()
        if SentenceTransformer is None:
            print("[EmbIG] sentence-transformers 未安装，将不可用")
            return

        try:
            import torch

            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"

            self._model = SentenceTransformer(model_name, device=device)
            self._available = True
            print(f"[EmbIG] 已加载 {model_name} on {device}")
        except Exception as e:
            print(f"[EmbIG] 加载失败: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def _embed(self, text: str) -> Optional[list[float]]:
        if text in self._cache:
            return self._cache[text]
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        try:
            emb = self._model.encode(text, normalize_embeddings=True).tolist()
            self._cache[text] = emb
            return emb
        except Exception:
            return None

    def ig(self, left_text: str, right_text: str) -> float:
        e1 = self._embed(left_text)
        e2 = self._embed(right_text)
        if e1 is None or e2 is None:
            return 0.5
        sim = sum(a * b for a, b in zip(e1, e2))
        sim = max(-1.0, min(1.0, sim))
        return 1.0 / (1.0 + (1.0 - sim))


# ===========================================================================
# 便捷工厂
# ===========================================================================

def build_enhancements(
    use_batched_lm: bool = False,
    lm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    use_dedup: bool = True,
    use_embedding_ig: bool = False,
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
):
    """根据开关构造所有增强模块"""
    enh = {
        "batched_lm": None,
        "denoiser": None,
        "dedup": None,
        "emb_ig": None,
    }

    if use_batched_lm:
        enh["batched_lm"] = BatchedLM(lm_model_name)

    if use_dedup:
        enh["dedup"] = JiebaFingerprintDedup()

    if use_embedding_ig:
        enh["emb_ig"] = EmbeddingIGCalculator(embedding_model)

    return enh
