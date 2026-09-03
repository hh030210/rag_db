"""
Integrated Chunking System
基于三阶段分片算法的文档智能分片器
"""

import json
import re
import math
import heapq
import statistics
import argparse
import pathlib
import os
from dataclasses import dataclass, field
from typing import Optional
from math import sqrt
from time import sleep


# ==============================================================================
# 辅助工具类
# ==============================================================================

class TextCounter:
    """字符频率计数器，用于信息增益计算"""
    def __init__(self):
        self.data = {}

    def update(self, text: str):
        for ch in text:
            self.data[ch] = self.data.get(ch, 0) + 1

    def items(self):
        return self.data.items()

    def __len__(self):
        return sum(self.data.values())


# ==============================================================================
# Round 1: 体裁检测与参数推荐
# ==============================================================================

# 内置默认推荐表（无 LLM API Key 时使用）
GENRE_RECOMMENDATIONS = {
    "doc":   {"l_min": 400,  "l_max": 1000, "reasoning": "默认技术文档"},
    "news":  {"l_min": 300,  "l_max": 600,  "reasoning": "新闻资讯宜短"},
    "paper": {"l_min": 500,  "l_max": 1200, "reasoning": "学术论文可较长"},
    "novel": {"l_min": 400,  "l_max": 800,  "reasoning": "小说故事适中"},
    "chat":  {"l_min": 200,  "l_max": 500,  "reasoning": "对话记录更短"},
}

SYSTEM_PROMPT = """你是一个专业的文档分片专家。根据文档的体裁和内容长度，
为分片算法推荐合适的最小和最大分片长度。

输出格式要求为JSON，包含：
- genre: 体裁类型
- l_min: 最小分片长度（字符数）
- l_max: 最大分片长度（字符数）
- reasoning: 推荐理由（简短）

注意：
- 对于新闻/短资讯，l_max 应较小（500-800）
- 对于技术文档/论文，l_max 可以较大（800-1500）
- 对于小说/故事，l_max 适中（600-1200）
- l_min 通常为 l_max 的 40%-60%"""


def detect_genre_hint(content: str) -> Optional[str]:
    """
    通过关键词检测体裁（不调用 LLM 时的快速判断）
    """
    content_lower = content.lower()

    if any(kw in content_lower for kw in ["摘要", "abstract", "参考文献", "引言", "结论"]):
        return "paper"
    if any(kw in content_lower for kw in ["日报", "快讯", "据报道", "新华社"]):
        return "news"
    if any(kw in content_lower for kw in ["\u201c", "\u201d", '"', "他说着", "她说道", "心想"]):
        return "novel"
    if content.count("：") / max(len(content), 1) > 0.02:
        return "chat"

    return "doc"


def call_llm_recommend(content: str, api_key: str, base_url: str, model: str) -> Optional[dict]:
    """
    通过 LLM API 获取分片参数推荐（3 次指数退避重试）
    """
    if not api_key:
        return None

    for attempt in range(3):
        try:
            import urllib.request
            import urllib.error

            print(f"[LLM] call {model} attempt {attempt+1}/3 ...")

            payload = json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"doc preview 500 chars:\\n{content[:500]}"}
                ],
                "temperature": 0.1,
                "max_tokens": 200,
                "stream": True,
            }).encode("utf-8")

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            req = urllib.request.Request(
                f"{base_url.rstrip('/')}/chat/completions",
                data=payload,
                headers=headers,
                method="POST",
            )

            collected = []
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        c = delta.get("content")
                        if c:
                            collected.append(c)
                    except json.JSONDecodeError:
                        continue

            full_text = "".join(collected)
            s = full_text.find("{")
            e = full_text.rfind("}") + 1
            if s != -1 and e != 0:
                parsed = json.loads(full_text[s:e])
                print(f"[LLM] {model} -> genre={parsed.get('genre')}, l_min={parsed.get('l_min')}, l_max={parsed.get('l_max')}")
                return parsed
            else:
                print(f"[LLM] {model} no JSON: {full_text[:200]}")
                return None
        except urllib.error.HTTPError as ex:
            if ex.code in (429, 500, 502, 503):
                w = (2 ** attempt) * 5
                print(f"[LLM] HTTP {ex.code} retry after {w}s ...")
                sleep(w)
                continue
            body = ex.read().decode("utf-8", errors="replace")
            print(f"[LLM] HTTP {ex.code}: {body[:300]}")
            break
        except Exception as ex2:
            print(f"[LLM] fail (attempt {attempt+1}): {ex2}")
            sleep((2 ** attempt) * 5)
            continue

    return None


# ==============================================================================
# Round 2: PPL 计算
# ==============================================================================

@dataclass
class ChunkBlock:
    """分片块"""
    text: str
    length: int
    l_min: int
    l_max: int


class CharNgramPPLScorer:
    """
    字符级 N-gram 困惑度计算器（无需外部依赖）
    P(ch|prev) = (count(prev,ch) + 1) / (count(prev) + |vocab|)
    PPL = exp(-1/N * sum(log(P(ch_i|prev_i))))
    """

    def __init__(self, n: int = 2):
        self.n = n
        self.vocab_size = 65536  # 假设 Unicode 范围
        self.bigram_counts: dict[tuple, int] = {}
        self.unigram_counts: dict[str, int] = {}
        self._trained = False
        self._trained_corpus = ""

    def train(self, corpus: str):
        """训练 N-gram 模型"""
        for i in range(len(corpus) - self.n + 1):
            ngram = tuple(corpus[i:i + self.n])
            self.bigram_counts[ngram] = self.bigram_counts.get(ngram, 0) + 1

        for ch in corpus:
            self.unigram_counts[ch] = self.unigram_counts.get(ch, 0) + 1

        self._trained = True
        self._trained_corpus = corpus

    def _score_sentence(self, context: str, sentence: str) -> float:
        """计算一个句子的 PPL（使用前 n-1 个字符作为上下文）"""
        if self._trained_corpus != context:
            self.train(context)

        text = context + sentence
        nll_sum = 0.0
        char_count = 0

        for i in range(self.n - 1, len(text)):
            context_chars = text[i - self.n + 1:i]
            ch = text[i]
            key = tuple(list(context_chars) + [ch])
            ctx_key = tuple(context_chars)

            ctx_total = self.unigram_counts.get(ctx_key[0], 0) if ctx_key else len(self.unigram_counts)

            count = self.bigram_counts.get(key, 0)
            prob = (count + 1) / (ctx_total + self.vocab_size)

            if prob > 0:
                nll_sum += math.log(prob)
            char_count += 1

        if char_count == 0:
            return 100.0

        return math.exp(-nll_sum / char_count)

    def score(self, context: str, sentence: str) -> float:
        """对外接口：计算 PPL"""
        return self._score_sentence(context, sentence)


class LocalHuggingFacePPLScorer:
    """
    使用本地 HuggingFace 因果语言模型计算 PPL
    PPL = exp(mean(NLL))
    """

    def __init__(self, model_name_or_path: str):
        self.model_name = model_name_or_path
        self.model = None
        self.tokenizer = None
        self._device = None

    def _ensure_model_loaded(self):
        if self.model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True, torch_dtype=torch.float16
            )
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self._device)
            self.model.eval()
        except ImportError:
            raise ImportError(
                "请安装 torch 和 transformers: pip install torch transformers"
            )

    def score(self, context: str, sentence: str) -> float:
        """计算句子的 PPL"""
        self._ensure_model_loaded()

        import torch
        from transformers import AutoModelForCausalLM

        full_text = context + sentence

        try:
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                nll = outputs.loss.item() * inputs["input_ids"].shape[1]

            return math.exp(nll / inputs["input_ids"].shape[1])
        except Exception:
            return 100.0


# ==============================================================================
# Round 3: 信息增益计算器
# ==============================================================================

class IGCalculator:
    """信息增益计算器，基于字符频率向量的余弦相似度"""

    def embedding(self, text: str):
        """将文本表示为字符频率的归一化向量"""
        counter = {}
        for ch in text:
            counter[ch] = counter.get(ch, 0) + 1

        norm = sqrt(sum(v * v for v in counter.values())) or 1.0
        return {k: v / norm for k, v in counter.items()}

    def ig(self, left: ChunkBlock, right: ChunkBlock) -> float:
        """
        两个文本块的信息增益 = 向量相似度
        sim = sum(e1[k] * e2[k])（归一化向量点积）
        ig = 1 / (1 + (1 - sim)) -> sim 越高，ig 越高
        """
        e1 = self.embedding(left.text)
        e2 = self.embedding(right.text)

        keys = set(e1.keys()) & set(e2.keys())
        sim = max(-1.0, min(1.0, sum(e1[k] * e2[k] for k in keys)))
        return 1.0 / (1.0 + (1.0 - sim))


# ==============================================================================
# 结构边界检测
# ==============================================================================

STRUCT_PATTERNS = [
    re.compile(r"^\s*={3,}.+={0,}\s*$", re.I),   # ===== 标题 =====
    re.compile(r"^\s*#{1,6}\s+.+$"),               # Markdown 标题
    re.compile(r"^\s*-{20,}\s*$"),                  # 长分隔线
    re.compile(r"^\s*-{3}\s*$"),                   # --- 分隔线
]

SENT_PATTERN = re.compile(
    r"[^。！？；!?;\n" + "\u201c\u201d" + r"]+[。！？；!?;" + "\u201c\u201d" + r"]?"
)


def split_by_structure(text: str):
    """
    按文档结构（标题/分隔线）将文本拆分成若干 subfiles
    """
    lines = text.split("\n")
    subfiles = []
    current = []

    for line in lines:
        is_boundary = any(p.match(line) for p in STRUCT_PATTERNS)
        if is_boundary and current:
            content = "\n".join(current).strip()
            if content:
                subfiles.append(content)
            current = []
        current.append(line)

    if current:
        content = "\n".join(current).strip()
        if content:
            subfiles.append(content)

    return subfiles


def split_into_sentences(text: str):
    """
    将文本按句子拆分（中文优先）
    """
    sentences = []
    matches = SENT_PATTERN.findall(text)
    for m in matches:
        s = m.strip()
        if s:
            sentences.append(s)
    return sentences


# ==============================================================================
# 统计工具
# ==============================================================================

def calc_mean_std(values):
    """计算均值和标准差"""
    valid = [v for v in values if v is not None and not math.isnan(v) and not math.isinf(v)]
    if not valid:
        return 100.0, 50.0
    return statistics.mean(valid), statistics.stdev(valid) if len(valid) > 1 else 0.0


def calc_mad(values, k: float = 3.5):
    """
    计算 MAD（中位数绝对偏差）阈值，比 3-sigma 对重尾分布更鲁棒。
    Z-score for normal distribution: 1.4826 * MAD ≈ std under normality.
    """
    valid = [v for v in values if v is not None and not math.isnan(v) and not math.isinf(v)]
    if not valid:
        return 1e6
    median = statistics.median(valid)
    mad = statistics.median([abs(v - median) for v in valid])
    if mad < 1e-9:
        return median + k * 1e-3
    threshold = median + k * mad * 1.4826
    return threshold


# ==============================================================================
# 主分片器
# ==============================================================================

class IntegratedChunker:
    """
    三阶段智能分片器

    Round 1: 结构拆分 + LLM 推荐分片参数
    Round 2: PPL（困惑度）去噪与切分
    Round 3: 策略优化融合
    """

    def __init__(self, args):
        self.window_w = getattr(args, "window_w", 3)
        self.beta_small = getattr(args, "beta_small", 0.8)
        self.beta = getattr(args, "beta", 1.1)
        self.mad_k = getattr(args, "mad_k", 3.5)
        self.denoise = getattr(args, "denoise", True)
        self.disable_structure_split = getattr(args, "disable_structure_split", False)
        self.disable_optimization = getattr(args, "disable_optimization", False)
        self.ppl_model_name = getattr(args, "ppl_model_name", "")
        self.llm_api_key = getattr(args, "llm_api_key", "")
        self.llm_base_url = getattr(args, "llm_base_url", "https://api.openai.com/v1")
        self.llm_model = getattr(args, "llm_model", "gpt-3.5-turbo")
        self.ig_calc = IGCalculator()
        self.line_mode = getattr(args, "line_mode", False)
        self.llm_sample_interval = getattr(args, "llm_sample_interval", 300)
        self._denoise_stats = {
            "round2_sentences_before": 0,
            "round2_sentences_after": 0,
            "round2_sentences_removed": 0,
            "round2_chars_before": 0,
            "round2_chars_after": 0,
            "round2_chars_removed": 0,
            "round2_fallback_restore_count": 0,
        }

        # 初始化 PPL 计算器
        self._ppl_scorer = None
        if self.ppl_model_name:
            self._ppl_scorer = LocalHuggingFacePPLScorer(self.ppl_model_name)
        else:
            self._ppl_scorer = CharNgramPPLScorer()

    def _get_recommendation(self, content: str, subfile_idx: int, line_idx: Optional[int] = None) -> tuple[str, int, int]:
        """获取分片参数推荐（line_mode 下按采样间隔调用 LLM，其余走内置表）"""
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
            l_min = rec["l_min"]
            l_max = rec["l_max"]

        return genre, l_min, l_max

    def _compute_ppls(self, sentences: list[str], context_text: str = "") -> list[float]:
        """计算所有句子的 PPL（统一在全文语料上训练 CharNgram，避免 context 污染）"""
        ppls = []
        full_corpus = context_text or "".join(sentences)
        if isinstance(self._ppl_scorer, CharNgramPPLScorer):
            self._ppl_scorer.train(full_corpus)

        for i, sent in enumerate(sentences):
            ctx_start = max(0, i - self.window_w)
            context = "".join(sentences[ctx_start:i])
            try:
                ppl = self._ppl_scorer.score(context, sent)
                ppl = min(ppl, 1e6)  # 防止极端值
            except Exception:
                ppl = 100.0
            ppls.append(ppl)

        return ppls

    def _round2_denoise_and_split(
        self, sentences: list[str], l_max: int
    ) -> list[str]:
        """
        Round 2: PPL 去噪 + 切分
        """
        chars_before = sum(len(s) for s in sentences)
        self._denoise_stats["round2_sentences_before"] += len(sentences)
        self._denoise_stats["round2_chars_before"] += chars_before

        if len(sentences) < 2:
            self._denoise_stats["round2_sentences_after"] += len(sentences)
            self._denoise_stats["round2_chars_after"] += chars_before
            return ["".join(sentences)] if sentences else []

        context_text = "".join(sentences)
        ppls = self._compute_ppls(sentences, context_text)

        # 去噪：移除 PPL 异常高的句子（改用 MAD，对重尾分布更鲁棒）
        if self.denoise:
            t1 = calc_mad(ppls, k=self.mad_k)
            denoised = [s for i, s in enumerate(sentences) if ppls[i] is None or ppls[i] <= t1]
            ppls_denoised = [p for p in ppls if p is not None and p <= t1]
        else:
            denoised = sentences
            ppls_denoised = ppls

        chars_after = sum(len(s) for s in denoised)
        self._denoise_stats["round2_sentences_after"] += len(denoised)
        self._denoise_stats["round2_sentences_removed"] += len(sentences) - len(denoised)
        self._denoise_stats["round2_chars_after"] += chars_after
        self._denoise_stats["round2_chars_removed"] += chars_before - chars_after

        if not denoised:
            self._denoise_stats["round2_fallback_restore_count"] += 1
            return ["".join(sentences)]

        # 重新计算去噪后的 PPL，用于切分
        ppls2 = self._compute_ppls(denoised, "".join(denoised))

        # 切分：PPL 突变处切分
        mean2, std2 = calc_mean_std(ppls2)
        t2 = mean2 + std2

        chunks = []
        curr = []

        for i, s in enumerate(denoised):
            if i > 0 and ppls2[i] > t2:
                chunks.append("".join(curr))
                curr = [s]
            else:
                curr.append(s)

        if curr:
            chunks.append("".join(curr))

        return chunks

    def _round3_optimize(self, chunks: list[ChunkBlock]) -> list[ChunkBlock]:
        """
        Round 3: 策略优化融合
        - 处理超大 chunk（贪心 IG 融合）
        - 融合过小 chunk
        """
        final = []

        for b in chunks:
            if b.length <= b.l_max:
                final.append(b)
                continue

            units = [ChunkBlock(s, len(s), b.l_min, b.l_max) for s in split_into_sentences(b.text)]
            units = [u for u in units if u.length > 0]
            if not units:
                final.append(b)
                continue

            ig_heap = []
            for i in range(len(units) - 1):
                if units[i].length + units[i + 1].length <= b.l_max:
                    ig = self.ig_calc.ig(units[i], units[i + 1])
                    heapq.heappush(ig_heap, (-ig, i, i + 1))

            done = [False] * len(units)
            while ig_heap:
                neg_ig, i, j = heapq.heappop(ig_heap)
                if done[i] or done[j]:
                    continue
                merged = ChunkBlock(
                    units[i].text + units[j].text,
                    units[i].length + units[j].length,
                    b.l_min, b.l_max,
                )
                # i 作为合并后的活动单元保留，j 才是被消费的单元。
                # 原实现把 i、j 都标记为 done，导致合并后的文本在下面
                # 的遍历中被跳过，造成第三阶段静默丢失文本。
                done[j] = True
                units[i] = merged
                ni = i + 1
                while ni < len(units) and done[ni]:
                    ni += 1
                if ni < len(units) and units[i].length + units[ni].length <= b.l_max:
                    ig2 = self.ig_calc.ig(units[i], units[ni])
                    heapq.heappush(ig_heap, (-ig2, i, ni))

            overflow_text = ""
            for idx_u, u in enumerate(units):
                if done[idx_u]:
                    continue
                if u.length <= b.l_max:
                    if overflow_text:
                        combined = u.text + overflow_text
                        if len(combined) <= b.l_max:
                            final.append(ChunkBlock(combined, len(combined), b.l_min, b.l_max))
                            overflow_text = ""
                        else:
                            final.append(u)
                            overflow_text = ""
                    else:
                        final.append(u)
                else:
                    final.append(ChunkBlock(u.text[:b.l_max], b.l_max, b.l_min, b.l_max))
                    overflow_text += u.text[b.l_max:]

            if overflow_text:
                ob = ChunkBlock(overflow_text, len(overflow_text), b.l_min, b.l_max)
                if final and final[-1].length + ob.length <= b.l_max * 1.1:
                    final[-1] = ChunkBlock(
                        final[-1].text + ob.text,
                        final[-1].length + ob.length,
                        b.l_min, b.l_max,
                    )
                else:
                    final.append(ob)

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

    def chunk_subfile(self, content: str, doc_id: str, subfile_idx: int,
                      line_idx: Optional[int] = None) -> list[ChunkBlock]:
        """
        对单个子文件执行 Round 2 和 Round 3
        """
        genre, l_min, l_max = self._get_recommendation(content, subfile_idx, line_idx)

        # Round 2: PPL 去噪与切分
        sentences = split_into_sentences(content)
        raw_chunks = self._round2_denoise_and_split(sentences, l_max)

        chunk_blocks = [
            ChunkBlock(c, len(c), l_min, l_max)
            for c in raw_chunks
        ]

        # Round 3: 策略优化融合
        optimized = chunk_blocks if self.disable_optimization else self._round3_optimize(chunk_blocks)

        return optimized

    def chunk_document(self, content: str, filename: str,
                       line_idx: Optional[int] = None) -> list[dict]:
        """
        对整个文档执行完整的分片流程
        """
        doc_id = pathlib.Path(filename).stem

        # Round 1: 结构拆分
        subfiles = [content] if self.disable_structure_split else split_by_structure(content)

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

    def run(self, input_path: pathlib.Path, output_path: pathlib.Path):
        """
        运行分片流程
        """
        output_path.mkdir(parents=True, exist_ok=True)

        all_results = []
        summary = {
            "version": "integrated_v3",
            "denoise_enabled": bool(self.denoise),
            "structure_split_enabled": not bool(self.disable_structure_split),
            "optimization_enabled": not bool(self.disable_optimization),
            "ppl_scorer": "local_huggingface" if self.ppl_model_name else "char_ngram",
            "window_w": self.window_w,
            "beta_small": self.beta_small,
            "beta": self.beta,
            "mad_k": self.mad_k,
            "line_mode": self.line_mode,
            "llm_sample_interval": self.llm_sample_interval,
            "denoise_stats": self._denoise_stats,
            "total_files": 0,
            "total_chunks": 0,
            "input_chars": 0,
            "files": [],
        }

        if input_path.is_file():
            files = [input_path]
        else:
            files = list(input_path.glob("*.txt")) + list(input_path.glob("*.md"))

        for fpath in files:
            content = fpath.read_text(encoding="utf-8")

            if self.line_mode:
                # 每行作为一个独立 doc
                lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
                summary["input_chars"] += sum(len(line) for line in lines)
                for line_idx, line_content in enumerate(lines):
                    doc_id = f"{fpath.stem}_line_{line_idx:05d}"
                    chunks = self.chunk_document(line_content, doc_id, line_idx=line_idx)
                    for c in chunks:
                        c["doc_id"] = doc_id
                        c["source_file"] = fpath.name
                    all_results.extend(chunks)
                    summary["total_chunks"] += len(chunks)
                    if line_idx == 0 or len(chunks) > 0:
                        pass
                summary["total_files"] += 1
                summary["files"].append({
                    "filename": fpath.name,
                    "lines": len(lines),
                    "total_chunks": summary["total_chunks"],
                })
            else:
                summary["input_chars"] += len(content)
                chunks = self.chunk_document(content, fpath.name)
                all_results.extend(chunks)
                summary["total_files"] += 1
                summary["total_chunks"] += len(chunks)
                summary["files"].append({
                    "filename": fpath.name,
                    "chunks": len(chunks),
                })

        # 输出
        chunks_json = [{"chunk_text": c["chunk_text"], "chunk_len": c["chunk_len"]} for c in all_results]
        summary["output_chars"] = sum(c["chunk_len"] for c in chunks_json)
        (output_path / "all_chunks_chunks.json").write_text(
            json.dumps(chunks_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_path / "all_chunks_chunks.txt").write_text(
            "\n\n---\n\n".join(c["chunk_text"] for c in chunks_json), encoding="utf-8"
        )
        (output_path / "all_chunks_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"分片完成：{summary['total_files']} 个文件，{summary['total_chunks']} 个 chunks")
        print(f"结果保存在: {output_path}")


# ==============================================================================
# 命令行入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="三阶段智能文档分片器")
    parser.add_argument("--input", type=str, required=True, help="输入文件或目录")
    parser.add_argument("--line_mode", action="store_true",
        help="将输入文件的每一行视为一个独立文档进行分片")
    parser.add_argument("--llm_sample_interval", type=int, default=300,
        help="line_mode 下每隔多少行调用一次 LLM（默认 300，设为 0 则全部跳过 LLM）")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--llm_api_key", type=str, default=os.environ.get("CHUNK_LLM_API_KEY", ""), help="LLM API Key（默认从 CHUNK_LLM_API_KEY 读取）")
    parser.add_argument("--llm_base_url", type=str, default=os.environ.get("CHUNK_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"), help="LLM API Base URL")
    parser.add_argument("--llm_model", type=str, default=os.environ.get("CHUNK_LLM_MODEL", "qwen3-8b"), help="LLM 模型名称")
    parser.add_argument("--ppl_model_name", type=str, default="", help="本地 PPL 模型路径（可选）")
    parser.add_argument("--window_w", type=int, default=3, help="PPL 上下文窗口大小")
    parser.add_argument("--beta_small", type=float, default=0.8, help="过小 chunk 判断系数")
    parser.add_argument("--beta", type=float, default=1.1, help="合并长度上限系数")
    parser.add_argument("--mad_k", type=float, default=3.5, help="MAD 去噪阈值系数（默认 3.5）")
    parser.add_argument("--denoise", type=lambda x: x.lower() != "false", default=True, help="是否启用去噪")
    parser.add_argument("--disable_structure_split", action="store_true",
        help="消融实验：跳过第一阶段结构拆分")
    parser.add_argument("--disable_optimization", action="store_true",
        help="消融实验：跳过第三阶段策略优化融合")

    args = parser.parse_args()
    chunker = IntegratedChunker(args)
    chunker.run(pathlib.Path(args.input), pathlib.Path(args.output))


if __name__ == "__main__":
    main()
