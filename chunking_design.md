# 分片（Chunking）系统设计文档

> 基于 `integrated_chunker.py` 的三阶段分片算法详解
> 适用于在新项目中复现和实验验证

---

## 目录

1. [系统总览](#1-系统总览)
2. [Round 1：按文档结构拆分 + LLM 推荐分片参数](#2-round-1按文档结构拆分--llm-推荐分片参数)
3. [Round 2：PPL（困惑度）去噪与切分](#3-round-2ppl困惑度去噪与切分)
4. [Round 3：策略优化融合](#4-round-3策略优化融合)
5. [关键参数汇总](#5-关键参数汇总)
6. [最小依赖与复现指南](#6-最小依赖与复现指南)
7. [调用方式](#7-调用方式)

---

## 1. 系统总览

整个分片流程分为 **三大轮（Round）**，是 `integrated_chunker.py` 的核心逻辑：

```
原始文档
  │
  ▼
Round 1: 结构拆分 + LLM 推荐分片参数
  │
  ▼
Round 2: PPL（困惑度）去噪与切分
  │
  ▼
Round 3: 策略优化融合
  │
  ▼
最终 chunks
```

**数据流说明**：

- Round 1 将一篇文档按结构标题/分隔线切分成若干"子文件"（subfiles），每个子文件独立获取自己的 `l_min` / `l_max` 分片长度参数。
- Round 2 对每个子文件内部，使用困惑度（PPL）进行异常句子的去噪，并基于 PPL 的突变检测切分点。
- Round 3 处理 Round 2 产生的超大 chunk（用信息增益贪心融合）和过小 chunk（合并到前一个 chunk）。

---

## 2. Round 1：按文档结构拆分 + LLM 推荐分片参数

### 2.1 结构边界检测

对原始文本按行分割，用正则表达式检测以下结构边界：

```python
patterns = [
    re.compile(r"^\s*={3,}.+={0,}\s*$", re.I),  # ===== 标题 =====
    re.compile(r"^\s*#{1,6}\s+.+$"),            # Markdown 标题
    re.compile(r"^\s*-{20,}\s*$"),               # --------... (30个分隔线)
    re.compile(r"^\s*-{3}\s*$"),                 # --- 分隔线
]
```

将文档按这些结构标题/分隔线切分成若干"子文件"（subfiles），每个子文件包含一段连续的正文。

### 2.2 LLM 动态推荐分片长度

对每个子文件，通过 LLM 判断其体裁（genre）并推荐 `l_min` 和 `l_max`（分片的最小/最大字符数）：

```python
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
```

**内置的默认推荐表**（当无 LLM API Key 时使用）：

| 体裁 | l_min | l_max | 说明 |
|------|-------|-------|------|
| `doc`（技术文档） | 400 | 1000 | 默认值 |
| `news`（新闻资讯） | 300 | 600 | 较短 |
| `paper`（学术论文） | 500 | 1200 | 可较长 |
| `novel`（小说故事） | 400 | 800 | 适中 |
| `chat`（对话记录） | 200 | 500 | 更短 |

**体裁自动检测逻辑**（`_detect_genre_hint`）：

```python
# 检测论文特征
if any(kw in content_lower for kw in ["摘要", "abstract", "参考文献", "引言", "结论"]):
    return "paper"
# 检测新闻特征
if any(kw in content_lower for kw in ["日报", "快讯", "据报道", "新华社"]):
    return "news"
# 检测小说特征
if any(kw in content_lower for kw in ['"', '"', "他说着", "她说道", "心想"]):
    return "novel"
# 检测对话特征（通过冒号密度）
if content.count("：") / max(len(content), 1) > 0.02:
    return "chat"
```

### 2.3 输出

Round 1 结束后，每个子文件获得以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `doc_id` | string | 文档 ID，如 `filename_000` |
| `file_name` | string | 子文件名，如 `filename_sub_000.txt` |
| `genre` | string | 体裁类型 |
| `l_min` | int | 最小分片长度（字符数） |
| `l_max` | int | 最大分片长度（字符数） |
| `content` | string | 子文件的原始文本内容 |

---

## 3. Round 2：PPL（困惑度）去噪与切分

### 3.1 句子拆分

用正则将子文件内容拆成句子（中文优先，按 `。！？；!?;` 断句）：

```python
SENT_PATTERN = re.compile(r"[^。！？；!?;\n]+[。！？；!?;]?")
```

### 3.2 PPL 计算（Perplexity，困惑度）

**核心思想**：PPL 衡量一个句子在给定上下文中出现的"意外程度"。PPL 突然升高意味着话题/语义的跳转。

使用两种 PPL 计算器（优先使用本地 HuggingFace 模型）：

#### 方案 A：本地因果语言模型（`LocalHuggingFacePPLScorer`）

```python
# 使用因果语言模型（Casual LM）计算 PPL
# PPL = exp(mean(NLL))
# PPL 越高，表示句子越"不符合"上下文预期，越可能是异常内容

# 模型使用 torch + transformers 加载
self.model = AutoModelForCausalLM.from_pretrained(model_path)
```

#### 方案 B：字符级 N-gram 回退（`CharNgramPPLScorer`）

无外部依赖时的备选方案，统计字符 bigram 的条件概率：

```python
# P(ch|prev) = (count(prev,ch) + 1) / (count(prev) + |vocab|)
# PPL = exp(-1/N * sum(log(P(ch_i|prev_i))))
```

**窗口机制**：PPL 不是只看当前句子，而是看它前面 `window_w=3` 个句子作为上下文：

```python
# 对每个句子 i，计算 PPL 时使用前 window_w 个句子作为 context
ppls[i] = ppl_scorer.score("".join(sents[max(0, i-window_w):i]), sents[i])
```

### 3.3 去噪（移除异常句子）

计算所有句子的 PPL 后，用 **均值 + 3倍标准差** 作为阈值，高于阈值的句子被视为"异常"（可能是 OCR 错误、乱码、无关广告等），直接丢弃：

```python
mean, std = calc_mean_std(valid_ppls)
t1 = mean + 3 * std  # 去噪阈值（宽松）

denoised = [s for i, s in enumerate(sents) if ppls[i] is None or ppls[i] <= t1]
```

> **注意**：均值 + 3σ 是统计学中的经典异常值检测阈值（对应正态分布下 99.7% 置信区间）。

### 3.4 PPL 切分点检测

在去噪后的句子上重新计算 PPL，用 **均值 + 1倍标准差** 作为切分阈值。当某个句子 PPL 突然升高（`> t2`），说明这里发生了话题/语义转换，就在该点切分：

```python
t2 = mean + std  # 切分阈值（比去噪更敏感）

chunks = []
curr = []
for i, s in enumerate(denoised):
    if i > 0 and ppls2[i] > t2:
        chunks.append("".join(curr))  # 在 PPL 突变点切分
        curr = [s]
    else:
        curr.append(s)
if curr:
    chunks.append("".join(curr))
```

**两个阈值的区别**：

| 阈值 | 公式 | 用途 | 宽松度 |
|------|------|------|--------|
| 去噪阈值 | 均值 + 3σ | 去除极端异常句子 | 宽松（只去离群点） |
| 切分阈值 | 均值 + 1σ | 识别语义跳转点 | 敏感（捕捉变化） |

---

## 4. Round 3：策略优化融合

### 4.1 处理超大 chunk（贪心 IG 融合）

如果 Round 2 产生的某个 chunk 长度超过了 `l_max`，执行以下步骤：

1. 将这个大 chunk 再细拆成更小的句子单元
2. 用**信息增益（Information Gain, IG）**贪心地合并相邻单元，直到总长度不超过 `l_max`

**信息增益计算**：

```python
class IGCalculator:
    def embedding(self, text: str):
        # 将文本表示为字符频率的归一化向量
        vec = {ch: count for ch, count in text_counter.items()}
        norm = sqrt(sum(v*v for v in vec.values()))
        return {k: v/norm for k, v in vec.items()}

    def ig(self, left: ChunkBlock, right: ChunkBlock) -> float:
        # 两个文本块的信息增益 = 向量相似度
        # sim = sum(e1[k] * e2[k])（归一化向量点积）
        # ig = 1 / (1 + (1 - sim))  ->  sim 越高，ig 越高
        e1, e2 = self.embedding(left.text), self.embedding(right.text)
        keys = set(e1.keys()) & set(e2.keys())
        sim = max(-1.0, min(1.0, sum(e1[k] * e2[k] for k in keys)))
        return 1.0 / (1.0 + (1.0 - sim))
```

**贪心融合逻辑**：

```python
while len(units) > 1:
    best_ig = -1
    best_i = -1
    for i in range(len(units) - 1):
        # 只考虑合并后不超过 l_max 的相邻对
        if units[i].length + units[i+1].length <= b.l_max:
            ig = self.ig_calc.ig(units[i], units[i+1])
            if ig > best_ig:
                best_ig = ig
                best_i = i
    if best_i == -1:
        break  # 没有可合并的对了
    # 合并 IG 最高的一对
    units[best_i].text += units[best_i+1].text
    units.pop(best_i + 1)
```

> **设计意图**：每次合并语义最相似（信息增益最高）的相邻句子，确保大 chunk 被切分后仍然保持语义连贯性。

### 4.2 融合过小 chunk

如果某个 chunk 长度小于 `l_min * beta_small`（`beta_small=0.8`），尝试将它合并到前一个 chunk 中：

```python
for b in processed:
    if not final or b.length >= b.l_min * self.beta_small:
        # 长度足够，直接保留
        final.append(b)
    else:
        # 过小，尝试合并到前一个 chunk
        if final[-1].length + b.length <= b.l_max * self.beta:
            final[-1].text += b.text  # 合并
        else:
            final.append(b)  # 合并会超长，单独保留
```

### 4.3 最终输出格式

```python
{
    "chunk_text": "分片文本内容",
    "chunk_len": 文本长度（字符数）
}
```

---

## 5. 关键参数汇总

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window_w` | 3 | PPL 计算的上下文窗口大小（向前看几个句子作为 context） |
| `beta_small` | 0.8 | 过小 chunk 判断系数：`length < l_min * beta_small` 时尝试合并 |
| `beta` | 1.1 | 合并长度上限系数：合并后不超过 `l_max * beta` |
| `denoise` | True | 是否启用 PPL 去噪（默认开启） |
| `ppl_model_name` | 空 | 本地 HuggingFace 模型的路径或名称（不填则用 CharNgramPPLScorer） |
| `llm_api_key` | 空 | LLM API Key（不填则使用默认推荐表） |
| `llm_base_url` | `https://api.openai.com/v1` | LLM API Base URL |
| `llm_model` | `gpt-3.5-turbo` | LLM 模型名称 |

---

## 6. 最小依赖与复现指南

### 6.1 纯 Python 依赖（无需外部模型）

`integrated_chunker.py` 的分片核心只依赖 Python 标准库：

```python
# 必需的 Python 标准库
import json, re, math, statistics, argparse, pathlib
# 仅用于调用 LLM 时：
# import urllib.request  (标准库)
```

如果你的 LLM API Key 也为空，系统会自动：
- 使用 `CharNgramPPLScorer`（字符 bigram 替代深度学习 PPL）
- 使用内置的默认推荐表（不调用 LLM）

### 6.2 推荐的生产配置

```python
# 推荐的完整依赖（pip install）
torch>=2.0.0
transformers>=4.30.0

# 本地 PPL 模型建议使用因果语言模型
# 建议：Qwen2-0.5B-Instruct / Qwen2-1.5B-Instruct
# 体积适中，效果远好于字符 N-gram
```

### 6.3 在新项目中复现的最小代码结构

```
your_new_project/
├── integrated_chunker.py    # 直接复制整个文件
├── requirements.txt
└── data/
    └── input/
        └── your_doc.txt
```

```bash
# 完整调用示例（无 LLM）
python integrated_chunker.py \
    --input ./data/input \
    --output ./output_chunks

# 完整调用示例（有 LLM）
python integrated_chunker.py \
    --input ./data/input \
    --output ./output_chunks \
    --llm_api_key your_api_key \
    --llm_base_url https://api.openai.com/v1 \
    --llm_model gpt-4o

# 完整调用示例（有本地 PPL 模型）
python integrated_chunker.py \
    --input ./data/input \
    --output ./output_chunks \
    --ppl_model_name ./models/Qwen2-0.5B-Instruct
```

### 6.4 关键输出文件

| 文件 | 内容 |
|------|------|
| `all_chunks_chunks.json` | 所有分片的完整结果（含每个 subfile 的 chunks 列表） |
| `all_chunks_chunks.txt` | 所有分片的纯文本版本 |
| `all_chunks_summary.json` | 汇总信息（文件数、chunk 总数、各子文件的分片统计） |

---

## 7. 调用方式

### 7.1 通过 pipeline.py 调用（完整流水线）

```bash
# 完整流水线
python pipeline.py --all --input ./data_input/test_data

# 仅执行 Step 2（分片）
python pipeline.py --step 2 --input ./data_input/test_data

# 指定步骤范围
python pipeline.py --from_step 2 --to_step 6 --input ./data_input/test_data
```

### 7.2 直接调用 integrated_chunker.py（仅分片）

```bash
python integrated_chunker.py \
    --input ./data_input/test_data \
    --output ./output_chunks \
    --llm_api_key your_key \
    --llm_base_url https://api.openai.com/v1 \
    --llm_model gpt-3.5-turbo \
    --window_w 3 \
    --beta_small 0.8 \
    --beta 1.1
```

### 7.3 Python 代码中调用

```python
from pathlib import Path
import argparse
from integrated_chunker import IntegratedChunker

args = argparse.Namespace(
    input="./data",
    output="./output",
    llm_api_key="",              # 不填则用默认推荐表
    llm_base_url="https://...",
    llm_model="gpt-4o",
    ppl_model_name="",           # 不填则用 CharNgramPPLScorer
    window_w=3,
    beta_small=0.8,
    beta=1.1,
    recommendation_config="",
    denoise=True,
)

chunker = IntegratedChunker(args)
chunker.run(Path(args.input), Path(args.output))
```

---

*文档生成时间：2026-06-03*
