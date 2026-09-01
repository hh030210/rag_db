# Integrated Chunking System v2

基于三阶段分片算法的文档智能分片器，v2 在原版基础上进行了系统性改进。

---

## 整体流程

```
输入文档 (文件或目录)
       │
       ▼
┌──────────────────────────────────────────────┐
│         Round 1: 结构拆分 + 体裁检测           │
│  • 按标题 / Markdown / 分隔线拆分              │
│  • 关键词或 LLM 识别体裁 → 推荐 l_min / l_max  │
└─────────────────┬────────────────────────────┘
                  ▼
┌──────────────────────────────────────────────┐
│       Round 2: PPL 去噪 + 切分                 │
│  • CharNgram 模型计算每句困惑度                  │
│  • MAD 阈值去噪（移除异常高 PPL 句子）           │
│  • PPL 突变处切分 chunk                        │
└─────────────────┬────────────────────────────┘
                  ▼
┌──────────────────────────────────────────────┐
│            Round 3: 策略优化融合                │
│  • 堆优化贪心 IG 融合超大 chunk                 │
│  • 溢出文本续接下游（不再丢失）                 │
│  • 合并过小 chunk                              │
└─────────────────┬────────────────────────────┘
                  ▼
          Chunk JSON / TXT 输出
```

---

## 三阶段算法详解

### Round 1：结构拆分 + 体裁检测

**结构拆分** 使用 `STRUCT_PATTERNS` 正则匹配文档语义段落边界：

| 模式 | 用途 |
|------|------|
| `^=+\s*(.+?)\s*=+$` | 等号包围的标题 `==== 标题 ====` |
| `^#{1,6}\s+` | Markdown 标题 `# ~ ######` |
| `^\s*-{3,}\s*$` | 长分隔线 `----` |
| `^\s*-{3}\s*$` | 短分隔线 `---` |

每个 subfile 独立处理，避免跨段落切分。

**体裁检测** 支持两条路径：

| 路径 | 触发条件 | 说明 |
|------|----------|------|
| 关键词快速判断 | 无 LLM API Key | 检测"摘要/abstract/日报/小说引号"等关键词 |
| LLM 智能推荐 | 有 API Key 且满足采样间隔 | 调用 LLM 流式解析前500字符，输出 JSON {genre, l_min, l_max} |

内置体裁推荐表（无 LLM 时兜底）：

| 体裁 | l_min | l_max | 适用场景 |
|------|-------|-------|----------|
| doc | 400 | 1000 | 技术文档 |
| news | 300 | 600 | 新闻资讯 |
| paper | 500 | 1200 | 学术论文 |
| novel | 400 | 800 | 小说故事 |
| chat | 200 | 500 | 对话记录 |

### Round 2：PPL 去噪与切分

**困惑度计算** 支持两种实现：

**CharNgramPPLScorer**（默认，无外部依赖）：

- 字符级 bigram 模型：`P(ch|prev) = (count(prev,ch)+1)/(count(prev)+|vocab|)`
- `PPL = exp(-1/N * Σ log P(ch_i|prev_i))`
- 上下文窗口：前 `window_w` 个句子

**LocalHuggingFacePPLScorer**（GPU 加速）：

- 调用本地因果语言模型（如 Qwen）
- `PPL = exp(mean(NLL))`
- 通过 `--ppl_model_name` 参数指定模型路径

**去噪策略**（v2 改进）：使用 MAD（中位数绝对偏差）替代 3-sigma

- PPL 服从重尾分布，标准差会被极端值严重拉偏
- `MAD = median(|x_i - median(x)|)`，对异常值鲁棒
- 阈值：`threshold = median + 3.5 * MAD * 1.4826`（归一化到正态等价）

**切分策略**：计算去噪后每句 PPL，对超过 `mean + std` 的突变处切分。

### Round 3：策略优化融合（v2 重写）

**超大 chunk 处理**：

```
超大 chunk → split_into_sentences → units[]
    ↓
堆优化贪心 IG 融合：
    计算相邻 unit 对的 IG
    用最小堆按 IG 从大到小弹出可合并对
    合并后更新相邻关系，O(n log n)
    ↓
超长 unit 截断头部保留 → 溢出文本续接下一个 chunk
    （v1 直接丢弃 → v2 不再丢失内容）
```

**IG（信息增益）计算**：基于字符频率向量的余弦相似度

```python
e1 = normalize(char_freq_vector(chunk_i))
e2 = normalize(char_freq_vector(chunk_i+1))
sim = dot(e1, e2)
ig = 1.0 / (1.0 + (1 - sim))   # IG 越高 = 越相似
```

**过小 chunk 合并**：如果 chunk 长度 < `l_min * beta_small`，尝试与前一个 chunk 合并（总长不超过 `l_max * beta`）。

---

## v2 改进点

### 改进 1：Round 3 溢出文本续接（内容不再丢失）

**v1 问题**：超长 unit 无法融合时被直接截断，`u.text[:b.l_max]` 之后的字符永久丢失。

**v2 方案**：

```python
overflow_text = ""
for u in units:
    if u.length <= b.l_max:
        if overflow_text:
            combined = u.text + overflow_text   # 溢出优先续接下游
            if len(combined) <= b.l_max:
                final.append(...)
                overflow_text = ""
    else:
        final.append(ChunkBlock(u.text[:b.l_max], ...))  # 头部保留
        overflow_text += u.text[b.l_max:]                  # 尾部续接

if overflow_text:
    # 剩余溢出作为独立块或合并到上一块
```

### 改进 2：PPL 去噪改用 MAD（中位数绝对偏差）

**v1 问题**：`t1 = mean + 3 * std`，PPL 重尾分布下 σ 受极端值影响严重。

**v2 方案**：`t1 = calc_mad(ppls, k=3.5)`，对重尾分布鲁棒性显著提升。

### 改进 3：PPL context 污染修复（统一全文训练）

**v1 问题**：`_compute_ppls` 接收 `context_text` 但实际并不使用全文，导致 CharNgram 训练在 `context_text` 上，而后续 `score(context, sent)` 又使用前 `window_w` 句作为 context，两者不一致。

**v2 方案**：在 `_compute_ppls` 开头统一调用 `train(full_corpus)`，确保模型在完整语料上训练。

### 改进 4：CharNgram 训练缓存（避免重复训练）

**v1 问题**：每句调用 `_score_sentence` 时检查 `if not self._trained`，而 context 每句都变，导致每句都重建计数表。

**v2 方案**：

- 新增 `_trained_corpus` 字段
- `train()` 检查 corpus 是否变化，相同则跳过
- 配合改进 3 的统一训练，实际只需训练一次

### 改进 5：IG 贪心融合改用堆优化

**v1 问题**：每次合并都要 O(n) 扫描找最佳对，总复杂度 O(n²)。

**v2 方案**：最小堆 `(-ig, i, j)` 维护所有可合并对，每次弹出最高 IG 的对，合并后更新相邻关系，复杂度 O(n log n)。

### 改进 6：LLM 调用指数退避重试

**v1 问题**：网络超时 / 429 / 500 直接返回 None，退化为不准确的默认参数。

**v2 方案**：

```python
for attempt in range(3):
    try:
        # LLM call
    except HTTPError as e:
        if e.code in (429, 500, 502, 503):
            sleep((2 ** attempt) * 5)  # 5s, 10s, 20s
            continue
    except Exception:
        sleep((2 ** attempt) * 5)
        continue
```

### 改进 7：Sentence 正则增强 + Genre 引号修复

**Sentence 正则**：将中文弯引号 `""`（U+201C/U+201D）加入字符集，正确处理：

- 中文弯引号包裹的对话：`"他说："你好。"她回答："好的。""`
- 冒号后的多个句子：`回答：第一点。第二点。`

**Genre 引号**：列表中第二个 `"` 改为 `"\u201c"`、`"\u201d"`，能正确匹配中文引号。

---

## 命令行用法

### 基础用法

```bash
python integrated_chunker.py --input ./data/news --output ./output/news_chunks
```

### 完整参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | 必填 | 输入文件或目录路径 |
| `--output` | 必填 | 输出目录路径 |
| `--line_mode` | False | 每行视为独立文档 |
| `--llm_api_key` | `sk-...` | LLM API Key（为空则使用内置推荐表） |
| `--llm_base_url` | 阿里云 DashScope | API Base URL |
| `--llm_model` | `qwen3-8b` | LLM 模型名称 |
| `--llm_sample_interval` | 300 | line_mode 下每隔多少行调用一次 LLM（0=全跳过） |
| `--ppl_model_name` | 空 | 本地 HuggingFace PPL 模型路径（空则用 CharNgram） |
| `--window_w` | 3 | PPL 上下文窗口大小（句子数） |
| `--beta_small` | 0.8 | 过小 chunk 判断系数 |
| `--beta` | 1.1 | 合并长度上限系数 |
| `--denoise` | True | 是否启用 PPL 去噪 |

### 示例

```bash
# 新闻文档，line_mode 每行独立处理
python integrated_chunker.py \
    --input ./data/news.txt --output ./output/news \
    --line_mode --llm_sample_interval 100

# 技术文档，使用本地 GPU PPL 模型
python integrated_chunker.py \
    --input ./data/docs/ --output ./output/docs \
    --ppl_model_name ./models/qwen2-1.5b \
    --window_w 5 --beta 1.2
```

---

## 输出格式

运行后生成三个文件：

### `all_chunks_chunks.json`

```json
[
  {"doc_id": "article", "subfile_id": "article_sub_000",
   "chunk_text": "...", "chunk_len": 1024}
]
```

### `all_chunks_chunks.txt`

每段 chunk 之间用 `---` 分隔。

### `all_chunks_summary.json`

```json
{
  "total_files": 5,
  "total_chunks": 47,
  "files": [{"filename": "a.txt", "chunks": 12}, ...]
}
```

---

## 参数调优指南

| 体裁 | 建议 l_min | 建议 l_max |
|------|-----------|------------|
| 新闻 | 200-300 | 400-600 |
| 论文 | 400-600 | 800-1200 |
| 小说 | 300-500 | 600-900 |
| 对话 | 150-300 | 300-500 |

| PPL 参数 | 调优建议 |
|----------|----------|
| `window_w` | 增大→更长上下文，PPL 更稳定 |
| `denoise` | True（默认），仅在确认无异常句时关闭 |
| `beta_small` | 增大→更激进合并过小 chunk |
| `beta` | 增大→允许更大合并块 |

| 场景 | 推荐 PPL 计算器 |
|------|----------------|
| 无 GPU，追求速度 | CharNgramPPLScorer（默认） |
| 有 GPU，CPU 瓶颈 | LocalHuggingFacePPLScorer |
| 文档 > 10万字 | 建议用 HuggingFace，避免 N-gram 稀疏 |

---

## 架构设计

```
integrated_chunker.py
├── TextCounter                  # 字符频率计数器（IG计算用）
├── GENRE_RECOMMENDATIONS        # 体裁参数推荐表
├── SYSTEM_PROMPT                # LLM system prompt
├── STRUCT_PATTERNS              # 文档结构边界正则
├── SENT_PATTERN                 # 句子切分正则（v2 增强引号）
├── detect_genre_hint()          # 关键词体裁检测（v2 修复引号）
├── call_llm_recommend()         # LLM API 调用（v2 指数退避重试）
├── CharNgramPPLScorer           # 字符级 N-gram PPL（v2 训练缓存）
│   ├── __init__
│   ├── train(corpus)            # 相同 corpus 跳过
│   ├── _score_sentence()
│   └── score(context, sentence)
├── LocalHuggingFacePPLScorer    # HuggingFace PPL（GPU 加速）
├── IGCalculator                 # 信息增益计算
├── calc_mean_std()              # 均值/标准差
├── calc_mad()                   # MAD 阈值（v2新增）
├── split_by_structure()         # 结构边界拆分
├── split_into_sentences()       # 句子拆分（v2 增强引号）
└── IntegratedChunker
    ├── __init__
    ├── _get_recommendation()    # 获取分片参数
    ├── _compute_ppls()          # 计算困惑度（v2 全文训练一次）
    ├── _round2_denoise_and_split()  # Round 2（v2 MAD 去噪）
    ├── _round3_optimize()       # Round 3（v2 堆优化+溢出续接）
    ├── chunk_subfile()          # 子文件分片
    ├── chunk_document()         # 整文档分片
    └── run()                    # 批量处理入口
```

### 数据流

```
chunk_document(content, filename)
    │
    ├─ split_by_structure() → subfiles[]
    │     │
    │     └─ for each subfile:
    │           ├─ _get_recommendation() → (genre, l_min, l_max)
    │           ├─ split_into_sentences() → sentences[]
    │           ├─ _round2_denoise_and_split()
    │           │     ├─ _compute_ppls()  # 在 full_corpus 上训练一次
    │           │     ├─ calc_mad() → t1  # MAD 阈值
    │           │     ├─ 剔除 ppl > t1 的句子
    │           │     ├─ 重新计算 ppls
    │           │     └─ PPL 突变处切分 → chunks[]
    │           └─ _round3_optimize()
    │                 ├─ 堆优化贪心 IG 融合超大 chunk
    │                 ├─ 溢出文本续接下游       # v2 内容不再丢失
    │                 └─ 合并过小 chunk
    │
    └─ 输出 JSON 列表
```

---

## 性能对比（v1 vs v2）

| 维度 | v1 | v2 | 改进 |
|------|----|----|------|
| Round 3 复杂度 | O(n²) | O(n log n) | 堆优化 |
| PPL 训练次数 | 每句一次 | 全文一次 | 缓存 + 统一训练 |
| 异常 PPL 噪声过滤 | 3-sigma（脆弱） | MAD（鲁棒） | 重尾分布 |
| 超大 chunk 内容保留 | 截断丢失 | 溢出续接下游 | 完整保留 |
| LLM 调用成功率 | 一次失败即放弃 | 3 次重试退避 | 网络容错 |
| 中文引号体裁识别 | 不生效 | 正确匹配 | 引号字符修复 |