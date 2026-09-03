# 智能分片方法实验介绍

## 1. 简介

本实验提出并实现了一种**三阶段智能分片算法**（Three-Stage Intelligent Chunker），用于在 RAG（Retrieval-Augmented Generation）系统中将原始语料切分为语义连贯的分片。目标是通过更合理的语义边界切分，提升下游检索质量。

核心动机：固定长度分块（如按 128 token 硬切）会破坏句子间的语义关联，本方法通过"结构感知 + 困惑度 + 信息增益"三阶段处理，让分片边界对齐语义转折点。

---

## 2. 方法设计

### 2.1 整体流程

```
原始语料（每行一篇新闻）
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ 阶段一：体裁检测 + 分片参数推荐                                │
│  - 关键词快速检测（paper / news / novel / chat / doc）     │
│  - Qwen3-8B LLM 推荐 l_min / l_max（流式调用）               │
│  - 无 API Key 时回退到内置默认表                              │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ 阶段二：PPL 降噪 + 边界检测                                  │
│  - 字符级 Bigram N-gram PPL（无需 GPU）                     │
│  - 可选：本地 HuggingFace LM PPL（--ppl_model_name）       │
│  - 剔除离群句：mean + 3σ 阈值                                │
│  - 切分点：PPL 跳变 mean + 1σ 阈值                          │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ 阶段三：信息增益（IG）融合优化                                │
│  - 过大的 chunk：贪婪 IG 合并（字符频率向量余弦相似度）        │
│  - 过小的 chunk：合并至邻居（阈值 beta_small * l_min）        │
└──────────────────────────────────────────────────────────┘
        │
        ▼
output_chunks/all_chunks_chunks.json
```

### 2.2 阶段一：体裁检测与参数推荐

#### 关键词快速检测

```python
GENRE_RECOMMENDATIONS = {
    "doc":   {"l_min": 400,  "l_max": 1000, "reasoning": "默认技术文档"},
    "news":  {"l_min": 300,  "l_max": 600,  "reasoning": "新闻资讯宜短"},
    "paper": {"l_min": 500,  "l_max": 1200, "reasoning": "学术论文可较长"},
    "novel": {"l_min": 400,  "l_max": 800,  "reasoning": "小说故事适中"},
    "chat":  {"l_min": 200,  "l_max": 500,  "reasoning": "对话记录更短"},
}
```

检测逻辑：根据 `摘要/abstract/参考文献` → paper；`日报/快讯/新华社` → news；`他说着/心想` → novel；`：`占比 > 2% → chat；默认 → doc。

#### LLM 参数推荐

调用 Qwen3-8B（DashScope 兼容接口）流式输出 JSON 推荐：

```json
{"genre": "news", "l_min": 300, "l_max": 600, "reasoning": "..."}
```

> 兼容 Qwen3 思考模式（`stream=True`），无 API Key 时回退到默认表。

### 2.3 阶段二：PPL 计算与边界检测

#### 字符级 Bigram PPL（默认）

$$P(ch|prev) = \frac{\text{count}(prev, ch) + 1}{\text{count}(prev) + |V|}$$

$$PPL = \exp\left(-\frac{1}{N}\sum_i \log P(ch_i|prev_i)\right)$$

- 词典大小 |V| = 65536（Unicode 范围）
- 不依赖任何深度学习模型，**纯 CPU 即可运行**
- 训练阶段：对整个语料统计 unigram/bigram 频次

#### 本地 LM PPL（可选）

`--ppl_model_name` 指定后加载 HuggingFace 因果 LM，按 token 级 NLL 算 PPL：
- 精度：`torch.float16`
- 自动 `device_map="auto"`（CUDA 优先）
- 输入最大长度 512 token

#### 降噪与切分

| 阶段 | 阈值 | 动作 |
|---|---|---|
| 降噪 | PPL > mean + 3σ | 删除该句 |
| 切分 | PPL 跳变 > mean + 1σ | 在此处断开 |

### 2.4 阶段三：信息增益融合

字符频率向量的余弦相似度作为相似性度量：

$$e_i = \frac{\text{counter}_i}{\|\text{counter}_i\|}, \quad \text{sim} = \sum_k e_1[k] \cdot e_2[k]$$

$$\text{IG}(A, B) = \frac{1}{1 + (1 - \text{sim})}$$

融合规则：
- **过大 chunk**（> `l_max`）：寻找 IG 最低的相邻块进行合并，直到 ≤ `l_max`
- **过小 chunk**（< `β_small * l_min`）：若两邻居 IG 较高则合并

---

## 3. 实现

### 3.1 关键文件

| 文件 | 行数 | 作用 |
|---|---:|---|
| `integrated_chunker.py` | 723 | 三阶段分片主程序 |
| `convert_chunks_to_txt.py` | 21 | JSON → 单目录 .txt |
| `output_chunks/convert_chunks.py` | 43 | JSON → 分目录 .txt（绕过 Windows 单目录文件数限制） |
| `analyze_chunks.py` | 42 | 分块统计脚本 |
| `run_crud_exp.py` | 303 | 端到端 CRUD 评测脚本 |

### 3.2 关键类

```python
# 字符级 PPL（无需依赖）
class CharNgramPPLScorer:
    def train(self, corpus: str)
    def score(self, context: str, sentence: str) -> float

# 本地 LM PPL（可选）
class LocalHuggingFacePPLScorer:
    def score(self, context: str, sentence: str) -> float

# 信息增益（基于字符频率向量）
class IGCalculator:
    def embedding(self, text: str) -> dict
    def ig(self, left: ChunkBlock, right: ChunkBlock) -> float

# 主分片器
class IntegratedChunker:
    def __init__(self, l_min, l_max, window_w, beta, beta_small, denoise)
    def chunk_document(self, content: str) -> list[ChunkBlock]
    def process_file(self, input_path, output_path)
```

### 3.3 CLI 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input` | 必填 | 输入文件/目录 |
| `--output` | 必填 | 输出目录 |
| `--line_mode` | off | 每行视为独立文档 |
| `--llm_api_key` | - | LLM API Key（无则用默认表） |
| `--llm_base_url` | DashScope | LLM API Base URL |
| `--llm_model` | `qwen3-8b` | LLM 模型 |
| `--ppl_model_name` | `""` | 本地 LM PPL 模型（空 = 字符 n-gram） |
| `--window_w` | 3 | PPL 上下文窗口大小 |
| `--beta` | 1.1 | 过大块合并上限系数 |
| `--beta_small` | 0.8 | 过小块合并下限系数 |
| `--denoise` | True | 启用 PPL 降噪 |
| `--llm_sample_interval` | 300 | line_mode 下 LLM 调用频率 |

---

## 4. 分片输出与统计

### 4.1 输出文件

```
output_chunks/
├── all_chunks_chunks.json     ← 主结果
├── all_chunks_chunks.txt      ← 文本格式（--- 分隔）
├── all_chunks_summary.json    ← 统计信息
├── convert_chunks.py
└── chunks_dir/                ← 最终用于检索
    ├── dir_0000/chunk_000000.txt ... chunk_000999.txt
    ├── dir_0001/...
    └── dir_0016/chunk_016400.txt
```

### 4.2 实际分片统计

| 指标 | 数值 |
|---|---|
| 原始语料 | 10,451 行（`db_qa.txt`） |
| **生成 chunk 数** | **16,401** |
| **平均 chunk 长度** | **505.5 字符** |
| 最长 chunk | 1,291 字符 |
| 最短 chunk | 5 字符 |
| **压缩比** | **1.57x**（每行平均切 1.57 段） |
| 总字符数 | 8,290,966 |

#### 长度分布

| 区间 | 数量 | 占比 |
|---|---:|---:|
| < 200 | 1,348 | 8.2% |
| 200 – 400 | 4,158 | 25.4% |
| 400 – 600 | 5,478 | **33.4%**（主峰） |
| 600 – 800 | 3,818 | 23.3% |
| ≥ 800 | 1,599 | 9.7% |

> 分布以 400-600 为主峰（与 news 体裁 `l_max=600` 一致），符合预期。

### 4.3 示例 chunks

```json
[
  {
    "chunk_text": "，正文：建设"一刻钟便民服务圈"事关百姓冷暖、民生福祉。朝阳区大屯街道...",
    "chunk_len": 481
  },
  {
    "chunk_text": "街道依托"众享大屯"生活服务品牌，整合优质服务资源...",
    "chunk_len": 395
  },
  {
    "chunk_text": "2023-08-11 19:29:38，正文：新华社北京8月11日电...",
    "chunk_len": 86
  }
]
```

可以看到同一主题（便民服务圈）被切分为两段（481 + 395 字符），这是阶段三 IG 融合在 600 阈值处合理拆分的结果。

---

## 5. 下游 RAG 评测

### 5.1 评测环境

| 组件 | 选型 |
|---|---|
| 评测框架 | CRUD RAG（基于 Meta-Chunking/eval/CRUD） |
| 检索 | Milvus + BAAI/bge-base-zh-v1.5（768 维） |
| LLM | Qwen2.5-7B-Instruct（SiliconFlow API） |
| 召回 top_k | 8 |
| 任务 | `quest_answer`（1Doc / 2Docs / 3Docs） |

### 5.2 评测指标

| 指标 | 计算方式 | 范围 |
|---|---|---|
| BLEU-1/2/3/4 | n-gram 匹配（jieba） | 0-1 |
| BLEU-avg | BLEU / brevity_penalty | 0-1 |
| ROUGE-L | 最长公共子序列（jieba） | 0-1 |
| BERTScore | `text2vec-base-chinese` 语义相似度 | 0-1 |

### 5.3 实验结果（top_k=8, Qwen2.5-7B）

| 子任务 | bleu-avg | bleu-1 | bleu-4 | rouge-L | bertScore | 有效样本 |
|---|---:|---:|---:|---:|---:|---:|
| **QA-1Doc** | 0.1595 | 0.3361 | 0.0996 | **0.3645** | 0.7198 | 3,121 |
| **QA-2Docs** | 0.1055 | 0.3439 | 0.0532 | 0.2733 | 0.8162 | 3,094 |
| **QA-3Docs** | 0.0990 | 0.3670 | 0.0453 | 0.2655 | **0.8323** | 3,102 |

> 总样本 3,199/任务，有效率 > 96%，约 78 条/任务因 token 超限被跳过。

---

## 6. 使用方法

### 6.1 运行分片

```bash
# 阶段一 + 二 + 三（端到端）
python integrated_chunker.py \
    --input ./data/db_qa.txt \
    --line_mode \
    --output ./output_chunks \
    --llm_api_key <KEY> \
    --llm_model qwen3-8b
```

### 6.2 转换为评测格式

```bash
# 生成 chunks_dir/
python output_chunks/convert_chunks.py
```

### 6.3 运行 CRUD 评测

```bash
cd Meta-Chunking/eval/CRUD

python quick_start.py \
    --model_name qwen_api \
    --data_path "C:\Users\胡铭强\Desktop\chunk_code\data\split_merged.json" \
    --docs_path "C:\Users\胡铭强\Desktop\chunk_code\output_chunks\chunks_dir" \
    --collection_name meta_chunks_full_v2 \
    --retrieve_top_k 8 \
    --task quest_answer \
    --construct_index \
    --bert_score_eval
```

---

## 7. 与 Meta-Chunking 的对比

| 维度 | 本方法 | Meta-Chunking |
|---|---|---|
| 阶段一 | 结构切分 + LLM 推荐 l_min/l_max | VLLM PPL |
| PPL 计算 | 字符 Bigram（默认）/ 本地 LM（可选） | VLLM 远程服务 |
| 依赖 | 仅 `urllib` + 可选 `torch/transformers` | 必须 VLLM 服务 |
| 阶段三融合 | 信息增益（字符频率向量） | Margin Sampling + 语义相似度 |
| 资源消耗 | **CPU 即可跑** | GPU + VLLM |
| 体裁适应 | 5 类内置 + LLM 推荐 | 无 |
| 可定制性 | 高（替换任意 scorer） | 较低 |

**核心优势**：
- 轻量（无 GPU 也能跑完 8,290,966 字符语料）
- 自适应（按体裁调参）
- 透明（每个阶段都可观察、可替换）

---

## 8. 局限与后续

### 8.1 已知局限

1. 字符 n-gram PPL 对中文语义敏感度弱于基于 LM 的 PPL
2. 关键词体裁检测对长尾领域（法律、医疗）效果差
3. IG 融合的字符频率向量粒度较粗

### 8.2 后续工作

- 引入基于 LLM 的 PPL scorer 提升边界检测精度
- 增加 doc/PDF 等多格式解析
- 收集 gold-split 标注以做端到端优化
