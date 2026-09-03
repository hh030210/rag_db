# Meta-Chunking RAG 实验报告

## 1. 实验目标

评估基于 **Meta-Chunking** 语义分块策略构建的 RAG 系统在中文新闻问答任务上的表现。核心问题：相比朴素的固定长度分块，Meta-Chunking 能否通过更合理的语义边界切分提升检索质量，进而提升生成质量。

实验基于 **CRUD RAG** 评测框架，在三个子任务上评估：单文档问答（1Doc）、双文档问答（2Docs）、三文档问答（3Docs）。

---

## 2. 数据集

### 2.1 原始语料

| 项目 | 内容 |
|---|---|
| 文件 | `data/db_qa.txt` |
| 格式 | 每行一篇中文新闻报道，标题+正文 |
| 总文档数 | ~10,441 篇 |
| 领域 | 新华网中文新闻 |

### 2.2 评测问答对

| 文件 | `data/split_merged.json` |
|---|---|
| 来源 | 从原始语料中筛选、标注 |
| 子任务 | `questanswer_1doc`、`questanswer_2docs`、`questanswer_3docs` |
| 每个子任务样本数 | 3,199 条 |
| 每条字段 | `ID`、`questions`、`answers` |

```json
{
  "ID": "64fa9b27b82641eb8ecbe14c",
  "questions": "国家卫生健康委在2023年7月28日开展的...",
  "answers": "启明行动是为了防控儿童青少年的近视问题..."
}
```

---

## 3. 分块策略：Meta-Chunking

### 3.1 为什么需要智能分块

固定长度分块（如按 128 token 硬切）会导致两种问题：
- **过度切割**：将语义连贯的段落强行断开，破坏句子间依赖关系
- **欠分割**：将两个不相关的主题合并进同一个 chunk，引入检索噪声

### 3.2 三阶段分块流程

由 `integrated_chunker.py` 实现：

```
原始语料 (db_qa.txt)
      │
      ▼
┌─────────────────────────────────────────────┐
│  阶段一：结构化预切分 + LLM 参数推荐           │
│  - 按 markdown 标题、分隔符切分                │
│  - 调用 Qwen3-8B API 推荐 l_min / l_max      │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  阶段二：PPL（困惑度）降噪 + 边界检测         │
│  - 计算句级字符 n-gram PPL                   │
│  - 剔除高困惑度离群句（均值 + 3σ）             │
│  - 在 PPL 跳变处切分（均值 + 1σ）             │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  阶段三：信息增益（IG）融合优化                │
│  - 对过大 chunk 做贪婪 IG 合并                │
│  - 对过小 chunk 合并至邻居                    │
└─────────────────────────────────────────────┘
      │
      ▼
all_chunks_chunks.json
```

**关键超参数**：

| 参数 | 值 | 说明 |
|---|---|---|
| `--line_mode` | on | 每行视为独立文档 |
| `--window_w` | 3 | PPL 上下文窗口大小 |
| `--beta` | 1.1 | 过小块合并上限系数 |
| `--beta_small` | 0.8 | 过小块合并下限系数 |
| `--denoise` | True | 启用 PPL 降噪 |

### 3.3 分块结果

分块输出经 `convert_chunks.py` 整理为单个 `.txt` 文件目录：

```
output_chunks/chunks_dir/
├── dir_0000/
│   ├── chunk_000000.txt
│   └── ...
└── dir_0016/
    └── chunk_016400.txt
```

- **总 chunk 数**：~16,401 个
- **存储格式**：每文件一 chunk，纯文本

---

## 4. 检索系统

### 4.1 技术栈

| 组件 | 技术选型 |
|---|---|
| 向量数据库 | **Milvus**（本地部署，端口 19530） |
| Embedding 模型 | `BAAI/bge-base-zh-v1.5`（768 维） |
| Embedding 客户端 | LangChain `LangchainEmbedding` 封装 |
| 检索器类型 | `base`（纯向量检索），可选 BM25 / Hybrid |

### 4.2 索引构建

首次运行时需加 `--construct_index`：

1. 遍历 `chunks_dir` 下所有 `.txt` 文件
2. 按换行切分文本行（忽略 < 10 字符的行），构建 `Node` 对象
3. 用 BGE 模型编码所有 Node，批量写入 Milvus
4. Milvus 分批写入上限 8,000 条，分 3 批完成（0 / 8000 / 16000）

> **注意**：索引构建仅需首次执行，后续复用同一 collection 可去掉 `--construct_index`。

### 4.3 检索参数

| 参数 | 值 | 说明 |
|---|---|---|
| `--retrieve_top_k` | 8 | 每次查询召回 8 个 chunk |
| `--retriever_name` | base | 纯向量检索 |

检索流程：
```
用户问题 → BGE 编码 → Milvus 相似度搜索 → 取 top 8 chunks → 以 "\n-----\n" 连接 → 送入 LLM
```

---

## 5. 生成模型

| 配置项 | 值 |
|---|---|
| API 提供方 | SiliconFlow（OpenAI 兼容接口） |
| API Base | `https://api.siliconflow.cn/v1` |
| 模型 | `Qwen/Qwen2.5-7B-Instruct` |
| temperature | 0.1 |
| max_new_tokens | 1280 |
| top_p | 0.9 |
| top_k | 5 |

---

## 6. 评测流程

### 6.1 端到端 pipeline

```
split_merged.json（3,199 条/子任务）
      │
      ▼
┌──────────────────────────────────────────────┐
│  BaseEvaluator（多线程并行，8 threads）         │
│                                              │
│  for each sample:                            │
│    1. retrieve_docs(question) → 8 chunks     │
│    2. model_generation(question + chunks)    │
│    3. scoring(prediction, ground_truth)       │
└──────────────────────────────────────────────┘
      │
      ▼
output/{collection}_top{k}_{model}/{TaskName}_{model}.json
```

### 6.2 Prompt 模板（RAG 生成）

```
你是一位新闻编辑，现在，你被提供了1个问题，和根据这些问题检索到的文档，
请分别检索内容和你自身的知识回答这些问题...

问题：{question}

检索到的文档：{search_documents}

请给出你的回答（回答的文本写在<response></response>之间。
```

### 6.3 评测指标

| 指标 | 计算方式 | 说明 |
|---|---|---|
| **BLEU-1/2/3/4** | n-gram 匹配率（jieba 分词） | 衡量生成文本与参考答案的字面重叠 |
| **BLEU-avg** | BLEU / brevity_penalty | 无 brevity penalty 修正版 |
| **ROUGE-L** | 最长公共子序列 | 衡量生成文本对参考答案的覆盖度 |
| **BERTScore** | `text2vec-base-chinese` 语义相似度 | 衡量生成内容的语义质量 |
| **QA_avg_F1** | 子问题分解 + 词级 F1 | GPT 辅助（需加 `--quest_eval`） |
| **QA_recall** | 子问题回答率（1 - "无法推断"比例） | GPT 辅助（需加 `--quest_eval`） |

> 默认评测仅启用 BLEU、ROUGE-L、BERTscore；`--quest_eval` 额外启用 GPT 驱动的子问题分解 F1（计算代价显著更高）。

---

## 7. 实验配置

### 7.1 完整命令

```powershell
cd "C:\Users\胡铭强\Desktop\chunk_code\Meta-Chunking\eval\CRUD"

$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_OFFLINE = "0"
$env:TRANSFORMERS_OFFLINE = "0"

python quick_start.py `
    --model_name qwen_api `
    --temperature 0.1 `
    --max_new_tokens 1280 `
    --data_path "C:\Users\胡铭强\Desktop\chunk_code\data\split_merged.json" `
    --shuffle True `
    --docs_path "C:\Users\胡铭强\Desktop\chunk_code\output_chunks\chunks_dir" `
    --docs_type txt `
    --chunk_size 128 `
    --chunk_overlap 0 `
    --retriever_name base `
    --collection_name meta_chunks_full_v2 `
    --retrieve_top_k 8 `
    --task quest_answer `
    --num_threads 8 `
    --show_progress_bar True `
    --construct_index `
    --bert_score_eval
```

### 7.2 参数说明

| 参数 | 值 | 含义 |
|---|---|---|
| `--model_name` | qwen_api | SiliconFlow Qwen2.5-7B-Instruct |
| `--temperature` | 0.1 | 低随机性，保证答案稳定性 |
| `--max_new_tokens` | 1280 | 最大生成 1280 token |
| `--data_path` | .../data/split_merged.json | 评测 QA 对（绝对路径） |
| `--shuffle` | True | 打乱评测顺序 |
| `--docs_path` | .../output_chunks/chunks_dir | 分块后的 chunk 文件目录 |
| `--docs_type` | txt | chunk 文件类型 |
| `--chunk_size` | 128 | **传递参数**（实际 chunk 已预分好，不影响） |
| `--chunk_overlap` | 0 | **传递参数** |
| `--construct_index` | ✓ | 首次运行建索引，后续可省略 |
| `--retriever_name` | base | 纯向量检索 |
| `--collection_name` | meta_chunks_full_v2 | Milvus 集合名 |
| `--retrieve_top_k` | 8 | 每次召回 8 个 chunk |
| `--task` | quest_answer | 同时评测 1Doc / 2Docs / 3Docs |
| `--num_threads` | 8 | 8 线程并发请求 API |
| `--bert_score_eval` | ✓ | 启用 BERTScore 语义指标 |
| `--quest_eval` | — | 未启用（GPT 辅助 F1，默认关闭） |

---

## 8. 实验结果

### 8.1 运行信息

| 项目 | 值 |
|---|---|
| 运行时间 | 2026-06-16 10:28 → 13:35（约 3 小时 7 分钟） |
| 有效样本 | 1Doc: 3,121 / 2Docs: 3,094 / 3Docs: 3,102（共 9,317 条） |
| 跳过样本 | ~78 条/任务（因 token 超限或异常，< 3%） |
| API 并发 | 8 threads，实际吞吐量 ~1.2 it/s |

### 8.2 评测指标

| 子任务 | bleu-avg | bleu-1 | bleu-4 | rouge-L | bertScore | avg_length (token) | 有效样本 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **QA-1Doc** | **0.1595** | 0.3361 | **0.0996** | **0.3645** | 0.7198 | 104 | 3121 |
| **QA-2Docs** | 0.1055 | 0.3439 | 0.0532 | 0.2733 | 0.8162 | 191 | 3094 |
| **QA-3Docs** | 0.0990 | **0.3670** | 0.0453 | 0.2655 | **0.8323** | 206 | 3102 |

### 8.3 结果解读

- **ROUGE-L 随文档数递减**（0.36 → 0.27 → 0.27）：上下文变长后，生成内容与参考答案的字面重叠难度增大，符合预期。
- **BERTScore 随文档数递增**（0.72 → 0.82 → 0.83）：模型生成的答案在语义层面反而更接近参考答案，说明长上下文中检索到的相关内容更丰富。
- **bleu-4 显著下滑**（0.10 → 0.05 → 0.05）：长答案的高阶 n-gram 匹配极为困难，BLEU-4 的局限性在此体现。
- **生成答案长度**：1Doc ~104 token → 2Docs ~191 → 3Docs ~206，随上下文量级合理扩张。

---

## 9. 实验记录文件

### 9.1 关键文件一览

| 文件 | 作用 |
|---|---|
| `integrated_chunker.py` | Meta-Chunking 三阶段分块 |
| `convert_chunks_to_txt.py` | JSON chunk → 目录格式转换 |
| `output_chunks/convert_chunks.py` | 批量分发到子目录（绕过 Windows 单目录文件数限制） |
| `run_crud_exp.py` | 实验流程编排脚本 |
| `Meta-Chunking/eval/CRUD/quick_start.py` | CRUD 评测入口 |
| `src/datasets/xinhua.py` | QA 数据集加载 |
| `src/retrievers/base.py` | Milvus 检索器封装 |
| `src/metric/common.py` | BLEU / ROUGE / BERTScore 计算 |
| `src/metric/quest_eval.py` | GPT 辅助子问题分解 F1 |
| `src/configs/real_config.py` | 模型 API 配置 |

### 9.2 输出文件

```
Meta-Chunking/eval/CRUD/output/meta_chunks_full_v2_top8_Qwen_API_Chat/
├── QuestAnswer1Doc_Qwen_Qwen2.5-7B-Instruct.json   (3.96 MB)
├── QuestAnswer2Docs_Qwen_Qwen2.5-7B-Instruct.json  (5.47 MB)
└── QuestAnswer3Docs_Qwen_Qwen2.5-7B-Instruct.json  (5.89 MB)
```

日志文件：
```
C:\Users\胡铭强\Desktop\chunk_code\eval_full_top8_v2.log
```

---

## 10. 后续对比实验建议

| 对比实验 | 修改参数 | 预期发现 |
|---|---|---|
| **top_k=4 对比** | `--retrieve_top_k 4` | 减少噪声 chunk 对生成的影响 |
| **Naive Chunking 对比** | 使用固定长度分块的 `docs_path` | 基线对照，量化 Meta-Chunking 增益 |
| **启用 quest_eval** | 加 `--quest_eval` | 获取 GPT 驱动的语义 F1 / Recall |
| **Hybrid Retriever** | `--retriever_name hybrid` | 验证向量+关键词融合效果 |

> ⚠️ 注意：`--quest_eval` 需要 GPT API Key，结果更接近真实语义对齐，但计算耗时约增加 3-5 倍。
