# 实验结果索引

本项目的实验结果分布在根目录、`code1/chapter2`、`code1/chapter3` 和多个第四章应用副本中。部分代码通过绝对路径或固定相对路径读取结果，因此当前阶段采用“建立索引、保留原路径”的方式整理，不直接移动结果目录。

## 1. 根目录 RAG 实验结果

| 目录 | 内容 | 是否属于运行依赖 |
|---|---|---|
| `experiment_data/` | 维度元数据、倒排索引、标签向量、查询缓存、Prompt/标签中间结果 | 是，检索和维度流程会读取 |
| `experiment_results/` | 根目录混合检索、排序和评测结果，主要为 JSONL/Markdown | 否，评测归档 |
| `fusion_results/` | 融合问答/检索实验结果 | 否，评测归档 |
| `output_chunks/` | 三轮分片、指代消解后的 Chunk、日志和汇总 | 是，入库流程可能读取 |
| `output_chunks_unified/` | 简化入库流程生成的统一 Chunk | 否，按需重新生成 |
| `output_no_ingest/` | 只分片/抽取、不写 MySQL/Qdrant 的实验输出 | 否，调试和对比用 |
| `cache/` | 语料增强进度和汇总缓存 | 按脚本使用 |

## 2. Chapter 2：文档转换与去噪

| 目录 | 内容 |
|---|---|
| `code1/chapter2/datas/` | PDF、PPT、Word 等原始数据，以及解析、规则去噪、LLM 去噪和最终文本 |
| `code1/chapter2/experiments/denoising_evaluation/` | 去噪质量评估结果 |
| `code1/chapter2/experiments/exp1/` | 规则去噪实验 |
| `code1/chapter2/experiments/exp2/` | 去噪/聚类/可视化实验 |
| `code1/chapter2/conversion_log.txt`、`llm_interaction.log` | 文档转换和 LLM 调用日志 |

Chapter 2 的原始文档和各阶段文本目前保留原目录，因为 `main.py` 和实验脚本使用这些目录作为输入输出位置。

## 3. Chapter 3：RAG、Prompt 演化与评测

主线代码位于 `code1/chapter3/codes/bylw_rag/`，结果按照研究主题划分如下：

### 3.1 Prompt 优化主线

| 目录 | 内容 |
|---|---|
| `new_experiments/datas/` | tourist、PubMedQA、NQ、PopQA、TriviaQA 数据 |
| `new_experiments/prompt_library/` | 每个问题的 Prompt 迭代库 |
| `new_experiments/iteration_results/` | Prompt 迭代过程和每轮结果 |
| `new_experiments/clustering_results/` | 问题聚类、聚类中心和聚类摘要 |
| `new_experiments/optimized_prompts/` | 聚类后的优化 Prompt |
| `new_experiments/only_q_results/` | 只使用问题文本的消融实验 |
| `new_experiments/q_a_prompt_attr_results/` | 问题、答案和 Prompt 属性联合实验 |
| `new_experiments/pubmedQA_results/` | PubMedQA 专项结果 |
| `new_experiments/top-k-*` | Top-K Prompt 或检索文档数量实验 |

### 3.2 检索与噪声实验

| 目录 | 内容 |
|---|---|
| `experiments/retrieval/level1_format_noise/` | 格式噪声 |
| `experiments/retrieval/level2_semantic_noise/` | 语义噪声 |
| `experiments/retrieval/level3_irrelevant_noise/` | 无关内容噪声 |
| `experiments/retrieval/retrieval_results/` | 检索原始结果 |
| `experiments/retrieval/metrics_results/` | 基础指标 |
| `experiments/retrieval/detailed_metrics/` | 详细指标第一版 |
| `experiments/retrieval/detailed_metrics_2/` | 详细指标第二版/汇总版 |
| `experiments/retrieval/llm_evaluation_results/` | LLM 评测结果 |
| `experiments/retrieval/visualization/` | 可视化结果 |

### 3.3 不同数据集的问答评测

| 目录 | 内容 |
|---|---|
| `experiments/tourist/`、`experiments/tourist2/` | 景区问答与 Prompt 演化实验 |
| `experiments/nq_results/` | Natural Questions 结果 |
| `experiments/popQA_results/` | PopQA 结果 |
| `experiments/triviaQA_results/` | TriviaQA 结果 |
| `experiments/results/` | 早期统一实验结果和消融实验 |

### 3.4 当前运行依赖

以下目录虽然位于备份路径，但当前根目录 Prompt 优化和交互式问答代码仍会读取它们，暂时不能删除或移动：

```text
code1/chapter3_backup/codes/bylw_rag/new_experiments/clustering_results/
code1/chapter3_backup/codes/bylw_rag/new_experiments/optimized_prompts/
code1/chapter3_backup/codes/bylw_rag/new_experiments/prompt_library/
code1/chapter3_backup/codes/bylw_rag/new_experiments/iteration_results/
```

## 4. Chapter 4：集成 Web 系统结果

`code1/chapter4` 是主版本，`chapter4_new`、`chapter4_upgrade`、`chapter4_xxx` 是不同迭代版本。每个版本的运行数据主要包括：

| 目录 | 内容 |
|---|---|
| `uploads/` | 用户上传的原始文件 |
| `outputs/` | 文件处理各阶段输出 |
| `logs/` | 处理日志 |
| `chroma_db/` | ChromaDB 向量数据 |
| `rag_data/` | RAG 元数据和测试数据 |

这些目录属于运行数据，不应和 Chapter 4 源码一起删除。后续可以在确认版本后，将旧版本的运行数据单独归档。

## 5. 整理原则

1. `code1/chapter2`、`code1/chapter3`、`code1/chapter4` 主线代码全部保留。
2. 先整理索引，再迁移或删除结果；不改变代码正在使用的路径。
3. `chapter3_backup` 先视为“当前景区 Prompt 运行数据”，不是普通备份。
4. 只有经过引用扫描、确认不会被脚本读取的重复结果，才移动到归档目录。
5. 生成物、日志和评测结果与源码分离，但不直接删除原始实验数据。
