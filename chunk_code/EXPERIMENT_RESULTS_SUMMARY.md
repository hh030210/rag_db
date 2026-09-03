# `chunk_code` 已有实验结果汇总

> 整理时间：2026-08-11  
> 整理范围：`chunk_code` 当前已有的分片输出、CRUD/QuestAnswer 结果、Boundary Clarity 报告、运行日志和结果摘要。  
> 说明：本文件只整理已有结果，没有重新运行实验，也没有修改原始输出、代码和日志。

## 1. 结论先看

当前最完整、最有比较价值的结果是 **Integrated Chunks vs Meta-Chunking full v2，top-k=8**：

- Integrated Chunks 在 1Doc、2Docs、3Docs 三个任务上的 ROUGE-L 和 BERTScore 都高于 Meta-Chunking v2。
- ROUGE-L 相对提升约 `+17.0% / +10.9% / +11.1%`；BERTScore 相对提升约 `+13.3% / +5.5% / +4.5%`。
- 代价是生成答案显著变长，平均长度增加约 `68%–84%`。因此当前结论是“信息覆盖更充分”，还不能直接等同于“最终产品体验更好”。
- Baseline v1 已完成建索引，但完整 QA 评测没有完成，当前不能宣称三方对比（Baseline / Meta / Integrated）已经闭环。
- Boundary Clarity 只有 Integrated 的部分结果；7B 与 14B 评估样本数不一致，需要统一失败处理后复跑。
- Chunk Stickiness 和 Relation Coherence 当前仍是待运行状态。
- 增强版 `output_enhanced_v2` 去重后从 15,151 降到 8,034 个 chunks，减少约 `47.0%`，明显高于设计文档预期的 `5%–15%`，必须单独验证是否误删内容。

## 2. 实验资产与状态

| 实验/结果 | 主要文件或目录 | 当前状态 | 能否用于最终结论 |
|---|---|---|---|
| Baseline v1 固定长度分片 | `output_baseline_v1/` | 分片完成；建索引完成；QA 评测缺失 | 暂不能做完整三方对比 |
| Meta-Chunking full v2 | `Meta-Chunking/eval/CRUD/output/meta_chunks_full_v2_top8_Qwen_API_Chat/` | 1/2/3 Docs QA 已有结果 | 可以作为当前主基线 |
| Integrated Chunks top-8 | `Meta-Chunking/eval/CRUD/output/eval_integrated_top8_top8_Qwen_API_Chat/` | 1/2/3 Docs QA 已有结果 | 可以与 Meta v2 做当前二方比较 |
| Integrated 结构化输出 | `data/db_qa_chunks/`、`output_chunks/` | 存在多个版本，chunk 数不同 | 使用前需绑定具体评测目录 |
| Enhanced v2 去重版 | `output_enhanced_v2/` | 分片和去重完成 | 需要召回/误删复验 |
| Meta v2 top-4 | `meta_chunks_final_top4_Qwen_API_Chat/` | 每个任务仅 `N=2` | 只能看作调试/小样本验证 |
| top-4 smoke test | `meta_chunks_smoke_test2_top4_Qwen_API_Chat/` | 每个任务仅 `N=1` | 不能做效果结论 |
| Boundary Clarity | `data/boundary_clarity_summary.json` | 7B/14B 部分完成 | 可作初步边界分析，需复跑 |
| Chunk Stickiness | `Meta-Chunking/MoC/our _metrics/eval_chunk_stickiness.py` | 尚未运行 | 无结果 |
| Relation Coherence | `Meta-Chunking/MoC/our _metrics/relation_eval*.py` | 尚未形成当前有效结果 | 无结果 |

## 3. 分片数量与长度统计

以下统计直接根据当前 JSON 输出中的 `chunk_len` 计算；不同目录代表不同运行版本，不能只按目录名认为它们是同一轮实验。

| 输出 | chunks | 总字符数 | 平均长度 | 最短 | 最长 | 备注 |
|---|---:|---:|---:|---:|---:|---|
| `output_baseline_v1/all_chunks_chunks.json` | 16,022 | 8,291,143 | 517.5 | 5 | 1,269 | 固定长度 baseline 输出 |
| `output_chunks/all_chunks_chunks.json` | 16,401 | 8,290,966 | 505.5 | 5 | 1,291 | 另一版分片输出，常用于早期报告 |
| `data/db_qa_chunks/all_chunks_chunks.json` | 15,929 | 7,559,532 | 474.6 | 1 | 1,269 | 报告中的 Integrated Chunks 版本 |
| `output_enhanced_v2/all_chunks_chunks.json` | 8,034 | 4,148,416 | 516.4 | 4 | 1,269 | 去重后结果；摘要记录去重前 15,151 |

所有主要输出都来自同一份 `db_qa.txt`（摘要记录 10,451 行），但不同运行版本的结构处理、参数或去重状态并不完全相同。后续评测必须明确记录：输入文件、输出目录、chunk 数、是否去重、PPL 类型、IG 类型、top-k 和生成模型。

## 4. 主实验：QuestAnswer QA，top-k=8

### 4.1 实验条件

- 数据：`data/split_merged.json`，报告记录 3,199 个 QA 样本，分为 1Doc、2Docs、3Docs。
- Embedding：`BAAI/bge-base-zh-v1.5`，768 维。
- 向量库：Milvus，默认 `localhost:19530`。
- 生成模型：`Qwen/Qwen2.5-7B-Instruct`，temperature `0.1`，max_new_tokens `1280`。
- 检索：纯向量检索，`retrieve_top_k=8`。
- 比较对象：Meta-Chunking full v2 与 Integrated Chunks。

### 4.2 Meta v2 与 Integrated 的结果

| 任务 | 指标 | Meta v2 | Integrated | 绝对变化 | 相对变化 |
|---|---|---:|---:|---:|---:|
| 1Doc | ROUGE-L | 0.3645 | **0.4264** | +0.0619 | **+17.0%** |
| 1Doc | BERTScore | 0.7198 | **0.8152** | +0.0954 | **+13.3%** |
| 1Doc | 平均答案长度 | 104.1 | 175.0 | +70.9 | +68.1% |
| 1Doc | 有效样本数 | 3,121 | 3,151 | +30 | — |
| 2Docs | ROUGE-L | 0.2733 | **0.3032** | +0.0299 | **+10.9%** |
| 2Docs | BERTScore | 0.8162 | **0.8608** | +0.0446 | **+5.5%** |
| 2Docs | 平均答案长度 | 190.8 | 351.7 | +160.9 | +84.3% |
| 2Docs | 有效样本数 | 3,094 | 3,132 | +38 | — |
| 3Docs | ROUGE-L | 0.2655 | **0.2950** | +0.0295 | **+11.1%** |
| 3Docs | BERTScore | 0.8323 | **0.8700** | +0.0377 | **+4.5%** |
| 3Docs | 平均答案长度 | 206.2 | 378.1 | +171.9 | +83.4% |
| 3Docs | 有效样本数 | 3,102 | 3,146 | +44 | — |

### 4.3 主实验可得出的结论

1. Integrated 在三个文档数量场景上都领先，单文档增益最大。
2. 多文档场景中增益仍然存在，但被 top-8 已经召回较多相关内容的效果部分稀释。
3. Integrated 生成答案明显更长。需要通过 top-k=4、简洁回答提示词和人工抽样确认长答案是有效信息还是冗余。
4. 这仍然是二方比较。Baseline v1 没有完成同条件 QA 评测，所以不能说 Integrated 相比固定长度 baseline 已经提升了多少。

### 4.4 BLEU 指标存在原始文件与报告不一致

现有 Markdown 报告多处写“两个 chunker 的 BLEU-1~4 均为 0”，但直接读取原始评测 JSON 可见：

- Integrated JSON 中 BLEU 各项确实为 0。
- Meta v2 JSON 中 BLEU 非零，例如 1Doc 的 BLEU-1 `0.3361`、BLEU-4 `0.0996`。

因此本次汇总不使用 BLEU 来支持 Integrated 的优越性。后续应确认是否存在指标版本、输出覆盖、统计脚本或报告抄录差异，再决定是否纳入 BLEU 对比。

## 5. top-4 结果：仅作冒烟验证

当前存在 `meta_chunks_final_top4` 和 `meta_chunks_smoke_test2_top4`，但样本量分别是每个任务 `N=2` 和 `N=1`。例如 `meta_chunks_final_top4` 的 1Doc/2Docs/3Docs ROUGE-L 为 `0.3408/0.4222/0.3083`，但样本太少，不能和 top-8 的 3,000 级样本结果比较。

这批文件只能证明 top-4 评测链路曾经跑通，不能证明 top-4 的质量、长度或成本收益。需要对 Integrated Chunks 在完整 QA 集上重新跑 top-4。

## 6. Boundary Clarity

### 6.1 指标

- `no_semantic_similarity = 1 - cosine_similarity`：跨 chunk 语义差异，越高通常越好。
- `no_transition_naturalness = 1 - LLM transition score`：跨 chunk 过渡不自然度，按当前报告定义越高通常越好。
- 语义相似度使用本地 `all-MiniLM-L6-v2`；过渡评分调用 SiliconFlow 的 Qwen2.5-7B/14B。

### 6.2 已有结果

| 评估模型 | 评估对数 | no_semantic_similarity | no_transition_naturalness | 耗时 |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | 14,272 | 0.3346 ± 0.1309 | **0.6346 ± 0.3740** | 12,571 秒，约 3.5 小时 |
| Qwen2.5-14B | 15,429 | 0.3357 ± 0.1305 | 0.5876 ± 0.3553 | 15,304 秒，约 4.3 小时 |

`no_semantic_similarity` 在两个模型下几乎一致，因为它主要由本地 MiniLM 计算。14B 的过渡不自然度反而低于 7B，可能是评估模型标定、对齐倾向或指标设计造成的差异，不能直接解释为分片质量变差。

### 6.3 可信度问题

7B 和 14B 的评估对数不同，原报告说明 API 失败时 `no_semantic_similarity` 和 `no_transition_naturalness` 没有成对丢弃。因此当前均值只能作为初步结果；重跑时应对每个 chunk pair 使用同一成功标记，任一指标失败就整体丢弃该 pair。

目前没有 Baseline v1 的 Boundary Clarity 结果，因此没有边界质量的基线对照。

## 7. 运行与评测过程中的异常

这些异常不一定使已有 JSON 完全无效，但说明复现实验前需要先修复或记录：

- `eval_full_top8.log`：索引完成后曾出现 `UnboundLocalError`，说明 QA 数据加载路径或任务分支有问题。
- `data/eval_integrated.log`：多次出现 Milvus `search_data value [None] is illegal`，说明部分查询的 embedding 或检索输入为空。
- `eval_full_top8_v2.log`：出现 `HuggingfaceEmbeddings` 缺少 `get_agg_embedding_from_queries` 和 `ZeroDivisionError` 警告，但最终仍写出了 Meta v2 结果 JSON。
- `data/build_baseline.log`：Baseline 载入 16,022 chunks，建索引约 189.2 秒。
- `data/build_integrated.log`：Integrated 载入 15,929 chunks，建索引约 185.5 秒。

因此后续结果需要同时保存：完整命令、依赖版本、失败样本数、有效样本数、collection 名称和原始日志，避免只保留最终平均指标。

## 8. 尚未完成的结果

按优先级排序：

1. **高优先级：补跑 Baseline v1 完整 QA**，建议单线程、失败样本跳过并记录，形成 Baseline / Meta / Integrated 三方表。
2. **高优先级：统一评测实现**，修复空 embedding、缺少 query embedding 方法和除零问题，并重新生成两种已完成结果，确认指标可比。
3. **高优先级：完整跑 Integrated top-4**，比较答案长度是否下降、BERTScore/ROUGE-L 是否保持。
4. **中优先级：修复并复跑 Boundary Clarity**，保证两项指标按同一 pair 统计。
5. **中优先级：抽样 30–50 条答案人工检查**，区分“有效的详细答案”和“冗余长答案”。
6. **中优先级：验证 Enhanced v2 去重**，对被合并/删除的 chunks 做重复率、召回率和人工误删检查。
7. **低优先级：运行 Chunk Stickiness 与 Relation Coherence**，并补充 Baseline 的同类指标。

## 9. 结果文件索引

- 分片统计：`output_baseline_v1/all_chunks_summary.json`、`output_chunks/all_chunks_summary.json`、`output_enhanced_v2/all_chunks_summary.json`、`data/db_qa_chunks/all_chunks_summary.json`
- 主 QA 原始结果：`Meta-Chunking/eval/CRUD/output/meta_chunks_full_v2_top8_Qwen_API_Chat/`、`Meta-Chunking/eval/CRUD/output/eval_integrated_top8_top8_Qwen_API_Chat/`
- top-4 冒烟结果：`Meta-Chunking/eval/CRUD/output/meta_chunks_final_top4_Qwen_API_Chat/`、`Meta-Chunking/eval/CRUD/output/meta_chunks_smoke_test2_top4_Qwen_API_Chat/`
- 边界结果：`data/boundary_clarity_summary.json`、`data/BOUNDARY_CLARITY_REPORT.md`
- 主要旧报告：`experiment_report.md`、`META_CHUNKING_EXPERIMENTS_REPORT.md`、`data/chunker_comparison_report.md`
- 设计和方法说明：`chunking_design.md`、`custom_chunking_intro.md`、`ENHANCEMENTS_README.md`

## 10. 去噪开关正式分片结果

本次针对 `data/db_qa.txt` 使用三阶段 `integrated_chunker.py` 做了严格开关对照。两组的 PPL、结构切分、Round 3 参数和输入完全相同，唯一变化是 `--denoise`。

| 组别 | Round 2 输入句子 | 删除句子 | 删除字符 | 最终 chunks | 最终字符数 |
|---|---:|---:|---:|---:|---:|
| `denoise_off` | 227,933 | 0 | 0 | 17,342 | 8,283,275 |
| `denoise_on` | 227,933 | 5,946 | 158,312 | 16,980 | 8,125,369 |

结果位置：

```text
results/denoise_ablation_db_qa_textsafe/
├── denoise_off/   # 不去噪
├── denoise_on/    # 启用 MAD 去噪
└── ablation_manifest.json
```

当前结论仅限于“开关确实改变了 Round 2 的句子保留结果”：去噪组删除了约 `2.61%` 的 Round 2 输入字符，最终 chunk 数减少约 `2.09%`。问答评测尚未在本机执行，因为本机没有 `torch`、`transformers`、`pymilvus`、`llama_index` 和 Milvus 服务；不能把这组分片统计直接当成去噪有效性的最终证据。

本地已准备好 QA 对照脚本 `run_denoise_qa_ablation.py`，它调用项目正确的 `Meta-Chunking/eval/CRUD/quick_start.py`，而不是旧的硬编码 Windows 建索引脚本。正式 QA 完成后，应把 Recall@k、证据保留率、ROUGE-L、BERTScore、答案长度和有效样本数追加到本节。
