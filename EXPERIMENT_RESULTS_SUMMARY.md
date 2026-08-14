# 去噪分片实验结果汇总

更新时间：2026-08-14  
数据集：`db_qa.txt` 及其问答数据  
对照组：`denoise_off`（关闭第二阶段去噪）与 `denoise_on`（开启第二阶段去噪）  
问答模型：`Qwen/Qwen3-8B`  

## 1. 结论摘要

目前问答实验已经完成，但整个切片质量实验还不是最终完整报告：

- 全量 QA 的 1-doc、2-doc、3-doc 均已完成，两组共 6 个结果 JSON 已生成。
- Boundary Clarity 已完成，去噪组的平均边界语义分离度略低。
- Chunk Stickiness 的四个 shard 已完成，合并后的加权统计显示去噪组结构熵略低，改善幅度很小。
- 检索证据保留实验已完成，去噪组 top-k 答案二元组覆盖率略低。
- MAD 阈值敏感性已完成，阈值越大，删除内容越少、chunk 数越多。
- Relation 质量评测仍在运行；API 边界过渡自然度和人工删除内容标注尚未开展。

现有结果不支持直接写出“去噪明显提升整体效果”。更准确的阶段性表述是：去噪减少了部分内容并略微改变了分片结构，但当前 QA 和自动检索证据指标没有显示稳定收益。

## 2. 全量问答实验

### 2.1 已完成范围

每个条件的有效问答规模为：1-doc 3199 条、2-doc 3189 条、3-doc 3188 条。两组共 6 个结果 JSON 已全部生成；其中 1-doc 的 on 组有 1 条无效记录，整体有效数为 3198。

### 2.2 汇总指标

| 任务 | 条件 | 有效结果数 | BLEU-avg | ROUGE-L | 平均答案长度 |
|---|---:|---:|---:|---:|---:|
| 1-doc | off | 3199 | 0.300412 | 0.483048 | 172.71 |
| 1-doc | on | 3198 | 0.301921 | 0.484601 | 171.81 |
| 2-doc | off | 3189 | 0.125607 | 0.278837 | 492.45 |
| 2-doc | on | 3189 | 0.124543 | 0.277601 | 494.51 |
| 3-doc | off | 3188 | 0.117881 | 0.273088 | 535.86 |
| 3-doc | on | 3188 | 0.118271 | 0.273494 | 535.45 |

`bertScore` 当前结果全部为 0，不能作为有效比较指标。

### 2.3 off/on 差异

| 任务 | BLEU-avg 变化（on-off） | ROUGE-L 变化（on-off） | 长度变化 |
|---|---:|---:|---:|
| 1-doc | +0.001509（约 +0.50%） | +0.001554（约 +0.32%） | -0.91 |
| 2-doc | -0.001064（约 -0.85%） | -0.001237（约 -0.44%） | +2.06 |
| 3-doc | +0.000390（约 +0.33%） | +0.000406（约 +0.15%） | -0.41 |

按相同问题做配对计算时，1-doc 的 BLEU-avg/ROUGE-L 平均变化为 `+0.001414/+0.001402`，2-doc 为 `-0.001064/-0.001237`，3-doc 为 `+0.000390/+0.000406`。三类任务中两类略升、一类略降，但提升幅度均很小，不能说明去噪带来稳定 QA 提升。

### 2.4 结果位置

服务器目录：

`/home/humq/chunk_code/results/denoise_ablation_db_qa_textsafe/qa_full_qwen3_dual2_20260811/`

已完成文件：

- `off/.../QuestAnswer1Doc_Qwen_Qwen3-8B.json`
- `on/.../QuestAnswer1Doc_Qwen_Qwen3-8B.json`
- `off/.../QuestAnswer2Docs_Qwen_Qwen3-8B.json`
- `on/.../QuestAnswer2Docs_Qwen_Qwen3-8B.json`
- `off/.../QuestAnswer3Docs_Qwen_Qwen3-8B.json`
- `on/.../QuestAnswer3Docs_Qwen_Qwen3-8B.json`

## 3. Boundary Clarity

指标定义为相邻 chunk embedding 的 `1 - cosine_similarity`，越高表示边界语义分离越明显。使用 v2 结果，chunk 映射 `unmatched_chunks=0`。

| 条件 | chunk 数 | 边界数 | 平均分离度 | 中位数 | 标准差 |
|---|---:|---:|---:|---:|---:|
| off | 17,342 | 6,990 | 0.205856 | 0.177526 | 0.110237 |
| on | 16,980 | 6,630 | 0.203916 | 0.177040 | 0.108014 |

去噪组平均值下降约 0.001939（约 -0.94%），95% bootstrap 区间有明显重叠。因此从该指标看，当前没有观察到边界清晰度提升。

结果文件：

- `/home/humq/chunk_code/metric_results/boundary_semantic_off_v2.json`
- `/home/humq/chunk_code/metric_results/boundary_semantic_on_v2.json`

## 4. Chunk Stickiness

四个 shard 已完成。下面是按每个 shard 的有效 chunk 数加权后的汇总；该指标使用 PPL 图结构熵，脚本定义为结构熵越低、chunk 内部黏连度越高。

| 条件 | 输入 chunk 数 | 评测 chunk 数 | 跳过数 | G1 加权均值 | G3 加权均值 |
|---|---:|---:|---:|---:|---:|
| off | 17,309 | 15,955 | 1,354 | 2.630070 | 2.753054 |
| on | 16,958 | 15,670 | 1,288 | 2.623998 | 2.746021 |

去噪组相对变化：G1 为 -0.006072（约 -0.23%），G3 为 -0.007033（约 -0.26%）。方向上属于轻微改善，但幅度很小，并且两组输入 chunk 数不同，最终应补充按文档或配对 chunk 的统计检验。

结果文件：

- `/home/humq/chunk_code/metric_results/fast_stickiness_off_shard0.json` 至 `shard3.json`
- `/home/humq/chunk_code/metric_results/fast_stickiness_on_shard0.json` 至 `shard3.json`

## 5. 检索证据保留实验

该实验在相同问题下检索 top-1、top-5、top-10 上下文，用参考答案二元组覆盖率作为自动代理指标，同时统计 exact answer hit rate。每个条件有效样本数均为 9,576：1-doc 3,199、2-doc 3,189、3-doc 3,188。

### 5.1 答案二元组覆盖率

| 任务 | k | off | on | on-off |
|---|---:|---:|---:|---:|
| 1-doc | 1 | 0.653437 | 0.648319 | -0.005118 |
| 1-doc | 5 | 0.799542 | 0.797841 | -0.001702 |
| 1-doc | 10 | 0.833119 | 0.831309 | -0.001810 |
| 2-doc | 1 | 0.381780 | 0.381452 | -0.000329 |
| 2-doc | 5 | 0.653523 | 0.651394 | -0.002130 |
| 2-doc | 10 | 0.721174 | 0.718282 | -0.002892 |
| 3-doc | 1 | 0.342028 | 0.340592 | -0.001437 |
| 3-doc | 5 | 0.619197 | 0.615959 | -0.003238 |
| 3-doc | 10 | 0.696856 | 0.693784 | -0.003072 |

各任务、各 k 下 on 均略低于 off。exact answer hit rate 整体很低，2-doc 和 3-doc 基本为 0，因此只能作为辅助观察，不能替代人工 gold evidence 标注。

结果文件：

- `/home/humq/chunk_code/metric_results/retrieval_evidence_full_20260812/off.json`
- `/home/humq/chunk_code/metric_results/retrieval_evidence_full_20260812/on.json`

## 6. MAD 阈值敏感性

三组实验均开启去噪，仅改变第二阶段 MAD 阈值 `k`。输入为同一份 `db_qa.txt`，输入字符数 8,314,681，第二阶段处理前句子数 227,933。

| MAD k | 输出 chunk 数 | 删除句子数 | 删除句子比例 | 删除字符数 | 删除字符比例 |
|---:|---:|---:|---:|---:|---:|
| 2.5 | 16,683 | 11,419 | 5.01% | 318,880 | 3.85% |
| 3.0 | 16,867 | 8,080 | 3.54% | 220,885 | 2.66% |
| 4.0 | 17,081 | 4,521 | 1.98% | 116,278 | 1.40% |

该实验确认阈值对去噪强度有明显影响：`k` 越小，删除越激进；`k` 越大，保留内容越多。它说明参数敏感性，不能单独证明某个阈值下的去噪内容一定是噪声。

结果目录：

`/home/humq/chunk_code/results/denoise_threshold_sensitivity_20260812/mad_k_*/all_chunks_summary.json`

## 7. 尚未完成或尚未开展

- Relation 质量评测仍在 GPU 0/1 运行，最终 JSON 尚未生成。
- Chunk Stickiness 的加权汇总尚未另存为独立 summary JSON。
- API 边界过渡自然度实验尚未开展。
- 删除句子是否为真实噪声的人工 gold 标注尚未开展。
- 尚未基于全部最终指标做 paired bootstrap、显著性检验和最终结论表。

## 8. 当前可写入报告的阶段性结论

在当前已完成结果内，去噪方法的主要作用是减少部分句子和字符，并使 chunk 数下降。结构质量方面，Chunk Stickiness 的结构熵有极小幅度下降，但 Boundary Clarity 略有下降；下游 QA 在 1-doc、3-doc 上略有提升，在 2-doc 上略有下降；自动检索证据覆盖率在三类任务上均略有下降。因此当前证据更适合表述为“去噪改变了分片结构，局部结构指标有轻微改善，但尚未证明整体检索问答收益”，不能表述为“去噪方法已经被充分证明有效”。
