# 去噪方法有效性评测方案

> 目标：证明去噪方法确实删除了无效/异常内容，同时没有删除有效信息，并且能够改善或至少不损害下游检索和问答。  
> 本方案针对 `/Users/a1234/chunk_code` 当前代码和结果设计，暂不修改分片器实现。

## 1. 先确认当前实现

当前有两个需要区分的“去噪”实现：

### 1.1 `integrated_chunker.py` 的 MAD 去噪

该实现会：

1. 先为句子计算 PPL。
2. 计算 `median + 3.5 × 1.4826 × MAD` 阈值。
3. 删除 PPL 超过阈值的句子。
4. 对剩余句子重新计算 PPL，再进行边界切分。

这是当前可以直接做开关消融的真实去噪方法。

### 1.2 `enhanced_chunker.py` 的 `LengthAwareDenoiser`

源码的注释意图是“只有高 PPL 且短于 `l_min` 的句子才删除”，但当前三个分支都会把句子 append 到 `kept_sents`，因此实际不会删除任何句子。`denoise_min_length_ratio` 当前不能改变删除行为。

所以现在不能用 `output_enhanced_v2` 证明长度感知去噪有效。该输出从 15,151 降到 8,034，主要是开启了 jieba fingerprint 去重，不能把这 47% 的减少归因于去噪。

## 2. 实验必须固定的条件

每组实验只能改变去噪条件，以下内容必须完全相同：

- 输入文件和文档顺序
- `line_mode`、`l_min/l_max`、`window_w`、PPL 模型
- 结构切分和信息增益合并逻辑
- 是否去重：验证去噪时必须关闭 dedup
- Embedding 模型、Milvus collection 构建方式和距离度量
- 生成模型、prompt、temperature、max_new_tokens
- QA 数据、查询顺序、top-k 和线程数
- 评测代码版本和依赖版本

`integrated_chunker.py` 源码含有 API key 默认值。为了让实验可复现并避免外部 LLM 参数推荐影响结果，运行时应显式传 `--llm_api_key "" --llm_sample_interval 0`，或先改成环境变量配置。

## 3. 第一层：句子级去噪消融

### 3.0 已完成的本地正式分片

针对 `/Users/a1234/chunk_code/data/db_qa.txt`，已使用同一份三阶段代码和同一组参数完成两次正式分片：

| 组别 | Round 2 输入句子 | 删除句子 | 删除字符 | 最终 chunks | 输出字符数 |
|---|---:|---:|---:|---:|---:|
| `denoise_off` | 227,933 | 0 | 0 | 17,342 | 8,283,275 |
| `denoise_on` | 227,933 | 5,946 | 158,312 | 16,980 | 8,125,369 |

两组使用 `char_ngram` PPL、`window_w=3`、`beta_small=0.8`、`beta=1.1`、`line_mode=true`，并关闭了 LLM 参数推荐（`llm_sample_interval=0`）。结果目录为：

```text
results/denoise_ablation_db_qa_textsafe/
├── denoise_off/
│   ├── all_chunks_chunks.json
│   ├── all_chunks_chunks.txt
│   ├── all_chunks_summary.json
│   └── docs/
├── denoise_on/
│   ├── all_chunks_chunks.json
│   ├── all_chunks_chunks.txt
│   ├── all_chunks_summary.json
│   └── docs/
└── ablation_manifest.json
```

这两组已经完成分片和评测文档转换，但还没有运行 Milvus/QuestAnswer 问答。因此目前只能确认去噪确实删除了 5,946 个高 PPL 句子，不能仅凭这个数字确认它们都是无效内容。

### 3.1 最少需要的两组

| 组别 | `--denoise` | 目的 |
|---|---|---|
| A：不去噪 | `false` | 控制组，保留所有 PPL 异常句子 |
| B：MAD 去噪 | `true` | 当前方法 |

运行命令：

```bash
cd /home/humq/chunk_code

python integrated_chunker.py \
  --input data/db_qa.txt \
  --line_mode \
  --output results/denoise_ablation/no_denoise \
  --denoise false \
  --llm_api_key "" \
  --llm_sample_interval 0

python integrated_chunker.py \
  --input data/db_qa.txt \
  --line_mode \
  --output results/denoise_ablation/mad_denoise \
  --denoise true \
  --llm_api_key "" \
  --llm_sample_interval 0
```

两组都要保存：chunk JSON、summary、完整日志和运行参数。不要开启 `--use_dedup`，否则无法分辨“去噪删除”和“重复 chunk 删除”。

### 3.2 阈值敏感性

如果 B 组有效，再增加：

| 组别 | MAD `k` | 目的 |
|---|---:|---|
| C | 2.5 | 更激进地删除异常句 |
| D | 3.0 | 中等阈值 |
| E | 3.5 | 当前默认 |
| F | 4.0 | 更保守地删除 |

每个阈值都要报告删除比例和下游指标，不能只挑效果最好的阈值。

## 4. 第二层：建立“噪声”真值

只看 PPL 或 chunk 数量不能证明删除的是噪声，需要建立可验证的标签。

### 4.1 合成噪声实验

从原始语料中抽取干净句子，人工或脚本插入已知噪声：

- 随机字符、乱码、截断句
- HTML/网页导航、版权、广告和重复模板
- 与当前文档主题无关的句子
- 重复句、重复新闻导语
- 异常时间戳、编号、日志行

因为插入位置和内容已知，可以直接计算：

```text
噪声召回率 = 删除的合成噪声 / 全部合成噪声
删除精确率 = 删除的合成噪声 / 全部被删除句子
噪声 F1 = precision 与 recall 的调和平均
有效内容保留率 = 保留下来的原始干净字符 / 原始干净字符
```

合成噪声的作用是验证“检测能力”，但不能替代真实语料验证，因为真实新闻中的短句、数字和时间也可能是有效信息。

### 4.2 人工标注实验

从原始句子中分层抽样 500–1,000 条，至少覆盖：

- PPL 低、中、高分位
- 短句和长句
- 新闻、说明文、问答等不同文体
- 数字/时间戳密集句子

由两名标注者独立标记：

- `useful`：对回答问题或保持文档语义有帮助
- `noise`：删除后不影响文档事实和问答
- `uncertain`：无法判断，单独统计

报告 Cohen’s Kappa，先确认标注一致性，再计算去噪 precision、recall、F1 和有效内容误删率。高 PPL 不应直接等同于噪声，尤其要检查新闻时间、数字和专有名词句子。

## 5. 第三层：下游检索评测

将 A/B 两组分别建索引，使用相同 QA 集和检索参数。建议至少报告：

- Recall@1、Recall@5、Recall@10
- MRR、nDCG@k
- context precision：召回 chunks 中真正包含答案证据的比例
- context recall：答案所需证据是否被召回
- 被删除句子是否包含 QA 的 gold evidence

最关键的保护指标是：

```text
证据保留率 = 去噪后仍能召回 gold evidence 的 QA 数 / 全部 QA 数
```

如果去噪组删除了很多句子，但 Recall@k 和证据保留率不下降，才说明删除内容大概率是检索噪声；如果 chunk 数变少但证据召回也下降，就不能宣称方法有效。

## 6. 第四层：端到端问答评测

在完全相同的生成配置下，比较 A/B 两组：

| 指标 | 作用 |
|---|---|
| ROUGE-L | 答案与参考答案的字面覆盖 |
| BERTScore | 语义相似度 |
| answerable rate | 是否包含回答问题所需事实 |
| hallucination rate | 是否引入检索内容之外的事实 |
| 平均答案长度 | 去噪是否减少无效上下文带来的冗余 |
| 延迟、token 消耗 | 去噪的工程收益 |

当前项目已有 ROUGE-L、BERTScore 和答案长度，可以继续使用；但必须补充“证据是否保留”和“是否幻觉”，否则只能证明平均文本指标变化，不能证明去噪删除的是无效内容。

## 7. 统计方法和判定标准

所有 QA 结果使用相同问题做配对比较，不要只比较两个总体均值。建议：

- 对 QA 结果做 1,000 次 paired bootstrap，报告均值差和 95% 置信区间。
- 对每个问题的分数差做 Wilcoxon signed-rank test。
- 同时报告绝对差，不只报告相对百分比。
- 若使用外部 LLM，temperature 设为 0；若不能设为 0，至少重复 3 个随机种子。
- 预先写下通过标准，避免跑完后挑选有利指标。

可以先采用下面的工程判定标准，之后根据业务要求调整：

1. 合成噪声 F1 高于不去噪基线，且删除精确率达到预设目标。
2. 人工标注的有效内容误删率低于 1%–2%。
3. 去噪后 gold evidence 保留率和 Recall@k 不低于不去噪组，或下降不超过预设容忍度（例如 1 个百分点）。
4. ROUGE-L/BERTScore 至少不显著下降；如果显著提升，同时答案长度和延迟下降，才可以称为“有效且有收益”。
5. 结论在不同文体和不同 PPL 阈值下基本稳定，而不是只在单一数据集或单一阈值上成立。

## 8. 推荐的执行顺序

1. 先修复或明确 `LengthAwareDenoiser` 的 no-op 实现；在修复前只评估 `integrated_chunker.py` 的 MAD 去噪。
2. 运行 `no_denoise` 与 `mad_denoise` 两组，确认真实删除数量和删除句子样例。
3. 做合成噪声测试，得到可重复的 precision/recall/F1。
4. 做 500–1,000 条人工标注，检查高 PPL 短句是否被误删。
5. 关闭 dedup，完成两组独立建索引和 Recall@k/证据保留率对比。
6. 运行同一批 QA 的端到端 ROUGE-L、BERTScore、答案长度和幻觉率评测。
7. 最后再比较 MAD 阈值和长度保护参数，并确定默认配置。

### 8.1 正确的问答实验入口

项目内真正负责完整 QuestAnswer QA 的入口是：

```text
Meta-Chunking/eval/CRUD/quick_start.py
```

`build_unified.py` 只负责向 Milvus 建索引，`run_crud_exp.py` 是早期包装脚本，当前不作为这次主实验入口。为保证两组参数一致，已新增：

```text
run_denoise_qa_ablation.py
```

在已准备好 Milvus、Embedding 和 Qwen API 环境后运行：

```bash
export QWEN_OPENAI_API_KEY='你的 API Key'
python run_denoise_qa_ablation.py --bert_score_eval
```

该脚本会：

1. 对 `denoise_off/docs` 建立独立 collection 并跑 1Doc/2Docs/3Docs。
2. 对 `denoise_on/docs` 建立另一个独立 collection 并跑相同 QA。
3. 默认使用 `qwen_api`、top-k=8、单线程，减少多线程崩溃和 API 限速影响。
4. 将结果写入 `results/denoise_ablation_db_qa_textsafe/qa_output/`。
5. 将两组实际命令写入 `qa_ablation_manifest.json`。

## 9. 最终结论模板

只有满足下面的证据链，才建议在报告中写“去噪有效”：

```text
PPL/MAD 能识别人工定义和人工标注的真实噪声
→ 删除精确率和召回率达到预设标准
→ 有效句子误删率很低
→ QA gold evidence 保留，Recall@k 不下降
→ ROUGE-L/BERTScore 不下降或提升
→ 上下文长度、延迟或 token 成本下降
```

如果只能证明“删除了很多句子”或“chunk 数减少”，结论应写成“去噪改变了分片规模”，而不是“去噪有效”。
