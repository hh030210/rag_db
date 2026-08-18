# Meta-Chunking & Integrated Chunks 综合实验报告

> **项目**: Meta-Chunking - 基于语义连贯性的动态文本分块
> **实验时间**: 0807

---

## 一、实验总览

本项目围绕"**动态语义感知分块（Dynamic Semantic-Aware Chunking）**"展开，并补充了第二阶段去噪开关消融实验：

| # | 实验名称 | 评测对象 | 核心指标 | 状态 |
|---|---------|---------|---------|------|
| 1 | **QuestAnswer QA 评测** | RAG Pipeline (chunker + retriever + generator) | ROUGE-L, BERT-Score | ✅ 完成 |
| 2 | **Boundary Clarity** | chunk 边界质量 | no_sim, no_trans_nat | ✅ 完成（含 v2） |
| 3 | **Chunk Stickiness** | chunk 内部语义黏连度 | structural_entropy | ✅ shard 完成 |
| 4 | **Relation Coherence** | 跨 chunk 关系保持 | structural_entropy (relation graph) | ⏳ 运行中 |
| 5 | **Denoise Ablation** | 第二阶段去噪的有效性 | QA、边界、黏连、证据保留 | ✅ QA/辅助实验完成 |

---

## 二、QuestAnswer QA 评测（主实验）

### 2.1 实验设计

**目的**: 端到端评测 RAG pipeline 中，不同分块策略对问答质量的影响。

**Pipeline 架构**:

```
[待分块文档]
    ↓
[Chunker] → Baseline | Meta-Chunking v2 | Integrated Chunks
    ↓
[Embedding (BAAI/bge-base-zh-v1.5)] → Milvus 向量数据库
    ↓
[Retriever] → Top-4 or Top-8 相关 chunks
    ↓
[Generator (Qwen2.5-7B-Instruct)] → 生成答案
    ↓
[Eval] → ROUGE-L, BERT-Score vs Gold Answer
```

**数据集**:
- 来源: `data/split_merged.json` — 3199 个问答对（基于医疗/新闻领域新闻）
- 每个样本包含来自 1 / 2 / 3 份文档的新闻拼接，作为检索背景

**评测集**:

| 任务 | 说明 | Gold 来源 |
|------|------|---------|
| QuestAnswer**1Doc** | 单文档场景，问题围绕 1 份文档的事实 | `split_merged.json` 中的 ground-truth 答案 |
| QuestAnswer**2Docs** | 双文档场景，需综合 2 份文档信息 | 同上 |
| QuestAnswer**3Docs** | 三文档场景，需综合 3 份文档信息 | 同上 |

**评测指标**:
- `ROUGE-L`: 生成答案与 Gold 答案的最长公共子序列覆盖率（越高越接近参考）
- `BERT-Score`: 基于 `bert-base-chinese` 的语义相似度（越高越语义相近）
- `avg.length`: 生成答案的平均 token 数（反映检索上下文的丰富程度）

**检索配置**:
- Embedding: `BAAI/bge-base-zh-v1.5` (dim=768)
- 向量库: Milvus (`localhost:19530`)
- 生成模型: `Qwen/Qwen2.5-7B-Instruct` (temperature=0.1, max_new_tokens=1280)

### 2.2 被评测的分块策略

| 名称 | 分块方式 | 说明 | chunks 数量 |
|------|---------|------|-------------|
| **Baseline v1** | 固定长度 (chunk_size=128) | LlamaIndex 固定切分 | ~16,022 |
| **Meta-Chunking full v2** | 基于 PPL + 语义相似度的动态边界 | 原版 meta-chunking，top-8 检索 | ~15,926 |
| **Meta-Chunking final (top-4)** | 同上，但 top-4 检索 | smoke test，N=2 | - |
| **smoke_test2 (top-4)** | smoke test | 仅验证 pipeline 可运行，N=1 | - |
| **Integrated Chunks** | Meta-Chunking 后处理整合 | 相邻相似 chunk 合并，top-8 检索 | ~15,929 |

### 2.3 评测结果

#### 2.3.1 主要对比：Integrated Chunks vs Meta-Chunking v2 (Top-8)

| Chunker | Task | N | ROUGE-L | BERT-Score | avg_length (tokens) |
|---------|------|---|---------|------------|---------------------|
| **Integrated Chunks** (top-8) | 1Doc | 3,151 | **0.4264** | **0.8152** | 175.0 |
| Meta-Chunking v2 (top-8) | 1Doc | 3,121 | 0.3645 | 0.7198 | 104.1 |
| **Improvement** | | | **+17.0%** | **+13.3%** | +68.1% |
| **Integrated Chunks** (top-8) | 2Docs | 3,132 | **0.3032** | **0.8608** | 351.7 |
| Meta-Chunking v2 (top-8) | 2Docs | 3,094 | 0.2733 | 0.8162 | 190.8 |
| **Improvement** | | | **+10.9%** | **+5.5%** | +84.3% |
| **Integrated Chunks** (top-8) | 3Docs | 3,146 | **0.2950** | **0.8700** | 378.1 |
| Meta-Chunking v2 (top-8) | 3Docs | 3,102 | 0.2655 | 0.8323 | 206.2 |
| **Improvement** | | | **+11.1%** | **+4.5%** | +83.4% |

> 注: BLEU-1~4 在所有 chunker 上均为 0（典型现象：生成答案与参考答案的精确 n-gram 重合度天然很低）



#### 2.3.2 第二阶段去噪开关消融结果

为验证三阶段分片中第二阶段去噪的有效性，新增 `denoise_off`/`denoise_on` 对照。两组使用同一份数据、同一检索配置和 `Qwen/Qwen3-8B`，唯一变量是第二阶段去噪开关。以下结果均保留三位小数。

| 任务 | 条件 | 有效结果数 | BLEU-avg | ROUGE-L |
|---|---:|---:|---:|---:|
| 1-doc | off | 3199 | 0.281 | 0.464 |
| 1-doc | on | 3198 | 0.302 | 0.485 |
| 2-doc | off | 3189 | 0.114 | 0.251 |
| 2-doc | on | 3189 | 0.125 | 0.278 |
| 3-doc | off | 3188 | 0.102 | 0.255 |
| 3-doc | on | 3188 | 0.118 | 0.273 |

`on - off` 差值为：

| 任务 | BLEU-avg | ROUGE-L |
|---|---:|---:|
| 1-doc | +0.021 | +0.021 |
| 2-doc | +0.011 | +0.027 | 
| 3-doc | +0.016 | +0.018 | 

结果显示，去噪组在 1-doc、3-doc 上略有提升，在 2-doc 上略有下降；差异幅度较小，暂不能证明去噪能够稳定提升端到端 QA 效果。1-doc on 组有 1 条无效记录，`bertScore=0.000`，均未用于有效结论。



