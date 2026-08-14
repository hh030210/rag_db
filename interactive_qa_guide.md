# Interactive QA 交互式检索问答文档

> 交互式 RAG 问答工具 | 配置驱动生成版 | 2026-04-23

---

## 目录

1. [概述](#1-概述)
2. [文件架构](#2-文件架构)
3. [配置文件详解](#3-配置文件详解)
4. [命令行接口](#4-命令行接口)
5. [交互命令参考](#5-交互命令参考)
6. [生成流程](#6-生成流程)
7. [依赖模块](#7-依赖模块)

---

## 1. 概述

### 1.1 目标

提供交互式的 RAG 问答体验：用户输入问题 → 从 Milvus/MySQL 混合检索 → 返回语义相关 chunks → 可选调用 LLM 生成答案。

支持三种检索模式和两种重排方式：

**检索模式：**

| 模式 | 说明 |
|------|------|
| `fusion` | 维度检索 + 语义检索融合（默认） |
| `dim` | 纯维度检索（向量 + 标签 RRF） |
| `sem` | 纯语义检索（向量 ANN） |

**重排模式（仅 fusion 模式生效）：**

| 模式 | 说明 |
|------|------|
| `score` | 得分重排：根据两种检索的排名赋分，综合排序（默认） |
| `interleaved` | 交替去重：交替从两种检索结果中选取，保证多样性 |

### 1.2 核心能力

- **三种检索模式实时切换**：运行时通过 `/mode` 命令切换
- **两种重排模式可选**：运行时通过 `/rerank` 命令切换
- **LLM 问答生成**：基于检索到的 chunks 调用通义千问生成答案
- **运行时参数调优**：Top-K、alpha 权重、LLM 温度等均可热修改
- **verbose 调试模式**：显示维度约束、得分分解等内部信息

---

## 2. 文件架构

```
interactive_qa_config.yaml  →  gen_interactive_qa.py  →  interactive_qa.py
         (配置源)                  (代码生成器)              (最终运行脚本)
```

### 2.1 各文件职责

| 文件 | 角色 | 是否需要手动编辑 |
|------|------|----------------|
| `interactive_qa_config.yaml` | 配置源（YAML） | ✅ 是，修改配置的入口 |
| `interactive_qa_template.py` | 模板文件（含占位符 `{{KEY}}`） | ❌ 否，由生成器使用 |
| `gen_interactive_qa.py` | 代码生成器 | ❌ 否，运行它来生成脚本 |
| `interactive_qa.py` | 可执行脚本（由生成器产出） | ❌ 否，直接 `python` 运行 |

### 2.2 数据流

```
interactive_qa_config.yaml
        │
        ▼  python gen_interactive_qa.py
interactive_qa_template.py  ──占位符替换──►  interactive_qa.py
                                                 │
                                                 ▼  python interactive_qa.py
                                            用户输入问题
                                                 │
                                                 ▼
                              ┌──────────────────────────────┐
                              │ retrieval_fusion_eval.py       │
                              │  DimensionSearcher (维度检索)   │
                              │  SemanticSearcher (语义检索)    │
                              │  rrf_fuse_all (RRF 融合)       │
                              └──────────────────────────────┘
                                                 │
                                                 ▼
                              ┌──────────────────────────────┐
                              │ compare_fusion_methods.py     │
                              │  call_dashscope (LLM 调用)    │
                              └──────────────────────────────┘
                                                 │
                                                 ▼
                                          检索结果 / LLM 回答
```

---

## 3. 配置文件详解

`interactive_qa_config.yaml` 包含以下配置节：

### 3.1 LLM 配置 (llm)

```yaml
llm:
  api_key: ""  # DashScope API Key；请通过 DASHSCOPE_API_KEY 注入
  model: "qwen-plus"                               # 模型名称
  max_retries: 3                                   # 最大重试次数
  retry_sleep: 5                                   # 重试等待秒数
  min_interval: 0.1                               # API 调用最小间隔（秒，Rate Limiter）
```

### 3.2 检索配置 (retrieval)

```yaml
retrieval:
  mode: "fusion"      # fusion | dim | sem  检索模式
  top_k: 5             # 返回结果数量
  dim_alpha: 0.5       # 维度检索权重（fusion/dim 模式）
  sem_alpha: 0.5       # 语义检索权重（fusion 模式）
  rerank_mode: "score"  # 重排模式：score(得分融合) | interleaved(交替去重)
```

### 3.3 问答配置 (qa)

```yaml
qa:
  enable: true         # 是否开启 LLM 问答生成
  temperature: 0.1     # LLM 温度（创造性 vs 确定性）
  max_tokens: 512     # LLM 最大 Token 数
```

### 3.4 展示配置 (display)

```yaml
display:
  chars: 300          # 每条检索结果展示字符数
  verbose: false      # 详细模式（显示维度约束、得分分解）
```

### 3.5 数据库配置 (database)

```yaml
database:
  # Milvus Collection 名称（从 db_config.py 读取，此处可覆盖）
  # 留空表示从 db_config.py 自动读取
  collection_name: ""
```

---

## 4. 命令行接口

### 4.1 生成脚本

```bash
# 使用默认配置生成 interactive_qa.py
python gen_interactive_qa.py

# 指定配置文件
python gen_interactive_qa.py --config custom_config.yaml

# 预览生成结果（不写入文件）
python gen_interactive_qa.py --dry-run
```

### 4.2 运行问答

```bash
# 直接运行
python interactive_qa.py

# PowerShell 管道输入（UTF-8 编码）
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Get-Content query.txt | python interactive_qa.py

# 或设置环境变量
$env:PYTHONIOENCODING = "utf-8"
Get-Content query.txt | python interactive_qa.py
```

---

## 5. 交互命令参考

启动 `python interactive_qa.py` 后，在 `>>> ` 提示符下输入命令或问题。

### 5.1 检索模式

| 命令 | 说明 |
|------|------|
| `/mode fusion` | 切换为融合检索（默认，维度+语义 RRF） |
| `/mode dim` | 切换为维度检索（向量+标签 RRF） |
| `/mode sem` | 切换为语义检索（纯向量 ANN） |

### 5.2 重排模式

| 命令 | 说明 |
|------|------|
| `/rerank score` | 切换为得分重排（默认，根据排名赋分综合排序） |
| `/rerank inter` | 切换为交替去重（平衡两种检索结果） |

### 5.3 参数调整

| 命令 | 说明 | 示例 |
|------|------|------|
| `/k N` | 设置 Top-K | `/k 3` |
| `/alpha N` | 设置维度检索权重 alpha | `/alpha 0.3` |
| `/temp N` | 设置 LLM 温度 | `/temp 0.2` |
| `/max N` | 设置 LLM 最大 Token 数 | `/max 256` |
| `/limit N` | 设置每条结果展示字符数 | `/limit 500` |

### 5.4 功能开关

| 命令 | 说明 |
|------|------|
| `/qa` | 开启/关闭 LLM 问答生成 |
| `/verbose` | 开启/关闭详细模式（维度约束、得分分解） |

### 5.5 系统命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助信息 |
| `/exit` 或 `/quit` | 退出程序 |

### 5.6 普通问题

直接输入问题文本即可执行检索：

```
>>> 故宫的历史有多久
>>> 杭州有哪些著名景点
```

---

## 6. 生成流程

### 6.1 为什么需要代码生成器？

`interactive_qa.py` 的配置参数（API Key、模型、权重等）需要硬编码在 Python 文件中，以便运行时高效访问而不依赖外部文件。通过生成器机制：

1. **配置与代码分离** — 所有参数集中在 YAML 文件中，版本管理友好
2. **模板可复用** — 模板文件定义结构，占位符定义变量
3. **一键更新** — 修改 YAML 后运行 `gen_interactive_qa.py` 即可更新脚本

### 6.2 占位符对照表

| 模板占位符 | 生成后示例 | 来源字段 |
|-----------|----------|---------|
| `{{DS_API_KEY}}` | `"sk-40c..."` | `llm.api_key` |
| `{{DS_MODEL}}` | `"qwen-plus"` | `llm.model` |
| `{{DS_MAX_RETRIES}}` | `3` | `llm.max_retries` |
| `{{DS_RETRY_SLEEP}}` | `5` | `llm.retry_sleep` |
| `{{DS_MIN_INTERVAL}}` | `0.1` | `llm.min_interval` |
| `{{RETRIEVAL_MODE}}` | `"fusion"` | `retrieval.mode` |
| `{{RETRIEVAL_TOP_K}}` | `5` | `retrieval.top_k` |
| `{{DIM_ALPHA}}` | `0.5` | `retrieval.dim_alpha` |
| `{{SEM_ALPHA}}` | `0.5` | `retrieval.sem_alpha` |
| `{{RERANK_MODE}}` | `"score"` | `retrieval.rerank_mode` |
| `{{LLM_TEMPERATURE}}` | `0.1` | `qa.temperature` |
| `{{LLM_MAX_TOKENS}}` | `512` | `qa.max_tokens` |
| `{{ENABLE_QA}}` | `True` | `qa.enable` |
| `{{DISPLAY_CHARS}}` | `300` | `display.chars` |
| `{{VERBOSE}}` | `False` | `display.verbose` |
| `{{COLLECTION_NAME}}` | `None` | `database.collection_name` |

### 6.3 修改配置的推荐流程

```
1. 编辑 interactive_qa_config.yaml
2. 运行 python gen_interactive_qa.py
3. 运行 python interactive_qa.py
```

---

## 7. 依赖模块

`interactive_qa.py` 依赖以下模块，它们应位于同一项目目录：

| 模块 | 职责 |
|------|------|
| `retrieval_fusion_eval.py` | 提供 `DimensionSearcher`、`SemanticSearcher`、`rrf_fuse_all`、`SearchConfig`、`_load_bge_encoder` |
| `compare_fusion_methods.py` | 提供 `_RateLimiter`、`call_dashscope`（LLM 调用封装） |
| `db_config.py` | 提供数据库连接配置（Collection 名称回退读取） |

### 7.1 外部依赖

```bash
pip install dashscope pymilvus FlagEmbedding PyYAML
```

- **dashscope**：通义千问 API 调用
- **pymilvus**：向量数据库 Milvus 客户端
- **FlagEmbedding**：BGE-M3 向量编码器
- **PyYAML**：YAML 配置文件解析（仅 gen_interactive_qa.py 需要）

---

*文档生成时间: 2026-05-06*
