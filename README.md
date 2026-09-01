# RAG_DB_slim

> 实验结果和 `code1` 三章主线代码的目录说明见 [EXPERIMENT_RESULTS_INDEX.md](EXPERIMENT_RESULTS_INDEX.md) 与 [code1/README.md](code1/README.md)。

> 混合检索增强生成系统 | 维度感知 + 语义融合 + 提示优化

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 系统架构](#2-系统架构)
- [3. 核心模块](#3-核心模块)
- [4. 数据处理流水线](#4-数据处理流水线)
- [5. 检索引擎](#5-检索引擎)
- [6. 交互问答](#6-交互问答)
- [7. 融合评测](#7-融合评测)
- [8. Module 3 提示优化](#8-module-3-提示优化)
- [9. Web 服务（fusion_app）](#9-web-服务fusion_app)
- [10. 快速开始](#10-快速开始)
- [11. 项目结构](#11-项目结构)
- [12]依赖与环境

---

## 1. 项目概述

**RAG_DB_slim** 是一个混合 RAG（Retrieval-Augmented Generation）系统，将文档处理流水线与多模式检索引擎结合，实现结构化知识库的高质量问答。

### 核心能力

| 能力 | 说明 |
|------|------|
| **混合存储** | MySQL（维度元数据）+ Qdrant/Milvus（向量索引）双库架构 |
| **维度感知检索** | 从非结构化文档中自动挖掘结构化维度（景区、历史、地理等），支持精确过滤 |
| **语义向量检索** | BGE-M3 多语言 Embedding，支持 1024 维稠密向量 |
| **RRF 融合排序** | Reciprocal Rank Fusion 将维度结果与语义结果按排名融合 |
| **指代消解** | 处理分片文本中的代词/指示词消解，保证上下文一致性 |
| **提示优化（Module 3）** | 基于聚类的协同优化，为不同问题类型生成专用 Prompt |

### 数据域

系统当前处理**旅游景区领域**（颐和园、西湖、龙门石窟、明十三陵、张家界、少林寺、南孔庙等景区），涵盖景区介绍与运营信息两类文档。

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                          用户查询                                 │
│                      "颐和园的历史有多久？"                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    fusion_app  Web 服务                          │
│              (检索引擎 / 问答生成 / 流水线管理)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│   语义检索       │  │   维度检索       │  │   LLM 生成答案       │
│ Qdrant HNSW    │  │ 倒排索引匹配    │  │  SiliconFlow Qwen   │
│ chunk_text_vec │  │ 29 个维度标签   │  │  通义千问            │
└─────────────────┘  └─────────────────┘  └─────────────────────┘
          │                   │                   │
          └───────── RRF 融合 ──────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Top-K Chunks  │
                    │  + 最终答案      │
                    └─────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                     数据处理流水线                                 │
│  原始文档 → 分片 → 指代消解 → MySQL 写入 → 维度抽取 → Qdrant 迁移  │
└─────────────────────────────────────────────────────────────────┘
```

### 双数据库架构

| 数据库 | 用途 | 数据内容 |
|--------|------|----------|
| **MySQL** | 维度元数据存储 | Chunk 基础信息 + 51 个维度列（dim_景区、dim_历史等） |
| **Qdrant** | 向量检索引擎 | 799 个 chunk 点，每个 3 个命名向量字段 |

---

## 3. 核心模块

### 3.1 配置文件

| 文件 | 说明 |
|------|------|
| `db_config.yaml` | 主配置：MySQL、Qdrant、BGE-M3、LLM API |
| `db_config.py` | 配置加载器，支持 `${VAR}` 和 `${VAR:-default}` 环境变量语法 |
| `fusion_app/config.yaml` | fusion_app 专用配置，合并了 RAG_DB_slim 与 code1 的能力 |

### 3.2 数据模型（schemas.py）

- `RetrievalMode` — 检索模式枚举：`semantic` | `dimension` | `fusion`
- `SearchRequest` / `SearchResponse` — 检索 API 请求/响应
- `ChatRequest` / `ChatResponse` — 问答 API 请求/响应
- `PipelineRunRequest` — 流水线执行请求

---

## 4. 数据处理流水线

**入口：** `pipeline_qdrant.py`（Qdrant 版）或 `pipeline.py`（Milvus 版）

### 六步流程

```
原始文档 (.txt/.md)
        │
        ▼
┌───────────────────┐
│ Step 1: 初始化数据库  │
│ MySQL 表 + Qdrant  │
│ Collection（含 3 个  │
│ 命名向量字段）        │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Step 2: 智能分片     │
│ integrated_chunker │
│ 三轮分片策略：       │
│ 结构感知→PPL→IG    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Step 3: 指代消解    │
│ coreference_       │
│ resolver.py        │
│ 规则 + LLM 验证    │
│ 一致性检查          │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Step 4: 写入 MySQL │
│ MainIndex 表       │
│ profile_json 列    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Step 5: 维度抽取    │
│ DimensionMiner    │
│ LLM 归纳 29 个维度  │
│ 标签生成            │
│ 维度→MySQL 列       │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Step 6: Qdrant 迁移 │
│ 构建倒排索引         │
│ inverted_index.json│
│ BGE-M3 编码 3 向量  │
│ HNSW 索引          │
└───────────────────┘
        │
        ▼
  输出 chunks (799 个)
```

### 维度系统

系统从文档中自动挖掘了 **29 个结构化维度**，包括：

| 维度类别 | 示例维度 |
|----------|----------|
| 景区信息 | 景区、历史、时间、地理、规模 |
| 建筑设施 | 殿堂、楼阁、廊榭、假山叠石、湖光山色 |
| 人物事件 | 人物身份、历史事件、历史人物、相关人物 |
| 学科领域 | 哲学思想、文学艺术、建筑学 |

维度以 `dim_景区`、`dim_历史` 等列存储在 MySQL，并建立倒排索引（`inverted_index.json`）供检索使用。

---

## 5. 检索引擎

### 5.1 三种检索模式

| 模式 | 说明 | 算法 |
|------|------|------|
| `semantic` | 纯语义向量检索 | Qdrant HNSW ANN 搜索，cosine 相似度 |
| `dimension` | 纯维度标签匹配 | 倒排索引交集，精确过滤 |
| `fusion` | 维度 + 语义融合（默认） | RRF（Reciprocal Rank Fusion） |

### 5.2 RRF 融合算法

```
RRF_score(chunk) = α × (1 / (k + dim_rank)) + (1-α) × (1 / (k + sem_rank))
```

- `k = 60`（RRF 平滑参数）
- `α ∈ [0, 1]`：`α=0` 纯语义，`α=1` 纯维度
- 默认 `α = 0.2`（偏重语义）

### 5.3 Qdrant 向量字段

| 向量字段 | 来源 | 维度 |
|----------|------|------|
| `chunk_text_vec` | Chunk 正文 | 1024 |
| `doc_title_vec` | 所属文档标题 | 1024 |
| `chunk_title_vec` | Chunk 小标题 | 1024 |

### 5.4 核心文件

| 文件 | 职责 |
|------|------|
| `retrieval_fusion_eval.py` | 检索器类（`DimensionSearcher`、`SemanticSearcher`）、RRF 融合、配置 |
| `compare_fusion_methods.py` | 对比不同融合策略（RRF vs 加权分数）的评测脚本 |
| `fusion_app/app/services/retrieval_engine.py` | fusion_app 的检索引擎封装，支持降级方案 |

---

## 6. 交互问答

**入口：** `interactive_qa.py`（由 `gen_interactive_qa.py` 从模板生成）

### 交互命令

| 命令 | 说明 |
|------|------|
| `/mode fusion\|dim\|sem` | 切换检索模式 |
| `/rerank score\|inter` | 切换重排模式 |
| `/k N` | 设置 Top-K（默认 5） |
| `/alpha N` | 设置维度权重（默认 0.2） |
| `/temp N` | LLM 温度参数 |
| `/max N` | LLM 最大 Token 数 |
| `/qa` | 开启/关闭问答生成 |
| `/verbose` | 显示维度约束和得分分解 |
| `/help` | 显示帮助 |

### 使用方式

```bash
# 交互模式（管道输入）
Get-Content query.txt | python interactive_qa.py

# 单条问答
python interactive_qa.py --question "颐和园的历史有多久？"
```

---

## 7. 融合评测

**入口：** `retrieval_fusion_eval.py`

提供 `DimensionSearcher` 和 `SemanticSearcher` 两个检索器类，支持：

- 独立检索模式（`sem_only`、`dim_only`）
- 融合模式（`rrf`、`score`）
- 可配置 `alpha` 权重

---

## 8. Module 3 提示优化

**目录：** `code1/chapter3/codes/bylw_rag/new_experiments/`

Module 3 是在 RAG 检索之后、LLM 生成之前的**提示优化层**，包含：

### 五步流水线

| Step | 文件 | 产物 |
|------|------|------|
| Step 1 | 数据准备 | `tourist_train.json`、`tourist_eval.json`（问答对） |
| Step 2 | Prompt 迭代 | 初始 Prompt → LLM 评估 → 改进 |
| Step 3 | K-Means 聚类 | 8 个问题类型聚类，生成聚类中心向量 |
| Step 4 | 协同优化 | 每个聚类生成专用优化 Prompt |
| Step 5 | 推理评测 | 批量推理 + BLEU/ROUGE/LLM 评分 |

### 产物目录

```
code1/chapter3/codes/bylw_rag/new_experiments/
├── clustering_results/tourist/     # 聚类中心向量（cluster_results.json）
├── optimized_prompts/tourist/      # 优化后的 Prompts
├── datas/tourist/
│   ├── tourist_train.json          # 训练问答对
│   └── tourist_eval.json           # 评测问答对
└── results/                        # 推理结果
```

---

## 9. Web 服务（fusion_app）

**目录：** `fusion_app/`

FastAPI Web 服务，整合检索引擎、MySQL 存储、Embedding 模型和 LLM 生成。

### 服务架构

```
fusion_app/
├── app/
│   ├── models/schemas.py           # Pydantic 数据模型
│   ├── services/
│   │   └── retrieval_engine.py     # 检索融合引擎（RRF）
│   └── core/
│       ├── mysql_service.py        # MySQL CRUD
│       ├── embedding_service.py     # BGE-M3 / BGE-large 编码
│       └── qdrant_client.py         # Qdrant 连接
├── config.py                       # 配置管理（YAML + 环境变量）
├── config.yaml                     # 服务配置
└── docker-entrypoint.sh            # 容器启动脚本
```

### 启动方式

```bash
# Docker Compose
docker-compose up -d fusion_app

# 或直接运行
cd fusion_app
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API 端点（待实现/扩展）

| 端点 | 方法 | 说明 |
|------|------|------|
| `/search` | POST | 检索接口（支持 fusion/semantic/dimension 模式） |
| `/chat` | POST | 问答接口 |
| `/pipeline/run` | POST | 触发数据处理流水线 |
| `/db/init` | POST | 初始化数据库 |

---

## 10. 快速开始

### 10.1 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Qdrant（Docker）
docker-compose -f docker-compose-qdrant.yml up -d

# 确认 Qdrant 就绪
curl http://localhost:6333/readyz
```

### 10.2 数据处理（一次性）

```bash
# 准备原始文档到 data_input/ 目录

# 完整流水线（6 步）
python pipeline_qdrant.py --all --input ./data_input

# 或分步执行
python pipeline_qdrant.py --from_step 1 --to_step 6 --input ./data_input
```

### 10.3 检索问答

```bash
# 交互模式
python interactive_qa.py

# 单条测试（融合检索，默认 alpha=0.2）
python rag_fusion_pipeline.py --mode single --question "颐和园的历史有多久？"

# 纯语义对比
python rag_fusion_pipeline.py --mode single --question "颐和园的历史有多久？" \
    --retrieval-mode semantic

# 批量评测
python rag_fusion_pipeline.py --mode eval --limit 20 \
    --retrieval-mode fusion --alpha 0.2 --output fusion_results/my_test
```

### 10.4 Web 服务

```bash
# 启动 fusion_app
cd fusion_app
uvicorn app.main:app --reload --port 8000
```

---

## 11. 项目结构

```
RAG_DB_slim/
├── # ===== 数据处理 =====
├── pipeline_qdrant.py          # 6步流水线（Qdrant 版）
├── pipeline.py                 # 6步流水线（Milvus 版）
├── coreference_resolver.py     # 指代消解模块
│
├── # ===== 检索与问答 =====
├── interactive_qa.py           # 交互式问答（CLI）
├── rag_fusion_pipeline.py      # 融合 Pipeline + Module 3 集成
├── retrieval_fusion_eval.py     # 检索器（维度/语义/融合）
├── compare_fusion_methods.py    # 融合策略对比评测
│
├── # ===== 核心服务（fusion_app）=====
├── fusion_app/
│   ├── app/
│   │   ├── models/schemas.py   # Pydantic 数据模型
│   │   ├── services/
│   │   │   └── retrieval_engine.py  # 检索融合引擎
│   │   └── core/
│   │       ├── mysql_service.py     # MySQL 服务
│   │       ├── embedding_service.py # Embedding 服务
│   │       └── qdrant_client.py     # Qdrant 客户端
│   ├── config.py               # 配置管理
│   └── config.yaml             # 服务配置
│
├── # ===== Module 3 提示优化（code1）=====
├── code1/chapter3/codes/bylw_rag/new_experiments/
│   ├── Tourist_step1_data_preparation.py
│   ├── Tourist_step2_prompt_iteration.py
│   ├── Tourist_step3_kmeans_clustering.py  # 聚类中心
│   ├── Tourist_step4_collective_optimization.py  # 优化 Prompts
│   └── Tourist_step5_inference_multithread_v_new_ds.py  # 推理评测
│
├── # ===== 索引产物 =====
├── experiment_data/
│   ├── inverted_index.json     # 维度倒排索引（29 个维度）
│   ├── dimension_metadata.json  # 维度元数据
│   └── tags_output.json         # 标签输出
│
├── # ===== 配置 =====
├── db_config.yaml              # 主配置
├── db_config.py                # 配置加载器
├── fusion_app/config.yaml      # fusion_app 配置
│
├── # ===== 容器化 =====
├── Dockerfile                  # Python 3.10-slim 镜像
├── docker-compose.yml          # 完整服务编排
├── docker-compose-qdrant.yml   # 独立 Qdrant 服务
└── docker-entrypoint.sh        # 容器启动脚本
│
├── # ===== 文档 =====
├── README.md                   # 本文件
├── RAG_Module3_Fusion_Design.md  # Module 3 融合设计文档
├── pipeline_usage_guide.md     # 流水线使用指南
└── interactive_qa_guide.md    # 交互问答指南
```

---

## 12. 依赖与环境

### Python 版本

**Python 3.10+**

### 核心依赖

```
mysql-connector-python    # MySQL 连接
pymilvus                  # Milvus 向量库（备用）
qdrant-client             # Qdrant 向量库
FlagEmbedding             # BGE-M3 编码器
scikit-learn              # K-Means 聚类
scipy                     # 科学计算
PyYAML                    # 配置管理
dashscope                 # 通义千问 API（DashScope）
sentence-transformers     # Sentence-BERT
numpy                     # 数值计算
tqdm                      # 进度条
pydantic                  # 数据验证（fusion_app）
fastapi                   # Web 框架（fusion_app）
uvicorn                   # ASGI 服务器（fusion_app）
```

### 模型

| 模型 | 路径 | 用途 |
|------|------|------|
| BGE-M3 | `model/bge-m3/` | Chunk 向量编码 |
| BGE-large-zh-v1.5 | `code1/models/embedding/bge-large-zh-v1.5/` | code1 检索编码 |
| Qwen2.5-7B-Instruct | SiliconFlow API | LLM 生成（fusion_app） |
| Qwen-plus | DashScope API | LLM 生成（rag_fusion_pipeline） |

### API 密钥

| 服务 | 配置项 |
|------|--------|
| DashScope | `DS_API_KEY` 环境变量或 `db_config.yaml` |
| SiliconFlow | `SILICONFLOW_API_KEY` 或 `fusion_app/config.yaml` |

---

## 附录：检索模式对比

| 模式 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| `semantic` | 语义模糊的开放式问题 | 覆盖广、召回高 | 可能偏离精确维度 |
| `dimension` | 维度明确的结构化查询 | 精确过滤、解释性强 | 依赖维度标签质量 |
| `fusion` | 通用场景（默认推荐） | 平衡精确性与覆盖率 | 需要调优 alpha |

**调参建议：** `alpha=0.2` 偏重语义，`alpha=0.5` 均衡，`alpha=0.8` 偏重维度。可通过 `--alpha` 参数在命令行快速实验。
