# Pipeline 工作流文档

> RAG DB 数据处理流水线 | 重构版 | 2026-04-17

---

## 目录

1. [概述](#1-概述)
2. [流水线架构](#2-流水线架构)
3. [各步骤详解](#3-各步骤详解)
4. [命令行接口](#4-命令行接口)
5. [数据流向](#5-数据流向)
6. [配置文件](#6-配置文件)
7. [中间产物](#7-中间产物)

---

## 1. 概述

### 1.1 目标

将原始文本数据经过智能分片、维度抽取与打标，最终存入 MySQL (RDB) + Milvus (VecDB) 的混合存储结构，支持维度感知的语义检索。

### 1.2 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| 流水线主控 | `pipeline.py` | 调度 5 个 Step，管理数据库连接 |
| 智能分片 | `integrated_chunker.py` | 三轮智能分片 (结构→PPL→IG) |
| LLM 服务 | `code/llm_service.py` | 通义千问 API 封装 |
| 配置管理 | `db_config.py` | MySQL / Milvus / Embedding 配置 |

### 1.3 环境依赖

```bash
pip install mysql-connector-python pymilvus FlagEmbedding scikit-learn scipy tqdm
```

---

## 2. 流水线架构

```
┌─────────────────────────────────────────────────────────────┐
│                     输入: 原始文本文件                        │
│                   (.txt / .md / .json)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 初始化数据库                                        │
│  ├─ MySQL MainIndex 表 (重建/创建)                          │
│  └─ Milvus Collection (Schema + 索引)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: 分片处理                                            │
│  └─ integrated_chunker.py                                   │
│      Round 1: 结构边界拆分 + LLM 推荐 l_min/l_max           │
│      Round 2: PPL 去噪 + 语义切分                          │
│      Round 3: IG 贪心融合 + 小块合并                         │
│  输出: output_chunks/all_chunks_chunks.json                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: 分片入库                                            │
│  ├─ 强制重建 MySQL MainIndex 表                             │
│  └─ 批量写入: doc_id, corpus_id, doc_text, profile_json     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: 维度抽取与打标                                      │
│  ├─ 4a 维度挖掘 (聚类采样 → LLM归纳 → 迭代优化)            │
│  ├─ 4b 添加维度列到 MySQL                                  │
│  ├─ 4c LLM 批量打标签 (断点续传)                          │
│  └─ 4d 标签回写 MySQL                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: 索引构建 + Milvus 迁移                             │
│  ├─ 5a 构建检索索引 (倒排索引 + 维度元数据 + 标签向量)      │
│  └─ 5b 全量迁移到 Milvus (含维度列)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     输出: 完整检索系统                         │
│  MySQL (维度标签) + Milvus (向量+维度) + 外部索引文件        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 各步骤详解

### Step 1: 初始化数据库

**目标：** 创建 MySQL 和 Milvus 的存储结构。

**1.1 MySQL**

- 数据库: `RAG_DB` (utf8mb4)
- 表: `MainIndex`
- 标准列:

| 列名 | 类型 | 说明 |
|------|------|------|
| `doc_id` | VARCHAR(255) PK | 文档唯一标识 |
| `corpus_id` | VARCHAR(255) | 语料库 ID |
| `doc_text` | LONGTEXT | 文档正文 |
| `profile_json` | JSON | 元数据 (file_name, genre, l_min, l_max, chunk_index) |
| `updated_at` | DATETIME | 更新时间 |

**1.2 Milvus**

- Collection: `corpus_chunks`
- Schema 字段 (固定部分):

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `chunk_id` | VARCHAR(256) PK | 主键 |
| `doc_id_link` | VARCHAR(256) | 关联 doc_id (partition_key) |
| `doc_title` | VARCHAR(1024) | 文档标题 |
| `chunk_gen_title` | VARCHAR(1024) | Chunk 标题 |
| `chunk_text` | VARCHAR(65535) | Chunk 正文 |
| `chunk_index` | INT64 | Chunk 序号 |
| `chunk_up_cid` | VARCHAR(256) | 上游 Chunk ID |
| `chunk_down_cid` | VARCHAR(256) | 下游 Chunk ID |
| `chunk_len` | INT64 | Chunk 长度 |
| `doc_title_vec` | FLOAT[dim] | 标题向量 |
| `chunk_title_vec` | FLOAT[dim] | Chunk 标题向量 |
| `chunk_text_vec` | FLOAT[dim] | 正文向量 |
| `genre` | VARCHAR(64) | 体裁类型 |
| `l_min` | INT64 | 分片下界 |
| `l_max` | INT64 | 分片上界 |

- 索引: 3 个向量字段分别建立 IVF_FLAT 索引 (COSINE 相似度)

---

### Step 2: 分片处理

**目标：** 将原始文本切分为语义完整的 chunks。

**调用命令：**

```bash
python integrated_chunker.py \
    --input ./data_input/test_data \
    --output ./output_chunks \
    --llm_api_key $DASHSCOPE_API_KEY \
    --llm_model qwen-plus
```

**三轮分片算法：**

| 轮次 | 算法 | 关键参数 |
|------|------|----------|
| Round 1 | 结构边界拆分 | 按 `=`/`#`/`---` 分隔符划分，LLM 推荐 l_min/l_max |
| Round 2 | PPL 去噪切分 | 困惑度过滤噪声句，阈值: mean + 3σ / mean + σ |
| Round 3 | IG 贪心融合 | 超大分片用信息增益贪心合并，过小分片向上合并 |

**输出文件：**

| 文件 | 说明 |
|------|------|
| `output_chunks/all_chunks_chunks.json` | 完整分片结果 |
| `output_chunks/all_chunks_summary.json` | 汇总统计 |
| `output_chunks/all_chunks_chunks.txt` | 纯文本分片 (可选) |

---

### Step 3: 分片入库

**目标：** 将分片结果持久化到 MySQL。

**行为：**
- **强制重建表**（DROP + CREATE），每次运行都会清空旧数据
- 批量逐条插入，每 100 条打印进度

**写入字段映射：**

| 分片字段 | MySQL 列 | 说明 |
|----------|----------|------|
| doc_id | `doc_id` | 格式: `{文件名}_sub_{序号:03d}` |
| 正文 | `doc_text` | LONGTEXT |
| 文件名 | `profile_json.file_name` | JSON 嵌套 |
| 体裁 | `profile_json.genre` | JSON 嵌套 |
| 分片参数 | `profile_json.l_min/l_max` | JSON 嵌套 |
| 序号 | `profile_json.chunk_index` | JSON 嵌套 |

---

### Step 4: 维度抽取与打标

**目标：** 从数据中自动发现维度结构，并打上标签。

**4a 维度挖掘**

```
数据加载 → BGE-M3 向量化 → TF-IDF/KMeans 聚类(50簇)
    → 每簇采样质心5个 + 边界5个 → ~500 篇代表性文档
    → LLM 归纳候选维度 (通义千问 qwen-plus)
    → 迭代优化 (最多3轮):
        覆盖率检查 < 20% → LLM 决策
        辨识度检查 (熵) < 1.0 → LLM 决策
        冗余检测 (向量相似度 + Jaccard) → LLM 决策
    → 输出: V_core.json (核心维度列表)
```

**4b 添加维度列**

```sql
ALTER TABLE MainIndex ADD COLUMN dim_维度名 TEXT NULL;
```

**4c 标签生成**

```python
# 断点续传: 已处理 doc_id 记录在 tags_output.json 中
for chunk in MySQL:
    if doc_id in 已处理: continue
    extracted = LLM.extract_batch_dimensions(text, dims)
    # 一次 LLM 调用抽多个维度，每文档最多 1500 字
    if not extracted:
        fallback = LLM.extract_keywords_fallback(text)
    tags_output[doc_id] = extracted or fallback
    if processed % 50 == 0: 保存
```

**4d 标签回写**

```sql
UPDATE MainIndex
SET dim_适宜人群 = '儿童,成人', dim_功效作用 = '退热,止痛'
WHERE doc_id = 'xxx';
```

---

### Step 5: 索引构建 + Milvus 迁移

**5a 构建检索索引**

| 索引文件 | 结构 | 用途 |
|----------|------|------|
| `inverted_index.json` | `{维度: {值: [doc_id...]}}` | 标签→文档倒排索引 |
| `dimension_metadata.json` | `{维度: {is_enum, value_count, values}}` | 维度元数据 |
| `tag_vectors.pkl` | `{维度: {values: [], vectors: np.array}}` | 开放维度软匹配向量 |

**5b 全量迁移到 Milvus**

```
加载 MySQL 全量数据
    → 读取 tags_output.json 关联维度标签
    → BGE-M3 三类向量编码 (doc_title_vec, chunk_title_vec, chunk_text_vec)
    → 构建 entities (固定字段 + 动态维度字段)
    → 批量 insert 到 Milvus Collection
```

---

## 4. 命令行接口

### 基本用法

```bash
# 单步执行
python pipeline.py --step 1
python pipeline.py --step 2 --input ./data_input/test_data
python pipeline.py --step 3
python pipeline.py --step 4
python pipeline.py --step 5

# 完整流程
python pipeline.py --all --input ./data_input/test_data

# 指定步骤范围
python pipeline.py --from_step 2 --to_step 4 --input ./data_input/test_data

# 强制重建数据库 (Step 1)
python pipeline.py --step 1 --force

# 跳过特定步骤
python pipeline.py --all --input ./data_input/test_data --skip_step2
python pipeline.py --all --input ./data_input/test_data --skip_step2 --skip_step4

# 指定语料库 ID
python pipeline.py --step 3 --corpus_id my_corpus_001

# 指定分片文件路径 (Step 3)
python pipeline.py --step 3 --chunks_file ./output_chunks/all_chunks_chunks.json
```

### 完整参数列表

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--step N` | 执行单个步骤 | - |
| `--from_step N` | 起始步骤 | - |
| `--to_step N` | 结束步骤 | - |
| `--all` | 执行完整流程 (Step 1-5) | False |
| `--skip_step1` | 跳过 Step 1 | False |
| `--skip_step2` | 跳过 Step 2 | False |
| `--skip_step3` | 跳过 Step 3 | False |
| `--skip_step4` | 跳过 Step 4 | False |
| `--skip_step5` | 跳过 Step 5 | False |
| `--input`, `-i` | 输入文件或目录 | "" |
| `--chunks_file` | 分片文件路径 | output_chunks/all_chunks_chunks.json |
| `--force`, `-f` | 强制重建数据库 | False |
| `--corpus_id` | 语料库 ID | "" |

### 跳步策略建议

| 场景 | 命令 |
|------|------|
| 分片已完成，只需重新入库 | `--all --skip_step2` |
| 维度已挖掘完成，只需重新打标 | `--all --skip_step2 --skip_step3` |
| 只想迁移到 Milvus | `--from_step 5 --to_step 5` |
| 分片+入库后，跳过已有维度 | `--from_step 4 --to_step 5 --skip_step4` |

---

## 5. 数据流向

### 5.1 输入

支持 3 种输入格式：

| 格式 | 处理方式 |
|------|----------|
| `.txt` 文件 | 直接读取文本内容 |
| `.md` 文件 | 直接读取 Markdown 内容 |
| `.json` 文件 | 解析 JSON 结构 |
| **目录** | 递归扫描目录下所有 .txt/.md/.json 文件 |

### 5.2 中间产物 (experiment_data/)

```
experiment_data/
├── V_cand.json              # 候选维度 (LLM 初始归纳)
├── V_core.json              # 核心维度 (迭代优化后)
├── tags_output.json         # 标签结果 {doc_id: {dim: [val1, val2]}}
├── inverted_index.json      # 倒排索引
├── dimension_metadata.json  # 维度元数据
└── tag_vectors.pkl          # 开放维度向量 (pickle)
```

### 5.3 最终存储

**MySQL MainIndex:**

```
doc_id | corpus_id | doc_text | profile_json | dim_适宜人群 | dim_功效作用 | ...
```

**Milvus Collection:**

```
chunk_id | doc_id_link | doc_title | chunk_text | doc_title_vec | chunk_text_vec | dim_适宜人群 | ...
```

---

## 6. 配置文件

通过 `db_config.py` 管理，支持 3 级优先级：环境变量 > 配置文件 > 代码默认值。

### 6.1 环境变量

```bash
export MYSQL_HOST=127.0.0.1
export MYSQL_PORT=3306
export MYSQL_USER=root
export MYSQL_PASSWORD=123456
export MYSQL_DATABASE=RAG_DB
export MILVUS_HOST=127.0.0.1
export MILVUS_PORT=19530
export VECDB_COLLECTION=corpus_chunks
export EMBEDDING_MODEL=bge-m3
export LLM_API_KEY=your_key
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=qwen-plus
export DASHSCOPE_API_KEY=your_key
```

### 6.2 配置文件 (db_config.yaml / db_config.json)

```yaml
rdb:
  host: 127.0.0.1
  port: 3306
  user: root
  password: 123456
  database: RAG_DB
  table: MainIndex

vecdb:
  host: 127.0.0.1
  port: 19530
  collection_name: corpus_chunks
  vector_dim: 1024
  index_type: IVF_FLAT
  metric_type: COSINE
  nlist: 1024

embedding:
  model_name: bge-m3
  batch_size: 64
  device: auto

llm:
  api_key: ""
  base_url: https://api.openai.com/v1
  model: qwen-plus
```

---

## 7. 中间产物

### 7.1 experiment_data/ 目录

| 文件 | 生命周期 | 用途 |
|------|----------|------|
| `V_cand.json` | Step 4a | LLM 归纳的候选维度，删除后重新生成 |
| `V_core.json` | Step 4a | 迭代优化后的核心维度，删除后重新生成 |
| `tags_output.json` | Step 4c | 标签结果，断点续传，删除后重新生成 |
| `inverted_index.json` | Step 5a | 倒排索引，删除后重新构建 |
| `dimension_metadata.json` | Step 5a | 维度元数据，删除后重新构建 |
| `tag_vectors.pkl` | Step 5a | 开放维度向量，删除后重新构建 |

### 7.2 output_chunks/ 目录

| 文件 | 生命周期 | 用途 |
|------|----------|------|
| `all_chunks_chunks.json` | Step 2 | 完整分片结果，Step 3 入库 |
| `all_chunks_summary.json` | Step 2 | 分片汇总统计 |

### 7.3 断点续传

Step 4c (标签生成) 支持断点续传：

```python
# 如果 tags_output.json 已存在，则只处理未完成的 doc_id
if tags_output_path.exists():
    with open(tags_output_path, "r") as f:
        results = json.load(f)
    processed_ids = set(results.keys())
```

---

## 附录: 关键算法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `K_CLUSTERS` | 50 | 聚类簇数 |
| `N_CORE_SAMPLES` | 5 | 每簇质心采样数 |
| `N_BOUND_SAMPLES` | 5 | 每簇边界采样数 |
| `TH_COV` | 0.20 | 覆盖率阈值 |
| `TH_DIS` | 1.0 | 辨识度(熵)阈值 |
| `TH_DIFF` | 0.3 | 冗余度阈值 (1 - min(sim_def, sim_data)) |
| `MAX_ITER` | 3 | 维度优化最大迭代轮数 |
| `batch_size` | 64 | Embedding 批处理大小 |
| `chunk_text_truncate` | 2000 | 迁移 Milvus 时正文截断长度 |

---

*文档生成时间: 2026-04-17*
