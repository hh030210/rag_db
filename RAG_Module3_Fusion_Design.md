# RAG Pipeline 与 code1 Module 3 融合方案

> 文档版本：2026-06-03
> 目标：将 `pipeline_qdrant.py` 的检索结果接入 `code1/chapter3` 的 Module 3 提示优化评测流程

---

## 一、现状梳理

### 1.1 pipeline_qdrant.py 流水线（已跑通）

```
测试数据 (14 md)
  │
  ▼ Step 1  初始化 MySQL + Qdrant
  │
  ▼ Step 2  integrated_chunker 分块 → 799 chunks
  │
  ▼ Step 3  coreference_resolver 指代消解
  │
  ▼ Step 4  批量写入 MySQL (MainIndex) — 799 条
  │
  ▼ Step 5  维度抽取 + 打标 — 51 维度 / 849 标签
  │
  ▼ Step 6  Qdrant 全量迁移 — 799 points (3 向量字段)
```

**最终存储状态：**

| 数据库 | 内容 |
|--------|------|
| MySQL `Main_index.MainIndex` | 799 rows，20 列（含51个 `dim_*` 维度列） |
| Qdrant `rag_chunks` | 799 points，3 向量字段 (doc_title_vec, chunk_title_vec, chunk_text_vec)，13 payload 字段 |
| `experiment_data/` | inverted_index.json, dimension_metadata.json, tag_vectors.pkl |

---

### 1.2 code1 Module 3 流程（提示迭代优化）

```
Step 1  Prompt 初始化
  │ 游客领域初始 prompt 模板
  ▼
Step 2  Prompt 迭代优化（LLM 多轮反馈）
  │ 每个问题迭代 3 轮优化
  ▼
Step 3  K-Means 聚类
  │ 问题按语义聚类，生成 cluster_centers
  ▼
Step 4  群智优化
  │ 每个 cluster 生成最优 prompt_module (role/task/constraints/template)
  ▼
Step 5  推理评测 ← 融合目标接入点
  │ 加载聚类中心 + 优化 prompts
  │ 对 tourist_eval.json 逐条推理
  │ 计算机器指标(BLEU/ROUGE/F1) + LLM 评分
```

**Step 5 关键数据依赖：**

| 字段 | 用途 |
|------|------|
| `id` | 问题唯一标识 |
| `question` | 问题文本 |
| `answer` | 参考答案（用于评测） |
| `source` | **预检索好的文档文本列表**，用于构建上下文 |
| `cluster_results.json` | Step 3 输出，聚类中心向量 |
| `optimized_prompts/cluster_*/` | Step 4 输出，每个聚类的最优 prompt |

---

## 二、核心鸿沟：数据格式不匹配

这是两个系统无法直接对接的根本原因。

### 2.1 Step 5 期望的输入格式（tourist_eval.json）

```python
# 来自 Tourist_step5_inference_multithread_v_new_ds.py 第 677-681 行
question_id  = generate_unique_id(item)    # str: id + hash
original_id  = str(item.get('id', ''))    # str: 原始ID
question     = item.get('question', '')    # str: 问题文本
reference_answer = item.get('answer', '')   # str: 参考答案
source       = item.get('source', [])      # List[str]: 预检索文档文本列表
```

### 2.2 interactive_qa.py 实际输出的检索结果

```python
# 来自 retrieval_fusion_eval.py do_search() 返回的 top_chunks
top_chunks = [
    {
        "chunk_id": str,       # 如 "西陵峡-景区介绍_000"
        "chunk_text": str,     # chunk 原始文本
        "doc_title": str,      # 所属文档标题
        "final_score": float,  # 融合得分
        "dim_score": float,    # 维度检索得分
        "sem_score": float,     # 语义检索得分
        "genre": str,          # 如 "景区介绍"
        "chunk_index": int,
        "chunk_len": int,
        # ... 其他 payload 字段
    },
    ...
]
```

### 2.3 字段映射表

| Step 5 期望字段 | interactive_qa.py 对应字段 | 转换逻辑 |
|----------------|--------------------------|---------|
| `id` | `chunk_id` | 直接映射 |
| `question` | `chunk_text` | **核心差异**：Step 5 把 chunk 当"问题"，interactive_qa 把 chunk 当"上下文" |
| `answer` | — | 需要额外处理（LLM 生成或留空） |
| `source` | `[c["chunk_text"] for c in top_chunks]` | 取 Top-K chunk 文本列表 |

### 2.4 设计决策：语义反转问题

> **关键洞察：** Module 3 的 `tourist_eval.json` 中，每个 entry 是一个**问答对**，而我们的 pipeline 里每个 entry 是一个**知识 chunk**。

两种融合路径的语义差异：

| 路径 | 语义 | 适合场景 |
|------|------|---------|
| **A: Chunk 即问题** | 每个 chunk 的文本作为"question"，source 取 top_chunks 作为"上下文" | 把 Module 3 当作"chunk 自检+优化"工具 |
| **B: 真实问题集** | 先收集一批用户真实问题 → interactive_qa 检索 → 接入 Module 3 推理 | 端到端评测，真实评测场景 |

**推荐路径 B**，因为：
- 路径 A 语义扭曲，Module 3 的聚类中心是基于"问答对"训练的，套在 chunk 上无意义
- 路径 B 符合 Module 3 原设计意图：评测真实问题的回答质量
- 用户最终需要的是一个"用户问真实问题 → pipeline 检索 → Module 3 优化回答"的完整链路

---

## 三、融合方案：三条路径

### 路径 A — 适配层脚本（推荐）

#### 3.1 核心思想

新建 `rag_fusion_pipeline.py`，作为两个系统的**桥梁**：

```
真实问题列表 (JSON)
  │
  ▼ Step 1: interactive_qa 检索
  top_chunks (chunk_id, chunk_text, final_score, ...)
  │
  ▼ Step 2: 适配器转换
  items = [{question, answer?, source: [chunk_text x top_k]}]
  │
  ├─ 加载 cluster_centers (Step 3 产物)
  ├─ 加载 optimized_prompts (Step 4 产物)
  │
  ▼ Step 3: 找最相似聚类
  best_cluster_id, cluster_sim
  │
  ▼ Step 4: 构建 Prompt
  build_full_prompt(prompt_module, question, context)
  │
  ▼ Step 5: LLM 推理 + 评测
  生成答案 + BLEU/ROUGE/F1 + LLM 评分
```

#### 3.2 适配器实现

```python
# rag_fusion_pipeline.py 核心片段

def adapt_top_chunks_to_step5(top_chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    将 interactive_qa 的 top_chunks 转换为 Step 5 期望的格式。

    关键：Module 3 是评测"真实问题"，
    这里用 top_chunks 中的 chunk_text 作为 source（上下文），
    用户问题来自外部问题集。
    """
    items = []
    for chunk in top_chunks[:top_k]:
        items.append({
            "id": chunk.get("chunk_id", ""),
            "question": chunk.get("chunk_text", ""),
            "answer": "",  # 评测时由外部提供
            "source": [c.get("chunk_text", "") for c in top_chunks],
            "doc_title": chunk.get("doc_title", ""),
            "genre": chunk.get("genre", ""),
            # 保留原始检索得分
            "final_score": chunk.get("final_score", 0.0),
            "dim_score": chunk.get("dim_score", 0.0),
            "sem_score": chunk.get("sem_score", 0.0),
        })
    return items


class RagFusionPipeline:
    """
    融合 pipeline_qdrant 的检索能力 + code1 Module 3 的提示优化能力
    """

    def __init__(self, config: dict):
        # 1. 初始化检索器（复用 retrieval_fusion_eval.py）
        self.dim_searcher = DimensionSearcher(...)
        self.sem_searcher = SemanticSearcher(...)
        self.bge_encoder = _load_bge_encoder(...)

        # 2. 加载 Module 3 产物（Step 3/4 产出）
        self.cluster_centers = self._load_cluster_centers()
        self.optimized_prompts = self._load_optimized_prompts()

        # 3. LLM 客户端
        self.llm_client = LLMClient(...)

    def _load_cluster_centers(self) -> Dict[int, np.ndarray]:
        cluster_file = Path("code1/.../clustering_results/tourist/cluster_results.json")
        with open(cluster_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {
            c['cluster_id']: np.array(c['center'])
            for c in data.get('clusters', [])
        }

    def _load_optimized_prompts(self) -> Dict[int, List[Dict]]:
        prompts = {}
        for cluster_dir in Path("code1/.../optimized_prompts/tourist").glob("cluster_*"):
            cluster_id = int(cluster_dir.name.replace("cluster_", ""))
            prompts[cluster_id] = []
            for pf in cluster_dir.glob("optimized_prompt_*.json"):
                with open(pf, 'r', encoding='utf-8') as f:
                    prompts[cluster_id].append(json.load(f))
        return prompts

    def find_best_cluster(self, question: str) -> Tuple[int, float]:
        """用 BGE 编码问题，找最相似聚类"""
        q_emb = self.bge_encoder.encode([question])
        best_id, best_sim = None, -1
        for cid, center in self.cluster_centers.items():
            sim = np.dot(q_emb, center) / (np.linalg.norm(q_emb) * np.linalg.norm(center))
            if sim > best_sim:
                best_sim, best_id = sim, cid
        return best_id, best_sim

    def build_full_prompt(self, prompt_module: Dict, question: str, context: str) -> str:
        """复用 Step 5 的 prompt 构建逻辑"""
        role      = prompt_module.get('role', '')
        task      = prompt_module.get('task', '')
        output_fmt = prompt_module.get('output_format', '')
        constraints = prompt_module.get('constraints', [])
        template = prompt_module.get('prompt_template',
            "基于以下上下文回答问题：\n\n上下文：\n{context}\n\n问题：{question}\n\n回答：")
        c_text = "\n".join(f"- {c}" for c in constraints)
        return f"{role}\n\n{task}\n\n{output_fmt}\n\n约束条件：\n{c_text}\n\n"
               f"基于以下信息回答问题：\n\n上下文信息：\n{context}\n\n问题：{question}\n\n请提供准确、详细的回答。"

    def run(self, question: str, top_k: int = 5) -> Dict:
        """
        端到端：检索 → 找聚类 → 优化 Prompt → LLM 推理
        """
        # Step A: 检索（复用 interactive_qa 逻辑）
        top_chunks = self._retrieve(question, top_k)

        # Step B: 构建上下文
        source = [c['chunk_text'] for c in top_chunks]
        context = "\n".join(source[:top_k])

        # Step C: 找最相似聚类 + 获取优化 prompt
        best_cluster_id, cluster_sim = self.find_best_cluster(question)
        if best_cluster_id is None:
            return {"error": "No matching cluster"}

        prompts = self.optimized_prompts.get(best_cluster_id, [])
        if not prompts:
            return {"error": f"No prompt for cluster {best_cluster_id}"}

        prompt_module = prompts[0].get('prompt_module', {})
        full_prompt = self.build_full_prompt(prompt_module, question, context)

        # Step D: LLM 推理
        answer, success = self.llm_client.generate(full_prompt, temperature=0.3)

        return {
            "question": question,
            "context": context,
            "top_chunks": top_chunks,
            "matched_cluster": best_cluster_id,
            "cluster_similarity": cluster_sim,
            "used_prompt": prompts[0],
            "answer": answer,
            "success": success,
        }
```

#### 3.3 关键依赖复用

| 依赖文件 | 复用内容 |
|---------|---------|
| `retrieval_fusion_eval.py` | `DimensionSearcher`, `SemanticSearcher`, `rrf_fuse_all`, `_load_bge_encoder` |
| `compare_fusion_methods.py` | `call_dashscope`, `_RateLimiter` |
| `db_config.yaml` / `db_config.py` | 数据库连接配置 |
| `code1/.../clustering_results/` | Step 3 聚类中心向量 |
| `code1/.../optimized_prompts/` | Step 4 优化后的 prompt 模块 |

#### 3.4 输出格式

```json
{
  "question": "颐和园的历史有多久？",
  "answer": "颐和园始建于清朝乾隆年间...",
  "context": "颐和园是中国现存规模最大的皇家园林...",
  "matched_cluster": 3,
  "cluster_similarity": 0.847,
  "used_prompt": {
    "prompt_id": 1,
    "cluster_category": "景区历史",
    "prompt_module": {
      "role": "你是专业的旅游咨询助手...",
      "task": "准确回答关于景区历史的问题...",
      "constraints": ["只基于提供的上下文", "使用中文回答"]
    }
  },
  "top_chunks": [...],
  "success": true
}
```

---

### 路径 B — 改造 interactive_qa.py（集成 Module 3）

#### 3.5 核心思想

直接在 `interactive_qa.py` 中增加"提示优化"模式，在已有检索结果基础上，用 Module 3 的优化 prompt 重新构建 LLM 问答。

#### 3.6 改动点

在 `interactive_qa.py` 的 `do_qa()` 方法中，替换默认系统 prompt：

```python
# 原有：使用固定系统 prompt
qa_prompt = f"{SYSTEM_PROMPT}\n\n上下文：\n{context}\n\n问题：{query}\n\n回答："

# 改为：加载 Module 3 优化后的 prompt
best_cluster_id, cluster_sim = fusion_pipeline.find_best_cluster(query)
prompt_module = optimized_prompts[best_cluster_id][0]['prompt_module']
qa_prompt = fusion_pipeline.build_full_prompt(prompt_module, query, context)
```

#### 3.7 优缺点

| 优点 | 缺点 |
|------|------|
| 改动集中在一处 | interactive_qa 是交互式工具，引入 Module 3 增加了启动时的模型加载开销 |
| 用户可以随时在"固定 Prompt 模式"和"优化 Prompt 模式"间切换 | 依赖 code1 的聚类结果，耦合较紧 |
| 实时体验最佳提示优化的效果 | 需要处理 Module 3 和 interactive_qa 的模型（BGE）共用问题 |

---

### 路径 C — 替换 retrieve_context（最小改动）

#### 3.8 核心思想

**不新建文件，不改 interactive_qa**，只修改 `Tourist_step5_inference_multithread_v_new_ds.py` 中的 `retrieve_context()` 方法，使其从 Qdrant/MySQL 实时检索而不是读静态 `source` 字段。

```python
# 改动：替换 Tourist_step5_inference_multithread_v_new_ds.py 第 500 行

def retrieve_context(self, question: str, source: List[str]) -> Tuple[str, List[Dict]]:
    # 原有逻辑（注释掉）：
    # contexts = source[:self.retrieval_k] if len(source) > self.retrieval_k else source
    # retrieved_contexts = [{'rank': i+1, 'text': text} for i, text in enumerate(contexts)]
    # return "\n".join(contexts), retrieved_contexts

    # 新逻辑：从 Qdrant 实时检索
    from qdrant_client import QdrantClient
    client = QdrantClient(url="http://127.0.0.1:6333")
    results = client.search(
        collection_name="rag_chunks",
        query_vector=self.embedder.encode([question])[0],
        limit=self.retrieval_k
    )
    retrieved_contexts = []
    for r in results:
        retrieved_contexts.append({
            'rank': r.id,
            'text': r.payload.get('chunk_text', ''),
            'score': r.score,
            'doc_title': r.payload.get('doc_title', ''),
        })
    context = "\n".join([c['text'] for c in retrieved_contexts])
    return context, retrieved_contexts
```

#### 3.9 优缺点

| 优点 | 缺点 |
|------|------|
| 改动量最小 | 只解决了 retrieval 部分，Step 3/4 的聚类和优化 prompts 仍然依赖原 tourist 数据 |
| 复用已有 pipeline 的检索能力 | `tourist_eval.json` 仍然需要（提供 question + answer 字段） |
| | 语义上仍然是"chunk 作为上下文"，未解决根本设计问题 |

---

## 四、推荐实施顺序

```
Phase 1: 路径 A — 适配层脚本（1-2天）
  ├─ 新建 rag_fusion_pipeline.py
  ├─ 复用 retrieval_fusion_eval.py 的检索逻辑
  ├─ 复用 Tourist_step5 的 prompt 构建逻辑
  ├─ 复用 Step 3/4 的聚类中心 + 优化 prompts
  └─ 输出标准化 JSON 结果

Phase 2: 路径 B — 集成到 interactive_qa（1天）
  ├─ 在 interactive_qa_config.yaml 增加"prompt_mode: module3"
  ├─ 条件加载 Module 3 资源
  └─ 用户可通过 /prompt module3 切换

Phase 3: 路径 C — 最小化改造（0.5天，可选）
  └─ 替换 retrieve_context，实时检索替代静态 source
```

---

## 五、数据流全景图

```
┌──────────────────────────────────────────────────────────────────┐
│                     pipeline_qdrant.py (已跑通)                    │
│                                                                   │
│  测试数据 → 分块 → 指代消解 → MySQL → 维度打标 → Qdrant迁移        │
│                                           │                       │
│                                   ┌───────┴───────┐               │
│                                   │  MySQL        │  Qdrant       │
│                                   │  MainIndex    │  rag_chunks   │
│                                   └───────────────┘               │
└──────────────────────────────────────────────────────────────────┘
                                        │
                               ┌────────┴────────┐
                               ▼                 ▼
                    ┌────────────────┐  ┌──────────────────┐
                    │  DimensionSearcher │  │ SemanticSearcher  │
                    │  (维度+标签检索)   │  │ (向量 ANN 检索)   │
                    └────────┬────────┘  └────────┬─────────┘
                             │                    │
                             └────────┬───────────┘
                                      ▼
                             ┌────────────────┐
                             │ rrf_fuse_all   │  retrieval_fusion_eval.py
                             │ (RRF 融合)     │
                             └────────┬───────┘
                                      │
                                      ▼
                             top_chunks (chunk_text, final_score, ...)
                                      │
                              ┌───────┴────────┐
                              │                │
                              ▼                ▼
               ┌──────────────────┐    ┌──────────────────┐
               │   路径 A 适配层    │    │   路径 B 集成    │
               │ rag_fusion_pipeline│    │ interactive_qa   │
               │                  │    │ + Module 3 Prompt │
               └────────┬─────────┘    └────────┬──────────┘
                        │                      │
                        └──────────┬───────────┘
                                   ▼
                    ┌───────────────────────────┐
                    │   code1 Module 3 资源      │
                    │  cluster_centers           │  (Step 3 产物)
                    │  optimized_prompts/        │  (Step 4 产物)
                    └──────────┬──────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                      ▼
          find_best_cluster         build_full_prompt
          (向量相似度)              (prompt_module 构建)
                    │                      │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │    LLMClient         │
                    │  call_dashscope      │
                    │  temperature=0.3     │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  生成答案 + 评测指标   │
                    │  BLEU/ROUGE/F1/LLM评分 │
                    └──────────────────────┘
```

---

## 六、待确认事项

1. **真实问题集来源**：路径 A/B 需要一批评测问题（question + answer）。是用 pipeline 里已有的 chunk 自生成，还是外部导入？
2. **Step 3/4 产物路径**：`code1/.../clustering_results/` 和 `optimized_prompts/` 的实际路径是什么？（目前代码中是硬编码 `i:\bylw_final\...`，需要适配到 `D:\RAG_DB_slim\code1`）
3. **LLM 客户端统一**：Step 5 用 SiliconFlow API (`sk-ahwf...`)，interactive_qa 用 DashScope API (`sk-40c...`)，是否需要统一？
4. **评测模式**：是只需要 LLM 生成答案（路径 A），还是需要完整的 BLEU/ROUGE 评测指标（路径 B）？

---

## 七、文件清单

| 文件 | 角色 | 状态 |
|------|------|------|
| `pipeline_qdrant.py` | 数据处理流水线 | ✅ 已跑通 |
| `interactive_qa.py` | 交互式检索问答 | ✅ 已跑通 |
| `retrieval_fusion_eval.py` | 检索核心（Dimension + Semantic + RRF） | ✅ 已跑通 |
| `compare_fusion_methods.py` | LLM 调用封装 | ✅ 已就绪 |
| `coreference_resolver.py` | 指代消解 | ✅ 已跑通 |
| `code1/.../Tourist_step5_inference_*.py` | Module 3 推理评测 | 🔲 待集成 |
| `code1/.../clustering_results/` | Step 3 聚类中心 | 🔲 待接入 |
| `code1/.../optimized_prompts/` | Step 4 优化 prompts | 🔲 待接入 |
| `rag_fusion_pipeline.py` | **新增：融合适配层** | 🔲 待实现 |
