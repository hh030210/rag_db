# RAG 检索服务 API 使用说明

> 最近更新：2026-07-22
> 适用：`/opt/search_service`（FastAPI 端口 8200，nginx 反向代理到公网 80 端口）
> **推荐公网地址：** `http://81.70.191.196:80/search`

---

## Prompt 优化服务 API（端口 8000 / 公网 80）

> 适用：`/root/app/core`（FastAPI 端口 8000，nginx 反向代理到公网 80 端口）
> **推荐公网地址：** `http://81.70.191.196:80/`

服务包含三个独立接口，分别对应不同的优化粒度：

| 接口 | 类型 | 功能 |
|---|---|---|
| `/optimize` | 场景级 | 分析 query 语义结构（子查询、实体、聚类） |
| `/optimize_case` | 案例级 | 定向改进 system prompt（可选结合场景级结果） |
| `/passthrough` | 案例级 | 定向改进 system prompt（不依赖场景级，独立使用） |

### /passthrough — 案例级 Prompt 改进（推荐）

**公网地址：** `http://81.70.191.196:80/passthrough`

对用户提供的 system prompt 做定向改进，输出改进后的版本。适合独立使用，不依赖场景级分析结果。

**请求体：**

```json
{
  "query":    "南孔庙的开放时间？",
  "prompt":   "你是一个景区导游助手，请用简洁的语言回答游客的问题。",
  "direction": "简洁"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query` | string | ✅ | 原始用户问题（1~500 字符） |
| `prompt` | string | ✅ | 当前使用的 system prompt |
| `direction` | string | ✅ | 迭代方向：`简洁` / `详细` / `专业化` / `友好` / `精确` / `全面`，也支持自定义描述文字 |

**成功响应（HTTP 200）：**

```json
{
  "ok": true,
  "code": 200,
  "data": {
    "original_query": "南孔庙的开放时间？",
    "refined_prompt": "你是一个景区导游助手。请用简洁、专业的语言直接回答游客关于景区开放时间的问题...",
    "direction": "简洁",
    "scenario_used": false,
    "llm_time": 1.23,
    "total_time": 1.45
  },
  "error": null
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `refined_prompt` | string | 改进后的 system prompt 正文 |
| `direction` | string | 实际使用的迭代方向 |
| `scenario_used` | bool | 是否使用了场景级上下文（本接口始终 false） |
| `llm_time` | float | LLM 调用耗时（秒） |
| `total_time` | float | 接口总耗时（秒） |

**错误码：**

| `code` | 含义 |
|---|---|
| `200` | 成功 |
| `400` | 缺少必填字段或 API Key 非法 |
| `500` | 服务异常（LLM 调用失败等） |

---

### /optimize_case — 案例级 Prompt 改进（可选结合场景上下文）

**公网地址：** `http://81.70.191.196:80/optimize_case`

与 `/passthrough` 功能相同，但支持可选传入场景级分析结果（来自 `/optimize` 接口的输出），在场景约束下做更精准的 prompt 改进。

**请求体：**

```json
{
  "query":           "南孔庙的开放时间？",
  "prompt":          "你是一个景区导游助手，请用简洁的语言回答游客的问题。",
  "direction":       "简洁",
  "scenario_result":  null
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query` | string | ✅ | 原始用户问题（1~500 字符） |
| `prompt` | string | ✅ | 当前使用的 system prompt |
| `direction` | string | ✅ | 迭代方向 |
| `scenario_result` | object | ❌ | 场景级结果（来自 `/optimize` 接口），传入后 prompt 改进会遵循场景约束 |

**成功响应（HTTP 200）：**

```json
{
  "ok": true,
  "code": 200,
  "data": {
    "original_query": "南孔庙的开放时间？",
    "refined_prompt": "你是一个景区导游助手。请用简洁、专业的语言直接回答...",
    "direction": "简洁",
    "scenario_used": false,
    "llm_time": 1.23,
    "total_time": 1.45
  },
  "error": null
}
```

---

### /optimize — 场景级 Query 分析

**公网地址：** `http://81.70.191.196:80/optimize`

分析用户 query 的语义结构，返回子查询拆分、实体术语提取、聚类匹配结果。可作为案例级接口的输入。

**请求体：**

```json
{
  "query": "南孔庙的开放时间和门票价格？"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query` | string | ✅ | 原始问题（1~500 字符） |
| `use_llm_subqueries` | bool | ❌ | 是否用 LLM 拆分子查询，默认 true |
| `use_entity_extraction` | bool | ❌ | 是否提取实体术语，默认 true |
| `use_cluster_prompt` | bool | ❌ | 是否匹配 PromptModule 聚类，默认 true |
| `cluster_top_n` | int | ❌ | 返回前 N 个聚类，默认 2 |
| `return_optimized_prompt` | bool | ❌ | 是否额外生成优化后的 system prompt，默认 true |

**成功响应（HTTP 200）：**

```json
{
  "ok": true,
  "code": 200,
  "data": {
    "original_query": "南孔庙的开放时间和门票价格？",
    "fusion_query": "南孔庙 开放时间 门票价格",
    "sub_queries": ["南孔庙的开放时间", "南孔庙的门票价格"],
    "entity_terms": ["南孔庙"],
    "cluster_id": 3,
    "cluster_sim": 0.87,
    "prompt_module": { "name": "景区开放时间问答", "template": "..." },
    "top_clusters": [...],
    "top_prompts": [...],
    "optimized_prompt": "你是一个景区 FAQ 助手...",
    "optimize_time": 1.2,
    "total_time": 1.45
  },
  "error": null
}
```

---

### /healthz — 健康检查

**公网地址：** `http://81.70.191.196:80/healthz`

```json
{ "ok": true, "service": "prompt-iteration-optimizer" }
```

---

## RAG 检索服务 API 使用说明

## 服务地址

**公网（推荐）：**
```
http://81.70.191.196:80/search
```

**内网直连：**
```
http://81.70.191.196:8200/search
```

## 通用说明

- 请求体使用 `Content-Type: application/json`（UTF-8）
- 接口返回统一结构：`{ ok, code, data, error }`
- 向量库后端：**Qdrant**（127.0.0.1:6333），语料集合 `unified_corpus`、维度标签集合 `dimension_tags`
- 编码模型：服务端预加载 BGE-M3（`/opt/search_service/models/bge-m3`），首次调用后常驻内存
- ChunkHit 中所有字段均可能为 `null`，按需读取

## 三种检索模式

| Mode | 召回来源 | 适用场景 |
|---|---|---|
| `sem` | 纯语义（BGE 向量 ANN） | 含义模糊、跨领域、同义词查询 |
| `dim` | 维度标签匹配 + 重排 | 结构化概念查询（人物、地名、文化标签） |
| `fusion` | 语义 + 维度融合 | 综合召回，覆盖面最广 |

**fusion 模式的权重规则：**

- 不传 `alpha_dim` / `alpha_sem` → **自适应权重**（内部基于 P/T/C/U 因子动态计算，`fusion_strategy` 控制融合方式）
- **两者都传** → 走固定权重分支（`alpha_dim + alpha_sem ≤ 1`，未给的一侧按 1 - 另一侧算）
- 只传一个 → 当成固定权重，另一侧自动按 1 - 此值补足

---

## 接口列表

### 检索

```
POST /search
```

**请求体参数：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `query` | string | ✅ | — | 原始查询，1–500 字符 |
| `mode` | string | ❌ | `"sem"` | `sem` / `dim` / `fusion` |
| `top_k` | int | ❌ | `8` | 返回数量，1–50 |
| `alpha_dim` | float? | ❌ | `null` | 维度权重 0–1，传了就强制走固定权重 |
| `alpha_sem` | float? | ❌ | `null` | 语义权重 0–1，传了就强制走固定权重 |
| `fusion_strategy` | string? | ❌ | `"adaptive"` | `adaptive` / `rrf` / `score`，仅 fusion 模式生效 |

**请求示例：**

```bash
# 语义检索
curl -s -X POST http://81.70.191.196:80/search \
  -H "Content-Type: application/json" \
  -d '{"query":"良渚玉器的文化内涵","mode":"sem","top_k":5}'

# fusion（自适应）
curl -s -X POST http://81.70.191.196:80/search \
  -H "Content-Type: application/json" \
  -d '{"query":"良渚玉器的文化内涵","mode":"fusion","top_k":5}'

# fusion（固定权重 + RRF）
curl -s -X POST http://81.70.191.196:80/search \
  -H "Content-Type: application/json" \
  -d '{"query":"良渚玉器的文化内涵","mode":"fusion","top_k":5,"alpha_dim":0.3,"alpha_sem":0.7,"fusion_strategy":"rrf"}'
```

**Python 调用示例：**

```python
import requests

resp = requests.post(
    "http://81.70.191.196:80/search",
    json={
        "query": "良渚玉器的文化内涵",
        "mode": "fusion",
        "top_k": 5,
        # 可选: "alpha_dim": 0.3, "alpha_sem": 0.7, "fusion_strategy": "rrf"
    },
    timeout=30,
)
data = resp.json()
if data["ok"]:
    # 推荐直接读 fusion_results（最全的合并视图）
    for h in data["data"]["fusion_results"]:
        print(h["fusion_rank"], h["chunk_id"], round(h["final_score"], 4), h["doc_title"])
    # 仅 sem: 改读 data["data"]["sem_results"]
    # 仅 dim: 改读 data["data"]["dim_results"]
```

**返回结构：**

| 字段 | 类型 | 说明 |
|---|---|---|
| `ok` | boolean | 是否成功 |
| `code` | int | 状态码，`200` 成功 |
| `data` | object? | 成功时存在，结构见下 |
| `error` | string? | 失败时写原因 |

**`data` 子项（三组结果）：**

| 字段 | 类型 | 出现场景 |
|---|---|---|
| `sem_results` | ChunkHit[] | mode=`sem` 时主输出；mode=`fusion` 时为分量 |
| `dim_results` | ChunkHit[] | mode=`dim` 时主输出；mode=`fusion` 时为分量 |
| `fusion_results` | ChunkHit[] | 仅 mode=`fusion` 有；`source="dim+sem"`，含 `final_score` |

**ChunkHit 字段：**

| 字段 | 类型 | 说明 |
|---|---|---|
| `chunk_id` | string? | 文档片段唯一 ID |
| `doc_title` | string? | 所属文档标题 |
| `score` | float? | 原始召回分数 |
| `final_score` | float? | 融合最终分（仅 `fusion_results` 非空） |
| `dim_rank` | int? | 在 dim 召回中的排名（1-based），`dim_results` / `fusion_results` 可能有值 |
| `sem_rank` | int? | 在 sem 召回中的排名（1-based），`sem_results` / `fusion_results` 可能有值 |
| `fusion_rank` | int? | 在 fusion 列表中的排名（1-based），仅 `fusion_results` 非空 |
| `source` | string? | `"sem"` / `"dim"` / `"dim+sem"` |
| `dim_name` | string? | 命中维度名，如 `culture_entities` |
| `tag_name` | string? | 命中的标签名，如 `良渚文化` |
| `tag_hits` | array? | 多标签命中明细：`[{ "tag": "...", "dim": "...", "score": 0.64 }, ...]` |
| `evidence` | array? | 命中证据串：`["culture_entities:良渚文化", "exhibit_entities:玉琮"]` |
| `chunk_text` | string? | 截断后的片段正文 |

---

### 三种模式返回对比

#### mode=sem（纯语义）

`sem_results` 有值；`dim_results` / `fusion_results` 为空数组。

```jsonc
{
  "ok": true,
  "code": 200,
  "data": {
    "sem_results": [
      {
        "chunk_id": "良渚玉琮文化_c0001",
        "doc_title": "良渚文化.txt",
        "score": 0.513,
        "sem_rank": 1,
        "source": "sem",
        "chunk_text": "……"
      }
      // 共 top_k 条
    ],
    "dim_results": [],
    "fusion_results": []
  },
  "error": null
}
```

#### mode=dim（维度召回）

`dim_results` 有值；`sem_results` / `fusion_results` 为空数组。

```jsonc
{
  "ok": true,
  "code": 200,
  "data": {
    "sem_results": [],
    "dim_results": [
      {
        "chunk_id": "良渚玉琮文化_c0001",
        "doc_title": "良渚文化.txt",
        "score": 0.493,
        "dim_rank": 1,
        "source": "dim",
        "dim_name": "culture_entities",
        "tag_name": "良渚文化",
        "tag_hits": [
          { "tag": "良渚文化",   "dim": "culture_entities", "score": 0.64 },
          { "tag": "浙江文化",   "dim": "culture_entities", "score": 0.42 }
        ],
        "evidence": [
          "culture_entities:良渚文化",
          "culture_entities:浙江文化"
        ],
        "chunk_text": "……"
      }
      // 共 top_k 条
    ],
    "fusion_results": []
  },
  "error": null
}
```

#### mode=fusion（语义 + 维度融合）

三组都有值；`fusion_results[i].source="dim+sem"`，`final_score` 与 `fusion_rank` 必有值。

```jsonc
{
  "ok": true,
  "code": 200,
  "data": {
    "sem_results":    [ /* top_k 条，source="sem" */ ],
    "dim_results":    [ /* top_k 条，source="dim" */ ],
    "fusion_results": [
      {
        "chunk_id":    "良渚玉琮文化_c0001",
        "doc_title":   "良渚文化.txt",
        "score":       0.487,
        "final_score": 0.0162,
        "dim_rank":    2,
        "sem_rank":    1,
        "fusion_rank": 1,
        "source":      "dim+sem",
        "dim_name":    "culture_entities",
        "tag_name":    "良渚文化",
        "tag_hits": [
          { "tag": "良渚文化", "dim": "culture_entities", "score": 0.64 },
          { "tag": "浙江文化", "dim": "culture_entities", "score": 0.42 }
        ],
        "evidence": [
          "culture_entities:良渚文化",
          "culture_entities:浙江文化"
        ],
        "chunk_text": "……"
      }
      // 共 top_k 条
    ]
  },
  "error": null
}
```

---

## 错误码

| `code` | HTTP 状态码 | 含义 | 典型场景 |
|---|---|---|---|
| `200` | 200 | 成功 | — |
| `400` | 200 | 请求体校验失败（业务层） | `mode` / `fusion_strategy` 非法值 |
| `500` | 200 | 服务异常 | Qdrant 不可用、BGE 编码失败、内部异常（`error` 字段写具体原因） |
| `422` | **422** | Pydantic 校验失败（HTTP 层） | 缺 `query`、`top_k` 越界、`alpha_*` 越界 |

> 与 8000 端口的 Prompt 服务不同：本服务**错误时 HTTP 状态码也可能是 4xx/5xx**（不强制 200）。建议优先看 `ok` 字段判断。

---

## 错误响应示例

**mode 非法：**

```json
{
  "ok": false,
  "code": 400,
  "data": null,
  "error": "invalid mode: oops，应为 sem / dim / fusion"
}
```

**缺 query（HTTP 422）：**

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "query"],
      "msg": "Field required",
      "input": { "mode": "sem", "top_k": 3 }
    }
  ]
}
```

**alpha 越界（HTTP 422）：**

```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "alpha_dim"],
      "msg": "Input should be less than or equal to 1",
      "input": 1.5,
      "ctx": { "le": 1.0 }
    }
  ]
}
```
