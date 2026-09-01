# Prompt Iteration Optimizer API 使用说明

> 最近更新：2026-07-14  
> 版本：v1.1（新增 `Authorization: Bearer` Header 传 Key）

## 服务地址

```
http://81.70.191.196:8000
```

## 通用说明

- 请求体使用 `Content-Type: application/json`（UTF-8）
- DashScope API Key **不在服务端持久化**，每次调用由调用方提供
- 接口返回统一结构：`{ ok, code, data, error }`
- 默认模型路径为 `/root/mingqiang/model/bge-m3`，可通过环境变量 `BGE_MODEL_PATH` 覆盖

## DashScope API Key 传递方式

`/optimize` 接口支持 **2 种** 传 Key 方式，**优先级 Header > Body**。

### 方式 1：HTTP Header（✅ 推荐）

```bash
Authorization: Bearer sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

也支持裸 token（去掉 `Bearer ` 前缀）：

```bash
Authorization: sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

**优势**：

- 不进请求体 body，避免 key 出现在服务端日志 / 分布式 tracing / 监控抓包里
- 标准 OAuth2 风格，便于网关、APISIX、Kong 等统一鉴权层对接
- 调用方只需设置一次 header，body 保持干净

### 方式 2：请求体字段（向后兼容）

```json
{
  "api_key": "sk-你的DashScopeKey",
  "query": "北京旅游三天推荐"
}
```

> 老调用方零修改即可继续工作；新代码建议改用 Header 方式。

### 错误行为

Header 和 Body 都没传 → 返回 400 + 明确报错：

```json
{
  "ok": false,
  "code": 400,
  "data": null,
  "error": "DashScope API Key 未提供：请通过 Header `Authorization: Bearer <key>` 或在请求 body `api_key` 字段传入。"
}
```

---

## 接口列表

### 1. 健康检查

```
GET /healthz
```

**请求示例：**

```bash
curl -s http://81.70.191.196:8000/healthz
```

**返回示例：**

```json
{
  "ok": true,
  "service": "prompt-iteration-optimizer"
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|---|---|---|
| `ok` | boolean | 服务是否正常 |
| `service` | string | 服务名称 |

---

### 2. 场景级 prompt 迭代优化

```
POST /optimize
```

**用途：**

将用户原始 query 转换为子查询、实体词、聚类 PromptModule 匹配结果，并通过 LLM 生成**优化后的 system prompt**。

**鉴权 / Header 传参：**

| Header | 必填 | 说明 |
|---|---|---|
| `Authorization` | ✅ 推荐 | `Bearer sk-xxxxxxxx` 或裸 `sk-xxxxxxxx` |
| `Content-Type` | ✅ | `application/json` |

**请求体参数：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `api_key` | string | 否（Header 已传则可省） | - | DashScope API Key，仅当 Header 未传时使用 |
| `query` | string | ✅ | - | 原始查询，1–500 字符 |
| `use_llm_subqueries` | boolean | 否 | `true` | 是否使用 LLM 生成子查询 |
| `use_entity_extraction` | boolean | 否 | `true` | 是否提取实体词 |
| `use_cluster_prompt` | boolean | 否 | `true` | 是否使用聚类 PromptModule 匹配 |
| `cluster_top_n` | integer | 否 | `2` | 返回 Top-N 个聚类结果，范围 1–5 |
| `base_prompt` | string | 否 | `null` | 兜底 system prompt，聚类未命中且 LLM 失败时使用 |
| `return_optimized_prompt` | boolean | 否 | `true` | 是否额外 LLM 生成"优化后的 system prompt"；设为 `false` 可只拿到分类结果（耗时 1–2s） |

#### 调用示例

##### 方式 A：Header 传 Key（推荐）

```bash
curl -s -X POST http://81.70.191.196:8000/optimize \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-你的DashScopeKey" \
  -d '{
    "query": "北京旅游三天推荐",
    "use_llm_subqueries": true,
    "use_entity_extraction": true,
    "use_cluster_prompt": true,
    "cluster_top_n": 2
  }'
```

##### 方式 B：Body 传 Key（向后兼容）

```bash
curl -s -X POST http://81.70.191.196:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "sk-你的DashScopeKey",
    "query": "北京旅游三天推荐",
    "use_llm_subqueries": true,
    "use_entity_extraction": true,
    "use_cluster_prompt": true,
    "cluster_top_n": 2
  }'
```

##### Python 调用示例（Header 方式）

```python
import requests

resp = requests.post(
    "http://81.70.191.196:8000/optimize",
    headers={
        "Authorization": "Bearer sk-你的DashScopeKey",
        "Content-Type": "application/json",
    },
    json={
        "query": "北京旅游三天推荐",
        "use_llm_subqueries": True,
        "use_entity_extraction": True,
        "use_cluster_prompt": True,
        "cluster_top_n": 2,
    },
    timeout=60,
)
data = resp.json()
print(data["code"], data["ok"])
if data["ok"]:
    print("子查询:", data["data"]["sub_queries"])
    print("实体:", data["data"]["entity_terms"])
    print("命中聚类:", data["data"]["cluster_id"], "相似度:", data["data"]["cluster_sim"])
    print("\n【优化后 prompt】")
    print(data["data"]["optimized_prompt"])
```

##### Python 调用示例（Body 方式，兼容）

```python
resp = requests.post(
    "http://81.70.191.196:8000/optimize",
    json={
        "api_key": "sk-你的DashScopeKey",
        "query": "北京旅游三天推荐",
    },
    timeout=60,
)
```

**返回字段说明：**

| 字段 | 类型 | 说明 |
|---|---|---|
| `ok` | boolean | 是否成功 |
| `code` | integer | 状态码，200 表示成功 |
| `data.original_query` | string | 原始查询 |
| `data.sub_queries` | array[string] | LLM 拆解出的 2–4 个子查询 |
| `data.entity_terms` | array[string] | 提取的实体术语 |
| `data.top_clusters` | array | 命中的聚类 Top-N，元素形如 `[cluster_id, similarity]` |
| `data.cluster_id` | integer? | 命中的最佳聚类 ID，未命中为 `null` |
| `data.cluster_sim` | float | 最佳聚类相似度 |
| `data.prompt_module` | dict? | 命中的聚类 PromptModule 模板，未命中为 `null` |
| `data.fusion_query` | string | 子查询 + 实体用 `\|` 拼接的融合检索串 |
| **`data.optimized_prompt`** | **string** | **基于子查询/实体/聚类 PromptModule 由 LLM 重写生成的最终 system prompt** |
| `data.optimize_time` | float | 子查询/实体/聚类 耗时（不含 LLM 生成 optimized_prompt） |
| `data.total_time` | float | 整个 `/optimize` 请求的总耗时（秒） |
| `error` | string? | 错误信息，成功时为 `null` |

**返回示例：**

```json
{
  "ok": true,
  "code": 200,
  "data": {
    "original_query": "北京旅游三天推荐",
    "fusion_query": "北京旅游景点推荐 | 北京三天行程安排与交通住宿建议 | 2024年北京春季旅游最佳路线",
    "sub_queries": [
      "北京旅游景点推荐",
      "北京三天行程安排与交通住宿建议",
      "2024年北京春季旅游最佳路线"
    ],
    "entity_terms": ["北京", "旅游", "三天", "推荐"],
    "cluster_id": null,
    "cluster_sim": 0.0,
    "prompt_module": null,
    "top_clusters": [],
    "top_prompts": [],
    "optimized_prompt": "你是一个专业的旅游规划助手，专注于为中国游客提供精准、实用、时效性强的本地化旅行建议……",
    "optimize_time": 1.57,
    "total_time": 5.14
  },
  "error": null
}
```

**性能参考：**

| 配置 | 典型耗时 | 主要耗时来源 |
|---|---|---|
| `return_optimized_prompt=true`（默认） | **5–8 s** | 1× LLM 子查询 + 1× 实体 + 1× 生成 optimized_prompt + BGE 编码 |
| `return_optimized_prompt=false` | **1–2 s** | 仅子查询 + 实体 + 聚类匹配 |

> BGE 编码器和聚类数据已在服务端预加载到全局单例，**不会**每次请求重新加载。

**错误码：**

| 状态码 | 含义 | 典型场景 |
|---|---|---|
| `200` | 成功 | — |
| `400` | 请求体校验失败 | Header + Body 都没传 Key |
| `422` | Pydantic 校验失败 | 缺 `query`、`cluster_top_n` 越界等 |
| `500` | 服务异常 | 聚类数据未加载、LLM 调用失败等（`error` 字段写具体原因） |

---

### 3. 占位 / 透传接口

```
POST /passthrough
```

**用途：** 原样透传 `query / prompt / direction`，当前**不做任何处理**，供上层预留"原始 prompt + 迭代方向 → 输出原始 prompt"的协议占位（`processed: false`）。

**请求体参数：**

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query` | string | ✅ | 原始问题，1–500 字符 |
| `prompt` | string | ✅ | 原始 prompt |
| `direction` | string | 否 | 迭代方向（描述文本），当前仅透传 |

**请求示例：**

```bash
curl -s -X POST http://81.70.191.196:8000/passthrough \
  -H "Content-Type: application/json" \
  -d '{
    "query": "北京旅游三天推荐",
    "prompt": "你是一个旅游助手",
    "direction": "让回答更口语化"
  }'
```

**Python 调用示例：**

```python
import requests

resp = requests.post(
    "http://81.70.191.196:8000/passthrough",
    json={
        "query": "北京旅游三天推荐",
        "prompt": "你是一个旅游助手",
        "direction": "让回答更口语化",
    },
)
print(resp.json())
```

**返回示例：**

```json
{
  "ok": true,
  "code": 200,
  "data": {
    "query": "北京旅游三天推荐",
    "prompt": "你是一个旅游助手",
    "direction": "让回答更口语化",
    "processed": false
  },
  "error": null
}
```

---

## 附录：完整错误响应一览

| 场景 | HTTP | `code` | `error` 内容 |
|---|---|---|---|
| Header + Body 都没传 Key | 200 | 400 | "DashScope API Key 未提供：请通过 Header `Authorization: Bearer <key>` 或在请求 body `api_key` 字段传入。" |
| Body 缺 `query` | 422 | - | Pydantic 校验错误详情 |
| `cluster_top_n` 越界 | 422 | - | Pydantic 校验错误详情 |
| LLM 调用失败 | 200 | 500 | 异常 message |
| 聚类数据未加载 | 200 | 500 | 异常 message |
> 接口内部错误统一用 `{ ok:false, code, error }` 形式返回，HTTP 状态码仍是 200。如需严格按 HTTP 状态码判断，请使用 `/healthz` 做探活。
