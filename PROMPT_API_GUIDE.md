# Prompt 优化服务 API 使用说明

> 最近更新：2026-07-22
> 适用：`/root/app/core`（FastAPI 端口 8000，nginx 反向代理到公网 80 端口）
> **推荐公网地址：** `http://81.70.191.196:80/`

---

## /refine_with_gt — 基于标准答案的自动 Prompt 改进

**公网地址：** `http://81.70.191.196:80/refine_with_gt`

传入原始问题、当前 prompt 和标准答案（ground_truth），自动推断迭代方向并改进 prompt。

**流程：**
1. 根据 query 调用 search 接口检索相关上下文
2. 将参考答案 + 检索上下文 + 当前 prompt 传入模型，自动推断最优迭代方向
3. 用推断出的方向对 prompt 进行多轮优化

**请求体：**

```json
{
  "query":          "南孔庙的开放时间？",
  "prompt":          "你是一个景区导游助手，请用简洁的语言回答游客的问题。",
  "ground_truth":    "南孔庙开放时间为每日 8:30-17:00。"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query` | string | ✅ | 原始问题（1~500 字符） |
| `prompt` | string | ✅ | 当前使用的 system prompt |
| `ground_truth` | string | ✅ | 标准答案（用户提供的参考答案） |
| `search_url` | string | ❌ | Search 服务地址，默认 `http://81.70.191.196:80/search` |
| `max_iterations` | int | ❌ | 最大迭代次数，默认 3 |

**成功响应（HTTP 200）：**

```json
{
  "ok": true,
  "code": 200,
  "data": {
    "original_query":    "南孔庙的开放时间？",
    "inferred_direction": "精确",
    "refined_prompt":    "你是一个景区 FAQ 助手。请根据检索到的上下文，用准确、完整的方式回答游客关于景区开放时间的问题...",
    "iterations":        3,
    "process_time":      4.21,
    "total_time":        5.05
  },
  "error": null
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `inferred_direction` | string | 从参考答案与当前 prompt 对比推断出的迭代方向（简洁/详细/专业化/友好/精确/全面） |
| `refined_prompt` | string | 改进后的 system prompt 正文 |
| `iterations` | int | 实际迭代轮数 |
| `process_time` | float | 优化服务调用总耗时（秒） |
| `total_time` | float | 接口总耗时（秒） |

---

## /optimize_case — 案例级 Prompt 改进

**公网地址：** `http://81.70.191.196:80/optimize_case`

对用户提供的 system prompt 进行定向改进。支持多轮迭代，默认迭代 3 轮。

**请求体：**

```json
{
  "query":     "南孔庙的开放时间？",
  "prompt":    "你是一个景区导游助手，请用简洁的语言回答游客的问题。",
  "direction": "简洁"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query` | string | ✅ | 原始问题（1~500 字符） |
| `prompt` | string | ✅ | 当前使用的 system prompt |
| `direction` | string | ✅ | 迭代方向 |

**direction 可选值：** `简洁` / `详细` / `专业化` / `友好` / `精确` / `全面`

**成功响应（HTTP 200）：**

```json
{
  "ok": true,
  "code": 200,
  "data": {
    "original_query": "南孔庙的开放时间？",
    "refined_prompt": "你是一个景区导游助手。请用简洁、专业的语言直接回答...",
    "direction": "简洁",
    "iterations": 3,
    "process_time": 3.69,
    "total_time": 3.95
  },
  "error": null
}
```

---

## /healthz — 健康检查

**公网地址：** `http://81.70.191.196:80/healthz`

```json
{ "ok": true, "service": "prompt-iteration-optimizer" }
```
