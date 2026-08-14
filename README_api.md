# Prompt Iteration Optimizer API

把 `interactive_qa.py` 中"问题聚类提示词迭代优化"逻辑抽离并部署为 FastAPI HTTP 服务。

## 文件结构

```
prompt_iteration_optimizer.py    # core（纯逻辑，BGE + LLM 调用 + 聚类匹配）
prompt_iteration_service.py      # service 层（无 IO，单例 + 纯函数封装）
api_server.py                    # FastAPI HTTP 入口
requirements_api.txt             # 依赖声明
```

部署到服务器时把以上文件整体上传，保持如下相对路径布局：

```
<server_root>/
├── prompt_iteration_optimizer.py
├── prompt_iteration_service.py
├── api_server.py
├── requirements_api.txt
└── code1/
    └── chapter3_backup/
        └── codes/
            └── bylw_rag/
                └── new_experiments/
                    ├── clustering_results/
                    │   └── tourist/
                    │       └── cluster_results.json
                    └── iteration_results/
                        └── tourist/
                            ├── tourist_question_*/
                            │   └── final_prompt.json
                            └── tourist_question_cluster_mapping.json
```

## 部署

```bash
# 1. 安装依赖
pip install -r requirements_api.txt

# 2. 设置监听环境变量（仅本机服务监听的地址）
export HOST=0.0.0.0
export PORT=8000

# 3. 启动（开发）
python api_server.py

# 或启动（生产，推荐 workers=1 避免重复加载 BGE）：
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
```

## API 协议

### POST /optimize

场景级 prompt 迭代优化。**每次调用必须由调用方在请求体里传 `api_key`**，
服务端不做任何 Key 缓存或环境变量兜底，避免把本地 Key 上传到服务器。

请求：

```json
{
  "api_key": "sk-xxxx",
  "query": "南孔庙的开放时间是什么时候？",
  "use_llm_subqueries": true,
  "use_entity_extraction": true,
  "use_cluster_prompt": true,
  "cluster_top_n": 2
}
```

响应（成功）：

```json
{
  "ok": true,
  "code": 200,
  "data": {
    "original_query": "...",
    "fusion_query": "子查询1 | 子查询2 | ...",
    "sub_queries": ["...", "..."],
    "entity_terms": ["...", "..."],
    "cluster_id": 7,
    "cluster_sim": 0.82,
    "prompt_module": { "P_sys": "...", "I_t": "...", "C_t": "...", "F_t": "...", "U_t": "..." },
    "top_clusters": [[7, 0.82], [3, 0.65]],
    "top_prompts": [{...}, {...}],
    "optimize_time": 1.20,
    "total_time": 1.25
  }
}
```

响应（失败）：

```json
{
  "ok": false,
  "code": 400,
  "error": "api_key 必须由调用方显式传入，不允许空值或服务端兜底。"
}
```

或：

```json
{
  "ok": false,
  "code": 500,
  "error": "DashScope 错误: ..."
}
```

### POST /passthrough

占位接口，原样透传，不做处理。

请求：

```json
{
  "query": "南孔庙的开放时间是什么时候？",
  "prompt": "...原始 prompt 文本...",
  "direction": "更简洁地回答"
}
```

响应：

```json
{
  "ok": true,
  "code": 200,
  "data": {
    "query": "...",
    "prompt": "...",
    "direction": "更简洁地回答",
    "processed": false
  }
}
```

### GET /healthz

健康检查：

```json
{ "ok": true, "service": "prompt-iteration-optimizer" }
```

## cURL 示例

部署到 `81.70.191.196:8000` 后：

```bash
# 健康检查
curl http://81.70.191.196:8000/healthz

# optimize（由调用方传入 api_key）
curl -X POST http://81.70.191.196:8000/optimize \
  -H 'Content-Type: application/json' \
  -d '{
        "api_key": "sk-你的真实key",
        "query": "南孔庙的开放时间是什么时候？"
      }'

# passthrough（不需要 api_key）
curl -X POST http://81.70.191.196:8000/passthrough \
  -H 'Content-Type: application/json' \
  -d '{"query":"南孔庙的开放时间？","prompt":"你是助手。","direction":"更简洁"}'
```

## 服务端 Key 处理约定（重要）

| 行为 | 说明 |
|---|---|
| 读环境变量 | ❌ 禁用（避免上传到服务器后泄露本地 Key） |
| 硬编码 | ❌ 禁用（同上） |
| 缓存 Key | ❌ 每次都要求重传，避免长期明文持有 |
| 调用方传 | ✅ 必传路径 |

如需长期持有 Key，建议在调用方侧做加密管理或接入公司内部的密钥管理服务（KMS），
而不是服务端做持久化。

## workers 与 BGE 单例

- `PromptIterationOptimizer` 在 `prompt_iteration_service.get_optimizer()` 中以模块级单例形式构造
- 首次请求触发 BGE 编码器加载与聚类中心读取
- `--workers 1` 时所有请求复用同一单例
- `--workers > 1` 时每个 worker 各持一份单例（会重复加载 BGE），按 GPU 显存决定
