# search_service_deploy · 部署交付包

> 完整 RAG 检索微服务（独立部署版）

## 包含

```
search_service_deploy/
├── README.md                  # 当前文件
├── service/                   # 检索服务代码（FastAPI / BGE-M3 / Qdrant）
├── deploy/                    # docker-compose + Dockerfile + 部署脚本
├── data/                      # 数据放置说明
└── scripts/                   # 上传 / 远程命令脚本
```

## 一句话架构

```
┌──────────────────────────────┐
│  FastAPI search_api_server   │  :8100
│   ├─ /search                 │
│   ├─ /healthz                │
│   └─ /config                 │
└────────────┬─────────────────┘
             │
   ┌─────────┴──────────┐
   ▼                    ▼
┌──────────────┐ ┌─────────────────┐
│ BGE-M3 编码  │ │ Qdrant HTTP API │  :6333
│  (CPU 推理)  │ │  - unified_corpus│
└──────────────┘ │  - dimension_tags│
                └─────────────────┘
```

## 功能

| 能力 | 入口接口 |
|---|---|
| **纯语义检索** (BGE-M3 → 向量 ANN) | `mode=sem` |
| **纯维度检索** (3 种粗召回 / 2 种精排 / 2 种内部融合) | `mode=dim` |
| **融合检索** (RRF / score / 自适应权重) | `mode=fusion` |
| **外部权重覆盖** | `alpha_dim` / `alpha_sem` |
| **Top-K 返回** | 任何模式都返回 8；fusion 模式同时返回三路 |

## 三步上服务器

### 第一步：服务器前置条件

- Docker 24+ + docker compose v2
- 已经装了 `qdrant/qdrant:v1.12.4`（也可以用 `docker compose pull` 下载）
- BGE-M3 模型放到服务器，如 `/root/mingqiang/model/bge-m3`

### 第二步：把整个仓库拷贝到服务器

任选其一（**在 PowerShell 本地**）：

```bash
# A. 用 paramiko 上传
cd D:\RAG_DB_slim
python search_service_deploy/scripts/pack_and_upload.py \
  --host 81.70.191.196 \
  --user root \
  --password YOUR_PASSWORD \
  --remote-dir /root/mingqiang

# B. 手工 scp
scp -r search_service_deploy root@81.70.191.196:/root/mingqiang/
```

服务器上得到：

```
/root/mingqiang/search_service_deploy/
```

### 第三步：服务端执行部署

SSH 到服务器：

```bash
# 解包（如果用 tar.gz 的话）
cd /root/mingqiang
rm -rf search_service_deploy
tar -xzf search_service_deploy.tar.gz

# 启服务
cd search_service_deploy/deploy
bash deploy_server.sh
```

脚本会：
1. 复制到 `/opt/search_service/`
2. 软链 BGE-M3 模型 → `models/bge-m3`
3. `docker compose up -d qdrant`（等健康）
4. `docker compose build --no-cache search` + `up -d search`

部署完成后：

- Qdrant HTTP: `http://81.70.191.196:6333/healthz`
- Search API: `http://81.70.191.196:8100/healthz`
- Search 调用：`POST http://81.70.191.196:8100/search`

### 第四步：灌库（独立操作）

把 `chunks.jsonl` 上传到服务器：

```bash
# 本地
scp chunks.jsonl root@81.70.191.196:/opt/search_service/data/

# 服务器
cd /opt/search_service/deploy
bash load_data.sh /data/chunks.jsonl
```

完成后 `unified_corpus` + `dimension_tags` collection 就有了数据。

## 接口规范

详见 `service/README.md`。最常见的 3 种调用：

### 1. 纯语义检索

```bash
curl -s -X POST http://81.70.191.196:8100/search \
  -H "Content-Type: application/json" \
  -d '{"query":"北京春天玩什么","mode":"sem","top_k":8}'
```

### 2. 纯维度检索

```bash
curl -s -X POST http://81.70.191.196:8100/search \
  -H "Content-Type: application/json" \
  -d '{"query":"杭州西湖春天","mode":"dim","top_k":8}'
```

### 3. 融合检索 + 自适应权重

```bash
curl -s -X POST http://81.70.191.196:8100/search \
  -H "Content-Type: application/json" \
  -d '{"query":"江南烟雨小镇","mode":"fusion","top_k":8}'
```

返回中会多一个 `weight_breakdown`，告诉你为什么算出来的权重是 0.62/0.38：

```json
{
  "ok": true,
  "data": {
    "mode": "fusion",
    "alpha_dim": 0.62, "alpha_sem": 0.38, "alpha_source": "adaptive",
    "weight_breakdown": {
      "structural_confidence_P_q": 0.75,
      "label_evidence_T_q": 0.6,
      "concentration_C_dim": 0.55,
      "concentration_C_sem": 0.7,
      "utility_U_dim": 1.9, "utility_U_sem": 1.55
    },
    "dim_results": [...8条...],
    "sem_results": [...8条...],
    "fusion_results": [...8条...]
  }
}
```

### 4. 融合检索 + 用户传权重（覆盖）

```bash
curl -s -X POST http://81.70.191.196:8100/search \
  -H "Content-Type: application/json" \
  -d '{
    "query":"江南烟雨小镇",
    "mode":"fusion","top_k":8,
    "alpha_dim":0.7, "alpha_sem":0.3
  }'
```

返回 `alpha_source` 会变 `fixed`，不再走自适应。

### 5. 融合检索 + 用户只传一个权重

```bash
curl -s -X POST http://81.70.191.196:8100/search \
  -H "Content-Type: application/json" \
  -d '{"query":"...","mode":"fusion","top_k":8,"alpha_dim":0.8}'
```

`alpha_sem` 自动 = 0.2，依然 fixed。

## 与 `/optimize` 服务的衔接

| 服务 | 端口 | 作用 |
|---|---|---|
| `/optimize` | `8000` | Prompt 优化（复用既有） |
| `/search`   | `8100` | 检索（新建） |

二者**部署独立**，通过 `nginx` 或内网打通即可。如果要在 `/optimize` 内部"先做 prompt 优化、再做检索"，把 `/search` 当作一个普通 HTTP 后端调用就行。

## 维护

```bash
# 看日志
docker compose -f /opt/search_service/deploy/docker-compose.yml logs -f search

# 重启
docker compose -f /opt/search_service/deploy/docker-compose.yml restart search

# 重新灌库（先清空 collection）
docker compose -f /opt/search_service/deploy/docker-compose.yml exec qdrant \
  curl -X DELETE http://localhost:6333/collections/unified_corpus
bash load_data.sh /data/chunks.jsonl
```

## API 详细字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query` | string | ✅ | 1-500 字符 |
| `mode` | string | 否 | `sem` / `dim` / `fusion`（默认 `sem`） |
| `top_k` | int | 否 | 1-50（默认 8） |
| `alpha_dim` | float | 否 | 0-1 |
| `alpha_sem` | float | 否 | 0-1 |
| `fusion_strategy` | string | 否 | `adaptive` / `rrf` / `score`（默认 `adaptive`） |

返回：

| 字段 | 说明 |
|---|---|
| `data.alpha_dim` / `data.alpha_sem` | 实际使用的权重 |
| `data.alpha_source` | `fixed` 或 `adaptive` |
| `data.weight_breakdown` | （adaptive 模式下）P_q/T_q/C_dim/C_sem/U_dim/U_sem |
| `data.sem_results` / `data.dim_results` / `data.fusion_results` | 各 8 条 |
| `data.elapsed` | 总耗时 |

错误：

- 模式非法 → `code=400`
- 编码器未加载 / Qdrant 不可达 → `code=500`
- 统一格式 `{ok, code, data, error}`
