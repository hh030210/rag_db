# 服务器部署清单

> 目标：在 Linux 服务器上把"问题聚类提示词迭代优化"逻辑部署为 FastAPI 服务，
> 提供场景级、案例级与**案例级专利方案迭代优化**全套 HTTP 接口：
>
> 场景级：`/optimize`
> 案例级（轻量）：`/optimize_case` / `/passthrough` / `/refine_with_gt`
> **案例级（专利方案，2026-07-23 新增）：**
>   - `/case_iter/metrics`         生成评估指标 + 权重
>   - `/case_iter/initial_prompt`  生成结构化初始 prompt
>   - `/case_iter/iterate`         Algorithm 1 多轮迭代 + 收敛
>   - `/case_iter/cluster`         案例级聚类（复用 BGE）
>   - `/case_iter/prototypes`      抽取通用原型 prompt
>   - `/case_iter/archive`         优化档案结构化封装 / 落盘

## 1. 文件清单

### 核心代码（5 个 .py + 1 个 docs）

```
prompt_iteration_optimizer.py    # core（自包含，含 BGE 加载/编码）
prompt_iteration_service.py      # service 层（无 IO，单例 + 纯函数）
case_level_service.py            # 案例级 prompt 定向改进（可选依赖 scenario_result）
case_level_optimizer_service.py  # 案例级专利方案迭代优化（metrics / iterate / cluster / prototypes / archive）
api_server.py                    # FastAPI HTTP 入口（含全部接口）
README_api.md                    # API 文档
```

### 启动 / 安装脚本（Linux 用）

```
setup.sh                         # 一键安装依赖
start.sh                         # 启动 uvicorn（workers=1）
smoke_test.sh                    # 启动后冒烟测试
pack_upload.sh                   # 在 Linux 上打包
pack_upload.ps1                  # 在 Windows 上打包（PowerShell）
```

### Python 依赖声明

```
requirements_api.txt             # fastapi / uvicorn / pydantic
requirements_bge.txt             # sentence-transformers（FlagEmbedding 失败时回退）
requirements_numpy.txt           # numpy
```

### 训练产物（必备）

```
code1/chapter3_backup/codes/bylw_rag/new_experiments/
├── clustering_results/
│   └── tourist/
│       └── cluster_results.json
└── iteration_results/
    └── tourist/
        ├── tourist_question_cluster_mapping.json
        └── tourist_question_*/
            └── final_prompt.json
```

### BGE 模型（必备）

```
model/bge-m3/
└── ...（BGE-M3 模型权重）
```

> 服务器上若把 BGE 模型放在其他位置，可用环境变量 `BGE_MODEL_PATH` 覆盖。

## 2. 上传后服务器目录布局

```
<server_root>/
├── prompt_iteration_optimizer.py
├── prompt_iteration_service.py
├── api_server.py
├── README_api.md
├── DEPLOY.md
├── requirements_api.txt
├── requirements_bge.txt
├── requirements_numpy.txt
├── setup.sh
├── start.sh
├── smoke_test.sh
├── model/
│   └── bge-m3/
└── code1/
    └── chapter3_backup/
        └── codes/
            └── bylw_rag/
                └── new_experiments/
                    ├── clustering_results/tourist/cluster_results.json
                    └── iteration_results/tourist/...
```

## 3. 部署步骤

```bash
# 0. 上传 zip 到服务器
scp dist/prompt_iteration_optimizer_<时间戳>.zip user@81.70.191.196:~
ssh user@81.70.191.196
unzip prompt_iteration_optimizer_<时间戳>.zip -d app/
cd app/core

# 1. 安装依赖
chmod +x setup.sh start.sh smoke_test.sh
./setup.sh

# 2. 启动服务
./start.sh

# 3. 冒烟测试
./smoke_test.sh

# 4. 用 nginx / systemd 做反向代理 / 守护进程（可选）
```

## 4. systemd 守护（推荐）

新建 `/etc/systemd/system/prompt-optimizer.service`：

```ini
[Unit]
Description=Prompt Iteration Optimizer
After=network.target

[Service]
Type=simple
User=app
WorkingDirectory=/home/app/core
Environment=HOST=0.0.0.0
Environment=PORT=8000
ExecStart=/usr/bin/env bash start.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

启用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now prompt-optimizer
sudo systemctl status prompt-optimizer
```

## 5. 调用样例

```bash
# 健康检查
curl http://81.70.191.196:8000/healthz

# /optimize（场景级）
curl -X POST http://81.70.191.196:8000/optimize \
  -H 'Content-Type: application/json' \
  -d '{
        "api_key": "sk-你的真实key",
        "query": "南孔庙的开放时间是什么时候？"
      }'

# /optimize_case（案例级，可选传入场景级结果）
# 先调 /optimize 拿到 scenario_result，再传入本接口做定向改进
curl -X POST http://81.70.191.196:8000/optimize_case \
  -H 'Content-Type: application/json' \
  -d '{
        "api_key": "sk-你的真实key",
        "query": "南孔庙的开放时间是什么时候？",
        "prompt": "你是助手。",
        "direction": "详细",
        "scenario_result": { /* /optimize 返回的完整 data 对象 */ }
      }'

# /passthrough（不需要 api_key）
curl -X POST http://81.70.191.196:8000/passthrough \
  -H 'Content-Type: application/json' \
  -d '{"query":"南孔庙的开放时间？","prompt":"你是助手。","direction":"更简洁"}'
```

## 6. Key 安全约定

| 行为 | 说明 |
|---|---|
| 服务端硬编码 Key | ❌ 禁用 |
| 服务端读环境变量 Key | ❌ 禁用 |
| 服务端持久化 Key | ❌ 禁用 |
| 调用方每次传入 Key | ✅ 唯一允许路径 |

## 7. 端口 / 防火墙

服务默认监听 `0.0.0.0:8000`。如果对外暴露在公网，建议：

- 用 nginx 反向代理到 443/HTTPS
- 加 IP 白名单 / basic auth
- 不要直接暴露 8000 端口