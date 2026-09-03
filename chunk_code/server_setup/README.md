# 服务器部署说明：去噪 QA 实验环境

目标目录：`/home/humq/milvus_denoise`

## 组件布局

- Rootless Docker 数据和 socket：`/home/humq/milvus_denoise/docker-data`、`run/`
- Milvus 持久化数据：`/home/humq/milvus_denoise/volumes/`
- Python 环境：建议 `/home/humq/envs/denoise_qa`
- Hugging Face 缓存和模型：`/home/humq/models`、`/home/humq/hf_cache`
- 去噪两组实验输出：`/home/humq/chunk_code/results/denoise_ablation_db_qa_textsafe/`

## Milvus Lite（无 Docker）

```bash
source /home/humq/milvus_denoise/server_setup/env_denoise_qa.sh
test -n "$DENOISE_MILVUS_URI"
```

`DENOISE_MILVUS_URI` 指向 `/home/humq/milvus_denoise/milvus_lite.db`。PyMilvus 2.6.17 会通过 `milvus-lite` 在当前 Python 进程内启动本地 Milvus Lite，不需要 Docker、systemd、端口或管理员权限。

旧的 Docker compose 和 rootless 脚本仅作为历史备份保留，不再是运行路径。

## Python 环境

进入实验 shell 后先加载公共路径配置：

```bash
source /home/humq/milvus_denoise/server_setup/env_denoise_qa.sh
```

不要修改 `/opt/conda/envs/testbase`。可以先克隆它以复用 CUDA PyTorch：

```bash
/opt/conda/bin/conda create \
  --prefix /home/humq/envs/denoise_qa \
  --clone /opt/conda/envs/testbase -y

/opt/conda/envs/denoise_qa/bin/pip install \
  -r /home/humq/chunk_code/server_setup/requirements_denoise_qa.txt
```

如果克隆环境空间或依赖冲突，应改为新建 Python 3.10 环境，并单独安装与服务器 CUDA 驱动匹配的 PyTorch。

## 模型和密钥

```bash
export HF_HOME=/home/humq/hf_cache
export TRANSFORMERS_CACHE=/home/humq/hf_cache
export QWEN_OPENAI_API_KEY='只在当前 shell 设置，不写入文件'
export QWEN_OPENAI_API_BASE='https://dashscope.aliyuncs.com/compatible-mode/v1'
export QWEN_OPENAI_MODEL_NAME='qwen-plus'
export QWEN_MIN_INTERVAL='10'
# 使用 SiliconFlow 的 Qwen/Qwen3-8B 时可关闭思考模式以降低单请求耗时
# export QWEN_ENABLE_THINKING='false'
```

Embedding 使用 `BAAI/bge-base-zh-v1.5`，第一次运行会下载到 `HF_HOME`。如果服务器不能访问 Hugging Face，应先在可联网机器下载后上传到 `/home/humq/hf_cache`。

## 两组 QA

```bash
cd /home/humq/chunk_code
/home/humq/envs/denoise_qa/bin/python run_denoise_qa_ablation.py \
  --ablation_root /home/humq/chunk_code/results/denoise_ablation_db_qa_textsafe \
  --data_path /home/humq/chunk_code/data/split_merged.json \
  --bert_score_eval
```

正式运行前建议先用 `--num_threads 1` 做小样本验证；当前服务器实测 GPU 2 可被 PyTorch 正常使用，可通过 `CUDA_VISIBLE_DEVICES=2` 选择。API key 只通过当前 shell 的 `QWEN_OPENAI_API_KEY` 注入，不写入配置文件或结果清单。
Qwen API 客户端默认每次请求至少间隔 10 秒，遇到 429/5xx 会指数退避；如服务商给出更严格的 RPM 限制，只增大 `QWEN_MIN_INTERVAL`，不要并发请求。

该命令调用项目正式入口 `Meta-Chunking/eval/CRUD/quick_start.py`，对 `denoise_off` 和 `denoise_on` 使用不同 Milvus collection，其他 QA 参数保持一致。
