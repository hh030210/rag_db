# 非人工论文实验队列

该目录只安排可自动复现的实验，人工 Chunk 质量评价不在队列中。服务器上默认使用 `/home/humq/chunk_code` 作为代码根目录，并将结果写入其 `results/nonhuman_paper_queue_20260825/`。

执行顺序为：

1. 生成 200、300、400 字符机械切分和三阶段切分结果。
2. 将分片转换成统一的检索文档目录。
3. 在相同问答数据、Embedding、检索 Top-k 和 Qwen API 配置下，依次运行四种方法的全量问答。
4. 对四种方法的逐问题结果做配对 Bootstrap 置信区间和符号检验。
5. 运行三阶段方法的第一阶段结构拆分、第二阶段去噪、第三阶段优化融合消融，并保存分片统计。

API 问答必须在服务器当前 shell 中设置 `QWEN_OPENAI_API_KEY`，并建议设置 `QWEN_OPENAI_API_BASE=https://api.siliconflow.cn/v1`、`QWEN_OPENAI_MODEL_NAME=Qwen/Qwen3-8B`、`QWEN_MIN_INTERVAL=10`。队列单线程串行执行，避免请求过频。

启动全部非人工实验：

```bash
source /home/humq/chunk_code/server_setup/env_denoise_qa.sh
export QWEN_OPENAI_API_BASE="https://api.siliconflow.cn/v1"
export QWEN_OPENAI_MODEL_NAME="Qwen/Qwen3-8B"
export QWEN_MIN_INTERVAL="10"
nohup bash /home/humq/chunk_code/paper_experiments/run_nonhuman_queue.sh --stage all > /home/humq/logs/nonhuman_paper_queue.log 2>&1 &
echo $!
```

只先准备分片：`--stage prepare`；准备完成后再执行 `--stage qa`；问答完成后执行 `--stage stats`。每一步有 `.done.json` 标记，可从中断处继续。
