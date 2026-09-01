#!/usr/bin/env bash

# Source this file in the server shell before running the denoise QA ablation.
export DENOISE_QA_ROOT="/home/humq/milvus_denoise"
export DENOISE_QA_ENV="/home/humq/envs/denoise_qa"
export HF_HOME="/home/humq/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export BGE_MODEL_PATH="/home/humq/hf_cache/bge-base-zh-v1.5"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"
export QWEN_MIN_INTERVAL="10"  # API 请求最小间隔（秒），单线程串行调用
# Qwen/Qwen3-8B 可设置为 false 关闭思考模式，减少单请求耗时：
# export QWEN_ENABLE_THINKING="false"

# Use Milvus Lite without Docker. The database file is persistent under /home.
export DENOISE_MILVUS_URI="/home/humq/milvus_denoise/milvus_lite.db"

# Set these only in the current shell; do not write real API keys to this file.
# export QWEN_OPENAI_API_KEY="..."
# DashScope / 阿里云百炼示例：
# export QWEN_OPENAI_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
# export QWEN_OPENAI_MODEL_NAME="qwen-plus"
# SiliconFlow 也可使用：
# export QWEN_OPENAI_API_BASE="https://api.siliconflow.cn/v1"
# export QWEN_OPENAI_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
