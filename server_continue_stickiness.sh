#!/usr/bin/env bash
set -u

wait_for_pid() {
  pid="$1"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
}

wait_for_pid 392525
env CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/humq/envs/denoise_qa/bin/python -u /home/humq/chunk_code/fast_ppl_metrics.py \
  --chunks_dir /home/humq/chunk_code/results/denoise_ablation_db_qa_textsafe/denoise_off/docs \
  --source /home/humq/chunk_code/data/db_qa.txt \
  --adapter /home/humq/chunk_code/paper_metric_adapter.py \
  --model /home/humq/hf_cache/Qwen2-1.5B-Instruct \
  --output /home/humq/chunk_code/metric_results/fast_stickiness_off_shard2.json \
  --metrics stickiness --start_chunk 9000 --end_chunk 13000 --batch_size 8 \
  > /home/humq/logs/fast_stickiness_off_shard2.log 2>&1 &

wait_for_pid 392522
env CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/humq/envs/denoise_qa/bin/python -u /home/humq/chunk_code/fast_ppl_metrics.py \
  --chunks_dir /home/humq/chunk_code/results/denoise_ablation_db_qa_textsafe/denoise_off/docs \
  --source /home/humq/chunk_code/data/db_qa.txt \
  --adapter /home/humq/chunk_code/paper_metric_adapter.py \
  --model /home/humq/hf_cache/Qwen2-1.5B-Instruct \
  --output /home/humq/chunk_code/metric_results/fast_stickiness_off_shard3.json \
  --metrics stickiness --start_chunk 13000 --end_chunk 17309 --batch_size 8 \
  > /home/humq/logs/fast_stickiness_off_shard3.log 2>&1 &

wait_for_pid 394369
env CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/humq/envs/denoise_qa/bin/python -u /home/humq/chunk_code/fast_ppl_metrics.py \
  --chunks_dir /home/humq/chunk_code/results/denoise_ablation_db_qa_textsafe/denoise_on/docs \
  --source /home/humq/chunk_code/data/db_qa.txt \
  --adapter /home/humq/chunk_code/paper_metric_adapter.py \
  --model /home/humq/hf_cache/Qwen2-1.5B-Instruct \
  --output /home/humq/chunk_code/metric_results/fast_stickiness_on_shard2.json \
  --metrics stickiness --start_chunk 9000 --end_chunk 13000 --batch_size 8 \
  > /home/humq/logs/fast_stickiness_on_shard2.log 2>&1 &

wait_for_pid 394486
env CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/humq/envs/denoise_qa/bin/python -u /home/humq/chunk_code/fast_ppl_metrics.py \
  --chunks_dir /home/humq/chunk_code/results/denoise_ablation_db_qa_textsafe/denoise_on/docs \
  --source /home/humq/chunk_code/data/db_qa.txt \
  --adapter /home/humq/chunk_code/paper_metric_adapter.py \
  --model /home/humq/hf_cache/Qwen2-1.5B-Instruct \
  --output /home/humq/chunk_code/metric_results/fast_stickiness_on_shard3.json \
  --metrics stickiness --start_chunk 13000 --end_chunk 16958 --batch_size 8 \
  > /home/humq/logs/fast_stickiness_on_shard3.log 2>&1 &
