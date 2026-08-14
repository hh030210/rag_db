#!/usr/bin/env bash
set -u

watch_run() {
  old_pid="$1"
  gpu="$2"
  chunks_dir="$3"
  start="$4"
  end="$5"
  output="$6"
  log="$7"
  while kill -0 "$old_pid" 2>/dev/null; do
    sleep 30
  done
  env CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /home/humq/envs/denoise_qa/bin/python -u /home/humq/chunk_code/fast_ppl_metrics.py \
    --chunks_dir "$chunks_dir" \
    --source /home/humq/chunk_code/data/db_qa.txt \
    --adapter /home/humq/chunk_code/paper_metric_adapter.py \
    --model /home/humq/hf_cache/Qwen2-1.5B-Instruct \
    --output "$output" --metrics stickiness \
    --start_chunk "$start" --end_chunk "$end" --batch_size 8 \
    > "$log" 2>&1
}

watch_run 392525 4 \
  /home/humq/chunk_code/results/denoise_ablation_db_qa_textsafe/denoise_off/docs \
  9000 13000 /home/humq/chunk_code/metric_results/fast_stickiness_off_shard2.json \
  /home/humq/logs/fast_stickiness_off_shard2.log &
watch_run 392522 5 \
  /home/humq/chunk_code/results/denoise_ablation_db_qa_textsafe/denoise_off/docs \
  13000 17309 /home/humq/chunk_code/metric_results/fast_stickiness_off_shard3.json \
  /home/humq/logs/fast_stickiness_off_shard3.log &
watch_run 394369 6 \
  /home/humq/chunk_code/results/denoise_ablation_db_qa_textsafe/denoise_on/docs \
  9000 13000 /home/humq/chunk_code/metric_results/fast_stickiness_on_shard2.json \
  /home/humq/logs/fast_stickiness_on_shard2.log &
watch_run 394486 7 \
  /home/humq/chunk_code/results/denoise_ablation_db_qa_textsafe/denoise_on/docs \
  13000 16958 /home/humq/chunk_code/metric_results/fast_stickiness_on_shard3.json \
  /home/humq/logs/fast_stickiness_on_shard3.log &
wait
