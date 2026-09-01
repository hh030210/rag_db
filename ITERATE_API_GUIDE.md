# POST /case_iter/iterate

案例级 Prompt 多轮迭代优化接口。根据评估指标对 Prompt 进行多轮打分、改写、验证，直到收敛。

---

## 请求

**Header**

| 字段 | 必填 | 说明 |
|------|------|------|
| `Authorization` | 是 | `Bearer <DashScope API Key>` |

**Body（JSON）**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | string | ✅ | — | 用户原始问题 |
| `prompt` | string | ✅ | — | 首版 Prompt（可由 `/case_iter/initial_prompt` 生成） |
| `ground_truth` | string | ✅ | — | 参考答案，用于评估生成质量 |
| `search_url` | string | ❌ | `http://81.70.191.196:80/search` | Search 服务地址（接口内部自动调用） |
| `max_iterations` | int | ❌ | 5 | 最大迭代轮数（1~20） |
| `score_threshold` | float | ❌ | 85.0 | 综合得分达标线，达到即停止（0~100） |
| `min_delta` | float | ❌ | 0.5 | 收敛灵敏阈值（0~10） |

**`metrics` 子项格式**

```json
{
  "metric_name": "回答准确性",
  "description": "回答的事实与参考答案是否一致",
  "weight": 0.30
}
```

---

## 响应

**成功（HTTP 200）**

```json
{
  "ok": true,
  "code": 200,
  "data": {
    "best_prompt": "...",
    "best_score": 82.5,
    "iterations": 3,
    "stop_reason": "converged(no_gain_for_2_iters)",
    "history": [...],
    "total_llm_time": 12.34,
    "total_time": 15.67
  }
}
```

**`data` 字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| `best_prompt` | string | 迭代全程最佳 Prompt |
| `best_score` | float | 最佳综合得分（0~100） |
| `iterations` | int | 实际执行轮数 |
| `stop_reason` | string | 停止原因 |
| `history` | list | 每轮详细记录 |
| `total_llm_time` | float | LLM 调用总耗时（秒） |
| `total_time` | float | 接口总耗时（秒） |

**`stop_reason` 取值**

| 值 | 含义 |
|----|------|
| `reached_threshold(X)` | 综合得分达到阈值停止 |
| `converged(no_gain_for_N_iters)` | 连续 N 轮改进幅度过小停止 |
| `max_iterations(N)` | 达到最大迭代轮数停止 |
| `refine_no_change` | 改写未产生新内容停止 |
| `monotonic_constraint_failed(all_retries_degraded)` | 重试后仍不满足单调性停止 |
| `generate_error` | 生成答案异常停止 |

**`history` 每轮记录字段**

| 字段 | 类型 | 说明 |
|------|------|------|
| `iter` | int | 迭代轮次编号（从 0 开始） |
| `prompt` | string | 本轮使用的 Prompt |
| `answer` | string | 模型生成的答案 |
| `score` | float | 本轮综合得分 |
| `per_metric_scores` | dict | 本轮各指标得分 `{指标名: 得分}` |
| `delta` | float | 相对 best_score 的提升量 |
| `judge_time` | float | judge 打分耗时 |
| `answer_time` | float | 生成答案耗时 |

**失败（HTTP 4xx/5xx）**

```json
{
  "ok": false,
  "code": 400,
  "error": "metrics 不能为空..."
}
```

---

## 调用示例

```bash
curl -X POST http://81.70.191.196:8000/case_iter/iterate \
  -H "Authorization: Bearer sk-xxxxxxxxxxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "南孔庙几点开门？",
    "prompt": "你是一个景区导游。请基于私域上下文回答用户问题。",
    "ground_truth": "南孔庙开放时间为08:00-17:30。",
    "metrics": [
      {"metric_name": "回答准确性", "description": "事实正确", "weight": 0.50},
      {"metric_name": "格式规范性", "description": "格式好", "weight": 0.50}
    ]
  }'
```

---

## 实现逻辑

### 核心算法流程（Algorithm 1）

```
输入: query, initial_prompt, ground_truth, metrics, search_url, generate_answer_fn
输出: best_prompt, best_score, iterations, stop_reason, history

1. 初始化
   current_prompt ← initial_prompt
   best_prompt ← initial_prompt
   best_score ← -1.0
   prev_metric_scores ← None        # 上一轮各指标得分
   history ← []
   total_llm_time ← 0

2. 迭代（最多 max_iterations 轮）

   2.1 生成答案
       candidate ← generate_answer_fn(current_prompt)
       answer_time ← 耗时

   2.2 LLM-as-a-Judge 多维打分
       judge ← judge_answer(query, candidate, ground_truth, retrieved_context, metrics)
       score ← score_answer(judge, metrics)
       current_metric_scores ← 各指标得分 {指标名: 分数}

   2.3 记录历史 & 更新 best
       history.append({iter, prompt, answer, judge, score, current_metric_scores, delta, ...})
       if score > best_score:
           best_score ← score
           best_prompt ← current_prompt

   2.4 收敛判断（任一满足即停止）
       if score >= score_threshold:
           stop_reason ← "reached_threshold"
           break

       if delta < min_delta:
           consecutive_no_gain += 1
       else:
           consecutive_no_gain = 0
       if consecutive_no_gain >= score_window:
           stop_reason ← "converged(no_gain)"
           break

   2.5 单调性约束改写（核心创新点）

       # 规则：每项指标得分不得低于上一轮（允许 ±0.01 浮点容差）
       # 任一指标回退 → 触发重试（最多 3 次），仍失败则放弃改写、终止迭代

       for retry ∈ [0, 1, 2]:
           new_prompt ← refine_with_eval(current_prompt, judge)

           if new_prompt == current_prompt:
               stop_reason ← "refine_no_change"; break

           # 用新 prompt 重新打分验证
           candidate_retry ← generate_answer_fn(new_prompt)
           judge_retry ← judge_answer(..., candidate_retry, ...)
           retry_scores ← 各指标得分

           # 单调性验证
           if prev_metric_scores is not None:
               for each metric m:
                   if retry_scores[m] < prev_metric_scores[m] - 0.01:
                       标记该指标回退; 继续重试

           if 所有指标均满足单调性:
               current_prompt ← new_prompt
               prev_metric_scores ← retry_scores
               break

       if 重试全部失败:
           stop_reason ← "monotonic_constraint_failed(all_retries_degraded)"
           break

3. 返回 best_prompt, best_score, iterations, history, stop_reason, total_llm_time
```

### 关键函数说明

| 函数 | 职责 | 调用位置 |
|------|------|----------|
| `iterate_prompt_until_converged()` | Algorithm 1 主循环 | `api_server.py` 第 623 行 |
| `judge_answer()` | LLM-as-a-Judge 多维打分 | Algorithm 第 2.2 步 |
| `score_answer()` | 根据权重计算加权综合得分 | Algorithm 第 2.2 步 |
| `refine_with_eval()` | 基于反馈改写 prompt | Algorithm 第 2.5 步 |
| `_answer_with_internal_llm()` | 内部 LLM 充当 `generate_answer_fn` | `api_server.py` 第 601 行 |

### Judge 打分 Prompt 模板

```
请根据评估指标集，对【模型生成的答案】进行多维打分。

【用户问题】
{query}

【私域上下文】
{retrieved_context}

【参考答案】
{ground_truth}

【模型生成的答案（待评分）】
{candidate_answer}

【评估指标集（每项 0~100 分）】
{metrics_block}

输出 JSON：{"scores": [{"metric_name": "...", "score": 0~100, "reason": "..."}]}
```

### Refine 改写 Prompt 模板

```
请根据【当前 prompt 的多维评估反馈】，对 prompt 进行一次定向改写。

【改写要求】
1. 必须保证所有指标不低于上一轮得分：
   - 得分 < 85：必须显式补强相应约束
   - 得分 ≥ 90：保持其约束不变
   - 严禁以牺牲任一维度为代价换取综合分提升
2. 严禁新增与当前评估问题无关的冗余内容
3. 输出新 prompt 正文（中文、结构化）
```

### 加权得分计算

```python
def score_answer(judge_result, metrics):
    weight_map = {m["metric_name"]: m["weight"] for m in metrics}
    score_map  = {s["metric_name"]: s["score"] for s in judge_result["scores"]}

    total = Σ(score_map[name] * weight_map[name]) / Σ weight_map[name]
    return round(total, 2)  # 归一化到 0~100
```

---

## 注意事项

- 本接口内部使用 LLM 模拟生成答案；生产环境建议在外部调用下游模型后传入结果。
- `metrics` 建议使用 `/case_iter/metrics` 接口生成，以保证权重合理分配。
- 单调性约束确保每次改写不会让任意指标回退，是本算法保证稳定收敛的关键机制。
