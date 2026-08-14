"""
compare_fusion_methods.py

对 7 个实验文件的 fusion 检索结果进行 LLM 语义排序比较。

功能：
  - 加载 7 个 fusion 实验结果 JSON
  - 按 qid 匹配各方法的 qa_result.qa_context
  - 将 7 种 fusion 的 qa_context 拼接后发给阿里云 DashScope LLM 进行语义排名
  - 汇总各方法的胜率、平均排名等指标

使用方法：
  python compare_fusion_methods.py
    --input_dir output/unified_retrieval
    --files events_ws2_test_k5.json events_ws4_test_k5.json merged_chunk_200_test_k5.json
             merged_chunk_300_test_k5.json merged_chunk_400_test_k5.json merged_chunk_500_test_k5.json
             unified_test_collection_k5.json
    --output output/fusion_llm_comparison.json
    --max_workers 2
"""

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import dashscope
import yaml

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

# ---- Milvus / BGE 检索依赖 ----
from pymilvus import Collection, connections, utility
from DB_GM.generated_code.search_vec_all_in_one import SearchEngine_vec_all_in_one

# ---- DashScope LLM 配置（从 db_config.yaml qgen 字段读取） ----
try:
    from db_config import get_config
    _cfg = get_config()
    _qgen_cfg = getattr(_cfg, 'qgen', None)
except Exception:
    _qgen_cfg = None


def _resolve_dashscope_cfg():
    if _qgen_cfg is not None:
        cfg_api_key  = getattr(_qgen_cfg, 'api_key',  None) or ""
        cfg_base_url = getattr(_qgen_cfg, 'base_url',  "") or ""
        cfg_model    = getattr(_qgen_cfg, 'model',      "") or "qwen-plus"
    else:
        cfg_api_key = cfg_base_url = cfg_model = ""

    api_key = os.getenv("DASHSCOPE_API_KEY") or cfg_api_key or ""
    base_url = os.getenv("DASHSCOPE_BASE_URL") or cfg_base_url or ""
    model = os.getenv("DASHSCOPE_MODEL_NAME") or cfg_model or "qwen-plus"

    if base_url:
        dashscope.base_http_api_url = base_url.rstrip("/")

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }


_ds_cfg = _resolve_dashscope_cfg()
DS_API_KEY = _ds_cfg["api_key"]
DS_MODEL   = _ds_cfg["model"]

DS_MAX_WORKERS  = int(os.getenv("LLM_EVAL_MAX_WORKERS", "2"))
DS_MAX_RETRIES  = int(os.getenv("LLM_EVAL_MAX_RETRIES", "6"))
DS_RETRY_SLEEP  = float(os.getenv("LLM_EVAL_RETRY_SLEEP_SECONDS", "2.0"))
DS_MIN_INTERVAL = float(os.getenv("LLM_EVAL_MIN_REQUEST_INTERVAL_SECONDS", "0.5"))
# 每个方法的 qa_context 最大字符数（7个方法相加不能超出 LLM 上下文窗口）
DS_MAX_CHARS_PER_METHOD = int(os.getenv("LLM_EVAL_MAX_CHARS_PER_METHOD", "8000"))

# ---- Milvus / 检索配置（从 db_config.yaml 读取） ----
try:
    from db_config import get_config
    _vec_cfg = get_config().vecdb
    MILVUS_HOST = getattr(_vec_cfg, 'host', '127.0.0.1')
    MILVUS_PORT = str(getattr(_vec_cfg, 'port', '19532'))
except Exception:
    MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19532")

MILVUS_CONN_ALIAS = "compare_fusion_methods"


def _ensure_milvus_connection():
    """建立 Milvus 连接（全局复用，同进程只连一次）"""
    if not connections.has_connection(MILVUS_CONN_ALIAS):
        connections.connect(MILVUS_CONN_ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)


# ---- 语义检索器：复用 retrieval_fusion_eval.py 的 SearchEngine ----

class SemanticRetriever:
    """
    语义检索器，对标 retrieval_fusion_eval.py 中的 SemanticSearcher。
    使用 BGE-M3 向量 + Milvus ANN 搜索，从指定的 collection 中检索 top-k chunks。
    """

    def __init__(self, collection_name: str, anns_field: str = "chunk_text_vec"):
        self.collection_name = collection_name
        self.anns_field = anns_field
        self.engine = SearchEngine_vec_all_in_one(collection_name=collection_name)

    def search(self, query_text: str, top_k: int = 1) -> Dict[str, Any]:
        """
        语义检索，返回 top_k 条结果。
        与 retrieval_fusion_eval.py SemanticSearcher.search() 保持一致。

        返回字段：
          chunk_id, doc_id_link, score, doc_title, chunk_gen_title,
          chunk_text, chunk_text_full
        """
        try:
            results = self.engine.search(
                query_text,
                top_k=top_k,
                anns_field=self.anns_field,
            )
        except Exception as e:
            return {"query": query_text, "top_k": top_k, "results": [], "error": str(e)}

        output = []
        for rank, r in enumerate(results, 1):
            chunk_text_full = r.get("chunk_text_full") or r.get("chunk_text") or ""
            output.append({
                "rank": rank,
                "chunk_id": str(r.get("chunk_id", "")),
                "doc_id_link": str(r.get("doc_id_link", "")),
                "score": round(float(r.get("score", 0.0)), 6),
                "doc_title": r.get("doc_title", "") or "",
                "chunk_gen_title": r.get("chunk_gen_title", "") or "",
                "chunk_text": chunk_text_full[:200].replace("\n", " ") if chunk_text_full else "",
                "chunk_text_full": chunk_text_full,
            })

        return {"query": query_text, "top_k": top_k, "results": output}


# ---- 检索结果缓存：method -> qid -> retrieval_result ----
_RETRIEVAL_CACHE: Dict[str, Dict] = {}
_CACHE_LOCK = threading.Lock()


class _RetrievalTask:
    """封装检索任务，用于并发提交"""

    def __init__(
        self,
        method_name: str,
        collection_name: str,
        qid: str,
        query_text: str,
        top_k: int,
        retriever: SemanticRetriever,
    ):
        self.method_name = method_name
        self.collection_name = collection_name
        self.qid = qid
        self.query_text = query_text
        self.top_k = top_k
        self.retriever = retriever

    def run(self) -> Dict[str, Any]:
        cache_key = f"{self.method_name}|{self.qid}"
        with _CACHE_LOCK:
            if cache_key in _RETRIEVAL_CACHE:
                return _RETRIEVAL_CACHE[cache_key]

        result = self.retriever.search(self.query_text, top_k=self.top_k)
        with _CACHE_LOCK:
            _RETRIEVAL_CACHE[cache_key] = result
        return result


def concurrent_retrieve(
    method_names: List[str],
    collection_names: Dict[str, str],
    qid_query_map: Dict[str, str],
    top_k: int = 1,
    max_workers: int = 4,
) -> None:
    """
    并发执行所有语义检索，填充全局缓存 _RETRIEVAL_CACHE。
    method_names: 方法名列表
    collection_names: {method_name: collection_name}
    qid_query_map: {qid: query_text}
    """
    tasks = []
    retrievers: Dict[str, SemanticRetriever] = {}

    for method_name in method_names:
        coll = collection_names.get(method_name, "")
        if not coll:
            print(f"  [WARN] method {method_name} 没有 collection，跳过")
            continue
        retrievers[method_name] = SemanticRetriever(coll)

    valid_methods = [m for m in method_names if m in retrievers]
    for qid, query_text in qid_query_map.items():
        for method_name in valid_methods:
            tasks.append(_RetrievalTask(
                method_name=method_name,
                collection_name=collection_names[method_name],
                qid=qid,
                query_text=query_text,
                top_k=top_k,
                retriever=retrievers[method_name],
            ))

    print(f"\n{'='*70}")
    print(f"  语义检索阶段  |  收集 {len(tasks)} 个检索任务  |  并发 {max_workers}")
    print(f"{'='*70}")

    completed = 0
    errors = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(t.run): t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                future.result()
                completed += 1
            except Exception as e:
                errors += 1
                print(f"  [Retrieval Error] {task.method_name}/{task.qid}: {e}")

            if (completed + errors) % 200 == 0 or (completed + errors) == len(tasks):
                print(f"  检索进度: {completed + errors}/{len(tasks)} | 成功: {completed} | 错误: {errors}")

    print(f"  检索完成  |  成功: {completed}  |  错误: {errors}")


def get_cached_retrieval(method_name: str, qid: str) -> Dict[str, Any]:
    """从缓存读取检索结果，无缓存返回空"""
    return _RETRIEVAL_CACHE.get(f"{method_name}|{qid}", {})


class _RateLimiter:
    def __init__(self, min_interval_seconds: float):
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._lock = threading.Lock()
        self._next_allowed_time = 0.0

    def wait(self):
        if self.min_interval_seconds <= 0:
            return
        sleep_for = 0.0
        with self._lock:
            now = time.time()
            if now < self._next_allowed_time:
                sleep_for = self._next_allowed_time - now
                now = self._next_allowed_time
            self._next_allowed_time = now + self.min_interval_seconds
        if sleep_for > 0:
            time.sleep(sleep_for)


_DS_RATE_LIMITER = _RateLimiter(DS_MIN_INTERVAL)


def call_dashscope(messages, temperature=0.0):
    """调用阿里云 DashScope API（dashscope SDK）"""
    if not DS_API_KEY:
        raise RuntimeError("MISSING_DASHSCOPE_API_KEY")

    last_error = None
    for attempt in range(DS_MAX_RETRIES):
        try:
            _DS_RATE_LIMITER.wait()
            resp = dashscope.Generation.call(
                api_key=DS_API_KEY,
                model=DS_MODEL,
                messages=messages,
                temperature=float(temperature),
                result_format="message",
            )

            if resp.status_code == 200:
                return resp.output["choices"][0]["message"]["content"]

            if resp.status_code == 429:
                backoff = (DS_RETRY_SLEEP * (2 ** attempt)) + random.uniform(0, 0.5)
                time.sleep(min(backoff, 60.0))
                last_error = f"HTTP_{resp.status_code}_429"
                continue

            if 500 <= resp.status_code < 600:
                backoff = (DS_RETRY_SLEEP * (2 ** attempt)) + random.uniform(0, 0.5)
                time.sleep(min(backoff, 60.0))
                last_error = f"HTTP_{resp.status_code}"
                continue

            last_error = f"HTTP_{resp.status_code} - {getattr(resp, 'message', '')}"
            time.sleep(DS_RETRY_SLEEP * (attempt + 1))

        except Exception as e:
            last_error = str(e)
            time.sleep(DS_RETRY_SLEEP * (attempt + 1))

    raise RuntimeError(f"DASHSCOPE_API_FAILED:{last_error}")


# ---- Prompt ----

RANKING_PROMPT_TEMPLATE = """|# Role
你是一个专业的检索质量评估专家。

# Task
给定一个用户问题（Query）、标准答案来源（Sources）以及 N 种融合检索方法返回的 QA 上下文（Context），
请根据以下核心维度对各方法进行排名。

# 核心评估维度（按优先级排序）
1. **Source 命中**：context 是否**直接包含** Sources 中列出的关键事实
   - 如果 Sources 中提到的人物、年份、地点、数量等在 context 中**逐字出现或同义出现**，则命中
   - 命中的 source 越多，排名越靠前
2. **Query 覆盖度**：context 是否能**完整回答** Query 提出的所有问题
   - 能完整回答 → 排名靠前
   - 只能回答部分 → 排名中等
   - 无法回答 → 排名靠后

# 次要考量（可忽略）
- 冗余信息：context 中包含与 Query 和 Sources 无关的内容，**不影响排名**
- 上下文长度：不作为排名依据

# Context 特殊标记说明
- 融合 Top-1 模式：Context 开头会标注 `[融合 Top-1 | chunk_id: xxx | dim_rank: x | sem_rank: x]`
- dim_rank / sem_rank 仅表示该 chunk 在各自检索中的排名，不影响最终排名

# 评分规则
| 情况 | 评级 | 排名 |
|------|------|------|
| 包含全部 Sources + 完整回答 Query | 高质量 | 靠前 |
| 包含部分 Sources + 部分回答 Query | 中等质量 | 中等 |
| Source 命中少 + 难以回答 Query | 低质量 | 靠后 |
| 无有效检索结果（context 为空/无内容） | 无效 | 最后 |

# 重要约束
- **必须对全部 N 种方法进行排名，每种方法出现且仅出现一次**
- reason 需简要说明该方法命中了哪些 Sources 或缺失了什么（30字以内）
- 输出格式严格为纯 JSON，无任何 Markdown 标记

# 输出格式
{{
  "ranking": [
    {{"rank": 1, "method": "<方法名>", "reason": "<理由>"}},
    {{"rank": 2, "method": "<方法名>", "reason": "<理由>"}},
    ...
  ],
  "summary": "<总体分析，50字以内>"
}}

# 输入
Query: {query}

Sources（标准答案来源，context 应尽可能包含以下事实）:
{sources_text}

Methods (每种方法的上下文):
{methods_text}
"""


def build_sources_text(sources: List[str]) -> str:
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(sources))


def build_methods_text(methods: Dict[str, str]) -> str:
    """methods: {method_name: context_text}"""
    lines = []
    for name, ctx in methods.items():
        ctx_display = ctx.strip() if ctx.strip() else "[无有效检索结果]"
        if len(ctx_display) > DS_MAX_CHARS_PER_METHOD:
            ctx_display = ctx_display[:DS_MAX_CHARS_PER_METHOD] + "..."
        lines.append(f"[{name}]\n{ctx_display}")
    return "\n\n".join(lines)


def build_prompt(query: str, sources: List[str], methods: Dict[str, str]) -> List[Dict[str, str]]:
    sources_text = build_sources_text(sources)
    methods_text = build_methods_text(methods)

    prompt = RANKING_PROMPT_TEMPLATE.format(
        query=query,
        sources_text=sources_text,
        methods_text=methods_text,
    )
    return [{"role": "user", "content": prompt}]


# ---- 结果解析 ----

def parse_ranking_response(content: str, expected_count: int) -> Optional[Dict[str, Any]]:
    """从 LLM 返回中解析 JSON 排名结果
    
    Args:
        content: LLM 返回的原始内容
        expected_count: 期望的方法数量（从命令行参数传入）
    """
    try:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE).strip()
            content = re.sub(r"\s*```$", "", content, flags=re.IGNORECASE).strip()

        result = json.loads(content)

        if "ranking" not in result or "summary" not in result:
            return None

        # 动态检查方法数量（允许少一些但不能多）
        actual_count = len(result["ranking"])
        if actual_count < expected_count * 0.5:  # 至少要有 50% 的方法
            return None

        for item in result["ranking"]:
            if "rank" not in item or "method" not in item:
                return None

        # 如果实际数量少于期望，补充缺失的排名
        if actual_count < expected_count:
            ranked_methods = {item["method"] for item in result["ranking"]}
            all_methods = None  # 需要从外部传入，这里无法访问
            # 已有足够信息，标记为部分结果
            result["_partial"] = True

        return result
    except json.JSONDecodeError as e:
        # JSON 不完整，尝试修复：补全末尾的 ] 和 }
        try:
            fixed = content
            if not fixed.endswith("]"):
                # 找到最后一个完整的对象
                last_brace = fixed.rfind("},")
                if last_brace != -1:
                    fixed = fixed[:last_brace + 1] + "]"
                # 尝试补全 summary
                if '"summary":' not in fixed:
                    fixed = fixed.rstrip(", \n") + ',\n  "summary": ""}'
                else:
                    fixed = fixed.rstrip(", \n") + "}"
            result = json.loads(fixed)
            if "ranking" in result and "summary" in result:
                actual_count = len(result["ranking"])
                if actual_count >= expected_count * 0.5:
                    result["_truncated"] = True
                    return result
        except Exception:
            pass
        return None
    except Exception:
        return None


# ---- 主评估逻辑 ----

def evaluate_single_query(args) -> Dict[str, Any]:
    """对单条 query 进行 N 路 fusion 排名评估"""
    qid, query, sources, methods, max_retries = args
    expected_count = len(methods)

    prompt = build_prompt(query, sources, methods)
    last_error = None
    last_raw_content = None

    for attempt in range(max_retries):
        try:
            content = call_dashscope(prompt, temperature=0.1)
            parsed = parse_ranking_response(content, expected_count)

            if parsed is not None:
                return {
                    "qid": qid,
                    "query": query,
                    "sources": sources,
                    "ranking": parsed["ranking"],
                    "summary": parsed["summary"],
                    "error": None,
                    "_raw_content": content if parsed.get("_truncated") or parsed.get("_partial") else None,
                }

            last_error = f"PARSE_FAILED: {content[:500]}"
            last_raw_content = content
        except Exception as e:
            last_error = str(e)
            last_raw_content = None

        if attempt < max_retries - 1:
            time.sleep(DS_RETRY_SLEEP * (attempt + 1))

    return {
        "qid": qid,
        "query": query,
        "sources": sources,
        "ranking": None,
        "summary": None,
        "error": last_error,
        "_raw_content": last_raw_content if last_error else None,
    }


def load_fusion_results(file_path: Path) -> Dict[str, Any]:
    """加载单个实验 JSON，返回 method_name → qid → qa_context 的映射
    
    method_name 生成规则：
    - 优先使用文件的特殊名称映射（如 ours_test_collection_k5.json -> fusion_test_collection）
    - 其次使用 config.dim_collection（如 merged_chunk_200 -> fusion_merged_chunk_200）
    - 最后使用文件名（不含 _test 后缀）
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    config = data.get("config", {})
    
    # 特殊映射：文件名 -> method_name 后缀（用于区分同一 collection 的不同实验）
    _file_method_suffix_map = {
        "ours_test_collection_k5.json": "test_collection",      # 原 test_collection 实验
        "unified_test_collection_k5.json": "nankong_chunks",    # 实际对应 nankong_chunks
    }
    
    fname = Path(file_path).name
    if fname in _file_method_suffix_map:
        method_suffix = _file_method_suffix_map[fname]
    else:
        method_suffix = config.get("dim_collection", file_path.stem.replace("_test_k5", ""))
    
    method_name = f"fusion_{method_suffix}"

    detailed = data.get("detailed_results", [])
    qid_map = {}
    for entry in detailed:
        qid = entry.get("qid")
        if qid is None:
            continue
        qid = str(qid)
        qa_result = entry.get("qa_result", {})
        qa_context = qa_result.get("qa_context", "")
        qid_map[qid] = {
            "method_name": method_name,
            "query": entry.get("query", ""),
            "sources": entry.get("sources", []),
            "qa_context": qa_context,
            "dim_hit": entry.get("dim_hit", False),
            "sem_hit": entry.get("sem_hit", False),
            "fusion_hit": entry.get("fusion_hit", False),
            "dim_recall": entry.get("dim_recall", 0.0),
            "sem_recall": entry.get("sem_recall", 0.0),
            "fusion_recall": entry.get("fusion_recall", 0.0),
            # 保留原始检索结果用于融合 top-k 分析
            "dim_results": entry.get("dim_results", []),
            "sem_results": entry.get("sem_results", []),
            "fusion_results": entry.get("fusion_results", []),
        }

    return method_name, qid_map


def compute_global_ranking(query_results: List[Dict], method_names: List[str]) -> Dict[str, Any]:
    """
    基于 Borda Count 计算全局综合排名。
    每种方法在每个 query 中获得的 Borda 分数 = (n - rank + 1)，n=7种方法
    总分越高越优，最后按总分排序得到 global_rank。
    """
    n_methods = len(method_names)

    # 方法名 → Borda 总分
    borda_scores = {m: 0 for m in method_names}
    # 方法名 → 胜出次数（排名第1）
    win_counts = {m: 0 for m in method_names}
    # 方法名 → 各 rank 的计数
    rank_counts = {m: {r: 0 for r in range(1, n_methods + 1)} for m in method_names}
    # 方法名 → 所有 query 的排名列表
    rank_lists = {m: [] for m in method_names}

    valid = 0
    for qr in query_results:
        if qr.get("error") or not qr.get("ranking"):
            continue
        valid += 1
        for item in qr["ranking"]:
            m = item["method"]
            r = item["rank"]
            if m in borda_scores:
                borda_scores[m] += (n_methods - r + 1)
                rank_lists[m].append(r)
                rank_counts[m][r] = rank_counts[m].get(r, 0) + 1
                if r == 1:
                    win_counts[m] += 1

    # 按 Borda 总分排序得到全局排名
    sorted_methods = sorted(method_names, key=lambda m: borda_scores[m], reverse=True)
    global_ranking = []
    prev_score = None
    prev_global_rank = 0
    for i, m in enumerate(sorted_methods):
        global_rank = i + 1
        if borda_scores[m] == prev_score:
            global_rank = prev_global_rank
        else:
            prev_global_rank = global_rank
        prev_score = borda_scores[m]

        ranks = rank_lists[m]
        avg_rank = round(sum(ranks) / len(ranks), 4) if ranks else 0.0
        total = sum(rank_counts[m].values()) if sum(rank_counts[m].values()) > 0 else 1

        global_ranking.append({
            "global_rank": global_rank,
            "method": m,
            "borda_score": borda_scores[m],
            "win_count": win_counts[m],
            "win_rate": round(win_counts[m] / total, 4),
            "avg_rank": avg_rank,
            "rank_distribution": {str(k): v for k, v in sorted(rank_counts[m].items())},
            "score_per_query": round(borda_scores[m] / total, 2),
        })

    return {
        "global_ranking": global_ranking,
        "borda_scores": borda_scores,
        "win_counts": win_counts,
    }


def aggregate_results(query_results: List[Dict], method_names: List[str]) -> Dict[str, Any]:
    """汇总各方法的胜率、平均排名等指标"""
    all_ranks = {m: [] for m in method_names}
    win_counts = {m: 0 for m in method_names}
    rank2_counts = {m: 0 for m in method_names}
    rank3_counts = {m: 0 for m in method_names}
    last_counts = {m: 0 for m in method_names}
    valid_count = 0
    n_methods = len(method_names)

    for qr in query_results:
        if qr.get("error") or not qr.get("ranking"):
            continue
        valid_count += 1
        for item in qr["ranking"]:
            m = item["method"]
            r = item["rank"]
            if m in all_ranks:
                all_ranks[m].append(r)
                if r == 1:
                    win_counts[m] += 1
                elif r == 2:
                    rank2_counts[m] += 1
                elif r == 3:
                    rank3_counts[m] += 1
                if r == n_methods:
                    last_counts[m] += 1

    total = valid_count if valid_count > 0 else 1

    avg_rank = {}
    for m in method_names:
        ranks = all_ranks[m]
        avg_rank[m] = round(sum(ranks) / len(ranks), 4) if ranks else 0.0

    win_rate = {}
    for m in method_names:
        win_rate[m] = round(win_counts[m] / total, 4)

    # dim/sem/fusion 对比统计
    collection_stats = {}
    for m in method_names:
        parts = m.split("_", 2)
        if len(parts) >= 3 and parts[0] == "fusion":
            coll = parts[1]
            if coll not in collection_stats:
                collection_stats[coll] = {}
            collection_stats[coll][m] = {
                "avg_rank": avg_rank[m],
                "win_count": win_counts[m],
                "win_rate": win_rate[m],
            }

    return {
        "total_queries": len(query_results),
        "valid_count": valid_count,
        "win_count": win_counts,
        "rank2_count": rank2_counts,
        "rank3_count": rank3_counts,
        "last_count": last_counts,
        "avg_rank": avg_rank,
        "win_rate": win_rate,
        "by_collection": collection_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM 语义排序比较 7 种 fusion 检索方法")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="output/unified_retrieval",
        help="输入 JSON 文件所在目录",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "events_ws2_test_k5.json",
            "events_ws4_test_k5.json",
            "merged_chunk_200_test_k5.json",
            "merged_chunk_300_test_k5.json",
            "merged_chunk_400_test_k5.json",
            "merged_chunk_500_test_k5.json",
            "unified_test_collection_k5.json",
        ],
        help="要比较的 JSON 文件列表",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/fusion_llm_comparison.json",
        help="输出结果路径",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="LLM 并发数",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从已有输出文件恢复（跳过已评估的 qid）",
    )
    # ---- 检索模式参数 ----
    parser.add_argument(
        "--use_retrieval",
        action="store_true",
        help="开启语义检索模式：直接查 Milvus collection 而非使用实验 JSON 中的 qa_context",
    )
    parser.add_argument(
        "--retrieval_top_k",
        type=int,
        default=1,
        help="检索模式下的 top_k（默认 1，即只取最相关的一条）",
    )
    parser.add_argument(
        "--retrieval_cache",
        type=str,
        default="",
        help="检索结果缓存文件路径（便于断点恢复），不指定则不缓存",
    )
    parser.add_argument(
        "--fusion_top1",
        action="store_true",
        help="使用 JSON 中融合检索结果的 top-1 chunk_text 进行评估（不使用 qa_context）",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 加载所有文件 ----
    print(f"\n{'='*70}")
    print(f"  加载实验文件")
    print(f"{'='*70}")

    all_data = {}  # method_name -> qid_map
    method_names = []

    for fname in args.files:
        fpath = input_dir / fname
        if not fpath.exists():
            print(f"  [WARN] 文件不存在，跳过: {fpath}")
            continue

        method_name, qid_map = load_fusion_results(fpath)
        all_data[method_name] = qid_map
        method_names.append(method_name)
        print(f"  [{method_name}] 加载了 {len(qid_map)} 条样本: {fpath.name}")

    if len(all_data) < 2:
        print("  [ERROR] 有效方法少于 2 种，退出。")
        return

    # ---- 构建评估任务 ----
    # 找到所有方法的公共 qid
    if method_names:
        common_qids = set(all_data[method_names[0]].keys())
        for mn in method_names[1:]:
            common_qids &= set(all_data[mn].keys())
        common_qids = sorted(common_qids)
        print(f"\n  公共 qid 数量: {len(common_qids)}")
    else:
        common_qids = []

    # ---- 建立 method_name -> collection_name 的映射 ----
    # 从文件名直接映射（与 load_fusion_results 中的逻辑保持一致）
    collection_names: Dict[str, str] = {}
    for fname in args.files:
        # 文件名 -> method_name -> collection
        method_suffix_map = {
            "ours_test_collection_k5.json": "test_collection",
            "unified_test_collection_k5.json": "nankong_chunks",
        }
        if fname in method_suffix_map:
            method_name = f"fusion_{method_suffix_map[fname]}"
            collection_names[method_name] = method_suffix_map[fname]

    # 只有 collection_names 中存在的 method 才会参与检索和评估
    valid_methods = [m for m in method_names if m in collection_names]

    # ---- 检索模式：先批量查 Milvus，再组装评估任务 ----
    if args.use_retrieval:
        _ensure_milvus_connection()

        # 构建 qid -> query_text 映射
        qid_query_map: Dict[str, str] = {}
        for qid in common_qids:
            entry0 = all_data[method_names[0]].get(qid, {})
            qid_query_map[qid] = entry0.get("query", "")

        # 尝试从缓存文件加载已有检索结果
        retrieval_cache_file = Path(args.retrieval_cache) if args.retrieval_cache else None
        if retrieval_cache_file and retrieval_cache_file.exists():
            try:
                with open(retrieval_cache_file, encoding="utf-8") as f:
                    cached = json.load(f)
                for key, val in cached.items():
                    _RETRIEVAL_CACHE[key] = val
                print(f"  [Retrieval] 从缓存加载了 {len(cached)} 条记录")
            except Exception as e:
                print(f"  [Retrieval] 缓存加载失败: {e}，重新检索")

        # 统计需要新检索的任务数（只统计当前任务范围内的缓存）
        total_needed = len(valid_methods) * len(common_qids)
        needed_keys = {f"{mn}|{qid}" for mn in valid_methods for qid in common_qids}
        already_cached = len(needed_keys & set(_RETRIEVAL_CACHE.keys()))
        print(f"\n  检索缓存: {already_cached}/{total_needed} 已命中")

        if already_cached < total_needed:
            # 执行并发语义检索
            concurrent_retrieve(
                method_names=valid_methods,
                collection_names=collection_names,
                qid_query_map=qid_query_map,
                top_k=args.retrieval_top_k,
                max_workers=args.max_workers,
            )

            # 保存检索缓存
            if retrieval_cache_file:
                try:
                    retrieval_cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(retrieval_cache_file, "w", encoding="utf-8") as f:
                        json.dump(_RETRIEVAL_CACHE, f, ensure_ascii=False)
                    print(f"  [Retrieval] 缓存已保存至 {retrieval_cache_file}")
                except Exception as e:
                    print(f"  [Retrieval] 缓存保存失败: {e}")

        # 从缓存构建评估任务（用检索结果替代 qa_context）
        eval_tasks = []
        for qid in common_qids:
            entry0 = all_data[method_names[0]].get(qid, {})
            query = entry0.get("query", "")
            sources = entry0.get("sources", [])

            methods_context: Dict[str, str] = {}
            for mn in valid_methods:
                cached_result = get_cached_retrieval(mn, qid)
                results_list = cached_result.get("results", [])
                if results_list:
                    # 将 top_k 条 chunk 拼接为完整上下文
                    chunks = []
                    for rank, item in enumerate(results_list, 1):
                        ctx = item.get("chunk_text_full", "") or item.get("chunk_text", "")
                        chunk_id = item.get("chunk_id", "")
                        score = item.get("score", 0.0)
                        chunks.append(f"[Chunk-{rank} | id: {chunk_id} | score: {score}]\n{ctx}")
                    methods_context[mn] = "\n\n".join(chunks)
                else:
                    methods_context[mn] = ""

            eval_tasks.append((qid, query, sources, methods_context, DS_MAX_RETRIES))

        retrieval_mode_note = f"（检索模式：top-{args.retrieval_top_k} 语义检索）"
    else:
        # ---- 原始模式：从实验 JSON 的 qa_result.qa_context 或 fusion_results 取值 ----
        eval_tasks = []
        for qid in common_qids:
            entry = all_data[method_names[0]][qid]
            query = entry.get("query", "")
            sources = entry.get("sources", [])

            methods_context = {}
            for mn in method_names:
                qid_entry = all_data[mn].get(qid, {})
                if args.fusion_top1:
                    # 使用 JSON 中融合检索结果的 top-1 chunk_text
                    fusion_results = qid_entry.get("fusion_results", [])
                    if fusion_results and len(fusion_results) > 0:
                        top1_result = fusion_results[0]
                        ctx = top1_result.get("chunk_text", "")
                        chunk_id = top1_result.get("chunk_id", "")
                        dim_rank = top1_result.get("dim_rank", 0)
                        sem_rank = top1_result.get("sem_rank", 0)
                        # 添加 rank 信息便于评估参考
                        methods_context[mn] = f"[融合 Top-1 | chunk_id: {chunk_id} | dim_rank: {dim_rank} | sem_rank: {sem_rank}]\n{ctx}"
                    else:
                        methods_context[mn] = "[融合结果为空]"
                else:
                    qa_context = qid_entry.get("qa_context", "")
                    methods_context[mn] = qa_context

            eval_tasks.append((qid, query, sources, methods_context, DS_MAX_RETRIES))

        if args.fusion_top1:
            retrieval_mode_note = "（融合 Top-1 模式：使用 JSON 中融合检索 top-1 chunk）"
        else:
            retrieval_mode_note = ""

    print(f"  待评估 query 总数: {len(eval_tasks)}")
    print(f"  LLM 模型: {DS_MODEL}")
    print(f"  并发数: {args.max_workers} {retrieval_mode_note}")

    # ---- 断点恢复 ----
    existing_results = []
    processed_qids = set()
    if args.resume and output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                existing = json.load(f)
            existing_results = existing.get("query_results", [])
            for qr in existing_results:
                if "qid" in qr:
                    processed_qids.add(qr["qid"])
            print(f"  断点恢复: 已跳过 {len(processed_qids)} 条已有结果")
        except Exception as e:
            print(f"  断点恢复读取失败: {e}, 从头开始")

    pending_tasks = [t for t in eval_tasks if t[0] not in processed_qids]
    print(f"  本次实际评估: {len(pending_tasks)} 条")

    # ---- 并发评估 ----
    results_lock = threading.Lock()
    results_so_far = list(existing_results)

    print(f"\n{'='*70}")
    print(f"  LLM 评估阶段")
    print(f"{'='*70}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(evaluate_single_query, task): task[0]
            for task in pending_tasks
        }

        completed = 0
        errors_this_run = 0

        for future in concurrent.futures.as_completed(futures):
            qid = futures[future]
            try:
                result = future.result()
                with results_lock:
                    results_so_far.append(result)
                    if result.get("error"):
                        errors_this_run += 1
                        err_msg = str(result["error"])
                        # 解析失败时显示更多内容
                        if "PARSE_FAILED" in err_msg:
                            print(f"  [错误] qid={qid}: {err_msg[:600]}")
                        else:
                            print(f"  [错误] qid={qid}: {err_msg[:200]}")
                        # 如果有原始内容，保存到 _raw_content
                        if result.get("_raw_content"):
                            result["_raw_content"] = result["_raw_content"][:2000]
                    completed += 1

                if completed % 20 == 0 or completed == len(pending_tasks):
                    # 打印最近的错误信息（便于排查）
                    err_results = [r for r in results_so_far if r.get("error")]
                    if err_results:
                        last_err = err_results[-1]
                        err_msg = str(last_err["error"])
                        if "PARSE_FAILED" in err_msg:
                            print(f"  [最近错误] qid={last_err['qid']}: {err_msg[:600]}")
                        else:
                            print(f"  [最近错误] qid={last_err['qid']}: {err_msg[:150]}")

                    print(f"  进度: {completed}/{len(pending_tasks)} | "
                          f"有效: {completed - errors_this_run} | "
                          f"错误: {errors_this_run}")

                    # 增量保存
                    agg = aggregate_results(results_so_far, method_names)
                    grk = compute_global_ranking(results_so_far, method_names)
                    tmp = {
                        "config": {
                            "methods": method_names,
                            "collections": collection_names,
                            "use_retrieval": args.use_retrieval,
                            "retrieval_top_k": args.retrieval_top_k,
                            "model": DS_MODEL,
                        },
                        "query_results": list(results_so_far),
                        "metrics": agg,
                        "global_ranking": grk,
                    }
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(tmp, f, ensure_ascii=False, indent=2)

            except Exception as e:
                with results_lock:
                    results_so_far.append({
                        "qid": qid,
                        "ranking": None,
                        "summary": None,
                        "error": str(e),
                    })
                    errors_this_run += 1
                    completed += 1
                print(f"  [{qid}] 异常: {e}")

    # ---- 最终汇总 ----
    final_agg = aggregate_results(results_so_far, method_names)
    global_rank = compute_global_ranking(results_so_far, method_names)

    final_output = {
        "config": {
            "methods": method_names,
            "collections": collection_names,
            "use_retrieval": args.use_retrieval,
            "retrieval_top_k": args.retrieval_top_k,
            "model": DS_MODEL,
        },
        "query_results": results_so_far,
        "metrics": final_agg,
        "global_ranking": global_rank,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    # ---- 打印汇总 ----
    print(f"\n{'='*70}")
    print(f"  评估完成  |  有效样本: {final_agg['valid_count']}/{final_agg['total_queries']}")
    print(f"{'='*70}")

    # 1. 每种方法的胜率、平均排名、分布统计
    print(f"\n{'方法':<35} {'胜率(Rank#1)':<14} {'Avg Rank':<12} {'Rank#2':<10} {'Rank#3':<10}")
    print("-" * 85)
    for m in method_names:
        print(f"  {m:<33} {final_agg['win_rate'][m]:<14.4f} "
              f"{final_agg['avg_rank'][m]:<12.4f} "
              f"{final_agg['rank2_count'][m]:<10} "
              f"{final_agg['rank3_count'][m]:<10}")

    # 2. 全局综合排名（Borda Count）
    print(f"\n{'='*70}")
    print(f"  全局综合排名（Borda Count）")
    print(f"{'='*70}")
    print(f"\n{'全局排名':<10} {'方法':<35} {'Borda总分':<12} {'胜率':<10} {'Avg Rank':<12} {'每题均分':<10}")
    print("-" * 93)
    for gr in global_rank["global_ranking"]:
        print(f"  #{gr['global_rank']:<8} {gr['method']:<33} "
              f"{gr['borda_score']:<12} {gr['win_rate']:<10.4f} "
              f"{gr['avg_rank']:<12.4f} {gr['score_per_query']:<10.2f}")

    # 3. 排名分布矩阵
    n_methods = len(method_names)
    print(f"\n{'='*70}")
    print(f"  各方法排名分布矩阵（每个 query 中排名第 N 的次数）")
    print(f"{'='*70}")
    print(f"\n{'方法':<28}", end="")
    for r in range(1, n_methods + 1):
        print(f"  #{r:<6}", end="")
    print()
    print("-" * (30 + n_methods * 9))

    for gr in global_rank["global_ranking"]:
        m = gr["method"]
        print(f"  {m:<26}", end="")
        dist = gr["rank_distribution"]
        for r in range(1, n_methods + 1):
            cnt = dist.get(str(r), 0)
            print(f"  {cnt:<6}", end="")
        print()

    # 4. 成对比较矩阵（方法A vs 方法B 的胜出次数）
    print(f"\n{'='*70}")
    print(f"  成对比较矩阵（A 行 vs B 列，A 优于 B 的次数）")
    print(f"{'='*70}")

    pair_matrix = {m: {n: 0 for n in method_names} for m in method_names}
    for qr in results_so_far:
        if qr.get("error") or not qr.get("ranking"):
            continue
        rank_map = {}
        for item in qr["ranking"]:
            r = item.get("rank")
            m = item.get("method")
            if r is not None and m is not None:
                rank_map[m] = r
        for am in rank_map:
            for bm in rank_map:
                if am != bm and am in pair_matrix and bm in pair_matrix[am]:
                    if rank_map[am] < rank_map[bm]:
                        pair_matrix[am][bm] += 1

    col_width = 12
    header = f"  {'A\\B':<26}"
    for bm in method_names:
        short = bm.replace("fusion_", "")
        header += f" {short:<{col_width}}"
    print(header)

    print("-" * (30 + len(method_names) * (col_width + 1)))
    for am in method_names:
        row = f"  {am:<26}"
        for bm in method_names:
            val = pair_matrix[am].get(bm, 0)
            row += f" {val:<{col_width}}"
        print(row)

    print(f"\n输出已保存至: {output_path}")


if __name__ == "__main__":
    main()
