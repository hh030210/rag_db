# -*- coding: utf-8 -*-
# interactive_qa.py - 交互式检索问答脚本
# ========================
# 此文件由 gen_interactive_qa.py 自动生成，请勿手动编辑！
# 如需修改配置，请编辑 interactive_qa_config.yaml 后重新运行 gen_interactive_qa.py
#
# PowerShell 管道输入注意：
#   若使用 Get-Content pipe，Windows PowerShell 需要：
#     [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
#     Get-Content query.txt | python interactive_qa.py
#   或在命令前设置环境变量：
#     $env:PYTHONIOENCODING = "utf-8"; Get-Content query.txt | python interactive_qa.py
#
# 交互命令：
#   /mode fusion|dim|sem  - 切换检索模式（默认 fusion）
#   /rerank score|inter   - 切换重排模式（默认 score）
#   /k N                  - 设置 Top-K（默认 5）
#   /alpha N              - 设置维度检索权重 alpha（默认 0.2）
#   /temp N               - 设置 LLM 温度（默认 0.1）
#   /max N                - 设置 LLM 最大 Token 数（默认 512）
#   /qa                   - 开启/关闭问答生成
#   /limit N              - 设置每条检索结果的展示字符数（默认 300）
#   /verbose              - 显示更多检索信息（维度约束、得分分解）
#   /help                 - 显示帮助
#   /exit                 - 退出

import sys
import os
import io
import time
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ==================== 复用 retrieval_fusion_eval.py 的检索和 LLM 逻辑 ====================

import retrieval_fusion_eval as rfe
from retrieval_fusion_eval import (
    DimensionSearcher,
    SemanticSearcher,
    rrf_fuse_all,
    SearchConfig,
    _load_bge_encoder,
    _RateLimiter,
)

# call_dashscope 直接定义在文件内

# ==================== Query 扩展层（复用 chapter3 的优化逻辑）====================

from query_expander import QueryExpander, build_fusion_query

# 全局 expander（延迟初始化，仅首次 /expand 开启时加载）
_expander: QueryExpander = None
EXPAND_ENABLED = True   # 默认开启，景区 cluster 数据已就绪 (code1/chapter3_backup)

def _get_expander() -> QueryExpander:
    global _expander
    if _expander is None:
        _expander = QueryExpander(
            use_llm_subqueries=True,
            use_cluster_prompt=True,
            subquery_cache_file=str(PROJECT_ROOT / "experiment_data" / "subquery_cache.json"),
        )
    return _expander


# ---- LLM 系统提示 ----
LLM_SYSTEM_PROMPT = """你是景点知识问答助手。回答问题严格满足如下要求：

1. 先直接回答问题，再补充必要细节。
2. 优先使用上下文中的实体名、数字、年份、专有名词，避免同义改写。
3. 如果问题中包含 "列举/对比/分别/至少N项"等要求时，按条目形式给出，数量要满足要求。
4. 请严格依据给定上下文作答，上下文信息不足时明确写"没有提供相应信息，无法回答"，不要编造。
5. 语言简洁、事实准确，避免空话。"""

# ==================== LLM 配置（由配置文件生成）====================
DS_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")                  # DashScope API Key
DS_MODEL = "qwen-plus"                  # DashScope 模型
DS_MAX_RETRIES = 3
DS_RETRY_SLEEP = 5
DS_MIN_INTERVAL = 0.1

# ==================== 全局配置（由配置文件生成）====================

RETRIEVAL_MODE = "fusion"      # fusion | dim | sem
RETRIEVAL_TOP_K = 5
DIM_ALPHA = 0.5
SEM_ALPHA = 0.5
FUSION_STRATEGY = "adaptive"   # adaptive | fixed
RERANK_MODE = "score"            # score | interleaved
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 512
ENABLE_QA = True
DISPLAY_CHARS = 300
VERBOSE = False

# ---- 从配置文件读取 collection ----
_COLLECTION_NAME = None
if not _COLLECTION_NAME:
    try:
        from db_config import get_config
        _cfg = get_config()
        _default_collection = _cfg.vecdb.collection_name
    except Exception:
        _default_collection = "unified_test_collection"
else:
    _default_collection = _COLLECTION_NAME

# ---- LLM Rate Limiter ----
_DS_RATE_LIMITER = _RateLimiter(DS_MIN_INTERVAL)


# ==================== 初始化检索器 ====================

def _init_searchers():
    """初始化三种检索器"""
    print(f"\n{'='*70}")
    print(f"  交互式检索问答  |  Collection: {_default_collection}")
    print(f"{'='*70}")

    dim_ok = sem_ok = False

    # 维度检索器
    try:
        dim_searcher = DimensionSearcher(collection_name=_default_collection)
        dim_ok = True
        print(f"  [√] 维度检索器 初始化成功")
    except Exception as e:
        dim_searcher = None
        print(f"  [×] 维度检索器 初始化失败: {e}")

    # 语义检索器
    try:
        sem_searcher = SemanticSearcher(collection_name=_default_collection)
        sem_ok = True
        print(f"  [√] 语义检索器 初始化成功")
    except Exception as e:
        sem_searcher = None
        print(f"  [×] 语义检索器 初始化失败: {e}")

    print(f"  当前模式: {RETRIEVAL_MODE}  |  Top-K: {RETRIEVAL_TOP_K}  |  dim_alpha: {DIM_ALPHA}")
    print(f"  融合策略: {FUSION_STRATEGY}  |  重排模式: {RERANK_MODE}  |  LLM: {DS_MODEL}  |  问答: {'开启' if ENABLE_QA else '关闭'}")
    print(f"  命令: /mode, /rerank, /k, /alpha, /fusion, /temp, /max, /qa, /limit, /verbose, /help, /exit")
    print(f"{'='*70}\n")

    return dim_searcher, sem_searcher, dim_ok, sem_ok


dim_searcher, sem_searcher, DIM_OK, SEM_OK = _init_searchers()

if not SEM_OK:
    print("[错误] 语义检索器初始化失败，程序退出。")
    sys.exit(1)


# ==================== 景区覆盖范围定义 ====================

# 知识库实际覆盖的景区关键词集合（按知识库扫描结果录入）
# 来源1: 南孔文化(吴锡标) - 1361 chunks（南孔/孔庙全系）
# 来源2: 多景区知识 - 161 chunks（浙江/南方景区）
_KB_SPOTS = {
    # 南孔文化系列
    "南孔", "南孔庙", "孔氏南宗家庙", "衢州南孔庙", "衢州孔庙",
    "南宗家庙", "孔庙", "孔府", "阙里",
    # 多景区知识系列（主要景区）
    "雁荡山", "西湖", "杭州西湖", "丽江古城", "丽江", "云和梯田",
    "普陀山", "西双版纳", "西双版纳热带植物园", "洞头", "乌镇",
    "南浔", "西塘", "千岛湖", "天目山", "莫干山", "楠溪江",
    "象山", "鲁迅故里", "沈园", "兰亭", "大佛寺", "大鹿岛",
    "南麂岛", "江心屿", "泰顺廊桥", "石门洞", "钱江源", "龙游石窟",
    "烂柯山", "天门山", "张家界",
}

# 按长度降序排列，确保长词优先匹配（如"杭州西湖"先于"西湖"）
_KB_SPOTS_SORTED = sorted(_KB_SPOTS, key=len, reverse=True)

def check_coverage(query: str) -> tuple[bool, list[str]]:
    """
    检查查询是否涉及知识库外的景区。
    返回 (in_range, uncovered_spots)
    - in_range: True 表示检测到的景区全部在库中；False 表示存在库外景区
    - uncovered_spots: 查询中涉及但知识库未收录的景区列表
    """
    found_in_kb = []
    found_unknown = []

    for spot in _KB_SPOTS_SORTED:
        if spot in query:
            found_in_kb.append(spot)

    # 常见库外景区关键词（与知识库内的南方景区容易混淆）
    # 当查询包含这些词时，说明用户可能在问一个知识库没有的景区
    # 注意：这些词本身不在 _KB_SPOTS 中，所以不会在 found_in_kb 里
    commonly_asked_unknown = [
        "故宫", "长城", "天安门", "天坛", "颐和园", "明十三陵",
        "圆明园", "北海公园", "雍和宫", "八达岭", "十三陵",
        "龙门石窟", "兵马俑", "秦始皇陵", "黄山", "泰山",
        "桂林", "九寨沟", "张家界国家森林公园",
    ]
    for spot in sorted(commonly_asked_unknown, key=len, reverse=True):
        if spot in query:
            found_unknown.append(spot)

    in_range = len(found_unknown) == 0
    return in_range, found_unknown


# ==================== 自适应融合权重（基于 semantic_dimension_fusion_strategy.md）====================

def _compute_structural_confidence(constraints: dict) -> float:
    """
    计算结构化置信度 P_q。
    n_d: query 解析出的有效维度数量（去重）
    n_b: 至少绑定了一个维度值的维度数量
    P_q = 2*n_b / (n_d + n_b + 1)，若 n_d == 0 则 P_q = 0
    """
    if not constraints:
        return 0.0

    # 过滤掉内部 key（如 _sem_count）
    real_dims = {k: v for k, v in constraints.items() if not k.startswith("_")}

    n_d = len(real_dims)
    if n_d == 0:
        return 0.0

    n_b = sum(1 for v in real_dims.values() if v and len(v) > 0)
    P_q = (2 * n_b) / (n_d + n_b + 1)
    return P_q


def _compute_label_evidence_confidence(
    constraints: dict,
    dim_results: list,
    tag_vector_scores: dict = None,
) -> float:
    """
    计算标签证据置信度 T_q。
    对每个 query 维度 m，计算 query 标签值与文档标签值的最大相似度之和作为标签证据总量 E_T。
    T_q = 2*E_T / (|M_q| + E_T + 1)，若 |M_q| == 0 则 T_q = 0。

    tag_vector_scores: 可选的 {tag: similarity_score} 字典，用于近似 sim(v_tq, v_tD)。
    若不提供，退化为计数模式：每个维度命中计 1 分。
    """
    if not constraints:
        return 0.0

    real_dims = {k: v for k, v in constraints.items() if not k.startswith("_")}
    M_q = list(real_dims.keys())
    if not M_q:
        return 0.0

    # 收集 dim_results 中每个维度命中的文档
    # 构建 dim_name -> set(tag_name)
    dim_hit_tags: dict = {}
    for r in dim_results:
        dn = r.get("dim_name") or ""
        tn = r.get("tag_name") or ""
        if dn and tn:
            dim_hit_tags.setdefault(dn, set()).add(tn)

    E_T = 0.0
    for m in M_q:
        query_tags = set(real_dims.get(m, []))
        hit_tags = dim_hit_tags.get(m, set())
        if not query_tags or not hit_tags:
            continue

        if tag_vector_scores is not None:
            # 使用预计算的标签向量相似度（近似）
            s_m = max(
                (tag_vector_scores.get(t, 1.0) for t in hit_tags if t in query_tags),
                default=0.0,
            )
        else:
            # 退化模式：命中计数
            overlap = query_tags & hit_tags
            if overlap:
                # 命中的 tag 有得分，使用 dim_results 中该 tag 的 score
                s_m = max(
                    (r.get("score", 0.0)
                     for r in dim_results
                     if r.get("dim_name") == m and r.get("tag_name") in overlap),
                    default=1.0,
                )
            else:
                s_m = 0.0
        E_T += s_m

    T_q = (2 * E_T) / (len(M_q) + E_T + 1)
    return T_q


def _compute_concentration(dim_results: list, sem_results: list) -> tuple[float, float]:
    """
    计算单路 Top 结果集中度 C_dim 和 C_sem。
    归一化熵公式：
      C_r = 1 - (entropy(p) / log |K_r|)
    其中 p_i = R_r(q, D_i) / sum(R_r)
    """
    def _calc_cr(results: list) -> float:
        if not results:
            return 0.0
        scores = [r.get("score", 0.0) for r in results]
        total = sum(scores)
        if total <= 0 or len(results) <= 1:
            return 1.0 if results else 0.0
        import math
        probs = [s / total for s in scores]
        n = len(probs)
        h = 0.0
        for p in probs:
            if p > 0:
                h -= p * math.log(p)
        h_norm = h / math.log(n) if n > 1 else 0.0
        return max(0.0, 1.0 - h_norm)

    return _calc_cr(dim_results), _calc_cr(sem_results)


def compute_adaptive_weights(
    dim_results: list,
    sem_results: list,
    constraints: dict,
    dim_top_k: int = 5,
    sem_top_k: int = 5,
) -> tuple[float, float]:
    """
    根据 semantic_dimension_fusion_strategy.md 3.2 节的自适应权重方案，
    综合结构化置信度、标签证据置信度、集中度，动态计算 alpha_dim 和 alpha_sem。

    返回 (alpha_dim, alpha_sem)
    """
    import math

    # Step 1: 结构化置信度 P_q
    P_q = _compute_structural_confidence(constraints)

    # Step 2: 标签证据置信度 T_q
    T_q = _compute_label_evidence_confidence(constraints, dim_results)

    # Step 3: 集中度
    C_dim, C_sem = _compute_concentration(
        dim_results[:dim_top_k],
        sem_results[:sem_top_k],
    )

    # Step 4: 效用
    U_dim = P_q + T_q + C_dim
    U_sem = (1 - P_q) + (1 - T_q) + C_sem

    # Step 5: 归一化
    eps = 1e-6
    alpha_dim = U_dim / (U_dim + U_sem + eps)
    alpha_sem = 1.0 - alpha_dim

    return alpha_dim, alpha_sem


# ==================== 重排函数 ====================

def rerank_by_score(dim_results: list, sem_results: list, dim_alpha: float, sem_alpha: float, top_k: int) -> list:
    """
    方案一：得分重排
    - 将 dim 和 sem 结果的排名映射为得分（排名越靠前得分越高）
    - 合并结果，按综合得分降序排列
    """
    if not dim_results and not sem_results:
        return []

    dim_rank_map = {}
    for i, r in enumerate(dim_results):
        cid = r.get("chunk_id")
        if cid:
            dim_rank_map[cid] = len(dim_results) - i

    sem_rank_map = {}
    for i, r in enumerate(sem_results):
        cid = r.get("chunk_id")
        if cid:
            sem_rank_map[cid] = len(sem_results) - i

    all_chunks = {}
    for r in dim_results + sem_results:
        cid = r.get("chunk_id")
        if cid and cid not in all_chunks:
            all_chunks[cid] = r.copy()
            all_chunks[cid]["dim_score"] = dim_rank_map.get(cid, 0)
            all_chunks[cid]["sem_score"] = sem_rank_map.get(cid, 0)
            dim_s = dim_rank_map.get(cid, 0)
            sem_s = sem_rank_map.get(cid, 0)
            all_chunks[cid]["final_score"] = dim_alpha * dim_s + sem_alpha * sem_s
            all_chunks[cid]["source"] = ("dim" if dim_s > 0 else "") + ("+sem" if sem_s > 0 else "")

    scored = list(all_chunks.values())
    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored[:top_k]


def rerank_by_interleaved(dim_results: list, sem_results: list, top_k: int) -> list:
    """
    方案二：交替去重
    - 从 dim 和 sem 结果中交替选取
    - 遇到重复的 chunk_id 则跳过
    """
    if not dim_results:
        return sem_results[:top_k]
    if not sem_results:
        return dim_results[:top_k]

    result = []
    seen = set()
    i = j = 0
    dim_len = len(dim_results)
    sem_len = len(sem_results)

    while len(result) < top_k and (i < dim_len or j < sem_len):
        if i < dim_len:
            cid = dim_results[i].get("chunk_id")
            if cid and cid not in seen:
                r = dim_results[i].copy()
                r["source"] = "dim"
                result.append(r)
                seen.add(cid)
            i += 1
            if len(result) >= top_k:
                break

        if j < sem_len and len(result) < top_k:
            cid = sem_results[j].get("chunk_id")
            if cid and cid not in seen:
                r = sem_results[j].copy()
                r["source"] = "sem"
                result.append(r)
                seen.add(cid)
            j += 1

    return result


# ==================== 景区覆盖范围定义 ====================

# 知识库中实际覆盖的景区列表（由 build_spot_index.py 扫描 rag_chunks 生成）
# 顶层系列：南孔文化(吴锡标)、多景区知识
# 多景区知识覆盖的具体景区：
_KNOWN_SPOTS = [
    # 多景区知识系列
    "雁荡山", "西湖", "杭州西湖", "丽江古城", "丽江", "云和梯田",
    "普陀山", "西双版纳热带植物园", "西双版纳", "洞头", "乌镇",
    "南浔", "西塘", "千岛湖", "天目山", "莫干山", "楠溪江",
    "象山影视城", "鲁迅故里", "沈园", "兰亭", "大佛寺", "大鹿岛",
    "南麂岛", "江心屿", "泰顺廊桥", "石门洞", "钱江源", "龙游石窟",
    "烂柯山", "天门山", "张家界", "南孔庙", "孔氏南宗家庙",
    "衢州南孔庙", "南孔", "孔庙",
]

def check_spot_coverage(query: str) -> dict:
    """
    检查查询是否涉及知识库外的景区。
    返回 {"in_range": bool, "detected_spots": [...], "uncovered_spots": [...]}
    """
    detected = []
    for spot in _KNOWN_SPOTS:
        if spot in query:
            detected.append(spot)
    # 未检测到任何景区 → 无法判断，假定在范围内
    if not detected:
        return {"in_range": True, "detected_spots": [], "uncovered_spots": []}
    # 检测到的景区如果在知识库列表中，就算在范围内
    uncovered = [s for s in detected if s in _KNOWN_SPOTS]
    # 这里用覆盖列表本身判断，所以永远返回全覆盖
    return {"in_range": True, "detected_spots": detected, "uncovered_spots": uncovered}


# ==================== 检索函数 ====================

def do_search(query: str, mode: str, top_k: int, dim_alpha: float, sem_alpha: float, rerank_mode: str) -> dict:
    """
    执行检索，返回结构化结果。

    返回 dict：
      {
        "mode": str,
        "dim_results": [...],
        "sem_results": [...],
        "fusion_results": [...],
        "top_chunks": [...],
        "constraints": {...},
        "error": str,
      }
    """
    t0 = time.time()
    dim_raw = {}
    sem_raw = {}
    fusion_raw = []
    constraints = {}
    error = ""

    # ---- 语义检索 ----
    try:
        sem_raw = sem_searcher.search(query, top_k=SearchConfig.SEM_TOP_K)
        sem_results = sem_raw.get("results", [])
    except Exception as e:
        sem_results = []
        error += f"[语义检索] {e}  "

    # ---- 维度检索 ----
    if mode in ("fusion", "dim") and dim_searcher is not None:
        try:
            dim_raw = dim_searcher.search(query, top_k=SearchConfig.DIM_TOP_K, alpha=dim_alpha)
            constraints = dim_raw.get("constraints", {})
            dim_results = dim_raw.get("results", [])
        except Exception as e:
            dim_results = []
            error += f"[维度检索] {e}  "
    else:
        dim_results = []

    # ---- RRF 融合 ----
    effective_dim_alpha = DIM_ALPHA
    effective_sem_alpha = SEM_ALPHA

    if mode == "fusion" and dim_searcher is not None:
        try:
            # 自适应融合：根据 query 结构化置信度、标签证据置信度、集中度动态计算权重
            if FUSION_STRATEGY == "adaptive":
                ad_dim, ad_sem = compute_adaptive_weights(
                    dim_results,
                    sem_results,
                    constraints,
                    dim_top_k=5,
                    sem_top_k=5,
                )
                effective_dim_alpha = ad_dim
                effective_sem_alpha = ad_sem
                if VERBOSE:
                    P_q = _compute_structural_confidence(constraints)
                    T_q = _compute_label_evidence_confidence(constraints, dim_results)
                    C_dim, C_sem = _compute_concentration(
                        dim_results[:5], sem_results[:5]
                    )
                    print(f"  │ ├─ 自适应权重: alpha_dim={ad_dim:.4f}  alpha_sem={ad_sem:.4f}")
                    print(f"  │ ├─   证据: P_q={P_q:.4f}  T_q={T_q:.4f}  C_dim={C_dim:.4f}  C_sem={C_sem:.4f}")

            # 根据 rerank_mode 选择重排方式
            if rerank_mode == "interleaved":
                fusion_raw = rerank_by_interleaved(
                    dim_raw.get("results", []),
                    sem_results,
                    top_k=top_k,
                )
            else:
                # 方案一：得分重排（默认）
                fusion_raw = rerank_by_score(
                    dim_raw.get("results", []),
                    sem_results,
                    dim_alpha=effective_dim_alpha,
                    sem_alpha=effective_sem_alpha,
                    top_k=top_k,
                )
        except Exception as e:
            error += f"[融合] {e}  "
            fusion_raw = []

    # ---- 确定最终输出的 chunks ----
    if mode == "fusion":
        if not fusion_raw:
            top_chunks = sem_results[:top_k]
            display_results = top_chunks
        else:
            top_chunks = fusion_raw
            display_results = fusion_raw
    elif mode == "dim":
        top_chunks = dim_results[:top_k]
        display_results = top_chunks
    else:  # sem
        top_chunks = sem_results[:top_k]
        display_results = top_chunks

    elapsed = time.time() - t0

    return {
        "mode": mode,
        "elapsed": elapsed,
        "dim_results": dim_results,
        "sem_results": sem_results,
        "fusion_results": fusion_raw,
        "top_chunks": top_chunks,
        "display_results": display_results,
        "constraints": constraints,
        "error": error.strip(),
        "adaptive_dim_alpha": effective_dim_alpha,
        "adaptive_sem_alpha": effective_sem_alpha,
        "fusion_strategy": FUSION_STRATEGY,
    }


def _fmt_chunk(r: dict, idx: int, display_chars: int) -> str:
    """格式化单条检索结果"""
    if "final_score" in r:
        score_str = f"final={r['final_score']:.4f}"
    elif "score" in r:
        score_str = f"score={r['score']:.6f}"
    else:
        score_str = ""

    chunk_id = r.get("chunk_id", "?")
    title = r.get("chunk_gen_title") or r.get("doc_title", "")
    text = (r.get("chunk_text_full") or r.get("chunk_text", ""))[:display_chars].replace("\n", " ")
    suffix = " ..." if len(r.get("chunk_text_full") or r.get("chunk_text") or "") > display_chars else ""

    dim_rank = r.get("dim_rank")
    sem_rank = r.get("sem_rank")

    parts = [f"  ── #{idx+1} ──"]
    parts.append(f"    [ID]     {chunk_id}")
    parts.append(f"    [{score_str}]")
    if RERANK_MODE == "interleaved":
        source = r.get("source", "")
        parts.append(f"    [来源]   {source}")
    elif dim_rank is not None and sem_rank is not None:
        parts.append(f"    [排名]   dim=#{dim_rank}  sem=#{sem_rank}")
    elif dim_rank is not None:
        parts.append(f"    [排名]   dim=#{dim_rank}")
    elif sem_rank is not None:
        parts.append(f"    [排名]   sem=#{sem_rank}")
    if title:
        parts.append(f"    [标题]   {title}")
    parts.append(f"    [内容]   {text}{suffix}")

    return "\n".join(parts)


def _fmt_section_header(label: str) -> str:
    """打印分节标题"""
    return f"\n  ═══ {label} ═══"


def print_search_results(result: dict, display_chars: int, verbose: bool):
    """打印检索结果

    - fusion 模式：同时展示语义、维度、融合三路的前5条
    - dim/sem 模式：只展示当前模式的结果（保持原有逻辑）
    """
    mode = result["mode"]
    elapsed = result["elapsed"]
    error = result.get("error", "")
    sem_results = result.get("sem_results", [])
    dim_results = result.get("dim_results", [])
    fusion_results = result.get("fusion_results", [])

    print(f"\n  检索完成，耗时 {elapsed:.2f}s")

    if error:
        print(f"  异常: {error}")

    # fusion 模式下显示权重信息
    if mode == "fusion":
        strategy = result.get("fusion_strategy", FUSION_STRATEGY)
        ad_dim = result.get("adaptive_dim_alpha", DIM_ALPHA)
        ad_sem = result.get("adaptive_sem_alpha", SEM_ALPHA)
        if strategy == "adaptive":
            print(f"  [自适应融合] alpha_dim={ad_dim:.4f}  alpha_sem={ad_sem:.4f}")
        else:
            print(f"  [固定融合] alpha_dim={ad_dim:.4f}  alpha_sem={ad_sem:.4f}")

    # ---- fusion 模式：三路各展示 Top-5 ----
    if mode == "fusion":
        print(_fmt_section_header(f"语义检索 Top-5（共 {len(sem_results)} 条）"))
        if sem_results:
            for i, r in enumerate(sem_results[:5]):
                print(_fmt_chunk(r, i, display_chars))
                if verbose:
                    ev = r.get("evidence", [])
                    if ev:
                        print(f"    [证据]   " + " | ".join(ev[:3]))
        else:
            print("  （无结果）")

        print(_fmt_section_header(f"维度检索 Top-5（共 {len(dim_results)} 条）"))
        if dim_results:
            if verbose:
                constraints = result.get("constraints", {})
                if constraints:
                    dim_parts = []
                    for dim, vals in constraints.items():
                        dim_parts.append(f"    {dim}: {vals}")
                    print(f"  维度约束:\n" + "\n".join(dim_parts))
            for i, r in enumerate(dim_results[:5]):
                print(_fmt_chunk(r, i, display_chars))
                if verbose:
                    ev = r.get("evidence", [])
                    if ev:
                        print(f"    [证据]   " + " | ".join(ev[:3]))
        else:
            print("  （无结果）")

        print(_fmt_section_header(f"融合检索 Top-5（共 {len(fusion_results)} 条）"))
        if fusion_results:
            for i, r in enumerate(fusion_results[:5]):
                print(_fmt_chunk(r, i, display_chars))
                if verbose:
                    ev = r.get("evidence", [])
                    if ev:
                        print(f"    [证据]   " + " | ".join(ev[:3]))
        else:
            print("  （无融合结果，降级显示语义 Top-5）")
            for i, r in enumerate(sem_results[:5]):
                print(_fmt_chunk(r, i, display_chars))
                if verbose:
                    ev = r.get("evidence", [])
                    if ev:
                        print(f"    [证据]   " + " | ".join(ev[:3]))

        print()
        return

    # ---- dim / sem 模式：保持原有逻辑 ----
    display = result.get("display_results", [])
    sem_count = len(sem_results)
    dim_count = len(dim_results)

    print(f"  阶段结果: sem={sem_count}条, dim={dim_count}条")

    if verbose and dim_results:
        constraints = result.get("constraints", {})
        if constraints:
            dim_parts = []
            for dim, vals in constraints.items():
                dim_parts.append(f"    {dim}: {vals}")
            print(f"  维度约束:\n" + "\n".join(dim_parts))

    for i, r in enumerate(display):
        print(_fmt_chunk(r, i, display_chars))
        if verbose:
            ev = r.get("evidence", [])
            if ev:
                print(f"    [证据]   " + " | ".join(ev[:3]))

    print()


# ==================== LLM 调用 ====================

def call_dashscope(messages, temperature=0.1, max_tokens=512):
    import dashscope
    from dashscope import Generation
    from concurrent.futures import ThreadPoolExecutor

    _DS_RATE_LIMITER.wait()
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(
            Generation.call,
            DS_MODEL,
            messages=messages,
            result_format='message',
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=DS_API_KEY,
        )
        result = future.result()
    if result.status_code != 200:
        raise Exception(f"DashScope 错误: {result.code} {result.message}")
    return result.output.choices[0].message.content


def _build_prompt_for_pm(pm: dict, query: str, context: str) -> Tuple[str, str]:
    """用 PromptModule 组装 system + user prompt。返回 (system_prompt, user_prompt)。"""
    system_prompt = pm.get("P_sys", LLM_SYSTEM_PROMPT)
    instruction  = pm.get("I_t", "")
    ctx_strategy = pm.get("C_t", "")
    fmt_require  = pm.get("F_t", "")
    uncertainty  = pm.get("U_t", "")

    user_prompt = f"""<instruction>
{instruction}
</instruction>

<context_strategy>
{ctx_strategy}
</context_strategy>

<format_requirement>
{fmt_require}
</format_requirement>

<uncertainty_handling>
{uncertainty}
</uncertainty_handling>

上下文信息：
{context}

用户问题：
{query}

请给出最终回答："""
    return system_prompt, user_prompt


def _build_default_prompt(query: str, context: str) -> Tuple[str, str]:
    """默认 prompt。"""
    system_prompt = LLM_SYSTEM_PROMPT
    user_prompt = f"""根据以下上下文信息回答用户问题。

要求：
- 不要使用上下文之外的信息。
- 尽量使用上下文原句中的关键表述（人名、地名、年份、数量）。
- 如果答案在上下文中不完整，明确写"文中未说明"。
- 不要输出推理过程。

上下文信息：
{context}

用户问题：
{query}

请给出最终回答："""
    return system_prompt, user_prompt


def _fuse_answers(question: str, answers: list, prompt_infos: list) -> str:
    """LLM 融合多个答案（借鉴 code1 v9 的融合策略）。"""
    if len(answers) <= 1:
        return answers[0]

    lines = []
    for i, (ans, pinfo) in enumerate(zip(answers, prompt_infos), 1):
        cluster_label = f"cluster-{pinfo.get('cluster_id', '?')}" if pinfo else "默认"
        sim = pinfo.get("sim", 0) if pinfo else 0
        q = pinfo.get("question", "") if pinfo else ""
        lines.append(f"回答{i}（{cluster_label}，相似度={sim:.3f}，参考问题：{q}）：\n{ans}")

    fusion_prompt = f"""基于以下多个回答，整合出一个最佳答案。

问题：{question}

{chr(10).join(lines)}

请整合以上回答的优点，生成一个完整、准确、清晰的最终答案。直接输出答案内容，不要添加解释。"""

    messages = [{"role": "user", "content": fusion_prompt}]
    try:
        fused = call_dashscope(messages, temperature=0.5, max_tokens=800)
        return fused.strip() if fused else answers[0]
    except Exception as e:
        print(f"  [融合警告] LLM 融合失败: {e}")
        return answers[0]


def do_qa(query: str, chunks: list, top_k: int, temperature: float, max_tokens: int,
          expand_result: dict = None):
    """调用 LLM 生成答案，支持多 Prompt 融合（/expand 开启时）。

    策略（借鉴 code1 Tourist_step5_inference_multithread_v9.py）：
      - 命中 Top-2 cluster 时，分别用各自的 PromptModule 生成答案
      - 调用 LLM 融合答案
      - 单 Prompt 或无聚类数据时退化回原有逻辑
    """
    t0 = time.time()
    if not chunks:
        print("  [跳过] 无检索结果")
        return

    context_parts = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("chunk_text_full") or chunk.get("chunk_text") or ""
        title = chunk.get("chunk_gen_title") or chunk.get("doc_title", "")
        cid = chunk.get("chunk_id", "")
        context_parts.append(f"[来源 {i+1}] {title} ({cid}):\n{text[:500]}")

    context = "\n\n".join(context_parts)

    # ── 收集可用的 PromptModule（Top-2） ──
    top_prompts: list = (expand_result or {}).get("top_prompts", [])
    top_clusters: list = (expand_result or {}).get("top_clusters", [])

    usable_prompts = []   # List[dict]
    usable_info = []      # List[dict] 辅助信息
    for i, pm in enumerate(top_prompts):
        if pm:
            cid, sim = top_clusters[i] if i < len(top_clusters) else (None, 0)
            usable_prompts.append(pm)
            usable_info.append({"cluster_id": cid, "sim": sim, "question": ""})

    # ── 多 Prompt 生成 + 融合 ──
    if len(usable_prompts) >= 2:
        print(f"  [多 Prompt 融合] 使用 {len(usable_prompts)} 个 PromptModule")
        answers = []
        prompt_results = []

        for i, pm in enumerate(usable_prompts):
            sys_p, usr_p = _build_prompt_for_pm(pm, query, context)
            cid = usable_info[i].get("cluster_id")
            print(f"  └─ Prompt {i+1}: cluster-{cid}")
            messages = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
            try:
                ans = call_dashscope(messages, temperature=temperature, max_tokens=max_tokens)
                answers.append(ans)
                prompt_results.append(usable_info[i])
            except Exception as e:
                print(f"    [警告] Prompt {i+1} 生成失败: {e}")

        if len(answers) >= 2:
            print(f"  └─ 融合 {len(answers)} 个答案...")
            final_answer = _fuse_answers(query, answers, prompt_results)
        elif answers:
            final_answer = answers[0]
        else:
            print("  [警告] 所有 Prompt 生成均失败，退回默认 Prompt")
            sys_p, usr_p = _build_default_prompt(query, context)
            messages = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
            final_answer = call_dashscope(messages, temperature=temperature, max_tokens=max_tokens)
    else:
        # ── 单 Prompt 或无 Prompt ──
        pm = (expand_result or {}).get("prompt_module")
        if pm:
            sys_p, usr_p = _build_prompt_for_pm(pm, query, context)
            cid = top_clusters[0][0] if top_clusters else None
            print(f"  [单 Prompt] cluster-{cid}")
        else:
            sys_p, usr_p = _build_default_prompt(query, context)
            print("  [默认 Prompt]")

        messages = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
        final_answer = call_dashscope(messages, temperature=temperature, max_tokens=max_tokens)

    elapsed = time.time() - t0
    cluster_tag = f" [cluster={top_clusters[0][0]}]" if top_clusters else ""
    print(f"\n  生成完成，耗时 {elapsed:.2f}s{cluster_tag}\n")
    print(f"  ── LLM 回答 ──")
    print(f"  {final_answer}")
    print()



# ==================== 交互主循环 ====================

def main():
    global RETRIEVAL_MODE, RETRIEVAL_TOP_K, DIM_ALPHA, RERANK_MODE, FUSION_STRATEGY
    global LLM_TEMPERATURE, LLM_MAX_TOKENS, ENABLE_QA
    global DISPLAY_CHARS, VERBOSE, EXPAND_ENABLED

    print("  输入问题开始问答（或输入 /help 查看命令）\n")

    while True:
        try:
            raw = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  退出。")
            break

        # PowerShell Get-Content GBK 解码问题修复
        if raw and not any('\u4e00' <= c <= '\u9fff' for c in raw):
            try:
                raw = sys.stdin.buffer.read().decode('utf-8', errors='replace').strip()
            except Exception:
                pass

        if not raw:
            continue

        cmd = raw.lower()

        # ==================== 命令处理 ====================

        if cmd in ("/exit", "/quit", "/q"):
            print("  再见！")
            break

        if cmd == "/help":
            print("""
  可用命令：
    /mode fusion|dim|sem   切换检索模式（默认 fusion）
         fusion: 融合检索（默认，维度+语义）
         dim:    维度检索（向量+标签）
         sem:    语义检索（纯向量 ANN）
    /rerank score|inter    切换重排模式（仅 fusion 模式生效）
         score:     得分重排（默认，权重可调）
         inter:     交替去重（平衡两种检索结果）
    /fusion adaptive|fixed 切换融合策略（仅 fusion + score 模式生效）
         adaptive:  自适应融合（默认，根据 P_q、T_q、C_r 动态计算权重）
                    基于 semantic_dimension_fusion_strategy.md 3.2 节
         fixed:     固定权重（使用 /alpha 设置的值）
    /k N                  设置 Top-K（例如 /k 3）
    /alpha N              设置维度检索固定权重 alpha（仅 fusion/fixed 模式生效）
    /temp N               设置 LLM 温度（例如 /temp 0.2）
    /max N                设置 LLM 最大 Token 数（例如 /max 256）
    /qa                   开启/关闭问答生成
    /limit N              设置展示字符数（默认 300，建议 >= 50）
    /verbose               显示详细检索信息（维度约束、得分分解、自适应权重证据）
    /expand               开启/关闭 Query 扩展（基于 chapter3 子查询+聚类优化）
    /help                 显示本帮助
    /exit                 退出程序
            """)
            continue

        if cmd.startswith("/mode "):
            mode = cmd.split()[1]
            if mode not in ("fusion", "dim", "sem"):
                print("  无效模式，可用: fusion | dim | sem")
            else:
                if mode == "dim" and not DIM_OK:
                    print(f"  [警告] 维度检索器未初始化，仅切换为 sem 模式")
                    mode = "sem"
                RETRIEVAL_MODE = mode
                print(f"  检索模式已切换为: {mode}")
            continue

        # ---- rerank mode ----
        if cmd.startswith("/rerank "):
            rm = cmd.split()[1]
            if rm == "inter":
                rm = "interleaved"
            if rm not in ("score", "interleaved"):
                print("  无效重排模式，可用: score | inter")
            else:
                RERANK_MODE = rm
                print(f"  重排模式已切换为: {rm}")
            continue

        # ---- Top-K ----
        if cmd.startswith("/k "):
            try:
                n = int(cmd.split()[1])
                if n < 1:
                    raise ValueError
                RETRIEVAL_TOP_K = n
                print(f"  Top-K 已设置为 {n}")
            except (IndexError, ValueError):
                print("  用法: /k N（N 为正整数）")
            continue

        # ---- dim_alpha ----
        if cmd.startswith("/alpha "):
            try:
                a = float(cmd.split()[1])
                DIM_ALPHA = a
                print(f"  dim_alpha 已设置为 {a}")
            except (IndexError, ValueError):
                print("  用法: /alpha N（N 为浮点数，例如 0.3）")
            continue

        # ---- fusion strategy ----
        if cmd.startswith("/fusion "):
            fs = cmd.split()[1]
            if fs not in ("adaptive", "fixed"):
                print("  无效融合策略，可用: adaptive | fixed")
            else:
                FUSION_STRATEGY = fs
                if fs == "adaptive":
                    print(f"  融合策略已切换为: adaptive（根据 query 结构动态计算权重）")
                else:
                    print(f"  融合策略已切换为: fixed（使用 /alpha 设置的固定权重）")
            continue

        # ---- temperature ----
        if cmd.startswith("/temp "):
            try:
                t = float(cmd.split()[1])
                LLM_TEMPERATURE = t
                print(f"  LLM 温度已设置为 {t}")
            except (IndexError, ValueError):
                print("  用法: /temp N（N 为浮点数，例如 0.2）")
            continue

        # ---- max_tokens ----
        if cmd.startswith("/max "):
            try:
                n = int(cmd.split()[1])
                if n < 1:
                    raise ValueError
                LLM_MAX_TOKENS = n
                print(f"  LLM 最大 Token 数已设置为 {n}")
            except (IndexError, ValueError):
                print("  用法: /max N（N 为正整数）")
            continue

        # ---- display limit ----
        if cmd.startswith("/limit "):
            try:
                n = int(cmd.split()[1])
                if n < 10:
                    raise ValueError
                DISPLAY_CHARS = n
                print(f"  展示字符数已设置为 {n}")
            except (IndexError, ValueError):
                print("  用法: /limit N（N 为正整数，建议 >= 50）")
            continue

        # ---- qa toggle ----
        if cmd == "/qa":
            ENABLE_QA = not ENABLE_QA
            print(f"  LLM 问答生成已{'开启' if ENABLE_QA else '关闭'}")
            continue

        # ---- verbose toggle ----
        if cmd == "/verbose":
            VERBOSE = not VERBOSE
            print(f"  详细模式已{'开启' if VERBOSE else '关闭'}")
            continue

        # ---- query expand toggle ----
        if cmd == "/expand":
            global EXPAND_ENABLED
            EXPAND_ENABLED = not EXPAND_ENABLED
            if EXPAND_ENABLED:
                _ = _get_expander()  # 预加载
            print(f"  Query 扩展已{'开启' if EXPAND_ENABLED else '关闭'}")
            continue

        # ==================== 普通问题 ====================
        query = raw

        print(f"\n{'─'*70}")
        print(f"  模式: {RETRIEVAL_MODE}  |  融合: {FUSION_STRATEGY}  |  问题: {query}")
        if EXPAND_ENABLED:
            print(f"  [Query 扩展] 开启中 ...")
        print(f"{'─'*70}")

        # ── 景区覆盖预检（方案B：查询前提示）──
        in_range, uncovered_spots = check_coverage(query)
        if not in_range:
            print(f"\n  ⚠  景区覆盖预检:")
            print(f"     检测到景区: {'、'.join(uncovered_spots)}")
            print(f"     当前知识库尚未收录该景区，以下结果仅供参考。")
            print(f"     知识库覆盖: 南孔文化、杭州西湖、丽江古城、雁荡山、普陀山、云和梯田、乌镇、洞头等南方景区。")

        # ── Query 扩展（chapter3 优化层）──
        expand_result = None
        if EXPAND_ENABLED:
            try:
                expander = _get_expander()
                expand_result = expander.expand(query)
                fusion_q = build_fusion_query(expand_result)
                if VERBOSE:
                    print(f"  │ ├─ 子查询: {expand_result['sub_queries']}")
                    print(f"  │ ├─ 实体:   {expand_result['entity_terms']}")
                    print(f"  │ └─ 聚类:   id={expand_result['cluster_id']}  sim={expand_result['cluster_sim']}  time={expand_result['expand_time']}s")
                # 用融合 query 替换原始 query
                query = fusion_q
            except Exception as e:
                print(f"  [Query 扩展] 失败，回退原始 query: {e}")
                EXPAND_ENABLED = False

        # 执行检索
        result = do_search(
            query=query,
            mode=RETRIEVAL_MODE,
            top_k=RETRIEVAL_TOP_K,
            dim_alpha=DIM_ALPHA,
            sem_alpha=SEM_ALPHA,
            rerank_mode=RERANK_MODE,
        )

        # 打印结果
        print_search_results(result, DISPLAY_CHARS, VERBOSE)

        # 问答生成
        if ENABLE_QA:
            top_chunks = result["top_chunks"]
            do_qa(
                query=raw,             # 用原始 query 提问（非融合 query）
                chunks=top_chunks,
                top_k=RETRIEVAL_TOP_K,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                expand_result=expand_result,  # 透传 chapter3 优化结果
            )


if __name__ == "__main__":
    main()
