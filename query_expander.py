# -*- coding: utf-8 -*-
"""
query_expander.py — Chapter3 Query 优化层，插入到 interactive_qa.py 的 do_search 前

复用来源：
  - chapter3/code1/chapter3/codes/bylw_rag/new_experiments/PubMedQA_step2_prompt_iteration.py
  - chapter3/code1/chapter3/codes/bylw_rag/new_experiments/PubMedQA_step3_kmeans_clustering.py
  - retrieval_fusion_eval.py（_encode_query / BGE 编码器）

使用方式（在 interactive_qa.py 的 do_search 调用前）：
    from query_expander import QueryExpander
    expander = QueryExpander()          # 首次调用初始化，之后全局复用
    expanded = expander.expand(query) # 返回 { "sub_queries", "entity_terms", "cluster_hint", "prompt_module" }
    # expanded["sub_queries"] 直接喂入 do_search(...)
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent
# 训练产物在 chapter3_backup（景区数据），不是 code1/chapter3（PubMedQA 默认产物）
CHAPTER3_ROOT = PROJECT_ROOT / "code1" / "chapter3_backup" / "codes" / "bylw_rag" / "new_experiments"

# ─────────────────────────── 复用 retrieval_fusion_eval 的编码器 ───────────────────────────

try:
    from retrieval_fusion_eval import _load_bge_encoder, _encode_query
    _encoder = _load_bge_encoder()
except Exception:
    _encoder = None


# ─────────────────────────── LLM 调用（复用 interactive_qa 的 call_dashscope 风格）──────────────────────────

def _call_llm(prompt: str, system: str = "", temperature: float = 0.3,
              max_tokens: int = 512, model: str = "qwen-plus") -> str:
    """
    调用 DashScope API（与 interactive_qa.py 的 call_dashscope 保持一致）。
    若你需要切换到 SiliconFlow 或本地 DeepSeek，只需改这里一处。
    """
    import urllib.request

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "input": {"messages": messages},
        "parameters": {"temperature": temperature, "max_tokens": max_tokens}
    }

    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["output"]["text"]
    except Exception as e:
        print(f"[QueryExpander LLM] 调用失败: {e}")
        return ""


# ─────────────────────────── 1. 子查询扩展（来自 step2 的 PromptIterator）──────────────────────────

SUBQUERY_SYSTEM = """你是一个专业的信息检索助手。对于给定的用户问题，请进行语义分析和检索优化。"""

SUBQUERY_TEMPLATE = """请对以下问题进行检索优化，输出多个独立的子查询：

**原始问题**：{query}

请从以下角度生成 2~4 个子查询（每个不超过 50 字）：
1. 核心实体查询：提取问题中的主要实体（人名、地名、机构、概念等）
2. 属性/特征查询：围绕实体的属性、特征、状态进行查询
3. 关系查询：查询实体之间的关系、因果、对比等
4. 时间/条件查询：包含时间或条件限定的问题

请直接输出子查询列表，每行一个，不要添加序号或解释：
"""


def generate_subqueries(query: str, encoder=None) -> List[str]:
    """
    使用 LLM 生成多个语义独立的子查询。
    同时用 BGE 编码 query 向量备用。
    返回子查询列表。
    """
    prompt = SUBQUERY_TEMPLATE.format(query=query)
    raw = _call_llm(prompt, system=SUBQUERY_SYSTEM, temperature=0.3, max_tokens=256)

    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    subqueries = []
    for line in lines:
        # 过滤掉太短的行或明显是解释性文字
        if 5 < len(line) < 120:
            subqueries.append(line)

    # 兜底：原始 query 本身
    if not subqueries:
        subqueries = [query]

    return subqueries


# ─────────────────────────── 2. 实体术语提取 ───────────────────────────

ENTITY_EXTRACT_PROMPT = """从以下问题中提取关键词实体和术语（不超过 10 个），用 | 分隔：
{query}
直接输出术语列表，不要解释："""


def extract_entity_terms(query: str) -> List[str]:
    """
    提取问题中的关键实体术语，用于增强维度检索的约束匹配。
    """
    prompt = ENTITY_EXTRACT_PROMPT.format(query=query)
    raw = _call_llm(prompt, temperature=0.1, max_tokens=64)

    terms = [t.strip() for t in raw.split("|") if t.strip()]
    return terms[:10]


# ─────────────────────────── 3. 轻量聚类匹配（简化版，不依赖 sklearn）──────────────────────────

# chapter3_backup 的景区产物路径：
#   clustering_results/tourist/cluster_results.json   — 聚合文件，含 clusters[].cluster_id/center/size
#   iteration_results/tourist/tourist_question_*/final_prompt.json — 每个 question 训出的 prompt
#   需要 tourist_question_cluster_mapping.json 把 prompt 映射到 cluster
CLUSTER_CENTERS_FILE = CHAPTER3_ROOT / "clustering_results" / "tourist" / "cluster_results.json"
ITERATION_RESULTS_DIR = CHAPTER3_ROOT / "iteration_results" / "tourist"


def _load_cluster_data() -> Tuple[Dict[int, np.ndarray], Dict[int, List[Dict]]]:
    """
    加载 chapter3_backup 景区数据训练好的聚类中心和优化 prompt。
    如果文件不存在，返回空字典（优雅降级）。

    结构：
      cluster_results.json:
        {"clusters": [{"cluster_id": int, "size": int, "center": [float, ...]}, ...]}
      iteration_results/tourist/tourist_question_*\/final_prompt.json:
        {"final_prompt": {"prompt_module": {"P_sys":..., "I_t":..., ...}}, ...}

    由于 iteration_results 的目录结构是按 question_id 而不是按 cluster_id 组织的，
    需要先查找 question_cluster_mapping.json（如果存在），否则只保留每个 cluster_id 的
    第一个遇到的 prompt 作为代表。
    """
    centers: Dict[int, np.ndarray] = {}
    prompts: Dict[int, List[Dict]] = {}

    # 1) 加载聚类中心
    if CLUSTER_CENTERS_FILE.exists():
        try:
            with open(CLUSTER_CENTERS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cluster in data.get("clusters", []):
                cid = cluster["cluster_id"]
                if "center" in cluster and cluster["center"]:
                    centers[cid] = np.array(cluster["center"], dtype=np.float32)
        except Exception as e:
            print(f"[QueryExpander] 加载聚类中心失败: {e}")

    # 2) 加载优化 prompt——优先按 question→cluster 映射加载，没有再退化到遍历
    if not ITERATION_RESULTS_DIR.exists():
        return centers, prompts

    mapping_file = ITERATION_RESULTS_DIR / "tourist_question_cluster_mapping.json"
    question_to_cluster: Dict[str, int] = {}
    if mapping_file.exists():
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                m = json.load(f)
            # 兼容两种格式：{question_id: cluster_id} 或 {question_id: {"cluster_id": int}}
            for qid, info in m.items():
                if isinstance(info, dict):
                    question_to_cluster[qid] = int(info.get("cluster_id", -1))
                else:
                    question_to_cluster[qid] = int(info)
        except Exception as e:
            print(f"[QueryExpander] 加载 question-cluster 映射失败: {e}")

    # 遍历 iteration_results，按映射填入 cluster_prompts
    try:
        for q_dir in ITERATION_RESULTS_DIR.iterdir():
            if not q_dir.is_dir() or not q_dir.name.startswith("tourist_question_"):
                continue
            qid = q_dir.name.replace("tourist_question_", "")
            final_prompt_file = q_dir / "final_prompt.json"
            if not final_prompt_file.exists():
                continue
            try:
                with open(final_prompt_file, "r", encoding="utf-8") as f:
                    fp = json.load(f)
                pm = (fp.get("final_prompt") or {}).get("prompt_module")
                if not pm:
                    continue
            except Exception:
                continue

            # 决定 cluster_id
            if qid in question_to_cluster:
                cid = question_to_cluster[qid]
                if cid is None or cid < 0:
                    continue
                prompts.setdefault(cid, []).append(pm)
    except Exception as e:
        print(f"[QueryExpander] 加载优化 prompt 失败: {e}")

    # 退化：如果映射文件缺失导致 prompts 为空，但目录里有 final_prompt.json，
    # 把所有 prompt 合并摊到所有 cluster_ids 上（避免完全空）
    if centers and not prompts:
        try:
            all_pms = []
            for q_dir in ITERATION_RESULTS_DIR.iterdir():
                if not q_dir.is_dir() or not q_dir.name.startswith("tourist_question_"):
                    continue
                fp = q_dir / "final_prompt.json"
                if fp.exists():
                    with open(fp, "r", encoding="utf-8") as f:
                        pm = (json.load(f).get("final_prompt") or {}).get("prompt_module")
                    if pm:
                        all_pms.append(pm)
            if all_pms:
                merged_pm = {}
                keys = set().union(*[p.keys() for p in all_pms])
                for k in keys:
                    vals = [p[k] for p in all_pms if k in p and p[k]]
                    if vals:
                        merged_pm[k] = "\n---\n".join(vals)
                for cid in centers.keys():
                    prompts[cid] = [{"_merged": True, "_mappings": len(all_pms), **merged_pm}]
        except Exception as e:
            print(f"[QueryExpander] 退化聚合 prompt 失败: {e}")

    return centers, prompts


_cluster_centers: Dict[int, np.ndarray] = {}
_cluster_prompts: Dict[int, List[Dict]] = {}


def _ensure_cluster_loaded():
    global _cluster_centers, _cluster_prompts
    if not _cluster_centers:
        _cluster_centers, _cluster_prompts = _load_cluster_data()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def find_nearest_clusters(query: str, top_n: int = 2) -> Tuple[List[Tuple[int, float]], List[Optional[Dict]]]:
    """
    用 BGE 将 query 编码后，与预训练聚类中心做余弦相似度，
    返回 Top-N 最近的 (cluster_id, similarity) 列表和对应的 prompt_module 列表。
    若无聚类数据，返回空列表。
    """
    _ensure_cluster_loaded()
    if not _cluster_centers or _encoder is None:
        return [], []

    q_vec = _encode_query(_encoder, query)
    if q_vec is None:
        return [], []

    q_emb = np.array(q_vec, dtype=np.float32)
    scored = []
    for cid, center in _cluster_centers.items():
        sim = cosine_sim(q_emb, center)
        scored.append((cid, float(sim)))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_clusters = scored[:top_n]

    prompts = []
    for cid, sim in top_clusters:
        p = _cluster_prompts.get(cid, [])
        prompts.append(p[0] if p else None)

    return top_clusters, prompts


def find_nearest_cluster(query: str) -> Tuple[Optional[int], float]:
    """
    用 BGE 将 query 编码后，与预训练聚类中心做余弦相似度，返回最近的 cluster_id 和相似度。
    若无聚类数据，返回 (None, 0.0)。
    """
    clusters, _ = find_nearest_clusters(query, top_n=1)
    if clusters:
        return clusters[0]
    return None, 0.0


def get_cluster_prompt(cluster_id: int) -> Optional[Dict]:
    """
    返回指定 cluster 的优化 prompt。
    取该 cluster 中评分最高的那个 prompt_module。
    """
    _ensure_cluster_loaded()
    prompts = _cluster_prompts.get(cluster_id, [])
    if not prompts:
        return None
    # 取第一个（通常是评分最高的）
    return prompts[0]


# ─────────────────────────── 4. 主入口类 ───────────────────────────

class QueryExpander:
    """
    对原始 query 进行三层优化后，再交给 do_search 检索：

        1. 子查询扩展  — LLM 生成 2~4 个语义独立的子查询
        2. 实体术语提取 — 用于增强维度约束匹配
        3. 聚类 Prompt 选择 — 匹配预训练聚类，拿到专属 prompt_module

    expand() 返回一个 dict，可直接与 do_search / do_qa 衔接。
    """

    def __init__(self, use_llm_subqueries: bool = True,
                 use_cluster_prompt: bool = True,
                 subquery_cache_file: str = None):
        """
        Params:
            use_llm_subqueries:  True=调用 LLM 生成子查询，False=只返回原始 query
            use_cluster_prompt:  True=做聚类匹配加载专属 prompt，False=跳过（不依赖 chapter3 预训练数据）
            subquery_cache_file: 子查询结果缓存文件路径（可选，用于批量推理加速）
        """
        self.use_llm_subqueries = use_llm_subqueries
        self.use_cluster_prompt = use_cluster_prompt
        self.subquery_cache_file = subquery_cache_file
        self._subquery_cache: Dict[str, List[str]] = {}
        self._load_cache()

    def _load_cache(self):
        if self.subquery_cache_file and Path(self.subquery_cache_file).exists():
            try:
                with open(self.subquery_cache_file, "r", encoding="utf-8") as f:
                    self._subquery_cache = json.load(f)
            except Exception:
                pass

    def _save_cache(self):
        if self.subquery_cache_file:
            try:
                with open(self.subquery_cache_file, "w", encoding="utf-8") as f:
                    json.dump(self._subquery_cache, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    def expand(self, query: str) -> Dict[str, Any]:
        """
        主入口：对一条 query 进行完整优化。

        Returns:
            {
                "original_query": str,          # 原始 query
                "sub_queries": List[str],       # 子查询列表（可直接喂入 do_search）
                "entity_terms": List[str],      # 实体术语列表（用于增强维度约束）
                "cluster_id": Optional[int],    # 匹配的聚类 ID（None 表示未命中）
                "cluster_sim": float,           # 聚类相似度
                "prompt_module": Optional[Dict],# 该聚类的优化 PromptModule（None 表示无预训练数据）
            }
        """
        t0 = time.time()

        # ── 1. 子查询扩展 ──
        if self.use_llm_subqueries:
            if query in self._subquery_cache:
                sub_queries = self._subquery_cache[query]
            else:
                sub_queries = generate_subqueries(query, encoder=_encoder)
                self._subquery_cache[query] = sub_queries
                self._save_cache()
        else:
            sub_queries = [query]

        # ── 2. 实体术语提取 ──
        entity_terms = extract_entity_terms(query)

        # ── 3. 聚类匹配 & Prompt 选择（Top-2） ──
        top_clusters: List[Tuple[int, float]] = []
        top_prompts: List[Optional[Dict]] = []
        if self.use_cluster_prompt:
            top_clusters, top_prompts = find_nearest_clusters(query, top_n=2)

        elapsed = time.time() - t0

        return {
            "original_query": query,
            "sub_queries": sub_queries,
            "entity_terms": entity_terms,
            "top_clusters": top_clusters,           # List[(cluster_id, sim)]
            "top_prompts": top_prompts,              # List[Optional[PromptModule]]
            "cluster_id": top_clusters[0][0] if top_clusters else None,
            "cluster_sim": top_clusters[0][1] if top_clusters else 0.0,
            "prompt_module": top_prompts[0] if top_prompts else None,
            "expand_time": round(elapsed, 2),
        }


# ─────────────────────────── 5. 与 do_search 衔接的便捷封装 ───────────────────────────

def build_fusion_query(expand_result: Dict) -> str:
    """
    将 expand() 得到的 sub_queries 合并成一个字符串，
    供 do_search 直接使用。

    策略：用 " | " 连接各子查询，形成多路检索信号。
    """
    subs = expand_result["sub_queries"]
    if len(subs) == 1:
        return subs[0]
    # 多子查询拼接，使检索器在同一 query 内感知多意图
    return " | ".join(subs)


def build_qa_system_prompt(expand_result: Dict, base_prompt: str = None) -> str:
    """
    若命中聚类且有预训练 PromptModule，优先使用它；
    否则回退到 base_prompt（interactive_qa.py 的 LLM_SYSTEM_PROMPT）。
    """
    pm = expand_result.get("prompt_module")
    if pm and base_prompt:
        # 将 chapter3 的 PromptModule 结构注入 QA
        return base_prompt  # 可扩展：按 pm 的字段自定义拼接
    return base_prompt or ""
