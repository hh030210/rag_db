# -*- coding: utf-8 -*-
"""
prompt_iteration_optimizer.py
=============================
把 ``interactive_qa.py`` 中通过 ``query_expander.QueryExpander`` 实现的
"问题聚类提示词迭代优化" 抽离为独立脚本。

包含 4 大能力：

    1. SubqueryGenerator       - LLM 子查询扩展（来自 chapter3 step2 PromptIterator）
    2. EntityExtractor         - 实体术语提取（增强检索信号）
    3. ClusterPromptSelector   - 聚类匹配 + 优化 PromptModule 选择（来自 chapter3 step3 KMeans）
    4. PromptIterationOptimizer - 组合上述三步的便捷入口（= 旧的 QueryExpander）

输入任意一条 query，输出：

    {
        "original_query":  str,
        "sub_queries":     List[str],
        "entity_terms":    List[str],
        "top_clusters":    List[(cluster_id, sim)],   # Top-N
        "top_prompts":     List[Optional[Dict]],       # 与 top_clusters 一一对应
        "cluster_id":      Optional[int],
        "cluster_sim":     float,
        "prompt_module":   Optional[Dict],
        "optimize_time":   float,
    }

兼容说明：
    - 返回 dict 的所有键名（含 cluster_id/cluster_sim/prompt_module/sub_queries/entity_terms）
      100% 与 ``query_expander.QueryExpander.expand()`` 对齐，因此 ``interactive_qa.py``
      中所有 ``expand_result[...]`` 的取值代码不需要修改。
    - 只需将 ``interactive_qa.py`` 顶部 ``from query_expander import QueryExpander, build_fusion_query``
      替换为
          ``from prompt_iteration_optimizer import PromptIterationOptimizer, build_fusion_query``
      并把 ``QueryExpander(...)`` 改成 ``PromptIterationOptimizer(...)`` 即可。
"""

import os
import sys
import json
import time
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# ────────────────────────────────────────────
# 路径与全局配置
# ────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent

# 训练产物在 chapter3_backup（景区数据），不是 code1/chapter3（PubMedQA 默认产物）
CHAPTER3_ROOT = (
    PROJECT_ROOT
    / "code1"
    / "chapter3_backup"
    / "codes"
    / "bylw_rag"
    / "new_experiments"
)
CLUSTER_CENTERS_FILE = (
    CHAPTER3_ROOT / "clustering_results" / "tourist" / "cluster_results.json"
)
ITERATION_RESULTS_DIR = CHAPTER3_ROOT / "iteration_results" / "tourist"

# DashScope 默认模型名；API Key 不再硬编码，由调用方在运行时通过参数传入。
DEFAULT_LLM_MODEL = "qwen-plus"

# BGE-M3 模型本地路径（相对当前文件），便于服务自包含。
# 服务器上如果放到别处，可以传环境变量 BGE_MODEL_PATH 覆盖。
_BGE_DEFAULT_PATH = str((PROJECT_ROOT / "model" / "bge-m3").resolve())


# ────────────────────────────────────────────
# BGE-M3 编码器（自包含实现，不依赖 retrieval_fusion_eval）
# ────────────────────────────────────────────

def _load_bge_encoder(model_path: Optional[str] = None):
    """
    加载 BGE-M3 编码器（优先 FlagEmbedding，失败回退 SentenceTransformer）。

    模型查找顺序：
        1) 入参 model_path
        2) 环境变量 BGE_MODEL_PATH
        3) PROJECT_ROOT/model/bge-m3
        4) PROJECT_ROOT.parent/model/bge-m3
        5) PROJECT_ROOT.parent.parent/model/bge-m3
    """
    candidates = []
    if model_path:
        candidates.append(model_path)
    env_path = os.getenv("BGE_MODEL_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.append(_BGE_DEFAULT_PATH)
    parent = PROJECT_ROOT.parent
    candidates.append(str((parent / "model" / "bge-m3").resolve()))
    candidates.append(str((parent.parent / "model" / "bge-m3").resolve()))

    resolved = None
    for p in candidates:
        if p and os.path.exists(p):
            resolved = p
            break
    if not resolved:
        print("[警告] 未找到 BGE-M3 模型目录，编码器置为 None（聚类匹配将不可用）")
        return None

    # 优先 FlagEmbedding（与原 retriever 一致）
    try:
        from FlagEmbedding import BGEM3FlagModel
        return BGEM3FlagModel(resolved, use_fp16=False, device='cpu')
    except ImportError:
        pass
    except Exception as e:
        print(f"[警告] FlagEmbedding 加载失败 ({e})，回退到 SentenceTransformer")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(resolved, local_files_only=True)

        # 套一层 Proxy 让 SentenceTransformer 暴露与 BGEM3FlagModel 兼容的 encode 接口
        class _FlagProxy:
            def __init__(self, m):
                self._model = m

            def encode(self, texts, return_dense=False, **kwargs):
                kwargs.pop("normalize_embeddings", None)
                kwargs.pop("show_progress_bar", None)
                emb = self._model.encode(
                    list(texts),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                if return_dense:
                    return {"dense_vecs": emb}
                return emb

        return _FlagProxy(model)
    except Exception as e:
        print(f"[警告] SentenceTransformer 加载失败: {e}")
        return None


def _encode_query(encoder, query: str):
    """
    用 BGE 编码一条 query，返回稠密向量（numpy 或 list）。
    """
    if encoder is None:
        return None
    try:
        result = encoder.encode([query], return_dense=True)
        # FlagEmbedding BGEM3FlagModel.encode(dict) vs SentenceTransformer/FlagProxy 返回 array
        if isinstance(result, dict):
            if "dense_vecs" in result:
                vec = result["dense_vecs"]
                return vec[0] if hasattr(vec, "__len__") else vec
            return None
        # array-like: take first (list-of-arrays case)
        if hasattr(result, "__len__"):
            return result[0]
        return result
    except Exception as e:
        print(f"[BGE encode] 失败: {e}")
        return None


_encoder = _load_bge_encoder()


# ────────────────────────────────────────────
# 公共 LLM 调用
# ────────────────────────────────────────────

def _call_llm(
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 512,
    model: str = DEFAULT_LLM_MODEL,
    api_key: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """
    调用 DashScope OpenAI 兼容接口（与 ``query_expander._call_llm`` 行为一致）。

    Params:
        prompt: 用户输入
        system: 系统提示词（可选）
        temperature: 采样温度
        max_tokens: 最大输出 token 数
        model: DashScope 模型名
        api_key: DashScope API Key，**必传**，由上游调用方注入（如 HTTP 接口、
                interactive_qa.py 等），不在本模块硬编码，不读环境变量兜底，
                避免上传到服务器后泄露本地 Key。
        timeout: HTTP 超时秒数

    Returns:
        LLM 输出的文本（失败时返回空字符串）。
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "input": {"messages": messages},
        "parameters": {"temperature": temperature, "max_tokens": max_tokens},
    }

    if not api_key:
        raise ValueError(
            "[PromptIterationOptimizer] api_key 必须由调用方显式传入，"
            "不允许在本模块硬编码或从环境变量兜底，避免上传服务器后泄露。"
        )

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["output"]["text"]
    except Exception as e:
        print(f"[PromptIterationOptimizer LLM] 调用失败: {e}")
        return ""


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """两个向量的余弦相似度（零向量安全）。"""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ════════════════════════════════════════════════════════════════
# 能力 1：子查询扩展
# ════════════════════════════════════════════════════════════════

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


class SubqueryGenerator:
    """
    调用 LLM 把单条 query 拆成 2~4 个语义独立的子查询。
    来自 chapter3 step2 PromptIterator 的提示词迭代思路。

    用法：
        gen = SubqueryGenerator()
        sub_queries = gen.generate(query)
    """

    def __init__(
        self,
        system_prompt: str = SUBQUERY_SYSTEM,
        template: str = SUBQUERY_TEMPLATE,
        llm_temperature: float = 0.3,
        max_tokens: int = 256,
        model: str = DEFAULT_LLM_MODEL,
        api_key: Optional[str] = None,
    ):
        self.system_prompt = system_prompt
        self.template = template
        self.llm_temperature = llm_temperature
        self.max_tokens = max_tokens
        self.model = model
        self.api_key = api_key

    def generate(self, query: str, api_key: Optional[str] = None) -> List[str]:
        """
        对一条 query 生成子查询列表。失败兜底为 ``[query]``。

        api_key: 显式 DashScope Key，优先于构造时的 ``self.api_key``。
        """
        prompt = self.template.format(query=query)
        raw = _call_llm(
            prompt,
            system=self.system_prompt,
            temperature=self.llm_temperature,
            max_tokens=self.max_tokens,
            model=self.model,
            api_key=api_key or self.api_key,
        )

        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        sub_queries = []
        for line in lines:
            # 过滤掉太短的行或明显是解释性文字
            if 5 < len(line) < 120:
                sub_queries.append(line)

        # 兜底：原始 query 本身
        if not sub_queries:
            sub_queries = [query]

        return sub_queries


# ════════════════════════════════════════════════════════════════
# 能力 2：实体术语提取
# ════════════════════════════════════════════════════════════════

ENTITY_EXTRACT_PROMPT = """从以下问题中提取关键词实体和术语（不超过 10 个），用 | 分隔：
{query}
直接输出术语列表，不要解释："""


class EntityExtractor:
    """
    从 query 中抽取 ≤10 个关键实体/术语。
    主要用途：在 do_search 之外追加"实体增强"，提升维度约束匹配的召回率。
    """

    def __init__(
        self,
        template: str = ENTITY_EXTRACT_PROMPT,
        llm_temperature: float = 0.1,
        max_tokens: int = 64,
        max_terms: int = 10,
        model: str = DEFAULT_LLM_MODEL,
        api_key: Optional[str] = None,
    ):
        self.template = template
        self.llm_temperature = llm_temperature
        self.max_tokens = max_tokens
        self.max_terms = max_terms
        self.model = model
        self.api_key = api_key

    def extract(self, query: str, api_key: Optional[str] = None) -> List[str]:
        prompt = self.template.format(query=query)
        raw = _call_llm(
            prompt,
            temperature=self.llm_temperature,
            max_tokens=self.max_tokens,
            model=self.model,
            api_key=api_key or self.api_key,
        )
        terms = [t.strip() for t in raw.split("|") if t.strip()]
        return terms[: self.max_terms]


# ════════════════════════════════════════════════════════════════
# 能力 3：聚类匹配 + 优化 PromptModule 选择
# ════════════════════════════════════════════════════════════════

def _load_cluster_data() -> Tuple[Dict[int, np.ndarray], Dict[int, List[Dict]]]:
    """
    加载 chapter3_backup 景区数据训练好的聚类中心和优化 prompt。

    Returns:
        centers:  {cluster_id: np.ndarray(center_vector)}
        prompts:  {cluster_id: List[PromptModule dict]}

    若文件不存在，返回 ``({}, {})``（优雅降级）。
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
            print(f"[ClusterPromptSelector] 加载聚类中心失败: {e}")

    # 2) 加载优化 prompt——优先按 question→cluster 映射加载
    if not ITERATION_RESULTS_DIR.exists():
        return centers, prompts

    mapping_file = (
        ITERATION_RESULTS_DIR / "tourist_question_cluster_mapping.json"
    )
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
            print(f"[ClusterPromptSelector] 加载 question-cluster 映射失败: {e}")

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
        print(f"[ClusterPromptSelector] 加载优化 prompt 失败: {e}")

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
                merged_pm: Dict[str, Any] = {}
                keys = set().union(*[p.keys() for p in all_pms])
                for k in keys:
                    vals = [p[k] for p in all_pms if k in p and p[k]]
                    if vals:
                        merged_pm[k] = "\n---\n".join(vals)
                for cid in centers.keys():
                    prompts[cid] = [
                        {"_merged": True, "_mappings": len(all_pms), **merged_pm}
                    ]
        except Exception as e:
            print(f"[ClusterPromptSelector] 退化聚合 prompt 失败: {e}")

    return centers, prompts


class ClusterPromptSelector:
    """
    用 BGE 把 query 编码成向量，与 chapter3_backup 训练好的聚类中心做余弦相似度，
    返回 Top-N 最相似的 ``(cluster_id, similarity)`` 及其对应 PromptModule。

    Params:
        top_n: 默认返回 Top-2，与 interactive_qa.py 的多 Prompt 融合策略一致
        encoder: 可选 BGE 编码器（默认用模块级 ``_encoder``）
    """

    def __init__(self, top_n: int = 2, encoder=None):
        self.top_n = top_n
        self.encoder = encoder
        self._centers: Dict[int, np.ndarray] = {}
        self._prompts: Dict[int, List[Dict]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._centers, self._prompts = _load_cluster_data()
        self._loaded = True

    # —— 公开 API ——

    def is_available(self) -> bool:
        """是否有可用的聚类中心 + 编码器。"""
        self._ensure_loaded()
        return bool(self._centers) and (self.encoder or _encoder) is not None

    def find_top(self, query: str) -> Tuple[List[Tuple[int, float]], List[Optional[Dict]]]:
        """
        返回 ``(top_clusters, top_prompts)``：
            top_clusters: ``[(cluster_id, similarity), ...]`` 长度 ≤ top_n
            top_prompts:  与 top_clusters 一一对应的 PromptModule，缺失处为 None
        """
        self._ensure_loaded()
        if not self._centers:
            return [], []
        encoder = self.encoder or _encoder
        if encoder is None:
            return [], []

        q_vec = _encode_query(encoder, query)
        if q_vec is None:
            return [], []

        q_emb = np.array(q_vec, dtype=np.float32)
        scored = [
            (cid, cosine_sim(q_emb, center))
            for cid, center in self._centers.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: self.top_n]

        prompts = [
            (self._prompts.get(cid, []) or [None])[0] for cid, _ in top
        ]
        return top, prompts

    def find_best(self, query: str) -> Tuple[Optional[int], float]:
        """返回最近一个聚类的 ``(cluster_id, similarity)``，无数据返回 ``(None, 0.0)``。"""
        top, _ = self.find_top(query)
        if top:
            return top[0]
        return None, 0.0

    def get_prompt(self, cluster_id: int) -> Optional[Dict]:
        """返回指定 cluster 的代表 PromptModule。"""
        self._ensure_loaded()
        items = self._prompts.get(cluster_id, [])
        return items[0] if items else None


# ════════════════════════════════════════════════════════════════
# 主类：组合 3 个能力 → 与 ``QueryExpander.expand()`` 等价
# ════════════════════════════════════════════════════════════════

class PromptIterationOptimizer:
    """
    把一条 query 走完三层优化：

        Step 1: 子查询扩展（LLM 生成 2~4 个子查询，可选缓存）
        Step 2: 实体术语提取（用于增强检索）
        Step 3: 聚类匹配 → 优化 PromptModule 选择（Top-N，可空）

    返回的 dict 字段与原 ``QueryExpander.expand()`` 完全一致，
    因此可作为 ``interactive_qa.py`` 的 drop-in 替换。

    用法（替换示例）：

        # 老代码
        # from query_expander import QueryExpander, build_fusion_query
        # expander = QueryExpander(use_llm_subqueries=True, use_cluster_prompt=True,
        #                          subquery_cache_file=".../subquery_cache.json")
        # expand_result = expander.expand(query)

        # 新代码（其余 do_search / do_qa / 透传字段完全不需要改）
        # from prompt_iteration_optimizer import (
        #     PromptIterationOptimizer, build_fusion_query,
        # )
        # optimizer = PromptIterationOptimizer(use_llm_subqueries=True,
        #                                       use_cluster_prompt=True,
        #                                       subquery_cache_file=".../subquery_cache.json")
        # expand_result = optimizer.optimize(query)   # = optimizer.expand(query) 别名
    """

    def __init__(
        self,
        use_llm_subqueries: bool = True,
        use_entity_extraction: bool = True,
        use_cluster_prompt: bool = True,
        subquery_cache_file: Optional[str] = None,
        cluster_top_n: int = 2,
        api_key: Optional[str] = None,
        subquery_generator: Optional[SubqueryGenerator] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        cluster_selector: Optional[ClusterPromptSelector] = None,
    ):
        """
        Params:
            use_llm_subqueries:      True=调用 LLM 生成子查询
            use_entity_extraction:   True=调用 LLM 提取实体术语
            use_cluster_prompt:      True=加载聚类中心 + PromptModule，做 Top-N 匹配
            subquery_cache_file:     子查询缓存文件路径（批量推理加速用，可选）
            cluster_top_n:           聚类匹配的 Top-N（默认 2，与多 Prompt 融合对齐）
            api_key:                 DashScope API Key，可选；如未提供则要求
                                      ``expand/optimize`` 的调用方显式传入
            subquery_generator:      自定义 SubqueryGenerator（默认按内部默认参数构造）
            entity_extractor:        自定义 EntityExtractor
            cluster_selector:        自定义 ClusterPromptSelector
        """
        self.use_llm_subqueries = use_llm_subqueries
        self.use_entity_extraction = use_entity_extraction
        self.use_cluster_prompt = use_cluster_prompt
        self.subquery_cache_file = subquery_cache_file
        self.cluster_top_n = cluster_top_n
        self.api_key = api_key

        # 三个子能力（支持依赖注入，便于单独测试/替换）
        self.subquery_generator = subquery_generator or SubqueryGenerator(api_key=api_key)
        self.entity_extractor = entity_extractor or EntityExtractor(api_key=api_key)
        self.cluster_selector = cluster_selector or ClusterPromptSelector(
            top_n=cluster_top_n
        )

        # 子查询缓存
        self._subquery_cache: Dict[str, List[str]] = {}
        self._load_cache()

    # ----- 子查询缓存 -----

    def _load_cache(self) -> None:
        if self.subquery_cache_file and Path(self.subquery_cache_file).exists():
            try:
                with open(self.subquery_cache_file, "r", encoding="utf-8") as f:
                    self._subquery_cache = json.load(f)
            except Exception:
                pass

    def _save_cache(self) -> None:
        if not self.subquery_cache_file:
            return
        try:
            target = Path(self.subquery_cache_file)
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                json.dump(self._subquery_cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ----- 主入口 -----

    def optimize(self, query: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """= ``expand()``，主入口。返回字段与旧 ``QueryExpander.expand()`` 对齐。

        ``api_key``：DashScope Key，优先于构造时的 ``self.api_key``。
        """
        return self.expand(query, api_key=api_key)

    def expand(self, query: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        三步优化串联，返回 dict：
            {
                "original_query":  str,
                "sub_queries":     List[str],
                "entity_terms":    List[str],
                "top_clusters":    List[(cluster_id, sim)],
                "top_prompts":     List[Optional[Dict]],
                "cluster_id":      Optional[int],
                "cluster_sim":     float,
                "prompt_module":   Optional[Dict],
                "expand_time":     float,
            }
        """
        t0 = time.time()
        effective_api_key = api_key or self.api_key

        # ── Step 1: 子查询扩展（带缓存） ──
        if self.use_llm_subqueries:
            if query in self._subquery_cache:
                sub_queries = self._subquery_cache[query]
            else:
                sub_queries = self.subquery_generator.generate(
                    query, api_key=effective_api_key
                )
                self._subquery_cache[query] = sub_queries
                self._save_cache()
        else:
            sub_queries = [query]

        # ── Step 2: 实体术语提取 ──
        if self.use_entity_extraction:
            entity_terms = self.entity_extractor.extract(
                query, api_key=effective_api_key
            )
        else:
            entity_terms = []

        # ── Step 3: 聚类匹配 & Prompt 选择（Top-N） ──
        top_clusters: List[Tuple[int, float]] = []
        top_prompts: List[Optional[Dict]] = []
        if self.use_cluster_prompt:
            top_clusters, top_prompts = self.cluster_selector.find_top(query)

        elapsed = time.time() - t0

        return {
            "original_query": query,
            "sub_queries": sub_queries,
            "entity_terms": entity_terms,
            "top_clusters": top_clusters,            # List[(cluster_id, sim)]
            "top_prompts": top_prompts,             # List[Optional[PromptModule]]
            "cluster_id": top_clusters[0][0] if top_clusters else None,
            "cluster_sim": top_clusters[0][1] if top_clusters else 0.0,
            "prompt_module": top_prompts[0] if top_prompts else None,
            "optimize_time": round(elapsed, 2),
        }


# ────────────────────────────────────────────
# 与 do_search / do_qa 衔接的便捷函数
# ────────────────────────────────────────────

def build_fusion_query(expand_result: Dict) -> str:
    """
    把 ``optimize()`` 拿到的 ``sub_queries`` 拼成多路检索信号，
    供 ``do_search`` 直接使用。

    策略：用 ``" | "`` 连接各子查询，使检索器在同一 query 内感知多意图。
    """
    subs = expand_result["sub_queries"]
    if len(subs) == 1:
        return subs[0]
    return " | ".join(subs)


def build_qa_system_prompt(expand_result: Dict, base_prompt: str = None) -> str:
    """
    若命中聚类且有 PromptModule，优先使用；
    否则回退到 ``base_prompt``（即 ``interactive_qa.LLM_SYSTEM_PROMPT``）。
    预留接口：未来可按 ``prompt_module`` 字段自定义拼接。
    """
    pm = expand_result.get("prompt_module")
    if pm and base_prompt:
        return base_prompt
    return base_prompt or ""


# ────────────────────────────────────────────
# CLI：独立运行便于调试
# ────────────────────────────────────────────

def _cli_demo(queries: List[str], show_full: bool = False) -> None:
    """
    独立运行本脚本时，打印每条 query 的优化结果。

    Examples:
        python prompt_iteration_optimizer.py "南孔庙的开放时间？"
        python prompt_iteration_optimizer.py "南孔庙的开放时间？" "雁荡山有哪些特色？"
    """
    print("=" * 70)
    print("  Prompt Iteration Optimizer — 独立运行模式")
    print("  训练产物目录:", CHAPTER3_ROOT)
    print("  聚类中心文件:", CLUSTER_CENTERS_FILE)
    print("  迭代结果目录:", ITERATION_RESULTS_DIR)
    print("  BGE 编码器  :", "已加载" if _encoder else "未加载")
    print("=" * 70)

    optimizer = PromptIterationOptimizer()

    for q in queries:
        print(f"\n>>> {q}")
        result = optimizer.optimize(q)

        print(f"  ├─ sub_queries ({len(result['sub_queries'])}):")
        for i, sq in enumerate(result["sub_queries"], 1):
            print(f"  │   {i}. {sq}")

        print(f"  ├─ entity_terms ({len(result['entity_terms'])}):")
        print(f"  │   {' | '.join(result['entity_terms']) or '(none)'}")

        print(f"  ├─ cluster_id={result['cluster_id']}  sim={result['cluster_sim']:.4f}")
        if result["top_clusters"]:
            for (cid, sim), pm in zip(
                result["top_clusters"], result["top_prompts"]
            ):
                tag = f"cluster-{cid}  sim={sim:.4f}"
                pm_keys = list(pm.keys()) if pm else []
                print(f"  ├─ {tag}  prompt_keys={pm_keys}")
                if show_full and pm:
                    for k, v in pm.items():
                        preview = (str(v)[:80] + "…") if v and len(str(v)) > 80 else v
                        print(f"  │     · {k}: {preview}")
    else:
        print("  ├─ (无聚类数据，PromptModule 为空)")

    print(f"  └─ expand_time={result['expand_time']}s")

    print()


if __name__ == "__main__":
    """
    独立运行：
        python prompt_iteration_optimizer.py "问题1" "问题2" ...

    可选环境变量：
        PROMPT_OPT_FULL=1    打印完整 PromptModule 内容
    """
    args = sys.argv[1:]
    if not args:
        # 默认演示 3 条 query
        args = [
            "南孔庙的开放时间是什么时候？",
            "雁荡山有哪些特色景点？",
            "西湖十景分别是什么？",
        ]
    show_full = bool(int(os.getenv("PROMPT_OPT_FULL", "0")))
    _cli_demo(args, show_full=show_full)
