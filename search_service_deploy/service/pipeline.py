# -*- coding: utf-8 -*-
"""
pipeline.py
===========
编排 3 个 searcher + 自适应权重 + 顶层融合。

输入:
    query_text
    mode: "sem" | "dim" | "fusion"
    alpha_dim, alpha_sem  可选入参覆盖
    fusion_strategy: "adaptive" | "fixed" | "rrf" | "score"
    top_k

输出:
    {
        mode, query_text,
        alpha_dim, alpha_sem, alpha_source,           # 记录权重来源（fixed/adptive）
        dim_results, sem_results, fusion_results,      # 各自最多 8 条
        constraints, weight_breakdown,                # 自适应权重的诊断信息
        elapsed
    }
"""

import math
import time
from typing import Any, Dict, List, Optional

from semantic_searcher import SemanticSearcher
from dim_searcher import DimSearcher
from fusion import (
    compute_adaptive_weights,
    compute_structural_confidence,
    compute_label_evidence_confidence,
    compute_concentration,
    fuse_two,
    fuse_rrf,
    fuse_score,
)


# 默认 top_k（=题目要求的 8）
DEFAULT_TOP_K = 8


class SearchPipeline:
    def __init__(
        self,
        sem_searcher: SemanticSearcher,
        dim_searcher: DimSearcher,
        sem_top_k: int = 20,
        dim_top_k: int = 100,
    ):
        self.sem = sem_searcher
        self.dim = dim_searcher
        self.sem_top_k = sem_top_k
        self.dim_top_k = dim_top_k

    def search(
        self,
        query_text: str,
        mode: str = "sem",
        top_k: int = DEFAULT_TOP_K,
        alpha_dim: Optional[float] = None,
        alpha_sem: Optional[float] = None,
        fusion_strategy: str = "adaptive",
    ) -> Dict[str, Any]:
        """
        mode:
            "sem"     → 纯语义检索，返回 sem_top8
            "dim"     → 纯维度检索，返回 dim_top8
            "fusion"  → 维度+语义融合，返回 三路各 top8 + 最终 top8

        alpha_dim / alpha_sem:
            都给 → 用给的（覆盖权重计算）
            任一给一个 → 用给的，剩下的用 (1 - 此) 计算并校验
            都不给 → 自适应权重（仅 fusion 模式有效，其他模式给 0/1）
        """
        t0 = time.time()
        out: Dict[str, Any] = {
            "mode": mode,
            "query_text": query_text,
            "top_k": top_k,
            "alpha_dim": None,
            "alpha_sem": None,
            "alpha_source": None,         # "fixed" / "adaptive"
            "dim_results": [],
            "sem_results": [],
            "fusion_results": [],
            "constraints": {},
            "weight_breakdown": {},
            "elapsed": 0.0,
        }

        if mode not in ("sem", "dim", "fusion"):
            out["error"] = f"invalid mode: {mode}"
            out["elapsed"] = time.time() - t0
            return out

        # ─── 1. 各路召回 ───
        dim_results: List[Dict[str, Any]] = []
        constraints: Dict[str, Any] = {}
        if mode in ("dim", "fusion"):
            try:
                dim_results, constraints = self.dim.search(query_text, top_k=self.dim_top_k)
            except Exception as e:
                out["dim_error"] = str(e)

        sem_results: List[Dict[str, Any]] = []
        if mode in ("sem", "fusion"):
            try:
                sem_results = self.sem.search(query_text, top_k=self.sem_top_k)
            except Exception as e:
                out["sem_error"] = str(e)

        # ─── 2. 决定 α ───
        explicit_alpha = self._resolve_alpha(alpha_dim, alpha_sem)

        if mode == "fusion":
            if explicit_alpha is not None:
                a_dim, a_sem = explicit_alpha
                source = "fixed"
            else:
                # 自适应权重（综合 P_q / T_q / C_r）
                a_dim, a_sem = compute_adaptive_weights(
                    dim_results, sem_results, constraints, top_k=5,
                )
                source = "adaptive"
        elif mode == "dim":
            a_dim, a_sem, source = 1.0, 0.0, "fixed"
        else:  # sem
            a_dim, a_sem, source = 0.0, 1.0, "fixed"

        out["alpha_dim"] = a_dim
        out["alpha_sem"] = a_sem
        out["alpha_source"] = source

        # ─── 3. 自适应权重诊断（仅当 source=adaptive 时记录）───
        if mode == "fusion" and source == "adaptive":
            P_q = compute_structural_confidence(constraints)
            T_q = compute_label_evidence_confidence(constraints, dim_results)
            C_dim, C_sem = compute_concentration(dim_results[:5], sem_results[:5])
            U_dim = P_q + T_q + C_dim
            U_sem = (1 - P_q) + (1 - T_q) + C_sem
            out["weight_breakdown"] = {
                "structural_confidence_P_q": P_q,
                "label_evidence_T_q": T_q,
                "concentration_C_dim": C_dim,
                "concentration_C_sem": C_sem,
                "utility_U_dim": U_dim,
                "utility_U_sem": U_sem,
            }

        # ─── 4. 顶层融合（仅 fusion 模式）───
        fusion_results: List[Dict[str, Any]] = []
        if mode == "fusion":
            strat = fusion_strategy if fusion_strategy in ("rrf", "score") else "rrf"
            if explicit_alpha is not None and fusion_strategy == "fixed":
                strat = fusion_strategy
            else:
                # adaptive 权重 + RRF 配合更稳定
                strat = "rrf"
            fusion_results = fuse_two(
                dim_results, sem_results,
                alpha=a_dim, top_k=top_k,
                strategy=strat,
                rrf_k=60,
            )

        out["constraints"] = constraints
        out["dim_results"] = (dim_results or [])[:top_k]
        out["sem_results"] = (sem_results or [])[:top_k]
        out["fusion_results"] = fusion_results
        out["elapsed"] = round(time.time() - t0, 4)

        return out

    # ──────────────── 辅助：解析 / 校验外部 α ────────────────

    @staticmethod
    def _resolve_alpha(alpha_dim: Optional[float], alpha_sem: Optional[float]):
        """返回 (a_dim, a_sem) 或者 None（未指定）"""
        if alpha_dim is not None and alpha_sem is not None:
            s = alpha_dim + alpha_sem
            if s <= 0:
                return 0.5, 0.5
            return alpha_dim / s, alpha_sem / s
        if alpha_dim is not None:
            a_dim = max(0.0, min(1.0, float(alpha_dim)))
            return a_dim, 1.0 - a_dim
        if alpha_sem is not None:
            a_sem = max(0.0, min(1.0, float(alpha_sem)))
            return 1.0 - a_sem, a_sem
        return None
