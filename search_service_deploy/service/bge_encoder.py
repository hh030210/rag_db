# -*- coding: utf-8 -*-
"""
bge_encoder.py
==============
BGE-M3 embedding 加载与编码。
支持：
    - FlagEmbedding.BGEM3FlagModel（首选）
    - sentence-transformers + FlagProxy（回退）
"""

import os
from pathlib import Path
from typing import List, Optional


# BGE 模型候选路径（按优先级）
def _candidate_paths() -> List[Path]:
    here = Path(__file__).resolve().parent
    roots = [
        here.parent,                     # /app/../
        here.parent.parent,              # /app/../../
        Path(os.getenv("BGE_MODEL_PATH", "/models/bge-m3")),
        Path("/models/bge-m3"),
        Path("/opt/search_service/models/bge-m3"),
        Path("/mnt/models/bge-m3"),
        Path("/root/mingqiang/model/bge-m3"),
        Path("/root/app/core/model/bge-m3"),
    ]
    candidates = []
    for root in roots:
        if root.is_dir():
            candidates.append(root)
            candidates.append(root / "bge-m3")
        else:
            candidates.append(root / "bge-m3")
    seen = set()
    out = []
    for c in candidates:
        s = str(c)
        if s in seen:
            continue
        seen.add(s)
        out.append(c)
    return out


def _resolve_model_path() -> Optional[str]:
    # env 优先
    env_path = os.getenv("BGE_MODEL_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    for p in _candidate_paths():
        try:
            if p.exists() and p.is_dir():
                return str(p)
        except Exception:
            continue
    return None


class _FlagProxy:
    """让 sentence-transformers 暴露与 BGEM3FlagModel 兼容的 .encode 接口"""

    def __init__(self, model):
        self._model = model

    def encode(self, texts, return_dense=False, **kwargs):
        kwargs.pop("normalize_embeddings", None)
        kwargs.pop("show_progress_bar", None)
        emb = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if return_dense:
            return {"dense_vecs": emb}
        return emb


def load_bge_encoder():
    """加载 BGE-M3 编码器，失败返回 None"""
    model_path = _resolve_model_path()
    if not model_path:
        print("[bge_encoder] 警告: 未找到 BGE-M3 模型目录")
        return None

    # 优先 FlagEmbedding
    try:
        from FlagEmbedding import BGEM3FlagModel
        return BGEM3FlagModel(model_path, use_fp16=False, device="cpu")
    except ImportError:
        pass
    except Exception as e:
        print(f"[bge_encoder] 警告: FlagEmbedding 加载失败 ({e})")

    # 回退 sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path, local_files_only=True)
        return _FlagProxy(model)
    except Exception as e:
        print(f"[bge_encoder] 警告: SentenceTransformer 加载失败: {e}")
        return None


def encode_query(encoder, text: str) -> Optional[List[float]]:
    """编码单条查询 → 1024 维 list[float]，失败返回 None"""
    if encoder is None:
        return None
    cls_name = encoder.__class__.__name__
    try:
        if cls_name == "_FlagProxy":
            emb = encoder.encode([text], return_dense=True)
            vec = emb["dense_vecs"][0]
        elif cls_name in ("BGEM3FlagModel", "M3Embedder"):
            emb = encoder.encode([text], return_dense=True)
            vec = emb["dense_vecs"][0]
        else:
            emb = encoder.encode([text], normalize_embeddings=True, show_progress_bar=False)
            vec = emb[0]
        if hasattr(vec, "tolist"):
            return vec.tolist()
        return [float(x) for x in vec]
    except Exception as e:
        print(f"[bge_encoder] 编码失败: {e}")
        return None
