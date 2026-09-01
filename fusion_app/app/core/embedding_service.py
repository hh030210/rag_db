"""
Embedding 模型服务

统一管理 BGE-M3 和 BGE-large-zh-v1.5 模型加载和向量编码。
支持 CPU/CUDA 自动选择，支持批量编码。
"""

import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from config import get_fusion_config
    _cfg = get_fusion_config()
except Exception:
    _cfg = None


class EmbeddingService:
    """Embedding 模型服务"""

    _instance: Optional["EmbeddingService"] = None

    def __init__(self, config=None):
        self.cfg = config or _cfg
        self._model_bgem3 = None      # BGE-M3 (RAG_DB_slim)
        self._model_bgelarge = None    # BGE-large-zh-v1.5 (code1)
        self._device = "cpu"

    # ==================== BGE-M3 模型（RAG_DB_slim 风格）====================

    def load_bgem3(self) -> bool:
        """加载 BGE-M3 模型"""
        if self._model_bgem3 is not None:
            return True

        try:
            from sentence_transformers import SentenceTransformer

            model_cfg = self.cfg.embedding if self.cfg else None
            model_path = None

            if model_cfg:
                # 尝试多个路径
                base_paths = [
                    Path(model_cfg.model_path) if model_cfg.model_path else None,
                    _project_root / "model" / "bge-m3",
                    _project_root.parent / "model" / "bge-m3",
                    _project_root.parent.parent / "model" / "bge-m3",
                ]
                for p in base_paths:
                    if p and p.exists():
                        model_path = str(p)
                        break

            if model_path is None:
                model_path = "BAAI/bge-m3"

            self._model_bgem3 = SentenceTransformer(
                model_path,
                device=self._device,
                local_files_only=(model_path not in ["BAAI/bge-m3"]),
            )
            print(f"[Embedding] BGE-M3 加载成功: {model_path}")
            return True

        except Exception as e:
            print(f"[Embedding] BGE-M3 加载失败: {e}")
            self._model_bgem3 = None
            return False

    def encode_bgem3(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """使用 BGE-M3 编码文本"""
        if not self._model_bgem3 and not self.load_bgem3():
            raise RuntimeError("BGE-M3 模型加载失败")

        try:
            emb = self._model_bgem3.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=normalize,
            )
            return emb.tolist()
        except Exception as e:
            print(f"[Embedding] BGE-M3 编码失败: {e}")
            raise

    def encode_query_bgem3(self, query: str) -> List[float]:
        """使用 BGE-M3 编码查询（BGE-M3 建议加查询指令）"""
        prefixed = f"为这个句子生成表示以用于检索相关文章：{query}"
        return self.encode_bgem3([prefixed])[0]

    @property
    def bgem3_dimension(self) -> int:
        """BGE-M3 向量维度"""
        if self._model_bgem3:
            return self._model_bgem3.get_sentence_embedding_dimension()
        return 1024

    # ==================== BGE-large-zh-v1.5 模型（code1 风格）====================

    def load_bgelarge(self) -> bool:
        """加载 BGE-large-zh-v1.5 模型"""
        if self._model_bgelarge is not None:
            return True

        try:
            from sentence_transformers import SentenceTransformer

            model_cfg = self.cfg.embedding_legacy if self.cfg else None
            model_path = "BAAI/bge-large-zh-v1.5"

            if model_cfg and hasattr(model_cfg, "model_path"):
                p = Path(model_cfg.model_path)
                if p.exists():
                    model_path = str(p)

            self._model_bgelarge = SentenceTransformer(
                model_path,
                device=self._device,
                local_files_only=True,
            )
            print(f"[Embedding] BGE-large-zh-v1.5 加载成功: {model_path}")
            return True

        except Exception as e:
            print(f"[Embedding] BGE-large-zh-v1.5 加载失败: {e}")
            self._model_bgelarge = None
            return False

    def encode_bgelarge(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """使用 BGE-large 编码文本（带查询指令）"""
        if not self._model_bgelarge and not self.load_bgelarge():
            raise RuntimeError("BGE-large 模型加载失败")

        try:
            instruction = "为这个句子生成表示以用于检索相关文章："
            prefixed = [f"{instruction}{t}" for t in texts]
            emb = self._model_bgelarge.encode(
                prefixed,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=normalize,
            )
            return emb.tolist()
        except Exception as e:
            print(f"[Embedding] BGE-large 编码失败: {e}")
            raise

    def encode_query_bgelarge(self, query: str) -> List[float]:
        """使用 BGE-large 编码查询"""
        instruction = "为这个句子生成表示以用于检索相关文章："
        prefixed = f"{instruction}{query}"
        return self.encode_bgelarge([prefixed])[0]

    @property
    def bgelarge_dimension(self) -> int:
        """BGE-large 向量维度"""
        if self._model_bgelarge:
            return self._model_bgelarge.get_sentence_embedding_dimension()
        return 1024

    # ==================== 统一接口 ====================

    def encode(self, texts: List[str], model: str = "bgem3", normalize: bool = True) -> List[List[float]]:
        """统一编码接口"""
        if model == "bgem3":
            return self.encode_bgem3(texts, normalize=normalize)
        elif model == "bgelarge":
            return self.encode_bgelarge(texts, normalize=normalize)
        else:
            raise ValueError(f"不支持的模型: {model}")

    def encode_query(self, query: str, model: str = "bgem3") -> List[float]:
        """统一查询编码接口"""
        if model == "bgem3":
            return self.encode_query_bgem3(query)
        elif model == "bgelarge":
            return self.encode_query_bgelarge(query)
        else:
            raise ValueError(f"不支持的模型: {model}")

    def get_dimension(self, model: str = "bgem3") -> int:
        """获取指定模型的向量维度"""
        if model == "bgem3":
            return self.bgem3_dimension
        elif model == "bgelarge":
            return self.bgelarge_dimension
        return 1024

    def preload_all(self) -> Dict[str, bool]:
        """预加载所有模型"""
        return {
            "bgem3": self.load_bgem3(),
            "bgelarge": self.load_bgelarge(),
        }

    def unload_all(self):
        """卸载所有模型（释放内存）"""
        self._model_bgem3 = None
        self._model_bgelarge = None
        import gc
        gc.collect()


# ==================== 单例管理器 ====================

_embedding_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """获取 Embedding 服务实例"""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = EmbeddingService()
    return _embedding_instance


def init_embedding(config=None) -> EmbeddingService:
    """初始化 Embedding 服务"""
    global _embedding_instance
    _embedding_instance = EmbeddingService(config)
    return _embedding_instance
