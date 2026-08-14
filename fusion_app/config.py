"""
融合系统配置模块

读取 config.yaml，统一管理系统配置。
支持环境变量替换和配置合并。
"""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).parent.parent


def _expand_env_vars(value: Any) -> Any:
    """递归展开字符串中的环境变量引用"""
    if isinstance(value, str):
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
        def replacer(match):
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default)
        return re.sub(pattern, replacer, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


class FusionConfig:
    """融合系统配置类"""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = PROJECT_ROOT / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._config = _expand_env_vars(raw)

    def _ns(self, d: Dict) -> SimpleNamespace:
        if d is None:
            return SimpleNamespace()
        if not isinstance(d, dict):
            return d
        return SimpleNamespace(**{k: self._ns(v) for k, v in d.items()})

    @property
    def rdb(self) -> SimpleNamespace:
        return self._ns(self._config.get("rdb", {}))

    @property
    def vecdb_qdrant(self) -> SimpleNamespace:
        return self._ns(self._config.get("vecdb_qdrant", {}))

    @property
    def embedding(self) -> SimpleNamespace:
        return self._ns(self._config.get("embedding", {}))

    @property
    def embedding_legacy(self) -> SimpleNamespace:
        return self._ns(self._config.get("embedding_legacy", {}))

    @property
    def llm(self) -> SimpleNamespace:
        return self._ns(self._config.get("llm", {}))

    @property
    def llm_qwen3(self) -> SimpleNamespace:
        return self._ns(self._config.get("llm_qwen3", {}))

    @property
    def rag(self) -> SimpleNamespace:
        return self._ns(self._config.get("rag", {}))

    @property
    def retrieval(self) -> SimpleNamespace:
        return self._ns(self._config.get("retrieval", {}))

    @property
    def code1_data(self) -> SimpleNamespace:
        return self._ns(self._config.get("code1_data", {}))

    @property
    def server(self) -> SimpleNamespace:
        return self._ns(self._config.get("server", {}))

    @property
    def file_upload(self) -> SimpleNamespace:
        return self._ns(self._config.get("file_upload", {}))

    @property
    def pipeline(self) -> SimpleNamespace:
        return self._ns(self._config.get("pipeline", {}))


_config_instance: Optional[FusionConfig] = None


def get_fusion_config(config_path: str = None) -> FusionConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = FusionConfig(config_path)
    return _config_instance


if __name__ == "__main__":
    cfg = get_fusion_config()
    print("Qdrant:", cfg.vecdb_qdrant)
    print("LLM:", cfg.llm)
    print("Code1 data:", cfg.code1_data)
