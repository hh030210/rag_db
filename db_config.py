"""
db_config.py

从 db_config.yaml 读取配置，提供统一的配置访问接口。
支持环境变量替换，如 ${ENV_VAR} 或 ${ENV_VAR:-default}
"""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from types import SimpleNamespace

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent


def _expand_env_vars(value: Any) -> Any:
    """
    递归展开字符串中的环境变量引用。
    支持格式: ${VAR} 和 ${VAR:-default}
    """
    if isinstance(value, str):
        # 匹配 ${VAR} 或 ${VAR:-default}
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


class DBConfig:
    """数据库配置类"""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = PROJECT_ROOT / "db_config.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        # 展开环境变量
        self._config = _expand_env_vars(raw_config)

    def _to_namespace(self, d: Dict) -> SimpleNamespace:
        """将字典转换为 SimpleNamespace"""
        if d is None:
            return SimpleNamespace()
        if not isinstance(d, dict):
            return d
        return SimpleNamespace(**{k: self._to_namespace(v) for k, v in d.items()})

    @property
    def rdb(self) -> SimpleNamespace:
        """关系型数据库配置"""
        return self._to_namespace(self._config.get("rdb", {}))

    @property
    def vecdb(self) -> SimpleNamespace:
        """向量数据库配置"""
        return self._to_namespace(self._config.get("vecdb", {}))

    @property
    def embedding(self) -> SimpleNamespace:
        """Embedding 模型配置"""
        return self._to_namespace(self._config.get("embedding", {}))

    @property
    def api(self) -> SimpleNamespace:
        """API 配置"""
        return self._to_namespace(self._config.get("api", {}))

    @property
    def llm(self) -> SimpleNamespace:
        """LLM 配置"""
        return self._to_namespace(self._config.get("llm", {}))

    @property
    def qgen(self) -> SimpleNamespace:
        """问题生成配置"""
        return self._to_namespace(self._config.get("qgen", {}))

    @property
    def vecdb_qdrant(self) -> SimpleNamespace:
        """Qdrant 向量数据库配置"""
        return self._to_namespace(self._config.get("vecdb_qdrant", {}))


# 单例模式
_config_instance: Optional[DBConfig] = None


def get_config(config_path: str = None) -> DBConfig:
    """
    获取配置单例

    Args:
        config_path: 配置文件路径，默认使用项目根目录的 db_config.yaml

    Returns:
        DBConfig 实例
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = DBConfig(config_path)
    return _config_instance


if __name__ == "__main__":
    # 测试配置读取
    config = get_config()
    print("RDB Config:", config.rdb)
    print("VecDB Config:", config.vecdb)
    print("Embedding Config:", config.embedding)
