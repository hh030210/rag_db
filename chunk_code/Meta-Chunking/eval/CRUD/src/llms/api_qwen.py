"""
硅基流动 SiliconFlow / 任意 OpenAI-compatible API 调用 Qwen / GLM 等模型。
"""

import os
import threading
import time
from openai import OpenAI
from loguru import logger
from src.llms.base import BaseLLM
from importlib import import_module

try:
    conf = import_module("src.configs.real_config")
except ImportError:
    conf = import_module("src.configs.config")


# Conservative adaptive rate limiting:默认请求起始间隔至少 10 秒，避免
# SiliconFlow/DashScope 的 RPM/TPM 限制；成功后也不会低于该下限。
_LLM_RATE_LOCK = threading.Lock()
_LLM_LAST_CALL_TS = [0.0]
_LLM_MIN_INTERVAL = max(0.1, float(os.environ.get("QWEN_MIN_INTERVAL", "10")))
_LLM_ADAPTIVE_INTERVAL = [_LLM_MIN_INTERVAL]


def _rate_limit_sleep():
    with _LLM_RATE_LOCK:
        now = time.time()
        interval = _LLM_ADAPTIVE_INTERVAL[0]
        wait = interval - (now - _LLM_LAST_CALL_TS[0])
        if wait > 0:
            time.sleep(wait)
        _LLM_LAST_CALL_TS[0] = time.time()


class Qwen_API_Chat(BaseLLM):
    """通过 OpenAI-compatible API 调用 LLM(硅基流动 / DashScope / vLLM / Ollama 等)"""

    def __init__(self, model_name='qwen_api', temperature=0.1, max_new_tokens=1280,
                 top_p=0.9, top_k=5, **kwargs):
        super().__init__(model_name, temperature, max_new_tokens, top_p, top_k, **kwargs)

        api_key = conf.Qwen_OpenAI_API_Key
        api_base = conf.Qwen_OpenAI_API_Base
        model_name = conf.Qwen_OpenAI_Model_Name

        if not api_key or not api_base:
            raise ValueError(
                "请在 src/configs/real_config.py 中配置 Qwen_OpenAI_API_Key 和 Qwen_OpenAI_API_Base。\n"
                "示例:\n"
                "  Qwen_OpenAI_API_Key = 'sk-xxxxxx'\n"
                "  Qwen_OpenAI_API_Base = 'https://api.siliconflow.cn/v1'\n"
                "  Qwen_OpenAI_Model_Name = 'Pro/zai-org/GLM-4.7'  # 或 'Qwen/Qwen2.5-7B-Instruct' 等"
            )

        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name
        self.params['model_name'] = model_name

    def request(self, query: str) -> str:
        last_err = None
        for attempt in range(5):
            try:
                _rate_limit_sleep()
                request_kwargs = dict(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是一个有用的助手"},
                        {"role": "user", "content": query},
                    ],
                    temperature=self.params['temperature'],
                    max_tokens=self.params['max_new_tokens'],
                    top_p=self.params['top_p'],
                )
                thinking = os.environ.get("QWEN_ENABLE_THINKING")
                if thinking is not None:
                    request_kwargs["extra_body"] = {
                        "enable_thinking": thinking.strip().lower()
                        in {"1", "true", "yes", "on"}
                    }
                response = self.client.chat.completions.create(**request_kwargs)
                content = response.choices[0].message.content
                # 自适应降低间隔:上次成功则缩短
                with _LLM_RATE_LOCK:
                    _LLM_ADAPTIVE_INTERVAL[0] = max(
                        _LLM_MIN_INTERVAL, _LLM_ADAPTIVE_INTERVAL[0] * 0.8
                    )
                return content
            except Exception as e:
                last_err = e
                msg = str(e)
                # 429/5xx 时退避重试
                if any(kw in msg for kw in ['429', 'rate limit', 'TPM', 'RPM',
                                              '500', '502', '503', '504']):
                    wait = min(180, (2 ** attempt) * max(5, _LLM_MIN_INTERVAL * 2))
                    # 自适应拉长间隔
                    with _LLM_RATE_LOCK:
                        _LLM_ADAPTIVE_INTERVAL[0] = min(5.0, _LLM_ADAPTIVE_INTERVAL[0] * 1.5)
                    logger.warning(f"[LLM retry] attempt={attempt+1}, wait={wait}s, "
                                  f"adaptive_interval={_LLM_ADAPTIVE_INTERVAL[0]:.2f}s, err={msg[:150]}")
                    time.sleep(wait)
                    continue
                logger.warning(f"LLM API 请求失败: {e}")
                raise
        logger.warning(f"LLM API 5 次重试后仍失败: {last_err}")
        raise last_err
