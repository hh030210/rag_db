from typing import Callable

import evaluate
import jieba
from loguru import logger
from text2vec import Similarity
from evaluate.utils import file_utils as _efile

# 离线补丁：evaluate 库在 load() 时会对 local relative imports 做 HEAD 验证，
# 离线环境可能无响应；我们直接把 HEAD 行为跳过（用 fake ETag），文件本身已在本地缓存。
_orig_http_head = _efile.http_head
def _offline_http_head(url, *args, **kwargs):
    # 本地路径直接返回 fake response
    if not (url.startswith("http://") or url.startswith("https://")):
        class _FakeResp:
            ok = True
            status_code = 200
            headers = {"ETag": "offline"}
            def raise_for_status(self): pass
        return _FakeResp()
    return _orig_http_head(url, *args, **kwargs)
_efile.http_head = _offline_http_head

# 缓存 evaluate 加载的 metric，避免每次调用重复加载
_loaded_metrics = {}
_metric_lock = None
def _get_metric(name):
    global _metric_lock
    if _metric_lock is None:
        import threading
        _metric_lock = threading.Lock()
    if name not in _loaded_metrics:
        with _metric_lock:
            if name not in _loaded_metrics:  # double-check
                _loaded_metrics[name] = evaluate.load(f'src/.cache/huggingface/{name}')
    return _loaded_metrics[name]

# 模块导入时立即预加载 metric（线程安全 + 避免多线程 race）
try:
    for _m in ('bleu', 'rouge'):
        _get_metric(_m)
except Exception as _e:
    logger.warning(f"preload metrics failed (will retry on first use): {_e}")

# evaluate 的 Bleu.compute() / Rouge.compute() 内部不是线程安全的，
# 在多线程下可能出现 'NoneType is not iterable' 等问题。
# 用一把全局锁串行化 compute 调用。
_compute_lock = None
def _locked_compute(metric, **kwargs):
    global _compute_lock
    if _compute_lock is None:
        import threading
        _compute_lock = threading.Lock()
    with _compute_lock:
        return metric.compute(**kwargs)


def catch_all_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.warning(repr(e))
    return wrapper


@catch_all_exceptions
def bleu_score(
    continuation: str,
    reference: str,
    with_penalty = False
) -> tuple[float, float, float, float, float]:
    # 关键修复:空字符串会导致 evaluate.bleu 内部 ZeroDivisionError
    if not (continuation and continuation.strip()) or not (reference and reference.strip()):
        return (0.0, 0.0, 0.0, 0.0, 0.0) if with_penalty else 0.0
    f = lambda text: list(jieba.cut(text))
    bleu = _get_metric('bleu')
    results = _locked_compute(bleu, predictions=[continuation], references=[[reference]], tokenizer=f)

    bleu_avg = results['bleu'] or 0.0
    bleu1, bleu2, bleu3, bleu4 = (results['precisions'] + [0.0] * 4)[:4]
    brevity_penalty = results['brevity_penalty'] or 0.0

    # 所有任务评分器都按五元组解包；旧实现的默认分支只返回一个
    # float，导致 BLEU 计算成功后仍被上层异常处理降为全零。
    if not with_penalty and brevity_penalty:
        bleu_avg = bleu_avg / brevity_penalty
    return (
        float(bleu_avg),
        float(bleu1 or 0.0),
        float(bleu2 or 0.0),
        float(bleu3 or 0.0),
        float(bleu4 or 0.0),
    )


@catch_all_exceptions
def rougeL_score(
    continuation: str,
    reference: str
) -> float:
    f = lambda text: list(jieba.cut(text))
    rouge = _get_metric('rouge')
    results = _locked_compute(rouge, predictions=[continuation], references=[[reference]], tokenizer=f, rouge_types=['rougeL'])
    score = results['rougeL']
    return score


@catch_all_exceptions
def kw_precision(
    continuation: str,
    reference: str,
    kw_extracter: Callable[[str], list[str]],
    with_kw_list: bool = True
) -> float | tuple[float, list[str], list[str]]:
    """Measure the rationality of a generated continuation sentence with respect to the original news object."""
    kws = kw_extracter(continuation)
    if len(kws) == 0:
        return 0, [], [] if with_kw_list else 0
    appeared_kws = [kw for kw in kws if kw in reference]
    precision = len(appeared_kws) / len(kws)
    return precision, appeared_kws, kws if with_kw_list else precision


@catch_all_exceptions
def bert_score(
    continuation: str,
    reference: str
) -> float:
    """
    Note:
        Requesting the network to connect to Hugging Face. 
    """
    sim = Similarity(model_name_or_path="src/.cache/text2vec-base-chinese")
    score = sim.get_score(continuation, reference)
    return score


def classifications(
    predictions: list[bool],
    references: list[bool]
) -> tuple[float, float, float, float]:
    """
    Calculate accuracy, precision, recall, and F1 in a binary classification problem.

    Args:
        predictions (list[bool]): List of predicted values (0 or 1).
        references (list[bool]): List of true values (0 or 1).

    Returns:
        tuple: Accuracy, precision, recall, and F1 scores.

    """
    true_positive = sum(1 for a, b in zip(references, predictions) if a == 1 and b == 1)
    false_positive = sum(1 for a, b in zip(references, predictions) if a == 0 and b == 1)
    false_negative = sum(1 for a, b in zip(references, predictions) if a == 1 and b == 0)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    accuracy = sum(1 for a, b in zip(references, predictions) if a == b) / len(predictions) if len(predictions) > 0 else 0
    return accuracy, precision, recall, f1
