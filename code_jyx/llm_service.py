"""
llm_service.py

基于 DashScope (通义千问) 的 LLM 服务封装。
实现了 dimension_generate、tag_generate、query_parser 三个模块所需的所有 LLM 接口。

也支持 OpenAI 兼容协议（通过环境变量 LLM_OPENAI_COMPAT=1 启用），可用于阿里云 Maas /
SiliconFlow 等 OpenAI 风格 endpoint。

使用前请先设置环境变量：
    export DASHSCOPE_API_KEY="sk-..."
或在使用时直接传入 api_key 参数。

OpenAI 兼容模式：
    export LLM_OPENAI_COMPAT=1
    export LLM_BASE_URL="https://llm-xxx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    export LLM_API_KEY="sk-..."
    export LLM_MODEL="qwen3.6-max-preview"
"""

import os
import re
import json
import threading
import time
from typing import List, Dict, Optional, Any


# ============================================================
# OpenAI 兼容客户端（懒加载）
# ============================================================

_OPENAI_CLIENT_CACHE = {}


def _get_openai_client(base_url: str, api_key: str):
    """懒加载 OpenAI 客户端"""
    cache_key = (base_url, api_key[:12] if api_key else "")
    if cache_key in _OPENAI_CLIENT_CACHE:
        return _OPENAI_CLIENT_CACHE[cache_key]
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "OpenAI 兼容模式需要安装 openai 包，请运行：pip install openai"
        ) from e
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
    _OPENAI_CLIENT_CACHE[cache_key] = client
    return client


def _is_openai_compat_mode() -> bool:
    """是否启用 OpenAI 兼容模式"""
    return os.getenv("LLM_OPENAI_COMPAT", "").strip() in ("1", "true", "TRUE", "yes", "YES")


def _resolve_api_key() -> str:
    """统一解析 API Key：OpenAI 兼容模式优先读 LLM_API_KEY，否则读 DASHSCOPE_API_KEY"""
    if _is_openai_compat_mode():
        return (
            os.getenv("LLM_API_KEY", "")
            or os.getenv("DASHSCOPE_API_KEY", "")
            or ""
        )
    return os.getenv("DASHSCOPE_API_KEY", "")


def _resolve_base_url() -> str:
    """OpenAI 兼容模式下解析 base_url，默认阿里云 Maas"""
    return os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")


def _resolve_model(default: str = "qwen-plus") -> str:
    """OpenAI 兼容模式下解析模型名"""
    if _is_openai_compat_mode():
        return os.getenv("LLM_MODEL", default)
    return default


# dashscope 仅在非 OpenAI 兼容模式时才需要
if not _is_openai_compat_mode():
    try:
        from dashscope import Generation
    except ImportError:
        Generation = None  # 占位，运行时再校验
else:
    Generation = None


# ============================================================
# 提示词模板
# ============================================================

PROMPT_GENERATE_CANDIDATES = """你是一个专业的领域知识结构化专家。

请从以下文档集合中，归纳出能够全面描述该领域知识的所有关键维度（Dimension）。
每个维度应是一个简洁的名词短语（如"适宜人群"、"疾病类别"、"治疗方案"等）。

要求：
1. 维度应具有领域代表性，涵盖该领域的主要信息轴
2. 每个维度的值应当是离散的、可枚举的
3. 优先识别高频共性维度，兼顾领域特殊性
4. 维度之间应尽量独立，避免重复
5. 【禁止】不要生成"其他"、"其他信息"、"备注"、"杂项"等兜底性质的维度 —— 这类维度没有实际检索价值

请直接输出维度列表，用中文逗号分隔，不要包含任何解释：
维度1, 维度2, 维度3, ..."

文档示例（共 {n} 篇，仅供参考）:
---
{docs_snippet}
---
你的回答（维度列表）："""

PROMPT_EXTRACT_SINGLE = """你是一个领域知识抽取专家。

给定一个维度的名称，请从以下文档文本中抽取该维度的值。
如果文档中未提及该维度，请返回 null。

维度名称：{dim_name}
文档文本：
---
{text}
---

请直接输出该维度的值（中文），如果未提及则输出 "NULL"：
"""

PROMPT_EXTRACT_VALUES = """你是一个领域知识结构化评估专家。

请判断下面的文本是否包含“{dim_name}”维度，并提取该维度在文本中明确出现的全部值。

严格要求：
1. 只能依据当前文本，不得使用文本外的信息。
2. 一个文本可能有多个值，必须全部列出。
3. 如果没有明确值，返回空数组。
4. 值应尽量使用文本中的原词，不要改写、概括或推断。
5. 只输出 JSON，不要输出解释或 Markdown。

输出格式：
{{"values": ["值1", "值2"]}}

维度名称：{dim_name}
文本：
---
{text}
---
"""

PROMPT_EXTRACT_BATCH = """你是一个领域知识抽取专家。

给定多个维度的名称，请从以下文档文本中同时抽取每个维度的值。
只输出有明确提及的维度及其值，未提及的维度请忽略。

维度列表：{dims_list}

文档文本：
---
{text}
---

请以 JSON 格式输出，key 为维度名称，value 为该维度的值列表（数组）：
{{"维度名1": ["值1", "值2"], "维度名2": ["值3"]}}
如果某个维度在文档中未提及，请不要包含在输出中。
"""

PROMPT_EXTRACT_MULTI_BATCH = """你是一个领域知识抽取专家。

请分别阅读下面的多条文本，并为每条文本抽取指定维度的值。
只输出文本中有明确依据的维度；没有提及的维度使用空对象。

指定维度：{dims_list}

输出要求：
1. 只输出 JSON 对象，不要输出解释、Markdown 或思考过程。
2. 顶层 key 必须使用输入记录中的原始 ID。
3. 每个 ID 对应一个对象，对象的 key 必须是指定维度名称，value 必须是字符串数组。
4. 不要臆测文本中没有出现的信息。

输出格式示例：
{{"记录ID-1": {{"地理位置": ["衢州"], "建筑功能": ["祭祀孔子"]}}, "记录ID-2": {{}}}}

待处理记录：
{records_block}
"""

PROMPT_VALIDATE_SCHEMA_BATCH = """你是一个知识库 schema 验证专家。

当前任务不是给数据库写标签，而是验证“候选维度名称是否适合成为统一的知识库 schema”。
请分别阅读下面的多条 chunk，只判断指定候选维度在每条文本中是否存在明确证据，并列出
文本中原样出现的值，用于统计该维度的覆盖率和区分度。

严格要求：
1. 只能依据当前 chunk 文本，不得跨 chunk 推断或补全。
2. 只允许使用指定候选维度，不要创造新维度。
3. 没有明确值的维度不要输出。
4. 一个 chunk 的同一维度可以有多个值，全部放入数组。
5. 只输出 JSON，不要解释、Markdown 或思考过程。

候选维度：{dims_list}

输出格式：
{{"records": [{{"id": "原始ID", "dimensions": {{"维度名": ["值1", "值2"]}}}}]}}

待验证 chunk：
{records_block}
"""

PROMPT_KEYWORDS_FALLBACK = """你是一个关键词提取专家。

请从以下文本中提取 3-5 个最重要的关键词或短语，用于摘要描述。
关键词应反映文本的核心主题和关键信息。

文本：
---
{text}
---

请直接输出关键词列表，用中文逗号分隔：
"""

PROMPT_OPTIMIZE_DIMENSION = """你是一个维度工程专家，负责评估和优化知识维度的质量。

当前需要分析的维度：
- 维度名称：{dim_name}
- 问题类型：{issue_type}
- 诊断数据：{metric_data}
- 抽取样本：{samples}

请根据以上信息做出决策：

问题类型说明：
- "低覆盖率"：该维度只在少数文档中出现
- "低辨识度"：该维度的值太单一，缺乏区分能力
- "语义/数据冗余"：该维度与其他维度高度重叠

决策选项（只能选择其中一个）：
1. KEEP - 保留该维度，即使存在问题但仍有一定价值
2. DELETE - 删除该维度，其信息可被其他维度覆盖
3. RENAME - 重命名该维度，用更准确的概念替代（**不能**改名为"其他"等兜底名称）
4. SPLIT - 拆分为多个更细粒度的维度
5. MERGE - 将该维度与其他维度合并为新的维度（**不能**合并为"其他"等兜底名称）

请以 JSON 格式输出你的决策：
{{"action": "KEEP|DELETE|RENAME|SPLIT|MERGE", "reasoning": "决策理由（50字以内）", "new_dimensions": ["新维度1", "新维度2"]（仅 SPLIT/MERGE 时填写）}}
"""

PROMPT_MERGE_WITH_TARGETS = """你是一个维度工程专家，负责评估是否将一个"不达标"的维度与"指定候选目标"中的一个进行合并。

当前需要分析的维度：
- 维度名称：{dim_name}
- 问题类型：{issue_type}
- 诊断数据：{metric_data}
- 抽取样本（前 5 条）：{samples}

【重要】以下是允许的合并目标候选列表（请**只能**从此列表中选择一个作为合并目标，或选择 NOT_MERGE 表示不合并）：
{candidates_block}

任务：
1. 判断"维度 {dim_name}"是否可以合理地合并到候选列表中的某一个维度。
2. 如果可以合并：从候选列表中**精确选择一个目标**（名称必须完全一致），并给出合并后的新维度名称（可以沿用目标名，也可以用更准确的新名称）。
3. 如果都不合适（语义不相关 / 强合会破坏目标维度的纯净度 / 信息确实无价值）：返回 NOT_MERGE。

决策选项（只能选择其中一个）：
1. MERGE - 合并到候选列表中的某个目标维度
2. NOT_MERGE - 不与任何候选合并（保持原样或后续会被删除）

【重要】如果选择 MERGE：
- 合并后的新维度名称**不能**是"其他"、"其他信息"、"备注"、"杂项"等兜底名称
- 新维度必须是有明确语义的具体维度（如"适宜人群"、"疾病类别"等）

请以 JSON 格式输出：
{{"action": "MERGE|NOT_MERGE", "reasoning": "决策理由（50字以内）", "merge_target": "候选列表中的某一个维度名称（仅 MERGE 时填写，必须完全一致）", "new_dimensions": ["合并后的新维度名称"]（仅 MERGE 时填写）}}
"""

PROMPT_PARSE_QUERY = """你是一个查询意图理解专家。

给定一个用户查询和可用的维度列表，请分析该查询涉及了哪些维度及其对应的值。

维度列表（使用这些精确的名称作为 JSON 键）：
{dims_list}
{enum_info}
{schema_constraint}
用户查询："{query_text}"

请分析：
1. 查询中明确或隐含涉及了哪些维度？
2. 每个维度对应的值是什么？

请以 JSON 格式输出，key 必须是上述维度列表中的精确名称，value 为该维度的值列表：
{{"维度名1": ["值1", "值2"], "维度名2": ["值3"]}}
如果查询不涉及某个维度，请不要在输出中包含该维度。
"""


PROMPT_SCHEMA_CONSTRAINT = """
【重要】以下字段名称是 Milvus 数据库中的实际字段名，请务必使用这些精确名称作为 JSON 的 key，切勿自行创造维度名称：
{schema_fields}
"""


def _clip_text(text: str, max_chars: int) -> str:
    """限制提示词长度，同时保留文本首尾，避免只看前缀漏掉后半段维度。"""
    text = str(text or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n...[中间内容省略，仅用于控制长度]...\n{text[-tail:]}"


# ============================================================
# 核心类
# ============================================================

class DimensionMiningWithQwen:
    """
    通义千问驱动的维度挖掘服务。
    封装所有 LLM 调用逻辑，向上游模块提供统一的接口。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = None,
        base_url: Optional[str] = None,
    ):
        """
        Args:
            api_key: API Key。若为 None，则从环境变量读取。
                     OpenAI 兼容模式读 LLM_API_KEY，其他模式读 DASHSCOPE_API_KEY。
            model_name: 使用的模型名称。OpenAI 兼容模式默认从 LLM_MODEL 读取，否则默认 qwen-plus。
            base_url: 仅 OpenAI 兼容模式生效。
        """
        # 判断模式
        self.openai_compat = _is_openai_compat_mode()

        if self.openai_compat:
            self.api_key = api_key or _resolve_api_key()
            self.model_name = model_name or _resolve_model(default="qwen-plus")
            self.base_url = (base_url or _resolve_base_url()).rstrip("/")
        else:
            self.api_key = api_key or _resolve_api_key()
            self.model_name = model_name or "qwen-plus"
            self.base_url = None

        if not self.api_key:
            raise ValueError(
                "未找到 API Key。"
                "请设置环境变量 DASHSCOPE_API_KEY 或 LLM_API_KEY，"
                "或在构造函数中传入 api_key。"
            )

        # 通过环境变量控制请求间隔，避免批量维度抽取触发服务限流。
        # 默认 0 保持原有调用速度；服务器实验显式设置为 2 秒。
        try:
            self.api_interval = max(0.0, float(os.getenv("LLM_API_INTERVAL", "0")))
        except ValueError:
            self.api_interval = 0.0
        self._rate_lock = threading.Lock()
        self._last_request_at = 0.0

    def _wait_for_rate_limit(self):
        """在发起下一次 LLM 请求前等待指定间隔。"""
        if self.api_interval <= 0:
            return
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_at
            wait = self.api_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()

    # ----------------------------------------------------------
    # 内部调用方法
    # ----------------------------------------------------------

    def _call_llm(self, prompt: str, temperature: float = 0.7, timeout: int = 120) -> str:
        """
        通用的 LLM 调用封装。

        Args:
            prompt: 构造好的提示词。
            temperature: 温度参数，控制随机性。
            timeout: 超时秒数。

        Returns:
            LLM 输出的文本内容。
        """
        self._wait_for_rate_limit()
        if self.openai_compat:
            return self._call_llm_openai(prompt, temperature=temperature, timeout=timeout)

        messages = [{"role": "user", "content": prompt}]
        response = Generation.call(
            api_key=self.api_key,
            model=self.model_name,
            messages=messages,
            result_format="message",
            temperature=temperature,
            top_p=0.9,
            request_timeout=timeout
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"LLM 调用失败 [status={response.status_code}]: "
                f"{response.message}"
            )

        return response.output["choices"][0]["message"]["content"].strip()

    def _call_llm_openai(self, prompt: str, temperature: float = 0.7, timeout: int = 60) -> str:
        """OpenAI 兼容协议调用"""
        client = _get_openai_client(self.base_url, self.api_key)
        request_kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": 0.9,
            "timeout": timeout,
        }
        # Qwen3 默认可能输出较长的思考过程。维度标签只需要结构化短答案，
        # 关闭思考并设置较小的输出上限可显著降低批量标签生成的延迟。
        if self.model_name.lower().startswith("qwen3"):
            request_kwargs["max_tokens"] = 512
            request_kwargs["extra_body"] = {"enable_thinking": False}

        resp = client.chat.completions.create(**request_kwargs)
        if not resp.choices:
            raise RuntimeError(f"OpenAI 兼容 LLM 返回空 choices: {resp}")
        return (resp.choices[0].message.content or "").strip()

    def _call_llm_json(self, prompt: str, temperature: float = 0.3) -> Any:
        """
        调用 LLM 并尝试解析为 JSON 对象。

        Returns:
            解析后的 Python 对象（dict / list）。
        """
        text = self._call_llm(prompt, temperature=temperature)

        # 尝试从响应中提取 JSON 块
        # 优先匹配 ```json ... ``` 块
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            json_str = match.group(1)
        else:
            # 直接尝试解析整段文本
            json_str = text.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Qwen 偶尔会连续输出两个 JSON 对象或在 JSON 后附加说明。
            # 用 raw_decode 找到第一个完整对象，避免贪婪正则把两段拼在一起。
            decoder = json.JSONDecoder()
            for match2 in re.finditer(r"[\{\[]", json_str):
                try:
                    value, _ = decoder.raw_decode(json_str[match2.start():])
                    if isinstance(value, (dict, list)):
                        return value
                except json.JSONDecodeError:
                    continue
            raise ValueError(f"LLM 输出无法解析为 JSON:\n{text}")

    # ============================================================
    # Phase 2: 生成候选维度
    # ============================================================

    def generate_candidate_dimensions(self, docs: List[str]) -> List[str]:
        """
        从采样文档集合中归纳候选维度名称列表。

        Args:
            docs: 代表性文档文本列表。

        Returns:
            维度名称列表，例: ["适宜人群", "疾病类别", "治疗方案", ...]
        """
        if not docs:
            raise ValueError("文档列表为空")

        # 使用分布更均匀的样本，而不是只取列表开头的文档；长度和数量可由
        # 环境变量控制。维度发现阶段若只看 5*300 字，极易漏掉长文本后半段
        # 和少数景区中的有效信息轴。
        try:
            max_docs = max(5, int(os.getenv("DIM_CANDIDATE_DOCS", "20")))
        except ValueError:
            max_docs = 20
        try:
            max_chars = max(300, int(os.getenv("DIM_CANDIDATE_DOC_CHARS", "800")))
        except ValueError:
            max_chars = 800

        sample_count = min(max_docs, len(docs))
        if sample_count == 1:
            sample_indices = [0]
        else:
            sample_indices = [
                round(i * (len(docs) - 1) / (sample_count - 1))
                for i in range(sample_count)
            ]

        snippet_docs = []
        for index in sample_indices:
            doc = str(docs[index] or "")
            snippet_docs.append(_clip_text(doc, max_chars))

        docs_snippet = "\n\n---\n\n".join(snippet_docs)

        prompt = PROMPT_GENERATE_CANDIDATES.format(
            n=len(snippet_docs),
            docs_snippet=docs_snippet
        )

        raw = self._call_llm(prompt, temperature=0.7, timeout=120)

        # 解析常见的中英文逗号、顿号、分号和换行格式，并去除列表编号。
        dims = []
        for item in re.split(r"[，,、；;\n]+", raw):
            item = re.sub(r"^\s*(?:[-*•]|\d+[.)、])\s*", "", item).strip()
            item = item.strip("\"'[]()（）")
            if item and item not in dims:
                dims.append(item)

        try:
            max_candidates = max(1, int(os.getenv("MAX_DIM_CANDIDATES", "30")))
        except ValueError:
            max_candidates = 30
        return dims[:max_candidates]

    def extract_dimension_values(
        self,
        text: str,
        dim_name: str,
        max_chars: int = None,
    ) -> List[str]:
        """抽取当前文本中某一维度的全部明确值。

        该接口供维度发现阶段评估候选维度使用。与旧的单值接口相比，
        它不会因一个 chunk 中存在多个值而低估覆盖率和辨识度。
        """
        if not text or not dim_name:
            return []

        if max_chars is None:
            try:
                max_chars = max(0, int(os.getenv("DIM_VALIDATION_CHARS", "4000")))
            except ValueError:
                max_chars = 4000
        text = str(text)
        text = _clip_text(text, max_chars)

        prompt = PROMPT_EXTRACT_VALUES.format(dim_name=dim_name, text=text)
        try:
            result = self._call_llm_json(prompt, temperature=0.1)
        except Exception:
            return []

        if isinstance(result, dict):
            values = result.get("values", [])
        elif isinstance(result, list):
            values = result
        else:
            values = []

        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return []

        cleaned = []
        for value in values:
            value = str(value or "").strip()
            if not value or value.upper() in {"NULL", "NONE", "无", "未提及"}:
                continue
            if value not in cleaned:
                cleaned.append(value)
        return cleaned

    # ============================================================
    # Phase 3: 单维度单值抽取（迭代验证用）
    # ============================================================

    def extract_dimension_value(self, text: str, dim_name: str) -> Optional[str]:
        """
        从单篇文档中抽取指定维度的单个值。

        Args:
            text: 文档全文。
            dim_name: 维度名称。

        Returns:
            抽取到的值字符串，或 None（表示未命中）。
        """
        values = self.extract_dimension_values(text, dim_name)
        return values[0] if values else None

    # ============================================================
    # Phase 3: 维度优化决策
    # ============================================================

    def optimize_dimension(
        self,
        dim_name: str,
        issue_type: str,
        metric_data: str,
        sample_values: List[str]
    ) -> Dict[str, Any]:
        """
        根据诊断结果，让 LLM 对维度做出优化决策。

        Args:
            dim_name: 维度名称。
            issue_type: 问题类型 ("低覆盖率" | "低辨识度" | "语义/数据冗余")。
            metric_data: 诊断指标数据。
            sample_values: 抽取样本列表。

        Returns:
            {"action": str, "reasoning": str, "new_dimensions": List[str]}
        """
        samples_str = "\n".join(f"  - {v}" for v in sample_values[:10])

        prompt = PROMPT_OPTIMIZE_DIMENSION.format(
            dim_name=dim_name,
            issue_type=issue_type,
            metric_data=metric_data,
            samples=samples_str
        )

        result = self._call_llm_json(prompt, temperature=0.3)
        return result

    def merge_with_targets(
        self,
        dim_name: str,
        issue_type: str,
        metric_data: str,
        sample_values: List[str],
        candidate_targets: List[str]
    ) -> Dict[str, Any]:
        """
        让 LLM 在"指定的候选目标列表"中判断能否将 dim_name 与其中一个合并。

        用于"分阶段融合"流程：
          - Stage 1: 传入"合格维度"作为候选
          - Stage 2: 传入"其他不合格维度"作为候选
          - 任何阶段返回 NOT_MERGE 都意味着本阶段融合失败, 可推进到下一阶段

        Args:
            dim_name: 待融合的维度名称。
            issue_type: 问题类型 ("低覆盖率" | "低辨识度")。
            metric_data: 诊断指标数据 (含具体阈值/数值)。
            sample_values: 抽取样本列表。
            candidate_targets: 允许的合并目标候选列表 (必须从中精确选择)。

        Returns:
            {
              "action": "MERGE" | "NOT_MERGE",
              "reasoning": str,
              "merge_target": str,            # 仅 MERGE 时, 必须完全等于 candidate_targets 中某一项
              "new_dimensions": List[str]     # 仅 MERGE 时, 合并后的新维度名称 (1个)
            }
        """
        # 防御: 候选为空直接返回 NOT_MERGE, 不调 LLM
        if not candidate_targets:
            return {
                "action": "NOT_MERGE",
                "reasoning": "候选目标列表为空, 无可融合对象。",
                "merge_target": "",
                "new_dimensions": []
            }

        # 构造候选块 (带编号, 便于 LLM 引用)
        cand_lines = [f"  {i+1}. {name}" for i, name in enumerate(candidate_targets)]
        candidates_block = "\n".join(cand_lines)

        samples_str = "\n".join(f"  - {v}" for v in sample_values[:5])

        prompt = PROMPT_MERGE_WITH_TARGETS.format(
            dim_name=dim_name,
            issue_type=issue_type,
            metric_data=metric_data,
            samples=samples_str,
            candidates_block=candidates_block
        )

        try:
            result = self._call_llm_json(prompt, temperature=0.3)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"    [Warning] merge_with_targets JSON 解析失败: {e}")
            return {
                "action": "NOT_MERGE",
                "reasoning": f"LLM 输出解析失败: {e}",
                "merge_target": "",
                "new_dimensions": []
            }

        # 字段规整 + 安全校验
        if not isinstance(result, dict):
            return {
                "action": "NOT_MERGE",
                "reasoning": "LLM 返回非 dict 结构。",
                "merge_target": "",
                "new_dimensions": []
            }

        action = str(result.get("action", "")).upper()
        result["action"] = action

        if action == "MERGE":
            target = str(result.get("merge_target", "")).strip()
            # 安全校验: merge_target 必须在候选列表中, 否则强制 NOT_MERGE
            if target not in candidate_targets:
                print(f"    [Warning] LLM 返回的 merge_target='{target}' 不在候选列表中, 强制 NOT_MERGE")
                result["action"] = "NOT_MERGE"
                result["merge_target"] = ""
                result["new_dimensions"] = []
            else:
                result["merge_target"] = target
                # 确保 new_dimensions 至少有一个
                new_dims = result.get("new_dimensions") or []
                if not isinstance(new_dims, list):
                    new_dims = [str(new_dims)] if new_dims else []
                if not new_dims:
                    new_dims = [target]   # 默认沿用目标名
                result["new_dimensions"] = [str(d).strip() for d in new_dims if str(d).strip()]
        else:
            # 任何非 MERGE 都归一为 NOT_MERGE
            result["action"] = "NOT_MERGE"
            result["merge_target"] = ""
            result["new_dimensions"] = []

        return result

    # ============================================================
    # TagGenerate: 批量多维度抽取
    # ============================================================

    def extract_batch_dimensions(
        self,
        text: str,
        dims: List[str]
    ) -> Dict[str, List[str]]:
        """
        从单篇文档中一次性抽取多个维度的值。

        Args:
            text: 文档全文。
            dims: 维度名称列表。

        Returns:
            {dim_name: [val1, val2, ...]}，未命中的维度不出现在结果中。
        """
        if not text or not dims:
            return {}

        # 限制文本和维度数量
        truncated = text[:1500] if len(text) > 1500 else text
        dims_subset = dims[:15]  # 最多 15 个维度一次性抽取

        prompt = PROMPT_EXTRACT_BATCH.format(
            dims_list="、".join(dims_subset),
            text=truncated
        )

        try:
            result = self._call_llm_json(prompt, temperature=0.1)

            # 确保返回值是 dict
            if isinstance(result, list):
                return {}
            if not isinstance(result, dict):
                return {}

            # 清理：过滤空值和无效条目
            cleaned = {}
            for k, v in result.items():
                if not k or not v:
                    continue
                if isinstance(v, list) and len(v) > 0:
                    cleaned[k] = [str(x).strip() for x in v if x]
                elif isinstance(v, str) and v.strip():
                    cleaned[k] = [v.strip()]

            return cleaned

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[Warning] 批量抽取 JSON 解析失败，回退到单维度抽取: {e}")
            return self._extract_batch_fallback(text, dims_subset)

    def extract_multi_chunk_dimensions(
        self,
        records: List[Dict[str, str]],
        dims: List[str],
        max_text_chars: int = 1000,
    ) -> Dict[str, Dict[str, List[str]]]:
        """一次请求处理多条 chunk，返回 ``doc_id -> 维度标签``。

        标签抽取是批量实验的主要耗时环节。将多条 chunk 放入同一个
        JSON 请求可以减少请求次数，同时仍保留每条 chunk 的 ID 关联。
        """
        if not records or not dims:
            return {}

        records_block = []
        for record in records:
            doc_id = str(record.get("doc_id", ""))
            text = _clip_text(str(record.get("doc_text", "")), max_text_chars)
            records_block.append(f"ID: {doc_id}\n文本：\n---\n{text}\n---")

        prompt = PROMPT_EXTRACT_MULTI_BATCH.format(
            dims_list="、".join(dims[:15]),
            records_block="\n\n".join(records_block),
        )

        try:
            result = self._call_llm_json(prompt, temperature=0.1)
        except (json.JSONDecodeError, ValueError, RuntimeError) as e:
            print(f"[Warning] 多 chunk 批量抽取失败: {e}")
            return {}

        if not isinstance(result, dict):
            return {}

        valid_ids = {str(record.get("doc_id", "")) for record in records}
        cleaned: Dict[str, Dict[str, List[str]]] = {}
        for raw_doc_id, tag_map in result.items():
            raw_doc_id = str(raw_doc_id)
            doc_id = raw_doc_id
            if doc_id not in valid_ids:
                # 模型有时会把示例中的“记录ID-”前缀带入 key；按最长
                # 后缀匹配恢复原始 ID，避免 chunk 标签无法回填。
                matches = [candidate for candidate in valid_ids if raw_doc_id.endswith(candidate)]
                if matches:
                    doc_id = max(matches, key=len)
            if doc_id not in valid_ids or not isinstance(tag_map, dict):
                continue
            item: Dict[str, List[str]] = {}
            for dim, values in tag_map.items():
                if not dim or not values:
                    continue
                if isinstance(values, list):
                    values = [str(value).strip() for value in values if str(value).strip()]
                elif isinstance(values, str) and values.strip():
                    values = [values.strip()]
                else:
                    values = []
                if values:
                    item[str(dim).strip()] = values
            cleaned[doc_id] = item
        return cleaned

    def validate_dimension_schema_batch(
        self,
        records: List[Dict[str, str]],
        dims: List[str],
        max_text_chars: int = 4000,
    ) -> Dict[str, Dict[str, List[str]]]:
        """批量验证候选维度 schema，不执行标签写入。

        返回 ``chunk_id -> {dimension_name: [explicit values]}``，仅供维度
        覆盖率、取值多样性和冗余诊断使用。该接口与正式标签生成接口分开，
        调用方不会把结果写入 Qdrant/MySQL。
        """
        if not records or not dims:
            return {}

        records_block = []
        for record in records:
            doc_id = str(record.get("doc_id", ""))
            text = _clip_text(str(record.get("doc_text", "")), max_text_chars)
            records_block.append(f"ID: {doc_id}\n文本：\n---\n{text}\n---")

        prompt = PROMPT_VALIDATE_SCHEMA_BATCH.format(
            dims_list="、".join(dims),
            records_block="\n\n".join(records_block),
        )

        try:
            result = self._call_llm_json(prompt, temperature=0.1)
        except Exception as exc:
            print(f"[Warning] schema 批量验证失败: {exc}")
            return {}

        valid_ids = {str(record.get("doc_id", "")) for record in records}
        allowed_dims = {str(dim).strip() for dim in dims if str(dim).strip()}
        raw_records = result.get("records", []) if isinstance(result, dict) else []
        if not isinstance(raw_records, list):
            raw_records = []

        cleaned: Dict[str, Dict[str, List[str]]] = {}
        for item in raw_records:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("id", item.get("doc_id", ""))).strip()
            dimension_map = item.get("dimensions", {})
            if doc_id not in valid_ids or not isinstance(dimension_map, dict):
                continue

            cleaned_dims: Dict[str, List[str]] = {}
            for dim, values in dimension_map.items():
                dim = str(dim).strip()
                if dim not in allowed_dims:
                    continue
                if isinstance(values, str):
                    values = [values]
                if not isinstance(values, list):
                    continue
                values = list(dict.fromkeys(
                    str(value).strip() for value in values
                    if str(value).strip() and str(value).upper() not in {"NULL", "NONE", "无", "未提及"}
                ))
                if values:
                    cleaned_dims[dim] = values
            if cleaned_dims:
                cleaned[doc_id] = cleaned_dims
        return cleaned

    def _extract_batch_fallback(
        self,
        text: str,
        dims: List[str]
    ) -> Dict[str, List[str]]:
        """
        批量抽取失败时的兜底策略：逐维度调用。
        """
        results = {}
        for dim in dims:
            val = self.extract_dimension_value(text, dim)
            if val is not None:
                results[dim] = [val]
        return results

    # ============================================================
    # TagGenerate: 兜底关键词抽取
    # ============================================================

    def extract_keywords_fallback(self, text: str) -> List[str]:
        """
        当标准维度抽取全部失败时，用关键词兜底。

        Args:
            text: 文档全文。

        Returns:
            关键词列表。
        """
        if not text:
            return []

        truncated = text[:500] if len(text) > 500 else text

        prompt = PROMPT_KEYWORDS_FALLBACK.format(text=truncated)

        try:
            raw = self._call_llm(prompt, temperature=0.5)
            keywords = [k.strip() for k in re.split(r"[，,、]", raw) if k.strip()]
            return keywords[:5]
        except Exception:
            return []

    # ============================================================
    # QueryParser: 查询意图解析
    # ============================================================

    def parse_query_intent(
        self,
        query_text: str,
        dims: List[str],
        enum_values_map: Dict[str, List[str]],
        schema_dim_fields: List[str] = None
    ) -> Dict[str, List[str]]:
        """
        将自然语言查询解析为结构化的维度约束。

        Args:
            query_text: 用户查询文本。
            dims: 所有可用维度名称列表。
            enum_values_map: 枚举维度的候选值映射，例:
                {"适宜人群": ["儿童", "成人", "老人"], "疾病类别": [...]}

        Returns:
            {dim: [val1, val2]} 约束结构。
        """
        if not query_text:
            return {}

        # 构建枚举信息段落
        if enum_values_map:
            enum_lines = []
            for dim, vals in enum_values_map.items():
                enum_lines.append(f"- {dim}（可选值：{', '.join(vals[:20])})")
            enum_info = "枚举维度候选值：\n" + "\n".join(enum_lines)
        else:
            enum_info = "（无可用的枚举维度候选值）"

        # 构造 schema 字段名约束
        if schema_dim_fields:
            schema_constraint = PROMPT_SCHEMA_CONSTRAINT.format(
                schema_fields="、".join(schema_dim_fields)
            )
        else:
            schema_constraint = ""

        prompt = PROMPT_PARSE_QUERY.format(
            dims_list="、".join(dims),
            enum_info=enum_info,
            schema_constraint=schema_constraint,
            query_text=query_text
        )

        try:
            result = self._call_llm_json(prompt, temperature=0.3)

            if not isinstance(result, dict):
                return {}

            # 清理
            cleaned = {}
            for k, v in result.items():
                if not k or not v:
                    continue
                if isinstance(v, list):
                    cleaned[k] = [str(x).strip() for x in v if x]
                elif isinstance(v, str) and v.strip():
                    cleaned[k] = [v.strip()]

            return cleaned

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[Warning] 查询解析 JSON 解析失败: {e}")
            return {}


# ============================================================
# 便捷入口（直接 python llm_service.py 可测试连通性）
# ============================================================

if __name__ == "__main__":
    import pprint

    # 从环境变量读取 API Key（也可直接传入）
    miner = DimensionMiningWithQwen(
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        model_name="qwen-plus"
    )

    # 测试 1: 生成候选维度
    print("=== 测试 1: generate_candidate_dimensions ===")
    test_docs = [
        "儿童感冒发热，可使用布洛芬混悬液，每次5ml，每日3次。适用于3-12岁儿童。",
        "老年人腰腿痛，可服用氨基葡萄糖胶囊，每日2次，每次1粒。",
        "孕妇感冒应避免使用布洛芬，建议使用对乙酰氨基酚，并遵医嘱。"
    ]
    dims = miner.generate_candidate_dimensions(test_docs)
    print(f"候选维度: {dims}")

    # 测试 2: 批量抽取
    print("\n=== 测试 2: extract_batch_dimensions ===")
    test_text = "本品适用于3岁以上儿童及成人，用于缓解感冒引起的发热、头痛，鼻塞等症状。儿童用量请遵医嘱。"
    extracted = miner.extract_batch_dimensions(test_text, ["适宜人群", "功效作用", "疾病类别"])
    pprint.pprint(extracted)

    # 测试 3: 查询意图解析
    print("\n=== 测试 3: parse_query_intent ===")
    parsed = miner.parse_query_intent(
        "儿童发烧咳嗽应该怎么办",
        dims=["适宜人群", "疾病类别", "治疗方案", "功效作用"],
        enum_values_map={
            "适宜人群": ["儿童", "成人", "老人", "婴幼儿", "孕妇"],
            "疾病类别": ["呼吸道疾病", "发热相关", "消化系统疾病"]
        }
    )
    pprint.pprint(parsed)
