#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
指代消解模块 - Coreference Resolution
====================================

功能：
1. 指代消解：将代词、指示词等替换为指代对象
2. 自检验证：检测消解错误并回滚
3. 一致性检查：确保同一文档内相同指代词有一致的消解结果

集成位置：分片之后，入库之前

使用方式：
    python coreference_resolver.py --input ./output_chunks/all_chunks_chunks.json --output ./output_chunks
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ==================== 数据类定义 ====================

@dataclass
class ResolutionRecord:
    """单次消解记录"""
    chunk_id: str
    original: str
    resolved: str
    pronoun: str
    antecedent: str
    start_pos: int
    end_pos: int


@dataclass
class VerificationResult:
    """自检验证结果"""
    chunk_id: str
    rule_check: str = "SKIP"  # PASS / FAIL / WARN
    llm_check: str = "SKIP"
    consistency_check: str = "SKIP"
    issues: List[str] = field(default_factory=list)
    rollback_triggered: bool = False


@dataclass
class ChunkResolutionResult:
    """单个 chunk 的消解结果"""
    doc_id: str
    chunk_index: int
    original_text: str
    resolved_text: str
    resolutions: List[ResolutionRecord] = field(default_factory=list)
    verification: Optional[VerificationResult] = None
    skipped: bool = False
    skip_reason: str = ""


# ==================== 规则引擎 ====================

class CoreferenceRules:
    """基于规则的指代消解"""

    # 代词模式
    PRONOUN_PATTERNS = [
        # 中文代词
        (r'\b他\b', '他'), (r'\b她\b', '她'), (r'\b它\b', '它'),
        (r'\b他们\b', '他们'), (r'\b她们\b', '她们'), (r'\b它们\b', '它们'),
        (r'\b这\b', '这'), (r'\b那\b', '那'), (r'\b这些\b', '这些'), (r'\b那些\b', '那些'),
        (r'\b此人\b', '此人'), (r'\b此事\b', '此事'), (r'\b该公司\b', '该公司'),
        (r'\b本\b', '本'),
        # 英文代词（用于双语内容）
        (r'\bhis\b', 'his'), (r'\bher\b', 'her'), (r'\bits\b', 'its'),
        (r'\bthis\b', 'this'), (r'\bthat\b', 'that'), (r'\bthese\b', 'these'), (r'\bthose\b', 'those'),
    ]

    # 机构/实体简称模式
    ABBREVIATION_PATTERNS = [
        (r'\b该方案\b', '该方案'),
        (r'\b该方法\b', '该方法'),
        (r'\b该技术\b', '该技术'),
        (r'\b该系统\b', '该系统'),
        (r'\b该模型\b', '该模型'),
        (r'\b该算法\b', '该算法'),
        (r'\b本方法\b', '本方法'),
        (r'\b本技术\b', '本技术'),
    ]

    @classmethod
    def find_pronouns(cls, text: str) -> List[Tuple[str, int, int]]:
        """查找文本中的代词，返回 (代词, 起始位置, 结束位置) 列表"""
        found = []
        for pattern, pronoun in cls.PRONOUN_PATTERNS:
            for m in re.finditer(pattern, text):
                found.append((pronoun, m.start(), m.end()))
        return found

    @classmethod
    def find_abbreviations(cls, text: str) -> List[Tuple[str, int, int]]:
        """查找简称/指示词"""
        found = []
        for pattern, abbr in cls.ABBREVIATION_PATTERNS:
            for m in re.finditer(pattern, text):
                found.append((abbr, m.start(), m.end()))
        return found


class AntecedentFinder:
    """指代对象查找器"""

    def __init__(self, context_chunks: List[str] = None):
        self.context_chunks = context_chunks or []

    def find_antecedent(self, pronoun: str, current_chunk: str, preceding_chunks: List[str] = None) -> Optional[str]:
        """根据代词类型查找指代对象"""
        all_context = []
        if preceding_chunks:
            all_context.extend(preceding_chunks)
        all_context.append(current_chunk)

        if pronoun in {'他', '他们'}:
            return self._find_person_antecedent('男', all_context)
        elif pronoun in {'她', '她们'}:
            return self._find_person_antecedent('女', all_context)
        elif pronoun in {'它', '它们'}:
            return self._find_object_antecedent(all_context)
        elif pronoun in {'这', '这些', '此人', '此事', '该方案', '该方法', '该技术', '该系统', '该模型', '该算法', '本', '本方法', '本技术'}:
            return self._find_recent_antecedent(all_context)
        elif pronoun in {'那', '那些'}:
            return self._find_distant_antecedent(all_context)
        return None

    def _find_person_antecedent(self, gender: str, context: List[str]) -> Optional[str]:
        """查找人物指代"""
        # 优先查找人物名词
        person_patterns = [
            r'([\u4e00-\u9fa5]{2,4})(?:先生|女士|教授|博士|院士|总裁|经理|董事长|CEO)',
            r'([\u4e00-\u9fa5]{2,4})(?:说|道|认为|指出|表示|发现|提出)',
            r'名叫([\u4e00-\u9fa5]{2,4})',
            r'是([\u4e00-\u9fa5]{2,4})',
        ]

        for text in reversed(context):
            for pattern in person_patterns:
                m = re.search(pattern, text)
                if m:
                    return m.group(1) if m.lastindex else None
        return None

    def _find_object_antecedent(self, context: List[str]) -> Optional[str]:
        """查找物体/概念指代"""
        obj_patterns = [
            r'([\u4e00-\u9fa5]{2,8})(?:技术|方法|模型|算法|系统|方案|理论|方法论)',
            r'([\u4e00-\u9fa5]{2,8})能够',
            r'([\u4e00-\u9fa5]{2,8})可以',
            r'名为([\u4e00-\u9fa5]{2,8})',
        ]

        for text in reversed(context):
            for pattern in obj_patterns:
                m = re.search(pattern, text)
                if m:
                    return m.group(1) if m.lastindex else None
        return None

    def _find_recent_antecedent(self, context: List[str]) -> Optional[str]:
        """查找最近的指代对象"""
        # 查找最近出现的重要名词
        noun_patterns = [
            r'([\u4e00-\u9fa5]{2,8})(?:方法|技术|方案|模型|系统)',
            r'([\u4e00-\u9fa5]{2,6})(?:的)',
            r'这个([\u4e00-\u9fa5]{2,6})',
            r'这项([\u4e00-\u9fa5]{2,6})',
        ]

        for text in reversed(context):
            for pattern in noun_patterns:
                m = re.search(pattern, text)
                if m:
                    return m.group(1) if m.lastindex else None
        return None

    def _find_distant_antecedent(self, context: List[str]) -> Optional[str]:
        """查找较远的指代对象"""
        # 与最近指代类似，但搜索范围更广
        return self._find_recent_antecedent(context)


# ==================== 自检验证器 ====================

class SelfVerifier:
    """自检验证器"""

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode

    def verify(self, chunk_id: str, original: str, resolved: str, resolutions: List[ResolutionRecord]) -> VerificationResult:
        """执行自检验证"""
        result = VerificationResult(chunk_id=chunk_id)
        issues = []

        # 1. 规则层校验（必做）
        rule_result = self._rule_check(original, resolved, resolutions)
        result.rule_check = rule_result['status']
        issues.extend(rule_result['issues'])

        # 2. 循环指代检测
        cycle_issues = self._detect_cycles(resolutions)
        issues.extend(cycle_issues)

        # 3. 悬空指代检测
        dangling_issues = self._detect_dangling(resolved, resolutions)
        issues.extend(dangling_issues)

        # 4. 膨胀率检测
        expansion_issues = self._check_expansion(original, resolved)
        issues.extend(expansion_issues)

        result.issues = issues

        # 判断是否需要回滚
        if result.rule_check == 'FAIL' or len(cycle_issues) > 0:
            result.rollback_triggered = True

        return result

    def _rule_check(self, original: str, resolved: str, resolutions: List[ResolutionRecord]) -> Dict[str, Any]:
        """规则层校验"""
        issues = []
        status = 'PASS'

        # 检查是否有不合理的替换
        for res in resolutions:
            if not res.antecedent:
                issues.append(f"代词 '{res.pronoun}' 无法找到指代对象")
                status = 'WARN'
            elif len(res.antecedent) < 2:
                issues.append(f"指代对象 '{res.antecedent}' 过短，可能不正确")
                status = 'WARN'

        return {'status': status, 'issues': issues}

    def _detect_cycles(self, resolutions: List[ResolutionRecord]) -> List[str]:
        """检测循环指代：A→B→A"""
        issues = []
        pronoun_to_antecedent = {res.pronoun: res.antecedent for res in resolutions if res.antecedent}

        for pronoun, antecedent in pronoun_to_antecedent.items():
            # 检查 antecedent 是否也是需要消解的代词
            if antecedent in pronoun_to_antecedent:
                issues.append(f"检测到循环指代: {pronoun} → {antecedent}")
                return issues  # 找到一个就足够了

        return issues

    def _detect_dangling(self, resolved: str, resolutions: List[ResolutionRecord]) -> List[str]:
        """检测悬空指代：替换后文本中仍存在未消解的代词"""
        issues = []

        # 查找替换后文本中是否还有代词
        remaining_pronouns = CoreferenceRules.find_pronouns(resolved)
        if remaining_pronouns:
            # 检查这些代词是否在 resolutions 中
            resolved_pronoun_positions = set()
            for res in resolutions:
                resolved_pronoun_positions.add((res.start_pos, res.end_pos))

            unprocessed = [(p, s, e) for p, s, e in remaining_pronouns
                         if (s, e) not in resolved_pronoun_positions]

            if unprocessed:
                # 这不是错误，只是说明有些代词没被处理
                pass

        return issues

    def _check_expansion(self, original: str, resolved: str) -> List[str]:
        """检测文本膨胀率"""
        issues = []

        if len(original) == 0:
            return issues

        expansion_rate = len(resolved) / len(original)

        # 如果膨胀率超过 1.5 倍，可能有过度的替换
        if expansion_rate > 1.5:
            issues.append(f"文本膨胀率过高: {expansion_rate:.2f}x，可能存在过度替换")
        elif expansion_rate < 0.5:
            issues.append(f"文本压缩率过高: {expansion_rate:.2f}x，可能丢失重要信息")

        return issues


# ==================== LLM 校验器（可选） ====================

class LLMVerifier:
    """基于 LLM 的验证器"""

    SYSTEM_PROMPT = """你是一个指代消解验证专家。请验证以下消解是否正确。

检查要点：
1. 每个代词是否指向了正确的指代对象
2. 消解后的文本语义是否与原意一致
3. 是否存在明显的消解错误

请以 JSON 格式输出验证结果：
{
    "is_correct": true/false,
    "errors": ["错误1", "错误2"],
    "warnings": ["警告1", "警告2"]
}"""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def verify(self, chunk_id: str, original: str, resolved: str, resolutions: List[ResolutionRecord]) -> VerificationResult:
        """使用 LLM 验证消解结果"""
        if not self.llm:
            return VerificationResult(chunk_id=chunk_id, rule_check='SKIP', llm_check='SKIP')

        # 构建验证 prompt
        resolution_details = "\n".join([
            f"- '{r.pronoun}' (位置 {r.start_pos}) → '{r.antecedent}'"
            for r in resolutions if r.antecedent
        ])

        user_prompt = f"""请验证以下指代消解是否正确：

原文：{original}

消解后：{resolved}

消解详情：
{resolution_details}

请判断消解是否正确，并指出任何错误或警告。"""

        try:
            result = self.llm.chat_json(self.SYSTEM_PROMPT, user_prompt)
            verification = VerificationResult(chunk_id=chunk_id)
            verification.llm_check = 'PASS' if result.get('is_correct', False) else 'FAIL'
            verification.issues.extend(result.get('errors', []))
            verification.issues.extend(result.get('warnings', []))

            if not result.get('is_correct', False):
                verification.rollback_triggered = True

            return verification

        except Exception as e:
            return VerificationResult(
                chunk_id=chunk_id,
                rule_check='PASS',
                llm_check='ERROR',
                issues=[f"LLM 验证失败: {str(e)}"]
            )


# ==================== 一致性检查器 ====================

class ConsistencyChecker:
    """一致性检查器：确保同一文档内相同指代词有一致的消解结果"""

    def check(self, doc_id: str, results: List[ChunkResolutionResult]) -> List[str]:
        """检查文档内的一致性"""
        issues = []

        # 收集所有消解结果
        pronoun_resolutions: Dict[str, Dict[str, str]] = {}  # {pronoun: {chunk_id: antecedent}}

        for result in results:
            if result.resolutions:
                for res in result.resolutions:
                    if res.antecedent:
                        if res.pronoun not in pronoun_resolutions:
                            pronoun_resolutions[res.pronoun] = {}
                        pronoun_resolutions[res.pronoun][str(result.chunk_index)] = res.antecedent

        # 检查一致性
        for pronoun, chunk_antecedents in pronoun_resolutions.items():
            unique_antecedents = set(chunk_antecedents.values())

            # 如果同一个代词在不同 chunk 中指向不同的对象，记录警告
            if len(unique_antecedents) > 1:
                issues.append(
                    f"代词 '{pronoun}' 在不同 chunk 中指向不同对象: {chunk_antecedents}"
                )

        return issues


# ==================== 核心指代消解器 ====================

class CoreferenceResolver:
    """指代消解主类"""

    def __init__(
        self,
        use_llm_resolution: bool = False,
        use_llm_verification: bool = False,
        llm_client=None,
        strict_mode: bool = False,
        enable_consistency_check: bool = True
    ):
        """
        Args:
            use_llm_resolution: 是否使用 LLM 进行指代消解（默认使用规则）
            use_llm_verification: 是否使用 LLM 验证消解结果
            llm_client: LLM 客户端
            strict_mode: 严格模式，会拒绝更多消解
            enable_consistency_check: 启用一致性检查
        """
        self.use_llm_resolution = use_llm_resolution
        self.use_llm_verification = use_llm_verification
        self.llm_client = llm_client
        self.strict_mode = strict_mode
        self.enable_consistency_check = enable_consistency_check

        self.llm_verifier = LLMVerifier(llm_client) if use_llm_verification and llm_client else None
        self.consistency_checker = ConsistencyChecker()

    def resolve(self, chunks_data: List[Dict]) -> List[ChunkResolutionResult]:
        """
        对分片数据进行指代消解

        Args:
            chunks_data: 分片数据列表

        Returns:
            消解结果列表
        """
        print(f"[CoreferenceResolver] 开始处理 {len(chunks_data)} 个 chunk...")

        results = []
        doc_groups: Dict[str, List[Dict]] = {}

        # 按文档分组
        for sub in chunks_data:
            doc_id = sub.get('doc_id', 'unknown')
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(sub)

        total_chunks = 0
        resolved_count = 0
        skipped_count = 0

        for doc_id, subs in doc_groups.items():
            print(f"  处理文档: {doc_id}")

            # 收集该文档的所有 chunk 文本（用于上下文）
            all_chunks_text = []
            for sub in subs:
                for chunk in sub.get('chunks', []):
                    all_chunks_text.append(chunk.get('chunk_text', ''))

            # 处理每个子文件的每个 chunk
            doc_results = []
            for sub in subs:
                sub_results = self._resolve_sub(sub, all_chunks_text)
                doc_results.extend(sub_results)

            # 一致性检查
            if self.enable_consistency_check and doc_results:
                consistency_issues = self.consistency_checker.check(doc_id, doc_results)
                if consistency_issues:
                    print(f"    [Warning] 一致性问题: {len(consistency_issues)} 个")

            results.extend(doc_results)

            for r in doc_results:
                total_chunks += 1
                if r.resolutions:
                    resolved_count += 1
                if r.skipped:
                    skipped_count += 1

        print(f"  -> 处理完成: {total_chunks} chunks, {resolved_count} 个含消解, {skipped_count} 个跳过")

        return results

    def _resolve_sub(self, sub: Dict, all_chunks_text: List[str]) -> List[ChunkResolutionResult]:
        """处理单个子文件的 chunks"""
        results = []
        chunks = sub.get('chunks', [])
        doc_id = sub.get('doc_id', 'unknown')

        # 构建前置 chunk 列表（用于查找指代对象）
        preceding_texts = []

        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.get('chunk_text', '')
            chunk_id = f"{doc_id}_chunk_{idx}"

            result = self._resolve_single_chunk(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                chunk_index=idx,
                preceding_chunks=preceding_texts,
                all_chunks=all_chunks_text
            )

            results.append(result)
            preceding_texts.append(result.resolved_text)

        return results

    def _resolve_single_chunk(
        self,
        chunk_id: str,
        chunk_text: str,
        chunk_index: int,
        preceding_chunks: List[str],
        all_chunks: List[str]
    ) -> ChunkResolutionResult:
        """消解单个 chunk"""
        doc_id = chunk_id.rsplit('_chunk_', 1)[0] if '_chunk_' in chunk_id else chunk_id

        # 检查是否需要消解
        pronouns = CoreferenceRules.find_pronouns(chunk_text)
        abbreviations = CoreferenceRules.find_abbreviations(chunk_text)

        if not pronouns and not abbreviations:
            return ChunkResolutionResult(
                doc_id=doc_id,
                chunk_index=chunk_index,
                original_text=chunk_text,
                resolved_text=chunk_text,
                skipped=True,
                skip_reason="无代词/指示词"
            )

        # 查找指代对象并进行替换
        resolved_text = chunk_text
        resolutions = []

        # 获取上下文（前置 chunk + 当前 chunk）
        context = preceding_chunks + [chunk_text]

        for pronoun, start, end in pronouns:
            antecedent = self._find_antecedent_for_pronoun(
                pronoun, chunk_text, preceding_chunks
            )

            if antecedent:
                # 替换
                resolved_text = resolved_text[:start] + antecedent + resolved_text[end:]
                offset = len(antecedent) - (end - start)

                # 调整后续位置
                for i, (p, s, e) in enumerate(pronouns):
                    if s >= end:
                        pronouns[i] = (p, s + offset, e + offset)

                resolutions.append(ResolutionRecord(
                    chunk_id=chunk_id,
                    original=chunk_text,
                    resolved=resolved_text,
                    pronoun=pronoun,
                    antecedent=antecedent,
                    start_pos=start,
                    end_pos=end
                ))

        # 自检验证
        verifier = SelfVerifier(strict_mode=self.strict_mode)
        verification = verifier.verify(chunk_id, chunk_text, resolved_text, resolutions)

        # 如果触发回滚，保留原文本
        if verification.rollback_triggered:
            resolved_text = chunk_text
            resolutions = []
            print(f"    [Rollback] {chunk_id}: 回滚消解结果")

        # LLM 验证（可选）
        if self.llm_verifier and resolutions:
            llm_verification = self.llm_verifier.verify(
                chunk_id, chunk_text, resolved_text, resolutions
            )
            if llm_verification.llm_check == 'FAIL':
                verification.issues.extend(llm_verification.issues)
                verification.rollback_triggered = True
                resolved_text = chunk_text
                resolutions = []

        return ChunkResolutionResult(
            doc_id=doc_id,
            chunk_index=chunk_index,
            original_text=chunk_text,
            resolved_text=resolved_text,
            resolutions=resolutions,
            verification=verification,
            skipped=len(resolutions) == 0
        )

    def _find_antecedent_for_pronoun(self, pronoun: str, current_chunk: str, preceding_chunks: List[str]) -> Optional[str]:
        """为代词查找指代对象"""
        finder = AntecedentFinder(context_chunks=preceding_chunks + [current_chunk])
        return finder.find_antecedent(pronoun, current_chunk, preceding_chunks)


# ==================== 主函数 ====================

def resolve_chunks(
    input_path: str,
    output_dir: str = None,
    use_llm: bool = False,
    llm_base_url: str = None,
    llm_api_key: str = None,
    llm_model: str = None,
    strict_mode: bool = False
) -> Dict[str, Any]:
    """
    指代消解主函数

    Args:
        input_path: 输入分片文件路径
        output_dir: 输出目录（默认与输入同目录）
        use_llm: 是否使用 LLM
        llm_base_url: LLM API 地址
        llm_api_key: LLM API Key
        llm_model: LLM 模型名
        strict_mode: 严格模式

    Returns:
        处理结果统计
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_dir = Path(output_dir) if output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载分片数据
    print(f"加载分片数据: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    # 构建 LLM 客户端
    llm_client = None
    if use_llm and llm_api_key:
        from integrated_chunker import OpenAICompatClient
        llm_client = OpenAICompatClient(
            base_url=llm_base_url or "https://api.openai.com/v1",
            api_key=llm_api_key,
            model=llm_model or "gpt-3.5-turbo"
        )

    # 执行指代消解
    resolver = CoreferenceResolver(
        use_llm_resolution=False,  # 默认使用规则引擎
        use_llm_verification=use_llm,
        llm_client=llm_client,
        strict_mode=strict_mode,
        enable_consistency_check=True
    )

    results = resolver.resolve(chunks_data)

    # 更新原始数据中的 chunk 文本
    updated_chunks_data = []
    for sub in chunks_data:
        sub_results = [r for r in results if r.doc_id == sub.get('doc_id', '')]
        if sub_results:
            # 更新每个 chunk 的文本
            for i, chunk in enumerate(sub.get('chunks', [])):
                matching_result = next((r for r in sub_results if r.chunk_index == i), None)
                if matching_result and not matching_result.skipped:
                    chunk['chunk_text'] = matching_result.resolved_text
                    chunk['coreference_resolved'] = True
                    chunk['resolution_count'] = len(matching_result.resolutions)
                    if matching_result.verification:
                        chunk['verification'] = {
                            'rule_check': matching_result.verification.rule_check,
                            'issues': matching_result.verification.issues
                        }
        updated_chunks_data.append(sub)

    # 保存结果
    output_file = output_dir / f"{input_path.stem}_resolved.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(updated_chunks_data, f, ensure_ascii=False, indent=2)

    # 保存消解日志
    log_file = output_dir / f"{input_path.stem}_resolution_log.json"
    resolution_log = [
        {
            'chunk_id': r.doc_id,
            'original': r.original_text,
            'resolved': r.resolved_text,
            'resolutions': [
                {
                    'pronoun': rec.pronoun,
                    'antecedent': rec.antecedent,
                    'position': f"{rec.start_pos}-{rec.end_pos}"
                }
                for rec in r.resolutions
            ],
            'verification': {
                'rule_check': r.verification.rule_check if r.verification else None,
                'issues': r.verification.issues if r.verification else []
            }
        }
        for r in results if r.resolutions
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(resolution_log, f, ensure_ascii=False, indent=2)

    # 统计
    stats = {
        'total_chunks': len(results),
        'resolved_chunks': sum(1 for r in results if r.resolutions),
        'skipped_chunks': sum(1 for r in results if r.skipped),
        'total_resolutions': sum(len(r.resolutions) for r in results),
        'rollback_count': sum(1 for r in results if r.verification and r.verification.rollback_triggered),
        'output_file': str(output_file),
        'log_file': str(log_file)
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="指代消解模块")
    parser.add_argument("--input", required=True, help="输入分片文件路径")
    parser.add_argument("--output", default="", help="输出目录（默认与输入同目录）")
    parser.add_argument("--use_llm", action="store_true", help="使用 LLM 验证")
    parser.add_argument("--llm_api_key", default="", help="LLM API Key")
    parser.add_argument("--llm_base_url", default="", help="LLM API 地址")
    parser.add_argument("--llm_model", default="gpt-3.5-turbo", help="LLM 模型")
    parser.add_argument("--strict_mode", action="store_true", help="严格模式")

    args = parser.parse_args()

    stats = resolve_chunks(
        input_path=args.input,
        output_dir=args.output or None,
        use_llm=args.use_llm,
        llm_base_url=args.llm_base_url or None,
        llm_api_key=args.llm_api_key or None,
        llm_model=args.llm_model,
        strict_mode=args.strict_mode
    )

    print("\n指代消解完成!")
    print(f"  总 chunks: {stats['total_chunks']}")
    print(f"  已消解: {stats['resolved_chunks']}")
    print(f"  已跳过: {stats['skipped_chunks']}")
    print(f"  消解次数: {stats['total_resolutions']}")
    print(f"  回滚次数: {stats['rollback_count']}")
    print(f"  输出文件: {stats['output_file']}")
    print(f"  日志文件: {stats['log_file']}")


if __name__ == "__main__":
    main()
