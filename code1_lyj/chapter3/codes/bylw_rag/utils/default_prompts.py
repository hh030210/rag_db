"""
默认Prompt模板 - 针对NQ数据集优化
所有Prompt要求输出简短答案，适合Natural Questions数据集
"""
from typing import List
from core.prompt_module import PromptModule, StructuredPrompt


def create_system_prompt(domain: str) -> PromptModule:
    """创建系统Prompt - NQ风格"""
    content = '''You are a helpful assistant that answers questions based on the provided context. 
Provide concise, direct answers. If the answer is not in the context, say "unknown".
Always base your answer solely on the provided documents.'''

    return PromptModule(
        name=f"P_sys_{domain}",
        content=content,
        module_type="P_sys"
    )


def get_fact_retrieval_prompts(domain: str) -> List[StructuredPrompt]:
    """事实检索型Prompt - 4个变体，NQ风格简短回答"""
    P_sys = create_system_prompt(domain)
    prompts = []
    
    # 变体1: 标准事实检索 - 直接提取
    I_t_1 = PromptModule(
        name="I_t_fact_direct",
        content='''Answer the following question based ONLY on the provided documents.
Give a short, direct answer (1-5 words if possible).

Question: {{question}}''',
        module_type="I_t"
    )
    
    C_t_1 = PromptModule(
        name="C_t_fact_direct",
        content='''Documents:
{{context}}

Instructions:
- Find the exact answer in the documents above
- Copy the answer as it appears in the text
- Do not add any explanation''',
        module_type="C_t"
    )
    
    F_t_1 = PromptModule(
        name="F_t_short",
        content='''Output format:
Answer with just the answer text, nothing else.
Example: "Paris" or "The Battle of Hastings" or "42"
If not found in documents, output: "unknown"''',
        module_type="F_t"
    )
    
    U_t_1 = PromptModule(
        name="U_t_fact",
        content='''If the answer is not clearly stated in the documents, output "unknown".
Do not guess or use outside knowledge.''',
        module_type="U_t"
    )
    
    prompts.append(StructuredPrompt(
        name="fact_direct",
        question_type="fact_retrieval",
        domain=domain,
        P_sys=P_sys,
        I_t=I_t_1,
        C_t=C_t_1,
        F_t=F_t_1,
        U_t=U_t_1
    ))
    
    # 变体2: 强调精确匹配
    I_t_2 = PromptModule(
        name="I_t_fact_exact",
        content='''Find the exact answer to the question in the provided documents.
Extract the answer word-for-word from the text.

Question: {{question}}''',
        module_type="I_t"
    )
    
    C_t_2 = PromptModule(
        name="C_t_fact_exact",
        content='''Reference Documents:
{{context}}

Important: The answer must appear exactly in these documents.
Look for the specific words or phrase that answers the question.''',
        module_type="C_t"
    )
    
    F_t_2 = PromptModule(
        name="F_t_exact",
        content='''Output only the exact words from the document that answer the question.
No explanation, no preamble.
If not found: "unknown"''',
        module_type="F_t"
    )
    
    U_t_2 = PromptModule(
        name="U_t_exact",
        content='''Only answer if you find the exact words in the documents.
If uncertain, output "unknown".''',
        module_type="U_t"
    )
    
    prompts.append(StructuredPrompt(
        name="fact_exact",
        question_type="fact_retrieval",
        domain=domain,
        P_sys=P_sys,
        I_t=I_t_2,
        C_t=C_t_2,
        F_t=F_t_2,
        U_t=U_t_2
    ))
    
    # 变体3: 带引用的回答
    I_t_3 = PromptModule(
        name="I_t_fact_cited",
        content='''Answer the question using information from the documents.
Provide the answer and indicate which document it came from.

Question: {{question}}''',
        module_type="I_t"
    )
    
    C_t_3 = PromptModule(
        name="C_t_fact_cited",
        content='''Source Documents:
{{context}}

Task: Identify which document contains the answer and extract it.''',
        module_type="C_t"
    )
    
    F_t_3 = PromptModule(
        name="F_t_cited",
        content='''Format: Answer | Source
Example: "Paris | Document 1"
If not found: "unknown | none"''',
        module_type="F_t"
    )
    
    U_t_3 = PromptModule(
        name="U_t_cited",
        content='''If you cannot identify the specific source document, output "unknown | none".
Do not fabricate sources.''',
        module_type="U_t"
    )
    
    prompts.append(StructuredPrompt(
        name="fact_cited",
        question_type="fact_retrieval",
        domain=domain,
        P_sys=P_sys,
        I_t=I_t_3,
        C_t=C_t_3,
        F_t=F_t_3,
        U_t=U_t_3
    ))
    
    # 变体4: 多文档验证
    I_t_4 = PromptModule(
        name="I_t_fact_verified",
        content='''Find the answer to the question by checking all documents.
If multiple documents agree, provide that answer.
If documents disagree, note the conflict.

Question: {{question}}''',
        module_type="I_t"
    )
    
    C_t_4 = PromptModule(
        name="C_t_fact_verified",
        content='''Documents to verify:
{{context}}

Check: Do multiple documents confirm the same answer?''',
        module_type="C_t"
    )
    
    F_t_4 = PromptModule(
        name="F_t_verified",
        content='''Answer format:
- If consensus: "Answer: [answer]"
- If conflict: "Conflict: [answer1] vs [answer2]"
- If not found: "unknown"''',
        module_type="F_t"
    )
    
    U_t_4 = PromptModule(
        name="U_t_verified",
        content='''Report conflicts honestly. If documents disagree, state both answers.
If no clear answer, output "unknown".''',
        module_type="U_t"
    )
    
    prompts.append(StructuredPrompt(
        name="fact_verified",
        question_type="fact_retrieval",
        domain=domain,
        P_sys=P_sys,
        I_t=I_t_4,
        C_t=C_t_4,
        F_t=F_t_4,
        U_t=U_t_4
    ))
    
    return prompts


def get_subjective_opinion_prompts(domain: str) -> List[StructuredPrompt]:
    """主观观点型Prompt - 4个变体，NQ风格"""
    P_sys = create_system_prompt(domain)
    prompts = []
    
    # 变体1: 观点提取
    I_t_1 = PromptModule(
        name="I_t_opinion_extract",
        content='''Extract the main opinion or viewpoint from the documents regarding the question.
Give a brief summary (5-10 words).

Question: {{question}}''',
        module_type="I_t"
    )
    
    C_t_1 = PromptModule(
        name="C_t_opinion_extract",
        content='''Documents containing opinions:
{{context}}

Extract the key viewpoint expressed.''',
        module_type="C_t"
    )
    
    F_t_1 = PromptModule(
        name="F_t_opinion_short",
        content='''Output: Brief opinion summary (5-10 words max)
Example: "Supports climate action" or "Critical of the policy"
If no opinion stated: "no opinion stated"''',
        module_type="F_t"
    )
    
    U_t_1 = PromptModule(
        name="U_t_opinion",
        content='''If documents present facts without clear opinion, state "no opinion stated".
Do not infer opinions not explicitly expressed.''',
        module_type="U_t"
    )
    
    prompts.append(StructuredPrompt(
        name="opinion_extract",
        question_type="subjective_opinion",
        domain=domain,
        P_sys=P_sys,
        I_t=I_t_1,
        C_t=C_t_1,
        F_t=F_t_1,
        U_t=U_t_1
    ))
    
    # 变体2: 多观点对比
    I_t_2 = PromptModule(
        name="I_t_opinion_compare",
        content='''What viewpoints do the documents present about this question?
List the main perspectives briefly.

Question: {{question}}''',
        module_type="I_t"
    )
    
    C_t_2 = PromptModule(
        name="C_t_opinion_compare",
        content='''Documents with various viewpoints:
{{context}}

Identify different perspectives presented.''',
        module_type="C_t"
    )
    
    F_t_2 = PromptModule(
        name="F_t_opinion_list",
        content='''Format: Viewpoint 1; Viewpoint 2; ...
Keep each to 3-5 words.
Example: "Pro-reform; Anti-reform; Neutral"
If single viewpoint: "[viewpoint] | single source"''',
        module_type="F_t"
    )
    
    U_t_2 = PromptModule(
        name="U_t_opinion_multi",
        content='''If documents present only one side, note "limited perspectives".
If no clear viewpoints: "no stated opinions"''',
        module_type="U_t"
    )
    
    prompts.append(StructuredPrompt(
        name="opinion_compare",
        question_type="subjective_opinion",
        domain=domain,
        P_sys=P_sys,
        I_t=I_t_2,
        C_t=C_t_2,
        F_t=F_t_2,
        U_t=U_t_2
    ))
    
    # 变体3: 立场识别
    I_t_3 = PromptModule(
        name="I_t_opinion_stance",
        content='''Determine the stance of the documents on this issue.
Classify as: For, Against, Neutral, or Mixed.

Question: {{question}}''',
        module_type="I_t"
    )
    
    C_t_3 = PromptModule(
        name="C_t_opinion_stance",
        content='''Documents to analyze:
{{context}}

Determine the overall stance presented.''',
        module_type="C_t"
    )
    
    F_t_3 = PromptModule(
        name="F_t_stance",
        content='''Output: For / Against / Neutral / Mixed
Optionally add brief justification (max 5 words)
Example: "Against | cites economic concerns"''',
        module_type="F_t"
    )
    
    U_t_3 = PromptModule(
        name="U_t_stance",
        content='''If stance is unclear or documents conflict, output "Mixed | conflicting sources".
Do not force a classification.''',
        module_type="U_t"
    )
    
    prompts.append(StructuredPrompt(
        name="opinion_stance",
        question_type="subjective_opinion",
        domain=domain,
        P_sys=P_sys,
        I_t=I_t_3,
        C_t=C_t_3,
        F_t=F_t_3,
        U_t=U_t_3
    ))
    
    # 变体4: 情感分析
    I_t_4 = PromptModule(
        name="I_t_opinion_sentiment",
        content='''What is the sentiment of the documents toward the subject?
Positive, Negative, or Neutral?

Question: {{question}}''',
        module_type="I_t"
    )
    
    C_t_4 = PromptModule(
        name="C_t_opinion_sentiment",
        content='''Documents for sentiment analysis:
{{context}}

Analyze the tone and sentiment.''',
        module_type="C_t"
    )
    
    F_t_4 = PromptModule(
        name="F_t_sentiment",
        content='''Output: Positive / Negative / Neutral
Add key phrase if relevant (max 5 words)
Example: "Positive | praises innovation"''',
        module_type="F_t"
    )
    
    U_t_4 = PromptModule(
        name="U_t_sentiment",
        content='''If sentiment is mixed or unclear, output "Mixed".
If factual only: "Neutral | factual only"''',
        module_type="U_t"
    )
    
    prompts.append(StructuredPrompt(
        name="opinion_sentiment",
        question_type="subjective_opinion",
        domain=domain,
        P_sys=P_sys,
        I_t=I_t_4,
        C_t=C_t_4,
        F_t=F_t_4,
        U_t=U_t_4
    ))
    
    return prompts


def get_exploratory_open_prompts(domain: str) -> List[StructuredPrompt]:
    """探索开放型Prompt - 4个变体，NQ风格简短回答"""
    P_sys = create_system_prompt(domain)
    prompts = []
    
    # 变体1: 关键信息提取
    I_t_1 = PromptModule(
        name="I_t_explore_key",
        content='''Based on the documents, what are the key points related to this question?
List 2-3 brief points.

Question: {{question}}''',
        module_type="I_t"
    )
    
    C_t_1 = PromptModule(
        name="C_t_explore_key",
        content='''Reference documents:
{{context}}

Extract the most relevant information.''',
        module_type="C_t"
    )
    
    F_t_1 = PromptModule(
        name="F_t_key_points",
        content='''Format: Point 1; Point 2; Point 3
Each point max 5 words.
Example: "Founded in 1998; CEO is Smith; HQ in NY"
If limited info: "[single point] | limited info"''',
        module_type="F_t"
    )
    
    U_t_1 = PromptModule(
        name="U_t_explore",
        content='''If documents provide limited information, state what is known and note "incomplete".
Do not speculate beyond the documents.''',
        module_type="U_t"
    )
    
    prompts.append(StructuredPrompt(
        name="explore_key",
        question_type="exploratory_open",
        domain=domain,
        P_sys=P_sys,
        I_t=I_t_1,
        C_t=C_t_1,
        F_t=F_t_1,
        U_t=U_t_1
    ))
    
    # 变体2: 原因分析
    I_t_2 = PromptModule(
        name="I_t_explore_why",
        content='''According to the documents, why did this happen or what explains this?
Give a brief explanation (10 words max).

Question: {{question}}''',
        module_type="I_t"
    )
    
    C_t_2 = PromptModule(
        name="C_t_explore_why",
        content='''Documents with explanations:
{{context}}

Find the stated reasons or causes.''',
        module_type="C_t"
    )
    
    F_t_2 = PromptModule(
        name="F_t_why",
        content='''Output: Brief explanation (max 10 words)
Example: "Due to economic recession" or "Caused by policy change"
If no explanation: "no explanation in documents"''',
        module_type="F_t"
    )
    
    U_t_2 = PromptModule(
        name="U_t_why",
        content='''If documents describe what happened but not why, state "cause not stated".
Distinguish between stated and implied causes.''',
        module_type="U_t"
    )
    
    prompts.append(StructuredPrompt(
        name="explore_why",
        question_type="exploratory_open",
        domain=domain,
        P_sys=P_sys,
        I_t=I_t_2,
        C_t=C_t_2,
        F_t=F_t_2,
        U_t=U_t_2
    ))
    
    # 变体3: 影响识别
    I_t_3 = PromptModule(
        name="I_t_explore_impact",
        content='''What impact or effect is described in the documents?
State briefly (5-10 words).

Question: {{question}}''',
        module_type="I_t"
    )
    
    C_t_3 = PromptModule(
        name="C_t_explore_impact",
        content='''Documents describing impacts:
{{context}}

Identify the main effects mentioned.''',
        module_type="C_t"
    )
    
    F_t_3 = PromptModule(
        name="F_t_impact",
        content='''Output: Impact description (5-10 words)
Example: "Increased unemployment by 5%" or "Improved air quality"
If no impact mentioned: "no impact stated"''',
        module_type="F_t"
    )
    
    U_t_3 = PromptModule(
        name="U_t_impact",
        content='''If documents mention multiple impacts, list the main one or say "multiple impacts".
If speculative: "speculative | not confirmed"''',
        module_type="U_t"
    )
    
    prompts.append(StructuredPrompt(
        name="explore_impact",
        question_type="exploratory_open",
        domain=domain,
        P_sys=P_sys,
        I_t=I_t_3,
        C_t=C_t_3,
        F_t=F_t_3,
        U_t=U_t_3
    ))
    
    # 变体4: 关系识别
    I_t_4 = PromptModule(
        name="I_t_explore_relation",
        content='''What relationship or connection do the documents describe?
Describe briefly (5-10 words).

Question: {{question}}''',
        module_type="I_t"
    )
    
    C_t_4 = PromptModule(
        name="C_t_explore_relation",
        content='''Documents describing relationships:
{{context}}

Identify the connection between entities.''',
        module_type="C_t"
    )
    
    F_t_4 = PromptModule(
        name="F_t_relation",
        content='''Output: Relationship description (5-10 words)
Example: "Parent company of subsidiary" or "Caused by same factor"
If unclear: "relationship unclear"''',
        module_type="F_t"
    )
    
    U_t_4 = PromptModule(
        name="U_t_relation",
        content='''If relationship is implied but not stated, note "implied | not explicit".
If no relationship: "no relationship described"''',
        module_type="U_t"
    )
    
    prompts.append(StructuredPrompt(
        name="explore_relation",
        question_type="exploratory_open",
        domain=domain,
        P_sys=P_sys,
        I_t=I_t_4,
        C_t=C_t_4,
        F_t=F_t_4,
        U_t=U_t_4
    ))
    
    return prompts


def get_short_answer_prompts(domain: str) -> List[StructuredPrompt]:
    """简短回答型Prompt - 4个变体，专门针对NQ数据集的短答案"""
    P_sys = create_system_prompt(domain)
    prompts = []
    
    # 变体1: 极简回答
    I_t_1 = PromptModule(
        name="I_t_short_minimal",
        content='''Answer the question using 1-3 words from the documents.

Question: {{question}}''',
        module_type="I_t"
    )
    
    C_t_1 = PromptModule(
        name="C_t_short_minimal",
        content='''Context:
{{context}}

Find the shortest accurate answer in the text.''',
        module_type="C_t"
    )
    
    F_t_1 = PromptModule(
        name="F_t_minimal",
        content='''Output: 1-3 words only
Examples: "Paris", "1999", "Barack Obama", "The Great Gatsby"
If not found: "unknown"''',
        module_type="F_t"
    )
    
    U_t_1 = PromptModule(
        name="U_t_minimal",
        content='''If answer requires more than 3 words, provide the most concise version possible.
If uncertain: "unknown"''',
        module_type="U_t"
    )
    
    prompts.append(StructuredPrompt(
        name="short_minimal",
        question_type="short_answer",
        domain=domain,
        P_sys=P_sys,
        I_t=I_t_1,
        C_t=C_t_1,
        F_t=F_t_1,
        U_t=U_t_1
    ))
    
    # 变体2: 实体提取
    I_t_2 = PromptModule(
        name="I_t_short_entity",
        content='''Extract the named entity that answers this question.
Person, place, organization, or date.

Question: {{question}}''',
        module_type="I_t"
    )
    
    C_t_2 = PromptModule(
        name="C_t_short_entity",
        content='''Documents:
{{context}}

Look for named entities (names, places, dates, organizations).''',
        module_type="C_t"
    )
    
    F_t_2 = PromptModule(
        name="F_t_entity",
        content='''Output: Entity name only
Examples: "Albert Einstein", "New York City", "July 4, 1776"
If not found: "unknown"''',
        module_type="F_t"
    )
    
    U_t_2 = PromptModule(
        name="U_t_entity",
        content='''If multiple entities could answer, choose the most specific one.
If ambiguous: "ambiguous | [options]"''',
        module_type="U_t"
    )
    
    prompts.append(StructuredPrompt(
        name="short_entity",
        question_type="short_answer",
        domain=domain,
        P_sys=P_sys,
        I_t=I_t_2,
        C_t=C_t_2,
        F_t=F_t_2,
        U_t=U_t_2
    ))
    
    # 变体3: 短语提取
    I_t_3 = PromptModule(
        name="I_t_short_phrase",
        content='''Extract the exact phrase from the documents that answers this question.
Copy it word-for-word.

Question: {{question}}''',
        module_type="I_t"
    )
    
    C_t_3 = PromptModule(
        name="C_t_short_phrase",
        content='''Source text:
{{context}}

Copy the exact phrase that answers the question.''',
        module_type="C_t"
    )
    
    F_t_3 = PromptModule(
        name="F_t_phrase",
        content='''Output: Exact phrase from text
Do not paraphrase or modify.
If not found: "unknown"''',
        module_type="F_t"
    )
    
    U_t_3 = PromptModule(
        name="U_t_phrase",
        content='''Only copy text that appears exactly in the documents.
Do not change words or word order.''',
        module_type="U_t"
    )
    
    prompts.append(StructuredPrompt(
        name="short_phrase",
        question_type="short_answer",
        domain=domain,
        P_sys=P_sys,
        I_t=I_t_3,
        C_t=C_t_3,
        F_t=F_t_3,
        U_t=U_t_3
    ))
    
    # 变体4: 日期/数字提取
    I_t_4 = PromptModule(
        name="I_t_short_date",
        content='''Extract the date, year, or number that answers this question.

Question: {{question}}''',
        module_type="I_t"
    )
    
    C_t_4 = PromptModule(
        name="C_t_short_date",
        content='''Documents:
{{context}}

Find the specific date, year, or numerical answer.''',
        module_type="C_t"
    )
    
    F_t_4 = PromptModule(
        name="F_t_date",
        content='''Output: Date/Number only
Examples: "2020", "March 15, 2020", "42", "$50 million"
If not found: "unknown"''',
        module_type="F_t"
    )
    
    U_t_4 = PromptModule(
        name="U_t_date",
        content='''If multiple dates/numbers mentioned, provide the one most relevant to the question.
If approximate: "approx [value]"''',
        module_type="U_t"
    )
    
    prompts.append(StructuredPrompt(
        name="short_date",
        question_type="short_answer",
        domain=domain,
        P_sys=P_sys,
        I_t=I_t_4,
        C_t=C_t_4,
        F_t=F_t_4,
        U_t=U_t_4
    ))
    
    return prompts


def get_default_prompts(question_type: str, domain: str = "general") -> List[StructuredPrompt]:
    """
    获取默认Prompt集合
    
    Args:
        question_type: 问题类型
        domain: 领域
        
    Returns:
        Prompt列表（每种类型4个）
    """
    getters = {
        "fact_retrieval": get_fact_retrieval_prompts,
        "subjective_opinion": get_subjective_opinion_prompts,
        "exploratory_open": get_exploratory_open_prompts,
        "short_answer": get_short_answer_prompts
    }
    
    getter = getters.get(question_type)
    if not getter:
        raise ValueError(f"未知的问题类型: {question_type}")
    
    return getter(domain)
