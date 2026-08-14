import json
import requests
import re
import config
import os
from tqdm import tqdm
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def rule_based_clean(text):
    """
    独立规则去噪流程
    """
    # 1. 去除 VLM 占位符 <|LOC_...|> (如 <|LOC_399|>)
    text = re.sub(r'<\|LOC_\d+\|>', '', text)
    
    # 2. 去除控制字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # 3. 清理多余空格和换行
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

class LLMDenoiser:
    def __init__(self, chunk_size=2000, log_file=None, organize_log_file=None):
        self.api_key = config.API_KEY
        self.api_url = config.API_URL
        self.model_name = config.TEXT_MODEL_NAME
        self.chunk_size = chunk_size
        self.log_file = log_file if log_file else "llm_interaction.log"
        self.organize_log_file = organize_log_file if organize_log_file else "organize_interaction.log"
        self.max_workers = config.API_MAX_WORKERS
        self.log_lock = threading.Lock()
        self.default_noise_rules = [
            "格式类噪音：多余空格、连续换行、空行段落",
            "字符噪声：乱码字符、控制字符、异常编码符号",
            "结构性冗余：排版残留行首编号碎片、对齐符、断行符、页眉页脚、水印等"
        ]
        if self.log_file:
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
        if self.organize_log_file:
            log_dir = os.path.dirname(self.organize_log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

    def _log_interaction(self, prompt, response_content):
        with self.log_lock:
            mode = "w" if not os.path.exists(self.log_file) else "a"
            with open(self.log_file, mode, encoding="utf-8") as f:
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*50}\n")
                f.write(f"--- PROMPT ---{prompt}\n")
                f.write(f"--- RESPONSE ---{response_content}\n")
                f.write(f"{'='*50}\n")

    def _log_organize_interaction(self, prompt, response_content):
        with self.log_lock:
            mode = "w" if not os.path.exists(self.organize_log_file) else "a"
            with open(self.organize_log_file, mode, encoding="utf-8") as f:
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*50}\n")
                f.write(f"--- PROMPT ---{prompt}\n")
                f.write(f"--- RESPONSE ---{response_content}\n")
                f.write(f"{'='*50}\n")

    def _call_llm(self, prompt, desc="Calling LLM", show_progress=True):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一个高效的文本处理工具。必须严格按照 JSON 格式输出，禁止任何解释、禁止思考、禁止输出逻辑推理过程。"},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.01,
            "max_tokens": 4000,
            "stream": False
        }
        
        try:
            if show_progress:
                print(f"[进度] {desc}...")
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            self._log_interaction(prompt, content)
            try:
                return json.loads(content)
            except json.JSONDecodeError as je:
                print(f"\n[警告] JSON 解析失败，尝试正则恢复...")
                text_match = re.search(r'\"去噪后文本\"\s*:\s*\"(.*?)\"(?=\s*,\s*\"噪声内容\"|\s*\})', content, re.DOTALL)
                if text_match:
                    return {"去噪后文本": text_match.group(1).encode().decode('unicode_escape', errors='ignore'), "噪声内容": []}
                raise je
        except Exception as e:
            err_msg = f"LLM 调用失败: {e}"
            self._log_interaction(prompt, err_msg)
            return None

    def extract_noise_types(self, text):
        print("\n[日志] 开始提取领域噪声特征 (基于文本采样)...")
        sample_text = text[:4000]
        prompt = f"""您需要根据"说明"中的要求，结合给定的"输入文本"，分析该文本所属的业务领域，并自动总结该领域文本中可能存在的噪声类型。

【输入文本】
{sample_text}

【说明】
1. 识别领域背景。
2. 总结该领域常见的低价值噪声（如解析残留、无关页码）。
3. **特别注意**：请明确区分"专业知识内容"与"噪声"。技术原理、逻辑推导、开发背景、算法说明（如 RAG、Beam Search 等相关描述）属于核心知识，绝对不能判定为噪声。

【输出格式】
JSON 格式，包含"领域判定结果"与"噪声类型列表"。
"""
        res = self._call_llm(prompt, desc="提取噪声特征")
        if res:
            original_list = res.get("噪声类型列表", [])
            res["噪声类型列表"] = self.default_noise_rules + original_list
            print(f"[日志] 领域识别成功: {res.get('领域判定结果')}")
        return res

    def _semantic_split(self, text):
        sentences = re.findall(r'[^。！？\.!\?\n]+[。！？\.!\?\n]*', text)
        chunks = []
        current_chunk = ""
        for sent in sentences:
            if len(current_chunk) + len(sent) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = sent
                else:
                    sub_chunks = [sent[i:i + self.chunk_size] for i in range(0, len(sent), self.chunk_size)]
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1]
            else:
                current_chunk += sent
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def denoise_text(self, text, noise_info, file_type="PDF/PPT"):
        domain = noise_info.get("领域判定结果", "未知领域")
        noise_types = "\n".join([f"- {n}" for n in noise_info.get("噪声类型列表", [])])
        chunks = self._semantic_split(text)
        total_chunks = len(chunks)
        print(f"[日志] 文本长度 {len(text)}，基于语义切分为 {total_chunks} 个段落进行并行去噪...")
        print(f"[日志] 当前设置的并行 API 任务数 (max_workers): {self.max_workers}")
        final_results = [None] * total_chunks
        all_noise_content = []
        
        def process_single_chunk(index, chunk):
            prompt = f"""你现在的任务是修复并去噪一份由 {file_type} 转换而来的学术/技术文档。

【重要背景】
输入文本中存在大量"人为断行"。很多行可能以句号、逗号、括号或孤立数字开头，这通常是因为原始排版换行导致的。这些内容是正文的延续，**绝对不是噪声**！
【输入文本】
{chunk}

【噪声约束参考】
领域：{domain}
已知噪声模式：{noise_types}

【示例参考】
示例1：
输入：本系统采用检索增强生成（RAG）框架构建企业级知识问答服务。系统首先对多源异构企业数据进行统一解析与标准化转换，并通过规则过滤与语义去噪提升数据质量。在查询阶段，用户问题经过向量检索与全文检索获取候选文档片段，最终由大语言模型在可信上下文约束下生成回答。—— 第 12 页 ——版权所有 © 2021 XX科技有限公司Table 3-2 Error! Reference source not found.AAAAAAAAAAAAAAAAAAAA如需技术支持请联系：admin@example.com
输出：{{
  "去噪后文本": "本系统采用检索增强生成（RAG）框架构建企业级知识问答服务。系统首先对多源异构企业数据进行统一解析与标准化转换，并通过规则过滤与语义去噪提升数据质量。在查询阶段，用户问题经过向量检索与全文检索获取候选文档片段，最终由大语言模型在可信上下文约束下生成回答。",
  "噪声内容": [
      "—— 第 12 页 ——",
      "版权所有 © 2021 XX科技有限公司",
      "Table 3-2 Error! Reference source not found.",
      "AAAAAAAAAAAAAAAAAAAA",
      "如需技术支持请联系：admin@example.com"
  ]
}}

示例2：
输入：本 系 统   采 用   RAG  架 构
  
  
 用于   企业   知识   问答。 
 � � � 
 系统  首先  对   数据  进行  处理。
输出：{{
  "去噪后文本": "本系统采用 RAG 架构用于企业知识问答。系统首先对数据进行处理。",
  "噪声内容": [
    "本 系 统   采 用   RAG  架 构",
    "用于   企业   知识   问答。",
    "� � �",
    "系统  首先  对   数据  进行  处理。"
  ]
}}

示例3（噪声识别示例）：
输入：·21·敢于练字岗。日如常，Cont serious居事。生态友长，少年县 Mayldots。生命福报，华灯累计，风雪暖火。
输出：{{
  "去噪后文本": "·21·敢于练字岗。日如常居事。生态友长，少年县。生命福报，华灯累计，风雪暖火。",
  "噪声内容": [
    "Cont serious",
    "Mayldots"
  ]
}}

示例4（多语言噪声识别）：
输入：克 apk, Willie part<fcel>若为否则，则%(1g)不
输出：{{
  "去噪后文本": "克若为否则，则不",
  "噪声内容": [
    " apk, Willie part<fcel>",
    "%(1g)"
  ]
}}

示例5（页码和分隔符识别）：
输入：2.
--- Page Break ---
12
输出：{{
  "去噪后文本": "",
  "噪声内容": [
    "2.",
    "--- Page Break ---",
    "12"
  ]
}}

【噪声识别规则 - 必须严格遵守】
1. **上下文不连贯的外文片段**：删除与前后文主题完全无关的英文单词或短语（如示例3中的"Cont serious"、"Mayldots"）
2. **非中英语言内容**：删除所有非中文、非英文的语言内容（如泰米尔语、阿拉伯语、印地语等）
3. **代码片段和标记**：删除明显的代码片段、HTML标签、LaTeX命令等（如"<fcel>"、"%(1g)"、"\\text"等）
4. **无意义的混合文本**：删除中文中混杂的无意义英文、数字、符号组合（如"plaotherapy"、"renewal"、"space"等与主题无关的词）
5. **页码和分隔符**：删除页码标记（如"2.", "12"）、分页符（如"--- Page Break ---"）等排版标记
6. **乱码和编码错误**：删除明显的乱码字符、编码错误产生的无意义字符串
7. **极端噪声行**：对于整行都是噪声的行（如示例5），整行删除

【噪声处理原则】
1. **精准删除**：只删除识别出的噪声部分，保留行中有意义的内容
2. **保持连贯**：删除噪声后，确保剩余文本语义连贯、可读
3. **不删除有意义内容**：严禁删除专业术语、人名、地名、有意义的英文缩写等
4. **噪声聚合**：'噪声内容'数组中的每个元素必须是完整的剔除片段，**绝对禁止**将噪声拆分为单个字符

【说明 (禁令)】
1. **噪声识别与删除**：只能删除上述规则中定义的噪声类型。
2. **换行清理**：必须删除所有多余的空行和连续换行。输出文本应保持段落紧凑，行与行之间不应有不必要的空白。 
3. **保留核心**：严禁删除任何技术描述、算法逻辑（如 RAG, ICL, Self-RAG 相关内容）。  
4. **禁止改写**：直接剔除噪声，不得修改或总结原文。
5. **严禁删除断行存续内容**：凡是行首为标点符号（如"。、，；"）、括号、序号（如"10."）但后续紧跟有意义文字的内容，必须 100% 原样保留并尝试与前文逻辑连接。
6. **必须保留重要数字**：以下类型的数字绝对不能删除，必须保留在原文中：
   - 年份（如1980、1999、2004、19世纪、20世纪等）
   - 百分比（如79.5%、12.2%、53%等）
   - 统计数据（如样本数量、人数等）
   - 任何有实际意义的数字信息
   - 被切断的数字（如"19 20"应该理解为"19 20"，不能删除）
7. **保留所有有意义的文本**：这是最重要的原则，必须严格遵守：
   - 任何包含中文、英文或其他语言文字的文本都必须保留
   - 任何包含语义信息的文本都必须保留（即使有乱码或格式问题，如果有，只删除乱码等，而保留正常的文字）
   - 任何包含专业术语、人名、地名、机构名等的文本都必须保留
   - 即使文本看起来不完整或格式不正确，只要包含有意义的信息就必须保留
   - **核心原则**：只删除真正的噪声（重复的水印、页眉页脚、纯数字页码），保留所有有意义的文本内容
【输出格式】
JSON 格式，包含"去噪后文本"与"噪声内容"字段。
"""
            try:
                res = self._call_llm(prompt, show_progress=False)
                if res:
                    cleaned_chunk = res.get("去噪后文本", chunk)
                    cleaned_chunk = re.sub(r'\n([。，、；）])', r'\1', cleaned_chunk)
                    cleaned_chunk = re.sub(r'^\s+', '', cleaned_chunk, flags=re.MULTILINE)
                    cleaned_chunk = re.sub(r'\s+$', '', cleaned_chunk, flags=re.MULTILINE)
                    cleaned_chunk = re.sub(r'[ \t]+', ' ', cleaned_chunk)
                    return index, cleaned_chunk, res.get("噪声内容", [])
                else:
                    return index, chunk, []
            except Exception as e:
                print(f"\n[错误] 处理第 {index+1} 段时发生 API 异常: {e}")
                return index, chunk, []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {executor.submit(process_single_chunk, i, chunk): i for i, chunk in enumerate(chunks)}
            with tqdm(total=total_chunks, desc="并行去噪进度") as pbar:
                for future in as_completed(future_to_index):
                    try:
                        index, cleaned_chunk, noises = future.result()
                        final_results[index] = cleaned_chunk
                        if isinstance(noises, list):
                            all_noise_content.extend(noises)
                    except Exception as e:
                        print(f"\n[错误] 并行任务执行异常: {e}")
                    pbar.update(1)
        
        full_text = "".join(final_results)
        full_text = re.sub(r'^\s+', '', full_text, flags=re.MULTILINE)
        full_text = re.sub(r'\s+$', '', full_text, flags=re.MULTILINE)
        full_text = re.sub(r'[ \t]+', ' ', full_text)
        full_text = re.sub(r'\n{2,}', '\n', full_text)
        
        return {
            "去噪后文本": full_text.strip(),
            "噪声内容": all_noise_content
        }

    def organize_content(self, text, file_type="PDF/PPT"):
        print(f"\n[日志] 开始进行内容重组和归纳...")
        if len(text) > 6000:
            chunks = [text[i:i+6000] for i in range(0, len(text), 6000)]
        else:
            chunks = [text]
        all_paragraphs = []
        all_responses = []
        
        def process_single_chunk(index, chunk):
            prompt = f"""你现在的任务是对一份由 {file_type} 转换而来的学术/技术文档进行内容重组和归纳。

【输入文本】
{chunk}

【核心任务】
将零散的、断续的句子重新组织成连贯的段落。原文可能因为PDF解析或VLM解析导致句子被切断，你需要：
1. **连接断句**：将因为换行导致的断掉的句子重新连接起来
2. **段落重组**：将零散的句子按照语义逻辑组织成连贯的段落
3. **内容归纳**：将相关的句子合并成一个段落，形成有意义的语义单元
4. **删除冗余**：删除重复的内容、无意义的符号、参考文献列表等

【特别注意】
- **连接被切断的数字**：如"19 20"应该连接成"19 20"，"世纪 末"应该连接成"世纪末"
- **连接被切断的年份**：如"19 20"应该理解为"19 20"（19世纪20年代），"1999 2004"应该理解为"1999-2004"
- **删除多余空格**：删除字符之间的多余空格，如"19 20"应该变成"19 20"
- **确保段落连贯**：每个段落应该是一个完整的语义单元，不要有断断续续的感觉

【必须删除的内容】
1. 大量无意义的符号（如连续的引号、空格等）
2. 连续的标点符号（如 ， ， ， 等）
3. 参考文献列表（如作者、年份、出版社、页码等单独的参考文献信息）
4. 没有实际意义的句子（如只包含标点符号、数字、符号等）
5. 对RAG没有帮助的句子

【重要约束】
- 不得添加原文中没有的信息
- 不得改变原文的核心观点和结论
- 不得删除任何重要的数据或统计信息
- 确保输出文本的连贯性和可读性
- 每个段落应该是一个完整的语义单元，主题明确
- 段落之间用空行分隔
- 段落内部不要有换行符，应该是一整段连续的文字

【输出格式】
必须严格按照以下JSON格式输出：
{{
    "重组后段落": [
        "段落1的完整内容...",
        "段落2的完整内容...",
        "段落3的完整内容..."
    ]
}}

注意：
- 每个段落应该是一个完整的语义单元
- 段落内部不要有换行符，应该是一整段连续的文字
- 段落之间用空行分隔
- 不要添加任何额外的格式标记或说明
"""
            try:
                res = self._call_llm(prompt, show_progress=False)
                if res and isinstance(res, dict):
                    paragraphs = res.get("重组后段落", [])
                    if isinstance(paragraphs, list):
                        self._log_organize_interaction(prompt, json.dumps(res, ensure_ascii=False, indent=2))
                        return index, paragraphs, res
                    else:
                        organized_text = res.get("重组后文本", chunk)
                        self._log_organize_interaction(prompt, json.dumps(res, ensure_ascii=False, indent=2))
                        return index, [organized_text], res
                elif isinstance(res, str):
                    self._log_organize_interaction(prompt, res)
                    return index, [res], {"原始响应": res}
                else:
                    self._log_organize_interaction(prompt, chunk)
                    return index, [chunk], {"原始输入": chunk}
            except Exception as e:
                print(f"\n[错误] 内容重组时发生异常: {e}")
                self._log_organize_interaction(prompt, f"异常: {e}")
                return index, [chunk], {"异常": str(e)}
        
        total_chunks = len(chunks)
        final_results = [None] * total_chunks
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {executor.submit(process_single_chunk, i, chunk): i for i, chunk in enumerate(chunks)}
            with tqdm(total=total_chunks, desc="内容重组和归纳进度") as pbar:
                for future in as_completed(future_to_index):
                    try:
                        index, paragraphs, response = future.result()
                        final_results[index] = (paragraphs, response)
                    except Exception as e:
                        print(f"\n[错误] 并行任务执行异常: {e}")
                    pbar.update(1)
        
        for paragraphs, response in final_results:
            if paragraphs:
                all_paragraphs.extend(paragraphs)
            if response:
                all_responses.append(response)
        
        print(f"[日志] 内容重组完成，共生成 {len(all_paragraphs)} 个段落")
        return all_paragraphs, all_responses
