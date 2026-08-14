import pandas as pd
import json
import re
import numpy as np
from html import unescape
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """处理numpy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def clean_html(html_text):
    """将HTML转换为纯文本"""
    if not isinstance(html_text, str):
        return ""

    # 解码HTML实体
    text = unescape(html_text)

    # 移除script和style标签及其内容
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # 移除HTML标签
    text = re.sub(r'<[^>]+>', ' ', text)

    # 合并多个空格
    text = re.sub(r'\s+', ' ', text)

    # 去除首尾空格
    text = text.strip()

    return text


def extract_document_text(document_field):
    """从document字段提取HTML并转换为纯文本"""
    if isinstance(document_field, dict):
        html = document_field.get('html', '')
        return clean_html(html)
    elif isinstance(document_field, str):
        return clean_html(document_field)
    return ""


def extract_question_text(question_field):
    """从question字段提取问题文本"""
    if isinstance(question_field, dict):
        return question_field.get('text', str(question_field))
    elif isinstance(question_field, str):
        return question_field
    return str(question_field)


def to_python_type(obj):
    """将numpy类型转换为Python原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, bytes):
        return obj.decode('utf-8')
    return obj


def extract_annotations(annotations_field, document_text):
    """提取标注信息中的短答案和长答案"""
    result = {
        "short_answers": [],
        "long_answers": [],
        "yes_no_answer": None
    }

    if not isinstance(annotations_field, dict):
        return result

    # 获取各个字段的数组
    annotation_ids = annotations_field.get('id', [])
    long_answers_arr = annotations_field.get('long_answer', [])
    short_answers_arr = annotations_field.get('short_answers', [])
    yes_no_answers = annotations_field.get('yes_no_answer', [])

    # 转换为列表
    if hasattr(annotation_ids, '__iter__') and not isinstance(annotation_ids, (list, str)):
        annotation_ids = annotation_ids.tolist() if hasattr(annotation_ids, 'tolist') else list(annotation_ids)
    if hasattr(long_answers_arr, '__iter__') and not isinstance(long_answers_arr, (list, str)):
        long_answers_arr = long_answers_arr.tolist() if hasattr(long_answers_arr, 'tolist') else list(long_answers_arr)
    if hasattr(short_answers_arr, '__iter__') and not isinstance(short_answers_arr, (list, str)):
        short_answers_arr = short_answers_arr.tolist() if hasattr(short_answers_arr, 'tolist') else list(short_answers_arr)
    if hasattr(yes_no_answers, '__iter__') and not isinstance(yes_no_answers, (list, str)):
        yes_no_answers = yes_no_answers.tolist() if hasattr(yes_no_answers, 'tolist') else list(yes_no_answers)

    # 将document按token分割
    tokens = document_text.split()

    # 处理每个标注员的标注
    for i in range(len(annotation_ids)):
        # 提取短答案
        if i < len(short_answers_arr):
            sa = short_answers_arr[i]
            if isinstance(sa, dict):
                text_arr = to_python_type(sa.get('text', []))
                if isinstance(text_arr, list) and len(text_arr) > 0:
                    result["short_answers"].extend(text_arr)

                # 根据token位置提取文本
                start_token_arr = to_python_type(sa.get('start_token', []))
                end_token_arr = to_python_type(sa.get('end_token', []))

                if isinstance(start_token_arr, list) and isinstance(end_token_arr, list):
                    for s, e in zip(start_token_arr, end_token_arr):
                        if s >= 0 and e > s and s < len(tokens):
                            answer_text = ' '.join(tokens[s:e])
                            if answer_text and answer_text not in result["short_answers"]:
                                result["short_answers"].append(answer_text)

        # 提取长答案
        if i < len(long_answers_arr):
            la = long_answers_arr[i]
            if isinstance(la, dict):
                start_token = to_python_type(la.get('start_token', -1))
                end_token = to_python_type(la.get('end_token', -1))

                if start_token != -1 and end_token != -1 and start_token < end_token:
                    if start_token < len(tokens):
                        answer_text = ' '.join(tokens[start_token:end_token])
                        result["long_answers"].append({
                            "annotator_id": to_python_type(annotation_ids[i]) if i < len(annotation_ids) else None,
                            "start_token": start_token,
                            "end_token": end_token,
                            "text": answer_text
                        })

        # 提取是否/否答案
        if i < len(yes_no_answers):
            yes_no = to_python_type(yes_no_answers[i])
            if yes_no != -1 and result["yes_no_answer"] is None:
                result["yes_no_answer"] = "YES" if yes_no == 1 else "NO" if yes_no == 0 else None

    return result


def extract_long_answer_candidates(candidates_field, document_text):
    """提取长答案候选并转换为文本"""
    result = []

    if not isinstance(candidates_field, dict):
        return result

    # 获取各个字段
    top_level = candidates_field.get('top_level', [])
    start_tokens = candidates_field.get('start_token', [])
    end_tokens = candidates_field.get('end_token', [])
    start_bytes = candidates_field.get('start_byte', [])
    end_bytes = candidates_field.get('end_byte', [])

    # 转换为列表
    top_level = to_python_type(top_level)
    start_tokens = to_python_type(start_tokens)
    end_tokens = to_python_type(end_tokens)

    # 将document按token分割
    tokens = document_text.split()

    # 提取候选答案（只取前10个top_level为True的）
    count = 0
    for i, is_top in enumerate(top_level):
        if is_top and i < len(start_tokens) and i < len(end_tokens):
            start = int(start_tokens[i])
            end = int(end_tokens[i])

            if start >= 0 and end > start and start < len(tokens):
                candidate_text = ' '.join(tokens[start:end])
                result.append({
                    "index": i,
                    "start_token": start,
                    "end_token": end,
                    "text": candidate_text
                })
                count += 1
                if count >= 10:  # 只保留前10个候选
                    break

    return result


def convert_parquet_to_json(input_file, output_file, max_rows=None):
    """将Parquet文件转换为JSON格式"""
    print(f"正在读取文件: {input_file}")
    df = pd.read_parquet(input_file)

    if max_rows:
        df = df.head(max_rows)

    print(f"共读取 {len(df)} 行数据")

    results = []

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"处理第 {idx + 1}/{len(df)} 行...")

        # 提取document文本
        document_text = extract_document_text(row.get('document', {}))

        # 提取并清理数据
        record = {
            "id": str(row.get('id', '')),
            "question": extract_question_text(row.get('question', {})),
            "document": document_text,
            "long_answer_candidates": extract_long_answer_candidates(row.get('long_answer_candidates', {}), document_text),
            "annotations": extract_annotations(row.get('annotations', {}), document_text)
        }

        results.append(record)

    # 保存为JSON
    print(f"正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f"转换完成！共保存 {len(results)} 条记录")


if __name__ == "__main__":
    # 输入文件路径
    # input_file = r'i:\毕业论文最新版\Code\chapter3\datasets\wikipedia\natural_questions\validation-00000-of-00007.parquet'
    input_file = r'I:\毕业论文最新版\Code\chapter3\datasets\natural_questions\validation-00006-of-00007.parquet'

    # 输出文件路径
    output_file = r'I:\毕业论文最新版\Code\chapter3\datasets\natural_questions\validation-00006-of-00007.json'

    # 转换（可以设置max_rows限制行数，例如max_rows=100）
    convert_parquet_to_json(input_file, output_file, max_rows=None)
