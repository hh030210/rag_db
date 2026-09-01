import base64
import requests
import os
import json
import fitz  # PyMuPDF
from PIL import Image
import io
import config
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def image_to_base64(image):
    buffered = io.BytesIO()
    # 统一转换为 RGB 模式后再保存为 PNG
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_vlm_api(base64_image):
    headers = {
        "Authorization": f"Bearer {config.API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": config.MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请识别并提取图中的所有文字内容，保持原有段落结构。直接输出文字，不要任何解释。"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "stream": False
    }
    try:
        response = requests.post(config.API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"VLM 请求异常: {e}"

def parse_non_editable(file_path, file_type):
    """
    使用 PyMuPDF 将 PDF 转为图像，并通过 VLM 提取文字
    """
    page_images = []
    
    if file_type == 'pdf':
        print(f"[日志] 正在利用 PyMuPDF 解析 PDF: {os.path.basename(file_path)}")
        try:
            doc = fitz.open(file_path)
            for page in doc:
                # 设置缩放倍数，提升清晰度 (2.0 = 200%)
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_images.append(img)
            doc.close()
        except Exception as e:
            return f"PyMuPDF 转换失败: {e}"
    elif file_type == 'image':
        try:
            page_images = [Image.open(file_path)]
        except Exception as e:
            return f"图片打开失败: {e}"
    
    total_pages = len(page_images)
    print(f"[日志] 共计 {total_pages} 页，开始并行调用 VLM 进行增强识别...")
    
    full_text_list = [None] * total_pages
    page_data_log = {}
    log_lock = threading.Lock()
    
    # 准备 JSON 日志路径
    json_log_path = os.path.splitext(file_path)[0] + "_vlm_raw.json"
    
    def process_page(i, img):
        b64 = image_to_base64(img)
        text = call_vlm_api(b64)
        
        # 清理占位符的逻辑
        import re
        text_cleaned = re.sub(r'<\|LOC_\d+\|>', '', text)
        
        with log_lock:
            page_data_log[f"page_{i+1}"] = text
            page_data_log[f"page_{i+1}_cleaned"] = text_cleaned
            # 实时更新 JSON 日志，防止程序中途崩溃
            with open(json_log_path, 'w', encoding='utf-8') as f:
                json.dump(page_data_log, f, ensure_ascii=False, indent=4)
        
        return i, text

    max_workers = getattr(config, 'API_MAX_WORKERS', 4)
    print(f"[日志] VLM 并行进程数: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {executor.submit(process_page, i, img): i for i, img in enumerate(page_images)}
        
        with tqdm(total=total_pages, desc="VLM 提取进度") as pbar:
            for future in as_completed(future_to_page):
                try:
                    index, text = future.result()
                    full_text_list[index] = text
                except Exception as e:
                    print(f"\n[错误] 第 {future_to_page[future]+1} 页提取异常: {e}")
                    idx = future_to_page[future]
                    full_text_list[idx] = f"Error processing page {idx+1}"
                pbar.update(1)
    
    print(f"[日志] VLM 原始提取结果已保存至: {json_log_path}")
    return "\n\n--- Page Break ---\n\n".join(full_text_list)
