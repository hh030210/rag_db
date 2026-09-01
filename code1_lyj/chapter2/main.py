import os
import sys
import json
from datetime import datetime
from tqdm import tqdm
from utils.file_identifier import identify_file_type
from utils.editable_parser import parse_editable
from utils.vlm_parser import parse_non_editable
from utils.llm_denoiser import rule_based_clean, LLMDenoiser

# 转换日志文件路径
CONVERSION_LOG_FILE = "conversion_log.txt"

def log_conversion(file_path):
    """记录文件转换信息到日志文件"""
    try:
        # 获取当前时间
        conversion_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 获取文件名
        file_name = os.path.basename(file_path)
        # 确保日志文件目录存在
        log_dir = os.path.dirname(CONVERSION_LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # 追加写入日志
        with open(CONVERSION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"转换时间: {conversion_time}, 文件名: {file_name}, 路径: {file_path}\n")
        print(f"[日志] 转换记录已写入: {CONVERSION_LOG_FILE}")
    except Exception as e:
        print(f"[错误] 写入转换日志失败: {e}")

def process_file(file_path):
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    # 检查final.txt文件是否已存在，如果存在则跳过处理
    final_output_path = os.path.splitext(file_path)[0] + "_final.txt"
    if os.path.exists(final_output_path):
        print(f"[跳过] 文件已处理，final文件已存在: {final_output_path}")
        return

    # 记录文件转换信息
    log_conversion(file_path)

    print(f"\n{'='*20} 正在分析文件: {os.path.basename(file_path)} {'='*20}")
    file_type, is_editable = identify_file_type(file_path)
    print(f"文件类型: {file_type}, 可编辑: {is_editable}")

    if is_editable:
        raw_content = parse_editable(file_path, file_type)
    else:
        raw_content = parse_non_editable(file_path, file_type)

    if not raw_content:
        print("未能提取到有效内容。")
        return

    # 按照需求：在处理时去掉占位符，保存为 txt 时不包含占位符
    # 但保留原始 JSON (在 vlm_parser.py 中已保存) 以保留"带格式的内容"
    import re
    raw_content_cleaned = re.sub(r'<\|LOC_\d+\|>', '', raw_content)

    # 保存原始提取出的文本 (已去占位符)
    raw_output_path = os.path.splitext(file_path)[0] + "_raw.txt"
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        f.write(raw_content_cleaned)
    print(f"原始提取文本 (已去占位符) 已保存: {raw_output_path}")

    # --- 流程 (1): 仅使用规则进行处理 ---
    print(f"\n[任务] 执行流程(1): 仅规则去噪...")
    rule_only_text = rule_based_clean(raw_content_cleaned)
    rule_output_path = os.path.splitext(file_path)[0] + "_rule_only.txt"
    with open(rule_output_path, 'w', encoding='utf-8') as f:
        f.write(rule_only_text)
    print(f"规则去噪结果已保存: {rule_output_path}")

    # --- 流程 (2): 仅使用大模型进行处理 ---
    print(f"\n[任务] 执行流程(2): 纯大模型语义去噪...")
    # 生成基于文件名和时间的日志文件名
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{base_name}_llm_interaction_{timestamp}.log"
    # 生成基于文件名和时间的organize日志文件名
    organize_log_file = f"{base_name}_organize_interaction_{timestamp}.log"
    denoiser = LLMDenoiser(log_file=log_file, organize_log_file=organize_log_file)
    
    # 第一步: 提取噪声特征 (基于基础清理后的文本)
    noise_info = denoiser.extract_noise_types(raw_content_cleaned)
    
    if noise_info:
        # 第二步: 执行语义去噪 (基于基础清理后的文本)
        denoise_result = denoiser.denoise_text(raw_content_cleaned, noise_info, file_type)
        
        if denoise_result:
            llm_only_text = denoise_result.get("去噪后文本", "")
            noise_content = denoise_result.get("噪声内容", [])
            
            llm_output_path = os.path.splitext(file_path)[0] + "_llm_only.txt"
            with open(llm_output_path, 'w', encoding='utf-8') as f:
                f.write(llm_only_text)
            
            # 保存大模型提取的特征日志和识别出的噪声内容
            log_path = os.path.splitext(file_path)[0] + "_llm_denoise_log.json"
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "extracted_features": noise_info,
                    "identified_noise_fragments": noise_content
                }, f, ensure_ascii=False, indent=4)
            
            print(f"大模型去噪结果已保存: {llm_output_path}")
            print(f"大模型处理日志已保存: {log_path}")
            
            # 第三步: 内容重组和归纳 (基于LLM去噪后的文本)
            print(f"\n[任务] 执行流程(3): 内容重组和归纳...")
            
            # 添加进度条显示
            with tqdm(total=1, desc="内容重组和归纳") as pbar:
                paragraphs, responses = denoiser.organize_content(llm_only_text, file_type)
                pbar.update(1)
            
            # 保存最终重组文本为段落格式
            final_output_path = os.path.splitext(file_path)[0] + "_final.txt"
            with open(final_output_path, 'w', encoding='utf-8') as f:
                # 每个段落之间用空行分隔
                f.write('\n'.join(paragraphs))
            
            # 保存内容重组的JSON日志文件
            organize_log_path = os.path.splitext(file_path)[0] + "_organize_log.json"
            with open(organize_log_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "重组后段落": paragraphs,
                    "原始响应": responses
                }, f, ensure_ascii=False, indent=4)
            
            print(f"最终重组文本已保存: {final_output_path}")
            print(f"内容重组日志已保存: {organize_log_path}")
        else:
            print("大模型语义去噪执行失败。")
    else:
        print("大模型噪声特征提取失败。")

    print(f"\n{'='*20} 文件处理完成 {'='*20}")

def process_folder(folder_path):
    """处理文件夹中的PDF、PPT、Excel和Word文件"""
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return
    
    print(f"\n{'='*40} 开始处理文件夹: {folder_path} {'='*40}")
    
    # 支持的文件扩展名
    supported_extensions = [
        '.pdf', '.ppt', '.pptx',  # PDF和PPT文件
        '.xls', '.xlsx',           # Excel文件
        '.doc', '.docx'            # Word文件
    ]
    
    # 收集所有需要处理的文件
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件扩展名
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in supported_extensions:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    
    # 使用进度条显示文件处理进度
    total_files = len(all_files)
    print(f"[日志] 找到 {total_files} 个需要处理的文件")
    
    with tqdm(total=total_files, desc="文件处理进度") as pbar:
        for i, file_path in enumerate(all_files):
            process_file(file_path)
            pbar.update(1)
    
    print(f"\n{'='*40} 文件夹处理完成 {'='*40}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python main.py <文件路径或文件夹路径>")
    else:
        path = sys.argv[1]
        if os.path.isfile(path):
            process_file(path)
        elif os.path.isdir(path):
            process_folder(path)
        else:
            print(f"路径不存在: {path}")
