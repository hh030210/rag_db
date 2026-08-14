"""
景区数据统计脚本
统计tourist_project目录下所有md文件的字数信息
"""
import os
import re
from pathlib import Path
from collections import defaultdict

# 数据目录
DATA_DIR = Path(r"i:\bylw_final\Code\chapter3\datasets\tourist_project")

def count_chinese_chars(text: str) -> int:
    """统计中文字符数"""
    # 匹配所有中文字符
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return len(chinese_chars)

def count_total_chars(text: str) -> int:
    """统计总字符数（不含空白）"""
    return len(text.replace(' ', '').replace('\n', '').replace('\t', ''))

def count_words(text: str) -> int:
    """统计词数（中文按字，英文按词）"""
    # 中文字符数
    chinese_count = count_chinese_chars(text)
    # 英文单词数
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    return chinese_count + english_words

def analyze_file(file_path: Path) -> dict:
    """分析单个文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 统计各种指标
    total_chars = len(content)  # 总字符数（含空格和换行）
    content_chars = count_total_chars(content)  # 内容字符数（不含空白）
    chinese_chars = count_chinese_chars(content)  # 中文字符数
    word_count = count_words(content)  # 词数
    lines = content.count('\n') + 1  # 行数
    
    # 统计景点数量（根据POI标记）
    poi_count = len(re.findall(r'"[^"]+":\s*"[^"]+"\s*\n[-=]+', content))
    
    return {
        'file_name': file_path.name,
        'scenic_area': file_path.parent.name,
        'total_chars': total_chars,
        'content_chars': content_chars,
        'chinese_chars': chinese_chars,
        'word_count': word_count,
        'lines': lines,
        'poi_count': poi_count if poi_count > 0 else 1  # 如果没有POI标记，算作1个
    }

def format_number(num: int) -> str:
    """格式化数字，添加千位分隔符"""
    return f"{num:,}"

def main():
    print("=" * 80)
    print("景区数据统计报告")
    print("=" * 80)
    print(f"数据目录: {DATA_DIR}")
    print("=" * 80)
    
    # 收集所有md文件
    md_files = []
    for scenic_dir in DATA_DIR.iterdir():
        if scenic_dir.is_dir():
            for md_file in scenic_dir.glob("*.md"):
                if "介绍" in md_file.name or "运营" in md_file.name:
                    md_files.append(md_file)
    
    md_files = sorted(md_files)
    
    # 分析每个文件
    all_stats = []
    scenic_area_stats = defaultdict(lambda: {
        'files': 0,
        'total_chars': 0,
        'content_chars': 0,
        'chinese_chars': 0,
        'word_count': 0,
        'lines': 0,
        'poi_count': 0
    })
    
    print("\n【详细统计】\n")
    print(f"{'景区':<12} {'文件名':<25} {'中文字符':>10} {'总字符':>10} {'词数':>10} {'景点数':>8}")
    print("-" * 80)
    
    for md_file in md_files:
        stats = analyze_file(md_file)
        all_stats.append(stats)
        
        # 累加到景区统计
        area = stats['scenic_area']
        scenic_area_stats[area]['files'] += 1
        scenic_area_stats[area]['total_chars'] += stats['total_chars']
        scenic_area_stats[area]['content_chars'] += stats['content_chars']
        scenic_area_stats[area]['chinese_chars'] += stats['chinese_chars']
        scenic_area_stats[area]['word_count'] += stats['word_count']
        scenic_area_stats[area]['lines'] += stats['lines']
        scenic_area_stats[area]['poi_count'] += stats['poi_count']
        
        # 打印单行统计
        print(f"{stats['scenic_area']:<12} {stats['file_name']:<25} "
              f"{format_number(stats['chinese_chars']):>10} "
              f"{format_number(stats['content_chars']):>10} "
              f"{format_number(stats['word_count']):>10} "
              f"{stats['poi_count']:>8}")
    
    # 打印景区汇总
    print("\n" + "=" * 80)
    print("【景区汇总】\n")
    print(f"{'景区':<15} {'文件数':>8} {'中文字符':>12} {'总字符':>12} {'词数':>12} {'景点数':>10}")
    print("-" * 80)
    
    total_stats = {
        'files': 0,
        'total_chars': 0,
        'content_chars': 0,
        'chinese_chars': 0,
        'word_count': 0,
        'lines': 0,
        'poi_count': 0
    }
    
    for area in sorted(scenic_area_stats.keys()):
        s = scenic_area_stats[area]
        print(f"{area:<15} {s['files']:>8} {format_number(s['chinese_chars']):>12} "
              f"{format_number(s['content_chars']):>12} {format_number(s['word_count']):>12} "
              f"{s['poi_count']:>10}")
        
        # 累加总计
        for key in total_stats:
            total_stats[key] += s[key]
    
    # 打印总计
    print("-" * 80)
    print(f"{'总计':<15} {total_stats['files']:>8} {format_number(total_stats['chinese_chars']):>12} "
          f"{format_number(total_stats['content_chars']):>12} {format_number(total_stats['word_count']):>12} "
          f"{total_stats['poi_count']:>10}")
    
    print("\n" + "=" * 80)
    print("【统计说明】")
    print("  - 中文字符: 纯汉字数量")
    print("  - 总字符: 去除空白后的字符总数")
    print("  - 词数: 中文字符 + 英文单词数")
    print("  - 景点数: 根据POI标记统计的景点数量")
    print("=" * 80)

if __name__ == "__main__":
    main()
