def fix_bom(file_path):
    """移除文件开头的BOM字符"""
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # 移除BOM字符
    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]
        print(f"已移除文件 {file_path} 开头的BOM字符")
    
    # 写回文件
    with open(file_path, 'wb') as f:
        f.write(content)
    
    print(f"文件 {file_path} 已修复")

if __name__ == "__main__":
    file_path = "utils/editable_parser.py"
    fix_bom(file_path)
