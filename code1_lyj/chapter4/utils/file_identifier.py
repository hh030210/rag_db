# 尝试导入filetype模块，失败时设置为None
try:
    import filetype
    has_filetype = True
except ImportError:
    has_filetype = False
    print("[警告] 缺少filetype模块，将使用文件扩展名判断文件类型")

# 尝试导入pdfplumber模块，失败时设置为None
try:
    import pdfplumber
    has_pdfplumber = True
except ImportError:
    has_pdfplumber = False
    print("[警告] 缺少pdfplumber模块，无法检查PDF文件是否可编辑")

def identify_file_type(file_path):
    if has_filetype:
        kind = filetype.guess(file_path)
        if kind is None:
            ext = file_path.split('.')[-1].lower()
        else:
            ext = kind.extension.lower()
            print(f"使用filetype库识别MIME类型：{ext},filetype库输出：{kind}")
    else:
        # 使用文件扩展名判断文件类型
        ext = file_path.split('.')[-1].lower()
        print(f"使用文件扩展名识别文件类型：{ext}")
    
    if ext == 'pdf':
        is_editable = check_pdf_editable(file_path)
        return 'pdf', is_editable
    elif ext in ['docx', 'doc']:
        return 'docx', True
    elif ext in ['pptx', 'ppt']:
        return 'pptx', True
    elif ext in ['xlsx', 'xls']:
        return 'xlsx', True
    elif ext in ['jpg', 'png', 'bmp', 'tiff']:
        return 'image', False
    
    return ext, False

def check_pdf_editable(file_path):
    if not has_pdfplumber:
        print("[警告] 缺少pdfplumber模块，默认PDF文件为不可编辑")
        return False
    try:
        with pdfplumber.open(file_path) as pdf:
            for i in range(min(5, len(pdf.pages))):
                text = pdf.pages[i].extract_text()
                if text and len(text.strip()) > 10:
                    return True
    except Exception:
        pass
    return False
