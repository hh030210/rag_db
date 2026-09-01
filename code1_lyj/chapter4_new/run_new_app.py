#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动Tourist RAG演示系统
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from frontend.app_new import create_app

if __name__ == "__main__":
    import torch
    print("正在启动Tourist RAG演示系统...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7863, share=False)
