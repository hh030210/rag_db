@echo off
chcp 65001 >nul
cd /d D:\RAG_DB_slim
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=6
python pipeline_qdrant.py --step 6 > step6_out.txt 2>&1
