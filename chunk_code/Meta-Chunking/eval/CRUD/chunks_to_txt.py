"""
将 chunks JSON 转成每个 chunk 一个 .txt 文件，供 CRUD eval 的 BaseRetriever 读取。
- 输入: chunks.json  [{ "chunk_text": str, "chunk_len": int }, ...]
- 输出: out_dir/chunk_000000.txt, chunk_000001.txt, ...

为避免覆盖已有的 data/chunks_txt/，输出到 data/chunks_txt_integrated/ 与 data/chunks_txt_baseline/。
"""
import json
import os
import sys
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

CHUNKS_JSON = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:/Users/胡铭强/Desktop/chunk_code/data/db_qa_chunks/all_chunks_chunks.json"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else \
    r"C:/Users/胡铭强/Desktop/chunk_code/data/chunks_txt_integrated"


def main():
    if not os.path.isfile(CHUNKS_JSON):
        raise FileNotFoundError(f"chunks JSON 不存在: {CHUNKS_JSON}")

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[Step 1] 读取 {CHUNKS_JSON}")
    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[Step 1] 写入 {OUT_DIR}/  共 {len(chunks)} 个 chunk")
    for i, chunk in enumerate(tqdm(chunks, desc="Converting chunks")):
        text = chunk["chunk_text"]
        fname = f"chunk_{i:06d}.txt"
        with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as out:
            out.write(text)

    print(f"[Step 1] 完成! {len(chunks)} 个 .txt 已保存到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
