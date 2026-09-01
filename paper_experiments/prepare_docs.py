#!/usr/bin/env python3
"""Convert a chunk JSON list into one-line documents for the existing retriever."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def load_chunks(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("chunks") or data.get("data") or []
    texts: list[str] = []
    for item in data:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = item.get("chunk_text") or item.get("text") or ""
        else:
            text = ""
        text = " ".join(str(text).replace("\r", " ").replace("\n", " ").split())
        if text:
            texts.append(text)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_json", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    source = Path(args.chunks_json).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    marker = output / ".prepared.json"
    if marker.exists():
        print(json.dumps(json.loads(marker.read_text(encoding="utf-8")), ensure_ascii=False))
        return

    texts = load_chunks(source)
    if not texts:
        raise ValueError(f"没有从 {source} 读取到有效分片")

    tmp = output.with_name(output.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for index, text in enumerate(texts):
        (tmp / f"chunk_{index:06d}.txt").write_text(text, encoding="utf-8")
    summary = {
        "source": str(source),
        "output_dir": str(output),
        "documents": len(texts),
        "characters": sum(len(text) for text in texts),
    }
    (tmp / ".prepared.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if output.exists():
        shutil.rmtree(output)
    os.replace(tmp, output)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
