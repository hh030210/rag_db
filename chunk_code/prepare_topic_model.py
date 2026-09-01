#!/usr/bin/env python3
"""为四种切分结果准备统一的 TF-IDF + LSA 主题空间。"""

import argparse
import json
from pathlib import Path

import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--max_features", type=int, default=30000)
    args = parser.parse_args()

    source = Path(args.source)
    lines = [x.strip() for x in source.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(lines) < 2:
        raise ValueError("source has too few non-empty lines")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 3),
        min_df=2,
        max_features=args.max_features,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(lines)
    n_components = min(args.components, max(2, matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=0)
    svd.fit(matrix)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "svd": svd,
            "source": str(source),
            "source_lines": len(lines),
            "feature_count": int(matrix.shape[1]),
            "components": int(n_components),
        },
        output,
        compress=3,
    )
    print(json.dumps({
        "output": str(output),
        "source_lines": len(lines),
        "features": int(matrix.shape[1]),
        "components": int(n_components),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
