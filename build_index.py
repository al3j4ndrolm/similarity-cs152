import argparse
import json
from pathlib import Path

import numpy as np

from similarity.data import FLOWER_NAMES, load_flowers, load_news
from similarity.encoders import ClipEncoder, TextEncoder

INDEX_DIR = Path("index")


def build_flowers(limit):
    paths, labels = load_flowers(split="all")
    if limit:
        paths, labels = paths[:limit], labels[:limit]
    vectors = ClipEncoder().encode(paths)
    items = [{"label": FLOWER_NAMES[label], "path": str(path.resolve())} for path, label in zip(paths, labels)]
    np.save(INDEX_DIR / "flowers.npy", vectors)
    (INDEX_DIR / "flowers.json").write_text(json.dumps(items))
    print(f"flowers: {len(items)} images")


def build_news(limit):
    texts, labels, names = load_news("train")
    if limit:
        texts, labels = texts[:limit], labels[:limit]
    vectors = TextEncoder().encode(texts)
    items = [{"label": names[label], "text": text} for text, label in zip(texts, labels)]
    np.save(INDEX_DIR / "news.npy", vectors)
    (INDEX_DIR / "news.json").write_text(json.dumps(items))
    print(f"news: {len(items)} posts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    INDEX_DIR.mkdir(exist_ok=True)
    build_flowers(args.limit)
    build_news(args.limit)
