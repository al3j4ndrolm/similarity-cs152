import argparse
import csv
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from similarity.data import FLOWER_NAMES, flower_prompts, load_flowers, load_news
from similarity.encoders import ClipEncoder, ColorHistogramEncoder, TextEncoder, TfidfEncoder, open_image
from similarity.metrics import ap_by_class, evaluate, top1_confusion
from similarity.search import normalize, top_k

KS = (1, 5, 10, 20, 50, 100)
CACHE = Path("cache")
OUT = Path("results")
FIGS = OUT / "figures"

encode_speed = []


def encode_cached(encoder, items, key):
    CACHE.mkdir(exist_ok=True)
    file = CACHE / f"{key}_{encoder.name}.npy"
    if file.exists() and not isinstance(encoder, TfidfEncoder):
        return np.load(file)
    start = time.perf_counter()
    vectors = encoder.encode(items)
    seconds = time.perf_counter() - start
    encode_speed.append({"encoder": encoder.name, "data": key, "items": len(items),
                         "seconds": round(seconds, 2), "items_per_second": round(len(items) / seconds, 1)})
    print(f"  encoded {len(items)} {key} with {encoder.name} in {seconds:.1f}s")
    if not isinstance(encoder, TfidfEncoder):
        np.save(file, vectors)
    return vectors


def run_image_experiments(encoders, quick):
    print("Flowers-102: image to image")
    db_paths, db_labels = load_flowers(split="database")
    q_paths, q_labels = load_flowers(split="test")
    if quick:
        q_paths, q_labels = q_paths[::20], q_labels[::20]
    rows, details = [], {}
    for encoder in encoders:
        db = normalize(encode_cached(encoder, db_paths, "flowers_db"))
        q = normalize(encode_cached(encoder, q_paths, "flowers_test" + ("_quick" if quick else "")))
        result, ap = evaluate(q, q_labels, db, db_labels, KS)
        rows.append({"task": "image->image (Flowers-102)", "method": encoder.name, **result})
        details[encoder.name] = {"db": db, "q": q, "ap": ap}
        print(f"  {encoder.name}: mAP={result['mAP']:.3f} P@10={result['P@10']:.3f}")
    return rows, details, (db_paths, db_labels, q_paths, q_labels)


def run_text_to_image(clip_encoders, image_details, image_data):
    print("Flowers-102: text to image")
    db_paths, db_labels, q_paths, q_labels = image_data
    paths = list(db_paths) + list(q_paths)
    labels = np.concatenate([db_labels, q_labels])
    prompts = flower_prompts()
    prompt_labels = np.arange(len(prompts))
    rows, details = [], {}
    for encoder in clip_encoders:
        images = np.concatenate([image_details[encoder.name]["db"], image_details[encoder.name]["q"]])
        texts = normalize(encoder.encode(prompts))
        result, ap = evaluate(texts, prompt_labels, images, labels, KS)
        rows.append({"task": "text->image (Flowers-102)", "method": encoder.name, **result})
        details[encoder.name] = {"images": images, "texts": texts}
        print(f"  {encoder.name}: mAP={result['mAP']:.3f} P@10={result['P@10']:.3f}")
    return rows, details, (paths, labels)


def run_text_experiments(encoders, quick):
    print("20 Newsgroups: text to text")
    db_texts, db_labels, names = load_news("train")
    q_texts, q_labels, _ = load_news("test")
    if quick:
        q_texts, q_labels = q_texts[::20], q_labels[::20]
    rows, details = [], {}
    for encoder in encoders:
        if encoder == "tfidf":
            encoder = TfidfEncoder(db_texts)
        db = normalize(encode_cached(encoder, db_texts, "news_train"))
        q = normalize(encode_cached(encoder, q_texts, "news_test" + ("_quick" if quick else "")))
        result, ap = evaluate(q, q_labels, db, db_labels, KS)
        rows.append({"task": "text->text (20 Newsgroups)", "method": encoder.name, **result})
        details[encoder.name] = {"db": db, "q": q, "ap": ap, "encoder": encoder}
        print(f"  {encoder.name}: mAP={result['mAP']:.3f} P@10={result['P@10']:.3f}")
    return rows, details, (db_texts, db_labels, q_texts, q_labels, names)


def numpy_latency(db, queries, k=10):
    start = time.perf_counter()
    for q in queries:
        top_k(db, q[None], k)
    return (time.perf_counter() - start) / len(queries) * 1000


def faiss_latency(db, queries, k=10):
    try:
        import faiss
    except ImportError:
        return None
    db = np.ascontiguousarray(db, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    flat = faiss.IndexFlatIP(db.shape[1])
    flat.add(db)
    start = time.perf_counter()
    for q in queries:
        flat.search(q[None], k)
    flat_ms = (time.perf_counter() - start) / len(queries) * 1000

    hnsw = faiss.IndexHNSWFlat(db.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efSearch = 64
    start = time.perf_counter()
    hnsw.add(db)
    build_s = time.perf_counter() - start
    start = time.perf_counter()
    for q in queries:
        hnsw.search(q[None], k)
    hnsw_ms = (time.perf_counter() - start) / len(queries) * 1000

    _, exact = flat.search(queries, k)
    _, approx = hnsw.search(queries, k)
    overlap = np.mean([len(set(a) & set(b)) / k for a, b in zip(exact, approx)])
    return {"faiss_flat_ms": round(flat_ms, 4), "faiss_hnsw_ms": round(hnsw_ms, 4),
            "hnsw_build_s": round(build_s, 2), "hnsw_overlap_at_10": round(float(overlap), 4)}


def run_speed_tests(real_indexes):
    print("Search speed")
    rows = []
    rng = np.random.default_rng(0)
    tests = list(real_indexes)
    for size in (10_000, 50_000, 200_000):
        db = normalize(rng.normal(size=(size, 512)).astype(np.float32))
        q = normalize(rng.normal(size=(200, 512)).astype(np.float32))
        tests.append((f"synthetic {size:,} x 512", db, q))
    for name, db, q in tests:
        q = q[:200]
        row = {"index": name, "size": db.shape[0], "dims": db.shape[1],
               "numpy_ms": round(numpy_latency(db, q), 4)}
        extra = faiss_latency(db, q)
        if extra:
            row.update(extra)
        rows.append(row)
        print("  ", row)
    return rows


def save_csv(rows, file):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_pk(rows, file):
    tasks = ["image->image (Flowers-102)", "text->text (20 Newsgroups)", "text->image (Flowers-102)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, task in zip(axes, tasks):
        for row in rows:
            if row["task"] == task:
                ax.plot(KS, [row[f"P@{k}"] for k in KS], marker="o", label=row["method"])
        ax.set_xscale("log")
        ax.set_xticks(KS)
        ax.set_xticklabels(KS)
        ax.set_ylim(0, 1)
        ax.set_xlabel("k")
        ax.set_ylabel("Precision@k")
        ax.set_title(task)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(file, dpi=200)
    plt.close()


def plot_confusion(matrix, names, file):
    rates = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("topic of the closest post found")
    ax.set_ylabel("topic of the query post")
    fig.colorbar(image, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(file, dpi=200)
    plt.close()


def plot_topic_ap(main_rows, base_rows, main_name, base_name, file):
    names = [row[0] for row in main_rows]
    base = dict((row[0], row[1]) for row in base_rows)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(y - 0.2, [base[n] for n in names], height=0.4, label=base_name, color="#bbbbbb")
    ax.barh(y + 0.2, [row[1] for row in main_rows], height=0.4, label=main_name, color="#2f6db0")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("average precision")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(file, dpi=200)
    plt.close()


def plot_flower_classes(rows, file):
    picked = rows[:10] + rows[-10:]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#c0504d"] * 10 + ["#2f6db0"] * 10
    ax.barh(range(20), [row[1] for row in picked], color=colors)
    ax.set_yticks(range(20))
    ax.set_yticklabels([row[0] for row in picked], fontsize=8)
    ax.set_xlabel("average precision")
    ax.set_xlim(0, 1)
    ax.set_title("10 hardest and 10 easiest flower classes")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(file, dpi=200)
    plt.close()


def plot_image_examples(q_vectors, q_paths, q_labels, db_vectors, db_paths, db_labels, picks, file):
    fig, axes = plt.subplots(len(picks), 6, figsize=(14, 2.6 * len(picks)))
    for row, i in enumerate(picks):
        hits, sims = top_k(db_vectors, q_vectors[i][None], 5)
        axes[row, 0].imshow(open_image(q_paths[i]))
        axes[row, 0].set_title("QUERY\n" + FLOWER_NAMES[q_labels[i]], fontsize=9, fontweight="bold")
        for col, (j, s) in enumerate(zip(hits, sims), start=1):
            axes[row, col].imshow(open_image(db_paths[j]))
            mark = "match" if db_labels[j] == q_labels[i] else "miss"
            axes[row, col].set_title(f"{FLOWER_NAMES[db_labels[j]]}\n{s:.2f} ({mark})", fontsize=8,
                                     color="black" if mark == "match" else "#c0504d")
        for ax in axes[row]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(file, dpi=130)
    plt.close()


def plot_text_to_image(text_vectors, images, paths, labels, picks, file):
    fig, axes = plt.subplots(len(picks), 5, figsize=(12, 2.8 * len(picks)))
    for row, c in enumerate(picks):
        hits, sims = top_k(images, text_vectors[c][None], 5)
        for col, (j, s) in enumerate(zip(hits, sims)):
            axes[row, col].imshow(open_image(paths[j]))
            mark = "match" if labels[j] == c else "miss"
            axes[row, col].set_title(f"{FLOWER_NAMES[labels[j]]}\n{s:.2f} ({mark})", fontsize=8,
                                     color="black" if mark == "match" else "#c0504d")
            axes[row, col].axis("off")
        axes[row, 0].text(-0.1, 0.5, f'"{FLOWER_NAMES[c]}"', transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(file, dpi=130, bbox_inches="tight")
    plt.close()


def plot_latency(rows, file):
    synthetic = [row for row in rows if row["index"].startswith("synthetic")]
    sizes = [row["size"] for row in synthetic]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, [row["numpy_ms"] for row in synthetic], marker="o", label="NumPy exact")
    if "faiss_flat_ms" in synthetic[0]:
        ax.plot(sizes, [row["faiss_flat_ms"] for row in synthetic], marker="o", label="FAISS flat (exact)")
        ax.plot(sizes, [row["faiss_hnsw_ms"] for row in synthetic], marker="o", label="FAISS HNSW (approx.)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("vectors in the index")
    ax.set_ylabel("ms per query (k=10)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    plt.savefig(file, dpi=200)
    plt.close()


def text_examples(details, db_texts, db_labels, names, queries):
    examples = []
    for query in queries:
        entry = {"query": query, "results": {}}
        for method, d in details.items():
            hits, sims = top_k(d["db"], normalize(d["encoder"].encode([query])), 3)
            entry["results"][method] = [{"topic": names[db_labels[i]], "score": round(float(s), 3),
                                         "text": " ".join(db_texts[i].split())[:200]} for i, s in zip(hits, sims)]
        examples.append(entry)
    return examples


def environment():
    info = {"python": platform.python_version(), "machine": platform.machine()}
    try:
        import torch
        import transformers
        import sentence_transformers
        info["torch"] = torch.__version__
        info["transformers"] = transformers.__version__
        info["sentence_transformers"] = sentence_transformers.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        pass
    try:
        import faiss
        info["faiss"] = faiss.__version__
    except Exception:
        pass
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-large", action="store_true")
    args = parser.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    clip_b = ClipEncoder("openai/clip-vit-base-patch32")
    clip_encoders = [clip_b]
    text_encoders = ["tfidf", TextEncoder("sentence-transformers/all-MiniLM-L6-v2")]
    if not args.skip_large:
        clip_encoders.append(ClipEncoder("openai/clip-vit-large-patch14"))
        text_encoders.append(TextEncoder("sentence-transformers/all-mpnet-base-v2"))
    text_encoders.append(clip_b)

    image_rows, image_details, image_data = run_image_experiments([ColorHistogramEncoder()] + clip_encoders, args.quick)
    t2i_rows, t2i_details, t2i_data = run_text_to_image(clip_encoders, image_details, image_data)
    text_rows, text_details, text_data = run_text_experiments(text_encoders, args.quick)
    rows = image_rows + text_rows + t2i_rows

    db_paths, db_labels, q_paths, q_labels = image_data
    db_texts, news_db_labels, q_texts, news_q_labels, names = text_data

    main_image = clip_b.name
    main_text = "all-MiniLM-L6-v2"
    flower_classes = ap_by_class(image_details[main_image]["ap"], q_labels, FLOWER_NAMES)
    news_topics = ap_by_class(text_details[main_text]["ap"], news_q_labels, names)
    tfidf_topics = ap_by_class(text_details["tfidf"]["ap"], news_q_labels, names)
    confusion = top1_confusion(text_details[main_text]["q"], news_q_labels,
                               text_details[main_text]["db"], news_db_labels, len(names))

    speed_rows = run_speed_tests([
        ("Flowers-102 database (CLIP B/32)", image_details[main_image]["db"], image_details[main_image]["q"]),
        ("20 Newsgroups train (MiniLM)", text_details[main_text]["db"], text_details[main_text]["q"]),
    ])

    save_csv(rows, OUT / "metrics.csv")
    save_csv(speed_rows, OUT / "search_speed.csv")
    save_csv(encode_speed, OUT / "encode_speed.csv")
    save_csv([{"class": n, "ap": round(a, 4), "queries": c} for n, a, c in flower_classes], OUT / "flowers_ap_by_class.csv")
    save_csv([{"topic": n, "ap_minilm": round(a, 4), "ap_tfidf": round(dict((r[0], r[1]) for r in tfidf_topics)[n], 4),
               "queries": c} for n, a, c in news_topics], OUT / "news_ap_by_topic.csv")
    np.savetxt(OUT / "news_top1_confusion.csv", confusion, fmt="%d", delimiter=",",
               header=",".join(names), comments="")

    plot_pk(rows, FIGS / "precision_at_k.png")
    plot_confusion(confusion, names, FIGS / "news_confusion.png")
    plot_topic_ap(news_topics, tfidf_topics, main_text, "tfidf", FIGS / "news_ap_by_topic.png")
    plot_flower_classes(flower_classes, FIGS / "flowers_ap_by_class.png")
    plot_latency(speed_rows, FIGS / "search_latency.png")
    picks = [0, len(q_paths) // 3, 2 * len(q_paths) // 3]
    plot_image_examples(image_details[main_image]["q"], q_paths, q_labels, image_details[main_image]["db"],
                        db_paths, db_labels, picks, FIGS / "example_image_search.png")
    paths, labels = t2i_data
    plot_text_to_image(t2i_details[main_image]["texts"], t2i_details[main_image]["images"], paths, labels,
                       [5, 53, 73], FIGS / "example_text_to_image.png")

    queries = ["NASA is planning a new mission to send astronauts to Mars",
               "my graphics card driver keeps crashing when I render 3D scenes",
               "should I take antibiotics for a cold",
               "who will win the world series this year"]
    examples = text_examples({k: text_details[k] for k in ("tfidf", main_text)}, db_texts, news_db_labels, names, queries)

    summary = {"metrics": rows, "search_speed": speed_rows, "encode_speed": encode_speed,
               "flowers_hardest": flower_classes[:10], "flowers_easiest": flower_classes[-10:],
               "news_ap_by_topic": news_topics, "text_examples": examples,
               "quick_mode": args.quick, "environment": environment(),
               "total_minutes": round((time.perf_counter() - total_start) / 60, 1)}
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Done in {summary['total_minutes']} min. Results are in {OUT}/")


if __name__ == "__main__":
    main()
