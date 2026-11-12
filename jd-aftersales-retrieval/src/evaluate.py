"""评测脚本：对每个检索版本计算命中率、MRR、答案片段命中率和拒答效果。

用法：
    python src/evaluate.py --split dev  --versions v1 v2      # 开发集：调参、看 bad case
    python src/evaluate.py --split test --versions v1 v2 v3   # 测试集：最终报告，只跑一次

拒答阈值只在开发集上选定，测试集直接沿用开发集的阈值。
"""
import argparse
import json
import os
import time

import numpy as np

from retrieval import BM25, TfidfChar, chunk_docs, expand, load_kb, load_synonyms, rrf, strip_stop, tokenize

# 每个版本的配置。迭代过程记录在 docs/iteration_log.md
VERSIONS = {
    "v1": dict(desc="基线：BM25（字符二元组），300 字切块", chunk=300, title=False, retriever="bm25", stop=False, syn=None),
    "tfidf": dict(desc="对照：TF-IDF（字符 1-3 元组），300 字切块", chunk=300, title=False, retriever="tfidf", stop=False, syn=None),
    "v2": dict(desc="v1 + 去除疑问语气词", chunk=300, title=False, retriever="bm25", stop=True, syn=None),
    "v3": dict(desc="v2 + 口语同义词扩展", chunk=300, title=False, retriever="bm25", stop=True, syn="data/synonyms.txt"),
    "v4_200": dict(desc="v3 + 块前拼接标题，200 字切块", chunk=200, title=True, retriever="bm25", stop=True, syn="data/synonyms.txt"),
    "v4": dict(desc="v3 + 块前拼接标题，300 字切块", chunk=300, title=True, retriever="bm25", stop=True, syn="data/synonyms.txt"),
    "v4_500": dict(desc="v3 + 块前拼接标题，500 字切块", chunk=500, title=True, retriever="bm25", stop=True, syn="data/synonyms.txt"),
    "v5": dict(desc="v4 + BM25 与 TF-IDF 排名融合（RRF）", chunk=300, title=True, retriever="hybrid", stop=True, syn="data/synonyms.txt"),
    "v6": dict(desc="v5 + 拒答改用查询词覆盖率", chunk=300, title=True, retriever="hybrid", stop=True, syn="data/synonyms.txt", conf="coverage"),
}

NO_ANSWER = "无答案"


def load_questions(split):
    qs = [json.loads(l) for l in open("data/testset_split.jsonl", encoding="utf-8") if l.strip()]
    return [q for q in qs if q["split"] == split]


class System:
    def __init__(self, cfg, docs):
        self.cfg = cfg
        self.chunks = chunk_docs(docs, max_chars=cfg["chunk"], prepend_title=cfg["title"])
        texts = [c["text"] for c in self.chunks]
        self.bm25 = BM25(texts)
        self.tfidf = TfidfChar(texts) if cfg["retriever"] in ("tfidf", "hybrid") else None
        self.syn = load_synonyms(cfg["syn"]) if cfg["syn"] else []

    def rewrite(self, q):
        if self.syn:
            q = expand(q, self.syn)
        if self.cfg["stop"]:
            q = strip_stop(q)
        return q

    def search(self, question):
        q = self.rewrite(question)
        bm = self.bm25.scores(q)
        if self.cfg["retriever"] == "bm25":
            final = bm
        elif self.cfg["retriever"] == "tfidf":
            final = self.tfidf.scores(q)
        else:
            final = rrf([bm, self.tfidf.scores(q)])
        order = np.argsort(-final)
        top = order[0]
        if self.cfg.get("conf") == "coverage":
            conf = self.coverage(q, [self.chunks[i]["text"] for i in order[:3]])
        elif self.cfg["retriever"] == "tfidf":
            conf = float(final[top])
        else:
            conf = float(bm[top] / self.bm25.max_score(q))
        return [self.chunks[i] for i in order[:20]], conf, q

    def coverage(self, q, top_texts):
        """拒答置信度：查询词（按 IDF 加权）在前 3 个结果中出现的比例。"""
        terms = set(tokenize(q))
        found = set(tokenize("\n".join(top_texts)))
        w = {t: self.bm25.idf.get(t, max(self.bm25.idf.values())) for t in terms}
        total = sum(w.values()) or 1.0
        return sum(v for t, v in w.items() if t in found) / total


def evaluate(name, cfg, docs, split, threshold=None):
    sysm = System(cfg, docs)
    qs = load_questions(split)
    rows = []
    t0 = time.perf_counter()
    for q in qs:
        ranked, conf, rq = sysm.search(q["question"])
        gold_docs = {g["doc"] for g in q["gold"]}
        doc_rank, seen = None, []
        for c in ranked:
            if c["doc"] not in seen:
                seen.append(c["doc"])
                if c["doc"] in gold_docs and doc_rank is None:
                    doc_rank = len(seen)
        span3 = any(c["doc"] == g["doc"] and g["key"] in c["raw"] for c in ranked[:3] for g in q["gold"])
        rows.append(dict(id=q["id"], category=q["category"], question=q["question"], rewritten=rq,
                         doc_rank=doc_rank, span3=span3, conf=conf,
                         top3=[f"{c['doc']} {c['title']}" for c in ranked[:3]],
                         gold=[g["doc"] for g in q["gold"]]))
    latency = (time.perf_counter() - t0) / len(qs) * 1000

    ans = [r for r in rows if r["category"] != NO_ANSWER]
    neg = [r for r in rows if r["category"] == NO_ANSWER]
    if threshold is None:  # 只在开发集上选阈值：让“可回答题不误拒”和“无答案题被拒”两者平均最高
        cands = sorted({round(r["conf"], 4) for r in rows})
        best = max(cands, key=lambda t: (np.mean([r["conf"] >= t for r in ans]) + np.mean([r["conf"] < t for r in neg])) / 2)
        threshold = best

    def hit(k, subset):
        return np.mean([r["doc_rank"] is not None and r["doc_rank"] <= k for r in subset]) if subset else float("nan")

    metrics = {
        "版本": name, "说明": cfg["desc"], "题数": len(rows),
        "Hit@1": hit(1, ans), "Hit@3": hit(3, ans), "Hit@5": hit(5, ans),
        "MRR@10": np.mean([1 / r["doc_rank"] if r["doc_rank"] and r["doc_rank"] <= 10 else 0 for r in ans]),
        "答案片段命中@3": np.mean([r["span3"] for r in ans]),
        "拒答阈值": threshold,
        "拒答正确率": np.mean([r["conf"] < threshold for r in neg]),
        "误拒率": np.mean([r["conf"] < threshold for r in ans]),
        "平均耗时ms": latency,
    }
    for cat in ["直接查询", "口语化提问", "细节条款"]:
        metrics[f"Hit@3·{cat}"] = hit(3, [r for r in ans if r["category"] == cat])
    return metrics, rows


RAW_KEYS = {"拒答阈值", "平均耗时ms"}


def fmt(v, key=""):
    if isinstance(v, float) and key not in RAW_KEYS:
        return f"{v * 100:.1f}%"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--versions", nargs="+", default=list(VERSIONS))
    args = ap.parse_args()
    docs = load_kb()
    os.makedirs("results", exist_ok=True)
    thr_path = "results/thresholds_dev.json"
    thresholds = json.load(open(thr_path)) if os.path.exists(thr_path) else {}

    all_metrics = []
    for v in args.versions:
        thr = thresholds.get(v) if args.split == "test" else None
        if args.split == "test" and thr is None:
            raise SystemExit(f"{v} 还没有开发集阈值，请先跑 --split dev")
        m, rows = evaluate(v, VERSIONS[v], docs, args.split, thr)
        if args.split == "dev":
            thresholds[v] = m["拒答阈值"]
        all_metrics.append(m)
        with open(f"results/{args.split}_{v}_details.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.split == "dev":
        json.dump(thresholds, open(thr_path, "w"), indent=1)

    keys = list(all_metrics[0].keys())
    with open(f"results/{args.split}_metrics.md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n")
        for m in all_metrics:
            f.write("| " + " | ".join(fmt(m[k], k) for k in keys) + " |\n")
    for m in all_metrics:
        print(m["版本"], "｜", "  ".join(f"{k}={fmt(m[k], k)}" for k in keys if k not in ("版本", "说明")))


if __name__ == "__main__":
    main()
