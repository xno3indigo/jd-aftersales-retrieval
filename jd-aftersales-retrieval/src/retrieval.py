"""检索核心：文本切分、分词、BM25 / TF-IDF 检索、查询改写、结果融合。

所有方法都只依赖 numpy / scikit-learn，不调用任何外部模型或服务。
"""
import json
import math
import re
import unicodedata
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------- 文本处理

def normalize(text: str) -> str:
    """全角转半角、英文小写。"""
    return unicodedata.normalize("NFKC", text).lower()


_CJK = r"一-鿿"
_TOKEN_RE = re.compile(rf"[{_CJK}]+|[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    """中文按字符二元组切分，英文和数字保留整词（如 211、plus、windows）。"""
    tokens = []
    for seg in _TOKEN_RE.findall(normalize(text)):
        if re.match(rf"[{_CJK}]", seg):
            if len(seg) == 1:
                tokens.append(seg)
            else:
                tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
        else:
            tokens.append(seg)
    return tokens


# ---------------------------------------------------------------- 切分

def load_kb(path="data/kb.jsonl"):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def chunk_docs(docs, max_chars=300, overlap_lines=0, prepend_title=False):
    """按行切分，把相邻行合并到不超过 max_chars 的块；超长单行按字符硬切。

    prepend_title=True 时，在每个块前面拼上文档标题，便于标题信息参与匹配。
    """
    chunks = []
    for d in docs:
        lines = [l.strip() for l in d["text"].split("\n") if l.strip() and l.strip() != "|"]
        pieces = []
        for line in lines:
            while len(line) > max_chars:
                pieces.append(line[:max_chars]); line = line[max_chars:]
            pieces.append(line)
        buf = []
        i = 0
        while i < len(pieces):
            buf, size, j = [], 0, i
            while j < len(pieces) and (size + len(pieces[j]) <= max_chars or not buf):
                buf.append(pieces[j]); size += len(pieces[j]); j += 1
            body = "\n".join(buf)
            text = f"{d['title']}\n{body}" if prepend_title and not body.startswith(d["title"]) else body
            chunks.append({"doc": d["id"], "title": d["title"], "text": text, "raw": body})
            if j >= len(pieces):
                break
            i = max(j - overlap_lines, i + 1)
    return chunks


# ---------------------------------------------------------------- 查询改写

# 只去掉疑问语气、代词这类对检索没有帮助的词，不删除任何业务词
STOP_PHRASES = [
    "能不能", "可不可以", "行不行", "是不是", "有没有", "怎么办", "怎么样",
    "可以吗", "还能", "还会", "一般", "现在", "已经", "刚刚",
    "怎么", "怎样", "如何", "什么", "多少", "哪里", "哪些", "为啥", "为什么", "啥",
    "吗", "呢", "吧", "啊", "了", "的", "给我", "我", "你们", "你", "一下", "一个",
]
# 注意：不去掉“会”“能”“要”这类单字，它们会出现在“会员”“功能”“需要”等业务词里


def strip_stop(query: str) -> str:
    q = normalize(query)
    for p in sorted(STOP_PHRASES, key=len, reverse=True):
        q = q.replace(p, " ")
    return q


def load_synonyms(path):
    """同义词表：每行“口语词=规范词1,规范词2”。命中口语词时把规范词追加到查询里。"""
    table = []
    for line in open(path, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if not line or "=" not in line:
            continue
        src, dst = line.split("=", 1)
        table.append((src.strip(), [t.strip() for t in dst.split(",") if t.strip()]))
    return table


def expand(query: str, synonyms) -> str:
    q = normalize(query)
    extra = []
    for src, dsts in synonyms:
        if src in q:
            extra.extend(dsts)
    return query + (" " + " ".join(extra) if extra else "")


# ---------------------------------------------------------------- 检索器

class BM25:
    def __init__(self, texts, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lens = np.array([sum(c.values()) for c in self.docs], dtype=float)
        self.avgdl = self.lens.mean()
        df = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = Counter(tokenize(query))
        s = np.zeros(len(self.docs))
        for t in q:
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i, c in enumerate(self.docs):
                tf = c.get(t)
                if tf:
                    s[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl))
        return s

    def max_score(self, query: str) -> float:
        """查询能拿到的理论最高分，用来把分数归一到 0~1，作为拒答置信度。"""
        q = Counter(tokenize(query))
        return sum(self.idf.get(t, 0) * (self.k1 + 1) for t in q) or 1.0


class TfidfChar:
    def __init__(self, texts):
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True,
                                   preprocessor=lambda s: re.sub(r"\s+", " ", normalize(s)))
        self.mat = self.vec.fit_transform(texts)

    def scores(self, query: str) -> np.ndarray:
        return (self.mat @ self.vec.transform([query]).T).toarray().ravel()


def rrf(score_lists, k=60):
    """Reciprocal Rank Fusion：按各检索器的排名而不是分数融合。"""
    fused = np.zeros(len(score_lists[0]))
    for s in score_lists:
        ranks = np.empty(len(s)); ranks[np.argsort(-s)] = np.arange(1, len(s) + 1)
        fused += 1.0 / (k + ranks)
    return fused
