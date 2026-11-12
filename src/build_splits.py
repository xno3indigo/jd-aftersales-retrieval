"""校验测试集标注，并按类别分层、固定种子划分开发集/测试集（各约一半）。"""
import json, random
from collections import defaultdict

kb = {d["id"]: d for d in map(json.loads, open("data/kb.jsonl", encoding="utf-8"))}
qs = [json.loads(l) for l in open("data/testset.jsonl", encoding="utf-8") if l.strip()]

errors = []
for q in qs:
    for g in q["gold"]:
        if g["doc"] not in kb:
            errors.append((q["id"], "doc 不存在", g["doc"]))
        elif g["key"] not in kb[g["doc"]]["text"]:
            errors.append((q["id"], "key 不在文档中", g["doc"], g["key"]))
if errors:
    for e in errors: print(*e)
    raise SystemExit("标注有误，先修正")

by_cat = defaultdict(list)
for q in qs: by_cat[q["category"]].append(q)
rng = random.Random(20261023)
for cat, items in by_cat.items():
    items = sorted(items, key=lambda q: q["id"]); rng.shuffle(items)
    half = len(items) // 2
    for i, q in enumerate(items): q["split"] = "dev" if i < half else "test"

with open("data/testset_split.jsonl", "w", encoding="utf-8") as f:
    for q in sorted(qs, key=lambda q: q["id"]): f.write(json.dumps(q, ensure_ascii=False) + "\n")
for cat, items in by_cat.items():
    print(cat, "dev", sum(q["split"] == "dev" for q in items), "test", sum(q["split"] == "test" for q in items))
print("总计", len(qs))
