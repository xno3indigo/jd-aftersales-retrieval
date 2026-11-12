# 电商售后知识检索：评测与迭代

用京东帮助中心公开的售后规则搭建一个检索系统，回答“能不能退、运费谁出、退款多久到”这类售后问题，并用自建测试集评测检索效果、归因失败案例、迭代优化。

只评测检索环节，不包含大模型生成。完整过程和结论见 `docs/iteration_log.md`。

## 运行

只需要 Python 3、numpy、scikit-learn，不调用任何外部服务。

仓库不包含知识库原文。运行前先生成 `data/kb.jsonl`：在浏览器打开 https://help.jd.com/user/issue.html ，在开发者工具控制台运行 `scripts/fetch_kb_browser.js`，把下载的 `kb.jsonl` 放到 `data/` 目录。
帮助中心页面会更新，重新抓取的内容可能和评测时（2026-10-23）不完全一致，指标也可能略有出入；`results/` 中保存的是评测时的原始结果。

```bash
pip install numpy scikit-learn
python src/build_splits.py                         # 校验标注，划分开发集/测试集
python src/evaluate.py --split dev  --versions v1 v2 v3 v4 v5 v6    # 开发集：调参、看 bad case，并保存拒答阈值
python src/evaluate.py --split test --versions v1 v2 v3 v4 v5 v6    # 测试集：沿用开发集阈值，最终报告
```

结果写在 `results/`：`*_metrics.md` 是指标表，`*_details.jsonl` 是每道题的检索结果、排名和改写后的查询。

## 目录

| 路径 | 内容 |
|---|---|
| `data/kb.jsonl` | 知识库：129 篇规则原文，含来源链接（不随仓库提供，由 `scripts/fetch_kb_browser.js` 生成） |
| `scripts/fetch_kb_browser.js` | 知识库抓取脚本，在浏览器控制台运行 |
| `data/testset.jsonl` | 测试集原始标注：109 题，每题标注正确文档和答案原文片段 |
| `data/testset_split.jsonl` | 划分后的测试集（开发集 54 题、测试集 55 题） |
| `data/synonyms.txt` | 口语同义词表，每条标注了来源题号 |
| `src/retrieval.py` | 切分、分词、BM25、TF-IDF、查询改写、排名融合 |
| `src/evaluate.py` | 各版本配置和评测指标 |
| `docs/iteration_log.md` | 迭代记录、测试集结果、失败归因和结论 |

## 核心结果（留出测试集）

| | 基线 v1 | 最终 v5 |
|---|---|---|
| Hit@3 | 80.0% | 80.0% |
| 答案片段命中@3 | 73.3% | 77.8% |
| 口语题 Hit@3 | 55.0% | 55.0% |
| 误拒率 | 46.7% | 28.9% |

开发集上 Hit@3 从 68.2% 提升到 95.5%，但测试集持平，说明人工同义词表对开发集过拟合。详见迭代记录。

## 数据说明

知识库内容来自京东帮助中心公开页面（help.jd.com，2026-10-23 抓取），仅用于学习和评测，因此本仓库不包含原文，只提供抓取脚本。
测试集中的答案片段是从对应规则页面摘取的简短原文，用于自动判断检索是否命中。
