# LoCoMo B/B+TF 指标摘要（中文预览版）

结果目录：`out/locomo_b_btf_recheck_20260624`

## 检索延迟

| Method | Avg Parse | Avg Filter | Avg Search No Parse | Avg Total With Parse | p50 Total | p95 Total |
|--------|-----------|------------|---------------------|----------------------|-----------|----------|
| B | 0.000s | 0.000s | 0.105s | 0.105s | 0.104s | 0.147s |
| B+TF | 1.003s | 0.002s | 2.227s | 3.231s | 3.189s | 4.264s |

`Avg Search No Parse` 不包含 LLM query parsing，但仍包含 rewritten-query embedding、vector-cache flush 和 semantic ranking。若对照论文 Table 3，应优先看下面的 paper-like core search。

### Paper-like core search latency

Core search = temporal filter + block vector fetch + cosine scoring + sorting；不包含 LLM parsing、rewritten-query embedding API、vector-cache flush。

#### 全量问题（n=1540）

| Method | Core Mean | Core p50 | Core p95 | Initial Pool Mean | Filtered Pool Mean | Filtered Pool p50 |
|--------|-----------|----------|----------|-------------------|--------------------|-------------------|
| B | 0.1024s | 0.1023s | 0.1438s | 93.36 | 93.36 | 93.0 |
| B+TF | 0.1037s | 0.1078s | 0.1522s | 93.36 | 87.33 | 93.0 |

#### Category-2 时序问题（n=321）

| Method | Core Mean | Core p50 | Core p95 | Initial Pool Mean | Filtered Pool Mean | Filtered Pool p50 |
|--------|-----------|----------|----------|-------------------|--------------------|-------------------|
| B | 0.1010s | 0.1010s | 0.1473s | 91.65 | 91.65 | 93.0 |
| B+TF | 0.1001s | 0.1040s | 0.1507s | 91.65 | 83.88 | 86.0 |

B+TF parser 对 category-2 问题的输出：NONE=281，POINT=36，RANGE=4。

#### Category-2 POINT/RANGE-triggered 子集（n=40）

| Method | Core Mean | Core p50 | Core p95 | Initial Pool Mean | Filtered Pool Mean | Filtered Pool p50 |
|--------|-----------|----------|----------|-------------------|--------------------|-------------------|
| B | 0.1060s | 0.1053s | 0.1449s | 96.00 | 96.00 | 93.0 |
| B+TF | 0.0429s | 0.0085s | 0.1362s | 96.00 | 33.65 | 4.5 |

这个子集上 Time Filtering 有明显的 core-latency 收益，但在全量运行中被大量 `NONE` 解析结果以及在线 query embedding/cache flush 开销稀释。

## B+TF 解析分布

Parse source：

```text
LLM: 1192
FAST: 348
```

Time constraint：

```text
NONE: 1324
RANGE: 90
POINT: 119
BEFORE: 6
AFTER: 1
```

Fallback count：38

## Evidence 指标（top-5）

### 全量问题

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 | Queries with targets |
|--------|----------|--------------|-------|----------|----------------------|
| B | 0.6592 | 0.5608 | 0.8201 | 0.7580 | 1532/1540 |
| B+TF | 0.6066 | 0.5114 | 0.7708 | 0.7065 | 1532/1540 |

### Temporal-constrained subset（POINT+RANGE，n=209）

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 | Queries with targets |
|--------|----------|--------------|-------|----------|----------------------|
| B | 0.5459 | 0.5218 | 0.7225 | 0.7083 | 208/209 |
| B+TF | 0.5103 | 0.4821 | 0.6268 | 0.6042 | 208/209 |

### Any-temporal subset（n=216）

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 | Queries with targets |
|--------|----------|--------------|-------|----------|----------------------|
| B | 0.5374 | 0.5140 | 0.7222 | 0.7085 | 215/216 |
| B+TF | 0.5050 | 0.4778 | 0.6250 | 0.6031 | 215/216 |

## QA Evaluation

| Method | F1 | Precision | Recall | Accuracy | BLEU | Samples |
|--------|----|-----------|--------|----------|------|---------|
| B | 0.5133 | 0.5440 | 0.5409 | 0.2318 | 0.3892 | 1540 |
| B+TF | 0.4879 | 0.5151 | 0.5148 | 0.2195 | 0.3692 | 1540 |

## 按 category 统计

### B

| Category | Count | F1 | Precision | Recall | Accuracy | BLEU |
|----------|-------|----|-----------|--------|----------|------|
| Multi-hop (1) | 282 | 0.3636 | 0.3540 | 0.4296 | 0.0461 | 0.2371 |
| Temporal (2) | 321 | 0.5257 | 0.5511 | 0.5367 | 0.1931 | 0.4014 |
| Open reasoning (3) | 96 | 0.2689 | 0.2991 | 0.3290 | 0.1250 | 0.1925 |
| Single-hop (4) | 841 | 0.5867 | 0.6330 | 0.6040 | 0.3210 | 0.4580 |

### B+TF

| Category | Count | F1 | Precision | Recall | Accuracy | BLEU |
|----------|-------|----|-----------|--------|----------|------|
| Multi-hop (1) | 282 | 0.3572 | 0.3479 | 0.4246 | 0.0426 | 0.2336 |
| Temporal (2) | 321 | 0.5105 | 0.5328 | 0.5211 | 0.1807 | 0.3900 |
| Open reasoning (3) | 96 | 0.2419 | 0.2687 | 0.3034 | 0.0938 | 0.1584 |
| Single-hop (4) | 841 | 0.5512 | 0.5925 | 0.5667 | 0.3080 | 0.4307 |

