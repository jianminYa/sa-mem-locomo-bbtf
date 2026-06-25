# SA-Mem LoCoMo B/B+TF Reproduction Report (Warm-Latency Version)

## Changes from Previous Version

### 1. Bi-Temporal Filtering (Algorithm 2 Alignment)

**Code change**: Added `dispatch_temporal_filter` and `axis_mode` parameter from upstream `retrieval-locomo`.

- **axis_mode=auto**: QueryParser infers time axis (SESSION/EVENT/BOTH_UNION)
- **axis_mode=session**: Force session-time axis only
- **axis_mode=event**: Force event-time axis only
- **axis_mode=none**: Skip temporal filtering entirely

Default is `auto`, which uses regex-based `_infer_time_axis` to detect session vs event cues.

### 2. Warm-Cache Optimization

**Problem**: Each QA query created a new `EmbeddingStore` and reloaded the vector JSON.

**Fix**: Create `EmbeddingStore` once per user in the QA loop, pass it to `_score_and_rank`.

**Impact**: B search latency dropped from 0.265s to 0.100s (2.6x faster).

### 3. Fine-Grained Latency Fields

New fields in retrieval JSONL:
- `retrieval_latency_store_init`: EmbeddingStore creation time
- `retrieval_latency_query_vector`: Query embedding time
- `retrieval_latency_block_vector_fetch`: Block vector fetch time (sum over all blocks)
- `retrieval_latency_cosine`: Cosine similarity computation (sum)
- `retrieval_latency_sort`: Ranking sort time
- `retrieval_latency_flush`: Store flush time
- `retrieval_latency_rank_total`: Total rank phase (store_init + query_vec + block_vec + cosine + sort + flush)
- `retrieval_latency_search_no_parse`: filter + rank_total (core search latency)
- `retrieval_latency_total_with_parse`: parse + search_no_parse (end-to-end)
- `time_axis`: Inferred time axis (SESSION/EVENT/BOTH_UNION/NONE)

### 4. Complete-MRR (Paper Definition)

**Definition**: If all gold memories are in top-k, score = M / rank_max. Otherwise 0.

**Complement**: First-Relevant MRR (reciprocal rank of first relevant item) is also reported.

### 5. Fallback Mechanism

When temporal filtering returns 0 blocks, fallback to full pool for semantic ranking.

## Results

### Line Count Verification

| File | Count |
|------|-------|
| retrieval_baseline.jsonl | 1540 |
| retrieval_enhanced.jsonl | 1540 |
| generation_results_locomo_baseline.jsonl | 1540 |
| generation_results_locomo_time_filtering.jsonl | 1540 |

- Empty rankings: 0
- Fallback count: 35

### Retrieval Latency (Warm-Cache)

| Method | Avg Parse | Avg Filter | Avg Search No Parse | Avg Total With Parse | p50 | p95 |
|--------|-----------|------------|---------------------|----------------------|-----|-----|
| B | 0 | 0 | 0.100s | 0.100s | 0.101s | 0.142s |
| B+TF | 1.116s | 0.002s | 0.103s | 1.219s | 1.381s | 2.059s |

**Parse source**: LLM=1192, FAST=348

**Time constraint**: NONE=1325, RANGE=90, POINT=118, BEFORE=6, AFTER=1

**Time axis** (B+TF): Inferred by `_infer_time_axis`

### Evidence Metrics (top-5)

#### All Queries

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 | Queries w/ Targets |
|--------|----------|--------------|-------|----------|--------------------|
| B | 0.6592 | 0.5608 | 0.8201 | 0.7580 | 1532/1540 |
| B+TF | 0.6525 | 0.5531 | 0.8045 | 0.7416 | 1532/1540 |

#### Temporal-Constrained Subset (POINT + RANGE)

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 | Queries w/ Targets |
|--------|----------|--------------|-------|----------|--------------------|
| B+TF | 0.4944 | 0.4634 | 0.6058 | 0.5853 | 207/208 |

**Paper reference** (Table 5, SA-Mem ~1200 tokens): C-MRR 0.5510, Hit@k 0.8162, Recall@k 0.7475

**Note**: Our top-5 is not equivalent to the paper's controlled token budget.

### Evaluation Results

| Method | F1 | Precision | Recall | Accuracy | BLEU | Samples |
|--------|-----|-----------|--------|----------|------|---------|
| B | 0.5140 | 0.5438 | 0.5418 | 0.2338 | 0.3894 | 1540 |
| B+TF | 0.5035 | 0.5342 | 0.5298 | 0.2234 | 0.3792 | 1540 |

### By Category

#### B

| Category | Count | F1 | Precision | Recall | Accuracy | BLEU |
|----------|-------|-----|-----------|--------|----------|------|
| Multi-hop (1) | 282 | 0.3651 | 0.3555 | 0.4309 | 0.0496 | 0.2352 |
| Temporal (2) | 321 | 0.5284 | 0.5502 | 0.5412 | 0.1963 | 0.4058 |
| Open reasoning (3) | 96 | 0.2801 | 0.3088 | 0.3394 | 0.1354 | 0.2026 |
| Single-hop (4) | 841 | 0.5852 | 0.6313 | 0.6024 | 0.3210 | 0.4562 |

#### B+TF

| Category | Count | F1 | Precision | Recall | Accuracy | BLEU |
|----------|-------|-----|-----------|--------|----------|------|
| Multi-hop (1) | 282 | 0.3577 | 0.3493 | 0.4233 | 0.0390 | 0.2318 |
| Temporal (2) | 321 | 0.5280 | 0.5524 | 0.5375 | 0.1900 | 0.4028 |
| Open reasoning (3) | 96 | 0.2599 | 0.2836 | 0.3228 | 0.1146 | 0.1840 |
| Single-hop (4) | 841 | 0.5708 | 0.6179 | 0.5863 | 0.3103 | 0.4420 |

## Comparison with Paper

### What's Aligned

1. **QA metrics**: F1/BLEU within 1-2% of paper Table 1
2. **Retrieval latency**: Warm-cache B search ~0.1s (comparable to paper)
3. **Evidence metrics**: Hit@5 and Recall@5 close to paper Table 5
4. **Bi-temporal filtering**: Using auto axis mode (SESSION/EVENT/BOTH_UNION)

### What's Different

1. **Model**: We use gpt-4o-mini; paper may use different model
2. **Token budget**: Paper uses controlled token budget; we use top-k=5
3. **Build**: We don't report build latency (reusing existing memory blocks)
4. **Graph**: Not enabled (paper has graph experiments)

### Cannot Claim Full Reproduction

1. Build stage not reproduced (reusing pre-built memory blocks)
2. Token budget vs top-k difference
3. API randomness affects exact numbers
4. Some ablation experiments not run (session-only, event-only axis)

## Output Directory

```
out/locomo_b_btf_warm_latency_20260623/
├── retrieval_baseline.jsonl
├── retrieval_enhanced.jsonl
├── generation_results_locomo_baseline.jsonl
├── generation_results_locomo_time_filtering.jsonl
├── evaluation_summary_locomo_baseline.json
├── evaluation_summary_locomo_time_filtering.json
├── metrics_summary.json
└── metrics_summary.md
```

---

*Report generated: 2026-06-24*
*Version: warm-latency with bi-temporal filtering*
