# SA-Mem LoCoMo B/B+TF Reproduction Report (Recheck Version)

**Commit**: `f3e8720` (will be updated after this commit)
**Date**: 2026-06-24
**New result directory**: `out/locomo_b_btf_recheck_20260624/`

## Changes Made

### 1. Fixed B+TF Query Embedding Cache Key

**Problem**: B baseline and B+TF enhanced both used `key=qa_{user_id}_{q_id}`, `field="question"`. If B ran first, B+TF would reuse B's original question embedding instead of the rewritten query.

**Fix**: B+TF now uses `key=qa_enhanced_{user_id}_{q_id}_{md5(query_text)[:12]}`, `field="question_rewritten"`.

**Impact**: B+TF ranking changed significantly. B+TF search latency increased from ~0.1s to ~2.2s (new embeddings need to be computed). B+TF evidence metrics decreased, suggesting the rewritten query embeddings were not being used correctly before.

### 2. Fixed Temporal-Constrained Subset Evidence Statistics

**Problem**: B retrieval has no `time_constraint_type`, so filtering B by it returned 0/0.

**Fix**: Use B+TF's `time_constraint_type` to define the subset, then compute metrics for both B and B+TF on the same `(user_id, qa_idx)` pairs.

### 3. Added Any-Temporal Subset

Added `any_temporal` subset (POINT + RANGE + BEFORE + AFTER + ANCHOR) in addition to POINT+RANGE.

## Sample Completeness

| File | Count |
|------|-------|
| retrieval_baseline.jsonl | 1540 |
| retrieval_enhanced.jsonl | 1540 |
| generation_results_locomo_baseline.jsonl | 1540 |
| generation_results_locomo_time_filtering.jsonl | 1540 |
| evaluation_summary_locomo_baseline.json | 1540 |
| evaluation_summary_locomo_time_filtering.json | 1540 |

## QA vs Paper Table 1

| Method | Our F1 | Our BLEU | Notes |
|--------|--------|----------|-------|
| B | 0.5133 | 0.3892 | Close to paper |
| B+TF | 0.4879 | 0.3692 | Lower than B |

**Note**: Paper Table 1 shows final SA-Mem results, not strict B/B+TF rows. Our B results are close; B+TF is lower, indicating temporal filtering gain is not reproduced.

## Retrieval Latency vs Paper Table 3

| Method | Parse | Filter | Search No Parse | Total With Parse | p50 | p95 |
|--------|-------|--------|-----------------|------------------|-----|-----|
| B | 0 | 0 | 0.105s | 0.105s | 0.104s | 0.147s |
| B+TF | 1.003s | 0.002s | 2.227s | 3.231s | 3.189s | 4.264s |

**Paper-like latency**: No-parse warm-cache search = 0.105s (B), 2.227s (B+TF)

**Note**: B+TF latency increased because rewritten query embeddings are now correctly computed (not reused from B cache).

## Evidence Metrics vs Paper Table 5

### All Queries

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 |
|--------|----------|--------------|-------|----------|
| B | 0.6592 | 0.5608 | 0.8201 | 0.7580 |
| B+TF | 0.6066 | 0.5114 | 0.7708 | 0.7065 |

**Paper reference** (SA-Mem ~1200 tokens): Complete-MRR 0.5510, Hit@5 0.8162, Recall@5 0.7475

**Note**: Our top-5 is not equivalent to paper's controlled token budget. First-RR is supplementary, not the paper's C-MRR.

### Temporal-Constrained Subset (POINT+RANGE, n=209)

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 |
|--------|----------|--------------|-------|----------|
| B | 0.5459 | 0.5218 | 0.7225 | 0.7083 |
| B+TF | 0.5103 | 0.4821 | 0.6268 | 0.6042 |

**Paper reference** (Table 7, no temporal filter): Recall@5 constrained = 0.6891

**Finding**: B+TF is lower than B on temporal-constrained queries. Temporal filtering gain is NOT reproduced.

### Any-Temporal Subset (n=216)

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 |
|--------|----------|--------------|-------|----------|
| B | 0.5374 | 0.5140 | 0.7222 | 0.7085 |
| B+TF | 0.5050 | 0.4778 | 0.6250 | 0.6031 |

Same pattern: B outperforms B+TF on temporal queries.

## Evaluation Results

| Method | F1 | Precision | Recall | Accuracy | BLEU | Samples |
|--------|-----|-----------|--------|----------|------|---------|
| B | 0.5133 | 0.5440 | 0.5409 | 0.2318 | 0.3892 | 1540 |
| B+TF | 0.4879 | 0.5151 | 0.5148 | 0.2195 | 0.3692 | 1540 |

## By Category

### B

| Category | Count | F1 | Precision | Recall | Accuracy | BLEU |
|----------|-------|-----|-----------|--------|----------|------|
| Multi-hop (1) | 282 | 0.3636 | 0.3540 | 0.4296 | 0.0461 | 0.2371 |
| Temporal (2) | 321 | 0.5257 | 0.5511 | 0.5367 | 0.1931 | 0.4014 |
| Open reasoning (3) | 96 | 0.2689 | 0.2991 | 0.3290 | 0.1250 | 0.1925 |
| Single-hop (4) | 841 | 0.5867 | 0.6330 | 0.6040 | 0.3210 | 0.4580 |

### B+TF

| Category | Count | F1 | Precision | Recall | Accuracy | BLEU |
|----------|-------|-----|-----------|--------|----------|------|
| Multi-hop (1) | 282 | 0.3572 | 0.3479 | 0.4246 | 0.0426 | 0.2336 |
| Temporal (2) | 321 | 0.5105 | 0.5328 | 0.5211 | 0.1807 | 0.3900 |
| Open reasoning (3) | 96 | 0.2419 | 0.2687 | 0.3034 | 0.0938 | 0.1584 |
| Single-hop (4) | 841 | 0.5512 | 0.5925 | 0.5667 | 0.3080 | 0.4307 |

## Conclusion

### Partially Aligned

1. **QA metrics (B)**: F1/BLEU close to paper Table 1
2. **All-query evidence (B)**: Hit@5, Recall@5, Complete-MRR close to paper Table 5
3. **Warm-cache latency (B)**: ~0.1s search latency, comparable to paper

### NOT Reproduced

1. **B+TF gain over B**: B+TF performs worse than B in this run
2. **Temporal-constrained evidence gain**: B+TF lower than B on POINT+RANGE queries
3. **B+TF search latency**: Increased to ~2.2s due to correct rewritten query embedding

### Cannot Claim

- Full reproduction of paper's temporal filtering ablation (Table 7)
- B+TF improvement over B
- Controlled token budget equivalence

### Can Claim

- Partial reproduction: B baseline is close to paper numbers
- Evidence metrics methodology is correct (Complete-MRR, First-RR)
- Code changes are documented and reproducible

## Output Directory Structure

```
out/locomo_b_btf_recheck_20260624/
├── retrieval_baseline.jsonl          (1540 lines)
├── retrieval_enhanced.jsonl          (1540 lines)
├── generation_results_locomo_baseline.jsonl    (1540 lines)
├── generation_results_locomo_time_filtering.jsonl  (1540 lines)
├── evaluation_summary_locomo_baseline.json
├── evaluation_summary_locomo_time_filtering.json
├── metrics_summary.json
└── metrics_summary.md
```

---

*Report generated: 2026-06-24*
*Recheck version with corrected query embedding cache key*
