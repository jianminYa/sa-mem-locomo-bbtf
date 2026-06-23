# LoCoMo B/B+TF Metrics Summary (with latency fix)

Output directory: `out/locomo_b_btf_fix_latency_20260623`

## Retrieval Latency

| Method | Avg Parse | Avg Filter | Avg Rank | Avg Core No Parse | Avg Total With Parse | p50 Total | p95 Total |
|--------|-----------|------------|----------|-------------------|----------------------|-----------|----------|
| B | 0 | 0 | 0.265s | 0.265s | 0.265s | 0.265s | 0.348s |
| B+TF | 1.035s | 0.002s | 0.272s | 0.274s | 1.309s | 1.490s | 2.062s |

### B+TF Parse Source Distribution

{'LLM': 1192, 'FAST': 348}

### B+TF Time Constraint Distribution

{'NONE': 1324, 'RANGE': 90, 'POINT': 119, 'BEFORE': 6, 'AFTER': 1}

### Fallback Count: 32

## Evidence Metrics (top-5)

| Method | C-MRR | Hit@5 | Recall@5 |
|--------|-------|-------|----------|
| B | 0.6592 | 0.8201 | 0.7580 |
| B+TF | 0.6548 | 0.8052 | 0.7426 |

## Evaluation Results

| Method | F1 | Precision | Recall | Accuracy | BLEU | Samples |
|--------|-----|-----------|--------|----------|------|---------|
| B | 0.5150 | 0.5455 | 0.5419 | 0.2318 | 0.3901 | 1540 |
| B+TF | 0.5046 | 0.5351 | 0.5312 | 0.2240 | 0.3807 | 1540 |

## By Category


### B

| Category | Count | F1 | Precision | Recall | Accuracy | BLEU |
|----------|-------|-----|-----------|--------|----------|------|
| Multi-hop (1) | 282 | 0.3702 | 0.3606 | 0.4350 | 0.0496 | 0.2438 |
| Temporal (2) | 321 | 0.5288 | 0.5540 | 0.5378 | 0.1963 | 0.4056 |
| Open reasoning (3) | 96 | 0.2587 | 0.2840 | 0.3181 | 0.1146 | 0.1826 |
| Single-hop (4) | 841 | 0.5876 | 0.6341 | 0.6048 | 0.3199 | 0.4569 |

### B+TF

| Category | Count | F1 | Precision | Recall | Accuracy | BLEU |
|----------|-------|-----|-----------|--------|----------|------|
| Multi-hop (1) | 282 | 0.3593 | 0.3523 | 0.4212 | 0.0426 | 0.2333 |
| Temporal (2) | 321 | 0.5287 | 0.5524 | 0.5392 | 0.1931 | 0.4034 |
| Open reasoning (3) | 96 | 0.2656 | 0.2902 | 0.3337 | 0.1146 | 0.1779 |
| Single-hop (4) | 841 | 0.5714 | 0.6178 | 0.5875 | 0.3092 | 0.4446 |
