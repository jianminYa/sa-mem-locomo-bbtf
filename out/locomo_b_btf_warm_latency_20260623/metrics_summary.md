# LoCoMo B/B+TF Metrics Summary (with latency fix)

Output directory: `out/locomo_b_btf_warm_latency_20260623`

## Retrieval Latency

| Method | Avg Parse | Avg Filter | Avg Search No Parse | Avg Total With Parse | p50 Total | p95 Total |
|--------|-----------|------------|---------------------|----------------------|-----------|----------|
| B | 0.000s | 0.000s | 0.100s | 0.100s | 0.101s | 0.142s |
| B+TF | 1.116s | 0.002s | 0.103s | 1.219s | 1.381s | 2.059s |

### B+TF Parse Source Distribution

{'LLM': 1192, 'FAST': 348}

### B+TF Time Constraint Distribution

{'NONE': 1325, 'RANGE': 90, 'POINT': 118, 'BEFORE': 6, 'AFTER': 1}

### Fallback Count: 35

## Evidence Metrics (top-5)

### All Queries

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 | Queries w/ Targets |
|--------|----------|--------------|-------|----------|--------------------|
| B | 0.6592 | 0.5608 | 0.8201 | 0.7580 | 1532/1540 |
| B+TF | 0.6525 | 0.5531 | 0.8045 | 0.7416 | 1532/1540 |

### Temporal-Constrained Subset (POINT + RANGE)

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 | Queries w/ Targets |
|--------|----------|--------------|-------|----------|--------------------|
| B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| B+TF | 0.4944 | 0.4634 | 0.6058 | 0.5853 | 207/208 |

## Evaluation Results

| Method | F1 | Precision | Recall | Accuracy | BLEU | Samples |
|--------|-----|-----------|--------|----------|------|---------|
| B | 0.5140 | 0.5438 | 0.5418 | 0.2338 | 0.3894 | 1540 |
| B+TF | 0.5035 | 0.5342 | 0.5298 | 0.2234 | 0.3792 | 1540 |

## By Category


### B

| Category | Count | F1 | Precision | Recall | Accuracy | BLEU |
|----------|-------|-----|-----------|--------|----------|------|
| Multi-hop (1) | 282 | 0.3651 | 0.3555 | 0.4309 | 0.0496 | 0.2352 |
| Temporal (2) | 321 | 0.5284 | 0.5502 | 0.5412 | 0.1963 | 0.4058 |
| Open reasoning (3) | 96 | 0.2801 | 0.3088 | 0.3394 | 0.1354 | 0.2026 |
| Single-hop (4) | 841 | 0.5852 | 0.6313 | 0.6024 | 0.3210 | 0.4562 |

### B+TF

| Category | Count | F1 | Precision | Recall | Accuracy | BLEU |
|----------|-------|-----|-----------|--------|----------|------|
| Multi-hop (1) | 282 | 0.3577 | 0.3493 | 0.4233 | 0.0390 | 0.2318 |
| Temporal (2) | 321 | 0.5280 | 0.5524 | 0.5375 | 0.1900 | 0.4028 |
| Open reasoning (3) | 96 | 0.2599 | 0.2836 | 0.3228 | 0.1146 | 0.1840 |
| Single-hop (4) | 841 | 0.5708 | 0.6179 | 0.5863 | 0.3103 | 0.4420 |
