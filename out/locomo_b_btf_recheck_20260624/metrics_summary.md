# LoCoMo B/B+TF Metrics Summary (with latency fix)

Output directory: `out/locomo_b_btf_recheck_20260624`

## Retrieval Latency

| Method | Avg Parse | Avg Filter | Avg Search No Parse | Avg Total With Parse | p50 Total | p95 Total |
|--------|-----------|------------|---------------------|----------------------|-----------|----------|
| B | 0.000s | 0.000s | 0.105s | 0.105s | 0.104s | 0.147s |
| B+TF | 1.003s | 0.002s | 2.227s | 3.231s | 3.189s | 4.264s |

### B+TF Parse Source Distribution

{'LLM': 1192, 'FAST': 348}

### B+TF Time Constraint Distribution

{'NONE': 1324, 'RANGE': 90, 'POINT': 119, 'BEFORE': 6, 'AFTER': 1}

### Fallback Count: 38

## Evidence Metrics (top-5)

### All Queries

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 | Queries w/ Targets |
|--------|----------|--------------|-------|----------|--------------------|
| B | 0.6592 | 0.5608 | 0.8201 | 0.7580 | 1532/1540 |
| B+TF | 0.6066 | 0.5114 | 0.7708 | 0.7065 | 1532/1540 |

### Temporal-Constrained Subset (POINT+RANGE, n=209)

*Subset defined by B+TF time_constraint_type; same (user_id, qa_idx) used for both methods.*

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 | Queries w/ Targets |
|--------|----------|--------------|-------|----------|--------------------|
| B | 0.5459 | 0.5218 | 0.7225 | 0.7083 | 208/209 |
| B+TF | 0.5103 | 0.4821 | 0.6268 | 0.6042 | 208/209 |

### Temporal-Constrained Subset (any_temporal, n=216)

*Subset defined by B+TF time_constraint_type; same (user_id, qa_idx) used for both methods.*

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 | Queries w/ Targets |
|--------|----------|--------------|-------|----------|--------------------|
| B | 0.5374 | 0.5140 | 0.7222 | 0.7085 | 215/216 |
| B+TF | 0.5050 | 0.4778 | 0.6250 | 0.6031 | 215/216 |

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
