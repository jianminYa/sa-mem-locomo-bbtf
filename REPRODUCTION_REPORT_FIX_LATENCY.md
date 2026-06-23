# SA-Mem LoCoMo B/B+TF 复现报告（Latency Fix 版本）

## 修复内容

### 1. B+TF Temporal Filtering 空候选 Fallback

**旧行为**: `_filter_by_metadata()` 返回空列表 → `rankings=[]` → generation 跳过该 QA（共 23 条）

**新行为**: filter 后为空 → fallback 到全体 MemBlocks → semantic ranking → 继续生成答案

**代码改动**: `retrieval/retrieval_enhanced_locomo.py` 中 `_score_and_rank()` 方法

```python
if not filtered_pool:
    mx.logger.warning(
        "Temporal filtering returned 0 blocks; falling back to full pool for semantic ranking."
    )
    filtered_pool = pool
    fallback_to_full_pool = True
```

**效果**: B+TF generation 从 1517 条提升到 **1540 条**（与数据集完全匹配）

### 2. 新增 Retrieval Latency 字段

**B+TF Enhanced Retrieval**:
- `retrieval_latency_parse`: query parsing 耗时
- `retrieval_latency_filter`: metadata filtering 耗时
- `retrieval_latency_rank`: vector similarity ranking 耗时
- `retrieval_latency_total_with_parse`: parse + filter + rank
- `retrieval_latency_core_no_parse`: filter + rank（在线检索核心延迟）
- `fallback_to_full_pool`: 是否 fallback
- `filtered_pool_size` / `initial_pool_size`: 过滤前后候选池大小
- `parse_source`: FAST / LLM
- `query_intent`: STATIC / WINDOW / MISC / PLANNING

**B Baseline Retrieval**:
- 同样添加 latency 字段（parse=0, filter=0）

### 3. LIMIT_CONVERSATIONS=-1 切片 Bug 修复

**问题**: `raw_list[:-1]` 跳过最后一个 conversation

**修复**: `all_data if (limit is None or limit <= 0) else all_data[:limit]`

## 实验配置

| 参数 | 值 |
|------|-----|
| 数据集 | LoCoMo10 (10 conversations) |
| LLM | gpt-4o-mini |
| Embedding | text-embedding-3-small |
| Retrieval top-k | 5 |
| Generation top-n | 5 |
| Text mode | content |
| Graph | disabled |
| API proxy | yunwu.ai |

## 本次不重新 Build

复用 `out/locomo_b_btf_full/final_boxes_content.jsonl`（894 blocks）。
因此不报告 build token / build latency。

## 实验结果

### Line Count 验收

| 文件 | 条数 |
|------|------|
| retrieval_baseline.jsonl | **1540** |
| retrieval_enhanced.jsonl | **1540** |
| generation_results_locomo_baseline.jsonl | **1540** |
| generation_results_locomo_time_filtering.jsonl | **1540** |

- Enhanced retrieval empty rankings: **0**
- Fallback count: **32**

### Retrieval Latency

| Method | Avg Parse | Avg Filter | Avg Rank | Avg Core No Parse | Avg Total With Parse | p50 Total | p95 Total |
|--------|-----------|------------|----------|-------------------|----------------------|-----------|-----------|
| B | 0 | 0 | 0.265s | 0.265s | 0.265s | 0.265s | 0.348s |
| B+TF | 1.035s | 0.002s | 0.272s | 0.274s | 1.309s | 1.490s | 2.062s |

**Parse source 分布**: LLM=1192, FAST=348

**Time constraint 分布**: NONE=1324, RANGE=90, POINT=119, BEFORE=6, AFTER=1

**Fallback**: 32 次（temporal filtering 返回 0 时 fallback 到全池）

### Evidence Metrics (top-5)

| Method | C-MRR | Hit@5 | Recall@5 |
|--------|-------|-------|----------|
| B | 0.6592 | 0.8201 | 0.7580 |
| B+TF | 0.6548 | 0.8052 | 0.7426 |

### Evaluation Results

| Method | F1 | Precision | Recall | Accuracy | BLEU | Samples |
|--------|-----|-----------|--------|----------|------|---------|
| **B** | **0.5150** | **0.5455** | **0.5419** | **0.2318** | **0.3901** | 1540 |
| B+TF | 0.5046 | 0.5351 | 0.5312 | 0.2240 | 0.3807 | 1540 |

### By Category

#### B

| Category | Count | F1 | Precision | Recall | Accuracy | BLEU |
|----------|-------|-----|-----------|--------|----------|------|
| Multi-hop (1) | 282 | 0.3702 | 0.3606 | 0.4350 | 0.0496 | 0.2438 |
| Temporal (2) | 321 | 0.5288 | 0.5540 | 0.5378 | 0.1963 | 0.4056 |
| Open reasoning (3) | 96 | 0.2587 | 0.2840 | 0.3181 | 0.1146 | 0.1826 |
| Single-hop (4) | 841 | 0.5876 | 0.6341 | 0.6048 | 0.3199 | 0.4569 |

#### B+TF

| Category | Count | F1 | Precision | Recall | Accuracy | BLEU |
|----------|-------|-----|-----------|--------|----------|------|
| Multi-hop (1) | 282 | 0.3593 | 0.3523 | 0.4212 | 0.0426 | 0.2333 |
| Temporal (2) | 321 | 0.5287 | 0.5524 | 0.5392 | 0.1931 | 0.4034 |
| Open reasoning (3) | 96 | 0.2656 | 0.2902 | 0.3337 | 0.1146 | 0.1779 |
| Single-hop (4) | 841 | 0.5714 | 0.6178 | 0.5875 | 0.3092 | 0.4446 |

## 结果分析

1. **B 略优于 B+TF**（F1: 0.5150 vs 0.5046），与论文趋势一致
2. **B+TF 在 Cat 3 (Open reasoning) 上略优**（F1: 0.2656 vs 0.2587）
3. **Fallback 机制生效**: 32 个 temporal filtering 空候选查询被挽救
4. **Retrieval latency**: B+TF 的 parse 阶段增加约 1s 延迟（主要来自 LLM 调用）
5. **Evidence metrics**: B 的 C-MRR/Hit@5/Recall@5 略高于 B+TF

## 输出目录

```
out/locomo_b_btf_full/              # 旧结果（保留）
out/locomo_b_btf_fix_latency_20260623/  # 新结果
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

*报告生成时间: 2026-06-23*
*修复版本: latency fix with fallback*
