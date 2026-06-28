# LoCoMo B/B+TF 复现说明（中文预览版）

**基准提交**：`8d6ed17`  
**预览分支**：`upstream-repro-submission-preview-zh`  
**实验日期**：2026-06-24  
**结果目录**：`out/locomo_b_btf_recheck_20260624/`

## 摘要

这份说明记录了我们基于公开仓库代码对 SA-Mem 论文中 LoCoMo B/B+TF 路径的复现情况。复现重点是：

- **B**：基于 MemBlock 的普通向量检索。
- **B+TF**：在 B 的基础上加入 query parsing；解析得到的时间约束用于 Time Filtering，也就是在语义排序前通过 temporal index 裁剪候选 MemBlock。

主要观察如下：

- **B 基线基本对齐论文**：LoCoMo QA 指标、全量 evidence 指标、warm/core 检索延迟都接近论文报告值。
- **B+TF 的整体 QA/evidence 增益没有复现**：在当前公开 enhanced retrieval 路径下，B+TF 的 F1、BLEU、Hit@5、Recall@5、Complete-MRR 都低于 B。
- **B+TF 在真正触发时间约束的子集上有局部延迟收益**：category 2 中解析出 POINT/RANGE 的 40 个问题里，候选池明显缩小，core semantic ranking latency 降低；但全量运行中多数 category 2 问题被解析为 `NONE`，并且在线 rewritten-query embedding 和 cache flush 会主导端到端耗时。

因此，这份报告更适合作为一份“B/B+TF 公开代码路径复现记录”：它说明哪些指标对齐、哪些指标仍有差距，以及后续需要确认的实现细节。

## 实验范围

| 项目 | 设置 |
|------|------|
| 数据集 | LoCoMo10，10 个 conversation |
| QA 样本 | 1540 个非 category-5 QA |
| 方法 | B、B+TF |
| Graph retrieval | 关闭 |
| Retrieval top-k | 5 |
| Generation answer top-n | 5 |
| Generation context | `content` text mode |
| LLM | `gpt-4o-mini` |
| Embedding model | `text-embedding-3-small` |
| 结果目录 | `out/locomo_b_btf_recheck_20260624/` |

本次只覆盖非图版本的 B/B+TF 路径。B+HTM、graph expansion、HaluMem、受控 token-budget 实验不在本次运行范围内。

## Query Rewriting 说明

当前公开仓库代码中，`QueryParser` 已经包含 `rewritten_query` 字段；enhanced retrieval 会使用 `directive.rewritten_query` 作为向量排序的 query text。因此，query rewriting 不是本次复现从零新增的机制。

不过，论文没有给出 exact query-rewriting prompt，也没有提供将 query rewriting 与 temporal filtering 拆开的独立 ablation。README 中对 enhanced mode 的描述也主要是 Time Filtering。我们在 recheck 中修复了 query embedding cache key 冲突，使 B+TF 不再复用 B 的原始 question embedding。修复后，B+TF 实际使用 rewritten-query embedding，QA/evidence 指标出现下降。

因此，当前 B+TF 结果应理解为：**公开 enhanced retrieval 路径 + rewritten-query embedding 启用** 时的结果。为了进一步隔离 temporal filtering 本身，后续可以补一个 ablation：

- **B+TF-original-query**：保留 query parsing 和 temporal-index candidate pruning，但语义排序时使用原始 question embedding，而不是 rewritten-query embedding。

这个 ablation 可以帮助判断 B+TF 与论文不一致主要来自 temporal pruning、query rewriting 质量，还是二者交互。

### 定性例子

之前的 recheck 分析中有一些例子可以帮助理解 B+TF 差距。这些例子只作为诊断材料，不替代 no-rewrite ablation。

**Rewritten-query 行为。** Parser 会在语义排序前移除显式时间表达：

| 原始 query | rewritten query | 被移除的时间表达 |
|------------|-----------------|------------------|
| `What did Mel and her kids paint in their latest project in July 2023?` | `What did Mel and her kids paint in their latest project` | `in July 2023` |
| `What painting did Melanie show to Caroline on October 13, 2023?` | `What painting did Melanie show to Caroline` | `on October 13, 2023` |
| `Where did Caroline move from 4 years ago?` | `Where did Caroline move from` | `4 years ago` |

这与公开代码设计一致：时间信息被放入 `time_constraint`，rewritten query 用于向量排序。潜在问题是，部分时间表达本身也可能帮助语义消歧。

**B 更好的例子。** 有些失败看起来来自 temporal pruning 过窄：

| Query | Temporal filter effect | B top-5 | B+TF top-5 | 观察 |
|-------|------------------------|---------|------------|------|
| `What did Mel and her kids paint in their latest project in July 2023?` | POINT filter，pool `67 -> 3` | 命中 target `25` | 未命中 | 正确 block 被排除在 filtered pool 外。 |
| `Why did Maria sit with the little girl at the shelter event in February 2023?` | POINT filter，pool `86 -> 7` | 命中 target `127` | 未命中 | 候选池很小，但丢失了相关 block。 |

**B+TF 更好的例子。** 也有例子体现了 Time Filtering 的预期作用：

| Query | Temporal filter effect | B top-5 | B+TF top-5 | 观察 |
|-------|------------------------|---------|------------|------|
| `What painting did Melanie show to Caroline on October 13, 2023?` | POINT filter，pool `67 -> 4` | 未命中 | 命中 target `58` | temporal pruning 定位到了时间相关 block。 |

这些例子说明，当前问题不是简单的“Time Filtering 没有效果”。机制在部分 case 上有效，但本次运行中过度过滤和 rewritten-query mismatch 的负面 case 抵消并超过了正面收益。

## 复现流程

本次复现遵循完整 LoCoMo QA 流程：

1. **Build MemBlocks**：运行 `build_stage_locomo.py`。
   - graph 关闭。
   - 输出 `out/<run_id>/final_boxes_content.jsonl`。
2. **Retrieve evidence**：分别运行 B 和 B+TF。
   - B 输出 `retrieval_baseline.jsonl`。
   - B+TF 输出 `retrieval_enhanced.jsonl`。
   - retrieval top-k = 5。
3. **Generate answers**：运行 `generate_stage_locomo.py`。
   - B 使用 `retrieval_baseline.jsonl`。
   - B+TF 使用 `retrieval_enhanced.jsonl`。
   - generation 使用 `answer_topn=5` 和 `text_modes=content`。
4. **Evaluate QA**：运行 `evaluate_locomo.py`。
   - 统计 F1、precision、recall、accuracy、BLEU。
5. **Analyze metrics**：运行 `scripts/analyze_repro_metrics.py`。
   - 统计 Complete-MRR、First-RR、Hit@5、Recall@5、latency breakdown、temporal subset diagnostics。

辅助脚本：

- `scripts/run_locomo_b_btf.sh`：端到端运行 B/B+TF。
- `scripts/retrieve_locomo_b_btf.py`：给原始 retrieval entrypoint 暴露 top-k 参数。
- `scripts/analyze_repro_metrics.py`：汇总 QA、evidence、latency 指标。

## 代码与统计修正

### 1. 修复 B+TF query embedding cache key

**问题**：B 和 B+TF 原本都使用 `key=qa_{user_id}_{q_id}`、`field="question"`。如果先运行 B，B+TF 会命中 B 的原始 question embedding cache，而不是 rewritten query embedding。

**修复**：B+TF 使用包含 query hash 的独立 key：

```python
key = f"qa_enhanced_{user_id}_{q_id}_{md5(query_text)[:12]}"
field = "question_rewritten"
```

**影响**：修复后 B+TF ranking 发生明显变化。在线 no-parse wall time 也会增加，因为 rewritten-query embedding 和 cache flush 被计入检索路径。为了对齐论文 Table 3 的 search-stage latency，需要单独报告 paper-like core search。

### 2. 修复 temporal-constrained subset 统计

**问题**：B retrieval 没有 `time_constraint_type`，直接按 B 的该字段筛选会得到 0/0。

**修复**：使用 B+TF 的 `time_constraint_type` 定义 temporal subset，再在同一组 `(user_id, qa_idx)` 上分别计算 B 和 B+TF 指标。

### 3. 增加 any-temporal subset

除 POINT+RANGE 外，增加 `any_temporal` subset：POINT + RANGE + BEFORE + AFTER + ANCHOR。

### 4. 增加 warm-cache 和细粒度 latency 字段

检索阶段记录以下字段：parse、filter、query vector、block vector fetch、cosine、sort、flush、rank total、search without parse、total with parse。这样可以区分：

- 论文更接近的 search-stage latency。
- 在线 query parsing、rewritten-query embedding、cache flush 带来的端到端开销。

### 5. 增加可复现 metrics analyzer

`scripts/analyze_repro_metrics.py` 现在支持：

- paper-style Complete-MRR。
- supplementary First-RR。
- temporal subset 对齐比较。
- paper-like core search latency。
- category 2 和 category 2 POINT/RANGE-triggered latency diagnostics。
- Windows 下显式 UTF-8 读写。

## 样本完整性

| 文件 | 数量 |
|------|------|
| `retrieval_baseline.jsonl` | 1540 |
| `retrieval_enhanced.jsonl` | 1540 |
| `generation_results_locomo_baseline.jsonl` | 1540 |
| `generation_results_locomo_time_filtering.jsonl` | 1540 |
| `evaluation_summary_locomo_baseline.json` | 1540 |
| `evaluation_summary_locomo_time_filtering.json` | 1540 |

## QA 指标与论文 Table 1 对比

| Method | 本次 F1 | 本次 BLEU | 说明 |
|--------|---------|-----------|------|
| B | 0.5133 | 0.3892 | 接近论文 SA-Mem overall |
| B+TF | 0.4879 | 0.3692 | 低于 B |

论文 Table 1 报告的是 final SA-Mem 结果，不是严格的 B/B+TF 行。这里更适合看作公开代码 B/B+TF 路径与论文 final result 的近似对照。

论文参考值：SA-Mem overall F1 = 0.5203，BLEU = 0.3908。

## 检索延迟与论文 Table 3 对比

论文 Table 3 报告 search-stage latency。本次 retrieval JSONL 记录了多个嵌套 latency 字段，因此这里区分两个边界：

- **Online no-parse wall**：不含 LLM query parsing，但仍包含 rewritten-query embedding、vector-cache flush、semantic ranking。
- **Paper-like core search**：temporal filter + block vector fetch + cosine scoring + sorting；不含 LLM parsing、rewritten-query embedding API、vector-cache flush。

### Online components（1540 全量问题）

| Method | Parse | Filter | Search No Parse | Total With Parse | p50 | p95 |
|--------|-------|--------|-----------------|------------------|-----|-----|
| B | 0 | 0 | 0.105s | 0.105s | 0.104s | 0.147s |
| B+TF | 1.003s | 0.002s | 2.227s | 3.231s | 3.189s | 4.264s |

B+TF 的 `Search No Parse` 不是纯 search：它包含 rewritten-query embedding（mean 0.780s）和 vector-cache flush（mean 1.340s）。这解释了为什么在线 no-parse wall time 远高于论文延迟。

### Paper-like core search（1540 全量问题）

| Method | Core Mean | Core p50 | Core p95 | Initial Pool Mean | Filtered Pool Mean | Filtered Pool p50 |
|--------|-----------|----------|----------|-------------------|--------------------|-------------------|
| B | 0.1024s | 0.1023s | 0.1438s | 93.36 | 93.36 | 93.0 |
| B+TF | 0.1037s | 0.1078s | 0.1522s | 93.36 | 87.33 | 93.0 |

在 paper-like core search 边界下，B+TF 并不是 21x 慢，而是与 B 接近。但 B+TF 全量也没有明显更快，因为 temporal filtering 只把平均候选池从 93.36 降到 87.33，median filtered pool 仍是 93。

### Category 2 时序问题

category 2 是 LoCoMo temporal-question category。看全部 321 个 category-2 问题：

| Method | Core Mean | Core p50 | Core p95 | Initial Pool Mean | Filtered Pool Mean | Filtered Pool p50 |
|--------|-----------|----------|----------|-------------------|--------------------|-------------------|
| B | 0.1010s | 0.1010s | 0.1473s | 91.65 | 91.65 | 93.0 |
| B+TF | 0.1001s | 0.1040s | 0.1507s | 91.65 | 83.88 | 86.0 |

整体 category-2 效果不明显，因为 B+TF parser 只对 40/321 个 category-2 问题产生显式时间约束：

| B+TF Constraint Type | Count |
|----------------------|-------|
| NONE | 281 |
| POINT | 36 |
| RANGE | 4 |

只看这 40 个 POINT/RANGE-triggered category-2 问题：

| Method | Core Mean | Core p50 | Core p95 | Initial Pool Mean | Filtered Pool Mean | Filtered Pool p50 |
|--------|-----------|----------|----------|-------------------|--------------------|-------------------|
| B | 0.1060s | 0.1053s | 0.1449s | 96.00 | 96.00 | 93.0 |
| B+TF | 0.0429s | 0.0085s | 0.1362s | 96.00 | 33.65 | 4.5 |

这个子集上，Time Filtering 明显缩小候选池并降低 core semantic-ranking latency。但该局部收益在全量运行中被稀释。

## Evidence 指标与论文 Table 5 对比

### 全量问题

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 |
|--------|----------|--------------|-------|----------|
| B | 0.6592 | 0.5608 | 0.8201 | 0.7580 |
| B+TF | 0.6066 | 0.5114 | 0.7708 | 0.7065 |

论文 Table 5 参考值（SA-Mem 约 1200 tokens）：Complete-MRR = 0.5510，Hit@k = 0.8162，Recall@k = 0.7475。

注意：本次 top-5 不等同于论文 controlled token budget；First-RR 是补充指标，不是论文 C-MRR。

### Temporal-constrained subset（POINT+RANGE，n=209）

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 |
|--------|----------|--------------|-------|----------|
| B | 0.5459 | 0.5218 | 0.7225 | 0.7083 |
| B+TF | 0.5103 | 0.4821 | 0.6268 | 0.6042 |

论文 Table 7 参考值：No temporal filter constrained Recall@5 = 0.6891；Union Session & Event constrained Recall@5 = 0.7596。

本次 B+TF 在 temporal-constrained subset 上低于 B，说明 Table 7 的 temporal filtering gain 尚未复现。

### Any-temporal subset（n=216）

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 |
|--------|----------|--------------|-------|----------|
| B | 0.5374 | 0.5140 | 0.7222 | 0.7085 |
| B+TF | 0.5050 | 0.4778 | 0.6250 | 0.6031 |

趋势相同：B 在该子集上仍高于 B+TF。

## QA Evaluation Results

| Method | F1 | Precision | Recall | Accuracy | BLEU | Samples |
|--------|----|-----------|--------|----------|------|---------|
| B | 0.5133 | 0.5440 | 0.5409 | 0.2318 | 0.3892 | 1540 |
| B+TF | 0.4879 | 0.5151 | 0.5148 | 0.2195 | 0.3692 | 1540 |

### 按 category 统计

**B**

| Category | Count | F1 | Precision | Recall | Accuracy | BLEU |
|----------|-------|----|-----------|--------|----------|------|
| Multi-hop (1) | 282 | 0.3636 | 0.3540 | 0.4296 | 0.0461 | 0.2371 |
| Temporal (2) | 321 | 0.5257 | 0.5511 | 0.5367 | 0.1931 | 0.4014 |
| Open reasoning (3) | 96 | 0.2689 | 0.2991 | 0.3290 | 0.1250 | 0.1925 |
| Single-hop (4) | 841 | 0.5867 | 0.6330 | 0.6040 | 0.3210 | 0.4580 |

**B+TF**

| Category | Count | F1 | Precision | Recall | Accuracy | BLEU |
|----------|-------|----|-----------|--------|----------|------|
| Multi-hop (1) | 282 | 0.3572 | 0.3479 | 0.4246 | 0.0426 | 0.2336 |
| Temporal (2) | 321 | 0.5105 | 0.5328 | 0.5211 | 0.1807 | 0.3900 |
| Open reasoning (3) | 96 | 0.2419 | 0.2687 | 0.3034 | 0.0938 | 0.1584 |
| Single-hop (4) | 841 | 0.5512 | 0.5925 | 0.5667 | 0.3080 | 0.4307 |

## 小结

### 已对齐或接近

1. **B 的 QA 指标**：F1/BLEU 接近论文 Table 1 的 SA-Mem overall。
2. **B 的全量 evidence 指标**：Hit@5、Recall@5、Complete-MRR 接近论文 Table 5。
3. **B 的 warm/core latency**：约 0.1s，接近论文 Table 3 中 SA-Mem(B)。
4. **B+TF 的局部 latency 行为**：在 category-2 POINT/RANGE-triggered 子集上，候选池和 core search latency 明显下降。

### 尚未对齐或需要继续确认

1. **B+TF 整体 QA/evidence gain**：本次 B+TF 低于 B。
2. **Temporal-constrained evidence gain**：B+TF 在 POINT+RANGE 子集上低于 B。
3. **B+TF 端到端检索耗时**：online no-parse wall time 主要受 rewritten-query embedding 与 cache flush 影响。
4. **Query rewriting 与 temporal filtering 的影响拆分**：仍需要 B+TF-original-query ablation。
5. **Controlled token budget equivalence**：本次 top-5 设置不能完全等同论文 controlled token budget。

## 建议提交到原始仓库的内容

如果后续提交到 `RichardWang11/SA-Mem-Research`，建议新建干净分支，例如 `locomo-b-btf-reproduction`，不要把本实验仓库整体搬过去。

### 建议提交

| 文件 / 改动 | 用途 |
|-------------|------|
| `README.md` 短小节 | 简要说明 LoCoMo B/B+TF 复现并链接详细报告 |
| `docs/locomo_b_btf_reproduction.md` | 详细复现报告 |
| `docs/locomo_b_btf_metrics_summary.md` | 精简指标摘要 |
| `docs/locomo_b_btf_metrics_summary.json` | 机器可读指标摘要 |
| `scripts/run_locomo_b_btf.sh` | 端到端 B/B+TF runner |
| `scripts/retrieve_locomo_b_btf.py` | 暴露 retrieval top-k 的 wrapper |
| `scripts/analyze_repro_metrics.py` | QA、evidence、latency 分析脚本 |
| `retrieval/retrieval_impl_locomo.py` patch | B baseline warm-cache 与 latency 字段 |
| `retrieval/retrieval_enhanced_locomo.py` patch | B+TF query parsing、temporal-index candidate pruning、query-key 修复、latency 字段 |
| `retrieval/query_pasing_byllm.py` patch | time-axis parsing 与 temporal-filter dispatch helpers |
| `retrieval/retrieve_stage_enhanced_locomo.py` patch | axis mode 等 CLI 支持 |
| `generate_impl_locomo.py` patch（如仍需要） | `LIMIT_CONVERSATIONS=-1` 处理 |

### 不建议提交

| 文件 / 目录 | 原因 |
|-------------|------|
| `.env` | 包含 API key |
| `dataset/locomo10.json` | 数据集/license 应由作者或使用者自行准备 |
| 完整 `out/locomo_b_btf_*` 目录 | 大量运行产物，不适合进上游 PR |
| `out/**/vector_store/*.json` | embedding cache |
| `out/**/retrieval_*.jsonl` | 大型 raw retrieval 输出 |
| `out/**/generation_*.jsonl` | 大型 raw generation 输出 |
| `out/**/evaluation_*.jsonl` | 大型 raw evaluation 输出 |
| `out/**/token_stream.jsonl` | debug/token 日志 |
| `out/**/trace_build_process.jsonl` | build trace 日志 |
| `ANALYSIS_BTF_VS_B.md` 原文 | 可作为内部分析来源，但上游报告应吸收结论而不是原样提交 |
| `UPSTREAM_REPRO_SUBMISSION_PLAN.md` 原文 | 这是工作计划文档，不是上游复现材料 |

## 输出目录结构

本次完整运行结果位于：

```text
out/locomo_b_btf_recheck_20260624/
├── retrieval_baseline.jsonl
├── retrieval_enhanced.jsonl
├── generation_results_locomo_baseline.jsonl
├── generation_results_locomo_time_filtering.jsonl
├── evaluation_summary_locomo_baseline.json
├── evaluation_summary_locomo_time_filtering.json
├── metrics_summary.json
└── metrics_summary.md
```

上游 PR 中建议只提交 `docs/` 下的精简摘要，不提交完整 `out/`。

