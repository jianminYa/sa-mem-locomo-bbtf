# SA-Mem LoCoMo B / B+TF 复现报告

## 实验概述

复现 SA-Mem 论文中 LoCoMo 数据集上的 **B (Baseline)** 和 **B+TF (Baseline + Temporal Filtering)** 实验。

- **数据集**: LoCoMo10（10 个对话，1540 个非 category-5 QA 对）
- **模型**: gpt-4o-mini (LLM) + text-embedding-3-small (Embedding)
- **API 代理**: yunwu.ai
- **运行环境**: Ubuntu (DigitalOcean), conda Python 3.10

## 类别定义

| Category | 含义 | 样本数 |
|----------|------|--------|
| Cat 1 | 多跳推理 (Multi-hop) | 282 |
| Cat 2 | 时序推理 (Temporal) | 321 |
| Cat 3 | 开放推理 (Open reasoning) | 96 |
| Cat 4 | 单跳 (Single-hop) | 841 |
| Cat 5 | 跳过 | - |

## 配置参数

```bash
RETRIEVAL_TOP_K=5
ANSWER_TOP_N=5
TEXT_MODE=content
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
Graph: disabled
```

## 遇到的问题与解决方案

### 1. SimpleRetriever 缺少 graph_person_relations 参数

**问题**: `retrieval_impl_locomo.py` 中 `SimpleRetriever.__init__()` 不接受 `graph_person_relations` 参数，导致 retrieval 阶段报错。

**修复**: 在 `SimpleRetriever.__init__()` 中添加 `graph_person_relations` 参数。

### 2. LIMIT_CONVERSATIONS=-1 导致跳过最后一个 conversation

**问题**: `generate_impl_locomo.py` 中 `raw_list[: mx.Config.LIMIT_CONVERSATIONS]` 当 LIMIT_CONVERSATIONS=-1 时，Python 切片 `[:-1]` 会跳过最后一个元素，导致 user 9 未被处理。

**修复**: 改为 `all_data if (limit is None or limit <= 0) else all_data[:limit]`。

### 3. Build 阶段超时

**问题**: Build 阶段（LLM 调用 topic 分类和事件提取）每对话约需 15-20 分钟，10 个对话总计约 2-3 小时，容易超时。

**解决方案**: 使用 tmux 保活，分阶段运行。Build 完成后可跳过（`SKIP_BUILD=1`）。

### 4. Generation 输出文件追加导致重复

**问题**: Generation 阶段以 append 模式写入，重跑时产生重复条目。

**解决方案**: 跑完后按 `(user_id, qa_idx, text_mode, answer_topn)` 去重。

## 最终结果

### B (Baseline) — 1540 样本

| 指标 | 总体 | Cat 1 (多跳) | Cat 2 (时序) | Cat 3 (开放) | Cat 4 (单跳) |
|------|------|-------------|-------------|-------------|-------------|
| **F1** | **0.5134** | 0.3668 | 0.5286 | 0.2610 | 0.5855 |
| Precision | 0.5434 | 0.3586 | 0.5556 | 0.2848 | 0.6303 |
| Recall | 0.5407 | 0.4316 | 0.5361 | 0.3224 | 0.6039 |
| Accuracy | 0.2299 | 0.0461 | 0.1963 | 0.1146 | 0.3175 |
| BLEU | 0.3885 | 0.2399 | 0.4018 | 0.1806 | 0.4570 |

### B+TF (Time Filtering) — 1517 样本

| 指标 | 总体 | Cat 1 (多跳) | Cat 2 (时序) | Cat 3 (开放) | Cat 4 (单跳) |
|------|------|-------------|-------------|-------------|-------------|
| **F1** | **0.5051** | 0.3654 | 0.5310 | 0.2240 | 0.5748 |
| Precision | 0.5363 | 0.3588 | 0.5561 | 0.2506 | 0.6216 |
| Recall | 0.5300 | 0.4271 | 0.5393 | 0.2867 | 0.5891 |
| Accuracy | 0.2254 | 0.0426 | 0.1981 | 0.0745 | 0.3152 |
| BLEU | 0.3812 | 0.2377 | 0.4051 | 0.1560 | 0.4466 |

### 结果分析

- **B 略优于 B+TF**（F1: 0.5134 vs 0.5051），与论文趋势一致，gpt-4o-mini 模型下差异较小
- **Cat 2 (时序)** 上 B+TF 的 F1 略高（0.5310 vs 0.5286），temporal filtering 有轻微正向效果
- **Cat 3 (开放推理)** 是最难的类别，F1 仅 0.26
- **Cat 4 (单跳)** 表现最好，F1 达 0.59
- B+TF 有 23 条 generation 失败（1517 vs 1540）

## 运行命令

```bash
# 完整运行（含 build）
RUN_ID=locomo_b_btf_full bash scripts/run_locomo_b_btf.sh

# 跳过 build（已有 final_boxes_content.jsonl）
SKIP_BUILD=1 RUN_ID=locomo_b_btf_full bash scripts/run_locomo_b_btf.sh

# 单样本测试
LIMIT_CONVERSATIONS=1 RUN_ID=locomo_b_btf_sample1 bash scripts/run_locomo_b_btf.sh
```

## 文件结构

```
SA-Mem-Research/
├── dataset/locomo10.json                    # 数据集
├── scripts/
│   ├── run_locomo_b_btf.sh                  # 主运行脚本
│   └── retrieve_locomo_b_btf.py             # Retrieval 入口
├── out/locomo_b_btf_full/
│   ├── final_boxes_content.jsonl            # Memory blocks (894 blocks)
│   ├── retrieval_baseline.jsonl             # B retrieval (1540)
│   ├── retrieval_enhanced.jsonl             # B+TF retrieval (1540)
│   ├── generation_results_locomo_baseline.jsonl      # B generation (1540)
│   ├── generation_results_locomo_time_filtering.jsonl # B+TF generation (1517)
│   ├── evaluation_summary_locomo_baseline.json        # B evaluation
│   └── evaluation_summary_locomo_time_filtering.json  # B+TF evaluation
└── source code files...
```

## 修改的源码文件

1. `retrieval/retrieval_impl_locomo.py` — 添加 `graph_person_relations` 参数
2. `generate_impl_locomo.py` — 修复 LIMIT_CONVERSATIONS=-1 的切片问题

---

*报告生成时间: 2026-06-23*
