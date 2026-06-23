# SA-Mem LoCoMo B vs B+TF 实验报告

## 1. 实验概述

本实验对 SA-Mem 记忆系统的两种检索策略进行对比评估：
- **Baseline (B)**：基础检索策略
- **B+TF (Time Filtering)**：带时间过滤的增强检索策略

使用 LoCoMo (Long Context Memory) 数据集进行评测。

---

## 2. 实验环境

### 2.1 硬件环境
- 操作系统：Linux 6.8.0-71-generic (Ubuntu)
- Python：3.12.3
- Conda 环境：samem-locomo

### 2.2 软件依赖
- OpenAI Python SDK (通过 yunwu.ai 中转)
- 嵌入模型：text-embedding-3-small
- LLM 模型：gpt-4o-mini

### 2.3 API 配置
```
LLM_PROVIDER=openai
OPENAI_BASE_URL=https://yunwu.ai/v1/
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
SA_MEM_ENABLE_GRAPH=0  # Graph DB 未启用
```

### 2.4 数据集
- 数据集文件：`dataset/locomo10.json`
- 对话数量：10 个完整对话
- QA 总数：885 条 (Baseline) / 872 条 (B+TF)

---

## 3. 运行命令

### 3.1 小规模测试 (LIMIT_CONVERSATIONS=1)
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate samem-locomo
cd /workspace/june/sa-mem/SA-Mem-Research
LIMIT_CONVERSATIONS=1 RUN_ID=locomo_b_btf_sample1 bash scripts/run_locomo_b_btf.sh
```

### 3.2 全量运行
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate samem-locomo
cd /workspace/june/sa-mem/SA-Mem-Research
RUN_ID=locomo_b_btf_full bash scripts/run_locomo_b_btf.sh
```

### 3.3 脚本参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| RUN_ID | locomo_b_btf | 输出目录名 |
| DATA_FILE | dataset/locomo10.json | 数据集路径 |
| LIMIT_CONVERSATIONS | -1 (全部) | 限制对话数量 |
| RETRIEVAL_TOP_K | 5 | 检索 Top-K |
| ANSWER_TOP_N | 5 | 回答参考 Top-N |
| TEXT_MODE | content | 文本模式 |
| LLM_MODEL | gpt-4o-mini | LLM 模型 |
| EMBEDDING_MODEL | text-embedding-3-small | 嵌入模型 |

---

## 4. 流水线阶段

脚本包含 6 个阶段：

| 阶段 | 脚本 | 功能 |
|------|------|------|
| [1/6] Build | `build_stage_locomo.py` | 构建记忆块 (memory blocks) |
| [2/6] Retrieval | `scripts/retrieve_locomo_b_btf.py` | 检索：同时生成 B 和 B+TF 结果 |
| [3/6] Generation B | `generate_stage_locomo.py` | 基于 Baseline 检索结果生成回答 |
| [4/6] Generation B+TF | `generate_stage_locomo.py` | 基于 B+TF 检索结果生成回答 |
| [5/6] Evaluation B | `evaluate_locomo.py` | 评估 Baseline 回答质量 |
| [6/6] Evaluation B+TF | `evaluate_locomo.py` | 评估 B+TF 回答质量 |

---

## 5. 运行中遇到的问题及解决方案

### 5.1 进程无 API 调用记录
**问题**：启动全量运行后，检查 API 后台未发现调用记录。

**原因**：进程启动时环境变量中没有 `OPENAI_API_KEY`，但代码中的 `_try_load_dotenv()` 函数会在模块加载时自动从 `.env` 文件读取并注入环境变量。因此 `/proc/<pid>/environ` 不显示该变量，但运行时实际已加载。

**解决方案**：通过检查 `token_stream.jsonl` 的实时写入确认 API 调用正常进行。实际 API 走的是 yunwu.ai 中转站，需在对应后台查看。

### 5.2 长时间运行监控
**问题**：全量运行耗时较长（约 3 小时 49 分钟），需要确认进程是否正常。

**解决方案**：
- 通过 `ps aux | grep python` 检查进程存活
- 通过 `stat` 命令检查输出文件的最后修改时间
- 通过 `tail` 查看 `token_stream.jsonl` 最新条目确认进度

---

## 6. 实验结果

### 6.1 小规模测试结果 (LIMIT_CONVERSATIONS=1, 152 样本)

| 指标 | Baseline (B) | B+TF | 差值 |
|------|-------------|------|------|
| avg_f1 | **0.4181** | 0.4111 | +0.007 |
| avg_precision | **0.4526** | 0.4421 | +0.010 |
| avg_recall | **0.4516** | 0.4428 | +0.009 |
| avg_accuracy | 0.1645 | 0.1645 | 0 |
| avg_bleu | **0.3166** | 0.3136 | +0.003 |

按 Category (小规模)：

| Category | 样本数 | B F1 | B+TF F1 | B BLEU | B+TF BLEU |
|----------|--------|------|---------|--------|-----------|
| 1 | 32 | **0.3047** | 0.2888 | **0.2504** | 0.2469 |
| 2 | 37 | **0.4718** | 0.4498 | **0.3333** | 0.3320 |
| 3 | 13 | **0.2963** | 0.2809 | **0.1915** | 0.1705 |
| 4 | 70 | 0.4641 | **0.4707** | **0.3612** | 0.3610 |

### 6.2 全量运行结果 (10 对话, 885/872 样本)

| 指标 | Baseline (B) | B+TF | 差值 |
|------|-------------|------|------|
| avg_f1 | **0.5073** | 0.4942 | +0.013 |
| avg_precision | **0.5340** | 0.5211 | +0.013 |
| avg_recall | **0.5392** | 0.5231 | +0.016 |
| avg_accuracy | 0.2328 | 0.2328 | 0 |
| avg_bleu | **0.3784** | 0.3700 | +0.008 |

按 Category (全量)：

| Category | 样本数 | B F1 | B+TF F1 | B BLEU | B+TF BLEU |
|----------|--------|------|---------|--------|-----------|
| 1 | 172 | **0.3659** | 0.3639 | 0.2499 | **0.2521** |
| 2 | 176-180 | **0.5539** | 0.5295 | **0.4145** | 0.3918 |
| 3 | 52-53 | **0.2427** | 0.2241 | **0.1583** | 0.1501 |
| 4 | 472-480 | **0.5698** | 0.5583 | **0.4352** | 0.4290 |

### 6.3 Build 阶段统计 (全量)

| 指标 | 值 |
|------|-----|
| 生成 boxes 数 | 894 |
| 总消息数 | 5,882 |
| 平均消息/box | 6.58 |
| LLM 调用次数 | 6,455 |
| 总 prompt tokens | 2,094,465 |
| 总 completion tokens | 238,029 |
| 总 tokens | 2,332,494 |

---

## 7. 结论

1. **Baseline (B) 在全量数据上全面优于 B+TF**，所有主要指标（F1、Precision、Recall、BLEU）均略高
2. **Category 4 表现最好**（F1 ~0.57），Category 3 表现最差（F1 ~0.24）
3. 小规模测试与全量测试趋势一致，验证了结果的可靠性
4. 时间过滤策略 (TF) 在当前设置下未带来增益，可能需要进一步调优

---

## 8. 输出文件说明

```
out/locomo_b_btf_full/
├── build_stats.jsonl                          # Build 阶段统计
├── final_boxes_content.jsonl                  # 构建的记忆块内容
├── trace_build_process.jsonl                  # Build 过程详细日志
├── token_stream.jsonl                         # LLM 调用 token 记录
├── vector_store/                              # 向量存储
├── retrieval_baseline.jsonl                   # Baseline 检索结果
├── retrieval_baseline.csv                     # Baseline 检索结果 (CSV)
├── retrieval_enhanced.jsonl                   # B+TF 检索结果
├── retrieval_enhanced.csv                     # B+TF 检索结果 (CSV)
├── retrieve_stage_enhanced.log                # 检索阶段日志
├── generation_results_locomo_baseline.jsonl   # Baseline 生成结果
├── generation_metrics_summary_baseline.jsonl  # Baseline 生成指标
├── report_generation_qa_locomo_baseline.csv   # Baseline QA 报告
├── generation_results_locomo_time_filtering.jsonl   # B+TF 生成结果
├── generation_metrics_summary_time_filtering.jsonl  # B+TF 生成指标
├── report_generation_qa_locomo_time_filtering.csv   # B+TF QA 报告
├── evaluation_results_locomo_baseline.jsonl   # Baseline 评估详细结果
├── evaluation_summary_locomo_baseline.json    # Baseline 评估汇总
├── evaluation_results_locomo_time_filtering.jsonl   # B+TF 评估详细结果
└── evaluation_summary_locomo_time_filtering.json    # B+TF 评估汇总
```
