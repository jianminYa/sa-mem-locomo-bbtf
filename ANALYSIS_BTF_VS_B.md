# B+TF vs B 深度分析：为什么 Temporal Filtering 没有收益

**Date**: 2026-06-24
**Commit**: `7a2af3f`

## 一、核心发现

修正 query embedding cache key 隔离后，B+TF 性能显著下降：

| 指标 | warm (错误复用B) | recheck (正确隔离) | 变化 |
|------|-----------------|-------------------|------|
| F1 | 0.5035 | 0.4879 | -0.016 |
| Hit@5 | 0.8045 | 0.7708 | -0.034 |
| Recall@5 | 0.7416 | 0.7065 | -0.035 |
| Complete-MRR | 0.5531 | 0.5114 | -0.042 |

**结论**：之前的 B+TF 接近 B 是假象——B+TF 实际复用了 B 的原始 question embedding，没有使用 rewritten query。

## 二、Query Embedding Cache Key 问题

### 问题代码

B baseline 和 B+TF enhanced 都使用相同的 cache key：

```python
# B baseline
store.get_vector(f"qa_{user_id}_{q_id}", "question", original_question)

# B+TF enhanced (修改前)
store.get_vector(f"qa_{user_id}_{q_id}", "question", rewritten_query)
#                              ^^^^         ^^^^^^^^
#                              相同key       不同text
```

### EmbeddingStore 缓存逻辑

```python
def get_vector(self, key, field, text, note):
    if key in self.data and field in self.data[key]:
        return self.data[key][field]  # 命中缓存，忽略 text
    vec = self.worker.get_embedding(text)
    self.data.setdefault(key, {})[field] = vec
    return vec
```

### 后果

1. B 先运行，缓存 `qa_0_0 -> original_question_embedding`
2. B+TF 后运行，命中缓存，返回 `original_question_embedding`（忽略 rewritten_query）
3. B+TF 实际使用了 B 的 embedding，temporal filtering 只影响候选池大小

### 修复

```python
# B+TF enhanced (修改后)
import hashlib
query_hash = hashlib.md5(rewritten_query.encode()).hexdigest()[:12]
store.get_vector(
    f"qa_enhanced_{user_id}_{q_id}_{query_hash}",
    "question_rewritten",
    rewritten_query
)
```

## 三、Rewritten Query 质量分析

### Rewritten Query 示例

B+TF 的 query parser 会移除时间短语，生成 "干净" 的 query 用于语义检索：

| 原始 Query | Rewritten Query | 变化 |
|-----------|-----------------|------|
| "What did Mel and her kids paint in their latest project in July 2023?" | "What did Mel and her kids paint in their latest project" | 移除 "in July 2023" |
| "What painting did Melanie show to Caroline on October 13, 2023?" | "What painting did Melanie show to Caroline" | 移除 "on October 13, 2023" |
| "Where did Caroline move from 4 years ago?" | "Where did Caroline move from" | 移除 "4 years ago" |

### 为什么 Rewritten Query 效果更差

1. **时间词包含语义信息**：如 "painting" + "October 13" 比单独 "painting" 更精确
2. **语义空间不匹配**：block embedding 使用原始文本（含时间），query embedding 使用 rewritten（无时间）
3. **过度过滤 + 语义模糊**：temporal filtering 缩小候选池，但 rewritten query 又无法精确匹配

## 四、Temporal-Constrained 子集详细分析

### 统计

| 方法 | Targets Hit | Hit Rate | B更好 | B+TF更好 | 平局 |
|------|-------------|----------|-------|----------|------|
| B | 156/229 | 68.1% | - | - | - |
| B+TF | 133/229 | 58.1% | 35 | 15 | 158 |

### B 更好的例子（temporal filtering 过度过滤）

**Q2**: "What did Mel and her kids paint in their latest project in July 2023?"
- tc=POINT, pool 被缩小到 **3/67** (95% 被过滤)
- B top5: [46, 47, 49, 25, 30] → 命中 target 25 ✓
- B+TF top5: [57, 52, 38] → 未命中 ✗
- **问题**：temporal filtering 过于激进，把正确 block 过滤掉了

**Q12**: "Why did Maria sit with the little girl at the shelter event in February 2023?"
- tc=POINT, pool 被缩小到 **7/86** (92% 被过滤)
- B top5: [129, 130, 172, 143, 127] → 命中 target 127 ✓
- B+TF top5: [192, 136, 138, 175, 150] → 未命中 ✗

### B+TF 更好的例子（temporal filtering 正确过滤）

**Q5**: "What painting did Melanie show to Caroline on October 13, 2023?"
- tc=POINT, pool 被缩小到 **4/67** (94% 被过滤)
- B top5: [3, 35, 46, 49, 30] → 未命中 ✗
- B+TF top5: [57, 58, 59, 60] → 命中 target 58 ✓
- **优势**：temporal filtering 正确定位到时间相关的 blocks

### 模式总结

- **B 更好**：temporal filtering 过度过滤，正确 block 被排除
- **B+TF 更好**：temporal filtering 精确过滤，缩小搜索范围
- **总体**：过度过滤的负面影响 > 精确过滤的正面影响

## 五、检索延迟对比

| 方法 | warm (错误复用B) | recheck (正确隔离) | 变化 |
|------|-----------------|-------------------|------|
| B search | 0.100s | 0.105s | +5% (正常) |
| B+TF search | 0.103s | 2.227s | **+21.6x** |
| B+TF parse | 1.116s | 1.003s | -10% |
| B+TF total | 1.219s | 3.231s | +2.6x |

### B+TF search 延迟暴增原因

1. **之前**：复用 B 的缓存 embedding，只需读取 JSON（0.1s）
2. **现在**：需要为每个 query 计算新 embedding（2.2s）
3. **额外开销**：rewritten query 的 embedding 需要调用 embedding API

## 六、为什么之前 "接近论文"

### 原因

1. **Cache key 碰撞**：B 和 B+TF 使用相同的 `qa_{user_id}_{q_id}` key
2. **B 先运行**：embedding 被缓存到 vector_store JSON
3. **B+TF 命中缓存**：实际上使用了 B 的原始 question embedding
4. **Temporal filtering 独立生效**：只影响候选池大小，不影响 query embedding

### 结果

- B+TF ≈ B（因为 query embedding 相同）
- 看起来 "temporal filtering 没有明显收益"
- 实际上是 "temporal filtering + 原始 query embedding" vs "无 temporal filtering"

## 七、对论文 Table 7 的启示

论文 Table 7 显示 temporal filtering 有收益：

| 方法 | Recall@5 (constrained) |
|------|------------------------|
| No temporal filter | 0.6891 |
| Session-only | 0.7436 |
| Event-only | 0.7500 |
| Union Session & Event | 0.7596 |

### 可能的差异原因

1. **模型差异**：论文可能使用不同的 embedding 模型
2. **Query rewriting 策略**：论文的 rewritten query 可能保留了更多语义信息
3. **Temporal index 精度**：论文的 temporal index 可能更精确
4. **训练数据**：论文的 embedding 模型可能在时间推理任务上训练过

## 八、结论

### 已确认

1. B+TF 之前的结果被 B 的缓存 embedding 污染
2. 修正后 B+TF 性能下降，temporal filtering 没有收益
3. Rewritten query 的 embedding 质量不如原始 query

### 未解决

1. 如何改进 rewritten query 的 embedding 质量
2. 如何平衡 temporal filtering 的精确性和召回率
3. 论文的 temporal filtering 为什么有效

### 建议

1. 保留时间词的一部分语义信息（如 "last year" → "2023"）
2. 使用 hybrid retrieval（temporal filtering + semantic ranking 的加权组合）
3. 训练专门的时间感知 embedding 模型

---

*分析日期: 2026-06-24*
*数据来源: out/locomo_b_btf_recheck_20260624/*
