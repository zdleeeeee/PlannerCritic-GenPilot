# CriticPilot-Lite 最新优化总结

## 1. 当前目标

本项目基于 GenPilot 的 TTPO 流程加入 CriticPilot-Lite 机制，用于比较原始 baseline 与带轻量批判机制的 critic 版本。

目前脚本已经支持：

1. 自动按 batch 准备 prompts。
2. 自动运行 error analysis。
3. 自动运行 baseline 模式。
4. 自动运行 critic 模式。
5. 自动生成每个 batch 的比较报告。
6. 自动生成整体比较报告。

默认实验设置为：

```text
4 个 batch
每个 batch 5 个 prompts
总共测试 original_prompts_299.txt 的前 20 条 prompts
```

运行命令：

```bash
uv run python run_baseline.py
```

---

## 2. 输入 prompt 配置

修改文件：

- `run_baseline.py`
- `data/original_prompts.txt`

现在 `run_baseline.py` 会自动从：

```text
data/original_prompts_299.txt
```

按 batch 读取 prompts。

默认：

```python
BATCH_COUNT = 4
BATCH_SIZE = 5
```

对应关系：

```text
Batch 1: original_prompts_299.txt 第 0-4 行
Batch 2: original_prompts_299.txt 第 5-9 行
Batch 3: original_prompts_299.txt 第 10-14 行
Batch 4: original_prompts_299.txt 第 15-19 行
```

每个 batch 开始时，脚本会把当前 5 条 prompt 写入：

```text
data/original_prompts.txt
```

并清空：

```text
data/ori_img/*.png
```

然后重新生成当前 batch 的原始图像。

---

## 3. 运行输出结构

输出根目录：

```text
results/baseline/compare/
```

结构如下：

```text
results/baseline/compare/
├── batch_01/
│   ├── analysis/
│   ├── baseline/
│   ├── critic/
│   ├── compare_summary.json
│   └── compare_summary.md
├── batch_02/
│   ├── analysis/
│   ├── baseline/
│   ├── critic/
│   ├── compare_summary.json
│   └── compare_summary.md
├── batch_03/
├── batch_04/
├── compare_summary_all.json
└── compare_summary_all.md
```

每个 batch 内：

- `analysis/`：共享 error analysis 结果。
- `baseline/`：原始 baseline TTPO 输出。
- `critic/`：CriticPilot-Lite TTPO 输出。
- `compare_summary.md`：当前 batch 的指标比较。
- `compare_summary.json`：当前 batch 的结构化指标。

整体汇总：

```text
results/baseline/compare/compare_summary_all.md
results/baseline/compare/compare_summary_all.json
```

---

## 4. Error Analysis 自动化

修改文件：

- `run_baseline.py`
- `genpilot/error_analysis/caption.py`
- `genpilot/error_analysis/check_captions.py`
- `genpilot/error_analysis/question.py`
- `genpilot/error_analysis/qa.py`
- `genpilot/error_analysis/error_integration.py`
- `genpilot/error_analysis/error_mapping.py`

现在不需要手动运行：

```bash
./error_analysis_pipline.sh
```

`run_baseline.py` 会自动执行：

```text
decom_prompt.py
reform_decomposed_prompt.py
generate_image.py
caption.py
check_captions.py
reform_jsonl.py
question.py
qa.py
error_integration.py
reform_jsonl.py
error_mapping.py
reform_jsonl.py
```

每个 batch 的分析结果写到：

```text
results/baseline/compare/batch_xx/analysis/
```

TTPO 需要的关键文件是：

```text
errors_reformed.jsonl
find_error.jsonl
questions.jsonl
```

---

## 5. Error Analysis 并发控制

为避免 API 429 限流，给多个 error analysis 脚本新增了：

```bash
--workers
```

涉及脚本：

```text
genpilot/error_analysis/caption.py
genpilot/error_analysis/check_captions.py
genpilot/error_analysis/question.py
genpilot/error_analysis/qa.py
genpilot/error_analysis/error_integration.py
genpilot/error_analysis/error_mapping.py
```

默认：

```python
ERROR_ANALYSIS_WORKERS = BATCH_SIZE
```

也就是默认每个 batch 内最多 5 个 API worker。

如果遇到 429，可以降低：

```bash
ERROR_ANALYSIS_WORKERS=1 uv run python run_baseline.py
```

---

## 6. Baseline / Critic 双模式

修改文件：

- `run_baseline.py`
- `genpilot/ttpo.py`

`ttpo.py` 新增参数：

```bash
--critic_mode baseline
--critic_mode critic
--analysis_folder <path>
```

### baseline 模式

关闭：

```text
ErrorAggregator
L1
L3
history
```

也就是尽量保持原始 GenPilot TTPO 控制逻辑。

### critic 模式

开启：

```text
ErrorAggregator
L1 局部重试
L3 全局重置
history
```

`run_baseline.py` 每个 batch 会先跑 baseline，再跑 critic，并用同一份 error analysis 输入做公平比较。

---

## 7. Error Aggregator

修改文件：

- `genpilot/utils_all/error_taxonomy.py`
- `genpilot/ttpo.py`

新增函数：

```python
aggregate_errors(errors, sentence="", max_errors_per_fragment=5)
count_error_items(errors)
```

作用：

1. 支持字符串、列表、字典、嵌套结构。
2. 过滤空错误、`None`、`NO ERROR`。
3. 去重重复 error。
4. 调用已有 `classify_error()` 分类。
5. 按优先级保留代表错误。
6. 每个 fragment 默认最多保留 5 条 error。

聚合优先级：

```text
object_error
counting_error
text_rendering_error
spatial_relation_error
action_error
attribute_error
style_error
other_error
```

在 critic 模式下，`process_case()` 会在调用 `process_single_j()` 前执行：

```python
aggregate_case_errors(image_data[i], log_file)
```

日志示例：

```text
[ErrorAggregator] fragment=1, errors=18->5
```

---

## 8. L1 局部重试

修改文件：

- `genpilot/ttpo.py`

### 触发条件

当前轮所有候选 prompt 评分后：

```python
max(scores) < l1_score_threshold
```

当前默认：

```text
L1_SCORE_THRESHOLD=13.0
MAX_L1_RETRIES=2
```

总分范围是 3-15，所以阈值 13 表示：

```text
三项平均分 < 4.33 时触发 L1
```

### 动作

L1 触发后会重新生成更多候选：

```python
num_candidates * 2
```

并重新生成图像与评分。

日志示例：

```text
[L1Check] max_score=12.0, threshold=13.0, will_retry=True
[L1] Trigger local retry 1/2: max_score=12.0, threshold=13.0
[L1] Retry complete: new_max_score=14.0
```

---

## 9. L3 全局重置

修改文件：

- `genpilot/ttpo.py`

### 触发条件

连续若干轮分数变化小于阈值：

```python
abs(current_best_score - last_best_score) < l3_score_delta
```

当前默认：

```text
L3_STAGNATION_WINDOW=2
L3_SCORE_DELTA=1.0
NUM_ITERATIONS=4
```

也就是连续 2 轮分数变化小于 1 分时触发 L3。

### 动作

触发后清空：

```python
history = []
modify_history = []
```

保留当前 prompt，不回滚到原始 prompt。

日志示例：

```text
[L3Check] current_best=13.0, last_best=13.0, score_change=0.0, stagnation=2/2
[L3] Trigger global reset: stagnation=2, max_score=13.0
```

---

## 10. TTPO 触发参数

`ttpo.py` 新增参数：

```bash
--l1_score_threshold
--max_l1_retries
--l3_stagnation_window
--l3_score_delta
```

`run_baseline.py` 默认读取环境变量：

```text
NUM_ITERATIONS=4
L1_SCORE_THRESHOLD=13.0
MAX_L1_RETRIES=2
L3_STAGNATION_WINDOW=2
L3_SCORE_DELTA=1.0
```

如果要更激进：

```bash
L1_SCORE_THRESHOLD=14 NUM_ITERATIONS=5 uv run python run_baseline.py
```

如果要更保守：

```bash
L1_SCORE_THRESHOLD=12 L3_STAGNATION_WINDOW=3 uv run python run_baseline.py
```

---

## 11. 候选 prompt 与修改句成对去重

修改文件：

- `genpilot/ttpo.py`

新增函数：

```python
_dedupe_candidate_pairs(candidate_prompts, modified_sentences)
```

原代码分别对 `candidate_prompts` 和 `modified_sentence` 做：

```python
list(set(...))
```

这会打乱索引对应关系。

现在改为按 pair 去重，避免：

```python
modified_sentence[best_prompt_index]
```

取错修改句。

---

## 12. 聚类数量保护

修改文件：

- `genpilot/ttpo.py`

原来：

```python
labels = cluster_prompts(candidate_prompts, num_clusters)
```

如果候选数少于 `num_clusters`，可能导致 KMeans 报错。

现在：

```python
effective_num_clusters = max(1, min(num_clusters, len(candidate_prompts)))
labels = cluster_prompts(candidate_prompts, effective_num_clusters)
```

---

## 13. MLLM JSON 解析修复

修改文件：

- `genpilot/prompts/rate_feedback.txt`
- `genpilot/utils_all/scorer.py`
- `tests/test_scorer_json.py`

### Prompt 修复

原评分 prompt 中的 JSON 示例不是严格合法 JSON，存在：

1. 占位符。
2. 尾随逗号。
3. reason 未使用字符串。
4. score 容易诱导成数组。

现在 `rate_feedback.txt` 明确要求：

```text
Return ONLY one fenced JSON block.
Each score must be an integer from 1 to 5.
Do not use arrays for scores.
Do not use placeholders.
Do not use trailing commas.
```

### Parser 修复

新增：

```python
parse_rating_response(response_text)
```

支持解析：

1. fenced JSON。
2. 纯 JSON。
3. 文本中夹带 JSON。
4. 带尾随逗号的 JSON。
5. score 是字符串或 `[4]` 的情况。

如果多次失败，会返回 fallback 低分，避免卡死。

---

## 14. 评分重试修复

修改文件：

- `genpilot/utils_all/scorer.py`

原来：

```python
while rate_result is None or count_num < 2:
```

即使评分成功也会额外请求两次；如果一直失败，还可能造成长时间重复调用。

现在改为：

```python
while rate_result is None and count_num < 2:
```

并在最终失败时使用 fallback rating。

---

## 15. 比较指标解析 bug 修复

修改文件：

- `run_baseline.py`
- `tests/test_compare_metrics.py`

原先比较脚本把：

```text
np.float64
```

里的 `64` 误识别成分数，导致报告里出现不可能的：

```text
64.000
```

现在新增：

```python
parse_score_line(line)
```

只解析真正的 score 行：

```text
[np.float64(12.0), np.float64(11.0), np.float64(13.0)]
[12.0, 11.0, 13.0]
```

并忽略 history/reason 中的 `np.float64(...)`。

---

## 16. 当前 batch_01 最新结果

最新 batch_01 报告：

```text
results/baseline/compare/batch_01/compare_summary.md
```

结果：

```text
| Metric | Baseline | Critic | Delta |
| --- | ---: | ---: | ---: |
| Max candidate score | 15.000 | 15.000 | 0.000 |
| Avg round-best score | 13.500 | 13.605 | 0.105 |
| Scored rounds | 28 | 38 | 10 |
| Final prompts | 4 | 5 | 1 |
| L1 triggers | 0 | 13 | 13 |
| L3 triggers | 0 | 4 | 4 |
```

### 解释

Critic 最大候选分与 baseline 持平：

```text
15.0 vs 15.0
```

Critic 平均 round-best score 小幅提升：

```text
13.500 -> 13.605
Delta = +0.105
```

Critic 触发：

```text
L1: 13 次
L3: 4 次
```

说明当前参数下 L1/L3 已经实际参与优化。

---

## 17. Batch_01 分 case 结果

### Baseline

```text
case 0: rounds 4,  max 13.0, avg 12.50
case 1: rounds 1,  max 15.0, avg 15.00
case 2: rounds 8,  max 15.0, avg 12.875
case 3: rounds 5,  max 15.0, avg 13.80
case 4: rounds 10, max 15.0, avg 14.10
```

### Critic

```text
case 0: rounds 11, max 15.0, avg 13.273, L1 4, L3 2
case 1: rounds 1,  max 15.0, avg 15.00,  L1 0, L3 0
case 2: rounds 8,  max 14.0, avg 13.25,  L1 3, L3 1
case 3: rounds 8,  max 15.0, avg 13.75,  L1 4, L3 1
case 4: rounds 10, max 15.0, avg 14.00,  L1 2, L3 0
```

### 观察

明显提升：

```text
case 0: max 13 -> 15, avg 12.50 -> 13.273
```

小幅提升：

```text
case 2: avg 12.875 -> 13.25
```

基本持平或略低：

```text
case 3: avg 13.80 -> 13.75
case 4: avg 14.10 -> 14.00
```

---

## 18. 如何解读当前结果

当前 batch_01 可以说明：

1. CriticPilot-Lite 机制确实生效。
2. L1/L3 在新参数下已经触发。
3. Critic 在 batch_01 上有轻微正向收益。
4. case 0 的 counting/visibility 场景提升最明显。
5. 这个提升有额外计算成本：scored rounds 从 28 增加到 38。

谨慎表述：

> CriticPilot-Lite shows a weak but positive improvement signal in batch_01. It improves the average round-best score from 13.50 to 13.61 and activates 13 L1 retries and 4 L3 resets. The strongest improvement appears in the cabbage counting/visibility case, while other cases are mostly tied or only marginally changed.

---

## 19. 当前还需要注意的问题

### 19.1 API 429 限流

TTPO 阶段仍可能出现：

```text
Rate limit exceeded on tokens
```

原因是 batch 内并发、candidate 数量、L1 重试和多模态 QA 评分叠加。

如果遇到 429，建议使用：

```bash
CASE_WORKERS=2 SENTENCE_WORKERS=1 ERROR_ANALYSIS_WORKERS=3 uv run python run_baseline.py
```

更稳：

```bash
CASE_WORKERS=1 SENTENCE_WORKERS=1 ERROR_ANALYSIS_WORKERS=1 uv run python run_baseline.py
```

### 19.2 Avg round-best score 不是最终图质量

`Avg round-best score` 衡量的是 TTPO 过程中每轮最佳候选的平均分，不等于最终 prompt 重新生成图后的最终质量。

更严格的最终比较需要：

```text
baseline prompt_final.txt -> 重新生成最终图 -> 评分
critic prompt_final.txt   -> 重新生成最终图 -> 评分
```

### 19.3 L1/L3 有额外成本

Critic 的 scored rounds 增加，说明机制带来更多图像生成和 MLLM 评分调用。

需要在报告中同时讨论质量提升和计算成本。

---

## 20. 已新增/修改的测试

新增或更新：

```text
tests/test_error_taxonomy.py
tests/test_scorer_json.py
tests/test_compare_metrics.py
```

测试内容：

1. 错误分类和 ErrorAggregator。
2. MLLM JSON 解析。
3. 比较指标分数解析。

已验证过：

```bash
uv run python tests/test_error_taxonomy.py
uv run python tests/test_scorer_json.py
uv run python tests/test_compare_metrics.py
uv run python -m py_compile run_baseline.py genpilot/ttpo.py genpilot/utils_all/scorer.py
```

---

## 21. 核心修改文件列表

### 实验入口

```text
run_baseline.py
```

### TTPO 主流程

```text
genpilot/ttpo.py
```

### Error Aggregator

```text
genpilot/utils_all/error_taxonomy.py
```

### Scoring / JSON parser

```text
genpilot/utils_all/scorer.py
genpilot/prompts/rate_feedback.txt
```

### Error analysis 并发控制

```text
genpilot/error_analysis/caption.py
genpilot/error_analysis/check_captions.py
genpilot/error_analysis/question.py
genpilot/error_analysis/qa.py
genpilot/error_analysis/error_integration.py
genpilot/error_analysis/error_mapping.py
```

### 配置示例

```text
.env.example
```

### 测试

```text
tests/test_error_taxonomy.py
tests/test_scorer_json.py
tests/test_compare_metrics.py
```
