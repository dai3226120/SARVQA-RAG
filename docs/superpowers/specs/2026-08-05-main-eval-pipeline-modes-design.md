# main_eval.py 流水线模式配置设计

日期：2026-08-05
状态：已批准（方案 A）

## 背景

`benchmark/main_eval.py` 当前固定执行 预测 → 评估 → 分析 三步流水线。需求：可通过配置选择只跑部分步骤：

- **full**：预测 + 评估 + 分析（现状行为，更新 Excel 统计表）
- **eval**：评估 + 分析（跳过预测，需引用之前的预测结果）
- **analyze**：仅分析（跳过预测与评估，需引用之前的评估结果）

配置方式沿用现有风格：文件顶部模块级常量（与 `MODEL_KEY` 一致），不支持命令行参数。

## 配置项

在 `main_eval.py` 文件顶部（`MODEL_KEY` 附近）新增：

```python
# ====================== 流水线模式配置 ======================
# "full"   : 预测 + 评估 + 分析（完整流水线，结束后更新 Excel 统计表）
# "eval"   : 评估 + 分析（跳过预测，需引用之前的预测结果）
# "analyze": 仅分析（跳过预测与评估，需引用之前的评估结果）
PIPELINE_MODE = "full"
RESULT_TIMESTAMP = ""  # eval/analyze 必填：引用 RESULT_DIR/<模型类型>/<文件标签>/<时间戳>/ 下的历史结果
```

## 历史结果定位

引用目录 = `RESULT_DIR / model_type / file_tag / RESULT_TIMESTAMP`：

- `eval` 模式：取引用目录下 `{file_tag}_{DATASET_TAG}_predicted_question_latest.csv`
- `analyze` 模式：取引用目录下 `{file_tag}_{DATASET_TAG}_benchmark_latest.csv`

## 启动校验

新增校验函数（如 `validate_pipeline_mode()`），在 `main()` 开始时执行，任一失败立即报错退出、不进入流水线：

1. `PIPELINE_MODE` 不在 `("full", "eval", "analyze")` 中
2. `eval`/`analyze` 模式下 `RESULT_TIMESTAMP` 为空
3. 引用目录不存在（报错时列出该模型实际存在的历史时间戳目录，方便修改配置）
4. 引用目标文件不存在

## 流程控制（main() 分支）

| 模式 | 执行 | 输出目录 | Excel |
|---|---|---|---|
| `full` | 预测 → 评估 → 分析（现状不变） | 新建时间戳目录 | 更新 |
| `eval` | 评估（读引用预测结果）→ 分析 | 新建时间戳目录（不污染历史结果） | 跳过 |
| `analyze` | 分析（读引用评估结果） | 新建时间戳目录 | 跳过 |

- 每次运行都生成自己的新时间戳输出目录，写入评估/分析产物与图表
- 启动横幅与配置打印中显示当前模式与引用的历史运行时间戳
- `full` 模式行为与现状完全一致

## 不改动的内容

- `run_prediction` / `run_benchmark` / `run_analysis` 三个函数签名不变
- `core` / `models` / `config` 包零改动
- 仅修改 `benchmark/main_eval.py` 一个文件

## 错误处理

- 校验失败：打印明确中文错误信息 + 可用时间戳列表，`main()` 直接 return
- 流水线中途步骤失败：沿用现有逻辑（打印失败信息、终止流程），`full` 模式下仍调用 `update_excel_summary` 记录失败（与现状一致）；`eval`/`analyze` 模式不调用
