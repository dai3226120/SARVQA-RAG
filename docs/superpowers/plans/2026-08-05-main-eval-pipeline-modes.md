# main_eval.py 流水线模式配置 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `benchmark/main_eval.py` 增加流水线模式配置，支持 full（预测+评估+分析）/ eval（评估+分析）/ analyze（仅分析）三种模式，非 full 模式通过 `RESULT_TIMESTAMP` 引用历史运行结果。

**Architecture:** 在文件顶部 `MODEL_KEY` 之后新增 `PIPELINE_MODE` / `RESULT_TIMESTAMP` 两个常量；新增 `resolve_pipeline_mode()` 校验函数（模式合法性、时间戳必填、引用目录/文件存在性）；`main()` 按模式走三个分支，`full` 分支为现有代码原样保留并更新 Excel，`eval`/`analyze` 分支跳过预测/评估且不更新 Excel。每次运行仍生成独立的新时间戳输出目录。

**Tech Stack:** Python 3（pandas / matplotlib / requests），无测试框架（本项目无 pytest 基础设施，验证方式为真实数据运行脚本）。

## Global Constraints

- 仅修改 `benchmark/main_eval.py` 一个文件；`core` / `models` / `config` 包零改动
- `run_prediction` / `run_benchmark` / `run_analysis` 三个函数签名不变，`main()` 中 `full` 分支的行为与现状完全一致（包括失败时仍调用 `update_excel_summary` 记录失败）
- 配置方式为文件顶部模块级常量（与 `MODEL_KEY` 风格一致），不引入命令行参数
- 常量命名与注释用中文注释、英文标识符，与现有代码风格一致
- 每次运行（含 eval/analyze 模式）生成自己的新时间戳输出目录，不写入被引用的历史目录
- 执行验证命令时必须从项目根目录 `C:\vibe coding\SARVQA-RAG-trae` 运行
- 注意：`eval` 模式的评估步骤会调用豆包语义匹配 API（`benchmark/core/benchmarker.py` 中 `call_chat_api`），验证该模式会产生 API 调用

---

### Task 1: 流水线模式常量 + 校验函数 + main() 接入

**Files:**
- Modify: `benchmark/main_eval.py`

**Interfaces:**
- Produces:
  - 模块常量 `PIPELINE_MODE`（默认 `"full"`）、`RESULT_TIMESTAMP`（默认 `""`）
  - `resolve_pipeline_mode(model_type: str, file_tag: str) -> tuple | None`：返回 `(mode, ref_file)`；`mode` ∈ `("full", "eval", "analyze")`；`ref_file` 为 eval/analyze 模式解析出的历史结果文件绝对路径，full 模式为 `None`；校验失败打印错误并返回 `None`
  - `_print_available_timestamps(base_dir: str) -> None`：打印历史时间戳目录列表
- Consumes: 现有 `cfg.path_config.RESULT_DIR`、`cfg.get_file_tag()`、`datetime`、`os`（均已导入/可用）

- [ ] **Step 1: 新增流水线模式常量**

在 `benchmark/main_eval.py` 第 68 行 `MODEL_KEY = "qwen37-plus"` 之后（`_MODEL_REGISTRY` 之前）插入：

```python

# ====================== 流水线模式配置 ======================
# 可选模式:
#   "full"   : 预测 + 评估 + 分析（完整流水线，结束后更新 Excel 统计表）
#   "eval"   : 评估 + 分析（跳过预测，需引用之前的预测结果）
#   "analyze": 仅分析（跳过预测与评估，需引用之前的评估结果）
PIPELINE_MODE = "full"
RESULT_TIMESTAMP = ""  # eval/analyze 必填：引用 RESULT_DIR/<模型类型>/<文件标签>/<时间戳>/ 下的历史结果

_PIPELINE_MODES = ("full", "eval", "analyze")
```

- [ ] **Step 2: 新增校验函数**

在 `# ====================== 步骤 1：模型预测 ======================`（`run_prediction` 定义）之前插入：

```python
# ====================== 流水线模式解析与校验 ======================
def _print_available_timestamps(base_dir: str):
    """打印 base_dir 下可用的历史运行时间戳目录（供用户修正 RESULT_TIMESTAMP）"""
    if not os.path.isdir(base_dir):
        print(f"   目录不存在: {base_dir}")
        return
    timestamps = sorted(
        (d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))),
        reverse=True,
    )
    if timestamps:
        print(f"   可用历史时间戳: {timestamps}")
    else:
        print(f"   目录 {base_dir} 下没有历史运行目录")


def resolve_pipeline_mode(model_type: str, file_tag: str):
    """校验并解析流水线模式配置。

    返回 (mode, ref_file):
        - mode 为 "full" / "eval" / "analyze"
        - ref_file 为 eval/analyze 模式下引用的历史结果文件，full 模式下为 None
    校验失败时打印错误信息并返回 None（调用方应终止流程）。
    """
    if PIPELINE_MODE not in _PIPELINE_MODES:
        print(f"❌ 无效的 PIPELINE_MODE: {PIPELINE_MODE}（可选: {' / '.join(_PIPELINE_MODES)}）")
        return None

    if PIPELINE_MODE == "full":
        return ("full", None)

    # eval / analyze 模式：解析引用的历史结果
    base_dir = os.path.join(cfg.path_config.RESULT_DIR, model_type, file_tag)
    if not RESULT_TIMESTAMP:
        print(f"❌ {PIPELINE_MODE} 模式必须指定 RESULT_TIMESTAMP（引用之前某次运行的结果）")
        _print_available_timestamps(base_dir)
        return None

    ref_dir = os.path.join(base_dir, RESULT_TIMESTAMP)
    if not os.path.isdir(ref_dir):
        print(f"❌ 引用目录不存在: {ref_dir}")
        _print_available_timestamps(base_dir)
        return None

    if PIPELINE_MODE == "eval":
        ref_file = os.path.join(ref_dir, f"{file_tag}_{DATASET_TAG}_predicted_question_latest.csv")
    else:  # analyze
        ref_file = os.path.join(ref_dir, f"{file_tag}_{DATASET_TAG}_benchmark_latest.csv")

    if not os.path.isfile(ref_file):
        print(f"❌ 引用文件不存在: {ref_file}")
        files = os.listdir(ref_dir)
        if files:
            print(f"   目录内可用文件: {files}")
        return None

    return (PIPELINE_MODE, ref_file)
```

- [ ] **Step 3: 修改 `main()` —— 模式解析 + full 分支**

将 `main()` 中现有 `entry = _MODEL_REGISTRY.get(MODEL_KEY)` 与 `if not entry:` 之间的 `model_type`/`file_tag` 赋值之后、`# ---- 生成统一时间戳和输出目录 ----` 之前，插入模式解析；并在 `print_separator(char="-")` 之后、`# ---- 三步流水线 ----` 之前，把流水线整体包进 `if mode == "full":` 分支，缩进现有三步流水线代码（其内容原样保留，含失败时调用 `update_excel_summary` 的两处）。具体改动：

```python
    model_type = entry["model_type"]
    file_tag = entry["file_tag"]

    # ---- 校验流水线模式（full/eval/analyze）----
    resolved = resolve_pipeline_mode(model_type, file_tag)
    if resolved is None:
        return
    mode, ref_file = resolved
```

在 `# ---- 生成统一时间戳和输出目录 ----` 之后、`print(f"\n📋 当前配置:")` 块中，`print(f"   - 模型: {MODEL_KEY}")` 之前插入两行：

```python
    mode_desc = {"full": "预测 + 评估 + 分析", "eval": "评估 + 分析（跳过预测）",
                 "analyze": "仅分析（跳过预测与评估）"}[mode]
    print(f"   - 流水线模式: {mode}（{mode_desc}）")
    if ref_file:
        print(f"   - 引用历史结果: {ref_file}")
```

在 `print_separator(char="-")` 之后将三步流水线改为：

```python
    print_separator(char="-")

    if mode == "full":
        # ---- 三步流水线 ----
        print("\n" + "=" * 80)
        print("🚀 开始执行完整评估流程")
        print("=" * 80)

        pred_result = run_prediction(MODEL_KEY, output_dir)
        if not pred_result["success"]:
            print("❌ 模型预测失败，终止流程")
            update_excel_summary(timestamp, model_type, file_tag, output_dir,
                                 pred_result, {"success": False, "stats": {}},
                                 {"success": False, "stats": {}})
            return
        print("\n" + "=" * 80)

        bench_result = run_benchmark(file_tag, output_dir, pred_result.get("output_file"))
        if not bench_result["success"]:
            print("❌ 指标评估失败，终止流程")
            update_excel_summary(timestamp, model_type, file_tag, output_dir,
                                 pred_result, bench_result,
                                 {"success": False, "stats": {}})
            return
        print("\n" + "=" * 80)

        analysis_result = run_analysis(file_tag, output_dir, bench_result.get("output_file"))

        # ---- 更新 Excel 统计表 ----
        update_excel_summary(timestamp, model_type, file_tag, output_dir,
                             pred_result, bench_result, analysis_result)

        if analysis_result["success"]:
            print("\n" + "=" * 80)
            print("🎉 完整评估流程结束！")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("⚠️ 评估流程结束（结果分析失败）")
            print("=" * 80)
```

（`elif mode == "eval":` 分支留到 Task 2 添加。本任务结束时 `main()` 中 mode 为 eval/analyze 时不会走到任何分支，但由于校验阶段已在 eval/analyze 带合法配置时会通过，Task 2 必须紧接着完成。）

- [ ] **Step 4: 验证无效模式报错退出**

```bash
cd "C:\vibe coding\SARVQA-RAG-trae"
sed -i 's/^PIPELINE_MODE = "full"/PIPELINE_MODE = "bad"/' benchmark/main_eval.py
python benchmark/main_eval.py
```
Expected: 打印 `❌ 无效的 PIPELINE_MODE: bad（可选: full / eval / analyze）` 后立即退出，**未**出现 "模型预测" 字样（无任何 API 调用）。

- [ ] **Step 5: 验证 eval 模式缺时间戳报错**

```bash
sed -i 's/^PIPELINE_MODE = "bad"/PIPELINE_MODE = "eval"/' benchmark/main_eval.py
python benchmark/main_eval.py
```
Expected: 打印 `❌ eval 模式必须指定 RESULT_TIMESTAMP` 且随后列出 `可用历史时间戳: ['20260715_224359', '20260715_223935', ...]`（qwen37-plus 的历史目录），立即退出，无 API 调用。

- [ ] **Step 6: 验证引用目录/文件不存在报错**

```bash
sed -i 's/^RESULT_TIMESTAMP = ""/RESULT_TIMESTAMP = "99999999_000000"/' benchmark/main_eval.py
python benchmark/main_eval.py
```
Expected: 打印 `❌ 引用目录不存在: .../qwen37-plus/qwen37-plus/99999999_000000` 并列出可用时间戳，立即退出。

```bash
sed -i 's/^RESULT_TIMESTAMP = "99999999_000000"/RESULT_TIMESTAMP = "20260715_224359"/' benchmark/main_eval.py
python benchmark/main_eval.py
```
Expected: 校验通过（引用目录与文件均存在），但 Task 2 尚未添加 eval 分支，`main()` 打印配置后无分支执行、脚本正常结束。此行为将在 Task 2 被替换为真正的 评估→分析 流程，故不做断言。

- [ ] **Step 7: 还原配置并提交**

⚠️ 此时实现尚未提交，**切勿使用 `git checkout -- benchmark/main_eval.py` 还原**（会把整个任务的未提交改动一并还原）。用反向 sed 还原两个常量：

```bash
sed -i 's/^PIPELINE_MODE = "eval"/PIPELINE_MODE = "full"/' benchmark/main_eval.py
sed -i 's/^RESULT_TIMESTAMP = "20260715_224359"/RESULT_TIMESTAMP = ""/' benchmark/main_eval.py
grep -n 'PIPELINE_MODE\|RESULT_TIMESTAMP' benchmark/main_eval.py
git add benchmark/main_eval.py docs/superpowers/plans/2026-08-05-main-eval-pipeline-modes.md
git commit -m "feat(benchmark): main_eval 支持流水线模式配置与历史结果引用校验"
```
Expected: `grep` 显示 `PIPELINE_MODE = "full"`、`RESULT_TIMESTAMP = ""`（还原后默认值），提交成功。

### Task 2: main() 增加 eval / analyze 分支

**Files:**
- Modify: `benchmark/main_eval.py`

**Interfaces:**
- Consumes: Task 1 的 `resolve_pipeline_mode`（返回 `(mode, ref_file)`）、`PIPELINE_MODE` / `RESULT_TIMESTAMP` 常量、`run_benchmark(file_tag, output_dir, predicted_csv)`、`run_analysis(file_tag, output_dir, benchmark_csv)`（签名均未变）
- Produces: `main()` 完整支持三种模式；eval/analyze 模式不调用 `update_excel_summary`

- [ ] **Step 1: 添加 eval / analyze 分支**

将 Task 1 中 `if mode == "full":` 分支的结束处（`🎉 完整评估流程结束！` 打印块的 `print("=" * 80)` 之后、与 `main()` 末尾对齐的位置）追加：

```python

    elif mode == "eval":
        # ---- 评估 + 分析（跳过预测，不更新 Excel）----
        print("\n" + "=" * 80)
        print("🚀 开始执行 评估→分析 流程（跳过预测）")
        print("=" * 80)

        bench_result = run_benchmark(file_tag, output_dir, ref_file)
        if not bench_result["success"]:
            print("❌ 指标评估失败，终止流程")
            return
        print("\n" + "=" * 80)

        analysis_result = run_analysis(file_tag, output_dir, bench_result.get("output_file"))

        if analysis_result["success"]:
            print("\n" + "=" * 80)
            print("🎉 评估→分析流程结束！")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("⚠️ 评估→分析流程结束（结果分析失败）")
            print("=" * 80)

    else:  # analyze
        # ---- 仅分析（跳过预测与评估，不更新 Excel）----
        print("\n" + "=" * 80)
        print("🚀 开始执行 仅分析 流程（跳过预测与评估）")
        print("=" * 80)

        analysis_result = run_analysis(file_tag, output_dir, ref_file)

        if analysis_result["success"]:
            print("\n" + "=" * 80)
            print("🎉 仅分析流程结束！")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("⚠️ 仅分析流程结束（结果分析失败）")
            print("=" * 80)
```

- [ ] **Step 2: 语法检查**

```bash
cd "C:\vibe coding\SARVQA-RAG-trae"
python -m py_compile benchmark/main_eval.py
```
Expected: 无输出、退出码 0。

- [ ] **Step 3: 端到端验证 analyze 模式（离线，无 API 调用）**

```bash
sed -i 's/^PIPELINE_MODE = "full"/PIPELINE_MODE = "analyze"/' benchmark/main_eval.py
sed -i 's/^RESULT_TIMESTAMP = ""/RESULT_TIMESTAMP = "20260715_224359"/' benchmark/main_eval.py
ls -la --time-style=full-iso "benchmark/result/eval_summary.xlsx" | awk '{print $6, $7}'   # 记录本次运行前的 mtime
python benchmark/main_eval.py
```
Expected: 打印 `🚀 开始执行 仅分析 流程`、`步骤 3/3: 结果分析`、`📁 图表已保存至: .../qwen37-plus/qwen37-plus/<新时间戳>/plots`、`🎉 仅分析流程结束！`；无 "模型预测" 字样。

```bash
ls "benchmark/result/qwen37-plus/qwen37-plus/" | head -5
ls "benchmark/result/qwen37-plus/qwen37-plus/<新时间戳>/plots" | head -5
```
Expected: 出现新的时间戳目录，`plots` 内有生成的图表文件（如 `*_confidence.png` 等）。

- [ ] **Step 4: 端到端验证 eval 模式（含语义匹配 API 调用，耗时较长）**

```bash
sed -i 's/^PIPELINE_MODE = "analyze"/PIPELINE_MODE = "eval"/' benchmark/main_eval.py
python benchmark/main_eval.py
```
Expected: 打印 `🚀 开始执行 评估→分析 流程（跳过预测）`、`步骤 2/3: 指标评估`（输入为引用目录的 `qwen37-plus_val_predicted_question_latest.csv`）、`步骤 3/3: 结果分析`、`🎉 评估→分析流程结束！`；新的时间戳目录中出现 `qwen37-plus_val_benchmark_latest.csv` 与 `plots/`。

- [ ] **Step 5: 验证 eval/analyze 模式不更新 eval_summary.xlsx**

```bash
ls -la --time-style=full-iso "benchmark/result/eval_summary.xlsx" | awk '{print $6, $7}'
```
与 Step 3 记录的值对比：Expected 完全一致（full 模式才写 Excel）。若期间有人运行过 full 模式则该文件 mtime 已变，属正常，不阻塞本步验收。

- [ ] **Step 6: 还原配置并提交**

⚠️ 此时实现尚未提交，**切勿使用 `git checkout -- benchmark/main_eval.py` 还原**（会把整个任务的未提交改动一并还原）。用反向 sed 还原两个常量：

```bash
sed -i 's/^PIPELINE_MODE = "eval"/PIPELINE_MODE = "full"/' benchmark/main_eval.py
sed -i 's/^RESULT_TIMESTAMP = "20260715_224359"/RESULT_TIMESTAMP = ""/' benchmark/main_eval.py
grep -n 'PIPELINE_MODE\|RESULT_TIMESTAMP' benchmark/main_eval.py
git add benchmark/main_eval.py
git commit -m "feat(benchmark): main_eval 支持评估+分析 / 仅分析模式，跳过预测时引用历史结果"
```
Expected: `grep` 显示 `PIPELINE_MODE = "full"` 与 `RESULT_TIMESTAMP = ""`；提交成功。

- [ ] **Step 7: 最终回归（full 模式校验通过、不产生新目录）**

```bash
cd "C:\vibe coding\SARVQA-RAG-trae" && git log --oneline -3
python -m py_compile benchmark/main_eval.py
```
Expected: 最近 3 条提交为设计文档/计划/两个功能提交；`py_compile` 无输出。**注意**：不运行 full 模式（会产生模型 API 调用与费用）；full 分支代码与改动前逐字一致，由 Step 3/4 的成功与代码审查共同保证。

