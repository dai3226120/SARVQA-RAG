# 隶属度历史日志形成机制改造 设计文档

日期: 2026-08-07
状态: 已批准（用户确认无需调整）

## 背景与目标

当前隶属度系统（`agent/rag/membership/`）存在以下问题：

1. **日志库来源不聚焦**：`agent/data/rag_feedback_logs.csv`（12.5k 条）记录了全部反馈，
   初始化时没有"仅由预测正确的会话记录形成隶属度"的明确机制。
2. **相似度向量不含回答**：日志向量文本为 `Question + Retrieved Slices(仅 id)`，
   用户要求隶属度相似度使用 **问题+切片+回答** 的综合向量化。
3. **查重键粗粒度**：当前按 `id`（图片文件名）保留最新，无法区分"同图不同问题"，
   也没有"分数更高则替换"的更新语义。
4. **入库无批量管控**：记录随写随入，缺少"检测待入库量 → 阈值 → 批量入库"的手动流程。
5. **召回路径未对比**：直接日志库召回 vs 日志库拿 id 再查切片库，两者效率未做过实验验证。

## 需求（用户确认）

1. **初始化**：隶属度日志由"最终预测正确的会话记录"形成。
   数据源为 `agent/dataset_split/test.csv`（2 万条，id/image/question/answer 四列），
   重跑预测（VLM agent + LLM 语义判断），只保留 correct=1 的记录。
2. **运行期新记录**：每次会话形成记录，按 **图片+问题** 联合查重：
   - 新记录 → 加入
   - 旧记录且分数更高 → 替换
   - 记录字段：问题 + 切片 + 回答，以及正确性分数（0/1 与连续分都存）、时间戳
3. **入库方法**（手动调用，放 data 文件夹）：检测待入库新记录条数，可设阈值，
   超过阈值执行批量入库。
4. **召回方式对比**：对比 (1) 直接从历史日志库直接召回 vs (2) 日志库拿 id →
   切片库检索召回，选效率高的；批量约 5000 次查询。

## 方案选择

选定 **方案 A：改造现有 LogManager**（用户确认）：

- 现有 12.5k 旧库 schema 不兼容新格式，初始化重跑本就要重建，顺势换新 schema；
- 改造点集中（schema / 向量文本 / 去重策略 / 批量 upsert），
  `SemanticCacheSystem`、`MembershipHybridService` 等调用方接口零改动；
- `VlmEvaluator` + `LlmJudge` 完全复用，无需新写预测逻辑。
- 未选方案 B（独立新模块）：两库并存、代码重复；未选方案 C（纯脚本层）：功能重复、无统一入口。

## 架构与组件

```
┌─────────────────────────────────────────────────────────────┐
│ 初始化        init_membership_logs.py（agent/data/，手动）    │
│   test.csv → VLM 预测 → LLM 判断 → correct=1 → 综合向量入库   │
├─────────────────────────────────────────────────────────────┤
│ 运行期        在线系统会话 → append_pending_record()          │
│   → agent/data/pending_records.csv（暂存区）                 │
├─────────────────────────────────────────────────────────────┤
│ 入库          flush_pending_logs.py（agent/data/，手动）      │
│   检测暂存条数 → 超阈值 → 图片+问题 查重/高分替换 → 批量入库    │
│   → 备份并清空暂存区                                          │
├─────────────────────────────────────────────────────────────┤
│ 实验          compare_recall_strategies.py（agent/data/）     │
│   5000 次查询：策略1 直接召回 vs 策略2 id→切片库召回，比效率    │
├─────────────────────────────────────────────────────────────┤
│ 核心改造      log_manager.py（schema/向量文本/去重/批量 upsert）│
│ 复用不变      vlm_evaluator.py / llm_judge.py /              │
│              degree_calculator.py / membership_service.py     │
└─────────────────────────────────────────────────────────────┘
```

## 日志 Schema（`agent/data/rag_feedback_logs.csv` 重建）

| 列 | 说明 | 变化 |
|---|---|---|
| `id` | 图片文件名（如 `train_sar5475.png`） | 保留 |
| `question` | 问题文本 | 保留 |
| `retrieved_slices` | 切片 id 列表，`\|` 分隔 | 保留 |
| `retrieved_slices_content` | 切片内容（入库时从切片库取回拼接） | **新增** |
| `predicted_text` | 模型回答 | 保留 |
| `correct` | 0/1 正确性（LLM 语义判断） | **新增** |
| `correctness_score` | 连续分数（VLM 评估） | 保留 |
| `timestamp` | 时间戳 | 保留 |

旧 CSV 加载时缺失新列自动补默认值（沿用现有 `_init_incremental` 的补列逻辑）。

## 查重与替换策略

- 查重键 = `id` + `question`（图片+问题联合）。
- 新记录 → 追加。
- 旧记录 → 比较 `correctness_score`：
  - 新分数 **更高** → 整体替换（CSV 行更新 + 向量库删旧向量 add 新向量）；
  - 不高于旧值 → 丢弃（记入统计）。
- 同分保新：分数相等时保留时间戳更晚的一条。
- 向量库唯一 id = `log_<id>_<question_md5前8位>`（确定性 id，替换可定位删除；
  不再使用现有的 `log_<id>_<时间戳后缀>`，后者每条插入都生成新 id 无法替换）。

## 综合向量化

```
向量文本 = "Question: {question}\n"
           "Retrieved Slices: {切片内容，每片截断 ~200 字符，总量截断至嵌入模型上限}\n"
           "Answer: {predicted_text}"
```

- 切片内容从 `SliceStore.get_by_ids()` 取回（入库时一次性批量获取）。
- 切片内容与回答的截断策略：每片截断 200 字符；整体再按嵌入模型最大 token 截断
  （HuggingFace 嵌入模型，配置在 `model_conf['huggingface_embedding_model_name']`）。
- `MembershipCalculator` 与查询侧零改动：query 仍为问题文本，相似度仍为 cosine；
  变化的只是日志集合中存储的向量文本。

## 初始化流程（`agent/data/init_membership_logs.py`）

参数：
- `--csv`：默认 `agent/dataset_split/test.csv`
- `--max-rows`：默认 20000（控制单次运行量）
- `--force`：清空重建；不带则交互确认

流程：
1. 备份现有 `rag_feedback_logs.csv` → `rag_feedback_logs_backup_<时间戳>.csv`。
2. 读 test.csv → `VlmEvaluator.call_vlm_agent` 预测（回答 + 切片id + 连续分数 + 时间戳）。
3. `LlmJudge` 语义判断（ground truth = test.csv 的 answer 列）→ `correct` 0/1。
4. 只保留 correct=1 的记录（初始化 = 最终预测正确的会话记录）。
5. 按 图片+问题 查重去重 → 写新 log CSV（含 `retrieved_slices_content`）。
6. 清空重建 `rag_test_logs` 集合 → 综合向量全量批量入库。
7. 打印：预测总数 / correct=1 数 / 最终入库数 / 耗时。

## 运行期暂存与批量入库

### 暂存区 `agent/data/pending_records.csv`

在线系统每会话追加一条：
```
id, question, retrieved_slices, predicted_text, correct, correctness_score, timestamp
```
写入函数 `append_pending_record()` 放 `agent/rag/membership/staging.py`，
使用 `utils/thread_lock` 保证线程安全。正确性 0/1 由在线系统提供
（用户反馈或 LLM 自评；在线场景无 ground truth，与初始化路径不同）。

### 入库脚本（`agent/data/flush_pending_logs.py`）

参数：`--threshold N`（默认 50）。

流程：
1. 读 pending_records.csv → 条数 < threshold → 打印"未达阈值，跳过"并以 0 退出码退出。
2. ≥ threshold → 按 图片+问题 查重合并进 log CSV（新加 / 高分替换 / 丢弃），
   同步更新向量库（新增批量 add；被替换记录按确定性 id 删旧加新）。
3. 入库成功 → pending 备份为 `pending_records_backup_<时间戳>.csv` → 清空原文件。
4. 打印统计：新增 X / 替换 Y / 丢弃 Z / 下次待处理 N。

## 召回策略对比实验（`agent/data/compare_recall_strategies.py`）

从 `agent/dataset_split/test.csv` 抽 N=5000 条查询（`--n` 可调），
top k 默认取 `rag_config.membership_k`（当前 50）：

```
策略 1「直接日志库召回」：query → 日志库向量检索(top k) → 上下文 =
        日志 metadata 中的切片内容 + 回答（不查切片库）
策略 2「id→切片库召回」：query → 日志库向量检索(top k) → 收集切片 id →
        SliceStore.get_by_ids 取切片内容 → 上下文
```

公平性：两策略产出同质的最终上下文（切片内容 + 回答），只差"切片内容从哪来"这一跳。
输出：两策略总耗时 / 平均耗时 / P50 / P95 / 耗时构成拆解（向量检索 vs 切片读取）/ 结论。

## 错误处理

- **初始化失败**：先备份旧库再重建；任一步失败打印错误并保留备份，不自动回滚（人工决策）。
- **入库失败**：CSV 合并成功但向量库失败 → 保留 pending 不清空（下次重跑幂等）；
  向量库 count 与 CSV 行数不一致由现有增量同步机制（`LogManager._check_and_reload_vector_db`）兜底。
- **pending 文件缺失/损坏**：提示并退出，不误清空。
- **并发**：替换与暂存写入均在 `utils/thread_lock` 锁内执行，防在线系统并发写 pending 时读脏。

## 测试

1. **单元测试**：查重/替换决策表（新加 / 高分替换 / 低分丢弃 / 同分保新）、
   向量文本构建与截断、确定性 id 生成。
2. **集成测试**：
   - test.csv 抽 20 条跑通初始化 → 断言入库数 = correct=1 数、向量可检索、正确性过滤生效；
   - 伪造 10 条 pending（含 1 条与库中重复且分数更高）→ flush → 断言新增/替换/清空正确；
   - 阈值测试：pending 5 条 + threshold 50 → 跳过。
3. **实验脚本**输出两策略对比结果作为验收证据。

## 涉及文件

| 文件 | 动作 |
|---|---|
| `agent/rag/membership/log_manager.py` | 改造：schema、向量文本、查重/替换、批量 upsert |
| `agent/rag/membership/staging.py` | 新增：`append_pending_record()` 线程安全写入 |
| `agent/rag/membership/cache_system.py` | 适配：evaluate_and_log 写新 schema |
| `agent/data/init_membership_logs.py` | 新增：初始化脚本 |
| `agent/data/flush_pending_logs.py` | 新增：批量入库脚本 |
| `agent/data/compare_recall_strategies.py` | 新增：召回方式对比实验 |
| `agent/rag/core/config.py` | 可能：新增 pending 路径配置 |
| 测试文件 | 新增：查重/替换/向量文本单元测试 + 集成测试 |
