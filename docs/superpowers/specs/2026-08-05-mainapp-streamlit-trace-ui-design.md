# mainapp.py Streamlit 展示优化设计（结构化 Trace + 双区 UI）

日期：2026-08-05
状态：已批准（方案 A：结构化 Trace）

## 背景

`agent/mainapp.py` 当前只有简单的聊天流：上传图片 → 输入问题 → 流式回答，RAG 检索过程完全不可见。需求：

1. 上传图片 + 输入问题，通过工具调用 RAG
2. RAG 过程可视化：先算隶属度（权重可配置）→ 命中阈值则历史日志快速召回，未命中则切片库检索
3. **工具输出格式化后放在另一个窗口（侧边栏）展示**
4. **主窗口展示最终召回的切片 + AI 回复**
5. 模型可切换

已确认的关键决策：

- 工具给 LLM 的返回字符串**保持原样**，UI 全部从新增的 `RetrievalTrace` 结构化数据渲染
- 检索参数（w1/w2/阈值/k）**运行时滑杆生效，不写回 yaml**
- 对话为**图片级多轮追问**（换图 = 新会话，历史会话可回看）
- 模型下拉**全部列出**，按能力分组：多模态组默认 `doubao-seed-2.0-mini`，文本组默认 `doubao-1.5-lite`
- 多模型切换不做专门测试

## 架构

```
┌─────────────────────────── Streamlit 页面 ───────────────────────────┐
│  侧边栏（另一个窗口）                      │  主区域                    │
│  ├ 模型选择（分组下拉）                     │  会话流：                  │
│  ├ 配置滑杆（w1/w2/阈值/k）                │   ① 用户消息（含图片缩略图）│
│  ├ 本次检索过程（阶段0/1/2 + 判定 + 耗时）  │   ② 召回切片卡片（得分徽标）│
│  ├ 会话级统计（命中率/平均耗时）            │   ③ AI 回复（流式 markdown）│
│  └ 历史会话列表                            │   底部固定栏（上传+输入）   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ execute_stream(query, image, history)
                 ┌─────────────▼──────────────┐
                 │   Agent 注册表（按模型选择）  │
                 │  doubao-seed→MainAgent 一步式│
                 │  internvl3.5/2→两步式(收集RAG │
                 │  +视觉回答)；文本模型→一步式  │
                 └─────────────┬──────────────┘
                               │ rag_rscsv 工具（给 LLM 的字符串不变）
                 ┌─────────────▼──────────────┐
                 │ MembershipHybridService    │
                 │  ├ 新增 RetrievalTrace 记录器│
                 │  ├ retrieve() 透传 w1/w2     │
                 │  └ get_last_trace()         │
                 └─────────────┬──────────────┘
                               │ 计算器透传 w1/w2
                 ┌─────────────▼──────────────┐
                 │ MembershipCalculator       │
                 └────────────────────────────┘
```

## 改动文件清单

| 文件 | 改动 |
|---|---|
| `agent/rag/services/membership_service.py` | 新增 `RetrievalTrace` dataclass；`hybrid_retrieve`/`retrieve` 增加 `w1/w2` 参数并记录各阶段明细+耗时；`get_last_trace()` |
| `agent/rag/membership/degree_calculator.py` | `calculate()` 透传 `w1/w2`（目前只能构造时定） |
| `agent/mainagent.py` | `MainAgent.__init__` 接受 `model` 参数（重建 agent 图，不重建 Chroma 存储）；`execute_stream` 接受历史消息列表（多轮）；暴露 `get_last_trace()` |
| `agent/mainagent_internVL.py` | 同样参数化 `vision_model`（两步式已存在，只需参数化）；`execute_stream` 同样支持历史消息（多轮追问） |
| `model/factory.py` | 接入 qwen3.7-plus（model.yml 已有配置） |
| `agent/agent_registry.py`（新增） | 模型 key → 代理类/模型实例/是否支持图像 的映射 |
| `agent/mainapp.py` | 整体重写 UI（侧边栏 + 主区 + 会话管理） |

不改动：`benchmark/`、`tools/`、工具返回给 LLM 的字符串格式、检索逻辑本身。

## RetrievalTrace 结构

每次 `hybrid_retrieve` 调用生成一份，存服务实例 `_last_trace`，UI 按 `trace.query == 本轮问题` 匹配（防串扰）：

```python
@dataclass
class RetrievalTrace:
    query: str
    decision: str                    # "membership_hit" | "slice_fallback"
    params_used: dict                # {w1, w2, fit_threshold, membership_k, slice_k, top_p}
    rag_context: str | None          # 阶段0
    stage0_latency: float
    membership: dict | None          # 阶段1: max_membership, qualified_log_count,
                                     #   top_logs: [{id, question, similarity,
                                     #               correctness, membership_degree, slice_ids}]
    stage1_latency: float
    slices: list[dict] | None        # 最终切片: [{slice_id, score,
                                     #   score_type("membership"/"similarity"), content}]
    stage2_latency: float
    total_latency: float
    error: str | None
```

## 数据流（一次问答）

```
上传新图片 → 创建新会话 session{image, turns:[]}
输入问题  → 追加 turn 占位
          → execute_stream(query, image, history)
              · 模型推理 → 调用 rag_rscsv
              · 工具执行 → hybrid_retrieve(query, w1, w2, fit_threshold, slice_k, top_p)
                  ├ 阶段0: RAG知识库检索(可选)
                  ├ 阶段1: 隶属度计算(日志库k条) → μ=w1×sim+w2×correctness → 与阈值比较
                  ├ 命中 → 按 μ 取切片 top_p 条
                  └ 未命中 → 阶段2 切片相似度检索
              · 记录 RetrievalTrace（各阶段耗时）
              · 模型生成回答（流式）
  ← 回答流式渲染
  ← 完成 → trace 写入 turns[-1].trace；侧边栏渲染过程；主区渲染切片卡片；统计刷新
```

关键机制：

- **滑杆实时生效**：提问时把滑杆值作为参数传入 `hybrid_retrieve`，不写 yaml；w2 联动 1-w1（计算器已有"和≠1 自动归一化"兜底）
- **多轮追问**：会话内每轮传 `history=[(human, ai), ...]`，agent 接收完整文本历史，图片只出现在最新一轮 HumanMessage；换图 → 新会话
- **模型切换**：重建 agent 图（`create_agent`），Chroma 存储实例不重建；已有历史用新模型继续回答，侧边栏提示"已切换到 xxx"

## 侧边栏设计

```
🤖 模型选择 [下拉·分组]
   多模态: doubao-seed-2.0-mini(默认) / internvl3.5-8b / internvl2-8b
   文本:   doubao-1.5-lite(默认) / qwen3-max / qwen3.7-plus
   ⚠️(文本模型+已传图) 当前模型不支持图像，将仅用文本回答
──────────────
⚙️ 检索参数（运行时生效）
   w1 [滑杆0~1] = 0.9 | w2 [联动] = 0.1 | fit_threshold [滑杆0~1] = 0.65
   membership_k [数字] = 50 | slice_k [数字] = 50 | top_p [数字] = 9
──────────────
🔍 本次检索过程（最新一轮）
   ✅ 隶属度命中 μ_max=0.873 ≥ 0.65   （或 🔄 未命中→降级切片检索 / 灰: 本轮无检索）
   ├ 阶段0 RAG参考  [▸展开] 12ms
   ├ 阶段1 隶属度   [▸展开] 38ms  （表格: id|历史问题|sim|correct|μ；合格日志 3/50）
   ├ 阶段2 切片检索 ──（未触发或对应渲染）
   └ 总耗时 63ms
──────────────
📊 会话统计：隶属度命中率 68.4% (13/19) · 平均检索耗时 52ms
──────────────
💬 会话列表：历史会话点击回看（侧边栏联动切换该会话最后一轮 trace）
```

渲染规则：

- 决策徽标：命中绿 / 降级橙 / 无检索灰
- 阶段0/1/2 用 `st.expander`；阶段1 明细用表格（id、历史问题、相似度、正确率、μ、关联切片数）
- **切片正文不进侧边栏**（主区统一展示，避免重复）
- 历史轮次的 trace 在会话内用 expander 展开查看

## 主区域设计

- **每轮 = 一个卡片容器**（新轮加在会话流顶部，贴近输入框）：
  1. 用户消息（文本 + 图片缩略图）
  2. "召回切片"区块：每片一行——得分徽标（`μ`/`sim`）+ slice_id + 内容截断（>200 字符，点击展开全文）；无结果显示占位
  3. AI 回复（流式 markdown）
  4. 行尾小字：`第N轮工具过程 ▸（跳转侧边栏）| 耗时 xx ms | ✅命中/🔄降级`
- 流式期间 `st.status("思考中...")` 显示 agent 当前步骤（如"调用工具 rag_rscsv"），完成收起
- 底部固定栏：图片上传（缩略图 + ✕清除）+ 输入框；无图可提问（纯文本）
- 空态：欢迎语 + 使用说明

## 错误处理与边界

| 场景 | 处理 |
|---|---|
| 图片读取/编码失败 | agent 兜底回复错误信息；侧边栏标注"本轮图片读取失败" |
| 文本模型 + 已传图片 | 常驻警告，仍可提问（纯文本） |
| 日志库为空 | 隶属度返回空 → 自动降级阶段2，trace 记录原因 |
| 切片检索无结果 | 主区显示"未检索到相关切片"占位 |
| 工具调用异常 | middleware 已记日志；trace.error 记录，侧边栏红色错误态 |
| 参数异常 | 计算器归一化 + 滑杆约束 |
| 模型切换失败 | 捕获异常，模型选择回滚，提示检查端点 |
| 本轮未调检索工具 | 侧边栏"本轮无检索调用"，主区不渲染切片区块 |
| trace 串扰 | 按 query 文本匹配，匹配不到取最新 |
| 首次构建 Chroma | 保留自动构建行为，UI 初始化 spinner + 侧边栏提示 |
| rerun 状态丢失 | 会话数据全部存 st.session_state |

**不做**（YAGNI）：图片裁剪/缩放编辑器、多用户隔离、trace 落盘持久化、自动写回 yaml。

## 测试与验证

1. **服务层**：临时脚本跑 `hybrid_retrieve`（自定义 w1/w2/阈值），断言：① 返回字符串格式不变（工具契约）；② `get_last_trace()` 结构完整、`params_used` 与实际一致、`decision` 正确
2. **边界参数**：w1+w2≠1 归一化、阈值=1（必降级）、阈值=0（必命中）
3. **回归**：`python agent/mainagent.py` CLI 问答，确认 agent 行为与改动前一致
4. **UI 冒烟**：`streamlit run agent/mainapp.py` 手动走查：上传图→提问→侧边栏明细/命中判定→追问→换模型→换图开新会话→回看历史会话
5. **基准回归**：benchmark/main_eval.py 不修改，抽 1 个 MODEL_KEY 跑小样本确认不因 trace 改动出错

（多模型切换不做专门测试）
