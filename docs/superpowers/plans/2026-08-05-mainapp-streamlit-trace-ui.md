# mainapp.py Streamlit 展示优化（结构化 Trace + 双区 UI）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重写 `agent/mainapp.py`，让 RAG 检索过程（隶属度计算/命中判定/切片检索）在侧边栏可见，主区展示召回切片与 AI 回复，支持运行时滑杆配置与模型切换。

**Architecture:** 方案 A（已批准 spec）：给 `MembershipHybridService` 增加 `RetrievalTrace` 记录器与 `set_runtime_params` 运行时参数机制（工具签名不变），`degree_calculator`/`cache_system` 透传 w1/w2；agent 类参数化 model 并支持多轮历史；新增 `agent_registry.py` 模型注册表；`mainapp.py` 重写为侧边栏（模型/滑杆/过程/统计/会话）+ 主区（会话流）双区 UI。

**Tech Stack:** Streamlit, LangChain/LangGraph (create_agent), ChromaDB, ChatOpenAI/ChatTongyi

## Global Constraints

- **工具返回字符串格式不允许改变**（给 LLM 的契约，UI 从 trace 渲染，不解析字符串）
- `benchmark/` 与 `tools/` 零改动
- 检索参数运行时生效，**不写回 yaml 配置文件**
- 代码注释用中文，风格与现有文件一致
- 临时验证脚本用后即删（`agent/rag/verify_*.py`、`agent/verify_*.py`）
- commit message 用中文（见 `.trae/rules/git-commit-message.md`），格式 `feat: ...` / `refactor: ...`，结尾带 `更新时间：YYYY-MM-DD`
- 所有任务在项目根目录 `C:\vibe coding\SARVQA-RAG-trae` 下操作；运行脚本用 `python agent/xxx.py`（win32 + Git Bash）

---

### Task 1: 隶属度计算器与缓存系统 w1/w2 透传

**Files:**
- Modify: `agent/rag/membership/degree_calculator.py:37-59`（`calculate` 签名与权重解析）
- Modify: `agent/rag/membership/cache_system.py:59-77`（`calculate_membership_degree` 透传）
- Create (temp): `agent/rag/verify_w1w2.py`（用后删除）

**Interfaces:**
- Produces（Task 2 依赖）:
  ```python
  # degree_calculator.MembershipCalculator
  def calculate(self, query: str, k: int = None, fit_threshold: float = None,
                top_p: int = None, w1: float = None, w2: float = None) -> dict
  # cache_system.SemanticCacheSystem
  def calculate_membership_degree(self, query: str, k: int = None,
                                  fit_threshold: float = None, top_p: int = None,
                                  w1: float = None, w2: float = None) -> dict
  ```

- [ ] **Step 1: 写临时验证脚本 `agent/rag/verify_w1w2.py`**

```python
"""临时验证：w1/w2 透传后 membership == w1*sim + w2*correctness（用后删除）"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # agent 目录
from rag.membership.cache_system import SemanticCacheSystem

QUERY = "Is there evidence of water bodies, like ponds or streams, in this landscape?"

sys_obj = SemanticCacheSystem()
calc = sys_obj._calculator

for w1, w2, field in [(1.0, 0.0, "similarity"), (0.0, 1.0, "correctness_score")]:
    res = calc.calculate(QUERY, k=5, w1=w1, w2=w2)
    logs = res.get("top_logs", [])
    print(f"w1={w1}, w2={w2}: {len(logs)} 条日志")
    for log in logs:
        assert abs(log["membership_degree"] - log[field]) < 1e-6, \
            f"{log['id']}: membership={log['membership_degree']} != {field}={log[field]}"
    # 权重和不为 1 时归一化后仍满足线性关系
    res_norm = calc.calculate(QUERY, k=5, w1=0.6, w2=0.6)
    for log in res_norm.get("top_logs", []):
        expected = (0.6 * log["similarity"] + 0.6 * log["correctness_score"]) / 1.2
        assert abs(log["membership_degree"] - expected) < 1e-6, log["id"]
print("PASS")
```

- [ ] **Step 2: 运行验证脚本，确认当前失败**

Run: `python agent/rag/verify_w1w2.py`
Expected: FAIL——`TypeError: calculate() got an unexpected keyword argument 'w1'`（或 AssertionError，取决于默认权重恰好是 1/0 的巧合；无论哪种都说明透传未实现）

- [ ] **Step 3: 修改 `degree_calculator.py` 的 `calculate`**

在 `fit_threshold = fit_threshold or rag_config.fit_threshold` 之后、`# 验证权重和为 1` 之前，把权重解析改为（原 `w1, w2 = self._w1, self._w2` 替换为）：

```python
        k = k or rag_config.membership_k
        fit_threshold = fit_threshold or rag_config.fit_threshold
        top_p = top_p or rag_config.top_p

        # 解析权重：显式传入 > 构造时默认 > 配置默认；和不为 1 自动归一化
        w1 = w1 if w1 is not None else self._w1
        w2 = w2 if w2 is not None else self._w2
        if abs(w1 + w2 - 1.0) > 1e-6:
            logger.warning("权重和不为1，进行自动归一化: w1=%.2f, w2=%.2f", w1, w2)
            total = w1 + w2
            w1 = w1 / total
            w2 = w2 / total
```

同时把 `calculate` 签名（第 37-43 行）增加两个参数：

```python
    def calculate(
        self,
        query: str,
        k: int = None,
        fit_threshold: float = None,
        top_p: int = None,
        w1: float = None,
        w2: float = None,
    ) -> dict:
```

- [ ] **Step 4: 修改 `cache_system.py` 的 `calculate_membership_degree`**

```python
    def calculate_membership_degree(
        self, query: str, k: int = None, fit_threshold: float = None,
        top_p: int = None, w1: float = None, w2: float = None,
    ) -> dict:
        return self._calculator.calculate(
            query, k=k, fit_threshold=fit_threshold, top_p=top_p, w1=w1, w2=w2
        )
```

- [ ] **Step 5: 运行验证脚本，确认通过**

Run: `python agent/rag/verify_w1w2.py`
Expected: `w1=1.0, w2=0.0: N 条日志`、`w1=0.0, w2=1.0: N 条日志`、`PASS`（N≥0；日志库为空时无断言，脚本仍 PASS）

- [ ] **Step 6: 提交**

```bash
git add agent/rag/membership/degree_calculator.py agent/rag/membership/cache_system.py
git commit -m "feat: 隶属度计算器与缓存系统支持 w1/w2 运行时透传

- degree_calculator.calculate 新增 w1/w2 参数，显式传入优先于构造默认
- cache_system.calculate_membership_degree 同步透传
- 权重和不为 1 时保持自动归一化行为不变

更新时间：2026-08-05"
```

---

### Task 2: 检索服务 RetrievalTrace 记录 + set_runtime_params + get_last_trace

**Files:**
- Modify: `agent/rag/services/membership_service.py`（整体，见下）
- Create (temp): `agent/rag/verify_trace.py`（用后删除）

**Interfaces:**
- Consumes: Task 1 的 `calculate_membership_degree(..., w1, w2)` 透传
- Produces（Task 4/5/7 依赖）:
  ```python
  @dataclass
  class RetrievalTrace:
      query: str
      decision: str                    # "membership_hit" | "slice_fallback"
      params_used: dict                # {w1, w2, fit_threshold, membership_k, slice_k, top_p}
      rag_context: str | None = None
      stage0_latency: float = 0.0      # ms
      membership: dict | None = None   # {max_membership, qualified_log_count,
                                       #  qualified_memberships, top_logs, final_slices}
      stage1_latency: float = 0.0
      slices: list | None = None       # [{slice_id, score, score_type, content}]
      stage2_latency: float = 0.0
      total_latency: float = 0.0
      error: str | None = None

  def set_runtime_params(self, **kwargs) -> None     # 合法 key: w1 w2 fit_threshold membership_k slice_k top_p
  def clear_runtime_params(self) -> None
  def get_last_trace(self) -> RetrievalTrace | None
  def hybrid_retrieve(self, query, slice_k=None, membership_k=None, top_p=None,
                      fit_threshold=None, w1=None, w2=None) -> str   # 签名向后兼容
  def retrieve(self, query) -> str                    # 签名不变（工具契约）
  ```

- [ ] **Step 1: 写临时验证脚本 `agent/rag/verify_trace.py`**

```python
"""临时验证：RetrievalTrace 结构 + 工具字符串契约不变 + 运行时参数生效（用后删除）"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # agent 目录
from rag.services.membership_service import MembershipHybridService

QUERY = "Is there evidence of water bodies, like ponds or streams, in this landscape?"

svc = MembershipHybridService()

# 1) 字符串契约：命中/降级/无结果 三种格式之一
out = svc.retrieve(QUERY)
assert "【匹配隶属度缓存】" in out or "【匹配基础切片】" in out or "未检索到相关遥感问答参考资料" in out, out[:120]

# 2) trace 结构完整
trace = svc.get_last_trace()
assert trace is not None and trace.query == QUERY
assert trace.decision in ("membership_hit", "slice_fallback")
assert set(trace.params_used) == {"w1", "w2", "fit_threshold", "membership_k", "slice_k", "top_p"}
assert trace.stage0_latency >= 0 and trace.stage1_latency >= 0 and trace.total_latency >= 0
if trace.decision == "slice_fallback":
    assert trace.stage2_latency >= 0
    assert all(s["score_type"] == "similarity" for s in (trace.slices or []))
if trace.decision == "membership_hit":
    assert all(s["score_type"] == "membership" for s in (trace.slices or []))
print(f"decision={trace.decision}, μ_max={trace.membership and trace.membership['max_membership']}, slices={len(trace.slices or [])}")

# 3) set_runtime_params 生效且 μ 复核线性关系
svc.set_runtime_params(w1=0.5, w2=0.5, fit_threshold=0.3)
svc.retrieve(QUERY)
t2 = svc.get_last_trace()
assert t2.params_used["w1"] == 0.5 and t2.params_used["w2"] == 0.5 and t2.params_used["fit_threshold"] == 0.3
if t2.membership and t2.membership.get("top_logs"):
    for log in t2.membership["top_logs"]:
        assert abs(log["membership_degree"] - (0.5 * log["similarity"] + 0.5 * log["correctness_score"])) < 1e-6
svc.clear_runtime_params()
t3 = svc.get_last_trace()
assert t3.params_used["w1"] == svc._resolve("w1", 0.9)  # 清除后回落默认（w1 配置默认 0.9）

# 4) 阈值=1.01 必降级
svc.set_runtime_params(fit_threshold=1.01)
svc.retrieve(QUERY)
assert svc.get_last_trace().decision == "slice_fallback"
svc.clear_runtime_params()
print("PASS")
```

- [ ] **Step 2: 运行验证脚本，确认当前失败**

Run: `python agent/rag/verify_trace.py`
Expected: FAIL——`AttributeError: 'MembershipHybridService' object has no attribute 'get_last_trace'`

- [ ] **Step 3: 在 `membership_service.py` 顶部新增 dataclass 与 time 导入**

文件第 1-12 行 import 区域之后（`from utils.logger_handler import logger` 之后）追加：

```python
import time
from dataclasses import dataclass


@dataclass
class RetrievalTrace:
    """一次 hybrid_retrieve 调用的完整过程记录（供 UI 展示，工具字符串不受影响）"""

    query: str
    decision: str                      # "membership_hit" | "slice_fallback"
    params_used: dict                  # 实际使用的检索参数
    rag_context: str | None = None     # 阶段0
    stage0_latency: float = 0.0
    membership: dict | None = None     # 阶段1: max_membership / qualified_log_count / top_logs / final_slices
    stage1_latency: float = 0.0
    slices: list | None = None         # 最终切片: [{slice_id, score, score_type, content}]
    stage2_latency: float = 0.0
    total_latency: float = 0.0
    error: str | None = None
```

- [ ] **Step 4: `__init__` 增加运行时参数与 trace 槽位**

在 `__init__` 末尾（`self._enable_rag_context = rag_config.enable_rag_context` 之后）追加：

```python
        # 6. 运行时参数覆盖（UI 滑杆设置；retrieve 合并后生效，不写配置文件）
        self._runtime_params: dict = {}
        self._last_trace: RetrievalTrace | None = None
```

- [ ] **Step 5: 新增参数/接口方法**

在 `get_membership_stats` 方法之前插入：

```python
    def set_runtime_params(self, **kwargs):
        """设置运行时参数覆盖（UI 在提问前调用；工具签名不变，参数经此透传）"""
        valid = {"w1", "w2", "fit_threshold", "membership_k", "slice_k", "top_p"}
        self._runtime_params = {k: v for k, v in kwargs.items() if k in valid and v is not None}

    def clear_runtime_params(self):
        """清除运行时参数覆盖，回落配置文件默认值"""
        self._runtime_params = {}

    def _resolve(self, key: str, default):
        """参数解析：运行时覆盖 > 默认值"""
        return self._runtime_params.get(key, default)

    def get_last_trace(self) -> RetrievalTrace | None:
        """获取最近一次检索的过程记录（供 UI 展示）"""
        return self._last_trace
```

- [ ] **Step 6: 重写 `hybrid_retrieve`（第 80-129 行）**

```python
    def hybrid_retrieve(
        self,
        query: str,
        slice_k: int = None,
        membership_k: int = None,
        top_p: int = None,
        fit_threshold: float = None,
        w1: float = None,
        w2: float = None,
    ) -> str:
        """
        执行三阶段混合检索（含过程记录）

        Args:
            query: 查询文本
            slice_k / membership_k / top_p / fit_threshold / w1 / w2:
                检索参数，显式传入 > 运行时覆盖(set_runtime_params) > 配置默认

        Returns:
            格式化的检索结果字符串（契约不变）
        """
        # 参数解析：显式传入 > 运行时覆盖 > 配置默认
        slice_k = slice_k or self._resolve("slice_k", self._slice_k)
        membership_k = membership_k or self._resolve("membership_k", self._membership_k)
        top_p = top_p or self._resolve("top_p", self._top_p)
        fit_threshold = fit_threshold or self._resolve("fit_threshold", self._fit_threshold)
        w1 = w1 if w1 is not None else self._resolve("w1", rag_config.w1)
        w2 = w2 if w2 is not None else self._resolve("w2", rag_config.w2)

        total_start = time.time()
        trace = RetrievalTrace(
            query=query,
            decision="slice_fallback",
            params_used={
                "w1": w1, "w2": w2, "fit_threshold": fit_threshold,
                "membership_k": membership_k, "slice_k": slice_k, "top_p": top_p,
            },
        )

        try:
            # ==========================================
            # 阶段 0: RAG 向量检索（通过 chroma.yml → retrieval.enable_rag_context 控制）
            # ==========================================
            t0 = time.time()
            rag_context = self._knowledge_service.retrieve_context(query) if self._enable_rag_context else ""
            trace.rag_context = rag_context or None
            trace.stage0_latency = (time.time() - t0) * 1000

            # ==========================================
            # 阶段 1: 隶属度计算（缓存拦截与校验）
            # ==========================================
            MembershipHybridService._total_calls += 1
            t1 = time.time()
            result_str, membership_trace = self._retrieve_by_membership(
                query, membership_k, fit_threshold, top_p, w1, w2
            )
            trace.membership = membership_trace
            trace.stage1_latency = (time.time() - t1) * 1000

            if result_str is not None:
                trace.decision = "membership_hit"
                trace.slices = (membership_trace or {}).get("final_slices", [])
                trace.total_latency = (time.time() - total_start) * 1000
                self._last_trace = trace
                if rag_context:
                    return f"【RAG检索参考】\n{rag_context}\n\n==============================\n\n{result_str}"
                return result_str

            if membership_trace is None:
                trace.error = "隶属度计算过程异常，已降级到基础检索"

            # ==========================================
            # 阶段 2: 基础切片检索（降级回退）
            # ==========================================
            t2 = time.time()
            rscsv_result, slices_trace = self._retrieve_by_similarity(query, slice_k, top_p)
            trace.slices = slices_trace
            trace.stage2_latency = (time.time() - t2) * 1000
            trace.total_latency = (time.time() - total_start) * 1000
            self._last_trace = trace
            if rag_context:
                return f"【RAG检索参考】\n{rag_context}\n\n==============================\n\n{rscsv_result}"
            return rscsv_result
        except Exception as e:
            logger.error(f"[hybrid_retrieve] 检索过程异常: {e}", exc_info=True)
            trace.error = str(e)
            trace.total_latency = (time.time() - total_start) * 1000
            self._last_trace = trace
            raise
```

- [ ] **Step 7: 重写 `_retrieve_by_membership`（第 141-218 行），返回 `(str|None, dict|None)`**

```python
    def _retrieve_by_membership(
        self, query: str, membership_k: int, fit_threshold: float, top_p: int,
        w1: float, w2: float,
    ) -> tuple[str | None, dict | None]:
        """
        阶段 1: 隶属度检索
        Returns: (检索结果字符串或 None, 过程记录 dict 或 None)
        """
        trace_data = None
        try:
            membership_result = self._cache_system.calculate_membership_degree(
                query, k=membership_k, fit_threshold=fit_threshold, top_p=top_p, w1=w1, w2=w2
            )
            max_membership = membership_result.get("max_membership", 0.0)
            qualified_log_count = membership_result.get("qualified_log_count", 0)
            qualified_memberships = membership_result.get("qualified_memberships", [])
            top_logs = membership_result.get("top_logs", [])

            memberships_str = ", ".join([f"{m:.4f}" for m in qualified_memberships])

            trace_data = {
                "max_membership": max_membership,
                "qualified_log_count": qualified_log_count,
                "qualified_memberships": qualified_memberships,
                "top_logs": top_logs,
                "final_slices": [],
            }

            if qualified_log_count > 0 and membership_result.get("weighted_slices"):
                logger.info(
                    f"【隶属度命中】合格日志: {qualified_log_count}条，"
                    f"隶属度(从大到小): [{memberships_str}]"
                )

                top_slice_info = membership_result["weighted_slices"][:top_p]
                top_slice_ids = [s["slice_id"] for s in top_slice_info]
                logger.info(f" 正在尝试从切片库获取以下 ID 的数据: {top_slice_ids}")

                slices_data = self._store.get_by_ids(top_slice_ids)
                documents = slices_data.get("documents", [])
                metadatas = slices_data.get("metadatas", [])

                logger.info(f" 切片库实际返回了 {len(documents)} 条文档内容")

                if documents:
                    MembershipHybridService._hit_calls += 1

                    # 按隶属度得分排序（同时携带 slice_id 供 trace 使用）
                    doc_with_membership = []
                    for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                        slice_id = (
                            metadata.get("slice_id")
                            if isinstance(metadata, dict)
                            else top_slice_ids[i]
                        )
                        membership_degree = next(
                            (
                                s["membership_degree"]
                                for s in top_slice_info
                                if s["slice_id"] == slice_id
                            ),
                            0.0,
                        )
                        doc_with_membership.append((doc, membership_degree, slice_id))

                    doc_with_membership.sort(key=lambda x: x[1], reverse=True)

                    trace_data["final_slices"] = [
                        {"slice_id": sid, "score": score,
                         "score_type": "membership", "content": doc}
                        for doc, score, sid in doc_with_membership
                    ]

                    content = "\n---\n".join(
                        [
                            f"隶属度得分: {score:.4f}\n{doc}"
                            for doc, score, _ in doc_with_membership
                        ]
                    )
                    return (
                        f"【匹配隶属度缓存】合格日志={qualified_log_count}条，"
                        f"隶属度(从大到小): [{memberships_str}] "
                        f"(共检索{membership_k}条，"
                        f"按隶属度排序后保留{len(doc_with_membership)}条)  \n{content}",
                        trace_data,
                    )
            else:
                logger.info(
                    f"【隶属度未命中/未达标】合格日志: {qualified_log_count}条，"
                    f"最大隶属度: {max_membership:.4f}"
                )

        except Exception as e:
            logger.error(f"隶属度计算过程发生异常，降级到基础检索: {str(e)}")

        return None, trace_data
```

- [ ] **Step 8: 重写 `_retrieve_by_similarity`（第 220-238 行），返回 `(str, list)`**

```python
    def _retrieve_by_similarity(self, query: str, slice_k: int, top_p: int) -> tuple[str, list]:
        """阶段 2: 基础切片检索（降级回退），返回 (结果字符串, 切片trace列表)"""
        slice_results = self._store.similarity_search_with_scores(query, k=slice_k)
        if slice_results:
            sorted_results = sorted(slice_results, key=lambda x: x[1], reverse=True)
            top_results = sorted_results[:top_p]

            slices_trace = [
                {
                    "slice_id": doc.metadata.get("slice_id", ""),
                    "score": float(score),
                    "score_type": "similarity",
                    "content": doc.page_content,
                }
                for doc, score in top_results
            ]

            content = "\n---\n".join(
                [
                    f"相似度得分: {score:.4f}\n{doc.page_content}"
                    for doc, score in top_results
                ]
            )
            return (
                f"【匹配基础切片】(共检索{slice_k}条，"
                f"按相似度排序后保留{len(top_results)}条)  \n{content}",
                slices_trace,
            )

        return "未检索到相关遥感问答参考资料。", []
```

- [ ] **Step 9: 简化 `retrieve`（第 131-139 行）——签名不变，参数在 hybrid_retrieve 内部解析**

```python
    def retrieve(self, query: str) -> str:
        """实现 BaseRetriever 接口（参数经 set_runtime_params 运行时配置）"""
        return self.hybrid_retrieve(query)
```

- [ ] **Step 10: 运行验证脚本，确认通过**

Run: `python agent/rag/verify_trace.py`
Expected: 打印 decision/μ_max/slices 行 + `PASS`。若日志库为空（无 chroma 数据），第 4 步"阈值=1.01 必降级"仍成立

- [ ] **Step 11: 提交**

```bash
git add agent/rag/services/membership_service.py
git commit -m "feat: 检索服务新增 RetrievalTrace 过程记录与运行时参数机制

- hybrid_retrieve 记录阶段0/1/2 明细、参数与耗时，get_last_trace 供 UI 展示
- set_runtime_params/clear_runtime_params 实现滑杆参数透传（工具签名不变）
- _retrieve_by_membership/_retrieve_by_similarity 附带结构化 trace 返回
- 工具返回字符串格式与检索逻辑零改动

更新时间：2026-08-05"
```

---

### Task 3: model/factory.py 接入 qwen3.7-plus

**Files:**
- Modify: `model/factory.py`（`DoubaoLiteModelFactory` 之后追加）

**Interfaces:**
- Produces（Task 6 依赖）:
  ```python
  qwen37_plus_model: ChatOpenAI   # model/factory 模块级实例
  ```

- [ ] **Step 1: 在 `factory.py` 的 `DoubaoLiteModelFactory` 类之后（第 87 行后）追加类与实例**

```python
# Qwen3.7 Plus 模型工厂：用于文本问答（阿里云 MaaS，OpenAI 兼容）
class Qwen37PlusModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatOpenAI(
            model_name=model_conf['qwen37_plus_model_name'],
            api_key=os.environ.get('DASHSCOPE_API_KEY') or model_conf.get('qwen37_plus_api_key', ''),
            base_url=model_conf['qwen37_plus_api_endpoint'],
            temperature=0.7,
            streaming=True,
        )
```

在模块底部实例列表（`doubao_1_5_lite_model = ...` 行后）追加：

```python
qwen37_plus_model = Qwen37PlusModelFactory().generate()  # Qwen3.7 Plus 文本模型
```

- [ ] **Step 2: 验证导入**

Run: `python -c "import sys; sys.path.insert(0, 'agent'); sys.path.insert(0, '.'); from model.factory import qwen37_plus_model; print(qwen37_plus_model.model_name)"`
Expected: 打印 `qwen3.7-plus`（不发起网络请求）

- [ ] **Step 3: 提交**

```bash
git add model/factory.py
git commit -m "feat: model/factory 接入 qwen3.7-plus 文本模型

- 新增 Qwen37PlusModelFactory 与 qwen37_plus_model 实例
- API key 复用 DASHSCOPE_API_KEY，兼容 model.yml 已有端点配置

更新时间：2026-08-05"
```

---

### Task 4: mainagent.py 模型参数化 + 多轮历史 + get_last_trace

**Files:**
- Modify: `agent/mainagent.py`
- Create (temp): `agent/verify_mainagent.py`（用后删除）

**Interfaces:**
- Consumes: Task 2 的 `rag_rscsv_service.get_last_trace()`（`tools.agent_tools` 模块级单例）
- Produces（Task 6/7 依赖）:
  ```python
  class MainAgent:
      def __init__(self, model=None, tools=None) -> None
      def rebuild_agent(self, model) -> None
      def execute_stream(self, query: str, image_file=None, history: list[dict] | None = None)
          # history: [{"role": "user"|"assistant", "content": str}, ...]（不含最新一轮）
      def get_last_trace(self) -> RetrievalTrace | None
      def _build_messages(self, query, image_file, history=None) -> list  # 可单测的纯构造
  ```

- [ ] **Step 1: 写临时验证脚本 `agent/verify_mainagent.py`**

```python
"""临时验证：MainAgent 消息构造与历史支持（不调 API；用后删除）"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # agent 目录
from mainagent import MainAgent

agent = MainAgent()

# 1) 历史消息构造：user/ai/user 顺序
msgs = agent._build_messages("q2", None, history=[
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1"},
])
assert len(msgs) == 3, len(msgs)
assert msgs[0].type == "human" and msgs[1].type == "ai" and msgs[2].type == "human"

# 2) 带图片：内容为 [text, image_url]
fake = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
fake.type = "image/png"
msgs2 = agent._build_messages("q", fake, history=None)
content = msgs2[0].content
assert isinstance(content, list) and len(content) == 2, content
assert content[0]["type"] == "text" and content[1]["type"] == "image_url"

# 3) get_last_trace 存在且返回 None（尚未检索）
assert agent.get_last_trace() is None or hasattr(agent.get_last_trace(), "query")
print("PASS")
```

- [ ] **Step 2: 运行验证脚本，确认当前失败**

Run: `python agent/verify_mainagent.py`
Expected: FAIL——`TypeError: MainAgent() got an unexpected keyword argument`（构造无 model 参数）或 `AttributeError: 'MainAgent' object has no attribute 'get_last_trace'`

- [ ] **Step 3: 修改 import 与 `__init__`**

`from langchain_core.messages import HumanMessage` 改为：

```python
from langchain_core.messages import HumanMessage, AIMessage
```

`from tools.agent_tools import rag_summarize,get_weather,get_user_location, get_user_id, rag_rscsv, to_openai_tools` 改为（追加 `rag_rscsv_service`）：

```python
from tools.agent_tools import rag_summarize, get_weather, get_user_location, get_user_id, rag_rscsv, to_openai_tools, rag_rscsv_service
```

`__init__`（第 26-41 行）替换为：

```python
    def __init__(self, model=None, tools=None):
        """Args:
            model: 聊天模型实例（None 时用默认 doubao_seed_20_mini_model）
            tools: 工具列表（None 时用默认 [rag_rscsv]）
        """
        self.tools = tools if tools is not None else [rag_rscsv]
        self.model = model
        self.rag_output = ""  # 累积 RAG 工具输出文本

        self.agent = create_agent(
            model=self.model if self.model is not None else doubao_seed_20_mini_model,
            tools=self.tools,
            system_prompt=load_system_prompts(),
            middleware=[log_before_model, monitor_tool]
        )

    def rebuild_agent(self, model):
        """切换模型：只重建 agent 图，Chroma 存储不重建（省时）"""
        self.model = model
        self.agent = create_agent(
            model=model,
            tools=self.tools,
            system_prompt=load_system_prompts(),
            middleware=[log_before_model, monitor_tool]
        )

    def _build_messages(self, query: str, image_file=None, history=None) -> list:
        """构造输入消息：历史文本（纯文本）+ 最新一轮（含可选图片）"""
        messages = []
        for h in history or []:
            content = h.get("content", "")
            if h.get("role") == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

        content = [{"type": "text", "text": query}]
        if image_file is not None:
            try:
                if hasattr(image_file, "seek"):
                    image_file.seek(0)
                image_bytes = image_file.read()
                image_type = getattr(image_file, "type", "image/jpeg") or "image/jpeg"
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{base64_image}"}}
                )
            except Exception as e:
                logger.error(f"[MainAgent] 读取图像数据失败: {e}", exc_info=True)
                raise ValueError("无法读取上传图像，请重试。")
        messages.append(HumanMessage(content=content))
        return messages

    def get_last_trace(self):
        """获取最后一次 RAG 检索的过程记录（RetrievalTrace 或 None）"""
        return rag_rscsv_service.get_last_trace()
```

- [ ] **Step 4: `execute_stream` 使用 `_build_messages` 并支持 history**

第 44-63 行（`def execute_stream` 开头到 `input_dict = ...`）替换为：

```python
    # 多模态输入版本
    def execute_stream(self, query: str, image_file=None, history=None):
        """流式执行 agent 问答

        Args:
            query: 本轮问题
            image_file: 当前图片（file-like，可带 .type 属性）
            history: 之前轮次 [{"role": "user"|"assistant", "content": str}, ...]
        """
        try:
            input_dict = {"messages": self._build_messages(query, image_file, history)}
        except ValueError as e:
            yield str(e)
            return
```

- [ ] **Step 5: 运行验证脚本，确认通过**

Run: `python agent/verify_mainagent.py`
Expected: `PASS`

- [ ] **Step 6: CLI 回归**

Run: `python agent/mainagent.py`
Expected: 与改动前一致的 agent 问答输出（`mainagent.py` 的 `__main__` 自带图片不存在时的纯文本降级；`.env` 已配置 doubao key 时正常输出回答）

- [ ] **Step 7: 提交**

```bash
git add agent/mainagent.py
git commit -m "feat: MainAgent 支持模型参数化、多轮历史与检索过程透出

- __init__/rebuild_agent 接受 model 参数，切换模型仅重建 agent 图
- execute_stream 支持 history 多轮追问，_build_messages 纯函数便于验证
- get_last_trace 透出 rag_rscsv 服务的 RetrievalTrace

更新时间：2026-08-05"
```

---

### Task 5: mainagent_internVL.py 视觉模型参数化 + 历史 + get_last_trace

**Files:**
- Modify: `agent/mainagent_internVL.py`

**Interfaces:**
- Consumes: Task 2 的 `rag_rscsv_service.get_last_trace()`
- Produces（Task 6 依赖）:
  ```python
  class MainAgent:   # mainagent_internVL.MainAgent（两步式）
      def __init__(self, model=None, vision_model=None, tools=None) -> None
      def execute_stream(self, query: str, image_file=None, history=None)
      def get_last_trace(self) -> RetrievalTrace | None
  ```

- [ ] **Step 1: 修改 import 与 `__init__`**

`from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage` 改为（追加 AIMessage）：

```python
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
```

`from tools.agent_tools import ...` 追加 `rag_rscsv_service`（同 Task 4 Step 3）。

`__init__`（第 26-42 行）替换为：

```python
    def __init__(self, model=None, vision_model=None, tools=None):
        """两步式 agent：agent 收集 RAG 信息（文本）→ vision_model 多模态回答

        Args:
            model: 收集 RAG 的 agent 模型（None 时默认 doubao_seed_20_mini_model）
            vision_model: 最终视觉回答模型（None 时默认 internvl3_5_8b_model）
        """
        self.tools = tools if tools is not None else [rag_summarize, rag_rscsv]

        self.agent = create_agent(
            model=model if model is not None else doubao_seed_20_mini_model,
            tools=self.tools,
            system_prompt=load_system_prompts(),
            middleware=[log_before_model, monitor_tool]
        )

        # 2. internvl 作为多模态视觉最终回答模型
        self.vision_model = vision_model if vision_model is not None else internvl3_5_8b_model

    def _build_agent_messages(self, query: str, history=None) -> list:
        """构造 agent（文本）输入消息：历史文本 + 当前问题"""
        messages = []
        for h in history or []:
            content = h.get("content", "")
            if h.get("role") == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=query))
        return messages

    def get_last_trace(self):
        """获取最后一次 RAG 检索的过程记录（RetrievalTrace 或 None）"""
        return rag_rscsv_service.get_last_trace()
```

- [ ] **Step 2: `execute_stream` 支持 history**

第 64-68 行：

```python
        # ----- 第一步：用 doubao agent 收集 RAG 信息（仅文本） -----
        try:
            # agent 只收文本，避免浪费视觉推理
            agent_input = {"messages": HumanMessage(content=query)}
```

替换为：

```python
        # ----- 第一步：用 doubao agent 收集 RAG 信息（仅文本） -----
        try:
            # agent 只收文本，避免浪费视觉推理
            agent_input = {"messages": self._build_agent_messages(query, history)}
```

其余（vision 部分）不变——最终视觉回答只取最新问题+图片+RAG 上下文。

- [ ] **Step 3: 验证**

Run: `python -c "import sys; sys.path.insert(0, 'agent'); from mainagent_internVL import MainAgent; a = MainAgent(); m = a._build_agent_messages('q2', [{'role': 'user', 'content': 'q1'}, {'role': 'assistant', 'content': 'a1'}]); assert [x.type for x in m] == ['human', 'ai', 'human']; print('PASS')"`
Expected: `PASS`（internvl 端点为远程 GPU 服务器，不在此步发起请求）

- [ ] **Step 4: 提交**

```bash
git add agent/mainagent_internVL.py
git commit -m "feat: MainAgent(internVL两步式) 支持视觉模型参数化与多轮历史

- __init__ 接受 model/vision_model 参数，默认行为不变
- execute_stream 支持 history，agent 文本输入走 _build_agent_messages
- get_last_trace 透出检索过程记录

更新时间：2026-08-05"
```

---

### Task 6: 新增 agent_registry.py 模型注册表

**Files:**
- Create: `agent/agent_registry.py`

**Interfaces:**
- Consumes: Task 3（qwen37_plus_model）、Task 4/5（MainAgent / MainAgentInternVL）
- Produces（Task 7 依赖）:
  ```python
  MODEL_REGISTRY: dict[str, dict]  # key → {class, model, supports_image, group, label}
  DEFAULT_MODEL_KEY = "doubao-seed-2.0-mini"      # 多模态默认
  DEFAULT_TEXT_MODEL_KEY = "doubao-1.5-lite"      # 文本默认
  def build_agent(model_key: str) -> MainAgent   # 按 key 创建 agent 实例
  ```

- [ ] **Step 1: 写 `agent/agent_registry.py`**

```python
"""Agent 模型注册表：模型 key → 代理类/模型实例/能力分组

供 mainapp.py 模型切换下拉使用；build_agent 按 key 创建 agent 实例。
"""
from model.factory import (
    chat_model,
    doubao_seed_20_mini_model,
    internvl2_8b_model,
    internvl3_5_8b_model,
    doubao_1_5_lite_model,
    qwen37_plus_model,
)
from mainagent import MainAgent
from mainagent_internVL import MainAgent as MainAgentInternVL

MODEL_REGISTRY = {
    # ── 多模态 VLM ──
    "doubao-seed-2.0-mini": {
        "class": MainAgent, "model": doubao_seed_20_mini_model,
        "supports_image": True, "group": "多模态", "label": "Doubao Seed 2.0 Mini",
    },
    "internvl3.5-8b": {
        "class": MainAgentInternVL, "model": internvl3_5_8b_model,
        "supports_image": True, "group": "多模态", "label": "InternVL3.5-8B",
    },
    "internvl2-8b": {
        "class": MainAgentInternVL, "model": internvl2_8b_model,
        "supports_image": True, "group": "多模态", "label": "InternVL2-8B",
    },
    # ── 文本 LLM ──
    "doubao-1.5-lite": {
        "class": MainAgent, "model": doubao_1_5_lite_model,
        "supports_image": False, "group": "文本", "label": "Doubao 1.5 Lite",
    },
    "qwen3-max": {
        "class": MainAgent, "model": chat_model,
        "supports_image": False, "group": "文本", "label": "Qwen3 Max",
    },
    "qwen3.7-plus": {
        "class": MainAgent, "model": qwen37_plus_model,
        "supports_image": False, "group": "文本", "label": "Qwen3.7 Plus",
    },
}

DEFAULT_MODEL_KEY = "doubao-seed-2.0-mini"   # 多模态默认
DEFAULT_TEXT_MODEL_KEY = "doubao-1.5-lite"   # 文本默认


def build_agent(model_key: str):
    """按 key 创建 agent 实例（internvl 两步式需指定 vision_model）"""
    info = MODEL_REGISTRY[model_key]
    if info["class"] is MainAgentInternVL:
        return info["class"](vision_model=info["model"])
    return info["class"](model=info["model"])
```

- [ ] **Step 2: 验证**

Run: `python -c "import sys; sys.path.insert(0, 'agent'); from agent_registry import MODEL_REGISTRY, DEFAULT_MODEL_KEY, DEFAULT_TEXT_MODEL_KEY, build_agent; assert set(MODEL_REGISTRY) == {'doubao-seed-2.0-mini','internvl3.5-8b','internvl2-8b','doubao-1.5-lite','qwen3-max','qwen3.7-plus'}; assert MODEL_REGISTRY[DEFAULT_MODEL_KEY]['group'] == '多模态' and MODEL_REGISTRY[DEFAULT_TEXT_MODEL_KEY]['group'] == '文本'; a = build_agent(DEFAULT_MODEL_KEY); print(type(a).__name__, 'PASS')"`
Expected: `MainAgent PASS`

- [ ] **Step 3: 提交**

```bash
git add agent/agent_registry.py
git commit -m "feat: 新增 agent_registry 模型注册表

- 6 个模型按能力分组（多模态 VLM / 文本 LLM），默认值 doubao-seed 与 doubao-1.5-lite
- build_agent 统一创建入口，internvl 走两步式（指定 vision_model）

更新时间：2026-08-05"
```

---

### Task 7: mainapp.py 整体重写（双区 UI）

**Files:**
- Rewrite: `agent/mainapp.py`

**Interfaces:**
- Consumes: Task 2（`RetrievalTrace` 字段、`set_runtime_params`）、Task 4/5（`execute_stream(query, image_file, history)`、`get_last_trace()`）、Task 6（`MODEL_REGISTRY` / `build_agent`）、`tools.middleware.ToolLatencyTracker`、`rag.services.membership_service.MembershipHybridService`
- Produces: 可运行的 Streamlit 应用（无对外接口）

- [ ] **Step 1: 整体重写 `agent/mainapp.py`**

```python
"""SAR 遥感问答系统 - Streamlit UI

侧边栏（工具过程窗口）：模型选择 / 检索参数滑杆 / 本次检索过程（阶段0/1/2+判定+耗时）/ 会话统计 / 会话列表
主区域：会话流（用户消息 → 召回切片卡片 → AI 回复，最新在上）
底部固定栏：图片上传（预览+清除）+ 提问输入
"""
import io
import time
from dataclasses import asdict
from datetime import datetime

import streamlit as st
from PIL import Image

from mainagent import MainAgent  # noqa: F401  # 确保 agent 目录在 sys.path
from agent_registry import MODEL_REGISTRY, DEFAULT_MODEL_KEY, DEFAULT_TEXT_MODEL_KEY, build_agent
from tools.agent_tools import rag_rscsv_service
from rag.services.membership_service import MembershipHybridService
from tools.middleware import ToolLatencyTracker

st.set_page_config(page_title="SAR遥感问答系统", layout="wide")

st.markdown("""
<style>
/* 给聊天内容区加底部内边距，防止被底部栏遮挡 */
.stAppViewContainer {
    padding-bottom: 220px !important;
}
/* 固定底部栏：贴底、全屏宽、顶层显示、白色背景、顶部边框 */
.focused-bottom-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: #ffffff;
    z-index: 9999;
    padding: 10px 20px;
    border-top: 1px solid #e5e7eb;
}
/* 隐藏文件上传器默认标签 */
.stFileUploader > label {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# ───────────────────────── 状态初始化 ─────────────────────────
def init_state():
    if "agent_info" not in st.session_state:
        st.session_state["agent_info"] = {
            "key": DEFAULT_MODEL_KEY,
            "agent": build_agent(DEFAULT_MODEL_KEY),
        }
    if "sessions" not in st.session_state:
        st.session_state["sessions"] = []  # [{id, name, image|None, turns:[]}]
    if "active_session_id" not in st.session_state:
        st.session_state["active_session_id"] = None
    if "staged_image" not in st.session_state:
        st.session_state["staged_image"] = None  # {"bytes","type","name"} | None
    if "params" not in st.session_state:
        st.session_state["params"] = {
            "w1": 0.9, "w2": 0.1, "fit_threshold": 0.65,
            "membership_k": 50, "slice_k": 50, "top_p": 9,
        }


def get_active_session():
    sid = st.session_state["active_session_id"]
    for s in st.session_state["sessions"]:
        if s["id"] == sid:
            return s
    return None


def new_session(image_data):
    session = {
        "id": time.time_ns(),
        "name": image_data["name"] if image_data else "文本会话",
        "image": image_data,  # None 表示纯文本会话
        "turns": [],          # [{user_text, user_image, ai_text, trace, latency_ms, error}]
    }
    st.session_state["sessions"].append(session)
    st.session_state["active_session_id"] = session["id"]
    return session


def session_image_file(session):
    """把会话图片包装成 file-like（带 .type），供 execute_stream 使用"""
    img = session.get("image")
    if not img:
        return None
    bio = io.BytesIO(img["bytes"])
    bio.type = img.get("type") or "image/png"
    return bio


# ───────────────────────── 侧边栏 ─────────────────────────
def render_sidebar():
    with st.sidebar:
        st.header("🤖 模型选择")
        registry_keys = list(MODEL_REGISTRY.keys())
        option_map = {
            f"{MODEL_REGISTRY[k]['group']}: {MODEL_REGISTRY[k]['label']}": k
            for k in registry_keys
        }
        current = st.session_state["agent_info"]["key"]
        current_opt = f"{MODEL_REGISTRY[current]['group']}: {MODEL_REGISTRY[current]['label']}"
        options = list(option_map.keys())
        sel = st.selectbox("模型", options, index=options.index(current_opt))
        sel_key = option_map[sel]
        if sel_key != current:
            try:
                st.session_state["agent_info"] = {"key": sel_key, "agent": build_agent(sel_key)}
                st.toast(f"已切换到 {sel}")
            except Exception as e:
                st.error(f"模型切换失败（保持原模型）：{e}")

        if not MODEL_REGISTRY[sel_key]["supports_image"]:
            has_img = get_active_session() and get_active_session().get("image") is not None
            if has_img or st.session_state["staged_image"] is not None:
                st.warning("⚠️ 当前模型不支持图像，将仅用文本回答。")

        st.divider()
        st.header("⚙️ 检索参数")
        p = st.session_state["params"]
        w1 = st.slider("w1 相似度权重", 0.0, 1.0, p["w1"], 0.05)
        st.caption(f"w2 正确率权重 = {1 - w1:.2f}（自动联动，和为 1）")
        fit_threshold = st.slider("fit_threshold 命中阈值", 0.0, 1.0, p["fit_threshold"], 0.05)
        membership_k = st.number_input("membership_k 日志检索数", 1, 200, p["membership_k"])
        slice_k = st.number_input("slice_k 切片检索数", 1, 200, p["slice_k"])
        top_p = st.number_input("top_p 保留数", 1, 50, p["top_p"])
        st.session_state["params"] = {
            "w1": w1, "w2": round(1 - w1, 2), "fit_threshold": fit_threshold,
            "membership_k": int(membership_k), "slice_k": int(slice_k), "top_p": int(top_p),
        }

        st.divider()
        st.header("🔍 本次检索过程")
        session = get_active_session()
        turn = session["turns"][-1] if session and session["turns"] else None
        trace = turn["trace"] if turn else None
        if trace is None:
            st.caption("本轮无检索调用")
        else:
            render_trace(trace)

        st.divider()
        st.header("📊 会话统计")
        stats = MembershipHybridService.get_membership_stats_static()
        avg_latency = ToolLatencyTracker.get_global_avg_latency()
        st.markdown(
            f"隶属度命中率 **{stats['hit_rate'] * 100:.1f}%**"
            f"（{stats['hit_calls']}/{stats['total_calls']}）"
        )
        st.markdown(f"平均检索耗时 **{avg_latency * 1000:.0f} ms**")

        st.divider()
        st.header("💬 会话")
        if st.button("➕ 新建会话", use_container_width=True):
            st.session_state["active_session_id"] = None
            st.session_state["staged_image"] = None
            st.rerun()
        for s in reversed(st.session_state["sessions"]):
            label = f"{s['name']}（{len(s['turns'])}轮）"
            if st.button(label, key=f"session_{s['id']}", use_container_width=True):
                st.session_state["active_session_id"] = s["id"]
                st.session_state["staged_image"] = None
                st.rerun()


def render_trace(trace: dict):
    """侧边栏：渲染一份 RetrievalTrace"""
    decision = trace.get("decision")
    p = trace.get("params_used", {})
    m = trace.get("membership") or {}
    max_mu = m.get("max_membership", 0.0)
    threshold = p.get("fit_threshold")
    if decision == "membership_hit":
        st.success(f"✅ 隶属度命中  μ_max={max_mu:.4f} ≥ {threshold}")
    elif decision == "slice_fallback":
        st.warning(f"🔄 未命中 → 降级切片检索  μ_max={max_mu:.4f} < {threshold}")
    st.caption(
        f"参数: w1={p.get('w1')} w2={p.get('w2')} 阈值={threshold} "
        f"membership_k={p.get('membership_k')} slice_k={p.get('slice_k')} top_p={p.get('top_p')}"
    )
    if trace.get("error"):
        st.error(f"检索异常: {trace['error']}")

    if trace.get("rag_context"):
        with st.expander(f"阶段0 RAG参考 · {trace.get('stage0_latency', 0):.0f}ms"):
            st.markdown(trace["rag_context"])

    if m:
        with st.expander(f"阶段1 隶属度 · {trace.get('stage1_latency', 0):.0f}ms"):
            st.markdown(
                f"合格日志 **{m.get('qualified_log_count')}** 条，"
                f"max μ = **{m.get('max_membership', 0.0):.4f}**"
            )
            logs = m.get("top_logs") or []
            if logs:
                st.dataframe(
                    [
                        {
                            "id": log.get("id"),
                            "历史问题": (log.get("question") or "")[:60],
                            "sim": round(log.get("similarity", 0.0), 4),
                            "correct": round(log.get("correctness_score", 0.0), 4),
                            "μ": round(log.get("membership_degree", 0.0), 4),
                            "切片数": len(log.get("retrieved_slices") or []),
                        }
                        for log in logs
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    if decision == "slice_fallback":
        with st.expander(f"阶段2 切片检索 · {trace.get('stage2_latency', 0):.0f}ms"):
            slices = trace.get("slices") or []
            st.markdown(f"保留 **{len(slices)}** 条切片")
            for s in slices:
                st.markdown(f"`{s.get('slice_id')}` · {s.get('score_type')} **{s.get('score', 0.0):.4f}**")

    st.caption(f"总耗时 {trace.get('total_latency', 0):.0f} ms")


# ───────────────────────── 主区域 ─────────────────────────
def render_slice_card(s: dict):
    """主区：单条切片卡片（得分徽标 + slice_id + 内容截断/展开）"""
    score = s.get("score", 0.0)
    stype = s.get("score_type", "")
    badge = f"μ {score:.3f}" if stype == "membership" else f"sim {score:.3f}"
    content = s.get("content", "")
    slice_id = s.get("slice_id", "")
    if len(content) > 200:
        with st.expander(f"[{badge}] 切片 `{slice_id}` — 点击展开全文"):
            st.markdown(content)
    else:
        st.markdown(f"[{badge}] 切片 `{slice_id}`: {content}")


def render_turn(turn: dict, idx: int):
    """主区：渲染一轮问答（用户消息 → 召回切片 → AI 回复 → 行尾小字）"""
    with st.container(border=True):
        st.markdown(f"**第 {idx} 轮**")
        if turn["user_image"]:
            col1, col2 = st.columns([1, 5])
            col1.image(Image.open(io.BytesIO(turn["user_image"]["bytes"])), width=140,
                       caption=turn["user_image"]["name"])
            col2.markdown(turn["user_text"])
        else:
            st.markdown(turn["user_text"])

        slices = (turn.get("trace") or {}).get("slices") or []
        if slices:
            st.markdown(f"**召回切片（{len(slices)} 条）**")
            for s in slices:
                render_slice_card(s)
        elif turn.get("trace") is not None:
            st.caption("未检索到相关切片")

        if turn.get("error"):
            st.error(turn["error"])
        else:
            st.chat_message("assistant").markdown(turn.get("ai_text") or "（无回答）")

        trace = turn.get("trace")
        if trace:
            tag = "✅ 命中" if trace.get("decision") == "membership_hit" else "🔄 降级"
            st.caption(f"第{idx}轮工具过程见侧边栏 | 耗时 {turn.get('latency_ms', 0):.0f} ms | {tag}")


def render_main():
    st.title("🛰️ SAR遥感问答系统")
    session = get_active_session()
    if not st.session_state["sessions"]:
        st.info(
            "上传一张 SAR 遥感图并提问，系统将展示检索过程（侧边栏）与最终答案（主区）。\n\n"
            "支持追问同一图片的多个问题；换图或点「新建会话」开启新会话。"
        )
        return
    # 最新一轮在上（贴近输入体验），历史在下；跳过正在流式渲染的本轮（避免与现场渲染重复）
    for idx in range(len(session["turns"]), 0, -1):
        turn = session["turns"][idx - 1]
        if turn.get("ai_text") == "" and turn.get("error") is None and turn.get("trace") is None:
            continue
        render_turn(turn, idx)


# ───────────────────────── 底部固定栏 ─────────────────────────
def render_bottom_bar() -> str | None:
    """渲染底部固定栏，返回用户输入的问题（无输入返回 None）"""
    with st.container():
        st.markdown('<div class="focused-bottom-bar">', unsafe_allow_html=True)

        staged = st.session_state["staged_image"]
        if staged is not None:
            try:
                preview_img = Image.open(io.BytesIO(staged["bytes"]))
                c1, c2, c3 = st.columns([1, 4, 1])
                c1.image(preview_img, width=80)
                c2.caption(f"待发送图片: {staged['name']}")
                if c3.button("✕ 清除", key="clear_image"):
                    st.session_state["staged_image"] = None
                    st.rerun()
            except Exception as e:
                st.error(f"图片预览失败: {str(e)}")

        prompt = st.chat_input("请输入您关于遥感图像的问题...")

        uploaded_image = st.file_uploader(
            "上传图片",
            type=["png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff"],
            accept_multiple_files=False,
            key="image_uploader",
        )
        if uploaded_image is not None:
            st.session_state["staged_image"] = {
                "bytes": uploaded_image.getvalue(),
                "type": uploaded_image.type or "image/png",
                "name": uploaded_image.name,
            }
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
    return prompt


# ───────────────────────── 提问处理 ─────────────────────────
def handle_prompt(prompt: str):
    """处理一轮提问：会话判定 → 滑杆生效 → 执行 agent → 抓取 trace"""
    params = st.session_state["params"]
    rag_rscsv_service.set_runtime_params(**params)  # 滑杆参数实时生效

    staged = st.session_state["staged_image"]
    session = get_active_session()
    staged_differs = (
        staged is not None
        and (session is None or session.get("image") is None or session["image"]["bytes"] != staged["bytes"])
    )
    if staged_differs or (staged is None and session is None):
        session = new_session(staged)  # 新图片 / 无会话 → 新会话（staged 可为 None）

    turn_image = session.get("image")  # 追问时取会话图片（若 staged 与之一致）
    history = []
    for t in session["turns"]:
        history.append({"role": "user", "content": t["user_text"]})
        history.append({"role": "assistant", "content": t.get("ai_text") or ""})

    turn = {
        "user_text": prompt,
        "user_image": turn_image,
        "ai_text": "",
        "trace": None,
        "latency_ms": 0,
        "error": None,
    }
    session["turns"].append(turn)

    # ── 现场渲染本轮（流式）──
    st.chat_message("user").markdown(prompt)
    if turn_image:
        st.image(Image.open(io.BytesIO(turn_image["bytes"])), width=160,
                 caption=turn_image["name"])
    agent = st.session_state["agent_info"]["agent"]
    chunks = []
    start = time.time()
    try:
        with st.status("思考中... 调用工具 rag_rscsv 检索", expanded=False) as status:
            status.write("模型推理中...")
            placeholder = st.chat_message("assistant").empty()
            for chunk in agent.execute_stream(prompt, image_file=session_image_file(session), history=history):
                chunks.append(chunk)
                placeholder.markdown("".join(chunks))
            turn["ai_text"] = "".join(chunks)
            status.update(label="完成", state="complete")
    except Exception as e:
        turn["error"] = str(e)
        st.error(f"执行出错：{e}")
    turn["latency_ms"] = (time.time() - start) * 1000

    # 抓取本次检索过程（按 query 匹配，防上一轮残留）；转为 dict 供 UI 渲染
    trace = agent.get_last_trace()
    turn["trace"] = (
        asdict(trace) if (trace is not None and trace.query == prompt) else None
    )

    rag_rscsv_service.clear_runtime_params()
    st.session_state["staged_image"] = None
    st.rerun()


# ───────────────────────── 入口 ─────────────────────────
init_state()
render_sidebar()
prompt = render_bottom_bar()
if prompt:
    handle_prompt(prompt)
render_main()
```

- [ ] **Step 2: 语法检查**

Run: `python -m py_compile agent/mainapp.py`
Expected: 无输出（0 退出码）

- [ ] **Step 3: 启动 UI 冒烟**

Run: `streamlit run agent/mainapp.py`（后台启动或另开终端），浏览器打开后按下述清单走查：

1. 空态：显示欢迎语（"上传一张 SAR 遥感图并提问..."）
2. 侧边栏：模型下拉默认"多模态: Doubao Seed 2.0 Mini"，滑杆默认 w1=0.9 / 阈值 0.65
3. 上传一张 SAR 图（如 `C:\dataset\SAR-TEXT\SAR-TEXT-data\Image\QXSLAB_SAROPT\7756.png`，不存在则任意 png）→ 底部栏出现缩略图
4. 提问（如 "Is there evidence of water bodies in this landscape?"）→ 回答流式输出；完成后：
   - 主区出现本轮卡片：用户消息（含图缩略图）→ "召回切片" 卡片（徽标 `μ`/`sim` + slice_id）→ AI 回复 → 行尾"✅ 命中 / 🔄 降级 + 耗时"
   - 侧边栏"本次检索过程"：命中/降级徽标、参数行、阶段0/1/2 expander、总耗时
5. 追问第二个问题 → 新卡片在上，历史卡片仍在
6. 把 w1 滑到 0.5、阈值滑到 0.3 → 再提问 → 侧边栏参数行显示新值
7. 切换"文本: Doubao 1.5 Lite" → toast"已切换到"；有图时出现"不支持图像"警告；提问走纯文本
8. 切回多模态；点"➕ 新建会话" → 空态；上传新图提问 → 新会话；侧边栏会话列表出现两个会话，点击可切回
9. 无图片直接提问 → 正常回答，侧边栏"本轮无检索调用"或文本检索
10. 图片预览失败/API 报错 → 错误信息可见，应用不崩

- [ ] **Step 4: 提交**

```bash
git add agent/mainapp.py
git commit -m "feat: mainapp 重写为双区 UI（侧边栏工具过程 + 主区切片与回复）

- 侧边栏：模型分组下拉/检索参数滑杆/本次检索过程(阶段0/1/2+判定+耗时)/会话统计/会话列表
- 主区：图片级多轮会话流，每轮 用户消息→召回切片卡片→AI回复，最新在上
- 底部固定栏：图片上传预览+清除、提问输入
- 滑杆参数经 set_runtime_params 实时生效，工具契约不变

更新时间：2026-08-05"
```

---

### Task 8: 端到端验证与清理

**Files:**
- Delete (temp): `agent/rag/verify_w1w2.py`、`agent/rag/verify_trace.py`、`agent/verify_mainagent.py`
- Run: 回归验证

- [ ] **Step 1: 清理临时验证脚本**

```bash
rm agent/rag/verify_w1w2.py agent/rag/verify_trace.py agent/verify_mainagent.py
```

- [ ] **Step 2: CLI 回归（agent 主链路）**

Run: `python agent/mainagent.py`
Expected: 与 Task 4 Step 6 一致——正常输出回答；确认工具字符串格式与改动前一致（含 `【匹配隶属度缓存】`/`【匹配基础切片】` 头）

- [ ] **Step 3: UI 完整冒烟**

Run: `streamlit run agent/mainapp.py`
Expected: Task 7 Step 3 清单全部通过

- [ ] **Step 4: 基准回归（best-effort）**

Run: `python benchmark/main_eval.py`（检查文件顶部 `MODEL_KEY`/`PIPELINE_MODE` 配置；若当前配置指向的评估数据集路径不存在或样本量过大，将 `MODEL_KEY` 临时设为 `agent-text-doubao-seed` 并跑通"预测"步骤的小样本后还原——benchmark 文件本身零改动）
Expected: 预测步骤正常产出（检索服务改动不破坏 benchmark 路径）；若环境/数据不可用，记录原因并在交付说明中注明"未执行"

- [ ] **Step 5: 提交清理**

```bash
git add -u
git commit -m "chore: 清理临时验证脚本

- 删除 verify_w1w2 / verify_trace / verify_mainagent

更新时间：2026-08-05"
```

---

## 自审记录（写完计划后逐项核对）

- **Spec 覆盖**：RetrievalTrace ✔(T2)、w1/w2 透传 ✔(T1)、滑杆运行时生效 ✔(T2 set_runtime_params + T7)、模型分组下拉+双默认值 ✔(T6/T7)、图片级多轮 ✔(T4/T5/T7)、侧边栏过程面板 ✔(T7 render_trace)、主区切片卡片 ✔(T7 render_slice_card)、qwen3.7-plus ✔(T3)、验证清单 ✔(T8)
- **占位符扫描**：所有步骤含完整代码；无 TBD
- **类型一致性**：`RetrievalTrace` 字段在 T2 定义、T7 渲染同名使用；`execute_stream(query, image_file, history)` 在 T4/T5 定义、T7 调用一致；`set_runtime_params(**params)` 的 key 集与 T7 滑杆字典 key 一致；`trace.query == prompt` 匹配规则在 T2/T4 定义、T7 使用一致
