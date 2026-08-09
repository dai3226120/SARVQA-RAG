"""SAR 遥感问答系统 - Streamlit UI

侧边栏（工具过程窗口）：模型选择 / 检索参数滑杆 / 本次检索过程（阶段0/1/2+判定+耗时）/ 会话统计 / 会话列表
主区域：会话流（用户消息 → 召回切片卡片 → AI 回复，最新在上）
底部固定栏：图片上传（预览+清除）+ 提问输入
"""
import io
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import streamlit as st
from PIL import Image

# streamlit run 时只有脚本目录（agent/）会自动进 sys.path，项目根目录不会；
# mainagent 等模块在第 3 行就 import utils.*，须在任何项目内导入前注入路径
_current_file = Path(__file__).absolute()
_agent_dir = _current_file.parent
_project_root = _agent_dir.parent
for _p in (_project_root, _agent_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from agent_registry import MODEL_REGISTRY, DEFAULT_MODEL_KEY, build_agent
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
/* 侧边栏占满整个视口高度（streamlit 用内联 height:auto 覆盖样式表，需 !important；
   否则主内容短时侧边栏只占内容高度，下方露出页面背景） */
[data-testid="stSidebar"] {
    position: sticky !important;
    top: 0 !important;
    height: 100vh !important;
}
/* 侧边栏内容底部留白，避免被固定底部栏遮挡 */
[data-testid="stSidebarContent"] {
    padding-bottom: 140px;
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
            "slice_k": 50,
        }


def get_active_session():
    sid = st.session_state["active_session_id"]
    for s in st.session_state["sessions"]:
        if s["id"] == sid:
            return s
    return None


def sync_stats_scope():
    """把会话级统计范围对齐到当前活动会话（切会话/新会话后调用，保证侧边栏统计与当前会话一致）"""
    session = get_active_session()
    sid = session["id"] if session else None
    MembershipHybridService.set_current_session(sid)
    ToolLatencyTracker.set_current_session(sid)


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
        # membership_k / top_p 已弃用（隶属度固定 top-1；检索多少就送多少），不再提供滑杆
        slice_k = st.number_input("slice_k 切片检索数", 1, 200, p["slice_k"])
        st.session_state["params"] = {
            "w1": w1, "w2": round(1 - w1, 2), "fit_threshold": fit_threshold,
            "slice_k": int(slice_k),
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
        # 会话级统计：先对齐到当前活动会话，再读取该会话的累计值
        sync_stats_scope()
        stats = MembershipHybridService.get_session_stats_static()
        latency_stats = ToolLatencyTracker.get_session_stats()
        st.markdown(
            f"隶属度命中率 **{stats['hit_rate'] * 100:.1f}%**"
            f"（{stats['hit_calls']}/{stats['total_calls']}）"
        )
        st.markdown(f"平均检索耗时 **{latency_stats['avg_latency'] * 1000:.0f} ms**")

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
        f"slice_k={p.get('slice_k')}"
    )
    if trace.get("error"):
        st.error(f"检索异常: {trace['error']}")

    if trace.get("rag_context"):
        with st.expander(f"阶段0 RAG参考 · {trace.get('stage0_latency', 0):.0f}ms"):
            st.markdown(trace["rag_context"])

    if m:
        with st.expander(f"阶段1 隶属度 · {trace.get('stage1_latency', 0):.0f}ms"):
            st.markdown(
                f"归属语境分类 **{m.get('qualified_log_count')}** 个，"
                f"max μ = **{m.get('max_membership', 0.0):.4f}**"
            )
            clusters = m.get("top_logs") or []
            if clusters:
                st.dataframe(
                    [
                        {
                            "类别": c.get("cluster_id"),
                            "中心相似度": round(c.get("center_sim", 0.0), 4),
                            "μ": round(c.get("mu", 0.0), 4),
                            "类内切片数": c.get("slice_count", 0),
                        }
                        for c in clusters
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
        if turn.get("user_render_error"):
            # 现场渲染失败（如损坏图片），回看时降级为纯文本，跳过图片渲染
            st.markdown(turn["user_text"])
        elif turn["user_image"]:
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
    if session is None or not st.session_state["sessions"]:
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
def _on_image_upload():
    """file_uploader on_change 回调：在 widget 实例化前把上传暂存为待发送图片"""
    uploaded = st.session_state.get("image_uploader")
    if uploaded is not None:
        st.session_state["staged_image"] = {
            "bytes": uploaded.getvalue(),
            "type": uploaded.type or "image/png",
            "name": uploaded.name,
        }
    else:
        # 用户在 uploader 内移除文件时，同步清除待发送缓存
        st.session_state["staged_image"] = None


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

        # 上传通过 on_change 回调暂存（回调先于 widget 实例化执行，修改 state 合法；
        # 且仅值变化时触发，rerun 不会重入，无死循环）
        st.file_uploader(
            "上传图片",
            type=["png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff"],
            accept_multiple_files=False,
            key="image_uploader",
            on_change=_on_image_upload,
        )

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

    # 新建/切换会话后，会话级统计范围对齐到当前会话
    sync_stats_scope()

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
    try:
        st.chat_message("user").markdown(prompt)
        if turn_image:
            st.image(Image.open(io.BytesIO(turn_image["bytes"])), width=160,
                     caption=turn_image["name"])
    except Exception as e:
        # 损坏图片等渲染异常不阻断问答流程
        turn["user_render_error"] = str(e)
        st.warning(f"用户消息渲染失败（不影响回答）：{e}")
    agent = st.session_state["agent_info"]["agent"]
    # 文本模型不传图（spec：仅用文本回答）；图片仍参与用户消息的 UI 渲染
    agent_key = st.session_state["agent_info"]["key"]
    supports_image = MODEL_REGISTRY[agent_key]["supports_image"]
    img_file = session_image_file(session) if supports_image else None
    prev_trace = agent.get_last_trace()  # 提问前的 trace（用于判断本轮是否新增检索）
    chunks = []
    start = time.time()
    try:
        with st.status("思考中... 调用工具 rag_rscsv 检索", expanded=True) as status:
            status.write("模型推理中...")
            placeholder = st.chat_message("assistant").empty()
            for chunk in agent.execute_stream(prompt, image_file=img_file, history=history):
                chunks.append(chunk)
                placeholder.markdown("".join(chunks))
            turn["ai_text"] = "".join(chunks)
            status.update(label="完成", state="complete")
    except Exception as e:
        turn["error"] = str(e)
        status.update(label="出错", state="error")
        st.error(f"执行出错：{e}")
    finally:
        # 无论成功与否，都恢复运行期参数并清空待发送图片
        rag_rscsv_service.clear_runtime_params()
        st.session_state["staged_image"] = None
    turn["latency_ms"] = (time.time() - start) * 1000

    # 抓取本次检索过程（转为 dict 供 UI 渲染）：
    # 本轮未触发新检索（trace 对象未变）→ 不显示，避免上一轮残留串扰；
    # 本轮有检索 → 直接取最新 trace（LLM 改写问题导致 query 不精确一致时
    # 也按 spec 兜底显示，不再误报"本轮无检索调用"）
    trace = agent.get_last_trace()
    turn["trace"] = asdict(trace) if (trace is not None and trace is not prev_trace) else None

    st.rerun()


# ───────────────────────── 入口 ─────────────────────────
init_state()
render_sidebar()
prompt = render_bottom_bar()
if prompt:
    handle_prompt(prompt)
render_main()
