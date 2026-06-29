# import os
# import sys
# from pathlib import Path

# # 1. 确定项目根目录（假设 agent 文件夹位于项目根目录下）
# current_file = Path(__file__).absolute()          # mainapp.py 的完整路径
# agent_dir = current_file.parent                    # agent 目录
# project_root = agent_dir.parent                    # 项目根目录

# # 2. 切换到项目根目录（重要！）
# os.chdir(project_root)

# # 3. 将项目根目录和 agent 目录都加入 Python 搜索路径
# for p in (project_root, agent_dir):
#     if str(p) not in sys.path:
#         sys.path.insert(0, str(p))

# # 可选：打印当前工作目录和环境，方便调试
# print(f"[mainapp] Working directory: {os.getcwd()}")
# print(f"[mainapp] Project root: {project_root}")
# print(f"[mainapp] Agent dir: {agent_dir}")

import streamlit as st
from mainagent import MainAgent
from PIL import Image
import io

st.markdown("""
<style>
/* 给聊天内容区加底部内边距，防止被底部栏遮挡 */
.stAppViewContainer {
    padding-bottom: 180px !important;
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

# 设置页面基本信息
st.title("SAR遥感问答系统")
st.divider()

# 初始化 session_state
if "agent" not in st.session_state:
    st.session_state["agent"] = MainAgent()

if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]

# 缓存当前准备上传的单张图片
if "temp_uploaded_file" not in st.session_state:
    st.session_state["temp_uploaded_file"] = None

# 1. 渲染历史聊天记录
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 2. 构造固定的底层输入区域
with st.container():
    st.markdown('<div class="focused-bottom-bar">', unsafe_allow_html=True)
    
    # 【第一层：图片预览区 → 固定在输入框正上方】
    if st.session_state["temp_uploaded_file"] is not None:
        try:
            # 【修复3：用BytesIO处理图片，兼容TIF/PNG/JPG所有格式，解决预览失败】
            img_bytes = st.session_state["temp_uploaded_file"].getvalue()
            preview_img = Image.open(io.BytesIO(img_bytes))
            # 显示缩略图（小尺寸，不撑开布局）
            st.image(preview_img, width=100, caption="待上传遥感图")
        except Exception as e:
            st.error(f"图片预览失败: {str(e)}")
    
    # 【第二层：聊天输入框】
    prompt = st.chat_input("请输入您关于遥感图像的问题...")
    
    # 【第三层：图片上传按钮】
    uploaded_image = st.file_uploader(
        "上传图片", 
        type=["png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff"],
        accept_multiple_files=False,
        key="image_uploader"
    )
    # 实时更新单图缓存
    if uploaded_image is not None:
        st.session_state["temp_uploaded_file"] = uploaded_image

    st.markdown('</div>', unsafe_allow_html=True)


# 3. 处理发送与大模型交互逻辑
if prompt:
    # 获取当前准备好的图片
    current_image = st.session_state["temp_uploaded_file"]

    # 在聊天框展示用户发送的内容
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_message = []

    with st.spinner("系统正在处理中..."):
        # 执行推理，传入当前单张图片
        res_stream = st.session_state["agent"].execute_stream(prompt, image_file=current_image)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        # 流式输出
        st.chat_message("assistant").write_stream(capture(res_stream, response_message))
        st.session_state["message"].append({"role": "assistant", "content": "".join(response_message)})
    
    # 交互完成后，清空上传图片缓存，并重置页面状态
    st.session_state["temp_uploaded_file"] = None
    st.rerun()