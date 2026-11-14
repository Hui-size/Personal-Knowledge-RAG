import os
import streamlit as st
import subprocess
import sys
from pathlib import Path
from rag_core import answer_question, reload_knowledge_base

# 1. 设置网页基本信息
st.set_page_config(
    page_title="Personal Knowledge RAG",
    page_icon="📚",
)

st.title("Personal Knowledge RAG")
st.caption("基于 LangChain + FAISS + DeepSeek 的个人知识库助手")

# 知识库文件上传
with st.sidebar:
    st.header("知识库管理")

    uploaded_files = st.file_uploader(
        "上传学习资料",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )

    if st.button("保存上传文件", disabled=not uploaded_files):

        folder_map = {
            ".pdf": "pdf",
            ".md": "markdown",
            ".txt": "txt",
        }

        saved_files = []

        for uploaded_file in uploaded_files:
            file_name = Path(uploaded_file.name).name
            suffix = Path(file_name).suffix.lower()

            target_dir = Path("data") / folder_map[suffix]
            target_dir.mkdir(parents=True, exist_ok=True)

            target_path = target_dir / file_name

            target_path.write_bytes(
                uploaded_file.getvalue()
            )

            saved_files.append(str(target_path))

        st.success(f"成功保存 {len(saved_files)} 个文件！")

        for file_path in saved_files:
            st.write(file_path)
        
    st.divider()

    if st.button("更新知识库"):

        with st.spinner("正在检测文件变化并更新知识库..."):

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            result = subprocess.run(
                [sys.executable, "update_index.py"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )

        if result.returncode == 0:
            reload_knowledge_base()

            st.success("知识库更新检查完成！")

            if result.stdout:
                st.code(result.stdout)

        else:
            st.error("知识库更新失败！")

            st.code(result.stderr or result.stdout)



# 2. 初始化对话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# 知识库文件列表与对话管理
with st.sidebar:
    st.divider()
    st.subheader("当前知识库文件")

    data_dir = Path("data")
    file_list = []

    for folder in ["pdf", "markdown", "txt"]:
        folder_path = data_dir / folder

        if folder_path.exists():
            file_list.extend(
                p for p in folder_path.iterdir() if p.is_file()
            )

    if file_list:
        for file_path in sorted(file_list):
            st.write(f"📄 {file_path.relative_to(data_dir)}")
    else:
        st.caption("知识库暂无文件")

    st.divider()

    if st.button("清空当前对话"):
        st.session_state.messages = []
        st.session_state.chat_history = []


# 3. 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("查看参考资料"):
                for source in message["sources"]:
                    st.write(source)


# 4. 接收用户输入
query = st.chat_input("请输入你的问题...")

if query:

    # 显示用户问题
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # 5. 调用 RAG 核心功能
    with st.chat_message("assistant"):

        with st.spinner("正在检索知识库并生成回答..."):
            answer, sources, search_query = answer_question(
                query,
                st.session_state.chat_history,
            )

        st.markdown(answer)

        if search_query != query:
            st.caption(f"检索问题：{search_query}")

        if sources:
            with st.expander("查看参考资料"):
                for source in sources:
                    st.write(source)

    # 6. 保存本轮消息
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
