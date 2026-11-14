
import json
import os
from pathlib import Path

import faiss
import numpy as np

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder#在 Prompt 中预留历史消息的位置
from langchain_core.messages import HumanMessage, AIMessage #保存用户发送的消息;保存 AI 返回的消息


load_dotenv()
#定义文件夹路径
VECTORSTORE_DIR = Path("vectorstore")#创建一个 Path 路径对象，指向当前工作目录下的 vectorstore 文件夹。
MAX_DISTANCE = 0.95
# 1. 加载 FAISS 索引
index = faiss.read_index(
    str(VECTORSTORE_DIR / "index.faiss")    #将 Path 对象转换成字符串路径。
)

# 2. 加载原始文本
with open(VECTORSTORE_DIR / "chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

if index.ntotal != len(chunks):
    raise ValueError("FAISS 索引与文本数据数量不一致")

# 3. 初始化 Embedding 模型
embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("PARATERA_API_KEY"),
    base_url=os.getenv("PARATERA_BASE_URL"),
)

# 4. 初始化 DeepSeek 模型
model = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL"),
    api_key=os.getenv("PARATERA_API_KEY"),
    base_url=os.getenv("PARATERA_BASE_URL"),
)


# 5. 创建问题改写 Prompt
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "根据对话历史，将用户的最新问题改写成一个独立、完整的检索问题。"
            "保留用户原本的提问意图。"
            "如果最新问题本身已经完整，直接返回原问题。"
            "只输出改写后的问题，不要回答。"
        ),
        MessagesPlaceholder(variable_name="chat_history"),#在 Prompt 中预留一个位置，用来放入历史聊天记录
        ("human", "{input}"),
    ]
)

contextualize_chain = contextualize_prompt | model


# 6. 创建 RAG 问答 Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个个人知识库助手。请优先根据提供的资料回答问题。"
            "不要编造资料中没有的信息。"
            "如果资料不足，请明确说明。"
            "回答中的事实性内容需要标注对应的资料编号。"
            "只能使用提供的资料编号，不得编造引用编号。"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "用户问题：{question}\n\n"
            "参考资料：\n{context}"
        ),
    ]
)

chain = prompt | model
# 初始化聊天历史记录
chat_history = []

# 7. 用户提问
while True:
    query = input("\n请输入问题（输入 exit 退出）：")

    if query.lower() == "exit":
        break

    if not query.strip():
        continue

    # 8. 根据历史对话改写问题
    if chat_history:
        rewritten_query = contextualize_chain.invoke(
            {
                "chat_history": chat_history,
                "input": query,
            }
        )

        search_query = rewritten_query.content.strip() or query
    else:
        search_query = query

    print("\n用于检索的问题：", search_query)

    # 9. 将问题转换为向量
    query_vector = embeddings.embed_query(search_query)

    query_vector = np.asarray(
        [query_vector],
        dtype="float32"
    )

    # 10. FAISS 检索
    k = min(3, index.ntotal)
    distances, indices = index.search(query_vector, k)

    # 11. 获取符合阈值的原始文本
    context_parts = []
    source_lines = []

    for idx, distance in zip(indices[0], distances[0]):
        if idx == -1 or distance > MAX_DISTANCE:
            continue

        chunk = chunks[int(idx)]
        metadata = chunk["metadata"]

        source = metadata["source"]

        if "page" in metadata:
            source += f"（第 {metadata['page']} 页）"

        reference_id = len(context_parts) + 1

        context_parts.append(
            f"[资料{reference_id}]\n"
            f"来源：{source}\n"
            f"内容：{chunk['page_content']}"
        )

        source_lines.append(
            f"[资料{reference_id}] {source}"
        )

    # 12. 判断是否找到相关资料
    if not context_parts:
        fallback = "知识库中没有找到足够相关的资料，暂时无法回答。"

        print("\n" + fallback)

        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=fallback))

        chat_history = chat_history[-6:]

        continue

    context = "\n\n".join(context_parts)

    # 13. 调用 DeepSeek 生成回答
    response = chain.invoke(
        {
            "question": query,
            "context": context,
            "chat_history": chat_history,
        }
    )

    print("\nDeepSeek 回答：")
    print(response.content)

    print("\n本次参考资料：")

    for source in source_lines:
        print(source)

    # 14. 保存本轮对话
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response.content))

    # 只保留最近三轮对话
    chat_history = chat_history[-6:]
