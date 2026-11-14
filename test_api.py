
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

api_key = os.getenv("PARATERA_API_KEY")
base_url = os.getenv("PARATERA_BASE_URL")

# 1. 测试 Embedding API
try:
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        api_key=api_key,
        base_url=base_url,
    )

    vector = embeddings.embed_query("RAG 是一种结合向量检索与大语言模型的技术。")

    print("Embedding API 连接成功")
    print("向量维度：", len(vector))
    print("向量前五个值：", vector[:5])

except Exception as e:
    print("Embedding API 测试失败：", e)


# 2. 测试 DeepSeek API
try:
    model = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        api_key=api_key,
        base_url=base_url,
    )

    response = model.invoke("请只回复：DeepSeek API 连接成功。")

    print("\nDeepSeek API 连接成功")
    print("模型回复：", response.content)

except Exception as e:
    print("DeepSeek API 测试失败：", e)
