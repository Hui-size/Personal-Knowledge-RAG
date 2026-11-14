
import json #用于 Python 数据与 JSON 格式之间的转换。
import os   #Python 自带的操作系统接口模块
from pathlib import Path

import faiss
import numpy as np

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


load_dotenv()#加载环境变量

VECTORSTORE_DIR = Path("vectorstore")   #定义一个变量，表示向量数据库的保存目录。
MAX_DISTANCE = 0.95
# 1. 加载 FAISS 索引
index = faiss.read_index(
    str(VECTORSTORE_DIR / "index.faiss")
)

# 2. 加载原始文本
with open(VECTORSTORE_DIR / "chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)   #从文件对象 f 中读取 JSON 数据；JSON 数据转换为 Python

if index.ntotal != len(chunks):
    raise ValueError("FAISS 索引与文本数据数量不一致")

# 3. 初始化 Embedding 模型
embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("PARATERA_API_KEY"),
    base_url=os.getenv("PARATERA_BASE_URL"),
)

# 4. 输入用户问题
query = input("请输入你的问题：")

# 5. 将问题转换成向量
query_vector = embeddings.embed_query(query)

query_vector = np.asarray(
    [query_vector],#增加一个外层维度，让数据变成二维数组。注意FAISS 的 search() 接收的是二维查询向量数组
    dtype="float32"
)

# 6. 使用 FAISS 检索
k = 3
distances, indices = index.search(query_vector, k)

# 7. 输出检索结果

print("\n检索结果：")

found = False#标记变量，用来记录有没有找到符合条件的资料

for rank, (idx, distance) in enumerate(
    zip(indices[0], distances[0]),
    start=1
):
    if idx == -1 or distance > MAX_DISTANCE:#表示 FAISS 没有返回有效索引。后面表示距离超过我们设定的阈值。
        continue

    found = True

    chunk = chunks[int(idx)]
    metadata = chunk["metadata"]

    print(f"\n第 {rank} 条结果")
    print("距离值：", float(distance))
    print("来源：", metadata["source"])

    if "page" in metadata:
        print("页码：", metadata["page"])

    print("内容：")
    print(chunk["page_content"])

if not found:
    print("没有找到足够相关的资料。")
