import hashlib
import os
import json
from pathlib import Path

import faiss
import numpy as np

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


load_dotenv()

DATA_DIR = Path("data")
VECTORSTORE_DIR = Path("vectorstore")


def load_documents(file_paths=None):
    documents = []

    if file_paths is None:
        file_paths = DATA_DIR.rglob("*")

    for file_path in file_paths:
        file_path = Path(file_path)
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            reader = PdfReader(str(file_path))#使用 pypdf 打开 PDF 文件

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or "" #逐页读取 PDF 内容

                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(file_path),
                                "page": page_number,
                            },
                        )
                    )

        elif suffix in [".md", ".txt"]:
            text = file_path.read_text(encoding="utf-8")

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                        },
                    )
                )

    return documents

# 计算文件的 SHA256 哈希值
def calculate_file_hash(file_path):
    return hashlib.sha256(
        file_path.read_bytes()#读取文件的二进制内容。
    ).hexdigest()#计算 SHA256 哈希值，并将结果转换成十六进制字符串。


# 获取所有资料的文件哈希记录
def build_manifest():#遍历 data 文件夹中的资料，逐个计算哈希值。
    manifest = {}

    for file_path in DATA_DIR.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in [".pdf", ".md", ".txt"]:
            continue

        file_key = file_path.relative_to(DATA_DIR).as_posix()

        manifest[file_key] = calculate_file_hash(file_path)

    return manifest


if __name__ == "__main__":
    # 1. 读取文档
    documents = load_documents()
    print(f"读取到 {len(documents)} 个文档单元")

    # 2. 文本切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"切分后得到 {len(chunks)} 个文本块")

    if not chunks:
        raise ValueError("没有读取到可用于建立知识库的文本")

    # 3. 初始化 Embedding 模型
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        api_key=os.getenv("PARATERA_API_KEY"),
        base_url=os.getenv("PARATERA_BASE_URL"),
    )

    # 4. 将文本转换为向量
    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts) #发送给 GLM-Embedding-3，获取对应的向量。

    vectors = np.asarray(vectors, dtype="float32")  #将返回的向量转换成 NumPy 数组，并指定为 float32 类型。

    print("Embedding 向量化完成")
    print("向量维度：", vectors.shape[1])

    # 5. 创建 FAISS 索引
    index = faiss.IndexFlatL2(vectors.shape[1]) #使用欧氏距离进行精确向量搜索。括号里面获取向量的维度
    index.add(vectors)  #添加到 FAISS 索引中

    print("FAISS 索引建立成功")
    print("向量数量：", index.ntotal)

    # 6. 保存 FAISS 索引
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)  #向量数据库的保存路径；创建文件夹；上级目录不存在，也一起创建；文件夹已经存在，不会报错

    faiss.write_index(#FAISS 索引保存到本地。
        index,
        str(VECTORSTORE_DIR / "index.faiss")#这是 Path 对象的路径拼接操作。
    )

    # 7. 保存文本及元数据
    chunk_data = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    #原始文本和对应的 metadata 保存为 JSON 文件。with 语句可以在代码块执行结束后自动关闭文件
    with open(VECTORSTORE_DIR / "chunks.json", "w", encoding="utf-8") as f:#将打开的文件对象命名为 f
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)  #将Python数据写入JSON 文件,保留中文，不转换成 Unicode 编码,设置 JSON 缩进为2格，方便阅读
    
    # 保存文件哈希记录
    manifest = build_manifest()
    with open(VECTORSTORE_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("文件哈希记录保存成功！")
    print("知识库保存成功！")
