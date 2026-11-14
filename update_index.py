
import json
import os
from pathlib import Path

import faiss
import numpy as np

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from build_index import DATA_DIR, VECTORSTORE_DIR, load_documents, build_manifest


load_dotenv()

# 1. 加载原来的知识库
index = faiss.read_index(
    str(VECTORSTORE_DIR / "index.faiss")
)

with open(VECTORSTORE_DIR / "chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open(VECTORSTORE_DIR / "manifest.json", "r", encoding="utf-8") as f:
    old_manifest = json.load(f)

if index.ntotal != len(chunks):
    raise ValueError("FAISS 索引与文本数据数量不一致")


# 2. 获取当前文件状态
current_manifest = build_manifest()
                #当前文件清单           之前保存的文件清单
added_files = set(current_manifest) - set(old_manifest)#set()：将数据转换成集合,-：集合差集，找出当前有、旧清单没有的文件

modified_files = {
    path for path in current_manifest.keys() & old_manifest.keys()  #.keys()：获取字典中的所有键（文件路径）;集合交集，找出两个清单中都存在的文件
    if current_manifest[path] != old_manifest[path]#哈希值不同，就说明文件内容发生了变化。
}

deleted_files = set(old_manifest) - set(current_manifest)

changed_files = added_files | modified_files | deleted_files

print("新增文件：", len(added_files))
print("修改文件：", len(modified_files))
print("删除文件：", len(deleted_files))

if not changed_files:
    print("知识库没有变化，无需更新！")
    raise SystemExit#主动抛出程序退出异常，结束当前程序


# 3. 保留没有变化的文本块和向量
retained_chunks = []
retained_vectors = []

for i, chunk in enumerate(chunks):
    file_key = Path(
        chunk["metadata"]["source"]
    ).relative_to(DATA_DIR).as_posix()

    if file_key not in changed_files:
        #保存需要保留的旧文本块。将当前文本块 chunk 添加进去
        retained_chunks.append(chunk)
        retained_vectors.append(index.reconstruct(i))   #根据 FAISS 索引位置，重新取出之前保存的向量。也就是取出旧向量


# 4. 读取新增和修改的文件
updated_files = sorted(added_files | modified_files)

documents = load_documents(
    [DATA_DIR / file_path for file_path in updated_files]
)

# 5. 切分新读取的文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]
)

new_chunks = text_splitter.split_documents(documents)

# 6. 只对新增和修改的文本进行向量化
new_vectors = []

if new_chunks:
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        api_key=os.getenv("PARATERA_API_KEY"),
        base_url=os.getenv("PARATERA_BASE_URL"),
    )

    texts = [chunk.page_content for chunk in new_chunks]
    new_vectors = embeddings.embed_documents(texts)


# 7. 合并旧向量和新向量
all_vectors = retained_vectors + new_vectors

new_index = faiss.IndexFlatL2(index.d)

if all_vectors:
    all_vectors = np.asarray(
        all_vectors,
        dtype="float32"
    )

    new_index.add(all_vectors)


# 8. 合并原始文本
updated_chunks = retained_chunks + [
    {
        "page_content": chunk.page_content,
        "metadata": chunk.metadata,
    }
    for chunk in new_chunks
]


# 9. 保存更新后的知识库
faiss.write_index(
    new_index,
    str(VECTORSTORE_DIR / "index.faiss")
)

with open(VECTORSTORE_DIR / "chunks.json", "w", encoding="utf-8") as f:
    json.dump(updated_chunks, f, ensure_ascii=False, indent=2)

with open(VECTORSTORE_DIR / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(current_manifest, f, ensure_ascii=False, indent=2)


print("\n知识库增量更新成功！")
print("新增文本块：", len(new_chunks))
print("保留文本块：", len(retained_chunks))
print("当前向量总数：", new_index.ntotal)
