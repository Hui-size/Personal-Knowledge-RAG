# Personal Knowledge RAG

基于 Python、LangChain、FAISS 和 DeepSeek API 构建的个人知识库问答系统。

本项目能够读取个人学习资料，将文档转换为向量并存储到 FAISS，通过向量检索找到相关内容，再调用 DeepSeek API 生成基于知识库的回答。

## 一、项目简介

Personal Knowledge RAG 是一个面向个人学习资料管理的 RAG（Retrieval-Augmented Generation，检索增强生成）应用。

用户可以上传 PDF、Markdown、TXT 等学习资料，系统自动完成文档读取、文本切分、Embedding 向量化和 FAISS 索引构建，并通过 Web 界面进行智能问答。

## 二、主要功能

1. **多格式文档读取**
   - 支持 PDF、Markdown、TXT。
   - 自动读取文档内容并进行文本切分。
   - 保留文档来源和 PDF 页码等元数据。

2. **向量知识库构建**
   - 使用 GLM-Embedding-3 API 生成文本向量。
   - 使用 FAISS 保存和检索向量。
   - 使用 IndexFlatL2 进行精确向量搜索。

3. **知识库增量更新**
   - 使用 SHA256 检测文件变化。
   - 自动识别新增、修改和删除的文件。
   - 避免对未发生变化的文件重复向量化。

4. **智能问答**
   - 使用 DeepSeek-V4.1-Flash 生成回答。
   - 根据检索到的知识库资料回答问题。
   - 知识库资料不足时提示用户。

5. **多轮对话**
   - 保存当前会话的对话历史。
   - 根据历史对话改写不完整的问题。
   - 支持清空当前对话。

6. **Web 操作界面**
   - 使用 Streamlit 构建 Web 界面。
   - 支持多个文件同时上传。
   - 支持网页触发知识库增量更新。
   - 展示知识库文件列表和参考资料来源。

## 三、技术栈

| 技术 | 用途 |
|---|---|
| Python | 项目开发语言 |
| LangChain | RAG 应用组件与流程组织 |
| FAISS | 向量存储与检索 |
| GLM-Embedding-3 | 文本向量化 |
| DeepSeek-V4.1-Flash | 智能问答 |
| Streamlit | Web 界面 |
| NumPy | 向量数据处理 |
| PyPDF | PDF 文档读取 |

模型 API 通过并行云提供，无需在本地部署 Embedding 或聊天模型。

## 四、项目架构

```text
个人学习资料
     |
     v
PDF / Markdown / TXT
     |
     v
文档读取与文本切分
     |
     v
GLM-Embedding-3
     |
     v
FAISS 向量索引
     |
     v
用户输入问题
     |
     v
上下文问题改写
     |
     v
FAISS 相似向量检索
     |
     v
相关文档与元数据
     |
     v
DeepSeek-V4.1-Flash
     |
     v
生成答案与参考资料
```

## 五、项目目录结构

```text
Personal_Knowledge_RAG/
│
├── .venv/                 # Python 虚拟环境
│
├── data/                  # 原始学习资料
│   ├── pdf/
│   ├── markdown/
│   └── txt/
│
├── vectorstore/           # FAISS 知识库
│   ├── index.faiss
│   ├── chunks.json
│   └── manifest.json
│
├── app.py                 # Streamlit Web 界面
├── rag_core.py            # RAG 核心功能
├── build_index.py         # 初始知识库构建
├── update_index.py        # 知识库增量更新
├── search.py              # 命令行向量检索
├── chat.py                # 命令行智能问答
├── test_api.py            # API 连接测试
│
├── requirements.txt       # 项目依赖
├── .env                   # API 配置
├── .gitignore             # Git 忽略配置
└── README.md              # 项目说明
```

## 六、环境配置

开发环境：

- 操作系统：Windows
- 编辑器：Visual Studio Code
- Python：使用与项目依赖兼容的 Python 版本
- 虚拟环境：项目内部 `.venv`

### 1. 克隆项目

```powershell
git clone 你的GitHub仓库地址
cd Personal_Knowledge_RAG
```

### 2. 创建虚拟环境

```powershell
python -m venv .venv
```

激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. 安装项目依赖

```powershell
python -m pip install -r requirements.txt
```

### 4. 配置 API

在项目根目录创建 `.env` 文件：

```env
PARATERA_API_KEY=你的并行云API密钥
PARATERA_BASE_URL=并行云提供的API地址

DEEPSEEK_MODEL=你的DeepSeek模型名称
EMBEDDING_MODEL=你的Embedding模型名称
```

当前项目使用的模型：

- 聊天模型：DeepSeek-V4.1-Flash
- Embedding 模型：GLM-Embedding-3

请根据并行云平台提供的实际 API 配置填写模型名称和接口地址。

不要将真实 API 密钥提交到 GitHub。

## 七、使用方法

### 1. 准备学习资料

将个人学习资料放入对应目录：

```text
data/
├── pdf/
├── markdown/
└── txt/
```

### 2. 首次构建知识库

```powershell
python build_index.py
```

程序读取资料、切分文本、生成向量并构建 FAISS 索引。

### 3. 增量更新知识库

```powershell
python update_index.py
```

程序检测文件变化，并根据新增、修改、删除情况更新知识库。

### 4. 启动 Web 应用

```powershell
streamlit run app.py
```

浏览器访问：

```text
http://localhost:8501
```

进入个人知识库助手，即可上传学习资料、更新知识库和进行智能问答。

### 5. 命令行使用

向量检索：

```powershell
python search.py
```

命令行智能问答：

```powershell
python chat.py
```

## 八、数据与对话管理

- 原始学习资料保存在 `data/`。
- FAISS 索引保存在 `vectorstore/`。
- `chunks.json` 保存文本块及其元数据。
- `manifest.json` 保存文件变化检测所需的信息。
- 当前多轮对话历史保存在 Streamlit 会话状态中，程序重启后会话历史清空。

## 九、项目特点

1. 使用 API 调用模型，无需本地部署大语言模型和 Embedding 模型。
2. 使用 FAISS 实现本地向量存储与检索。
3. 使用 SHA256 实现知识库增量更新。
4. 通过上下文改写改善多轮对话检索。
5. 支持文档来源追踪，方便核对答案依据。
6. 使用 Streamlit 提供简单易用的 Web 操作界面。

## 十、注意事项

- 首次运行 Web 应用前，需要先完成知识库构建。
- `.env` 包含 API 密钥，不应上传到公开仓库。
- `.venv/`、`data/` 和 `vectorstore/` 默认由 `.gitignore` 排除。
- 克隆项目后，需要自行准备学习资料并重新构建知识库。
- 本项目第一版以基础 RAG 为核心，暂未引入复杂的 RAG-Fusion、Reranker 等高级检索技术。

---

**项目名称：** Personal Knowledge RAG

**开发语言：** Python

**核心技术：** LangChain、FAISS、DeepSeek API、GLM-Embedding-3、Streamlit