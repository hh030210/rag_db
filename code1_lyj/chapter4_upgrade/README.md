# 智能文档处理与RAG系统

基于 FastAPI 和 Gradio 的智能文档处理系统，集成第二章（文件转换与处理系统）和第三章（RAG检索增强生成系统）。

## 系统架构

```
chapter4/
├── app/
│   ├── main.py              # FastAPI 主应用
│   └── rag_routes.py        # RAG系统API路由（第三章）
├── frontend/
│   ├── app.py               # Gradio 前端主应用（含章节切换）
│   └── pages/
│       ├── __init__.py
│       └── rag_page.py      # RAG系统前端页面（第三章）
├── core/
│   ├── file_manager.py      # 文件管理器
│   └── file_processor.py    # 文件处理器
├── models/
│   └── schemas.py           # Pydantic 数据模型
├── uploads/                 # 上传文件存储目录
├── outputs/                 # 处理结果输出目录
├── config.py                # 系统配置
├── requirements.txt         # 依赖包
└── README.md               # 项目文档
```

## 功能模块

### 📄 第二章：文件转换与处理系统

支持 PDF、Word、PPT、Excel、TXT 等格式的文件转换、去噪和内容重组。

**功能标签页：**
- 📤 文件上传 - 批量上传多种格式文件
- 📋 文件列表 - 查看所有上传文件
- ⚙️ 文件处理 - 四阶段处理流程（原始提取→规则去噪→LLM去噪→内容重组）
- 👁️ 内容预览 - 预览各阶段处理结果
- ⬇️ 下载文件 - 下载各阶段处理结果
- 🗑️ 删除文件 - 删除文件及处理结果

### 🔍 第三章：RAG检索增强生成系统

基于向量数据库的文档检索与智能问答系统。

**功能标签页：**
- 📚 向量库管理 - 创建、查看、删除向量库
- 📄 文档向量化 - 将处理后的文档添加到向量库
- 🔍 相似度检索 - 基于语义相似度的文档检索
- 💬 智能问答 - 基于检索结果的智能对话

## 技术框架介绍

### 1. 后端框架 - FastAPI

**FastAPI** 是一个现代、高性能的 Python Web 框架，用于构建 API。

**核心特性：**
- **异步支持**：基于 Starlette 和 Pydantic，原生支持异步处理
- **自动文档**：自动生成 OpenAPI 和 Swagger UI 文档
- **数据验证**：基于 Pydantic 的类型检查和数据验证
- **高性能**：与 Node.js 和 Go 相当的性能
- **后台任务**：内置后台任务处理支持

**本系统使用场景：**
- 文件上传和下载 API
- 文件处理状态管理
- 后台异步处理任务
- RESTful API 设计
- RAG向量库管理API

### 2. 前端框架 - Gradio

**Gradio** 是一个 Python 库，用于快速创建和共享机器学习模型的演示界面。

**核心特性：**
- **纯 Python**：无需前端开发经验
- **快速构建**：几行代码即可创建交互式界面
- **组件丰富**：支持文件上传、文本框、按钮等多种组件
- **主题定制**：支持多种主题和样式定制
- **易于部署**：可轻松部署为 Web 应用

**本系统使用场景：**
- 文件上传界面
- 处理进度展示
- 内容预览界面
- 下载链接生成
- RAG向量库管理界面
- 智能问答对话界面

### 3. 文件处理流程

系统采用**四阶段处理流程**：

```
原始文件 → 原始文本提取 → 规则去噪 → LLM去噪 → 内容重组 → 最终输出
```

**阶段说明：**

1. **原始文本提取 (raw)**
   - 支持 PDF、Word、PPT、Excel、TXT 等格式
   - 使用 pdfplumber、python-docx、python-pptx、openpyxl 等库
   - 提取纯文本内容

2. **规则去噪 (rule_denoised)**
   - 去除 VLM 占位符（如 `<|LOC_399|>`）
   - 去除控制字符
   - 合并多余空格和空行
   - 可集成 chapter2 的规则去噪器

3. **LLM去噪 (llm_denoised)**
   - 基于大语言模型的语义去噪
   - 识别并删除与上下文无关的内容
   - 保留专业术语和有意义的信息
   - 可集成 chapter2 的 LLM 去噪器

4. **内容重组 (organized)**
   - 连接断句
   - 段落重组
   - 删除冗余内容
   - 可集成 chapter2 的内容重组功能

### 4. RAG系统流程

```
文档处理 → 文本分块 → 向量化 → 向量库存储 → 语义检索 → LLM生成回答
```

**流程说明：**

1. **文档向量化**
   - 选择已处理完成的文档
   - 配置分块参数（分块大小、重叠大小）
   - 生成文本向量并存储到向量库

2. **相似度检索**
   - 输入查询内容
   - 计算查询向量与文档向量的相似度
   - 返回最相关的文本片段

3. **智能问答**
   - 基于检索到的相关文档片段
   - 调用大语言模型生成回答
   - 提供参考来源

### 5. 数据流架构

```
┌─────────────────────────────────────────────────────────────┐
│                      前端界面 (Gradio)                       │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │  第二章：文件处理    │    │  第三章：RAG系统            │ │
│  │  - 文件上传          │    │  - 向量库管理               │ │
│  │  - 文件处理          │◄──►│  - 文档向量化               │ │
│  │  - 内容预览          │    │  - 相似度检索               │ │
│  │  - 下载删除          │    │  - 智能问答                 │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      后端API (FastAPI)                       │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │  文件管理API         │    │  RAG系统API                 │ │
│  │  /api/files/*        │    │  /api/rag/*                 │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
┌─────────────────────┐      ┌─────────────────────────────┐
│   文件存储系统       │      │     RAG向量库系统           │
│  ┌───────────────┐  │      │  ┌─────────────────────┐    │
│  │  uploads/     │  │      │  │  向量库存储         │    │
│  │  原始文件      │  │      │  │  - 集合管理         │    │
│  └───────────────┘  │      │  │  - 文档索引         │    │
│  ┌───────────────┐  │      │  │  - 向量检索         │    │
│  │  outputs/     │  │      │  └─────────────────────┘    │
│  │  处理结果      │  │      │  ┌─────────────────────┐    │
│  └───────────────┘  │      │  │  LLM服务            │    │
└─────────────────────┘      │  │  - 文本生成         │    │
                             │  │  - 问答生成         │    │
                             │  └─────────────────────┘    │
                             └─────────────────────────────┘
```

### 6. 核心组件

#### 文件管理器 (FileManager)
- 负责文件的存储、检索和状态管理
- 使用 JSON 文件持久化元数据
- 支持文件的上传、下载、删除操作
- 管理各阶段处理结果

#### 文件处理器 (FileProcessor)
- 负责文件转换和处理流程
- 异步处理，不阻塞主线程
- 支持多阶段处理流水线
- 详细的日志记录

#### RAG系统 (RAGPage)
- 向量库管理（创建、删除、查询）
- 文档向量化（分块、嵌入、存储）
- 语义检索（相似度计算、结果排序）
- 智能问答（上下文构建、LLM调用）

#### Pydantic 模型
- 定义数据结构和类型
- 自动数据验证
- API 文档生成

## 安装和运行

### 1. 安装依赖

```bash
cd i:\毕业论文最新版\Code\chapter4
pip install -r requirements.txt
```

### 2. 配置环境变量（可选）

```bash
# 在 config.py 中修改配置，或设置环境变量
set API_KEY=your-api-key
set API_URL=https://api.example.com/v1/chat/completions
set TEXT_MODEL_NAME=gpt-4
```

### 3. 启动后端服务

```bash
python app/main.py
```

后端服务将在 http://0.0.0.0:8000 启动

### 4. 启动前端界面

在另一个终端中：

```bash
python frontend/app.py
```

前端界面将在 http://0.0.0.0:7860 启动

### 5. 访问系统

- API 文档：http://localhost:8000/docs
- 前端界面：http://localhost:7860

## 使用流程

### 第二章：文件处理流程

1. **上传文件**
   - 进入"📄 第二章：文件转换与处理系统"
   - 在"📤 文件上传"标签页上传文件

2. **处理文件**
   - 切换到"⚙️ 文件处理"标签页
   - 选择要处理的文件
   - 选择处理阶段
   - 点击"开始批量处理"

3. **查看结果**
   - 在"👁️ 内容预览"标签页查看各阶段结果
   - 在"⬇️ 下载文件"标签页下载处理结果

### 第三章：RAG系统流程

1. **创建向量库**
   - 切换到"🔍 第三章：RAG检索增强生成系统"
   - 在"📚 向量库管理"标签页创建新向量库

2. **添加文档**
   - 在"📄 文档向量化"标签页
   - 选择目标向量库
   - 选择已处理完成的文档
   - 点击"添加到向量库"

3. **检索问答**
   - 在"🔍 相似度检索"标签页进行文档检索
   - 在"💬 智能问答"标签页进行智能对话

## API 接口列表

### 文件管理接口（第二章）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/files/upload | 上传单个文件 |
| POST | /api/files/upload-batch | 批量上传文件 |
| GET | /api/files | 获取文件列表 |
| GET | /api/files/{file_id} | 获取文件信息 |
| DELETE | /api/files/{file_id} | 删除文件 |
| POST | /api/files/{file_id}/process | 处理文件 |
| GET | /api/files/{file_id}/preview | 预览文件内容 |
| GET | /api/files/{file_id}/download/{stage} | 下载处理后的文件 |

### RAG系统接口（第三章）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/rag/collections | 获取向量库列表 |
| POST | /api/rag/collections | 创建向量库 |
| GET | /api/rag/collections/{id} | 获取向量库信息 |
| DELETE | /api/rag/collections/{id} | 删除向量库 |
| GET | /api/rag/collections/{id}/documents | 获取向量库文档列表 |
| POST | /api/rag/collections/{id}/documents | 添加文档到向量库 |
| DELETE | /api/rag/collections/{id}/documents/{doc_id} | 从向量库删除文档 |
| POST | /api/rag/collections/{id}/search | 相似度检索 |
| POST | /api/rag/chat | RAG智能问答 |
| GET | /api/rag/health | RAG系统健康检查 |

## 与 Chapter2 集成

系统预留了与 chapter2 的集成接口：

### 1. 规则去噪集成
在 `core/file_processor.py` 的 `_rule_based_denoise` 方法中，可以导入 chapter2 的规则去噪器：

```python
from experiments.exp1.rule_based_denoiser import RuleBasedDenoiser

denoiser = RuleBasedDenoiser()
cleaned_text, noise_categories = denoiser.detect_and_remove_noise(text)
```

### 2. LLM去噪集成
在 `_llm_denoise` 方法中，可以导入 chapter2 的 LLM 去噪器：

```python
from utils.llm_denoiser import LLMDenoiser

denoiser = LLMDenoiser()
result = denoiser.denoise_text(text, noise_info)
```

### 3. 内容重组集成
在 `_organize_content` 方法中，可以导入 chapter2 的内容重组功能：

```python
paragraphs, responses = denoiser.organize_content(text)
```

## 后续扩展建议

### 1. RAG功能完善
- 集成真实的向量数据库（Chroma、Milvus、Pinecone）
- 使用真实的Embedding模型（OpenAI、Sentence-BERT）
- 集成真实的LLM服务（OpenAI、Claude、本地模型）
- 添加多轮对话记忆功能
- 支持多模态RAG（图片、表格）

### 2. 用户管理
- 添加用户认证和授权
- 实现多用户隔离
- 添加操作日志

### 3. 性能优化
- 添加 Redis 缓存
- 实现任务队列（如 Celery）
- 支持分布式处理

### 4. 监控和告警
- 添加 Prometheus 监控
- 实现异常告警
- 添加性能指标收集

## 技术栈总结

| 层级 | 技术 | 用途 |
|------|------|------|
| 前端 | Gradio | 用户界面 |
| 后端 | FastAPI | API 服务 |
| 数据处理 | pdfplumber, python-docx 等 | 文件解析 |
| 数据验证 | Pydantic | 数据模型 |
| 日志 | Loguru | 日志记录 |
| 异步 | asyncio | 异步处理 |
| 文件存储 | 本地文件系统 | 文件存储 |
| 向量库 | 内存存储（可扩展） | 向量存储 |

## 许可证

MIT License
