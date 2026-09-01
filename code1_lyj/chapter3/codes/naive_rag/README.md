# 朴素RAG系统 (Naive RAG)

基于ChromaDB和BGE-large-zh-v1.5的朴素RAG实现，用于Natural Questions数据集问答。

## 功能特性

- **多数据集支持**: 每个数据集有独立的向量库
- **GPU加速**: 使用CUDA进行向量化编码
- **文档切片**: 智能文档切分，保持语义完整
- **向量检索**: 基于余弦相似度的文档检索

## 环境要求

- Python 3.10+
- CUDA 12.1+
- NVIDIA GPU

## 安装

在bylw conda环境下安装依赖：

```bash
# 安装GPU版PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装RAG依赖
pip install chromadb sentence-transformers
```

## 项目结构

```
naive_rag/
├── config.py           # 配置文件
├── embedder.py         # BGE编码器（GPU支持）
├── chunker.py          # 文档切片模块
├── vector_store.py     # ChromaDB向量存储
├── naive_rag.py        # RAG核心实现
├── main.py             # 主程序入口
└── README.md           # 本文件
```

## 使用方法

### 1. 构建索引

首次使用需要构建向量索引：

```bash
C:\Users\45508\anaconda3\envs\bylw\python.exe main.py \
    --dataset nq_validation \
    --index "I:\毕业论文最新版\Code\chapter3\datasets\natural_questions\validation-00000-of-00007.json"
```

参数说明：
- `--dataset`: 数据集名称，用于创建独立的向量库
- `--index`: JSON数据文件路径

### 2. 单次查询

```bash
C:\Users\45508\anaconda3\envs\bylw\python.exe main.py \
    --dataset nq_validation \
    --query "what is the longest english word in the dictionary" \
    --top_k 5
```

输出示例：
```
============================================================
朴素RAG系统
============================================================
✓ GPU可用: NVIDIA GeForce RTX 2070 SUPER
✓ 模型加载完成，使用设备: cuda
✓ 向量维度: 1024
✓ RAG系统初始化完成，使用数据集: nq_validation

检索相关问题: who proposed that electrons behave like waves and particles
  编码批次 1/1...
✓ 检索到 5 个相关文档块
  [1] 分数: 0.8234 | Wave–particle duality is the concept in quantum mechanics that every particle...
  [2] 分数: 0.7891 | In 1924, Louis de Broglie proposed that all moving particles—particularly subatomic particles such as electrons—exhibit a degree of wave-like behavior...
  [3] 分数: 0.7654 | The wave theory had prevailed—or at least it seemed to...

============================================================
查询结果
============================================================
问题: who proposed that electrons behave like waves and particles

回答:
根据检索到的最相关文档：

In 1924, Louis de Broglie proposed that all moving particles—particularly subatomic particles such as electrons—exhibit a degree of wave-like behavior...
```

### 3. 交互模式

```bash
C:\Users\45508\anaconda3\envs\bylw\python.exe main.py --dataset nq_validation
```

交互示例：
```
进入交互模式（输入'quit'退出）
------------------------------------------------------------

请输入问题: how many senators are there in the us senate

------------------------------------------------------------
回答:
根据检索到的最相关文档：

The United States Senate consists of 100 members, two from each of the 50 states...
------------------------------------------------------------

请输入问题: quit
再见！
```

### 4. 评估性能

在数据集上评估RAG检索准确率：

```bash
C:\Users\45508\anaconda3\envs\bylw\python.exe main.py \
    --dataset nq_validation \
    --index "I:\毕业论文最新版\Code\chapter3\datasets\natural_questions\validation-00000-of-00007.json" \
    --evaluate \
    --max_eval 100
```

输出示例：
```
============================================================
评估结果
============================================================
总样本数: 100
正确数: 67
准确率: 67.00%
```

### 5. 重新索引

如果需要重新构建索引（会删除现有索引）：

```bash
C:\Users\45508\anaconda3\envs\bylw\python.exe main.py \
    --dataset nq_validation \
    --index "I:\毕业论文最新版\Code\chapter3\datasets\natural_questions\validation-00000-of-00007.json" \
    --reindex
```

## 配置参数

在 `config.py` 中可以修改以下参数：

```python
# 模型路径
EMBEDDING_MODEL_PATH = r"I:\毕业论文最新版\Code\models\embedding\bge-large-zh-v1.5"

# 向量数据库根目录
VECTOR_DB_ROOT = BASE_DIR / "vector_dbs"

# 切片配置
CHUNK_SIZE = 512        # 每个文档块的最大字符数
CHUNK_OVERLAP = 50      # 文档块之间的重叠字符数

# 检索配置
TOP_K = 5               # 默认检索的文档块数量
```

## 支持的命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集名称（用于选择向量库） | `nq_validation` |
| `--index` | JSON文件路径，用于构建索引 | 无 |
| `--query` | 单个查询问题 | 无 |
| `--top_k` | 检索的文档块数量 | `5` |
| `--reindex` | 重新构建索引 | `False` |
| `--evaluate` | 在数据集上评估性能 | `False` |
| `--max_eval` | 评估时最大样本数 | `100` |

## 多数据集使用示例

可以为不同的数据集创建独立的向量库：

```bash
# 为NQ验证集创建向量库
python main.py --dataset nq_validation --index "path/to/nq_validation.json"

# 为NQ测试集创建向量库
python main.py --dataset nq_test --index "path/to/nq_test.json"

# 为自定义数据集创建向量库
python main.py --dataset my_dataset --index "path/to/my_data.json"
```

每个数据集的向量库独立存储在 `vector_dbs/{dataset_name}/` 目录下。

## 注意事项

1. **GPU内存**: 首次加载BGE-large-zh-v1.5模型需要约2GB显存
2. **向量库位置**: 向量库默认存储在项目目录下的 `vector_dbs/` 文件夹中
3. **编码速度**: GPU编码速度约为CPU的10-20倍

## 故障排除

### GPU不可用

检查CUDA和PyTorch安装：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果返回`False`，需要重新安装GPU版PyTorch：
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 内存不足

如果GPU内存不足，可以：
1. 减小 `CHUNK_SIZE` 参数
2. 减小批处理大小（在 `embedder.py` 中修改）
3. 使用CPU运行（修改 `config.py` 中的 `DEVICE = "cpu"`）
