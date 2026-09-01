# code1 章节代码与实验结果

`code1` 是论文/系统主线代码区，不能按普通历史代码整体删除。

## 章节职责

- `chapter2/`：多格式文档解析、OCR/VLM 提取、规则去噪、LLM 去噪和内容重组。
- `chapter3/`：基础 RAG、Prompt 迭代、问题聚类、检索评测和多数据集实验。
- `chapter4/`：FastAPI + Gradio 集成应用，连接文档处理和 RAG 问答。
- `chapter3_backup/`：景区 Prompt 聚类与优化产物的运行依赖，同时也是历史实验留档，暂不删除。
- `chapter4_new/`、`chapter4_upgrade/`、`chapter4_xxx/`：第四章不同迭代版本，需在确认当前使用版本后再做版本归并。

实验结果的分类入口见根目录 [EXPERIMENT_RESULTS_INDEX.md](../EXPERIMENT_RESULTS_INDEX.md)。当前整理策略是不改变原始路径，先建立索引，再对确认无引用的重复结果做归档。
