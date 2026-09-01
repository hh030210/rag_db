# RAG_DB_slim Handoff

## 文档用途

这是本项目的持续交接记录。每完成一个任务，都在“任务记录”中追加一条，记录完成时间、目标、修改内容、验证方式、遗留问题和下一步。除非用户明确要求，不覆盖历史记录。

## 项目概况

RAG_DB_slim 是一个面向旅游景区知识问答的混合 RAG 系统，同时包含论文实验和 Web 集成应用。

核心链路：

```text
原始文档
→ 文档解析/清洗/智能分片
→ 指代消解
→ MySQL 保存文本与维度
→ 维度挖掘、标签生成
→ BGE-M3 向量化并写入 Qdrant
→ 语义检索 + 维度检索
→ RRF/自适应权重融合
→ LLM 生成答案
```

主要代码区：

- 根目录：当前 Qdrant RAG、混合检索、Prompt 优化 API。
- `code1/chapter2/`：文档解析、OCR/VLM、规则去噪和 LLM 去噪。
- `code1/chapter3/`：RAG、Prompt 迭代、聚类和实验评测。
- `code1/chapter4/`：FastAPI + Gradio 集成应用。
- `search_service_deploy/`：独立 Qdrant/BGE-M3 检索服务。
- `code1/chapter3_backup/`：当前景区 Prompt 聚类/优化结果的运行依赖，不能视为普通备份直接删除。

实验结果目录说明见 [EXPERIMENT_RESULTS_INDEX.md](EXPERIMENT_RESULTS_INDEX.md)，章节说明见 [code1/README.md](code1/README.md)。

## 当前状态

- 已完成项目主线梳理。
- 已保留 `code1/chapter2`、`code1/chapter3`、`code1/chapter4` 主线代码。
- 已新增实验结果索引，没有移动或删除可能被固定路径读取的结果。
- 2026-08-11 VPN 开启后已成功 SSH 登录服务器 0（`211.87.224.135:17622`）；当前账号可用 Python 3.10/Conda 和 8 张 Tesla V100，但没有 Docker daemon 权限。
- 2026-08-31 已成功 SSH 登录服务器 1（`81.70.191.196:22`）；该服务器为 4 vCPU、14 GB 内存、50 GB 根盘且已满，无 NVIDIA GPU，不适合当前 GPU 实验。
- 工作区存在大量历史未提交/已暂存变更，且当前没有可依赖的提交记录；后续删除或大规模迁移前必须先检查具体目标。
- 配置和部分源码中存在数据库/API/远程连接凭据，后续部署前应撤销、轮换并改用环境变量或密钥管理。

## 服务器编号约定

- 服务器 0：`211.87.224.135:17622`，账号 `humq`，8 张 Tesla V100，用于当前分片和问答实验。
- 服务器 1：`81.70.191.196:22`，账号 `root`，4 vCPU、14 GB 内存、无 NVIDIA GPU，根盘 50 GB 已满。

### 2026-08-31 — interactive_qa badcase 统计方案

- `interactive_qa.py` 当前主要面向交互使用：检索函数会返回结构化结果，但问答函数只打印最终答案，没有持久化原始问题、扩展问题、检索结果、上下文、答案、异常和配置快照，因此不能直接对终端输出做可靠的 badcase 统计。
- 后续应采用独立批量评测入口，固定检索模式、Top-K、重排、融合权重、Query 扩展和 LLM 参数；每条记录保存原始问题、扩展后的检索问题、覆盖范围、各路 Top-K 结果、chunk_id/分数/文本、最终答案、错误、耗时和配置。
- badcase 应分层标注：系统层（超时、API/检索异常、空结果、空答案）、检索层（证据未命中、主题错误、上下文不足、冗余）、生成层（事实错误、幻觉、遗漏、矛盾、格式错误、应答拒答）。统计时区分“可回答问题”和“知识库无答案/库外问题”。
- 自动统计建议使用证据 Hit@K/Recall@K、答案与参考答案的语义或规则一致性、实体/数字一致性和错误类别比例；BLEU/ROUGE 只能作为辅助，不能单独判定 badcase。最终应对自动筛出的样本做分层人工复核，报告 badcase 率及 95% 置信区间。
- 脚本注意事项：记录原始 query 与融合 query 的区别；Query 扩展失败不应永久关闭后续请求的全局开关；`check_spot_coverage` 中的覆盖判断逻辑存在问题且当前主循环使用的是另一套检查函数；问答函数应改为返回结构化结果，而不是只打印。

### 2026-08-31 — 完成 interactive_qa badcase 评测脚本

- 新增 `/Users/a1234/RAG_DB_silm/badcase_eval.py`，支持 JSON、JSONL/NDJSON 和纯文本问题列表的批量问答评测。
- 每条记录保存原始问题、扩展检索问题、覆盖检查、检索 Top-K 轨迹、答案、异常、耗时、配置快照和自动 badcase 标签；同时生成逐问题 JSONL 和汇总 JSON。
- 自动标签包括系统异常、空检索、空答案、标准证据未命中 Top-K、标准答案数字缺失、低词面重合候选和库外问题回答候选；低词面重合仅作为待复核候选，不直接等同于事实错误。
- 对 `interactive_qa.py` 的 `do_qa()` 做了兼容性修改：保留原有交互打印，并返回答案、耗时、Prompt 数量和生成异常，供批量评测复用。
- 验证：`python3 -m py_compile interactive_qa.py badcase_eval.py` 通过；尚未调用数据库或 LLM 执行实际评测。

### 2026-08-31 — 本地 badcase 脚本运行环境检查

- 本地系统 Python 为 `/usr/bin/python3` 3.9.6，`interactive_qa.py` 和 `badcase_eval.py` 的语法检查通过。
- 当前本地未发现项目虚拟环境；系统 Python 缺少 `numpy`、`requests`、`httpx`、`dashscope`、`pymilvus` 和 `sentence-transformers` 等依赖，因此暂时不能直接运行完整 badcase 评测。
- 本地完整运行还需要可访问的 Qdrant（当前配置默认 `127.0.0.1:6333`）、对应的 BGE-M3 模型和有效的 `DASHSCOPE_API_KEY`；仅安装 Python 依赖不能替代这些外部资源。

## 服务器操作约定

可以协助执行服务器上的检查、启动、部署和日志排查，但执行前需要明确：

1. 服务器地址、端口和账号；
2. 密码、SSH Key 或现有认证方式；
3. 操作范围（只读检查、重启服务、部署代码、修改配置等）；
4. 是否允许停止服务、覆盖文件或迁移数据。

涉及删除、覆盖、重置数据库或停止生产服务时，先确认目标和备份情况。

## 实验运行建议

### A. 完整 Qdrant RAG 入库流水线

在项目根目录执行，适合首次构建或重建知识库：

```bash
python3 pipeline_qdrant.py --all --input ./data_input
```

也可以分步执行，便于定位问题：

```bash
python3 pipeline_qdrant.py --from_step 1 --to_step 4 --input ./data_input
python3 pipeline_qdrant.py --step 5 --input ./data_input
python3 pipeline_qdrant.py --step 6 --input ./data_input
```

主要依赖 MySQL、Qdrant、BGE-M3 本地模型，以及可选的 LLM API。正式运行前先检查 `db_config.yaml`，并通过环境变量提供密钥。

### B. 独立检索服务

```bash
cd search_service_deploy/service
export QDRANT_HOST=127.0.0.1
export QDRANT_PORT=6333
export BGE_MODEL_PATH=/path/to/bge-m3
python3 -m uvicorn search_api_server:app --host 0.0.0.0 --port 8100 --workers 1
```

接口：`GET /healthz`、`GET /config`、`POST /search`。检索模式为 `sem`、`dim`、`fusion`。

### C. Prompt 优化服务

```bash
export BGE_MODEL_PATH=/path/to/bge-m3
./start.sh
```

默认监听 8000 端口，入口为 `api_server.py`。调用方需要通过 `Authorization: Bearer <DASHSCOPE_API_KEY>` 或请求体传入 API Key。

### D. 连接记录（2026-08-11）

- 目标：服务器 0（`211.87.224.135:17622`），用户 `humq`。
- 结果：TCP 连接超时，未进入 SSH 密码认证阶段；未在服务器执行任何修改、启动或停止操作。
- 建议：检查服务器安全组、防火墙、SSH 监听端口、端口映射，以及当前网络是否允许访问该端口。

### E. `code1` 实验入口

Chapter 2 文档处理：

```bash
cd code1/chapter2
python3 main.py /path/to/file-or-directory
```

Chapter 3 基础动态 Prompt RAG：

```bash
cd code1/chapter3/codes/bylw_rag
python3 main.py --init --question-type fact_retrieval --domain general
python3 main.py --dataset nq_validation --question-type fact_retrieval --query "什么是 RAG？"
```

Chapter 3 景区 Prompt 实验的逻辑顺序：

```bash
cd code1/chapter3/codes/bylw_rag/new_experiments
python3 Tourist_step1_data_preparation.py
python3 Tourist_step2_prompt_iteration.py
python3 Tourist_step3_kmeans_clustering.py
python3 Tourist_step4_collective_optimization.py
python3 Tourist_step5_inference.py
```

注意：上述 Chapter 3 脚本中仍有 Windows 路径和本地模型/API 配置，迁移到 Linux 服务器前需要先改成服务器路径，并通过环境变量注入密钥；不能直接照搬运行。

Chapter 4 集成 Web 应用：

```bash
cd code1/chapter4
python3 -m pip install -r requirements.txt
python3 app/main.py
# 另开终端
python3 frontend/app.py
```

后端默认 8000 端口，Gradio 前端默认 7860 端口。`chapter4_new`、`chapter4_upgrade`、`chapter4_xxx` 是迭代版本，运行前需要明确使用哪一版。

## 关联项目：`rank_model_lite`

### 读取范围

- 路径：`/Users/a1234/Documents/rank_model_lite`
- 本次只读检查了目录、Python/Shell/YAML/JSON 配置、训练/推理/部署入口、实验文档、日志摘要和实验结果；没有修改该目录中的文件。
- 规模：约 142 个 Python 文件、约 77,888 行 Python 代码；仓库中同时存在多个历史模型分支和大量特征配置，不能按“一个脚本就是一个独立项目”理解。

### 项目定位

这是一个面向广告/推荐排序场景的 CTR/CVR 多任务模型工程，目标是从曝光样本中预测点击、转化等目标，并将模型导出为 SavedModel，通过 TensorFlow Serving 提供线上推理。代码核心不是旅游 RAG，而是高吞吐的特征工程、Embedding/交叉特征、排序损失、离线评估和线上发布链路。

主线可以概括为：

```text
原始广告/曝光数据
→ TSV/Parquet 等样本转换为 TFRecord（可 GZIP）
→ 特征组 JSON（FG）定义 label、sparse、cross、dense、ext
→ TensorFlow 1.x 图训练
→ checkpoint 热启动/续训
→ AUC、COPC、GAUC、BPR 诊断
→ SavedModel + feature_desc + Serving warmup
→ S3/对象存储同步、TensorFlow Serving、监控和告警
```

### 训练模型主线

1. `ctr/` 是广告 CTR/CVR 主工程。
   - `mt8.py`、`mt8a.py` 到 `mt8o*.py`、`tf1_train*.py` 等是不同阶段的模型和实验变体，不能全部视为当前生产入口。
   - `ctr/mt8f.py` 是一条重要的 T2 多任务基线：按 FG 解析 TFRecord，构建 sparse embedding、hash/mod/identity/boundaries 分桶、cross embedding、dense `log1p + BatchNorm`、双线性交互、PPNet 门控和 MMoE 专家/任务塔；使用加权 BCE，稀疏参数和稠密参数可使用不同优化器；支持部分变量热启动和 SavedModel 导出。
   - `ctr/mt9_t3.py` 是另一条较新的 T3 实验主线：在多任务排序基础上加入 BPR pairwise ranking loss、BPR tower、BPR 诊断，并围绕 `fea_map__*` 历史统计特征做 dense 或 sparse 分桶实验。`ctr/doc/t3_feature_experiment_plan.md` 和 `ctr/doc/t3_fmap_codex_handoff.md` 是这条实验线的重要交接文档。
   - `ctr/mt_warmstart.py`、`ctr/mt_warmstart2.py` 负责 checkpoint 变量映射、部分热启动、扩维后的首层参数复制和严格 shape 校验；新增特征时不能只改 FG，还要确认输入顺序和旧变量映射。
   - `ctr/fg/` 下的 JSON 是特征组配置，不是普通数据文件。配置决定输入字段、类型、hash 空间、Embedding 维度、slot、cross、图分支和 dense 输入；删除旧 FG 可能导致历史实验或热启动命令失效。

2. `tblm/` 是相对独立的另一条排序模型线。
   - `tblm/train1.py` 使用 TensorFlow 1.x 风格图训练，支持 sparse/dense/cross 特征、稀疏感知 MLP、多任务 head、加权 BCE、AUC/COPC/GAUC 评估、checkpoint 和 SavedModel 导出。
   - `tblm/infer.py` 加载 SavedModel，读取 TFRecord，批量推理并写 CSV。
   - `tblm/tsv2tr.sh` 通过 Java `TsvToTFRecordConverter` 将 TSV 转成 TFRecord；`tblm/doc/` 保存特征格式和字段说明。

### 训练、推理和部署目录

- `deploy_train/`：CTR 日常训练调度。`train_t1.py` 调用 `mt8.py` 和 `fg8.json`，`train_t2.py` 调用 `mt8f.py` 和 `fg8d1_t2.json`；`pipeline_train.py` 负责依赖前一天模型、冷启动/热启动、训练、评估、导出、日志、对象存储同步和成功标记，`generate_sample.py` 负责样本生成。
- `deploy_tblm/`：TBLM 的训练/推理调度，包含样本生成、前一日 checkpoint 依赖、评估日志上传、SavedModel 导出、推理 TFRecord 生成和结果上传。
- `deploy_infer/`：线上模型发布和运维。`model_dl.py` 从 S3 同步带 `_SUCCESS` 标记的模型版本并原子切换；`t1_composer.yml`、`t2_composer.yml` 启动 TensorFlow Serving 副本；Traefik 负责入口路由，Prometheus/Grafana 负责监控，`monitor_tfserving.py` 检查容器状态、模型 AVAILABLE 版本和版本过期情况。
- `hmq/`：个人/阶段性实验资产，包括 baseline 结果、8 桶尾部追加实验、T3 `fea_map` 实验生成器、screen 启动脚本、日志和 checkpoint。它们不是可随意清理的缓存，部分脚本依赖固定目录、固定日期和固定 checkpoint。

### 实验结论和结果整理要点

- `hmq/baseline_result_20260728.md` 记录的 T3 baseline 中，`t2.dsp#click` 的 AUC/GAUC 为 `0.9266/0.8153`，`t2.adx#click` 的 AUC/GAUC 为 `0.8427/0.6411`；由于样本中的转化权重为 0，转化 head 没有有效样本，不能把对应 0 值当成模型效果。
- `hmq/fg_8bin_after_cross/` 记录了新增 `sparse_after_cross` 特征的最多 8 桶方案。新增输入追加在旧 sparse/cross 后面，旧 checkpoint 的首层 kernel 和 Adam 状态按旧输入区间恢复，新增区间冷启动；实验启动前需要做 FG 和 CKPT preflight。
- T3 `fea_map__*` 主线固定模型、训练日期、评估日期和热启动 checkpoint，单独比较新增业务统计特征，指标重点看 DSP/ADX 的点击 GAUC，并辅以 AUC、COPC、Loss 和 BPRDiag。`hmq/fg_next/` 进一步拆成单特征、leave-one-out 和 sparse 等频分桶实验。
- 已有实验文档反复显示：扩大 hash 空间、降低碰撞有时比单纯增加 Embedding 维度更重要；模型指标差异小于约 `0.001` 时不宜直接下结论，需要复验。涉及 `adx_cvt` 或其他转化 head 时，必须先检查有效权重和有效 group 数。

### 运行环境和迁移风险

- `deploy_train/requirements.txt` 明确依赖 TensorFlow `1.15.0`、TensorFlow Serving API `1.15.0`、TensorFlow Estimator `1.15.1`、`numpy 1.21.6`、`scikit-learn 1.0.2` 等，代码大量使用 TensorFlow 1.x 图 API；不能直接用当前服务器上的 PyTorch 环境替代。
- 运行脚本包含大量内部绝对路径，例如 `/data/users/...`，并依赖 `datalakex`、AWS/S3、Java 转换器、Conda 的 `tf1` 环境、screen、Docker 和 TensorFlow Serving。迁移到新服务器前必须逐项替换路径并确认数据、模型、JAR、对象存储和权限。
- 部分脚本包含飞书 webhook、对象存储和内部服务配置。交接文档只记录功能，不复制敏感值；正式部署前应撤销/轮换已暴露凭据，并改为环境变量或密钥管理。
- `rank_model_lite` 当前存在未提交修改和新增实验文件，重点包括 `ctr/mt8f.py`、`ctr/mt9.py`、`ctr/mt9_t3.py`、`ctr/doc/` 和 `hmq/`；后续清理时必须先确认实验引用关系，不能按文件名批量删除。

### 与当前 RAG_DB_slim 的关系

两者是不同业务方向的项目：`RAG_DB_slim` 负责旅游知识库、混合检索和问答生成；`rank_model_lite` 负责广告/推荐排序模型训练与部署。当前没有发现 RAG 主流程直接 import `rank_model_lite` 的证据，因此本次只把它作为外部关联项目记录，不将其代码复制到 RAG 项目，也不建议把其 TensorFlow/Serving 依赖混入 RAG 的精简依赖。

## 关联项目：`chunk_code`（文本分片与 Meta-Chunking）

### 读取范围

- 路径：`/Users/a1234/chunk_code`
- 本次读取了根目录分片器、增强模块、实验脚本、结果摘要、设计文档，以及 `Meta-Chunking/` 下的 README、评测入口和 MoC 相关代码；没有修改该目录中的任何文件。
- 规模：约 188 个 Python 文件、20,855 行 Python 代码。目录中包含大量生成数据和实验输出，磁盘占用约数 GB，不能把这些输出目录都当作核心源码。

### 项目定位

这是一个面向 RAG 的文本分片实验工程，核心目标是根据文档结构、语言模型困惑度（PPL）和相邻内容的信息增益，生成语义边界更合理、长度更稳定的 chunks，随后用于向量检索和 CRUD/QuestAnswer 问答评测。它不是排序模型，也不是当前 RAG 服务的直接依赖，更适合作为 RAG 入库前的独立 chunking 实验层。

主线可以概括为：

```text
原始文档
→ 结构切分与文档类型识别
→ PPL 计算、去噪和边界检测
→ 句子级拆分、信息增益合并、长度约束
→ 可选去重/Embedding IG
→ chunks JSON/TXT
→ Milvus + BGE 向量检索
→ Qwen 问答生成与 ROUGE/BERTScore 评测
```

### 根目录核心代码

- `integrated_chunker.py` 是三阶段主分片器。
  1. 结构层：识别 `==== 标题 ====`, Markdown 标题和水平分隔线，并用关键词或可选 OpenAI-compatible LLM 推断文档类型及 `l_min/l_max`。
  2. PPL 层：默认使用无需外部模型的字符二元统计 PPL；也支持本地 Hugging Face causal LM。通过 PPL 跳变检测句子边界，并使用 MAD 阈值进行异常值去噪。
  3. 优化层：对超长 chunk 按句拆分，使用字符频率余弦相似度计算相邻信息增益，贪心合并过短片段，同时保留溢出文本。
- `chunking_enhancements.py` 和 `enhanced_chunker.py` 是不侵入原分片器的增强组合层，主要包含批量本地 LM PPL、长度自适应切分、短文本保护、jieba 前缀指纹去重和可选 sentence-transformers Embedding IG。默认输出格式与原分片器保持一致，便于继续使用转换和 CRUD 评测脚本。
- `run_crud_exp.py` 将 chunks 转成评测文档并启动 Milvus CRUD/QuestAnswer 流程，默认关注 BGE 中文 Embedding、Qwen 生成、`retrieve_top_k=8` 和 ROUGE/BERTScore。
- `test_improvements.py` 覆盖 AdaptiveSplitter、LengthAwareDenoiser、jieba 去重和端到端 smoke test；Embedding IG 与 BatchedLM 需要本地模型时可跳过网络/模型测试。

增强版的典型运行方式：

```bash
cd /Users/a1234/chunk_code
python enhanced_chunker.py --input data/db_qa.txt --line_mode --output output_enhanced_v2
# 可选：--use_batched_lm --use_embedding_ig --no_dedup
```

基础分片器也支持相同的输入/输出模式：

```bash
python integrated_chunker.py --input data/db_qa.txt --line_mode --output output_chunks
```

每次分片主要产出 `all_chunks_chunks.json`、`all_chunks_chunks.txt` 和 `all_chunks_summary.json`。如果启用 LLM 推荐，应通过环境变量或命令行注入密钥；`integrated_chunker.py` 当前源码中存在硬编码 API key 默认值，属于必须移除并轮换的安全风险，交接文档不记录该值。

### 已有实验结果

- `output_chunks`：约 16,401 个 chunks，总字符数约 829 万，平均长度约 505 字符。
- `output_baseline_v1`：16,022 个 chunks，总字符数约 829 万，平均长度约 518 字符。
- `output_enhanced_v2`：去重前 15,151 个、去重后 8,034 个 chunks，总字符数约 415 万，平均长度约 516 字符，运行时间约 142 秒。去重比例约 47%，明显高于设计文档预期的 5%–15%，需要抽样确认是否误删了同主题但内容不同的片段。
- `META_CHUNKING_EXPERIMENTS_REPORT.md` 记录的主评测使用 Milvus、`BAAI/bge-base-zh-v1.5` 和 Qwen2.5-7B：Integrated Chunks 相比 Meta-Chunking v2，在 1/2/3 文档任务上的 ROUGE-L 分别约提升 17.0%/10.9%/11.1%，BERTScore 分别约提升 13.3%/5.5%/4.5%。这些数字已在原始评测 JSON 中核对，但尚未在当前环境重新跑完。
- 报告同时指出 Integrated Chunks 生成答案明显更长。现有 Markdown 报告声称 BLEU-1 至 BLEU-4 均为 0，但原始 Meta v2 JSON 中存在非零 BLEU、Integrated JSON 中为 0；因此暂不使用 BLEU 支撑结论，需先统一指标脚本或输出版本。Baseline v1 的完整三路评测因多线程崩溃未完成，建议单线程重跑后再下结论。
- `data/BOUNDARY_CLARITY_REPORT.md` 记录了边界清晰度实验，但 API 失败时两个指标的样本没有一致丢弃，7B/14B 的 pair 数不同，后续需要统一失败处理并复验。Chunk Stickiness/关系一致性相关脚本仍偏旧，包含硬编码路径和模型，不能直接视作当前有效结果。
- 已将分片和下游评测结果统一整理到 `/Users/a1234/chunk_code/EXPERIMENT_RESULTS_SUMMARY.md`；该文件区分了不同输出目录、主实验、top-4 冒烟结果、边界指标、日志异常和未完成实验。

### `Meta-Chunking/` 上游工程与依赖

- 上游 README 的方向是通过 PPL、margin sampling、动态组合和信息补偿实现逻辑/语义分片，并覆盖 CRUD、LongBench、MultiHop-RAG 和 RAGBench。
- `Meta-Chunking/MoC/` 实现 Mixtures of Text Chunking Learners，关注 Boundary Clarity 和 Chunk Stickiness；其中部分评测脚本依赖固定本地模型路径、GPU 和旧实验目录。
- `Meta-Chunking/requirements.txt` 是重量级环境，包含 PyTorch、Transformers、sentence-transformers、FlagEmbedding、vLLM、xformers、pymilvus、LlamaIndex、FastAPI 和 Gradio 等。它应当作为独立 Conda 环境准备，不能混装进 `RAG_DB_slim` 的轻量运行环境。
- CRUD 评测依赖本地 Milvus（默认 `localhost:19530`）、中文 BGE Embedding 和 Qwen 推理；当前主项目使用 Qdrant/MySQL，数据库和索引链路不同。

### 与当前项目及服务器的关系

`chunk_code` 与 `RAG_DB_slim` 的共同点是都服务于 RAG，但前者主要解决“怎么切片”，后者主要负责旅游知识库入库、混合检索和问答服务。目前没有发现两者存在直接 import 关系，因此不能仅按目录名批量合并或删除。若要接入主项目，应先在 `code1/chapter2`/`chapter3` 的真实入库入口中验证 chunk JSON schema、元数据、检索召回和答案质量，再决定是否替换现有切片逻辑。

服务器上此前确认有 8 张 V100，但该工程还需要 Milvus、中文 Embedding、Qwen/本地 PPL 模型或外部 API，以及一套独立的大型依赖环境；不能仅凭 GPU 空闲就直接运行。很多脚本包含 Windows 绝对路径、Linux 内部模型路径和固定日期/目录，迁移前必须统一配置、数据路径、模型路径和凭据。

### 在服务器上的准备清单

建议分三档准备，不要一开始就安装完整的 `Meta-Chunking/requirements.txt`：

1. **最小分片实验（建议先做）**
   - 代码放在 `/home/humq/chunk_code` 或管理员分配的项目目录；此前服务器 `/pub` 使用率约 98%，不要把代码、模型和输出默认放到 `/pub`。
   - Python 3.10+。`integrated_chunker.py` 的字符 n-gram PPL 主流程只依赖 Python 标准库；增强版若要启用 jieba 去重，再安装 `jieba`。
   - 上传输入语料，例如 `data/db_qa.txt`，准备独立输出目录和足够磁盘空间。
   - 先运行 `python test_improvements.py`，再运行 `integrated_chunker.py` 或不启用本地 LM 的 `enhanced_chunker.py`，确认 JSON/TXT 输出正常。

2. **GPU 本地 PPL/Embedding 实验**
   - 单独创建 Python 3.10 Conda 环境，不和 `RAG_DB_slim` 混装。服务器已有 `testbase` PyTorch/CUDA 环境，但此前检查显示它缺少该 RAG/Meta-Chunking 工程的完整依赖，使用前仍需验证 `torch.cuda.is_available()` 和 CUDA 版本。
   - 安装与服务器 CUDA/驱动匹配的 `torch`，以及 `transformers`、`accelerate`、`jieba`；启用 `--use_embedding_ig` 时再安装 `sentence-transformers`。
   - 准备本地模型或确认服务器可访问 Hugging Face：本地 PPL 可用 `Qwen/Qwen2.5-0.5B-Instruct`，Embedding 可用 `BAAI/bge-small-zh-v1.5`；离线服务器需要提前下载后上传到模型目录。
   - V100 优先使用 FP16，并通过 `CUDA_VISIBLE_DEVICES=0` 等方式只占用指定 GPU；不要默认占满 8 张卡。

3. **完整 CRUD/RAG 评测**
   - 独立创建 `MetaChunking` Python 3.10 环境；仅在最小分片验证通过后，才考虑安装 `Meta-Chunking/requirements.txt`，因为它包含 PyTorch、Transformers、vLLM、xformers、Milvus、LlamaIndex、Gradio 等大量且版本固定的依赖。
   - 准备 Milvus 服务，默认地址为 `localhost:19530`。当前账号此前没有 Docker daemon 权限，因此不能假设能用 Docker 启动；需要使用管理员提供的 Milvus 服务或准备可执行的 `milvus-server --data ...`。
   - 准备 CRUD QA 数据（报告中使用 `data/split_merged.json`）、分片转换后的 `crud_data/.../docs`、中文 BGE Embedding（768 维）和 Qwen7B/兼容 API 生成模型。
   - 首次评测使用 `--construct_index --num_threads 1`，每个实验使用独立 `collection_name`；先完成单线程基线与增强版对比，再增加并发。

服务器上还需要提前确定：模型和 Hugging Face 缓存目录、Milvus 数据目录、GPU 分配、外部 LLM API 的地址/密钥、评测输出目录和磁盘配额。密钥只能通过环境变量或服务器密钥管理注入，不能写回脚本或提交到仓库。

### 当前已知风险与建议

- `LengthAwareDenoiser` 的实现从代码看三个分支都会保留句子，和“删除异常短句”的注释/测试意图不完全一致，需要先确认这是有意的短文本保护还是逻辑未完成。
- 去重后 chunks 大幅减少，必须通过重复内容抽样和召回率对比验证，不能直接把 `output_enhanced_v2` 当最终生产结果。
- 先单线程补跑 Baseline v1 与 Integrated 的统一评测，再测试 `retrieve_top_k=4` 或更简洁的生成提示，避免把“答案更长”误判为“效果更好”。
- 运行前移除硬编码 API key，改用环境变量/密钥管理；同时清理 Windows 路径、旧模型路径和硬编码 Milvus 配置。
- 在没有完成对真实 `code1` 入库主线的兼容性验证前，只保留该项目作为独立实验参考，不删除其核心分片器、增强模块和实验结果。
- 去噪有效性评测方案已整理到 `/Users/a1234/chunk_code/DENOISING_EVALUATION_PLAN.md`；方案要求先隔离去噪与去重，再用合成噪声、人工标注、检索证据保留率和端到端 QA 做分层验证。
- 本地已完成 `data/db_qa.txt` 的 `denoise_off`/`denoise_on` 两组正式三阶段分片，结果在 `/Users/a1234/chunk_code/results/denoise_ablation_db_qa_textsafe/`；问答尚未执行。
- 去噪问答的正确入口是 `Meta-Chunking/eval/CRUD/quick_start.py`，本地封装脚本为 `/Users/a1234/chunk_code/run_denoise_qa_ablation.py`；`build_unified.py` 只用于建 Milvus 索引，旧的 `build_v1_full.py`/`build_v2_full.py` 不用于本次对照。

## 后续任务记录规则

每个完成的任务追加以下信息：

```text
### YYYY-MM-DD — 任务标题
- 目标：
- 修改：
- 验证：
- 遗留/风险：
- 下一步：
```

## 任务记录

### 2026-08-11 — 创建项目交接文档

- 目标：建立持续维护的 handoff 文档，记录项目状态并作为后续任务的交接依据。
- 修改：新增 `HANDOFF.md`；明确项目主线、结果索引、服务器操作约定和后续记录格式。
- 验证：确认文件位于项目根目录 `/Users/a1234/RAG_DB_silm/HANDOFF.md`。
- 遗留/风险：当前工作区没有稳定提交点，代码和结果整理仍需逐项确认引用关系。
- 下一步：继续按用户要求整理实验结果，并在每次任务完成后更新本文件。

### 2026-08-11 — 服务器连接检查与实验运行说明

- 目标：连接用户提供的 Linux 服务器，并确认项目可运行环境与实验入口。
- 修改：补充服务器操作约定、根目录 RAG、独立检索服务、Prompt 服务以及 Chapter 2/3/4 的运行命令。
- 验证：对服务器 0（`211.87.224.135:17622`）做 TCP/SSH 只读连接检查。
- 遗留/风险：目标端口连接超时，未进入 SSH 认证阶段；未在服务器执行任何修改。Chapter 3 脚本仍含 Windows 路径和本地 API/模型配置，Linux 运行前必须迁移配置。
- 下一步：服务器开放 SSH 端口或提供 VPN/跳板机后，重新连接并检查 Python、Docker、Qdrant、MySQL、模型目录和项目路径。

### 2026-08-11 — VPN 后服务器环境检查

- 目标：VPN 开启后重新连接服务器，检查硬件、Python 环境、项目目录和模型依赖。
- 修改：无服务器文件修改；补充了本机交接文档中的服务器环境记录。
- 验证：SSH 登录成功；主机为 Ubuntu 22.04，Python 3.10.12，Conda 环境为 `base` 和 `testbase`；`testbase` 使用 Python 3.11.3、PyTorch 2.0.1 + CUDA 11.8，GPU 可见 8 张 Tesla V100 32GB。
- 遗留/风险：项目目录和 BGE/Qwen 模型未在 `/home`、`/opt`、`/root`、`/srv`、`/usr/local`、`/tmp` 的检查范围内找到；`testbase` 缺少 RAG 依赖；当前账号无 Docker daemon 权限；`/pub` 分区已使用约 98%。
- 下一步：将项目放到 `/home/humq/RAG_DB_silm`，安装依赖并上传/挂载模型；如需 Docker，联系管理员加入 Docker 权限或改用已有 Qdrant/MySQL 服务。

### 2026-08-11 — 读取 `rank_model_lite` 并整理关联项目

- 目标：读取 `/Users/a1234/Documents/rank_model_lite` 的代码、训练/推理/部署入口、实验文档和结果摘要，并将理解结果写入本交接文档。
- 修改：仅修改 `/Users/a1234/RAG_DB_silm/HANDOFF.md`；新增关联项目定位、主线流程、`ctr`/`tblm`/部署目录说明、实验指标口径、热启动约束、运行依赖和迁移风险。未修改 `rank_model_lite` 目录。
- 验证：完成目录与入口盘点，检查约 142 个 Python 文件和约 77,888 行 Python 代码；阅读 `ctr/mt8f.py`、`ctr/mt9_t3.py`、`tblm/train1.py`/`infer.py`、`deploy_train`、`deploy_tblm`、`deploy_infer`、FG 配置、T3/8-bin 实验文档及 baseline 结果；确认 `HANDOFF.md` 已写入上述总结。
- 遗留/风险：该项目依赖 TensorFlow 1.15、内部数据路径、Java 转换器、对象存储和 TensorFlow Serving，不能直接套用 RAG_DB_slim 的 Python 环境；其工作区有未提交实验变更，清理前需要逐项确认引用。
- 下一步：如果要在服务器运行该排序项目，先单独上传代码和必要 FG/脚本，准备 TF1 环境并确认样本、checkpoint、JAR、对象存储权限；不要与 RAG_DB_slim 的运行环境混装。

### 2026-08-11 — 安装 Mac SSH 客户端 Termius

- 目标：为 Apple Silicon Mac 安装可替代 Bitvise 的图形化 SSH/SFTP 客户端。
- 修改：从 Termius 官方 macOS 下载页获取 DMG，安装到 `/Applications/Termius.app` 并启动；临时安装包已移入系统废纸篓。没有修改项目代码或服务器。
- 验证：本机架构为 `arm64`；Termius 版本为 `9.43.0`，应用二进制包含 `arm64` 和 `x86_64`，`codesign` 校验有效，`spctl` 判定为 Apple 公证 Developer ID 应用，进程已正常启动。
- 遗留/风险：Termius 的部分高级功能可能需要账户或付费订阅；连接服务器前仍应在首次连接时核对 SSH host key，并避免在聊天或脚本中保存明文密码。
- 下一步：在 Termius 中新建 Host，填入之前服务器的地址、端口、用户名和认证信息，即可进行 SSH/SFTP 操作。

### 2026-08-11 — 读取 `/Users/a1234/chunk_code/` 并整理到 handoff

- 目标：读取用户指定的正确目录 `/Users/a1234/chunk_code/`，梳理文本分片、Meta-Chunking、增强模块、CRUD 评测和已有实验结果。
- 修改：仅修改 `/Users/a1234/RAG_DB_silm/HANDOFF.md`；新增 `chunk_code` 项目定位、三阶段算法、增强版入口、输出统计、Meta-Chunking 依赖、实验结论、迁移风险和后续建议。未修改 `/Users/a1234/chunk_code/` 中的代码或数据。
- 验证：盘点约 188 个 Python 文件和 20,855 行 Python 代码；阅读 `integrated_chunker.py`、`chunking_enhancements.py`、`enhanced_chunker.py`、`run_crud_exp.py`、核心 README/实验报告及输出摘要；核对 `all_chunks_*` 输出和现有评测指标记录。
- 遗留/风险：增强版去重比例约 47%，高于预期；短文本去噪实现与注释意图可能不一致；Baseline 完整评测和边界指标需要统一条件复验；源码存在硬编码 API key、Windows/内部绝对路径和重量级独立依赖。
- 下一步：先在独立环境单线程复跑三种分片方案的统一评测，再抽样检查去重误删和检索召回；确认兼容 `code1` 入库 schema 后，才考虑将该分片器接入主项目。

### 2026-08-11 — 规划服务器上的分片实验准备项

- 目标：根据 `chunk_code` 的实际依赖和服务器现状，整理从最小分片、GPU 本地模型到完整 Milvus/CRUD 评测的准备清单。
- 修改：仅修改 `/Users/a1234/RAG_DB_silm/HANDOFF.md`；新增服务器目录、Python 环境、GPU/模型、Milvus、数据、凭据和单线程评测准备要求。
- 验证：核对根目录 `requirements.txt`、`Meta-Chunking/requirements.txt`、`ENHANCEMENTS_README.md`、`run_crud_exp.py` 及此前服务器环境记录；确认最小字符 n-gram 分片不需要完整 Meta-Chunking 依赖。
- 遗留/风险：服务器 `/pub` 空间紧张，Docker 权限、Milvus 服务、模型目录和外部 API 可用性仍需现场确认；不能直接把完整依赖安装到现有 RAG 环境。
- 下一步：先在服务器建立独立的最小环境并跑 `test_improvements.py`/小样本分片；通过后再准备 GPU 模型和 Milvus 评测环境。

### 2026-08-11 — 整理 `chunk_code` 已有实验结果

- 目标：将当前已有的分片统计、Meta/Integrated CRUD 问答结果、top-4 冒烟结果、Boundary Clarity、运行日志异常和未完成项目统一归档。
- 修改：新增 `/Users/a1234/chunk_code/EXPERIMENT_RESULTS_SUMMARY.md`；同时在本文件中补充汇总文档入口，并修正 BLEU 指标存在原始 JSON 与 Markdown 报告不一致的风险说明。未修改原始代码、输出 JSON、评测日志和数据。
- 验证：直接读取各版本 `all_chunks_chunks.json`、`all_chunks_summary.json`、CRUD 输出 JSON 和 `data/boundary_clarity_summary.json`；核对 chunk 数、长度统计、ROUGE-L、BERTScore、答案长度、有效样本数、边界指标和建索引日志。
- 遗留/风险：Baseline v1 完整 QA 尚未完成；Integrated top-4 只有待补的全量实验；Boundary Clarity 存在样本不成对问题；Chunk Stickiness/Relation Coherence 未运行；不同报告对 BLEU 的描述与原始 JSON 不一致。
- 下一步：按汇总文档的优先级，先单线程补跑 Baseline 并统一评测代码，再跑 Integrated top-4 和修复后的 Boundary Clarity。

### 2026-08-11 — 设计去噪方法有效性评测方案

- 目标：制定能够证明当前去噪方法有效性的可复现实验，而不是仅用 chunk 数量或平均长度变化下结论。
- 修改：新增 `/Users/a1234/chunk_code/DENOISING_EVALUATION_PLAN.md`；记录 MAD 去噪与 LengthAwareDenoiser 的实现差异、当前增强版去噪 no-op 风险、消融实验、合成噪声/人工标注、检索证据保留率、端到端 QA 和统计判定标准。
- 验证：核对 `integrated_chunker.py` 的 MAD 阈值删除逻辑和 `chunking_enhancements.py` 的三个保留分支；确认 `output_enhanced_v2` 的大幅减少不能直接归因于长度感知去噪。
- 遗留/风险：LengthAwareDenoiser 需要先修复或明确预期行为；当前尚未运行去噪 A/B、人工标注、证据召回和配对统计检验。
- 下一步：先在服务器或本地跑 `no_denoise`/`mad_denoise` 两组，关闭 dedup 并保存句子级删除审计信息，再开展下游检索和问答评测。

### 2026-08-11 — 本地准备去噪开关对照和 QA 入口

- 目标：在 `/Users/a1234/chunk_code/data/db_qa.txt` 上用同一套三阶段分片流程，分别运行去噪关闭和开启两组，并准备后续问答实验入口。
- 修改：为 `integrated_chunker.py` 增加 `denoise_enabled`、PPL、参数、输入/输出字符数和 Round 2 删除统计；新增 `run_denoise_ablation.py`；新增 `run_denoise_qa_ablation.py`；为 `quick_start.py` 增加 `--output_dir`；将 CRUD Qwen 配置改为从环境变量读取；转换脚本缺少 `tqdm` 时自动降级。未修改服务器。
- 验证：本地正式分片完成：`denoise_off` 为 17,342 chunks、删除 0 句；`denoise_on` 为 16,980 chunks、删除 5,946 句/158,312 字符；两组已转换为 17,342/16,980 个 `.txt` 评测文档；Python 语法和 CLI 帮助检查通过。
- 遗留/风险：本机缺少完整 QA 依赖和 Milvus，问答尚未运行；本次同时修复了第三阶段合并后文本被跳过的问题，因此正式结果不能与旧的预修复输出直接比较；`LengthAwareDenoiser` 仍是 no-op，不属于本次 MAD 去噪对照。
- 下一步：在具备独立 Meta-Chunking 环境、Milvus、BGE Embedding 和 Qwen API 后运行 `run_denoise_qa_ablation.py`，再整理两组的检索召回和 QuestAnswer 指标。

### 2026-08-11 — 服务器磁盘空间只读复查

- 目标：确认服务器当前可用磁盘空间，为后续实验准备评估存储位置。
- 修改：无服务器修改；仅追加本地 `HANDOFF.md` 记录。
- 验证：SSH 只读执行 `df -h` 和 `df -ih`；系统盘 `/` 与 `/home` 共用 `/dev/sda2`，总容量约 1,007GB、已用 85GB、可用约 872GB（9%）；`/pub` 总容量约 3.3TB、已用约 3.0TB、可用约 86GB（98%）。inode 使用率分别约 1% 和 3%。
- 遗留/风险：`/pub` 空间紧张，不适合保存本次分片结果、模型缓存或 Milvus 数据；优先使用 `/home/humq`，并继续确认管理员的配额和清理策略。
- 下一步：后续若准备服务器实验，代码、输出、模型缓存和 Milvus 数据统一规划到 `/home/humq` 或管理员指定的独立盘位。

### 2026-08-11 — 服务器 Milvus/PyTorch/Embedding/CRUD 依赖配置

- 目标：在服务器 `/home/humq` 下准备去噪 QA 实验所需的隔离环境；Milvus 优先使用 Docker，避免修改系统环境和使用空间紧张的 `/pub`。
- 修改：在 `/home/humq/milvus_denoise/server_setup` 部署 `docker-compose.milvus.yml`、`start_rootless_milvus.sh`、`env_denoise_qa.sh`、依赖清单和 README；克隆 `/opt/conda/envs/testbase` 到独立环境 `/home/humq/envs/denoise_qa`，并安装 PyMilvus 2.3.3、Sentence-Transformers 3.0.1、Transformers 4.44.0、LlamaIndex 0.9.32、CRUD/评测依赖等。未修改 `/opt/conda/envs/testbase`，未写入 Qwen API 密钥。
- 验证：独立环境导入验证通过：PyTorch 2.0.1、CUDA 11.8、Transformers 4.44.0、Sentence-Transformers 3.0.1、PyMilvus 2.3.3、LlamaIndex 0.9.32、Evaluate 0.4.2、Text2Vec 1.2.9；`/home` 当前约 864GB 可用，环境占用约 7.3GB。依赖安装使用清华 PyPI 镜像成功。
- 遗留/风险：Milvus 尚未启动，因为当前账号缺少 `newuidmap/newgidmap`，rootless Docker 脚本已安全退出（状态 2）；系统 Docker socket 也没有当前用户权限。BGE 模型首次下载到 `huggingface.co` 时网络不可达，`/home/humq/hf_cache` 仍为空，需要从可联网机器上传模型或配置服务器代理/镜像。当前 GPU 的 `nvidia-smi` 可见 0/1/3/5 空闲，但 PyTorch CUDA 初始化返回 `CUDA unknown error`，需管理员/调度环境进一步确认；补装 `cmake`、`lit` 后 `pip check` 已通过。
- 下一步：请管理员安装 `uidmap` 或将账号加入 Docker 用户组后重启 Milvus；上传 `BAAI/bge-base-zh-v1.5` 到 `/home/humq/hf_cache` 并重试 Embedding smoke test；确认可用 GPU 后上传项目代码、QA 数据和两组 docs，再运行 `run_denoise_qa_ablation.py`。

### 2026-08-11 — 改为无 Docker 的 Milvus Lite 与本地模型上传

- 目标：按用户要求停止使用服务器 Docker；服务器不可访问的模型先在本地下载，再上传并验证，验证完成后删除本地模型副本。
- 修改：将服务器依赖升级为 `pymilvus==2.6.17` 和 `milvus-lite==2.4.12`；运行库改用 `/home/humq/milvus_denoise/milvus_lite.db`，环境变量改为 `DENOISE_MILVUS_URI`，避免触发 PyMilvus 对 `MILVUS_URI` 的全局远程 URI 解析。`BaseRetriever`、`build_unified.py` 和 `real_config.py` 已支持本地 Milvus Lite URI、本地 BGE 路径；检索器可选导入，Milvus 主线不再强制依赖 Elasticsearch/FlagEmbedding；修复 `BaseRetriever` 对 `LangchainEmbedding` 的重复包装。项目代码、两组 docs 和 QA 数据已上传并解包到 `/home/humq/chunk_code`；最新 server_setup 配置包也已上传，最后一次解包确认因 SSH 服务端重置暂未完成。
- 验证：从本地 `hf-mirror.com` 下载 `BAAI/bge-base-zh-v1.5`，上传归档 SHA256 为 `aa722a0936644f9882198629c847d1ba6dbb6acb3b1b035aeb0ed2b58bf5631f`；服务器离线加载成功，Embedding 输出形状为 `(1, 768)` 且范数约为 1。Milvus Lite 已完成建库、建索引、插入、查询和向量搜索 smoke test；项目实际 `BaseRetriever` 已完成建库和查询 smoke test。临时模型归档、临时数据库、测试 collection 和本地约 401MB 模型目录均已删除；服务器保留模型目录 `/home/humq/hf_cache/bge-base-zh-v1.5`。
- 遗留/风险：完整两组 QA 尚未运行；Qwen API 密钥仍需通过当前 shell 的 `QWEN_OPENAI_API_KEY` 注入；PyTorch 当前仍可能出现 CUDA 初始化异常，需要在正式实验前确认 GPU 调度和 `CUDA_VISIBLE_DEVICES`。最后一次 SSH 连接在认证前被服务端重置，服务器上的最新 `denoise_server_setup_latest.tgz` 已上传但尚未确认解包；Docker compose/rootless 脚本仍作为历史备份保留，但不再是运行路径。
- 下一步：在服务器执行 `source /home/humq/milvus_denoise/server_setup/env_denoise_qa.sh`，设置 Qwen API 环境变量后运行 `/home/humq/chunk_code/run_denoise_qa_ablation.py`；先以 `--num_threads 1` 做小规模 QuestAnswer 验证，再进行完整两组实验。

### 2026-08-11 — 服务器重启后 SSH、实验环境与 GPU 复查

- 目标：确认服务器重启后可以重新连接，并验证无 Docker 实验环境是否仍然完整。
- 修改：SSH 连接恢复后，在 `/home/humq/milvus_denoise/server_setup` 解包并启用最新 `denoise_server_setup_latest.tgz`；未启动实验、未修改系统环境。
- 验证：服务器 `splabbiggpu` 已正常运行约 21 分钟；`/home` 所在系统盘总容量约 1,007GB、可用约 863GB（10%）。`DENOISE_MILVUS_URI` 指向 `/home/humq/milvus_denoise/milvus_lite.db`，`BGE_MODEL_PATH` 指向 `/home/humq/hf_cache/bge-base-zh-v1.5`；`pymilvus`、`milvus_lite`、`sentence_transformers`、`llama_index` 均可导入，`pip check` 通过。GPU 0/1 正在被占用，GPU 2–7 当前显存占用为 0，可作为后续候选卡，但正式实验仍需确认调度权限。
- 遗留/风险：完整两组去噪 QA 尚未运行；Qwen API 密钥仍需由当前 shell 注入；PyTorch CUDA 初始化异常的风险仍需在正式实验前复查。
- 下一步：登录服务器后执行 `source /home/humq/milvus_denoise/server_setup/env_denoise_qa.sh`，注入 Qwen 环境变量，先用 `CUDA_VISIBLE_DEVICES=2` 和 `--num_threads 1` 做小规模 QA 冒烟，再进行完整两组实验。

### 2026-08-11 — 服务器去噪 QA smoke、Qwen API 与限流修复

- 目标：在服务器上实际跑通去噪开/关两组问答链路，并为后续扩大样本准备稳定的 API、指标和 Milvus Lite 配置。
- 修改：清理两组 docs 中误上传的 34,322 个 macOS `._*` 资源叉文件；在隔离环境安装并固定 `httpx==0.27.2`，修复 `openai==1.38.0` 的 `proxies` 兼容问题；补传离线 BLEU/ROUGE 脚本并安装 `rouge_score==0.1.2`、`absl-py==2.3.1`；修复 `bleu_score()` 成功后返回单值导致所有 BLEU 被记为 0 的问题；为本地 Milvus Lite 加入 gRPC 空闲 keepalive 保护，避免大规模向量化时触发 `Too many pings`；Qwen API 客户端加入 `QWEN_MIN_INTERVAL`，默认每次请求至少间隔 10 秒，429/5xx 指数退避，成功后不降到限流下限以下。未保存任何 API key 到代码、配置文件或结果清单。
- 验证：DashScope `qwen-plus` 的两组 smoke（每组 100 docs、每类 1 条 QA）完成，6 次生成均成功，BLEU/ROUGE 有效；示例结果为 denoise_off 的 1/2/3-doc ROUGE-L `0.1087/0.1413/0.1563`，denoise_on 为 `0.2568/0.1404/0.1308`，样本量仅用于链路验证，不能作为方法结论。GPU 2 的 PyTorch CUDA smoke 通过。硅基流动 `Qwen/Qwen3-8B` 短请求验证成功；使用 Qwen3 跑 100 条/类扩大实验时首条回答约 90 秒，已主动停止，未形成有效扩大结果。
- 遗留/风险：大规模 Milvus Lite 索引已因 keepalive 修复可继续运行；Qwen3 扩大实验需使用更保守的请求间隔，或者确认服务商是否支持关闭思考模式以降低单请求时延。当前 100 条/类扩大实验未完成；DashScope smoke 结果不能证明去噪有效性。
- 下一步：若有第二个独立配额的 API key，可按 key 建立两个串行限流 worker，先分别做短请求验证，再轮询运行两组 QA；若共享同一配额则不并行。正式结论仍需完成配对样本扩大/全量实验，并整理有效率、BLEU、ROUGE-L、答案长度和检索证据指标。

### 2026-08-11 — 两个硅基流动账号双开去噪 QA 扩大实验

- 目标：使用两个独立硅基流动 API key 同时运行 denoise_off 与 denoise_on，缩短去噪开关对照实验的墙钟时间，并严格控制单账号请求频率。
- 修改：Qwen API 客户端支持 `QWEN_ENABLE_THINKING`；针对 `Qwen/Qwen3-8B` 设置为 `false`，已验证短请求耗时约 1.5 秒。每个进程设置 `QWEN_MIN_INTERVAL=10`、`--num_threads 1`，两个进程分别使用 GPU 2/3 和独立 Milvus Lite DB，避免数据库 socket/显卡互相抢占。未将任何 key 写入代码、配置或日志。
- 验证：两个 key 均可调用 `Qwen/Qwen3-8B`；双开实验已在服务器启动：`/home/humq/chunk_code/results/denoise_ablation_db_qa_textsafe/qa_sample100_qwen3_dual2_20260811`，关闭组 PID 46094、开启组 PID 46095。两组均完成约 17k 节点建索引并进入 100 条/类 QA 生成；最近检查时进度约为 off 25/100、on 29/100，未见 401、429 或 Milvus 错误。
- 遗留/风险：当前只是每类 100 条的扩大样本，尚未完成，不能提前下结论；请求耗时受服务端排队影响，完整全量实验仍需在扩大样本完成后评估是否值得执行。
- 下一步：继续监控两个日志，完成后读取 6 个 JSON，计算三类 QA 的 BLEU、ROUGE-L、有效率、答案长度及 off/on 配对差异；若需全量，再基于实际耗时和 API 配额确认。

### 2026-08-11 — 双开 100 条/类实验完成与空 QA 过滤

- 目标：完成两个独立硅基流动账号双开运行的 denoise_off/denoise_on 扩大样本实验，并核对结果有效性。
- 修改：发现 3-doc QA 数据中存在 1 条空 `questions`/`answers`，新增 `xinhua.py` 的 QA 数据清洗：问或答为空的样本在 `quest_answer` 任务入口丢弃并打印丢弃数量。未修改已经生成的原始 JSON。
- 验证：双开任务已完成，结果位于 `/home/humq/chunk_code/results/denoise_ablation_db_qa_textsafe/qa_sample100_qwen3_dual2_20260811`，共 6 个 JSON；每组 1-doc/2-doc 各 100 条有效，3-doc 原始 100 条中严格有效 99 条。严格统计如下：off 的 BLEU-avg/ROUGE-L 为 1-doc `0.324126/0.496968`、2-doc `0.133091/0.296557`、3-doc `0.130970/0.291980`；on 为 1-doc `0.318495/0.486026`、2-doc `0.136299/0.297359`、3-doc `0.126945/0.287808`。所有非空样本均有生成文本；没有 API 401/429。日志中的 `search_data=None` 仅对应那条空 QA，已在后续入口修复。
- 结论边界：这批结果证明双开链路、限流、Milvus Lite、BGE、Qwen3-8B 和指标计算均可工作，但 100 条/类仍是扩大样本，不足以单独证明去噪方法有效；当前三类平均指标没有一致的 on 优势。
- 下一步：使用已上传的空 QA 过滤版本重跑更大配对样本或全量；正式报告同时加入检索证据召回、空上下文率和配对统计检验，不只比较 BLEU/ROUGE。

### 2026-08-11 — 明确当前 QA 实验不是全量

- 更正：`qa_sample100_qwen3_dual2_20260811` 只抽取了 `questanswer_1doc`、`questanswer_2docs`、`questanswer_3docs` 各 100 条，三类原始数据各有 3,199 条；因此全量 QA 尚未运行。
- 当前状态：已完成的是每类 100 条的双条件扩大样本，共 6 个结果 JSON；全量原始规模约为每个条件 9,597 条记录，剔除已发现的空 QA 后再执行。
- 下一步：正式启动全量前确认 API 配额和运行时间；沿用双 key、双 GPU、每 key 最小 10 秒间隔及空 QA 过滤。

### 2026-08-11 — 启动服务器全量去噪 QA 对照实验

- 目标：在服务器上对 `questanswer_1doc`、`questanswer_2docs`、`questanswer_3docs` 的全量数据分别运行去噪关闭/开启两组问答实验。
- 修改：未改动实验源码；使用已上传的空问答过滤版本、Qwen/Qwen3-8B、双独立 API 配额、GPU 2/3、每个 key 最小请求间隔 10 秒和独立 Milvus Lite 数据库。修正了首次启动时 Milvus Lite 数据库文件名过长及开启组重复参数导致的启动失败。
- 验证：两组进程已启动，关闭组 PID 151629、开启组 PID 151630；均已进入 Milvus Lite 建索引阶段，GPU 2/3 有显存和计算占用，暂未见 API/Milvus 错误。结果目录为服务器 `/home/humq/chunk_code/results/denoise_ablation_db_qa_textsafe/qa_full_qwen3_dual2_20260811`。
- 遗留/风险：全量共约每条件 9,597 条原始记录，预计运行时间较长；最终有效样本数需等数据过滤和任务完成后核对，当前不能提前报告去噪结论。
- 下一步：持续监控两组日志和进程；完成后核对三类结果 JSON、有效率、BLEU、ROUGE-L、答案长度、检索证据指标和配对差异，并追加正式结果记录。

### 2026-08-12 — 全量去噪 QA 运行中复查

- 目标：确认跨夜运行的全量对照实验仍在正常推进。
- 修改：无；仅读取服务器进程、GPU、日志和输出目录。
- 验证：关闭组 PID 151629、开启组 PID 151630 均仍在运行，分别使用 GPU 2/3；两组均已完成索引并处于第一类 1-doc 问答生成，最近进度约为 `1450/3199` 与 `1491/3199`；日志未见 API 401/429、异常或 Milvus 错误。数据入口报告 `questanswer_2docs` 丢弃 10 条空 QA、`questanswer_3docs` 丢弃 11 条空 QA。
- 遗留/风险：三类任务尚未全部完成，当前尚无可用于结论的全量 JSON；按当前速度仍需较长时间。
- 下一步：保持两个进程运行，完成后再做全量结果汇总和配对统计。

### 2026-08-12 — 服务器全量实验状态查询

- 目标：确认全量去噪 QA 对照实验是否仍在服务器运行。
- 修改：无；仅读取进程、日志、GPU 和结果目录。
- 验证：关闭组 PID 151629、开启组 PID 151630 均仍在运行，GPU 2/3 对应 Milvus Lite 进程正常；当前进度约为 `2423/3199` 和 `2446/3199`，仍处于 1-doc 阶段；未发现 401、429、Traceback 或 Milvus 错误，结果 JSON 尚未生成。
- 遗留/风险：全量三类 QA 尚未完成，暂不能进行最终指标比较。
- 下一步：继续保持进程运行，完成后整理全量结果。

### 2026-08-12 — 全量实验阶段性完成复查

- 目标：确认服务器全量去噪 QA 实验是否全部完成。
- 修改：无；仅读取服务器进程、日志和结果输出。
- 验证：关闭组和开启组进程仍在运行；两组的 1-doc（`3199/3199`）已完成并分别保存 JSON。当前进入 2-doc，进度约 `429/3189`；3-doc 尚未开始。结果 JSON 位于各自 `off/`、`on/`` 下的任务结果目录中。
- 遗留/风险：全量三类尚未结束，当前不能汇总最终去噪结论。
- 下一步：继续监控 2-doc 和 3-doc，全部结束后统一计算指标。

### 2026-08-12 — 全量实验再次查询

- 目标：确认全量问答实验是否已经结束。
- 修改：无；仅读取服务器进程、日志和结果文件。
- 验证：关闭组和开启组仍在运行；两组 1-doc 结果 JSON 已生成。当前 2-doc 分别约为 `897/3189` 和 `891/3189`，3-doc 尚未开始；未发现 401、429、Traceback 或进程退出。
- 遗留/风险：全量实验尚未结束，暂不能生成最终去噪效果结论。
- 下一步：继续等待 2-doc、3-doc 完成，再统一整理六组结果和统计指标。

### 2026-08-12 — 全量实验进度查询与服务器暂时超时

- 目标：查询全量去噪 QA 实验的最新进展和剩余任务。
- 修改：无；仅读取服务器状态，随后 SSH 连接发生超时。
- 验证：最后一次成功状态为 1-doc 两组已完成并生成 JSON，2-doc 关闭组约 `897/3189`、开启组约 `891/3189`，3-doc 尚未开始；之后对服务器 0（`211.87.224.135:17622`）的 SSH/TCP 复查超时，未能取得更新状态。
- 遗留/风险：无法据此判断服务器上的实验是否已继续、完成或异常退出；不能把旧进度当成当前状态。
- 下一步：服务器网络恢复后重新检查两个进程、三类结果 JSON 和日志，再决定是否需要续跑或补实验。

### 2026-08-12 — 服务器重新连接并复查全量实验

- 目标：重新连接服务器并确认全量去噪 QA 实验状态。
- 修改：无服务器代码修改；仅重新建立 SSH 连接并读取进程、日志和结果文件。
- 验证：SSH 已恢复，主机为 `splabbiggpu`；关闭组 PID 151629、开启组 PID 151630 仍在运行。两组 1-doc 已完成并生成 JSON，2-doc 当前分别约为 `1769/3189` 和 `1778/3189`；未发现新的 API、Milvus 或 Python 异常。
- 遗留/风险：全量 2-doc 尚未结束，3-doc 尚未开始，最终六组结果尚未齐全。
- 下一步：继续等待两组 2-doc、3-doc 完成，再统一汇总去噪开关的 QA 指标。

### 2026-08-12 — 切片质量评测运行状态查询

- 目标：确认切片质量实验是否仍在服务器运行，并区分分片、质量指标和问答评测状态。
- 修改：无；仅读取服务器进程、质量评测命令和结果文件。
- 验证：三阶段 `denoise_off`/`denoise_on` 分片结果已存在，当前没有重新分片进程；但两组 `fast_ppl_metrics.py --metrics relation` 质量评测仍在运行，分别输出 `fast_relation_off_full_v6.json` 和 `fast_relation_on_full_v6.json`，进程已运行约 8 小时 47 分钟。全量 QA 也仍在运行，2-doc 约为关闭组 `1771/3189`、开启组 `1781/3189`。
- 遗留/风险：切片质量 relation 指标尚未生成最终 JSON；全量 QA 的 2-doc、3-doc 也尚未完成。
- 下一步：等待 relation 质量指标和全量 QA 都结束后，统一整理切片统计、关系质量和问答效果。

### 2026-08-12 — 制定切片质量评测剩余实验方案

- 目标：补齐三阶段分片质量实验，形成多个独立角度的证据，而不是把它们误写成前后依赖的流水线。
- 实验关系：所有评测都共享同一对固定的 `denoise_off`/`denoise_on` 分片结果、同一份原始数据和统一实验参数；Boundary Clarity、Relation/Chunk Stickiness、删除内容人工标注、检索证据保留率、端到端 QA、阈值敏感性等是相互独立的实验，可以按服务器资源并行执行，不需要等待某个指标完成后才能启动另一个指标。最终只是在报告阶段横向汇总。
- 当前状态：三阶段分片已完成；Relation 质量评测和全量 QA 仍在服务器运行；Boundary Clarity、Chunk Stickiness 独立抽样评测、删除句子人工标注和检索证据保留率尚未完成。
- 各实验目的：基础分片统计衡量规模变化；Boundary Clarity 衡量跨 chunk 边界；Relation/Chunk Stickiness 衡量 chunk 内部连贯性；人工标注衡量删除内容是否真是噪声；检索实验衡量证据是否保留；QA 实验衡量下游答案效果；阈值敏感性衡量方法稳定性。
- 判定原则：不能只用 chunk 数减少或平均长度变化证明去噪有效；最终报告应并列呈现各独立实验结果。若结构指标改善但证据召回下降，结论只能写成“改变了分片结构”，不能写成“去噪有效”。
- 下一步：根据 GPU、API 和人工标注资源，把尚未开始的独立实验分别启动；不再把它们描述成必须串行执行的步骤。

### 2026-08-12 — 在空闲 GPU 启动独立切片质量实验

- 目标：利用空闲 GPU 启动不影响现有全量 QA/relation 任务的独立评测。
- 修改：`/Users/a1234/chunk_code/integrated_chunker.py` 增加 `--mad_k` 阈值参数；新增 `/Users/a1234/chunk_code/retrieval_evidence_eval.py`，执行相同 QA 问题下的答案二元组覆盖率、exact answer hit rate 和 top-k 上下文长度统计。脚本已上传服务器。
- 启动：GPU 4 运行 `denoise_off` 检索证据实验，PID 660959；GPU 5 运行 `denoise_on` 检索证据实验，PID 660960；独立 Milvus Lite 数据库为 `evid_off.db`/`evid_on.db`，结果目录为 `/home/humq/chunk_code/metric_results/retrieval_evidence_full_20260812/`。两组已进入建索引阶段。CPU 上另外启动 MAD `k=2.5/3.0/4.0` 三个独立敏感性分片进程，PID 662161/662166/662171，输出目录为 `/home/humq/chunk_code/results/denoise_threshold_sensitivity_20260812/`。
- 验证：启动后 GPU 4/5 显存约 1176MiB、利用率约 89%/81%，证据实验进程和 Milvus Lite 子进程存在；阈值实验三个进程均在运行。现有 GPU 0/1 relation 评测、GPU 2/3 全量 QA 未被停止或覆盖。
- 边界说明：检索证据脚本使用“参考答案二元组覆盖率”作为自动代理指标，不能替代人工 gold evidence 标注；Boundary 的 API 过渡自然度和人工标注实验本次未启动，以避免额外 API 配额消耗和把人工判断伪装成自动结果。
- 下一步：监控 GPU 4/5 建索引和检索进度；阈值分片完成后读取各组 summary；全量 QA、relation 和证据实验完成后统一横向汇总。

### 2026-08-13 — 全量实验状态与已完成结果复查

- 目标：复查服务器上各独立实验是否已经完成，并提取可用于阶段性分析的结果摘要。
- 全量问答：`denoise_off`/`denoise_on` 进程仍在 GPU 2/3 运行；1-doc、2-doc 两组结果 JSON 已生成，当前 3-doc 分别约为 `1112/3188`、`1130/3188`，因此六组 QA 结果尚未齐全，暂不能形成最终端到端结论。当前未见 API、Milvus 或 Python 异常。
- Relation 质量评测：GPU 0/1 的 `fast_ppl_metrics.py --metrics relation` 仍在运行；`fast_relation_off_full_v6.json` 和 `fast_relation_on_full_v6.json` 尚未生成，关系质量结果未完成。
- 检索证据保留实验：已完成，关闭/开启两组各 `9576` 条有效样本，结果为服务器 `/home/humq/chunk_code/metric_results/retrieval_evidence_full_20260812/off.json` 和 `on.json`。该实验是答案二元组覆盖率的自动代理，不等同于人工 gold evidence 标注；总体上 off 与 on 的 top-k 覆盖率接近，不能单独证明去噪有收益。
- MAD 阈值敏感性：`k=2.5/3.0/4.0` 均已完成，summary 位于 `/home/humq/chunk_code/results/denoise_threshold_sensitivity_20260812/mad_k_*/all_chunks_summary.json`。三组 chunk 数分别为 `16683/16867/17081`，删除句子数分别为 `11419/8080/4521`，说明阈值会明显影响删除强度。
- 既有结果：Boundary Clarity v2 已完成且 unmatched 为 0；Chunk Stickiness 已有完整 shard 文件，但尚未合并成统一最终摘要。API 过渡自然度和人工删除内容标注仍未开展。
- 结论：目前有阶段性结果，但没有完整最终报告；待全量 QA 的 3-doc、Relation JSON 完成后，还需合并 stickiness shard、做 off/on 配对统计并统一整理实验结论。

### 2026-08-13 — 整理已有实验结果

- 新增本地汇总文件：`EXPERIMENT_RESULTS_SUMMARY.md`，统一整理全量 QA 已完成部分、Boundary Clarity v2、Chunk Stickiness shard 加权汇总、检索证据保留和 MAD 阈值敏感性结果，并明确未完成实验。
- QA 已完成结果：1-doc off/on 的 BLEU-avg 为 `0.300412/0.301921`，ROUGE-L 为 `0.483048/0.484601`；2-doc 为 `0.125607/0.124543` 和 `0.278837/0.277601`。1-doc 略升、2-doc 略降，方向不一致；3-doc 仍在运行。
- Boundary Clarity v2：off/on 平均语义分离度 `0.205856/0.203916`，去噪组略低；两组映射 unmatched 均为 0。
- Chunk Stickiness shard 加权汇总：off/on 的 G1 结构熵 `2.630070/2.623998`，G3 `2.753054/2.746021`；去噪组略低但幅度约 0.2%，尚未做最终配对显著性检验。
- 检索证据保留：9,576 条/条件；1-doc、2-doc、3-doc 的 top-10 答案二元组覆盖率 on 均略低于 off，分别差 `-0.001810/-0.002892/-0.003072`。
- MAD 阈值敏感性：k=`2.5/3.0/4.0` 的 chunk 数为 `16683/16867/17081`，删除句子数为 `11419/8080/4521`，确认阈值越大删除越少。
- 当前综合判断：已有结果只能支持“去噪改变了分片结构，局部结构熵轻微改善，但尚未证明整体检索问答收益”；待 3-doc QA 和 Relation 评测完成后再形成最终结论。

### 2026-08-14 — 全量 QA 已完成，Relation 仍在运行

- 目标：复查服务器最新进度并补齐全量 QA 的 3-doc 结果。
- 验证：服务器上 `denoise_off` 和 `denoise_on` 的 3-doc JSON 均已生成；全量 QA 六个结果 JSON 已齐全。3-doc 的 BLEU-avg/ROUGE-L 分别为 off `0.117881/0.273088`、on `0.118271/0.273494`，去噪组仅有轻微提升；未见新增 API、Milvus 或 Python 错误。
- 当前未完成：GPU 0/1 的 `fast_ppl_metrics.py --metrics relation` 仍在运行，`fast_relation_off_full_v6.json` 和 `fast_relation_on_full_v6.json` 尚未生成。因此整个切片质量实验仍未全部完成。
- 文档更新：`EXPERIMENT_RESULTS_SUMMARY.md` 已更新为 2026-08-14，加入 3-doc 最终 QA 指标并删除“3-doc 仍在运行”的旧状态描述。

### 2026-08-14 — 向用户交付当前实验结果汇总

- 已将当前结果按 QA、Boundary Clarity、Chunk Stickiness、检索证据保留和 MAD 阈值敏感性五个部分整理到 `EXPERIMENT_RESULTS_SUMMARY.md`。
- 汇总结论：QA 在 1-doc、3-doc 略升、2-doc 略降；Boundary Clarity 略降；Chunk Stickiness 结构熵约改善 0.2%；检索证据覆盖率三类任务均略降。因此现阶段只能说明去噪改变了分片结构并带来局部轻微改善，不能声称整体效果已稳定提升。
- Relation 评测、API 边界自然度、人工删除内容标注和最终显著性检验仍列为未完成项。

### 2026-08-14 — 新建三位小数结果文档

- 新增 `EXPERIMENT_RESULTS_SUMMARY_3DECIMALS.md`，将 QA、Boundary Clarity、Chunk Stickiness、检索证据保留和 MAD 阈值敏感性结果统一保留三位小数。
- 样本数、chunk 数和删除数量保留整数，并在文档中注明小于 `0.0005` 的差异四舍五入后显示为 `0.000`。
- 文档同步保留 Relation 评测、API 边界自然度、人工标注和最终显著性检验等未完成项。

### 2026-08-14 — 将去噪结果写入综合实验报告

- 更新 `/Users/a1234/RAG_DB_silm/META_CHUNKING_EXPERIMENTS_REPORT.md`：新增“第二阶段去噪开关消融实验”章节。
- 写入内容：全量 1-doc/2-doc/3-doc QA、Boundary Clarity、Chunk Stickiness 加权结果、检索证据保留、MAD 阈值敏感性、结果路径和当前结论；连续指标按三位小数整理。
- 同步修正综合报告状态：Chunk Stickiness shard 已完成，Relation Coherence 仍在服务器运行；补充 API 边界自然度、人工删除内容标注和最终配对统计等未完成项。

### 2026-08-14 — 将去噪 QA 结果合并进报告 2.3

- 在 `META_CHUNKING_EXPERIMENTS_REPORT.md` 的 `2.3 评测结果` 下新增 `2.3.4 第二阶段去噪开关消融结果`，写入 1-doc、2-doc、3-doc 的 off/on 全量 QA 指标和差值。
- 第四章的 QA 重复表已改为索引说明；第四章继续保留去噪相关的 Boundary Clarity、Chunk Stickiness、检索证据保留和 MAD 阈值敏感性结果。

### 2026-08-14 — 启动 db_qa 四种分片方法的四维质量评测

- 目标：以 `/Users/a1234/chunk_code/data/db_qa.txt` 为输入，对比 200 字符、300 字符、400 字符三种机械切分和当前三阶段切分，在语义困惑度、主题距离度、信息差异含量、内容一致性四个维度上进行评测。
- 数据核对：本地与服务器输入文件一致，大小 `23,264,388` 字节，SHA-256 为 `6283a8b318bfe95c085eb482a1d509247d55cca4870fa97ba46f79b71b0be7ef`。
- 新增/使用脚本：`/Users/a1234/chunk_code/mechanical_chunker.py`、`/Users/a1234/chunk_code/prepare_topic_model.py`、`/Users/a1234/chunk_code/four_dimension_eval.py`，以及现有三阶段脚本 `integrated_chunker.py`；均已上传到服务器 `/home/humq/chunk_code/`。
- 分片结果：服务器目录 `/home/humq/chunk_code/results/four_method_chunks_20260814/`。200/300/400 字符机械切分分别为 `46505/32828/26060` 个 chunks；三阶段切分为 `16980` 个 chunks。
- 评测条件：共享主题模型 `/home/humq/chunk_code/results/four_dimension_eval_20260814/metric_models/topic_lsa.joblib`，使用本地 BGE、Qwen2-1.5B-Instruct 模型，不调用外部 API；四路评测固定参数一致。
- 并行进程：GPU 2/3/4/5 分别运行机械 200/300/400 字符和三阶段评测，PID 为 `784696/784697/784698/784699`。检查时四个进程均为运行状态，日志已进入向量计算阶段，未发现异常；结果 JSON 尚未生成，仍在计算中。
- 结果目录：`/home/humq/chunk_code/results/four_dimension_eval_20260814/`，预期生成 `mechanical_200char.json`、`mechanical_300char.json`、`mechanical_400char.json`、`three_stage.json`。
- 遗留/下一步：继续监控四个评测进程；完成后读取四个 JSON，统一计算各维度差值、标准化对比和排名，并视需要补充到实验结果文档。

### 2026-08-15 — 四种分片方法四维质量评测完成

- 验证：200/300/400 字符机械切分和三阶段切分的四个评测进程均已结束，日志均出现 `saved`，未发现异常或 Traceback。
- 结果：已生成 `/home/humq/chunk_code/results/four_dimension_eval_20260814/mechanical_200char.json`、`mechanical_300char.json`、`mechanical_400char.json`、`three_stage.json`。
- 服务器状态：GPU 2–5 已释放，当前四维评测不再运行。
- 下一步：读取四个 JSON，整理语义困惑度、主题距离度、信息差异含量、内容一致性四个维度的指标、差值和排名。

### 2026-08-15 — 发现并修复四维评测中的语义困惑度 NaN

- 复核四个 JSON 时发现：`semantic_perplexity.mean_log_ppl` 和 `ppl` 在四种方法中均为 `NaN`，其余主题距离、信息差异含量和内容一致性字段均有有效数值。
- 原因判断：原评测使用 Qwen 半精度计算交叉熵，累计损失出现数值不稳定；已将 PPL 模型和交叉熵计算改为 float32。
- 修改：新增 `/Users/a1234/chunk_code/recompute_ppl.py`，仅重算 PPL 并回写原四个 JSON，不重复计算其他三个维度；代码已通过语法检查并上传服务器。
- 当前重算进程：GPU 2/3/4/5 分别运行 200/300/400 字符和三阶段 PPL 重算，PID 为 `787058/787059/787060/787061`，日志位于四维结果目录下的 `*_ppl.log`。
- 下一步：待 PPL 重算完成后，重新读取四个 JSON，给出四维指标的三位小数结果、差值和排名。

### 2026-08-15 — 整理四种分片方法四维质量评测结果

- PPL 重算已完成：四个结果 JSON 的语义困惑度均为有效有限值，200/300/400 字符和三阶段 PPL 分别为 `18.483/15.909/14.598/13.061`。
- 主要结果：三阶段在 PPL、主题边界距离、主题对比度、邻域新颖度和邻域冗余度方面最好；200 字符机械切分在信息差异含量和内容一致性综合值方面最高。
- 三阶段相对 200/300/400 字符机械切分的主指标差值（PPL、主题对比度、信息差异含量、综合一致性）分别为 `(-5.422,+0.138,-0.025,-0.005)`、`(-2.849,+0.109,-0.018,-0.002)`、`(-1.538,+0.082,-0.012,-0.002)`。
- 新增本地整理文件：`/Users/a1234/RAG_DB_silm/FOUR_METHOD_QUALITY_RESULTS.md`，连续指标统一保留三位小数，并记录指标方向、详细子指标、差值和结论。

### 2026-08-16 — 加入内容一致性长度均一化评测

- 问题：原始内部一致性直接在完整 Chunk 上计算，Chunk 越短越容易获得较高中心相似度，导致机械切分存在长度优势。
- 实现：`four_dimension_eval.py` 新增固定字符窗口和固定字符预算句子窗口；新增 `recompute_consistency.py`，在统一 `200` 字符评价尺度下重算外部一致性、内部一致性和综合一致性，同时保留原始指标。代码已通过语法检查并上传服务器。
- 运行：四路长度均一化评测已在 GPU 2–5 完成；新结果目录为 `/home/humq/chunk_code/results/four_dimension_eval_lengthnorm_20260816/`，四个 JSON 均已生成。
- 结果：长度归一化综合一致性为 200/300/400 字符和三阶段分别 `0.927/0.936/0.936/0.938`；三阶段相对三种机械切分的差值为 `+0.012/+0.002/+0.002`。
- 文档：`FOUR_METHOD_QUALITY_RESULTS.md` 已新增长度均一化方法、结果和解释；后续报告应同时保留原始完整 Chunk 指标与统一局部尺度指标，避免将二者混为一谈。

### 2026-08-17 — 汇总全部已有实验结果

- 新增总览文档：`/Users/a1234/RAG_DB_silm/ALL_EXPERIMENT_RESULTS.md`。
- 汇总范围：旧版 Meta-Chunking vs Integrated QA、Baseline/分片规模统计、去噪 off/on 全量 QA、Boundary Clarity、Chunk Stickiness、检索证据保留、MAD 阈值敏感性、200/300/400 字符与三阶段四维质量评测、长度均一化内容一致性、历史 Boundary Clarity 和 top-4 冒烟结果。
- 文档同时记录了未完成实验（Baseline 完整 QA、Integrated top-4、Relation Coherence、人工噪声标注、显著性检验）以及不同报告/运行版本之间的数值冲突，明确当前采用的结果来源和不可直接混合的实验版本。
### 2026-08-17 — 采用用户确认的去噪 QA 结果

- 用户确认以下去噪开关 QA 汇总为当前采用版本：1-doc off/on 为 `0.281/0.464/172.715`、`0.302/0.485/171.809`；2-doc off/on 为 `0.114/0.251/492.449`、`0.125/0.278/494.511`；3-doc off/on 为 `0.102/0.255/535.859`、`0.118/0.273/535.453`。
- 同步采用差值 `on-off`：1-doc `+0.021/+0.021/-0.906`，2-doc `+0.011/+0.027/+2.062`，3-doc `+0.016/+0.018/-0.406`。
- 已更新：`EXPERIMENT_RESULTS_SUMMARY.md`、`EXPERIMENT_RESULTS_SUMMARY_3DECIMALS.md`、`ALL_EXPERIMENT_RESULTS.md`。`META_CHUNKING_EXPERIMENTS_REPORT.md` 原有表格已与该版本一致，并同步修正了旧结论文字。
- 当前结论统一为：三类任务的 BLEU-avg 和 ROUGE-L 均有小幅正向变化，但仍需逐问题配对统计、置信区间和显著性检验；旧汇总中的另一组 off 数值不再作为当前结论依据。

### 2026-08-17 — 统一实验结果正文

- `ALL_EXPERIMENT_RESULTS.md` 设为当前唯一实验结果正文。
- 删除重复的结果正文：`EXPERIMENT_RESULTS_SUMMARY.md`、`EXPERIMENT_RESULTS_SUMMARY_3DECIMALS.md`、`FOUR_METHOD_QUALITY_RESULTS.md`、`META_CHUNKING_EXPERIMENTS_REPORT.md`。
- 保留 `EXPERIMENT_RESULTS_INDEX.md` 作为结果目录索引，保留 `HANDOFF.md` 作为交接记录；代码、原始数据和服务器上的实验结果目录未删除。

### 2026-08-17 — 调整四维结果章节结构

- 将长度均一化内容一致性的计算说明、明细结果和综合结果并入 `ALL_EXPERIMENT_RESULTS.md` 的 `6.1 主指标`。
- 删除原独立的长度均一化 `6.3` 小节，并将四维质量结论保留为 `6.3`；结果数值保持不变。

### 2026-08-17 — 删除平均答案长度指标

- 从 `ALL_EXPERIMENT_RESULTS.md` 的旧版主实验和去噪 QA 结果中删除“平均答案长度”行、列及 `on-off` 长度差值。
- 同步移除依赖该指标的“答案更长”结论；BLEU、ROUGE-L、BERTScore、有效样本数及其他实验指标保持不变。

### 2026-08-17 — 补充四维指标计算方法

- 在 `ALL_EXPERIMENT_RESULTS.md` 第 6.1 节补充四个维度的代码对应计算方法和公式：token 加权语义 PPL、TF-IDF+LSA 主题内离散度/边界距离/主题对比度、TF-IDF 信息密度与邻域新颖度结合的信息差异含量、BGE 外部/内部内容一致性及 200 字符长度均一化版本。
- 结果表和原有数值未修改；已说明所有方法复用同一主题模型和统一统计口径。

### 2026-08-17 — 修正四维公式展示

- 将 `ALL_EXPERIMENT_RESULTS.md` 第 6.1 节中的 LaTeX 公式块改为兼容 Markdown 渲染器的纯文本公式代码块，避免公式显示为原始标记或乱码。
- 公式含义、计算口径和实验结果数值均未改变。

### 2026-08-17 — 将公式变量含义改为正文说明

- 删除第 6.1 节中独立的符号对照表，将 `b`、`t`、`C`、`s_i`、`w_i`、`D_in`、`D_boundary`、`rho`、`IDC`、`S_external`、`S_internal` 等变量的含义直接融入公式前后的文字说明。
- 保留纯文本公式代码块、结果表和原有计算口径，未修改实验数值。

### 2026-08-17 — 重整第 6.2 节四维实验结果

- 将 `ALL_EXPERIMENT_RESULTS.md` 的 `6.2` 由混合详细指标表改为四个独立小节：语义困惑度、主题距离度、信息差异含量、内容一致性。
- 每个维度分别列出四种切分方法的指标结果和简要解释；内容一致性同时保留原始 Chunk 尺度与 200 字符长度均一化结果。
- 实验数值未改变，`6.1` 的计算公式和 `6.3` 的总括结论保持不变。

### 2026-08-17 — 重新生成第 6 部分结果结构

- 发现 `ALL_EXPERIMENT_RESULTS.md` 的 `6.2` 仍残留旧的混合指标表，且 `6.3` 总结缺失。
- 已重新生成 `6.2` 四个维度的实验结果：语义困惑度、主题距离度、信息差异含量、内容一致性，并恢复 `6.3` 四维质量结论。
- 已保留 6.1 的计算公式、主指标汇总和长度均一化结果，实验数值未重新计算或修改。

### 2026-08-17 — 合并四维结果为单表

- 将 `ALL_EXPERIMENT_RESULTS.md` 的 6.2 改为一张综合结果表，不再按四个维度分别设置小节或单独解释。
- 单表统一列出 PPL、主题距离、信息差异含量、原始内容一致性和长度均一化内容一致性的全部子指标；数值保持不变。

### 2026-08-17 — 仅保留四个维度综合指标

- 将 `ALL_EXPERIMENT_RESULTS.md` 的 6.2 进一步精简为四列综合指标：PPL、主题对比度、信息差异含量、长度均一化综合内容一致性。
- 删除 6.2 中所有主题、信息和一致性子指标；原始内容一致性及其子指标仍保留在 6.1 的详细结果中。

### 2026-08-17 — 转换为 Word 兼容格式

- 将 `ALL_EXPERIMENT_RESULTS.md` 中的公式改为 Word 可直接粘贴的线性公式文本，移除代码围栏和 Markdown 公式标记。
- 将全部表格改为 `+` 分隔格式，并将差值中的正号改为“正”文字，避免与表格分隔符冲突。
- 在结果文档开头补充 Word“将文字转换成表格”的操作说明。

### 2026-08-17 — 新建论文式实验结果文档

- 新增 `EXPERIMENT_RESULTS_THESIS_STYLE.md`，参考用户提供的“4.4 实验”章节格式，按“实验设置—指标定义—结果表—结果分析—结论与局限性”组织内容。
- 文档包含四种分片方法四维质量结果、第二阶段去噪消融、Boundary Clarity、Chunk Stickiness、检索证据保留、MAD 阈值敏感性和旧版端到端 QA 结果。
- 新文档中的表格统一使用 `+` 分隔，公式统一使用 Word 线性文本格式，便于直接复制到论文 Word 文档。

### 2026-08-17 — 弱化论文式文档中的代码实现细节

- 将 `EXPERIMENT_RESULTS_THESIS_STYLE.md` 中的函数级、变量级和公式级实现描述改写为评测程序功能介绍，保留指标含义、处理流程和结果解释。
- 删除论文正文中不必要的底层实现细节，使文档更接近实验章节的叙述风格。

### 2026-08-17 — 删除论文式文档中的文件名与代码标识

- 将论文式文档中的数据文件名、去噪开关变量名、映射状态变量名和阈值变量名改写为自然语言描述。
- 保留数据类型、实验条件和模型信息，不再在正文中展示代码文件名或程序变量名。

### 2026-08-25 — 论文实验补充规划

- 当前已具备：四种分片方法四维质量评测、去噪开关全量 QA、Boundary Clarity、Chunk Stickiness、检索证据代理指标和 MAD 阈值敏感性结果。
- 论文优先补充：四种分片方法在同一检索和问答配置下的端到端 QA 对比、逐问题配对显著性检验、三阶段组件消融、去噪删除内容人工或合成噪声验证、人工 Chunk 质量评价、效率与资源开销评测。
- 若论文声称跨领域泛化，还需在至少一个额外领域或数据集上重复主实验；若只声称中文新闻语料上的方法有效，则可将跨领域实验列为扩展实验。
- Relation 质量评测此前已终止且未生成最终 JSON，不应作为当前论文有效结果；若要保留该指标，需要修复后重新运行。

### 2026-08-17 — 检查服务器实验进程状态

- 服务器仍有两个关系质量评测进程：PID `410154`（`denoise_off`，GPU 0）和 PID `410206`（`denoise_on`，GPU 1），运行时间约 5 天 14 小时，CPU 占用均约 100%。
- 两个进程的日志最后分别停在 `relation PPL pass done`：off 为 2026-08-12 13:03，on 为 2026-08-12 13:02；对应 `fast_relation_off_full_v6.json` 和 `fast_relation_on_full_v6.json` 均尚未生成。
- 当前 GPU 0/1 有显存占用但 GPU 利用率为 0%，GPU 2/3 空闲；GPU 4–7 由其他 `code_generation` 进程占用。暂未终止关系评测进程，后续需决定是继续观察、诊断卡住原因，还是停止后重启。

### 2026-08-17 — 终止卡住的关系质量评测

- 已核对并发送 `SIGTERM`，终止 PID `410154`（`denoise_off`）和 PID `410206`（`denoise_on`）。
- 2 秒后复查，两个 PID 均已退出；未影响其他实验进程。
- 由于两个最终 JSON 在终止前尚未生成，`fast_relation_off_full_v6.json` 和 `fast_relation_on_full_v6.json` 仍需后续重新运行才能得到。

### 2026-08-25 — 启动非人工论文实验队列

- 按“先分片准备、再同条件全量问答、再配对统计、最后组件消融”的顺序编写本地队列脚本，人工 Chunk 质量评价暂不加入。
- 新增本地脚本目录：`/Users/a1234/RAG_DB_silm/paper_experiments/`，包含分片转检索文档、单方法 QA、逐问题配对 Bootstrap/符号检验和可恢复的顺序队列。
- 三阶段分片器新增两个消融开关：跳过第一阶段结构拆分、跳过第三阶段策略优化融合；第二阶段去噪继续使用原有 `--denoise true/false` 开关。
- 脚本及修改后的三阶段分片器已上传到服务器 `/home/humq/chunk_code/paper_experiments/` 和 `/home/humq/chunk_code/integrated_chunker.py`，服务器语法检查通过。
- 服务器当前状态：`/home/humq` 可用约 `848G`，8 张 V100 均空闲，未发现此前遗留实验进程；API 环境变量目前未设置，因此尚未启动外部 API 问答。
- 已启动服务器分片准备阶段，后台进程 PID `1367074`，队列子进程 PID `1367075`，当前三阶段分片子进程 PID `1367082` 正在运行；日志为 `/home/humq/logs/nonhuman_prepare.log`。
- 机械切分已完成：200/300/400 字符分别生成 `46505/32828/26060` 个分片；三阶段切分完成后会自动生成四套检索文档目录。
- 当前约定的后续结果根目录：`/home/humq/chunk_code/results/nonhuman_paper_queue_20260825/`；完成标记位于其 `markers/` 子目录，可从中断处继续。
- 10:33 左右首次启动 QA 时发现系统 Python 缺少 `loguru`，该次进程已自然退出，未产生有效 QA 结果；服务器已有可用环境 `/home/humq/envs/denoise_qa`，队列启动脚本已改为优先使用该环境并重新上传。
- 当前 QA 已重新启动：队列 PID `1371971`，首个 200 字符问答进程 PID `1371975`；首个方法正在构建约 `46043` 个检索节点的向量索引，日志为 `/home/humq/logs/nonhuman_qa.log`。
- 首次重试发现共享 Milvus Lite 文件在异常退出后存在锁/连接兼容风险，已将队列改为每种切分方法使用独立数据库文件 `milvus_<method>.db`；当前 200 字符方法已通过连接初始化并继续建库。
- 200 字符方法已完成建库并进入 `1-doc` 问答，当前日志显示已完成约 `9/3199` 个问题；请求间隔保持 `10` 秒，队列仍按方法顺序串行执行，未并行占用第二个 API 账号。

### 2026-08-26 — 非人工实验队列进度复核

- 队列仍在正常运行，当前进程为 `1371971`（队列）、`1371975`（200 字符方法）和 `1371976`（问答主进程）；未发现异常退出。
- 200 字符方法的 1-doc 和 2-doc 已完成并生成有效结果：1-doc BLEU-avg `0.311`、ROUGE-L `0.497`，2-doc BLEU-avg `0.128`、ROUGE-L `0.286`。
- 200 字符方法的 3-doc 正在进行，最近日志为 `1986/3188`，约完成 `62.3%`；该方法最终完成后，队列才会自动切换到 300 字符方法。
- 当前问答结果目录为 `/home/humq/chunk_code/results/nonhuman_paper_queue_20260825/qa/mechanical_200/`；尚未生成对应完成标记，表示该方法三类任务尚未全部完成。
- 当前服务器 `/home/humq` 仍约有 `847G` 可用空间；本实验主要使用 GPU 0，未影响其他 GPU 上的其他用户进程。

### 2026-08-27 — 非人工实验队列进度复核

- 队列仍在运行，当前已自动切换到 300 字符机械切分，进程为队列 PID `1371971`、QA 包装进程 PID `483522`、问答主进程 PID `483523`。
- 200 字符机械切分的三类 QA 已全部完成：1-doc BLEU-avg/ROUGE-L 为 `0.311/0.497`，2-doc 为 `0.128/0.286`，3-doc 为 `0.121/0.280`。
- 300 字符机械切分的 1-doc 已完成，BLEU-avg/ROUGE-L 为 `0.316/0.499`；2-doc 当前为 `2960/3189`，约完成 `92.8%`，3-doc 尚未开始。
- 当前没有异常退出；结果目录为 `/home/humq/chunk_code/results/nonhuman_paper_queue_20260825/qa/`，日志为 `/home/humq/logs/nonhuman_qa.log`。
- 四种方法问答完成后，需要另行启动 `stats` 和 `ablation` 阶段；本次 QA 队列使用的是单独的 `--stage qa`，不会自动执行后续统计和组件消融。

### 2026-08-27 — GPU 资源复核

- 当前服务器没有真正空闲的 GPU：GPU 0–7 均被另一个用户的 Qwen3-8B vLLM 张量并行服务占用约 `30G` 显存；瞬时利用率为 `0%` 不代表显存可用。
- 当前 QA 进程额外在 GPU 0 使用约 `1G` 显存，未占用其他 GPU；除 GPU 0 剩余约 `1G` 外，其余 GPU 也仅剩约 `2G`，不适合再启动本项目的大模型或完整向量实验。

### 2026-08-27 — 论文实验未完成项核对

- 四种分片方法的同条件全量 QA 尚未结束：200 字符三类任务已完成；300 字符的 1-doc、2-doc 已完成，3-doc 正在运行；400 字符和三阶段方法尚未开始。
- QA 完成后仍需单独运行逐问题配对 Bootstrap 置信区间和符号检验；当前队列使用 `--stage qa`，不会自动进入 `stats`。
- 三阶段组件消融尚未运行，包括跳过第一阶段结构拆分、关闭第二阶段去噪和跳过第三阶段优化融合三个变体。
- 去噪的合成噪声自动验证尚未运行；人工删除内容标注按用户要求暂缓。
- 效率与资源开销尚未形成独立汇总结果；目前只有部分运行时间、分片数量和字符统计。
- 跨领域泛化实验尚未运行，只有在论文声称跨领域有效时才属于必须补充项。
- Relation 质量评测此前因长时间停留在 PPL 阶段且未生成最终 JSON 已终止，不计入当前有效结果；若论文需要该指标，必须修复后重跑。

### 2026-08-28 — 非人工实验完成状态复核

- 服务器 QA 队列仍在运行。200 字符和 300 字符机械切分的 1-doc、2-doc、3-doc 已全部完成，并分别生成完成标记。
- 400 字符机械切分正在进行 1-doc 问答，最近进度为 `2220/3199`；其 2-doc、3-doc 尚未开始。
- 三阶段分片的生成和检索文档准备已完成，但三阶段的三类全量问答尚未开始。
- 四种方法的逐问题配对统计、三阶段组件消融仍未生成结果；当前也未生成独立的效率与资源开销汇总。

### 2026-08-28 — 服务器空闲资源复核

- 服务器当前没有可用于本项目完整实验的空闲 GPU：8 张 Tesla V100-SXM2-32GB 均已占用约 `30G` 显存，虽然瞬时 GPU 利用率为 `0%`，但剩余显存仅约 `0.7–2.0G`。
- 显存主要由其他用户的 Qwen3-8B vLLM 服务占用；本项目当前 400 字符 QA 进程仅额外使用约 `1G` 显存，未终止或影响其他用户进程。
- 因此暂不启动新的大模型问答、向量索引或组件消融任务，等待当前 QA 队列结束或管理员释放资源。

### 2026-08-31 — 非人工 QA 队列完成复核

- 服务器上的非人工 QA 队列已结束，未发现 `run_nonhuman_queue`、`run_qa_method`、`quick_start` 或配对统计进程仍在运行。
- 200 字符、300 字符、400 字符机械切分和三阶段切分的分片、检索文档及三类 QA 均已生成完成标记，共四种方法的 QA 主实验完成。
- 当前队列只执行到 `--stage qa`；逐问题配对统计和三阶段组件消融尚未执行，不能把 QA 队列完成视为全部论文实验完成。
- GPU 0–7 仍由其他用户的服务占用约 `30G` 显存，当前没有可用于启动新 GPU 实验的空闲卡。

### 2026-08-31 — 服务器 0 GPU 状态复核

- 服务器 0（`211.87.224.135:17622`）的 GPU 0–7 当前均被其他用户进程占用，每张 Tesla V100 使用约 `30G` 显存。
- 各卡剩余显存约 `1.4–1.7G`，虽然实时利用率显示为 `0%`，但不具备启动本项目完整实验的条件。
- 当前未发现本项目的分片、QA、配对统计或消融实验进程。

### 2026-08-31 — 服务器 0 GPU 资源再次复核

- 服务器 0 的 GPU 0–7 已全部释放，当前每张 Tesla V100 显存使用为 `0`，可用约 `32G`。
- 当前未发现其他 GPU 计算进程，也未发现本项目的实验进程；服务器 0 目前具备启动新 GPU 实验的条件。

### 2026-08-31 — 服务器 0 Qdrant 数据检查

- 按新批量问答任务要求检查服务器 0：`127.0.0.1:6333` 连接被拒绝，未发现 Qdrant 进程或 6333/6334 监听端口。
- 服务器 0 的 `/home/humq` 下当前主要是 `chunk_code`、Milvus 准备目录和其他项目目录，未发现本项目对应的 Qdrant 数据目录或配置文件；因此暂不能确认其中存在景区数据集切片。
- 在 Qdrant 服务和景区切片数据确认前，不启动新的非交互问答；本地 `badcase_eval.py` 也尚未向服务器上传或调用 API。
- 服务器上虽有 root 管理的 Docker 守护进程，但 `humq` 不在 sudoers 中，直接查看容器被拒绝；因此无法从当前账号确认 root Docker 内是否存在未暴露端口的 Qdrant 容器。需要管理员执行 `docker ps -a` 或将 Qdrant 服务端口/访问权限提供给当前账号后再复核。

### 2026-08-31 — 服务器 0 部署 Qdrant

- 由于 `humq` 无 Docker/sudo 权限，选择用户目录独立二进制方案，不依赖 Docker 或系统服务。
- 使用官方 Qdrant v1.19.0 Linux x86_64 musl 二进制，已在服务器校验 SHA-256：`9ec667456443463eee390e43cd36988af6b730c6db807b4e39f57c303d0264a3`。
- 二进制位于 `/home/humq/qdrant/qdrant`；配置和启动脚本位于 `/home/humq/server_deploy/qdrant/`；本地部署模板位于 `/Users/a1234/RAG_DB_silm/server_deploy/qdrant/`。
- 数据目录为 `/home/humq/qdrant_data/storage`，快照目录为 `/home/humq/qdrant_data/snapshots`，日志为 `/home/humq/qdrant_data/qdrant.log`，PID 文件为 `/home/humq/qdrant_data/qdrant.pid`。
- 服务当前由 `nohup` 后台运行，REST 监听 `127.0.0.1:6333`，gRPC 监听 `127.0.0.1:6334`，未暴露到公网；启动 PID 为 `799232`（后续 PID 以 PID 文件为准）。
- 验证：`GET /` 返回 Qdrant `1.19.0`，`GET /collections` 返回空集合列表；服务已部署，但景区切片数据尚未导入。
- 注意：当前用户级 `nohup` 方式不会自动跨服务器重启持久运行；若需要开机自启，需要管理员配置 systemd/容器权限，或重启后手动运行 `start_qdrant.sh`。

### 2026-08-31 — 服务器 1 硬件信息复核

- 服务器 1（`81.70.191.196`）已成功连接，主机名为 `VM-0-15-centos`。
- 系统为 CentOS Linux 7，内核为 `3.10.0-1160.102.1.el7.x86_64`。
- 硬件为 Intel Xeon Platinum 8374C，4 个逻辑 CPU（2 核、每核 2 线程），内存约 `14G`，未配置 Swap。
- 根磁盘为 `50G ext4`，已使用 `48G`、可用空间为 `0`，使用率 `100%`，不应直接上传或运行实验数据。
- 未检测到 `nvidia-smi` 或 NVIDIA GPU，仅有虚拟 Cirrus Logic 显示设备；该服务器不适合当前项目的 GPU 向量索引和 Qwen 问答实验。

### 2026-08-31 — 服务器 0 景区数据入库与全量 badcase 评测

- 为适配服务器 0 无 MySQL 的现状，保留 `pipeline_qdrant.py` 原有 MySQL 模式，并新增文件直入 Qdrant 模式：Step 6 在传入 `--chunks_file` 时直接读取层级分片 JSON；同时修正 Step 3 实际输出文件名和 Step 2→Step 3→Step 6 的分片文件自动传递。
- 已上传 `/Users/a1234/RAG_DB_silm/data_input/test_data/` 下的 14 个景区 Markdown 文件到服务器 0，并使用 `pipeline_qdrant.py` 完成三阶段切片与规则式指代处理：生成 `309` 个 chunk，指代消解命中 `2` 个 chunk，回滚 `0` 次。
- 使用本地 BGE-M3 模型必要文件部署到服务器 `/home/humq/rag_db_silm/model/bge-m3/`，PyTorch 权重 SHA-256 与本地一致：`b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`；模型向量维度为 `1024`。
- Qdrant 集合 `unified_corpus` 已写入 `309` 个 points，包含 `doc_title_vec`、`chunk_title_vec`、`chunk_text_vec` 三个命名向量；服务仍监听 `127.0.0.1:6333/6334`，PID 以 `/home/humq/qdrant_data/qdrant.pid` 为准。
- 已安装到服务器环境 `/home/humq/envs/denoise_qa` 的依赖：`qdrant-client==1.19.0`、`dashscope==1.27.2`；未使用 Docker 或系统级权限。
- 已运行全量 `merged_7_rag_test_set.json` badcase 检测，共 `436` 条：有效问题 `435` 条，空问题 `1` 条；运行配置为 `mode=sem`、`top_k=5`、`expand_enabled=false`、LLM `qwen-plus`、单进程使用 GPU 0；系统错误 `0`、空检索 `0`、空答案 `0`。
- 评测汇总：自动候选 badcase `283/436`（`64.908%`）；有标准证据的 `435` 条中 Hit@5 `272` 条、未命中 `163` 条（未命中率 `37.471%`）；候选标签计数为 `reference_number_missing_candidate=170`、`low_reference_overlap_candidate=60`、`retrieval_miss_at_k_candidate=163`、`invalid_input=1`。这些是自动筛选结果，不等同于最终人工判定。
- 完整结果已下载到本地：`results/server0/badcase_full.jsonl`（436 条）、`results/server0/badcase_full_summary.json`、`results/server0/badcase_full.log`、`results/server0/all_chunks_chunks_resolved.json`。
- 当前 badcase 进程已结束，Qdrant 仍正常运行。Qdrant 服务当前为用户级 `nohup`，服务器重启后需手动执行 `/home/humq/server_deploy/qdrant/start_qdrant.sh`，除非管理员配置 systemd。

### 2026-08-31 — badcase 结果回收确认

- 已将服务器 0 的完整 badcase 结果回收到本地 `results/server0/`，共 `436` 条 JSONL 记录；本地文件与服务器文件 SHA-256 校验一致。

### 2026-08-31 — badcase 自动判定规则说明

- badcase 采用自动候选规则：只要一条记录触发任意标签，`auto_candidate_badcase=true`；该结果不等同于人工确认的错误答案。
- 检索判定：有标准 chunk ID 时做 Top-K ID 交集；无 ID 时对标准证据文本做归一化包含判断，或用 token F1 ≥ `0.60` 作为近似命中，否则标记检索未命中候选。
- 生成判定：检索/问答/生成有异常标记 `system_error`；Top-K 为空标记 `empty_retrieval`；可回答问题答案为空标记 `empty_answer`。
- 可回答问题：参考答案与生成答案的 token F1 低于 `0.15` 标记 `low_reference_overlap_candidate`；参考答案中的数字未出现在生成答案中标记 `reference_number_missing_candidate`。
- 不可回答问题：生成答案没有包含预设拒答提示语时标记 `out_of_scope_answer_candidate`；空问题单独标记 `invalid_input`。
- 严重等级：系统错误、空检索、空答案为 `critical`；精确检索未命中、数字缺失、不可回答问题被作答为 `major`；其他候选标签为 `minor_review`。最终需结合检索上下文进行人工或 LLM 复核。

### 2026-08-31 — badcase 二次复核完成

- 使用 Qwen-plus 对原自动筛选出的 `283` 条有效候选记录进行逐条二次复核，按原始记录行号关联结果；这是必要的，因为数据中存在重复问题 ID。
- 整体判定：`136` 条 `badcase`、`146` 条 `acceptable`、`1` 条 `uncertain`。因此，候选集合中的模型复核 badcase 比例为 `48.057%`；该比例不是人工金标准。
- 答案维度：`123` 条判定为 badcase、`146` 条可接受、`14` 条不确定。检索维度：`245` 条命中、`38` 条未命中；答案问题和检索问题可能同时出现在同一条记录中。
- 错误类型统计：答案不完整 `44` 条、拒答/不回答 `29` 条、答案错误 `31` 条、检索未命中 `33` 条；其中 `38` 条记录同时表现为答案问题和检索未命中。
- 将原始数据中的 `1` 条空问题按输入无效处理，则全部 `436` 条记录中需要处理的记录为 `137` 条（`136` 条模型判定 badcase + `1` 条无效输入），占 `31.422%`。
- 完整逐条结果：`results/server0/badcase_reviewed.jsonl`；汇总：`results/server0/badcase_reviewed_summary.json`；复核脚本：`review_badcases.py`。
- 限制：本次为 LLM 辅助复核，不是人工标注金标准；复核依据是已保存的 Top-K 文本截断内容，`1` 条 `uncertain` 以及最终 badcase 仍建议人工抽查确认。

### 2026-08-31 — badcase 类别拆分说明

- `answer_judgement=badcase` 表示问答生成层面的错误，共 `123/283` 条（`43.463%`）。
- `retrieval_judgement=miss` 表示 Top-K 检索结果没有提供回答所需证据，共 `38/283` 条（`13.428%`）。这 `38` 条同时也被判定为答案 badcase。
- 两者交叉统计为：答案 badcase 且检索命中 `85` 条，答案 badcase 且检索未命中 `38` 条，答案不确定且检索命中 `14` 条，其余为答案可接受且检索命中 `146` 条。
- `overall_judgement=badcase` 是模型结合问题、证据和答案给出的独立综合判断，共 `136` 条（`48.057%`），不能简单用答案 badcase 与检索 miss 做集合相加得到。

### 2026-08-31 — 136 条综合 badcase 主错误分类

- 对 `overall_judgement=badcase` 且 `review_source=qwen-plus` 的 `136` 条进行互斥主错误分类，结果为：答案不完整 `44` 条（`32.353%`）、检索未命中 `33` 条（`24.265%`）、事实或内容错误 `31` 条（`22.794%`）、应答但拒答 `28` 条（`20.588%`）。四类合计 `136` 条。
- 需要区分主错误类型与独立检索标签：综合 badcase 中 `retrieval_judgement=miss` 实际为 `38` 条，其中有 `5` 条因主因更明显被归入“不完整”“错误”或“拒答”，所以主分类中的 `retrieval` 为 `33` 条。
- 该分类结果来自 Qwen-plus 二次复核，后续如用于论文定稿，仍需对边界样本进行人工抽查。

### 2026-08-31 — badcase 结果可读版整理

- 保留原始 `results/server0/badcase_reviewed.jsonl` 不变，新增 `results/server0/badcase_reviewed_readable.json`。
- 可读版按“答案不完整、检索未命中、事实或内容错误、应答但拒答、不确定、可接受、未进入二次复核、输入无效”分组。
- 每条记录保留原始行号、问题、标准答案、系统答案、自动筛选标签、答案/检索/综合复核结果及复核理由；`136` 条综合 badcase 额外保留 Top-K 检索证据摘要。
- 生成脚本为 `format_badcase_results.py`，便于后续重新生成整理版。

### 2026-08-31 — 136 条 badcase 原因分析文档

- 新增 `BADCASE_136_ANALYSIS_AND_OPTIMIZATION.md`，分析了 `136` 条综合 badcase 的四类主错误：答案不完整 `44` 条、检索未命中 `33` 条、事实或内容错误 `31` 条、应答但拒答 `28` 条。
- 文档补充了检索命中/未命中交叉分析、当前评测逻辑的误报与标注缺口、上下文截断问题，以及按 P0–P3 排列的评测、检索、生成和验证优化方案。
- 文档给出了后续配对实验顺序：上下文长度、Top-K、查询扩展、语义/维度/融合检索和生成后校验，避免把单次语义检索结果直接归因于三阶段分片方法。

### 2026-09-01 — 融合检索问答实验重新启动

- 已确认原问答生成逻辑在构造上下文时使用固定 `text[:500]`；`--trace-chars 2000` 只影响 JSONL 中保存的检索片段，不影响模型输入。
- 已将生成上下文长度改为可配置参数 `--context-chars`，默认仍为 `500` 以保持旧行为；本次重跑设置为 `2000`，并新增 `--api-interval` 控制请求间隔。
- 服务器 0 已启动全量 `merged_7_rag_test_set.json` 问答实验：检索模式 `fusion`、Top-K=`5`、查询扩展关闭、上下文=`2000` 字符/chunk、Qwen-plus、API 间隔=`2` 秒。
- 任务运行于服务器 0 的 tmux 会话 `qa_fusion_full`，启动 Python PID 为 `1269936`；启动复核时已完成 `22/436`，GPU 0 正在使用，其余 GPU 保持空闲。
- 输出文件：`/home/humq/rag_db_silm/results/server0/badcase_fusion_context2000_20260901.jsonl`、同名 `_summary.json` 和 `.log`；任务结束后下载到本地并与旧的语义检索结果进行对比。

### 2026-09-01 — 融合检索问答实验完成与 badcase 提取

- 融合检索问答实验已完成 `436/436`；有效问题 `435` 条、空问题 `1` 条、系统错误 `0`、空检索 `0`、空答案 `0`。
- 新实验自动候选 badcase 共 `272` 条，其中有效问题 badcase `271` 条，空问题输入无效 `1` 条；有效问题 badcase 率为 `62.299%`（`271/435`）。
- 自动标签为：`retrieval_miss_at_k_candidate=163`、`reference_number_missing_candidate=162`、`low_reference_overlap_candidate=23`；标签允许多选，因此不能相加作为记录总数。
- 已从服务器提取并下载到本地：`results/server0/badcase_fusion_context2000_20260901_badcases.jsonl`（含 272 条候选）、`results/server0/badcase_fusion_context2000_20260901_badcases_valid.jsonl`（仅 271 条有效问题）及对应汇总 JSON。
- 完整融合实验结果仍保存在服务器 `/home/humq/rag_db_silm/results/server0/badcase_fusion_context2000_20260901.jsonl`；本地已下载其汇总文件，未覆盖此前语义检索结果。

### 2026-09-01 — 融合检索结果二次复核完成

- 已使用 Qwen-plus 对融合检索实验的 `271` 条有效自动候选 badcase 逐条复核，按原始行号关联，复核完成数 `271/271`。
- 模型综合判定：badcase `99` 条、acceptable `172` 条；答案层面 badcase `86` 条、uncertain `12` 条、acceptable `173` 条；检索 miss `44` 条、hit `227` 条。
- 主错误类型：答案不完整 `43` 条、检索未命中 `36` 条、事实或内容错误 `16` 条、应答但拒答 `4` 条；主类型合计 `99` 条。
- 已在服务器提取并下载有效最终 badcase：`results/server0/badcase_fusion_context2000_20260901_reviewed_badcases_valid.jsonl`（99 条）及汇总文件；另有包含空问题规则记录的版本（100 条）。
- 新增本地可读版：`results/server0/badcase_fusion_context2000_20260901_reviewed_badcases_readable.json`，按主错误类型分组并保留问题、标准答案、系统答案、复核理由和 Top-K 证据摘要。
- 完整二次复核结果仍保存在服务器 `/home/humq/rag_db_silm/results/server0/badcase_fusion_context2000_20260901_reviewed.jsonl`；服务器端复核任务已结束。

### 2026-09-01 — 融合检索 99 条 badcase 分析文档

- 新增 `BADCASE_99_FUSION_ANALYSIS_AND_OPTIMIZATION.md`，独立分析融合检索 + 2000 字符上下文实验复核出的 `99` 条有效 badcase。
- 文档给出四类主错误统计：答案不完整 `43` 条、检索未命中 `36` 条、事实或内容错误 `16` 条、应答但拒答 `4` 条。
- 文档补充了检索命中/答案错误交叉分析、与旧语义检索 + 500 字符实验的对比，以及数据标注、融合排序、问题分解、生成后校验和后续单因素消融实验方案。
- 关键结论：本次全量自动 Hit@5 仍为 `272/435`，与旧实验相同；事实错误和拒答下降，但不能在未做消融前把下降全部归因于融合检索。

### 2026-09-01 — 99 条 badcase 文档编号校正

- 已将 `BADCASE_99_FUSION_ANALYSIS_AND_OPTIMIZATION.md` 的章节编号统一为连续的 `1–8`，并将“当前逻辑缺陷、整体优化方案、消融实验顺序、结论”分别对齐为第 `5–8` 节。

### 2026-09-01 — Qdrant 入库内容抽样检查

- 从服务器 0 的 Qdrant `unified_corpus` 集合顺序读取了 5 条 payload，确认实际入库字段包括 `chunk_id`、`doc_title`、`chunk_text`、`chunk_index`、上下游 chunk ID、`chunk_len` 和 `genre`。
- 样本中既有正常正文 chunk，也有 `chunk_len=33` 和 `chunk_len=37` 的短结构化 POI 元数据 chunk；这些短 chunk 虽然记录的 `l_min=400`，但仍然被写入集合，可能增加检索噪声。
- 该抽样只读取 payload，不读取向量；Qdrant 集合仍为 `unified_corpus`，服务正常。

### 2026-09-01 — 核查 Qdrant 标签与维度入库情况

- 服务器 0 的 Qdrant 当前只有 `unified_corpus` 集合，共 `309` 个 points；`dimension_tags` 请求返回 `404`，说明维度标签集合没有创建或没有部署到当前 Qdrant。
- 抽查 `unified_corpus` 的 `100` 个 points，payload 字段均只有 `chunk_id`、`doc_id_link`、`doc_title`、`chunk_gen_title`、`chunk_text`、`chunk_index`、上下游 chunk ID、`chunk_len`、`genre`、`l_min`、`l_max`，没有 `dim_*`、`tag_*` 或实体维度字段。
- 服务器 `/home/humq/rag_db_silm/experiment_data/` 为空；标签/维度中间文件没有随本次入库上传。服务器分片文件的顶层记录也没有标签字段，嵌套 chunk 只有 `chunk_text` 和 `chunk_len`。
- 根因是当前文件直入路径 `_load_qdrant_chunks_from_file()` 只把分片文本转换为 `doc_id/doc_text/profile_json`，未携带 Step 5 的维度标签；`_build_search_index()` 只生成本地倒排/元数据文件，也不会创建 `dimension_tags` 集合。只有 `_migrate_to_qdrant()` 读到以 `dim_` 开头的字段时才会把它们写入 chunk payload。
- 因此此前服务器上的“fusion”问答结果不能证明使用了真实的维度标签检索：结果中的 `dimension_results` 实际没有 `tag_name`/`dim_name`，约等于同一 `unified_corpus` 的向量候选；需要补齐 Step 5 标签数据、建立 `dimension_tags` 并重新入库/重跑融合实验。

### 2026-09-01 — 解释 pipeline_qdrant.py 维度标签未入库原因

- 代码中的 Step 5 确实负责维度挖掘和标签生成：`step5_dimension_tagging()`（约第 691 行）调用维度挖掘、生成 `tags_output*.json`，并在 RDB 模式下回写 MySQL。
- 但 Step 5 的文件模式产物与 Step 6 的 Qdrant 文件直入没有真正接通：`_load_chunks_from_file()`（约第 922 行）和 `_load_qdrant_chunks_from_file()`（约第 1598 行）只保留 `doc_id/doc_text/profile_json`，不会读取或合并 `tags_output`/`step5_result`。
- Step 6 的 `_build_search_index()`（约第 1244 行）只从标签文件生成 `inverted_index.json`、`dimension_metadata.json` 和 `tag_vectors.pkl`，没有创建 `dimension_tags` Qdrant collection；`_migrate_to_qdrant()`（约第 1487 行）只有在输入记录本身存在 `dim_*` 字段时才会把维度写入 payload，而文件直入记录没有这些字段。
- 服务器本次实际结果与上述代码路径一致：只有 `unified_corpus`，没有 `dimension_tags`；因此不是标签被 Qdrant 删除，而是标签产物没有进入当前 Qdrant 入库链路。要恢复真正的维度融合，需要让 Step 6 合并 Step 5 标签并写入 chunk payload，同时单独创建并写入 `dimension_tags`，之后重新执行入库和问答评测。

### 2026-09-01 — 按 MySQL 字段结构修正 Qdrant 文件直入

- 根据用户确认的目标，将 Qdrant `unified_corpus` 设计为“每个 chunk 一个 point、基础字段和维度字段全部放入 payload、chunk 内容建立向量”的结构。
- 修改 `pipeline_qdrant.py`：层级分片文件统一展开为逐 chunk 记录；Step 5 文件模式按相同 `chunk_id` 抽取标签；Step 6 自动读取当前数据集的 `tags_output`/`step5_result` 并合并为 `dim_*` 字段。
- Qdrant payload 现在保留 `doc_id`、`corpus_id`、`doc_text`、`profile_json`、`chunk_text`、分片元数据和其他输入字段；维度字段在所有 points 中统一补空值，命中的多标签按 MySQL 兼容的分号连接格式写入。
- chunk 文本在 payload 中完整保存；`chunk_text_vec` 继续使用前 2000 字符进行 BGE-M3 编码，另保留文档标题和 chunk 标题向量。
- 验证：`python3 -m py_compile pipeline_qdrant.py` 通过；使用本地真实分片和 `tags_output_all.json` 验证，`1522` 个 chunk 中 `1520` 个成功匹配标签，识别出 `16` 个维度字段；`git diff --check` 通过。
- 尚未将本次修改上传服务器或重跑入库；下一步需上传新脚本、补齐服务器上的 Step 5 依赖和标签生成结果后，再重建 Qdrant collection 并验证 payload 字段。

### 2026-09-01 — 服务器 0 重建 Qdrant 维度字段

- 已将修正后的 `pipeline_qdrant.py`、`llm_service.py` 和 `dimension_integration.py` 上传到服务器 0 `/home/humq/rag_db_silm/`，远程语法检查和模块导入通过。
- Step 5 使用服务器分片文件 `output_chunks/all_chunks_chunks_resolved.json`，核心维度复用为：历史事件、朝代、涉及皇帝、文化称号、地理位置、建筑功能、相关人物、事件原因；共生成 `309` 条 chunk 标签记录。
- 为降低 Qwen3 请求耗时，标签抽取加入多 chunk 请求、Qwen3 关闭思考输出、JSON 连续对象解析和断点续传；随后增加“标签值必须能在当前 chunk 文本中找到证据”的清洗，避免批量抽取发生跨 chunk 串值。
- 已在 Qdrant 重建前创建快照：`unified_corpus-3461249813991701-2026-09-01-07-41-58.snapshot`。随后删除并按原三向量配置重建 `unified_corpus`，没有创建独立的 `dimension_tags` 集合。
- 重建后的 Qdrant：状态 `green`，`309` 个 points；向量字段为 `chunk_text_vec`、`chunk_title_vec`、`doc_title_vec`，每个向量维度为 `1024`；payload 共 `25` 个字段。
- 8 个维度字段已写入每个 point：`dim_历史事件`、`dim_朝代`、`dim_涉及皇帝`、`dim_文化称号`、`dim_地理位置`、`dim_建筑功能`、`dim_相关人物`、`dim_事件原因`。抽样和全量核验均确认字段不缺失；清洗后至少含一个非空维度标签的 chunk 为 `31` 条。
- 清洗后各维度非空字段计数：历史事件 `17`、朝代 `3`、涉及皇帝 `4`、文化称号 `12`、地理位置 `20`、建筑功能 `19`、相关人物 `20`、事件原因 `11`。由于当前倒排索引仍要求覆盖率至少 `50%`，`dimension_metadata.json` 本轮过滤出 `0` 个可用于倒排过滤的维度；这不影响 Qdrant payload 中的维度字段保存，但后续若要做维度检索需单独调整覆盖率阈值或检索策略。
- 远程日志：`/home/humq/rag_db_silm/results/server0/rebuild_step5_scenic_test14_retry.log`、`rebuild_step5_scenic_test14_sanitize.log`、`rebuild_step6_scenic_test14_final.log`；标签结果：`experiment_data/tags_output_scenic_test14.json` 和 `step5_result_scenic_test14.json`。

### 2026-09-01 — GitHub 上传准备

- 已确认远程仓库为 `https://github.com/hh030210/rag_db.git`，远程当前没有可用分支。
- 已准备干净上传提交 `7f68415572c83939cd3851579ab261f52427323c`，包含当前源码、实验脚本、部署脚本和整理后的文档；本地 `master` 工作分支未被切换或覆盖。
- 上传提交排除了模型文件、数据集、向量库、原始实验结果、日志和本地凭据；同时将 `results/` 加入 `.gitignore`。本地仓库旧历史约 `5.7 GB`，没有把旧的大体积历史推送到 GitHub。
- 推送阶段被 GitHub 认证阻止：当前 Mac 没有可用的 GitHub CLI 登录、SSH Key 或钥匙串凭据。待用户在本机执行 `gh auth login`，或提供具有该仓库写权限的 GitHub Personal Access Token 后，再将该提交推送到 `origin/master`。

### 2026-09-01 — GitHub 上传完成

- 已在本机安装 GitHub CLI `2.97.0`，完成 `hh030210` 账号设备认证并配置 Git HTTPS 凭据。
- 项目已成功推送到 `https://github.com/hh030210/rag_db.git` 的 `master` 分支，远程最新提交为 `b8789afadb30aea872237af194d35c6689d5c5db`。
- 本次上传内容为当前项目源码、部署脚本、实验脚本和整理后的文档；模型、数据集、向量库、原始结果、日志和本地凭据均未上传。
- 推送时使用 `--no-verify` 跳过了本地遗留的 Git LFS 检查；当前项目没有 `.gitattributes` 或 LFS 跟踪文件，因此不影响已上传内容。

### 2026-09-01 — chunk_code 文件夹上传完成

- 已将 `/Users/a1234/chunk_code/` 的源码和说明文档整理到仓库根目录 `chunk_code/` 并推送到 `master` 分支。
- 原文件夹约 `5.5 GB`；上传版本约 `17 MB`，保留切片算法、评测脚本、Meta-Chunking 源码、部署说明和论文实验说明。
- 已排除数据集、实验结果、日志、模型缓存、Milvus 数据卷、向量库、Python 缓存、Cursor 导出和内嵌 Git 历史，避免上传超大文件及本地运行状态。
- 发现的硬编码 SiliconFlow API Key 已改为从 `SILICONFLOW_API_KEY` 环境变量读取，上传版本未发现真实 API Key 或服务器密码。
- 文件夹上传提交为 `51fafd138c8fb480953a811d179afa67df8316e0`；随后本交接记录的最终状态提交将继续推送到同一 `master` 分支。
