# 项目全部实验结果汇总

更新时间：2026-08-17

本文中的公式采用 Word 可直接粘贴的线性文本形式；表格采用 `+` 作为列分隔符。复制表格内容到 Word 后，使用“插入 → 表格 → 将文字转换成表格”，分隔符选择“其他字符：`+`”即可转换为正常表格。

## 1. 项目与实验范围

项目围绕中文新闻语料上的语义感知分片和 RAG 问答展开，主要包含两条实验主线：

1. 比较固定长度、Meta-Chunking 和 Integrated Chunks 对端到端 RAG 问答的影响。
2. 在三阶段分片中隔离第二阶段去噪开关，比较去噪前后分片结构、检索证据和问答效果。

本次新增的独立质量实验，进一步比较 200/300/400 字符机械切分和三阶段切分的：

- 语义困惑度
- 主题距离度
- 信息差异含量
- 内容一致性

## 2. 数据集与实验版本

项目+内容
主语料+data/db_qa.txt，约 10,451 行中文新闻
QA 数据+data/split_merged.json，1-doc/2-doc/3-doc 三类问答
当前四方法评测输入+/home/humq/chunk_code/data/db_qa.txt
Embedding+BAAI/bge-base-zh-v1.5
旧版端到端 QA 模型+Qwen/Qwen2.5-7B-Instruct
去噪消融 QA 模型+Qwen/Qwen3-8B
四维 PPL 模型+本地 Qwen2-1.5B-Instruct

旧版 Meta/Integrated QA 与去噪开关 QA 不是同一轮实验：前者使用旧版分片和 Qwen2.5-7B，后者使用 `db_qa.txt` 的去噪开关结果和 Qwen3-8B，不能直接合并成一个百分比结论。

## 3. 分片规模统计

### 3.1 历史分片版本

分片输出+Chunk 数+总字符数+平均长度+说明
output_baseline_v1+16,022+8,291,143+517.5+固定长度 Baseline v1
output_chunks+16,401+8,290,966+505.5+早期 Integrated 输出
data/db_qa_chunks+15,929+7,559,532+474.6+旧报告中的 Integrated 版本
output_enhanced_v2+8,034+4,148,416+516.4+去重增强版，需单独验证误删

`output_enhanced_v2` 相比其摘要记录的去重前 15,151 个 Chunk 减少约 47.0%，幅度明显高于设计预期，不能与后续 MAD 去噪结果混为一谈。

### 3.2 当前四方法评测分片

方法+Chunk 数+平均 Chunk 长度（约）
200 字符机械切分+46,505+177.3
300 字符机械切分+32,828+252.8
400 字符机械切分+26,060+318.4
三阶段切分+16,980+478.5

三阶段分片使用 `line_mode`，第二阶段默认启用 MAD/PPL 去噪；当前结果目录为 `/home/humq/chunk_code/results/four_method_chunks_20260814/`。

## 4. 主实验：Integrated Chunks 与 Meta-Chunking v2

实验条件：top-k=8，BGE embedding，Qwen2.5-7B-Instruct，数据集为 `split_merged.json`。该实验是当前旧版 RAG 主线中最完整的二方比较。

任务+指标+Meta-Chunking v2+Integrated Chunks+绝对变化+相对变化
1Doc+ROUGE-L+0.3645+0.4264+正0.0619+正17.0%
1Doc+BERTScore+0.7198+0.8152+正0.0954+正13.3%
1Doc+有效样本数+3,121+3,151+正30+—
2Docs+ROUGE-L+0.2733+0.3032+正0.0299+正10.9%
2Docs+BERTScore+0.8162+0.8608+正0.0446+正5.5%
2Docs+有效样本数+3,094+3,132+正38+—
3Docs+ROUGE-L+0.2655+0.2950+正0.0295+正11.1%
3Docs+BERTScore+0.8323+0.8700+正0.0377+正4.5%
3Docs+有效样本数+3,102+3,146+正44+—

结论：Integrated Chunks 在三个任务上的 ROUGE-L 和 BERTScore 均高于 Meta-Chunking v2。Baseline v1 只完成了建索引，没有完成同条件全量 QA，因此尚未形成 Baseline/Meta/Integrated 三方闭环比较。
## 5. 去噪开关消融实验

实验只改变三阶段分片第二阶段的去噪开关：

- `denoise_off`：关闭去噪
- `denoise_on`：开启 MAD/PPL 去噪

### 5.1 分片变化

条件+Round 2 输入句子+删除句子+删除字符+最终 Chunk 数+最终字符数
off+227,933+0+0+17,342+8,283,275
on+227,933+5,946+158,312+16,980+8,125,369

去噪组删除约 2.61% 的 Round 2 输入字符，最终 Chunk 数减少约 2.09%。

### 5.2 全量 QA 结果

任务+条件+有效结果数+BLEU-avg+ROUGE-L
1-doc+off+3,199+0.281+0.464
1-doc+on+3,198+0.302+0.485
2-doc+off+3,189+0.114+0.251
2-doc+on+3,189+0.125+0.278
3-doc+off+3,188+0.102+0.255
3-doc+on+3,188+0.118+0.273

`on - off` 差值：

任务+BLEU-avg+ROUGE-L
1-doc+正0.021+正0.021
2-doc+正0.011+正0.027
3-doc+正0.016+正0.018

按当前采用版本，三类任务的 BLEU-avg 和 ROUGE-L 均有小幅提升；`bertScore` 当前为 0，不能作为有效比较指标。由于尚未完成逐问题配对统计和显著性检验，暂不能据此断言提升具有统计稳定性。


## 6. 四种分片方法的四维质量评测

输入为同一份 `db_qa.txt`，比较 200/300/400 字符机械切分和三阶段切分。PPL 使用本地 Qwen2-1.5B-Instruct，主题空间使用共享 TF-IDF+LSA，内容向量使用 BGE。

### 6.1 主指标

#### 6.1.1 语义困惑度

对每个 Chunk 使用固定的本地 `Qwen2-1.5B-Instruct` 因果语言模型计算 token 级交叉熵，并按有效 token 数汇总：

Word 线性公式：

L_PPL = − Σ(b,t)[m(b,t) × ln pθ(x(b,t+1) | x(b,≤t))] ÷ Σ(b,t)m(b,t)

PPL = exp(L_PPL)

在公式中，`b` 是第 `b` 个 Chunk，`t` 是当前 Chunk 内的 token 预测位置，`x_(b,t+1)` 是该位置对应的真实下一个 token。`p_theta(...)` 是模型预测该真实 token 的概率，`theta` 表示模型参数；`m_(b,t)` 是有效 token 掩码，有效位置取 1、padding 位置取 0。`L_PPL` 是所有有效 token 的平均负对数似然，也就是平均交叉熵；`exp(...)` 表示以自然常数为底的指数函数。

预测标签相对输入右移一位。每个 Chunk 最长输入 1,024 tokens，最终是所有有效 token 的加权平均，而不是各 Chunk PPL 的简单平均。PPL 越低表示文本对语言模型越容易预测，语义连贯性通常越好。

#### 6.1.2 主题距离度

先将 Chunk 和 Chunk 内句子映射到同一个 TF-IDF+LSA 主题空间。对 Chunk `C` 中的句子 `s_i`，以句子字符数为权重：

Word 线性公式：

w_i = len(s_i) ÷ Σk len(s_k)

c_C = Σi[w_i × z(s_i)]

这里，`C` 表示一个 Chunk，`s_i` 表示 Chunk `C` 中的第 `i` 个句子，`len(s_i)` 是该句子的字符数。`w_i` 是句子长度权重，所有句子的 `w_i` 之和为 1；`z(s_i)` 是句子的 LSA 主题向量；`c_C` 是 Chunk `C` 的加权主题中心向量。Chunk 内主题离散度为：

Word 线性公式：

D_in(C) = Σi[w_i × (1 − cos(z(s_i), c_C))]

其中，`D_in(C)` 是 Chunk `C` 的主题内离散度，`cos(a,b)` 是向量 `a` 和 `b` 的余弦相似度，因此 `1 - cos(...)` 就是主题距离，数值越大表示越不相似。

对所有含至少两个句子的 Chunk 求平均，得到 `intra_topic_dispersion_mean`。相邻 Chunk 的主题边界距离为：

Word 线性公式：

D_boundary = [Σ(j=1..n−1)(1 − cos(z(C_j), z(C_(j+1))))] ÷ (n − 1)

这里，`n` 是全部 Chunk 的数量，`C_j` 是第 `j` 个 Chunk，`C_(j+1)` 是它的相邻后继 Chunk；`D_boundary` 是所有相邻 Chunk 边界距离的平均值。

报告中的主题对比度定义为：

Word 线性公式：

D_contrast = D_boundary − mean(D_in)

其中，`mean(D_in)` 是所有符合条件 Chunk 的主题内离散度平均值，`D_contrast` 是主题对比度；边界越远且 Chunk 内部越集中时，`D_contrast` 越高。

主题内离散度越低、边界距离和主题对比度越高，说明 Chunk 内部越集中、相邻 Chunk 之间的主题边界越清晰。

#### 6.1.3 信息差异含量

对每个 Chunk 的字符级 TF-IDF 向量 `x_j`，先计算单位字符的信息密度：

Word 线性公式：

rho_j = Σk[x(j,k)] ÷ max(len(C_j), 1)

这里，`C_j` 是第 `j` 个 Chunk，`x_j` 是 Chunk `C_j` 的字符级 TF-IDF 向量，`x_(j,k)` 是该向量的第 `k` 个 TF-IDF 特征值，`rho_j` 是 Chunk `C_j` 的单位字符 TF-IDF 信息密度。

再使用相邻 Chunk 的 TF-IDF 余弦相似度计算邻域新颖度。设 `q_j` 为 Chunk 与可用相邻 Chunk 的平均余弦相似度，则：

Word 线性公式：

N_j = 1 − q_j

R_j = 1 − N_j

其中，`q_j` 是 Chunk `C_j` 与可用相邻 Chunk 的平均 TF-IDF 余弦相似度，`N_j` 是邻域新颖度，与邻居越不相似时越高；`R_j` 是邻域冗余度，与邻居越相似时越高。首尾 Chunk 只使用实际存在的一个邻居；如果只有一个 Chunk，代码将其新颖度设为 1。最终的信息差异含量为信息密度与邻域新颖度的乘积：

Word 线性公式：

IDC_j = rho_j × N_j

IDC = Σ(j=1..n)IDC_j ÷ n

其中，`IDC_j` 是第 `j` 个 Chunk 的信息差异含量，`IDC` 是全部 Chunk 的平均信息差异含量，`n` 是 Chunk 总数。

`information_difference_content_mean` 越高，表示 Chunk 同时包含更多 TF-IDF 信息，并且与相邻 Chunk 的重复越少；邻域冗余度越低越好。

#### 6.1.4 内容一致性

使用同一个 BGE Embedding 模型，将原文行、Chunk 和 Chunk 内句子编码为向量，并按字符数加权计算中心向量。切分前后的外部内容一致性为：

Word 线性公式：

v_before = Σi[(len(u_i) ÷ Σk len(u_k)) × Emb(u_i)]

v_after = Σj[(len(C_j) ÷ Σk len(C_k)) × Emb(C_j)]

这里，`u_i` 是切分前原始文档中的第 `i` 个非空文本行，`C_j` 是切分后得到的第 `j` 个 Chunk，`Emb(x)` 是 BGE Embedding 模型生成的文本向量。`v_before` 是原始文本的长度加权中心向量，`v_after` 是 Chunk 文本的长度加权中心向量。

Word 线性公式：

S_external = cos(v_before, v_after)

这里的 `S_external` 是切分前后外部内容一致性，数值越高表示整体内容越接近；`cos(...)` 仍表示两个中心向量的余弦相似度。

对每个至少包含两个句子的 Chunk，计算句子向量的长度加权中心 `c_C`，其内部一致性为：

Word 线性公式：

S_internal(C) = Σi[w_i × cos(Emb(s_i), c_C)]

这里，`s_i` 是 Chunk `C` 内的第 `i` 个句子，`w_i` 是句子字符数权重，`c_C` 是 Chunk `C` 内句子 Embedding 的长度加权中心，`S_internal(C)` 是单个 Chunk 的内部一致性。所有符合条件 Chunk 的 `S_internal(C)` 平均值就是 `internal_consistency`，有效 Chunk 数除以 Chunk 总数就是 `internal_coverage`。原始综合内容一致性为：

Word 线性公式：

S_combined = 0.5 × S_external + 0.5 × S_internal

其中，`S_combined` 是原始综合内容一致性，两个 `0.5` 表示外部一致性和内部一致性各占 50%。

为消除 Chunk 长度差异带来的偏置，长度均一化版本把原文和 Chunk 切成统一 200 字符窗口；内部一致性则把每个 Chunk 的句子按约 200 字符预算组成局部窗口，仅对包含至少两个句子的窗口计算。归一化外部一致性仍使用窗口长度加权中心，归一化内部一致性按窗口字符数加权平均，覆盖率定义为有效窗口数除以窗口总数：

Word 线性公式：

S_combined_norm = 0.5 × S_external_norm + 0.5 × S_internal_norm

其中，`S_external_norm` 是将原文和 Chunk 切成统一 200 字符窗口后得到的外部一致性，`S_internal_norm` 是将句子按约 200 字符预算组成局部窗口后得到的内部一致性，`S_combined_norm` 是长度均一化后的综合内容一致性；有效窗口数除以窗口总数得到 `normalized_coverage`。

原始指标反映完整 Chunk 的整体一致性，长度均一化指标反映相同局部尺度下的一致性。PPL 越低越好；主题对比度、信息差异含量和内容一致性越高越好。

方法+PPL（越低越好）+主题对比度（越高越好）+信息差异含量（越高越好）+原始内容一致性综合（越高越好）+长度均一化内容一致性综合（越高越好）
200 字符机械切分+18.483+0.210+0.060+0.917+0.927
300 字符机械切分+15.909+0.239+0.053+0.915+0.936
400 字符机械切分+14.598+0.266+0.048+0.914+0.936
三阶段切分+13.061+0.348+0.035+0.912+0.938

三阶段在 PPL、主题边界和主题对比度方面最好；长度均一化后，三阶段的综合一致性最高。
综合一致性为 `0.5 × 归一化外部一致性 + 0.5 × 归一化内部一致性`。

方法+归一化外部一致性+归一化内部一致性+归一化内部覆盖率+归一化综合一致性
200 字符机械切分+1.000+0.853+0.949+0.927
300 字符机械切分+0.999+0.873+0.874+0.936
400 字符机械切分+1.000+0.872+0.775+0.936
三阶段切分+1.000+0.877+0.813+0.938

长度归一化后，三阶段综合一致性高于三种机械切分，说明机械切分原始优势主要受到短 Chunk 长度效应影响。原始指标反映完整 Chunk 的整体一致性，归一化指标反映相同局部尺度下的一致性。

### 6.2 四个维度综合实验结果

以下表格只保留四个维度各自的综合指标；内容一致性采用 200 字符长度均一化后的综合值，以避免不同 Chunk 长度造成比较偏差。

方法+语义困惑度 PPL+主题对比度+信息差异含量+长度均一化综合内容一致性
200 字符机械切分+18.483+0.210+0.060+0.927
300 字符机械切分+15.909+0.239+0.053+0.936
400 字符机械切分+14.598+0.266+0.048+0.936
三阶段切分+13.061+0.348+0.035+0.938
