# 语义检索、维度检索与融合策略

本文统一描述三部分内容：

1. 语义检索：基于 query 与文档文本向量相似度的召回路径。
2. 维度检索：基于维度、标签和维度标签向量的召回与精排路径。
3. 双路融合策略：只考虑语义检索结果与维度检索结果之间的最终融合。

注意：本文的融合策略先忽略维度检索内部的检索融合，也就是暂不讨论维度检索内部混入的那次语义检索，不引入内部标签权重 $\lambda_q$。这里将维度检索视为已经产出 `dim_results` 的独立路径，再与语义检索产出的 `sem_results` 做最终融合。

本文统一用 $\mathbf{v}_x$ 表示对象 $x$ 的向量化结果，不再单独引入编码函数。例如，$\mathbf{v}_q$ 表示 query 向量，$\mathbf{v}_D$ 表示文档或 chunk 向量，$\mathbf{v}_{m,t}$ 表示维度标签 $(m,t)$ 的向量，$\mathbf{v}_{t_q}$ 与 $\mathbf{v}_{t_D}$ 分别表示 query 标签值和文档标签值的向量。

## 1. 语义检索

语义检索将 query 和文档或 chunk 表示到同一个向量空间中，通过向量相似度衡量二者在语义上的接近程度。

语义相似度得分为：

$$
S_{sem}(q,D)=sim(\mathbf{v}_q,\mathbf{v}_D)
$$

纯语义检索候选集合为：

$$
\mathcal{D}_{sem}=TopK_D(S_{sem}(q,D))
$$

## 2. 维度检索

设 query 解析出的结构化维度-标签约束为 $C_q$，文档 $D$ 自身的维度标签集合为 $T_D$。其中 $Q_m=C_q[m]$ 表示 query 在维度 $m$ 下的标签值集合，$T_D[m]$ 表示文档在维度 $m$ 下的标签值集合。

### 2.1 第一阶段：初步召回

#### 2.1.1 方法一：query 向量-D 向量

直接使用 query 向量与文档或 chunk 向量的相似度召回候选集合：

$$
\mathcal{D}_{cand}=TopK_D(sim(\mathbf{v}_q,\mathbf{v}_D))
$$

#### 2.1.2 方法二：query 维度-D 维度

将 query 解析为结构化维度-标签约束：

$$
C_q=\{m_i:[t_{i1},t_{i2},...]\}
$$

query 涉及的维度集合为：

$$
M_q=\{m\mid m\in C_q\}
$$

候选文档集合为：

$$
\mathcal{D}_{cand}=\{D \mid \exists m\in M_q,\ T_D[m]\neq\varnothing\}
$$

也就是说，只要文档拥有 query 中要求的维度，就先进入候选集合。该阶段只检查维度是否存在，不强制要求标签值完全匹配。

#### 2.1.3 方法三：query 向量-维度标签向量

计算 query 向量与每个维度标签向量的相似度，召回得分最高的若干维度标签：

$$
L_q=TopK_{(m,t)}(sim(\mathbf{v}_q,\mathbf{v}_{m,t}))
$$

得到召回标签 $L_q$ 后，直接筛选文档自身标签集合 $T_D$ 中包含这些召回维度-标签的文档：

$$
\mathcal{D}_{cand}
=\{D \mid T_D \cap L_q \neq \varnothing\}
$$

对候选 chunk $D$，设它在本次召回中命中的维度标签集合为：

$$
A_D^q=\{(m,t)\mid (m,t)\in L_q,\ (m,t)\in T_D\}
$$

### 2.2 第二阶段：精排

精排只在初步召回候选集合内进行。该阶段可以使用更细粒度、更耗时的得分函数。

#### 2.2.1 方法一：query 向量与维度-标签向量

数据 $D$ 的维度标签向量得分可以采用平均值：

$$
S(q,D)=\frac{1}{|A_D|}\sum_{(m,t)\in A_D} sim(\mathbf{v}_q,\mathbf{v}_{m,t})
$$

其中，$A_D$ 可以是数据 $D$ 包含的所有维度-标签，即 $T_D$；也可以是 2.1.3 中的 $A_D^q$。

注意，代码中在这里混入了一次语义检索。

#### 2.2.2 方法二：标签匹配得分

标签匹配得分按 $M_q$ 中的维度逐维计算：

$$
S(q,D)=\sum_{m\in M_q}
\begin{cases}
\max\limits_{t_q\in Q_m,\ t_D\in T_D[m]} sim(\mathbf{v}_{t_q},\mathbf{v}_{t_D}), & T_D[m]\neq\varnothing \\
0, & T_D[m]=\varnothing
\end{cases}
$$

## ！！！！！ 维度检索两阶段 融合策略！！！！！！！！！！！
1.进行RRF融合一次
2.直接使用精排的得分


## 3. 语义检索与维度检索的融合策略

本节只整理最终双路融合，即将维度检索结果 `dim_results` 与纯语义检索结果 `sem_results` 融合。维度检索内部如何融合文本语义得分与标签匹配得分，暂不纳入本文策略。

### 3.1 最终双路融合公式

最终融合得分定义为：

$$
S_{final}(q,D)=\alpha_{sem} R_{sem}(q,D)+\alpha_{dim} R_{dim}(q,D)
$$

其中，$R_{sem}(q,D)$ 和 $R_{dim}(q,D)$ 分别表示两条检索路径归一化后的文档得分。$\alpha_{sem}$ 和 $\alpha_{dim}$ 本质上是在衡量系统对两条检索路径的信任程度。

### 3.2 规则型自适应权重方案

令 $K_{dim}$ 与 $K_{sem}$ 分别表示维度检索和语义检索的 Top 结果集合，$K_r$ 表示其中任一路的 Top 结果集合。

#### 3.2.1 判断 query 是否适合标签化

如果 query 能被清晰解析成维度约束，维度检索更可能提供稳定增益。

定义结构化置信度 $P_q$：

$$
P_q=
\begin{cases}
\frac{2n_b}{n_d+n_b+1}, & n_d>0 \\
0, & n_d=0
\end{cases}
$$

其中：

- $n_d$：query 解析出的有效维度数量，去重后统计。
- $n_b$：至少绑定了一个维度值的维度数量。

#### 3.2.2 判断标签证据是否可靠

需要看 query 解析出的维度值是否能在文档标签中形成有效命中。如果标签缺失、只命中少数维度，或命中多为低相似度软匹配，就不应过度相信维度检索路径。

对每个 query 维度 $m\in M_q$，定义该维度的标签证据得分：

$$
s_m=
\begin{cases}
\max\limits_{D\in K_{dim},\ t_q\in Q_m,\ t_D\in T_D[m]} sim(\mathbf{v}_{t_q},\mathbf{v}_{t_D}), & \exists D\in K_{dim}:T_D[m]\neq\varnothing \\
0, & \text{otherwise}
\end{cases}
$$

定义标签证据总量：

$$
E_T=\sum_{m\in M_q}s_m
$$

定义标签证据置信度 $T_q$：

$$
T_q=
\begin{cases}
\frac{2E_T}{|M_q|+E_T+1}, & |M_q|>0 \\
0, & |M_q|=0
\end{cases}
$$

该公式同时考虑两个因素：

- 标签证据数量：$E_T$ 越大，说明越多 query 维度能在文档标签中找到有效证据。
- 标签证据缺口：$|M_q|-E_T$ 越大，说明越多维度没有命中，或只形成了低相似度软匹配。

#### 3.2.3 判断两路检索结果是否稳定

定义单路 Top 结果集中度。对任一路 $r\in\{dim,sem\}$，当该路 Top 结果得分总量大于 0 时，先将得分归一化：

$$
p_i^r=\frac{R_r(q,D_i)}{\sum_{D_j\in K_r}R_r(q,D_j)}
$$

用归一化熵计算集中度：

$$
C_r=
\begin{cases}
1-\frac{-\sum_{D_i\in K_r}p_i^r\log p_i^r}{\log |K_r|}, & |K_r|>1,\ \sum_{D_j\in K_r}R_r(q,D_j)>0 \\
1, & |K_r|=1,\ \sum_{D_j\in K_r}R_r(q,D_j)>0 \\
0, & |K_r|=0\ \text{or}\ \sum_{D_j\in K_r}R_r(q,D_j)=0
\end{cases}
$$

#### 3.2.4 生成最终融合权重

最终融合先计算两路效用。这里不再引入额外权重系数，而是把前面三类证据直接作为两路检索的支持证据：

- query 越适合结构化，越支持维度检索；越不适合结构化，越支持语义检索。
- 标签证据越可靠，越支持维度检索；标签证据越弱，越支持语义检索。
- 某一路结果越稳定，越支持该路检索。

因此定义维度检索效用：

$$
U_{dim}=P_q+T_q+C_{dim}
$$

定义语义检索效用：

$$
U_{sem}=(1-P_q)+(1-T_q)+C_{sem}
$$

再归一化：

$$
\alpha_{dim}=\frac{U_{dim}}{U_{dim}+U_{sem}+\epsilon}
$$

$$
\alpha_{sem}=1-\alpha_{dim}
$$

其中，$\epsilon>0$ 用于避免分母为 0。
