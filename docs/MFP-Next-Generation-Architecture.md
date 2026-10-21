# MFP 下一代模型架构设想书

> 三条完整路线已在现有项目实现，具体接口、实际容量和评价协议见第 8 节。设计机制的性能与创新收益仍待正式训练及消融验证；前文预算是设计时的估计，不能代替实际参数测量。

## 1. 结论与数据依据

**优先实现 TriRelay：三模态证据交互与标签反馈网络。** 另保留 GeoFactor 异构因子图、ProtoRoute 多原型专家两条独立完整路线。三者在表征组织、信息传播和决策方式上有实质区别。

当前序列模型的单 seed Validation Macro-AP：major15 的 ESM-35M 为 0.45164、ESM-650M 为 **0.47782**；fine39 分别为 **0.43251** 和 0.42167。扩大 PLM 有价值，但不能代替结构、理化及标签建模；这些数值也不足以确定下一代模型的最佳编码器。

现有数据包含最长 74 残基的序列、D 构型注记、Cα 坐标、骨架/接触边、DSSP、φ/ψ/ω、ASA/RSA、部分明确来源的置信度，以及 515 项理化和化学描述符。major15 明确骨架训练集 39,478 条，其中 34,387 条有可用主结构；fine39 相应为 39,455 条和 34,367 条。结构缺失是正常输入情况。详见 [`structure_schema.json`](../data/therapeutic_2025/structure_schema.json)、[`physchem_schema.json`](../data/therapeutic_2025/physchem_schema.json) 和 [`README.md`](../data/therapeutic_2025/README.md)。

理化描述符大多是计算代理量，并非独立实验测量；现有坐标代表选定结构，而非构象轨迹。标签的 0 是未标注，`__unknown` 是亚类未知。不能把未共现解释为互斥，也不能把同一大类下的亚类强制归一化成单标签分布。

## 2. 本地同类研究的结构与评价

本节依据根目录的 `同类研究论文.zip` 中 9 篇 PDF。所列 checkpoint 规则只记录论文正文中可确认的依据；论文重视某个 Test 指标，不等于按它选 epoch。

| 方法 | 输入、编码与完整架构 | 损失及可借鉴设计 | 报告指标与选择依据 |
|---|---|---|---|
| [TPpred-LE，2023](https://doi.org/10.1186/s12915-023-01740-w) | 序列 Transformer encoder、标签 embedding decoder、标签—序列交互及功能特异分类器 | BCE；共享表示后训练特异分类器，MCRT 处理少样本功能 | 样本 Jaccard、Macro-F1、逐类 AUC/MCC/F1；按最小 Val loss 选超参数，逐 epoch checkpoint 规则未明确 |
| [iMFP-LG，2024](https://doi.org/10.1093/gpbjnl/qzae084) | 可微调 TAPE：12 层、768 维；每标签投影、两层六头标签 GAT、独立二分类器 | BCE + FGM；样本条件化标签节点和关系传播 | 样本 Precision/Recall/Jaccard、Exact Match、Hamming Loss，逐类敏感度/特异度；最佳 epoch 指标未明确 |
| [MFP-MFL，2025](https://doi.org/10.3390/ijms26031317) | ESM-2、ProtT5、RoBERTa 各自表征与标签图分类，再加权集成 | 分类监督 + FGM；多序列表征互补、标签图 | 常见五项多标签指标；最佳 checkpoint 指标未明确 |
| [AMCL，2025](https://doi.org/10.1186/s12915-025-02273-0) | 可学习序列 embedding、核 3/4/5 的 CNN、池化、多层前馈分类及对比学习路径 | FDL + 论文定义的 DBL、难样本多标签对比；学习标签重叠与长尾表示 | 五项多标签指标、Macro/Micro 及逐类 P/R/F1；最佳 checkpoint 指标未明确 |
| [MCMFPP，2025](https://doi.org/10.1021/acs.jcim.5c01693) | SLFE：ESMC 960 维 + 可学习 128 维 → TextCNN → MLP；CFEC：增强和对比学习；head/tail 分别加权融合分类概率 | MLFDL；CFEC 加 MulSupCon。不同学习目标的分类器可互补 | 五项多标签指标、Macro-F1、逐类 SEN/SPE/F1；训练集五折以 0.5×Jaccard + 0.5×Macro-F1 选超参数，未明确 epoch 选模 |
| [METFAN，2025](https://doi.org/10.1021/acsomega.5c07552) | 序列 embedding → MHSA/四尺度 TextCNN；ESM2、ProtT5 全局向量各经瓶颈残差及通道重标定；残差 FAN 融合后分类 | MLFDL；局部/全局分工与 PLM 适配，但池化后交互缺少残基级对应 | 样本 Jaccard、Macro-F1、逐类 AUC/MCC/F1；有独立 Val，最佳 checkpoint 指标未明确 |
| [SeekTP，2025](https://doi.org/10.1093/bib/bbaf569) | ESM-1b、one-hot、二级结构/RSA 与预测接触图 → 三层四头 GAT；并行 MHSA/四尺度 TextCNN；融合 FFN | MLFDL；空间邻域与序列邻域并行建模 | 五项多标签指标、逐类 AUROC/AUPR/F1；最佳 checkpoint 指标未明确 |
| [SA-MTP，2026](https://doi.org/10.1093/bib/bbag361) | ESM2-650M → 256 维；二级结构概率及接触先验 → 动态稀疏图 → 两层多头 GAT → 标签交叉注意力/FiLM | BCEWithLogits；结构不确定性和标签相关残基读取 | 逐类 AUC/MCC/F1、样本 P/R/F1；Val 表现早停 patience=15，具体指标未明确；逐类阈值依 Val F1/MCC |
| [PDeepPP，预印本](https://arxiv.org/abs/2502.15610) | ESM2-650M + 可学习 embedding；并行四层 Transformer 的 TransLinear 与 PosCNN，融合分类；主要覆盖多个二分类和 PTM 任务 | CE + TIM 启发的信息熵目标；较深的局部/全局双路表示 | ACC/BACC/Sn/Sp/MCC/ROC-AUC/PR-AUC；逐 epoch 选模指标未明确 |

同类论文已普遍使用多分支、PLM 适配、局部/全局表示、标签特异分类、关系传播或辅助目标。仅在现有 PLM 向量后加一层 attention、GNN 或替换 loss，不能构成完整下一代架构。

### 本项目统一指标协议

| 环节 | 指标与决策 |
|---|---|
| 训练监控 | 总 loss、分类和辅助 loss、学习率、模态权重；每轮 Val Macro-AP、Micro-AP、固定 0.5 阈值的 Macro/Micro-F1、head/medium/tail Macro-AP |
| checkpoint | **Val Macro-AP 最大 → Micro-AP 最大 → 更早 epoch**，延续当前协议 |
| 阈值 | 仅在最佳 checkpoint 的 Validation 预测上确定；主结果沿用全局阈值最大化样本 Jaccard；逐类收缩阈值作为独立实验 |
| Test 主报告 | Macro-AP、Micro-AP、Macro-F1、Micro-F1、样本 Jaccard |
| Test 补充 | 样本 P/R/F1、Exact Match、Hamming Loss、Micro-P/R；逐类 AP/F1/MCC/support、Macro-AUROC |
| 分层结果 | 按训练正例数对标签排序，major15 各 5 类、fine39 各 13 类构成 head/medium/tail，并报告各类实际 support；另报有/无结构、D 注记、单/多标注样本 |

论文中常称的 Accuracy/ACCexample 是**样本 Jaccard**，Coverage 是**样本平均 Recall**，Absolute true 是 Exact Match，Absolute false 是 Hamming Loss。AP 继续使用 `average_precision_score` 的非插值定义，区别于梯形积分 PR-AUC。Val 上拟合阈值后的 F1/Jaccard 是拟合内结果；最终判断依独立 Test。

## 3. 前沿方法与可迁移思想

| 来源 | 可用思想与边界 |
|---|---|
| [ProSST，NeurIPS 2024](https://proceedings.nips.cc/paper_files/paper/2024/hash/3ed57b293db0aab7cc30c44f45262348-Abstract-Conference.html)、[GearNet，ICLR 2023](https://arxiv.org/abs/2203.06125) | 序列—结构交互、几何关系编码。本项目已有连续几何，可直接利用，不必先训练结构 tokenizer |
| [CSSFE，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Du_Category-Specific_Selective_Feature_Enhancement_for_Long-Tailed_Multi-Label_Image_Classification_ICCV_2025_paper.html) | 长尾也影响标签相关局部特征，应增强尾类自己的证据读取 |
| [标签分组，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_Correlative_and_Discriminative_Label_Grouping_for_Multi-Label_Visual_Prompt_Tuning_CVPR_2025_paper.pdf)、[反事实多标签，ICML 2024](https://proceedings.mlr.press/v235/xie24i.html) | 同时建模相关性与判别性，保留直接证据通路，避免共现捷径；不能把零共现当生物学互斥 |
| [MultiMAE，ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136970341.pdf)、[BALM，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Nguyen_BALM_A_Model-Agnostic_Framework_for_Balanced_Multimodal_Learning_under_Imbalanced_CVPR_2026_paper.html) | 跨模态掩码预测与缺模态优化；优先用可用性 mask、分支监督和模态 dropout |
| [信息瓶颈对齐，ICML 2025](https://proceedings.mlr.press/v267/almudevar25a.html)、[MulSupCon，AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29619) | 对比学习需要正确处理私有信息与标签重叠；无共同标注不能一律当真负对 |
| [IGNORE，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05081.pdf)、[Crisp，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1dc9fbdb6b4d9955ad377cb983232c9f-Abstract-Conference.html) | 未标注标签与类别先验值得研究；原方法主要面向单正例多标签，不能直接套用理论条件 |
| [Static-Dynamic Fusion，ICML 2025](https://proceedings.mlr.press/v267/guo25b.html) | 构象动态可能提供互补信息；本项目没有 MD 轨迹，不能宣称现有静态结构具备此能力 |

[ASL，ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf) 和 [Distribution-Balanced Loss，ECCV 2020](https://arxiv.org/abs/2007.09654) 仍适合作为成熟损失对照。

## 4. 三条可独立训练的完整路线

### TriRelay：三模态证据交互与标签反馈

序列语义/局部编码、几何图编码、分组理化编码 → 三轮双向模态交互 → 每标签读取三种证据 → 样本×标签×模态自适应融合 → 条件标签关系和高阶因子 → 标签证据反馈到模态 token → 第二轮读取 → 直接证据与关系修正共同分类。

首套规格：4 层序列编码、4 层几何消息传递、2 层理化 Transformer、3 个交互块、2 轮标签读取，以及直接分类和关系推断双通路。研究假设是不同功能需要读取同一条肽中的不同模态和残基证据，且标签反馈可修正第一轮遗漏的证据。

### GeoFactor：异构几何因子图与结构化预测

将残基、连续/空间片段、理化属性组、功能标签设为四类节点；骨架、接触、残基—片段、属性—片段及片段—标签形成类型化边。6 层消息传递融合局部与全局，标签间使用样本条件化的成对及高阶因子，4 步可微近似推断得到结构化多标签输出。缺结构时保留序列/属性图，不建立虚假空间边。训练使用结构化分类损失、一元头辅助监督、几何掩码重建和长尾加权。

可研究的能量形式：

$$E(y\mid x)=-\sum_c u_c(x)y_c-\sum_{c<d}a_{cd}(x)y_cy_d-\sum_k b_k(x)\prod_{c\in\mathcal H_k}y_c.$$

其核心问题是共享空间片段和理化因子能否解释多功能性，而非仅拟合标签统计共现。

### ProtoRoute：标签条件的多原型专家

配置语义/局部、几何、理化、跨模态协同四个完整专家，分别有 4、4、3、3 层编码/交互。结合数据质量评估，每标签的专属原型容量调整为按独立正例簇分配 1–4 个，保留共享原型库；几何专家按有结构正例簇单独分配，零支持时不路由该标签到几何专家。每个样本—标签对独立生成软专家权重，融合原型相似度与标签关系修正。尾类残差向共享原型收缩。其他标签的未标注样本不自动成为负原型。

其核心问题是功能内部多样性与不同样本的最佳专家组合。

| 维度 | TriRelay | GeoFactor | ProtoRoute |
|---|---|---|---|
| 表征组织 | 三条模态流 | 统一异构图 | 四个专门专家 |
| 交互机制 | 双向注意力、残基对齐、标签反馈 | 类型化消息传递与因子推断 | 标签条件软路由 |
| 标签模型 | 独立证据 token + 条件关系 | 标签变量节点 | 每标签多原型 |
| 决策 | 直接分类 + 关系残差 | 展开式结构化推断 | 专家混合与原型匹配 |
| 下游可训练参数预算，不含冻结 PLM | 约 15–25M | 约 12–25M | 约 20–35M |
| 主要待验证风险 | 融合冗余、自强化反馈 | 推断稳定性 | 原型坍塌、路由失衡 |

参数区间是设计预算，不是已实例化测量值。**推荐 TriRelay 优先实现**：与现有数据字段对应最直接，模态与标签模块的贡献也较容易定位。GeoFactor 和 ProtoRoute 保持独立完整模型身份。

## 5. TriRelay 的实现规格

### 数据流与编码

```text
ESM 残基 [B,L,1280] + AA/D 注记 ──> 4 层序列编码 ─────────┐
Cα/接触/角度/DSSP/ASA/RSA ──> 4 层几何编码 ──────────────┤
515 项理化/化学值及 mask ──> 25 token、2 层属性编码 ──────┤
                                                         ↓
                         3 个三模态交互块 → 标签读取/融合
                                                ↓
                              条件关系/高阶因子 → 标签反馈
                                                ↓
                             第二轮读取 → 最终独立 logits [B,C]
```

主设置隐藏维度 $d=256$、8 个注意力头、FFN 维度 1024；同一架构分别训练 major15 与 fine39。ESM2-650M 从 1280 投影到 256，ESM2-35M 保留为同架构编码器对照。

**序列分支。** 冻结 PLM 的逐残基表示与可学习 AA、D 注记和位置 embedding，经门控残差结合。4 层局部—全局混合块各含核 3/5/7 卷积、残基 self-attention、FFN、残差和归一化。输出 `[B,L,256]`，保留局部证据及全局 token。当前 ESM 将大小写归一，同骨架 D/L 变体需要此独立化学路径才能区别。

**结构分支。** 节点使用 AA/D、DSSP、角度 sin/cos、ASA/RSA，以及对应缺失 mask；边使用邻接/非局部接触类型、Cα 距离 RBF、序列间隔和局部几何方向。4 层关系消息传递保留浅层与深层特征。距离对镜像不敏感，故保留有符号角和 D 注记；需要时加入局部有符号三重积。输出预测应对平移旋转不变，但不强求镜像不变。置信度没有明确来源时使用“未知”状态，不能伪装成低或高置信度。缺结构样本跳过空间消息，不制造零坐标图。

**理化分支。** 400 维 DPC 按首位残基形成 20 token；AAC、L/D 组成、序列理化代理、RDKit 代理、化学注记各形成 1 token，合计 25 个。每组的数值、可用性 mask 和组 embedding → 256 维，再经 2 层属性 Transformer。来源冲突及解析状态主要控制可靠性，不当作生物功能证据。结构附带的 25 个全局标量归入结构分支，不重复包装成独立模态。

### 三模态交互与标签决策

**序列↔结构：**残基位置对齐，双向 cross-attention；空间距离与邻接类型作为 attention bias。**序列↔理化：**残基读取属性 token，属性反向聚合相关残基。**结构↔理化：**属性调制空间边消息：

$$m_{ij}=\gamma(P)\odot\phi(H_i,H_j,\operatorname{RBF}(d_{ij}),t_{ij})+\beta(P).$$

这里是可学习条件变换，不预先规定某种电荷或疏水关系决定功能。全局 token 双向读取残基 token，避免过早池化丢失定位信息。

每个标签 $c$ 用独立 query 从序列、结构、理化分别读取证据：

$$z_c^m=\operatorname{CrossAttn}(q_c,H^m),\qquad
\alpha_c^m=\operatorname{MaskedSoftmax}_m f(q_c,z_c^m,r^m),\qquad
z_c=\sum_m\alpha_c^m z_c^m.$$

融合权重依赖**样本、标签和模态可用性/质量**；缺结构时其权重严格为零。保留模态私有残差，使几何或构型特异信息不被公共空间完全抹平。

每标签首先产生直接 logit，获得独立监督。训练集正例共现可构造平滑的关系先验，由当前样本条件化修正；低支持共现向中性先验收缩，零共现不建立互斥。高阶关系使用有足够训练支持的三元/小组合与少量潜在功能组，对成员表示作非线性联合聚合，而非仅用一次标签相似矩阵。初始候选：最多 32 个经验组合、支持数起点 20、8 个潜在组、2 层关系更新；均待验证。尾类没有可靠经验组合时仍可经潜在组共享表示。

$$\ell_c=\ell_c^{\mathrm{direct}}+g_c(x)\,\Delta\ell_c^{\mathrm{relation}},\qquad 0\le g_c(x)\le1.$$

**标签反馈是主研究机制。** 第一轮标签表示回传至相关模态 token，调制一次共享交互块并进行第二轮读取。训练与推理采用相同两轮计算，反馈只用当前样本的预测表示，不输入真实标签。直接证据头持续受监督，防止关系模块仅靠标签共现相互拉高概率。

### 长尾、目标函数与 Tensor Shape

长尾同时作用于表示和决策：标签独立 query 读取局部残基；分类器使用共享基、低秩标签系数及标签残差 $w_c=w_{shared}+Ua_c+\delta_c$，对尾类残差适度收缩；训练正例数产生**截断平方根权重**；稀少共现先验收缩。不要默认重复采样整条多标签样本，否则尾类伴随的头类也被同步放大。

首套完整训练目标：

$$\mathcal L=\mathcal L_{weighted\text{-}ASL}
+0.2\mathcal L_{direct}
+0.1\mathcal L_{branch}
+0.05\mathcal L_{cross\text{-}mask}.$$

主项用于最终分类；直接证据项抑制共现捷径；有效模态的分支监督防止分支失效；跨模态掩码项预测原本可观测的角度/DSSP/属性。掩码目标及其等价副本须同时从输入隐藏，例如预测距离时不能仍在空间边特征中直接读取同一距离。各项按有效元素归一化，系数为起点。ASL 是处理观测标签的实用损失，不证明未标注项是真阴性。

| Tensor | Shape |
|---|---|
| ESM-650M 残基输入 | `[B,L,1280]` |
| 序列、结构隐藏表示 | 各 `[B,L,256]` |
| 坐标、角度 sin/cos | `[B,L,3]`、`[B,L,6]` |
| 理化值和可用性 | 各 `[B,515]` |
| 理化 token | `[B,25,256]` |
| 标签 query | `[C,256]`，`C=15/39` |
| 各标签的三模态证据 | `[B,C,3,256]` |
| 自适应模态权重 | `[B,C,3]` |
| 标签表示与动态关系 | `[B,C,256]`、`[B,C,C]` |
| 标签—高阶因子关联 | `[B,C,K]` |
| 最终 logits / 概率 | `[B,C]` |

## 6. 与同类研究的复杂度对标

| 维度 | 本地同类研究代表 | TriRelay |
|---|---|---|
| 输入模态 | 多为序列多表征；SeekTP/SA-MTP 加结构信号 | 序列、连续几何、理化/化学描述符 |
| 编码深度 | TAPE 12 层；PDeepPP 下游 Transformer 4 层；结构 GAT 2–3 层 | 冻结 ESM2 33 层；下游序列 4、几何 4、属性 2 层 |
| 编码分支与层级 | METFAN 三路；SeekTP 图/序列并行；残基和全局 | 三路独立，保留残基、片段/空间邻域、全局、标签、高阶因子 |
| 局部与全局 | TextCNN、多头注意力、池化 | 多尺度卷积、几何邻域、全局 token 回写 |
| 模态交互 | 拼接、FAN、结构图约束 | 三轮双向交互、属性调制空间消息 |
| 标签关系 | 标签 Transformer 或 GAT | 条件成对关系、非线性高阶因子及反馈 |
| 融合与分类 | 加权集成、FiLM、独立分类头 | 每样本每标签融合；直接证据 + 关系残差 |
| Loss | BCE、FDL/DBL、对比、FGM | 加权 ASL + 直接/分支监督 + 跨模态掩码目标 |
| 参数与计算 | 文献冻结/微调/集成口径不同，完整参数不可直接排名 | 下游预算 15–25M；主要计算含 $O(L^2d+LPd+CLd+C^2d+CKd)$ 和线性层 $O(Nd^2)$ |

当前 MFP-Nova 下游仅约 0.13–0.24M 可训练参数；iMFP-LG 的 MFTP 标签独立投影仅 $21\times768\times768$ 权重已约 12.4M，尚不含 GAT 和可训练 TAPE。METFAN 的 6272 维融合及宽前馈可能更大，但论文总参数口径并不统一。TriRelay 将容量放在有职责分工的几何、交互与标签读取上；是否容量充分通过同架构 $d=256/384$ 比较。肽长 $L\le74$、标签数 $C\le39$，交互计算可以实施。

## 7. 训练、推理及首轮消融

**训练：**使用与当前结果相同的明确骨架队列及簇划分，保留缺结构样本；复用现有冻结 ESM 残基缓存，新增结构/理化加载。首轮实现即包含完整三模态、标签交互和反馈架构。初始 AdamW 学习率 $2\times10^{-4}$、weight decay $10^{-3}$、dropout 0.15、5 epoch warmup、最多 100 epoch、patience 15。batch 从 32 的设计值起，实际按显存调整。辅助项前 5 epoch 逐渐升权；对有效辅助模态作适度遮蔽，但不改原始可用性记录或标签。以 Val Macro-AP 选 checkpoint，再仅以 Val 选阈值。正式比较固定多个 seed，报告均值与标准差。

**推理：**输入序列和可用的结构/理化字段 → 分支编码 → 三模态交互 → 两轮标签读取/反馈 → sigmoid → 应用 Validation 固定阈值。推理不使用真实标签、Test 共现统计或同批其他样本的信息。训练建立的预处理参数和标签先验随模型保存。暂未明确骨架的 HELM/符号样本可另设覆盖实验；与当前结果比较时先保持相同队列。

**首轮模块消融：**

| 比较 | 回答的问题 |
|---|---|
| 关闭标签反馈 | 第二轮是否找到新的有效证据？ |
| 三模态交互改为最终融合 | 增益来自多输入还是交互？ |
| 关闭理化条件空间消息 | 属性是否有助于几何传播？ |
| 标签条件融合改为样本级统一融合 | 不同功能是否需要不同模态？ |
| 去掉高阶因子、保留成对关系 | 高阶关系是否超出普通标签图？ |
| 关闭关系模块、保留直接证据 | 标签依赖的净收益是什么？ |
| 关闭长尾共享/加权、cross-mask、分支监督 | 各专门机制是否帮助主任务？ |
| 不使用 D 注记和有符号几何 | 相同骨架上的构型信息是否有用？ |
| ESM35M/650M、隐藏维 256/384 | PLM 规模和下游容量分别有什么作用？ |

研究重点是**理化条件化的序列—结构交互、每标签的模态证据分配与反馈读取、直接证据支撑的高阶标签推理**。这些机制目前是可实现的研究假设，性能与独创性仍须通过同队列实验、消融和独立 Test 检验。

## 8. 已实现架构与统一实验协议

### 实际实现

代码位于 `src/mfp_nova/`，三套独立文件为 `trirelay.py`、`geofactor.py`、`protoroute.py`。`research_layers.py` 仅共享神经网络积木，`multimodal.py` 共享只读数据与 Train 统计，`research.py` 共享训练/恢复/评价/推理。使用普通 PyTorch，没有模型注册器、图框架依赖或远程调度。

| 模型 | 完整深度与决策机制 | major15 参数 | fine39 参数 |
|---|---|---:|---:|
| TriRelay | 序列 4、几何 4、属性 2、双向交互 3；两轮读取、反馈、两层成对/经验及潜在高阶关系、直接/关系残差分类 | 35,196,329 | 35,209,345 |
| TriRelay 共享—私有候选（默认关闭） | 保留上述全部深度；每模态共享残差投影、两层私有 FFN、两轮标签条件私有读取及质量感知门控 | 39,147,692 | 39,160,708 |
| GeoFactor | 序列 2、属性 2；六层异构消息传递；连续 4/8 残基片段及空间片段；一元/二元/经验高阶/潜在共享因子，四步阻尼推断 | 30,998,706 | 31,011,722 |
| ProtoRoute | 序列 4、几何 4、属性 3、协同交互 3；四专家标签读取、共享库与残差原型、标签条件路由、关系残差分类 | 32,327,070 | 32,451,062 |

测量条件：隐藏维 256、8 heads、ESM-650M 残基输入 1280；本地原始明确骨架训练队列及其化学词表。参数不含离线冻结 ESM。容量高于早期预算，来自全通道多尺度卷积、多向交互与完整图传播；未为了贴合预算删除核心机制。650M/35M、隐藏维度和模块开关均可比较，参数会随配置、词表和标签变化。

### TriRelay 共享交互＋私有证据候选（2026-10-21）

`--shared-private` 启用，`--no-shared-private` 关闭；原配置和旧 checkpoint 默认保持原架构。编码器输出各自形成完整 256 维的共享/私有流，没有对半缩窄原交互网络。共享流经原三轮双向交互、关系反馈和 refine；私有流经过两个扩张倍数为 2 的残差 FFN，绕过交互与反馈回写，供两轮标签读取使用。共享投影同为扩张倍数 2 的残差 FFN。两轮复用同一组私有注意力与门控，不新增一个完整编码器。

对样本 i、标签 c、模态 m，标签 query 读取共享/私有证据后，执行：

```text
g_icm = sigmoid(MLP_m([query_ic, shared_icm, private_icm, quality_im]))
e_icm = shared_icm + g_icm * private_icm
alpha_icm = masked_softmax_m(score([query_ic, e_icm, quality_im]))
```

缺失模态的 g 和 alpha 严格为零。第一轮融合证据进入既有关系推理；预测标签表示仅回写共享 token；第二轮以关系上下文重新查询同一批私有 token，门控限制私有证据的残差幅度。最终使用原长尾分类器及关系残差决策。门控形状 `[B,C,3]`，两类证据均为 `[B,C,3,256]`；原残基/全局 token 及 mask 保持不变。

这里“私有”指绕过 relay 的独立可学习路径，**不保证统计独立，也不声称完全单模态**：几何编码器保留原有理化条件调制。共享编码器仍由分类与重建共同优化，私有流也不是冻结的原始特征。没有正交、互信息或全模态强制对齐损失。

总 Loss 仍是主分类＋0.2 直接头＋0.1 分支＋0.05 重建，warmup 沿用原规则。0.1 分支预算在有效模态的融合分支与私有分支间等分，不另加一份 0.1；重建头只读取共享流，但第一轮融合标签会反馈至共享流，因此重建梯度仍可间接到达私有路径，不宣称梯度隔离。两个分支复用原长尾分类器。新增参数 3,951,363，约增加 11.2%。

本轮完整 major15/fine39、seed42、EMA-only 协议验证此单项机制：consistency/function auxiliary 均为 0、重建梯度为原 `sum`，不叠加 cap/project，不使用 Test。对照为已有独立 EMA-only 训练；同轨迹 `online_best.pt` 仅提供在线/EMA参考，不能充当无私有流对照。是否保护稀有模态证据、有无实际收益，需由后续 GPU 完整训练与标签/缺模态分组结果判断。

实现检查：12 项相关 CPU 测试通过，覆盖两任务、D/L/padding/缺模态、两轮私有 token 不被回写、关键路径梯度、旧模型兼容、EMA 精确恢复及独立 Val 重评。真实 ESM-650M 缓存另做两任务各 8 样本、2 optimizer step 的完整规模 CPU 检查，全部选定模块有非零有限梯度及参数更新；诊断记录位于 `runs/diagnostics/trirelay_shared_private_20261021/real_cache_check.json`。这是实现验证，不是性能/收敛结果；本轮未运行 GPU 或 Test。

### 数据和 Tensor 约定

默认继续使用原始明确骨架队列，不使用已有未验证合成肽。保留无结构和 D 肽。序列输入为 ESM `[B,L,D]`、AA/D/残基化学 token `[B,L]`、全局修饰 token `[B,T]`；G 不赋予真实 D 构型。未定位 PTM/连接文本保留为全局化学语义，不虚构局部修饰位置。

结构输入为坐标 `[B,L,3]`、几何 `[B,L,16]`、DSSP `[B,L]`、类型边 `[B,L,L]`、25 个全局统计及观察 mask。16 个几何通道依次为角度 sin/cos 6、角度观察 3、验证置信度/100 1、标准化 log1p(ASA)/RSA 2、置信度观察 1、ASA/RSA 观察 2、有符号 Cα 键三重积 1。骨架和接触边分开；无结构不制造空间关系。

理化为 `[B,515]` 加同形观察 mask，分为 20 个 DPC 行 token 和 AAC、L/D、序列代理、RDKit 代理、化学注记五组，共 25 token。节点和描述符标准化、化学词表、标签先验、经验高阶组合、长尾及原型容量全部仅由原始 Train 拟合，并随 checkpoint 保存。RDKit 值仍是母体线性肽代理量；NaN 不是零值，填补不取消观察 mask。置信度未知单独表示，不强制作为低质量；可靠性用于消息、交互和路由，不宣称是实际构象准确度。

模型 `forward(batch)` 返回包含 `logits[B,C]` 的字典，`compute_loss(batch, settings)` 返回总 Loss 和分项；推理不消费 `y`。

### Loss 和增强

共同分类采用正例长尾加权 ASL，未标注项默认权重 0.3。fine39 中出现 `__unknown` 时，同父类具体亚类的未标注项进一步乘 0.25；仍是弱监督假设，不是已验证的 PU 风险估计。未共现不建立硬互斥。

- TriRelay：主分类 + 0.2 直接证据 + 0.1 有效分支监督 + 0.05 重建。
- GeoFactor：主分类 + 0.2 一元监督 + 0.05 重建 + 0.001 因子正则。经验高阶因子要求至少 20 正例/10 簇，最多 32 个，共享非线性核并按簇支持收缩；另有 8 个潜在共享因子。推断使用预测概率，不输入真实标签。
- ProtoRoute：主分类 + 0.2 直接监督 + 0.1 专家监督 + 0.1 已标注正例原型吸引 + 0.01 多样性 + 0.01 可用专家平衡 + 0.05 重建。初始容量 `min(4,max(1,正例簇数//100))`，几何零支持容量为 0；该启发式待消融。共享库随机初始化并由 Train 学习，不宣称已完成类内聚类。

初始训练增强为结构/属性 dropout 0.1 和观察字段遮蔽 0.15，始终保留序列。AAC/L-D 和质量/RDKit 的重复组联合遮蔽；隐藏局部几何时同步隐藏相应全局摘要。重建损失按属性组平均、仅对实际观察目标计算。坐标仍保留时角度/ASA/DSSP 重建是结构条件去噪，不宣称独立的跨模态结构真值或新构象生成。没有氨基酸替换、构型改变或坐标噪声后继承功能标签。

### 评价定义

设标注集合为 `Y_i`，预测集合为 `P_i`，样本数 N、标签数 C：

| 指标 | 实际公式/口径 |
|---|---|
| Accuracy / sample_jaccard | `mean_i |Y_i∩P_i|/|Y_i∪P_i|` |
| 样本 Precision | `mean_i |Y_i∩P_i|/|P_i|` |
| 样本 Recall / Coverage | `mean_i |Y_i∩P_i|/|Y_i|`，不是 Coverage Error |
| Absolute True / Exact Match | `mean_i 1[Y_i=P_i]` |
| Absolute False / Hamming Loss | `sum_i |Y_i△P_i|/(N*C)` |
| Sensitivity / Specificity | `TP/(TP+FN)` / `TN/(TN+FP)`，无对应分母时 null |
| MCC | `(TP*TN-FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))`，无分母时 null |
| Macro/Micro P/R/F1 | 标签等权 / 汇总标签混淆计数；F1 单独计算，不由 Macro P/R 调和代替 |
| AP | sklearn `average_precision_score`；无正例 null，不纳入 Macro-AP；Micro-AP 展平 |
| AUROC | 缺正例或未标注对照时 null；Macro 对可计算标签平均 |
| Head/Medium/Tail Macro-AP | 固定原始明确骨架 Train 支持：≥1000 / 200–999 / <200，跨模型不改变 |

集合指标空分母取 0。Specificity、MCC、AUROC 及其他指标均是**相对已有标注的描述性指标**，不把未标注项解释为实验阴性。输出包括逐标签 AP/F1/MCC/P/R/Sensitivity/Specificity/AUROC/Support。

MFP 主口径为明确的**独立大功能轴**：fine39 先映射至 15 个功能轴，同轴亚类只计一个功能；generic antimicrobial 在已有 antibacterial/fungal/parasitic/viral 轴时不再重复计数。≥2 个轴定义本项目 MFP 阳性。`mfp_detection` 报识别多功能样本的二分类 P/R/F1；`mfp_function_prediction` 报真实 MFP 子集上的功能轴样本平均 P/R/F1。这个操作定义不等同于论文原始 21,130 条计数，不用 fine39 标签数替代独立功能数。

每 epoch 记录总/分项 Loss、学习率、耗时和 Validation 指标；按 Val Macro-AP、Micro-AP、更早 epoch 选 checkpoint。主实验在最佳模型 Val 上拟合全局阈值，网格 0.05–0.95，最大化样本 Jaccard；逐标签 Val F1 阈值另作 `per_label` 实验。Test 仅独立评价，不自动调参、选模型或回写阈值。

评价另输出结构可用性、来源证据、已知/未知置信度、非甘氨酸 D、长度 2–8/9–16/17–32/33+、单/多功能分组。实验结构的 ID 来源未确认，不能虚构实验结构子组，其字段为 null。

### 使用和限制

训练、恢复、消融、推理和多 seed 汇总命令见根目录 README。默认 FP32/AdamW、5 epoch warmup、cosine decay、辅助权重升温、梯度裁剪；V100 不使用 BF16。`best.pt` 用于固定阈值推理，`last.pt` 保存优化器、scheduler、最佳状态和随机状态。恢复续写到新的运行目录，不覆盖历史；恢复保持原模型和实验协议，允许设备、路径、总 epoch 与 worker 调整。

已执行相关模型/训练/指标测试；完整规格在本地真实字段加随机 ESM 上通过反向检查。当前本机没有实际 ESM 缓存，未启动正式 GPU 实验。GPU 显存/吞吐、正式 Test、多 seed 和机制收益尚待服务器训练验证；合成短训练只能说明流程可运行。

服务器真实运行准备增加 FP32/FP16/BF16 autocast（V100 拒绝 BF16）、GradScaler 状态保存、梯度累积和 `max_steps` 诊断上限。短训练统计使用完整原始 Train，不因优化子集缩小而删除高阶因子或压缩原型容量。代表样本按结构可用性、D 注记、长度和 Train 标签覆盖选取，原划分不变；Validation 仅选输入类型覆盖，不据其标签拟合先验。`runtime_check.py` 通过同一训练入口做实际缓存 GPU profile、短训练和恢复；只有服务器真正运行后才有实测显存/吞吐/Loss 记录，当前不作已验证声明。
