# MFP 下一代模型架构设想书

2026-10-23 更新：当前项目现状、第四模型 **HelixSet** 及可切换评价协议见第 9 节。第 1–8 节保留此前三模型的设计与阶段记录；其中早期“尚未训练”的叙述不代表当前项目状态。HelixSet 两层几何上下文版已完成 major15、seed42 的全量 Train/Val；当前加深到四层的配置仅经静态检查，未启动训练或测试。

> 三条完整路线已在现有项目实现，具体接口、实际容量和评价协议见第 8 节。设计机制的性能与创新收益仍待正式训练及消融验证；前文预算是设计时的估计，不能代替实际参数测量。

## 1. 结论与数据依据

**优先实现 TriRelay：三模态证据交互与标签反馈网络。** 另保留 GeoFactor 异构因子图、ProtoRoute 多原型专家两条独立完整路线。三者在表征组织、信息传播和决策方式上有实质区别。

当前序列模型的单 seed Validation Macro-AP：major15 的 ESM-35M 为 0.45164、ESM-650M 为 **0.47782**；fine39 分别为 **0.43251** 和 0.42167。扩大 PLM 有价值，但不能代替结构、理化及标签建模；这些数值也不足以确定下一代模型的最佳编码器。

现有数据包含最长 74 残基的序列、D 构型注记、Cα 坐标、骨架/接触边、DSSP、φ/ψ/ω、ASA/RSA、部分明确来源的置信度，以及 515 项理化和化学描述符。major15 明确骨架训练集 39,478 条，其中 34,387 条有可用主结构；fine39 相应为 39,455 条和 34,367 条。结构缺失是正常输入情况。详见 [`structure_schema.json`](../data/therapeutic_2025/structure_schema.json)、[`physchem_schema.json`](../data/therapeutic_2025/physchem_schema.json) 和 [`README.md`](../data/therapeutic_2025/README.md)。

理化描述符大多是计算代理量，并非独立实验测量；现有坐标代表选定结构，而非构象轨迹。标签的 0 是未标注，`__unknown` 是亚类未知。不能把未共现解释为互斥，也不能把同一大类下的亚类强制归一化成单标签分布。

## 2. 本地同类研究的结构与评价

本节依据 `docs/references/同类研究论文.zip` 中 9 篇 PDF。所列 checkpoint 规则只记录论文正文中可确认的依据；论文重视某个 Test 指标，不等于按它选 epoch。

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

四模型统一位于 `src/mfp_nova/models/`：`trirelay.py`、`geofactor.py`、`protoroute.py`、`helixset.py`。本节保留前三模型的架构与历史协议，HelixSet 详见第 9 节。包根目录的 `research_layers.py` 共享神经网络积木，`multimodal.py` 共享只读数据与 Train 统计，`research.py` 共享训练/恢复/评价/推理。使用普通 PyTorch，没有模型注册器、图框架依赖或远程调度。

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

后续RTX5070完整seed42训练已完成，major15/fine39最佳轮13/15、早停轮28/30、显存7.20/7.61GiB，MFP检测F1均低于EMA-only。先通过`mfp_nova.private_evidence`导出两份best.pt的完整Val及同权重推理消融。消融在两次读取都将`g * private_evidence`置零，第一轮标签关系、反馈和第二轮query据此重新计算；保留新模型学得的共享投影和全部其他权重，使用同一保存阈值。它检验模型对私有残差的依赖，不等价于重训原架构，不预设移除残差后能改善。运行时开关不进入state_dict、不改变旧权重加载或训练默认值；正式GPU消融结果待回传。

正式Val消融现已回传：full重评复现原记录，去除私有残差使两任务MacroAP分别下降13.10/10.05pp；私有路径被实际使用，但相对原EMA-only的综合净收益仍不足。下一项`--branch-supervision shared_private`将现有0.1分支预算等分给第一次读取的共享/私有证据，替代原来的融合/私有分支监督。两路读取都在关系反馈前，使用同一无关系上下文的label query；共享分类目标没有通往私有投影/读取器/门控的计算图依赖。最终混合分类、直接头、两轮反馈、高阶推理及重建保留，不增加参数。两分支原始Loss单独记录，聚合后仍乘原branch系数及warmup。新实验从头训练，是否改善分工与泛化待Val检验，不将消融依赖等同于新架构收益。

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

## 9. 当前现状与第四模型 HelixSet（2026-10-22）

### 9.1 当前项目已经完成什么

当前项目已有数据接口、冻结 ESM 缓存接口、四种独立多模态架构、EMA、恢复训练、独立评价和五 Seed 汇总。前三模型已有正式实验；HelixSet 两层版已完成 major15、seed42 的全量 Train/Val，四层版待运行验证。默认原样本明确骨架队列：major15 Train/Val/Test 为 39,478/8,534/8,187，fine39 为 39,455/8,529/8,180。结构、D 注记、修饰及 515 项理化描述符已进入研究模型输入，早期 sequence 基线说明中的“尚不消费”不适用于四个研究架构。

三模型、双任务、固定五 Seed 的首轮训练与独立 Test 已完成。以下是**已有 Test 文件中记录的五 Seed 均值（%）**，本次没有重新评价模型或计算预测；完整标准差见[原始汇总](../runs/analysis/rtx5070_five_seed_20261018/test_analysis.json)。Test 数据仅用于说明已完成实验现状，不参与此次架构、阈值或超参数选择。

| 任务 | 模型 | Macro-AP | Acc↑ | Prec↑ | Cov↑ | AT↑ | AF↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| major15 | TriRelay | 52.98 | 60.02 | 64.26 | 67.77 | 48.28 | 6.43 |
| major15 | GeoFactor | 51.98 | 58.02 | 61.66 | 66.76 | 46.11 | 6.88 |
| major15 | ProtoRoute | 53.16 | 60.45 | 64.54 | 68.26 | 48.89 | 6.45 |
| fine39 | TriRelay | 52.44 | 54.50 | 58.39 | 64.00 | 40.73 | 3.35 |
| fine39 | GeoFactor | 50.83 | 54.43 | 58.49 | 63.13 | 41.29 | 3.36 |
| fine39 | ProtoRoute | 53.09 | 54.15 | 58.14 | 63.62 | 40.25 | 3.38 |

ProtoRoute 的 Macro-AP 均值略高，不是所有指标或所有 Seed 的统一赢家。三个模型的集合预测仍明显偏低；AF 较低受多标签稀疏性影响，不能替代 AT 或 Acc。传统五指标不包含 Macro-AP：Acc 是样本 Jaccard，Prec/Cov 是样本 Precision/Recall，AT 是完整标签集合匹配率，AF 是 Hamming Loss。

后续开发应与上述首轮 Test 区分：

- TriRelay 的五 Seed EMA **Validation** Macro-AP：major15 为 55.98±0.27%，fine39 为 54.39±0.61%；相同训练轨迹的在线最佳约为 53.24%/51.79%。对应 EMA Acc 62.53%/55.63%，AT 50.46%/41.06%。来源：[major15](../runs/analysis/ema_five_seed_20261019/analysis.json)、[fine39](../runs/analysis/ema_fine39_five_seed_20261019/analysis.json)。EMA 有一致的排序收益，尚未证明能把完整集合预测带到 70%。
- GeoFactor/ProtoRoute 已有两任务 seed42 EMA Validation 结果，尚不足以替代完整五 Seed 比较；见[已有报告](../runs/analysis/ema_geofactor_protoroute_seed42_20261020/report.md)。
- 最新 TriRelay RMS 竞争式融合的 seed42 Validation Macro-AP 在 major15 达到 57.14%，Acc 63.23%、AT 50.93%；fine39 则出现 Macro-AP/Macro-F1 提高、集合预测/MFP 下降的取舍。它不是双任务统一改善方案，见[已有完整分析](../runs/analysis/trirelay_competitive_seed42_20261022/report.md)。归一化加法方案的源码和既有实现检查已记录在 README，本次未发现其正式性能结论，不能替它填成绩。
- [首次读取诊断](../runs/analysis/trirelay_firstread_seed42_20261022/report.md)表明，增强分支监督后部分分支 AP 大幅提高，最终增益仍小；首次融合高度偏向序列，而且个别标签在下游处理后 AP 回落。这是需要改变信息组织方式的依据，但不是“某一个门控已被证明导致全部退步”的因果结论。

### 9.2 设计取舍与区别

第四模型命名 **HelixSet：残基对交互与多尺度潜变量集合解码网络**。“Helix”是模型名称，不限定输入为螺旋肽。实现与其他三模型统一位于 [`models/helixset.py`](../src/mfp_nova/models/helixset.py)，参数位于 [`helixset.yaml`](../configs/helixset.yaml)。复用本项目的基础序列/属性编码层、数据和训练器；主干、信息传播及解码机制独立于前三模型。单次训练、独立评价、预测、GPU 检查和五 Seed 批量入口均已纳入四模型；此次没有执行模型检查或训练。

| 维度 | TriRelay | GeoFactor | ProtoRoute | HelixSet |
|---|---|---|---|---|
| 主表示 | 三模态流与标签证据 | 残基/片段/属性/标签异构图 | 四专家与共享/专属原型 | 残基状态、显式残基对状态、多尺度记忆和潜变量 |
| 高阶计算 | 标签关系/经验高阶共现 | 标签能量因子展开推断 | 原型匹配后的标签关系 | 残基三角乘法交互，交替 incoming/outgoing |
| 融合 | 模态间 relay、标签反馈 | 多类型图消息传递 | 标签条件专家路由 | 每种记忆分别读取、分别归一化后拼接投影 |
| 标签解码 | 读证据→关系→反馈→再次读 | 一元分数→因子推断 | 专家分数→原型/关系修正 | 读取潜变量与细粒度证据→标签条件残基对读取→标签交互 |
| 直接 ESM 分类 | 研究模型已有各自直接头，并非本列统一含义 | 同左 | 同左 | **没有；ESM 只提供残基输入** |
| 辅助目标 | direct、分支、重建 | 一元、重建、因子正则 | direct、分支、原型、均衡、重建 | 潜变量分类和中间解码层分类 |

设计假设：把高阶容量放在**残基证据形成阶段**，将深层多模态融合与标签解码分开；解码层同时保留细粒度证据读取，减少固定潜变量压缩后少数标签信息丢失。前三模型及其全部代码保留用于对照，不用新模型覆盖旧实现。

### 9.3 完整数据流

```text
冻结 ESM 残基 + AA/D/化学注记
       ↓ 3 层局部卷积/全局注意力，d=384
序列残基 ← 同位置结构几何的置信度调制注入
       ↑ 角度/DSSP 节点经 4 层距离/边偏置的几何上下文更新
       ↕ 4 层残基—残基对更新
       ├─ 48 维学习残基对：序列间隔 + 残基条件乘法/三角更新
       └─ 48 维几何残基对：近/远距离 + 边/置信度 + 质量加权三角更新
               两路分别归一化，几何路乘质量后形成注意力偏置 → 更新残基
       ↓
原残基 + 宽度 3/7/15 有序片段 + 全局化学 token（按尺度校正读取先验）
       │                  25 个理化 token       局部/全局结构 token
       └──────────────────────┼────────────────────────┘
                              ↓ 分源读取、分别归一化、拼接融合
                         32 个潜变量 × 6 层
                              ↓
15/39 个固定语义标签 query × 3 层解码
每层：读取潜变量 + 片段 + 理化 + 结构
          → 以当前标签状态读取显式残基对 → 标签交互 → FFN
                              ↓
                     标签专属权重 → 独立 logits
```

具体实现：

1. **分开的学习残基对与几何残基对。** 有符号/绝对序列间隔投影为 48 维学习 pair state，每层接受残基条件乘法、三角更新和 FFN。几何输入另行投影为 48 维：16 个近距离 RBF（中心 0–20 Å、宽度 1.5 Å）、8 个远距离 RBF（中心 24–56 Å、宽度 5 Å）、`log(1+d)/log(65)` 与 `max(d−56,0)/(d+1)` 两个连续尾部特征、3 类边 one-hot、坐标对可用性、置信度对已知状态及置信度乘积，共 32 通道。距离超过最后一个 RBF 后仍可区分；缺坐标与自对角的显式几何贡献为零。两路三角计算都交替 incoming/outgoing、使用 FP32 累积，分别按真实残基数/有结构残基数归一化。几何三角输入边先乘端点质量乘积，输出再乘目标边质量；其输出线性层无 bias、无 LayerNorm，避免零路径变为非零或弱路径被直接归一化放大。残基注意力将学习 pair bias 与质量调制后的几何 bias 相加。学习残基对可经已融合的残基间接含结构信息，因此不能把它称为严格纯序列流；缺结构时仍正常计算序列关系。三角更新不是三角不等式约束，也不预测坐标。
2. **有序多尺度片段与计数校正。** 宽度 3/7/15 各自使用 depthwise 卷积提取顺序敏感特征，同时取得 mask 归一化均值，再拼接投影。步长为 1/3/7，保留原残基分辨率。窗口中心必须属于真实序列，避免 padding 多出片段。潜变量层和解码层读取片段时，对每个有效 key 加上 `−log(该尺度有效 token 数)`；原残基、三种窗口和全局化学 token 共五组，在内容分数相同时每组总先验质量相同。该校正缓解短尺度因 token 多而获得更大总权重，最终权重仍由内容决定，不是强制各尺度贡献相等。全局 token 也因此获得更大先验份额，收益需消融验证。卷积激活仍是待验证的局部模式。
3. **潜变量融合。** 32 个可学习潜变量在每层分别读取片段、属性和结构，三路输出分别无仿射 LayerNorm 后拼接投影，再做潜变量 self-attention/FFN。不存在三模态竞争 softmax，也不通过重复相加未归一化的共享/私有证据扩大分数。模型仍可能学会忽略某模态，需实验分析，不能因移除竞争 softmax 就声称已解决模态失衡。
4. **四层几何上下文及相对/绝对质量控制。** 原 16 维几何特征（含有符号角、可用性、置信度、暴露度和三重积）与 DSSP 先生成节点，再通过四层结构 attention/FFN 更新；距离/边状态提供 attention bias，只在有效结构节点间传播，FFN 扩展倍数为 2。已知置信度权重 `q=0.3+0.7*confidence`，未知为中性 1，缺结构为 0。上下文 attention 加 `log(q_key)`，消息经无仿射 LayerNorm 后再乘查询质量与平均来源质量。更新后的节点用于同位置几何注入、结构记忆，并以质量加权均值补充全局结构 token。潜变量层和标签层的结构读取同样在归一化后乘平均可靠性。显式几何 pair 的 attention bias、标签 pair 读取和三角路径分别在归一化后或无归一化的输出处施加质量权重，避免把序列 pair 一起关闭。无结构时几何贡献为零；这是启发式调制，不能保证最终 logits 对置信度单调，也不能消除已融合表示中的间接结构影响。采用现有 D 注记及有符号几何信息，不消除镜像差异；本次未运行旋转/平移检查。
5. **标签条件残基对解码。** 每层标签先分别读取四类记忆（潜变量、片段、属性、结构），各自归一化后拼接投影；再用当前标签状态对显式残基对评分，读取端点与 pair state，最后执行标签间 attention。这样 pair state 同时通过残基注意力偏置和标签读取两条路径影响预测，不必只间接压缩到残基或潜变量。残基对证据与标签交互各有初始约 0.1 的可学习残差尺度。结构/属性和片段仍可直接进入标签状态；这些读取位于完整主网络内部，不是直接 ESM 分类旁路。无经验共现矩阵、硬标签个数、互斥 softmax 或标签真值输入；最终仍是独立标签 logits，不是联合集合概率模型。
6. **长尾与监督。** 每标签有独立 query 和输出权重，不按尾类簇数压缩输出残差；继续使用现有 Train 正例权重、弱未标注 ASL 及 fine39 `__unknown` 对应的未标注降权。主损失 + 0.2 潜变量分类 + 0.1 中间解码分类均值；辅助权重沿用 5 轮 warmup。历史统一接口中的 `direct` 在本模型专指潜变量辅助头，`branch` 指前两层解码辅助监督；这些 logits 不与最终 logits 混合。不增加描述符重建、伪负标签、标签数量预测或新数据增强。

残基对读取的分数为 `s(c,i,j) = [q_left(c)·k_left(i) + q_right(c)·k_right(j) + q_pair(c)·z(i,j)] / sqrt(3*d_pair)`，其中 `z=LN(学习 pair)+q_i*q_j*tanh(LN(几何 pair))`，缺坐标时第二项为零。在有效有序残基对上 softmax，排除 padding 与 `i=j`；少于两个有效残基时返回零证据。对同一组权重分别计算两个端点的边缘加权和与 pair state 加权和，再拼接投影。评分、softmax 和加权累积采用 FP32。`pair_interactions=false` 同时置零评分中的 pair 项和读出的 pair value，保留端点评分/端点值、温度、输出维度及完整上游主干；它仅隔离解码器显式 pair state 的增益，不是移除整个模型的成对交互。缺结构时仍可读取序列条件残基对，权重不代表实验接触概率。`model(batch, return_evidence=True)` 可返回最后一层 `pair_attention`，形状为 `[B,C,L,L]`；关闭 `pair_readout` 时为 `None`。常规训练不输出该字段；注意力权重不构成因果解释。

采用 d=384、8 heads、geometry 4 层、pair 4 层、latent 6 层、decoder 3 层。2026-10-23 将独立几何上下文从2层加深到4层，后三部分深度不变；训练使用 microbatch16×累积4，保持有效 batch64。残基三角部分为 `O(B L³ d_pair)`，现在学习/几何两路各计算一次，三角收缩运算量约为此前的两倍；四层几何上下文还增加结构 self-attention/FFN。每层标签对读取为 `O(B C L² d_pair + B C L d)`，通过边缘加权避免创建 `[B,C,L,L,d]` 张量。潜变量以固定 M=32 控制融合成本；当前短肽 L≤74。默认 FP32、microbatch16×累积4、EMA0.999、dropout0.2、weight decay0.01。**实际参数数目、峰值显存和吞吐尚未测量**；不承诺默认 batch 必然合适。本次新增第3、4个几何上下文层的参数；两层版checkpoint不能直接加载为四层版，深度对照应重新初始化训练。前三模型不受影响。

这些修改依据此前仅针对 Train 的数据阅读：major15 的有结构非局部残基对中约 27.5% 距离超过 20 Å、约 8.0% 达到 32 Å，结构残基中约 26.6% 有已知置信度。前两项按残基对计数，不是肽比例。它们支持改进距离覆盖与未知质量处理，不足以证明层数不足或保证性能提升。`geometry_depth=4` 是当前深度对照设置，不是已确定的最优深度。此前两层版在 RTX 5070、major15、seed42、batch32×累积2 下完成40轮，第25轮EMA模型达到 Val Acc63.49%、AT53.25%、Macro-AP55.18%；Macro-AP在第11轮最高57.44%，此后下降。该记录表明存在过拟合，增加深度的收益需要单独验证。四层版应使用新运行名从头训练，不能从两层版checkpoint恢复；批量大小变化也可能影响计算轨迹。

可借鉴机制有明确来源：[AlphaFold 的残基对/三角更新](https://www.nature.com/articles/s41586-021-03819-2)、[Perceiver IO 的潜变量处理](https://arxiv.org/abs/2107.14795)、[Query2Label 的标签条件读取](https://arxiv.org/abs/2107.10834)。本实现独立编写；这些机制本身不是本项目首创。潜在贡献是面向不完整标注短肽的“顺序敏感片段—置信度残基对—分源潜变量—细粒度标签解码”组织方式，是否具有论文级新颖性和性能贡献仍需检索及消融证明。

### 9.4 每次训练选择指标模式

新增 `--metric-suite traditional5|task_specific`，第四模型默认 `traditional5`，前三研究模型不传该选项时维持原 `task_specific` 选模行为。它切换主日志、主结果字段及 checkpoint/早停目标；**不偷偷修改标签、数据、训练损失或指标定义**。

| 模式 | 主报告 | 最佳轮次与早停 | 阈值 |
|---|---|---|---|
| traditional5 | Acc、Prec、Cov、AT、AF | 每轮 Val 拟合阈值后的 Acc 最大；同分 AT 最大；再同分保留更早轮 | 默认每轮 Val 按 Jaccard 拟合全局阈值；最终保存最佳轮阈值 |
| task_specific | Macro/Micro-AP、Macro/Micro-F1、Jaccard、Head/Medium/Tail AP、MFP 检测与功能集合 P/R/F1 | Val Macro-AP 最大；同分 Micro-AP 最大；再同分更早轮 | 最佳轮 Val 按 Jaccard 拟合全局阈值 |

每轮原 `train` / `validation` 及分组指标仍使用固定 0.5 便于观察曲线；传统模式另存 `selection_metrics`、`selection_threshold`、`selection_key`，不要与固定阈值列混用。阈值网格沿用 0.05–0.95/0.01，同分先靠近0.5再较小值。`per_label` 仍是可选独立实验，沿用 Val 逐标签 F1 阈值；传统模式仍以该阈值下的样本 Acc 选轮，而不是声称逐标签 F1 阈值等于全局 Jaccard 最优。

EMA 和在线参考按同一指标模式各选最佳轮；训练停止由主评价轨迹控制。`results.json.primary_metrics` 和独立评价的 `primary_metrics` 保存所选主报告，完整指标保留在 `validation`/`metrics` 中。`metric_protocol` 记录选模与阈值定义，checkpoint 配置携带模式；恢复训练必须沿用原模式。`best_macro_ap` 表示所选最佳轮的 AP，在传统模式下不代表全程最高 AP。不同模式不能混成同一组五 Seed 均值。

### 9.5 70%目标、验证边界与后续比较

将目标明确为：**主要集合指标 Acc/Prec/Cov 达到 70%，AT 同样以 70% 为挑战目标；任务特定模式关注 Macro-AP/Micro-F1 达到 70% 并报告 MFP-F1；AF 应降低，绝不以 AF≥70% 为成功。** 不用 `1−AF` 冒充传统 AF，不把 Micro-AP 已超过70%解释为所有关键指标达标。提高阈值或换选模目标不能直接提高 AP，也不能保证 Acc、AT、Precision、Recall 同时达到目标。

同类论文的 75%–80% 不能直接移植为本项目预期值：必须核对标签数/含义、样本队列、同源划分、未标注处理、训练数据与阈值口径。原始数据论文和本项目修复队列也不是完全相同评价对象。此处保留现有划分，不能为追逐指标改用随机拆分、删除难标签、用 Test 调参或选最好的 Seed。

两层版已由用户完成一次真实GPU训练，回传了配置、完整逐轮记录及结果；本地只读取这些记录，未重新加载checkpoint或复算预测。四层版尚未经运行验证，不能沿用两层版性能结论。用户授权运行以后，需要首先确认 mask/padding、缺结构、D 注记、梯度和新选模/恢复链路；通过相关实现检查后才有意义开展完整 Validation。正式比较固定 seed42/101/202/303/404，统一任务、编码器、划分、评价模式和阈值策略；当前配置正则及有效 batch 等与旧模型有差异，整体性能比较不能直接当作单一模块因果贡献。

必要消融使用现有 `--ablate` 接口，下列布尔开关默认均开启：

| 关闭开关 | 关闭的路径 | 保留的路径及比较含义 |
|---|---|---|
| `triangles` | incoming/outgoing 三角乘法 | 残基条件 pair 更新、pair FFN、残基注意力偏置与标签对读取仍保留 |
| `pair_readout` | 标签直接读取残基对 | pair 主干与残基注意力偏置仍保留，用于区分直接读取的贡献 |
| `fine_evidence` | 解码层直接读取片段、理化和结构 | 潜变量仍读取全部模态，标签对读取独立保留 |
| `label_context` | 标签间 attention | 所有输入证据读取保留 |
| `geometry_context` | 四层独立几何上下文更新 | 初始几何节点、全局结构描述符、几何 pair 路径仍保留 |
| `geometry_quality` | 人为的结构质量乘法及 log-prior | 可用性 mask 保留，未知/已知置信度输入仍保留；用于比较显式质量权重的贡献 |
| `distance_multiscale` | 8 个远距离 RBF 和两个连续尾部通道（置零） | 原 16 个近距离 RBF、所有参数维度和其他特征保留；不是旧架构完整复现 |
| `scale_balance` | 每尺度有效 token 数的先验校正 | 多尺度 token、mask、卷积及内容 attention 保留 |
| `pair_interactions` | 标签对读取中的 pair-state 分数和 pair value | 端点读取及上游 pair 主干仍保留；需保持 `pair_readout=true` 才有比较意义 |

同时关闭 `fine_evidence pair_readout` 才是“标签只读取潜变量”的对照，不等于移除结构/理化输入。geometry/pair/latent 深度与中间监督系数可通过配置改变；不要把“关闭计算但保留参数”的消融当成等参数训练成本比较。优先固定主干深度，比较完整模型与新增机制的单项消融，再决定是否需要更深的结构网络。这些开关及新增读取均未运行验证。Test 在方案确定后仅作独立最终评价；既有 Test 已被历史开发者查看，应如实披露这是固定历史保留集，不能声称从未接触过的新盲测。
