# MFP-Nova V3.0 基础架构研究

2026-11-16。HH 系列暂停。V3.0 以氨基酸序列为基础模型的主要研究对象。首轮用现有项目骨架序列直接训练多标签预测。化学注记和实测变体进入后续独立对照。

## 1. 从 TheraPep-AI 到 HH，瓶颈在哪里

| 阶段 | 实际引入的能力 | 同协议内能支持的判断 |
|---|---|---|
| TheraPep-AI | 多尺度 CNN 和统计合成背景训练。作者发布任务含 48 个功能位及治疗肽判别 | 局部序列模式是强基线。背景拒识与已测无活性属于不同问题，作者集与 Fine39 的指标分开解读 |
| HelixSet | 双 PLM、理化和单结构输入，标签条件证据及残基对建模 | 项目统一传统口径下，Fine39 Test 58.78%→59.03% Jaccard，Macro AP +5.49 pp、尾类 AP +10.97 pp。Major15 Test Jaccard 64.34%→64.23%。复杂路径主要重分配了收益 |
| TheraSet | 保留化学 CNN 直接头，新增序列、属性、几何和原型读取用于分数修正 | 同一 Fine39 Validation 上，历史化学 CNN 57.78%、完整设置 58.14%、去标签上下文后的单原型设置 61.05%。多原型和标签上下文缺少保留理由。61.05% 是当前最强单模型开发基线 |
| HH V1.0–V1.7 | 在已有表示或分数上反复试验软分解、关系、编辑和联合集合预测 | V1.6 单分量／完整联合头为 59.52%／59.48%。V1.7 为 61.224%，比 61.046% 高 0.178 pp，区间跨零。验证集中预测集合改变了，稳定超过基线的证据仍缺 |
| HH V2.0 | 新建轻量 CNN，尝试家族辅助、局部读出和全肽 PLM 融合 | 首轮直接头 47.96%，家族与粒度辅助 42.25%，完整 44.18%；恢复原化学 CNN 到 56.72%。匹配实验中加入 ESM／位置融合为 52.47%，下降 4.25 pp。残基候选头单独 56.72%，补标签后 61.03% |

[作者 TheraPep-AI 项目](https://github.com/terra-quantum-public/tq-therapep-ai) · [TheraPep-AI 预印本](https://doi.org/10.64898/2026.06.07.730473) · [HelixSet 同口径 Test](TheraPep-AI-与五模型性能比较-2026-10-31.md) · [TheraSet 消融](TheraSet-Ablation-Analysis-2026-11-08.md) · [TheraSet 后续结果](TheraSet-Followup-Analysis-2026-11-08.md) · [HH V1.7](HH_SetReread_Architecture_and_Training_Plan.md) · [HH V2.0](HH_SetReread_V2_Architecture_and_Training_Plan.md)

**已有模型有架构区别，判别信号的增量有限。** 几代模型主要围绕同一批短序列、来源相近的 PLM、代理结构和不完整记录重新读出或融合。V3 优先研究**序列中的局部模式、残基顺序及跨跨度上下文如何共同区分不同功能**。已有结果不足以把平台瓶颈全归于架构，或估计 70% 的可达性。

同批 seed 42 的 CNN 对照里，Fine39 固定 0.50 时，加入化学注记使 Jaccard 从 **56.19% 到 57.78%**、Macro AP 从 **51.74% 到 55.72%**。traditional5 下 Jaccard 从 **57.01% 到 58.80%**，两组分别使用 0.70 和 0.73。化学注记有增益线索，现有消融无法分开 D/L、端基及非天然单体的贡献，固定阈值下，按验证序列簇重采样的条件 95% 差值区间为 +0.77 至 +2.37 pp，跨 seed 效果也待复核。后续在同一序列主干上测独立增量。 [TheraSet 消融](TheraSet-Ablation-Analysis-2026-11-08.md)

## 2. 首轮排除的机制

| 机制 | 项目证据及处理 |
|---|---|
| 增加标签全局共现、标签因子、多原型、集合重排 | TheraSet 去因子／单原型更强，HH 多轮重排与联合集合头未稳定改善。V3 直接输出标签，无标签图和候选编辑 |
| 编码后把全肽 PLM 摘要加给所有标签 | 匹配化学 CNN 对照为 −4.25 pp，两个标签组为 −9.83 pp。V3 首轮单路氨基酸序列编码 |
| 新增注意查询、log-mean-exp 候选或更多分段后读出 | 新候选与纯 CNN 的漏标前 3 覆盖分别 2,029 与 2,036 条，补标签 ACC 61.03% 与 61.22%。首轮保持统一直接读出以隔离编码收益 |
| 家族辅助与记录粒度同时塑造共享表示 | V2.0 辅助组多报 6,539 个未标注位，仅多命中 1,195 个标注正位。V3 固定任务映射，测定信号按实际对象和端点进入独立训练目标 |
| 静态结构全面参与配对、完整三角更新、低置信近邻填补 | 已有去 pair、去结构的改善证据不足以证明可靠净收益；结构有无也可能标记来源。V3 首轮只沿序列建立局部关系 |
| 同源近邻、随机突变、拼接、伪标签补齐功能 | 同序列可有不同的实体，同源变体会有活性悬崖。保留真实观测，训练增强只允许同一实体的等价序列化 |

## 3. 本地数据能支持哪些假设

读取当前 Fine39 原始 Train 39,455 条（25,244 簇）与 Validation 8,529 条（5,010 簇），按 row_id 对齐记录、canonical backbone、HELM 与目标。此次统计未用 Test。

| 记录层面的可用信息 | Train | Validation |
|---|---:|---:|
| HELM 可解析且除常见端基后核心长度与骨架一致 | 39,446 | 8,529 |
| 存在多字符核心单体 token | 839 | 174 |
| 非 Gly 的小写手性标记 | 851 | 173 |
| HELM 连接字段有内容 | 22 | 4 |
| 端基注记与独立字段欠一致（包含独立字段空缺） | 37,846 | 8,208 |
| 主结构记录可用 | 34,367 | 7,489 |

相同项目解析骨架而 HELM token 或连接字段不同的组，在 Train 有 128 组／273 行，Validation 有 35 组／77 行，其中 27 个验证组的记录标签集也不同。它们适合检查表示能否区分记录，不能直接监督活性差值。将模型改善局限在这 77 行，即使用满每行 1.0 Jaccard 的理论增量，整体也最多 +0.90 pp。单靠环化或非天然分支覆盖不到 70% 目标所需的整体提升。项目的 actual_chemical_entity_verified 字段全为 False，记录表示的是有出处可查的化学注记，不能升级为完整实测分子身份。 [本地记录](../data/therapeutic_2025/records.csv) · [数据处理报告](../data/therapeutic_2025/docs/therapeutic_2025_完整处理与前后对照报告.md)

当前两菌株 MIC 面板有 1,059 条测量记录，表中标记实体精确匹配的 Train 子集含 134 条记录、48 个不同肽；双菌株四状态标签表仅 35 个肽。按 PMID/文章 ID、菌株、原始单位、已记录介质、接种、pH、盐条件分组，再筛同长度单残基差异，发现 0 对适合直接用于该单点变体训练的候选。不同报告条件合并会产生近似对，但不能用于主结论。现有面板只适合辅助行为审查，后续可选的实测差值监督需要额外获取原始测定。 [MIC 记录](../runs/analysis/gram_mic_panel/mic_measurements.tsv) · [双菌株状态](../runs/analysis/gram_mic_panel/four_state_support.tsv)

**近期可用的外部入口是 DBAASP V4。** 它保留肽实体、SMILES、修饰、菌株、浓度、条件和出处，也覆盖抗生物膜、抗癌与溶血等端点。先从原始 assay 提取同研究系列的相近变体，优先匹配同一单位、对象和报告条件，保留大于／小于形式的删失。保留与旧划分之间的实体、同源簇及来源关系，接触 Val/Test 邻居的外部变体不用于训练。PathoMIC 预印本报告 74,751 条记录、424 个病原种，可用于定位广覆盖的来源；其序列—病原聚合结果需要回溯原始条件才能用于差值监督。本次尚未下载构建扩充训练集。 [DBAASP V4 原文](https://doi.org/10.1093/nar/gkag1050) · [PathoMIC 作者预印本](https://arxiv.org/abs/2608.26228)

## 4. V3.0 架构提案

**让氨基酸序列的多尺度上下文决定相邻模式如何组合，由新主干直接输出所有功能。** 首轮检验序列本身能否提高多功能肽和低同源肽的标签排序。高可信修饰和同测定变体在后续单独比较。

### 计算机多标签论文如何改变 V3

| 本地编号与实质机制 | 迁移到肽的取舍 |
|---|---|
| 001 Two-Way Loss、009 DRO、069 MulSupCon。分别比较单个样本上的多个标签、单个标签上的多个样本，或按标签重叠构造对比 | 取双向比较，替换比较依据。同对象下按实测差比较两个肽，同一变体对上的差值按对象比较。Fine39 的全部零位不参与硬负排序，两条肽也不因共享一个已记录标签就在所有功能维度上拉近。 |
| 002 PatchCT、017 PIAA、054 RAM。分别做局部到标签的双向匹配、恢复 patch 判别信号及知识约束匹配 | 取“次要信号要在全局汇聚前形成”。让逐残基状态先与不同跨度的序列上下文交互，优先测序列本身能否改善第二、第三标签排序。上一轮残基候选改的是读出，补标签后只有 61.03%。额外标签查询、传输损失和候选头留作可分辨的对照。 |
| 003 相关／判别双分组、011 虚假任务相关、016 ProCI。分别以共现和判别关系成组、用前后重采样识别相关漂移、用上下文原型近似混杂 | **用实测的“变化或保持”替代共现先验。** 同一编辑在对象 A 改变活性、在对象 B 的差值低于重复测量分辨率时，监督只改变 A 的响应。B 的保持也必须来自同测定数据。标签图与混杂原型在第一版退出，按簇和出处诊断共现漂移。 |
| 020 DARC、064 Patching、065 Causal Label Correlations。分别利用额外标注或受控替换区分对象本身与共现场景 | 把干预单位具体化为已测肽变体。以同研究、同对象、同端点和相容条件限制比较。匹配只能减轻部分混杂；序列遮挡和残基替换只作模型灵敏度检查，标注不自动继承。 |
| 006 BridgeGap、007 SPA、038 SalSPML、045 CRISP。分别处理缺标对归因、概率校准、选择性记录和类先验的影响 | 明确预测目标分成 Fine39 记录概率和测定响应。没有测定覆盖时，缺测不构造硬负，记录频率不代替活性先验。保持弱标签协议做架构比较，另评校准。 |
| 015 MambaML、018 F-measure、072 动态标签顺序与 084 期望测试效用 | 分别作为序列算子、预测排序或效用对照；已有 HH 多轮更换后端，优先把实验预算用于提升新输入上的判别和留出簇泛化。 |

[Two-Way Loss](https://openaccess.thecvf.com/content/CVPR2023/html/Kobayashi_Two-Way_Multi-Label_Loss_CVPR_2023_paper.html) · [PatchCT](https://openaccess.thecvf.com/content/ICCV2023/html/Li_PatchCT_Aligning_Patch_Set_and_Label_Set_with_Conditional_Transport_ICCV_2023_paper.html) · [标签双分组](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_Correlative_and_Discriminative_Label_Grouping_for_Multi-Label_Visual_Prompt_Tuning_CVPR_2025_paper.html) · [虚假任务相关](https://openaccess.thecvf.com/content/CVPR2025/html/Chai_Identifying_and_Mitigating_Spurious_Correlation_in_Multi-Task_Learning_CVPR_2025_paper.html) · [PIAA](https://proceedings.mlr.press/v306/wang26am.html) · [DARC](https://openaccess.thecvf.com/content/CVPR2026/html/Liao_DARC_Dual_Adjustment_Reasoning_with_Counterfactuals_for_Trustworthy_Chest_X-ray_CVPR_2026_paper.html) · [Patching](https://proceedings.mlr.press/v235/xie24i.html)

一个可直接检验的多标签归纳偏置如下。定义 $M_{pa}=1$ 表示变体对 $p$ 在对象 $a$ 上两侧都测了，且比较条件相容。固定 $a$ 沿分子方向比较观测到的有符号差值；固定 $p$ 沿对象方向比较 $\Delta_{pa}-\Delta_{pb}$，只在 $M_{pa}M_{pb}=1$ 时计算。保持项只来自已测变化落在相应测定误差范围内的对，仍为区间约束。两类损失回传到第 4.1 节的局部编码器。在第 5 节的外部实验中分开消融绝对值、单对象差值、跨对象差值；单对象收益与跨对象迁移分别报告。若后者只改善 MIC、Fine39 多标签排序无变化，便不能归为本项目的多标签贡献。当前面板没有达标变体对，第一轮仍可测主干比较。

### 4.1 主模型：有序间距带和双中心化

**首轮设计假设。** 同一条肽中多个局部模式的排列可能影响哪几个标签同时成立。按各间距带的两端特征构造双中心化的联合响应。联合项先写回残基表示，再由统一直接头产生全部标签。首轮优先检验三件事，依次为扩大感受野、有序乘性交互和按端点双中心化。

#### 输入和局部编码

输入为原项目解析的氨基酸骨架和有效位 mask，长度上限为 80。使用 AA34 查表，即 20 维氨基酸身份与 14 维由身份确定的固定属性。作为输入匹配控制，全部首轮模型使用同一个查表和 TheraSet 序列 CNN。CNN 每层包含核 3、5、7、9，每尺度 128 通道，共两层，得到 $H\in\mathbb R^{L\times512}$。池化前保留所有残基。最大局部感受野是 17 个残基，尺寸来自本地基线作为受控起点。保留 512 通道直到最终池化，避免把 V2.0 曾经同时压缩宽度和池化路径的问题引入对照。AA34 是否优于可学词嵌入以后可单独比较。 [TheraPep-AI 作者实现](https://github.com/terra-quantum-public/tq-therapep-ai) · [本地 MotifCNN](../src/mfp_nova/models/theraset.py) · [AMCL](https://doi.org/10.1186/s12915-025-02273-0)

#### 骨架与长度约定

保留已解析的 canonical backbone 和原划分。Fine39 中经 HELM 无歧义恢复骨架的行数为 Train 283、Validation 60、Test 43，其余为原始 20-AA 记录。一律按残基身份编码会折叠手性和部分修饰差异。Validation 中相同大写骨架且记录标签集不同的有 74 组、191 行，可用作输入歧义诊断，差异也可能源于测定和记录范围。

Fine39、Major15、Rich48 的 Train 均仅 1 条超过 50，长度为 74；三个任务的 Validation 和 Test 最大均为 50。Fine39 该样本 ID 为 145450，唯一标注是 immunoactive__unknown。V3 首轮保留 80 残基的输入上限与 74 残基的完整样本，按批补齐和使用 mask。以 batch 64 为例，唯一超长样本每轮通常只影响一个批次的补齐宽度，因此没有必要只为该样本统一压到 50。若要复现文献的 50 长度规范，单独报告筛除或真实裁剪的策略；首尾拼接会引入一条原序列不存在的邻接，对 V3 的间距交互尤其敏感。

旧 ESM 路线的 encoder.max_length 对含特殊符号的 tokenizer 序列计数。当前编码器超限会报错，80 是旧缓存的配置值，不能通过直接改成 50 来实现保留 50 个残基。V3 纯序列首轮不依赖该缓存。 [骨架解析](../data/therapeutic_2025/scripts/multimodal_common.py) · [现有动态补齐](../src/mfp_nova/multimodal.py) · [旧编码器 token 限制](../src/mfp_nova/encoder.py) · [TPpred-LE](https://doi.org/10.1186/s12915-023-01740-w) · [METFAN](https://pubs.acs.org/acsodf/article/10/44/53019/3658208/METFAN-Multisource-Enhanced-Therapeutic-Peptide)

#### 有序交互从早期局部特征读取

两层 CNN 的 512 维残基图仍用于直接读出。交互块以第一层的核宽 3 和 5 的两组残基图为输入，拼成 $G_i\in\mathbb R^{256}$，得到两个角色向量 $u_i=\tanh(W_U\mathrm{LN}(G_i))$、$v_i=\tanh(W_V\mathrm{LN}(G_i))$，秩取 32。第一层最大感受野为 5。将局部特征来源前移，是因为 Fine39 Validation 的 5,296/8,529 条肽长度不超过 17，第二层最宽分支的中央特征已经可能覆盖整条肽。两套角色投影保留顺序差异。局部窗口仍可能重叠，响应属于序列模型特征。 [Bilinear CNN](https://openaccess.thecvf.com/content_iccv_2015/html/Lin_Bilinear_CNN_Models_ICCV_2015_paper.html) · [本地序列 CNN](../src/mfp_nova/models/theraset.py)

#### 用连续间距带覆盖 1 至 16 个残基

建议起点为 1–2、3–5、6–9、10–16 四个带。在 Fine39 Validation 上，旧的五个离散跨度 1/2/4/8/16 覆盖 512,304 个合法位置对，占 1–16 范围全部 1,349,288 个位置对的 37.97%。固定跨度与连续带只是采样策略。区间内的平均会压缩具体间距，强响应仅存在于单一位移或区间里响应异号时可能被削弱。若发现这一现象，再比较完整的 16 个精确位移。最长间距带 10–16 在 6,036 条验证肽上有效。 [iFeature 间隔残基对](https://academic.oup.com/bioinformatics/article/34/14/2499/4924718) · [BPNet 顺序与间距评估](https://pmc.ncbi.nlm.nih.gov/articles/PMC8812996/)

#### 在每个带的可用端点上做双中心化

前一版 $q_{ij}-\mu(x)$ 可以校正打乱局部状态后的期望，但若左端通道在各位置都是常数 $u_i=a$，它仍可能输出随 $v_j$ 变化的一阶项。这不足以区分**联合响应**与单端强响应。首轮修订为双中心化。对间距带 $b=[l,h]$，记 $E_b=\{(i,j):l\leq j-i\leq h\}$，集合仅含真实残基；$d_b^+(i)$、$d_b^-(j)$ 分别为各端点出现的次数，$N_b=|E_b|$。对两个角色按各自在该间距带出现的次数取均值：

$$
\bar u_b={\sum_i d_b^+(i)u_i\over N_b},\qquad
\bar v_b={\sum_j d_b^-(j)v_j\over N_b},\qquad
e_{ij,b}=(u_i-\bar u_b)\odot(v_j-\bar v_b).
$$

这样，任一端的某通道在可用位置上是常数时，该通道的交互严格归零。任一端整体平移一个常向量，交互保持不变。对 $E_b$ 所有边求平均等价于原乘积均值减去两个端点均值的乘积。左右端各自计数能处理首尾可用位置本身的不对称。在给定肽的合法位置对上，这是按通道计算的协方差；它可能仍受序列长度、端点模式和窗口重叠影响。高于均值与低于均值的共同变化都可贡献正值，正值不能单独解读为两个基序同时存在。只有一条边时，两端恰为自身均值，交互为零。强全局丰度和单端证据由原直接路径保留。 [二阶视觉表示中的协方差](https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Is_Second-Order_Information_ICCV_2017_paper.pdf)

#### 区间求和、回写和并行预测

对于左端 $i$，该带的交互均值可写成 $(u_i-\bar u_b)\odot(\operatorname{mean}_{j:\ (i,j)\in E_b} v_j-\bar v_b)$。右端写回采用对称的区间求和。同一角色的序列前缀和用两次索引取得一个连续区间的和，无需构造所有实际位置对。空区间和空带贡献为零，只对有邻居位置归一。以 32 维角色向量、4 个带和左右两组 512 维回写投影计算，新增核心参数约 49,408（不含 LayerNorm），计算量为 $O(BL(dr+Kr))$，其中 $K=4$，$d=512$。两组回写使用无偏置线性投影并零初始化，保留直接路径的初始输出。使用已有通道的最大池化与 $512\to1024\to512\to39$ 独立 sigmoid 分类头。在固定算子下验证过区间前缀和与显式逐对运算数值等价。训练速度和显存需实测。 [本地使用前缀累积的 MotifSpan](../src/mfp_nova/models/motifspan.py) · [PIAA 的局部信息保留](https://proceedings.mlr.press/v306/wang26am.html)

与已有 MotifSpan 的差别是取早期序列局部图，在实际的间距区间对两个角色各自去均值，再写回原序列位置。MotifSpan 已经做了有序关系及中心化，采用软分段和距离衰减关系，标签读出负责分数纠正。上述差别用同参数消融检验。二阶统计、残差和前缀和本身已有先例。

#### 监督、验收和同参数消融

- 首轮保持项目现有弱标注 ASL。阳性是记录标签，未记录按既有弱权重处理。同一划分、相同 seed、同一优化策略和固定 0.50 阈值，按 Validation 样本 Jaccard 选 epoch。保留共享的 R-Drop／EMA 作为训练配方时，所有组按相同日程开启。来源或测试标签不进入特征。Two-Way 和 DRO 启发比较样本内及标签内排序，现有 Fine39 未测位不能作为硬负。 [ASL](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html) · [Two-Way](https://openaccess.thecvf.com/content/CVPR2023/html/Kobayashi_Two-Way_Multi-Label_Loss_CVPR_2023_paper.html) · [Fine39 当前弱损失](../src/mfp_nova/research_layers.py)
- 一组纯 CNN。其余三组使用相同的早层状态、间距带、回写通路和参数，分别使用中心化加法 $(u_i-\bar u_b)+(v_j-\bar v_b)$、未中心化乘积 $u_i\odot v_j$ 和双中心化乘积 $e_{ij,b}$。若完整组只超过纯 CNN，结论限于新增传播通路。若还超过加法与原乘积组，再用共享的模型结构比较对称关系，或只置乱交互块的序列位置并逆映射回写，检查是否依赖原始的排列。置乱只作敏感性诊断，无功能标签继承。
- 主表保留整体 Jaccard、Macro AP、Micro F1 和独立多功能 F1。分层报告单标签、二标签及三标签以上，核对前五至少命中二／三个标签的行数以及原完全匹配保留。Fine39 Validation 有 407 组相同氨基酸组成但不同顺序的解析序列，合计 1,136 个不同序列。80 组的记录标签集也有差异。该切片用于诊断顺序，仍可能有化学或标注差异。配对比较沿用序列簇重采样，并用相同一组 seed 检查复现。最终 Test 留作独立评价。 [本地 TheraSet 分层比较](TheraSet-Followup-Analysis-2026-11-08.md) · [PIAA](https://proceedings.mlr.press/v306/wang26am.html)

### 4.2 后续实验用测定差值监督编码

先做逐条有测量的条件回归，条件 $a$ 至少包含可核验的目标对象和测定端点，剂量单位和上下界在数据层处理。回归读出用节点贡献的均值，并共享上游编码器。

$$
s(x,a)=b(a)+L^{-1}\sum_{i=1}^L w(a)^\top h_i(x).
$$

对于同长度、已对齐且同条件的两个实体 $x,x'$，定义 $\Delta_a=s(x',a)-s(x,a)$。在确定性前向中，若某节点的全部受感邻域在两侧完全一致，其贡献严格相消。这来自逐节点局部传播和共享参数。训练配对时两侧关闭随机 dropout 或共享随机掩码，避免差异由随机性引入。对已测差值使用 Huber；只有测量上下界能稳健确定方向时，才用方向约束。反向配对自动得到相反差值。这里的严格相消要求测定读出只使用未引入全肽中心化的局部状态。第 4.1 节的带内角色均值可能受远处残基影响，局部编辑也可能改变该参照，因此该证明不能直接用于完整中心化主干。将来若研究局部可解释的效应，可从中心化前的状态做此独立回归读出。

对同一变体对同时测了对象 $a,b$ 的情况，可监督 $\Delta_a-\Delta_b$ 对应的两种测定变化之差。该量抵消共同的活性变化分量，检验模型是否把选择性变化错学成普遍增强。首轮先核计四格齐备且协议相容的系列；同为 MIC 的数据先按条件形成各自的对数浓度比，异种端点不直接相减。序列单点替换、立体翻转和端基变化分组报告。Fine39 的未标注目标仍沿用弱权重，不以 MIC 结果推断整个 Gram 大类的阳性或阴性。

### 4.3 创新边界

化学图、边条件卷积、肽专用预训练、单目标突变差值、靶选择性悬崖都已有先例。PepLand 和 HELM-BERT 是实体表示对照。SA-MTP 是结构条件图与标签感知头的近邻。最接近差值监督的是小分子 UCN，该方法已经用成对活性差与不同子结构训练图模型，推理时无需配对。蛋白变体基准也已有相似的 held-out 检验。V3 主线的可检验贡献是局部序列模式与跨跨度上下文的结合能否改善**同源簇留出的多功能判别**。化学表示和实测监督的增量分别检验。序列条件化、卷积、差值损失等组件不作为首创主张。 [ECC](https://arxiv.org/abs/1704.02901) · [PepLand](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315545/) · [HELM-BERT](https://doi.org/10.1021/acs.jcim.6c00451) · [SA-MTP](https://doi.org/10.1093/bib/bbag361) · [UCN](https://doi.org/10.1186/s13321-023-00733-9) · [Matched Peptides](https://pmc.ncbi.nlm.nih.gov/articles/PMC4658635/) · [选择性悬崖](https://pubmed.ncbi.nlm.nih.gov/26358784/) · [AMPCliff](https://doi.org/10.1016/j.jare.2025.04.046) · [肽突变扫描与膜选择性](https://pmc.ncbi.nlm.nih.gov/articles/PMC12044605/)

## 5. 第一轮实验怎样回答问题

1. **先比较序列编码。** 按第 4.1 节的纯 CNN、同参数加法、未中心化乘积及双中心化乘积四组比较。原数据仍用 Fine39 同源簇划分。历史序列 CNN 的 56.19% 是可量化的起点，完整 TheraSet 的 61.05% 是开发参照。两份历史成绩都需要与本轮的统一协议结果区分。
2. **验证实测监督。** 外部训练对准备好后，用同一新主干分别比较仅绝对测量与加入成对差值，控制测量集合、训练步数和 Fine39 微调配方。两对象同测样本覆盖足够时，单列选择性目标的增量。记录差值误差、方向正确率、常规样本与悬崖分层结果。对照强基线含简单描述符和同一模型绝对分数相减，避免把共有支架记忆当机制。 [AMPCliff](https://doi.org/10.1016/j.jare.2025.04.046) · [成熟药物发现活动悬崖基准](https://doi.org/10.1038/s42004-024-01204-4)
3. **测总体迁移。** 主指标是原同源簇 Fine39 的样本 Jaccard；同时看 Macro AP、Micro F1、独立多功能 F1、第一标签及前五中的第二、第三标注标签、原已完全匹配的保留。按长度、化学记录和功能数分组，特别检查普通线性肽是否获益。正式比较复用相同 seeds 和划分；Test 仅在模型和阈值选定后独立评价。70% 是性能目标。达到它要求多数样本上的泛化收益，不能用小分组提升或真值挑选上限替代。

## 6. 实现规格与首轮训练

模型入口为 CLI 的 peptide_v3。输入为现有 canonical backbone 和 mask，按原始 Train/Validation 行训练与选模。每个批次动态补齐，保留 74 残基的样本并允许最长 80 个残基。训练、评估和单序列推理直接使用残基，不需生成语言模型特征。Fine39 是首轮研究任务，Major15 与 Rich48 可进入相同模型和数据入口，各自使用原标签顺序及划分。

| 模式 | 每个间距带的关系算子 | 归因 |
|---|---|---|
| cnn | 原 512 维卷积表示直接池化 | 相同输入的序列基线 |
| additive | 双中心化后的两端相加 | 新增多跨度传播和参数量 |
| product | 两端原始特征相乘 | 增量来自乘法的候选对照 |
| centered | 双中心化后的两端相乘 | V3.0 完整模式 |

加法与两组乘法使用相同的早层特征、间距带、回写方式和参数数量。每个端点先按各带的有效邻居数取均值，再对可用间距带取平均。带权重按方向和特征通道学习，初值为 1。最后两组回写投影分别将两端 32 维表示投到 512 维并加回卷积图。FP32 前缀累积和均值根据 mask 计算。空的间距带贡献为零。

首轮各组从头训练，同一 seed 42、100 轮上限、batch 64、FP32、学习率 0.0002。前 5 轮预热，patience 15，第 6 轮起分别开 EMA 和渐增 R-Drop，执行固定阈值 0.50 与 Validation Jaccard 选模。每组保存原行号和标签顺序的逐样本预测。首轮可先核对效应方向，独立 Test 在进一步确认后使用。默认配置记录这些数值。

### V100 三卡后台

在空闲物理 GPU 上运行。第四组会在同一张卡的第一组完成后启动。

~~~bash
cd ~/wuzhencheng/projects/MFP-Nova
conda activate MFP-Nova
GPUS=(0 1 2)  # 改成实际空闲的物理卡号
CPU_THREADS=6
TASK=fine39
SEED=42
RUN="peptide_v3_${TASK}_s${SEED}_$(date +%Y%m%d_%H%M%S)"

launch_v3() {
  local gpu="$1"
  shift
  nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$CPU_THREADS" \
    MKL_NUM_THREADS="$CPU_THREADS" PYTHONPATH="$PWD/src" \
    MFP_RUN="$RUN" MFP_TASK="$TASK" MFP_SEED="$SEED" bash -c '
    set -e
    for mode in "$@"; do
      printf "START %s %s\n" "$mode" "$(date)"
      python -u -m mfp_nova.cli train --model peptide_v3 --task "$MFP_TASK" \
        --v3-mode "$mode" --seed "$MFP_SEED" --device cuda:0 --precision fp32 \
        --batch-size 64 --epochs 100 --metric-suite jaccard \
        --name "$MFP_RUN/$mode" > "runs/$MFP_RUN/$mode.log" 2>&1
      printf "DONE %s %s\n" "$mode" "$(date)"
    done
  ' _ "$@" > "runs/$RUN/gpu${gpu}_queue.log" 2>&1 < /dev/null &
  printf "GPU=%s PID=%s RUN=runs/%s\n" "$gpu" "$!" "$RUN"
}

if git pull --ff-only origin main && mkdir -p "runs/$RUN"; then
  launch_v3 "${GPUS[0]}" cnn
  launch_v3 "${GPUS[1]}" additive
  launch_v3 "${GPUS[2]}" centered product
fi
~~~

### RTX 5070 单卡顺序

~~~powershell
Set-Location 'D:\Data\Code\MFP-Nova'
conda activate MFP-Nova
git pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw 'Git 同步失败' }
$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '6'
$env:MKL_NUM_THREADS = '6'
$env:PYTHONPATH = "$PWD/src"
$task = 'fine39'
$seed = 42
$run = "peptide_v3_${task}_s${seed}_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Path "runs/$run" | Out-Null
foreach ($mode in 'cnn', 'additive', 'product', 'centered') {
    python -u -m mfp_nova.cli train --model peptide_v3 --task $task `
      --v3-mode $mode --seed $seed --device cuda:0 --precision fp32 `
      --batch-size 64 --epochs 100 --metric-suite jaccard `
      --name "$run/$mode" 2>&1 | Tee-Object -FilePath "runs/$run/$mode.log"
    if ($LASTEXITCODE -ne 0) { throw "训练失败 $mode" }
}
~~~

### Fine39 验收与打包

在启动批次的 V100 终端执行。四组均写出正式结果后运行比较和打包部分。

~~~bash
for mode in cnn additive product centered; do
  if test -s "runs/$RUN/$mode/results.json"; then
    printf "DONE %s\n" "$mode"
  else
    printf "PENDING %s\n" "$mode"
  fi
done

PYTHONPATH="$PWD/src" python scripts/compare_validation_predictions.py \
  "runs/$RUN/additive" "runs/$RUN/centered" > "runs/$RUN/centered_vs_additive.json"
PYTHONPATH="$PWD/src" python scripts/compare_validation_predictions.py \
  "runs/$RUN/product" "runs/$RUN/centered" > "runs/$RUN/centered_vs_product.json"

tar -czf "runs/$RUN/${RUN}_validation.tar.gz" -C "runs/$RUN" \
  centered_vs_additive.json centered_vs_product.json \
  {cnn,additive,product,centered}.log \
  {cnn,additive,product,centered}/{metadata.json,history.jsonl,results.json,best_validation.npz}
~~~

## 附录：87 篇逐篇阅读记录

对本地 87 份 PDF 提取原文文字，逐篇检查摘要和可提取结论；核心 MFP、化学表示、部分标注及干预论文补读方法与对照。085 的本地文字层缺损，以官方出版 PDF 与作者稿交叉识别。逐条记录的是机制和 V3 使用判断，所链接的评测数字应按原论文任务解读。每条提供官方出处与本地 PDF。

### 多标签视觉和学习理论


- **001 · Two-Way Multi-Label Loss**（2023）。样本内和标签内双向相对损失，扩大判别间隔。尾类排序对照须先区分未测与阴性。 [官方出处](https://openaccess.thecvf.com/content/CVPR2023/html/Kobayashi_Two-Way_Multi-Label_Loss_CVPR_2023_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/001_Two_Way_Multi_Label_Loss.pdf)

- **002 · PatchCT: Aligning Patch Set and Label Set with Conditional Transport for Multi-Label Image Classification**（2023）。双向条件传输对齐 patch 与标签原型。肽残基可服务多个功能，未测标签也无法定义完整传输边际。 [官方出处](https://openaccess.thecvf.com/content/ICCV2023/html/Li_PatchCT_Aligning_Patch_Set_and_Label_Set_with_Conditional_Transport_ICCV_2023_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/002_PatchCT_Aligning_Patch_Set_and_Label_Set_with_Conditional_Transpo.pdf)

- **003 · Correlative and Discriminative Label Grouping for Multi-Label Visual Prompt Tuning**（2025）。共现与互斥分别分组学习提示。Gram+ 与 Gram− 可同时阳性，互斥须有可验证依据。 [官方出处](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_Correlative_and_Discriminative_Label_Grouping_for_Multi-Label_Visual_Prompt_Tuning_CVPR_2025_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/003_Correlative_and_Discriminative_Label_Grouping_for_Multi_Label_Vis.pdf)

- **004 · Multi-Label Knowledge Distillation**（2023）。按标签蒸馏二元 logits 和标签级嵌入。仅用于以后压缩对照，教师无法提供新测量。 [官方出处](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Multi-Label_Knowledge_Distillation_ICCV_2023_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/004_Multi_Label_Knowledge_Distillation.pdf)

- **005 · Multi-Label Self-Supervised Learning with Scene Images**（2023）。用嵌入字典产生多伪标签并做 BCE 预训练。近邻伪标签可能重复数据库偏差。 [官方出处](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Multi-Label_Self-Supervised_Learning_with_Scene_Images_ICCV_2023_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/005_Multi_Label_Self_Supervised_Learning_with_Scene_Images.pdf)

- **006 · Bridging the Gap Between Model Explanations in Partially Annotated Multi-Label Classification**（2023）。BoostLU 恢复部分标注条件下降低的归因幅度。归因热图无法验证活性残基。 [官方出处](https://openaccess.thecvf.com/content/CVPR2023/html/Kim_Bridging_the_Gap_Between_Model_Explanations_in_Partially_Annotated_Multi-Label_CVPR_2023_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/006_Bridging_the_Gap_Between_Model_Explanations_in_Partially_Annotate.pdf)

- **007 · Towards Calibrated Multi-label Deep Neural Networks**（2024）。SPA 与标签对正则处理概率校准。校准可评估记录概率，真实活性还受缺测约束。 [官方出处](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Towards_Calibrated_Multi-label_Deep_Neural_Networks_CVPR_2024_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/007_Towards_Calibrated_Multi_label_Deep_Neural_Networks.pdf)

- **008 · Modeling Label Correlations with Latent Context for Multi-Label Recognition**（2024）。从多层上下文推导样本条件标签关联。与既有标签上下文方案相近，仅保留关联偏移诊断。 [官方出处](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04855.pdf) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/008_Modeling_Label_Correlations_with_Latent_Context_for_Multi_Label_R.pdf)

- **009 · Distributionally Robust Loss for Long-Tailed Multi-Label Image Classification**（2024）。类内 LSEP 和负梯度约束减少长尾过拟合。负梯度约束不能认定未测为阴性。 [官方出处](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4926_ECCV_2024_paper.php) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/009_Distributionally_Robust_Loss_for_Long_Tailed_Multi_Label_Image_Cl.pdf)

- **010 · Multi-Label Cluster Discrimination for Visual Representation Learning**（2024）。多语义聚类分配代替单簇伪监督。可检查表征覆盖，序列同源簇不能当作功能真值。 [官方出处](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3972_ECCV_2024_paper.php) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/010_Multi_Label_Cluster_Discrimination_for_Visual_Representation_Lear.pdf)

- **011 · Identifying and Mitigating Spurious Correlation in Multi-Task Learning**（2025）。识别共享训练中的虚假任务相关。用于家族辅助负迁移的机制解释和分环境检查。 [官方出处](https://openaccess.thecvf.com/content/CVPR2025/html/Chai_Identifying_and_Mitigating_Spurious_Correlation_in_Multi-Task_Learning_CVPR_2025_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/011_Identifying_and_Mitigating_Spurious_Correlation_in_Multi_Task_Lea.pdf)

- **012 · SPARC: Score Prompting and Adaptive Fusion for Zero-Shot Multi-Label Recognition in Vision-Language Models**（2025）。分数标准化配合复合提示解决视听零样本偏置。借用分数尺度检查，缺同构的肽文本空间。 [官方出处](https://openaccess.thecvf.com/content/CVPR2025/html/Miller_SPARC_Score_Prompting_and_Adaptive_Fusion_for_Zero-Shot_Multi-Label_Recognition_CVPR_2025_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/012_SPARC_Score_Prompting_and_Adaptive_Fusion_for_Zero_Shot_Multi_Lab.pdf)

- **013 · Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification**（2025）。用 CAM 选局部视图并去偏蒸馏 CLIP。剪出的肽片段可能是活性不同的新分子。 [官方出处](https://openaccess.thecvf.com/content/CVPR2025/html/Kim_Classifier-guided_CLIP_Distillation_for_Unsupervised_Multi-label_Classification_CVPR_2025_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/013_Classifier_guided_CLIP_Distillation_for_Unsupervised_Multi_label_.pdf)

- **014 · Category-Specific Selective Feature Enhancement for Long-Tailed Multi-Label Image Classification**（2025）。类激活区域和渐进增强改善尾类特征。可审查尾类局部证据，热图仍需位点测定支持。 [官方出处](https://openaccess.thecvf.com/content/ICCV2025/html/Du_Category-Specific_Selective_Feature_Enhancement_for_Long-Tailed_Multi-Label_Image_Classification_ICCV_2025_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/014_Category_Specific_Selective_Feature_Enhancement_for_Long_Tailed_M.pdf)

- **015 · MambaML: Exploring State Space Models for Multi-Label Image Classification**（2025）。双向 Mamba 聚合图像及标签状态。短肽先验证化学信号，换序列算子本身已有先例。 [官方出处](https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_MambaML_Exploring_State_Space_Models_for_Multi-Label_Image_Classification_ICCV_2025_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/015_MambaML_Exploring_State_Space_Models_for_Multi_Label_Image_Classi.pdf)

- **016 · Prototype-based Causal Intervention for Multi-Label Image Classification**（2026）。动态上下文原型建模视觉混杂。原型增加在项目中缺少收益，肽库共现不足以识别混杂。 [官方出处](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Prototype-based_Causal_Intervention_for_Multi-Label_Image_Classification_CVPR_2026_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/016_Prototype_based_Causal_Intervention_for_Multi_Label_Image_Classif.pdf)

- **017 · \[CLS\] is Not Enough: Multi-Label Recognition via Patch-Level Inference and Adaptive Aggregation**（2026）。patch 级推断和自适应聚合补足 CLS。局部保留有依据，V2.0 已近似测试只改聚合。 [官方出处](https://proceedings.mlr.press/v306/wang26am.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/017_CLS_is_Not_Enough_Multi_Label_Recognition_via_Patch_Level_Inferen.pdf)

- **018 · Revisiting F-measure Optimization in Multi-Label Classification: A Sampling-based Approach**（2026）。采样估计 F-measure 的联合决策概率，FFT 加速。F 与 Jaccard 需分别验证，HH 后处理表现有限。 [官方出处](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Revisiting_F-measure_Optimization_in_Multi-Label_Classification_A_Sampling-based_Approach_CVPR_2026_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/018_Revisiting_F_measure_Optimization_in_Multi_Label_Classification_A.pdf)

- **019 · Rethinking BCE Loss for Multi-Label Image Recognition with Fine-Tuning**（2026）。文本语义协方差约束置信结构。缺少肽语义对齐时，标签共现可能引回已测试的偏置。 [官方出处](https://openaccess.thecvf.com/CVPR2026?day=2026-06-07) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/019_Rethinking_BCE_Loss_for_Multi_Label_Image_Recognition_with_Fine_T.pdf)

- **020 · DARC: Dual Adjustment Reasoning with Counterfactuals for Trustworthy Chest X-ray Classification**（2026）。胸片反事实区分病灶共现和背景，使用额外标注。肽应使用实际实体编辑和实验记录。 [官方出处](https://openaccess.thecvf.com/content/CVPR2026/html/Liao_DARC_Dual_Adjustment_Reasoning_with_Counterfactuals_for_Trustworthy_Chest_X-ray_CVPR_2026_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/020_DARC_Dual_Adjustment_Reasoning_with_Counterfactuals_for_Trustwort.pdf)

- **021 · FedMPT: Federated Multi-Label Prompt Tuning of Vision-Language Models**（2026）。联邦环境用 LLM 条件和传输对齐缓解相关漂移。可借来源分组验证，无联邦训练需求。 [官方出处](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_FedMPT_Federated_Multi-Label_Prompt_Tuning_of_Vision-Language_Models_CVPR_2026_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/021_FedMPT_Federated_Multi_Label_Prompt_Tuning_of_Vision_Language_Mod.pdf)

- **022 · PatchDEMUX: A Certifiably Robust Framework for Multi-label Classifiers Against Adversarial Patches**（2025）。逐标签二元认证防御视觉局部攻击。真实化学编辑可以改变活性，评价其可分辨性。 [官方出处](https://openaccess.thecvf.com/content/CVPR2025/html/Jacob_PatchDEMUX_A_Certifiably_Robust_Framework_for_Multi-label_Classifiers_Against_Adversarial_CVPR_2025_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/022_PatchDEMUX_A_Certifiably_Robust_Framework_for_Multi_label_Classif.pdf)

- **023 · Compositional Targeted Multi-Label Universal Perturbations**（2025）。用凸锥组合标签扰动。独立假设可启发寻找反例，不能假定每种肽编辑的效应可加。 [官方出处](https://openaccess.thecvf.com/content/CVPR2025/html/Mahmood_Compositional_Targeted_Multi-Label_Universal_Perturbations_CVPR_2025_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/023_Compositional_Targeted_Multi_Label_Universal_Perturbations.pdf)

- **024 · Revisiting Sparsity Constraint Under High-Rank Property in Partial Multi-Label Learning**（2026）。部分候选标签下用稀疏扰动维持高秩。避免强行低秩，候选标签设定与 Fine39 缺测不同。 [官方出处](https://openaccess.thecvf.com/content/CVPR2026/papers/Si_Revisiting_Sparsity_Constraint_Under_High-Rank_Property_in_Partial_Multi-Label_Learning_CVPR_2026_paper.pdf) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/024_Revisiting_Sparsity_Constraint_Under_High_Rank_Property_in_Partia.pdf)

- **025 · FedHarmony: Harmonizing Heterogeneous Label Correlations in Federated Multi-Label Learning**（2026）。跨联邦客户端用共识相关抑制漂移。可报告跨来源共现变化，联邦聚合无需迁移。 [官方出处](https://openaccess.thecvf.com/content/CVPR2026/papers/Kou_FedHarmony_Harmonizing_Heterogeneous_Label_Correlations_in_Federated_Multi-Label_Learning_CVPR_2026_paper.pdf) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/025_FedHarmony_Harmonizing_Heterogeneous_Label_Correlations_in_Federa.pdf)

### 肽功能、化学表示及构象


- **026 · AMCL: supervised contrastive learning with hard sample mining for multi-functional therapeutic peptide prediction**（2025）。AMCL 用 TextCNN、重叠对比、困难样本和 Focal Dice／DBL。反转与近似残基替换保活性须实测。 [官方出处](https://doi.org/10.1186/s12915-025-02273-0) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/026_AMCL_supervised_contrastive_learning_with_hard_sample_mining_for_.pdf)

- **027 · TPpred-CMvL: prediction of multi-functional therapeutic peptide using contrast multi-view learning**（2025）。TPpred-CMvL 对比 TAPE 与 PSSM 多视图。借用同任务单视图／融合消融，相关视图不保证提供新信息。 [官方出处](https://doi.org/10.1186/s12915-025-02466-7) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/027_TPpred_CMvL_prediction_of_multi_functional_therapeutic_peptide_us.pdf)

- **028 · SA-MTP: a structure-aware framework for multifunctional therapeutic peptide annotation**（2026）。SA-MTP 的概率二级结构、PLM 接触先验、动态 GAT 和标签 FiLM 已属直接近邻；论文是 15 类协议。 [官方出处](https://doi.org/10.1093/bib/bbag361) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/028_SA_MTP_a_structure_aware_framework_for_multifunctional_therapeuti.pdf)

- **029 · Peptide language pragmatic analysis and two-stage hierarchical learning framework for therapeutic peptide prediction**（2026）。TPpred-PepPA 用 ProtT5 与两阶段层次学习应对长尾。层次辅助在本地应暂停作为主线。 [官方出处](https://doi.org/10.1186/s12915-026-02642-3) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/029_Peptide_language_pragmatic_analysis_and_two_stage_hierarchical_le.pdf)

- **030 · PepBERT: Lightweight language models for bioactive peptide representation**（2026）。PepBERT 在短肽语料上预训练 1.86M／4.94M 编码器。肽专用轻量预训练可作外部基线。 [官方出处](https://doi.org/10.1016/j.fufo.2026.100999) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/030_PepBERT_Lightweight_language_models_for_bioactive_peptide_represe.pdf)

- **031 · PepLand: a large-scale pre-trained peptide representation model for a comprehensive landscape of both canonical and non-canonical amino acids**（2025）。PepLand 用原子／片段异构图自监督，含非标准残基。分子图与多粒度预训练已有先例。 [官方出处](https://doi.org/10.1093/bib/bbaf367) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/031_PepLand_a_large_scale_pre_trained_peptide_representation_model_fo.pdf)

- **032 · HELM-BERT: Topology-Aware Representations for Chemically Modified Peptides**（2026）。HELM-BERT 显式编码单体和连接，比较 SMILES 及 scaffold 分割。不能用其通透性成绩替代 Fine39。 [官方出处](https://doi.org/10.1021/acs.jcim.6c00451) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/032_HELM_BERT_Topology_Aware_Representations_for_Chemically_Modified_.pdf)

- **033 · Peptide structural plasticity is predictable from sequence and environment**（2026）。ApexFold 按溶剂条件预测 CD 二级结构分布，并做时间外验证。环境条件须独立于结构可信度处理。 [官方出处](https://pmc.ncbi.nlm.nih.gov/articles/PMC13640665/) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/033_Peptide_structural_plasticity_is_predictable_from_sequence_and_en.pdf)

- **034 · Ensemble-Aware Structural Encoding (EASE): Conditioning Protein Encoders on Peptide Conformational Ensembles**（2026）。EASE 对构象聚类选代表并排列无关聚合。首轮跳过依赖大规模可信构象采样的路线。 [官方出处](https://arxiv.org/abs/2610.27638) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/01_%E9%87%8D%E7%82%B9%E8%A7%86%E8%A7%89%E5%92%8CMFP%E8%AE%BA%E6%96%87/034_Ensemble_Aware_Structural_Encoding_EASE_Conditioning_Protein_Enco.pdf)

### 不完整监督、跨视图和多标签方法


- **035 · Label-Guided Representation Learning for Incomplete Multi-View Multi-Label Classification**（2026）。语义原型混合先验、类特异后验融合和信息瓶颈。本地多原型与共享融合结果不支持复用为主贡献。 [官方出处](https://proceedings.mlr.press/v306/li26cg.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/035_Label_Guided_Representation_Learning_for_Incomplete_Multi_View_Mu.pdf)

- **036 · Multi-Label Test-Time Adaptation with Bayesian Conditional Priors**（2026）。对高置信锚标签作 PMI 式贝叶斯后验修正。属于暂停的后验路线，条件先验可能包含来源偏差。 [官方出处](https://proceedings.mlr.press/v306/li26jw.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/036_Multi_Label_Test_Time_Adaptation_with_Bayesian_Conditional_Priors.pdf)

- **037 · NBCG: Nash-Bargained Causal Game for Long-Tailed Multi-Label NLP**（2026）。神经结构方程造标签联盟并用 Nash 目标平衡梯度。可测竞争梯度，因果边需独立证据。 [官方出处](https://proceedings.mlr.press/v306/yang26do.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/037_NBCG_Nash_Bargained_Causal_Game_for_Long_Tailed_Multi_Label_NLP.pdf)

- **038 · One Coin Has Two Sides: Single Positive Multi Label Learning from Salient Annotations**（2026）。SalSPML 建模优先记录显著类，PiSA 用原型拒绝。提供非随机缺测的类比。 [官方出处](https://proceedings.mlr.press/v306/wang26dl.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/038_One_Coin_Has_Two_Sides_Single_Positive_Multi_Label_Learning_from_.pdf)

- **039 · Rethinking Consistent Multi-Label Classification Under Inexact Supervision**（2026）。从候选和互补弱标注推导一致风险。普通零位不满足其可靠候选或阴性的观测条件。 [官方出处](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a7c1a61d395611ccfd9672b35720f102-Abstract-Conference.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/039_Rethinking_Consistent_Multi_Label_Classification_Under_Inexact_Su.pdf)

- **040 · Unlocking the Power of Co-Occurrence in CLIP: A DualPrompt-Driven Method for Training-Free Zero-Shot Multi-Label Classification**（2026）。DualPrompt 用相关提示恢复 CLIP 忽略的共同对象。支持检查次功能遗漏，提示共现无法补物理证据。 [官方出处](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cb825b4e5d0665a339e8d1aa08d1ca7e-Abstract-Conference.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/040_Unlocking_the_Power_of_Co_Occurrence_in_CLIP_A_DualPrompt_Driven_.pdf)

- **041 · DF2-VB: Dual-level Fuzzy Fusion with View-specific Boosting for Multi-view Multi-label Classification**（2026）。特征和决策双层模糊融合。先核输入互补性，本地全肽加和路径降低多功能表现。 [官方出处](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_DF2-VB_Dual-level_Fuzzy_Fusion_with_View-specific_Boosting_for_Multi-view_Multi-label_CVPR_2026_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/041_DF2_VB_Dual_level_Fuzzy_Fusion_with_View_specific_Boosting_for_Mu.pdf)

- **042 · Label-Wise uncertainty decomposition for Multi-label Classification by Maximizing Type II Likelihood**（2026）。Type-II likelihood 分解标签不确定性的变异、偏差与展宽。留作置信诊断。 [官方出处](https://proceedings.mlr.press/v337/li26i.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/042_Label_Wise_uncertainty_decomposition_for_Multi_label_Classificati.pdf)

- **043 · Permutation-Consistent Variational Encoding for Incomplete Multi-View Multi-Label Classification**（2026）。变分信息瓶颈与跨视图排列一致性。不变性可用于分子图重编号，不能任意重排序列。 [官方出处](https://proceedings.iclr.cc/paper_files/paper/2026/hash/fd8f0d6175c930e860bef686a6fcaeeb-Abstract-Conference.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/043_Permutation_Consistent_Variational_Encoding_for_Incomplete_Multi_.pdf)

- **044 · Multi-Label Learning with Contrastive Cluster Self-Supervision for 3D Hierarchical Semantic Segmentation**（2026）。3D 分割用解耦头及簇级对比解决层次冲突。可借用监督冲突消融，层次含义不同。 [官方出处](https://proceedings.mlr.press/v306/cao26j.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/044_Multi_Label_Learning_with_Contrastive_Cluster_Self_Supervision_fo.pdf)

- **045 · Can Class-Priors Help Single-Positive Multi-Label Learning?**（2025）。CRISP 估单正例类先验及无偏风险。Fine39 标签频率是记录频率，真实活性先验需额外数据。 [官方出处](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1dc9fbdb6b4d9955ad377cb983232c9f-Abstract-Conference.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/045_Can_Class_Priors_Help_Single_Positive_Multi_Label_Learning.pdf)

- **046 · Copula Based Trainable Calibration Error Estimator of Multi-Label Classification with Label Interdependencies**（2025）。用 copula 可微估联合校准。当前只能对记录事件校准，联合生物活性缺真值。 [官方出处](https://proceedings.mlr.press/v258/panda25a.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/046_Copula_Based_Trainable_Calibration_Error_Estimator_of_Multi_Label.pdf)

- **047 · More Reliable Pseudo-labels, Better Performance: A Generalized Approach to Single Positive Multi-label Learning**（2025）。稳健伪标签损失与多焦点强化。伪标签不能充当未测活性的真值。 [官方出处](https://openaccess.thecvf.com/content/ICCV2025/html/Tran_More_Reliable_Pseudo-labels_Better_Performance_A_Generalized_Approach_to_Single_ICCV_2025_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/047_More_Reliable_Pseudo_labels_Better_Performance_A_Generalized_Appr.pdf)

- **048 · Multi-Label Text Classification with Label Attention Aware and Correlation Aware Contrastive Learning**（2025）。标签注意与相关感知对比联合训练。既有查询与关系类实验已覆盖关键组件。 [官方出处](https://ijcai.org/proceedings/2025/936) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/048_Multi_Label_Text_Classification_with_Label_Attention_Aware_and_Co.pdf)

- **049 · Theory-Inspired Deep Multi-View Multi-Label Learning with Incomplete Views and Noisy Labels**（2025）。信息瓶颈分离共享／私有特征并建模噪声转移。没有可识别的噪声转移矩阵，暂停相同视图重构。 [官方出处](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Theory-Inspired_Deep_Multi-View_Multi-Label_Learning_with_Incomplete_Views_and_Noisy_CVPR_2025_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/049_Theory_Inspired_Deep_Multi_View_Multi_Label_Learning_with_Incompl.pdf)

- **050 · ComRank: Ranking Loss for Multi-Label Complementary Label Learning**（2025）。ComRank 让确知互补标签排序更低。做相应对照须先有确证阴性与测定条件。 [官方出处](https://proceedings.neurips.cc/paper_files/paper/2025/hash/adedc33539eafdf3d2c14a5e0032db6e-Abstract-Conference.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/050_ComRank_Ranking_Loss_for_Multi_Label_Complementary_Label_Learning.pdf)

- **051 · Incomplete Multi-View Multi-label Learning via Disentangled Representation and Label Semantic Embedding**（2025）。VAE 解耦视图共享／私有信息并做语义补全。不同同源表示不会增加未测实验。 [官方出处](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_Incomplete_Multi-View_Multi-label_Learning_via_Disentangled_Representation_and_Label_Semantic_CVPR_2025_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/051_Incomplete_Multi_View_Multi_label_Learning_via_Disentangled_Repre.pdf)

- **052 · Learning Compact Semantic Information for Incomplete Multi-View Missing Multi-Label Classification**（2025）。COME 提取跨视图紧凑共享特征及双分支补标签。首轮跳过互相补全以免放大共现偏差。 [官方出处](https://proceedings.mlr.press/v267/wen25c.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/052_Learning_Compact_Semantic_Information_for_Incomplete_Multi_View_M.pdf)

- **053 · Multi-Label Node Classification with Label Influence Propagation**（2025）。拆开图传播与变换并定量建模标签影响。可启发邻接扰动消融，分子键不同于标签统计边。 [官方出处](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2862ccf01e3843c81623b246895bcc45-Abstract-Conference.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/053_Multi_Label_Node_Classification_with_Label_Influence_Propagation.pdf)

- **054 · Recover and Match: Open-Vocabulary Multi-Label Recognition through Knowledge-Constrained Optimal Transport**（2025）。RAM 以局部适配和知识约束 OT 排除无关视觉区域。肽任务缺少可确认的功能位点边际。 [官方出处](https://openaccess.thecvf.com/content/CVPR2025/html/Tan_Recover_and_Match_Open-Vocabulary_Multi-Label_Recognition_through_Knowledge-Constrained_Optimal_Transport_CVPR_2025_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/054_Recover_and_Match_Open_Vocabulary_Multi_Label_Recognition_through.pdf)

- **055 · Theory-Driven Label-Specific Representation for Incomplete Multi-View Multi-Label Learning**（2025）。TDLSR 用拓扑及原型补缺并约束视图信息。可比较缺结构组，近邻插补不等于测完结构。 [官方出处](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b07052dc9c644f40efbc813a76d08f36-Abstract-Conference.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/055_Theory_Driven_Label_Specific_Representation_for_Incomplete_Multi_.pdf)

- **056 · Cut out and Replay: A Simple yet Versatile Strategy for Multi-Label Online Continual Learning**（2025）。CUTER 裁出类别区域用于在线经验回放。肽片段为新分子，需实测才能继承标签。 [官方出处](https://proceedings.mlr.press/v267/wang25bg.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/056_Cut_out_and_Replay_A_Simple_yet_Versatile_Strategy_for_Multi_Labe.pdf)

- **057 · L3A: Label-Augmented Analytic Adaptation for Multi-Label Class Incremental Learning**（2025）。L3A 用旧类伪标签及加权闭式分类器做类增量。可待标签字典增类再试。 [官方出处](https://proceedings.mlr.press/v267/zhang25y.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/057_L3A_Label_Augmented_Analytic_Adaptation_for_Multi_Label_Class_Inc.pdf)

- **058 · Language-Driven Multi-Label Zero-Shot Learning with Semantic Granularity**（2025）。按属性、层次、情境重建类别文本。可用于梳理 assay 粒度，泛标签不能自动补成具体子类。 [官方出处](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Language-Driven_Multi-Label_Zero-Shot_Learning_with_Semantic_Granularity_ICCV_2025_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/058_Language_Driven_Multi_Label_Zero_Shot_Learning_with_Semantic_Gran.pdf)

- **059 · Large Language Models Do Multi-Label Classification Differently**（2025）。自回归 LLM 每步偏单一标签，初始排序与最终输出不同。单独测第二标签并比较并行输出。 [官方出处](https://aclanthology.org/2025.emnlp-main.126/) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/059_Large_Language_Models_Do_Multi_Label_Classification_Differently.pdf)

- **060 · Leveraging Text-to-Text Transformers as Classifier Chain for Few-Shot Multi-Label Classification**（2025）。T5 分类链建模标签依赖并蒸馏至少样本模型。固定集合任务不引入人为标签序列。 [官方出处](https://aclanthology.org/2025.emnlp-main.1368/) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/060_Leveraging_Text_to_Text_Transformers_as_Classifier_Chain_for_Few_.pdf)

- **061 · Multi-Label Test-Time Adaptation with Bound Entropy Minimization**（2025）。BEM 通过增强与文本检索估计标签数，再适配多标签熵。可报告标签数误差，Test 保持独立。 [官方出处](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1e616bde0438cb10cb6adf076ae7d336-Abstract-Conference.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/061_Multi_Label_Test_Time_Adaptation_with_Bound_Entropy_Minimization.pdf)

- **062 · Tensorized Multi-View Multi-Label Classification via Laplace Tensor Rank**（2025）。Laplace 张量秩联合建模跨视图与标签关系。低秩可能合并不同功能，暂缓该路线。 [官方出处](https://proceedings.mlr.press/v267/zhong25c.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/062_Tensorized_Multi_View_Multi_Label_Classification_via_Laplace_Tens.pdf)

- **063 · Tight and Fast Bounds for Multi-Label Learning**（2025）。用平滑代理的向量收缩推导泛化界。理论用于损失设计，不能当真实 ACC 预测。 [官方出处](https://proceedings.mlr.press/v267/zhang25dd.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/063_Tight_and_Fast_Bounds_for_Multi_Label_Learning.pdf)

- **064 · Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training**（2024）。训练时通过拼接分离自身目标与共现背景。肽上用真实同测变体，避免随机拼接代替实验干预。 [官方出处](https://proceedings.mlr.press/v235/xie24i.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/064_Counterfactual_Reasoning_for_Multi_Label_Image_Classification_via.pdf)

- **065 · In Pursuit of Causal Label Correlations for Multi-label Image Recognition**（2024）。通过上下文干预区分有用和虚假相关。借鉴对照思想，观察匹配不能认定随机因果实验。 [官方出处](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5c54e016197805946481d786d80a662e-Abstract.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/065_In_Pursuit_of_Causal_Label_Correlations_for_Multi_label_Image_Rec.pdf)

- **066 · Toward Robustness in Multi-Label Classification: A Data Augmentation Strategy against Imbalance and Noise**（2024）。BalanceMix 同时缓解长尾和标签噪声。混合物及拼接序列须另有活性证据。 [官方出处](https://ojs.aaai.org/index.php/AAAI/article/view/30157) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/066_Toward_Robustness_in_Multi_Label_Classification_A_Data_Augmentati.pdf)

- **067 · Attention-Induced Embedding Imputation for Incomplete Multi-View Partial Multi-Label Classification**（2024）。跨视图注意力插补缺失嵌入。现阶段保留几何缺失状态。 [官方出处](https://ojs.aaai.org/index.php/AAAI/article/view/29293) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/067_Attention_Induced_Embedding_Imputation_for_Incomplete_Multi_View_.pdf)

- **068 · Multi-Label Learning with Stronger Consistency Guarantees**（2024）。多标签 logistic 与更强 H-consistency 界。适作损失对照，依赖适用的标注与目标条件。 [官方出处](https://proceedings.neurips.cc/paper_files/paper/2024/hash/04ad66d02234541aac5143de7876e880-Abstract.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/068_Multi_Label_Learning_with_Stronger_Consistency_Guarantees.pdf)

- **069 · Multi-Label Supervised Contrastive Learning**（2024）。MulSupCon 按标签重叠定义加权对比。实测差异对照比仅靠记录重叠更适合检验突变敏感性。 [官方出处](https://ojs.aaai.org/index.php/AAAI/article/view/29619) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/069_Multi_Label_Supervised_Contrastive_Learning.pdf)

- **070 · TagCLIP: A Local-to-Global Framework to Enhance Open-Vocabulary Multi-Label Classification of CLIP Without Training**（2024）。TagCLIP 从局部到全局聚合。V2.0 局部候选说明只改读出难获净收益。 [官方出处](https://ojs.aaai.org/index.php/AAAI/article/view/28139) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/070_TagCLIP_A_Local_to_Global_Framework_to_Enhance_Open_Vocabulary_Mu.pdf)

- **071 · View-Category Interactive Sharing Transformer for Incomplete Multi-View Multi-Label Learning**（2024）。VIST 视图与标签交互，并用 KNN 补缺。近邻补分的 HH 表现弱，缺失化学不交给近邻猜测。 [官方出处](https://openaccess.thecvf.com/content/CVPR2024/html/Ou_View-Category_Interactive_Sharing_Transformer_for_Incomplete_Multi-View_Multi-Label_Learning_CVPR_2024_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/071_View_Category_Interactive_Sharing_Transformer_for_Incomplete_Mult.pdf)

- **072 · Enhancing Multi-Label Classification via Dynamic Label-Order Learning**（2024）。学习样本自适应标签生成顺序。用于排序诊断，固定集合无需人为序列顺序。 [官方出处](https://ojs.aaai.org/index.php/AAAI/article/view/29814) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/072_Enhancing_Multi_Label_Classification_via_Dynamic_Label_Order_Lear.pdf)

- **073 · Language-Driven Cross-Modal Classifier for Zero-Shot Multi-Label Image Recognition**（2024）。以合成语言监督跨模态转移零样本分类器。项目缺稳定的分子—测定文本对齐。 [官方出处](https://proceedings.mlr.press/v235/liu24bq.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/073_Language_Driven_Cross_Modal_Classifier_for_Zero_Shot_Multi_Label_.pdf)

- **074 · Multi-Label Open Set Recognition**（2024）。SLAN 利用子标签结构识别开放集。项目泛 unknown 代表家族内记录粒度。 [官方出处](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0aee38a6fe9fffc8b658cfb1d872c1d5-Abstract.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/074_Multi_Label_Open_Set_Recognition.pdf)

- **075 · Unbiased Multi-Label Learning from Crowdsourced Annotations**（2024）。以多标注者转移矩阵构造无偏风险。来源间缺充分同实体并行记录，来源名不足以确定可靠性。 [官方出处](https://proceedings.mlr.press/v235/xia24a.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/075_Unbiased_Multi_Label_Learning_from_Crowdsourced_Annotations.pdf)

- **076 · Learning in Imperfect Environment: Multi-Label Classification with Long-Tailed Distribution and Partial Labels**（2023）。COMIC 同时处理长尾和部分标注。伪标签循环可能持续强化少见功能的缺测。 [官方出处](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Learning_in_Imperfect_Environment_Multi-Label_Classification_with_Long-Tailed_Distribution_and_ICCV_2023_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/076_Learning_in_Imperfect_Environment_Multi_Label_Classification_with.pdf)

- **077 · An Effective Deployment of Contrastive Learning in Multi-label Text Classification**（2023）。比较严格、类内、Jaccard 等五种对比标签定义。把实测匹配差值作为生物变体对比的候选目标。 [官方出处](https://aclanthology.org/2023.findings-acl.556/) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/077_An_Effective_Deployment_of_Contrastive_Learning_in_Multi_label_Te.pdf)

- **078 · Exploring Structured Semantic Prior for Multi Label Recognition With Incomplete Labels**（2023）。用结构化跨模态语义先验补标注。词义关系能解释字典，无法替代未测实验。 [官方出处](https://openaccess.thecvf.com/content/CVPR2023/html/Ding_Exploring_Structured_Semantic_Prior_for_Multi_Label_Recognition_With_Incomplete_CVPR_2023_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/078_Exploring_Structured_Semantic_Prior_for_Multi_Label_Recognition_W.pdf)

- **079 · Holistic Label Correction for Noisy Multi-Label Classification**（2023）。用标签依赖做噪声校正。当前以缺测为主，强行校正可能重现 V2.0 多报。 [官方出处](https://openaccess.thecvf.com/content/ICCV2023/html/Xia_Holistic_Label_Correction_for_Noisy_Multi-Label_Classification_ICCV_2023_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/079_Holistic_Label_Correction_for_Noisy_Multi_Label_Classification.pdf)

- **080 · Partial Multi-Label Learning with Probabilistic Graphical Disambiguation**（2023）。用概率图对候选标签做潜变量消歧。候选生成与 Fine39 不同，缺测与活性仍难识别。 [官方出处](https://proceedings.neurips.cc/paper_files/paper/2023/hash/04e05ba5cbc36044f6499d1edf15247e-Abstract-Conference.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/080_Partial_Multi_Label_Learning_with_Probabilistic_Graphical_Disambi.pdf)

- **081 · Scene-Aware Label Graph Learning for Multi-Label Image Classification**（2023）。用场景条件图学习局部共现。测定对象及条件可进入实测任务，标签共现图暂停。 [官方出处](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Scene-Aware_Label_Graph_Learning_for_Multi-Label_Image_Classification_ICCV_2023_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/081_Scene_Aware_Label_Graph_Learning_for_Multi_Label_Image_Classifica.pdf)

- **082 · Towards Better Representations for Multi-Label Text Classification with Multi-granularity Information**（2023）。对比学习与多粒度辅助优化文本表示。肽的新多粒度算子须做参数匹配、同输入对照。 [官方出处](https://aclanthology.org/2023.findings-emnlp.635/) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/082_Towards_Better_Representations_for_Multi_Label_Text_Classificatio.pdf)

- **083 · CDUL: CLIP-Driven Unsupervised Learning for Multi-Label Image Classification**（2023）。局部与全局 CLIP 聚合产生迭代伪标签。首轮跳过伪标签循环。 [官方出处](https://openaccess.thecvf.com/content/ICCV2023/html/Abdelfattah_CDUL_CLIP-Driven_Unsupervised_Learning_for_Multi-Label_Image_Classification_ICCV_2023_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/083_CDUL_CLIP_Driven_Unsupervised_Learning_for_Multi_Label_Image_Clas.pdf)

- **084 · Generalized test utilities for long-tail performance in extreme multi-label classification**（2023）。用期望测试效用区分 top-k 长尾价值和缺测。候选真值上限与实际决策收益需分开。 [官方出处](https://proceedings.neurips.cc/paper_files/paper/2023/hash/46994b3d6dd0fd5fca5f780af6259db5-Abstract-Conference.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/084_Generalized_test_utilities_for_long_tail_performance_in_extreme_m.pdf)

- **085 · Knowledge Restore and Transfer for Multi-Label Class-Incremental Learning**（2023）。KRT 动态伪标签与增量交叉注意迁移旧类。适用于后续增类，原 PDF 文字层有缺损，补读作者稿。 [官方出处](https://openaccess.thecvf.com/content/ICCV2023/html/Dong_Knowledge_Restore_and_Transfer_for_Multi-Label_Class-Incremental_Learning_ICCV_2023_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/085_Knowledge_Restore_and_Transfer_for_Multi_Label_Class_Incremental_.pdf) · [作者稿](https://arxiv.org/abs/2302.13334)

- **086 · Open Set Action Recognition via Multi-Label Evidential Learning**（2023）。MULE 用 Beta 证据估计同时动作的不确定性并去偏。可启发校准，置信分布不是实验证据。 [官方出处](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Open_Set_Action_Recognition_via_Multi-Label_Evidential_Learning_CVPR_2023_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/086_Open_Set_Action_Recognition_via_Multi_Label_Evidential_Learning.pdf)

- **087 · Texts as Images in Prompt Tuning for Multi-Label Image Recognition**（2023）。TaI 以跨模态已对齐的文本替代图像做提示学习。项目缺该对齐，首轮保持分子输入。 [官方出处](https://openaccess.thecvf.com/content/CVPR2023/html/Guo_Texts_as_Images_in_Prompt_Tuning_for_Multi-Label_Image_Recognition_CVPR_2023_paper.html) · [本地 PDF](../%E5%9B%BE%E6%96%87%E7%A0%94%E7%A9%B6%E6%8A%A5%E5%91%8A_87%E7%AF%87%E8%AE%BA%E6%96%87%E6%9C%AC%E5%9C%B0%E9%93%BE%E6%8E%A5%E6%95%B4%E5%90%88%E5%8C%85/PDF%E8%AE%BA%E6%96%87/02_%E5%85%B6%E4%BB%96%E8%B7%A8%E9%A2%86%E5%9F%9F%E5%8F%82%E8%80%83/087_Texts_as_Images_in_Prompt_Tuning_for_Multi_Label_Image_Recognitio.pdf)
