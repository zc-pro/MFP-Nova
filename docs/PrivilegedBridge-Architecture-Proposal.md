# PrivilegedBridge：用训练期模态残差指导纯序列部署的架构设计

> **实现更新（2026-11-08）：**核心主配置已实现，见[模型源码](../src/mfp_nova/models/privilegedbridge.py)和[配置](../configs/privilegedbridge.yaml)。已做CPU模型及训练链路验证，正式GPU实验待执行；本页较早段落保留原设计历史，当前运行方式见[README](../README.md#新三模型实现状态与运行)。


> 设计日期：2026-10-30。本文是待验证的研究方案；尚未实现、训练或取得性能结果。只建议新增一种独立算法，不替换现有五模型。公共比较与文献梳理见 [MFP 架构研究综述](MFP-Architecture-Design-Study.md)。

## 1. 研究问题与可检验假设

**跨领域研究补充（2026-11-07）：**[总览第8.3、8.7节](MFP-Architecture-Design-Study.md#83-三个方案目前有哪些不能算强创新的地方)收紧原创性判断。MARS已有完整/缺失模态的特权残差利用，交叉拟合蒸馏也已有研究；固定同基座时，残差SmoothL1与对应logit差值损失可完全等价。本文的实际差异须来自分量目标、学生基座更新及可迁移任务信息，不能单以“残差”命名支持创新；部署可行性与方法首创性分别判断。

现有模型普遍把结构、理化属性和序列表征放入同一预测过程；即使支持输入模态缺失，训练时利用的证据组合与仅给出序列的实际推理仍可能不同。HelixSet 已有模态屏蔽、R-Drop 和双冻结 ESM，仍出现训练指标持续上升而 Validation 指标下降的历史现象；这说明继续增加融合容量不一定有效，但不能据此证明几何分支就是过拟合原因。

PrivilegedBridge 提出一个不同问题：**能否先学出“额外模态相对固定序列基座改变了哪些标签判断”，再把可从序列预测的部分迁移到小型桥接头，部署时完全取消额外模态输入？**

设计包含三个可证伪假设：

1. **增量证据假设。** 当前 515 维属性及已观测结构中，存在相对序列基座有用的增量。Teacher 在相同序列表征与相同数据协议下应优于基座，至少在部分预先定义的群组成立。
2. **可迁移性假设。** 部分增量可以由序列隐状态近似。如果完整 Teacher 有效、Student 无效，则增量可能依赖序列无法确定的信息，或者桥接容量不足，不能把蒸馏失败解释成数据问题。
3. **低维约束假设。** 迁移每标签的少量任务相关残差，比复制完整残基对张量或强迫重建坐标更适合当前样本规模。必须与普通最终 logits 蒸馏、同容量无蒸馏模型比较。

“训练期特权信息”只表示**本实验在训练阶段开放、部署阶段不要求提供**的现有输入。515 维理化属性中有一部分本来即可由序列计算，它们不是新生物学测量，也不因此包含序列以外的信息；对应实验应解释为归纳偏置迁移。预测结构同样不等于实验结构。

### 1.1 固定基座机制对照

| 项目 | 固定基座机制对照，非主配置 |
|---|---|
| 序列基座 | 冻结单650M＋AA34，d192，两级CNN＋一层6头Transformer＋每标签一次读取 |
| Student可见输入 | 固定标签状态R0与固定H0的全局均值g0，拼接384维；首轮关闭D/修饰输入 |
| Teacher可见输入 | 额外观察到的515维性质及局部/全局结构；两模态独立残差，零证据时归零 |
| 训练 | A训练基座；B固定基座训练Teacher；C固定二者，优先训练两个384→128→1标量桥 |
| Student目标 | 主分类＋0.2分量残差SmoothL1；R-Drop/EMA首轮关闭 |
| 必要竞争 | 无KD、普通KD及mask/质量Teacher；共享64维隐桥为机制对照，可适应Student为性能竞争条件 |
| 适用任务 | major15/fine39完整检验；therapep49只有S0，不能作为迁移机制验证 |

本路线不以“结构可以全部还原”为前提。它只研究现有训练信息中，哪些预测校正可以由部署时可见的固定序列特征估计；无法迁移时应接受更简单的基座或普通KD。

### 1.2 性能与研究价值的定位（2026-11-05）

当前三阶段训练的实现和归因复杂度最高。此前共享隐桥的固定残差函数可能压低Student上限，现已将标量桥提升为首选；固定基座和每模态±2限幅仍需验证。它首先是**训练信息利用与无结构部署方法**，不是最有把握在三个任务都取得最高指标的统一主模型。therapep49缺少本方案所需的真实配套模态，只有S0时不计作第三个完整新机制的验证。

性能优先并不要求永久冻结Student下游网络。保留固定参考版本用于研究“迁移哪些增量”，同时增加第6.7节的可适应Student作为直接竞争条件。Teacher权重与目标仍固定，Student可以适应任务；实际最终采用者由Validation决定。若完整多模态Teacher明显优于所有纯序列Student，且应用允许推理使用这些输入，也应保留Teacher作为性能候选，明确它的部署条件，而非为了小模型目标主动舍弃性能。

普通KD、特权学习、中心化残差都不是独立的新方法。分量监督、缺失处理和固定读出是否必要，需要在强Student、普通KD及同预算继续监督训练之上证明。只让一个较弱的固定S0变好，创新证据不够。

### 1.3 实施主配置与适用条件（2026-11-07）

主配置仍为单650M＋AA34、d192序列基座和两个384→128→1标量桥。A训练基座；B固定S0训练两个确定性Teacher分支；C复制A权重为Student并更新其下游主干和桥，ESM固定。部署只保留一份Student和两桥。

**本轮修正：C阶段主目标改为“实际可观察分量子集的预测对齐”，权重0.2、SmoothL1 beta=1、前5轮升权。** 原分量残差监督仅保留为明确消融，不与新目标默认叠加。原因是Student基座更新后，再要求每条桥固定重现原S0残差，可能重复补偿已经被Student主干学会的差异。主干lr2e-5、桥lr2e-4；其余预算见第7.5节。

这项修正使目标与输出一致，但必须承认它属于观察子集上的logit蒸馏，不是新损失。独立研究价值集中在零参考Teacher是否提供真实内容增量，以及这些增量能否在相同部署输入和预算下超越普通Teacher。若仅有普通KD收益，按普通KD报告。

A/C沿用匹配基线R-Drop/EMA；B的固定基座和Teacher残差函数无dropout。Teacher不共享持续更新的Student参数。质量ρ仅作已定义的输入质量权重，不表示Teacher正确概率。全部Val和观察/缺失模态组都评价Student实际部署输出。

完整机制适用于major15/fine39。作者therapep48/49纯序列队列只运行S0；不能以S0优于CNN声称验证了特权迁移。 **2026-11-08澄清：这是当前序列加载路径的限制，不是数据来源的根本限制。现已按总览“作者48维队列的多模态回填可行性”建立rich48；保留作者标签、成员与split并正确回填可核实属性/结构和mask，即可在该视图运行完整A/B/C。其结果明确标为增加训练期模态的实验。**另设650M→150M-F的纯序列KD会改变研究问题，本轮不默认加入。

## 2. 与五个现有模型的关系

以下按当前源码、默认配置及 README 描述，不把历史可选分支误写为全部默认启用。

| 模型 | 当前核心机制 | 本方案保留的经验 | 主要区别及必要对照 |
|---|---|---|---|
| TriRelay | 序列/结构/属性交互、标签反馈、标签关系；另有可选 shared/private 分支 | 单模态证据应能独立读取，缺失模态不得产生伪证据 | shared/private 在同一模型内保留不同信息路径，不保证统计解耦；本方案固定基座和 Teacher，跨训练阶段迁移有明确零参考的预测增量 |
| GeoFactor | 残基、连续/空间片段、属性、标签异构图；高阶因子推断 | 结构与功能存在局部及高阶联系，但应分辨观察和推断 | 不在部署时保留异构图或标签因子迭代；先验证局部结构汇总是否足以指导 Student |
| ProtoRoute | 标签条件专家选择、共享多原型、关系残差 | 不同标签需要不同证据 | 不做部署时多专家竞争；两类训练期证据转换为 Student 的两组加性校正；不新增原型库 |
| HelixSet | 默认双冻结 ESM、完整残基对与几何上下文、潜变量与集合解码，R-Drop | 冻结预训练表征、干净推理与训练一致性 | 不重建或蒸馏完整 pair 张量；不增加三角更新，不把两个网络随机 dropout 的一致性叫作特权蒸馏 |
| TheraSet | TheraPep 式两级 CNN 直接头，多原型条件读取多尺度/模态/pair证据，标签残差 | 简单 CNN 有用，直接序列通路应保持可用 | TheraSet 的额外证据参与运行时残差；本方案将Teacher证据编码器移出部署，首选仅保留序列Student与标量桥 |

已有 `training_views` 对结构/属性做随机丢弃与局部遮盖，使一个模型适应不同观察组合；已有 R-Drop 对同一输入的两次随机网络预测施加一致性。两者均不等于“固定有额外输入的 Teacher 指导只接收序列的 Student”。主配置使用匹配R-Drop/EMA，成对机制隔离实验才关闭；不把训练配方差异归给Teacher。

当前 TheraSet 单 Seed Validation 与 HelixSet 的微小差异不能作为本方案预期增益；现有不同选模协议、不同 ESM 输入和不同训练设置的历史成绩也不能直接排出五模型名次。

## 3. TheraPep-AI 与视觉研究带来的具体启发

TheraPep-AI 官方实现以氨基酸编码和多核 CNN 预测多标签，发布配置使用 34 维编码、3/5/7/9 卷积核及五模型集成；其 README 还采用周期填充、随机循环位移和合成背景。本方案只借鉴可直接监督的局部序列主干，采用项目已有 `MotifCNN` 的独立适配思路；不复制五模型集成、周期假设、固定 50 长度或生成负例协议。[官方仓库](https://github.com/terra-quantum-public/tq-therapep-ai)

用户提供的TheraPep-AI复现包进一步确立它为**同原始数据来源的主要baseline**；其therapep49 CSV与本项目完全对齐，F1复现完整Test×49 Micro-F1为79.319%、正肽功能为76.250%。两个现有实验均含背景，positive_only不是无背景训练。该任务仍只比较本方案S0，不能借与CNN的差异声称验证特权迁移；major15/fine39需另训作者CNN的任务适配baseline。详见[复现baseline研究](TheraPep-AI-Metric-Alignment.md#11-用户复现包作为主要baseline的依据)。

迁移机制的直接证据来自相同ESM/相同S0下的监督桥、普通KD、残差KD对照；与无ESM的TheraPep五成员则是系统级比较。Teacher的额外输入和三阶段训练、ESM提取及推理都计入成本。原作者BCE与本项目弱ASL的影响独立解释，不将它算作残差桥创新。若最终部署仍运行650M/双ESM，不能仅凭小桥参数宣称比14.25M的纯CNN集成更轻。

本次“三大顶刊”采用 TPAMI、IJCV、TIP 的工作口径；CVPR、ICCV、ECCV 是会议。下列均有可核实的正式期刊记录，方法迁移只作为研究启发：

| 来源 | 已核实的方法层面事实 | 迁移到本项目的边界 |
|---|---|---|
| [Tu 等，TPAMI 2025，GLSCL](https://doi.org/10.1109/TPAMI.2025.3596394) | 分开处理遮盖后的全局语义和局部 token 补全 | 不强迫所有层、所有模态使用同一蒸馏目标；本方案先迁移标签增量，局部隐特征仅作后续消融。其最早预印本为 2023，不能称为 2025 全新提出 |
| [Yu 等，IJCV 2024，M²A²E/AMA](https://link.springer.com/article/10.1007/s11263-024-02055-1) | 非对称多模态遮盖、轻量适配及缺失模态实验 | 支持按模态角色设计训练输入；脸部 RGB/IR/Depth 的对齐方式和性能结论不可直接外推到肽 |
| [Wang 等，TIP 2024，LSKD](https://doi.org/10.1109/TIP.2024.3372448) | 根据层语义使用不同蒸馏方式，并屏蔽受背景语义漂移影响的目标 | 本方案屏蔽没有真实观察支撑的 Teacher 分量；不照搬语义分割背景类，也不引入双层权重优化 |
| [Lei Wang、Piotr Koniusz，IJCV 2025](https://link.springer.com/article/10.1007/s11263-025-02513-4) | 利用辅助模态特征训练低成本特征预测路径，并研究特征不确定性 | 只预测任务相关隐证据，不生成坐标、DSSP 或结构文件；预测特征不能登记成真实结构 |
| [Liu 等，IJCV 2026](https://link.springer.com/article/10.1007/s11263-026-03025-5) | 公开摘要讨论不完整多视图多标签、共享信息压缩与标签原型 | 提醒区分“模态缺失”和“标签未标注”；不默认加入信息瓶颈、原型与逆最优传输。正文受订阅限制，本次仅据公开摘要/出版信息；该文是 ICML 2024 工作的扩展 |

具体书目信息与一手链接见文末。本方案没有声称这些研究已经验证“蒸馏可提升治疗肽预测”，也没有声称以下组合具有已确认的文献首创性。

## 4. 输入、张量与监督口径

研究范围是`major15`、`fine39`、`therapep49`三个独立任务，输出`C=15/39/49`。完整的特权模态迁移假设主要在前两者验证；第三任务缺少统一特权观察，按下文限定退化为序列基座比较。标签空间不拼接，不恢复旧48位目标。

| 字段/符号 | 形状 | 使用位置 |
|---|---|---|
| `aa`, `mask` | `[B,L]` | 全阶段；沿当前残基顺序，padding 不参与卷积池化或 attention |
| `d`, `residue_chem`, `chemical` | `[B,L]` / `[B,K_chem]` | 已有注释可用时作为序列侧输入；普通 FASTA 没有注释则使用既有空注释约定，不推断缺失修饰 |
| `esm`, `esm_peptide` | 650M `[B,L,1280]`、150M-F `[B,L,640]` | 统一归因用单650M；单150M-F及双路竞争，基座/Teacher/Student必须用同一条件；下文1920维是双路实例 |
| `phys`, `phys_mask` | `[B,515]` | 仅 Teacher，依 `PROPERTY_GROUPS` 分成 25 组 |
| `geometry`, `struct_mask`, `dssp` | `[B,L,16]`, `[B,L]`, `[B,L]` | 仅 Teacher；结构存在与置信度已知是两个不同状态 |
| `struct_global`, `struct_global_mask` | `[B,25]` | 仅 Teacher；投影输入拼接值和 mask 共 50 维 |
| `confidence_known` | `[B,L]` | Teacher 有效观察加权；不是类别标签 |
| `coords`, `edge_type`, `quality` | 当前已有字段 | 基础版本不读坐标、不建 pair、不用来源特征做分类；保留源数据接口，不生成替代观测 |
| `y` | `[B,C]` | 仅进入 loss 与 Train 统计；绝不进入 Teacher/Student forward 或 query |
| `row_id` | `[B]` | 保持数据和缓存对齐，不作为特征 |

`L` 是当前批次有效最大残基长度；保持项目现有样本范围与缓存长度，不引入 TheraPep 的固定 50 截断。当前配置缓存 `max_length: 80`，这不是本设计另行改变数据范围的依据。

需要区分三种 mask：

- `train_original_mask` / `val_original_mask` / `test_original_mask` 是样本队列选择，必须沿用；不是每标签“已确认阴性”矩阵。
- `mask`、`struct_mask`、`phys_mask` 等定义输入观察有效性。
- 本设计新增蒸馏权重来自模态是否观察到、观察质量和现有未标注权重，不伪造 `label_mask` 字段。

主任务继续调用 `observed_label_loss`：0 是未标注的弱惩罚项，默认权重 0.3；fine39 样本中 `__unknown` 对应父功能已知而具体子类未知时，相应具体子类的零项再乘 0.25。完整 Teacher、序列基座和 Student 使用完全相同的语义。

`therapep49`是单独数据协议，当前对所有来源统一只开放AA、固定残基属性和所选ESM，关闭修饰/515维性质/结构，防止模态有无泄漏来源。因没有可用特权模态，**该任务只训练序列基座S0，跳过B/C，不能声称验证了PrivilegedBridge迁移机制**。只有另行定义实际序列Teacher时才有普通KD实验，本设计不默认增加它。

该任务按用户要求研究两种Train队列：全部发布行，或仅`is_therapeutic=True`的真实/恢复正肽；二者均保留49输出，按共同正肽Val前48功能列Micro-F1@0.5选模，比较同一正肽Test功能指标，并另报完整背景诊断。无背景条件的第49目标恒1，不能评价其治疗性判别学习；这两条件不等同于therapep48的正肽48输出训练；后者按独立协议运行S0。具体成员数、作者默认口径及公平预算见[总览第2.5节](MFP-Architecture-Design-Study.md)。这两种条件均不为正肽单独补结构、不新生成背景，选项尚未实现。

两种队列仍使用现有监督函数：正肽未标功能弱降权；若包含背景，背景功能0和全局标志0权重为1。去合成背景不等于删去正肽未标注项。该区别不改变major15/fine39的未知0语义。

## 5. 总体架构

~~~mermaid
flowchart LR
    X[同一序列与冻结ESM缓存] --> REF[固定S0参照]
    X --> STU[从S0初始化的可更新Student]
    REF --> Z0[固定z0与R0]
    PG[已观察属性和结构及mask] --> TEA[两个Teacher读取器及零参考残差]
    Z0 --> TEA
    TEA --> TARGET[固定的观察子集预测目标]
    STU --> ZS[学生序列logit]
    STU --> BR[两个序列标量桥]
    ZS --> OUT[部署输出 加全部桥]
    BR --> OUT
    ZS --> SUB[训练辅助输出 只加被观察的桥]
    BR --> SUB
    TARGET -.SmoothL1.-> SUB
    OUT --> LOSS[原弱标注任务损失]
~~~

A只训练S0；B保持S0为eval，仅训练Teacher；C保持S0/Teacher为eval并停止梯度，另训练Student。固定基座机制对照可复用S0前向；主配置Student更新后不能复用R0代替Rθ。训练辅助子集输出读取mask，但Student特征网络和实际部署forward不读取mask、结构或性质。

不将模型权重固定这一算法步骤扩展为工程审批流程。

## 6. 模块数学定义

### 6.1 可直接监督的序列基座

以 \(d=192\)、CNN 每核通道 \(c=64\)、序列侧6个attention heads为初始配置；Teacher的128维attention另用4个heads，各自头维均为32。此规模是起点，不是性能结论。小型MLP均采用 `Linear→GELU→Linear`，除非明确说明，不追加隐藏层。

1. 复用项目已有 34 维氨基酸查表编码思路，输入投影到 64；已有 `d`、残基修饰及全局修饰分别嵌入到 64 并按当前 `MotifCNN` 方式相加。两级 3/5/7/9 卷积得到四个 `[B,L,64]` map，拼接投影到 192。每层依 `mask` 重新清零 padding，并保留线性序列边界。
2. 所选冻结ESM各路做无参数LayerNorm，再投影到192；首轮单650M，双路时先拼接。与CNN残基表示及可训练192维位置嵌入相加，逐位置LayerNorm后乘mask，再经过一层TransformerBlock得到 \(H_0\in\mathbb R^{B\times L\times192}\)。Transformer为6头、FFN扩展4、GELU、dropout0.2，屏蔽无效key且输出重乘mask；位置按样本真实N→C顺序。该路径不建立独立残基pair状态。
3. 每标签可训练 query \(q_c\in\mathbb R^{192}\) 读取 \(H_0\)，得到 \(R_0\in\mathbb R^{B\times C\times192}\)。固定标签顺序，一标签一 query，不增加多原型或标签间自注意力。
   同时计算无参数的 `g0=masked_mean(H0)`，形状 `[B,192]`，作为Student共同序列上下文；不更改基座z0，不增加Teacher输入。
4. \(z^0_{bc}=\langle w_c,R^0_{bc}\rangle/\sqrt{192}+b_c\)。仅本阶段用任务监督训练基座；ESM 始终冻结。基座选定后，其投影、CNN、Transformer、query 和分类头一并固定。

普通 CNN 直接头与 label-readout 的基座比较可以只做一次小规模消融；本研究的核心是桥接，不应把基座结构搜索扩张成另一项研究。

### 6.2 低维 Teacher 证据读取

属性分支复用 25 组分词方式，每组投影值与其观察 mask，得到 \(P\in\mathbb R^{B\times25\times128}\)。缺失数值在标准化后清零，mask 单独保留，插补值不当作观察证据。基础版本不加属性 Transformer。用 `R0` 投影为 query 做一次 4-head masked cross-attention，随后线性压到 \(U^P\in\mathbb R^{B\times C\times64}\)。全组缺失时输出严格为零；输出投影之后也乘模态存在 mask，避免线性层偏置重新引入非零值。

结构分支将有效 `geometry` 投影到 128，加入 DSSP embedding；25 维全局结构连同其 mask 投影为一个额外 token，得到 \(G\in\mathbb R^{B\times(L+1)\times128}\)。只用局部标量/角度特征和全局描述符，不使用坐标重建损失、距离矩阵或几何三角更新。经过一次 4-head masked cross-attention 和 64 维投影，得到 \(U^G\in\mathbb R^{B\times C\times64}\)，最终同样乘观察存在 mask。

定义 \(a^P_b\) 为存在至少一个有效属性组，\(a^G_b\) 为存在有效结构残基或有效全局结构，均为 0/1。全局结构 mask 必须与原始结构有效性一致；不能凭插补值构造存在标记。

结构读取可按已有可信度字段设置 attention bias：已知置信度残基使用 \(q_i=0.3+0.7\,\mathrm{clip}(g_{i,9},0,1)\)，置信度未知但结构已观测时取中性权重 1。未知不等于零质量。全缺失仍输出零。实现中ρG为有效结构token的q均值，全局token质量取1。此为计算权重，不是经过校准的“结构正确概率”；应单独报告置信度未知组。

落实到算子时，bias取 `log(q_i)` 加到有效key的attention logits，全局token权重取1，无效key屏蔽。全缺失样本直接跳过该模态attention并回填零，不能先对全负无穷的行做softmax再乘零。`aP/aG` 判断的是输入观察；属性mask、全局结构mask及质量本身也可能带有来源信息，掩码正确并不等于已经排除来源捷径。

### 6.3 明确零参考的 Teacher 残差

每个模态 \(m\in\{P,G\}\) 使用一个跨标签共享小 MLP：

\[
f_m:\mathbb R^{192+64}\rightarrow\mathbb R^{128}\rightarrow\mathbb R,
\qquad
r^m_{bc}=2\tanh\left(\frac{f_m([R^0_{bc},U^m_{bc}])-f_m([R^0_{bc},0])}{2}\right).
\]

Teacher 输出为：

\[
z^T=z^0+a^P r^P+a^G r^G.
\]

减去同一函数的零证据输出，使该分量在证据向量为零时恰为零；固定基座提供可重复的参照。`f_m` 不使用 dropout，确保零证据前后的两次调用是同一个确定函数；B/C 阶段基座始终为 `eval`。`tanh` 将每模态最大校正幅度限制在 2 个 logit 单位，避免少量结构样本让残差头任意放大；数值 2 是待验证设计参数，需用“不限幅”消融判断是否压制真正强证据。

这里的 \(r^P,r^G\) 是**该模型中对基座预测的加性变化**。它们不具有唯一统计分解，也不是生物学因果贡献。两种模态相关时仍可能分摊或重复同一证据，不能称为已实现共享/私有信息解耦。

### 6.4 共享64维隐桥：保留为机制对照

当前首选为第6.6节直接标量桥。下述共享隐桥保留完整定义，用于检验复用Teacher固定读出是否真的有益，不再因为维度更高而默认它更强。

每标签的基座状态可能只保留当前分类所需信息，丢掉仍能预测Teacher增量的其他序列特征。故Student输入从单独R0收敛为 `X_bc=[R0_bc;g0_b]`，将同一个g0广播到标签轴，宽度384。两个小MLP \(g_P,g_G:384\rightarrow128\rightarrow64\) 预测：

\[
\hat U^m=g_m([R^0;g^0]),\qquad
\hat r^m=2\tanh\left(\frac{f_m([R^0,\hat U^m])-f_m([R^0,0])}{2}\right),
\qquad z^S=z^0+\hat r^P+\hat r^G.
\]

阶段 C 中 `f_m` 及 `R0` 坐标固定，**仅 `g_m` 更新**，防止一边追逐 Teacher 残差、一边改变残差坐标系。Student 不接收 `aP/aG`、结构来源、数据来源或 Teacher 注意力；部署中两组残差都由序列估计。

g0来自同一固定H0，B/C均随基座eval而确定，计算只需一次masked mean，不新增ESM前向或attention。Teacher的残差函数仍读 `[R0,U]`，零参考和残差分解不变。理想平方风险下，扩大条件信息不能提高最优可达风险；有限样本和小桥不保证获益，因此保留“仅R0”的小型消融。它不恢复H0本身未包含的信息，也不解释为全序列结构重建。

将 `g_m` 最后一层初始化为零，因此 Student 起点严格等于基座。`f_m` 固定但对输入仍可微；实现时仅关闭 `f_m` 参数梯度，不能在 Student 路径套 `no_grad()`，否则桥接器无法学习。

Teacher残差函数使用正常非零初始化，不将其输入证据通路与Student桥同时初始化为零。在Student起点 `Uhat=0`，有 `∂rhat/∂Uhat=∂f(R0,U)/∂U|U=0`，因为 `tanh` 此时导数为1。因此桥能否启动取决于固定Teacher头在零证据附近的局部梯度，而非仅取决于loss是否有限。未来一次真实batch backward检查桥末层有梯度即可；无需每步计算完整Jacobian。桥隐藏层首步尚无梯度属于末层零初始化的正常现象，后续应随末层更新开始学习。

64维 `Uhat` 只经一个标量函数映射到该模态的每标签残差，通常不可辨识：许多隐向量会产生相同输出。默认只约束任务残差，不给 `Uhat` 添加逐维MSE，也不把它当成恢复出的真实属性或结构嵌入。如果固定 `f` 在Student可达区域饱和或平坦，即使Teacher有效也可能难以迁移；这属于共享读出假设的失败，不能默认堆深桥网络解决。

预测出的结构分量只能标注为“结构指导的序列估计校正”。它不是该序列具有某种真实结构的证据，也不能替代生物学实验解释。

### 6.5 创新候选在哪里，不能成立的主张是什么

知识蒸馏、特权信息学习、固定编码器与残差头都是已有方法。本文当前创新候选是：**以固定序列决策为Teacher参照、使Teacher模态校正在零证据时消失、按实际观察到的分量迁移**；部署时复用固定残差函数只是共享隐桥的额外假设，须单独证明。它试图把“额外信息改变了什么”和“序列能够预测多少改变”拆成可测试的问题，主配置预测对齐属于KD；方法价值须来自Teacher构造和任务证据，不能用重新命名支持原创性。

一个必须承认的等价情形是：若全部模态齐全、只蒸馏残差总和、使用平方损失且基座固定，则总残差 MSE 与最终 logits MSE 完全等价，因为两边的基座相消。真正需要实验支持的差异来自 P/G 分量目标、缺失分量屏蔽、零参考与固定读出函数；这也是必须做“合计残差”和“普通 KD”对照的原因。分量监督可能减少两分支误差互相抵消，也可能在模态高度相关时施加不必要约束。

对于无法从序列恢复的构象细节，本方案没有自动识别或剔除机制。以平方损失的理想无容量限制版本为例，令序列状态为 `X`、蒸馏权重为 `W = a × rho × omega`，最优预测为 `E[W × r | X] / E[W | X]`；仅在权重和观察机制满足额外条件时才是通常的 `E[r | X]`。实际SmoothL1学到的是相应加权稳健中心，还受桥容量、限幅及主监督影响；共享隐桥额外受固定f约束。它不是每条肽真实结构增量的恢复器。

因此，可行性来自已有张量、普通 attention/MLP、有限部署参数和可直接检查的损失，而不是依赖不可验证的“结构信息全部可蒸馏”假设。若分量蒸馏不优于普通 KD，就采用更简单的方法；不以复杂度本身证明创新。

### 6.6 将“真实模态增量”与“增加容量”分开

零参考只保证 `U=0` 时残差消失。Teacher读取器有bias、可见模态mask且使用序列query，完全可能输出常量或主要由 `R0` 决定的非零 `U`；此时仍能形成额外序列分类器。因此 `Teacher>S0` 是必要观察，**不足以证明结构/性质内容有用**。

增加一个直接可实现的匹配Teacher对照：保留完全相同的读取器、mask、质量权重、query、残差函数及B阶段训练预算，把标准化后的已观察属性/几何/全局结构数值置零、DSSP替换为同一占位embedding；质量只用于与主方案相同的外部权重。该对照必须从头训练B，不能只在完整Teacher推理时突然清空数值。它保留可用性、质量及新增容量，移除具体内容，是“mask/质量+序列”对照，并非严格无信息Teacher。

完整Teacher相对该对照在同一Validation、尤其共同可观察样本上有增量，才支持“额外内容值得传递”；若二者相当，优先解释为新增容量、缺失模式或质量捷径。这个对照也不能证明生物学因果，且不能完全解决观察机制偏差，但比仅与较小基座比较更接近本研究问题。

当前优先采用直接标量桥：每模态 `h_m:384→128→1`，跨标签共享，通过 `rhat_m=2 tanh(h_m([R0;g0])/2)` 输出，末层零初始化；此处R0公式用于固定基座对照；主配置读Rθ/gθ，采用第7.3节预测对齐目标，不部署Teacher的f。两桥合计98,818个参数（含bias），共享64维桥为115,072且还需固定f。两者用相同序列上下文；共享隐桥仅在证明其读出约束有益后保留，不是第四个模型。

这一调整来自监督维度：每标签/模态只监督一个标量残差，局部Jacobian对64维U的秩至多1，不能识别其余方向。共享网络跨样本和标签会耦合这些自由度，不能机械声称每个样本都有63个独立可调参数，但仅凭该损失仍不足以解释64维“证据坐标”。给U加MSE会改变研究目标且可能强迫复制无用信息，因此本轮不以新增特征损失补救。

相关一手研究提醒，Teacher可能携带对下游无用的信息，而更贴近Teacher也不必然有更好的泛化。本方案据此把**验证集任务指标**放在残差拟合误差之前；不引入这些论文的复杂信息量估计或多层优化。[Dissanayake等，AISTATS 2025](https://proceedings.mlr.press/v258/dissanayake25a.html)；[Stanton等，NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/376c6b9ff3bedbbea56751a84fffc10c-Abstract.html)。这是对实验解释的借鉴，并非这些论文已验证本方案。

### 6.7 主配置：可适应Student与预测一致的目标

\[
\hat r_\theta^m=2\tanh(h_m([R_\theta;g_\theta])/2),\qquad
z^{deploy}=z_\theta^{seq}+\hat r_\theta^P+\hat r_\theta^G.
\]

两桥末层零初始化，Student初始与选定A基座的确定性预测一致；训练时dropout会产生通常的随机差异。其余桥层正常初始化。Teacher始终相对固定z0生成rT，不随Student变化。

原分量目标存在可定位的冲突：假设z0=0、rT=1，Student主干已经将zseq学到1；若桥仍被要求输出1，部署logit变成2。任务损失可能抵消这项压力，但原目标本身并不保证与Teacher预测对齐。故主配置用第7.3节的实际预测差，主干和桥共同承担可迁移增量。

两桥分解仍不唯一：若总是同时观察P/G，总和损失只识别总修正。部分观察提供子集约束，但缺失机制可能有偏，不保证识别生物独立贡献。因此展示的是模型校正分量，不是恢复的结构/属性事实。

若要检验固定分量目标，明确用同一Student、同预算，将主KD目标替换为分量SmoothL1；不同时加入两种损失再解释分量作用。固定S0、只训桥的版本另作机制对照。

### 6.8 观察质量不等于任务增量，相关模态也不自动可分

结构置信度ρ只描述既有输入质量权重，不代表该模态对某标签有预测增量；a只表示观察存在。即使二者很高，Teacher也可能学错。保持分量目标不阈值化、不用低熵自动宣布可靠。完整Teacher和mask/质量Teacher的验证对照仍是判断内容价值的依据，不能仅凭ρ决定增加蒸馏强度。

P/G相关时，联合输出可能容许分量相互补偿，分支辅助损失只是施加偏好，不使分解唯一。主配置已经使用观察子集的预测对齐。固定基座时，它退化为对应总和残差SmoothL1；二者完全等价，不重复训练。学生基座更新时，预测对齐包含zseq−z0，旧总和残差目标不包含，不能混称同一算法。分量监督作为替换目标，用于检查是否值得约束P/G分摊。

CVPR2025 DMMIvNL区分视图缺失与标签噪声，并讨论共享/特有信息；它提供了相关问题背景，但其噪声转移与信息瓶颈假设不能直接套在本项目“0为未标注”的目标上。[CVF原文](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Theory-Inspired_Deep_Multi-View_Multi-Label_Learning_with_Incomplete_Views_and_Noisy_CVPR_2025_paper.html) 因此本轮只加强有效目标和对照，不新增VAE、标签修复或自适应置信门控。

## 7. 训练目标与缺失模态

### 7.1 阶段 A：基座

\[
\mathcal L_A=\mathrm{observed\_label\_loss}(z^0,y).
\]

用完整原始 Train 队列，不筛成“结构齐全子集”；按统一 Validation 指标选基座。保留相同 seed 和原样本设定。阶段 A 的权重随后固定为阶段 B/C 的共同起点。

### 7.2 阶段 B：Teacher

只更新两个证据读取器与残差函数。初始方案：

\[
\mathcal L_B=\mathcal L_{obs}(z^T,y)
+0.2\,\operatorname{Mean}_{m:a^m\neq0}\mathcal L_{obs}(z^0+a^m r^m,y;\,a^m)
+10^{-3}\,\operatorname{Mean}_{b,c,m:a^m_b=1}(r^m_{bc})^2.
\]

分支损失只在该模态存在的样本上计算；这里传给 `observed_label_loss` 的 mask 是 `[B,C]` 广播后的**分支适用性**，不是负标签确认。无观察的模态无分支损失。轻量二次项抑制在少数模态可用样本上过大的增量，不引入复杂不确定性网络。

实现须把 `a_m[:,None]` 显式展开为与logits相同的 `[B,C]` 再传入共享loss。当前函数分子广播后对B×C求和，分母直接取传入mask的元素和；只传 `[B,1]` 会把分支loss意外放大C倍。模态分支项先各自在观察到的 `样本×标签` 项上求均值，再对本批至少一行可用的模态取均值。联合主项沿全部B×C均值；完全缺失行只贡献固定常数，不产生Teacher梯度。这样的归约定义避免标签数15/39改变辅助项的有效权重。

训练仍使用相同全部 Train 样本；完全无特权输入的样本在 Teacher 阶段只得到固定 \(z^0\)，不会为分支产生梯度，属于有意定义。整个 batch 均无特权观察时，跳过该批 Teacher 优化器更新；其预测仍定义为基座，不能对无可训练依赖的 loss 强行 backward。Teacher 权重由 Validation 选择，但 Validation 不参与梯度、预处理拟合或生成训练目标。

### 7.3 阶段 C：观察子集的预测对齐

Teacher及S0进入eval、输出停止梯度。a_b^m表示实际观察，ρP=1，ρG按第6.2节现有结构质量定义。令I_b=1{Σ_m a_b^m>0}，有观察时取q_b=Σ_m a_b^mρ_b^m/Σ_m a_b^m，无观察时q_b=0。这些权重固定、不由Student控制。

定义Teacher目标与Student训练辅助输出：

\[
t_{bc}=\operatorname{sg}(z^0_{bc}+\sum_m a_b^m r^T_{bcm}),\quad
s^A_{bc}=z^{seq}_{bc}+\sum_m a_b^m\hat r_{bcm}.
\]

主任务始终使用部署输出zdeploy，而非sA。令ω_bc=y_bc+(1−y_bc)u_bc，u来自已有未标注权重：

\[
\mathcal L_{KD}
=\frac{\sum_{bc}I_bq_b\omega_{bc}\operatorname{SmoothL1}_{\beta=1}(s^A_{bc},t_{bc})}
{\max(1,\sum_{bc}I_bq_b\omega_{bc})},\qquad
\mathcal L_C=\mathcal L_{obs}(z^{deploy},y)+\lambda\mathcal L_{KD}.
\]

λ前5个C阶段epoch升至0.2；FP32归约。不加伪标签、完整特征MSE或新的不确定性网络。某模态缺失时，仅从辅助子集输出和目标中移除该分量，不给它零残差真值；该桥仍通过部署形式的主监督学习。全部模态缺失时KD为0，Student正常更新。

R-Drop开启时，两次Student前向分别计算主监督和KD，再求均值；R-Drop项只比较两次实际部署logit。固定Teacher目标在两次Student前向间复用；不额外生成不同Teacher噪声。Student EMA覆盖下游与两桥，验证和部署使用同一选定参数口径。

**分量对照。** 将上述KD替换为Σ aρ ω SmoothL1(rhat,rT)/max(1,Σ aρ ω)，其他设置不变。若固定基座，还可拟合2C个Train常数作校正对照；常数改变阈值指标、不改变单标签排序。主配置可适应Student时，常数对照也须使用相同的可适应主干和C预算，不能把额外主干训练归给逐肽桥。

**普通KD对照。** 使用同一Teacher、同一子集输出、同一SmoothL1时，就是主配置的等价实现，不重复当作独立实验。真正有区别的对照是：同参数Student无KD；使用不做零参考分解的普通Teacher但保持输入/预算匹配；或固定其他条件将SmoothL1替换为逐标签Bernoulli KL。比较Teacher架构与比较KD距离分别进行，避免混为一个变化。

本方法不识别未知的真实负标签；ω只沿用已有监督语义。教师训练集拟合好也不能代替跨簇验证。

### 7.4 为什么不默认做自监督遮盖补全

当前 ESM 缓存来自完整序列，简单把缓存中某个位置清零，并不能去掉该残基对其他位置上下文的影响。因此“遮盖缓存后重建原残基”不是严格 masked-language 实验，可能形成容易的泄漏式目标。基础版本不加此损失。若后续确需研究，必须重新计算被遮盖序列的 ESM 表征，或明确称为特征扰动一致性，并单独计入成本。

同理，不强制 Student 恢复完整 `geometry[16]` 或 515 个数值：其中有不可由骨架唯一确定的构象、实验条件和缺失机制，精确回归可能训练模型记忆测量来源。

### 7.5 可执行的起始训练预算

采用下列起点，先检验算法而不同时搜索大量训练日程。各阶段按自己的 Validation 主输出选模：A 用 `z0`、B 用 `zT`、C 用实际部署形式 `zS`。

| 阶段 | 更新参数 | 最多 epoch / patience | 学习率及 warmup | 其他设置 |
|---|---|---|---|---|
| A | 序列基座下游参数，ESM固定 | 100 / 15 | AdamW，2e-4，warmup5，随后cosine | weight_decay0.01，clip5，有效batch64 |
| B | Teacher模态读取器与残差函数 | 30 / 8 | 新建AdamW，2e-4，warmup2，随后cosine | 同上；基座固定且eval |
| C | Student下游及两个标量桥；S0/Teacher不更新 | 30 / 8 | 新建AdamW，主干2e-5、桥2e-4，warmup2，随后cosine | 实际预测KD前5轮渐增；主任务与R-Drop使用部署输出 |

本表为主配置；固定基座机制对照仅改C的更新范围，成对关闭EMA/R-Drop时明确标识。全流程最多160个数据遍历epoch，B/C每轮计算量不同，应实测GPU时间。匹配的无KD和普通KD学生复用相同A/B及C上限，另用等总GPU时间的监督基线检验额外训练预算的影响。日程参数只由Train/Val决定。

## 8. 科研正确性与局限

沿用已有簇划分、`row_id` 对齐、原样本 mask、标签顺序和 Train 统计即可。Teacher 的额外字段只能来自当前训练行，不引入同源验证肽的外部检索。Train 学权重，Validation 选模型/阈值，Test 最后独立评价；不重新划分合并数据，不在 Test 上比较多个蒸馏超参数。

Teacher 的训练集预测可用于常规蒸馏，这是标准训练目标而非独立泛化证据；Teacher 可能记忆训练标注并将噪声传给 Student。无需默认建立交叉拟合 Teacher 群，只需报告 Teacher 的 Train/Val 差距及 Student 相对同容量监督对照的差异；只有结果指向严重 Teacher 记忆时，再研究更强的 Teacher 正则化。

Teacher 有模态可用性选择偏差，屏蔽 loss 不能消除 missing-not-at-random 问题。现有来源字段只用于结果分组；不输入类别头，也不按标签生成模态存在状态。重点观察结构缺失、来源、长度、D 注释和置信度未知组的表现。如果收益集中在某一来源，必须报告这种局限。

双 ESM 中肽微调编码器的预训练数据与当前 Val/Test 是否重叠，不能仅由项目缓存证明。沿用现有权重时保留来源说明，并用全模型统一单 650M 的对照判断新增架构效果；不声称更小 150M 天然更无泄漏或泛化更好。

现已核实该150M-F来自PepBenchmark的短肽MLM继续预训练，而非本项目功能标签微调。650M也有UniRef预训练曝光，单650M对照不能充当“无重叠”证明。选择三种表征时，`S0`输入投影分别为1280/640/1920到192，后续结构固定；每种条件需重新训练A及适用的B/C。双路较两个单路是否有真实增量，应按同任务Val比较，完整依据见[总览第2.4节](MFP-Architecture-Design-Study.md)。

本方案默认认为两类特权信息有部分可预测增量；这一假设可能不成立。物理构象/修饰并非单一 canonical backbone 的确定函数，强迫完全蒸馏有不可约误差。主任务损失、残差限幅与低蒸馏权重只能缓和，不能创造不存在的信息。

## 9. 推理与训练伪代码

以下是算法规格，不是现有API或训练命令。

~~~text
A: train S0 on original Train; select using Validation
reference = selected_S0 in eval, parameters fixed
B: train Teacher(reference_states, observed_properties_and_structure)
   select using Validation; set Teacher to eval, parameters fixed
student = copy(selected_S0)
bridges = two_MLPs_with_zero_output_initialization()

C, each Train batch:
    no_grad:
        R0, z0, g0 = reference(sequence_inputs)
        rT, available, quality = teacher(R0, observed_inputs)
        target = z0 + sum(available * rT)

    for each required Student dropout pass:
        R, zseq, g = student(sequence_inputs)
        rhat = 2 * tanh(bridges(concat(R, g)) / 2)
        deployed = zseq + sum(rhat)
        subset = zseq + sum(available * rhat)
        task_loss = observed_label_loss(deployed, y)
        kd_loss = weighted_SmoothL1(subset, stopgrad(target))
    average task and KD losses
    add matched RDrop only on deployed logits
    update student and bridges; update their EMA as configured

Inference:
    R, zseq, g = selected_student(sequence_inputs)
    rhat = 2 * tanh(selected_bridges(concat(R, g)) / 2)
    return sigmoid(zseq + sum(rhat))
~~~

推理不运行reference/Teacher，不读取available、quality、属性或结构。展示桥值时称“预测校正分量”，不称实测模态证据。

## 10. 复杂度与参数预算

以下是手工数量级估计，不是参数统计或 GPU 实测。词表大小和实际投影共享方式会改变精确值。

| 部分 | 预计参数/计算特征 | 说明 |
|---|---|---|
| `S0` 下游可训练部分 | 约 1.5–2.3M | 约 0.49M 两级 c64 CNN、0.37M 双 ESM 投影、约 0.44M 一层 Transformer，加读取头和嵌入 |
| Teacher 额外读取器及两个残差头 | 约 0.3–0.7M | 128 维模态读取、64 维证据和小 MLP；仅 B 阶段训练 |
| Student 两个标量桥 | 98,818 | 两个384→128→1 MLP，含bias；共享隐桥对照为115,072 |
| 最终部署下游总参数 | S0＋98,818，按上述S0范围约1.6–2.4M | 共享隐桥对照另保留固定残差头；各项均待实现统计，原宽泛估计不用于精确比较 |
| 冻结编码器 | 单650M / 单150M-F / 双路约800M | 由当次受控表征条件决定；不能只报告小头并声称整个系统只有2M |

以 \(N_P=25\)、Teacher 宽度 \(d_T=128\)，Student 下游主要成本为：

\[
O\left(BL c^2\sum k+BLd_{in}d+BLd^2+BL^2d+BCLd+BCdr\right).
\]

一层序列 self-attention 仍是 \(L^2\)，不能称为全线性模型；没有 `[B,L,L,pair_dim]` 状态及三角更新。Teacher 额外读取近似 \(O(BC(N_P+L)d_T)\)，其激活在 C 阶段不反向，但前向仍有耗时。

整个研究成本应报告 A+B+C 总 GPU 时间、各阶段更新参数、冻结编码器缓存提取时间、部署端完整序列编码延迟与已缓存特征延迟。**Student 推理便宜不代表三阶段训练更便宜。**

单 650M 是五模型共同表征的架构对照；单 150M 是独立部署成本实验，需要重新训练相应基座/Teacher/Student，不能把双编码器基座的投影直接截掉。150M 的效果及预训练重叠问题单独报告。

## 11. 最小实现顺序

本轮不修改代码。后续若实施，采用普通模型文件与现有训练流程即可：

1. 实现紧凑序列基座并复用数据、缓存、Train 统计、`observed_label_loss`；先跑 forward/backward、padding 不变性与一段短训练。
2. 实现两个 Teacher 读取器和中心化残差函数。检查单模态全缺失时其输出为零、所有模态缺失时等于基座、`y` 不在 forward 输入路径。
3. 接入三阶段参数更新范围及标量桥。用一次backward确认桥有梯度、Teacher目标停止梯度；共享隐桥对照另确认固定f不截断输入梯度。
4. 在现有运行配置中显式记录基座/Teacher 关联、阶段和蒸馏权重；只保存恢复与评价必要内容。不建立模型工厂、证据库或训练平台。
5. 完成一次短训练后再做完整 seed42 对照；仅当假设得到初步支持时运行统一多 Seed 比较。已有通过且无关的检查不重复执行。

只需增加真正对应错误风险的测试：mask、梯度边界、缺失损失项、padding 和相同 Student 输入下额外模态字段变化不影响推理。不需要哈希链、审批清单或额外交付报告。

## 12. 实验、消融与失败判据

### 12.1 先隔离算法收益，再比较大模型

| 实验 | 要回答的问题 |
|---|---|
| `S0` 单独训练 | 固定基座本身的水平 |
| `S0` 固定＋同容量两个桥，仅主任务监督 | 加容量/分阶段训练是否已解释收益 |
| 仅对共享隐桥：无KD监督对照使用相同Teacher固定残差函数 | 排除固定读出函数本身带来的收益；标量桥首选不部署该函数 |
| 普通Teacher＋匹配KD | 零参考Teacher构造是否超出相同额外输入、相同预算的常规Teacher |
| Teacher 完整观察、Teacher 无结构、Teacher 无属性 | 是否真有可迁移的增量，哪个模态提供增量 |
| PrivilegedBridge 完整设计 | 增量是否进入 Student |
| TheraSet 现有 seq/ESM 对照及完整模型 | 相同表征下，部署成本和性能取舍 |
| 五模型单650M匹配比较；单150M-F与双路各自作为竞争条件 | 分离架构与预训练编码器差异，双路需要对两个单路证明增量 |

标量桥使用同初始化、同容量、同预算的无KD监督对照。若研究共享隐桥，再补使用同一个Teacher固定f而蒸馏权重为零的对照，因为固定f本身已经吸收训练期信息；不能把其收益全部归给残差KD。

执行顺序：S0、完整Teacher与mask/质量Teacher；复用A/B，比较同容量无KD与本主配置。随后只针对得到的差异比较Teacher构造、分量目标或固定Student。固定基座下等价的总和/预测差损失不重复训练。therapep48/49当前仅做S0的对应任务，不声称完整机制覆盖。

### 12.2 小而有解释力的消融

- 仅属性桥、仅结构桥：确定哪部分值得保留。
- 中心化残差 vs 普通加性残差：验证零参考是否有用，避免仅靠额外非线性容量获益。
- 分量残差 vs 合计残差：检查是否真的需要可分的 P/G 目标。
- Student只读R0 vs读取 `[R0;g0]`：检验标签压缩之外的固定主干摘要是否有可迁移信息；相同输入的监督桥、普通KD与残差KD同步比较。
- 限幅 2 vs 不限幅：检查稳定性收益与表达能力损失。
- 观察/质量加权 vs 仅观察 mask：置信度字段是否有效，未知置信度组是否受损。
- 可适应Student（主配置）vs固定基座仅训桥：同一目标与两桥，明确更新范围。
- 双 ESM vs 全五模型统一单 650M；单 150M 另列部署轴。

不要在一次首轮实验中穷举全部组合。先基座、匹配监督对照、普通 KD、完整设计四条主线，再做能解释主要差异的消融。

### 12.3 统一协议与观察指标

major15/fine39独立训练，沿现有原样本与簇划分，完整比较统一`traditional5/global`并报告AP和独立多功能指标。therapep49只比较序列基座：作者口径默认all49 Micro-F1@0.5，有/无背景受控比较使用共同正肽Val前48功能列Micro-F1@0.5；不使用major15功能投影。训练日程与有效batch对匹配对照一致，适用时三阶段总预算单独记录。

正式多 Seed 使用已有 `42,101,202,303,404`。先 seed42 探索，最后只对选定方案做必要多 Seed 和独立 Test；历史其他协议成绩作为背景而非显著性对照。按样本簇配对分析模型差值，避免相似序列被当成独立观察。

预先关注结构可用/缺失、置信度已知/未知、长度 2–8/9–16/17–32/33+、D 注释、来源、单/多功能组。小组样本少时报告样本数和不确定性。预测标签数不等于独立功能数，fine39 的 `__unknown` 不当作额外独立功能。

### 12.4 何时应放弃或简化

1. Teacher 相对 `S0` 没有稳定 Validation 增量：没有证据支持再扩大 Student 桥；直接保留基座。
   完整Teacher若不优于相同容量的mask/质量Teacher，也不足以支持“真实模态内容迁移”；先简化为序列容量对照。
2. Teacher 有效而 Student 不优于同容量监督模型：增量不可预测或蒸馏无效，应报告失败，不直接堆更深桥接器。
3. 零参考Teacher或分量目标不优于相应普通KD对照：采用简单方案；不保留已被等价式消除的创新主张。
   直接标量桥若相当或更好，则删除共享64维隐桥及固定残差函数的部署依赖，而非将更多隐维数当作创新。
4. 总体改善伴随结构缺失组、多功能组或短肽明显退步：不能称为解决缺失模态泛化；根据目标取舍或不采用。
5. 只有属性桥有效且属性完全由序列便宜计算：比较部署直接计算属性的简单模型，特权蒸馏未必有实用优势。
6. 下游头更小但总训练时间明显增加、完整 ESM 推理延迟几乎不变：收益只能表述为删除结构依赖，不能笼统宣称高效。

## 13. V100 与 RTX 5070 的适配原则

V100 使用按现有约定克隆得到的 `MFP-Nova` 环境与 PyTorch 2.4.1+cu121；RTX 5070 使用 Windows 配置对应的 2.8.0+cu128。两个平台使用相同选定的冻结编码器条件与FP32建立科学比较，首轮共同650M；不为了本方案升级V100原环境。

两个平台保持任务、seed、原样本、有效 batch、主损失、阶段预算及选模协议一致。可调整 micro-batch 与梯度累积控制显存；V100 的 GPU 选择、CPU 线程与后台运行仍用现有训练入口方式。Windows 多进程保留 main guard。若后续尝试混合精度，分别测量，不将跨设备数值差异当算法增益。

本方案尚无已注册的模型入口、三阶段 CLI 或现成配置，故不提供会误导执行的训练命令。这里仅给实现与资源原则；实现时再同时提供两个平台的真实命令。本文没有启动任何 GPU 任务。

## 14. 参考文献与核实范围

1. **TheraPep-AI 官方仓库**，terra-quantum-public。核实 README、公开模型入口与项目中的独立适配关系；仓库不是视觉期刊论文。[GitHub](https://github.com/terra-quantum-public/tq-therapep-ai)
2. **Rong-Cheng Tu, Yatai Ji, Jie Jiang, Weijie Kong, Chengfei Cai, Wenzhe Zhao, Hongfa Wang, Yujiu Yang, Wei Liu.** *Global and Local Semantic Completion Learning for Vision-Language Pre-Training.* IEEE TPAMI, **47(12):11065–11079, 2025**；online 2025-08-06。正式 DOI：[10.1109/TPAMI.2025.3596394](https://doi.org/10.1109/TPAMI.2025.3596394)。[作者预印本](https://arxiv.org/abs/2306.07096) 首次发布于 2023 年；本文不将预印本年份和期刊年份混用。
3. **Zitong Yu, Rizhao Cai, Yawen Cui, Xin Liu, Yongjian Hu, Alex C. Kot.** *Rethinking Vision Transformer and Masked Autoencoder in Multimodal Face Anti-Spoofing.* IJCV, **132:5217–5238, 2024**；online 2024-06-05。公开全文：[10.1007/s11263-024-02055-1](https://link.springer.com/article/10.1007/s11263-024-02055-1)。
4. **Qilong Wang, Yiwen Wu, Liu Yang, Wangmeng Zuo, Qinghua Hu.** *Layer-Specific Knowledge Distillation for Class Incremental Semantic Segmentation.* IEEE TIP, **33:1977–1989, 2024**；online 2024-03-12。正式 DOI：[10.1109/TIP.2024.3372448](https://doi.org/10.1109/TIP.2024.3372448)；[作者官方仓库](https://github.com/wwwwyw/LSKD)。本设计只借鉴层/目标适配与有效位置屏蔽，不复述未核实的实现细节。
5. **Lei Wang, Piotr Koniusz.** *Feature Hallucination for Self-supervised Action Recognition.* IJCV, **133:7612–7646, 2025**；online 2025-08-07，issue November 2025。公开全文：[10.1007/s11263-025-02513-4](https://link.springer.com/article/10.1007/s11263-025-02513-4)。
6. **Chengliang Liu, Bo Li, Jie Wen, Yong Xu, Wenqi Ren, Xiaochun Cao, Chao Huang.** *Semantic Invariant Representation Learning and Data-Driven Label Prototype Modeling for Incomplete Multi-view Multi-label Classification.* IJCV, **134, article 427, 2026**；online 2026-09-10。[10.1007/s11263-026-03025-5](https://link.springer.com/article/10.1007/s11263-026-03025-5)。仅据公开摘要与出版说明；作者说明它是 ICML 2024 SIP 工作的扩展，不是全部方法始于 2026。

本地实现依据：[五模型与当前协议](../README.md)、[数据输入](../src/mfp_nova/multimodal.py)、[共同网络层及未标注损失](../src/mfp_nova/research_layers.py)、[TriRelay](../src/mfp_nova/models/trirelay.py)、[GeoFactor](../src/mfp_nova/models/geofactor.py)、[ProtoRoute](../src/mfp_nova/models/protoroute.py)、[HelixSet](../src/mfp_nova/models/helixset.py)、[TheraSet](../src/mfp_nova/models/theraset.py)。
