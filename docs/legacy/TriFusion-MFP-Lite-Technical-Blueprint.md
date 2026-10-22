# TriFusion-MFP-Lite Technical Blueprint

基于 TriFusion-MFP 实际实现与已完成实验的技术资产提取。分析日期：2026-10-14；本地代码基线：`2b38514909bdb43032660999a372f83745ef7aba`。本文是 Lite 重建依据，不改变原项目科学合同，也不表示 Lite 已完成实现或验证。

原仓库根目录：`D:/Data/Code/VScode/trifusion/trifusion-mfp`。下文代码路径相对此根目录；外部数据、输出路径分别相对 `TRIFUSION_DATA_ROOT`、`TRIFUSION_OUTPUT_ROOT`。文末提供可点击的实际源码清单。

证据标记：**实现事实**＝已阅读的源码；**本地实验记录**＝同步的轻量结果、Validation 片段、CSV 或 history；**历史报告**＝既有研究报告中的结果；**用户服务器证据**＝负责人提供且已有记录的服务器结论；**建议/假设**＝尚未得到对应新实验支持。

本次只做代码、配置、历史记录和轻量元数据分析；未执行模型、加载权重、读取科研样本或新预测。对早期混合结果 JSON，仅提取预定的 Validation/训练元数据片段，没有整份反序列化或解析 `test_evaluation`。历史 Test 数值只转述已经发布的汇总表；没有读取 Test 特征、标签或预测。没有重新计算任何 SHA256。

## 1. 项目现状与核心结论

**Lite 首选复现：S4，即冻结 ESM-2 最后一层全局表示 + S2 多尺度残基卷积分支 + 可学习全局局部尺度 + 47 标签线性头 + S4-ASL。**

这条路线在完整 sequence-eligible 人群的五 seed 配对实验中具有明确的正向 Macro-AUPRC 证据。S8-B 是同一计算架构在 geometry-eligible 子集上的重新训练，不是另一种更强的序列网络。最新 D2/D3-A 没有取代该序列锚点的证据。

Lite 默认预测链应为：

```text
肽原始序列
  → 已验证的 PLM 序列规范化
  → 冻结 ESM-2 / 最后一层残基状态
  ├─ 独立 B0 masked-mean 全局 embedding [B,1280]
  └─ 残基状态 [B,L,1280] + mask [B,L]
       → Conv1d kernels 3/5/7，各 128 channels
       → GELU + masked max
       → concat[384] → Linear[1280] → GELU → 无 affine LayerNorm
  → 全局 embedding + sigmoid(local_scale) × 局部表示
  → Linear(1280,47) → logits → sigmoid
  → Validation 拟合的阈值策略 → 47 标签二态输出
```

训练使用 ASL；阈值属于训练完成后的后处理，不能参与模型 forward。首版不加入 ProtT5、HELM-BERT、几何专家、Function Query、标签关系网络、hierarchy loss、Trust Router 或 coarse-to-fine decoder。

需要保留的最小学习型融合机制只有 **S2 已验证架构中的一个局部尺度标量**；它不是新建的样本/标签门控模块。删除这个标量或把它改成固定值会改变成熟模型。原项目的审批链、执行状态机、逐源码 SHA 绑定和设备门禁无需迁移。

“Lite”首先表示代码和训练管线轻量：可训练网络约 301 万参数。原始序列在线推理仍需要约 650M 参数的 ESM-2；冻结和离线缓存不等于压缩编码器，也不证明能够直接在移动端部署。

## 2. 模型演进及实验结果对比

### 2.1 比较人群必须分开

| 人群 | 总数 | Train | Validation | Test | 用途 |
|---|---:|---:|---:|---:|---|
| Canonical / split_v1 全集 | 58,583 | 46,803 | 5,918 | 5,862 | 固定 ID 与 homology 分组划分 |
| Sequence-eligible | 57,388 | 45,648 | 5,900 | 5,840 | B0、S1、S2、S2-GC、S3、S4；S5–S7 复用 S4 分数 |
| Geometry-eligible | 54,088 | 43,023 | 5,566 | 5,499 | S8-B/G/GC、S9-I0、D1、D2、D3-A |

集合关系为 `Geometry ⊆ Sequence-eligible ⊆ Canonical Assignment`；下游筛选保持原 split，不重新划分。不得用 S4 的 5,900 条 Validation 与 S8-B 的 5,566 条 Validation 做无条件性能排名。

以下标准差均为 seed 间样本标准差 `ddof=1`；五 seed 是同一个 split 上的五次初始化，不是五折交叉验证。

### 2.2 Sequence-eligible 人群：五 seed 真实结果

每个阶段 seeds=0–4，300 完整 epochs，Validation Macro→Micro→更早 epoch 选模。Validation 指标在本次从同步结果的白名单片段计算；与既有四阶段 CSV、S3/S4 closeout 对照一致。

| 阶段 | 架构/唯一主要变化 | 参数量* | Val Macro-AUPRC | Val Micro-AUPRC | Val Macro-F1† | best epochs（seed0→4） |
|---|---|---:|---:|---:|---:|---|
| B0 e300 | frozen ESM-2 mean → Linear47；BCE | 60,207 | 0.280447 ± 0.000241 | 0.493385 ± 0.003113 | 0.221654 ± 0.000599 | 275,274,273,299,298 |
| S1-last4 | layers30–33 全局 softmax LayerMix + Linear47；BCE | 60,211 | 0.278327 ± 0.001738 | 0.462289 ± 0.007925 | 0.247538 ± 0.013935 | 88,130,86,71,31 |
| S2-GC | B0 全局向量 → 1280→1152→1280 非线性残差；BCE | 3,011,760 | 0.308185 ± 0.008544 | 0.480046 ± 0.024922 | 0.276962 ± 0.016975 | 22,25,24,16,20 |
| S2 | B0 全局 + shared multi-scale residue CNN；BCE | 3,010,992 | 0.330305 ± 0.009724 | 0.471508 ± 0.037439 | 0.317094 ± 0.025185 | 9,13,89,11,25 |
| S3-FSL | S2 的 masked max 改为 47 function query attention；BCE | 3,029,040 | 0.308331 ± 0.013588 | 0.462020 ± 0.027965 | 0.299052 ± 0.014630 | 37,13,7,12,33 |
| **S4-ASL** | **S2 架构不变，仅 BCE→ASL** | **3,010,992** | **0.359016 ± 0.004451** | **0.475708 ± 0.019725** | **0.295008 ± 0.021435** | **5,8,4,7,5** |

\* 不计 frozen ESM-2；这些训练网络只读取缓存。† 每个模型在完整 Validation 上选择自己的单一 global Jaccard 阈值，再评价完整 Validation；这是 in-sample threshold-fit 结果，不能与后文 OOF Macro-F1混为一组。

关键配对证据：

- S2−S2-GC：Validation Macro `+0.022120±0.013882`，5/5 正；参数差只有 768。局部分支的收益不能完全由参数量解释，但不是生物学 motif 因果证明。
- S3−S2：Validation Macro `−0.021974±0.014493`，0/5 正；当前 query-attention 组合为负向消融。
- S4−S2：Validation Macro `+0.028711±0.011043`，5/5 正；同时 Validation Macro-F1 `−0.022085`、Exact Match `−0.001932`。ASL 没有全面提升全部指标。
- B0 e80/e200/e300 的五 seed Validation Macro 分别约 `0.273333/0.278238/0.280447`；它们是不同训练预算，不能当作 15 个独立同配置 seed。e300 是已经完成的最终 baseline，不建议再重复延长 B0 horizon。

证据：`docs/08_reports/s2_gc_closeout_v1/four_stage_summary_v1.csv`、`docs/04_experiments/B0_S1_S2_S2GC_SUMMARY.zh-CN.md`、S3/S4 五 seed closeout；实际轻量结果根分别为 `b0/formal_e300`、`s1/formal_last4_e300`、`s2/formal_multiscale_e300`、`s2/global_capacity_control_e300`、`s3/formal_fsl_e300`、`s4/formal_asl_e300`。

历史发布的 Test Macro-AUPRC：B0 `0.221902±0.000297`，S1 `0.215398±0.003996`，S2-GC `0.233251±0.003676`，S2 `0.255548±0.007798`，S3 `0.239953±0.008870`，S4 `0.262920±0.007760`。S2/S4 Test Macro-F1分别为 `0.257612±0.018137`、`0.241888±0.013479`。这些是原项目已经观察过的内部确认结果，不能作为 Lite 新方案的全新独立 holdout，也不能用来选择 Lite 超参数。

### 2.3 S5–S7：同一 S4 scores 的阈值消融，不是新模型

| 阶段 | 实际变化 | Validation 证据 | 结论 |
|---|---|---|---|
| S5-Free | 47 个逐标签 F1 最优阈值；5-fold cross-fit | OOF Macro-F1：Global `0.297757±0.022528` → Free `0.387691±0.007699`；5/5 正 | Macro-F1收益明确；Micro-F1 `0.540748→0.481149`、Jaccard `0.430824→0.332263`、Exact `0.177627→0.015424`，代价明显 |
| S6-Regularized | `t_c=t_global+λ(t_free,c−t_global)`；旧选择规则 | 25 fold fits 和5个 full fits 全选 λ=0 | 完全退回 Global，未保留 Macro 增益；当前 selector 不迁移 |
| S7-Retention | 同一 shrinkage，要求保留50% Macro-F1收益，再最大化 Jaccard | full-Validation refit平均保留53.36% Macro增益、恢复72.71% Jaccard损失；5/5达到原标准 | 正向折中策略，仍有 secondary trade-off；不是普遍最优 policy |

S7 full-Validation selected λ为 `[0.65,0.75,0.70,0.70,0.65]`；cross-fitted 25 个 calibration fits 的均值是0.704。它们是不同统计对象，不能把0.704或众数0.70当成所有样本、所有 seed 的固定最优值。

S5的 OOF 结果是在held-out fold上合并评价；S7 primary是full-Validation refit。二者结果不能在不同 scope 下直接比较出“谁的泛化最好”。既有Test Macro-F1汇总：Global `0.241888`、S5-Free `0.271826`、S7 `0.270657`；仅作历史描述，不能由此选 Lite policy。

证据：`S5_C_LABELWISE_THRESHOLD_CLOSEOUT.zh-CN.md`、`S6_C_REGULARIZED_THRESHOLD_CLOSEOUT.zh-CN.md`、`S7_C_RETENTION_CONSTRAINED_POLICY_CLOSEOUT.zh-CN.md`；本次另读了同步的 S7 `full_validation_selection.csv`，未读取任何score NPZ。

### 2.4 Geometry-eligible 人群：序列仍是最强可靠锚点

| 模型 | seed/预算 | 核心变化 | Val Macro-AUPRC | Val Micro-AUPRC | Val Macro-F1 | 结论与边界 |
|---|---|---|---:|---:|---:|---|
| S8-B | 0–4 / 300 epochs | S4式S2序列架构在54,088子集重训 | 0.355738 ± 0.009933 | 0.465539 ± 0.016088 | 0.316579 ± 0.018542 | 匹配子集的最佳比较锚点；best epochs=6,8,7,9,8 |
| S8-G | 0–4 / 300 | 2层GVP-like geometry→1280；representation residual，joint训练 | 0.348489 ± 0.007113 | 0.474699 ± 0.020458 | 0.304047 ± 0.014137 | G−B Macro均值−0.007249，只有1/5正；未保留胜过B的证据 |
| S8-GC | 0–4 / 300 | sequence representation→1280→64→1280容量残差 | 0.345377 ± 0.005338 | 0.464198 ± 0.013783 | 0.307056 ± 0.020837 | GC−B Macro均值−0.010361，只有1/5正；best epochs=6,11,12,7,6 |
| S9-I0 ResGeo-FLR | 0–2 / 300正式epochs | 稀疏typed几何、sequence-geometry交互、function queries、label attention、gate residual | 相对同seed S8-B mean delta=−0.010695；0/3正 | 未取得完整汇总 | 未报告 | `TIER2_PROMOTION_NOT_MET`；用户提供paired summary；不能称整个机制分别被证伪 |
| D1 MAIN | 0–2 / 60 | geometry-only typed blocks + 47 function query readout | 0.150460；单seed见下 | 未报告完整汇总 | 未报告 | MAIN−mean Control平均+0.027707，3/3正；证明几何单独有信号，未证明对序列有增量 |
| D2 fixed-α | 0–2 / 60 | frozen S8-B + 可训练同seed D1几何logit residual | 0.343684 ± 0.012563 | 0.447403 ± 0.016811 | 未执行阈值评价 | 对三seed S8-B平均delta−0.007483，1/3正；`D2_NOT_GO` |
| D3-A θ-only | seed0 / 10 | frozen S8-B与D2-selected几何；只学习一个有界尺度 | 0.3266669229277257（best epoch1） | 本地未取得可靠best值 | 未评价 | `FEASIBILITY_PASS`仅为工程可行；未超过seed0 D2/S8-B |

D2 Micro均值/SD由下表权威Validation参考值计算，不能当作新推理。S8数据来自15个正式运行的closeout与配对CSV；本次读取了seed0 resolved config，未重新验证权重。S9-I0结论来自 `PHASE_1_S9_I0_F.md` 末尾用户结果记录，其完整各seed Micro/best-epoch资料不足。

同seed直接比较（源：`configs/evaluation/s9_diag01_input_authority_v2.json`）：

| Seed | S8-B selected epoch | S8-B Macro / Micro | D2 selected epoch | D2 Macro / Micro | D2−B Macro |
|---:|---:|---|---:|---|---:|
| 0 | 6 | 0.355715401061 / 0.449233634950 | 1 | 0.329614075954 / 0.437609781995 | −0.026101325107 |
| 1 | 8 | 0.355197305880 / 0.482266375185 | 2 | 0.353778068334 / 0.437785746579 | −0.001419237545 |
| 2 | 7 | 0.342588870739 / 0.468529416417 | 2 | 0.347659783491 / 0.466814940784 | +0.005070912752 |

三seed S8-B reference均值为0.351167193；它不同于五seed S8-B均值0.355737865。不能挑选两种seed范围中更好看的值混用。

D1 MAIN/Control-1 Macro：seed0 `0.143488605365/0.125368809543`，seed1 `0.156433429319/0.120410461885`，seed2 `0.151457409318/0.122480300121`；MAIN best epochs=35,53,29。prevalence baseline=0.059679858145。此表仍保持既有报告的“用户服务器证据”边界。

已安全提供的三个D2 60-row histories记录 `epoch,train_asl,val_macro_auprc,val_micro_auprc` 等：

| Seed | Epoch1 Train ASL → Epoch60 | best Macro → final Macro | best时 Micro → final Micro |
|---:|---|---|---|
| 0 | 0.012264786 → 0.003185515 | 0.329614076 → 0.244132686 | 0.437609782 → 0.379465241 |
| 1 | 0.007963072 → 0.002349844 | 0.353778068 → 0.283752395 | 0.437785747 → 0.404833273 |
| 2 | 0.011213526 → 0.003222093 | 0.347659783 → 0.266249238 | 0.466814941 → 0.417744209 |

这支持“持续优化与Validation ranking退化具有时间关联”，不确证残差尺度过大、ASL是唯一原因或结构无用。history中的 `trainable_parameter_count=53` 实际与53个梯度参数张量相对应，不是网络只有53个scalar；真实D1模型参数量为1,373,367。

D3-A最终θ≈0.821654、α≈0.165594以及逐epoch指标下降，是负责人后续提供的服务器证据；本地closeout早先写的是这些数值尚未取得。本文采用较新的已提供事实，同时不宣称本地有完整10-row Pilot history。DIAG01已签署，但当前没有正式九点α结果，不能宣称已经知道最优α或47标签收益分布。

### 2.5 B1、A1–A4与历史命名

`B1` 在原路线中是Function Query设计阶段，后来明确拆成S1→S2→S3；没有找到独立B1完整训练结果。原B2–B8描述是设计路线，不能补造为已完成实验。

当前源码、配置和可见Git历史没有能够对应“独立模型A1、A2、A3、A4”的四组可比较正式结果。仓库中的D2-A1是artifact/comparator审计，不是一个新模型；D2-C1是smoke/runner准备。若外部曾用A1–A4命名四种模态或消融，需要原始映射证据才能关联，本文不自行赋予架构和分数。

### 2.6 简要Git演进位置

| 节点 | 可核实含义 |
|---|---|
| `913e5dd` | B0序列模型实现 |
| `19a8678` | S1-last4实现 |
| `f5bd9d8` / `a7ed3fd` | S2实现及Canonical序列join修复 |
| `e7f074e` | S2-GC容量对照 |
| `6af8d10` / `829f826` | S3实现 / residue provenance节点 |
| `20df42186c45ccf69d0cd17d2fa442c150f0cfc9` | S4实际结果记录的training commit |
| `6e1262bca129d508a5ae771c2b9f947c93eb8008` | S8正式训练身份/容量配置节点 |
| `0423126` / `70ea338` | S9-I0 NextGen / static geometry cache |
| `d7e7c2c` / `4454d5695aedf1dcb6da2a0ded23f035bf7f1a2e` | D1实现 / 输入dispatch修复；后者也是D2 payload记录的启动Git HEAD |
| `a5eb215`、`e389a06`、`e582a0d` | 后补录D2 / D3数据路径 / D3运行快照；不是对历史训练源码完整覆盖的追认 |

Lite可以使用当前代码中已验证的计算逻辑；不应把后补录commit当作过去实验实际执行时的完整代码身份。

## 3. 最优已验证模型的选择依据

| 用户关心的“最好” | 可支持的结论 | 不能越过的边界 |
|---|---|---|
| 综合表现 | 以原Primary Macro-AUPRC为主，推荐S4作为Lite主干；保留S2+BCE作为指标权衡参考 | 没有事先定义全指标综合utility；不存在已证明在Macro、Micro、F1、Exact上都最优的模型 |
| Validation Macro-AUPRC | Sequence-eligible内S4五seed mean=0.359015602599；单seed最高是S4 seed4=0.365853819755 | 单seed最高不等于更可靠的方法；不能把“选最佳seed”冒充五seed均值或公平协议 |
| 多seed稳定性 | B0 Macro seed SD最小=0.000241；高性能路线中S4 SD=0.004451，低于S2/S3 | 初始化稳定不等于不同数据集、split稳定；只有一个固定split |
| 性能/开销 | S4免去几何输入和联合网络，较几何路线更简洁；B0只有60,207可训练参数，是最低训练头开销选项 | 缺少同设备独占条件的全面延迟/能耗测量，不能宣称严格Pareto最优；ESM编码开销仍在 |
| 直接迁移Lite | **S2MultiScaleMotifModel + compute_s4_asl_loss + 原Validation selector/metric** | 不直接迁移旧Runner治理壳；S4+S7是可选决策组合，未被原项目自动选为production |

首版建议保存global阈值结果；如明确重视Macro-F1，可保留S5/S7后处理作为配置选项，同时报告sample-level trade-off。不能只报告最漂亮的阈值指标。

### S4可以直接复用的五个checkpoint

共同历史输出根：`s4/formal_asl_e300/seed_{s}/checkpoints/s4_asl_best_seed_{s}.pt`。本地轻量结果含服务器路径；checkpoint binary本轮未加载，不声称本地都有权重。

| Seed | Best epoch | Val Macro | Val Micro | global threshold | best checkpoint局部尺度g |
|---:|---:|---:|---:|---:|---:|
| 0 | 5 | 0.354063728918 | 0.457219127426 | 0.70 | 0.170372858644 |
| 1 | 8 | 0.357017263891 | 0.463804213651 | 0.66 | 0.219766810536 |
| 2 | 4 | 0.357665109692 | 0.473543580427 | 0.73 | 0.154541537166 |
| 3 | 7 | 0.360478090738 | 0.475583964918 | 0.72 | 0.198711037636 |
| 4 | 5 | 0.365853819755 | 0.508389199744 | 0.71 | 0.168960526586 |

这些阈值和g是对应checkpoint的结果，不是新模型应固定使用的“最优超参数”。新训练仍初始化原raw local_scale=-2，并在各自Validation拟合阈值。没有新验证时，也不能把seed4权重的性能外推为生产泛化表现。

## 4. 最优模型完整架构与数据流

### 4.1 数据与编码器

1. 读取已规范化的 `plm_normalized_sequence`；标签单独读取为 `[B,47]`。原序列及真实化学身份保留。
2. `B0BatchCollator`调用Hugging Face tokenizer：`padding=True,truncation=True,max_length=256,return_special_tokens_mask=True`。
3. `residue_mask=attention_mask.bool() & ~special_tokens_mask.bool()`，不把BOS/EOS/padding计入pooling。
4. frozen ESM-2处于eval、no_grad；读取 `outputs.last_hidden_state`（已核验的layer33输出），FP32。
5. B0 global缓存是masked-mean `[N,1280]`；S2 residue缓存是FP32 `[R_total,1280]` + int64 offsets `[N+1]`。ID、split、target列序一致。
6. 训练时按indices取global行，按offsets取residue片段，batch内右侧零padding为 `[B,L_batch_max,1280]`，布尔mask为 `[B,L_batch_max]`。

**首轮复现应继续使用独立的B0 global特征。** S2生成器做过抽样mean-equivalence（`atol=rtol=1e-5`），但这不证明所有样本全量逐位一致。旧项目明确未把S2 mean候选认可为原B0的无损替代；Lite不能仅因维度相同就替换已有global缓存。未来从同一次明确的encoder输出同时构建global/residue可简化新缓存，但必须说明其与历史特征的数值边界，不能未经复现就冒称原S4完全相同。

### 4.2 真实forward公式

定义global为 `G∈R^(B×1280)`，residue为 `R∈R^(B×L×1280)`，mask为 `M∈{0,1}^(B×L)`。为避免混淆，下面用β表示原模型代码名为`alpha`的局部raw scalar：

\[
A_k=\operatorname{GELU}(\operatorname{Conv1d}_k(R^\top)),\quad k\in\{3,5,7\},
\]

\[
v_{k,b,c}=\max_{\ell:M_{b\ell}=1} A_{k,b,c,\ell},\qquad
u=\operatorname{LN}_{\text{no affine}}(\operatorname{GELU}(W_p[v_3;v_5;v_7]+b_p)),
\]

\[
g=\sigma(\beta),\quad\beta_{init}=-2,\qquad h=G+g u,\qquad z=W_c h+b_c,\quad p=\sigma(z).
\]

`g`是全模型共享的一个可学习scalar，不是每标签gate。G本身不被局部卷积再次编码；分类头在Train上与CNN、projection和β联合训练。

### 4.3 Tensor shape与连接

| 实际步骤 | 输入 | 输出 | 关键约束 |
|---|---|---|---|
| tokenizer/ESM | `[B,T]` tokens/masks | `[B,T,1280]` | 特殊token只在encoder上下文存在，pooling排除 |
| global masked mean | residue states +mask | `[B,1280]` | denominator为有效residue数，FP32 accumulation |
| residue collate | ragged slices | `[B,L,1280]` + `[B,L]` | zero padding；每样本至少一个valid residue |
| transpose/3个Conv1d | `[B,1280,L]` | 各`[B,128,L]` | stride1,padding=k//2,bias=True |
| GELU + masked max | 各scale activations | 各`[B,128]` | invalid time positions设`−inf`后max |
| concat | 三个pooled向量 | `[B,384]` | scale顺序3,5,7固定 |
| projection/GELU/LN | `[B,384]` | `[B,1280]` | LayerNorm无affine |
| residual fusion | global +局部 | `[B,1280]` | FP32；global + sigmoid(β)×local |
| classifier | `[B,1280]` | `[B,47]` logits | 无softmax，不强制单标签 |
| sigmoid/metrics | `[N,47]` | 概率/排名 | GPU sigmoid FP32后CPU numpy，指标再转float64 |
| threshold | 概率 + scalar或47-vector | bool`[N,47]` | 比较`p>=t`；不强制top-k或至少一个positive |

mask只在pooling屏蔽无效position。原模型没有在每个conv前重新mask输入；因此collator的**零padding**是计算语义的一部分，不能用任意非零pad向量替换。

## 5. 各模块数学原理、参数与实现

### 5.1 Frozen ESM-2

- 实际模型：`facebook/esm2_t33_650M_UR50D`，revision=`08e4846e537177426273712802403f7ba8261b6c`；33层、hidden1280。
- 采用Hugging Face `AutoModel`/`AutoTokenizer`；底层encoder和tokenizer由该revision确定。不要把fair-esm或其他layer输出接口直接视为相同实现。
- frozen eval、no_grad；没有验证PEFT、last-K解冻、full fine-tuning、ProtT5或双PLM增益。
- 原S2缓存提取的batch=16与B0历史batch=512属于执行批大小差异；不能要求它们相等，但也不能据此声称所有FP32输出逐位相等。
- `max_length=256`是tokenizer长度上限，包含special tokens；不是“保留256个residue”。原raw序列审计长度2–74；S2构建器要求valid token count等于normalized sequence length，拒绝静默截断。

代码：`src/trifusion/models/b0_sequence.py::{B0SequenceModel,masked_mean_fp32}`，`src/trifusion/data/b0_batch.py::{B0BatchCollator,build_residue_mask}`，`scripts/30_build_b0_embedding_cache.py`，`scripts/36_build_s2_residue_cache.py`。

### 5.2 Multi-scale contextual local branch

该分支不是raw one-hot motif扫描器。卷积作用于已带上下文的ESM residue embeddings；kernel3/5/7提供不同邻域长度，masked max挑选强局部activation。

| 组件 | 参数数 |
|---|---:|
| 三个Conv1d（1280→128） | 2,457,984 |
| Linear384→1280 | 492,800 |
| LayerNorm1280，affine=False | 0 |
| raw scale β | 1 |
| Linear1280→47 | 60,207 |
| **合计** | **3,010,992** |

GELU使用PyTorch默认`approximate="none"`；LayerNorm默认eps=1e-5，未经过单独优化。局部模型没有Dropout模块。不能擅自加dropout、affine LN、extra projection、attention替换masked max，再称原成熟配置。

S2-GC证明global nonlinear capacity本身有收益；S2相对近似等参数GC还有稳定Macro收益。由此保留local branch。没有真实motif注释、位点扰动或生物学验证，不能称已经发现功能motif。

### 5.3 ASL：必须保留focusing detach

实际 `compute_s4_asl_loss`：令 `p=σ(z)`、`n=min(1,1−p+0.05)`、`q=1−n`，

\[
L_{base}=-y\log(\max(p,10^{-8}))-(1-y)\log(\max(n,10^{-8})),
\]

\[
w=y(1-p)^0+(1-y)q^4,\qquad
L=\operatorname{mean}_{B\times47}(L_{base}\cdot\operatorname{stopgrad}(w)).
\]

- `gamma_pos=0,gamma_neg=4,clip=0.05,eps=1e-8`。
- log/base部分保留梯度，**只有focusing factor detach**。直接使用一个同名第三方ASL类不能保证计算相同。
- mean除以实际最后batch的`B×47`，不按positive数量归一化，不按label单独重加权。
- 不使用pos_weight、class-balanced sampler、额外hierarchy loss或经验label关系loss。
- 这是既有ASL思想的项目实现/验证，不是本项目提出的新损失理论。五seed实验支持其对当前Macro AP主目标的价值，不能推导它直接优化AP或改善概率校准。

### 5.4 AP/AUPRC的真实定义

实际函数位于 `src/trifusion/metrics/b0_metrics.py`：

- `average_precision`用 `np.lexsort((np.arange(N),-scores))` 排序。
- 单标签AP=`Σ_{rank为positive}precision_at_rank / positive_count`，不是trapezoidal PR面积。
- **同分数按原行index稳定排序，未先按tie score组聚合。** 因此不能默认等于sklearn `average_precision_score`的tie语义；样本顺序是复现输入之一。
- 无正例标签返回NaN，在结果中保存None，Macro用`nanmean`只平均有正例的labels；不是无正例AP记零后除以47。
- Micro把`[N,47]`按C-order展开后计算同一AP；label列序及sample行序都影响同分数的tie顺序。
- logits先由原设备的FP32 sigmoid转概率，再转numpy/float64评价。用float64 logits直接重新sigmoid可能改变饱和处ties；精确历史复现应保留这条路径。

### 5.5 F1与旧命名的实际细节

`thresholded_metrics`的Macro-F1等权平均47个label；源码使用`np.divide(...,out=np.ones(...),where=denominator!=0)`，所以某label的`TP=FP=FN=0`时F1为1。Micro-F1空denominator记0，sample Precision/Coverage/Jaccard空denominator也记0。

这一Macro-F1细节与源码注释“Empty denominators contribute zero”及部分zero-division文字不一致。它不影响当前AP结论，但可能影响rare-label或cross-fit F1解释。**精确历史复现保留此实现并明确披露；若Lite采用标准zero_division=0规则，须单独命名评价协议并与旧值分开，不能直接声称数值等价。** 本次不修改原指标代码。

旧 `legacy_accuracy` 实际是sample-wise Jaccard；`legacy_precision/coverage`是sample平均，而新版micro_precision/micro_recall是全局TP归约。不要用“Accuracy”或“Coverage”名称误比较不同公式。

### 5.6 阈值与MFP关系

Global：Validation grid0.05–0.95，步0.01，最大化sample-wise Jaccard；tie依次最小symmetric difference、最靠近0.5、再较小阈值。

S5：每label最大化calibration F1；tie最靠近该fold global，再较大阈值；calibration无positive或无negative时回退global。5-fold cross-fit只适用于threshold evaluation，不能消除此前同Validation选checkpoint的影响。

S7：`t_c(λ)=t_g+λ(t_free,c−t_g)`；λ为0,0.05,…,1。要求Macro-F1保留free−global增益至少50%，再最大化calibration Jaccard，tie选更小λ；没有有效reference/feasible候选时明确失败。不能把旧S6所有λ=0结果叫作成功正则化。

输出47个二态标签，可有零个positive。`predicted_label_count>=2`只能叫多标签预测数量；`configs/ontology/mfp_independence_v1.yaml`的`independent_pairs=[]`，正式生物学MFP独立关系尚未建立。不能以预测出两个层级相关标签就宣称已证明多功能肽。15个major group映射也仍为`groups: []`，主干不使用它们。

## 6. 已实现且实验验证有效的创新点

| 资产 | 支持证据 | 适用范围 / Lite处理 |
|---|---|---|
| Frozen global+multi-scale local contextual融合 | S2超过B0，并在近似等参数S2-GC对照中五seed Macro全胜 | 默认保留；有效的是当前组合，不把每个kernel分别宣称已验证最优 |
| 在相同S2网络上采用ASL | S4−S2 Val Macro五seed全正、mean+0.028711 | 默认保留；保留focusing detach，披露F1 trade-off |
| Label-wise decision boundary | S5 OOF Macro-F1五seed全正 | 可选后处理，不进forward；不默认替代global |
| Macro-retention constrained shrinkage | S7非端点选择与retention/recovery通过 | 可选决策研究资产；不硬编码全局λ=0.70 |
| D1 geometry relation-aware+function readout | geometry-only MAIN超过mean control，3/3 | 保留为可选研究参考，不放入Lite主路径，不声称融合增益 |
| Frozen features/caching | 实际多seed训练可在cache-native小网络上完成；S9 static/dynamic已有parity证据 | 复用离线缓存策略，不继承哈希审批壳；缓存不是新的预测算法 |

ESM-2、CNN、ASL、attention、GVP都是已有方法；项目价值主要是受控的组合设计和机制消融证据，不能把这些模块的名字包装成原创理论。

## 7. 已实现但无效或证据不足的创新点

### 7.1 不进入Lite默认路径

- **S1-last4全局LayerMix**：Macro不胜B0；不重复相同last4实验。其他layer selection方案没有自动被否定，但没有可迁移的正结果。
- **S3当前Function Query local attention**：0/5胜S2；不把query bank误列为最优序列模型的核心组件。attention pooling、query初始化与其他因素共同改变，不能归因到“所有function specificity都无效”。
- **S8-G整体GVP融合**：G−B Macro为负，复杂度/数据要求增加。不能用G−GC的4/5胜忽略G未胜sequence baseline。
- **S9-I0复杂ResGeo-FLR**：sample-conditioned gate、label attention等已实现，但组合未晋级；没有单模块消融足以说明其中每块独立收益。
- **D2固定正α logit残差**：D2_NOT_GO，且后期Val退化；不重复同一3-seed60epoch训练，不在Lite默认挂上几何logits。
- **D3-A θ-only优化**：单seed工程PASS、Macro未改善；ASL推动α增加与Val下降相伴，不证明“减小α必然好”。
- **S6旧regularized selector**：全部λ=0；不照搬其失败selection policy。

### 7.2 已实现但验证范围有限

- D1有geometry-only信号，但没有完整的sequence-incremental证据。
- D3-A只有seed0 Pilot，不能作三seed确认结论；finite θ对应α严格大于0，不能真正把现有wrapper设成α=0精确旁路。
- D2真实α=0分支直接返回S8-B tensor，已取得三seedValidation控制点证据；这验证恢复/旁路计算，不证明非零残差有益。
- D3-A θ=0在**共享中间量fusion**上应与D2 α0精确一致；两次独立GPU expert forward存在微小数值波动，不能将其当作权重不同。R3-v2保留final logits1e-5、metrics1e-6；其规则修改是已知FIX02后提出的初始化验收，不是独立性能证据。
- DIAG01程序/合成测试/审批已经完成，实际九点α结果未取得；不能据此保留新的label/sample gate作为有效模块。

## 8. 尚未实现或尚未取得正结果的创新方向

以下是研究候选，不是Lite首版必建组件。优先级是本文建议，依据当前证据而非新实验。

| 候选 | 数学/实现构想与解决问题 | 与成熟主干关系 | 复杂度 / 优先级 / 风险 |
|---|---|---|---|
| 数据支持的label-conditioned融合 | 对同seed冻结专家用共享小网络或受约束label尺度：`z_ic=z_seq,ic+a_ic δ_ic`；共享先验/稀疏正则控制长尾自由度 | 可外挂S4/S8-B；先要证明per-label residual异质性 | 中；条件性P1。D3-B草案258→16→1共享MLP共4,161参数；尚未训练，不能照搬为默认 |
| 可靠性感知sample融合 | `a_i=f(validity/coverage/quality,expert disagreement)`，缺结构精确回退 | 辅助分支可选 | 中；P2。有效mask不是可信quality或uncertainty，来源/质量缺失仍是障碍 |
| ASL与ranking目标的对照 | 在同架构上单独考察ranking surrogate或校准；不与新geometry gate一起改 | 可替换loss插件 | 中；条件性P1。现有负迁移只给目标冲突动机，没证明某新loss能改善AP |
| PEFT/小型PLM/蒸馏 | 用成熟S4为reference，减少在线650M成本 | encoder可替换，需重新验证token/features | 中高；真正资源受限时P1，否则P2。没有已验证更小encoder保持成绩的证据 |
| Chemistry/PTM specialist | 保留D-AA、端基、非标准monomer/HELM语义；先descriptor baseline | sequence保持mandatory，chemistry可选 | 中高；P2。预处理代码已有，encoder与融合有效性未证实 |
| Train-only标签关系/低秩共享 | 用Train support/co-occurrence构造关系，或共享低秩label表示 | head插件 | 中；P2/P3。没有当前成熟图/经验relation收益；禁止用Val/Test拟合关系 |
| ProtT5双PLM | 新编码器再融合 | encoder接口允许以后替换 | 高；P3。当前没有正结果，首版不开双encoder |
| Hierarchy/coarse-to-fine/MoE/distillation | 原大架构中的group/query/retrieval/teacher计划 | 非首版 | 高；P3。group与独立关系语义未完成，复杂度大且归因困难 |

最值得保留的未决研究任务是已经准备好的冻结专家α/per-label诊断；它能够减少盲目增加融合模块的风险。但Lite核心复现不依赖诊断完成，也无需为这项迁移重建旧隔离治理框架。

## 9. 数据处理与实验协议

### 9.1 therapeutic_2025与标签语义

实际源是 `raw/therapeutic_2025/main.xlsx` 的Data表，58,583个ID，11个字段；不是一个已经由项目证实的公有benchmark排名。Canonical构建保留所有ID，没有把ineligible样本从原数据删除。

Raw `Label encoding`有48个二态位置；`Growth_Regeneration/Growth_regeneration`两个别名按logical OR合并，形成47维。不要从Function层级path补充未标注positive：训练目标是已观察标签位，0表示当前未标注/未观察，而不是完备生物学阴性证明。

固定47列顺序（位置为1-based，实际矩阵index为position−1）：

| 位置 | label | 位置 | label |
|---:|---|---:|---|
| 1 | Analgesic | 25 | Antimalarial |
| 2 | Angiogenic | 26 | Antimicrobial |
| 3 | Anti-gram+ | 27 | Antioxidant |
| 4 | Anti-gram- | 28 | Antiparasitic |
| 5 | AntiAngiogenesis | 29 | Antiplasmodial |
| 6 | AntiBreastcancer | 30 | Antitrypanosomic |
| 7 | AntiCervicalcancer | 31 | Antiviral |
| 8 | AntiColoncancer | 32 | Blood-brain_barrier_penetrating |
| 9 | AntiHCV | 33 | Cell_Communication |
| 10 | AntiHIV | 34 | Cell_penetrating |
| 11 | AntiHSV | 35 | Drug_delivery |
| 12 | AntiLivercancer | 36 | Glucose_metabolism |
| 13 | AntiLungcancer | 37 | Growth_Regeneration |
| 14 | AntiMERS-Cov | 38 | Growth_regulatory |
| 15 | AntiProstatecancer | 39 | Immunoactive |
| 16 | AntiSARS | 40 | Lipid_metabolism |
| 17 | AntiSkincancer | 41 | Metabolic_regulatory |
| 18 | AntiTubercular | 42 | Neuropeptide |
| 19 | Antibacterial | 43 | Opioid |
| 20 | Anticancer | 44 | Osteogenic |
| 21 | Antifungal | 45 | Quorum_sensing |
| 22 | Antihypertensive | 46 | Thrombolytic |
| 23 | Antiinflammatory | 47 | Tumor_homing |
| 24 | Antileishmania | — | — |

来源：`configs/ontology/label_schema_v1.yaml`与`label_aliases_v1.yaml`。不要从字母排序重建列顺序。

### 9.2 序列规范化

`normalize_sequence`保留original；PLM副本执行NFKC、移除whitespace、标准ASCII残基大写；非标准ASCII字母映射X；非法符号/非ASCII可生成X但标记invalid_for_plm。小写残基的D-AA解释保留warning，不能仅凭大小写宣称真实化学身份。

最终eligibility使用Canonical `sequence_plm_available`，不是重新从文本状态猜测；1,195行不可用于原PLM路径，因此直接比较集是57,388。不要把excluded行当成模型失败或negative样本，也不要把D-AA映射后的序列当作真实L-AA化学证明。

代码：`sequence_modification_normalization.py::normalize_sequence`、`canonical_dataset.py`、`b0_eligibility.py::derive_eligibility_rows`。HELM语义修复仅供canonical/chemistry记录，最优网络不消费PTM/HELM输入。

### 9.3 去重与同源隔离

- Canonical保留58,583行；原raw观察有7组重复Sequence、14行。处理重点是分组隔离，不是任意删除重复ID或混合标签。
- 正式split使用exact原序列、相同PLM序列及通过homology验证的连接分量；同分量不可跨split。
- identity≥0.80，双向coverage≥0.90；split seed20260920，目标比例80/10/10；固定assignment最终counts见2.1。
- MMseqs2用于candidate discovery，不直接把其cluster输出当作最终split；project-owned Needleman–Wunsch验证使用match+1、mismatch−1、gap−1及确定性tie；含X位置不计identity的comparable denominator，双向coverage按non-gap paired residues/原长度计算。
- short peptide候选补充和既有audit有特定长度边界；不要用一个简化“MMseqs random group split”替代原assignment，并声称数据协议相同。
- 保留原assignment最省工作。accepted warnings包括Train序列更短、label cardinality更低，不能为了分布好看重划后继续比较旧成绩。

源码：`homology_sensitivity.py::{global_alignment_metrics,verify_candidate}`、`formal_split.py`；正式assignment为 `splits/v1/split_assignment_v1.jsonl`。构造split可以做事前stratification；它不授权从Val/Test构建模型label关系。

### 9.4 原缓存与Lite输入建议

| 源 | 实际格式/用途 | Lite应如何处理 |
|---|---|---|
| B0 | `interim/feature_cache/b0_embedding_v1/b0_embedding_cache_v1.npz`；global、IDs、splits、targets的混合压缩cache | 原始训练可整体物化；新Lite优先明确split shard，避免训练代码解压Test payload |
| S2 | `s2_residue_v1/`中的residue_values、offsets、IDs、splits、targets NPY | 保留ragged布局，mmap只选当前split；可在Lite准备阶段生成独立split arrays |
| D3已隔离B0 | `d3a_b0_train_validation_isolated_v1/{train,validation}_{embeddings,ids}.npy` | 仅覆盖geometry子集48,589 TV行，不能当成57,388主实验全体的替代 |
| S9 static geometry | node/edge features、ptr、local index、mask的NPY | 首版不需要；未来结构实验用既有row mapping，不在线重建每batch全图 |

原S4 loader会打开全cohort B0 NPZ与S2 cache并核对targets；随后只有Train行用于梯度、Validation用于选模/阈值，Test在最后评价。**物化Test特征与Test参与训练是不同事实。** Lite可改为split文件，从数据布局自然实现train/val/test分离；这属于输入工程简化，不能顺带改变特征值、列序和人群。

现有D3 reader允许TV逻辑切片，S2/S9仍是mixed-file mmap；没有“物理零Test字节I/O”证明。过去B0 split导出经历一次获批的混合NPZ解压接触Test embedding，不能改写成端到端零Test接触。Lite无需复制这种审批体系；从起始准备split-specific特征即可保持清楚的数据访问边界。

### 9.5 结构、3Di与质量

有PDB文件名候选54,722；结构可解析54,301，421是HTTP404内容；sequence-structure mapping EXACT54,086、chemistry-aware2，合成54,088几何人群。filename命中不等于有效geometry。

S8用N/CA/C/O `[N,4,3]`，CA kNN16、cutoff20Å，node scalar6/vector3、edge distance+16RBF/vector方向；2个GVP-like block，scalar64/vector16，mean readout后投影1280。

S9/D1改为node24、edge32、typed稀疏关系：backbone、12Å spatial top8、8Å contact且sequence separation>3、long-range separation≥5且distance≤12Å。D1 hidden256、heads4、blocks2、FFN512、47 queries、dropout0。

有atom-validity、node mask、angle/torsion有效性等信息，可以描述缺失/完整度；它们不是完整的结构置信度或source authority。S8 cache原source字段UNKNOWN，后验头部审计有28,788明确ESMFold证据、25,300未解析，不能据此把全cache标为高质量实验结构。没有确认3Di/SaProt编码在当前正式模型里使用；它们是研究候选。

### 9.6 正确性必需项与可舍弃工程措施

必须保留：immutable raw概念、明确ID/47-column mapping、同源split、feature/target对齐、mask、Train-only更新、Val-only选择、独立Test评价、固定seed与评价公式、完整训练曲线、结果不覆盖。

Lite可舍弃：审批manifest、状态机、逐文件代码SHA、版本号副本堆叠、server preimage安装器、重复identity层、命令PLAN_ONLY包装、自动GPU空闲判定、multi-stagefreeze工具。用一个普通YAML、Git commit、简单run metadata和独立run目录即可。

## 10. 关键训练、验证和推理参数

下面“验证”表示该值在有效组合中使用，不表示它经过单因素搜索而证明最优。

| 参数 | 直接复现S4的值 | 证据/状态 |
|---|---|---|
| 编码器/revision | ESM-2 t33 650M，`08e4846e537177426273712802403f7ba8261b6c` | 真实特征生成身份；未比较所有encoder |
| feature layer/dim | `last_hidden_state`，33层输出，1280 | 源码与缓存生成逻辑 |
| max_length | 256 tokenizer tokens，padding/special-token mask | 实际使用；不是已搜索最优 |
| encoder精度/状态 | FP32，frozen，eval/no_grad | 正式cache与训练边界；不启用训练AMP |
| local kernels/channels | `[3,5,7]` / 每scale128 | 有效组合，没做逐kernel最优搜索 |
| local projection | 384→1280，GELU；LN affine=False,eps默认1e-5 | 原模型实际实现 |
| local raw scale init | −2，`g=σ(β)` | 必须与geo α和D3 θ区分 |
| head/dropout | Linear1280→47；local dropout=0 | 实际使用，未单独优化 |
| physical/effective batch | 256/256 | 源码与五个S4结果 |
| gradient accumulation | 1 | 一batch一step；末尾batch保留 |
| optimizer | AdamW；lr=0.001，wd=0.0001 | 原默认被正式使用；非搜索最优 |
| AdamW betas/eps | (0.9,0.999) / 1e-8 | S4真实training_environment记录 |
| amsgrad/foreach/fused | false / None / None | 历史Torch defaults；非额外算法 |
| loss | S4-ASL，γ+=0,γ−=4,m=0.05,eps=1e-8，focusing detach | 相对BCE有明确Macro增益；未搜索其他ASL设置 |
| reduction | 实际B×47 mean | 不能用全局sum或batch-size固定除数 |
| epochs | 300完整epochs；保存Val-best | 历史可复现预算；best常在4–8，不等于事前只训8epochs已验证 |
| scheduler / clipping / early stopping | 无 / 无 / 关闭 | 保持成熟协议；不能直接迁移D3的10epochs |
| seeds | [0,1,2,3,4]，split固定 | Python random/NumPy/Torch/CUDA seed |
| batch shuffle | `default_rng(seed+epoch).permutation(train_indices)` | epoch从1开始 |
| model init | PyTorch默认Conv/Linear init；β确定性−2 | 保留构建顺序以复现同seed初始化 |
| selection | Val Macro降序→Micro降序→早epoch | train loss和Test不选checkpoint |
| validation order | 保持cache原Validation行序，无shuffle | AP tie规则需要稳定顺序 |
| threshold | global grid0.05..0.95，步0.01，max Jaccard | 默认简单策略；S5/S7可选 |
| 微调策略 | 只训local CNN/projection/scale/head；ESM不微调 | 已验证；PEFT未验证 |
| environment | Python3.12.14、Torch2.4.1+cu121、CUDA12.1、Transformers4.44.2、V100 | 历史实际环境；本轮未安装/升级 |

完整序列人群每epoch `ceil(45648/256)=179`个Train batches、24个Validation batches；300epochs共有53,700次预期更新/seed，五seed268,500次。geometry子集每epoch169/22 batches是另一配置，不能套给完整S4。

S8训练capacity记录：S8-B batch256 probe peak allocated约290MiB、reserved310MiB；是Train-only小网络容量探测，不含完整online ESM模型，不是Lite推理显存承诺。S8 formal平均单seedwall runtime B约2.648小时、G4.287小时，受共享服务器/I/O波动影响，不据此做严格速度排名。本次没有新的latency或VRAM测量。

保存/加载：历史S4 checkpoint含 `model_state_dict,optimizer_state_dict,epoch,best_record,history,seed`及缓存身份；state keys是S2类的`local_convs.*`,`local_projection.*`,`alpha`,`classifier.*`，LN无affine权重。旧trainer恢复用`weights_only=False`；本轮不加载旧binary。Lite新保存可用纯tensor state_dict和JSON/YAML metadata，安全加载`weights_only=True`、`load_state_dict(strict=True)`。旧checkpoint的安全加载兼容性没有在本文中被重新验证，不加自动不安全fallback。

## 11. 可直接复用的代码与配置清单

“复用”主要是拷贝核心公式与函数到新的干净模块，不是把整个旧package和其治理依赖图带入Lite。

| 必要资产 | 实际对象/入口 | Lite处理 |
|---|---|---|
| [S2成熟主干](D:/Data/Code/VScode/trifusion/trifusion-mfp/src/trifusion/models/s2_multiscale_motif.py) | `S2MultiScaleMotifModel`,`S2ModelConfig` | 拷贝计算图；配置化但保持默认数值/shape |
| [S4损失](D:/Data/Code/VScode/trifusion/trifusion-mfp/src/trifusion/losses/s4_asl.py) | `compute_s4_asl_loss` | 直接复用数学与detach语义 |
| [官方指标](D:/Data/Code/VScode/trifusion/trifusion-mfp/src/trifusion/metrics/b0_metrics.py) | `average_precision`,`ranking_metrics`,`evaluate_probabilities`,`select_checkpoint_epoch`,`select_validation_threshold` | 精简保留；写清tie/empty-label规则 |
| [S2 batch/train](D:/Data/Code/VScode/trifusion/trifusion-mfp/src/trifusion/training/s2_trainer.py) | `_make_batch`,`_evaluate`,`seed_everything`,`train_s2_multiscale` | 参考数值路径；拆离自动Test与旧cache身份依赖 |
| [S4执行适配](D:/Data/Code/VScode/trifusion/trifusion-mfp/src/trifusion/training/s4_trainer.py) | `build_model_for_seed`,`build_optimizer`,`ASL_CONTRACT`,`train_s4_asl` | 参考超参数；不照搬load_verified_inputs治理 |
| [原S4 CLI](D:/Data/Code/VScode/trifusion/trifusion-mfp/scripts/41_train_s4_asl.py) | 正式入口 | 参考调用；Lite新建一个简单train CLI |
| [原S4预注册](D:/Data/Code/VScode/trifusion/trifusion-mfp/docs/04_experiments/S4_ASL_PREREGISTRATION_DRAFT.zh-CN.md) | 方法参数与评价协议 | 提取到一个新YAML；不是Lite审批链 |
| [B0 encoder/pooling](D:/Data/Code/VScode/trifusion/trifusion-mfp/src/trifusion/models/b0_sequence.py) | `B0SequenceModel`,`masked_mean_fp32` | 提取frozen encoder/mean逻辑 |
| [tokenizer/mask](D:/Data/Code/VScode/trifusion/trifusion-mfp/src/trifusion/data/b0_batch.py) | `B0BatchCollator`,`build_residue_mask` | 保留special token/padding逻辑 |
| [residue提取](D:/Data/Code/VScode/trifusion/trifusion-mfp/scripts/36_build_s2_residue_cache.py) | `_build_residue_values`,`validate_last_hidden_state_contract` | 参考ragged写法，不移植整套manifest流程 |
| [B0缓存提取](D:/Data/Code/VScode/trifusion/trifusion-mfp/scripts/30_build_b0_embedding_cache.py) | 实际global特征来源 | 保留特征定义与精度 |
| [标签schema](D:/Data/Code/VScode/trifusion/trifusion-mfp/configs/ontology/label_schema_v1.yaml) | `canonical_label_order` | 新项目保留一个47-label文件，不复制未确认groups |
| [标签alias](D:/Data/Code/VScode/trifusion/trifusion-mfp/configs/ontology/label_aliases_v1.yaml) | canonical alias规则 | 清洗阶段复用；模型不读alias |
| [序列规范化](D:/Data/Code/VScode/trifusion/trifusion-mfp/src/trifusion/data/sequence_modification_normalization.py) | `normalize_sequence` | 提取纯函数，不带原审计/日志逻辑 |
| [逐标签threshold](D:/Data/Code/VScode/trifusion/trifusion-mfp/src/trifusion/evaluation/s5_labelwise_policy.py) | `fit_thresholds`,`assign_folds`,`cross_fit` | 可选calibration，不默认引入所有formal报表 |
| [S7策略](D:/Data/Code/VScode/trifusion/trifusion-mfp/src/trifusion/evaluation/s7_retention_constrained_policy.py) | retention与lambda selector | 可选，连同所依赖shrinkage数学一起精简 |

原S4没有独立的 `configs/experiment/s4.yaml`：方法值来自 `S2ModelConfig`、`s2_trainer` constants、`s4_trainer.ASL_CONTRACT`及真实结果中的training_environment。不能引用不存在的“原最优配置文件”。

只在研究geometry时参考：

- [S8-G](D:/Data/Code/VScode/trifusion/trifusion-mfp/src/trifusion/models/s8_geometry_expert.py)：`S8GeometryExpert`,`S8GeometryExpertModel`,`build_s8_model`；S8-B最终返回S2模型。
- [D1](D:/Data/Code/VScode/trifusion/trifusion-mfp/src/trifusion/models/s9_geometry_only_d1.py)：`S9GeometryOnlyD1Model`,`S9GeometryMeanControlD1Model`。
- [D2](D:/Data/Code/VScode/trifusion/trifusion-mfp/src/trifusion/models/s9_i1_d2.py)：`S9I1D2FixedScaleResidual`；geo δ是专家logits，不是预先标定的error residual。
- [D3-A](D:/Data/Code/VScode/trifusion/trifusion-mfp/src/trifusion/models/s9_i1_d3a.py)：`S9I1D3ABoundedGlobalResidualScale`；仅作为负/未充分验证研究参考。
- [S9 typed几何](D:/Data/Code/VScode/trifusion/trifusion-mfp/src/trifusion/structure/s9_geometry.py)、[static cache](D:/Data/Code/VScode/trifusion/trifusion-mfp/src/trifusion/data/s9_static_geometry_cache.py)。

历史证据入口：

- [四阶段真实汇总CSV](D:/Data/Code/VScode/trifusion/trifusion-mfp/docs/08_reports/s2_gc_closeout_v1/four_stage_summary_v1.csv)
- [S4五seed closeout](D:/Data/Code/VScode/trifusion/trifusion-mfp/docs/08_reports/S4_C_FIVE_SEED_PAIRED_CLOSEOUT.zh-CN.md)
- [S8五seed closeout](D:/Data/Code/VScode/trifusion/trifusion-mfp/docs/08_reports/S8_C_FIVE_SEED_GEOMETRY_CLOSEOUT.zh-CN.md)及[配对CSV](D:/Data/Code/VScode/trifusion/trifusion-mfp/docs/08_reports/S8_C_METRIC_PAIRED_AUDIT_V1.csv)
- [D1 closeout](D:/Data/Code/VScode/trifusion/trifusion-mfp/docs/08_reports/S9_I1_D1_TIER2_CLOSEOUT.zh-CN.md)
- [D2结果与history分析](D:/Data/Code/VScode/trifusion/trifusion-mfp/docs/08_reports/S9_I1_D2_TIER2_NO_GO_DIAGNOSIS_AND_D3_CONCEPTS.zh-CN.md)
- [D3-A closeout](D:/Data/Code/VScode/trifusion/trifusion-mfp/docs/08_reports/S9_I1_D3_A_SEED0_PILOT_CLOSEOUT.zh-CN.md)
- [三seedValidation参考/身份](D:/Data/Code/VScode/trifusion/trifusion-mfp/configs/evaluation/s9_diag01_input_authority_v2.json)，只是数据/结果来源，不要求在Lite复刻authority体系。

## 12. TriFusion-MFP-Lite最小重构方案

### 12.1 首版目录与职责

```text
trifusion-mfp-lite/
  pyproject.toml
  README.md
  .gitignore
  configs/
    s4_sequence.yaml          # 单一成熟配置；后续变化由Git跟踪
    labels.yaml               # 固定47列与目标语义
  src/trifusion_lite/
    data.py                   # split dataset + ID join + ragged collate
    encoder.py                # frozen ESM-2 adapter与特征提取
    model.py                  # global-local CNN + scalar fusion + Linear47
    losses.py                 # exact S4-ASL；BCE仅用于已有参考
    metrics.py                # AP/F1/selector；显式历史兼容语义
    thresholds.py             # global默认；label-wise/S7可选
    train.py                  # Train更新 + Validation选checkpoint
    evaluate.py               # common pure inference/evaluation函数
  scripts/
    prepare_features.py
    train.py
    calibrate.py
    test.py
    predict.py
  tests/
    test_model.py
    test_data_alignment.py
    test_loss_metrics.py
    test_checkpoint.py
```

只有这几个直接可替换的函数/模块；不先建立registry/plugin框架、继承层级、factory链和多版本状态对象。`encoder`返回global/residue/mask；`model`接受这些tensor返回logits；`loss`接受logits/targets；`thresholds`处理probabilities。没有必要构建泛化到所有模态的大型接口。

### 12.2 可直接实施的配置值

```yaml
task:
  num_labels: 47
  label_order: configs/labels.yaml
data:
  cohort: sequence_eligible  # 57,388；不要意外变成54,088
  split: split_v1
encoder:
  name: facebook/esm2_t33_650M_UR50D
  revision: 08e4846e537177426273712802403f7ba8261b6c
  output: last_hidden_state
  hidden_size: 1280
  max_length: 256
  frozen: true
model:
  kernels: [3, 5, 7]
  channels_per_scale: 128
  projection_size: 1280
  local_scale_init: -2.0
  layer_norm_affine: false
  layer_norm_eps: 0.00001
  dropout: 0.0
loss:
  name: s4_asl
  gamma_pos: 0
  gamma_neg: 4
  clip: 0.05
  eps: 0.00000001
  detach_focusing: true
  reduction: mean
train:
  epochs: 300
  seeds: [0, 1, 2, 3, 4]
  batch_size: 256
  gradient_accumulation: 1
  precision: fp32
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0001
  betas: [0.9, 0.999]
  eps: 0.00000001
  scheduler: null
  gradient_clipping: null
  early_stopping: false
evaluation:
  checkpoint_order: [validation_macro_ap, validation_micro_ap, earlier_epoch]
  ap_ties: sample_index_stable
  no_positive_ap: exclude_from_macro
  macro_f1_zero_denominator: 1  # 原实现兼容；未来改规则需分开报告
  threshold_policy: global_jaccard
  threshold_grid: [0.05, 0.95, 0.01]
```

这段是**Lite建议配置**，不是旧仓库已经存在的配置文件，也不是本轮新训练验证出的最优设置。

### 12.3 简洁训练/验证/测试流程

1. 一次准备固定split与47-label mapping；复用已有特征或按同一encoder语义生成split-specific缓存。用shape、ID、offset、列序的普通断言保证正确性。
2. Train脚本只打开Train/Validation数据。每epoch按原seed+epoch顺序更新，记录Train ASL与Validation Macro/Micro；仅保存best checkpoint及history。
3. calibrate脚本载入该checkpoint，只使用Validation概率拟合默认global阈值；有需要时运行明确命名的label-wise/S7 policy。
4. Test脚本独立执行，只应用已确定的checkpoint与阈值。Test不选架构/seed/epoch/阈值；不用旧Test已观察成绩作为“全新独立”结论。
5. predict脚本只需要sequence和对应feature接口，输出47个概率与二态labels；没有标签输入，没有dummy PDB/HELM，不强制top-k。

run目录例如 `runs/s4_sequence/seed_0/`；已有run不能静默覆盖，重跑使用新的明确run name。这是实验产物管理的简单边界，不需要attempt状态机或审批工具。

### 12.4 Git与复现记录

Git跟踪源码、一个配置、labels、小型结果汇总和测试；不跟踪数据、特征、checkpoint、预测矩阵、环境凭据或pytest scratch。配置修改通过Git历史表示，不维护无数v1/v2副本。

每run一个metadata JSON足够：Git commit、配置副本/实际值、seed、cohort/split名称与counts、encoder revision、label order、Python/Torch/CUDA/device、best epoch、threshold、主要指标。可以记录必要的数据版本信息，但不建立逐源码SHA审批依赖。

Windows开发→GitHub固定commit→Ubuntu checkout/worktree的标准流程可复用；无需复制SCP安装器或几十个源码SHA清单。原研究checkpoint仍作为只读外部资产。

### 12.5 最小测试与复现验收

只保留能发现科学错误的测试：

- special-token/zero-padding/masked-pooling；global/residue/target ID join与47列序。
- exact模型shape/参数数、frozen encoder没有梯度，local/head有梯度。
- ASL手算小例和focusing detach；AP ties、无正例、Macro-F1 empty-denominator约定。
- 保存/恢复state_dict后eval logits一致；Val-only selector与threshold；Train流程不打开Test。

先用合成小batch验证移植，然后在已有合法资源上复现固定配置。若有原S4 checkpoint与相同features，优先做相同batch计算一致性，而不是重新做S1/S3/S8负向消融。只有输入/evaluator相同才拿旧数值作复现目标；CPU/GPU backend变化要诚实记录数值边界，不能用新结果倒推容差。

保留B0作为很小的sanity/performance reference；不必重训五seed S1、S2-GC、S3、S8-G、D2、D3-A来再次证明已知结论。未来创新必须是在成熟base复现之后独立改变一个变量，而不是首版“顺便”堆模块。

## 13. 尚待确认的问题与潜在风险

1. **综合winner没有唯一客观定义。** S4对原Primary最强，B0对Micro与seed稳定性有优势，S2对global-threshold Macro-F1有优势，S5/S7处理另一决策目标。本文推荐是Macro优先的工程取舍，不追认原production selection。
2. **S4历史权重可用性。** 五个服务器checkpoint路径已明确；本轮没有下载或加载binary。直接迁移权重需要可访问这些文件，不能由轻量JSON假装权重已经存在。
3. **S2→B0无损派生尚未全量证明。** 同源encoder与抽样allclose不是逐样本无损证明；首版保持独立B0 global来源。
4. **AP tie与F1 empty-label规则不标准。** 改为常见库默认值会改变历史结果。先做精确兼容，再明确评价协议的后续版本，不能悄悄“修正”后混比较。
5. **特征precision影响。** FP16提取、换tokenizer、换layer输出、缩短max_length、换encoder都会改变成熟输入；没有既有证据可保证性能。
6. **validation复用偏倚。** S2–D3反复利用同Validation设计与选模；早期Test也已观察。Lite复现是复现项目内部证据，独立泛化主张仍需预先设计的新外部评估。
7. **长尾与不完整标注。** Macro会放大极少正例标签影响；Train support固定分箱`≤10/11–50/51–200/>200`对应2/6/14/25 labels。support只能从Train计算；0不是完备阴性证明。
8. **可解释性有限。** CNN activation不等于生物学motif，geometry function query不等于已证明的功能机制，θ/α变化不证明尺度错误。没有DIAG01结果时不创造label-level结论。
9. **旧文档有历史快照。** DEVELOPMENT_PLAN写S4为PEFT路线、S1未执行，或部分config还写PENDING，不能覆盖真实完成记录；实际S4是ASL loss-only阶段。D2/D3概念文档的“尚未实现/未执行”也已被后续记录更新。保留历史而依据实现与较新可核实结果判断。
10. **MFP/group schema未完成。** 不把47二态输出擅自升级成独立功能关系判定；首版不实现hierarchy/mfp-independence模块。

**可直接重建的成熟资产已经明确：S2计算图、S4-ASL、固定数据/标签映射和官方评价路径。未决的融合机制、阈值多目标取舍及外部泛化，不妨碍先完成一个干净的sequence-only Lite复现。**
