# 同类论文模型在 MFP-Nova 数据集上的重训练可行性分析

**2026-11-04，五模型原数据首轮复现已完成，当前研究目标转为 TheraPep 三任务重训练。** 首节给出最新数据核对和适配方案。后面的 V100 章节保留首轮结果与命令，11月2日 TheraPep 分析和编号1–7节为历史判断，涉及可用资产、风险和实施顺序时以最新研究为准。

## TheraPep48 已运行结果与下一步（2026-11-04）

依据本轮V100粘贴日志，以下均为seed42、Test 5566条、48功能的统一指标。iMFP-LG只有1轮试跑，不能作为100轮正式结果。不同方法沿用各自训练和决策设置。

| 模型 | 训练阶段 | Micro-F1 | Macro-F1 | Micro-AP | Macro-AP | Jaccard | 完全匹配 | Hamming loss |
|---|---|---|---|---|---|---|---|---|
| METFAN | 347轮早停，恢复最佳297轮 | .420454 | .475387 | .826793 | .457008 | .255248 | .000000 | .106776 |
| iMFP-LG | **1轮试跑** | .677273 | .158323 | .753688 | .295072 | .541721 | .328063 | .030296 |
| AMCL | 固定200轮 | .580437 | .215667 | .656806 | .270277 | .526802 | .346389 | .033522 |

METFAN对Antileishmania、Antimalarial、Antitrypanosomic、Blood-brain_barrier_penetrating四列的5566条Test全部预测阳性；实际阳性分别5、6、4、9条。四列贡献22240个假阳性，占全部25193个假阳性的88.28%。平均每条预测6.39个标签，实际为2.46个。需要先读出实际阈值和Val分数，不能将问题简单归因于模型没有学到信号。原生Macro-F1 .496221把Test完全为空且预测也为空的Angiogenic计为1，统一值.475387计为0，差1/48。

iMFP-LG仍有28个Test有阳性类别未预测任何阳性，AMCL正式200轮有17个。AMCL平均预测1.38个标签，Micro-Precision .808325、Recall .452785，主要表现为漏检。Test Angiogenic无正例，Macro-AP分母47，Macro-F1仍用全48类。AMCL的1轮结果仅验证流程，正式结果来自seed42_epochs200。

### V100：先诊断METFAN，同时补齐iMFP-LG正式训练

本次新增 `metfan_therapep.py diagnose`。从原最佳checkpoint导出的Val/Test分数进行CPU分析，不重新训练或生成语言模型特征，保留原comparison.json、NPZ、权重。输出独立的threshold_analysis.json，包括原Val粗网格、预先固定0.5、Val精确F1三种决策的完整指标、逐类支持度、阈值和预测阳性数。精确F1仅在Val不同分数处搜索，相同最优F1取较高阈值；单类别Val列仍固定0.5。先拟合新阈值再读取Test；不根据Test挑策略，也不重新选择checkpoint。这是已有checkpoint的补充决策实验，不能称为重新训练后的改进模型。拟合后的Val成绩为拟合内结果。

在已完成的TheraPep48任务上使用以下命令。当前三个作业的工作区不更新，用单独Git工作区读取新脚本。

```bash
export BASE="$HOME/wuzhencheng/projects/MFP-Nova"
export NEXT_PROJECT="$HOME/wuzhencheng/projects/MFP-Nova-paper-next"
git -C "$BASE" fetch origin main
git -C "$BASE" worktree add --detach "$NEXT_PROJECT" origin/main
export NEXT_KIT="$NEXT_PROJECT/scripts/paper_repro"
export PAPER_ROOT="$HOME/paper-repro"
export MODERN_PYTHON="$HOME/.conda/envs/repro-modern/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export metfan_out="$PAPER_ROOT/therapep_runs/metfan_therapep48_20261104_011944"

CUDA_VISIBLE_DEVICES="" "$MODERN_PYTHON" "$NEXT_KIT/metfan_therapep.py" diagnose \
  --output "$metfan_out" --seed 42 >"$metfan_out/threshold_analysis.log" 2>&1
cat "$metfan_out/threshold_analysis.log"
```

若NEXT_PROJECT已经由上述命令创建，再次使用时复用该目录，不重复worktree add。输出会给出四列的实际原生/精确阈值；现有日志尚未包含这些阈值，不能预先断言一定为0。精确搜索也可能合法地选择整列阳性，效果要以真实输出为准。

iMFP-LG正式训练复用之前准备的数据和已应用的作者仓库补丁，不重复prepare或apply。先执行下面的进程/GPU查看，确认GPU0仍分配给本实验，且没有已启动的100轮任务；若正式任务已在运行或完成，继续查看其train.log即可。

```bash
pgrep -af '[i]mfp_therapep.py.*train'
nvidia-smi
```

确认尚未启动正式任务后运行：

```bash
export IMFP_KIT="$HOME/wuzhencheng/projects/MFP-Nova-imfp/scripts/paper_repro"
export IMFP_PYTHON="$HOME/.conda/envs/repro-imfp/bin/python"
export imfp_out="$HOME/paper-repro/therapep_runs/imfp_therapep48_20261104_015157"
export GPU_ID=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup "$IMFP_PYTHON" "$IMFP_KIT/imfp_therapep.py" train \
  --output "$imfp_out" --seed 42 --epochs 100 --batch-size 32 \
  >"$imfp_out/train.log" 2>&1 < /dev/null &
echo "PID=$! LOG=$imfp_out/train.log"
tail -f "$imfp_out/train.log"
```

这从相同通用TAPE权重重新初始化任务模型，保留1轮试跑，输出training/seed42_epochs100。按试跑1398秒/轮估算，100轮约39小时，加最终评估，实际受硬件负载影响。固定100轮不因本次Test表现改变。

### 准备TPpred-LE和SA-MTP的新样本特征

新增structure_inputs.py只导出输入，还没有实现这两模型的TheraPep训练。一次生成供两方法使用的原始全长查询、各split的seqs.fasta/labels.csv及row_id/ID/feature_id映射。仅特征查询按完全相同序列合并，训练和评价仍保留全部行及原顺序。TheraPep48共55651条不同序列，Train/Val/Test仍为44524/5565/5566，训练中的74残基序列保留；模型最终按各自原生规则处理长度。不同数据任务的feature_id编号不能按位置互换，需要通过序列对应。

```bash
export structure_out="$PAPER_ROOT/therapep_runs/structure_therapep48_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="" "$MODERN_PYTHON" "$NEXT_KIT/structure_inputs.py" \
  --task therapep48 --data-root "$BASE/data/therapep48" --output "$structure_out"

# 只检查当前环境是否能找到程序，不安装、不下载数据库、不启动特征计算。
command -v psiblast || true
command -v blastdbcmd || true
command -v runpsipred || true
command -v runpsipred_single || true
printf 'BLASTDB=%s\n' "${BLASTDB:-}"
df -h "$PAPER_ROOT"
```

这里列出的PSIPRED命令是资源定位线索，不表示作者要求使用single模式。下一步需落实可用PSI-BLAST数据库、PSIPRED版本/模型与匹配的SS2生成流程，再接训练入口。TPpred-LE原搜索是3轮、evalue .01、默认6线程；数据库不足不能把所有查询静默替换为BLOSUM。SA-MTP必须为新序列生成H/E/C概率；不能把旧MTP15的SS2套到新编号或把缺结构的全连接图结果当完整方法。查询导出不占用训练GPU。

## TheraPep 三任务重训练研究（2026-11-04）

五模型已完成原数据首轮训练和 Test。下一步沿用 TheraPep-AI 发布划分，分别训练 `therapep49`、`therapep49_positive`、`therapep48`。各方法保留自己的预处理、特征、损失、训练阶段和原生决策，同时输出一致定义的评价指标。下面依据当前九份 CSV 和已复现的五个提交重新核查，覆盖此前仅做静态估计的判断。

### 数据和输出

| CLI任务 | 数据目录 | Train / Val / Test | 目标 |
|---|---|---|---|
| therapep49 | data/therapep_ai | 131,947 / 16,457 / 16,537 | 48功能 + is_therapeutic，含全零合成背景 |
| therapep49_positive | data/therapep49_positive | 44,524 / 5,565 / 5,566 | 相同正肽的49维目标，最后一维恒1 |
| therapep48 | data/therapep48 | 44,524 / 5,565 / 5,566 | 相同正肽的48功能目标，删去治疗标志 |

两个变体与完整数据的正肽逐行核对通过，序列、功能标签和顺序一致，两个变体的 row_id/ID 也一致。项目实际任务名使用上表小写标识。第48个功能 Tumor_homing 保留。

本次补充统计发现，正肽 Train 有4条重复记录，标签相同；混合 Train/Val/Test 分别有14/1/1条重复记录。混合 Train 与 Val、Train 与 Test 各有4条相同序列，均为全零背景，标签无冲突。两个正肽变体的三个划分之间没有完全相同序列。保留作者原始行和划分，不按序列建字典覆盖记录；编码缓存可按序列复用，训练样本仍按行读取。完全相同序列的统计不代表同源隔离结论。

Train 中48个功能都有正例，最少为 AntiTubercular 9条、Angiogenic 11条。Val 的这两类没有正例，Test 的 Angiogenic 没有正例。原 TPpred-LE/SA-MTP 指标函数逐类无条件调用 roc_auc_score，在当前 sklearn 环境会因单类别真值而中断，需要对不可定义的 AUROC 返回空值并记录有效类别数。全48功能的 F1 按固定标签集合计算；Macro-AP 应明确无正例类的处理及分母，具体指标复用方式见下一段。

Val 缺正例的类别和恒正治疗标志不搜索类别阈值，预先用0.5。其余可搜索的阈值只在 Val 确定。SA-MTP 的原15类 F1/MCC/fixed 表不直接按位置迁移，建议在全部可识别功能类上采用作者通用 F1 搜索作为新任务起点，写明这项迁移设置。

具体接入时复用项目 `evaluate_metrics`，Macro-AP 按有正例类别平均，并报告 `ap_labels_with_positives` 和逐类空值。该字段与原生PR曲线梯形面积AUPR区分。Macro-F1继续保留完整标签集合。

### 五模型的具体改造

| 模型 | 可以直接保留 | 必须修改 | 当前可行性 |
|---|---|---|---|
| METFAN | TextCNN、两种语言模型、池化、特征优化、FDL、Val Macro-F1选模及现有Val阈值补丁 | 按行读取CSV；输出48/49；同一有效行索引生成两套特征；单类别阈值处理；导出身份和连续分数 | 最适合作为第一个三任务训练实现 |
| iMFP-LG | 已取得的TAPE、微调、FGM、标签GAT、BCE、原固定日程与0.5阈值 | NPY序列字典改为CSV行记录；config.num_labels；确认优化器是否训练FC/CLSFC；测量49节点图显存 | 三任务均可，已具备正确预训练资产 |
| AMCL | TextCNN、FDL/DBL/对比损失、增强操作、原训练日程与阈值规则 | 所有21维常量；当前Train增强和类频率；功能列与治疗标志分开处理；恒正类DBL数值；阈值拟合改为Val | 48任务优先，49任务需明确损失与增强适配 |
| TPpred-LE | one-hot/PSSM、标签解码、两个训练算法、分类器重采样 | 15维label_input与Embedding；数据读入；新PSSM映射；最佳轮保存；缺失类别评价 | 可行，主要准备成本是PSSM |
| SA-MTP | 冻结ESM2、PSIPRED SS2、动态图、GAT、FiLM、原早停 | 15维token与Embedding；新SS2和序列ID；最佳轮保存；新任务阈值和单类别评价 | 可行，主要准备成本是SS2及残基缓存 |

**METFAN。** 本地用每个任务4条真实 Train 的序列/标签，加随机1280/1024维池化输入，检查作者完整融合模型48/49维输出及 BCE 反向，均为有限值。该检查只验证接口和梯度，不是语言模型特征验证、原FDL完整训练或真实性能。TheraPep Train 长度超过50的行分别为混合版3条、两个正肽版各1条；Val/Test 无超长行。保留作者过滤规则时，实际 Train 为131,944/44,523/44,523条，对全部模态使用同一索引。

**iMFP-LG。** BCE 接受全零行和恒正列；TAPE必须继续微调，不能改成冻结ESM替代。原代码将批内标签节点拼成一个大图，32批量时从21类改到49类，注意力矩阵元素数由451,584增到2,458,624，约5.44倍；这只描述图矩阵，不能据此估算整模型显存倍数。先在16GB V100 实测batch32，确有OOM再降低。FC/CLSFC优化器遗漏是原数据复现时已确认的问题，建议新任务加入优化器并清楚命名为修正设置；原结果仍保留。

**AMCL。** 新增强只来源于相应 Train。重读原DA源码可知，其 `positions_to_check=range(21)` 覆盖全部21功能，实际对所有单标签Train行增强，没有在该函数中做尾类筛选。因此新入口按前48个功能判断单标签，保留N=3和四条新增序列的原操作；49维分类目标仍含治疗标志。功能相似度也只用48个功能。DBL相关的头尾频率和全部类频率从当前增强Train重新计算，21位硬编码需去除。

本次提取作者 DBL 类，在小型合成标签例上检查，CPU测试将硬编码CUDA张量映射到CPU。混合49的全零背景例得到有限loss和梯度；positive49的恒正列得到NaN；48功能例正常。因此明确需要处理的是恒正类初始偏置 log(0)，不能仅凭 repeat_rate=0 就断言混合版DBL会失败。建议功能列继续采用DBL，恒正治疗标志的DBL子项改为普通BCE，并保持原组合权重、记录输出归一化方式；正式训练前再以真实小批检验原FDL/DBL/对比组合。

**TPpred-LE。** 本次直接提取并运行作者 square 采样器的小型标签例。混合49全零行得到非零且有限的采样权重，原实现已经考虑每一目标类的正负状态，无需主动排除背景。positive49恒正列最终权重同样有限，负类分配的空除法会产生警告；可只对实际存在的状态计算，避免无意义除零，保持原采样分布。此前“全零行可能被赋零采样权重”的泛化怀疑不适用于已核查的这个 Sampler。PSSM沿用作者搜索和无匹配时BLOSUM回退，统计实际覆盖率，不能将全体PSSM人为换成BLOSUM。

**SA-MTP。** 当前10,237份SS2和PSSM来自旧MTP15样本，不能按编号套给TheraPep。对新样本确认可生成作者所需H/E/C概率；正肽与合成背景用同一生成流程。缺SS2回退全连接图的运行应标为结构缺失配置。最佳轮须即时保存或复制参数，随后在恢复的权重上拟合Val阈值。该修正同样用于TPpred-LE。新模型均随机初始化任务相关头，不加载旧MTP/MFTP的监督分类权重。

### 可复用资产与资源

V100 已有 ESM2-650M、ProtT5和TAPE 权重，以及三个可用Conda环境，可直接继续使用。旧数据的微调权重、SS2、PSSM和预计算特征只有在序列、编码器、残基顺序、长度处理及数值设置一致时才能逐条复用。MFP-Nova原主任务的特征不能按行号直接套入TheraPep。

建议先对完整混合队列生成通用特征，再按row_id为两个正肽变体取子集。训练任务各自独立，PSSM/SS2和冻结表征可复用。METFAN 的 ESM2/ProtT5 均值特征与 SA-MTP 的残基表征应分别按其原处理生成；前者本轮用FP16提取，后者原实现为FP32，直接互换会改变数值设置。

按50残基、1280维、FP32估算，SA-MTP 单独混合Train的padding表征约31.46 GiB，三个划分合计约39.32 GiB；实际内存还含原始缓存、图和模型。三个正肽划分合计约13.27 GiB。迁移前需改为按批读取/填充或合适的磁盘数组，避免一次同时持有多份padding拷贝。METFAN两支均值特征合计2304维，混合三划分约1.42 GiB，资源门槛明显较低。以上是张量体积估算，没有宣称实测显存和工期。

### 训练、比较和实施顺序

每个模型在三个任务分别训练，共15个首轮训练配置。seed先用42作适配试验；固定方法设置后再做必要多seed。TPpred-LE的一个配置包含原两阶段，不能把15个配置直接理解为15个单阶段GPU作业。各方法沿用自己的batch、优化器、损失、训练阶段和选模原则。

每个任务先报告完整Test，再将三次训练固定下来的权重、阈值投到同一批5,566条正肽的前48个功能上。主表给Micro/Macro-F1、Micro/Macro-AP、样本Jaccard、完全匹配率、Hamming loss。需要时给固定0.5的补充结果，用已有分数计算，沿用原生决策的主结果也应保留。

混合49另外报告治疗标志和合成背景功能误报率。positive49的治疗标志恒1，不能据整体49维分数判断功能提高。完全背景正确预测的Jaccard按项目定义计0，混合全Test的样本Jaccard即使完美预测也只有约33.66%；因此不能拿它与正肽任务的Jaccard直接比较。

推荐实施顺序如下。

1. **METFAN先完成therapep48。** 增加一个普通CSV入口，保留数据行身份、两套特征和当前Val阈值修正。真实小批检验原FDL和参数更新，短训练后再跑一个完整seed。
2. **METFAN扩展positive49和mixed49，同时接iMFP-LG。** 这两种方法无需先建设PSSM/SS2流水线；iMFP-LG先确认优化器修正和49节点显存。
3. **AMCL先48后49。** 固定尾类与增强定义，先验证恒正类DBL适配，再跑两个49任务。
4. **准备TPpred-LE的PSSM与SA-MTP的SS2。** 落实生成覆盖、CPU/存储条件后再完成两种结构/进化方法的三个任务。避免先占用GPU等待未就绪的输入。
5. **与项目模型对照。** 读取同任务、同split的TheraPep-AI和HelixSet结果，核对模型版本和选模设置。此前原数据五模型分数不直接放入TheraPep表。

### 下一步执行入口

已增加 `scripts/paper_repro/metfan_therapep.py`，支持三个任务的CPU数据准备、特征生成、训练三个动作。第一项完整实验先运行 therapep48 / seed42。原 `run_v100.sh` 继续用于原论文数据，TheraPep改用新入口。

本地已检查三任务的导出条数和行身份，验证旧Val阈值补丁与新增TheraPep补丁可依次应用。对每个任务取4条真实序列/标签，用随机池化特征运行作者完整融合模型、原FDL、反向和一次参数更新，结果有限。另用小型融合测试替身、随机池化输入和真实CSV小批，跑通原生FDL/Adam一轮训练到最佳权重恢复、固定阈值、身份对齐NPZ及指标JSON的CLI流程。未在本机生成真实语言模型特征或进行完整训练，这些检查不产生模型性能结果。

首次在V100的MFP-Nova目录同步脚本。若此前仅按本会话命令将服务器 `metfan_features.py` 的单行设置改成 `use_fast=False`，这项修复已包含在新提交中，可先 `git restore -- scripts/paper_repro/metfan_features.py` 再拉取；如还做了其他本地修改，先检查 `git diff` 保留所需修改。不要在运行中的源码目录更新。

```bash
cd "$HOME/wuzhencheng/projects/MFP-Nova"
git pull --ff-only origin main
conda activate repro-modern
export KIT="$PWD/scripts/paper_repro"
export PAPER_ROOT="$HOME/paper-repro"

# 原始TheraPep CSV或变体尚未放在服务器时，使用项目已有命令准备。
python -m mfp_nova.cli download-therapep
python -m mfp_nova.cli prepare-therapep-variants

# METFAN原仓库已经应用过metfan_val_thresholds.patch，再应用这份增量补丁一次。
git -C "$PAPER_ROOT/METFAN" apply --ignore-space-change "$KIT/metfan_therapep.patch"

# out为本轮新目录，准备动作自行创建。后续两个动作继续使用同一个out。
export out="$PAPER_ROOT/therapep_runs/metfan_therapep48_$(date +%Y%m%d_%H%M%S)"
python "$KIT/metfan_therapep.py" prepare --task therapep48 --output "$out"
```

预期显示Train源数据44,524条、保留44,523条，Val 5,565条、Test 5,566条。输出保留row_id/ID的映射和48类顺序；过滤仅涉及Train的一条超长序列，原始CSV不修改。

沿用分配的GPU，此处按用户此前明确选择的物理GPU1执行。提取和训练顺序在同一个后台作业内完成，特征失败会停止作业。

```bash
export METFAN_ESM2="$PAPER_ROOT/pretrained/esm2_t33_650M_UR50D"
export METFAN_PROTT5="$PAPER_ROOT/pretrained/prot_t5_xl_half_uniref50-enc"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export METFAN_PYTHON="$CONDA_PREFIX/bin/python"
"$METFAN_PYTHON" -c 'import sys, torch; print(sys.executable, torch.__version__)'
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup bash -c '
  "$1" "$2" features --output "$3" &&
  "$1" "$2" train --output "$3" --seed 42
' _ "$METFAN_PYTHON" "$KIT/metfan_therapep.py" "$out" >"$out/run.log" 2>&1 < /dev/null &
echo "PID=$! LOG=$out/run.log"
tail -f "$out/run.log"
```

结果位于 `$out/training/seed42/0/textcnn_esm2_mean_prot_t5_mean_optimized/`，包含最佳checkpoint、原生结果、带row_id/ID与类名的Val/Test连续分数NPZ，以及 `comparison.json`。后者提供完整Test、正肽48功能视图、Macro/Micro-F1/AP、不可定义指标原因和最佳epoch。新入口按seed分别保存训练目录，支持复用同一套特征开展后续seed。

首版各任务单独提取特征；跨任务按row_id复用混合队列表征仍属后续优化。48任务完整训练和Test验证后，两个49任务只需将prepare的 `--task` 改为对应名称，并使用各自的新out，再执行features/train。单类别Val的阈值固定为0.5，不会使用Test标签重新拟合阈值。

### iMFP-LG 的 TheraPep 准备

已增加 `imfp_therapep.py` 和 `imfp_therapep.patch`。CSV分支逐行读取，保留重复记录；NPY原输入分支保留。三个任务的类别数由本项目标签定义提供。TAPE、FGM、BCE和标签GAT保持原实现，最终权重按固定训练日程选取，决策沿用严格 `probability > 0.5`。

新入口显式开启 `train_task_heads`，将原来遗漏的FC/CLSFC参数纳入任务组，与GAT同用5e-4学习率，TAPE保持5e-5。结果记录为 `method_variant=task_heads_trained`。原NPY复现入口默认不启用这一参数，保持此前的优化器参数范围。补丁同时按原始批次顺序累积预测，导出Val/Test连续分数和身份，以便检查标签对齐；图拼接的已弃用 `np.row_stack` 使用等价 `np.vstack`。

METFAN 运行期间，为并行准备另一个模型，使用独立Git工作区读取新增入口。下面仅fetch及新增工作区，当前METFAN项目检出的源码和HEAD不更新；每个模型的源码提交记录因此与实际运行版本相符。该Git工作区读取原项目的只读CSV，不复制数据。

```bash
export BASE="$HOME/wuzhencheng/projects/MFP-Nova"
export IMFP_PROJECT="$HOME/wuzhencheng/projects/MFP-Nova-imfp"
git -C "$BASE" fetch origin main
git -C "$BASE" worktree add --detach "$IMFP_PROJECT" origin/main
export IMFP_KIT="$IMFP_PROJECT/scripts/paper_repro"
export PAPER_ROOT="$HOME/paper-repro"

conda activate repro-imfp
export IMFP_PYTHON="$CONDA_PREFIX/bin/python"
"$IMFP_PYTHON" -c 'import sys, torch; from transformers import AdamW; print(sys.executable, torch.__version__)'

# 仅对已经完成原数据复现的iMFP-LG作者仓库应用一次。
git -C "$PAPER_ROOT/iMFP-LG" apply --ignore-space-change "$IMFP_KIT/imfp_therapep.patch"
export imfp_out="$PAPER_ROOT/therapep_runs/imfp_therapep48_$(date +%Y%m%d_%H%M%S)"
"$IMFP_PYTHON" "$IMFP_KIT/imfp_therapep.py" prepare \
  --task therapep48 --data-root "$BASE/data/therapep48" --output "$imfp_out"
```

预期保留Train 44,524、Val 5,565、Test 5,566条。该方法原生按批padding，74残基训练序列仍在TAPE位置长度范围内，故不像METFAN一样过滤该条。模型直接微调TAPE，不额外提取冻结特征。

如GPU0已分配且空闲，可与GPU1上的METFAN分别运行。先做一轮连通性和耗时检查；这是短训练结果，不记为正式100轮性能。必须使用环境Python的绝对路径，避免新Bash解释器变化。

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup "$IMFP_PYTHON" "$IMFP_KIT/imfp_therapep.py" train \
  --output "$imfp_out" --seed 42 --epochs 1 --batch-size 32 \
  >"$imfp_out/short_train.log" 2>&1 < /dev/null &
echo "PID=$! LOG=$imfp_out/short_train.log"
tail -f "$imfp_out/short_train.log"
```

确认这一轮能完成后，用相同out和 `--epochs 100` 提交正式实验，日志改为 `train.log`。两项分别写入 `training/seed42_epochs1/`、`training/seed42_epochs100/`，均随机初始化任务模型，只加载相同的通用TAPE权重。数据预备结果复用，短训练权重不作为正式训练初始化。Val/Test分数、原始ID、固定阈值和 `comparison.json` 均写入各自训练目录。

本地用单层、768维的随机BERT初始化替身和真实CSV小批，覆盖48功能、49恒正标志及混合全零背景，检查原GAT/BCE/FGM的一轮执行、FC/CLSFC实际更新、重复行及输出对齐。本地Transformers较新，测试时仅在临时副本将已移除的HF AdamW换为Torch AdamW并匹配基础超参数；提交的代码仍使用V100已验证的Transformers4.24/HF AdamW。原始TAPE权重和16GB V100的完整批量运行仍需服务器短训练验证。

### AMCL 的 TheraPep 准备

新增 `amcl_therapep.py` 和 `amcl_therapep.patch`。prepare动作只使用当前任务Train，调用作者增强函数，按前48功能判断单标签并以N=3生成反转、相似残基替换和反向翻译变体。Val和Test保持原始序列及行身份。训练继续使用原TextCNN、FDL/DBL权重0.99/0.01、0.3倍对比损失、flooding、Adam及原余弦调度。此入口的阈值改在Val拟合，预测仍保留作者的 `sigmoid(10*(logit-t)) > t` 决策，AP从普通sigmoid(logit)连续分数计算。

对positive49，DBL的功能48列正常计算，恒正治疗标志的该子项改用BCE，再按49个输出归一化；FDL仍覆盖全部49列。对mixed49保留全部49列的原DBL。两者对比损失均只使用功能48列，避免治疗标志使任意正肽之间产生额外功能重叠。Val中单一真值的类别不搜索阈值，固定t=0.5，此t仍按作者上述决策解释。

真实therapep48的CPU增强检查得到Train 44,524条，其中17,310条为单功能标签行，增强后为113,764条，即44,524 + 4×17,310。原始Train有一条74残基序列，保留；Train/Val/Test分别padding到74/50/50。模型在这两种长度下均保持原池化输出尺寸。小批测试采用至少50的padding，符合原TextCNN固定池化维度要求。

METFAN及iMFP-LG仍在运行时，另建AMCL的Git工作区。操作仅fetch和新增工作区，不更新正在运行的两个项目检出的HEAD。数据、权重不复制。

```bash
export BASE="$HOME/wuzhencheng/projects/MFP-Nova"
export AMCL_PROJECT="$HOME/wuzhencheng/projects/MFP-Nova-amcl"
git -C "$BASE" fetch origin main
git -C "$BASE" worktree add --detach "$AMCL_PROJECT" origin/main
export AMCL_KIT="$AMCL_PROJECT/scripts/paper_repro"
export PAPER_ROOT="$HOME/paper-repro"
conda activate repro-modern
export AMCL_PYTHON="$HOME/.conda/envs/repro-modern/bin/python"

git -C "$PAPER_ROOT/AMCL" apply --ignore-space-change "$AMCL_KIT/amcl_therapep.patch"
export amcl_out="$PAPER_ROOT/therapep_runs/amcl_therapep48_$(date +%Y%m%d_%H%M%S)"
"$AMCL_PYTHON" "$AMCL_KIT/amcl_therapep.py" prepare \
  --task therapep48 --data-root "$BASE/data/therapep48" --output "$amcl_out" --seed 42
```

以上仅准备CPU增强，不占用训练GPU。两个现有GPU作业完成或有其他已分配空闲卡后，再选择该卡短训练。下面物理GPU0只作用户已确认可用时的示例。

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0 \
nohup "$AMCL_PYTHON" "$AMCL_KIT/amcl_therapep.py" train \
  --output "$amcl_out" --seed 42 --epochs 1 --batch-size 256 \
  >"$amcl_out/short_train.log" 2>&1 < /dev/null &
echo "PID=$! LOG=$amcl_out/short_train.log"
tail -f "$amcl_out/short_train.log"
```

短训练正常后，正式设置为 `--epochs 200`，日志用train.log。两个实验分别写入 `training/seed42_epochs1/` 和 `training/seed42_epochs200/`；模型均从随机初始化开始，复用同一份Train增强文件。最终保存model.pth、Val阈值、带logits/普通概率/原生二值决策/row_id/ID的Val和Test NPZ，以及完整Test和正肽48功能视图的comparison.json。源码只在DA脚本直接运行时执行其旧默认流程，导入增强函数不再读取作者机器的TP_test路径。原始21类训练入口默认仍读其原增强文件。

本地已对三个任务用合成覆盖标签的小批跑通原TextCNN、原组合损失、对比学习、flooding、Adam参数更新、Val阈值和按行导出，包含恒正49列及全零背景。另用真实完整therapep48验证了增强条数。完整V100训练及真实Test性能尚待执行。准备positive49/mixed49时使用独立out，指定相应任务及其真实CSV根目录，复用同一套代码。

## V100 上复现论文原始实验

### 首轮实际运行结果（2026-11-03）

五模型的逐项效果分析、论文对照与源码差异集中在 [原数据复现效果分析](D:/Data/Code/MFP-Nova/docs/五模型-V100-原数据复现效果分析.md)。

以下来自用户提供的 V100 训练日志和 Test CSV，均为单次运行；此处未重新执行远端推理。五个选定方法已各完成一次原始主基准训练和独立测试。AMCL 另完成作者发布权重复测，五指标与论文表 1 在日志三位小数精度上一致。

MTP15 的 Train/Val/Test 为 8,190/1,023/1,024。下表为百分数；横线表示本次提供的输出没有该项。

| 方法/结果 | 样本 Jaccard | Micro-F1 | Macro-F1 | 完全匹配率 | Hamming loss ↓ |
|---|---:|---:|---:|---:|---:|
| TPpred-LE 发布权重 | 55.39 | 60.95 | 43.79 | 43.75 | 7.50 |
| TPpred-LE seed46 阶段A | 53.68 | 59.45 | 42.81 | 42.29 | 7.61 |
| TPpred-LE seed46 阶段B | 52.32 | 58.08 | 43.18 | 36.33 | 9.15 |
| SA-MTP seed50 混合阈值 | — | 58.90 | 43.02 | 38.18 | 8.89 |
| METFAN seed42 完整融合 | 59.69 | — | 48.60 | 46.73 | 8.60 |

TPpred-LE 结果目录为 `~/paper-repro/results/TPpred-LE.O4kGdp/`。SA-MTP 为 `SA-MTP.kgEohV/`，Epoch 13 的 Val loss 最低为 0.200873，第28轮早停；公开实现的引用保存实际留下第28轮参数，应标为原代码结果。TPpred-LE 同样保留了原引用保存行为。METFAN 为 `METFAN.aa5che/0/textcnn_esm2_mean_prot_t5_mean_optimized/`，第239轮早停，恢复第189轮保存的权重，最佳 Val Macro-F1 为0.5640。METFAN 使用本项目重建的 ESM2/ProtT5 特征和 Val 阈值补丁，精确 checkpoint 和池化设置见后文。

MFTP21 结果同样为百分数。iMFP-LG 原 NPY Test 读取1,975条；AMCL 原编码器过滤后为1,969条。保留原生处理时须注明这6条覆盖差异。

| 方法/结果 | 样本 Precision | 样本 Recall | 样本 Jaccard | 完全匹配率 | Hamming loss ↓ |
|---|---:|---:|---:|---:|---:|
| AMCL 发布权重复测 | 76.2 | 71.9 | 70.7 | 65.2 | 3.2 |
| AMCL 重建训练入口，200轮 | 75.0 | 70.7 | 69.5 | 63.7 | 3.3 |
| iMFP-LG 编号0 / seed0，100轮 | 70.0 | 71.2 | 65.8 | 57.0 | 3.8 |

AMCL 本次目录为 `~/paper-repro/results/AMCL.2xm3sd/`，权重 `amcl.pth`。iMFP-LG 将最终权重保存为对应 `iMFP-LG.*/model0.bin`，日志在第100轮之后打印的“on val”五项实际调用 `TestModel(test_iter, ...)`，属于独立 Test。原入口未输出 Macro/Micro-F1，不从样本均值 Precision/Recall 推算这两项。原来的 `MFTP_alldata` 全数据任务权重未用于此测试。当前结果支持原数据训练通路已跑通；稳定的重复实验及最佳轮保存修正结果尚待完成。

**本轮重点复现五个模型，每个先在作者主划分上完成一次训练与测试。** MTP15 选 TPpred-LE、SA-MTP、METFAN，共用 1,024 条原始测试记录；MFTP21 选 AMCL、iMFP-LG，共用 1,975 条原始测试记录。各自保留原生预处理，汇报实际进入评分的样本数，两个数据集分表。新增的 iMFP-LG 优先做 MFTP21 单个模型编号，MFBP5 留到后续；METFAN 排最后，先落实 ESM2/ProtT5 特征，再运行完整融合模型，用验证集确定阈值后单独报告独立测试成绩。

第一项选 **AMCL 发布权重复测**，第一项从头训练选 **TPpred-LE**。这样可以分别回答“能否得到论文的测试输出”和“能否从原始训练数据得到相近表现”。使用各仓库原格式、原划分、原特征、原训练日程及指标。本地已保存九个项目的数据文件，研究用的统一 CSV 可用于核对样本数，运行作者脚本直接读其原文件即可。[原始材料清单](../data/literature_benchmarks/README.md) · [基准与划分](../data/literature_consolidated/README.md)

| 项目 | 推荐的第一次实验 | 启动前需要解决的实际问题 |
|---|---|---|
| AMCL | MFTP21，原权重独立测试 | 仓库含权重、增强训练集和 Test。默认入口用增强 Train 拟合阈值，再评 Test。按源码允许字符在本地文件计数，增强 Train 34,764 行中进入编码 34,664 行，Test 1,975 行中进入 1,969 行。从头训练需把优化器和组合损失接到已有训练类。[源码](https://github.com/xialab-ahu/AMCL/blob/master/train_test.py) |
| TPpred-LE | MTP15，先复测一份发布权重，再跑一个 seed 的训练与分类器重训练 | Train/Val/Test 为 8,190/1,023/1,024。Git 已含五份权重，另下载作者 PSSM 并解到 `features/pssm/*.pssm`。索引中有序列但文件缺失会报错。作者的 Algorithm 1 与 Algorithm 2 可分别记录。[材料与 PSSM](https://bliulab.net/TPpred-LE/data/) · [训练](https://github.com/HongWuL/TPpred-LE/blob/main/train.sh) · [重训](https://github.com/HongWuL/TPpred-LE/blob/main/retrain.sh) |
| iMFP-LG | 原 MFTP21 的一个模型编号，随后再做 MFBP5 | 下载作者链接中的 TAPE，以 `pretrain_model/tapebert/` 供程序读取 config、vocab、权重。进入 `Task/` 运行；MFTP 默认 batch 32、100 epoch、循环 10 个编号。可在此目录用 `python -c 'from TaskForMultiPeptide_MFTP import TrainConfig, train; train(TrainConfig(0))'` 单独训练首个编号。发布的 `MFTP_alldata` 预测权重与独立测试重训分开。MFBP5 原划分有 69 条 Train/Test 同序列，保留作者划分并注明。[作者说明与下载入口](https://github.com/chen-bioinfo/iMFP-LG) |
| SA-MTP | 原 MTP15 主基准 | 完整仓库已有 10,237 份 SS2，按 split 放在 `texture/` 下。为原 ESM2-650M 准备权重，loader 可增量生成序列缓存。从根目录创建 `logs results trained_models` 后用 `python -u main.py -act t e -task original_mtp15`；默认动作 `v` 进入可视化。确认读取 SS2，缺失会退为全连接图。低同源 40% 独立划分尚未取得。[入口](https://github.com/LZW-TECH/SA-MTP/blob/main/main.py) |
| MFP-MFL | 清洗版 MFBP5，先跑一个特征分支检查训练 | 仓库已有原样本预计算特征，清洗版为 4,732/1,185。先修训练数据路径、补作者训练入口引用但缺失的 FGM。完整方法还需三个分支各十个成员和集成。集成脚本按 Test 标签找阈值，分清公开脚本输出与独立评估。[代码](https://github.com/zhouworks/Multifunctional-peptide-classification/tree/main/model_GAT) |
| MCMFPP | 原 MFTP21 上双分类器发布权重复测，随后分支训练及融合 | 取得 ESMC-300M 权重。源码启用 FlashAttention 且 CUDA 路径转 BF16，V100 上先改为普通注意力、FP32，对照发布结果后再跑 `train_slfe.py`、`train_cfec.py` 和融合。保留这项精度差异及作者过滤后的实际样本数。[上游仓库](https://github.com/zhao-project/MCMFPP) |
| METFAN | 原 MTP15，先落实完整特征组合 | Git 有三份序列划分，默认仅 TextCNN。按论文准备 ESM2、ProtT5 对应池化特征。公开评估在传入的 Test 上拟合阈值，此路径的输出单列说明；独立指标在 Val 拟合阈值后算 Test。[配置与源码](https://github.com/szlstart/METFAN) |
| TPpred-CMvL | 原 MTP15，优先取作者完整代码和数据包 | 已发布 PSSM 和部分权重，仍要定位 TAPE 语义特征、替换作者机器路径，核对第一阶段保存的权重是否被第二阶段加载。默认只跑第二阶段。本次未取得 Zenodo 压缩包。[仓库](https://github.com/Dominiczx/TPpred-CMvL) |
| PDeepPP | 选已公开的一项原生二分类任务，如 ACE | 本地已有 32 个原始任务。选定一个任务，先匹配该任务的表征及 Hugging Face 权重；从头训练前修复 loader 参数和训练标签/测试特征错配。按本任务的 ACC、AUC、MCC 等核对。[代码及权重说明](https://github.com/fondress/PDeepPP) |
| SeekTP | 暂缓，优先找原始 materials.zip | 项目此前记录材料链接失效。需要作者筛选的 8 类划分（6,388/1,592）、特征与 gat.pkl 才能用 TestGAT.py 复测；公开仓库未发现完整训练入口。[仓库](https://github.com/researchprotein/MFTP) |

### V100 操作命令（首次安装与单次训练）

**1．同步四个小文件。** 从本机 MFP-Nova 的 main 分支用 GitHub 同步 `scripts/paper_repro/` 和本文。当前新增文件在本地工作区，尚未推送。在项目目录中执行 `git add -- scripts/paper_repro docs/literature-model-comparison-feasibility.md`、`git commit --only -m "Add V100 original paper reproduction commands" -- scripts/paper_repro docs/literature-model-comparison-feasibility.md`、`git push origin main`。服务器用密码 SSH 登录；在未运行实验的 V100 项目检出中 `git pull --ff-only origin main`。以下第一行在 V100 的 MFP-Nova 项目根目录执行，路径变量在当前 Bash 会话有效。

```bash
export KIT="$PWD/scripts/paper_repro"
export PAPER_ROOT="$HOME/paper-repro"
mkdir -p "$PAPER_ROOT"
cd "$PAPER_ROOT"
git clone https://github.com/HongWuL/TPpred-LE.git && git -C TPpred-LE checkout 9913cc9
git clone https://github.com/xialab-ahu/AMCL.git && git -C AMCL checkout 627c637
git clone https://github.com/LZW-TECH/SA-MTP.git && git -C SA-MTP checkout 3e23883
git clone https://github.com/chen-bioinfo/iMFP-LG.git && git -C iMFP-LG checkout 911da39
git clone https://github.com/szlstart/METFAN.git && git -C METFAN checkout 048fdf7
git -C METFAN apply --ignore-space-change "$KIT/metfan_val_thresholds.patch"
```

**2．创建三个 Linux 环境。** 以下按入口实际导入的库安装。TPpred-LE 和 SA-MTP 共用旧 PyTorch/NumPy，采用 Python 3.9（作者为 3.7）。SA-MTP 原 `environment.yml` 含 Windows 的 vc/ucrt 构建。AMCL 和 METFAN 复用 V100 既有 CUDA 12.1 构建的克隆；AMCL 作者使用 2.5.1+cu124。先用 `nvidia-smi` 检查当前卡和驱动。连接外网或包源的步骤可先在登录会话完成。

```bash
conda create -y -n repro-mtp python=3.9 pip
conda activate repro-mtp
python -m pip install 'torch==1.11.0+cu113' --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install numpy==1.21.6 scipy==1.7.3 pandas==1.3.5 scikit-learn==1.0.2 matplotlib==3.5.3 seaborn==0.11.2 pyyaml tqdm transformers==4.25.1 'huggingface-hub==0.16.4'

conda create -y -n repro-imfp python=3.9 pip
conda activate repro-imfp
python -m pip install 'torch==1.12.0+cu113' --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install 'numpy<2' scikit-learn six transformers==4.24.0 'huggingface-hub==0.16.4'

conda create -y -n repro-modern --clone MFP-Nova
conda activate repro-modern
python -m pip install 'numpy==1.26.4' 'transformers==4.44.2' 'matplotlib==3.9.2' 'contourpy==1.3.0' 'seaborn==0.13.2' sentencepiece protobuf pandas scikit-learn tqdm
```

**3．准备论文输入。** 五个仓库已经带有各自的主划分。TPpred-LE 首选作者原 PSSM；如果已 clone SA-MTP，则可直接复用同一 MTP15 的 PSSM。下面做一次索引文件名检查再建立链接，省去重新跑 PSI-BLAST。只在新 clone 上运行。

```bash
cd "$PAPER_ROOT/TPpred-LE"
python - <<'PY'
from pathlib import Path
import pickle
with Path("data/pssm_seq2fn.pkl").open("rb") as f:
    names = set(pickle.load(f).values())
base = Path("../SA-MTP/features/pssm").resolve()
missing = [s for s in names if not (base / s).is_file()]
if missing:
    raise SystemExit(f"Missing {len(missing)} PSSMs. Use the author's PSSM archive.")
Path("features").mkdir(exist_ok=True)
Path("features/pssm").symlink_to(base, target_is_directory=True)
print(len(names), "indexed PSSM files available")
PY
```

- **TPpred-LE** 也可从 [作者材料页](https://bliulab.net/TPpred-LE/data/)下载 [PSSM 压缩包](https://bliulab.net/TPpred-LE/static/download/pssm.zip)，解压至仓库的 `features/pssm/*.pssm`。
- **iMFP-LG** 从[作者 TAPE 文件夹](https://drive.google.com/drive/folders/1g9ygTC8azNd6S_UnKsTSdcc1wgY90BLk)取 Hugging Face 格式的预训练权重、`config.json` 和 `vocab.txt`，放至 `$PAPER_ROOT/iMFP-LG/pretrain_model/tapebert/`。可用 `(cd "$PAPER_ROOT/iMFP-LG/Task" && conda run -n repro-imfp python -c 'from transformers import BertModel,BertTokenizer; p="../pretrain_model/tapebert"; BertTokenizer.from_pretrained(p); BertModel.from_pretrained(p)')` 检查加载。无需下载 `MFTP_alldata`。
- **SA-MTP、METFAN** 会用 Hugging Face 权重。服务器可访问 Hugging Face 时，首轮自动下载。离线时，先在可联网机器下载完整模型目录并传到 V100。SA-MTP 的 `utils/esm2_encoder.py` 中把 `model_name` 默认值指到同一份 ESM2 本地目录；METFAN 支持 `export METFAN_ESM2=/实际路径/esm2` 和 `export METFAN_PROTT5=/实际路径/prot_t5`。SA-MTP 使用仓库 `texture/{train,val,test}` 的 SS2，日志若出现“未找到ss2文件”，先检查这些文件和 FASTA ID。
- **AMCL** 使用 Git 中的 `dataset/augmented_train.txt`、`dataset/test.txt` 和发布权重，无需重新生成增强数据。入口先复测发布权重，再随机初始化完整训练 200 epoch，并用原代码在增强 Train 上确定阈值、在 Test 上打分。

**4．在已分配的一张 V100 上顺序运行。** 将 0 改成已分配的物理卡号；若集群已设置 `CUDA_VISIBLE_DEVICES`，沿用现值。线程数按分配修改。建议首次只向下列脚本传 `TPpred-LE`，然后按同样方式传其余四个名称。若资产已齐，也可一次传五个。任一模型失败后队列停止，修正后仅提交该模型和余下模型。

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
log=$(mktemp "$PAPER_ROOT/queue.XXXXXX")
nohup bash "$KIT/run_v100.sh" TPpred-LE AMCL SA-MTP iMFP-LG METFAN >"$log" 2>&1 < /dev/null &
echo "PID=$!   queue log=$log"
tail -f "$log"
```

队列日志打印每个实验的绝对目录。进入对应目录用 `tail -f run.log` 看逐轮日志。`Ctrl+C` 只结束 tail，SSH 可断开。重登后使用 `ls -td ~/paper-repro/results/* | head` 找本次目录。重试会创建新的结果目录。METFAN 两个特征都成功生成后，如仅重跑训练，可设置 `export METFAN_FEATURES=/原运行目录/features` 复用同一原始划分、相同编码器的六份张量。

| 模型 | 脚本实际做什么 | 首次应核对 |
|---|---|---|
| TPpred-LE | 发布权重复测 → seed 46 的 Algorithm 1 → square 采样的 Algorithm 2；两个阶段权重分别写入本次目录 | 发布权重及阶段 A/B 的独立 Test CSV；完整 Test 为 1,024 条 |
| AMCL | 作者权重复测 → Adam + 原 CosineScheduler、batch 256、lr 0.0007、FDL/DBL 0.99/0.01、原对比损失及 flooding → 保存新权重并测试 | 以每次的 run.log 区分两组五指标；原代码另追加写入仓库 result/model_select.csv。论文表 1 报告 0.762/0.719/0.707/0.652/0.032，测试实际进入编码 1,969 条 |
| SA-MTP | seed 50，显式 `-act t e`，原动态构图、FiLM、Val 阈值策略及独立 Test | 是否用全 SS2，权重及逐样本/逐类 Test CSV；主划分不等同于 40% identity 实验 |
| iMFP-LG | MFTP21、编号 0、batch 32、100 epoch、FGM；只训练一个成员，本次显式设随机种子 0 | 本次目录 model0.bin 与训练末尾的五指标；作者日志字符串写作 val，实际传入 test_iter |
| METFAN | 先按原 loader 的顺序、过滤条件生成 1280/1024 维按残基均值池化特征，再跑一次 TextCNN + ESM2 + ProtT5，启用特征优化。使用最佳 Val Macro-F1 对应阈值计算 Test | 日志应显示三个有效分支；结果子目录含 best_model.pth、results.json、阈值和有效分支名 |

**与论文/仓库的差异。** AMCL 公开默认入口只测试；本次补接训练类。DBL 显式采用 `rebalance`，余弦调度采用代码默认无 warmup、末端学习率 0，沿用公开循环的 FP32（论文描述 AMP）。这些属于重建配置。METFAN 论文明确两支的维度和平均池化，未在所查材料中确认 checkpoint 精确版本；选用匹配维度的 [ESM2-650M](https://huggingface.co/facebook/esm2_t33_650M_UR50D) 和 [ProtT5-XL-UniRef50 半精度 encoder](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc)，使用最后一层、排除 padding/特殊 token 的均值。分类器保留仓库的特征优化规则，Test 改用 Val 阈值，单列为重建特征下的独立测试结果。TPpred-LE 和 SA-MTP 保留公开代码对最佳模型的引用保存行为（后续训练可改变该引用），不把保存权重解释成严格的最佳轮快照。本地已用真实 AMCL 小批验证 FP32 组合损失和一次参数更新（CPU 测试时映射了上游硬编码的 CUDA 张量），并检验 METFAN 补丁可应用且固定阈值评价不再拟合 Test。V100 环境、TAPE 下载、两个语言模型完整特征和五个完整训练是待现场验证项。

## TheraPep 三任务：现在能做什么

**可行。建议以 `therapep48` 为同类论文功能预测主比较，`therapep49` 为作者原任务对照，`therapep49_positive` 用来隔离额外输出一个恒正标志的影响。** 每个模型在各任务分别训练；比较的是同一数据上的完整论文方法。保留作者编码、增强、损失、选模和阈值策略，只有标签维度、样本身份、数学上未定义的边界情况和 Test 调参问题需要适配。只有 TheraPep-AI 自身在作者原任务上的实验可称原任务复现；其余论文是在新基准上重训。

### 本地输入与共享测试视图

已直接读取本地九份 CSV，按各 split 逐行核对两个变体与完整版本正肽的序列、row_id、ID、标签和顺序。三个任务不需再下载或重新划分。CLI 任务名是 `therapep49`，`therapep_ai` 是其数据目录名。[项目读取与变体生成](../src/mfp_nova/therapep.py) · [官方数据发布](https://github.com/terra-quantum-public/tq-therapep-ai/releases/tag/assets)。

| 任务 | Train / Val / Test | 标签与样本 | 可以回答的问题 |
|---|---|---|---|
| therapep49 | 131,947 / 16,457 / 16,537 | 前48列功能 + 第49列治疗标志；各 split 有 87,423 / 10,892 / 10,971 条全零合成背景 | 在作者发布的混合任务上，模型能否同时拒识背景并预测正肽功能？ |
| therapep49_positive | 44,524 / 5,565 / 5,566 | 保留同一批正肽和49列，末列恒为1 | 相同正肽上保留治疗标志是否改变功能预测？不能验证正负鉴别 |
| therapep48 | 44,524 / 5,565 / 5,566 | 与上一行逐条相同，仅删除末列；第48列仍是 Tumor_homing | 已知是治疗肽时，48个功能预测得怎样？最接近原多功能肽论文的任务 |

**判读方法：** 每个任务先报告其完整 Test，再把三次独立训练都投到相同的 **5,566 条正肽 × 前48个功能**：固定各自训练所选权重与原生决策，给出 Micro-F1、Macro-AP/F1、样本 Jaccard。比较混合49与正肽49时输出数相同、训练背景不同；比较正肽49与正肽48时样本相同、输出不同。为了把差异解释为背景或输出的作用，同一模型的这组配对应尽量保持 seed 和选模原则相同，并注明选模目标是否包含标志。正肽49整体分与正肽48整体分不能直接相减解释成功能提升。

混合任务另报治疗标志与背景任意功能误报率；背景是合成指定负例，不是实验确认无活性。项目把空集对空集的 Jaccard 计0，故全混合 Test 的样本 Jaccard 即使完美预测也至多 5,566/16,537 ≈ 33.66%；全预测为空集反而可得约66.34%的完全匹配率。正肽49的治疗标志不具备 AUROC 判别意义。跨模型保持相同指标定义；“预测至少两个功能”不能把治疗标志算入，也不能忽略48列里的父子标签关系。

**本基准的特殊情况：** 三个任务48个功能在 Train 中均有正例；Val 的 Angiogenic 和 AntiTubercular 各有0个正例，Test 的 Angiogenic 为0。不删输出；逐类阈值搜索对这些 Val 类别没有有效正例依据，预先指定沿用原固定阈值或 Val 全局阈值，不看 Test 选择。Macro-AP 等应注明有效类别分母。保留作者发布的重复行与划分，外部 loader 不能以序列作为唯一主键覆盖样本。既有分析在正肽 Test 映射到 2,316 条训练已见簇样本，约41.61%，故此基准不支持严格未见同源簇泛化结论。这里用发布的标准大写序列，无须移植原 major15/fine39 的手性与结构字段。

### 哪些论文适合迁移

| 方法 | 对三个任务的判断 | 最小改造和实际代价 |
|---|---|---|
| **AMCL** | **48 首选；两个49版有条件可行** | 保留 TextCNN、FDL/DBL、对比学习、Train 增强和原生阈值。原增强按特定列且标签和等于1；在两个49版中每条正肽至少有一个功能和一个标志，全向量判断将使单功能增强完全失效。按前48列选功能与计算功能相似度，分类仍输出全部目标；把旧类别索引按当前 Train 类频率重建。 |
| **iMFP-LG** | 三者都适合；如需一个贯穿三任务且改动少的语言模型对照，优先选它 | TAPE微调、FGM、标签GAT和 BCE 可保留；按行读 CSV，绕开原以序列为键的 NPY 字典；令标签节点数为48/49。恒正列与全零行对普通 BCE 是合法输入。作者 [README](https://github.com/chen-bioinfo/iMFP-LG#readme) 给了 TAPE 下载入口；本次未下载验证。混合任务有约三倍训练样本，49节点的批图开销须实测。 |
| **SA-MTP** | 三者均有研究价值；与 HelixSet 比较结构证据时最值得加 | 保留 ESM2-650M + PSIPRED SS2 + 作者构图/标签解码，按当前标签改 token/Embedding。必须对正肽和背景用同一套序列特征生成方式，否则“是否有结构特征”可泄漏背景身份；不能让所有缺 SS2 的行默认为全连接图后仍称完整方法。原15类混合阈值不可直接按位置复制。 |
| **TPpred-LE** | 三者均可作经典对照，前提是有 PSSM 生成条件 | 保留 one-hot/PSSM、作者截断及两阶段训练，用当前标签训练新分类头。若启用上游按标签采样的阶段，要检查全零行是否被赋零权重，以及恒正列有没有可用的负样本，明确该阶段适用范围。 |
| **METFAN** | 有 ESM2/ProtT5 生成条件后可做 | 保留双方语言模型池化与 TextCNN。原最大长度50的过滤在混合 Train 会去掉3行（1正、2背景），两个正肽 Train 各去掉1行；Val/Test 均无超长行。按同一行掩码切全部特征。逐类阈值仅在 Val 拟合。 |
| **MCMFPP** | 48 可考虑；混合49需先修复空标签对比损失 | 两分支、ESMC 和融合均保留。CFEC 以标签交并比加权，在全零背景行与所有样本重叠均为0时仍在分子减去自身相似度，可能对负数取 log。分类应继续学习背景，在功能对比项排除无有效正对的锚点并披露；头尾类和单功能增强按功能列计算，标志独立处理。 |
| **TPpred-CMvL / MFP-MFL** | 可迁移但不作首批 | 前者要恢复 TAPE/PSSM 两阶段并适配空标签、功能配对与输出宽度；后者需三个特征分支、补缺失 FGM、保留完整集成并修正 Test 阈值拟合。 |
| **SeekTP / PDeepPP** | 暂不纳入 | 更换数据并没有补齐 SeekTP 的训练入口。PDeepPP 若在正肽49上按列训练二分类，第49列甚至只有一个类别；把它改成共享多标签模型属于新变体。 |

**AMCL 的证据边界：** 本次在 CPU 用上次核查的作者源码对每个任务各取8条真实 Train 记录（混合版4正/4背景），用原 PadEncode、TextCNN 和 Focal Dice + 0.3倍原对比损失做随机初始化的单批前向/反向。输出依次为 [8,49]、[8,49]、[8,48]，损失和参数梯度均有限。没有运行 DBL、增强、优化器更新或测试集推理。这只说明模型主体能接入。原 DBL 初始偏置含 `log(N/frequency-1)`，正肽49的恒正标志会得到 log(0)；若选用 rebalance 分支，混合49的背景行 repeat_rate 为0。正式版本应对恒正标志使用有限的普通监督项、为全零行明确定义负项权重，仅在功能列执行依赖功能频率/正对的机制，保留作者方法的其余部分，并注明适配差异。[AMCL 损失](https://github.com/xialab-ahu/AMCL/blob/627c63784e108eb19471787cc7f4b599f4c30f01/losses.py) · [增强条件](https://github.com/xialab-ahu/AMCL/blob/627c63784e108eb19471787cc7f4b599f4c30f01/DA.py#L96)。

### 现成基线与建议顺序

1. **直接利用已有 TheraPep-AI 结果。** [10月31日实测记录，第3节](TheraPep-AI-与五模型性能比较-2026-10-31.md) 已包含作者发布权重、原数据重训和两个变体独立重训结果。按 Micro-F1@0.5 选模的混合49整体 Test：发布权重79.04%，本地重训79.32%；三种训练在同一正肽48功能视图下的 Test Micro-F1 分别为 **76.25%、78.81%、78.99%**。后两者是项目变体的既有单次结果，不是作者论文声称的变体性能。无需为可行性判断重复训练 CNN。
2. **先做 therapep48 / seed42 的 AMCL、HelixSet、TheraPep-AI 同队列对照**，已有同配置结果直接复用。再以 iMFP-LG 补充可微调语言模型路线；若关注结构建模，落实 SS2 后加 SA-MTP。保留各论文固定日程、损失与决策；只有需要拟合的规则才用 Val。不能将 HelixSet 在 major15/fine39 的分数挪进这张表。
3. **把选定方法分别在混合49、正肽49重新训练，再作共同视图比较**；不要用48维模型补一个常数当成49维独立训练。对于依赖功能标签的机制，先在真实小批上覆盖全零背景和恒正列。固定方法设置后再做正式多 seed；不要求把每篇论文都套进 HelixSet 的两套选模训练。
4. 项目现有五模型研究入口，HelixSet 已有 [RTX 5070](../scripts/run_helixset_therapep.ps1) 与 [V100](../scripts/run_helixset_therapep.sh) 的三任务脚本。它们训练项目模型，不会运行外部论文。两变体可按原 row_id 复用混合版相同编码器的缓存；本次本机 `features/` 未见 `therapep_ai/` 专用缓存，不能把原主数据的缓存当作可用。未在本项目运行目录确认到 HelixSet 三任务的完整 Test 汇总；远端进度未核实。本次未启动正式训练。

## 1. 结论与比较目标

**可以推进，但没有一个仓库能在当前数据上仅修改文件路径就完成可靠的重训练和独立测试。** 多数已有可复用的模型实现；主要工作是适配当前标签、生成各自需要的特征、补接训练入口，以及处理少数会影响结果正确性的源码问题。不需要重新搭建通用训练平台。

按本次明确的需求，主实验是：**用当前数据集重新训练各论文自己的完整方法，与 TriRelay、GeoFactor、ProtoRoute、HelixSet 比较，重点观察 HelixSet。** 保留论文各自的输入表示、预处理、训练增强、损失、优化器、批大小、训练阶段、选模依据及决策规则，不统一替换为 HelixSet 的 ESM、ASL 或全局阈值。

建议选择：

| 选择范围 | 建议模型 | 作用及条件 |
|---|---|---|
| 优先实施 | **AMCL** | 无大型语言模型特征依赖，适合作为第一项接入；保留增强、对比学习与组合损失，需要补接训练入口 |
| 第一批研究对照 | **iMFP-LG** | 可训练 TAPE + 标签图，补足与冻结 ESM 模型的差异；先解决缺失的 TAPE/Hugging Face 格式资产 |
| 第一批研究对照 | **SA-MTP** | ESM-2 + 二级结构图，对 HelixSet 最有针对性；需要为当前样本生成 PSIPRED SS2 |
| 第四项经典对照 | **TPpred-LE** | one-hot/PSSM + 标签嵌入；训练实现较完整，但需准备当前数据的 PSSM |
| 扩展候选 | **METFAN** | 多来源蛋白语言模型特征融合；补特征生成并修正测试阈值拟合问题后有比较价值 |
| 后续再安排 | MCMFPP、TPpred-CMvL、MFP-MFL | 方法相关，但有较多特征、训练流程或依赖修复工作 |
| 暂不列入第一批 | SeekTP、PDeepPP | 前者缺少公开完整训练流程；后者任务形态不同，转多标签属于方法扩展 |

若只选三篇，建议 **AMCL + iMFP-LG + SA-MTP**；若希望兼顾经典方法，增加 **TPpred-LE**。这是根据方法代表性和公开实现判断的优先顺序，不是性能预测。SA-MTP 和 iMFP-LG 的外部资产尚未就绪，不能把它们描述为已经可以直接运行。

## 2. 核查范围与证据边界

本次读取了本地 `docs/references/同类研究论文.zip` 的 9 篇论文，补查此前已整理的 TPpred-CMvL 论文和仓库；读取当前项目的数据接口、ESM 编码、损失、配置和评价实现。

对下列 10 个仓库，实际通过 HTTPS 完成了 `git ls-remote ... HEAD`，读取目录树，并下载小型源码和配置作静态分析。**Git 访问成功不等于运行成功**：本次没有完整 clone 大型仓库、安装论文环境、下载模型权重或执行论文模型的 forward/backward、训练和测试。报告中的“可改造”是源码级可行性判断，具体硬件耗时与显存尚未实测。

| 模型 | 论文与作者仓库 | 核查 HEAD | 源码层面判断 |
|---|---|---|---|
| TPpred-LE | [论文](https://doi.org/10.1186/s12915-023-01740-w) · [仓库](https://github.com/HongWuL/TPpred-LE) | `9913cc9` | 有训练、重训练、独立测试；需新特征和类别参数化 |
| iMFP-LG | [论文](https://doi.org/10.1093/gpbjnl/qzae084) · [仓库](https://github.com/chen-bioinfo/iMFP-LG) | `911da39` | 有训练与预测；依赖未随仓库发布的本地预训练模型目录 |
| MFP-MFL | [论文](https://doi.org/10.3390/ijms26031317) · [仓库](https://github.com/zhouworks/Multifunctional-peptide-classification) | `1f7b35f` | 有 GAT 训练和集成预测；缺特征生成及所引用的 FGM 模块 |
| AMCL | [论文](https://doi.org/10.1186/s12915-025-02273-0) · [仓库](https://github.com/xialab-ahu/AMCL) | `627c637` | 模型、增强、损失和训练类均有；默认入口实际加载权重预测 |
| MCMFPP | [论文](https://doi.org/10.1021/acs.jcim.5c01693) · [上游仓库](https://github.com/zhao-project/MCMFPP) | `c8b18c5` | 有两个分类器训练与融合；ESMC、精度和设备适配较多 |
| METFAN | [论文](https://doi.org/10.1021/acsomega.5c07552) · [仓库](https://github.com/szlstart/METFAN) | `048fdf7` | 有训练与测试；缺预训练特征提取流程，评估存在 Test 阈值拟合 |
| SeekTP | [论文](https://doi.org/10.1093/bib/bbaf569) · [仓库](https://github.com/researchprotein/MFTP) | `8c0b3f2` | 主要为已训练模型推理；不具备直接重训练闭环 |
| SA-MTP | [论文](https://doi.org/10.1093/bib/bbag361) · [仓库](https://github.com/LZW-TECH/SA-MTP) | `3e23883` | 有训练、构图与测试；需新 SS2、标签参数化和最佳权重保存修复 |
| PDeepPP | [论文](https://arxiv.org/abs/2502.15610) · [仓库](https://github.com/fondress/PDeepPP) | `1c3daa4` | 原生单输出二分类，且训练数据装载有明确错误 |
| TPpred-CMvL（补查） | [论文](https://link.springer.com/article/10.1186/s12915-025-02466-7) · [仓库](https://github.com/Dominiczx/TPpred-CMvL) | `114f3e6` | 有多阶段代码，但默认执行路径、阶段衔接和模型保存需整理 |

MFP-MFL 的旧地址 `Zhou-Jianren/Multifunctional-peptide-classification` 已重定向至表中地址。MCMFPP 论文写的是 [xialab-ahu/MCMFPP](https://github.com/xialab-ahu/MCMFPP)，GitHub 元信息显示它是 `zhao-project/MCMFPP` 的 fork；以下详细源码结论针对所核查的上游版本，不假定两个分支永远相同。

## 3. 同一个数据集，保留各自的方法

### 3.1 共同数据来源和任务

数据使用当前只读 `data/therapeutic_2025/`，不是把以前下载的 MTP15、MFTP21、MFBP5 拼接进来。两个任务分别训练：

| 任务 | 标签数 | 基础 Train | 基础 Validation | 基础 Test |
|---|---:|---:|---:|---:|
| major15 | 15 | 39,478 | 8,534 | 8,187 |
| fine39 | 39 | 39,455 | 8,529 | 8,180 |

以上数量本次直接读取数据确认，对应 `PeptideData.indices(split, augment=False, sequence_only=True)` 的基础队列，即既有原样本 mask 与 `canonical_available` 的交集。当前发布的任务 NPZ 包含行级划分/增强 mask；不要凭空把所有标签值 0 生成“已证实阴性”的监督 mask。

具体接入保留 `row_id`、ID、`cluster_id` 和 `label_names` 顺序，以当前 `Y` 为目标。**major15 与上游 MTP 的 15 标签不是同一套类别**，不能保留原仓库的 AMP/TXP 等输出列后直接比较；fine39 必须扩展为当前 39 个输出，保留 `__unknown` 的含义。

模型可按自己的规则在 Train 内过滤、重采样或增强，并记录处理后的实际数量；共同基准的来源和划分不变。不能重新随机分割全量数据，也不能为追求高分删掉难预测标签。最终主表应覆盖相同的 Test 队列；若方法确实无法处理其中某些样本，要明确覆盖范围，不能把少测样本的结果混作完整 Test 成绩。

### 3.2 应当保留的论文处理手段

| 内容 | 主实验处理原则 |
|---|---|
| 编码与特征 | 保留各论文的 one-hot、PSSM、TAPE、ESM-2、ProtT5、ESMC、SS2 或接触图；不全部换成现有 ESM650M |
| 序列长度与 padding | 尽量遵循作者截断、补齐和掩码规则，不统一强制长度 80 |
| 增强与采样 | 保留方法的训练增强、平衡采样和对比学习；增强只能从本项目 Train 产生，不能使用作者旧数据的增强文件 |
| 模型与损失 | 保留原网络、BCE/FDL/DBL/对比损失等；不统一替换成 HelixSet 的 ASL |
| 优化策略 | 保留各自学习率、训练阶段、批大小、日程和冻结/微调方式，不要求相同 epoch 或优化器 |
| 选模 | 保留有效的论文选模依据，例如 Val loss、Val F1 或固定训练轮数；没有早停的方法不为统一形式强加早停 |
| 决策 | 保留固定阈值、逐类阈值或头尾类决策机制；需拟合/选择的阈值和超参数使用 Validation |
| 评价汇总 | 用一致的指标定义计算各模型最终预测，原生阈值结果作为主表；同时保留连续分数用于 AP |

这回答的是“同一数据上哪个完整方法更有效”，收益可能来自输入、训练和决策的共同作用，不能全部归因于网络结构。统一编码器、统一损失或统一阈值只能作为另行说明的控制实验，本次不把它们作为主实验要求。

### 3.3 已确认的本地适配细节

- **大小写与手性：** 骨架包含大小写标准氨基酸。当前 ESM 编码器在输入时转大写，而原字符串和 D 标记保留。论文通常接受大写 20-AA 序列，接入时采用相同的残基身份视图，不能把小写误判为非法氨基酸然后大量删除样本，也不能按大写序列直接合并标签或重新去重。纯序列方法不获得额外手性输入，这属于方法本身的信息差异。
- **长度：** 两任务的 Train 各有 1 条长度 74 的样本；其余 Train 以及全部 Val/Test 不超过 50，最短为 2。TPpred-LE/SA-MTP 的前后端截断可按原方式保留；METFAN 的 `PadEncode` 对超长样本直接跳过，若沿用应明确 Train 少 1 条，并同步对齐其所有特征。不要仅删除标签或某一模态。短序列需按作者 padding 方式进入模型，不能自行扩写成生物序列。
- **未标注语义：** 0 表示未标注。保留作者 BCE 等目标时，0 是训练中的代理负项，不是实验验证阴性；报告指标也是相对现有标注的表现。HelixSet 的未标注权重 0.3 和 fine39 模糊亚类降权继续属于 HelixSet 方法，不强制移植给其他模型。
- **已有特征复用：** SA-MTP 明确使用 `facebook/esm2_t33_650M_UR50D` 的末层残基表征，与当前缓存路线相容，可按 row_id 读取并转换接口，说明当前缓存 FP16 存储这一数值细节。其他方法只有在编码器、层、token 处理与池化方式确实一致时才复用；TAPE/ESM-1b/ESMC 不能拿 ESM-2 替代。

本地依据：[数据接口](../src/mfp_nova/data.py)、[ESM 编码](../src/mfp_nova/encoder.py)、[现有监督损失](../src/mfp_nova/research_layers.py)、[评价协议](../src/mfp_nova/metric_protocol.py)。

## 4. 各仓库的具体判断

### 4.1 AMCL：优先接入，保留完整训练方法

**原方法：** TextCNN、训练序列增强、多标签监督对比学习与困难样本处理、FDL/DBL 组合损失，以及类别自适应阈值。它能作为不依赖大型蛋白语言模型的完整方法对照，补足当前四模型的比较维度。

**已确认源码：** `pep_main.py → TrainAndTest()` 默认读取 `saved_models/model_select+TextCNN0.pth`，不是从头训练；真正的训练循环在 `train.py::DataTrain.train_step()`。该循环包含分类损失、0.3 倍对比损失及 flooding，但未接入默认入口，也未提供本项目需要的结果导出。[默认执行路径](https://github.com/xialab-ahu/AMCL/blob/master/train_test.py#L83)、[训练类](https://github.com/xialab-ahu/AMCL/blob/master/train.py#L30)。

**必要改造：** 接入当前三份数据；`output_size` 改为 15/39；从当前 Train 重新运行 `DA.py` 的增强并重新计算 DBL 类频率，不能使用原 `augmented_train.txt`；补接优化器、训练、权重保存和预测导出。`losses.py` 的频率来源路径以及阈值函数中 `[0] * 21` 都需要适配。[增强代码](https://github.com/xialab-ahu/AMCL/blob/master/DA.py)、[阈值逻辑](https://github.com/xialab-ahu/AMCL/blob/master/train.py#L138)。

**保真注意：** 保留作者组合损失及其阈值变换。代码不是简单的 `sigmoid(logit) > threshold`，不能把它“简化”后仍称原版 AMCL；原入口在 Train 上拟合阈值，本项目将拟合输入改为 Val，并注明这项协议调整。论文描述 AMP，而当前训练类未见 autocast；接入时明确选择论文描述的训练精度还是公开代码的实际精度。

**判断：可基于 clone 改造，代码工作量中等、外部特征准备负担较低，建议最先做。**

### 4.2 iMFP-LG：有完整训练主体，预训练资产是主要缺口

**原方法：** 可训练 TAPE 蛋白语言模型与功能标签 GAT，带 FGM 对抗训练。TAPE 会参与任务微调，不能用冻结 ESM 特征替代，否则不再是论文方法。

**已确认源码：** `Task/TaskForMultiPeptide_MFTP.py` 有优化器、常规及对抗反向传播、训练与测试；模型大部分循环使用 `config.num_labels`，由 21 类改成 15/39 比较直接。原入口按固定轮数训练并保存，未提供独立 Val 选模循环，不能仅因它有 Train/Test 两份文件就指控其使用 Test 选模。[训练入口](https://github.com/chen-bioinfo/iMFP-LG/blob/master/Task/TaskForMultiPeptide_MFTP.py)、[模型](https://github.com/chen-bioinfo/iMFP-LG/blob/master/models/BertForMultiLabelWithGAT_MFTP.py)。

**实际缺口：** 代码要求 `pretrain_model/tapebert/` 内有权重、`config.json`、`vocab.txt`，使用 Hugging Face `BertModel/BertTokenizer` 读取，但当前仓库树未提供该目录或转换脚本。需要取得匹配的 TAPE 资产或完成其到该实现的转换；换成一般 BERT/ProtBERT 不能算忠实复现。

**必要改造：** 当前数据和标签接入、类别数与图尺寸、设备与路径、固定 seed、概率导出；按论文原固定日程和固定阈值执行，若另行调参则使用既有 Val。旧 `transformers.AdamW` 接口需要随所选环境适配。fine39 的标签图更大，需要实测批大小，但不要先重写 GAT 算法。

**判断：可改造、方法价值高；在拿到正确 TAPE 资产前，不承诺仅 clone 后即可启动。**

### 4.3 SA-MTP：与 HelixSet 最相关，但 SS2 不能省略

**原方法：** 冻结 ESM-2 650M、PSIPRED 二级结构概率、结构图/GAT、FiLM 与标签解码。它可对照 HelixSet 的几何与残基对建模，但两者结构来源不同，应报告为完整方法比较。

**已确认源码：** 有 ESM 提取、SS2 读取、图构建和 Train/Val/Test 入口。`dataset.py` 的标签 token 范围和 `models/tppred.py` 的标签 Embedding 均写死 15；fine39 必须同步参数化。[数据类](https://github.com/LZW-TECH/SA-MTP/blob/main/dataset.py#L24)、[模型](https://github.com/LZW-TECH/SA-MTP/blob/main/models/tppred.py)、[特征编码](https://github.com/LZW-TECH/SA-MTP/blob/main/utils/esm2_encoder.py)。

**关键问题：** 缺少某个 SS2 时，`graph_builder.py` 会改用全连接图。这能让程序继续跑，却不是完整结构方法。应为当前样本生成 SS2，不拿原 MTP 样本的 `texture/` 文件套用，也不把项目的 DSSP 硬标签直接充作 PSIPRED 的 H/E/C 概率。[SS2 缺失分支](https://github.com/LZW-TECH/SA-MTP/blob/main/utils/graph_builder.py#L250)。

**另一处必要修复：** `train_all()` 在最优轮只做 `best_model = model`，训练结束后才保存；该引用会随后续优化继续变化，应在改进时保存状态。保留原 Val loss 选模和 patience=15，而非换成 HelixSet 的 Jaccard 选模。[选模与保存](https://github.com/LZW-TECH/SA-MTP/blob/main/run_network.py#L407)。

**保真注意：** 图中名为 contact map 的量实际由 ESM 残基表征余弦相似度得到；不是独立预测的三维接触矩阵。按作者实现保留并在方法描述中准确命名。原逐类 F1/MCC/fixed 混合阈值表绑定 MTP 类别，不能按列位置套到当前 15/39 类；应采用作者通用阈值优化机制在 Val 上拟合，并注明原类别专属配置没有一一对应关系。[图构建](https://github.com/LZW-TECH/SA-MTP/blob/main/utils/graph_builder.py#L119)、[原配置](https://github.com/LZW-TECH/SA-MTP/blob/main/config.yaml)。

**判断：可改造，优先比较价值高；外部准备主要是 PSIPRED/SS2，而不是下载原论文数据。**

### 4.4 TPpred-LE：经典对照，PSSM 成本高于模型接入成本

**原方法：** one-hot + PSSM、Transformer 标签嵌入和功能特定分类器；仓库有两阶段训练，另有 MCRT 入口。主比较应按论文主结果所对应设置执行，不把某个局部训练阶段当成完整模型。

**已确认源码：** `run_network.py` 包含训练、重训练和独立测试；输入为 FASTA + 带列名标签 CSV。与 SA-MTP 一样存在 15 类写死和 `best_model = model` 的可变引用保存问题，需要参数化并真正保存选中轮状态。[训练实现](https://github.com/HongWuL/TPpred-LE/blob/main/run_network.py#L196)、[标签嵌入](https://github.com/HongWuL/TPpred-LE/blob/main/models/tppred.py#L24)。

**必要特征：** 对当前骨架生成 PSSM，重建序列与特征的对应关系；仓库原特征映射不能用于新样本。作者提供 PSI-BLAST 调用以及无匹配时的 BLOSUM 回退，可保留这一本来就存在的处理并报告实际回退比例，不能为了省事把所有 PSSM 换成 one-hot 或 ESM。[输入组合](https://github.com/HongWuL/TPpred-LE/blob/main/utils/load_data.py#L76)、[PSSM/BLOSUM 路线](https://github.com/HongWuL/TPpred-LE/blob/main/utils/encoding_methods.py#L127)、[搜索脚本](https://github.com/HongWuL/TPpred-LE/blob/main/utils/psiblast_search.py)。

保留作者长度 50 的前后端拼接截断策略；当前仅 1 条训练样本受影响。处理旧 NumPy `np.int` 等兼容问题，不必为旧依赖把 RTX 5070 的 PyTorch 降回 cu113 构建。

**判断：可改造，适合经典基线；主要不确定成本是序列数据库和 PSSM 生成，不能按“轻量 Transformer”估算整个方法的成本。**

### 4.5 METFAN：有价值的扩展候选，不能直接沿用评估脚本

**原方法：** TextCNN 局部特征 + ESM2/ProtT5 的全局特征、特征优化与融合网络，配 Focal Dice Loss。

**公开实现缺口：** 仓库只有模型、特征加载器和训练/评估代码，没有当前样本的 ESM2/ProtT5 提取脚本。默认 `features=['textcnn']`，不存在的特征会被忽略；直接跑默认入口不等于复现完整 METFAN。需按论文生成相应池化特征、显式启用完整组合与优化模块。[默认配置](https://github.com/szlstart/METFAN/blob/main/config.py)、[特征加载器](https://github.com/szlstart/METFAN/blob/main/feature_loader.py#L44)。

**明确的评估问题：** `train.py::evaluate_model()` 用 `test_labels, test_preds` 找阈值；`evaluation.py::evaluate()` 又对传入数据拟合逐类 F1 阈值。把测试数据传入原函数会得到 Test 拟合成绩，应保留逐类 F1 方法，但在 Val 拟合后把阈值传给 Test，不能直接调用原函数出主表。[测试流程](https://github.com/szlstart/METFAN/blob/main/train.py#L407)、[评估函数](https://github.com/szlstart/METFAN/blob/main/evaluation.py#L189)。这描述的是当前公开代码路径，不据此推断论文所有实验的执行方式。

论文描述 ESM2 和 ProtT5 的均值池化，代码却保留 `esm1v_max` 等候选名称；不能依据默认特征名猜论文最终编码器。需要落实具体 checkpoint 和池化设置。原 loader 会过滤超长序列，需把同样的筛选应用到对应预训练特征，并记录 Train 数量变化。[数据处理](https://github.com/szlstart/METFAN/blob/main/data_utils.py#L34)。

**判断：可改造，工作量中等；比单纯换数据接口多出特征生成与评估修复，适合第二批。**

### 4.6 MCMFPP：训练链存在，ESMC 与设备适配需要单独处理

**原方法：** SLFE 与 CFEC 两个分类器，使用 ESMC、卷积、单功能样本增强、监督对比学习，以及头尾类加权预测融合。不能只训练其中一个分类器并以 MCMFPP 名义比较。

**已确认源码：** 有 `train_slfe.py`、`train_cfec.py`、`predict_MCMFPP.py`。类别数 21、标签名及头尾类索引写死，需要改成当前任务，并按作者划分原则用当前 Train 支持度重新确定头尾类。原融合权重、双阈值机制可以保留；需要重新选择参数时只能用 Val。[融合入口](https://github.com/zhao-project/MCMFPP/blob/main/predict_MCMFPP.py#L5)。

**设备实际障碍：** `models.py` 导入阶段就构建 ESMC、读取本地 300M 权重，并设置 `use_flash_attn=True`；凡是 CUDA 就转 BF16。这不能直接照搬到 V100。Windows 侧也不宜假设旧 FlashAttention 依赖可直接工作。可把冻结 ESMC 特征离线提取，按设备选择支持的标准注意力/精度，保留同一 ESMC 模型与表征，记录数值设置差异。[ESMC 初始化](https://github.com/zhao-project/MCMFPP/blob/main/models.py#L200)、[FlashAttention 硬件与系统支持](https://github.com/Dao-AILab/flash-attention#installation-and-features)。

两分类器的训练循环会访问名为 Test 的 loader；接入时移出训练阶段，正式 Test 在最终模型和融合规则确定后评价。分类器的原定训练日程可保留，不必改成四模型的统一 100 轮。

**判断：可以基于原代码推进，工作量偏高。ESMC 与两个分类器的总成本应单独计入；不要求按 README 的 RTX 4060 才能训练，也不能在其他设备上直接使用作者任务权重冒充重训练。**

### 4.7 TPpred-CMvL：双视图方法相关，但不适合作为第一项接入

**原方法：** TAPE 语义特征与 PSSM 进化特征、多视图对比学习及后续分类阶段。[论文方法](https://link.springer.com/article/10.1186/s12915-025-02466-7)。

**源码问题：** README 推荐的 `tape_model.py` 含作者机器 `/home/xkr/.../semantic*.npy` 路径；默认只执行 `train_second()`，第一阶段调用被注释，多处保存模型的调用也被注释。分类阶段创建 `FModel2()`，需要核实并恢复论文要求的第一阶段参数传递，而不是默认把两个阶段当成已经连通。`pssmModel.py` 多个输出头写死 15，另一路 `models/TapeModel.py` 也固定 15。[特征入口](https://github.com/Dominiczx/TPpred-CMvL/blob/main/tape_model.py#L207)、[阶段与保存](https://github.com/Dominiczx/TPpred-CMvL/blob/main/tape_model.py#L606)、[分类头](https://github.com/Dominiczx/TPpred-CMvL/blob/main/pssmModel.py#L123)。

**必要改造：** 新样本 TAPE/PSSM、类别数与标签列、对比阶段数据索引/样本数量、阶段衔接和权重保存。保留原多视图学习及分类损失；把训练中反复访问 Test 的代码移到最终评价。现有发布代码与论文完整流程的差异必须写清楚，不能仅把路径替换完就宣称复现。

**判断：能够复用模型代码，但流程整理工作偏多，建议后续再做。**

### 4.8 MFP-MFL：集成方法成本较高，不能只拿一支 GAT 代表全文

**原方法：** ESM-2、ProtT5、RoBERTa 特征分别进入标签 GAT，再作预测集成。

**公开实现问题：** `GAT_train.py` 引用 `from model.FGM import FGM`，但当前完整仓库树中没有对应模块；默认数据路径也与发布的 `data/` 目录不一致。仓库发布的是原样本特征，没有新样本三个编码器的完整提取流程。[训练入口](https://github.com/zhouworks/Multifunctional-peptide-classification/blob/main/model_GAT/GAT_train.py)。

`predict.py` 默认融合 30 个成员，即三类特征各 10 个模型，且用测试标签重新计算阈值。保留原集成思想时，要明确外层重复实验与内层集成成员的区别；不能把一个成员标成完整 MFP-MFL，也不能把 30 个成员当作 30 次独立的完整方法结果。[集成与阈值](https://github.com/zhouworks/Multifunctional-peptide-classification/blob/main/model_GAT/predict.py#L90)。

**必要改造：** 补缺失 FGM、落实三个预训练编码器及池化方式、适配 15/39 类并新增 Val 特征/决策流程；保留各分支与融合，阈值只在 Val 拟合。若先只做 ESM-GAT，应明确标为该方法的单分支变体。

**判断：可部分复用，但完整方法的资产和训练负担偏高，不建议第一批。**

### 4.9 SeekTP：原数据下载问题不是此次的主要障碍，缺训练流程才是

**原方法：** ESM-1b、NetSurfP-3.0 二级结构/RSA、trRosetta 接触信息，以及 GAT、注意力和 CNN。其原实验筛选为 8 类，但当前比较应适配完整 15/39 类，不能照搬筛掉稀有类别的任务范围。

**公开代码实际提供：** README 运行的是 `TestGAT.py`；该文件定义模型后加载 `gat.pkl` 进行预测。仓库只有少量推理、特征读取和损失代码，未发现完整训练脚本；读取器固定 8 位标签、50 个残基、1304 维特征。[使用说明](https://github.com/researchprotein/MFTP)、[推理入口](https://github.com/researchprotein/MFTP/blob/main/TestGAT.py)、[特征读取](https://github.com/researchprotein/MFTP/blob/main/utility.py)。

此前本地记录显示其百度网盘材料链接失效；本次没有重复验证该网盘状态。由于现在使用本项目数据，缺作者原 Train/Test 本身并不阻止研究；真正缺的是新样本特征生成与训练执行细节。可以按论文自行实现，但应称“基于论文和公开推理实现的重建”，不能称直接 clone 后重训练。

**判断：不属于简单接口适配，暂不推荐第一批。**

### 4.10 PDeepPP：可作未来扩展，不宜列为原生多标签对手

**任务差异：** 本地论文和发布数据涉及多个独立二分类/位点任务。公开 `PredictModule` 最后一层为 `Linear(64, 1)`；它不是一次输出当前 15/39 个功能的多标签模型。[模型输出](https://github.com/fondress/PDeepPP/blob/main/train_test/util/model.py#L69)。

**直接可见的运行问题：** `train.py` 调用 `get_dataloaders(batch_size)`，定义却要求 `(batch_size, args)`；loader 把 `test_representations.npy` 赋给训练标签，把 `train_labels.npy` 赋给测试特征。标签 loss 中又使用 `label.view(-1)`，不能只把最后一层改成 39 就结束。[训练调用](https://github.com/fondress/PDeepPP/blob/main/train_test/train.py)、[数据装载](https://github.com/fondress/PDeepPP/blob/main/train_test/util/data_loader.py)、[损失接口](https://github.com/fondress/PDeepPP/blob/main/train_test/util/loss_function.py)。

若将来纳入，可考虑每类独立训练一个原生二分类模型，或做共享编码器的多标签扩展。两者训练规模与方法含义不同，且当前 0 并非确认阴性；必须单独命名并解释，不能默认为论文原始多标签实现。

**判断：本次不建议选入主比较。**

## 5. 与当前四模型比较时如何报告

### 保留原生决策，但统一指标含义

主表使用各方法自己的有效选模/决策规则，并列明选模指标、固定或自适应阈值、输入模态和是否微调语言模型。**不要求所有方法改为 HelixSet 的 traditional5/global。** 当前前三模型的 AP 选模与 HelixSet 的传统指标选模可以作为各自方法设置保留，但需要显式列出，不能声称它们进行了统一选模控制实验。

最终对相同 Test 样本计算：传统五指标（Acc=样本 Jaccard、Precision、Coverage、AT=集合完全匹配率、AF=Hamming loss）、Macro/Micro AP、Macro/Micro F1，以及现有定义的独立多功能识别指标。AF 越低越好；不把它解释成传统二分类的假阳性率，不把标签数当成独立生物功能数。AP 从连续分数计算，不能从已阈值化的标签重新计算。

不同原生阈值有利于各自优化目标，这属于完整方法结果。报告时以多指标表展示，避免仅按某一种方法特别优化的指标宣布全面优胜。若后续需要区分“概率排序能力”和“决策规则收益”，可以使用已有预测做补充分析，无需现在统一重写各论文决策。

### 样本与训练信息的使用边界

原论文的 PSSM/SS2/固定预训练编码器可以分别用于 Train/Val/Test 的特征生成；这种不使用功能标签的推理不是 Test 调参。标签频率、类别图统计、标准化/PCA 拟合、增强来源、监督对比学习或任务相关微调只使用 Train。原作者固定的超参数可直接作为起点；重新搜索时用既有 Val 或 Train 内的按簇划分，不能重新做跨全量随机 KFold。

作者提供的旧任务分类权重不能作为“当前数据重训练”成绩，也不宜用来初始化本次任务头：这些权重使用过另一套功能标注，且与本项目样本可能重叠。通用预训练蛋白模型仍按论文使用，任务相关模块重新训练。不能把“重新训练”误解为连通用蛋白语言模型也必须从零预训练。

当前公开方法多数只看序列及由序列预测的结构，HelixSet 还可使用项目已有几何、理化、修饰/手性信息。主比较可以保留这些差异，但结论应是完整方法性能；如果要单独证明 HelixSet 网络结构优势，再安排信息来源相同的消融，本轮不强制扩大实验量。

### 当前结果状态

依据当前 README，TriRelay、GeoFactor、ProtoRoute 已有两任务各五 Seed 的首轮独立 Test；HelixSet 在 2026-10-24 完成整体架构重构，当前文档尚无重构后真实训练/测试结论。旧 HelixSet 的 Val 数值不能作为新结构的 Test 成绩。

已有三模型结果若数据队列、标签、模型版本和对应方法设置一致，可作为其原生配置结果保留并统一汇总；仅靠旧 `results.json` 的数字不一定能补出所有新指标，需要对应概率或可用 checkpoint。主表明确每一结果使用的模型版本与设置，不凭此次静态检查判断 HelixSet 优于任何论文模型。

## 6. 建议实施顺序与工作量

1. **先接 AMCL，跑通一个任务、一个 seed。** 建立最小的数据读取和预测导出，再接作者增强、完整 loss、训练与原生决策；只做必要的真实小批 forward/backward、标签对齐和短训练检查。
2. **准备 iMFP-LG 的正确 TAPE 资产与 SA-MTP 的 SS2。** 两项准备互不依赖。SA-MTP 可复用相容的冻结 ESM 缓存，但不能省掉 SS2；iMFP-LG 保留 TAPE 微调。
3. **按选定论文扩展至两任务。** TPpred-LE 需要安排 PSSM 生成；若已有 ProtT5 特征生成条件，也可优先实施 METFAN。不要因某模型前几轮 Val 表现较差就删除该对照。
4. **完成正式重复实验。** 建议沿用项目 seed：42、101、202、303、404，报告均值与标准差；固定每种方法自己的超参数后再重复，不逐 seed 单独调参。新 HelixSet 也需要对应的真实实验。

前三个外部方法 × 两任务 × 五 Seed 是 **30 组完整方法实验**；加入 TPpred-LE 后为 **40 组**。加上项目四模型对应的 40 组，共 80 组结果单元，但符合口径的既有结果可复用，**不等于必须新启动 80 次训练**。两阶段方法、双分类器或多成员集成的一组实验包含多个内部训练，不能把结果单元数当作 GPU 作业数。

| 方法 | 接入工作量（静态估计） | 主要额外资源 |
|---|---|---|
| AMCL | 中：补接现成训练类、增强/类频率、决策和导出 | 对比学习的真实批大小、训练增强样本 |
| iMFP-LG | 中；预训练格式取得/转换有不确定性 | TAPE 微调与 FGM 的显存/训练开销 |
| SA-MTP | 中：标签、缓存/SS2 对接、最佳权重保存 | PSIPRED 工具与所需数据库/计算 |
| TPpred-LE | 中；PSSM 准备可能占主要时间 | PSI-BLAST、搜索数据库与 CPU/I/O |
| METFAN | 中：特征生成、完整配置、阈值修复 | ProtT5 与具体 ESM 模型的特征提取 |
| MCMFPP | 中高：ESMC 设备路线、双分类器和融合 | ESMC 特征及两个模型训练 |
| TPpred-CMvL | 高：恢复多阶段流程和新特征 | TAPE、PSSM、对比学习 |
| MFP-MFL | 高：补模块、三类特征、完整集成 | 三套特征与较多集成成员 |
| SeekTP | 高：训练与特征流水线重建 | ESM-1b、NetSurfP、trRosetta |
| PDeepPP | 高，且涉及任务扩展 | 多个二分类模型或另行设计的多标签版本 |

以上是改造复杂度判断，不是实测工期。各方法先用一小段真实训练测量单轮耗时/显存，再估算完整任务；没有依据给出“几小时全部跑完”。监督对比学习的 microbatch 会影响正负样本集合，梯度累积不能自动等价于作者的大批对比损失，需要在资源调整时明确记录。

## 7. 两平台执行准备与最小实现边界

- **RTX 5070 / Windows：** 按项目约定保留支持 Blackwell 的 PyTorch 2.8.0+cu128 路线。不要直接安装论文旧 `requirements` 覆盖为 cu113/cu116。需要编译扩展的图包和 FlashAttention 单独处理；Windows 多进程入口保留 main guard，数据线程按实际情况设置。
- **V100 / Ubuntu：** 保留从 `trifusion_mfp` 克隆的独立 `MFP-Nova` 环境及 PyTorch 2.4.1+cu121。ESMC 的 BF16/FlashAttention 路线需要调整，不能未经评估原样运行。训练命令在实现完成后同时提供 GPU、CPU 线程选择及 `nohup` 后台运行形式。
- **必要依赖冲突：** 若某论文确实需要冲突的依赖，使用独立环境运行该方法，不升级或降级现有四模型环境。通用预训练特征可以离线生成，数据和小型源码遵守现有三机同步约定。
- **代码组织：** 选中哪些方法后再正式 clone 并逐项修改。优先在上游模型实现上加一个普通数据接入/运行脚本，共用当前数据读取和评价函数；保留作者训练逻辑，不构建注册器、通用模型工厂或强制包进 `research.py` 的所有辅助损失。
- **必要输出：** 方法名/配置、seed、代码版本、数据标识、选中 epoch、环境、Test 指标，以及按 row_id 对齐的 Val/Test 连续分数、原生二值预测和决策参数。这样既能核对真实样本对齐，也能统一汇总，不需要复制源码保存历史或生成额外审计文件。

本次只增加这份分析报告及文档导航。主数据、四模型源码、配置和训练结果没有修改，没有启动 GPU 任务。后续最具体的起点是：**保留原方法的 AMCL 重训练接入，同时解决 iMFP-LG 预训练资产和 SA-MTP 的 SS2 生成条件。**
