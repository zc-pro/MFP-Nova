# HH-SetReread V1.7 架构与训练计划

V1.7 在原 Fine39 同源簇划分上以样本平均 Jaccard 为主指标。长期目标是 **70% ACC**。参考模型是 seed 42 的 Fine39 TheraSet `no_label_context_single_prototype/micro_f1`，固定标签阈值 0.50，Validation ACC **61.0463%**。原始 Train 39,455 条，Validation 8,529 条，长度上限维持 80。V1.6 的联合集合模型可从 Git 提交 `81dc864` 查阅。

## 1. 设计依据

| 对照 | Fine39 Validation ACC | 对架构的约束 |
|---|---:|---|
| 原 TheraSet | 61.05% | 每条肽始终可以保留参考集合 |
| V1.6 单分量 / 完整联合头 | 59.52% / 59.48% | 保留原判别路径。扩大集合表达需要先证明真实标签排序改善 |
| 历史变体、Train 近邻、短片段的固定数量纠错 | 61.87% | 局部序列和近邻有独立于参考分数的有效信号 |
| 同证据、同候选的标签概率与加删换诊断 | **62.29%** | 先学标签身份，再计算各动作的收益。直接预测动作收益得 62.13%，两者差的簇配对区间跨零 |
| 历史三 TheraSet 概率平均和自适应截断 | 62.82% | 额外的开发对照。三份权重曾使用同一历史 Validation 选模 |
| 真值挑选单次最有利动作 | 79.05% | 仅说明候选覆盖；预测时无法使用真值选动作 |

诊断的 62.29% 使用历史变体分数、轻量短串模型和簇留出评分器。V1.7 正式实现使用**单个** TheraSet、可训练残基读出和近邻，因此须重新验证结果。此前只允许替换时完全修复 73 条错类单标签，允许所有动作时降到 41 条。独立多功能 F1 升至 58.42%，第三正确功能轴仍净损失。架构将标签身份和共存证据分别输出，并分别记录正确轴恢复与丢失。

## 2. 输入与参考输出

~~~mermaid
flowchart LR
    X["肽及缓存特征"] --> T["固定 TheraSet"]
    X --> R["按标签读取局部片段与端点"]
    X --> N["原 Train 簇近邻"]
    T --> I["标签身份分数"]
    R --> I
    N --> I
    T --> C["已选标签与候选的共存证据"]
    R --> C
    T --> D["保留、单加、单删、单换"]
    I --> D
    C --> D
    D --> Y["Fine39 标签集合"]
~~~

沿用多模态 TheraSet checkpoint 及两种现有残基缓存 `features/esm2_650m`、`features/esm2_peptide_150m`。参考分数为 \(u\in\mathbb R^{39}\)，参考集合为 \(S=\{c:u_c\ge0\}\)。Train 的参考分数来自 V1.3 的三份同源簇折外预测，Validation 与推理使用原完整 TheraSet。训练新头时只读取预先生成的参考分数，TheraSet 参数固定。调用模型推理时仍会正常运行原多模态路径。

可选近邻读取肽 ESM150M 残基均值，经 L2 归一化后检索 16 个原 Train 簇代表。每个类别读取支持率、已记录该类的最高相似度和最高整体近邻相似度。Train 查询排除其整个折，Val/Test 查询使用原 Train 库。最后的 checkpoint 携带该库，可对输入独立推理。

## 3. 标签局部证据读出

1. 两种 ESM 视图分别按残基归一化，拼接投影到 \(d=64\)，再加 20 种氨基酸的嵌入。Padding 位置归零。
2. 宽度 3 和 7 的逐通道卷积读取不同跨度的局部环境，与原残基表示拼接。加入距 N/C 端的归一化位置及两端前三个残基的指示。核宽是首轮对照配置，输入长度由原 mask 控制。
3. 39 个标签各用一个查询做掩码注意力池化，保留值为零的空槽，得到每类局部向量 \(e_c\)。较短肽仍可用同一组卷积，位于真实序列之外的值为零。
4. 共享身份头接收 \(e_c\)、标签嵌入、参考 logit 的有符号对数、参考是否选中及近邻支持。输出
   \[
   \ell_c=2\sigma(s_c)u_c+b_c+r_c,\qquad p_c=\sigma(\ell_c).
   \]
   \(s_c,b_c\) 与残差末层初始为零，因此初始身份分数与 TheraSet 一致。训练后 \(p_c\) 用于细标签排名和动作评分，参考集合 \(S\) 仍由 \(u\) 定义。

训练目标是 Fine39 **记录标签**。使用原样本的 39 位 BCE，模型学习标签被记录的概率。零位表示未记录，分数与损失不推定真实生物学阴性。`__unknown` 属于当前细标签记录，不用未知亚类标签制造独立功能轴。

## 4. 共存证据

对 \(c\notin S\)，\(A(S)\) 表示按项目规则投影到 15 个有效轴。只有 \(S\) 非空、\(A(S\cup\{c\})\) 增加一个轴且保留已有轴时，\(c\) 才进入共存分支的适用范围。已有具体抗微生物轴时，泛 antimicrobial 轴按项目规则被抑制。由泛轴改成具体轴的候选仍可参与 Fine39 动作。

分支对参考集合内的标签证据做均值，得到背景 \(h_S\)。候选的局部证据、背景、逐元素交互及身份特征供共享小头输出 \(q_c\)。在符合适用范围且 \(c\) 有记录的训练位上，\(q_c\) 的正例要求该新轴有记录、至少一个原预测轴也有记录。这是“已知候选标签有记录”条件下的共存事件。目标为：

\[
L=L_{\mathrm{recorded\ label}}+0.2L_{\mathrm{coexistence}}.
\]

当一批中没有适用训练位时，第二项为零。以经过同一 Fine39 投影的正确轴统计补充评价，分别记录跨入和跌出两个、三个正确轴的样本数。

## 5. 推理与验证

设 \(\hat h=\sum_{c\in S}p_c\)、\(\hat u=|S|+\sum_{c\notin S}p_c\)。动作加入或移除标签使预测大小改变 \(\Delta s\)，交集的估计改变 \(\Delta h\)。统一的候选收益为

\[
 \widehat{\Delta J}=
 \frac{\hat h+\Delta h}{\hat u+\Delta s-\Delta h}
 -\frac{\hat h}{\hat u}.
\]

这是代入预期交集和并集的近似，实际验证在记录集合上计算 Jaccard。单加时 \(\Delta h=p_c,\ \Delta s=1\)。单删时 \(\Delta h=-p_d,\ \Delta s=-1\)。单换时 \(\Delta h=p_c-p_d,\ \Delta s=0\)。对符合适用范围的单加，另加 \(0.05p_c(2q_c-1)\) 作为共存证据的有限调整，可通过 `--ablate coexistence` 消融。固定大小的替换收益对 \(p_c-p_d\) 单调，取未选中最高分和已选中最低分即可。

每条肽至多执行一个单加、单删或单换，保留视为零收益动作。默认让三种动作竞争，单种对照使用 `--hh-v17-actions add|drop|swap`。保留空参考预测，且不通过删除制造新的空集合。每轮在 Validation 上使用真实 Jaccard 拟合一个非负编辑阈值，用 Jaccard 和 Macro AP 选择 checkpoint。Test 直接使用保存的决策。Validation 小样本检查只能衡量运行情况。

训练保存 `metadata.json`、`history.jsonl`、`best.pt`、`results.json`、`best_validation.npz`。验证输出含逐行参考分数、标签概率、实际集合、增删标签与 Jaccard 变化。结果报告参考基线、高置信错类命中、完整修复错类单标签、被改坏的原正确样本、第二及第三记录标签排名和正确轴变动。加入新分支后，优先检查这些指标及原 TheraSet、无新增序列证据、无共存、仅替换对照。只分数对照用 `--ablate evidence neighbors coexistence`，无近邻版本用 `--ablate neighbors`。

## 6. 训练命令

使用已经包含 V1.7 的 checkout。两个平台使用同一 seed 157、FP32、batch 64、学习率 \(3\times10^{-4}\)、最多 30 轮、patience 6。首次正式训练需要参考权重和 V1.3 的三个**完整**折外文件。检查本地路径，RTX 5070 可将这些小型训练输入及权重放到同一相对路径或修改变量。缓存保留长度 80。

### V100（Ubuntu Bash）

~~~bash
cd ~/wuzhencheng/projects/MFP-Nova
conda activate MFP-Nova
GPU=0
CPU_THREADS=6
BASE=runs/theraset_followup_seed42_v100/fine39/no_label_context_single_prototype/micro_f1/best.pt
OOF_PREFIX=runs/hh_v13_fine39_20261115_024558
OOF=("${OOF_PREFIX}_fold0/oof_logits.npz" "${OOF_PREFIX}_fold1/oof_logits.npz" "${OOF_PREFIX}_fold2/oof_logits.npz")
RUN="runs/hh_v17_fine39_s157_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"

nohup env CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" \
  PYTHONPATH="$PWD/src" python -u -m mfp_nova.cli train \
  --model hh_setreread_v17 --task fine39 --seed 157 \
  --device cuda:0 --precision fp32 --batch-size 64 --num-workers 0 \
  --epochs 30 --learning-rate 0.0003 --metric-suite jaccard \
  --initialize-from "$BASE" --hh-oof-files "${OOF[@]}" \
  --name "${RUN#runs/}/full" > "$RUN/full.log" 2>&1 < /dev/null &
printf 'PID=%s LOG=%s\n' "$!" "$RUN/full.log"
~~~

### RTX 5070（Windows PowerShell）

~~~powershell
Set-Location 'D:\Data\Code\MFP-Nova'
conda activate MFP-Nova
$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '6'
$env:MKL_NUM_THREADS = '6'
$env:PYTHONPATH = "$PWD/src"
$base = 'runs/theraset_followup_seed42_v100/fine39/no_label_context_single_prototype/micro_f1/best.pt'
$oofPrefix = 'runs/hh_v13_fine39_20261115_024558'
$oof = 0..2 | ForEach-Object { "${oofPrefix}_fold$_/oof_logits.npz" }
$run = "hh_v17_fine39_s157_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Path "runs/$run" | Out-Null

python -u -m mfp_nova.cli train --model hh_setreread_v17 --task fine39 --seed 157 `
  --device cuda:0 --precision fp32 --batch-size 64 --num-workers 0 `
  --epochs 30 --learning-rate 0.0003 --metric-suite jaccard `
  --initialize-from "$base" --hh-oof-files @oof --name "$run/full" 2>&1 |
  Tee-Object -FilePath "runs/$run/full.log"
~~~

以上 PowerShell 命令的续行符是反引号。正式跑完后，主结果位于 `runs/<实验目录>/full/results.json`。需要复算时，把对应权重传给 `evaluate --split val --hh-diagnostics`。若 RTX 或 V100 没有折外文件，可以迁移三个 V1.3 导出，或在三个同源簇折上用 `train --model theraset --task fine39 --hh-oof-fold 0|1|2` 分别重建，再把三条实际生成路径传给 `--hh-oof-files`。重建时使用现有长度 80 的两套特征缓存。
