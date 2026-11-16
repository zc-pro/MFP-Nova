# MFP-Nova V5.0 架构书

V5.0 在 Fine39 原同源簇划分上检验两个问题。局部序列、全序列上下文和残基对能否各自形成有用的标签预测。功能之间的语义关系能否帮助这些路径找到更有区分力的肽内证据。

历史主基线为 TheraSet 的 `no_label_context_single_prototype`，Validation ACC **61.0463%**，固定阈值 0.50，按验证 Micro-F1 选模。V5.0 主指标为样本平均 Jaccard，记为 ACC。新实验用同批按 ACC 选出的原 TheraSet 作主要对照。目标为 70%。 [历史基线](../README.md#fine39-性能基线)

## 1. 实验依据

| 实验 | 观察 | 对 V5.0 的设计决定 |
|---|---|---|
| B0 局部结构对照 | AA34 50.89%，加入局部最大池化 53.47%，再加入局部 CNN 57.32%，均为对应开发对照 | 保留 CNN 的多尺度残基特征图作为局部主干 |
| TheraSet 消融 | Fine39 固定 0.50 下，纯 CNN 56.19%，化学 CNN 57.78%，完整 TheraSet 58.14%，关闭标签上下文且每标签一个原型后 61.05% | 保留强配置作对照。控制标签关系的作用位置，优先检查各路自己的预测 |
| HH V1.1–V1.7 | V1.1 在同权重打开关系仅约 +0.005 pp。后续补标签、编辑和联合集合头几次出现专项改善，整体收益小。V1.7 为 61.224%，配对区间跨零 | 功能关系用于生成标签查询，作用在汇聚前。新模型直接输出 39 个分数 |
| HH V2.0 | 化学 CNN 56.72%，加入该实现的全肽 ESM/位置融合后 52.47%。两项同时改变 | 保留局部 CNN 与 ESM 的逐残基表示，各自读出与分类 |
| V3.0 | 原 CNN 55.20%，加法 55.09%，原始乘法 53.86%，双中心乘法 53.47% | 保留已有 pair 作为独立来源进行对照。V3.0 距离带乘法结束迭代 |
| V4.0 第 38 轮同轮对照 | A/B/C ACC 分别为 60.36% / 60.83% / 60.10%。C 的条件残基 NLL 较 B 降低约 5.6% | 表征辅助任务的收益要在功能排名上验证 |
| V4.0 最终验证 | 按 Micro-F1 选出的 A/B/C 分别为 60.9073% / 61.1010% / 60.6536% ACC。B 对 A 为 +0.194 pp，按序列簇配对的条件 95% 区间为约 [−0.69, +1.15] pp。B/C 的条件残基 NLL 为 1.3890 / 1.3042，C 净少命中 211 个已标注标签 | 背景顺序分支作为研究对照。V5.0 主实验优先分辨独立取证和真实功能关系的贡献，首轮沿用第 2 节的序列特征 |

[历史结构与消融对照](MFP-Nova-V3-Base-Architecture-Research.md) · [V4.0 设计和训练协议](MFP-Nova-V4-Architecture.md) · [V2.0 融合实验](HH_SetReread_V2_Architecture_and_Training_Plan.md#7-化学-cnn-与-esm位置融合对照)

### 1.1 分开读取的直接证据

按 row_id、标签顺序和目标对齐三份历史 Fine39 验证预测。每组均用固定 0.50 决策。强基线有 4,184 条完全匹配，2,266 条零命中。

| 历史模型 | ACC | 保住基线完全匹配 | 救回基线零命中 | 零命中子集至少一个真标签进 Top-5 |
|---|---:|---:|---:|---:|
| 最强 TheraSet | 61.05% | 4,184 | — | 1,380 |
| `no_pair` | 59.54% | 3,334 | 577 | 1,446 |
| `no_plm` | 58.67% | 3,229 | 609 | 1,397 |
| 三组概率等权平均 | 62.22% | 3,858 | 313 | 1,498 |

三个 Top-5 候选表按样本取并集可覆盖 1,789 条零命中肽，多出 409 条。并集最多可含 15 个标签，正式模型与基线仍比较同样的 Top-5。按真值为每条肽挑选三个已存在集合中最好的一个可得 71.86%。这是对已保存输出的诊断上限。固定等权平均取得 62.22%，同时改坏了 326 条基线完全匹配。V5.0 因此同时检查新增命中和原有正确预测的损失。历史 `no_pair` 和 `no_plm` 来自原 `full` 配置，当前互补结果也包含各模型独立训练及其他结构差异。 [三组历史组合依据](HH_SetReread_V2_Architecture_and_Training_Plan.md#1-设计依据)

V4.0 完成后补做固定 0.50 的历史组合诊断。在原三模型中加上 V4.0 背景组，四模型等权平均为 62.40%。若由背景组替换原强 TheraSet，三模型平均降至 61.85%。这些是验证集开发结果，背景顺序保留作可选对照。

## 2. 输入与三条证据通路

设批大小为 \(B\)，肽长为 \(L\)，Fine39 标签数为 \(C=39\)。首轮沿用原样本、长度上限 80、化学和结构字段的既有缺失掩码、双 ESM 缓存及标签顺序。输入张量与历史样本按现行缓存合同对齐。

1. **局部 CNN。** 复用 TheraSet 的 AA34、化学注记和两级 \(3,5,7,9\) CNN。第二级四份逐残基特征各自投影后供标签读取。原全局 CNN 池化特征仅供局部路径使用。
2. **ESM 上下文。** 继续用现有冻结的 ESM650M 和肽 ESM150M 残基输入，分别归一化，拼接投影后加位置表示。复用现有上下文层。标签直接读取残基上下文和可用的理化 token，理化字段继续使用独立掩码。
3. **残基对与结构。** 从 CNN 的逐残基图形成端点状态，复用现有有符号间距、距离 RBF 和可用结构质量。端点使用 CNN 特征，使这一路的残基对可以提供独立于 ESM 的判断。结构 token 与 pair 分开保留掩码。长度小于 2 时按可用结构证据评分，没有可用证据时由组合权重处理。

各分支只编码一次供自己的 39 个标签共享。证据查询、分类投影和残差状态分别学习。首轮读出宽度 256，注意力头数 8。实现中的 `shared` 和 `split` 共用同样的取证模块和参数。两组先分别形成三路标签状态，`shared` 对可用状态取均值后给三个分类头，`split` 让每个分类头只读取自己的状态。这样在相同参数与取证计算下比较融合位置。 [三路读取器源码](../src/mfp_nova/models/theraset_v5.py)

## 3. 功能矩阵如何指导残基取证

设 \(R\in\mathbb R^{L\times L}\) 代表肽内部的残基关系，\(E\in\mathbb R^{L\times C}\) 代表残基到功能的模型证据，\(G\in\mathbb R^{C\times C}\) 描述全局功能语义邻接。对于 pair 路径，证据的位置索引为 \((i,j)\)，需要逐残基诊断时分别报告两个端点的注意力边缘和。训练标注 \(Y\in\{0,1\}^{B\times C}\) 位于整条肽，逐残基证据经由肽级监督学习。

AGKphormer 用疾病的 MeSH 语义、代谢物的分子指纹以及已知关联推导的相似性，辅助预测代谢物–疾病边。V5.0 迁移的是**同侧语义辅助跨类型取证**。本项目的可观测边是肽–功能，内部需要学习残基–功能。标签邻接基于已定义的功能体系，查询每条肽时产生该样本自己的证据矩阵。 [AGKphormer](https://www.sciencedirect.com/science/article/abs/pii/S1877750325001061) · [PIAA 的局部多标签推断](https://proceedings.mlr.press/v306/wang26am.html)

### 3.1 首轮语义邻接

- 直接复用项目 Fine39 到 Major15 功能轴的已定义映射。若两个具体标签投影到同一个明确功能轴，在它们之间加入无向语义边。
- `__unknown` 表示有父功能记录而亚类未知。这些标签也独立输出。首轮置为孤立点，使用自己的查询和分类监督。
- 对有邻居的行归一化。记 \(g_c=\mathbf1[\sum_d G_{cd}>0]\)。孤立点保留基础路径。
- 已标注共现作为单独的 Train 统计对照。例如 Train 中两个 Gram 标签共同出现 8,823 次，抗 HIV/抗 HCV 只共同出现 17 次。语义相近与共同记录是两种观测。首版用语义邻接学习共享位置线索，未标注位继续按现有弱监督处理。

[Fine39 功能映射和已标注共现实现](../src/mfp_nova/multimodal.py)

### 3.2 保留本标签查询，另建关系查询

对第 \(v\) 条证据路径，让标签 \(c\) 的原查询是 \(q^v_c\)。原查询直接访问这条肽在该路径的特征。语义邻居汇聚和辅助查询定义如下。

\[
m^v_c=\sum_d G_{cd}q^v_d,
\qquad
\tilde q^v_c=q^v_c+g_c\,W^v_r[m^v_c\,\|\,(q^v_c-m^v_c)].
\]

原查询与关系查询使用同一证据张量，两次读出的状态分别记为 \(u^v_c\) 与 \(\tilde u^v_c\)。读出沿用各路径的掩码与质量权重，在池化前产生位置权重。分类器读取下式的状态。

\[
u^v_c+g_cW_o^v(\tilde u^v_c-u^v_c).
\]

\(W_o^v\) 在同一路的标签间共享并零初始化。孤立标签的关系增量为零。可在同权重下关闭增量，查看关系查询改变了哪些位置和分数。邻居查询提供家族线索，原查询与邻居的差保留目标标签身份。实际是否学出区分，按同家族候选排序和真实图对照验证。

例如抗 HIV 和抗 HCV 都属于抗病毒轴。关系查询可提示模型优先比较与抗病毒有关的残基上下文。抗 HIV 分数仍由本肽的 CNN、ESM、pair 证据各自计算，两项功能都保有自己的输出和监督。原 TheraSet 的关系层读取已经融合的标签状态。V5.0 在各路读取残基或残基对之前构造关系查询。 [旧标签上下文的位置](../src/mfp_nova/models/theraset.py#L180)

## 4. 独立评分与训练目标

每路由自己的读出状态给出 \(z^v_c\)。每条肽按有效证据路径数均匀汇总概率。设 \(m_v(x)\) 是路径可用标志。

\[
p_c(x)=
\frac{\sum_v m_v(x)\sigma(z^v_c(x))}{\sum_v m_v(x)}.
\]

CNN 对每条合法输入可用。上下文路径在存在 ESM 或可用理化字段时可用。pair/结构路径至少有一种合法证据时可用。融合先固定等权。这样可以直接检验独立取证带来的增量，后续如需学习权重，再用训练簇折外预测与固定平均比较。

训练沿用 TheraSet 的弱标签 ASL、R-Drop 和 EMA。最终概率换回数值稳定的 logit 计算 ASL。每路都以同一个整肽标注独立受监督，辅助项对可用路径取平均，共同初始权重 0.2。

\[
\mathcal L
=\mathcal L_{\mathrm{ASL}}(\operatorname{logit}(p),Y)
+0.2\,\operatorname{mean}_{v\;\mathrm{available}}\mathcal L_{\mathrm{ASL}}(z^v,Y)
+\lambda_R(t)\mathcal L_{\mathrm{RDrop}}.
\]

R-Drop 沿用原启动与爬升日程，作用于最终输出。主项和分支的负向部分继续使用原未标注弱权重与同家族亚类的进一步降权。V4.0 的残基 NLL 留在 V4.0 结果和推理诊断中，首轮 V5.0 的训练目标由分类决定。 [现有弱标签和 R-Drop 入口](../src/mfp_nova/models/theraset.py#L345)

## 5. 对照与验收

| 对照 | 证据输入 | 功能关系 | 要回答的问题 |
|---|---|---|---|
| 原 TheraSet | 现有强基线 | 关闭标签上下文，每标签一个活跃原型 | 同批按 ACC 选模的基线 |
| 共享证据的三头 | 三个读出均可访问统一的证据集合 | 各标签独立查询 | 三头的容量与独立分类损失本身能带来多少 |
| 分开证据的三头 | 每头只访问自己的证据 | 各标签独立查询 | 分开读取是否保留互补信息 |
| V5.0 语义查询 | 与上一组相同 | 增加真实功能关系查询 | 关系能否改善同家族标签的区分 |
| V5.0 打乱关系对照 | 与上一组相同 | 打乱语义边，保留各标签的邻居数和孤立点 | 增益来自真实关系还是额外查询容量 |

首批训练原 TheraSet、共享三头、分开三头。关系组与打乱关系组使用相同的关系模块、初始权重和训练协议。同无关系的分开三头共同给出因果对照。若语义邻接规模过小而难以构造有代表性的度保持随机图，报告打乱图与语义图实际重合率，并优先用相同维度的零关系或自身查询作为补充容量对照。

主实验保留 Fine39 原 Train/Validation/Test 簇划分。首批三组与后续关系组使用同一套 seed 和阈值。第一批可用 seed 42 定位效应。各组按固定 0.50 的 Validation 样本 Jaccard 选模并早停。多 seed 验证保持协议一致。Test 在全模型协议确定后作为独立评估。

报告整体 ACC、Macro-AP、Micro-F1 和独立多功能 F1。分别导出三路及最终的按行概率，计算下列指标。

- 真正新增的前五候选，包括基线零命中和置信度至少 0.90 的零命中子集。
- 两项及三项以上标注样本的 Top-5 覆盖与标准指标，独立功能轴另列。
- 同家族混淆候选之间的排序变化，按记录语义解释未标注位。
- 基线完全匹配集合的保留率，新命中、多报、漏标、净 Jaccard 和逐簇配对区间。

用同一权重分别关闭关系增量和各证据路径，可观察推理敏感性。重训对照用于衡量机制的学习收益。历史 61.05% 与 62.22% 等权组合同时保留为开发参照。V5.0 的主要检验是同批分开证据能否超过共享证据，并且真实语义关系能否进一步超过打乱关系。70% 的判断以实际训练和独立评估为准。

## 6. 实现边界

复用现有数据加载、CNN、ESM 缓存和可用性掩码。以 `train --model theraset --v5-mode` 进入五个实验配置。旧 TheraSet 和 V4.0 使用各自的既有入口。验证重点为缺结构时的输出、只有一个残基时的 pair 掩码、功能邻接与标签次序一致、孤立标签的关系增量为零，以及三路梯度均能进入对应证据路径。

三头版本的 `best_validation.npz` 同时记录最终概率、`view_probabilities`、`view_available` 和 `view_names`。语义查询与打乱图的边数和重合率出现在 `results.json` 的 `v5_graph`。同权重验证支持 `evaluate --v5-without-relation` 和 `evaluate --v5-without-view local|context|pair`，继续使用 checkpoint 保存的阈值。

本地用真实 Fine39 输入对五组分别进行两步 CPU 短训，并重载最佳 checkpoint。三头共享与分开均为 7,063,477 个训练参数，语义和打乱图均为 7,653,301。语义图与打乱图均有 34 条无向边，重合 13 条。GPU 显存及正式性能以实际运行记录为准。

## 7. 三卡 V100 启动与验收

### 首批启动

```bash
cd ~/wuzhencheng/projects/MFP-Nova
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MFP-Nova
PY="$CONDA_PREFIX/bin/python"
CPU_THREADS=6
RUN="runs/v5_fine39_s42_$(date +%Y%m%d_%H%M%S)"
mkdir -p runs
mkdir "$RUN"

launch_v5() {
  local mode="$1" gpu="$2"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$CPU_THREADS" \
    MKL_NUM_THREADS="$CPU_THREADS" PYTHONPATH="$PWD/src" \
    "$PY" -u -m mfp_nova.cli train \
    --model theraset --v5-mode "$mode" --task fine39 --seed 42 \
    --device cuda:0 --precision fp32 --batch-size 64 --accumulation-steps 1 \
    --num-workers 0 --epochs 100 --metric-suite jaccard \
    --name "${RUN#runs/}/$mode" > "$RUN/$mode.log" 2>&1 < /dev/null &
  printf '%s GPU=%s PID=%s log=%s\n' "$mode" "$gpu" "$!" "$RUN/$mode.log"
}

launch_v5 baseline 2
launch_v5 shared 3
launch_v5 split 4
```

按实际空闲卡号改最后三行。首批完成后，在保留 `RUN` 和 `launch_v5` 的终端运行语义图对照。

```bash
launch_v5 semantic 2
launch_v5 shuffled 3
```

### 查看

```bash
nvidia-smi
for log in "$RUN"/*.log; do
  mode="${log##*/}"
  mode="${mode%.log}"
  printf '\n%s\n' "$mode"
  if test -s "$RUN/$mode/results.json"; then
    echo "已完成"
  fi
  tail -n 8 "$log"
done
tail -F "$RUN"/{baseline,shared,split}.log
```

关系批次可改用 `tail -F "$RUN"/{semantic,shuffled}.log`。如已重新连接 SSH，先把 `RUN` 设为启动时打印的实际日志所在目录。

### 验收与打包

```bash
(
  set -e
  modes=(baseline shared split)
  if [[ -d "$RUN/semantic" || -d "$RUN/shuffled" ]]; then
    modes+=(semantic shuffled)
  fi
  dirs=()
  files=()
  for mode in "${modes[@]}"; do
    for file in results.json metadata.json history.jsonl best_validation.npz; do
      test -s "$RUN/$mode/$file" || { echo "等待 $mode/$file"; exit 1; }
      files+=("$mode/$file")
    done
    dirs+=("$RUN/$mode")
  done
  report="acceptance_$(date +%Y%m%d_%H%M%S)"
  mkdir "$RUN/$report"
  PYTHONPATH="$PWD/src" "$CONDA_PREFIX/bin/python" -m mfp_nova.cli summarize \
    --runs "${dirs[@]}" --split val --output "$RUN/$report/summary.json"
  pairs=(baseline:split shared:split)
  if [[ ${#modes[@]} -eq 5 ]]; then
    pairs+=(split:semantic shuffled:semantic)
  fi
  for pair in "${pairs[@]}"; do
    base="${pair%:*}"
    candidate="${pair#*:}"
    PYTHONPATH="$PWD/src" "$CONDA_PREFIX/bin/python" scripts/compare_validation_predictions.py \
      "$RUN/$base" "$RUN/$candidate" > "$RUN/$report/${base}_vs_${candidate}.json"
  done
  tar -czf "$RUN/$report.tar.gz" -C "$RUN" "${files[@]}" "$report"
  ls -lh "$RUN/$report.tar.gz"
)
```
