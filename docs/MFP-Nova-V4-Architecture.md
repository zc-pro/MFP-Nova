# MFP-Nova V4.0 架构与训练

## 研究问题

在 Fine39 原同源簇划分上，以 TheraSet 的 `no_label_context_single_prototype/micro_f1` 为主基线。历史验证 ACC 为 **61.0463%**，其中 ACC 是样本平均 Jaccard，验证阈值为 0.50。原始训练集 39,455 条，验证集 8,529 条，长度上限 80。70% 是长期目标。借助真值删除基线所有额外预测，只能把 ACC 提到 68.29%，需要把漏掉的真实标签也排上来。 [基线结果](../README.md#fine39-性能基线)

V4.0 要检验固定氨基酸组成以后，**功能条件下的残基顺序预测，能否改善 TheraSet 的标签证据表征**。模型沿用同一 TheraSet 主体及其输入，新增支路只用规范化氨基酸骨架。

## 已做的本地探针

对齐基线最佳验证概率与 Fine39 的行和标签。所有序列探针只在原 Train 拟合。叠加诊断按验证序列簇分五折，每次在四折估计每类中心及一个全局叠加系数，在另一折应用，最终沿用 0.50 概率阈值。这个步骤是基线已做验证选模后的诊断。表中分数不进入正式结果。

| 序列探针 | 独立 Macro-AP | 线性叠加后的 ACC |
|---|---:|---:|
| TheraSet 历史基线 | 57.11% | 61.046% |
| 平滑频率，左右各一残基，减去组成分数 | 32.47%（扣组成前） | 61.185% |
| 小型双侧遮蔽 CNN | 13.83% | 61.056% |
| 小型双方向自回归 CNN | 18.38% | 61.043% |
| 小型双方向固定组成自回归 CNN | 12.33% | 61.093% |

频率探针叠加后多命中 96 个标注正位，同时多报 186 个未标注位，基线已有的 4,184 条完全匹配保住 4,118 条。三个神经探针用 seed 42，在 CPU 上各训练 6 轮，batch 约 192。卷积宽度 32，核大小 3，膨胀率 1、2、4，共享左右卷积权重，输出上下文宽度 48。训练仅优化背景与功能条件下的残基交叉熵。供拟合的叠加系数为 0、0.25、0.5、1、2、4、8、16、32。

这些探针把加权分类之前的残基分布作为单独特征。正式模型需要比较残基顺序监督能否在共训时改善分类；它的强度要用相同读出、相同容量的对照衡量。

## 固定组成的残基顺序建模

对规范化序列 \(x=(x_1,\ldots,x_L)\) 计算真实残基计数 \(C_a\)。从 N 端读取时，第 \(i\) 步剩余可用个数为

\[
r_i(a)=C_a-\sum_{j<i}\mathbf 1(x_j=a).
\]

因果卷积给出 \(h_i\)。从 N 端时仅看 \(x_{<i}\)，从 C 端时用同一运算处理逆向遍历，结果映射回真实残基位置。两个方向共享卷积权重。位置表示先做因果卷积再平移一位。背景和条件头使用相同的剩余计数掩码，令 \(c=0\) 表示背景。

\[
p_c(a\mid x_{<i},C)=
\frac{r_i(a)\exp(s_{i,c,a})}{\sum_b r_i(b)\exp(s_{i,c,b})}.
\]

实现中对 \(\log r_i(a)+s_{i,c,a}\) 做 FP32 `log_softmax`，只让 \(r_i(a)>0\) 的残基参与。Padding 在归一化前使用有限占位分数，随后排除在损失和汇聚之外。只剩一种类型时背景和条件概率均为 1，证据为零。Fine39 原 Train 和 Validation 的规范化骨架均只包含 20 种标准残基，其中分别有 79 条与 23 条为单一残基类型的重复序列，它们的新增顺序证据应全为零。计数先验本身在各分数相等时给同一组成的所有不同排列同样概率。对 `AAKL` 的 12 个不同排列做了数值枚举，背景和条件概率各自求和均为 1（浮点精度内）。

背景分数 \(s_{i,0}=W_0 h_i+b_0\)。功能条件分数在背景上加低秩项。首版宽度可沿用探针尺度，令 \(R=16\)，\(q_c\) 为标签向量，\(U\) 为共享的残基输出矩阵。

\[
s_{i,c,a}=s_{i,0,a}+\sum_{k=1}^R U_{a,k}(W_hh_i)_kq_{c,k}.
\]

因此，同一实际残基在两个条件下的证据为 \(d_{i,c}=\log p_c(x_i)-\log p_0(x_i)\)。沿一个真实遍历方向求和，等于该方向归一化模型在固定组成下的序列对数似然比。以真实位置对齐两个方向后取平均，得到供分类使用的证据。正式解码只需对已知序列并行评分。反向遍历共享同一真实肽和标注。

固定组成的含义可以用概率分解写清楚。

\[
\log\frac{P(x\mid Y_c=1)}{P(x)}
=
\log\frac{P(C\mid Y_c=1)}{P(C)}
+
\log\frac{P(x\mid C,Y_c=1)}{P(x\mid C)}.
\]

新增支路针对最后一项。TheraSet 主体仍可从原始序列学习组成和整体证据。此处 \(Y_c=1\) 指数据中有标签记录。公式给出总体分布的关系，有限样本中的神经分数由训练与验证确定。

## 接入 TheraSet

1. 使用当前 batch 的 `aa` 和 `mask`。氨基酸分支消费当前缓存使用的规范化骨架。原 TheraSet 的两个 ESM 缓存沿用原协议。它们的逐残基编码可见全序列，避免把它们引入因果残基预测支路。
2. 汇聚每类的正证据、负证据各自的 masked mean 和 masked max，形成每样本每标签四维特征。符号分开保留，防止相反位置证据在求平均时抵消。各标签独立读取，同一残基允许支持多个功能。
3. 将共享线性投影写入原单原型最后一层标签状态，在 `output_norm` 之前融合。写入权重零初始化。原标签状态、监督的 CNN 直接头和最终分类通路继续提供强基线。分类和残基任务同时训练。最终仍输出 39 个 sigmoid 分数。
4. 保留原弱标签分类目标与 TheraSet 的直接头、证据辅助头、R-Drop、EMA。在原分类损失上加 \(\lambda_0\mathcal L_0+\lambda_c\mathcal L_c\)。\(\mathcal L_0\) 是所有真实训练序列的背景残基 NLL。\(\mathcal L_c\) 仅在 \(Y_c=1\) 的样本×标签对上计算，以有效残基数归一化，并用训练标签频次倒平方根归一化加权。优先固定 \(\lambda_0=\lambda_c=0.01\) 做第一轮，记录两项加权损失与分类项的量级。
5. 背景头的输出直接由背景 NLL 学习。在条件 NLL 和差分读出中对背景分数用停止梯度。分类梯度通过条件项和共享因果上下文回传。R-Drop 的两个 TheraSet 视图共享同一次确定性的氨基酸支路前向。两项验证残基 NLL 按各自分母累计，记录在 `history.jsonl` 和 `results.json` 的 `validation_order_nll`，与模型选取用的分类指标分别报告。

FP32、batch 64、长度 80 时，每方向的完整 \([B,L,39,20]\) 残基 logit 张量约 15.2 MiB。内存实测须包含梯度、临时张量和 R-Drop。训练和推理按两个遍历方向、39 个标签并行评分。条件残基损失的分子和分母均按已标注正位计算。

## 确认结构贡献

| 运行 | 训练内容 | 回答的问题 |
|---|---|---|
| A | 原 TheraSet，关闭标签上下文，每类一个活跃原型 | 匹配训练条件下的主基线 |
| B | TheraSet 加同样的固定组成支路，保留背景 NLL，\(\lambda_c=0\) | 新增容量、显式顺序建模和背景残基监督的收益 |
| C | 与 B 相同，引入已标注正例的功能条件残基 NLL | 功能条件生成监督提供的独立增量 |

B 的条件参数仍可通过分类目标学习，B/C 的推理结构和参数量相同。A/B/C 从头训练，同 seed、划分、输入、训练与选模协议。首轮统一复用历史构建已验证的 FP32、batch 64、累积 1、固定 0.50 阈值与 `micro_f1` 选模设置。历史 61.05% 以及同批 A 同时报告。项目现有 Jaccard 主目标和漏标排序诊断照常复算。

主报告包含样本 Jaccard、Macro-AP、Micro-F1、独立多功能 F1、二标签及三标签以上的 Top-5 覆盖、基线完全匹配保留、仅漏标/仅多报/两者兼有三组的变化，以及相同组成但排列不同的真实肽。使用按序列簇配对区间辅助判读。正式多 seed 采用共同的 seed 列表与相同选模口径，最终 Test 保留作独立评估。

构建时检查因果状态、归一化、padding 和权重兼容性。改动某一方向尚未读到的位置时，相应上下文状态应一致。小多重集枚举的概率和应为 1。改变 batch padding 应保持同一肽的分数。只剩一种可用残基时差分应为零，梯度应有限。零写入下加载同一 TheraSet 权重应复现原输出。

本地 CPU 已用真实 Fine39 数据及双 ESM 缓存完成 A/B/C 的两步短训练，并重载各自的 EMA 权重。A 的训练参数为 10,404,206，B/C 均为 10,419,506，额外 15,300 个参数。B/C 以同结构、同初始权重开始，条件损失产生了不同的参数更新。V100 和 RTX 5070 的显存与正式性能待实测。

## 首轮训练命令

模型入口是 `train --model theraset --v4-mode`。A、B、C 对应 `baseline`、`background`、`conditional`。命令从已同步本次源码的项目根目录运行。使用原 ESM650M 和肽 ESM150M 的 Train/Val 缓存。

### V100 三卡后台

```bash
cd ~/wuzhencheng/projects/MFP-Nova

(
  set -e
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate MFP-Nova
  GPUS=(1 2 3)
  CPU_THREADS=6
  MFP_PYTHON="$CONDA_PREFIX/bin/python"
  "$MFP_PYTHON" -c 'import torch; print(torch.__version__)'
  MFP_RUN="v4_fine39_s42_$(date +%Y%m%d_%H%M%S)"
  mkdir -p runs
  mkdir "runs/$MFP_RUN"
  modes=(baseline background conditional)

  for i in "${!modes[@]}"; do
    mode="${modes[$i]}"
    log="$PWD/runs/$MFP_RUN/$mode.log"
    nohup env CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
      OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" \
      PYTHONPATH="$PWD/src" "$MFP_PYTHON" -u -m mfp_nova.cli train \
      --model theraset --v4-mode "$mode" --task fine39 --seed 42 \
      --device cuda:0 --precision fp32 --batch-size 64 --accumulation-steps 1 \
      --num-workers 0 --epochs 100 --metric-suite micro_f1 \
      --name "$MFP_RUN/$mode" > "$log" 2>&1 < /dev/null &
    printf '%s  GPU=%s  PID=%s  log=%s\n' "$mode" "${GPUS[$i]}" "$!" "$log"
  done
)
```

将 `GPUS` 改为当前三张空闲物理卡号。每个进程只看到一张设备，CLI 中都使用 `cuda:0`。

### RTX 5070 PowerShell 顺序

```powershell
conda activate MFP-Nova
$MfpPython = Join-Path $env:CONDA_PREFIX 'python.exe'
$env:PYTHONPATH = Join-Path $PWD.Path 'src'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '6'
$env:MKL_NUM_THREADS = '6'
$MfpRun = 'v4_fine39_s42_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$RunDir = Join-Path 'runs' $MfpRun
New-Item -ItemType Directory -Path $RunDir -ErrorAction Stop | Out-Null
$Common = @('--model','theraset','--task','fine39','--seed','42',
    '--device','cuda:0','--precision','fp32','--batch-size','64','--accumulation-steps','1',
    '--num-workers','0','--epochs','100','--metric-suite','micro_f1')
foreach ($Mode in 'baseline','background','conditional') {
    $Log = Join-Path $RunDir "$Mode.log"
    & $MfpPython -u -m mfp_nova.cli train @Common --v4-mode $Mode --name "$MfpRun/$Mode" 2>&1 |
        Tee-Object -FilePath $Log
    if ($LASTEXITCODE -ne 0) { throw "V4 $Mode failed. Log $Log" }
}
```

训练完成后比较三个子目录中的 `results.json` 与 `best_validation.npz`。每个配置的启动日志位于同一批实验目录。按 `row_id` 对齐后复核配对样本 Jaccard、额外标签及漏标、Top-5 覆盖和完全匹配保留。

## 文献关系

- PIAA 提供了按局部片段产生标签证据再汇聚的参考。V4.0 的局部量使用真实残基在两种条件下的相对预测概率。 [PIAA](https://proceedings.mlr.press/v306/wang26am.html)
- Salazar 等用逐位置遮蔽计算伪对数似然。V4.0 将单方向定义为显式链式条件分布，使同一组成内的不重复排列具有归一化概率。 [Masked Language Model Scoring](https://aclanthology.org/2020.acl-main.240/)
- NLP 词序恢复已经研究给定词项多重集的条件生成。ProGen 已展示基于功能标签的生物序列条件化。Ren 等研究了用生成模型的比值消减背景统计。V4.0 的项目创新候选是把这三者落实为短肽的逐残基标签证据，在强多标签基线上检验条件监督的增量。 [词序恢复](https://aclanthology.org/2022.coling-1.567/) · [ProGen](https://doi.org/10.1038/s41587-022-01618-2) · [背景似然比](https://proceedings.neurips.cc/paper/2019/hash/1e79596878b2320cac26dd792a6c51c9-Abstract.html)
