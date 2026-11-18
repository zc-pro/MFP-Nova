# XBridge-MFP

## 研究动机

2026-11-18 的 Fine39 科研审计中，TheraSet 的固定 0.50 Validation Jaccard 为 61.2704%，8,529 条里有 2,351 条零命中。距最近 Train 序列的全局编辑相似度低于 0.70 的 3,753 条样本贡献了约 55% 的总 Jaccard 缺口。审计中至少两轴和三轴的样本分别有 47.20% 和 27.64% 恢复了对应数量的已记录功能轴。XBridge 优先检验功能证据能否跨簇迁移，以及证据能否改善正确标签的排序。

## 实现与设计修订

1. **功能证据。** 保留冻结 ESM-650M 的残基信息，将其与 Peptide ESM-150M 及逐残基化学词嵌入融合。每个 Fine39 标签用自己的查询在有效残基上读出证据，输出 39 个独立 logit。缓存模式用于低成本对照。在线模式在 Peptide ESM 最后四层 Q/V 上启用 rank-8 LoRA，编码器其余参数固定。CLS、EOS 和 PAD 由同 revision 的 tokenizer 定义，读取证据时只使用有效残基。
2. **跨簇桥接。** 每个训练批可加入一个额外正例对，先选标签，再选两个不同簇，最后在簇内选行。全部原训练行仍遍历一次。损失在 batch 内同标签、不同簇的已记录正例对上计算 `max(0, 0.50 - cosine)²`，先对标签内正例对平均，再对参与标签平均。权重设为 0 时仍用同一配对采样，形成等曝光对照。独立分类监督、证据标准差和 margin 处于活跃区间的配对比例一起记录。正例桥接单项允许常量解，且同一功能可能有不同机制。
3. **逐轴覆盖。** 复用项目独立轴投影和泛抗微生物抑制规则，在每个已记录有效轴内对真实 Fine39 阳性 logits 做温度归一化的 log-mean-exp，再计算正例 softplus。逐样本均分各轴梯度。此项可能推动已经命中的功能分数继续增加，因此保留 `--xbridge-axis-weight 0.015` 供消融，默认权重为 0。基础监督复用项目 ASL，零位按未标注位处理。
4. **折外残差。** 单独训练和评估证据头后，可接入冻结 TheraSet。训练用三折簇折外 logits，推理用同一训练范围的全量参考，最终 `Z = U + R`。残差末层初值为零，参考固定为 eval 并以 FP32 前向。首步主损失更新残差末层，独立分类头为证据与 LoRA 提供梯度。推理使用已保存的参考权重和适配参数。折外与全量参考的置信分布仍可能不同。

模型在 `src/mfp_nova/models/xbridge.py`，采样在 `src/mfp_nova/xbridge_data.py`。训练和评估使用共有的 `mfp-nova` 入口。默认按固定 0.50 Jaccard 选模，附带 Macro-AP、样本 Precision/Coverage、独立轴 MFP-F1、零命中数、Top-1/Top-5 命中数和轴恢复数。残差运行还输出相同样本上的参考、辅助头及新增和丢失标签统计。

## 已完成的验证

- 在线模式使用真实 Peptide ESM checkpoint 和 Fine39 缓存，完成 24 条训练样本、24 条 Train 内留出样本的两轮 CPU FP32 探测，并从 `last.pt` 续训一轮。检查长度 2、10、22、74 的真实样本，LoRA 置零后对 FP16 缓存的有效残基平均绝对误差为 `2.07e-5`，最大为 `9.43e-4`。LoRA B 梯度非零，训练后保存的 B 权重也发生变化。检查点记录 LoRA 和证据头，冻结 ESM 从配置中的固定 revision 加载。
- 三个 Train 内嵌套簇折各跑一轮 TheraSet，导出折外 logits，再用独立的同范围参考跑完两轮 XBridge 残差。在线和残差检查点均已重载并验证无标签 NPZ 推理，重算分数与导出分数一致。单元测试覆盖残差初始等价、冻结参考、首步主／辅助梯度、padding、空配对和覆盖轴语义。短跑所用参考权重为诊断权重，历史 V100 参考归档提供了预测与元数据。
- Seed 42 的同预算消融各使用 768 条 Train 样本、768 条 Train 内簇留出样本（618 个簇）、6 轮和同一配对采样。各组第 5 轮的固定 0.50 指标如下。MFP-F1 指真实多轴样本上的独立轴功能预测 F1。

| 配置 | Jaccard | Precision | Coverage | MFP-F1 | 零命中 | 恢复 ≥2 轴 `/211` | 恢复 ≥3 轴 `/57` |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASL，保留配对采样 | 24.51% | 25.95% | 67.57% | 54.27% | 202 | 107 | 11 |
| ASL + 桥接 | 24.81% | 26.29% | 67.27% | 53.46% | 202 | 103 | 9 |
| ASL + 逐轴覆盖 | 21.76% | 22.62% | 76.19% | 56.06% | 139 | 137 | 16 |
| ASL + 桥接 + 逐轴覆盖 | 21.86% | 22.73% | 76.44% | 56.03% | 139 | 140 | 16 |

桥接单项 Jaccard 差为 **+0.29 个百分点**，在这批样本上按簇配对重采样的描述区间为 −0.17 到 +0.76 个百分点。它未减少零命中，恢复两轴、三轴的条数略降。桥接与逐轴覆盖同时启用时，相对仅桥接增加了 119 个已记录命中位和 1,051 个未标注预测位。当前权重更适合保留为覆盖率消融。以上结果来自少量、按诊断规则抽样的内部样本，测量的是单次运行，正式 Fine39 指标和两台 GPU 的在线吞吐仍待测。

运行记录见 `runs/xbridge_asl_control_20261118_100737`、`runs/xbridge_axis_control_20261118_100403`、`runs/xbridge_cached_pair_ablation`、`runs/xbridge_cached_pair_probe`、`runs/xbridge_lora_smoke` 和 `runs/xbridge_oof_smoke_20261118_100538`。`best_validation.npz` 含原始 `row_id`、标签和簇 ID。内部划分的导出 `split` 为 `train_inner`。

## 运行与下一步实验

先在 `--xbridge-inner-fold 0` 下固定 seed、样本、阈值和训练预算，比较缓存与在线 LoRA。每种表征同时跑默认桥接权重和 `--xbridge-bridge-weight 0`，用独立证据头的 Top-1/Top-5、零命中及 Jaccard 判断增量。以下命令在真实数据上跑两轮短训练，检查单卡资源和数值。正式内部对照移除 `--limit-samples 128 --max-steps 16 --epochs 2`，保持两组相同配置。缓存对照省略 `--xbridge-online`。

**RTX 5070 · PowerShell**

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$run = "xbridge_inner0_s42_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Path "runs/$run" | Out-Null
mfp-nova train --model xbridge --xbridge-online --xbridge-inner-fold 0 --seed 42 `
  --device cuda:0 --precision fp16 --batch-size 8 --accumulation-steps 8 `
  --limit-samples 128 --max-steps 16 --epochs 2 --name "$run/online" 2>&1 |
  Tee-Object -FilePath "runs/$run/online.log"
```

**V100 · Ubuntu Bash**

```bash
run="xbridge_inner0_s42_$(date +%Y%m%d_%H%M%S)"
mkdir -p "runs/$run"
nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  mfp-nova train --model xbridge --xbridge-online --xbridge-inner-fold 0 --seed 42 \
  --device cuda:0 --precision fp16 --batch-size 8 --accumulation-steps 8 \
  --limit-samples 128 --max-steps 16 --epochs 2 --name "$run/online" \
  >"runs/$run/online.log" 2>&1 </dev/null &
echo "PID=$!  runs/$run/online.log"
```

证据头具备互补排序后，再用 `--model theraset --task fine39 --hh-oof-fold 0/1/2`（每折单独启动）生成三个 `oof_logits.npz`。全训练范围的参考使用同范围 TheraSet V5 baseline 的 `best.pt`。内部对照时，三折教师、参考和 XBridge 都带同一个 `--xbridge-inner-fold 0`。给 XBridge 同时传入 `--xbridge-reference <best.pt>` 和 `--xbridge-oof <fold0.npz> <fold1.npz> <fold2.npz>`。折内训练采用固定轮数，预测折只用于导出分数。对相同留出行比较参考、辅助头和残差总分，优先看低相似度、零命中净恢复和多轴恢复的变化。

转入原始 Fine39 协议时从所有训练命令去掉 `--xbridge-inner-fold`，使用原 Train 范围的折外参考文件。Validation 选模，Test 用于最终评价。`mfp-nova evaluate --checkpoint <best.pt> --split val --output <validation.json> --predictions <validation.npz>` 导出带逐行信息的验证结果。续训用 `--model xbridge --resume <last.pt>`，复用保存的训练范围和 OOF 输入。预测入口 `mfp-nova predict --checkpoint <best.pt>` 支持项目已有的序列或预处理 NPZ 输入。
