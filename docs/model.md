# 初始序列方案与评价协议

## 依据与边界

参考根目录技术蓝图中 S2/S4 的正向证据：冻结上下文编码器、全局/局部特征互补、ASL 适合长尾多标签、Validation 选模与阈值校准。没有继承旧源码、47 标签、特征、权重或实验指标。

新数据默认 fine39：明确骨架 Train 39,455、Val 8,529、Test 8,180。全监督 Train 标签正例数 57–10,798，长尾明显；序列短，最长 74。major15 可用同一框架独立训练，但不同标签粒度、监督队列和数值不能直接当模型优劣排名。

初版选冻结 `facebook/esm2_t12_35M_UR50D`，12 层、输出维度 480，来自 [官方 ESM 模型表](https://github.com/facebookresearch/esm/blob/main/README.md) 和 [模型配置](https://huggingface.co/facebook/esm2_t12_35M_UR50D/blob/main/config.json)。35M 用于控制双节点特征提取成本，并非已证明保持旧 650M 效果。

## 计算

去除 BOS/EOS/PAD 的冻结末层残基状态为 R，有效残基 mask 为 M。

```text
R [B,L,480]
→ Linear(480,128) → affine LayerNorm → GELU → 将无效位置置零，得到 X
├─ 按 M 做 mean → global [B,128]
└─ Conv1d kernels 3/5，每尺度64 channels → GELU → masked max
   → local [B,128]
concat(global, local) [B,256] → Dropout(0.1) → Linear(256,C) → logits
```

紧凑共享投影在卷积之前，减少训练参数；全局/局部拼接采用直接分类，不引入旧模型局部尺度参数。fine39 训练网络 137,511 参数，major15 为 131,343，不含冻结 ESM。全局 mean 使用投影后的残基，属于新的计算定义；不能称为旧 S4 等价复现。卷积前将所有 padding 清零，max 排除 padding；混合长度 batch 不改变短序列的有效特征。

ASL 令 p=sigmoid(z)，n=min(1,1-p+clip)：

```text
base = y*log(p) + (1-y)*log(n)
weight = y*(1-p)^gamma_pos + (1-y)*(1-n)^gamma_neg
loss = -mean(base * stopgrad(weight))
```

正项用稳定 logsigmoid，负项 log 下限 1e-8。首版 γ+=0、γ−=2、clip=0.03，是较温和的负样本抑制起点；不认为旧 γ−=4、clip=0.05 必然适用。单独 BCE 对照由配置选择，不叠加额外损失。

## 初始参数

| 参数 | 值 | 依据 |
|---|---|---|
| token 上限 | 80（含特殊 token） | 最长74；不静默截断 |
| ESM 提取 batch | 32 | 短肽、小编码器的保守起点；GPU 尚待测量 |
| 缓存 | FP32 提取，FP16 保存 | 约百万残基时480维缓存约1 GB；数值/性能差异需实验 |
| 投影/卷积 | 128 / 3,5 / 每尺度64 | 短肽与紧凑可训练网络 |
| Dropout | 0.1 | 小型网络的适度正则化，尚未单因素验证 |
| AdamW | lr=3e-4, wd=1e-4 | 独立初始学习率，非最优结论 |
| head batch | 128 | 小模型 FP32；实际吞吐待 GPU 检查 |
| 预算 | 60 epochs, patience=10 | 开发期有界训练，以 Val 选择；正式比较统一预算 |
| seed | 42 | 日常单次开发；正式比较另用固定多 seed 报均值/样本标准差 |

默认训练 FP32，不启用 V100 不支持的 BF16。固定 seed 初始化与 DataLoader shuffle；不保证跨 GPU、跨 PyTorch 完全逐位一致。Windows 默认 workers=0。

## 指标与决策

- AP 使用 sklearn `average_precision_score` 的非插值阶梯定义；同分数统一聚合，不按旧项目样本索引逐位排序。这里用 Macro-AP/Micro-AP 表示通常所称 AUPRC，不是梯形 PR 积分。
- 标签没有正例时 AP=null，从 Macro-AP 排除，记录有效标签数量；Micro-AP 展平所有列。没有任何正例时 AP=null，Val 无法进行选模则明确报错。
- Macro-F1 对全部任务标签等权平均，Precision/Recall/F1/Jaccard 的空分母取 0，与旧 Macro-F1 空分母取 1 的实现不同。
- 模型选择按 Val Macro-AP、Micro-AP、更早 epoch。Test 只在独立评价命令中使用固定 checkpoint 和阈值。
- 最佳 checkpoint 在完整 Validation 拟合唯一全局阈值，网格 0.05–0.95、步长0.01，最大化 sample Jaccard；同分优先接近0.5，再较小阈值。比较概率采用 >=。
- 校准后的 Val F1/Jaccard 是拟合内指标；独立 Test 才用于最终评价，不宣称这些 Val 阈值指标是交叉拟合泛化结果。

原数据中的 D/HELM/PTM、结构和理化特征已保留，当前模型不消费。D/L 或修饰不同而骨架相同的样本上下文特征一致，并受已有簇隔离约束；本模型不能据此区分完整化学实体。未知0不能视为完备阴性；CNN activation 也不能直接称为已验证 motif。

下一步先在 GPU 验证提取/存储精度、显存和单 seed 基线；再在同队列/同预算下比较全局与全局局部、ASL 与 BCE。构型注记或可靠结构融合有研究动机，但当前未实现或验证其增益。
