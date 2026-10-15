# MFP-Nova 真实数据验证与 GPU 训练准备实际报告

日期：2026-10-15。状态：服务器执行代码已准备；两台服务器的实测结果尚待回传。不能将本报告解释为 GPU 验证已通过。

## 1. 实际环境与缓存

- 本机：RTX 3050 Laptop，4096 MiB，检查时使用约 867 MiB、利用率约 7%，未发现 Python 训练进程。项目环境为 PyTorch 2.8.0+cpu，CUDA 不可用。
- 用户确认真实 ESM-35M/650M 缓存仅位于 RTX 5070 服务器；本机没有可读取的 MFP 缓存或共享路径。35M/650M 的预期残基维度分别为 480/1280，服务器缓存的实际维度、元数据与逐行对齐尚未远程复核。
- RTX：项目约定路径 `D:\Data\ChatGPT\mfp\MFP-Nova`，Conda `MFP-Nova`，原方案 Torch 2.8.0+cu128。本机与 RTX 不建立 SSH，更新通过 Git，执行在 UU 远程终端完成。
- V100：项目约定路径 `~/wuzhencheng/projects/MFP-Nova`，Conda `MFP-Nova`，保留 Torch 2.4.1+cu121。当前非交互 SSH 未通过密码认证，尚未取得实际 GPU 占用或缓存列表；不安装公钥、不改服务器环境。
- 按用户最新要求，不在本机训练或生成替代缓存；不使用随机特征冒充真实 ESM，不重新生成大规模特征。

## 2. 三模型实际状态

| 模型 | major15 / fine39 下游参数（前轮本地完整规格） | RTX GPU 检查 | V100 GPU 检查 | 本轮真实 Loss / 显存 / 吞吐 |
|---|---:|---|---|---|
| TriRelay | 35,196,329 / 35,209,345 | 尚未执行 | 尚未执行 | 不可报告 |
| GeoFactor | 30,998,706 / 31,011,722 | 尚未执行 | 尚未执行 | 不可报告 |
| ProtoRoute | 32,327,070 / 32,451,062 | 尚未执行 | 尚未执行 | 不可报告 |

上表参数条件为隐藏维 256、8 heads、650M 残基输入，使用本地原始 Train 化学词表，不含冻结 ESM；不是本轮服务器资源测量。

## 3. 本轮发现和修复

1. 原 `limit_samples` 以子集重算标签先验，可能使高阶经验因子消失、原型容量全部缩到 1。已改为保留完整原始 Train 统计，只限制优化样本，保持完整模型机制与统一长尾分层。
2. 原前 N 行诊断不保证包含 D/无结构/长肽。已增加代表样本选择，并将所选真实 row_id 正确映射回 ESM 缓存位置；不重排缓存、不改变划分。
3. 原训练只有 FP32，缺少有界 optimizer step 诊断。已接入 autocast、FP16 GradScaler、梯度累积和 max_steps，保存恢复 scaler/global step，记录 AMP 跳过更新。FP16 分类 Loss 在 FP32 计算。
4. 新增 `gpu-check`：复用真实缓存检查、完整模型资源测量、关键模块输出与梯度更新检查，接着调用共享 trainer 做短训练、Validation、checkpoint 和恢复；不启动 Test 或多 seed。

## 4. 已执行的相关检查

- CPU 定向回归：代表样本与 ESM 缓存位置一致；累积梯度和有界更新次数；完整 Train 统计保留；checkpoint/scaler/global step 保存恢复；既有 TriRelay 训练/推理流程。使用测试 fixture，仅说明代码链路，不是真实数据训练或 GPU 结论。
- 现有数据只读，未修改迁移包、划分、真实缓存或历史结果。统一评价仍调用同一 metrics 实现；Val Macro-AP → Micro-AP → 更早 epoch，阈值仅 Val 拟合。
- major15/fine39 的真实服务器链路均为待执行。实际数值稳定性、混合精度、GPU 显存、速度、Loss 变化不能提前给出。

## 5. 两台服务器运行方式

先确认项目没有运行中的训练，再更新源码并激活已有环境，执行 `python -m pip install --no-deps -e .`。不升级服务器 Torch/CUDA。

只读检查：

```bash
nvidia-smi
mfp-nova gpu-check --esm 650m --device cuda:0 --inspect-only
```

RTX 5070：

```bash
mfp-nova gpu-check --esm 650m --device cuda:0 --tasks major15 fine39 --precisions fp32 fp16 --batch-sizes 2 4 8 16 --steps 32 --samples 128 --output runs/gpucheck_rtx5070_20261015
```

V100：

```bash
mfp-nova gpu-check --esm 650m --device cuda:0 --tasks major15 fine39 --precisions fp32 fp16 --batch-sizes 2 4 8 16 --steps 32 --samples 128 --output runs/gpucheck_v100_20261015
```

只选择已分配且空闲的设备；如不是 0，改 device。V100 不启用 BF16。缺少迁移包或缓存时先传输当前项目的现有文件，或指定 data-root/feature-root，不以旧项目缓存替代。

脚本按 TriRelay → GeoFactor → ProtoRoute，分别完成 major15/fine39。记录参数、allocated/reserved 峰值、前向/反向/optimizer/总 step 时间、吞吐和粗略 epoch 时间，检查 padding、缺结构及关键模块更新；只将成功配置用于 32 step 短训练并恢复 2 step。结果写入新目录，失败明确报告，不覆盖历史、不删核心模块。

## 6. 正式训练准备

初始资源探测从 batch 2 起，逐步至 16，比较 FP32/FP16；这些是待测批量，不是已验证推荐值。OOM 优先减 batch 或配合梯度累积，不削减架构。正式 batch/精度以每台服务器实测结果为依据。

```bash
mfp-nova train --model trirelay --task major15 --seed 42 --device cuda:0 --precision fp16 --batch-size 8 --name trirelay_major15_esm650m_seed42
mfp-nova train --model geofactor --task major15 --seed 42 --device cuda:0 --precision fp16 --batch-size 8 --name geofactor_major15_esm650m_seed42
mfp-nova train --model protoroute --task major15 --seed 42 --device cuda:0 --precision fp16 --batch-size 8 --name protoroute_major15_esm650m_seed42
```

上例 batch/精度需在 gpu-check 通过后确认；fine39 替换 task 并使用独立 run。当前不启动完整正式训练。两台服务器运行后，使用各自 `summary.json`、`*_profile.json` 和短训练 `history.jsonl` 回填本报告，才能形成最终实测结论。
