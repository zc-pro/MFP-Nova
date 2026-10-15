# MFP-Nova 真实数据验证与 GPU 训练准备实际报告

日期：2026-10-15。状态：RTX 5070 已完成三模型 × 两任务的 ESM-650M、seed 42 全量训练，六个正式 results.json 已回传并归入本机 runs/，详见第 7 节。下一步优先执行同协议的五 Seed 比较，35M 对照后置。本机承担开发与结果整理，不启动本机 GPU 训练。按用户最新要求，暂不继续处理 V100；下文保留其已取得的历史检查结果。

## 1. 实际环境与缓存

- 本机：RTX 3050 Laptop，4096 MiB，检查时使用约 867 MiB、利用率约 7%，未发现 Python 训练进程。项目环境为 PyTorch 2.8.0+cpu，CUDA 不可用。
- 真实缓存最初仅在 RTX 5070，随后用户通过 UU 传入本机 `features/esm2_650m` 和 `features/esm2_35m`，两者均有 Train/Val 的元数据及六个预期缓存文件；35M 元数据为 480 维（并非用户口述的 15M）。GPU 实测仅运行 650M：Train 39,478 行、Val 8,534 行，残基维度 1280，max_length 80，float16 存储，数据标识 `2026-10-14`。六个组合通过现有加载器的样本 ID、序列及缓存对应检查；未在本机启动训练或重新生成特征。
- RTX：项目路径 `D:\Data\ChatGPT\mfp\MFP-Nova`，Conda `MFP-Nova`，实测 Torch 2.8.0+cu128、CUDA 12.8 可用，RTX 5070 总显存 12,227 MiB。检查开始时占用 1,762 MiB、利用率 1%，列表为桌面应用，未显示 Python 训练进程；此占用不是模型峰值显存。本机与 RTX 不建立 SSH，更新通过 Git，执行在 UU 远程终端完成。
- V100：项目路径 `~/wuzhencheng/projects/MFP-Nova`，Conda `MFP-Nova`，实测 Torch 2.4.1+cu121、CUDA 12.1 可用；GPU 0 为 Tesla V100-PCIE-16GB-LS，总显存 16,384 MiB，最新检查时占用 5 MiB、利用率 0%，无其他计算进程。用户上传后，默认 `features/esm2_650m` 和 `features/esm2_35m` 均成功读到 Train 39,478 / Val 8,534 行的元数据，维度分别为 1280 / 480，数据标识与 resolved_revision 均与 RTX 记录一致。inspect-only 本身不加载数组与多模态数据包；后续 650M 完整 GPU 检查已运行结束，见下文；不改服务器环境。
- 按用户最新要求，不在本机训练或生成替代缓存；不使用随机特征冒充真实 ESM，不重新生成大规模特征。

## 2. 三模型实际状态

| 模型 | major15 / fine39 下游参数（前轮本地完整规格） | RTX GPU 检查 | V100 GPU 检查 | 本轮真实 Loss / 显存 / 吞吐 |
|---|---:|---|---|---|
| TriRelay | 35,196,329 / 35,209,345 | 两任务均运行结束 | 两任务均运行结束 | 短训练及资源见下文；RTX 正式结果见第 7 节 |
| GeoFactor | 30,998,706 / 31,011,722 | 两任务均运行结束 | 两任务均运行结束 | 短训练及资源见下文；RTX 正式结果见第 7 节 |
| ProtoRoute | 32,327,070 / 32,451,062 | 两任务均运行结束 | 两任务均运行结束 | 短训练及资源见下文；RTX 正式结果见第 7 节 |

上表参数条件为隐藏维 256、8 heads、650M 残基输入，使用本地原始 Train 化学词表，不含冻结 ESM；不是本轮服务器资源测量。

用户实际执行 `gpu-check --esm 650m --device cuda:0 --models trirelay geofactor protoroute --tasks major15 fine39 --batch-sizes 2 4 8 16 --precisions fp32 fp16 --steps 32 --samples 128`。各组合显示选取 128 个真实 Train 样本，日志如下：

| 模型 | 任务 | 首条 epoch Loss | 末条 epoch Loss | 日志中的最高诊断 Val Macro-AP |
|---|---|---:|---:|---:|
| TriRelay | major15 | 0.382950 | 0.245595 | 0.169190 |
| TriRelay | fine39 | 0.328621 | 0.202533 | 0.099086 |
| GeoFactor | major15 | 1.594887 | 0.331794 | 0.117402 |
| GeoFactor | fine39 | 1.761826 | 0.323814 | 0.102476 |
| ProtoRoute | major15 | 0.471770 | 0.269284 | 0.169643 |
| ProtoRoute | fine39 | 0.415544 | 0.241604 | 0.124520 |

这些是首末日志值，包含脚本的短训练及恢复阶段，不是完整数据训练曲线。不同模型有不同辅助 Loss，绝对 Loss 不用于横向排名。小样本 Val AP 只用于诊断，不能与此前全量基线比较或据此选择模型。

按当前脚本的顺序和失败行为，六组合全部结束说明每个组合至少有一项完整模型资源配置通过关键参数梯度/更新检查，并完成共享 trainer 的短训练、checkpoint 恢复及独立 Val 读取。用户随后从 `summary.json` 导出以下测量值，六组合均选中 batch 16、FP32：

| 模型 | 任务 | 峰值 allocated MiB | 平均 step ms | samples/s |
|---|---|---:|---:|---:|
| TriRelay | major15 | 2032.8 | 183.1 | 87.4 |
| TriRelay | fine39 | 2106.8 | 170.0 | 94.1 |
| GeoFactor | major15 | 2122.9 | 163.8 | 97.7 |
| GeoFactor | fine39 | 2168.0 | 174.8 | 91.5 |
| ProtoRoute | major15 | 1872.3 | 157.0 | 101.9 |
| ProtoRoute | fine39 | 1924.2 | 155.7 | 102.7 |

这是代表性真实 batch 的短测；吞吐最高与最低相差约 17.5%，不足以据此评价模型性能。峰值是 PyTorch allocated 显存，不含桌面应用、部分 CUDA 上下文及 allocator 的未用 reserved 空间，不等于整卡占用。ESM-650M 已预计算，本次训练完整下游模型，不在线训练 650M 主干。按约 3.95 万 Train 样本和上述吞吐线性外推，训练部分约 6.4–7.5 分钟/epoch；这不是正式全量实测，不含完整 Val、初始化和保存等耗时，实际会受长度与数据读取影响。

脚本在检查通过的配置中按 samples/s 选择配置。FP32 被选中不能区分 FP16 是更慢还是未通过梯度/更新检查；也不能由当前表格判断所有 batch/精度是否通过。服务器实际参数量、reserved 峰值、前向/反向拆分、AMP 跳步和逐模块记录仍需 `*_profile.json`、`history.jsonl`。

### FP32 大 batch 补测及最终资源选择

用户进一步执行同六组合的 `--batch-sizes 32 64 128 --precisions fp32 --steps 4 --samples 128`，每组合短训练及恢复正常结束。以下数据来自用户导出的 `*_profile.json`；全部 18 行的 status 均为 `ok`。每个单元格表示“峰值 allocated MiB / samples/s”：

| 模型 | 任务 | batch 32 | batch 64 | batch 128 |
|---|---|---:|---:|---:|
| TriRelay | major15 | 3616.6 / 163.2 | 6791.9 / 283.7 | 13127.8 / 12.6 |
| TriRelay | fine39 | 3788.6 / 171.7 | 7099.6 / 284.0 | 13730.6 / 10.6 |
| GeoFactor | major15 | 3870.7 / 194.2 | 7291.7 / 303.1 | 14197.9 / 5.8 |
| GeoFactor | fine39 | 3985.3 / 189.9 | 7504.6 / 304.2 | 14603.4 / 4.7 |
| ProtoRoute | major15 | 3327.6 / 195.6 | 6241.6 / 325.2 | 12059.1 / 21.1 |
| ProtoRoute | fine39 | 3434.9 / 191.0 | 6427.4 / 306.5 | 12421.8 / 19.5 |

统一选择 batch 64 / FP32：六组合均为本次吞吐最佳，相比 batch 32 提升约 56%–74%，allocated 峰值 6.10–7.33 GiB。batch 128 的 allocated 峰值为 12,059–14,603 MiB，达到或超过物理显存 12,227 MiB（另有桌面等占用），吞吐骤降至 4.7–21.1 samples/s。status `ok` 仅表示脚本的数值/更新条件通过，不能证明分配都驻留于独立显存或资源效率良好。

根据超出物理显存的分配量及吞吐骤降，最合理的解释是 Windows WDDM / NVIDIA 系统内存回退造成分页或数据迁移；本次没有直接采集共享内存曲线，故标记为推断。NVIDIA [系统内存回退说明](https://nvidia.custhelp.com/app/answers/detail/a_id/5490/~/system-memory-fallback-for-stable-diffusion) 说明显存耗尽时应用可继续执行但速度降低。当前通过 batch 64 解决实际资源问题，不需要更改驱动策略或再扩大 batch。

按 batch 64 的实测吞吐线性外推，约 3.95 万 Train 样本的训练部分为 2.0–2.3 分钟/epoch；仍不包含完整 Val、初始化和保存，且真实长度与数据读取会影响耗时。这是短测资源估计，不是全量训练耗时或泛化表现结论。

### V100 真实数据 GPU 短训练

用户回传命令为 `gpu-check --esm 650m --device cuda:0 --models trirelay geofactor protoroute --tasks major15 fine39 --batch-sizes 32 64 128 --precisions fp32 fp16 --steps 32 --samples 128`。六组合均从 Checking 输出进入训练，打印至 epoch 34 后正常返回终端，无 traceback。按脚本流程，已执行完整缓存和数据加载、至少一项有效资源配置的关键梯度/参数更新检查、共享 trainer 短训练、checkpoint 恢复及独立 Val 读取。

| 模型 | 任务 | 首条 epoch Loss | 末条 epoch Loss |
|---|---|---:|---:|
| TriRelay | major15 | 0.462949 | 0.205145 |
| TriRelay | fine39 | 0.381858 | 0.192974 |
| GeoFactor | major15 | 0.672648 | 0.252435 |
| GeoFactor | fine39 | 0.982272 | 0.195151 |
| ProtoRoute | major15 | 0.559270 | 0.226096 |
| ProtoRoute | fine39 | 0.498265 | 0.238074 |

这是反复使用 128 个代表性 Train 样本的有界检查，epoch 34 不代表 34 轮全量训练。配置是 32 次 optimizer step 尝试加恢复阶段 2 次，若选中 FP16，真实 AMP 跳步数仍需 history 确认。当前日志未显示 NaN/Inf 或错误，不能据此推断所有 batch/精度通过或 FP16 无跳步。

GeoFactor 的 epoch 2 总 Loss 分别达到 7.351419 / 7.095124，随后整体回落，末尾约 0.25 / 0.20。用户回传的分项记录确认，尖峰来自 reconstruction：major15 为 143.610764，fine39 为 139.522278；按 0.05 权重分别贡献约 7.180538 / 6.976114，即总 Loss 的 97.7% / 98.3%。同期 classification 仅为 0.142288 / 0.099138，direct 平稳，factor 的加权贡献很小，两个任务前六轮 skipped_amp_updates 均为 0。可排除这几轮总 Loss 尖峰由主分类项或 AMP 跳步直接造成，但尚未拆分重建内部的理化、几何、DSSP 项，不能断言具体子目标或根本原因。

源码确认重建由按组平均的描述符 MSE、几何 MSE 和 DSSP 交叉熵组成；这些目标尺度不同。当前未发现可由已有日志直接证明的计算错误，不修改正在运行的 RTX 实现或删除重建模块。正式配置已有 5 epoch 的学习率和辅助目标 warmup，诊断特意用 warmup=0 / aux_warmup=1，两者优化过程不同；不能用短测尖峰预测全量收敛。后续依据正式 history 中重建加权占比与分类/Val 趋势决定是否调整重建目标尺度或权重。小样本 Val AP 不用于模型排名、硬件比较或全量泛化判断。

用户随后回传 V100 全部资源表：六组合 × FP32/FP16 × batch 32/64/128 的 36 行均为 status=ok；每个组合吞吐最高都是 batch 128 / FP16。关键两种配置比较如下：

| 模型 | 任务 | FP32 / 64 samples/s | FP32 / 64 allocated MiB | FP16 / 128 samples/s | FP16 / 128 allocated MiB |
|---|---|---:|---:|---:|---:|
| TriRelay | major15 | 79.9 | 6792.9 | 96.3 | 7564.5 |
| TriRelay | fine39 | 83.8 | 7101.6 | 106.2 | 7919.1 |
| GeoFactor | major15 | 92.1 | 7291.7 | 101.3 | 7952.8 |
| GeoFactor | fine39 | 92.7 | 7508.6 | 108.7 | 8204.8 |
| ProtoRoute | major15 | 94.5 | 6238.7 | 100.1 | 6962.7 |
| ProtoRoute | fine39 | 58.4 | 6428.6 | 65.4 | 7232.1 |

FP16 / 128 allocated 峰值约 6.80–8.01 GiB，相比 FP32 / 64 吞吐增益约 6%–27%。FP32 / 128 allocated 约 12,058–14,603 MiB，显存余量较小且吞吐并非所有组合都提升，不作为统一推荐。ProtoRoute fine39 在该机本次测试中明显慢于 major15，当前只有总 step 吞吐，不能归因于某个模块、硬件故障或标签数，暂不因此追加重测或改架构。

V100 实际参数量、reserved 峰值和全部训练 epoch 的 AMP 跳步总数仍未回传，不能用前六轮 GeoFactor 的 0 外推全部训练。650M 数据与数组读取已实际发生；35M 目前只确认元数据可读，尚未执行三模型 GPU 检查。

## 3. 本轮发现和修复

1. 原 `limit_samples` 以子集重算标签先验，可能使高阶经验因子消失、原型容量全部缩到 1。已改为保留完整原始 Train 统计，只限制优化样本，保持完整模型机制与统一长尾分层。
2. 原前 N 行诊断不保证包含 D/无结构/长肽。已增加代表样本选择，并将所选真实 row_id 正确映射回 ESM 缓存位置；不重排缓存、不改变划分。
3. 原训练只有 FP32，缺少有界 optimizer step 诊断。已接入 autocast、FP16 GradScaler、梯度累积和 max_steps，保存恢复 scaler/global step，记录 AMP 跳过更新。FP16 分类 Loss 在 FP32 计算。
4. 新增 `gpu-check`：复用真实缓存检查、完整模型资源测量、关键模块输出与梯度更新检查，接着调用共享 trainer 做短训练、Validation、checkpoint 和恢复；不启动 Test 或多 seed。

## 4. 已执行的相关检查

- CPU 定向回归：代表样本与 ESM 缓存位置一致；累积梯度和有界更新次数；完整 Train 统计保留；checkpoint/scaler/global step 保存恢复；既有 TriRelay 训练/推理流程。使用测试 fixture，仅说明代码链路，不是真实数据训练或 GPU 结论。
- 现有数据只读，未修改迁移包、划分、真实缓存或历史结果。统一评价仍调用同一 metrics 实现；Val Macro-AP → Micro-AP → 更早 epoch，阈值仅 Val 拟合。
- RTX 的 major15/fine39 六组合真实链路已运行结束，batch 64 / FP32 为统一起始选择，已进入正式训练。V100 六组合 GPU 短训练也已运行结束，36 项资源状态均为 ok，GeoFactor 的早期重建尖峰已定位；完整逐模块记录、全部 AMP 跳步仍未回传。ESM-35M 尚未执行模型 GPU 检查。

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

RTX 起始正式训练采用补测确定的 batch 64 / FP32，650M 缓存、seed 42、既有完整模型和默认学习率 2e-4、100 epochs 上限、patience 15；不附加诊断样本或 step 限制，不按 batch 倍数机械放大学习率。每 epoch 约 617 个 optimizer step（无梯度累积），保持六组合统一。V100 若承担同一正式比较协议，沿用已通过的 batch 64 / FP32，与 RTX 保持有效 batch 和精度一致；V100 若采用资源最优的 batch 128 / FP16，应单独标记为不同训练设置，不能把差异仅解释为 GPU 速度或模型差异。无需削减架构或继续做更大 batch 探测。

```bash
mfp-nova train --model trirelay --task major15 --esm 650m --seed 42 --device cuda:0 --precision fp32 --batch-size 64 --name trirelay_major15_esm650m_seed42_rtx5070_fp32_b64
mfp-nova train --model geofactor --task major15 --esm 650m --seed 42 --device cuda:0 --precision fp32 --batch-size 64 --name geofactor_major15_esm650m_seed42_rtx5070_fp32_b64
mfp-nova train --model protoroute --task major15 --esm 650m --seed 42 --device cuda:0 --precision fp32 --batch-size 64 --name protoroute_major15_esm650m_seed42_rtx5070_fp32_b64
```

fine39 替换 task 及 name 中的任务名称，顺序执行并使用独立 run；本机未代为启动正式训练。当前证据足以按上述设置开展全量 Train/Val 单次正式训练，不必重复全部诊断。Test 仍独立留待最终评价。RTX 已在用户终端训练；V100 数据与运行链路就绪，正式实验应明确任务分工及 batch/精度协议，不默认重复 RTX 的同配置同 seed 实验。

## 7. RTX 5070 首轮全量训练与下一步对照

来源：用户回传的 `results.zip` 六个正式结果，run 标识为 `20261015_105336`。共同设置为 ESM-650M、seed 42、FP32、batch 64、accumulation 1、global 阈值、Val 拟合，diagnostic_subset=false。Train/Val 每轮为 617 个 optimizer step，六份结果的逐标签 Val 正例数与本机当前 major15/fine39 队列一致。不是 Test 结果或多 seed 均值。

| 模型 | 任务 | Macro-AP | Micro-AP | Macro-F1 | Micro-F1 | Jaccard | Tail AP | MFP 检测 F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TriRelay | major15 | 0.540361 | 0.698735 | 0.533829 | 0.643729 | 0.616967 | 0.165782 | 0.505152 |
| GeoFactor | major15 | 0.528925 | 0.683219 | 0.491238 | 0.627734 | 0.594457 | 0.114600 | 0.520319 |
| ProtoRoute | major15 | 0.537225 | 0.688001 | 0.509786 | 0.631151 | 0.591137 | 0.232171 | 0.496069 |
| TriRelay | fine39 | 0.518048 | 0.628616 | 0.518930 | 0.586960 | 0.540010 | 0.447529 | 0.495767 |
| GeoFactor | fine39 | 0.506050 | 0.624103 | 0.516135 | 0.592903 | 0.548995 | 0.418741 | 0.526949 |
| ProtoRoute | fine39 | 0.514995 | 0.636843 | 0.521421 | 0.598489 | 0.555383 | 0.428661 | 0.514027 |

| 模型 | major15 最佳/结束 epoch | fine39 最佳/结束 epoch | major15/fine39 分钟 | major15/fine39 allocated GiB |
|---|---|---|---|---|
| TriRelay | 20/35 | 21/36 | 78.7/82.4 | 6.62/6.93 |
| GeoFactor | 14/29 | 46/61 | 85.3/134.7 | 7.12/7.34 |
| ProtoRoute | 12/27 | 26/41 | 54.1/89.2 | 6.08/6.27 |

训练器记录耗时合计约 8 小时 44 分钟，不含此前 ESM 提取；停止轮数不同，不能用总时长直接代替同条件 step 性能。返回的实际可训练参数量与第 2 节本地完整规格一致。

结论：TriRelay 在 major15 的主要综合指标领先，并在两个任务的主指标 Macro-AP 领先；ProtoRoute 在 fine39 的 Micro-AP、Macro/Micro-F1、Jaccard 更高；GeoFactor 在两任务的 MFP 检测 F1 更高。TriRelay 与 ProtoRoute 的 Macro-AP 差距仅约 0.31 个百分点，单 seed 不能确定稳定优胜。fine39 尾类共 18 个，TriRelay 的 Tail AP 最高；major15 尾类仅 thrombolytic 一个，不能据此概括一般长尾能力。antiinflammatory 等标签仍明显薄弱；MFP 检测 F1 约 0.50–0.53，尚有改进空间。结果均按观测标注评价，0 不代表实验确认阴性。

六组都在最佳 epoch 后 15 轮早停；后期总 Loss 继续下降但 Val AP 回落，不靠盲目延长训练解决。目前不因单 seed 结果调整架构、Loss 或主选模指标，也不使用 Test 调参。

本机将 ZIP 中原始 results.json 放到对应真实 run 名称下的 runs/，受 .gitignore 忽略；只有结果文件，未伪造 metadata.json、history 或 checkpoint。模型权重和完整日志仍在 RTX，因此本机尚不能执行这些 checkpoint 的独立评价。原有 `mfp-nova summarize` 需要各 run 原始 metadata.json；新增的固定协议汇总 `python -m mfp_nova.experiments summary` 可读这些旧结果，并明确标记本机缺少完整元数据。

下一步：固定 Seed 为 42、101、202、303、404，复用六组 42 结果，补齐其余 24 组 ESM-650M、batch 64、FP32 实验，不修改模型、Loss 或其他正式超参数。本机的 35M 缓存可读不代表三模型 35M GPU 训练已经验证。当前未启动新增正式训练或 Test，V100 暂不处理。

## 8. 完整 Epoch 指标与分批五 Seed 执行

三模型共用训练器，每轮输出四行：运行标识、Train、Val、Extra。Train/Val 均包含 Acc（样本 Jaccard）、Prec（样本 Precision）、Cov（样本 Recall）、AT（Exact Match）、AF（Hamming Loss），以及 Macro/Micro-AP、Macro/Micro-F1、MFP-F1；Extra 显示 Val Head/Medium/Tail AP、学习率、耗时、峰值显存、最佳 epoch 和早停计数。

`history.jsonl` 保存全部逐标签、分组及训练状态指标。Train 复用当轮带增强、Dropout、权重持续更新的训练预测，不增加 Train 前向；Val 在 eval 模式遍历一次，所有指标和分组共用预测。Val Loss 为主分类 Loss，不混入随机重建目标。每轮监控阈值固定为 0.5，最终最佳模型仍仅在 Val 拟合阈值；旧 42 未记录的历史不补造。

新增 `scripts/run_multiseed.ps1`，默认串行运行一组，可指定三组或显式 `-All`。已完成结果跳过；中断任务从原 `last.pt` 恢复到新运行目录，沿用 Trainer 的 optimizer、scheduler、随机状态和早停状态。损坏或不兼容 checkpoint 明确报错；没有 checkpoint 的尝试使用新目录重新训练，不覆盖历史。仅依据可识别的本机活跃训练命令标记 running，其余未完成 checkpoint 不猜测为正在运行。

只读进度和汇总由 `python -m mfp_nova.experiments plan/summary` 提供；同一 Seed 只计一次，阶段性结果注明实际 n，样本标准差在 n=1 时为 NA/null。逐标签和已有分组可单独查询，旧结果没有的分组标记缺失。完整启动、恢复、汇总和单次执行策略命令集中在 README 的“快速使用：RTX 5070 分批五 Seed”。

验证范围：新增指标/分批脚本测试和既有汇总回归共 26 项通过。三模型使用 CPU fixture 检查在线 Train 预测复用、完整 Val、指标落盘、并列选模和末次 checkpoint 恢复；PowerShell 5.1/7 使用受控替身检查批次上限、跳过、恢复和失败停止。修正 PowerShell 7 测试捕获文本的编码后，受影响的 6 项复查通过。真实结果目录执行只读 DryRun 与汇总，确认为 6/30 完成、24 待运行。本轮未执行新增 Seed 的 GPU 训练，新增指标在正式 RTX 全量训练中的 CPU 统计耗时尚未实测。
