# TheraSet后续测试：统一使用V100

日期：2026-11-08。依据已完成的54组seed42消融，第一轮只新增fine39的`no_label_context_single_prototype`，分别按`traditional5`和`micro_f1`训练，共2组。原full和no_label_context沿用同平台、同设置的既有结果，不重新选取跨设备最高成绩。

后续统一在V100训练；设置为seed42、FP32、物理batch64、累积1、原数据划分、原损失与EMA/R-Drop日程。保留pair、PLM、理化、结构和化学输入；同时关闭label_context和prototype_readout。单原型沿用原消融的active mask实现，尚未裁剪稠密查询。每套协议仍记录传统五指标、Micro/Macro-F1和AP，但分别按自己的目标选模。

## 文件与运行行为

- `scripts/run_theraset_followup.sh`：V100，两张卡各执行一套协议；指定一张卡时顺序执行。
- V100脚本支持`plan/run/pause/pack/diagnose`。重复run跳过已完成组，从未完成组的last.pt续训；暂停保留原优化器、调度器、EMA与随机状态。使用同一个RunRoot续训。
- 新任务使用独立followup目录，原54组默认清单仍为54组。
- 默认复用已有Train/Val缓存并离线运行，不重新下载模型或提取特征。以下训练命令均在V100的Bash终端执行。

## 代码同步

源码通过GitHub的main分支同步。V100只需拉取本次提交，然后执行下面的plan/run命令；无需在RTX5070启动任务。

## V100：GPU2、GPU3后台并行

先在V100 Bash同步，GPU2运行traditional5，GPU3运行micro_f1。单卡显存占用相互独立。

```bash
cd ~/wuzhencheng/projects/MFP-Nova
conda activate MFP-Nova
git pull --ff-only origin main || exit

export PYTHON_BIN="$(command -v python)"
export RUN_ROOT=runs/theraset_followup_seed42_v100
export GPUS='2 3'
export CPU_THREADS=4
mkdir -p "$RUN_ROOT"

bash scripts/run_theraset_followup.sh plan
nohup bash scripts/run_theraset_followup.sh run >> "$RUN_ROOT/launcher.log" 2>&1 < /dev/null &
echo "Launcher PID=$!"
```

只使用一张卡时改为`export GPUS='2'`，两套协议会顺序执行。脚本不会启动其他变体或其他任务。查看日志：

```bash
tail -n 30 -f "$RUN_ROOT/traditional5_run.log" "$RUN_ROOT/micro_f1_run.log"
```

可以在同一服务器任意已激活MFP-Nova环境的窗口暂停；新窗口需要先重新设置上述RUN_ROOT：

```bash
bash scripts/run_theraset_followup.sh pause
```

等旧进程退出后，使用相同目录续训：

```bash
rm -f "$RUN_ROOT/PAUSE"
nohup bash scripts/run_theraset_followup.sh run >> "$RUN_ROOT/launcher.log" 2>&1 < /dev/null &
echo "Launcher PID=$!"
```

完成后打包：

```bash
bash scripts/run_theraset_followup.sh pack
```

默认包名示例：

```text
theraset_no_label_context_single_prototype_seed42_v100_fine39-b64_traditional5-microf1_2of2_YYYYMMDD_HHMMSS.zip
```

ZIP位于RunRoot的上级runs目录，包含结果、历史、预测、诊断和日志，不含checkpoint、ESM缓存或原始数据。`2of2`表示训练完成数；summary.csv另外记录diagnostics_completed。

## 证据修正诊断与旧对照

每组训练完成后，脚本在其best.pt上额外执行一次完整Val推理，生成：

- `evidence_diagnostics.json`：CNN直接头与最终输出的指标，统一0.5及保存阈值下的修正/损害计数，各标签门值和修正幅度、长度/D注记/结构等子群统计。
- `evidence_validation.npz`：对齐row_id/ID/标签的直接头概率、最终概率、修正值、门值、原型权重及按原型聚合的模态/尺度权重。

诊断固定为FP32，不重新拟合阈值，也不读取Test。直接头是同一训练模型中的辅助监督头，不是独立训练的CNN基线；其决策使用最终头相同阈值，不能视为分别最优校准的比较。0表示未标注，修正FP只针对记录标签统计。模态权重本身不能证明因果贡献。

需要诊断旧full、no_label_context，必须仍有它们的best.pt；已下载的结果ZIP不含权重。V100已知旧根目录如下，可复用权重而不重训：

```bash
export REUSE_ROOT=runs/theraset_ablation_seed42_v100
VARIANTS='full no_label_context' nohup bash scripts/run_theraset_followup.sh diagnose \
  >> "$RUN_ROOT/controls_diagnostics.log" 2>&1 < /dev/null &
echo "Diagnostics PID=$!"
```

单纯诊断失败不会删除已完成训练结果。新组合若训练已完成但自动诊断中断，重复run会跳过训练并补齐诊断；也可直接执行diagnose。

若要将3种配置×2套协议的6组结果及诊断共同打包，执行：

```bash
VARIANTS='full no_label_context no_label_context_single_prototype' bash scripts/run_theraset_followup.sh pack
```



pack只寻找配置匹配的已完成对照，不启动训练。结果中保留原任务目录与指标；复用来源记在summary.csv。缺少旧对照时会显示未完成数，而不会自动补训。

## 结果出来后再安排的步骤

1. 组合配置先与no_label_context比较，再与原full比较；查看Micro-F1、Jaccard、AP、MFP和修正诊断，不把单项收益相加。
2. 有保留价值时，再在major15/therapep49运行同一组合；相应batch自动为128/64。通过Bash的TASKS环境变量明确指定，默认不会扩展。
3. fine39的去因子＋去pair也已支持，名称为`no_label_context_no_pair`；待第一轮分析后再启动。
4. 相同输入简单头、学习率比较和seed101/202不在本轮自动执行，避免同时改变多个因素。最终Test留到候选选择完成后。
