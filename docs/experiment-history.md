# 实验与开发记录

本文由原 README 迁入，保留历次安装说明、实验命令、指标与开发记录。历史三模型的数字与协议按当时范围解释；当前四模型目录与入口以[项目首页](../README.md)为准。此次整理没有执行训练或测试。

**2026-10-22：新增第四模型 HelixSet 和可切换评价模式。** 三模型现状、五 Seed Test 对比及第四模型技术说明集中在[架构说明第 9 节](../docs/MFP-Next-Generation-Architecture.md#9-当前现状与第四模型-helixset2026-10-22)。HelixSet 已写入源码，按本次用户要求**尚未运行或测试**，无性能结论；没有 ESM 直接预测通路。使用方式见本文末尾，其余历史实验记录保留。

独立的治疗肽多标签识别科研项目。已完成 TriRelay、GeoFactor、ProtoRoute 两任务各五 Seed 的首轮训练及独立评价；当前针对 TriRelay 的完整 major15/fine39、seed42 开展 EMA 和算法改进。所有核心代码独立编写，新候选继续仅用 Validation 比较。

## 本地数据与任务

本地迁移包完整解压至 `data/therapeutic_2025/`，原包保留。数据目录包含上游处理脚本，但本项目不调用或复制这些实现。

同类论文公开数据另存于 `data/literature_benchmarks/`，论文对应的数据集、原始划分、作者来源和下载失败原因见[数据说明](../data/literature_benchmarks/README.md)。本次取得 9 个方法目录的 183 个文件，约 49.19 MiB；SeekTP 原始下载链接失效，SA-MTP 低同源划分和 PDeepPP 部分任务数据尚未取得。各数据保持上游原格式，没有合并进当前训练队列。

按数据集归并的版本见 [`data/literature_consolidated/`](../data/literature_consolidated/README.md)：MTP15、MFTP21 各保留一份，MFBP5 共用样本表并保留原始/清洗两套划分，PDeepPP 的 32 个任务分别保存。不同数据集没有融合；原版 MFBP Train/Test 的 69 条相同序列重叠如实保留并说明。

实际读取统计见 [data_summary.json](../docs/data_summary.json)。原记录 58,583 条，增强候选 648 条，共 59,231 行；默认增强关闭。标签来自处理后的 Function 语义，不使用原 48 位 `Label encoding`。

| 任务/队列 | 标签 | Train | Validation | Test |
|---|---:|---:|---:|---:|
| major15 全监督原样本 | 15 | 41,008 | 8,788 | 8,787 |
| major15 明确骨架序列队列 | 15 | 39,478 | 8,534 | 8,187 |
| fine39 全监督原样本 | 39 | 40,967 | 8,780 | 8,779 |
| fine39 明确骨架序列队列（默认） | 39 | 39,455 | 8,529 | 8,180 |

2,384 条原记录没有明确标准骨架，仍保留在原数据中；当前 ESM 路径只使用可编码队列。fine39 的 57 条任务内全零记录由监督 mask 排除。原样本骨架长度 2–74，四分位数 9/15/24，95 分位 42；有 1,173 条 D 注记记录。输入保留 D/HELM/PTM 数据，但当前序列模型仅消费大写骨架，尚不区分构型/修饰。

两任务都是多标签二分类，0 表示未标注，而非实验阴性。fine39 的 `__unknown` 指未标具体亚类。预测两个标签不能直接认定两种独立生物学功能。

沿用新数据提供的约 70/15/15 修复划分。检查发现跨集合簇和相同明确骨架均为 0；没有重新运行上游全对相似度计算。新协议为 90% 编辑相似关系和专业候选复核，与旧蓝图的 80% alignment identity、80/10/10 划分不同。旧项目成绩不能作为本项目成绩或直接对照。

## 项目结构

```text
configs/sequence.yaml      # 冻结 ESM-2 35M 轻量对照
configs/sequence_650m.yaml # 冻结 ESM-2 650M 基线候选；两任务均可 --track 切换
src/mfp_nova/
  analysis.py              # 样本、标签、划分和 Train 共现统计
  data.py                  # 当前数组接口、ID/row_id 对齐、缓存 batch
  encoder.py               # 冻结 ESM、按 split 提取残基缓存
  model.py                 # 紧凑全局/局部序列模型
  losses.py                # ASL；Trainer 支持 BCE 对照
  metrics.py               # sklearn AP/F1/Jaccard、全局阈值
  trainer.py               # Train 更新、Val 选模、checkpoint
  inference.py             # 独立评价、序列预测
  cli.py                   # 统一命令入口
tests/test_core.py         # 必要正确性测试和两 epoch 合成训练
docs/model.md              # 设计依据、参数与评价协议
docs/data_summary.json     # 实际读取的必要统计，含完整标签顺序
```

## 安装

推荐 Python 3.12。GPU 构建按设备选择：V100 从已存在的 `trifusion_mfp` 克隆出独立的 `MFP-Nova` 环境（PyTorch 2.4.1+cu121），开发机和 RTX 5070 使用 PyTorch 2.8.0 的适用构建。当前代码所需的 PyTorch API 在 2.4 中已有，项目最低依赖调整为 2.4.1；V100 的 CPU 合成训练与评价测试已通过，真实 ESM/GPU 链路尚待验证。安装命令依据 [PyTorch 官方历史版本页](https://pytorch.org/get-started/previous-versions/)，`weights_only` 接口见 [PyTorch 2.4 文档](https://docs.pytorch.org/docs/2.4/generated/torch.load.html)。

两训练端建议共用以下主要 Python 依赖版本，PyTorch/CUDA 分别适配硬件：

| 组件 | V100 已有环境 | RTX 5070 目标配置 |
|---|---|---|
| Python | 3.12.14 | 3.12 |
| PyTorch / CUDA 构建 | 2.4.1+cu121 | 2.8.0+cu128 |
| Transformers | 4.44.2 | 4.44.2 |
| NumPy | 1.26.4 | 1.26.4 |
| scikit-learn | 1.5.2 | 1.5.2 |
| PyYAML | 6.0.2 | 6.0.2 |

RTX 5070 属于 Blackwell 架构；不能照搬 V100 的 2.4.1+cu121 构建。PyTorch 2.7 开始提供 Blackwell/CUDA 12.8 支持，当前 RTX 安装方案使用 2.8.0+cu128，见 [官方发布说明](https://pytorch.org/blog/pytorch-2-7/)。两台机器可以使用同一份代码、数据划分和实验配置，但不承诺跨设备逐位一致；运行记录会保存各自的 Torch/CUDA 版本。无需复制 Linux 的完整 Conda 导出配置到 Windows。

Windows 开发机（CPU）：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pytest -q
```

Windows RTX 5070：通过 UU 远程桌面在该机器打开 Anaconda PowerShell Prompt 或已能运行 Conda 的 PowerShell，建议项目放在用户目录的 `Projects/MFP-Nova`。下面代码使用 PowerShell 语法；Git clone 应单独执行，等认证和克隆完成后再执行后续命令。该机器需要已安装 Git 和 Conda。环境配置为 [environment-rtx5070.yaml](../configs/environment-rtx5070.yaml)，Conda 创建 Python 3.12 并通过 pip 安装指定 CUDA 12.8 构建；固定 `torch==2.8.0+cu128`，以明确选择 GPU wheel。首次安装执行：

```powershell
$taskProjectParent = Join-Path $env:USERPROFILE 'Projects'
New-Item -ItemType Directory -Path $taskProjectParent -Force | Out-Null
Set-Location $taskProjectParent
git clone https://github.com/zc-pro/MFP-Nova.git
Set-Location MFP-Nova
conda env create -f configs/environment-rtx5070.yaml
conda activate MFP-Nova
python -m pip install --no-deps -e .
python -m pip check
mfp-nova --help
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import torch; x=torch.ones(8,8,device='cuda'); print('CUDA matmul:', (x@x).sum().item())"
python -m pytest -q
```

Ubuntu V100：项目已在 `~/wuzhencheng/projects/MFP-Nova` 克隆成功。用户已确认 `trifusion_mfp` 的上述依赖版本和 `torch.cuda.is_available()=True`。使用 [Conda 官方克隆方式](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#cloning-an-environment) 创建独立的 `MFP-Nova`，保留原环境及 CUDA 12.1 构建；不要直接复制环境目录后改名。以下命令在登录后的 Ubuntu 终端执行；克隆命令只在首次创建环境时运行，后续直接 activate：

```bash
conda create --name MFP-Nova --clone trifusion_mfp -y
conda activate MFP-Nova
cd ~/wuzhencheng/projects/MFP-Nova
git pull --ff-only
python -m pip install --no-deps -e .
mfp-nova --help
python -m pytest -q
```

服务器测试使用 CPU 合成输入。数据集尚未传入时，真实数据测试会跳过；这些结果不代表 ESM 特征提取或 GPU 训练性能。如果环境没有 pytest，反馈缺失提示后只补必要测试依赖，不执行会升级整个旧环境的安装命令。

两台训练端都使用各自的 Conda 环境 `MFP-Nova`，每次打开新终端后先激活环境。环境配置文件只用于首次创建 RTX 环境；已有项目目录时先 `git pull --ff-only` 获取该文件。GitHub 是私有仓库，本机与 RTX 使用 HTTPS 认证；V100 的专用 Deploy key 拉取已成功。V100 项目已安装，5 项 CPU 测试通过，1 项因数据未传入跳过。RTX 环境已完成依赖和项目安装，确认 Python 3.12.15、PyTorch `2.8.0+cu128`、CUDA `12.8`、可用性 `True`、设备为 NVIDIA GeForce RTX 5070，GPU 小矩阵乘法输出 `512.0`。RTX 项目测试尚待执行。CPU 测试不代表真实 ESM 特征提取或 GPU 训练验证。

RTX 项目实际位于 `D:\Data\ChatGPT\mfp\MFP-Nova`。CMD 若提示 `Run 'conda init' before 'conda activate'`，可用以下命令激活已有环境；若完整 YAML 安装中断，单独安装依赖以显示下载进度，无需重新创建环境：

```cmd
call D:\App\Code\Anaconda\Scripts\activate.bat MFP-Nova
python -m pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 transformers==4.44.2 scikit-learn==1.5.2 PyYAML==6.0.2 pytest==8.4.2
python -m pip install --no-deps -e .
```

数据经 UU 传到 RTX 5070，经 SFTP/SCP 传到 V100；完整解压后，两端建议目录如下。外部数据盘可以通过 `--data-root` 指定。

```text
MFP-Nova/
  .venv/                     # 开发机 venv 可放这里；两服务器环境由 Conda 管理
  data/therapeutic_2025/      # 迁移包解压内容，直接包含 records.csv、arrays/、tracks/
  features/                  # 后续本机提取的特征
  runs/                      # 该机器的实验输出
```

## 使用

从仓库根目录执行命令，相对路径以当前目录为基准。服务器可用 `--data-root`、`--feature-root` 设置本机路径，无硬编码盘符。

```text
mfp-nova analyze --data-root data/therapeutic_2025 --output docs/data_summary.json
mfp-nova features --device cuda:0 --batch-size 32 --splits train val
mfp-nova train --device cuda:0 --track fine39 --name fine39_seq_seed42
mfp-nova train --device cuda:0 --track major15 --name major15_seq_seed42
```

首次特征提取会下载所选配置的 ESM-2 权重。默认仅提取 Train/Validation；同一编码器的原样本缓存可由两个标签任务共用。数据骨架保留大小写构型注记；只在送入 ESM 时转为大写，原始序列、D mask 和缓存的序列字段均保留。提取 FP32、存储 FP16，使用 NPY ragged 残基、offsets、row_id、ID 与序列，读取采用 mmap，不重复保存全局向量。模型权重 revision 自动记录；后续推理加载同一 revision。正式实验可将配置 revision 改为已记录的确定 commit。完整特征目录不覆盖。

若提取中断，`--resume-from N` 可复用已成功写入的前 N 条记录，从日志中最后一条 `train: N/总数` 读取 N。它只应用于 `--splits` 的第一个划分，其余划分正常新建；必须保持相同数据顺序、编码器权重、max_length 和缓存目录。未完成缓存只包含预分配的 `residues.npy`，末尾未写入区域不能用于训练。请勿删除需要续提取的文件。例如两次提取都已成功写入 38,336 条时：

```text
mfp-nova features --config configs/sequence.yaml --device cuda:0 --splits train val --resume-from 38336
mfp-nova features --config configs/sequence_650m.yaml --device cuda:0 --splits train val --resume-from 38336
```

ESM 使用末层残基状态，不启用未预训练的 pooler。Windows symlink 缓存提示、tokenizer cleanup 提示和 Xet 转普通 HTTP 下载提示不影响当前特征计算，无需为它们重新安装环境。

35M 和 650M 分别运行独立实验，在相同划分、seed 和下游训练协议下比较。无需重建 Conda 环境，配置如下：

| 配置 | 冻结编码器 | 残基维度 | 提取 batch | 特征目录 |
|---|---|---:|---:|---|
| `configs/sequence.yaml`（CLI 默认） | ESM-2 35M | 480 | 32 | `features/esm2_35m` |
| `configs/sequence_650m.yaml` | ESM-2 650M | 1280 | 8 | `features/esm2_650m` |

按顺序提取和训练，使用不同实验名：

```text
mfp-nova features --config configs/sequence.yaml --device cuda:0 --splits train val
mfp-nova train --config configs/sequence.yaml --device cuda:0 --name fine39_esm35m_seed42
mfp-nova features --config configs/sequence_650m.yaml --device cuda:0 --splits train val
mfp-nova train --config configs/sequence_650m.yaml --device cuda:0 --name fine39_esm650m_seed42
```

650M 的提取 batch 8 是尚未测量的起点，可通过 `features --batch-size 4` 调小；两套配置均保持训练 batch 128。模型从缓存自动读取输入维度并投影到128维。已有完整 35M 缓存可直接复用，无需重复提取。两种编码器的缓存必须分开放置；650M 的同队列残基数组约为35M的2.67倍，不代表总计算耗时比例。

确定方案后才提取 Test。650M 的评价需显式指定对应缓存，序列预测则自动使用 checkpoint 中记录的编码器：

```text
mfp-nova features --config configs/sequence_650m.yaml --device cuda:0 --splits test
mfp-nova evaluate --checkpoint runs/fine39_esm650m_seed42/best.pt --feature-root features/esm2_650m --device cuda:0 --output runs/fine39_esm650m_seed42/test.json
mfp-nova predict --checkpoint runs/fine39_esm650m_seed42/best.pt --sequences peptides.txt --device cuda:0 --output runs/fine39_esm650m_seed42/predictions.json
```

训练仅打开 Train/Validation 缓存，按 Validation Macro-AP 选最佳 checkpoint，Micro-AP 打破同分，完全相同则保留更早 epoch。训练后加载最佳模型，在 Validation 上拟合全局 Jaccard 阈值；结果中的 Val F1/Jaccard 为阈值拟合内估计。Test 未参与训练、选模或阈值拟合。输出 `runs/<name>/best.pt`、`history.jsonl`、`metadata.json`、`results.json`，已有运行目录拒绝覆盖。

可用 `--epochs`、`--batch-size`、`--seed`、`--loss` 和 `--global-only` 覆盖当前配置，实际生效配置写入 checkpoint 与 metadata，不需要复制配置文件。特征提取的 `--batch-size` 控制 ESM 批次，训练/评价的该选项控制分类模型批次。以下示例共用已提取的 Train/Val 缓存，各次运行只更改一个研究因素：

```text
mfp-nova train --device cpu --epochs 2 --name fine39_smoke_seed42
mfp-nova train --device cuda:0 --seed 42 --name fine39_seq_seed42
mfp-nova train --device cuda:0 --seed 42 --global-only --name fine39_global_seed42
mfp-nova train --device cuda:0 --seed 42 --loss bce --name fine39_bce_seed42
```

两轮短训练用于检查完整队列的运行链路，不用于报告最终模型成绩。`history.jsonl` 记录每轮训练与验证耗时，`results.json` 记录完成轮数和训练至阈值拟合结束的总耗时；`metadata.json` 记录可训练参数量（不含冻结 ESM）、主要依赖版本和设备。

确定方案后，独立提取并评价 Test：

```text
mfp-nova features --device cuda:0 --splits test
mfp-nova evaluate --checkpoint runs/fine39_seq_seed42/best.pt --device cuda:0 --output runs/fine39_seq_seed42/test.json --predictions runs/fine39_seq_seed42/test_predictions.npz
mfp-nova predict --checkpoint runs/fine39_seq_seed42/best.pt --sequences peptides.txt --device cuda:0 --output predictions.json
```

`peptides.txt` 每行一个明确、非空的 20-AA 骨架，可保留表示构型的小写字母；ESM 输入视图转为大写，返回结果保留输入字符串。非标准化学表示不进行猜测转换，超过 token 上限明确报错，不静默截断。推理返回所有标签概率和阈值通过的标签，可全部为阴性。原始修饰肽的化学身份需要独立处理。

评价 JSON 中的 `label_ap` 和 `label_positive_support` 按 `labels` 顺序对应。可选 `--predictions` 导出 NPZ：`row_id`、`ID`、`canonical_backbone`、`label_names`、`targets`、`probabilities`、`threshold` 及划分、任务、编码器版本信息。行顺序与当前任务监督队列一致，独立于缓存保存顺序，便于后续错误分析；用 `np.load(path, allow_pickle=False)` 读取。结果与概率文件拒绝覆盖已有文件，大型概率文件放在被 Git 忽略的 `runs/` 中。

消融通过上述 CLI 选项或修改 `model.use_local`、`loss.name` 实现，使用新实验名。增强 `data.augment: true` 仅使用当前任务提供的训练候选，需要新的含增强特征目录；fine39 为 648 条，major15 为 60 条。暂未实现 CV/增强 CV，不将固定划分增强用于开发折。

## GitHub 与服务器

GitHub 私有仓库已创建：[zc-pro/MFP-Nova](https://github.com/zc-pro/MFP-Nova)。本机为开发端，两台服务器为独立训练端；源码通过 GitHub 同步，数据与运行产物单独传输。

```text
git add <CHANGED_SOURCE_FILES>
git commit -m "Describe the research change"
git push origin main
```

三机按以下方式协作：

| 设备/用途 | 连接与同步方式 |
|---|---|
| 本机开发 → GitHub | HTTPS Git，已通过现有 Git Credential Manager 登录并完成推送 |
| Ubuntu V100 管理 | `ssh MFP-Nova-V100`，使用 `ubuntu@192.168.13.171:22` 的密码登录 |
| V100 ↔ GitHub | SSH Git，专用 Deploy key 已成功免密拉取；写入权限勾选 Allow write access 后可 push |
| RTX 5070 ↔ GitHub | HTTPS Git，在 RTX 5070 上通过 UU 远程操作终端完成 clone/pull |
| 本机 ↔ RTX 5070 大文件 | 用户指定的 UU 远程文件传输，同步数据、特征、权重和完整运行输出 |

本机与 RTX 5070 不建立 SSH 连接，也不需要为此配置 sshd、端口或公钥。本机登录 V100 使用密码；V100 访问 GitHub 使用独立的项目 Deploy key。本机已将 `MFP-Nova-V100` 别名调整为交互式密码认证，并移除了 RTX 5070 项目 SSH 别名。

V100 的 SSH 密码在本机终端提示中输入。本机与 RTX 的私有 GitHub 仓库 HTTPS 认证各自完成：优先使用已有 Git Credential Manager/浏览器登录；终端需要 HTTPS 凭据时使用 GitHub token，不使用 GitHub 账户密码。凭据不放入 clone URL、脚本、配置或 Git 仓库。现有本机 GitHub 登录无需重新设置。

两台服务器的安装命令见上面的安装章节。RTX 项目位于 `D:\Data\ChatGPT\mfp\MFP-Nova`，环境位于 `D:\App\Code\Anaconda\envs\MFP-Nova`，已完成 ESM-35M/650M 特征提取、序列模型及三套多模态模型的 seed=42 正式训练。当前继续在本机开发、RTX 执行分批五 Seed 训练；暂不操作 V100。

对于 V100，可在本机 PowerShell 登录：

```powershell
ssh MFP-Nova-V100
```

对于 RTX 5070，在 UU 远程桌面内打开 PowerShell，执行 Git 和环境安装命令。源码与小型配置通过 GitHub，同一台机器上的 `.venv` 保留在该机器，不通过 UU 复制另一台机器的 Python 环境。

数据迁移包通过 UU 传到 RTX 5070 后，完整解压到项目的 `data/therapeutic_2025/`，或使用 `--data-root` 指定外部数据目录。V100 的数据可通过密码认证的 SFTP/SCP 单独传输。特征、权重、checkpoint 和完整运行输出均已被 Git 忽略；小型实验指标汇总可以单独放到 `results/` 后用 Git 提交，其余产物通过 UU 或 V100 的 SFTP/SCP 回传。

日常同步使用标准 Git，在训练停止后更新源码：

```text
git pull --ff-only
```

正式实验可 `git checkout <COMMIT>`。本机提交代码并推送后，两台训练端拉取；共享 V100 按服务器使用约定取得资源。不同运行使用独立名称，不在运行中的目录更新源码，不覆盖既有实验产物。

当前保留序列对照，并实现 TriRelay、GeoFactor、ProtoRoute 三套完整多模态模型；结构、D/PTM、标签关系和长尾机制均有实际代码。下一步在现有明确骨架队列上正式训练和消融，模型性能以独立 Test 和多 seed 实验为准。

本地已用 Python 3.12、PyTorch 2.8.0 CPU 完成 6 项针对性测试：真实数据对齐/划分、padding 不变性、ASL 数值和梯度、指标语义、冻结编码器接口与缓存写入、两 epoch 合成训练和 checkpoint/评价加载。编码器接口测试使用替身，不下载预训练权重；这些检查不代表真实 ESM 特征或 GPU 性能验证。pytest 临时文件配置在 `.pytest_tmp`，避免本机系统临时目录的权限问题。

## 四模型入口与三模型历史实验

### 快速使用：RTX 5070 分批五 Seed

当前批量入口已包含四模型，`all` 对应 40 组；HelixSet 使用自身配置中的 microbatch16、累积4、EMA0.999 与传统指标选模。下段 30 组/24 组数字保留原三模型的历史计划，不表示 HelixSet 已完成实验。

固定 Seed 为 **42、101、202、303、404**，三模型 × 两任务共 30 组。已有 seed=42 六组正式结果直接复用，新增 24 组保持 **ESM-650M、FP32、batch 64、无梯度累积**，沿用现有架构、Loss、学习率计划、100 epoch 上限和 patience=15。仅使用 Train/Validation，不启动 Test；不根据阶段性排名修改 Seed 或协议。

在 RTX 无 MFP 训练进程运行时执行 `git pull --ff-only`，然后在项目根目录 `D:\Data\ChatGPT\mfp\MFP-Nova` 的已激活 `MFP-Nova` PowerShell 中使用以下命令。已有 editable 安装无需重装依赖。本机用于开发与整理结果，不执行正式 GPU 训练。

```powershell
# A. 直接训练一组；完成后会被批量脚本识别并跳过
python -m mfp_nova.cli train --model trirelay --task major15 --esm 650m --seed 101 --device cuda:0 --precision fp32 --batch-size 64 --name trirelay_major15_esm650m_seed101_rtx5070_fp32_b64

# B. 查看当前四模型 40 组的状态与剩余计划；不加载 checkpoint、不启动训练
.\scripts\run_multiseed.ps1 -DryRun

# C. 每次只执行下一组（默认就是 MaxRuns=1），完成后返回终端
.\scripts\run_multiseed.ps1 -MaxRuns 1

# D. 本次最多连续执行三组
.\scripts\run_multiseed.ps1 -MaxRuns 3

# E. 仅处理某个模型
.\scripts\run_multiseed.ps1 -Model trirelay -MaxRuns 1

# F. 仅处理指定模型、任务和固定清单中的 Seed
.\scripts\run_multiseed.ps1 -Model protoroute -Task fine39 -Seeds "101,202" -MaxRuns 1

# G. 用户明确选择时才连续执行全部剩余任务；不能与 MaxRuns 同时指定
.\scripts\run_multiseed.ps1 -All

# H. 中断后重新执行相同命令：有 last.pt 自动恢复，无 checkpoint 从新运行目录重做
.\scripts\run_multiseed.ps1 -Model trirelay -Task major15 -Seeds "101" -MaxRuns 1

# I. 查看完成数、当前剩余、实际检测到的运行进程、最佳 Epoch 和主要指标
python -m mfp_nova.experiments plan

# J. 随时查看阶段性 mean ± 样本 std；每个模型的五 Seed 分别汇总
python -m mfp_nova.experiments summary --task major15
python -m mfp_nova.experiments summary --task fine39

# 同一个 Seed 的四模型记录（按各自评价模式说明）
python -m mfp_nova.experiments summary --seeds "42"

# 逐标签、可靠分组查询，以及完整机器可读汇总
python -m mfp_nova.experiments summary --task fine39 --label antiinflammatory
python -m mfp_nova.experiments summary --task fine39 --group structure_missing
python -m mfp_nova.experiments summary --json
```

需要直接指定 checkpoint 时，使用新名称，保留原运行目录：

```powershell
$resumeName = "trirelay_major15_esm650m_seed101_rtx5070_fp32_b64_resume_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
python -m mfp_nova.cli train --model trirelay --task major15 --esm 650m --seed 101 --device cuda:0 --precision fp32 --batch-size 64 --resume runs/trirelay_major15_esm650m_seed101_rtx5070_fp32_b64/last.pt --name $resumeName
```

脚本兼容 Windows PowerShell 5.1 / PowerShell 7，Seed 用逗号分隔字符串。若当前执行策略阻止 ps1，可单次启动，不需要管理员权限或永久修改系统策略：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_multiseed.ps1 -DryRun
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_multiseed.ps1 -MaxRuns 1
```

使用已有 Python 环境，脚本也支持 `-Python "D:\App\Code\Anaconda\envs\MFP-Nova\python.exe"` 和 `-Device cuda:0`。默认前台、单 GPU 串行；不创建后台服务、计划任务或开机启动，不关闭程序或更改驱动、电源设置。办公/游戏时间由用户安排，建议训练与大型 3D 游戏错开；一组结束后脚本默认退出。Ctrl+C 中断后从最近完成 epoch 的 last.pt 继续，当前 epoch 可能重做；不伪装为精确 batch 中途续接。

状态来自 results.json、metadata.json、last.pt 和当前进程命令：`completed` 已完成，`running` 是检测到对应 --name 的存活进程，`unfinished_checkpoint` / `unfinished_no_checkpoint` 表示未观察到对应运行进程但存在未完成目录，`not_started` 尚未开始，`incompatible` 与固定协议不符。没有 --name 或无法识别的外部进程不作猜测。恢复损坏的 checkpoint 时 Trainer 明确报错，脚本停止，不悄悄从头覆盖；无 checkpoint 的旧尝试保留，新尝试使用 retry 名称。一次脚本串行运行；不提供跨多个终端抢启动的调度/锁服务。

汇总仅纳入已完成且属于本协议的结果，每个 Seed 只计一次，不按 Validation 分数挑选重复运行。阶段性显示实际 n/5；n=1 时 std=NA/null，只有五个固定 Seed 都完成才标记 five-seed。缺失或不可计算指标单独报告有效 Seed 数。旧 seed=42 原始结果不回写：传统五项可从已记录的同义指标精确映射，旧文件没有的 Sample-F1、历史 Train 指标和分组不会补造。本机只有其 results.json 时显示 results-only 说明，不能声称已经核对未回传的完整 metadata/checkpoint。

### 每 Epoch 指标与记录口径

完整定义、计算公式、取值范围、零分母处理及指标间联系见 [指标说明书](../docs/MFP-Metrics-Guide.md)。

三套研究模型的终端每轮输出四行：身份与固定监控阈值、Train、Val、Extra。Train/Val 都显示 Loss、Acc、Prec、Cov、AT、AF、Macro/Micro-AP、Macro/Micro-F1、MFP-F1；Extra 显示 Val Head/Medium/Tail AP、LR、epoch/累计耗时、显存峰值、当前最佳 epoch 与早停计数。

传统五项定义为样本均值，设真实标签集合为 Y、预测集合为 P：

| 终端名 | JSON 字段 | 定义 |
|---|---|---|
| Acc | accuracy / sample_jaccard | 平均 \|Y∩P\| / \|Y∪P\| |
| Prec | precision / sample_precision | 平均 \|Y∩P\| / \|P\| |
| Cov | coverage / sample_recall | 平均 \|Y∩P\| / \|Y\| |
| AT | absolute_true / exact_match | 平均 1[Y=P] |
| AF | absolute_false / hamming_loss | 平均标签位错误率；越低越好 |

Acc 不等于 Exact Match，Cov 不等于 Coverage Error 或 Micro-Recall。零分母沿用 sklearn 的 zero_division=0；不可用 AP/AUROC/MCC 等为 null，原因放在 `undefined_metrics`，其中标签索引对应 metadata.labels。各组无样本时 metrics=null、reason=no_samples；未知结构置信度不归为低质量，置信度区间也不等价于跨来源校准后的结构可靠性。

`history.jsonl` 每行保存 `train`、`validation` 的全部全局/逐标签指标，包含 AP/AUROC、Macro/Micro P/R/F1、Sample-F1、MCC、Sensitivity/Specificity、固定 Train 分层 AP，以及独立功能映射下的 MFP P/R/F1、真实/预测数、TP/FP/FN、真实多功能样本的功能集合指标。`validation_groups` 保存有无结构、D/L、长度、独立单/多功能、已知来源/置信度分组的样本数和指标；全局逐标签数组按 metadata.labels 顺序，不在每组重复保存。Head/Medium/Tail 是 Train 支持度定义的标签分层，并记录参与 AP 的标签数，不伪装成互斥的样本分组。

`losses` / `loss_settings` 记录 Train 总/分项 Loss 和实际 warmup 后权重；另有 `validation_loss`、LR、epoch/累计秒数、每轮峰值 GPU 显存、best_epoch / best_macro_ap / best_micro_ap、early_stopping_count 和 AMP 跳步。JSONL 可以按 epoch 直接读取绘图，无需另建日志系统：

```powershell
Get-Content runs/trirelay_major15_esm650m_seed101_rtx5070_fp32_b64/history.jsonl |
    ForEach-Object { $_ | ConvertFrom-Json } |
    Select-Object epoch, @{Name='TrainLoss';Expression={$_.losses.total}}, validation_loss,
        @{Name='ValMacroAP';Expression={$_.validation.macro_ap}}, best_epoch, early_stopping_count
```

Train 在线指标复用当轮带增强、Dropout、持续更新权重的前向预测，不额外完整前向，不能当成某个固定模型的完整 Train 评估。Val 每轮在 eval 模式完整前向一次，同一批预测供全局和 CPU 分组统计使用。Val Loss 是相同主分类准则在干净输入下的均值，**不包含随机遮蔽重建等辅助目标**，不能与 Train 总 Loss 直接做数值高低比较。

Epoch 阈值固定 0.5，不逐轮调阈值；checkpoint 仍按 Val Macro-AP → Micro-AP → 更早 epoch。训练结束才对最佳模型按现有 Val Jaccard 策略拟合最终阈值，results.json / best.pt 保持兼容，新增最终 `validation_groups`。固定阈值的历史指标与最终拟合阈值指标明确区分；旧 seed=42 不补造历史，Test 不参与本轮。

实现与数学口径集中在 [下一代模型技术说明](../docs/MFP-Next-Generation-Architecture.md#8-已实现架构与统一实验协议)。三模型共享原始明确骨架数据、固定标签顺序、Train-only 统计和评价协议，保持独立架构。默认 ESM-650M、隐藏维 256、8 heads；major15/fine39 的下游参数分别约为 TriRelay 35.20/35.21M、GeoFactor 31.00/31.01M、ProtoRoute 32.33/32.45M，不含冻结 ESM。

在项目根目录、已激活环境中运行：

```bash
mfp-nova train --model trirelay --task major15 --seed 42 --device cuda:0
mfp-nova train --model geofactor --task major15 --seed 42 --device cuda:0
mfp-nova train --model protoroute --task major15 --seed 42 --device cuda:0
```

把 `--task` 改为 `fine39` 即可切换任务。也可使用 `python -m mfp_nova.cli`。四模型独立配置为 `configs/trirelay.yaml`、`configs/geofactor.yaml`、`configs/protoroute.yaml`、`configs/helixset.yaml`；默认输出 `runs/<model>_<task>_seed<seed>/`，已存在时明确拒绝覆盖。

`--esm 35m` 同时选择 35M 编码器和其缓存；`--feature-root` 可覆盖缓存路径。两种编码器比较请使用不同 `--name`：

```bash
mfp-nova train --model trirelay --task fine39 --esm 35m --seed 42 --device cuda:0 --name trirelay_fine39_esm35m_seed42
mfp-nova train --model trirelay --task fine39 --esm 650m --seed 42 --device cuda:0 --name trirelay_fine39_esm650m_seed42
```

RTX 5070 已完成三模型 × major15/fine39 的 ESM-650M、seed 42 全量训练，统一使用 `--batch-size 64 --precision fp32`。Validation Macro-AP 分别为 TriRelay 0.540361/0.518048、GeoFactor 0.528925/0.506050、ProtoRoute 0.537225/0.514995。完整指标和资源成本见 [实际验证与训练记录](../docs/MFP-Real-Data-GPU-Validation.md#7-rtx-5070-首轮全量训练与下一步对照)。目前尚无三模型的 35M 全量结果。

下一步按下面的五 Seed 协议验证稳定性，35M 架构对照暂后置；不因中途排名改变模型、Loss、数据、Seed 或优化设置。

不修改迁移包、划分或已有缓存。运行新模型需要同一编码器的 Train/Val 残基缓存。训练仅读取 Train/Val，Test 独立执行；只有尚无 Test 缓存时才执行对应提取：

```bash
mfp-nova features --esm 650m --device cuda:0 --splits test
mfp-nova evaluate --checkpoint runs/trirelay_major15_seed42/best.pt --split test --device cuda:0 --output runs/trirelay_major15_seed42/test.json --predictions runs/trirelay_major15_seed42/test_predictions.npz
```

评价默认使用 checkpoint 的缓存路径和 batch，可用 `--feature-root`、`--data-root`、`--batch-size` 覆盖。输出完整指标与结构/来源/置信度/D/长度/单多功能分组。缺可信实验结构来源时不生成虚假实验结构指标。

恢复使用 `last.pt`，输出到新目录；`best.pt` 含最佳模型和 Validation 固定阈值，供评价/推理：

```bash
mfp-nova train --model trirelay --task major15 --resume runs/trirelay_major15_seed42/last.pt --epochs 100 --device cuda:0 --name trirelay_major15_seed42_continued
mfp-nova train --model trirelay --task major15 --ablate feedback high_order --device cuda:0 --name trirelay_major15_no_feedback_no_high_order_seed42
mfp-nova train --model geofactor --task major15 --ablate high_order --device cuda:0 --name geofactor_major15_pairwise_seed42
mfp-nova train --model protoroute --task major15 --ablate shared_prototypes --device cuda:0 --name protoroute_major15_no_shared_prototypes_seed42
```

恢复保留原模型、统计、Loss、seed 和训练协议，仅调整执行位置、设备、总 epoch 与 worker；它继续原来的学习率计划，不偷偷重启计划。其他消融开关见各模型配置；Loss 权重可在现有配置直接修改并用不同实验名称识别。`--threshold-strategy per_label` 是单独的 Val 逐标签阈值实验，主协议保持 global。

完整多模态独立推理使用 NPZ 原始数值输入，预处理从 checkpoint 自动加载：

```bash
mfp-nova predict --checkpoint runs/trirelay_major15_seed42/best.pt --inputs samples.npz --device cuda:0 --output predictions.json
```

`samples.npz` 的必需字段：`esm[B,L,D]`、`aa[B,L]`、`d[B,L]`、`mask[B,L]`、`phys_raw[B,515]`。AA 的 1–20 顺序为 `ACDEFGHIKLMNPQRSTVWY`，0 为 padding；未观察描述符用 NaN。可选字段：`coords[B,L,3]`（Å）、`struct_mask`、`angles_degrees[B,L,3]`、`dssp`（0–8，未知 -1）、`ASA_A2`、`RSA_unclipped`、`confidence_verified`（仅已识别来源的 0–100 置信度）、`struct_global_raw[B,25]`、字符串 `chemical_tokens[B,T]`/`residue_chem_tokens[B,L]`、`structure_source_known[B]`。列顺序遵循迁移包 schema；缺结构必须不设置有效结构 mask。接口为 `multimodal.prepare_inputs` → 模型 `forward`，不需要目标标签。

已有 `collate_multimodal` 导出的 checkpoint 标准化张量 NPZ 也可直接输入。只有序列时可用 `--sequences peptides.txt`，会在线调用冻结 ESM，并将结构/理化标为缺失；输出明确标记 `sequence_and_stereochemistry_only`，不假装已利用完整模态，也不自动推断末端修饰。

每个 run 记录配置、seed、Git commit/dirty、数据和编码器标识、参数量、最佳 epoch、完整指标、Loss 分项、LR、耗时及 CUDA 峰值分配显存。分批多 Seed 运行方式见下方快速使用，不自动占用整晚或全天。

汇总不启动训练、不按 Test 选模型。各 run 的 Test 文件统一命名 `test.json`：

```bash
mfp-nova summarize --runs runs/trirelay_major15_seed42 runs/trirelay_major15_seed101 runs/trirelay_major15_seed202 --split val --output results/trirelay_major15_val.json
mfp-nova summarize --runs runs/trirelay_major15_seed42 runs/geofactor_major15_seed42 runs/protoroute_major15_seed42 --split test --output results/major15_test_comparison.json
```

同任务/编码器/数据队列/配置协议的 seed 汇总 mean ± 样本 std；不同模型、任务、消融、Loss、阈值协议分别列出，不混合平均。只有一个 seed 时 std=NA/null，并记录有效 seed 数。支持传统指标、核心指标、逐标签指标及参数/显存/时间；诊断子集不进入正式汇总。

必要的 CPU 流程检查可用 `--device cpu --limit-samples 16 --epochs 1 --name <unique_name>`，需要实际缓存；结果标记为 diagnostic subset，不能报告为正式性能。本轮相关检查见 `tests/test_research.py`：两标签空间、缺结构、D 输入、padding、多原型、关键模块梯度、训练恢复、输入预处理和独立推理。未自动占用服务器 GPU，正式多 seed、GPU 资源与模型增益尚待实验。

## 两台服务器的真实缓存 GPU 短训练

实际训练在 RTX 5070 和 V100 上运行，本机只做开发与 CPU 检查。先查看 `nvidia-smi`，确认当前项目没有运行中的训练后再 `git pull --ff-only`；仅使用已分配且空闲的 GPU。更新代码后 `python -m pip install --no-deps -e .`，不升级服务器 Torch/CUDA。

先检查所选 GPU 和缓存元数据，不训练、不下载、不重建特征：

```bash
mfp-nova gpu-check --esm 650m --device cuda:0 --inspect-only
```

RTX 5070（已激活 `MFP-Nova`，项目 `D:\Data\ChatGPT\mfp\MFP-Nova`）：

```bash
mfp-nova gpu-check --esm 650m --device cuda:0 --tasks major15 fine39 --batch-sizes 2 4 8 16 --precisions fp32 fp16 --steps 32 --samples 128 --output runs/gpucheck_rtx5070_20261015
```

V100（密码 SSH 登录后，`conda activate MFP-Nova`，项目 `~/wuzhencheng/projects/MFP-Nova`）：

```bash
mfp-nova gpu-check --esm 650m --device cuda:0 --tasks major15 fine39 --batch-sizes 2 4 8 16 --precisions fp32 fp16 --steps 32 --samples 128 --output runs/gpucheck_v100_20261015
```

V100 不用 BF16。若空闲且已分配的 GPU 不是 0，替换 `--device`；设置了 `CUDA_VISIBLE_DEVICES` 时遵循其映射。该命令检查其他计算进程，遇到占用时退出，不抢占、不终止他人任务。RTX 不使用 SSH。

当前默认依次执行 TriRelay → GeoFactor → ProtoRoute → HelixSet，每模型覆盖 major15/fine39；本节原有实测记录仅覆盖当时执行的模型。复用服务器已有真实缓存，读出 ID/骨架与缓存 row_id 对齐；选取结构存在/缺失、非甘氨酸 D、不同长度及标签覆盖的少量原始样本。模型深度、专家、反馈及因子全部保留。诊断优化子集不会重拟合缩小标签先验：标准化、长尾分层、高阶关系和原型容量继续来自完整原始 Train。

每个 batch/精度先 warmup，再计时 3 step，记录参数量、峰值 allocated/reserved 显存、前向/反向/optimizer 时间、含数据准备的 step 时间与吞吐、按代表 batch 粗估的训练 epoch 时间、padding/缺结构输出和关键模块梯度/更新。OOM 只记录失败批量并停止继续增大该精度，不删模块。Profiler 的代表 batch 和计时开销会影响估计，不能替代正式整轮吞吐。

随后使用成功且关键模块有梯度更新的实测配置，运行 32 个 optimizer step、Diagnostic Validation、checkpoint，再恢复 2 step，并从恢复的最佳模型独立计算 Diagnostic Validation。小样本指标不用于性能结论。`summary.json`、各模型任务的 `*_profile.json`、`history.jsonl` 和 checkpoint 都保存在指定的新目录；目录存在则拒绝覆盖。不会读取 Test 缓存或启动正式多 seed。

若 V100 尚无当前 MFP 数据包/ESM 缓存，先通过 UU 回传到本机，再用密码认证 SFTP/SCP 转移现有文件；也可指定 `--data-root`、`--feature-root` 指向当前项目的数据和缓存。不能把旧项目缓存、随机特征当成这次真实验证。35M 可用 `--esm 35m` 单独执行，使用不同输出目录。

正式训练入口另支持：

```bash
mfp-nova train --model trirelay --task major15 --precision fp16 --batch-size 8 --accumulation-steps 2 --device cuda:0 --name trirelay_major15_esm650m_fp16_seed42
```

`--max-steps` 是诊断上限，按 optimizer 尝试步计数，AMP 跳过的更新单独记录；含此参数的 run 不进入正式比较。恢复诊断运行可增大 `--max-steps`，从下一轮开始继续，不承诺部分 epoch 内逐样本精确续接。最终 batch/精度应依据两台服务器各自实测选择，不把另一台或本机的结果代入。

## 下一阶段候选：辅助目标日程与独立功能决策

以下为尚待真实 Validation 对比的可选算法，默认训练和历史 checkpoint 行为不变。保留三个完整模型，不新增架构层数。既有 `data.augment=false` 关闭迁移包增强样本，但训练内的模态 dropout、理化/结构特征 masking 与重建仍由 Loss 配置控制；本次不重复增加这些扰动，也不修改序列、D/L 或结构真值。

训练候选仅对重建辅助目标增加 warmup 后余弦衰减：`--reconstruction-decay 5 20 --reconstruction-floor 0.2` 表示第 5 轮保持原权重，第 20 轮降至原权重的 20%，之后保持。原重建权重 0.05 最终为 0.01；其他辅助目标、分类 Loss、架构、checkpoint 选模规则不变。动机是训练后期 Val 平台与较高辅助 Loss 占比；占比不能证明梯度冲突，收益必须实测。实际权重记录在每轮 `loss_settings` 中，恢复训练读取原 checkpoint 日程。

先仅导出已有 TriRelay / major15 / seed42 的完整 Val 概率（服务器旧版已经支持此命令，无需为了导出更新正在运行的项目）：

```powershell
Set-Location "D:\Data\ChatGPT\mfp\MFP-Nova"
$mfpPython = "D:\App\Code\Anaconda\envs\MFP-Nova\python.exe"
$baseline = "runs/trirelay_major15_esm650m_seed42_rtx5070_fp32_b64_20261015_105336"
$tag = Get-Date -Format "yyyyMMdd_HHmmss"
& $mfpPython -m mfp_nova.cli evaluate --checkpoint "$baseline/best.pt" --split val --device cuda:0 --batch-size 64 --output "$baseline/val_decision_$tag.json" --predictions "$baseline/val_predictions_$tag.npz"
```

回传新生成的 JSON 和 NPZ 即可，不必传模型、ESM 缓存或 Test。NPZ 含原始概率、目标、标签顺序、row_id 和已保存阈值。独立功能决策实验可在本机 CPU 上运行：

```powershell
.\.venv\Scripts\python.exe -m mfp_nova.decisions --predictions "实际回传的val_predictions文件.npz" --task major15 --output "runs/analysis/decision_major15_seed42_唯一名称"
```

实验比较原保存阈值、全局阈值扫描，以及保持最强活跃独立功能、提高其他功能证据要求的决策。沿用独立功能映射，抗菌亚型和泛抗微生物不重复计数。所有规则保留原概率，所以 AP/AUROC 不变，可能改变 MFP-F1 和传统五指标。输出 `candidates.csv`、`results.json`、`decision.json`；候选默认允许 Jaccard 下降最多 0.005（0.5 个百分点），在此范围最大化 MFP-F1，同时报告 Pareto 候选；这只是显式权衡参数，不是已证明的最优设置。完整 Val 拟合结果有选择乐观性；按簇划分的两折决策交叉拟合仅检查额外决策搜索，checkpoint 和原阈值已经用全 Val 选择，因此也不是独立泛化证据。默认不采用候选，不使用 Test 调参。

保留候选后，研究模型的 `evaluate` / `predict` 可通过 `--decision <decision.json>` 读取其决策；不改原 checkpoint。无该参数时完全沿用原阈值。

以下为已完成的单 Seed 重建日程对比命令，保留用于复现。本次候选未获整体收益，不建议直接扩展到多 Seed；如主动复现，在 RTX 5070 空闲且更新代码后单独运行（不使用原五 Seed 批量脚本）：

```powershell
$tag = Get-Date -Format "yyyyMMdd_HHmmss"
$candidate = "trirelay_major15_reconstruction_decay_seed42_$tag"
& $mfpPython -m mfp_nova.cli train --model trirelay --task major15 --esm 650m --seed 42 --device cuda:0 --precision fp32 --batch-size 64 --reconstruction-decay 5 20 --reconstruction-floor 0.2 --name $candidate
& $mfpPython -m mfp_nova.cli summarize --runs $baseline "runs/$candidate" --split val --output "runs/analysis/reconstruction_comparison_$tag.json"
```

保持原最大 epoch 与学习率日程，以便和既有 seed42 比较。新日程单列协议，不进入原固定五 Seed 汇总。必须同时观察 Macro-AP、MFP P/R/F1、五指标及逐标签变化；单 Seed 的小幅上涨不能宣称稳定提升。本轮旧 Test 已被分析，不作为候选选择依据。

修改前原文件保存在仓库外 `D:\Data\Code\MFP-Nova-originals\20261018-generalization`：`src/mfp_nova/{research,metrics,cli}.py`，原 commit 为 `8c7eac346b4ba6f23a24701e58941941b06b0e48`。需要恢复时，在无运行任务且保存后续工作后，将对应原文件复制回同名路径；其余正常开发历史由 Git 管理。本轮未修改三个模型架构文件。

2026-10-18 已完成真实 Val 决策实验：TriRelay / major15 / seed42，共 8,534 条，输入为 `val_predictions_seed42_20261018_114244.zip`。本地输出位于 `runs/analysis/decision_major15_seed42_20261018/study`，完整候选在 `candidates.csv`，原阈值与默认权衡候选在 `results.json`，零 Jaccard 下降预算的补充诊断在 `strict_jaccard_crossfit.json`。

| 指标 | 原阈值 0.68 | 全局阈值 0.61 | 主阈值 0.68 / 其他独立功能阈值 0.69 |
| --- | ---: | ---: | ---: |
| Macro-AP | 0.540361 | 0.540361 | 0.540361 |
| MFP Precision | 0.471551 | 0.442645 | 0.480932 |
| MFP Recall | 0.543909 | 0.619452 | 0.535883 |
| MFP-F1 | 0.505152 | 0.516332 | 0.506923 |
| Accuracy / Jaccard | 0.616967 | 0.612756 | 0.617346 |
| Precision | 0.665093 | 0.655133 | 0.666171 |
| Coverage | 0.690116 | 0.712635 | 0.687749 |
| Absolute True | 0.498242 | 0.474807 | 0.500469 |
| Absolute False（低为好） | 0.064065 | 0.067659 | 0.063604 |
| Micro-F1 | 0.643729 | 0.639770 | 0.644454 |

默认 0.005 Jaccard 预算选择全局 0.61；MFP 真阳性增加 160、假阳性增加 361，AT 下降 2.34 个百分点，不建议直接替换主协议。零下降预算选择功能阈值 0.69；减少 66 个假阳性，同时损失 17 个真阳性，MFP-F1 仅提高 0.177 个百分点，保留为低成本候选，不能宣称稳定提升。后者另存 `decision_balanced.json`，未修改默认阈值或已有权重。两种预算各自的两个簇分组拟合均选择相同参数，因此各自交叉拟合汇总恰好等于全 Val 拟合结果；这不消除既有 checkpoint / 原阈值全 Val 选择的偏差。固定决策的 2,000 次配对簇重采样中，该功能规则的 MFP-F1 增益 95% 百分位区间为 −0.170～+0.477 个百分点，跨零，且不校正候选选择偏差。结论限于此模型、任务与 Seed。

2026-10-18 RTX 5070 实际重建日程训练已完成：`trirelay_major15_reconstruction_decay_seed42_20261018_200703`。25 个连续 epoch、15,425 个 optimizer step，Loss 有限，第 10 轮最佳，连续 15 轮无改善正常早停。参数量 35,196,329、峰值 allocated 显存约 6.62 GiB，均与原实验相同。Train/Val 为 39,478/8,534，数据标识、ESM 元信息、标签顺序、模型/Loss 配置和主要训练参数一致，仅配置新增重建衰减两个选项。实际系数第 5/10/15/20 轮分别为 0.05/0.04/0.02/0.01。

| 最佳 checkpoint 的 Val 指标 | 原 seed42 | 重建衰减 seed42 | 差值（百分点） |
| --- | ---: | ---: | ---: |
| Macro-AP | 0.540361 | 0.538387 | −0.197 |
| Micro-AP | 0.698735 | 0.686636 | −1.210 |
| Macro-F1 | 0.533829 | 0.510739 | −2.309 |
| Micro-F1 | 0.643729 | 0.629660 | −1.407 |
| MFP-F1 | 0.505152 | 0.489683 | −1.547 |
| Accuracy / Jaccard | 0.616967 | 0.587320 | −2.965 |
| Precision | 0.665093 | 0.632738 | −3.235 |
| Coverage | 0.690116 | 0.679136 | −1.098 |
| Absolute True | 0.498242 | 0.453363 | −4.488 |
| Absolute False（低为好） | 0.064065 | 0.066971 | +0.291 |

以上两组各自使用 Val 拟合阈值（原 0.68、候选 0.71），不能拿训练日志的固定 0.50 阈值指标替代最终结果。MFP 真阳性 1,152→1,151，假阳性 1,291→1,432，未改善多功能识别。候选训练后期 online Train Macro-AP 到 0.9697，Val 约 0.52，分类训练 Loss 仍下降而干净 Val 分类 Loss 回升；降低总 Loss 没有消除泛化平台。总体不采用此日程，不扩大实验。部分 AP 上升（drug_delivery +4.61、antiparasitic +4.14 个百分点），但 cell_communication −5.36、growth_regeneration −4.46；major15 的 Tail-AP 仅对应 thrombolytic 一个标签，本次亦下降。

归因边界：原代码 commit `0cbf428`、候选 `88c1ba9`，期间增加指标记录与恢复逻辑；所查 TriRelay 前向与 Loss 公式未变。第 5 轮尚未开始衰减，Val Macro-AP 已为 0.457015 vs 0.464487；现有 seed 设置未启用强制确定性，差异原因未定位，不能把全程差异严格归因于日程。原实验 35 轮/78.66 分钟，候选 25 轮/58.77 分钟，耗时减少来自更早停止，不是吞吐提升证明。结果与图保存于本地 `runs/analysis/reconstruction_decay_seed42_20261018/`。本轮未使用 Test；保留原训练默认值与权重，结束这两个候选的首轮验证，尚无足够证据替换原正式方案。

## 下一项候选：同次训练的 EMA 权重对照

重建日程实验停止扩展后，新增可选 EMA（指数移动平均）候选。研究假设是对更新中的参数做平均可能缓解 Val 波动，尚未证明适用于本数据。它不修改模型架构、Loss、数据增强、学习率或默认配置，不与重建衰减叠加。使用 [PyTorch AveragedModel / EMA](https://docs.pytorch.org/docs/2.8/optim.html#weight-averaging-swa-and-ema)，第一次成功更新复制在线权重，后续按 `ema = decay * ema + (1-decay) * online` 更新。现有模型的 buffers 是先验和索引，直接同步，不做平均；无 BatchNorm 统计重估。EMA 不参与反向，也不产生伪标签。

固定首个实验：TriRelay / major15 / seed42 / FP32 / batch64，`decay=0.999`，第 6 轮起启用；原重建权重仍为 0.05。EMA 只计成功 optimizer step，不计梯度累积微批次和 AMP 跳步。在每轮同时评价在线权重与 EMA，额外的验证迭代隔离 CPU RNG，避免 DataLoader 基础种子消耗干扰后续训练。CPU 测试检查在线轨迹与无 EMA 时一致，不宣称 GPU 位级重现。

- 常规 `Val` 行：预热期为在线权重，EMA 启用后为平均权重。仍按 Macro-AP → Micro-AP → 更早 epoch 选模与早停。
- 新增 `Weights / OnlineVal` 行：标记当前权重来源、EMA 更新次数，以及在线权重的 AP、MFP-F1 和五指标。
- `best.pt`：主分支最佳权重，`weight_source` 明确标注 `ema` 或 `online_warmup`；`online_best.pt`：同次训练在线权重的独立最佳 checkpoint。两者分别在 Val 拟合阈值，均可用原 evaluate/predict 入口。
- `last.pt`：保存在线模型、优化器、EMA 状态、两组最佳状态与 RNG；恢复继续同一日程。不能在旧非 EMA checkpoint 上临时添加 EMA 并伪装成同一实验。
- `results.json` 的顶层 `validation` 为主分支，`online_reference.validation` 为在线对照；`history.jsonl` 保存逐轮两组指标。这个对照共享训练轨迹和主分支停止时刻，不能当作独立训练的第二个 Seed。

额外开销为一份平均模型、一次每步参数平均以及 EMA 启用后每轮额外一次 Val 前向，无额外训练反向。RTX 5070 的首轮实测见下。新实验自动与原五 Seed 协议分开，原结果不覆盖。

RTX 5070 空闲、无运行中训练时，在 PowerShell 执行一次：

```powershell
Set-Location "D:\Data\ChatGPT\mfp\MFP-Nova"
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw "Git 更新失败" }
$mfpPython = "D:\App\Code\Anaconda\envs\MFP-Nova\python.exe"
$tag = Get-Date -Format "yyyyMMdd_HHmmss"
$run = "trirelay_major15_ema_seed42_$tag"
& $mfpPython -m mfp_nova.cli train --model trirelay --task major15 --esm 650m --seed 42 --device cuda:0 --precision fp32 --batch-size 64 --ema-decay 0.999 --ema-start-epoch 6 --name $run
if ($LASTEXITCODE -ne 0) { throw "训练失败，请保留输出；可使用新目录从 last.pt 恢复" }
$zip = Join-Path ([Environment]::GetFolderPath("Desktop")) "ema_results_$tag.zip"
Compress-Archive -LiteralPath "runs/$run/metadata.json","runs/$run/results.json","runs/$run/history.jsonl" -DestinationPath $zip
Write-Host "回传：$zip"
```

先比较该单次训练内两组最佳 checkpoint 的 AP、MFP P/R/F1、五指标、逐标签及分组结果，再参考历史 seed42；不以单一指标的小幅变化决定采用，不使用 Test 调参。本轮修改前原始 `research.py`、`cli.py` 已保存在仓库外 `D:\Data\Code\MFP-Nova-originals\20261018-ema`，对应 commit `2830405`；恢复方式与前节相同。模型架构文件未修改。

本地相关 8 项检查通过：成功步平均/跳步/先验 buffer、在线训练轨迹、完整 epoch 边界恢复、两组权重独立评价、终止后恢复、三模型原有恢复接口及固定五 Seed 隔离等。真实 ESM-650M 缓存另完成完整 35,196,329 参数 TriRelay 的两步 CPU 诊断，生成 `best.pt` 与 `online_best.pt`，EMA 更新次数为 2；仅验证接口，未作为性能证据。该诊断为了覆盖平均操作从第 1 轮启用 EMA，正式候选仍从第 6 轮开始。RTX 5070 首轮真实训练结果如下，多 Seed 收益尚待验证。

### EMA 首轮真实 Validation 结果（2026-10-18）

回传文件 `ema_results_20261018_215416.zip`，运行名 `trirelay_major15_ema_seed42_20261018_215416`，代码 `5ac5c7f`。完整 Train/Val 为 39,478/8,534；原重建系数全程保持 0.05。34 轮连续历史、20,978 个 optimizer step、17,893 次 EMA 更新；Loss 有限，FP32 无跳步。EMA 在第 19 轮最佳，之后 15 轮无改善正常早停；同次在线权重最佳为第 10 轮。两组分别使用保存的 Val 阈值 0.72/0.70，历史原 seed42 阈值为 0.68。

| 最佳 checkpoint 的 Val 指标 | 历史原 seed42 | 同次在线权重 | EMA | EMA − 在线（百分点） |
| --- | ---: | ---: | ---: | ---: |
| Macro-AP | 0.540361 | 0.535077 | 0.561444 | +2.637 |
| Micro-AP | 0.698735 | 0.690731 | 0.711217 | +2.049 |
| Macro-F1 | 0.533829 | 0.517180 | 0.551003 | +3.382 |
| Micro-F1 | 0.643729 | 0.638206 | 0.661585 | +2.338 |
| Accuracy / Jaccard | 0.616967 | 0.597955 | 0.633324 | +3.537 |
| Precision | 0.665093 | 0.639309 | 0.679996 | +4.069 |
| Coverage | 0.690116 | 0.687052 | 0.697060 | +1.001 |
| Absolute True | 0.498242 | 0.468245 | 0.522498 | +5.425 |
| Absolute False（低为好） | 0.064065 | 0.065542 | 0.060003 | −0.554 |
| MFP Precision | 0.471551 | 0.464519 | 0.509761 | +4.524 |
| MFP Recall | 0.543909 | 0.571766 | 0.554769 | −1.700 |
| MFP-F1 | 0.505152 | 0.512593 | 0.531314 | +1.872 |

EMA 在启用后的 29/29 轮 Macro-AP 均高于同轮在线权重，15 个标签中 13 个 AP 高于在线最佳；这支持继续研究参数平均，但相邻 epoch 相关，不能当作 29 次独立实验。主要增益包括 growth_regeneration +9.51、thrombolytic +6.75、cell_communication +5.28、anticancer +4.32 个百分点；drug_delivery −3.08、antiparasitic −0.64。相对历史原 seed42，14/15 个标签 AP 上升，antibacterial 下降 1.39 个百分点。Head/Medium/Tail AP 为 0.615653/0.477824/0.216007；major15 Tail 仅一个 thrombolytic 标签，Val 正例仅 37 条，不能推广为全面长尾改善。

MFP 相对同次在线最佳，按现有标注统计的假阳性 1,396→1,130，真阳性 1,211→1,175，精确率提高而召回下降；不能把未标注的假阳性解释为实验确认阴性。Macro-AUROC 0.896761→0.893140 也略降。结构缺失组 1,040 条的 Macro-AP 0.53108→0.56440，有结构组 7,494 条 0.53462→0.55950，收益不只来自有结构样本。训练日志按固定 0.50 阈值监控，不能与上表拟合阈值后的结果混用。

实际参数量仍为 35,196,329，峰值 allocated 显存 6.756 GiB，较历史原 seed42 多约 134 MiB；完整运行 88.40 分钟，平均每轮约 155.3 秒。包含双分支 Val 与收尾评价，不能将不同停止轮数的历史总时长直接解释为 EMA 吞吐开销。此次只检查回传配置、结果和历史，未在本机重新加载这次 GPU checkpoint。

结论：保留 EMA 为优先候选，默认正式配置暂不替换。下一步固定同一组参数，只增加 major15 的 seed101、202，每次仍保存同次在线最佳作为对照；不叠加重建衰减、不搜索 decay、不使用 Test。若多个 Seed 都有相近的 AP、五指标及 MFP 改善，再决定补齐五 Seed 或扩展 fine39。当前证据是一个 Seed 的 Validation 改善，不能宣称统计显著或独立 Test 提升。原始回传、衍生统计和图存于本机 `runs/analysis/ema_seed42_20261018/`（不提交大型运行产物）。

### EMA 五 Seed Validation 汇总（2026-10-19）

新增 seed101/202/303/404 回传与原 EMA seed42 合并，共五次完整 TriRelay / major15 / ESM-650M 实验。除 seed 外配置一致，标签顺序、缓存元信息及 Train/Val 39,478/8,534 一致。新四次 commit `f4dfe0e` 与 seed42 的 `5ac5c7f` 仅 README 不同。历史连续、Loss 有限，全部以 EMA 最佳后 15 轮无改善正常早停；没有发现回传记录中的训练错误。此次未重新加载服务器 checkpoint。

主比较为每次训练内独立选出的在线最佳与 EMA 最佳，均按 Macro-AP / Micro-AP 选择并各自在 Val 拟合阈值；二者共享训练轨迹和 EMA 控制的停止时刻。下表为五 Seed 均值 ± 样本标准差（ddof=1）；这表示随机种子间波动，不是置信区间。

| Val 指标 | 同次在线最佳 | EMA 最佳 | 平均差值（百分点） | 改善 Seed 数 |
| --- | ---: | ---: | ---: | ---: |
| Macro-AP | 0.5324 ± 0.0018 | 0.5598 ± 0.0027 | +2.735 | 5/5 |
| Micro-AP | 0.6929 ± 0.0076 | 0.7177 ± 0.0061 | +2.482 | 5/5 |
| Macro-F1 | 0.5157 ± 0.0016 | 0.5453 ± 0.0052 | +2.952 | 5/5 |
| Micro-F1 | 0.6403 ± 0.0055 | 0.6591 ± 0.0044 | +1.880 | 5/5 |
| Accuracy / Jaccard | 0.6059 ± 0.0073 | 0.6253 ± 0.0082 | +1.943 | 4/5 |
| Precision | 0.6540 ± 0.0107 | 0.6740 ± 0.0071 | +1.993 | 4/5 |
| Coverage | 0.6792 ± 0.0084 | 0.6984 ± 0.0052 | +1.925 | 5/5 |
| Absolute True | 0.4864 ± 0.0155 | 0.5046 ± 0.0141 | +1.826 | 4/5 |
| Absolute False（低为好） | 0.0640 ± 0.0017 | 0.0606 ± 0.0008 | -0.342 | 5/5 |
| MFP Precision | 0.4807 ± 0.0293 | 0.4863 ± 0.0156 | +0.558 | 4/5 |
| MFP Recall | 0.5358 ± 0.0209 | 0.5457 ± 0.0147 | +0.992 | 4/5 |
| MFP-F1 | 0.5061 ± 0.0154 | 0.5142 ± 0.0132 | +0.809 | 4/5 |
| Head-AP | 0.5955 ± 0.0026 | 0.6223 ± 0.0043 | +2.681 | 5/5 |
| Medium-AP | 0.4382 ± 0.0142 | 0.4632 ± 0.0126 | +2.503 | 5/5 |
| Tail-AP | 0.1216 ± 0.0193 | 0.1619 ± 0.0365 | +4.021 | 4/5 |

| Seed | 在线 Macro-AP | EMA Macro-AP | EMA 最佳 / 完成 epoch | 在线 / EMA 阈值 | 耗时（分钟） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.535077 | 0.561444 | 19 / 34 | 0.70 / 0.72 | 88.40 |
| 101 | 0.533706 | 0.562755 | 13 / 28 | 0.69 / 0.72 | 71.64 |
| 202 | 0.531360 | 0.557016 | 12 / 27 | 0.67 / 0.73 | 68.71 |
| 303 | 0.530830 | 0.556867 | 15 / 30 | 0.73 / 0.71 | 76.76 |
| 404 | 0.531274 | 0.560895 | 12 / 27 | 0.72 / 0.71 | 69.53 |

Macro-AP 每 Seed 提升 2.566～2.962 个百分点，配对增益为 2.735 ± 0.184 个百分点；Micro-AP、Macro/Micro-F1 全部改善。新增四个 Seed 单独的 Macro-AP 平均增益约 2.759 个百分点，支持 seed42 发现可重复。EMA Macro-AP 的跨 Seed 标准差 0.002692 高于在线 0.001849，不能声称所有稳定性指标都改善。

传统五指标均值全部改善，但 seed202 的 Jaccard/Precision/Absolute True 分别下降 0.120/0.160/0.762 个百分点；Coverage 和 Absolute False 在五 Seed 全部改善。MFP-F1 仅 4/5 改善，平均增益 0.809 ± 1.336 个百分点：seed303 下降 1.259 个百分点，对应真阳性 +35、假阳性 +203、Precision −3.922、Recall +1.653 个百分点。因此 EMA 的主要证据在排序与总体标签预测，不能宣称稳定解决多功能误报。MFP 假阳性仍是相对于现有标注的统计，不代表实验确认阴性。

15 个标签的平均 AP 全部增加，其中 10 个在五 Seed 全部改善。均值增益最大的包括 growth_regeneration +5.033、metabolic_regulatory +4.685、anticancer +4.139 个百分点；antiparasitic 仅 +0.037。Tail-AP 平均提升 4.021 个百分点，但仅对应 thrombolytic（Val 正例 37），seed303 略降，不能推广为普遍长尾改善。结构存在组 Macro-AP 0.52780→0.55690，缺失组 0.55395→0.57349，两组均 5/5 改善。

新四次累计 4.777 小时（约 4 小时 47 分钟），峰值 allocated 显存 6.756～6.763 GiB，参数量仍 35,196,329。原始回传和完整衍生统计位于 `runs/analysis/ema_five_seed_20261019/`，对比图为 `comparison.png`。本轮仅分析与文档更新，没有训练或修改算法。

下一步将 EMA 保留为 TriRelay / major15 的优先训练候选，并固定相同 decay=0.999、start_epoch=6，对 fine39 先做 seed42 的同次在线/EMA 对照，确认另一标签空间的适用性；不叠加其他优化或搜索阈值策略。major15 的 MFP 短板单独保留，不能因总体 AP 改善而忽略。结果仍来自同一个反复用于模型开发的 Validation 集，五 Seed 不等于五个独立数据集；不新增 Test 调参，不将此均值宣称为 Test 性能。

### TriRelay / fine39 五 Seed EMA 结果（2026-10-19）

本次回传五个 `ema_results_fine39_seed*_20261019_090117.zip`，代码均为 `18b0eb3`、工作树干净，除 seed 外配置一致。完整模型 35,209,345 个可训练参数，FP32 / batch64 / ESM-650M，EMA decay=0.999、start_epoch=6；原重建权重 0.05。实际 Train/Val 为 39,455/8,529，与本地 fine39 原始任务 mask 加 canonical_available 的计数一致；不同于 major15 的 39,478/8,534，并非本次丢失缓存。ESM 缓存覆盖更大的序列集合，任务按现有 row_id 与 mask 取子集。未修改划分或读取 Test 评价。

五组历史连续、Loss 有限，均在 EMA 最佳轮之后 15 轮正常早停。31/32/34/31/32 个 epoch，总计 7.001 小时；峰值 allocated 显存 7.060～7.068 GiB。此次检查是回传配置、结果与历史分析，未在本机独立加载这批 GPU checkpoint。

| Seed | 在线最佳 Macro-AP | EMA 最佳 Macro-AP | 差值（百分点） | EMA 最佳 / 完成轮 | 在线 / EMA 阈值 | 分钟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.514691 | 0.535915 | +2.122 | 16 / 31 | 0.71 / 0.65 | 81.69 |
| 101 | 0.516347 | 0.543796 | +2.745 | 17 / 32 | 0.68 / 0.64 | 84.64 |
| 202 | 0.513547 | 0.543223 | +2.968 | 19 / 34 | 0.62 / 0.68 | 87.71 |
| 303 | 0.528397 | 0.553055 | +2.466 | 16 / 31 | 0.66 / 0.67 | 82.49 |
| 404 | 0.516602 | 0.543351 | +2.675 | 17 / 32 | 0.67 / 0.66 | 83.53 |

下表为五 Seed 均值 ± 样本标准差（ddof=1），不是置信区间。主对照为同次训练的独立在线最佳 checkpoint 与 EMA 最佳 checkpoint，均以 Val Macro-AP / Micro-AP 选模；停止时刻由主分支控制，在线对照不等同于独立训练的另一组 Seed。阈值各自在 Val 最大化样本 Jaccard，不能与日志固定 0.50 阈值直接比较。

| Val 指标 | 在线最佳 | EMA 最佳 | 平均差值（百分点） | 改善 Seed 数 |
| --- | ---: | ---: | ---: | ---: |
| Macro-AP | 0.5179 ± 0.0060 | 0.5439 ± 0.0061 | +2.595 | 5/5 |
| Micro-AP | 0.6268 ± 0.0050 | 0.6541 ± 0.0087 | +2.736 | 5/5 |
| Macro-F1 | 0.5114 ± 0.0065 | 0.5345 ± 0.0053 | +2.314 | 5/5 |
| Micro-F1 | 0.5903 ± 0.0031 | 0.6045 ± 0.0082 | +1.428 | 5/5 |
| Accuracy / Jaccard | 0.5398 ± 0.0063 | 0.5563 ± 0.0085 | +1.651 | 5/5 |
| Precision（样本平均） | 0.5830 ± 0.0088 | 0.5965 ± 0.0115 | +1.346 | 5/5 |
| Coverage（样本平均 Recall） | 0.6287 ± 0.0128 | 0.6547 ± 0.0136 | +2.597 | 5/5 |
| Absolute True | 0.4010 ± 0.0130 | 0.4106 ± 0.0189 | +0.961 | 4/5 |
| Absolute False（低为好） | 0.0339 ± 0.0008 | 0.0331 ± 0.0014 | -0.082 | 4/5 |
| MFP Precision | 0.4657 ± 0.0224 | 0.4557 ± 0.0192 | -1.000 | 2/5 |
| MFP Recall | 0.5324 ± 0.0458 | 0.5527 ± 0.0386 | +2.021 | 3/5 |
| MFP-F1 | 0.4959 ± 0.0227 | 0.4983 ± 0.0078 | +0.248 | 3/5 |
| Head-AP（13 标签） | 0.5669 ± 0.0043 | 0.5909 ± 0.0064 | +2.405 | 5/5 |
| Medium-AP（8 标签） | 0.6080 ± 0.0110 | 0.6327 ± 0.0086 | +2.465 | 5/5 |
| Tail-AP（18 标签） | 0.4425 ± 0.0083 | 0.4704 ± 0.0103 | +2.790 | 5/5 |
| Macro-AUROC | 0.9016 ± 0.0084 | 0.9149 ± 0.0035 | +1.331 | 5/5 |
| Micro-AUROC | 0.9572 ± 0.0019 | 0.9625 ± 0.0013 | +0.526 | 5/5 |
| Macro-MCC | 0.4972 ± 0.0075 | 0.5200 ± 0.0060 | +2.278 | 5/5 |

结论分层：Macro/Micro-AP、Macro/Micro-F1 和样本 Jaccard 全部 5/5 改善；Macro-AP 配对增益 2.595 ± 0.319 个百分点，与 major15 的 2.735 ± 0.184 接近，支持同一组 EMA 设置可用于两个标签空间。不能因此宣称它对另两模型同样有效，也不能将两个任务当作十个独立数据集。EMA 的跨 Seed Macro-AP 标准差 0.00609，在线为 0.00599，并未普遍降低指标波动。

传统五指标均值全部改善，Jaccard、Precision、Coverage 为 5/5；seed42 的 Absolute True 下降 1.395 个百分点、Absolute False 增加 0.110 个百分点，同时 Coverage 增加 5.024 个百分点。该 Seed 在线/EMA 阈值为 0.71/0.65，预测集合发生精确率/召回权衡，不能把降低阈值本身当作错误或证明校准改善。AP 的提升不依赖阈值，表明排序收益确实存在。

MFP-F1 平均仅 0.495864→0.498349（+0.248 ± 2.139 个百分点），没有一致改善证据：seed42 +0.005、101 +2.907、202 +1.857、303 −1.264、404 −2.262 个百分点。平均 MFP Precision 下降 1.000 个百分点，Recall 增加 2.021 个百分点；每 Seed 平均真阳性 +42.6、假阳性 +107.8。seed42 真阳性 +116、假阳性 +337，F1 几乎不变；seed404 真阳性 −116、假阳性 −128，但精确率仍略降，召回明显下降。全局阈值优化 Jaccard，不是独立功能检测 F1；fine39 具体子类预测改善不必然改善独立功能数判断。`__unknown` 仍表示亚类未知，不能把预测标签数直接视为独立功能数，未标注也不是实验确认阴性。

39 个标签平均 AP 中 35 个上升、4 个下降，其中 16 个在五 Seed 全部上升。

| 标签 | Val 正例数 | 在线平均 AP | EMA 平均 AP | 差值（百分点） | 改善数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| analgesic | 18 | 0.4242 | 0.4896 | +6.539 | 5/5 |
| antihcv | 104 | 0.9145 | 0.9676 | +5.308 | 5/5 |
| anticervicalcancer | 47 | 0.6060 | 0.6591 | +5.307 | 4/5 |
| growth_regulatory | 36 | 0.1458 | 0.1911 | +4.524 | 4/5 |
| quorum_sensing | 30 | 0.6319 | 0.6755 | +4.357 | 5/5 |
| antiviral__unknown | 473 | 0.5334 | 0.5744 | +4.101 | 5/5 |
| antitrypanosomic | 12 | 0.3897 | 0.3735 | -1.615 | 2/5 |
| thrombolytic__unknown | 37 | 0.1466 | 0.1340 | -1.260 | 3/5 |
| antiparasitic__unknown | 81 | 0.5192 | 0.5149 | -0.434 | 3/5 |
| antioxidant__unknown | 235 | 0.2264 | 0.2248 | -0.162 | 3/5 |

Head/Medium/Tail 使用 Train 正例数 ≥1000、200～999、<200 的固定分层。fine39 Tail 包含 18 标签，五 Seed 全部提升，证据覆盖面比 major15 单个尾类更广；但抗炎 antiinflammatory AP 仍仅 0.0195，thrombolytic__unknown 为 0.1340，不能声称尾类问题已经解决。少样本标签的 AP 波动和异常高分需结合支持度解释；此轮没有额外开展数据审计。

有结构组 7,489 条 Macro-AP 0.51694→0.54323，缺结构组 1,040 条 0.49616→0.51573，两组均 5/5 改善。但缺结构组 MFP-F1 平均下降 0.849 个百分点；D_annotation 组 173 条 AP 平均提高、MFP-F1 下降 2.122 个百分点，且只有 21 个标签有正例，不能与完整 39 标签宏平均直接比较。结构证据的排序利用改善，不等于所有子群的多功能决策改善。

收敛方面，EMA 最佳出现在 16～19 轮，后期 Train 在线增强 Macro-AP 达 0.952～0.961，EMA Val Macro-AP 从最佳 0.536～0.553 回落到 0.512～0.538；干净 Val 分类 Loss 从最佳 0.0188～0.0205 回升至 0.0255～0.0270，训练分类 Loss 继续下降。Train 与 Val 口径不同，但后期 Val 自身退化已说明 EMA 未消除过拟合；延长训练并非当前首选。

建议：TriRelay 的两个任务均保留固定 EMA 为优先候选，不再搜索 decay 或为了 MFP 改变主评价协议。若继续验证通用性，下一批可让 GeoFactor、ProtoRoute 各在 major15/fine39 做 seed42 同次在线—EMA 对照，共四次；不直接扩展为两模型全部五 Seed。MFP 决策问题独立保留，避免把分类 AP 改善误当成已解决多功能检测。本轮不使用 Test 选择策略，所有结论限于同一个用于开发的 Validation 集；正式评价前可独立加载所选 checkpoint 复核 Val。

回传文件、全指标/全标签/分组配对统计及图位于 `runs/analysis/ema_fine39_five_seed_20261019/`：`analysis.json`、`comparison.png`、`learning_curves.png`。未修改模型、数据或配置，未启动训练。

### TriRelay 下一项候选：受控视图的 EMA 教师一致性

此项仅针对 TriRelay / major15、fine39 / seed42，正式实验使用完整原 Train/Val；不抽样、不重划分、不重提取 ESM。完整模型、既有模态 Dropout/分组 Mask、原分类及重建目标均保留。默认配置仍不启用新目标，GeoFactor、ProtoRoute 不受影响。

教师是无梯度、eval 模式的 EMA 模型，读取原本可用的完整输入；在线学生继续读取既有训练视图。教师在 optimizer step **之前**提供目标，在成功更新之后按原规则平均。已知正例不做蒸馏，继续使用原分类监督，避免低分教师反向压制已知阳性；未标注项并不被宣布为真实阴性。

新增目标为逐标签 Bernoulli `KL(teacher || student)`，不能用跨标签 softmax。未标注项按教师分数 `>=0.5` / `<0.5` 分成两组，各组先在标签内部平均，再对有效标签平均；`L_consistency = L_high + 0.1 * L_low`。0.5 只是分组规则，不是伪标签确认阈值；不给数据写回硬伪标签。该目标属于一致性正则，不是无偏 PU 风险。

固定首轮设置：`--consistency-weight 0.1 --consistency-start-epoch 7 --consistency-ramp-epochs 5`。EMA仍为0.999、第6轮开始；第6轮只初始化和积累EMA，第7～11轮的一致性系数依次为0.02、0.04、0.06、0.08、0.10，之后保持0.10。此组数值是待验证候选，不代表已经优于EMA-only。

RTX 5070 **PowerShell** 中，在该项目没有正在运行的训练、且本次代码已同步后执行（两任务串行）：

```powershell
Set-Location "D:\Data\ChatGPT\mfp\MFP-Nova"
$mfpPython = "D:\App\Code\Anaconda\envs\MFP-Nova\python.exe"
$tag = Get-Date -Format "yyyyMMdd_HHmmss"
foreach ($task in @("major15", "fine39")) {
    $run = "trirelay_${task}_esm650m_ema_consistency_seed42_$tag"
    & $mfpPython -u -m mfp_nova.cli train --model trirelay --task $task --esm 650m --seed 42 --device cuda:0 --precision fp32 --batch-size 64 --ema-decay 0.999 --ema-start-epoch 6 --consistency-weight 0.1 --consistency-start-epoch 7 --consistency-ramp-epochs 5 --name $run
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $run" }
}
```

不要在 CMD 中直接粘贴 PowerShell 变量语法；可先输入 `powershell.exe -NoProfile` 再运行上述代码。正式命令不带 `--limit-samples`、`--max-steps` 或短 `--epochs`。

每轮新增 `Consistency` 行，记录当前权重、原始KL及高/低分组KL；`history.jsonl`、metadata 与checkpoint保存相应配置。Val分类Loss仍只计算原分类目标，checkpoint与阈值规则保持原协议。`online_best.pt`和`best.pt`都来自**受新Loss影响的同一训练**，因此判断新算法收益必须与之前保存的EMA-only seed42比较，不能把本次在线分支当成旧算法对照。

候选可以从自己的 `last.pt` 恢复，省略一致性参数会自动使用已保存设置；不能向旧EMA-only checkpoint临时加入新目标。首次比较关注Macro-AP、五指标和MFP Precision/Recall/F1，并保留D构型、缺结构分组；不使用Test选择新机制。教师增加每步一次前向，原batch64与耗时需要RTX实际确认，不宣称已有GPU性能收益。

修改前的 `research.py`、`cli.py`、README 及Git提交号保存在仓库外 `D:\Data\Code\MFP-Nova-originals\20261020-trirelay-consistency`。原始三模型架构文件、YAML、数据、缓存与历史checkpoint未改动。

本机相关14项CPU检查通过：稳定Bernoulli KL、正例/监督范围mask、教师停止梯度、高低分组预算、权重日程、15/39标签训练与独立Val重评、精确断点恢复及旧EMA-only checkpoint不能临时改目标。原EMA关闭新目标时的训练轨迹检查也通过。另用真实ESM-650M缓存，对完整35,196,329/35,209,345参数的两任务模型各做2步CPU检查，覆盖长度50/5/2、D构型、有/无结构与padding；logits、Loss、梯度有限，反馈参数更新且一致性KL非零。诊断记录位于本机`runs/diagnostics/trirelay_consistency_20261020/real_cache_check.json`，它提前启用EMA只为覆盖接口，不能作为正式性能或收敛证据。尚未执行此次候选的GPU训练。

### 一致性实验后的 Val 概率配对诊断

2026-10-20 两任务 seed42 已回传。相对各自 EMA-only：major15 Macro-AP +1.178 pp、Macro-F1 −0.591 pp、MFP-F1 −1.641 pp；fine39 Macro-AP +0.308 pp、Macro-F1 −3.019 pp、MFP-F1 +0.894 pp。fine39 五指标改善，但尾组 Macro-F1 0.4713→0.4080，缺结构组 MFP-F1 两任务均下降。当前一致性权重0.1保留为可选候选，不替换默认方案，不追加多 Seed。详细记录位于本机`runs/analysis/trirelay_{major15,fine39}_consistency_seed42_20261020/`。

下一步先复用已有最佳 checkpoint 做四次 **Val** 前向并导出概率，不重新训练、不重新拟合模型、不访问 Test。RTX 5070 空闲且项目没有正在运行的训练时，在 PowerShell 执行：

```powershell
Set-Location "D:\Data\ChatGPT\mfp\MFP-Nova"
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw "Git同步失败" }
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\export_trirelay_val.ps1 -Python "D:\App\Code\Anaconda\envs\MFP-Nova\python.exe" -Device cuda:0
```

脚本固定读取以下四个运行的`best.pt`（不是`online_best.pt`或`last.pt`）：

| 任务 | EMA-only | EMA＋一致性 |
| --- | --- | --- |
| major15 | `trirelay_major15_ema_seed42_20261018_215416` | `trirelay_major15_esm650m_ema_consistency_seed42_20261020_093530` |
| fine39 | `trirelay_fine39_ema_seed42_20261019_090117` | `trirelay_fine39_esm650m_ema_consistency_seed42_20261020_125726` |

输出写入新建`runs/trirelay_val_comparison_seed42_时间/`，每组一个JSON及一个NPZ，文件前缀为`major15_ema_only`、`major15_consistency`、`fine39_ema_only`、`fine39_consistency`。最终自动打包到桌面`trirelay_val_comparison_seed42_时间.zip`，不包含权重或ESM缓存。若某次导出失败，脚本停止并保留已完成文件；原checkpoint、训练结果与阈值不改写。

回传后在本机CPU运行配对诊断，例如（按实际解压路径替换）：

```powershell
.\.venv\Scripts\python.exe -m mfp_nova.decisions --reference "解压目录/major15_ema_only.npz" --predictions "解压目录/major15_consistency.npz" --task major15 --output "runs/analysis/val_probability_major15_唯一名称"
.\.venv\Scripts\python.exe -m mfp_nova.decisions --reference "解压目录/fine39_ema_only.npz" --predictions "解压目录/fine39_consistency.npz" --task fine39 --output "runs/analysis/val_probability_fine39_唯一名称"
```

### TriRelay 当前候选：跨粒度功能辅助监督

2026-10-20 的真实 Val 概率诊断表明，一致性权重 0.1 的退步不能仅靠阈值修复：逐标签簇两折诊断中 major15/fine39 Macro-F1 分别下降约 1.566/1.415 个百分点，fine39 尾类 F1 从 0.4736 降至 0.4232。该方案保留可选，不追加其他 Seed。本轮在 **EMA-only 基准**上单独研究跨粒度辅助监督；只开发 TriRelay，完整 major15/fine39，固定 seed42，不叠加教师一致性或新的 PU 伪标签。

两任务分别训练一个完整 TriRelay，并增加一个读取最终标签证据的辅助头：

| 主任务 | 主输出 / 评价 | 辅助监督 | 完整 Train / Val | 有辅助 Train 监督 |
|---|---|---|---|---:|
| major15 | 原 15 类 | 实际 fine39 亚类标签 | 39,478 / 8,534 | 39,455 |
| fine39 | 原 39 类 | 实际 major15 功能轴标签 | 39,455 / 8,529 | 39,455 |

这不是把两个任务合并成一个训练、不是把缺行填成阴性，也不是给 major15 重复加一个同目标分类器。两任务分别使用自己的主划分；辅助标签以 `row_id` 对齐，只对两任务共同有效的 **Train** 行启用。major15 的另 23 行仍参与主任务，辅助分类与层级 Loss 均屏蔽。辅助权重、先验及长尾收缩统计只来自这些实际 Train 行。当前共同 Train 中，fine39 映射到 major15 的父功能正例无矛盾；major15 还提供 71 个不能由 fine39 正例投影得到的已标注轴单元，因此直接使用实际 major15 标签，而非仅从 fine39 自行生成目标。

新头输入为主分类所用的最终证据 `E = H_final + gate * (H_relation - H_final)`，不是读取真值或主概率。辅助 Query 由功能族内证据均值与可学习标签 Query 组成，通过带功能族偏置的交叉 Attention 读取全部主标签上下文，再经过前馈层和现有长尾分类器。主序列/几何/理化编码、三轮交互、关系推理和标签反馈全部保留。辅助梯度回到完整证据网络；主输出仍使用原分类头，**不混合辅助概率、不增加新的预测阈值**。

新增训练目标：

```text
L = L_original + lambda_aux * L_observed(aux_logits, actual_other_task_labels, aux_mask)
               + lambda_hierarchy * mean_valid(ReLU(p_subtype - p_parent)^2)
```

候选设置 `lambda_aux=0.1`、`lambda_hierarchy=0.02`，随既有辅助 warmup 在第 1～5 轮线性升至目标值。辅助分类继续使用原 ASL 风格弱未标注项、Train 正例权重和 `__unknown` 语义；0 仍非确认阴性。层级项是分数上的柔性父子存在约束，允许多亚类同时存在，`__unknown` 不与已知亚类互斥；它不保证真实概率校准。泛抗微生物与具体抗菌可同时存在，独立功能数继续在原评价函数中去重；不把“只标了一个功能”强行当作真实单功能阴性。

新功能默认关闭；`--function-aux-weight` 为正时启用，`--function-hierarchy-weight 0` 可单独消融层级项。不传新参数仍实例化旧架构，旧 checkpoint 可照常加载；候选必须从新运行开始，不能把旧 EMA-only / 一致性 `last.pt` 当作新目标的续训起点。

RTX 5070 在当前训练结束后执行以下 **PowerShell** 命令。两组串行，完整数据，没有 `--limit-samples` / `--max-steps`；若只想先跑一组，将任务列表改为单个 `'fine39'`。当前 batch64/FP32 为沿用的起点，新增候选的显存与速度仍需 RTX 实测。

```powershell
Set-Location 'D:\Data\ChatGPT\mfp\MFP-Nova'
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Git 更新失败，先处理上方错误。' }
$mfpPython = 'D:\App\Code\Anaconda\envs\MFP-Nova\python.exe'
& $mfpPython -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw '项目安装失败。' }
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'
foreach ($task in @('fine39', 'major15')) {
    $run = "trirelay_${task}_esm650m_ema_functionaux_seed42_$tag"
    & $mfpPython -u -m mfp_nova.cli train `
        --model trirelay --task $task --esm 650m --seed 42 `
        --device cuda:0 --precision fp32 --batch-size 64 `
        --ema-decay 0.999 --ema-start-epoch 6 --consistency-weight 0 `
        --function-aux-weight 0.1 --function-hierarchy-weight 0.02 `
        --name $run
    if ($LASTEXITCODE -ne 0) { throw "$task 训练失败；后续任务未启动。" }
}
```

每轮新增 `FunctionAux` 行，显示辅助任务、分类权重/Loss、层级权重/Loss；0 层级 Loss 表示当前 batch 未违反约束，并非模块未启用。两项 Loss 与权重写入 `history.jsonl`，metadata 记录辅助标签顺序与 Train 支持行数，checkpoint 保存辅助参数与统计，支持自己的 `last.pt` 精确恢复。恢复时指定原任务和自己的 `last.pt`，使用新的唯一 `--name`；省略辅助参数会沿用 checkpoint 中的原设置。

主任务的 Val 分类 Loss、Macro-AP/Micro-AP 选模、Val 全局 Jaccard 阈值、传统五指标和独立 MFP 指标继续原协议。辅助 Val/Test 标签不进入训练或阈值拟合；本轮不运行 Test。对比对象是各任务已有的 **EMA-only seed42 独立运行**，不是此次受到辅助 Loss 影响的 `online_best.pt`。这轮增加了另一标签粒度的 Train 监督，论文比较应明确说明该实验设定。

本机已通过新机制 7 项 CPU 检查和原 EMA/一致性相关 14 项检查，覆盖辅助范围/Train-only 统计、父子语义、完整结构梯度、两任务短训练、精确恢复与独立 Val 复现。真实 ESM-650M 缓存的完整模型两任务各做 2 步 CPU 检查，记录在 `runs/diagnostics/trirelay_function_aux_20261020/real_cache_check.json`；覆盖有/无结构、D 构型、padding 和 major15 无辅助标签行。参数量为 major15 **36,016,304**、fine39 **36,016,240**（不含冻结 ESM）。这些仅证明代码链路可运行，尚未证明 GPU 收敛、性能或资源收益。

### TriRelay 下一项候选：主任务优先的重建梯度平衡

跨粒度辅助监督的两个 seed42 完整实验已经返回：相对 EMA-only，major15/fine39 Macro-AP 分别 +0.660/+0.092 个百分点，但独立 MFP-F1 分别 −0.535/−1.454 个百分点；当前权重0.1/0.02不作为默认，不追加其他Seed。详细分析在本机 `runs/analysis/trirelay_functionaux_seed42_20261020/report.md`。

依据项目外改进书第6.2节，本轮回到 **原完整 TriRelay + EMA**，仅研究原重建任务的梯度作用。真实 ESM-650M Train 的完整模型各做6步CPU探查，序列投影层观察到 major15 3/6、fine39 2/6 批次主辅梯度夹角为负；反馈层分别2/6、4/6。部分层的加权重建梯度远大于分类梯度。这是**新初始化模型的局部短训练证据**，不是正式服务器checkpoint的梯度结论，也不能证明其造成之前的指标退步。记录位于 `runs/diagnostics/trirelay_reconstruction_gradient_20261020/probe.json`。

设主目标为原功能分类组（最终主头＋加权直接头＋加权分支头），辅助目标为原加权重建组。只在两个目标共同使用的参数上，计算：

```text
g_main = grad(L_classification + w_direct*L_direct + w_branch*L_branch)
g_aux  = grad(w_reconstruction*L_reconstruction)

若 dot(g_main,g_aux)<0:
    g_aux = g_aux - dot(g_main,g_aux)/||g_main||² * g_main
g_aux *= min(1, rho*||g_main||/||g_aux||)
g = g_main + g_aux                       # 共同参数
```

`rho=1` 是本轮固定候选；无共享主梯度时不让重建单独移动共享参数。主任务私有参数保留原分类梯度，重建头私有参数保留原重建梯度，不把辅助修正注入分类器私有参数。方向投影与范数平衡都在**共同参数整体**上完成，不是逐层或逐标签约束。它保留原重建任务、原0.05权重及辅助warmup，改变的是梯度组合，而非删除重建模块或再次实施重建权重衰减。

这是项目适配的非对称主辅梯度方法，不宣称复现AuxiNash或完整PCGrad。约束只在每个microbatch的原始梯度空间成立；梯度累积的交叉项、AdamW预条件及参数更新仍可能改变方向，因此不保证下一step的分类Loss、Val指标或长期收敛单调改善。辅助组范围不包含新的跨粒度监督；本轮指令将其与一致性均关闭。

新功能默认关闭；`--reconstruction-gradient sum` 为旧梯度求和，`project` 启用上述投影和范数平衡，`cap` 仅限制范数而保留原重建梯度方向。`--reconstruction-grad-ratio` 只控制共享重建梯度的上限，默认1。model参数量恢复为 major15 **35,196,329**、fine39 **35,209,345**，冻结ESM不计入；输入、模型深度、反馈、关系推理和评价协议不变。

RTX5070 的 **PowerShell** 正式实验命令如下，每组使用完整原数据、seed42、EMA、FP32/batch64；从新运行开始，不接旧目标的 `last.pt`。先fine39，再major15，串行执行：

```powershell
Set-Location 'D:\Data\ChatGPT\mfp\MFP-Nova'
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Git 更新失败。' }
$mfpPython = 'D:\App\Code\Anaconda\envs\MFP-Nova\python.exe'
& $mfpPython -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw '项目安装失败。' }
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'
foreach ($task in @('fine39', 'major15')) {
    $run = "trirelay_${task}_esm650m_ema_recongrad_seed42_$tag"
    & $mfpPython -u -m mfp_nova.cli train `
        --model trirelay --task $task --esm 650m --seed 42 `
        --device cuda:0 --precision fp32 --batch-size 64 `
        --ema-decay 0.999 --ema-start-epoch 6 `
        --consistency-weight 0 --function-aux-weight 0 `
        --reconstruction-gradient project --reconstruction-grad-ratio 1 `
        --name $run
    if ($LASTEXITCODE -ne 0) { throw "$task 训练失败；后续任务未启动。" }
}
```

每轮新增 `ReconGrad` 行与对应history字段：冲突批次数、范数受限批次数、投影前平均cosine、加权主辅范数比及施加后的范数比。分母为梯度有限的microbatch数；AMP溢出仍由已有GradScaler跳步处理，FP32非有限梯度仍明确报错，不用零梯度掩盖。无梯度范数时相关诊断为NA。总Loss仍记录原加权目标和，**投影后的梯度通常不对应这个标量的普通梯度**，metadata明确记录这一点。

自己的 `last.pt` 可恢复并自动沿用投影参数；不能通过恢复旧checkpoint临时改变优化规则。正式比较继续Val Macro-AP最大、Micro-AP并列判定、Val全局Jaccard阈值，关注五指标、尾类和独立MFP P/R/F1，不按Test结果调参。对照为原EMA-only独立运行，`online_best.pt`仅为同轨迹在线参考。

本轮8项优化相关检查与3项原EMA检查覆盖投影/范数上限、私有梯度、损失缩放与累积、两任务短训练、精确恢复、独立Val重评与实验汇总区分。真实缓存的完整模型另作两任务各2步CPU检查，结果在 `runs/diagnostics/trirelay_reconstruction_gradient_20261020/real_cache_check.json`。每step对同一前向图做两组梯度计算，GPU耗时及额外梯度内存需实测，不能沿用原训练速度或宣称已提高指标。本轮不占用远端GPU、不执行正式Test。

诊断按`row_id`和标签名称对齐两份完整Val概率，核对当前观测标签；输出原保存阈值、共享0.50/参考阈值/候选阈值、逐标签Val拟合与两折簇交叉拟合结果。共享阈值用于研究分数尺度差异；逐标签全Val拟合有选择乐观性，交叉拟合仅隔离额外阈值拟合，checkpoint仍用完整Val选择，均不能当作独立Test结论。所有阈值实验都不改写原checkpoint、不自动采用。AP保持原始概率口径，未标注0不解释为实验确认阴性，分数分位数不解释为真实概率校准。

本地输出仅`results.json`、`label_diagnostics.csv`和`report.md`，包含逐标签AP/F1、TP/观测FP/FN、正例/未标注分数分位数、近阈值数量及缺结构/D构型分组。相关15项CPU测试通过，并确认旧的真实8534×15 Val NPZ仍可读取；当前四份正式概率尚在服务器待导出，本机未执行四checkpoint重评。此次只增加分析入口和导出脚本，没有修改训练算法或配置；修改前的`decisions.py`和README保存在仓库外`D:\Data\Code\MFP-Nova-originals\20261020-val-probability-comparison`。

### 重建梯度单项消融：仅限制范数

完整 fine39 / seed42 的 `cap, rho=1` 已完成RTX5070训练及Val概率配对：相对EMA-only，保存阈值下Jaccard由0.5456升至0.5574、多功能组功能集合F1由0.6134升至0.6319，但MFP检测F1由0.5117降至0.5060。相对`project`，cap的Macro/Micro-F1及MFP检测F1改善，标签收益仍不均匀。三方案均第16轮最佳、第31轮早停，不能把这次单seed结果推广为普适收益。下一步只补 **TriRelay / 完整major15 / seed42 / cap / rho=1**，检查另一任务的适用性；不追加新Loss、其他seed或Test，原默认模式仍为`sum`。

`cap`在共同参数上使用原始加权重建梯度：

```text
g_aux *= min(1, rho * ||g_main|| / ||g_aux||)
g = g_main + g_aux
```

与`project`相比仅取消负内积时的方向投影，其余共享参数定义、零主梯度处理、私有梯度、EMA、数据、损失权重及训练协议相同。`cap`保留负向分量，仍可能抵消部分主梯度；不保证下一步主Loss或Val改善。它仍对同一前向图计算两组梯度，不能假设恢复到原EMA-only速度。

以下RTX5070与V100命令二选一，在所选项目空闲时同步并运行；与既有RTX对照做单项比较时优先使用RTX。V100使用相同实验协议，但不同硬件及PyTorch构建可能产生不同轨迹，不能作为额外独立seed合并。均从新运行开始，不恢复`project`或EMA-only的`last.pt`。

RTX5070 PowerShell：

```powershell
Set-Location 'D:\Data\ChatGPT\mfp\MFP-Nova'
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Git 更新失败。' }
$mfpPython = 'D:\App\Code\Anaconda\envs\MFP-Nova\python.exe'
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'
$run = "trirelay_major15_esm650m_ema_reconcap_seed42_rtx5070_$tag"
& $mfpPython -u -m mfp_nova.cli train `
    --model trirelay --task major15 --esm 650m --seed 42 `
    --device cuda:0 --precision fp32 --batch-size 64 `
    --ema-decay 0.999 --ema-start-epoch 6 `
    --consistency-weight 0 --function-aux-weight 0 `
    --reconstruction-gradient cap --reconstruction-grad-ratio 1 `
    --name $run
if ($LASTEXITCODE -ne 0) { throw 'major15 cap 训练失败。' }
```

V100 Bash（确认空闲GPU后修改GPU_ID，后台运行支持断开SSH；默认CPU线程4）：

```bash
(
set -e
cd ~/wuzhencheng/projects/MFP-Nova
git pull --ff-only
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MFP-Nova
GPU_ID=1
CPU_THREADS=4
tag="$(date +%Y%m%d_%H%M%S)"
run="trirelay_major15_esm650m_ema_reconcap_seed42_v100_${tag}"
mkdir -p runs
nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS="$CPU_THREADS" \
    MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" \
    "$(command -v python)" -u -m mfp_nova.cli train \
    --model trirelay --task major15 --esm 650m --seed 42 \
    --device cuda:0 --precision fp32 --batch-size 64 \
    --ema-decay 0.999 --ema-start-epoch 6 \
    --consistency-weight 0 --function-aux-weight 0 \
    --reconstruction-gradient cap --reconstruction-grad-ratio 1 \
    --name "$run" > "runs/${run}.log" 2>&1 < /dev/null &
printf 'PID=%s\n日志=runs/%s.log\n' "$!" "$run"
)
```

`ReconGrad`显示`mode=cap`、观测冲突`conflict`、实际投影`projected=0`及范数限制`capped`。history保存真实mode与`projected_batches`，避免把观测到冲突误当作进行了投影。旧`project`行为保留；恢复自己的`last.pt`会沿用模式和rho，跨模式恢复仍会报错。

继续以Val Macro-AP选checkpoint、Micro-AP并列判定、Val全局Jaccard选阈值。完成后独立导出最佳EMA的Val JSON与NPZ，与既有EMA-only及`project`进行相同口径比较，同时看传统五指标、MFP detection和`mfp_function_prediction`。不提前运行Test或改选checkpoint。本轮相关梯度及恢复测试和真实缓存CPU检查只验证实现，不能当作GPU性能或模型收益证据。

### 当前实验：TriRelay 共享交互＋私有证据

major15 的 cap 实验已完成，AP 的微小提升没有带来 F1/Jaccard/MFP 的一致改善，因此不继续扩展重建梯度网格。改进书 7.1 的共享—私有机制也已完成两任务seed42训练：保留原完整深度和交互，增加绕过 relay/反馈回写的私有证据流；两轮标签条件读取及质量感知门控将其融入决策。完整数据仍为 major15 Train39478/Val8534、fine39 Train39455/Val8529；没有重新划分或缩小正式训练集。

开关 `--shared-private`，默认关闭，旧模型及旧 checkpoint 保持可用。实际参数 major15 **39,147,692**、fine39 **39,160,708**，不含冻结 ESM。总 Loss 和原分支预算不增加；private 分支共享原 0.1 分支预算，重建只读取共享流。详细数据流和限制见[模型技术说明](../docs/MFP-Next-Generation-Architecture.md)。本轮关闭此前一致性、功能辅助和重建梯度控制，只与已有 EMA-only 对照；新架构从新运行训练，不能接旧架构的 last.pt。

以下保留已完成实验的训练命令供复现；当前下一步执行文末的Val导出及私有路径推理消融，无需重训。两平台命令二选一，在所选项目无训练进程时更新源码。训练协议为固定seed42、完整数据、FP32/batch64、EMA0.999/start6；不跑新seed或Test。RTX5070实际峰值major15为7.20GiB、fine39为7.61GiB，EMA阶段每轮约169/177秒，分别第13/15轮最佳、第28/30轮早停，两组总计约165分钟。V100尚未实测此候选。

RTX5070，**PowerShell**：

```powershell
Set-Location 'D:\Data\ChatGPT\mfp\MFP-Nova'
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Git 更新失败。' }
$mfpPython = 'D:\App\Code\Anaconda\envs\MFP-Nova\python.exe'
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'
$artifacts = @()
foreach ($task in @('fine39', 'major15')) {
    $run = "trirelay_${task}_esm650m_ema_sharedprivate_seed42_rtx5070_$tag"
    & $mfpPython -u -m mfp_nova.cli train `
        --model trirelay --task $task --esm 650m --seed 42 `
        --device cuda:0 --precision fp32 --batch-size 64 `
        --ema-decay 0.999 --ema-start-epoch 6 --shared-private `
        --consistency-weight 0 --function-aux-weight 0 `
        --reconstruction-gradient sum --name $run
    if ($LASTEXITCODE -ne 0) { throw "$task 训练失败；后续任务未启动。" }
    & $mfpPython -u -m mfp_nova.cli evaluate `
        --checkpoint "runs/$run/best.pt" --split val --device cuda:0 --batch-size 64 `
        --output "runs/$run/val.json" --predictions "runs/$run/val_predictions.npz"
    if ($LASTEXITCODE -ne 0) { throw "$task Val 导出失败。" }
    foreach ($file in @('metadata.json','results.json','history.jsonl','val.json','val_predictions.npz')) {
        $artifacts += "runs/$run/$file"
    }
}
$zip = Join-Path ([Environment]::GetFolderPath('Desktop')) "trirelay_sharedprivate_seed42_$tag.zip"
# 直接压缩并保留运行目录；不复制输出、不压缩 pt 权重。
& $mfpPython -c "import sys,zipfile; z=zipfile.ZipFile(sys.argv[1],'x',zipfile.ZIP_DEFLATED); [z.write(p,p) for p in sys.argv[2:]]; z.close()" $zip $artifacts
if ($LASTEXITCODE -ne 0) { throw '结果压缩失败。' }
Write-Host "回传文件：$zip"
```

V100，**Ubuntu Bash**。`GPU_ID`选空闲物理GPU，`CPU_THREADS`可选2/4/8等；后台整个串行队列，关闭SSH后继续。`CUDA_VISIBLE_DEVICES`映射后，进程内始终使用`cuda:0`。

```bash
(
set -e
cd ~/wuzhencheng/projects/MFP-Nova
git pull --ff-only
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MFP-Nova
GPU_ID=1
CPU_THREADS=4
tag="$(date +%Y%m%d_%H%M%S)"
mkdir -p runs
nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS="$CPU_THREADS" \
    MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" \
    bash -c '
set -e
for task in fine39 major15; do
    run="trirelay_${task}_esm650m_ema_sharedprivate_seed42_v100_${2}"
    "$1" -u -m mfp_nova.cli train \
        --model trirelay --task "$task" --esm 650m --seed 42 \
        --device cuda:0 --precision fp32 --batch-size 64 \
        --ema-decay 0.999 --ema-start-epoch 6 --shared-private \
        --consistency-weight 0 --function-aux-weight 0 \
        --reconstruction-gradient sum --name "$run"
    "$1" -u -m mfp_nova.cli evaluate \
        --checkpoint "runs/$run/best.pt" --split val --device cuda:0 --batch-size 64 \
        --output "runs/$run/val.json" --predictions "runs/$run/val_predictions.npz"
done
tar -czf "runs/trirelay_sharedprivate_seed42_${2}.tar.gz" \
    runs/trirelay_*_esm650m_ema_sharedprivate_seed42_v100_${2}/{metadata.json,results.json,history.jsonl,val.json,val_predictions.npz}
' _ "$(command -v python)" "$tag" > "runs/sharedprivate_${tag}.log" 2>&1 < /dev/null &
printf 'PID=%s\n日志=runs/sharedprivate_%s.log\n' "$!" "$tag"
)
```

完成后回传压缩文件，比较 Macro/Micro-AP、F1、传统五指标、MFP检测及功能集合F1、逐标签和缺结构/D型分组；独立 Val 重评必须复现保存阈值下的结果。共享/私有路径的收益仍待实验验证，不把门控权重直接解释成生物因果作用。

### 下一步：同 checkpoint 的完整 Val 导出与私有证据推理消融

已完成的共享—私有训练相对EMA-only：major15 MacroAP 0.5614→0.5636、MacroF1 0.5510→0.5536，但MFP检测F1 0.5313→0.5056；fine39 MacroAP 0.5359→0.5374、MicroF1 0.5947→0.6004，但MFP检测F1 0.5117→0.4788、功能集合F1 0.6134→0.6039。当前不替换默认算法、不扩展五seed，先利用已有权重定位私有路径与决策差异。

`python -m mfp_nova.private_evidence`对每个run的`best.pt`做两次完整Val推理：`full`保留原计算；`without_private`仅将**两轮读取中的私有证据残差**置零，保留全部权重、共享投影、交互、反馈、高阶关系及原保存阈值，重新计算反馈上下文和融合权重。没有重训或重新拟合阈值，不能将其视作原EMA-only架构或重训的无私有流模型。推理仍计算私有表示，不用于测量移除模块后的加速。

两任务共四次Val推理，输出各自`full.json/.npz`、`without_private.json/.npz`和一份`summary.json`，自动生成同名ZIP，不含checkpoint/ESM。ZIP现同时直接收录原run的`metadata.json/results.json/history.jsonl`，不复制权重或修改源记录，便于后续检查训练Loss。summary记录完整推理相对原results的指标复现差值、阈值是否一致、消融差值、逐标签AP/F1变化及预测改变数量。NPZ保留row_id、ID、标签、目标、原阈值及消融说明；不覆盖原run。没有多功能标注或无正例时，无法定义的指标保持null。

RTX5070 PowerShell，使用本次回传包对应的两份实际运行目录。在项目空闲时更新后执行，当前无需重新安装依赖：

```powershell
Set-Location 'D:\Data\ChatGPT\mfp\MFP-Nova'
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Git 更新失败。' }
$mfpPython = 'D:\App\Code\Anaconda\envs\MFP-Nova\python.exe'
$out = 'runs/sharedprivate_val_seed42_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $mfpPython -u -m mfp_nova.private_evidence `
    --runs `
    'runs/trirelay_major15_esm650m_ema_sharedprivate_seed42_rtx5070_20261021_155324' `
    'runs/trirelay_fine39_esm650m_ema_sharedprivate_seed42_rtx5070_20261021_171644' `
    --device cuda:0 --batch-size 64 --output $out
if ($LASTEXITCODE -ne 0) { throw 'Val导出或消融失败，请查看错误。' }
Write-Host "回传：$out.zip"
```

V100 Bash：仅在**这两份相同的checkpoint和对应metadata/results已经传到下面目录**时运行；本机当前没有这两份正式权重，不替用户假定V100已完成传输。优先直接在RTX5070执行，不为V100补训。若已有V100本机独立运行，应替换为对应run目录，结果单独标记，不能冒充RTX同权重复现。

```bash
(
set -e
cd ~/wuzhencheng/projects/MFP-Nova
git pull --ff-only
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MFP-Nova
GPU_ID=1
CPU_THREADS=4
out="runs/sharedprivate_val_seed42_v100_$(date +%Y%m%d_%H%M%S)"
nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS="$CPU_THREADS" \
    MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" \
    "$(command -v python)" -u -m mfp_nova.private_evidence \
    --runs \
    runs/trirelay_major15_esm650m_ema_sharedprivate_seed42_rtx5070_20261021_155324 \
    runs/trirelay_fine39_esm650m_ema_sharedprivate_seed42_rtx5070_20261021_171644 \
    --device cuda:0 --batch-size 64 --output "$out" \
    > "${out}.log" 2>&1 < /dev/null &
printf 'PID=%s\n日志=%s.log\n回传=%s.zip\n' "$!" "$out" "$out"
)
```

需要单独操作时，统一评价入口也支持`evaluate --without-private-evidence`，必须指向共享—私有TriRelay checkpoint。正常`evaluate`与所有训练默认计算保持不变；旧checkpoint不增加参数或state_dict键。当前本地4项相关检查覆盖两任务、两轮残差移除、mask、严格权重不变、阈值不变、独立导出及压缩；小型fixture结果仅验证实现，不报告成真实模型收益。正式四次GPU推理仍需在服务器执行。

### 当前下一项：反馈前共享／私有证据对等监督

四次正式Val推理已完成：full复现原结果；移除私有残差使major15/fine39 MacroAP分别由0.5636/0.5374降至0.4326/0.4370，当前模型已依赖私有路径。full在全局阈值扫描中的最高MFP检测F1仍低于EMA-only原结果。下一轮只检验共享／私有分工的监督调整，不扩展seed、不用Test。

开关为`--branch-supervision shared_private`，同时需要`--shared-private`；原默认`mixed_private`保留。新分支监督来自**第一次读取、标签反馈之前**的共享证据和未混合私有证据，二者均使用无关系上下文的原label query，不再以已混合的分支作为共享监督的替身。第二轮读取、动态融合、直接分类、反馈和高阶关系均照常工作。复用既有注意力输出和长尾分类器，不新增参数或注意力层；major15/fine39仍为39,147,692/39,160,708参数。

```text
L_branch = 0.5 * mean_available_modalities(L_shared_first_read)
         + 0.5 * mean_available_modalities(L_private_first_read)
L_total  = L_classification + 0.2 L_direct + 0.1 L_branch + 0.05 L_reconstruction
```

aux warmup仍沿用原规则，完整权重后两路径各占0.05；共享目标不向私有投影、私有读取器或门控反传，但两者仍共同训练编码器和原分类器，不宣称统计独立。只改分支监督内容，不增加总辅助预算；一致性/功能辅助关闭，重建梯度用sum。history新增`branch_shared`、`branch_private`，保留`branch`聚合；每epoch新增`Branches`行，显示两项原始Loss及各自有效权重。这是待实验检验的候选，不预先保证指标提高。

新Loss从头创建新运行，不恢复上轮`last.pt`；恢复自己的新实验则可省略模式参数，Trainer会从checkpoint沿用，显式跨模式恢复会报错。推理无需指定该训练Loss开关，旧/新checkpoint的state_dict键和形状不变。Val MacroAP选模、MicroAP并列判定和原Val阈值策略保持不变。

以下每段只运行一个任务。先设`task=major15`完成后，另行设为`fine39`再执行；不会自动启动第二组。两平台选一个运行，在项目空闲时先`git pull --ff-only`。没有依赖或环境升级。

RTX5070 PowerShell：

```powershell
Set-Location 'D:\Data\ChatGPT\mfp\MFP-Nova'
$mfpPython = 'D:\App\Code\Anaconda\envs\MFP-Nova\python.exe'
$task = 'major15'   # 第二组单独改成 'fine39'
$run = "trirelay_${task}_esm650m_ema_sharedsupervision_seed42_rtx5070_" + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $mfpPython -u -m mfp_nova.cli train `
    --model trirelay --task $task --esm 650m --seed 42 `
    --device cuda:0 --precision fp32 --batch-size 64 `
    --ema-decay 0.999 --ema-start-epoch 6 --shared-private `
    --branch-supervision shared_private `
    --consistency-weight 0 --function-aux-weight 0 `
    --reconstruction-gradient sum --name $run
if ($LASTEXITCODE -ne 0) { throw "$task 训练失败。" }
& $mfpPython -u -m mfp_nova.private_evidence `
    --runs "runs/$run" --device cuda:0 --batch-size 64 --output "runs/${run}_val"
if ($LASTEXITCODE -ne 0) { throw "$task Val导出失败。" }
Write-Host "回传：runs/${run}_val.zip"
```

V100 Bash，选择空闲GPU和CPU线程；整段训练及后续导出在nohup后台执行，断开SSH后继续：

```bash
(
set -e
cd ~/wuzhencheng/projects/MFP-Nova
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MFP-Nova
GPU_ID=1
CPU_THREADS=4
task=major15   # 第二组单独改成 fine39
run="trirelay_${task}_esm650m_ema_sharedsupervision_seed42_v100_$(date +%Y%m%d_%H%M%S)"
mkdir -p runs
nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS="$CPU_THREADS" \
    MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" \
    bash -c '
set -e
"$1" -u -m mfp_nova.cli train \
    --model trirelay --task "$2" --esm 650m --seed 42 \
    --device cuda:0 --precision fp32 --batch-size 64 \
    --ema-decay 0.999 --ema-start-epoch 6 --shared-private \
    --branch-supervision shared_private \
    --consistency-weight 0 --function-aux-weight 0 \
    --reconstruction-gradient sum --name "$3"
"$1" -u -m mfp_nova.private_evidence \
    --runs "runs/$3" --device cuda:0 --batch-size 64 --output "runs/${3}_val"
' _ "$(command -v python)" "$task" "$run" > "runs/${run}.log" 2>&1 < /dev/null &
printf 'PID=%s\n日志=runs/%s.log\n回传=runs/%s_val.zip\n' "$!" "$run" "$run"
)
```

导出后回传每任务的`*_val.zip`：同时含训练三份记录、完整Val与同权重私有残差消融的JSON/NPZ和summary，不需再单独压缩。对照为前一轮shared-private候选；原EMA-only及fine39 cap继续保留作参照。

本地6项相关检查覆盖Loss预算和mask、共享目标的私有梯度隔离、参数/前向输出不变、两任务短训练、EMA/精确恢复/独立Val及导出打包；真实650M缓存另作两任务各8样本2步的完整规模CPU检查，关键路径均有有限梯度和参数更新。记录在`runs/diagnostics/trirelay_branch_supervision_20261021/real_cache_check.json`。未启动远端GPU或正式Test，模型收益仍待本轮完整训练。

### 当前诊断：首次读取的共享／私有／混合证据

对等监督完整Val结果已回传：fine39 MacroAP=0.540201、Jaccard=0.567590、MFP-F1=0.515029，优于前轮共享私有；major15 MacroAP=0.561389、Jaccard=0.622325，尚未优于原EMA-only。private-off的最终输出会改变反馈分布，不能当成首次读取共享头的性能。

`python -m mfp_nova.branch_evidence`直接读取现有`best.pt`，一次完整Val前向同时观察11个真实读出：三模态各自的`shared_*`、未乘门控的`private_*`、`mixed_*`（共享＋门控私有），以及原`direct`融合分类和`final`最终分类。前9个均发生在标签反馈之前；共享流此时已完成跨模态relay，不能解释成原始单模态编码器。复用既有分类器，不构造未经训练的“纯共享融合分类器”，不删除私有路径或关闭完整模型机制。

输出`branches.json`、`readouts.npz`、总`summary.json`，自动打包并附原metadata/results/history，不含权重。NPZ概率轴为`[Val行, 11个head_names, 标签]`，附`head_available`、首次门控和融合权重；无效模态行保留对齐但不计入该head指标，不能忽略mask后直接统计。每头同时报告自身可用行和三模态共同可用行指标，后者便于比较同一批样本。没有共同可用行时指标明确为null。结构可用不等于可靠；门控幅值不等于因果贡献。

所有头沿用checkpoint原保存阈值，**不拟合分支阈值、不更新权重、不使用Test**。分支分数未单独校准，优先比较Macro/Micro-AP及逐标签AP；F1、MFP和五指标只作统一阈值下诊断。`final_reproduction_deltas`显示最终输出相对原results的差异，不自动改写原成绩。需同时诊断前轮与本轮四个checkpoint才能判断共享头是否改善；不需要重训。

RTX5070 PowerShell，在项目无运行中训练时更新代码后执行：

```powershell
Set-Location 'D:\Data\ChatGPT\mfp\MFP-Nova'
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Git更新失败。' }
$mfpPython = 'D:\App\Code\Anaconda\envs\MFP-Nova\python.exe'
$runs = @(
    'runs/trirelay_major15_esm650m_ema_sharedprivate_seed42_rtx5070_20261021_155324'
    'runs/trirelay_fine39_esm650m_ema_sharedprivate_seed42_rtx5070_20261021_171644'
    'runs/trirelay_major15_esm650m_ema_sharedsupervision_seed42_rtx5070_20261022_002000'
    'runs/trirelay_fine39_esm650m_ema_sharedsupervision_seed42_rtx5070_20261022_002000'
)
$out = 'runs/trirelay_firstread_seed42_rtx5070_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $mfpPython -u -m mfp_nova.branch_evidence --runs $runs `
    --device cuda:0 --batch-size 64 --cpu-threads 4 --output $out
if ($LASTEXITCODE -ne 0) { throw '首次读取Val诊断失败。' }
Write-Host "回传压缩包：${out}.zip"
```

V100 Bash：仅当**上述四组相同RTX5070权重和原metadata/results/history已通过现有大文件中转放在同名目录**时执行；没有则优先在RTX5070运行，不为此重训。V100独立训练的权重不能冒充RTX同权重复现。选择空闲GPU和CPU线程；断开SSH后继续：

```bash
(
set -e
cd ~/wuzhencheng/projects/MFP-Nova
git pull --ff-only
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MFP-Nova
GPU_ID=1
CPU_THREADS=4
out="runs/trirelay_firstread_seed42_v100_$(date +%Y%m%d_%H%M%S)"
nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS="$CPU_THREADS" \
    MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" \
    "$(command -v python)" -u -m mfp_nova.branch_evidence --runs \
    runs/trirelay_major15_esm650m_ema_sharedprivate_seed42_rtx5070_20261021_155324 \
    runs/trirelay_fine39_esm650m_ema_sharedprivate_seed42_rtx5070_20261021_171644 \
    runs/trirelay_major15_esm650m_ema_sharedsupervision_seed42_rtx5070_20261022_002000 \
    runs/trirelay_fine39_esm650m_ema_sharedsupervision_seed42_rtx5070_20261022_002000 \
    --device cuda:0 --batch-size 64 --cpu-threads "$CPU_THREADS" --output "$out" \
    > "${out}.log" 2>&1 < /dev/null &
printf 'PID=%s\n日志=%s.log\n回传=%s.zip\n' "$!" "$out" "$out"
)
```

本地相关检查覆盖15/39标签、默认预测/参数不变、前反馈张量取值、D/L与缺结构、阈值不变、无可用结构、完整Val导出及ZIP。另对真实650M缓存各6条Val输入进行完整容量CPU前向，记录在`runs/diagnostics/trirelay_first_read_20261022/real_cache_check.json`；使用未训练模型验证输入链路，不报告性能。正式checkpoint的完整GPU诊断仍需在服务器执行。

### 当前融合对照：RMS竞争式与RMS加法

竞争式融合已完成完整seed42训练：major15 MacroAP=0.571396、Jaccard=0.632288，多数指标优于前轮；fine39 MacroAP=0.545289，但Jaccard=0.550318、MFP-F1=0.499779，低于归一化前对等监督的0.567590/0.515029。门控饱和和分支过量预测缓解，首次融合仍高度集中于序列。不能把机制改善等同于最终性能全面提升。

本轮新增`--evidence-fusion normalized_residual`，需要`--shared-private`。与`normalized_competitive`保留相同RMS归一化、门控输入、纯分支监督，只把共享系数从`1-g`改为`1`。旧默认`residual`和已完成的竞争式均保留；state_dict键及参数量不变：major15=39,147,692、fine39=39,160,708。两种归一化模式在**两次读取**中均执行：

```text
S_hat = S / sqrt(mean(S^2, feature_dim) + 1e-5)
P_hat = P / sqrt(mean(P^2, feature_dim) + 1e-5)
g     = sigmoid(existing_gate(query, S_hat, P_hat, quality))
E_competitive = (1-g) * S_hat + g * P_hat
E_rmsadd      = S_hat + g * P_hat
```

RMS统计使用FP32，无额外可学习参数；纯共享／私有分类监督也读取S_hat/P_hat，避免训练和混合使用不同尺度。门控仍是样本—标签—模态条件，缺模态仍排除；完整编码深度、三轮交互、第二轮反馈、标签关系、高阶机制、重建及原优化策略均保留。

竞争式的混合logit位于两条纯分支之间；本轮归一化加法则保留叠加表达，混合特征RMS上界约为2，但不再保证logit位于纯分支之间。新模式可能恢复有用叠加，也可能重新引入评分偏移，需实验验证。它与竞争式的差别只有组合公式；与原始加法对照可研究加法路径下的RMS作用。这三项并非完整2×2因子设计，不预先声称所有交互效应已被分离。

继续固定TriRelay、完整major15/fine39、seed42，直接对照为已完成竞争式，归一化前对等监督作为共同参考；不叠加PU、consistency、functionaux或cap。分支预算仍0.1=0.05+0.05，direct=0.2、重建=0.05、原warmup不变。Val MacroAP选模、MicroAP并列判定及Val全局阈值策略不变，不使用Test。

从头创建新运行，**不恢复其他融合模式的last.pt**。恢复自身训练时可省略融合选项，会从checkpoint沿用；显式跨模式恢复报错。console的`Branches`行应包含`fusion=normalized_residual`；metadata/model_args保存该设置。新模式`--without-private-evidence`只移除归一化私有残差，共享系数始终为1；竞争式的g=0干预还会将共享系数恢复为1。导出分别标明，不混淆两种干预。

以下命令依次训练major15、fine39，最后一次性导出两份最佳权重的11头完整Val结果并自动压缩。无需再重复private-off或重训对照。运行前在项目无训练时同步Git；不升级环境，不覆盖旧运行。RTX与V100选择一个平台执行，不能把同seed不同硬件运行冒充独立seed。

RTX5070 PowerShell：

```powershell
Set-Location 'D:\Data\ChatGPT\mfp\MFP-Nova'
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Git更新失败。' }
$mfpPython = 'D:\App\Code\Anaconda\envs\MFP-Nova\python.exe'
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'
$runs = @()
foreach ($task in @('major15','fine39')) {
    $run = "trirelay_${task}_esm650m_ema_rmsadd_seed42_rtx5070_$tag"
    & $mfpPython -u -m mfp_nova.cli train `
        --model trirelay --task $task --esm 650m --seed 42 `
        --device cuda:0 --precision fp32 --batch-size 64 `
        --ema-decay 0.999 --ema-start-epoch 6 --shared-private `
        --branch-supervision shared_private --evidence-fusion normalized_residual `
        --consistency-weight 0 --function-aux-weight 0 `
        --reconstruction-gradient sum --name $run
    if ($LASTEXITCODE -ne 0) { throw "$task 训练失败；后续任务未启动。" }
    $runs += "runs/$run"
}
$out = "runs/trirelay_rmsadd_seed42_rtx5070_${tag}_val"
& $mfpPython -u -m mfp_nova.branch_evidence --runs $runs `
    --device cuda:0 --batch-size 64 --cpu-threads 4 --output $out
if ($LASTEXITCODE -ne 0) { throw '完整Val导出失败。' }
Write-Host "回传：${out}.zip"
```

V100 Bash：自有空闲GPU运行，修改GPU_ID与CPU_THREADS；nohup同时覆盖两组训练和最终导出，关闭SSH后继续。CUDA_VISIBLE_DEVICES选择物理卡后，进程内仍使用cuda:0。

```bash
(
set -e
cd ~/wuzhencheng/projects/MFP-Nova
git pull --ff-only
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MFP-Nova
GPU_ID=1
CPU_THREADS=4
tag=$(date +%Y%m%d_%H%M%S)
mkdir -p runs
nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS="$CPU_THREADS" \
    MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" \
    bash -c '
set -e
python_bin="$1"
tag="$2"
cpu_threads="$3"
run_paths=()
for task in major15 fine39; do
    run="trirelay_${task}_esm650m_ema_rmsadd_seed42_v100_${tag}"
    "$python_bin" -u -m mfp_nova.cli train \
        --model trirelay --task "$task" --esm 650m --seed 42 \
        --device cuda:0 --precision fp32 --batch-size 64 \
        --ema-decay 0.999 --ema-start-epoch 6 --shared-private \
        --branch-supervision shared_private --evidence-fusion normalized_residual \
        --consistency-weight 0 --function-aux-weight 0 \
        --reconstruction-gradient sum --name "$run"
    run_paths+=("runs/$run")
done
"$python_bin" -u -m mfp_nova.branch_evidence --runs "${run_paths[@]}" \
    --device cuda:0 --batch-size 64 --cpu-threads "$cpu_threads" \
    --output "runs/trirelay_rmsadd_seed42_v100_${tag}_val"
' _ "$(command -v python)" "$tag" "$CPU_THREADS" \
    > "runs/trirelay_rmsadd_seed42_v100_${tag}.log" 2>&1 < /dev/null &
printf 'PID=%s\n日志=runs/trirelay_rmsadd_seed42_v100_%s.log\n回传=runs/trirelay_rmsadd_seed42_v100_%s_val.zip\n' "$!" "$tag" "$tag"
)
```

本轮7项相关检查覆盖两任务、两轮归一化加法/范围、数值与梯度、D/L及缺失输入、默认residual兼容、EMA/精确恢复、独立评价和11头打包。同权重首次读取还确认归一化分支与门控与竞争式完全一致，只在组合处产生差异。另对真实650M Train输入各8样本执行完整容量CPU两步更新，关键路径有有限非零梯度、门控参数更新；记录在`runs/diagnostics/trirelay_normalized_residual_20261022/real_cache_check.json`，未覆盖前次记录。这些检查不报告性能；本轮正式GPU收敛、最终指标和资源成本仍待服务器运行，未自动启动远端任务。

## 第四模型 HelixSet：源码候选与指标模式

新增[模型源码](../src/mfp_nova/models/helixset.py)、[配置](../configs/helixset.yaml)和[评价协议](../src/mfp_nova/metric_protocol.py)。HelixSet 使用残基对三角交互、有序多尺度片段、分源潜变量融合和证据优先的标签解码；没有直接 ESM 分类旁路。`direct` 损失字段在此模型表示潜变量辅助监督，`branch` 表示中间解码层监督，不是三模态分支分类。

标签解码现在直接读取标签相关的残基对及两个端点，将显式 pair state 接到最终标签证据中；结构读出在归一化后保留绝对可靠性尺度。`pair_readout` 和 `fine_evidence` 默认开启，可通过现有 `--ablate` 分别关闭标签对读取和细粒度记忆读取；同时关闭两者得到标签只读取潜变量的对照。`model(batch, return_evidence=True)` 可返回最后一层标签对注意力，用于后续分析；其权重不等同于生物学接触或因果解释。

**本次没有运行任何模型、测试、训练或独立评价。以下命令仅供之后手动使用，不表示已验证可运行或达到 70%。** 模型实际参数量、显存及性能仍待实测。现有数据、划分、前三模型和历史实验不变；新架构及完整项目现状见[技术说明第 9 节](../docs/MFP-Next-Generation-Architecture.md#9-当前现状与第四模型-helixset2026-10-22)。

| 训练选项 | 主指标 | checkpoint / 早停目标 |
|---|---|---|
| `--metric-suite traditional5`（HelixSet 默认） | Acc、Prec、Cov、AT、AF | 每轮 Val 拟合阈值后的 Acc；同分 AT，再同分更早 epoch |
| `--metric-suite task_specific` | Macro/Micro-AP、Macro/Micro-F1、Jaccard、分层 AP、MFP 指标 | Val Macro-AP；同分 Micro-AP，再同分更早 epoch |

AF 是错误率，越低越好。两种模式不改变训练损失。主日志按模式展示，完整诊断指标继续保存；固定0.5监控指标与传统模式用于选模的 `selection_metrics` 分开记录。最终 `primary_metrics` 给出所选指标组。独立评价自动读取 checkpoint 中的模式与 Val 阈值，不能在 Test 重拟合。恢复必须沿用保存的模式；前三研究模型不指定新选项时仍按原 AP 协议运行。

下面两端均使用 seed42、ESM650M、FP32、microbatch16×累积4（名义有效batch64）、同一 YAML、EMA0.999。模型较大，当前 batch 是未经显存测量的起点；调整 microbatch 时同步调整累积步数并记录。将任务变量改为 `fine39` 可运行另一任务，将指标变量改为 `task_specific` 可切换模式。

RTX 5070 / Windows PowerShell：

```powershell
conda activate MFP-Nova
Set-Location D:\Data\ChatGPT\mfp\MFP-Nova
$task = 'major15'
$suite = 'traditional5'
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
python -u -m mfp_nova.cli train --model helixset --task $task --seed 42 `
    --metric-suite $suite --device cuda:0 --precision fp32 `
    --batch-size 16 --accumulation-steps 4 `
    --name "helixset_${task}_${suite}_seed42_rtx5070_${tag}"
```

V100 / Ubuntu Bash：使用已克隆的 `MFP-Nova` 环境；`GPU` 选择获准使用的物理 GPU，进程内为 `cuda:0`，`CPU_THREADS` 控制线程；`nohup` 支持关闭 SSH 后继续运行。这里不连接服务器或修改系统配置。

```bash
conda activate MFP-Nova
cd ~/wuzhencheng/projects/MFP-Nova
task=major15
suite=traditional5
GPU=0
CPU_THREADS=8
tag=$(date +%Y%m%d_%H%M%S)
run="helixset_${task}_${suite}_seed42_v100_${tag}"
mkdir -p runs
nohup env CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS="$CPU_THREADS" \
    MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" \
    python -u -m mfp_nova.cli train --model helixset --task "$task" --seed 42 \
    --metric-suite "$suite" --device cuda:0 --precision fp32 \
    --batch-size 16 --accumulation-steps 4 --name "$run" \
    > "runs/${run}.log" 2>&1 < /dev/null &
printf 'PID=%s  Log=runs/%s.log\n' "$!" "$run"
```

现有 `evaluate`、`predict` 和 `summarize` 通过统一研究模型加载入口支持该模型的 checkpoint；这些扩展本次未执行验证。`gpu-check`、`mfp_nova.experiments` 和 `run_multiseed.ps1` 均已纳入四模型。`all` 为 40 组计划；原三模型的历史设置保留，HelixSet 使用其 YAML 中的 batch、累积、EMA 与指标模式，不套用旧三模型的 batch64 单步设置。此次没有执行这些入口。


## TriRelay 关系视图鲁棒性（2026-10-22）

不再追加融合公式试验，本轮以已完成的 RMS 竞争融合为对照，实现改进书第 7.2 节的关系视图一致性与证据锚定候选。完整三模态编码、三轮交互、两轮读取、关系反馈和高阶机制均保留；不增加可训练参数。

`--relation-view-weight 0.1 --relation-dropout 0.2` 启用：一次完整模型计算后，复用第二轮证据，仅重算最后的标签关系层。遮蔽 20% 非自环的成对注意力边及经验/潜在因子；自环保留，原始支持度收缩不变。原完整关系视图停止梯度，监督遮蔽视图；关系前证据概率 ≤0.2 或 ≥0.8 时，额外约束最终预测。目标为 `L_view = KL(full → masked) + 0.25 KL(confident evidence → full)`，系数从第 6 轮起用 5 轮升至 0.1。

两项 KL 均使用既有逐标签 Bernoulli 一致性：仅作用于未标注项，教师高/低分预算为 1/0.1，已知正例仍由原分类目标监督，不被低教师分反向覆盖。不更改标签，不估计 PU 类别先验。锚点已经包含此前的反馈，是最后关系层之前的证据，不声称完全没有关系信息。遮蔽视图只用于训练；推理保持完整模型，不能把它描述为完整反馈图的两次独立前向。

旧配置默认关闭，旧 checkpoint 不增加 state key。恢复保留新目标设置；主辅梯度优化也包含新 Loss。记录 `relation_stability`、`relation_anchor`、`relation_view`，控制台每轮显示 `RelationView`。主实验仍使用原 AP 选模与 Val 全局 Jaccard 阈值，不采用另一候选模型的传统指标选模模式。

同步源码后，RTX5070 PowerShell：

```powershell
Set-Location 'D:\Data\ChatGPT\mfp\MFP-Nova'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_relation_views.ps1 -CpuThreads 4
```

V100 Bash：先激活 MFP-Nova；GPU_ID 为获准使用的物理卡，CPU_THREADS 可调整，nohup 支持关闭 SSH 后继续运行：

```bash
cd ~/wuzhencheng/projects/MFP-Nova
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MFP-Nova
GPU_ID=1
CPU_THREADS=4
tag=$(date +%Y%m%d_%H%M%S)
mkdir -p runs
nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" CPU_THREADS="$CPU_THREADS" PYTHON_BIN="$(command -v python)" \
    bash scripts/run_relation_views.sh > "runs/relationview_${tag}.log" 2>&1 < /dev/null &
printf 'PID=%s  Log=runs/relationview_%s.log\n' "$!" "$tag"
```

两个脚本默认串行训练完整 major15/fine39、seed42、650M、FP32、batch64、EMA0.999，其他候选 Loss 关闭；自动导出完整 Val 11 头并打包 metadata/results/history/概率，不含权重。只跑一项时 PowerShell 加 `-Tasks fine39`，Bash 在脚本路径后加 `fine39`。不自动启动服务器任务、不使用 Test，不覆盖历史运行目录。

## 2026-10-28：TheraSet 两任务 seed42 正式 Train/Val 结果

来源：`theraset_seed42_rtx5070_results_20261028_035242.zip`。代码 `6bef2d2a70859d83f7010b61b43dbc2df9c01cb3`，训练记录为干净工作区、数据2026-10-14、原样本、双冻结ESM缓存；RTX5070 / PyTorch2.8.0+cu128 / FP32 / batch64×累积1 / seed42。沿用ASL弱负项0.3、R-Drop0.1、EMA0.999、traditional5及Validation全局阈值。

| 项目 | major15 | fine39 |
|---|---:|---:|
| Train / Validation 样本数 | 39,478 / 8,534 | 39,455 / 8,529 |
| 训练结束 / 主候选epoch | 39 / 24 | 33 / 18 |
| 主候选权重 / 阈值 | EMA / 0.74 | EMA / 0.68 |
| Accuracy（样本Jaccard） | 65.0756% | 59.5821% |
| 样本Precision / Coverage | 69.6270% / 71.1531% | 63.5863% / 66.6297% |
| 完全匹配率AT / Hamming loss | 54.5114% / 5.4972% | 47.7664% / 2.8846% |
| Macro-AP / Micro-AP | 58.4839% / 74.4229% | 56.5149% / 69.0188% |
| Macro-F1 / Micro-F1 | 56.7611% / 68.5834% | 55.7485% / 64.1536% |
| 独立多功能检测F1 | 56.2757% | 54.0054% |
| 原始运行总耗时 | 76.92分钟 | 89.02分钟 |
| 第10轮起平均epoch耗时 | 124.06秒 | 171.40秒 |
| 正式运行显存峰值 | 4.20GiB | 7.50GiB |

检查了两任务主候选与AP候选的完整Validation概率：row_id、ID、cluster_id、标签顺序、目标和样本范围均与本地数据匹配，无重复或非有限概率；Train/Val簇交叉为0。使用现有指标代码复算，最大数值差为2.22e−16，重新拟合的全局阈值与保存结果一致。epoch连续、每轮617次更新、EMA更新总数、R-Drop预热和patience15均与配置一致，分项loss有限、AMP跳步为0。GPU profile在长度74、batch32/64、完整R-Drop下均成功，列出的关键模块梯度有限、非零且实际更新。

按预定Jaccard主目标应保留两个`best.pt`。AP候选为major15 epoch17（阈值0.73、Macro-AP59.2683%、Micro-AP74.8331%）和fine39 epoch16（阈值0.66、Macro-AP57.0042%、Micro-AP69.4111%），仅作为Validation备选，不能混用各候选最优数字。包内没有checkpoint，能核对的是概率、记录及诊断；没有重新运行权重前向，也没有Test结果。

与本地已有HelixSet双表征R-Drop/seed42的major15结果逐样本对齐：TheraSet主候选的Jaccard +0.1130个百分点、AT −0.2226个百分点、Macro-AP +0.9158个百分点、Micro-AP +2.3264个百分点、Micro-F1 +0.9032个百分点。5,013个Validation簇的2,000次配对簇bootstrap给出Jaccard差异95%区间[−0.8268,+1.0526]个百分点，AT差异区间[−1.5267,+1.0335]个百分点；均跨零，不能宣称整体严格优胜。该区间是在已选checkpoint/阈值上的探索性估计，不包含选模与跨seed不确定性。

TheraSet major15可训练参数较HelixSet减少85.53%，实测每轮约快6.16倍，正式运行峰值显存减少45.98%。这是两种实际执行配置（TheraSet64×1、HelixSet16×4）的联合效率结果，不是固定物理batch下的纯架构速度对照，且不含新序列ESM编码成本。

分组取舍：major15的D注记组173条Jaccard +12.09个百分点，缺结构组1,040条 +1.98，2–8aa组 +1.07、9–16aa组 +1.11；17–32aa组 −1.86、AT −2.98。小分组为探索结果。major15唯一Train尾类thrombolytic的AP从HelixSet29.94%降至21.84%；不能从整体Macro-AP改善推断尾类都改善。

两任务Validation ASL均在epoch8最低，此后训练loss继续下降、Validation loss持续上升；最终训练Macro-AP约99.30%/99.11%，Validation Macro-AP58%左右/55%左右，存在明显过拟合。loss最低epoch与Jaccard/AP最优epoch不同不构成选模错误；当前记录支持保留早期候选，不支持继续单纯延长训练。

弱项包括fine39 antiinflammatory：Val40个正例只召回1个，AP2.14%、F1 3.03%；growth_regulatory召回7/36，thrombolytic__unknown召回7/37。独立多功能检测仍漏掉major15的885/2,118和fine39的908/2,108；真正同时恢复至少两个已标注独立功能的比例仅47.69%/46.54%。0为未标注，统计中的FP不是经实验确认的无活性。

结论：这两次是有效的完整seed42 Train/Val结果，支持TheraSet作为较低成本候选继续比较；尚不能证明对TheraPep适配CNN的净增益，也不能确认各创新模块有效或跨seed稳定。下一步先补CNN对照及有针对性的消融，再做统一多Seed和一次独立Test评价。分析数据与图保存在`runs/analysis/theraset_seed42_20261028/`，不提交运行输出。

### 2026-11-08：TermiScale fine39 / seed42 V100首轮正式训练日志

来源为用户回传终端日志，运行名termiscale_fine39_seed42_v100_20261108_201826。尚未取得本次results.json、metadata.json、概率或checkpoint，以下不是独立重载复算结果。单650M缓存、AA34、FP32、batch64×累积1、workers2、OMP/MKL6、物理GPU1；无Test结果。

日志覆盖epoch1–45。第30轮后15轮选模目标未改进，第45轮达到正常早停条件；完成落盘仍以results.json为准。主选模traditional5/global，EMA；第30轮阈值0.62：Val样本Jaccard50.556%、完全匹配34.881%、样本Prec54.22%、Cov62.48%、Hamming3.62%，该候选Macro-AP约45.44%。这些是Selection阈值下的结果；Val行固定0.5的监控值不混入。第21轮可见候选Macro-AP约46.34%，AP备选最终以best_ap记录确认，不把它拼到第30轮主候选上。

收敛：Val主ASL在第13轮约0.0211最低，随后升至第45轮0.0384；Train总loss持续降至0.0028，后期出现明显过拟合。Train为随机在线权重、Val多为EMA，二者并非同一确定模型的精确训练/验证差；但loss与排序/选模走势共同支持泛化停滞。第6轮R-Drop从0.02启动，第10轮达0.1；EMA每轮增加617次，至第45轮24,680次，与617更新/轮一致。

资源：第6–45轮epoch Time中位40.45秒、Train22.75秒、Val11.2秒、DataWait0.3秒；后三者中位数不能直接相加解释同一轮全部时间。第30–45轮epoch中位38.55秒；日志累计约31分钟，完整R-Drop/EMA峰值allocated约0.77GiB。不是整卡显存或包括全部启动/收尾成本的端到端计时。

性能参照：既有V100 fine39/traditional5 TheraSet纯AA34 CNN为Jaccard57.01%、AT44.54%、Macro-AP51.38%；完整TheraSet为58.72%/47.12%/54.97%。本次主候选分别落后纯CNN约6.45/9.66/5.94个百分点；两者架构、输入与初始化轨迹不完全相同，不当作单个模块的净效应。当前可以确认低开销，尚不支持性能超越。

后续先取得完整训练产物确认选模/数据范围，并按原协议完成MotifSpan。TermiScale优先比较均匀LME、固定端点混合、静态尺度等必要消融及同输入强CNN，定位端点/路由归纳偏置；不因单次落后直接扩容、增大有效batch或改Test选模。不把这份日志当成已有五seed或独立Test结论。

### 2026-11-09：新候选四组 fine39 / seed42 后续实验

用户回传四份正式运行的 `summarize --split val` 输出。共享单650M+AA34、FP32、batch64×1、R-Drop/EMA 和 `traditional5/global`，均为单 seed。三项选模相关指标取同一主候选。耗时为训练器记录的整次运行时长，显存为 CUDA 峰值已分配量。

| 实验 | Jaccard % | AT % | Macro-AP % | 多功能检测 F1 % | 分钟 | GiB |
|---|---:|---:|---:|---:|---:|---:|
| MotifSpan 主配置 | 51.70 | 35.28 | 49.50 | 47.20 | 273.8 | 0.379 |
| TermiScale 均匀池化 | 50.33 | 34.19 | 45.58 | 47.04 | 90.2 | 0.763 |
| TermiScale 去样本条件混合 | 50.59 | 35.07 | 45.47 | 46.95 | 100.8 | 0.766 |
| TermiScale 等权多尺度 | 50.73 | 34.72 | 45.58 | 48.46 | 97.0 | 0.622 |

MotifSpan 主候选为第43轮 EMA，阈值0.66，第58轮早停。相对等权多尺度，39个标签中30个 AP 提升，尾部组 Macro-AP 提升4.13个百分点，Jaccard 提升0.96个百分点，多功能检测 F1 降低1.26个百分点。三个 TermiScale 对照与首轮主配置接近。下一组运行 `--motifspan-global-only`，以相同输入和训练协议检验完整 MotifSpan 三类附加读出的联合增益。运行命令见 [README](../README.md#motifspan-同输入全局读出对照)。
