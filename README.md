# MFP-Nova

独立、轻量的治疗肽多标签识别科研框架。当前完成数据接入和最小训练链路，尚未开展真实 GPU 性能实验。技术蓝图用于参考已验证的算法思想；所有核心代码独立编写。

## 本地数据与任务

本地迁移包完整解压至 `data/therapeutic_2025/`，原包保留。数据目录包含上游处理脚本，但本项目不调用或复制这些实现。

实际读取统计见 [data_summary.json](docs/data_summary.json)。原记录 58,583 条，增强候选 648 条，共 59,231 行；默认增强关闭。标签来自处理后的 Function 语义，不使用原 48 位 `Label encoding`。

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

Windows RTX 5070：通过 UU 远程桌面在该机器打开 Anaconda PowerShell Prompt 或已能运行 Conda 的 PowerShell，建议项目放在用户目录的 `Projects/MFP-Nova`。下面代码使用 PowerShell 语法；Git clone 应单独执行，等认证和克隆完成后再执行后续命令。该机器需要已安装 Git 和 Conda。环境配置为 [environment-rtx5070.yaml](configs/environment-rtx5070.yaml)，Conda 创建 Python 3.12 并通过 pip 安装指定 CUDA 12.8 构建；固定 `torch==2.8.0+cu128`，以明确选择 GPU wheel。首次安装执行：

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

两台服务器的安装命令见上面的安装章节：V100 项目位于 `~/wuzhencheng/projects/MFP-Nova`，独立 Conda 环境 `MFP-Nova` 已完成克隆和安装；RTX 项目位于 `D:\Data\ChatGPT\mfp\MFP-Nova`，环境位于 `D:\App\Code\Anaconda\envs\MFP-Nova`。RTX 已完成 ESM-35M/650M 特征提取及 major15/fine39 单 seed 序列模型训练。三套新模型本轮在本地实现，没有自动部署到服务器或启动正式 GPU 训练。

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

## 三套完整研究模型

实现与数学口径集中在 [下一代模型技术说明](docs/MFP-Next-Generation-Architecture.md#8-已实现架构与统一实验协议)。三模型共享原始明确骨架数据、固定标签顺序、Train-only 统计和评价协议，保持独立架构。默认 ESM-650M、隐藏维 256、8 heads；major15/fine39 的下游参数分别约为 TriRelay 35.20/35.21M、GeoFactor 31.00/31.01M、ProtoRoute 32.33/32.45M，不含冻结 ESM。

在项目根目录、已激活环境中运行：

```bash
mfp-nova train --model trirelay --task major15 --seed 42 --device cuda:0
mfp-nova train --model geofactor --task major15 --seed 42 --device cuda:0
mfp-nova train --model protoroute --task major15 --seed 42 --device cuda:0
```

把 `--task` 改为 `fine39` 即可切换任务。也可使用 `python -m mfp_nova.cli`。模型独立配置分别为 `configs/trirelay.yaml`、`configs/geofactor.yaml`、`configs/protoroute.yaml`；默认输出 `runs/<model>_<task>_seed<seed>/`，已存在时明确拒绝覆盖。

`--esm 35m` 同时选择 35M 编码器和其缓存；`--feature-root` 可覆盖缓存路径。两种编码器比较请使用不同 `--name`：

```bash
mfp-nova train --model trirelay --task fine39 --esm 35m --seed 42 --device cuda:0 --name trirelay_fine39_esm35m_seed42
mfp-nova train --model trirelay --task fine39 --esm 650m --seed 42 --device cuda:0 --name trirelay_fine39_esm650m_seed42
```

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

每个 run 记录配置、seed、Git commit/dirty、数据和编码器标识、参数量、最佳 epoch、Loss 分项、LR、耗时及 CUDA 峰值分配显存。正式多 seed 可在训练端依次运行，例如 PowerShell：

```powershell
foreach ($experimentSeed in 42, 123, 456) {
    mfp-nova train --model trirelay --task major15 --seed $experimentSeed --device cuda:0
}
```

汇总不启动训练、不按 Test 选模型。各 run 的 Test 文件统一命名 `test.json`：

```bash
mfp-nova summarize --runs runs/trirelay_major15_seed42 runs/trirelay_major15_seed123 runs/trirelay_major15_seed456 --split val --output results/trirelay_major15_val.json
mfp-nova summarize --runs runs/trirelay_major15_seed42 runs/geofactor_major15_seed42 runs/protoroute_major15_seed42 --split test --output results/major15_test_comparison.json
```

同任务/编码器/数据队列/配置协议的 seed 汇总 mean ± 样本 std；不同模型、任务、消融、Loss、阈值协议分别列出，不混合平均。只有一个 seed 时 std=0，并记录有效 seed 数。支持传统指标、核心指标、逐标签指标及参数/显存/时间；诊断子集不进入正式汇总。

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

默认依次执行 TriRelay → GeoFactor → ProtoRoute，每模型完成 major15/fine39。复用服务器已有真实缓存，读出 ID/骨架与缓存 row_id 对齐；选取结构存在/缺失、非甘氨酸 D、不同长度及标签覆盖的少量原始样本。模型深度、专家、反馈及因子全部保留。诊断优化子集不会重拟合缩小标签先验：标准化、长尾分层、高阶关系和原型容量继续来自完整原始 Train。

每个 batch/精度先 warmup，再计时 3 step，记录参数量、峰值 allocated/reserved 显存、前向/反向/optimizer 时间、含数据准备的 step 时间与吞吐、按代表 batch 粗估的训练 epoch 时间、padding/缺结构输出和关键模块梯度/更新。OOM 只记录失败批量并停止继续增大该精度，不删模块。Profiler 的代表 batch 和计时开销会影响估计，不能替代正式整轮吞吐。

随后使用成功且关键模块有梯度更新的实测配置，运行 32 个 optimizer step、Diagnostic Validation、checkpoint，再恢复 2 step，并从恢复的最佳模型独立计算 Diagnostic Validation。小样本指标不用于性能结论。`summary.json`、各模型任务的 `*_profile.json`、`history.jsonl` 和 checkpoint 都保存在指定的新目录；目录存在则拒绝覆盖。不会读取 Test 缓存或启动正式多 seed。

若 V100 尚无当前 MFP 数据包/ESM 缓存，先通过 UU 回传到本机，再用密码认证 SFTP/SCP 转移现有文件；也可指定 `--data-root`、`--feature-root` 指向当前项目的数据和缓存。不能把旧项目缓存、随机特征当成这次真实验证。35M 可用 `--esm 35m` 单独执行，使用不同输出目录。

正式训练入口另支持：

```bash
mfp-nova train --model trirelay --task major15 --precision fp16 --batch-size 8 --accumulation-steps 2 --device cuda:0 --name trirelay_major15_esm650m_fp16_seed42
```

`--max-steps` 是诊断上限，按 optimizer 尝试步计数，AMP 跳过的更新单独记录；含此参数的 run 不进入正式比较。恢复诊断运行可增大 `--max-steps`，从下一轮开始继续，不承诺部分 epoch 内逐样本精确续接。最终 batch/精度应依据两台服务器各自实测选择，不把另一台或本机的结果代入。
