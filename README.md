# MFP-Nova

独立的治疗肽多标签识别科研项目。已实现序列对照与 TriRelay、GeoFactor、ProtoRoute 三套多模态模型，并完成 RTX 5070 上六组 seed=42 正式训练。当前完善逐轮指标和分批五 Seed 实验；所有核心代码独立编写。

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

## 三套完整研究模型

### 快速使用：RTX 5070 分批五 Seed

固定 Seed 为 **42、101、202、303、404**，三模型 × 两任务共 30 组。已有 seed=42 六组正式结果直接复用，新增 24 组保持 **ESM-650M、FP32、batch 64、无梯度累积**，沿用现有架构、Loss、学习率计划、100 epoch 上限和 patience=15。仅使用 Train/Validation，不启动 Test；不根据阶段性排名修改 Seed 或协议。

在 RTX 无 MFP 训练进程运行时执行 `git pull --ff-only`，然后在项目根目录 `D:\Data\ChatGPT\mfp\MFP-Nova` 的已激活 `MFP-Nova` PowerShell 中使用以下命令。已有 editable 安装无需重装依赖。本机用于开发与整理结果，不执行正式 GPU 训练。

```powershell
# A. 直接训练一组；完成后会被批量脚本识别并跳过
python -m mfp_nova.cli train --model trirelay --task major15 --esm 650m --seed 101 --device cuda:0 --precision fp32 --batch-size 64 --name trirelay_major15_esm650m_seed101_rtx5070_fp32_b64

# B. 查看 30 组的状态与剩余计划；不加载 checkpoint、不启动训练
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

# J. 随时查看阶段性 mean ± 样本 std；30 组完成后同命令给出五 Seed 对比
python -m mfp_nova.experiments summary --task major15
python -m mfp_nova.experiments summary --task fine39

# 同一个 Seed 的三模型对比
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

完整定义、计算公式、取值范围、零分母处理及指标间联系见 [指标说明书](docs/MFP-Metrics-Guide.md)。

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

RTX 5070 已完成三模型 × major15/fine39 的 ESM-650M、seed 42 全量训练，统一使用 `--batch-size 64 --precision fp32`。Validation Macro-AP 分别为 TriRelay 0.540361/0.518048、GeoFactor 0.528925/0.506050、ProtoRoute 0.537225/0.514995。完整指标和资源成本见 [实际验证与训练记录](docs/MFP-Real-Data-GPU-Validation.md#7-rtx-5070-首轮全量训练与下一步对照)。目前尚无三模型的 35M 全量结果。

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

默认依次执行 TriRelay → GeoFactor → ProtoRoute，每模型完成 major15/fine39。复用服务器已有真实缓存，读出 ID/骨架与缓存 row_id 对齐；选取结构存在/缺失、非甘氨酸 D、不同长度及标签覆盖的少量原始样本。模型深度、专家、反馈及因子全部保留。诊断优化子集不会重拟合缩小标签先验：标准化、长尾分层、高阶关系和原型容量继续来自完整原始 Train。

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

需要单 Seed 训练对比时，在 RTX 5070 空闲且更新代码后单独运行（不使用原五 Seed 批量脚本）：

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

默认 0.005 Jaccard 预算选择全局 0.61；MFP 真阳性增加 160、假阳性增加 361，AT 下降 2.34 个百分点，不建议直接替换主协议。零下降预算选择功能阈值 0.69；减少 66 个假阳性，同时损失 17 个真阳性，MFP-F1 仅提高 0.177 个百分点，保留为低成本候选，不能宣称稳定提升。后者另存 `decision_balanced.json`，未修改默认阈值或已有权重。两种预算各自的两个簇分组拟合均选择相同参数，因此各自交叉拟合汇总恰好等于全 Val 拟合结果；这不消除既有 checkpoint / 原阈值全 Val 选择的偏差。此结果仅涉及单模型、单任务、单 Seed；重建日程的真实训练比较仍未执行，下一步只运行上方一个候选训练，先不扩展到 30 组。
