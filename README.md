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
configs/sequence.yaml      # 一个初始配置，两任务通过 --track 切换
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

推荐 Python 3.12。GPU 构建按设备选择：V100 从已存在的 `trifusion_mfp` 克隆出独立的 `MFP-Nova` 环境（PyTorch 2.4.1+cu121），开发机和 RTX 5070 使用 PyTorch 2.8.0 的适用构建。当前代码所需的 PyTorch API 在 2.4 中已有，项目最低依赖调整为 2.4.1；这属于 API 检查结论，服务器上的实际链路仍需测试。安装命令依据 [PyTorch 官方历史版本页](https://pytorch.org/get-started/previous-versions/)，`weights_only` 接口见 [PyTorch 2.4 文档](https://docs.pytorch.org/docs/2.4/generated/torch.load.html)。

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

Windows RTX 5070：通过 UU 远程桌面在该机器打开 PowerShell，建议项目放在用户目录的 `Projects/MFP-Nova`。该机器需要已安装 Git 和 Python 3.12。首次安装执行：

```powershell
$taskProjectParent = Join-Path $env:USERPROFILE 'Projects'
New-Item -ItemType Directory -Path $taskProjectParent -Force | Out-Null
Set-Location $taskProjectParent
git clone https://github.com/zc-pro/MFP-Nova.git
Set-Location MFP-Nova
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install numpy==1.26.4 transformers==4.44.2 scikit-learn==1.5.2 PyYAML==6.0.2
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\mfp-nova.exe --help
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
.\.venv\Scripts\python.exe -c "import torch; x=torch.ones(8,8,device='cuda'); print('CUDA matmul:', (x@x).sum().item())"
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

RTX 的命令直接调用 `.venv` 中的可执行文件，无需激活；如使用 Conda，则在该机器单独创建 Python 3.12 环境并执行同样的 pip 安装步骤。GitHub 是私有仓库，按 Git 提示完成各自的 HTTPS 认证。CUDA 可用性应为 `True`，RTX 的小矩阵核验证应输出 `512.0`。服务器端安装和模型测试尚未实际确认，不把本机 2.8.0 CPU 测试当作 V100 2.4.1 或 RTX 5070 验证。

数据经 UU 传到 RTX 5070，经 SFTP/SCP 传到 V100；完整解压后，两端建议目录如下。外部数据盘可以通过 `--data-root` 指定。

```text
MFP-Nova/
  .venv/                     # Windows venv 可放这里；V100 使用独立 Conda 环境 MFP-Nova
  data/therapeutic_2025/      # 迁移包解压内容，直接包含 records.csv、arrays/、tracks/
  features/                  # 后续本机提取的特征
  runs/                      # 该机器的实验输出
```

## 使用

从仓库根目录执行命令，相对路径以当前目录为基准。服务器可用 `--data-root`、`--feature-root` 设置本机路径，无硬编码盘符。

```text
mfp-nova analyze --data-root data/therapeutic_2025 --output docs/data_summary.json
mfp-nova features --device cuda:0 --splits train val
mfp-nova train --device cuda:0 --track fine39 --name fine39_seq_seed42
mfp-nova train --device cuda:0 --track major15 --name major15_seq_seed42
```

首次特征提取会下载 ESM-2 35M。默认仅提取 Train/Validation，原样本缓存两任务共用。提取 FP32、存储 FP16，使用 NPY ragged 残基、offsets、row_id、ID 与序列，读取采用 mmap，不重复保存全局向量。模型权重 revision 自动记录；后续推理加载同一 revision。正式实验可将配置 revision 改为已记录的确定 commit。特征目录已经存在时不覆盖，提取中断后使用新缓存目录。

训练仅打开 Train/Validation 缓存，按 Validation Macro-AP 选最佳 checkpoint，Micro-AP 打破同分，完全相同则保留更早 epoch。训练后加载最佳模型，在 Validation 上拟合全局 Jaccard 阈值；结果中的 Val F1/Jaccard 为阈值拟合内估计。Test 未参与训练、选模或阈值拟合。输出 `runs/<name>/best.pt`、`history.jsonl`、`metadata.json`、`results.json`，已有运行目录拒绝覆盖。

确定方案后，独立提取并评价 Test：

```text
mfp-nova features --device cuda:0 --splits test
mfp-nova evaluate --checkpoint runs/fine39_seq_seed42/best.pt --device cuda:0 --output runs/fine39_seq_seed42/test.json
mfp-nova predict --checkpoint runs/fine39_seq_seed42/best.pt --sequences peptides.txt --device cuda:0 --output predictions.json
```

`peptides.txt` 每行一个明确、非空、大写 20-AA 骨架；非标准化学表示不进行猜测转换，超过 token 上限明确报错，不静默截断。推理返回所有标签概率和阈值通过的标签，可全部为阴性。原始修饰肽的化学身份需要独立处理。

消融直接修改配置并使用新实验名：`model.use_local: false` 为全局特征对照，`loss.name: bce` 为损失对照。增强 `data.augment: true` 仅使用当前任务提供的训练候选，需要新的含增强特征目录；fine39 为 648 条，major15 为 60 条。暂未实现 CV/增强 CV，不将固定划分增强用于开发折。

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
| V100 ↔ GitHub | HTTPS Git，在服务器独立完成 GitHub 认证后 clone/pull |
| RTX 5070 ↔ GitHub | HTTPS Git，在 RTX 5070 上通过 UU 远程操作终端完成 clone/pull |
| 本机 ↔ RTX 5070 大文件 | 用户指定的 UU 远程文件传输，同步数据、特征、权重和完整运行输出 |

本机与 RTX 5070 不建立 SSH 连接，也不需要为此配置 sshd、端口或公钥。V100 使用密码登录，不安装公钥。两台训练端的项目目录在首次登录/UU 远程操作后确定。本机已将 `MFP-Nova-V100` 别名调整为交互式密码认证，并移除了 RTX 5070 项目 SSH 别名。

V100 的 SSH 密码在本机终端提示中输入。私有 GitHub 仓库的 HTTPS 认证在每台机器各自完成：优先使用已有 Git Credential Manager/浏览器登录；终端需要 HTTPS 凭据时使用 GitHub token，不使用 GitHub 账户密码。凭据不放入 clone URL、脚本、配置或 Git 仓库。现有本机 GitHub 登录无需重新设置。

两台服务器的安装命令见上面的安装章节：V100 已确认项目位于 `~/wuzhencheng/projects/MFP-Nova`，从 `trifusion_mfp` 克隆创建独立 Conda 环境 `MFP-Nova`；RTX 5070 建议项目位于 `$env:USERPROFILE/Projects/MFP-Nova`，在 Windows 上独立创建 `.venv` 或 Conda 环境。RTX 目录与环境尚未实际确认；尚未执行正式 GPU 训练。

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

当前提供数据与框架基础，不宣称模型效果。下一步是 GPU 端特征提取和单 seed 基线，随后针对局部分支/损失做单变量对照；结构、D/PTM 表示和标签专属学习均留待有依据的后续研究。

本地已用 Python 3.12、PyTorch 2.8.0 CPU 完成 6 项针对性测试：真实数据对齐/划分、padding 不变性、ASL 数值和梯度、指标语义、冻结编码器接口与缓存写入、两 epoch 合成训练和 checkpoint/评价加载。编码器接口测试使用替身，不下载预训练权重；这些检查不代表真实 ESM 特征或 GPU 性能验证。pytest 临时文件配置在 `.pytest_cache/tmp`，避免本机系统临时目录的权限问题。
