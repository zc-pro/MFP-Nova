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

推荐 Python 3.12。先在独立环境安装适合设备的 PyTorch，再安装项目。以下选择固定 2.8.0，避免两种 GPU 默认安装到不适合的 CUDA 构建；不是要求所有机器使用完全一致的环境。安装命令依据 [PyTorch 官方历史版本页](https://pytorch.org/get-started/previous-versions/)。

Windows 开发机（CPU）：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pytest -q
```

Windows RTX 5070：在该设备新建同样的 Python 环境，将 PyTorch 安装命令换为：

```powershell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]"
```

Ubuntu V100：使用已有 Python 3.12 或可用的 Python ≥3.10，不修改系统 CUDA。此构建需要服务器驱动支持；当前未检查远程驱动。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[dev]'
```

两 GPU 端在实际训练前检查设备是否可用（此命令不启动训练）：

```text
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
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

服务器先设置各自的 GitHub 只读 Deploy Key，再 `git clone git@github-mfp-nova:zc-pro/MFP-Nova.git`；后续在训练停止后 `git pull --ff-only`，正式实验可 `git checkout <COMMIT>`。用 `ssh <USER>@<HOST>` 登录后，在服务器项目目录运行上述训练命令。共享 V100 需按服务器使用约定取得资源，不自动启动远程实验。

三机连接使用两种独立身份：本机 SSH 公钥用于登录两台服务器；每台服务器自己的 SSH 公钥作为该私有仓库的只读 Deploy Key，用于拉取代码。私钥留在生成它的机器，GitHub 登录凭据不写入项目或复制到服务器。

服务器各自在用户目录生成 GitHub 密钥，已有同名文件时直接使用，勿覆盖：

```text
ssh-keygen -t ed25519 -f <USER_SSH_DIR>/id_ed25519_mfp_github -C mfp-nova-server
```

将对应 `.pub` 的内容添加到仓库 Settings → Deploy keys，关闭 Allow write access。每台服务器在自己的用户 SSH config 中追加以下项目别名，使用各自生成的私钥：

```text
Host github-mfp-nova
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_mfp_github
  IdentitiesOnly yes
```

执行 `ssh -T github-mfp-nova` 验证；成功认证但不提供 shell 时退出码为 1，这是 GitHub 的正常行为。首次连接应核对 [GitHub 公布的主机指纹](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints)。

本机已生成专用登录密钥 `%USERPROFILE%/.ssh/id_ed25519_mfp_nova`，用于本机到服务器的 SSH 连接，不放入 Git 仓库。已在本机用户 SSH config 设置两个项目专用别名，均使用该密钥：

| 训练端 | 本机 SSH 别名 | 连接地址 | 项目目录 |
|---|---|---|---|
| Ubuntu V100 | `MFP-Nova-V100` | `ubuntu@192.168.13.171:22` | 登录后确认 |
| Windows RTX 5070 | `MFP-Nova-RTX5070` | `wuzhencheng@10.1.44.34:22` | 登录后确认 |

Ubuntu 主机已确认，网络可达，但公钥/非交互认证尚未成功。可在本机 PowerShell 执行以下命令，用当前服务器密码登录并追加项目公钥：

```powershell
Get-Content "$env:USERPROFILE\.ssh\id_ed25519_mfp_nova.pub" | ssh MFP-Nova-V100 'umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys; chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys'
```

密码在 SSH 的终端提示中输入，不写入文件或聊天。可用 `ssh -o BatchMode=yes MFP-Nova-V100 "id -un"` 检查登录。

RTX 5070 的 SSH 端口目前从本机连接超时；本机为 `10.1.44.154/23`，与目标地址处于同一网段。需要在该机器确认当前 IP、`Get-Service sshd` 和 `Get-NetTCPConnection -LocalPort 22 -State Listen` 的结果，恢复 SSH 连通后再进行项目搭建。端口恢复后首次连接需确认该服务器主机身份，再为 `wuzhencheng` 设置公钥登录；Windows 管理员账号与普通账号的公钥位置可能不同，应根据实际账号权限和 sshd 配置处理。

Windows RTX 5070 若尚未提供 SSH 服务，应由该机器管理员按 [Microsoft OpenSSH Server 安装说明](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse) 启用服务与防火墙，再提供用户名、地址、端口。本项目不会自动修改远程系统服务或防火墙。

Git 只同步源码、小型配置和统计。数据迁移包、特征、权重、checkpoint 和运行输出已忽略；可通过 SFTP/SCP 独立传输原迁移包并完整解压。例如 `scp therapeutic_2025_跨电脑开发迁移包.zip <USER>@<HOST>:<DATA_DIR>/`；结果回传可用 `scp -r <USER>@<HOST>:<PROJECT_DIR>/runs/<RUN_NAME> ./runs/`，本机选择新的运行名避免覆盖。不要在运行中的项目目录更新源码。

当前提供数据与框架基础，不宣称模型效果。下一步是 GPU 端特征提取和单 seed 基线，随后针对局部分支/损失做单变量对照；结构、D/PTM 表示和标签专属学习均留待有依据的后续研究。

本地已用 Python 3.12、PyTorch 2.8.0 CPU 完成 6 项针对性测试：真实数据对齐/划分、padding 不变性、ASL 数值和梯度、指标语义、冻结编码器接口与缓存写入、两 epoch 合成训练和 checkpoint/评价加载。编码器接口测试使用替身，不下载预训练权重；这些检查不代表真实 ESM 特征或 GPU 性能验证。pytest 临时文件配置在 `.pytest_cache/tmp`，避免本机系统临时目录的权限问题。
