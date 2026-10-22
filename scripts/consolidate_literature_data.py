"""Group published benchmarks by dataset, retaining their original split protocols.

This only prepares data files. It does not import models or start experiments.
"""
import argparse
import csv
import gzip
import pickle
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.core.multiarray import _reconstruct, scalar


MTP_LABELS = "AMP TXP ABP AIP AVP ACP AFP DDV CPP CCC APP AAP AHTP PBP QSP".split()
MFTP_LABELS = "AAP ABP ACP ACVP ADP AEP AFP AHIVP AHP AIP AMRSAP APP ATP AVP BBP BIP CPP DPPIP QSP SBP THP".split()
MFBP_LABELS = "ACP ADP AHP AIP AMP".split()
MFL_LABELS = "AMP ACP ADP AHP AIP".split()


def read_fasta(path):
    records, name, fragments = [], None, []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(fragments)))
            name, fragments = line[1:], []
        elif line.strip():
            fragments.append(line.strip())
    if name is not None:
        records.append((name, "".join(fragments)))
    return records


def bit_records(path, width):
    records = []
    for bits, sequence in read_fasta(path):
        if len(bits) != width or set(bits) - {"0", "1"}:
            raise ValueError(f"Invalid label encoding in {path}: {bits}")
        records.append((sequence, tuple(map(int, bits))))
    return records


def mtp_records(folder, split):
    sequences = read_fasta(folder / split / "seqs.fasta")
    with (folder / split / "labels.csv").open(encoding="utf-8-sig", newline="") as f:
        labels = list(csv.reader(f))
    if labels[0] != MTP_LABELS or len(sequences) != len(labels) - 1:
        raise ValueError(f"Sequence/label alignment differs in {folder / split}")
    return [(sequence, tuple(map(int, row))) for (_, sequence), row in zip(sequences, labels[1:])]


class NumpyDatasetReader(pickle.Unpickler):
    """Read the author's NumPy dictionaries without enabling arbitrary pickle globals."""

    def find_class(self, module, name):
        allowed = {("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype,
                   ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
                   ("numpy.core.multiarray", "scalar"): scalar}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported dataset object: {module}.{name}")
        return allowed[module, name]


def mfbp_records(path):
    with path.open("rb") as f:
        np.lib.format.read_magic(f)
        np.lib.format.read_array_header_1_0(f)
        data = NumpyDatasetReader(f).load().item()
    return [(sequence, tuple(map(int, values))) for sequence, values in data.items()]


def mfl_records(path):
    order = [MFL_LABELS.index(label) for label in MFBP_LABELS]
    records = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            values = list(map(int, row["label"].split(",")))
            records.append((row["text_a"], tuple(values[i] for i in order)))
    return records


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_benchmark(output, name, labels, protocols):
    # Identical sequences with different labels remain different records. No
    # annotation union, negative inference, or reassignment between splits.
    pool = sorted({record for splits in protocols.values() for rows in splits.values() for record in rows})
    ids = {record: f"{name}_{i:06d}" for i, record in enumerate(pool, 1)}
    folder = output / name
    write_csv(folder / "samples.csv", ["sample_id", "sequence", *labels],
              ((ids[record], record[0], *record[1]) for record in pool))
    stats = []
    for protocol, splits in protocols.items():
        memberships = ((ids[record], split, i) for split, rows in splits.items() for i, record in enumerate(rows))
        write_csv(folder / "splits" / f"{protocol}.csv", ["sample_id", "split", "source_row"], memberships)
        split_names = list(splits)
        overlap = sum(len({s for s, _ in splits[a]} & {s for s, _ in splits[b]})
                      for i, a in enumerate(split_names) for b in split_names[i + 1:])
        sizes = "/".join(f"{split}={len(rows):,}" for split, rows in splits.items())
        stats.append(f"| {name} | {protocol} | {len(pool):,} | {sizes} | {overlap} |")
    return stats


def consolidate(source, output):
    mtp = {split: mtp_records(source / "tppred_le/datasets/data", split) for split in ("train", "val", "test")}
    for split, records in mtp.items():
        variants = [mtp_records(source / "sa_mtp/datasets/data", split),
                    mtp_records(source / "tppred_cmvl/datasets", split),
                    bit_records(source / "metfan" / f"{split}.txt", 15)]
        if any(records != other for other in variants):
            raise ValueError(f"MTP15 published versions differ in {split}; keep them separate")
    mftp = {split: bit_records(source / "imfp_lg/data/MFTP" / f"{split}.txt", 21) for split in ("train", "test")}
    for split, records in mftp.items():
        for directory in ("amcl/dataset", "mcmfpp/MFTP"):
            if records != bit_records(source / directory / f"{split}.txt", 21):
                raise ValueError(f"MFTP21 published versions differ: {directory}/{split}")
    mfbp = {split: mfbp_records(source / "imfp_lg/data/MFBP" / f"{split}_data.npy") for split in ("train", "test")}
    cleaned = {split: mfl_records(source / "mfp_mfl/data" / f"{split}.tsv") for split in ("train", "test")}
    original_pool = {record for rows in mfbp.values() for record in rows}
    cleaned_pool = {record for rows in cleaned.values() for record in rows}
    if original_pool != cleaned_pool:
        raise ValueError("MFBP5 original and cleaned sample pools differ; keep them separate")

    output.mkdir(parents=True)
    stats = []
    stats += write_benchmark(output, "MTP15", MTP_LABELS, {"published": mtp})
    stats += write_benchmark(output, "MFTP21", MFTP_LABELS, {"published_raw": mftp})
    stats += write_benchmark(output, "MFBP5", MFBP_LABELS,
                             {"imfp_lg_original": mfbp, "mfp_mfl_cleaned": cleaned})

    augmented = bit_records(source / "amcl/dataset/augmented_train.txt", 21)
    path = output / "MFTP21/amcl_augmented_train.csv.gz"
    with path.open("xb") as raw, gzip.open(raw, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_row", "sequence", *MFTP_LABELS])
        writer.writerows((i, seq, *target) for i, (seq, target) in enumerate(augmented))

    tasks = []
    for task in sorted((source / "pdeeppp/data").iterdir()):
        if not task.is_dir():
            continue
        files = sorted(task.glob("*.xlsx"))
        counts = Counter()
        for file in files:
            split = "train" if "train" in file.stem.lower() else "test" if "test" in file.stem.lower() else None
            if split is None:
                raise ValueError(f"Unrecognized PDeepPP split: {file}")
            counts[split] += 1
            target = output / "PDeepPP" / task.name / f"{split}.xlsx"
            target.parent.mkdir(parents=True, exist_ok=True)
            with file.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        if counts != {"train": 1, "test": 1}:
            raise ValueError(f"Incomplete PDeepPP task: {task.name}")
        tasks.append(task.name)

    description = """# 按数据集归并的同类论文数据

2026-10-22。以数据集为单位整理 `../literature_benchmarks/` 中已经下载的文件。同一数据集只保存一份样本表，并列出使用它的论文；保留不同的原始划分协议。各数据集之间没有融合，也没有加入 `therapeutic_2025/`。

## 目录与论文对应

| 目录 | 使用该数据集的论文 | 归并依据 |
|---|---|---|
| `MTP15/` | TPpred-LE、METFAN、SA-MTP、TPpred-CMvL | 四者发布的 Train/Val/Test 序列、15 位标签及记录顺序逐条一致；合并为一份 `samples.csv` 和 `splits/published.csv` |
| `MFTP21/` | iMFP-LG、AMCL、MCMFPP | 三者发布的原始 Train/Test 序列、21 位标签及记录顺序逐条一致；合并为一份样本表和 `splits/published_raw.csv` |
| `MFBP5/` | iMFP-LG、MFP-MFL | 二者的唯一序列与标签集合完全相同；共用一份 5,917 条样本表，分别保留原始与清洗后的划分文件 |
| `PDeepPP/<任务名>/` | PDeepPP | 32 个独立任务分别保存，各保留 `train.xlsx` 和 `test.xlsx`；main/alternative 任务保持独立 |

## 样本与划分

前三个目录使用相同接口：

- `samples.csv`：`sample_id,sequence,<各标签列>`。标签列名就是位编码的含义。按序列与标签向量共同去重；不合并标签、不删除非标准残基、不推断阴性、不重新划分。
- `splits/*.csv`：`sample_id,split,source_row`。`source_row` 为对应上游划分中的零起始记录序号，保留原始顺序；按 `sample_id` 关联样本表即可恢复该协议。
- 每个数据集的 ID 仅用于本目录，不与当前项目 `row_id` 混用。标签值沿用原作者多标签编码，0 不扩展解释为实验阴性。

| 数据集 | 划分协议 | 唯一样本记录数 | 原始划分记录数 | 跨划分相同序列对数 |
|---|---|---:|---|---:|
""" + "\n".join(stats) + """

### MFBP5 的两种划分

原版 iMFP-LG 的 Train/Test 共 5,986 条记录，其中有 69 条序列同时存在于 Train 与 Test，且标签相同，因此唯一序列只有 5,917 条。本目录将重复样本保存一次，原始两次划分归属仍保留在 `splits/imfp_lg_original.csv`；没有静默修复原作者基准。

MFP-MFL 的清洗版本也覆盖这 5,917 条唯一序列，但采用 4,732/1,185 的无相同序列交叉划分，见 `splits/mfp_mfl_cleaned.csv`。两种协议不能混着报告成绩。原始 MFP-MFL TSV 的列顺序为 AMP/ACP/ADP/AHP/AIP，本目录按标签名称对齐为 ACP/ADP/AHP/AIP/AMP；对齐后全部序列及标签与 MFBP5 一致。

### MFTP21 的原始数据与增强数据

原始划分保留 7,899/1,975 条，尚未按 MCMFPP 论文的非标准氨基酸过滤规则处理。`amcl_augmented_train.csv.gz` 单独保存作者的 """ + f"{len(augmented):,}" + """ 条增强训练记录，不进入原始样本池，不改变 Test，也不能算作新增真实样本。

### PDeepPP

Excel 数据内容按原文件复制，仅将文件名统一为 `train.xlsx`、`test.xlsx`。生物活性二分类与 PTM 位点窗口保持各自任务定义；没有把 PTM 窗口当作完整治疗肽，没有跨任务去重或合并标签。目录名保留作者命名。

已取得的任务：""" + "、".join(f"`{task}`" for task in tasks) + """。

## 未取得的基准

- SeekTP 的 8 标签、7,980 条筛选子集：作者百度网盘链接不存在，没有用完整 MFTP21 替代。
- SA-MTP 的 40% identity 低同源划分：未找到可明确识别的发布文件；`MTP15` 仅代表已取得的主基准。
- PDeepPP 未公开取得的任务数据：没有补造，模型权重未当作数据集。

作者仓库、论文链接、下载故障和 MCMFPP 数量差异见 [原始下载说明](../literature_benchmarks/README.md)。本次只执行数据整理与必要的样本/标签/划分核对，没有运行模型、训练、测试或性能评价。整理程序位于 `scripts/consolidate_literature_data.py`；已有输出目录不会覆盖。
"""
    (output / "README.md").write_text(description, encoding="utf-8")
    print("\n".join(stats))
    print(f"PDeepPP: {len(tasks)} independent tasks; AMCL augmentation: {len(augmented):,} records")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/literature_benchmarks"))
    parser.add_argument("--output", type=Path, default=Path("data/literature_consolidated"))
    args = parser.parse_args()
    consolidate(args.source, args.output)
