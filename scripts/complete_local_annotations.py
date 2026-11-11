"""Recover supported training positives and list local benchmark leads for review."""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from openpyxl import load_workbook


TASKS = ("major15", "fine39", "rich48")
STANDARD = set("ACDEFGHIKLMNPQRSTVWY")
MODIFICATIONS = ("N-terminal modification", "C-terminal modification", "Post translation modifications")

# Only map benchmark endpoints whose names correspond to a project label. In
# fine39, a broad positive cannot determine that a subtype is unknown.
TARGETS = {
    "ABP": ("antibacterial", None, "Antibacterial"),
    "AFP": ("antifungal", None, "Antifungal"),
    "AVP": ("antiviral", None, "Antiviral"),
    "APP": ("antiparasitic", None, "Antiparasitic"),
    "AMP": ("antimicrobial", None, "Antimicrobial"),
    "ACP": ("anticancer", None, "Anticancer"),
    "AAP": ("antioxidant", None, "Antioxidant"),
    "AHTP": ("antihypertensive", None, "Antihypertensive"),
    "AHP": ("antihypertensive", None, "Antihypertensive"),
    "AIP": (None, "antiinflammatory", "Antiinflammatory"),
    "AHIVP": (None, "antihiv", "AntiHIV"),
    "CPP": (None, "cell_penetrating", "Cell_penetrating"),
    "BBP": (None, "blood_brain_barrier_penetrating", "Blood-brain_barrier_penetrating"),
    "QSP": (None, "quorum_sensing", "Quorum_sensing"),
    "DDV": ("drug_delivery", None, "Drug_delivery"),
    "neuro": ("neuropeptide", None, "Neuropeptide"),
    "antimalarial": (None, None, "Antimalarial"),
}
WORKBOOK_ENDPOINTS = {
    "Antibacterial": "ABP", "Antifungal": "AFP", "Antiviral": "AVP", "Antiparasitic": "APP",
    "Antimicrobial": "AMP", "Anticancer_main": "ACP", "Anticancer_alternative": "ACP",
    "Antioxidant": "AAP", "Antimalarial_main": "antimalarial", "Antimalarial_alternative": "antimalarial",
    "neuro": "neuro", "BBP": "BBP", "Quorum": "QSP",
}
GRAM_POSITIVE = set("S. aureus|S. epidermidis|B. subtilis|B. cereus|E. faecalis|E. faecium|L. monocytogenes|M. luteus|S. pyogenes|B. megaterium|S. mutans|S. pneumonia|S. agalactiae".split("|"))
GRAM_NEGATIVE = set("E. coli|P. aeruginosa|S. typhimurium|K. pneumoniae|A. baumannii|S. enterica|P. syringae|E. cloacae|P. pyocyaneus|P. mirabilis|S. sonnei".split("|"))
FUNGI = set("C. albicans|S. cerevisiae|C. neoformans|F. oxysporum|C. glabrata|C. tropicalis|C. parapsilosis|A. fumigatus|A. niger".split("|"))
MIC_LABELS = {
    "major15": {"gram_positive": "antibacterial", "gram_negative": "antibacterial", "fungus": "antifungal"},
    "fine39": {"gram_positive": "anti_gram+", "gram_negative": "anti_gram_"},
}
MAX_POSITIVE_MIC_UM = 16


def read_records(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {int(record["row_id"]): record for record in csv.DictReader(stream)}


def chemistry_key(sequence, record):
    helm = record["HELM notation"].strip()
    if not helm:
        return None
    modifications = [re.sub(r"\s+", "", record[field]).casefold() for field in MODIFICATIONS]
    return sequence, helm, *modifications


def benchmark_matches(root):
    matches = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def add(sequence, endpoint, value, source):
        if sequence and set(sequence) <= STANDARD:
            matches[sequence][endpoint][value].append(source)

    for collection in ("MTP15", "MFTP21", "MFBP5"):
        path = root / collection / "samples.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                for endpoint in TARGETS.keys() & row.keys():
                    if row[endpoint] == "1":
                        add(row["sequence"], endpoint, 1, f"{collection}/samples.csv:{row['sample_id']}")

    for folder, endpoint in WORKBOOK_ENDPOINTS.items():
        for split in ("train", "test"):
            path = root / "PDeepPP" / folder / f"{split}.xlsx"
            workbook = load_workbook(path, read_only=True, data_only=True)
            for sheet in workbook:
                rows = sheet.values
                if tuple(next(rows, ())) != ("sequence", "label"):
                    continue
                for number, (sequence, value) in enumerate(rows, 2):
                    if isinstance(sequence, str) and value in (0, 1):
                        add(sequence, endpoint, int(value), f"PDeepPP/{folder}/{split}.xlsx:{sheet.title}!{number}")
            workbook.close()
    return matches


def grampa_assays(path):
    assays = defaultdict(list)
    groups = {target: group for targets, group in ((GRAM_POSITIVE, "gram_positive"),
              (GRAM_NEGATIVE, "gram_negative"), (FUNGI, "fungus")) for target in targets}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for record in csv.DictReader(stream):
            group = groups.get(record["bacterium"])
            if not group or record["unit"] != "uM" or not record["value"]:
                continue
            log_mic = float(record["value"])
            if math.isfinite(log_mic):
                assays[record["sequence"]].append((record, group, 10 ** log_mic))
    return assays


def smep_assays(path):
    from pypdf import PdfReader

    assays = defaultdict(list)
    expression = r"^\s*([ACDEFGHIKLMNPQRSTVWY]+)-NH2\s+\d+\s+(>?\d+)\s*$"
    pdf = PdfReader(path)
    for index in (18, 19):
        table = pdf.pages[index].extract_text(extraction_mode="layout")
        for sequence, mic in re.findall(expression, table, re.M):
            assays[sequence].append((index + 1, mic))
    if sum(map(len, assays.values())) != 67:
        raise ValueError("SMEP Supplementary Table 2 must contain its published 67 measurements")
    return assays


def same_grampa_chemistry(record, assay):
    sequence = record["Sequence"]
    expected = "PEPTIDE1{[ac]." + ".".join(sequence) + ".[am]}$$$$"
    return (record["terminal_annotation_conflict"] == "False"
            and record["chemical_annotation_conflict"] == "False"
            and not record["Post translation modifications"]
            and record["N-terminal modification"] == "Acetylation"
            and record["C-terminal modification"] == "Amidation"
            and record["HELM notation"] == expected
            and assay["modifications"] == "[N-Terminus: ACT, C-Terminus: AMD]"
            and assay["datasource_has_modifications"] == "True")


def assemble_task(root, track, task_index, records, main_records, matches, output, confirmed, review,
                  grampa, smep, assays):
    with np.load(root / "arrays/sequences.npz", allow_pickle=False) as archive:
        row_ids, ids, sequences, available = (archive[key] for key in
                                                ("row_id", "ID", "canonical_backbone", "canonical_available"))
    with np.load(root / "tracks" / track / "dataset.npz", allow_pickle=False) as archive:
        if not np.array_equal(row_ids, archive["row_id"]):
            raise ValueError(f"Sequence and label rows differ for {track}")
        labels, targets, groups = (archive[key] for key in ("label_names", "Y", "cluster_id"))
        indices = np.flatnonzero(available & archive["train_original_mask"])
    columns = {label: c for c, label in enumerate(labels)}
    chemistry_groups = defaultdict(list)
    updated = {}
    counts = Counter()

    def main_record(record):
        if track != "rich48":
            return record
        row = record["source_row_id"]
        return main_records[int(row)] if row else None

    def entry(i, label, status, source, record):
        counts[status] += 1
        return [track, int(row_ids[i]), str(ids[i]), str(groups[i]), label, status, source,
                record.get("row_id", "") if record else "", record.get("Reference", "") if record else ""]

    for i in indices:
        record = records[int(row_ids[i])]
        primary = main_record(record)
        # rich48 donor links identify a backbone match to the main table. Use
        # those links to locate evidence, without transferring donor labels.
        if track != "rich48":
            key = chemistry_key(sequences[i], record)
            if key is not None:
                chemistry_groups[key].append(i)
        sequence = record["Sequence"]
        grampa_candidates = grampa.get(sequence, ())
        if grampa_candidates:
            counts["grampa_sequence_overlap"] += 1
        matching_assays = [(assay, group, mic) for assay, group, mic in grampa_candidates
                           if track != "rich48" and same_grampa_chemistry(record, assay)]
        if matching_assays:
            counts["grampa_chemistry_matched_rows"] += 1
        positives = {}
        for assay, group, mic in matching_assays:
            source = f"GRAMPA/grampa.csv:index={assay['']}; {assay['url_source']}"
            assays.writerow([track, int(row_ids[i]), str(ids[i]), sequence, "matched_sequence_and_caps",
                             assay["bacterium"], assay["strain"], format(mic, ".8g"), "uM", source])
            counts["grampa_chemistry_matched_assays"] += 1
            label = MIC_LABELS[track].get(group)
            if label and mic <= MAX_POSITIVE_MIC_UM and not targets[i, columns[label]]:
                positives.setdefault(label, []).append(source)
        for label, sources in positives.items():
            new_targets = updated.setdefault(i, targets[i].copy())
            new_targets[columns[label]] = 1
            confirmed.writerow(entry(i, label, "recovered_positive_from_mic", " | ".join(sources), primary))
        for page, mic in smep.get(sequence, ()):
            assays.writerow([track, int(row_ids[i]), str(ids[i]), sequence, "sequence_match_only",
                             "S. aureus", "", mic, "ug/mL", f"SMEP/Supplementary Information.pdf:page={page}, Table 2"])
            counts["smep_sequence_matched_assays"] += 1
            counts["smep_above_maximum_assays"] += mic.startswith(">")
        for endpoint, values in matches.get(sequence, {}).items():
            label = TARGETS[endpoint][task_index]
            if label not in columns:
                continue
            present = bool(targets[i, columns[label]])
            if not present and values.get(1):
                review.writerow(entry(i, label, "benchmark_positive_to_verify", " | ".join(values[1]), primary))
            if values.get(0):
                status = "benchmark_zero_vs_project_positive" if present else "benchmark_zero_to_verify"
                review.writerow(entry(i, label, status, " | ".join(values[0]), primary))

    for members in chemistry_groups.values():
        if len(members) < 2:
            continue
        union = targets[members].any(0)
        for i in members:
            dest = records[int(row_ids[i])]
            for c in np.flatnonzero(union & ~targets[i].astype(bool)):
                source = next(j for j in members if targets[j, c])
                donor = records[int(row_ids[source])]
                label = str(labels[c])
                source_link = f"therapeutic_2025/records.csv:row_id={int(row_ids[source])}"
                conflict = any(record[key] == "True" for record in (dest, donor)
                               for key in ("terminal_annotation_conflict", "chemical_annotation_conflict"))
                if conflict or groups[i] != groups[source] or not donor["Reference"].strip():
                    review.writerow(entry(i, label, "matched_record_to_verify", source_link, donor))
                    continue
                new_targets = updated.setdefault(i, targets[i].copy())
                if new_targets[c]:
                    continue
                new_targets[c] = 1
                confirmed.writerow(entry(i, label, "recovered_positive", source_link, donor))

    if updated:
        changed = np.array(sorted(updated), dtype=int)
        y = np.stack([updated[i] for i in changed])
        np.savez_compressed(output / f"{track}.npz", row_id=row_ids[changed], ID=ids[changed],
                            label_names=labels, Y=y, observed=y.astype(bool))
    return {"train_rows": len(indices), "updated_rows": len(updated), **dict(counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--benchmarks", type=Path, default=Path("data/literature_consolidated"))
    parser.add_argument("--grampa", type=Path, help="Local GRAMPA grampa.csv with log10 activity concentrations")
    parser.add_argument("--smep-supplement", type=Path, help="Local SMEP Supplementary Information PDF")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matches = benchmark_matches(args.benchmarks)
    grampa = grampa_assays(args.grampa) if args.grampa else {}
    smep = smep_assays(args.smep_supplement) if args.smep_supplement else {}
    main_records = read_records(args.data_root / "records.csv")
    args.output.mkdir(parents=True, exist_ok=False)
    header = ("task", "row_id", "ID", "cluster_id", "label", "status", "evidence_source",
              "main_row_id", "main_reference")
    with (args.output / "recovered_positives.tsv").open("w", encoding="utf-8-sig", newline="") as accepted, \
         (args.output / "review_candidates.tsv").open("w", encoding="utf-8-sig", newline="") as candidates, \
         (args.output / "assay_evidence.tsv").open("w", encoding="utf-8-sig", newline="") as measurements:
        confirmed, review, assays = (csv.writer(stream, delimiter="\t")
                                     for stream in (accepted, candidates, measurements))
        for writer in (confirmed, review):
            writer.writerow(header)
        assays.writerow(("task", "row_id", "ID", "sequence", "match", "target", "strain", "value", "unit", "source"))
        for task_index, track in enumerate(TASKS):
            root = args.data_root / "rich48" if track == "rich48" else args.data_root
            records = read_records(root / "records.csv") if track == "rich48" else main_records
            counts = assemble_task(root, track, task_index, records, main_records, matches, args.output,
                                   confirmed, review, grampa, smep, assays)
            print(track, json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
