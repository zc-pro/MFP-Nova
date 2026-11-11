"""Summarize a fixed Gram+/Gram- MIC panel on fine39's existing Train rows."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from mfp_nova.hypothesis_data import SequenceStudyData


PANEL = ("Staphylococcus aureus ATCC 25923", "Escherichia coli ATCC 25922")
PANEL_COLUMNS = ("anti_gram+", "anti_gram_")
THRESHOLDS = {"ug/mL": (32, 128), "uM": (10, 40)}
STATES = ("00", "10", "01", "11")
RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")
MIN_FAMILIES = 5


def mic_class(concentration, unit):
    """Keep a censored value on the side of the cutoff implied by its bound."""
    unit = unit.strip().lower().replace("µ", "u").replace("μ", "u").replace(" ", "")
    unit = {"ug/ml": "ug/mL", "mg/l": "ug/mL", "um": "uM"}.get(unit)
    raw = concentration.strip().replace("≤", "<=").replace("≥", ">=")
    match = re.fullmatch(r"(<=|>=|<|>|=)?\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", raw)
    if not unit or not match:
        return unit, None
    relation, number = match.groups()
    value = float(number)
    positive, inactive = THRESHOLDS[unit]
    if 0 < value <= positive and relation not in (">", ">="):
        return unit, 1
    if value >= inactive and relation not in ("<", "<="):
        return unit, 0
    return unit, None


def exact_chemistry(row, card):
    if (row["terminal_annotation_conflict"] == "True" or row["chemical_annotation_conflict"] == "True"
            or row["Post translation modifications"] or not row["Sequence"]
            or set(row["Sequence"].upper()) - RESIDUES or card["sequence"] != row["Sequence"]):
        return False
    if any(card[key] for key in ("unusualAminoAcids", "intrachainBonds", "interchainBonds", "coordinationBonds")):
        return False
    n_term = (card.get("nTerminus") or {}).get("name")
    c_term = (card.get("cTerminus") or {}).get("name")
    if n_term not in (None, "ACT") or c_term not in (None, "AMD"):
        return False
    n_term = ["[ac]"] if n_term else []
    c_term = ["[am]"] if c_term else []
    # DBAASP spells D residues with lowercase letters. Their positions must
    # match the explicit D tokens in the project's HELM sequence.
    residues = [f"[d{aa.upper()}]" if aa.islower() else aa for aa in row["Sequence"]]
    helm = "PEPTIDE1{" + ".".join([*n_term, *residues, *c_term]) + "}$$$$"
    return row["HELM notation"].strip() == helm


def activity_rows(card):
    articles = card["articles"] or []
    for assay in card["targetActivities"]:
        target = (assay.get("targetSpecies") or {}).get("name")
        measure = (assay.get("activityMeasureGroup") or {}).get("name")
        if target not in PANEL or measure != "MIC" or assay.get("activityMeasureValue") != "MIC":
            continue
        raw = assay.get("concentration") or ""
        original_unit = (assay.get("unit") or {}).get("name") or ""
        unit, state = mic_class(raw, original_unit)
        ref = str(assay.get("reference") or "").strip()
        article = articles[int(ref) - 1] if ref.isdigit() and 1 <= int(ref) <= len(articles) else {}
        pmid = (article.get("pubmed") or {}).get("pubmedId")
        article_id = article.get("id") if article.get("title") else None
        condition = ((assay.get("medium") or {}).get("name") or "", *(assay.get(k) or "" for k in
                    ("cfu", "ph", "ionicStrength", "saltType", "note")))
        yield target, raw, original_unit, unit, state, ref, pmid, article_id, condition, assay["id"]


def card_panel_states(assays):
    measurements = defaultdict(lambda: defaultdict(set))
    for target, _, _, unit, state, _, pmid, article_id, condition, _ in assays:
        if unit and (pmid or article_id):
            measurements[pmid, article_id, unit, condition][target].add(state)
    for (pmid, article_id, unit, condition), measures in measurements.items():
        both = [measures.get(target, set()) for target in PANEL]
        if all(len(values) == 1 and None not in values for values in both):
            yield "".join(str(next(iter(values))) for values in both), pmid, article_id, unit, condition


def write_rows(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(fieldnames)
        writer.writerows(rows)


def export_annotations(data, accepted, output):
    positions = {int(row_id): index for index, row_id in enumerate(data.rows)}
    rows = [positions[row_id] for row_id, *_ in accepted]
    columns = [data.labels.index(label) for label in PANEL_COLUMNS]
    targets, observed = data.targets.copy(), data.observed.copy()
    targets[:, columns] = 0
    observed[:, columns] = False
    panel_targets = np.array([[int(bit) for bit in row[-1]] for row in accepted]).reshape(-1, len(columns))
    targets[np.ix_(rows, columns)] = panel_targets
    observed[np.ix_(rows, columns)] = True
    np.savez_compressed(output, row_id=data.rows, ID=data.ids.astype(str), label_names=data.labels,
                        Y=targets, observed=observed)


def summarize(root, output, export):
    data = SequenceStudyData(root, "fine39")
    train = data.indices("train")
    folds = data.train_folds(train, 5)
    by_row = {int(data.rows[i]): (str(data.ids[i]), int(data.clusters[i]), int(fold)) for i, fold in zip(train, folds)}
    # The raw-sequence backbone ignores stereochemistry and terminal modifications.
    sequence_input = {int(data.rows[i]): data.sequences[i].strip().upper() for i in train}
    by_sequence = defaultdict(list)
    with (root / "records.csv").open(encoding="utf-8-sig", newline="") as stream:
        for record in csv.DictReader(stream):
            if int(record["row_id"]) in by_row:
                by_sequence[record["Sequence"]].append(record)
    resolved = {}
    source_matches = output / "primary_source_matches.tsv"
    if source_matches.is_file():
        with source_matches.open(encoding="utf-8-sig", newline="") as stream:
            resolved = {(int(row["row_id"]), int(row["dbaasp_id"]), row["PMID"]): row["primary_source"]
                        for row in csv.DictReader(stream, delimiter="\t")}

    raw, findings = [], defaultdict(list)
    cards, paired_cards, exact_cards = 0, 0, 0
    with (output / "dbaasp_records.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            card = json.loads(line)
            cards += 1
            assays = list(activity_rows(card))
            paired = list(card_panel_states(assays))
            paired_cards += bool(paired)
            exact = [(record, exact_chemistry(record, card)) for record in by_sequence[card["sequence"]]]
            exact_cards += any(match for _, match in exact)
            for record, match in exact:
                row_id = int(record["row_id"])
                local_pmids = set(re.findall(r"PMID\s*:\s*(\d+)", record["Reference"], re.I))
                for target, mic, original_unit, unit, state, ref, pmid, article_id, condition, assay_id in assays:
                    reference = resolved.get((row_id, card["id"], pmid)) if pmid in local_pmids else None
                    raw.append((row_id, card["id"], match, reference, assay_id, target, mic, original_unit,
                                unit, state, ref, pmid, article_id, *condition))
                for state, pmid, article_id, unit, condition in paired:
                    reference = resolved.get((row_id, card["id"], pmid)) if pmid in local_pmids else None
                    findings[row_id].append((match, reference, state, card["id"], pmid, article_id,
                                             pmid in local_pmids, unit, condition))

    accepted = []
    evidence = []
    for row_id, observations in sorted(findings.items()):
        verified = {state for match, reference, state, *_ in observations if match or reference}
        status = "conflicting_matched_panels" if len(verified) > 1 else "sequence_match_only"
        if len(verified) == 1:
            state = next(iter(verified))
            accepted.append((row_id, *by_row[row_id], state))
            status = "matched_panel"
        for match, reference, state, card, pmid, article_id, shared_source, unit, condition in observations:
            evidence.append((row_id, *by_row[row_id], status if match or reference else "sequence_match_only",
                             state, card, pmid, article_id, shared_source, reference, unit, *condition))

    by_input = defaultdict(list)
    for row in accepted:
        by_input[sequence_input[row[0]]].append(row)
    conflicts = {row[0] for group in by_input.values() if len({row[-1] for row in group}) > 1 for row in group}
    accepted = [row for row in accepted if row[0] not in conflicts]
    evidence = [(*row[:4], "conflicting_sequence_input", *row[5:])
                if row[0] in conflicts and row[4] == "matched_panel" else row for row in evidence]

    support = []
    for state in STATES:
        rows = [(family, fold) for _, _, family, fold, actual in accepted if actual == state]
        families = {family for family, _ in rows}
        complements = [len({family for family, assigned in rows if assigned != fold}) for fold in range(5)]
        support.append((state, len(rows), len(families), *complements, max(0, MIN_FAMILIES - len(families)),
                        max(0, 10 - len(families))))

    condition_cols = ("medium", "cfu", "ph", "ionic_strength", "salt_type", "note")
    write_rows(output / "mic_measurements.tsv", ("row_id", "dbaasp_id", "exact_chemistry", "primary_source", "assay_id", "target",
               "mic_as_reported", "unit_as_reported", "threshold_unit", "panel_label", "reference_index", "PMID",
               "dbaasp_article_id", *condition_cols), raw)
    write_rows(output / "paired_measurements.tsv", ("row_id", "ID", "cluster_id", "fold", "match", "state",
               "dbaasp_id", "PMID", "dbaasp_article_id", "shares_project_PMID", "primary_source", "unit", *condition_cols), evidence)
    write_rows(output / "panel_labels.tsv", ("row_id", "ID", "cluster_id", "fold", "state", *PANEL),
               ((*row, int(row[-1][0]), int(row[-1][1])) for row in accepted))
    write_rows(output / "four_state_support.tsv", ("state", "train_rows", "homology_groups",
               *(f"groups_outside_fold_{fold}" for fold in range(5)), "groups_short_of_5", "groups_short_of_10"), support)
    missing_folds = {row[0]: {fold for fold, count in enumerate(row[3:8]) if count < MIN_FAMILIES}
                     for row in support}
    existing = {(state, family) for _, _, family, _, state in accepted}
    leads = {}
    for row_id, _, family, fold, match, state, card, pmid, _, shared, _, *_ in evidence:
        key = state, family
        helps_folds = sorted(missing_folds[state] - {fold})
        if match == "sequence_match_only" and helps_folds and key not in existing:
            if key not in leads or shared and not leads[key][-1]:
                leads[key] = (state, family, fold, row_id, card, pmid, ",".join(map(str, helps_folds)), shared)
    ordered = sorted(leads.values(), key=lambda row: (not row[-1], -len(row[-2]), STATES.index(row[0]), row[1]))
    write_rows(output / "review_candidates.tsv", ("state", "cluster_id", "fold", "row_id", "dbaasp_id", "PMID",
               "helps_folds", "shares_project_PMID"), ordered)
    (output / "panel_definition.json").write_text(json.dumps({"track": "fine39", "split": "train",
        "targets_in_state_order": PANEL, "active_at_or_below_and_inactive_at_or_above": THRESHOLDS,
        "annotation_columns_in_state_order": PANEL_COLUMNS,
        "annotation_scope": "Exploratory training only. The two columns are strain endpoints; unmeasured bits are masked.",
        "unlabeled_weight": 0,
        "database": "https://dbaasp.org/v3/api-docs", "cards_checked": cards,
        "matching": "Exact sequence and simple linear HELM chemistry, or form resolved using the referenced study",
        "pairing": "Same card and article, matching reported conditions and unit",
        "conflicting_raw_sequence_rows_excluded": len(conflicts),
    }, indent=2), encoding="utf-8")
    if export:
        export_annotations(data, accepted, output / "fine39_two_strain_train_annotations.npz")
    print(f"Cards={cards}, exact chemistry cards={exact_cards}, cards with comparable definitive pairs={paired_cards}")
    print(f"Excluded {len(conflicts)} matched rows with conflicting states for the same model input")
    for state, rows, families, *rest in support:
        print(f"{state}: rows={rows}, groups={families}, fold complements={rest[:5]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--output", type=Path, required=True, help="Directory with the saved DBAASP cards")
    parser.add_argument("--export-annotations", action="store_true", help="Write an isolated Train-only MIC overlay")
    args = parser.parse_args()
    summarize(args.data_root, args.output, args.export_annotations)


if __name__ == "__main__":
    main()
