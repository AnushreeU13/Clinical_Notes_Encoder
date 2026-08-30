"""Sample a random gold set of coded diagnosis/medication entities, paired
with the source note text, for manual (or LLM-assisted first-pass) grading.

Output is a JSONL file with one row per sampled entity:
{patient_id, note_id, entity_type, entity_text, assigned_code,
 assigned_description, note_text, verdict, notes}

`verdict` and `notes` are left blank for the grader to fill in (correct /
incorrect / unsure, plus a short reason).
"""
import argparse
import json
import random
from pathlib import Path


def load_note_text_index(patients_path: Path) -> dict[tuple[str, str], str]:
    index = {}
    with open(patients_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            patient_id = p["demographics"]["patient_id"]
            for note in p["clinical_notes"]:
                index[(patient_id, note["note_id"])] = note["text"]
    return index


def load_codeable_rows(coded_path: Path) -> tuple[list[dict], list[dict]]:
    diagnoses, medications = [], []
    with open(coded_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["entity_type"] == "diagnosis":
                diagnoses.append(r)
            elif r["entity_type"] == "medication":
                medications.append(r)
    return diagnoses, medications


def sample_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


def build_gold_set(
    coded_path: Path, patients_path: Path, n_per_type: int, seed: int
) -> list[dict]:
    diagnoses, medications = load_codeable_rows(coded_path)
    note_text_index = load_note_text_index(patients_path)

    sampled = sample_rows(diagnoses, n_per_type, seed) + sample_rows(medications, n_per_type, seed)

    gold_set = []
    for r in sampled:
        note_text = note_text_index.get((r["patient_id"], r["note_id"]), "")
        if r["entity_type"] == "diagnosis":
            assigned_code = r.get("icd10_code")
            assigned_description = r.get("icd10_description")
        else:
            assigned_code = r.get("rxnorm_code")
            assigned_description = r.get("rxnorm_name")
        gold_set.append({
            "patient_id": r["patient_id"],
            "note_id": r["note_id"],
            "entity_type": r["entity_type"],
            "entity_text": r["entity_text"],
            "assigned_code": assigned_code,
            "assigned_description": assigned_description,
            "note_text": note_text,
            "verdict": None,
            "notes": None,
        })
    return gold_set


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coded", type=Path, default=Path("data/parsed/coded_records.jsonl"))
    parser.add_argument("--patients", type=Path, default=Path("data/parsed/patients.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/parsed/gold_set.jsonl"))
    parser.add_argument("--n-per-type", type=int, default=50,
                         help="Number of diagnoses AND number of medications to sample (each)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gold_set = build_gold_set(args.coded, args.patients, args.n_per_type, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in gold_set:
            f.write(json.dumps(row) + "\n")

    print(f"Sampled {len(gold_set)} entities -> {args.output}")


if __name__ == "__main__":
    main()
