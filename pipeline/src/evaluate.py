"""Evaluate the coding pipeline's output.

Note on ground truth vs. CLAUDE.md's original design: CLAUDE.md calls for
exact-match and category-match comparison against "Synthea's ground truth
ICD-10 codes." As established in parse_fhir.py, Synthea's standard FHIR
export has no ICD-10 anywhere -- Condition resources are SNOMED-CT only. So
there is no ICD-10 ground truth to exact/category-match against.

Instead, this evaluates:
  - match_method breakdown for diagnoses (direct / llm_validated / no_match)
    and ICD-10 coverage (% of diagnoses that got any code at all)
  - RxNorm coverage for medications (% that got a non-null rxnorm_code)
  - a "grounded rate": for each diagnosis entity extracted from a clinical
    note, whether it shares meaningful vocabulary with one of that same
    patient's actual Condition entries (from parse_fhir.py's structured
    output) -- a proxy for "did the NER stage find something real" in the
    absence of a true labeled dataset.
  - counts for procedures/symptoms (no code-mapping stage exists for these,
    so no accuracy metric applies, only volume)

Reported per entity type, per CLAUDE.md's requirement.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_ground_truth_diagnoses(patients: list[dict]) -> dict[str, list[set[str]]]:
    """patient_id -> list of token sets, one per recorded Condition."""
    ground_truth = {}
    for patient in patients:
        patient_id = patient["demographics"]["patient_id"]
        ground_truth[patient_id] = [
            _tokenize(d["text"]) for d in patient["diagnoses"] if d.get("text")
        ]
    return ground_truth


def is_grounded(entity_tokens: set[str], condition_token_sets: list[set[str]]) -> bool:
    return any(entity_tokens & tokens for tokens in condition_token_sets)


def evaluate_diagnoses(coded: list[dict], ground_truth: dict[str, list[set[str]]]) -> dict:
    diagnoses = [r for r in coded if r["entity_type"] == "diagnosis"]
    if not diagnoses:
        return {"total": 0}

    method_counts = Counter(r["match_method"] for r in diagnoses)
    n_coded = sum(1 for r in diagnoses if r.get("icd10_code"))

    n_grounded = 0
    for r in diagnoses:
        conditions = ground_truth.get(r["patient_id"], [])
        if is_grounded(_tokenize(r["entity_text"]), conditions):
            n_grounded += 1

    total = len(diagnoses)
    return {
        "total": total,
        "match_method_counts": dict(method_counts),
        "match_method_pct": {k: round(100 * v / total, 1) for k, v in method_counts.items()},
        "icd10_coverage_pct": round(100 * n_coded / total, 1),
        "grounded_in_chart_pct": round(100 * n_grounded / total, 1),
    }


def evaluate_medications(coded: list[dict]) -> dict:
    medications = [r for r in coded if r["entity_type"] == "medication"]
    if not medications:
        return {"total": 0}

    n_with_code = sum(1 for r in medications if r.get("rxnorm_code"))
    total = len(medications)
    return {
        "total": total,
        "rxnorm_coverage_pct": round(100 * n_with_code / total, 1),
        "validated_pct": 0.0,  # no `rxnorm` package exists on PyPI to validate against
    }


def evaluate_passthrough(coded: list[dict], entity_type: str) -> dict:
    return {"total": sum(1 for r in coded if r["entity_type"] == entity_type)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=Path, default=Path("data/parsed/patients.jsonl"))
    parser.add_argument("--coded", type=Path, default=Path("data/parsed/coded_records.jsonl"))
    args = parser.parse_args()

    patients = load_jsonl(args.patients)
    coded = load_jsonl(args.coded)
    ground_truth = build_ground_truth_diagnoses(patients)

    report = {
        "diagnosis": evaluate_diagnoses(coded, ground_truth),
        "medication": evaluate_medications(coded),
        "procedure": evaluate_passthrough(coded, "procedure"),
        "symptom": evaluate_passthrough(coded, "symptom"),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
