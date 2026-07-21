"""Parse Synthea FHIR R4 patient bundles into structured patient records.

Note on ground truth: Synthea's standard FHIR export codes Condition/Procedure
using SNOMED-CT (http://snomed.info/sct), not ICD-10. ICD-10 only appears in
Synthea's separate BlueButton2/RIF claims exporter, which this project does not
use. So `diagnoses[].code`/`system` below reflect the real source coding
(SNOMED-CT) rather than ICD-10 ground truth -- ICD-10 mapping happens later in
map_codes.py via the icd10-cm package + LLaMA validation.
"""
import argparse
import base64
import json
from pathlib import Path


def load_bundle(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_by_type(bundle: dict) -> dict[str, list[dict]]:
    by_type: dict[str, list[dict]] = {}
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")
        if resource_type:
            by_type.setdefault(resource_type, []).append(resource)
    return by_type


def index_by_full_url(bundle: dict) -> dict[str, dict]:
    return {
        entry["fullUrl"]: entry["resource"]
        for entry in bundle.get("entry", [])
        if "fullUrl" in entry
    }


def first_coding(codeable_concept: dict | None) -> dict:
    if not codeable_concept:
        return {"code": None, "system": None, "text": None}
    codings = codeable_concept.get("coding", [])
    coding = codings[0] if codings else {}
    return {
        "code": coding.get("code"),
        "system": coding.get("system"),
        "text": codeable_concept.get("text") or coding.get("display"),
    }


def parse_demographics(patient: dict) -> dict:
    name = patient.get("name", [{}])[0]
    given = " ".join(name.get("given", []))
    family = name.get("family", "")
    address = patient.get("address", [{}])[0]
    race = ethnicity = None
    for ext in patient.get("extension", []):
        if ext["url"].endswith("us-core-race"):
            race = next(
                (e["valueCoding"]["display"] for e in ext.get("extension", [])
                 if e["url"] == "ombCategory"), None,
            )
        elif ext["url"].endswith("us-core-ethnicity"):
            ethnicity = next(
                (e["valueCoding"]["display"] for e in ext.get("extension", [])
                 if e["url"] == "ombCategory"), None,
            )
    return {
        "patient_id": patient.get("id"),
        "name": f"{given} {family}".strip(),
        "gender": patient.get("gender"),
        "birth_date": patient.get("birthDate"),
        "race": race,
        "ethnicity": ethnicity,
        "city": address.get("city"),
        "state": address.get("state"),
    }


def parse_diagnoses(conditions: list[dict]) -> list[dict]:
    diagnoses = []
    for c in conditions:
        coding = first_coding(c.get("code"))
        diagnoses.append({
            **coding,
            "clinical_status": (c.get("clinicalStatus", {}).get("coding", [{}])[0].get("code")),
            "onset_date": c.get("onsetDateTime"),
        })
    return diagnoses


def parse_medications(medication_requests: list[dict], resources_by_url: dict[str, dict]) -> list[dict]:
    medications = []
    for m in medication_requests:
        if "medicationCodeableConcept" in m:
            coding = first_coding(m["medicationCodeableConcept"])
        else:
            ref = m.get("medicationReference", {}).get("reference")
            medication_resource = resources_by_url.get(ref, {})
            coding = first_coding(medication_resource.get("code"))
        medications.append({
            **coding,
            "status": m.get("status"),
            "authored_on": m.get("authoredOn"),
        })
    return medications


def parse_procedures(procedures: list[dict]) -> list[dict]:
    result = []
    for p in procedures:
        coding = first_coding(p.get("code"))
        performed = p.get("performedDateTime") or p.get("performedPeriod", {}).get("start")
        result.append({
            **coding,
            "status": p.get("status"),
            "performed_date": performed,
        })
    return result


def parse_clinical_notes(document_references: list[dict]) -> list[dict]:
    notes = []
    for doc in document_references:
        doc_type = first_coding(doc.get("type"))
        for content in doc.get("content", []):
            attachment = content.get("attachment", {})
            data = attachment.get("data")
            content_type = attachment.get("contentType", "")
            if not data or "text/plain" not in content_type:
                continue
            text = base64.b64decode(data).decode("utf-8")
            notes.append({
                "note_id": doc.get("id"),
                "date": doc.get("date"),
                "type": doc_type["text"],
                "text": text,
            })
    return notes


def parse_patient_bundle(bundle: dict) -> dict | None:
    by_type = index_by_type(bundle)
    patients = by_type.get("Patient", [])
    if not patients:
        return None
    resources_by_url = index_by_full_url(bundle)
    return {
        "demographics": parse_demographics(patients[0]),
        "diagnoses": parse_diagnoses(by_type.get("Condition", [])),
        "medications": parse_medications(by_type.get("MedicationRequest", []), resources_by_url),
        "procedures": parse_procedures(by_type.get("Procedure", [])),
        "clinical_notes": parse_clinical_notes(by_type.get("DocumentReference", [])),
    }


def parse_all(input_dir: Path) -> list[dict]:
    records = []
    for path in sorted(input_dir.glob("*.json")):
        bundle = load_bundle(path)
        record = parse_patient_bundle(bundle)
        if record is not None:
            records.append(record)
    return records


def save_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/synthea_output"))
    parser.add_argument("--output", type=Path, default=Path("data/parsed/patients.jsonl"))
    args = parser.parse_args()

    records = parse_all(args.input_dir)
    save_jsonl(records, args.output)
    print(f"Parsed {len(records)} patients -> {args.output}")


if __name__ == "__main__":
    main()
