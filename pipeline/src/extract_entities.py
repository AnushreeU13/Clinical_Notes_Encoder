"""Extract structured clinical entities (diagnoses, medications, procedures,
symptoms) from clinical notes using LLaMA via the Groq API.

Input: data/parsed/patients.jsonl (from parse_fhir.py), one JSON object per
patient with a "clinical_notes" list.

Output: one JSON object per clinical note, with the parsed entities plus the
raw LLaMA response for debugging.
"""
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError

from utils import call_with_backoff, call_with_retry

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL = "llama-3.1-8b-instant"
PROMPT_VERSION = "v1"
REQUIRED_KEYS = ("diagnoses", "medications", "procedures", "symptoms")

SYSTEM_PROMPT = """You are a clinical NLP assistant. Extract clinical entities from the \
note below and return ONLY a JSON object with exactly these four keys, each a list of \
strings (use [] if none are mentioned): diagnoses, medications, procedures, symptoms."""

SIMPLIFIED_PROMPT = """Extract diagnoses, medications, procedures, and symptoms from this \
clinical note. Return only a JSON object with those four keys, each a list of strings."""


def is_valid_extraction(obj) -> bool:
    return isinstance(obj, dict) and all(
        key in obj and isinstance(obj[key], list) for key in REQUIRED_KEYS
    )


def empty_extraction() -> dict:
    return {key: [] for key in REQUIRED_KEYS}


def call_llama(client: Groq, system_prompt: str, note_text: str) -> str:
    def do_call() -> str:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": note_text},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return response.choices[0].message.content

    return call_with_backoff(do_call)


def extract_entities(client: Groq, note_text: str) -> dict:
    parsed, raw = call_with_retry(
        primary_call=lambda: call_llama(client, SYSTEM_PROMPT, note_text),
        fallback_call=lambda: call_llama(client, SIMPLIFIED_PROMPT, note_text),
        validate_fn=is_valid_extraction,
    )
    return {
        "entities": parsed if parsed is not None else empty_extraction(),
        "raw_response": raw,
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
        "valid": parsed is not None,
    }


def iter_notes(patients_path: Path, limit_notes: int | None, limit_patients: int | None):
    count = n_patients = 0
    with open(patients_path, encoding="utf-8") as f:
        for line in f:
            if limit_patients is not None and n_patients >= limit_patients:
                return
            patient = json.loads(line)
            patient_id = patient["demographics"]["patient_id"]
            for note in patient["clinical_notes"]:
                if limit_notes is not None and count >= limit_notes:
                    return
                yield patient_id, note
                count += 1
            n_patients += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/parsed/patients.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/parsed/entity_extractions.jsonl"))
    parser.add_argument("--limit-notes", type=int, default=None,
                         help="Process at most this many notes (for smoke-testing before a full run)")
    parser.add_argument("--limit-patients", type=int, default=None,
                         help="Process at most this many patients")
    parser.add_argument("--sleep-seconds", type=float, default=7.0,
                         help="Delay between API calls to stay under Groq's 6000 TPM free-tier limit")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set -- add it to pipeline/.env")
    client = Groq(api_key=api_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_processed = n_invalid = 0
    stopped_early = None
    with open(args.output, "w", encoding="utf-8") as out_f:
        for patient_id, note in iter_notes(args.input, args.limit_notes, args.limit_patients):
            try:
                result = extract_entities(client, note["text"])
            except RateLimitError as e:
                stopped_early = str(e)
                break
            if not result["valid"]:
                n_invalid += 1
            record = {
                "patient_id": patient_id,
                "note_date": note["date"],
                "note_type": note["type"],
                **result,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            n_processed += 1
            time.sleep(args.sleep_seconds)

    print(f"Processed {n_processed} notes ({n_invalid} fell back to empty extraction) -> {args.output}")
    if stopped_early:
        print(f"Stopped early due to a rate/quota limit: {stopped_early}")


if __name__ == "__main__":
    main()
