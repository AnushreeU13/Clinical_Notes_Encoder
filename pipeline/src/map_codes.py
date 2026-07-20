"""Map extracted clinical entities to ICD-10 (diagnoses) and RxNorm (medications).

Note on tooling reality vs. CLAUDE.md's assumptions:
- The installed `icd10-cm` package has no text-search API -- only `find(code)`
  (exact lookup) and a raw `codes` dict of code -> [billable, description].
  `search_icd10()` below implements the text search over that dict ourselves,
  in place of the `icd10.search(...)` call CLAUDE.md assumed exists.
- No PyPI package named `rxnorm` exists at all, so medication mapping always
  goes through the LLaMA-only fallback path, flagged unvalidated -- this is
  the documented fallback, not a workaround.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import icd10
from dotenv import load_dotenv
from groq import Groq, RateLimitError

from utils import call_with_backoff, call_with_retry

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL = "llama-3.1-8b-instant"
DIRECT_MATCH_THRESHOLD = 0.999

_ICD10_INDEX = None


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _icd10_index() -> list[tuple[str, set[str], str, bool]]:
    global _ICD10_INDEX
    if _ICD10_INDEX is None:
        _ICD10_INDEX = [
            (code, _tokenize(description), description, billable)
            for code, (billable, description) in icd10.codes.items()
        ]
    return _ICD10_INDEX


def search_icd10(query: str, top_n: int = 3) -> list[dict]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    scored = []
    for code, desc_tokens, description, billable in _icd10_index():
        overlap = query_tokens & desc_tokens
        if not overlap:
            continue
        score = len(overlap) / len(query_tokens)
        scored.append((score, code, description, billable))
    scored.sort(key=lambda item: (-item[0], item[1]))
    results = []
    for score, code, description, billable in scored[:top_n]:
        formatted = icd10.find(code)
        results.append({
            "code": formatted.code if formatted else code,
            "description": description,
            "billable": billable,
            "score": round(score, 3),
        })
    return results


def _call_json(client: Groq, prompt: str) -> str:
    def do_call() -> str:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return response.choices[0].message.content

    return call_with_backoff(do_call)


def disambiguate_icd10(client: Groq, entity_text: str, candidates: list[dict]) -> str | None:
    options = "; ".join(f"{c['code']}: {c['description']}" for c in candidates)
    prompt = (
        f"Given the clinical mention '{entity_text}', which of these ICD-10 codes is "
        f"most appropriate? Options: {options}. "
        'Return only JSON: {"code": "<one of the option codes>"}'
    )
    valid_codes = {c["code"] for c in candidates}

    def is_valid(obj):
        return isinstance(obj, dict) and obj.get("code") in valid_codes

    parsed, _ = call_with_retry(
        primary_call=lambda: _call_json(client, prompt),
        fallback_call=lambda: _call_json(client, prompt),
        validate_fn=is_valid,
    )
    return parsed["code"] if parsed else None


def map_diagnosis(client: Groq, entity_text: str) -> dict:
    candidates = search_icd10(entity_text)
    if not candidates:
        return {
            "entity_text": entity_text, "icd10_code": None,
            "icd10_description": None, "match_method": "no_match",
        }

    unique_top_match = len(candidates) == 1 or candidates[0]["score"] > candidates[1]["score"]
    if unique_top_match and candidates[0]["score"] >= DIRECT_MATCH_THRESHOLD:
        best = candidates[0]
        return {
            "entity_text": entity_text, "icd10_code": best["code"],
            "icd10_description": best["description"], "match_method": "direct",
        }

    chosen_code = disambiguate_icd10(client, entity_text, candidates)
    chosen = next((c for c in candidates if c["code"] == chosen_code), candidates[0])
    method = "llm_validated" if chosen_code else "llm_fallback_top_candidate"
    return {
        "entity_text": entity_text, "icd10_code": chosen["code"],
        "icd10_description": chosen["description"], "match_method": method,
    }


RXNORM_PROMPT = (
    "Normalize this medication mention to its generic drug name and suggest a plausible "
    'RxNorm concept. Return only JSON: {{"generic_name": "...", "rxnorm_code": "..." or null, '
    '"rxnorm_name": "..." or null}}. Medication mention: "{text}"'
)


def is_valid_rxnorm(obj) -> bool:
    return isinstance(obj, dict) and {"generic_name", "rxnorm_code", "rxnorm_name"} <= obj.keys()


def map_medication(client: Groq, entity_text: str) -> dict:
    prompt = RXNORM_PROMPT.format(text=entity_text)
    parsed, _ = call_with_retry(
        primary_call=lambda: _call_json(client, prompt),
        fallback_call=lambda: _call_json(client, prompt),
        validate_fn=is_valid_rxnorm,
    )
    if parsed is None:
        return {
            "entity_text": entity_text, "generic_name": None,
            "rxnorm_code": None, "rxnorm_name": None, "validated": False,
        }
    return {
        "entity_text": entity_text,
        "generic_name": parsed["generic_name"],
        "rxnorm_code": parsed["rxnorm_code"],
        "rxnorm_name": parsed["rxnorm_name"],
        "validated": False,  # no `rxnorm` package exists on PyPI to validate against
    }


def iter_entity_records(path: Path, limit: int | None):
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if limit is not None and count >= limit:
                return
            yield json.loads(line)
            count += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/parsed/entity_extractions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/parsed/coded_records.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many notes")
    parser.add_argument("--sleep-seconds", type=float, default=7.0,
                         help="Delay between API calls to stay under Groq's 6000 TPM free-tier limit")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set -- add it to pipeline/.env")
    client = Groq(api_key=api_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"diagnosis": 0, "medication": 0, "procedure": 0, "symptom": 0}
    stopped_early = None

    with open(args.output, "w", encoding="utf-8") as out_f:

        def write(row: dict) -> None:
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()

        for record in iter_entity_records(args.input, args.limit):
            base = {
                "patient_id": record["patient_id"],
                "note_date": record["note_date"],
                "note_type": record["note_type"],
            }
            try:
                for text in record["entities"]["diagnoses"]:
                    write({**base, "entity_type": "diagnosis", **map_diagnosis(client, text)})
                    counts["diagnosis"] += 1
                    time.sleep(args.sleep_seconds)
                for text in record["entities"]["medications"]:
                    write({**base, "entity_type": "medication", **map_medication(client, text)})
                    counts["medication"] += 1
                    time.sleep(args.sleep_seconds)
            except RateLimitError as e:
                stopped_early = str(e)
                break
            for text in record["entities"]["procedures"]:
                write({**base, "entity_type": "procedure", "entity_text": text})
                counts["procedure"] += 1
            for text in record["entities"]["symptoms"]:
                write({**base, "entity_type": "symptom", "entity_text": text})
                counts["symptom"] += 1

    print(f"Coded records -> {args.output}: {counts}")
    if stopped_early:
        print(f"Stopped early due to a rate/quota limit: {stopped_early}")


if __name__ == "__main__":
    main()
