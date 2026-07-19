# Clinical NLP Coding Pipeline

## What This Project Does
Ingests synthetic patient data (FHIR R4 JSON from Synthea), extracts clinical entities
using LLaMA, maps them to ICD-10 and RxNorm codes using Python packages only,
stores everything in Databricks Delta Lake, and tracks experiments with MLflow.

## Pipeline Stages
```
Synthea FHIR R4 JSON
       ↓
Parse FHIR → extract patient records, clinical notes, conditions
       ↓
LLaMA NER → structured JSON: diagnoses, medications, procedures, symptoms
       ↓
ICD-10 / RxNorm mapping → icd10-cm Python package + LLaMA validation
       ↓
Delta Lake → store raw + coded records as Delta tables
       ↓
MLflow → log every run: model, prompt version, extraction F1, mapping accuracy
       ↓
Evaluate → compare against Synthea ground truth ICD-10 codes
```

---

## Ownership Split

### Claude Code owns (write, run, debug directly):
- FHIR JSON parsing
- LLaMA entity extraction with structured output
- ICD-10 / RxNorm mapping using Python packages
- PySpark / Delta Lake notebook code
- MLflow logging
- Evaluation scripts

### User runs in Databricks (Claude Code writes the code + gives exact instructions):
For any Databricks step, Claude Code will:
1. Write the complete notebook code
2. Say exactly: "Open Databricks → create a new notebook → paste this → run"
3. Specify cluster runtime, libraries to install via UI, and exact Delta table paths
4. Never assume the user can run Databricks steps locally

---

## Tech Stack

| Stage | Tool |
|---|---|
| Synthetic data | Synthea (user clones and runs the repo) |
| FHIR parsing | Raw Python JSON — no heavy FHIR library |
| NER / Entity extraction | LLaMA 3 via Groq API (fast, free tier) |
| ICD-10 mapping | `icd10-cm` Python package — no CSV/Excel tables |
| RxNorm | LLaMA suggests concept name; `rxnorm` package for validation if available |
| Storage | Databricks Delta Lake (PySpark) |
| Experiment tracking | MLflow (pre-installed in Databricks) |
| Evaluation | scikit-learn precision / recall / F1 |

Do NOT download or load external ICD-10 CSV/Excel files. Use the `icd10-cm`
package exclusively for all code lookups.

---

## Project Structure
```
clinical-nlp-pipeline/
├── CLAUDE.md
├── requirements.txt
├── data/
│   └── synthea_output/          # User places Synthea /output folder here
├── src/
│   ├── parse_fhir.py            # Parse FHIR JSON → structured patient records
│   ├── extract_entities.py      # LLaMA NER → structured entity JSON
│   ├── map_codes.py             # Map entities → ICD-10 / RxNorm codes
│   ├── evaluate.py              # Compare pipeline output to Synthea ground truth
│   └── utils.py                 # Shared helpers (JSON validation, retries)
├── notebooks/
│   └── pipeline_databricks.py   # Databricks-ready notebook (uses % magic commands)
└── mlflow/
    └── tracking.py              # MLflow run logging helpers
```

---

## Implementation Rules

### FHIR Parsing
- Synthea outputs one JSON file per patient under `/output/fhir/`
- Each file contains a FHIR Bundle with resourceType entries:
  `Patient`, `Condition`, `MedicationRequest`, `Procedure`, `Observation`, `Encounter`
- Extract from each patient:
  - Demographics from `Patient`
  - Diagnoses from `Condition` (includes ground truth ICD-10 in `code.coding`)
  - Medications from `MedicationRequest`
  - Procedures from `Procedure`
  - Clinical notes from `DocumentReference` if present
- Store ground truth ICD-10 codes from Synthea separately — these are used for evaluation

### LLaMA / Entity Extraction
Always use structured JSON output. The extraction prompt must return exactly:
```json
{
  "diagnoses": ["type 2 diabetes", "hypertension"],
  "medications": ["metformin 500mg", "lisinopril"],
  "procedures": ["blood glucose test"],
  "symptoms": ["fatigue", "increased thirst"]
}
```
- Use Groq API: `from groq import Groq` with `response_format={"type": "json_object"}`
- Always validate the returned JSON (try/except) — if malformed, retry once with a
  simplified prompt before skipping the record
- Log the raw LLaMA response alongside the parsed output for debugging

### ICD-10 Mapping
```python
import icd10
results = icd10.search("type 2 diabetes")   # returns list of Code objects
# Each Code has: .code, .description, .billable
```
- Search each extracted diagnosis string against `icd10.search()`
- If multiple results, take top 3 candidates and pass back to LLaMA:
  "Given the clinical context '{note}', which of these ICD-10 codes is most appropriate?
   Options: {candidates}. Return only the code."
- Always store: entity_text, icd10_code, icd10_description, match_method (direct/llm_validated)

### RxNorm / Medications
- Use LLaMA to normalize the medication name to its generic form and suggest RxNorm concept
- If `rxnorm` Python package is available, validate. If not, store LLaMA output and flag as unvalidated
- Do not use any external API for RxNorm unless the package fails

### Delta Lake (Databricks)
When writing PySpark/Delta code:
- Always write it as a complete Databricks notebook cell (with `# COMMAND ----------` separators)
- Specify at the top of each notebook: minimum cluster runtime (e.g., DBR 13.x ML),
  and any pip installs needed (`%pip install icd10-cm groq`)
- Use these table names consistently:
  - `clinical_pipeline.raw.patients`
  - `clinical_pipeline.raw.fhir_records`
  - `clinical_pipeline.processed.entity_extractions`
  - `clinical_pipeline.processed.coded_records`
  - `clinical_pipeline.evaluation.results`
- Always include instructions: "In Databricks: go to Catalog → create schema 'clinical_pipeline'
  before running this cell"

### MLflow
- Log every pipeline run with:
  - `llama_model`: model name used
  - `prompt_version`: prompt template version
  - `num_records`: records processed
  - `extraction_f1`: NER F1 against ground truth
  - `icd10_exact_match`: % exact code matches
  - `icd10_category_match`: % matches at 3-character category level
- In Databricks, MLflow is pre-installed — just `import mlflow`

### Evaluation
- Synthea `Condition` resources include ICD-10 codes as ground truth
- Measure two levels:
  - Exact match: pipeline code == Synthea code
  - Category match: first 3 characters match (e.g., E11 for all Type 2 diabetes variants)
- Report per entity type (diagnosis, medication, procedure separately)

---

## How to Handle Databricks Steps

When a step must happen in Databricks, respond in this format:

```
[DATABRICKS STEP — user action required]

1. Go to: <exact location in Databricks UI>
2. Do: <exact action>
3. Then run this code:

<complete notebook code>

4. Expected output: <what success looks like>
```

---

## Starting Point

Begin by asking the user:
1. Have you cloned and run Synthea? If yes, where is the /output/fhir/ folder on your machine?
2. Do you have a Groq API key? (free at console.groq.com)
3. Do you have a Databricks workspace URL and personal access token?

Then start with `src/parse_fhir.py`.
