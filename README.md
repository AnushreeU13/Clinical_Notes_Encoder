# Clinical_Notes_Encoder

clinical charts → codes, clean and familiar to anyone in healthcare

Ingests synthetic patient data (FHIR R4 JSON), extracts clinical entities with LLaMA,
maps them to ICD-10 / RxNorm codes, stores results in Databricks Delta Lake, and
tracks experiments with MLflow.

## Structure

```
Clinical_Notes_Encoder/
├── synthea/    # Synthetic patient data generator (Java, vendored from synthea-international/synthea)
└── pipeline/   # NLP coding pipeline (Python) — see pipeline/CLAUDE.md for full spec
```

- `synthea/` — run this to generate FHIR R4 patient bundles under `synthea/output/fhir/`.
- `pipeline/` — parses FHIR, runs LLaMA NER, maps ICD-10/RxNorm, writes Delta Lake tables, logs to MLflow.
  See [pipeline/CLAUDE.md](pipeline/CLAUDE.md) for the full pipeline design and conventions.

## Getting started

1. Generate data: from `synthea/`, run `./run_synthea` (or `run_synthea.bat` on Windows).
2. Copy/point `synthea/output/fhir/` into `pipeline/data/synthea_output/`.
3. In `pipeline/`, copy `.env.example` to `.env` and fill in `GROQ_API_KEY` (and Databricks host/token once available).
4. `pip install -r pipeline/requirements.txt`
5. Run the pipeline stages in `pipeline/src/`.
