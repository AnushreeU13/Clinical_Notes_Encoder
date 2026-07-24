# Databricks notebook source
# MAGIC %md
# MAGIC # Clinical NLP Pipeline: Delta Lake + MLflow
# MAGIC
# MAGIC **Cluster**: Serverless (Databricks Free Edition) or any DBR 13.x+ ML runtime.
# MAGIC No cluster creation needed on Free Edition -- this notebook auto-attaches to
# MAGIC serverless compute.
# MAGIC
# MAGIC **Before running**: this workspace's Unity Catalog needs a `clinical_pipeline`
# MAGIC catalog and the `raw`/`processed`/`evaluation` schemas. The cell below creates
# MAGIC them for you (`CREATE ... IF NOT EXISTS`), but if it fails because you don't
# MAGIC have catalog-creation rights, go to **Catalog -> Create catalog** in the UI
# MAGIC first, name it `clinical_pipeline`, then re-run.
# MAGIC
# MAGIC **Data source**: this notebook reads the pipeline's local JSONL outputs
# MAGIC (`data/parsed/*.jsonl`) directly from this Git folder -- no external upload
# MAGIC needed, since the whole repo (incl. that data) is already synced here.

# COMMAND ----------

# MAGIC %pip install icd10-cm groq python-dotenv

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC Adjust `pipeline_path` if your Git folder isn't at the default location
# MAGIC (Workspace sidebar -> your Git folder -> right-click `pipeline` -> Copy path).

# COMMAND ----------

dbutils.widgets.text(
    "pipeline_path",
    "/Workspace/Users/au11@illinois.edu/Clinical_Notes_Encoder/pipeline",
    "Path to pipeline/ folder in this Git folder",
)
PIPELINE_PATH = dbutils.widgets.get("pipeline_path")
PATIENTS_JSONL = f"{PIPELINE_PATH}/data/parsed/patients.jsonl"
ENTITY_EXTRACTIONS_JSONL = f"{PIPELINE_PATH}/data/parsed/entity_extractions.jsonl"
CODED_RECORDS_JSONL = f"{PIPELINE_PATH}/data/parsed/coded_records.jsonl"

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS clinical_pipeline;
# MAGIC CREATE SCHEMA IF NOT EXISTS clinical_pipeline.raw;
# MAGIC CREATE SCHEMA IF NOT EXISTS clinical_pipeline.processed;
# MAGIC CREATE SCHEMA IF NOT EXISTS clinical_pipeline.evaluation;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load parsed FHIR patients -> `raw.patients` (demographics) + `raw.fhir_records` (full per-patient record)

# COMMAND ----------

from pyspark.sql.functions import col

patients_df = spark.read.json(PATIENTS_JSONL)

patients_demographics_df = patients_df.select("demographics.*")
(
    patients_demographics_df.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("clinical_pipeline.raw.patients")
)

(
    patients_df.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("clinical_pipeline.raw.fhir_records")
)

print(f"raw.patients: {patients_demographics_df.count()} rows")
print(f"raw.fhir_records: {patients_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load LLaMA entity extractions -> `processed.entity_extractions`

# COMMAND ----------

entity_extractions_df = spark.read.json(ENTITY_EXTRACTIONS_JSONL)
(
    entity_extractions_df.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("clinical_pipeline.processed.entity_extractions")
)
print(f"processed.entity_extractions: {entity_extractions_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load ICD-10/RxNorm coded records -> `processed.coded_records`

# COMMAND ----------

coded_records_df = spark.read.json(CODED_RECORDS_JSONL)
(
    coded_records_df.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("clinical_pipeline.processed.coded_records")
)
print(f"processed.coded_records: {coded_records_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate -> `evaluation.results`
# MAGIC
# MAGIC Note: CLAUDE.md's original eval design (`icd10_exact_match`,
# MAGIC `icd10_category_match`, `extraction_f1` against Synthea ground truth) assumed
# MAGIC Synthea's FHIR export carries ICD-10 ground truth. It doesn't -- Condition
# MAGIC resources are SNOMED-CT only (see parse_fhir.py). The metrics below
# MAGIC (from src/evaluate.py) substitute: ICD-10 coverage, match-method breakdown,
# MAGIC and a "grounded in chart" rate in place of exact/category match and F1.

# COMMAND ----------

import sys

sys.path.insert(0, f"{PIPELINE_PATH}/src")
import evaluate as evaluate_module  # noqa: E402

patients_records = [row.asDict(recursive=True) for row in patients_df.collect()]
coded_records = [row.asDict(recursive=True) for row in coded_records_df.collect()]

ground_truth = evaluate_module.build_ground_truth_diagnoses(patients_records)
report = {
    "diagnosis": evaluate_module.evaluate_diagnoses(coded_records, ground_truth),
    "medication": evaluate_module.evaluate_medications(coded_records),
    "procedure": evaluate_module.evaluate_passthrough(coded_records, "procedure"),
    "symptom": evaluate_module.evaluate_passthrough(coded_records, "symptom"),
}
print(report)

# COMMAND ----------

import json

report_rows = [
    {"entity_type": entity_type, "metrics_json": json.dumps(metrics)}
    for entity_type, metrics in report.items()
]
results_df = spark.createDataFrame(report_rows)
(
    results_df.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("clinical_pipeline.evaluation.results")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow: log this run

# COMMAND ----------

import mlflow

with mlflow.start_run(run_name="clinical_nlp_pipeline"):
    mlflow.log_param("llama_model", "llama-3.1-8b-instant")
    mlflow.log_param("prompt_version", "v1")
    mlflow.log_metric("num_patients", patients_df.count())
    mlflow.log_metric("num_notes_extracted", entity_extractions_df.count())
    mlflow.log_metric("num_diagnoses_coded", report["diagnosis"].get("total", 0))
    mlflow.log_metric("num_medications_coded", report["medication"].get("total", 0))
    mlflow.log_metric("icd10_coverage_pct", report["diagnosis"].get("icd10_coverage_pct", 0.0))
    mlflow.log_metric("grounded_in_chart_pct", report["diagnosis"].get("grounded_in_chart_pct", 0.0))
    mlflow.log_metric("rxnorm_coverage_pct", report["medication"].get("rxnorm_coverage_pct", 0.0))
    mlflow.log_dict(report, "evaluation_report.json")

print("Logged run to MLflow.")
