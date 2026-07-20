-- Queries for the Clinical Notes Encoder Databricks Dashboard.
-- Run these in SQL Editor (against the Serverless Starter Warehouse) and save
-- each as a query, then add each saved query to a new Dashboard as a widget.
-- See notebooks/pipeline_databricks.py for how these tables get populated.

-- 1. Patient count by gender
SELECT gender, COUNT(*) AS patient_count
FROM clinical_pipeline.raw.patients
GROUP BY gender;

-- 2. Patient age distribution (10-year buckets)
SELECT
  FLOOR(DATEDIFF(CURRENT_DATE(), CAST(birth_date AS DATE)) / 365.25 / 10) * 10 AS age_bucket,
  COUNT(*) AS patient_count
FROM clinical_pipeline.raw.patients
GROUP BY 1
ORDER BY 1;

-- 3. Extraction progress: notes extracted vs. total notes in the corpus
SELECT
  (SELECT COUNT(*) FROM clinical_pipeline.processed.entity_extractions) AS notes_extracted,
  (SELECT SUM(SIZE(clinical_notes)) FROM clinical_pipeline.raw.fhir_records) AS notes_total;

-- 4. Extracted entity volume by type
SELECT entity_type, COUNT(*) AS count
FROM clinical_pipeline.processed.coded_records
GROUP BY entity_type
ORDER BY count DESC;

-- 5. Diagnosis ICD-10 match method breakdown
SELECT match_method, COUNT(*) AS count
FROM clinical_pipeline.processed.coded_records
WHERE entity_type = 'diagnosis'
GROUP BY match_method;

-- 6. Medication RxNorm mapping coverage
SELECT
  CASE WHEN rxnorm_code IS NOT NULL THEN 'mapped' ELSE 'unmapped' END AS rxnorm_status,
  COUNT(*) AS count
FROM clinical_pipeline.processed.coded_records
WHERE entity_type = 'medication'
GROUP BY 1;

-- 7. Top 15 most-mentioned ICD-10 diagnosis codes
SELECT icd10_code, icd10_description, COUNT(*) AS mentions
FROM clinical_pipeline.processed.coded_records
WHERE entity_type = 'diagnosis' AND icd10_code IS NOT NULL
GROUP BY icd10_code, icd10_description
ORDER BY mentions DESC
LIMIT 15;

-- 8. Evaluation summary metrics (parsed from evaluate.py's JSON output)
SELECT
  entity_type,
  get_json_object(metrics_json, '$.total') AS total,
  get_json_object(metrics_json, '$.icd10_coverage_pct') AS icd10_coverage_pct,
  get_json_object(metrics_json, '$.grounded_in_chart_pct') AS grounded_in_chart_pct,
  get_json_object(metrics_json, '$.rxnorm_coverage_pct') AS rxnorm_coverage_pct
FROM clinical_pipeline.evaluation.results;
