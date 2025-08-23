# humaein_screening_scenario1
A compact pipeline to ingest messy EMR claim exports, unify their schema, classify denial reasons (heuristics + mocked LLM), and surface automated resubmission candidates with audit-ready outputs. Includes a documented Jupyter notebook (canonical, for review) and a runnable pipeline.py (CLI + fuzzy ingestion, metrics, and rejection logging).

---

# Claim Resubmission Pipeline

Hi — I built this pipeline to take messy claim exports from different EMR systems, standardize them, classify denial reasons, and flag claims we can safely resubmit automatically. There are two versions of the work in this repo:

* **Jupyter notebook** — my step-by-step, annotated, explanation-first version. This is the canonical artifact you should open for grading or to understand why each step exists.
* **`pipeline.py`** — a compact, runnable CLI version of the same logic. Use this to run the pipeline from the command line, integrate into tests, or exercise the FastAPI scaffold.

---

## What the pipeline does (high level)

It reads one or more input files (CSV or JSON), automatically map each file’s column names to a single canonical schema, combine them into one table, then:

1. Normalize the data (dates, missing values, `status`, `patient_id`).
2. Classify denial reasons using heuristic rules plus a deterministic, mocked LLM for ambiguous phrases.
3. Flag claims as eligible for automated resubmission when all rules are met:

   * `status == "denied"`,
   * `patient_id` is present and not `"Unknown"`,
   * submission is older than 7 days (reference date: `2025-07-30`),
   * denial reason is known retryable, or inferred retryable by the classifier.
4. Produce outputs for automation and audit:

   * `resubmission_candidates.json` — the list of candidate claims, including `recommended_changes` (rule-based) and `source_system` derived from the filename stem.
   * `pipeline_metrics_summary_<run_id>.json` — run metrics (counts, per-source breakdowns).
   * `rejection_log.csv` — ambiguous / non-retryable rows for manual review (can be disabled).

**Important:** `recommended_changes` are produced by deterministic rule mappings (not by the mocked LLM). The classifier only decides retryability vs non-retryable/ambiguous.

---

## Files in this repo

* `notebook.ipynb` (or similarly named Jupyter notebook) — canonical, annotated implementation with explanations and checks.
* `pipeline.py` — runnable CLI pipeline (ingest + fuzzy mapping, normalize, classify, write outputs).
* `data/` — example input files used for testing (e.g. `emr_alpha.csv`, `emr_beta.json`).
* Output files produced by the pipeline: `resubmission_candidates.json`, `pipeline_metrics_summary_<run_id>.json`, `rejection_log.csv`.

---

## Quick setup (Windows CMD)

Open a terminal in the project folder and run:

```cmd
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install pandas numpy
# optional for Jupyter/fastapi:
pip install jupyterlab
pip install fastapi "uvicorn[standard]"
```
---

## How to run

### 1) Use the Jupyter notebook (recommended for review / grading)

* Open the notebook and run cells in order. The notebook contains the step-by-step narrative and interactive checks.

### 2) Use the CLI (`pipeline.py`) to run the pipeline on files

Examples (from project root):

Run on two files — pipeline will infer column mapping and use filename stems as `source_system`:

```cmd
python pipeline.py data/emr_alpha.csv data/emr_beta.json
```

Provide explicit per-file `source_system` names:

```cmd
python pipeline.py data/emr_alpha.csv data/emr_beta.json --sources alpha beta
```

Disable writing the rejection log:

```cmd
python pipeline.py data/emr_alpha.csv data/emr_beta.json --no-write-rejection-log
```

**Outputs produced**

* `resubmission_candidates.json` — JSON array of candidates with fields like:

```json
{
  "claim_id": "A124",
  "patient_id": "P002",
  "submitted_at": "2025-07-10T00:00:00",
  "denial_reason": "Incorrect NPI",
  "_denial_classification": "known_retryable",
  "_denial_confidence": 0.99,
  "source_system": "alpha",
  "recommended_changes": "Review NPI number and resubmit"
}
```

* `pipeline_metrics_summary_<run_id>.json` — run-level metrics.
* `rejection_log.csv` — ambiguous and known non-retryable rows for manual review (unless disabled).

---

## Programmatic usage (Python API)

If you prefer to call functions directly from Python code, the main helpers are:

* `ingest_and_unify(input_paths, sources=None, schema=None, user_mapping=None)`
  Reads files, auto maps columns to the canonical schema, concatenates them, and returns a unified `DataFrame`. You may pass `user_mapping` to override inferred mappings.

* `normalize_df(df)`
  Normalizes `patient_id`, `status`, `denial_reason`, and parses `submitted_at`.

* `compute_resubmission_eligibility(df, reference_date=..., write_rejection_log=True)`
  Enriches the `DataFrame` with `_denial_classification`, `_denial_confidence`, `_patient_present`, `_days_since_submitted`, `resubmission_eligible`, and `reason`.

* `write_resubmission_candidates(df, out_path="resubmission_candidates.json")`
  Atomically writes the candidates JSON.

* `compute_and_write_metrics(df, out_dir=".", write_json=True)`
  Computes and writes run metrics.

Example:

```python
from pipeline import ingest_and_unify, normalize_df, compute_resubmission_eligibility, write_resubmission_candidates, compute_and_write_metrics

df = ingest_and_unify(["data/emr_alpha.csv", "data/emr_beta.json"])
df_norm = normalize_df(df)
enriched = compute_resubmission_eligibility(df_norm, write_rejection_log=True)
write_resubmission_candidates(enriched)
compute_and_write_metrics(enriched)
```

---

## Troubleshooting & verification checklist

If flagged counts don’t match expectations, check the following:

1. **Column mapping**

   * Inspect `ingest_and_unify` log lines — it reports how each source column mapped to canonical fields.
   * If a mapping is incorrect, call `ingest_and_unify(..., user_mapping={...})` to override.

2. **`source_system` correctness**

   * By default, the pipeline uses the filename stem as the `source_system` (e.g., `emr_alpha.csv` → `emr_alpha` → `"alpha"` in outputs). When running the CLI you can provide `--sources` to override.

3. **Normalization**

   * Confirm `patient_id`, `denial_reason`, and `submitted_at` were cleaned and parsed correctly (inspect `df_norm`).

4. **Classifier decisions**

   * Inspect `_denial_classification` and `_denial_confidence` to see why a denial reason was labeled.

5. **Eligibility details**

   * Check `_days_since_submitted`, `_patient_present`, and `resubmission_eligible` columns.

6. **Rejection log**

   * Open `rejection_log.csv` to see ambiguous/non-retryable rows written for manual review.

---

## Notes & design decisions

* The notebook is the **primary** artifact for grading and understanding the pipeline rationale.
* `pipeline.py` is a runnable form for convenience and integration; it implements the same core logic.
* Denial classification is hybrid: deterministic heuristics first, plus a mocked LLM for ambiguous phrases; fallback heuristics are used when confidence is low.
* `recommended_changes` are produced from deterministic rule mappings (so results are predictable and auditable).
* Ingest mapping is fuzzy (exact match → synonyms → fuzzy matching). If automatic mapping gets it wrong, use `user_mapping` to override.
* The reference date used in the eligibility rule is `2025-07-30`.
