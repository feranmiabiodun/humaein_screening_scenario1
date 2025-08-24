# Resubmission Pipeline

## Overview

This repository contains a resubmission-classification pipeline and a small FastAPI service that exposes the pipeline over HTTP.

* **Use the Jupyter notebook** `HumaeinScreening1_notebook` for grading, review, narrative, experiments and a step-by-step walkthrough. The notebook is the authoritative document for graders.
* **Use `pipeline.py`** for programmatic/CLI runs and to back the FastAPI service (`pipeline_service.py`). `pipeline.py` exports the core functions (ingest, normalize, classify, compute eligibility, and atomic writers).
* **Use `pipeline_service.py`** (the FastAPI app) to run the pipeline via HTTP multipart uploads. The API calls the same `pipeline.py` internals so behavior matches CLI runs.

---

## Repo contents (what to expect)

* `data/` — sample input files used for testing (e.g. `emr_alpha.csv`, `emr_beta.json`).
* `HumaeinScreening1_notebook` — Jupyter notebook (narrative, experiments, grading).
* `pipeline.py` — the merged pipeline implementation (ingest, normalization, classifier, eligibility, writers).
* `pipeline_service.py` — FastAPI service that invokes the pipeline.
* Output artifacts (created when running):

  * `resubmission_candidates_<runid>.json`
  * `pipeline_metrics_summary_<uuid>.json`
  * `rejection_<runid>.csv`
* `.gitignore`, `requirements` — dependency list, etc.
---

## High-level architecture & flow

1. **Ingest & Unify** (`ingest_and_unify`)

   * Read CSV/JSON input files. Infer column mapping to canonical schema (exact name -> synonyms -> fuzzy). Optional user mapping overrides inference. Ensures `source_system` is present.

2. **Normalize** (`normalize_df`)

   * Clean `patient_id`, `status`, `denial_reason`, parse `submitted_at` and standardize sentinel values.

3. **Classify** (`classify_with_strategy`)

   * Heuristics-first (keyword lists and compiled regex) with optional hybrid mocked LLM. Mocked LLM is deterministic for reproducible runs.

4. **Eligibility & caching** (`compute_resubmission_eligibility`)

   * Vectorized classification caching by unique `denial_reason`. Computes `_denial_classification`, `_denial_confidence`, `_days_since_submitted`, `_patient_present`, and `resubmission_eligible`. Produces human-readable `reason` (e.g., `known_retryable`, `non_retryable`, `too_recent`, `missing_patient_id`, `previously_approved`).

5. **Writers & outputs**

   * `write_resubmission_candidates` writes `resubmission_candidates_<runid>.json` atomically. Each candidate includes a `recommended_changes` suggestion.
   * `compute_and_write_metrics` writes `pipeline_metrics_summary_<uuid>.json` with run-level metrics.
   * `_atomic_write_rejection_log` appends or creates `rejection_<runid>.csv` for rows selected by the rejection mask.

---

## What the notebook is for

Open `HumaeinScreening1_notebook` for:

* Project goals, assumptions and narrative reasoning.
* Data exploration and sample runs.
* Design rationale for heuristics, hybrid strategy, caching and file-writing.
  **This notebook is intended for grading and review.**

`pipeline.py` is provided for production-style usage (CLI & API) and is more compact—use it to run or deploy the pipeline.

---

## Running locally (Windows examples)

**Activate venv and install deps**

```powershell
# create and activate venv (if not yet)
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell
# or: .venv\Scripts\activate.bat  (CMD)

# install requirements
pip install -r requirements
# FastAPI file uploads need python-multipart
pip install python-multipart
```

### Run the pipeline from CLI (writes outputs to current directory by default)

The CLI in `pipeline.py` writes outputs into the current working directory unless you pass `--out-dir`. Example:

```powershell
.venv\Scripts\python.exe pipeline.py data\emr_alpha.csv data\emr_beta.json --out-dir . --keep-outputs
```

After the run you will see:

* `resubmission_candidates_<runid>.json`
* `pipeline_metrics_summary_<uuid>.json`
* `rejection_<runid>.csv`

### Run the FastAPI service

Start the server:

```powershell
.venv\Scripts\python -m uvicorn pipeline_service:app --reload
```

Then use `curl` (Windows example) or any HTTP client to call `/run-pipeline/` (multipart/form-data):

```powershell
curl -X POST "http://127.0.0.1:8000/run-pipeline/" ^
 -F "files=@data\emr_alpha.csv" ^
 -F "files=@data\emr_beta.json" ^
 -F "write_candidates=true" ^
 -F "write_metrics=true" ^
 -F "keep_outputs=true"
```

**API behavior:** the FastAPI endpoint accepts multipart files and optional parameters:

* `files` (repeatable): CSV or JSON file uploads.
* `sources`: optional source tokens (one or one-per-file).
* `column_map_json`: optional JSON mapping canonical\_field -> source column to override inference.
* `write_candidates`, `write_metrics`, `keep_outputs` (booleans): control writing behavior.
* `reference_date` (optional): override date used for age calculations.

**API response** includes:

* `run_id`, `total_claims`, `total_denied`, `total_flagged_for_resubmission`, `candidates_returned_inline`, `candidates_path` (if written), `metrics` (object), `rejection_log_path` (if written).

Open `http://127.0.0.1:8000/docs` for the interactive OpenAPI UI.

---

## Output formats (summary)

**`resubmission_candidates_<runid>.json`** — list of candidates; each item:

```json
{
  "claim_id": "A123",
  "patient_id": "P001",
  "submitted_at": "2025-07-01T00:00:00",
  "denial_reason": "Missing modifier",
  "_denial_classification": "known_retryable",
  "_denial_confidence": 0.99,
  "source_system": "alpha",
  "recommended_changes": "Add missing modifier and resubmit"
}
```

**`pipeline_metrics_summary_<uuid>.json`** — run-level metrics: run id, timestamp, totals, counts-by-source, flagged\_by\_source, excluded\_reasons, denial\_classification\_counts.

**`rejection_<runid>.csv`** — rows captured by the configured rejection mask; includes canonical columns and computed fields (`_denial_classification`, `_denial_confidence`, `_days_since_submitted`, `_patient_present`, `resubmission_eligible`, `reason`). Rejection mask is configurable in `pipeline.py`.

---

## Configurable knobs (edit `pipeline.py`)

At top of `pipeline.py`:

* `USE_HYBRID` — enable heuristics + LLM hybrid.
* `LLM_CONF_THRESHOLD` — LLM confidence cutoff.
* `FALLBACK_TO_HEURISTICS_IF_LOW_CONF` — fallback behavior.
* `REFERENCE_DATE` — default reference date for age calculation.
* `REJECTION_LOG_PATH` — default rejection log path (overridden per-run).

Other tunables: `KNOWN_RETRYABLE`, `KNOWN_NON_RETRYABLE`, `RETRY_KEYWORDS`, `NON_RETRY_KEYWORDS`, `infer_column_mapping(...).synonyms` and `fuzzy_cutoff`.

---

## Troubleshooting & tips

* If FastAPI raises `Form data requires "python-multipart"`, install it: `pip install python-multipart`.
* To force writing outputs into the folder where you run the command use `--out-dir .` (or pass an absolute path).
* Logging: `pipeline.py` configures `logging.basicConfig(level=logging.INFO)`. If you don’t see INFO logs, another component may have configured logging differently.
* The LLM used here is a **deterministic mock** (`mocked_llm_infer`) for reproducibility. 

---

## Testing & grading 

* Open the notebook and run the demonstration cells — these are the canonical examples for graders.
* Run `pipeline.py` with `--out-dir .` and compare `resubmission_candidates` fields and `recommended_changes` to examples in the notebook.
* Use the API to test file uploads and to confirm output paths and metrics match CLI runs.
