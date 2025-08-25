# fastapi.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import List, Optional
import tempfile
import os
import json
from pathlib import Path
from uuid import uuid4
import shutil
import traceback

# import the pipeline functions
from pipeline import (
    ingest_and_unify,
    normalize_df,
    compute_resubmission_eligibility,
    write_resubmission_candidates,
    compute_and_write_metrics,
    REFERENCE_DATE,
)

app = FastAPI(title="Resubmission Pipeline API")

# safety / operational best-effort
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB per file
MAX_FILES = 10                       # max number of files per request
MAX_RETURN_CANDIDATES = 1000         # limit how many candidate records returned inline by default


def _cleanup_dir(path: str) -> None:
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except Exception:
        # swallow errcandidate
        pass


@app.post("/run-pipeline/")
async def run_pipeline(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    sources: Optional[str] = Form(None),               # JSON array string or single token
    user_mapping_json: Optional[str] = Form(None),     # JSON string mapping canonical -> source col
    reference_date: Optional[str] = Form(None),       # optional override, e.g. "2025-08-24"
    write_rejection_log: bool = Form(True),
    write_candidates: bool = Form(True),
    write_metrics: bool = Form(True),
    keep_outputs: bool = Form(False),                  # if True, don't delete tmp dir (for debugging)
    return_candidate_limit: Optional[int] = Form(None) # limit number of candidates returned inline
):
    """
    Upload one or more CSV/JSON files and run the merged pipeline on them.
    Returns a compact summary and candidate records.
    """

    if len(files) > MAX_FILES:
        raise HTTPException(status_code=413, detail=f"too many files (max {MAX_FILES})")

    run_id = uuid4().hex
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"pipeline_run_{run_id}_"))
    saved_paths = []

    try:
        # 1) save uploads to tmp_dir
        for f in files:
            contents = await f.read()
            if len(contents) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"{f.filename} too large")
            suffix = Path(f.filename).suffix or ".csv"
            path = tmp_dir / f"{Path(f.filename).stem}{suffix}"
            path.write_bytes(contents)
            saved_paths.append(str(path))

        # 2) parse sources param (supports single token or JSON list)
        sources_list = None
        if sources:
            try:
                parsed = json.loads(sources)
                if isinstance(parsed, list):
                    sources_list = parsed
                elif isinstance(parsed, str):
                    sources_list = [parsed]
            except Exception:
                # fallback: treat as single token string
                sources_list = [sources]

        # 3) parse user_mapping JSON (canonical_field -> source_column)
        user_map = None
        if user_mapping_json:
            try:
                parsed = json.loads(user_mapping_json)
                if isinstance(parsed, dict):
                    user_map = parsed
                else:
                    raise ValueError("user_mapping_json must be a JSON object")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"bad user_mapping_json: {e}")

        # 4) ingest_and_unify (this handles fuzzy mapping, source_system inference, dtype enforcement)
        df = ingest_and_unify(saved_paths, sources=sources_list, user_mapping=user_map)

        # 5) normalization
        df = normalize_df(df)

        # 6) compute eligibility (use per-request rejection log to avoid cross-request collisions)
        rejection_log_path = str(tmp_dir / f"rejection_{run_id}.csv")
        ref_date_arg = REFERENCE_DATE if not reference_date else reference_date
        enriched = compute_resubmission_eligibility(
            df,
            reference_date=ref_date_arg,
            write_rejection_log=write_rejection_log,
            rejection_log_path=rejection_log_path
        )

        # 7) write candidates JSON & metrics (per-request files)
        candidates_path = None
        metrics = None
        if write_candidates:
            candidates_path = str(tmp_dir / f"resubmission_candidates_{run_id}.json")
            write_resubmission_candidates(enriched, out_path=candidates_path)
        if write_metrics:
            # compute_and_write_metrics generates a file in out_dir and returns metrics dict
            metrics = compute_and_write_metrics(enriched, out_dir=str(tmp_dir), write_json=True)

        # 8) prepare compact candidates payload
        candidates_df = enriched[enriched["resubmission_eligible"].astype(bool)].copy()
        total_candidates = len(candidates_df)
        limit = return_candidate_limit if (return_candidate_limit is not None) else MAX_RETURN_CANDIDATES
        if limit is not None and total_candidates > limit:
            # return only first N inline, but still persist full candidates file
            candidates_slice = candidates_df.head(limit).to_dict(orient="records")
            truncated = True
        else:
            candidates_slice = candidates_df.to_dict(orient="records")
            truncated = False

        response = {
            "run_id": run_id,
            "total_claims": len(enriched),
            "total_denied": int(enriched['status'].astype(str).str.lower().eq('denied').sum()),
            "total_flagged_for_resubmission": int(enriched['resubmission_eligible'].astype(bool).sum()),
            "candidates_returned_inline": len(candidates_slice),
            "candidates_truncated": truncated,
            "candidates_path": candidates_path,
            "metrics": metrics,
            "rejection_log_path": rejection_log_path,
        }
        # include inline candidates
        response["candidates"] = candidates_slice

        # 9) schedule cleanup unless user requested keep_outputs
        if not keep_outputs:
            background_tasks.add_task(_cleanup_dir, str(tmp_dir))
        else:
            # if keeping outputs, warn in response where things are
            response["kept_output_dir"] = str(tmp_dir)

        return response

    except HTTPException:
        # re-raise HTTP exceptions
        # schedule cleanup on error
        background_tasks.add_task(_cleanup_dir, str(tmp_dir))
        raise
    except Exception as e:
        # general failure
        tb = traceback.format_exc()
        # schedule cleanup
        background_tasks.add_task(_cleanup_dir, str(tmp_dir))
        raise HTTPException(status_code=500, detail={"error": "pipeline failed", "exception": str(e), "trace": tb})
