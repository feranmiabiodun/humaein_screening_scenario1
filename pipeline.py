# pipeline.py
# Pipeline: ingestion (fuzzy mapping), normalization, classifier, eligibility,
# atomic writers, plus helpers: recommend_from_reason, write_resubmission_candidates,
# compute_and_write_metrics.
from __future__ import annotations
import sys
import os
import re
import json
import uuid
import shutil
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable, Iterable

import difflib
import pandas as pd
import numpy as np

import argparse

# -----------------------
# Module-level defaults
# -----------------------
logging.basicConfig(level=logging.INFO)                    # simple logger config
logger = logging.getLogger("resubmission_pipeline")        # module logger

USE_HYBRID = True                                          # heuristics + LLM hybrid
LLM_CONF_THRESHOLD = 0.80                                  # LLM confidence cutoff
FALLBACK_TO_HEURISTICS_IF_LOW_CONF = True                  # fallback behavior
REFERENCE_DATE = pd.to_datetime("2025-07-30")              # default reference date
REJECTION_LOG_PATH = "rejection_log.csv"                   # rejection log path

# -----------------------
# Ingestion / mapping helpers
# -----------------------
def _normalize_col_name(name: str) -> str:
    """Normalize a column name to a compact token for matching."""
    n = str(name or "").strip().lower()
    n = re.sub(r"[^a-z0-9]+", "", n)
    return n

def infer_column_mapping(
    source_columns: Iterable[str],
    schema: Iterable[str],
    synonyms: Optional[Dict[str, Iterable[str]]] = None,
    fuzzy_cutoff: float = 0.6,
) -> Dict[str, Optional[str]]:
    """
    Infer mapping canonical_field -> source_column (or None).
    Strategy: exact normalized match -> synonyms -> fuzzy match.
    """
    src_cols = list(source_columns)
    norm_to_src = { _normalize_col_name(c): c for c in src_cols }
    schema_list = list(schema)
    synonyms = synonyms or {
        "claim_id": ["id", "claimid", "claimno", "claim_number", "claim"],
        "patient_id": ["member", "memberid", "patient", "patientid", "subscriber"],
        "procedure_code": ["code", "proc", "procedure", "procedurecode", "cpt", "hcpcs"],
        "denial_reason": ["error_msg", "error", "reason", "denial", "denialreason", "errorreason"],
        "status": ["claim_status", "statuscode", "state"],
        "submitted_at": ["date", "submitted", "submission_date", "date_submitted", "created_at"],
        "source_system": ["source", "system", "source_system"]
    }

    norm_synonyms = { field: {_normalize_col_name(s) for s in syns} for field, syns in synonyms.items() }
    result: Dict[str, Optional[str]] = {field: None for field in schema_list}
    used_src = set()

    # exact normalized name match
    for field in schema_list:
        nfield = _normalize_col_name(field)
        if nfield in norm_to_src:
            src = norm_to_src[nfield]
            result[field] = src
            used_src.add(src)

    # synonyms match
    for field in schema_list:
        if result[field] is not None:
            continue
        for token in norm_synonyms.get(field, set()):
            if token in norm_to_src:
                src = norm_to_src[token]
                if src not in used_src:
                    result[field] = src
                    used_src.add(src)
                    break

    # fuzzy match remaining
    remaining_src_norms = [k for k,v in norm_to_src.items() if v not in used_src]
    for field in schema_list:
        if result[field] is not None:
            continue
        candidates = difflib.get_close_matches(_normalize_col_name(field), remaining_src_norms, n=1, cutoff=fuzzy_cutoff)
        if candidates:
            chosen_norm = candidates[0]
            src = norm_to_src[chosen_norm]
            result[field] = src
            used_src.add(src)
            remaining_src_norms.remove(chosen_norm)

    return {k: (v if v is not None else None) for k,v in result.items()}

def ingest_and_unify(
    input_paths: list,
    sources: Optional[list] = None,
    schema: Optional[list] = None,
    user_mapping: Optional[Dict[str, str]] = None,
    fuzzy_cutoff: float = 0.6,
) -> pd.DataFrame:
    """Read files, infer mapping per-file, map to canonical schema, concat and enforce types."""
    schema = schema or ["claim_id","patient_id","procedure_code","denial_reason","status","submitted_at","source_system"]
    frames = []
    input_paths = [Path(p) for p in input_paths]

    # sources logic
    if sources is None:
        sources = [p.stem for p in input_paths]
    elif len(sources) == 1 and len(input_paths) > 1:
        sources = [sources[0]] * len(input_paths)
    elif len(sources) != len(input_paths):
        raise ValueError("sources must be None, length 1, or same length as input_paths")

    for p, src in zip(input_paths, sources):
        # read file
        if p.suffix.lower() in {".csv", ".txt"}:
            df = pd.read_csv(p)
        elif p.suffix.lower() in {".json"}:
            df = pd.read_json(p)
        else:
            raise ValueError(f"Unsupported file type: {p.suffix} for {p}")

        # infer mapping for this file
        inferred = infer_column_mapping(df.columns, schema, fuzzy_cutoff=fuzzy_cutoff)
        if user_mapping:
            for k,v in (user_mapping.items()):
                if k in schema and v:
                    inferred[k] = v

        # build source_col -> canonical_field rename map
        rename_map = { src_col: field for field, src_col in inferred.items() if src_col }

        # log mapping
        logger.info("Ingest: file=%s source=%s mapped=%s missing=%s",
                    p.name, src, {k:v for k,v in inferred.items() if v}, [k for k,v in inferred.items() if v is None])

        # rename columns (source -> canonical)
        df = df.rename(columns=rename_map)

        # compute short source_system token from provided src (last token after underscore)
        src_short = str(src).split("_")[-1] if src is not None else str(src)

        # ensure source_system column exists and fill empties with short token
        if "source_system" not in df.columns or df["source_system"].astype(str).str.strip().eq("").all():
            df["source_system"] = src_short
        else:
            df["source_system"] = df["source_system"].fillna("").astype(str)
            df.loc[df["source_system"].str.strip() == "", "source_system"] = src_short
            # also normalize any explicit values to their short form if they look like "<prefix>_<token>"
            df["source_system"] = df["source_system"].astype(str).apply(lambda v: v.split("_")[-1] if isinstance(v, str) and "_" in v else v)

        # create missing canonical columns
        missing_cols = [c for c in schema if c not in df.columns]
        for c in missing_cols:
            df[c] = pd.NA

        # keep canonical order
        df_normcols = df.reindex(columns=schema)
        frames.append(df_normcols)

    df_unified = pd.concat(frames, ignore_index=True, sort=False)

    # enforce dtypes (strings except submitted_at)
    dtype_map = {c: "string" for c in schema if c != "submitted_at"}
    df_unified = df_unified.astype(dtype_map, copy=False)

    # parse submitted_at
    if "submitted_at" in df_unified.columns:
        df_unified["submitted_at"] = pd.to_datetime(df_unified["submitted_at"], errors="coerce")

    return df_unified

# -----------------------
# Normalization (single source of truth)
# -----------------------
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize patient_id, status, denial_reason and submitted_at for downstream processing."""
    df = df.copy()
    sentinels = {'', ' ', 'none', 'null', 'nan', 'n/a', 'na'}

    # patient_id -> "Unknown" for missing-like
    if "patient_id" in df.columns:
        pid = df["patient_id"].astype("string").str.strip()
        pid_lower = pid.str.lower().fillna("")
        mask_pid_missing = pid.isna() | pid_lower.isin(sentinels)
        df.loc[mask_pid_missing, "patient_id"] = "Unknown"
        df["patient_id"] = df["patient_id"].astype("string").str.strip()

    # status -> normalized to 'approved'/'denied'
    if "status" in df.columns:
        df["status"] = df["status"].astype("string").str.strip().str.lower().fillna("")
        df["status"] = df["status"].replace({"approve": "approved", "deny": "denied"})

    # denial_reason -> pd.NA for missing-like, else trimmed
    if "denial_reason" in df.columns:
        dr = df["denial_reason"].astype("string")
        dr_stripped = dr.str.strip()
        dr_lower = dr_stripped.str.lower().fillna("")
        mask_dr_missing = dr.isna() | dr_lower.isin(sentinels)
        df.loc[mask_dr_missing, "denial_reason"] = pd.NA
        df.loc[~mask_dr_missing, "denial_reason"] = dr_stripped[~mask_dr_missing]

    # submitted_at -> parse datetime
    if "submitted_at" in df.columns:
        df["submitted_at"] = pd.to_datetime(df["submitted_at"], errors="coerce")

    return df

# -----------------------
# Heuristics & patterns
# -----------------------
KNOWN_RETRYABLE = ["missing modifier", "incorrect npi", "prior auth required"]
KNOWN_NON_RETRYABLE = ["authorization expired", "incorrect provider type"]
RETRY_KEYWORDS = ["missing", "modifier", "npi", "prior auth", "auth required", "authorization required", "auth"]
NON_RETRY_KEYWORDS = ["expired", "authorization expired", "provider type", "not billable", "not covered"]

def _compile_word_patterns(phrases):
    """Compile whole-word regex patterns (ignore case)."""
    compiled = {}
    for p in phrases:
        try:
            compiled[p] = re.compile(r"\b" + re.escape(p) + r"\b", flags=re.IGNORECASE)
        except Exception:
            compiled[p] = re.compile(re.escape(p), flags=re.IGNORECASE)
    return compiled

_COMPILED_KNOWN_RETRYABLE = _compile_word_patterns(KNOWN_RETRYABLE)
_COMPILED_KNOWN_NON_RETRYABLE = _compile_word_patterns(KNOWN_NON_RETRYABLE)

def _word_match_precompiled(text: Optional[str], compiled_patterns: Dict[str, re.Pattern]) -> bool:
    """Return True if any compiled pattern matches the text."""
    if text is None:
        return False
    t = str(text)
    for pat in compiled_patterns.values():
        if pat.search(t):
            return True
    return False

# -----------------------
# Mocked LLM (deterministic)
# -----------------------
def mocked_llm_infer(text: Optional[str]) -> Dict[str, Any]:
    """Deterministic mocked LLM classifier for ambiguous denial reasons."""
    t = (text or "").strip().lower()
    mapping = {
        "missing modifier": ("known_retryable", 0.98),
        "incorrect npi": ("known_retryable", 0.95),
        "prior auth required": ("known_retryable", 0.97),
        "authorization expired": ("known_non_retryable", 0.99),
        "incorrect provider type": ("known_non_retryable", 0.99),
    }
    if t in mapping:
        return {"label": mapping[t][0], "confidence": mapping[t][1]}
    if any(k in t for k in ["incorrect procedure", "form incomplete", "not billable", "procedure", "form"]):
        return {"label": "ambiguous", "confidence": 0.60}
    if t == "" or t in {"none", "null", "nan"}:
        return {"label": "ambiguous", "confidence": 0.50}
    return {"label": "ambiguous", "confidence": 0.55}

# -----------------------
# Classifier + strategy
# -----------------------
def classify_with_strategy(
    reason: Optional[str],
    use_heuristic_for_ambiguous: bool = True,
    use_hybrid: bool = USE_HYBRID,
    llm_conf_threshold: float = LLM_CONF_THRESHOLD,
    fallback_to_heuristics_if_low_conf: bool = FALLBACK_TO_HEURISTICS_IF_LOW_CONF,
    llm_infer_fn: Callable[[Optional[str]], Dict[str, Any]] = mocked_llm_infer,
) -> Dict[str, Any]:
    """Classify a denial reason using heuristics then mocked LLM with fallbacks."""
    if reason is None or (isinstance(reason, float) and np.isnan(reason)):
        return {"label": "missing", "confidence": 1.0, "source": "heuristic"}
    text = str(reason).strip()
    if text == "":
        return {"label": "missing", "confidence": 1.0, "source": "heuristic"}
    text_l = text.lower()
    if text_l in {"none", "null", "nan"}:
        return {"label": "missing", "confidence": 1.0, "source": "heuristic_text_normalized"}

    # heuristics
    if use_hybrid and use_heuristic_for_ambiguous:
        if _word_match_precompiled(text, _COMPILED_KNOWN_NON_RETRYABLE):
            return {"label": "known_non_retryable", "confidence": 0.99, "source": "heuristic"}
        if _word_match_precompiled(text, _COMPILED_KNOWN_RETRYABLE):
            return {"label": "known_retryable", "confidence": 0.99, "source": "heuristic"}
        for kw in NON_RETRY_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", text, flags=re.IGNORECASE):
                return {"label": "heuristic_non_retryable", "confidence": 0.85, "source": "heuristic"}
        for kw in RETRY_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", text, flags=re.IGNORECASE):
                return {"label": "heuristic_retryable", "confidence": 0.85, "source": "heuristic"}

    # LLM (mock)
    llm_out = llm_infer_fn(text)
    llm_label = llm_out.get("label", "ambiguous")
    llm_conf = float(llm_out.get("confidence", 0.0))

    if not use_hybrid:
        return {"label": llm_label or "ambiguous", "confidence": llm_conf, "source": "llm"}
    if llm_conf >= llm_conf_threshold:
        return {"label": llm_label or "ambiguous", "confidence": llm_conf, "source": "llm"}

    # fallback
    if fallback_to_heuristics_if_low_conf:
        if any(kw in text_l for kw in RETRY_KEYWORDS):
            return {"label": "heuristic_retryable", "confidence": 0.50, "source": "fallback"}
        if any(kw in text_l for kw in NON_RETRY_KEYWORDS):
            return {"label": "heuristic_non_retryable", "confidence": 0.50, "source": "fallback"}
        return {"label": "ambiguous", "confidence": llm_conf, "source": "llm_low_conf"}
    else:
        return {"label": llm_label or "ambiguous", "confidence": llm_conf, "source": "llm_low_conf"}

# -----------------------
# Atomic rejection log writer (fixed append behavior)
# -----------------------
def _atomic_write_rejection_log(rejects: pd.DataFrame, target_path: str) -> None:
    """Write rejection rows to CSV atomically (create or append) WITHOUT duplicating headers."""
    target = Path(target_path)
    if len(rejects) == 0:
        logger.info("No rejected rows to write to %s", target_path)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    # write temp CSV first (contains header)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, newline="", suffix=".csv", encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
        rejects.to_csv(tmp_path, index=False)
    try:
        if not target.exists():
            # atomic replace for first-time write
            os.replace(tmp_path, target)
            logger.info("Wrote %d rejected rows to %s (created)", len(rejects), target_path)
        else:
            # append without header to avoid duplicate headers
            # use pandas append mode (not atomic) as it's simpler and avoids header duplication
            rejects.to_csv(target, mode="a", header=False, index=False)
            logger.info("Appended %d rejected rows to %s", len(rejects), target_path)
            # cleanup tmp
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        logger.exception("Failed to write rejection log atomically; attempting fallback append.")
        try:
            rejects.to_csv(target, mode="a", header=not target.exists(), index=False)
            logger.info("Fallback wrote %d rejected rows to %s", len(rejects), target_path)
        except Exception:
            logger.exception("Fallback write also failed.")

# -----------------------
# Core compute: eligibility (vectorized + cached)
# -----------------------
def compute_resubmission_eligibility(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
    write_rejection_log: bool = True,
    rejection_log_path: str = REJECTION_LOG_PATH,
    allow_hybrid: bool = USE_HYBRID,
    llm_conf_threshold: float = LLM_CONF_THRESHOLD,
    llm_infer_fn: Callable[[Optional[str]], Dict[str, Any]] = mocked_llm_infer,
) -> pd.DataFrame:
    """Enrich df with classification and resubmission eligibility columns."""
    required_cols = {"status", "patient_id", "submitted_at", "denial_reason"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        logger.warning("Missing required columns %s — creating with pd.NA and continuing", missing_cols)
        df = df.copy()
        for c in missing_cols:
            df[c] = pd.NA
    else:
        df = df.copy()

    # normalize submitted_at and set reference date
    df["submitted_at"] = pd.to_datetime(df["submitted_at"], errors="coerce")
    reference_date = pd.to_datetime(reference_date) if reference_date is not None else REFERENCE_DATE

    # cache classification per unique denial_reason
    placeholder_missing = "<__MISSING__>"
    map_keys = df["denial_reason"].fillna(placeholder_missing)
    unique_reasons = pd.Index(map_keys.unique())
    cache: Dict[str, Dict[str, Any]] = {}
    for key in unique_reasons:
        if key == placeholder_missing:
            cache[key] = {"label": "missing", "confidence": 1.0, "source": "heuristic"}
        else:
            cache[key] = classify_with_strategy(
                key,
                use_heuristic_for_ambiguous=True,
                use_hybrid=allow_hybrid,
                llm_conf_threshold=llm_conf_threshold,
                fallback_to_heuristics_if_low_conf=FALLBACK_TO_HEURISTICS_IF_LOW_CONF,
                llm_infer_fn=llm_infer_fn,
            )

    # map cached results back to dataframe
    df["_denial_classification"] = map_keys.map(lambda k: cache.get(k, {"label": "ambiguous"})["label"])
    df["_denial_confidence"] = map_keys.map(lambda k: float(cache.get(k, {"confidence": 0.0})["confidence"]))
    df["_denial_inference_source"] = map_keys.map(lambda k: cache.get(k, {"source": "unknown"})["source"])

    # compute days since submitted (numeric)
    df["_days_since_submitted"] = (pd.to_datetime(reference_date) - df["submitted_at"]).dt.days

    # patient presence check (treat "Unknown" as missing)
    patient_series = df["patient_id"].astype(str).str.strip()
    df["_patient_present"] = df["patient_id"].notna() & patient_series.ne("") & patient_series.str.lower().ne("unknown")

    # eligibility rule
    allowed_classes_for_resubmit = {"known_retryable", "heuristic_retryable"}
    is_denied = df["status"].astype(str).str.strip().str.lower() == "denied"
    days_ok = df["_days_since_submitted"].notna() & (df["_days_since_submitted"] > 7)
    class_allowed = df["_denial_classification"].isin(allowed_classes_for_resubmit)
    df["resubmission_eligible"] = is_denied & df["_patient_present"] & days_ok & class_allowed

    # human-readable reason with precedence
    reason = pd.Series("ambiguous_classification", index=df.index)

    # normalized status to detect previously approved rows
    status_lower = df["status"].astype(str).str.strip().str.lower()

    # mark rows that are NOT denied:
    # - if they were previously approved, label "previously_approved"
    reason.loc[~is_denied & (status_lower == "approved")] = "previously_approved"
    reason.loc[~is_denied & (status_lower != "approved")] = "not_denied"

    pid_missing = ~df["_patient_present"]
    days_missing = df["_days_since_submitted"].isna()
    too_recent = df["_days_since_submitted"] <= 7
    is_allowed = df["_denial_classification"].isin(allowed_classes_for_resubmit)
    is_known_non_retry = df["_denial_classification"].isin(["known_non_retryable", "heuristic_non_retryable"])
    is_missing_denial = df["_denial_classification"] == "missing"

    # for denied rows, preserve the existing precedence and labels
    reason.loc[is_denied & pid_missing] = "missing_patient_id"
    reason.loc[is_denied & days_missing] = "missing_submitted_at"
    reason.loc[is_denied & too_recent] = "too_recent"
    reason.loc[is_denied & is_allowed] = df.loc[is_denied & is_allowed, "_denial_classification"]
    reason.loc[is_denied & is_known_non_retry] = "non_retryable"
    reason.loc[is_denied & is_missing_denial] = "missing_denial_reason"

    df["reason"] = reason


    # write rejection log (ambiguous + non-retryable) but ONLY for denied claims
    rejection_mask = ~df["resubmission_eligible"].astype(bool)
    rejects = df[rejection_mask]
    if write_rejection_log:
        try:
            _atomic_write_rejection_log(rejects, rejection_log_path)
        except Exception:
            logger.exception("Failed to write rejection log")

    return df

# ----- Added helpers from second script -----

def recommend_from_reason(reason_text: Optional[str], classification: Optional[str] = None) -> str:
    # small deterministic recommender for human-readable suggested action
    if reason_text is None or (isinstance(reason_text, float) and pd.isna(reason_text)):
        if classification in ("known_retryable", "heuristic_retryable"):
            return "Review denial note, correct the issue, and resubmit"
        return "Review denial reason and decide manually"
    t = str(reason_text).strip().lower()
    if "incorrect npi" in t or re.search(r'\bnpi\b', t):
        return "Review NPI number and resubmit"
    if "missing modifier" in t or "modifier" in t:
        return "Add missing modifier and resubmit"
    if "prior auth" in t or "prior authorization" in t or "auth required" in t:
        return "Obtain prior authorization and resubmit"
    if "form incomplete" in t or "incomplete" in t:
        return "Complete missing form fields and resubmit"
    if "incorrect procedure" in t or "incorrect procedure code" in t:
        return "Verify procedure code (CPT/HCPCS) and resubmit"
    if "not billable" in t:
        return "Not typically retryable — review payer rules; consider manual appeal"
    if classification in ("known_retryable", "heuristic_retryable"):
        return "Review denial note, correct the issue, and resubmit"
    return "Manual review recommended"

def write_resubmission_candidates(df: pd.DataFrame, out_path: str = "resubmission_candidates.json") -> str:
    candidates = df[df["resubmission_eligible"].astype(bool)].copy()
    out_list = []
    seen = set()
    for _, row in candidates.iterrows():
        cid = None if pd.isna(row.get("claim_id")) else str(row.get("claim_id")).strip()
        if cid and cid in seen:
            continue
        if cid:
            seen.add(cid)
        out_item = {
            "claim_id": cid if cid else None,
            "patient_id": None if pd.isna(row.get("patient_id")) else row.get("patient_id"),
            "submitted_at": None if pd.isna(row.get("submitted_at")) else pd.to_datetime(row.get("submitted_at")).isoformat(),
            "denial_reason": None if pd.isna(row.get("denial_reason")) else row.get("denial_reason"),
            "_denial_classification": row.get("_denial_classification"),
            "_denial_confidence": float(row.get("_denial_confidence")) if pd.notna(row.get("_denial_confidence")) else None,
            "source_system": row.get("source_system") if pd.notna(row.get("source_system")) else "unknown",
            "recommended_changes": recommend_from_reason(row.get("denial_reason"), row.get("_denial_classification"))
        }
        out_list.append(out_item)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", encoding="utf-8") as fh:
            tmp = fh.name
            json.dump(out_list, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, out_path)
        logger.info("Wrote %d candidates to %s", len(out_list), out_path)
    except Exception as e:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
        logger.exception("Failed to write candidates JSON: %s", e)
        raise
    return out_path

def compute_and_write_metrics(df: pd.DataFrame, out_dir: str = ".", write_json: bool = True) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())
    total_claims = int(len(df))
    total_denied = int(df['status'].astype(str).str.lower().eq('denied').sum())
    total_flagged = int(df['resubmission_eligible'].astype(bool).sum())
    total_excluded = int(total_denied - total_flagged)
    counts_by_source = df['source_system'].value_counts(dropna=False).to_dict()
    flagged_by_source = df[df['resubmission_eligible']].groupby('source_system').size()
    flagged_by_source = flagged_by_source.reindex(index=list(counts_by_source.keys()), fill_value=0).to_dict()
    excluded_reasons = df[~df['resubmission_eligible']]['reason'].value_counts(dropna=False).to_dict()
    denial_class_counts = df['_denial_classification'].value_counts(dropna=False).to_dict()
    metrics = {
        "run_id": run_id,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "total_claims": total_claims,
        "total_denied": total_denied,
        "total_flagged_for_resubmission": total_flagged,
        "total_excluded_denied": total_excluded,
        "counts_by_source": counts_by_source,
        "flagged_by_source": flagged_by_source,
        "excluded_reasons": excluded_reasons,
        "denial_classification_counts": denial_class_counts
    }
    if write_json:
        fname = os.path.join(out_dir, f"pipeline_metrics_summary_{run_id}.json")
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", dir=out_dir) as fh:
                tmp = fh.name
                json.dump(metrics, fh, indent=2)
            os.replace(tmp, fname)
            logger.info("Wrote metrics JSON: %s flagged=%d", fname, total_flagged)
        except Exception as e:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
            logger.exception("Failed to write metrics JSON: %s", e)
            raise
    logger.info("metrics: run_id=%s total=%d denied=%d flagged=%d excluded_denied=%d by_source=%s",
                metrics["run_id"], total_claims, total_denied, total_flagged, total_excluded, counts_by_source)
    return metrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run pipeline (ingest -> normalize -> compute -> write).")
    parser.add_argument("inputs", nargs="+", help="Input files (csv/json).")
    parser.add_argument("--sources", nargs="*", help="Optional list of source tokens.")
    parser.add_argument("--user-mapping-json", help="JSON string mapping canonical_field -> source_column")
    parser.add_argument("--reference-date", default=None, help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", default=None, help="Directory to write outputs. Default = current working directory.")
    args = parser.parse_args()

    run_id = uuid.uuid4().hex

    # Use explicit out-dir if passed, otherwise use current working directory
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path.cwd()

    # optional user mapping
    user_map = None
    if args.user_mapping_json:
        user_map = json.loads(args.user_mapping_json)

    # pipeline flow (calls existing functions from this module)
    df = ingest_and_unify(args.inputs, sources=args.sources, user_mapping=user_map)
    df = normalize_df(df)

    rejection_log_path = str(out_dir / f"rejection_{run_id}.csv")
    enriched = compute_resubmission_eligibility(
        df,
        reference_date=args.reference_date,
        write_rejection_log=True,
        rejection_log_path=rejection_log_path
    )

    candidates_path = str(out_dir / f"resubmission_candidates_{run_id}.json")
    write_resubmission_candidates(enriched, out_path=candidates_path)

    metrics = compute_and_write_metrics(enriched, out_dir=str(out_dir), write_json=True)

    # summary
    print("Pipeline run complete.")
    print("run_id:", run_id)
    print("out_dir:", out_dir)
    print("candidates_path:", candidates_path)
    print("metrics_run_id:", metrics.get("run_id"))
    print("rejection_log_path:", rejection_log_path)
