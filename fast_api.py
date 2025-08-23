# fastapi_app_with_mapping.py
from fastapi import FastAPI, File, UploadFile, Form
from typing import Any, Dict, Optional
import io, json, pandas as pd, re

app = FastAPI(title="Claims Resubmission API (with column mapping)")

# a small dictionary of likely column-name synonyms -> canonical name
SYNONYMS = {
    "id": "claim_id",
    "claimid": "claim_id",
    "claim": "claim_id",
    "member": "patient_id",
    "patient": "patient_id",
    "patientid": "patient_id",
    "code": "procedure_code",
    "procedure": "procedure_code",
    "error_msg": "denial_reason",
    "error": "denial_reason",
    "denial": "denial_reason",
    "date": "submitted_at",
    "submitted": "submitted_at",
    "status": "status",
    "source": "source_system",
    "source_system": "source_system",
}

def _auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to rename columns by matching lowercased column tokens to SYNONYMS.
    Leaves columns unchanged if no mapping found.
    """
    rename = {}
    for col in df.columns:
        key = re.sub(r"\W+", "", str(col).strip().lower())  # remove punctuation/whitespace
        if key in SYNONYMS:
            rename[col] = SYNONYMS[key]
        else:
            # try partial-token matching (e.g. "error_msg" -> "errormsg" -> matches)
            for token, canon in SYNONYMS.items():
                if token in key and canon not in rename.values():
                    rename[col] = canon
                    break
    if rename:
        return df.rename(columns=rename)
    return df

def _apply_user_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Apply a user-provided mapping dict (source_col -> canonical_col).
    Only renames columns that are present in the uploaded dataframe.
    """
    safe_map = {src: dst for src, dst in mapping.items() if src in df.columns and isinstance(dst, str)}
    return df.rename(columns=safe_map)

@app.post("/upload")
async def upload_and_run(
    file: UploadFile = File(...),
    source_system: str = Form("uploaded"),
    column_map_json: Optional[str] = Form(None),  # optional JSON string: {"id":"claim_id", ...}
    write_rejection_log: bool = Form(False),
) -> Any:
    """
    Accept CSV or JSON file, optionally a JSON mapping of columns, run pipeline, return candidates.
    - column_map_json: optional JSON string mapping original column names to canonical names.
    - source_system: tag to set on incoming rows (unless column_map_json already maps such a column).
    """
    content = await file.read()
    try:
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_json(io.BytesIO(content))
    except Exception as e:
        return {"error": "failed to parse upload", "detail": str(e)}

    # 1) Try to apply user-provided mapping (if any)
    if column_map_json:
        try:
            user_map = json.loads(column_map_json)
            if isinstance(user_map, dict):
                df = _apply_user_mapping(df, user_map)
        except Exception as e:
            return {"error": "failed to parse column_map_json", "detail": str(e)}

    # 2) Auto-map common synonyms for unmapped columns
    df = _auto_map_columns(df)

    # 3) Ensure source_system present (user can override via mapping or form field)
    if "source_system" not in df.columns:
        df["source_system"] = source_system

    # 4) Now call your existing pipeline functions (assumes they are importable/in-scope)
    try:
        enriched = compute_resubmission_eligibility(df, reference_date=REFERENCE_DATE, write_rejection_log=write_rejection_log)
    except Exception as e:
        return {"error": "pipeline execution failed", "detail": str(e)}

    # 5) Prepare a compact candidates payload
    candidates_df = enriched[enriched["resubmission_eligible"].astype(bool)]
    candidates = candidates_df.to_dict(orient="records")

    return {"count": len(candidates), "candidates": candidates}
