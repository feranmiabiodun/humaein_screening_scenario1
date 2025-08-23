# run_pipeline.py
import sys
import pandas as pd
from pathlib import Path
from pipeline import (
    compute_resubmission_eligibility,
    write_resubmission_candidates,
    compute_and_write_metrics,
    REFERENCE_DATE,
)

def load_input(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(p)
    if p.suffix.lower() in {".json"}:
        return pd.read_json(p)
    raise ValueError("Unsupported file type: " + p.suffix)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_pipeline.py <input_path> <source_system>")
        sys.exit(2)

    input_path = sys.argv[1]
    source_system = sys.argv[2]

    df = load_input(input_path)
    # attach source_system if not present
    if "source_system" not in df.columns:
        df["source_system"] = source_system

    # run pipeline (won't write rejection log by default here; set write_rejection_log=True if desired)
    enriched = compute_resubmission_eligibility(df, reference_date=REFERENCE_DATE, write_rejection_log=True)

    # write artifacts
    candidates_path = write_resubmission_candidates(enriched, out_path="resubmission_candidates.json")
    metrics = compute_and_write_metrics(enriched, out_dir=".", write_json=True)

    # print brief summary
    print(f"Input: {input_path}  |  Rows: {len(df)}")
    print(f"Candidates written to: {candidates_path}")
    print(f"Metrics run_id: {metrics['run_id']}  |  total_claims: {metrics['total_claims']}  flagged: {metrics['total_flagged_for_resubmission']}")
