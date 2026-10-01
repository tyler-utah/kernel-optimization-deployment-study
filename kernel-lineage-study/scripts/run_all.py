"""Re-derive every dataset, figure and report from cached inputs (no network), then run the
consistency check. Staged agent outputs (registries, screening/coding batches, move catalogs,
audits, failure histories, narratives, recoding) are inputs and are not regenerated here.

  python scripts/run_all.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ["anchors.py"],               # idempotent: upstream anchors + relation appends (after any recode apply)
    ["moves_final.py", "namemap"],
    ["assemble.py"],
    ["moves_final.py", "csv"],
    ["survival.py", "presence"],
    ["survival.py", "moves"],
    ["blame.py"],
    ["assemble.py"],            # re-assemble so code survival reflects the (cached) blame table
    ["moves_final.py", "csv"],
    ["snapshots.py"],
    ["failures.py"],
    ["analysis.py"],
    ["figures.py"],
    ["reports.py"],
    ["consistency.py"],
]

if __name__ == "__main__":
    sys.path.insert(0, HERE)
    import status as S
    S.log("run_all start")
    for step in STEPS:
        print(">>", " ".join(step), flush=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, step[0]), *step[1:]], cwd=HERE)
        if r.returncode != 0:
            S.log(f"run_all failed at {' '.join(step)} (exit {r.returncode})")
            sys.exit(r.returncode)
    S.log("run_all complete")
