"""Atomically update the persistent study checkpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
STATUS = ARTIFACT_ROOT / "results" / "study-status.json"


def update_phase(phase: str, status: str, outputs: list[str], note: str, resume: str) -> None:
    data = json.loads(STATUS.read_text(encoding="utf-8"))
    now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    entry = data["phases"][phase]
    if entry["started_at_utc"] is None:
        entry["started_at_utc"] = now
    entry["status"] = status
    if outputs:
        entry["outputs"] = outputs
    if note:
        entry["notes"] = note
    entry["completed_at_utc"] = now if status == "complete" else None
    data["current_phase"] = phase
    data["resume_next"] = resume
    data["updated_at_utc"] = now
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, STATUS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase")
    parser.add_argument("status", choices=["pending", "in_progress", "complete", "blocked"])
    parser.add_argument("--output", action="append", default=[])
    parser.add_argument("--note", default="")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--validation-iteration", type=int)
    parser.add_argument("--validation-sample")
    parser.add_argument("--validation-labels")
    parser.add_argument("--validation-metrics")
    parser.add_argument("--macro-f1", type=float)
    parser.add_argument("--kernel-perf-precision", type=float)
    parser.add_argument("--passed-threshold", action="store_true")
    parser.add_argument("--failure")
    args = parser.parse_args()
    update_phase(args.phase, args.status, args.output, args.note, args.resume)
    if args.validation_iteration is not None or args.failure:
        data = json.loads(STATUS.read_text(encoding="utf-8"))
        if args.validation_iteration is not None:
            data["validation"].update({
                "iteration": args.validation_iteration,
                "sample_file": args.validation_sample,
                "labels_file": args.validation_labels,
                "metrics_file": args.validation_metrics,
                "macro_f1": args.macro_f1,
                "kernel_perf_precision": args.kernel_perf_precision,
                "passed_threshold": args.passed_threshold,
            })
        if args.failure and not any(
            item.get("phase") == args.phase and item.get("error") == args.failure
            for item in data["failures"]
        ):
            data["failures"].append({
                "phase": args.phase,
                "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                "error": args.failure,
                "resume": args.resume,
            })
        tmp = STATUS.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, STATUS)


if __name__ == "__main__":
    main()
