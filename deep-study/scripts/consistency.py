"""Consistency check: every tagged number in the deep-study reports must equal a
value recomputed from a named CSV row.

Tag syntax (placed immediately after the number it certifies, same line):

    **3,214** <!-- claim:data/a1-population-summary.csv::repo=vllm::confirmed_performance::int -->

fields (separated by ::): csv path (relative to experiments/deep-study), row filter (k=v&k=v), column, format
formats: int, pct0, pct1 (value*100), dec1, dec2, dec3, raw
The certified number is the last number token preceding the tag on that line.
"""

from __future__ import annotations

import json
import re
import sys

from common import DEEP, atomic_json, now_utc, read_csv, update_status

REPORTS = ["kernel-pr-delivery-report.md", "kernel-failures-report.md", "paper-to-production-report.md",
           "production-kernel-provenance-report.md", "SYNTHESIS.md"]
TAG = re.compile(r"<!--\s*claim:(.+?)::(.*?)::(.+?)::(\w+)\s*-->")
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def fmt(value: str, kind: str) -> str:
    v = float(value)
    return {"int": lambda: f"{round(v):,}", "pct0": lambda: f"{v * 100:.0f}", "pct1": lambda: f"{v * 100:.1f}",
            "dec1": lambda: f"{v:.1f}", "dec2": lambda: f"{v:.2f}", "dec3": lambda: f"{v:.3f}", "raw": lambda: value}[kind]()


def run() -> dict:
    cache = {}
    results = []
    for rep in REPORTS:
        path = DEEP / rep
        if not path.exists():
            results.append({"report": rep, "status": "missing_report"})
            continue
        for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in TAG.finditer(line):
                csv_path, filt, col, kind = m.groups()
                before = re.sub(r"<!--.*?-->", "", line[:m.start()])
                nums = NUM.findall(before)
                shown = nums[-1] if nums else None
                rec = {"report": rep, "line": ln, "csv": csv_path, "filter": filt, "column": col, "format": kind, "shown": shown}
                try:
                    if csv_path not in cache:
                        cache[csv_path] = read_csv(DEEP / csv_path)
                    conds = [c.split("=", 1) for c in filt.split("&") if c]
                    rows = [r for r in cache[csv_path] if all(r.get(k) == v for k, v in conds)]
                    if len(rows) != 1:
                        rec.update(status="fail", reason=f"{len(rows)} matching rows")
                    else:
                        expect = fmt(rows[0][col], kind)
                        rec["expected"] = expect
                        ok = shown is not None and shown.replace(",", "") == expect.replace(",", "")
                        rec["status"] = "pass" if ok else "fail"
                except Exception as e:  # noqa: BLE001
                    rec.update(status="fail", reason=str(e)[:200])
                results.append(rec)
    summary = {"checked_at_utc": now_utc(), "claims": sum(1 for r in results if "line" in r),
               "passed": sum(1 for r in results if r.get("status") == "pass"),
               "failed": [r for r in results if r.get("status") != "pass"], "results": results}
    atomic_json(DEEP / "data" / "consistency-check.json", summary)
    update_status(lambda d: d.__setitem__("last_consistency_check", {k: summary[k] for k in ["checked_at_utc", "claims", "passed"]} | {"failed": len(summary["failed"])}))
    print(f"claims {summary['claims']} passed {summary['passed']} failed {len(summary['failed'])}")
    for f in summary["failed"][:40]:
        print(" FAIL", json.dumps(f)[:300])
    return summary


if __name__ == "__main__":
    s = run()
    sys.exit(1 if s["failed"] else 0)
