"""Verify every marked executive-summary number against generated CSVs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"
REPORT = EXP / "system-evolution-report.md"


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_value(value: float, style: str) -> str:
    if style == "pct1":
        return f"{value:.1%}"
    if style == "pp1":
        return f"{value * 100:+.1f} percentage points"
    if style == "num1":
        return f"{value:.1f}"
    if style == "int":
        return f"{int(value):,}"
    if style == "days1":
        return f"{value / 24:.1f} days"
    raise ValueError(f"unknown claim format: {style}")


def main() -> None:
    text = REPORT.read_text(encoding="utf-8")
    match = re.search(r"## Executive summary\s+(.*?)(?=\n## )", text, re.S)
    if not match:
        raise ValueError("Executive summary section not found")
    executive = match.group(1)
    headline = {r["finding_id"]: r for r in rows(RESULTS / "headline-numbers.csv")}
    pr = {(r["repo"], r["category"]): r for r in rows(RESULTS / "pr-evidence-summary.csv")}
    rules = {(r["repo"], r["rule_or_signal"]): r for r in rows(RESULTS / "rule-compliance-observed.csv")}
    claims = []
    marker = re.compile(r"<!-- claim:(headline|pr|rule):([^:]+)(?::([^:]+))?(?::([^:]+))?:([^:]+) -->")
    for line in executive.splitlines():
        for source, first, second, third, style in marker.findall(line):
            if source == "headline":
                finding_id = first
                if finding_id not in headline:
                    raise KeyError(f"missing headline finding {finding_id}")
                raw = float(headline[finding_id]["estimate"])
                source_file = "experiments/results/headline-numbers.csv"
                claim_id = finding_id
            elif source == "pr":
                repo, category, column = first, second, third
                key = (repo, category)
                if key not in pr:
                    raise KeyError(f"missing PR summary row {key}")
                raw = float(pr[key][column])
                source_file = "experiments/results/pr-evidence-summary.csv"
                claim_id = f"{repo}-{category}-{column}"
            else:
                repo, rule, column = first, second, third
                key = (repo, rule)
                if key not in rules:
                    raise KeyError(f"missing compliance row {key}")
                raw = float(rules[key][column])
                source_file = "experiments/results/rule-compliance-observed.csv"
                claim_id = f"{repo}-{rule}-{column}"
            expected = format_value(raw, style)
            claims.append({
                "claim_id": claim_id,
                "source_file": source_file,
                "expected_text": expected,
                "report_line": line.strip(),
                "exact_match": expected in line,
            })
    if not claims:
        raise ValueError("No executive-summary claim markers found")
    result = {
        "report": "experiments/system-evolution-report.md",
        "claims_checked": len(claims),
        "all_exact": all(item["exact_match"] for item in claims),
        "claims": claims,
    }
    (RESULTS / "consistency-check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["all_exact"]:
        raise SystemExit("One or more executive-summary claims do not match generated CSVs")
    print(f"verified {len(claims)} executive-summary numerical claims")


if __name__ == "__main__":
    main()
