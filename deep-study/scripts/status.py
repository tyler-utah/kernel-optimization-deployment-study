"""Create, show, and verify the deep-study checkpoint (STATUS.json).

Usage (from the project root):
    python experiments/deep-study/scripts/status.py init     # only if missing
    python experiments/deep-study/scripts/status.py show
    python experiments/deep-study/scripts/status.py verify   # re-check claimed outputs
"""

from __future__ import annotations

import hashlib
import sys

from common import (DEEP, HEADS, REPOS, ROOT, STATUS, A_START, B_START, CUTOFF, load_status,
                    now_utc, rel, save_status)


def sub(required, desc, unit="records"):
    return {"status": "pending", "completed": 0, "required": required, "unit": unit,
            "description": desc, "next": "", "outputs": [], "notes": ""}


def init() -> None:
    if STATUS.exists():
        print("STATUS.json exists; not overwriting")
        return
    data = {
        "schema_version": 1,
        "created_at_utc": now_utc(),
        "study": "Deep empirical follow-up: kernel delivery, failures, and papers to production",
        "study_cutoff_utc": CUTOFF.isoformat(),
        "windows": {
            "A_pr_delivery_12m": [A_START.date().isoformat(), CUTOFF.date().isoformat()],
            "B_failures_24m": [B_START.date().isoformat(), CUTOFF.date().isoformat()],
        },
        "repository_heads": {REPOS[k]: v for k, v in HEADS.items()},
        "first_study_inputs": {
            "status": "experiments/results/study-status.json",
            "rest_recent_prs": "experiments/data/github/{repo}/recent_prs/page-*.json",
            "rest_open_prs": "experiments/data/github/{repo}/open_prs/page-*.json",
            "lifecycle_rows": "experiments/results/pr-lifecycle-{repo}.csv",
            "first_parent_logs": "experiments/data/{repo}-log.txt",
            "blobless_git": "experiments/data/repos/{repo}",
            "rules": "experiments/results/contribution-rules.csv",
        },
        "work_packages": {
            "A": {"status": "pending", "title": "Deep PR delivery study", "subtasks": {
                "A1_population": sub(934, "Complete high-precision perf-PR population; adjudicate all 589+345 open-queue candidates and every ambiguous candidate", "adjudicated open-queue candidates"),
                "A2_metadata": sub(1, "Full-population metadata, KM survival, bootstrap effect sizes", "analysis"),
                "A3_collect": sub(None, "Detailed review/comment/timeline cache: all confirmed perf PRs (<=2500/repo) + >=500 matched non-perf per repo", "PR timelines"),
                "A3_code": sub(None, "Contextual coding of every detailed PR; second pass >=200/repo", "coded PRs"),
                "A3_second_pass": sub(400, "Fresh second-pass coding of >=200 records per repo and agreement", "records"),
                "A4_rules": sub(1, "Compliance with codified rules; recurring undocumented requests (>=10 PRs)", "analysis"),
                "A5_cases": sub(12, "12 verified PR case histories (4 fast, 4 slow, 4 abandoned/reverted/superseded)", "case histories"),
            }},
            "B": {"status": "pending", "title": "Confirmed kernel failures and reverts", "subtasks": {
                "B1_reverts": sub(None, "24-month revert census, context-confirmed", "revert records"),
                "B2_corpus": sub(150, ">=150 confirmed kernel-correctness fixes, coded", "confirmed cases"),
                "B3_counterfactual": sub(150, "Counterfactual detection technique + second pass on every case", "cases"),
                "B4_cases": sub(10, ">=10 failure case studies", "case studies"),
            }},
            "C": {"status": "pending", "title": "Conference paper census and deployment ladder", "subtasks": {
                "C1_census": sub(None, "Classify all MLSys 2025 (61) and ASPLOS 2025 research papers; second pass positives/ambiguous", "papers"),
                "C2_eval_audit": sub(None, "Evaluation audit for every kernel-style positive", "positives"),
                "C3_deployment": sub(None, "L0-L5/S ladder for every positive; >=20 negatives per conference checked", "papers"),
                "C4_latency": sub(1, "Adoption latency + funnel figure", "analysis"),
            }},
            "D": {"status": "pending", "title": "Reverse provenance of deployed kernels", "subtasks": {
                "D1_provenance": sub(40, ">=40 kernel/backend families with verified provenance", "families"),
            }},
            "R": {"status": "pending", "title": "Reports, synthesis, consistency", "subtasks": {
                "R1_reports": sub(7, "METHODS + 4 reports + SYNTHESIS + STATUS", "files"),
                "R2_consistency": sub(1, "Every numerical executive claim linked to CSV row", "check"),
            }},
        },
        "api": {"cursors": {}, "cache_paths": {
            "compact_metadata": "experiments/deep-study/data/cache/{repo}/meta/",
            "graphql_enrichment": "experiments/deep-study/data/cache/{repo}/enrich/batch-*.json",
            "detailed_timelines": "experiments/deep-study/data/cache/{repo}/detail/batch-*.json",
            "revert_context": "experiments/deep-study/data/cache/{repo}/reverts/",
            "papers": "experiments/deep-study/data/cache/papers/",
        }},
        "failures": [],
        "outputs": {},
        "next_action": "Run: python experiments/deep-study/scripts/a1_population.py compact",
        "resume_instructions": (
            "Read this file; run `python experiments/deep-study/scripts/status.py verify`; then run the "
            "command in next_action. Every collector reuses cached batches. Subagent coding batches live in "
            "experiments/deep-study/coding/batches/<task>/; a batch is complete when its *-output.jsonl passes "
            "`python experiments/deep-study/scripts/batches.py check <task>`."
        ),
        "last_sanity_check": None,
    }
    save_status(data)
    print("initialized", STATUS)


def verify() -> None:
    data = load_status()
    report = {"at_utc": now_utc(), "missing": [], "present": 0}
    for path, meta in list(data.get("outputs", {}).items()):
        full = ROOT / path
        if full.exists():
            report["present"] += 1
            meta["exists"] = True
            meta["bytes"] = full.stat().st_size if full.is_file() else None
            if full.is_file() and full.stat().st_size < 50_000_000:
                meta["sha256_12"] = hashlib.sha256(full.read_bytes()).hexdigest()[:12]
        else:
            meta["exists"] = False
            report["missing"].append(path)
    data["last_sanity_check"] = report
    save_status(data)
    print(report)


def register(paths: list[str], note: str = "") -> None:
    data = load_status()
    for p in paths:
        full = (ROOT / p) if not p.startswith(str(ROOT)) else p
        key = rel(full)
        data.setdefault("outputs", {})[key] = {"note": note, "registered_at_utc": now_utc(),
                                              "exists": (ROOT / key).exists()}
    save_status(data)


def show() -> None:
    data = load_status()
    for pkg, node in data["work_packages"].items():
        print(f"{pkg} [{node['status']}] {node['title']}")
        for name, s in node["subtasks"].items():
            print(f"   {name:18s} {s['status']:11s} {s['completed']}/{s['required']} {s.get('next','')[:90]}")
    print("next:", data.get("next_action"))
    print("failures:", len(data.get("failures", [])))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "init":
        init()
    elif cmd == "verify":
        verify()
    elif cmd == "register":
        register(sys.argv[2:])
    else:
        show()
