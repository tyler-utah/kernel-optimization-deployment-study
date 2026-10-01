"""Validate subagent coding batches against per-task schemas.

    python experiments/deep-study/scripts/batches.py check <task> [--quiet]
    python experiments/deep-study/scripts/batches.py status

A batch is complete when ``batch-NNN-output.jsonl`` contains exactly one valid
object per input id. ``status`` records per-task completion in STATUS.json.
"""

from __future__ import annotations

import json
import sys

from common import BATCHES, now_utc, read_jsonl, update_status

ANY = None
SCHEMAS = {
    "b1_reverts": {
        "revert_status": {"confirmed_revert", "partial_revert", "explicit_rollback", "reland", "not_revert"},
        "reverted_prs": ANY, "reverted_title": ANY,
        "reverted_class": {"kernel_performance", "kernel_correctness", "hardware_backend", "other", "na"},
        "reverted_touches_kernel": {"yes", "no", "unknown"},
        "revert_reason": {"correctness_or_accuracy", "crash_or_hang", "performance_regression", "ci_or_test_failure",
                          "build_or_dependency", "hardware_specific_breakage", "api_or_compat_break",
                          "premature_or_process", "unstated", "other", "na"},
        "hardware_specific": {"yes", "no", "unknown"},
        "reason_evidence": ANY, "confidence": {"high", "medium", "low"}, "rationale": ANY,
    },
    "a1_adjudication": {
        "classification": {"confirmed_performance", "not_performance", "uncertain"},
        "perf_type": {"kernel_optimization", "new_kernel_or_fusion", "kernel_tuning_config", "precision_format",
                      "system_performance", "perf_regression_fix", "na"},
        "evidence": ANY, "rationale": ANY, "confidence": {"high", "medium", "low"},
    },
    "c1_census": {
        "kernel_style": {"yes", "no", "ambiguous"},
        "category": {"kernel", "fusion_family", "precision_format", "scheduling_layout", "compiler_dsl",
                     "agentic_generation", "not_kernel_style"},
        "secondary_categories": ANY, "technique_name": ANY, "target_workload": ANY,
        "rationale": ANY, "confidence": {"high", "medium", "low"}, "sections_read": ANY,
    },
}
# second-pass tasks reuse the first-pass schema
SCHEMAS["a1_adjudication_p2"] = SCHEMAS["a1_adjudication"]
SCHEMAS["c1_census_p2"] = SCHEMAS["c1_census"]


def load_schema(task: str):
    if task in SCHEMAS:
        return SCHEMAS[task]
    path = BATCHES / task / "schema.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {k: (set(v) if isinstance(v, list) else ANY) for k, v in raw.items()}
    raise KeyError(task)


def check(task: str, quiet: bool = False) -> dict:
    schema = load_schema(task)
    directory = BATCHES / task
    result = {"task": task, "batches": 0, "complete": 0, "incomplete": [], "records": 0, "errors": []}
    for inp in sorted(directory.glob("batch-*-input.jsonl")):
        result["batches"] += 1
        ids = [r["id"] for r in read_jsonl(inp)]
        out = inp.with_name(inp.name.replace("-input", "-output"))
        if not out.exists():
            result["incomplete"].append(inp.name)
            continue
        try:
            rows = read_jsonl(out)
        except Exception as e:  # noqa: BLE001
            result["incomplete"].append(inp.name)
            result["errors"].append(f"{out.name}: unreadable {e}")
            continue
        got = {}
        errs = []
        for r in rows:
            if r.get("id") not in ids:
                errs.append(f"unknown id {r.get('id')}")
                continue
            for field, allowed in schema.items():
                if field not in r:
                    errs.append(f"{r['id']}: missing {field}")
                elif allowed is not ANY and r[field] not in allowed:
                    errs.append(f"{r['id']}: bad {field}={r[field]!r}")
            got[r["id"]] = r
        missing = [i for i in ids if i not in got]
        if missing:
            errs.append(f"missing ids {missing[:5]}… ({len(missing)})")
        if errs:
            result["incomplete"].append(inp.name)
            result["errors"].extend(f"{out.name}: {e}" for e in errs[:8])
        else:
            result["complete"] += 1
            result["records"] += len(ids)
    if not quiet:
        print(json.dumps(result, indent=1)[:4000])
    return result


def record_status(task: str) -> dict:
    res = check(task, quiet=True)
    value = {k: res[k] for k in ["batches", "complete", "records"]} | {"incomplete": res["incomplete"][:30], "at_utc": now_utc()}
    update_status(lambda d: d.setdefault("coding_batches", {}).__setitem__(task, value))
    return res


def load_outputs(task: str) -> dict[str, dict]:
    out = {}
    for path in sorted((BATCHES / task).glob("batch-*-output.jsonl")):
        for r in read_jsonl(path):
            out[r["id"]] = r
    return out


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(sys.argv[2], quiet="--quiet" in sys.argv)
    elif sys.argv[1] == "status":
        for t in sys.argv[2:]:
            r = record_status(t)
            print(t, r["complete"], "/", r["batches"], "batches;", r["records"], "records")
