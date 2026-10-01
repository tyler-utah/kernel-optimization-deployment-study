"""B2/B3 merge: first pass + independent second pass, agreement, reconciliation,
final kernel-correctness case table and B figures.

agree      - agreement pass1 vs pass2 -> data/failure-coding-agreement.csv ; writes
             reconciliation batches (task b3_reconcile) for every disagreement
final      - final labels (agreed or reconciled) -> data/kernel-correctness-cases.csv
"""

from __future__ import annotations

import collections
import json
import sys

from batches import load_outputs
from b2_failures import D, load_ctx
from common import BATCHES, DATA, REPOS, atomic_csv, iso, read_jsonl, rng, set_subtask, write_jsonl

FIELDS = ["case_status", "failure_class", "hardware_specific", "counterfactual_primary"]


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs)
    cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def inputs(task: str) -> dict[str, dict]:
    out = {}
    for p in sorted((BATCHES / task).glob("batch-*-input.jsonl")):
        for r in read_jsonl(p):
            out[r["id"]] = r
    return out


def agree() -> None:
    p1 = load_outputs("b2_failures")
    p2 = load_outputs("b3_second_pass")
    ids = sorted(p2)
    rows = []
    for f in FIELDS:
        pairs = [(p1[i][f], p2[i][f]) for i in ids]
        if f != "case_status":  # agreement on content fields among cases both call confirmed
            pairs = [(p1[i][f], p2[i][f]) for i in ids if p2[i]["case_status"] == "confirmed_kernel_correctness"]
        rows.append({"field": f, "n": len(pairs), "percent_agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4),
                     "cohen_kappa": round(kappa(pairs), 4), "comparison": "first pass vs independent second pass (model-model; not human IRR)"})
    both = [i for i in ids if p2[i]["case_status"] == "confirmed_kernel_correctness"]
    overlap = [len(set(p1[i]["counterfactual_all"]) & set(p2[i]["counterfactual_all"])) > 0 for i in both]
    rows.append({"field": "counterfactual_all_overlap", "n": len(both), "percent_agreement": round(sum(overlap) / len(both), 4),
                 "cohen_kappa": "", "comparison": "share of cases where the two plausible-technique sets intersect"})
    atomic_csv(DATA / "failure-coding-agreement.csv", rows)
    for r in rows:
        print(r)
    dis = [i for i in ids if any(p1[i][f] != p2[i][f] for f in FIELDS)]
    inp = inputs("b2_failures")
    items = []
    for i in dis:
        d = dict(inp[i])
        d["coder_A"] = {k: p1[i][k] for k in FIELDS + ["counterfactual_all", "rationale", "counterfactual_rationale"]}
        d["coder_B"] = {k: p2[i][k] for k in FIELDS + ["counterfactual_all", "counterfactual_rationale"]}
        items.append(d)
    rng("b3rec").shuffle(items)
    out = BATCHES / "b3_reconcile"
    out.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(items), 30):
        write_jsonl(out / f"batch-{k // 30:03d}-input.jsonl", items[k:k + 30])
    schema = {"case_status": ["confirmed_kernel_correctness", "not_kernel_correctness", "insufficient_information"],
              "failure_class": ["numerical_precision", "nondeterminism_race_sync", "warp_subgroup_mask", "shape_alignment_edge",
                                "memory_safety_oob", "hardware_compiler_specific", "integration_backend_cudagraph",
                                "perf_regression_as_correctness", "other", "na"],
              "hardware_specific": ["yes", "no", "unknown", "na"],
              "counterfactual_primary": ["randomized_edge_tests", "hidden_input_distributions", "sanitizer_memcheck",
                                         "determinism_schedule_perturbation", "formal_equivalence", "warp_race_static_analysis",
                                         "e2e_serving_tests_only", "hardware_matrix_ci", "not_enough_information", "na"],
              "counterfactual_all": None, "adjudication_rationale": None}
    (out / "schema.json").write_text(json.dumps(schema))
    print("disagreements", len(dis), "of", len(ids), "-> b3_reconcile batches", (len(items) + 29) // 30)


def final() -> None:
    p1 = load_outputs("b2_failures")
    p2 = load_outputs("b3_second_pass")
    rec = load_outputs("b3_reconcile")
    intro = load_outputs("b2_intro")
    inp = inputs("b2_failures")
    frames = {r: {x["sha"][:10]: x for x in read_jsonl(D / f"frame-{r}.jsonl")} for r in REPOS}
    ictx = {r: load_ctx(r, "intro") for r in REPOS}
    rows = []
    for cid, a in p1.items():
        d = inp[cid]
        b = p2.get(cid)
        final_vals = {}
        for f in FIELDS:
            if b is None:
                final_vals[f] = a[f]
            elif cid in rec:
                final_vals[f] = rec[cid][f]
            else:
                final_vals[f] = a[f]
        cf_all = rec[cid]["counterfactual_all"] if cid in rec else a["counterfactual_all"]
        src = "first_pass_only_not_confirmed" if b is None else ("reconciled" if cid in rec else "agreed")
        repo = d["repo"]
        # introducing PR evidence and time-to-fix
        ev = [v for k, v in intro.items() if k.startswith(cid + "|")]
        intro_prs = [int(k.split("|")[1]) for k in intro if k.startswith(cid + "|")]
        merged = [iso(ictx[repo][n]["mergedAt"]) for n in intro_prs if n in ictx[repo] and ictx[repo][n].get("mergedAt")]
        fix_date = iso(d["merged"])
        ttf = (fix_date - min(merged)).total_seconds() / 86400 if merged else None
        rows.append({
            "case_id": cid, "repo": repo, "fix_pr": d["pr"] or "", "fix_sha": d["sha"], "fix_date": d["merged"], "url": d["url"],
            "subject": d["subject"], "frame_rank": frames[repo][d["sha"][:10]]["frame_rank"],
            "case_status": final_vals["case_status"], "label_source": src,
            "failure_class": final_vals["failure_class"], "secondary_classes": ";".join(a.get("secondary_classes") or []),
            "symptom": a["symptom"], "affected_hardware": a["affected_hardware"], "affected_kernel": a["affected_kernel"],
            "hardware_specific": final_vals["hardware_specific"],
            "introducing_ref": a["introducing_ref"], "introducing_evidence": a["introducing_evidence"],
            "introducing_pr_merged": min(merged).isoformat() if merged else "",
            "days_intro_to_fix": round(ttf, 2) if ttf is not None and ttf >= 0 else "",
            "introducing_pr_is_performance": ev[0]["introducing_pr_is_performance"] if ev else "",
            "introducing_pr_reported_correctness_tests": ev[0]["reported_correctness_tests"] if ev else "",
            "introducing_pr_reported_accuracy_eval": ev[0]["reported_accuracy_eval"] if ev else "",
            "introducing_pr_reported_multi_hardware": ev[0]["reported_multi_hardware"] if ev else "",
            "regression_test_added": a["regression_test_added"], "detected_by": a["detected_by"],
            "escaped_ci": a["escaped_ci"], "consequence": a["consequence"],
            "counterfactual_primary": final_vals["counterfactual_primary"], "counterfactual_all": ";".join(cf_all or []),
            "counterfactual_primary_pass1": a["counterfactual_primary"], "counterfactual_primary_pass2": b["counterfactual_primary"] if b else "",
            "failure_class_pass1": a["failure_class"], "failure_class_pass2": b["failure_class"] if b else "",
            "evidence_snippet": a["evidence_snippet"], "rationale": a["rationale"],
            "counterfactual_rationale": (rec[cid]["adjudication_rationale"] if cid in rec else a["counterfactual_rationale"]),
            "confidence": a["confidence"],
        })
    rows.sort(key=lambda r: (r["repo"], r["frame_rank"]))
    atomic_csv(DATA / "kernel-correctness-cases.csv", rows)
    conf = [r for r in rows if r["case_status"] == "confirmed_kernel_correctness"]
    print("cases", len(rows), "final confirmed", len(conf), collections.Counter(r["repo"] for r in conf))
    set_subtask("B", "B2_corpus", status="complete", completed=len(conf), required=150,
                outputs=["experiments/deep-study/data/kernel-correctness-cases.csv"],
                notes=f"{len(rows)} sampled fix commits screened (ranks 0-219 per repo of 2,354-commit frame); {len(conf)} confirmed after second pass + reconciliation.",
                next="done")
    set_subtask("B", "B3_counterfactual", status="complete", completed=len(conf), required=len(conf),
                outputs=["experiments/deep-study/data/kernel-correctness-cases.csv", "experiments/deep-study/data/failure-coding-agreement.csv"],
                next="done")


if __name__ == "__main__":
    {"agree": agree, "final": final}[sys.argv[1]]()
