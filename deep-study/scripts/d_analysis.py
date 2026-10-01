"""D: reverse provenance merge.

agree  - pass1 (data/cache/d/provenance-<repo>.csv) vs independent pass2 (d_second_pass);
         reconciliation batches for disagreements (task d_reconcile)
final  - data/production-kernel-provenance.csv (+ agreement csv)
"""

from __future__ import annotations

import collections
import json
import sys

from b_analysis import kappa
from batches import load_outputs
from common import BATCHES, CACHE, DATA, REPOS, atomic_csv, read_csv, rng, set_subtask, write_jsonl


def pass1() -> dict[str, dict]:
    out = {}
    for repo in REPOS:
        for i, r in enumerate(read_csv(CACHE / "d" / f"provenance-{repo}.csv")):
            out[f"{repo}:{i:03d}"] = r
    return out


def agree() -> None:
    p1, p2 = pass1(), load_outputs("d_second_pass")
    ids = sorted(p2)
    pairs = [(p1[i]["provenance_class"], p2[i]["provenance_class"]) for i in ids]
    academic = [(a == "academic_paper", b == "academic_paper") for a, b in pairs]
    rows = [{"field": "provenance_class", "n": len(pairs), "percent_agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4),
             "cohen_kappa": round(kappa(pairs), 4), "comparison": "first pass vs independent second pass (model-model; not human IRR)"},
            {"field": "academic_paper_vs_other", "n": len(academic), "percent_agreement": round(sum(a == b for a, b in academic) / len(academic), 4),
             "cohen_kappa": round(kappa([(str(a), str(b)) for a, b in academic]), 4), "comparison": "binary collapse"}]
    atomic_csv(DATA / "provenance-agreement.csv", rows)
    print(rows)
    print(collections.Counter(pairs).most_common(12))
    dis = [i for i in ids if p1[i]["provenance_class"] != p2[i]["provenance_class"]]
    items = []
    for i in dis:
        a, b = p1[i], p2[i]
        items.append({"id": i, "repo": a["repo"], "operator_class": a["operator_class"], "family": a["family"],
                      "implementation_name": a["implementation_name"], "source_paths": a["source_paths"],
                      "first_intro_ref": a["first_intro_ref"],
                      "coder_A": {k: a[k] for k in ["provenance_class", "upstream_or_vendored_source", "cited_paper_or_system", "provenance_evidence", "evidence_urls", "notes"]},
                      "coder_B": {k: b[k] for k in ["provenance_class", "upstream_or_vendored_source", "cited_paper_or_system", "provenance_evidence", "evidence_urls"]}})
    rng("drec").shuffle(items)
    out = BATCHES / "d_reconcile"
    out.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(items), 40):
        write_jsonl(out / f"batch-{k // 40:03d}-input.jsonl", items[k:k + 40])
    (out / "schema.json").write_text(json.dumps({
        "provenance_class": ["academic_paper", "vendor_library", "company_engineering", "community_contribution", "unclear"],
        "cited_paper_or_system": None, "provenance_evidence": None, "evidence_urls": None, "adjudication_rationale": None}))
    print("disagreements", len(dis))


def final() -> None:
    p1, p2, rec = pass1(), load_outputs("d_second_pass"), load_outputs("d_reconcile")
    rows = []
    for i, a in p1.items():
        b = p2.get(i, {})
        r = dict(a)
        r["record_id"] = i
        r["provenance_class_pass1"] = a["provenance_class"]
        r["provenance_class_pass2"] = b.get("provenance_class", "")
        if i in rec:
            r["provenance_class"] = rec[i]["provenance_class"]
            r["cited_paper_or_system"] = rec[i].get("cited_paper_or_system") or a["cited_paper_or_system"]
            r["provenance_evidence"] = rec[i].get("provenance_evidence") or a["provenance_evidence"]
            r["evidence_urls"] = rec[i].get("evidence_urls") or a["evidence_urls"]
            r["label_source"] = "reconciled"
            r["adjudication_rationale"] = rec[i].get("adjudication_rationale", "")
        else:
            r["label_source"] = "agreed" if b else "first_pass_only"
            r["adjudication_rationale"] = ""
        rows.append(r)
    fields = ["record_id", "repo", "operator_class", "family", "implementation_name", "source_paths", "supported_hardware",
              "upstream_or_vendored_source", "cited_paper_or_system", "provenance_class", "provenance_evidence", "evidence_urls",
              "first_intro_ref", "first_intro_date", "status", "status_evidence", "confidence", "label_source",
              "provenance_class_pass1", "provenance_class_pass2", "adjudication_rationale", "notes"]
    atomic_csv(DATA / "production-kernel-provenance.csv", rows, fields)
    c = collections.Counter(r["provenance_class"] for r in rows)
    print(len(rows), c)
    set_subtask("D", "D1_provenance", status="complete", completed=len(rows), required=40,
                outputs=["experiments/deep-study/data/production-kernel-provenance.csv", "experiments/deep-study/data/provenance-agreement.csv"],
                notes=f"{len(rows)} implementation records (vLLM+SGLang); two independent passes + reconciliation.", next="done")


if __name__ == "__main__":
    {"agree": agree, "final": final}[sys.argv[1]]()
