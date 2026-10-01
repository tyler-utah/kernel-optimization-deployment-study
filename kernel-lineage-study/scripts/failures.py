"""WP-G: link deep-study failure and revert records to lineage events.

  python failures.py links     -> data/failure-links.csv
"""
import csv
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402

csv.field_size_limit(10 ** 9)
DEEP = os.path.join(os.path.dirname(S.STUDY), "deep-study", "data")
DATA = os.path.join(S.STUDY, "data")
SLUG = {"vllm": "vllm-project/vllm", "sglang": "sgl-project/sglang", "vllm-project/vllm": "vllm-project/vllm",
        "sgl-project/sglang": "sgl-project/sglang"}


def rcsv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prnum(s):
    m = re.search(r"#?(\d{2,6})", s or "")
    return int(m.group(1)) if m else None


def links():
    events = rcsv(os.path.join(DATA, "lineage-events.csv"))
    by_pr = defaultdict(list)
    for e in events:
        if e["pr_number"]:
            by_pr[(e["repo"], int(e["pr_number"]))].append(e)
    rows = []
    for c in rcsv(os.path.join(DEEP, "kernel-correctness-cases.csv")):
        if c["case_status"] != "confirmed_kernel_correctness":
            continue
        slug = SLUG.get(c["repo"])
        fix = by_pr.get((slug, prnum(c["fix_pr"])), []) if prnum(c["fix_pr"]) else []
        intro = by_pr.get((slug, prnum(c["introducing_ref"])), []) if prnum(c["introducing_ref"]) else []
        for e in fix or [None]:
            if e is None and not intro:
                continue
            rows.append({"source": "kernel-correctness-cases", "record_id": c["case_id"], "repo": slug,
                         "lineage": (e or intro[0])["lineage"], "fix_or_revert_pr": c["fix_pr"],
                         "fix_event_id": e["event_id"] if e else "", "introducing_ref": c["introducing_ref"],
                         "introducing_event_id": ";".join(x["event_id"] for x in intro if not e or x["lineage"] == e["lineage"]),
                         "failure_class": c["failure_class"], "symptom": c["symptom"][:200], "affected_kernel": c["affected_kernel"],
                         "hardware_specific": c["hardware_specific"], "regression_test_added": c["regression_test_added"],
                         "detected_by": c["detected_by"], "consequence": c["consequence"],
                         "days_intro_to_fix": c["days_intro_to_fix"], "url": c["url"]})
    for r in rcsv(os.path.join(DEEP, "confirmed-reverts.csv")):
        if r["revert_status"] == "not_revert":
            continue
        slug = SLUG.get(r["repo"])
        rev = by_pr.get((slug, prnum(r["revert_pr"])), []) if prnum(r["revert_pr"]) else []
        tg = []
        for t in re.findall(r"\d+", r["reverted_prs"] or ""):
            tg += by_pr.get((slug, int(t)), [])
        if not rev and not tg:
            continue
        for e in rev or [None]:
            L = (e or tg[0])["lineage"]
            rows.append({"source": "confirmed-reverts", "record_id": r["revert_sha"][:12], "repo": slug, "lineage": L,
                         "fix_or_revert_pr": r["revert_pr"], "fix_event_id": e["event_id"] if e else "",
                         "introducing_ref": r["reverted_prs"],
                         "introducing_event_id": ";".join(x["event_id"] for x in tg if x["lineage"] == L),
                         "failure_class": r["revert_reason"], "symptom": (r["reason_evidence"] or "")[:200],
                         "affected_kernel": r["reverted_title"][:100], "hardware_specific": r["hardware_specific"],
                         "regression_test_added": "", "detected_by": "", "consequence": r["revert_status"],
                         "days_intro_to_fix": r["days_to_revert"], "url": r["url"]})
    out = os.path.join(DATA, "failure-links.csv")
    cols = list(rows[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print("failure links", len(rows), Counter((r["lineage"], r["source"]) for r in rows))


if __name__ == "__main__":
    links()
