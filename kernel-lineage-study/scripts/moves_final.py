"""WP-E outputs from the final move catalogs (+ assumption audits).

  python moves_final.py namemap   -> staging/move-name-map.json
  python moves_final.py csv       -> data/optimization-moves.csv, data/move-assumptions.csv, data/move-biographies.csv
"""
import csv
import glob
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402

csv.field_size_limit(10 ** 9)
DATA = os.path.join(S.STUDY, "data")
MOVES = os.path.join(S.STUDY, "staging", "moves")


def finals():
    out = {}
    for L in ("L1", "L2", "L3"):
        fn = os.path.join(MOVES, f"{L}-moves-final.json")
        out[L] = json.load(open(fn, encoding="utf-8"))
    return out


def namemap():
    mp = {}
    for L, d in finals().items():
        for k, v in (d.get("name_map") or {}).items():
            if v and v != "one_off":
                mp[k] = v
    json.dump(mp, open(os.path.join(S.STUDY, "staging", "move-name-map.json"), "w", encoding="utf-8"), indent=1)
    print("name map entries", len(mp))


def rcsv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def w(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", os.path.basename(path), len(rows))


def to_csv():
    events = rcsv(os.path.join(DATA, "lineage-events.csv"))
    ev_ids = {e["event_id"] for e in events}
    by_pr = {(e["repo"], e["pr_number"]): e["event_id"] for e in events if e["pr_number"]}
    by_sha = {e["commit_sha"][:10]: e["event_id"] for e in events}
    mrows, arows, brows = [], [], []
    for L, d in finals().items():
        audit_fn = os.path.join(MOVES, f"{L}-assumptions-audit.json")
        audit = json.load(open(audit_fn, encoding="utf-8"))["assumptions"] if os.path.exists(audit_fn) else []
        uses = Counter()
        for e in events:
            for m in e["optimization_move_ids"].split(";"):
                if m:
                    uses[m] += 1
        for m in d["moves"]:
            fo = m.get("first_observed") or {}
            fo_ev = by_pr.get((fo.get("repo", ""), str(fo.get("pr") or ""))) or by_sha.get((fo.get("sha") or "")[:10], "")
            if not fo_ev:
                fo_ev = f"external:{fo.get('repo','')}#{fo.get('pr') or (fo.get('sha') or '')[:10]}"
            coded = sorted((e["date"], e["event_id"]) for e in events if m["move_id"] in e["optimization_move_ids"].split(";"))
            cat_date = fo.get("date") or ""
            earliest_coded = coded[0] if coded else None
            if earliest_coded and (not cat_date or earliest_coded[0][:10] < cat_date[:10]):
                first_event, first_basis = earliest_coded[1], "earliest_coded_event"
            else:
                first_event, first_basis = fo_ev, "catalog_first_observed"
            later = [x for x in (m.get("later_uses") or []) if x in ev_ids]
            fails = m.get("known_failure_refs") or [f.get("ref") for f in m.get("known_failures") or []]
            asm = [a for a in audit if a.get("move_id") == m["move_id"]]
            src = asm if asm else m.get("assumptions") or []
            st = Counter(a.get("status") for a in src)
            mrows.append({
                "move_id": m["move_id"], "name": m.get("name", ""), "category": m.get("category", ""),
                "first_observed_event": first_event, "description": (m.get("description") or "")[:600],
                "semantic_preconditions": " | ".join(m.get("semantic_preconditions") or [])[:600],
                "hardware_preconditions": " | ".join(m.get("hardware_preconditions") or [])[:600],
                "framework_preconditions": " | ".join(m.get("framework_preconditions") or [])[:600],
                "assumption_status": "; ".join(f"{k}={v}" for k, v in sorted(st.items())) + (" (spec-checked audit)" if asm else " (catalog)"),
                "performance_evidence": (m.get("performance_evidence") or "")[:400],
                "correctness_evidence": (m.get("correctness_evidence") or "")[:400],
                "later_uses": ";".join(later), "known_failures": ";".join(str(x) for x in fails if x),
                "current_status": m.get("current_status", ""), "confidence": m.get("confidence", ""),
                "evidence_urls": " ".join(m.get("evidence_urls") or []),
                # extras
                "lineage": L, "first_observed_basis": first_basis, "catalog_first_observed_event": fo_ev,
                "first_observed_repo": fo.get("repo", ""), "first_observed_date": fo.get("date", ""),
                "earliest_coded_event_date": earliest_coded[0][:10] if earliest_coded else "",
                "first_observed_pr": fo.get("pr") or "", "n_coded_events": uses.get(m["move_id"], 0),
                "n_occurrences": len(m.get("occurrences") or []),
                "cross_repo_occurrences": sum(1 for o in m.get("occurrences") or [] if o.get("repo") and o.get("repo") != fo.get("repo")),
                "unacknowledged_recurrences": sum(1 for o in m.get("occurrences") or [] if (o.get("acknowledges_prior") or "").startswith("no")
                                                  and o.get("relation") in ("reimplemented", "rediscovered", "ported")),
                "has_biography": int(bool(m.get("biography"))),
            })
            for a in src:
                arows.append({"lineage": L, "move_id": m["move_id"], "assumption": a.get("assumption", ""),
                              "kind": a.get("kind", ""), "load_bearing_for": a.get("load_bearing_for", ""),
                              "platform": a.get("platform", ""), "status": a.get("status", ""),
                              "code_evidence": a.get("code_evidence") or a.get("source", ""),
                              "spec_checked": a.get("spec_checked", ""), "excerpt": a.get("spec_excerpt") or a.get("excerpt", ""),
                              "notes": (a.get("notes") or "")[:300], "source_file": "audit" if asm else "catalog"})
            b = m.get("biography")
            if b:
                brows.append({"lineage": L, "move_id": m["move_id"], "first_appearance": b.get("first_appearance", ""),
                              "problem": b.get("problem", ""), "preconditions": b.get("preconditions", ""),
                              "survival": b.get("survival", ""), "recurrence": b.get("recurrence", ""),
                              "failures": b.get("failures", ""), "acknowledgement": b.get("acknowledgement", ""),
                              "event_ids": ";".join(x for x in b.get("event_ids", [])),
                              "invalid_event_ids": ";".join(x for x in b.get("event_ids", []) if x not in ev_ids),
                              "evidence_types": ";".join(m.get("evidence_types") or []),
                              "evidence_urls": " ".join(b.get("evidence_urls") or []), "confidence": b.get("confidence", "")})
    w(os.path.join(DATA, "optimization-moves.csv"), mrows)
    w(os.path.join(DATA, "move-assumptions.csv"), arows)
    w(os.path.join(DATA, "move-biographies.csv"), brows)


if __name__ == "__main__":
    {"namemap": namemap, "csv": to_csv}[sys.argv[1]]()
