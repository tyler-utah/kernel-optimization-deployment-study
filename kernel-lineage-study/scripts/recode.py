"""WP-I1: reproducible stratified recode sample, comparison, reconciliation.

  python recode.py sample      -> recoding/sample-input.{md,jsonl}, recoding/sample-key.json (first-pass, hidden)
  python recode.py compare     -> recoding/agreement.csv, recoding/disagreements.jsonl
  python recode.py finalize    -> recoding/reconciliation-summary.csv (after reconciliation.jsonl exists)

Sample: seed 20260928; per lineage 12 verified events, stratified 6 key-type
(introduce/optimize/port/replace/change_default/remove/revert/reland/integrate/deprecate)
and 6 other types, drawn uniformly within stratum. Total 36 (>= 30, >= 10 per lineage).
"""
import csv
import json
import os
import random
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402
from batches import dossier, load_registries, corpus_flags, render_md  # noqa: E402

csv.field_size_limit(10 ** 9)
REC = os.path.join(S.STUDY, "recoding")
DATA = os.path.join(S.STUDY, "data")
SEED = 20260928
KEY = {"introduce", "optimize", "port", "replace", "change_default", "remove", "revert", "reland", "integrate", "deprecate"}


def rcsv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample():
    os.makedirs(REC, exist_ok=True)
    events = [e for e in rcsv(os.path.join(DATA, "lineage-events.csv")) if e["candidate_id"]]
    rng = random.Random(SEED)
    chosen = []
    for L in ("L1", "L2", "L3"):
        ev = sorted([e for e in events if e["lineage"] == L], key=lambda e: e["event_id"])
        key = [e for e in ev if e["event_type"] in KEY]
        oth = [e for e in ev if e["event_type"] not in KEY]
        chosen += rng.sample(key, 6) + rng.sample(oth, 6)
    cands = {json.loads(l)["candidate_id"]: json.loads(l) for l in open(os.path.join(S.STUDY, "staging", "candidates.jsonl"), encoding="utf-8")}
    regs, flags = load_registries(), corpus_flags()
    recs = []
    for e in chosen:
        d = dossier(cands[e["candidate_id"]], regs, flags, "code")
        d["id"] = e["event_id"]
        d.pop("artifact_hints", None)
        d["artifact_hints"] = []
        d["deep_study"] = []  # hide deep-study labels too (they could cue event type)
        recs.append(d)
    with open(os.path.join(REC, "sample-input.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(REC, "sample-input.md"), "w", encoding="utf-8") as f:
        f.write(render_md(recs, "code"))
    key = {e["event_id"]: {k: e[k] for k in ("event_type", "primary_cause", "parent_event_ids", "optimization_move_ids",
                                             "artifact_ids", "lineage", "pr_number", "commit_sha", "repo")} for e in chosen}
    json.dump(key, open(os.path.join(REC, "sample-key.json"), "w", encoding="utf-8"), indent=1)
    print("sampled", len(recs), Counter(e["lineage"] for e in chosen), Counter(e["event_type"] in KEY for e in chosen))


def kappa(pairs):
    n = len(pairs)
    if not n:
        return ""
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def compare():
    key = json.load(open(os.path.join(REC, "sample-key.json"), encoding="utf-8"))
    out = {json.loads(l)["event_id"]: json.loads(l) for l in open(os.path.join(REC, "sample-output.jsonl"), encoding="utf-8") if l.strip()}
    events = {e["event_id"]: e for e in rcsv(os.path.join(DATA, "lineage-events.csv"))}
    edges = rcsv(os.path.join(DATA, "lineage-edges.csv"))
    ext_fn = os.path.join(DATA, "external-references.csv")
    ext = [x for x in rcsv(ext_fn) if x["relation"] not in ("integrates", "wraps")] if os.path.exists(ext_fn) else []
    rows, dis = [], []
    fields = {"event_type": [], "primary_cause": [], "ancestry": [], "moves": [], "artifacts": []}
    for eid, k in key.items():
        o = out.get(eid)
        if not o:
            continue
        fields["event_type"].append((k["event_type"], o.get("event_type")))
        fields["primary_cause"].append((k["primary_cause"], o.get("primary_cause")))
        # ancestry agreement: compare explicit (non-textual) predecessors; if neither coder names one, agree (textual only)
        anc = o.get("ancestry") or {}
        tgt = (anc.get("target") or "").strip()
        parents = [p for p in k["parent_event_ids"].split(";") if p]
        explicit = [x for x in edges if x["to_event_id"] == eid and x["edge_type"] != "textual_successor"]
        exp_prs = {events[x["from_event_id"]]["pr_number"] for x in explicit if x["from_event_id"] in events}
        exp_shas = {events[x["from_event_id"]]["commit_sha"] for x in explicit if x["from_event_id"] in events}
        ext_targets = [x for x in ext if x["event_id"] == eid]
        exp_prs |= {m.group(1) for x in ext_targets for m in [re.search(r"(\d{2,6})", x["target"])] if m}
        rec_textual = anc.get("edge_type") in ("textual_successor", "none", "") or tgt in ("", "previous_change_on_artifact")
        m = re.search(r"(\d{2,6})", tgt)
        if rec_textual:
            ok = "agree_textual" if not explicit and not ext_targets else "disagree_missed_explicit"
        elif (m and m.group(1) in exp_prs) or (len(tgt) >= 7 and any(s.startswith(tgt[:7]) for s in exp_shas)):
            ok = "agree_explicit"
        elif not explicit and not ext_targets:
            ok = "disagree_extra_explicit"
        else:
            ok = "disagree_different_target"
        fields["ancestry"].append((ok, "agree" if ok.startswith("agree") else "disagree"))
        m1 = set(x for x in k["optimization_move_ids"].split(";") if x)
        m2 = set(o.get("move_ids") or [])
        mv = "agree" if m1 == m2 else ("partial" if m1 & m2 else "disagree")
        fields["moves"].append((mv, mv))
        a1 = set(x for x in k["artifact_ids"].split(";") if x)
        a2 = set(o.get("artifact_ids") or [])
        av = "agree" if a1 == a2 else ("partial" if a1 & a2 else "disagree")
        fields["artifacts"].append((av, av))
        row = {"event_id": eid, "lineage": k["lineage"], "type_1": k["event_type"], "type_2": o.get("event_type"),
               "cause_1": k["primary_cause"], "cause_2": o.get("primary_cause"), "ancestry": ok,
               "ancestry_2": json.dumps(anc), "parents_1": k["parent_event_ids"], "moves_1": ";".join(sorted(m1)),
               "moves_2": ";".join(sorted(m2)), "moves": mv, "artifacts": av}
        rows.append(row)
        if row["type_1"] != row["type_2"] or row["cause_1"] != row["cause_2"] or not ok.startswith("agree") or mv == "disagree":
            dis.append(row)
    with open(os.path.join(REC, "recode-pairs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    agg = []
    n = len(rows)
    for fld in ("event_type", "primary_cause"):
        pairs = fields[fld]
        agg.append({"field": fld, "n": n, "percent_agreement": round(sum(1 for a, b in pairs if a == b) / n, 4), "cohen_kappa": kappa(pairs)})
    for fld in ("ancestry", "moves", "artifacts"):
        vals = [a for a, _ in fields[fld]]
        agg.append({"field": fld, "n": n, "percent_agreement": round(sum(1 for v in vals if v.startswith("agree")) / n, 4),
                    "cohen_kappa": "", "partial": sum(1 for v in vals if v == "partial")})
    for L in ("L1", "L2", "L3"):
        sub = [r for r in rows if r["lineage"] == L]
        agg.append({"field": f"event_type[{L}]", "n": len(sub), "percent_agreement": round(sum(1 for r in sub if r["type_1"] == r["type_2"]) / max(1, len(sub)), 4),
                    "cohen_kappa": kappa([(r["type_1"], r["type_2"]) for r in sub])})
    with open(os.path.join(REC, "agreement.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["field", "n", "percent_agreement", "cohen_kappa", "partial"])
        w.writeheader()
        w.writerows(agg)
    with open(os.path.join(REC, "disagreements.jsonl"), "w", encoding="utf-8") as f:
        for r in dis:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("compared", n, "disagreements", len(dis))
    for a in agg:
        print(a)


def apply():
    """Turn adjudicated final values that differ from first pass into staging/code-overrides.jsonl."""
    key = json.load(open(os.path.join(REC, "sample-key.json"), encoding="utf-8"))
    rec = [json.loads(l) for l in open(os.path.join(REC, "reconciliation.jsonl"), encoding="utf-8") if l.strip()]
    import glob
    first = {}
    for fn in glob.glob(os.path.join(S.STUDY, "staging", "code", "L*", "batch-*-output.jsonl")):
        for line in open(fn, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                first[r["candidate_id"]] = r
    events = {e["event_id"]: e for e in rcsv(os.path.join(DATA, "lineage-events.csv"))}
    ov, rows = [], []
    for x in rec:
        eid = x["event_id"]
        cid = x.get("candidate_id") or events.get(eid, {}).get("candidate_id")
        if not cid or cid not in first:
            continue
        k = key.get(eid, {})
        o = {"candidate_id": cid}
        if "event_type" in x and x["event_type"].get("final") and x["event_type"]["final"] != k.get("event_type"):
            o["event_type"] = x["event_type"]["final"]
        if "primary_cause" in x and x["primary_cause"].get("final") and x["primary_cause"]["final"] != k.get("primary_cause"):
            o["primary_cause"] = x["primary_cause"]["final"]
        if "move_ids" in x and x["move_ids"].get("winner") in ("second", "neither"):
            o["moves"] = [{"move_id": m, "name": "", "category": "", "mechanism": "adjudicated in I1 reconciliation",
                           "source": "reconciliation", "status": "reconciled"} for m in x["move_ids"].get("final") or []]
        anc = x.get("ancestry") or {}
        if anc.get("winner") in ("second", "neither") and anc.get("final_target") not in (None, "", "textual_only"):
            rels = list(first[cid].get("relations") or [])
            rels.append({"target": anc["final_target"], "repo": anc.get("final_repo") or events.get(eid, {}).get("repo", ""),
                         "relation": anc.get("final_edge_type") or "historical_connection_uncertain",
                         "evidence": ("I1 reconciliation: " + (anc.get("rationale") or ""))[:280]})
            o["relations"] = rels
        if len(o) > 1:
            o["notes"] = (first[cid].get("notes") or "") + " [I1 reconciliation override]"
            ov.append(o)
        rows.append({"event_id": eid, "fields_overridden": ";".join(k2 for k2 in o if k2 not in ("candidate_id", "notes"))})
    with open(os.path.join(S.STUDY, "staging", "code-overrides.jsonl"), "w", encoding="utf-8") as f:
        for o in ov:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    summ = Counter()
    for x in rec:
        for fld in ("event_type", "primary_cause", "ancestry", "move_ids"):
            if fld in x:
                summ[(fld, x[fld].get("winner"))] += 1
    with open(os.path.join(REC, "reconciliation-summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["field", "winner", "n"])
        w.writeheader()
        for (fld, win), n in sorted(summ.items()):
            w.writerow({"field": fld, "winner": win, "n": n})
    print("overrides", len(ov), dict(summ))


if __name__ == "__main__":
    {"sample": sample, "compare": compare, "apply": apply}[sys.argv[1]]()
