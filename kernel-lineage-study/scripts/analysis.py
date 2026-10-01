"""WP-F analysis: release transitions, survival, repairs, causes.

  python analysis.py all

Inputs : data/lineage-events.csv, data/artifact-release-presence.csv, data/move-release-presence.csv,
         staging/registry/*.json, staging/moves/*.json, cache/git/<repo>-release-index.json
Outputs: data/release-transitions.csv, data/artifact-survival.csv, data/move-survival.csv,
         data/km-artifact-lifetime.csv, data/km-first-repair.csv, data/artifact-milestones.csv,
         data/lineage-metrics.csv (tidy: lineage, metric, value, n, denominator, note)
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402

csv.field_size_limit(10 ** 9)
DATA = os.path.join(S.STUDY, "data")
STAGING = os.path.join(S.STUDY, "staging")
LREPO = {"L1": "sglang", "L2": "sglang", "L3": "vllm"}
ORDER = ["direct_carry", "parameter_retune", "adapter_repair", "derivation_repair", "manual_reimplementation",
         "backend_replacement", "obsolete"]
RANK = {c: i for i, c in enumerate(ORDER)}
EV2CLASS = {"retune": "parameter_retune", "adapt_framework": "adapter_repair", "integrate": "adapter_repair",
            "repair_build_dependency": "adapter_repair", "optimize": "derivation_repair",
            "extend_support": "derivation_repair", "repair_correctness": "derivation_repair",
            "repair_performance": "derivation_repair", "replace": "backend_replacement",
            "change_default": "backend_replacement", "remove": "obsolete", "deprecate": "obsolete"}
REPAIR = {"repair_correctness", "repair_performance", "repair_build_dependency"}
MILESTONE_TYPES = {"first_repair": {"repair_correctness", "repair_performance"}, "first_revert": {"revert"},
                   "first_replacement": {"replace", "change_default"}, "first_removal": {"remove"}}


def rcsv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wcsv(path, rows, cols=None):
    cols = cols or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def d(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")[:25]) if "T" in s else datetime.fromisoformat(s + "T00:00:00+00:00")


def releases(repo):
    rel = json.load(open(os.path.join(S.STUDY, "cache", "git", f"{repo}-release-index.json"), encoding="utf-8"))
    finals = sorted([r for r in rel if r["final"]], key=lambda r: r["release_date"])
    return finals


def registry():
    arts = {}
    for L in ("L1", "L2", "L3"):
        for a in json.load(open(os.path.join(STAGING, "registry", f"{L}-registry.json"), encoding="utf-8"))["artifacts"]:
            a["lineage"] = L
            arts[a["artifact_id"]] = a
    return arts


def moves_catalog():
    out = {}
    for L in ("L1", "L2", "L3"):
        fn = os.path.join(STAGING, "moves", f"{L}-moves-final.json")
        if not os.path.exists(fn):
            fn = os.path.join(STAGING, "moves", f"{L}-moves.json")
        if os.path.exists(fn):
            for m in json.load(open(fn, encoding="utf-8"))["moves"]:
                m["lineage"] = L
                out[m["move_id"]] = m
    return out


def km(durations_events):
    """Kaplan-Meier: list of (t, event:1/0). Returns [(t, S(t), n_at_risk, d)] and median."""
    data = sorted(durations_events)
    times = sorted({t for t, e in data if e})
    surv, out, s = [], [], 1.0
    n = len(data)
    for t in times:
        at_risk = sum(1 for x, _ in data if x >= t)
        dd = sum(1 for x, e in data if x == t and e)
        if at_risk:
            s *= (1 - dd / at_risk)
        out.append((t, s, at_risk, dd))
    median = next((t for t, sv, _, _ in out if sv <= 0.5), None)
    return out, median, n


def main():
    events = rcsv(os.path.join(DATA, "lineage-events.csv"))
    pres = rcsv(os.path.join(DATA, "artifact-release-presence.csv"))
    mpres_fn = os.path.join(DATA, "move-release-presence.csv")
    mpres = rcsv(mpres_fn) if os.path.exists(mpres_fn) else []
    arts = registry()
    moves = moves_catalog()
    metrics = []

    def metric(L, name, value, n="", den="", note=""):
        share = ""
        if n == "" and isinstance(value, (int, float)) and den not in ("", 0, "0"):
            n = value
        try:
            if n != "" and den not in ("", 0, "0"):
                share = round(float(n) / float(den), 4)
        except (TypeError, ValueError):
            share = ""
        metrics.append({"lineage": L, "metric": name, "value": value, "n": n, "denominator": den, "share": share, "note": note})

    # ------------- release transitions -------------
    trans_rows = []
    per_art_class = []
    for L, repo in LREPO.items():
        rels = releases(repo)
        tags = [r["tag"] for r in rels]
        lp = [p for p in pres if p["lineage"] == L]
        present = defaultdict(dict)  # tag -> artifact -> row
        for p in lp:
            present[p["release"]][p["artifact_id"]] = p
        first_idx = min((tags.index(p["release"]) for p in lp if p["present"] == "1" and p["release"] in tags), default=0)
        ev_L = [e for e in events if e["lineage"] == L]
        start_idx = max(0, first_idx)
        ev_first = [tags.index(e["release"]) for e in ev_L if e["release"] in tags]
        if ev_first:
            start_idx = min(start_idx, max(0, min(ev_first) - 1))
        landing = defaultdict(list)
        for e in ev_L:
            landing[e["release"]].append(e)
        mp = defaultdict(dict)
        for m in mpres:
            if m["lineage"] == L:
                mp[m["release"]][m["move_id"]] = int(m["present"])
        move_arts = {mid: {o.get("artifact_id") for o in (m.get("occurrences") or []) + [m.get("first_observed") or {}] if o.get("artifact_id")}
                     for mid, m in moves.items() if m["lineage"] == L}
        pairs = list(zip(tags[start_idx:-1], tags[start_idx + 1:]))
        pairs.append((tags[-1], "UNRELEASED_AT_CUTOFF"))
        for fr, to in pairs:
            evs = landing.get(to, []) if to != "UNRELEASED_AT_CUTOFF" else [e for e in ev_L if not e["release"].startswith("v")]
            # per-artifact classification
            cls = {}
            a_status = {}
            for aid in set(present.get(fr, {})) | set(present.get(to, {})):
                a_to = present.get(to, {}).get(aid)
                a_fr = present.get(fr, {}).get(aid)
                was = a_fr and a_fr["present"] == "1"
                now = a_to and a_to["present"] == "1"
                if not was and not now:
                    continue
                st = "carried"
                if was and not now:
                    st = "removed"
                elif now and not was:
                    st = "added"
                elif a_to and a_to["status_vs_prev"] == "modified":
                    st = "modified"
                a_status[aid] = st
            ev_by_art = defaultdict(list)
            for e in evs:
                for a in e["artifact_ids"].split(";"):
                    if a:
                        ev_by_art[a].append(e)
            for aid in set(a_status) | set(ev_by_art):
                if a_status.get(aid) == "added" and not any(x["event_type"] in ("replace", "change_default") for x in ev_by_art.get(aid, [])):
                    a_cls = None  # new role or new artifact: not a rebase of existing knowledge
                    a_pred = [p for p in arts.get(aid, {}).get("predecessors", []) if p.get("relation") in ("reimplementation_of", "replaces", "ported_from", "textual_successor")]
                    if a_pred and to != "UNRELEASED_AT_CUTOFF":
                        a_cls = "manual_reimplementation"
                else:
                    cands = [EV2CLASS[x["event_type"]] for x in ev_by_art.get(aid, []) if x["event_type"] in EV2CLASS]
                    for x in ev_by_art.get(aid, []):
                        if x["event_type"] in ("introduce", "port") and aid in arts and any(
                                p.get("relation") in ("reimplementation_of", "replaces") for p in arts[aid].get("predecessors", [])):
                            cands.append("manual_reimplementation")
                    if a_status.get(aid) == "removed":
                        ended = (arts.get(aid, {}).get("ended") or {}).get("how", "")
                        cands.append("backend_replacement" if "replaced" in ended or "merged_into" in ended else "obsolete")
                    a_cls = max(cands, key=lambda c: RANK[c]) if cands else ("direct_carry" if a_status.get(aid) in ("carried", "modified") else None)
                if a_cls:
                    cls[aid] = a_cls
                    per_art_class.append({"lineage": L, "from_release": fr, "to_release": to, "artifact_id": aid,
                                          "presence_status": a_status.get(aid, "n/a"), "rebase_class": a_cls,
                                          "events": ";".join(x["event_id"] for x in ev_by_art.get(aid, []))})
            counts = Counter(cls.values())
            non_carry = [c for c in cls.values() if c != "direct_carry"]
            rclass = max(non_carry, key=lambda c: RANK[c]) if non_carry else ("direct_carry" if cls else "unclear")
            mfr, mto = mp.get(fr, {}), mp.get(to, {})
            preserved = sorted(m for m in mfr if mfr[m] and mto.get(m))
            new = sorted(m for m in mto if mto[m] and not mfr.get(m))
            lost = sorted(m for m in mfr if mfr[m] and not mto.get(m)) if to != "UNRELEASED_AT_CUTOFF" else []
            repaired = sorted({mid for e in evs if e["event_type"] in REPAIR for mid in moves if moves[mid]["lineage"] == L and (
                mid in e["optimization_move_ids"].split(";") or (move_arts.get(mid, set()) & set(e["artifact_ids"].split(";"))))})
            spec = [f"{e['event_id']}: {e['spec_change'][:60]}" for e in evs if e["spec_change"] and e["spec_change"] != "none"]
            frame = [f"{e['event_id']}: {(e['integration_change'] if e['integration_change'] != 'none' else e['title'])[:60]}"
                     for e in evs if e["primary_cause"] == "framework_integration" or (e["integration_change"] and e["integration_change"] != "none")]
            ev_types = Counter(e["event_type"] for e in evs)
            trans_rows.append({
                "lineage": L, "from_release": fr, "to_release": to,
                "spec_delta": f"{len(spec)} events" + (": " + " | ".join(spec[:3]) if spec else ""),
                "framework_delta": f"{len(frame)} events" + (": " + " | ".join(frame[:3]) if frame else ""),
                "artifacts_carried": sum(1 for s in a_status.values() if s == "carried"),
                "artifacts_modified": sum(1 for s in a_status.values() if s == "modified"),
                "artifacts_added": sum(1 for s in a_status.values() if s == "added"),
                "artifacts_removed": sum(1 for s in a_status.values() if s == "removed"),
                "moves_preserved": ";".join(preserved), "moves_repaired": ";".join(repaired),
                "moves_lost": ";".join(lost), "moves_new": ";".join(new),
                "rebase_class": rclass,
                "evidence": f"{len(evs)} landing events {dict(ev_types)}; per-artifact classes {dict(counts)}; ids: " + ";".join(e["event_id"] for e in evs[:8]),
                "confidence": "low" if rclass == "unclear" else ("high" if evs and all(e["confidence"] == "high" for e in evs) else "medium"),
                "n_landing_events": len(evs), "n_repair_events": sum(ev_types[t] for t in REPAIR),
                "repair_events_by_type": json.dumps({t: ev_types[t] for t in REPAIR if ev_types[t]}),
                "rebase_class_counts": json.dumps(dict(counts)),
            })
    cols = ["lineage", "from_release", "to_release", "spec_delta", "framework_delta", "artifacts_carried", "artifacts_modified",
            "artifacts_added", "artifacts_removed", "moves_preserved", "moves_repaired", "moves_lost", "moves_new",
            "rebase_class", "evidence", "confidence", "n_landing_events", "n_repair_events", "repair_events_by_type",
            "rebase_class_counts"]
    wcsv(os.path.join(DATA, "release-transitions.csv"), trans_rows, cols)
    wcsv(os.path.join(DATA, "release-transition-artifacts.csv"), per_art_class)

    # ------------- survival across k releases -------------
    surv_rows = []
    for kind, rows, key in (("artifact", pres, "artifact_id"), ("move", mpres, "move_id")):
        for L, repo in LREPO.items():
            tags = [r["tag"] for r in releases(repo)]
            byk = defaultdict(dict)
            for r in rows:
                if r["lineage"] == L and r["release"] in tags:
                    byk[r[key]][r["release"]] = int(r["present"])
            for k in (1, 2, 3):
                at_risk = surv = 0
                at_risk_t = surv_t = 0  # transition-based (all present releases)
                for item, pr in byk.items():
                    idx = [i for i, t in enumerate(tags) if pr.get(t)]
                    if not idx:
                        continue
                    e = idx[0]
                    if e + k < len(tags):
                        at_risk += 1
                        surv += int(bool(pr.get(tags[e + k])))
                    for i in idx:
                        if i + k < len(tags):
                            at_risk_t += 1
                            surv_t += int(bool(pr.get(tags[i + k])))
                surv_rows.append({"kind": kind, "lineage": L, "k_releases": k, "basis": "entry_cohort", "at_risk": at_risk,
                                  "survived": surv, "share": round(surv / at_risk, 4) if at_risk else ""})
                surv_rows.append({"kind": kind, "lineage": L, "k_releases": k, "basis": "all_present_releases", "at_risk": at_risk_t,
                                  "survived": surv_t, "share": round(surv_t / at_risk_t, 4) if at_risk_t else ""})
        # pooled
        for k in (1, 2, 3):
            for basis in ("entry_cohort", "all_present_releases"):
                sub = [r for r in surv_rows if r["kind"] == kind and r["k_releases"] == k and r["basis"] == basis and r["lineage"] != "all"]
                a = sum(r["at_risk"] for r in sub)
                s = sum(r["survived"] for r in sub)
                surv_rows.append({"kind": kind, "lineage": "all", "k_releases": k, "basis": basis, "at_risk": a, "survived": s,
                                  "share": round(s / a, 4) if a else ""})
    wcsv(os.path.join(DATA, "survival-by-releases.csv"), surv_rows)
    # textual identity: artifact present and byte-identical (no change) across the next k releases
    for L, repo in LREPO.items():
        tags = [r["tag"] for r in releases(repo)]
        st = defaultdict(dict)
        for p in pres:
            if p["lineage"] == L and p["release"] in tags:
                st[p["artifact_id"]][p["release"]] = p["status_vs_prev"] if p["present"] == "1" else "absent"
        for k in (1, 2, 3):
            at = ok = 0
            for aid, s in st.items():
                for i, t in enumerate(tags):
                    if s.get(t) in ("carried", "modified", "added") and i + k < len(tags):
                        at += 1
                        ok += int(all(s.get(tags[j]) == "carried" for j in range(i + 1, i + k + 1)))
            surv_rows.append({"kind": "artifact_unchanged", "lineage": L, "k_releases": k, "basis": "all_present_releases",
                              "at_risk": at, "survived": ok, "share": round(ok / at, 4) if at else ""})
    for k in (1, 2, 3):
        sub = [r for r in surv_rows if r["kind"] == "artifact_unchanged" and r["k_releases"] == k and r["lineage"] != "all"]
        a, s = sum(r["at_risk"] for r in sub), sum(r["survived"] for r in sub)
        surv_rows.append({"kind": "artifact_unchanged", "lineage": "all", "k_releases": k, "basis": "all_present_releases",
                          "at_risk": a, "survived": s, "share": round(s / a, 4) if a else ""})
    wcsv(os.path.join(DATA, "survival-by-releases.csv"), surv_rows)
    # code survival of event contributions by event-year cohort
    for L in ("L1", "L2", "L3", "all"):
        ev = [e for e in events if (L == "all" or e["lineage"] == L) and e["survives_at_cutoff"] in ("code", "concept_only", "no")]
        for y in sorted({e["date"][:4] for e in ev}):
            sub = [e for e in ev if e["date"][:4] == y]
            c = sum(1 for e in sub if e["survives_at_cutoff"] == "code")
            cc = sum(1 for e in sub if e["survives_at_cutoff"] == "concept_only")
            metric(L, f"event_code_survival_share:{y}", round(c / len(sub), 4), c, len(sub), "events with >=1 blamed line at cutoff")
            metric(L, f"event_concept_only_share:{y}", round(cc / len(sub), 4), cc, len(sub), "no surviving lines but a move still live")

    # ------------- artifact milestones + KM -------------
    ms_rows = []
    km_life, km_rep = [], []
    cutoff = d(S.CUTOFF)
    ev_by_art = defaultdict(list)
    for e in events:
        for a in e["artifact_ids"].split(";"):
            if a:
                ev_by_art[a].append(e)
    for aid, a in arts.items():
        intro = (a.get("introduced") or {}).get("date")
        if not intro or len(intro) < 10 or "?" in intro:
            continue
        t0 = d(intro[:10])
        if t0 > cutoff:
            continue
        evs = sorted(ev_by_art.get(aid, []), key=lambda e: e["date"])
        row = {"lineage": a["lineage"], "artifact_id": aid, "kind": a.get("kind"), "introduced": intro[:10],
               "n_events": len(evs)}
        for name, types in MILESTONE_TYPES.items():
            hit = next((e for e in evs if e["event_type"] in types and d(e["date"][:10]) >= t0), None)
            row[name + "_days"] = (d(hit["date"][:10]) - t0).days if hit else ""
            row[name + "_event"] = hit["event_id"] if hit else ""
        ended = a.get("ended") or {}
        end_date = ended.get("date")
        row["ended"] = end_date or ""
        row["ended_how"] = ended.get("how", "") if ended else ""
        if end_date and len(end_date) >= 10 and d(end_date[:10]) <= cutoff:
            life = (d(end_date[:10]) - t0).days
            km_life.append((life, 1, a["lineage"]))
            row["lifetime_days"] = life
            row["censored"] = 0
        else:
            life = (cutoff - t0).days
            km_life.append((life, 0, a["lineage"]))
            row["lifetime_days"] = life
            row["censored"] = 1
        rep = row["first_repair_days"]
        if rep != "":
            km_rep.append((rep, 1, a["lineage"]))
        else:
            km_rep.append((min(life, (cutoff - t0).days), 0, a["lineage"]))
        ms_rows.append(row)
    wcsv(os.path.join(DATA, "artifact-milestones.csv"), ms_rows)
    for name, data in (("km-artifact-lifetime.csv", km_life), ("km-first-repair.csv", km_rep)):
        out = []
        for L in ("L1", "L2", "L3", "all"):
            sub = [(t, e) for t, e, l in data if L == "all" or l == L]
            curve, median, n = km(sub)
            for t, s, r, dd in curve:
                out.append({"lineage": L, "t_days": t, "survival": round(s, 4), "at_risk": r, "events": dd})
            metric(L, name.replace(".csv", "") + "_median_days", median if median is not None else "not_reached", n, n,
                   f"KM; events={sum(e for _, e in sub)}")
        wcsv(os.path.join(DATA, name), out)

    # ------------- event-level metrics -------------
    for L in ("L1", "L2", "L3", "all"):
        ev = [e for e in events if L == "all" or e["lineage"] == L]
        n = len(ev)
        metric(L, "verified_events", n, n, n)
        for t, c in Counter(e["event_type"] for e in ev).items():
            metric(L, f"event_type:{t}", c, c, n)
        for t, c in Counter(e["primary_cause"] for e in ev).items():
            metric(L, f"primary_cause:{t}", c, c, n)
        repl = sum(1 for e in ev if e["event_type"] in ("replace", "change_default", "remove", "deprecate") or (
            e["event_type"] in ("introduce", "port") and any(p.get("relation") in ("reimplementation_of", "replaces")
                                                             for a in e["artifact_ids"].split(";") for p in arts.get(a, {}).get("predecessors", []))))
        incr = sum(1 for e in ev if e["event_type"] in ("optimize", "retune", "repair_correctness", "repair_performance",
                                                          "repair_build_dependency", "adapt_framework", "extend_support", "integrate"))
        metric(L, "replacement_events", repl, repl, n, "replace/change_default/remove/deprecate + introduce/port with reimplementation/replaces predecessor")
        metric(L, "incremental_events", incr, incr, n, "optimize/retune/repair_*/adapt/extend/integrate")
        for s, c in Counter(e["survives_at_cutoff"] for e in ev).items():
            metric(L, f"survives_at_cutoff:{s}", c, c, n)
        for o, c in Counter(e["outcome"] for e in ev).items():
            metric(L, f"outcome:{o}", c, c, n)
        spec = sum(1 for e in ev if e["spec_change"] and e["spec_change"] != "none" and not e["spec_change"].startswith("newly_supported"))
        newly = sum(1 for e in ev if e["spec_change"].startswith("newly_supported"))
        metric(L, "spec_change_events", spec, spec, n, "contract change (excl. newly_supported)")
        metric(L, "newly_supported_events", newly, newly, n)
        rel = [e for e in ev if e["release"].startswith("v")]
        metric(L, "events_in_a_release", len(rel), len(rel), n)
        metric(L, "events_unreleased_at_cutoff", sum(1 for e in ev if e["release"] in ("unreleased_at_cutoff",)), "", n)
        metric(L, "events_wheel_not_pinned", sum(1 for e in ev if "wheel" in e["release"]), "", n)
    tr = [t for t in trans_rows if t["to_release"] != "UNRELEASED_AT_CUTOFF"]
    for L in ("L1", "L2", "L3", "all"):
        sub = [t for t in tr if L == "all" or t["lineage"] == L]
        if not sub:
            continue
        reps = sorted(int(t["n_repair_events"]) for t in sub)
        metric(L, "transitions", len(sub), len(sub), len(sub))
        metric(L, "repairs_per_transition_mean", round(sum(reps) / len(reps), 2), sum(reps), len(sub))
        metric(L, "repairs_per_transition_median", reps[len(reps) // 2], "", len(sub))
        metric(L, "transitions_with_repair", sum(1 for x in reps if x), sum(1 for x in reps if x), len(sub))
        for c, k in Counter(t["rebase_class"] for t in sub).items():
            metric(L, f"rebase_class:{c}", k, k, len(sub))
    pac = [p for p in per_art_class if p["to_release"] != "UNRELEASED_AT_CUTOFF"]
    for L in ("L1", "L2", "L3", "all"):
        sub = [p for p in pac if L == "all" or p["lineage"] == L]
        for c, k in Counter(p["rebase_class"] for p in sub).items():
            metric(L, f"artifact_transition_class:{c}", k, k, len(sub))
        # direct_carry although the artifact's files changed textually (only via commits screened out of the lineage)
        dcm = sum(1 for p in sub if p["rebase_class"] == "direct_carry" and p["presence_status"] == "modified")
        metric(L, "artifact_transition_direct_carry_textually_modified", dcm, dcm, len(sub))
    for r in surv_rows:
        metric(r["lineage"], f"{r['kind']}_survival_k{r['k_releases']}_{r['basis']}", r["share"], r["survived"], r["at_risk"])
    extra_metrics(metric, events, arts)
    wcsv(os.path.join(DATA, "lineage-metrics.csv"), metrics)
    print("transitions", len(trans_rows), "artifact-transition rows", len(per_art_class), "metrics", len(metrics))


def extra_metrics(metric, events, arts):
    """Counts used in reports, so every printed number maps to one CSV row."""
    cands = rcsv(os.path.join(DATA, "candidate-events.csv"))
    edges = rcsv(os.path.join(DATA, "lineage-edges.csv"))
    ev = {e["event_id"]: e for e in events}
    for L in ("L1", "L2", "L3", "all"):
        cs = [c for c in cands if L == "all" or c["lineage"] == L]
        metric(L, "candidates", len(cs), len(cs), len(cs))
        for k, v in Counter(c["screening_label"] for c in cs).items():
            metric(L, f"candidate_label:{k}", v, v, len(cs))
        for k, v in Counter(c["adjudication_status"] for c in cs).items():
            metric(L, f"candidate_status:{k}", v, v, len(cs))
        strat = Counter()
        for c in cs:
            for s in set(x.split("(")[0] for x in c["discovery_source"].split(";") if x):
                strat[s] += 1
        for k, v in strat.items():
            metric(L, f"discovery:{k}", v, v, len(cs))
        single = Counter(c["discovery_source"] for c in cs if ";" not in c["discovery_source"])
        ver_single = Counter(c["discovery_source"] for c in cs if ";" not in c["discovery_source"] and c["adjudication_status"] == "verified")
        for k, v in ver_single.items():
            metric(L, f"verified_single_strategy:{k.split('(')[0]}", v, v, single[k])
        a = [x for x in arts.values() if L == "all" or x["lineage"] == L]
        metric(L, "artifacts_registered", len(a), len(a), len(a))
        metric(L, "artifacts_live_at_cutoff", sum(1 for x in a if x.get("live_at_cutoff")), "", len(a))
        metric(L, "artifacts_ended", sum(1 for x in a if x.get("ended")), "", len(a))
        metric(L, "artifacts_upstream", sum(1 for x in a if x.get("kind") in ("upstream_kernel", "dependency_pin")), "", len(a))
        es = [x for x in edges if L == "all" or x["lineage"] == L]
        metric(L, "edges", len(es), len(es), len(es))
        for k, v in Counter(x["edge_type"] for x in es).items():
            metric(L, f"edge_type:{k}", v, v, len(es))
        xrepo = [x for x in es if x["from_event_id"] in ev and x["to_event_id"] in ev and ev[x["from_event_id"]]["repo"] != ev[x["to_event_id"]]["repo"]]
        metric(L, "cross_repo_edges", len(xrepo), len(xrepo), len(es))
        xl = [x for x in es if x["from_event_id"][:2] != x["to_event_id"][:2]]
        metric(L, "cross_lineage_edges", len(xl), len(xl), len(es))
        evl = [e for e in events if L == "all" or e["lineage"] == L]
        metric(L, "events_with_moves", sum(1 for e in evl if e["optimization_move_ids"]), "", len(evl))
        metric(L, "key_events", sum(1 for e in evl if e["key_event"] == "1"), "", len(evl))
        metric(L, "events_hardware_specific", sum(1 for e in evl if e["hardware_scope"] and not set(e["hardware_scope"].split(";")) <= {"all_cuda", "unspecified", ""}), "", len(evl))
        metric(L, "events_perf_claim", sum(1 for e in evl if e["performance_claim"] and e["performance_claim"] != "none"), "", len(evl))
        metric(L, "events_correctness_evidence", sum(1 for e in evl if e["correctness_evidence"] and e["correctness_evidence"] != "none"), "", len(evl))
        metric(L, "events_default_change", sum(1 for e in evl if e["default_change"] and e["default_change"] != "none"), "", len(evl))
        metric(L, "events_integration_change", sum(1 for e in evl if e["integration_change"] and e["integration_change"] != "none"), "", len(evl))
        conf = Counter(e["confidence"] for e in evl)
        for k, v in conf.items():
            metric(L, f"confidence:{k}", v, v, len(evl))
        spec_req = Counter(e["spec_change"].split(":")[0].strip() for e in evl if e["spec_change"] and e["spec_change"] != "none")
        for k, v in spec_req.items():
            metric(L, f"spec_requirement:{k}", v, v, len(evl))
        rels = Counter(e["release_route"].split(">=")[0] for e in evl)
        for k, v in rels.items():
            metric(L, f"release_route:{k}", v, v, len(evl))
    mv_fn = os.path.join(DATA, "optimization-moves.csv")
    if os.path.exists(mv_fn):
        mv = rcsv(mv_fn)
        for mm in mv:
            used = [e for e in events if mm["move_id"] in e["optimization_move_ids"].split(";")]
            rep = [e for e in used if e["event_type"] in ("repair_correctness", "repair_performance", "repair_build_dependency", "revert")]
            metric(mm["lineage"], f"move_coded_events:{mm['move_id']}", len(used), len(used), len(used))
            metric(mm["lineage"], f"move_repair_events:{mm['move_id']}", len(rep), len(rep), len(used) or "")
        for L in ("L1", "L2", "L3", "all"):
            sub = [mm for mm in mv if L == "all" or mm["lineage"] == L]
            with_rep = [mm for mm in sub if any(e["event_type"] in ("repair_correctness", "repair_performance", "repair_build_dependency", "revert")
                                                and mm["move_id"] in e["optimization_move_ids"].split(";") for e in events)]
            metric(L, "moves_with_repair_events", len(with_rep), len(with_rep), len(sub))
            used_all = [e for e in events if (L == "all" or e["lineage"] == L) and e["optimization_move_ids"]]
            rep_all = [e for e in used_all if e["event_type"] in ("repair_correctness", "repair_performance", "repair_build_dependency", "revert")]
            metric(L, "move_tagged_repair_events", len(rep_all), len(rep_all), len(used_all))
        for L in ("L1", "L2", "L3", "all"):
            m = [x for x in mv if L == "all" or x["lineage"] == L]
            metric(L, "moves", len(m), len(m), len(m))
            metric(L, "moves_with_biography", sum(1 for x in m if x["has_biography"] == "1"), "", len(m))
            metric(L, "moves_cross_repo", sum(1 for x in m if int(x["cross_repo_occurrences"] or 0) > 0), "", len(m))
            metric(L, "moves_with_unacknowledged_recurrence", sum(1 for x in m if int(x["unacknowledged_recurrences"] or 0) > 0), "", len(m))
            for k, v in Counter(x["current_status"].split(":")[0] for x in m).items():
                metric(L, f"move_status:{k}", v, v, len(m))
    ma_fn = os.path.join(DATA, "move-assumptions.csv")
    if os.path.exists(ma_fn):
        ma = rcsv(ma_fn)
        for L in ("L1", "L2", "L3", "all"):
            a = [x for x in ma if (L == "all" or x["lineage"] == L) and x["source_file"] == "audit"]
            metric(L, "audited_assumptions", len(a), len(a), len(a))
            for k, v in Counter(x["status"] for x in a).items():
                metric(L, f"assumption_status:{k}", v, v, len(a))
            for k, v in Counter(x["kind"] for x in a).items():
                metric(L, f"assumption_kind:{k}", v, v, len(a))
            hw = [x for x in a if x["kind"] in ("hardware", "programming_model")]
            for k, v in Counter(x["status"] for x in hw).items():
                metric(L, f"hw_assumption_status:{k}", v, v, len(hw))
    import glob as _g
    fails = []
    for fn in _g.glob(os.path.join(STAGING, "failures", "*-failures.json")):
        d = json.load(open(fn, encoding="utf-8"))
        for h in d["histories"]:
            h["lineage"] = d["lineage"]
            fails.append(h)
        c = d.get("census") or {}
        for k in ("linked_failures", "with_introducing_change", "reverts", "relands"):
            if k in c:
                metric(d["lineage"], f"failure_census:{k}", c[k], c[k], "")
    for L in ("L1", "L2", "L3", "all"):
        f = [h for h in fails if L == "all" or h["lineage"] == L]
        metric(L, "failure_histories", len(f), len(f), len(f))
        metric(L, "failure_histories_no_durable_encoding", sum(1 for h in f if "no durable encoding found" in (h.get("durable_encoding") or [])), "", len(f))
        for k in ("test", "guard", "comment", "documented_requirement", "abstraction"):
            metric(L, f"failure_histories_encoding:{k}", sum(1 for h in f if k in (h.get("durable_encoding") or [])), "", len(f))
        for k, v in Counter(h.get("invalidation_scope") for h in f).items():
            metric(L, f"failure_scope:{k}", v, v, len(f))
        def rep_class(h):
            t = (h.get("repeat_assumption") or "").strip().lower()
            if not t or t.startswith("none"):
                return "none_found"
            if t.startswith("possible"):
                return "possible"
            return "verified"
        rep = sum(1 for h in f if rep_class(h) == "verified")
        pos = sum(1 for h in f if rep_class(h) == "possible")
        metric(L, "failure_histories_repeat_assumption", rep, rep, len(f))
        metric(L, "failure_histories_repeat_possible", pos, pos, len(f))
        metric(L, "failure_histories_repeat_none_found", len(f) - rep - pos, len(f) - rep - pos, len(f))


if __name__ == "__main__":
    main()
