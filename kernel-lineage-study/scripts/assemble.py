"""Assemble study datasets from staged outputs.

  python assemble.py all

Inputs : staging/candidates.jsonl, staging/screen/*/batch-*-output.jsonl,
         staging/code/*/batch-*-output.jsonl, staging/registry/*-registry.json,
         staging/registry/new-artifact-map.json (optional), staging/moves/*-moves.json,
         staging/move-name-map.json (optional), staging/upstream-events.json (optional),
         data/cutoff-line-survival.csv, cache/git/*
Outputs: data/candidate-events.csv, data/lineage-events.csv, data/lineage-edges.csv,
         data/artifacts.csv, data/artifact-snapshots.csv
"""
import csv
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402
from releases import Releases  # noqa: E402

csv.field_size_limit(10 ** 9)
STAGING = os.path.join(S.STUDY, "staging")
DATA = os.path.join(S.STUDY, "data")
SLUG = {"sglang": "sgl-project/sglang", "vllm": "vllm-project/vllm"}
REPO_KEY = {v: k for k, v in SLUG.items()}
LREPO = {"L1": "sglang", "L2": "sglang", "L3": "vllm"}
GITDIR = {r: os.path.join(S.STUDY, "cache", "git", f"{r}.git") for r in ("sglang", "vllm")}
KEY_TYPES = {"introduce", "optimize", "port", "replace", "change_default", "remove", "revert", "reland", "integrate", "deprecate"}


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def outputs(mode):
    res = {}
    for fn in sorted(glob.glob(os.path.join(STAGING, mode, "L*", "batch-*-output.jsonl"))):
        for r in jl(fn):
            r["_batch"] = os.path.relpath(fn, STAGING)
            res[r["candidate_id"]] = r
    # later passes (e.g. recode/reconcile overrides) live in staging/<mode>-overrides.jsonl
    ov = os.path.join(STAGING, f"{mode}-overrides.jsonl")
    if os.path.exists(ov):
        for r in jl(ov):
            res.setdefault(r["candidate_id"], {}).update(r)
    return res


def pr_url(repo_slug, pr):
    return f"https://github.com/{repo_slug}/pull/{pr}" if pr else ""


def commit_url(repo_slug, sha):
    return f"https://github.com/{repo_slug}/commit/{sha}" if sha else ""


def load_registry():
    arts = {}
    for L in ("L1", "L2", "L3"):
        d = json.load(open(os.path.join(STAGING, "registry", f"{L}-registry.json"), encoding="utf-8"))
        for a in d["artifacts"]:
            a["lineage"] = L
            arts[a["artifact_id"]] = a
    mp = {}
    fn = os.path.join(STAGING, "registry", "new-artifact-map.json")
    if os.path.exists(fn):
        mp = json.load(open(fn, encoding="utf-8"))
    for fn in glob.glob(os.path.join(STAGING, "registry", "L*-new-map.json")):
        for k, v in json.load(open(fn, encoding="utf-8")).items():
            mp[k] = v
    return arts, mp


def load_moves():
    moves = {}
    for fn in glob.glob(os.path.join(STAGING, "moves", "*-moves.json")):
        d = json.load(open(fn, encoding="utf-8"))
        for m in d["moves"]:
            m["lineage"] = d["lineage"]
            moves[m["move_id"]] = m
    name_map = {}
    fn = os.path.join(STAGING, "move-name-map.json")
    if os.path.exists(fn):
        name_map = json.load(open(fn, encoding="utf-8"))
    return moves, name_map


def fp_files():
    out = {}
    for repo in ("sglang", "vllm"):
        for r in jl(os.path.join(S.STUDY, "cache", "git", f"{repo}-fp.jsonl")):
            out[(repo, r["sha"])] = [f[-1] for f in r["files"]] + [f[1] for f in r["files"] if len(f) == 3]
    return out


def line_survival():
    surv = defaultdict(int)       # (repo, commit) -> lines in lineage files
    by_art = defaultdict(int)     # (repo, commit, artifact) -> lines
    fn = os.path.join(DATA, "cutoff-line-survival.csv")
    if os.path.exists(fn):
        for r in csv.DictReader(open(fn, encoding="utf-8")):
            surv[(r["repo"], r["commit"])] += int(r["lines"])
            by_art[(r["repo"], r["commit"], r["artifact_id"])] += int(r["lines"])
    return surv, by_art


def norm_art(a, mp):
    a = a.strip()
    return mp.get(a, a)


def resolve_target(t, repo_slug, ev_by_pr, ev_by_sha):
    t = (t or "").strip()
    m = re.search(r"(?:#|pull/|PR\s*#?)(\d{2,6})", t)
    if m:
        return ev_by_pr.get((repo_slug, int(m.group(1))))
    m = re.search(r"\b([0-9a-f]{7,40})\b", t)
    if m:
        for (rs, sha), e in ev_by_sha.items():
            if rs == repo_slug and sha.startswith(m.group(1)):
                return e
    return None


def main():
    cands = jl(os.path.join(STAGING, "candidates.jsonl"))
    screen = outputs("screen")
    code = outputs("code")
    arts, amap = load_registry()
    moves, mname = load_moves()
    files = fp_files()
    surv, surv_art = line_survival()
    R = Releases()

    # ---------------- candidate-events.csv ----------------
    oos_only = set()
    for cid, k in code.items():
        if k.get("final_label", "verified_lineage_event") == "verified_lineage_event":
            mapped = [norm_art(a, amap) for a in k.get("artifact_ids", [])]
            if mapped and all(a == "OUT_OF_SCOPE" for a in mapped):
                oos_only.add(cid)
    cand_rows = []
    for c in cands:
        s = screen.get(c["candidate_id"], {})
        k = code.get(c["candidate_id"])
        label = s.get("screening_label", "")
        status = "screened" if s else "unscreened"
        reason = s.get("rejection_reason", "")
        one_pass = k is not None and "_batch" in k and ("batch-900" in k["_batch"] or "batch-901" in k["_batch"])
        if one_pass:
            # cross_reference (batch-900) and hand-verified upstream anchors (batch-901) are screened and
            # coded in one pass; that adjudication overrides any stage-1 label
            kind = "cross_reference" if "batch-900" in k["_batch"] else "upstream_anchor"
            label = k.get("final_label", "verified_lineage_event")
            status = "verified" if label == "verified_lineage_event" else f"adjudicated_{kind}"
            reason = "" if label == "verified_lineage_event" else (k.get("rejection_reason") or f"{kind}_not_event")
        elif label == "verified_lineage_event":
            if k is None:
                status = "awaiting_stage2"
            elif k.get("final_label", "verified_lineage_event") == "verified_lineage_event":
                status = "verified"
            else:
                status = "rejected_stage2"
                label = k["final_label"]
                reason = k.get("rejection_reason", "") or "stage2_overturned"
        if c["candidate_id"] in oos_only:
            status, label, reason = "rejected_reconciliation", "out_of_scope", "other_operator (artifact reconciliation: all artifacts out of scope)"
        cand_rows.append({
            "candidate_id": c["candidate_id"], "lineage": c["lineage"], "repo": c["repo"], "date": c["date"],
            "commit_sha": c["commit_sha"], "pr_number": c["pr_number"], "title": c["title"],
            "source_paths": ";".join(c["source_paths"][:12]), "discovery_source": ";".join(c["discovery_source"]),
            "screening_label": label, "rejection_reason": reason, "adjudication_status": status,
            "evidence_urls": " ".join(x for x in (pr_url(c["repo"], c["pr_number"]), commit_url(c["repo"], c["commit_sha"])) if x),
            "screening_rationale": ((k or {}).get("notes") if (status == "rejected_stage2" or (one_pass and not s)) and (k or {}).get("notes")
                                    else s.get("rationale", "")),
            "artifact_ids": ";".join(norm_art(a, amap) for a in (s.get("artifact_ids") or ((k or {}).get("artifact_ids") if one_pass else None) or [])),
            "stratum": c.get("stratum", ""),
        })
    write_csv(os.path.join(DATA, "candidate-events.csv"), cand_rows)

    # ---------------- lineage-events.csv ----------------
    events = []
    cmap = {c["candidate_id"]: c for c in cands}
    for cid, k in code.items():
        if k.get("final_label", "verified_lineage_event") != "verified_lineage_event" or cid not in cmap or cid in oos_only:
            continue
        c = cmap[cid]
        repo = REPO_KEY[c["repo"]]
        paths = files.get((repo, c["commit_sha"]), c["source_paths"])
        rel = R.first_release(repo, c["commit_sha"], c["pr_number"], paths)
        a_ids = []
        for a in k.get("artifact_ids", []):
            a2 = norm_art(a, amap)
            if a2 != "OUT_OF_SCOPE" and a2 not in a_ids:
                a_ids.append(a2)
        mv = []
        for m in k.get("moves", []) or []:
            mid = m.get("move_id") or mname.get(m.get("name", ""), "")
            if not mid and m.get("name") in moves:
                mid = m["name"]
            if mid and mid not in mv:
                mv.append(mid)
        events.append({
            "event_id": f"{c['lineage']}-E-{c['commit_sha'][:10]}", "lineage": c["lineage"], "date": c["date"],
            "repo": c["repo"], "release": rel["tag"] or rel["route"], "commit_sha": c["commit_sha"],
            "pr_number": c["pr_number"], "event_type": k["event_type"], "artifact_ids": ";".join(a_ids),
            "parent_event_ids": "", "spec_change": k.get("spec_change", "none"),
            "integration_change": k.get("integration_change", "none"), "optimization_move_ids": ";".join(mv),
            "hardware_scope": ";".join(k.get("hardware_scope", []) or []),
            "performance_claim": k.get("performance_claim", "none"),
            "correctness_evidence": k.get("correctness_evidence", "none"), "outcome": "", "survives_at_cutoff": "",
            "confidence": k.get("confidence", ""), "evidence_excerpt": k.get("evidence_excerpt", ""),
            "evidence_urls": " ".join(x for x in (pr_url(c["repo"], c["pr_number"]), commit_url(c["repo"], c["commit_sha"])) if x),
            # extras
            "primary_cause": k.get("primary_cause", ""), "default_change": k.get("default_change", "none"),
            "evidence_types": ";".join(k.get("evidence_types", []) or []), "release_route": rel["route"],
            "release_date": rel["release_date"], "code_release": rel["code_tag"], "wheel_version": rel["wheel_version"],
            "key_event": int(k["event_type"] in KEY_TYPES), "candidate_id": cid, "title": c["title"],
            "move_names_raw": ";".join((m.get("name") or "") for m in (k.get("moves") or [])),
            "surviving_lines": surv.get((repo, c["commit_sha"]), 0),
            "_relations": k.get("relations", []) or [], "_assumptions": k.get("assumptions", []) or [],
            "_moves_raw": k.get("moves", []) or [], "notes": k.get("notes", ""),
        })
    # upstream / anchor events
    up = os.path.join(STAGING, "upstream-events.json")
    if os.path.exists(up):
        for u in json.load(open(up, encoding="utf-8")):
            u.setdefault("_relations", [])
            u.setdefault("_assumptions", [])
            u.setdefault("_moves_raw", [])
            u.setdefault("surviving_lines", "")
            u.setdefault("key_event", 1)
            events.append(u)
    events.sort(key=lambda e: (e["lineage"], e["date"], e["event_id"]))
    ev_by_pr, ev_by_sha, by_id = {}, {}, {}
    for e in events:
        by_id[e["event_id"]] = e
        if e.get("pr_number"):
            ev_by_pr.setdefault((e["repo"], int(e["pr_number"])), []).append(e)
        if e.get("commit_sha"):
            ev_by_sha.setdefault((e["repo"], e["commit_sha"]), []).append(e)

    def same_lineage(lst, L):
        if not lst:
            return None
        for e in lst:
            if e["lineage"] == L:
                return e
        return lst[0]

    edges = []
    eid = 0

    def add_edge(L, a, b, t, ev, conf, urls, retro=False):
        nonlocal eid
        eid += 1
        edges.append({"edge_id": f"X{eid:05d}", "lineage": L, "from_event_id": a, "to_event_id": b, "edge_type": t,
                      "evidence": ev + (" [retrospective]" if retro else ""), "confidence": conf, "evidence_urls": urls})

    # textual succession on the same artifact
    last_on_art = {}
    parents = defaultdict(list)
    for e in events:
        for a in [x for x in e["artifact_ids"].split(";") if x]:
            prev = last_on_art.get((e["lineage"], a))
            if prev and e["event_type"] not in ("introduce", "port") and prev["event_id"] != e["event_id"]:
                if prev["event_id"] not in parents[e["event_id"]]:
                    parents[e["event_id"]].append(prev["event_id"])
                    add_edge(e["lineage"], prev["event_id"], e["event_id"], "textual_successor",
                             f"same artifact {a}; previous lineage event touching it", "high", e["evidence_urls"])
            last_on_art[(e["lineage"], a)] = e
    # explicit relations
    rel_map = {"reverts": "reverts", "relands": "relands", "repairs": "repairs", "ported_from": "ported_from",
               "optimized_from": "optimized_from", "reimplementation_of": "reimplementation_of", "replaces": "replaces",
               "integrates": "integrates", "wraps": "wraps", "forked_from": "forked_from",
               "historical_connection_uncertain": "historical_connection_uncertain",
               "shares_optimization_move": "shares_optimization_move", "follows_up": "textual_successor",
               "supersedes": "replaces"}
    unresolved = []
    ext_rows = []
    supp = {}
    cand_by_pr = defaultdict(dict)
    for c in cands:
        if c["pr_number"]:
            cand_by_pr[(c["repo"], int(c["pr_number"]))][c["lineage"]] = c
    fp_pr = {}
    for repo in ("sglang", "vllm"):
        for r in jl(os.path.join(S.STUDY, "cache", "git", f"{repo}-fp.jsonl")):
            if r["pr"] is not None:
                fp_pr.setdefault((SLUG[repo], r["pr"]), r)
    art_events = defaultdict(list)
    for e in events:
        for a in [x for x in e["artifact_ids"].split(";") if x]:
            art_events[(e["lineage"], a)].append(e)

    def classify_target(e, r):
        t = (r.get("target") or "").strip()
        tslug = r.get("repo") or e["repo"]
        if t in arts or t.split()[0] in arts:
            aid = t.split()[0]
            prior = [x for x in art_events.get((e["lineage"], aid), []) if x["date"] < e["date"]]
            return ("artifact_ref", prior[-1] if prior else None, aid)
        if re.search(r"(?i)issue|discussion", t):
            return ("issue", None, t)
        m = re.search(r"(?:#|pull/|PR\s*#?)(\d{2,6})", t)
        if tslug in REPO_KEY and m:
            n = int(m.group(1))
            key = (tslug, n)
            if key in fp_pr:
                cl = cand_by_pr.get(key, {})
                other = [x for x in ev_by_pr.get(key, []) if x["lineage"] != e["lineage"]]
                if other:
                    return ("other_lineage_event", other[0], f"{tslug}#{n}")
                if e["lineage"] in cl:
                    return ("candidate_not_event:" + (screen.get(cl[e["lineage"]]["candidate_id"], {}).get("screening_label", "?")), None, f"{tslug}#{n}")
                return ("merged_pr_not_candidate", None, f"{tslug}#{n}")
            return ("pr_not_on_main_or_issue", None, f"{tslug}#{n}")
        if tslug not in REPO_KEY:
            return ("upstream_external", None, f"{tslug}:{t}")
        return ("unparseable", None, t)

    for e in events:
        for r in e["_relations"]:
            et = rel_map.get(r.get("relation", ""))
            if not et:
                continue
            tslug = r.get("repo") or e["repo"]
            tgt = resolve_target(r.get("target", ""), tslug, {k: same_lineage(v, e["lineage"]) for k, v in ev_by_pr.items()},
                                 {k: same_lineage(v, e["lineage"]) for k, v in ev_by_sha.items()})
            status = "resolved_event"
            if not tgt or tgt["event_id"] == e["event_id"]:
                status, tgt2, ref = classify_target(e, r)
                tgt = tgt2 if tgt2 is not None and tgt2["event_id"] != e["event_id"] else None
                ext_rows.append({"event_id": e["event_id"], "lineage": e["lineage"], "relation": r.get("relation", ""),
                                 "target_repo": tslug, "target": (r.get("target") or "")[:120], "resolution": status,
                                 "resolved_event_id": tgt["event_id"] if tgt else "", "evidence": (r.get("evidence") or "")[:200]})
                if tgt is None:
                    unresolved.append({"event_id": e["event_id"], "relation": r, "resolution": status})
                    if (status.startswith("candidate_not_event") or status == "merged_pr_not_candidate") and \
                            r.get("relation") in ("reverts", "relands", "repairs", "ported_from", "optimized_from", "replaces", "reimplementation_of"):
                        key = (e["lineage"], ref)
                        supp.setdefault(key, {"lineage": e["lineage"], "ref": ref, "cited_by": [], "status": status})
                        supp[key]["cited_by"].append(f"{e['event_id']}:{r.get('relation')}")
                    continue
            earlier, later = (tgt, e) if tgt["date"] <= e["date"] else (e, tgt)
            retro = earlier is e
            add_edge(e["lineage"], earlier["event_id"], later["event_id"], et, (r.get("evidence") or "")[:300],
                     e["confidence"] or "medium", e["evidence_urls"], retro=retro and et not in ("replaces",))
            if earlier["event_id"] not in parents[later["event_id"]]:
                parents[later["event_id"]].append(earlier["event_id"])
    if ext_rows:
        write_csv(os.path.join(DATA, "external-references.csv"), ext_rows)
    json.dump(list(supp.values()), open(os.path.join(STAGING, "supplementary-targets.json"), "w", encoding="utf-8"), indent=1)
    # outcomes and survival
    reverted = {x["from_event_id"] for x in edges if x["edge_type"] == "reverts"}
    later_touch = defaultdict(bool)
    for e in events:
        for a in [x for x in e["artifact_ids"].split(";") if x]:
            pass
    seen_art_after = defaultdict(list)
    for e in events:
        for a in [x for x in e["artifact_ids"].split(";") if x]:
            seen_art_after[(e["lineage"], a)].append(e)
    for e in events:
        e["parent_event_ids"] = ";".join(parents.get(e["event_id"], []))
        a_list = [x for x in e["artifact_ids"].split(";") if x]
        ended = [arts[a]["ended"] for a in a_list if a in arts and arts[a].get("ended")]
        modified = any(x["date"] > e["date"] for a in a_list for x in seen_art_after[(e["lineage"], a)])
        if e["event_id"] in reverted:
            e["outcome"] = "merged_reverted"
        elif a_list and len(ended) == len([a for a in a_list if a in arts]) and ended:
            e["outcome"] = "merged_replaced" if any("replaced" in (x.get("how") or "") for x in ended) else "merged_removed"
        elif modified:
            e["outcome"] = "merged_modified_later"
        else:
            e["outcome"] = "merged_survives"
        kinds = [arts[a]["kind"] for a in a_list if a in arts]
        if e.get("surviving_lines"):
            e["survives_at_cutoff"] = "code"
        elif any((moves.get(m, {}).get("current_status") or "").startswith("live") for m in e["optimization_move_ids"].split(";") if m):
            e["survives_at_cutoff"] = "concept_only"
        elif kinds and all(k in ("upstream_kernel", "dependency_pin") for k in kinds):
            e["survives_at_cutoff"] = "unclear"
        elif e["event_type"] in ("revert", "remove", "deprecate"):
            e["survives_at_cutoff"] = "not_applicable"
        else:
            e["survives_at_cutoff"] = "no"
    cols = ["event_id", "lineage", "date", "repo", "release", "commit_sha", "pr_number", "event_type", "artifact_ids",
            "parent_event_ids", "spec_change", "integration_change", "optimization_move_ids", "hardware_scope",
            "performance_claim", "correctness_evidence", "outcome", "survives_at_cutoff", "confidence",
            "evidence_excerpt", "evidence_urls", "primary_cause", "default_change", "evidence_types", "release_route",
            "release_date", "code_release", "wheel_version", "key_event", "candidate_id", "title", "move_names_raw",
            "surviving_lines", "notes"]
    write_csv(os.path.join(DATA, "lineage-events.csv"), [{k: e.get(k, "") for k in cols} for e in events])
    write_csv(os.path.join(DATA, "lineage-edges.csv"), edges)
    json.dump(unresolved, open(os.path.join(STAGING, "unresolved-relations.json"), "w", encoding="utf-8"), indent=1)
    # assumptions table (per event) for WP-E
    arows = []
    for e in events:
        for a in e["_assumptions"]:
            arows.append({"event_id": e["event_id"], "lineage": e["lineage"], "assumption": a.get("assumption", ""),
                          "status": a.get("status", ""), "source": a.get("source", ""), "excerpt": a.get("excerpt", "")})
    if arows:
        write_csv(os.path.join(DATA, "event-assumptions.csv"), arows)
    # raw move mentions for consolidation
    mrows = []
    for e in events:
        for m in e["_moves_raw"]:
            mrows.append({"event_id": e["event_id"], "lineage": e["lineage"], "date": e["date"], "name": m.get("name", ""),
                          "move_id": m.get("move_id", ""), "category": m.get("category", ""), "status": m.get("status", ""),
                          "mechanism": m.get("mechanism", ""), "source": m.get("source", "")})
    if mrows:
        write_csv(os.path.join(STAGING, "move-mentions.csv"), mrows)
    print("candidates", len(cand_rows), "events", len(events), "edges", len(edges), "unresolved relations", len(unresolved))


def write_csv(path, rows):
    if not rows:
        print("no rows for", path)
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


if __name__ == "__main__":
    main()
