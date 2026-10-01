"""Validate and summarize staging/registry/<L>-registry.json files.

  python registry_check.py [L1 L2 L3]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
REQ = ["artifact_id", "name", "kind", "role", "language", "origin", "path_history", "key_symbols", "introduced",
       "ended", "live_at_cutoff", "predecessors", "hardware", "framework_interface", "dispatch_status_history",
       "tests", "benchmarks", "path_regexes"]


def check(L, verbose=True):
    fn = os.path.join(STUDY, "staging", "registry", f"{L}-registry.json")
    d = json.load(open(fn, encoding="utf-8"))
    arts = d["artifacts"]
    ids = [a["artifact_id"] for a in arts]
    problems = []
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids")
    idset = set(ids)
    for a in arts:
        miss = [k for k in REQ if k not in a]
        if miss:
            problems.append(f"{a.get('artifact_id')}: missing {miss}")
        for rx in a.get("path_regexes", []):
            try:
                re.compile(rx)
            except re.error as e:
                problems.append(f"{a['artifact_id']}: bad regex {rx}: {e}")
        for p in a.get("predecessors", []):
            if p.get("artifact_id") not in idset:
                problems.append(f"{a['artifact_id']}: predecessor {p.get('artifact_id')} not in registry")
        if not a.get("path_regexes") and a.get("origin", "").startswith("in_tree"):
            problems.append(f"{a['artifact_id']}: no path_regexes")
    if verbose:
        for a in arts:
            intro = a.get("introduced") or {}
            end = a.get("ended") or {}
            print(f"{a['artifact_id']:48s} {a['kind'][:10]:10s} {a['origin'][:14]:14s} {str(intro.get('date'))[:10]} #{intro.get('pr')} "
                  f"end={str(end.get('date'))[:10] if end else '-'} live={a.get('live_at_cutoff')} paths={len(a.get('path_history', []))} rx={len(a.get('path_regexes', []))}")
    print(L, len(arts), "artifacts;", len(d.get("excluded_artifacts", [])), "excluded;", len(d.get("unresolved", [])), "unresolved;",
          len(problems), "problems")
    for p in problems[:30]:
        print("  PROBLEM", p)
    return d


if __name__ == "__main__":
    for L in sys.argv[1:] or ["L1", "L2", "L3"]:
        if os.path.exists(os.path.join(STUDY, "staging", "registry", f"{L}-registry.json")):
            check(L)
