"""WP-C1: artifacts.csv and artifact-snapshots.csv

  python snapshots.py
"""
import csv
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402
from survival import GITDIR, LREPO, releases, tree, in_tree, CUTOFF  # noqa: E402

csv.field_size_limit(10 ** 9)
DATA = os.path.join(S.STUDY, "data")
STAGING = os.path.join(S.STUDY, "staging")


def rcsv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


_BLOB_CACHE = {}


def blobs(repo, commit, paths):
    """(path, blob) for the given paths at commit, from one cached `ls-tree -r` per commit."""
    if not paths:
        return []
    key = (repo, commit)
    if key not in _BLOB_CACHE:
        fn = os.path.join(S.STUDY, "cache", "git", "trees", f"{repo}-{commit[:12]}-blobs.txt")
        if os.path.exists(fn):
            lines = open(fn, encoding="utf-8").read().split("\n")
        else:
            p = subprocess.run(["git", f"--git-dir={GITDIR[repo]}", "ls-tree", "-r", commit], capture_output=True)
            lines = p.stdout.decode("utf-8", "replace").replace("\r\n", "\n").strip().split("\n")
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            open(fn, "w", encoding="utf-8").write("\n".join(lines))
        m = {}
        for line in lines:
            if "\t" in line:
                meta, path = line.split("\t", 1)
                parts = meta.split()
                if len(parts) >= 3:
                    m[path] = parts[2]
        _BLOB_CACHE[key] = m
    m = _BLOB_CACHE[key]
    return sorted((p, m[p]) for p in paths if p in m)


def status_at(a, date):
    st = ""
    for h in sorted(a.get("dispatch_status_history") or [], key=lambda h: h.get("date") or ""):
        if (h.get("date") or "") <= date[:10]:
            st = f"{h.get('status')} ({h.get('scope','')[:60]}; {h.get('ref','')})"
    return st


def main():
    arows, srows = [], []
    pres = rcsv(os.path.join(DATA, "artifact-release-presence.csv"))
    for L in ("L1", "L2", "L3"):
        repo = LREPO[L]
        reg = json.load(open(os.path.join(STAGING, "registry", f"{L}-registry.json"), encoding="utf-8"))["artifacts"]
        succ = {}
        for a in reg:
            for p in a.get("predecessors", []):
                succ.setdefault(p.get("artifact_id"), []).append(f"{a['artifact_id']}:{p.get('relation')}")
        rels = {r["tag"]: r for r in releases(repo)}
        rels["CUTOFF"] = {"tag": "CUTOFF", "commit": CUTOFF[repo], "release_date": S.CUTOFF}
        union = re.compile("|".join(f"(?:{x})" for a in reg if in_tree(a) for x in a["path_regexes"]))
        filtered = {}

        def ftree(commit):
            if commit not in filtered:
                t = tree(repo, commit)
                filtered[commit] = ([p for p in t if union.search(p)], set(t))
            return filtered[commit]
        for a in reg:
            intro, end = a.get("introduced") or {}, a.get("ended") or {}
            arows.append({
                "artifact_id": a["artifact_id"], "lineage": L, "name": a.get("name", ""), "kind": a.get("kind", ""),
                "role": a.get("role", ""), "language": a.get("language", ""), "origin": a.get("origin", ""),
                "upstream": a.get("upstream", ""), "introduced_date": intro.get("date", ""), "introduced_pr": intro.get("pr", ""),
                "introduced_sha": intro.get("sha", ""), "ended_date": end.get("date", "") if end else "",
                "ended_how": end.get("how", "") if end else "", "live_at_cutoff": a.get("live_at_cutoff"),
                "hardware": ";".join(a.get("hardware") or []), "framework_interface": (a.get("framework_interface") or "")[:300],
                "predecessors": ";".join(f"{p.get('artifact_id')}:{p.get('relation')}" for p in a.get("predecessors", [])),
                "successors": ";".join(succ.get(a["artifact_id"], [])),
                "dispatch_status_history": " | ".join(f"{h.get('date')}:{h.get('status')}:{(h.get('scope') or '')[:40]}" for h in a.get("dispatch_status_history") or []),
                "paths_history": " | ".join(f"{ph.get('path')}[{ph.get('from')}..{ph.get('to') or 'live'}]" for ph in a.get("path_history") or []),
                "tests": ";".join(a.get("tests") or []), "benchmarks": ";".join(a.get("benchmarks") or []),
                "added_in": a.get("added_in", "registry"), "in_tree_tracked": int(in_tree(a)),
            })
            if not in_tree(a):
                continue
            rxs = [re.compile(x) for x in a["path_regexes"]]
            test_paths = set((a.get("tests") or []) + (a.get("benchmarks") or []))
            rows = [p for p in pres if p["lineage"] == L and p["artifact_id"] == a["artifact_id"]]
            snap_tags = []
            first = next((p for p in rows if p["present"] == "1"), None)
            if first:
                snap_tags.append((first["release"], "first_release_present"))
            for p in rows:
                if p["status_vs_prev"] in ("modified", "removed") or (p["status_vs_prev"] == "added" and p is not first):
                    snap_tags.append((p["release"], p["status_vs_prev"]))
            snap_tags.append(("CUTOFF", "cutoff"))
            seen = set()
            for tag, why in snap_tags:
                if tag in seen or tag not in rels:
                    continue
                seen.add(tag)
                r = rels[tag]
                sub, allpaths = ftree(r["commit"])
                matched = [p for p in sub if any(x.search(p) for x in rxs)]
                bl = blobs(repo, r["commit"], matched)
                h = hashlib.sha1("\n".join(f"{p}:{b}" for p, b in bl).encode()).hexdigest()[:16] if bl else ""
                tests_present = sorted(test_paths & allpaths)
                srows.append({
                    "artifact_id": a["artifact_id"], "lineage": L, "release": tag, "commit": r["commit"],
                    "release_date": r["release_date"][:10], "snapshot_reason": why, "present": int(bool(matched)),
                    "source_paths": ";".join(matched[:12]) + (f";(+{len(matched) - 12})" if len(matched) > 12 else ""),
                    "implementation_language": a.get("language", ""), "origin": a.get("origin", ""), "upstream": a.get("upstream", ""),
                    "supported_hardware": ";".join(a.get("hardware") or []),
                    "framework_interface": (a.get("framework_interface") or "")[:200],
                    "dispatch_status": status_at(a, r["release_date"]),
                    "tests_benchmarks_present": ";".join(tests_present[:8]),
                    "predecessor_evidence": " | ".join(f"{p.get('artifact_id')}:{p.get('relation')}:{(p.get('evidence') or '')[:80]}" for p in a.get("predecessors", [])),
                    "successors": ";".join(succ.get(a["artifact_id"], [])),
                    "content_hash": h, "blob_ids": ";".join(b[:12] for _, b in bl[:12]),
                })
        print(L, "artifacts", len(reg))
    for name, rows in (("artifacts.csv", arows), ("artifact-snapshots.csv", srows)):
        with open(os.path.join(DATA, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", name, len(rows))


if __name__ == "__main__":
    main()
