"""WP-F: release-pinned artifact presence/change and move-signature presence.

  python survival.py presence            -> data/artifact-release-presence.csv
  python survival.py overlaps            -> print artifacts whose regexes claim the same cutoff path
  python survival.py moves               -> data/move-release-presence.csv (needs staging/move-signatures.json)

Presence of an in-tree artifact at a release = at least one file in the release tree
matches one of its registry path_regexes. Change across a transition = at least one
matched file differs between the two release trees (git diff --name-only).
Upstream-only artifacts (no in-tree path) are excluded from path presence.
"""
import csv
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402
from study_config import CUTOFF_COMMITS  # noqa: E402

GITDIR = {r: os.path.join(S.STUDY, "cache", "git", f"{r}.git") for r in ("sglang", "vllm")}
CUTOFF = CUTOFF_COMMITS
LREPO = {"L1": "sglang", "L2": "sglang", "L3": "vllm"}
TREE_CACHE = os.path.join(S.STUDY, "cache", "git", "trees")


def git(repo, *args):
    p = subprocess.run(["git", f"--git-dir={GITDIR[repo]}", *args], capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode("utf-8", "replace"))
    return p.stdout.decode("utf-8", "replace").replace("\r\n", "\n")


def tree(repo, commit):
    os.makedirs(TREE_CACHE, exist_ok=True)
    fn = os.path.join(TREE_CACHE, f"{repo}-{commit[:12]}.txt")
    if os.path.exists(fn):
        return open(fn, encoding="utf-8").read().split("\n")
    t = git(repo, "ls-tree", "-r", "--name-only", commit).strip().split("\n")
    open(fn, "w", encoding="utf-8").write("\n".join(t))
    return t


def releases(repo):
    rel = json.load(open(os.path.join(S.STUDY, "cache", "git", f"{repo}-release-index.json"), encoding="utf-8"))
    finals = [r for r in rel if r["final"]]
    finals.sort(key=lambda r: r["release_date"])
    finals.append({"tag": "CUTOFF", "commit": CUTOFF[repo], "release_date": S.CUTOFF, "final": True})
    return finals


def registry(L):
    d = json.load(open(os.path.join(S.STUDY, "staging", "registry", f"{L}-registry.json"), encoding="utf-8"))
    return d["artifacts"]


def in_tree(a):
    """In-tree code unit: kernels, adapters, backends, runners, dispatchers, build rules, reference impls.
    Upstream kernels and dependency pins are tracked through their adapters/pins, not by path presence."""
    return a.get("kind") not in ("upstream_kernel", "dependency_pin") and bool(a.get("path_regexes"))


def presence():
    rows = []
    for L in ("L1", "L2", "L3"):
        repo = LREPO[L]
        arts = [a for a in registry(L) if in_tree(a) and a.get("path_regexes")]
        comp = {a["artifact_id"]: [re.compile(x) for x in a["path_regexes"]] for a in arts}
        union = re.compile("|".join(f"(?:{x})" for a in arts for x in a["path_regexes"]))
        rels = releases(repo)
        prev_commit, prev_match = None, None
        for r in rels:
            t = [p for p in tree(repo, r["commit"]) if union.search(p)]
            match = {aid: sorted(p for p in t if any(x.search(p) for x in rxs)) for aid, rxs in comp.items()}
            changed = set()
            if prev_commit:
                changed = set(git(repo, "diff", "--name-only", "--no-renames", prev_commit, r["commit"]).split("\n"))
            for aid, ps in match.items():
                was = bool(prev_match and prev_match.get(aid))
                rows.append({"lineage": L, "repo": repo, "artifact_id": aid, "release": r["tag"],
                             "release_date": r["release_date"][:10], "present": int(bool(ps)), "n_files": len(ps),
                             "changed_since_prev": int(bool(ps) and was and any(p in changed for p in ps + prev_match[aid])),
                             "status_vs_prev": ("absent" if not ps and not was else "added" if ps and not was else
                                                "removed" if was and not ps else
                                                "modified" if any(p in changed for p in ps + prev_match[aid]) else "carried")})
            prev_commit, prev_match = r["commit"], match
        print(L, len(arts), "in-tree artifacts x", len(rels), "releases")
    out = os.path.join(S.STUDY, "data", "artifact-release-presence.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, len(rows))


def overlaps():
    for L in ("L1", "L2", "L3"):
        repo = LREPO[L]
        t = tree(repo, CUTOFF[repo])
        claims = defaultdict(list)
        for a in registry(L):
            for p in t:
                if any(re.search(x, p) for x in a.get("path_regexes", [])):
                    claims[p].append(a["artifact_id"])
        multi = {p: ids for p, ids in claims.items() if len(ids) > 1}
        print(L, "cutoff paths claimed:", len(claims), "multi-claimed:", len(multi))
        for p, ids in sorted(multi.items())[:40]:
            print("   ", p, ids)


def _scan_move(L, repo, m, sig, cfn):
    rx, specs = sig["regex"], sig.get("pathspec") or []
    minf = int(sig.get("min_files") or 1)
    mrows = []
    for r in releases(repo):
        cmd = ["git", f"--git-dir={GITDIR[repo]}", "grep", "-l", "-E", rx, r["commit"], "--", *specs]
        p = subprocess.run(cmd, capture_output=True, env=dict(os.environ, GIT_NO_LAZY_FETCH="1"))
        if p.returncode not in (0, 1):
            # A missing blob aborts git grep; an empty result would be a false "absent".
            # Retry once allowing lazy fetch, and never cache a failed scan.
            sys.stderr.write(f"grep retry (lazy fetch) {m['move_id']} {r['tag']}\n")
            p = subprocess.run(cmd, capture_output=True, timeout=1800)
            if p.returncode not in (0, 1):
                raise RuntimeError(f"git grep failed for {m['move_id']} at {r['tag']}: "
                                   f"{p.stderr.decode('utf-8', 'replace')[:300]}")
        hits = [x for x in p.stdout.decode("utf-8", "replace").split("\n") if x.strip()]
        mrows.append({"lineage": L, "repo": repo, "move_id": m["move_id"], "release": r["tag"],
                      "release_date": r["release_date"][:10], "present": int(len(hits) >= minf), "n_files": len(hits),
                      "files": ";".join(h.split(":", 1)[-1] for h in hits[:6])})
    with open(cfn, "w", encoding="utf-8") as f:
        json.dump(mrows, f)
    return mrows


def move_presence(rescan=False):
    """rescan=True ignores cached scans (use after prefetching blobs for new signature paths)."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cache_dir = os.path.join(S.STUDY, "cache", "git", "move-scan")
    os.makedirs(cache_dir, exist_ok=True)
    # seed cache from a previous full CSV run (same signature => same result)
    prev_fn = os.path.join(S.STUDY, "data", "move-release-presence.csv")
    prev = defaultdict(list)
    if os.path.exists(prev_fn) and not rescan:
        for r in csv.DictReader(open(prev_fn, encoding="utf-8")):
            prev[r["move_id"]].append(r)
    catalogs = []
    for L in ("L1", "L2", "L3"):
        fin = os.path.join(S.STUDY, "staging", "moves", f"{L}-moves-final.json")
        base = os.path.join(S.STUDY, "staging", "moves", f"{L}-moves.json")
        catalogs.append(fin if os.path.exists(fin) else base)
    base_sigs = {}
    for L in ("L1", "L2", "L3"):
        base = os.path.join(S.STUDY, "staging", "moves", f"{L}-moves.json")
        if os.path.exists(base):
            for m in json.load(open(base, encoding="utf-8"))["moves"]:
                base_sigs[m["move_id"]] = json.dumps(m.get("signature") or {}, sort_keys=True)
    results, jobs = {}, []
    for fn in catalogs:
        d = json.load(open(fn, encoding="utf-8"))
        L = d["lineage"]
        for m in d["moves"]:
            sig = m.get("signature") or {}
            if not sig.get("regex") or not sig.get("pathspec"):
                continue
            idx = len(results) + len(jobs)
            key = hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:12]
            cfn = os.path.join(cache_dir, f"{m['move_id']}-{key}.json")
            if not rescan and os.path.exists(cfn):
                results[idx] = json.load(open(cfn, encoding="utf-8"))
                continue
            if not rescan and prev.get(m["move_id"]) and base_sigs.get(m["move_id"]) == json.dumps(sig, sort_keys=True):
                results[idx] = prev[m["move_id"]]
                with open(cfn, "w", encoding="utf-8") as f:
                    json.dump(results[idx], f)
                continue
            jobs.append((idx, L, LREPO[L], m, sig, cfn))
    workers = int(os.environ.get("MOVE_SCAN_WORKERS", "6"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_scan_move, L, repo, m, sig, cfn): (idx, L, m["move_id"]) for idx, L, repo, m, sig, cfn in jobs}
        for fu in as_completed(futs):
            idx, L, mid = futs[fu]
            results[idx] = fu.result()
            print(L, mid, "scanned", flush=True)
    rows = [row for idx in sorted(results) for row in results[idx]]
    out = os.path.join(S.STUDY, "data", "move-release-presence.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, len(rows))


if __name__ == "__main__":
    import glob  # noqa: F401
    cmd = sys.argv[1]
    if cmd == "presence":
        presence()
    elif cmd == "overlaps":
        overlaps()
    elif cmd == "moves":
        move_presence(rescan="--rescan" in sys.argv)
