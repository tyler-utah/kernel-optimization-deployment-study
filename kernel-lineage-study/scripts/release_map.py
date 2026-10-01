"""WP-F: release map and first-containing-release assignment.

  python release_map.py            -> data/release-map.csv, cache/git/<repo>-release-index.json

Release definition: a GitHub Release (non-draft) whose tag exists locally,
published at or before the cutoff. Final releases (not marked prerelease and
without rc/a/b suffix) form the release sequence used for transitions;
pre-releases are recorded but flagged.

Containment of a main (first-parent) commit C in release tag T:
  * ancestry: C is an ancestor of merge-base(T, main)  -> 'branch_point'
  * cherry-pick: T's side branch (merge-base..T) contains a commit whose
    subject carries the same PR number as C                -> 'cherry_pick'
The first containing release is the earliest-published final release that
contains C by either route.
"""
import csv
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402
import gh  # noqa: E402
from study_config import CUTOFF_COMMITS, CUTOFF_ISO  # noqa: E402

GITDIR = {r: os.path.join(S.STUDY, "cache", "git", f"{r}.git") for r in ("sglang", "vllm")}
CUTOFF_COMMIT = CUTOFF_COMMITS
PR_RE = re.compile(r"\(#(\d+)\)")
PRE_RE = re.compile(r"(rc\d*|a\d+|b\d+|dev\d*)$")


def _utc(s):
    from datetime import datetime, timezone
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _add_days(iso, d):
    from datetime import datetime, timedelta, timezone
    return (datetime.fromisoformat(iso.replace("Z", "+00:00")) + timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(repo, *args):
    p = subprocess.run(["git", f"--git-dir={GITDIR[repo]}", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode:
        raise RuntimeError(p.stderr)
    return p.stdout


def fp_list(repo):
    with open(os.path.join(S.STUDY, "cache", "git", f"{repo}-fp.jsonl"), encoding="utf-8") as f:
        recs = [json.loads(l) for l in f]
    return recs  # newest first


def build(repo):
    slug = gh.REPO_SLUG[repo]
    rels = gh.rest(f"repos/{slug}/releases?per_page=100", paginate=True)
    recs = fp_list(repo)
    idx = {r["sha"]: i for i, r in enumerate(recs)}  # 0 = newest
    pr_to_idx = {}
    for i, r in enumerate(recs):
        if r["pr"] is not None and r["pr"] not in pr_to_idx:
            pr_to_idx[r["pr"]] = i
    rows = []
    for rel in rels:
        if rel.get("draft"):
            continue
        tag = rel["tag_name"]
        if not tag.startswith("v"):
            continue  # gateway-v*, proto-v*: separate products
        pub = rel.get("published_at")
        if not pub or pub > CUTOFF_ISO:
            continue
        try:
            commit = git(repo, "rev-parse", f"{tag}^{{commit}}").strip()
        except RuntimeError:
            rows.append({"repo": repo, "tag": tag, "commit": "", "note": "tag missing locally"})
            continue
        cdate = git(repo, "log", "-1", "--format=%cI", commit).strip()
        mb = git(repo, "merge-base", commit, CUTOFF_COMMIT[repo]).strip()
        on_main = mb == commit
        side = [] if on_main else git(repo, "log", "--format=%H\x1f%s", f"{mb}..{commit}").splitlines()
        cherry_prs = sorted({int(m) for line in side for m in PR_RE.findall(line.split("\x1f", 1)[-1])})
        prerelease = bool(rel.get("prerelease")) or bool(PRE_RE.search(tag))
        tagger = git(repo, "for-each-ref", f"refs/tags/{tag}", "--format=%(taggerdate:iso-strict)").strip()
        release_date = pub
        date_source = "github_published_at"
        if tagger and _utc(tagger) < pub:
            release_date, date_source = _utc(tagger), "annotated_tag_date"
        # tag re-pointed after publication: cap ancestry containment at the release date
        moved = _utc(cdate) > _add_days(release_date, 0.5)
        eff_bp = idx.get(mb)
        if moved and eff_bp is not None:
            j = eff_bp
            while j < len(recs) and _utc(recs[j]["cdate"]) > release_date:
                j += 1
            eff_bp = j
        rows.append({
            "release_date": release_date, "release_date_source": date_source,
            "tag_moved_after_publication": moved, "effective_branch_point_fp_index": eff_bp,
            "repo": repo, "tag": tag, "commit": commit, "commit_date": cdate,
            "published_at": pub, "prerelease": prerelease, "final": not prerelease,
            "on_main": on_main, "branch_point": mb, "branch_point_fp_index": idx.get(mb),
            "branch_point_date": recs[idx[mb]]["cdate"] if mb in idx else "",
            "side_commits": len(side), "cherry_pick_prs": cherry_prs,
            "release_url": rel.get("html_url"), "name": rel.get("name") or "",
        })
    rows = [r for r in rows if r.get("commit")]
    rows.sort(key=lambda r: r["release_date"])
    finals = [r for r in rows if r["final"]]
    for i, r in enumerate(finals):
        r["seq"] = i + 1
        r["prev_final"] = finals[i - 1]["tag"] if i else ""
    out = os.path.join(S.STUDY, "cache", "git", f"{repo}-release-index.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    return rows


def first_release(repo, sha, pr, rows, recs_idx):
    """Return (tag, route, published_at) of earliest final release containing sha/pr."""
    i = recs_idx.get(sha)
    best = None
    for r in rows:
        if not r["final"]:
            continue
        route = None
        bp = r["effective_branch_point_fp_index"]
        if i is not None and bp is not None and i >= bp:
            route = "branch_point"
        elif pr is not None and pr in r["cherry_pick_prs"]:
            route = "cherry_pick"
        if route and (best is None or r["release_date"] < best[2]):
            best = (r["tag"], route + ("_tag_moved" if r["tag_moved_after_publication"] else ""), r["release_date"])
    return best


def write_csv(all_rows):
    path = os.path.join(S.STUDY, "data", "release-map.csv")
    cols = ["repo", "tag", "seq", "final", "prerelease", "release_date", "release_date_source", "published_at",
            "tag_moved_after_publication", "commit", "commit_date", "on_main",
            "branch_point", "branch_point_date", "side_commits", "n_cherry_pick_prs", "prev_final", "release_url"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            d = {k: r.get(k, "") for k in cols}
            d["n_cherry_pick_prs"] = len(r.get("cherry_pick_prs", []))
            w.writerow(d)
    print("wrote", path, len(all_rows))


if __name__ == "__main__":
    allrows = []
    for repo in ("sglang", "vllm"):
        rows = build(repo)
        print(repo, len(rows), "releases;", sum(r["final"] for r in rows), "final;",
              sum(1 for r in rows if not r["on_main"]), "off-main tags")
        allrows += rows
    write_csv(allrows)
