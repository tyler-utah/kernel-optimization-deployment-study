"""Prefetch every blob reachable at release commits (and the cutoff) under the
move-signature pathspecs and the in-tree artifact paths, so that git grep / blame
never trigger per-blob lazy fetches.

  python prefetch_release_blobs.py
"""
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402
from blobs import GITDIR, missing, fetch  # noqa: E402
from survival import releases, LREPO  # noqa: E402


def ls_blobs(repo, commit, specs):
    p = subprocess.run(["git", f"--git-dir={GITDIR[repo]}", "ls-tree", "-r", commit, "--", *specs], capture_output=True)
    out = set()
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "blob":
            out.add(parts[2])
    return out


if __name__ == "__main__":
    specs = {"sglang": set(), "vllm": set()}
    for fn in (glob.glob(os.path.join(S.STUDY, "staging", "moves", "*-moves.json"))
               + glob.glob(os.path.join(S.STUDY, "staging", "moves", "*-moves-final.json"))):
        d = json.load(open(fn, encoding="utf-8"))
        for m in d["moves"]:
            for sp in (m.get("signature") or {}).get("pathspec") or []:
                specs[LREPO[d["lineage"]]].add(sp)
    for repo in specs:
        want = set()
        # ls-tree does not understand :(glob) magic reliably; strip magic and globs to directory prefixes
        clean = set()
        for sp in specs[repo]:
            sp = re.sub(r"^:\([^)]*\)", "", sp)
            sp = re.split(r"[*?\[]", sp)[0].rstrip("/")
            if sp:
                clean.add(sp)
        rels = releases(repo)
        for r in rels:
            want |= ls_blobs(repo, r["commit"], sorted(clean))
        miss = missing(repo, sorted(want))
        print(repo, "pathspecs", len(clean), "blobs at releases", len(want), "missing", len(miss), flush=True)
        if miss:
            S.log(f"network batch start: release-blob prefetch {repo} ({len(miss)})")
            fetch(repo, miss)
            S.log(f"network batch end: release-blob prefetch {repo}")
