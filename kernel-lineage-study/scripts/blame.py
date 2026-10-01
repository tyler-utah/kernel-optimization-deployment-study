"""Code survival at the cutoff via git blame.

  python blame.py            -> cache/git/blame-<repo>.json  {file: {commit_sha: lines}}
                                data/cutoff-line-survival.csv (repo, lineage, artifact_id, file, commit, lines)

For every file present at the cutoff that matches an in-tree registry artifact's
path_regexes, run `git blame -w -M --line-porcelain <cutoff> -- <file>` and count
surviving lines per originating commit. An event "survives as code" iff at least one
line it introduced is still attributed to it at the cutoff (whole-file renames followed by
blame's rename detection; -C copy detection disabled (it needs blobs outside the lineage pathspecs)).
"""
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402
from survival import GITDIR, CUTOFF, LREPO, tree, registry  # noqa: E402

HEX = re.compile(r"^([0-9a-f]{40}) \d+ \d+")


def blame(repo, path):
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1")
    p = subprocess.run(["git", f"--git-dir={GITDIR[repo]}", "blame", "-w", "-M", "--line-porcelain",
                        CUTOFF[repo], "--", path], capture_output=True, env=env)
    if p.returncode:
        return {"__error__": p.stderr.decode("utf-8", "replace")[:200]}
    c = Counter()
    for line in p.stdout.decode("utf-8", "replace").split("\n"):
        m = HEX.match(line)
        if m:
            c[m.group(1)] += 1
    return dict(c)


def main():
    rows = []
    for repo in ("sglang", "vllm"):
        t = tree(repo, CUTOFF[repo])
        owners = defaultdict(set)  # file -> {(L, artifact)}
        for L, r in LREPO.items():
            if r != repo:
                continue
            for a in registry(L):
                if a.get("kind") in ("upstream_kernel", "dependency_pin"):
                    continue
                rxs = [re.compile(x) for x in a.get("path_regexes", [])]
                for p in t:
                    if any(x.search(p) for x in rxs):
                        owners[p].add((L, a["artifact_id"]))
        cache_fn = os.path.join(S.STUDY, "cache", "git", f"blame-{repo}.json")
        done = json.load(open(cache_fn, encoding="utf-8")) if os.path.exists(cache_fn) else {}
        todo = [p for p in owners if p not in done]
        print(repo, len(owners), "files;", len(todo), "to blame", flush=True)
        with ThreadPoolExecutor(max_workers=8) as ex:
            for p, res in zip(todo, ex.map(lambda p: blame(repo, p), todo)):
                done[p] = res
        json.dump(done, open(cache_fn, "w", encoding="utf-8"))
        for p, res in done.items():
            if p not in owners or "__error__" in res:
                continue
            for (L, aid) in sorted(owners[p]):
                for sha, n in res.items():
                    rows.append({"repo": repo, "lineage": L, "artifact_id": aid, "file": p, "commit": sha, "lines": n})
    out = os.path.join(S.STUDY, "data", "cutoff-line-survival.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["repo", "lineage", "artifact_id", "file", "commit", "lines"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, len(rows))


if __name__ == "__main__":
    main()
