"""Path census helpers over cache/git/<repo>-fp.jsonl.

python paths.py <repo> <regex> [--min N]
  prints every path ever touched matching regex with first/last touch date,
  number of commits, and whether it exists at the cutoff (last status != D).
"""
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)


def load(repo):
    with open(os.path.join(STUDY, "cache", "git", f"{repo}-fp.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def census(recs, rx):
    rx = re.compile(rx)
    info = defaultdict(lambda: {"n": 0, "first": None, "last": None, "last_status": None, "first_sha": None, "last_sha": None})
    for r in reversed(recs):  # chronological (oldest first)
        for fe in r["files"]:
            p = fe[-1]
            if not rx.search(p):
                continue
            d = info[p]
            d["n"] += 1
            if d["first"] is None:
                d["first"], d["first_sha"] = r["cdate"][:10], r["sha"][:10]
            d["last"], d["last_sha"], d["last_status"] = r["cdate"][:10], r["sha"][:10], fe[0]
    return info


if __name__ == "__main__":
    repo, rx = sys.argv[1], sys.argv[2]
    mn = int(sys.argv[sys.argv.index("--min") + 1]) if "--min" in sys.argv else 1
    info = census(load(repo), rx)
    for p, d in sorted(info.items(), key=lambda kv: kv[1]["first"]):
        if d["n"] >= mn:
            alive = "DEL" if d["last_status"] == "D" else "live"
            print(f"{d['first']} {d['last']} {d['n']:4d} {alive:4s} {p}")
    print(len(info), "paths")
