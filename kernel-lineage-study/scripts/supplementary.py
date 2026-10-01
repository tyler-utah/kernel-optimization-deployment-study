"""WP-B cross_reference strategy: adjudicate PRs cited by coded relations
(reverts/relands/repairs/ported_from/optimized_from/replaces/reimplementation_of)
that are not yet lineage events.

  python supplementary.py build   -> appends new candidates to staging/candidates.jsonl
                                     (discovery_source cross_reference) and writes
                                     staging/code/<L>/batch-9NN-input.{md,jsonl}
"""
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402
from batches import dossier, load_registries, corpus_flags, render_md  # noqa: E402

STAGING = os.path.join(S.STUDY, "staging")
SLUG = {"sglang": "sgl-project/sglang", "vllm": "vllm-project/vllm"}
REPO_KEY = {v: k for k, v in SLUG.items()}


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build():
    targets = json.load(open(os.path.join(STAGING, "supplementary-targets.json"), encoding="utf-8"))
    cands = jl(os.path.join(STAGING, "candidates.jsonl"))
    cid_set = {c["candidate_id"]: c for c in cands}
    fp = {}
    for repo in SLUG:
        for r in jl(os.path.join(S.STUDY, "cache", "git", f"{repo}-fp.jsonl")):
            if r["pr"] is not None:
                fp.setdefault((SLUG[repo], r["pr"]), r)
    new_cands, batch = [], defaultdict(list)
    for t in targets:
        m = re.match(r"(.+?)#(\d+)$", t["ref"])
        if not m:
            continue
        slug, n = m.group(1), int(m.group(2))
        r = fp.get((slug, n))
        if not r:
            continue
        L = t["lineage"]
        cid = f"{L}-{r['sha'][:10]}"
        if cid in cid_set:
            c = cid_set[cid]
            if "cross_reference" not in c["discovery_source"]:
                c["discovery_source"].append("cross_reference")
        else:
            c = {"candidate_id": cid, "lineage": L, "repo": slug, "date": r["cdate"], "commit_sha": r["sha"],
                 "pr_number": r["pr"], "title": r["subject"], "source_paths": [f[-1] for f in r["files"]][:12],
                 "n_files": len(r["files"]), "discovery_source": ["cross_reference"], "stratum": "X"}
            cands.append(c)
            cid_set[cid] = c
            new_cands.append(c)
        batch[L].append((c, t))
    with open(os.path.join(STAGING, "candidates.jsonl"), "w", encoding="utf-8") as f:
        for c in sorted(cands, key=lambda c: (c["lineage"], c["date"])):
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    # fetch PR light records for new candidates
    import gh
    by_repo = defaultdict(set)
    for c, _ in [x for v in batch.values() for x in v]:
        by_repo[REPO_KEY[c["repo"]]].add(int(c["pr_number"]))
    for repo, nums in by_repo.items():
        gh.prs_light(repo, nums)
    regs, flags = load_registries(), corpus_flags()
    for L, items in batch.items():
        d = os.path.join(STAGING, "code", L)
        recs = []
        for c, t in sorted(items, key=lambda x: x[0]["date"]):
            rec = dossier(c, regs, flags, "code")
            rec["stage1"] = (f"CROSS-REFERENCE target ({t['status']}); cited by {', '.join(t['cited_by'][:4])}. "
                             f"Decide whether this change is itself a verified {L} lineage event (it may be in another repository).")
            recs.append(rec)
        stem = os.path.join(d, "batch-900-input")
        with open(stem + ".jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(stem + ".md", "w", encoding="utf-8") as f:
            f.write(render_md(recs, "code"))
        print(L, len(recs), "supplementary records ->", stem)
    print("new candidates added:", len(new_cands))


if __name__ == "__main__":
    build()
