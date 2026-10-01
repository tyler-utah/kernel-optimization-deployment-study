"""A5: select candidate PRs for 12 verified case histories (seeded, reproducible).

Slots per repository: 2 fast merges, 2 slow/many-round merges, 1 reverted, 1 superseded/abandoned.
Three ranked candidates per slot are written so the verifying agent can skip a
candidate whose public record is too thin; the final choice is recorded.
"""

from __future__ import annotations

import collections

from common import CACHE, DATA, REPOS, atomic_csv, iso, read_csv, read_jsonl, rng


def main() -> None:
    corpus = [r for r in read_csv(DATA / "a3-detailed-corpus.csv") if r["group"] == "performance"]
    meta = {}
    for repo in REPOS:
        for r in read_jsonl(CACHE / "a1" / f"population-{repo}.jsonl"):
            meta[(repo, r["number"])] = r
    met = {}
    for repo in REPOS:
        for m in read_jsonl(CACHE / "a3" / f"metrics-{repo}.jsonl"):
            met[(repo, m["number"])] = m
    popmeta = {(r["repo"], int(r["number"])): r for r in read_csv(DATA / "pr-population-metadata.csv")}
    reverted = set()
    for r in read_csv(DATA / "confirmed-reverts.csv"):
        if r["is_revert_or_rollback"] == "1":
            for n in filter(None, r["reverted_prs"].split(";")):
                reverted.add((r["repo"], int(n)))
    rows = []
    for repo in REPOS:
        pool = collections.defaultdict(list)
        for c in corpus:
            if c["repo"] != repo:
                continue
            k = (repo, int(c["number"]))
            m, mm, pm = meta[k], met.get(k), popmeta.get(k)
            if not mm or not pm:
                continue
            rich = mm["n_substantive_reviewer_msgs"] >= 2 and len(m["body_clean"]) >= 400
            days = float(pm["duration_days"])
            if c["state"] == "merged" and k in reverted and rich:
                pool["reverted"].append(k)
            elif c["state"] == "merged" and days <= 2 and m["sig_kernel_path"] and rich:
                pool["fast"].append(k)
            elif c["state"] == "merged" and (days >= 60 or mm["substantive_review_rounds"] >= 8) and m["sig_kernel_path"] and rich:
                pool["slow"].append(k)
            elif c["state"] == "closed" and rich and m["sig_kernel_path"]:
                pool["superseded_or_abandoned" if pm["superseded_signal"] == "1" else "abandoned"].append(k)
        slots = [("fast", 2), ("slow", 2), ("reverted", 1), ("superseded_or_abandoned", 1)]
        for slot, n in slots:
            cand = sorted(pool[slot]) or sorted(pool["abandoned"])
            picks = rng(f"a5-{repo}-{slot}").sample(cand, min(len(cand), 3 * n))
            for i, k in enumerate(picks):
                pm, mm = popmeta[k], met[k]
                rows.append({"repo": repo, "slot": slot, "slot_index": i // 3, "rank": i % 3, "number": k[1],
                             "title": meta[k]["title"], "state": pm["state"], "duration_days": pm["duration_days"],
                             "substantive_review_rounds": mm["substantive_review_rounds"], "n_human_reviewers": mm["n_human_reviewers"],
                             "reverted": int(k in reverted), "superseded_signal": pm["superseded_signal"],
                             "url": f"https://github.com/{REPOS[repo]}/pull/{k[1]}", "pool_size": len(cand)})
    atomic_csv(DATA / "a5-case-candidates.csv", rows)
    print(collections.Counter((r["repo"], r["slot"]) for r in rows))


if __name__ == "__main__":
    main()
