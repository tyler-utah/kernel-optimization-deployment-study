"""A2: full-population PR metadata analysis (complete 12-month population).

run - joins the adjudicated population with metadata, reverts and superseded
      signals; Kaplan-Meier (open censored at cutoff; closed-unmerged censored
      at closure) and Aalen-Johansen cumulative incidence of merge (closure as
      competing event); 1,000-replicate bootstrap CIs; queue age; figures.
"""

from __future__ import annotations

import collections
import re

import numpy as np

from common import (A_START, CACHE, CUTOFF, DATA, FIG, REPOS, atomic_csv, iso, read_csv, read_jsonl)
from a1_metadata import load_enriched

B = 1000


def zseed(*parts) -> int:
    import zlib
    return zlib.crc32(repr(parts).encode()) & 0xFFFFFFFF

HORIZONS = [1, 7, 30, 90]
SUPERSEDE_RE = re.compile(r"supersed|replaced by|in favou?r of|duplicate of|continued in|moved to #|landed (?:in|via)|"
                          r"merged (?:in|via|as part of) #|covered by #|folded into|re-?opened as|new pr|closing (?:this )?(?:as|since|because).{0,60}#\d+", re.I)


def km_curve(t: np.ndarray, e: np.ndarray):
    order = np.argsort(t, kind="stable")
    t, e = t[order], e[order]
    uniq, idx = np.unique(t, return_index=True)
    n = len(t)
    at_risk = n - idx
    events = np.add.reduceat(e, idx)
    s = np.cumprod(1 - events / at_risk)
    return uniq, s


def km_at(t, e, h):
    u, s = km_curve(t, e)
    k = np.searchsorted(u, h, side="right") - 1
    return 1.0 if k < 0 else float(s[k])


def km_median(t, e):
    u, s = km_curve(t, e)
    k = np.nonzero(s <= 0.5)[0]
    return float(u[k[0]]) if len(k) else float("nan")


def cif(t: np.ndarray, cause: np.ndarray, h: float, which: int = 1) -> float:
    """Aalen-Johansen CIF for cause `which` (1=merge, 2=closed); 0=censored."""
    order = np.argsort(t, kind="stable")
    t, c = t[order], cause[order]
    uniq, idx = np.unique(t, return_index=True)
    at_risk = len(t) - idx
    d_any = np.add.reduceat((c > 0).astype(float), idx)
    d_k = np.add.reduceat((c == which).astype(float), idx)
    s_prev = np.concatenate([[1.0], np.cumprod(1 - d_any / at_risk)[:-1]])
    inc = s_prev * d_k / at_risk
    mask = uniq <= h
    return float(inc[mask].sum())


def rmst(t, e, h):
    u, s = km_curve(t, e)
    u2 = np.concatenate([[0.0], u[u <= h], [h]])
    s2 = np.concatenate([[1.0], s[u <= h]])
    return float(np.sum(np.diff(u2) * s2))


def boot(fn, groups: dict[str, tuple], seed: int):
    rs = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        res = {}
        for g, arrs in groups.items():
            n = len(arrs[0])
            ix = rs.integers(0, n, n)
            res[g] = tuple(a[ix] for a in arrs)
        out.append(fn(res))
    a = np.array(out, dtype=float)
    a = a[~np.isnan(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (float("nan"), float("nan"))


def load(repo: str) -> list[dict]:
    pop = {int(r["number"]): r for r in read_csv(DATA / "performance-pr-population.csv") if r["repo"] == repo}
    meta = {r["number"]: r for r in read_jsonl(CACHE / "a1" / f"population-{repo}.jsonl")}
    enr = load_enriched(repo)
    reverted = collections.defaultdict(list)
    for r in read_csv(DATA / "confirmed-reverts.csv"):
        if r["repo"] == repo and r["is_revert_or_rollback"] == "1":
            for n in filter(None, r["reverted_prs"].split(";")):
                reverted[int(n)].append(r["revert_date"])
    rows = []
    for n, p in pop.items():
        m = meta[n]
        node = enr.get(n, {})
        created = iso(m["created"])
        state = m["state"]
        end = iso(m["merged"]) if state == "merged" else (iso(m["closed"]) if state == "closed" else CUTOFF)
        dur_days = (end - created).total_seconds() / 86400
        sup = 0
        if state == "closed":
            for ev in ((node.get("timelineItems") or {}).get("nodes") or []):
                if ev.get("__typename") == "ClosedEvent" and (ev.get("closer") or {}).get("__typename") in ("PullRequest", "Commit"):
                    sup = 1
                if ev.get("__typename") == "CrossReferencedEvent" and ev.get("willCloseTarget"):
                    sup = 1
            for c in ((node.get("comments") or {}).get("nodes") or []):
                if SUPERSEDE_RE.search(c.get("body") or ""):
                    sup = 1
        rows.append({
            "repo": repo, "number": n, "classification": p["classification"], "tier": m["tier"], "in_window": m["in_window"],
            "state": state, "created": m["created"], "merged": m["merged"], "closed": m["closed"],
            "author_type": m["author_type"], "additions": m["additions"], "deletions": m["deletions"],
            "changed_files": m["changed_files"], "labels": ";".join(m["labels"]),
            "duration_days": round(dur_days, 4),
            "event_merged": int(state == "merged"), "event_closed": int(state == "closed"),
            "reverted": int(n in reverted), "superseded_signal": sup,
            "queue_age_days": round(dur_days, 3) if state == "open" else "",
        })
    return rows


def summarize(rows: list[dict], repo: str) -> tuple[list, list, list]:
    groups = {"performance": [r for r in rows if r["classification"] == "confirmed_performance" and r["in_window"]],
              "non_performance": [r for r in rows if r["classification"] == "not_performance" and r["in_window"]]}
    summary, surv, effects = [], [], []
    arr = {}
    for g, rs in groups.items():
        t = np.array([r["duration_days"] for r in rs], float)
        e = np.array([r["event_merged"] for r in rs], float)
        cause = np.array([1 if r["event_merged"] else (2 if r["event_closed"] else 0) for r in rs], int)
        arr[g] = (t, e, cause)
        size = np.array([(r["additions"] or 0) + (r["deletions"] or 0) for r in rs if r["additions"] is not None], float)
        files = np.array([r["changed_files"] or 0 for r in rs if r["changed_files"] is not None], float)
        n = len(rs)
        st = collections.Counter(r["state"] for r in rs)
        labels = collections.Counter(l for r in rs for l in r["labels"].split(";") if l)
        summary.append({
            "repo": repo, "group": g, "n": n, "merged": st["merged"], "closed_unmerged": st["closed"], "open": st["open"],
            "merged_share": round(st["merged"] / n, 4), "closed_share": round(st["closed"] / n, 4), "open_share": round(st["open"] / n, 4),
            "median_lines_changed": float(np.median(size)) if len(size) else "", "median_changed_files": float(np.median(files)) if len(files) else "",
            "bot_authored_share": round(sum(r["author_type"] == "bot" for r in rs) / n, 4),
            "reverted_n": sum(r["reverted"] for r in rs), "reverted_share_of_merged": round(sum(r["reverted"] for r in rs if r["state"] == "merged") / max(1, st["merged"]), 4),
            "superseded_signal_n": sum(r["superseded_signal"] for r in rs),
            "superseded_signal_share_of_closed": round(sum(r["superseded_signal"] for r in rs) / max(1, st["closed"]), 4),
            "km_median_days_to_merge": round(km_median(t, e), 3),
            "median_days_to_merge_among_merged": round(float(np.median(t[e == 1])), 3) if e.sum() else "",
            "top_labels": "; ".join(f"{k} ({v})" for k, v in labels.most_common(8)),
        })
        for h in HORIZONS:
            lo_hi = boot(lambda res, h=h, g=g: cif(res[g][0], res[g][2], h), {g: arr[g]}, seed=zseed(repo, g, h))
            surv.append({"repo": repo, "group": g, "horizon_days": h,
                         "km_merged_prob_closed_censored": round(1 - km_at(t, e, h), 4),
                         "cif_merged": round(cif(t, cause, h), 4), "cif_merged_ci_low": round(lo_hi[0], 4), "cif_merged_ci_high": round(lo_hi[1], 4),
                         "cif_closed": round(cif(t, cause, h, 2), 4)})
    P, N = "performance", "non_performance"
    specs = [
        ("km_median_days_to_merge_diff", lambda a: km_median(a[P][0], a[P][1]) - km_median(a[N][0], a[N][1])),
        ("km_median_days_to_merge_ratio", lambda a: km_median(a[P][0], a[P][1]) / max(1e-9, km_median(a[N][0], a[N][1]))),
        ("cif_merged_30d_diff", lambda a: cif(a[P][0], a[P][2], 30) - cif(a[N][0], a[N][2], 30)),
        ("cif_merged_7d_diff", lambda a: cif(a[P][0], a[P][2], 7) - cif(a[N][0], a[N][2], 7)),
        ("rmst_90d_days_diff", lambda a: rmst(a[P][0], a[P][1], 90) - rmst(a[N][0], a[N][1], 90)),
        ("closed_unmerged_share_diff", lambda a: float((a[P][2] == 2).mean() - (a[N][2] == 2).mean())),
    ]
    for name, fn in specs:
        est = fn(arr)
        lo, hi = boot(fn, arr, seed=zseed(repo, name))
        effects.append({"repo": repo, "effect": name, "estimate": round(est, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                        "n_performance": len(arr[P][0]), "n_non_performance": len(arr[N][0]), "bootstrap_reps": B,
                        "stratum": "all"})
    # size-stratified KM medians (descriptive control for PR size)
    sizes = np.array([(r["additions"] or 0) + (r["deletions"] or 0) for g in groups.values() for r in g], float)
    cuts = np.percentile(sizes, [33.3, 66.7])
    for lo_c, hi_c, lab in [(-1, cuts[0], "small"), (cuts[0], cuts[1], "medium"), (cuts[1], 1e12, "large")]:
        sub = {}
        for g, rs in groups.items():
            rr = [r for r in rs if lo_c < (r["additions"] or 0) + (r["deletions"] or 0) <= hi_c]
            sub[g] = (np.array([r["duration_days"] for r in rr], float), np.array([r["event_merged"] for r in rr], float),
                      np.array([1 if r["event_merged"] else (2 if r["event_closed"] else 0) for r in rr], int))
        fn = specs[0][1]
        lo, hi = boot(fn, sub, seed=zseed(repo, lab))
        effects.append({"repo": repo, "effect": "km_median_days_to_merge_diff", "estimate": round(fn(sub), 4), "ci95_low": round(lo, 4),
                        "ci95_high": round(hi, 4), "n_performance": len(sub[P][0]), "n_non_performance": len(sub[N][0]),
                        "bootstrap_reps": B, "stratum": f"size_{lab} (lines {max(0, lo_c):.0f}-{hi_c:.0f})"})
    return summary, surv, effects


def queue(rows: list[dict], repo: str) -> list[dict]:
    out = []
    for g, cls in [("performance", "confirmed_performance"), ("non_performance", "not_performance"), ("uncertain", "uncertain")]:
        ages = np.array([r["duration_days"] for r in rows if r["state"] == "open" and r["classification"] == cls], float)
        if not len(ages):
            continue
        out.append({"repo": repo, "group": g, "open_at_cutoff": len(ages), "median_age_days": round(float(np.median(ages)), 2),
                    "p75_age_days": round(float(np.percentile(ages, 75)), 2), "share_older_than_90d": round(float((ages > 90).mean()), 4),
                    "share_older_than_30d": round(float((ages > 30).mean()), 4)})
    return out


def figures(all_rows: dict[str, list[dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, repo in zip(axes, REPOS):
        for g, cls, col in [("performance", "confirmed_performance", "#d62728"), ("non-performance", "not_performance", "#1f77b4")]:
            rs = [r for r in all_rows[repo] if r["classification"] == cls and r["in_window"]]
            t = np.array([r["duration_days"] for r in rs], float)
            cause = np.array([1 if r["event_merged"] else (2 if r["event_closed"] else 0) for r in rs], int)
            e = (cause == 1).astype(float)
            u, s = km_curve(t, e)
            ax.step(u, 1 - s, where="post", color=col, lw=1.8, label=f"{g} KM (closed censored), n={len(rs):,}")
            hs = np.linspace(0, 120, 241)
            ax.plot(hs, [cif(t, cause, h) for h in hs], color=col, lw=1.2, ls="--", label=f"{g} CIF (closure competing)")
        ax.set_xlim(0, 120)
        ax.set_title(f"{repo}: time to merge (PRs created {A_START.date()}..{CUTOFF.date()})", fontsize=9)
        ax.set_xlabel("days since PR creation")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("share merged")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "a2-time-to-merge-km-cif.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, repo in zip(axes, REPOS):
        for g, cls, col in [("performance", "confirmed_performance", "#d62728"), ("non-performance", "not_performance", "#1f77b4")]:
            ages = [r["duration_days"] for r in all_rows[repo] if r["state"] == "open" and r["classification"] == cls]
            ax.hist(ages, bins=np.arange(0, 400, 10), color=col, alpha=.55, density=True, label=f"{g} (n={len(ages):,})")
        ax.set_title(f"{repo}: age of PRs open at {CUTOFF.date()}", fontsize=9)
        ax.set_xlabel("age (days)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "a2-open-queue-age.png", dpi=160)
    plt.close(fig)


def run() -> None:
    allrows, summ, surv, eff, q = {}, [], [], [], []
    for repo in REPOS:
        rows = load(repo)
        allrows[repo] = rows
        s, v, e = summarize(rows, repo)
        summ += s
        surv += v
        eff += e
        q += queue(rows, repo)
    atomic_csv(DATA / "pr-population-metadata.csv", [r for rs in allrows.values() for r in rs])
    atomic_csv(DATA / "a2-population-summary.csv", summ)
    atomic_csv(DATA / "a2-survival.csv", surv)
    atomic_csv(DATA / "a2-effects.csv", eff)
    atomic_csv(DATA / "a2-open-queue.csv", q)
    figures(allrows)
    for r in summ:
        print({k: r[k] for k in ["repo", "group", "n", "merged_share", "closed_share", "open_share", "km_median_days_to_merge", "reverted_n"]})
    for r in eff:
        print(r["repo"], r["effect"], r["stratum"], r["estimate"], (r["ci95_low"], r["ci95_high"]))


if __name__ == "__main__":
    run()
