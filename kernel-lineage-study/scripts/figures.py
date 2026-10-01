"""WP-H figures (PNG + SVG) and editable DAG sources (DOT + Mermaid).

  python figures.py all
"""
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402
from study_config import CUTOFF_DATE  # noqa: E402

csv.field_size_limit(10 ** 9)
DATA = os.path.join(S.STUDY, "data")
FIG = os.path.join(S.STUDY, "figures")
STAGING = os.path.join(S.STUDY, "staging")
LREPO = {"L1": "sglang", "L2": "sglang", "L3": "vllm"}
LNAME = {"L1": "SGLang MoE alignment / routing / top-k / fusion", "L2": "SGLang MLA / FlashInfer MLA / FlashMLA",
         "L3": "vLLM attention implementation family"}
CUTOFF = datetime(2026, 9, 28, 23, 59)
EV_STYLE = {"introduce": ("^", "#1b7837"), "port": ("^", "#5aae61"), "integrate": ("^", "#a6dba0"),
            "optimize": ("o", "#2166ac"), "retune": ("o", "#92c5de"), "extend_support": (".", "#999999"),
            "adapt_framework": ("s", "#fdb863"), "repair_correctness": ("x", "#d6604d"),
            "repair_performance": ("x", "#f4a582"), "repair_build_dependency": ("x", "#b2abd2"),
            "revert": ("v", "#b2182b"), "reland": ("v", "#ef8a62"), "change_default": ("D", "#762a83"),
            "replace": ("D", "#c2a5cf"), "deprecate": ("P", "#666666"), "remove": ("X", "#000000")}
CAUSE_COL = {"specification": "#1b9e77", "framework_integration": "#d95f02", "hardware_compiler": "#7570b3",
             "performance": "#e7298a", "correctness": "#66a61e", "build_dependency": "#e6ab02", "maintenance": "#a6761d"}
CLASS_COL = {"direct_carry": "#d9d9d9", "parameter_retune": "#9ecae1", "adapter_repair": "#fdae6b",
             "derivation_repair": "#fb6a4a", "manual_reimplementation": "#9e9ac8", "backend_replacement": "#6a51a3",
             "obsolete": "#252525", "unclear": "#ffffff"}
EDGE_COL = {"ported_from": "#1b7837", "reimplementation_of": "#762a83", "replaces": "#b2182b", "forked_from": "#2166ac",
            "wraps": "#999999", "integrates": "#999999", "textual_successor": "#4d4d4d",
            "historical_connection_uncertain": "#bbbbbb", "optimized_from": "#2166ac"}


def rcsv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dt(s):
    s = (s or "")[:10]
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=160 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print("figure", name)


def registry(L):
    return json.load(open(os.path.join(STAGING, "registry", f"{L}-registry.json"), encoding="utf-8"))["artifacts"]


def moves(L):
    fn = os.path.join(STAGING, "moves", f"{L}-moves-final.json")
    if not os.path.exists(fn):
        fn = os.path.join(STAGING, "moves", f"{L}-moves.json")
    return json.load(open(fn, encoding="utf-8"))["moves"] if os.path.exists(fn) else []


def releases(repo):
    rel = json.load(open(os.path.join(S.STUDY, "cache", "git", f"{repo}-release-index.json"), encoding="utf-8"))
    return sorted([r for r in rel if r["final"]], key=lambda r: r["release_date"])


def branch_of(L, aid):
    parts = aid.split(".")
    fam = parts[1] if len(parts) > 1 else aid
    groups = {
        "L1": {"align": "alignment", "triton": "Triton fused MoE", "routing": "routing / top-k", "reduce": "reduction / scaling",
               "ep": "EP / DeepEP", "runner": "runners & backends", "cutlass": "runners & backends", "upstream": "upstream",
               "hardware": "hardware variants", "ref": "Triton fused MoE"},
        "L2": {"model": "MLA forward", "optimization": "MLA forward", "kernel": "kernels", "backend": "backends",
               "upstream": "upstream", "build": "backends", "pool": "KV cache contract", "dispatch": "dispatch",
               "runner": "dispatch"},
        "L3": {"paged": "PagedAttention & cache", "cache": "PagedAttention & cache", "xformers": "FlashAttention & xFormers",
               "flash_attn": "FlashAttention & xFormers", "flashinfer": "FlashInfer & TRT-LLM", "triton": "Triton",
               "merge": "Triton", "rocm": "ROCm / AITER", "mla": "MLA", "dispatch": "dispatch", "platform": "dispatch",
               "flex_attention": "other backends", "tree_attention": "other backends", "blocksparse": "other backends"},
    }
    return groups[L].get(fam, "other")


# ---------------- artifact lane DAGs ----------------
def artifact_lanes(L, events, edges):
    arts = registry(L)
    ev = [e for e in events if e["lineage"] == L]
    by_art = defaultdict(list)
    for e in ev:
        for a in e["artifact_ids"].split(";"):
            if a:
                by_art[a].append(e)
    branches = defaultdict(list)
    for a in arts:
        branches[branch_of(L, a["artifact_id"])].append(a)
    rels = releases(LREPO[L])
    order = [b for b in branches if b != "upstream"] + (["upstream"] if "upstream" in branches else [])
    # split into pages of <= 18 lanes so each figure stays legible
    pages, cur = [], []
    for b in order:
        if cur and len(cur) + len(branches[b]) > 18:
            pages.append(cur)
            cur = []
        cur += [(b, a) for a in sorted(branches[b], key=lambda a: (a.get("introduced") or {}).get("date") or "9")]
    if cur:
        pages.append(cur)
    idx_all = {}
    for pi, page in enumerate(pages, 1):
        fig, ax = plt.subplots(figsize=(15, 0.42 * len(page) + 2.2))
        ypos = {}
        for i, (b, a) in enumerate(page):
            y = len(page) - i
            ypos[a["artifact_id"]] = y
            idx_all[a["artifact_id"]] = (pi, y)
            t0 = dt((a.get("introduced") or {}).get("date"))
            t1 = dt((a.get("ended") or {}).get("date")) if a.get("ended") else CUTOFF
            if t0:
                col = "#bdbdbd" if a.get("kind") in ("upstream_kernel", "dependency_pin") else "#9ecae1"
                ax.plot([t0, t1 or CUTOFF], [y, y], lw=6, color=col, solid_capstyle="butt", zorder=1)
                if a.get("ended"):
                    ax.plot([t1], [y], marker="|", ms=14, color="black", zorder=3)
            for e in by_art.get(a["artifact_id"], []):
                m, c = EV_STYLE.get(e["event_type"], ("o", "#444444"))
                big = e["event_type"] in ("introduce", "port", "optimize", "revert", "reland", "change_default", "replace", "remove")
                ax.scatter([dt(e["date"])], [y], marker=m, s=38 if big else 10, color=c, zorder=4,
                           edgecolors="black" if big else "none", linewidths=0.3)
            label = a["artifact_id"].split(".", 1)[1] if "." in a["artifact_id"] else a["artifact_id"]
            ax.text(dt("2023-01-01") if L == "L3" else dt("2024-04-01"), y, f"[{b}] {label}", fontsize=7, va="center", ha="right")
        # registry predecessor arrows within the page
        for a in [x for _, x in page]:
            for p in a.get("predecessors", []):
                if p.get("artifact_id") in ypos and p.get("relation") in EDGE_COL:
                    t = dt((a.get("introduced") or {}).get("date"))
                    if t:
                        ax.annotate("", xy=(t, ypos[a["artifact_id"]]), xytext=(t, ypos[p["artifact_id"]]),
                                    arrowprops=dict(arrowstyle="->", color=EDGE_COL[p["relation"]], lw=1.0, alpha=0.8,
                                                    connectionstyle="arc3,rad=0.25"))
        for r in rels:
            t = dt(r["release_date"])
            if t:
                ax.axvline(t, color="#eeeeee", lw=0.6, zorder=0)
        ax.set_yticks([])
        ax.set_xlim(dt("2023-01-01") if L == "L3" else dt("2024-04-01"), CUTOFF)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        handles = [plt.Line2D([], [], marker=v[0], color=v[1], ls="", label=k) for k, v in EV_STYLE.items()]
        handles += [plt.Line2D([], [], color=c, label=f"edge: {k}") for k, c in EDGE_COL.items() if k in ("ported_from", "reimplementation_of", "replaces", "forked_from")]
        ax.legend(handles=handles, fontsize=6, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
        ax.set_title(f"{L}: {LNAME[L]} — artifact lineage (page {pi}/{len(pages)}; bars = lifespan, grey lines = releases)", fontsize=10)
        save(fig, f"{L.lower()}-artifact-lineage-p{pi}")
    # DOT + Mermaid sources
    dot = [f'digraph "{L}_artifacts" {{', "  rankdir=LR; node [shape=box, fontsize=10];"]
    mer = ["graph LR"]
    safe = lambda s: re.sub(r"[^A-Za-z0-9_]", "_", s)
    for a in arts:
        intro = (a.get("introduced") or {}).get("date") or "?"
        end = (a.get("ended") or {}).get("date") if a.get("ended") else "live"
        lab = f"{a['artifact_id']}\\n{intro[:10]} → {str(end)[:10]}\\n{a.get('language','')}"
        dot.append(f'  "{a["artifact_id"]}" [label="{lab}"];')
        mer.append(f'  {safe(a["artifact_id"])}["{a["artifact_id"]}<br/>{intro[:10]} → {str(end)[:10]}"]')
    for a in arts:
        for p in a.get("predecessors", []):
            if p.get("artifact_id"):
                dot.append(f'  "{p["artifact_id"]}" -> "{a["artifact_id"]}" [label="{p.get("relation","")}"];')
                mer.append(f'  {safe(p["artifact_id"])} -->|{p.get("relation","")}| {safe(a["artifact_id"])}')
    dot.append("}")
    open(os.path.join(FIG, f"{L.lower()}-artifact-lineage.dot"), "w", encoding="utf-8").write("\n".join(dot))
    open(os.path.join(FIG, f"{L.lower()}-artifact-lineage.mmd"), "w", encoding="utf-8").write("\n".join(mer))


# ---------------- move DAGs ----------------
def move_dag(L, events, mpres):
    mv = moves(L)
    if not mv:
        return
    ev = [e for e in events if e["lineage"] == L]
    fig, ax = plt.subplots(figsize=(15, 0.55 * len(mv) + 2.2))
    rel_col = {"first": "#1b7837", "same_artifact_modified": "#2166ac", "ported": "#5aae61", "reimplemented": "#762a83",
               "rediscovered": "#e7298a", "removed": "#000000", "repaired": "#d6604d"}
    pres_by = defaultdict(dict)
    for p in mpres:
        if p["lineage"] == L:
            pres_by[p["move_id"]][p["release_date"]] = int(p["present"])
    for i, m in enumerate(mv):
        y = len(mv) - i
        # presence band from release scans
        ds = sorted(pres_by.get(m["move_id"], {}).items())
        for (d0, p0), (d1, _) in zip(ds, ds[1:] + [(CUTOFF.strftime("%Y-%m-%d"), 0)]):
            if p0:
                ax.plot([dt(d0), dt(d1)], [y, y], lw=6, color="#c7e9c0", solid_capstyle="butt", zorder=1)
        occ = [(m.get("first_observed") or {}) | {"relation": "first"}] + (m.get("occurrences") or [])
        pts = []
        for o in occ:
            t = dt(o.get("date"))
            if not t:
                continue
            c = rel_col.get(o.get("relation"), "#444444")
            mk = "s" if "vllm" in (o.get("repo") or "") and L != "L3" or ("sglang" in (o.get("repo") or "") and L == "L3") else "o"
            ax.scatter([t], [y], color=c, marker=mk, s=40, zorder=3, edgecolors="black", linewidths=0.3)
            pts.append(t)
        for e in ev:
            if m["move_id"] in e["optimization_move_ids"].split(";"):
                ax.scatter([dt(e["date"])], [y + 0.18], marker="|", color="#636363", s=25, zorder=2)
        ax.text(dt("2023-01-01") if L == "L3" else dt("2024-04-01"), y, m["move_id"].replace(f"M-{L}-", ""), fontsize=7, ha="right", va="center")
    ax.set_yticks([])
    ax.set_xlim(dt("2023-01-01") if L == "L3" else dt("2024-04-01"), CUTOFF)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    handles = [plt.Line2D([], [], marker="o", color=c, ls="", label=k) for k, c in rel_col.items()]
    handles.append(plt.Line2D([], [], lw=6, color="#c7e9c0", label="signature present in release"))
    handles.append(plt.Line2D([], [], marker="s", color="#999999", ls="", label="occurrence in other repo"))
    handles.append(plt.Line2D([], [], marker="|", color="#636363", ls="", label="coded event using move"))
    ax.legend(handles=handles, fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.set_title(f"{L}: optimization-move biographies (occurrences, release-scan presence, coded events)", fontsize=10)
    save(fig, f"{L.lower()}-move-dag")
    dot = [f'digraph "{L}_moves" {{', "  rankdir=LR; node [shape=ellipse, fontsize=9];"]
    mer = ["graph LR"]
    safe = lambda s: re.sub(r"[^A-Za-z0-9_]", "_", s)
    for m in mv:
        occ = [(m.get("first_observed") or {}) | {"relation": "first"}] + (m.get("occurrences") or [])
        prev = None
        for j, o in enumerate(occ):
            nid = f'{m["move_id"]}#{j}'
            lab = f'{m["move_id"]}\\n{o.get("repo","")}\\n{o.get("date","")} #{o.get("pr","")}\\n{o.get("artifact_id","")}'
            dot.append(f'  "{nid}" [label="{lab}"];')
            mer.append(f'  {safe(nid)}["{m["move_id"]}<br/>{o.get("repo","")} {o.get("date","")}<br/>{o.get("relation","")}"]')
            if prev:
                dot.append(f'  "{prev}" -> "{nid}" [label="{o.get("relation","")}"];')
                mer.append(f"  {safe(prev)} -->|{o.get('relation','')}| {safe(nid)}")
            prev = nid
    dot.append("}")
    open(os.path.join(FIG, f"{L.lower()}-move-dag.dot"), "w", encoding="utf-8").write("\n".join(dot))
    open(os.path.join(FIG, f"{L.lower()}-move-dag.mmd"), "w", encoding="utf-8").write("\n".join(mer))


# ---------------- release-overlaid timelines ----------------
def release_timeline(L, events, pres):
    repo = LREPO[L]
    rels = releases(repo)
    ev = [e for e in events if e["lineage"] == L]
    types = ["introduce", "port", "integrate", "optimize", "retune", "extend_support", "adapt_framework",
             "repair_correctness", "repair_performance", "repair_build_dependency", "revert", "reland",
             "change_default", "replace", "deprecate", "remove"]
    per = defaultdict(Counter)
    for e in ev:
        per[e["release"]][e["event_type"]] += 1
    tags = [r["tag"] for r in rels if per.get(r["tag"])]
    first = next((i for i, r in enumerate(rels) if per.get(r["tag"])), 0)
    shown = rels[first:] + [{"tag": "unreleased", "release_date": CUTOFF_DATE}]
    per["unreleased"] = sum((per[k] for k in per if not k.startswith("v")), Counter())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    x = list(range(len(shown)))
    bottom = [0] * len(shown)
    for t in types:
        vals = [per[r["tag"]][t] for r in shown]
        ax1.bar(x, vals, bottom=bottom, color=EV_STYLE[t][1], label=t, width=0.8)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax1.set_ylabel("lineage events landing\nin release")
    ax1.legend(fontsize=6, ncol=4)
    cnt = Counter()
    for p in pres:
        if p["lineage"] == L and p["present"] == "1":
            cnt[p["release"]] += 1
    ax2.plot(x, [cnt.get(r["tag"], cnt.get("CUTOFF", 0) if r["tag"] == "unreleased" else 0) for r in shown], color="#2166ac", marker=".")
    ax2.set_ylabel("in-tree artifacts\npresent")
    ax2.set_xticks(x)
    step = 2 if len(shown) > 60 else 1
    ax2.set_xticklabels([(f"{r['tag']}" if len(shown) > 60 else f"{r['tag']}\n{r['release_date'][:7]}") if i % step == 0 else ""
                         for i, r in enumerate(shown)], rotation=90, fontsize=6)
    ax1.set_title(f"{L}: {LNAME[L]} — events by first containing release (event type) and artifact count", fontsize=10)
    save(fig, f"{L.lower()}-release-timeline")


# ---------------- cross-lineage figures ----------------
def synthesis_figs(events, surv, kmf, kmr, trans, tart):
    # 1 survival: artifacts (present / unchanged) vs moves
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    series = (("artifact", "artifact path present", "#9ecae1"), ("artifact_unchanged", "artifact textually unchanged", "#fdae6b"),
              ("move", "move signature present", "#a1d99b"))
    for ax, L in zip(axes, ("L1", "L2", "L3")):
        for j, (kind, lab, col) in enumerate(series):
            ys = []
            for k in (1, 2, 3):
                r = next((s for s in surv if s["kind"] == kind and s["lineage"] == L and s["k_releases"] == str(k)
                          and s["basis"] == "all_present_releases"), None)
                ys.append(float(r["share"]) if r and r["share"] else 0)
            xs = [k + (j - 1) * 0.27 for k in (1, 2, 3)]
            ax.bar(xs, ys, width=0.27, label=lab, color=col)
            for x, y in zip(xs, ys):
                ax.text(x, y + 0.01, f"{y:.0%}", ha="center", fontsize=6)
        ax.set_title(L, fontsize=9)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("releases later")
    axes[0].set_ylabel("share of (item, release) pairs")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Code identity vs mechanism persistence across 1–3 subsequent releases", fontsize=10)
    save(fig, "synthesis-survival-artifact-vs-move")
    # 2 KM curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, title in ((axes[0], kmf, "artifact lifetime (days since introduction)"),
                            (axes[1], kmr, "time to first correctness/performance repair")):
        for L, col in (("L1", "#1b9e77"), ("L2", "#d95f02"), ("L3", "#7570b3")):
            pts = [(float(r["t_days"]), float(r["survival"])) for r in data if r["lineage"] == L]
            xs, ys = [0], [1.0]
            for t, s in pts:
                xs += [t, t]
                ys += [ys[-1], s]
            ax.plot(xs, ys, color=col, label=L)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("days")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Kaplan–Meier S(t)")
    save(fig, "synthesis-km-artifacts")
    # 3 causes
    fig, ax = plt.subplots(figsize=(9, 3.6))
    causes = list(CAUSE_COL)
    for i, L in enumerate(("L1", "L2", "L3")):
        ev = [e for e in events if e["lineage"] == L]
        n = len(ev) or 1
        left = 0
        c = Counter(e["primary_cause"] for e in ev)
        for k in causes:
            w = c[k] / n
            ax.barh(i, w, left=left, color=CAUSE_COL[k], label=k if i == 0 else None)
            if w > 0.06:
                ax.text(left + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=7)
            left += w
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([f"{L} (n={sum(1 for e in events if e['lineage']==L)})" for L in ("L1", "L2", "L3")])
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    ax.set_title("Primary cause of verified lineage events", fontsize=10)
    save(fig, "synthesis-causes")
    # 4 rebase classes per artifact-transition
    fig, ax = plt.subplots(figsize=(9, 3.6))
    order = list(CLASS_COL)
    for i, L in enumerate(("L1", "L2", "L3")):
        sub = [t for t in tart if t["lineage"] == L and t["to_release"] != "UNRELEASED_AT_CUTOFF"]
        n = len(sub) or 1
        c = Counter(t["rebase_class"] for t in sub)
        left = 0
        for k in order:
            w = c[k] / n
            ax.barh(i, w, left=left, color=CLASS_COL[k], edgecolor="#777777", lw=0.3, label=k if i == 0 else None)
            if w > 0.06:
                ax.text(left + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if k in ("backend_replacement", "obsolete") else "black")
            left += w
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["L1", "L2", "L3"])
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    ax.set_title("How each artifact crossed each release boundary (per artifact × transition)", fontsize=10)
    save(fig, "synthesis-rebase-classes")
    # 5 repairs per transition over time
    fig, ax = plt.subplots(figsize=(12, 3.6))
    for L, col in (("L1", "#1b9e77"), ("L2", "#d95f02"), ("L3", "#7570b3")):
        rels = {r["tag"]: r["release_date"] for r in releases(LREPO[L])}
        pts = [(dt(rels.get(t["to_release"], "")), int(t["n_repair_events"])) for t in trans if t["lineage"] == L and t["to_release"] in rels]
        pts = [p for p in pts if p[0]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=".", color=col, label=L)
    ax.set_ylabel("repair events landing")
    ax.set_title("Repairs per release transition", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, "synthesis-repairs-per-transition")


def main():
    events = rcsv(os.path.join(DATA, "lineage-events.csv"))
    edges = rcsv(os.path.join(DATA, "lineage-edges.csv")) if os.path.exists(os.path.join(DATA, "lineage-edges.csv")) else []
    pres = rcsv(os.path.join(DATA, "artifact-release-presence.csv"))
    mp_fn = os.path.join(DATA, "move-release-presence.csv")
    mpres = rcsv(mp_fn) if os.path.exists(mp_fn) else []
    for L in ("L1", "L2", "L3"):
        artifact_lanes(L, events, edges)
        move_dag(L, events, mpres)
        release_timeline(L, events, pres)
    surv = rcsv(os.path.join(DATA, "survival-by-releases.csv"))
    kmf = rcsv(os.path.join(DATA, "km-artifact-lifetime.csv"))
    kmr = rcsv(os.path.join(DATA, "km-first-repair.csv"))
    trans = rcsv(os.path.join(DATA, "release-transitions.csv"))
    tart = rcsv(os.path.join(DATA, "release-transition-artifacts.csv"))
    synthesis_figs(events, surv, kmf, kmr, trans, tart)


if __name__ == "__main__":
    main()
