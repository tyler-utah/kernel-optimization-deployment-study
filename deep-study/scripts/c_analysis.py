"""C: paper census merge, second pass, C2/C3 batches, funnel and latency.

census      - merge first pass + full-text re-reads; write second-pass batches for every
              yes/ambiguous paper (task c1_census_p2)
final       - agreement + adjudication -> data/paper-census.csv
deploy      - C2/C3 batches for every kernel-style positive + >=20 random negatives per venue
ladder      - merge C2/C3 -> data/paper-deployment-ladder.csv; funnel + latency figures
"""

from __future__ import annotations

import collections
import json
import sys

import numpy as np

from b_analysis import kappa
from batches import load_outputs
from common import BATCHES, CUTOFF, DATA, FIG, atomic_csv, iso, read_json, read_jsonl, rng, write_jsonl

CAT = ["kernel", "fusion_family", "precision_format", "scheduling_layout", "compiler_dsl", "agentic_generation", "not_kernel_style"]


def dossiers() -> dict[str, dict]:
    return {d["id"]: d for d in read_json(DATA / "paper-dossiers.json")}


def first_pass() -> dict[str, dict]:
    p1 = load_outputs("c1_census")
    for k, v in load_outputs("c1_fulltext").items():
        merged = dict(p1[k])
        merged.update({x: v[x] for x in ["kernel_style", "category", "secondary_categories", "technique_name", "target_workload",
                                          "rationale", "confidence", "sections_read"]})
        merged["fulltext_url_used"] = v.get("fulltext_url", "")
        merged["fulltext_hunt"] = v.get("fulltext_found", "")
        p1[k] = merged
    return p1


def census() -> None:
    p1 = first_pass()
    ds = dossiers()
    items = []
    for pid, o in p1.items():
        if o["kernel_style"] in ("yes", "ambiguous"):
            d = dict(ds[pid])
            if o.get("fulltext_url_used"):
                d["fulltext_url_found_by_first_coder"] = o["fulltext_url_used"]
            items.append(d)
    rng("c1p2").shuffle(items)
    out = BATCHES / "c1_census_p2"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(items), 12):
        write_jsonl(out / f"batch-{i // 12:03d}-input.jsonl", items[i:i + 12])
    print("second-pass papers", len(items), collections.Counter(p1[i["id"]]["kernel_style"] for i in items))


def final() -> None:
    p1 = first_pass()
    p2 = load_outputs("c1_census_p2")
    rec = load_outputs("c1_reconcile") if (BATCHES / "c1_reconcile").exists() else {}
    ds = dossiers()
    ids = sorted(p2)
    pairs_ks = [(p1[i]["kernel_style"], p2[i]["kernel_style"]) for i in ids]
    pairs_cat = [(p1[i]["category"], p2[i]["category"]) for i in ids]
    agr = [{"field": "kernel_style", "n": len(ids), "percent_agreement": round(sum(a == b for a, b in pairs_ks) / len(ids), 4),
            "cohen_kappa": round(kappa(pairs_ks) or 0, 4), "scope": "papers first-coded yes/ambiguous; model-model second pass"},
           {"field": "category", "n": len(ids), "percent_agreement": round(sum(a == b for a, b in pairs_cat) / len(ids), 4),
            "cohen_kappa": round(kappa(pairs_cat) or 0, 4), "scope": "same"}]
    atomic_csv(DATA / "paper-census-agreement.csv", agr)
    rows = []
    for pid, o in sorted(p1.items()):
        d = ds[pid]
        b = p2.get(pid)
        if pid in rec:
            fin, src = rec[pid], "reconciled"
        elif b and (b["kernel_style"], b["category"]) != (o["kernel_style"], o["category"]):
            fin, src = None, "DISAGREEMENT_PENDING"
        elif b:
            fin, src = o, "agreed_second_pass"
        else:
            fin, src = o, "first_pass_negative"
        fin = fin or o
        rows.append({"id": pid, "venue": d["venue"], "title": d["title"], "authors": d["authors"], "url": d["url"], "doi": d["doi"],
                     "arxiv": d["arxiv"], "fulltext_available": int(d["fulltext_available"] or o.get("fulltext_hunt") == "yes"),
                     "sections_read": ";".join(o.get("sections_read") or []),
                     "kernel_style": fin["kernel_style"], "category": fin["category"],
                     "secondary_categories": ";".join(fin.get("secondary_categories") or []),
                     "technique_name": fin.get("technique_name", o.get("technique_name", "")),
                     "target_workload": fin.get("target_workload", o.get("target_workload", "")),
                     "rationale": fin.get("rationale") or fin.get("adjudication_rationale", ""), "confidence": fin.get("confidence", ""),
                     "label_source": src, "kernel_style_pass1": o["kernel_style"], "kernel_style_pass2": b["kernel_style"] if b else "",
                     "category_pass1": o["category"], "category_pass2": b["category"] if b else ""})
    atomic_csv(DATA / "paper-census.csv", rows)
    pend = [r for r in rows if r["label_source"] == "DISAGREEMENT_PENDING"]
    print("census rows", len(rows), "pending disagreements", len(pend), agr)
    if pend and not rec:
        items = []
        for r in pend:
            d = dict(ds[r["id"]])
            d["coder_A"] = {k: p1[r["id"]][k] for k in ["kernel_style", "category", "rationale"]}
            d["coder_B"] = {k: p2[r["id"]][k] for k in ["kernel_style", "category", "rationale"]}
            items.append(d)
        out = BATCHES / "c1_reconcile"
        out.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(items), 12):
            write_jsonl(out / f"batch-{i // 12:03d}-input.jsonl", items[i:i + 12])
        (out / "schema.json").write_text(json.dumps({"kernel_style": ["yes", "no", "ambiguous"], "category": CAT,
                                                     "secondary_categories": None, "technique_name": None, "target_workload": None,
                                                     "adjudication_rationale": None, "confidence": ["high", "medium", "low"]}))
        print("wrote c1_reconcile batches", (len(items) + 11) // 12)


def deploy() -> None:
    from common import read_csv
    rows = read_csv(DATA / "paper-census.csv")
    ds = dossiers()
    pos = [r for r in rows if r["kernel_style"] == "yes"]
    amb = [r for r in rows if r["kernel_style"] == "ambiguous"]
    neg = {}
    for v in ["MLSys 2025", "ASPLOS 2025"]:
        pool = [r for r in rows if r["kernel_style"] == "no" and r["venue"] == v]
        neg[v] = rng(f"c3-neg-{v}").sample(pool, min(20, len(pool)))
    items = []
    for r in pos + amb:
        d = dict(ds[r["id"]])
        d.update({"census_category": r["category"], "technique_name": r["technique_name"], "role": "positive" if r["kernel_style"] == "yes" else "ambiguous"})
        items.append(d)
    for v, rs in neg.items():
        for r in rs:
            d = dict(ds[r["id"]])
            d.update({"census_category": r["category"], "technique_name": r["technique_name"], "role": "negative_check"})
            items.append(d)
    out = BATCHES / "c3_deploy"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(items), 6):
        write_jsonl(out / f"batch-{i // 6:03d}-input.jsonl", items[i:i + 6])
    lv = ["L0", "L1", "L2", "L3", "L4", "L5"]
    (out / "schema.json").write_text(json.dumps({
        "eval_scope": ["microbenchmark_only", "end_to_end_only", "both", "na"], "baselines": None,
        "baseline_versions_stated": ["yes", "partial", "no", "na"], "n_hardware_targets": None, "hardware_targets": None,
        "hardware_vendors": None, "correctness_validation": ["none_reported", "tolerance_vs_reference", "model_accuracy_eval", "both", "formal_or_exact", "na"],
        "public_code": ["yes", "no"], "code_url": None, "artifact_badges": None, "claimed_production": ["yes", "no"],
        "claimed_production_quote": None, "ladder_level": lv, "l2_evidence": None, "l3_evidence": None, "l4_evidence": None,
        "l5_evidence": None, "in_progress_signal": None, "preprint_date": None, "first_code_date": None,
        "first_downstream_pr_date": None, "downstream_merge_date": None, "first_default_or_doc_date": None,
        "search_log": None, "confidence": ["high", "medium", "low"], "notes": None}))
    print("c3 items", len(items), "positives", len(pos), "ambiguous", len(amb), "negatives", sum(len(x) for x in neg.values()))


CONF_DATE = {"MLSys 2025": "2025-05-12", "ASPLOS 2025": "2025-03-30"}
LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, (c - r) / d), min(1.0, (c + r) / d))


def months(a, b):
    if not a or not b:
        return None
    try:
        return round((iso(b[:10] + "T00:00:00Z") - iso(a[:10] + "T00:00:00Z")).days / 30.44, 1)
    except Exception:  # noqa: BLE001
        return None


def ladder() -> None:
    from common import read_csv
    census = {r["id"]: r for r in read_csv(DATA / "paper-census.csv")}
    dep = load_outputs("c3_deploy")
    ver = load_outputs("c3_verify") if (BATCHES / "c3_verify").exists() else {}
    for pid, v in ver.items():
        o = dict(dep[pid])
        o["first_pass_level"] = o["ladder_level"]
        o["ladder_level"] = v["verified_level"]
        for k in ["public_code", "code_url", "l2_evidence", "l3_evidence", "l4_evidence", "l5_evidence", "first_code_date",
                  "first_downstream_pr_date", "downstream_merge_date", "first_default_or_doc_date"]:
            o[k] = v.get(k, o.get(k, ""))
        o["verification_notes"] = v.get("verification_notes", "")
        o["is_missed_kernel_style"] = v.get("is_missed_kernel_style", "na")
        o["verified"] = "yes"
        dep[pid] = o
    items = {}
    for p in sorted((BATCHES / "c3_deploy").glob("batch-*-input.jsonl")):
        for r in read_jsonl(p):
            items[r["id"]] = r
    rows = []
    for pid, o in dep.items():
        c = census[pid]
        role = items[pid]["role"]
        lvl = o["ladder_level"]
        rows.append({"id": pid, "venue": c["venue"], "title": c["title"], "role": role, "kernel_style": c["kernel_style"],
                     "category": c["category"], "technique_name": c["technique_name"], "target_workload": c["target_workload"],
                     **{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v) for k, v in o.items() if k != "id"},
                     "level_num": LEVELS.index(lvl), "reached_L1": int(LEVELS.index(lvl) >= 1), "reached_L2": int(LEVELS.index(lvl) >= 2),
                     "reached_L3": int(LEVELS.index(lvl) >= 3), "reached_L4": int(LEVELS.index(lvl) >= 4), "reached_L5": int(lvl == "L5"),
                     "months_conf_to_first_downstream_pr": months(CONF_DATE[c["venue"]], o.get("first_downstream_pr_date")),
                     "months_preprint_to_first_downstream_pr": months(o.get("preprint_date"), o.get("first_downstream_pr_date")),
                     "months_preprint_to_merge": months(o.get("preprint_date"), o.get("downstream_merge_date")),
                     "months_preprint_to_default_or_doc": months(o.get("preprint_date"), o.get("first_default_or_doc_date"))})
    rows.sort(key=lambda r: r["id"])
    for r in rows:
        pub = r.get("preprint_date") or CONF_DATE[r["venue"]]
        m = r.get("downstream_merge_date") or ""
        r["adoption_precedes_publication"] = int(bool(m) and m[:10] < pub[:10]) if r["level_num"] >= 4 else ""
    atomic_csv(DATA / "paper-deployment-ladder.csv", rows)
    # funnel table
    fun = []
    for v in ["MLSys 2025", "ASPLOS 2025", "both"]:
        allp = [r for r in census.values() if v == "both" or r["venue"] == v]
        pos = [r for r in rows if r["role"] == "positive" and (v == "both" or r["venue"] == v)]
        stages = [("all_papers", len(allp)), ("kernel_style", len(pos))] + [
            (f"reached_{s}", sum(r[f"reached_{s}"] for r in pos)) for s in ["L1", "L2", "L3", "L4", "L5"]]
        stages.append(("self_reported_production_S", sum(r["claimed_production"] == "yes" for r in pos)))
        stages.append(("L4plus_framework_code_before_publication", sum(1 for r in pos if r["adoption_precedes_publication"] == 1)))
        stages.append(("L4plus_adopted_after_publication", sum(1 for r in pos if r["adoption_precedes_publication"] == 0)))
        for name, k in stages:
            den = len(pos) if name.startswith(("reached", "self", "L4plus")) else len(allp)
            lo, hi = wilson(k, den)
            fun.append({"venue": v, "stage": name, "n": k, "denominator": den, "share": round(k / den, 4) if den else "",
                        "wilson95_low": round(lo, 4), "wilson95_high": round(hi, 4)})
        neg = [r for r in rows if r["role"] == "negative_check" and (v == "both" or r["venue"] == v)]
        k = sum(r["reached_L3"] for r in neg)
        lo, hi = wilson(k, len(neg))
        fun.append({"venue": v, "stage": "negatives_with_L3plus_evidence", "n": k, "denominator": len(neg),
                    "share": round(k / len(neg), 4) if neg else "", "wilson95_low": round(lo, 4), "wilson95_high": round(hi, 4)})
        km = sum(1 for r in neg if r.get("is_missed_kernel_style") == "yes" and r["reached_L3"])
        lo, hi = wilson(km, len(neg))
        fun.append({"venue": v, "stage": "negatives_missed_kernel_style_with_L3plus", "n": km, "denominator": len(neg),
                    "share": round(km / len(neg), 4) if neg else "", "wilson95_low": round(lo, 4), "wilson95_high": round(hi, 4)})
    atomic_csv(DATA / "paper-funnel.csv", fun)
    # evaluation audit summary (positives)
    aud = []
    pos = [r for r in rows if r["role"] == "positive"]
    for v in ["MLSys 2025", "ASPLOS 2025", "both"]:
        pp = [r for r in pos if v == "both" or r["venue"] == v]
        for field in ["eval_scope", "baseline_versions_stated", "correctness_validation", "public_code", "claimed_production"]:
            for k, n in collections.Counter(r[field] for r in pp).most_common():
                aud.append({"venue": v, "field": field, "value": k, "n": n, "denominator": len(pp), "share": round(n / len(pp), 4)})
        hw = [int(r["n_hardware_targets"]) for r in pp if str(r["n_hardware_targets"]).isdigit()]
        vend = [len({x.strip().lower() for x in (r["hardware_vendors"] or "").split(";") if x.strip()}) for r in pp]
        vend = [v_ for v_ in vend if v_ > 0]
        if vend:
            aud.append({"venue": v, "field": "hardware_vendors", "value": "single_vendor", "n": sum(x == 1 for x in vend), "denominator": len(vend),
                        "share": round(sum(x == 1 for x in vend) / len(vend), 4)})
        if hw:
            aud.append({"venue": v, "field": "n_hardware_targets", "value": "median", "n": float(np.median(hw)), "denominator": len(hw), "share": ""})
            aud.append({"venue": v, "field": "n_hardware_targets", "value": "single_target", "n": sum(h == 1 for h in hw), "denominator": len(hw),
                        "share": round(sum(h == 1 for h in hw) / len(hw), 4)})
        badges = collections.Counter(b for r in pp for b in (r["artifact_badges"] or "").split(";") if b)
        for k, n in badges.most_common():
            aud.append({"venue": v, "field": "artifact_badges", "value": k, "n": n, "denominator": len(pp), "share": round(n / len(pp), 4)})
    atomic_csv(DATA / "paper-eval-audit-summary.csv", aud)
    figures_c(rows, fun)
    print("ladder rows", len(rows))
    for f in fun:
        print(f)


def figures_c(rows, fun) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    stages = ["all_papers", "kernel_style", "reached_L1", "reached_L2", "reached_L3", "reached_L4", "reached_L5"]
    labels = ["all\npapers", "kernel-\nstyle", "code\n(L1)", "maintained\n(L2)", "kernel lib.\n(L3)", "framework\n(L4)", "default/doc\n(L5)"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, v in zip(axes, ["MLSys 2025", "ASPLOS 2025"]):
        vals = [next(f["n"] for f in fun if f["venue"] == v and f["stage"] == s) for s in stages]
        top = max(vals)
        x = np.arange(len(stages)) * 2.0
        w = 0.5
        for i, (xi, n) in enumerate(zip(x, vals)):
            h = n / top
            ax.add_patch(plt.Rectangle((xi - w / 2, 0.5 - h / 2), w, h, color="#1f4e79"))
            ax.text(xi, 0.5 + h / 2 + 0.03, f"{n}", ha="center", fontsize=9, fontweight="bold")
            ax.text(xi, -0.08, labels[i], ha="center", va="top", fontsize=7.5, rotation=0, wrap=True)
            if i + 1 < len(vals):
                h2 = vals[i + 1] / top
                ax.add_patch(Polygon([(xi + w / 2, 0.5 - h / 2), (x[i + 1] - w / 2, 0.5 - h2 / 2), (x[i + 1] - w / 2, 0.5 + h2 / 2),
                                      (xi + w / 2, 0.5 - h / 2 + h2)], closed=True, color="#9dc3e6", alpha=.8))
                if n - vals[i + 1] > 0:
                    ax.text((xi + x[i + 1]) / 2, 0.5 + h / 2 - 0.02, f"-{n - vals[i + 1]}", ha="center", fontsize=7, color="#c00000")
        ax.set_xlim(-1, x[-1] + 1)
        ax.set_ylim(-0.25, 1.15)
        ax.axis("off")
        ax.set_title(f"{v}: paper-to-production evidence ladder (highest observable level, cutoff {CUTOFF.date()})", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "c-paper-to-production-funnel.png", dpi=170)
    plt.close(fig)
    lat = [r for r in rows if r["role"] == "positive" and r["level_num"] >= 3]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    series = [("preprint → first downstream PR", "months_preprint_to_first_downstream_pr"), ("preprint → merge", "months_preprint_to_merge"),
              ("preprint → default/doc", "months_preprint_to_default_or_doc")]
    for j, (lab, key) in enumerate(series):
        ys = [r[key] for r in lat if r[key] is not None]
        ax.scatter(ys, np.full(len(ys), j) + RSJ(len(ys)), s=28, alpha=.8)
        if ys:
            ax.text(max(ys) + 0.5, j, f"n={len(ys)}, median={np.median(ys):.1f} mo", va="center", fontsize=8)
    ax.set_yticks(range(len(series)), [s[0] for s in series], fontsize=8)
    ax.set_xlabel("months")
    ax.axvline(0, color="grey", lw=.8)
    ax.set_title(f"Adoption latency for kernel-style papers with L3+ evidence (n={len(lat)})", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "c-adoption-latency.png", dpi=170)
    plt.close(fig)


def RSJ(n):
    return np.random.default_rng(7).uniform(-0.12, 0.12, n)


def verify_batches() -> None:
    dep = load_outputs("c3_deploy")
    items = []
    for p in sorted((BATCHES / "c3_deploy").glob("batch-*-input.jsonl")):
        for r in read_jsonl(p):
            if r["role"] in ("positive", "ambiguous") or (r["role"] == "negative_check" and dep[r["id"]]["ladder_level"] in ("L3", "L4", "L5")):
                o = dep[r["id"]]
                items.append({"id": r["id"], "venue": r["venue"], "title": r["title"], "role": r["role"], "technique_name": r.get("technique_name", ""),
                              "arxiv": r.get("arxiv", ""), "code_links_in_paper": r.get("code_links", []),
                              "first_pass": {k: o[k] for k in ["ladder_level", "public_code", "code_url", "l2_evidence", "l3_evidence", "l4_evidence",
                                                              "l5_evidence", "in_progress_signal", "preprint_date", "first_code_date",
                                                              "first_downstream_pr_date", "downstream_merge_date", "first_default_or_doc_date", "claimed_production"]}})
    out = BATCHES / "c3_verify"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(items), 8):
        write_jsonl(out / f"batch-{i // 8:03d}-input.jsonl", items[i:i + 8])
    (out / "schema.json").write_text(json.dumps({
        "verified_level": LEVELS, "public_code": ["yes", "no"], "code_url": None, "l2_evidence": None, "l3_evidence": None,
        "l4_evidence": None, "l5_evidence": None, "first_code_date": None, "first_downstream_pr_date": None, "downstream_merge_date": None,
        "first_default_or_doc_date": None, "changed_from_first_pass": ["yes", "no"], "verification_notes": None,
        "is_missed_kernel_style": ["yes", "no", "na"]}))
    print("c3_verify items", len(items))


if __name__ == "__main__":
    {"census": census, "final": final, "deploy": deploy, "ladder": ladder, "verify": verify_batches}[sys.argv[1]]()
