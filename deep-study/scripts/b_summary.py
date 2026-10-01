"""B summaries and figures: revert census, failure taxonomy, counterfactual matrix."""

from __future__ import annotations

import collections
import csv

import numpy as np

from common import DATA, FIG, REPOS, atomic_csv, read_csv

RS = np.random.default_rng(20260929)


def boot_share(x: np.ndarray, reps: int = 2000):
    if len(x) == 0:
        return (float("nan"), float("nan"))
    bs = [x[RS.integers(0, len(x), len(x))].mean() for _ in range(reps)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def reverts() -> list[dict]:
    rows = read_csv(DATA / "confirmed-reverts.csv")
    out = []
    for repo in list(REPOS) + ["both"]:
        rr = [r for r in rows if (repo == "both" or r["repo"] == repo)]
        conf = [r for r in rr if r["is_revert_or_rollback"] == "1"]
        out.append({"repo": repo, "measure": "candidates", "category": "all", "n": len(rr), "denominator": len(rr), "share": 1.0})
        out.append({"repo": repo, "measure": "confirmed_reverts_or_rollbacks", "category": "all", "n": len(conf), "denominator": len(rr),
                    "share": round(len(conf) / len(rr), 4)})
        for field in ["reverted_class", "revert_reason", "revert_status", "hardware_specific", "reverted_touches_kernel"]:
            c = collections.Counter(r[field] for r in conf)
            for k, v in c.most_common():
                x = np.array([r[field] == k for r in conf], float)
                lo, hi = boot_share(x)
                out.append({"repo": repo, "measure": field, "category": k, "n": v, "denominator": len(conf),
                            "share": round(v / len(conf), 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)})
        for cls in ["kernel_performance", "kernel_correctness", "hardware_backend", "other", "all"]:
            d = np.array([float(r["days_to_revert"]) for r in conf if r["days_to_revert"] and (cls == "all" or r["reverted_class"] == cls)])
            if len(d):
                out.append({"repo": repo, "measure": "median_days_to_revert", "category": cls, "n": len(d), "denominator": len(d),
                            "share": round(float(np.median(d)), 3), "ci95_low": round(float(np.percentile(d, 25)), 3),
                            "ci95_high": round(float(np.percentile(d, 75)), 3), "note": "ci columns hold IQR for this measure"})
        kp = [r for r in conf if r["reverted_class"] == "kernel_performance"]
        for k, v in collections.Counter(r["revert_reason"] for r in kp).most_common():
            out.append({"repo": repo, "measure": "kernel_performance_revert_reason", "category": k, "n": v, "denominator": len(kp),
                        "share": round(v / len(kp), 4)})
    atomic_csv(DATA / "b1-revert-summary.csv", out)
    return out


def failures() -> list[dict]:
    rows = read_csv(DATA / "kernel-correctness-cases.csv")
    frame = {"vllm": 1416, "sglang": 938}
    out = []
    for repo in list(REPOS) + ["both"]:
        rr = [r for r in rows if repo == "both" or r["repo"] == repo]
        conf = [r for r in rr if r["case_status"] == "confirmed_kernel_correctness"]
        x = np.array([r["case_status"] == "confirmed_kernel_correctness" for r in rr], float)
        lo, hi = boot_share(x)
        fsize = sum(frame.values()) if repo == "both" else frame[repo]
        out.append({"repo": repo, "measure": "screened", "category": "all", "n": len(rr), "denominator": fsize, "share": round(len(rr) / fsize, 4)})
        out.append({"repo": repo, "measure": "confirmed_rate_in_sample", "category": "confirmed", "n": len(conf), "denominator": len(rr),
                    "share": round(len(conf) / len(rr), 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)})
        out.append({"repo": repo, "measure": "estimated_confirmed_in_frame", "category": "confirmed", "n": round(fsize * len(conf) / len(rr)),
                    "denominator": fsize, "share": round(len(conf) / len(rr), 4), "ci95_low": round(fsize * lo), "ci95_high": round(fsize * hi),
                    "note": "ci columns are counts for this measure"})
        for field in ["failure_class", "symptom", "hardware_specific", "counterfactual_primary", "detected_by", "escaped_ci",
                      "consequence", "regression_test_added", "introducing_pr_is_performance",
                      "introducing_pr_reported_correctness_tests", "introducing_pr_reported_accuracy_eval",
                      "introducing_pr_reported_multi_hardware"]:
            base = [r for r in conf if r[field] not in ("",)]
            for k, v in collections.Counter(r[field] for r in base).most_common():
                xx = np.array([r[field] == k for r in base], float)
                lo, hi = boot_share(xx)
                out.append({"repo": repo, "measure": field, "category": k, "n": v, "denominator": len(base),
                            "share": round(v / len(base), 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)})
        # any-plausible technique (multi-label)
        for tech in ["randomized_edge_tests", "hidden_input_distributions", "sanitizer_memcheck", "determinism_schedule_perturbation",
                     "formal_equivalence", "warp_race_static_analysis", "e2e_serving_tests_only", "hardware_matrix_ci"]:
            xx = np.array([tech in r["counterfactual_all"].split(";") for r in conf], float)
            lo, hi = boot_share(xx)
            out.append({"repo": repo, "measure": "counterfactual_any", "category": tech, "n": int(xx.sum()), "denominator": len(conf),
                        "share": round(float(xx.mean()), 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)})
        d = np.array([float(r["days_intro_to_fix"]) for r in conf if r["days_intro_to_fix"]])
        if len(d):
            out.append({"repo": repo, "measure": "median_days_intro_to_fix", "category": "traceable", "n": len(d), "denominator": len(conf),
                        "share": round(float(np.median(d)), 2), "ci95_low": round(float(np.percentile(d, 25)), 2),
                        "ci95_high": round(float(np.percentile(d, 75)), 2), "note": "ci columns hold IQR"})
    atomic_csv(DATA / "b2-failure-summary.csv", out)
    return out


def figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in read_csv(DATA / "kernel-correctness-cases.csv") if r["case_status"] == "confirmed_kernel_correctness"]
    classes = [c for c, _ in collections.Counter(r["failure_class"] for r in rows).most_common()]
    techs = ["randomized_edge_tests", "hidden_input_distributions", "sanitizer_memcheck", "determinism_schedule_perturbation",
             "formal_equivalence", "warp_race_static_analysis", "e2e_serving_tests_only", "hardware_matrix_ci", "not_enough_information"]
    m = np.zeros((len(classes), len(techs)))
    for r in rows:
        m[classes.index(r["failure_class"]), techs.index(r["counterfactual_primary"])] += 1
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.imshow(m, cmap="Reds")
    ax.set_xticks(range(len(techs)), [t.replace("_", "\n") for t in techs], fontsize=7)
    ax.set_yticks(range(len(classes)), [f"{c} (n={int(m[i].sum())})" for i, c in enumerate(classes)], fontsize=8)
    for i in range(len(classes)):
        for j in range(len(techs)):
            if m[i, j]:
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title(f"Confirmed kernel-correctness fixes (n={len(rows)}): failure class x most plausible pre-merge detector (adjudicated)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "b3-failure-vs-detector-matrix.png", dpi=170)
    plt.close(fig)

    rev = [r for r in read_csv(DATA / "confirmed-reverts.csv") if r["is_revert_or_rollback"] == "1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    cls = ["kernel_performance", "kernel_correctness", "hardware_backend", "other"]
    for ax, repo in zip(axes, REPOS):
        rr = [r for r in rev if r["repo"] == repo]
        reasons = [k for k, _ in collections.Counter(r["revert_reason"] for r in rev).most_common()]
        bottom = np.zeros(len(cls))
        cmap = plt.get_cmap("tab10")
        for i, reason in enumerate(reasons):
            vals = np.array([sum(1 for r in rr if r["reverted_class"] == c and r["revert_reason"] == reason) for c in cls])
            ax.bar(cls, vals, bottom=bottom, color=cmap(i % 10), label=reason)
            bottom += vals
        ax.set_title(f"{repo}: confirmed reverts/rollbacks by reverted work (n={len(rr)}, 24 months)", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
    axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "b1-reverts-by-class-reason.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    reverts()
    failures()
    figures()
    print("ok")
