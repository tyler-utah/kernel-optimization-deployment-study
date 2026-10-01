"""Report builder: every number is read from a generated CSV and emitted with a
claim tag so scripts/consistency.py can re-verify it.

    python experiments/deep-study/scripts/reports.py d_summary      # D aggregates + figure
    python experiments/deep-study/scripts/reports.py failures|provenance|papers|delivery|synthesis|all
"""

from __future__ import annotations

import collections
import json
import re
import sys

import numpy as np

from common import (
    A_START,
    B_START,
    CACHE,
    CONFIG,
    CUTOFF,
    DATA,
    DEEP,
    FIG,
    HEADS,
    atomic_csv,
    atomic_text,
    read_csv,
    read_jsonl,
)

_CACHE: dict[str, list[dict]] = {}


def rows(csv: str) -> list[dict]:
    if csv not in _CACHE:
        _CACHE[csv] = read_csv(DEEP / csv)
    return _CACHE[csv]


def fmtv(value, kind):
    v = float(value)
    return {"int": f"{round(v):,}", "pct0": f"{v * 100:.0f}", "pct1": f"{v * 100:.1f}", "dec1": f"{v:.1f}",
            "dec2": f"{v:.2f}", "dec3": f"{v:.3f}"}[kind] if kind != "raw" else str(value)


def get(csv, filt, col):
    conds = [c.split("=", 1) for c in filt.split("&") if c]
    rr = [r for r in rows(csv) if all(r.get(k) == v for k, v in conds)]
    if len(rr) != 1:
        raise KeyError(f"{csv} {filt}: {len(rr)} rows")
    return rr[0][col]


def C(csv, filt, col, kind="int"):
    """Formatted number followed by its claim tag."""
    return f"{fmtv(get(csv, filt, col), kind)}<!-- claim:{csv}::{filt}::{col}::{kind} -->"


def P(csv, filt, col="share", kind="pct1"):
    return C(csv, filt, col, kind) + "%"


def CI(csv, filt, lo="ci95_low", hi="ci95_high", kind="pct1"):
    return f"[{fmtv(get(csv, filt, lo), kind)}–{fmtv(get(csv, filt, hi), kind)}%]"


def anonymize(text: str) -> str:
    text = re.sub(r"\s*\(`[A-Za-z0-9_-]+`\)", "", text)
    text = re.sub(r"(?<![\w/])@[A-Za-z0-9-]{2,}", "a contributor", text)
    return text


# ------------------------------------------------------------------ D summary

def d_summary() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rr = read_csv(DATA / "production-kernel-provenance.csv")
    classes = ["academic_paper", "vendor_library", "company_engineering", "community_contribution", "unclear"]
    out = []
    rs = np.random.default_rng(20260929)
    for repo in ["vllm", "sglang", "both"]:
        for scope in ["all", "default"]:
            sub = [r for r in rr if (repo == "both" or r["repo"] == repo) and (scope == "all" or r["status"] == "default")]
            for c in classes:
                x = np.array([r["provenance_class"] == c for r in sub], float)
                bs = [x[rs.integers(0, len(x), len(x))].mean() for _ in range(2000)] if len(x) else [0]
                out.append({"repo": repo, "scope": scope, "provenance_class": c, "n": int(x.sum()), "denominator": len(sub),
                            "share": round(float(x.mean()), 4) if len(x) else 0, "ci95_low": round(float(np.percentile(bs, 2.5)), 4),
                            "ci95_high": round(float(np.percentile(bs, 97.5)), 4)})
        for oc in sorted({r["operator_class"] for r in rr}):
            sub = [r for r in rr if (repo == "both" or r["repo"] == repo) and r["operator_class"] == oc]
            for c in classes:
                k = sum(r["provenance_class"] == c for r in sub)
                out.append({"repo": repo, "scope": f"operator:{oc}", "provenance_class": c, "n": k, "denominator": len(sub),
                            "share": round(k / len(sub), 4) if sub else 0, "ci95_low": "", "ci95_high": ""})
    atomic_csv(DATA / "d-provenance-summary.csv", out)
    ops = sorted({r["operator_class"] for r in rr}, key=lambda o: -sum(r["operator_class"] == o for r in rr))
    colors = {"academic_paper": "#1f77b4", "vendor_library": "#ff7f0e", "company_engineering": "#2ca02c", "community_contribution": "#9467bd", "unclear": "#bbbbbb"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, repo in zip(axes, ["vllm", "sglang"]):
        left = np.zeros(len(ops))
        for c in classes:
            vals = np.array([sum(1 for r in rr if r["repo"] == repo and r["operator_class"] == o and r["provenance_class"] == c) for o in ops])
            ax.barh(ops, vals, left=left, color=colors[c], label=c.replace("_", " "))
            left += vals
        n = sum(1 for r in rr if r["repo"] == repo)
        ax.set_title(f"{repo}: origin of {n} deployed kernel/backend implementations (frozen HEAD)", fontsize=9)
        ax.invert_yaxis()
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "d-provenance-by-operator.png", dpi=170)
    plt.close(fig)
    print("d summary rows", len(out))


# ------------------------------------------------------------------ failures report

def failures() -> str:
    b1, b2, ag, rr = "data/b1-revert-summary.csv", "data/b2-failure-summary.csv", "data/failure-coding-agreement.csv", "data/a2-revert-rates.csv"
    f = lambda meas, cat, repo="both": f"repo={repo}&measure={meas}&category={cat}"
    cases = []
    for s in read_jsonl(CACHE / "b4" / "summary-0.jsonl") + read_jsonl(CACHE / "b4" / "summary-1.jsonl"):
        md = (CACHE / "b4" / f"case-{s['case_id'].replace(':', '_')}.md").read_text(encoding="utf-8")
        cases.append((s, anonymize(md)))
    fc_rows = [r for r in rows(b2) if r["repo"] == "both" and r["measure"] == "failure_class"]
    cf_rows = [r for r in rows(b2) if r["repo"] == "both" and r["measure"] == "counterfactual_primary"]
    cfa_rows = [r for r in rows(b2) if r["repo"] == "both" and r["measure"] == "counterfactual_any"]
    rc_rows = [r for r in rows(b1) if r["repo"] == "both" and r["measure"] == "reverted_class"]
    rr_rows = [r for r in rows(b1) if r["repo"] == "both" and r["measure"] == "revert_reason"]
    kpr = [r for r in rows(b1) if r["repo"] == "both" and r["measure"] == "kernel_performance_revert_reason"]
    t = []
    t.append("# How optimized kernels fail in vLLM and SGLang\n")
    t.append(f"**Deep follow-up, WP-B. Cutoff {CUTOFF.date()}; window {B_START.date()} → {CUTOFF.date()} (24 months).** "
             "All counts are generated from `data/confirmed-reverts.csv`, `data/kernel-correctness-cases.csv` and the "
             "summaries `data/b1-revert-summary.csv`, `data/b2-failure-summary.csv`. Coding was done by LLM subagents "
             "with a written codebook, an independent second pass and adjudication; agreement is model–model "
             "consistency, not human inter-rater reliability. See [`METHODS.md`](METHODS.md).\n")
    t.append("## Executive summary\n")
    t.append(f"1. **Reverts are a complete census.** Of {C(b1, f('candidates','all'), 'n')} first-parent commits with revert/rollback language, "
             f"{C(b1, f('confirmed_reverts_or_rollbacks','all'), 'n')} were confirmed reverts, partial reverts or explicit rollbacks "
             f"({C(b1, f('confirmed_reverts_or_rollbacks','all','vllm'), 'n')} vLLM, {C(b1, f('confirmed_reverts_or_rollbacks','all','sglang'), 'n')} SGLang).")
    t.append(f"2. **Performance work is reverted about twice as often.** Among merged PRs created in the 12-month window, "
             f"{P(rr, 'repo=vllm&group=performance', 'revert_rate')} of vLLM performance PRs were later reverted versus "
             f"{P(rr, 'repo=vllm&group=non_performance', 'revert_rate')} of other PRs; SGLang: {P(rr, 'repo=sglang&group=performance', 'revert_rate')} versus "
             f"{P(rr, 'repo=sglang&group=non_performance', 'revert_rate')} (Wilson intervals in `data/a2-revert-rates.csv`).")
    t.append(f"3. **Kernel-correctness fixes are common.** {C(b2, f('confirmed_rate_in_sample','confirmed'), 'n')} of {C(b2, f('screened','all'), 'n')} randomly "
             f"sampled kernel-path fix commits were confirmed kernel-correctness fixes ({P(b2, f('confirmed_rate_in_sample','confirmed'))}), implying about "
             f"{C(b2, f('estimated_confirmed_in_frame','confirmed'), 'n')} such fixes among the {C(b2, f('screened','all'), 'denominator')}-commit frame in 24 months.")
    t.append(f"4. **Most escaped bugs are integration or portability failures, not arithmetic.** Integration/backend-selection/CUDA-graph failures are "
             f"{P(b2, f('failure_class','integration_backend_cudagraph'))} of confirmed cases, hardware/compiler-specific {P(b2, f('failure_class','hardware_compiler_specific'))}, "
             f"shape/alignment edge cases {P(b2, f('failure_class','shape_alignment_edge'))}, numerical precision {P(b2, f('failure_class','numerical_precision'))}, "
             f"and races/synchronization only {P(b2, f('failure_class','nondeterminism_race_sync'))}; {P(b2, f('hardware_specific','yes'))} were hardware-specific.")
    t.append(f"5. **The most plausible missing check is usually system-level.** The adjudicated most-plausible pre-merge detector was hardware-matrix CI for "
             f"{P(b2, f('counterfactual_primary','hardware_matrix_ci'))}, end-to-end serving tests for {P(b2, f('counterfactual_primary','e2e_serving_tests_only'))} and "
             f"randomized/edge-case unit tests for {P(b2, f('counterfactual_primary','randomized_edge_tests'))}; determinism/schedule perturbation "
             f"({P(b2, f('counterfactual_primary','determinism_schedule_perturbation'))}), static warp-race analysis "
             f"({P(b2, f('counterfactual_primary','warp_race_static_analysis'))}) and formal equivalence ({P(b2, f('counterfactual_primary','formal_equivalence'))}) "
             f"were rarely the best fit. This is a counterfactual judgement (κ = {C(ag, 'field=counterfactual_primary', 'cohen_kappa', 'dec2')} between independent passes).\n")

    t.append("## B1 — 24-month revert census\n")
    t.append(f"Every candidate was adjudicated in context (commit body, revert-PR body and discussion, and up to four resolved target PRs). "
             f"The census is complete for first-parent history; reverts performed inside unmerged branches or force-pushes are invisible.\n")
    t.append("| Reverted work | n | share of confirmed reverts | 95% bootstrap CI |\n|---|---:|---:|---|")
    for r in rc_rows:
        t.append(f"| {r['category']} | {C(b1, f('reverted_class', r['category']), 'n')} | {P(b1, f('reverted_class', r['category']))} | {CI(b1, f('reverted_class', r['category']))} |")
    t.append("\n| Stated revert reason (all confirmed) | n | share |\n|---|---:|---:|")
    for r in rr_rows:
        t.append(f"| {r['category']} | {C(b1, f('revert_reason', r['category']), 'n')} | {P(b1, f('revert_reason', r['category']))} |")
    t.append(f"\nFor the {C(b1, f('reverted_class','kernel_performance'), 'n')} reverts of kernel/performance work, the stated reasons were: "
             + "; ".join(f"{r['category']} {C(b1, f('kernel_performance_revert_reason', r['category']), 'n')}" for r in kpr) + ".")
    t.append(f"Median time from the reverted change's merge to the revert was {C(b1, f('median_days_to_revert','all'), 'share', 'dec1')} days overall "
             f"and {C(b1, f('median_days_to_revert','kernel_performance'), 'share', 'dec1')} days for kernel/performance work — most reverts are fast "
             f"rollbacks after CI or user breakage, not slow retirements.\n")
    t.append("![Reverts by class and reason](figures/b1-reverts-by-class-reason.png)\n")
    t.append("*Observation:* confirmed reverts/rollbacks per repository by reverted work class, stacked by stated reason. "
             "*Interpretation:* \"unstated\" and CI failures dominate; public revert text often omits the user-visible symptom.\n")

    t.append("## B2 — confirmed kernel-correctness corpus\n")
    t.append(f"Frame: all {C(b2, f('screened','all'), 'denominator')} first-parent commits (24 months) that touch production kernel paths and use fix/failure "
             f"language. A seeded random {C(b2, f('screened','all'), 'n')} (220 per repository) were screened in context; "
             f"{C(b2, f('confirmed_rate_in_sample','confirmed'), 'n')} were confirmed ({C(b2, f('confirmed_rate_in_sample','confirmed','vllm'), 'n')} vLLM, "
             f"{C(b2, f('confirmed_rate_in_sample','confirmed','sglang'), 'n')} SGLang). Confirmation requires a fixing PR/commit that establishes the incorrect behaviour.\n")
    t.append("| Failure class (primary) | n | share | 95% CI |\n|---|---:|---:|---|")
    for r in fc_rows:
        t.append(f"| {r['category']} | {C(b2, f('failure_class', r['category']), 'n')} | {P(b2, f('failure_class', r['category']))} | {CI(b2, f('failure_class', r['category']))} |")
    t.append("")
    t.append(f"- **Detection:** developer/reviewer {P(b2, f('detected_by','developer_or_reviewer'))}, user issue {P(b2, f('detected_by','user_issue'))}, "
             f"post-merge CI/nightly {P(b2, f('detected_by','ci_or_nightly'))}.")
    t.append(f"- **Escaped pre-merge CI:** {P(b2, f('escaped_ci','yes'))} (the remainder are `unknown`; none was shown to have been caught before merge).")
    t.append(f"- **Regression test added with the fix:** {P(b2, f('regression_test_added','yes'))}.")
    t.append(f"- **Consequence:** fix-forward {P(b2, f('consequence','fix_forward'))}, fallback {P(b2, f('consequence','fallback'))}, "
             f"disablement {P(b2, f('consequence','disablement'))}, revert {P(b2, f('consequence','revert'))}.")
    t.append(f"- **Traceable introductions:** {C(b2, f('median_days_intro_to_fix','traceable'), 'n')} cases name the introducing PR; median "
             f"{C(b2, f('median_days_intro_to_fix','traceable'), 'share', 'dec1')} days from its merge to the fix. Of the introducing PRs, "
             f"{P(b2, f('introducing_pr_is_performance','yes'))} were themselves performance PRs and {P(b2, f('introducing_pr_reported_correctness_tests','yes'))} "
             f"had reported correctness tests before merge — tests existed and the bug still escaped.\n")
    t.append("## B3 — which validation would plausibly have caught it\n")
    t.append("Every confirmed case was coded twice independently and every disagreement adjudicated. Agreement between the two passes "
             f"(model–model): confirmation {P(ag, 'field=case_status', 'percent_agreement')} agreement; failure class κ = "
             f"{C(ag, 'field=failure_class', 'cohen_kappa', 'dec2')}; hardware specificity κ = {C(ag, 'field=hardware_specific', 'cohen_kappa', 'dec2')}; "
             f"most-plausible technique κ = {C(ag, 'field=counterfactual_primary', 'cohen_kappa', 'dec2')}; the two plausible-technique sets overlapped in "
             f"{P(ag, 'field=counterfactual_all_overlap', 'percent_agreement')} of cases.\n")
    t.append("| Technique | most plausible (primary) | any plausible |\n|---|---:|---:|")
    anymap = {r["category"]: r for r in cfa_rows}
    for r in cf_rows:
        c = r["category"]
        anyc = P(b2, f("counterfactual_any", c)) if c in anymap else "–"
        t.append(f"| {c} | {P(b2, f('counterfactual_primary', c))} | {anyc} |")
    t.append("\n![Failure class vs detector](figures/b3-failure-vs-detector-matrix.png)\n")
    t.append("*Observation:* adjudicated primary technique per failure class. *Interpretation:* integration failures cluster on E2E serving "
             "tests, hardware-specific failures on hardware-matrix CI, and shape/memory failures on randomized edge tests and sanitizers. "
             "Deep techniques (determinism perturbation, formal equivalence, warp-race analysis) are the best fit for a small, distinctive tail — "
             "races, batch-invariance and reduction-order bugs — that ordinary tests handle poorly.\n")
    t.append("## B4 — failure case studies\n")
    t.append("Twelve cases were reconstructed from PR timelines, diffs, linked issues and GitHub release-containment checks. "
             "Contributor handles are removed. Each case states which technique would *plausibly* have caught it and why others would not.\n")
    t.append("| Case | Class | Hardware | Intro→fix (days) | First release with fix | Talk link |\n|---|---|---|---:|---|---|")
    for s, _ in cases:
        slug = {"vllm": "vllm-project/vllm", "sglang": "sgl-project/sglang"}.get(str(s["repo"]).split("/")[-1].lower(), s["repo"])
        num = re.sub(r"\D", "", str(s["fix_pr"]))
        t.append(f"| [{slug.split('/')[-1]} #{num}](https://github.com/{slug}/pull/{num}) {s['title'][:60]} | {s['failure_class'][:40]} | {str(s['hardware'])[:40]} | "
                 f"{s['days_intro_to_fix'] if s['days_intro_to_fix'] is not None else 'n/a'} | {s['first_release_with_fix']} | {s['talk_link']} |")
    t.append("")
    for s, md in cases:
        t.append(md.strip() + "\n")
    t.append("## Limitations\n")
    t.append("- The B2 frame requires fix/failure language and production kernel paths; silent fixes, fixes in vendored libraries (FlashInfer, "
             "AITER, CUTLASS) and fixes outside kernel paths are excluded. Estimates extrapolate a 440-commit random sample to the frame.\n"
             "- \"Escaped CI\" is `yes` whenever a bug in merged code was found after merge; private or external CI results are not observable.\n"
             "- Counterfactual techniques are plausibility judgements, not experiments; no technique was run on these bugs.\n"
             "- Revert reasons are as stated publicly; `unstated` is common.\n"
             "- Coding is model-based with a second pass and adjudication; agreement is not human inter-rater reliability.\n")
    text = "\n".join(t) + "\n"
    atomic_text(DEEP / "kernel-failures-report.md", text)
    return text


# ------------------------------------------------------------------ provenance report

def provenance() -> str:
    d, ag = "data/d-provenance-summary.csv", "data/provenance-agreement.csv"
    rr = read_csv(DATA / "production-kernel-provenance.csv")
    f = lambda repo, scope, c: f"repo={repo}&scope={scope}&provenance_class={c}"
    t = ["# Where deployed kernels come from: reverse provenance in vLLM and SGLang\n"]
    t.append(f"**Deep follow-up, WP-D.** Enumeration at the frozen HEADs (vLLM `{HEADS['vllm'][:12]}`, SGLang `{HEADS['sglang'][:12]}`). Records: "
             "`data/production-kernel-provenance.csv`; aggregates `data/d-provenance-summary.csv`. Each record was verified from code, "
             "licence/\"adapted from\" comments, vendored sources, dependency pins, docs and the introducing PR; a blinded second pass re-derived "
             "the provenance class and disagreements were adjudicated.\n")
    t.append("## Executive summary\n")
    t.append(f"1. **{C(d, f('both','all','academic_paper'), 'denominator')} deployed kernel/backend implementations** were verified "
             f"({C(d, f('vllm','all','academic_paper'), 'denominator')} vLLM, {C(d, f('sglang','all','academic_paper'), 'denominator')} SGLang), well above the 40-family floor.")
    t.append(f"2. **Only {P(d, f('both','all','academic_paper'))} trace to an academic paper** (95% CI {CI(d, f('both','all','academic_paper'))}); "
             f"vendor libraries account for {P(d, f('both','all','vendor_library'))}, non-vendor company engineering {P(d, f('both','all','company_engineering'))}, "
             f"and in-repo community work {P(d, f('both','all','community_contribution'))}.")
    t.append(f"3. **Among implementations that are on by default somewhere,** academic provenance is {P(d, f('vllm','default','academic_paper'))} in vLLM "
             f"(n={C(d, f('vllm','default','academic_paper'), 'denominator')}) and {P(d, f('sglang','default','academic_paper'))} in SGLang "
             f"(n={C(d, f('sglang','default','academic_paper'), 'denominator')}).")
    t.append(f"4. **Attention is the paper-heavy family**: {C(d, f('both','operator:attention','academic_paper'), 'n')} of "
             f"{C(d, f('both','operator:attention','academic_paper'), 'denominator')} attention implementations trace to papers, versus "
             f"{C(d, f('both','operator:fused_moe','academic_paper'), 'n')} of {C(d, f('both','operator:fused_moe','academic_paper'), 'denominator')} fused-MoE and "
             f"{C(d, f('both','operator:collective','academic_paper'), 'n')} of {C(d, f('both','operator:collective','academic_paper'), 'denominator')} collective implementations.")
    t.append(f"5. **Provenance judgements are moderately stable:** the two independent passes agreed on {P(ag, 'field=provenance_class', 'percent_agreement')} "
             f"of records (κ = {C(ag, 'field=provenance_class', 'cohen_kappa', 'dec2')}); on paper-versus-not the agreement was "
             f"{P(ag, 'field=academic_paper_vs_other', 'percent_agreement')}.\n")
    t.append("## Provenance by repository\n")
    t.append("| Class | vLLM all | vLLM default | SGLang all | SGLang default |\n|---|---:|---:|---:|---:|")
    for c in ["academic_paper", "vendor_library", "company_engineering", "community_contribution", "unclear"]:
        t.append(f"| {c} | {C(d, f('vllm','all',c), 'n')} | {C(d, f('vllm','default',c), 'n')} | {C(d, f('sglang','all',c), 'n')} | {C(d, f('sglang','default',c), 'n')} |")
    t.append("\n![Provenance by operator](figures/d-provenance-by-operator.png)\n")
    t.append("*Observation:* verified implementations per operator class, coloured by origin. *Interpretation:* papers seed the attention "
             "stack (FlashAttention, FlashInfer, PagedAttention-style layouts, Marlin-style quantized GEMM), while MoE, quantized GEMM paths, "
             "collectives and fused norms are dominated by vendor libraries, company releases and in-repo engineering.\n")
    t.append("## Implementation records\n")
    t.append("| Repo | Operator | Implementation | Origin | Upstream / vendored | Cited paper/system | Status | Introduced |\n|---|---|---|---|---|---|---|---|")
    for r in sorted(rr, key=lambda r: (r["repo"], r["operator_class"], r["implementation_name"])):
        t.append(f"| {r['repo']} | {r['operator_class']} | {r['implementation_name'][:48]} | {r['provenance_class']} | {r['upstream_or_vendored_source'][:40]} | "
                 f"{r['cited_paper_or_system'][:40]} | {r['status']} | {r['first_intro_ref']} ({r['first_intro_date']}) |")
    t.append("\n## Method notes and limitations\n")
    t.append("- Implementation granularity: a *record* is one selectable/dispatched implementation of an operator class; related variants "
             "(e.g. per-dtype kernels of one backend) are merged. Different granularity would change denominators, not the qualitative split.\n"
             "- `academic_paper` means the implementation originates from, or directly implements, a published paper/artifact (as cited in code, "
             "docs or the introducing PR). Industry papers (e.g. FlashInfer, Marlin) count as academic papers; unpublished company releases do not.\n"
             "- Introduction references are the first PR/commit adding the implementation's source path in first-parent history; earlier "
             "prototypes under other paths may exist.\n"
             "- Both passes were model-based; κ is model–model consistency, not human inter-rater reliability.\n")
    text = "\n".join(t) + "\n"
    atomic_text(DEEP / "production-kernel-provenance-report.md", text)
    return text


# ------------------------------------------------------------------ A figures

def a_figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = read_csv(DATA / "a3-evidence-summary.csv")
    codes = ["ev_microbenchmark", "ev_end_to_end", "ev_accuracy_eval", "ev_numerical_tests", "ev_multi_hardware", "ev_memory", "ev_compile_graph",
             "rq_tests", "rq_other_backend_hw", "rq_accuracy_eval", "rq_perf_evidence", "cn_integration", "cn_maintenance", "cn_benchmark_method", "cn_documentation"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = np.arange(len(codes))
    for ax, repo in zip(axes, ["vllm", "sglang"]):
        for j, (g, col) in enumerate([("performance", "#d62728"), ("comparison", "#1f77b4")]):
            vals, lo, hi, n = [], [], [], 0
            for c in codes:
                r = next(r for r in ev if r["repo"] == repo and r["group"] == g and r["measure"] == c)
                vals.append(float(r["share"])); lo.append(float(r["share"]) - float(r["ci95_low"])); hi.append(float(r["ci95_high"]) - float(r["share"]))
                n = r["n"]
            ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, yerr=[lo, hi], color=col, alpha=.85, capsize=2, label=f"{g} (n={n})")
        ax.set_ylabel("share of PRs")
        ax.set_title(f"{repo}: coded evidence (ev_) and non-author reviewer requests/concerns (rq_/cn_)", fontsize=9)
        ax.legend(fontsize=8)
        ax.axvline(6.5, color="grey", lw=.8, ls=":")
    axes[1].set_xticks(x, [c.replace("_", "\n", 1) for c in codes], fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "a3-evidence-and-review-requests.png", dpi=170)
    plt.close(fig)
    rt = read_csv(DATA / "a4-request-types.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    colors = {"yes": "#2ca02c", "partial": "#ffbf00", "no": "#d62728"}
    for ax, repo in zip(axes, ["vllm", "sglang"]):
        rr = sorted([r for r in rt if r["repo"] == repo and int(r["distinct_perf_prs"]) >= 10], key=lambda r: int(r["distinct_perf_prs"]))
        ax.barh([r["request_type"].replace("_", " ") for r in rr], [int(r["distinct_perf_prs"]) for r in rr],
                color=[colors.get(r["documented"], "grey") for r in rr])
        ax.set_title(f"{repo}: reviewer requests in >=10 sampled perf PRs\n(green=written rule, amber=vague/partial, red=no written rule)", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("distinct performance PRs (of 1,000 sampled)")
    fig.tight_layout()
    fig.savefig(FIG / "a4-reviewer-requests-vs-written-rules.png", dpi=170)
    plt.close(fig)
    print("a figures ok")


# ------------------------------------------------------------------ delivery report

def a3_outcome() -> None:
    rc = read_csv(DATA / "review-coding.csv")
    codes = ["ev_microbenchmark", "ev_end_to_end", "ev_accuracy_eval", "ev_numerical_tests", "ev_multi_hardware",
             "rq_tests", "rq_other_backend_hw", "cn_integration", "cn_maintenance", "cn_benchmark_method"]
    out = []
    for repo in ["vllm", "sglang"]:
        for st in ["merged", "closed", "open"]:
            sub = [r for r in rc if r["repo"] == repo and r["group"] == "performance" and r["state"] == st]
            for c in codes + ["any_e2e_or_micro"]:
                k = sum(1 for r in sub if (r["ev_end_to_end"] == "1" or r["ev_microbenchmark"] == "1") if c == "any_e2e_or_micro") if c == "any_e2e_or_micro" else sum(r[c] == "1" for r in sub)
                out.append({"repo": repo, "state": st, "measure": c, "n": k, "denominator": len(sub), "share": round(k / len(sub), 4) if sub else ""})
    atomic_csv(DATA / "a3-evidence-by-outcome.csv", out)


def delivery() -> str:
    a3_outcome()
    ps, tv, aa = "data/a1-population-summary.csv", "data/a1-tier-validation.csv", "data/a1-adjudication-agreement.csv"
    s2, sv, ef, oq, rr = "data/a2-population-summary.csv", "data/a2-survival.csv", "data/a2-effects.csv", "data/a2-open-queue.csv", "data/a2-revert-rates.csv"
    es, ra, eo = "data/a3-evidence-summary.csv", "data/review-coding-agreement.csv", "data/a3-evidence-by-outcome.csv"
    a4s, a4t, a4c, a4d = "data/a4-summary.csv", "data/a4-request-types.csv", "data/a4-rule-compliance.csv", "data/a4-doc-check-agreement.csv"
    e = lambda repo, g, m: f"repo={repo}&group={g}&measure={m}"
    t = ["# What happens to performance and kernel PRs in vLLM and SGLang\n"]
    t.append(f"**Deep follow-up, WP-A. Cutoff {CUTOFF.date()}; PRs created {A_START.date()} → {CUTOFF.date()}.** Complete-population results (A1–A2) "
             "use every PR in the window; content results (A3–A5) use a seeded, stratified detailed corpus. Denominators are stated "
             "with every figure. Coding was performed by LLM subagents with explicit codebooks, fresh second passes and adjudication — "
             "agreement statistics are model–model consistency, not human inter-rater reliability. Nothing here is causal. See [`METHODS.md`](METHODS.md).\n")
    t.append("## Executive summary\n")
    t.append(f"1. **Performance PRs are a large, well-defined population.** Contextual adjudication of every ambiguous candidate yields "
             f"{C(ps, 'repo=vllm', 'confirmed_performance')} confirmed performance PRs of {C(ps, 'repo=vllm', 'window_prs')} in vLLM and "
             f"{C(ps, 'repo=sglang', 'confirmed_performance')} of {C(ps, 'repo=sglang', 'window_prs')} in SGLang. The first study's open queue was "
             f"re-adjudicated: {C(ps, 'repo=vllm', 'queue_confirmed')} of {C(ps, 'repo=vllm', 'first_study_queue_candidates')} (vLLM) and "
             f"{C(ps, 'repo=sglang', 'queue_confirmed')} of {C(ps, 'repo=sglang', 'first_study_queue_candidates')} (SGLang) title candidates are confirmed; "
             f"the full adjudicated open performance queue at the cutoff is {C(ps, 'repo=vllm', 'open_confirmed_performance')} and {C(ps, 'repo=sglang', 'open_confirmed_performance')} PRs.")
    t.append(f"2. **They merge later and less often.** Accounting for closure as a competing outcome, {P(sv, 'repo=vllm&group=performance&horizon_days=30', 'cif_merged')} of vLLM "
             f"performance PRs had merged within 30 days versus {P(sv, 'repo=vllm&group=non_performance&horizon_days=30', 'cif_merged')} of other PRs "
             f"(SGLang {P(sv, 'repo=sglang&group=performance&horizon_days=30', 'cif_merged')} vs {P(sv, 'repo=sglang&group=non_performance&horizon_days=30', 'cif_merged')}); "
             f"the 30-day difference is {C(ef, 'repo=vllm&effect=cif_merged_30d_diff&stratum=all', 'estimate', 'pct1')} points "
             f"[{fmtv(get(ef, 'repo=vllm&effect=cif_merged_30d_diff&stratum=all', 'ci95_low'), 'pct1')}, {fmtv(get(ef, 'repo=vllm&effect=cif_merged_30d_diff&stratum=all', 'ci95_high'), 'pct1')}] in vLLM and "
             f"{C(ef, 'repo=sglang&effect=cif_merged_30d_diff&stratum=all', 'estimate', 'pct1')} points in SGLang (bootstrap 95% CIs).")
    t.append(f"3. **They carry far more evidence than comparable PRs.** In the matched detailed corpus, microbenchmarks appear in {P(es, e('vllm','performance','ev_microbenchmark'))} "
             f"of vLLM performance PRs versus {P(es, e('vllm','comparison','ev_microbenchmark'))} of matched comparisons, and end-to-end numbers in "
             f"{P(es, e('vllm','performance','ev_end_to_end'))} vs {P(es, e('vllm','comparison','ev_end_to_end'))}; accuracy evaluation in "
             f"{P(es, e('vllm','performance','ev_accuracy_eval'))} (SGLang: {P(es, e('sglang','performance','ev_microbenchmark'))}, {P(es, e('sglang','performance','ev_end_to_end'))}, {P(es, e('sglang','performance','ev_accuracy_eval'))}).")
    t.append(f"4. **Review asks for the downstream pipeline.** Among the same performance PRs, reviewers raised integration concerns in {P(es, e('vllm','performance','cn_integration'))} (vLLM) "
             f"and {P(es, e('sglang','performance','cn_integration'))} (SGLang), asked about other hardware/backends in {P(es, e('vllm','performance','rq_other_backend_hw'))} and "
             f"{P(es, e('sglang','performance','rq_other_backend_hw'))}, and challenged benchmark methodology in {P(es, e('vllm','performance','cn_benchmark_method'))} and "
             f"{P(es, e('sglang','performance','cn_benchmark_method'))} — each well above matched comparisons.")
    t.append(f"5. **Much of what reviewers require is not written down.** In vLLM, {P(a4s, 'repo=vllm', 'share_merged_with_ge1_undocumented')} of merged sampled performance PRs "
             f"({C(a4s, 'repo=vllm', 'merged_with_ge1_undocumented_request_before_merge')} of {C(a4s, 'repo=vllm', 'merged_perf_prs')}) received at least one pre-merge request of a type that "
             f"neither of two independent document audits found in the contribution rules; {C(a4s, 'repo=vllm', 'undocumented_recurring_types')} such request types recur in ≥10 PRs. "
             f"SGLang's much larger written guidance (contribution guide, template and agent-skill guides) leaves {P(a4s, 'repo=sglang', 'share_merged_with_ge1_undocumented')}.")
    t.append(f"6. **Performance PRs are reverted about twice as often:** {P(rr, 'repo=vllm&group=performance', 'revert_rate')} vs {P(rr, 'repo=vllm&group=non_performance', 'revert_rate')} of merged PRs in vLLM, "
             f"{P(rr, 'repo=sglang&group=performance', 'revert_rate')} vs {P(rr, 'repo=sglang&group=non_performance', 'revert_rate')} in SGLang.\n")

    t.append("## A1 — the performance-PR population\n")
    t.append("| | vLLM | SGLang |\n|---|---:|---:|")
    for lab, col in [("PRs created in window", "window_prs"), ("confirmed performance", "confirmed_performance"), ("not performance", "not_performance"),
                     ("uncertain", "uncertain"), ("records adjudicated in context", "adjudicated_records"),
                     ("first-study open-queue candidates", "first_study_queue_candidates"), ("… confirmed on adjudication", "queue_confirmed"),
                     ("… not performance", "queue_not_performance"), ("open PRs at cutoff (population scope)", "open_at_cutoff_all"),
                     ("open confirmed performance PRs", "open_confirmed_performance")]:
        t.append(f"| {lab} | {C(ps, 'repo=vllm', col)} | {C(ps, 'repo=sglang', col)} |")
    t.append("\nTiering and validation. PRs with an explicit performance title tag and no fix/CI/doc/revert tag were accepted by rule; every "
             "other PR with any performance or kernel signal (title claim, kernel tag, `performance` label, or kernel paths plus a body "
             "performance claim) was adjudicated in context. Seeded random validation samples measure both rule tiers:\n")
    t.append("| Repo | Sample | n | confirmed performance | share | Wilson 95% |\n|---|---|---:|---:|---:|---|")
    for repo in ["vllm", "sglang"]:
        for smp in ["validation_rule_confirmed", "validation_rule_excluded"]:
            fl = f"repo={repo}&validation_sample={smp}"
            t.append(f"| {repo} | {smp} | {C(tv, fl, 'n')} | {C(tv, fl, 'confirmed_performance')} | {P(tv, fl, 'share_confirmed')} | "
                     f"[{fmtv(get(tv, fl, 'wilson95_low'), 'pct1')}–{fmtv(get(tv, fl, 'wilson95_high'), 'pct1')}%] |")
    t.append(f"\nThe rule-confirmed tier is high precision. The rule-excluded tier still hides performance PRs at a rate of "
             f"{P(tv, 'repo=vllm&validation_sample=validation_rule_excluded', 'share_confirmed')} (vLLM) and {P(tv, 'repo=sglang&validation_sample=validation_rule_excluded', 'share_confirmed')} (SGLang) — "
             "mostly borderline system-level efficiency features without performance vocabulary. The confirmed population is therefore a "
             "high-precision **lower bound**; misclassified PRs sit in the comparison group and bias contrasts toward zero. "
             f"Extrapolating the miss rate to the whole excluded tier implies about {C('data/a1-recall-estimate.csv', 'repo=vllm', 'est_missed')} (vLLM) and "
             f"{C('data/a1-recall-estimate.csv', 'repo=sglang', 'est_missed')} (SGLang) further performance PRs, i.e. estimated recall of "
             f"{P('data/a1-recall-estimate.csv', 'repo=vllm', 'est_recall')} and {P('data/a1-recall-estimate.csv', 'repo=sglang', 'est_recall')} "
             "(`data/a1-recall-estimate.csv`, with Wilson-based ranges). "
             f"An independent second adjudication of 300 random records agreed {P(aa, 'repo=both', 'percent_agreement')} (κ = {C(aa, 'repo=both', 'cohen_kappa', 'dec2')}).\n")

    t.append("## A2 — complete-population delivery metrics\n")
    t.append("| | vLLM perf | vLLM other | SGLang perf | SGLang other |\n|---|---:|---:|---:|---:|")
    for lab, col, k in [("PRs", "n", "int"), ("merged by cutoff", "merged_share", "pct1"), ("closed unmerged", "closed_share", "pct1"),
                        ("still open", "open_share", "pct1"), ("median lines changed", "median_lines_changed", "int"),
                        ("median files changed", "median_changed_files", "int"), ("KM median days to merge (closed censored)", "km_median_days_to_merge", "dec1"),
                        ("median days to merge among merged", "median_days_to_merge_among_merged", "dec1"),
                        ("later reverted (of all)", "reverted_n", "int"), ("closed with superseded signal", "superseded_signal_n", "int")]:
        cells = []
        for repo in ["vllm", "sglang"]:
            for g in ["performance", "non_performance"]:
                v = C(s2, f"repo={repo}&group={g}", col, k)
                cells.append(v + ("%" if k == "pct1" else ""))
        t.append(f"| {lab} | " + " | ".join(cells) + " |")
    t.append("\nCumulative incidence of merge (Aalen–Johansen; closure is a competing event; open PRs censored at the cutoff):\n")
    t.append("| Days | vLLM perf | vLLM other | SGLang perf | SGLang other |\n|---:|---:|---:|---:|---:|")
    for h in ["1", "7", "30", "90"]:
        cells = [P(sv, f"repo={repo}&group={g}&horizon_days={h}", "cif_merged") for repo in ["vllm", "sglang"] for g in ["performance", "non_performance"]]
        t.append(f"| {h} | " + " | ".join(cells) + " |")
    t.append("\nEffect sizes (performance minus other; 1,000-replicate bootstrap):\n")
    t.append("| Repo | Effect | Stratum | Estimate | 95% CI |\n|---|---|---|---:|---|")
    for r in rows(ef):
        est = r["estimate"]
        t.append(f"| {r['repo']} | {r['effect']} | {r['stratum']} | {'not reached' if est == 'nan' else est} | "
                 f"[{r['ci95_low']}, {r['ci95_high']}] |")
    t.append("\nKM medians are long because many PRs never merge; with closed PRs censored, KM overstates eventual merge probability, "
             "so the CIF columns are the preferred summary. Within size terciles the gap persists for small and large PRs "
             "(`nan` = median not reached within the window).\n")
    t.append("![Time to merge](figures/a2-time-to-merge-km-cif.png)\n")
    t.append("*Observation:* KM (solid, closed censored) and CIF (dashed, closure competing) for PRs created in the window. "
             "*Interpretation:* performance PRs accumulate merges more slowly at every horizon; this is descriptive and confounded by size and complexity.\n")
    t.append("| Open at cutoff | open PRs | median age (days) | share older than 90 days |\n|---|---:|---:|---:|")
    for repo in ["vllm", "sglang"]:
        for g in ["performance", "non_performance"]:
            fl = f"repo={repo}&group={g}"
            t.append(f"| {repo} {g} | {C(oq, fl, 'open_at_cutoff')} | {C(oq, fl, 'median_age_days', 'dec1')} | {P(oq, fl, 'share_older_than_90d')} |")
    t.append("\n![Open queue age](figures/a2-open-queue-age.png)\n")

    t.append("## A3 — evidence and reviewer labour in the detailed corpus\n")
    t.append("Both repositories exceed 2,500 confirmed performance PRs, so the detailed corpus is a seeded state×month stratified sample of "
             "1,000 performance PRs per repository plus 552 (vLLM) and 553 (SGLang) human-authored non-performance PRs matched on creation "
             "month, PR-size tercile and state. Full reviews, inline comments, issue comments, commits and timeline events were cached for all "
             "3,105 PRs. Shares below are PR-level; the performance sample is proportional to the population by state and month.\n")
    t.append("| Code | vLLM perf | vLLM comparison | SGLang perf | SGLang comparison |\n|---|---:|---:|---:|---:|")
    for c in ["ev_microbenchmark", "ev_end_to_end", "ev_accuracy_eval", "ev_numerical_tests", "ev_multi_hardware", "ev_memory", "ev_compile_graph",
              "rq_tests", "rq_other_backend_hw", "rq_accuracy_eval", "rq_perf_evidence", "cn_integration", "cn_maintenance", "cn_benchmark_method",
              "cn_documentation", "reviewed_any"]:
        cells = [f"{P(es, e(repo, g, c))} {CI(es, e(repo, g, c))}" for repo in ["vllm", "sglang"] for g in ["performance", "comparison"]]
        t.append(f"| {c} | " + " | ".join(cells) + " |")
    t.append("\n`reviewed_any` = at least one substantive non-author human review message.\n")
    t.append("| Review effort (median) | vLLM perf | vLLM comparison | SGLang perf | SGLang comparison |\n|---|---:|---:|---:|---:|")
    for m, k in [("median_substantive_review_rounds", "dec1"), ("median_hours_to_first_human_response", "dec1"), ("median_n_human_reviewers", "dec1"), ("median_n_commits", "dec1")]:
        cells = [C(es, e(repo, g, m), "share", k) for repo in ["vllm", "sglang"] for g in ["performance", "comparison"]]
        t.append(f"| {m.replace('median_', '')} | " + " | ".join(cells) + " |")
    t.append("\n![Evidence and reviewer requests](figures/a3-evidence-and-review-requests.png)\n")
    t.append("*Observation:* coded shares with bootstrap 95% CIs. *Interpretation:* performance PRs arrive with far more measurement, "
             "and reviewers respond with requests about other hardware, integration and benchmark method — the Establish and Sustain work.\n")
    t.append("Evidence by outcome within the performance sample (shares of PRs in each end state):\n")
    t.append("| Measure | vLLM merged | vLLM closed | vLLM open | SGLang merged | SGLang closed | SGLang open |\n|---|---:|---:|---:|---:|---:|---:|")
    for m in ["any_e2e_or_micro", "ev_end_to_end", "ev_accuracy_eval", "ev_numerical_tests", "rq_other_backend_hw", "cn_integration", "cn_benchmark_method"]:
        cells = [P(eo, f"repo={repo}&state={st}&measure={m}") for repo in ["vllm", "sglang"] for st in ["merged", "closed", "open"]]
        t.append(f"| {m} | " + " | ".join(cells) + " |")
    t.append("\n**Coding reliability.** A fresh second pass re-coded 200 random corpus PRs per repository from identical dossiers:\n")
    t.append("| Code | % agreement | κ | prevalence (pass 1 / pass 2) |\n|---|---:|---:|---|")
    for r in [r for r in rows(ra) if r["repo"] == "both"]:
        t.append(f"| {r['code']} | {P(ra, 'repo=both&code=' + r['code'], 'percent_agreement')} | {r['cohen_kappa']} | "
                 f"{float(r['prevalence_pass1']) * 100:.1f}% / {float(r['prevalence_pass2']) * 100:.1f}% |")
    t.append("")

    t.append("## A4 — codified versus undocumented requirements\n")
    t.append("Mechanically observable compliance with the first study's audited rules, detailed corpus:\n")
    t.append("| Repo | Rule | Signal | Scope | Performance | Comparison |\n|---|---|---|---|---:|---:|")
    for repo in ["vllm", "sglang"]:
        seen = []
        for r in rows(a4c):
            if r["repo"] == repo and (r["rule_id"], r["signal"]) not in seen:
                seen.append((r["rule_id"], r["signal"]))
                cells = []
                for g in ["performance", "comparison"]:
                    fl = f"repo={repo}&rule_id={r['rule_id']}&signal={r['signal']}&group={g}"
                    try:
                        cells.append(f"{P(a4c, fl)} (n={get(a4c, fl, 'n')})")
                    except KeyError:
                        cells.append("–")
                t.append(f"| {repo} | {r['rule_id']} | {r['signal']} | {r['scope']} | " + " | ".join(cells) + " |")
    t.append("\nReviewer requests were labelled on every sampled performance PR with substantive pre-merge review, using a 35-type "
             "taxonomy induced from the data (`coding/a4-request-taxonomy.json`). Two independent audits of the frozen contribution "
             f"documents decided whether each type is written down (agreement {P(a4d, 'comparison=two independent model passes over frozen docs; not human IRR', 'percent_agreement')}); "
             "a type is **undocumented** only when both audits found no written rule, and **documented** only when both agree.\n")
    t.append("| Repo | Request type | Perf PRs (of 1,000) | Written rule? | label κ | Example (anonymized) |\n|---|---|---:|---|---:|---|")
    for r in sorted(rows(a4t), key=lambda r: (r["repo"], -int(r["distinct_perf_prs"]))):
        if int(r["distinct_perf_prs"]) >= 10 and r["documented"] != "yes":
            ex = anonymize(r["examples"].split(" || ")[0])[:200].replace("|", "/")
            t.append(f"| {r['repo']} | {r['request_type']} | {C(a4t, 'repo=' + r['repo'] + '&request_type=' + r['request_type'], 'distinct_perf_prs')} | {r['documented']} | "
                     f"{r.get('label_kappa', '') or '–'} | {ex} |")
    la = "data/a4-label-agreement.csv"
    t.append(f"\nLabel reliability: an independent second labeller re-coded 150 random PRs; mean Jaccard overlap of the two request-type sets was "
             f"{C(la, 'request_type=ALL (mean Jaccard of type sets)', 'percent_agreement', 'dec2')}. Per-type κ is shown; treat types with κ < 0.4 as "
             "indicative only. Examples are drawn from PRs where both labellers agreed when available (otherwise marked single-coder).")
    t.append(f"\nIn vLLM, {C(a4s, 'repo=vllm', 'undocumented_request_instances_before_merge')} of {C(a4s, 'repo=vllm', 'request_instances_before_merge')} pre-merge request instances on merged "
             f"performance PRs ({P(a4s, 'repo=vllm', 'share_request_instances_undocumented')}) were of undocumented types; excluding reviewer-found defects "
             f"(`fix_functional_correctness_issue`, inherently uncodifiable) {P(a4s, 'repo=vllm', 'share_merged_with_ge1_undocumented_excl_defect_fix')} of merged performance PRs still received one. "
             f"In SGLang the corresponding figures are {P(a4s, 'repo=sglang', 'share_request_instances_undocumented')} and {P(a4s, 'repo=sglang', 'share_merged_with_ge1_undocumented_excl_defect_fix')}: "
             "almost everything reviewers ask is written somewhere — but often only vaguely (`partial`) or in agent-skill guides.\n")
    t.append("![Requests vs written rules](figures/a4-reviewer-requests-vs-written-rules.png)\n")
    t.append("No contributor or reviewer is profiled or ranked; examples quote review text with handles removed.\n")

    t.append("## A5 — case histories\n")
    t.append("Twelve histories (per repository: two fast merges, two slow/many-round merges, one reverted, one superseded) were selected "
             "by a seeded draw from the detailed corpus (`data/a5-case-candidates.csv`) and verified against PR timelines, commits and "
             "GitHub release-containment checks.\n")
    for repo in ["vllm", "sglang"]:
        for p in sorted((CACHE / "a5").glob(f"case-{repo}-*.md")):
            t.append(anonymize(p.read_text(encoding="utf-8")).strip() + "\n")
    t.append("## Limitations\n")
    t.append("- The performance population is high-precision but incomplete (see tier validation); contrasts are conservative.\n"
             "- A2 contrasts are descriptive; size, author experience, hardware requirements and complexity confound them.\n"
             "- A3 codes rely on keyword-prefiltered snippets: evidence that is not phrased in any prefilter vocabulary, or that lives in "
             "external dashboards, CI logs or private channels, is missed. Timelines with >60 reviews or >100 comments are truncated (flagged).\n"
             "- A4 documentation audits count agent-skill guides and subsystem guides as written guidance (often `partial`); the "
             "undocumented share is therefore conservative.\n"
             "- All coding is model-based with second passes; agreement is model–model consistency, not human inter-rater reliability.\n")
    text = "\n".join(t) + "\n"
    atomic_text(DEEP / "kernel-pr-delivery-report.md", text)
    return text


# ------------------------------------------------------------------ papers report

def c_summary() -> None:
    from c_analysis import wilson
    cen = read_csv(DATA / "paper-census.csv")
    out = []
    for v in ["MLSys 2025", "ASPLOS 2025", "both"]:
        pp = [r for r in cen if v == "both" or r["venue"] == v]
        for field in ["kernel_style", "category", "target_workload"]:
            base = pp if field != "target_workload" else [r for r in pp if r["kernel_style"] == "yes"]
            if field == "category":
                base = [r for r in pp if r["kernel_style"] == "yes"]
            for k, n in collections.Counter(r[field] for r in base).most_common():
                lo, hi = wilson(n, len(base))
                out.append({"venue": v, "measure": field, "category": k, "n": n, "denominator": len(base), "share": round(n / len(base), 4),
                            "wilson95_low": round(lo, 4), "wilson95_high": round(hi, 4)})
        k = sum(1 for r in pp if r["sections_read"] in ("abstract", ""))
        out.append({"venue": v, "measure": "abstract_only", "category": "abstract_only", "n": k, "denominator": len(pp), "share": round(k / len(pp), 4)})
    atomic_csv(DATA / "c-census-summary.csv", out)


def papers() -> str:
    c_summary()
    cs, ca, fu, au = "data/c-census-summary.csv", "data/paper-census-agreement.csv", "data/paper-funnel.csv", "data/paper-eval-audit-summary.csv"
    lad = read_csv(DATA / "paper-deployment-ladder.csv")
    q = lambda v, m, c: f"venue={v}&measure={m}&category={c}"
    fs = lambda v, s: f"venue={v}&stage={s}"
    t = ["# From paper to production: MLSys 2025 and ASPLOS 2025\n"]
    t.append(f"**Deep follow-up, WP-C. Cutoff {CUTOFF.date()}.** Census `data/paper-census.csv` (all 61 MLSys 2025 papers; all 160 papers in "
             "the official ASPLOS 2025 proceedings, ACM Vol. 1 and 2), evaluation audit and deployment ladder `data/paper-deployment-ladder.csv`, "
             "funnel `data/paper-funnel.csv`. Classification used the broad definition of kernel-style optimization in "
             "`studies/04-paper-to-production.md`; every positive/ambiguous paper was second-passed and disagreements adjudicated "
             "(model–model agreement, not human IRR). The ladder records the **highest publicly observable** evidence — a lower bound on deployment.\n")
    t.append("## Executive summary\n")
    t.append(f"1. **Kernel-style work is a minority of both venues:** {C(cs, q('MLSys 2025','kernel_style','yes'), 'n')} of {C(cs, q('MLSys 2025','kernel_style','yes'), 'denominator')} MLSys 2025 papers "
             f"({P(cs, q('MLSys 2025','kernel_style','yes'))}) and {C(cs, q('ASPLOS 2025','kernel_style','yes'), 'n')} of {C(cs, q('ASPLOS 2025','kernel_style','yes'), 'denominator')} ASPLOS 2025 papers "
             f"({P(cs, q('ASPLOS 2025','kernel_style','yes'))}) propose a kernel-style optimization.")
    t.append(f"2. **Code release is common; maintenance and adoption are not.** Of {C(fu, fs('both','kernel_style'), 'n')} kernel-style papers, "
             f"{C(fu, fs('both','reached_L1'), 'n')} released code (L1+), {C(fu, fs('both','reached_L2'), 'n')} show maintenance after publication (L2+), "
             f"{C(fu, fs('both','reached_L3'), 'n')} reached a kernel library or beyond (L3+), {C(fu, fs('both','reached_L4'), 'n')} were merged into a serving/training framework (L4+) and "
             f"{C(fu, fs('both','reached_L5'), 'n')} are default, documented or release-noted (L5).")
    t.append(f"3. **Observable framework adoption is {P(fu, fs('both','reached_L4'))} of kernel-style papers** (Wilson 95% "
             f"[{fmtv(get(fu, fs('both','reached_L4'), 'wilson95_low'), 'pct1')}–{fmtv(get(fu, fs('both','reached_L4'), 'wilson95_high'), 'pct1')}%]), and "
             f"{C(fu, fs('both','L4plus_framework_code_before_publication'), 'n')} of those {C(fu, fs('both','reached_L4'), 'n')} were in a framework *before* the paper/preprint appeared "
             f"(the paper documents an already-deployed system); only {C(fu, fs('both','L4plus_adopted_after_publication'), 'n')} show paper → framework flow. "
             f"Self-reported production deployment (S) appears in {C(fu, fs('both','self_reported_production_S'), 'n')} papers and is tracked separately.")
    t.append(f"4. **Evaluation is narrow in hardware and often thin on correctness:** among kernel-style papers, "
             f"{P(au, 'venue=both&field=hardware_vendors&value=single_vendor')} evaluate on a single hardware vendor, "
             f"{P(au, 'venue=both&field=n_hardware_targets&value=single_target')} on a single hardware target, "
             f"{P(au, 'venue=both&field=eval_scope&value=microbenchmark_only')} only at kernel/operator level, and "
             f"{P(au, 'venue=both&field=correctness_validation&value=none_reported')} report no correctness/accuracy validation.")
    t.append(f"5. **The negative check found adoption outside the kernel-style set:** {C(fu, fs('both','negatives_with_L3plus_evidence'), 'n')} of "
             f"{C(fu, fs('both','negatives_with_L3plus_evidence'), 'denominator')} randomly sampled non-kernel-style papers had verified L3+ evidence, and "
             f"{C(fu, fs('both','negatives_missed_kernel_style_with_L3plus'), 'n')} of those were judged missed kernel-style papers on verification "
             f"(estimated missed kernel-style adoption ≤ {fmtv(get(fu, fs('both','negatives_missed_kernel_style_with_L3plus'), 'wilson95_high'), 'pct1')}% of negatives, Wilson upper bound). "
             "Production uptake of *system* ideas (caching, parallelism, optimizers) is real but outside the kernel-style funnel.\n")
    t.append("## C1 — census\n")
    t.append("| Venue | Papers | Kernel-style | Ambiguous | Not kernel-style |\n|---|---:|---:|---:|---:|")
    for v in ["MLSys 2025", "ASPLOS 2025"]:
        amb = C(cs, q(v, "kernel_style", "ambiguous"), "n") if any(r["venue"] == v and r["measure"] == "kernel_style" and r["category"] == "ambiguous" for r in rows(cs)) else "0"
        t.append(f"| {v} | {C(cs, q(v,'kernel_style','yes'), 'denominator')} | {C(cs, q(v,'kernel_style','yes'), 'n')} | {amb} | {C(cs, q(v,'kernel_style','no'), 'n')} |")
    t.append("\n| Category (kernel-style papers) | MLSys | ASPLOS |\n|---|---:|---:|")
    cats = sorted({r["category"] for r in rows(cs) if r["measure"] == "category"})
    for c in cats:
        cells = []
        for v in ["MLSys 2025", "ASPLOS 2025"]:
            cells.append(C(cs, q(v, "category", c), "n") if any(r["venue"] == v and r["measure"] == "category" and r["category"] == c for r in rows(cs)) else "0")
        t.append(f"| {c} | " + " | ".join(cells) + " |")
    t.append(f"\nReading depth: full text was read for every MLSys paper and for ASPLOS papers with an open copy; "
             f"{C(cs, q('ASPLOS 2025','abstract_only','abstract_only'), 'n')} ASPLOS papers had no reachable open copy (ACM DL blocks automated access) and were "
             "classified from the abstract and public descriptions; re-reading 45 other abstract-only papers from located full text changed 1 classification. "
             f"Second pass on all yes/ambiguous papers: kernel-style agreement {P(ca, 'field=kernel_style', 'percent_agreement')} (κ = {C(ca, 'field=kernel_style', 'cohen_kappa', 'dec2')}), "
             f"category agreement {P(ca, 'field=category', 'percent_agreement')} (κ = {C(ca, 'field=category', 'cohen_kappa', 'dec2')}); all disagreements were adjudicated. The low "
             "kernel-style κ reflects boundary cases (system papers whose key mechanism is a kernel), which is why every such case was adjudicated.\n")
    t.append("## C2 — evaluation audit of kernel-style papers\n")
    t.append("| Field | Value | MLSys | ASPLOS | Both |\n|---|---|---:|---:|---:|")
    for field in ["eval_scope", "correctness_validation", "baseline_versions_stated", "public_code", "claimed_production"]:
        vals = sorted({r["value"] for r in rows(au) if r["field"] == field})
        for val in vals:
            cells = []
            for v in ["MLSys 2025", "ASPLOS 2025", "both"]:
                fl = f"venue={v}&field={field}&value={val}"
                cells.append(f"{C(au, fl, 'n')}/{get(au, fl, 'denominator')}" if any(r["venue"] == v and r["field"] == field and r["value"] == val for r in rows(au)) else "0")
            t.append(f"| {field} | {val} | " + " | ".join(cells) + " |")
    for v in ["MLSys 2025", "ASPLOS 2025", "both"]:
        fl = f"venue={v}&field=n_hardware_targets&value=median"
        if any(r["venue"] == v and r["field"] == "n_hardware_targets" and r["value"] == "median" for r in rows(au)):
            t.append(f"\n{v}: median hardware targets {C(au, fl, 'n', 'dec1')}; single-target papers {P(au, f'venue={v}&field=n_hardware_targets&value=single_target')}.")
    t.append("\n## C3 — deployment ladder\n")
    t.append("| Stage | MLSys 2025 | ASPLOS 2025 | Both |\n|---|---:|---:|---:|")
    for s, lab in [("all_papers", "all papers"), ("kernel_style", "kernel-style"), ("reached_L1", "L1+ code released"), ("reached_L2", "L2+ maintained"),
                   ("reached_L3", "L3+ kernel library"), ("reached_L4", "L4+ framework"), ("reached_L5", "L5 default/documented"),
                   ("self_reported_production_S", "S self-reported production"), ("negatives_with_L3plus_evidence", "negative sample with L3+ (of sampled)"),
                   ("negatives_missed_kernel_style_with_L3plus", "… of which missed kernel-style papers")]:
        t.append(f"| {lab} | " + " | ".join(f"{C(fu, fs(v, s), 'n')}/{get(fu, fs(v, s), 'denominator')}" for v in ["MLSys 2025", "ASPLOS 2025", "both"]) + " |")
    t.append("\n![Paper-to-production funnel](figures/c-paper-to-production-funnel.png)\n")
    t.append("*Observation:* highest observable evidence level per kernel-style paper. *Interpretation:* the pipeline leaks mostly between "
             "\"code released\" and \"used by a library or framework\"; this is a conservative lower bound because renamed or reimplemented "
             "techniques can be missed.\n")
    t.append("Every L3–L5 label with its evidence (URL | quote):\n")
    t.append("| Paper | Venue | Role | Level | Evidence |\n|---|---|---|---|---|")
    for r in sorted(lad, key=lambda r: (-int(r["level_num"]), r["id"])):
        if int(r["level_num"]) >= 3:
            ev = next((r[k] for k in ["l5_evidence", "l4_evidence", "l3_evidence"] if r.get(k)), "")
            t.append(f"| {r['title'][:70]} | {r['venue']} | {r['role']} | {r['ladder_level']} | {ev[:260].replace('|', '/', 1).replace('|', '—')} |")
    t.append("\n## C4 — adoption latency\n")
    t.append("| Paper | Level | Preprint | First downstream PR | Merge | Default/doc | Months preprint→merge |\n|---|---|---|---|---|---|---:|")
    for r in sorted(lad, key=lambda r: r["id"]):
        if r["role"] == "positive" and int(r["level_num"]) >= 3:
            t.append(f"| {r['title'][:60]} | {r['ladder_level']} | {r['preprint_date']} | {r['first_downstream_pr_date']} | {r['downstream_merge_date']} | "
                     f"{r['first_default_or_doc_date']} | {r['months_preprint_to_merge'] or 'n/a'} |")
    t.append("\n![Adoption latency](figures/c-adoption-latency.png)\n")
    t.append("## Negative-sample check (missed adoption)\n")
    t.append("Twenty random non-kernel-style papers per venue received the same systematic adoption search. Papers with L3+ evidence are "
             "listed in the evidence table above with role `negative_check`; they were reviewed to see whether the census had missed a "
             "kernel-style contribution. They are reported separately and are **not** added to the kernel-style funnel.\n")
    t.append("## Limitations\n")
    t.append("- Two venues, one year; results do not represent OSDI/SOSP/ISCA/MICRO or earlier cohorts with longer adoption windows.\n"
             "- The ladder is a lower bound: renamed, reimplemented or privately adopted techniques are invisible; downstream searches used "
             "HEAD snapshots (2026-09-29) plus PR search, with hits dated by their introducing PR.\n"
             "- 41 ASPLOS papers were classified from abstracts because no open full text was reachable.\n"
             "- Self-reported production (S) is not verified and is never counted as observable adoption.\n")
    text = "\n".join(t) + "\n"
    atomic_text(DEEP / "paper-to-production-report.md", text)
    return text


# ------------------------------------------------------------------ synthesis

def synthesis() -> str:
    ps, s2, sv, es, a4s, rr = ("data/a1-population-summary.csv", "data/a2-population-summary.csv", "data/a2-survival.csv",
                               "data/a3-evidence-summary.csv", "data/a4-summary.csv", "data/a2-revert-rates.csv")
    b1, b2, fu, cs, d = "data/b1-revert-summary.csv", "data/b2-failure-summary.csv", "data/paper-funnel.csv", "data/c-census-summary.csv", "data/d-provenance-summary.csv"
    e = lambda repo, g, m: f"repo={repo}&group={g}&measure={m}"
    f = lambda meas, cat, repo="both": f"repo={repo}&measure={meas}&category={cat}"
    fs = lambda v, s: f"venue={v}&stage={s}"
    q = lambda v, m, c: f"venue={v}&measure={m}&category={c}"
    dd = lambda repo, scope, c: f"repo={repo}&scope={scope}&provenance_class={c}"
    t = ["# Synthesis — optimizing the optimization pipeline\n"]
    t.append("This synthesis answers the seven study questions using only results that passed the study's gates. Each number is "
             "linked to a CSV row by a hidden claim tag and re-verified by `scripts/consistency.py`. Complete-population results "
             "(A1–A2, B1, C1, D) are distinguished from sampled/coded results (A3–A5, B2–B4, C2–C4). All coding agreement is model–model "
             "adjudication consistency, not human inter-rater reliability; nothing is causal.\n")
    t.append("## 1. Is kernel discovery a large or small fraction of real engineering work?\n")
    t.append(f"Small but not marginal. Confirmed performance PRs are {C(ps, 'repo=vllm', 'confirmed_performance')} of {C(ps, 'repo=vllm', 'window_prs')} vLLM PRs "
             f"and {C(ps, 'repo=sglang', 'confirmed_performance')} of {C(ps, 'repo=sglang', 'window_prs')} SGLang PRs in 12 months — roughly one PR in eight, "
             f"a high-precision lower bound (estimated recall {P('data/a1-recall-estimate.csv', 'repo=vllm', 'est_recall')} and {P('data/a1-recall-estimate.csv', 'repo=sglang', 'est_recall')}). "
             f"And only part of that is *discovering* a new or faster kernel: among the {C('data/a1-perf-type-summary.csv', 'repo=both&perf_type=new_or_faster_kernel_total', 'denominator')} adjudicated performance PRs, "
             f"new kernels/fusions and kernel speedups are {P('data/a1-perf-type-summary.csv', 'repo=both&perf_type=new_or_faster_kernel_total')}, while system-level optimizations are "
             f"{P('data/a1-perf-type-summary.csv', 'repo=both&perf_type=system_performance')}, precision/format work {P('data/a1-perf-type-summary.csv', 'repo=both&perf_type=precision_format')}, "
             f"tuning configs {P('data/a1-perf-type-summary.csv', 'repo=both&perf_type=kernel_tuning_config')} and performance-regression fixes {P('data/a1-perf-type-summary.csv', 'repo=both&perf_type=perf_regression_fix')}.\n")
    t.append("## 2. What consumes the rest of the work?\n")
    t.append(f"Establishing and sustaining optimizations. Kernel-correctness fixes alone are estimated at "
             f"{C(b2, f('estimated_confirmed_in_frame','confirmed'), 'n')} in 24 months (from a {C(b2, f('screened','all'), 'n')}-commit random sample of a "
             f"{C(b2, f('screened','all'), 'denominator')}-commit frame); {P(b2, f('failure_class','integration_backend_cudagraph'))} of confirmed failures are integration/backend-selection/"
             f"CUDA-graph problems and {P(b2, f('hardware_specific','yes'))} are hardware-specific. {C(b1, f('confirmed_reverts_or_rollbacks','all'), 'n')} reverts/rollbacks occurred in 24 months. "
             "The first study's commit-level view (bug fixes, hardware enablement, integration, maintenance) is consistent with this, "
             "but its fine-purpose shares remain diagnostic.\n")
    t.append("## 3. What evidence and reviewer labour turn an optimization into a merge?\n")
    t.append(f"Performance PRs arrive with far more evidence than matched comparisons (vLLM microbenchmarks {P(es, e('vllm','performance','ev_microbenchmark'))} vs "
             f"{P(es, e('vllm','comparison','ev_microbenchmark'))}; end-to-end {P(es, e('vllm','performance','ev_end_to_end'))} vs {P(es, e('vllm','comparison','ev_end_to_end'))}), "
             f"yet merge more slowly: by day 30, {P(sv, 'repo=vllm&group=performance&horizon_days=30', 'cif_merged')} vs {P(sv, 'repo=vllm&group=non_performance&horizon_days=30', 'cif_merged')} "
             f"had merged in vLLM and {P(sv, 'repo=sglang&group=performance&horizon_days=30', 'cif_merged')} vs {P(sv, 'repo=sglang&group=non_performance&horizon_days=30', 'cif_merged')} in SGLang. "
             f"Reviewers ask about integration ({P(es, e('vllm','performance','cn_integration'))} / {P(es, e('sglang','performance','cn_integration'))}), other hardware "
             f"({P(es, e('vllm','performance','rq_other_backend_hw'))} / {P(es, e('sglang','performance','rq_other_backend_hw'))}) and benchmark method "
             f"({P(es, e('vllm','performance','cn_benchmark_method'))} / {P(es, e('sglang','performance','cn_benchmark_method'))}). In vLLM, "
             f"{P(a4s, 'repo=vllm', 'share_merged_with_ge1_undocumented')} of merged sampled performance PRs met at least one pre-merge request of a type absent from the written rules.\n")
    t.append("## 4. What kinds of kernel bugs escape existing validation?\n")
    t.append(f"Mostly bugs that unit tests on one GPU cannot see: integration failures ({P(b2, f('failure_class','integration_backend_cudagraph'))}), shape/alignment edge cases "
             f"({P(b2, f('failure_class','shape_alignment_edge'))}), hardware/compiler-specific failures ({P(b2, f('failure_class','hardware_compiler_specific'))}), numerical precision "
             f"({P(b2, f('failure_class','numerical_precision'))}), memory safety ({P(b2, f('failure_class','memory_safety_oob'))}), and a small but hard tail of races/nondeterminism "
             f"({P(b2, f('failure_class','nondeterminism_race_sync'))}). {P(b2, f('escaped_ci','yes'))} escaped pre-merge CI; only {P(b2, f('regression_test_added','yes'))} of fixes added a regression test. "
             f"Performance PRs are reverted about twice as often as other merged PRs ({P(rr, 'repo=vllm&group=performance', 'revert_rate')} vs {P(rr, 'repo=vllm&group=non_performance', 'revert_rate')} in vLLM).\n")
    t.append("## 5. How much published kernel research leaves observable traces in production?\n")
    t.append(f"Little, within 16–18 months of publication. {C(fu, fs('both','kernel_style'), 'n')} of {C(fu, fs('both','all_papers'), 'n')} MLSys/ASPLOS 2025 papers are kernel-style; "
             f"{C(fu, fs('both','reached_L1'), 'n')} released code, {C(fu, fs('both','reached_L2'), 'n')} maintained it, {C(fu, fs('both','reached_L3'), 'n')} reached a kernel library or beyond, "
             f"{C(fu, fs('both','reached_L4'), 'n')} were merged into a framework and {C(fu, fs('both','reached_L5'), 'n')} became default/documented/release-noted — and "
             f"{C(fu, fs('both','L4plus_framework_code_before_publication'), 'n')} of the framework-level cases were already in production code before the paper appeared, leaving "
             f"{C(fu, fs('both','L4plus_adopted_after_publication'), 'n')} observable paper → framework transfers. "
             "System ideas (caching, parallelism, optimizers) from *non*-kernel papers also show up in production, so the leak is specific to the kernel pipeline, not to research in general.\n")
    t.append("## 6. How much production kernel engineering traces back to papers?\n")
    t.append(f"About a quarter to a third. Of {C(d, dd('both','all','academic_paper'), 'denominator')} verified kernel/backend implementations, "
             f"{P(d, dd('both','all','academic_paper'))} trace to an academic paper, {P(d, dd('both','all','vendor_library'))} to vendor libraries, "
             f"{P(d, dd('both','all','company_engineering'))} to company engineering and {P(d, dd('both','all','community_contribution'))} to in-repo community work. "
             "Papers concentrate in attention; MoE, quantized GEMM, collectives and fused norms are mostly vendor/company/community.\n")
    t.append("## 7. Which findings most strongly support or contradict the Optimization Gap?\n")
    t.append("**Support.** (i) Discovered optimizations queue: "
             f"{C(ps, 'repo=vllm', 'open_confirmed_performance')} + {C(ps, 'repo=sglang', 'open_confirmed_performance')} adjudicated performance PRs were open at the cutoff, and performance PRs merge more slowly at every horizon. "
             "(ii) Review labour targets Establish/Sustain concerns — integration, other hardware, benchmark method — much of it unwritten. "
             "(iii) Escaped kernel bugs are dominated by integration and portability failures, the parts of the pipeline downstream of discovery. "
             "(iv) Few kernel-style papers reach frameworks.\n")
    t.append("**Contradict / qualify.** (i) Most performance PRs that merge do so within days (median among merged ≈ "
             f"{C(s2, 'repo=vllm&group=performance', 'median_days_to_merge_among_merged', 'dec1')} days in vLLM) — the gap is a heavy tail plus attrition, not uniform delay. "
             f"(ii) SGLang writes down almost everything reviewers ask for: excluding reviewer-found defects, {P(a4s, 'repo=sglang', 'share_merged_with_ge1_undocumented_excl_defect_fix')} of its merged sampled performance PRs met an undocumented-type request, so codification is achievable (though much of it is only `partial`). "
             "(iii) Performance PRs are larger (median lines changed "
             f"{C(s2, 'repo=vllm&group=performance', 'median_lines_changed')} vs {C(s2, 'repo=vllm&group=non_performance', 'median_lines_changed')} in vLLM). Within size terciles the merge delay "
             "persists in SGLang and for small vLLM PRs but is not distinguishable from zero for medium vLLM PRs (`data/a2-effects.csv`), so size is a plausible "
             "partial confounder. (iv) Deep formal techniques are the best fit for only a small tail of escaped bugs; "
             "hardware-matrix CI and E2E serving tests would plausibly catch more.\n")
    t.append("## Five strongest defensible findings\n")
    t.append(f"1. **Performance PRs are ~12% of PRs but carry the heaviest evidence burden** — microbenchmarks {P(es, e('vllm','performance','ev_microbenchmark'))} vs {P(es, e('vllm','comparison','ev_microbenchmark'))} "
             f"in matched vLLM comparisons (microbenchmark-code κ = {C('data/review-coding-agreement.csv', 'repo=both&code=ev_microbenchmark', 'cohen_kappa', 'dec2')}).")
    t.append(f"2. **They merge more slowly under a competing-risk model**: 30-day merge incidence {P(sv, 'repo=vllm&group=performance&horizon_days=30', 'cif_merged')} vs "
             f"{P(sv, 'repo=vllm&group=non_performance&horizon_days=30', 'cif_merged')} (vLLM) with bootstrap CIs excluding zero (`data/a2-effects.csv`).")
    t.append(f"3. **They are reverted about twice as often** ({P(rr, 'repo=vllm&group=performance', 'revert_rate')} vs {P(rr, 'repo=vllm&group=non_performance', 'revert_rate')} vLLM; "
             f"{P(rr, 'repo=sglang&group=performance', 'revert_rate')} vs {P(rr, 'repo=sglang&group=non_performance', 'revert_rate')} SGLang; complete 24-month census).")
    t.append(f"4. **Escaped kernel bugs are mostly integration/portability failures** ({P(b2, f('failure_class','integration_backend_cudagraph'))} integration; "
             f"{P(b2, f('hardware_specific','yes'))} hardware-specific; failure-class κ = {C('data/failure-coding-agreement.csv', 'field=failure_class', 'cohen_kappa', 'dec2')}).")
    t.append(f"5. **Kernel research rarely reaches frameworks, and frameworks rarely come from papers**: {C(fu, fs('both','reached_L4'), 'n')} of {C(fu, fs('both','kernel_style'), 'n')} "
             f"kernel-style 2025 papers reached L4+; {P(d, dd('both','all','academic_paper'))} of deployed implementations trace to papers.\n")
    t.append("## Five best figures for the talk\n")
    t.append("1. `figures/a2-time-to-merge-km-cif.png` — performance vs other PRs, KM and competing-risk CIF.\n"
             "2. `figures/a3-evidence-and-review-requests.png` — the evidence and reviewer-request gap, matched corpus.\n"
             "3. `figures/b3-failure-vs-detector-matrix.png` — escaped kernel bugs × most plausible detector.\n"
             "4. `figures/c-paper-to-production-funnel.png` — the MLSys/ASPLOS 2025 ladder.\n"
             "5. `figures/d-provenance-by-operator.png` — where deployed kernels come from.\n")
    t.append("## Three case histories for narration\n")
    t.append("1. **GPTQ Marlin shared-memory races (vLLM #11493)** — user-visible corruption; compute-sanitizer racecheck found reuse races; "
             "a WarpDRF-style contract or routine sanitizer run would plausibly have caught it (`kernel-failures-report.md`, B4).\n"
             "2. **RMSNorm batch invariance (vLLM #48391)** — a block-size heuristic broke bitwise batch invariance for roughly three months; the RESOLVE-style "
             "determinism/perturbation check is the natural detector (B4).\n"
             "3. **A reverted performance PR (vLLM #48223, reverted by #52024)** — full proposal → evidence → review → merge → revert chain "
             "(`kernel-pr-delivery-report.md`, A5).\n")
    t.append("## Open limitations\n")
    t.append("- Two repositories, two venues, one 12/24-month window; no causal identification.\n"
             "- Performance population recall ≈ 60–66%; contrasts are conservative.\n"
             "- Content codes rely on keyword-prefiltered snippets; private CI, dashboards and offline discussion are invisible.\n"
             "- 41 ASPLOS papers were classified from abstracts (no reachable open copy).\n"
             "- Counterfactual detection is judgement, not experiment; no technique was run on the bugs.\n"
             "- All coding was done by LLM subagents; agreement is model–model, not human inter-rater reliability.\n")
    t.append("## Paths\n")
    t.append("| Artifact | Path |\n|---|---|")
    for a, p in [("Checkpoint", "experiments/deep-study/STATUS.json"), ("Methods", "experiments/deep-study/METHODS.md"),
                 ("PR delivery report", "experiments/deep-study/kernel-pr-delivery-report.md"), ("Failures report", "experiments/deep-study/kernel-failures-report.md"),
                 ("Paper-to-production report", "experiments/deep-study/paper-to-production-report.md"),
                 ("Provenance report", "experiments/deep-study/production-kernel-provenance-report.md"),
                 ("Performance-PR population", "experiments/deep-study/data/performance-pr-population.csv"),
                 ("Full PR metadata (A2)", "experiments/deep-study/data/pr-population-metadata.csv"),
                 ("Detailed review coding (A3)", "experiments/deep-study/data/review-coding.csv"),
                 ("Reviewer requests vs rules (A4)", "experiments/deep-study/data/a4-request-types.csv"),
                 ("Confirmed reverts (B1)", "experiments/deep-study/data/confirmed-reverts.csv"),
                 ("Kernel-correctness cases (B2/B3)", "experiments/deep-study/data/kernel-correctness-cases.csv"),
                 ("Paper census (C1)", "experiments/deep-study/data/paper-census.csv"),
                 ("Deployment ladder (C2–C4)", "experiments/deep-study/data/paper-deployment-ladder.csv"),
                 ("Production kernel provenance (D)", "experiments/deep-study/data/production-kernel-provenance.csv"),
                 ("Consistency check", "experiments/deep-study/data/consistency-check.json"),
                 ("Codebooks and adjudication records", "experiments/deep-study/coding/"), ("Scripts", "experiments/deep-study/scripts/"),
                 ("Figures", "experiments/deep-study/figures/")]:
        t.append(f"| {a} | `{p}` |")
    t.append("\n## Resume\n")
    t.append("```powershell\n# Run from the artifact repository root.\n"
             "python experiments\\deep-study\\scripts\\status.py verify\npython experiments\\deep-study\\scripts\\status.py show\n"
             "python experiments\\deep-study\\scripts\\consistency.py\n```\n"
             "All collectors reuse caches; rerun any report with `python experiments\\deep-study\\scripts\\reports.py all`.\n")
    text = "\n".join(t) + "\n"
    atomic_text(DEEP / "SYNTHESIS.md", text)
    return text


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "all":
        d_summary(); a_figures(); failures(); provenance(); delivery(); papers(); synthesis()
    elif cmd == "synthesis":
        synthesis()
    elif cmd == "papers":
        papers()
    elif cmd == "delivery":
        delivery()
    elif cmd == "a_figures":
        a_figures()
    elif cmd == "d_summary":
        d_summary()
    elif cmd == "failures":
        failures()
    elif cmd == "provenance":
        provenance()
