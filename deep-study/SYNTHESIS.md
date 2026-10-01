# Synthesis — optimizing the optimization pipeline

This synthesis answers the seven study questions using only results that passed the study's gates. Each number is linked to a CSV row by a hidden claim tag and re-verified by `scripts/consistency.py`. Complete-population results (A1–A2, B1, C1, D) are distinguished from sampled/coded results (A3–A5, B2–B4, C2–C4). All coding agreement is model–model adjudication consistency, not human inter-rater reliability; nothing is causal.

## 1. Is kernel discovery a large or small fraction of real engineering work?

Small but not marginal. Confirmed performance PRs are 3,214<!-- claim:data/a1-population-summary.csv::repo=vllm::confirmed_performance::int --> of 25,573<!-- claim:data/a1-population-summary.csv::repo=vllm::window_prs::int --> vLLM PRs and 3,094<!-- claim:data/a1-population-summary.csv::repo=sglang::confirmed_performance::int --> of 26,004<!-- claim:data/a1-population-summary.csv::repo=sglang::window_prs::int --> SGLang PRs in 12 months — roughly one PR in eight, a high-precision lower bound (estimated recall 66.3<!-- claim:data/a1-recall-estimate.csv::repo=vllm::est_recall::pct1 -->% and 60.5<!-- claim:data/a1-recall-estimate.csv::repo=sglang::est_recall::pct1 -->%). And only part of that is *discovering* a new or faster kernel: among the 5,417<!-- claim:data/a1-perf-type-summary.csv::repo=both&perf_type=new_or_faster_kernel_total::denominator::int --> adjudicated performance PRs, new kernels/fusions and kernel speedups are 38.8<!-- claim:data/a1-perf-type-summary.csv::repo=both&perf_type=new_or_faster_kernel_total::share::pct1 -->%, while system-level optimizations are 41.6<!-- claim:data/a1-perf-type-summary.csv::repo=both&perf_type=system_performance::share::pct1 -->%, precision/format work 11.4<!-- claim:data/a1-perf-type-summary.csv::repo=both&perf_type=precision_format::share::pct1 -->%, tuning configs 5.6<!-- claim:data/a1-perf-type-summary.csv::repo=both&perf_type=kernel_tuning_config::share::pct1 -->% and performance-regression fixes 2.6<!-- claim:data/a1-perf-type-summary.csv::repo=both&perf_type=perf_regression_fix::share::pct1 -->%.

## 2. What consumes the rest of the work?

Establishing and sustaining optimizations. Kernel-correctness fixes alone are estimated at 1,787<!-- claim:data/b2-failure-summary.csv::repo=both&measure=estimated_confirmed_in_frame&category=confirmed::n::int --> in 24 months (from a 440<!-- claim:data/b2-failure-summary.csv::repo=both&measure=screened&category=all::n::int -->-commit random sample of a 2,354<!-- claim:data/b2-failure-summary.csv::repo=both&measure=screened&category=all::denominator::int -->-commit frame); 41.0<!-- claim:data/b2-failure-summary.csv::repo=both&measure=failure_class&category=integration_backend_cudagraph::share::pct1 -->% of confirmed failures are integration/backend-selection/CUDA-graph problems and 55.1<!-- claim:data/b2-failure-summary.csv::repo=both&measure=hardware_specific&category=yes::share::pct1 -->% are hardware-specific. 467<!-- claim:data/b1-revert-summary.csv::repo=both&measure=confirmed_reverts_or_rollbacks&category=all::n::int --> reverts/rollbacks occurred in 24 months. The first study's commit-level view (bug fixes, hardware enablement, integration, maintenance) is consistent with this, but its fine-purpose shares remain diagnostic.

## 3. What evidence and reviewer labour turn an optimization into a merge?

Performance PRs arrive with far more evidence than matched comparisons (vLLM microbenchmarks 40.0<!-- claim:data/a3-evidence-summary.csv::repo=vllm&group=performance&measure=ev_microbenchmark::share::pct1 -->% vs 3.8<!-- claim:data/a3-evidence-summary.csv::repo=vllm&group=comparison&measure=ev_microbenchmark::share::pct1 -->%; end-to-end 49.7<!-- claim:data/a3-evidence-summary.csv::repo=vllm&group=performance&measure=ev_end_to_end::share::pct1 -->% vs 13.2<!-- claim:data/a3-evidence-summary.csv::repo=vllm&group=comparison&measure=ev_end_to_end::share::pct1 -->%), yet merge more slowly: by day 30, 37.5<!-- claim:data/a2-survival.csv::repo=vllm&group=performance&horizon_days=30::cif_merged::pct1 -->% vs 46.0<!-- claim:data/a2-survival.csv::repo=vllm&group=non_performance&horizon_days=30::cif_merged::pct1 -->% had merged in vLLM and 42.9<!-- claim:data/a2-survival.csv::repo=sglang&group=performance&horizon_days=30::cif_merged::pct1 -->% vs 53.0<!-- claim:data/a2-survival.csv::repo=sglang&group=non_performance&horizon_days=30::cif_merged::pct1 -->% in SGLang. Reviewers ask about integration (27.1<!-- claim:data/a3-evidence-summary.csv::repo=vllm&group=performance&measure=cn_integration::share::pct1 -->% / 23.3<!-- claim:data/a3-evidence-summary.csv::repo=sglang&group=performance&measure=cn_integration::share::pct1 -->%), other hardware (19.5<!-- claim:data/a3-evidence-summary.csv::repo=vllm&group=performance&measure=rq_other_backend_hw::share::pct1 -->% / 14.7<!-- claim:data/a3-evidence-summary.csv::repo=sglang&group=performance&measure=rq_other_backend_hw::share::pct1 -->%) and benchmark method (20.2<!-- claim:data/a3-evidence-summary.csv::repo=vllm&group=performance&measure=cn_benchmark_method::share::pct1 -->% / 11.5<!-- claim:data/a3-evidence-summary.csv::repo=sglang&group=performance&measure=cn_benchmark_method::share::pct1 -->%). In vLLM, 44.6<!-- claim:data/a4-summary.csv::repo=vllm::share_merged_with_ge1_undocumented::pct1 -->% of merged sampled performance PRs met at least one pre-merge request of a type absent from the written rules.

## 4. What kinds of kernel bugs escape existing validation?

Mostly bugs that unit tests on one GPU cannot see: integration failures (41.0<!-- claim:data/b2-failure-summary.csv::repo=both&measure=failure_class&category=integration_backend_cudagraph::share::pct1 -->%), shape/alignment edge cases (18.6<!-- claim:data/b2-failure-summary.csv::repo=both&measure=failure_class&category=shape_alignment_edge::share::pct1 -->%), hardware/compiler-specific failures (16.2<!-- claim:data/b2-failure-summary.csv::repo=both&measure=failure_class&category=hardware_compiler_specific::share::pct1 -->%), numerical precision (9.3<!-- claim:data/b2-failure-summary.csv::repo=both&measure=failure_class&category=numerical_precision::share::pct1 -->%), memory safety (8.4<!-- claim:data/b2-failure-summary.csv::repo=both&measure=failure_class&category=memory_safety_oob::share::pct1 -->%), and a small but hard tail of races/nondeterminism (3.0<!-- claim:data/b2-failure-summary.csv::repo=both&measure=failure_class&category=nondeterminism_race_sync::share::pct1 -->%). 80.2<!-- claim:data/b2-failure-summary.csv::repo=both&measure=escaped_ci&category=yes::share::pct1 -->% escaped pre-merge CI; only 28.4<!-- claim:data/b2-failure-summary.csv::repo=both&measure=regression_test_added&category=yes::share::pct1 -->% of fixes added a regression test. Performance PRs are reverted about twice as often as other merged PRs (1.8<!-- claim:data/a2-revert-rates.csv::repo=vllm&group=performance::revert_rate::pct1 -->% vs 0.9<!-- claim:data/a2-revert-rates.csv::repo=vllm&group=non_performance::revert_rate::pct1 -->% in vLLM).

## 5. How much published kernel research leaves observable traces in production?

Little, within 16–18 months of publication. 39<!-- claim:data/paper-funnel.csv::venue=both&stage=kernel_style::n::int --> of 221<!-- claim:data/paper-funnel.csv::venue=both&stage=all_papers::n::int --> MLSys/ASPLOS 2025 papers are kernel-style; 21<!-- claim:data/paper-funnel.csv::venue=both&stage=reached_L1::n::int --> released code, 11<!-- claim:data/paper-funnel.csv::venue=both&stage=reached_L2::n::int --> maintained it, 6<!-- claim:data/paper-funnel.csv::venue=both&stage=reached_L3::n::int --> reached a kernel library or beyond, 4<!-- claim:data/paper-funnel.csv::venue=both&stage=reached_L4::n::int --> were merged into a framework and 4<!-- claim:data/paper-funnel.csv::venue=both&stage=reached_L5::n::int --> became default/documented/release-noted — and 2<!-- claim:data/paper-funnel.csv::venue=both&stage=L4plus_framework_code_before_publication::n::int --> of the framework-level cases were already in production code before the paper appeared, leaving 2<!-- claim:data/paper-funnel.csv::venue=both&stage=L4plus_adopted_after_publication::n::int --> observable paper → framework transfers. System ideas (caching, parallelism, optimizers) from *non*-kernel papers also show up in production, so the leak is specific to the kernel pipeline, not to research in general.

## 6. How much production kernel engineering traces back to papers?

About a quarter to a third. Of 204<!-- claim:data/d-provenance-summary.csv::repo=both&scope=all&provenance_class=academic_paper::denominator::int --> verified kernel/backend implementations, 28.9<!-- claim:data/d-provenance-summary.csv::repo=both&scope=all&provenance_class=academic_paper::share::pct1 -->% trace to an academic paper, 29.4<!-- claim:data/d-provenance-summary.csv::repo=both&scope=all&provenance_class=vendor_library::share::pct1 -->% to vendor libraries, 18.6<!-- claim:data/d-provenance-summary.csv::repo=both&scope=all&provenance_class=company_engineering::share::pct1 -->% to company engineering and 23.0<!-- claim:data/d-provenance-summary.csv::repo=both&scope=all&provenance_class=community_contribution::share::pct1 -->% to in-repo community work. Papers concentrate in attention; MoE, quantized GEMM, collectives and fused norms are mostly vendor/company/community.

## 7. Which findings most strongly support or contradict the Optimization Gap?

**Support.** (i) Discovered optimizations queue: 1,022<!-- claim:data/a1-population-summary.csv::repo=vllm::open_confirmed_performance::int --> + 784<!-- claim:data/a1-population-summary.csv::repo=sglang::open_confirmed_performance::int --> adjudicated performance PRs were open at the cutoff, and performance PRs merge more slowly at every horizon. (ii) Review labour targets Establish/Sustain concerns — integration, other hardware, benchmark method — much of it unwritten. (iii) Escaped kernel bugs are dominated by integration and portability failures, the parts of the pipeline downstream of discovery. (iv) Few kernel-style papers reach frameworks.

**Contradict / qualify.** (i) Most performance PRs that merge do so within days (median among merged ≈ 5.2<!-- claim:data/a2-population-summary.csv::repo=vllm&group=performance::median_days_to_merge_among_merged::dec1 --> days in vLLM) — the gap is a heavy tail plus attrition, not uniform delay. (ii) SGLang writes down almost everything reviewers ask for: excluding reviewer-found defects, 0.0<!-- claim:data/a4-summary.csv::repo=sglang::share_merged_with_ge1_undocumented_excl_defect_fix::pct1 -->% of its merged sampled performance PRs met an undocumented-type request, so codification is achievable (though much of it is only `partial`). (iii) Performance PRs are larger (median lines changed 223<!-- claim:data/a2-population-summary.csv::repo=vllm&group=performance::median_lines_changed::int --> vs 62<!-- claim:data/a2-population-summary.csv::repo=vllm&group=non_performance::median_lines_changed::int --> in vLLM). Within size terciles the merge delay persists in SGLang and for small vLLM PRs but is not distinguishable from zero for medium vLLM PRs (`data/a2-effects.csv`), so size is a plausible partial confounder. (iv) Deep formal techniques are the best fit for only a small tail of escaped bugs; hardware-matrix CI and E2E serving tests would plausibly catch more.

## Five strongest defensible findings

1. **Performance PRs are ~12% of PRs but carry the heaviest evidence burden** — microbenchmarks 40.0<!-- claim:data/a3-evidence-summary.csv::repo=vllm&group=performance&measure=ev_microbenchmark::share::pct1 -->% vs 3.8<!-- claim:data/a3-evidence-summary.csv::repo=vllm&group=comparison&measure=ev_microbenchmark::share::pct1 -->% in matched vLLM comparisons (microbenchmark-code κ = 0.89<!-- claim:data/review-coding-agreement.csv::repo=both&code=ev_microbenchmark::cohen_kappa::dec2 -->).
2. **They merge more slowly under a competing-risk model**: 30-day merge incidence 37.5<!-- claim:data/a2-survival.csv::repo=vllm&group=performance&horizon_days=30::cif_merged::pct1 -->% vs 46.0<!-- claim:data/a2-survival.csv::repo=vllm&group=non_performance&horizon_days=30::cif_merged::pct1 -->% (vLLM) with bootstrap CIs excluding zero (`data/a2-effects.csv`).
3. **They are reverted about twice as often** (1.8<!-- claim:data/a2-revert-rates.csv::repo=vllm&group=performance::revert_rate::pct1 -->% vs 0.9<!-- claim:data/a2-revert-rates.csv::repo=vllm&group=non_performance::revert_rate::pct1 -->% vLLM; 2.4<!-- claim:data/a2-revert-rates.csv::repo=sglang&group=performance::revert_rate::pct1 -->% vs 1.1<!-- claim:data/a2-revert-rates.csv::repo=sglang&group=non_performance::revert_rate::pct1 -->% SGLang; complete 24-month census).
4. **Escaped kernel bugs are mostly integration/portability failures** (41.0<!-- claim:data/b2-failure-summary.csv::repo=both&measure=failure_class&category=integration_backend_cudagraph::share::pct1 -->% integration; 55.1<!-- claim:data/b2-failure-summary.csv::repo=both&measure=hardware_specific&category=yes::share::pct1 -->% hardware-specific; failure-class κ = 0.83<!-- claim:data/failure-coding-agreement.csv::field=failure_class::cohen_kappa::dec2 -->).
5. **Kernel research rarely reaches frameworks, and frameworks rarely come from papers**: 4<!-- claim:data/paper-funnel.csv::venue=both&stage=reached_L4::n::int --> of 39<!-- claim:data/paper-funnel.csv::venue=both&stage=kernel_style::n::int --> kernel-style 2025 papers reached L4+; 28.9<!-- claim:data/d-provenance-summary.csv::repo=both&scope=all&provenance_class=academic_paper::share::pct1 -->% of deployed implementations trace to papers.

## Five best figures for the talk

1. `figures/a2-time-to-merge-km-cif.png` — performance vs other PRs, KM and competing-risk CIF.
2. `figures/a3-evidence-and-review-requests.png` — the evidence and reviewer-request gap, matched corpus.
3. `figures/b3-failure-vs-detector-matrix.png` — escaped kernel bugs × most plausible detector.
4. `figures/c-paper-to-production-funnel.png` — the MLSys/ASPLOS 2025 ladder.
5. `figures/d-provenance-by-operator.png` — where deployed kernels come from.

## Three case histories for narration

1. **GPTQ Marlin shared-memory races (vLLM #11493)** — user-visible corruption; compute-sanitizer racecheck found reuse races; a WarpDRF-style contract or routine sanitizer run would plausibly have caught it (`kernel-failures-report.md`, B4).
2. **RMSNorm batch invariance (vLLM #48391)** — a block-size heuristic broke bitwise batch invariance for roughly three months; the RESOLVE-style determinism/perturbation check is the natural detector (B4).
3. **A reverted performance PR (vLLM #48223, reverted by #52024)** — full proposal → evidence → review → merge → revert chain (`kernel-pr-delivery-report.md`, A5).

## Open limitations

- Two repositories, two venues, one 12/24-month window; no causal identification.
- Performance population recall ≈ 60–66%; contrasts are conservative.
- Content codes rely on keyword-prefiltered snippets; private CI, dashboards and offline discussion are invisible.
- 41 ASPLOS papers were classified from abstracts (no reachable open copy).
- Counterfactual detection is judgement, not experiment; no technique was run on the bugs.
- All coding was done by LLM subagents; agreement is model–model, not human inter-rater reliability.

## Paths

| Artifact | Path |
|---|---|
| Checkpoint | `experiments/deep-study/STATUS.json` |
| Methods | `experiments/deep-study/METHODS.md` |
| PR delivery report | `experiments/deep-study/kernel-pr-delivery-report.md` |
| Failures report | `experiments/deep-study/kernel-failures-report.md` |
| Paper-to-production report | `experiments/deep-study/paper-to-production-report.md` |
| Provenance report | `experiments/deep-study/production-kernel-provenance-report.md` |
| Performance-PR population | `experiments/deep-study/data/performance-pr-population.csv` |
| Full PR metadata (A2) | `experiments/deep-study/data/pr-population-metadata.csv` |
| Detailed review coding (A3) | `experiments/deep-study/data/review-coding.csv` |
| Reviewer requests vs rules (A4) | `experiments/deep-study/data/a4-request-types.csv` |
| Confirmed reverts (B1) | `experiments/deep-study/data/confirmed-reverts.csv` |
| Kernel-correctness cases (B2/B3) | `experiments/deep-study/data/kernel-correctness-cases.csv` |
| Paper census (C1) | `experiments/deep-study/data/paper-census.csv` |
| Deployment ladder (C2–C4) | `experiments/deep-study/data/paper-deployment-ladder.csv` |
| Production kernel provenance (D) | `experiments/deep-study/data/production-kernel-provenance.csv` |
| Consistency check | `experiments/deep-study/data/consistency-check.json` |
| Codebooks and adjudication records | `experiments/deep-study/coding/` |
| Scripts | `experiments/deep-study/scripts/` |
| Figures | `experiments/deep-study/figures/` |

## Resume

```powershell
# Run from the artifact repository root.
python experiments\deep-study\scripts\status.py verify
python experiments\deep-study\scripts\status.py show
python experiments\deep-study\scripts\consistency.py
```
All collectors reuse caches; rerun any report with `python experiments\deep-study\scripts\reports.py all`.

