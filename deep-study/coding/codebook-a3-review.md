# Codebook A3 — evidence and reviewer-request coding for detailed PRs (v1.0)

Unit: one PR in the detailed corpus (confirmed performance PRs and a matched
non-performance comparison set; the coder is **not told** which). Input:
title, state, size, labels, kernel paths, the opening of the PR body, and
**keyword-prefiltered candidate snippets** grouped by code. Each snippet
carries its source: `body` (PR description), `author` (PR-author comment),
`review`/`comment`/`inline` (non-author human), with a short actor role
(`author`, `member`, `contributor`, `none`) and date. Prefilters are broad; a
snippet appearing under a code is **only a candidate**. Final positive labels
require that the snippet, read in context, actually satisfies the definition.
If a code has no snippets it is negative (`0`).

## Evidence codes (what the PR author/contributors provided)

Positive when the body or an author comment **reports** it (plans or requests
don't count; a reviewer-posted result counts for `ev_*` too, noted in the
evidence).

| Code | Definition |
|---|---|
| `ev_microbenchmark` | kernel/operator-level measurements (op latency, kernel time, TFLOPS, bandwidth, microbenchmark tables) |
| `ev_end_to_end` | serving or offline end-to-end measurements with a model (throughput tok/s, TTFT/TPOT/ITL, request latency, benchmark_serving outputs) |
| `ev_accuracy_eval` | model-quality evaluation results (GSM8K, MMLU, lm-eval, perplexity, accuracy scores) |
| `ev_numerical_tests` | numerical/correctness tests: kernel-vs-reference comparisons, allclose/tolerance, new/updated unit tests, "tests pass" naming specific correctness tests |
| `ev_multi_hardware` | results or tests on ≥2 hardware types/vendors/backends |
| `ev_memory` | memory-usage evidence (peak memory, KV-cache capacity, memory savings) |
| `ev_compile_graph` | evidence or discussion of CUDA-graph / torch.compile / piecewise-graph compatibility |

## Reviewer codes (non-author humans only; bots and CI commands excluded)

| Code | Definition |
|---|---|
| `rq_tests` | reviewer asks for tests (unit, correctness, regression, CI coverage) |
| `rq_other_backend_hw` | reviewer asks about/for another backend, hardware, vendor, or configuration (e.g. "does this work on AMD/Hopper/FA2?", "please test on B200") |
| `rq_accuracy_eval` | reviewer asks for accuracy/model-quality evaluation |
| `rq_perf_evidence` | reviewer asks for performance numbers or more benchmark data |
| `cn_integration` | integration concerns: interaction with other features/paths (CUDA graphs, torch.compile, spec decode, TP/EP/PP, other backends, default selection, config flags, API) |
| `cn_maintenance` | maintenance/complexity concerns: duplication, complexity, abstraction, code placement, ownership, extra dependency, dead code, readability |
| `cn_benchmark_method` | benchmark-methodology concerns: baseline, workload/shape choice, warmup/variance, config, missing e2e, fairness of comparison |
| `cn_documentation` | docs concerns: asks for docs, comments, docstrings, README/usage notes, release notes |

## Other fields

| Field | Values |
|---|---|
| `superseded` | for closed-unmerged PRs: `yes` if text states it was superseded/replaced/landed elsewhere, `no`, `unknown`; `na` for merged/open |
| `evidence` | object mapping each positive code to a ≤160-char verbatim quote from its snippet |
| `notes` | optional short note |

Output values for codes are integers `0`/`1`.
