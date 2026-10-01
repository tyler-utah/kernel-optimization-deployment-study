# Codebook B2/B3 — confirmed kernel-correctness fixes and validation counterfactuals (v1.0)

Unit: one merged fixing commit/PR (first-parent, 2024-09-29 → 2026-09-28) in
vLLM or SGLang that touches production kernel paths and whose subject uses
fix/failure language. Input shows the PR body, linked issues, discussion
excerpts, changed files (+/-), kernel paths and test paths in the fix.

## Stage 1 — is it a confirmed kernel-correctness fix?

`case_status`:
- `confirmed_kernel_correctness` — the change fixes an actual **incorrect
  behaviour** (wrong/inaccurate output, NaN/Inf, crash/exception, illegal
  memory access, hang, nondeterminism, failure to run on supported
  hardware/config, or a severe performance/availability failure) whose
  root cause lies in an optimized kernel, its launch/dispatch/configuration
  code, or its integration into the serving path (backend selection, CUDA
  graphs, torch.compile, quantization kernel wiring). The fixing PR or a
  maintainer comment must establish the fix.
- `not_kernel_correctness` — a feature, optimization, refactor, test-only,
  typo, build/lint fix unrelated to kernel behaviour, or a bug whose root
  cause is outside kernel code (e.g. model config parsing, API, scheduler
  logic that merely touched a kernel path).
- `insufficient_information` — cannot tell even after reading everything.

For `not_kernel_correctness`/`insufficient_information`, set every remaining
categorical field to `na` (lists empty) and fill only `rationale`.

## Stage 2 — failure coding (confirmed cases only)

| Field | Values / meaning |
|---|---|
| `failure_class` | primary root-cause class: `numerical_precision` (precision, accumulation, overflow, tolerance, scale/quant math), `nondeterminism_race_sync` (data race, missing barrier/fence, stream sync, nondeterministic reduction), `warp_subgroup_mask` (warp/subgroup participation, shuffle/ballot masks, lane assumptions, wave64 vs warp32), `shape_alignment_edge` (odd/zero/large sizes, alignment, strides, padding, head-dim/block-size edge cases), `memory_safety_oob` (OOB/illegal access, uninitialised memory, buffer overrun, wrong pointer/offset), `hardware_compiler_specific` (only on one arch/vendor/driver/compiler/Triton version), `integration_backend_cudagraph` (backend selection, dispatch conditions, CUDA-graph capture/replay, torch.compile, config plumbing), `perf_regression_as_correctness` (slowdown/timeout/OOM presented as a failure), `other` |
| `secondary_classes` | list of other applicable classes (may be empty) |
| `symptom` | `wrong_output_or_accuracy`, `nan_inf`, `crash_or_exception`, `illegal_memory_access`, `hang_deadlock`, `nondeterministic_output`, `compile_or_build_failure`, `performance_or_availability`, `other` |
| `affected_hardware` | free text, e.g. `NVIDIA Blackwell (sm100)`, `AMD MI300X`, `all CUDA`, `unknown` |
| `affected_kernel` | free text: kernel/backend, e.g. `FlashInfer MLA decode`, `Triton fused_moe`, `custom all-reduce` |
| `hardware_specific` | `yes` / `no` / `unknown` |
| `introducing_ref` | `#NNNN` or commit SHA **only if the record states it** (e.g. "regression from #123", "introduced by", "broken since"); else `unknown` |
| `introducing_evidence` | ≤160-char quote supporting `introducing_ref`, or "" |
| `regression_test_added` | `yes` if the fix adds/changes a test that exercises this failure (see `test_paths_in_fix` and body), `no`, `unknown` |
| `detected_by` | `user_issue` (issue/user report), `ci_or_nightly` (post-merge CI, nightly, accuracy dashboards), `developer_or_reviewer` (found while developing/reviewing other work), `downstream_or_partner` (another project/vendor/team), `unknown` |
| `escaped_ci` | `yes` (the bug was in merged code and was found outside the introducing PR's pre-merge CI), `no` (evidence it never reached a merged state), `unknown` |
| `consequence` | `revert` (fix reverts earlier change), `disablement` (feature/kernel/backend disabled or gated off), `fallback` (routes to a slower/safer path), `fix_forward` (kernel/integration corrected), `unknown` |
| `evidence_snippet` | ≤220-char verbatim quote establishing the failure and the fix |
| `rationale` | one or two sentences |
| `confidence` | `high` / `medium` / `low` |

## Stage 3 — validation counterfactual (B3, confirmed cases only)

Which technique could **plausibly** have detected this failure *before merge*?
This is a counterfactual judgement; choose the cheapest technique that would
plausibly have exposed it given the evidence.

| Value | Choose when |
|---|---|
| `randomized_edge_tests` | a kernel-vs-reference unit test with more random shapes/dtypes/edge sizes (odd, zero, huge, unaligned) would expose it |
| `hidden_input_distributions` | exposure needs realistic or adversarial value/workload distributions (outliers, extreme scales, skewed MoE routing, long contexts, specific quant scales, special tokens) that standard `randn` tests miss |
| `sanitizer_memcheck` | a dynamic checker (compute-sanitizer memcheck/initcheck/racecheck/synccheck, ASan, bounds-checked build) on an ordinary test input would flag it |
| `determinism_schedule_perturbation` | repeated runs / varied batch composition, split sizes, launch configs or stream timing with bitwise or invariance checks would expose it (RESOLVE-style) |
| `formal_equivalence` | only a semantic equivalence argument over all inputs (index math, reduction order, masking logic vs a reference) would reliably catch it |
| `warp_race_static_analysis` | a static warp-level race / barrier / mask analysis (WarpDRF-style) would flag it regardless of inputs |
| `e2e_serving_tests_only` | only visible when the full serving stack runs (scheduler + CUDA graphs + backend selection + real model), SWE-Serve-style E2E |
| `hardware_matrix_ci` | existing tests would fail if run on the affected GPU/vendor/driver/compiler, which CI did not cover |
| `not_enough_information` | the record does not reveal enough about the root cause |

Fields: `counterfactual_primary` (one value above), `counterfactual_all`
(list of all plausible values, including the primary), `counterfactual_rationale`
(one or two sentences grounded in the evidence; never claim certainty).
