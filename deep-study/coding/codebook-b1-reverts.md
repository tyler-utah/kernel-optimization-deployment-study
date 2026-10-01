# Codebook B1 — revert and rollback census (v1.0)

Unit: one first-parent commit on the default branch of vLLM or SGLang, committed
2024-09-29 → 2026-09-28, whose subject contains revert/rollback/back-out/undo/
reapply/reland language. Each input record contains the commit subject and
body, the revert PR body and up to 8 non-bot comments/reviews, and up to four
candidate *targets* (the reverted PR/commit) with their title, body excerpt,
labels and changed paths.

Read the whole record. Decide from **context**, not from the word "revert".

## Fields (one JSON object per input `id`)

| Field | Values |
|---|---|
| `id` | copy from input |
| `revert_status` | `confirmed_revert` (fully undoes one or more earlier merged changes), `partial_revert` (undoes part of an earlier change, e.g. one file or one default), `explicit_rollback` (not a git revert, but explicitly rolls a feature/default/dependency back to earlier behaviour), `reland` (re-applies previously reverted work), `not_revert` (the word is used in another sense, e.g. "fix revert logic", "revert to fp32 inside kernel" as a design description, test named revert) |
| `reverted_prs` | list of integer PR numbers that were reverted (empty if unknown or not a revert) |
| `reverted_title` | title of the (main) reverted change, or "" |
| `reverted_class` | for reverts/rollbacks only: `kernel_performance` (the reverted change was a performance optimization or new/modified kernel, fusion, kernel tuning, perf-motivated backend/default switch), `kernel_correctness` (the reverted change was itself a kernel correctness fix), `hardware_backend` (support/enablement for a hardware platform or backend, e.g. ROCm, NPU, XPU, TPU, CPU, Blackwell), `other` (CI, models, API, scheduler features, refactors, deps, docs …); `na` for `not_revert` |
| `reverted_touches_kernel` | `yes` / `no` / `unknown` — did the reverted change modify kernel implementation code (CUDA/Triton/CUTLASS/C++ kernels, sgl-kernel, csrc, fused_moe, attention backends, quantization kernels)? |
| `revert_reason` | `correctness_or_accuracy` (wrong outputs, accuracy drop, NaN, garbage), `crash_or_hang` (exceptions, illegal memory access, deadlock, OOM), `performance_regression`, `ci_or_test_failure` (broke CI/tests without a user-visible bug being described), `build_or_dependency`, `hardware_specific_breakage` (breaks one platform/GPU/vendor), `api_or_compat_break` (breaks users, configs, downstream projects), `premature_or_process` (merged without review/approval, needs more discussion, wrong branch), `unstated`, `other`; `na` for `not_revert` |
| `hardware_specific` | `yes` if the stated failure is limited to specific hardware/backend, else `no`/`unknown` |
| `reason_evidence` | ≤200-char verbatim quote from the record supporting the reason ("" if unstated) |
| `confidence` | `high` / `medium` / `low` |
| `rationale` | one sentence explaining the classification |

Rules:

1. A `Revert "X"` title plus matching target is sufficient for `confirmed_revert`
   even without body text; the reason is then `unstated` unless comments state it.
2. `reverted_class` describes the **reverted** change, not the revert commit.
   Prefer the target title/body/paths over the revert's own paths.
3. If a record reverts several PRs of different classes, classify by the
   primary one and mention the others in `rationale`.
4. A performance PR that is reverted because it produced wrong outputs is
   `reverted_class=kernel_performance`, `revert_reason=correctness_or_accuracy`.
5. Never invent PR numbers. If the target is unknown, leave `reverted_prs` empty
   and set `confidence` no higher than `medium`.
