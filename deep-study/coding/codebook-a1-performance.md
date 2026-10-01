# Codebook A1 — performance-PR adjudication (v1.0)

Unit: one pull request in vllm-project/vllm or sgl-project/sglang. Input shows
title, state, labels, size, changed paths (kernel paths flagged), and the PR
body with template boilerplate removed.

## Construct

A **confirmed performance PR** is a PR whose *primary stated purpose* is to make
the system faster or more resource-efficient (latency, throughput, memory,
launch/CPU overhead, communication), including:

- new or improved kernels and fused operators whose motivation is efficiency;
- kernel tuning configurations (e.g. fused-MoE/GEMM tile configs for a GPU);
- precision/format work that exists for efficiency (FP8/FP4/INT4 kernels,
  KV-cache quantization), when presented as an efficiency feature;
- switching a backend/default to a faster implementation;
- system-level optimizations (scheduler, CPU overhead, CUDA-graph capture
  for speed, communication overlap, caching) with an explicit efficiency aim;
- fixing a **performance regression**.

It is **not** a performance PR when the primary purpose is: a bug/crash/
accuracy fix (even in a kernel), new model support, a feature/API, hardware
enablement without an efficiency claim, refactor/cleanup, tests/CI/benchmark
tooling only, docs, or dependency bumps. A PR that adds a benchmark *script*
without optimizing anything is not performance. Use `uncertain` only when the
record genuinely does not reveal the purpose.

## Fields

| Field | Values |
|---|---|
| `id` | copy from input |
| `classification` | `confirmed_performance`, `not_performance`, `uncertain` |
| `perf_type` | for confirmed: `kernel_optimization` (faster existing kernel), `new_kernel_or_fusion` (new kernel/fused op/backend kernel for speed), `kernel_tuning_config`, `precision_format` (low-precision kernel paths for efficiency), `system_performance` (non-kernel optimization), `perf_regression_fix`; otherwise `na` |
| `evidence` | ≤160-char verbatim quote from title/body/labels/paths supporting the decision; prefix with the source, e.g. `title: …`, `body: …`, `paths: …` |
| `rationale` | one sentence |
| `confidence` | `high` / `medium` / `low` |

Rules: decide from the author's stated purpose, then paths. A `[Perf]` tag
is strong but not decisive (e.g. `[Perf] Fix crash in X` is a bug fix). A
`[Kernel]` tag is not decisive (kernel bug fixes and kernel refactors are not
performance). A benchmark table in the body is strong evidence of performance
purpose when the change itself modifies production code.
