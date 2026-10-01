# Commit and PR classification codebook

Version 4.0. Study cutoff: 2026-09-28.

Version 2.0 was refined after the first blinded validation iteration failed the
preregistered thresholds. The failed v1 samples, labels, metrics, and confusion
matrices are retained with `-v1` filenames. Refinements gave documentation,
CI/test/build-only, and explicit refactor signals precedence over incidental
hardware/model words; narrowed the new-workload keywords; and tightened
kernel-implementation paths. Version 2.0 also missed the threshold on a fresh
sample; its evidence is retained with `-v2` filenames. Version 3.0 expands
repository-specific kernel/dispatch paths and distinguishes explicit
model/API/core tags from incidental keywords. It is evaluated on a third fresh
sample rather than reporting fit to either refinement sample. That sample also
missed the threshold and is retained with `-v3` filenames. Version 4.0 narrows
hardware enablement versus hardware-specific tuning, adds explicit
model/workload integration signals, and is evaluated on a fourth fresh sample.

## Unit and dimensions

The commit unit is a first-parent commit. Both core projects normally squash a
merged pull request into one first-parent commit, so the commit is usually also
a merged-PR proxy; the PR lifecycle analysis uses actual PR records.

Classification is deliberately two-dimensional:

1. **Primary purpose** answers why the change was made. Exactly one category is
   assigned.
2. **Kernel relationship and claims** are independent booleans. A hardware
   port, model integration, fix, test, or cleanup can touch kernel code without
   becoming a performance-purpose change.

Evidence precedence is contributor-chosen title tags or conventional prefixes,
then changed paths and file types, then title/body keywords. Ambiguous records
are assigned a lower confidence and included in the validation oversample.

## Primary-purpose categories

| Value | Include | Exclude / boundary |
|---|---|---|
| `kernel_performance` | Explicit optimization, latency, throughput, memory, fusion, faster kernel, quantized-kernel, or benchmark-driven performance purpose | A kernel path alone; correctness fixes; ports whose stated purpose is hardware support |
| `kernel_correctness` | A fix for wrong results, crashes, races, hangs, numerical errors, or dispatch errors in kernel implementation | Kernel tests without an implementation fix; generic runtime fixes |
| `bugfix` | Correctness/reliability fix outside kernel implementation | Refactoring framed as cleanup; kernel implementation fixes |
| `hardware_backend` | Enablement or porting for GPU/accelerator/CPU vendors, architectures, or backends | Performance tuning that merely names a GPU but does not enable/port it |
| `model_support` | Add or integrate a named model, architecture, weight format, tokenizer, or model-specific path | General multimodal/diffusion infrastructure; kernel optimization used by many models |
| `core_runtime_serving` | Scheduling, caching, distributed execution, serving engine, memory management, compilation, speculative decoding, metrics, and core features | Public API/router-only work; explicit bug fixes |
| `api_frontend_router` | OpenAI/API protocol, CLI, frontend, gateway, router, parser, request/response surface | Internal engine changes |
| `test_ci_build_benchmark_deps` | Tests-only, CI, packaging, build, release, dependency, benchmark harness, or developer tooling | Source fixes that also add tests; those retain the source-change purpose |
| `documentation` | Documentation-only | Source changes with documentation |
| `maintenance_refactor` | Refactor, cleanup, removal, rename, formatting, logging, typing, or revert where no more specific purpose is evident | User-visible feature or explicit correctness fix |
| `new_workloads` | Diffusion/image/video generation, RL/training integration, embeddings/pooling as a new workload family, or broad multimodal generation infrastructure | One named multimodal model, which is `model_support` |
| `other` | Insufficient evidence or genuinely cross-cutting administrative change | Do not use merely because a change spans multiple paths |

When multiple purposes appear, choose the user-visible objective stated in the
title. Priority for conflicting explicit signals is correctness fix, hardware
enablement, new workload/model support, performance, then integration/supporting
work. A performance change that adds tests remains `kernel_performance`.

## Orthogonal fields

- `touches_kernel_implementation`: at least one changed production source file
  is a GPU/accelerator kernel, kernel DSL implementation, native extension, or
  kernel dispatch implementation.
- `touches_kernel_tests_only`: kernel-specific tests or benchmarks are changed,
  but no kernel implementation file is changed.
- `hardware_backend`: title or production path identifies vendor, architecture,
  accelerator, or backend work.
- `optimization_claim`: title explicitly claims performance, memory, fusion,
  throughput, latency, speedup, or optimization. A `[Kernel]` tag alone is not
  an optimization claim.
- `correctness_fix_claim`: title explicitly claims fix/correction/reliability,
  including crash, race, hang, OOM, NaN, wrong output, or regression.
- `kernel_touch_reason`: for implementation-touching records only, one of
  `optimization`, `correctness`, `hardware_port`, `model_integration`,
  `test_ci`, `maintenance`, `other`, derived primarily from the primary purpose.

Files under tests, benchmarks, docs, examples, `.github`, build metadata, or
developer tooling never count as kernel implementation solely because their
path contains `kernel`. Generated or vendored code counts only when the commit
actively changes that implementation; this is noted as a validity threat.

## Signal and confidence

- `tag`, confidence `high`: an unambiguous contributor tag/prefix determines
  purpose.
- `title`, confidence `high` or `medium`: explicit title wording determines
  purpose.
- `path`, confidence `medium`: production paths determine purpose.
- `fallback`, confidence `low`: no decisive signal; usually
  `core_runtime_serving` or `other`.

## Validation labeling protocol

Samples are stratified by predicted class, kernel touch, hardware/bug/ambiguous
signals, and quarter. Labelers see the commit subject and changed paths but not
the prediction. They assign `human_primary_purpose`,
`human_touches_kernel_implementation`, and `label_notes` using this document.
The sample retains `sample_weight` so validation can report both instrument
quality and population-adjusted estimates. No independent second coder was
available; no inter-rater agreement is claimed.
