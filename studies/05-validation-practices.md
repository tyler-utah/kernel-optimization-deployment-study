# Study 5: Validation Practices in the Wild

**Question:** How do real frameworks validate their optimized kernels today?

**Talk hook:** *§2.* The related work showed benchmark checks are weak: the 374× identity kernel, 14.5% reward hacks, and the 3-in-200 rare bug. Are production frameworks' own kernel tests any stronger?

## Hypotheses
- H1: Kernel tests overwhelmingly use a few random inputs with fixed seeds and loose tolerances (`atol`/`rtol` around 1e-2 for low precision).
- H2: Few repos run race or memory sanitizers (`compute-sanitizer --tool racecheck/memcheck`) or determinism checks in CI.
- H3: CI covers a small subset of the hardware the code claims to support. Many backends are "supported" but not tested per commit.

## Method
1. **Static mining of test code** in vLLM, SGLang, FlashInfer, llama.cpp, and PyTorch kernel tests:
   - Tolerance values: extract `atol`/`rtol`/`assert_close` arguments and plot the distribution by dtype.
   - Input generation: `torch.randn`/`rand`, seeds, and whether edge cases (NaN, Inf, zeros, extreme scales, negative values) appear.
   - Presence of determinism or repeat-run tests, sanitizer invocations, and fuzzing.
2. **CI configuration audit:**
   - Parse GitHub Actions and Buildkite configs to find which GPU types run which kernel tests, and how often (per PR, nightly, release).
   - Compare against the hardware matrix each project claims to support.
3. **Accuracy gating:** do perf PRs require an lm-eval or GSM8K check? (This overlaps with Study 2's evidence extraction.)

## Candidate slides
- "The median tolerance in FP16/BF16 kernel tests across these repos is atol = X."
- "K of 5 frameworks run a race detector in CI. None run determinism tests." (If H2 holds.)
- "Claimed hardware: N platforms. Hardware tested per PR: M."

## Pitfalls
- Tolerances are sometimes computed dynamically. Record those as "dynamic", and don't drop them.
- Buildkite and other CI configs may live outside the repo. Note it when coverage can't be determined.

## Effort
★. Mostly static analysis of test files and CI configs. It is quick and a good first task for an agent.
