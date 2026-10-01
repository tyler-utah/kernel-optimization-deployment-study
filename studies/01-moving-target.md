# Study 1: The Moving Target

**Question:** How fast does the software around an optimized kernel change, and how long does optimized kernel code survive?

**Talk hook:** *Sustain / Maintainability.* "Optimized code decays. Models change, shapes change, hardware changes, compilers change, libraries change." This study puts numbers on that sentence.

## Hypotheses
- H1: Serving engines (vLLM, SGLang) change so fast that a kernel tuned today is integrated into a substantially different system within weeks.
- H2: Kernel code has a *shorter* half-life than the rest of the codebase, because it is rewritten for new hardware, dtypes, and shapes.
- H3: The interfaces kernels plug into (attention-backend APIs, MoE dispatch, quantization-method registries) churn repeatedly, so an optimization must be re-integrated even when the kernel itself hasn't changed.

## Data
- Full clones of vLLM, SGLang, llama.cpp, FlashInfer, and PyTorch (`aten/src/ATen/native/cuda`, `torch/_inductor`).
- Release tags and dependency pins over time: the torch, CUDA, FlashInfer, and Triton versions required by vLLM and SGLang.

## Method
1. **Churn rate:** commits per day, lines changed per week, and releases per month. Split kernel paths from everything else (path rules in [../README.md](../README.md)).
2. **Survival analysis (`git blame` over time):** for kernel lines added in each quarter, the fraction still present after 3, 6, and 12 months, using Kaplan–Meier curves. Compare kernel vs. non-kernel code.
3. **Interface churn:** track the files that define kernel-facing interfaces (for example vLLM's `vllm/attention/backends/*`, `vllm/model_executor/layers/fused_moe/*`, and SGLang's equivalents). Count breaking signature changes and the number of backends over time.
4. **Dependency treadmill:** how often each engine bumps its torch, CUDA, FlashInfer, or Triton pin. Each bump is a re-validation event for every custom kernel.

## Candidate slides
- "vLLM and SGLang each merge about 45 commits a day." (The sizing probe already supports this.)
- "The median line of kernel code in X survives N months."
- "vLLM has had K attention backends; the backend interface changed M times in 12 months."
- "Every custom kernel in X had to survive P PyTorch and CUDA upgrades last year."

## Pitfalls
- Moved or renamed files break naive blame. Use `git log --follow` and `-M -C` detection.
- Vendored code (for example vLLM's flash-attention fork) inflates churn. Tag vendored directories separately.
- Reformatting commits (linters) inflate churn. Exclude them via `.git-blame-ignore-revs` or by detecting whitespace-only diffs.

## Effort
★★. Mostly git scripting. Survival analysis over the full history of a big repo takes a few hours of compute.
