# Study 6: Portability Load

**Question:** What does each new GPU generation, vendor, or backend cost a framework, in kernel code, PRs, and time?

**Talk hook:** *Open problem: Portability.* "Can the intent of this optimization survive somewhere else?" Today the answer is: re-implement it, per architecture. It also connects to SIMT-Step and WarpDRF, since semantics differ across vendors.

## Hypotheses
- H1: A large and growing share of kernel code is architecture-specific (`sm90`/`sm100`, `gfx942`/`gfx950`, Metal, Vulkan, and so on).
- H2: Support for a new GPU (for example Blackwell B200, AMD MI355) arrives as a long tail of PRs spread over months, not a single port.
- H3: Optimizations reach NVIDIA first. The lag for other vendors is measurable.

## Method
1. **Architecture-specific code census:**
   - Count files, kernels, and lines guarded by `__CUDA_ARCH__`, `sm_XX` dispatch, ROCm `gfx` checks, or backend-specific directories (`ggml-cuda`, `ggml-metal`, `ggml-vulkan`, `ggml-sycl`, ...). Track this over time.
   - Count attention and GEMM *backend variants* per engine.
2. **New-hardware cohorts:** for each launch (H100, B200, MI300X, MI355), collect the PRs mentioning it (titles, tags like `[Hardware][AMD]`, and paths), then plot their cumulative count and lines changed over time since launch.
3. **Cross-vendor lag:** for a set of named optimizations (for example FP8 GEMM, MLA attention, FP4 MoE), find the date each first worked on NVIDIA vs. AMD vs. other backends.
4. **llama.cpp as the extreme case:** count how many backends implement each ggml op, and the lag between the first and the Nth backend.

## Candidate slides
- "Each new GPU generation triggered N PRs over M months in vLLM."
- "The same optimization reached AMD a median of D days after NVIDIA."
- "llama.cpp: K backends × J ops; X% of op implementations exist for only one backend."

## Effort
★★. Path and tag based, and mostly scriptable.
