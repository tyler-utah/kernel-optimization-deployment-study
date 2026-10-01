# Study 3: What Breaks

**Question:** When optimized kernels fail in real frameworks, *how* do they fail, and which validation techniques would have caught them?

**Talk hook:** *§2 RESOLVE and §3 WarpDRF.* This supplies real-world motivation for determinism testing, reduction and proof, and warp-level contracts. It shows that the failure classes RESOLVE and WarpDRF target are not hypothetical.

## Hypotheses
- H1: A substantial share of kernel correctness bugs are *numerical* (precision, accumulation order, overflow), and they slip past tolerance-based tests.
- H2: Concurrency bugs (races, missing barriers, warp-sync masks) are rarer but take longer to diagnose and fix.
- H3: Many bugs are *hardware-specific*: they appear only on one GPU generation or vendor, or under a new compiler.
- H4: A notable fraction of kernel bugs are introduced by perf PRs and later reverted or fixed.

## Data
- Issues and PRs from vLLM, SGLang, llama.cpp, FlashInfer, and PyTorch (CUDA kernels), last 12–24 months.
- Revert commits (`git log --grep "^Revert"`), and "Fixes #" or "regression from #" links.

## Method
1. **Scoped version (cheap, do first): reverts.**
   - Find all reverts.
   - Classify the reverted change (kernel or not, perf or not).
   - Measure time-to-revert and extract the stated reason.
2. **Full version:**
   - Collect issues and PRs that fix kernel correctness problems, using path rules on the fixing PR plus keywords (wrong output, NaN, accuracy drop, garbage tokens, race, hang, illegal memory access, mismatch).
   - Hand-code a sample of about 100–150 with a taxonomy:
     - numerical / precision
     - race / synchronization (incl. warp-level)
     - shape or edge case (alignment, odd sizes, empty batch)
     - memory safety (OOB, illegal access)
     - hardware / compiler specific
     - integration (CUDA graphs, backend selection, torch.compile)
3. **Warp-level spotlight (WarpDRF tie-in):** grep fixing diffs for `__shfl_*_sync`, `__syncwarp`, `__activemask`, `__ballot_sync`, mask changes, and subgroup ops in Vulkan/Metal backends.
4. **Counterfactual detection column:** for each coded bug, which technique would plausibly have caught it?
   - more random tests
   - hidden input distributions (KernelBench-Verified style)
   - determinism / perturbation testing (RESOLVE)
   - formal equivalence (RESOLVE / Kuiper)
   - a warp-DRF check (WarpDRF)
   - only E2E testing (SWE-Serve style)

## Candidate slides
- "Of N kernel correctness bugs in vLLM and SGLang, X% were numerical, Y% concurrency, Z% hardware-specific."
- "K% of reverted PRs were performance changes, reverted a median of D days after landing."
- "Real bug gallery": 3–4 short case studies (a race, a tolerance-hidden numeric bug, a warp mask bug) that lead into RESOLVE and WarpDRF.
- "What would have caught it" matrix: bugs × techniques. This is the justification for layering validation techniques.

## Pitfalls
- Issue reports are noisy (user error, driver issues). Only count bugs with a confirmed fix.
- The counterfactual column is a judgment call. Have two coders and report agreement, or present it qualitatively.

## Effort
★★★ for the full version, since it needs manual coding. ★ for the reverts-only scoped version.
