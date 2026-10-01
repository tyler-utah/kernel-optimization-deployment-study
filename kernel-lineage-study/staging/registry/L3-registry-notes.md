# L3 vLLM attention implementation family registry notes

## Family tree (dates and PRs)
- 2023-03 PR #3 custom CUDA PagedAttention V1 (`csrc/attention_kernels.cu`).
  - 2023-05 PR #53 refactor to `csrc/attention/attention_kernels.cu`.
  - 2023-10 PR #1348 branches V2 split-KV: partitions, `exp_sums`, `max_logits`, reduce kernel.
  - 2024-11 PR #10091 splits header + `paged_attention_v1.cu` + `paged_attention_v2.cu`.
  - 2026-05 PR #43717 moves to `csrc/libtorch_stable/attention/`.
  - 2026-07 PR #47361 deletes CUDA PagedAttention decode ops; cache write kernels remain.
- 2023-02 -> live KV-cache write contract: `reshape_and_cache` / `reshape_and_cache_flash` moves to libtorch_stable in PR #43717.
- 2024-03 PR #3005/#3462 backend refactor: xFormers, FlashAttention V0, paged wrapper, abstract interface, selector.
  - 2025-09 PR #25351 removes V0 attention backends.
  - 2026-01 PR #31916 moves `vllm/attention` to `vllm/v1/attention` and model-layer locations.
- FlashAttention delivery/selection branch:
  - 2024-03 PR #3005 imports upstream `flash_attn` (`flash-attn==2.5.6` thirdparty/setup.py; Docker install PR #3396).
  - 2024-05 PR #4686 switches delivery to `vllm-flash-attn==2.5.8.post1` pip wheel; wheel pins bump until PR #8245.
  - 2024-09 PR #8245 replaces the wheel with inline CMake FetchContent from `vllm-project/flash-attention` at tag `013f0c4f...`; PR #8699 adds `vllm/vllm_flash_attn/.gitkeep`.
  - 2025-02 PR #13747 externalizes the build into `cmake/external_projects/vllm_flash_attn.cmake`; 63 unique GIT_TAG changes through cutoff, final tag `9cd61de...`.
  - 2025-01 PR #12093 introduces FA3 `fa_version`: SM90 prefers FA3 if supported, otherwise FA2; env override then `VLLM_FLASH_ATTN_VERSION`.
  - 2026-03 PR #32974 adds FA4; cutoff `get_flash_attn_version` prefers FA3 on SM90, FA4 on SM100, otherwise FA2 with fallbacks.
- FlashInfer branch:
  - 2024-05 PR #4353 V0 FlashInfer decode backend; 2025-04 PR #16684 V1 FlashInfer backend.
  - 2025-07 PR #19825 adds FlashInfer TRTLLM-gen decode path for SM100; 2025-08 PR #22095 adds TRTLLM prefill and generalizes `VLLM_USE_TRTLLM_ATTENTION`.
  - 2026-07 PR #43232 adds XQA as a sibling decode path: SM90/SM12x XQA, SM100+ TRTLLM_GEN.
- Triton branch: 2024-03 prefix prefill; 2024-04 PR #3643 ROCm Triton FA; 2025-01 PR #12528 decode/MLA; 2025-03 PR #14152 chunked prefill; 2025-05 PR #16828 unified attention; 2026-09 Triton FlashInfer-compatible path.
- Merge branch: 2025-02 PR #12639 Triton LSE merge; 2025-04 PR #16173 CUDA LSE merge, moved to stable ABI in PR #43361.
- ROCm branch: 2024-09 PR #8310 custom paged attention; 2025-02 PR #12790 V1 ROCm; 2025-06 PR #18596 AITER FA; 2025-10 PR #25507 AITER unified.
- MLA branch: 2025-01 PR #12528 Triton V0; 2025-02 PRs #13747/#13867 FlashMLA V0/V1; 2025-04/#16032 and 2025-06/#17625 CUTLASS; 2025-09 FlashAttn/FlashInfer/FlashMLA sparse; 2025-05/#17523 and 2025-11/#26670/#28701 ROCm AITER; 2026-02 PR #33451 FlashInfer sparse.
- Dispatch spine: selector (#3462), registry (#25893 -> #31916), abstract/backend protocol (#3462/#31916), platform defaults (`cuda.py`, `rocm.py`, #6080).

## Ten important ancestry facts
1. PagedAttention V2 is a branch from V1: PR #1348 adds `PARTITION_SIZE`, `max_num_partitions`, `exp_sums`, `max_logits`, and a reduce kernel.
2. PR #10091 is a source split: git stat renames `attention_kernels.cu => attention_kernels.cuh` and adds V1/V2 `.cu` files.
3. PR #43717 mechanically moves attention/cache kernels into `csrc/libtorch_stable`; identity is preserved for cache and merge kernels.
4. CUDA PagedAttention paths under `csrc/attention` end on 2026-05-29 as a move to `libtorch_stable` (#43717); the artifact ends when libtorch_stable sources/bindings are deleted by PR #47361 on 2026-07-02.
5. The paged KV layout contract survives through `reshape_and_cache` and `reshape_and_cache_flash` in `libtorch_stable/cache_kernels.cu`.
6. FlashAttention delivery has four stages: upstream `flash-attn` (#3005/#3396), `vllm-flash-attn` wheel (#4686), inline CMake FetchContent (#8245/#8699), externalized CMake FetchContent (#13747).
7. FA version selection: PR #12093 introduced FA3 selection/override; cutoff rules prefer FA3 on SM90, FA4 on SM100, else FA2, with compatibility fallbacks.
8. FlashInfer TRTLLM-gen predates XQA: PR #19825 adds SM100 decode, PR #22095 prefill; PR #43232 adds the XQA sibling for SM90/SM12x.
9. ROCm has both a custom paged-attention lineage (`csrc/rocm/attention.cu`) and later AITER/Triton branches selected by `rocm.py` priority logic.
10. MLA is a parallel subfamily with FlashMLA, CUTLASS, FlashAttention, FlashInfer/TRTLLM and AITER branches sharing the V1 registry/interface.

## Open questions
- Did Triton decode attention borrow code from SGLang or LightLLM? Missing: Need PR body/review or code-similarity proof for PR #12528.
- Exact full SHAs/PRs for late sparse MLA variants and CUTLASS SM100 stable-ABI move. Missing: Path census gives dates; targeted git log should be extended before event coding.
- Upstream-side FlashInfer/TensorRT-LLM kernel ancestry for XQA and MLA TRTLLM-gen. Missing: Need upstream FlashInfer/NVIDIA PRs; local vLLM code proves integration only.
