### L3-b9cdc85207  (L3, 2026-03-31, sha b9cdc85207b3, PR #38508)
TITLE: [ROCm][CI] Fix Whisper translation test attention backend selection (#38508)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/entrypoints/openai/speech_to_text/test_translation_validation.py (+39/-6)
LABELS: rocm, ready
BODY: test_translation_validation.py used ROCM_AITER_FA for all models including Whisper, but ROCM_AITER_FA does not support ENCODER_DECODER cross-attention. In this PR, we replace shared rocm_aiter_fa_attention fixture with model-aware _get_rocm_attention_config() that selects ROCM_AITER_UNIFIED_ATTN (MI3XX) or TRITON_ATTN for Whisper, keeping ROCM_AITER_FA for decoder-only models. ⏎  ⏎ ### Test plan ⏎  ⏎  ⏎ cc @kenroche

### L3-202f147cf2  (L3, 2026-03-31, sha 202f147cf213, PR #38631)
TITLE: Fix MLA runs when use_inductor_graph_partition=True (#38631)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+6/-4)
LABELS: ready
BODY: Before this PR, running offline inference with ⏎  ⏎ ``` ⏎ llm = LLM(model="deepseek-ai/DeepSeek-V2-Lite", ⏎           compilation_config=CompilationConfig( ⏎             use_inductor_graph_partition=True, ⏎            ), ⏎     ) ⏎ ``` ⏎ would yield gibberish-y output, despite good eval results. With the changes from this PR, the output is sane. ⏎  ⏎ ### Eval ⏎  ⏎ Ran with ⏎ ``` ⏎ lm-eval --model vllm --model_args '{"pretrained": "deepseek-ai/DeepSeek-V2-Lite",  …[truncated]

### L3-d9b90a07ac  (L3, 2026-03-31, sha d9b90a07aced, PR #36286)
TITLE: [MoE Refactor] Migrate Unquantized to Full Oracle Flow (#36286)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+1/-0); tests/kernels/moe/test_moe.py (+15/-3); tests/kernels/moe/test_unquantized_backend_selection.py (+64/-46); tests/quantization/test_blackwell_moe.py (+7/-0); vllm/lora/layers/fused_moe.py (+3/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py (+148/-0); vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py (+0/-141); vllm/model_executor/layers/fused_moe/fused_moe.py (+4/-0); vllm/model_executor/layers/fused_moe/modular_kernel.py (+12/-0); vllm/model_executor/layers/fused_moe/oracle/unquantized.py (+267/-163); (+1 more)
LABELS: ready, nvidia, ready-run-all-tests
BODY: ## Purpose ⏎  ⏎ Migrate the unquantized MoE (BF16) code path from the legacy kernel initialization pattern to the modern modular pattern already used by FP8 and NvFP4. ⏎  ⏎ The CPU backend is **not** migrated and remains on the old path due to interface differences (see below). ⏎  ⏎ ## Background ⏎  ⏎ There are unquantized ⏎ - Monolithic backends (CPU, FlashInfer TRTLLM) ⏎ - Non-monolithic backends (Triton, AITER, FlashInfer CUTLASS, XPU) ⏎  ⏎ #### In the old path ⏎  ⏎ - Mo …[truncated]

### L3-0fab52f0aa  (L3, 2026-03-31, sha 0fab52f0aa0f, PR #38148)
TITLE: Fix NaN from stale FP4 scale padding in create_fp4_scale_tensor (#38148)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/_custom_ops.py (+2/-2)
LABELS: ready, v1
BODY: ## Summary ⏎  ⏎ - Zero-fill FP4 scale tensors in `create_fp4_scale_tensor` (`torch.empty` → `torch.zeros`) ⏎ - Fixes NaN contamination in MoE expert outputs on Blackwell (GB200) with NVFP4 quantization ⏎  ⏎ ## Root cause ⏎  ⏎ `create_fp4_scale_tensor` allocates the swizzled scale tensor with `torch.empty`. When the number of rows `m` is less than `rounded_m` (rounded up to 128 for the tile boundary), the padding rows' scales contain stale GPU memory. If that m …[truncated]

### L3-31a719bcd3  (L3, 2026-03-31, sha 31a719bcd37a, PR #37887)
TITLE: [ROCm][perf] fix Aiter sparse MLA with MTP>1 (#37887)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/spec_decode/dflash.py (+5/-5); vllm/v1/spec_decode/eagle.py (+22/-20)
LABELS: rocm, speculative-decoding, ready, v1
BODY: ## Purpose ⏎  ⏎ Enable speculative decoding with the MTP method and num_speculative_tokens > 1 for DeepSeek v3.2 and ROCM_AITER_MLA_SPARSE ⏎  ⏎ The check in the eagle `SpecDecodeBaseProposer.propose` only checked the type of the last value of `attn_metadata`, which depends on the order of `self.draft_attn_groups`. A change in iteration order caused the checked type for DeepSeek v3.2 to change from `ROCMAiterMLASparseMetadata` to `DeepseekV32IndexerMe …[truncated]

### L3-40bb175027  (L3, 2026-03-31, sha 40bb17502736, PR #33825)
TITLE: [vLLM IR] 1/N Implement IR skeleton and rms_norm op (#33825)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/kernels.yaml (+10/-0); .github/CODEOWNERS (+4/-0); tests/compile/backend.py (+12/-4); tests/compile/fusions_e2e/test_tp1_quant.py (+3/-0); tests/compile/fusions_e2e/test_tp2_ar_rms.py (+3/-0); tests/compile/fusions_e2e/test_tp2_async_tp.py (+2/-0); tests/compile/passes/distributed/test_sequence_parallelism.py (+7/-7); tests/compile/passes/ir/__init__.py (+0/-0); tests/compile/passes/ir/test_lowering.py (+69/-0); tests/compile/passes/test_fusion.py (+10/-8); (+39 more)
LABELS: rocm, intel-gpu, ready, torch.compile, ci/build, nvidia, ready-run-all-tests, vllm-ir
BODY: ## Purpose ⏎  ⏎ This PR implements the foundational infrastructure for vLLM IR (Intermediate Representation), a functional IR system for vLLM custom operations, starting with the `rms_norm` operation. This is the first of many PRs to addresses RFC #32358. ⏎  ⏎ **What is vLLM IR?** vLLM IR is a functional intermediate representation that separates operation semantics from implementation and dispatching. It serves as a higher-level torch dialect with t …[truncated]

### L3-eb47454987  (L3, 2026-04-01, sha eb4745498745, PR #36178)
TITLE: [Bugfix][MLA] Add logits size budget to sparse indexer prefill chunking (#36178)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+7/-0); vllm/v1/attention/backends/mla/indexer.py (+89/-24); tests/v1/attention/test_sparse_mla_backends.py (+76/-0); vllm/model_executor/layers/sparse_attn_indexer.py (+19/-7)
LABELS: bug, ready, v1
BODY: Alternative to https://github.com/vllm-project/vllm/pull/35488, credit to @haosdent  ⏎  ⏎ ## Summary ⏎ - Adds a logits tensor size constraint to sparse MLA indexer prefill chunking to prevent CUDA OOM ⏎ - Introduces `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` env var (default 512 MB) to bound the [M, N] float32 logits tensor ⏎ - Replaces `split_prefill_chunks` with `split_indexer_prefill_chunks` that respects both workspace and logits size constraints ⏎  ⏎ ## Test pla …[truncated]

### L3-c49497726b  (L3, 2026-04-01, sha c49497726ba4, PR #32914)
TITLE: [ROCm][perf] Shuffle KV cache to use paged_attention_common (#32914)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+38/-4); vllm/_aiter_ops.py (+51/-0)
LABELS: rocm, ready, ci/build, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ For Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 model, currently `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` performs worse on small concurrencies, compared to  `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=0`. This PR fixes the issue using `paged_attention_common` from aiter (see https://github.com/ROCm/aiter/pull/1821). ⏎  ⏎ ## Test Plan ⏎ For input and output lengths of 1k and 8k and concurrencies from 8, 18, 32, 64, 128, compare current main branch with and w …[truncated]

### L3-116f4be405  (L3, 2026-04-01, sha 116f4be405ff, PR #38659)
TITLE: [1/N][Cleanup] Standardize on use of `is_quantized_kv_cache` (#38659)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.rocm.aiter_unified, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.flashmla_sparse, L3.mla.flashinfer_sparse, L3.dispatch.abstract_interface, L3.platform.cuda_selection, L3.flex_attention
FILES: vllm/model_executor/layers/attention/mla_attention.py (+5/-4); vllm/v1/attention/backend.py (+0/-4); vllm/v1/attention/backends/cpu_attn.py (+1/-1); vllm/v1/attention/backends/flash_attn.py (+5/-5); vllm/v1/attention/backends/flash_attn_diffkv.py (+2/-1); vllm/v1/attention/backends/flashinfer.py (+9/-10); vllm/v1/attention/backends/flex_attention.py (+1/-2); vllm/v1/attention/backends/mla/cutlass_mla.py (+1/-1); vllm/v1/attention/backends/mla/flashattn_mla.py (+2/-2); vllm/v1/attention/backends/mla/flashinfer_mla.py (+3/-3); (+18 more)
LABELS: rocm, intel-gpu, ready, v1, cpu, nvidia
BODY: ## Purpose ⏎ This PR will be the first in a series of PRs to resolve some tech debt with KV cache dtypes. This simply standardizes on the use of `is_quantized_kv_cache` throughout the codebase in lieu of `startswith("fp8")`. The change will be useful down the road as we work towards changes like #38124 ⏎  ⏎ ## Test Plan ⏎ CI ⏎  ⏎ ## Test Result ⏎ TBD ⏎  ⏎ --- ⏎ [details omitted]

### L3-36d7f19897  (L3, 2026-04-01, sha 36d7f1989784, PR #38676)
TITLE: [CPU] Support head_size 512 in cpu_attn (#38676)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+1/-1); csrc/cpu/generate_cpu_attn_dispatch.py (+1/-1); docs/design/attention_backends.md (+1/-1); tests/kernels/attention/test_cpu_attn.py (+1/-1)
LABELS: documentation, ready, v1, cpu
BODY: ## Purpose ⏎  ⏎ Add 512 head_size support. ⏎  ⏎ ## Test Plan ⏎  ⏎ CI tests ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-5e30e9b9a9  (L3, 2026-04-01, sha 5e30e9b9a9b0, PR #38359)
TITLE: [Bugfix] Revert "Zero-init MLA attention output buffers to prevent NaN from CUDA graph padding" (#38359)
SOURCES: path_core, subject_keyword, corpus:confirmed-reverts, body_keyword
ARTIFACT_HINTS: L3.mla.cutlass_v1_backend, L3.mla.flashinfer
FILES: vllm/v1/attention/backends/mla/cutlass_mla.py (+1/-14); vllm/v1/attention/backends/mla/flashinfer_mla.py (+0/-44)
LABELS: bug, ready, v1, nvidia
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s)  reason=performance_regression
BODY: ## Summary ⏎  ⏎ - Restores the original `torch.empty` allocation, removing the overhead of pre-allocated zero-init buffers and the `out=` workaround in FlashInfer MLA. ⏎  ⏎ ## Test plan ⏎  ⏎  ⏎ 🤖 Generated with [Claude Code](https://claude.com/claude-code)

### L3-6183cae1bd  (L3, 2026-04-01, sha 6183cae1bd8d, PR #38730)
TITLE: [Bugfix] Restrict TRTLLM attention to SM100, fixing GB300 (SM103) hang (#38730)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+4/-4); docs/design/attention_backends.md (+1/-1); tools/pre_commit/generate_attention_backend_docs.py (+9/-5)
LABELS: bug, documentation, ready, nvidia
ISSUES: #38729 [Bug] All models hang on GB300 (SM103) with FlashInfer 0.6.7
DEEP_STUDY: deep-study: this PR was reverted by PR 40032 (confirmed_revert, reason=build_or_dependency)
BODY: ## Summary ⏎  ⏎ All models hang indefinitely on GB300 (SM103, CC 10.3) during inference with large batch sizes. The GPU shows 99% SM utilization and 0% memory bandwidth. This is a regression introduced by the FlashInfer 0.6.6 to 0.6.7 upgrade, where TRTLLM attention kernels are no longer forward-compatible with SM103. ⏎  ⏎ GB200 (SM100) is not affected. ⏎  ⏎ Related FlashInfer issue: [flashinfer-ai/flashinfer#2939](https://github.com/flashinfer-ai/flashinfer …[truncated]

### L3-dc0428ebb8  (L3, 2026-04-01, sha dc0428ebb879, PR #37940)
TITLE: [NIXL][BUG] Fix Triton heterogeneous TP (#37940)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+13/-4); vllm/v1/attention/ops/triton_reshape_and_cache_flash.py (+9/-3); tests/v1/kv_connector/nixl_integration/config_sweep_accuracy_test.sh (+7/-0); tests/v1/kv_connector/unit/test_nixl_connector.py (+17/-15); vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py (+16/-0)
LABELS: bug, ready, v1, kv-connector
BODY: co-authored with @ZhanqiuHu  ⏎  ⏎ ## Purpose ⏎  ⏎ - Fix Triton Attn Heterogeneous TP Disagg: #37703 ⏎ - Also fixes Gemma with Heterogeneous TP bug, also caused by Triton Backend: #37333 ⏎ - Enable cross-layer TP disagg for Triton, which now has the same KV cache layout as FlashInfer ⏎  ⏎ ## Test Plan ⏎  ⏎ In `tests/v1/kv_connector/nixl_integration/config_sweep_accuracy_test.sh`, replace `tp_configs` with the following: ⏎  ⏎ Fixed GEMMA tests: ⏎ ``` ⏎ "GPU_MEMO …[truncated]

### L3-5f96f9aff1  (L3, 2026-04-02, sha 5f96f9aff10f, PR #38684)
TITLE: [Perf] DSV3.2 Indexer Fused Weights Projection (#38684)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/deepseek_mtp.py (+3/-0); vllm/model_executor/models/deepseek_v2.py (+22/-14)
LABELS: ready, deepseek
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Fuse the WK and Weights_Proj projections in the DSV3.2 Indexer. This is an alternative optimization to https://github.com/vllm-project/vllm/pull/35968, which overlaps the projections instead of fusing them. Doing the fusion provides a greater speedup: ⏎  ⏎ Benchmark timings for DSV3.2 NVFP4 on 8xB200 (TP8, No Specdec) ⏎  ⏎ ### BS128 8k/1k ⏎ ``` ⏎ Fused WK+WeightsProj (Decode):  21.6 ms ⏎ Baseline             (Decode):  22.3 ms ⏎ ``` ⏎  ⏎ ###  …[truncated]

### L3-2ce3d0ce36  (L3, 2026-04-02, sha 2ce3d0ce360b, PR #38378)
TITLE: [Feature] KV cache per-token-head INT8/FP8 quantization (#38378)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.triton.unified_attention, L3.triton.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/model_executor/layers/attention/attention.py (+8/-2); vllm/model_executor/layers/attention/chunked_local_attention.py (+2/-0); vllm/model_executor/layers/attention/cross_attention.py (+6/-1); vllm/model_executor/layers/attention/static_sink_attention.py (+2/-0); vllm/v1/attention/backend.py (+10/-1); vllm/v1/attention/backends/triton_attn.py (+132/-18); vllm/v1/attention/ops/triton_reshape_and_cache_flash.py (+198/-1); vllm/v1/attention/ops/triton_unified_attention.py (+189/-36); docs/design/attention_backends.md (+1/-1); tests/models/quantization/test_per_token_kv_cache.py (+94/-0); (+6 more)
LABELS: documentation, rocm, ready, v1, quantization
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: Continue of PR: https://github.com/vllm-project/vllm/pull/36893 with changes requested by @mgoin ⏎ At comment: ⏎ https://github.com/vllm-project/vllm/pull/36893#issuecomment-4143255216 ⏎  ⏎ This PR adds per-token-head kv cache quantization to the Triton attention backend ⏎  ⏎ INT8_PER_TOKEN, FP8_PER_TOKEN ⏎ ```md ⏎ | Metric | FP16 | FP8 | INT8_PER_TOKEN | FP8_PER_TOKEN | ⏎ |---|---|---|---|---| ⏎ | Duration (s) | 89.17 | 91.21 | **80.95 ✅** | 104.45 | ⏎ | R …[truncated]

### L3-16a65e4173  (L3, 2026-04-02, sha 16a65e41736c, PR #38427)
TITLE: [Bugfix] Enable batch-invariant Triton matmul on all Ampere GPUs (SM 8x)  (#38427)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/batch_invariant.py (+3/-5)
LABELS: bug, ready
ISSUES: #38286 [Feature]: Batch invariance on 3090
BODY: ## Purpose                                                                                                                                                                        ⏎     ⏎ Enable batch-invariant Triton matmul on all Ampere GPUs (SM 80, 86, 87), fixing batch invariance on RTX 3090/3080/A6000/Jetson Orin. ⏎                    ⏎ The Triton matmul overrides in `batch_invariant.py` were only registered for SM 80 (A100), SM 89 (4090), and SM  …[truncated]

### L3-d9408ffba3  (L3, 2026-04-02, sha d9408ffba3c8, PR #33529)
TITLE: Triton MLA perf fixes (#33529)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.triton.decode_attention
FILES: vllm/v1/attention/backends/mla/triton_mla.py (+22/-1); vllm/v1/attention/ops/triton_decode_attention.py (+47/-25)
LABELS: performance, ready, v1, deepseek
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Triton MLA on sm120 performance degrades on batch size 1 as context length increases. ⏎  ⏎ This perf issue has been bugging me for a while as it made deepseek and Kimi k2 unusable, and when Kimi k2.5 was released I finally got around to digging into it. I'm familiar with w/ cuda but this is my first foray into triton.  ⏎  ⏎ The primary issues are suboptimal kv splitting during low batch count resulting in underutilized SM and unnecessary load …[truncated]

### L3-ecd5443dbc  (L3, 2026-04-02, sha ecd5443dbcd1, PR #38062)
TITLE: Bump helion dependency from 0.3.2 to 0.3.3 (#38062)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: .buildkite/test-amd.yaml (+1/-1); .buildkite/test_areas/kernels.yaml (+1/-1); setup.py (+4/-1)
LABELS: ready, ci/build
BODY: ## Summary ⏎ - Bumps the pinned `helion` optional dependency version from 0.3.2 to 0.3.3 in `setup.py`. ⏎  ⏎ ## Test plan ⏎ - Verified that `pip install -e ".[helion]"` resolves the new version correctly. ⏎  ⏎ 🤖 Generated with [Claude Code](https://claude.com/claude-code)

### L3-58262dec6e  (L3, 2026-04-02, sha 58262dec6e81, PR #38791)
TITLE: [Bugfix] Fix test mocks after SM100 restriction in #38730 (#38791)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_use_trtllm_attention.py (+3/-3)
LABELS: bug, ready, nvidia
DEEP_STUDY: deep-study: this PR was reverted by PR 40032 (confirmed_revert, reason=build_or_dependency)
BODY: ## Summary ⏎  ⏎ - Update three test mocks in `test_use_trtllm_attention.py` from `is_device_capability_family` to `is_device_capability` to match the production code change in #38730 ⏎ - This is a direct follow-up to #38730 which changed `supports_trtllm_attention()` to use `is_device_capability(100)` instead of `is_device_capability_family(100)` ⏎ - Supersedes the auto-revert #38773 ⏎  ⏎ ## Test plan ⏎  ⏎ ## Why CI missed this on #38730 ⏎  ⏎ The "Kernels Attention  …[truncated]

### L3-c6f722b93e  (L3, 2026-04-02, sha c6f722b93e8e, PR #38770)
TITLE: [CPU] Support gelu act in cpu_fused_moe (#38770)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/v1/attention/backends/cpu_attn.py (+2/-1); csrc/cpu/cpu_fused_moe.cpp (+43/-1); tests/kernels/moe/test_cpu_fused_moe.py (+1/-1); vllm/envs.py (+5/-0); vllm/model_executor/layers/fused_moe/cpu_fused_moe.py (+8/-0)
LABELS: ready, v1, cpu
BODY: ## Purpose ⏎  ⏎ - Add gelu act in cpu_fused_moe ⏎  ⏎ ## Test Plan ⏎  ⏎ CI tests ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-cb3935a8fc  (L3, 2026-04-02, sha cb3935a8fc19, PR #38690)
TITLE: [FA4] Update flash-attention to latest upstream FA4 (#38690)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); requirements/cuda.txt (+2/-2)
LABELS: ready, ci/build, nvidia
BODY: Testing PR for updating FA4 to latest upstream

### L3-188defbd0b  (L3, 2026-04-02, sha 188defbd0bad, PR #38792)
TITLE: [CI] Add flashinfer.py to attention test source deps (#38792)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/kernels.yaml (+1/-0)
LABELS: ready, ci/build
BODY: ## Summary ⏎  ⏎ - Add `vllm/utils/flashinfer.py` to the source file dependencies of the "Kernels Attention Test" job ⏎ - Without this, changes to `flashinfer.py` do not trigger the attention test job in PR CI. The tests only run in nightly, which is how #38730 passed PR CI but broke nightly (fixed in #38791)

### L3-7b743ba953  (L3, 2026-04-02, sha 7b743ba953b0, PR #38836)
TITLE: [CI] Fix: pass string cache_dtype in test_register_kv_caches (#38836)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_nixl_connector.py (+1/-1)
LABELS: ready, v1, kv-connector
BODY: ## Summary ⏎ - Fix `test_register_kv_caches[TRITON_ATTN-True]` failure related to #38378 ⏎ - The test passed `torch.bfloat16` (a `torch.dtype` object) as `cache_dtype`, but `allocate_uniform_kv_caches` expects a `CacheDType` string (e.g. `"bfloat16"`) ⏎ - This caused `AttributeError: 'torch.dtype' object has no attribute 'startswith'` in `get_kv_quant_mode` ⏎  ⏎ ## Test plan

### L3-8b141ed8c3  (L3, 2026-04-02, sha 8b141ed8c328, PR #36298)
TITLE: full cudagraph for flex-attn (#36298)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+91/-0); tests/compile/fullgraph/test_full_cudagraph.py (+0/-11); tests/kernels/test_flex_attention.py (+53/-0); vllm/v1/worker/gpu_model_runner.py (+1/-0)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Make vllm flex-attention backend support full cudagraph. Previously it only support piecewise cudagraphs. ⏎  ⏎ There are mainly 2 reasons that full cudagraph does not work upfront for flex-attn ⏎ 1. the warm-up run and the capture-run use different max-seq-len. That causes re-compilation during graph capture and fail. ⏎ 2. there are multiple tensors prepared when creating the flex attn metadata and used by model.forward. Those tensors does not use persis …[truncated]

### L3-1f5ec2889c  (L3, 2026-04-02, sha 1f5ec2889c41, PR #36205)
TITLE: [mla] Support fused FP8/NVFP4 output quantization in MLA attention (#35792) (#36205)
SOURCES: path_core, path_integration+keyword, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.common_v1, L3.dispatch.abstract_interface
FILES: vllm/compilation/passes/fusion/mla_attn_quant_fusion.py (+262/-0); vllm/compilation/passes/pass_manager.py (+2/-0); vllm/config/compilation.py (+1/-1); vllm/model_executor/layers/attention/mla_attention.py (+49/-5); vllm/v1/attention/backend.py (+21/-0); .buildkite/test_areas/compile.yaml (+2/-0); docs/design/fusions.md (+18/-2); tests/compile/fusions_e2e/conftest.py (+9/-3); tests/compile/fusions_e2e/models.py (+32/-1); tests/compile/fusions_e2e/test_tp1_quant.py (+9/-2); (+2 more)
LABELS: documentation, performance, ready, ci/build, v1
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ Phase 1 (of 2) for issue #35792. ⏎  ⏎ Adds fused output quantization (FP8 static and NVFP4) for MLA attention, eliminating a separate quant kernel per layer by folding ⏎ quantization into the `unified_mla_attention_with_output` custom op. ⏎  ⏎ ## Changes ⏎  ⏎ - `mla_attention.py`: Implement fused output quant in `forward_impl` via temp-buffer swap. ⏎ - `mla_attn_quant_fusion.py` (new): Pattern matcher + `MLAAttnFusionPass` for MLA attn→quant fusion. ⏎  …[truncated]

### L3-3bc2734dd0  (L3, 2026-04-03, sha 3bc2734dd03d, PR #36518)
TITLE: [Kernel] Fuse FP8 output quantization into merge_attn_states (#36518)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.merge.triton_lse, L3.merge.cuda_lse
FILES: csrc/attention/merge_attn_states.cu (+133/-41); csrc/ops.h (+2/-1); csrc/torch_bindings.cpp (+2/-1); vllm/_custom_ops.py (+2/-0); vllm/v1/attention/ops/merge_attn_states.py (+25/-10); vllm/v1/attention/ops/triton_merge_attn_states.py (+24/-2); benchmarks/fused_kernels/merge_attn_states_benchmarks.py (+264/-0); tests/kernels/attention/test_merge_attn_states.py (+64/-15)
LABELS: documentation, performance, rocm, ready, ci/build, v1, llama, qwen, nvidia
ISSUES: #33097 [Feature]: Fuse FP8 output quantization into merge_attn_states (DCP / cascade paths)
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎ Closes #33097  ⏎  ⏎ - Add optional `output_scale` parameter to `merge_attn_states` (CUDA kernel, Triton kernel, Python bindings, dispatcher) for fused FP8 static per-tensor quantization ⏎ - When `output_scale` is provided, the kernel quantizes merged attention output directly to FP8 during the final store, eliminating a separate quantization kernel launch and BF16 memory round-trip ⏎ - Backward compatible — existing callers that omit `output_s …[truncated]

### L3-66e86f1dbd  (L3, 2026-04-03, sha 66e86f1dbd56, PR #37416)
TITLE: [Kernel] Mamba support different layout for Conv state (#37416)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/models/language/generation/test_hybrid.py (+23/-0); vllm/envs.py (+8/-0); vllm/model_executor/layers/kda.py (+11/-5); vllm/model_executor/layers/mamba/gdn_linear_attn.py (+15/-2); vllm/model_executor/layers/mamba/mamba_mixer.py (+7/-3); vllm/model_executor/layers/mamba/mamba_mixer2.py (+10/-4); vllm/model_executor/layers/mamba/mamba_utils.py (+73/-17); vllm/model_executor/layers/mamba/ops/causal_conv1d.py (+0/-4); vllm/model_executor/layers/mamba/short_conv.py (+6/-2); vllm/model_executor/models/olmo_hybrid.py (+8/-1); (+1 more)
LABELS: ready, qwen
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: This PR proposes swapping the 2D layout of the Conv state from `(state_len, dim)` to `(dim, state_len)`, similarly to the work done for NHD->HND layout for the same purposes of efficient indexing in PD. ⏎ This new layout leads to overall better performance (TTFT particularly) as well as enabling HeterogeneousTP support for PD disagg deployments. ⏎  ⏎ ## The problem ⏎  ⏎ Mamba layout is currently inefficient for heterogeneous TP in disaggregated scenar …[truncated]

### L3-4a06e1246e  (L3, 2026-04-03, sha 4a06e1246e30, PR #38460)
TITLE: [Perf] Batch KV cache swap copies via cuMemcpyBatchAsync (#38460)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+55/-0); csrc/cache.h (+4/-0); csrc/torch_bindings.cpp (+6/-0); vllm/_custom_ops.py (+16/-0); vllm/v1/kv_offload/worker/cpu_gpu.py (+37/-15)
LABELS: ready, v1
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:bd8bd52308 (fix PR 38919) || deep-study performance PR (system_performance)
BODY: Replace per-layer per-block `swap_blocks` calls in the KV cache offloading ⏎ handler with a single `swap_blocks_batch` call that submits all copies in ⏎ one driver invocation. ⏎  ⏎ On CUDA 12.8+ this uses `cuMemcpyBatchAsync`; on older CUDA/ROCm it falls ⏎ back to a flat `cudaMemcpyAsync` loop with `cudaMemcpyDefault`. Zero extra ⏎ GPU memory. No behavior change. ⏎  ⏎ Supersedes #38216 (rebased clean). ⏎  ⏎ ## Benchmark Results ⏎  ⏎ **Hardware:** 8xH100 80GB HBM3, CUDA  …[truncated]

### L3-1b117cb0ac  (L3, 2026-04-03, sha 1b117cb0ac51, PR #38615)
TITLE: [ROCm] Fix aiter persistent mode mla with q/o nhead<16 for kimi-k2.5 tp8 (#38615)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+4/-3)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎ AiterMLAImpl already contains a head-padding mechanism: when num_heads is 4 or 8 (e.g., Kimi-K2.5 with 64 heads under TP=8 yields 8 heads per GPU), the q tensor is padded to 16 heads via repeat_interleave before being passed to the kernel. ⏎  ⏎ persistent mode mla implementation allocated buffer for the original head count (e.g., 8), while the kernel receives a q tensor with 16 heads — causing a shape mismatch between the pre-allocated  …[truncated]

### L3-ee3cf45739  (L3, 2026-04-03, sha ee3cf457398e, PR #33657)
TITLE: [XPU] Initial support for GDN attention on Qwen3-next/Qwen3.5 (#33657)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/layernorm.py (+5/-0); vllm/model_executor/layers/mamba/gdn_linear_attn.py (+94/-0); vllm/platforms/xpu.py (+51/-0)
LABELS: rocm, intel-gpu, ready, v1, qwen, nvidia
BODY: ## Purpose ⏎ This PR enables Qwen3-next/Qwen3.5 support for XPU path, using triton attention due to k/v in-contiguous not supported. Need mamba cache block size fix like #37467  ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ VLLM_WORKER_MULTIPROC_METHOD=spawn python3 examples/basic/offline_inference/generate.py --model Qwen/Qwen3.5-9B  --enforce-eager  --attention-backend=TRITON_ATTN ⏎  ⏎ ``` ⏎ ## Test Result ⏎  ⏎ <img width="733" height="187" alt="image" src="https://github.c …[truncated]

### L3-5f1de2b14b  (L3, 2026-04-03, sha 5f1de2b14b1d, PR #38758)
TITLE: [Model Runner V2] Add config validation for not-yet-supported features (#38758)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/model_runner_v2.yaml (+0/-1); vllm/config/vllm.py (+46/-0)
LABELS: ready, ci/build, mrv2
BODY: Not necessarily exhaustive yet.

### L3-580090db6b  (L3, 2026-04-03, sha 580090db6bcb, PR #38325)
TITLE: [Kernel] Add swapAB support for SM120 CUTLASS blockwise FP8 GEMM  (#38325)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh (+82/-26)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎ This PR adds swap AB kernel support for the SM120 (Blackwell) blockwise FP8 scaled GEMM.  ⏎  ⏎ For decode-phase workloads and small-batch inference, the M dimension of the GEMM is very small. Without swap AB, the tile partitioning along M is highly inefficient — most threads within a CTA tile are idle. By transposing the problem as `D = (B^T @ A^T)^T`, the small dimension is moved to N, enabling better tile coverage and higher SM occupa …[truncated]

### L3-062f1a2d70  (L3, 2026-04-03, sha 062f1a2d706c, PR #38915)
TITLE: [Bug] Fix compile error for `swap_blocks_batch` in CUDA 13 (#38915)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+17/-8)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎  ⏎ Originally ⏎  ⏎ ```bash ⏎ [1/3] Building CUDA object CMakeFiles/_C.dir/csrc/cache_kernels.cu.o ⏎ FAILED: [code=255] CMakeFiles/_C.dir/csrc/cache_kernels.cu.o  ⏎ ccache /usr/local/cuda-13.0/bin/nvcc -forward-unknown-to-host-compiler -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1 -DPy_LIMITED_API=3 -DTORCH_EXTENSION_NAME=_C -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_NVSHMEM -DUSE_RPC -DUSE_TENSORPIPE -D_C_EXPORTS -I/home/yewentao2 …[truncated]

### L3-a5a623d961  (L3, 2026-04-04, sha a5a623d96163, PR #38859)
TITLE: [Bugfix] Re-enable Renormalize routing for TRT-LLM MoE experts (#38859)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py (+2/-5); vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+5/-7)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎  ⏎ Re-enable `Renormalize` and `RenormalizeNaive` routing for TRT-LLM MoE experts (BF16 and FP8). ⏎  ⏎ These were disabled in #37591 because the monolithic kernel's internal Renormalize routing produced output uncorrelated with the modular/Triton kernel for Qwen3.5 models. The root cause was a flashinfer bug (https://github.com/flashinfer-ai/flashinfer/issues/2822), fixed in 0.6.7. ⏎  ⏎ ## Test Plan ⏎  ⏎ Verified with reproduction scripts th …[truncated]

### L3-99e5539a67  (L3, 2026-04-05, sha 99e5539a6701, PR #38981)
TITLE: [Perf][GDN] Align TMA usage with upstream FLA (#38981)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fla/ops/utils.py (+7/-3)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Align `is_tma_supported` in vLLM's GDN kernels with upstream FLA behavior: TMA is now disabled by default and requires `FLA_USE_TMA=1` to enable. ⏎  ⏎ vLLM's copy of FLA GDN kernels was forked before [fla-org/flash-linear-attention#607](https://github.com/fla-org/flash-linear-attention/issues/607) / [2eade97](https://github.com/fla-org/flash-linear-attention/commit/2eade97), so `is_tma_supported` is unconditionally `True` on SM90+ (Ho …[truncated]

### L3-228023b3a5  (L3, 2026-04-05, sha 228023b3a58f, PR #38990)
TITLE: [Bugfix][MoE] Fix 6-8% decode regression: prefer multi-stream shared expert overlap (#38990)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/runner/shared_experts.py (+9/-9)
LABELS: bug, ready, ready-run-all-tests
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ** NOTE (rob): this regression does not impact any official vLLM release ** ⏎  ⏎ ## Summary ⏎  ⏎ Fix **6-8% decode throughput regression** on MoE models with TP-only configurations (no EP/EPLB), introduced by #35153. ⏎  ⏎ ### Root cause ⏎  ⏎ In `SharedExperts._determine_shared_experts_order()`, the `_has_external_experts` check is evaluated **before** the aux-stream overlap check. For TP-only configs, `_has_external_experts` returns `True` (neither EPLB  …[truncated]

### L3-1af6f78ae5  (L3, 2026-04-05, sha 1af6f78ae5a1, PR #38993)
TITLE: [Perf] Change Trtllm fp8 MoE to use Shuffled Weights and BlockMajorK Layout (#38993)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+28/-5); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+38/-0); vllm/model_executor/warmup/deep_gemm_warmup.py (+6/-8)
LABELS: ready, nvidia, ready-run-all-tests
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Change Trtllm fp8 MoE to use Shuffled weights and BlockMajorK layout. Adapted from examples in https://github.com/flashinfer-ai/flashinfer/blob/19329d838236803c59c8f571d42910b1c0c2a18e/tests/moe/test_trtllm_gen_fused_moe.py#L838 ⏎  ⏎ Benchmarking shows this improves performance across the board. ⏎  ⏎ ## Test Plan ⏎ Ensure model evals look good for both the modular (MiniMaxAI/MiniMax-M2.5) and monolithic (deepseek-ai/DeepSeek-R1) kernel cod …[truncated]

### L3-f6983f01de  (L3, 2026-04-05, sha f6983f01de2b, PR #37512)
TITLE: MiniMax-M2: add Eagle3 speculative decoding support (#37512)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/registry.py (+6/-0); vllm/config/speculative.py (+1/-0); vllm/model_executor/models/minimax_m2.py (+16/-5); vllm/model_executor/models/registry.py (+1/-0)
LABELS: new-model, ready
BODY: - Add SupportsEagle3 interface to MiniMaxM2ForCausalLM with aux hidden state collection in MiniMaxM2Model.forward() ⏎ - Register Eagle3MiniMaxM2ForCausalLM in speculative decoding models ⏎ - Add "minimax" to eagle3_target_supported whitelist ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-608914de30  (L3, 2026-04-06, sha 608914de3038, PR #38944)
TITLE: [Core] Re-enable Inductor pre-grad passes in standalone compile (torch>=2.12) (#38944)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/compilation/compiler_interface.py (+2/-2)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Re-enable Inductor pre-grad passes in vLLM's standalone compile path. ⏎ PyTorch 2.12+ no longer runs pre-grad passes before the cache lookup ⏎ (`pre_grad_pass_timing = "default"`), so the monkey-patch that bypassed ⏎ `_recursive_pre_grad_passes` is no longer needed. Removing it re-enables ⏎ pre-grad passes, including any custom passes registered via PyTorch's ⏎ pass infrastructure, with no impact on compile time. ⏎  ⏎ ## Test Plan ⏎  ⏎ Bench …[truncated]

### L3-780ba37458  (L3, 2026-04-06, sha 780ba3745836, PR #38501)
TITLE: [ROCm][Quantization] Add asymmetric INT8 quantization support to TritonInt8ScaledMMLinearKernel (#38501)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/lm-eval-harness/configs/Meta-Llama-4-Maverick-17B-128E-Instruct-FP8.yaml (+3/-0); .buildkite/lm-eval-harness/configs/Qwen2.5-VL-3B-Instruct-FP8-dynamic.yaml (+3/-0); .buildkite/lm-eval-harness/configs/Qwen3-235B-A22B-Instruct-2507-FP8.yaml (+3/-0); .buildkite/lm-eval-harness/configs/models-small-rocm.txt (+1/-0); .buildkite/lm-eval-harness/test_lm_eval_correctness.py (+32/-0); .buildkite/test-amd.yaml (+18/-0); vllm/model_executor/kernels/linear/scaled_mm/triton.py (+73/-14)
LABELS: rocm, ready, ci/build
BODY: - Adds asymmetric (AZP) INT8 quantization support to `TritonInt8ScaledMMLinearKernel`, which previously only supported symmetric input quantization. ⏎ - This unblocks asymmetric INT8 models (e.g. W8-Channel-A8-Dynamic-Asym-Per-Token) on ROCm, where the Triton kernel is the only viable INT8 fallback when AITER is not enabled. ⏎ - The implementation follows the same math as `CutlassInt8ScaledMMLinearKernel`: precomputes `azp_adj` (weight column sums) …[truncated]

### L3-47e605092b  (L3, 2026-04-06, sha 47e605092b7f, PR #38879)
TITLE: [Gemma4] Enable Fast Prefill Optimization (#38879)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/gemma4.py (+369/-47)
LABELS: new-model, ready, multi-modality, tool-calling
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ Add `--kv-sharing-fast-prefill` support for Gemma 4 models, porting the YOCO (You Only Cache Once) fast prefill optimization from Gemma3n. When enabled, the cross-decoder layers (KV-shared) skip prefill tokens and only process decode tokens, significantly reducing prefill latency and improving throughput under concurrent load. ⏎  ⏎ shout-out to @sarckk for the original optimzation (https://github.com/vllm-project/vllm/pull/22628) ⏎  ⏎ # …[truncated]

### L3-4ae218c122  (L3, 2026-04-06, sha 4ae218c122f7, PR #38842)
TITLE: [Refactor] Remove unused dead code (#38842)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.flashmla_v1_adapter
FILES: vllm/v1/attention/ops/flashmla.py (+0/-13); vllm/model_executor/models/mlp_speculator.py (+0/-53); vllm/v1/executor/ray_distributed_executor.py (+0/-8)
LABELS: speculative-decoding, ready, v1
BODY: ## Purpose ⏎  ⏎ Remove unused dead code

### L3-e8ebbdde83  (L3, 2026-04-06, sha e8ebbdde8304, PR #38251)
TITLE: [Quantization] Add FlashInfer CuteDSL batched experts backend for NVFP4 MoE (#38251)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+18/-0); tests/kernels/moe/test_cutedsl_moe.py (+1/-1); vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_batched_moe.py (+353/-0); vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py (+64/-244); vllm/model_executor/layers/fused_moe/oracle/nvfp4.py (+46/-2); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+95/-1)
LABELS: ready, nvidia, ready-run-all-tests
BODY: #38050 #38169

### L3-00d7b497b3  (L3, 2026-04-06, sha 00d7b497b336, PR #35733)
TITLE: [NVFP4] Support NVFP4 dense models from `modelopt` and `compressed-tensors` on AMD Instinct MI300, MI355X and Hopper through emulation (#35733)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.platform.rocm_selection
FILES: tests/models/quantization/test_nvfp4.py (+15/-4); tests/quantization/test_compressed_tensors.py (+1/-4); vllm/envs.py (+5/-0); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py (+23/-0); vllm/model_executor/layers/quantization/modelopt.py (+19/-0); vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py (+1/-1); vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py (+22/-9); vllm/model_executor/layers/quantization/utils/nvfp4_utils.py (+99/-40); vllm/platforms/rocm.py (+1/-0); vllm/utils/import_utils.py (+5/-0)
LABELS: rocm, ready
BODY: ## Purpose ⏎  ⏎ NVFP4 models as https://huggingface.co/RedHatAI/Qwen3-8B-NVFP4 or https://huggingface.co/nvidia/Qwen3-8B-NVFP4 are supported on Blackwell, but not runnable on other devices, nor tested. ⏎  ⏎ This PR: ⏎ - Makes it so that the emulation dispatch `backend = NvFp4LinearBackend.EMULATION` is by default selected on ROCm ⏎ - Fix correctness of NVFP4 models loading when using emulation backend (wrongful div/mul by global scale). ⏎ - Fix CUDA Graph capt …[truncated]

### L3-9c81f35b1a  (L3, 2026-04-06, sha 9c81f35b1ae6, PR #38819)
TITLE: [Attention][MLA] Re-enable FA4 as default MLA prefill backend (#38819)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/attention.py (+1/-1)
LABELS: ready, nvidia
BODY: Reverts vllm-project/vllm#38562 ⏎  ⏎ NaN issue resulting in correctness problems for MLA models https://github.com/vllm-project/vllm/issues/36763 has been resolved by updating FA4 (https://github.com/vllm-project/vllm/pull/38690) to capture the upstream fix https://github.com/Dao-AILab/flash-attention/commit/02931551ece7eb7f36e94302ad79daee6beda2e6 ⏎  ⏎ This PR makes FA4 the default again due to its superior performance (see benchmarks in https://git …[truncated]

### L3-94fbb09894  (L3, 2026-04-06, sha 94fbb09894a0, PR #38799)
TITLE: [EASY] Drop duplicate KV-cache initialization (#38799)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/attention.py (+0/-3)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ In KV-cache quantization initialization (`_init_kv_cache_quant`), quantization method (`quant_method`) is defined twice (duplicate). This PR drop not-used one for brevity. ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ pytest -sv tests/models/quantization/test_fp8.py ⏎ ``` ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-5c35517a3e  (L3, 2026-04-07, sha 5c35517a3e66, PR #39123)
TITLE: [ROCm] Remove unused IS_FNUZ parameter from reshape_and_cache_shuffle_kernel (#39123)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+0/-2)
LABELS: rocm, ready, v1
BODY: `IS_FNUZ` is declared as a `tl.constexpr` kernel parameter and computed at the call site via `current_platform.fp8_dtype() == torch.float8_e4m3fnuz`, but it's never referenced in the kernel body (lines 233-266). This means every call to `reshape_and_cache_shuffle` does a platform check for nothing. ⏎  ⏎ Also, since `IS_FNUZ` is a `tl.constexpr`, different values would cause Triton to JIT-compile separate kernel variants — doubling compile time for no …[truncated]

### L3-96b5004b71  (L3, 2026-04-07, sha 96b5004b7110, PR #37636)
TITLE: [KVConnector] Support 3FS KVConnector (#37636)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: setup.py (+1/-0); tests/v1/kv_connector/unit/test_hf3fs_client.py (+284/-0); tests/v1/kv_connector/unit/test_hf3fs_connector.py (+230/-0); tests/v1/kv_connector/unit/test_hf3fs_metadata_server.py (+193/-0); vllm/distributed/kv_transfer/kv_connector/factory.py (+5/-2); vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/__init__.py (+0/-0); vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_client.py (+298/-0); vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py (+1195/-0); vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_metadata_server.py (+530/-0); vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/__init__.py (+0/-0); (+4 more)
LABELS: ready, ci/build, v1, kv-connector, nvidia
BODY: ## Overview ⏎ This PR introduces the implementation of the 3FS KVConnector for vLLM. ⏎  ⏎ The 3FS KVConnector enables efficient offloading and sharing of KV caches across nodes, significantly accelerating long-context inference scenarios. Alongside the core implementation, we provide the 3FS Operator for one-click deployment and a mini3fs setup for easy local verification. ⏎  ⏎ Deployment Operator: [aliyun/kvc-3fs-operator](https://github.com/aliyun/k …[truncated]

### L3-70406eb1dc  (L3, 2026-04-07, sha 70406eb1dc19, PR #39125)
TITLE: [Attention][V0 Deprecation] Deprecate accept output buffer (#39125)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.rocm.aiter_unified, L3.mla.common_v1, L3.mla.flashmla_sparse, L3.mla.flashinfer_sparse, L3.mla.rocm_aiter_sparse, L3.dispatch.abstract_interface, L3.flex_attention, L3.tree_attention
FILES: vllm/model_executor/layers/attention/attention.py (+53/-96); vllm/model_executor/layers/attention/cross_attention.py (+1/-1); vllm/model_executor/layers/attention/mla_attention.py (+21/-76); vllm/v1/attention/backend.py (+1/-5); vllm/v1/attention/backends/cpu_attn.py (+1/-3); vllm/v1/attention/backends/flash_attn.py (+1/-3); vllm/v1/attention/backends/flash_attn_diffkv.py (+1/-2); vllm/v1/attention/backends/flashinfer.py (+1/-4); vllm/v1/attention/backends/flex_attention.py (+1/-3); vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+0/-1); (+12 more)
LABELS: rocm, intel-gpu, ready, v1, cpu, nvidia
BODY: `accept_output_buffer` is a holdover from V0; all V1 backends accept output buffers for PIECEWISE cudagraphs

### L3-5daf62271d  (L3, 2026-04-07, sha 5daf62271d20, PR #38496)
TITLE: [Model Runner V2] Fuse probabilistic rejection sample kernels (#38496)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/model_runner_v2.yaml (+2/-0); tests/v1/spec_decode/test_probabilistic_rejection_sampler_utils.py (+215/-0); vllm/v1/worker/gpu/sample/gumbel.py (+54/-25); vllm/v1/worker/gpu/spec_decode/probabilistic_rejection_sampler_utils.py (+612/-0); vllm/v1/worker/gpu/spec_decode/rejection_sampler.py (+3/-352)
LABELS: speculative-decoding, ready, ci/build, v1, mrv2
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: This PR accomplishes the following: ⏎ - Fuses kernels to improve rejection sample performance ⏎ - Adds test for rejection sampler correctness (via chi-squared goodness-of-fit test), which checks that the sampled tokens match the target probability distribution up to some threshold. ⏎  ⏎ `probabilistic_rejection_sample` no longer calls `torch.softmax` for either the target or draft logits. In fact, now the order of memory allocations in probabilistic_ …[truncated]

### L3-92b9afeecd  (L3, 2026-04-08, sha 92b9afeecde8, PR #39088)
TITLE: [XPU] Quick fix for TritonMLA to remove cuda hardcode (#39088)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/triton_mla.py (+2/-1); vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py (+1/-1)
LABELS: intel-gpu, ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ Run deepseek-v2-lite family with TrtionMLA on Intel GPU ⏎  ⏎ ## Test Plan ⏎  ⏎ Test with XPU MOE ⏎  ⏎ ``` ⏎ pip install vllm_xpu_kernels@https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.5/vllm_xpu_kernels-0.1.5-cp38-abi3-manylinux_2_28_x86_64.whl ⏎ python ../examples/basic/offline_inference/generate.py --model deepseek-ai/DeepSeek-V2-Lite --tensor-parallel-size 2 --enforce-eager ⏎ ``` ⏎ ``` ⏎ ---------------------------- …[truncated]

### L3-0e9f0a516c  (L3, 2026-04-08, sha 0e9f0a516c98, PR #38580)
TITLE: [ROCm][CI-Build] Cherry pick triton BUFFER_OPS fix and update AITER (#38580)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+2/-2); docker/Dockerfile.rocm_base (+5/-1)
LABELS: rocm, ready, ci/build
BODY: Cherry pick https://github.com/triton-lang/triton/pull/9541 into the triton build ⏎ Silence the spammy warning that used to be handled by Wno-unused-result ⏎ Bump AITER version

### L3-56c976c1b5  (L3, 2026-04-08, sha 56c976c1b5cc, PR #38817)
TITLE: [ROCm] Enable fused_silu_mul_block_quant on ROCm (#38817)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+2/-2); csrc/ops.h (+0/-2); csrc/quantization/fused_kernels/quant_conversions.cuh (+1/-1); csrc/quantization/w8a8/fp8/common.cuh (+1/-1); csrc/torch_bindings.cpp (+12/-11); tests/kernels/core/test_fused_silu_mul_block_quant.py (+10/-10); vllm/compilation/passes/fusion/act_quant_fusion.py (+2/-2)
LABELS: rocm, ready, ci/build
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: Another follow up for #32996 ⏎ This time properly enabling the new kernel on ROCm instead of guarding ⏎  ⏎ Include path changes are needed because the hipify script would ignore absolute include paths and multiple slightly different versions of the same header would end up being included, causing symbol redefinition errors. ⏎  ⏎ Setting the device index globally in the test solves the IMA error from torch on ROCm

### L3-308cec5864  (L3, 2026-04-08, sha 308cec586489, PR #38814)
TITLE: [FlashAttention] Symlink FA4 instead of copying when using `VLLM_FLASH_ATTN_SRC_DIR` (#38814)
SOURCES: path_core, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+27/-15); vllm/vllm_flash_attn/__init__.py (+20/-1)
LABELS: performance, ready, ci/build
BODY: ## Purpose ⏎ FA4 is written in CuTe DSL, which uses JIT caching so it doesn't require AOT compilation. When installing FA4, cmake just copies the files in the `cute/` directory into `vllm/vllm_flash_attn`. This means that when working on FA4 locally, we need to run the build step before testing, in order for the files to be copied. ⏎  ⏎ This PR symlinks the files instead of copying when `VLLM_FLASH_ATTN_SRC_DIR` is set in order to eliminate the need …[truncated]

### L3-b55d830ec7  (L3, 2026-04-08, sha b55d830ec782, PR #37421)
TITLE: [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode (#37421)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+0/-12); .buildkite/test_areas/kernels.yaml (+3/-2); csrc/ops.h (+3/-3); csrc/persistent_topk.cuh (+1321/-0); csrc/topk.cu (+139/-358); csrc/torch_bindings.cpp (+3/-4); tests/kernels/test_top_k_per_row.py (+540/-78); vllm/model_executor/layers/sparse_attn_indexer.py (+24/-24); vllm/model_executor/models/deepseek_v2.py (+6/-2)
LABELS: performance, ready, ci/build, v1, deepseek, nvidia
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎   Redesigns the persistent TopK kernel used by DSA as a true persistent scheduler with dynamic per-row path selection. ⏎  ⏎   This supersedes and closes #34265, which took a CUDAGraph-specialization approach. Instead, this PR follows a persistent scheduler pattern where a single fixed-grid kernel dynamically dispatches each row to the appropriate path at runtime. ⏎  ⏎ ## Problem ⏎  ⏎   As #34265 demonstrated, there are four different topK …[truncated]

### L3-e24e0a43a4  (L3, 2026-04-08, sha e24e0a43a4f6, PR #38835)
TITLE: [Attention] relax the head dim 512 and paged kv for sm90+FA4 (#38835)
SOURCES: path_core, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fork_build, L3.flash_attn.fa4_cutedsl, L3.flash_attn.fa_utils
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/v1/attention/backends/fa_utils.py (+11/-0); vllm/v1/attention/backends/flash_attn.py (+17/-2); vllm/vllm_flash_attn/flash_attn_interface.py (+0/-7)
LABELS: ready, ci/build, v1
BODY: ## Purpose ⏎ This PR updates the checks for FA4+SM90 in order to unblock the head dim 512 and page KV for SM90. ⏎  ⏎ vLLm Flash Attention PR dependency: https://github.com/vllm-project/flash-attention/pull/130 ⏎  ⏎ Related Flash Attention PRs: ⏎ - https://github.com/Dao-AILab/flash-attention/pull/2422 ⏎ - https://github.com/Dao-AILab/flash-attention/pull/2415 ⏎  ⏎ ## Test Plan ⏎  ⏎ Test with [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B- …[truncated]

### L3-eb4205fee5  (L3, 2026-04-08, sha eb4205fee52d, PR #37980)
TITLE: [UX] Integrate DeepGEMM into vLLM wheel via CMake (#37980)
SOURCES: dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-0); cmake/external_projects/deepgemm.cmake (+151/-0); docker/Dockerfile (+1/-26); docker/versions.json (+0/-3); setup.py (+29/-0); .gitignore (+3/-0); docs/assets/contributing/dockerfile-stages-dependency.png (+0/-0); tests/kernels/moe/test_silu_mul_fp8_quant_deep_gemm.py (+5/-3); tools/install_deepgemm.sh (+1/-0); vllm/utils/deep_gemm.py (+51/-3); (+2 more)
LABELS: documentation, ready, ci/build, v1, deepseek
BODY: ## Summary ⏎  ⏎ - Bundles DeepGEMM into the vLLM wheel via CMake's FetchContent, so users no longer need to manually run `tools/install_deepgemm.sh` ⏎ - Builds DeepGEMM's pybind11 `_C` extension and vendors the Python package + JIT include headers into `vllm/third_party/deep_gemm/` ⏎ - Removes the separate DeepGEMM build/install steps from the Dockerfile since it's now part of the wheel ⏎ - All `deep_gemm` imports are routed through `vllm/utils/deep_gemm.p …[truncated]

### L3-92fbec391b  (L3, 2026-04-08, sha 92fbec391b0a, PR #38989)
TITLE: [Bug] Fix routing bias dtype for trtllm per-block fp8 moe (#38989)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+5/-0)
LABELS: bug, ready, nvidia
ISSUES: #38931 [Bug]: Deepseek R1 produces incorrect output | #39179 [Bug]: GLM5 on B300 generates garbage output
BODY: ## Purpose ⏎ Fix [[Bug]: Deepseek R1 produces incorrect output](https://github.com/vllm-project/vllm/issues/38931) ⏎ FIX https://github.com/vllm-project/vllm/issues/39179 ⏎ Flashinfer v0.6.7 requires bf16 routing bias dtype for trtllm MoE. We have done this in https://github.com/vllm-project/vllm/pull/38423 for nvfp4 and fp8 per-tensor, but haven't done for fp8 per-block. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎ main: ⏎ ``` ⏎ vllm serve deepseek-ai/DeepSeek …[truncated]

### L3-8332078cfd  (L3, 2026-04-08, sha 8332078cfdbd, PR #39315)
TITLE: [Bugfix] FlashInfer MXINT4 MoE crashes, missing do_finalize (#39315)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_marlin_vs_trtllm_mxint4.py (+93/-0); vllm/model_executor/layers/quantization/utils/flashinfer_mxint4_moe.py (+5/-1)
LABELS: bug, ready, nvidia
ISSUES: #39245 [Bug]: VLLM_USE_FLASHINFER_MOE_INT4 broken
BODY: ## Purpose ⏎  ⏎ In FlashInfer 0.6.4 the interface to the fused MoE backends changed. MXINT4 did not get updated. This means that `VLLM_USE_FLASHINFER_MOE_INT4` has been unusable in vLLM since v0.15.1.  ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/39245 ⏎  ⏎ ## Test Plan ⏎  ⏎ Launches fine with the flag enabled. GSM8k pass, 1k1k runs ~4% faster with this flag enabled ⏎  ⏎ ## Test Result ⏎  ⏎ [details omitted]

### L3-2e98406048  (L3, 2026-04-08, sha 2e9840604879, PR #38865)
TITLE: [Refactor] Improve indexer decode path metadata preparation (#38865)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+127/-75); csrc/sampler.cu (+24/-9); csrc/topk.cu (+4/-2); vllm/model_executor/layers/sparse_attn_indexer.py (+7/-16)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ ### What changed                                                                                                                                                ⏎                                                                                                                                                                 ⏎ This PR refactors the decode path of DeepseekV32IndexerMetadataBuilder and the supporting C++ kernels. No behavior …[truncated]

### L3-9e78555743  (L3, 2026-04-08, sha 9e78555743c1, PR #38950)
TITLE: [Docker] Add fastsafetensors to NVIDIA Dockerfile (#38950)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+3/-1); docker/Dockerfile.rocm (+2/-1); requirements/cuda.txt (+3/-0); requirements/rocm-test.txt (+3/-1); requirements/rocm.txt (+3/-1)
LABELS: rocm, ready, ci/build, nvidia
BODY: ## Summary ⏎ - Add `fastsafetensors = 0.2.2` to `requirements/cuda.txt` to enable faster safetensors model loading via GPU Direct Storage ⏎ - Install `libnuma-dev` in the `vllm-base` Dockerfile stage as a runtime dependency for fastsafetensors (see #20384) ⏎  ⏎ ## Test plan ⏎  ⏎  ⏎ AI assistance was used (Claude). This is not duplicating existing PR #29410 which is about enabling fastsafetensors as the *default* loader — this PR simply makes the package  …[truncated]

### L3-ba4a78eb5d  (L3, 2026-04-08, sha ba4a78eb5d2e, PR #39286)
TITLE: [torch.compile] Allow usage of Opaque Objects in PyTorch 2.11 (#39286)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/compilation/compiler_interface.py (+4/-44); vllm/compilation/wrapper.py (+7/-0); vllm/env_override.py (+54/-0); vllm/utils/torch_utils.py (+1/-1); vllm/v1/worker/gpu_model_runner.py (+3/-0)
LABELS: ready, v1
BODY: We turned this off temporarily in the pt2.11 upgrade because there was some failing tests and I wanted the pt2.11 upgrade to go in first. This PR turns Opaque Objects back on and fixes the tests. ⏎  ⏎ The problem with the tests was that we need to monkeypatch inductor to fix an issue with opaque objects. We were only monkeypatching inductor in the VLLM_COMPILE path (not in the DYNAMO_ONCE/STOCK_TORCH_COMPILE) paths. This PR makes it so that the pat …[truncated]

### L3-f3c7941ec8  (L3, 2026-04-09, sha f3c7941ec8d3, PR #39181)
TITLE: [Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next (#39181)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen2_moe.py (+3/-0); vllm/model_executor/models/qwen3_next.py (+1/-0)
LABELS: bug, ready, qwen
BODY: ## Purpose ⏎ Do not shard shared experts weights when sequence parallel is enabled to fix precision issue for Qwen3.5/Qwen3-Next with EP. At present, when sequence_parallel is enabled, shared experts would apply tensor parallel and do not do all reduce at the end of calculation, which results in the precision issue. ⏎ My environment(collected by `python vllm/collect_env.py`, I tested on vllm 0.18.0, which has no difference with latest main branch i …[truncated]

### L3-2e9034c998  (L3, 2026-04-09, sha 2e9034c998e2, PR #33892)
TITLE: [W8A8 Block Linear Refactor][2/N] Remove W8A8Fp8BlockLinearOp and adopt Fp8 block linear kernel selections. (#33892)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+5/-7); benchmarks/kernels/benchmark_block_fp8_gemm.py (+12/-7); tests/compile/passes/distributed/test_fusion_all_reduce.py (+11/-4); tests/compile/passes/distributed/test_sequence_parallelism.py (+1/-0); tests/compile/passes/test_functionalization.py (+3/-0); tests/compile/passes/test_fusion.py (+49/-51); tests/compile/passes/test_fusion_attn.py (+2/-0); tests/compile/passes/test_mla_attn_quant_fusion.py (+2/-0); tests/compile/passes/test_silu_mul_quant_fusion.py (+23/-10); tests/conftest.py (+3/-2); (+25 more)
LABELS: performance, rocm, ready, nvidia, ready-run-all-tests
BODY: ## Purpose ⏎ This PR  refactors block scaled linear kernel into kernel abstraction. ⏎  ⏎ changes: ⏎ - Introduces `MMLinearKernel` base interface for all linear kernels. ⏎ - Introduces `Params`, `Fp8Params` and `Int8Params`,  classes to access layer params in structured format. ⏎ - Introduces `DynamicMMLinearKernel` which is a type of `MMLinearKernel` with two main properties of base and fallback kernels that are variant of  `MMLinearKernel`. this class switc …[truncated]

### L3-aec18492d0  (L3, 2026-04-09, sha aec18492d0c0, PR #39219)
TITLE: [CI] Fix mypy for `vllm/v1/ops` (#39219)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.triton.prefix_prefill, L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/prefix_prefill.py (+3/-1); vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+22/-22); vllm/v1/attention/ops/vit_attn_wrappers.py (+3/-0); tools/pre_commit/mypy.py (+0/-2)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎  ⏎ Part of the https://github.com/vllm-project/vllm/issues/26533 ⏎  ⏎ `pre-commit run --hook-stage manual mypy-3.13 -a` ⏎  ⏎ Before ⏎  ⏎ ```bash ⏎ vllm/v1/attention/ops/vit_attn_wrappers.py:295: error: Argument 1 to "len" has incompatible type "Any | None"; expected "Sized"  [arg-type] ⏎ vllm/v1/attention/ops/vit_attn_wrappers.py:296: error: Argument 1 to "len" has incompatible type "Any | None"; expected "Sized"  [arg-type] ⏎ vllm/v1/attention …[truncated]

### L3-ef5a226819  (L3, 2026-04-09, sha ef5a226819e0, PR #38935)
TITLE: [PD][HeteroArch]Fix accuracy issue with CPU_ATTN as Decoder and Flash_ATTN as prefiller (#38935)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/platforms/cpu.py (+40/-0); tests/v1/kv_connector/unit/test_nixl_connector.py (+4/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py (+49/-0)
LABELS: intel-gpu, ready, v1, cpu, kv-connector
BODY: ## Purpose ⏎  ⏎ Fix the error reported in #38710  ⏎  ⏎ When using CPU as decoder to work with Hetero Arch Platform (CUDA, Intel GPU, Intel Gaudi ...) as prefiller, there is an accuracy issue due to CPU KV layout need addition `packing` step ⏎  ⏎ In this PR, we fixed this issue by following cpu_attn kv Packing method to pack FlashAttn plain KV to CPU packed KV. ⏎ The work is done in platforms/cpu.py - `pack_kv_cache()` ⏎  ⏎ ## Changes proposed ⏎  ⏎ 1. In Nix …[truncated]

### L3-56e19d7ee2  (L3, 2026-04-09, sha 56e19d7ee206, PR #39353)
TITLE: [Model Runner V2] Fix flex attention kv blocks calculation issue (#39353)
SOURCES: path_core
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+6/-9)
LABELS: ready, v1, mrv2
BODY: ## Purpose ⏎  ⏎ `VLLM_USE_V2_MODEL_RUNNER=1 pytest -s tests/v1/e2e/general/test_async_scheduling.py` ⏎  ⏎ ```bash ⏎ (EngineCore pid=2359877) Process EngineCore: ⏎ (EngineCore pid=2359877) Traceback (most recent call last): ⏎ (EngineCore pid=2359877)   File "/usr/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap ⏎ (EngineCore pid=2359877)     self.run() ⏎ (EngineCore pid=2359877)   File "/usr/lib/python3.12/multiprocessing/process.py", li …[truncated]

### L3-2800706f06  (L3, 2026-04-09, sha 2800706f0649, PR #39129)
TITLE: [Refactor] Move NVFP4 GEMM management into NvFp4LinearKernel (#39129)
SOURCES: corpus:production-kernel-provenance, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/__init__.py (+133/-0); vllm/model_executor/kernels/linear/nvfp4/__init__.py (+12/-0); vllm/model_executor/kernels/linear/nvfp4/base.py (+68/-0); vllm/model_executor/kernels/linear/nvfp4/cutlass.py (+80/-0); vllm/model_executor/kernels/linear/nvfp4/emulation.py (+49/-0); vllm/model_executor/kernels/linear/nvfp4/fbgemm.py (+69/-0); vllm/model_executor/kernels/linear/nvfp4/flashinfer.py (+218/-0); vllm/model_executor/kernels/linear/nvfp4/marlin.py (+57/-0); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py (+4/-19); vllm/model_executor/layers/quantization/modelopt.py (+7/-20); (+1 more)
LABELS: ready, nvidia, quantization
BODY: ## Purpose ⏎  ⏎ Move NVFP4 GEMM management into the kernels/linear/ abstraction, matching the pattern used by FP8/Int8/MP kernels. Each backend (CUTLASS, FlashInfer, Marlin, FBGEMM, emulation) is now its own `NvFp4LinearKernel` subclass. Kernel selection goes through `init_nvfp4_linear_kernel()`.  ⏎  ⏎ The old NvFp4LinearBackend enum, select_nvfp4_linear_backend, convert_to_nvfp4_linear_kernel_format, apply_nvfp4_linear, and Nvfp4LinearOp are all rem …[truncated]

### L3-8a34c5087a  (L3, 2026-04-09, sha 8a34c5087aa7, PR #39122)
TITLE: [ROCm] Remove unnecessary fp8 roundtrip in gather cache NHD dequant (#39122)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+6/-4)
LABELS: rocm, ready, v1, quantization
BODY: In the `cp_mha_gather_cache_kernel` NHD DEQUANT path, after loading fp8 values from the KV cache and multiplying by scale (in float32), the result is cast back to the original fp8 dtype before being stored to the output buffer: ⏎  ⏎ ```python ⏎ k_dtype = k_reg.dtype          # fp8 ⏎ k_reg = (k_reg.to(tl.float32) * k_scale).to(k_dtype)  # float32 -> fp8 roundtrip ⏎ tl.store(key_ptr_offset + col_offsets, k_reg)           # stored to bf16 buffer ⏎ ``` ⏎  ⏎ The outp …[truncated]

### L3-467d3247c3  (L3, 2026-04-09, sha 467d3247c3bd, PR #38856)
TITLE: [LMCache] vLLM Block Allocation Event (#38856)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py (+72/-0)
LABELS: ready, kv-connector
BODY: ## Purpose ⏎  ⏎ Send vLLM block allocation event to LMCache for trace ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-e0613702ad  (L3, 2026-04-09, sha e0613702ade9, PR #36092)
TITLE: [ROCm] Fix AITER ops fake impl and minor bugs (#36092)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/_aiter_ops.py (+8/-4)
LABELS: rocm, ready
BODY: ## Summary ⏎  ⏎ Fix three ROCm-specific bugs in AITER ops and platform code: ⏎  ⏎ - **`_rocm_aiter_fused_topk_fake` returns `None` instead of tuple**: The fake/meta implementation for `rocm_aiter_fused_topk` custom op returned `None` (with `pass` body), while the real implementation returns `(topk_weights, topk_indices)`. This would cause failures during `torch.compile` tracing in FakeTensor mode, since downstream code expects a tuple of tensors. ⏎ - **`sh …[truncated]

### L3-58c0a928c9  (L3, 2026-04-09, sha 58c0a928c99c, PR #38922)
TITLE: [Bugfix] Fix broken explicit unquantized kv cache dtype support (#38922)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: csrc/attention/dtype_fp8.cuh (+16/-0); csrc/quantization/w8a8/fp8/amd/quant_utils.cuh (+13/-14); csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh (+23/-36)
LABELS: bug, ready
BODY: PLEASE FILL IN THE PR DESCRIPTION HERE ENSURING ALL CHECKLIST ITEMS (AT THE BOTTOM) HAVE BEEN CONSIDERED. ⏎  ⏎ ## Purpose ⏎ `--kv-cache-dtype bfloat16` etc is broken: ⏎ ``` ⏎ (Worker_TP1 pid=2915725) ERROR 04-04 00:12:28 [multiproc_executor.py:963]   File "/home/mozf/develop-projects/vllm/vllm/v1/attention/backends/flash_attn.py", line 847, in do_kv_cache_update ⏎ (Worker_TP0 pid=2915724) ERROR 04-04 00:12:28 [multiproc_executor.py:963]            ^^^^ …[truncated]

### L3-447ce22212  (L3, 2026-04-10, sha 447ce22212e5, PR #39471)
TITLE: [GGUF] Support non-standard quant types with prefix (e.g. UD-IQ1_S) (#39471)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/transformers_utils/test_utils.py (+27/-0); vllm/transformers_utils/gguf_utils.py (+38/-3)
LABELS: ready
ISSUES: #39469 [Feature]: Support non-standard GGUF quant type prefixes (e.g. Unsloth Dynamic UD-IQ1_S )
BODY: ## Purpose ⏎ Support non-standard quant types with prefix (e.g. UD-IQ1_S ) ⏎  ⏎ Fixes: #39469 ⏎  ⏎ ## Test Plan ⏎  ⏎ ```bash ⏎ vllm serve unsloth/Qwen3-0.6B-GGUF:UD-IQ1_S --tokenizer Qwen/Qwen3-0.6B ⏎ ``` ⏎  ⏎ ## Test Result ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ --- ⏎ [details omitted]

### L3-f44afef6d6  (L3, 2026-04-10, sha f44afef6d61f, PR #38123)
TITLE: [compile] Allow strings in custom ops without regressing compilation times (#38123)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.common_v1, L3.mla.rocm_aiter_sparse
FILES: vllm/model_executor/layers/attention/attention.py (+18/-9); vllm/model_executor/layers/attention/kv_transfer_utils.py (+2/-1); vllm/model_executor/layers/attention/mla_attention.py (+18/-7); vllm/model_executor/layers/attention/static_sink_attention.py (+12/-4); vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+6/-2); tests/compile/passes/test_rope_kvcache_fusion.py (+2/-1); vllm/compilation/passes/fusion/attn_quant_fusion.py (+182/-50); vllm/compilation/passes/fusion/mla_attn_quant_fusion.py (+166/-36); vllm/compilation/passes/fusion/rope_kvcache_fusion.py (+72/-25); vllm/envs.py (+4/-0); (+6 more)
LABELS: rocm, ready, v1, qwen
BODY: This is a follow-up to https://github.com/vllm-project/vllm/pull/35475 to extend the fix to all custom operators, not just the MOE custom ops. ⏎  ⏎ Previously, string inputs to custom ops would regress compilation times. The problem goes: ⏎ - a transformer model (e.g. llama3-70b) has 80 identical layers ⏎ - we capture a full graph and the split the graph on the attention operations. ⏎ - this produces 81 subgraphs: the middle 79 graphs are all identica …[truncated]

### L3-8f121f7879  (L3, 2026-04-10, sha 8f121f787966, PR #32936)
TITLE: [Model Runner V2] support auto resolve cudagraph mode/sizes based on attn backend (#32936)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/compilation.py (+148/-0); vllm/v1/worker/gpu/attn_utils.py (+39/-5); vllm/v1/worker/gpu/dp_utils.py (+43/-1); vllm/v1/worker/gpu/model_runner.py (+36/-30); vllm/v1/worker/gpu/spec_decode/eagle/speculator.py (+17/-32); vllm/v1/worker/gpu_model_runner.py (+11/-145)
LABELS: ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ A follow-up PR of #32771 and #32820.  ⏎  ⏎ After #32820 we can select any attention backend, but some of them have limitations for CUDA-graph. This PR, like Model-Runner V1, adds a CUDA-graph check that adjusts the cudagraph mode & capture_sizes according to the attention backend. For example, with FLASHINFER + spec decode, a user-specified FULL_AND_PIECEWISE is automatically resolved to PIECEWISE. ⏎  ⏎ ``` ⏎ [WARNING 01-23 20:32:46 [compilatio …[truncated]

### L3-d468322dc1  (L3, 2026-04-10, sha d468322dc1c8, PR #37352)
TITLE: [Kernel][Hardware][AMD] Add TritonW4A16LinearKernel for ROCm (#37352)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/quantization/test_rocm_compressed_tensors_w4a16.py (+35/-0); tests/kernels/quantization/test_triton_w4a16.py (+304/-0); tests/kernels/quantization/test_w4a16_kernel_selection.py (+53/-0); vllm/model_executor/kernels/linear/__init__.py (+5/-0); vllm/model_executor/kernels/linear/mixed_precision/__init__.py (+4/-0); vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py (+430/-0)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Summary ⏎ - Add Triton-based W4A16 GEMM kernel for INT4 weight / FP16 activation inference on AMD MI300 (ROCm) ⏎ - Supports symmetric (uint4b8) and asymmetric (uint4) quantization with grouped scales ⏎ - Registers as preferred mixed-precision linear kernel on ROCm platform ⏎  ⏎ ## AI Disclosure ⏎ This PR includes AI-assisted code (Claude). All code has been reviewed and tested on MI300 hardware. ⏎  ⏎ ## Test Plan ⏎  ⏎ All tests passed on MI300 (gfx942).

### L3-f976e3b98b  (L3, 2026-04-10, sha f976e3b98ba4, PR #37539)
TITLE: [Performance] Remove unnecessary zero-fill of MLA decode output tensor in Aiter backend (#37539)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+1/-1)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Problem ⏎  ⏎ During the MLA decode phase on ROCm with the Aiter backend, a ⏎ vectorized_elementwise GPU kernel is launched to zero-fill the output tensor ⏎ before the MLA kernel overwrites it entirely. This adds unnecessary kernel ⏎ launch latency on the critical decode path. ⏎  ⏎ ## Root ⏎  ⏎ In `rocm_aiter_mla.py`, the output tensor `o` is allocated with `torch.zeros`: ⏎  ⏎ ```python ⏎ o = torch.zeros( ⏎     B, ⏎     self.num_heads, ⏎     self.kv_lora_rank, ⏎     dtype=att …[truncated]

### L3-51cfc0e76c  (L3, 2026-04-10, sha 51cfc0e76c8d, PR #39183)
TITLE: perf(moe): add tuned fused_moe config for RTX PRO 6000 Blackwell Server Edition (#39183)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/configs/E=256,N=384,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8,block_shape=[128,128].json (+147/-0); vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8,block_shape=[128,128].json (+147/-0); vllm/model_executor/layers/fused_moe/configs/E=64,N=1536,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8,block_shape=[128,128].json (+147/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Add tuned fused_moe Triton kernel configs for NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120) with FP8 block-quantized MoE. ⏎  ⏎ Three configs added: ⏎ - **E=256, N=512** (TP=1) -- covers Qwen3.5-35B-A3B-FP8 and similar 256-expert models at TP=1 ⏎ - **E=256, N=384** (TP=4, tensor parallel) -- covers 256-expert models with intermediate_size sharded across 4 GPUs ⏎ - **E=64, N=1536** (EP=4, expert parallel) -- covers MiniMax-M2.5 and  …[truncated]

### L3-c1cc7344fb  (L3, 2026-04-10, sha c1cc7344fb72, PR #38455)
TITLE: [ROCm] Add RDNA 3.5/4 device IDs (gfx1150, gfx1151, gfx1201) (#38455)
SOURCES: release_notes
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+6/-1)
LABELS: rocm, ready
BODY: ## Summary ⏎  ⏎ Adds 3 missing entries to `_ROCM_DEVICE_ID_NAME_MAP` in `vllm/platforms/rocm.py`: ⏎  ⏎ | Device ID | Name | Architecture | Hardware | ⏎ |-----------|------|-------------|----------| ⏎ | `0x150e` | `AMD_Radeon_890M` | gfx1150 | Strix Point APU | ⏎ | `0x1586` | `AMD_Radeon_8060S` | gfx1151 | Strix Halo APU | ⏎ | `0x7550` | `AMD_Radeon_RX9070XT` | gfx1201 | Navi 48 discrete | ⏎  ⏎ Without these entries, `get_device_name()` falls back to `amdsmi["market_ …[truncated]

### L3-e7cfd7c5b9  (L3, 2026-04-10, sha e7cfd7c5b9a1, PR #39450)
TITLE: Add Gemma4 Eagle3 support (#39450)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/speculative.py (+1/-0); vllm/model_executor/models/gemma4.py (+20/-5); vllm/model_executor/models/gemma4_mm.py (+12/-2); vllm/v1/core/single_type_kv_cache_manager.py (+9/-3); vllm/v1/spec_decode/eagle.py (+1/-0)
LABELS: speculative-decoding, ready, v1
BODY: ## Purpose ⏎ Enables Eagle3 style speculative decoding on Gemma4 models. ⏎  ⏎ ## Test Plan ⏎  ⏎ Test model: RedHatAI/gemma-4-31B-it-speculator.eagle3 ⏎ Served locally and verified it ran and produced reasonable acceptance rates / lengths ⏎  ⏎ ## Test Result ⏎ ``` ⏎ 127.0.0.1:55132 - "POST /v1/chat/completions HTTP/1.1" 200 OK ⏎ (APIServer pid=2359019) INFO 04-09 20:54:46 [loggers.py:259] Engine 000: Avg prompt throughput: 4.7 tokens/s, Avg generation throug …[truncated]

### L3-3dd60971de  (L3, 2026-04-10, sha 3dd60971de5a, PR #28443)
TITLE: [feat]: make DCP error msg clearer (#28443)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/cp_utils.py (+5/-4)
LABELS: ready, v1
BODY: Ref: https://github.com/vllm-project/vllm/issues/28407  ⏎  ⏎ Just making the msg clearer. ⏎ cc @lucaswilkinson  ⏎  ⏎ --- ⏎  ⏎ > [!NOTE] ⏎ > Enhances DCP validation during worker init. ⏎ >  ⏎ > - When `dcp_world_size > 1`, validates attention backends by asserting `need_to_return_lse_for_decode` on each layer `impl` and surfaces a clearer error explaining DCP requires softmax LSE during decode and to check `VLLM_ATTENTION_BACKEND`. ⏎ > - No functional changes to KV cac …[truncated]

### L3-49d20346e4  (L3, 2026-04-10, sha 49d20346e411, PR #38794)
TITLE: [Perf] Reduce H2D pageable memory copies (#38794)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+20/-18); vllm/v1/worker/gpu_model_runner.py (+42/-7)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, intel-gpu, speculative-decoding, ready, ci/build
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Fix H2D (Host-to-Device) pageable memory copy bottleneck in  attention Triton kernel, which caused pipeline bubbles between transformer blocks. ⏎  ⏎ **Root Cause**: `mm_prefix_range_tensor()` recomputes and transfers data from CPU to GPU on every attention call, generating redundant H2D pageable copies. ⏎  ⏎ **Solution**: Cache the computed `mm_prefix_range_tensor` result after first computation to eliminate repeated H2D transfers. ⏎  ⏎ **P …[truncated]

### L3-42c6bb4b75  (L3, 2026-04-10, sha 42c6bb4b7510, PR #39509)
TITLE: [ROCm] [AITER] Revert AITER version to v0.1.10.post3 (#39509)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm_base (+1/-1)
LABELS: rocm, ready, ci/build
DEEP_STUDY: deep-study revert record: explicit_rollback of PR(s)  reason=build_or_dependency
BODY: ## Purpose ⏎  ⏎ The AITER v0.1.12 tag is moving https://github.com/ROCm/aiter/issues/2691 . ⏎  ⏎ Moreover, there are many known issues with the initial commit of v0.1.12: ⏎  ⏎ 1. DeepSeek blockscaled gemm `RuntimeError: This GEMM is not supported!` https://github.com/vllm-project/vllm/issues/39485 ⏎  ⏎ 2. https://github.com/vllm-project/vllm/issues/39303 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-e816a8811f  (L3, 2026-04-10, sha e816a8811f2f, PR #39002)
TITLE: [Bugfix] Fix FlashInfer crash with kv_cache_dtype_skip_layers (#39002)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+10/-3); tests/compile/passes/test_fusion_attn.py (+5/-8)
LABELS: bug, ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ Fix FlashInfer attention crashing when `kv_cache_dtype_skip_layers` is used (introduced in #33695). ⏎  ⏎ The metadata builder read the global `cache_config.cache_dtype` (e.g., `"fp8"`) instead of the per-group `kv_cache_spec.kv_quant_mode`, so skipped layers were still treated as fp8.  ⏎  ⏎ ## Test Plan ⏎  ⏎ On B200s (where FlashInfer attention is default)  ⏎ ```bash ⏎ .venv/bin/python -m pytest tests/quantization/test_fp8.py::test_kv_cache …[truncated]

### L3-ecd1ea1363  (L3, 2026-04-11, sha ecd1ea13634e, PR #37045)
TITLE: [Kernel] Porting the TRTLLM minimax_allreduce_rms kernels (#37045)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: .buildkite/test_areas/kernels.yaml (+14/-1); CMakeLists.txt (+2/-0); csrc/minimax_reduce_rms_kernel.cu (+879/-0); csrc/minimax_reduce_rms_kernel.h (+79/-0); csrc/ops.h (+12/-0); csrc/torch_bindings.cpp (+23/-0); tests/kernels/core/test_minimax_reduce_rms.py (+152/-0); vllm/_custom_ops.py (+35/-0); vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py (+340/-0); vllm/compilation/passes/pass_manager.py (+4/-0); (+4 more)
LABELS: rocm, ready, ci/build
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ See: https://github.com/NVIDIA/TensorRT-LLM/pull/12163 ⏎  ⏎ ## Plan ⏎  ⏎  ⏎ ## Test Plan ⏎  ⏎  ⏎ ### Accuracy Verification(https://github.com/vllm-project/vllm/pull/37045/commits/69f231c7b211a9088bf591679fcdced35d4e9199) ⏎  ⏎ - vLLM script(TP4) ⏎ ```bash ⏎ vllm serve MiniMaxAI/MiniMax-M2.5 \ ⏎   --tensor-parallel-size 4 \ ⏎   --tool-call-parser minimax_m2 \ ⏎   --reasoning-parser minimax_m2_append_think  \ ⏎   --served-model-name m25 \ ⏎   --trust-remo …[truncated]

### L3-bd8bd52308  (L3, 2026-04-11, sha bd8bd52308ea, PR #38919)
TITLE: [Bugfix] Runtime driver check for cuMemcpyBatchAsync in swap_blocks_batch (#38919)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+42/-30)
LABELS: bug, ready, nvidia
DEEP_STUDY: deep-study correctness case vllm:bd8bd52308: class=hardware_compiler_specific; symptom=crash_or_exception; introducing=#38460
BODY: Fixes two issues introduced by `swap_blocks_batch` (#38460): ⏎  ⏎ 1. **`undefined symbol: cuMemcpyBatchAsync`** on CUDA drivers < 12.8 (@JaheimLee) — pre-built wheels hard-link the symbol, crashing at `import vllm._C` time on older drivers. ⏎ 2. **Compile error on CUDA 13.0** (@bbrowning, @eugr) — CUDA 13.0 headers `#define cuMemcpyBatchAsync cuMemcpyBatchAsync_v2` (8 params), breaking the original 9-param call. ⏎  ⏎ #38915 fixed problem 2 with compil …[truncated]

### L3-639402f5a2  (L3, 2026-04-12, sha 639402f5a24a, PR #37731)
TITLE: Support FP8 KVCache on XPU (#37731)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fa_utils
FILES: vllm/v1/attention/backends/fa_utils.py (+6/-0); vllm/v1/attention/backends/flash_attn.py (+7/-2); .buildkite/intel_jobs/test-intel.yaml (+1/-0); vllm/_xpu_ops.py (+3/-0)
LABELS: intel-gpu, ready, ci/build, v1
BODY: ## Purpose ⏎  ⏎ Support FP8 KV on XPU. Currently, only support per tensor scale for KVCache and fp8 query will be supported later. ⏎  ⏎ depends on: ⏎  ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ python examples/offline_inference/data_parallel.py --model Qwen/Qwen3-0.6B --enforce-eager --no-enable-expert-parallel --kv-cache-dtype fp8 ⏎  ⏎ DP rank 0, Prompt: 'Hello, my name is', Generated text: ' Josh and I am in the area of what is known as 3D printing' ⏎ DP rank 0, Prompt: ' …[truncated]

### L3-cae984060f  (L3, 2026-04-12, sha cae984060f6f, PR #39201)
TITLE: [compile] Enable AOT compile with batch invariance mode. (#39201)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+1/-4)
LABELS: ready
BODY: Summary: ⏎ Previously we disable AOT compile when batch invariance mode was enabled. ⏎ However, this is never verified to be necessary always. As batch invariance ⏎ feature improves, we should be able to compose it with AOT compile without issue. ⏎  ⏎ So we just remove this override from the envs.py and verify test_batch_invariance.py ⏎ continue to pass. ⏎  ⏎ Test Plan: ⏎ pytest tests/v1/determinism/test_batch_invariance.py -v ⏎ VLLM_BATCH_INVARIANT=1 vllm …[truncated]

### L3-4beeb0689c  (L3, 2026-04-12, sha 4beeb0689cb6, PR #37376)
TITLE:  fused qknorm+rope kernel optimization for SM9.0 (#37376)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/async_util.cuh (+100/-0); csrc/fused_qknorm_rope_kernel.cu (+388/-9); csrc/ops.h (+3/-2); csrc/torch_bindings.cpp (+2/-1); vllm/_custom_ops.py (+2/-0); vllm/compilation/passes/fusion/qk_norm_rope_fusion.py (+1/-0); vllm/compilation/passes/utility/fix_functionalization.py (+1/-0)
LABELS: performance, ready, ci/build, qwen, nvidia
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎ The qknorm+rope fused kernel performs worse than the unfused Triton kernel on H100. ⏎ The main reason is that the 1-head per warp pattern does not perform well with large token batches. ⏎  ⏎ This PR improves the qknorm+rope kernel by dynamically adjusting the workload per warp ⏎ based on the number of tokens in the batch. The threshold value was calculated via offline benchmarking. ⏎  ⏎ Related issue: https://github.com/vllm-project/vllm/is …[truncated]

### L3-ccd0d1d906  (L3, 2026-04-13, sha ccd0d1d9067a, PR #39225)
TITLE: [Bug] Fix rocm sparse attn indexer issue (#39225)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+5/-0)
LABELS: bug, rocm, ready, v1
BODY: ## Purpose ⏎  ⏎ Fixes https://github.com/vllm-project/vllm/pull/39219#discussion_r3047394497 ⏎  ⏎ In `vllm/model_executor/layers/sparse_attn_indexer.py`, we have  ⏎  ⏎ ```py ⏎     attn_metadata = attn_metadata[k_cache_prefix] ⏎     assert isinstance(attn_metadata, DeepseekV32IndexerMetadata) ⏎     slot_mapping = attn_metadata.slot_mapping ⏎     has_decode = attn_metadata.num_decodes > 0 ⏎     has_prefill = attn_metadata.num_prefills > 0 ⏎     num_decode_toke …[truncated]

### L3-d8ddb31644  (L3, 2026-04-13, sha d8ddb316444e, PR #39418)
TITLE: [Bugfix][CT] Fix KV cache scale handling (#39418)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py (+11/-0)
LABELS: bug, documentation, ready, quantization
BODY: ## Summary ⏎ - Fix KV cache scale handling in compressed-tensors quantization ⏎ - Ensures `_k_scale_float` and `_v_scale_float` are properly set for KV cache quantization ⏎  ⏎ Flashinfer uses both `_k_scale` and `_k_scale_float`: ⏎ https://github.com/vllm-project/vllm/blob/6c749399b7dc939a592e30d5964d7bdf255427aa/vllm/v1/attention/backends/flashinfer.py#L1510 ⏎ https://github.com/vllm-project/vllm/blob/6c749399b7dc939a592e30d5964d7bdf255427aa/vllm/v1/a …[truncated]

### L3-ccf90ba784  (L3, 2026-04-13, sha ccf90ba78490, PR #37588)
TITLE: [Model Runner V2] Add full cuda graph support for eagle prefill (#37588)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/model_runner.py (+0/-6); vllm/v1/worker/gpu/spec_decode/eagle/cudagraph.py (+6/-16); vllm/v1/worker/gpu/spec_decode/eagle/speculator.py (+204/-112)
LABELS: ready, v1, nvidia, mrv2
BODY: # Purpose ⏎ FULL cudagraphs are currently only used for the position 1+ drafting phase. In this PR, I apply FULL cudagraphs to the Eagle prefill path as well to reduce the CPU dispatch overhead in `EagleSpeculator.propose`. ⏎  ⏎ # Benchmarks ⏎  ⏎ ## H200 ⏎ I ran an exhaustive set of accuracy and performance benchmarks across several models (Llama3, Qwen3, Mimo, GLM 4.7 Flash), parallelizations (TP, EP, DP), and spec decode types (Eagle-1, Eagle-3, MTP) …[truncated]

### L3-c687bf226a  (L3, 2026-04-13, sha c687bf226a8e, PR #38810)
TITLE: [LMCache][MP] optimize save when mla enabled (#38810)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/__init__.py (+2/-0); vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/multi_process_adapter.py (+78/-14); vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py (+40/-28)
LABELS: ready, kv-connector
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: When MLA is enabled, store or retrieve requests only need to be sent once in multi workers, which can greatly reduce the number of requests in the server. ⏎ The current PR only modifies store requests and will modify retrieve requests in the next request.

### L3-e64b39ea71  (L3, 2026-04-14, sha e64b39ea7114, PR #39119)
TITLE: [ROCm] Align AiterFlashAttentionImpl attn_type check with backend (#39119)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+6/-2)
LABELS: rocm, ready, v1
BODY: `AiterFlashAttentionBackend.supports_attn_type()` correctly rejects `ENCODER_DECODER` with a detailed comment about why cross-attention would produce wrong results (cu_seqlens_k set to decoder query_start_loc + causal=True). ⏎  ⏎ But `AiterFlashAttentionImpl.__init__` accepts both `DECODER` and `ENCODER_DECODER`. If ENCODER_DECODER were ever passed through (e.g., via direct instantiation or a future code path), the impl would silently produce incorre …[truncated]

### L3-240f2636ca  (L3, 2026-04-14, sha 240f2636ca49, PR #39510)
TITLE: [Kernel] Support TRTLLM GEN NVFP4 MoE for non-512-aligned hidden dims via weight padding (#39510)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: tests/quantization/test_trtllm_nvfp4_hidden_dim_padding.py (+62/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+12/-3); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+8/-0); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+42/-0)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Summary ⏎ - Relaxes `_supports_shape` check from `hidden_dim % 512` to always True or TRTLLM NVFP4 MoE kernel - we pad in load time the weights to enable running this kernel (might cause perf gain for some shapes - added a warninig for this) ⏎ - Zero-pads MoE weights to next multiple of 256 (no need 512 alignment) at load time .  ⏎ - Runtime activation padding/output slicing handled by existing MoE runner infrastructure — no custom padding in `ap …[truncated]

### L3-dc8df110bc  (L3, 2026-04-14, sha dc8df110bc8a, PR #39752)
TITLE: add warning when FP8 KV cache misses prefill query quantization (#39752)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+13/-0)
LABELS: ready, deepseek, nvidia
ISSUES: #39751 [Performance]: Add warning log for FP8 KV cache without prefill query quantization
BODY: ## Purpose ⏎  ⏎ add a startup warning log when FP8 KV cache is enabled (`--kv-cache-dtype fp8`) but `use_prefill_query_quantization` is not set, so users are aware of the FP8 prefill attention option. ⏎  ⏎ Resolves #39751  ⏎  ⏎ **Background**: Per PR #31195 by @pavanimajety, FP8 prefill query quantization was intentionally gated behind `use_prefill_query_quantization=true` because it regresses short-sequence performance (~20% at ISL=1024). However, for …[truncated]

### L3-ecf5ff7ce3  (L3, 2026-04-14, sha ecf5ff7ce3ac, PR #36162)
TITLE: [Mamba] Flashinfer selective_state_update (#36162)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/__init__.py (+3/-0); vllm/config/cache.py (+0/-34); vllm/config/mamba.py (+76/-0); vllm/config/vllm.py (+15/-0); vllm/engine/arg_utils.py (+42/-11); vllm/v1/worker/gpu/model_runner.py (+4/-0); vllm/v1/worker/gpu_model_runner.py (+4/-0); tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml (+1/-0); tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml (+1/-0); tests/kernels/mamba/test_ssu_dispatch.py (+92/-0); (+5 more)
LABELS: ready, ci/build, v1
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Add wrapper for FI's selective_state_update kernel, with a runtime dispatcher, connected to a config field, to select between it and the existing triton implementation. ⏎  ⏎ As suggested by @tdoublep in #35753, I've introduced `MambaConfig` in this PR, and a followup (or this, if you'd prefer) could move config fields relevant to Mamba to it. ⏎  ⏎ ## Test Plan ⏎ New test file for the dispatcher's functionality. ⏎ Add tests e2e for Nemotro …[truncated]

### L3-65b9808960  (L3, 2026-04-14, sha 65b9808960d7, PR #39825)
TITLE: [Bugfix] Disable FlashInfer CUTLASS MoE on SM121 (DGX Spark) (#39825)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py (+8/-1)
LABELS: bug, ready, nvidia
BODY: The bf16 unquantized CUTLASS MoE GEMM in flashinfer <= 0.6.7 has no `Relu2` template instantiation and throws `Invalid activation type` at `moe_gemm_template_dispatch.h:1042` when running Nemotron-H (MTP drafter) on SM121. Narrow the SM120 family gate to exact SM120 so the oracle falls back to Triton on SM121. ⏎  ⏎ Fixed upstream by https://github.com/flashinfer-ai/flashinfer/pull/2926 (merged 2026-04-01, not in a stable release yet). The guard can b …[truncated]

### L3-80118853f4  (L3, 2026-04-14, sha 80118853f42a, PR #38061)
TITLE: [MM][Perf][CG] Support ViT full CUDA graph for Qwen3-VL video inference (#38061)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: docs/design/cuda_graphs_multimodal.md (+66/-7); tests/v1/cudagraph/test_encoder_cudagraph.py (+312/-1); vllm/config/compilation.py (+20/-4); vllm/model_executor/models/interfaces.py (+10/-1); vllm/model_executor/models/qwen3_vl.py (+138/-42); vllm/v1/worker/encoder_cudagraph.py (+33/-11); vllm/v1/worker/encoder_cudagraph_defs.py (+4/-2)
LABELS: documentation, performance, ready, v1, multi-modality, qwen, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Following https://github.com/vllm-project/vllm/pull/35963 (only supports image inference), this PR continues to work on it to support video inference for Qwen3-VL. ⏎  ⏎ **TODO:** ⏎  ⏎  ⏎ ## 🤖 AI Summary ⏎  ⏎ Following #35963 (ViT full CUDA graph support for image inference), this PR extends the encoder CUDA graph framework to support **video inference for Qwen3-VL**. Previously, the CUDA graph capture/replay path only handled image inputs ( …[truncated]

### L3-23f3760217  (L3, 2026-04-14, sha 23f376021761, PR #39754)
TITLE: [Bugfix][ROCm]: Allow `gpt_oss_mxfp4` quantization method on rocm (#39754)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+1/-0)
LABELS: bug, rocm, ready, gpt-oss
BODY: ## Purpose ⏎ #39604 added the `gpt_oss_mxfp4` quantization method, but did not add it to the list of `supported_quantization` methods that `rocm.py` maintains. Running `vllm serve --model openai/gpt-oss-120b` thus gives me the following error: ⏎  ⏎ ``` ⏎ VLLM_ROCM_USE_AITER=1 vllm bench latency --model openai/gpt-oss-120b --batch-size 32 --input-len 16 --output-len 16 --num-iters-warmup 1 ⏎  --num-iters 3 --attention-backend ROCM_AITER_UNIFIED_ATTN ⏎ W …[truncated]

### L3-2faad08362  (L3, 2026-04-14, sha 2faad08362ff, PR #39718)
TITLE: [compile] Nest inductor cache under AOT compile dir (#39718)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/compilation/decorators.py (+10/-0)
LABELS: ready, verified
BODY: ## Purpose ⏎  ⏎   With `VLLM_USE_AOT_COMPILE`, the mega-artifact is saved under `~/.cache/vllm/torch_compile_cache/torch_aot_compile/{hash}/rank_{r}_{dp}/model`, but on load Triton and Inductor unpack their on-disk state (fx_graph / aotautograd / triton cubins / generated `.py`) to                                                                                                                          ⏎   `/tmp/torchinductor_$USER/` by default. Unles …[truncated]

### L3-3bfe55a037  (L3, 2026-04-14, sha 3bfe55a03758, PR #39773)
TITLE: [Model Runner V2] Disable piecewise cudagraph mode fallback for eagle draft decodes (#39773)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/spec_decode/eagle/speculator.py (+19/-9)
LABELS: ready, v1, nvidia, mrv2
BODY: # Purpose ⏎ There were two issues present with speculator's cudagraph implementation: ⏎ 1. `last_token_indices` was not being padded before prefill, allowing stale values to remain in the buffer from previous runs and cause OOB errors during a gather op. ⏎ 2. The eagle draft decodes are currently able to run in PIECEWISE mode. This is problematic for at least one attention backend (FlashInfer), where during PIECEWISE decode with single-token batches …[truncated]

### L3-f4b42df048  (L3, 2026-04-14, sha f4b42df04847, PR #38479)
TITLE: [Attention Backend] TurboQuant: 2-bit KV cache compression with 4x capacity (#38479)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:production-kernel-provenance, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.dispatch.registry, L3.platform.cuda_selection
FILES: pyproject.toml (+3/-0); vllm/config/attention.py (+5/-0); vllm/config/cache.py (+4/-0); vllm/engine/arg_utils.py (+38/-0); vllm/model_executor/layers/attention/attention.py (+82/-0); vllm/platforms/cuda.py (+5/-0); vllm/platforms/xpu.py (+6/-0); vllm/utils/torch_utils.py (+4/-0); vllm/v1/attention/backends/registry.py (+1/-0); vllm/v1/attention/backends/turboquant_attn.py (+812/-0); (+17 more)
LABELS: documentation, rocm, intel-gpu, ready, ci/build, v1, nvidia, quantization
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ### Summary ⏎  ⏎ TurboQuant adds **online KV cache compression** to vLLM's v1 attention backend using PolarQuant (WHT rotation + Lloyd-Max scalar quantization) for keys and uniform quantization for values. All quantization happens at store time via fused Triton kernels — no offline calibration, model changes, or weight modifications required. Just set `--kv-cache-dtype turboquant_k8v4`. ⏎  ⏎ ### Compression Presets (Qwen3-4B, head_dim=128) ⏎  ⏎ | Prese …[truncated]

### L3-be0c855ebd  (L3, 2026-04-14, sha be0c855ebd6e, PR #37206)
TITLE: [KV Offload] Unified memory layout for offloading workers (#37206)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_offload/test_cpu_gpu.py (+32/-7); tests/v1/kv_offload/test_shared_offload_region.py (+625/-0); vllm/v1/kv_offload/cpu/shared_offload_region.py (+192/-0); vllm/v1/kv_offload/worker/cpu_gpu.py (+102/-48)
LABELS: ready, v1, kv-connector
BODY: This creates a mmapped memory region that is shared between offloading connector workers. ⏎  ⏎ ## Purpose ⏎ ### Before this PR ⏎  ⏎ Each TP worker independently allocated its own pinned CPU tensors using `torch.zeros(..., pin_memory=True)`. There was no shared memory between workers — each worker had a completely separate, private CPU buffer: ⏎  ⏎ ``` ⏎ Worker 0: [ blk0 | blk1 | blk2 | ... ]  (private pinned memory) ⏎ Worker 1: [ blk0 | blk1 | blk2 | ...  …[truncated]

### L3-799973af4e  (L3, 2026-04-14, sha 799973af4e61, PR #39851)
TITLE: [CI][NIXL] Fix PD CI breakage: pin nixl-cu{12,13} versions (#39851)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/kv_connectors.txt (+2/-0)
LABELS: ready, ci/build, kv-connector
BODY: `nixl-cu12==1.0.1` dropped on PyPI today (19:38 UTC) and ships `nixl_ep` compiled against `libcudart.so.12` — crashes on CUDA 13 CI runners. Our `< 0.10.0` constraint only pins the meta-package, not the backends: ⏎  ⏎ ``` ⏎ [2026-04-14T21:21:43Z]  + nixl==0.9.0 ⏎ [2026-04-14T21:21:43Z]  + nixl-cu12==1.0.1 ⏎ [2026-04-14T21:21:43Z]  + nixl-cu13==1.0.1 ⏎ ``` ⏎  ⏎ - Passing: https://buildkite.com/vllm/ci/builds/61159 (`nixl-cu12==1.0.0`, no `nixl_ep`) ⏎ - Fai …[truncated]

### L3-ed33310552  (L3, 2026-04-15, sha ed333105520c, PR #39837)
TITLE: [KVConnector][LMCache] Propagate cache_salt through MP connector for per-user cache isolation (#39837)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py (+17/-2)
LABELS: ready, kv-connector
BODY: ## Summary ⏎  ⏎ Propagate `request.cache_salt` through the LMCache MP connector to enable per-user cache isolation. This is the vLLM-side counterpart to LMCache/LMCache#3029. ⏎  ⏎ ## Changes ⏎  ⏎ - `LMCacheMPRequestTracker.__init__`: store `request.cache_salt or ""` ⏎ - `LMCacheMPRequestMetadata`: add `cache_salt: str = ""` field ⏎ - `GetStoreMetadata` / `GetRetrieveMetadata`: copy `cache_salt` from tracker ⏎ - `get_num_new_matched_tokens`: pass `cache_sa …[truncated]

### L3-55e1a8e103  (L3, 2026-04-15, sha 55e1a8e1035b, PR #39596)
TITLE: [Mooncake] Fix mixed MLA+Eagle block-size validation (#39596)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_mooncake_connector.py (+45/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py (+20/-3)
LABELS: ready, v1, kv-connector
BODY: ## Purpose ⏎  ⏎ Fix Mooncake registration for mixed MLA + Eagle models. ⏎  ⏎ This branch makes two related changes: ⏎  ⏎ 1. Remove Mooncake's shape-derived registration check: ⏎    `kernel_block_size = cache.shape[-2 if self.use_mla else -3]` ⏎    `assert self.block_size == kernel_block_size` ⏎  ⏎    That check is incorrect for mixed MLA + Eagle because `self.use_mla` is model-global, but Eagle/GQA cache tensors do not follow the MLA shape convention. When …[truncated]

### L3-c77e596e2e  (L3, 2026-04-15, sha c77e596e2eb6, PR #39932)
TITLE: [FlashAttention] Don't overwrite `flash_attn_interface.py` when installing precompiled (#39932)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: setup.py (+11/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ When installing with `VLLM_USE_PRECOMPILED=1`, we should skip copying `flash_attn_interface.py`, which is now source-controlled in vLLM. This mimics similar logic in `vllm_flash_attn.cmake` ⏎  ⏎ ## Test Plan ⏎ 1. Make local changes to `flash_attn_interface.py` ⏎ 2. Run `VLLM_USE_PRECOMPILED=1 uv pip install -e . --no-build-isolation` ⏎  ⏎ ## Test Result ⏎ main: Changes to `flash_attn_interface.py` are overwritten ⏎ PR: Changes to `flash_attn_ …[truncated]

### L3-41488f2acd  (L3, 2026-04-15, sha 41488f2acdc5, PR #39724)
TITLE: [Bugfix][NIXL] Fix `_logical_to_kernel_block_ids` conversion for non-mamba models (#39724)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_nixl_connector_hma.py (+93/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py (+4/-0)
LABELS: bug, ready, v1, kv-connector
BODY: ## Purpose ⏎  ⏎ Error: Assertion failure on Blackwell (FlashInfer, kernel block size < logical block size) ⏎  ⏎ Root cause: #37635 refactored `_logical_to_kernel_block_ids` into `_logical_to_remote_kernel_block_ids` for mamba models but dropped the conversion for non-Mamba models (e.g., pure FA models).  ⏎  ⏎ This PR fix the bug by adding back the `_logical_to_kernel_block_ids` to the non-mamba path.  ⏎  ⏎  ⏎ ## Test Plan ⏎ CI tests on B200.  ⏎  ⏎ ## Test Re …[truncated]

### L3-ac3dac545b  (L3, 2026-04-15, sha ac3dac545b28, PR #38928)
TITLE: [Bugfix][Perf] Indexer upcast WK to BF16 for fusion (#38928)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/deepseek_mtp.py (+14/-11); vllm/model_executor/models/deepseek_v2.py (+70/-53)
LABELS: bug, performance, ready, deepseek
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Purpose ⏎  ⏎ Alternative fix to https://github.com/vllm-project/vllm/pull/38870/ which maintains the fusion. ⏎  ⏎ Performance: ⏎  ⏎ - B200 TP8 FP8 BS1 8k/1k ⏎ - Compares this PR (top), multi-stream (#35968), and baseline (#38870) ⏎ ``` ⏎ Upcast+Fused WK: (Decode): 11.90 ms ⏎ Upcast+Fused WK: (TTFT):   375.0 ms ⏎  ⏎ Multi-Stream:    (Decode): 12.74 ms ⏎ Multi-Stream:    (TTFT):   376.5 ms ⏎  ⏎ Separate:        (Decode): 12.59 ms ⏎ Separate:        (TTFT):   37 …[truncated]

### L3-951dca8019  (L3, 2026-04-15, sha 951dca8019f6, PR #38657)
TITLE: [compile] Invoke split FX graph by codegen. (#38657)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/compilation/backends.py (+23/-2); vllm/compilation/caching.py (+28/-7); vllm/compilation/codegen.py (+155/-0)
LABELS: ready-run-all-tests
BODY: Summary: ⏎  ⏎ This PR reduces inference loop runtime overhead by codegen-ing slightly faster Python ⏎ code instead of invoking the FX graph directly after compilation. ⏎  ⏎ Context: ⏎  ⏎ Today VllmBackend returns a callable as a FX GraphModule with multiple submodules with the following code: ⏎ ``` ⏎ def forward(self, ...): ⏎     self.submod_0(...) ⏎     self.submod_1(...) ⏎     ... ⏎ ``` ⏎ FX graph execution has some overhead due to: ⏎ 1. getattr() calls to fe …[truncated]

### L3-3beb57a238  (L3, 2026-04-15, sha 3beb57a238b8, PR #39676)
TITLE: [XPU] properly handle q_descale on XPU as quant query input not supported (#39676)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+3/-1)
LABELS: intel-gpu, ready, v1
BODY: ## Purpose ⏎ quant query input on XPU is not supported, so we need pass `None` in encoder attention path like full attention. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ VLLM_XPU_FP8_DTYPE=e4m3 VLLM_XPU_USE_SAMPLER_KERNEL=0 VLLM_USE_V1=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python3 -m vllm.entrypoints.openai.api_server --model BAAI/bge-reranker-large --enforce-eager --port 8000 --host 0.0.0.0 --trust-remote-code --gpu-memory-util=0.95 --no …[truncated]

### L3-5f7fab881a  (L3, 2026-04-16, sha 5f7fab881a13, PR #33773)
TITLE: [ROCm][FEAT] Integrate aiter gemm w8a8 ptpc (#33773)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/utils.py (+1/-0); vllm/_aiter_ops.py (+97/-9); vllm/model_executor/kernels/linear/__init__.py (+9/-3); vllm/model_executor/kernels/linear/scaled_mm/aiter.py (+160/-4); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py (+4/-5); vllm/model_executor/layers/quantization/fbgemm_fp8.py (+2/-0); vllm/model_executor/layers/quantization/fp8.py (+2/-2); vllm/model_executor/layers/quantization/modelopt.py (+2/-0); vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py (+2/-0)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ This PR integrates aiter PTPC kernels to optimize FP8 inference performance on AMD ROCm platforms. By leveraging bpreshuffle and ck operators from the aiter library, we provide a significant performance uplift for mainstream Large Language Models compared to the standard torch.scaled_mm baseline. ⏎  ⏎ Key enhancements: ⏎  ⏎ Integrated bpreshuffle (Priority 1): Optimized weight-shuffling kernel for maximum throughput. ⏎  ⏎ Integrated ck (Priority  …[truncated]

### L3-9965f501a8  (L3, 2026-04-16, sha 9965f501a892, PR #39922)
TITLE: [Nixl] Bump Nixl version to 0.10.1 (#39922)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/kv_connectors.txt (+3/-3)
LABELS: ready, ci/build, kv-connector
BODY: As highlighted here https://github.com/vllm-project/vllm/pull/39797#issuecomment-4252561348 we're having some issues updating to nixl 1.0 due to the reported bug listed above. ⏎  ⏎ I am upping the patch version for now, given that @ZhanqiuHu recently addressed issues we've had with nixl-cu13 pinning version on >=0.10 (https://github.com/vllm-project/vllm/issues/36676).

### L3-bf9a5ddb24  (L3, 2026-04-16, sha bf9a5ddb24af, PR #39458)
TITLE: [MLA] Optimize mla indexer prepare uniform decode for MTP > 1 (#39458)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+100/-43)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: # Purpose ⏎ While profiling DeepSeek-V3.2 + NVFP4 with MTP > 1 speculative decoding, I noticed that `_prepare_decode_tensors` in `vllm/v1/attention/backends/mla/indexer.py` was adding per-step overhead from pytorch ops like `repeat_interleave`. This function is called once per decode step to build MLA attention metadata, and the expensive path is taken whenever decode lengths exceed what the kernel natively supports, which is 2. This is the common …[truncated]

### L3-79e799ebbd  (L3, 2026-04-16, sha 79e799ebbd0a, PR #40057)
TITLE: [Bugfix] Temporarily disable B200 fp4 MoE layer tests (#40057)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_moe_layer.py (+8/-0)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ Disable the `modelopt_fp4` tests on B200 for now.  To fix the underlying issue, the `Dockerfile` can be updated to install `libcublas-dev` instead of `libcublas` or we can wait for a newer version of flashinfer, e.g. 0.6.8rc1. ⏎  ⏎ See https://github.com/vllm-project/vllm/issues/39525 ⏎  ⏎ ## Test Plan ⏎  ⏎ Run B200 MoE layer test ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-6b2b7bd0eb  (L3, 2026-04-17, sha 6b2b7bd0ebd4, PR #37332)
TITLE: Add nvfp4 support to reshape_and_cache_flash (#37332)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flash_attn.fork_inline_cmake, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: csrc/cache_kernels.cu (+22/-2); vllm/model_executor/layers/attention/attention.py (+4/-2); vllm/v1/attention/backends/flashinfer.py (+61/-11); CMakeLists.txt (+14/-0); csrc/nvfp4_kv_cache_kernels.cu (+275/-0); tests/kernels/attention/test_cache.py (+99/-21); tests/kernels/quantization/nvfp4_utils.py (+54/-0); vllm/config/cache.py (+1/-0); vllm/utils/torch_utils.py (+122/-12); vllm/v1/kv_cache_interface.py (+27/-3)
LABELS: documentation, ready, ci/build, v1, nvidia, verified
BODY: add nvfp4 support to reshape_and_cache_flash by introducing a new CUDA kernel for reshape_and_cache_flash that quantize and store k, v into kv_cache. The per page layout inside kv_cache is [k_data, k_scale, v_data, v_scale]. The layout is designed such that the last dimension of data and last two dimensions of scale are contiguous, which is a requirement of the kernel. A nvfp4_kv_cache_split_views function is added to split and reshape the kv_cac …[truncated]

### L3-1174723eba  (L3, 2026-04-17, sha 1174723eba17, PR #40060)
TITLE: Fix TURBOQUANT backend selection in cuda.py (#40060)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: docs/design/attention_backends.md (+2/-0); vllm/platforms/cuda.py (+2/-5)
LABELS: bug, documentation, ready, nvidia
BODY: ## Purpose ⏎  ⏎ Added TURBOQUANT to the selection list of attention backends and removed specialized case. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-4c47710bf7  (L3, 2026-04-17, sha 4c47710bf707, PR #40078)
TITLE: [CI/Build] Apply ruff formatter to pass pre-commit (#40078)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py (+3/-5)
LABELS: ready, ci-failure
BODY: ## Purpose ⏎ Fix the pre-commit CI pipeline on the `main` branch. ⏎  ⏎ The pre-commit CI has been failing since commit 29057d3bee1e4a5f84a41eb0cbd2f67b9fa35816, which is currently blocking new PRs from being auto-merged. This PR resolves those issues. ⏎  ⏎ ## Test Plan ⏎ - Verify that the pre-commit CI passes on this PR. ⏎ - (Optional) Successfully run `pre-commit run --all-files` locally. ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ git commit -s -m "Apply ruff formatter …[truncated]

### L3-c0c98b8b9a  (L3, 2026-04-17, sha c0c98b8b9a39, PR #40105)
TITLE: [Bugfix] Add Marlin kernel in block scaled mm kernel selection. (#40105)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/__init__.py (+16/-2)
LABELS: bug, ready
BODY: PLEASE FILL IN THE PR DESCRIPTION HERE ENSURING ALL CHECKLIST ITEMS (AT THE BOTTOM) HAVE BEEN CONSIDERED. ⏎  ⏎ ## Purpose ⏎ This PR fixes the issue in https://github.com/vllm-project/vllm/issues/39610 ⏎  ⏎ ## Test Plan ⏎ On A100  ⏎ ``` ⏎ vllm serve Qwen/Qwen3.5-27B-FP8 -tp 2  ⏎ ``` ⏎ ``` ⏎ lm_eval --model local-completions \ ⏎ --tasks gsm8k \ ⏎ --model_args model=Qwen/Qwen3.5-27B-FP8,base_url=http://localhost:8000/v1/completions \ ⏎ --trust_remote_code \ ⏎ --nu …[truncated]

### L3-6ef1efd51f  (L3, 2026-04-17, sha 6ef1efd51f11, PR #39953)
TITLE: [ROCm] Fix TurboQuant on ROCm: backend routing, flash-attn compat, int64 overflow (#39953)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+1/-0); vllm/v1/attention/backends/turboquant_attn.py (+3/-11); vllm/v1/attention/ops/triton_turboquant_decode.py (+6/-6); vllm/v1/attention/ops/triton_turboquant_store.py (+12/-6)
LABELS: bug, rocm, ready, v1
BODY: ## Purpose ⏎ Route turboquant_* kv-cache-dtype to TurboQuantBackend on ROCm ⏎ Wrap flash_attn_varlen_func on ROCm to handle out= keyword argument (API mismatch with upstream flash-attn) ⏎ Cast block indices and slot offsets to int64 in Triton TQ decode/store kernels to prevent int32 overflow on large KV caches ⏎  ⏎ ## Tests Done ⏎ Verified with GPT-OSS-120B on AMD MI300X (TP=2) at C=2, 4, 8, 64 with 8K input / 1K output — zero failures ⏎ Unit tests (tes …[truncated]

### L3-a8bffaa133  (L3, 2026-04-17, sha a8bffaa13337, PR #37463)
TITLE: [Kernel] Add MXFP4 W4A4 CUTLASS MoE kernel for SM100 (#37463)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: .buildkite/test_areas/kernels.yaml (+1/-0); CMakeLists.txt (+3/-1); csrc/libtorch_stable/ops.h (+23/-0); csrc/libtorch_stable/quantization/fp4/mxfp4_blockwise_moe_kernel.cu (+468/-0); csrc/libtorch_stable/quantization/fp4/mxfp4_experts_quant.cu (+422/-0); csrc/libtorch_stable/torch_bindings.cpp (+22/-0); tests/kernels/moe/test_mxfp4_moe.py (+248/-0); vllm/_custom_ops.py (+135/-0); vllm/model_executor/layers/fused_moe/config.py (+19/-0); vllm/model_executor/layers/fused_moe/cutlass_moe.py (+295/-0); (+1 more)
LABELS: ready, ci/build, nvidia, quantization
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Adds an MXFP4 W4A4 MoE kernel path using CUTLASS on SM100f Blackwell. This enables W4A4 inference for MXFP4-quantized MoE models using the `mx_float4_t` type with E8M0 block-32 scales, distinct from the existing NVFP4 path which uses `nv_float4_t` with E4M3 block-16 scales and a global scale. ⏎  ⏎ This PR includes a new grouped GEMM kernel `mxfp4_blockwise_moe_kernel.cu` and activation quantization kernel (with fused SiLU+MuL) `mxfp4_expe …[truncated]

### L3-251c18d1f8  (L3, 2026-04-17, sha 251c18d1f89c, PR #39957)
TITLE: skip fp8e4b15 on xpu (#39957)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/triton_turboquant_decode.py (+9/-3); tests/quantization/test_turboquant.py (+17/-15)
LABELS: intel-gpu, ready, v1
BODY: ## Purpose ⏎  ⏎ re-land https://github.com/vllm-project/vllm/pull/38479/changes/a8d08c6b29b4fb9c600ad5f654183e1349747b02#diff-318fa76eea4526501ec8f06665dbcafb711753b6ab37735cf816992ca6768094R22-R23 ⏎  ⏎ ## Test Plan ⏎  ⏎ allow test on xpu ⏎  ⏎ will follow https://github.com/vllm-project/vllm/issues/39158 for refactor later. ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-87518c3027  (L3, 2026-04-18, sha 87518c302797, PR #39967)
TITLE: [ZenCPU] AMD Zen CPU Backend with supported dtypes via zentorch weekly (#39967)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: setup.py (+3/-1); vllm/platforms/zen_cpu.py (+8/-0)
LABELS: rocm, ready, ci/build
BODY: ## Purpose ⏎ Update the AMD Zen CPU platform (`ZenCpuPlatform`) to correctly declare supported dtypes and switch the zentorch dependency to the weekly release channel. ⏎ This PR fixes the two minor issues introduced in https://github.com/vllm-project/vllm/issues/35089 ⏎ 1. **Declare `supported_dtypes` for Zen CPU**: AMD Zen CPUs do not support float16 compute natively. The base `CpuPlatform` includes `torch.float16` in its `supported_dtypes`, which  …[truncated]

### L3-b5f6c5f834  (L3, 2026-04-18, sha b5f6c5f8343d, PR #39909)
TITLE: Added general ND x ND matmul and unit test for it (#39909)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/determinism/test_matmul_batch_invariant.py (+105/-0); vllm/model_executor/layers/batch_invariant.py (+24/-31)
LABELS: ready, v1
ISSUES: #38892 [Bug]: matmul_batch_invariant does not handle all torch.matmul dimension combinations (4D x 3D for gemma4-E2B)
BODY: ## Purpose ⏎  ⏎ Add a general ND x ND batch invariant matmul branch. The Gemma4-E2B model has a 4D x 3D matmul, which is not covered by the batch invariant implementation at the moment. This PR fixes: #38892 and the general implementation ensures all shapes supported by torch.matmul are now supported by batch invariant matmul too. ⏎  ⏎ ## Test Plan ⏎  ⏎ Created a new test file which tests first the implementation with against the torch.matmul implement …[truncated]

### L3-ed0622e3a8  (L3, 2026-04-18, sha ed0622e3a809, PR #40194)
TITLE: [Attention] TurboQuant: remove redundant random signs, add prior art attribution (#40194)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/attention.py (+1/-18); vllm/v1/attention/backends/turboquant_attn.py (+9/-13); tests/quantization/test_turboquant.py (+14/-42); vllm/model_executor/layers/quantization/turboquant/__init__.py (+12/-5); vllm/model_executor/layers/quantization/turboquant/config.py (+17/-7); vllm/model_executor/layers/quantization/turboquant/quantizer.py (+0/-18)
LABELS: ready, v1, quantization
BODY: ## Summary ⏎  ⏎ - **Remove per-layer random sign flips** from the Hadamard rotation. The Hadamard signs should have no effect on Lloyd-Max quantization quality; first, because the quantizer is symmetric around zero, and second because the randomness will add variance. In this PR, all layers now share the same pure Hadamard matrix, removing the `_tq_signs` buffer, `generate_wht_signs()`, the per-layer seed logic, and the `seed` field from `TurboQuan …[truncated]

### L3-4b7f5ea1a0  (L3, 2026-04-19, sha 4b7f5ea1a06c, PR #40010)
TITLE: [KV Connector] Allow metrics of multiple connectors of same types in multi connector. (#40010)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py (+11/-1)
LABELS: bug, ready, kv-connector
BODY: ## Purpose ⏎  ⏎   Fix stats collection and Prometheus metric registration in `MultiConnector` when multiple connectors of the same class are configured (e.g., two `OffloadingConnector` instances ⏎   for different transfer types). ⏎  ⏎   **Two bugs fixed:** ⏎  ⏎   1. **Stats overwrite** — `get_kv_connector_stats()` keyed stats by class name, so a second connector of the same type silently overwrote the first's stats. Fixed by aggregating ⏎   into the exis …[truncated]

### L3-45232a454e  (L3, 2026-04-19, sha 45232a454e4c, PR #39083)
TITLE: [FEAT] [Perf] [Gemma4] Fused Gemma4 Routing Function Triton (#39083)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: pyproject.toml (+1/-0); tests/kernels/moe/test_gemma4router.py (+57/-0); vllm/model_executor/models/gemma4.py (+122/-16)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Improve the performance of Gemma4 by introducing triton fused routing function. ⏎  ⏎ The custom routing function introduces many synchronizations point and read write to global memory. ⏎  ⏎ Moreover, the custom routing function is not captured under torch compile. ⏎  ⏎ ## Test Plan ⏎  ⏎ Perform microbenchmark on the triton kernel versus torch ⏎  ⏎ Perform end to end testing on A100, H100, MI300X, and B60 (TP4, FP8 online quant): Accuracy and  …[truncated]

### L3-d1135a5087  (L3, 2026-04-19, sha d1135a508705, PR #40273)
TITLE: Fix MoE backend selection for LoRA (unquantized MoE) (#40273)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_unquantized_backend_selection.py (+85/-0); vllm/model_executor/layers/fused_moe/oracle/unquantized.py (+5/-0)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ When using LoRA adapters with Nemotron Nano BF16: ⏎ https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 ⏎  ⏎ The following error was raised: ⏎  ⏎ ``` ⏎ Using FlashInfer CUTLASS Unquantized MoE backend out of potential backends: ['FlashInfer TRTLLM', 'FlashInfer CUTLASS', 'TRITON', 'BATCHED_TRITON']. ⏎ ... ⏎ File "/my_home/workspace/my_vllm/vllm/lora/layers/fused_moe.py", line 164, in _inject_lora_into_fused_moe ⏎ assert isinsta …[truncated]

### L3-f150107efd  (L3, 2026-04-19, sha f150107efd15, PR #39120)
TITLE: [ROCm] Fix cu_seqlens_q off-by-one in AITER FA speculative decode path (#39120)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+2/-2)
LABELS: rocm, ready, v1
BODY: In the speculative decode path of `AiterFlashAttentionImpl` (when `decode_max_query_len > 1`), `cu_seqlens_q` is sliced as `query_start_loc[:num_decodes]`, giving `num_decodes` elements. But `cu_seqlens_q` for `unified_attention` is a cumulative length array that needs `num_seqs + 1` entries (including the leading 0). ⏎  ⏎ The correct pattern is already used in the fallback path 33 lines later (line 1228-1229), which correctly does `[:num_decodes + 1 …[truncated]

### L3-7243e02aa1  (L3, 2026-04-20, sha 7243e02aa1c6, PR #39616)
TITLE: [ROCm][Feature] Enable AITER MLA attention backend to work with Eagle3 speculative decoding on ROCm (#39616)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+129/-57); docs/design/attention_backends.md (+1/-1)
LABELS: documentation, rocm, ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary                                                                                       ⏎                                                                                                    ⏎   Enable AITER MLA attention backend to work with Eagle3 speculative decoding on ROCm.             ⏎                                                                                                    ⏎   AITER MLA is the fastest MLA attention backend on  …[truncated]

### L3-3a30eaa1d7  (L3, 2026-04-20, sha 3a30eaa1d7b6, PR #37712)
TITLE: Properly enable wvSplitK fp8 path for RDNA (#37712)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/scaled_mm/rocm.py (+3/-3)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-50dd4cb427  (L3, 2026-04-20, sha 50dd4cb42726, PR #36276)
TITLE: [EPLB] Add nixl-based eplb communicator (#36276)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/serving/expert_parallel_deployment.md (+1/-0); tests/distributed/test_eplb_execute.py (+14/-3); vllm/config/parallel.py (+2/-1); vllm/distributed/eplb/eplb_communicator.py (+443/-3); vllm/distributed/eplb/rebalance_execute.py (+20/-14); vllm/distributed/kv_transfer/kv_connector/v1/nixl/stats.py (+1/-3); vllm/distributed/kv_transfer/kv_connector/v1/nixl/utils.py (+0/-46); vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py (+1/-2); vllm/distributed/nixl_utils.py (+54/-0); vllm/v1/worker/gpu_model_runner.py (+20/-8)
LABELS: documentation, ready, ci/build, v1, kv-connector
BODY: ## Purpose ⏎ Add Nixl EPLB communicator as another alternative EPLB communicator that allows avoiding hangs in sync and async EPLB caused by NCCL. ⏎  ⏎ ## Validation ⏎ EPLB sync ⏎ ``` ⏎ vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --no-enable-prefix-caching -tp 1 -dp 4 --max-num-seqs 256 --enable-expert-parallel --port 8000 --gpu-memory-utilization 0.8 --max-model-len 4096 --async-scheduling --disable-nccl-for-dp-synchronization --all2all-backend al …[truncated]

### L3-191e3fdaa1  (L3, 2026-04-20, sha 191e3fdaa1fd, PR #39959)
TITLE: Update flashinfer to 0.6.8 (#39959)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: docker/Dockerfile (+3/-24); docker/Dockerfile.nightly_torch (+2/-5); docker/versions.json (+1/-1); requirements/cuda.txt (+2/-2); vllm/utils/flashinfer.py (+3/-4); docs/design/attention_backends.md (+1/-1); tests/kernels/attention/test_use_trtllm_attention.py (+1/-1); tests/kernels/moe/test_ocp_mx_moe.py (+17/-15); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py (+1/-8)
LABELS: documentation, ready, ci/build, nvidia, verified
BODY: ## Purpose ⏎  ⏎ Update flashinfer to 0.6.8 and re-enable FlashInfer CUTLASS MoE on SM121. ⏎ `download_trtllm_headers` was replaced with download-cubin upstream. ⏎  ⏎ See also: https://github.com/vllm-project/vllm/pull/39825 ⏎  ⏎ ## Test Plan ⏎  ⏎ No crashes in EngineCore in `flashinfer_cutlass_fused_moe`.

### L3-b42e878ec0  (L3, 2026-04-20, sha b42e878ec018, PR #40053)
TITLE: [Bug] Fix dcp error message (#40053)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/cp_utils.py (+1/-1)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ `VLLM_ATTENTION_BACKEND` is deprecated long time ago, update the error message here

### L3-2390caf157  (L3, 2026-04-20, sha 2390caf157cb, PR #38371)
TITLE: Enable building MoRI with AMD AINIC stack (#38371)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm (+53/-3); docker/Dockerfile.rocm_base (+1/-1)
LABELS: rocm, ready, ci/build
BODY: ## Summary ⏎  ⏎  ⏎ Updates the ROCm  image build (`docker/Dockerfile.rocm` and `docker/Dockerfile.rocm_base`) so MORI can be built with an optional **AMD AINIC (Pensando / ionic)** or **BNXT (broadcom)" NIC stack, following the same approach as SGLang’s [docker/rocm.Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile#L366) MORI section. ⏎  ⏎ Also bumps the default **MORI** git pin to `v1.1.0`, adds **MoriIO proxy** Pytho …[truncated]

### L3-fb5635d3f9  (L3, 2026-04-20, sha fb5635d3f906, PR #39242)
TITLE: [ROCm] Add MLA dual RMS norm fusion (Q, KV) pass for DeepSeek/Kimi-K2 (#39242)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:kernel-correctness-cases(introducing), corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/compilation/passes/fusion/rocm_aiter_fusion.py (+106/-1); vllm/compilation/passes/pass_manager.py (+4/-0); vllm/config/compilation.py (+9/-0); vllm/config/vllm.py (+11/-0); docs/design/fusions.md (+40/-0); docs/design/optimization_levels.md (+1/-0); tests/compile/passes/test_fuse_mla_dual_rms_norm.py (+148/-0); vllm/_aiter_ops.py (+42/-0)
LABELS: documentation, rocm, ready, deepseek
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:21b086d0aa (fix PR 40386) || deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ Add a `torch.inductor` post-grad pattern-matcher pass that fuses the paired `q_a_layernorm` and `kv_a_layernorm` RMS norm operations in DeepSeek-V3 / Kimi-K2 MLA attention into a single `fused_qk_rmsnorm` HIP kernel call via AITer. ⏎  ⏎ In the unfused graph, each MLA layer runs two separate `rms_norm` calls on the q and kv compressed latents: ⏎  ⏎ ``` ⏎ qkv_lora = gemm(hidden) ⏎ q_c, kv_lora = split(qkv_lora, [1536, 576]) ⏎ kv_c, k_pe   =  …[truncated]

### L3-21b086d0aa  (L3, 2026-04-20, sha 21b086d0aaea, PR #40386)
TITLE: [ROCm] Hotfix: guard MLA dual RMS norm fusion against older AITer versions (#40386)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+3/-3); vllm/_aiter_ops.py (+24/-1)
LABELS: rocm, ready
DEEP_STUDY: deep-study correctness case vllm:21b086d0aa: class=hardware_compiler_specific; symptom=crash_or_exception; introducing=#39242
BODY: The fuse_mla_dual_rms_norm pass (PR #39242) requires aiter.ops.fused_qk_norm_rope_cache_quant.fused_qk_rmsnorm, which was added in AITer PR #2442. The upstream Dockerfile.rocm_base pins aiter v0.1.10.post3 which does not include this kernel, causing an ImportError at runtime when the pass is auto-enabled at O1+. Add a cached availability probe (_check_aiter_fused_qk_rmsnorm) that checks whether the kernel exists in the installed aiter. The enable …[truncated]

### L3-47fcb8ca68  (L3, 2026-04-20, sha 47fcb8ca68c1, PR #39733)
TITLE: [Core] Pass donate_graph_module=True to standalone_compile (#39733)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/compilation/compiler_interface.py (+3/-0)
LABELS: ready
BODY: ## Purpose ⏎ Pass donate_graph_module=True to standalone_compile for PyTorch >= 2.13dev. ⏎ This allows standalone_compile to take ownership of the graph module, ⏎ avoiding unnecessary copies. ⏎  ⏎ ## Test Plan ⏎ Compile benchmark: 1 repeat, Llama-3-70B, TP=4, cold compile. ⏎  ⏎ ## Test Result ⏎ No regression: ⏎ - Current: 29.1s (n=1) ⏎ - Parent:  29.4s (n=1) ⏎  ⏎ Expected improvement is ~0.1s which is hard to measure with the ⏎ end-to-end benchmark. Confirmed the copy elimi …[truncated]

### L3-8b1f3bebca  (L3, 2026-04-20, sha 8b1f3bebcab8, PR #39843)
TITLE: [LMCache MP Connector] Add num_lmcache_extra_cached_token in KVTransferParams (#39843)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py (+20/-1)
LABELS: ready, kv-connector
BODY: ## Purpose ⏎ Add `num_lmcache_extra_cached_token` to the LMCache MP mode's kv_transfer_params in request and response  ⏎  ⏎ ## Test Plan ⏎  ⏎   ⏎ ## Test Result ⏎ Pending ⏎  ⏎ --- ⏎ [details omitted]

### L3-e06de7f005  (L3, 2026-04-20, sha e06de7f0057f, PR #39627)
TITLE: [XPU] enable triton attention test on XPU by removing cuda device binding (#39627)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_triton_decode_attention.py (+31/-24); tests/kernels/attention/test_triton_prefill_attention.py (+17/-10); tests/kernels/attention/test_triton_unified_attention.py (+4/-2)
LABELS: intel-gpu, ready, nvidia
BODY: ## Purpose ⏎ Triton attention tests can't run on XPU as device bound to cuda. This PR remove the binding by using `current_platform.device_type`. ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-18563f2072  (L3, 2026-04-21, sha 18563f20727a, PR #40086)
TITLE: [Misc] Reduce attention logging levels (#40086)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/attention.py (+1/-1)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ <img width="1663" height="770" alt="image" src="https://github.com/user-attachments/assets/82eceb78-872d-41fa-a4ee-0d8fff2d259d" /> ⏎ Not important messages for default case. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted] ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ <https://docs.vllm.ai/en/latest/contributing>** (anything written below this line will be removed by GitHub Actions)

### L3-6d85b36a9f  (L3, 2026-04-21, sha 6d85b36a9fa2, PR #40032)
TITLE: Revert #38730 and #38791  (#40032)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+1/-0); tests/kernels/attention/test_use_trtllm_attention.py (+2/-2); tools/pre_commit/generate_attention_backend_docs.py (+5/-9)
LABELS: documentation, ready, nvidia
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 38730;38791 reason=build_or_dependency
BODY: Revert #38730 and connected #38791.  ⏎  ⏎ #38730 is workaround for https://github.com/flashinfer-ai/flashinfer/issues/2939. https://github.com/flashinfer-ai/flashinfer/issues/2939 is fixed.  ⏎  ⏎ #38730 was reverted in v0.19.1 https://github.com/vllm-project/vllm/commit/cfad6a509c37e863733805a5b351c076e3b0c0db ⏎  ⏎ But by some reason wasn't reverted in main.

### L3-936e0b79aa  (L3, 2026-04-21, sha 936e0b79aa93, PR #40445)
TITLE: [MM][CG] Optimize default `max_frames_per_batch` auto-infer for ViT CUDA graph video inference (#40445)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/design/cuda_graphs_multimodal.md (+2/-1); vllm/config/compilation.py (+9/-6); vllm/model_executor/models/interfaces.py (+6/-0); vllm/model_executor/models/qwen3_vl.py (+10/-0); vllm/multimodal/registry.py (+3/-0); vllm/v1/worker/encoder_cudagraph.py (+16/-8)
LABELS: documentation, ready, v1, multi-modality, qwen, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ The previous auto-inference logic for `encoder_cudagraph_max_frames_per_batch` in the ViT encoder CUDA graph path used a hardcoded formula of `max_batch_size * 2`, which was both inaccurate and noted as a TODO. For models like Qwen3-VL, the actual max frames per video can be far greater than 2, causing the pre-allocated `cu_seqlens` buffer to be either too small (risking OOM or incorrect behavior) or unnecessarily large. ⏎  ⏎ This PR  …[truncated]

### L3-f95c11a848  (L3, 2026-04-21, sha f95c11a848c7, PR #39703)
TITLE: [Feat] dflash support for ROCm (#39703)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+75/-29)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ There's already dflash support for nvidia https://github.com/vllm-project/vllm/pull/36847. This PR is for dflash support for ROCm ⏎  ⏎ ## Benchmark ⏎  ⏎ Model: `Qwen/Qwen3-8B` ⏎ Drafter: `z-lab/Qwen3-8B-DFlash-b16` ⏎ Num prompts per run: `8` ⏎ Max new tokens per request: `128` ⏎ Concurrencies: `[32, 16, 8, 1]` ⏎ Datasets (official dflash benchmark): `['gsm8k', 'math500', 'humaneval', 'mbpp', 'mt-bench']` ⏎ GPU: `AMD MI300` ⏎  ⏎ I measured the f …[truncated]

### L3-28c222157b  (L3, 2026-04-21, sha 28c222157bbf, PR #39391)
TITLE: fix: clamp NaN/Inf in topk_softmax to prevent duplicate expert IDs (#39391)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: csrc/moe/topk_softmax_kernels.cu (+19/-2); tests/kernels/moe/test_fused_topk.py (+67/-0)
LABELS: ready, verified
ISSUES: #39077 [Bug]: qwen 3.5 crash with mtp | #39244 [Bug]: CUDA illegal memory access with FlashInfer MoE FP8 on Qwen3.5-397B (num_tokens > 256)
BODY: ### Purpose ⏎ Fix https://github.com/vllm-project/vllm/issues/39244 ⏎  ⏎ Fix CUDA illegal memory access crash when serving MoE models (e.g., Qwen3.5-397B-A17B-FP8) with FlashInfer CUTLASS MoE and CUDA graphs at high concurrency on H200. ⏎  ⏎ CUDA graph replay pads the batch to the nearest capture size. Padded tokens have degenerate hidden states that produce NaN gating logits. The `topkGating` kernel's softmax outputs all-NaN, and the argmax loop pick …[truncated]

### L3-66cc3fa559  (L3, 2026-04-21, sha 66cc3fa559d7, PR #39937)
TITLE: [Model Runner V2] Multiple prompt logprobs support (#39937)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/sample/prompt_logprob.py (+48/-18)
LABELS: ready, v1, mrv2
BODY: ## Purpose ⏎  ⏎ Part of the https://github.com/vllm-project/vllm/pull/39337 ⏎  ⏎ Multiple prompt logprobs support ⏎  ⏎ ## Test ⏎  ⏎ `VLLM_USE_V2_MODEL_RUNNER=1 pytest tests/v1/sample/test_logprobs.py -k prompt_logprobs_with_chunking_and_preemption` ⏎  ⏎ Originnaly ⏎  ⏎ ```bash ⏎ __________________________________ test_prompt_logprobs_with_chunking_and_preemption ___________________________________ ⏎  ⏎     def test_prompt_logprobs_with_chunking_and_preemption() …[truncated]

### L3-9a6a66f3b8  (L3, 2026-04-21, sha 9a6a66f3b837, PR #39833)
TITLE: [MRv2]fix: model accuracy regression caused by reusing the stale last_sampled_tokens and draft_tokens (#39833)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/states.py (+12/-0)
LABELS: bug, ready, v1, mrv2
BODY: ## Summary ⏎ - Fix ~13% GSM8K accuracy regression in PD disaggregation caused by stale `last_sampled_tokens` and `draft_tokens` leaking between requests when V2 model runner reuses request state slots ⏎ - In `add_request()`, initialize `last_sampled_tokens` to the last computed token and zero out `draft_tokens` ⏎  ⏎ ## Root Cause ⏎ In PD disagg, the decode side receives requests with `num_computed_tokens == prefill_len` (fully prefilled). The first it …[truncated]

### L3-9f39b380d0  (L3, 2026-04-21, sha 9f39b380d070, PR #39546)
TITLE: [Bugfix] Fix spec decode test failures on Blackwell (SM100+) (#39546)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+18/-5); .buildkite/test_areas/spec_decode.yaml (+34/-0)
LABELS: bug, ready, ci/build, v1, nvidia
BODY: Summary                                ⏎                                                                                                                                                                                                                                                      ⏎   Fix speculative decoding on Blackwell (SM100+) GPUs by eliminating a stale GPU buffer read and removing unnecessary CPU→GPU synchronization in the FlashInfer atte …[truncated]

### L3-16688b26a6  (L3, 2026-04-21, sha 16688b26a6fb, PR #40413)
TITLE: [Perf] Optimize batch invariant with fused rms norm, 2.1% E2E latency improvement (#40413)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/determinism/test_rms_norm_batch_invariant.py (+88/-1); vllm/_custom_ops.py (+1/-0); vllm/model_executor/layers/layernorm.py (+0/-4)
LABELS: ready, v1
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:b6cbba8bc8 (fix PR 48391) || deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ `fused_add_rms_norm` is already batch invariant, we don't need to call the triton kernel ⏎  ⏎ ## Test ⏎  ⏎ Acc covered in unit test ⏎  ⏎ `VLLM_BATCH_INVARIANT=1 vllm bench latency   --model=RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8   --attention-backend=TRITON_ATTN   -cc.custom_ops+=+rms_norm   -cc.pass_config.fuse_norm_quant=False` ⏎  ⏎ ```bash ⏎ # Now ⏎ Avg latency: 0.5136070239978532 seconds ⏎ 10% percentile latency: 0.5083566858433187 second …[truncated]

### L3-3951d3eacd  (L3, 2026-04-21, sha 3951d3eacde3, PR #40159)
TITLE: [MyPy] Enable mypy for `vllm/model_executor/layers/` (#40159)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/attention.py (+15/-6); vllm/model_executor/layers/attention/chunked_local_attention.py (+1/-1); vllm/model_executor/layers/attention/cross_attention.py (+9/-4); vllm/model_executor/layers/attention/encoder_only_attention.py (+2/-2); vllm/model_executor/layers/attention/mla_attention.py (+17/-10); tools/pre_commit/mypy.py (+0/-1); vllm/model_executor/layers/activation.py (+15/-12); vllm/model_executor/layers/fused_moe/all2all_utils.py (+7/-4); vllm/model_executor/layers/fused_moe/config.py (+2/-1); vllm/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py (+3/-3); (+18 more)
LABELS: ready, nvidia
BODY: Part of https://github.com/vllm-project/vllm/issues/26533 ⏎  ⏎ `$ pre-commit run -a --hook-stage manual mypy-3.10` ⏎  ⏎ Before: ⏎  ⏎ ```bash ⏎ vllm/model_executor/layers/activation.py:671: error: "warning_once" of "_VllmLogger" does not return a value (it only ever returns None)  [func-returns-value] ⏎ vllm/model_executor/layers/activation.py:706: error: Need type annotation for "_ACTIVATION_AND_MUL_REGISTRY"  [var-annotated] ⏎ vllm/model_executor/layers/ …[truncated]

### L3-4506319a28  (L3, 2026-04-21, sha 4506319a2862, PR #38877)
TITLE: [compile] mla + group fp8 fusion (#38877)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.dispatch.abstract_interface
FILES: vllm/compilation/passes/fusion/mla_attn_quant_fusion.py (+259/-18); vllm/model_executor/layers/attention/mla_attention.py (+96/-10); vllm/v1/attention/backend.py (+14/-2); docs/design/fusions.md (+2/-2); tests/compile/fusions_e2e/conftest.py (+16/-0); tests/compile/fusions_e2e/models.py (+19/-5); tests/compile/fusions_e2e/test_tp1_quant.py (+3/-2); tests/compile/fusions_e2e/test_tp2_ar_rms.py (+3/-2); tests/compile/passes/test_mla_attn_quant_fusion.py (+99/-18)
LABELS: documentation, ready, v1, verified
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ Completes phase 1 (group fp8) of #35792  ⏎  ⏎ - Adds `MLAAttnFp8GroupQuantPattern` for DeepSeekV3 MLA + group FP8 quant fusion ⏎   - pattern match covers group fp8 flags of col_major, e8m0 and TMA alignment ⏎   - Leverage existing `output_scale` and `output_block_scale` to derive group FP8 quant key ⏎ - Fix pattern match and test for nvfp4 ⏎   - fixed pattern match on previous nvfp4 ⏎   - use `DeepSeek-R1-0528-NVFP4-v2` in e2e test, previous …[truncated]

### L3-123674879e  (L3, 2026-04-21, sha 123674879e8e, PR #39823)
TITLE: [Model] Add block-local attention and YaRN for local layers to Gemma3 (#39823)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.unified_attention, L3.triton.v1_backend
FILES: vllm/model_executor/layers/attention/attention.py (+6/-0); vllm/v1/attention/backends/triton_attn.py (+3/-0); vllm/v1/attention/ops/triton_unified_attention.py (+55/-8); docs/models/supported_models.md (+1/-0); tests/models/registry.py (+4/-0); vllm/model_executor/models/registry.py (+1/-0); vllm/model_executor/models/rnj1.py (+470/-0)
LABELS: documentation, new-model, ready, v1
BODY: ## Purpose ⏎  ⏎ This adds support for an upcoming model from Essential AI in the Rnj-1 series. ⏎  ⏎ Rnj-1 used a similar enough architecture to Gemma3 that no code changes were needed (see config [here](https://huggingface.co/EssentialAI/rnj-1-instruct/blob/main/config.json)). This upcoming model also has a similar architecture, but it requires two changes, which we've implemented as options on the Gemma3 model file: ⏎  ⏎ - Gemma3 uses YaRN for global  …[truncated]

### L3-d622e27d2b  (L3, 2026-04-22, sha d622e27d2be9, PR #35737)
TITLE: [NVFP4] NVFP4 MOE emulation fallback for H100/MI300/MI350, standardize `TritonExperts` usage for OCP MX emulation (#35737)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/evals/gsm8k/configs/models-mi3xx.txt (+2/-0); tests/models/quantization/test_nvfp4.py (+23/-0); vllm/config/kernel.py (+5/-1); vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py (+164/-0); vllm/model_executor/layers/fused_moe/experts/ocp_mx_emulation_moe.py (+186/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+5/-1); vllm/model_executor/layers/fused_moe/fused_moe.py (+11/-47); vllm/model_executor/layers/fused_moe/oracle/mxfp4.py (+22/-0); vllm/model_executor/layers/fused_moe/oracle/nvfp4.py (+42/-0); vllm/model_executor/layers/fused_moe/utils.py (+42/-1); (+2 more)
LABELS: rocm, ready, nvidia, quantization
BODY: ## Purpose ⏎  ⏎ This PR enables running NVFP4 MOE models on Hopper and AMD Instinct MI300, MI350. ⏎  ⏎ This is useful for researchers, anybody trying out microscaling formats, and people who would like to run e.g. https://huggingface.co/nvidia/Qwen3-30B-A3B-NVFP4 or https://huggingface.co/RedHatAI/Qwen3-30B-A3B-NVFP4 on non-Blackwell devices. ⏎  ⏎ This PR also refactors `quark_moe.py` to stop using the functional `fused_experts` function for OCP MX qua …[truncated]

### L3-cefa5281a7  (L3, 2026-04-22, sha cefa5281a752, PR #39835)
TITLE: [ROCm][P/D][MORI][BugFix] Ensure correct api is used when making requests to prefill / decode nodes (#39835)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py (+27/-12); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py (+1/-1)
LABELS: bug, documentation, rocm, ready, kv-connector
BODY: ## Purpose ⏎ This PR fixes the MORI IO KV connector so it uses the correct API when making requests to prefill and decode instances. ⏎ Currently, this is broken since the request URL has a` /v1/completions` suffix which should instead be just `/v1`.  Furthermore, the routes `/v1/completion`s and `/v1/chat/completions` are not differentiated by the handler, which contributes to the problem. Instead, this PR creates individual routes which then call  …[truncated]

### L3-6d09769700  (L3, 2026-04-22, sha 6d097697001e, PR #40176)
TITLE: [ROCm] Support non-causal attention in ROCM_ATTN (#40176)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.triton.prefix_prefill, L3.triton.chunked_prefill_paged_decode, L3.rocm.v1_rocm_attn, L3.rocm.aiter_unified
FILES: vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+6/-2); vllm/v1/attention/backends/rocm_attn.py (+11/-3); vllm/v1/attention/ops/chunked_prefill_paged_decode.py (+2/-0); vllm/v1/attention/ops/prefix_prefill.py (+22/-8)
LABELS: rocm, ready, v1
BODY: Fix DFlash spec decoding with ROCM_ATTN by adding support for bidirectional attention over query tokens. This issue was exposed after https://github.com/vllm-project/vllm/pull/38300/changes added `test_dflash_speculators_model`, which fails on ROCm before this PR. The default behavior of ROCM_ATTN should remain unchanged with this PR ⏎  ⏎ Since I swapped the FlashAttentionMetadata import in rocm_attn with RocmAttentionMetadata, I had to also swap i …[truncated]

### L3-29f64c5f5e  (L3, 2026-04-22, sha 29f64c5f5e63, PR #40394)
TITLE: FlexAttention non-causal support (#40394)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+29/-10); tests/v1/attention/test_attention_backends.py (+73/-4); tests/v1/attention/utils.py (+20/-3)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Currently only the FLASH_ATTN backend supports non-causal attention. This presents an issue when serving models like DFlash speculators, which require non-casual attention on devices like A100s that don't support FLASH_ATTN implementation. ⏎  ⏎ This pr adds support for non-casual attention to the FLEX_ATTENTION backend for decoder models. ⏎  ⏎ ## Test Plan ⏎ Ran  ⏎ ```bash ⏎ vllm serve nm-testing/dflash-qwen3-8b-speculators --attention-bac …[truncated]

### L3-8f87eb4622  (L3, 2026-04-22, sha 8f87eb4622cf, PR #40540)
TITLE: [Refactor] Clean up log once `scope="local"` (#40540)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fa_utils, L3.mla.common_v1, L3.platform.cuda_selection
FILES: vllm/model_executor/layers/attention/attention.py (+0/-1); vllm/model_executor/layers/attention/mla_attention.py (+5/-12); vllm/model_executor/layers/attention/mm_encoder_attention.py (+1/-3); vllm/v1/attention/backends/fa_utils.py (+0/-1); vllm/v1/attention/backends/flash_attn.py (+0/-1); vllm/compilation/backends.py (+3/-13); vllm/compilation/decorators.py (+0/-1); vllm/compilation/monitor.py (+1/-2); vllm/config/scheduler.py (+0/-1); vllm/config/vllm.py (+1/-10); (+34 more)
LABELS: ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ `scope="local"` has been a default option for log once, let's clean up the current caller.

### L3-fe57be7809  (L3, 2026-04-22, sha fe57be780967, PR #40580)
TITLE: [MM][CG] Support `--enable-vit-cuda-graph` option for VLM examples (#40580)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: examples/offline_inference/vision_language.py (+35/-5); vllm/model_executor/models/qwen3_vl.py (+5/-1)
LABELS: documentation, ready, qwen, nvidia
BODY: ## Purpose ⏎ Support `--enable-vit-cuda-graph` option for VLM examples. ⏎  ⏎ ## Test Plan ⏎ ```bash ⏎ python examples/offline_inference/vision_language.py -m qwen3_vl --modality "image" --enable-vit-cuda-graph ⏎ python examples/offline_inference/vision_language.py -m qwen3_vl --modality "video" --enable-vit-cuda-graph ⏎ ``` ⏎  ⏎ ## Test Result ⏎ modality="image": ⏎ ``` ⏎ -------------------------------------------------- ⏎ This image captures a beautiful and  …[truncated]

### L3-4b7869d6bc  (L3, 2026-04-23, sha 4b7869d6bc64, PR #40037)
TITLE: [ROCm] Add gfx1102/gfx1103 support (#40037)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.rocm.custom_paged
FILES: csrc/rocm/attention.cu (+2/-11); CMakeLists.txt (+1/-1); csrc/rocm/skinny_gemms.cu (+18/-23)
LABELS: rocm, ready, ci/build
BODY: gfx1103 (RDNA 3, e.g. Radeon 780M iGPU) was missing from the HIP_SUPPORTED_ARCHS list in CMakeLists.txt and from the compile-time architecture macros in skinny_gemms.cu and attention.cu. ⏎  ⏎ The skinny GEMM and attention kernels defined __HIP__GFX1X__ and __HIP__GFX11__ by listing individual arch targets (__gfx1100__, __gfx1101__, __gfx1150__, __gfx1151__) but missed __gfx1103__. This caused wvSplitK kernel bodies to compile as UNREACHABLE_CODE (a …[truncated]

### L3-fe9c3d6c5f  (L3, 2026-04-23, sha fe9c3d6c5f66, PR #40092)
TITLE: [TurboQuant] enable FA3/FA4 for prefill paths (#40092)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+7/-2); vllm/v1/attention/backends/turboquant_attn.py (+44/-9); tests/evals/gsm8k/configs/Qwen3-4B-TQ-k3v4nc.yaml (+1/-1); tests/evals/gsm8k/configs/Qwen3-4B-TQ-k8v4.yaml (+1/-1); tests/evals/gsm8k/configs/Qwen3-4B-TQ-t3nc.yaml (+1/-1); tests/evals/gsm8k/configs/Qwen3-4B-TQ-t4nc.yaml (+1/-1)
LABELS: ready, v1, quantization
BODY: ## Purpose ⏎  ⏎ Resolves part of https://github.com/vllm-project/vllm/issues/40069 (Backend Coverage: extend `flash_attn_varlen_func` support to FA3/4). ⏎  ⏎ Two issues fixed: ⏎  ⏎ 1. **FA version passthrough:** TurboQuant prefill paths call `flash_attn_varlen_func` without the `fa_version` kwarg, so on Hopper (SM90) the call defaults to FA2 instead of leveraging FA3, and on Blackwell (SM100) it misses FA4 entirely. The standard FlashAttention backend  …[truncated]

### L3-ac58e2a170  (L3, 2026-04-23, sha ac58e2a1704b, PR #39565)
TITLE: [Fix][MoRI] Align MoRI-IO message format with P2pNcclConnector and vllm-router (#39565)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py (+91/-66); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py (+72/-5); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py (+40/-20)
LABELS: documentation, rocm, ready, kv-connector
ISSUES: #38692 [Bug]: parity with CUDA & parity with rocm sglang: vLLM router doesn't current support MoRI kvcache connector
BODY: ## Purpose ⏎  ⏎ Fixes https://github.com/vllm-project/vllm/issues/38692. ⏎  ⏎ This PR aligns the message formats of the MoRI-IO KV Connector with the P2pNcclConnector, **making MoRI-IO itself compatible with vllm-router** with minimal changes required on the router side. ⏎  ⏎ The changes made are: ⏎ - embed peer connection information (ZMQ addresses) directly into the request_id.  ⏎     - This change eliminates the need for the router to explicitly pass  …[truncated]

### L3-2196bac135  (L3, 2026-04-23, sha 2196bac1359a, PR #39684)
TITLE: [Compilation] Refactor SiluMul activation+quant Fusion Pass (#39684)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/fusions_e2e/conftest.py (+7/-1); tests/compile/passes/test_silu_mul_quant_fusion.py (+1/-1); vllm/compilation/passes/fusion/act_quant_fusion.py (+59/-72); vllm/compilation/passes/fusion/rocm_aiter_fusion.py (+14/-29)
LABELS: ready, verified
BODY: Cleaned up the SiluMul activation+quant fusion passes to align with the existing pattern matcher infrastructure. ⏎  ⏎ ### Changes: ⏎ - Refactored SiluMul activation+quant fusion passes to use VllmFusionPatternMatcherPass / VllmPatternReplacement. ⏎ - Fixed a small bug in test_silu_mul_quant_fusion.py ⏎  ⏎ ## Test Plan ⏎ run `vllm/tests/compile/passes/test_silu_mul_quant_fusion.py`. ⏎  ⏎ ## Test Result ⏎ ``` ⏎ ================= 80 passed, 48 skipped, 23 warn …[truncated]

### L3-424033f4fc  (L3, 2026-04-23, sha 424033f4fcee, PR #40627)
TITLE: [Bugfix] Include inductor and functorch configs in compilation cache key (#40627)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/h100/test_startup.py (+2/-0); tests/compile/test_config.py (+18/-0); vllm/compilation/compiler_interface.py (+20/-2)
LABELS: bug, ready-run-all-tests
BODY: Previously, changing TorchInductor configs (e.g. TORCHINDUCTOR_ENABLE_PDL) or aotautograd (functorch) configs would not affect vLLM's compilation cache, causing stale cached artifacts to be loaded instead of recompiling with the new config. ⏎  ⏎ Add inductor_config.save_config_portable() and ⏎ functorch_config.save_config_portable() to get_inductor_factors(), which flows into compiler_hash for all compilation paths (InductorStandalone, Inductor, and …[truncated]

### L3-0098db9ec1  (L3, 2026-04-23, sha 0098db9ec1b1, PR #40015)
TITLE: [ROCm] Implement GPU-to-NUMA-node detection (#40015)
SOURCES: release_notes
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: docs/configuration/optimization.md (+3/-3); vllm/platforms/rocm.py (+28/-0)
LABELS: documentation, rocm, ready
BODY: ## Purpose ⏎ Implement automatic NUMA topology detection on ROCm platforms via the amdsmi library (which is already a dependency). ⏎  ⏎ ## Test Plan ⏎ Run vLLM with `--numa-bind`, observe correct topology detection ⏎  ⏎ ## Test Result ⏎ Topology is detected correctly on the test system with 8 GPUs connected to 2 NUMA nodes: ⏎ ```INFO 04-16 12:57:41 [numa_utils.py:111] Auto-detected NUMA nodes for GPUs: [0, 0, 0, 0, 1, 1, 1, 1]``` ⏎  ⏎ --- ⏎ [details omitted …[truncated]

### L3-98a242ff61  (L3, 2026-04-23, sha 98a242ff61f7, PR #40151)
TITLE: [compile] Skip FX graph deserialiaztion on loading, further reducing warm compile time. (#40151)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/compilation/backends.py (+4/-5); vllm/compilation/caching.py (+20/-15); vllm/compilation/codegen.py (+74/-31)
LABELS: ready
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Summary: ⏎  ⏎ Following up https://github.com/vllm-project/vllm/pull/38657, we utilize the recently-added Python execution code as the source of truth ⏎ to run compiled function moving forward. This means it's not required to serialize the split FX graph anymore, which brings the ⏎ warm compile time further down to sub 2 second level. ⏎  ⏎ Test Plan: ⏎ Before the change ⏎ ``` ⏎ Model                              Cold Compile (s)  Warm Compile Avg (s) ⏎ --- …[truncated]

### L3-b7a2605020  (L3, 2026-04-23, sha b7a26050200e, PR #40193)
TITLE: [Bugfix] Make Attention Backend Auto-Selection Batch-Invariance-Aware (#40193)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.triton.v1_backend, L3.mla.flashattn, L3.dispatch.selector, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+7/-0); vllm/v1/attention/backends/flash_attn.py (+4/-0); vllm/v1/attention/backends/mla/flashattn_mla.py (+4/-0); vllm/v1/attention/backends/mla/triton_mla.py (+4/-0); vllm/v1/attention/backends/triton_attn.py (+4/-0); vllm/v1/attention/selector.py (+10/-1); vllm/v1/worker/gpu_worker.py (+1/-2); examples/rl/rlhf_async_new_apis.py (+1/-8); vllm/model_executor/layers/batch_invariant.py (+3/-39)
LABELS: bug, documentation, ready, v1
ISSUES: #40173 Automatically select highest priority batch-invariant attention backend
BODY: Fixes #40173 ⏎  ⏎ Makes attention backend selection batch-invariance-aware, enabling auto-selection of the highest-priority backend that supports batch invariance when none is explicitly specified. Specific changes: ⏎  ⏎ * Adds `supports_batch_invariance()` to `AttentionBackend`, defaulting to `False`. ⏎ * Adds `is_batch_invariant` to `AttentionSelectorConfig`. ⏎ * Updates `validate_configuration()` to reject backends that do not support batch invarian …[truncated]

### L3-7f95a66cbf  (L3, 2026-04-23, sha 7f95a66cbffc, PR #39233)
TITLE: [NVIDIA] Add sm_110 (Jetson Thor) to CUDA 13.0 build targets (#39233)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+2/-2); docker/docker-bake.hcl (+1/-1); docker/versions.json (+1/-1); tools/flashinfer-build.sh (+1/-1)
LABELS: ready, ci/build, nvidia
BODY: ## Summary ⏎  ⏎ - Add compute capability `11.0` (`sm_110`, Jetson Thor) to all `TORCH_CUDA_ARCH_LIST` configurations that feed CUDA 13.0+ builds ⏎ - The official `vllm/vllm-openai:v0.19.0-cu130` image ships **without** `sm_110` kernels, making it completely unusable on Jetson Thor — precompiled CUDA kernels for Marlin (GPTQ-INT4), CUTLASS (`cutlass_scaled_mm`, FP8), and flash-attn (BF16) all fail with `CUDA error: no kernel image is available for ex …[truncated]

### L3-e9ba519f45  (L3, 2026-04-23, sha e9ba519f450f, PR #39167)
TITLE: [DP][Ray] Pin DP control bundle to same node as first GPU bundle (#39167)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/engine/utils.py (+27/-2)
LABELS: ready, v1
BODY: ## Summary ⏎ Fixes a bug affecting multi-node Data Parallel serving with the Ray backend: ⏎  ⏎ **Ray placement-group control-bundle drift in multi-node `span` DP layouts** — the engine actor is scheduled on the final CPU-only bundle of the placement group. That bundle was previously unconstrained, so ⏎   Ray could place the engine actor on an unrelated node instead of the node that owns the group's first GPU bundle. Fixed by pinning the control bundl …[truncated]

### L3-8824f50f1f  (L3, 2026-04-23, sha 8824f50f1f14, PR #40623)
TITLE: [CI] Split disaggregated tests into own test-area (#40623)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/disaggregated.yaml (+98/-0); .buildkite/test_areas/distributed.yaml (+0/-85); tests/v1/kv_connector/nixl_integration/config_sweep_accuracy_test.sh (+2/-1)
LABELS: ready, ci/build, v1, kv-connector
BODY: I feel PD disaggregation code/relevance has grown enough over the past year that it's better handled with a separated harness. ⏎ Similarly to what's been done here https://github.com/vllm-project/vllm/pull/36945, we're also enabling distributed jobs to be smaller and more independently run, while allowing more budget for targeting PD setups in isolation (which are inherently costly to run..).  ⏎   ⏎ While at it, I also added a FlashInfer eval run wh …[truncated]
