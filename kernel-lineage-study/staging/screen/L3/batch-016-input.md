### L3-c5e3c40877  (L3, 2026-06-25, sha c5e3c40877c2, PR #46628)
TITLE: Fix P/D with DP Supervisor (#46628)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/engine/core.py (+2/-2)
LABELS: ready, v1, kv-connector
BODY: SUMMARY: ⏎ * prior to this PR, each DP rank in DP Supervisor mode ended up with the same engine_id ⏎ * this caused garbage GSM8k

### L3-e8c24a7695  (L3, 2026-06-25, sha e8c24a769576, PR #46643)
TITLE: [Kernel] Vectorized fp32 `moe_sum` reduction and support any topk (#46643)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/moe/moe_align_sum_kernels.cu (+161/-47); tests/kernels/moe/test_moe.py (+16/-4)
LABELS: performance, ready, deepseek
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ `moe_sum` only specialized topk ≤ 4 and fell through to `at::sum_out` for larger topk, so every DeepSeek-V3 / GLM / Qwen MoE (topk=8) paid PyTorch's generic reduction on the critical path. The old topk ≤ 4 kernels also accumulated in bf16, which is lossy across 8 terms. ⏎  ⏎ This replaces the lot with one 16B-vectorized, fp32-accumulating kernel: ⏎  ⏎ - Compile-time unrolled for common topk (1/2/4/6/8/9), runtime fallback otherwise. ⏎ -  …[truncated]

### L3-1744adc256  (L3, 2026-06-25, sha 1744adc256b8, PR #45666)
TITLE: [ROCM] [Communication] Add INT3 quantization method for quickreduce (#45666)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: csrc/custom_quickreduce.cu (+10/-0); csrc/quickreduce/base.h (+23/-0); csrc/quickreduce/quick_reduce.h (+26/-4); csrc/quickreduce/quick_reduce_impl.cuh (+163/-1); tests/distributed/test_quick_all_reduce.py (+2/-2); vllm/distributed/device_communicators/quick_all_reduce.py (+37/-10); vllm/envs.py (+3/-3)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: 1.According to our kernel experiments, INT3 demonstrated better performance under TP2 (see figure below), but ⏎ under TP=4 and TP=8 conditions, the performance of INT3 and INT4 was very close. ⏎ This is likely because, as TP increases, the number of GPUs that need to communicate with each other grows exponentially, thereby requiring more synchronization and scheduling time; ⏎ In this scenario, synchronization and latency overhead—rather than the amo …[truncated]

### L3-2396d91e93  (L3, 2026-06-25, sha 2396d91e9312, PR #44029)
TITLE: [CPU][Spec Decode] Enable DFlash SD for CPU (#44029)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+4/-0); csrc/cpu/spec_decode_utils.cpp (+83/-0); csrc/cpu/torch_bindings.cpp (+23/-0); docs/design/attention_backends.md (+1/-1); vllm/model_executor/models/qwen3_dflash.py (+1/-1); vllm/utils/cpu_triton_utils.py (+134/-0); vllm/v1/spec_decode/dflash.py (+6/-4); vllm/v1/worker/cpu_model_runner.py (+14/-2)
LABELS: documentation, speculative-decoding, ready, v1, qwen, cpu, verified, build-docs
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ Enable DFlash speculative decoding on the vLLM CPU backend. ⏎  ⏎ ## Test Plan ⏎ Srart server: ⏎ ``` ⏎ VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=40 \ ⏎ vllm serve meta-llama/Llama-3.1-8B-Instruct \ ⏎   --dtype bfloat16 \ ⏎   --enforce-eager \ ⏎   --no-enable-prefix-caching \ ⏎   --trust-remote-code \ ⏎   --speculative-config '{"method": "dflash", "model": "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat", "num_speculative_tokens": 8}' ⏎ ``` ⏎ bench …[truncated]

### L3-c63cd4906c  (L3, 2026-06-25, sha c63cd4906c2a, PR #46546)
TITLE: [ROCm][ [Perf] sparse attention optimization on minimax-m3  (#46546)
SOURCES: subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/models/minimax_m3/amd/ops/index_topk.py (+939/-0); vllm/models/minimax_m3/amd/ops/sparse_attn.py (+271/-0); vllm/models/minimax_m3/common/indexer.py (+14/-5); vllm/models/minimax_m3/common/ops/sparse_attn.py (+0/-26); vllm/models/minimax_m3/common/sparse_attention.py (+14/-5)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎ Extend the work of PR https://github.com/vllm-project/vllm/pull/46035 (gfx942 focus) to address gfx950 platform. ⏎  ⏎ Validated on gfx950 to ensure Perf win on minimax-m3-mxfp8 model without accuracy loss before filing up the PR. ⏎  ⏎ Co-Authored-By: @yueliu14 <yue.liu4@amd.com> ⏎  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ **Perf (8k/1k, concurrency 64):** ⏎ | metric | base | opt | Δ | ⏎ |---|---:|---:|---:| ⏎ | Output throughput | 1269.98 | 1351. …[truncated]

### L3-27da2a2ac4  (L3, 2026-06-25, sha 27da2a2ac477, PR #46691)
TITLE: [Hardware][AMD][CI] Use Triton-based AITER MHA for LM Eval Qwen-3.5 Models Tests (#46691)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/vit_attn_wrappers.py (+5/-1); .buildkite/test-amd.yaml (+3/-4); tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-MXFP4-AITER-TP2.yaml (+1/-1)
LABELS: rocm, ready, ci/build, v1, qwen
BODY: ## Purpose ⏎ Use Triton-based (rather than CK-based) AITER MHA in the LM Eval Qwen-3.5 35B AITER test. This sidesteps the issue reported in https://github.com/vllm-project/vllm/pull/46520, and also reverts the changes made in that PR. ⏎  ⏎ ## Test Plan ⏎ `pytest -sv evals/gsm8k/test_gsm8k_correctness.py --config-list-file=configs/models-qwen35-mi355.txt` ⏎ This is run as part of the LM Eval Qwen-3.5 Models test group on MI355 in AMD CI. ⏎  ⏎ ## Test Res …[truncated]

### L3-ae7c8ec223  (L3, 2026-06-25, sha ae7c8ec223e4, PR #46696)
TITLE: [Rust Frontend] Switch `rustls` to `native-tls`/OpenSSL (#46696)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/scripts/run-rust-frontend-cargo-ci.sh (+18/-0); rust/Cargo.lock (+21/-417); rust/Cargo.toml (+6/-6); rust/deny.toml (+15/-0); rust/src/text/Cargo.toml (+1/-0); rust/src/tokenizer/Cargo.toml (+2/-0); rust/src/tokenizer/benches/hf.rs (+12/-8); rust/src/tokenizer/benches/tiktoken.rs (+14/-10)
LABELS: ready, ci/build, rust
BODY: ## Purpose ⏎  ⏎ Related discussions: ⏎ - #46052 @wseaton ⏎ - https://github.com/vllm-project/vllm/pull/45890#issuecomment-4783509287 @tahsintunan  ⏎  ⏎ Eliminate `rustls` from the Rust frontend dependency tree and keep the HTTP/TLS stack on `native-tls` / OpenSSL, which is friendlier for compliance-sensitive deployments. ⏎  ⏎ Also adds a `cargo-deny` bans check for the crypto provider crates we want to keep out of the Rust frontend dependency graph into  …[truncated]

### L3-cc7981599e  (L3, 2026-06-25, sha cc7981599eac, PR #46405)
TITLE: [Refactor] Remove dead kernel code (#46405)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: csrc/custom_all_reduce_test.cu (+0/-361); csrc/ops.h (+0/-6); vllm/_custom_ops.py (+0/-64)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Remove dead kernel code

### L3-5314665bad  (L3, 2026-06-25, sha 5314665badcb, PR #46770)
TITLE: [Model Runner V2][DFlash] Enable dflash attention backend selection (#46770)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/spec_decode/dflash/utils.py (+3/-1)
LABELS: ready, v1
BODY: # Context ⏎ Currently, the speculative config's attention backend is not applied to the DFlash drfat model in Model Runner V2. The consequence is that if the attention backend defaults to FlashInfer, I get the error: ⏎ ``` ⏎ EngineCore pid=1280660)   File "/home/gdelfin/vllm/vllm/v1/executor/multiproc_executor.py", line 402, in collective_rpc ⏎ (EngineCore pid=1280660)     return future if non_block else future.result() ⏎ (EngineCore pid=1280660)      …[truncated]

### L3-02a1f23711  (L3, 2026-06-25, sha 02a1f23711c5, PR #46761)
TITLE: [DFlash] Fuse precompute kv per-layer rmsnorms (#46761)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/layernorm_kernels.cu (+25/-8); tests/kernels/core/test_batched_weight_rms_norm.py (+70/-0); vllm/model_executor/models/qwen3_dflash.py (+13/-10)
LABELS: ready, qwen, dflash
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: # Context ⏎ Currently in `DFlashQwen3Model` during `precompute_and_store_context_kv` we invoke `rms_norm` for each layer in a loop, resulting in multiple kernel launches. This is done because rms_norm only supports broadcasting the single `[hidden_size]` weights to all rows of the input. ⏎  ⏎ <img width="1861" height="388" alt="image" src="https://github.com/user-attachments/assets/4c5d3040-c990-4cd9-bb65-f5e66c598d7d" /> ⏎  ⏎  ⏎ # This PR ⏎ Adds suppor …[truncated]

### L3-d980a3cc6e  (L3, 2026-06-26, sha d980a3cc6ed9, PR #46780)
TITLE: [ROCm] Fix AITER_UNIFIED_ATTN Dispatching After AITER Bump (#46780)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_unified
FILES: vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+11/-0); .buildkite/hardware_tests/amd.yaml (+2/-0); docs/design/attention_backends.md (+1/-1); tests/compile/passes/test_fusion_attn.py (+5/-0); tests/kernels/attention/test_rocm_aiter_unified_attn.py (+2/-1); tests/v1/attention/test_rocm_attention_backends_selection.py (+9/-1)
LABELS: documentation, rocm, ready, ci/build, v1
BODY: After bumping AITER to v0.1.16.post2, we are seeing failures such as the following: ⏎  ⏎ ``` ⏎ tests/models/multimodal/generation/test_whisper.py::test_models[True-5-half-openai/whisper-large-v3-turbo] ⏎  ⏎ (EngineCore pid=7984) ERROR 06-26 00:15:40 [core.py:1233]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ (EngineCore pid=7984) ERROR 06-26 00:15:40 [core.py:1233]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/atten …[truncated]

### L3-8e394244a5  (L3, 2026-06-26, sha 8e394244a59a, PR #46419)
TITLE: [ROCm]Enable AITER MoE backend for MiniMax-M3-MXFP4 (#46419)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/config.py (+2/-0); vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py (+22/-8); vllm/model_executor/layers/fused_moe/layer.py (+2/-0); vllm/models/minimax_m3/amd/model.py (+1/-0)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, intel-gpu, speculative-decoding, ready, ci/build
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: [ROCm][feature] Enable MiniMax-M3-MXFP4 with AITER MoE ⏎  ⏎ This feature requires AITER version bump (latest version).  ⏎  ⏎ **Accuracy test:** ⏎ 1. vLLM server start: VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0 vllm serve  /data/amd-MiniMax-M3-MXFP4/   --block-size 128   -tp 8   --attention-backend TRITON_ATTN   --tool-call-parser minimax_m3   --enable-auto-tool-choice   --reasoning-parser minimax_m3 --moe-backend a …[truncated]

### L3-8921c4be88  (L3, 2026-06-26, sha 8921c4be88ef, PR #46122)
TITLE: [ROCm] [Performance] Optimize aiter moe for DeepSeekV4 (#46122)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/oracle/mxfp4.py (+28/-15)
LABELS: rocm, ready, deepseek
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ NOTE: This feature is validated with aiter `v0.1.15.post1`, we require the AITER on upstream to be updated. ⏎  ⏎ `Mxfp4MoeBackend.AITER_MXFP4_BF16` is introduced only for DeepSeek V4 original weights. ⏎  ⏎ It follows the speed of light reference (ATOM repo) https://github.com/ROCm/ATOM/blob/d7964d50be17a3910dec1d22cf1d4f6205764cb4/recipes/mesh/multi-node-atom.md?plain=1#L74 in preparing the weights. ⏎  ⏎ ## Test Plan ⏎  ⏎ 1. lmeval before a …[truncated]

### L3-c2507fb293  (L3, 2026-06-26, sha c2507fb2937a, PR #46545)
TITLE: [ROCm] [MoE] [Perf] Shared-expert fusion for bias-routed MoE; enable on MiniMax-M3 mxfp8 model (#46545)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/mxfp8_native_moe.py (+7/-1); vllm/model_executor/layers/fused_moe/layer.py (+27/-16); vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py (+26/-0); vllm/model_executor/layers/fused_moe/router/router_factory.py (+3/-0); vllm/models/minimax_m3/amd/model.py (+47/-3)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Perf optimization ⏎  ⏎ MiniMax-M3 runs its single shared expert as a separate dense MLP **every MoE layer** ⏎ (a `gate_up` GEMM + activation + `down` GEMM on a side stream, ×60 layers). Folding it ⏎ into the routed grouped GEMM removes those per-layer launches, which is the dominant cost ⏎ at low/medium concurrency (decode is launch-bound). The fusion is **numerically equivalent** ⏎ to the separate-MLP path. ⏎  ⏎ - **`router/fused_topk_bias_r …[truncated]

### L3-4e07ca2c92  (L3, 2026-06-26, sha 4e07ca2c9284, PR #44800)
TITLE: [Core] Add `VLLM_GPU_SYNC_CHECK` env var (#44800)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/utils_/test_gpu_sync_debug.py (+61/-0); vllm/compilation/compiler_interface.py (+28/-0); vllm/envs.py (+8/-0); vllm/utils/gpu_sync_debug.py (+165/-0); vllm/v1/worker/gpu_worker.py (+17/-0)
LABELS: ready, v1
BODY: vLLM now uses asynchronous scheduling by default and in the majority of cases. Performance relies on the absence of any gpu<->cpu synchronizations on the main cuda stream, but such syncs can be opaque and it is easy for them to creep in accidentally. ⏎  ⏎ This change adds a `VLLM_GPU_SYNC_CHECK` env var which enables `torch.cuda.set_sync_debug_mode` for the model forward pass and sampler, so that we can easily check for such syncs. ⏎  ⏎ A new `gpu_sy …[truncated]

### L3-c6554f321c  (L3, 2026-06-26, sha c6554f321ce4, PR #46769)
TITLE: [CPU] Fix macOS/Apple Silicon hang by enabling OpenMP in the build (#46769)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: csrc/cpu/mla_decode.cpp (+1/-1); .github/actionlint.yaml (+2/-0); .github/workflows/macos-smoke-test.yml (+17/-8); cmake/cpu_extension.cmake (+3/-0); csrc/cpu/cpu_attn_impl.hpp (+3/-3); csrc/cpu/cpu_fused_moe.cpp (+1/-1); csrc/cpu/cpu_types.hpp (+16/-0); csrc/cpu/cpu_wna16.cpp (+1/-1); csrc/cpu/dnnl_kernels.cpp (+1/-1); docs/getting_started/installation/cpu.apple.inc.md (+4/-0)
LABELS: documentation, ready, ci/build, cpu
BODY: ## Purpose ⏎  ⏎ macOS CPU builds never passed `-fopenmp` (regression from #16086, which added `omp.h` but not the flag), so the kernels' `#pragma omp parallel` regions were compiled out and ran serially while `omp_get_max_threads()` still reported all cores. The attention split-KV path then spins on an in-region barrier (`guard_counter != thread_num`) waiting for a thread team that never starts → deadlock on any prompt long enough to hit it (short pr …[truncated]

### L3-af16446bf3  (L3, 2026-06-26, sha af16446bf39d, PR #44465)
TITLE: Vram semaphore infra (#44465)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/cuda.txt (+1/-0); tests/multimodal/test_gpu_ipc_memory.py (+145/-0); tests/multimodal/test_video.py (+234/-0); tests/v1/worker/test_gpu_worker.py (+116/-0); vllm/config/model.py (+4/-0); vllm/config/multimodal.py (+10/-0); vllm/engine/arg_utils.py (+6/-0); vllm/multimodal/gpu_ipc_memory.py (+147/-0); vllm/multimodal/video.py (+331/-13); vllm/renderers/base.py (+11/-0); (+1 more)
LABELS: ready, ci/build, v1, multi-modality, nvidia
BODY: ## Purpose ⏎ Continuing the work described in RFC #30839, this change introduces two new things: ⏎  ⏎ 1. A new **video decoding backend based on `pynvvideocodec`** which is a lightweight dependency enabling HW video decoding on all NVIDIA GPUs. Video decoding always takes place using GPU index 0. ⏎ 2. A **VRAM Semaphore**. The semaphore enables the GPU video decoding backend to logically allocate N bytes from a pool, or block if unavailable. This ens …[truncated]

### L3-c6dd32a810  (L3, 2026-06-26, sha c6dd32a810aa, PR #46762)
TITLE: [ModelRunner V2] Support realtime embeddings (#46762)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_encoder_runner.py (+3/-3); vllm/model_executor/models/diffusion_gemma.py (+1/-1); vllm/v1/worker/gpu/mm/encoder_runner.py (+20/-17); vllm/v1/worker/gpu/model_runner.py (+18/-13); vllm/v1/worker/gpu/model_states/default.py (+4/-0); vllm/v1/worker/gpu/model_states/interface.py (+6/-2)
LABELS: ready, v1, mrv2
BODY: Realtime models like voxtral require embeddings for decode steps too.

### L3-d0f800811b  (L3, 2026-06-26, sha d0f800811bb8, PR #46644)
TITLE: [Build] Update vllm to point to vllm-project/flash-attention commit that builds FA3 with torch stable API.  (#46644)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, corpus:confirmed-reverts(reverted), body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: rocm, ready, ci/build
DEEP_STUDY: deep-study: this PR was reverted by PR 48269 (explicit_rollback, reason=hardware_specific_breakage)
BODY: ## Purpose ⏎ As a part of https://github.com/vllm-project/vllm/issues/26946. Updated vllm_flash_attn.cmake to use FA fork that builds a libtorch ABI stable FA3 library.  ⏎  ⏎ Currently, this is pointing towards my flash-attention fork but if this passes the CI, we will merge that fork with vllm-project/flash-attention, then we will point to the new tag.  ⏎  ⏎ see https://github.com/vllm-project/flash-attention/pull/152 to see the commit it is pointing …[truncated]

### L3-c40d307731  (L3, 2026-06-26, sha c40d307731b8, PR #36701)
TITLE: [Core] Remove FlashAttention block size restriction for hybrid models (#36701)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+0/-17)
LABELS: ready, v1
BODY: ## Summary ⏎  ⏎ - Remove the block size restriction in `FlashAttentionBackend.get_supported_kernel_block_sizes()` that limited hybrid models with float32 Mamba cache to block sizes `[16, 32, 64]` ⏎ - This restriction was introduced in #27753 to work around NaN propagation from stale fp32 Mamba data in reused KV cache blocks ⏎ - #35219 has since solved the root cause by zeroing freshly allocated KV cache blocks via `KVBlockZeroer`, making the restriction  …[truncated]

### L3-2ff76a5e85  (L3, 2026-06-26, sha 2ff76a5e856e, PR #46760)
TITLE: [ROCm][Bugfix] Pass num_kv_splits to aiter mla_reduce_v1 (#46760)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+3/-0)
LABELS: bug, rocm, ready, v1
BODY: ## Purpose ⏎  ⏎ #46692 bumped AITER to `v0.1.16.post2`, which includes ROCm/aiter#3391. That aiter PR added a **required `num_kv_splits` argument** to `mla_reduce_v1` (now position 7 of the registered op schema). The FP8 ASM prefill reduce call in the AITER dense MLA backend (`rocm_aiter_mla.py`) was not updated, so on `main` today the `out_3d` Tensor is bound to the `num_kv_splits` (int) slot and **every MLA model crashes at prefill on gfx950** (Dee …[truncated]

### L3-091d13976c  (L3, 2026-06-27, sha 091d13976c1c, PR #46891)
TITLE: [ROCm][CI] Add TRITON_ATTN score absolute tolerance floor (#46891)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/entrypoints/pooling/scoring/test_cross_encoder_online_vision.py (+8/-3)
LABELS: rocm, ready
BODY: ## Purpose ⏎  ⏎ The `text_vs_text` score (~0.10) in `test_cross_encoder_online_vision.py` drifts ~0.008 absolute on gfx942/ROCm 7.2 under the `TRITON_ATTN` backend. Because the value is near zero, this small fixed absolute drift reads as ~7.9% relative error and exceeds the tight 0.045 relative tolerance ~@~T even though the larger scores (0.53, 0.74) stay well within it. ⏎  ⏎ Relative tolerance is the wrong metric for a probability near zero. `TRITO …[truncated]

### L3-867fd5e8ed  (L3, 2026-06-27, sha 867fd5e8ed6b, PR #46184)
TITLE: [ROCm][Perf] Use flydsl moe with Minimax-M3 mxfp8 weights on gfx950 and implemented moe-backend selection (#46184)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_mxfp8_aiter_backend_selection.py (+135/-0); vllm/_aiter_ops.py (+30/-0); vllm/model_executor/layers/fused_moe/experts/aiter_mxfp8_moe.py (+166/-0); vllm/model_executor/layers/fused_moe/oracle/fp8.py (+8/-2); vllm/model_executor/layers/fused_moe/oracle/mxfp8.py (+35/-2)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎ [ROCm][Perf] Use flydsl moe with Minimax-M3 mxfp8 weights on gfx950 ⏎  ⏎ aiter's flydls moe has shown perf improvement on various scenrios on mxfp8 serving on gfx950. This PR is to integrate the  ⏎ support, and also have the capability to fall back to triton mxfp8 dot-scaled implementation. ⏎  ⏎ Implementation and usability decisions: ⏎  ⏎ Reused moe-backend aiter and triton, and refactored the moe selection logic. ⏎  ⏎ Gating (the usability d …[truncated]

### L3-51a99565c3  (L3, 2026-06-27, sha 51a99565c398, PR #46474)
TITLE: [ROCm][Perf] Fused shared expert for Minimax M3 (#46474)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/minimax_m3/amd/model.py (+46/-13)
LABELS: rocm, ready, verified
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose  ⏎ Fuse MiniMax-M3 shared expert into the routed MoE call, a follow-up PR for https://github.com/vllm-project/vllm/pull/46419 ⏎ ## Test Plan ⏎ vLLM server start: `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0 .venv/bin/python -m vllm.entrypoints.cli.main serve amd/MiniMax-M3-MXFP4 --block-size 128 -tp 4 --max-model-len 9472 --attention-backend TRITON_ATTN --tool- …[truncated]

### L3-35e3850fa9  (L3, 2026-06-27, sha 35e3850fa949, PR #46915)
TITLE: [Bugfix][Test] Fix test_flashinfer_cutlass_mxfp4_fused_moe on sm90 (stale weight/scale interleave) (#46915)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_ocp_mx_moe.py (+14/-15)
LABELS: bug, ready, nvidia
ISSUES: #46585 [Bug]: test_flashinfer_cutlass_mxfp4_fused_moe accuracy mismatch on H20 (sm90) — 89% mismatch vs 20% threshold
BODY: ## Purpose ⏎  ⏎ Fixes the stale weight/scale preparation in `test_flashinfer_cutlass_mxfp4_fused_moe`, which fails on Hopper (sm90) with ~89% mismatch (vs the 0.2 threshold). ⏎  ⏎ Closes #46585. ⏎  ⏎ ## Root cause ⏎  ⏎ The test prepared inputs for the SM90 mixed-input cutlass MoE GEMM (`cutlass_fused_moe(use_w4_group_scaling=True)`) with: ⏎ - a hand-rolled `_interleave_scales_lastdim_by4` for scales, and ⏎ - **no interleave at all** for the packed 4-bit weights. ⏎  ⏎ Th …[truncated]

### L3-9036c89ee4  (L3, 2026-06-27, sha 9036c89ee410, PR #46928)
TITLE: [Hardware][AMD][CI] Patch Whisper multi LoRA test to use TRITON_ATTN for now (#46928)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/lora/test_whisper.py (+10/-2)
LABELS: rocm, ready
BODY: ## Purpose ⏎ This PR temporarily patches `tests/lora/test_whisper.py::test_whisper_multi_lora` to use `TRITON_ATTN`, as this test is currently failing on `main` and blocking CI. This test started failing after https://github.com/vllm-project/vllm/pull/46780, which had the effect of switching the test from using `ROCM_AITER_UNIFIED_ATTN` to `ROCM_ATTN` (because the former currently does not support FP16 dtypes after the AITER bump). ⏎  ⏎ It remains u …[truncated]

### L3-c6741b2ad4  (L3, 2026-06-27, sha c6741b2ad48a, PR #46564)
TITLE: [Model] Support Unlimited OCR (#46564)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.abstract_interface, L3.flex_attention
FILES: vllm/model_executor/layers/attention/__init__.py (+2/-0); vllm/model_executor/layers/attention/rswa_attention.py (+37/-0); vllm/v1/attention/backend.py (+8/-0); vllm/v1/attention/backends/flash_attn.py (+114/-3); vllm/v1/attention/backends/flex_attention.py (+148/-4); docs/models/supported_models.md (+1/-0); tests/models/registry.py (+3/-0); tests/v1/core/test_single_type_kv_cache_manager.py (+51/-1); vllm/config/model.py (+4/-0); vllm/config/model_arch.py (+3/-0); (+25 more)
LABELS: documentation, new-model, ready, v1, deepseek
BODY: ## Purpose ⏎  ⏎ Support https://huggingface.co/baidu/Unlimited-OCR ⏎  ⏎ ## Test ⏎  ⏎ OmniDocBench ⏎  ⏎  ⏎ | Metric | FA4 | FlexAttention | Paper v1.6 | ⏎   |---|---|---|---| ⏎   | Overall ↑ | 92.12 | 92.38 | 93.92 | ⏎   | Text Edit ↓ | 0.089 | 0.087 | 0.042 | ⏎   | Formula CDM ↑ | 95.34 | 95.53 | 95.79 | ⏎   | Formula Edit ↓ | 0.108 | 0.105 | — | ⏎   | Table TEDS ↑ | 89.90 | 90.33 | 90.16 | ⏎   | Table TEDS-S ↑ | 93.27 | 93.60 | 93.32 | ⏎   | Read-order Edit ↓ |  …[truncated]

### L3-c7ca0bccae  (L3, 2026-06-28, sha c7ca0bccae66, PR #44313)
TITLE: [ROCm][Perf] Add Fused Shared Expert (FSE) support for GLM-4.5/6/7 (#44313)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/glm4_moe.py (+138/-63); vllm/model_executor/models/glm4_moe_mtp.py (+116/-42)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Extend the AITER Fused Shared Expert (FSE) path — originally added for DeepSeek-V2/V3 (#28540) and Qwen3-Next (#39280) — to the GLM-4 MoE family (GLM-4.5, GLM-4.6, GLM-4.7). When `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`, the shared expert is folded into the AITER `FusedMoE` kernel as `n_shared_experts` extra expert slots, eliminating the separate shared-expert MLP forward pass at low/medium concurrency. ⏎  ⏎ This PR extends FSE to bo …[truncated]

### L3-5ecae3266c  (L3, 2026-06-28, sha 5ecae3266cd5, PR #45033)
TITLE: [ROCm][Perf][MLA] Add AITER FlashAttention MLA prefill backend (`ROCM_AITER_FA`) (#45033)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py (+121/-0); vllm/v1/attention/backends/mla/prefill/registry.py (+4/-0); vllm/v1/attention/backends/mla/prefill/selector.py (+8/-0); tests/v1/attention/test_mla_prefill_registry.py (+17/-0); tests/v1/attention/test_mla_prefill_selector.py (+119/-0)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ Adds an AITER-backed MLA prefill backend that calls `aiter.flash_attn_varlen_func` ⏎ directly instead of falling back to the upstream CK-tile FlashAttention path on ROCm. ⏎  ⏎ It dispatches the fast `aiter::fmha_fwd_*` kernel on gfx950 and, unlike the CK ⏎ fallback, needs **no V padding** (native qk=192 / v=128 head dims) and returns ⏎ `softmax_lse` already shaped `(nheads, total_q)` (**no LSE transpose**). This is the ⏎ fp16/bf16 generic …[truncated]

### L3-95528527ea  (L3, 2026-06-28, sha 95528527eab9, PR #46855)
TITLE: [Bugfix][Mooncake] Fix Mooncake lookup prefixes with DCP > 1 (#46855)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_mooncake_store_worker.py (+48/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py (+4/-2); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py (+34/-8)
LABELS: bug, ready, v1, kv-connector
BODY: ## Purpose ⏎ Current Mooncake store lookup only iterates over TP and PP ranks and fails to consider DCP and PCP ranks. This results in lookup to return True despite keys for certain ranks are missing. ⏎  ⏎ ``` ⏎ self._lookup_key_prefixes = tuple( ⏎             tuple( ⏎                 PoolKey.build_prefix(db.metadata, tp_rank=tp, pp_rank=pp) ⏎                 for tp in range(tp_count) ⏎                 for pp in range(self.pp_size) ⏎             ) ⏎        …[truncated]

### L3-4dfbf1503b  (L3, 2026-06-28, sha 4dfbf1503b4b, PR #41026)
TITLE: [Model] Add support for openai/privacy-filter (#41026)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+1/-0); docs/models/pooling_models/token_classify.md (+1/-0); tests/models/language/pooling/test_token_classification.py (+47/-0); tests/models/registry.py (+4/-0); vllm/model_executor/layers/fused_moe/oracle/unquantized.py (+9/-0); vllm/model_executor/models/gpt_oss.py (+32/-5); vllm/model_executor/models/openai_privacy_filter.py (+127/-0); vllm/model_executor/models/registry.py (+4/-0)
LABELS: documentation, new-model, ready, v1, gpt-oss, verified
BODY: ## Purpose ⏎  ⏎ Adds support for [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) (`OpenAIPrivacyFilterForTokenClassification`): a gpt-oss-style MoE encoder (GQA 14/2, YaRN RoPE, 128 experts top-4, attention sinks) repurposed as a bidirectional token classifier for PII detection. Every layer uses non-causal attention with a banded ±`sliding_window` mask. The LM head is replaced with a 33-class BIOES score head. ⏎  ⏎ The new mod …[truncated]

### L3-6149187a4c  (L3, 2026-06-29, sha 6149187a4cca, PR #46819)
TITLE: [Kernel] Triton MLA logits workspace (#46819)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/triton_mla.py (+64/-29)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: `TritonMLAImpl.forward_mqa` allocates a fresh float32 scratch tensor attn_logits on every decode forward call. ⏎ This PR addresses the TODO https://github.com/vllm-project/vllm/blob/bf292b5f6b537d154fc09a3b232f89cbc66827f5/vllm/v1/attention/backends/mla/triton_mla.py#L185-L196 ⏎  ⏎ by allocating a logits workspace ahead of time, using a single buffer that layer can share. This makes it so that we don't pay the allocation price at every step.  ⏎ The a …[truncated]

### L3-debec6440b  (L3, 2026-06-29, sha debec6440b89, PR #46756)
TITLE: Add MiniMax-M3 modelopt nvfp4 support (#46756)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/quantization/test_modelopt.py (+6/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+59/-9); vllm/model_executor/layers/quantization/modelopt.py (+27/-0); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+1/-0)
LABELS: ready, nvidia, quantization
BODY: ## Summary ⏎  ⏎ Ports https://github.com/vllm-project/vllm/pull/46380 from `minimax-m3-perf` onto current `main`. ⏎  ⏎ The original PR was merged into `minimax-m3-perf`, but current `main` was still missing the relevant support points when checked locally: ModelOpt mixed MXFP8 dispatch, the parent-prefix fallback for fused projections, and the NVFP4 MoE SwiGLU-OAI alpha/beta/clamp wiring. ⏎  ⏎ ## Changes ⏎  ⏎ - Add ModelOpt mixed-precision MXFP8 Linear a …[truncated]

### L3-4708292d48  (L3, 2026-06-29, sha 4708292d48f3, PR #46683)
TITLE: Bump flashinfer version to 0.6.13 (#46683)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); docker/Dockerfile.nightly_torch (+2/-2); docker/versions.json (+1/-1); requirements/cuda.txt (+2/-2); tests/evals/gsm8k/test_gsm8k_correctness.py (+9/-2); vllm/model_executor/warmup/kernel_warmup.py (+14/-7)
LABELS: ready, ci/build, nvidia, ready-run-all-tests
BODY: ## Purpose ⏎ Bump flashinfer version to 0.6.13 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-a309d4fe60  (L3, 2026-06-29, sha a309d4fe60be, PR #43729)
TITLE: Support DCP with FlashInfer MLA (#43729)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer
FILES: vllm/v1/attention/backends/mla/flashinfer_mla.py (+10/-4); docs/design/attention_backends.md (+1/-1)
LABELS: documentation, ready, ci/build, v1, nvidia
BODY: FlashInfer MLA supports LSE since https://github.com/flashinfer-ai/flashinfer/pull/3116 ⏎ This PR can be merged once it's included in GA.

### L3-61ab70ec3b  (L3, 2026-06-29, sha 61ab70ec3bd1, PR #42406)
TITLE: [Model Runner V2] support mamba hybrid models align prefix cache (#42406)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/mamba/test_precopy_mamba_align.py (+180/-0); tests/v1/e2e/general/test_mamba_prefix_cache.py (+280/-16); vllm/config/vllm.py (+0/-11); vllm/model_executor/models/diffusion_gemma.py (+3/-1); vllm/v1/worker/gpu/model_runner.py (+13/-1); vllm/v1/worker/gpu/model_states/interface.py (+16/-1); vllm/v1/worker/gpu/model_states/mamba_hybrid.py (+174/-8); vllm/v1/worker/gpu/warmup.py (+12/-3); vllm/v1/worker/mamba_utils.py (+367/-91)
LABELS: ready, v1, mrv2
BODY: ## Purpose ⏎  ⏎ A follow PR of https://github.com/vllm-project/vllm/pull/35520 ⏎  ⏎ Support `mamba_cache_mode="align"` for Mamba prefix caching with Model Runner V2. The implementation keeps the align-mode state handling inside `MambaHybridModelState`, avoids CPU/GPU synchronization in the hot path. ⏎  ⏎ ## Design ⏎  ⏎ 1. **No post-copy needed**: The kernel reads from the src block (previous running block) and writes to the window block (which becomes th …[truncated]

### L3-8fc1b2d046  (L3, 2026-06-29, sha 8fc1b2d046f4, PR #46659)
TITLE: Fix FA4 dynamic_causal for full attention layers (#46659)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+4/-1)
LABELS: ready, ci/build, v1
BODY: Depends on https://github.com/vllm-project/flash-attention/pull/154, merge that first ⏎  ⏎ ## Purpose ⏎ Currently, FA4 doesn't actually use dynamic causal in full attention layers because the backend hands the per-request causal tensor to the kernel as `dynamic_causal` but then calls the kernel with `causal=False`, so it never actually reads the causal tensor. This causes it to run full bidirectional attention for every request. This PR fixes this. …[truncated]

### L3-43916891b2  (L3, 2026-06-29, sha 43916891b222, PR #46346)
TITLE: [GDN] Improve kkt kernel of CuteDSL prefill backend (#46346)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/cute_utils/__init__.py (+29/-15); vllm/cute_utils/_tcgen05.py (+34/-30); vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/__init__.py (+23/-28); vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/kernel_h.py (+89/-90); vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/kernel_kkt_inv_uw.py (+297/-231); vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/kernel_o.py (+82/-80)
LABELS: ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ This a a follow-up of #43273. In this PR, I only focused on optimizing the matrix inverse pipeline inside the KKT kernel. A quick recap on the KKT kernel timeline ⏎ <img width="3684" height="648" alt="kkt_old" src="https://github.com/user-attachments/assets/8197c715-a2fc-46a0-98e0-7fe61888fd15" /> ⏎  ⏎ This is a bit more detailed than the timeline shown previously in #43273, because the extra steps shown above (`Prepare gate/beta` and  …[truncated]

### L3-5b4cb69523  (L3, 2026-06-29, sha 5b4cb6952310, PR #47079)
TITLE: [Bugfix][MLA] Fix LSE log-base mismatch in DCP + FlashInfer MLA decode (#47079)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashinfer, L3.dispatch.abstract_interface
FILES: vllm/model_executor/layers/attention/mla_attention.py (+2/-2); vllm/v1/attention/backend.py (+11/-0); vllm/v1/attention/backends/mla/flashinfer_mla.py (+7/-0)
LABELS: bug, ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ FlashInfer's trtllm-gen MLA decode kernel returns LSE in **log2** (per flashinfer's own reference at `flashinfer/trace/templates/attention.py:81`: `logsumexp / log(2.0)`), but `MLAAttention`'s DCP combine in `vllm/model_executor/layers/attention/mla_attention.py` hard-codes `is_lse_base_on_e=True` when calling `cp_lse_ag_out_rs` / `dcp_a2a_lse_reduce`, asserting natural log. The mismatch makes the per-shard re-weighting use the wrong  …[truncated]

### L3-b153dd3f28  (L3, 2026-06-29, sha b153dd3f2811, PR #47074)
TITLE: [Bugfix] Use larger workspace size for Flashinfer MLA LSE (#47074)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer
FILES: vllm/v1/attention/backends/mla/flashinfer_mla.py (+18/-9)
LABELS: bug, ready, v1, nvidia
BODY: ## Purpose ⏎ When using Flashinfer MLA LSE with Kimi K2.6, the following error is thrown due to insufficient workspace size: ⏎ ``` ⏎  ERROR 06-29 13:11:30 [multiproc_executor.py:1000]   File "/vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py", line 202, in forward_mqa ⏎  ERROR 06-29 13:11:30 [multiproc_executor.py:1000]     kernel_out = trtllm_batch_decode_with_kv_cache_mla( ⏎  ERROR 06-29 13:11:30 [multiproc_executor.py:1000]                  ^^ …[truncated]

### L3-49e28e8e91  (L3, 2026-06-29, sha 49e28e8e91ad, PR #44010)
TITLE: [Kernel][Helion][1/N] Add Helion kernel for fused_qk_norm_rope (#44010)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/helion/test_fused_qk_norm_rope.py (+261/-0); vllm/kernels/helion/configs/fused_qk_norm_rope/nvidia_b200.json (+2612/-0); vllm/kernels/helion/configs/fused_qk_norm_rope/nvidia_h100.json (+2722/-0); vllm/kernels/helion/ops/fused_qk_norm_rope.py (+316/-0)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ This PR is to add Helion kernel for ```fused_qk_norm_rope``` operation. This is a subtask for https://github.com/vllm-project/vllm/issues/32962. ⏎  ⏎ ### Kernel level benchmark ⏎ **Environment** ⏎ Python: 3.12.12 ⏎ Pytorch: 2.11.0 ⏎ Cuda: 13.0 ⏎ Helion: 1.0.0 ⏎  ⏎ **Benchmark Setup** ⏎ Latency measure: ```triton.testing.do_bench_cudagraph(rep=1000)``` ⏎ Baseline: ```torch.compile``` and ```torch.ops._C.fused_qk_norm_rope``` ⏎  ⏎ **Autotuning Setup …[truncated]

### L3-0feca7ffa8  (L3, 2026-06-29, sha 0feca7ffa8f6, PR #46807)
TITLE: PD disagg with Mooncake Connector: GDN support (Qwen3.5) and MLA support (Deepseek-V4-Flash) (#46807)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_mooncake_connector.py (+36/-4); tests/v1/kv_connector/unit/test_mooncake_connector_hma.py (+2/-0); tests/v1/kv_connector/unit/test_mooncake_connector_hybrid_mamba.py (+386/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py (+308/-96); vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py (+1/-1); vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py (+1/-1); vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py (+1/-1)
LABELS: ready, v1, qwen, deepseek, kv-connector, verified
BODY: ## Purpose ⏎  ⏎ Mooncake P/D disaggregation did not work correctly for hybrid KV-cache models such as Qwen3.5, where one model has both full-attention KV cache groups and GDN state groups represented as `MambaSpec`. Nixl now has supported GDN pd disagg here https://github.com/vllm-project/vllm/pull/41869, but mooncake connector lacks this implementation. ⏎  ⏎ The previous Mooncake path treated request block IDs and registered cache regions too much l …[truncated]

### L3-77654d080c  (L3, 2026-06-30, sha 77654d080c61, PR #46777)
TITLE: [KVTransfer] MultiConnector: merge kv_transfer_params dicts across connectors (#46777)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py (+9/-6)
LABELS: ready, kv-connector
BODY: **Description:** ⏎ `MultiConnector.request_finished` previously raised a `RuntimeError` when more than one child connector returned a non-`None` `kv_transfer_params`. This prevented combining connectors that each contribute disjoint keys to the params dictionary—for example, a caching connector returning `cached_token_stats` alongside a P/D connector returning transfer metadata.

### L3-aab7af0bcb  (L3, 2026-06-30, sha aab7af0bcbf9, PR #46997)
TITLE: [Bugfix][ROCm][MLA] Pass q/kv dtypes to get_mla_metadata_v1 in FP8 decode (#46997)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+6/-0); .buildkite/test-amd.yaml (+1/-0); tests/kernels/attention/test_rocm_aiter_mla_decode_metadata.py (+202/-0)
LABELS: bug, rocm, ready, ci/build, v1
BODY: ## Purpose ⏎  ⏎ Fix a silent accuracy regression for MLA models served with an **FP8 KV cache** on **gfx950 (MI355X)** using the AITER MLA backend. With `amd-aiter == 0.1.16.post2`, Kimi-K2.5 at TP2 (32 q-heads/rank) produces garbage — gsm8k drops from ~0.93 to ~0. ⏎  ⏎ ### Root cause ⏎  ⏎ `AiterMLAMetadataBuilder` builds the persistent split/reduce metadata for decode via `get_mla_metadata_v1`, but did **not** forward `dtype_q` / `dtype_kv` (they defa …[truncated]

### L3-bec232a914  (L3, 2026-06-30, sha bec232a9146b, PR #42285)
TITLE: Secondary tier implementation for PD disaggregation (#42285)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/features/kv_offloading_usage.md (+14/-0); tests/v1/kv_offload/tiering/p2p/__init__.py (+0/-0); tests/v1/kv_offload/tiering/p2p/p2p_connector_proxy.py (+331/-0); tests/v1/kv_offload/tiering/p2p/run_accuracy_test.sh (+312/-0); tests/v1/kv_offload/tiering/p2p/test_data_transport.py (+330/-0); tests/v1/kv_offload/tiering/p2p/test_manager.py (+1370/-0); tests/v1/kv_offload/tiering/p2p/test_sessions.py (+1645/-0); tests/v1/kv_offload/tiering/p2p/test_zmq_transport.py (+230/-0); vllm/v1/kv_offload/tiering/factory.py (+6/-0); vllm/v1/kv_offload/tiering/p2p/__init__.py (+0/-0); (+12 more)
LABELS: documentation, ready, v1, kv-connector
BODY: In this design PD disaggregation is based on the vLLM CPU KV cache which is per vLLM instance and it is in canonical layout (single TP unified block size). The PD Connector is a secondary tier that implements the `SecondaryTierManager` interface. Orchestration between the primary tier (CPU Manager) and secondary tiers is done by the `TieringManager`, which is transparent to the secondary tiers. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details o …[truncated]

### L3-a7732537f4  (L3, 2026-06-30, sha a7732537f430, PR #47039)
TITLE: [Bugfix] Restore part of bugfix #42650 after accidental deletion in #43241 (#47039)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/flashinfer.py (+5/-3); vllm/v1/attention/backends/triton_attn.py (+5/-3); vllm/v1/attention/backends/utils.py (+26/-0)
LABELS: bug, ready, v1, nvidia
BODY: This PR restores the upstream https://github.com/vllm-project/vllm/pull/42650 behavior that appears to have been accidentally reverted during https://github.com/vllm-project/vllm/pull/43241 final conflict-resolution/rebase cycle. ⏎  ⏎ ### Relevant upstream timeline: ⏎  ⏎ - vLLM #42650 merged on 2026-05-22 as `c7624bea5` (`[Bugfix] Source num_qo_heads from Attention layers in Flashinfer/Triton metadata builders`). That PR added `get_num_attention_head …[truncated]

### L3-c8f9c156a5  (L3, 2026-06-30, sha c8f9c156a5f8, PR #46993)
TITLE: [ROCm][V1][MLA] Clone prefill backend state per metadata builder (#46993)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+4/-1); vllm/v1/attention/backends/mla/prefill/base.py (+11/-0); tests/v1/attention/test_mla_prefill_registry.py (+22/-0)
LABELS: rocm, ready, v1
BODY: This prevents V1 MLA prefill metadata from being shared across metadata builders. DBO creates metadata builders per ubatch. MLA prefill backends store prepared prefill metadata on the backend object. Reusing the same backend instance from the static forward context lets one ubatch overwrite another ubatch's prepared metadata. ⏎  ⏎ Proposed changes: ⏎ - Add `MLAPrefillBackend.clone()`. ⏎ - Have `MLACommonMetadataBuilder` clone the prefill backend from …[truncated]

### L3-248d1fbb71  (L3, 2026-06-30, sha 248d1fbb711b, PR #46182)
TITLE: [Feat][1/N] CuTeDSL warmup infrastructure, FA4 MLA (#46182)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_build, L3.flash_attn.fa4_cutedsl, L3.flash_attn.fa_utils
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/config/kernel.py (+9/-1); vllm/v1/attention/backends/fa_utils.py (+76/-0); vllm/v1/attention/backends/mla/prefill/flash_attn.py (+49/-0); vllm/vllm_flash_attn/__init__.py (+2/-0); vllm/vllm_flash_attn/flash_attn_interface.py (+60/-0); vllm/model_executor/warmup/cutedsl_warmup.py (+113/-0); vllm/model_executor/warmup/fa4_cutedsl_config.py (+204/-0); vllm/model_executor/warmup/kernel_warmup.py (+4/-0)
LABELS: ready, ci/build, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Depends on: https://github.com/vllm-project/vllm/pull/46167 and https://github.com/vllm-project/flash-attention/pull/150 ⏎  ⏎ ## Description ⏎ This is the first PR in a broader CuTeDSL warmup series. The initial study focuses only on the FA4 MLA prefill path. ⏎  ⏎ This PR adds a generic CuTeDSL warmup hook and one provider for FA4 MLA prefill. The goal is to compile the relevant FA4 CuTeDSL kernels during model warmup, before serving real requests, so …[truncated]

### L3-4236514098  (L3, 2026-06-30, sha 42365140980d, PR #47004)
TITLE: [ROCm][CI][Multimodal] Use ROCm-aware FA availability check for Unlimited-OCR (#47004)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/config.py (+1/-1)
LABELS: rocm, ready
BODY: Use the attention backend helper for the Unlimited-OCR FlashAttention 4 availability check. Unlimited-OCR defaults to FlashAttention 4 when it is available, otherwise it falls back to FlexAttention. The config check imported `is_fa_version_supported` directly from `vllm.vllm_flash_attn`, which is CUDA-only in this ROCm environment. On ROCm that import can fail before the fallback decision is reached, so `test_model_tensor_schema[baidu/Unlimited-O …[truncated]

### L3-fcaa84efa7  (L3, 2026-06-30, sha fcaa84efa7a9, PR #47050)
TITLE: [BugFix] Gate MRV2 mixed sparse-MLA warmup on `max_num_seqs` > 1 (#47050)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/warmup.py (+1/-1); tests/v1/worker/test_mixed_warmup_gate.py (+30/-0); vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py (+3/-3)
LABELS: bug, ready, v1, nvidia
BODY: Thanks to @ZeldaHuang for finding this

### L3-f098ee70c7  (L3, 2026-06-30, sha f098ee70c730, PR #47090)
TITLE: [GLM5] Support FlashMLA FP8 KV cache (Hopper & Blackwell) (#47090)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_fused_deepseek_v32_norm_rope.py (+139/-0); vllm/models/deepseek_v32/nvidia/attention.py (+33/-13); vllm/models/deepseek_v32/nvidia/kernels.py (+135/-31)
LABELS: ready, deepseek
BODY: Adds FlashMLA backend support to the new GLM5/DSV3.2 model definition. This PR modifies the existing fused kernels to support FlashMLA's FP8 KV cache layout and support BF16 query (which is required for Hopper).

### L3-8cf7c4d8ad  (L3, 2026-06-30, sha 8cf7c4d8ad60, PR #46020)
TITLE: [Attention Backend] add HPC-Ops Attention backend (#46020)
SOURCES: path_core, path_integration+keyword, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.dispatch.registry
FILES: vllm/config/compilation.py (+1/-0); vllm/model_executor/layers/attention/attention.py (+3/-1); vllm/v1/attention/backends/hpc_attn.py (+469/-0); vllm/v1/attention/backends/registry.py (+6/-0); docs/design/attention_backends.md (+1/-0); vllm/model_executor/layers/hpc/__init__.py (+11/-0); vllm/model_executor/layers/hpc/hpc_module.py (+18/-0); vllm/model_executor/layers/hpc/rope_norm.py (+408/-0); vllm/model_executor/model_loader/utils.py (+10/-0); vllm/model_executor/models/hy_v3.py (+51/-13)
LABELS: documentation, performance, ready, v1, verified
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Support hpc-ops attention backend. [HPC-Ops](https://github.com/Tencent/hpc-ops) is a production-grade, high-performance, and easy-to-use operator library for LLM inference, developed by the Tencent Hunyuan AI Infra team. ⏎  ⏎ **You can enable hpc-ops attention backend by specify `--attention_backend HPC_ATTN --kv-cache-dtype fp8_e4m3 --block-size 64`.** ⏎  ⏎ WARNING: The HPC attention backend currently supports only the Hy3-FP8 model.  …[truncated]

### L3-a264e41975  (L3, 2026-06-30, sha a264e419751a, PR #47219)
TITLE: [Distributed] Default FlashInfer allreduce to mnnvl on single node (#47219)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/distributed/device_communicators/flashinfer_all_reduce.py (+38/-22)
LABELS: ready, nvidia
BODY: ## Purpose ⏎  ⏎ The single-node default for the FlashInfer fused allreduce backend was pinned ⏎ to `trtllm` to work around an `mnnvl` cudagraph capture/replay hang ⏎ (#35772). That hang was root-caused and fixed upstream in FlashInfer ⏎ ([flashinfer-ai/flashinfer#3304](https://github.com/flashinfer-ai/flashinfer/pull/3304), ⏎ released in `>= 0.6.12`; vLLM currently pins `flashinfer-python==0.6.13`), so ⏎ the workaround is no longer needed. ⏎  ⏎ This PR makes `VLLM …[truncated]

### L3-9969466a59  (L3, 2026-06-30, sha 9969466a5978, PR #46104)
TITLE: [Spec Decode] Support SWA + DFlash for MiMo (#46104)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+33/-17); tests/models/registry.py (+1/-1); vllm/model_executor/models/mimo_v2.py (+16/-3); vllm/model_executor/models/qwen3_dflash.py (+193/-4)
LABELS: ready, v1, qwen
BODY: ## Purpose ⏎  ⏎ Adds support for various combinations of causality and sliding-window for DFlash layers. ⏎ Enables all-causal SWA and all-non-causal SWA for DFlash, and sets up a model-implementation foundation from which to build out hybrid SWA+Full DFlash support in MRV2. ⏎  ⏎ Also adds support for loading the mask embedding from a file ("mask_embedding.pt") in the model checkpoint path; since the [MiMo checkpoint](https://huggingface.co/XiaomiMiMo/ …[truncated]

### L3-9a08a5118e  (L3, 2026-06-30, sha 9a08a5118e4c, PR #47164)
TITLE: fix: skip cooperative top-K on SM120 (#47164)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/sparse_attn_indexer.py (+1/-0)
LABELS: ready, verified
BODY: ## Summary ⏎  ⏎ Disable cooperative top-K on SM120, where clustered kernel launches fail with `cudaErrorInvalidValue`. SM120 will use the existing persistent top-K fallback instead. ⏎  ⏎ ## Testing ⏎  ⏎ Manually verified the failure on an SM120 GPU for cluster sizes 4, 8, and 16, and verified successful server startup with the fallback.

### L3-91055efd36  (L3, 2026-06-30, sha 91055efd363a, PR #47134)
TITLE: [XPU] C++ implementation for get_memory_info (#47134)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/platforms/xpu.py (+72/-0)
LABELS: intel-gpu, ready
BODY: Torch.xpu.mem_get_info returns wrong memory info with latest UMD. ⏎ More details please refer to https://github.com/vllm-project/vllm-xpu-kernels/blob/70387bfdcc15f85628ed31b8faf62af537096435/tests/test_get_memory_info.py#L16 ⏎ Use API of kernels to replace it. ⏎ Refer to https://github.com/jikunshang/vllm/commit/bd4f96f97fa03cf70dda72ac9098b72f3fdb8477. ⏎ Add some bounds and type check.

### L3-92c7fac640  (L3, 2026-06-30, sha 92c7fac640fa, PR #45739)
TITLE: [Perf] Restore zero-init of swizzled NVFP4 scale buffer to recover Blackwell decode throughput (#45739)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/_custom_ops.py (+7/-1)
LABELS: ready, verified
ISSUES: #45741 [Perf]: Severe NVFP4 decode throughput regression on Blackwell (B300/GB300): uninitialized swizzled scale buffer in `create_fp4_scale_tensor`
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Fixes a severe NVFP4 decode regression on Blackwell (B300 / GB300) introduced by #42988 ("[Perf] zeros -> empty to remove additional fill", commit `b29cbf0652`). ⏎  ⏎ `create_fp4_scale_tensor` (`vllm/_custom_ops.py`) allocates the **swizzled** scale-factor buffer with shape `(round_up(m, 128), round_up(n // 16, 4) // 4)`, i.e. padded in both the M (to 128) and packed-K (to 4) dimensions. The NVFP4 quant kernel (`cvt_fp16_to_fp4`) writ …[truncated]

### L3-dc148dc4d7  (L3, 2026-06-30, sha dc148dc4d763, PR #47157)
TITLE: [CI][Bugfix] Fix `Hybrid SSM NixlConnector PD prefix cache test (2 GPUs)` (#47157)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/nixl_integration/run_mamba_prefix_cache_test.sh (+2/-0)
LABELS: bug, ready, v1, kv-connector
BODY: Fix flaky CI test https://buildkite.com/vllm/ci/builds/74969/list?sid=019f1009-547c-4b3c-9eae-3e500fda953a&tab=output, in which the output of two identical requests might be ~slightly different, causing the assertion to trigger. ⏎  ⏎ I currently suspect this might be due to a recent change to FA https://github.com/vllm-project/vllm/pull/36701, which should still be the default on hopper ⏎ ``` ⏎ [2026-06-28T21:10:21Z] ^[[0;36m(EngineCore pid=954)^[[0; …[truncated]

### L3-c5200d3565  (L3, 2026-07-01, sha c5200d3565e6, PR #46076)
TITLE: [Attention][DSA] support dcp for FLASHINFER_MLA_SPARSE (#46076)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.flashinfer_sparse, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+104/-17); vllm/v1/attention/backends/mla/indexer.py (+176/-12); vllm/v1/attention/backends/mla/sparse_utils.py (+158/-11); vllm/v1/attention/backends/utils.py (+12/-12); vllm/v1/attention/ops/common.py (+1/-0); docs/design/attention_backends.md (+1/-1); tests/v1/attention/test_indexer_dcp_localize.py (+951/-0); vllm/model_executor/kernels/attention/__init__.py (+2/-0); vllm/model_executor/kernels/attention/dsa/__init__.py (+2/-0); vllm/model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py (+420/-0); (+2 more)
LABELS: documentation, ready, v1, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ add dcp support for GLM 5.2 ⏎  ⏎  ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824 ⏎ vllm serve zai-org/GLM-5.2-FP8 \ ⏎     --host 0.0.0.0 \ ⏎     --port 8000 \ ⏎     --kv-cache-dtype fp8_e4m3 \ ⏎     --tensor-parallel-size 8 \ ⏎     --decode-context-parallel-size 4 \ ⏎     --tool-call-parser glm47 \ ⏎     --enable-auto-tool-choice \ ⏎     --reasoning-parser glm45 ⏎ ``` ⏎  ⏎ logs ⏎ ``` ⏎ DCP 4, TP 8 ⏎ (EngineCore pid=88967) INF …[truncated]

### L3-f5a8d73377  (L3, 2026-07-01, sha f5a8d73377d0, PR #46995)
TITLE: [Spec Decode] DSpark (#46995)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/sparse_swa.py (+139/-23); tests/models/registry.py (+13/-0); tests/models/test_registry.py (+4/-0); tests/v1/attention/test_dspark_noncausal_sparse_mla.py (+529/-0); tests/v1/e2e/spec_decode/test_spec_decode.py (+61/-0); vllm/benchmarks/datasets/datasets.py (+1/-0); vllm/config/speculative.py (+36/-3); vllm/config/vllm.py (+24/-5); vllm/model_executor/models/qwen3_dflash.py (+9/-3); vllm/model_executor/models/qwen3_dspark.py (+153/-0); (+14 more)
LABELS: performance, new-model, speculative-decoding, ready, v1, qwen
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Adds support for [DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark) speculative decoding, using both DeepSeek-V4 DSpark models as well as [DeepSeek-trained Qwen3-DSpark](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7) models. ⏎  ⏎ ## Design ⏎  ⏎ DSpark uses non-causal sliding-window attention. To implement this, instead of manually reimplementing the MLA attention, we instead utilize the existing SparseMLA ba …[truncated]

### L3-024b06b0dc  (L3, 2026-07-01, sha 024b06b0dc0c, PR #42748)
TITLE: [Bugfix] Expose usage field in GenerateResponse for disaggregated serving (#42748)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/entrypoints/scale_out/token_in_token_out/protocol.py (+3/-1); vllm/entrypoints/scale_out/token_in_token_out/serving.py (+1/-1)
LABELS: bug, frontend, ready
BODY: ## Summary ⏎  ⏎ `serving.py` already constructs `UsageInfo` (including `prompt_tokens_details.cached_tokens` when `--enable-prompt-tokens-details` is set) and passes it to the `GenerateResponse` constructor. However, the `usage` field was missing from the `GenerateResponse` model, so pydantic silently dropped it. ⏎  ⏎ **Fix:** add `usage: UsageInfo | None = None` to `GenerateResponse`, mirroring `GenerateStreamResponse` which already has this field. …[truncated]

### L3-4787f2dd1b  (L3, 2026-07-01, sha 4787f2dd1b57, PR #47305)
TITLE: [Bugfix] Don't read KV cache past `seq_len` in triton paged attn kernels (#47305)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.triton.chunked_prefill_paged_decode
FILES: vllm/v1/attention/ops/chunked_prefill_paged_decode.py (+7/-2); vllm/v1/attention/ops/triton_attention_helpers.py (+6/-4)
LABELS: bug, ready, v1
BODY: The Triton unified_attention and ROCm paged-decode kernels load KV cache slots beyond seq_len (the unwritten tail of the last partial block) and rely on the score mask to drop them. That is unsafe when the tail is non-finite: masked positions get softmax weight 0, but 0 * NaN = NaN still poisons the output. In `compute_tile_loop_bounds` the causal-derived `max_seq_prefix_len` can additionally overshoot seq_len for non-causal (cross-)attention, wi …[truncated]

### L3-fa248139a0  (L3, 2026-07-01, sha fa248139a020, PR #45723)
TITLE: [MoE] Plumb gemm1_alpha/beta/clamp_limit into TRT-LLM FP8 MoE (#45723)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+46/-3); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_mxfp8.py (+2/-0); vllm/model_executor/layers/quantization/online/mxfp8.py (+2/-0); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+2/-0)
LABELS: ready, nvidia
BODY: ## Purpose ⏎  ⏎ The FlashInfer FP8 block-scale MoE kernels — `trtllm_fp8_block_scale_moe` and `trtllm_fp8_block_scale_routed_moe` — accept three optional per-expert SwiGLU parameters (`gemm1_alpha`, `gemm1_beta`, `gemm1_clamp_limit`) that realize the OAI SwiGLU variant (`X2 * sigmoid(alpha*X2) * (X1 + beta)` with clamping) for MXFP8. The FP8 TRT-LLM experts did not pass them, so MXFP8 models using clamped/OAI SwiGLU (e.g. MiniMax-M3) could not run co …[truncated]

### L3-e196268bad  (L3, 2026-07-01, sha e196268bade5, PR #47338)
TITLE: [Docker] Remove unused Dockerfile.nightly_torch (#47338)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.nightly_torch (+0/-326)
LABELS: ci/build
BODY: ## Purpose ⏎ Remove `docker/Dockerfile.nightly_torch`, which is dead/unreferenced. ⏎  ⏎ ## Why it's safe ⏎ `Dockerfile.nightly_torch` (added in #16936) is not referenced anywhere: ⏎ - vLLM repo: no hits in `.buildkite/` pipelines, `docker/*.hcl` bake files, `image_build*.sh` scripts, `.github/workflows`, or docs. ⏎ - `vllm-project/ci-infra`: no references (`ci.hcl`, `bake.hcl`, `bootstrap.sh`, code search all empty). ⏎  ⏎ Building vLLM against torch nightly is h …[truncated]

### L3-ed41aa270a  (L3, 2026-07-01, sha ed41aa270a9e, PR #43950)
TITLE: [ROCm][DSV4] Use aiter mHC pre/post as the default ROCm path (#43950)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mhc.py (+24/-32); vllm/models/deepseek_v4/amd/model.py (+6/-4); vllm/models/deepseek_v4/amd/mtp.py (+2/-3)
LABELS: rocm, ready, ci/build, verified
BODY: ## Purpose ⏎  ⏎ Re-enable the **aiter** multi-head-consensus (mHC) pre/post ops as the preferred ROCm path for DeepSeek V4. #43679 introduced the tilelang fused post+pre mHC kernel and explicitly left a hook to switch back to the (faster) aiter mHC kernels once an aiter release containing the `mhc_pre_gemm_sqrsum_kernel` race-condition fix was available. This is that follow-up. ⏎  ⏎ ### Dispatch / fallback path ⏎  ⏎ Selection is purely **capability based — n …[truncated]

### L3-2b753ad200  (L3, 2026-07-01, sha 2b753ad200d5, PR #47093)
TITLE: [Spec Decode] DSpark speculators checkpoint support (#47093)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_dspark.py (+37/-5); vllm/models/deepseek_v4/nvidia/dspark.py (+11/-0); vllm/transformers_utils/configs/speculators/algos.py (+44/-0); vllm/v1/worker/gpu/spec_decode/dspark/speculator.py (+48/-11)
LABELS: performance, new-model, ready, v1, qwen
BODY: Validated with [Qwen3-8B](https://huggingface.co/mgoin/Qwen3-8B-speculator.dspark-reasoning) and [GLM 5.2 DSpark (preview)](https://huggingface.co/mgoin/GLM-5.2-speculator.dspark-preview#greedy-temperature0) ⏎  ⏎ ``` ⏎ vllm serve zai-org/GLM-5.2-FP8 -tp 8 \ ⏎   --speculative-config '{"method":"dspark","model":"mgoin/GLM-5.2-speculator.dspark-preview","num_speculative_tokens":7,"attention_backend":"FLASH_ATTN","draft_sample_method":"probabilistic"}' ⏎  …[truncated]

### L3-2665ed704b  (L3, 2026-07-02, sha 2665ed704b04, PR #46838)
TITLE: [Bugfix][Kernel] Correct FlashInfer CUTLASS MoE tuning token bound (#46838)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py (+0/-2)
LABELS: bug, ready, nvidia, verified
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Purpose ⏎  ⏎ Previously, #34542 refactored the FlashInfer CUTLASS fused MoE backend and passed the CUDA graph max capture size (typically 512) as `tune_max_num_tokens` for flashinfer_cutlass_fused_moe. However, `max_capture_size` describes graph capture capacity and can be unrelated to the scheduler/MoE max batched token limit, so using it can select a tuning configuration for the wrong token range and thus cause suboptimal perf unexpectedly. ⏎  …[truncated]

### L3-a47f38f825  (L3, 2026-07-02, sha a47f38f82569, PR #47383)
TITLE: [Bugfix][Model Runner V2][Spec Decode] Fix int32 offset overflow in block verification kernels (#47383)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_rejection_sampler_i64.py (+144/-0); vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py (+4/-4)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ `_compute_cumulative_log_p_kernel` and `_compute_local_residual_mass_kernel` (used when `rejection_sample_method="block"`) multiply int32 logit/request-state indices by vocab-scale strides before int64 promotion. With a GLM-scale vocab (~155k tokens) the byte offset wraps once `logit_idx * vocab >= 2**31` (`logit_idx >= ~13.8k`, reachable with large `max_num_seqs` × `1 + num_speculative_tokens`), or once `req_state_idx * num_speculati …[truncated]

### L3-d715b3aa1e  (L3, 2026-07-02, sha d715b3aa1ea6, PR #47361)
TITLE: Delete PagedAttention (#47361)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+0/-2); csrc/libtorch_stable/attention/paged_attention_v1.cu (+0/-190); csrc/libtorch_stable/attention/paged_attention_v2.cu (+0/-202); vllm/_custom_ops.py (+0/-94); benchmarks/kernels/benchmark_paged_attention.py (+39/-94); csrc/libtorch_stable/attention/attention_kernels.cuh (+0/-667); csrc/libtorch_stable/ops.h (+0/-26); csrc/libtorch_stable/torch_bindings.cpp (+0/-30); tests/kernels/attention/test_attention.py (+51/-167)
LABELS: performance, ready, ci/build
BODY: ## Purpose ⏎  ⏎ It is time. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-d29125c085  (L3, 2026-07-02, sha d29125c0852e, PR #43232)
TITLE: Xqa decode kernels (#43232)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+37/-18); vllm/v1/attention/backends/flashinfer.py (+286/-92); docs/design/attention_backends.md (+3/-2); tests/kernels/attention/test_use_trtllm_attention.py (+16/-1); tests/v1/attention/test_attention_backends.py (+120/-4); tests/v1/attention/test_trtllm_attention_integration.py (+6/-3); tools/pre_commit/generate_attention_backend_docs.py (+70/-23)
LABELS: documentation, performance, ready, v1, nvidia, verified
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:3dd910da42 (fix PR 47908) || deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Enable the FlashInfer TRTLLM/XQA decode path on Hopper (SM90) for vLLM V1 FlashInfer attention. ⏎  ⏎ This makes TRTLLM attention support phase-aware, so decode can use FlashInfer's `trtllm_batch_decode_with_kv_cache` XQA path while prefill can remain on the existing FlashInfer native path when TRTLLM prefill is not supported. The branch also tracks prefill and decode query dtypes separately, since SM90 XQA decode requires BF16/FP16 qu …[truncated]

### L3-09663abde0  (L3, 2026-07-02, sha 09663abde0f5, PR #44977)
TITLE: [ROCm][MLA] Fuse MLA q/kv RMSNorm + FP8 per-token quant in the FP8 attention path (#44977)
SOURCES: path_integration+keyword, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/compilation/passes/fusion/rocm_aiter_fusion.py (+135/-0); docs/design/fusions.md (+25/-4); tests/compile/passes/test_fuse_mla_dual_rms_norm.py (+143/-1); vllm/_aiter_ops.py (+68/-0)
LABELS: documentation, rocm, ready
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ The `MLADualRMSNormFusionPass` fuses the paired `q_a_layernorm` + ⏎ `kv_a_layernorm` of MLA attention into AITER's `fused_qk_rmsnorm` HIP kernel, but ⏎ that fusion silently stops firing once attention is **FP8-quantized** — the ⏎ earlier `RocmAiterRMSNormQuantFusionPass` folds the q-side ⏎ `rms_norm → fp8 per-token quant` into a single ⏎ `rocm_aiter_rmsnorm_fused_dynamic_quant` op, so the q latent is no longer a bare ⏎ `rms_norm` and the  …[truncated]

### L3-84b9c2762f  (L3, 2026-07-02, sha 84b9c2762f5f, PR #47304)
TITLE: Update DeepGEMM tag to point to latest nv-dev branch for sm120 support (#47304)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: cmake/external_projects/deepgemm.cmake (+2/-1); tools/install_deepgemm.sh (+2/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ This PR https://github.com/vllm-project/vllm/pull/43477 landed in vLLM without bumping up our commit tag for DeepGEMM for the SM120 support, which resulted in many failures for models that trigger DeepGEMM kernels i.e. https://github.com/vllm-project/vllm/issues/47169, https://github.com/vllm-project/vllm/issues/47266 ⏎  ⏎ For now we should bump the DeepGEMM commit for nv-dev (which includes https://github.com/deepseek-ai/DeepGEMM/pul …[truncated]

### L3-de2a8fc042  (L3, 2026-07-02, sha de2a8fc042b6, PR #47128)
TITLE: [ROCm] [PyTorch] Move to stable abi since ROCm upgraded to torch 2.11 (#47128)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+17/-29); csrc/cuda_view.cu (+0/-60); csrc/libtorch_stable/ops.h (+3/-4); csrc/libtorch_stable/torch_bindings.cpp (+2/-7); csrc/ops.h (+0/-3); csrc/torch_bindings.cpp (+0/-28)
LABELS: rocm, ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ Since ROCm has upgraded to torch 2.11 in PR https://github.com/vllm-project/vllm/pull/45362 , this PR follow up to clean up the previous fallback loop https://github.com/vllm-project/vllm/pull/44648 . So we are moving back to stable_abi. ⏎  ⏎ ## Test Plan ⏎  ⏎  ⏎  ⏎ ## Test Result ⏎  ⏎ - `pytest -svvvvv tests/kernels/core/test_uva.py` ⏎  ⏎ ``` ⏎ ======================== 4 passed, 16 warnings in 1.43s ======================== ⏎ ``` ⏎  ⏎ - `python3 …[truncated]

### L3-a2f713002d  (L3, 2026-07-02, sha a2f713002df9, PR #44443)
TITLE: [ModelRunner V2] Enable by default for all dense models (#44443)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/model_runner_v2.yaml (+1/-3); tests/distributed/test_multiproc_executor.py (+8/-3); tests/distributed/test_ray_v2_executor.py (+7/-1); tests/test_config.py (+23/-1); tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py (+8/-4); vllm/config/vllm.py (+8/-5)
LABELS: rocm, ready, ci/build, v1, multi-modality, kv-connector, ready-run-all-tests, mrv2
BODY: ## Purpose ⏎  ⏎ Part of https://github.com/vllm-project/vllm/issues/41286 ⏎  ⏎ ## Test ⏎  ⏎ Covered in CI

### L3-41de1380c2  (L3, 2026-07-02, sha 41de1380c23d, PR #47485)
TITLE: [BugFix] Derive FlashInfer Q dtype from resolved per-group builder state (#47485)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+21/-19)
LABELS: bug, ready, v1, ci-failure, nvidia
BODY: Fixes the Quantization CI failures from #43232: `get_q_data_type` decides from the global config, so FP8-Q gets selected on archs with no fp8 tensor-core path (L4 crashes with `fp8 tensor core is not supported in fa2 backend`) and for unquantized `--kv-cache-dtype-skip-layers` groups. Use the builder's group-aware `self.cache_dtype` instead and only quantize Q on SM90/SM100. ⏎  ⏎ Verified with the previously-failing `tests/quantization/test_fp8.py` t …[truncated]

### L3-979f5511d7  (L3, 2026-07-02, sha 979f5511d78b, PR #47217)
TITLE: [Bugfix][Gemma4] Keep image bidirectional attention within the sliding window (#47217)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.triton.unified_attention, L3.triton.v1_backend
FILES: vllm/model_executor/layers/attention/attention.py (+4/-0); vllm/v1/attention/backends/flash_attn.py (+32/-4); vllm/v1/attention/backends/triton_attn.py (+3/-0); vllm/v1/attention/ops/triton_attention_helpers.py (+13/-2); vllm/v1/attention/ops/triton_unified_attention.py (+9/-0); vllm/model_executor/models/gemma4.py (+7/-0); vllm/model_executor/models/gemma4_mm.py (+7/-0); vllm/v1/worker/gpu_model_runner.py (+14/-3)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ Gemma4 vision models that use blockwise bidirectional image attention ⏎ (`use_bidirectional_attention="vision"` → 12B, 26B-A4B, 31B) silently fell back ⏎ to **causal-only** attention over image tokens whenever a single image's ⏎ soft-token span exceeded the text sliding window (e.g. ~1100 soft tokens at ⏎ `max_soft_tokens=1120` vs `sliding_window=1024`). This degraded vision/OCR ⏎ quality — catastrophic on the encoder-free **12B** (no vi …[truncated]

### L3-1f486d96a1  (L3, 2026-07-03, sha 1f486d96a173, PR #47102)
TITLE: Add Triton Backend for Unlimited-OCR R-SWA (#47102)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.triton.unified_attention, L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+23/-0); vllm/v1/attention/ops/triton_attention_helpers.py (+10/-1); vllm/v1/attention/ops/triton_unified_attention.py (+16/-1); vllm/model_executor/models/config.py (+14/-6); vllm/model_executor/models/unlimited_ocr.py (+5/-9)
LABELS: ready, v1
BODY: ## Summary ⏎  ⏎ This PR adds explicit TritonAttention support for Unlimited-OCR's R-SWA decode ⏎ mask. The Triton path implements the mask with: ⏎  ⏎ ```python ⏎ if USE_R_SWA: ⏎     prefix_len = tl.load(rswa_prefix_lens_ptr + seq_idx) ⏎     in_prefix = seq_offset[None, :] < prefix_len ⏎     in_window = (query_abs_pos - seq_offset) < R_SWA_WINDOW ⏎     seq_mask = seq_mask & (in_prefix | in_window) ⏎ ``` ⏎  ⏎ This keeps prompt/image prefix tokens globally visib …[truncated]

### L3-f63dca6838  (L3, 2026-07-03, sha f63dca68385c, PR #47035)
TITLE: [ROCm] Fix encoder-decoder cross-attention KV layout aliasing (#47035)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L3.rocm.aiter_unified
FILES: vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+56/-41); vllm/v1/worker/gpu_model_runner.py (+33/-1); tests/entrypoints/speech_to_text/transcription/test_transcription_validation_whisper.py (+2/-1)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎ Encoder-decoder models (Whisper, Nemotron-Parse, ...) share one raw KV allocation between decoder self-attention (`ROCM_ATTN`, K/V-first) and cross-attention (`AITER`/`TRITON`, blocks-first). The mismatched stride conventions over the same bytes cause aliasing → garbage / repeated-token output. ⏎  ⏎ Changes: ⏎ - Reconcile layouts **once at allocation** (`gpu_model_runner.py`): when shared attention groups disagree on layout, normalize K/ …[truncated]

### L3-07516fda67  (L3, 2026-07-03, sha 07516fda67d2, PR #45953)
TITLE: [MRV2][SD] Make Dynamic SD comatible with Full Cuda Graphs (#45953)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/features/speculative_decoding/dynamic_speculative_decoding.md (+2/-5); tests/test_config.py (+31/-0); tests/v1/spec_decode/test_dynamic_sd_cug.py (+328/-0); vllm/config/compilation.py (+6/-2); vllm/config/vllm.py (+3/-4); vllm/v1/worker/gpu/cudagraph_utils.py (+58/-19); vllm/v1/worker/gpu/model_runner.py (+4/-3); vllm/v1/worker/gpu_model_runner.py (+4/-3)
LABELS: documentation, speculative-decoding, ready, v1, nvidia
BODY: Followup to DSD: https://github.com/vllm-project/vllm/pull/32374 which ran only with MRv1 and Piecewise graph. ⏎  ⏎ This PR mainly enables DSD with MRv2 + FCG. DSD + MRv1 is still on PW. ⏎  ⏎ Currently, we capture FCG for decode only for given `K` say `K=3` which meant only multiple of `K+1=4` size of graph are captured. However, in DSD the chosen optimal draft `k` can be `<K` during scheduling which would lead to missing the graph. This PR captures  …[truncated]

### L3-4c3c17d43b  (L3, 2026-07-04, sha 4c3c17d43bbd, PR #47567)
TITLE: [ROCm] Disable persistent sparse-MLA kernel for chunked-prefill continuations (#47567)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+12/-2)
LABELS: rocm, ready, v1
ISSUES: #47042 [Bug][ROCm] GLM-5.2-FP8 sparse MLA decode degenerates at long context on gfx942 (MI325X)
BODY: # [ROCm] Disable persistent sparse-MLA kernel for chunked-prefill continuations ⏎  ⏎ ## Summary ⏎ Fixes #47042. GLM-4.6/4.5 / DeepSeek-V3.2-style sparse-MLA (DSA) models on ROCm ⏎ produce correct output up to ~20K tokens and then collapse into repetition / ⏎ garbage at longer contexts. ⏎  ⏎ Root cause: the AITER **persistent MLA work-stealing kernel** ⏎ (`get_mla_metadata_v1` + the `work_meta_data` persistent path in ⏎ `aiter.mla.mla_decode_fwd`, enabled unconditi …[truncated]

### L3-67ff0ae30f  (L3, 2026-07-04, sha 67ff0ae30fe6, PR #42890)
TITLE: Support nvfp4 kv with kv-cache-dtype-skip-layers sliding_window (#42890)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/attention.py (+51/-1); docs/features/quantization/quantized_kvcache.md (+26/-0); vllm/config/cache.py (+6/-0); vllm/platforms/interface.py (+117/-0); vllm/v1/worker/gpu/attn_utils.py (+18/-2); vllm/v1/worker/gpu_model_runner.py (+9/-1)
LABELS: documentation, ready, v1, tool-calling
BODY: ## Purpose ⏎ NVFP4 kv does not support `--kv-cache-dtype-skip-layers sliding_window` before this PR ⏎ - [reverted] Remove the padding and scaling logic of kv cache in mamba ⏎ - Update the kv cache scaling and padding logic in `unify_kv_cache_spec_page_size`. New logic: ⏎   - Find the "primary" by smallest bytes per token in all AttentionSpec ⏎   - Set the target page byte size as a multiple of primary page bytes that barely larger than maximum page by …[truncated]

### L3-26eb87204d  (L3, 2026-07-04, sha 26eb87204d58, PR #45844)
TITLE: [Bugfix] Fix CPU split-KV scratchpad sizing (#45844)
SOURCES: path_core, subject_keyword, corpus:kernel-correctness-cases
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+1/-1); csrc/cpu/cpu_attn_impl.hpp (+56/-14)
LABELS: bug, ready, v1, cpu, verified
DEEP_STUDY: deep-study correctness case vllm:26eb87204d: class=memory_safety_oob; symptom=hang_deadlock; introducing=unknown
BODY: ## Purpose ⏎  ⏎ Fix a CPU split-KV hang seen with `google/gemma-4-26B-A4B-it`. ⏎  ⏎ The worker could get stuck in `reduce_splits` waiting for a split completion flag. The root cause was the split-KV scratchpad was sized using the final output element size / smaller fallback q-head count, while the runtime split path writes temporary split outputs as `float` and can process more q-heads per split. ⏎  ⏎ This could under-size the split-KV reduction buffer …[truncated]

### L3-f1445f6dbd  (L3, 2026-07-04, sha f1445f6dbd34, PR #47551)
TITLE: [CI] Bump `huggingface-hub` from `v1.10.2` to `v1.22.0` (#47551)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+2/-2); .buildkite/test_areas/misc.yaml (+1/-1); requirements/test/cpu.txt (+4/-4); requirements/test/cuda.txt (+4/-4); requirements/test/rocm.txt (+4/-4); requirements/test/xpu.txt (+4/-4); tests/benchmarks/test_bfcl_dataset.py (+1/-1); tests/benchmarks/test_custom_dataset_seed.py (+1/-1); tests/benchmarks/test_random_dataset.py (+1/-1); tests/benchmarks/test_random_multimodal_dataset_video.py (+1/-1); (+15 more)
LABELS: performance, structured-output, ready, ci/build, v1, tool-calling, cpu, nvidia
BODY: This release includes https://github.com/huggingface/huggingface_hub/pull/4394:  ⏎  ⏎ > `snapshot_download` now caches a repository's file listing on disk under a new `trees/` folder, so re-downloading a commit that's already cached costs a single network call — resolving the branch or tag to a commit hash — instead of one metadata request per file. The listing is immutable per commit and shared by both `snapshot_download` and `hf_hub_download`; for  …[truncated]

### L3-4a6bf3c77f  (L3, 2026-07-04, sha 4a6bf3c77f05, PR #47519)
TITLE: [ROCm][CI] Fix Kernels and Kernels attention test failures (#47519)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_rocm_aiter_mla_decode_metadata.py (+1/-1); tests/kernels/moe/test_ocp_mx_moe.py (+10/-1)
LABELS: rocm, ready
BODY: ## Purpose ⏎  ⏎ This PR fixes failures on ROCm tests MI355 Kernels and Kernels Attention. The tests were failing due to a clone operation being attempted on the `prefill_backend` which was set to None for testing. ⏎  ⏎ This PR also fixes a failure on the Kernels MoE test (`test_ocp_mx_moe.py`), which was failing due to the distributed environment not being initialized and an assertion failing due to `rocm_aiter_ops.is_enabled()` returning False. Note …[truncated]

### L3-34b560b725  (L3, 2026-07-04, sha 34b560b725b4, PR #47332)
TITLE: [Bugfix][Gemma4] Fix FA4 mm_prefix mask: add sliding window and absolute q_idx (#47332)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+90/-50)
LABELS: bug, ready, v1
ISSUES: #47300 [Bug]: [Gemma4] gibberish output for long inputs with images on SM90 FlashAttn4
BODY: ### Purpose ⏎  ⏎ Fixes https://github.com/vllm-project/vllm/issues/47300 ⏎  ⏎ Gemma4 vision models emit total gibberish on the FlashAttention-4 (CuTeDSL, SM90) backend when the context is long (>=~1k tokens past the sliding window) and contains at least one image. The Triton backend produces correct output on the same input. ⏎  ⏎ Root cause — two bugs in `_make_mm_prefix_mask_mod` (`vllm/v1/attention/backends/flash_attn.py`): ⏎  ⏎ - Missing sliding windo …[truncated]

### L3-1a308c449c  (L3, 2026-07-04, sha 1a308c449c93, PR #43645)
TITLE: [XPU] Add W8A8 FP8 linear kernel with multi-granularity quant support (#43645)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/intel_jobs/test-intel.yaml (+48/-1); docs/features/quantization/online.md (+2/-0); vllm/config/kernel.py (+6/-1); vllm/model_executor/kernels/linear/__init__.py (+12/-3); vllm/model_executor/kernels/linear/scaled_mm/xpu.py (+107/-4); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py (+13/-3)
LABELS: documentation, intel-gpu, ready, ci/build
BODY: ## Purpose ⏎ This PR adds `XPUW8A8FP8LinearKernel` and `XPUW8A16FP8LinearKernel` for XPU, ⏎ replacing the previous `XPUFP8ScaledMMLinearKernel` with proper support for ⏎ multiple weight/activation quantization granularities. ⏎  ⏎ ### Changes ⏎ - Add `XPUW8A8FP8LinearKernel` and `XPUW8A16FP8LinearKernel`, replacing `XPUFP8ScaledMMLinearKernel` ⏎ - Select W8A8 or W8A16 kernel based on whether activation quantization is present ⏎ - Ensure dynamic activation …[truncated]

### L3-d2afe39647  (L3, 2026-07-04, sha d2afe39647d2, PR #47597)
TITLE: [Bugfix][Frontend] Preserve default sampling params in batch chat (#47597)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/entrypoints/openai/chat_completion/protocol.py (+4/-4)
LABELS: bug, frontend, ready
BODY: ## Purpose ⏎  ⏎ `/v1/chat/completions/batch` converts each conversation into a `ChatCompletionRequest`, but the conversion currently dumps Pydantic defaults too. A batch request that omits sampling params still forwards `temperature=0.7`, `top_p=1.0`, `min_p=0.0`, and `repetition_penalty=1.0`, so it silently overrides model or server default sampling params that a normal `/v1/chat/completions` request would use. ⏎  ⏎ This keeps only fields explicitly …[truncated]

### L3-8974ed89cd  (L3, 2026-07-05, sha 8974ed89cd0b, PR #44461)
TITLE: [Bugfix][Voxtral Realtime] Fix token feedback timeout silent hang (#44461)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/voxtral_realtime.py (+3/-6)
LABELS: bug, ready, multi-modality, mistral
ISSUES: #35863 [Bug]: Voxtral-Realtime stops returning transcribed text starting from the 3rd concurrent session | #36015 [Bug]: Realtime audio transcription (Voxtral) silently hangs after ~10 minutes due to unhandled TimeoutError in background task
BODY: ## Purpose ⏎  ⏎ Closes #36015. ⏎ Closes #35863. ⏎  ⏎ Voxtral realtime has a background `feed_tokens()` task that moves engine output tokens from `input_stream` into the realtime transcription buffer. That task currently wraps `input_stream.get()` in `asyncio.wait_for(..., VLLM_ENGINE_ITERATION_TIMEOUT_S)`. A normal idle interval with no engine output can therefore raise `TimeoutError` and kill the feedback task while the websocket session remains open …[truncated]

### L3-cc1d020d01  (L3, 2026-07-05, sha cc1d020d0194, PR #46942)
TITLE: [MRV2] Enable mm prefix bidi attention support on MRV2 (#46942)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/models_multimodal.yaml (+2/-1); tests/models/multimodal/generation/test_mm_prefix_lm.py (+119/-0); vllm/config/model.py (+1/-1); vllm/v1/worker/gpu/attn_utils.py (+27/-0); vllm/v1/worker/gpu/model_states/default.py (+16/-1)
LABELS: ready, ci/build, v1, multi-modality
BODY: ## Purpose ⏎ - Wire mm prefix attn mask computation to MRV2 ⏎ - Actually, `tests/models/multimodal/generation/test_common.py`'s gemma3 test can't catch the difference of with/without mm_prefixlm attention, because their logprobs are still quite similar. ⏎ - This PR adds test to cover this regression. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ VLLM_USE_V2_MODEL_RUNNER=1 pytest -s -v tests/models/multimodal/generation/test_mm_prefix_lm.py ⏎ ``` ⏎  ⏎ ## Test Result ⏎ Test shou …[truncated]

### L3-95a248faed  (L3, 2026-07-05, sha 95a248faed67, PR #47433)
TITLE: [Attention Backend] HPC_ATTN backend support mtp and dynamic scheduled attention (#47433)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/hpc_attn.py (+100/-15); docs/design/attention_backends.md (+1/-1); vllm/model_executor/layers/hpc/rope_norm.py (+22/-14)
LABELS: documentation, v1, verified
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-16f8110935  (L3, 2026-07-06, sha 16f811093570, PR #47532)
TITLE: [Bugfix][CPU][RISC-V] Fix VLEN detection for RVV attention path (#47532)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+13/-16); cmake/cpu_extension.cmake (+11/-1); csrc/cpu/cpu_attn.cpp (+2/-1)
LABELS: bug, ready, ci/build, v1, cpu
BODY: ## Purpose ⏎ Three bugs in the RISC-V VLEN detection chain caused the RVV-optimized attention kernel to be silently disabled or the build to fail: ⏎  ⏎ 1. cpu_attn_has_isa("rvv") only checked __riscv_v_min_vlen == 128, excluding VLEN=256 hardware (e.g. Spacemit X100). This is inconsistent with cpu_attn_rvv.hpp which supports both 128 and 256. ⏎  ⏎ 2. CMake VLEN auto-detection read /proc/cpuinfo unconditionally, which describes the build host — not the …[truncated]

### L3-598d51153a  (L3, 2026-07-06, sha 598d51153a15, PR #47589)
TITLE: [Bugfix][Distributed] Delegate MNNVL allreduce one-shot selection (#47589)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/passes/distributed/test_fusion_all_reduce.py (+30/-0); vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+30/-13)
LABELS: bug, ready
ISSUES: #47284 mnnvl allreduce buffer size error on single-node TP8 after #47219
BODY: ## Purpose ⏎  ⏎ Fixes #47284. ⏎  ⏎ After #47219, vLLM can default FlashInfer fused allreduce/RMSNorm to the `mnnvl` workspace backend on single-node setups. The fusion call still forced `use_oneshot` from vLLM's legacy per-rank threshold table, which can disagree with FlashInfer MNNVL's AUTO workspace sizing. When vLLM forces `use_oneshot=True` for a shape that FlashInfer sized for AUTO/two-shot, FlashInfer reports an insufficient workspace. ⏎  ⏎ This PR kee …[truncated]

### L3-095adf1fdc  (L3, 2026-07-06, sha 095adf1fdc93, PR #47671)
TITLE: [Bugfix] Fix int32 overflow in triton_decode_attention page offsets (#47671)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.triton.decode_attention
FILES: vllm/v1/attention/ops/triton_decode_attention.py (+2/-2)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ `_fwd_kernel_stage1` / `_fwd_grouped_kernel_stage1` load `kv_page_number` as int32 and multiply it by the KV buffer page stride in int32. With MLA c_kv (576 dims) and PAGE_SIZE 480 the page stride is ~276k, so page numbers above ~7.7k overflow to negative offsets → CUDA illegal memory access. Hybrid models hit this quickly: all layer groups share one block pool, so a 90GB KV buffer already holds ~163k pages. The error is async, so t …[truncated]

### L3-f2aaf59151  (L3, 2026-07-06, sha f2aaf5915102, PR #44880)
TITLE: [Feature] Support MTP speculative decoding for Bailing hybrid models (#44880)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/config/test_bailing_mtp_config.py (+52/-0); tests/models/registry.py (+6/-0); tests/v1/attention/test_linear_attention_metadata_builder.py (+188/-0); vllm/config/speculative.py (+16/-0); vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py (+322/-3); vllm/model_executor/models/bailing_moe_linear.py (+4/-0); vllm/model_executor/models/bailing_moe_mtp.py (+380/-0); vllm/model_executor/models/registry.py (+1/-0); vllm/transformers_utils/model_arch_config_convertor.py (+7/-0); vllm/v1/attention/backends/linear_attn.py (+212/-1); (+1 more)
LABELS: new-model, speculative-decoding, ready, v1, verified
BODY: ## Purpose ⏎  ⏎ Add MTP (Multi-Token Prediction) speculative decoding support for Bailing MoE hybrid models, and enhance the linear attention backend to support speculative decode with CUDAGraph. ⏎  ⏎ **Key changes:** ⏎  ⏎ - **New `BailingMoeV25MTPModel`**: Add the draft model implementation for Bailing MoE V2.5 MTP speculative decoding (`bailing_moe_mtp.py`), including `BailingMoeV25MultiTokenPredictor` with shared embedding/lm_head support and MoE ex …[truncated]

### L3-f70caef48b  (L3, 2026-07-06, sha f70caef48b92, PR #47474)
TITLE: [Perf] Cache `token_to_req_indices` for dsv4, 5x~6x kernel performance improvement (#47474)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+27/-0); vllm/v1/attention/backends/mla/sparse_swa.py (+3/-5); vllm/models/deepseek_v4/compressor.py (+3/-6); vllm/models/deepseek_v4/sparse_mla.py (+1/-14)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ So we save two additional copy from CPU to GPU ⏎  ⏎ ## Test ⏎  ⏎ ### Acc ⏎  ⏎ ```bash ⏎ vllm serve deepseek-ai/DeepSeek-V4-Flash   --trust-remote-code   --kv-cache-dtype fp8   --block-size 256   --enable-expert-parallel   --tensor-parallel-size 4   --attention_config.use_fp4_indexer_cache=True   --moe-backend deep_gemm_mega_moe   --tokenizer-mode deepseek_v4   --tool-call-parser deepseek_v4   --enable-auto-tool-choice   --reasoning-parser  …[truncated]

### L3-740f379fae  (L3, 2026-07-06, sha 740f379faefd, PR #46065)
TITLE: [ROCm][AITER] Directly Implement AITER Custom All-reduce in CudaCommunicator (#46065)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: .buildkite/test-amd.yaml (+5/-1); tests/compile/fusions_e2e/conftest.py (+1/-0); tests/compile/passes/distributed/test_fusion_all_reduce.py (+9/-1); tests/distributed/test_rocm_aiter_custom_ar.py (+134/-0); tests/utils.py (+40/-0); tests/v1/e2e/general/test_rocm_aiter_custom_ar.py (+118/-0); vllm/_aiter_ops.py (+34/-95); vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+10/-39); vllm/config/vllm.py (+7/-2); vllm/distributed/device_communicators/aiter_custom_all_reduce.py (+95/-0); (+2 more)
LABELS: rocm, ready, ci/build, v1, nvidia, verified
BODY: ## Purpose ⏎ Previously, with the AITER fused AR-RMSNorm enabled, two separate custom-allreduce instances were live: vLLM's `CustomAllreduce` for the plain allreduce, and a standalone AITER `CustomAllreduce` (`rocm_aiter_ops._CUSTOM_ALL_REDUCE`) for the fused op, each with its own IPC buffers, both registered during CUDA graph capture. Performance was the main reason it's done this way rather than by dropping the AITER path directly in the `CudaCo …[truncated]

### L3-b1384f5ec6  (L3, 2026-07-06, sha b1384f5ec6b0, PR #43328)
TITLE: Enable B12x backend for non-gated MoEs (like Nemotron)  (#43328)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer_b12x_moe.py (+157/-22); tests/kernels/moe/utils.py (+2/-1); vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py (+66/-17)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Summary ⏎  ⏎ Stacked on top of #40082. ⏎  ⏎ This PR refines the FlashInfer B12x MoE integration by switching the SM12x MoE path to FlashInfer's `B12xMoEWrapper` API and adding ReLU2 / non-gated MoE coverage. ⏎  ⏎ Key changes: ⏎ - Use `B12xMoEWrapper` for `FlashInferB12xExperts` ⏎ - Keep BF16 hidden states as unquantized inputs; B12x handles FP4 activation quantization internally ⏎ - Support both SiLU gated MoE and ReLU2 non-gated MoE ⏎ - Add ReLU2 test  …[truncated]

### L3-04adc8843b  (L3, 2026-07-06, sha 04adc8843bbe, PR #47716)
TITLE: [Bugfix]Fix DeepSeek-V4 fp8_ds_mla KV cache reshape (#47716)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/sparse_swa.py (+2/-0); vllm/models/deepseek_v4/attention.py (+6/-1); vllm/v1/worker/gpu_model_runner.py (+6/-1)
LABELS: bug, ready, v1, deepseek
ISSUES: #47648 [Bug]: DeepSeek-V4-Flash-DSpark fails on H200/SM90 with FlashMLA KV cache shape mismatch
BODY: Fixes #47648. ⏎  ⏎ DeepSeek-V4 FlashMLA uses a custom `fp8_ds_mla` KV layout with 584 bytes per token. The main MLA and SWA cache specs already carry `cache_dtype_str`, but they did not consistently expose `kv_quant_mode`, and the GPU model runner used the global cache dtype when computing backend KV cache shapes. ⏎  ⏎ As a result, DSpark serving on H200 could allocate/reshape SWA KV cache as the semantic 512-byte head size, then fail in FlashMLA spa …[truncated]

### L3-7a90eb98ab  (L3, 2026-07-06, sha 7a90eb98ab70, PR #47091)
TITLE: [Bugfix] [Gemma4] Fix Gemma4 MTP draft model layers ignoring quant_config (#47091)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/gemma4_mtp.py (+13/-4)
LABELS: bug, ready, quantization, verified
BODY: ## Purpose ⏎  ⏎ This PR fixes quantization support for the Gemma4 Eagle (MTP) draft model in speculative decoding. Previously, all layers in the Gemma4 MTP draft model were hardcoded to `quant_config=None`, causing quantized draft checkpoints to silently load incorrectly and produce 0% draft token acceptance. ⏎  ⏎ This PR addresses the following: ⏎  ⏎ - Use `get_draft_quant_config` in `Gemma4MultiTokenPredictor` and `Gemma4MTP` to obtain the draft mode …[truncated]

### L3-8f0e75e16b  (L3, 2026-07-06, sha 8f0e75e16b96, PR #47481)
TITLE: [ROCm][CI] Adding nixl multiconn (#47481)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+53/-0); tests/v1/kv_connector/nixl_integration/run_mamba_prefix_cache_test.sh (+15/-5); tests/v1/kv_connector/nixl_integration/run_multi_connector_accuracy_test.sh (+10/-1); tests/v1/kv_connector/nixl_integration/run_multi_connector_edge_case_test.sh (+10/-1)
LABELS: rocm, ready, ci/build, v1, kv-connector
BODY: Adds MI300 ROCm CI jobs for Hybrid SSM NixlConnector prefix cache coverage and MultiConnector NIXL plus offloading accuracy and edge case coverage. The new jobs run on the 2-GPU MI300 pool and use the TRITON_ATTN backend. The NIXL integration scripts now accept ATTENTION_BACKEND and VLLM_SERVE_EXTRA_ARGS so the same scripts can be reused across backend-specific ROCm coverage. The MultiConnector scripts also resolve the repo root without requiring …[truncated]

### L3-736f1a5907  (L3, 2026-07-06, sha 736f1a590714, PR #47688)
TITLE: [XPU] Route mm_prefix models to Triton attention backend (#47688)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/platforms/xpu.py (+9/-0)
LABELS: intel-gpu, ready
BODY: ## Purpose ⏎ mm_prefix (prefix-LM bidirectional mask) is only supported by FA4, which is unavailable on XPU, so fall back to Triton attention. ⏎  ⏎ ## Test Plan ⏎ `pytest -s -v tests/models/multimodal/generation/test_mm_prefix_lm.py::test_mm_prefix_lm_e2e` ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-26c754d847  (L3, 2026-07-06, sha 26c754d8475e, PR #47116)
TITLE: [XPU][Bugfix] Do not transpose weight_scale_inv at load time (#47116)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/scaled_mm/xpu.py (+2/-24)
LABELS: bug, intel-gpu, ready
BODY: ## Summary ⏎  ⏎ The MLA attention's `_o_proj` path (via `_get_cached_wo_a_bf16` in `rocm_aiter_mla_sparse.py`) directly reads `weight_scale_inv` and assumes the original `[N/128, K/128]` checkpoint layout for manual dequantization. ⏎  ⏎ Pre-transposing to `[K/128, N/128]` in `process_weights_after_loading` caused `_expand_2d_block_scales` to apply wrong scale values (row/col blocks swapped), silently degrading GSM8K accuracy from 96.5% to 89% on DeepSeek …[truncated]

### L3-b136cc2c2c  (L3, 2026-07-06, sha b136cc2c2c59, PR #45965)
TITLE: [Bugfix][Model] Add stability window to DiffusionGemma to match HF stability_threshold semantics (#45965)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/diffusion_gemma.py (+4/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎ DiffusionGemma's denoise sampler commits a canvas one step too early, dropping the first answer token on short, high-confidence prompts (e.g. "The capital of France is" returns "" instead of "...Paris."). ⏎  ⏎ The checkpoint sets `stability_threshold` in the HF convention (`StableAndConfidentStoppingCriteria`): value `k` requires `k + 1` consecutive identical argmax canvases. `_compiled_sample_step` currently treats it as a sliding-wind …[truncated]

### L3-ae098abe3f  (L3, 2026-07-06, sha ae098abe3fff, PR #47726)
TITLE: [CI] Fix some errors on `main` (#47726)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/distributed/test_weight_transfer.py (+37/-13); vllm/model_executor/warmup/kernel_warmup.py (+2/-9)
LABELS: ready
BODY: - https://github.com/vllm-project/vllm/pull/44353 checked `self.device.index` to try and get the device index, but `self.device` is a `str` in tests so they were passing the `str.index` bound method to `packed_ipc_consumer`. This PR fixes that. ⏎ - https://github.com/vllm-project/vllm/pull/46683 enabled persistent cache for several a2a backends during flashinfer autotuning. This caused timeouts in several tests where non-leader ranks are waiting f …[truncated]

### L3-8b79971bb9  (L3, 2026-07-06, sha 8b79971bb9f9, PR #43597)
TITLE: attention: pass None for unused args in unified attention TD path (#43597)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/v1/attention/ops/triton_unified_attention.py (+29/-20)
LABELS: ready, v1, verified
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎ Performance follow-up to #40327 (UA+TD). On XPU we measured +1-10% throughput in some configurations. ⏎  ⏎ Co-authored with: @quinnlp

### L3-482e5524fe  (L3, 2026-07-06, sha 482e5524fe12, PR #47276)
TITLE: [Bugfix][ROCm] Fix memory access fault in AITER MLA backend for DPA+FP8 KV  (#47276)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+1/-0)
LABELS: bug, rocm, ready, v1
BODY: ## Purpose ⏎  ⏎ Fix memory access fault in DSv3 when running DPA with FP8 KV. Fix is to cast Q to FP8 whenever KV is FP8 as Q is already quantized to FP8 in this case and the MLA decode kernel expects Q to be FP8 as well. ⏎  ⏎ **The issue is only observed with DP, not with TP**. This is because MLA metadata tiles are sized based on dtypes OR when there are TP8 head shapes. The following snippet from [AITER](https://github.com/ROCm/aiter/blob/90ee2737 …[truncated]

### L3-9fde043f54  (L3, 2026-07-07, sha 9fde043f5452, PR #43994)
TITLE: [Kernel][Helion][1/N] Add Helion kernel for silu_and_mul_per_block_quant (#43994)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/helion/test_silu_and_mul_per_block_quant.py (+224/-0); vllm/kernels/helion/configs/silu_and_mul_per_block_quant/nvidia_b200.json (+2099/-0); vllm/kernels/helion/configs/silu_and_mul_per_block_quant/nvidia_h100.json (+2207/-0); vllm/kernels/helion/ops/silu_and_mul_per_block_quant.py (+252/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ This PR is to add Helion kernel for ```silu_and_mul_per_block_quant``` operation. This is a subtask for https://github.com/vllm-project/vllm/issues/32962. ⏎  ⏎ ### Kernel level benchmark ⏎ **Environment** ⏎ Python: 3.12.12 ⏎ Pytorch: 2.11.0 ⏎ Cuda: 13.0 ⏎ Helion: 1.0.0 ⏎  ⏎ **Benchmark Setup** ⏎ Latency measure: ```triton.testing.do_bench_cudagraph(rep=1000)``` ⏎ Baseline: ```torch.compile``` and ```torch.ops._C.silu_and_mul_per_block_quant ``` …[truncated]

### L3-cbb5f045be  (L3, 2026-07-07, sha cbb5f045beb1, PR #46904)
TITLE: [ROCm][CI] Refresh ROCm base images when docker rocm_base changes (#46904)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.rocm.aiter_unified
FILES: vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+3/-1); .buildkite/ci_config_rocm.yaml (+1/-0); .buildkite/hardware_tests/amd.yaml (+32/-29); .buildkite/scripts/ci-bake-rocm.sh (+25/-3); .buildkite/scripts/hardware_ci/run-amd-test.sh (+25/-1); .buildkite/scripts/rocm/build-ci-base.sh (+32/-0); .buildkite/scripts/rocm/build-test-image.sh (+57/-0); .buildkite/scripts/rocm/refresh-base-image.sh (+513/-0); .buildkite/scripts/rocm/smoke-test-image.sh (+32/-0); docker/Dockerfile.rocm_base (+1/-1); (+4 more)
LABELS: documentation, rocm, ready, ci/build, v1
BODY: ROCm `Dockerfile.rocm_base` changes currently do not have a clean release path through CI. A PR can change the base image recipe while the rest of the ROCm image stack is still validated against older base/ci_base images. This makes base-image changes harder to reason about and easy to under-test. ⏎  ⏎ This PR adds an explicit base-refresh lane to the AMD image build: ⏎  ⏎ - Check whether `docker/Dockerfile.rocm_base` changed before doing any expensi …[truncated]

### L3-69f3150981  (L3, 2026-07-07, sha 69f3150981e4, PR #47253)
TITLE: [XPU] Fix PP accuracy on XPU device (#47253)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/pp_utils.py (+5/-0)
LABELS: intel-gpu, ready, v1
BODY: Fix the accuracy issue on XPU device.

### L3-d3e69fd671  (L3, 2026-07-07, sha d3e69fd6714e, PR #47081)
TITLE: [Perf] Use blocking CUDA events to avoid busy polling cuda driver lock (#47081)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/async_utils.py (+4/-2); vllm/v1/worker/gpu/spec_decode/utils.py (+2/-1); vllm/v1/worker/gpu_model_runner.py (+7/-3)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ When async scheduling is enabled, the per-step CUDA-event syncs default to spin-wait events that busy-poll the CUDA driver lock. Under per-step TP collective contention the main-thread spin can balloon on a *rotating* rank, making it arrive late at the eager `cross_device_reduce` all-reduce while the other ranks spin-wait on it — the occasional "long all-reduce" straggle. This makes those events **blocking (sleep)** instead, for both  …[truncated]

### L3-066f02ae94  (L3, 2026-07-07, sha 066f02ae94c7, PR #47427)
TITLE: [MoE] FI autotuning: max bucket = max token count [e.g. `DP_size*MNBT`] (#47427)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py (+2/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+7/-1); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+8/-1); vllm/model_executor/layers/fused_moe/utils.py (+21/-0)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: vLLM does not pass `tune_max_num_tokens` to FlashInfer’s BF16, FP8, and NVFP4 TRT-LLM MoE kernels. FlashInfer therefore ends autotuning at its default maximum bucket of 8192. ⏎  ⏎ This can be too low. The estimated maximum token count is: `DP size * max_num_batched_tokens` ⏎  ⏎ Larger inputs still run, but use a tactic tuned for a smaller bucket. ⏎  ⏎ This PR passes that estimate to FlashInfer while keeping 8192 as the minimum tuning range. MXFP4 is un …[truncated]

### L3-2f71b2bd9f  (L3, 2026-07-07, sha 2f71b2bd9f69, PR #47685)
TITLE: [ROCm] Align mixed encoder-decoder KV cache views in V2 runner (#47685)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/attn_utils.py (+82/-0)
LABELS: rocm, ready, v1
BODY: Fixes regression in encoder-decoder models that share one KV cache allocation across attention backends with different physical layouts, introduced by: ⏎ - #47035 ⏎  ⏎ That PR normalized the legacy model runner path, but the failing Buildkite job uses the V2 model runner. In the V2 path, `attn_utils._reshape_kv_cache` still left shared decoder/cross-attention allocations with mixed layout semantics: decoder self-attention used ROCm's K/V-first layou …[truncated]

### L3-abe41f28de  (L3, 2026-07-07, sha abe41f28de8f, PR #47835)
TITLE: Upgrade tpu-inference to v0.24.0 (#47835)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/tpu.txt (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ Upgrade tpu-inference to latest stable release v0.24.0 ⏎  ⏎ ## Test Plan ⏎  ⏎ Verified on tpu-inference CI. ⏎  ⏎ ## Test Result ⏎  ⏎ Success. ⏎  ⏎ --- ⏎ [details omitted]

### L3-6e35c5e5af  (L3, 2026-07-07, sha 6e35c5e5af40, PR #47731)
TITLE: [ROCm][CI] Minimize comment in RocmAttention q_scale check (#47731)
SOURCES: path_core, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn
FILES: vllm/v1/attention/backends/rocm_attn.py (+2/-7)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎  ⏎ Follow-up to #46148 addressing review feedback from @Rohan138: the explanatory ⏎ comment above the fp8-query `q_scale` guard in `RocmAttentionImpl.forward()` was ⏎ too long. This PR shortens it to a concise two lines. ⏎  ⏎ **No functional change** — only the comment is edited; the guard logic is ⏎ untouched. ⏎  ⏎ ## Test Plan ⏎  ⏎ Comment-only change; no new tests required. Behavior is unchanged and remains ⏎ covered by the existing test that …[truncated]

### L3-4aceabf8c1  (L3, 2026-07-07, sha 4aceabf8c1a4, PR #47766)
TITLE: [ROCm][Bugfix] Key sparse-MLA persistent metadata on per-request context lengths (#47766)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+12/-15)
LABELS: bug, rocm, ready, v1
BODY: ## Purpose ⏎  ⏎ Fixes the sparse-MLA persistent-metadata cache collision behind #47042 ⏎ (GLM-4.6 / DeepSeek-V3.2-style DSA models on ROCm produce correct output up to ⏎ ~20K tokens, then collapse into repetition/garbage at longer context). ⏎  ⏎ The ROCm sparse-MLA backend caches the persistent work-stealing metadata and ⏎ recomputes it only when `metadata_key` changes. The key was keyed on ⏎ `min(seq_lens, topk)`, but the metadata actually depends on the per-to …[truncated]

### L3-3dd910da42  (L3, 2026-07-07, sha 3dd910da4222, PR #47908)
TITLE: [Bugfix] Allow non-contiguous query in FlashInfer FP8 query quantization (#47908)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+1/-1)
LABELS: bug, ready, v1, nvidia
ISSUES: #47905 [Bug] Nemotron + FlashInfer + FP8 KV cache crashes on Hopper: assert query.is_contiguous() in maybe_quant_query
DEEP_STUDY: deep-study correctness case vllm:3dd910da42: class=shape_alignment_edge; symptom=crash_or_exception; introducing=#43232
BODY: Fix #47905  ⏎  ⏎ ## Purpose ⏎  ⏎ Fix an `AssertionError` in the FlashInfer attention backend when running models that slice their query from a fused QKV projection (e.g. Nemotron) with FP8 KV cache. ⏎  ⏎ #43232 moved the query→FP8 quantization inline into `FlashInferImpl.maybe_quant_query`, guarded by `assert query.is_contiguous()`. Models such as Nemotron produce their query via `q, k, v = qkv.split(...)`, which returns a **view** into the fused QKV t …[truncated]

### L3-9021589498  (L3, 2026-07-07, sha 902158949803, PR #47502)
TITLE: [Minimax-M3] Using tok_sparse_select from MSA instead of triton kernels (#47502)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: cmake/external_projects/fmha_sm100.cmake (+1/-1); tests/kernels/attention/test_minimax_m3.py (+4/-4); vllm/models/minimax_m3/common/indexer.py (+7/-0); vllm/models/minimax_m3/common/ops/__init__.py (+2/-0); vllm/models/minimax_m3/common/ops/index_topk.py (+67/-15); vllm/models/minimax_m3/common/ops/sparse_attn.py (+7/-11); vllm/models/minimax_m3/nvidia/indexer_msa.py (+135/-35); vllm/models/minimax_m3/nvidia/model.py (+4/-3); vllm/models/minimax_m3/nvidia/sparse_attention_msa.py (+6/-3)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ Replace the two per-path Triton top-k implementations in the MiniMax-M3 MSA (SM100/Blackwell) indexer with a **single MSA `sparse_topk_select`** over one unified score buffer shared by decode and prefill. ⏎  ⏎ **Before**, the MSA indexer ran two independent top-k paths: ⏎ - **decode** → Triton fused `minimax_m3_index_decode` (block-score + split-K top-k) writing `topk_indices_buffer[:, :nd]` ⏎ - **prefill** → fmha_sm100 `OnlyScore` → Tr …[truncated]

### L3-80eb01e93d  (L3, 2026-07-07, sha 80eb01e93dcd, PR #47493)
TITLE: [Bugfix] DSV4 TP16 garbage output (#47493)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/sparse_swa.py (+2/-1); vllm/models/deepseek_v4/attention.py (+4/-4); vllm/models/deepseek_v4/common/ops/cache_utils.py (+35/-0); vllm/models/deepseek_v4/compressor.py (+1/-1); vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py (+19/-0)
LABELS: bug, ready, v1, nvidia
BODY: Fixes DeepSeek-V4 garbage output (gsm8k ~0% #47783) with the `FLASHINFER_MLA_SPARSE_DSV4` backend on the packed KV regression from #44577, coauthored by @LucasWilkinson. Remapping is adopted from #47895. ⏎  ⏎ ## Problem ⏎  ⏎ #44577 packs a block's KV components (indexer + compressor + SWA) into one contiguous per-block allocation, so a layer's per-block stride becomes the whole block size. The FlashInfer sparse-MLA decode reads the pool by **flat glo …[truncated]

### L3-2afa3f7e95  (L3, 2026-07-07, sha 2afa3f7e9502, PR #47631)
TITLE: [Perf] Minimax M3 - Support cross-layer allreduce-norm fusion (#47631)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/prefill/selector.py (+1/-1); vllm/model_executor/layers/fused_moe/layer.py (+13/-0); vllm/model_executor/layers/fused_moe/runner/moe_runner.py (+7/-1); vllm/model_executor/models/deepseek_v2.py (+2/-0); vllm/models/deepseek_v32/nvidia/model.py (+4/-4); vllm/models/deepseek_v32/nvidia/mtp.py (+1/-1); vllm/models/minimax_m3/nvidia/model.py (+37/-10)
LABELS: ready, v1, deepseek
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ This PR supports fusing Minimax M3's MoE layers' allreduce with next's layer's input norm. ⏎  ⏎ ## Performance ⏎ TP2 with InfX benchmarking: **13406 tok/s -> 13709 tok/s (2.2% speedup)** ⏎ ``` ⏎ vllm serve nvidia/MiniMax-M3-NVFP4 \ ⏎   --tensor-parallel-size 2 \ ⏎   --kv-cache-dtype fp8 \ ⏎   --max-model-len 9472 \ ⏎   --block-size 128 \ ⏎   --language-model-only \ ⏎   --max-cudagraph-capture-size 2048 \ ⏎   --max-num-batched-tokens 16384 \ ⏎   -- …[truncated]

### L3-55da232db6  (L3, 2026-07-07, sha 55da232db696, PR #45207)
TITLE: [Bugfix] Pad Mamba page size instead of scaling block_size in unify_kv_cache_spec_page_size (#45207)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_kv_cache_utils.py (+85/-0); vllm/v1/core/kv_cache_utils.py (+16/-5)
LABELS: bug, ready, v1
ISSUES: #43626 [Bug]: AssertionError at kv_cache_utils.py:1042 — dense draft model + hybrid-attention main (DeltaNet+SWA) fails in unify_kv_cache_spec_page_size
BODY: ## Purpose ⏎  ⏎ Fixes #43626. ⏎  ⏎ `unify_kv_cache_spec_page_size` unifies differing KV page sizes by scaling `block_size` of the smaller-page layers. That is a no-op for `MambaSpec`, whose `page_size_bytes` is determined by its state shapes (and an optional `page_size_padded` override) and does not change with `block_size` — so the post-scaling `assert new_spec.page_size_bytes == max_page_size` fires as a bare `AssertionError` at engine init. ⏎  ⏎ This is r …[truncated]

### L3-d9e57ea82e  (L3, 2026-07-08, sha d9e57ea82e3e, PR #46117)
TITLE: [ROCm][Perf] MXFP8 dense-linear + grouped-MoE GEMM optimizations for MiniMax-M3 (#46117)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_minimax_m3_amd_ops.py (+87/-0); vllm/model_executor/kernels/linear/mxfp8/rocm_native.py (+76/-7); vllm/model_executor/layers/fused_moe/experts/mxfp8_native_moe.py (+63/-35)
LABELS: rocm, ready, verified
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ Numerically-equivalent Triton-kernel optimizations for the native MXFP8 path of MiniMax-M3 on ⏎ AMD MI355X (CDNA4 / gfx950). Math is unchanged (`tl.dot_scaled`, fp32 accumulate; outputs ⏎ numerically identical to the current kernels, tol 6e-2). **AMD/ROCm-only**: both kernels are ⏎ gated on `current_platform.is_rocm() and supports_mx()` (gfx95x); on other platforms the oracle ⏎ selects the existing paths, so nothing here affects non-ROC …[truncated]

### L3-7cc2e8e74f  (L3, 2026-07-08, sha 7cc2e8e74fa0, PR #47911)
TITLE: fix: hash speculative draft model config (#47911)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/speculative.py (+3/-1)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ AOT compilation cache keys include VllmConfig.compute_hash(), which delegates to SpeculativeConfig.compute_hash(). For DFlash and related aux-hidden-state methods, the speculative hash only distinguished whether aux states were used and which aux layer IDs were returned. ⏎  ⏎ That is insufficient when two deployments share an AOT cache root but use different draft model configs with the same aux layer IDs. The second deployment can lo …[truncated]

### L3-8347c6e6e1  (L3, 2026-07-08, sha 8347c6e6e18a, PR #47995)
TITLE: updated flash_attn GIT_TAG to point to torch Stable ABI FA3 commit (#47995)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ Redoing the PR #46644 because the flash-attention `GIT_TAG` got changed to an older tag in #46182, getting rid of the torch Stable ABI FA3 build.  ⏎  ⏎ cc @Harry-Chen @janeyx99 @LopezCastroRoberto ⏎  ⏎ ## Test Plan ⏎ build with new tag and utilize [torch-abi-audit](https://pypi.org/project/torch-abi-audit/) to make sure FA3 only uses stable symbols.  The actual unit test where ran in PR #46644.  ⏎  ⏎ ## Test Result ⏎ ``` ⏎   -- extensions -- ⏎  …[truncated]

### L3-f05603fa28  (L3, 2026-07-08, sha f05603fa287a, PR #47801)
TITLE: [Bugfix][DCP] Cast LSE to fp32 in a2a combine to fix bf16 bitcast crash (#47801)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/dcp_alltoall.py (+4/-0)
LABELS: bug, ready, v1
ISSUES: #47800 [Bug]: DCP --dcp-comm-backend a2a crashes with "Cannot bitcast data-type of size 16 to data-type of size 32" on MLA models
BODY: # [Bugfix][DCP] Cast LSE to fp32 in a2a combine to fix bf16 bitcast crash ⏎  ⏎ ## Purpose ⏎  ⏎ Fixes #47800. ⏎  ⏎ The DCP all-to-all combine backend (`--dcp-comm-backend a2a`) crashes at ⏎ startup on MLA models whose decode backend returns a non-fp32 softmax LSE: ⏎  ⏎ ``` ⏎ ValueError: Cannot bitcast data-type of size 16 to data-type of size 32 ⏎ ``` ⏎  ⏎ `dcp_a2a_lse_reduce` (in `vllm/v1/attention/ops/dcp_alltoall.py`) and its ⏎ pack/unpack Triton kernels treat the LSE as  …[truncated]

### L3-0d2f4e7c9c  (L3, 2026-07-08, sha 0d2f4e7c9c88, PR #46661)
TITLE: Allow FlashInfer A2A backends for TRTLLM FP8 MoE Modular (#46661)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+2/-0)
LABELS: ready, nvidia
BODY: ## Purpose ⏎  ⏎ Allow FlashInfer A2A backends for TRTLLM FP8 MoE Modular. Port of #46551 to main ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-c2ecd0f888  (L3, 2026-07-08, sha c2ecd0f88822, PR #42642)
TITLE: Fix FlashAttention MLA prefill V unpadding (#42642)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+4/-0); vllm/v1/attention/backends/mla/prefill/flash_attn.py (+0/-4); vllm/v1/attention/ops/merge_attn_states.py (+8/-0)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Fix a regression in the FlashAttention MLA prefill path introduced when the prefill implementations were split out in #32623. ⏎  ⏎ Before #32623, the FlashAttention MLA helper padded `V` when the selected FlashAttention implementation did not support different QK/V head dimensions. The padded output was kept through the context/suffix `merge_attn_states` path, and only then sliced back to `v_head_dim` in `MLACommonImpl.forward_mha`. ⏎  ⏎ Afte …[truncated]

### L3-2c64b4c1cc  (L3, 2026-07-08, sha 2c64b4c1cc18, PR #47158)
TITLE: [ROCm] fixed aiter master flag and expert parallelism compatibility on minimax-m3-mxfp8 (#47158)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_minimax_m3_amd_ops.py (+94/-0); vllm/model_executor/layers/fused_moe/experts/aiter_mxfp8_moe.py (+13/-12)
LABELS: rocm, ready
BODY: ## Purpose ⏎  ⏎ Fix the situation the accuracy is bad when --enable-expert-parallel is on and VLLM_ROCM_USE_AITER=1. ⏎  ⏎ Validation  on gfx950) ⏎ Kernel level: FlyDSL mxfp8 EP additivity test — sum of EP-shard outputs == full no-mask output, cosine 1.0, max-diff 0.0. Confirms the kernel + 0/1-mask contract are correct. ⏎ Reproduced the bug + fix side-by-side: master-OFF (idx→convert) cos=1.0, master-ON FIXED (mask as-is) cos=1.0, master-ON BUGGY (mask …[truncated]

### L3-49abadaedb  (L3, 2026-07-08, sha 49abadaedba2, PR #47894)
TITLE: [ROCm][Bugfix] Fix empty-tensor .max() crash in AITER FA (#47894)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+5/-1)
LABELS: bug, rocm, ready, v1
BODY: ## Purpose ⏎ On ROCm, the AITER FA backend crashes during warmup for speech-to-text models (Whisper, Voxtral) when `num_chunks == 0`: `cu_seq_lens_cpu` is empty and `.max()` raises `RuntimeError: max(): ... input.numel() == 0`, killing EngineCore. ⏎  ⏎ **Fix:** use `0` when there are no chunks instead of calling `.max()` on an empty tensor. ⏎ ## Test Plan ⏎ ``` ⏎  pytest -v -s entrypoints/speech_to_text/realtime/test_realtime_validation.py  -k "Voxtral …[truncated]

### L3-26831949b4  (L3, 2026-07-08, sha 26831949b48a, PR #47912)
TITLE: [ROCm] Fix pooling startup workspace lock (#47912)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.v1_backend, L3.rocm.v1_rocm_attn
FILES: vllm/v1/attention/backends/rocm_attn.py (+3/-0); vllm/v1/attention/backends/triton_attn.py (+1/-0); vllm/v1/attention/ops/triton_prefill_attention.py (+14/-2); vllm/v1/worker/gpu_worker.py (+22/-3)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study: this PR was reverted by PR 48154 (partial_revert, reason=ci_or_test_failure)
BODY: Motivation: https://buildkite.com/vllm/ci/builds/76664/list?sid=019f3900-a887-45a4-bd39-f1e71a7beaee&tab=output ⏎  ⏎ `Language Models Test (Extended Pooling)` failed in `test_openai_privacy_filter` while starting `openai/privacy-filter`. This model exposes the startup bug because it combines token-classification pooling, a GptOss/FusedMoE backbone, ROCm modular MoE workspace allocation, and a post-capture warmup shape larger than the graph-capture sh …[truncated]

### L3-2c17d33f42  (L3, 2026-07-08, sha 2c17d33f4291, PR #47144)
TITLE: [Bugfix][ROCm] Change AttentionCGSuppoort in TritonMLA to UNIFORM_SINGLE_TOKEN_DECODE (#47144)
SOURCES: path_core, subject_keyword, corpus:kernel-correctness-cases
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/triton_mla.py (+3/-1)
LABELS: bug, rocm, ready, v1
DEEP_STUDY: deep-study correctness case vllm:2c17d33f42: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=#42885
BODY: ## Purpose ⏎ TritonMLAMetadataBuilder advertised _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH (added in #42885) while keeping the inherited query_len_support = SINGLE_ONLY (so reorder_batch_threshold = 1). ⏎  ⏎ With speculative decoding enabled, the UNIFORM_BATCH claim bypasses a downgrade in resolve_cudagraph_mode_and_sizes (which only triggers when min_cg_support < UNIFORM_BATCH).  ⏎ Full decode cudagraph capture then proceeds with max_que …[truncated]

### L3-95d6d6f4bb  (L3, 2026-07-08, sha 95d6d6f4bba8, PR #48046)
TITLE: [Bugfix] Use int8 workspace for FlashInfer MLA decode (#48046)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer, L3.mla.flashinfer_sparse
FILES: vllm/v1/attention/backends/mla/flashinfer_mla.py (+3/-1); vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+3/-1); tests/kernels/attention/test_flashinfer_mla_decode.py (+87/-45)
LABELS: bug, ready, v1, nvidia
BODY: FlashInfer's CuteDSL MLA-decode tactic requires an int8 workspace_buffer, but vLLM allocated it as uint8. The trtllm-gen path (used for normal inference) views the buffer as uint8, so the mismatch is invisible until the FlashInfer autotuner enumerates the CuteDSL tactic. Model Runner V2's warmup autotunes trtllm_batch_decode_mla, so once V2 became the default runner for MoE models it hit `workspace_buffer must be torch.int8`, crashing every DeepS …[truncated]

### L3-089e412878  (L3, 2026-07-09, sha 089e41287824, PR #45182)
TITLE: [Perf] Integrate TRTLLM BF16 MoE Modular Kernel  (#45182)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_trtllm_bf16_moe.py (+145/-0); tests/kernels/moe/test_unquantized_backend_selection.py (+169/-5); vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py (+139/-23); vllm/model_executor/layers/fused_moe/oracle/base.py (+4/-2); vllm/model_executor/layers/fused_moe/oracle/unquantized.py (+28/-25)
LABELS: ready, nvidia, verified
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Add TRTLLM BF16 MoE Modular Kernel. The trtllm monolithic kernel has restrictions over the routing method of the model. This PR enables more models (e.g., glm5.1) to use the trtllm moe backend with all2all backend by adding the modular version. ⏎  ⏎ ## Test Plan ⏎ Tested with GLM-5.1  ⏎ ``` ⏎ vllm serve /lustre/fsw/coreai_dlfw_dev/kaihangj/models/GLM-5.1 \ ⏎     --dtype bfloat16 \ ⏎     --moe-backend flashinfer_trtllm \ ⏎     --enable-expert- …[truncated]

### L3-b0dec2a11b  (L3, 2026-07-09, sha b0dec2a11b91, PR #45149)
TITLE: [ROCM][DSV32][Perf][MTP] Enable UNIFORM_BATCH CG mode in rocm_aiter_mla_sparse (#45149)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+4/-3)
LABELS: rocm, ready, v1, verified
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ In order to improve performance especially in CPU bound cases when MTP is enabled UNIFORM_BATCH CG mode needs to be enabled in rocm_aiter_mla_sparse.py. It has not been enabled mostly due to testing reasons and lack of focus for MTP. ⏎  ⏎ ## Test Plan ⏎ Extensive testing of DeepSeekV3.2 with MTP on on MI350X and MI355X HW, both performance and lm_eval. Basically mostly on the "kicking the tires" style where various concurrencies were tes …[truncated]

### L3-bbb0f945ff  (L3, 2026-07-09, sha bbb0f945ff34, PR #47404)
TITLE: [ROCm] Synchronize sparse MLA metadata before graph replay (#47404)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+3/-0); tests/kernels/attention/test_rocm_aiter_mla_sparse_metadata_sync.py (+125/-0)
LABELS: rocm, ready, v1
ISSUES: #47196 [Bug]: [ROCm][gfx942] GPU memory access fault with MTP spec-decode + sparse-MLA decode under cuda graph (GLM-5.1-FP8)
BODY: Fixes https://github.com/vllm-project/vllm/issues/47196 ⏎  ⏎ ## Summary ⏎  ⏎ This patch orders ROCm AITER sparse MLA persistent metadata updates before graph replay consumes those metadata buffers. ⏎  ⏎ The sparse MLA backend precomputes persistent metadata with `get_mla_metadata_v1()`. The resulting buffers are later passed to the AITER sparse MLA decode kernel. When CUDA graph replay consumes those persistent buffers, the metadata write must be order …[truncated]

### L3-e08a915146  (L3, 2026-07-09, sha e08a91514681, PR #48135)
TITLE: [Bugfix] Preserve tensor causal metadata for grouped attention (#48135)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/attn_utils.py (+4/-2)
LABELS: bug, ready, v1
BODY: Fix for nightly CI failure: [Buildkite #77195, LM Eval Small Models Distributed (2xB200)](https://buildkite.com/vllm/ci/builds/77195#019f4388-f4da-4b16-9373-58229478cb23). ⏎  ⏎ ## Summary ⏎ - Preserve per-request `torch.Tensor` causal masks when grouped attention metadata is built. ⏎ - Keep the per-KV-cache-group `Mapping[int, bool]` causal override path used by hybrid DFlash drafters. ⏎  ⏎ ## Regression ⏎ - Breaking PR: [#47914 - [Spec Decode] Support  …[truncated]

### L3-95ed0feaa5  (L3, 2026-07-09, sha 95ed0feaa5cd, PR #40996)
TITLE: DCP supports hybrid attention (#40996)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/model_executor/layers/attention_layer_base.py (+1/-0); vllm/v1/attention/backend.py (+2/-0); vllm/v1/attention/backends/flash_attn.py (+176/-45); tests/distributed/test_context_parallel.py (+8/-0); tests/distributed/test_pynccl.py (+47/-0); tests/models/language/generation/test_hybrid.py (+13/-1); tests/models/multimodal/generation/test_vit_cudagraph.py (+1/-0); tests/test_config.py (+4/-4); tests/v1/streaming_input/test_gpu_model_runner_streaming.py (+1/-0); tests/v1/worker/test_cp_utils.py (+45/-0); (+16 more)
LABELS: tpu, speculative-decoding, ready, v1, multi-modality, cpu, nvidia, verified
BODY: ## Purpose ⏎  ⏎ Add DCP support for hybrid-attention models. Hybrid-attention models such as `Qwen/Qwen3.5-0.8B` contain both DCP-capable full-attention layers and non-DCP layers. This PR enables DCP for supported ⏎ attention groups without globally blocking hybrid-attention DCP, while keeping ⏎ non-DCP groups on local cache/state handling. ⏎  ⏎ ## Test Plan ⏎  ⏎ Run focused hybrid-attention DCP validation with ⏎ `tests/distributed/test_context_parallel.p …[truncated]

### L3-f1a5adddb8  (L3, 2026-07-09, sha f1a5adddb815, PR #48144)
TITLE: update marlin M size for EP (#48144)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/marlin_moe.py (+7/-2)
LABELS: ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎ Updates the Marlin MoE `M` dimension size to estimate the number of valid tokens per rank. When using DP/EP, the `hidden_states.shape[0]` dimension is padded to `max_tokens_per_rank * num_ranks`. Scaling this by `num_experts / num_global_experts` recovers an estimated number of valid (i.e. non-padding) tokens per rank. ⏎  ⏎ ## Test Plan ⏎  ⏎ ### Benchmark results ⏎ Benchmarked `deepseek-ai/DeepSeek-V4-Flash` on 8xH200 with DP/EP=8 ⏎  ⏎ [deta …[truncated]

### L3-a0f6d767e4  (L3, 2026-07-10, sha a0f6d767e42a, PR #48169)
TITLE: [ROCm][CI] Move remaining engine/samplers AMD steps to mi325_1 (#48169)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/engine.yaml (+2/-2); .buildkite/test_areas/samplers.yaml (+1/-1)
LABELS: rocm, ready, ci/build
BODY: ## Purpose ⏎  ⏎ Three mirrored AMD CI steps in `.buildkite/test_areas/` were still pinned to the `mi250_1` agent pool (gfx90a): the async-scheduling e2e step and the general e2e step in `engine.yaml`, and the FlashInfer sampler step in `samplers.yaml`. Every other step under `test_areas/` already runs on `mi325_1`, and these three were the only remaining `mi250_1` references in that directory. This moves them onto `mi325_1` so all mirrored AMD step …[truncated]

### L3-433f291195  (L3, 2026-07-10, sha 433f291195de, PR #48186)
TITLE: [CI] Right-size test-area timeouts from nightly durations (#48186)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/attention.yaml (+3/-3); .buildkite/test_areas/basic_correctness.yaml (+2/-2); .buildkite/test_areas/benchmarks.yaml (+2/-2); .buildkite/test_areas/compile.yaml (+11/-11); .buildkite/test_areas/cuda.yaml (+2/-2); .buildkite/test_areas/disaggregated.yaml (+12/-12); .buildkite/test_areas/distributed.yaml (+10/-10); .buildkite/test_areas/docker.yaml (+1/-1); .buildkite/test_areas/e2e_integration.yaml (+5/-5); .buildkite/test_areas/engine.yaml (+11/-11); (+20 more)
LABELS: ci/build, nvidia, rust
BODY: ## Purpose ⏎  ⏎ Right-size `timeout_in_minutes` across **all** `.buildkite/test_areas/*.yaml` files based on the actual per-job runtimes observed in the nightly full CI run ([build 77252](https://buildkite.com/vllm/ci/builds/77252), `source: schedule`, "Full CI run - nightly"). ⏎  ⏎ The nightly full run relaxes per-step timeouts and runs every job to completion, so it exposes true runtimes. Many premerge timeouts are either well below the real runtime (r …[truncated]

### L3-88e5e2c57b  (L3, 2026-07-10, sha 88e5e2c57be8, PR #47366)
TITLE: [CI/Build][AMD] Fix ROCm OOM in eagle_correctness_heavy by reserving CUDA graph memory (#47366)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/e2e/spec_decode/test_spec_decode.py (+7/-0); vllm/v1/worker/gpu_worker.py (+7/-5)
LABELS: rocm, ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ test_eagle_correctness_heavy builds a reference engine and a speculative engine back to back in one process, and its parametrized cases run sequentially. On the AMD "V1 e2e (4 GPUs)" step the TRITON_ATTN Llama-4-Scout cases (llama4_eagle, llama4_eagle_mm) aborted at engine init with HSA_STATUS_ERROR_OUT_OF_RESOURCES ("Available Free mem : 0 MB") right after CUDA graph capture, while the same model passed on ROCM_AITER_FA. ⏎  ⏎ There a …[truncated]

### L3-b12cca6a23  (L3, 2026-07-10, sha b12cca6a2352, PR #39988)
TITLE: [Bugfix] Fix turboquant FP8 cast failure for BF16 models on Ampere GPUs (#39988)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/triton_turboquant_store.py (+1/-1)
LABELS: bug, ready, v1, quantization
BODY: ## Purpose ⏎  ⏎   Fix turboquant attention backend crash when running BF16 models (e.g. Qwen2). ⏎  ⏎   Triton's `convert_custom_float8` only supports FP16/FP32 inputs. When the model's KV dtype is BF16, the direct BF16 → FP8 cast in `_tq_fused_store_fp8` triggers an `AssertionError`. Fix by casting to FP32 first, which is safe and lossless for all input dtypes (FP16, BF16, FP32). ⏎  ⏎   ## Test Plan ⏎  ⏎   ```bash ⏎   .venv/bin/python -m vllm.entrypoints. …[truncated]

### L3-735def4fcf  (L3, 2026-07-10, sha 735def4fcf39, PR #48045)
TITLE: [Bugfix] Fix FlashMLA dense fp8 metadata crash (num_sm_parts clamp) (#48045)
SOURCES: path_core, subject_keyword, dependency_pin, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_build
FILES: cmake/external_projects/flashmla.cmake (+1/-1)
LABELS: bug, ready, ci/build
ISSUES: #47935 [Bug]: DP/EP with fp8 KV Cache Brokens for FlashMLA
BODY: ## Summary ⏎  ⏎ Fixes #47935 ⏎  ⏎ ## Root cause ⏎  ⏎ FlashMLA's `get_mla_decoding_metadata_dense_fp8` computes ⏎  ⏎ ``` ⏎ int num_sm_parts = sm_count / num_heads_k / cutlass::ceil_div(num_heads_per_head_k, block_size_m); ⏎ ``` ⏎  ⏎ This can be 0 when `ceil_div(num_q_tokens_per_head_k, block_size_m) > sm_count` (with `num_heads_k == 1`). The bf16/fp16 version clamps the minimum to 1: https://github.com/vllm-project/FlashMLA/blob/a6ec2ba7bd0a7dff98b3f4d3e6b52b …[truncated]

### L3-29fd688892  (L3, 2026-07-10, sha 29fd68889221, PR #48268)
TITLE: Add VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS and skip CuTeDSL fp4_gemm autotuning by default (#48268)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+14/-0); vllm/model_executor/warmup/kernel_warmup.py (+32/-2)
LABELS: ready, startup-ux, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: CuTe-DSL `mm_fp4` autotuning JIT-compiles a fresh kernel per candidate tactic, adding 7+ minutes to warmup, while its fallback tactic is already the analytical heuristic — so skipping it is free. This passes `skip_ops={"fp4_gemm"}` (FlashInfer 0.6.13, flashinfer-ai/flashinfer#3396) to the warmup autotune, scoped to when the CuTe-DSL NVFP4 linear kernel is actually selected since all `mm_fp4` backends tune under the shared `fp4_gemm` op name. `VLL …[truncated]

### L3-9c18e90f6c  (L3, 2026-07-10, sha 9c18e90f6c94, PR #47314)
TITLE: [BugFix] Fix packed HND KV cache reshape for FlashAttention (#47314)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/attn_utils.py (+1/-1)
LABELS: bug, ready, v1
ISSUES: #47054 [Bug]: CPUOffloading + enable_cross_layers_blocks + gptoss-120b
BODY: Fixes #47054. ⏎  ⏎ Summary: ⏎ - Preserve packed KV cache physical layout before permuting back to backend logical shape. ⏎ - This keeps FlashAttention HND caches logically shaped as `[blocks, 2, block_size, heads, dim]`. ⏎  ⏎ Repro used during debugging: ⏎  ⏎ ```bash ⏎ MODEL="openai/gpt-oss-120b" ⏎ TP_SIZE=2 ⏎ CPU_BYTES=26843545600 ⏎ KV_TRANSFER_CONFIG=$(cat <<EOF_JSON ⏎ { ⏎   "kv_connector": "OffloadingConnector", ⏎   "kv_role": "kv_both", ⏎   "kv_connector_ext …[truncated]

### L3-4a6440acef  (L3, 2026-07-11, sha 4a6440acefbd, PR #47867)
TITLE: Bump Transformers version to 5.13.0 (#47867)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/test/cpu.txt (+2/-2); requirements/test/cuda.in (+1/-1); requirements/test/cuda.txt (+2/-2); requirements/test/nightly-torch.txt (+1/-1); requirements/test/rocm.in (+1/-1); requirements/test/rocm.txt (+2/-2); requirements/test/xpu.in (+1/-1); requirements/test/xpu.txt (+2/-2); vllm/model_executor/models/granitemoehybrid.py (+4/-0); vllm/model_executor/models/hunyuan_vision.py (+11/-3); (+2 more)
LABELS: ci/build, cpu, nvidia, ready-run-all-tests
BODY: PRs needed for Transformers patch: ⏎  ⏎ - https://github.com/huggingface/transformers/pull/47148 ⏎ - https://github.com/huggingface/transformers/pull/47174 ⏎ - https://github.com/huggingface/transformers/pull/47245 ⏎  ⏎ --- ⏎  ⏎ Errors resolved: ⏎  ⏎ - `AttributeError: 'str' object has no attribute '__module__'` - https://github.com/huggingface/transformers/pull/46876 added `if key.__module__.startswith("transformers."):` which does not tolerate `str` keys …[truncated]

### L3-092387963c  (L3, 2026-07-11, sha 092387963c09, PR #46276)
TITLE: [BugFix] weights processing peak memory reduction for nvfp4 MoE layers (#46276)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer_b12x_moe.py (+12/-21); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+24/-8)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎ When serving MoE models such as nvidia/Qwen3.6-35B-A3B-NVFP4 with moe_backend flashinfer_b12x on 16GB of vram, [reorder_w1w3_to_w3w1 ](https://github.com/vllm-project/vllm/blob/b80ce9dd2f30913b3b054308a09bd2d86ec6202f/vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py#L32) returns 2 new replacement tensors (weight, scale) which result in OOM. This PR implements in-place chunk swaps to mitigate memory spikes on more re …[truncated]

### L3-bec0a4ede6  (L3, 2026-07-11, sha bec0a4ede621, PR #48269)
TITLE: [Revert] [Build] Update vllm ...builds FA3 with torch stable API (#48269)
SOURCES: path_core, subject_keyword, dependency_pin, corpus:confirmed-reverts, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build
DEEP_STUDY: deep-study revert record: explicit_rollback of PR(s) 46644 reason=hardware_specific_breakage
BODY: ## Summary ⏎ - Effectively a revert of https://github.com/vllm-project/vllm/pull/46644 ⏎ - Bump the bundled vllm-flash-attn source pin from `b3964b1d8b95d8e8447435668ab169a2700bab65` to `bb9a72e7dde0dc614ffc663e052cd6a19ce73a42`. ⏎ - Picks up vllm-project/flash-attention#160, which reverts vllm-project/flash-attention#152 after it broke `FLASH_ATTN_MLA_SPARSE` in vllm-project/vllm#48221. ⏎  ⏎ ## Tests ⏎ - `git diff --check` ⏎ - `uvx pre-commit run --fil …[truncated]

### L3-51878e5b6e  (L3, 2026-07-11, sha 51878e5b6e4c, PR #44455)
TITLE: [2/N][KV-Cache Layout Refactor] Pack K/V into the content dim across attention backends (#44455)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.aiter_fa, L3.rocm.aiter_unified, L3.dispatch.abstract_interface, L3.platform.rocm_selection, L3.flex_attention
FILES: vllm/platforms/rocm.py (+16/-2); vllm/utils/torch_utils.py (+11/-45); vllm/v1/attention/backend.py (+6/-6); vllm/v1/attention/backends/flash_attn.py (+18/-13); vllm/v1/attention/backends/flash_attn_diffkv.py (+18/-24); vllm/v1/attention/backends/flashinfer.py (+97/-43); vllm/v1/attention/backends/flex_attention.py (+11/-7); vllm/v1/attention/backends/rocm_aiter_fa.py (+8/-4); vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+4/-5); vllm/v1/attention/backends/triton_attn.py (+60/-42); (+21 more)
LABELS: documentation, rocm, intel-gpu, ready, ci/build, v1, kv-connector, nvidia, rust
BODY: PR #42374 (first part of RFC #42082) has been split into 4 PRs: ⏎  ⏎ #44454 [1/N][KV-Cache Layout Refactor] Refactor DSV4 KV cache config ⏎ -> #44455 [2/N][KV-Cache Layout Refactor] Pack K/V into the content dim across attention backends ⏎ #44456 [3/N][KV-Cache Layout Refactor] Standardize Mamba cache; drop `get_transfer_cache_regions` ⏎ #44458 [4/N][KV-Cache Layout Refactor] Standardize KV cache layout ⏎  ⏎ ## Summary ⏎  ⏎ Packs K and V into the content  …[truncated]

### L3-19069bcbd5  (L3, 2026-07-11, sha 19069bcbd5be, PR #48335)
TITLE:  FP32 router GEMV optimization (#48335)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/fp32_router_gemm.cu (+179/-56); csrc/libtorch_stable/fp32_router_gemm_entry.cu (+8/-3); tests/kernels/test_fp32_router_gemm.py (+94/-5); vllm/model_executor/layers/fused_moe/router/gate_linear.py (+3/-3)
LABELS: ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-5f8e73cb8b  (L3, 2026-07-12, sha 5f8e73cb8b8d, PR #48330)
TITLE: [Bugfix] Guard mixed-dtype allreduce RMSNorm quant fusions (#48330)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/passes/distributed/test_fusion_all_reduce.py (+32/-1); vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+12/-2)
LABELS: bug, ready
ISSUES: #48324 [Bug]: FlashInfer fused allreduce + residual RMSNorm + quant produces corrupted output with FP32 norm weights
DEEP_STUDY: deep-study correctness case vllm:5f8e73cb8b: class=numerical_precision; symptom=wrong_output_or_accuracy; introducing=#21069
BODY: ## Summary ⏎  ⏎ Fixes #48324. ⏎  ⏎ The residual FlashInfer allreduce + RMSNorm + static-quantization patterns could ⏎ match graphs where the activation and RMSNorm weight have different dtypes. ⏎  ⏎ This occurs with Qwen/Gemma-style RMSNorm in ⏎ `nvidia/Qwen3.6-27B-NVFP4`: the residual stream is BF16, while the effective ⏎ RMSNorm weight is FP32 due to the `weight.float() + 1.0` computation. Selecting ⏎ the fused quantized operation for this graph corrupts …[truncated]

### L3-ee5a89f4d7  (L3, 2026-07-12, sha ee5a89f4d7b8, PR #47287)
TITLE: [ROCm][MiniMax-M3] Add AITER sparse paged attention (#47287)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/_custom_ops.py (+7/-0); csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu (+126/-75); csrc/libtorch_stable/ops.h (+1/-1); csrc/libtorch_stable/torch_bindings.cpp (+1/-1); tests/kernels/attention/test_minimax_m3.py (+41/-0); tests/kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py (+130/-13); tests/kernels/test_minimax_m3_sparse_attn_fp8_scale.py (+189/-0); vllm/models/minimax_m3/amd/model.py (+214/-25); vllm/models/minimax_m3/amd/ops/sparse_attn.py (+65/-8); vllm/models/minimax_m3/amd/ops/sparse_pa.py (+466/-0); (+5 more)
LABELS: rocm, ready, ci/build, v1
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: This PR adds an opt-in ROCm AITER implementation for MiniMax-M3 sparse attention. ⏎  ⏎ MiniMax-M3 sparse attention selects top-k logical blocks of 128 tokens. This PR maps those selected 128-token blocks into AITER page-16 block tables and runs AITER Gluon paged attention over only the selected KV pages. ⏎ - Uses AITER's page-16 paged-attention path for MiniMax-M3 sparse attention. ⏎ - Preserves the fast FP8 KV-cache path by passing KV scales into sp …[truncated]

### L3-bea70c7cfc  (L3, 2026-07-13, sha bea70c7cfc10, PR #48011)
TITLE: [Attention] Make sliding-window support an explicit backend capability (#48011)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.flex_attention
FILES: vllm/model_executor/layers/attention/attention.py (+1/-0); vllm/v1/attention/backend.py (+7/-0); vllm/v1/attention/backends/cpu_attn.py (+4/-0); vllm/v1/attention/backends/flash_attn.py (+4/-0); vllm/v1/attention/backends/flashinfer.py (+4/-0); vllm/v1/attention/backends/flex_attention.py (+4/-0); vllm/v1/attention/backends/rocm_aiter_fa.py (+4/-0); vllm/v1/attention/backends/rocm_attn.py (+4/-0); vllm/v1/attention/backends/triton_attn.py (+4/-0); vllm/v1/attention/selector.py (+5/-0)
LABELS: rocm, ready, v1, cpu, nvidia
BODY: 1/2 of https://github.com/vllm-project/vllm/pull/48012 , needed to get proper backend selection support.  ⏎ I split the PRs because this one is the one which has a ~very tiny impact on existing auto-selector logic, as explained below.  ⏎ cc @MatthewBonanni  ⏎  ⏎ ### Problem ⏎ Attention backend selection models every capability constraint explicitly (`supports_sink`, `is_sparse`, `is_mla`, …) except sliding window, so the selector can hand a sliding-wi …[truncated]

### L3-05fa8183a6  (L3, 2026-07-13, sha 05fa8183a6e2, PR #46090)
TITLE: [CPU][Spec Decode] Support DFlash speculative decoding for GDN models on CPU (#46090)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: csrc/cpu/sgl-kernels/fla.cpp (+255/-0); csrc/cpu/torch_bindings.cpp (+17/-0); vllm/_custom_ops.py (+34/-0); vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py (+442/-9); vllm/model_executor/layers/utils.py (+7/-1)
LABELS: documentation, speculative-decoding, ready, v1, qwen, cpu, verified
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ This PR builds on #44029 (DFlash spec decode on CPU). It extends DFlash speculative decoding on the vLLM CPU backend to **Gated DeltaNet (GDN) hybrid models** (Qwen3.5 / Qwen3.6). ⏎  ⏎ ## Test Plan ⏎  ⏎ start server ⏎ ``` ⏎ VLLM_TARGET_DEVICE=cpu VLLM_CPU_KVCACHE_SPACE=40 \ ⏎ vllm serve Intel/Qwen3.5-9B-int4-AutoRound \ ⏎   --dtype bfloat16 \ ⏎   --enforce-eager \ ⏎   --no-enable-prefix-caching \ ⏎   --trust-remote-code \ ⏎   --max-num-batched-to …[truncated]

### L3-56a357ed33  (L3, 2026-07-13, sha 56a357ed33de, PR #48256)
TITLE: [Bugfix][KV Cache] Don't route uniform-page-size MLA+SWA models into DeepseekV4 packing (#48256)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_kv_cache_utils.py (+43/-0); vllm/v1/core/kv_cache_utils.py (+5/-0)
LABELS: bug, ready, v1, deepseek
BODY: `group_and_unify_kv_cache_specs()` targets DeepseekV4, where MLA and sliding-window MLA layers have different page sizes and must be tuple-packed.  ⏎ It fired for any model containing a SlidingWindowMLASpec, so a non-DeepseekV4 model with a uniform page size was wrongly pushed into the packing path.  ⏎ This results in creating "one-element bins", while unecessarily triggering separate paths for things like PD eg ⏎ https://github.com/vllm-project/vll …[truncated]

### L3-26587f9519  (L3, 2026-07-13, sha 26587f9519e2, PR #48261)
TITLE: [BugFix][ModelRunner V2] Fix stale attn metadata in speculator prefill cudagraph capture (#48261)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/cudagraph_utils.py (+17/-35); vllm/v1/worker/gpu/model_runner.py (+7/-3); vllm/v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py (+11/-45); vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py (+14/-12); vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py (+2/-3); vllm/v1/worker/gpu/spec_decode/dflash/speculator.py (+11/-2); vllm/v1/worker/gpu/spec_decode/speculator.py (+10/-8)
LABELS: bug, ready, v1, nvidia
BODY: The V2 speculator's prefill FULL cudagraph capture reused the attention states built earlier during the target model's capture. Those metadata objects are views into per-builder persistent buffers (e.g. FlashAttention's ⏎ AOT scheduler_metadata, dummy `query_start_loc`/`seq_len`s), which every subsequent build overwrites. By the time the speculator captured its prefill graphs, the buffer contents matched the target's last-captured batch ⏎ descripto …[truncated]

### L3-7fc97042c3  (L3, 2026-07-13, sha 7fc97042c30e, PR #48180)
TITLE: Add DCP + Eagle support for Tokenspeed MLA backends (#48180)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: requirements/cuda.txt (+1/-1); vllm/utils/flashinfer.py (+5/-3); vllm/v1/attention/backends/flashinfer.py (+76/-8); vllm/v1/attention/backends/mla/tokenspeed_mla.py (+48/-10); docs/design/attention_backends.md (+1/-1); tests/kernels/attention/test_use_trtllm_attention.py (+11/-0); tests/v1/attention/test_flashinfer_dcp_spec_reorder.py (+73/-0); tests/v1/attention/test_mla_backends.py (+98/-0)
LABELS: documentation, ready, ci/build, v1, nvidia
BODY: ## Purpose ⏎ Adds support to run DCP + Tokenspeed_MLA with Eagle. ⏎  ⏎ Builds off of https://github.com/vllm-project/vllm/pull/47915/changes  ⏎ ## Test Plan ⏎ Validated with unit test and Kimi models and the lightseekai-org's two eagle heads. ⏎  ⏎ ## Test Result ⏎ GSM8k is validated.  ⏎  ⏎ Perf (Green- this PR, Blue- No DCP) -- validated with Agentic Benchmark: ⏎ <img width="1133" height="602" alt="image" src="https://github.com/user-attachments/assets/8615 …[truncated]

### L3-75fe92a316  (L3, 2026-07-13, sha 75fe92a3162a, PR #48064)
TITLE: [Distributed][Perf] Enable FlashInfer MNNVL allreduce RMS quant fusion (#48064)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+33/-9); vllm/distributed/device_communicators/flashinfer_all_reduce.py (+41/-13)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ Enable AR + RMSNorm quant fusion to use the MNNVL backend, which current FI already supports ⏎  ⏎ <img width="1632" height="852" alt="image" src="https://github.com/user-attachments/assets/acf40be8-7efb-4360-b179-9141969360ca" /> ⏎  ⏎ `QuantType)2` is the FP4 quant path https://github.com/flashinfer-ai/flashinfer/blob/release-v0.6.13/include/flashinfer/comm/trtllm_mnnvl_allreduce.cuh#L43-L47 ⏎  ⏎ ```cpp ⏎ enum class QuantType : int { ⏎   kN …[truncated]

### L3-21472f32ea  (L3, 2026-07-13, sha 21472f32ea6c, PR #48287)
TITLE: add pad-aware swiglu limit kernel (#48287)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/activation.py (+19/-2); vllm/model_executor/layers/fused_moe/experts/marlin_moe.py (+8/-0); vllm/model_executor/layers/fused_moe/modular_kernel.py (+10/-1); vllm/model_executor/layers/fused_moe/utils.py (+103/-1)
LABELS: ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎ When serving with expert-parallelism enabled, the MoE inputs and intermediate states are padded along the token dimension to handle the worst case (i.e. all tokens routed to a single rank; see https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py#L32-L33).  ⏎  ⏎ Most MoE kernels have internal logic to ignore such padding tokens during GEMM execution; however, the MoE activation …[truncated]

### L3-382bbd5144  (L3, 2026-07-13, sha 382bbd51448b, PR #40977)
TITLE: [ROCm][Kernel] Add HybridW4A16LinearKernel: Triton prefill + HIP skinny decode (#40977)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-0); benchmarks/kernels/benchmark_rdna_hybrid_w4a16_gemm.py (+201/-0); csrc/rocm/ops.h (+6/-0); csrc/rocm/skinny_gemms_int4.cu (+795/-0); csrc/rocm/torch_bindings.cpp (+8/-0); tests/kernels/quantization/test_rdna_hybrid_w4a16.py (+525/-0); tests/kernels/quantization/test_w4a16_kernel_selection.py (+24/-5); vllm/_custom_ops.py (+36/-0); vllm/model_executor/kernels/linear/__init__.py (+5/-0); vllm/model_executor/kernels/linear/mixed_precision/__init__.py (+4/-0); (+1 more)
LABELS: performance, rocm, ready, ci/build
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: Add a hybrid W4A16 linear kernel for ROCm gfx11/gfx12 that routes between two GEMM implementations based on batch size: ⏎ - M <= 5: HIP wvSplitK_int4_g skinny GEMM (optimized for single-token decode) ⏎ - M > 5: Triton fused dequant GEMM (optimized for prefill/batched inference) ⏎  ⏎ Both paths share a single weight tensor in the AWQ-style packed shuffle layout [N, K//8]. Supports both symmetric (uint4b8, bias=8) and asymmetric (uint4, per-group zero- …[truncated]

### L3-0762f2afeb  (L3, 2026-07-13, sha 0762f2afeb74, PR #42562)
TITLE: [Perf][Feat] Add generic cuteDSL LL BF16 router (GEMM) (#42562)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/kernels.yaml (+5/-0); tests/kernels/test_ll_bf16_gemm.py (+553/-0); vllm/model_executor/kernels/linear/cute_dsl/__init__.py (+0/-0); vllm/model_executor/kernels/linear/cute_dsl/_ll_bf16_dotprod.py (+313/-0); vllm/model_executor/kernels/linear/cute_dsl/_ll_bf16_splitk.py (+585/-0); vllm/model_executor/kernels/linear/cute_dsl/ll_bf16.py (+282/-0); vllm/model_executor/layers/fused_moe/router/gate_linear.py (+46/-8); vllm/model_executor/warmup/kernel_warmup.py (+28/-0)
LABELS: ready, ci/build
DEEP_STUDY: deep-study performance PR ()
BODY: ## Motivation ⏎  ⏎  vLLM already ships a LL router GEMM compatible with DeepSeek-V3/V4-Pro requirements, but it is locked to specific shapes: ⏎  ⏎  ```python ⏎   DSV3_SUPPORTED_NUM_EXPERTS = [256, 384] ⏎   DSV3_SUPPORTED_HIDDEN_SIZES = [7168] ⏎ ``` ⏎   ⏎ Models that fall outside these dimensions (e.g. DeepSeek-V4-Flash with K=14400, N=256) silently fall back to cuBLAS, leaving performance on the table. ⏎  ⏎ This PR adds a generic cuteDSL router GEMM that is …[truncated]

### L3-c4f5cd60da  (L3, 2026-07-14, sha c4f5cd60dae3, PR #47327)
TITLE: [1/N] Add dense MHA path for sparse MLA short sequences (#47327)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases(introducing), corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_sparse, L3.mla.flashinfer_sparse, L3.mla.rocm_aiter_sparse, L3.mla.flashattn_sparse, L3.dispatch.abstract_interface
FILES: vllm/config/attention.py (+5/-0); vllm/model_executor/layers/attention/mla_attention.py (+315/-255); vllm/model_executor/layers/attention/sparse_mla_attention.py (+316/-0); vllm/v1/attention/backend.py (+4/-81); vllm/v1/attention/backends/mla/flashattn_mla_sparse.py (+36/-63); vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+37/-94); vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py (+4/-2); vllm/v1/attention/backends/mla/flashmla_sparse.py (+55/-73); vllm/v1/attention/backends/mla/indexer.py (+6/-3); vllm/v1/attention/backends/mla/prefill/flash_attn.py (+16/-3); (+11 more)
LABELS: documentation, performance, rocm, intel-gpu, ready, v1, nvidia
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:28158b2fc3 (fix PR 48886) || deep-study performance PR (system_performance)
BODY: Split up from https://github.com/vllm-project/vllm/pull/34744 ⏎  ⏎ ## Purpose ⏎ Adds a fast path when all prefills in a batch have `seq_len <= topk`, routing to dense MHA instead of sparse MQA (subject to a `q_len` threshold) to obtain a substantial speedup. ⏎  ⏎ This is also a precursor to a masked MHA implementation (see https://github.com/vllm-project/vllm/pull/34744), which will enable speedups at `seq_len > topk`. ⏎  ⏎ ## Test Plan ⏎  ⏎ ### Layer-lev …[truncated]

### L3-9e289c553c  (L3, 2026-07-14, sha 9e289c553c8c, PR #44462)
TITLE: up FI fp8 moe topk to 32 (#44462)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+1/-1)
LABELS: ready, nvidia
BODY: ## Purpose ⏎ The FlashInfer backend now supports topk up to 32. See [here](https://github.com/flashinfer-ai/flashinfer/blob/56d537a106024eb25f4d4a186eadc226990a9185/include/flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh#L699-L710)  ⏎  ⏎ The model I tested this on is not OSS so I cannot give a repro, but verified smoke tests work. ⏎  ⏎  ⏎ While there isn't some specific test to cover this, I've run: ⏎ ``` ⏎ pytest -v -s tests/kernels/moe/test_flashin …[truncated]

### L3-b50ef9c6ed  (L3, 2026-07-14, sha b50ef9c6ed97, PR #44849)
TITLE: [ROCm][MiniMax-M2] Dispatch fused QK-norm + AllReduce via AITER (#44849)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py (+12/-0)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ⚠️ **Requires:** AITER build containing ROCm/aiter#3163 (Optionally ROCm/aiter#3189 for additional performance) ⏎  ⏎ Co-developed with @pbkowalski who wrote the initial dispatch re-wiring  ⏎  ⏎ Adds ROCm dispatch for MiniMax-M2's QK-norm custom op to use AITER's fused kernel (`custom_fused_qknorm_ar` from ROCm/aiter#3163 + aiter#3189 further pushes the perf) ⏎  ⏎ Supersedes #42602 (closed as that used a compile-time FX pass which has now been removed u …[truncated]

### L3-95aab66e95  (L3, 2026-07-14, sha 95aab66e9585, PR #47984)
TITLE: [ROCm][MiniMax-M3][Spec Decode] Support speculative decode with AITER sparse PA (#47984)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_minimax_m3.py (+140/-0); vllm/models/minimax_m3/amd/ops/sparse_pa.py (+41/-10); vllm/models/minimax_m3/amd/sparse_attention_msa.py (+1/-5)
LABELS: rocm, ready, v1
BODY: ## Summary ⏎  ⏎ This PR enables MiniMax-M3 speculative decode on the ROCm AITER sparse paged-attention path introduced in #47287. ⏎  ⏎ The core change maps flattened speculative decode rows back to their request-local query positions: ⏎  ⏎ ```text ⏎ req_id = row // decode_query_len ⏎ local_q = row % decode_query_len ⏎ query_abs_pos = seq_lens[req_id] - decode_query_len + local_q ⏎ ``` ⏎  ⏎ Then it reuses the existing prefill sparse block-table builder to pro …[truncated]

### L3-b2f7d2560a  (L3, 2026-07-14, sha b2f7d2560ae6, PR #48520)
TITLE: [Bugfix] Make MLA+SWA check the layer's backend, not the model config (#48520)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/attention.py (+2/-2)
LABELS: bug, speculative-decoding, ready
BODY: ## Purpose ⏎  ⏎ `Attention.get_kv_cache_spec` asserted on the model-level `model_config.use_mla`, rejecting a regular sliding-window layer in a drafter merely because the target model is MLA. Replaced with a per-layer check as `self.attn_backend.is_mla()`, preserving the real invariant (an MLA layer must not use a sliding window) without the false positive. Fixes https://huggingface.co/shanjiaz/dspark-mistral-small-119b ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test R …[truncated]

### L3-50ac1c7bab  (L3, 2026-07-14, sha 50ac1c7bab47, PR #45781)
TITLE: [Misc] Rename VLLM_TRITON_ATTN_USE_TD to VLLM_TRITON_USE_TD (#45781)
SOURCES: path_core, path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.triton.unified_attention, L3.triton.v1_backend
FILES: vllm/envs.py (+22/-3); vllm/v1/attention/backends/triton_attn.py (+3/-3); vllm/v1/attention/ops/triton_unified_attention.py (+1/-1)
LABELS: ready, v1, verified
BODY: # [Misc] Rename `VLLM_TRITON_ATTN_USE_TD` to `VLLM_TRITON_USE_TD` ⏎  ⏎ ## Purpose ⏎  ⏎ Follow-up to the tensor-descriptor (TD) pilot #40327 and the TD-adoption RFC #42545. The pilot shipped the TD opt-in behind the attention-specific `VLLM_TRITON_ATTN_USE_TD`; the RFC extends TD across vLLM's Triton kernels behind a single, general flag. This renames the variable to `VLLM_TRITON_USE_TD`. ⏎  ⏎ The old name stays registered (so it does not trip the unkno …[truncated]

### L3-7e950521b3  (L3, 2026-07-14, sha 7e950521b39f, PR #48428)
TITLE: fix: size FlashInfer prefill workspace to batch head footprint (#48428)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+19/-0)
LABELS: bug, ready, v1, nvidia
BODY: The fixed 394 MiB workspace default overflows for wide-head models at the 8192-token prefill chunk on sm_120, where FlashInfer hard-errors on batch_prefill_tmp_v instead of growing. Size it to max_num_batched_tokens ⏎ * num_qo_heads * head_dim, floored at the configured default. ⏎  ⏎  ⏎  ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-37aa52821d  (L3, 2026-07-14, sha 37aa52821df8, PR #48174)
TITLE: Build with ABI stable FlashMLA (#48174)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_build
FILES: cmake/external_projects/flashmla.cmake (+18/-13)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ Part of https://github.com/vllm-project/vllm/issues/26946. Test out stable FlashMLA extensions by moving the pin past https://github.com/vllm-project/FlashMLA/pull/15 and https://github.com/vllm-project/FlashMLA/pull/16 ⏎  ⏎ ## Test Plan ⏎  ⏎ CI should still all be green.   ⏎  ⏎ In the standalone fork: 4748 tests passed. ⏎ ``` ⏎   ┌────────────────────────────────────────────────┬──────────────────────────────┐ ⏎   │                   Test f …[truncated]

### L3-05d4f8bba3  (L3, 2026-07-14, sha 05d4f8bba3aa, PR #48647)
TITLE: [ROCm][CI] fix flashinfer import check (#48647)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/attention/test_flashinfer_dcp_spec_reorder.py (+6/-4)
LABELS: rocm, ready, v1, nvidia
BODY: cuda platform check needs to be done before from flashinfer backend import ⏎  ⏎ Resolves ([buildkite test fail link](https://buildkite.com/vllm/ci/builds/78059/canvas?jid=019f61ac-fb57-42cd-93c9-8bcd99ee3c9f&tab=output)):  ⏎ ``` ⏎ from vllm.v1.attention.backends import flashinfer as flashinfer_backend ⏎ /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py:12: in <module> ⏎ from flashinfer import ( ⏎ E   ModuleNotFoundError: N …[truncated]

### L3-313d01f507  (L3, 2026-07-14, sha 313d01f507fc, PR #48631)
TITLE: [CI][Bugfix] Fix FlashAttention reported MLA dimension support (#48631)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/prefill/base.py (+20/-8); vllm/v1/attention/backends/mla/prefill/flash_attn.py (+19/-9); docs/design/attention_backends.md (+1/-1); tests/v1/attention/test_mla_prefill_selector.py (+34/-18); tools/pre_commit/generate_attention_backend_docs.py (+139/-0)
LABELS: bug, documentation, ready, v1
BODY: Alternative to https://github.com/vllm-project/vllm/pull/48609 ⏎  ⏎ ## Purpose ⏎ FA2 and FA3 support GLM-5 dimensions. FA4 does not. Report this correctly. ⏎  ⏎ Fixes the following CI failures (build [#77969](https://buildkite.com/vllm/ci/builds/77969)): ⏎  ⏎ - **Basic Models Tests (Extra Initialization) 1** — `test_can_initialize_large_subset[Glm4MoeLiteForCausalLM]` ⏎ - **Basic Models Tests (Extra Initialization) 2** — `test_can_initialize_large_subset …[truncated]
