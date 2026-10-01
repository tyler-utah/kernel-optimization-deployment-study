### L2-4aadf94146  (L2, 2026-07-15, sha 4aadf94146b1, PR #30795)
TITLE: [Kernel] Relocate vendored fla and mamba kernel trees to sglang.kernels (RFC #29630, Phase 2.5, 7/7) (#30795)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.dispatch.attention_registry
FILES: benchmark/bench_linear_attention/bench_cutedsl_kda_decode.py (+2/-2); benchmark/bench_linear_attention/bench_fused_gate_cumsum.py (+3/-3); benchmark/bench_linear_attention/bench_gdn_decode.py (+2/-2); benchmark/bench_linear_attention/bench_gdn_prefill.py (+2/-2); benchmark/bench_linear_attention/bench_gdn_prefill_cutedsl.py (+4/-4); benchmark/bench_linear_attention/bench_kda_decode.py (+2/-2); benchmark/bench_linear_attention/bench_kda_prefill_cutedsl.py (+1/-1); benchmark/fla/benchmark_layernorm_gated.py (+2/-2); benchmark/kernels/decoding_attention_triton/triton_flashinfer_cudnn.py (+1/-1); python/sglang/kernels/ops/attention/__init__.py (+16/-0); (+78 more)
LABELS: deepseek, blackwell, run-ci, mthreads, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ Phase 2.5 sweep **7 of 7** (migration plan in RFC #29630): pure directory relocations of the two vendored linear-state kernel libraries. Large diffstat, zero logic change — kept separate precisely so the trivial-but-huge diff doesn't drown the substantive sweeps. ⏎  ⏎ ## Modifications ⏎  ⏎ | From | To | Content | ⏎ |---|---|---| ⏎ | `srt/layers/attention/fla/` | `ops/attention/fla/` | flash-linear-attention port: 34 Triton kernels / 18 files ( …[truncated]

### L2-e78051a419  (L2, 2026-07-15, sha e78051a4192a, PR #30355)
TITLE: [AMD] [Fix] Fix --attention-backend triton work for DeepSeek MLA on MI355 (null-K + decode dispatch + RoPE) (#30355)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_backend_handler.py (+5/-1); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+8/-7); test/registered/unit/models/test_deepseek_mla_dispatch.py (+67/-0)
LABELS: amd, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ DeepSeek MLA models could not run with `--attention-backend triton` on gfx95 (MI300/MI355) when `SGLANG_USE_AITER=1` (the ROCm image default). Triton is the common NVIDIA/AMD path and should not GPU-fault or produce wrong output on a valid backend selection. ⏎  ⏎ Root cause is a recurring anti-pattern: several gfx95/aiter fused MLA paths are gated on env vars (`SGLANG_USE_AITER`, `SGLANG_ROCM_FUSED_DECODE_MLA`) or on "not in the absorb …[truncated]

### L2-3101c1258c  (L2, 2026-07-15, sha 3101c1258c4d, PR #30012)
TITLE: [DSv4] Use BF16 instead of FP32 for indexer score computation (#30012)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/dsv4/indexer.py (+4/-4)
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ Switch the sparse MLA indexer's QK score BMM from FP32 to BF16, enabling Tensor Core (WMMA) dispatch instead of FP32 SIMT SGEMM. ⏎  ⏎     This eliminates: ⏎     - FP32 SIMT SGEMM (cutlass_80_simt_sgemm): 408ms -> 95ms via Tensor Core ⏎     - Associated BF16<->FP32 dtype conversion copies: ~190ms -> 0ms ⏎      ⏎     Benchmarks on 8x RTX PRO 5000 72GB (DeepSeek-V4 Flash, TP8): ⏎     - E2E serving (cc=8, in=4096, out=150): +19% output thro …[truncated]

### L2-dc60f65661  (L2, 2026-07-15, sha dc60f6566123, PR #31385)
TITLE: chore: bump tokenspeed_mla to 0.1.8 (#31385)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎ Bump `tokenspeed_mla` from 0.1.7 to 0.1.8 ([upstream release](https://github.com/lightseekorg/tokenspeed/pull/504)). ⏎  ⏎ Note: 0.1.9 is not currently installable with sglang — it hard-pins `apache-tvm-ffi==0.1.12`, which conflicts with `sgl-deep-gemm==0.1.4` (pins `apache-tvm-ffi==0.1.11`, no newer release available). 0.1.8 keeps the same dependency envelope as 0.1.7 (`apache-tvm-ffi<=0.1.11,>=0.1.5`, same `tokenspeed-triton` floor),  …[truncated]

### L2-01b003255a  (L2, 2026-07-16, sha 01b003255a0b, PR #26852)
TITLE: [AMD]Reuse fused FP8 KV cache write on standard aiter prefill/decode (#26852)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+42/-6); test/registered/attention/test_fused_fp8_kv_write.py (+437/-0)
LABELS: amd, run-ci, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎ The fused bf16→fp8 KV cache write kernel (launch_reshape_and_cache_flash) was only enabled on the unified-attention path (use_triton_unified_attention). On the standard aiter path (read via paged_attention_ragged), FP8 KV writes fell back to set_kv_buffer, which performs a separate divide + cast + store_kvcache — i.e. extra elementwise kernels and intermediate DRAM traffic per K/V per attention layer. ⏎  ⏎ The KV write only produces  …[truncated]

### L2-e2d021d4ab  (L2, 2026-07-16, sha e2d021d4ab2f, PR #30238)
TITLE: [AMD] Support two batch overlap with MTP on DeepSeekV4 (#30238)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/dsv4/compress.py (+15/-0); python/sglang/srt/models/deepseek_v4.py (+3/-1); test/registered/amd/test_deepseek_v4_pro_fp4_tbo_mtp.py (+149/-0)
LABELS: deepseek, run-ci, jit-kernel
BODY: ## Motivation ⏎ TBO failed when enable DeepSeek V4 MTP ⏎  ⏎ <img width="1292" height="451" alt="image" src="https://github.com/user-attachments/assets/91f96a7c-3c0b-4780-8f2f-e00d00452ef2" /> ⏎  ⏎ ## Modifications ⏎  ⏎ Codebase allowed `TARGET_VERIFY` requests to enter the prefill-only TBO strategy by mistake. The gate now uses `is_extend_without_speculative()`, so only real prefill/extend batches use the prefill TBO path, while decode/target-verify are …[truncated]

### L2-22453ca63c  (L2, 2026-07-16, sha 22453ca63c46, PR #31390)
TITLE: docker: build HPC-Ops into the GPU image (#31390)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+34/-2)
LABELS: run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Per the discussion with the Hunyuan HPC-Ops team: depend on the `hpc` library directly in our image so that (a) the opt-in `hpc_ops` attention (#30540) and MoE runner (#30541) backends work out of the box (`import hpc` today requires a manual source build), and (b) future kernel-side improvements land by bumping one pinned commit, with no re-porting — the Python API stays stable. ⏎  ⏎ HPC-Ops is MIT-licensed, so bundling it in the imag …[truncated]

### L2-e5f9804e26  (L2, 2026-07-16, sha e5f9804e26c4, PR #31241)
TITLE: Refining fused A GEMM dispatch (#31241)
SOURCES: path_core
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+2/-4); python/sglang/jit_kernel/csrc/gemm/dsv3_fused_a_gemm.cuh (+41/-22); python/sglang/jit_kernel/fused_a_gemm.py (+0/-10); python/sglang/srt/models/deepseek_v2.py (+17/-0); test/registered/jit/test_dsv3_fused_a_gemm.py (+6/-4)
LABELS: deepseek, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: For num_tokens 1 to 16, this will be better than the best selected tactic of CuteDSL GEMM. However for num tokens from 17 to 96, the DSL one can still achieve better than https://github.com/sgl-project/sglang/pull/30117 ⏎  ⏎ The idea of tileM = 32 is created by @zhou9402 https://github.com/vllm-project/vllm/pull/48597, thanks to the author. ⏎  ⏎ 1. Do not override the fused a GEMM, as in it's applied scenario, it will always be better. ⏎ 2. Use it for …[truncated]

### L2-b296e1a503  (L2, 2026-07-16, sha b296e1a5035b, PR #30535)
TITLE: [hicache]: add  mamba concurrency io transfer kernel (#30535)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/csrc/kvcacheio/transfer_mamba.cuh (+188/-0); python/sglang/jit_kernel/transfer_mamba.py (+83/-0); python/sglang/srt/mem_cache/memory_pool_host.py (+38/-85); test/registered/jit/test_transfer_mamba.py (+349/-0)
LABELS: run-ci, jit-kernel, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ SGLang's HiCache supports Mamba models (e.g., Qwen-Next), but the Mamba state transfer between host and device previously relied on `direct` I/O backend only (pure DMA). This PR introduces a dedicated JIT-compiled CUDA kernel for Mamba KV cache transfer, providing a `kernel` I/O backend option that uses cache-bypass loads/stores with software pipelining for efficient PCIe transfer. The block-cooperative design achieves up to **14 …[truncated]

### L2-9910ef8167  (L2, 2026-07-16, sha 9910ef816754, PR #31090)
TITLE: flashmla: sync-free spec via device-side draft-extend (#31090)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.flashmla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashmla_backend.py (+139/-42); python/sglang/srt/speculative/draft_utils.py (+3/-8); python/sglang/srt/speculative/eagle_worker_v2.py (+6/-0)
LABELS: high priority, run-ci, run-ci-extra
BODY: Extend the sync-free treatment to the FlashMLA backend: device-side fixed-q draft-extend (pad/unpad, no flashinfer host-planned plan), preallocated tree-mask scratch, static block-table bounds when the CPU mirror is absent, and `needs_cpu_seq_lens = False`. Also declare the flag on the fa3/flashmla multi-step draft wrappers so `decide_needs_cpu_seq_lens` sees it. Follows the pattern of #29589. ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_ …[truncated]

### L2-ead703815e  (L2, 2026-07-16, sha ead703815ef3, PR #31514)
TITLE: [DCP] Enable decode context parallel for Kimi K2.5 NVFP4 (#31514)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/dcp/comm.py (+1/-1); python/sglang/srt/models/kimi_k25.py (+29/-0)
LABELS: blackwell
BODY: ## Summary ⏎  ⏎ Enables decode context parallelism (DCP) end-to-end for `nvidia/Kimi-K2.5-NVFP4`, which previously crashed under `--dcp-size > 1`. Two model-side wiring gaps are fixed, plus a registered nightly test. ⏎  ⏎ - **`kimi_k25.py`**: `KimiK25ForConditionalGeneration` now delegates `prepare_context_parallel_metadata_for_dcp` to its inner DeepSeek-V3 `language_model`. The extend runner (`eager_runner._execute_extend`) only builds `attn_dcp_met …[truncated]

### L2-96dd96c02b  (L2, 2026-07-16, sha 96dd96c02baa, PR #31532)
TITLE: [DCP] Auto-disable tc_piecewise and breakable prefill CUDA graphs under DCP (#31532)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+10/-0)
BODY: ## Summary ⏎  ⏎ When decode context parallel (DCP, `dcp_size > 1`) is enabled, the extend ⏎ attention path is metadata-dependent (`attn_dcp_metadata` / ⏎ `dcp_local_prefix_kv_indices`). Both prefill CUDA graph capture backends ⏎ build a *dummy* extend forward batch with `attn_dcp_metadata=None`, so the ⏎ MLA prepare logic dereferences `None` and crashes at capture time. ⏎  ⏎ This PR gates both prefill CUDA graph backends off DCP in `server_args`: ⏎  ⏎ - `_disable_tc …[truncated]

### L2-6e3be088a9  (L2, 2026-07-16, sha 6e3be088a9b7, PR #31527)
TITLE: [Spec] Allocate the verify tree-mask scratch on the target backend only (#31527)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+3/-1); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+16/-0); python/sglang/srt/layers/attention/flashattention_backend.py (+4/-1); python/sglang/srt/layers/attention/triton_backend.py (+3/-1)
LABELS: deepseek, blackwell, run-ci
BODY: The worker fetches the tree-mask scratch (get_verify_buffers_to_fill_after_draft) from the target attention backend only, but fa3 / trtllm_mla / triton also allocated it on their draft-extend instances -- a dead worst-case FULL_MASK buffer (hundreds of MB at long context). Gate the allocation with is_draft_runner, and add the missing DSV4 override + preallocation so build_tree stops dynamically allocating a bs * max_context_len mask every verify  …[truncated]

### L2-d67aa05697  (L2, 2026-07-17, sha d67aa0569727, PR #31502)
TITLE: Bump FlashInfer to 0.6.15 and revert regressions (#31502)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+15/-2); python/sglang/srt/models/deepseek_v2.py (+0/-8); python/sglang/srt/utils/common.py (+1/-1); test/registered/models_e2e/test_dsa_glm52_nvfp4_tp_mtp.py (+1/-1)
LABELS: high priority, dependencies, deepseek, blackwell, run-ci, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study revert record: partial_revert of PR(s)  reason=crash_or_hang || deep-study: this PR was reverted by PR 31625 (confirmed_revert, reason=performance_regression)
BODY: --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #29549992666](https://github.com/sgl-project/sglang/actions/runs/29549992666) ⏎ Latest PR Test (Extra): :no_entry_sign: [Run #29549992592](https://github.com/sgl-project/sglang/actions/runs/29549992592)

### L2-2c64b7782e  (L2, 2026-07-17, sha 2c64b7782e03, PR #31368)
TITLE: [AMD][PD] Fix early-send cached-prefix KV racing the prefill forward on mori (#31368)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/mori/conn.py (+9/-0); python/sglang/srt/disaggregation/prefill.py (+9/-0)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ Fixes a decode-accuracy regression from #29316 (early-send of cached-prefix KV) on the **mori** transfer backend, by adding the forward-completion synchronization that the RDMA read was missing. Early-send stays **enabled** — this is a "fix forward", not a disable. ⏎  ⏎ Supersedes #31239 (which disabled the optimization globally). Thanks @cctry for pushing back on that approach and for the AMD-RDMA hint — it was correct. ⏎  ⏎ ## Root cause ⏎  ⏎ ` …[truncated]

### L2-7355e0cb87  (L2, 2026-07-17, sha 7355e0cb875f, PR #30731)
TITLE: [NPU] custom-ops adapt (#30731)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/npu.Dockerfile (+12/-0); .github/workflows/release-docker-npu-nightly.yml (+1/-1); .github/workflows/release-docker-npu.yml (+1/-1); scripts/ci/npu/npu_ci_install_dependency.sh (+14/-1)
LABELS: npu, run-ci
BODY: --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #29400990590](https://github.com/sgl-project/sglang/actions/runs/29400990590) ⏎ Latest PR Test (Extra): :x: [Run #29400990408](https://github.com/sgl-project/sglang/actions/runs/29400990408)

### L2-ec6a3163b7  (L2, 2026-07-17, sha ec6a3163b7ac, PR #21601)
TITLE: [Feature] Add FP4 KV Cache Design and support SM120 GPUs (#21601)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.pool.mla_token_kv, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/attention_registry.py (+24/-8); docs_new/docs/advanced_features/attention_backend.mdx (+2/-1); docs_new/docs/advanced_features/quantized_kv_cache.mdx (+8/-3); docs_new/docs/advanced_features/server_arguments.mdx (+2/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+244/-35); python/sglang/srt/layers/attention/trtllm_mha_backend.py (+96/-22); python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py (+480/-69); python/sglang/srt/layers/quantization/kvfp4_tensor.py (+61/-35); python/sglang/srt/mem_cache/kv_cache_configurator.py (+51/-21); python/sglang/srt/mem_cache/kv_cache_dtype.py (+13/-4); (+9 more)
LABELS: documentation, quant, blackwell, run-ci, bypass-fastfail
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Summary ⏎   Add NVFP4 (FP4 E2M1) KV cache quantization support for SM120 GPUs, reducing KV cache memory by ~2x compared to FP8 with no accuracy loss on GSM8K/GPQA/LongBenchV2-128k. ⏎  ⏎   ### Usage ⏎  ⏎   ```bash ⏎ # server ⏎   python3 -m sglang.launch_server \ ⏎       --model-path Qwen/Qwen3.5-35B-A3B-FP8 \ ⏎       --kv-cache-dtype nvfp4 \ ⏎       --prefill-attention-backend flashinfer \ ⏎       --decode-attention-backend trtllm_mha \ ⏎       --disable-r …[truncated]

### L2-0ad0ff2e9e  (L2, 2026-07-17, sha 0ad0ff2e9ee4, PR #31618)
TITLE: chore: bump sglang-kernel version to 0.4.5 (#31618)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sglang-kernel` version to `0.4.5` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.4.5` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the `sglang-kernel` dependency are updat …[truncated]

### L2-304a529558  (L2, 2026-07-17, sha 304a52955834, PR #31625)
TITLE: Revert "Bump FlashInfer to 0.6.15 and revert regressions" (#31625)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+2/-15); python/sglang/srt/models/deepseek_v2.py (+8/-0); python/sglang/srt/utils/common.py (+1/-1); test/registered/models_e2e/test_dsa_glm52_nvfp4_tp_mtp.py (+1/-1)
LABELS: dependencies, deepseek, blackwell
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 31502 reason=performance_regression
BODY: Reverts sgl-project/sglang#31502, since the new flashinfer version caused performance regression ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #29621280754](https://github.com/sgl-project/sglang/actions/runs/29621280754) ⏎ Latest PR Test (Extra): :x: [Run #29621280651](https://github.com/sgl-project/sglang/actions/runs/29621280651)

### L2-24a8944e15  (L2, 2026-07-17, sha 24a8944e151f, PR #31391)
TITLE: fix: enable Kimi multimodal breakable prefill cuda graph replay (#31391)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/radix_attention.py (+4/-9); python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py (+1/-0); python/sglang/srt/model_executor/model_runner_components/layer_setup.py (+12/-7); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+31/-21); python/sglang/srt/model_executor/runner_backend_utils/tc_piecewise_cuda_graph/context_manager.py (+3/-0); test/registered/unit/configs/test_multimodal_piecewise_cuda_graph.py (+47/-0); test/registered/unit/model_executor/model_runner_components/test_layer_setup.py (+36/-0)
LABELS: Multi-modal, run-ci
BODY: ## What changed ⏎  ⏎ - Keep multimodal `input_embeds` in breakable prefill CUDA graph capture/replay. ⏎ - Register the DeepSeek-style MHA companion on NVIDIA as well as HIP and restore the captured one-shot MHA state during replay. ⏎ - Keep non-zero-prefix Kimi chunks eager until that distinct KV-materialization path has its own graph. ⏎ - Add CPU regression coverage for multimodal graph eligibility, the non-zero-prefix gate, and MHA companion discovery. ⏎  ⏎  …[truncated]

### L2-7a896215e7  (L2, 2026-07-17, sha 7a896215e765, PR #31619)
TITLE: [CP] Migrate MLA prefill CP (DeepSeek V3) to CP-v2 zigzag strategy (#31619)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.fa3_fa4_mla
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+32/-14); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+8/-2); python/sglang/srt/models/deepseek_nextn.py (+8/-5); python/sglang/srt/models/deepseek_v2.py (+9/-7); python/sglang/srt/layers/cp/utils.py (+1/-0); python/sglang/srt/layers/cp/zigzag.py (+15/-0); python/sglang/srt/model_executor/runner/eager_runner.py (+50/-48)
LABELS: deepseek, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Part of the Prefill CP refactor roadmap (#27252) and the Context Parallelism roadmap (#21788). #28421 landed the CP-v2 `ZigzagCPStrategy` for GQA/MHA (Qwen3) and the model-agnostic eager-runner boundary. This PR ports **MLA prefill CP for DeepSeek V3/R1** (originally #23292) onto that abstraction and makes the strategy own the MLA latent all-gather. ⏎  ⏎ `DeepseekV3ForCausalLM` is added to `CP_V2_DEFAULT_MODEL_CLASSES`, so CP-v2 is its …[truncated]

### L2-72c4ed1a3f  (L2, 2026-07-17, sha 72c4ed1a3ff9, PR #31468)
TITLE: [Spec] DFlash: remove per-step host syncs so the CPU runs a full step ahead (spec-v2 overlap) (#31468)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+7/-4); python/sglang/kernels/ops/speculative/cache_locs.py (+68/-0); python/sglang/srt/environ.py (+2/-0); python/sglang/srt/layers/attention/hybrid_attn_backend.py (+6/-0); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/speculative/dflash_info_v2.py (+13/-3); python/sglang/srt/speculative/dflash_worker_v2.py (+146/-40); python/sglang/srt/speculative/eagle_info.py (+6/-1); python/sglang/srt/speculative/ngram_info.py (+7/-2); test/registered/unit/model_executor/model_runner_components/test_attention_backend_setup.py (+2/-0); (+1 more)
LABELS: blackwell, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Under spec-v2 overlap scheduling, DFlash stalls the host every decode step: nsys shows a large `cudaStreamSynchronize` between the draft and verify cuda graphs, so `cudaGraphLaunch` for step N+1 is only issued **after** step N has finished (launch lead ≈ −50 µs). EAGLE3 spec-v2 runs a full step ahead; DFlash should too. This PR removes every per-step host↔device sync on that path. Result: the CPU queues step N+1 a full step early …[truncated]

### L2-faf6894093  (L2, 2026-07-18, sha faf68940939a, PR #30272)
TITLE: Implement SM120 DeepSeek V4 flashinfer_mxfp4 moe runner backend + TP2 (#30272)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.kernel.flash_mla_sm120, L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/kernels/ops/attention/flash_mla_sm120.py (+33/-23); docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+3/-4); docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx (+3/-4); python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h (+2/-5); python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py (+7/-4); python/sglang/srt/arg_groups/overrides.py (+5/-6); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+8/-3); python/sglang/srt/layers/attention/dsv4/indexer.py (+19/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py (+2/-1); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py (+45/-22); (+8 more)
LABELS: documentation, dependencies, deepseek, run-ci, jit-kernel, run-ci-extra
BODY: ## Motivation ⏎  ⏎ DeepSeek-V4-Flash (FP8 checkpoint, MXFP4 experts) could not be served on SM120 (Blackwell desktop/workstation, e.g. RTX PRO 6000). The failures encountered while bringing up TP2/TP4 on 2–4 GPUs were: ⏎  ⏎ 1. The first MoE forward crashed because Marlin dereferenced masked `-1` expert blocks under EP. ⏎ 2. TP2 could not allocate a viable KV cache because FP32 loader containers inflated expert scales by about 16 GB/rank at EP2. ⏎ 3. The firs …[truncated]

### L2-c68392c535  (L2, 2026-07-19, sha c68392c535e9, PR #31675)
TITLE: [AMD] Fix DeepSeek MLA prefill shape mismatch on HIP eager fallback (missing mha_companion_layers) (#31675)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/runner/eager_runner.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎ PR31391, DeepSeek MLA prefill on AMD (HIP) crashes when the prefill runner uses `tc_piecewise`: ⏎     RuntimeError: shape '[-1, 16, 576]' is invalid for input of size 36003840 ⏎      ⏎ failed run : https://github.com/sgl-project/sglang/actions/runs/29644059344/job/88079191396 ⏎ ### Root cause ⏎ PR #31391 replaced the per-layer `_pcg_mha_companion` attribute (used to swap `attn_mqa` -> `attn_mha` so the attention backend sees the correct …[truncated]

### L2-b8ec544946  (L2, 2026-07-19, sha b8ec544946f1, PR #30514)
TITLE: [DSA] Integrate Q8KV8 FP8 Sparse MLA Prefill into the DSA Backend (DeepSeek-V3.2) (#30514)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/kernels/ops/attention/dsa/dequant_k_cache.py (+132/-0); python/sglang/kernels/ops/attention/utils.py (+6/-0); python/sglang/srt/layers/attention/dsa_backend.py (+211/-9); python/sglang/srt/server_args.py (+1/-0); docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx (+1/-1); docs_new/docs/advanced_features/attention_backend.mdx (+6/-0); docs_new/docs/advanced_features/server_arguments.mdx (+1/-1); python/sglang/jit_kernel/csrc/sparse_mla_q8kv8_prefill_sm90/kernel.cuh (+190/-83); python/sglang/kernels/ops/kvcache/cache_ops.py (+67/-0); test/registered/jit/test_sparse_mla_q8kv8_prefill_sm90.py (+186/-1)
LABELS: documentation, quant, deepseek, run-ci, jit-kernel, bypass-fastfail
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Summary ⏎ This PR integrates the SM90 Q8KV8 FP8 sparse-MLA prefill kernel from PR-1 (#25751) into SGLang's **DSA prefill backend** (`dsa_backend.py`), so DeepSeek-V3.2 serving runs **native FP8 query × FP8 KV** sparse attention during prefill end-to-end. It is opt-in via `--dsa-prefill-backend flashmla_sparse_q8` (the `--nsa-prefill-backend` alias still works; requires `fp8_e4m3` KV; the bf16 baseline remains `flashmla_sparse`), and adds the co …[truncated]

### L2-688a6d23f1  (L2, 2026-07-19, sha 688a6d23f144, PR #31705)
TITLE: [DeepSeek-V4] Fix idle-rank dummy-extend sparse-prefill crash under DP breakable CUDA graph (#31705)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/deepseek_v4_backend.py (+2/-0)
LABELS: high priority, deepseek, run-ci, bypass-fastfail
DEEP_STUDY: deep-study correctness case sglang:688a6d23f1: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=#30898
BODY: ## Problem ⏎  ⏎ `test_deepseek_v4_flash_fp4_b200.py` (base-c b200 deepep, BCG recipe: TP4/DP4/DeepEP, `--enable-mixed-chunk --cuda-graph-backend-prefill breakable`) crashes deterministically on idle DP ranks: ⏎  ⏎ ``` ⏎ _forward_prefill_sparse: assert core_attn_metadata.c4_sparse_raw_indices is not None ⏎ AssertionError: sparse-prefill c4 path requires c4_sparse_raw_indices ⏎ ``` ⏎  ⏎ ## Root cause ⏎  ⏎ Latent defect from #30898, surfaced by #31487. ⏎  ⏎ Under DP attentio …[truncated]

### L2-a03ca46a28  (L2, 2026-07-19, sha a03ca46a2847, PR #31474)
TITLE: Fix KDA prefix caching under mamba extra_buffer and enable it for kimi_linear (#31474)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+3/-1); python/sglang/kernels/ops/attention/fla/kda.py (+9/-3); python/sglang/srt/arg_groups/overrides.py (+1/-0); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+4/-0); python/sglang/srt/layers/attention/linear/kda_backend.py (+43/-2); python/sglang/srt/layers/attention/linear/kernels/kda_cutedsl.py (+7/-0); python/sglang/srt/layers/attention/linear/kernels/kda_flashkda.py (+7/-2); python/sglang/srt/layers/attention/linear/kernels/kda_triton.py (+2/-0); python/sglang/test/kits/kl_divergence_kit.py (+3/-0); python/sglang/test/kl_test_utils.py (+39/-9); (+1 more)
LABELS: blackwell, run-ci, bypass-fastfail, run-ci-extra
BODY: Enable kimi_linear to run with `--mamba-radix-cache-strategy extra_buffer`, which is required for attention backends with page_size > 1 (e.g. trtllm_mla, since the KDA mamba pool otherwise requires page_size=1). ⏎  ⏎ - The KDA backend never wrote mamba track snapshots, so the conv/SSM states donated to the radix cache on request completion were garbage and prefix-cache hits restored wrong KDA state. Wire up the same track path GDN uses (decode: cop …[truncated]

### L2-02236fa38c  (L2, 2026-07-19, sha 02236fa38cb0, PR #31681)
TITLE: Add Inkling model support (#31681)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.kernel.cutedsl_flash_fwd_mla_sm100, L2.dispatch.attention_registry
FILES: .codespellrc (+1/-1); docker/Dockerfile (+5/-0); docker/rocm.Dockerfile (+33/-12); python/pyproject.toml (+8/-0); python/pyproject_other.toml (+9/-1); python/sglang/jit_kernel/csrc/inkling/causal_conv1d.cuh (+217/-0); python/sglang/jit_kernel/csrc/inkling/draft_extend_sconv.cuh (+147/-0); python/sglang/jit_kernel/csrc/inkling/fused_decode_update.cuh (+187/-0); python/sglang/jit_kernel/csrc/inkling/gather_scatter_sconv.cuh (+109/-0); python/sglang/jit_kernel/csrc/inkling/inkling_all_reduce.cuh (+672/-0); (+269 more)
LABELS: documentation, high priority, quant, amd, dependencies, lora, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
BODY: Supersedes #31358 — rebased on the latest `main`. ⏎  ⏎ ## Motivation ⏎  ⏎ Add serving support for the **Inkling** model family — a hybrid-attention Mixture-of-Experts architecture that interleaves sliding-window and full softmax attention with Mamba2 linear-attention layers, an NVFP4-quantized MoE, optional vision/audio multimodal towers, and native multi-token-prediction (MTP) speculative decoding. ⏎  ⏎ ## Modifications ⏎  ⏎ - **Model**: `InklingForConditionalG …[truncated]

### L2-3d82dacd58  (L2, 2026-07-20, sha 3d82dacd580a, PR #31714)
TITLE: Bump CuTe DSL to 4.6.0 (#31714)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+3/-3); python/sglang/kernels/ops/attention/cute_utils/_tcgen05.py (+4/-4); python/sglang/multimodal_gen/test/test_utils.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1); scripts/ci/cuda/ci_install_dependency.sh (+0/-29)
LABELS: dependencies, run-ci, diffusion, run-ci-extra, release-highlight
BODY: ## Summary ⏎  ⏎ - CuTe DSL: `4.5.2` → `4.6.0` ⏎ - FlashAttention-4: `4.0.0b15` → `>=4.0.0b16` ⏎ - Quack: `>=0.4.1` → `>=0.6.1` ⏎ - `nvvm.Tcgen05GroupKind` → `nvvm.CTAGroupKind` ⏎ - `nvvm.tcgen05_commit_arrive` → `nvvm.tcgen05_commit` ⏎ - Updates the related CuTe DSL deprecation warning filter ⏎ - Removes the old CU13 wheel force-reinstall workaround, which is no longer needed with 4.6.0 ⏎  ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run # …[truncated]

### L2-11a4c2d057  (L2, 2026-07-22, sha 11a4c2d05771, PR #31814)
TITLE: config: read resolved config via namespace accessors (#31814)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.backend.sparse_mla_adapters, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+6/-6); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+6/-4); python/sglang/srt/batch_overlap/two_batch_overlap.py (+10/-5); python/sglang/srt/configs/inkling.py (+2/-2); python/sglang/srt/constrained/grammar_manager.py (+2/-1); python/sglang/srt/disaggregation/common/conn.py (+3/-6); python/sglang/srt/disaggregation/decode.py (+4/-6); python/sglang/srt/disaggregation/encode_grpc_server.py (+4/-3); python/sglang/srt/disaggregation/encode_server.py (+17/-22); python/sglang/srt/disaggregation/mooncake/conn.py (+4/-7); (+152 more)
LABELS: Multi-modal, deepseek, blackwell, npu, mthreads, apple-silicon
DEEP_STUDY: deep-study: this PR was reverted by PR 32100 (confirmed_revert, reason=crash_or_hang)
BODY: Stacked on https://github.com/sgl-project/sglang/pull/31813. Part of a stacked series introducing a structured RuntimeContext configuration API (resolved config read through domain namespaces; ServerArgs becomes the read-only record). Incremental and behavior-preserving. ⏎  ⏎ Migrate resolved-config reads from the flat get_server_args()/self.server_args ⏎ surface to the domain namespace accessors (get_model()/get_serving()/get_exec()/ ⏎ get_schedule()/ge …[truncated]

### L2-2f4f2362fb  (L2, 2026-07-22, sha 2f4f2362fbad, PR #31202)
TITLE: Delete sgl-kernel AOT `bmm_fp8`, use `flashinfer.bmm_fp8` (#31202)
SOURCES: path_core
ARTIFACT_HINTS: L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+1/-24); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py (+1/-1); python/sglang/kernels/ops/gemm/__init__.py (+28/-1); python/sglang/srt/layers/quantization/fp8_utils.py (+30/-0); python/sglang/srt/models/minicpm3.py (+1/-26); python/sglang/srt/models/sarvam_moe.py (+2/-1); sgl-kernel/CMakeLists.txt (+0/-1); sgl-kernel/csrc/common_extension.cc (+0/-6); sgl-kernel/csrc/common_extension_musa.cc (+0/-6); sgl-kernel/csrc/gemm/bmm_fp8.cu (+0/-75); (+5 more)
LABELS: amd, sgl-kernel, run-ci, mthreads, bypass-fastfail
BODY: No need to keep it. The supported CC is the same (SM89+) ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #29743907240](https://github.com/sgl-project/sglang/actions/runs/29743907240) ⏎ Latest PR Test (Extra): :x: [Run #29743907047](https://github.com/sgl-project/sglang/actions/runs/29743907047)

### L2-246b3c3eaf  (L2, 2026-07-22, sha 246b3c3eafde, PR #31666)
TITLE: [Kernel] Phase 3+4: move JIT infra + operator groups into sglang.kernels (RFC #29630) (#31666)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.tokenspeed_mla, L2.kernel.set_mla_kv_buffer, L2.kernel.concat_mla, L2.kernel.mla_kv_pack_quantize_fp8, L2.runner.cuda_graph_mla
FILES: python/sglang/jit_kernel/concat_mla.py (+1/-1); python/sglang/jit_kernel/__main__.py (+4/-4); python/sglang/jit_kernel/activation.py (+3/-166); python/sglang/jit_kernel/add_constant.py (+1/-1); python/sglang/jit_kernel/all_reduce.py (+2/-2); python/sglang/jit_kernel/awq_dequantize.py (+1/-1); python/sglang/jit_kernel/awq_marlin_repack.py (+1/-1); python/sglang/jit_kernel/benchmark/marker.py (+1/-1); python/sglang/jit_kernel/clamp_position.py (+1/-1); python/sglang/jit_kernel/diffusion/causal_conv3d_cat_pad.py (+1/-1); (+145 more)
LABELS: quant, lora, hicache, blackwell, run-ci, jit-kernel, bypass-maintenance, bypass-fastfail, run-ci-extra
BODY: RFC #29630 **Phase 3 + Phase 4 (consolidated)** — the whole JIT-namespace move lands as one PR. (Per request, the five per-group PRs are folded in here; #31693–#31697 are closed in favor of this.) ⏎  ⏎ ### Phase 3 — shared infra ⏎ `sglang/jit_kernel/utils/` → **`sglang/kernels/jit/`** (compile pipeline, arch, deps, common). ~90 import sites rewritten. `KERNEL_PATH` resolves via `find_spec('sglang.jit_kernel')`, so `csrc/`, `include/`, and the build CLI …[truncated]

### L2-f5dcbe8f14  (L2, 2026-07-22, sha f5dcbe8f142f, PR #32100)
TITLE: Revert RuntimeContext config-namespace reads/roles (#31813–#31817) (#32100)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+6/-6); python/sglang/srt/arg_groups/overrides.py (+11/-9); python/sglang/srt/batch_overlap/two_batch_overlap.py (+6/-10); python/sglang/srt/configs/inkling.py (+2/-2); python/sglang/srt/constrained/grammar_manager.py (+1/-2); python/sglang/srt/disaggregation/common/conn.py (+8/-5); python/sglang/srt/disaggregation/decode.py (+6/-4); python/sglang/srt/disaggregation/encode_grpc_server.py (+3/-4); python/sglang/srt/disaggregation/encode_server.py (+24/-20); python/sglang/srt/disaggregation/mooncake/conn.py (+8/-5); (+177 more)
LABELS: high priority, Multi-modal, deepseek, blackwell, npu, run-ci, mthreads, apple-silicon, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 31813;31814;31815;31816;31817 reason=crash_or_hang
BODY: Reverts the upper half of the RuntimeContext config-namespace migration — ⏎ **#31813, #31814, #31815, #31816, #31817** — which merged without full ⏎ verification and carries correctness defects: ⏎  ⏎ - **Load-time fusion fallback desync.** `determine_num_fused_shared_experts()` / ⏎   `_maybe_autodisable_shared_experts_fusion()` call `declare_load_time_override()`, ⏎   which updates the published `ServerArgs`, but the migrated construction-time ⏎   readers cons …[truncated]

### L2-0c29c8fece  (L2, 2026-07-22, sha 0c29c8fecee1, PR #31927)
TITLE: Bump FlashInfer to 0.6.15.post1 (#31927)
SOURCES: path_core, symbol_pickaxe, dependency_pin, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.trtllm_mla, L2.backend.sparse_mla_adapters, L2.runner.cuda_graph_mla
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+41/-0); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/dsa_backend.py (+22/-0); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+15/-2); python/sglang/srt/models/deepseek_v2.py (+0/-8); python/sglang/srt/utils/common.py (+1/-1); python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py (+1/-0); python/sglang/test/kits/attention_unittest/attention_methods/mla_attention.py (+1/-0); (+1 more)
LABELS: dependencies, deepseek, blackwell, run-ci, bypass-fastfail, release-highlight
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Summary ⏎  ⏎ - bump `flashinfer_python`, `flashinfer-cubin`, and the optional JIT cache from 0.6.14 to 0.6.15.post1 ⏎ - reland the FlashInfer 0.6.15 API compatibility and regression cleanup from #31502 ⏎ - restore the GLM-5.2 NVFP4 performance threshold after removing the obsolete correction-bias workaround ⏎  ⏎ ## Why ⏎  ⏎ The original 0.6.15 bump was reverted in #31625 after host-side regressions reduced long-context serving throughput. FlashInfer 0.6.15.po …[truncated]

### L2-977ea336cd  (L2, 2026-07-22, sha 977ea336cd3e, PR #32015)
TITLE: [Kernel] Phase 4 batch-2: migrate JIT operator groups into kernels.ops (no shims) (RFC #29630) (#32015)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.tokenspeed_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/tokenspeed_mla_backend.py (+1/-1); python/sglang/jit_kernel/benchmark/utils.py (+1/-1); python/sglang/jit_kernel/dsv4/attn.py (+3/-1); python/sglang/jit_kernel/tests/utils.py (+1/-1); python/sglang/kernels/ops/communication/all_reduce.py (+0/-0); python/sglang/kernels/ops/communication/mp.py (+0/-0); python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py (+0/-0); python/sglang/kernels/ops/gemm/cutedsl_dsv3_fused_a_gemm.py (+0/-0); python/sglang/kernels/ops/gemm/fp8_blockwise_gemm.py (+0/-0); python/sglang/kernels/ops/gemm/fused_a_gemm.py (+2/-2); (+85 more)
LABELS: quant, deepseek, hicache, blackwell, run-ci, jit-kernel, bypass-maintenance, bypass-fastfail, run-ci-extra
BODY: RFC #29630 Phase 4 — **consolidated** batch-2 (the directly-imported JIT operator groups, in one PR per request): ⏎  ⏎ | group | operators | ⏎ |---|---| ⏎ | gemm | cutedsl_bf16_gemm, cutedsl_dsv3_fused_a_gemm, fp8_blockwise_gemm, fused_a_gemm | ⏎ | communication | all_reduce, mp | ⏎ | layernorm | fused_eh_norm, rmsnorm_hf | ⏎ | speculative | ngram_corpus, resolve_future_token_ids, ngram_embedding | ⏎ | mamba | transfer_mamba, inkling_sconv | ⏎ | quantization | awq …[truncated]

### L2-74338e94f1  (L2, 2026-07-22, sha 74338e94f10e, PR #32045)
TITLE: [Kernel] Phase 4 batch-3: migrate tangled JIT subsystems + new groups into kernels.ops (RFC #29630) (#32045)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.kernel.cutedsl_flash_fwd_mla_sm100, L2.backend.tokenspeed_mla, L2.backend.sparse_mla_adapters, L2.kernel.concat_mla, L2.kernel.mla_kv_pack_quantize_fp8, L2.runner.cuda_graph_mla
FILES: python/sglang/kernels/ops/attention/concat_mla.py (+0/-0); python/sglang/jit_kernel/benchmark/kv_canary/utils.py (+1/-1); python/sglang/jit_kernel/dsa/__init__.py (+0/-30); python/sglang/jit_kernel/dsv4/__init__.py (+0/-63); python/sglang/jit_kernel/kv_canary/plan/__init__.py (+0/-1); python/sglang/jit_kernel/minimax_m3/__init__.py (+0/-25); python/sglang/jit_kernel/tests/deepseek_v4/common.py (+4/-1); python/sglang/jit_kernel/tests/kv_canary/_canary_helpers.py (+4/-4); python/sglang/jit_kernel/tests/kv_canary/_constants.py (+1/-1); python/sglang/jit_kernel/tests/kv_canary/_differential.py (+13/-13); (+379 more)
LABELS: documentation, quant, amd, dependencies, lora, Multi-modal, deepseek, blackwell, npu, run-ci
BODY: RFC #29630 Phase 4 — **batch-3** (tangled subsystems + new groups + splits, no shims: delete + rewrite call sites). ⏎  ⏎ **Stage 1 — clean moves:** 19 attention ops (flash_attention{,_v3,_v4}, concat_mla, cutedsl_gdn/kda/paged_mqa, rope, fused_qknorm_rope/minimax_qknorm_rope, hadamard, clamp_position, add_constant, mla_kv_pack_quantize_fp8, ...), 8 moe ops, timestep_embedding; `flash_attn/` → attention; **new groups** `lplb/`, `kv_canary/`; `dsv32/`  …[truncated]

### L2-0a6d1930c3  (L2, 2026-07-22, sha 0a6d1930c360, PR #30540)
TITLE: [Attention Backend] Add HPC-Ops attention backend (#30540)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/attention_registry.py (+17/-0); docs_new/docs/advanced_features/attention_backend.mdx (+32/-1); python/sglang/srt/arg_groups/overrides.py (+10/-0); python/sglang/srt/layers/attention/hpc_ops_backend.py (+674/-0); python/sglang/srt/models/hunyuan_v3.py (+56/-0); python/sglang/srt/server_args.py (+1/-0)
LABELS: documentation, performance, run-ci, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Add a new opt-in attention backend `hpc_ops` wrapping the paged MHA kernels from [HPC-Ops](https://github.com/Tencent/hpc-ops), the production-grade operator library for LLM inference from the Tencent Hunyuan AI Infra team. ⏎  ⏎ This is the first step of the HPC-Ops <-> SGLang integration discussed with the Hunyuan team: land Attention and MoE backends first (MoE in a separate PR), then evaluate the other operators (allreduce fusion, r …[truncated]

### L2-99f636a86f  (L2, 2026-07-23, sha 99f636a86fb8, PR #32072)
TITLE: [Kernel] RFC #29630 finale: retire sglang.jit_kernel into sglang.kernels (#32072)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.kernel.set_mla_kv_buffer, L2.kernel.concat_mla
FILES: python/sglang/jit_kernel/set_mla_kv_buffer.py (+0/-5); .claude/skills/add-jit-kernel/SKILL.md (+35/-35); .claude/skills/add-sgl-kernel/SKILL.md (+1/-1); .claude/skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md (+18/-18); .claude/skills/llm-torch-profiler-analysis/scripts/triage_kernel_helpers.py (+11/-9); .claude/skills/write-sglang-test/SKILL.md (+3/-3); .github/CODEOWNERS (+4/-4); .github/MAINTAINER.md (+1/-1); .github/labeler.yml (+1/-1); .github/workflows/_pr-test-check-changes.yml (+1/-2); (+344 more)
LABELS: documentation, quant, amd, dependencies, lora, Multi-modal, deepseek, hicache, sgl-kernel, blackwell
BODY: ## Summary ⏎  ⏎ Structural **finale** of the `sglang.jit_kernel` → `sglang.kernels` migration (RFC #29630). After batch-1/2/3 (#31666, #32015, #32045) moved every operator into `sglang.kernels.ops.<group>`, this PR removes the `sglang.jit_kernel` package **entirely**. ⏎  ⏎ ## What changed ⏎  ⏎ - **Build infra moved** into `sglang/kernels/jit/`: `csrc/`, `include/`, `__main__.py`, `benchmark/`, `tests/`, `.clang-format` (git-tracked as renames). ⏎ - **`KERNEL_P …[truncated]

### L2-11b0e5c5ad  (L2, 2026-07-23, sha 11b0e5c5add9, PR #32148)
TITLE: [Kernel] Classification cleanup: unify _jit_ naming, drop empty/model groups, add elementwise (RFC #29630) (#32148)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.set_mla_kv_buffer
FILES: python/sglang/kernels/ops/kvcache/mla_buffer.py (+2/-2); python/sglang/kernels/ops/kvcache/set_mla_kv_buffer.py (+3/-7); .claude/skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md (+1/-1); .claude/skills/llm-torch-profiler-analysis/scripts/triage_kernel_helpers.py (+1/-1); benchmark/kernels/bench_fused_gate_sigmoid_mul_add.py (+1/-1); benchmark/kernels/bench_fused_sigmoid_mul.py (+1/-1); python/sglang/kernels/README.md (+3/-3); python/sglang/kernels/ops/__init__.py (+1/-2); python/sglang/kernels/ops/activation/__init__.py (+4/-4); python/sglang/kernels/ops/activation/activation.py (+4/-4); (+92 more)
LABELS: documentation, quant, Multi-modal, deepseek, sgl-kernel, run-ci, diffusion, jit-kernel, bypass-maintenance, bypass-fastfail
BODY: ## Summary ⏎  ⏎ Post-migration review follow-up on `python/sglang/kernels/` — tightens classification and naming consistency. Almost entirely `git mv` renames + import rewrites. ⏎  ⏎ ## Changes ⏎  ⏎ - **Unify JIT-op naming** — drop the leftover `_jit_` prefix (a batch-1 shim-era artifact) from 8 modules so JIT-backed ops read consistently with the rest (`flash_attention.py`, `rope.py`, …): `activation/activation`, `layernorm/norm`, `gemm/dsv3_{fused_a,router …[truncated]

### L2-ebe3ab29e4  (L2, 2026-07-23, sha ebe3ab29e485, PR #27657)
TITLE: [DeepSeek V4] CP decode opt: slice repeat attention weights to local TP partition (#27657)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/models/deepseek_v2.py (+40/-10); python/sglang/srt/models/deepseek_v4.py (+60/-25); python/sglang/srt/models/deepseek_v4_dspark.py (+2/-12); python/sglang/srt/server_args.py (+18/-0); python/sglang/srt/layers/cp/cp_decode_attn_tp.py (+213/-0); python/sglang/srt/layers/linear.py (+3/-2)
LABELS: high priority, deepseek, run-ci, run-ci-extra, release-highlight
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ In DeepSeek-V4's NSA prefill context parallel mode, the attention linears (`wo_a`, `wo_b`, `wq_b`) are initialized with `tp_size=1` and weights are repeated across all CP ranks. For CP8, this means 8× redundant attention linear computation during decode.  ⏎  ⏎ By slicing the repeated weights to only the local TP partition (matching what normal TP8 would compute), we can eliminate this redundancy and reduce decode latency. ⏎  ⏎ ## …[truncated]

### L2-3d0c6bf57f  (L2, 2026-07-23, sha 3d0c6bf57fd1, PR #32181)
TITLE: [Fix] Fix trtllm_mla backend + fp8 kv cache without rope (#32181)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/kernels/ops/attention/utils.py (+12/-0); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+35/-32); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+2/-1)
LABELS: blackwell, run-ci, jit-kernel, bypass-fastfail
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-15d73f1e03  (L2, 2026-07-23, sha 15d73f1e038e, PR #32239)
TITLE: Fix dynamo recompile limit in allreduce and bf16 gemm (#32239)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/distributed/parallel_state.py (+88/-26); python/sglang/srt/layers/communicator.py (+5/-2); python/sglang/srt/layers/quantization/unquant.py (+35/-6); python/sglang/srt/server_args.py (+13/-0)
LABELS: quant, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-35e25f5356  (L2, 2026-07-24, sha 35e25f53567c, PR #21637)
TITLE: [Feature] DCP: A2A + FlashInfer-MNNVL comm backends and q-replicate (Helix) (#21637)
SOURCES: path_core, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+61/-10); python/sglang/kernels/ops/attention/dcp_kernels.py (+196/-1); python/sglang/srt/distributed/device_communicators/pynccl.py (+54/-0); python/sglang/srt/distributed/parallel_state.py (+6/-1); python/sglang/srt/layers/activation.py (+44/-3); python/sglang/srt/layers/dcp/__init__.py (+4/-0); python/sglang/srt/layers/dcp/comm.py (+246/-2); python/sglang/srt/model_executor/model_runner.py (+44/-0); python/sglang/srt/model_executor/runner/base_runner.py (+14/-0); python/sglang/srt/models/deepseek_v2.py (+5/-0); (+3 more)
LABELS: deepseek, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight
BODY: ## Summary ⏎  ⏎ Adds pluggable communication backends and q-replicate (Helix) to the DeepSeek-MLA ⏎ **decode context-parallel (DCP)** path, on top of the merged DCP base (#14194): ⏎  ⏎ ``` ⏎ --dcp-comm-backend {ag_rs, a2a, fi_a2a}    --dcp-replicate-q-proj ⏎ ``` ⏎  ⏎ ## What's added ⏎  ⏎ - **`a2a`** (NCCL All-to-All): exchanges packed *(attention output + fp32 LSE)* in a ⏎   **single collective per layer**, then a one-shot Triton LSE combine (base-2). Adds a …[truncated]

### L2-b954e9cf3d  (L2, 2026-07-24, sha b954e9cf3dad, PR #30822)
TITLE: [6/6][kimi-deterministic] Use deterministic seeded coins for EAGLE rejection sampling (#30822)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.fa3_fa4_mla, L2.dispatch.server_args_defaults
FILES: python/sglang/kernels/ops/attention/flash_attention.py (+1/-0); python/sglang/kernels/ops/attention/flash_attention_v4.py (+5/-2); python/sglang/srt/batch_overlap/two_batch_overlap.py (+1/-0); python/sglang/srt/layers/attention/flashattention_backend.py (+4/-1); python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-2); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+12/-3); python/sglang/srt/layers/quantization/unquant.py (+0/-2); python/sglang/srt/layers/sampler.py (+15/-0); python/sglang/srt/model_executor/forward_batch_info.py (+13/-0); python/sglang/srt/models/deepseek_common/attention_backend_handler.py (+11/-4); (+9 more)
LABELS: quant, dependencies, deepseek, jit-kernel, bypass-fastfail
BODY: > **Stacked series (6/6, tip)**: stacked on `kimi-deterministic/5-logsoftmax-decode-logprobs` (#30821), so the diff below includes the whole series. This PR's own changes are the last 4 commits (ending 18f2f50bf — incl. the murmur_hash32 relocation to `sglang.kernels`). **CI on this PR covers the combined series.** ⏎  ⏎ ## Motivation ⏎  ⏎ EAGLE verify draws rejection-sampling coins from `torch.rand`, so speculative outputs are not reproducible even with  …[truncated]

### L2-2428f56145  (L2, 2026-07-24, sha 2428f5614561, PR #32262)
TITLE: [Bugfix] Fix Kimi-Linear state transfer across heterogeneous TP (#32262)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/configs/mamba_utils.py (+6/-0); python/sglang/srt/disaggregation/base/conn.py (+2/-0); python/sglang/srt/disaggregation/mooncake/conn.py (+26/-21); python/sglang/srt/disaggregation/nixl/conn.py (+36/-20); python/sglang/srt/disaggregation/utils.py (+61/-0); python/sglang/srt/mem_cache/memory_pool.py (+55/-85); python/sglang/srt/mem_cache/unified_memory_pool.py (+8/-0); python/sglang/srt/models/kimi_linear.py (+1/-1); python/sglang/test/server_fixtures/disaggregation_fixture.py (+6/-3); test/registered/disaggregation/test_disaggregation_kimi_linear.py (+105/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Kimi-Linear stores each convolution-state slot as `[K - 1, sharded_channels]`. Heterogeneous attention-TP transfer previously selected the wrong slice axis and represented the channel shard as one contiguous block, producing incorrect state descriptors for both NIXL and Mooncake. ⏎  ⏎ On the Mooncake aggregation path, source ranks whose hybrid-MLA KV is replicated also skipped their TP-sharded linear-attention state. The KV copy can be …[truncated]

### L2-1e69765bae  (L2, 2026-07-24, sha 1e69765bae5b, PR #27059)
TITLE: Add FP4 Indexer for DeepSeek V4 on SM120 (#27059)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/deepseek_v4_backend.py (+5/-0); python/sglang/srt/layers/attention/dsv4/metadata.py (+8/-3); python/sglang/srt/server_args.py (+4/-2); test/registered/kernels/benchmark/attention/bench_dsv4_fp4_indexer.py (+7/-4); test/registered/unit/layers/test_dsv4_nonpaged_indexer.py (+52/-1)
LABELS: deepseek, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ Follow-up to #26209, extends the FP4 Indexer to SM120 (depends on https://github.com/sgl-project/DeepGEMM/pull/56).  ⏎  ⏎ ## Accuracy ⏎  ⏎ DeepSeek V4 Flash GSM8K ⏎ |                                 | Accuracy | ⏎ |-------------------|----------| ⏎ | Default FP8            |    95.5% | ⏎ | FP4 indexer            |    93.0% | ⏎  ⏎ ## Performance ⏎  ⏎ Two sets of results are reported because they were collected on different SGLang software sta …[truncated]

### L2-71015f3fea  (L2, 2026-07-24, sha 71015f3fea7c, PR #31346)
TITLE: fix(dsa): fail fast on fp8_e4m3 KV with tilelang DSA backend on CUDA (#31346)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/overrides.py (+24/-0); test/registered/unit/test_dsa_tilelang_fp8_validation.py (+41/-0)
BODY: ## Motivation ⏎  ⏎ `--kv-cache-dtype fp8_e4m3` and the tilelang DSA backend (`--dsa-prefill-backend tilelang` / `--dsa-decode-backend tilelang`, or the deprecated `--nsa-*-backend` aliases) are each valid on their own, and on CUDA the server accepts them together. The combination is not implemented there. ⏎  ⏎ The tilelang DSA sparse-attention kernel only has an fp8_e4m3 KV path under ROCm/HIP, in the `if _is_hip:` branch of `tilelang_sparse_fwd`. On CUD …[truncated]

### L2-82fe0f041a  (L2, 2026-07-24, sha 82fe0f041aec, PR #32288)
TITLE: Fix stale flashinfer-MLA fallback poisoning spec verify capture (trtllm_mla + tc_piecewise) (#32288)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+10/-1); python/sglang/srt/server_args.py (+0/-13)
LABELS: high priority, blackwell, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ Follow-up to #32239. The kimi_k26 (Kimi-K2.6-NVFP4 + DFLASH, trtllm_mla, tc_piecewise) server hangs/crashes at serving: CUDA coredump shows a deterministic all-rank `Warp Illegal Address` in `flashinfer::mla::BatchMLAPagedAttentionKernel`. ⏎  ⏎ Root defect: `TRTLLMMLABackend.forward_extend` routes to the flashinfer-MLA implementation based on `forward_prefill_metadata.fallback_to_flashinfer_impl` — persistent state left behind by the l …[truncated]

### L2-91f386a5b2  (L2, 2026-07-25, sha 91f386a5b2b3, PR #32270)
TITLE: fix(disagg): support pipeline-parallel hybrid-linear transfer (#32270)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/disaggregation/base/conn.py (+2/-0); python/sglang/srt/disaggregation/decode.py (+6/-0); python/sglang/srt/disaggregation/mooncake/conn.py (+105/-17); python/sglang/srt/disaggregation/nixl/conn.py (+100/-28); python/sglang/srt/disaggregation/prefill.py (+6/-0); python/sglang/srt/disaggregation/utils.py (+54/-0); python/sglang/srt/mem_cache/memory_pool.py (+21/-0); python/sglang/srt/mem_cache/unified_memory_pool.py (+2/-0); python/sglang/srt/model_executor/pool_configurator.py (+16/-1); python/sglang/srt/models/kimi_linear.py (+20/-8); (+2 more)
LABELS: ready-to-merge, run-ci
BODY: ## Motivation ⏎  ⏎ Pipeline-parallel prefill workers own only the hybrid-linear layers assigned to their pipeline stage, while a non-pipeline decode worker registers all layers. The existing transfer path paired state and compact full-attention KV buffers by local ordinal, so later pipeline stages could write into the wrong decode buffers. NIXL completion notifications also used a pipeline-local engine rank, allowing sources from different pipeline s …[truncated]

### L2-34454c06b8  (L2, 2026-07-27, sha 34454c06b891, PR #32496)
TITLE: [Refactor] Tidy server_args.py section grouping and drop unused alias (#32496)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+24/-28)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Small readability cleanup of `server_args.py`: ⏎  ⏎ - Move `CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS` next to `ATTENTION_BACKEND_CHOICES` so the related attention-backend lists live together, and simplify `add_chunked_prefix_cache_attention_backend`. ⏎ - Drop the `SPECULATIVE_DRAFT_MODEL_QUANTIZATION_CHOICES` alias and reference `QUANTIZATION_CHOICES` directly at its only use site. ⏎ - Add section header comments for the ngram spe …[truncated]

### L2-2abb1d2c37  (L2, 2026-07-27, sha 2abb1d2c3744, PR #31992)
TITLE: fix(hisparse): correct DSA KV memory budget (#31992)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/pool_configurator.py (+21/-3); test/registered/unit/model_executor/test_hisparse_pool_configurator.py (+76/-0)
LABELS: run-ci, run-ci-extra
BODY: ## Summary ⏎  ⏎ - reuse `calculate_mla_kv_cache_dim()` when profiling MLA KV memory so the budget matches the layout allocated by `DSATokenToKVPool`, including scaled FP8 KV ⏎ - scale only the HiSparse DSA indexer term by `host_to_device_ratio` ⏎ - add CPU regression coverage for BF16 and FP8 HiSparse layouts ⏎  ⏎ ## Root cause ⏎  ⏎ `HiSparseDSATokenToKVPool` allocates the indexer buffer with `size * host_to_device_ratio`, while the profiler budgeted it at `size …[truncated]

### L2-4e5a05148a  (L2, 2026-07-28, sha 4e5a05148a2b, PR #30825)
TITLE: [FullCG] Support chunked cached-prefix prefill (#30825)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla
FILES: python/sglang/srt/layers/attention/base_attn_backend.py (+21/-0); python/sglang/srt/layers/attention/flashattention_backend.py (+4/-0); python/sglang/srt/layers/attention/hybrid_attn_backend.py (+14/-0); python/sglang/srt/model_executor/cuda_graph_config.py (+17/-3); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+316/-24); python/sglang/srt/model_executor/runner/shape_key.py (+3/-2); test/registered/cuda_graph/full_prefill/test_full_cuda_graph_prefill.py (+103/-8); test/registered/unit/configs/test_multimodal_piecewise_cuda_graph.py (+1/-0); test/registered/unit/model_executor/runner/test_prefill_cuda_graph_padding.py (+1/-0); test/registered/unit/model_executor/test_prefill_cuda_graph_runner.py (+266/-0)
LABELS: Multi-modal, run-ci, bypass-fastfail
BODY: Depends on #31050. ⏎  ⏎ ## Why is this change needed? ⏎  ⏎ Full prefill CUDA graphs capture a suffix-only topology. MLA cached-prefix requests add a prefix-attention loop and LSE merge, so replaying the ordinary graph can silently omit cached-prefix attention. ⏎  ⏎ Capturing the entire supported prefix as one aggregate workspace is correct but expensive: MLA-to-MHA expansion must materialize every captured prefix token simultaneously. This change bounds that …[truncated]

### L2-d9cf7b0a8b  (L2, 2026-07-28, sha d9cf7b0a8b25, PR #32219)
TITLE: [MTP] Cut spec-v2 host-seam overhead in hybrid-linear MTP decode (#32219)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/mamba/mamba_state_indices_triton.py (+75/-0); python/sglang/kernels/ops/speculative/cache_locs.py (+68/-0); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+75/-8); python/sglang/srt/layers/attention/linear/kda_backend.py (+7/-0); python/sglang/srt/speculative/eagle_utils.py (+5/-3); test/registered/attention/unittests/hybrid_linear/test_verify_buffer_fixup_hook.py (+131/-0); test/registered/kernels/ops/mamba/test_fused_replay_state_indices.py (+192/-0)
LABELS: speculative-decoding, run-ci, jit-kernel, run-ci-extra, linear-attention, release-highlight, user-tps
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ With MTP (NEXTN) speculative decoding under spec-v2 overlap scheduling, each decode step consists of three CUDA graphs: ⏎  ⏎ ``` ⏎ [draft graph]   NEXTN draft model, k autoregressive passes -> draft-token chain ⏎ [verify graph]  target model, one forward over the draft tokens (TARGET_VERIFY) ⏎ [extend graph]  draft model backfills KV for the accepted tokens (DRAFT_EXTEND) ⏎ ``` ⏎  ⏎ Between the graphs live eager "seams": accept sampling, …[truncated]

### L2-7f438a6031  (L2, 2026-07-28, sha 7f438a6031ec, PR #26928)
TITLE: feat: SM120 (Blackwell Desktop) support for GLM-5.1 inference (#26928)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.kernel.flash_mla_sm120, L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/kernels/ops/attention/flash_mla_sm120.py (+82/-0); python/sglang/srt/arg_groups/overrides.py (+19/-0); python/sglang/srt/layers/attention/dsa_backend.py (+82/-2); python/sglang/srt/server_args.py (+1/-0); scripts/release/bump_flashinfer_version.py (+6/-3); test/manual/test_dsa_alias_cli_registry_env.py (+1/-0); test/registered/unit/model_executor/test_pool_configurator.py (+43/-1); test/registered/unit/test_flashinfer_sparse_mla.py (+123/-0); test/registered/unit/test_model_overrides.py (+14/-0)
LABELS: dependencies, run-ci, jit-kernel
BODY: Related issue: https://github.com/sgl-project/sglang/issues/26087 ⏎  ⏎ Reference vLLM implementation: https://github.com/vllm-project/vllm/pull/43477 ⏎  ⏎ Old version prebuilt image: [`void0110/sglang-sm120-glm51:main-deepgemm324-flashinfer3395-cu130-20260602`](https://hub.docker.com/r/void0110/sglang-sm120-glm51/tags?name=main-deepgemm324-flashinfer3395-cu130-20260602) ⏎  ⏎ ## Motivation ⏎  ⏎ GLM-5.1 NVFP4 uses DeepSeek Sparse Attention (DSA) with an FP …[truncated]

### L2-86ee545388  (L2, 2026-07-28, sha 86ee54538867, PR #32592)
TITLE: docs(cookbook): update Kimi-K3 GB200 recipes from measured 4x4 runs (#32592)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx (+110/-4)
LABELS: documentation
BODY: Follow-up to #32542. Updates the GB200 recipes for Kimi-K3. ⏎  ⏎ Work done in collaboration with @leejnau. ⏎  ⏎ The GB200 cells were derived from the GB300 recipe. This replaces the PD-disaggregated ones with measured configurations, and adds two pieces GB200 was missing: a prefill cell and a `multiNodeHints` entry. ⏎  ⏎ ## Changes ⏎  ⏎ | Cell | Before | After | ⏎ |---|---|---| ⏎ | prefill / default | *(none)* | `--tp-size 1 --pp-size 16` | ⏎ | prefill / long-context  …[truncated]

### L2-ef6c07008b  (L2, 2026-07-28, sha ef6c07008b5e, PR #32612)
TITLE: Support DCP for Kimi Linear model (#32612)
SOURCES: path_core, symbol_pickaxe, corpus:production-kernel-provenance
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.trtllm_mla, L2.backend.cutedsl_mla, L2.backend.tokenspeed_mla, L2.pool.mla_token_kv, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/attention_registry.py (+2/-2); python/sglang/srt/layers/attention/cutedsl_mla_backend.py (+451/-0); python/sglang/srt/layers/attention/tokenspeed_mla_backend.py (+291/-5); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+55/-3); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+28/-13); python/sglang/kernels/ops/attention/dcp_kernels.py (+34/-0); python/sglang/srt/layers/attention/flashinfer_backend.py (+3/-15); python/sglang/srt/mem_cache/memory_pool.py (+6/-0); python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py (+7/-11); python/sglang/srt/model_executor/runner/eager_runner.py (+11/-7); (+7 more)
LABELS: blackwell, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
BODY: ## Summary ⏎  ⏎ - Support DCP, as well as its compatibility with PD Disagg/DSpark/Hicache for Kimi series models ⏎  ⏎ ## Co-authors ⏎  ⏎ The implementation commit includes the requested co-author trailers for the two additional authors from PR #205: ⏎  ⏎ - Julien Lin <jullin@nvidia.com> ⏎ - kpham-sgl <khoa.pham@radixark.ai> ⏎  ⏎ ## Testing ⏎  ⏎ - full `pre-commit run --all-files` ⏎ - relevant suite: **153 passed, 29 skipped, 12 subtests passed** ⏎ - per review feedback, the t …[truncated]

### L2-d12ea3e9ba  (L2, 2026-07-29, sha d12ea3e9ba9f, PR #32760)
TITLE: docker: add Kimi K3 images (#32760)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/kimi_k3/apply_deepep_k3_patch.sh (+169/-0); docker/kimi_k3/apply_deepgemm_situ_patch.py (+140/-0); docker/kimi_k3/flashinfer-perkz-dcp-0.6.15.txt (+5639/-0); docker/kimi_k3/kimi_k3_cu12.Dockerfile (+136/-0); docker/kimi_k3/kimi_k3_cu13.Dockerfile (+125/-0); docker/rocm.Dockerfile (+4/-4)
LABELS: amd
BODY: ## Summary ⏎  ⏎ Extract the Docker-only changes from #32541 into a dedicated pull request. ⏎  ⏎ - add CUDA 12 and CUDA 13 Kimi K3 Dockerfiles ⏎ - add the associated DeepEP, DeepGEMM, and FlashInfer patch inputs ⏎ - include the Kimi K3 ROCm Dockerfile update ⏎  ⏎ ## Why ⏎  ⏎ Keeping image build changes separate lets the Kimi K3 runtime support and image publication review independently. ⏎  ⏎ ## Validation ⏎  ⏎ - `git diff --check` ⏎ - repository pre-commit hooks ⏎  ⏎ --- ⏎ ### CI St …[truncated]

### L2-e5c46ff07d  (L2, 2026-07-29, sha e5c46ff07d78, PR #32818)
TITLE: [Fix] Route asymmetric-KV models to fa4 on SM100 and pin MiMoV2 FP8 MoE to flashinfer_trtllm (#32818)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx (+2/-0); docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx (+7/-3); python/sglang/srt/arg_groups/overrides.py (+13/-2); python/sglang/srt/configs/model_config.py (+11/-0); python/sglang/srt/server_args.py (+4/-0); test/registered/unit/test_model_overrides.py (+51/-16)
LABELS: documentation
BODY: MiMo-V2.5 cannot be deployed on SM100/SM103 with default flags (#31243). The SM100 auto-dispatch picks `trtllm_mha`, whose paged-KV kernel requires equal K/V row widths, but MiMoV2 has asymmetric KV (`head_dim` 192, `v_head_dim` 128), so decode CUDA-graph capture aborts with `Check failed: key_cache.size(i) == value_cache.size(i) (37074 vs. 24716)`. The reporter had to fall back to triton with CUDA graphs disabled. ⏎  ⏎ Three changes: ⏎  ⏎ 1. `ModelConfi …[truncated]

### L2-3c9efaf3e1  (L2, 2026-07-29, sha 3c9efaf3e192, PR #32834)
TITLE: [docs] Kimi-K3: widen the H200 High-Throughput recipe to 4x8 TP32/EP32 (#32834)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx (+3/-3); docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx (+21/-9)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ The Kimi-K3 cookbook's **H200 / Unified / High-Throughput** cell was a copy of Balanced (2×8 TP16/EP16) with `--mem-fraction-static 0.90` and `extra_buffer_lazy`. The shape actually run for that operating point is **TP32/EP32 across 4 nodes**, so the page shipped a recipe nobody serves. ⏎  ⏎ ## Modifications ⏎  ⏎ `docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx` — the H200 Unified High-Throughput cell: ⏎  ⏎ - `nnodes: 2 → 4`, `--tp-size`/ …[truncated]

### L2-c32c4ef79c  (L2, 2026-07-29, sha c32c4ef79cf5, PR #32648)
TITLE: [Kernel] Move sgl-kernel under sglang.kernels.aot (#32648)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L2.build.flashmla_sgl_kernel, L2.kernel.cutlass_mla, L2.kernel.concat_mla
FILES: docker/arm64.Dockerfile (+1/-1); docker/rocm.Dockerfile (+2/-2); docker/xeon.Dockerfile (+1/-1); python/pyproject.toml (+8/-0); python/sglang/kernels/aot/CMakeLists.txt (+0/-0); .claude/skills/add-jit-kernel/SKILL.md (+1/-1); .claude/skills/add-sgl-kernel/SKILL.md (+38/-38); .claude/skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md (+1/-1); .claude/skills/llm-torch-profiler-analysis/scripts/triage_kernel_helpers.py (+2/-2); .github/CODEOWNERS (+2/-2); (+360 more)
LABELS: documentation, quant, amd, dependencies, Multi-modal, deepseek, speculative-decoding, sgl-kernel, npu, run-ci
BODY: ## Summary ⏎  ⏎ - Move the standalone `sgl-kernel` source tree to `python/sglang/kernels/aot` so SGLang's AOT and JIT kernel implementations live under one kernel namespace. ⏎ - Preserve the published distribution name (`sglang-kernel`), Python import (`sgl_kernel`), operator registrations, versioning, API, and ABI. ⏎ - Update CUDA/ROCm/CPU/ARM build, test, release, Docker, CI, CODEOWNERS, labeler, and developer paths to the new source location. ⏎ - Keep t …[truncated]

### L2-983e4aa18d  (L2, 2026-07-29, sha 983e4aa18d7e, PR #32620)
TITLE: Eliminate redundant DSA state transfers (Mooncake) (#32620)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/mooncake/conn.py (+15/-7)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ With Prefill CP + Decode TP and `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1`, DSA MLA models send the same bytes many times over, wasting bandwidth and inflating TTFT. ⏎  ⏎ Take Prefill CP8 + Decode TP8 as an example. ⏎  ⏎ - **`ALL_CP_RANKS_TRANSFER=0`** — only CP rank 0 transfers, once per decode rank: 8 full-size transfers. ⏎ - **`ALL_CP_RANKS_TRANSFER=1`** — every CP rank transfers. Each rank sends 1/8 of the KV cache to all dec …[truncated]

### L2-f46d5f25b4  (L2, 2026-07-30, sha f46d5f25b4c3, PR #30482)
TITLE: [4/N][CP] Support interleave strategy for cp v2 (#30482)
SOURCES: path_core
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+15/-6); python/sglang/srt/layers/attention/dsa/dsa_indexer.py (+23/-8); python/sglang/srt/layers/attention/dsa/utils.py (+21/-12); python/sglang/srt/layers/attention/dsa_backend.py (+45/-9); python/sglang/srt/layers/cp/base.py (+38/-6); python/sglang/srt/layers/cp/interleave.py (+186/-21); python/sglang/srt/layers/cp/utils.py (+59/-7); python/sglang/srt/layers/cp/zigzag.py (+7/-2); python/sglang/srt/model_executor/runner/eager_runner.py (+17/-14); python/sglang/srt/models/deepseek_nextn.py (+31/-23); (+5 more)
LABELS: deepseek, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-e4a40a71f8  (L2, 2026-07-30, sha e4a40a71f84c, PR #31888)
TITLE: [DSA] Q8KV8 FP8 Sparse Prefill on GLM-5.2 & DeepSeek-V3.2: Q8-Path & Shared-Path Optimizations (#31888)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters
FILES: python/sglang/kernels/jit/csrc/sparse_mla_q8kv8_prefill_sm90/entry.cuh (+41/-0); python/sglang/kernels/ops/attention/sparse_mla_q8kv8_prefill_sm90.py (+74/-4); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+174/-0); benchmark/kernels/deepseek/benchmark_q8kv8_kv_gather.py (+262/-0); benchmark/kernels/deepseek/benchmark_q8kv8_q_prep.py (+486/-0); docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx (+1/-1); python/sglang/kernels/jit/csrc/qprep_bf16_fp8_sm90/entry.cuh (+84/-0); python/sglang/kernels/jit/csrc/qprep_bf16_fp8_sm90/kernel.cuh (+516/-0); python/sglang/kernels/jit/csrc/qprep_bf16_fp8_sm90/params.h (+52/-0); python/sglang/kernels/ops/attention/dsa/dequant_k_cache.py (+237/-2); (+20 more)
LABELS: documentation, quant, deepseek, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: > Part of the Q8KV8 FP8 sparse-attention roadmap #25746; follow-up to #25751 (kernel) and ⏎ > #30514 (DSA integration), both now in main. Throughout, ⏎ > **q8** = the Q8KV8 fp8 sparse-prefill backend and **q16** = the bf16-sparse FlashMLA ⏎ > baseline; both arms always run the fp8 KV cache. Scope: **GLM-5.2 + DeepSeek-V3.2** (shared ⏎ > levers measured on both; the q8-only levers apply to any DSA architecture using this backend). ⏎  ⏎ ## TL;DR ⏎  ⏎ - **G …[truncated]

### L2-425349b799  (L2, 2026-07-30, sha 425349b799f1, PR #31128)
TITLE: [Perf][DSA] Pass topk_length to flash_mla_sparse_fwd in the sparse attention path (#31128)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/dsa_backend.py (+16/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ `sgl_kernel.flash_mla_sparse_fwd` accepts an optional per-row `topk_length` tensor, and the vendored FlashMLA sparse prefill kernels (both sm90 and sm100) early-exit their top-k loop after `ceil_div(topk_length[row], B_TOPK)` blocks. The DSA backend (`dsa_backend.py::_forward_flashmla_sparse`, the DeepSeek-V3.2 / GLM-5 path) never passed it, so every row whose context is shorter than `index_topk` scanned the full `-1`-padded top-k  …[truncated]

### L2-e23ccb15f0  (L2, 2026-07-30, sha e23ccb15f075, PR #32971)
TITLE: [unified-memory] Support MLA-hybrid-Mamba (Kimi-Linear) on the Triton backend (#32971)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/kernels/ops/attention/fla/chunk_delta_h.py (+11/-3); python/sglang/srt/layers/attention/triton_backend.py (+18/-4); python/sglang/srt/mem_cache/memory_pool.py (+13/-1); python/sglang/srt/mem_cache/kv_cache_configurator.py (+6/-3); python/sglang/srt/mem_cache/layout/page_major.py (+68/-0); python/sglang/srt/mem_cache/multi_ended_allocator.py (+76/-1); python/sglang/srt/mem_cache/unified_memory_pool.py (+267/-36); test/registered/unit/mem_cache/test_full_loc_fast_path.py (+110/-0); test/registered/unit/mem_cache/test_unified_mla_gpu_parity.py (+203/-0); test/registered/unit/mem_cache/test_unified_mla_views.py (+391/-0)
LABELS: jit-kernel
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## What this does ⏎  ⏎ Extends `--enable-unified-memory` to **MLA-hybrid-Mamba** models (Kimi-Linear), on the Triton attention backend. ⏎  ⏎ Unified memory replaces the two static pools — a full-KV pool and a Mamba/linear-attention state pool, split once at startup by `--mamba-full-memory-ratio` and never rebalanced — with a single shared buffer that two `MultiEndedAllocator`s grow into from opposite ends. The pain it removes is one side idling while the …[truncated]

### L2-5c6635d8f3  (L2, 2026-07-30, sha 5c6635d8f3f4, PR #32920)
TITLE: [Spec] Compact the target-verify mask when nothing reads it (#32920)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.backend.flashmla, L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashmla_backend.py (+17/-12); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+17/-11); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+0/-8); python/sglang/srt/layers/attention/base_attn_backend.py (+5/-15); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+15/-18); python/sglang/srt/layers/attention/flashattention_backend.py (+17/-26); python/sglang/srt/layers/attention/hybrid_attn_backend.py (+17/-4); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+10/-11); python/sglang/srt/layers/attention/tbo_backend.py (+14/-5); python/sglang/srt/layers/attention/triton_backend.py (+23/-18); (+7 more)
LABELS: deepseek, blackwell, npu, run-ci
ISSUES: #32050 [DSV4][Spec] FULL verify tree-mask fill touches ~1 GiB per draft step
BODY: ## Summary ⏎  ⏎ - Give the target-verify mask a single owner (`VerifyMask`), replacing five hand-rolled allocations that disagreed on size, dtype and gate ⏎ - Let a mask nobody reads use the compact `QLEN_ONLY` layout: at `max_bs=256, num_draft_tokens=4, max_context_len=1M` the DeepSeek-V4 buffer goes from **1,073,745,920 cells (~1 GiB per rank) to 4,096** (fixes #32050, supersedes #32060) ⏎ - **Four backends qualify**, not just DeepSeek-V4: FlashMLA and …[truncated]

### L2-33c27d8e7f  (L2, 2026-07-31, sha 33c27d8e7f4f, PR #32972)
TITLE: [unified-memory] Let Kimi-Linear use the paged MLA attention backends (#32972)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.trtllm_mla, L2.pool.mla_token_kv, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+76/-0); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+64/-3); python/sglang/srt/mem_cache/memory_pool.py (+8/-3); python/sglang/srt/server_args.py (+24/-4); python/sglang/kernels/ops/kvcache/kv_indices.py (+15/-1); python/sglang/srt/mem_cache/multi_ended_allocator.py (+9/-0); test/registered/models_e2e/test_kimi_linear_unified_memory.py (+77/-0); test/registered/unit/mem_cache/test_unified_mla_dense_block_table.py (+375/-0); test/registered/unit/server_args/test_page_major_backend_allowlist.py (+111/-0)
LABELS: blackwell, jit-kernel, bypass-fastfail
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: > Stacked on #32971 (base is `feature/unified-memory-kimi-linear-triton`). Review that one first; this PR's diff is just the two commits on top. ⏎  ⏎ Some refactor work is needed after Kimi K3 is merged to main. ⏎  ⏎ ## What this does ⏎  ⏎ With the dense per-layer MLA views from #32971 in place, the **stock paged MLA kernels can read the unified pool directly** — only their `kv_indices` / block tables need remapping to dense ids. No change to the physical la …[truncated]

### L2-2573190b93  (L2, 2026-07-31, sha 2573190b9377, PR #32837)
TITLE: feat: support Kimi Linear PD disaggregation with DCP (#32837)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/pd_disaggregation_hook.py (+25/-0); python/sglang/srt/disaggregation/base/conn.py (+1/-0); python/sglang/srt/disaggregation/common/conn.py (+57/-5); python/sglang/srt/disaggregation/common/utils.py (+69/-0); python/sglang/srt/disaggregation/fake/conn.py (+1/-0); python/sglang/srt/disaggregation/mooncake/conn.py (+179/-16); python/sglang/srt/disaggregation/mori/conn.py (+1/-0); python/sglang/srt/disaggregation/nixl/conn.py (+158/-24); python/sglang/srt/disaggregation/prefill.py (+12/-3); python/sglang/srt/disaggregation/utils.py (+25/-0); (+3 more)
LABELS: run-ci, bypass-fastfail, run-ci-extra
BODY: ## Summary ⏎  ⏎ Add PD disaggregation support for transferring dense prefill MLA rows into rank-owned decode DCP layouts. ⏎  ⏎ - support Mooncake and NIXL transfer backends ⏎ - preserve exact final-page token counts across physical/virtual page and chunk boundaries ⏎ - support heterogeneous prefill/decode attention TP and matching DCP layouts ⏎ - compose pipeline-parallel layer mapping with DCP token relayout ⏎ - complete zero-owned NIXL chunks with standalone n …[truncated]

### L2-d3222bcc3a  (L2, 2026-07-31, sha d3222bcc3a79, PR #33046)
TITLE: [unified-memory] Support fa3, the default MLA backend on pre-Blackwell hosts (#33046)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/kernels/ops/attention/metadata.py (+31/-0); python/sglang/srt/layers/attention/flashattention_backend.py (+38/-0); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+1/-45); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+1/-1); python/sglang/srt/layers/attention/unified_mem_hooks.py (+70/-0); python/sglang/srt/server_args.py (+3/-1); test/registered/models_e2e/test_kimi_linear_unified_memory.py (+21/-39); test/registered/unit/mem_cache/test_unified_mla_dense_block_table.py (+110/-5); test/registered/unit/server_args/test_page_major_backend_allowlist.py (+15/-6)
LABELS: blackwell, run-ci, jit-kernel, bypass-fastfail
BODY: Follow-up to #32971 / #32972. Those two made `--enable-unified-memory` work for Kimi-Linear, but the page-major full-attention allowlist still omitted `fa3` — which is exactly what an unspecified `--attention-backend` resolves to for an MLA model on Hopper. So on H100/H200 the feature **failed at startup under its own default configuration**, and you had to know to pass `--attention-backend triton`. ⏎  ⏎ This is the pre-Blackwell counterpart of the s …[truncated]

### L2-77c77a3da8  (L2, 2026-07-31, sha 77c77a3da879, PR #33023)
TITLE: feat(inkling): migrate short convs onto the ShortConv attention backend (#33023)
SOURCES: path_core
ARTIFACT_HINTS: L2.dispatch.attention_registry
FILES: python/sglang/srt/layers/attention/attention_registry.py (+26/-1); python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py (+2/-0); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+8/-1); python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py (+572/-0); python/sglang/srt/layers/attention/linear/short_conv_backend.py (+23/-11); python/sglang/srt/models/inkling.py (+0/-32); python/sglang/srt/models/inkling_common/kernels/comm.py (+1/-1); python/sglang/srt/models/inkling_common/kernels/sconv.py (+69/-20); python/sglang/srt/models/inkling_common/sconv.py (+61/-342); python/sglang/srt/speculative/dflash_worker_v2.py (+3/-17); (+6 more)
LABELS: npu, bypass-fastfail
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ Applies the `ShortConvAttnBackend` sidecar from #29867 to **Inkling**, and fixes the 4x redundant per-step metadata prep that motivated it. ⏎  ⏎ `ShortConvolution` no longer touches the pool. It reads a per-(layer, step) handle from `get_attn_backend().conv_state_metadata()`, and the new `InklingShortConvAttnBackend` owns the step-global conv metadata: ⏎  ⏎ - the per-request slot gather + virtual->physical translate ⏎ - the fused `query_start_l …[truncated]

### L2-55b6769b0e  (L2, 2026-07-31, sha 55b6769b0ede, PR #33013)
TITLE: config: read resolved config via namespace accessors (#33013)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.backend.flashmla, L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.backend.aiter_mla, L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+6/-6); python/sglang/srt/layers/attention/flashmla_backend.py (+2/-2); python/sglang/srt/arg_groups/overrides.py (+6/-11); python/sglang/srt/batch_overlap/two_batch_overlap.py (+4/-4); python/sglang/srt/configs/inkling.py (+2/-2); python/sglang/srt/debug_utils/dumper.py (+2/-1); python/sglang/srt/disaggregation/base/conn.py (+1/-0); python/sglang/srt/disaggregation/common/conn.py (+6/-5); python/sglang/srt/disaggregation/decode.py (+7/-6); python/sglang/srt/disaggregation/encode_grpc_server.py (+4/-3); (+177 more)
LABELS: Multi-modal, deepseek, speculative-decoding, ready-to-merge, blackwell, npu, mthreads, apple-silicon
BODY: Part 3 of the 3-PR stack (base: #33012). RFC: #30696. Re-lands the reader migration reverted in #32100, regenerated from scratch against current main with the revert's defects fixed at their origin. ⏎  ⏎ - Mechanical sweep (AST-based, alias-aware): `get_server_args().FIELD` / `self.server_args.FIELD` / local-alias reads flip to the namespace accessors (`get_exec()` / `get_memory()` / …), routed by each field's NS metadata — 628 reads across 160 files …[truncated]

### L2-3e0f7c3f30  (L2, 2026-07-31, sha 3e0f7c3f30e2, PR #31987)
TITLE: [BCG][3/N] Enable bcg on dsa & deepep a2a backend (#31987)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: python/sglang/benchmark/one_batch.py (+1/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+66/-1); python/sglang/srt/managers/scheduler.py (+9/-0); python/sglang/srt/managers/scheduler_components/dp_attn.py (+42/-19); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+98/-42); python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py (+1/-0); python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py (+19/-5); python/sglang/srt/models/deepseek_v2.py (+8/-0); python/sglang/srt/server_args.py (+53/-28); python/sglang/srt/speculative/eagle_worker_common.py (+5/-5); (+3 more)
LABELS: deepseek, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-1496bfee93  (L2, 2026-07-31, sha 1496bfee93bc, PR #32828)
TITLE: [Kimi] Support DCP + DSpark (ported from kimi-k3 branch) (#32828)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L2.backend.tokenspeed_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/tokenspeed_mla_backend.py (+15/-3); python/sglang/srt/mem_cache/common.py (+6/-13); python/sglang/srt/mem_cache/kv_cache_configurator.py (+10/-0); python/sglang/srt/model_executor/pool_configurator.py (+1/-1); python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py (+64/-0); test/registered/dcp/test_kimi_linear_dcp_dspark4.py (+222/-0); test/registered/dcp/test_tokenspeed_mla_dcp_metadata.py (+87/-0); test/registered/unit/mem_cache/test_paged_free_segment.py (+40/-0)
LABELS: high priority, run-ci, bypass-fastfail, run-ci-extra
BODY: Port the Kimi Linear DCP + DSPARK feature from the `kimi-k3` branch, and add its test coverage. ⏎  ⏎ `main` had the DCP half (#32612) but not the DSPARK half for a hybrid linear-attention target: the target's KDA/mamba recurrent state was never committed to the accepted length after verify, so it drifted from the accepted token sequence every step. ⏎  ⏎ ## Changes ⏎  ⏎ - `speculative/dspark_components/dspark_worker_v2.py` — commit the last accepted ver …[truncated]

### L2-2fd78ec2d7  (L2, 2026-08-01, sha 2fd78ec2d751, PR #31221)
TITLE: [AMD] Derive AITER verify tokens-per-req from input shape (#31221)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+22/-6)
LABELS: documentation, quant, amd, dependencies, lora, Multi-modal, deepseek, speculative-decoding, hicache, sgl-kernel
DEEP_STUDY: deep-study: this PR was reverted by PR 38575 (reland, reason=correctness_or_accuracy)
BODY: In the AITER attention backend's target_verify and CUDA graph paths, the per-request verify token count was taken from the fixed init-time num_draft_tokens (self.num_draft_tokens / spec_info.draft_token_num). This is incorrect when the actual number of verify tokens per request varies at runtime, for example, SBD spec decoding. ⏎  ⏎ Derive it from the exact-sized capture/replay input instead (input_ids.shape[0] // batch_size) and thread it through  …[truncated]

### L2-df55e911d6  (L2, 2026-08-01, sha df55e911d621, PR #33168)
TITLE: Fix the chunked-prefix-cache gate writing config the backends never read (#33168)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/model_runner.py (+5/-6); python/sglang/srt/model_executor/model_runner_components/misc_utils.py (+6/-5); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+2/-3); python/sglang/srt/speculative/draft_worker_common.py (+23/-13); test/registered/unit/model_executor/test_chunked_prefix_cache_gate.py (+72/-0); test/registered/unit/test_server_args_writer_ratchet.py (+1/-1)
LABELS: ready-to-merge
BODY: ## Motivation ⏎  ⏎ The load-time chunked-prefix gate (`maybe_disable_chunked_prefix_cache`) still writes its `ServerArgs` instance, but every reader has moved to the published config: the attention backends assert / branch on `get_schedule().disable_chunked_prefix_cache` when they initialize. The flip never reaches them, so an MLA model on a backend outside `CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS` keeps chunked prefix enabled instead of be …[truncated]

### L2-47d8b5b749  (L2, 2026-08-01, sha 47d8b5b749fd, PR #33170)
TITLE: config: route parallel config-leaf reads through get_parallel() (#33170)
SOURCES: path_core
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+2/-2); python/sglang/srt/batch_overlap/two_batch_overlap.py (+1/-1); python/sglang/srt/disaggregation/common/conn.py (+10/-13); python/sglang/srt/disaggregation/mooncake/conn.py (+2/-2); python/sglang/srt/distributed/device_communicators/triton_symm_mem_ag.py (+4/-4); python/sglang/srt/elastic_ep/elastic_ep.py (+2/-4); python/sglang/srt/eplb/expert_location.py (+30/-21); python/sglang/srt/layers/attention/dsa/utils.py (+2/-2); python/sglang/srt/layers/communicator.py (+5/-11); python/sglang/srt/layers/cp/cp_decode_attn_tp.py (+2/-2); (+71 more)
LABELS: amd, deepseek, ready-to-merge
BODY: Part 1/4 of the config-namespace follow-up stack (RFC: #30696; follows the merged #33011–#33013). Based on #33168 (the chunked-prefix gate fix) so the stack tests with it; merge #33168 first, then the members in order. ⏎  ⏎ ## Motivation ⏎  ⏎ The parallel namespace was the last reader family left on `get_server_args()`: 106 config-leaf reads (`enable_dp_lm_head`, `enable_dp_attention`, `pp_async_batch_depth`, `dp_size`, `ep_join_rank_offset`, `dwdp_size` …[truncated]

### L2-bae8eb8d6c  (L2, 2026-08-01, sha bae8eb8d6caa, PR #30971)
TITLE: [minimax-m3] fp8 attention GEMMs on SM100 (fp8_e4m3 KV + trtllm_mha) (#30971)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.pool.mla_token_kv, L2.dispatch.server_args_defaults
FILES: python/sglang/kernels/jit/csrc/minimax/minimax_decode_topk.cuh (+54/-7); python/sglang/kernels/ops/attention/minimax_decode_topk.py (+11/-4); python/sglang/kernels/ops/attention/minimax_sparse/common/utils.py (+73/-17); python/sglang/kernels/ops/attention/minimax_sparse/decode/flash_with_topk_idx.py (+56/-12); python/sglang/kernels/ops/attention/minimax_sparse/decode/topk_sparse.py (+32/-12); python/sglang/kernels/ops/attention/minimax_sparse/prefill/flash_with_topk_idx.py (+49/-9); python/sglang/kernels/ops/attention/minimax_sparse/prefill/topk_sparse.py (+33/-8); python/sglang/srt/arg_groups/overrides.py (+59/-8); python/sglang/srt/environ.py (+4/-0); python/sglang/srt/layers/attention/minimax_sparse_backend.py (+73/-9); (+13 more)
LABELS: performance, blackwell, run-ci, jit-kernel, run-ci-extra, kernel
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ MiniMax-M3 currently runs all attention GEMMs in bf16: with `--kv-cache-dtype fp8_e4m3` the fp8 cache is widened back to bf16 on load (widening-dequant contract), and the lightning-indexer cache stays bf16 entirely. On Blackwell, both the MSA fmha_sm100 kernels and trtllm-gen's dense kernels have native fp8 paths, so we can run the sparse / MSA / indexer / dense attention GEMMs in fp8_e4m3 end-to-end: fp8 tensor-core MMAs plus an …[truncated]

### L2-fb207b72b0  (L2, 2026-08-01, sha fb207b72b02a, PR #32890)
TITLE: feat(kernels): port standalone Kimi K3 kernels (#32890)
SOURCES: path_core, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.kernel.set_mla_kv_buffer, L2.kernel.concat_mla, L2.kernel.set_mla_kv_concat_q
FILES: python/sglang/kernels/jit/csrc/elementwise/concat_mla.cuh (+20/-13); python/sglang/kernels/jit/csrc/elementwise/set_mla_kv_buffer.cuh (+3/-36); python/sglang/kernels/jit/csrc/elementwise/set_mla_kv_concat_q.cuh (+607/-0); python/sglang/kernels/jit/csrc/kimi_k3/mla_output_gate.cuh (+84/-0); python/sglang/kernels/ops/attention/concat_mla.py (+9/-2); python/sglang/kernels/jit/csrc/attention/fixup_zero_kv.cuh (+39/-24); python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh (+1064/-0); python/sglang/kernels/jit/csrc/attention/kda_packed_decode.cuh (+240/-0); python/sglang/kernels/jit/csrc/attention/kda_prefill.cu (+3871/-0); python/sglang/kernels/jit/csrc/elementwise/add3.cuh (+93/-0); (+74 more)
LABELS: documentation, high priority, quant, Multi-modal, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Port the standalone kernels from the Kimi K3 Day0 work to `main` first. Keeping ⏎ the kernels and their direct tests separate makes the reusable pieces reviewable ⏎ before the more invasive model, scheduler, and serving integration changes. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add the shared JIT/TMA support and generic prerequisite kernels used by Kimi ⏎   K3. ⏎ - Port the standalone KDA kernels, including ReplaySSM, packed decode, and the ⏎   NVIDIA/PTX pr …[truncated]

### L2-e2cf21b9e5  (L2, 2026-08-01, sha e2cf21b9e561, PR #33025)
TITLE: [Kimi K3] Add reasoning, tool-call, and OpenAI serving support (#33025)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+1/-1); python/sglang/srt/constrained/base_grammar_backend.py (+2/-2); python/sglang/srt/constrained/grammar_manager.py (+1/-1); python/sglang/srt/constrained/reasoner_grammar_backend.py (+63/-55); python/sglang/srt/entrypoints/openai/chat_encoding.py (+5/-1); python/sglang/srt/entrypoints/openai/protocol.py (+34/-25); python/sglang/srt/entrypoints/openai/serving_chat.py (+272/-42); python/sglang/srt/entrypoints/openai/serving_responses.py (+26/-10); python/sglang/srt/function_call/base_format_detector.py (+15/-1); python/sglang/srt/function_call/function_call_parser.py (+22/-3); (+24 more)
LABELS: high priority, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Summary ⏎  ⏎ - add Kimi K3 XTML reasoning and native tool-call parsers ⏎ - add XGrammar structural constraints for automatic, required, and named tool choice ⏎ - auto-detect both parsers from Kimi K3 model configuration ⏎ - support Kimi K3 rendering in Chat Completions and Responses, including dynamic message tools, wire-format tool and response schemas, multimodal prompt IDs, and native tool-call IDs ⏎ - preserve the reasoning state resolved during templ …[truncated]

### L2-ebb1c88d23  (L2, 2026-08-02, sha ebb1c88d232e, PR #33334)
TITLE: config: stop writing config onto the published ServerArgs at three sites (#33334)
SOURCES: path_core
ARTIFACT_HINTS: L2.dispatch.attention_registry
FILES: python/sglang/srt/layers/attention/attention_registry.py (+10/-3); python/sglang/srt/constrained/base_grammar_backend.py (+2/-2); python/sglang/srt/layers/attention/linear/gdn_backend.py (+9/-12); python/sglang/srt/layers/attention/linear/utils.py (+4/-2); python/sglang/srt/mem_cache/unified_radix_cache.py (+0/-11); test/registered/unit/constrained/test_base_grammar_backend.py (+10/-4); test/registered/unit/layers/attention/test_gdn_prefill_backend_policy.py (+5/-16); test/registered/unit/layers/attention/test_linear_attn_config.py (+92/-0); test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py (+3/-0); test/registered/unit/test_server_args_writer_ratchet.py (+1/-1)
BODY: > Part 1 of five, all based on `main` and meant to merge in order. ⏎ > First of five; nothing precedes it. ⏎ > ⏎ > Replaces #33238, which was closed unmerged: GitHub treated the previous ⏎ > chained-base series as a stack, which blocks both base retargeting and ⏎ > every merge path except the async endpoint. The review discussion and the ⏎ > triage of each round of comments is on #33238; the code here is identical to ⏎ > that PR's final revision. ⏎  ⏎ ## What ⏎  ⏎ Thr …[truncated]

### L2-b8109b5d63  (L2, 2026-08-02, sha b8109b5d63d3, PR #33338)
TITLE: config: retire the last process-global config field reads (#33338)
SOURCES: path_core
ARTIFACT_HINTS: L2.dispatch.attention_registry
FILES: python/sglang/srt/layers/attention/attention_registry.py (+3/-1); .claude/skills/sglang-runtime-context/SKILL.md (+4/-4); python/sglang/kernels/ops/layernorm/mhc.py (+2/-2); python/sglang/srt/batch_overlap/two_batch_overlap.py (+2/-2); python/sglang/srt/layers/rotary_embedding/mrope.py (+2/-3); python/sglang/srt/managers/mm_utils.py (+4/-3); python/sglang/srt/mem_cache/allocation.py (+3/-3); python/sglang/srt/models/gpt_oss.py (+1/-2); python/sglang/srt/models/inkling_common/dense_mlp.py (+2/-2); python/sglang/srt/multimodal/processors/base_processor.py (+34/-26); (+3 more)
LABELS: documentation, hicache, jit-kernel
BODY: > Part 5 of five, all based on `main` and meant to merge in order. ⏎ > Applies on top of parts 1–4. Until they land, this PR's diff includes them. ⏎ > ⏎ > Replaces #33244, which was closed unmerged: GitHub treated the previous ⏎ > chained-base series as a stack, which blocks both base retargeting and ⏎ > every merge path except the async endpoint. The review discussion and the ⏎ > triage of each round of comments is on #33244; the code here is identical to ⏎ > …[truncated]

### L2-1a3bea77f2  (L2, 2026-08-02, sha 1a3bea77f2ab, PR #33112)
TITLE: [Feat] DCP + HiCache L2 Support (ported from kimi-k3) (#33112)
SOURCES: symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.pool.mla_token_kv, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/managers/scheduler_components/metrics_reporter.py (+3/-4); python/sglang/srt/mem_cache/hiradix_cache.py (+5/-0); python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py (+9/-0); python/sglang/srt/mem_cache/memory_pool_host.py (+3/-0); python/sglang/srt/mem_cache/pool_host/base.py (+43/-5); python/sglang/srt/mem_cache/pool_host/mla.py (+14/-0); python/sglang/srt/server_args.py (+43/-0); test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dcp.py (+112/-0); test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py (+210/-0)
LABELS: hicache, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-204e0fbac0  (L2, 2026-08-02, sha 204e0fbac0c3, PR #32320)
TITLE: [SM120] Only split touched SWA pages in FlashMLA page-split kernel (#32320)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.kernel.flash_mla_sm120
FILES: python/sglang/kernels/ops/attention/flash_mla_sm120.py (+89/-7); test/registered/kernels/ops/attention/test_flash_mla_backends.py (+130/-0)
LABELS: jit-kernel
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: On SM120 the FlashMLA decode path splits the entire SWA KV pool (pbs=256 -> pbs=64) on every attention layer every decode step, but only ~2*batch pages are actually referenced by the sparse indices. Add a mark + masked-copy pass so only touched pages are copied; the persistent dst buffer keeps stale (unreferenced) data for the rest. ⏎  ⏎ Fully fixed-grid / no D2H sync, so CUDA graph capture/replay is unaffected. Numerically equivalent to the full s …[truncated]

### L2-92087ef4d2  (L2, 2026-08-03, sha 92087ef4d29f, PR #33432)
TITLE: fix(mem_cache): state the MLA KV bound in the DCP index space (#33432)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/mem_cache/memory_pool.py (+7/-1)
LABELS: run-ci, bypass-fastfail
BODY: set_mla_kv_buffer receives a widened logical loc under DCP — set_mla_kv_buffer_kernel keeps `loc % DCP_WORLD_SIZE == DCP_RANK` and then divides by the world size to reach the physical row — but the bounds check compared that logical index against the physical capacity, so it fires spuriously once the pool is small enough for logical indices to exceed it. ⏎  ⏎ Repro on 4x GB300 with test/registered/dcp/test_kimi_linear_dcp4.py and SGLANG_ENABLE_ASYN …[truncated]

### L2-1e64fc1563  (L2, 2026-08-03, sha 1e64fc15633f, PR #33065)
TITLE: [Fix] Honor FlashMLA natural-log LSE in DCP reduction (#33065)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.optimization.weight_absorption
FILES: python/sglang/kernels/ops/attention/dcp_kernels.py (+10/-4); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+12/-3); python/sglang/srt/layers/dcp/comm.py (+12/-2); test/registered/kernels/test_dcp_lse_combine.py (+29/-0)
LABELS: jit-kernel
ISSUES: #33064 [Bug] FlashMLA DCP combines natural-log LSE with base-2 correction
BODY: ## Motivation ⏎  ⏎ FlashMLA returns natural-log softmax LSE, while the MLA DCP merge paths currently treat all decode LSE as base-2. As a result, FlashMLA + DCP computes incorrect cross-rank softmax weights and attention output. ⏎  ⏎ For partition functions `Z = [2, 8]` and a second-shard local output of `10`, the correct contribution is `8`. Interpreting `[ln(2), ln(8)]` with `exp2` produces `7.2330317` instead. ⏎  ⏎ Fixes #33064. ⏎  ⏎ ## Modifications ⏎  ⏎ - Make  …[truncated]

### L2-48dcadc770  (L2, 2026-08-03, sha 48dcadc7703b, PR #31500)
TITLE: [AMD][DI][CI] 5/N Add DSV4 wide-EP16 4-node 2P1D nightly recipes (#31500)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: scripts/ci/slurm/nightly-configs.yaml (+143/-0); scripts/ci/slurm/recipes/mi355x-fp4/dsv4flash/1k1k/2p1d-ep16-mtp.yaml (+107/-0); scripts/ci/slurm/recipes/mi355x-fp4/dsv4flash/1k1k/2p1d-ep16.yaml (+95/-0); scripts/ci/slurm/recipes/mi355x-fp4/dsv4pro/1k1k/2p1d-ep16-mtp.yaml (+107/-0); scripts/ci/slurm/recipes/mi355x-fp4/dsv4pro/1k1k/2p1d-ep16.yaml (+95/-0); scripts/ci/slurm/recipes/mi355x-fp8/dsv4flash/1k1k/2p1d-ep16-mtp.yaml (+107/-0); scripts/ci/slurm/recipes/mi355x-fp8/dsv4flash/1k1k/2p1d-ep16.yaml (+95/-0); scripts/ci/slurm/recipes/mi355x-fp8/dsv4pro/1k1k/2p1d-ep16-mtp.yaml (+107/-0); scripts/ci/slurm/recipes/mi355x-fp8/dsv4pro/1k1k/2p1d-ep16.yaml (+95/-0)
LABELS: amd
BODY: Coauthor: @yctseng0211 @bingxche @michaelzhang-ai ⏎  ⏎ ## What this PR does ⏎  ⏎ Phase-4 of the AMD disaggregation nightly: add **DSV4 Wide-EP16** legs (no-MTP + MTP). The goal is to run **2P1D EP16 across 4 nodes** of the `mi355x` amd-sglang cluster. ⏎  ⏎ ## Topology (Oren's 2P1D config) ⏎  ⏎ - **2 prefill engines** — EP8 / TP8 / DP8, one node each; the router fans requests across both. ⏎ - **1 decode engine** — EP16 / TP16 / DP16, spanning 2 nodes. ⏎ - 4 …[truncated]

### L2-b57721ccf7  (L2, 2026-08-04, sha b57721ccf79b, PR #33427)
TITLE: Enable post-capture KV sizing with DP attention (#33427)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+37/-24)
LABELS: run-ci, bypass-maintenance
BODY: ## Motivation ⏎  ⏎ `ServerArgs.post_capture_kv_sizing_planned()` gates whether the ⏎ `mem_fraction_static` heuristic is allowed to skip the CUDA graph reserve and ⏎ let the runtime size the KV pool from measured free memory after capture. ⏎  ⏎ It currently disables post-capture sizing outright whenever `enable_dp_attention` ⏎ is set. The reason for that exclusion is that post-capture sizing is only safe ⏎ when execution is graph-covered: graphs retain their acti …[truncated]

### L2-1307968605  (L2, 2026-08-04, sha 1307968605fc, PR #32952)
TITLE: [JIT] Drop redundant per-kernel arch overrides (#32952)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/kernels/ops/attention/sparse_mla_q8kv8_prefill_sm90.py (+14/-15); python/sglang/kernels/jit/__main__.py (+7/-10); python/sglang/kernels/jit/utils/arch.py (+73/-11); python/sglang/kernels/ops/attention/qprep_bf16_fp8_sm90.py (+16/-17); python/sglang/kernels/ops/gemm/fp8_blockwise_gemm.py (+13/-23)
LABELS: enhancement, high priority, ready-for-review, run-ci, jit-kernel, run-ci-extra
BODY: > Generated by Claude. ⏎  ⏎ ## Motivation ⏎  ⏎ `get_jit_cuda_arch()` already resolves the local GPU to its arch-specific target (`_cuda_arch_suffix`: 9.x/10.x+ → `a`, 12.0 → `f`), so the three kernels that wrapped `load_jit` in `override_jit_cuda_arch` to force an `a` target were pinning what detection hands them anyway. Those overrides predate the default-suffix logic and are now noise. ⏎  ⏎ Note this is *sglang's* JIT default, not nvcc's — plain `-arch=sm_ …[truncated]

### L2-0753663b8e  (L2, 2026-08-04, sha 0753663b8ea7, PR #33586)
TITLE: [CI] Trim redundant B200 test registrations (#33586)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/attention/unittests/dense/test_fa3.py (+0/-1); test/registered/backends/test_flashinfer_trtllm_gen_attn_backend.py (+0/-65); test/registered/lora/test_lora_qwen3_5_35b_a3b_logprob_diff.py (+1/-1); test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py (+1/-1); test/registered/models_e2e/test_gpt_oss_4gpu_bf16.py (+0/-1); test/registered/models_e2e/test_qwen35_fp4_flashinfer.py (+0/-83); test/registered/models_e2e/test_qwen35_fp4_mtp.py (+1/-29); test/registered/spec/eagle/test_eagle_dp_attention.py (+1/-4); test/registered/spec/eagle/test_eagle_infer_beta_dp_attention.py (+0/-85)
LABELS: lora, run-ci
BODY: ## Summary ⏎ - Remove B200 registrations that duplicate default-path or Hopper coverage, and move hardware-agnostic tests off B200 runners ⏎ - `base-c-test-4-gpu-b200` est total: 90 min -> ~63 min; `nightly-4-gpu-b200` -5 min ⏎  ⏎ ## Removed (pin-equals-default / duplicate coverage) ⏎ - `test_qwen35_fp4_flashinfer.py` + the `TestQwen35FP4MTPFlashInfer` class: on SM100 the linear-attn (GDN) backend already defaults to flashinfer and full attention defaults  …[truncated]

### L2-abddb1c7e9  (L2, 2026-08-04, sha abddb1c7e9d6, PR #32541)
TITLE: [Kimi] Support kimi-k3 (#32541)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.backend.trtllm_mla, L2.backend.cutedsl_mla, L2.backend.tokenspeed_mla, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/attention_registry.py (+12/-0); python/sglang/srt/layers/attention/cutedsl_mla_backend.py (+64/-16); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+26/-3); .github/workflows/pr-test.yml (+16/-0); docker/Dockerfile (+1/-1); docs/docs/references/environment_variables.mdx (+2/-2); python/pyproject.toml (+1/-1); python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh (+3/-1); python/sglang/kernels/jit/csrc/moe/align_single_token.cuh (+107/-0); python/sglang/kernels/ops/moe/moe_align_single_token.py (+49/-0); (+129 more)
LABELS: documentation, high priority, quant, amd, dependencies, Multi-modal, hicache, blackwell, npu, run-ci
BODY: Day-0 support for the Kimi K3 model. ⏎  ⏎ #### Nvidia Support ⏎ Day 0 Cuda 13 image: `docker pull lmsysorg/sglang:kimi-k3` ⏎ Day 0 Cuda 12 image: `docker pull lmsysorg/sglang:kimi-k3-cu12` ⏎  ⏎ #### AMD Support ⏎ Day 0 image: `docker pull lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727` ⏎ More details: #32548 ⏎  ⏎ #### Links ⏎  ⏎ Cookbook: https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3 ⏎ Blog: https://www.lmsys.org/blog/2026-07-27-kimi-k3-da …[truncated]

### L2-53804d609c  (L2, 2026-08-04, sha 53804d609cb3, PR #32438)
TITLE: [CI][XPU] Stabilize XPU CI: pin UMD/IGC, retry infra flakes, right-size EAGLE3 (#32438)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/xpu.Dockerfile (+30/-4); .github/workflows/pr-test-xpu.yml (+28/-2); python/sglang/test/ci/ci_utils.py (+24/-0); python/sglang/test/test_utils.py (+3/-0); scripts/ci/xpu/xpu_ci_start_container.sh (+51/-0); test/registered/spec/eagle/test_spec_eagle_parity.py (+1/-1)
LABELS: intel, ci, xpu, run-ci
BODY: ## Summary ⏎  ⏎ Four changes to get B580 XPU CI green: ⏎  ⏎ - **`docker/xpu.Dockerfile`** — pin Level-Zero UMD, IGC, and libigdgmm; `apt-mark hold` to stop the rolling PPA from breaking the pinned host xe KMD. ⏎ - **`python/sglang/test/ci/ci_utils.py`** — retry transient GPU OOMs and server-start timeouts once; also retry per-file timeouts under `--enable-retry`. ⏎ - **`.github/workflows/pr-test-xpu.yml`** — pass `--enable-retry` to both stage-a and stage-b  …[truncated]

### L2-6c05aaae7e  (L2, 2026-08-04, sha 6c05aaae7e39, PR #33063)
TITLE: [trtllm_mha] perf: Stop allocating per-layer scratch inside the decode CUDA graph (#33063)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/trtllm_mha_backend.py (+35/-2); python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py (+1/-0)
LABELS: blackwell
BODY: ## Motivation ⏎  ⏎ Two allocations in the TRTLLM-MHA decode path emit a fill kernel per attention ⏎ layer per forward, and CUDA-graph capture bakes them into every replay: ⏎  ⏎ 1. FlashInfer allocates and zeroes a fresh multi-CTA KV counter buffer on every ⏎    `trtllm_batch_decode_with_kv_cache` call when the caller supplies none — even ⏎    though its own docstring says the kernel self-resets the counters at the end of ⏎    each launch and that callers …[truncated]

### L2-81c7a54ecd  (L2, 2026-08-05, sha 81c7a54ecda1, PR #33433)
TITLE: [NVIDIA] Use sm_100f instead of sm_100a for sgl-kernel and FlashMLA (#33433)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L2.build.flashmla_sgl_kernel
FILES: python/sglang/kernels/aot/cmake/flashmla.cmake (+2/-5); python/sglang/kernels/aot/CMakeLists.txt (+10/-2); python/sglang/kernels/aot/csrc/moe/fp8_blockwise_moe_kernel.cu (+3/-0)
LABELS: sgl-kernel, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ CUDA 12.9+ supports `sm_100f` which is able to run on all sm_10x-family GPUs, whereas `sm_100a` can only run on `sm_100a` GPUs. ⏎  ⏎ We can use this to reduce wheel size by removing redundant `sm_103` binaries while also being able to run on Rubin `sm_107` when it is available. ⏎  ⏎ See https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/ for more information. ⏎  ⏎ ##  …[truncated]

### L2-d2c405f19d  (L2, 2026-08-05, sha d2c405f19df9, PR #28040)
TITLE: [Intel GPU] DeepSeek V4 8/N: use sgl-kernel implementation of fused_k_norm_rope_flashmla on XPU (#28040)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dsv4/elementwise.py (+12/-4)
LABELS: intel, xpu, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-1a045669e4  (L2, 2026-08-05, sha 1a045669e497, PR #33641)
TITLE: [CI] Merge tokenizer worker tests and drop redundant triton attention e2e (#33641)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/attention/test_triton_attention_backend.py (+0/-71); test/registered/mla/test_flashmla.py (+0/-93); test/registered/mla/test_mla_flashinfer.py (+0/-84); test/registered/mla/test_mla_int8_deepseek_v3.py (+2/-63); test/registered/models_e2e/test_gpt_oss_4gpu_bf16.py (+0/-22); test/registered/tokenizer/test_multi_detokenizer.py (+0/-80); test/registered/tokenizer/test_multi_tokenizer.py (+5/-0); test/registered/unit/layers/quantization/test_int8_linear_methods.py (+126/-0)
LABELS: deepseek, run-ci
BODY: Trims on the per-commit H100 suites (base-b 1-gpu ~ -548s, base-c 4-gpu -220s), replacing redundant e2e matrices with existing or new layer-level coverage: ⏎  ⏎ - Merge `test_multi_detokenizer.py` into `test_multi_tokenizer.py`: the two flags are orthogonal, so one server now runs with both `--tokenizer-worker-num 8` and `--detokenizer-worker-num 4`; the detokenizer file's ttft test was identical to the tokenizer one modulo the flag. (-211s) ⏎ - Remove …[truncated]

### L2-c0ef548eef  (L2, 2026-08-05, sha c0ef548eefe5, PR #33363)
TITLE: [misc] Unify MLA `scaling` init and remove dead buffer / scaling code (#33363)
SOURCES: path_core, path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L2.backend.flashmla, L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+0/-1); python/sglang/srt/layers/attention/flashmla_backend.py (+0/-1); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+0/-1); python/sglang/srt/configs/model_config.py (+14/-34); python/sglang/srt/model_executor/cuda_graph_buffer_registry.py (+20/-1); python/sglang/srt/model_executor/input_buffers.py (+6/-35); test/registered/unit/configs/test_model_config_scaling.py (+29/-1)
LABELS: blackwell, npu, run-ci
BODY: ## Summary ⏎ - Route all five MLA branches through one `scaling` initializer, which also fixes `sm_scale` for a `rope_scaling` dict carrying neither `rope_type` nor `type` ⏎ - Drop dead code in the graph input buffers and in three MLA attention backends ⏎ - Make the two independent index-buffer reset paths assert they agree ⏎  ⏎ ## Fix: MLA `sm_scale` skipped the yarn mscale for type-less `rope_scaling` ⏎ - `ModelConfig` resolved `rope_type` with an extra `o …[truncated]

### L2-bebebb8f6c  (L2, 2026-08-05, sha bebebb8f6c8b, PR #)
TITLE: config: retire the alias-form process-global config reads
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: 
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L2-99cfc90658  (L2, 2026-08-05, sha 99cfc9065822, PR #)
TITLE: config: retire ServerArgs.override in favour of derive()
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py; python/sglang/srt/speculative/draft_worker_common.py
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L2-4ea227fa91  (L2, 2026-08-05, sha 4ea227fa91fd, PR #)
TITLE: config: the draft runner carries its own attention backend
SOURCES: path_core
ARTIFACT_HINTS: L2.dispatch.attention_registry
FILES: python/sglang/srt/layers/attention/attention_registry.py
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L2-3b4fac5b99  (L2, 2026-08-05, sha 3b4fac5b9967, PR #31865)
TITLE: [XPU] DeepSeek V4: use sgl-kernel-xpu implemetation of flash_mla_sparse_fwd for prefill (#31865)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/deepseek_v4_backend.py (+4/-1)
LABELS: deepseek, intel, xpu, run-ci, run-ci-extra
BODY: Depend on sgl-kernel-xpu implementation https://github.com/sgl-project/sgl-kernel-xpu/pull/337. ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [C …[truncated]

### L2-b3cdd016ba  (L2, 2026-08-05, sha b3cdd016baeb, PR #33556)
TITLE: Add Ling-3.0-flash cookbook (#33556)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/InclusionAI/Ling-3.0-flash.mdx (+206/-0); docs/cookbook/autoregressive/InclusionAI/Ring-2.6-1T.mdx (+0/-1); docs/cookbook/autoregressive/intro.mdx (+1/-1); docs/docs.json (+1/-0); docs/src/snippets/_deployment.jsx (+6/-1); docs/src/snippets/_playground.jsx (+64/-16); docs/src/snippets/configs/inclusionAI/ling-3.0-flash-benchmarks.jsx (+97/-0); docs/src/snippets/configs/inclusionAI/ling-3.0-flash.jsx (+615/-0)
LABELS: documentation, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Add the SGLang Cookbook page for [inclusionAI/Ling-3.0-flash](https://huggingface.co/inclusionAI/Ling-3.0-flash): a 124B-total / 5.1B-active hybrid KDA + MLA MoE with default thinking and structured tool calling. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add the config-driven cookbook page and Deployment/Playground widgets. ⏎ - Add single-node BF16 and FP8 recipes across H20-3e, H200, H800, H100, B200, and GB300. ⏎ - Add the page to the InclusionAI cookboo …[truncated]

### L2-c9506d023f  (L2, 2026-08-06, sha c9506d023f33, PR #33148)
TITLE: [Quantization] Route per-tensor FP8 checkpoints to FlashInfer on SM90 (#33148)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8_utils.py (+16/-8); python/sglang/srt/layers/quantization/modelopt_quant.py (+11/-7)
LABELS: quant, sgl-kernel, run-ci
ISSUES: #32993 [Feature] Route per-tensor FP8 checkpoints to FlashInfer on SM89/SM90
BODY: Closes #32993 ⏎  ⏎ ## Motivation ⏎  ⏎ A per-tensor FP8 checkpoint carries one scalar weight scale and one scalar activation scale. Those layers go to the AOT CUTLASS rowwise GEMM, which only takes a per-token activation vector and a per-channel weight vector. So both scalars get broadcast: the weight scale at load time in `convert_to_channelwise`, and the activation scale on every forward in `static_quant_fp8(repeat_scale=True)`. The GEMM then runs a …[truncated]

### L2-434e646282  (L2, 2026-08-06, sha 434e646282e5, PR #28836)
TITLE: [Deps] Upgrade CUDA PyTorch stack to 2.13 (#28836)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: docker/Dockerfile (+21/-27); docker/kimi_k3/kimi_k3_cu12.Dockerfile (+2/-2); docker/kimi_k3/kimi_k3_cu13.Dockerfile (+2/-2); docker/sgl-deep-gemm.Dockerfile (+1/-1); python/pyproject.toml (+4/-4); .github/workflows/_docker-build-and-publish.yml (+4/-4); .github/workflows/_pr-test-stage.yml (+1/-0); .github/workflows/nightly-72-gpu-gb200.yml (+1/-1); .github/workflows/pr-test-jit-kernel.yml (+1/-0); .github/workflows/pr-test-multimodal-gen.yml (+1/-0); (+24 more)
LABELS: documentation, high priority, quant, dependencies, Multi-modal, deepseek, sgl-kernel, ready-to-merge, blackwell, run-ci
BODY: ## Summary ⏎  ⏎ torch: 2.11.0 → 2.13.0 ⏎ torchvision: 0.26.0 → 0.28.0 ⏎ triton: 3.6.0 → 3.7.1 ⏎ torchaudio: stays at 2.11.0 ⏎ triton_kernels: 3.6.0 → 3.7.1 ⏎ torchcodec: 0.11.1 → 0.15.0 ⏎  ⏎ ## Test Plan ⏎  ⏎ CI ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  …[truncated]

### L2-971932d661  (L2, 2026-08-06, sha 971932d66117, PR #33650)
TITLE: [Kimi-K3] Allow DSPARK verify on cutedsl_mla (fold_sq) (#33650)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/overrides.py (+4/-2)
BODY: `_dspark_verify_on_decode_backend` caps `cutedsl_mla` at `q_len <= 4`, from when the cute-dsl MLA decode kernel rejected `q_len >= 5`. flashinfer's monolithic MLA decode now folds `seq_len_q` into the head dim (`fold_sq`, added in flashinfer #3309, gated-fixed in #3664, present in the pinned 0.6.15.post1), so `q_len > 4` is supported. ⏎  ⏎ K3 DSPARK verify uses `q_len = block_size(7) + 1 = 8`, which exceeded the cap, so verify silently fell back to …[truncated]

### L2-dd7e4c91e2  (L2, 2026-08-06, sha dd7e4c91e2e1, PR #33785)
TITLE: Fix Mistral-Large-3 EAGLE draft skipping DeepseekV2Model.__init__ (#33785)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/mistral_large_3_eagle.py (+6/-46); test/registered/8-gpu-models/test_mistral_large3.py (+19/-8)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ The nightly `nightly-test-general-8-gpu-b200` job fails on `test/registered/8-gpu-models/test_mistral_large3.py`, in the **`Mistral-Large-3-675B-Instruct-2512 [TP8+MTP]`** variant. Plain TP8 and NVFP4 pass accuracy and perf. ⏎  ⏎ > **Two independent problems, one stacked on the other.** Commit 1 fixes a real crash (`AttributeError`). Commit 2 suppresses a warmup-only NaN probe that the crash was hiding, matching what the Nemotron-3 nig …[truncated]

### L2-18e6c61c21  (L2, 2026-08-06, sha 18e6c61c21ad, PR #29677)
TITLE: [AMD] perf: compact Triton extend-attention for ragged prefill (AMD/HIP-only) (#29677)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/extend_attention.py (+100/-4); python/sglang/kernels/ops/attention/verify_splitkv.py (+6/-0); python/sglang/srt/environ.py (+11/-0); python/sglang/srt/layers/attention/triton_backend.py (+14/-0); python/sglang/srt/managers/schedule_policy.py (+103/-1); python/sglang/srt/managers/scheduler.py (+8/-0); test/registered/attention/test_triton_attention_kernels.py (+127/-0); test/registered/attention/test_verify_splitkv.py (+14/-0); test/registered/unit/managers/test_prefill_adder.py (+66/-1)
LABELS: documentation, high priority, amd, deepseek, npu, run-ci, jit-kernel, bypass-fastfail
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ This PR improves Triton extend-attention performance for ragged mixed-prefill batches by adding an optional **compact query-tile launch grid**. ⏎  ⏎ The legacy grid is rectangular and sized by `max_len_extend`: ⏎  ⏎ ```python ⏎ grid = (batch_size, head_num, ceil(max_len_extend / BLOCK_M)) ⏎ ``` ⏎  ⏎ In a mixed batch, every short decode/partial-prefill row pays tile work proportional to the longest extend row. The compact path launches on …[truncated]

### L2-8a22b8305d  (L2, 2026-08-07, sha 8a22b8305d59, PR #33956)
TITLE: docker: add Kimi K3 artifacts and build hpc-ops with C++20 (#33956)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+49/-3)
BODY: ## Motivation ⏎  ⏎ The generic CUDA image does not include the pinned TRT-LLM generated-MoE cubin pool or the FlashInfer CuTeDSL MLA decode-context-parallel runtime patch currently installed by the Kimi K3 CUDA 12 and CUDA 13 images. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add an independent builder stage that downloads the pinned cubin archive, verifies its SHA-256 digest, extracts it, and requires exactly 1,696 cubin files. ⏎ - Copy the verified pool into both framewor …[truncated]

### L2-1480687cff  (L2, 2026-08-07, sha 1480687cff5a, PR #33532)
TITLE: [CP]: Support CP V2 Strategy for dsv4 (#33532)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py (+16/-6); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+1/-0); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+32/-8); python/sglang/srt/layers/attention/dsa/utils.py (+3/-2); python/sglang/srt/layers/attention/dsv4/compressor.py (+3/-6); python/sglang/srt/layers/cp/utils.py (+42/-6); python/sglang/srt/model_executor/runner/eager_runner.py (+6/-0); python/sglang/srt/models/deepseek_v4.py (+26/-14); python/sglang/srt/models/deepseek_v4_nextn.py (+18/-6); test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py (+14/-11)
LABELS: deepseek, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-62a28197c0  (L2, 2026-08-07, sha 62a28197c071, PR #32556)
TITLE: Autotune flashinfer extend buckets at warmup (#32556)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.aiter_mla, L2.backend.sparse_mla_adapters, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+4/-0); python/sglang/srt/environ.py (+4/-0); python/sglang/srt/layers/attention/aiter_backend.py (+5/-0); python/sglang/srt/layers/attention/base_attn_backend.py (+5/-0); python/sglang/srt/layers/attention/dsa_backend.py (+4/-0); python/sglang/srt/layers/attention/flashinfer_backend.py (+4/-0); python/sglang/srt/layers/attention/hybrid_attn_backend.py (+4/-0); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+3/-0); python/sglang/srt/layers/attention/minimax_sparse_backend.py (+3/-0); python/sglang/srt/layers/attention/tbo_backend.py (+3/-0); (+6 more)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Problem ⏎  ⏎ The warmup flashinfer autotune runs a **decode-shaped** dummy forward (`BaseRunner.warmup` → `_autotune_buffers()`, 1 token/req), so tuned buckets only cover token counts up to the decode batch size. Larger prefill/extend batches fall outside the tuned buckets at serving time and run flashinfer's slow default heuristic: ⏎  ⏎ ``` ⏎ [AutoTuner]: No tuned config covers flashinfer::trtllm_fp4_block_scale_moe input_shapes=(torch.Size([16384 …[truncated]

### L2-9e3f6b746b  (L2, 2026-08-07, sha 9e3f6b746b56, PR #33665)
TITLE: fix(mamba): widen causal_conv1d token offsets to int64 (#33665)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/mamba/causal_conv1d_triton.py (+11/-5)
LABELS: run-ci, jit-kernel, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ `_causal_conv1d_fwd_kernel` indexes `x` and `o` with 32-bit arithmetic: ⏎  ⏎ ```python ⏎ x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim ⏎ ``` ⏎  ⏎ Under a channel-last layout `stride_x_token` is the full token pitch (for a fused QKV projection this is the whole projection width, e.g. 36864), so the product overflows int32 once the packed batch reaches `2**31 / stride_x_token` tokens. The offset wraps negative …[truncated]

### L2-07297049e9  (L2, 2026-08-07, sha 07297049e941, PR #33925)
TITLE: config: route DCP topology reads through get_parallel() (#33925)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.attention_registry
FILES: python/sglang/srt/layers/attention/attention_registry.py (+2/-2); python/sglang/srt/disaggregation/common/conn.py (+2/-2); python/sglang/srt/layers/attention/dsa/utils.py (+1/-1); python/sglang/srt/managers/scheduler.py (+3/-3); python/sglang/srt/managers/scheduler_components/invariant_checker.py (+3/-8); python/sglang/srt/mem_cache/allocation.py (+3/-3); python/sglang/srt/mem_cache/kv_cache_configurator.py (+4/-4); python/sglang/srt/model_executor/model_runner.py (+4/-3); python/sglang/srt/model_executor/pool_configurator.py (+2/-1); python/sglang/srt/model_executor/runner/base_runner.py (+4/-2); (+4 more)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Follow-up to #33170, which routed 106 parallel config-leaf reads through `get_parallel()` but deliberately left the five live-shadowed topology sizes (`tp/pp/dcp/attn_cp/moe_dp_size`) on `server_args`: ⏎  ⏎ > The five live-shadowed topology sizes keep their `server_args` reads: the live `@property` wins on the accessor, and conditionally-initialized groups would fail loud at unconditional call sites. ⏎  ⏎ This PR does the `dcp_size` slice. …[truncated]

### L2-572434e2f6  (L2, 2026-08-07, sha 572434e2f6a8, PR #33886)
TITLE: [diffusion] Z-Image bit-exact fused qk-norm (H200 Turbo 1024px e2e -6.4%) (#33886)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/diffusion/triton/zimage_native_norm.py (+160/-0); python/sglang/multimodal_gen/runtime/models/dits/zimage.py (+75/-13); python/sglang/multimodal_gen/test/unit/test_zimage_qknorm_fusion.py (+51/-0)
LABELS: run-ci, diffusion, jit-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ PR #29742 fixed Z-Image accuracy by keeping the model's native bf16 RMSNorm ⏎ trajectory, and on the main attention path (precomputed rope cache) it passes ⏎ `allow_inplace=False`, which forces the qk-norm onto the eager ⏎ `ZImageRMSNorm` chain. A torch-profiler pass over one Z-Image-Turbo denoise ⏎ step on H200 (default 1024x1024 preset, latest main) shows what that costs: ⏎  ⏎ | launching op | % of denoise-step GPU time | ⏎ |---|---| ⏎ | eager q …[truncated]

### L2-a5af27f49e  (L2, 2026-08-07, sha a5af27f49ed5, PR #33887)
TITLE: config: retire ServerArgs.derive; per-runner values are constructor arguments (#33887)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: .claude/skills/sglang-runtime-context/SKILL.md (+22/-22); python/sglang/srt/disaggregation/encode_server.py (+13/-5); python/sglang/srt/runtime_context.py (+12/-2); python/sglang/srt/server_args.py (+4/-45); python/sglang/test/kits/attention_unittest/runner_modes/speculative_draft_runner.py (+5/-5); python/sglang/test/test_utils.py (+22/-0); test/registered/attention/unittests/hybrid_linear/test_flashinfer_mla_chunk_metadata.py (+3/-2); test/registered/unit/test_runtime_context.py (+4/-6); test/registered/unit/test_runtime_context_override.py (+1/-3); test/registered/unit/test_server_args_derive.py (+0/-79); (+2 more)
LABELS: documentation, speculative-decoding, ready-to-merge
BODY: ## What ⏎  ⏎ `ServerArgs.derive(source, **fields)` — the copy-and-edit entry on the config object — is deleted. Nothing in the tree derives a config any more: ⏎  ⏎ - The last production call built the encoder DP worker's per-worker copy (`base_gpu_id=gpu_id, tp_size=1`). The `tp_size` write was dead (encoder DP mode already validates `--tp-size 1` before spawning workers), and the device is one instance's placement, not a configuration difference: `MMEnc …[truncated]

### L2-eda0ddc260  (L2, 2026-08-07, sha eda0ddc260d4, PR #33888)
TITLE: config: delete the dead get_server_args() bindings across the repo (#33888)
SOURCES: path_core
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+1/-2); python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py (+0/-1); python/sglang/srt/layers/moe/topk.py (+1/-1); python/sglang/srt/managers/schedule_batch.py (+0/-1); python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py (+1/-3); python/sglang/srt/mem_cache/storage/flexkv/flexkv_radix_cache.py (+0/-3); python/sglang/srt/mem_cache/storage/lmcache/lmc_radix_cache.py (+1/-2); python/sglang/srt/model_loader/loader.py (+0/-1); python/sglang/srt/models/deepseek_v2.py (+0/-3); python/sglang/srt/models/minimax_m3_vl.py (+1/-2); (+4 more)
LABELS: deepseek, ready-to-merge
BODY: ## What ⏎  ⏎ Deletes the 15 dead `server_args = get_server_args()` bindings that earlier bag-migration flips left behind, plus the imports they orphaned: `sampling_batch_info`, `deepseek_v2` (3), `forward_mla`, `minimax_m3_vl`, `routed_experts`, `indexer_topk`, `loader`, `deepseek_v4_memory_pool` (2), `schedule_batch`, `inkling_sconv_backend`, and the flexkv/lmcache commit-prefix paths. ⏎  ⏎ In `layers/moe/topk.py` the call itself is load-bearing — it pr …[truncated]

### L2-b61a06921e  (L2, 2026-08-07, sha b61a06921ef0, PR #33889)
TITLE: moe: the shared-experts-fusion decision is a per-runner value the loader installs (#33889)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: .claude/skills/sglang-runtime-context/SKILL.md (+15/-9); python/sglang/srt/arg_groups/overrides.py (+0/-12); python/sglang/srt/layers/moe/utils.py (+90/-0); python/sglang/srt/model_executor/model_runner.py (+8/-5); python/sglang/srt/model_executor/model_runner_components/load_model_utils.py (+4/-4); python/sglang/srt/model_loader/loader.py (+10/-0); python/sglang/srt/models/deepseek_nextn.py (+3/-1); python/sglang/srt/models/deepseek_ocr.py (+21/-5); python/sglang/srt/models/deepseek_v2.py (+37/-44); python/sglang/srt/models/deepseek_v4.py (+18/-27); (+30 more)
LABELS: documentation, Multi-modal, deepseek, speculative-decoding, diffusion
BODY: ## What ⏎  ⏎ A draft's construction used to rewrite the process config record. Two writers, two shapes: ⏎  ⏎ **1. The shared-experts-fusion decision becomes a per-runner value the loader installs.** `declare_load_time_override` wrote the fusion decision to the target's bags — and an MTP/nextn draft IS a DeepSeek/GLM/Qwen3.5/MiniMax model, so a draft whose checkpoint differs from the target's (quantization) corrupted the target's record. This is ancestral …[truncated]

### L2-6679d9b60c  (L2, 2026-08-07, sha 6679d9b60c4d, PR #33981)
TITLE: [AMD] Add K3 verified mla kernel for DSpark on triton backend (#33981)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/verify_mla.py (+647/-0); python/sglang/srt/layers/attention/triton_backend.py (+24/-3)
LABELS: amd, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Kimi-K3 DSpark throughput is slow, at high conc it is even slower than non-DSpark setting. ⏎  ⏎ Root cause: the target-verify attention runs through verify_splitkv, which is invoked on every MLA full-attention layer (~24 calls/step for K3). That kernel is designed for standard MHA, not MLA: it parallelizes one program per query head, so it re-reads the single shared MLA latent (h_kv=1) once per head (~12x redundant load at TP8) …[truncated]

### L2-86f373daff  (L2, 2026-08-07, sha 86f373daff12, PR #34044)
TITLE: docs(cookbook): DeepSeek-V4-Flash-0731 — drop chunked-prefill/autotune flags on B300 low-latency (#34044)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx (+1/-3)
LABELS: documentation, deepseek, run-ci, run-ci-extra
BODY: Drops `--chunked-prefill-size 4096` and `--disable-flashinfer-autotune` from the **Flash Official (0731)** low-latency recipe on **B300** (`flash-official` / `fp4`), and marks the cell verified. ⏎  ⏎ The cell has no benchmarks entry, so it keeps rendering pending. ⏎  ⏎ 🤖 Generated with [Claude Code](https://claude.com/claude-code) ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #31226045307](https://github.com/sgl-project/sglang/action …[truncated]

### L2-c9444deef4  (L2, 2026-08-07, sha c9444deef4a5, PR #34041)
TITLE: Docker: install DeepEP from release wheels (#34041)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+16/-89); .github/workflows/_docker-build-and-publish.yml (+0/-4); .github/workflows/nightly-72-gpu-gb200.yml (+0/-1); .github/workflows/release-docker-dev.yml (+0/-1)
BODY: ## Motivation ⏎  ⏎ `sgl-deep-ep==0.1.0` is now a released SGLang dependency. The main Dockerfile should consume the published wheels instead of cloning and compiling DeepEP for every image build. ⏎  ⏎ ## Modifications ⏎  ⏎ - Remove the `deepep_builder` stage and the framework-stage DeepEP wheel/source copies. ⏎ - Let CUDA 13 images install the public `sgl-deep-ep==0.1.0` wheel through the existing pyproject dependency. ⏎ - Preinstall `sgl-deep-ep==0.1.0+cu129` f …[truncated]

### L2-2c0188cc78  (L2, 2026-08-08, sha 2c0188cc7822, PR #34100)
TITLE: [Fix] Give the piecewise CUDA graph test stub an `hf_config` (#34100)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/unit/configs/test_multimodal_piecewise_cuda_graph.py (+4/-1)
LABELS: Multi-modal, run-ci
BODY: `base-a-test-cpu` is currently red on main: `test_trtllm_mla_stays_on_breakable_and_is_disabled_by_compatibility` raises `AttributeError: 'types.SimpleNamespace' object has no attribute 'hf_config'`. ⏎  ⏎ #32785 added an `is_deepseek_dsa(self.get_model_config().hf_config)` term to the MLA rule in `_disable_breakable_cudagraph_if_incompatible`, but that test's stub only sets `is_multimodal_piecewise_cuda_graph_supported`. It is the only case in the fi …[truncated]

### L2-c4f018ba1d  (L2, 2026-08-08, sha c4f018ba1dac, PR #31531)
TITLE: [Refactor] Separate ROCm-specific DeepSeek MHA and MLA forward paths (#31531)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_backend_handler.py (+17/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/__init__.py (+5/-1); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py (+9/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py (+18/-129); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha_rocm.py (+324/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+75/-452); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py (+1/-1); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py (+849/-0); python/sglang/srt/models/deepseek_v2.py (+29/-1); python/sglang/srt/batch_overlap/operations.py (+1/-1); (+2 more)
LABELS: amd, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Separate ROCm-specific DeepSeek attention logic from the shared MHA and MLA forward implementations to improve code organization and maintainability. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ - Added dedicated ROCm helpers in `forward_mha_rocm.py` and `forward_mla_rocm.py`. ⏎ - Updated the shared MHA and MLA forward paths to dispatch to the ROCm helpers. ⏎ - Reused the extracted ROCm absorb BMM helpers in the fused RoPE path. ⏎ - Preserved existin …[truncated]

### L2-6185ed8011  (L2, 2026-08-08, sha 6185ed801120, PR #34045)
TITLE: Add registered short-conv tests and backend extensions (#34045)
SOURCES: path_core
ARTIFACT_HINTS: L2.dispatch.attention_registry
FILES: python/sglang/srt/layers/attention/attention_registry.py (+5/-0); python/sglang/srt/configs/linear_attn_model_registry.py (+6/-1); python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py (+4/-0); python/sglang/srt/models/inkling_common/sconv.py (+48/-34); test/registered/kernels/ops/mamba/test_sconv_cache.py (+171/-0)
LABELS: run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎   The linear-attention model registry allows model integrations to select a backend, but registered models cannot currently customize how that backend is wrapped with full attention. Short-convolution layers also invoke their compute kernels directly, making it difficult to customize numerical implementations while reusing the existing cache and state-management lifecycle. ⏎  ⏎ ## Changes ⏎  ⏎ - Add optional config predicates and hyb …[truncated]

### L2-ba7abd4f92  (L2, 2026-08-08, sha ba7abd4f92de, PR #30964)
TITLE: [AMD] Support DeepSeek V4 DSpark on AMD HIP platform (#30964)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/runtime.py (+51/-0); python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py (+31/-0); python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py (+102/-27); python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py (+28/-0); python/sglang/srt/models/deepseek_v4_dspark.py (+27/-1); python/sglang/srt/speculative/dspark_components/dspark_kv_inject.py (+94/-15); python/sglang/srt/speculative/dspark_components/dspark_verify.py (+34/-9); python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py (+18/-0); test/registered/amd/test_deepseek_v4_pro_fp4_dspark.py (+201/-0)
LABELS: amd, deepseek, sgl-kernel, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Follow https://github.com/sgl-project/sglang/pull/30261, support dspark for deepseek v4 on AMD platform ⏎  ⏎ ## Modifications ⏎  ⏎ `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` ⏎ Enables HIP ragged verify CUDA graph support for DSpark. Adds ragged layout metadata handling, GPU prefill expansion/compression planning, token-tier graph keys, and a draft-worker token-count fix. Removes previous HIP NotImplementedEr …[truncated]

### L2-db3898fec1  (L2, 2026-08-08, sha db3898fec136, PR #32785)
TITLE: fix: avoid piecewise prefill graph for trtllm_mla (#32785)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+6/-0); test/registered/unit/configs/test_multimodal_piecewise_cuda_graph.py (+47/-3)
LABELS: Multi-modal, run-ci
ISSUES: #32655 [Bug] Kimi-K2.6 NVFP4 throughput regression after enabling piecewise prefill CUDA graph (#30889)
BODY: ## Motivation ⏎  ⏎ Fixes #32655. ⏎  ⏎ #30889 upgrades the default prefill CUDA graph backend from `breakable` to `tc_piecewise` for allowlisted multimodal models. Kimi-K2.6 on Blackwell auto-selects `trtllm_mla`, so that upgrade moves prefill capture onto the FlashInfer paged-MLA fallback. The fallback converts the full FP8 KV cache buffer and substantially regresses TTFT, while decode TPOT remains unchanged. ⏎  ⏎ This is orthogonal to #32288: that PR fixed  …[truncated]

### L2-4ad5bb5d9a  (L2, 2026-08-08, sha 4ad5bb5d9ae0, PR #33400)
TITLE: [jit_kernel] Move JIT kernels into namespace sglang (#33400)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.kernel.set_mla_kv_buffer, L2.kernel.concat_mla, L2.kernel.set_mla_kv_concat_q
FILES: .claude/skills/add-jit-kernel/SKILL.md (+3/-2); docs/docs/developer_guide/development_jit_kernel_guide.mdx (+9/-2); python/sglang/kernels/jit/csrc/add_constant.cuh (+2/-2); python/sglang/kernels/jit/csrc/attention/fixup_zero_kv.cuh (+2/-2); python/sglang/kernels/jit/csrc/attention/fused_fp8_qkv_kv_cache.cuh (+2/-2); python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh (+3/-3); python/sglang/kernels/jit/csrc/attention/kda_packed_decode.cuh (+2/-2); python/sglang/kernels/jit/csrc/deepseek_v32/indexer_k.cuh (+2/-2); python/sglang/kernels/jit/csrc/deepseek_v4/c128.cuh (+2/-2); python/sglang/kernels/jit/csrc/deepseek_v4/c128_online.cuh (+3/-7); (+168 more)
LABELS: documentation, quant, lora, hicache, run-ci, jit-kernel
DEEP_STUDY: deep-study: introduced the defect fixed in case sglang:ec5199b906 (fix PR 34106)
BODY: > Generated by Claude. ⏎  ⏎ ## Motivation ⏎  ⏎ Every JIT kernel used to sit in the global namespace (or an anonymous one), with a handful of files having already started on `namespace sglang` and reaching back out via `using namespace sglang;` / `sglang::`. This unifies that: all JIT C++ under `python/sglang/kernels/jit/` now lives in `namespace sglang`, and no `sglang::` qualification is needed anywhere inside it. ⏎  ⏎ ## Modifications ⏎  ⏎ - **`namespace sglan …[truncated]

### L2-967cac801c  (L2, 2026-08-09, sha 967cac801ce3, PR #34147)
TITLE: [AMD] [CI] Register the DeepSeek-V4-Pro-DSpark MI35x nightly job so its suite actually runs (#34147)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/nightly-test-amd-rocm720.yml (+40/-0)
LABELS: amd, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ [#30964](https://github.com/sgl-project/sglang/pull/30964) added `test/registered/amd/test_deepseek_v4_pro_fp4_dspark.py`, the only test covering the AMD DSpark unified-KV path, and registered it as: ⏎  ⏎ ```python ⏎ register_amd_ci( ⏎     est_time=7200, suite="nightly-amd-8-gpu-mi35x-deepseek-v4-pro-dspark", nightly=True ⏎ ) ⏎ ``` ⏎  ⏎ No job in `nightly-test-amd.yml` or `nightly-test-amd-rocm720.yml` invokes that suite name, so the test never ru …[truncated]

### L2-7120f3ee13  (L2, 2026-08-09, sha 7120f3ee13de, PR #33436)
TITLE: fix: support FA4 backend for GLM4.7-flash (#33436)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/flash_attention_v4.py (+65/-0); test/registered/kernels/ops/attention/test_flash_attention_4.py (+195/-18)
LABELS: run-ci, jit-kernel
DEEP_STUDY: deep-study correctness case sglang:7120f3ee13: class=shape_alignment_edge; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Motivation ⏎  ⏎ Enable GLM-4.7-Flash to serve correctly with `--attention-backend fa4` on Blackwell using default non-deterministic decode and CUDA Graphs. SGLang's absorbed MLA presents 20 Q/QV heads over one latent KV head at TP1, but FA4's packed kernel requires the ratio to be compatible with its 128-row tile. ⏎  ⏎ The current `pack_gqa=False` QV varlen decode path is incorrect for bs>1: queries after the first sequence can read another sequen …[truncated]

### L2-06f32bab6b  (L2, 2026-08-09, sha 06f32bab6baa, PR #33661)
TITLE: [BCG][5/N] MLA Fully Support (#33661)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/kernels/ops/attention/flash_attn/cute/interface.py (+4/-0); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+12/-5); python/sglang/srt/server_args.py (+3/-18); python/sglang/srt/configs/model_config.py (+0/-20); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+4/-24); python/sglang/srt/utils/common.py (+7/-4); test/registered/unit/configs/test_multimodal_piecewise_cuda_graph.py (+12/-7)
LABELS: Multi-modal, blackwell, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-0977b22431  (L2, 2026-08-10, sha 0977b22431cf, PR #34261)
TITLE: [AMD] Restore K3 MLA verify kernel path blocked by can_handle() guard (#34261)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/verify_mla.py (+84/-4)
LABELS: amd, run-ci, jit-kernel
BODY: **Only impact K3, mi355 code path.** ⏎  ⏎  ⏎  ⏎ ## Motivation & Modifications ⏎  ⏎  ⏎  ⏎  ⏎ In PR #33981 we added a triton MLA verify kernel (`verify_mla_fwd`) to improve Kimi-K3 DSpark perf. It reused `verify_splitkv.can_handle()` as its eligibility gate. ⏎  ⏎ PR #29677 (merged 2 days ago) added a guard to `can_handle()` rejecting cases where `q_head_dim != v_head_dim`. MLA is exactly that case, so `can_handle()` now returns `False` for K3 and `verify_mla_ …[truncated]

### L2-7738062294  (L2, 2026-08-10, sha 7738062294e2, PR #33974)
TITLE: [unified memory] Support DSPARK speculative decoding + fix two NaN root causes (page hand-out zeroing, CuTe int32 slot-stride wrap) (#33974)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+13/-6); python/sglang/kernels/ops/kimi_k3/kda_decode_mtp.py (+3/-0); python/sglang/kernels/ops/kvcache/zero_pages.py (+48/-0); python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py (+24/-9); python/sglang/srt/environ.py (+2/-0); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+6/-0); python/sglang/srt/mem_cache/kv_cache_configurator.py (+48/-0); python/sglang/srt/mem_cache/multi_ended_allocator.py (+20/-1); python/sglang/srt/mem_cache/unified_memory_pool.py (+22/-1); python/sglang/srt/server_args.py (+24/-3); (+3 more)
LABELS: blackwell, run-ci, jit-kernel, bypass-fastfail
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ `--enable-unified-memory` currently hard-blocks speculative decoding (`server_args.py`: `assert self.speculative_algorithm is None`). This PR wires chain DSPARK through the unified pool and fixes the two real NaN/corruption root causes found while validating it end to end on Kimi-K3 (TP8, 2x4 GB300) and Kimi-Linear-48B (TP1 x8 B300-class). Other speculative algorithms (EAGLE / tree / DFLASH / NGRAM) remain blocked pending their own …[truncated]

### L2-955aab8db1  (L2, 2026-08-10, sha 955aab8db103, PR #34213)
TITLE: [DCP] Reuse partial output in natural-log LSE merge (#34213)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/dcp/comm.py (+2/-2)
BODY: ## Motivation ⏎  ⏎ `cp_lse_ag_out_rs_mha` merges rank-local attention outputs and LSEs per query head. The previous expression allocated a second full FP32 `[tokens, heads, output_dim]` tensor even though the local partial output is not used after the merge. ⏎  ⏎ This merge is not specific to the model attention architecture: `output_dim` can be MHA `v_head_dim` or absorbed-MLA `kv_lora_rank`. In the Kimi-K3 TP8/DCP8 cached-extend reproduction, the absor …[truncated]

### L2-c80a38edcd  (L2, 2026-08-10, sha c80a38edcd2c, PR #34321)
TITLE: [Fix] Pin `cuda-tile` to 1.6.0rc5 to unblock Python 3.10 x86_64 installs (#34321)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-0)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎ - Pin `cuda-tile==1.6.0rc5` so resolution stops picking the incomplete `1.6.0rc6` prerelease ⏎  ⏎ ## Background ⏎ - `cuda-tile` is transitive: `sglang` -> `flashinfer_python[cu13]==0.6.15.post1` -> `cuda-tile>=1.4.0` ⏎ - On PyPI it is a `wheel_stub` placeholder sdist that fetches the real wheel from `pypi.nvidia.com` at build time ⏎ - `1.6.0rc6` was published without `cp310` and `cp313` linux x86_64 wheels, so the stub raises `RuntimeError: Didn …[truncated]

### L2-8a7c8a72d6  (L2, 2026-08-10, sha 8a7c8a72d64b, PR #33517)
TITLE: Fix NaN logits from deterministic Triton extend on the unified memory pool (#33517)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/triton_backend.py (+9/-0); test/registered/attention/test_unified_memory_deterministic.py (+118/-0)
LABELS: ready-to-merge
BODY: ## Motivation ⏎  ⏎ `--enable-unified-memory` + `--attention-backend triton` + `--enable-deterministic-inference` together produce `NaN` logits. Any two of the three are fine, which is why no existing test caught it. Found while adding a PD-disaggregation test in #33362; this is a pre-existing defect on `main`, unrelated to PD, and #33362 works around it by not setting those flags together. ⏎  ⏎ The failure is nastier than a crash: it only *aborts* when ` …[truncated]

### L2-ceeaec2078  (L2, 2026-08-10, sha ceeaec207887, PR #33362)
TITLE: [PD] Support --enable-unified-memory with PD disaggregation (kimi-linear MLA hybrid-Mamba) (#33362)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+23/-3); python/sglang/srt/disaggregation/decode.py (+29/-3); python/sglang/srt/disaggregation/mooncake/conn.py (+94/-24); python/sglang/srt/disaggregation/prefill.py (+30/-3); python/sglang/srt/disaggregation/utils.py (+44/-0); python/sglang/srt/managers/scheduler.py (+13/-1); python/sglang/srt/mem_cache/allocator/base.py (+10/-0); python/sglang/srt/mem_cache/kv_cache_configurator.py (+22/-5); python/sglang/srt/mem_cache/multi_ended_allocator.py (+36/-1); python/sglang/srt/mem_cache/unified_memory_pool.py (+61/-3); (+5 more)
LABELS: run-ci, bypass-fastfail
BODY: > **Stacked on #33517.** That PR fixes a pre-existing NaN in the deterministic Triton extend path on the unified memory pool, which this PR's test configuration would otherwise hit. Review/merge #33517 first; the diff shown here is only the PD-disaggregation change. ⏎  ⏎ ## Motivation ⏎  ⏎ `--enable-unified-memory` (the unified KV/mamba byte pool for hybrid-Mamba models, e.g. kimi-linear) was incompatible with PD disaggregation. The pool stores VIRTUAL i …[truncated]

### L2-7c7326ccb3  (L2, 2026-08-10, sha 7c7326ccb328, PR #31700)
TITLE: Fix DeepSeek-V4/DeepSeek-V4-Pro DP-attention gather semantics (#31700)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v4.py (+7/-2); python/sglang/srt/models/deepseek_v4_nextn.py (+7/-2)
LABELS: bug, deepseek, run-ci
ISSUES: #31699 [Bug]: DeepSeek-V4(-Pro) DP-attention (data-parallel-size>1) produces numerically-garbage output when moe_a2a_backend=none and attn_tp_size>1 — dp_gather_partial used where data is already replicated
BODY: ## Motivation ⏎  ⏎ Fixes #31699. ⏎  ⏎ DeepSeek-V4 DP-attention produces numerically invalid output when ⏎ `moe_a2a_backend=none`, `data_parallel_size>1`, and `attn_tp_size>1`. ⏎  ⏎ By the time the MoE gather runs, `self_attn` has already reduced its output ⏎ across the attention-TP group. The hidden states are therefore replicated ⏎ across attention-TP ranks. ⏎  ⏎ The existing `dp_gather_partial` calls treat those tensors as unreduced partial ⏎ contributions …[truncated]

### L2-0967885121  (L2, 2026-08-10, sha 0967885121ba, PR #34240)
TITLE: [DCP] Drop two per-layer launches from the MLA target-verify path (#34240)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.trtllm_mla, L2.backend.cutedsl_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/cutedsl_mla_backend.py (+2/-16); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+12/-8); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+1/-3); test/registered/dcp/test_dcp_layout_unit.py (+30/-0); test/registered/kernels/test_dcp_lse_combine.py (+5/-1)
LABELS: blackwell
BODY: ## Motivation ⏎  ⏎ Profiling Kimi-K3 @ 128k on 8xB300 (`--dcp-size 8` + DSpark, `cutedsl_mla`) shows that in every MLA layer of the captured target-verify graph, the window between the attention epilogue and the DCP all-to-all is mostly glue. Two of those kernels do no useful work at all: ⏎  ⏎ | kernel | µs/layer | what it is | ⏎ |---|---|---| ⏎ | `vectorized_elementwise<MulFunctor<float>>`, grid[1] | 1.76 | `lse * log2(e)` — rebase the cute-dsl kernel …[truncated]

### L2-77c90e7e54  (L2, 2026-08-10, sha 77c90e7e5493, PR #34283)
TITLE: Cookbook: add Ling-3.0-tiny (#34283)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/InclusionAI/Ling-3.0-tiny.mdx (+166/-0); docs/docs.json (+1/-0); docs/src/snippets/configs/inclusionAI/ling-3.0-tiny-benchmarks.jsx (+28/-0); docs/src/snippets/configs/inclusionAI/ling-3.0-tiny.jsx (+191/-0)
LABELS: documentation
BODY: ## Model ⏎  ⏎ [inclusionAI/Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny) — the compact variant of Ling-3.0-flash. A BailingMoeV3 hybrid-attention MoE (~7.9B total / ~1.2B active; KDA linear-attention interleaved with gated MLA, 128 experts). Open-sourced by inclusionAI. Adds a Cookbook page + deployment config, following the config-driven format (shared `_deployment.jsx` / `_playground.jsx` engines; this PR only adds model data + n …[truncated]

### L2-9d4be40124  (L2, 2026-08-10, sha 9d4be40124cc, PR #33865)
TITLE: Fix DSpark + DeepSeek V4 prefill CP compatibility (#33865)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/models/deepseek_v4.py (+10/-7); python/sglang/srt/models/deepseek_v4_dspark.py (+18/-9); python/sglang/srt/arg_groups/speculative_hook.py (+2/-1); python/sglang/srt/model_executor/runner/eager_runner.py (+17/-4); test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py (+44/-0)
LABELS: deepseek, speculative-decoding, run-ci
BODY: ## Summary ⏎ Two related fixes so DSpark speculative decoding works together with DeepSeek V4 prefill context parallelism (`--enable-nsa-prefill-context-parallel --nsa-prefill-cp-mode round-robin-split`) at PD co-located deployment. ⏎  ⏎ ## Changes ⏎ 1. **`deepseek_v4_dspark.py`: use lm_head's shard group for markov_w2 TP-shard** ⏎    `configure_tp_shard` hard-coded `attn_tp_group` for the per-step all-gather. Under prefill CP (`attn_cp_size == tp_siz …[truncated]

### L2-e74ea5b1d7  (L2, 2026-08-11, sha e74ea5b1d709, PR #31105)
TITLE: [ROCm/gfx95] Fix fp8 per-channel attention for Kimi-K2.7-code-mxfp4 o… (#31105)
SOURCES: path_core
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py (+4/-3); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha_rocm.py (+5/-7); python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py (+5/-0); python/sglang/srt/models/deepseek_common/utils.py (+17/-0); python/sglang/srt/models/deepseek_v2.py (+24/-5); test/registered/amd/accuracy/mi35x/test_kimi_k27_code_mxfp4_eval_mi35x.py (+184/-0); test/registered/amd/test_fp8_per_channel_detection.py (+75/-0)
LABELS: amd, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ Models with mixed quantization (mxfp4 MoE + fp8 per-channel attention, e.g. Kimi-K2.7-code-mxfp4) failed on gfx95 (MI355X) because the hardcoded `weight.dtype == float8_e4m3fn` checks in the forward pass unconditionally applied fused fp8 block-scale quantization kernels to all fp8 layers, including per-channel fp8 attention layers (QuarkW8A8Fp8) which cannot accept the resulting (fp8, group_scale[M, K/128]) tuple inputs. ⏎  ⏎ Root  …[truncated]

### L2-2c07ca5e8d  (L2, 2026-08-11, sha 2c07ca5e8da3, PR #33075)
TITLE: [Fix] Allow flashinfer_sparse_mla DSA backend for HiSparse on SM120 FP8 KV (#33075)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docs/docs/advanced_features/hisparse_guide.mdx (+2/-1); python/sglang/srt/arg_groups/hisparse_hook.py (+4/-5); test/registered/unit/server_args/test_server_args.py (+17/-0)
LABELS: documentation, run-ci
BODY: ## Motivation ⏎  ⏎ `--enable-hisparse` cannot start at all on SM120/SM121 (RTX PRO 6000 / RTX 5090) with FP8 KV cache, because two parts of the argument pipeline disagree about which DSA backend is legal. ⏎  ⏎ The resolution pass already selects `flashinfer_sparse_mla` for GLM DSA models on SM120 with `--kv-cache-dtype fp8_e4m3` — it is the only DSA kernel available on that architecture (`_dsa_split_backend_resolution`, `is_glm_sm12_fp8` arm in `pyth …[truncated]

### L2-c7c03ec53b  (L2, 2026-08-11, sha c7c03ec53b1e, PR #30700)
TITLE: [NVIDIA] Add flashinfer MNNVL backend for allreduce only (#30700)
SOURCES: release_notes
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/overrides.py (+1/-0); python/sglang/srt/distributed/bootstrap.py (+8/-0); python/sglang/srt/distributed/parallel_state.py (+85/-0); python/sglang/srt/layers/communicator.py (+12/-0); python/sglang/srt/layers/flashinfer_comm_fusion.py (+99/-0); test/registered/unit/layers/test_flashinfer_comm_fusion.py (+206/-6); test/registered/unit/layers/test_layer_communicator_fusion_gate.py (+65/-0)
LABELS: run-ci, release-highlight
BODY: ## Motivation ⏎  ⏎ SGLang already supports **fused** allreduce via FlashInfer (`kARResidualRMSNorm`: allreduce + residual + RMSNorm in one kernel) for DeepSeek-V3/V4 models. However, not every allreduce site can use the fused path — `RowParallelLinear`, attention output fallback, and MoE TP fallback all perform **pure (non-fused)** allreduce that currently always falls back to NCCL/custom-allreduce even when the FlashInfer mnnvl/trtllm workspace is …[truncated]

### L2-1ce515a53d  (L2, 2026-08-11, sha 1ce515a53def, PR #34464)
TITLE: Refocus LoRA tests on regression coverage (#34464)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/kernels/ops/gemm/test_chunked_sgmv_cuda_graph.py (+294/-0); test/registered/lora/test_embedding_lora_support.py (+125/-0); test/registered/lora/test_lora_eviction_policy.py (+0/-201); test/registered/lora/test_lora_openai_compatible.py (+32/-206); test/registered/unit/lora/test_eviction_policy.py (+83/-0); test/registered/unit/lora/test_lm_head_pruning.py (+87/-0); test/registered/unit/lora/test_lora_manager_tied_lm_head.py (+91/-0); test/registered/unit/managers/test_io_struct.py (+33/-0)
LABELS: lora, run-ci
BODY: ## Motivation ⏎  ⏎ Apply `.claude/rules/unit-test-admission.md` to the LoRA tests so each retained case names a concrete future regression. The broad cleanup in #34070 removed several valuable regressions together with redundant coverage; this restores only the focused contracts while pruning weak happy paths. ⏎  ⏎ ## Modifications ⏎  ⏎ > **Counting baseline:** the pictured LoRA suite immediately before #34070. Latest `main` had already deleted four of these …[truncated]

### L2-30cb848d4b  (L2, 2026-08-11, sha 30cb848d4b45, PR #34443)
TITLE: [Bugfix][DSA] Fix num_splits "(b+1)" crash on prefill-CP speculative decode (#34443)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/dsa/utils.py (+11/-5)
LABELS: run-ci
ISSUES: #30296 [Bug] GLM-5.2 EAGLE/MTP speculative fails with DSA attention TP (attn_tp_size>1)
BODY: ## Motivation ⏎  ⏎ Fixes #30296. Running a DSA model (GLM-5.x / DeepSeek-V3.2) with `--dsa-prefill-backend flashmla_kv --enable-prefill-cp --cp-strategy interleave` together with EAGLE/MTP speculative decoding crashes at warmup: ⏎  ⏎ ``` ⏎ RuntimeError: num_splits must have shape (b+1) ⏎ ``` ⏎  ⏎ This is a `TORCH_CHECK` inside `torch.ops.sgl_kernel.fwd_kvcache_mla`, which asserts `num_splits.shape[0] == q.shape[0] + 1`. ⏎  ⏎ ## Root cause ⏎  ⏎ `cal_padded_tokens` (`pyth …[truncated]

### L2-a3bd7d9401  (L2, 2026-08-12, sha a3bd7d94011d, PR #34372)
TITLE: Bump CuTeDSL to 4.6.2 (#34372)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); python/sglang/kernels/ops/attention/flash_attn/cute/interface.py (+0/-2); python/sglang/kernels/ops/attention/flash_attn/cute/pyproject.toml (+3/-3)
LABELS: high priority, dependencies, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight
BODY: ## Motivation ⏎  ⏎ This fixes an FA4 startup regression on Blackwell caused by the combination of `quack-kernels==0.6.3` and `nvidia-cutlass-dsl==4.6.0`. ⏎  ⏎ I hit this while starting `thinkingmachines/Inkling-Small-NVFP4` with `--attention-backend fa4` on SM103. During FlashInfer autotuning, the FA4 relative-bias kernel failed to compile: ⏎  ⏎ ```text ⏎ error[TYPE_UNSTABLE_JOIN]: `tBrS_cur` has type `None` on one path and `_Tensor` on another ⏎   --> f …[truncated]

### L2-00e57d74f0  (L2, 2026-08-12, sha 00e57d74f07b, PR #33997)
TITLE: Bump FlashInfer to 0.6.17 and remove Kimi K3 workarounds (#33997)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.dispatch.server_args_defaults, L2.backend.flashinfer_general_mla
FILES: docker/Dockerfile (+3/-49); docker/kimi_k3/flashinfer-perkz-dcp-0.6.15.txt (+0/-5639); docker/kimi_k3/kimi_k3_cu12.Dockerfile (+10/-47); docker/kimi_k3/kimi_k3_cu13.Dockerfile (+10/-47); python/pyproject.toml (+1/-1); docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx (+2/-11); docs/src/snippets/_playground.jsx (+1/-1); docs/src/snippets/configs/moonshotai/kimi-k3.jsx (+3/-7); python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh (+1/-1); python/sglang/kernels/ops/moe/trtllm_gen_moe.py (+0/-528); (+9 more)
LABELS: documentation, high priority, dependencies, run-ci, jit-kernel, release-highlight
BODY: ## Summary ⏎  ⏎ Bump FlashInfer to 0.6.17 and remove Kimi K3 workarounds ⏎  ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #31570458219](https://github.com/sgl-project/sglang/actions/runs/31570458219) ⏎ Latest PR Test (Extra): :x: [Run #31570457937](https://github.com/sgl-project/sglang/actions/runs/31570457937)

### L2-4f883636a2  (L2, 2026-08-12, sha 4f883636a2b3, PR #27689)
TITLE: [Perf] FlashInfer MLA: remove blocking D2H in spec-decode plan (#27689)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+127/-5)
LABELS: performance, deepseek, run-ci
DEEP_STUDY: deep-study performance PR ()
BODY: ## Motivation ⏎  ⏎ On H200, DeepSeek-R1 with MTP/EAGLE (`--speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3`) was **no faster than standard decoding**, which shouldn't happen. nsys showed 3 synchronous device→host copies per attention `plan()` in the target-verify and draft-extend steps. Being blocking, they stall the host thread that launches the next CUDA graph, breaking overlap scheduling — and the stalls negate  …[truncated]

### L2-6c6294b7be  (L2, 2026-08-12, sha 6c6294b7be23, PR #34614)
TITLE: [DCP] Fuse the a2a pack/unpack copies in the MLA LSE reduce (#34614)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/__init__.py (+2/-0); python/sglang/kernels/ops/attention/dcp_kernels.py (+81/-0); python/sglang/srt/layers/dcp/comm.py (+10/-46); test/registered/kernels/test_dcp_lse_combine.py (+41/-0)
LABELS: jit-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Follow-up to #34240. That PR removed two no-op launches from the DCP MLA verify path; profiling the same window showed the rest of it is the a2a buffer plumbing — four elementwise copies per MLA layer, per decode step, all on the critical path between the attention epilogue and the NCCL all-to-all: ⏎  ⏎ | kernel | what it moves | ⏎ |---|---| ⏎ | `direct_copy` | `reshaped_lse.contiguous()` — permute-contiguous of the fp32 LSE | ⏎ | `di …[truncated]

### L2-2d76d537e5  (L2, 2026-08-12, sha 2d76d537e551, PR #33945)
TITLE: feat: support deterministic FA4 for GLM-4.7-Flash (#33945)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docs/docs/advanced_features/server_arguments.mdx (+2/-2); python/sglang/srt/arg_groups/overrides.py (+1/-0); python/sglang/srt/layers/quantization/unquant.py (+8/-1); python/sglang/srt/server_args.py (+4/-3); python/sglang/test/test_deterministic_utils.py (+3/-1); test/registered/attention/test_glm4_moe_lite_deterministic.py (+68/-0)
LABELS: documentation, quant, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ Enable deterministic GLM-4.7-Flash inference with the FA4 attention backend on Blackwell. ⏎  ⏎ ## Modifications ⏎  ⏎ - Treat `Glm4MoeLiteForCausalLM` as an absorbed-MLA model for deterministic backend validation. ⏎ - Select the `torch` BF16 GEMM backend in deterministic mode and reject explicit `cutedsl`, whose kernel selection is batch-size dependent. ⏎ - Document the deterministic BF16 GEMM behavior. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Run on on …[truncated]

### L2-197832bcf5  (L2, 2026-08-12, sha 197832bcf536, PR #33465)
TITLE: [Kimi-K3][NPU]  Support Kimi-K3 on NPU (#33465)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.backend.npu_mla, L2.pool.mla_token_kv, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/kernels/ops/kimi_k3/mla_output_gate.py (+4/-1); python/sglang/srt/layers/attention/attention_registry.py (+10/-1); python/sglang/kernels/ops/elementwise/add3.py (+3/-1); python/sglang/kernels/ops/kimi_k3/__init__.py (+11/-6); python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py (+2/-1); python/sglang/multimodal_gen/runtime/layers/attention/backends/ascend_fa.py (+6/-0); python/sglang/srt/arg_groups/speculative_hook.py (+8/-4); python/sglang/srt/environ.py (+10/-0); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+46/-4); python/sglang/srt/hardware_backend/npu/attention/ascend_kda_backend.py (+634/-0); (+14 more)
LABELS: documentation, quant, dependencies, Multi-modal, deepseek, speculative-decoding, blackwell, npu, run-ci, diffusion
BODY: ## Summary ⏎  ⏎ Add Kimi-K3 support for Ascend NPU on top of upstream SGLang main and the GPU model integration from #32541. ⏎  ⏎ This PR keeps shared/GPU behavior intact where possible and dispatches NPU-specific implementations through the Ascend backend. The extracted Ascend Triton kernels are proposed separately in sgl-project/sgl-kernel-npu#658. ⏎  ⏎ ## Main changes ⏎  ⏎ - Full 93-layer Kimi-K3 execution on Ascend with TP64/DP4. ⏎ - DeepEP and Ascend …[truncated]

### L2-b501311fa1  (L2, 2026-08-13, sha b501311fa147, PR #33623)
TITLE: [Kimi K3] Fuse MLA gate projection into QKV-A GEMM (#33623)
SOURCES: subject_keyword, release_notes, corpus:confirmed-reverts(reverted), corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/gemm/dsv3_fused_a_gemm.cuh (+1/-0); python/sglang/srt/models/kimi_k3.py (+62/-4); test/registered/unit/models/test_kimi_k3_mla_gate_fusion.py (+118/-0)
LABELS: run-ci, jit-kernel, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study: this PR was reverted by PR 34642 (confirmed_revert, reason=ci_or_test_failure) || deep-study performance PR (new_kernel_or_fusion)
BODY: Replaces #33521, which was automatically closed when its base branch was deleted. Rebasing to main as requested in the review. ⏎  ⏎ ## Motivation ⏎  ⏎ Kimi-K3 MLA computes QKV-A and the TP-local output gate from the same hidden states. They currently run as separate GEMMs. Fuse them to reduce projection cost while keeping the gate output TP-local. ⏎  ⏎ ## Modifications ⏎  ⏎ - Merge unquantized BF16/FP16 QKV-A and g_proj weights after loading. ⏎ - Run one merged pr …[truncated]

### L2-aefe2d0207  (L2, 2026-08-13, sha aefe2d0207f0, PR #34642)
TITLE: Revert "[Kimi K3] Fuse MLA gate projection into QKV-A GEMM" (#34642)
SOURCES: subject_keyword, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/gemm/dsv3_fused_a_gemm.cuh (+0/-1); python/sglang/srt/models/kimi_k3.py (+4/-62); test/registered/unit/models/test_kimi_k3_mla_gate_fusion.py (+0/-118)
LABELS: jit-kernel
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 33623 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#33623 ⏎  ⏎ Breaks CI https://github.com/sgl-project/sglang/actions/runs/31650215286/job/94292793773#step:15:2133 and seems like a real regression given full 2048 tokens ⏎  ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #31653682855](https://github.com/sgl-project/sglang/actions/runs/31653682855) ⏎ Latest PR Test (Extra): :x: [Run #31653682699](https://github.com/sgl-project/sglang/actions/runs/31653682699)

### L2-652a2709d1  (L2, 2026-08-13, sha 652a2709d180, PR #34654)
TITLE: [Docs] Add decode context parallelism to advanced features (#34654)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/docs.json (+5/-0); docs/docs/advanced_features/dcp.mdx (+180/-0); docs/docs/advanced_features/overview.mdx (+1/-0); docs/docs/advanced_features/server_arguments.mdx (+19/-0)
LABELS: documentation
BODY: ## Summary ⏎ - Add a Decode Context Parallelism page under Advanced Features covering MLA KV striping, LSE merge, communication backends, and compositions with DPA, DSpark, PD, and HiCache L2. ⏎ - Register the page in the docs sidebar and overview, and document `--dcp-size`, `--dcp-comm-backend`, and `--dcp-replicate-q-proj` in server arguments. ⏎  ⏎ ## Test plan ⏎  ⏎ Made with [Cursor](https://cursor.com) ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_ch …[truncated]

### L2-81fe452810  (L2, 2026-08-13, sha 81fe452810d9, PR #34651)
TITLE: [DCP] Share one pack kernel between both a2a backends (#34651)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dcp_kernels.py (+43/-26); python/sglang/srt/layers/dcp/comm.py (+22/-12); test/registered/kernels/test_dcp_lse_combine.py (+41/-1)
LABELS: jit-kernel
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ #34614 fused the pack/unpack copies on the pynccl `a2a` path but left `fi_a2a` (FlashInfer MNNVL) untouched, so it still paid four materializing copies plus a zero-fill per MLA layer per decode step: ⏎  ⏎ ```python ⏎ partial_o = out.view(B, N, H_pr, D).permute(0, 2, 1, 3).contiguous()   # copy ⏎ softmax_stats = torch.zeros(B, H_pr, N, 2, ...)                        # FillFunctor ⏎ softmax_stats[..., 0] = lse_view                       …[truncated]

### L2-a34f81251f  (L2, 2026-08-13, sha a34f81251f78, PR #30762)
TITLE: fix(hicache/umbp): support DeepSeek-V4 hybrid HostPoolGroup (multi-po… (#30762)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/mem_cache/memory_pool_host.py (+4/-0); python/sglang/srt/mem_cache/storage/umbp/umbp_store.py (+279/-17); test/registered/hicache/test_hicache_storage_umbp_backend.py (+214/-0); test/registered/unit/mem_cache/test_hicache_staged_write_back_dispatch.py (+16/-0); test/registered/unit/mem_cache/test_umbp_store.py (+192/-13); test/run_suite.py (+1/-0)
LABELS: amd, hicache, run-ci
BODY: …ol v2) ⏎  ⏎ UMBPStore assumed mem_pool_host is a single KV-bearing pool. For the DeepSeek-V4 HiCache stack, mem_pool_host is a HostPoolGroup whose KV anchor is a LogicalHostPool that owns only page indices and holds no physical KV tensor (get_page_buffer_meta() returns None by design). The real KV state lives in page_first side pools (SWA / compressed KV / indexer / state), which the controller registers via register_mem_host_pool_v2() and drives  …[truncated]

### L2-ba1d980b35  (L2, 2026-08-13, sha ba1d980b353b, PR #31856)
TITLE: [AMD] Accelerate AITER unified-attention decode with scaled FP8 Q (#31856)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+11/-2); test/registered/attention/test_aiter_fp8_q_unified_attention.py (+304/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ### Motivation ⏎  ⏎ AITER unified attention already supports FP8 Q + FP8 KV. With BF16 Q, the FP8 KV path cannot select the native FP8-Q matrix-multiply path. Quantizing Q adds one graph node per full-attention layer, but can reduce the much larger `kernel_unified_attention_3d` cost at medium and high decode concurrency. ⏎  ⏎ Qwen3.5-397B-A17B-MXFP4 has 15 full-attention layers in one complete decode replay. The added Q-quant cost is therefore nearly …[truncated]

### L2-22dde1dd5b  (L2, 2026-08-14, sha 22dde1dd5b56, PR #31554)
TITLE: [Docs] Fill GLM-5.2 H200 FP8 speed cells (low-latency, balanced); fix MTP notation (#31554)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/GLM/GLM-5.2.mdx (+1/-1); docs/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx (+35/-4)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ The three H200 + FP8 speed cells for GLM-5.2 have been "pending re-measurement" since the acceptance-length-pinned methodology landed. We measured the low-latency and balanced cells on 8xH200 following that methodology, and while auditing the recipes found an MTP notation mismatch in the docs. ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Fill all three pending **H200 + FP8** speed cells in `glm-5.2-benchmarks.jsx`: low-latency (c=1/c=16), balanced (c=64/ …[truncated]

### L2-1af761a09a  (L2, 2026-08-14, sha 1af761a09a5a, PR #34019)
TITLE: [SM12x] Default the fused MHC post+pre path on (#34019)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+2/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ On sm120/sm121 the SM120 block in `server_args.py` sets `SGLANG_OPT_USE_TILELANG_MHC_PRE=False`, so DeepSeek-V4's `hc_pre` falls through to `hc_pre_torch_impl` — an fp32 `F.linear` of shape `[M, 16384] × [16384, 24]`. cuBLAS serves that with `cutlass_80_simt_sgemm`, i.e. **plain CUDA cores, no tensor cores at all**. ⏎  ⏎ On 2× DGX Spark (GB10 / sm_121, TP=2) running `deepseek-ai/DeepSeek-V4-Flash-0731` + DSPARK that fallback costs **93 …[truncated]

### L2-6cbfa791d6  (L2, 2026-08-15, sha 6cbfa791d699, PR #34517)
TITLE: [AMD][Spec] Accelerate Qwen3.5 verification with grouped-head shared KV (#34517)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/verify_mla.py (+41/-27); python/sglang/srt/configs/model_config.py (+9/-0); python/sglang/srt/layers/attention/triton_backend.py (+29/-14); test/registered/attention/test_verify_shared_kv.py (+235/-0)
LABELS: run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ Qwen3.5 uses grouped-query attention, where multiple query heads share a KV head. For Qwen3.5-397B under TP2, each rank has 16 query heads sharing one TP-local KV head. ⏎  ⏎ During EAGLE target verification, the existing split-KV path processes query heads independently and repeatedly scans the same prefix KV cache. This becomes increasingly expensive at high concurrency, when verification is memory-bandwidth-bound. ⏎  ⏎ PR [#33981]( …[truncated]

### L2-ab810e4052  (L2, 2026-08-15, sha ab810e40524c, PR #)
TITLE: config: each runner carries its own linear-attn kernel choice
SOURCES: path_core
ARTIFACT_HINTS: L2.dispatch.attention_registry
FILES: python/sglang/srt/layers/attention/attention_registry.py
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L2-d13d5c03ab  (L2, 2026-08-15, sha d13d5c03abea, PR #)
TITLE: config: decisions keyed on the attention backend read the configured pair
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/trtllm_mha_backend.py; python/sglang/srt/model_executor/model_runner_components/misc_utils.py; python/sglang/srt/speculative/draft_utils.py
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L2-4d0c5a89af  (L2, 2026-08-15, sha 4d0c5a89af70, PR #34837)
TITLE: [AMD] Add concat_and_cast_mha_k_pad_kernel to support 12-head and enable K3 aiter prefill kernel (#34837)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/kvcache/cache_ops.py (+72/-0)
LABELS: amd, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: With this PR, K3 can use the AITER prefill kernel, but must disable FP8 kernel, which does not support 12 heads. ⏎ Two changes to the server cmd (full cmd in the description above) ⏎ - `export SGLANG_AITER_FP8_PREFILL_ATTN=0` ⏎ - `--prefill-attention-backend aiter` ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ The triton `extend_attention_fwd` is much slower than the aiter kernel for MLA prefill. ⏎ - Triton kernel: `extend_attention_fwd()` in `python/sglang/kernels/ops/at …[truncated]

### L2-d22c4cc177  (L2, 2026-08-15, sha d22c4cc177c4, PR #30024)
TITLE: [AMD] perf(sgl-kernel): default block_quota=16 for MLA page_first KV gather… (#30024)
SOURCES: subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu (+125/-3); python/sglang/kernels/aot/python/sgl_kernel/kvcacheio.py (+25/-3)
LABELS: amd, hicache, sgl-kernel, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ROCm HiCache L2-L1 (host-device) load-back perf work: gets the JIT staged write-back kernel working on ROCm, widens the MLA gather copy to 128-bit, and fixes block_quota to honor SGLANG_HICACHE_BLOCK_QUOTA at runtime. ⏎  ⏎ - Widen MLA gather to 128-bit. 128-bit streaming load/store when aligned. +32% at block_quota=8, +36% for D2D on MI355X. ⏎ - Fix block_quota env override. Was resolved once at import, so the env var had no effect at runtime. Now r …[truncated]

### L2-4c0e85524d  (L2, 2026-08-15, sha 4c0e85524dc1, PR #30808)
TITLE: [AMD] [GLM5] Enable dense-MHA short-context prefill fallback on gfx950 (#30808)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/dsa_backend.py (+8/-4); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py (+16/-0)
LABELS: documentation, amd, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ On gfx950 (MI355X), GLM-5.2 DSA prefill always ran the triton sparse-MLA path, even at short context where the sparse indexer top-k + gather + mask overhead exceeds the KV it prunes. The dense-MHA prefill fallback — already used on NVIDIA SM90/SM100 and gated by `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` (default = model `index_topk`) — was hard-gated to NVIDIA and never taken on ROCm, despite the dense kernel (aiter `flash …[truncated]

### L2-a508d60295  (L2, 2026-08-16, sha a508d60295a9, PR #34245)
TITLE: [BCG][6/N] Allow prefill breakable CUDA graph for the Kimi archs (#34245)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/configs/model_config.py (+6/-0); python/sglang/srt/server_args.py (+5/-4); test/registered/unit/configs/test_multimodal_piecewise_cuda_graph.py (+2/-1)
LABELS: Multi-modal, run-ci, bypass-fastfail, run-ci-extra
BODY: Kimi configs always construct a vision_config (KimiK3Config defaults it to KimiK3VisionConfig() when absent), so ModelConfig.is_multimodal is True via the has_multimodal_subconfig sniff even when serving text only -- neither the arch name nor multimodal_model_archs is involved. The generic "multimodal model" rule in _disable_breakable_cudagraph_if_incompatible then forced prefill.backend to DISABLED on the default path. ⏎  ⏎ This only ever showed u …[truncated]

### L2-f7cb328eb7  (L2, 2026-08-16, sha f7cb328eb749, PR #31324)
TITLE: [AMD] [GLM5] Skip DSA decode indexer when kv_len <= index_topk (dense k-only fast path) (#31324)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/dsa/dsa_indexer.py (+67/-6); python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py (+91/-10); python/sglang/srt/model_executor/runner/shape_key.py (+4/-0); python/sglang/srt/model_executor/runner_utils/capture_mode.py (+18/-0)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎ - On GLM-5.2 DSA decode, when a request's `kv_len <= index_topk` the top-k selects **all** valid positions, so the indexer's logits GEMM + `paged_mqa_logits` + top-k selection is wasted work. Add a **k-only** fast path that skips the indexer, stores the K cache, and generates the identity index directly (`[0, 1, ..., kv_len-1, -1, ...]`), feeding the same sparse-MLA decode attention kernel. ⏎ - **CUDA-graph "Design A" dual-graph:** c …[truncated]

### L2-0fb040cbeb  (L2, 2026-08-16, sha 0fb040cbeb8c, PR #34889)
TITLE: [DCP]Localize HiCache DCP indices once per transfer, not per layer (#34889)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/mem_cache/pool_host/base.py (+5/-8); python/sglang/srt/mem_cache/pool_host/mla.py (+4/-4); test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py (+5/-5)
LABELS: hicache
BODY: ## Motivation ⏎  ⏎ Under DCP, `MLATokenToKVPoolHost.load_to_device_per_layer()` translated the widened logical slots into this rank's physical rows on every call: ⏎  ⏎ ```python ⏎ host_indices = self.dcp_kernel_indices(host_indices) ⏎ device_indices = self.dcp_kernel_indices(device_indices) ⏎ ``` ⏎  ⏎ The result is layer-independent, but the load loop calls the method once per layer. ⏎  ⏎ The cost is worse than the launch count suggests. `indices[indices %  …[truncated]

### L2-eb61cb2823  (L2, 2026-08-16, sha eb61cb28233b, PR #33480)
TITLE: [AMD] Support prefill context parallel two batch overlap for DeepSeek V4 (#33480)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/batch_overlap/operations_strategy.py (+24/-4); python/sglang/srt/batch_overlap/two_batch_overlap.py (+7/-0); python/sglang/srt/distributed/bootstrap.py (+6/-0); python/sglang/srt/distributed/parallel_state.py (+66/-0); python/sglang/srt/layers/attention/dsv4/compressor.py (+36/-0); python/sglang/srt/layers/dp_attention.py (+9/-0); python/sglang/srt/layers/utils/cp_utils.py (+50/-0); python/sglang/srt/models/deepseek_v4.py (+235/-15); python/sglang/srt/server_args.py (+6/-0); test/registered/amd/test_deepseek_v4_pro_fp4_cp_tbo.py (+155/-0)
LABELS: deepseek, run-ci, bypass-fastfail
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎ DeepSeek V4 prefill context parallelism (CP) distributes long-context prefill across attention CP ranks, but introduces several per-layer CP collectives around attention and MoE. Previously, the DeepSeek V4 two-batch-overlap (TBO) path only supported non-CP DP/EP workflows: ⏎ - `--enable-prefill-cp` with `--enable-two-batch-overlap` was rejected when DP attention was disabled. ⏎ - CP batches were explicitly excluded by the runtime TB …[truncated]

### L2-92bce3d7bb  (L2, 2026-08-17, sha 92bce3d7bb53, PR #30519)
TITLE: [AMD] [GLM5] fp8 MLA absorbed bmm for GLM-5.2 on gfx950 (#30519)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py (+18/-11); python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py (+11/-0)
LABELS: amd, deepseek, run-ci
BODY: ## Summary ⏎  ⏎ * On gfx950, GLM (`GlmMoeDsaForCausalLM`) MLA absorbed weights (`w_kc`/`w_vc`) load as bf16, so `forward_mla` falls to the slow per-batched `torch.bmm` (rocBLAS) path. Quantize them to per-tensor `fp8_e4m3fn` at load (mirroring the DeepSeek fp8 flow) so the q_nope / attn-output absorbed projections route through the fused aiter `batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant` kernel. ⏎ * Write the absorbed v_ …[truncated]

### L2-198a7b2fc9  (L2, 2026-08-17, sha 198a7b2fc990, PR #35062)
TITLE: [Misc] Clean up python/sglang package structure (#35062)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/kernels/ops/attention/sparse_mla_q8kv8_prefill_sm90.py (+1/-1); python/sglang/README.md (+24/-16); python/sglang/__init__.py (+25/-44); python/sglang/_platform_stubs.py (+219/-9); python/sglang/_triton_stub.py (+0/-228); python/sglang/eval/llama3_eval.py (+0/-315); python/sglang/eval/loogle_eval.py (+0/-164); python/sglang/kernels/aot/python/sgl_kernel/debug_utils.py (+1/-1); python/sglang/kernels/fused_op.py (+1/-1); python/sglang/kernels/kernel_api_logging.py (+0/-0); (+44 more)
LABELS: documentation, quant, hicache, sgl-kernel, run-ci, diffusion, jit-kernel
BODY: ## Motivation ⏎  ⏎ Clean up the top-level python/sglang package organization and remove stale modules. ⏎  ⏎ ## Modifications ⏎  ⏎ - Remove the obsolete eval package. ⏎ - Move frontend global configuration under sglang.lang and update internal imports. ⏎ - Move kernel API logging under sglang.kernels and update all callers. ⏎ - Simplify package initialization and move plugin loading to the top-level imports. ⏎ - Rewrite python/sglang/README.md to document the current …[truncated]

### L2-b42abbb1ba  (L2, 2026-08-17, sha b42abbb1ba08, PR #34316)
TITLE: [metrics] Fix prefill FLOPs estimate to count prefix and per-request causal pairs (#34316)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/scheduler_components/metrics_reporter.py (+15/-4); test/registered/unit/observability/test_forward_pass_metrics.py (+47/-0)
ISSUES: #34298 [Bug] Prefill FLOPs estimate ignores `prefix_lens`, so `est. prefill TFLOPS/s` degenerates into 1/latency across chunked-prefill chunks
BODY: ## Motivation ⏎  ⏎ Fixes #34298. ⏎  ⏎ `SchedulerMetricsReporter._estimate_prefill_perf` computes the causal attention pair count using only `sum(extend_lens)`. In other words, when a chunk of `c` tokens is resumed at prefix `P`, the estimator scores it as if it were a brand-new sequence of length `c`. This produces two independent errors. ⏎  ⏎ 1. Pairs against the cached prefix are dropped ⏎ The `c * P` attention pairs between the chunk and its cached p …[truncated]

### L2-fcdaaf8a5d  (L2, 2026-08-17, sha fcdaaf8a5d2c, PR #32313)
TITLE: [Feature] Optimize TP LMHead with All-to-All (#32313)
SOURCES: release_notes
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/overrides.py (+41/-1); python/sglang/srt/distributed/bootstrap.py (+53/-0); python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py (+4/-2); python/sglang/srt/layers/logits_processor.py (+80/-3); python/sglang/srt/server_args.py (+23/-4); test/registered/mock_model/test_e2e_tp.py (+24/-1); test/registered/unit/test_model_overrides.py (+1/-0)
LABELS: run-ci, release-highlight
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ When the model is deployed with pure DP and dp-attention, **--enable-dp-lm-head will suffer from poor gemm efficiency when batchsize of each rank is small**. For example, in DeepseekV4 Pro, we use dp=8 and --enable-dp-lm-head. In such case, the lmhead gemm itself costs **320us** when bs=36 on each dp rank. ⏎ <img width="557" height="122" alt="image" src="https://github.com/user-attachments/assets/c63cd93d-b672-4834-92d4-b046decfd6 …[truncated]
