### L1-3239baef25  (L1, 2026-09-03, sha 3239baef255d, PR #37760)
TITLE: [CI][NPU] Fix kimi_k2_6 16p in64k perf test and dsv4-flash testcases (#37760)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/test/ascend/e2e/run_npu_testcase.sh (+3/-0); test/registered/npu/accuracy/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_gpqa.py (+4/-1); test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_1p1d_16p_in8k_out1k_50ms.py (+12/-13); test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms.py (+12/-23); test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms.py (+12/-24); test/registered/npu/performance/kimi_k2_6/test_npu_kimi_k2_6_w4a8_16p_in64k_out1k_100ms.py (+1/-1)
LABELS: deepseek, npu
BODY: ## Motivation ⏎  ⏎ 1. The kimi_k2_6 W4A8 16p in64k/out1k performance test uses `--chunked-prefill-size 262144`. With the full 64K-token input prefilled in one chunk, peak activation memory is high and the server is prone to recomputation/oom-adjacent behavior on A3 devices, which destabilizes the nightly latency measurement. Reducing it to `32768` keeps prefill stable and restores reproducible end-to-end latency. ⏎ 2. The DSV4-Flash perf/accuracy te …[truncated]

### L1-4e37882a93  (L1, 2026-09-03, sha 4e37882a9374, PR #37332)
TITLE: [Diffusion][minimax-h3] Add SM120 support for SubBlock sparse attention (#37332)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py (+16/-0); python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/README.md (+19/-13); python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/__init__.py (+13/-5); python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py (+42/-12); python/sglang/multimodal_gen/runtime/platforms/cuda.py (+20/-9); python/sglang/multimodal_gen/test/unit/test_subblock_sparse_attention.py (+16/-3); test/registered/cpu/test_subblock_sparse_attention.py (+70/-1)
LABELS: documentation, run-ci, diffusion
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ SubBlock sparse attention provides architecture-specific paths for Hopper ⏎ (SM90) and datacenter Blackwell (SM100). An SM120 path for Blackwell devices ⏎ with compute capability 12.0 is currently missing. ⏎  ⏎ This PR adds SM120 support through FlashInfer's SM120 blk64 operator. ⏎  ⏎ In a fixed 8x RTX PRO 5000 MiniMax-H3 request benchmark, pure SubBlock 0.75 ⏎ reduced measured inference latency by 18.23%-45.89% (1.223x-1.848x) versus ⏎  …[truncated]

### L1-b42569a0f1  (L1, 2026-09-04, sha b42569a0f135, PR #31755)
TITLE: [Quant][ue8m0 fix] group requant_weight_ue8m0 reduce reserved gpu memory (#31755)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8_utils.py (+48/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ **How discover this issue?** ⏎ In the colocate training scenario, we noticed that after SGLang called `release_memory_occupation` for `kvcache`, the decrease in GPU memory was minimal. Even after releasing `kvcache` on B200, there was still an occupation of approximately ~141G (with MTP), which was clearly not expected ⏎  ⏎ **What is the reason?**（test in `sgl-project/DeepSeek-V4-Flash-FP8` 4card B200/GB200） ⏎ `requant_weight_ue8m0`  …[truncated]

### L1-72078cd7f5  (L1, 2026-09-04, sha 72078cd7f516, PR #35751)
TITLE: [XPU] Support GPT-OSS MXFP4 checkpoints on Intel XPU (#35751)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+8/-0); docs/docs/advanced_features/quantization.mdx (+1/-1); docs/docs/hardware-platforms/xpu.mdx (+37/-0); python/sglang/srt/layers/quantization/__init__.py (+5/-1); python/sglang/srt/layers/quantization/mxfp4.py (+64/-0)
LABELS: documentation, quant, intel, xpu, run-ci
BODY: ## Enable GPT-OSS MXFP4 checkpoints on Intel XPU via the native W4A16 grouped GEMM ⏎  ⏎ ### Motivation ⏎  ⏎ `openai/gpt-oss-20b` and `-120b` ship MXFP4-quantized MoE weights. Every SGLang MXFP4 kernel targets CUDA (`triton_kernels` / Marlin / FlashInfer cutlass), HIP (aiter) or CPU (AMX), so on `--device xpu` the checkpoint was rejected before any layer was built: ⏎  ⏎ ``` ⏎ ValueError: Unknown quantization method: mxfp4. Must be one of ['fp8', 'mxfp8', ...] ⏎   …[truncated]

### L1-55bf3380e0  (L1, 2026-09-04, sha 55bf3380e073, PR #36805)
TITLE: Support Hy4-preview (#36805)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.runner.deep_gemm, L1.ep.layer, L1.cutlass.adapters
FILES: python/sglang/kernels/ops/moe/ep_moe_kernels.py (+4/-1); python/sglang/kernels/ops/moe/triton_pad_expert_counts.py (+41/-0); python/sglang/kernels/ops/moe/triton_sigmoid_gate_mul.py (+2/-0); python/sglang/srt/layers/moe/cutlass_moe.py (+7/-1); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+112/-35); python/sglang/kernels/ops/layernorm/__init__.py (+2/-0); python/sglang/kernels/ops/layernorm/hy4_ihc.py (+443/-0); python/sglang/srt/arg_groups/model_hook.py (+27/-13); python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py (+44/-2); python/sglang/srt/arg_groups/overrides.py (+32/-1); (+37 more)
LABELS: high priority, quant, deepseek, speculative-decoding, run-ci, jit-kernel, run-ci-extra, release-highlight
BODY: ## Summary ⏎  ⏎ Adds support for the `hy_v4` architecture (Hy4-preview, text-only): MLA with DSA sparse attention, iHC, gated MLA, learned attention sinks, sigmoid-gated MoE, and MTP/NextN speculative drafting, together with MXFP8 quantization support. ⏎  ⏎ ## What is included ⏎  ⏎ ## Validation ⏎  ⏎ ## Limitations ⏎  ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #33934945618](https://github.com/sgl-project/sglang/actions/runs/339349456 …[truncated]

### L1-92a4d8b5ee  (L1, 2026-09-04, sha 92a4d8b5eee0, PR #33930)
TITLE: Clean logging under --weight-loader-prefetch-checkpoints (#33930)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-6); docs/cookbook/autoregressive/GLM/GLM-5.mdx (+0/-4); python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py (+1/-4); python/sglang/srt/layers/cp/cp_decode_attn_tp.py (+0/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-4); python/sglang/srt/layers/quantization/petit.py (+1/-4); python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py (+5/-3); python/sglang/srt/model_executor/model_runner_components/load_model_utils.py (+1/-2); python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py (+7/-1); python/sglang/srt/model_loader/loader.py (+1/-1); (+9 more)
LABELS: documentation, quant, Multi-modal, deepseek, run-ci, diffusion, bypass-fastfail
BODY: Delete logs that add noise. ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #33854447904](https://github.com/sgl-project/sglang/actions/runs/33854447904) ⏎ Latest PR Test (Extra): :x: [Run #33854447858](https://github.com/sgl-project/sglang/actions/runs/33854447858) ⏎ Latest PR Test (AMD ROCm 7.2): :x: [Run #33854448326](https://github.com/sgl-project/sglang/actions/runs/33854448326)

### L1-1e6f18bfeb  (L1, 2026-09-05, sha 1e6f18bfeb33, PR #32405)
TITLE: [MoE Refactor] Migrate SM100 trtllm-gen mxfp4 MoE onto MoeRunner (#32405)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm, L1.runner.flashinfer_cutlass
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py (+33/-7); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+236/-0); python/sglang/srt/layers/quantization/mxfp4.py (+52/-250); test/registered/unit/layers/quantization/test_mxfp4_situ_output.py (+51/-38); test/registered/unit/layers/quantization/test_mxfp4_sm100_trtllm_gen.py (+335/-0); test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py (+2/-8)
LABELS: run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ Part of the MoE refactor tracked in #8715.  ⏎  ⏎ cc @ch-wan  ⏎  ⏎ `Mxfp4MoEMethod` was the last quant method with a partially migrated path: the SM90 CUTLASS branch went through `MoeRunner` (#26489), but the SM100 trtllm-gen branch still called the kernel inline from `apply`, with `create_moe_runner` falling through to `pass` and a `TODO`. This migrates it, so both FlashInfer MXFP4 GPU paths in `Mxfp4MoEMethod` now use the shared run …[truncated]

### L1-97c6978369  (L1, 2026-09-06, sha 97c6978369ac, PR #36507)
TITLE: GLM-5.3-Flash support (#36507)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.runner.deep_gemm
FILES: python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+7/-6); python/sglang/kernels/ops/attention/dsa/quant_k_cache.py (+69/-5); python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py (+33/-21); python/sglang/kernels/ops/attention/dsa_metadata.py (+10/-6); python/sglang/kernels/ops/attention/fla/kda.py (+4/-5); python/sglang/srt/arg_groups/attention_hook.py (+1/-0); python/sglang/srt/arg_groups/cuda_graph_hook.py (+8/-1); python/sglang/srt/arg_groups/model_hook.py (+1/-0); python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py (+2/-1); python/sglang/srt/arg_groups/overrides.py (+3/-0); (+93 more)
LABELS: quant, amd, Multi-modal, deepseek, hicache, npu, run-ci, apple-silicon, jit-kernel, bypass-fastfail
BODY: Support GLM-5.3-Flash. ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #34024595258](https://github.com/sgl-project/sglang/actions/runs/34024595258) ⏎ Latest PR Test (Extra): :no_entry_sign: [Run #34024595202](https://github.com/sgl-project/sglang/actions/runs/34024595202) ⏎ Latest PR Test (AMD ROCm 7.2): :no_entry_sign: [Run #34024595305](https://github.com/sgl-project/sglang/actions/runs/34024595305)

### L1-6252993afe  (L1, 2026-09-06, sha 6252993afe07, PR #38238)
TITLE: chore: update CI test est_time values (#38238)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py (+1/-1); test/registered/8-gpu-models/test_deepseek_v32_indexcache.py (+1/-1); test/registered/attention/test_chunk_gated_delta_rule.py (+1/-1); test/registered/attention/test_create_kvindices.py (+1/-1); test/registered/attention/test_deterministic.py (+1/-1); test/registered/attention/test_flash_attention_4.py (+1/-1); test/registered/attention/test_gdn_fused_split_head_ratios.py (+1/-1); test/registered/attention/test_gdn_noncontiguous_stride.py (+1/-1); test/registered/attention/test_gdn_prefill_layout.py (+1/-1); test/registered/attention/test_gemma4_swa_triton_oob_regression.py (+1/-1); (+1006 more)
LABELS: quant, amd, lora, Multi-modal, deepseek, speculative-decoding, hicache, blackwell, npu, apple-silicon
BODY: ## Summary ⏎  ⏎ Refreshes `est_time` literals from [`sgl-project/sglang-ci-stats`](https://github.com/sgl-project/sglang-ci-stats)'s `model.json` (per-(suite, file) p90 over recent successful CI runs on `main`). ⏎  ⏎ This keeps the LPT load-balancing algorithm accurate for partitioning tests across parallel CI jobs, and serves as the static fallback when `compute_partitions` cannot fetch the live model at PR time. ⏎  ⏎ ### Significant est_time changes (203 o …[truncated]

### L1-ed82def55f  (L1, 2026-09-06, sha ed82def55fac, PR #38047)
TITLE: [Config] Round 6.2: the field declarations move to their namespaces, and the record is assembled from them (#38047)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/arg_utils.py (+49/-27); python/sglang/srt/arg_groups/choices.py (+289/-0); python/sglang/srt/arg_groups/field_order.py (+511/-0); python/sglang/srt/arg_groups/fields/__init__.py (+70/-0); python/sglang/srt/arg_groups/fields/device.py (+83/-0); python/sglang/srt/arg_groups/fields/disagg.py (+167/-0); python/sglang/srt/arg_groups/fields/exec_.py (+863/-0); python/sglang/srt/arg_groups/fields/lora.py (+119/-0); python/sglang/srt/arg_groups/fields/memory.py (+242/-0); python/sglang/srt/arg_groups/fields/mm.py (+154/-0); (+8 more)
LABELS: lora
BODY: Second of four; stacked on #38046. Mechanical relocation plus one design change ⏎ that the relocation makes possible. **Review by checking the identity proofs at ⏎ the bottom** -- nothing here is meant to change behaviour. ⏎  ⏎ ## The declarations move ⏎  ⏎ `ServerArgs` carried all 487 declarations in one 4,462-line file, each tagged ⏎ with an `NS("...")` marker naming the namespace it belongs to -- structure ⏎ supplied by annotation, in a file a namespace away  …[truncated]

### L1-b99175dc7d  (L1, 2026-09-06, sha b99175dc7d8d, PR #38049)
TITLE: [Config] Round 6.4: the runtime reads the bags, not the record (#38049)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/dwdp/dwdp_manager.py (+1/-1); python/sglang/srt/layers/moe/qwen35_flashinfer_fusion.py (+9/-6); python/sglang/srt/constrained/grammar_manager.py (+2/-1); python/sglang/srt/disaggregation/decode.py (+5/-9); python/sglang/srt/disaggregation/encoder/grpc_server.py (+2/-2); python/sglang/srt/disaggregation/encoder/http_server.py (+8/-8); python/sglang/srt/disaggregation/encoder/receiver.py (+1/-1); python/sglang/srt/disaggregation/encoder/runtime.py (+4/-3); python/sglang/srt/disaggregation/encoder/server.py (+6/-6); python/sglang/srt/distributed/bootstrap.py (+8/-11); (+75 more)
LABELS: quant, lora, Multi-modal, deepseek, hicache
BODY: Last of four; stacked on #38048. ⏎  ⏎ The record is the operator's input; the bags are what is in effect. A reader ⏎ that takes the record and reads a field off it gets the input, which is the ⏎ wrong one of the two whenever resolution decided something -- and the mistake is ⏎ silent, because for most fields and most launches the two agree. Several of ⏎ these files already read both ways, sometimes in the same expression: ⏎  ⏎ ```python ⏎ get_tokenizer( ⏎     get_se …[truncated]

### L1-30705c004c  (L1, 2026-09-07, sha 30705c004ca4, PR #33608)
TITLE: [Deepseek V4] Keep fp32 routing weights in the mxfp4 trtllm MoE (#33608)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py (+2/-5)
LABELS: high priority, run-ci, jit-kernel, bypass-fastfail
BODY: The packed `topk_ids` format is `(expert_id << 16) | bf16(weight)`, so it truncates the fp32 routing weights that `moe_fused_gate` produces. Pass the ids and weights unpacked instead, the kernel then keeps them in fp32, and the `PackTopkIds` launch goes away. ⏎  ⏎ The reference DeepSeek-V4 implementation keeps the routing weights in fp32 (`inference/model.py`, `Gate.forward`). ⏎  ⏎ Requires flashinfer 0.6.18 (https://github.com/flashinfer-ai/flashinf …[truncated]

### L1-28457f0dca  (L1, 2026-09-07, sha 28457f0dcab4, PR #37199)
TITLE: fix(gpt-oss): avoid duplicate MoE reduction with DP attention (#37199)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/gpt_oss.py (+16/-3); test/registered/models_e2e/test_gpt_oss_4gpu_mxfp4.py (+24/-2)
LABELS: high priority, run-ci, bypass-fastfail
ISSUES: #37187 [Bug] GPT-OSS with DP attention enabled produces garbage output (GSM8K 0/128 vs 124/128)
BODY: ## Motivation ⏎  ⏎ Fixes #37187. ⏎  ⏎ Enabling DP attention for GPT-OSS-120B with TP4/DP4/EP4 caused severe output corruption: ⏎  ⏎ | Configuration | GSM8K first 128 | MMLU fixed 128 subset | ⏎ | --- | ---: | ---: | ⏎ | DP attention disabled | 124/128 | 110/128 | ⏎ | DP attention enabled | 0/128 | 4/128 | ⏎  ⏎ Affected responses contained repeated Harmony channel markers, empty final ⏎ answers, and generations that frequently reached the 8192-token limit. ⏎  …[truncated]

### L1-31d28a2961  (L1, 2026-09-07, sha 31d28a2961b7, PR #38112)
TITLE: [NPU] Fix failed test cases in pr‑test‑npu and improve execution efficiency (#38112)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-npu.yml (+4/-4); python/sglang/multimodal_gen/test/server/ascend/conftest.py (+111/-0); python/sglang/multimodal_gen/test/server/ascend/testcase_configs_npu.py (+4/-0); test/registered/npu/basic_function/parallel_strategy/data_parallelism/test_npu_load_balance_method.py (+0/-1); test/registered/npu/basic_function/parallel_strategy/data_parallelism/test_npu_load_balance_method_pd_disaggregation.py (+0/-1); test/registered/npu/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep.py (+0/-105); test/registered/npu/basic_function/runtime_opts/test_npu_tp4_bf16.py (+0/-86); test/registered/npu/basic_function/speculative_inference/test_npu_speculative_attention_mode.py (+1/-66); test/registered/npu/basic_function/speculative_inference/test_npu_speculative_draft_attention_backend.py (+0/-113); test/registered/npu/basic_function/speculative_inference/test_npu_speculative_moe_a2a_backend.py (+4/-1); (+1 more)
LABELS: speculative-decoding, npu, run-ci, diffusion
BODY: ## Motivation ⏎ Fix failed test cases in pr‑test‑npu and improve execution efficiency ⏎  ⏎  ⏎ ## Modifications ⏎ ### 1、Delete non‑essential test cases from the pr‑test‑npu pipeline. ⏎  ⏎ 1. Remove test_npu_speculative_draft_attention_backend.py, covered by accuracy & performance test cases ⏎ 2. Add `--speculative‑draft‑attention‑backend=ascend` and `--speculative‑moe‑runner‑backend=auto` arguments to `test_npu_speculative_moe_a2a_backend.py` to supplemen …[truncated]

### L1-c99d906eff  (L1, 2026-09-07, sha c99d906effa8, PR #33591)
TITLE: Drop the routing bias casts in flashinfer trtllm MoE (#33591)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+1/-4); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxint4_moe.py (+1/-5)
LABELS: run-ci
BODY: This cast doesn't actually need to exist. it's already handled by the routing method ⏎  ⏎ Requires flashinfer 0.6.18 (https://github.com/flashinfer-ai/flashinfer/pull/3898) ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #33853194348](https://github.com/sgl-project/sglang/actions/runs/33853194348) ⏎ Latest PR Test (Extra): :x: [Run #33853194302](https://github.com/sgl-project/sglang/actions/runs/33853194302) ⏎ Latest PR Test (AMD ROCm 7.2 …[truncated]

### L1-dcebe8c473  (L1, 2026-09-07, sha dcebe8c4733a, PR #38116)
TITLE: [Kernel] Add fused MoE Triton configs for Qwen3.8-Flash-Next FP8 on NVIDIA H200 NVL (TP2+EP2) (#38116)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=256,N=640,device_name=NVIDIA_H200_NVL,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); benchmark/kernels/fused_moe_triton/common_utils.py (+1/-0)
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎ Serving `Qwen/Qwen3.8-Flash-Next-FP8` with `--tp 2 --ep-size 2` on NVIDIA **H200 NVL** logs: ⏎  ⏎ ``` ⏎ Using default MoE kernel config. Performance might be sub-optimal! Config file not found at ⏎ .../configs/triton_3_7_1/E=256,N=640,device_name=NVIDIA_H200_NVL,dtype=fp8_w8a8,block_shape=[128, 128].json ⏎ ``` ⏎  ⏎ There is currently no `NVIDIA_H200_NVL` config in the repo at all (the H200 NVL reports a different device name than the SXM H200, s …[truncated]

### L1-a8b2f36dee  (L1, 2026-09-07, sha a8b2f36dee1a, PR #37376)
TITLE: [kernel] add fused silu mul quant fp8 (#37376)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels
FILES: python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py (+96/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+73/-7); python/sglang/kernels/aot/benchmark/bench_silu_quant_fp8.py (+112/-0); test/registered/kernels/ops/quantization/test_fused_silu_mul_quant_fp8.py (+414/-0)
LABELS: quant, sgl-kernel, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study: this PR was reverted by PR 38381 (confirmed_revert, reason=correctness_or_accuracy) || deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ Reduce MoE launch and memory-traffic overhead by fusing `silu_and_mul` and `per_token_group_quant_fp8`. The fused Triton kernel computes the gated activation, derives per-token-group FP8 scales, and writes the FP8 down-projection input in one pass. ⏎  ⏎ > The no-flag auto-dispatch update is in [xieminghe1/sglang#1](https://github.com/xieminghe1/sglang/pull/1). The original PR head has **Allow edits from maintainers** disabled, so it mu …[truncated]

### L1-f4b75b5c36  (L1, 2026-09-07, sha f4b75b5c36ca, PR #37995)
TITLE: docs(cookbook): Qwen3.8-Flash-Next NVFP4 recipes for DGX Spark (1x, 2x) and RTX PRO 6000 (#37995)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx (+48/-4); docs/src/snippets/configs/Qwen/qwen3.8-flash-next-benchmarks.jsx (+181/-0); docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx (+562/-9)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ The Qwen3.8-Flash-Next cookbook page had NVFP4 recipes only for single-GPU B200/B300/GB300. This PR adds DGX Spark (GB10) recipes for the NVFP4 checkpoints, each verified on the hardware and reproduced from the exact command the Deploy panel emits: ⏎  ⏎ - 2x DGX Spark, TP=2, `RadixArk/Qwen3.8-Flash-Next-NVFP4`: low latency (in-checkpoint MTP head) and high throughput. ⏎ - 2x DGX Spark, TP=2, `nvidia/Qwen3.8-Flash-Next-NVFP4` (ModelOpt MI …[truncated]

### L1-85d39401c8  (L1, 2026-09-07, sha 85d39401c85d, PR #38293)
TITLE: [CP V1 Deprecation 3.5/5]  Deprecate HIP/NPU/MUSA prefill CP and remove legacy implementation (#38293)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/attention_hook.py (+1/-1); python/sglang/srt/arg_groups/cuda_graph_hook.py (+0/-4); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+0/-18); python/sglang/srt/arg_groups/field_order.py (+0/-4); python/sglang/srt/arg_groups/fields/parallel.py (+0/-4); python/sglang/srt/arg_groups/model_hook.py (+1/-3); python/sglang/srt/arg_groups/parallel_hook.py (+13/-104); python/sglang/srt/arg_groups/pipeline.py (+11/-18); python/sglang/srt/arg_groups/validation_hook.py (+0/-6); python/sglang/srt/batch_overlap/operations_strategy.py (+4/-24); (+37 more)
LABELS: amd, deepseek, npu, run-ci, mthreads, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Deprecate the legacy prefill context-parallel implementations on HIP/ROCm, Ascend NPU, and MUSA ahead of the upcoming CP refactor. CUDA's strategy-based prefill CP, decode context parallelism (DCP), and ordinary non-CP platform inference remain in scope for regression protection, not removal. ⏎  ⏎ ## Modifications ⏎  ⏎ - Reject `--enable-prefill-cp` on HIP/NPU/MUSA in the parallel hooks with: `Prefill CP on HIP/NPU/MUSA is deprecated; CP s …[truncated]

### L1-4dcecc7891  (L1, 2026-09-07, sha 4dcecc78916b, PR #38381)
TITLE: Revert "[kernel] add fused silu mul quant fp8" (#38381)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels
FILES: python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py (+0/-96); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+7/-73); python/sglang/kernels/aot/benchmark/bench_silu_quant_fp8.py (+0/-112); test/registered/kernels/ops/quantization/test_fused_silu_mul_quant_fp8.py (+0/-414)
LABELS: quant, sgl-kernel, jit-kernel
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 37376 reason=correctness_or_accuracy
BODY: Reverts sgl-project/sglang#37376 ⏎  ⏎ ## Why ⏎  ⏎ #37376 enables a fused silu+mul+FP8-quant kernel on the block-wise FP8 MoE path. It is not numerically equivalent to the two-kernel path it replaces, and the difference is large enough to cost speculative-decoding acceptance on GLM-5.2-FP8 EAGLE MTP. ⏎  ⏎ `test/registered/e2e/models/test_dsa_glm52_tp_mtp.py::test_bs_1_speed` asserts `accept_length > 4.0` and is currently failing on main. ⏎  ⏎ ## Bisect ⏎  ⏎ Two `Reru …[truncated]

### L1-5aa913e156  (L1, 2026-09-07, sha 5aa913e15639, PR #37133)
TITLE: [AMD][GLM-5.2] Keep GlmMoeDsa MoE e_score_correction_bias in fp32 (#37133)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+13/-2); python/sglang/srt/configs/model_config.py (+8/-2); python/sglang/srt/models/deepseek_v2.py (+4/-1); test/registered/unit/models/test_glmmoedsa_correction_bias_fp32.py (+141/-0)
LABELS: amd, deepseek, run-ci
BODY: ## Problem ⏎  ⏎ GLM-5.2's MoE scores experts with `sigmoid(logits) + e_score_correction_bias` and takes the top 8. ⏎  ⏎ That bias is fp32 in the checkpoint, with values in [6.817, 7.063] — a spread of 0.246 sitting up at 7. Near 7, bf16's smallest step is 0.03125, so it can only represent about 8 distinct values there. `MoEGate.__init__` casts the parameter to bf16 whenever a quant_config is present, so 238 distinct biases collapse into 8. ⏎  ⏎ The res …[truncated]

### L1-62a4a6ea0e  (L1, 2026-09-07, sha 62a4a6ea0edc, PR #37373)
TITLE: [NPU] Add NPU arch35 support and enhance DSV4 processing in DeepSeek-V4 (#37373)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+69/-1); python/sglang/srt/disaggregation/ascend/conn.py (+24/-0); python/sglang/srt/disaggregation/ascend/transfer_engine.py (+5/-1); python/sglang/srt/disaggregation/utils.py (+41/-0); python/sglang/srt/environ.py (+4/-0); python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py (+333/-109); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_common_hooks.py (+31/-4); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py (+135/-29); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_req_to_token_pool.py (+6/-0); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_rope.py (+54/-0); (+15 more)
LABELS: quant, amd, deepseek, hicache, npu, run-ci, diffusion, jit-kernel, unified-radix-cache, memory-pool
BODY: ## Motivation ⏎  ⏎   Enable DeepSeek-V4 inference on Ascend Atlas A5 (Ascend 950), including A5-specific quantization, attention, ⏎   memory-pool, and MoE execution paths, while keeping non-A5 NPU behavior unchanged. ⏎  ⏎   ## Modifications ⏎  ⏎   - Add Atlas A5 / Arch35 capability detection and route A5-specific NPU implementations accordingly. ⏎   - Add A5 support for DeepSeek-V4 sparse attention, compressor/indexer, FP8 KV cache layout, and cycle-stat …[truncated]

### L1-5097f9ac95  (L1, 2026-09-08, sha 5097f9ac95b0, PR #37325)
TITLE: Disable Hopper GLM shared-expert fusion for modelopt_fp4 Marlin (#37325)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+12/-0); test/registered/unit/models/test_shared_experts_fusion_gates.py (+49/-0)
LABELS: deepseek, run-ci
BODY: ## Summary ⏎  ⏎ - disable shared-expert fusion by default for `GlmMoeDsaForCausalLM` when using `modelopt_fp4` with the Marlin MoE backend on Hopper GPUs ⏎ - keep `--enforce-shared-experts-fusion` as an explicit override ⏎ - add unit coverage for the new Hopper gate and the explicit override path ⏎  ⏎ ## Repro ⏎  ⏎ Issue: #37268 ⏎  ⏎ I updated local `sglang` to `origin/main` (`079afaffb19abbe42c09f0ebb8322d6e9f6f2905`) and reproduced the problematic Hopper path on T …[truncated]

### L1-30e7a3072d  (L1, 2026-09-08, sha 30e7a3072d3f, PR #33631)
TITLE: Keep fp32 routing weights in the fp8 block-scale and bf16 trtllm MoE (#33631)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/kernels/ops/moe/__init__.py (+0/-10); python/sglang/kernels/ops/moe/pack_topk_ids.py (+0/-101); python/sglang/srt/layers/moe/flashinfer_trtllm_moe.py (+3/-1); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+35/-24); test/registered/kernels/ops/test_kimi_k3_prerequisite_ops.py (+13/-2)
LABELS: run-ci, jit-kernel
BODY: The packed `topk_ids` format is `(expert_id << 16) | bf16(weight)`, so the routed fp8 block-scale and bf16 paths truncate the fp32 routing weights the router produced. Pass the ids and weights unpacked instead, as the mxfp4 path does in #33608. ⏎  ⏎ The fused gate emits packed ids directly, and those already carry bf16 weights, so that case keeps the packed form. ⏎  ⏎ The nvfp4 `fused_experts_none_to_flashinfer_trtllm_fp4` call site was the last one stil …[truncated]

### L1-f4bbf12423  (L1, 2026-09-08, sha f4bbf12423b8, PR #38374)
TITLE: docs(cookbook): Qwen3.5 FP8 on B200/B300 — trtllm-gen MoE + symm mem (#38374)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/Qwen/Qwen3.5.mdx (+3/-1); docs/src/snippets/autoregressive/qwen35-deployment.jsx (+13/-2)
LABELS: documentation
BODY: Emit --moe-runner-backend flashinfer_trtllm for the FP8 MoE recipes on B200/B300. This is not a default: the auto-promotion to flashinfer_trtllm in arg_groups/overrides.py is gated to the DeepSeek arch family, and Fp8MoEMethod.create_moe_runner resolves "auto" to Triton for a TP-only run, so an FP8 Qwen3.5 MoE deployment otherwise lands on the Triton MoE runner. The NVFP4 recipes already set it. Restricted to the MoE sizes (397B-A17B / 122B-A10B  …[truncated]

### L1-91a45ea37e  (L1, 2026-09-08, sha 91a45ea37e04, PR #30345)
TITLE: [Intel][XPU][LoRA] Enable LoRA on Intel XPU (#30345)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/lora/lora_moe_runners.py (+1/-1); .github/workflows/pr-test-xpu.yml (+2/-2); python/sglang/srt/arg_groups/overrides.py (+2/-0); python/sglang/srt/layers/rotary_embedding/base.py (+14/-2); python/sglang/srt/layers/rotary_embedding/mrope.py (+9/-2); python/sglang/srt/lora/backend/base_backend.py (+6/-4); python/sglang/srt/lora/backend/chunked_backend.py (+1/-1); python/sglang/srt/lora/backend/torch_backend.py (+1/-1); python/sglang/srt/lora/backend/triton_backend.py (+1/-1); python/sglang/srt/lora/lora_overlap_loader.py (+11/-8); (+9 more)
LABELS: lora, intel, xpu, run-ci, run-ci-extra
BODY: Enable the LoRA functionality on XPU (in addition to CUDA/ROCm), and enable the corresponding unit tests. ⏎  ⏎ Source changes: ⏎ - backends (triton/chunked/torch): use torch.device(self.device) instead of a hard-coded "cuda". ⏎ - lora_moe_runners: route XPU to the pure-torch _naive_moe_lora_align_block_size fallback. ⏎ - rotary_embedding base.py / mrope.py: guard the XPU-only sgl_kernel imports (fused_qk_rope_with_cos_sin_cache_inplace, multimodal_rot …[truncated]

### L1-52fecfdf09  (L1, 2026-09-08, sha 52fecfdf0908, PR #37500)
TITLE: support qwen 3.8 flash next (#37500)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/common_utils.py (+7/-0); python/sglang/kernels/jit/csrc/attention/qsa_indexer.cuh (+486/-0); python/sglang/kernels/jit/csrc/elementwise/fast_topk.cuh (+291/-0); python/sglang/kernels/jit/csrc/elementwise/grouped_gemma_rmsnorm.cuh (+178/-0); python/sglang/kernels/jit/csrc/elementwise/hc_combine.cuh (+381/-0); python/sglang/kernels/kda_kernels/README.md (+1/-0); python/sglang/kernels/kda_kernels/qwen38_qsa_sm121/README.md (+51/-0); python/sglang/kernels/kda_kernels/qwen38_qsa_sm121/__init__.py (+83/-0); python/sglang/kernels/kda_kernels/qwen38_qsa_sm121/kernel.py (+269/-0); python/sglang/kernels/ops/attention/__init__.py (+86/-2); (+81 more)
LABELS: documentation, high priority, quant, speculative-decoding, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight, unified-radix-cache
BODY: initial pr: https://github.com/sgl-project/sglang/pull/36497 ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #34195586068](https://github.com/sgl-project/sglang/actions/runs/34195586068) ⏎ Latest PR Test (Extra): :white_check_mark: [Run #34273666129](https://github.com/sgl-project/sglang/actions/runs/34273666129) ⏎ Latest PR Test (AMD ROCm 7.2): :x: [Run #34195586146](https://github.com/sgl-project/sglang/actions/runs/34195586146)

### L1-5177a3ec08  (L1, 2026-09-08, sha 5177a3ec0854, PR #36557)
TITLE: MiniMax-M3: Triton split-K router GEMV with in-kernel fixup (#36557)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/minimax_m3.py (+14/-0); python/sglang/kernels/ops/gemm/router_gemv.py (+175/-0)
LABELS: amd, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: # MiniMax-M3: Triton split-K router GEMV with in-kernel split-K fixup ⏎  ⏎ The MiniMax-M3 MoE router gate is a skinny GEMV — `[M, K]` bf16 x `[N, K]` bf16 -> `[M, N]` fp32, with `N ~ 128` experts and `M <= 64` at decode — and the vendor BLAS solutions picked for that shape run far off the memory roofline on gfx950 (~0.1 TB/s for a ~1.6MB weight read at `M<=8`, `N=128`). This PR adds `python/sglang/kernels/ops/gemm/router_gemv.py`, a Triton split-K ke …[truncated]

### L1-db272201a2  (L1, 2026-09-08, sha db272201a2db, PR #38375)
TITLE: [Config] Retire get_global_server_args, and clear the deprecated flags that have a replacement (#38375)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.deepgemm_megamoe
FILES: .claude/skills/cookbook-add-model/references/authoring-reference.md (+1/-1); .claude/skills/cookbook-migrate-model/SKILL.md (+2/-2); .claude/skills/cookbook-migrate-model/references/dimension-mapping.md (+1/-1); .claude/skills/sglang-runtime-context/SKILL.md (+7/-5); benchmark/bench_linear_attention/bench_int8_checkpoint_reuse.py (+2/-2); docs/cookbook/autoregressive/DeepSeek/DeepSeek-V3.mdx (+1/-1); docs/cookbook/autoregressive/Meituan/LongCat-2.0.mdx (+1/-1); docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx (+1/-1); docs/docs/advanced_features/cuda_graph_for_multi_modal_encoder.mdx (+7/-4); docs/docs/advanced_features/pd_disaggregation.mdx (+3/-4); (+203 more)
LABELS: documentation, quant, lora, Multi-modal, deepseek, speculative-decoding, hicache, sgl-kernel, npu, run-ci
BODY: Two cleanups on the config tier, plus the review rounds they went through. ⏎ Separate commits, in order. ⏎  ⏎ ## `get_global_server_args` is retired ⏎  ⏎ It was the last spelling of "reach for the whole record and pick a field off ⏎ it". Nothing in `srt` called it any more -- the two remaining references were ⏎ bare imports, one a deliberate re-export -- so this is where keeping it costs ⏎ more than removing it. The name still worked, still returned a `ServerArg …[truncated]

### L1-1ad3eb09a9  (L1, 2026-09-09, sha 1ad3eb09a91f, PR #38590)
TITLE: [Attention] Size FlashInfer MLA indptr buffers to the padded max batch (#38590)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+12/-4)
BODY: ## Motivation ⏎  ⏎ Under MLP sync (DP attention, DeepEP, or `--moe-a2a-backend megamoe`) the eager and cuda-graph runners pad the request count to the attn-tp alignment (`get_eager_max_batch_size` / `get_cuda_graph_max_batch_size`), so the warm-up dummy batch and the largest captured decode batch can be wider than `req_to_token_pool.size`. ⏎  ⏎ `FlashInferMLAAttnBackend` sized `kv_indptr`, `qo_indptr` and `q_indptr_decode` to the raw pool size (and the c …[truncated]

### L1-72d5c5bb73  (L1, 2026-09-09, sha 72d5c5bb73ca, PR #38612)
TITLE: [Kimi-K3] Accept fp32 routing weights in the fused MoE finalize (#38612)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/moe/moe_finalize_fuse_shared.cu (+2/-3); python/sglang/srt/models/kimi_k3.py (+4/-3); python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh (+26/-14); python/sglang/kernels/jit/include/sgl_kernel/math.cuh (+6/-0); python/sglang/kernels/ops/kimi_k3/all_reduce.py (+2/-1); python/sglang/srt/layers/k3_ar_fusion.py (+0/-2); test/registered/kernels/ops/kimi_k3/test_ar_fusion.py (+15/-6)
LABELS: jit-kernel
BODY: ## Summary ⏎  ⏎ The trtllm-gen deferred finalize returns its routing weights at the dtype the caller passed in, so K3's unpacked `(topk_ids, topk_weights)` routing yields fp32 weights where the packed form yielded bf16. The fused finalize kernel matched bf16 only and a 72-token decode batch killed the server during CUDA-graph capture. #38588 stopped that by casting to bf16 at the call site. This PR makes the kernel take either dtype instead, and re …[truncated]

### L1-880d6fa64d  (L1, 2026-09-09, sha 880d6fa64d18, PR #30805)
TITLE: [DSv4] Integrate TRT-LLM DSv4 Attention for SM100/103 (#30805)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/deepseek_v4_hook.py (+41/-0); python/sglang/srt/arg_groups/fields/exec_.py (+13/-0); python/sglang/srt/layers/attention/attention_registry.py (+7/-4); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+112/-8); python/sglang/srt/layers/attention/deepseek_v4_trtllm_backend.py (+532/-0); python/sglang/srt/layers/attention/dsv4/compressor_trtllm.py (+120/-0); python/sglang/srt/layers/attention/dsv4/compressor_v2.py (+16/-0); python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py (+94/-1); python/sglang/srt/speculative/draft_utils.py (+7/-5); test/registered/attention/unittests/dsv4/test_deepseek_v4.py (+28/-0); (+3 more)
LABELS: high priority, deepseek, blackwell, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight, memory-pool
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Integrates TRT-LLM attention kernel for DSv4 style attention (CSA, HCA). ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ SGLANG_DSV4_ATTN_DECODE_BACKEND=flashmla/trtllm_gen \ ⏎ python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V4-Pro \ ⏎   --trust-remote-code --tp 8 --moe-runner-backend flashinfer_mxfp4 \ ⏎   --chunked-prefill-size 4096 --disable-flashinfer-autotune \ ⏎   --mem-fraction-static 0.88 --max-running-r …[truncated]

### L1-ffe98a4279  (L1, 2026-09-09, sha ffe98a4279ba, PR #37982)
TITLE: [Diffusion][MiniMax-H3] Add SM90 Sage compute for SubBlock sparse attention (#37982)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py (+69/-28); python/sglang/kernels/ops/attention/subblock_sage_fp8_sm90.py (+259/-0); python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py (+26/-0); python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/README.md (+36/-16); python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/__init__.py (+3/-3); python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py (+94/-18); python/sglang/multimodal_gen/test/unit/test_minimax_h3_admission.py (+1/-0); python/sglang/multimodal_gen/test/unit/test_subblock_sparse_attention.py (+41/-1); test/registered/cpu/test_subblock_sparse_attention.py (+231/-2); test/registered/kernel/attention/test_subblock_sage_fp8_sm90.py (+140/-0)
LABELS: documentation, run-ci, diffusion, jit-kernel
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎ Inspired by the combination of block-sparse attention and Sage FP8 compute in [FlashInfer's SM100 backend](https://github.com/flashinfer-ai/flashinfer/pull/4612), this PR adds optional SageAttention2 compute to MiniMax-H3's SubBlock sparse attention backend on SM90. ⏎  ⏎  ⏎ The new mode is opt-in. BF16 remains the default, and Sage compute currently targets SM90. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add `compute_mode="sage_fp8"` with online INT8- …[truncated]

### L1-69777c4d36  (L1, 2026-09-09, sha 69777c4d36c0, PR #38802)
TITLE: Add DeepSeek-V4.1 Flash cookbook (#38802)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+0/-1); docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx (+190/-0); docs/cookbook/autoregressive/intro.mdx (+1/-1); docs/docs.json (+1/-0); docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx (+375/-0); docs/src/snippets/configs/popular-models.jsx (+17/-0)
LABELS: documentation, deepseek
BODY: Adds the config-driven cookbook page for DeepSeek-V4.1 Flash, covering GB300, H200, B200, B300 and MI350X, plus the nav entry, the DeepSeek homepage card, and a popular-models entry. ⏎  ⏎ Docs only — six files, no runtime changes. ⏎  ⏎ ## Matrix shape ⏎  ⏎ Two operating points per platform: ⏎  ⏎ - **Low-Latency** — TP + EP with DSpark, the model's bundled three-stage draft. ⏎ - **High-Throughput** — speculation off, decode CUDA graph, wider concurrency. ⏎  ⏎ On GB300, …[truncated]

### L1-2b1c4e4c85  (L1, 2026-09-10, sha 2b1c4e4c854a, PR #38667)
TITLE: [NPU] Set DEEPEP_HYBRID_DEPLOYMENT=1 for collocated DeepEP test cases (#38667)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/npu/accuracy/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_gpqa.py (+1/-0); test/registered/npu/accuracy/glm5_2/test_npu_glm_5_2_w4a8_16p_gpqa.py (+1/-0); test/registered/npu/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_deepseek_v3_2_w8a8.py (+1/-0); test/registered/npu/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_qwen3_480b.py (+1/-0); test/registered/npu/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_qwen3_next.py (+1/-0); test/registered/npu/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_low_latency_deepseek_v3_2_w8a8.py (+1/-0); test/registered/npu/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_low_latency_qwen3_480b.py (+1/-0); test/registered/npu/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_low_latency_qwen3_next.py (+1/-0); test/registered/npu/basic_function/parallel_strategy/expert_parallelism/test_npu_eplb_min_rebalancing_utilization_threshold.py (+2/-0); test/registered/npu/basic_function/parallel_strategy/expert_parallelism/test_npu_expert_distribution_recorder_mode.py (+1/-0); (+4 more)
LABELS: deepseek, npu, run-ci
BODY: ## Motivation ⏎  ⏎ Prefill/Decode collocated (hybrid) deployments on Ascend NPU with the DeepEP backend require the `DEEPEP_HYBRID_DEPLOYMENT=1` environment variable. Some registered NPU test cases launch SGLang with `--moe-a2a-backend deepep` in collocated deployments (deepep-mode auto / low_latency / normal) but do not set this variable, so the DeepEP layer may not run with the correct hybrid-deployment configuration. ⏎  ⏎ ## Modifications ⏎  ⏎ Add `"DEEPE …[truncated]

### L1-1b77f498a0  (L1, 2026-09-10, sha 1b77f498a0f7, PR #31470)
TITLE: [NVIDIA] Support flashinfer Mega Moe (#31470)
SOURCES: path_core, symbol_pickaxe, release_notes, corpus:production-kernel-provenance, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework, L1.runner.flashinfer_trtllm, L1.runner.flashinfer_megamoe, L1.ep.other_dispatchers, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/sglang/srt/layers/moe/flashinfer_megamoe.py (+672/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-0); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+31/-9); python/sglang/srt/layers/moe/moe_runner/runner.py (+7/-0); python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py (+85/-22); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+2/-0); python/sglang/srt/layers/moe/utils.py (+43/-3); docs/docs/advanced_features/server_arguments.mdx (+7/-1); docs/docs/references/environment_variables.mdx (+15/-0); python/pyproject.toml (+1/-0); (+21 more)
LABELS: documentation, high priority, quant, dependencies, deepseek, run-ci, bypass-fastfail, release-highlight
BODY: Fork from https://github.com/djns99/sglang/tree/djns99/mega_moe_flashinfer ⏎ @djns99 is the main author of this PR.  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get …[truncated]

### L1-7152c14384  (L1, 2026-09-10, sha 7152c143841d, PR #34459)
TITLE: Fix DeepSeek-V4 routing: sqrtsoftplus underflow and unfloored renorm (#34459)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.fused_gate, L1.routing.hash_topk
FILES: python/sglang/kernels/jit/csrc/deepseek_v4/hash_topk.cuh (+5/-1); python/sglang/kernels/jit/csrc/moe/moe_fused_gate.cuh (+2/-2); python/sglang/kernels/ops/moe/moe_fused_gate.py (+7/-2); python/sglang/srt/layers/moe/hash_topk.py (+9/-2)
LABELS: run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ Similar bug to https://github.com/sgl-project/sglang/issues/30989 ⏎  ⏎ DeepSeek-V4's `sqrtsoftplus` gate has four routing implementations, and each was missing a different piece of the numerics its reference gate provides (`inference/model.py` `Gate.forward`, `F.softplus`). Same class as flashinfer-ai/flashinfer#3803. ⏎  ⏎ 1. **Triton router** (default, layers 3-42) computed `log(1.0 + exp(x))`, which in fp32 rounds to `log(1.0) == 0 …[truncated]

### L1-c0b790cf7f  (L1, 2026-09-10, sha c0b790cf7fe6, PR #32114)
TITLE: Delete cutlass_mla, non-Marlin GPTQ, AWQ AOT kernel, and Dual Chunk Flash Attention (#32114)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/aot/CMakeLists.txt (+0/-10); .github/labeler.yml (+0/-1); docs/cookbook/autoregressive/DeepSeek/DeepSeek-V3.mdx (+1/-1); docs/docs/advanced_features/attention_backend.mdx (+0/-16); docs/docs/advanced_features/quantization.mdx (+4/-4); docs/docs/advanced_features/server_arguments.mdx (+3/-3); examples/runtime/engine/offline_batch_inference_qwen_1m.py (+0/-71); python/sglang/kernels/aot/benchmark/bench_awq_dequant.py (+0/-151); python/sglang/kernels/aot/benchmark/bench_cutlass_mla.py (+0/-173); python/sglang/kernels/aot/csrc/attention/cutlass_mla_kernel.cu (+0/-274); (+61 more)
LABELS: documentation, quant, speculative-decoding, sgl-kernel, blackwell, run-ci, mthreads, bypass-fastfail
ISSUES: #32111 Deprecate CUTLASS MLA attention backend | #32112 Deprecate legacy (non-Marlin) GPTQ kernel and Dual Chunk Flash Attention backend
BODY: ## Summary ⏎ - Delete the `cutlass_mla` attention backend (kernel + Python integration): SM10.0-only decode kernel, already disabled on GB300 (SM10.3), falls through to `FlashInferMLABackend` for everything except plain decode, no CI coverage. ⏎ - Delete the non-Marlin GPTQ CUDA kernel (`gptq_gemm`/`gptq_shuffle`) and the `"gptq"` GPU quantization choice: superseded by `gptq_marlin` (fully JIT), which `GPTQMarlinConfig` already auto-upgrades compatib …[truncated]

### L1-52c191da52  (L1, 2026-09-10, sha 52c191da5239, PR #38404)
TITLE: [Deps] Retire the CUDA 12 lane (#38404)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: -
FILES: .github/workflows/release-whl-deepep.yml (+7/-103); docker/Dockerfile (+15/-77); docker/Dockerfile.cu134 (+13/-68); docker/sgl-deep-ep.Dockerfile (+2/-18); docker/sgl-deep-gemm.Dockerfile (+1/-2); scripts/build_sgl_deepep.sh (+3/-6); scripts/update_deepep_whl_index.py (+1/-1); .github/workflows/_docker-build-and-publish.yml (+21/-92); .github/workflows/patch-docker-dev.yml (+1/-1); .github/workflows/release-docker-dev.yml (+6/-8); (+28 more)
LABELS: documentation, sgl-kernel, jit-kernel, release-highlight
BODY: ## Summary ⏎  ⏎ Drops the CUDA 12.9 wheels and images, per the [advance notice][notice] — prerequisite for the PyTorch 2.14 upgrade. ⏎  ⏎ Already-published `-cu129` / `-cu12` image tags are unaffected. v0.5.19 is the last release with a CUDA 12 lane. ⏎  ⏎ [notice]: https://sgl-fru7574.slack.com/archives/C064NB2TAP9/p1786520889229779 ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #34440714447](https://github.com/sgl-project/sglang/actions/runs/3444 …[truncated]

### L1-dc2157dcd6  (L1, 2026-09-10, sha dc2157dcd62d, PR #38830)
TITLE: [JIT] Port the expert-pack MXFP4 kernels to load_jit and fix their launch limits (#38830)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/moe/expert_pack_mxfp4.cu (+0/-565); python/sglang/kernels/jit/csrc/moe/expert_pack_mxfp4.cuh (+777/-0); python/sglang/kernels/ops/moe/expert_pack_mxfp4.py (+34/-16)
LABELS: jit-kernel
BODY: ## Summary ⏎  ⏎ `expert_pack_mxfp4` was built with `torch.utils.cpp_extension.load`, whose `FileBaton` waits on a lock **file** and only releases it in a `finally`. A build killed mid-flight leaves the file behind, and every later run against that cache spins on it forever with no output — a 7s test can burn a full CI timeout, and the cache outlives the job. Ported to `load_jit`, which uses `fcntl.flock` (released by the kernel when the holder dies …[truncated]

### L1-b9899b04c1  (L1, 2026-09-10, sha b9899b04c1c6, PR #33939)
TITLE: [AMD] Add gfx1151 (Strix Halo / Ryzen AI MAX+) Docker image (#33939)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/patches/sgl-kernel-gfx1151.sh (+90/-0); docker/rocm-gfx1151.Dockerfile (+156/-0); .github/workflows/release-docker-amd-gfx1151-nightly.yml (+63/-0)
LABELS: amd, run-ci
BODY: Co-author: @hubertlu-tw, @ankith117 ⏎  ⏎ ## Motivation ⏎  ⏎ The existing ROCm image targets CDNA GPUs and includes components that do not support gfx1151. This PR adds a separate image for Strix Halo systems such as the Ryzen AI MAX+ 395. ⏎  ⏎ The scope is limited to building and running SGLang on gfx1151. General RDNA kernel support and performance tuning remain separate work. ⏎  ⏎ ## Changes ⏎  ⏎ - Add `docker/rocm-gfx1151.Dockerfile`, based on AMD's sta …[truncated]

### L1-92dffebe16  (L1, 2026-09-10, sha 92dffebe16f5, PR #38775)
TITLE: [NPU] Set DEEPEP_HYBRID_DEPLOYMENT for new DeepEP tests; switch glm5_2 to w8a8; tune nightly timeouts (#38775)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/nightly-test-npu.yml (+19/-18); python/sglang/test/ascend/e2e/test_npu_performance_utils.py (+1/-0); test/registered/npu/accuracy/glm5_2/test_npu_glm_5_2_w8a8_16p_gpqa.py (+13/-13); test/registered/npu/accuracy/glm5_top64_pruned/test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py (+1/-0); test/registered/npu/accuracy/kimi_k2_6/test_npu_kimi_k2_6_w4a8_16p_in64k_out1k_100ms_aime25.py (+1/-0); test/registered/npu/accuracy/kimi_k3/test_npu_kimi_k3_w4a8_32p_gpqa.py (+1/-0); test/registered/npu/basic_function/parallel_strategy/expert_parallelism/test_npu_moe_runner_backend.py (+1/-0)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ Enable the `DEEPEP_HYBRID_DEPLOYMENT=1` environment variable for NPU test cases that use the DeepEP backend in collocated deployments, switch the glm5_2 accuracy test to w8a8 weights, and tune NPU nightly workflow timeouts. ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Add `"DEEPEP_HYBRID_DEPLOYMENT": "1"` to the environment of 4 NPU test cases that run DeepEP in collocated deployments (multi-node PdMix or single-node): ⏎    - `test/registered/npu/accuracy/ …[truncated]

### L1-335f6aab27  (L1, 2026-09-11, sha 335f6aab27a4, PR #33672)
TITLE: [DSV4] Support raw-index output in TopK v2 (#33672)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dsv4/topk.py (+17/-3); python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh (+37/-11); python/sglang/srt/layers/attention/dsv4/indexer.py (+5/-4); test/registered/kernels/ops/attention/test_topk_v2.py (+35/-0); test/registered/unit/layers/test_dsv4_nonpaged_indexer.py (+73/-0)
LABELS: run-ci, jit-kernel, run-ci-extra
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎ Allow TopK v2 to populate a raw-index output buffer. DSV4 sparse prefill internally allocates `c4_sparse_raw_indices` even when `--enable-return-indexer-topk` is disabled. The previous `raw_indices is None` gate therefore forced the InfX sparse-prefill path to fall back to TopK v1. ⏎  ⏎ When `--enable-return-indexer-topk` is enabled, the capture-only temporary buffer could additionally take precedence over the functional sparse-prefil …[truncated]

### L1-ec30f19e4a  (L1, 2026-09-11, sha ec30f19e4ad3, PR #38578)
TITLE: [LoRA] Support MoE in full and breakable prefill CUDA graphs (#38578)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/lora/backend/base_backend.py (+37/-39); python/sglang/srt/lora/backend/chunked_backend.py (+8/-3); python/sglang/srt/lora/backend/triton_backend.py (+60/-6); python/sglang/srt/lora/layers.py (+5/-1); python/sglang/srt/lora/lora_manager.py (+31/-10); python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py (+1/-1); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+15/-15); test/registered/e2e/lora/test_lora_moe_prefill_cuda_graph.py (+189/-0); test/registered/lora/test_moe_lora_info.py (+171/-1)
LABELS: lora, run-ci
BODY: ## Motivation ⏎  ⏎ Enable MoE LoRA in full and breakable prefill CUDA graphs, including batches above 32 requests. ⏎  ⏎ ## Modifications ⏎  ⏎ - Separate prefill MoE LoRA scratch; preserve request slots and clear unused tails. ⏎ - Capture MoE LoRA hooks, including base-only batches. ⏎ - Compact Triton dense prefill LoRA grids. ⏎ - Size LoRA metadata from Breakable token/request capacity or Full's configured slots. Remove Triton's 32-request cap and avoid oversized  …[truncated]

### L1-7bc4eb3740  (L1, 2026-09-12, sha 7bc4eb374033, PR #39104)
TITLE: [AMD] Update MI355X MXFP4 HiCache defaults and quick-reduce quantization for Qwen3.5 cookbook (#39104)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/Qwen/Qwen3.5.mdx (+3/-3); docs/src/snippets/autoregressive/qwen35-deployment.jsx (+6/-6)
LABELS: documentation
BODY: ## Summary ⏎  ⏎ Update HiCache KV offloading defaults and quick-reduce quantization for the Qwen3.5-397B-A17B MXFP4 MI355X deployment, and bump the MI355X Docker image. ⏎  ⏎ Supersedes #35879 (closed due to branch recreation during conflict resolution). ⏎  ⏎ ## Changes ⏎  ⏎ ### Deployment Generator (`qwen35-deployment.jsx`) ⏎ - **HiCache IO backend**: `direct` → `kernel` ⏎ - **HiCache memory layout**: `page_first_direct` → `page_first` ⏎ - **ROCM_QUICK_REDUCE_QUANTIZA …[truncated]

### L1-a984c78330  (L1, 2026-09-12, sha a984c78330a8, PR #37565)
TITLE: [NPU] Support DFlash speculative decoding for MiMo-V2.5-Pro (mxfp4) (#37565)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/speculative_hook.py (+4/-2); python/sglang/srt/configs/model_config.py (+9/-0); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+67/-11); python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py (+112/-4); python/sglang/srt/layers/quantization/fp8.py (+95/-2); python/sglang/srt/model_loader/loader.py (+3/-0); python/sglang/srt/models/dflash.py (+23/-5); python/sglang/srt/models/mimo_v2.py (+62/-5); python/sglang/srt/speculative/dflash_disaggregation.py (+1/-1); python/sglang/srt/speculative/dflash_utils.py (+27/-0); (+5 more)
LABELS: speculative-decoding, npu, run-ci
BODY: ## Motivation ⏎  ⏎ Enable DFlash speculative decoding for MiMo-V2.5-Pro (mxfp4) on Ascend NPU, covering the full inference path: mxfp4 expert weight loading, DFlash draft/verify execution, NPU graph replay, DP attention, and PD disaggregation. DFlash was previously blocked on this stack by missing weight adaptation, graph-mode metadata bugs, and DP-attention layout mismatches. ⏎  ⏎ ## Modifications ⏎  ⏎ - MiMo-V2 mxfp4 on Ascend : load routed expert we …[truncated]

### L1-6657f7d844  (L1, 2026-09-12, sha 6657f7d8449f, PR #38328)
TITLE: [MoE][ROCm] Admit the unified Triton router on ROCm, including single-group routing (#38328)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py, L1.routing.fused_gate
FILES: python/sglang/kernels/ops/moe/moe_fused_gate.py (+1/-1); python/sglang/srt/layers/moe/topk.py (+20/-10); python/sglang/srt/layers/attention/wave_ops/decode_attention.py (+52/-0); test/registered/kernel/moe/test_jit_grouped_topk.py (+186/-0)
LABELS: amd, dependencies, run-ci, jit-kernel
BODY: The opt-in unified Triton router (`SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK`, off ⏎ by default) was gated on CUDA and on more than one expert group. Nothing in the ⏎ kernel is CUDA-specific, so admit ROCm and single-group routing. CUDA's condition ⏎ is unchanged, and with the flag off every path is byte-identical to before. ⏎  ⏎ Three things follow: ⏎  ⏎ - **Width.** The kernel wants the total top-k, shared slots included, but ⏎   `select_experts` passes ` …[truncated]

### L1-a8b5616303  (L1, 2026-09-12, sha a8b5616303f2, PR #39241)
TITLE: Fix DeepGEMM release dependencies and bound GPU validation (#39241)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/release-whl-deepgemm.yml (+34/-5)
BODY: The [latest DeepGEMM release validation](https://github.com/sgl-project/sglang/actions/runs/34678601083) times out on H200 and fails six test invocations on each Blackwell runner. Blackwell attention alone takes 106–195 minutes. The failures are missing TileLang/TileKernels reference dependencies and DeepEP loading NCCL 2.29.7 although its wheel requires 2.30.7. ⏎  ⏎ Use the bounded `--release` profile from https://github.com/sgl-project/DeepGEMM/pul …[truncated]

### L1-7763f666f3  (L1, 2026-09-12, sha 7763f666f348, PR #39245)
TITLE: Fix DeepGEMM release MegaMoE validation without RDMA (#39245)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/release-whl-deepgemm.yml (+3/-0)
BODY: The [DeepGEMM release B200 job](https://github.com/sgl-project/sglang/actions/runs/34726899652/job/103643335128) fails three MegaMoE invocations when DeepEP requests NCCL GIN on a runner without usable RDMA. Each release test runner uses one NVLink domain, so its reference dispatch/combine only needs intra-node communication. ⏎  ⏎ Set DeepEP's supported `EP_DISABLE_GIN=1` only on the GPU test step. This lets the existing reference comparisons and gra …[truncated]

### L1-5ebb16005d  (L1, 2026-09-13, sha 5ebb16005d28, PR #39171)
TITLE: [DeepSeek-V4.1] Bump FlashMLA to the fork's rebase head (v4.1 kernels) (#39171)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/aot/cmake/flashmla.cmake (+57/-43); python/sglang/kernels/aot/csrc/flashmla_extension.cc (+30/-4); python/sglang/test/kits/basic_decode_correctness_kit.py (+4/-1)
LABELS: sgl-kernel, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ It's just #38942 rebased on main branch ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main …[truncated]

### L1-cebca698e2  (L1, 2026-09-13, sha cebca698e2da, PR #39126)
TITLE: [Qwen3.8] Enable NVIDIA NVFP4 on DGX Spark with file-backed PLE and PDL router fix (#39126)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: python/sglang/kernels/jit/csrc/moe/route_radix.cuh (+3/-2); python/sglang/kernels/ops/moe/moe_fused_gate.py (+8/-6); docs/docs/advanced_features/server_arguments.mdx (+18/-0); docs/docs/references/environment_variables.mdx (+25/-0); python/sglang/srt/arg_groups/fields/exec_.py (+24/-0); python/sglang/srt/arg_groups/memory_hook.py (+6/-0); python/sglang/srt/arg_groups/model_overrides/qwen3_moe.py (+45/-2); python/sglang/srt/configs/qwen4_exp.py (+6/-0); python/sglang/srt/environ.py (+12/-0); python/sglang/srt/layers/quantization/fp8.py (+12/-0); (+11 more)
LABELS: documentation, quant, run-ci, jit-kernel
DEEP_STUDY: deep-study correctness case sglang:cebca698e2: class=nondeterminism_race_sync; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Motivation ⏎  ⏎ Combining these changes into one PR also saves CI resources by avoiding three separate PRs running overlapping CI suites. ⏎  ⏎ Enable single-node DGX Spark / GB10 deployment of `nvidia/Qwen3.8-Flash-Next-NVFP4` from upstream main, with or without its embedded MTP head. Main still lacks the NVIDIA mixed-precision loader handling and file-backed PLE storage needed for this deployment, and both PDL routers read bias before waiting for the …[truncated]

### L1-96a95171c7  (L1, 2026-09-13, sha 96a95171c714, PR #39346)
TITLE: chore: bump sglang-kernel version to 0.4.7 (#39346)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sglang-kernel` version to `0.4.7` across SGLang files to match the version defined in `python/sglang/kernels/aot/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.4.7` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The kernel version in `python/sglang/kernels/aot/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the `sglan …[truncated]

### L1-ca8ecc6a6f  (L1, 2026-09-14, sha ca8ecc6a6f06, PR #37382)
TITLE: [NPU] Support DSV4 host memory cache management (#37382)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/dsv4/c128_sidecar_component.py (+227/-20); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py (+78/-4); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py (+8/-0); python/sglang/srt/managers/schedule_batch.py (+10/-5); python/sglang/srt/mem_cache/allocation.py (+6/-0); python/sglang/srt/mem_cache/allocator/base.py (+11/-2); python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py (+118/-40); python/sglang/srt/mem_cache/memory_pool_host.py (+189/-2); python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py (+1/-3); python/sglang/srt/mem_cache/unified_radix_cache.py (+32/-10); (+2 more)
LABELS: hicache, run-ci, unified-radix-cache, memory-pool
BODY: ## Motivation ⏎  ⏎   DeepSeek-V4 on NPU maintains physically independent FULL, SWA, C4, and C128 cache regions. The existing hierarchical-cache path primarily assumed GPU-style KV-derived sidecars and global page geometry, so it could not correctly manage all DSV4 NPU cache components during host backup, device eviction, and ⏎   load-back. ⏎  ⏎   This PR adds DSV4 NPU host-memory cache management, including Ascend host/device transfers, independent C1 …[truncated]

### L1-2f5cc8e33e  (L1, 2026-09-14, sha 2f5cc8e33e97, PR #39358)
TITLE: [AMD] Align Qwen3.5 MI355X cookbook with AttnFP8-V2 and HiCache direct / page_first_direct (#39358)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/Qwen/Qwen3.5.mdx (+3/-3); docs/src/snippets/autoregressive/qwen35-deployment.jsx (+4/-4)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ InferenceX 8k1k (fixed-seq) and AgentX on MI355X now both serve `amd/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2`. The published Qwen3.5 cookbook still emits `amd/Qwen3.5-397B-A17B-MXFP4`, so the InferenceX recipe-link check fails with a MAJOR model-path contradiction. ⏎  ⏎ Separately, InferenceX AgentX launches HiCache with `--hicache-io-backend direct` and `--hicache-mem-layout page_first_direct`. The cookbook still emits `kernel` / `page_fir …[truncated]

### L1-2fca6d69aa  (L1, 2026-09-14, sha 2fca6d69aa93, PR #39371)
TITLE: bumping sgl-deep-gemm to 0.2.0 (#39371)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py (+9/-0); test/registered/unit/batch_invariant_ops/test_batch_invariant_ops.py (+28/-0)
LABELS: dependencies, run-ci, deterministic
BODY: Recreates [#39275](https://github.com/sgl-project/sglang/pull/39275) after its branch was accidentally deleted. Restores the same head commit (`f33320f18a`) and code diff. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ Test upgrading `sgl-deep-gemm` from 0.1.7 to 0.2.0rc0 while preserving SGLang batch-invariant outputs. DeepGEMM 0.2 defaults BF16 GEMM to cuBLASLt, whose reduction can change with batch size; the batch-invariant override must explicitly select the determin …[truncated]

### L1-7465e42b7a  (L1, 2026-09-14, sha 7465e42b7a12, PR #38833)
TITLE: [NPU][CI] Add CANN 9.1.0 and Ascend a5 nightly suites (#38833)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/_npu-single-node-test-stage.yml (+1/-0); .github/workflows/nightly-test-npu.yml (+166/-3); python/sglang/test/ascend/e2e/test_npu_performance_utils.py (+3/-0); test/registered/npu/accuracy/deepseek_v4_flash/test_npu_deepseek_v4_flash_fp8_4p_gpqa_a5.py (+143/-0); test/registered/npu/accuracy/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_gpqa.py (+6/-0); test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in32k_out1k_50ms.py (+0/-132); test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms.py (+1/-0); test/run_suite.py (+3/-0)
LABELS: deepseek, npu, run-ci
BODY: ## Motivation ⏎  ⏎ The NPU nightly pipeline currently only runs against the CANN 9.0.0 image (`main-cann9.0.0-a3`). We need to validate the DeepSeek-V4-Flash nightly tests against the new CANN 9.1.0 stack, and add a new Ascend a5 (Atlas 950) nightly accuracy suite for the DeepSeek-V4-Flash FP8 4P GPQA test. All original CANN 9.0.0 nightly jobs are kept unchanged and enabled. ⏎  ⏎ ## Modifications ⏎  ⏎ - `.github/workflows/nightly-test-npu.yml` ⏎   - Add two ne …[truncated]

### L1-5dde6e8f02  (L1, 2026-09-15, sha 5dde6e8f02e0, PR #39223)
TITLE: Fix MegaMoE buffer allocation and caching for effective SM budgets (#39223)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.deepgemm_megamoe
FILES: python/sglang/srt/layers/moe/mega_moe.py (+19/-17); test/registered/unit/layers/moe/test_mega_moe_deepgemm_api.py (+85/-3)
LABELS: run-ci, bypass-fastfail, run-ci-extra
BODY: ## Why is this change needed? ⏎  ⏎ DeepGEMM derives MegaMoE buffer size and layout from the runtime SM count ([buffer sizing](https://github.com/deepseek-ai/DeepGEMM/blob/66081d4c9c7d7c44f13fea402e5b622aa0f409c2/csrc/apis/mega_moe.hpp#L44)). SGLang applies an SM budget to kernel execution, but previously allocated outside that context and cached without the effective SM count. ⏎  ⏎ For the CPU test's 132-SM ambient count and 130-SM cap: ⏎  ⏎ ```diff ⏎ - Alloca …[truncated]

### L1-406c9c71d8  (L1, 2026-09-15, sha 406c9c71d833, PR #38160)
TITLE: [Feature] Support BF16 and batch-invariant inference with DeepEP v2 (#38160)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework, L1.runner.deep_gemm, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/kernels/ops/moe/ep_moe_kernels.py (+9/-2); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+26/-6); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+10/-5); python/sglang/srt/layers/moe/moe_runner/runner.py (+5/-3); python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py (+12/-4); python/sglang/srt/layers/moe/utils.py (+19/-0); python/sglang/srt/arg_groups/moe_hook.py (+0/-7); python/sglang/srt/distributed/parallel_state.py (+41/-1); test/manual/ep/test_deepep_v2_deterministic.py (+91/-0); test/registered/kernel/communication/test_deterministic_reduce_scatter.py (+80/-0); (+6 more)
LABELS: run-ci, jit-kernel, bypass-fastfail
BODY: ## Summary ⏎  ⏎ Enable BF16 checkpoints and batch-invariant inference with `--moe-a2a-backend deepep_v2`, and fix a missing-argument error in the existing FP8 prefill path. ⏎  ⏎ The wire dtype follows the expert weights: BF16 experts receive BF16 activations; supported blockwise FP8 experts receive FP8 activations and scales. With `--enable-deterministic-inference`, tokens retain identical results when changes in batch size move them between TP ranks. De …[truncated]

### L1-37ebacb50f  (L1, 2026-09-15, sha 37ebacb50f86, PR #31804)
TITLE: [EPLB] Drop defensive getattr for ep_dispatch_algorithm (#31804)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py, L1.routing.hash_topk
FILES: python/sglang/srt/layers/moe/hash_topk.py (+1/-2); python/sglang/srt/layers/moe/topk.py (+1/-2); test/registered/unit/eplb/test_waterfill_eplb.py (+6/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ `ExpertLocationDispatchInfo.ep_dispatch_algorithm` is a required field of the dataclass, so it is always present on any constructed instance. The two LP-dispatch call sites nonetheless read it through `getattr(expert_location_dispatch_info, "ep_dispatch_algorithm", None)`. This is dead defensive access: the default can never fire, and it would silently swallow an `AttributeError` if the field were ever renamed. Both call sites al …[truncated]

### L1-0dabef3d30  (L1, 2026-09-15, sha 0dabef3d3071, PR #39574)
TITLE: [misc] Fix tool-call index, graph padded-row count, and prefill-graph input_embeds refresh (#39574)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+4/-7); python/sglang/srt/entrypoints/openai/serving_chat.py (+4/-2); python/sglang/srt/model_executor/runner/base_runner.py (+9/-2); python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py (+9/-16); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+7/-4); python/sglang/srt/model_executor/runner_utils/buffers.py (+11/-3); python/sglang/srt/models/deepseek_v2.py (+10/-0); test/registered/unit/entrypoints/openai/test_serving_chat.py (+36/-0)
LABELS: high priority, deepseek, run-ci, bypass-fastfail
DEEP_STUDY: deep-study correctness case sglang:0dabef3d30: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Summary ⏎  ⏎ Five independent fixes in the OpenAI chat server, the CUDA graph runners, and the MoE runners. Each is one or two commits and stands on its own. ⏎  ⏎ ## Changes ⏎  ⏎ **1. Non-streaming `tool_calls[i].index` is the call ordinal.** ⏎ `_process_tool_calls` numbered calls by the detector's `tool_index`, which for most detectors is the tool's position in the request. Two calls to the same tool therefore both got `index=0`, while the streaming deltas …[truncated]

### L1-f920be4b09  (L1, 2026-09-15, sha f920be4b0973, PR #39155)
TITLE: [AMD] GLM-5.2 NextN: cast draft fused MoE to per-channel FP8 (#39155)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py (+51/-43); python/sglang/srt/models/glm4_moe.py (+112/-8); python/sglang/srt/environ.py (+3/-0); test/registered/unit/models/test_glm_nextn_moe_ptpc.py (+160/-0)
LABELS: deepseek, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: Re-submission of #38476 on behalf of @fanxingran. The four original commits retain their original author attribution; the original discussion and comments remain on #38476. ⏎  ⏎ ## Motivation ⏎  ⏎ Layer 78 of GLM-5.2 is the MTP draft layer. The MXFP4 checkpoint ships its routed ⏎ and shared experts in bf16 — 71.7 MB per expert against 19.0 MB for an MXFP4 ⏎ decoder layer — and MTP reads that layer once per draft step. In bandwidth-bound ⏎ decode it theref …[truncated]

### L1-935cbf24ec  (L1, 2026-09-15, sha 935cbf24ec55, PR #39613)
TITLE: Accept MXFP8 dispatch in FlashInfer A2A TRT-LLM MoE (#39613)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+21/-10)
BODY: ## Summary ⏎  ⏎ Serving an MXFP8 model with FlashInfer A2A and `--flashinfer-a2a-dispatch-type mxfp8` dies during init: ⏎  ⏎ ``` ⏎ TypeError: FlashInfer A2A + TRT-LLM Gen FP8 MoE requires a BF16 dispatch payload, got torch.float8_e4m3fn. ⏎ ``` ⏎  ⏎ #31470 made the dispatcher send an already-quantized `float8_e4m3fn` payload with its activation scales, and taught the runner to consume it — but the entry point's BF16-only guard still rejects it. Gate that  …[truncated]

### L1-a64be2e430  (L1, 2026-09-15, sha a64be2e43069, PR #38913)
TITLE: [Kernel] Add H20 block-FP8 MoE configs for GLM-5.3-Flash EP4/EP8 (#38913)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=36,N=2048,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/E=72,N=2048,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); benchmark/kernels/fused_moe_triton/common_utils.py (+1/-0)
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Summary ⏎  ⏎ Add H20 block-FP8 Triton MoE configs for GLM-5.3-Flash TP4/EP4 and TP8/EP8, generated with the official tuner. Recognize `Glm5NextForConditionalGeneration` in the tuner so it reads the nested text config. ⏎  ⏎ ## Performance ⏎  ⏎ NVIDIA H20 96 GB, PyTorch 2.13.0+cu130, Triton 3.7.1. Each baseline/tuned pair uses the same model, runtime and settings; only the two PR MoE configurations are removed for baseline and enabled for tuned. ⏎  ⏎ ### Offici …[truncated]

### L1-faaff1eca8  (L1, 2026-09-15, sha faaff1eca887, PR #39648)
TITLE: dsv4.1: Top-k kernels and candidate selection (#39648)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dsv4/topk.py (+86/-27); python/sglang/kernels/jit/csrc/deepseek_v4/block_amax.cuh (+155/-0); python/sglang/kernels/jit/csrc/deepseek_v4/candidate_block_table.cuh (+218/-0); python/sglang/kernels/jit/csrc/deepseek_v4/topk_bf16_small.cuh (+440/-0); python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh (+239/-162); python/sglang/kernels/jit/csrc/occupancy/cluster_probe.cuh (+55/-0); python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh (+338/-314); python/sglang/kernels/jit/utils/occupancy.py (+47/-0); python/sglang/kernels/ops/attention/dsv4/candidate_table.py (+93/-0); python/sglang/test/kits/dsa_metadata_kit.py (+3/-0); (+2 more)
LABELS: run-ci, jit-kernel
BODY: ## Summary ⏎ - Add the DeepSeek-V4.1 sparse-indexer kernels that sit beside Top-k v2: a bf16 small-row top-k, the two-level indexer's block-max keys and sorted block table, and a cluster-occupancy probe. ⏎ - Rework Top-k v2's edge handling and cluster dispatch (details below); the existing suite still passes and two new regression cases pin the fixed `-inf` tie behaviour. ⏎  ⏎ ## New kernels ⏎ - `topk_bf16_small` (`topk_transform_bf16_small`): bf16 top-k f …[truncated]

### L1-fc5a979f21  (L1, 2026-09-16, sha fc5a979f2121, PR #39678)
TITLE: [misc] Merge FlashInfer autotune caches across spec workers, pad MXFP4 TP shards, drop dead ngram attrs (#39678)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py (+53/-1); python/sglang/srt/layers/quantization/fp8.py (+1/-1); python/sglang/srt/layers/quantization/fp8_utils.py (+2/-1); python/sglang/srt/managers/scheduler.py (+0/-5); python/sglang/srt/model_executor/model_runner_components/weight_updater.py (+1/-1); python/sglang/srt/model_executor/runner/flashinfer_autotune.py (+11/-3)
LABELS: run-ci
BODY: Six independent small fixes, one per commit: merge target/draft FlashInfer autotune caches instead of letting the second load clear the first; pad flashinfer_mxfp4 TP shards to the 128-wide kernel alignment; keep DeepGEMM block-FP8 to 128x128 blocks; drop dead scheduler ngram attributes; encode the FP4 dequant table's negative zero; drop a stray f-string. ⏎  ⏎ ## Test cleanup ⏎  ⏎ The newly added MXFP4 TP-padding regression case was removed in follow-up  …[truncated]

### L1-a3bf25dc62  (L1, 2026-09-16, sha a3bf25dc620f, PR #39763)
TITLE: [AMD] Clamp MORI intranode grid GPUs (#39763)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+9/-0)
LABELS: amd, run-ci
BODY: ## Motivation ⏎  ⏎ this  supports gfx1250 deployment rather than one fixed labtopology. A rack can expose gfx1250 A0 GPUs as 192-CU SPX devices or split eachphysical GPU into two 96-CU DPX agents.  ⏎  ⏎ The MORI intranode path defaults to a 256-block grid. Its standard-MoE dispatch kernel ends with a grid-wide software barrier that requires every block to become resident. gfx1250 GPUs may expose fewer compute units than this grid size; for example, g …[truncated]

### L1-d4ad368ed9  (L1, 2026-09-16, sha d4ad368ed968, PR #28723)
TITLE: [Intel XPU] Enable fused_moe_triton tuning on XPU and add tuned DeepSeek-OCR-2 configs (#28723)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.helper_kernels
FILES: python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py (+5/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_8_0/E=64,N=896,device_name=Intel(R)_Arc(TM)_Pro_B60_Graphics.json (+146/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_8_0/E=64,N=896,device_name=Intel(R)_Arc(TM)_Pro_B60_Graphics_down.json (+164/-0); benchmark/kernels/fused_moe_triton/common_utils.py (+6/-0); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py (+60/-26); python/pyproject_xpu.toml (+6/-0); test/registered/unit/layers/moe/test_fused_moe_common_utils.py (+46/-0)
LABELS: dependencies, intel, xpu, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎ Enable `fused_moe_triton` tuning on Intel XPU and ship tuned DeepSeek-OCR-2 MoE ⏎ configs for Arc Pro B60. ⏎  ⏎ The `_sep` tuner is CUDA-only: graph capture, events, seeding, and timing go ⏎ through `torch.cuda.*` directly, and its ray worker pins every actor to device 0. ⏎ This makes it device-agnostic, fixes tuner issues that a full sweep exposes, and ⏎ adds the resulting `E=64,N=896` configs under `configs/triton_3_8_0/`. ⏎  ⏎ ## Modi …[truncated]

### L1-241a5b9823  (L1, 2026-09-16, sha 241a5b982373, PR #36576)
TITLE: MiniMax-M3: allow shared-experts fusion on ROCm gfx942 and newer (#36576)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-1); python/sglang/srt/models/minimax_m3.py (+4/-2); python/sglang/srt/models/minimax_m3_vl.py (+15/-4)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: # MiniMax-M3: allow shared-experts fusion on ROCm gfx942 and newer ⏎  ⏎ `shared_experts_fusion_disable_reason` currently refuses on any non-CUDA device, so ROCm never gets shared-experts fusion even where the hardware supports it. This lifts that guard and gates it at gfx942 instead. ⏎  ⏎ `minimax_m3_vl.py` carries the identical gate. `MiniMaxM3SparseForCausalLM` and `MiniMaxM3SparseForConditionalGeneration` are separate `EntryClass`es with separate  …[truncated]

### L1-fa8d22e665  (L1, 2026-09-16, sha fa8d22e665dc, PR #39910)
TITLE: [AMD][DSV4] Allow moe_a2a_backend='mori' with DSpark + dp attention (#39910)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/speculative_hook.py (+2/-2)
LABELS: amd, speculative-decoding
BODY: ## Motivation ⏎  ⏎ `_handle_dspark` gates DSpark + dp-attention to `moe_a2a_backend in {"none", "megamoe"}` and rejects everything else at argument-resolution time. This excludes `"mori"`, even though MoRI EP is a fully-supported a2a backend and is compatible with the DSpark draft-verify path. ⏎  ⏎ The exclusion was conservative rather than a real code gap: ⏎  ⏎ - Any non-`"none"` a2a backend on this path already requires `SGLANG_RAGGED_VERIFY_MODE=static`,  …[truncated]

### L1-f0bf652534  (L1, 2026-09-17, sha f0bf652534c5, PR #38526)
TITLE: Add Ling-3.0-flash-VL model support (#38526)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+73/-17); python/sglang/kernels/ops/attention/rotary_triton.py (+2/-2); python/sglang/srt/arg_groups/model_overrides/__init__.py (+1/-0); python/sglang/srt/arg_groups/model_overrides/bailing_moe_v3.py (+48/-0); python/sglang/srt/arg_groups/overrides.py (+2/-0); python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py (+3/-0); python/sglang/srt/configs/__init__.py (+4/-1); python/sglang/srt/configs/bailing_hybrid.py (+79/-1); python/sglang/srt/configs/bailing_moe_v2.py (+172/-0); python/sglang/srt/configs/hybrid_arch.py (+4/-0); (+27 more)
LABELS: run-ci, deterministic, jit-kernel, run-ci-extra
BODY: ## Summary ⏎  ⏎ This PR adds native SGLang support for [`inclusionAI/Ling-3.0-flash-VL`](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL), including text, image, and video serving through the OpenAI-compatible API. It supports the official [BF16](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL), [FP8](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL-fp8), [INT4](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL-int4), and [FP4](https …[truncated]

### L1-aebae58b8c  (L1, 2026-09-17, sha aebae58b8c78, PR #39920)
TITLE: [Moe] Fix flashinfer_trtllm silently dropping swiglu_limit clamped SwiGLU activation (#39920)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8.py (+6/-1)
LABELS: run-ci
ISSUES: #39797 [Question] GLM-5.3-Flash on 4x GB200: `--moe-runner-backend flashinfer_trtllm` scores ~1.3 gsm8k points below `deep_gemm` in repeated runs (6 vs 3 runs); one `triton` run agrees with deep_gemm
BODY: ## Motivation ⏎  ⏎ Models with a clamped SwiGLU activation (e.g. GLM-5.3-Flash / GLM-5-Next, Qwen3-Next style) plumb the clamp limit through `FusedMoE(swiglu_limit=...)` -> `MoeRunnerConfig.swiglu_limit`. All MoE runner backends are supposed to honor it: `triton` passes it into the fused kernel, `deep_gemm` applies it via `silu_and_mul_clamp` / `_apply_swiglu_limit`, but `flashinfer_trtllm` **silently ignores it**. ⏎  ⏎ The flashinfer_trtllm FP8 path …[truncated]

### L1-a98d921658  (L1, 2026-09-17, sha a98d921658b2, PR #39899)
TITLE: [DP Attn] Fix crash for no token all-gather case (#39899)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/logits_processor.py (+8/-3); test/registered/unit/model_executor/test_mlp_sync_pad_unpad.py (+26/-0)
LABELS: run-ci
BODY: ## Why? ⏎  ⏎ With DP attention, DeepEP, DP LM head, `--moe-dense-tp-size 1`, and breakable prefill graphs, one request can crash an idle DP rank: ⏎  ⏎ ```text ⏎ One request routed to rank 0 ⏎   | ⏎   +-> Rank 0: processes the prompt ⏎   | ⏎   +-> Rank 1: IDLE, 0 local tokens ⏎         | ⏎         v ⏎       Prefill graph padding creates a dummy EXTEND request of length 0 ⏎         | ⏎         v ⏎       Empty batch falls back to eager execution ⏎         | ⏎         v ⏎       Logits  …[truncated]

### L1-e970453b43  (L1, 2026-09-17, sha e970453b433e, PR #38420)
TITLE: [NPU]Refactor weight processing and add NPUSwigluLimit activation (#38420)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/activation.py (+29/-0); python/sglang/srt/layers/moe/moe_runner/ascend.py (+9/-1); python/sglang/srt/hardware_backend/npu/quantization/fp4_moe_methods.py (+69/-507); python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py (+107/-43); python/sglang/srt/layers/quantization/fp8.py (+2/-2); test/registered/unit/npu/quantization/test_fp4_moe_methods.py (+151/-341)
LABELS: documentation, npu, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎   The DeepSeek-V4 W4A8 MXFP4 MoE implementation on Ascend maintained a separate execution path for weight preprocessing, routing, grouped ⏎   matrix multiplication, and activation quantization. This duplicated functionality already available in the shared Ascend MoE runner and ⏎   made the quantized MoE path harder to maintain. ⏎  ⏎   This PR refactors the implementation to reuse the shared Ascend MoE infrastructure. It also integrat …[truncated]

### L1-11c35b8433  (L1, 2026-09-17, sha 11c35b8433e8, PR #38878)
TITLE: [AMD] Load fused shared experts for Qwen4-Exp and Qwen3.5 MTP (#38878)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen3_5_mtp.py (+13/-8); python/sglang/srt/models/qwen4_exp.py (+24/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ `Qwen3.8-Flash-Next` loads with its shared experts uninitialized whenever shared-expert fusion is active. There are two independent instances of this, one in the main model and one in the MTP draft. ⏎  ⏎ **Main model.** When fusion is enabled the model has no separate `mlp.shared_expert.*` parameters — the shared expert lives in routed slot `num_experts` of the fused MoE. The checkpoint still ships the tensors separately, and `Qwen4Exp …[truncated]

### L1-c46bf5e990  (L1, 2026-09-18, sha c46bf5e990bd, PR #40105)
TITLE: [MoE] Disable FlashInfer fused finalize by default for numerical accuracy (#40105)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: docs/docs/references/environment_variables.mdx (+2/-2); python/sglang/srt/environ.py (+1/-1)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ @humansand ⏎  ⏎ Default to unfused FlashInfer MoE finalize for better numerical accuracy. Users can opt in to fused atomic finalize with `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=1` when they prefer more aggressive performance. ⏎  ⏎ - **History:** [#28354](https://github.com/sgl-project/sglang/pull/28354) introduced this environment variable with `EnvBool(True)` and made deterministic inference force it to `0`. ⏎ - **Prior behavior:** immediatel …[truncated]

### L1-1e8699fda3  (L1, 2026-09-18, sha 1e8699fda39f, PR #32963)
TITLE: [NVIDIA][comm] Merge EP+MoE-TP post-experts all-reduces into one _TP reduction (#32963)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/__init__.py (+6/-0); python/sglang/srt/layers/moe/utils.py (+67/-24); python/sglang/srt/layers/communicator.py (+14/-11); python/sglang/srt/layers/flashinfer_comm_fusion.py (+35/-20); python/sglang/srt/models/deepseek_v2.py (+5/-18); python/sglang/srt/models/hunyuan_v3.py (+3/-27); python/sglang/srt/models/qwen2_moe.py (+15/-4); python/sglang/srt/models/qwen3_moe.py (+2/-14); test/registered/unit/layers/test_layer_communicator_fusion_gate.py (+230/-13)
LABELS: bug, deepseek, run-ci, bypass-fastfail
BODY: ##   Problem statement ⏎ With `--tp-size 4 --ep-size 2`, the post-experts reduction requires two ⏎   all-reduces over orthogonal groups (`_MOE_EP` then `_MOE_TP`). The allreduce ⏎   fusion gate was skipping *both* once `fuse_mlp_allreduce` was published, then ⏎  ⏎   Observed as GSM8k accuracy 0.012 on `nvidia/DeepSeek-V4-Flash-NVFP4` with ⏎   silently returning under-reduced activations with no error. ⏎  ⏎   Observed as GSM8k accuracy 0.012 on `nvidia/De …[truncated]

### L1-a6cf05817f  (L1, 2026-09-18, sha a6cf05817f11, PR #38798)
TITLE: dsv4.1: remaining model and runtime integration (#38798)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.routing.fused_gate, L1.runner.flashinfer_trtllm
FILES: python/sglang/kernels/ops/moe/moe_fused_gate.py (+139/-22); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+31/-19); python/sglang/srt/layers/moe/mhc_post_fusion.py (+48/-0); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+4/-0); docs/docs/advanced_features/server_arguments.mdx (+6/-0); docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx (+1/-1); python/sglang/kernels/ops/attention/dsv4/__init__.py (+7/-1); python/sglang/kernels/ops/attention/dsv4/c2_decode_pool.py (+151/-0); python/sglang/kernels/ops/attention/dsv4/decode_attention_sm100.py (+160/-0); python/sglang/kernels/ops/attention/dsv4/decode_attention_sm100_gluon.py (+190/-0); (+93 more)
LABELS: documentation, high priority, quant, dependencies, deepseek, hicache, npu, run-ci, jit-kernel, bypass-fastfail
DEEP_STUDY: deep-study: introduced the defect fixed in case sglang:2305242f51 (fix PR 40205)
BODY: - Add DeepSeek V4.1 support. ⏎  ⏎ Please use the image (commit is da64c5cbb8cf6bfd39be19da43573fdfd484c43a) instead of this branch; this branch is being refactored and is unstable. ⏎  ⏎ ## Stack ⏎ - Depends on #39666 (`dsv4.1-engram`). ⏎ - Restacked from `01d34d7b404451bce7f7b35e306de6fab395024d`; the original restack reproduced tree `d529d05f879d36770ad055b2a78dba65a0c675c4`. The standalone NVLink benchmark is now [archived in a gist](https://gist.github. …[truncated]

### L1-826d5170ae  (L1, 2026-09-18, sha 826d5170ae2e, PR #38780)
TITLE: [Bug] Guard FlashInfer CUTLASS MoE against 0-token inputs (#38780)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.runner.flashinfer_cutlass
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py (+4/-1)
LABELS: run-ci, bypass-fastfail, run-ci-extra
ISSUES: #38773 [Bug] flashinfer_mxfp4 crashes on 0-token MoE batches
BODY: TBO + DP attention can produce empty child batches. CUTLASS fused MoE asserts a non-null input_activations pointer, and PyTorch 0-row tensors have data_ptr()==0, which crashes flashinfer_mxfp4. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ Fixes #38773. ⏎  ⏎ FlashInfer CUTLASS flashinfer_mxfp4 asserts on 0-token MoE batches because empty PyTorch tensors have data_ptr() == 0. DP attention + TBO hits this on idle ranks / bs=1 decode (child_a or idle hidden states are [0,  …[truncated]

### L1-0e5347db82  (L1, 2026-09-18, sha 0e5347db8282, PR #40030)
TITLE: Support MXFP8 and deferred route weighting in DeepEP v2 (#40030)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework, L1.runner.deep_gemm, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/kernels/ops/moe/ep_moe_kernels.py (+2/-2); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+21/-5); python/sglang/srt/layers/moe/moe_runner/base.py (+3/-0); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+130/-57); python/sglang/srt/layers/moe/moe_runner/deep_gemm_sm120.py (+1/-0); python/sglang/srt/layers/moe/moe_runner/runner.py (+6/-0); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+4/-0); python/sglang/srt/layers/moe/token_dispatcher/base.py (+36/-0); python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py (+34/-11); python/sglang/kernels/jit/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh (+12/-14); (+10 more)
LABELS: quant, run-ci, jit-kernel
BODY: ## Summary ⏎  ⏎ DeepEP v2 rejects 1x32 MXFP8 experts and applies routing weights before model-specific expert finalization. Support group-32 activation scales through dispatch and DeepGEMM, with unweighted route outputs and an opt-in FP32 SiLU policy for registered model architectures. ⏎  ⏎ ## Motivation ⏎  ⏎ Activation scale groups must be tracked independently of weight scale groups. Models that transform each expert output also need those outputs before r …[truncated]

### L1-8ac39c66d8  (L1, 2026-09-18, sha 8ac39c66d837, PR #39589)
TITLE: [NPU] support kimi k3 on A5 and improve performance (#39589)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/activation.py (+27/-1); python/sglang/srt/layers/moe/moe_runner/ascend.py (+40/-12); python/sglang/kernels/ops/speculative/dspark/dspark_accept.py (+16/-4); python/sglang/srt/arg_groups/field_order.py (+1/-0); python/sglang/srt/arg_groups/fields/parallel.py (+6/-0); python/sglang/srt/arg_groups/parallel_hook.py (+40/-0); python/sglang/srt/arg_groups/pipeline.py (+2/-0); python/sglang/srt/arg_groups/resolution_hooks.py (+1/-0); python/sglang/srt/distributed/bootstrap.py (+1/-0); python/sglang/srt/distributed/parallel_state.py (+50/-2); (+17 more)
LABELS: npu, run-ci, jit-kernel, memory-pool
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Co-Authored-By: [hanwlax](https://github.com/hanwlax) ⏎ Co-Authored-By: [Hexq0210](https://github.com/Hexq0210) ⏎ Co-Authored-By: [McZyWu](https://github.com/McZyWu) ⏎ Co-Authored-By: [qybnb](https://github.com/qybnb) ⏎ Co-Authored-By: [sherdavincl9](https://github.com/sherdavincl9) ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎ support kimi k3 on A5 and improve performance. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ 1.add compressed w4a8 mxfp4 moe. ⏎ 2.shared expert: add fine-grained dual …[truncated]

### L1-d6090f92bf  (L1, 2026-09-18, sha d6090f92bf60, PR #39881)
TITLE: [NPU] Fuse MXFP4 W4A8 MoE gmm1 + swiglu + requant into one kernel (#39881)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/layers/moe/moe_runner/ascend.py (+21/-10); python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py (+61/-0)
LABELS: npu, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ The unfused MXFP4 W4A8 MoE chain runs three kernels per MoE layer for the gate/up projection: gmm1, a separate swiglu activation kernel, and a dynamic MX requant. NPU A5 already provides `npu_grouped_matmul_swiglu_quant_v2`, which folds all three into a single aclnn kernel. ⏎  ⏎ ## Modifications ⏎  ⏎ - `NPUW4A8MXFP4MoEMethod` gains `apply_fused_gmm1_swiglu`, mirroring `NPUMXFP8MoEMethod`: same fused-kernel call shape, with the FP4 we …[truncated]

### L1-7e6d5cbfac  (L1, 2026-09-18, sha 7e6d5cbfac4e, PR #39879)
TITLE: [NPU] Gate DFlash replay metadata refresh behind spec_algorithm check (#39879)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+11/-6); python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py (+127/-91)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ The DFlash support (#37565) added a metadata-refresh block to the common pre-planned replay path of npu_graph_runner.execute() and a seq_lens_cpu_list refresh in _apply_cuda_graph_metadata . Both run unconditionally, which regressed non-DFlash models: ⏎  ⏎ - DSA/DSV4 (+MTP) replay used to be sync-free; the unconditional seq_lens[:bs].cpu() in the backend now forces a D2H sync on every verify replay. ⏎ - npu_graph_runner reads forwar …[truncated]

### L1-6c7c5e78de  (L1, 2026-09-18, sha 6c7c5e78de0a, PR #39823)
TITLE: [NPU] Run arch35 block-FP8 dense linears on the native MXFP8 GEMM (#39823)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/quantization/w8a8_mxfp8.py (+95/-7); python/sglang/srt/layers/quantization/fp8.py (+4/-10)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ On NPU arch35 (A5), plain block-FP8 checkpoints (fp32 128x128 scales, `scale_fmt` unset) fell back to `triton_w8a8_block_fp8_linear` — a CUDA Triton tile GEMM that cannot use the NPU Cube units and runs far slower than the native quantized matmul. The fast native A5 chain (`npu_w8a8_mxfp8_linear`) was only reachable for `scale_fmt="ue8m0"` checkpoints. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add `requant_npu_arch35_block_fp8_to_mxfp8` in `w8a8_ …[truncated]

### L1-d0730a0e8b  (L1, 2026-09-18, sha d0730a0e8bfd, PR #40067)
TITLE: Give the attention-DP width and rank one home (#40067)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/debug_utils/dumper.py (+5/-4); python/sglang/srt/disaggregation/common/conn.py (+2/-6); python/sglang/srt/layers/dp_attention.py (+18/-51); python/sglang/srt/layers/engram.py (+2/-3); python/sglang/srt/managers/cache_controller.py (+1/-2); python/sglang/srt/managers/scheduler_components/dp_attn.py (+1/-2); python/sglang/srt/managers/scheduler_components/dynamic_chunk_sizer.py (+3/-4); python/sglang/srt/mem_cache/storage/umbp/umbp_store.py (+8/-10); python/sglang/srt/models/qwen4_exp.py (+2/-3); python/sglang/srt/runtime_context.py (+47/-11); (+1 more)
LABELS: hicache
BODY: ## Motivation ⏎  ⏎ `attn_dp_size` and `attn_dp_rank` have two homes: module globals in ⏎ `layers/dp_attention.py`, and the parallel namespace on the runtime context. ⏎ Readers are split between the two, so `get_parallel().override(attn_dp_size=...)` ⏎ reaches only half of them — the other half keeps answering with the module ⏎ global. A scope that cannot move every reader of a name is not a scope. ⏎  ⏎ This gives the two names one home, the context, and deletes  …[truncated]

### L1-afe71f4b9e  (L1, 2026-09-18, sha afe71f4b9e9c, PR #40068)
TITLE: Read process groups through the runtime context (#40068)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.hardware.cpu_npu_musa, L1.runner.deep_gemm, L1.runner.flashinfer_trtllm, L1.runner.flashinfer_cutlass, L1.runner.deepgemm_megamoe, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/hardware_backend/npu/moe/fuseep.py (+2/-3); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-8); python/sglang/srt/layers/moe/mega_moe.py (+2/-2); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+7/-6); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py (+5/-3); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+10/-6); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+2/-3); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+2/-2); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+4/-7); python/sglang/srt/layers/moe/topk.py (+2/-5); (+155 more)
LABELS: quant, amd, lora, Multi-modal, deepseek, hicache, mthreads, jit-kernel
BODY: ## Motivation ⏎  ⏎ The same process group can be asked for in two spellings: the free accessors in ⏎ `distributed/parallel_state.py` (`get_tp_group()` and friends) and the parallel ⏎ namespace on the runtime context (`get_parallel().tp_group`). Business code uses ⏎ both, so a scope that swaps a group — the draft worker's tensor-parallel scope, ⏎ for instance — has to be installed in both places to be seen, and any later ⏎ change to how a group is answered has  …[truncated]

### L1-d507accadc  (L1, 2026-09-18, sha d507accadc49, PR #40264)
TITLE: [Test] Drop dead and strictly-subsumed CI test registrations (#40264)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/manual/test_trtllm_mla.py (+0/-6); test/registered/debug_utils/test_dump_comparator.py (+1/-1); test/registered/e2e/models/test_nvidia_nemotron_3_super_bf16.py (+0/-69); test/registered/e2e/models_large/test_deepseek_v3_cutedsl_4gpu.py (+0/-89); test/registered/ep/test_deepep_large.py (+0/-61); test/registered/hicache/test_hicache_storage_file_backend.py (+0/-48); test/registered/hicache/test_hicache_storage_mooncake_backend.py (+0/-32); test/registered/input_embedding/test_input_embeddings.py (+0/-160); test/registered/kernels/benchmark/diffusion/bench_group_norm_silu.py (+1/-2); test/registered/kernels/benchmark/diffusion/bench_norm_impls.py (+1/-2); (+9 more)
LABELS: quant, lora, Multi-modal, deepseek, hicache, npu, run-ci
BODY: Deletion-and-cleanup from a full-tree audit (N² similarity + A1 existence scan across all 1592 registered test files). Every removed item is either never executed in CI, a strict subset of another registration, assertion-free, or a superseded oracle: ⏎  ⏎ **Round 1 (strict subset / disabled / assertion-free):** ⏎ - `test_nvidia_nemotron_3_super_bf16.py`: strict subset of the `_mtp` variant (same args list + EAGLE; gsm8k params byte-identical) ⏎ - `test_l …[truncated]

### L1-81363bf8cb  (L1, 2026-09-18, sha 81363bf8cb54, PR #36176)
TITLE: [kernel] Share the warp vectorized copy and enforce its alignment (#36176)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/moe/route_radix.cuh (+7/-7); .claude/skills/add-jit-kernel/SKILL.md (+46/-9); python/sglang/kernels/jit/benchmark/marker.py (+94/-24); python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh (+0/-14); python/sglang/kernels/jit/csrc/deepseek_v4/candidate_block_table.cuh (+1/-1); python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh (+2/-2); python/sglang/kernels/jit/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh (+1/-11); python/sglang/kernels/jit/csrc/deepseek_v4/topk_bf16_small.cuh (+3/-3); python/sglang/kernels/jit/csrc/elementwise/concat_mla.cuh (+45/-53); python/sglang/kernels/jit/csrc/elementwise/kvcache.cuh (+87/-223); (+24 more)
LABELS: documentation, quant, hicache, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
BODY: > Generated by Claude. ⏎  ⏎ ## Motivation ⏎  ⏎ Four JIT kernels hand-rolled the same thing: a warp cooperatively moving one contiguous row. `concat_mla`, `set_mla_kv_buffer`, `store_cache` and `set_mla_kv_concat_q` each had their own vector-width selection, their own tail handling, and their own idea of what alignment they required. `warp_inclusive_sum` existed in four separate copies. ⏎  ⏎ That duplication was not just untidy — it hid a bug class. The vecto …[truncated]

### L1-7714b182f2  (L1, 2026-09-18, sha 7714b182f223, PR #40187)
TITLE: [Bugfix] Fix top-1 MoE routing with non-unit scaling (#40187)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+5/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ `TestFusedMOE.test_single_expert_routing` intermittently fails on H100 for ⏎ `topk=1` with `routed_scaling_factor=1.5` (for example, ⏎ [this scheduled CI job](https://github.com/sgl-project/sglang/actions/runs/35338273981/job/105578255653)). ⏎  ⏎ For top-1 routing, `_use_intermediate` was false regardless of the routed ⏎ scaling factor, so the down-projection kernel wrote directly to ⏎ `out_hidden_states`. The non-unit scaling path then reduce …[truncated]

### L1-986959e3c4  (L1, 2026-09-19, sha 986959e3c4c9, PR #40197)
TITLE: [Refactor] Deduplicate kernel helpers and remove unused code (#40197)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa, L1.align.cuda_jit, L1.runner.marlin
FILES: python/sglang/kernels/jit/csrc/moe/moe_align_kernel.cu (+9/-27); python/sglang/kernels/jit/csrc/trtllm_lora_temp/moe_lora_merged_align_kernel.cu (+11/-29); python/sglang/kernels/ops/moe/trtllm_lora_temp/virtual_experts.py (+1/-192); python/sglang/srt/hardware_backend/npu/moe/topk.py (+0/-15); python/sglang/srt/lora/marlin_lora_temp/moe_runner.py (+1/-1); python/sglang/kernels/jit/csrc/attention/kda_packed_decode.cuh (+5/-17); python/sglang/kernels/jit/csrc/inkling/inkling_ar_fused_decode.cuh (+2/-2); python/sglang/kernels/jit/csrc/inkling/inkling_ar_scattered_sconv.cuh (+2/-2); python/sglang/kernels/jit/utils/__init__.py (+2/-0); python/sglang/kernels/jit/utils/arch.py (+8/-0); (+18 more)
LABELS: quant, amd, lora, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ Several kernel integrations carry copies of the same TMA builders, warp primitives, alignment code and compiler policy. Fixes then need to be repeated across implementations, while unused fallback/cache helpers obscure the active paths. Consolidate these implementations and remove unused private code without changing the kernel algorithms or dispatch thresholds. ⏎  ⏎ ## Modifications ⏎  ⏎ - Share the chunk and recurrent/output-state TMA de …[truncated]

### L1-9cc7da2ab0  (L1, 2026-09-19, sha 9cc7da2ab0b3, PR #38080)
TITLE: [MegaMoE] Wire Qwen MoE blocks to DeepGEMM MegaMoE (MXFP4 and NVFP4 experts) (#38080)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.deepgemm_megamoe
FILES: python/sglang/srt/arg_groups/mega_moe_hook.py (+172/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-0); python/sglang/srt/layers/moe/mega_moe.py (+105/-23); python/sglang/srt/layers/moe/mega_moe_sm90.py (+10/-13); python/sglang/srt/layers/quantization/modelopt_quant.py (+68/-0); python/sglang/srt/layers/quantization/mxfp4.py (+9/-1); python/sglang/srt/models/qwen2_moe.py (+71/-0); python/sglang/srt/models/qwen3_moe.py (+60/-1); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+0/-85); python/sglang/srt/arg_groups/model_hook.py (+0/-2); (+5 more)
LABELS: quant, deepseek, run-ci, bypass-fastfail
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ `--moe-a2a-backend megamoe` is wired only into `DeepseekV2MoE`, and only for MXFP4 experts. The Qwen MoE blocks (`Qwen2MoeSparseMoeBlock`: Qwen3.5-MoE / Qwen3-Next / Qwen2-MoE; `Qwen3MoeSparseMoeBlock`: Qwen3-MoE / Qwen3-VL-MoE) silently fall back to the standard dispatcher, and on an MXFP4 checkpoint would feed mega-packed weights to the regular FusedMoE runner. Qwen3.5 ships no MXFP4 export, so the useful checkpoint is NVFP4, whi …[truncated]

### L1-cb22f2451e  (L1, 2026-09-19, sha cb22f2451e8e, PR #40265)
TITLE: [Cleanup] Deduplicate kernel tests, diffusion fixtures and benchmark helpers (#40265)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: benchmark/kernels/attention/fa4_benchmark_utils.py (+0/-247); benchmark/kernels/deepep/deepep_utils.py (+0/-218); benchmark/kernels/deepep/tuning_deepep.py (+26/-2); benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py (+5/-83); benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm_blackwell.py (+4/-51); benchmark/kernels/deepseek/benchmark_deepgemm_fp8_group_gemm.py (+3/-36); benchmark/kernels/quantization/tuning_block_wise_kernel.py (+1/-33); benchmark/mmmu/data_utils.py (+0/-43); python/sglang/benchmark/deepseek_utils.py (+65/-0); python/sglang/multimodal_gen/test/unit/sana_wm/test_realtime_chain.py (+2/-21); (+15 more)
LABELS: deepseek, run-ci, diffusion
BODY: ## Motivation ⏎  ⏎ Kernel and diffusion tests repeat input/reference code, fixture setup and whole test bodies. Some benchmarks also carry copied or unused helpers and stale import paths. This cleanup removes 1,426 net lines across 26 files while retaining the numerical checks, backend-specific dispatch, input matrices and CI registrations. ⏎  ⏎ ## Modifications ⏎  ⏎ - Share paged-MQA fixtures, PyTorch reference and comparison logic between CuTe DSL and Deep …[truncated]

### L1-7fac84b639  (L1, 2026-09-19, sha 7fac84b6391b, PR #39704)
TITLE: [DSV4.1] Reduce mHC, metadata and small-batch router overhead (#39704)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: python/sglang/kernels/ops/moe/moe_fused_gate.py (+12/-1); python/sglang/srt/layers/moe/mhc_post_fusion.py (+9/-0); python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py (+37/-8); python/sglang/kernels/jit/csrc/deepseek_v4/mhc_post_combine_norm_prefill.cuh (+190/-0); python/sglang/kernels/jit/csrc/distributed/all_reduce_fusion.cuh (+49/-5); python/sglang/kernels/ops/attention/dsv4/fp4_indexer.py (+65/-1); python/sglang/kernels/ops/communication/all_reduce_mhc_combine.py (+121/-0); python/sglang/kernels/ops/layernorm/mhc_post_combine.py (+85/-0); python/sglang/kernels/ops/layernorm/mhc_post_combine_norm_prefill.py (+38/-0); python/sglang/kernels/ops/speculative/dspark/dspark_accept.py (+4/-1); (+5 more)
LABELS: deepseek, run-ci, jit-kernel, run-ci-extra, mergeable
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: > Retargeted to `main` at `a6cf05817f11d22023fd951a76255ef50fb09f49`, after #38798 and #40039 landed. The diff contains only this PR's optimizations (15 files), including the router PDL change previously consolidated from #39941. ⏎ > ⏎ > Conflict resolution preserves main's `post_experts_all_reduce` topology handling and the DSPARK fused-argmax opt-in, while retaining the mHC overlap/fusion changes. Local validation: all pre-commit checks on the chan …[truncated]

### L1-83e29d6c5a  (L1, 2026-09-19, sha 83e29d6c5aed, PR #38220)
TITLE: [perf] Optimize w4a8 MoE for glm5.2 on H200 (#38220)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.cutlass.w4a8
FILES: python/sglang/kernels/aot/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu (+89/-0)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR ()
BODY: ## Motivation ⏎  ⏎  ⏎  `dispatch_w4a8_moe_mm_sm90 (python/sglang/kernels/aot/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu)` picks a CUTLASS config for the W4A8 grouped GEMM by matching on (n, k). Every named branch was tuned for DeepSeek's 7168-hidden shapes on H20. GLM-5.2 (hidden=6144, moe_intermediate_size=2048, E=256, topk=8) matches none of them, so both of its production sharding layouts - TP-8 and EP-8 without DeepEP - fall through to the …[truncated]

### L1-d903351a66  (L1, 2026-09-20, sha d903351a669d, PR #38831)
TITLE: [NPU][bugfix] update low latency quantization input and update MXFP8 tests (#38831)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+86/-69); python/sglang/srt/layers/moe/utils.py (+2/-0); test/registered/unit/npu/quantization/test_fp4_moe_methods.py (+430/-0)
LABELS: run-ci, unified-radix-cache
BODY: ## Summary ⏎  ⏎ Align SGLang's DeepEP dispatch quantization interface with the explicit bool flags exposed by the current runtime. ⏎  ⏎ - Derive `use_fp8`, `use_mxfp4`, and `use_mxfp8` from the dispatcher output dtype. ⏎ - Forward those flags unchanged through both normal and low-latency dispatch. ⏎ - Add the `mxfp4` dispatcher output dtype and map it to `use_mxfp4=True`. ⏎ - Keep `use_fp8=True` as the cross-architecture request: DeepEP selects INT8 on  …[truncated]

### L1-983e643854  (L1, 2026-09-20, sha 983e643854f1, PR #40448)
TITLE: [Feature] support bf16 MoE router and mxfp4 MoE for MiMo V2 (#40448)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+6/-1); python/sglang/srt/models/dflash.py (+2/-1); python/sglang/srt/models/mimo_v2.py (+18/-4)
LABELS: run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎ Day0 support for MiMo-V2.6/MiMo-V2.6-Pro in SGLang. ⏎  ⏎ - mxfp4 MoE ⏎ - bf16 MoE router ⏎ - DFlash fo spec decoding ⏎  ⏎  ⏎ ## Launch command examples ⏎ MiMo V2.6 Flash model weights: https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL ⏎ MiMo V2.6 Pro model weights: https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL ⏎ ### **MiMo V2 flash:** ⏎ ``` ⏎ python3 -m sglang.launch_server \ ⏎               --model-path /model/MiMo-V2.5-FP4-DFlash/ \ ⏎  …[truncated]

### L1-42875bcd2a  (L1, 2026-09-20, sha 42875bcd2a7f, PR #38932)
TITLE: fix(modelopt): dispatch NVFP4 MoE on the cached backend, not the live global (#38932)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/modelopt_quant.py (+8/-4); test/registered/unit/layers/quantization/test_modelopt_nvfp4_moe_dispatch.py (+186/-0)
LABELS: bug, quant, blackwell, run-ci, bypass-fastfail
ISSUES: #38795 [Bug] NVFP4 + flashinfer_cutlass + --speculative-adaptive: CUDA-graph capture raises "Unsupported moe_runner_backend ... Use flashinfer_cutlass instead" for the backend already in use
BODY: ## Motivation ⏎  ⏎ Fixes #38795. ⏎  ⏎ `ModelOptNvFp4FusedMoEMethod.apply` resolves the MoE backend twice from two different sources, and under speculative decoding they disagree. ⏎  ⏎ `create_weights` caches it: ⏎  ⏎ ```python ⏎ self._moe_runner_backend = moe_runner_backend ⏎ ``` ⏎  ⏎ `apply` reads that cache for the error message and for the marlin/megamoe checks: ⏎  ⏎ ```python ⏎ moe_runner_backend = getattr(self, "_moe_runner_backend", get_moe_runner_backend()) ⏎ ``` ⏎  ⏎ but th …[truncated]

### L1-5f017ffabb  (L1, 2026-09-20, sha 5f017ffabb6a, PR #40392)
TITLE: Update test cases and performance testing framework (#40392)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/test/ascend/e2e/test_npu_performance_utils.py (+64/-0); test/registered/npu/accuracy/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_gpqa.py (+46/-10); test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms.py (+52/-18)
LABELS: deepseek, npu, run-ci
BODY: ## Motivation ⏎  ⏎ Update the DeepSeek-V4-Flash W8A8 8P accuracy/performance test cases on Ascend NPU to match the latest deployment configuration (DSPARK speculative decoding with 7 draft tokens, 192 max running requests, round-robin load balancing, enlarged DeepEP buffers), and extend the NPU performance test framework with an accept_rate baseline assertion.The method of starting the service is changed to python -m sglang.launch_server, which avo …[truncated]

### L1-d97aed2c90  (L1, 2026-09-21, sha d97aed2c908d, PR #40163)
TITLE: Fix TopK v2 fallback when 16-block cluster capacity is zero (#40163)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dsv4/topk.py (+15/-17); python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh (+2/-1); python/sglang/kernels/jit/utils/occupancy.py (+9/-3)
LABELS: run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ TopK v2 can launch an unsupported 16-block cluster even when the device occupancy probe reports zero capacity. The loader swallows the zero-capacity exception and retains the default C16 capacity of `7`. ⏎  ⏎ On NVIDIA H20, this was reproduced with `batch=4`, `seq_len=65536`, and `topk=512`, which failed with `cluster misconfiguration`. ⏎  ⏎ This PR passes the zero-capacity result into JIT compilation so the unsupported path is disabled an …[truncated]

### L1-3c71bb018a  (L1, 2026-09-21, sha 3c71bb018aa5, PR #37889)
TITLE: [AMD] Enable GLM DSA prefill top-k to the v2 kernel (#37889)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dsv4/topk.py (+59/-5); python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh (+185/-0); python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py (+87/-0); python/sglang/srt/layers/attention/dsa_backend.py (+3/-3); test/registered/kernels/ops/attention/test_topk_v2.py (+143/-0)
LABELS: amd, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ **Add a packed-row top-k kernel to the DSA top-k v2 family and route GLM-5.x prefill through it. The prefill top-k kernel drops ~66% per launch; at ISL 70000 / OSL 300 that is +4.9% token throughput per GPU, −3.5% median TPOT and −4.8% median TTFT (geomean over concurrency 4-64); GSM8k 0.937 against 0.935 on the baseline.** ⏎  ⏎ <img width="1692" height="965" alt="image" src="https://github.com/user-attachments/assets/0ccb560a-cf57 …[truncated]

### L1-ab03a8e7eb  (L1, 2026-09-21, sha ab03a8e7eb82, PR #40201)
TITLE: [Perf] Fork-safe import: no CUDA context at import time, lighter argument parsing (#40201)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/cli/utils.py (+12/-2); python/sglang/srt/distributed/parallel_state.py (+3/-1); python/sglang/srt/function_call/parser_names.py (+51/-0); python/sglang/srt/parser/reasoning_parser_names.py (+39/-0); python/sglang/srt/server_args.py (+51/-6); python/sglang/srt/utils/common.py (+62/-2); python/sglang/srt/utils/hf_transformers_patches.py (+4/-1); test/registered/unit/function_call/test_function_call_parser.py (+11/-0); test/registered/unit/parser/test_reasoning_parser.py (+10/-0); test/registered/unit/server_args/test_server_args.py (+67/-0); (+1 more)
LABELS: run-ci, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Following #39683, decompose it into a stack of PRs for better management. ⏎  ⏎ # **WHAT IS IN THIS PR:** ⏎  ⏎ Motivation: importing sglang created a CUDA context in every process through device probes, and `server_args` pulled the parser registries and `sglang.kernels` into every process that parses arguments. A process with a CUDA context cannot be a `fork()` parent, which rules out forkserver-based worker startup (PR 2), and the import cost lands o …[truncated]

### L1-90b3f8544c  (L1, 2026-09-21, sha 90b3f8544ca2, PR #39986)
TITLE: [AMD] Use Triton softmax routing for Qwen3.5 on gfx950 (#39986)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+36/-2); test/registered/amd/test_qwen35_moe_softmax_topk.py (+127/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ Qwen3.5-397B-A17B-MXFP4 spends about 0.59 ms of every TP4 target-verify pass in 60 AITER `topkGatingSoftmax` launches. SGLang already has a numerically matching Triton softmax router that runs on ROCm, but the AITER softmax branch always selects `aiter_fused_topk` first. On MI355X, the Triton router is 38-42% faster in isolation and cuts the routed portion of a target-verify graph by 51%. ⏎  ⏎ This change selects the Triton router only …[truncated]

### L1-b63f8416b3  (L1, 2026-09-21, sha b63f8416b3b7, PR #29189)
TITLE: [Feature] Gigachat 3.5 support (#29189)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs/docs/advanced_features/server_arguments.mdx (+2/-2); docs/docs/supported-models/generative_models.mdx (+5/-0); python/sglang/srt/arg_groups/model_overrides/__init__.py (+1/-0); python/sglang/srt/arg_groups/model_overrides/gigachat35.py (+26/-0); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/gigachat35.py (+252/-0); python/sglang/srt/configs/model_config.py (+7/-0); python/sglang/srt/function_call/function_call_parser.py (+2/-0); python/sglang/srt/function_call/gigachat35_detector.py (+151/-0); python/sglang/srt/layers/attention/attention_registry.py (+1/-2); (+7 more)
LABELS: documentation, run-ci
BODY: ## Motivation ⏎  ⏎ GigaChat-3.5-432B-A28B is a Mixture-of-Experts (MoE) language model with 432B total parameters and 28B active parameters. Built on a DeepSeek-V3-style backbone (MLA attention + DeepSeek MoE), it employs a hybrid attention architecture in which most layers use a Qwen3-Next Gated-Delta-Net (GDN) linear-attention block while a periodic subset retains full MLA attention. The architecture further incorporates gated RMSNorm with a low- …[truncated]

### L1-0c53fec476  (L1, 2026-09-21, sha 0c53fec4768a, PR #39775)
TITLE: [ROCm] fix: remove extra bf16 -> fp32 cast in jit grouped topk kernel path (#39775)
SOURCES: path_core, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+4/-0)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ On ROCm with `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1`, every MoE layer pays an ⏎ avoidable `bfloat16 -> float32` copy kernel on the routing bias. ⏎  ⏎ `select_experts` unconditionally converts `correction_bias` into the gating dtype ⏎ whenever `_use_aiter and use_grouped_topk`: ⏎  ⏎ ```python ⏎     if _use_aiter and use_grouped_topk and correction_bias is not None: ⏎         correction_bias = topk_config.correction_bias_for_dtype(route …[truncated]

### L1-56fee88e23  (L1, 2026-09-21, sha 56fee88e236b, PR #35504)
TITLE: fix(moe): support Llama4 NVFP4 router input weights on SM120 (#35504)
SOURCES: path_core, path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L1.runner.flashinfer_cutlass
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py (+32/-7); python/sglang/srt/layers/quantization/modelopt_quant.py (+0/-4)
LABELS: bug, quant, run-ci
ISSUES: #34192 [Bug] Llama4 NVFP4 MoE crashes on SM120/SM121: apply_router_weight_on_input is not supported for Flashinfer
BODY: ## Motivation ⏎  ⏎ Fixes https://github.com/sgl-project/sglang/issues/34192. ⏎  ⏎ Llama 4 constructs its MoE layers with `apply_router_weight_on_input=True`. For ModelOpt NVFP4 on SM120/SM121, SGLang selects the FlashInfer CUTLASS MoE runner, which previously rejected this configuration during server warmup. ⏎  ⏎ For top-1 routing, the required semantics can be implemented exactly by multiplying each token's hidden states by its router weight before ac …[truncated]

### L1-69d1e5cfe0  (L1, 2026-09-21, sha 69d1e5cfe06e, PR #40577)
TITLE: [Docs][NPU] Add MiMo-V2.5-Pro FP4 DFlash best practice on Ascend NPU (#40577)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/docs.json (+5/-0); docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/mimo_v2_5_pro.mdx (+170/-0)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ MiMo-V2.5-Pro (FP4) is now supported on Ascend NPUs with DFlash speculative decoding and PD disaggregation, but there is no deployment documentation for it. This PR adds a best-practice guide so users can deploy the model with a verified configuration. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add `docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/mimo_v2_5_pro.mdx`, covering: ⏎   - Common environment setup for both nodes (C …[truncated]

### L1-b44e248682  (L1, 2026-09-21, sha b44e2486824e, PR #38546)
TITLE: [AMD] [GLM-5.3-Flash Day 0] Enable FP8 and Quark MXFP4 MoE on gfx950 (#38546)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.runner.aiter, L1.upstream.aiter_moe
FILES: python/sglang/srt/layers/moe/moe_runner/aiter.py (+7/-4); python/sglang/srt/layers/quantization/fp8.py (+22/-0); python/sglang/srt/layers/quantization/quark/quark.py (+56/-3); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py (+24/-4); python/sglang/srt/models/glm5_next.py (+17/-3); test/registered/e2e/moe/test_glm53_flash_quark_moe_mi35x.py (+397/-0); test/registered/unit/layers/quantization/test_fp8_moe_runner_ownership.py (+48/-0); test/registered/unit/layers/quantization/test_quark_config.py (+185/-1)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Summary ⏎  ⏎ Replacement for #38037, which GitHub closed when its deleted support base was merged into main. ⏎  ⏎ Replacement for #37629, which was accidentally squash-merged and then reverted by #37880. This reapplies the same reviewed change on the current support-branch head. ⏎  ⏎ GLM-5.3-Flash ships two gfx950 MoE checkpoint paths that share one model architecture but use different quantization contracts: ⏎  ⏎ - `zai-org/GLM-5.3-Flash`: routed, shared, an …[truncated]

### L1-7b977ce5dc  (L1, 2026-09-22, sha 7b977ce5dcce, PR #40672)
TITLE: [Fix] Decide the MoE padded-row bound from the layer scatter mode (#40672)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.deepgemm_megamoe
FILES: python/sglang/srt/layers/moe/mega_moe.py (+1/-1); python/sglang/srt/layers/communicator.py (+19/-13); python/sglang/srt/model_executor/forward_batch_info.py (+44/-0); python/sglang/srt/models/bailing_moe.py (+1/-1); python/sglang/srt/models/bailing_moe_v3.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+5/-3); python/sglang/srt/models/deepseek_v4.py (+11/-19); python/sglang/srt/models/dots3_common/modeling.py (+2/-2); python/sglang/srt/models/exaone_moe.py (+1/-1); python/sglang/srt/models/glm4_moe.py (+2/-2); (+11 more)
LABELS: deepseek, run-ci, run-ci-extra, bypass-fail-fast, highest-priority
BODY: ## Summary ⏎  ⏎ A more general fix for the truncation reported in #40643, which #39574 introduced: rather than special-casing DP attention, the bound is decided from the layer communicator's scatter mode, which also covers the non-DP and CP gathered cases and the other MoE models on the same path. ⏎  ⏎ `forward_batch.num_token_non_padded` is **LOCAL** (`forward_batch_info.py`). It bounds a sparse MoE's input only while that input is this rank's own s …[truncated]

### L1-78980a3b0b  (L1, 2026-09-22, sha 78980a3b0bdd, PR #40795)
TITLE: [misc] Remove deprecated endpoints, env vars and aliases past two releases (#40795)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: .claude/skills/env-var-conventions/SKILL.md (+1/-1); .claude/skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md (+4/-4); docs/cookbook/autoregressive/GLM/GLM-4.5V.mdx (+1/-2); docs/cookbook/autoregressive/GLM/GLM-4.6V.mdx (+1/-2); docs/cookbook/autoregressive/Google/Gemma4.mdx (+3/-3); docs/cookbook/autoregressive/Qwen/Qwen3-VL.mdx (+1/-1); docs/cookbook/autoregressive/Qwen/Qwen3.5.mdx (+0/-1); docs/docs/advanced_features/server_arguments.mdx (+0/-12); docs/docs/developer_guide/benchmark_and_profiling.mdx (+8/-8); docs/docs/developer_guide/development_guide_using_docker.mdx (+1/-1); (+173 more)
LABELS: documentation, quant, amd, lora, Multi-modal, deepseek, speculative-decoding, hicache, sgl-kernel, blackwell
BODY: Remove deprecations that have shipped for at least two releases and have no in-repo users (dead shims, deprecated HTTP endpoints, env aliases, API aliases). ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #35799988862](https://github.com/sgl-project/sglang/actions/runs/35799988862) ⏎ Latest PR Test (Extra): :x: [Run #35799987967](https://github.com/sgl-project/sglang/actions/runs/35799987967) ⏎ Latest PR Test (AMD ROCm 10): :hourglass …[truncated]

### L1-28be39f72e  (L1, 2026-09-22, sha 28be39f72e7d, PR #40466)
TITLE: [deepep_v2] support GLM-5.3-Flash (Glm5NextForConditionalGeneration) (#40466)
SOURCES: subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/moe_model_registry.py (+1/-0); python/sglang/srt/models/glm5_next.py (+5/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ `--moe-a2a-backend deepep_v2` cannot be used with GLM-5.3-Flash ⏎ (`Glm5NextForConditionalGeneration`). There are two independent blockers. ⏎  ⏎ **1. The architecture is not registered as v2-eligible.** ⏎ `moe_model_registry._DEEPEP_V2_MODELS` does not list it, so ⏎ `validate_deepep_v2_model_architecture()` rejects the server args before the ⏎ engine starts. ⏎  ⏎ **2. `glm5_next.py` gates the a2a path on a v1-only predicate.** ⏎ `MoeA2ABackend.is_de …[truncated]

### L1-de123f38bb  (L1, 2026-09-23, sha de123f38bbc9, PR #33723)
TITLE: [3/N] elastic-ep: Recapture decode CUDA graphs after scale-up (#33723)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/nixl.py (+1/-1); python/sglang/srt/arg_groups/parallel_hook.py (+31/-10); python/sglang/srt/elastic_ep/elastic_ep.py (+30/-12); python/sglang/srt/layers/attention/base_attn_backend.py (+3/-0); python/sglang/srt/layers/attention/flashattention_backend.py (+9/-3); python/sglang/srt/layers/attention/hybrid_attn_backend.py (+3/-0); python/sglang/srt/layers/attention/tbo_backend.py (+4/-0); python/sglang/srt/layers/communicator.py (+10/-2); python/sglang/srt/layers/dp_attention.py (+9/-0); python/sglang/srt/managers/tp_worker.py (+7/-0); (+8 more)
LABELS: run-ci, bypass-fastfail
BODY: ## Summary ⏎  ⏎ This PR follows [PR #30553](https://github.com/sgl-project/sglang/pull/30553) ⏎ and adds FULL decode CUDA graph recapture after runtime Elastic EP scale-up. ⏎ Primary and joiner ranks rebuild their graphs for the expanded topology before ⏎ the new EP size is committed. ⏎  ⏎ The existing scale-up API and EPLB lifecycle are unchanged. ⏎  ⏎ ## Changes ⏎  ⏎ - Defer initial decode graph capture on joining ranks until they enter the ⏎   expanded wo …[truncated]

### L1-401d5aedf3  (L1, 2026-09-23, sha 401d5aedf391, PR #40879)
TITLE: [AMD] Drop the unreachable vLLM fallback from ROCm FP8 activation quant (#40879)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/quantization/fp8_kernel.py (+13/-36)
LABELS: jit-kernel
BODY: ## Motivation ⏎  ⏎ Follow-up to #40557: The reviewer asked why `_has_vllm` is still kept. ⏎  ⏎ `_has_vllm` in `fp8_kernel.py` is only ever set inside `if _is_hip:`, and only in the arm where `_use_aiter` is false: ⏎  ⏎ ```python ⏎ if _is_hip: ⏎     _has_vllm = False ⏎     if _use_aiter: ⏎         from aiter import dynamic_per_tensor_quant, ... ⏎     else: ⏎         try: ⏎             import vllm._C ⏎             _has_vllm = True ⏎         except ImportError: ⏎   …[truncated]

### L1-40048f6e51  (L1, 2026-09-23, sha 40048f6e5189, PR #34061)
TITLE: [dLLM] feat: support DiffusionGemma serving (#34061)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+20/-7); benchmark/dllm/README.md (+27/-0); benchmark/dllm/bench_diffusion_gemma.py (+166/-0); python/sglang/srt/arg_groups/dllm_hook.py (+4/-0); python/sglang/srt/arg_groups/model_overrides/gemma4.py (+10/-1); python/sglang/srt/arg_groups/overrides.py (+17/-0); python/sglang/srt/configs/model_config.py (+4/-0); python/sglang/srt/dllm/algorithm/__init__.py (+8/-5); python/sglang/srt/dllm/algorithm/base.py (+36/-5); python/sglang/srt/dllm/algorithm/gemma4_renoise.py (+364/-0); (+22 more)
LABELS: documentation, quant, Multi-modal, run-ci, run-ci-extra, bypass-fail-fast, highest-priority
ISSUES: #29904 [Bug] Unknown diffusion LLM: DiffusionGemmaForBlockDiffusion
BODY: ## Motivation ⏎  ⏎ SGLang rejects the DiffusionGemma architecture as unknown. This PR adds native BF16 text and image serving using the shared dLLM and FDFO paths from #27551 and #27877. ⏎  ⏎ Fixes #29904. ⏎  ⏎ ## Changes ⏎  ⏎ - Add the model, multimodal processor, weight loader, and Gemma4Renoise sampler, with TP1, TP2, and TP4 support. ⏎ - Encode causal context separately from each bidirectional 256-position denoising canvas, retaining request KV state  …[truncated]

### L1-2d25767759  (L1, 2026-09-23, sha 2d2576775981, PR #28417)
TITLE:  [NPU] Enable piecewise CUDA graph support on NPU (#28417)
SOURCES: path_core
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+11/-2); python/sglang/kernels/ops/attention/fla/layernorm_gated.py (+66/-1); python/sglang/kernels/ops/elementwise/elementwise.py (+11/-1); python/sglang/srt/arg_groups/cuda_graph_hook.py (+5/-1); python/sglang/srt/compilation/npu_piecewise_backend.py (+16/-1); python/sglang/srt/environ.py (+1/-0); python/sglang/srt/hardware_backend/npu/cmo.py (+50/-23); python/sglang/srt/models/qwen2_moe.py (+1/-1); python/sglang/srt/models/qwen3.py (+53/-3); python/sglang/srt/models/qwen3_5.py (+9/-3); (+2 more)
LABELS: npu, run-ci, piecewise-cuda-graph, jit-kernel
DEEP_STUDY: deep-study: this PR was reverted by PR 40895 (confirmed_revert, reason=other) || deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Piecewise CUDA Graph (PCG) was disabled on NPU, so NPU prefill could not use the ⏎ torch.compile-driven piecewise capture/replay path. Enabling it also requires the NPU ⏎ MoE, fused QKV/RMSNorm/RoPE, elementwise, and FLA layernorm-gated helpers to be visible ⏎ to Dynamo as traceable custom ops instead of unsupported Python or external-kernel code. ⏎  ⏎ This PR enables NPU PCG behind the opt-in ⏎ `SGLANG_NPU_ENABLE_PIECEWISE_CUDA_GRAPH= …[truncated]

### L1-77983865d8  (L1, 2026-09-23, sha 77983865d892, PR #39939)
TITLE: [Moe] Honor swiglu_limit clamped activation in flashinfer_cutlass runner (#39939)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.runner.flashinfer_cutlass
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py (+46/-0); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py (+15/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+33/-0); python/sglang/srt/layers/quantization/unquant.py (+15/-0); test/registered/unit/layers/moe/test_flashinfer_cutlass_swiglu_params.py (+98/-0)
LABELS: quant, blackwell, run-ci
BODY: ## Motivation ⏎  ⏎ Models with a clamped SwiGLU activation (GLM-5 / GLM-5-Next, Qwen3-Next, GPT-OSS; configured as `swiglu_limit` or `gemm1_clamp_limit` in `MoeRunnerConfig`) silently lose the clamp on the `flashinfer_cutlass` MoE runner and run plain SwiGLU experts, for two stacked reasons: the runner's `_run_flashinfer_cutlass` never passed the per-expert SwiGLU parameters to the kernel, and its quant-info construction sites never filled them. ⏎  …[truncated]

### L1-e2f4fedf04  (L1, 2026-09-23, sha e2f4fedf04fe, PR #40754)
TITLE: [AMD] Critical fix enabling Qwen3.8 FP8: restore dropped fused shared-expert weights (#40754)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen3_5.py (+101/-3)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Text-only Qwen3.5-MoE checkpoints (architecture `Qwen3_5MoeForCausalLM`, e.g. `Qwen3.8-2.4T-A95B-FP8`) load into `Qwen3_5MoeForCausalLM`, whose `load_weights` was missing the shared-expert fusion remapping that the multimodal `Qwen3_5MoeForConditionalGeneration.load_weights` already implements. On ROCm with aiter shared-expert fusion enabled (`SGLANG_USE_AITER=1`), `Qwen2MoeSparseMoeBlock` serves the shared expert as an extra fused …[truncated]

### L1-3fdd63a562  (L1, 2026-09-23, sha 3fdd63a5620d, PR #40445)
TITLE: [NPU] Fuse FIA KV-cache K/V writes into one npu_scatter_pa_kv_cache call (#40445)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/environ.py (+4/-0); python/sglang/srt/hardware_backend/npu/memory_pool_npu.py (+60/-16)
LABELS: npu, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 41132 (confirmed_revert, reason=unstated) || deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎ On the NPU FIA path ( ASCEND_USE_FIA=1 ), NPUMHATokenToKVPool.set_kv_buffer writes K and V into the paged KV cache with two separate npu_scatter_nd_update_ kernels — one per tensor, indexed by the same out_cache_loc . That is two kernel launches per layer on the decode critical path. ⏎  ⏎ MiMo-V2.5-Pro EP6 decode profiling (msprof, DeepEP LL, 57 MoE layers) shows the two aclnnScatterNdUpdate kernels cost ~11.1us/layer. ⏎  ⏎ torch_npu p …[truncated]

### L1-16e353d93e  (L1, 2026-09-23, sha 16e353d93e4e, PR #40438)
TITLE: [NPU] Skip fused gmm1+swiglu for swiglu_limit (SiLU-with-clamp) checkpoints (#40438)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/layers/moe/moe_runner/ascend.py (+22/-4)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ Follow-up to #39881. The fused gmm1 kernel ( npu_grouped_matmul_swiglu_quant_v2 ) applies plain SiLU with no clamp , but _uses_fused_gmm1 selected it for W4A8 MXFP4 MoE based only on the kernel type and use_fused_gmm1 — it never looked at the activation configuration. ⏎  ⏎ Checkpoints whose MoE activation is SiLU-with-clamp ( swiglu_limit , e.g. DSV4) therefore lost the clamp when the fused kernel was selected, producing incorrect  …[truncated]

### L1-32290dda2c  (L1, 2026-09-23, sha 32290dda2cea, PR #40204)
TITLE: [AMD] Small-M MXFP4 fused-MoE kernel for gfx950 (Qwen) (#40204)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.runner.aiter, L1.upstream.aiter_moe
FILES: python/sglang/kernels/ops/moe/smallm_moe_gfx950/__init__.py (+369/-0); python/sglang/kernels/ops/moe/smallm_moe_gfx950/smallm_moe.hip (+378/-0); python/sglang/srt/layers/moe/moe_runner/aiter.py (+36/-0); test/registered/amd/test_smallm_moe_gfx950.py (+177/-0)
LABELS: amd, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ For decode batches of 1 to ~40 tokens per rank, the MXFP4 MoE is the biggest cost per step on MI355X, and AITER ⏎ `fused_moe` only uses a small part of the HBM bandwidth there. This PR adds a HIP kernel pair for these shapes; above 40 tokens AITER is still used. ⏎  ⏎ ## Modifications ⏎  ⏎ - New `python/sglang/kernels/ops/moe/smallm_moe_gfx950/` (`__init__.py`, `smallm_moe.hip`): built with `hipcc` on ⏎   first use and launched through  …[truncated]

### L1-5c154c214d  (L1, 2026-09-23, sha 5c154c214df5, PR #40895)
TITLE: Revert " [NPU] Enable piecewise CUDA graph support on NPU" (#40895)
SOURCES: path_core
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+2/-11); python/sglang/kernels/ops/attention/fla/layernorm_gated.py (+1/-66); python/sglang/kernels/ops/elementwise/elementwise.py (+1/-11); python/sglang/srt/arg_groups/cuda_graph_hook.py (+1/-5); python/sglang/srt/compilation/npu_piecewise_backend.py (+1/-16); python/sglang/srt/environ.py (+0/-1); python/sglang/srt/hardware_backend/npu/cmo.py (+23/-50); python/sglang/srt/models/qwen2_moe.py (+1/-1); python/sglang/srt/models/qwen3.py (+3/-53); python/sglang/srt/models/qwen3_5.py (+3/-9); (+2 more)
LABELS: npu, run-ci, piecewise-cuda-graph, jit-kernel
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 28417 reason=other
BODY: Reverts sgl-project/sglang#28417 ⏎ NPU will support BCG only, PCG support will be dropped ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #35845757908](https://github.com/sgl-project/sglang/actions/runs/35845757908) ⏎ Latest PR Test (Extra): :x: [Run #35845757494](https://github.com/sgl-project/sglang/actions/runs/35845757494) ⏎ Latest PR Test (AMD ROCm 10): :no_entry_sign: [Run #35845757640](https://github.com/sgl-project/sglang/actio …[truncated]

### L1-3177d10ca6  (L1, 2026-09-24, sha 3177d10ca6f7, PR #39816)
TITLE: Refactor the Cute-DSL AR fusion to support DeepseekV2 archs (GLM-5.3, etc.) (#39816)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/cutedsl_ar_fusion.py (+478/-0); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+1/-7); python/sglang/srt/layers/moe/qwen35_flashinfer_fusion.py (+0/-350); python/sglang/srt/arg_groups/fields/exec_.py (+4/-1); python/sglang/srt/arg_groups/moe_hook.py (+1/-0); python/sglang/srt/arg_groups/overrides.py (+0/-14); python/sglang/srt/distributed/bootstrap.py (+4/-0); python/sglang/srt/environ.py (+15/-7); python/sglang/srt/layers/communicator.py (+32/-3); python/sglang/srt/layers/flashinfer_comm_fusion.py (+14/-1); (+10 more)
LABELS: documentation, deepseek, blackwell, run-ci, run-ci-extra, bypass-fail-fast, parallel-stages, max-concurrency
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Accuracy Tests ⏎  ⏎ `sgl-eval run gsm8k`, 1319 examples, 8x GB300, TP8, NVFP4 ⏎  ⏎ | model | cutedsl | mnnvl | no fusion | ⏎ | --- | --- | --- | --- | ⏎ | `RadixArk/GLM-5.3-NVFP4` | 97.04% | 96.82% | — | ⏎ | `RadixArk/Qwen3.8-2.4T-A95B-NVFP4` | 97.35% | — | 97.50% | ⏎  ⏎ Both fused patterns also match a torch reference to 0.031 (bf16 rounding) across all four M routes at hidden 7168 and 6144, including with HT forced off. Technically, TP for BS = 512 i …[truncated]

### L1-5c44214d6a  (L1, 2026-09-24, sha 5c44214d6ac3, PR #37213)
TITLE: [XPU] Qwen3.8-flash-next enablement (#37213)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py (+5/-5); python/sglang/srt/hardware_backend/xpu/kernels/fla/fused_sigmoid_gating_recurrent.py (+5/-0); python/sglang/srt/layers/attention/qsa/qsa_indexer.py (+26/-1); python/sglang/srt/layers/hyperconnection.py (+3/-3); test/registered/kernels/ops/qsa/test_qsa.py (+6/-1); test/registered/xpu/test_qsa_indexer_xpu.py (+238/-0)
LABELS: intel, xpu, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ Qwen/Qwen3.8-Flash-Next (generic model support merged into `main` via #37500) ⏎ introduces several architectural features new to SGLang — a sparse-attention ⏎ indexer (QSA), HyperConnection (gated multi-stream residual mixing), a ⏎ PLE/N-gram auxiliary embedding, and MTP/NEXTN speculative decoding on top of ⏎ a hybrid linear+full attention backbone. This work enables the model ⏎ end-to-end on the XPU backend, since the current impleme …[truncated]

### L1-62ae032c46  (L1, 2026-09-24, sha 62ae032c4601, PR #41097)
TITLE: [Refactor] Share the MoE output all-reduce between models (#41097)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/__init__.py (+2/-0); python/sglang/srt/layers/moe/utils.py (+39/-19); python/sglang/srt/models/bailing_moe.py (+2/-8); python/sglang/srt/models/bailing_moe_linear.py (+2/-8); python/sglang/srt/models/exaone_moe.py (+2/-8); python/sglang/srt/models/glm4_moe.py (+3/-10); python/sglang/srt/models/glm4_moe_lite.py (+3/-12); python/sglang/srt/models/gpt_oss.py (+2/-8); python/sglang/srt/models/interns2_mobius.py (+2/-12); python/sglang/srt/models/laguna.py (+2/-8); (+11 more)
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41084 ⏎ * #41083 ⏎ * #41082 ⏎ * __->__ #41097 ⏎ * #41081 ⏎ * #41080 ⏎ * #41079 ⏎  ⏎ ## Motivation ⏎  ⏎ Twenty-one MoE blocks repeat the same check before all-reducing their output over TP, and `should_skip_post_experts_all_reduce()` mixes two different questions in that check: whether the output still owes a sum at all, and whether a later step will run it. This moves the check into one helper and separates the two  …[truncated]

### L1-0b53305f48  (L1, 2026-09-24, sha 0b53305f480d, PR #36574)
TITLE: MiniMax-M3: MXFP8 dense-only block convert + aiter MXFP8 MoE on gfx950 (#36574)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/overrides.py (+3/-0); python/sglang/srt/environ.py (+3/-0); python/sglang/srt/layers/quantization/fp8.py (+79/-2); python/sglang/srt/layers/quantization/fp8_utils.py (+19/-0); test/registered/unit/test_model_overrides.py (+18/-0)
LABELS: amd, run-ci, jit-kernel, bypass-fail-fast
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: # MiniMax-M3: MXFP8 dense-only block convert + aiter MXFP8 MoE on gfx950 ⏎  ⏎ Branch: `m3/mxfp8-dense-block-convert` · base: `main` · **depends on #36559** (land it first) · `m3/fused-norm-fp8-quant` sits on top of this ⏎  ⏎ > **Merge order:** #36559 must land before this PR. This PR's aiter MXFP8 MoE path passes `activation` through `fused_moe_kwargs`. On `main`, the aiter runner also passes `activation=` explicitly, so `fused_moe` raises `TypeError: go …[truncated]

### L1-36f59982fa  (L1, 2026-09-24, sha 36f59982faaa, PR #36559)
TITLE: MoE: small-batch sorting path with fused mxfp8 quantisation (#36559)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.runner.triton, L1.runner.aiter, L1.upstream.aiter_moe
FILES: python/sglang/kernels/ops/moe/moe_sorting_small.py (+557/-0); python/sglang/kernels/ops/moe/mxfp8_moe_amd_gfx95.py (+7/-1); python/sglang/srt/layers/moe/moe_runner/aiter.py (+46/-10); python/sglang/srt/layers/moe/moe_runner/triton.py (+7/-0); python/sglang/srt/layers/quantization/unquant.py (+32/-8)
LABELS: quant, amd, run-ci, jit-kernel, bypass-fail-fast
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: # MoE: single-launch small-batch sorting with fused mxfp8 activation quant ⏎  ⏎ aiter's opus MoE sorting kernel is sized for thousands of tokens; at decode batch sizes it ⏎ costs roughly 7us/layer of launch and ramp overhead, and the separate ⏎ `fused_dynamic_mxfp8_quant_moe_sort` stage1 activation quant costs another ~4us on top. When ⏎ the number of routing pairs `P = M * topk` is at most 64, the whole sorting job — stable ⏎ sort-by-expert, per-expert bloc …[truncated]

### L1-e0b4d66733  (L1, 2026-09-24, sha e0b4d66733bd, PR #41061)
TITLE: [Perf] Lazy-load built-in model definitions and nixl_ep at startup (#41061)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/nixl.py (+22/-19); python/sglang/srt/model_loader/utils.py (+3/-2); python/sglang/srt/models/registry.py (+28/-3); test/registered/unit/models/test_model_registry.py (+91/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Importing `sglang.srt.models.registry` eagerly imports every module under `sglang.srt.models` (over 200 model files, on the order of 10-20 s on a typical dev box) even when `SGLANG_EXTERNAL_MODEL_PACKAGE` points at an out-of-tree package that provides the model actually being served. `sglang.srt.layers.moe.token_dispatcher.nixl` likewise probes `nixl_ep` at module import time, which every server pays for whether or not NIXL EP is u …[truncated]

### L1-ea5baf4022  (L1, 2026-09-24, sha ea5baf4022e4, PR #40922)
TITLE: [Refactor] Retire the model-specific Kimi K3 kernel namespace (#40922)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.runner.deep_gemm
FILES: python/sglang/kernels/ops/moe/__init__.py (+42/-1); python/sglang/kernels/ops/moe/_jit_situ_and_mul_quant.py (+0/-0); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+1/-1); python/sglang/kernels/README.md (+4/-0); python/sglang/kernels/ops/__init__.py (+0/-1); python/sglang/kernels/ops/activation/__init__.py (+23/-0); python/sglang/kernels/ops/activation/_jit_situ_and_mul.py (+0/-0); python/sglang/kernels/ops/attention/__init__.py (+52/-0); python/sglang/kernels/ops/attention/attn_res.py (+1/-1); python/sglang/kernels/ops/attention/attn_res_hip.py (+0/-0); (+42 more)
LABELS: documentation, quant, run-ci, jit-kernel, run-ci-extra
BODY: ## Motivation ⏎  ⏎ `sglang.kernels.ops.kimi_k3` collects unrelated activation, attention, GEMM, MoE and communication kernels under a model name. Retire this package and restore the logical operator grouping from #29630. ⏎  ⏎ ## Modifications ⏎  ⏎ - Move SiTU into `activation`, masked SiTU/quantization into `moe`, and the shape-tuned tiny GEMM adapter into `gemm`. ⏎ - Move attention-residual, MLA output gating, KDA MTP and FlyDSL KDA implementations into `atte …[truncated]

### L1-4142235c2b  (L1, 2026-09-24, sha 4142235c2bfa, PR #40524)
TITLE: [NPU] Update CANN version to 9.1.0 (#40524)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/npu.Dockerfile (+88/-30); .github/workflows/_npu-pr-test-stage.yml (+1/-1); .github/workflows/_npu-single-node-test-stage.yml (+2/-2); .github/workflows/coverage-collection-npu.yml (+1/-1); .github/workflows/diffusion-ci-gt-gen-npu.yml (+2/-2); .github/workflows/full-test-npu.yml (+2/-29); .github/workflows/nightly-test-npu.yml (+5/-150); .github/workflows/pr-test-npu.yml (+64/-22); .github/workflows/release-docker-npu-nightly.yml (+60/-27); .github/workflows/release-docker-npu.yml (+2/-2); (+5 more)
LABELS: dependencies, npu, run-ci
BODY: Motivation ⏎  ⏎  ⏎  ⏎ Move the SGLang NPU stack from CANN 9.0.0 / Python 3.11 to CANN 9.1.0 / Python 3.12, and align the ⏎ device matrix with what 9.1.0 actually ships. ⏎  ⏎ * CANN 9.1.0 is the current Ascend toolkit release, and the prebuilt base images ⏎   (`quay.io/ascend/cann:9.1.0-<device_type>-ubuntu22.04-py3.12`) are published for both `a3` and `950`. ⏎ * The `torch_npu` and `sgl-kernel-npu` builds we consume now target CANN 9.1.0 on Python 3.12: ⏎  …[truncated]

### L1-9c8340a2e5  (L1, 2026-09-24, sha 9c8340a2e57e, PR #41201)
TITLE: [DeepEP v2] Let a model package supply its per-rank prefill dispatch bound (#41201)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/moe_hook.py (+16/-4); python/sglang/srt/configs/moe_model_registry.py (+55/-23); test/registered/unit/layers/moe/test_hpc_ops_runner_guard.py (+14/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ `validate_deepep_v2_dispatch_token_budget` compares the whole prefill-buffer ceiling (`--chunked-prefill-size` / `--max-prefill-tokens`) against `SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK`. A model package that scatters each prefill chunk over its attention-TP group before the MoE dispatch (sequence-parallel prefill under DP attention) sends only a fraction of that ceiling from any one rank, so the unsharded check rejects c …[truncated]

### L1-515f5be77e  (L1, 2026-09-25, sha 515f5be77e74, PR #35619)
TITLE: [AMD] Integrate Aiter MegaMoEv2 for DeepSeek-V4 (#35619)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py, L1.runner.deepgemm_megamoe
FILES: python/sglang/srt/arg_groups/mega_moe_hook.py (+8/-2); python/sglang/srt/layers/moe/mega_moe.py (+32/-0); python/sglang/srt/layers/moe/mega_moe_flydsl.py (+316/-0); python/sglang/srt/layers/moe/topk.py (+32/-5); python/sglang/srt/environ.py (+6/-0); python/sglang/srt/eplb/eplb_map_record_fused.py (+114/-0); python/sglang/srt/eplb/expert_distribution.py (+75/-3); python/sglang/srt/layers/quantization/fp8.py (+184/-164); python/sglang/srt/managers/schedule_batch.py (+33/-10); python/sglang/srt/managers/scheduler_components/dp_attn.py (+0/-2); (+5 more)
LABELS: deepseek, run-ci, run-ci-extra
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: co-author: @hubertlu-tw  ⏎  ⏎ ## Summary ⏎ - add the packaged Aiter/FlyDSL MegaMoEv2 backend for DeepSeek-V4 with DP rank synchronization and graph-safe idle handling ⏎ - add prefill-only fused EPLB remap/recording to remove per-layer recorder overhead ⏎ - preserve stable SWA eviction cadence for the MegaMoE path under high DP concurrency and cover rank-sync, recorder, and mixed-slot behavior with regression tests ⏎  ⏎ ## Performance ⏎ Validated on 8×MI3 …[truncated]

### L1-402df23188  (L1, 2026-09-25, sha 402df23188db, PR #41195)
TITLE: [Fix] Stop counting a deferred FFN sum more than once: replicated TP1 shared expert, dense reduce_scatterv (#41195)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+14/-0); python/sglang/srt/layers/communicator.py (+10/-1); python/sglang/srt/layers/communicator_dsa_cp.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+7/-2); python/sglang/srt/models/laguna.py (+2/-1); python/sglang/srt/models/longcat_flash_nextn.py (+1/-0); python/sglang/srt/models/nemotron_h_utils.py (+1/-0); test/registered/unit/layers/moe/test_moe_output_reduction.py (+88/-2); test/registered/unit/layers/test_postprocess_reduce_scatterv.py (+194/-0)
LABELS: deepseek
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41200 ⏎ * #41199 ⏎ * #41191 ⏎ * #41198 ⏎ * #41197 ⏎ * #41196 ⏎ * __->__ #41195 ⏎ * #41194 ⏎ * #41193 ⏎  ⏎ ## Motivation ⏎  ⏎ This PR fixes two cases where a layer's output was summed more than it should be when its FFN reduction runs later than the FFN. ⏎  ⏎ 1. **Replicated TP1 shared expert.** With `SGLANG_SHARED_EXPERT_TP1=1`, Laguna and `DeepseekV2MoE` (DeepSeek-V2/V3, GLM5-Next without MHC) replicate the shared expert o …[truncated]

### L1-9d7f44bdbb  (L1, 2026-09-25, sha 9d7f44bdbb1e, PR #41196)
TITLE: [Refactor] Carry a deferred FFN all-reduce as UnreducedOutput and complete it in the next layer without the fused kernel (#41196)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/__init__.py (+2/-0); python/sglang/srt/layers/moe/cutedsl_ar_fusion.py (+4/-4); python/sglang/srt/batch_overlap/two_batch_overlap.py (+2/-0); python/sglang/srt/layers/communicator.py (+115/-44); python/sglang/srt/layers/communicator_mhc.py (+3/-0); python/sglang/srt/models/bailing_moe.py (+2/-2); python/sglang/srt/models/bailing_moe_v3.py (+4/-3); python/sglang/srt/models/deepseek_v2.py (+2/-1); python/sglang/srt/models/dots3_common/modeling.py (+2/-1); python/sglang/srt/models/glm4_moe.py (+2/-2); (+17 more)
LABELS: deepseek, npu
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41200 ⏎ * #41199 ⏎ * #41191 ⏎ * #41198 ⏎ * #41197 ⏎ * __->__ #41196 ⏎ * #41195 ⏎ * #41194 ⏎ * #41193 ⏎  ⏎ ## Motivation ⏎  ⏎ A decoder layer can leave its FFN all-reduce to the next layer's input norm. Until now it did so only when the fused all-reduce + residual + norm kernel took the batch, and it marked the output tensor with a `_sglang_needs_allreduce_fusion` attribute. ⏎  ⏎ - **The marker was easy to lose.** Slicing, v …[truncated]

### L1-3450d68d4e  (L1, 2026-09-25, sha 3450d68d4ede, PR #41200)
TITLE: [Refactor] Decide an FFN exit's completion once and declare the group it owes (#41200)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/__init__.py (+2/-0); python/sglang/srt/layers/moe/utils.py (+14/-13); python/sglang/srt/layers/communicator.py (+136/-107); python/sglang/srt/layers/flashinfer_comm_fusion.py (+3/-6); python/sglang/srt/models/nemotron_h.py (+6/-2); test/registered/unit/layers/test_communicator_ffn_exit.py (+81/-67); test/registered/unit/layers/test_layer_communicator_fusion_gate.py (+35/-17); test/registered/unit/layers/test_layernorm_allreduce_fusion.py (+17/-0); test/registered/unit/layers/test_postprocess_reduce_scatterv.py (+5/-4); test/registered/unit/layers/test_prepare_attn_steps.py (+30/-12); (+2 more)
LABELS: run-ci, run-ci-extra, bypass-fail-fast, parallel-stages
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * __->__ #41200 ⏎ * #41199 ⏎ * #41191 ⏎ * #41198 ⏎ * #41197 ⏎ * #41196 ⏎ * #41195 ⏎ * #41194 ⏎ * #41193 ⏎  ⏎ ## Motivation ⏎  ⏎ - **Decided twice.** `FfnExit` asked three decisions before the FFN ran. Then `finish()` chose again how the next layer completes the output (`_scatter_for_next_layer`, `_reduce_scatter_for_next_layer`), reading the postprocess function's identity and comparing step objects. ⏎ - **Group guessed by t …[truncated]

### L1-0fb699ac03  (L1, 2026-09-25, sha 0fb699ac03e2, PR #41067)
TITLE: [diffusion] model: support Ming-Image Design and Design-Layer (#41067)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels
FILES: python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py (+3/-1); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+9/-0); docs/cookbook/diffusion/inclusionAI/Ming-Image.mdx (+115/-0); docs/cookbook/diffusion/intro.mdx (+6/-0); docs/docs.json (+6/-0); docs/docs/sglang-diffusion/compatibility_matrix.mdx (+31/-0); docs/src/snippets/configs/inclusionAI/ming-image.jsx (+234/-0); docs/src/snippets/diffusion/comfyui-support.jsx (+36/-37); docs/src/snippets/diffusion/model-catalog.jsx (+5/-0); python/sglang/multimodal_gen/configs/models/dits/ming_image.py (+36/-0); (+23 more)
LABELS: documentation, npu, run-ci, diffusion, jit-kernel
BODY: ## Motivation ⏎  ⏎ Add native support for the public Ming-Image family: ⏎ - `inclusionAI/Ming-Image-0.1-Design`: text-to-image and single-image editing, with RGBA output. ⏎ - `inclusionAI/Ming-Image-0.1-Design-Layer`: decomposition into ordered transparent layers. ⏎  ⏎ ## Modifications ⏎  ⏎ - Implement the Bailing multimodal MoE encoder and query connector with native attention, TP linears, Qwen vision, and fused experts. Reuse Z-Image DiT blocks, the RGBA Qwen  …[truncated]

### L1-0967a013c2  (L1, 2026-09-25, sha 0967a013c2ab, PR #41291)
TITLE: [DSv4.1] Move the ratio-1/2 index top-k ops into kernels/ops/attention/dsv4 (#41291)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dsv4/topk.py (+40/-0); python/sglang/kernels/ops/attention/dsv4/candidate_blocks.py (+162/-2); python/sglang/kernels/ops/attention/dsv4/candidate_table.py (+29/-39); python/sglang/kernels/ops/attention/dsv4/index_logits.py (+126/-0); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+15/-14); python/sglang/srt/layers/attention/dsv4/candidate_indexer.py (+1/-84); python/sglang/srt/layers/attention/dsv4/candidate_indexer_deep_gemm.py (+29/-101); python/sglang/srt/layers/attention/dsv4/dense_prefill_indexer.py (+13/-56); python/sglang/srt/layers/attention/dsv4/indexer.py (+0/-27); test/registered/kernels/ops/attention/test_dsv41_prefill_sparse_indexer.py (+2/-4); (+1 more)
LABELS: deepseek, run-ci, jit-kernel
BODY: Move the plain-tensor index ops (torch block ids / masks, the JIT block top-k, the DeepGEMM sparse-table schedule and logits, the dense prefill score tiles, and a torch paged top-k) out of the attention layer modules into kernels/ops/attention/dsv4; callers re-import, behavior unchanged. ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #36205949148](https://github.com/sgl-project/sglang/actions/runs/36205949148) ⏎ Latest PR Test (Ext …[truncated]

### L1-8772916e06  (L1, 2026-09-25, sha 8772916e06e6, PR #41125)
TITLE: [DSv4.1] Move the low-ratio index top-k into dsv4/low_ratio_indexer (#41125)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dsv4/topk.py (+7/-7); python/sglang/kernels/ops/attention/dsv4/candidate_blocks.py (+46/-64); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+144/-560); python/sglang/srt/layers/attention/dsv4/candidate_indexer.py (+0/-194); python/sglang/srt/layers/attention/dsv4/candidate_indexer_deep_gemm.py (+0/-439); python/sglang/srt/layers/attention/dsv4/dense_prefill_indexer.py (+0/-234); python/sglang/srt/layers/attention/dsv4/metadata.py (+27/-0); python/sglang/srt/layers/attention/dsv4/v41_indexer/__init__.py (+134/-0); python/sglang/srt/layers/attention/dsv4/v41_indexer/dense_blocks.py (+345/-0); python/sglang/srt/layers/attention/dsv4/v41_indexer/full_topk.py (+184/-0); (+8 more)
LABELS: deepseek, npu, run-ci, jit-kernel, run-ci-extra, highest-priority
BODY: (auto generated by Claude) ⏎  ⏎ ## What ⏎  ⏎ Move the DSv4.1 ratio-1/2 (low-ratio) index top-k out of `deepseek_v4_backend.py` into `layers/attention/dsv4/low_ratio_indexer/`. The attention backend only dispatches dense / candidate source / candidate consumer layers and keeps an opaque `CandidateMetadata` on the forward metadata. ⏎  ⏎ - `DenseIndexer`: the plain top-k (DeepGEMM on SM100, the portable path with the fp4 decode logits kernel elsewhere) and the  …[truncated]

### L1-8fc3ce48da  (L1, 2026-09-26, sha 8fc3ce48da1c, PR #41252)
TITLE: [Refactor] Choose prepare_attn / prepare_mlp steps and fused kernels at construction (#41252)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/cutedsl_ar_fusion.py (+68/-95); python/sglang/srt/layers/communicator.py (+404/-337); python/sglang/srt/layers/communicator_dsa_cp.py (+13/-12); python/sglang/srt/layers/communicator_mhc.py (+18/-60); python/sglang/srt/models/deepseek_v2.py (+21/-6); python/sglang/srt/models/gigachat35.py (+6/-2); test/registered/unit/layers/moe/test_cutedsl_ar_fusion.py (+30/-4); test/registered/unit/layers/test_communicator_layout.py (+178/-41); test/registered/unit/layers/test_prepare_attn_steps.py (+2/-0); test/registered/unit/lora/test_triton_dp_attention_unit.py (+4/-3)
LABELS: deepseek
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41257 ⏎ * #41256 ⏎ * #41255 ⏎ * #41254 ⏎ * #41253 ⏎ * __->__ #41252 ⏎  ⏎ ## Motivation ⏎  ⏎ `prepare_mlp` went through `CommunicateWithAllReduceAndLayerNormFn.get_fn`, whose gather re-derived on every call facts fixed when the communicator is built: whether the residual must be gathered, whether attention DP is on, and which of the two DP orders applies. Whether a fused kernel may run was checked per call too. ⏎  ⏎ Ot …[truncated]

### L1-9cd6616c24  (L1, 2026-09-26, sha 9cd6616c24c4, PR #41256)
TITLE: [Refactor] Pick the two-batch-overlap split's layout moves once and remove execute (#41256)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/batch_overlap/two_batch_overlap.py (+14/-6); python/sglang/srt/layers/communicator.py (+0/-16)
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41257 ⏎ * __->__ #41256 ⏎ * #41255 ⏎ * #41254 ⏎ * #41253 ⏎ * #41252 ⏎  ⏎ ## Motivation ⏎  ⏎ The two-batch-overlap split moved its input to the splitter's attention-TP-full layout, then moved each microbatch on to the first layer's input layout. Both moves went through `CommunicateSummableTensorPairFn.execute`, which looks the move up in the table on every call. The split was `execute`'s only caller. ⏎  ⏎ ## Modificati …[truncated]

### L1-bd37e7f513  (L1, 2026-09-26, sha bd37e7f513b6, PR #41257)
TITLE: [Refactor] Choose a dense layer's boundaries under attention DP from both sides' declarations (#41257)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/cutedsl_ar_fusion.py (+3/-0); python/sglang/srt/layers/boundary_layout.py (+90/-1); python/sglang/srt/layers/communicator.py (+200/-45); python/sglang/srt/layers/communicator_dsa_cp.py (+3/-0); python/sglang/srt/layers/communicator_mhc.py (+3/-0); test/registered/unit/layers/test_communicator_ffn_exit.py (+11/-4); test/registered/unit/layers/test_communicator_layout.py (+2/-0); test/registered/unit/layers/test_declared_dense_boundary.py (+749/-0); test/registered/unit/layers/test_layer_communicator_fusion_gate.py (+12/-2); test/registered/unit/layers/test_postprocess_reduce_scatterv.py (+25/-1); (+1 more)
LABELS: run-ci, run-ci-extra, bypass-fail-fast, parallel-stages
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * __->__ #41257 ⏎ * #41256 ⏎ * #41255 ⏎ * #41254 ⏎ * #41253 ⏎ * #41252 ⏎  ⏎ ## Motivation ⏎  ⏎ The communicator chose every boundary's steps from scatter modes: ⏎ - `LayerScatterModes` enumerated each layer's modes; ⏎ - the token layouts only checked whether two modes are equivalent; ⏎ - on every batch the FFN exit re-read the producer's constructor flags, the layer's sparsity and its MLP mode to work out whether the FFN l …[truncated]

### L1-38ec649048  (L1, 2026-09-26, sha 38ec649048df, PR #41243)
TITLE: [Refactor] Restore logical kernel groups and test organization (#41243)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.routing.hash_topk, L1.runner.deep_gemm, L1.runner.deepgemm_megamoe, L1.cutlass.adapters
FILES: .claude/skills/add-jit-kernel/SKILL.md (+32/-2); .claude/skills/add-sgl-kernel/SKILL.md (+19/-0); .claude/skills/kernel-organization/SKILL.md (+82/-0); .claude/skills/write-sglang-test/SKILL.md (+4/-0); benchmark/kernels/decoding_attention_triton/triton_flashinfer_cudnn.py (+1/-1); benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py (+1/-1); benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm_blackwell.py (+2/-4); benchmark/kernels/quantization/README.md (+1/-1); benchmark/kernels/quantization/tuning_block_wise_kernel.py (+3/-3); python/sglang/kernels/README.md (+26/-3); (+347 more)
LABELS: documentation, quant, Multi-modal, deepseek, blackwell, run-ci, jit-kernel, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Recent additions have drifted from the logical operator groups in the [kernels RFC](https://github.com/sgl-project/sglang/issues/29630): MiniCPM is a top-level group, Qwen PLE combines unrelated operators, attention modules own standalone GEMM/MoE operations, and quantization modules own matmul implementations. The corresponding tests also mix numerical kernels, runtime integration, and server evaluation. ⏎  ⏎ This restores those bound …[truncated]

### L1-425a1f8f24  (L1, 2026-09-26, sha 425a1f8f247d, PR #41377)
TITLE: [AMD] Honor an explicit triton moe_runner_backend for mxfp8 on ROCm (#41377)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/overrides.py (+5/-1); test/registered/unit/test_mxfp8_moe_runner_backend.py (+92/-0)
BODY: ## Motivation ⏎  ⏎ On ROCm, an explicit `--moe-runner-backend triton` with mxfp8 quantization is ⏎ silently discarded and the server fails to start. ⏎  ⏎ `_moe_runner_backend_quant_constraints` adds `"triton"` to `allowed` only when ⏎ `is_gfx95_mxfp8`. On any other ROCm part the request falls through to the ⏎ `not in allowed` branch: ⏎  ⏎ ``` ⏎ mxfp8 quantization supports only cutlass, deep_gemm, flashinfer_megamoe, ⏎ flashinfer_trtllm, flashinfer_trtllm_ro …[truncated]

### L1-effb752188  (L1, 2026-09-26, sha effb75218808, PR #41019)
TITLE: dsv4.1-amd: KV cache layouts, FP4 indexer, compressor and router kernels (#41019)
SOURCES: path_core, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/moe/rocm_router_gate.py (+307/-0); python/sglang/kernels/jit/csrc/deepseek_v4/c1.cuh (+1/-1); python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope.cuh (+13/-2); python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope_hip.cuh (+156/-0); python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh (+79/-6); python/sglang/kernels/jit/csrc/deepseek_v4/store.cuh (+2/-0); python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp4_utils.cuh (+52/-0); python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kv_layout.cuh (+18/-21); python/sglang/kernels/ops/attention/dsv4/attn.py (+6/-4); python/sglang/kernels/ops/attention/dsv4/c2_decode_pool.py (+3/-3); (+16 more)
LABELS: documentation, quant, amd, deepseek, sgl-kernel, run-ci, jit-kernel, run-ci-extra, memory-pool
BODY: > This PR was "stack 2/4" of the DeepSeek-V4.1 AMD series (#41018 to #41021). It now follows the CUDA dsv4.1 layout (#39646, #39652, #39653, #39656, #39664, then #38798): kernel PRs by domain, each on `main` with no callers, then one integration PR. This is the second kernel PR; it does not depend on #41018. ⏎  ⏎ ## Summary ⏎ - Build the V4.1 KV layouts on ROCm (`include/sgl_kernel/deepseek_v4/kv_layout.cuh`, `fp4_utils.cuh`): `v41::store_row` quantize …[truncated]

### L1-b252aceffe  (L1, 2026-09-27, sha b252aceffecd, PR #41458)
TITLE: [AMD] Update v4 cookbook for megamoe, fp8 kv attn, BCG (#41458)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+74/-6); docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx (+3/-3)
LABELS: documentation, deepseek
BODY: ## Motivation & Modifications ⏎  ⏎  ⏎  ⏎  ⏎ Sync with https://github.com/SemiAnalysisAI/InferenceX/pull/3430. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ None. ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ None. ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://g …[truncated]

### L1-446d181e23  (L1, 2026-09-27, sha 446d181e23d0, PR #41417)
TITLE: [Refactor] Choose the boundaries of plain-TP dense layers and MoE layers from declarations (#41417)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/cutedsl_ar_fusion.py (+3/-3); python/sglang/srt/layers/boundary_layout.py (+50/-15); python/sglang/srt/layers/communicator.py (+114/-45); test/registered/unit/layers/moe/test_cutedsl_ar_fusion.py (+7/-4); test/registered/unit/layers/test_declared_decoder_boundary.py (+316/-90)
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * #41442 ⏎ * #41441 ⏎ * #41440 ⏎ * #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * #41431 ⏎ * #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * #41426 ⏎ * #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * __->__ #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ #41257 chose the boundaries of an attention followed by a dense MLP under attention DP from declarations. Every other layer still took i …[truncated]

### L1-846181f1c4  (L1, 2026-09-27, sha 846181f1c408, PR #41418)
TITLE: [Refactor] Give the fused prepare_mlp kernels an explicit contract (#41418)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/cutedsl_ar_fusion.py (+7/-1); python/sglang/srt/layers/communicator.py (+69/-37); test/registered/unit/layers/moe/test_cutedsl_ar_fusion.py (+8/-2); test/registered/unit/layers/test_declared_decoder_boundary.py (+67/-2)
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * #41442 ⏎ * #41441 ⏎ * #41440 ⏎ * #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * #41431 ⏎ * #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * #41426 ⏎ * #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * __->__ #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ `prepare_mlp` tries fused kernels that complete the attention output's all-reduce together with the residual add and the post-attention  …[truncated]

### L1-380a4d0332  (L1, 2026-09-27, sha 380a4d03329f, PR #41420)
TITLE: [Refactor] Run every batch of a layer from one BoundarySteps (#41420)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/communicator.py (+116/-141); python/sglang/srt/layers/communicator_dsa_cp.py (+13/-11); python/sglang/srt/layers/communicator_mhc.py (+4/-2); test/registered/unit/layers/test_communicator_ffn_exit.py (+30/-10); test/registered/unit/layers/test_communicator_layout.py (+8/-8); test/registered/unit/layers/test_declared_decoder_boundary.py (+160/-32); test/registered/unit/layers/test_layer_communicator_fusion_gate.py (+28/-9); test/registered/unit/layers/test_layernorm_sp.py (+23/-11); test/registered/unit/layers/test_postprocess_reduce_scatterv.py (+15/-4); test/registered/unit/layers/test_prepare_attn_steps.py (+8/-1); (+2 more)
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * #41442 ⏎ * #41441 ⏎ * #41440 ⏎ * #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * #41431 ⏎ * #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * #41426 ⏎ * #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * __->__ #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ A layer's boundary steps were held in two forms. ⏎ - **Ordinary batches** read seven separate attributes, which both the declared path and …[truncated]

### L1-1c919e401d  (L1, 2026-09-27, sha 1c919e401de4, PR #41422)
TITLE: [Fix] Keep one copy of CP-replicated rows in the DP gather (#41422)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/dp_attention.py (+18/-1); test/registered/unit/layers/test_dp_attention_cp_replicas.py (+153/-0)
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * #41442 ⏎ * #41441 ⏎ * #41440 ⏎ * #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * #41431 ⏎ * #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * #41426 ⏎ * #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * __->__ #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ Under attention DP with attention CP, the CP ranks of a DP group hold the same rows wherever a DP gather runs on them: decode, short pre …[truncated]

### L1-f5ae989a0e  (L1, 2026-09-27, sha f5ae989a0e94, PR #41424)
TITLE: [Refactor] Choose fully-DP dense and DSA / MLA prefill CP layers' steps from declarations (#41424)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py (+4/-4); python/sglang/srt/layers/boundary_layout.py (+1/-1); python/sglang/srt/layers/communicator.py (+201/-40); python/sglang/srt/layers/communicator_dsa_cp.py (+0/-153); python/sglang/srt/model_executor/model_runner.py (+1/-1); python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py (+5/-6); python/sglang/srt/models/deepseek_v2.py (+5/-4); test/registered/unit/layers/test_declared_attention_cp.py (+277/-0); test/registered/unit/layers/test_declared_decoder_boundary.py (+233/-29)
LABELS: deepseek
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * #41442 ⏎ * #41441 ⏎ * #41440 ⏎ * #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * #41431 ⏎ * #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * #41426 ⏎ * #41425 ⏎ * __->__ #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ - **A dense MLP on every rank (`moe_dense_tp_size=1`)** runs on each rank's own slice of the tokens and owes no sum, like a MoE dispatch …[truncated]

### L1-6f2d2437b6  (L1, 2026-09-27, sha 6f2d2437b63f, PR #41425)
TITLE: [Refactor] Run MHC layers on the shared boundary steps with MHC's residual operations (#41425)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/boundary_layout.py (+46/-0); python/sglang/srt/layers/communicator.py (+415/-105); python/sglang/srt/layers/communicator_mhc.py (+65/-413); python/sglang/srt/layers/dp_attention.py (+11/-22); python/sglang/srt/models/glm5_next.py (+0/-1); test/registered/unit/layers/test_communicator_ffn_exit.py (+10/-13); test/registered/unit/layers/test_communicator_layout.py (+39/-15); test/registered/unit/layers/test_declared_decoder_boundary.py (+340/-13); test/registered/unit/layers/test_dp_attention_cp_gather.py (+3/-1); test/registered/unit/layers/test_prepare_attn_steps.py (+2/-0); (+3 more)
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * #41442 ⏎ * #41441 ⏎ * #41440 ⏎ * #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * #41431 ⏎ * #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * #41426 ⏎ * __->__ #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ An MHC layer (GLM5-Next) still chose and ran its boundary steps on its own. ⏎ - **How it chose its steps:** ⏎   - its FFN input by the scatt …[truncated]

### L1-deaa7f1463  (L1, 2026-09-27, sha deaa7f14634e, PR #41426)
TITLE: [Refactor] Run two-batch-overlap layers on the declared boundaries (#41426)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/batch_overlap/two_batch_overlap.py (+7/-28); python/sglang/srt/layers/boundary_layout.py (+9/-2); python/sglang/srt/layers/communicator.py (+51/-8); python/sglang/srt/models/deepseek_v2.py (+0/-3); python/sglang/srt/models/dots3_common/modeling.py (+0/-3); python/sglang/srt/models/glm4_moe.py (+0/-3); python/sglang/srt/models/glm4_moe_lite.py (+0/-3); python/sglang/srt/models/glm5_next.py (+0/-3); python/sglang/srt/models/mimo_v2.py (+0/-8); python/sglang/srt/models/minimax_m2.py (+0/-2); (+6 more)
LABELS: deepseek, npu
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * #41442 ⏎ * #41441 ⏎ * #41440 ⏎ * #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * #41431 ⏎ * #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * __->__ #41426 ⏎ * #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ Under two-batch overlap, a dense MLP on every rank (`moe_dense_tp_size=1`) gathers its output over attention TP before a sparse layer, s …[truncated]

### L1-c289656434  (L1, 2026-09-27, sha c28965643489, PR #41427)
TITLE: [Refactor] Let the MoE declare whether its skipped reduction is one TP all-reduce (#41427)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/__init__.py (+2/-0); python/sglang/srt/layers/moe/utils.py (+22/-0); python/sglang/srt/layers/communicator.py (+4/-24); test/registered/unit/layers/test_declared_decoder_boundary.py (+16/-1); test/registered/unit/layers/test_layer_communicator_fusion_gate.py (+49/-11)
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * #41442 ⏎ * #41441 ⏎ * #41440 ⏎ * #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * #41431 ⏎ * #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * __->__ #41427 ⏎ * #41426 ⏎ * #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ An FFN may leave its all-reduce to the next layer's input only if the skipped reduction is exactly what the next layer would run: one fu …[truncated]

### L1-6f370c3bcd  (L1, 2026-09-27, sha 6f370c3bcd6a, PR #41429)
TITLE: [Refactor] Build each decoder boundary from the declarations of its two sides (#41429)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/boundary_layout.py (+65/-2); python/sglang/srt/layers/communicator.py (+268/-98); test/registered/unit/layers/moe/test_cutedsl_ar_fusion.py (+15/-0); test/registered/unit/layers/test_boundary_edges.py (+232/-0); test/registered/unit/layers/test_communicator_ffn_exit.py (+8/-0); test/registered/unit/layers/test_declared_decoder_boundary.py (+75/-7); test/registered/unit/layers/test_layer_communicator_fusion_gate.py (+8/-0); test/registered/unit/layers/test_layernorm_sp.py (+8/-0); test/registered/unit/layers/test_postprocess_reduce_scatterv.py (+8/-0); test/registered/unit/layers/test_prepare_attn_steps.py (+36/-2); (+2 more)
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * #41442 ⏎ * #41441 ⏎ * #41440 ⏎ * #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * #41431 ⏎ * #41430 ⏎ * __->__ #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * #41426 ⏎ * #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ - **One structure per layer.** A decoder layer's steps were chosen from one structure describing the whole layer (`DecoderLayerSides`).  …[truncated]

### L1-8c43c667cb  (L1, 2026-09-27, sha 8c43c667cbb4, PR #41430)
TITLE: [Refactor] Nemotron-H: build each layer's boundaries from its stage and the previous one (#41430)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/boundary_layout.py (+43/-1); python/sglang/srt/layers/communicator.py (+143/-16); python/sglang/srt/models/nemotron_h.py (+34/-86); python/sglang/srt/models/nemotron_h_mtp.py (+0/-1); python/sglang/srt/models/nemotron_h_utils.py (+95/-15); test/registered/unit/layers/test_communicator_ffn_exit.py (+31/-7); test/registered/unit/layers/test_declared_decoder_boundary.py (+26/-0); test/registered/unit/models/test_nemotron_h_aux_capture.py (+16/-7); test/registered/unit/models/test_nemotron_h_mtp_reduction.py (+1/-3); test/registered/unit/models/test_nemotron_h_stages.py (+171/-0)
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * #41442 ⏎ * #41441 ⏎ * #41440 ⏎ * #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * #41431 ⏎ * __->__ #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * #41426 ⏎ * #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ A Nemotron-H layer is a single stage: a Mamba / attention mixer, or an MLP / MoE FFN. Which stage completed a reduction was decided in t …[truncated]

### L1-f0ecf15c84  (L1, 2026-09-27, sha f0ecf15c84d2, PR #41431)
TITLE: [Refactor] Move CuTe DSL-fused layers onto the declared boundaries and FFN-exit kernel entries (#41431)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/cutedsl_ar_fusion.py (+94/-66); python/sglang/srt/layers/communicator.py (+71/-24); python/sglang/srt/models/deepseek_v2.py (+1/-0); python/sglang/srt/models/qwen2_moe.py (+1/-0); python/sglang/srt/models/qwen3_5.py (+6/-1); test/registered/unit/layers/moe/test_cutedsl_ar_fusion.py (+173/-21); test/registered/unit/layers/test_communicator_ffn_exit.py (+22/-3); test/registered/unit/layers/test_declared_decoder_boundary.py (+7/-10)
LABELS: deepseek
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * #41442 ⏎ * #41441 ⏎ * #41440 ⏎ * #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * __->__ #41431 ⏎ * #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * #41426 ⏎ * #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ The CuTe DSL all-reduce fusion (`CuteDSLFusionLayerCommunicator`) was the last communicator that still worked from scatter modes, and it …[truncated]

### L1-31329c222c  (L1, 2026-09-27, sha 31329c222cb0, PR #41432)
TITLE: [Fix] Run MoE layers under attention DP and GQA prefill CP on the declared DP × CP gather (#41432)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/communicator.py (+4/-2); test/registered/unit/layers/test_declared_decoder_boundary.py (+9/-1); test/registered/unit/layers/test_dp_attention_cp_gather.py (+11/-4)
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * #41442 ⏎ * #41441 ⏎ * #41440 ⏎ * #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * __->__ #41432 ⏎ * #41431 ⏎ * #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * #41426 ⏎ * #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ Under attention DP with GQA prefill CP, `_declared_sides` still sent MoE layers to the scatter-mode steps. On a CP extend, those steps g …[truncated]

### L1-0971450c16  (L1, 2026-09-27, sha 0971450c168f, PR #41436)
TITLE: [Fix] LongCat-Flash under attention DP: branch and merge the dense FFNs through the communicators (#41436)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/communicator.py (+92/-0); python/sglang/srt/models/longcat_flash.py (+21/-56); test/registered/unit/layers/test_communicator_ffn_exit.py (+29/-6); test/registered/unit/layers/test_declared_decoder_boundary.py (+140/-0); test/registered/unit/models/test_longcat_flash_shortcut.py (+16/-21)
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * #41442 ⏎ * #41441 ⏎ * #41440 ⏎ * #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * __->__ #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * #41431 ⏎ * #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * #41426 ⏎ * #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ Each LongCat-Flash layer runs a MoE shortcut beside a dense branch (mlp0 → attention 1 → mlp1). Both branches start from the same FFN in …[truncated]

### L1-427c9e05c1  (L1, 2026-09-27, sha 427c9e05c165, PR #41438)
TITLE: [Refactor] Choose every layer's boundaries from declarations and remove the scatter-mode selection (#41438)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py (+8/-20); python/sglang/srt/layers/attention/dsa/dsa_npu_indexer.py (+3/-12); python/sglang/srt/layers/communicator.py (+91/-425); python/sglang/srt/layers/communicator_mhc.py (+12/-10); python/sglang/srt/model_executor/forward_batch_info.py (+6/-10); python/sglang/srt/models/deepseek_v2.py (+9/-7); python/sglang/srt/models/gigachat35.py (+3/-1); python/sglang/srt/models/glm4_moe_lite.py (+3/-1); python/sglang/srt/models/glm5_next.py (+3/-1); test/registered/unit/layers/test_communicator_layout.py (+0/-332); (+2 more)
LABELS: deepseek, npu
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * #41442 ⏎ * #41441 ⏎ * #41440 ⏎ * #41439 ⏎ * __->__ #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * #41431 ⏎ * #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * #41426 ⏎ * #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ Two kinds of code still depended on scatter modes: ⏎ - **Readers outside the communicator.** ⏎   - The NPU MLA / DSA attention and the DSA N …[truncated]

### L1-0e907b755d  (L1, 2026-09-27, sha 0e907b755d73, PR #41439)
TITLE: [Refactor] Split the layer communicator into a package (move only) (#41439)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/cutedsl_ar_fusion.py (+2/-1); python/sglang/srt/batch_overlap/two_batch_overlap.py (+1/-1); python/sglang/srt/layers/boundary_layout.py (+0/-403); python/sglang/srt/layers/communicator.py (+0/-3244); python/sglang/srt/layers/communicator/__init__.py (+124/-0); python/sglang/srt/layers/communicator/adapters/__init__.py (+14/-0); python/sglang/srt/layers/communicator/adapters/attention.py (+188/-0); python/sglang/srt/layers/communicator/boundary.py (+969/-0); python/sglang/srt/layers/communicator/layer.py (+1348/-0); python/sglang/srt/layers/communicator/layout.py (+251/-0); (+30 more)
LABELS: amd, npu
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * #41442 ⏎ * #41441 ⏎ * #41440 ⏎ * __->__ #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * #41431 ⏎ * #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * #41426 ⏎ * #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ `layers/communicator.py` had grown to about 3,200 lines. It held the token layouts, the owed-output types, the attention-TP helpers, the …[truncated]

### L1-644014bc98  (L1, 2026-09-27, sha 644014bc981b, PR #41440)
TITLE: [Refactor] Declare each stage's residual read and update, and give each stage its own entry (#41440)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/cutedsl_ar_fusion.py (+1/-1); python/sglang/srt/layers/communicator/__init__.py (+12/-2); python/sglang/srt/layers/communicator/boundary.py (+138/-69); python/sglang/srt/layers/communicator/layer.py (+113/-69); python/sglang/srt/layers/communicator/ops.py (+42/-34); python/sglang/srt/layers/communicator/residual/__init__.py (+59/-32); python/sglang/srt/layers/communicator/residual/add_norm.py (+81/-23); python/sglang/srt/layers/communicator/residual/mhc.py (+100/-35); python/sglang/srt/models/nemotron_h_utils.py (+8/-6); python/sglang/test/communicator_patch.py (+2/-2); (+12 more)
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * #41442 ⏎ * #41441 ⏎ * __->__ #41440 ⏎ * #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * #41431 ⏎ * #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * #41426 ⏎ * #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ The boundary steps are chosen from declarations of each stage's rows and sums. How a stage writes its output into the residual, and how  …[truncated]

### L1-341d5273bd  (L1, 2026-09-27, sha 341d5273bd3f, PR #41441)
TITLE: [Refactor] Build the boundary into any stage with one construction (#41441)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/communicator/__init__.py (+4/-2); python/sglang/srt/layers/communicator/boundary.py (+180/-219); python/sglang/srt/layers/communicator/layer.py (+25/-27); python/sglang/srt/layers/communicator/ops.py (+87/-20); python/sglang/srt/models/nemotron_h_utils.py (+11/-10); test/registered/unit/layers/moe/test_cutedsl_ar_fusion.py (+10/-7); test/registered/unit/layers/test_boundary_edges.py (+178/-35); test/registered/unit/layers/test_communicator_ffn_exit.py (+17/-7); test/registered/unit/layers/test_declared_decoder_boundary.py (+133/-73); test/registered/unit/layers/test_dp_attention_cp_gather.py (+1/-1); (+6 more)
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * #41442 ⏎ * __->__ #41441 ⏎ * #41440 ⏎ * #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * #41431 ⏎ * #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * #41426 ⏎ * #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ `make_boundary` still routed on what its consumer was, via `reads=InputRead.ATTENTION`, `InputRead.FFN` or `None`, into two families of  …[truncated]

### L1-06d012eaca  (L1, 2026-09-27, sha 06d012eacad7, PR #41442)
TITLE: [Refactor] Give a layer its CuTe DSL kernels at construction instead of a subclass (#41442)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/cutedsl_ar_fusion.py (+109/-104); python/sglang/srt/layers/communicator/layer.py (+34/-8); python/sglang/srt/models/deepseek_v2.py (+5/-7); python/sglang/srt/models/qwen3_5.py (+9/-8); test/registered/unit/layers/moe/test_cutedsl_ar_fusion.py (+50/-33)
LABELS: deepseek
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * #41443 ⏎ * __->__ #41442 ⏎ * #41441 ⏎ * #41440 ⏎ * #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * #41431 ⏎ * #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * #41426 ⏎ * #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ The CuTe DSL all-reduce fusion still needed its own `LayerCommunicator` subclass, `CuteDSLFusionLayerCommunicator`: ⏎ - it overrode the th …[truncated]

### L1-81f27fb3a7  (L1, 2026-09-27, sha 81f27fb3a71d, PR #41443)
TITLE: [Refactor] Replace LayerScatterModes with LayerFacts and remove ScatterMode (#41443)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/communicator/__init__.py (+2/-4); python/sglang/srt/layers/communicator/layer.py (+32/-136); python/sglang/srt/layers/communicator/layout.py (+0/-40); python/sglang/srt/models/bailing_moe.py (+3/-3); python/sglang/srt/models/bailing_moe_linear.py (+3/-3); python/sglang/srt/models/bailing_moe_v3.py (+3/-3); python/sglang/srt/models/deepseek_v2.py (+3/-3); python/sglang/srt/models/dots3_common/modeling.py (+3/-3); python/sglang/srt/models/falcon_h1.py (+3/-3); python/sglang/srt/models/glm4_moe.py (+3/-3); (+39 more)
LABELS: amd, deepseek, run-ci, run-ci-extra, bypass-fail-fast, parallel-stages, max-concurrency
BODY: This PR is part of a stack (oldest at bottom): ⏎  ⏎ * __->__ #41443 ⏎ * #41442 ⏎ * #41441 ⏎ * #41440 ⏎ * #41439 ⏎ * #41438 ⏎ * #41437 ⏎ * #41436 ⏎ * #41435 ⏎ * #41434 ⏎ * #41433 ⏎ * #41432 ⏎ * #41431 ⏎ * #41430 ⏎ * #41429 ⏎ * #41428 ⏎ * #41427 ⏎ * #41426 ⏎ * #41425 ⏎ * #41424 ⏎ * #41423 ⏎ * #41422 ⏎ * #41421 ⏎ * #41420 ⏎ * #41419 ⏎ * #41418 ⏎ * #41417 ⏎  ⏎ ## Motivation ⏎  ⏎ Every layer's boundaries are now chosen from declarations that read five layer facts: whether the layer and each of its neighbours is sp …[truncated]

### L1-85be04978d  (L1, 2026-09-28, sha 85be04978d76, PR #40943)
TITLE: [AMD][DSV4] moe: enable shared-expert fusion on the grouped-topk path (megamoe) (#40943)
SOURCES: path_core, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-2)
LABELS: amd, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: > Stacked on top of the Aiter MegaMoEv2 integration, **sgl-project/sglang#35619** — that PR adds the FlyDSL MegaMoEv2 backend but requires `--disable-shared-experts-fusion` on ROCm because of the bug below. This change removes that restriction. Base is the `20260922` image commit; #35619 is bind-mounted on top for the megamoe path. ⏎  ⏎ ## Motivation ⏎  ⏎ `select_experts` already computes `num_fused_shared_experts_for_gate` — `0` on the per-rank fuse …[truncated]

### L1-fa443412c5  (L1, 2026-09-28, sha fa443412c558, PR #41161)
TITLE: [AMD] [GLM5] Fuse shared expert into AITER MoE on gfx950 (#41161)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/glm5_next.py (+70/-9); test/registered/kernels/ops/moe/test_glm53_shared_expert_fusion.py (+265/-0); test/registered/unit/layers/moe/test_fused_shared_expert_scaling.py (+32/-1); test/registered/unit/models/test_shared_experts_fusion_gates.py (+189/-4)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ GLM-5.3-Flash has 288 routed experts and one always-on shared expert. The current AITER path executes routed `E=288, topk=8` MoE, then launches the shared gate/up projection, clamped SiLU and quantization, shared down projection, and a BF16 routed-plus-shared add separately. ⏎  ⏎ This change appends the shared expert as expert ID 288 and dispatches one `E=289, topk=9` AITER MoE operation. The appended expert always has weight 1.0 becau …[truncated]

### L1-6be9c78cf4  (L1, 2026-09-28, sha 6be9c78cf442, PR #40664)
TITLE: [Intel GPU] Xpu/weekly simple model enablement 2026 09 21 (#40664)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+5/-2); .pre-commit-config.yaml (+1/-1); python/sglang/check_env.py (+57/-12); python/sglang/multimodal_gen/test/server/consistency_thresholds/intel_xpu_b60.json (+17/-0); python/sglang/multimodal_gen/test/server/perf_baselines/xpu_b60.json (+26/-1); python/sglang/multimodal_gen/test/test_utils.py (+4/-0); python/sglang/srt/arg_groups/choices.py (+1/-1); python/sglang/srt/arg_groups/platform_hook.py (+6/-0); python/sglang/srt/arg_groups/validation_hook.py (+14/-0); python/sglang/srt/layers/attention/minimax_sparse_backend.py (+0/-2); (+2 more)
LABELS: intel, xpu, run-ci, diffusion
BODY: ## Motivation ⏎  ⏎ Weekly consolidation of the followings: ⏎  ⏎ - #35267 — @Amrutha-M05: small XPU diffusion test baselines; mechanically applied, no unresolved review. Kernel-independent. ⏎ - #37037 — @CharlesXu-HQ: focused CPU/XPU `check_env` parity and tests; mechanically applied, no unresolved review. Kernel-independent. ⏎ - #38510 — @KMS07: focused fused-sampling XPU enablement; approved and XPU stage-A passed. Kernel PRs #182, #282, #288, and #36 …[truncated]

### L1-b07cb9e9e7  (L1, 2026-09-28, sha b07cb9e9e73a, PR #40987)
TITLE: [NVIDIA] Update deepgemm, deep-ep, sgl-kernel in CUDA 13.4 image, use cuda base image (#40987)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.cu134 (+27/-100)
BODY: ## Motivation ⏎  ⏎ Simplify CUDA 13.4 image and bring the wheels up to date with the main container. ⏎  ⏎ ## Modifications ⏎  ⏎ * Remove CUDA base build and use `nvidia/cuda:13.4.1-cudnn-devel-ubuntu24.04` instead. ⏎ * Update `sgl-kernel` from `0.4.6.post1` to `0.4.7` ⏎ * Update `sgl-deep-gemm` from `0.1.5.post2` to `0.2.0` ⏎ * Update `sgl-deep-ep` from `0.1.0` to `0.1.2` ⏎ * Remove C++20 patches as they are no longer needed. ⏎ * Rename `NCCL_VERSION`->`SGL …[truncated]

### L1-096b066fb4  (L1, 2026-09-28, sha 096b066fb4f6, PR #41020)
TITLE: dsv4.1-amd: gfx950 sparse decode attention and sorted top-k (#41020)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/aot/python/sgl_kernel/top_k.py (+7/-2); python/sglang/kernels/aot/benchmark/bench_dsv4_topk_transform.py (+76/-0); python/sglang/kernels/aot/csrc/common_extension_rocm.cc (+1/-1); python/sglang/kernels/aot/csrc/elementwise/deepseek_v4_topk.cu (+172/-6); python/sglang/kernels/aot/include/sgl_kernel_ops.h (+2/-1); python/sglang/kernels/aot/tests/test_topk.py (+45/-0); python/sglang/kernels/ops/attention/dsv4/attn_glue_hip.py (+426/-0); python/sglang/kernels/ops/attention/dsv4/candidate_blocks_hip.py (+408/-0); python/sglang/kernels/ops/attention/dsv4/compact_attention_hip.py (+403/-0); python/sglang/kernels/ops/attention/dsv4/decode_attention_sm100.py (+45/-3); (+7 more)
LABELS: documentation, quant, amd, deepseek, sgl-kernel, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, memory-pool
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: > This PR was "stack 3/4" of the DeepSeek-V4.1 AMD series (#41018 to #41021). It now follows the CUDA dsv4.1 layout (#39646, #39652, #39653, #39656, #39664, then #38798): kernel PRs by domain, each on `main` with no callers, then one integration PR. This is the third kernel PR; it does not depend on #41018 or #41019. ⏎  ⏎ ## Summary ⏎ - `ops/attention/sparse_decode_reduce_hip.py`: `gfx_sparse_split_reduce` combines the split-KV partials of aiter's gfx9 …[truncated]
