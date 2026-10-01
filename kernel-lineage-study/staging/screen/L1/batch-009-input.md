### L1-4a279d9c36  (L1, 2026-05-06, sha 4a279d9c3625, PR #24550)
TITLE: [R3] Avoid implicit CUDA sync in routed experts DP slicing (#24550)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/dp_attention.py (+17/-0); python/sglang/srt/state_capturer/routed_experts.py (+5/-5)
LABELS: run-ci
ISSUES: #24514 [R3] Non-DeepEP DP attention codepath causes implicit CUDA sync in routed experts overlap
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Fixes #24514  ⏎  ⏎ ## Modifications ⏎  ⏎ Compute the routed-experts DP local slice indices from `forward_batch.global_num_tokens_cpu` instead of `forward_batch.global_num_tokens_gpu`. ⏎  ⏎ This keeps the slice bounds as Python integers while preserving the existing indexing semantics: ⏎  ⏎ - non-CUDA-graph path: use the prefix sum of per-DP token counts ⏎ - CUDA graph path: use `dp_rank * cuda_graph_batch` ⏎ - DeepEP path remains unchanged …[truncated]

### L1-eaf074d50e  (L1, 2026-05-06, sha eaf074d50eef, PR #24487)
TITLE: propagate pytest exit code from test __main__ entries (#24487)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/tests/diffusion/test_norm_tanh_mul_add_norm_scale.py (+0/-88); python/sglang/jit_kernel/tests/test_cast.py (+0/-314); python/sglang/jit_kernel/tests/test_flash_attention_3.py (+0/-1359); python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py (+0/-448); python/sglang/multimodal_gen/test/server/test_tracing.py (+0/-155); python/sglang/multimodal_gen/test/unit/manual/test_patch_embed.py (+0/-292); test/registered/lora/test_lora_moe_runner.py (+0/-788); test/registered/lora/test_marlin_lora_correctness.py (+0/-289); test/registered/lora/test_sgemm_sorted_by_adapter.py (+0/-236); test/registered/unit/test_no_bare_pytest_main.py (+90/-0)
LABELS: lora, run-ci, diffusion, jit-kernel
BODY: ## Problem ⏎  ⏎ `run_files` in `python/sglang/test/ci/ci_utils.py` decides whether a ⏎ test file passed by reading the wrapping python script's ⏎ `process.returncode`. Files using ⏎  ⏎     if __name__ == "__main__": ⏎         pytest.main([__file__]) ⏎  ⏎ discard pytest's exit code, so the script returns 0 even when assertions ⏎ fail. The CI runner then logs the file as PASSED while pytest's stdout ⏎ reports the failure — silently masking real regressions. ⏎  ⏎ (Discovered …[truncated]

### L1-ecb786c8d7  (L1, 2026-05-06, sha ecb786c8d719, PR #24268)
TITLE: [Kernel] Deprecate DeepGemm in sgl kernel and apply custom wheel sgl-deep-gemm (#24268)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+10/-2); python/sglang/srt/layers/attention/nsa_backend.py (+19/-3); python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py (+10/-4); python/sglang/srt/layers/deep_gemm_wrapper/configurer.py (+2/-0); python/sglang/srt/layers/quantization/fp8_utils.py (+13/-0); scripts/ci/cuda/ci_install_dependency.sh (+6/-0); scripts/ci/cuda/warmup_deep_gemm.py (+8/-4); (+4 more)
LABELS: high priority, dependencies, deepseek, sgl-kernel, run-ci
BODY: ## Motivation ⏎ Ref:  ⏎ https://github.com/sgl-project/DeepGEMM/pull/26 ⏎ https://pypi.org/project/sgl-deep-gemm/ ⏎ #20745  ⏎  ⏎ Do the following one by one: ⏎  ⏎  ⏎ We will build a single wheel for deepgemm in sglang, rather than compiling it with sglang-kernel ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Mer …[truncated]

### L1-3b2c730320  (L1, 2026-05-07, sha 3b2c73032049, PR #24005)
TITLE: [AMD] Enable dual-stream MoE on ROCm (#24005)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+12/-0); docs/references/environment_variables.md (+1/-0); python/sglang/srt/environ.py (+3/-0); python/sglang/srt/models/deepseek_v2.py (+6/-1)
LABELS: documentation, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Add an opt-in env var that lets Deepseek Model allocate its alt CUDA/HIP stream on the ROCmpath. This is the same alt_stream consumed by DeepseekV2MoE.{forward_normal_dual_stream, forward_deepep, _fuse_shared_experts_inside_sbo} to overlap shared experts vs routed experts. Without this gate the AMD path runs single-stream regardless of the existing dual-stream code. ⏎ Recommend pairing the alt-stream overlap with Mori's CU-free AsyncLL ep kernel ( …[truncated]

### L1-35870d55ac  (L1, 2026-05-07, sha 35870d55aca7, PR #23882)
TITLE: Deepseek V4 (#23882)
SOURCES: path_core, symbol_pickaxe, dependency_pin, corpus:production-kernel-provenance, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.routing.topk_py, L1.routing.fused_gate, L1.routing.hash_topk, L1.runner.framework, L1.runner.triton, L1.runner.deep_gemm, L1.runner.marlin, L1.runner.deepgemm_megamoe, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+10/-0); python/pyproject.toml (+2/-1); python/sglang/jit_kernel/csrc/deepseek_v4/hash_topk.cuh (+214/-0); python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh (+219/-0); python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h (+105/-107); python/sglang/jit_kernel/csrc/moe/moe_fused_gate.cuh (+363/-0); python/sglang/jit_kernel/moe_fused_gate.py (+82/-0); .github/workflows/pr-test.yml (+106/-0); docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+0/-4); python/sglang/jit_kernel/csrc/deepseek_v4/c128.cuh (+522/-0); (+144 more)
LABELS: documentation, high priority, quant, dependencies, deepseek, speculative-decoding, sgl-kernel, blackwell, npu, benchmark
BODY: ## Rebase progress ⏎  ⏎ merge-base 0519b09 → target main ea794de (latest), 2880 commits across **29 batches** (~100/batch). **Done: 29 / 29. ✅** ⏎  ⏎ [details omitted] ⏎ [details omitted] ⏎ [details omitted] ⏎ [details omitted] ⏎ [details omitted] ⏎ [details omitted] ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ [details om …[truncated]

### L1-15e6572f21  (L1, 2026-05-07, sha 15e6572f2198, PR #23255)
TITLE: [MUSA][18/N] Add MUSA-optimized kernel implementations for hot ops (#23255)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/musa/moe_gemv_swiglu.mu (+846/-0); sgl-kernel/csrc/common_extension_musa.cc (+42/-0); sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.mu (+17/-8); sgl-kernel/csrc/musa/common.muh (+42/-0); sgl-kernel/csrc/musa/dtype.muh (+446/-0); sgl-kernel/csrc/musa/pos_encoding_contiguous.mu (+264/-0); sgl-kernel/csrc/musa/ternary.mu (+123/-0); sgl-kernel/csrc/musa/top_k_top_p_sampling.mu (+382/-0); sgl-kernel/include/musa/dispatch_utils.h (+34/-0); sgl-kernel/include/musa/integer_subbyte.h (+49/-0); (+5 more)
LABELS: quant, sgl-kernel, run-ci, mthreads
BODY: ## Motivation ⏎  ⏎ This PR continues the ongoing effort (tracked in https://github.com/sgl-project/sglang/issues/16565) to add full support for Moore Threads GPUs in SGLang by leveraging MUSA (Meta-computing Unified System Architecture) for LLM inference. ⏎  ⏎ The primary goal of this pr is to add MUSA-specific kernel support to sgl-kernel. ⏎  ⏎ It focuses only on the MUSA kernel path so that sgl-kernel can build successfully with setup_musa.py and be  …[truncated]

### L1-7d397ad23d  (L1, 2026-05-07, sha 7d397ad23ded, PR #18172)
TITLE: [NPU]Support model Trinity-mini for Npu, accuracy 90% (#18172)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+15/-24); python/sglang/srt/models/afmoe.py (+17/-4); python/sglang/test/ascend/test_ascend_utils.py (+1/-0); test/registered/ascend/llm_models/test_ascend_trinity_mini.py (+47/-0)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ Previously, model trinity mini is not supported for npu. ⏎  ⏎ ## Modifications ⏎  ⏎ As follows ⏎ custom_routing_function was used because the trinity mini model (afmoe.py) has its unique routing function. ⏎ ## Accuracy Tests ⏎ 90%, surpass GPU. ⏎ <img width="1300" height="162" alt="image" src="https://github.com/user-attachments/assets/06ecdd67-5dd9-453e-a1d4-72b4e9ee6a5c" /> ⏎ <img width="1517" height="160" alt="image" src="https://githu …[truncated]

### L1-55224fff08  (L1, 2026-05-08, sha 55224fff0851, PR #22123)
TITLE: Add Arm64 CPU Phase 1A CI bootstrap (#22123)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/arm64.Dockerfile (+52/-0); .github/workflows/pr-test-arm64.yml (+118/-0); python/sglang/srt/server_args.py (+4/-1); sgl-kernel/csrc/cpu/CMakeLists.txt (+17/-0); sgl-kernel/csrc/cpu/torch_extension_cpu.cpp (+29/-21); test/srt/cpu/test_server_args_backend.py (+35/-0); test/srt/run_suite.py (+17/-0)
LABELS: sgl-kernel, run-ci
BODY: ### Summary ⏎  ⏎ This PR adds **Arm64 CPU Phase 1A CI bootstrap** support. ⏎  ⏎ The goal is to give Arm64 a real PR-time build and functional test lane without disturbing the existing Xeon/x86 flow. This is an additive bootstrap step, not a full parity claim. ⏎  ⏎ ### What changed ⏎  ⏎ 1. Added a dedicated Arm64 PR workflow in `.github/workflows/pr-test-arm64.yml` ⏎ 2. Added a native Arm64 build image in `docker/arm64.Dockerfile` ⏎ 3. Defaulted Arm64 CPU b …[truncated]

### L1-461bc8af49  (L1, 2026-05-08, sha 461bc8af494c, PR #23708)
TITLE: [NPU][Doc] Update GLM-5 docs, enabling deepep by default (#23708)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: docs/platforms/ascend/ascend_npu_glm5_examples.md (+3/-3)
LABELS: documentation, npu, run-ci
BODY: ## Motivation ⏎  ⏎ When DeepEP is not enabled, there can be accuracy issues, so DeepEP is enabled by default. ⏎  ⏎ ## Modifications ⏎  ⏎ docs/platforms/ascend/ascend_npu_glm5_examples.md ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-mer …[truncated]

### L1-55d8223c2b  (L1, 2026-05-08, sha 55d8223c2b59, PR #16045)
TITLE: [sgl-kernel/cpu] support w8a8 int8 model for arm cpu (#16045)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/cpu/aarch64/moe.cpp (+323/-0); python/sglang/srt/layers/quantization/w8a8_int8.py (+12/-10); sgl-kernel/csrc/cpu/CMakeLists.txt (+18/-4); sgl-kernel/csrc/cpu/aarch64/gemm_int8.cpp (+132/-0); sgl-kernel/csrc/cpu/aarch64/op.h (+343/-0); sgl-kernel/csrc/cpu/gemm_int8.cpp (+2/-0); sgl-kernel/csrc/cpu/torch_extension_cpu.cpp (+2/-2)
LABELS: sgl-kernel, run-ci
BODY: ## Motivation ⏎  ⏎ Support W8A8 Int8 Dense and MoE model for Arm CPU. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ Tested accuracy with few_shot_gsm8k dataset, 200 questions. ⏎  ⏎ ``` ⏎ Dense model ⏎ ----------- ⏎ BF16: meta-llama/Llama-3.2-3B-Instruct ⏎ Accuray: 0.680 ⏎  ⏎ W8A8: RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8 ⏎ Accuray: 0.670 ⏎  ⏎ MoE model ⏎ --------- ⏎ BF16: Qwen/Qwen3-30B-A3B-Instruct-2507 ⏎ Accuray: 0.945 ⏎  ⏎ W8A8: ramblingpolymath/Qwen …[truncated]

### L1-f4b7e73699  (L1, 2026-05-09, sha f4b7e7369978, PR #24260)
TITLE: Enable trtllm-gen BF16 MoE for MTP (#24260)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+8/-6); python/sglang/srt/server_args.py (+2/-15)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ After https://github.com/flashinfer-ai/flashinfer/pull/2803, BF16 supports DSV3 routing, for unquantized MoE layer we can use it in MTP layer ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ Remove the guard. ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ GPQA on DSV3 NVFP4: ⏎  ⏎ ``` ⏎ python3 -m sglang.test.run_eval --port 30020 --eval-name gpqa --num-examples 198 --max-tokens 128000 --repeat 8 --top-p 0.95 --temperature 1.0 --thinking-mode deepseek-v3 ⏎ ``` ⏎  ⏎ 0.834 -> 0. …[truncated]

### L1-4b23f6bdc5  (L1, 2026-05-09, sha 4b23f6bdc50e, PR #24562)
TITLE: Fix performance regression on Deepseek V3 on `moe-runner-backend=triton` on SM90 (#24562)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py (+14/-6)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: When recent PyTorch upgrade to 2.11, we will use Triton 3.6.0, which does not have a tuned config, it will fall back to 3.3.1, which is not correct, we should prefer the more recent 3.5.1.

### L1-ef5e9f8aba  (L1, 2026-05-09, sha ef5e9f8abab1, PR #24793)
TITLE: [DSV4] Cherry pick missing commits from deepseek_v4 branch and enhance tests (#24793)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+2/-2); docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx (+0/-3); python/sglang/srt/entrypoints/openai/protocol.py (+5/-2); python/sglang/srt/function_call/deepseekv32_detector.py (+26/-10); python/sglang/srt/model_loader/loader.py (+5/-0); python/sglang/srt/model_loader/weight_utils.py (+33/-3); python/sglang/srt/server_args.py (+6/-0); scripts/ci/cuda/ci_install_dsv4_dep.sh (+161/-0); scripts/ci/cuda/ci_install_flash_mla.sh (+0/-35); scripts/ci/utils/slash_command_handler.py (+6/-0); (+5 more)
LABELS: documentation, deepseek
BODY: ## Motivation ⏎ - Cherry-pick https://github.com/sgl-project/sglang/commit/dbdd494429678cdfa9d792d2c69b8a5f9f7961d7 ⏎ - Cherry-pick https://github.com/sgl-project/sglang/commit/c13abdc9fc7518e2155feaa45b2c64bc4eadbf33 ⏎ - Enhance per-commit v4 test with: b200 fp4 depep test, b200 fp4 megamoe test, h200 fp8 deepep test ⏎ - Add some tools for v4 tests, such as slash commands ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profi …[truncated]

### L1-096ad02b06  (L1, 2026-05-09, sha 096ad02b0661, PR #24204)
TITLE: [Model] Laguna-XS.2 Model Support (#24204)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/docs/supported-models/generative_models.mdx (+5/-0); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/laguna.py (+209/-0); python/sglang/srt/configs/model_config.py (+9/-0); python/sglang/srt/function_call/function_call_parser.py (+2/-0); python/sglang/srt/function_call/poolside_v1_detector.py (+452/-0); python/sglang/srt/models/laguna.py (+787/-0); python/sglang/srt/parser/reasoning_parser.py (+10/-0); python/sglang/srt/utils/hf_transformers/common.py (+2/-0); test/registered/unit/entrypoints/openai/test_serving_chat.py (+46/-0); (+2 more)
LABELS: documentation, run-ci
BODY: ## Purpose ⏎  ⏎ Add native SGLang support for [poolside/Laguna-XS.2](https://huggingface.co/poolside/Laguna-XS.2), a hybrid sliding-window-attention MoE model. Without this, the checkpoint only loads via `--trust-remote-code` and falls back to the HF reference module, bypassing `RadixAttention`, `FusedMoE`, and SGLang's hybrid-SWA KV-cache pool. ⏎  ⏎ Special thanks to @kpham-sgl for taking the time to review. ⏎  ⏎ **Supported model:** ⏎  ⏎ | Model | Arch …[truncated]

### L1-d49fc092cb  (L1, 2026-05-09, sha d49fc092cb42, PR #23550)
TITLE: [Bug Fix] GLM-5.1: drop constexpr on page_indice_batch_offset, skip offloader post_init on draft worker, support N=32 in copy_to_gpu_no_ce (#23550)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/attention/nsa/index_buf_accessor.py (+1/-1); python/sglang/srt/model_executor/model_runner.py (+2/-1); sgl-kernel/csrc/elementwise/copy.cu (+2/-0)
LABELS: sgl-kernel, run-ci
BODY: ## Motivation ⏎  ⏎ Several issues were blocking the GLM-5.1 expert-offload path and causing excessive Triton recompilation on long-sequence / per-batch runs. This PR bundles the minimal fixes required to unblock it. ⏎  ⏎ ## Modifications ⏎  ⏎ - **NSA `get_k_and_s` kernel**: drop `tl.constexpr` from `page_indice_batch_offset` and pass it as a regular `int`. Its value is `page_indices.shape[1]`, which varies at runtime (long seqlen / per-batch), so keepi …[truncated]

### L1-cfd3fd00d0  (L1, 2026-05-09, sha cfd3fd00d048, PR #24854)
TITLE: [RL] Call torch.cuda.empty_cache() for `in-place` pause mode to avoid OOM (#24854)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/io_struct.py (+6/-1); python/sglang/srt/managers/scheduler.py (+9/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Post-weight-update processing (e.g. DeepSeek MLA w_kc/w_vc derivation, FP8 scale rebuild) creates transient CUDA allocations that fragment PyTorch's block cache. Without `empty_cache()`, reserved memory grows each iteration and eventually OOMs. ⏎  ⏎ | Pause mode | `flush_cache` called? | `empty_cache` before this PR | `empty_cache` after this PR | ⏎ |---|---|---|---| ⏎ | `abort` | Yes | Yes (via `flush_cache`) | Yes (via `flush_cache` + re …[truncated]

### L1-d3fd91ed97  (L1, 2026-05-10, sha d3fd91ed9726, PR #24696)
TITLE: [Gemma4] Optimize Gemm4 with fused Q/K/V RMSNorm + per-expert FP8 ckpt loader (#24696)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/gemma4_fused_ops.py (+115/-0); python/sglang/srt/models/gemma4_causal.py (+58/-15); python/sglang/srt/models/gemma4_mm.py (+35/-0); test/registered/models/test_gemma4_fp8_per_expert_loading.py (+109/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ The original motivation is to speedup Gemm4 model. Two unrelated issues on the Gemma4 FP8-MoE path: ⏎  ⏎ 1. **Attention RMSNorm is launch-bound.** Each Gemma4 attention layer issues three separate sgl-kernel `rmsnorm` launches (q/k/v). At small token counts the per-launch host overhead dominates — three back-to-back launches cost ~60μs regardless of M, even though the actual compute is trivial. ⏎  ⏎ 2. **compressed-tensors per-expert …[truncated]

### L1-8e2142c15a  (L1, 2026-05-10, sha 8e2142c15aac, PR #24850)
TITLE: [MoE] Fix NaN in flashinfer TRT-LLM A2A dispatch by sanitizing padding slots (#24850)
SOURCES: path_core, subject_keyword, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py (+1/-0)
LABELS: run-ci
DEEP_STUDY: deep-study correctness case sglang:8e2142c15a: class=memory_safety_oob; symptom=nan_inf; introducing=unknown
BODY: ## Motivation ⏎  ⏎ Pass `invalid_token_expert_id=-1` to `MoeAlltoAll.dispatch()` so the sanitize kernel marks padding slots (beyond recv_counters per source rank) with expert ID -1. Without this, the default (`None`) skips sanitization entirely, leaving uninitialized garbage expert IDs in the padding slots. The downstream TRTLLM MoE kernel then processes those garbage entries, reading uninitialized hidden states and producing NaN. ⏎  ⏎ ## Modificatio …[truncated]

### L1-1d80a1a9fe  (L1, 2026-05-11, sha 1d80a1a9fe6d, PR #23745)
TITLE: Use Cute-DSL NVFP4 quantization kernels (#23745)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm, L1.runner.flashinfer_cutedsl, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+1/-2); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+1/-1); python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py (+3/-1); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+4/-1); benchmark/kernels/quantization/bench_fp4_quant.py (+121/-121); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py (+3/-1); python/sglang/srt/layers/quantization/fp4_utils.py (+75/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+4/-13)
LABELS: quant, blackwell, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ <img width="1560" height="1040" alt="image" src="https://github.com/user-attachments/assets/4e5e0785-8e7c-4357-ad20-7ca70aa6d209" /> ⏎  ⏎ This can beat the original SGLang CUDA based kernel in all scenarios, after perf optimizations in https://github.com/flashinfer-ai/flashinfer/pull/2904. **Note, this is with `backend=cute-dsl`**, TRT-LLM `quantize_with_block_size` will still be much slower. We don't use this option. ⏎  ⏎  ⏎  ⏎ ## Mod …[truncated]

### L1-09a4828db9  (L1, 2026-05-11, sha 09a4828db94a, PR #23819)
TITLE: [NPU] Fix warmup error with --disable-cuda-graph and mtp (#23819)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+15/-1)
LABELS: npu, run-ci
BODY: ## Motivation ⏎ The server warmup encounters an error in forward_decode when --disable-cuda-graph is enabled. The root cause is the redundant padding tokens in the input tensor, which leads to a dimension mismatch for the NPU fused attention operator. ⏎ ``` ⏎ File "/xxx/sglang/srt/hardware_backend/npu/attention/ascend_backend.py", line 1846, in forward_decode ⏎ attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score( ⏎ ``` ⏎  ⏎ ## Modification ⏎  …[truncated]

### L1-958f35d1e0  (L1, 2026-05-11, sha 958f35d1e053, PR #24916)
TITLE: ci: run H20 stage with CUDA 13 (#24916)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+0/-1)
BODY: ## Motivation ⏎  ⏎ Start running the H20 CI stage with CUDA 13. ⏎  ⏎ The H20 runner is already using CUDA 13.0, but `stage-c-test-8-gpu-h20` currently forces CUDA 12.9 dependencies via `CU_VERSION: cu129`. This causes DeepEP extension build failure during dependency installation. ⏎  ⏎ Failure log: https://github.com/sgl-project/sglang/actions/runs/25635867709/job/75247521703#step:6:1 ⏎  ⏎ ```text ⏎ copying deep_ep/__init__.py -> build/lib.linux-x86_64-cpy …[truncated]

### L1-df441b8fea  (L1, 2026-05-11, sha df441b8feae3, PR #23827)
TITLE: [NPU] Support shared expert dual stream optimization (#23827)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/cmo.py (+29/-0); python/sglang/srt/models/qwen2_moe.py (+22/-1)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ On NPU, the shared expert and routed expert in MoE models are executed sequentially on the same stream, leaving compute resources underutilized. This PR introduces a dual stream optimization that overlaps shared expert computation with routed expert computation on independent NPU streams. ⏎  ⏎ ## Modifications ⏎  ⏎ - Added `shared_expert_on_independent_stream` in `cmo.py` to offload shared expert computation to independent NPU stream …[truncated]

### L1-e9dea79755  (L1, 2026-05-11, sha e9dea797555b, PR #24262)
TITLE: (3/n - prefill optimize)[LoRA][MoE] Optimize virtual experts: remove CPU-GPU sync & multi-block CUDA JIT histogram (#24262)
SOURCES: path_core, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.align.cuda_jit
FILES: python/sglang/jit_kernel/csrc/moe/moe_align_kernel.cu (+7/-5); python/sglang/srt/lora/triton_ops/virtual_experts.py (+100/-6); test/registered/lora/test_virtual_experts_kernels.py (+46/-25)
LABELS: high priority, lora, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ based on https://github.com/sgl-project/sglang/pull/24246 ⏎  ⏎ I add the jit but also keep the original in case our hardware can fall back to torch if fail on jit. ⏎  ⏎   ### Hidden side-benefit: ~68% drop in `AllReduce` kernel time on prefill-heavy LoRA workloads ⏎  ⏎   While cross-checking optimize-2 against pre-opt-2 traces (Qwen3-30B-A3B + LoRA, TP=4, GB300, ⏎   input=65536 / output=32, 10 prompts, `--max-concurrency 4`), I noticed a m …[truncated]

### L1-74d70af09a  (L1, 2026-05-11, sha 74d70af09a19, PR #23449)
TITLE: [Apple Silicon] Add Metal kernel support in sgl-kernel (#23449)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs_new/docs/hardware-platforms/apple_metal.mdx (+22/-2); docs_new/docs/hardware-platforms/mthreads_gpu.mdx (+1/-1); docs_new/docs/sglang-diffusion/installation.mdx (+1/-1); python/sglang/__init__.py (+3/-1); sgl-kernel/csrc/metal/README.md (+21/-0); sgl-kernel/csrc/metal/placeholder.cpp (+0/-0); sgl-kernel/csrc/metal/placeholder.metal (+0/-0); sgl-kernel/python/sgl_kernel/__init__.py (+218/-206); sgl-kernel/python/sgl_kernel/metal.py (+30/-0); sgl-kernel/setup_metal.py (+298/-0)
LABELS: documentation, dependencies, sgl-kernel, run-ci, mthreads, apple-silicon
BODY: ## Motivation ⏎  ⏎  ⏎ [#22868](https://github.com/sgl-project/sglang/pull/22868) introduces custom Metal kernels to SGLang. After some discussion, we'd like to stay aligned with SGLang's existing convention of keeping custom kernels separate from SRT wiring. This PR therefore moves the Metal kernel build into `sgl-kernel` so it can be installed independently, mirroring how the CUDA / ROCm / MUSA backends are organized. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ 1. Ad …[truncated]

### L1-ca3bc05fea  (L1, 2026-05-11, sha ca3bc05fea1a, PR #24949)
TITLE: Deepseek-v4-Pro share expert tp1 (#24949)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/environ.py (+1/-1); python/sglang/srt/model_executor/model_runner.py (+4/-2); python/sglang/srt/models/deepseek_v2.py (+26/-14)
LABELS: deepseek
BODY: ## Motivation ⏎  ⏎ Share Expert cannot be deployed using TP16, this PR implements a TP1 deployment of Share Expert. DeepSeekV4 branch PR is here: https://github.com/sgl-project/sglang/pull/23911 ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blo …[truncated]

### L1-d5f3254ed1  (L1, 2026-05-12, sha d5f3254ed1f1, PR #24452)
TITLE: [Dependency] Flashinfer 0.6.8post1 -> 0.6.11 (#24452)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+4/-4); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/flashinfer_comm_fusion.py (+11/-13); python/sglang/srt/layers/quantization/fp4_utils.py (+7/-7); python/sglang/srt/utils/common.py (+1/-1); test/registered/moe/test_cutedsl_moe.py (+9/-6)
LABELS: high priority, dependencies, run-ci, bypass-fastfail
DEEP_STUDY: deep-study: this PR was reverted by PR 25310 (explicit_rollback, reason=crash_or_hang)
BODY: Commits of interest: ⏎  ⏎ https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.10 ⏎ https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.9 ⏎  ⏎ Note: if this is not cherry-picked, maybe just use 0.6.11, because it will break main and bad performance of this features ⏎  ⏎ Confirming not falling back from TRTLLM allreduce fusio  ⏎ <img width="1452" height="738" alt="Screenshot 2026-05-06 at 3 48 23 PM" src="https://github.com/user-attachm …[truncated]

### L1-52d4c697bb  (L1, 2026-05-12, sha 52d4c697bb46, PR #25076)
TITLE: Fix fused_moe import for non-NPU devices (#25076)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/afmoe.py (+1/-3)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L1-66a9234246  (L1, 2026-05-12, sha 66a923424638, PR #24879)
TITLE: [AMD] support fp8 blockwise quantization combine for mori ep (#24879)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+77/-36); docker/rocm.Dockerfile (+1/-1)
LABELS: amd
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ This patch is to add FP8 blockwise quantization support for the combine in Mori EP dispatcher, complementing the existing FP8/FP4 dispatch quantization. Fix issue https://github.com/sgl-project/sglang/issues/24866 ⏎  ⏎ Work with https://github.com/ROCm/mori/pull/311 ⏎  ⏎ cc @Duyi-Wang @HaiShaw  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - Introduce `DispatchDtype` and `CombineDtype` enums to replace boolean flags (`fp8_dispatch`, `fp4_dispatch`), ma …[truncated]

### L1-51a9403104  (L1, 2026-05-13, sha 51a94031042a, PR #25129)
TITLE: Update flashinfer to 0.6.11.post1 (#25129)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1)
LABELS: dependencies, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 25310 (explicit_rollback, reason=crash_or_hang)
BODY: ## Motivation ⏎ https://github.com/flashinfer-ai/flashinfer/compare/v0.6.11...v0.6.11.post1 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github. …[truncated]

### L1-409d350fb6  (L1, 2026-05-13, sha 409d350fb6f6, PR #19329)
TITLE: Bugfix: fix symm not enabled due to incorrect registration of comm (#19329)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/token_dispatcher/standard.py (+4/-1); python/sglang/srt/layers/communicator.py (+10/-5); python/sglang/srt/layers/communicator_nsa_cp.py (+2/-1); python/sglang/srt/layers/dp_attention.py (+8/-8)
LABELS: run-ci
BODY: CC @ShangmingCai @nvcastet @Fridge003 @BBuf @yizhang2077  @ch-wan    PTAL, thx. ⏎  ⏎ ## Motivation ⏎  ⏎ fix symm not enabled due to incorrect registration of comm. ⏎  ⏎ 1. `get_local_dp_buffer` uses the group obtained by `get_tp_group` by default to register symm.   ⏎ 2. The `attn_cp_all_gather_into_tensor` operation uses the group obtained by `get_attention_cp_group` to perform allgather.   ⏎ 3. Steps 1 and 2 cause symm to fail when enabled because the  …[truncated]

### L1-22012ba1bc  (L1, 2026-05-13, sha 22012ba1bc21, PR #17392)
TITLE: Add BF16 support to EP-MoE for DeepGEMM (#17392)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.deep_gemm, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+141/-22); python/sglang/srt/layers/moe/ep_moe/layer.py (+5/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-1); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+159/-13); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+21/-13); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+2/-1); python/sglang/srt/layers/quantization/unquant.py (+23/-2); python/sglang/srt/server_args.py (+5/-0); python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py (+56/-0); python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py (+34/-0)
LABELS: documentation, quant, run-ci, model-gateway, mthreads
BODY: ## Motivation ⏎ We attempted to enable EP-MoE on BF16 models and found that BF16 data type was not supported in the DeepGEMM backend. This PR implements BF16 EP-MoE support using DeepGEMM's grouped BF16 GEMM kernels. ⏎  ⏎ ## Modifications ⏎ Following the same computational paradigm as FP8, we have implemented EP-MoE for BF16 using `deep_gemm.m_grouped_bf16_gemm_nt_contiguous` and `deep_gemm.m_grouped_bf16_gemm_nt_masked` from DeepGEMM. ⏎  ⏎ The changes …[truncated]

### L1-72b49bfac6  (L1, 2026-05-13, sha 72b49bfac6cb, PR #25113)
TITLE: docker, ci: swap GB DeepEP source from fzyzcjy fork to deepseek-ai/DeepEP@hybrid-ep (#25113)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+3/-3); scripts/ci/cuda/ci_install_deepep.sh (+3/-4)
BODY: ## Motivation ⏎  ⏎ Move the **Grace Blackwell** DeepEP build path off the personal `fzyzcjy/DeepEP@gb200_blog_part_2` fork and onto upstream `deepseek-ai/DeepEP@hybrid-ep`, where DeepSeek's active DeepEP work lands. This gets the GB300 staging image (`lmsysorg/sglang-staging:deepseek-v4-grace-blackwell-dev_arm64`) onto a maintained upstream source. ⏎  ⏎ ## Modifications ⏎  ⏎ Two files, GB-only paths: ⏎  ⏎ **`docker/Dockerfile`** — 3-line change: ⏎ - Default `GRACE …[truncated]

### L1-28758d37dd  (L1, 2026-05-13, sha 28758d37dd5c, PR #24816)
TITLE: Add FlashInfer SM90 cutlass MXFP4 MoE backend (W4A16) for GPT-OSS + DeepSeek-V4 (#24816)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+12/-0); python/sglang/srt/layers/quantization/fp8.py (+9/-0); python/sglang/srt/layers/quantization/mxfp4.py (+269/-1); python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py (+263/-0); python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py (+9/-1); python/sglang/test/bench_mxfp4_sm90_kernels.py (+366/-0); test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py (+70/-1); test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py (+544/-0)
LABELS: deepseek, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Summary ⏎  ⏎ Wires FlashInfer's SM90 mixed-input `cutlass_fused_moe(use_w4_group_scaling=True)` path ([FlashInfer PR #3084](https://github.com/flashinfer-ai/flashinfer/pull/3084)) into both SGLang MXFP4 entry points as an **opt-in** backend on Hopper: ⏎  ⏎ - **GPT-OSS** path: `Mxfp4MoEMethod` in `mxfp4.py`, dispatched from `Mxfp4Config`. ⏎ - **DeepSeek-V4** path: new `Mxfp4FlashinferCutlassMoEMethod`, sibling of ⏎   `Mxfp4MarlinMoEMethod` / `Mxfp4Fl …[truncated]

### L1-0a2615df24  (L1, 2026-05-13, sha 0a2615df24a2, PR #25182)
TITLE: chore: add vLLM SPDX copyright headers to ported files (#25182)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.openai_triton_kernels, L1.upstream.openai_triton_kernels
FILES: benchmark/hicache/bench_serving.py (+2/-0); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+2/-0); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py (+2/-0); python/sglang/bench_serving.py (+2/-0); python/sglang/multimodal_gen/runtime/distributed/communication_op.py (+1/-0); python/sglang/multimodal_gen/runtime/distributed/device_communicators/base_device_communicator.py (+1/-0); python/sglang/multimodal_gen/runtime/distributed/device_communicators/cpu_communicator.py (+1/-0); python/sglang/multimodal_gen/runtime/distributed/device_communicators/cuda_communicator.py (+1/-0); python/sglang/multimodal_gen/runtime/distributed/device_communicators/pynccl.py (+1/-0); python/sglang/multimodal_gen/runtime/distributed/device_communicators/pynccl_wrapper.py (+1/-0); (+126 more)
LABELS: quant, deepseek, sgl-kernel, blackwell, run-ci, piecewise-cuda-graph, diffusion
BODY: ## Summary ⏎  ⏎ Adds the canonical vLLM SPDX header ⏎  ⏎ ``` ⏎ # SPDX-License-Identifier: Apache-2.0 ⏎ # SPDX-FileCopyrightText: Copyright contributors to the vLLM project ⏎ ``` ⏎  ⏎ to every Python file in the repository that carries an `Adapted from https://github.com/vllm-project/vllm/...` attribution. ⏎  ⏎ ## Motivation ⏎  ⏎ The existing `# Adapted from ...` comments show that SGLang contributors have always intended to preserve attribution and acknowledge upstream wo …[truncated]

### L1-37f18438c5  (L1, 2026-05-13, sha 37f18438c593, PR #24986)
TITLE: [rebase]Deepseek_v4 support w4(mxfp4)a16 on hopper (#24986)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.marlin
FILES: python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh (+10/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py (+3/-7); python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py (+57/-12); python/sglang/srt/layers/quantization/marlin_utils_fp4.py (+32/-16); python/sglang/srt/layers/quantization/mxfp4.py (+40/-1); test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py (+2/-0); test/registered/dsv4/test_deepseek_v4_flash_fp8_h200.py (+2/-0)
LABELS: deepseek, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ According to @zhangxiaolei123456 https://github.com/sgl-project/sglang/pull/23686 ⏎  ⏎ ## Modifications ⏎  ⏎ Rebase the MXFP4 support from the deepseek_v4 branch onto the main branch. ⏎  ⏎ ## Accuracy Tests ⏎ ### V4 Flash ⏎ DeepSeek V4 Flash run command: ⏎ ``` ⏎ SGLANG_DSV4_FP4_EXPERTS=1 SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 GLOO_SOCKET_IFNAME=eth0 \ ⏎ sglang serve \ ⏎   --trust-remote-code \ ⏎   --model-path /0424/models/DeepSeek-V4-Flash \ ⏎   -- …[truncated]

### L1-af7511e0e8  (L1, 2026-05-13, sha af7511e0e86b, PR #24719)
TITLE: [sgl-model-gateway] Close PyO3 binding gaps and add regression tests (#24719)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-model-gateway/bindings/python/src/sglang_router/router.py (+21/-0); sgl-model-gateway/bindings/python/src/lib.rs (+43/-0); sgl-model-gateway/bindings/python/src/sglang_router/router_args.py (+74/-31); sgl-model-gateway/bindings/python/tests/test_pyo3_binding.py (+726/-0)
LABELS: run-ci, model-gateway
BODY: ## Summary ⏎  ⏎ - The Python `sglang_router` wrapper did not surface several `RouterConfig` fields the Rust binary exposes, plus a few CLI knobs were inert. This closes the actionable gaps without changing user-visible defaults. ⏎ - New tests exercise the PyO3 boundary directly (no `_Router` mocking) so future drift between `RouterArgs`, `Router.from_args`, and the Rust constructor signature surfaces immediately. ⏎  ⏎ ## What changed ⏎  ⏎ **`sgl-model-gateway/ …[truncated]

### L1-b7f856df70  (L1, 2026-05-13, sha b7f856df70c8, PR #25052)
TITLE: DeepSeek V4 w4a4 MegaMoE (#25052)
SOURCES: path_core, dependency_pin, release_notes
ARTIFACT_HINTS: L1.runner.deepgemm_megamoe, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/mega_moe.py (+52/-10); python/sglang/srt/environ.py (+11/-0); test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py (+0/-49); test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py (+148/-0)
LABELS: high priority, dependencies, deepseek, run-ci
BODY: It's #24444 rebased on main branch ⏎ Author: @pranjalssh  ⏎  ⏎ Requires a new release of sgl-deep-gemm after https://github.com/sgl-project/DeepGEMM/pull/31 merges ⏎  ⏎ Next steps: ⏎ - Update cookbook for w4a4 usage ⏎ - Add CI test for protection (Done) ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See  …[truncated]

### L1-c701a08765  (L1, 2026-05-13, sha c701a08765ba, PR #19290)
TITLE: feat: [2/2][DeepEP] Add waterfill load balancing for shared expert dispatch (#19290)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/deepep_waterfill.py (+584/-0); python/sglang/srt/layers/moe/topk.py (+48/-4); python/sglang/srt/model_executor/model_runner.py (+45/-0); python/sglang/srt/models/deepseek_v2.py (+3/-22); python/sglang/srt/server_args.py (+36/-1); docs/advanced_features/server_arguments.md (+1/-0); python/sglang/srt/environ.py (+3/-0); test/registered/unit/server_args/test_server_args.py (+41/-0)
LABELS: documentation, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ In DeepSeek V3/R1 with expert parallelism (EP), each rank processes a subset of routed experts plus the **full** shared expert. Since every token must visit the shared expert, it creates a fixed compute load on all ranks regardless of routed expert distribution. When routed expert load is imbalanced across ranks (common with skewed token distributions), the shared expert amplifies the bottleneck on already-overloaded ranks. ⏎ **Water …[truncated]

### L1-b71d74673c  (L1, 2026-05-13, sha b71d74673c46, PR #24253)
TITLE: ci: combine H200 8-GPU warmup steps and surface server log on every path (#24253)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+10/-13); scripts/ci/cuda/warmup_deep_gemm.py (+214/-14); scripts/ci/cuda/warmup_server.py (+44/-22)
BODY: ## Summary ⏎  ⏎ Two related changes to the per-commit H200 8-GPU CI warmup pipeline: ⏎  ⏎ ### 1. Remove the heavyweight `Warmup Server CUDA Graphs` step; expand the lightweight DeepGEMM step to cover all per-commit H200 models ⏎  ⏎ `warmup_server.py` was launching the full sglang server for `V3-0324:8` and `Ring-2.5-1T:8` to do a full model load + DeepGEMM JIT pre-compile + CUDA graph capture. The graph-capture portion is **wasted work** — CUDA graphs don't  …[truncated]

### L1-f7efff321d  (L1, 2026-05-13, sha f7efff321d00, PR #25215)
TITLE: [Docker] Fix several dependencies in Dockerfile (#25215)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+16/-9)
BODY: ## Motivation ⏎  ⏎ Following #25113 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and  …[truncated]

### L1-9e00b7ca95  (L1, 2026-05-13, sha 9e00b7ca95aa, PR #24575)
TITLE: [NPU] add zbal support for npu (#24575)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+5/-1); python/sglang/srt/distributed/parallel_state.py (+1/-1); python/sglang/srt/environ.py (+4/-0); python/sglang/srt/hardware_backend/npu/utils.py (+92/-0); python/sglang/srt/managers/scheduler.py (+13/-0); python/sglang/srt/model_executor/model_runner.py (+15/-1); python/sglang/srt/utils/common.py (+14/-2)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ zbal stands for Zero Buffer Acceleration Library based on memfabric, currently open-sourced as app part in https://gitcode.com/Ascend/memfabric_hybrid ⏎  ⏎  ⏎ ## How To Use ⏎ ``` ⏎ unset PYTORCH_NPU_ALLOC_CONF ⏎ export SGLANG_ZBAL_LOCAL_MEM_SIZE=61184  # zbal total allocate memory size in MB ⏎ export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 ⏎  ⏎ # if you want to use mix alloc (using vmm mode for weights & kv, using zbal gva mode for acti …[truncated]

### L1-85d9c77c57  (L1, 2026-05-13, sha 85d9c77c57b9, PR #25138)
TITLE: ci: extract cuda stage actions + runner_config mapping (#25138)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/_pr-test-stage.yml (+198/-0); .github/workflows/pr-test.yml (+227/-826); scripts/ci/runner_configs.py (+27/-0); scripts/ci/runner_configs.yml (+26/-0)
LABELS: high priority, lora, deepseek, blackwell, run-ci
BODY: Stack on top of #25197 (already merged): pulls every CUDA stage job in pr-test.yml into 3 composite actions and routes install-script selection through a single `runner_config` lookup key. ⏎  ⏎ ## Composite actions ⏎  ⏎ - `.github/actions/setup-cuda-test-stage` — checkout, stage-health / maintenance gates, sgl-kernel wheel download, install. Install script is now looked up by `runner_config` (no longer a caller input). ⏎ - `.github/actions/run-test-cuda-su …[truncated]

### L1-22dfcdaa04  (L1, 2026-05-14, sha 22dfcdaa0438, PR #25310)
TITLE: revert flashinfer 0.6.11 bumps (#25310)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+4/-4); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/flashinfer_comm_fusion.py (+13/-11); python/sglang/srt/layers/quantization/fp4_utils.py (+7/-7); python/sglang/srt/utils/common.py (+1/-1); test/registered/moe/test_cutedsl_moe.py (+6/-9)
LABELS: dependencies
DEEP_STUDY: deep-study revert record: explicit_rollback of PR(s) 24452;25129 reason=crash_or_hang
BODY: Reverts #24452 and #25129. The 0.6.11 bump causes a CUDA illegal-address crash in the triton_kernels mxfp4 MoE matmul (`_matmul_ogs_NNT_bf16xbf16xmxfp4_128x256x128x1_swiglu`) during piecewise CUDA graph capture for gpt-oss-120b on 4xH100. Same failure signature on the original PR CI and on main scheduled CI. Pin back to 0.6.8.post1 until upstream fixes the mxfp4 MoE path. ⏎  ⏎ Failure links: ⏎ - PR #24452 CI: https://github.com/sgl-project/sglang/actio …[truncated]

### L1-67096f48bf  (L1, 2026-05-14, sha 67096f48bffd, PR #25317)
TITLE: Revert "[MoE] Decouple Mega MoE from DeepEP backend" (#25317)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.deep_gemm, L1.runner.deepgemm_megamoe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-1); python/sglang/srt/layers/moe/mega_moe.py (+1/-2); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+1/-0); python/sglang/srt/layers/moe/utils.py (+0/-4); python/sglang/srt/layers/quantization/fp8.py (+1/-1); python/sglang/srt/server_args.py (+1/-28); docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx (+20/-6); test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py (+8/-2)
LABELS: documentation, deepseek
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 24884 reason=unstated
BODY: Reverts sgl-project/sglang#24884

### L1-ba214ef3d3  (L1, 2026-05-14, sha ba214ef3d363, PR #24725)
TITLE: ci: tag-gated nightly migration — foundation + 40 whole-file moves (#24725)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/_pr-test-check-changes.yml (+5/-0); .github/workflows/pr-test-extra.yml (+225/-0); .github/workflows/pr-test.yml (+0/-17); python/sglang/test/kits/streaming_session_kit.py (+441/-0); python/sglang/test/server_fixtures/hybrid_attn_backend_fixture.py (+102/-0); python/sglang/test/server_fixtures/ngram_fixture.py (+74/-0); python/sglang/test/server_fixtures/pcg_spec_fixture.py (+80/-0); python/sglang/test/server_fixtures/standalone_fixture.py (+114/-0); python/sglang/test/server_fixtures/streaming_session_fixture.py (+434/-0); scripts/ci/utils/compute_partitions.py (+2/-2); (+68 more)
LABELS: high priority, quant, lora, Multi-modal, deepseek, speculative-decoding, hicache, blackwell, run-ci, bypass-fastfail
BODY: ## Summary ⏎  ⏎ PR2 of the CI pruning effort. Introduces a **conditional-CI** middle tier driven by PR labels + per-test tags, then applies it to 59 nightly migrations and assorted cleanup. Combined with PR1 (#24721, merged), the per-commit CUDA budget drops from 1232 min → 753 min (**−305 min in this PR alone, −38.9% combined, ~8h total**). ⏎  ⏎ ## New vocabulary ⏎  ⏎ | Tier | When it runs | How it's defined | ⏎ |---|---|---| ⏎ | **baseline CI** | every PR comm …[truncated]

### L1-ad4994dc1d  (L1, 2026-05-14, sha ad4994dc1d6f, PR #25279)
TITLE: DeepseekV2MoE: defer shared experts when routed kernel is non-mutating (#25279)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+11/-3)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ When the routed-MoE kernel does **not** mutate its input (`inplace=False`, e.g. `flashinfer_trtllm_routed`), `hidden_states` is preserved across `self.experts(...)`. In that case the existing ordering — compute `shared_experts` **before** the routed call — forces the shared-experts activations and the routed-MoE workspace to coexist on-device, inflating transient peak memory at long prefill on Kimi-K2.5-NVFP4 / Blackwell. ⏎  ⏎ ## Modif …[truncated]

### L1-0c19540550  (L1, 2026-05-15, sha 0c19540550e1, PR #25335)
TITLE: [Fix] Fix gpt oss triton kernels and upgrade flashinfer back to 0.6.11.post1 (#25335)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.runner.openai_triton_kernels, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe, L1.upstream.openai_triton_kernels
FILES: docker/Dockerfile (+2/-2); python/pyproject.toml (+5/-5); python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+4/-3); python/sglang/srt/layers/moe/moe_runner/triton_kernels.py (+6/-2); python/sglang/srt/layers/moe/topk.py (+44/-1); python/sglang/srt/entrypoints/engine.py (+2/-2); python/sglang/srt/layers/flashinfer_comm_fusion.py (+11/-13); python/sglang/srt/layers/quantization/fp4_utils.py (+7/-7); python/sglang/srt/layers/quantization/mxfp4.py (+46/-3); python/sglang/srt/utils/common.py (+10/-2); (+3 more)
LABELS: high priority, dependencies, deepseek, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ co-author: @b8zhong @mmangkad  ⏎ Modified upon #25312  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sg …[truncated]

### L1-37f030a0de  (L1, 2026-05-15, sha 37f030a0de6d, PR #24884)
TITLE: [MoE] Decouple Mega MoE from DeepEP backend (#24884)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts(reverted), body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.deep_gemm, L1.runner.deepgemm_megamoe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-1); python/sglang/srt/layers/moe/mega_moe.py (+2/-1); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+0/-1); python/sglang/srt/layers/moe/utils.py (+4/-0); python/sglang/srt/layers/quantization/fp8.py (+1/-1); python/sglang/srt/server_args.py (+28/-1); docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx (+6/-20); test/registered/dsv4/test_deepseek_v4_flash_fp4_megamoe_b200.py (+2/-8)
LABELS: documentation, deepseek
DEEP_STUDY: deep-study: this PR was reverted by PR 25317 (confirmed_revert, reason=unstated)
BODY: ## Summary ⏎ - Auto-configure EP (`ep_size = tp_size`) when `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1` is set, so users no longer need `--moe-a2a-backend=deepep` or the `deep_ep` library to use Mega MoE. ⏎ - Mega MoE uses `deep_gemm.fp8_fp4_mega_moe()` for its own all-to-all communication and never touches the DeepEP dispatcher, so the dependency was unnecessary.

### L1-8d5b347edd  (L1, 2026-05-15, sha 8d5b347edd9b, PR #24906)
TITLE: Support Qwen3.5 NVFP4 MTP DeepEP (#24906)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+47/-2); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+4/-1); python/sglang/srt/layers/attention/linear/gdn_backend.py (+6/-2); python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Qwen3.5 NVFP4 disaggregated serving with DeepEP and NEXTN/MTP currently fails in the MTP draft path on GB300/SM100+. ⏎  ⏎ There are three separate issues: ⏎  ⏎ 1. The MTP draft MoE layer can have `quant_config=None` even though the target model uses NVFP4. The existing DeepEP low-latency path assumes either an FP4 quant config or the deprecated DeepGEMM path. ⏎ 2. DeepEP low-latency dispatch for Qwen3.5 needs a build with `kNumMaxTopK …[truncated]

### L1-54221dd998  (L1, 2026-05-15, sha 54221dd99814, PR #25379)
TITLE: feat(moe): reuse prev-layer output as symm_output for FP4 routed MoE (#25379)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework, L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+9/-0); python/sglang/srt/layers/moe/moe_runner/base.py (+17/-1); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+25/-7); python/sglang/srt/models/deepseek_v2.py (+31/-10); python/sglang/srt/layers/communicator.py (+3/-0)
LABELS: deepseek, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ For Kimi-K2.5-NVFP4 with \`flashinfer_trtllm_routed\` on TP=4, two sources of unnecessary tensor lifetime were identified and eliminated: ⏎  ⏎ 1. **symm_output reuse**: each of the 60 MoE decoder layers allocated a fresh \`symm_output\` buffer (60 × 235 MB = 14 GB churn per 16K-token prefill). The previous layer's MoE output (\`hidden_states_orig\`) is stale by the time the MoE kernel runs and can be reused in-place. ⏎ 2. **attn_inputs l …[truncated]

### L1-f9caf43095  (L1, 2026-05-15, sha f9caf4309531, PR #25394)
TITLE: [CI] slash handler: lookup `runs_on` from `runner_configs.yml` (#25394)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-states.yml (+1/-1); .github/workflows/rerun-test.yml (+106/-58); .github/workflows/slash-command-handler.yml (+1/-1); scripts/ci/utils/slash_command_handler.py (+142/-137); test/README.md (+3/-3); test/manual/4-gpu-models/test_qwen35_fp4_triton.py (+0/-3); test/manual/4-gpu-models/test_qwen3_next_models.py (+0/-3); test/manual/8-gpu-models/test_deepseek_v3_basic.py (+0/-3); test/manual/8-gpu-models/test_dsa_models_basic.py (+0/-3); test/manual/attention/test_fa3.py (+0/-3); (+23 more)
LABELS: documentation, quant, lora, deepseek, run-ci, run-ci-extra
BODY: ## Summary ⏎  ⏎ Make `runner_configs.yml` the single source of truth for `/rerun-test` (matches the main PR test pipeline) and refactor `rerun-test.yml` to consume it directly. Drop the hardcoded `CUDA_SUITE_TO_RUNNER` stage table and the bespoke `use_deepep` / `install_diffusion` flag soup. ⏎  ⏎ ## Changes ⏎  ⏎ **`slash_command_handler.py`** ⏎ - `detect_suite` resolves `runs_on`, `install_script`, `install_timeout`, `rdma_devices` from `runner_configs.yml` fo …[truncated]

### L1-34cb8e2842  (L1, 2026-05-15, sha 34cb8e28425d, PR #24130)
TITLE: fix(sgl-kernel): sm90 compile flashmla failed (#24130)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/cmake/flashmla.cmake (+26/-16); sgl-kernel/csrc/flashmla_extension.cc (+2/-0)
LABELS: sgl-kernel, run-ci
ISSUES: #24126 [Bug] sgl-kernel fails to compile FlashAttention on CUDA 12.8 with H20
BODY: ## Motivation ⏎  ⏎ ### close https://github.com/sgl-project/sglang/issues/24126 ⏎  ⏎ Fixes a FlashMLA build issue on Hopper GPUs such as H20 when buildingwith CUDA 12.8. ⏎  ⏎ In `sgl-kernel/cmake/flashmla.cmake`, the `sm100` Blackwell source files were always appended to `FlashMLA_SOURCES`, even when the build only targeted Hopper (`sm90a`). At the same time,`sm100` gencode was only added when `CUDA_VERSION > 12.8`. ⏎  ⏎ That mismatch meant a CUDA 12.8 H …[truncated]

### L1-b7d62bd724  (L1, 2026-05-15, sha b7d62bd72414, PR #25420)
TITLE: [CI] Rename basic CI `stage-a/b/c` -> `base-a/b/c` for symmetry with extra CI (#25420)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .claude/skills/add-jit-kernel/SKILL.md (+9/-9); .claude/skills/ci-workflow-guide/SKILL.md (+63/-64); .claude/skills/write-sglang-test/SKILL.md (+40/-41); .github/actions/check-pr-test-health/action.yml (+7/-7); .github/actions/wait-for-jobs/action.yml (+2/-2); .github/workflows/_pr-test-sgl-kernel-build.yml (+2/-2); .github/workflows/_pr-test-stage.yml (+4/-4); .github/workflows/ci-auto-bisect.yml (+1/-1); .github/workflows/pr-states.yml (+1/-1); .github/workflows/pr-test-extra.yml (+4/-4); (+463 more)
LABELS: documentation, quant, lora, Multi-modal, deepseek, speculative-decoding, hicache, blackwell, npu, run-ci
BODY: ## Summary ⏎ - Rename CUDA basic CI suites `stage-a/b/c-*` -> `base-a/b/c-*` so basic and extra CI share a parallel naming scheme (`pr-test.yml` + `pr-test-extra.yml`) ⏎ - Workflow display `name: PR Test` -> `name: PR Test Base`; `pr-states.yml` label `Latest PR Test` -> `Latest PR Test (Base)` ⏎ - Rename `check-stage-health` action -> `check-pr-test-health`; env `SKIP_STAGE_HEALTH_CHECK` -> `SKIP_PR_TEST_HEALTH_CHECK` ⏎ - Drop stale `stage-c-test-deepep …[truncated]

### L1-ce2506e1c6  (L1, 2026-05-15, sha ce2506e1c65c, PR #24314)
TITLE: Deprecate record_nolora_graph dual MoE CUDA graph capture (#24314)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+0/-25); python/sglang/srt/lora/lora_moe_runners.py (+17/-19); python/sglang/srt/model_executor/cuda_graph_runner.py (+22/-76); python/sglang/srt/server_args.py (+0/-9)
LABELS: lora, run-ci
BODY: ## Summary ⏎  ⏎ Reverts #22809. The dual-capture path (separate \"lora\" and \"nolora\" CUDA graphs per batch size) is a bug factory and not that fast. ⏎  ⏎  ⏎ ## What is removed ⏎  ⏎ - `ServerArgs.record_nolora_graph` and the `--record-nolora-graph` CLI flag ⏎ - `RECORD_NOLORA_GRAPH` global, `should_record_nolora_graph()`, and the triton-only guard in `initialize_moe_config()` ⏎ - `_capture_lora_variant` module state plus `get_capture_lora_variant()` / ` …[truncated]

### L1-bda01d2435  (L1, 2026-05-15, sha bda01d24357b, PR #25380)
TITLE: [Disagg] Fix MegaMoE topk_ids dtype mismatch and FakeKVManager missing kv_args (#25380)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.runner.deepgemm_megamoe
FILES: python/sglang/srt/layers/moe/mega_moe.py (+2/-2); python/sglang/srt/disaggregation/fake/conn.py (+1/-0)
BODY: ## Motivation ⏎  ⏎ When serving DeepSeek-V4-Pro with MegaMoE enabled (`SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1`) and disaggregation decode mode (`--disaggregation-transfer-backend fake --ep-dispatch-algorithm fake`), the server crashes during initialization with two independent errors: ⏎  ⏎ **Bug 1 — dtype mismatch in `mega_moe_pre_dispatch`:** ⏎ ``` ⏎ tvm.error.InternalError: Tensor match failed for Tensor<32, 6>[strides=<6, 1>, dtype=int64, device=cuda:0] …[truncated]

### L1-b2c6db0cc4  (L1, 2026-05-16, sha b2c6db0cc429, PR #25406)
TITLE: [MoE] Decouple Mega MoE from DeepEP backend (#25406)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.deep_gemm, L1.runner.deepgemm_megamoe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-1); python/sglang/srt/layers/moe/mega_moe.py (+2/-1); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+0/-1); python/sglang/srt/layers/moe/utils.py (+4/-0); python/sglang/srt/layers/quantization/fp8.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+1/-0); python/sglang/srt/server_args.py (+28/-1); python/sglang/srt/environ.py (+2/-3); test/manual/dsv4/test_b200_flash.py (+0/-1); test/manual/dsv4/test_b200_pro.py (+0/-1); (+8 more)
LABELS: documentation, high priority, deepseek, run-ci
BODY: ## Summary ⏎ - Add `megamoe` as an `--moe-a2a-backend` choice so users can enable Mega MoE without the `deep_ep` library or `--moe-a2a-backend deepep`. ⏎ - Keep `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE` env var as backward-compatible shortcut. ⏎ - Clean up dead/redundant env vars in test and cookbook. ⏎  ⏎ Re-land of #24884 ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): [Run #25952622517](https://github.com/sgl-project/sglang/actions/runs/2595262251 …[truncated]

### L1-aec4022e58  (L1, 2026-05-16, sha aec4022e58c6, PR #25424)
TITLE: [Spec] Clean up draft-window-size handling; extract spec arg setup to arg_groups (#25424)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/argparse_actions.py (+82/-0); python/sglang/srt/arg_groups/speculative_hook.py (+443/-0); python/sglang/srt/models/llama_eagle3.py (+12/-15); python/sglang/srt/server_args.py (+21/-473); python/sglang/srt/speculative/dflash_worker.py (+2/-3); python/sglang/srt/speculative/frozen_kv_mtp_worker.py (+1/-1); test/registered/unit/server_args/test_server_args.py (+13/-10)
LABELS: speculative-decoding, run-ci
BODY: ## Summary ⏎  ⏎ - Tighten `--speculative-draft-window-size` handling (validation, scope warning, deprecation alias) ⏎ - Clean up EAGLE-3 SWA wiring in `LlamaForCausalLMEagle3` (drop kwarg threading, post-init loop) ⏎ - Extract speculative arg setup from `server_args.py` to new `arg_groups/speculative_hook.py`, split per algorithm ⏎ - Move generic argparse `Action` classes to new `arg_groups/argparse_actions.py` ⏎  ⏎ References #24664. ⏎  ⏎ ## Changes ⏎  ⏎ **`--specula …[truncated]

### L1-229cadec04  (L1, 2026-05-16, sha 229cadec0409, PR #25499)
TITLE: Update logging for inplace setting in MoE layer (#25499)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-3)
BODY: Log info message when setting inplace to False for FlashInfer TRTLLM MoE backend. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](h …[truncated]

### L1-568ba7216a  (L1, 2026-05-17, sha 568ba7216a4a, PR #25522)
TITLE: Fix logging for inplace setting in the flashInfer-trtllm backend (#25522)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L1-be3c425788  (L1, 2026-05-17, sha be3c425788db, PR #23760)
TITLE: [MoE] Unify DeepEPMoE+MoriEPMoE through AITER MoeRunner pre/post-permute (#23760)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.runner.framework, L1.runner.aiter, L1.ep.layer, L1.ep.deepep_dispatcher, L1.ep.other_dispatchers, L1.upstream.aiter_moe
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+11/-285); python/sglang/srt/layers/moe/moe_runner/aiter.py (+332/-30); python/sglang/srt/layers/moe/moe_runner/runner.py (+2/-3); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+13/-0); python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+17/-2); python/sglang/srt/layers/moe/utils.py (+9/-0); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py (+1/-1); python/sglang/srt/layers/quantization/fp8.py (+1/-3); python/sglang/srt/layers/quantization/mxfp4.py (+2/-2); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py (+1/-1); (+3 more)
LABELS: quant, run-ci
BODY: ## Motivation ⏎  ⏎ Follow-up to the AITER MoE runner refactor in #23597. After that PR, every quant method (FP8, MXFP4, Quark W4A4 / INT4-FP8, Compressed-Tensors W8A8 FP8, Unquant) routes ``aiter.fused_moe`` through ``MoeRunner`` for the standard ``none`` a2a backend. The two remaining direct ``aiter.fused_moe`` call sites in ``ep_moe/layer.py`` — ``DeepEPMoE.forward_aiter`` and ``MoriEPMoE.run_moe_core`` — were carved out as separate roadmap items.  …[truncated]

### L1-7158a255eb  (L1, 2026-05-17, sha 7158a255ebec, PR #25525)
TITLE: [MoE Refactor] Migrate flashinfer_cutedsl + DeepEP to MoeRunner (#25525)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.runner.framework, L1.runner.flashinfer_cutedsl, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+7/-29); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+103/-13); python/sglang/srt/layers/moe/moe_runner/runner.py (+0/-2); python/sglang/srt/layers/quantization/modelopt_quant.py (+44/-101); test/registered/moe/test_cutedsl_moe.py (+2/-1)
LABELS: quant, run-ci
BODY: ## Motivation ⏎  ⏎ Part of the MoE refactor roadmap (#8715). The roadmap entry ⏎  ⏎ > [ ] `DeepEPMoE.forward_*` ⏎ >   - [ ] `flashinfer_cutedsl.py` + `modelopt_quant.py` ⏎  ⏎ calls for retiring the legacy DeepEP fused-MoE forward path that bypasses the unified `MoeRunner` framework. This PR deprecates the last two pieces of that path: `DeepEPMoE.forward_flashinfer_cutedsl` and `ModelOptNvFp4FusedMoEMethod.apply_without_routing_weights`. ⏎  ⏎ ## Modifications ⏎  ⏎ The  …[truncated]

### L1-6a21dd20b1  (L1, 2026-05-17, sha 6a21dd20b106, PR #25285)
TITLE: Fix EPLB mapping for TopK paths (#25285)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+0/-4); test/registered/cpu/test_topk.py (+34/-0)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ Fix two EPLB logical/physical expert mapping issues in TopK paths: ⏎  ⏎ - keep `biased_topk_impl` and `biased_topk_jit_kernel_impl` returning logical expert ids, then let `_post_process_topk_ids()` apply EPLB mapping exactly once ⏎ - when DeepEP shared-expert fusion runs with EPLB dispatch info, use `ExpertLocationDispatchInfo.num_physical_experts` for the DeepEP interleaved shared-expert remap instead of `router_logits.shape[1]` ⏎  ⏎ The secon …[truncated]

### L1-1f9eda4ea1  (L1, 2026-05-17, sha 1f9eda4ea183, PR #25540)
TITLE: Use DeepGEMM BF16 for unquantized DeepEP LL MoE (#25540)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+14/-45); python/sglang/srt/layers/quantization/unquant.py (+12/-0)
LABELS: quant, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ This is a follow-up to #24906. The merged implementation enabled BF16 DeepEP dispatch for every `quant_config=None` MoE layer, which is too broad. This PR narrows the behavior to the target case: unquantized DeepEP low-latency MoE on CUDA, used by the Qwen3.5 NVFP4 MTP/NextN draft MoE path. ⏎  ⏎ The Qwen3.5 NVFP4 main decode MoE can use `flashinfer_cutedsl + DeepEP`, but the MTP/NextN draft MoE is unquantized (`quant_config=None`). For …[truncated]

### L1-a080358cac  (L1, 2026-05-18, sha a080358cac7d, PR #22822)
TITLE: [Refactor] Refactor DeepEP dispatcher (#22822)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:kernel-correctness-cases(introducing), body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py (+57/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+6/-21); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+77/-31); python/sglang/srt/layers/moe/utils.py (+76/-0); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py (+12/-0); python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w4a4_int4_moe.py (+11/-3); python/sglang/srt/models/deepseek_nextn.py (+0/-6); python/sglang/srt/models/glm4_moe_nextn.py (+3/-1); python/sglang/srt/server_args.py (+10/-0); docs/advanced_features/server_arguments.md (+1/-0); (+20 more)
LABELS: documentation, deepseek, npu, run-ci
DEEP_STUDY: deep-study: introduced the defect fixed in case sglang:b421e60eed (fix PR 26389)
BODY: ## Description ⏎ Refactor the DeepEP dispatcher to introduce structured output dtype control, replacing the ⏎ `SGLANG_DEEPEP_BF16_DISPATCH` environment variable with a `DeepEPOutputDtype` enum and ⏎ automatic detection. This is part of a broader effort to make the dispatch pipeline robust for ⏎ quantized MoE models — especially on Ascend NPU — while reducing memory overhead and ⏎ simplifying user configuration. ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎ The previous dispa …[truncated]

### L1-866793c502  (L1, 2026-05-18, sha 866793c502b7, PR #24933)
TITLE: Amd/deepseek v4 rebase main 0509 (#24933)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+3/-1); python/sglang/jit_kernel/deepseek_v4.py (+26/-0); python/sglang/srt/environ.py (+7/-0); python/sglang/srt/layers/attention/attention_registry.py (+17/-4); python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py (+1265/-0); python/sglang/srt/layers/attention/dsv4/compress_hip.py (+455/-0); python/sglang/srt/layers/attention/dsv4/compressor.py (+16/-2); python/sglang/srt/layers/attention/dsv4/indexer.py (+8/-11); python/sglang/srt/layers/attention/hip_flash_mla.py (+197/-0); python/sglang/srt/layers/attention/nsa/index_buf_accessor.py (+0/-4); (+7 more)
LABELS: high priority, quant, amd, deepseek, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ Enable deepseek v4 model support (merge to main) to ROCm platform. ⏎  ⏎ This is a first PR to add ROCm deepseek v4 model support on SGLang main. ⏎  ⏎ After this PR merged, we can run the v4 flash/pro model on MI35x in eager mode. Then will have subsequent PRs to merge remaining DSv4 optimizations from amd/deepseek_v4 branch. ⏎  ⏎ Some following tasks is under cooking, included ⏎ - https://github.com/sgl-project/sglang/tree/amd/deepseek_ …[truncated]

### L1-d96e593fd0  (L1, 2026-05-18, sha d96e593fd017, PR #25571)
TITLE: [Benchmark] Add SGLANG_SIMULATE_UNIFORM_EXPERTS for balanced expert routing with dummy weights (#25571)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+12/-0); python/sglang/srt/environ.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ When benchmarking MoE models with `--load-format dummy`, random gate weights cause severe expert imbalance. This flag forces a uniform expert distribution, which represents the most optimistic (best-case) token routing. This is useful for benchmarking to set an **upper bound** on serving performance. ⏎  ⏎ ## Change ⏎  ⏎ Add `SGLANG_SIMULATE_UNIFORM_EXPERTS=1` env var that overrides the gating output with a deterministic round-robin e …[truncated]

### L1-6f892047ec  (L1, 2026-05-18, sha 6f892047ecad, PR #25509)
TITLE: [misc] Throw error when single batch overlap is enabled on Hopper  (#25509)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+5/-0)
LABELS: run-ci
BODY: ## Motivation ⏎ #25491 ⏎ This feature will be supported back later. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/ …[truncated]

### L1-878e6b8886  (L1, 2026-05-18, sha 878e6b8886ff, PR #25685)
TITLE: [SP] Fix runtime_max_tokens_per_rank for sequence parallelism (#25685)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py (+10/-5)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ - When using sequence parallelism (SP), the pre-scatter scheduler token count from `get_dp_global_num_tokens()` can exceed the A2A workspace cap, causing issues. ⏎ - This fix uses `x.shape[0]` (the post-scatter tensor size) when `dp_size==1` or SP is active, and reserves the max-across-ranks logic for true DP attention with multiple ranks. ⏎ - Reduces TTFT for llama4x with sequence parallelism enabled. ⏎  ⏎ ## Original commits ⏎  ⏎ - `886108914` ⏎  …[truncated]

### L1-745abd6cc0  (L1, 2026-05-18, sha 745abd6cc00d, PR #25688)
TITLE: Add no_combine support to cutlass_moe_fp4 (#25688)
SOURCES: path_core, path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+3/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-0)
LABELS: quant, run-ci
BODY: ## Summary ⏎  ⏎ - Add `no_combine` parameter to `cutlass_moe_fp4()` so it can return per-expert outputs without combining (matching the triton path). ⏎ - Wire the parameter through from `ModelOptNvFp4FusedMoEMethod` via `moe_runner_config.no_combine`. ⏎ - Enables FP4 MoE with TP1 in no-combine mode (needed for EP dispatch patterns). ⏎  ⏎ ## Original commits ⏎  ⏎ - `5f5737ed0` ⏎  ⏎ ## Test plan ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :hourglass_flowing_sand: [Run  …[truncated]

### L1-b79e4b1e68  (L1, 2026-05-18, sha b79e4b1e687b, PR #25690)
TITLE: [Fix] Try to fix error caused by latest cutedsl packages  (#25690)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); scripts/ci/cuda/ci_install_dependency.sh (+19/-2)
LABELS: dependencies, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Ref: https://github.com/sgl-project/sglang/actions/runs/26055697810/job/76604443741 ⏎ https://github.com/vllm-project/vllm/pull/40082#issuecomment-4349406309  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAIN …[truncated]

### L1-ba2ffcf156  (L1, 2026-05-18, sha ba2ffcf156b5, PR #25569)
TITLE: Add DeepSeekV4 fused MoE Triton autotune support (#25569)
SOURCES: subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/common_utils.py (+1/-0); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+5/-0)
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎ Add DeepSeekV4 fused MoE Triton autotune support to improve inference performance. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ sglang/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py ⏎ sglang/benchmark/kernels/fused_moe_triton/common_utils.py    ⏎  ⏎  ⏎ After enabling DeepSeekV4 fused MoE Triton autotune, the feature works normally: ⏎  ⏎ <img width="2594" height="1628" alt="c8542f34e18be4ed5dc7a9242916d73c" src="https://github.com/user-attac …[truncated]

### L1-78cb38ed5e  (L1, 2026-05-19, sha 78cb38ed5ec4, PR #22918)
TITLE: [FlashInfer v0.6.11] [RL] Support FlashInfer per-token NVFP4 MoE (#22918)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+23/-3); docs/references/environment_variables.md (+1/-0); docs_new/docs/references/environment_variables.mdx (+5/-0); python/sglang/srt/environ.py (+2/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+7/-0); test/registered/backends/test_flashinfer_trtllm_gen_moe_backend.py (+4/-20)
LABELS: documentation, high priority, quant, run-ci, bypass-fastfail
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎ @humansand ⏎  ⏎ FlashInfer PR: ⏎ - https://github.com/flashinfer-ai/flashinfer/pull/3027 ⏎  ⏎ With this recipe, the activation is always online per-token quantized. No QAT or calibration is required for NVFP4 checkpoint conversion. ⏎ ## Modifications ⏎ - When `SGLANG_FLASHINFER_PER_TOKEN_NVFP4_MOE` is true, ignore fp32 input activation scale in checkpoint, and use online per-token path instead ⏎ - Expand test coverage ⏎  ⏎  ⏎ ## Accuracy Test …[truncated]

### L1-d90bc65e30  (L1, 2026-05-19, sha d90bc65e3075, PR #25383)
TITLE: [NPU] Fix TypeError in get_state_buf_infos when index_head_dim is None on MLA (#25383)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/memory_pool_npu.py (+2/-0)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ In PD disaggregation scenarios, NPUMLATokenToKVPool.get_state_buf_infos() unconditionally accesses self.index_k_buffer , which is initialized to None when the model has no index head (i.e., index_head_dim is None ), causing TypeError: 'NoneType' object is not subscriptable . ⏎  ⏎ This issue was triggered by commit d7f4761a ([PD] Refactor hybrid state transfer), which added NPUMLATokenToKVPool to the isinstance check in setup_st …[truncated]

### L1-31e324391b  (L1, 2026-05-19, sha 31e324391bbc, PR #24611)
TITLE: [Codex] Opt Mistral Large performace  (#24611)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=128,N=1024,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=128,N=1024,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8_down.json (+146/-0); python/sglang/srt/server_args.py (+2/-1)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Summary ⏎  ⏎ - Add MistralLarge3 to the model list that auto-enables FlashInfer AllReduce Fusion on supported single-node multi-GPU deployments. ⏎ - Also handle wrapper configs whose outer architecture differs from the text backbone: `mistralai/Mistral-Small-4-119B-2603` loads as `PixtralForConditionalGeneration`, while the text backbone is `MistralLarge3ForCausalLM`. ⏎ - Add tuned FP8 MoE config files for the targeted Mistral path. ⏎ - Leave the explic …[truncated]

### L1-425dffbde3  (L1, 2026-05-19, sha 425dffbde339, PR #24934)
TITLE: DeepSeek V4 MTP Support CP (#24934)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v4_nextn.py (+59/-0); test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py (+46/-0)
LABELS: high priority, deepseek
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ `export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" ⏎ SGLANG_OPT_USE_JIT_INDEXER_METADATA=1 \ ⏎ SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \ ⏎ SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \ ⏎ sglang serve \ ⏎   --trust-remote-code \ ⏎   --model-path /ssd2/models/deepseek-ai/DeepSeek-V4-Pro/ \ ⏎   --tp 8 \ ⏎   --moe-a2a-backend deepep \ ⏎   --enable-nsa-prefill-context-parallel \ ⏎   --nsa-prefill-cp-mo …[truncated]

### L1-7fda3caea4  (L1, 2026-05-19, sha 7fda3caea41e, PR #25356)
TITLE: [AMD] test(sgl-kernel): seed RNG on ROCm in test_moe_topk_sigmoid to fix tie-break flake (#25356)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: sgl-kernel/tests/test_moe_topk_sigmoid.py (+13/-0)
LABELS: sgl-kernel, run-ci
BODY: ## Motivation ⏎  ⏎ `sgl-kernel/tests/test_moe_topk_sigmoid.py` has been intermittently failing on AMD CI (ROCm MI30x/MI35x) on the `atol=0` index comparison against `torch.topk`'s tie-break, e.g. ⏎  ⏎ ``` ⏎ Indices mismatch: torch=[..., 41, ...], SGLang=[..., 32, ...] ⏎ ``` ⏎  ⏎ The test inputs are `torch.randn(...)` without an explicit seed, so near-tied sigmoid scores show up randomly across runs. On ROCm, `torch.topk` is backed by hipCUB (vs CUB on CUDA), and …[truncated]

### L1-044649c23a  (L1, 2026-05-20, sha 044649c23abe, PR #22669)
TITLE: feat: Support flashinfer_cutedsl MoE runner with flashinfer alltoall backend (#22669)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.runner.flashinfer_cutedsl, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+92/-5); python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py (+17/-28); python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-1); python/sglang/srt/models/qwen2_moe.py (+22/-14); python/sglang/srt/server_args.py (+41/-4); test/registered/moe/test_flashinfer_a2a_cutedsl_v2.py (+89/-0)
LABELS: quant, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: Depends on https://github.com/flashinfer-ai/flashinfer/pull/3021 for topk=10. ⏎  ⏎ ## Summary ⏎  ⏎ - Enable CuteDSL FP4 MoE runner (`--moe-runner-backend flashinfer_cutedsl`) to work with FlashInfer one-sided alltoall dispatch (`--moe-a2a-backend flashinfer`) for DP attention + EP configurations ⏎ - Fixes multiple issues discovered when combining CuteDSL with alltoall in DP mode where idle ranks have 0 tokens ⏎  ⏎ ## Changes ⏎  ⏎ ### `server_args.py` ⏎ - A …[truncated]

### L1-65fe32379e  (L1, 2026-05-20, sha 65fe32379edf, PR #24641)
TITLE: [Intel GPU]Support fused_topk for XPU (#24641)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+25/-0)
BODY: ## Motivation ⏎ Support fused_topk for xpu. ⏎ ## Modifications ⏎ Add 'forward_xpu' pass for topk module, and it will use non torch-native pass when "top_k <= 8 and num_experts <= 256" for xpu device. ⏎  ⏎  ⏎  ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #26137614611](https://github.com/sgl-project/sglang/actions/runs/26137614611) ⏎ Latest PR Test (Extra): :x: [Run #26137614558](https://github.com/sgl-project/sglang/actions/runs/26137614558)

### L1-ce7141ef98  (L1, 2026-05-20, sha ce7141ef9876, PR #25860)
TITLE: add git gemm warpper for dispatch_bf16_fp32_backend (#25860)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/deepseek_v4.py (+7/-8)
LABELS: deepseek, jit-kernel
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎ We added a JIT DeepGemm code branch for _dispatch_bf16_fp32_backend to reduce cache generation during the runtime stage. ⏎  ⏎ ## Perf Test ⏎ server command： ⏎ ``` ⏎ SGLANG_SET_CPU_AFFINITY=1 \ ⏎   SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1 \ ⏎   SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1 \ ⏎   SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1 \ ⏎   SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=0 \ ⏎   SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1 \ ⏎   SGLANG_OPT_ …[truncated]

### L1-9f2bc24b35  (L1, 2026-05-20, sha 9f2bc24b3575, PR #25892)
TITLE: Fix/dsv4 flash eagle dummy ima (#25892)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.hash_topk
FILES: python/sglang/srt/layers/moe/hash_topk.py (+18/-0)
ISSUES: #25767 [Bug] DSV4-Flash fails with dummy weights
BODY: ## Motivation ⏎  ⏎ Closes https://github.com/sgl-project/sglang/issues/25767 ⏎  ⏎ `--load-format dummy` only initializes floating-point tensors, leaving `HashTopK.tid2eid` uninitialized because it is an integer lookup table. For DeepSeek-V4-Flash with `flashinfer_mxfp4`, this can produce invalid expert ids and trigger CUDA illegal memory accesses during hash top-k execution, including CUDA graph capture. ⏎  ⏎ ## Modifications ⏎  ⏎ Initialize `HashTopK.ti …[truncated]

### L1-3a6de13cd8  (L1, 2026-05-20, sha 3a6de13cd822, PR #25810)
TITLE: perf(dsv4): add MHC token-count prewarm (#25810)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/mhc.py (+18/-0); python/sglang/srt/model_executor/model_runner.py (+8/-1); python/sglang/srt/models/deepseek_v4.py (+110/-0); python/sglang/srt/models/deepseek_v4_nextn.py (+5/-0)
LABELS: deepseek
DEEP_STUDY: deep-study performance PR ()
BODY: ## Motivation ⏎  ⏎ DeepSeek V4 MHC pre can hit a large one-time cold cost when a rare token-count bucket first enters the server process. In internal DSV4 profiling runs, the long 20-40s forward stalls aligned with first-seen MHC pre split buckets in sparse mixed/decode tail. This patch moves that first-hit cost into startup warmup so live traffic and benchmark measurement do not pay it at the tail. ⏎  ⏎ ## Changes ⏎  ⏎ - Add an optional model-level ke …[truncated]

### L1-847cbada9c  (L1, 2026-05-20, sha 847cbada9c42, PR #25054)
TITLE: Support Gemma4 MoE NVFP4 (#25054)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+1/-1); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+21/-9); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py (+4/-1); python/sglang/srt/layers/quantization/fp8.py (+4/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+57/-17); python/sglang/srt/server_args.py (+7/-0); python/sglang/srt/managers/scheduler.py (+8/-1); python/sglang/srt/models/gemma4_causal.py (+81/-27); python/sglang/srt/models/gemma4_mm.py (+88/-61)
LABELS: high priority, quant, blackwell, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ Support NVFP4 Gemma4 MoE `nvidia/Gemma-4-26B-A4B-NVFP4`. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - Make flashinfer_trtllm the default MoE runner for Gemma4. And also fixed some geglu issue ⏎ - Fix model weights loading for NVFP4 format. ⏎  ⏎ Note: current PR doesn't work with torch compile.  ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ``` ⏎ python -m sglang.launch_server --model-path nvidia/Gemma-4-26B-A4B-NVFP4  --tp-size 1 --enable-torch-compile ⏎ ``` ⏎  ⏎ mm …[truncated]

### L1-e8608bdcb5  (L1, 2026-05-20, sha e8608bdcb544, PR #25367)
TITLE: Fix EPLB redundant experts with shared expert fusion and Waterfill (#25367)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+18/-6); python/sglang/srt/layers/moe/topk.py (+11/-3); python/sglang/srt/model_executor/model_runner.py (+7/-1)
LABELS: run-ci
BODY: ## Summary ⏎ - pass the physical routed expert count (logical routed + redundant) to DeepEP Waterfill ⏎ - detect fused shared checkpoint weights using the logical expert boundary when EPLB metadata is active ⏎ - map those shared weights back to the physical shared slot layout used by FusedMoE ⏎  ⏎ ## Testing ⏎ - PYTHONPYCACHEPREFIX=/tmp/sglang_pycache_check python -m py_compile python/sglang/srt/layers/moe/fused_moe_triton/layer.py python/sglang/srt/model_ex …[truncated]

### L1-8fa56a0ab1  (L1, 2026-05-20, sha 8fa56a0ab145, PR #25907)
TITLE: Fix FlashInfer A2A token cap sizing (#25907)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py (+6/-3)
LABELS: run-ci, run-ci-extra
BODY: ## Summary ⏎  ⏎ - Fix `max_num_tokens` in `FlashinferDispatcher`: the previous code multiplied the per-rank env var by `ep_size`, double-counting the scaling already done inside FlashInfer's workspace sizing (`moe_a2a_get_workspace_size_per_rank()`). ⏎ - Raise the default from 1024 to 16384 tokens per rank. ⏎  ⏎ ## Original commits ⏎  ⏎ - `9b4258cdc` ⏎  ⏎ ## Test plan ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #26204297059](https://github.com/sgl-project/ …[truncated]

### L1-19f55c0e6d  (L1, 2026-05-21, sha 19f55c0e6d6f, PR #25884)
TITLE: [Refactor] major JIT kernel clean up for dsv4 (#25884)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.routing.hash_topk, L1.runner.deep_gemm, L1.runner.deepgemm_megamoe
FILES: python/sglang/jit_kernel/dsv4/moe.py (+216/-0); python/sglang/jit_kernel/dsv4/topk.py (+88/-0); python/sglang/srt/layers/moe/hash_topk.py (+1/-1); python/sglang/srt/layers/moe/mega_moe.py (+1/-1); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+2/-2); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+1/-1); python/sglang/srt/layers/moe/topk.py (+1/-1); python/sglang/jit_kernel/csrc/deepseek_v4/topk_1024.cuh (+0/-336); python/sglang/jit_kernel/csrc/deepseek_v4/topk_v1.cuh (+13/-9); python/sglang/jit_kernel/deepseek_v4.py (+0/-1036); (+13 more)
LABELS: deepseek, jit-kernel
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Split deepseek v4.py into multiple files ⏎ 2. Reuse torch.mm instead of inline cpp cublas handler ⏎ 3. Unify topk.cuh and topk_1024.cuh (rename to topk_v1.cuh) ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINT …[truncated]

### L1-b765faee30  (L1, 2026-05-21, sha b765faee3049, PR #25678)
TITLE: [MoE Refactor] deprecate forward_npu and NpuFuseEPMoE (#25678)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.hardware.cpu_npu_musa, L1.ep.layer, L1.ep.other_dispatchers
FILES: python/sglang/srt/hardware_backend/npu/moe/fuseep.py (+171/-0); python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py (+123/-5); python/sglang/srt/layers/moe/ep_moe/layer.py (+6/-243); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+13/-14); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+0/-2); python/sglang/srt/layers/moe/token_dispatcher/fuseep.py (+0/-98); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+0/-1); python/sglang/srt/layers/quantization/unquant.py (+46/-1)
LABELS: quant, npu, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Continues the MoE refactor roadmap (#8715). Deprecates `DeepEPMoE.forward_npu` and the `NpuFuseEPMoE` class so the NPU paths fit the unified `FusedMoE.forward` → `dispatcher.dispatch` → `quant_method.apply` → `dispatcher.combine` pipeline. Tracks the roadmap items: ⏎  ⏎ ## Modifications ⏎  ⏎ Follows the mega_moe pattern (free-function bypass + quant_method weight hook) rather than introducing a new dispatcher/format. ⏎  ⏎ - `python/sglang/srt/ …[truncated]

### L1-a449ee4822  (L1, 2026-05-21, sha a449ee4822ee, PR #25576)
TITLE: [Deps] Use cu13 extra for nvidia cutlass dsl (#25576)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1)
LABELS: dependencies, run-ci
ISSUES: #25564 [Bug] Qwen-3.5 on B300 (sm_103) crashes in flash-attn-4 cute kernel — assertion at flash_fwd_sm100.py:162 (fix exists in Dao-AILab/flash-attention#2572; sglang needs to bump flash-attn-4)
BODY: ## Summary ⏎  ⏎ Add `[cu13]` extra to `nvidia-cutlass-dsl` dependency since we already default to CUDA 13 and optionally upgrade to 4.5.1 ⏎  ⏎ Should fix #25564 ⏎  ⏎ ## Test Plan ⏎  ⏎ CI ⏎  ⏎  ⏎  ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #26077315178](https://github.com/sgl-project/sglang/actions/runs/26077315178) ⏎ Latest PR Test (Extra): :warning: **Not enabled** -- add `run-ci-extra` label to opt in.

### L1-81d686d9fa  (L1, 2026-05-21, sha 81d686d9fa2f, PR #26004)
TITLE: Default MegaMoE to W4A8 for Max-Throughput recipe (#26004)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx (+13/-2)
LABELS: documentation, deepseek
BODY: ## Summary ⏎ - Auto-select MegaMoE W4A8 when switching to Max-Throughput recipe on supported hardware (Blackwell) ⏎ - Skip `--deepep-config` flag when MegaMoE is enabled (MegaMoE uses its own backend, not DeepEP) ⏎ - MegaMoE stays disabled on unsupported hardware (Hopper) or recipes (low-latency, cp) ⏎ - Users can still manually switch to W4A4 or Disabled after the auto-selection

### L1-4ea8282cb7  (L1, 2026-05-21, sha 4ea8282cb7ab, PR #25938)
TITLE: [Revert] nvidia-cutlass-dsl[cu13] 4.5.1 -> 4.5.0 (#25938)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1)
LABELS: high priority, dependencies, run-ci, bypass-fastfail
DEEP_STUDY: deep-study revert record: explicit_rollback of PR(s)  reason=build_or_dependency
BODY: ## Problem ⏎  ⏎ `nvidia-cutlass-dsl[cu13]` has **additive extras** on PyPI: both `-libs-base` AND `-libs-cu13` are installed together when `[cu13]` is requested. They write to the same `site-packages` paths with different content, causing a `GPUModuleOp TypeError` at kernel-compile time ([vllm-project/vllm#40082](https://github.com/vllm-project/vllm/issues/40082)). ⏎  ⏎ The correct libs package to keep depends on GPU family: ⏎  ⏎ | Runner | Required libs | W …[truncated]

### L1-b9ae8353d2  (L1, 2026-05-21, sha b9ae8353d2af, PR #25257)
TITLE: [NPU] Support model DeepSeek-OCR and DeepSeek-OCR-2 (#25257)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek.py (+11/-3)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ Support DeepSeek-OCR and DeepSeek-OCR-2 model on NPU platform. ⏎  ⏎ ## Modifications ⏎  ⏎ In NPU environment, `fused_moe_npu` from `sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu` should be used in `DeepseekMoE::forward()`. ⏎  ⏎ ## Accuracy Tests ⏎ Run the following command to start the SGLang inference service: ⏎ ``` ⏎ #!/bin/bash ⏎  ⏎ export PYTHONPATH=/root/RemoteDev/sglang/python:$PYTHONPATH ⏎ export SGLANG_LOG_PATH=/r …[truncated]

### L1-caa9f08294  (L1, 2026-05-21, sha caa9f0829408, PR #25958)
TITLE: [CI] Force-reinstall nvidia-cutlass-dsl-libs-cu13 last to avoid wheel-mix TypeError (#25958)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); scripts/ci/cuda/ci_install_dependency.sh (+29/-0)
LABELS: dependencies, run-ci, bypass-fastfail
BODY: ## Root cause ⏎  ⏎ `nvidia-cutlass-dsl[cu13]` has **additive PyPI extras** — installing it pulls in both `nvidia-cutlass-dsl-libs-base` AND `nvidia-cutlass-dsl-libs-cu13`. The two wheels ship **intentionally-different content for the same paths**: ⏎  ⏎ | Path | `-libs-base` | `-libs-cu13` | ⏎ |------|--------------|--------------| ⏎ | `cutlass/_mlir/dialects/_gpu_ops_gen.py` | calls `super().__init__(self.build_generic(...))` (new-style single object) | call …[truncated]

### L1-a24c374f84  (L1, 2026-05-21, sha a24c374f8444, PR #25531)
TITLE: [lora] Remove synchronous .any().item() guard in LoRA MoE prefill path (#25531)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/lora/lora_moe_runners.py (+9/-18); python/sglang/srt/lora/layers.py (+5/-0)
LABELS: lora, run-ci, run-ci-extra
BODY: ## Motivation ⏎ [before] ⏎ <img width="1259" height="643" alt="before" src="https://github.com/user-attachments/assets/834282f8-ee88-4781-9f4d-f1c29ec165c1" /> ⏎  ⏎ [after] ⏎ <img width="1262" height="719" alt="after" src="https://github.com/user-attachments/assets/8f051047-754e-4144-962e-bc1108d14d16" /> ⏎  ⏎  ⏎ During LoRA+MoE prefill, `_add_lora_gate_up_delta` calls `.any().item()` on a GPU tensor every layer to check whether any LoRA adapter is activ …[truncated]

### L1-88a37d7405  (L1, 2026-05-22, sha 88a37d740511, PR #26057)
TITLE: [docs] DeepSeek-V4 cookbook: split Quantization axis, add H100 SGLang FP8 (#26057)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+273/-68); docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx (+115/-25)
LABELS: documentation, deepseek
BODY: ## Motivation ⏎  ⏎ The DeepSeek-V4 cookbook currently encodes the FP4 / FP8 choice into the hardware radio label itself (`H200 (FP8)` vs `H200 (FP4)`, etc.). That conflates two orthogonal dimensions, hides the H100 SGLang-FP8 path entirely, and forces the user to mentally translate between "hardware I have" and "checkpoint I want to run." ⏎  ⏎ This PR splits the two: hardware row is just the GPU (B200 / B300 / GB200 / GB300 / H200 / H100), and a new **Qu …[truncated]

### L1-bd6c7e713c  (L1, 2026-05-22, sha bd6c7e713cd0, PR #26025)
TITLE: [fix] Fallback DeepGEMM activation for unsupported shapes (#26025)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.runner.deep_gemm
FILES: python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+5/-2)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ Fall back to the existing non-JIT masked SiLU+mul quant path when the DeepGEMM JIT EP activation kernel does not support the current `N/G` shape. Supported shapes still use the JIT activation fast path. (N/G is not multiple of 4) ⏎  ⏎ ## Error Reproduce ⏎  ⏎ ```bash ⏎ python3 -m sglang.launch_server \ ⏎   --model-path Qwen/Qwen3-30B-A3B-FP8 \ ⏎   --trust-remote-code \ ⏎   --host 0.0.0.0 \ ⏎   --port 30000 \ ⏎   --tensor-parallel-size 4 \ ⏎   - …[truncated]

### L1-2df9e8b4b3  (L1, 2026-05-22, sha 2df9e8b4b331, PR #25189)
TITLE: [perf] DeepSeekV3: drop redundant FP32 upcasts in trtllm MoE paths (#25189)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+2/-14); python/sglang/srt/models/deepseek_v2.py (+1/-0)
LABELS: high priority, deepseek, run-ci, bypass-fastfail
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎ Drop the explicit `.to(torch.float32)` upcasts on `router_logits` before each `trtllm_*_moe` call for DeepSeekV3 routing. flashinfer ≥ 0.6.8 (`trtllm_fp4_block_scale_moe`, `trtllm_fp8_block_scale_moe`, `trtllm_fp8_per_tensor_scale_moe`) accept BF16 `router_logits` directly, so the explicit FP32 upcasts are no longer required. ⏎  ⏎ flashinfer is pinned at `0.6.11.post1` in `python/pyproject.toml`, well past the 0.6.8 cutoff. ⏎  ⏎ ## Changes ⏎  ⏎ * …[truncated]

### L1-cadfa2d025  (L1, 2026-05-22, sha cadfa2d025d3, PR #23351)
TITLE: Support piecewise CUDA graph with NSA (#23351)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/hadamard.py (+9/-0); python/sglang/srt/compilation/piecewise_context_manager.py (+7/-0); python/sglang/srt/configs/model_config.py (+0/-3); python/sglang/srt/layers/attention/dsa/dsa_indexer.py (+167/-35); python/sglang/srt/layers/attention/dsa_backend.py (+14/-3); python/sglang/srt/layers/layernorm.py (+20/-1); python/sglang/srt/layers/radix_attention.py (+14/-0); python/sglang/srt/model_executor/model_runner.py (+6/-0); python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py (+4/-0); python/sglang/srt/models/deepseek_v2.py (+1/-0); (+2 more)
LABELS: deepseek, run-ci, piecewise-cuda-graph, jit-kernel, run-ci-extra
BODY: ## Motivation ⏎  ⏎ GLM-5/DSV3.2 currently doesn't allow piecewise CUDA graph due to incompatibilities in NSA attention backend and NSA indexer. This commit fixes the incompatibilities on the conditions ⏎ 1. is not context parallel ⏎ 2. is on CUDA ⏎  ⏎ `_store_index_k_cache` and `_get_topk_ragged` are excluded from CUDA graph for now, will revisit whether they can be in CUDA graph in a future PR. ⏎  ⏎ Benchmark results ⏎ <img width="2684" height="1055" alt …[truncated]

### L1-81cd338fcc  (L1, 2026-05-23, sha 81cd338fcc8b, PR #26164)
TITLE: [docs] DeepSeek-V4 cookbook: balanced MegaMoE cap, H200 Pro FP4 mem-frac, nsa-* compat, PD-disagg fixes (#26164)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx (+66/-7)
LABELS: documentation, deepseek
BODY: ## Summary ⏎  ⏎ Five small fixes to the DSV4 deployment-command generator (`docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`). ⏎  ⏎ ### 1. Blackwell + Balanced + MegaMoE → cap per-rank dispatch buffer at 4096 ⏎ Balanced recipe always runs MTP (1/1/2); the draft pass needs more GMEM headroom than max-throughput's `=8320`. Adds `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096` whenever `megamoe !== \"disabled\" && recipe === \"balan …[truncated]

### L1-b0ce16d0c5  (L1, 2026-05-23, sha b0ce16d0c577, PR #23292)
TITLE: [CP] 1/N: Support MLA Prefill Context Parallel (#23292)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+128/-56); python/sglang/srt/layers/communicator.py (+10/-4); python/sglang/srt/layers/communicator_dsa_cp.py (+3/-2); python/sglang/srt/layers/utils/cp_utils.py (+36/-19); python/sglang/srt/model_executor/cuda_graph_runner.py (+14/-3); python/sglang/srt/model_executor/model_runner.py (+7/-0); python/sglang/srt/models/deepseek_common/attention_backend_handler.py (+7/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+2/-1); python/sglang/srt/models/deepseek_nextn.py (+31/-8); python/sglang/srt/models/deepseek_v2.py (+73/-14); (+11 more)
LABELS: deepseek, run-ci, run-ci-extra
ISSUES: #22896 [Feature] Prefill Context Parallelism Support for MLA Models
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Part of Context Parallelism series https://github.com/sgl-project/sglang/issues/21788. Upon merge will close https://github.com/sgl-project/sglang/issues/22896 and https://github.com/sgl-project/sglang/issues/22692 ⏎  ⏎ Extend SGLang's prefill context parallelism (CP) to MLA-based models (DeepSeek V3 / R1, Kimi K2.5) on the `fa3` attention backend, unlocking multi-GPU long-prefill throughput for MLA architectures. ⏎  ⏎ **Insight.** M …[truncated]

### L1-89ff2bc111  (L1, 2026-05-23, sha 89ff2bc1115c, PR #26026)
TITLE: [bug fix] Fix 3 issues when using Gemma4 MTP (#26026)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/gemma4_causal.py (+11/-6); python/sglang/srt/models/gemma4_mtp.py (+2/-0); python/sglang/srt/server_args.py (+7/-5)
LABELS: run-ci
BODY: ## Modifications ⏎  ⏎ - pp_group is missing in MTP class due to it skipped Gemma4CausalLM.__init__ ⏎  ⏎ ``` ⏎      File "transformers/modeling_utils.py", line 1395, in post_init                                                                                  ⏎          self.init_weights()                                                                                                                         ⏎      File "transformers/modeling_utils.py", l …[truncated]

### L1-af8f66940e  (L1, 2026-05-23, sha af8f66940e9b, PR #25898)
TITLE: [AMD] Dsv4/pr1 fix run time issue (#25898)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/dsv4/moe.py (+32/-19); python/sglang/jit_kernel/dsv4/topk.py (+10/-4); python/sglang/jit_kernel/triton/hash_topk.py (+99/-0); sgl-kernel/python/sgl_kernel/top_k.py (+32/-0); python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh (+2/-2); python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh (+2/-2); python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh (+21/-9); python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh (+10/-4); python/sglang/jit_kernel/dsv4/attn.py (+13/-7); python/sglang/jit_kernel/dsv4/elementwise.py (+21/-4); (+22 more)
LABELS: amd, deepseek, sgl-kernel, run-ci, jit-kernel, run-ci-extra
BODY: ## Motivation ⏎  ⏎ DeepSeek-V4 (DSV4) on AMD MI300X/MI350X GPUs encounters multiple runtime failures when serving inference workloads: ⏎  ⏎ 1. **GPU fault (HSA_STATUS_ERROR_EXCEPTION)** — The core `CompressStatePool` is initialized without `swa_page_size`, defaulting to 0. This causes division-by-zero in `translate_from_swa_loc_to_state_loc()`, producing out-of-bounds `state_loc` indices (e.g., 525448 vs buffer size 470528) that trigger a hardware ex …[truncated]

### L1-7f45bcdd2a  (L1, 2026-05-24, sha 7f45bcdd2ab8, PR #25948)
TITLE: [dsv4] support eplb (#25948)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.hash_topk
FILES: python/sglang/srt/layers/moe/hash_topk.py (+4/-0); python/sglang/srt/models/deepseek_v4.py (+14/-6)
LABELS: deepseek
BODY: ## Motivation ⏎  ⏎ Support EPLB for deepseek v4 (with megaMoE disabled). ⏎  ⏎ ## Modifications ⏎  ⏎ Add the eplb context in the forward of DeepseekV4Model. Without the context, the layer_idx used in eplb distribution gatherer is None which causes crash in prefill node and stats inaccuracy issue in decode node. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Not relevant. ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎ With this pr, eplb works normally on both prefill and decode. ⏎  ⏎ Server …[truncated]

### L1-e27d4fb70f  (L1, 2026-05-25, sha e27d4fb70f33, PR #25775)
TITLE: [Perf][Qwen3.5] Add case 512 to topkGatingSoftmaxKernelLauncher, (#25775)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_softmax
FILES: sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu (+4/-1); sgl-kernel/benchmark/bench_moe_topk_softmax.py (+1/-1); sgl-kernel/tests/test_moe_topk_softmax.py (+1/-1)
LABELS: sgl-kernel, run-ci, run-ci-extra
DEEP_STUDY: deep-study performance PR ()
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ Add case 512 to topkGatingSoftmaxKernelLauncher, enabling the fused single-kernel path for models with 512 experts (e.g. Qwen3.5-397B-A17B). ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎ **num_experts=512, topk=10, bf16, H200：** ⏎  ⏎ | num_tokens | before (µs) | after (µs) | Δ  | ⏎ |:----------:|:-----------:|:----------:|:------:| ⏎ | 1          | 18.63       | 10.39      | **1.79x** | ⏎ | 2        …[truncated]

### L1-59cad671e2  (L1, 2026-05-25, sha 59cad671e2a8, PR #25391)
TITLE: Support DeepSeek V4 DeepEP Waterfill (#25391)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py, L1.routing.hash_topk
FILES: python/sglang/srt/layers/moe/hash_topk.py (+30/-2); python/sglang/srt/layers/moe/topk.py (+5/-10); python/sglang/srt/model_executor/model_runner.py (+7/-4); python/sglang/srt/models/deepseek_v4.py (+16/-0)
LABELS: deepseek
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ This PR adds DeepSeek V4 support for DeepEP Waterfill: ⏎  ⏎ - keep shared-expert fusion enabled for DeepSeek V4 when `--enable-deepep-waterfill` is set, so the shared expert can be dispatched through DeepEP as the extra Waterfill expert ⏎ - prepare both `TopK` and V4 `HashTopK` modules with `DeepEPWaterfillBalancer` ⏎ - add Waterfill expansion support to `HashTopK` ⏎ - record DeepSeek V4 expert distribution with layer context for EPLB pl …[truncated]

### L1-3f5e2c7688  (L1, 2026-05-25, sha 3f5e2c768825, PR #26208)
TITLE: [AMD] Dsv4/pr2 compressor opt (#26208)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py, L1.runner.aiter, L1.upstream.aiter_moe
FILES: python/sglang/srt/layers/moe/moe_runner/aiter.py (+5/-0); python/sglang/srt/layers/moe/topk.py (+39/-13); docs/diffusion/compatibility_matrix.md (+0/-2); python/sglang/srt/environ.py (+4/-0); python/sglang/srt/layers/activation.py (+14/-0); python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py (+13/-5); python/sglang/srt/layers/attention/dsv4/compress_hip.py (+5/-4); python/sglang/srt/layers/attention/dsv4/compressor.py (+37/-1); python/sglang/srt/layers/attention/dsv4/compressor_v2.py (+516/-25); python/sglang/srt/layers/attention/dsv4/fused_compress_triton.py (+954/-0); (+21 more)
LABELS: documentation, amd, deepseek, sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ This PR improves DeepSeek-V4 inference performance on AMD ROCm by reducing decode/prefill hot-path overhead in compressor, indexer, and fused attention execution. ⏎ It also consolidates kernel options so we can enable high-performance fused paths with clearer runtime flags while maintaining numerical correctness checks. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add DSV4 fused compress implementations (`fused_compress_kernel.py`, `fused_compress_tr …[truncated]

### L1-137168539a  (L1, 2026-05-26, sha 137168539a73, PR #19493)
TITLE: [Perf][Moe]improve cutlass_moe_fp4 performance by using apply_router_weight_on_i… (#19493)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+7/-5)
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎ when i running with this cmd ⏎ ```bash ⏎ model_path=/pfs/pfs-OqLB2M/models/DeepSeek-R1-0528-NVFP4-v2 ⏎ TP_SIZE=8 ⏎ DP_SIZE=8 ⏎ EP_SIZE=1 ⏎ NNODES=1 ⏎ NODE_RANK=0 ⏎ export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 ⏎ export NCCL_MNNVL_ENABLE=1 ⏎ export NCCL_CUMEM_ENABLE=1 ⏎ export PYTHONUNBUFFERED=1 ⏎ export SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 ⏎ export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 ⏎ python3 -m sglang.launch_server \ ⏎     --nnodes $ …[truncated]

### L1-6afebc278a  (L1, 2026-05-26, sha 6afebc278ad9, PR #26413)
TITLE: [docs] DeepSeek-V4 cookbook: note cu129 image for GB200 Pro DeepEP backend (#26413)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx (+14/-0)
LABELS: documentation, deepseek
BODY: ## Summary ⏎  ⏎ - The default `lmsysorg/sglang:latest` image ships CUDA 13 and does not include a compatible DeepEP supported `mnnvl`, so users hit failures unless they swap to `lmsysorg/sglang:latest-cu129`. ⏎ - Surface a `# NOTE:` line prepended to the generated command for the `(gb200, big, megamoe=disabled, deepep-in-flags)` combination — same shell-comment style as the existing GB200 multinode env hint right above it. ⏎  ⏎ The check uses `flags.s …[truncated]

### L1-7ef06bfc06  (L1, 2026-05-26, sha 7ef06bfc06ec, PR #26088)
TITLE: GLM-4.7-Flash: standalone MLA impl and MLA NextN/MTP (#26088)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+7/-4); python/sglang/srt/model_loader/weight_utils.py (+7/-1); python/sglang/srt/models/glm4_moe_lite.py (+603/-81); python/sglang/srt/models/glm4_moe_lite_nextn.py (+182/-0)
LABELS: run-ci, run-ci-extra
BODY: ## Motivation ⏎    ⏎   `glm4_moe_lite` (GLM-4.7-Flash, MLA) had a few issues when running with the ⏎   native SGLang implementation: ⏎  ⏎   - It subclassed the `deepseek_v2.py` wrapper classes, so DSV4/DSA/CP changes ⏎     there kept leaking in and breaking it. ⏎   - The MoE gate computed routing logits in the model dtype (bf16), but GLM ⏎     requires an FP32 gate projection — wrong-precision routing corrupts output. ⏎   - With MTP/EAGLE speculative deco …[truncated]

### L1-6c8128650e  (L1, 2026-05-26, sha 6c8128650e1f, PR #22627)
TITLE: [Bugfix] Fix flashinfer_cutlass MoE crash when intermediate_size_per_partition is not 16-aligned (#22627)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/modelopt_quant.py (+38/-0)
LABELS: documentation, quant, amd, dependencies, lora, Multi-modal, deepseek, speculative-decoding, hicache, sgl-kernel
BODY: ## Summary ⏎  ⏎ - Fix crash in `flashinfer_cutlass MoE backend ` when `intermediate_size_per_partition` is not a multiple of 16 (e.g., NemotronH FP8 at TP=8: 7688 / 8 = 961) ⏎ - Add 16-byte alignment padding for flashinfer_cutlass in` ModelOptFp8MoEMethod.process_weights_after_loading`, following the same post-load padding pattern as `ModelOptNvFp4FusedMoEMethod` for flashinfer_trtllm ⏎ - Move moe_runner_backend auto-resolution (`AUTO` → `FLASHINFER_ …[truncated]

### L1-98eb84497d  (L1, 2026-05-26, sha 98eb84497dca, PR #26148)
TITLE: [PP] Skip PP output communication for pure chunked prefill batches (#26148)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/environ.py (+3/-0); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/managers/scheduler.py (+5/-0); python/sglang/srt/managers/scheduler_components/batch_result_processor.py (+43/-0); python/sglang/srt/managers/scheduler_pp_mixin.py (+55/-5); python/sglang/srt/managers/utils.py (+5/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ In pipeline parallelism (PP), the last rank sends sampled `next_token_ids` to rank 0 after every forward pass. However, for chunked prefill requests that are **not** in their final chunk, `process_batch_result_prefill` only decrements a counter (`inflight_middle_chunks -= 1`) and completely discards the received tokens — they are never appended to `output_ids`. ⏎  ⏎ When an entire microbatch consists solely of non-final chunked p …[truncated]

### L1-d6032c04b6  (L1, 2026-05-26, sha d6032c04b665, PR #26451)
TITLE: [docs] Fix V4 Pro balanced recipe (#26451)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+2/-2); docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx (+4/-10)
LABELS: documentation, deepseek
BODY: ## Motivation ⏎  ⏎ In the interactive DeepSeek-V4 deployment generator (`docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`), the **Balanced** recipe emitted different MoE backends for the two model variants on Blackwell (B200/B300): ⏎  ⏎ - **Flash** → `--moe-a2a-backend deepep` ⏎ - **Pro** → `--moe-runner-backend flashinfer_mxfp4` (plus `--disable-flashinfer-autotune`, `--chunked-prefill-size 32768`, `--swa-full-tokens-ratio 0.1`) ⏎  ⏎ This PR a …[truncated]

### L1-14f81a67d9  (L1, 2026-05-27, sha 14f81a67d94c, PR #26421)
TITLE: chore: bump sglang-kernel version to 0.4.3 (#26421)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci, run-ci-extra
BODY: ## Summary ⏎  ⏎ This PR bumps the `sglang-kernel` version to `0.4.3` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.4.3` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the `sglang-kernel` dependency are updat …[truncated]

### L1-dea85c30f4  (L1, 2026-05-27, sha dea85c30f48e, PR #23837)
TITLE: Add Ling_2_6 (#23837)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json (+146/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e_down.json (+164/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20.json (+146/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20_down.json (+164/-0); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+5/-0); python/sglang/srt/layers/attention/linear/lightning_backend.py (+16/-6); python/sglang/srt/mem_cache/kv_cache_builder.py (+2/-0); python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py (+4/-3); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+80/-24); python/sglang/srt/models/bailing_moe_linear.py (+78/-34); (+2 more)
LABELS: deepseek, run-ci, run-ci-extra
BODY: ## Motivation ⏎ This is used for support Bailing models, Ling-2.6 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.g …[truncated]

### L1-d9d719b270  (L1, 2026-05-27, sha d9d719b27072, PR #26309)
TITLE: [npu] [bugfix] Add contiguous operation during quantized weight loading. (#26309)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py (+12/-4)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ Add a contiguous operation during quantized weight loading , which avoids the weight transmission error. ⏎  ⏎ ## Modifications ⏎  ⏎ fused_moe_method_npu.py ⏎  ⏎ ## before this pr ⏎  ⏎ Once EPLB is enabled, a weight redistribution process is triggered; this subsequently leads to the following error because the quantized model's weights are non-contiguous. ⏎  ⏎ <img width="373" height="73" alt="image" src="https://github.com/user-attachments …[truncated]

### L1-b4808d44da  (L1, 2026-05-28, sha b4808d44da2c, PR #25486)
TITLE: Use Cute-DSL MXFP8 quantize kernels (#25486)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py (+7/-2)
LABELS: run-ci
BODY: <img width="738" height="662" alt="Screenshot 2026-05-16 at 12 55 57 PM" src="https://github.com/user-attachments/assets/5615bb69-5d79-4c03-b56d-c536042c4c57" /> ⏎  ⏎ This is needed for Deepseek V4 ⏎  ⏎ BS = 1 MTP ⏎ Before: ⏎ <img width="1250" height="798" alt="Screenshot 2026-05-17 at 12 09 19 AM" src="https://github.com/user-attachments/assets/d27e2bf3-739c-48c4-9e0e-801b9fa702da" /> ⏎  ⏎ After: ⏎ <img width="1250" height="790" alt="Screenshot 2026-05-1 …[truncated]

### L1-714fdd9723  (L1, 2026-05-28, sha 714fdd972342, PR #25061)
TITLE: Fix MiniMax-M2.7 on CPU (#25061)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+14/-0); sgl-kernel/csrc/cpu/moe.cpp (+9/-6); python/sglang/srt/models/minimax_m2.py (+27/-0)
LABELS: sgl-kernel, intel, cpu, run-ci
BODY: ## Motivation ⏎ Fixes MiniMax-M2.7 on CPU ⏎  ⏎  ⏎ ## Modifications ⏎ - Make `fused_topk_cpu` fall back to torch native when `correction_bias is not None or scoring_func != "softmax"` since this is unsupported in kernel yet. ⏎ - Add `_forward_cpu` to `MiniMaxM2QKRMSNorm` and dispatch to it on CPU. ⏎ - Handle uneven TP sharding in `MiniMaxM2RMSNormTP.weight_loader`. ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ gsm8k: 0.930 ⏎ hellaswag: 0.750 ⏎ ```sh ⏎ # Server ⏎ python3 -m sgla …[truncated]

### L1-435c4ffb30  (L1, 2026-05-28, sha 435c4ffb3081, PR #26609)
TITLE: [CI] Clean DeepSeek V4 tests and installation scripts (#26609)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: scripts/ci/cuda/ci_install_deepep.sh (+24/-1); .github/workflows/pr-test-extra.yml (+15/-0); .github/workflows/pr-test.yml (+12/-12); docker/Dockerfile (+0/-8); scripts/ci/cuda/ci_install_dsv4_dep.sh (+0/-161); scripts/ci/runner_configs.yml (+2/-4); test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py (+1/-1); test/registered/disaggregation/test_disaggregation_dsv4.py (+1/-1); test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py (+2/-2); test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py (+2/-2); (+3 more)
LABELS: deepseek
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L1-e33bbbb467  (L1, 2026-05-28, sha e33bbbb467e4, PR #26402)
TITLE: [5/N] Quantization Refactor: GPTQ schemes and kernel split (#26402)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py (+382/-0); python/sglang/srt/hardware_backend/npu/quantization/gptq_kernels.py (+315/-0); python/sglang/srt/layers/quantization/__init__.py (+13/-1); python/sglang/srt/layers/quantization/auto_round.py (+2/-2); python/sglang/srt/layers/quantization/gptq.py (+0/-1571); python/sglang/srt/layers/quantization/gptq/__init__.py (+42/-0); python/sglang/srt/layers/quantization/gptq/gptq.py (+617/-0); python/sglang/srt/layers/quantization/gptq/schemes/__init__.py (+16/-0); python/sglang/srt/layers/quantization/gptq/schemes/gptq_linear.py (+168/-0); python/sglang/srt/layers/quantization/gptq/schemes/gptq_marlin.py (+154/-0); (+2 more)
LABELS: run-ci, run-ci-extra
BODY: ## Summary ⏎  ⏎ Refactor GPTQ quantization to follow the scheme/kernel split used by AWQ: ⏎  ⏎ - Move GPTQ config and method entry points into the `gptq` package. ⏎ - Split linear, Marlin linear, and MoE schemes into separate scheme modules. ⏎ - Move CUDA/NPU GPTQ kernel helpers into hardware backend specific modules. ⏎ - Preserve embedding handling through `get_linear_quant_method` so non-linear layers are not incorrectly assigned linear quant methods. ⏎  ⏎ ## Va …[truncated]

### L1-be32df33b9  (L1, 2026-05-28, sha be32df33b951, PR #26437)
TITLE: [MUSA] Fix startup with patched torchada (#26437)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: 3rdparty/amd/wheel/sglang/pyproject.toml (+1/-1); python/pyproject_other.toml (+1/-1); python/sglang/jit_kernel/utils.py (+7/-1); sgl-kernel/pyproject_musa.toml (+1/-1)
LABELS: dependencies, sgl-kernel, run-ci, mthreads, jit-kernel
BODY: ## Summary ⏎  ⏎ - Bump MUSA torchada requirements to `>=0.1.57` so SGLang picks up MooreThreads/torchada#70. ⏎ - Skip CUDA PDL arch probing on MUSA runtime. ⏎  ⏎ ## Motivation ⏎  ⏎ SGLang can fail during MUSA startup with: ⏎  ⏎ `AttributeError: module 'triton.language.extra' has no attribute 'cuda'` ⏎  ⏎ The torchada-side patch is available in https://github.com/MooreThreads/torchada/pull/70, so this updates the dependency and avoids the CUDA-specific PDL p …[truncated]

### L1-3bdea78ad1  (L1, 2026-05-29, sha 3bdea78ad11d, PR #26565)
TITLE: model: support Step-3.7-Flash (#26565)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+18/-2); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+1/-0); docs_new/cookbook/autoregressive/StepFun/Step-3.7-Flash.mdx (+324/-0); docs_new/cookbook/autoregressive/StepFun/Step3.5.mdx (+1/-1); docs_new/docs.json (+1/-0); docs_new/src/snippets/autoregressive/step-37-flash-deployment.jsx (+394/-0); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/model_config.py (+12/-1); python/sglang/srt/configs/step3p5.py (+2/-0); python/sglang/srt/configs/step3p7.py (+97/-0); (+7 more)
LABELS: documentation, high priority, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L1-1c2857b064  (L1, 2026-05-29, sha 1c2857b0646f, PR #25960)
TITLE: bugfix: --decrypted-draft-config-file not applied (#25960)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/speculative_hook.py (+9/-1)
LABELS: speculative-decoding, run-ci
BODY: ## Motivation ⏎  ⏎ Previously, --decrypted-draft-config-file was not applied when getting config.json for speculative draft model. ⏎  ⏎ server_args.speculative_algorithm = _resolve_speculative_algorithm_alias( ⏎                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/home/wzy/sgl-sglang/python/sglang/srt/arg_groups/speculative_hook.py", line 24, in _resolve_speculative_algorithm_alias ⏎     cfg = get_config( ⏎           ^ …[truncated]

### L1-7619e77b7b  (L1, 2026-05-29, sha 7619e77b7ba5, PR #26466)
TITLE: [NPU] chore: basic software upgrade (#26466)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/npu.Dockerfile (+21/-9); .github/workflows/nightly-test-npu.yml (+6/-6); .github/workflows/pr-test-npu.yml (+30/-19); .github/workflows/release-docker-npu-nightly.yml (+2/-2); .github/workflows/release-docker-npu.yml (+2/-2); scripts/ci/npu/npu_ci_install_dependency.sh (+14/-21)
LABELS: npu, run-ci
BODY: --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #26567132258](https://github.com/sgl-project/sglang/actions/runs/26567132258) ⏎ Latest PR Test (Extra): :x: [Run #26567132127](https://github.com/sgl-project/sglang/actions/runs/26567132127)

### L1-54b06f199c  (L1, 2026-05-29, sha 54b06f199c14, PR #24160)
TITLE: [lora] Share MoE LoRA Info (#24160)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/lora/lora_moe_runners.py (+2/-17); python/sglang/srt/lora/backend/ascend_backend.py (+2/-0); python/sglang/srt/lora/backend/base_backend.py (+186/-0); python/sglang/srt/lora/backend/chunked_backend.py (+2/-0); python/sglang/srt/lora/backend/torch_backend.py (+1/-0); python/sglang/srt/lora/backend/triton_backend.py (+1/-0); python/sglang/srt/lora/layers.py (+10/-29); python/sglang/srt/lora/utils.py (+19/-0); test/registered/lora/test_moe_lora_info.py (+87/-0)
LABELS: lora, npu, run-ci
BODY: ## Motivation ⏎  ⏎ In the current MoE LoRA design, a significant amount of information is computed ever layer despite information depending solely on just the batch info. ⏎  ⏎ ## Modifications ⏎  ⏎ This PR reworks the MoE LoRA design to enable batch-info-only dependent information to be computed once and then reused. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ This is a bitwise change. I've verified that the tensors are identical. ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎ In d …[truncated]

### L1-b1173c8c14  (L1, 2026-05-29, sha b1173c8c1469, PR #24582)
TITLE: [NPU] Enhance accuracy for model Step3_5 from 0 to 88% (#24582)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+135/-27); python/sglang/srt/layers/quantization/unquant.py (+20/-1)
LABELS: quant, npu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ refactor fia with sparse mode 4 supporting swa. ⏎ (align with cann 8.5, when qlen=1, pre&next tokens will be invalid for sparsemode4, fia squeezed mask to induct the true mask will fail as well ) ⏎ (cann 9 may support this) ⏎  ⏎ refactor activation function for moe part. ⏎ As follows. ⏎ this activation function is supported by gpu so the accuracy on gpu is normal,  (_swiglu_silu_clamp_mul in sglang/python/sglang …[truncated]

### L1-3ecf2c76ad  (L1, 2026-05-29, sha 3ecf2c76ad1b, PR #16775)
TITLE: [CPU] Add GPT-OSS model optimization for CPU (#16775)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-0); sgl-kernel/csrc/cpu/moe.cpp (+217/-37); sgl-kernel/csrc/cpu/moe.h (+106/-0); sgl-kernel/csrc/cpu/moe_fp8.cpp (+92/-55); python/sglang/srt/layers/amx_utils.py (+2/-0); python/sglang/srt/layers/attention/intel_amx_backend.py (+6/-0); python/sglang/srt/layers/quantization/__init__.py (+3/-1); python/sglang/srt/layers/quantization/awq/schemes/awq_cpu.py (+4/-0); python/sglang/srt/layers/quantization/fp8.py (+4/-0); python/sglang/srt/layers/quantization/gptq_cpu.py (+4/-0); (+25 more)
LABELS: quant, deepseek, sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎  ⏎ This PR (in collaboration with @jianan-gu) adds CPU-optimized support for GPT-OSS series, including the following changes: ⏎ - BF16 MoE kernel with MoE bias and swiglu activation support, ported from #12537  ⏎ - MXFP4 MoE kernel with MoE bias and swiglu activation support, depending on #14385  ⏎ - Attention with sink and sliding window support, ported from #12579, and depending on #8666  ⏎ - TP padding support, ported from #12539  …[truncated]

### L1-716e670d3d  (L1, 2026-05-30, sha 716e670d3dc0, PR #26696)
TITLE: [bugfix]: size CuteDSL MoE allgather buffers for the worst-case forward (#26696)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.runner.flashinfer_cutedsl
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+8/-18); python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-1); python/sglang/srt/server_args.py (+59/-31); test/registered/unit/server_args/test_server_args.py (+41/-0)
LABELS: documentation, quant, run-ci
BODY: ## Motivation ⏎  ⏎ `--moe-runner-backend flashinfer_cutedsl` on the standard (non-A2A) allgather path crashes at serving time: ⏎  ⏎ ``` ⏎ ValueError: num_tokens (4073) exceeds max_num_tokens (2048) ⏎ ``` ⏎  ⏎ Regression from #22669: `ensure_cutedsl_wrapper` sizes the `CuteDslMoEWrapper` CUDA-graph buffers from the **first forward's** token count (`max(num_tokens, 1)`) — a small startup capture batch. A larger real forward then trips FlashInfer's `run()` guard an …[truncated]

### L1-c93a559e5a  (L1, 2026-05-30, sha c93a559e5af1, PR #26710)
TITLE: Fix MoE LoRA wrapper exposing moe_runner_config (#26710)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/lora/layers.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Fix a MoE LoRA crash introduced after #25379. ⏎  ⏎ #25379  added direct access to `self.mlp.experts.moe_runner_config.inplace` in the DeepSeek/Kimi MoE forward path. When MoE LoRA is enabled, `self.mlp.experts` is wrapped by `FusedMoEWithLoRA`, which did not expose `moe_runner_config`, causing an `AttributeError` during CUDA graph capture. ⏎ <img width="1763" height="1163" alt="image" src="https://github.com/user-attachments/assets/ …[truncated]

### L1-0d9a2a9de3  (L1, 2026-05-30, sha 0d9a2a9de378, PR #26489)
TITLE: [MoE Refactor] Migrate SM90 Cutlass W4A16 to MoeRunner (#26489)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.runner.framework, L1.runner.flashinfer_mxfp4
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_mxfp4.py (+174/-0); python/sglang/srt/layers/moe/moe_runner/runner.py (+2/-0); python/sglang/srt/layers/quantization/mxfp4.py (+32/-65); python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py (+28/-45); test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py (+45/-6)
LABELS: run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ #24816 landed the SM90 cutlass MXFP4 path as a private `_apply_sm90_cutlass` helper that bypassed the unified `MoeRunner`, with `create_moe_runner` falling through to `pass` for `flashinfer_mxfp4`. #25525 then introduced the `register_fused_func((a2a, runner_backend))` pool and migrated `flashinfer_cutedsl` through it, leaving `flashinfer_mxfp4` as the next backend to clean up. This PR completes that migration so that GPT-OSS and …[truncated]

### L1-b421e60eed  (L1, 2026-05-30, sha b421e60eeddc, PR #26389)
TITLE: 【NPU】【bugfix】fix server error when mtp unquant (#26389)
SOURCES: path_core, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+2/-6); python/sglang/srt/models/deepseek_nextn.py (+69/-51); python/sglang/srt/models/qwen3_5_mtp.py (+43/-24); python/sglang/srt/models/qwen3_next_mtp.py (+33/-17); test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep.py (+1/-0)
LABELS: deepseek, npu, run-ci
DEEP_STUDY: deep-study correctness case sglang:b421e60eed: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=#22822
BODY: ## Motivation ⏎  ⏎ On NPU , the main model and draft model sometimes adapt different quantization schemes.With the ongoing refactoring of DeepEP, the environment variables for MTP quantization are currently affecting the main model incorrectly. ⏎  ⏎ ## Modifications ⏎  ⏎ python/sglang/srt/layers/moe/token_dispatcher/deepep.py ⏎ python/sglang/srt/models/deepseek_nextn.py ⏎ python/sglang/srt/models/qwen3_5_mtp.py ⏎ python/sglang/srt/models/qwen3_next_mtp.py …[truncated]

### L1-376635c1e3  (L1, 2026-05-30, sha 376635c1e3aa, PR #26123)
TITLE: Fix routed-experts device buffer overflow under DP attention (#26123)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/state_capturer/routed_experts.py (+6/-4)
LABELS: run-ci
BODY: The routed experts device buffer in RoutedExpertsCapturer can overflow when DP attention is enabled and max_running_requests is larger than chunked_prefill_size * dp_size. The buffer's first dim is currently max(chunked_prefill_size * dp_size, max_running_requests). On non-DeepEP DP-attention paths _get_local_slice goes through get_dp_local_slice_cpu, which returns local_start_pos = dp_rank * cuda_graph_batch under a cuda graph. On ranks with dp_ …[truncated]

### L1-a779791b3f  (L1, 2026-05-31, sha a779791b3f81, PR #26862)
TITLE: Add random-ids dataset, round-robin expert simulation, and kill_process_tree logging (#26862)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+51/-7); python/sglang/srt/environ.py (+1/-0); python/sglang/srt/utils/common.py (+5/-0); python/sglang/test/bench_one_batch_server_internal.py (+2/-2)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ - Add `random-ids` as a supported dataset option in `bench_one_batch_server_internal` for benchmarking with pre-generated token IDs. ⏎ - Add `SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS` env var for deterministic expert assignment in MoE layers, useful for reproducible benchmarking with dummy/random weights. Mutually exclusive with `SGLANG_SIMULATE_UNIFORM_EXPERTS`. ⏎ - Guard the existing uniform expert simulation against `k=0` edge case. ⏎ - Add l …[truncated]

### L1-524ba10eda  (L1, 2026-06-01, sha 524ba10eda1b, PR #24692)
TITLE: feat: SM120 (Blackwell Desktop) support for DeepSeek-V4 inference (#24692)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py (+454/-0); python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py (+72/-2); docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx (+47/-1); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+41/-18); python/sglang/srt/layers/attention/dsv4/indexer.py (+73/-1); python/sglang/srt/layers/attention/flash_mla_sm120.py (+252/-0); python/sglang/srt/layers/attention/flash_mla_sm120_triton.py (+370/-0); python/sglang/srt/layers/deep_gemm_wrapper/configurer.py (+3/-0); python/sglang/srt/server_args.py (+14/-0); test/registered/kernels/test_sm120_flash_mla.py (+465/-0); (+1 more)
LABELS: documentation, deepseek, run-ci, jit-kernel, run-ci-extra
BODY: ## Summary ⏎  ⏎ Adds full **SM120** (RTX PRO 6000 / RTX 5090 / DGX Spark, compute 12.0) support for DeepSeek-V4/V3 on SGLang. SM120 desktop Blackwell GPUs lack TMEM, tcgen05, and DeepGEMM support — this PR provides Triton-based fallback kernels for all critical paths and enables CUDA graph capture. ⏎  ⏎ ### Key changes ⏎  ⏎ **New kernels (7 files):** ⏎ - `mxfp4_moe_sm120_triton.py` — Triton fused MXFP4 dequant + GEMM for MoE experts (4.1x vs PyTorch per-GEMM) ⏎  …[truncated]

### L1-5700790c05  (L1, 2026-06-01, sha 5700790c0593, PR #24947)
TITLE: DeepSeek V4: Support context parallelism with fused MoE (non-DeepEP)  (#24947)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json (+164/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json (+164/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json (+164/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json (+164/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=384,N=384,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=384,N=384,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json (+164/-0); (+7 more)
LABELS: deepseek, run-ci, run-ci-extra
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎  ⏎   Currently, Context Parallelism (CP) in DeepSeek-V4 only works with DeepEP as the MoE all-to-all backend. Fused MoE, Marlin MoE, and other MoE backends are not supported under CP, which is a significant limitation for H20-3e             ⏎   deployments: ⏎  ⏎   - TP8 Fused MoE outperforms DeepEP on H20-3e by a considerable margin.                                                                                                       …[truncated]

### L1-1d7e2f6fb8  (L1, 2026-06-01, sha 1d7e2f6fb859, PR #22972)
TITLE: [NPU] fix normal DeepEP mode num_tokens_per_rdma_rank error caused by none (#22972)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/eplb/expert_distribution.py (+5/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ If `expert_distribution_recorder_mode = "per_token"`, attributeError is reported in the `on_deepep_dispatch_normal` method of the expert distribution recorder when the NPU is deployed on a single server. ⏎ <img width="1082" height="121" alt="image" src="https://github.com/user-attachments/assets/c8c45531-3535-4579-af7c-244856adf610" /> ⏎  ⏎ ## Modifications ⏎  ⏎ A check for the `None` value of `num_tokens_per_rdma_rank` is added to en …[truncated]

### L1-ff642ed936  (L1, 2026-06-01, sha ff642ed93697, PR #26303)
TITLE: [MoE] Extend kimi_k2_moe_fused_gate to support 256 experts (MiMo V2 Flash) (#26303)
SOURCES: path_core, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu (+190/-145); sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py (+13/-9)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ `kimi_k2_moe_fused_gate` is currently hard-coded for 384-expert routing (Kimi K2). MiMo V2 Flash uses the same DeepSeek-style `noaux_tc` routing with `num_expert_group=1` but has 256 experts. This PR extends the fast-path kernel to cover both 256 and 384 expert configurations from a single code path, so MiMo V2 Flash can take the same optimized fused gate as Kimi K2 instead of falling back to the slower generic biased grouped top …[truncated]

### L1-951fa05a09  (L1, 2026-06-01, sha 951fa05a09c7, PR #26473)
TITLE: [MoE] Support BF16 standard A2A with DeepGEMM runner (#26473)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin
ARTIFACT_HINTS: L1.runner.deep_gemm, L1.ep.layer, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/ep_moe/kernels.py (+36/-23); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+6/-0); python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py (+1/-1)
LABELS: dependencies, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ The main motivation of this PR is to fix the DeepGEMM MoE runner path when `--moe-runner-backend deep_gemm` is used with the standard MoE A2A backend. ⏎  ⏎ The primary issue is that `post_reorder_triton_kernel` did not handle expert 0 when combining routed outputs. As a result, routed outputs from expert 0 were skipped in the DeepGEMM runner combine step, which affected model accuracy. ⏎  ⏎ This PR also fixes BF16 model support for the sam …[truncated]

### L1-4226a6f13a  (L1, 2026-06-01, sha 4226a6f13aa6, PR #26884)
TITLE: [AMD] Fix GPT-OSS MXFP4 accuracy on ROCm AITER path (#26884)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.runner.aiter, L1.upstream.aiter_moe
FILES: python/sglang/srt/layers/moe/moe_runner/aiter.py (+12/-1); python/sglang/srt/environ.py (+5/-0); python/sglang/srt/layers/quantization/mxfp4.py (+51/-17); python/sglang/srt/server_args.py (+7/-0); test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py (+12/-2)
LABELS: amd, run-ci
DEEP_STUDY: deep-study correctness case sglang:4226a6f13a: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: CO-OWNER: @bingxche  ⏎  ⏎ ## Motivation ⏎  ⏎ On ROCm AITER, the GPT-OSS MXFP4 fused-MoE path was producing silently wrong outputs. With these server flags on MI355X (gfx950): ⏎  ⏎ ```bash ⏎ SGLANG_USE_AITER_MOE_GU_ITLV=False python3 -m sglang.launch_server \ ⏎   --model-path /path/to/gpt-oss-120b/ \ ⏎   --tp 1 --trust-remote-code \ ⏎   --mem-fraction-static 0.85 \ ⏎   --prefill-attention-backend aiter --decode-attention-backend aiter \ ⏎   --page-size 64 --d …[truncated]

### L1-b603f08c0c  (L1, 2026-06-02, sha b603f08c0cdf, PR #26643)
TITLE: [DP] Fix FlashInfer dispatcher workspace sizing and set_dp_buffer_len (#26643)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py (+12/-2); python/sglang/srt/model_executor/cuda_graph_runner.py (+12/-29); python/sglang/srt/model_executor/model_runner.py (+11/-30); python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py (+2/-0); python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+15/-32); python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+10/-18); python/sglang/srt/speculative/frozen_kv_mtp_cuda_graph_runner.py (+15/-32); python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py (+13/-18)
LABELS: run-ci
BODY: ## Summary ⏎ - Fix FlashInfer dispatcher workspace sizing to reflect max(chunked_prefill_size, max running req) ⏎ - set_dp_buffer_len sets the global_num_tokens_cpu based on global_dp_buffer_len ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #26807060375](https://github.com/sgl-project/sglang/actions/runs/26807060375) ⏎ Latest PR Test (Extra): :x: [Run #26807060145](https://github.com/sgl-project/sglang/actions/runs/26807060145)

### L1-68caf49154  (L1, 2026-06-02, sha 68caf49154ed, PR #26993)
TITLE: Update sgl-deep-gemm to 0.1.2 (#26993)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎ - Update the Python dependency pin for `sgl-deep-gemm` from 0.1.0 to 0.1.2. ⏎ - Update the Docker build ARG used for the `sgl-deep-gemm` wheel URL to 0.1.2. ⏎  ⏎ ## Test Plan ⏎ - `git diff --check` ⏎ - `python3 - <<'PY' ...` pyproject TOML parse confirming `sgl-deep-gemm==0.1.2` ⏎ - Commit hooks: TOML check, whitespace checks, codespell, and related pre-commit checks passed ⏎ - `gh release view v0.1.2 --repo sgl-project/whl --json tagName,assets` co …[truncated]

### L1-5ae8d286d2  (L1, 2026-06-02, sha 5ae8d286d2b8, PR #26502)
TITLE: perf(gemma4): single-launch fused router (topk + softmax + scale) (#26502)
SOURCES: subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/gemma4_fused_ops.py (+114/-0); python/sglang/srt/models/gemma4_causal.py (+9/-0); test/registered/kernels/test_gemma4_fused_routing.py (+106/-0)
LABELS: high priority, run-ci
DEEP_STUDY: deep-study performance PR ()
BODY: ## Motivation ⏎  ⏎ `Gemma4MoE.routing_function` emits four per-layer GPU kernels for every MoE ⏎ forward pass (one decode step touches ~30 routed-MoE layers): ⏎  ⏎ | Kernel | Decode share (Gemma-4-26B-A4B-IT + MTP, B200) | ⏎ |---|---| ⏎ | `at::native::sbtopk::gatherTopK<bf16,uint,2,false>` (torch.topk) | 2.4% | ⏎ | `at::native::bitonicSortKVInPlace<2,-1,16,16,bf16,...>` (torch.topk tie-break) | 2.0% | ⏎ | `at::native::index_elementwise_kernel<bf16>` (`per …[truncated]

### L1-b8d7351a74  (L1, 2026-06-02, sha b8d7351a74c3, PR #25655)
TITLE: Feat/add w4a16 moe support to nemotron (#25655)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.marlin
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py (+41/-14); python/sglang/srt/layers/moe/moe_runner/marlin.py (+14/-1); docs_new/docs/advanced_features/quantization.mdx (+15/-5); docs_new/docs/advanced_features/server_arguments.mdx (+3/-3); python/sglang/jit_kernel/csrc/gemm/marlin/marlin_template.h (+12/-18); python/sglang/jit_kernel/tests/test_gptq_marlin.py (+91/-2); python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py (+273/-1); python/sglang/jit_kernel/utils.py (+34/-0); python/sglang/srt/arg_groups/nemotron_h_hook.py (+14/-1); python/sglang/srt/layers/quantization/fp4_utils.py (+12/-1); (+9 more)
LABELS: documentation, high priority, quant, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ Serve NVFP4 modelopt checkpoints (modelopt_fp4) on Ampere/Hopper by routing them through Marlin W4A16 when native FP4. This is to support nemotron models ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ * Add Modelopt nvfp4 marlin path ⏎ * add non gated moe marlin support ⏎ * handle nvfp4 scaling format in kernel ⏎  ⏎  ⏎  ⏎ * implemented tests to make sure Linear/MoE layers are close to unquantized  ⏎ * ran nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 on A100 a …[truncated]

### L1-ac16dbf412  (L1, 2026-06-03, sha ac16dbf41250, PR #27049)
TITLE: docs: add DeepSeek-V4 EPLB Waterfill tips (#27049)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+41/-0)
LABELS: documentation
BODY: ## Summary ⏎ - add DeepSeek-V4 configuration tips for online EPLB with DeepEP Waterfill ⏎ - document the reproduction-style flow using a recorded EPLB expert-distribution .pt file plus --enable-deepep-waterfill ⏎ - note DeepEP mode and MegaMoE compatibility constraints ⏎  ⏎ ## Checks ⏎ - git diff --check ⏎ - pre-commit hooks from git commit ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #26857986793](https://github.com/sgl-project/sglang/ac …[truncated]

### L1-f790674ad8  (L1, 2026-06-03, sha f790674ad8a5, PR #26839)
TITLE: fix(moe): avoid unpacking None from masked deep_gemm without overlap when sbo enabled (#26839)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.runner.deep_gemm
FILES: python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+3/-1)
LABELS: run-ci
BODY: # Summary ⏎ Under Single Batch Overlap (SBO), DeepGemmRunnerCore._run_masked_gemm can crash during CUDA graph capture with: ⏎  ⏎ TypeError: cannot unpack non-iterable NoneType object ⏎   File ".../moe_runner/deep_gemm.py", line ..., in _run_masked_gemm ⏎     block_m, threshold = deep_gemm_return_value ⏎  ⏎ # Root cause ⏎ deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked only returns (block_m, threshold) when it is run with down-gemm overlap (i.e. overlap …[truncated]

### L1-7f706f4cfb  (L1, 2026-06-03, sha 7f706f4cfba0, PR #26854)
TITLE: [Deps] Bump FI to 0.6.12 and cutedsl to 4.5.2 (#26854)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+3/-3); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1)
LABELS: dependencies, run-ci, run-ci-extra
BODY: ## Summary ⏎  ⏎ Bump FI to 0.6.12 and cutedsl to 4.5.2 ⏎  ⏎ ## Test Plan ⏎  ⏎ CI ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #26841042733](https://github.com/sgl-project/sglang/actions/runs/26841042733) ⏎ Latest PR Test (Extra): :x: [Run #26841044109](https://github.com/sgl-project/sglang/actions/runs/26841044109)

### L1-293816ab14  (L1, 2026-06-03, sha 293816ab14af, PR #18005)
TITLE: [AMD][MXFP4] Online MXFP4 quantization 1/N - dense and MOE models w. original BF16 weight (#18005)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/__init__.py (+1/-0); python/sglang/srt/layers/quantization/quark/quark.py (+103/-3); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py (+68/-6); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py (+77/-10); python/sglang/srt/model_executor/model_runner.py (+14/-0); python/sglang/srt/server_args.py (+11/-6); docs_new/docs/advanced_features/quantization.mdx (+13/-1); python/sglang/srt/configs/model_config.py (+2/-0); python/sglang/srt/constants.py (+2/-0); python/sglang/srt/model_loader/loader.py (+24/-0); (+2 more)
LABELS: documentation, quant, amd, run-ci
BODY: As per title. This PR implements online MXFP4 quantization (targeting Instinct MI350X/MI355X). ⏎  ⏎ **For both dense and MOE layers, online MXFP4 quantization is run in weight loaders by wrapping the original `weight_loader` with the online quantization step, before eventually calling it.** ⏎  ⏎ Example: ⏎  ⏎ ```bash ⏎ sglang serve --model-path Qwen/Qwen3.5-397B-A17B \ ⏎ 	--tensor-parallel-size 4 --trust-remote-code \ ⏎ 	--mem-fraction-static 0.85 --atten …[truncated]

### L1-aa510bda45  (L1, 2026-06-03, sha aa510bda4505, PR #26349)
TITLE: Support specific pass of bias_grouped_topk for xpu (#26349)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+32/-0); test/registered/xpu/test_topk.py (+101/-0)
LABELS: intel, xpu, run-ci, run-ci-extra
BODY: ## Modifications ⏎ Use topk_sigmoid directly when num_groups is 1, just like cuda did. Also added unitest. ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #26803749635](https://github.com/sgl-project/sglang/actions/runs/26803749635) ⏎ Latest PR Test (Extra): :no_entry_sign: [Run #26858636266](https://github.com/sgl-project/sglang/actions/runs/26858636266)

### L1-c9ca56da8c  (L1, 2026-06-03, sha c9ca56da8c5e, PR #27091)
TITLE: Unify full→SWA index translation in init_forward_metadata; drop pool caches (#27091)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/environ.py (+0/-1); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+78/-4); python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py (+78/-4); python/sglang/srt/layers/attention/dsa_backend.py (+2/-1); python/sglang/srt/layers/attention/dsv4/compressor_v2.py (+4/-1); python/sglang/srt/layers/attention/dsv4/indexer.py (+2/-1); python/sglang/srt/layers/attention/flashattention_backend.py (+8/-7); python/sglang/srt/layers/attention/triton_backend.py (+0/-3); python/sglang/srt/layers/attention/trtllm_mha_backend.py (+6/-3); python/sglang/srt/mem_cache/base_swa_memory_pool.py (+0/-3); (+19 more)
LABELS: deepseek, hicache, blackwell, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ The full→SWA index translation (`translate_loc_from_full_to_swa`) was triggered in two inconsistent styles: ⏎  ⏎ - **Write path (DSV4 KV store):** translated `out_cache_loc` lazily at the *first SWA layer* via `DeepSeekV4TokenToKVPool.get_cached_swa_loc` (env-gated by `SGLANG_OPT_CACHE_SWA_TRANSLATION`), with the result cached across layers. ⏎ - **Read path (attention):** `SWAKVPool` translated on demand and memoized with a `(data_ptr, n …[truncated]

### L1-e4191708c9  (L1, 2026-06-03, sha e4191708c9d6, PR #26845)
TITLE: [Qwen3.5][AMD] Fix shared-expert ×ep_size over-count under allreduce-EP (#26845)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen2_moe.py (+14/-1)
LABELS: amd, run-ci
BODY: ## Motivation ⏎  ⏎ On the AMD/AITER path, fused shared expert was enabled for Qwen3.5 by #20736. When combined with allreduce-EP (`--expert-parallel-size > 1` and the default `--moe-a2a-backend none`), the shared-expert contribution is over-counted by exactly `ep_size`, which destroys generation quality. ⏎  ⏎ Measured on **MI355X** with `Qwen3.5-397B-A17B-FP8`, sglang `b6f71d585`, container `rocm/sgl-dev:v0.5.12.post1-rocm720-mi35x-20260524`, GSM8K 5-sho …[truncated]

### L1-8e836e7dc9  (L1, 2026-06-04, sha 8e836e7dc9b8, PR #26746)
TITLE: Support optional kwargs in AITER fused_moe runner (#26746)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L1.runner.aiter, L1.upstream.aiter_moe
FILES: python/sglang/srt/layers/moe/moe_runner/aiter.py (+34/-1); test/registered/unit/layers/moe/test_aiter_runner.py (+119/-0)
LABELS: documentation, amd, run-ci
BODY: PR description: ⏎  ⏎   ## Motivation ⏎  ⏎   Some AITER `fused_moe` integrations need to pass backend-specific options directly to the installed AITER kernel while keeping the generic SGLang MoE runner interface unchanged. This PR adds a small extension point for those optional kwargs and enables `no_combine` forwarding when the installed AITER package supports it. ⏎  ⏎   ## Modifications ⏎  ⏎   - Add optional `fused_moe_kwargs` to `AiterMoeQuantInfo` and …[truncated]

### L1-4cfebbb95f  (L1, 2026-06-04, sha 4cfebbb95f18, PR #25239)
TITLE: [FlashInfer v0.6.12] Support FlashInfer 4over6 NVFP4 (#25239)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+8/-1); docs_new/docs/references/environment_variables.mdx (+10/-0); python/sglang/srt/environ.py (+3/-0)
LABELS: documentation, quant, run-ci
BODY: ## Motivation ⏎ @humansand ⏎  ⏎ Parents: ⏎  ⏎ - https://github.com/flashinfer-ai/flashinfer/pull/3264 ⏎ - https://github.com/sgl-project/sglang/pull/22918 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-p …[truncated]

### L1-e76d36214b  (L1, 2026-06-04, sha e76d36214b0f, PR #26496)
TITLE: Changes for SM120 perf and usability for NVFP4 (#26496)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition.json (+298/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition_down.json (+335/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py (+14/-3); docs_new/docs/advanced_features/quantization.mdx (+2/-2); docs_new/docs/advanced_features/server_arguments.mdx (+1/-1); python/sglang/srt/layers/deep_gemm_wrapper/configurer.py (+2/-2); python/sglang/srt/layers/quantization/awq/awq.py (+4/-0); python/sglang/srt/layers/quantization/fp4_utils.py (+1/-7); python/sglang/srt/model_executor/model_runner.py (+19/-6); python/sglang/srt/server_args.py (+12/-1)
LABELS: documentation, quant, run-ci
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/issues/19637 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - Fix AWQ skip layers https://github.com/sgl-project/sglang/pull/20439 ⏎ - Switch the default back to CUTLASS, since: the PDL bug has been fixed, and the performance is better due to SwapAB + tileN in https://github.com/flashinfer-ai/flashinfer/pull/3152 for NVFP4/MXFP8 regular and grouped GEMM (blockwise FP8 still may need some updates) ⏎ - Disable DeepG …[truncated]

### L1-0aa72a9e76  (L1, 2026-06-04, sha 0aa72a9e7677, PR #27193)
TITLE: Replace skip_attn_backend_init with a batch-carried attention plan marker (+ staleness re-plan) (#27193)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/batch_overlap/two_batch_overlap.py (+6/-0); python/sglang/srt/hardware_backend/mlx/tp_worker.py (+1/-1); python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py (+1/-2); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+8/-6); python/sglang/srt/managers/tp_worker.py (+4/-6); python/sglang/srt/model_executor/cpu_graph_runner.py (+0/-1); python/sglang/srt/model_executor/cuda_graph_runner.py (+2/-2); python/sglang/srt/model_executor/forward_batch_info.py (+93/-0); python/sglang/srt/model_executor/model_runner.py (+6/-13); python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+3/-0); (+10 more)
LABELS: speculative-decoding, blackwell, npu, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ `skip_attn_backend_init` is a control-coupling boolean threaded through ~6 layers (`TpModelWorker.forward_batch_generation` → `ModelRunner.forward` → `_forward_raw` → `forward_decode` / `forward_extend` / graph-runner `replay`, plus the NPU/CPU/MLX mirrors). It means *"the attention metadata for this batch was already planned elsewhere — don't plan again"*, but the protocol has three structural problems: ⏎  ⏎ 1. **Unverifiable caller a …[truncated]

### L1-c6c1f1a29a  (L1, 2026-06-04, sha c6c1f1a29a2f, PR #27308)
TITLE: docs: sync legacy docs/-only updates into docs_new (Mintlify) (#27308)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/docs.json (+1/-0); docs_new/docs/advanced_features/quantization.mdx (+37/-0); docs_new/docs/advanced_features/server_arguments.mdx (+51/-1); docs_new/docs/basic_usage/sampling_params.mdx (+1/-1); docs_new/docs/developer_guide/benchmark_and_profiling.mdx (+3/-0); docs_new/docs/developer_guide/contribution_guide.mdx (+6/-5); docs_new/docs/hardware-platforms/ascend-npus/ascend_contribution_guide.mdx (+4/-4); docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_quantization.mdx (+129/-0); docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_qwen3_examples.mdx (+80/-0); docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx (+1/-1); (+8 more)
LABELS: documentation, quant, npu
BODY: ## Motivation ⏎  ⏎ Since the Mintlify documentation site (`docs_new/`) was introduced in #23001, a number of PRs landed documentation changes in the **legacy `docs/` tree only**, so `docs_new/` has drifted out of sync. This PR ports those `docs/`-only changes into the corresponding `docs_new/` pages. ⏎  ⏎ **Each commit maps to one source PR** (ordered by merge date) for easy review and traceability. ⏎  ⏎ ## Ported PRs (one commit each) ⏎  ⏎ | Source PR | `docs_n …[truncated]

### L1-4167f211b1  (L1, 2026-06-04, sha 4167f211b113, PR #27027)
TITLE: Solving the problem of test case failures caused by timeouts (#27027)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_low_latency_deepseek_v3_2_w8a8.py (+2/-0)
LABELS: deepseek, npu, run-ci
BODY: … to resolve test case failure issues ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ test_npu_deepep_auto_deepseek_v3_2_w8a8.py ⏎ This test case failed in the nightly pipeline, and the error message indicated that loading the weights took too long. The actual value of watchdog-timeout is 300. ⏎  ⏎ ## Modifications ⏎  ⏎ In the service startup parameters of this test case, watchdog-timeout is set to 900. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Check …[truncated]

### L1-8933ec8772  (L1, 2026-06-04, sha 8933ec877235, PR #26775)
TITLE: fix test cases failed on 5/30 in nightly pipeline (#26775)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/test/ascend/test_ascend_utils.py (+2/-1); test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_qwen3_next.py (+1/-0); test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_low_latency_qwen3_next.py (+1/-0); test/registered/ascend/basic_function/parameter/test_npu_no_overlap_scheduler.py (+1/-2)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ On May 29, the test cases that were skipped in the nightly build were executed again. ⏎ In the nightly pipeline automatically triggered on May 30, five test cases of those test cases failed to be executed due to the implementation problems of the test cases. ⏎  ⏎ ## Modifications ⏎  ⏎ test_npu_no_overlap_scheduler.py	 ⏎ This test case imports a method using an incorrect path.Therefore, I modified the import code block of this test case …[truncated]

### L1-aed0808e18  (L1, 2026-06-05, sha aed0808e1853, PR #27335)
TITLE: 6-5 nightly failed test case fix (#27335)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_deepseek_v3_2_w8a8.py (+2/-0); test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_low_latency_deepseek_v3_2_w8a8.py (+1/-0); test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_low_latency_qwen3_480b.py (+1/-0)
LABELS: deepseek, npu, run-ci
BODY: ## Motivation ⏎  ⏎ test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_deepseek_v3_2_w8a8.py ⏎ This test case failed due to a timeout when loading weights, so watchdog-timeout is set to 900 to avoid this issue. ⏎  ⏎ test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_low_latency_deepseek_v3_2_w8a8.py ⏎ test/registered/ascend/basic_function/parallel_strategy/expert_parall …[truncated]

### L1-c9f582a272  (L1, 2026-06-05, sha c9f582a272dc, PR #27329)
TITLE: [LoRA] Experimental fast LoRA path with `experimental_sgl_trtllm` MoE backend for FP8 and NVFP4 models (#27329)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.triton.moe_align, L1.routing.topk_py, L1.runner.flashinfer_trtllm
FILES: python/sglang/jit_kernel/csrc/trtllm_lora_temp/kimi_k2_moe_fused_gate.cuh (+453/-0); python/sglang/jit_kernel/csrc/trtllm_lora_temp/moe_lora_merged_align_kernel.cu (+589/-0); python/sglang/jit_kernel/csrc/trtllm_lora_temp/topk_softmax_pack.cuh (+412/-0); python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu (+1137/-0); python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/trtllm_fused_moe_kernel_launcher.cu (+3920/-0); python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/trtllm_fused_moe_runner.cu (+1098/-0); python/sglang/jit_kernel/trtllm_lora_temp/data/include/flashinfer/trtllm/fused_moe/DevKernel.h (+472/-0); python/sglang/jit_kernel/trtllm_lora_temp/data/include/flashinfer/trtllm/fused_moe/runner.h (+585/-0); python/sglang/jit_kernel/trtllm_lora_temp/kimi_k2_moe_fused_gate.py (+70/-0); python/sglang/jit_kernel/trtllm_lora_temp/moe_lora_merged_align.py (+144/-0); (+42 more)
LABELS: documentation, high priority, quant, lora, deepseek, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: > **⚠️ Experimental — for early adopters.** This is an opt-in fast LoRA path gated behind a single master switch `SGLANG_EXPERIMENTAL_LORA_OPTI` (default **off** ⇒ upstream behavior is byte-identical) and selected with `--moe-runner-backend experimental_sgl_trtllm`. The new logic is intentionally isolated in `*_temp` packages and is **actively being refactored** toward an upstream-clean form — env flags, the backend name, and file layout may stil …[truncated]

### L1-29591594f5  (L1, 2026-06-05, sha 29591594f599, PR #27150)
TITLE: Support Waterfill with dynamic EPLB (#27150)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+14/-4); python/sglang/srt/models/deepseek_v2.py (+7/-1); test/registered/unit/eplb/test_deepep_waterfill_eplb.py (+138/-0)
LABELS: deepseek, run-ci
BODY: ## Summary ⏎ - Fix fused shared expert MoE weight views used by online EPLB updates. ⏎ - Keep EPLB expert distribution stats on logical routed expert ids when shared experts are fused. ⏎ - Add focused EPLB unit coverage for fused shared experts, Waterfill, and non-DeepEP recorder ids. ⏎  ⏎ ## Tests ⏎ - `PYTHONPATH=python python -m pytest -q test/registered/unit/eplb` ⏎ - SGLang MMLU64, dynamic EPLB rb32: shared fusion red8 `Score: 0.891`, no NCCL timeout ⏎ - SGL …[truncated]

### L1-38ae22e08c  (L1, 2026-06-05, sha 38ae22e08c73, PR #26733)
TITLE: Nemotron perf changes (#26733)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/activation.py (+41/-0); python/sglang/jit_kernel/benchmark/bench_activation.py (+16/-0); python/sglang/jit_kernel/csrc/elementwise/activation.cuh (+72/-0); python/sglang/jit_kernel/tests/test_activation.py (+47/-1); python/sglang/srt/arg_groups/nemotron_h_hook.py (+27/-24); python/sglang/srt/layers/activation.py (+6/-2); python/sglang/srt/layers/attention/fla/layernorm_gated.py (+10/-0); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+4/-2); python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py (+12/-0); python/sglang/srt/layers/attention/mamba/mamba.py (+9/-3); (+5 more)
LABELS: quant, run-ci, jit-kernel, run-ci-extra
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## TODOs: ⏎ This PR aims to address the very obvious issues. There are still the following items: ⏎ 1. Ensure the right kernels ⏎ - When using the default command, it uses Flashinfer FA2 attention for full attention, instead we should use mamba `extra_buffer` and `trtllm_mha` for that command (always). For example, we'd use: ⏎ ``` ⏎ python3 -m sglang.launch_server \ ⏎   --model-path nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \ ⏎   --trust-remote-code …[truncated]

### L1-9da88e32e0  (L1, 2026-06-05, sha 9da88e32e096, PR #27401)
TITLE: [Cohere2Moe] Enable flashinfer_trtllm NVFP4 fused-MoE via SigmoidRenorm routing (#27401)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+7/-1); python/sglang/srt/models/cohere2_moe.py (+12/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## [Cohere2Moe] Enable flashinfer_trtllm NVFP4 fused-MoE via SigmoidRenorm routing ⏎  ⏎ ### Problem ⏎ `Cohere2MoeSparseMoeBlock` built its `FusedMoE` without `routing_method_type`. The `flashinfer_trtllm` NVFP4 ⏎ fused-MoE runner requires it — `compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` does ⏎ `assert layer.routing_method_type is not None`. So for NVFP4 Command-A-Plus checkpoints, selecting ⏎ `--moe-runner-backend flashinfer_trtllm` cra …[truncated]

### L1-9a48bf75f5  (L1, 2026-06-06, sha 9a48bf75f537, PR #27427)
TITLE: Add GB300 base C CI suite (#27427)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/_pr-test-stage.yml (+2/-0); .github/workflows/pr-test.yml (+15/-3); .github/workflows/rerun-test.yml (+7/-0); scripts/ci/runner_configs.yml (+3/-0); scripts/ci/utils/slash_command_handler.py (+19/-9); test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py (+1/-1); test/registered/disaggregation/test_disaggregation_aarch64.py (+4/-3); test/registered/utils/test_numa_utils.py (+37/-31); test/run_suite.py (+1/-1)
LABELS: deepseek
BODY: ## Summary ⏎ - Add `base-c-test-4-gpu-gb300` to PR Base C and move the former 4-GPU GB200 registrations to GB300. ⏎ - Add a `4-gpu-gb300` runner config that uses the DeePEP installer with `GRACE_BLACKWELL=1`. ⏎ - Thread the `grace_blackwell` runner config field through PR test stages and `/rerun-test` dispatch. ⏎  ⏎ ## Test Plan ⏎ - `python3 scripts/ci/runner_configs.py 4-gpu-gb300 && python3 scripts/ci/runner_configs.py deepep-4-gpu-b200` ⏎ - `python3 scripts …[truncated]

### L1-f57f8a8afd  (L1, 2026-06-06, sha f57f8a8afd84, PR #26588)
TITLE: Optimize Gemma4 H200 MoE and extend attention (#26588)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=128,N=704,device_name=NVIDIA_H200.json (+114/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=128,N=704,device_name=NVIDIA_H200_down.json (+114/-0); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+3/-1); python/sglang/srt/layers/gemma4_fused_ops.py (+106/-34)
LABELS: run-ci, run-ci-extra
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎ - Add H200 Triton fused MoE configs for Gemma4 `E=128,N=704` normal and down projections. ⏎ - Tune Hopper extend-attention block sizes for `Lq=129..256` to reduce TTFT/TPOT on Gemma4 prefill-heavy serving. ⏎ - Add a small-batch (M ≤ 256) per-head Triton kernel for Gemma4 QKV RMSNorm to lower MTP draft latency. ⏎ - ~~Add a Gemma4-specific Triton routing kernel that fuses top-k selection, top-k softmax, and per-expert scale into `gemma4_topk_ …[truncated]

### L1-4c8a022f38  (L1, 2026-06-06, sha 4c8a022f38e3, PR #27191)
TITLE: Fix DeepSeek V4 DP reduce scatter when use attention DP + MoE TP (#27191)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v4.py (+10/-2)
LABELS: deepseek, run-ci
BODY: ## Summary ⏎  ⏎ - Use `reduce_scatterv` for DeepSeek V4's TP-MoE gather return path when DP reduce-scatter is enabled. ⏎ - Keep the existing `dp_scatter` path for configurations that do not use DP reduce-scatter. ⏎  ⏎ ## Root Cause ⏎  ⏎ DeepSeek V4 has a hand-written `_use_tp_moe_gather` path. In DP attention with TP MoE, the MLP output can be TP-partial. When `should_use_dp_reduce_scatterv()` is true, scattering that partial tensor directly leaves the local h …[truncated]

### L1-10d33bd77e  (L1, 2026-06-07, sha 10d33bd77ea8, PR #22299)
TITLE: [AMD] Enable Piecewise CUDA Graph for AMD GPUs (#22299)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py, L1.routing.hash_topk
FILES: python/sglang/srt/layers/moe/hash_topk.py (+11/-2); python/sglang/srt/layers/moe/topk.py (+22/-0); python/sglang/srt/compilation/cuda_piecewise_backend.py (+21/-4); python/sglang/srt/compilation/piecewise_context_manager.py (+6/-0); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py (+134/-5); python/sglang/srt/layers/radix_attention.py (+29/-0); python/sglang/srt/model_executor/model_runner.py (+37/-6); python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py (+66/-12); test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py (+7/-2); test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py (+2/-1)
LABELS: amd, deepseek, run-ci, piecewise-cuda-graph, bypass-fastfail
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Enable Piecewise CUDA Graph (PCG) on AMD/ROCm with the aiter backend, while keeping the existing CUDA path unchanged. The PR also fixes a Dynamo recompilation fallback crash when batches exceed `piecewise_cuda_graph_max_tokens`. ⏎  ⏎ ### Tested configurations ⏎  ⏎ - **DeepSeek-R1-MXFP4-Preview** — TP=4, EP=4, EAGLE speculative decoding, FP8 KV cache, aiter backend ⏎ - **DeepSeek-R1-MXFP4-Preview** — TP=8, registered AMD PCG test ⏎ - ** …[truncated]

### L1-1c73ff8ad3  (L1, 2026-06-07, sha 1c73ff8ad3fd, PR #27063)
TITLE: [AMD] Optimize gpt-oss-120B performance (#27063)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/environ.py (+20/-0); python/sglang/srt/layers/attention/aiter_backend.py (+69/-13); python/sglang/srt/layers/attention/aiter_utils.py (+309/-0); python/sglang/srt/layers/attention/utils.py (+1185/-0); python/sglang/srt/layers/layernorm.py (+35/-0); python/sglang/srt/layers/quantization/mxfp4.py (+10/-3); python/sglang/srt/layers/rotary_embedding/base.py (+12/-7); python/sglang/srt/mem_cache/memory_pool.py (+118/-18); python/sglang/srt/models/gpt_oss.py (+74/-5); python/sglang/srt/models/utils.py (+25/-7); (+1 more)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Integrate Atom's `pa_decode_gluon` paged-attention decode kernel into ⏎ SGLang's AITER backend to accelerate gpt-oss-120b decode on ROCm ⏎ (MI300 / MI355).  The kernel requires a SHUFFLE 5D vectorized KV ⏎ cache layout that aiter's `mha_batch_prefill_func` also consumes ⏎ natively, so we add the layout as a new opt-in pool format ⏎ (`SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d`) and route the AITER backend ⏎ through it on both prefill a …[truncated]

### L1-61e4132bc2  (L1, 2026-06-08, sha 61e4132bc27f, PR #27537)
TITLE: [MUSA] bump torchada version to 0.1.59 and workaround PCG limitation. (#27537)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: 3rdparty/amd/wheel/sglang/pyproject.toml (+1/-1); python/pyproject_other.toml (+1/-1); sgl-kernel/pyproject_musa.toml (+1/-1)
LABELS: dependencies, sgl-kernel, mthreads
BODY: ## Motivation ⏎  ⏎ Upgrade `torchada` to `0.1.59` to pick up the MUSA CUDA Graph executable rotation fix for the long-standing PCG issue. ⏎  ⏎ SGLang’s piecewise CUDA Graph path can instantiate a large number of MUSA graph executables for deep models. On MUSA, the driver has a per-process limit of roughly 2048 live `musaGraphExec_t` objects, which can cause graph capture failures when PCG is enabled. ⏎  ⏎ `torchada 0.1.59` includes a transparent LRU ro …[truncated]

### L1-b047bb3e92  (L1, 2026-06-08, sha b047bb3e9271, PR #27387)
TITLE: build(sgl-kernel): support configurable mirrors for restricted networks (#27387)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+27/-18); sgl-kernel/cmake/flashmla.cmake (+11/-3); sgl-kernel/Dockerfile (+25/-5); sgl-kernel/Makefile (+7/-0); sgl-kernel/build.sh (+12/-1)
LABELS: sgl-kernel, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Make `sgl-kernel` Docker builds work in restricted-network or mirrored ⏎ environments (internal artifactory, region-blocked GitHub, etc.) by making ⏎ every external source a configurable mirror. Public defaults are preserved — ⏎ out-of-the-box behavior is unchanged. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Some opt-in env vars, each falling back to its public upstream when unset. ⏎ All threaded through `build.sh` → `docker buildx --build-arg`  …[truncated]

### L1-1f5dc2cdca  (L1, 2026-06-08, sha 1f5dc2cdca0a, PR #26786)
TITLE: [GPTQ] Refactor CPU quantization schemes (#26786)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/cpu/quantization/awq_kernels.py (+99/-0); python/sglang/srt/hardware_backend/cpu/quantization/gptq_kernels.py (+99/-0); python/sglang/srt/layers/quantization/__init__.py (+2/-4); python/sglang/srt/layers/quantization/auto_round.py (+6/-4); python/sglang/srt/layers/quantization/awq/schemes/awq_cpu.py (+4/-85); python/sglang/srt/layers/quantization/awq/schemes/awq_marlin.py (+8/-4); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py (+3/-1); python/sglang/srt/layers/quantization/gptq/__init__.py (+8/-9); python/sglang/srt/layers/quantization/gptq/gptq.py (+39/-7); python/sglang/srt/layers/quantization/gptq/schemes/__init__.py (+3/-0); (+3 more)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ This PR refactors GPTQ quantization scheme organization to align with the AWQ scheme/kernel split. ⏎  ⏎ - Moves GPTQ CPU AMX kernels into `hardware_backend/cpu/quantization`. ⏎ - Moves GPTQ CPU schemes under `layers/quantization/gptq/schemes` and removes the old top-level `gptq_cpu.py` module. ⏎ - Keeps platform-specific kernels behind scheme initialization instead of importing GPU/NPU/CPU kernels at package import time. ⏎ - Applies the same CP …[truncated]

### L1-dc24a26821  (L1, 2026-06-08, sha dc24a2682190, PR #27528)
TITLE: Fix GPT-OSS MXFP4 hidden size reshape on SM10X (#27528)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/gpt_oss.py (+2/-1)
BODY: ## Summary ⏎  ⏎ Fix a regression from #27063 where GPT-OSS used the backend-internal expert hidden size for the final MoE reshape. ⏎  ⏎ On SM10X with FlashInfer MXFP4, `self.experts.hidden_size` is padded from `2880` to `3072`, while `config.hidden_size` remains `2880`, causing warmup to fail: ⏎  ⏎ ```text ⏎ RuntimeError: shape '[4096, 3072]' is invalid for input of size 11796480 ⏎ ``` ⏎  ⏎ I also checked H200, where serving succeeds because it uses `moe_r …[truncated]

### L1-ca66e6fb5e  (L1, 2026-06-08, sha ca66e6fb5e5d, PR #25195)
TITLE: [BCG] Support breakable CUDA graph for DeepSeek V4 DP attention (#25195)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh (+3/-1); python/sglang/srt/batch_overlap/two_batch_overlap.py (+9/-0); python/sglang/srt/layers/attention/base_attn_backend.py (+30/-0); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+251/-26); python/sglang/srt/layers/attention/dsv4/compressor.py (+8/-3); python/sglang/srt/layers/attention/dsv4/indexer.py (+39/-15); python/sglang/srt/managers/schedule_batch.py (+2/-0); python/sglang/srt/managers/scheduler_components/dp_attn.py (+13/-2); python/sglang/srt/model_executor/breakable_cuda_graph_runner.py (+66/-9); python/sglang/srt/model_executor/forward_batch_info.py (+2/-0); (+3 more)
LABELS: deepseek, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ DeepSeek V4 with DP attention currently cannot use breakable piecewise CUDA graph for mixed/extend batches. This leaves the prefill/mixed path on eager kernel launches and can cause large host-side gaps under high concurrency. This PR adds the DeepSeek V4 and DP-attention plumbing required to make breakable piecewise CUDA graph capture/replay work for DSV4 mixed chunk workloads. ⏎  ⏎ Co-authored by: @Oasis-Git  ⏎  ⏎ ## Modifications …[truncated]

### L1-d7c8b9ab9f  (L1, 2026-06-09, sha d7c8b9ab9fcd, PR #27533)
TITLE: [Intel GPU] Enable fused_experts in fp8.py for quantized models on XPU (#27533)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8.py (+38/-0)
LABELS: intel, xpu, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]
