### L3-63c0889416  (L3, 2026-01-31, sha 63c0889416f0, PR #33462)
TITLE: [Misc] Fix flashinfer related tests (#33462)
SOURCES: path_core, path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+1/-1); tests/kernels/moe/test_moe.py (+1/-1); tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py (+1/-1); tests/kernels/quantization/test_fp8_quant.py (+2/-2); vllm/model_executor/layers/quantization/utils/nvfp4_utils.py (+4/-3)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ Fix CUDA errors from FlashInfer TRTLLM kernels on Blackwell. ⏎ ``` ⏎ CUDA Error: CUDA_ERROR_INVALID_VALUE /home/ubuntu/vllm/.venv/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/trtllm/fmha/fmhaKernels.cuh 301 ⏎ ``` ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-527bcd14d4  (L3, 2026-01-31, sha 527bcd14d465, PR #32492)
TITLE: [ROCM] Enable aiter attn backend for qwen3-next model (#32492)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+1/-1); docs/design/attention_backends.md (+1/-1)
LABELS: documentation, rocm, ready, ci/build, v1, qwen
BODY: ## Purpose ⏎ Currently aiter attn doesn't work for qwen3-next because the model uses non-standard block size (544). This issue can be fixed by #24486 which decouples kernel block size from page block size. The trunk fails because aiter attn returns multiple of 16 for get_supported_kernel_block_sizes, but in reality it doesn't support large block size like 544. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎  CUDA_VISIBLE_DEVICES=2 VLLM_ROCM_MOE_PADDING=0  VLLM_ROCM_USE_AITER=1  …[truncated]

### L3-cd86fff38f  (L3, 2026-02-01, sha cd86fff38fee, PR #33077)
TITLE: [BUGFIX] Fix hipErrorIllegalState in Qwen3-Omni during startup profiling allow inference Omni on ROCM (#33077)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_omni_moe_thinker.py (+31/-7)
LABELS: bug, rocm, ready, qwen
BODY: This PR fixes a critical crash occurring on AMD (ROCm) hardware when initializing the Qwen3-Omni model (specifically the Qwen3Omni_VisionTransformer). ⏎  ⏎ During the memory profiling phase (profile_run), the operation torch.repeat_interleave on the GPU triggers a hipErrorIllegalState, causing the worker to crash. This appears to be a stability issue with dynamic tensor shape operations on the current ROCm driver/backend when processing visual grid d …[truncated]

### L3-8869cd8ec1  (L3, 2026-02-01, sha 8869cd8ec1b2, PR #33510)
TITLE: Add MoE config for Super B200 TP2 (#33510)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/configs/E=512,N=1344,device_name=NVIDIA_B200.json (+131/-0)
LABELS: ready
BODY: When locally running Nemotron Super on B200 the following warning appears: ⏎  ⏎ ```bash ⏎ Using default MoE config. Performance might be sub-optimal! ⏎ ``` ⏎  ⏎ I used the `benchmark_moe.py` to create a JSON file for this use-case: ⏎ ```bash ⏎ python benchmarks/kernels/benchmark_moe.py \ ⏎   --model $MODEL_PATH \ ⏎   --trust-remote-code \ ⏎   --tp-size 2 \ ⏎   --tune \ ⏎   --batch-size 1 2 4 8 16 24 32 48 64 96 128 256 512 768 1024 1536 \ ⏎   --save-dir /.../vllm/model_exec …[truncated]

### L3-46b4a02794  (L3, 2026-02-01, sha 46b4a0279422, PR #33501)
TITLE: Fix DeepSeek V2 RoPE initialization error (#33501)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py (+0/-2)
LABELS: ready, deepseek
DEEP_STUDY: deep-study correctness case vllm:46b4a02794: class=hardware_compiler_specific; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎ Fix DeepSeek V2 RoPE initialization error. ⏎  ⏎ Before, one would get the following error output on TPUs ⏎ ``` ⏎ (vllm_env2) nhatt@t1v-n-b6d844bb-w-0:~$ cd tpu-inference ⏎ (vllm_env2) nhatt@t1v-n-b6d844bb-w-0:~/tpu-inference$ JAX_TRACEBACK_FILTERING=off VLLM_MLA_DISABLE=1 VLLM_LOGGING_LEVEL=DEBUG MODEL_IMPL_TYPE=vllm vllm serve gaunernst/DeepSeek-V2-Lite-Chat-FP8   --max-model-len=4096   --gpu-memory-utilization=0.98  --download-dir /dev/shm ⏎ DEB …[truncated]

### L3-0aca8b8c62  (L3, 2026-02-02, sha 0aca8b8c628e, PR #32790)
TITLE: [MoE] Enable Shared/Routed Overlap For Latent MoE (Nemotron-H) (#32790)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_shared_fused_moe_routed_transform.py (+162/-0); vllm/model_executor/layers/fused_moe/layer.py (+89/-16); vllm/model_executor/layers/fused_moe/shared_fused_moe.py (+22/-0); vllm/model_executor/models/nemotron_h.py (+30/-42)
LABELS: ready
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Enable parallel CUDA stream execution between shared and routed experts for latent MoE architectures (e.g., Nemotron-H). ⏎  ⏎ ## Problem ⏎  ⏎ Latent MoE compresses the input before routing to experts: ⏎ - **Routed experts** receive compressed input: `[S, moe_latent_size]` (e.g., 1024) ⏎ - **Shared experts** need original input: `[S, hidden_size]` (e.g., 4096) ⏎  ⏎ Previously, this dimension mismatch prevented the streaming optimization where shared and routed ex …[truncated]

### L3-089cd4f002  (L3, 2026-02-02, sha 089cd4f00248, PR #32224)
TITLE: fix cutlass_3x_gemm_fp8_blockwise on sm103a (#32224)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/cutlass_extensions/common.hpp (+88/-6); csrc/quantization/w8a8/cutlass/c3x/scaled_mm.cuh (+4/-4); csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm100_fp8_dispatch.cuh (+1/-1); csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm100_fp8_dispatch.cuh (+2/-2); csrc/quantization/w8a8/cutlass/scaled_mm_c2x.cuh (+0/-35); csrc/quantization/w8a8/cutlass/scaled_mm_c2x_sm89_fp8_dispatch.cuh (+17/-17); csrc/quantization/w8a8/cutlass/scaled_mm_c2x_sm89_int8_dispatch.cuh (+17/-17)
LABELS: ready, nvidia
BODY: ## Purpose ⏎  ⏎ When compiling with sm103a, the output of `cutlass_3x_gemm_fp8_blockwise` is garbage value. ⏎  ⏎ Updated the helpers in `csrc/cutlass_extensions/common.hpp` to include sm103a. ⏎  ⏎ Also added a runtime error message when the kernel is executed but not compiled. ⏎  ⏎ ## Test Plan ⏎  ⏎ `pytest tests/kernels/quantization/test_block_fp8.py::test_w8a8_block_fp8_cutlass_matmul` ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-5d1aef3004  (L3, 2026-02-02, sha 5d1aef3004f0, PR #33570)
TITLE: [UX] Format attention backend log line (#33570)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+2/-2)
LABELS: ready, nvidia
BODY: ## Purpose ⏎ Clean up attention backend log line to match MoE ⏎  ⏎ Main: ⏎ ``` ⏎ (EngineCore_DP0 pid=2126810) INFO 02-02 17:08:08 [cuda.py:364] Using FLASHMLA_SPARSE attention backend out of potential backends: ('FLASHMLA_SPARSE',) ⏎ (EngineCore_DP0 pid=2126810) INFO 02-02 17:08:08 [fp8.py:329] Using FLASHINFER_CUTLASS Fp8 MoE backend out of potential backends: ['AITER', 'FLASHINFER_TRTLLM', 'FLASHINFER_CUTLASS', 'DEEPGEMM', 'BATCHED_DEEPGEMM', 'TRITON', 'BA …[truncated]

### L3-e69c990c21  (L3, 2026-02-02, sha e69c990c216c, PR #30329)
TITLE: [Feature][CPU Backend]: Optimize ARM vectorization backend (#30329)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: csrc/cpu/mla_decode.cpp (+1/-3); csrc/cpu/cpu_attn_impl.hpp (+0/-11); csrc/cpu/cpu_types_arm.hpp (+551/-579); csrc/cpu/dnnl_kernels.cpp (+0/-2); csrc/cpu/utils.hpp (+0/-2)
LABELS: ready, cpu
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎ Implement vectorised backed based on `at::vec::Vectorized` for ARM, introduce highlevel FP32/FP16/BF16 vector wrappers and add new shared based class `VectorizedRegWrapper` capable of scaling with dtypes. ⏎  ⏎ - Implement ARM vectorisation based on `at::vec::Vectorized<T>` for `float`, `c10::Half` and `c10::BFloat16` with potential to extend for integer data types. ⏎  ⏎ - Introduce `NxVectorizedTVecReg<N, T>` and `VectorizedRegWrapper<DerivedC …[truncated]

### L3-9eb58f8cf1  (L3, 2026-02-02, sha 9eb58f8cf123, PR #32902)
TITLE: fix[ROCm]: Remove unconditional aiter import (#32902)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+3/-2); vllm/_aiter_ops.py (+12/-4); vllm/v1/spec_decode/eagle.py (+3/-1)
LABELS: rocm, speculative-decoding, ready, v1
BODY: ## Purpose ⏎ AITER modules were being imported unconditionally at module load time, triggering JIT compilation and warnings even when VLLM_ROCM_USE_AITER=0 (the default). ⏎  ⏎   [aiter] WARNING: NUMA balancing is enabled... ⏎   [aiter] import [module_aiter_enum] under .../aiter/jit/... ⏎  ⏎ This occurred because several code paths imported aiter without first checking the VLLM_ROCM_USE_AITER environment variable. ⏎  ⏎ ## Test Plan ⏎ Tested locally ⏎  ⏎ ## Test Result ⏎  …[truncated]

### L3-e10604480b  (L3, 2026-02-02, sha e10604480bb8, PR #33379)
TITLE: [XPU][1/N] Deprecate ipex and switch to vllm-xpu-kernels for xpu platform (#33379)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fa_utils, L3.dispatch.registry
FILES: vllm/v1/attention/backends/fa_utils.py (+3/-2); vllm/v1/attention/backends/registry.py (+0/-1); .buildkite/scripts/hardware_ci/run-xpu-test.sh (+1/-2); docker/Dockerfile.xpu (+10/-4); requirements/xpu.txt (+2/-2); vllm/_ipex_ops.py (+73/-360); vllm/config/model.py (+0/-1); vllm/model_executor/layers/activation.py (+29/-53); vllm/model_executor/layers/layernorm.py (+1/-18); vllm/model_executor/layers/linear.py (+0/-2); (+8 more)
LABELS: ready, ci/build, v1
BODY: ## Purpose ⏎ [1/N] https://github.com/vllm-project/vllm/issues/33214 ⏎ start from this PR, we will switch to vllm-xpu-kernels based kernel impl for Intel GPU. ⏎ this PR also upgrade dependency to oneapi 2025.3 and pytorch 2.10 for xpu platform. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-528e9b1490  (L3, 2026-02-02, sha 528e9b14900f, PR #33540)
TITLE: [Feature][Core] Support Fabric detection to adapt the MNNVL protocol for the GB series (#33540)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/cumem_allocator.cpp (+18/-1)
LABELS: ready
ISSUES: #33539 [Bug]: When using NIXL for PD separation on GB series cards, the MNNVL protocol cannot be used
BODY: ## Purpose ⏎  ⏎ Fix https://github.com/vllm-project/vllm/issues/33539 ⏎  ⏎ Thank you very much for the solution provided by @tvegas1  ⏎  ⏎ ## Test Plan ⏎  ⏎ Note that, currently, to use a custom allocator, sleep-mode must to be enabled. ⏎  ⏎ ``` ⏎ # Prefill - Tray 0 ⏎ export VLLM_NIXL_SIDE_CHANNEL_HOST=10.0.8.10 UCX_CUDA_IPC_ENABLE_MNNVL=y UCX_PROTO_INFO=y ⏎ vllm serve --gpu-memory-utilization=0.85 --tokenizer-mode=deepseek_v32 --no-enable-expert-parallel --model /home/ub …[truncated]

### L3-2267cb1cfd  (L3, 2026-02-03, sha 2267cb1cfd83, PR #23465)
TITLE: [Attention][FA3] Update FA3 to include new swizzle optimization (#23465)
SOURCES: path_core, subject_keyword, dependency_pin, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fork_build, L3.mla.flashattn
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/v1/attention/backends/flash_attn.py (+6/-1); vllm/v1/attention/backends/mla/flashattn_mla.py (+6/-1)
LABELS: documentation, ready, ci/build, unstale, v1
DEEP_STUDY: deep-study: this PR was reverted by PR 33841 (confirmed_revert, reason=hardware_specific_breakage) || deep-study: this PR was reverted by PR 34043 (reland, reason=other)
BODY: vLLM side of https://github.com/vllm-project/flash-attention/pull/82  ⏎  ⏎ `meta-llama/Meta-Llama-3-8B-Instruct`, 1xH100, 4k and 2k out ⏎ ``` ⏎ branch   rate     num_prompts  req/s    median TTFT (ms)   std TTFT     p99 TTFT     median TPOT (ms)   std TPOT     p99 TPOT     ⏎ ------   ----     -----------  -----    ----------------   --------     --------     ----------------   --------     --------     ⏎ MAIN     1.00     120          0.88     141.95         …[truncated]

### L3-2a99c5a6c8  (L3, 2026-02-03, sha 2a99c5a6c86d, PR #33613)
TITLE: [Bugfix] Disable TRTLLM FP8 MoE if router_logits_dtype==float32 and routing_method!=DeepSeekV3 (#33613)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py (+31/-4); vllm/model_executor/layers/fused_moe/oracle/fp8.py (+5/-5); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+3/-12); vllm/model_executor/layers/quantization/fp8.py (+3/-12); vllm/model_executor/models/minimax_m2.py (+1/-0)
LABELS: bug, ready, deepseek, nvidia
ISSUES: #33543 [Bug]: Some FP8 MoE models fail assertions on GB200
BODY: ## Purpose ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/33543 ⏎  ⏎ Also dedupe shared logic for routing_logits and routing_bias casting ⏎  ⏎ Opened a Flashinfer issue to track if the kernel ends up adding support for this https://github.com/flashinfer-ai/flashinfer/issues/2469 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ Now when running `MiniMaxAI/MiniMax-M2.1` on B200 FLASHINFER_TRTLLM MoE is not selected and accuracy is fine with DeepGEMM or Triton MoE selected …[truncated]

### L3-bd8da29a66  (L3, 2026-02-03, sha bd8da29a66ea, PR #33579)
TITLE: [Bugfix] Fix sparse MLA metadata building (#33579)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+22/-31)
LABELS: bug, ready
ISSUES: #33546 [Bug][DeepSeekV32]: AttributeError: 'FlashMLASparseMetadata' object has no attribute 'num_decodes'
BODY: ## Purpose ⏎ Fix https://github.com/vllm-project/vllm/issues/33546 ⏎ https://github.com/vllm-project/vllm/pull/33284 broke sparse MLA by moving logic from the backend to the layer without properly accounting for sparse backends. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ vllm serve deepseek-ai/DeepSeek-V3.2 -tp 8 -ep ⏎ ``` ⏎  ⏎ ## Test Result ⏎ Main: crashes during startup ⏎ ``` ⏎ AttributeError: 'FlashMLASparseMetadata' object has no attribute 'num_decodes' ⏎ ``` ⏎  ⏎ PR: ⏎ ``` ⏎ |Tasks|Version| …[truncated]

### L3-4dffc5e044  (L3, 2026-02-03, sha 4dffc5e04431, PR #32161)
TITLE: [CPU] Split attention dispatch by head_dim alignment (#32161)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: cmake/cpu_extension.cmake (+13/-0); csrc/cpu/cpu_attn.cpp (+21/-104); csrc/cpu/cpu_attn_amx.hpp (+1/-1); csrc/cpu/cpu_attn_neon.hpp (+1/-1); csrc/cpu/generate_cpu_attn_dispatch.py (+203/-0); tests/kernels/attention/test_cpu_attn.py (+2/-1)
LABELS: ready, ci/build, cpu
BODY: ## Purpose ⏎ Separate 32-aligned (AMX/NEON/VEC/VEC16) and 16-only (VEC16 only) head_dim dispatch paths to avoid redundant template instantiations and fix NEON and AMX compilation errors for head_dim 80/112 when static assertions are enabled ⏎  ⏎ Head dims divisible by 32 route through all ISA implementations, while head dims divisible by 16 but not 32 are restricted to VEC16 only, preventing unsupported ISA combinations from being instantiated during c …[truncated]

### L3-0e92298622  (L3, 2026-02-04, sha 0e922986222d, PR #33801)
TITLE: [Misc] Delay deprecation of CommonAttentionMetadata properties (#33801)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+2/-2)
LABELS: ready, v1
BODY: Deprecation is delayed due to snags with performance https://github.com/vllm-project/vllm/pull/33771 and CI instability slowing down https://github.com/vllm-project/vllm/pull/32073 ⏎  ⏎ Remove version for now until the timeline becomes more clear

### L3-4c8d1bf361  (L3, 2026-02-04, sha 4c8d1bf361c0, PR #33548)
TITLE: use ORJSONResponse when available to improve the efficiency of request process (#33548)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/entrypoints/pooling/embed/api_router.py (+18/-1)
LABELS: frontend, ready
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ use ORJSONResponse when available to improve the efficiency of handling embedding requests ⏎  ⏎ ## Test Plan ⏎  ⏎ start vllm with command below, and check benchmark performance for embeddings, running with 1 **H20** gpu.  ⏎ ```bash ⏎ python \ ⏎   -m vllm.entrypoints.openai.api_server \ ⏎   --model BAAI/bge-base-en-v1.5 \ ⏎   --port 12345 \ ⏎   --tensor-parallel-size 1 \ ⏎   --served-model-name auto \ ⏎   --max-num-batched-tokens 524288 \ ⏎   --max-num-seqs 1024 …[truncated]

### L3-824058076c  (L3, 2026-02-04, sha 824058076c56, PR #33291)
TITLE: [PERF] Change GDN Attention State Layout from [N, HV, K, V] to [N, HV, V, K] (#33291)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fla/ops/chunk.py (+4/-4); vllm/model_executor/layers/fla/ops/chunk_delta_h.py (+35/-35); vllm/model_executor/layers/fla/ops/chunk_o.py (+4/-4); vllm/model_executor/layers/fla/ops/fused_recurrent.py (+16/-16); vllm/model_executor/layers/fla/ops/kda.py (+1/-1); vllm/model_executor/layers/mamba/mamba_utils.py (+1/-1)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ This PR changes the recurrent state memory layout in GDN (Gated Delta Net) attention from `[N, HV, K, V]` to `[N, HV, V, K]` for improved memory access patterns and throughput. ⏎  ⏎ Behind speedup, also allows to use FI's GDN kernels ⏎  ⏎ ## Performance Results ⏎  ⏎ **Model**: `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` (TP=2) ⏎  ⏎ **Server**: ⏎ ```bash ⏎ VLLM_USE_FLASHINFER_MOE_FP4=1 vllm serve nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \ ⏎     -tp 2 --enabl …[truncated]

### L3-f67ee8b859  (L3, 2026-02-04, sha f67ee8b85921, PR #33782)
TITLE: [Perf] Optimize chat completion streaming performance (#33782)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/entrypoints/openai/chat_completion/serving.py (+17/-14)
LABELS: frontend, ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Optimize chat completion streaming performance ⏎  ⏎ Avoid repeatedly executing `is_reasoning_end(res.prompt_token_ids)`. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-45f8fd6f97  (L3, 2026-02-04, sha 45f8fd6f979d, PR #33688)
TITLE: [Feature] Enable `TRITON_ATTN` for Batch Invariance (#33688)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/v1/attention/ops/triton_unified_attention.py (+5/-1); docs/features/batch_invariance.md (+1/-0); tests/v1/determinism/utils.py (+1/-0); vllm/model_executor/layers/batch_invariant.py (+6/-3)
LABELS: documentation, ready, v1
BODY: ## Purpose ⏎ This PR adds `TRITON_ATTN` support for batch invariance. ⏎  ⏎ Related / parent issue: #27433 ⏎  ⏎ ## Test Plan ⏎ Run tests with and without the `or is_batch_invariant` check in the triton_unified_attention's `unified_attention` method. ⏎  ⏎ ## Test Result ⏎ Tests are run on a B200 (do not have access to a Hopper GPU to validate there 🙁) ⏎  ⏎ Without: ⏎  ⏎ ``` ⏎ CUDA_VISIBLE_DEVICES=2 VLLM_TEST_SEED=12345 pytest tests/v1/determinism/test_batch_invariance.py::tes …[truncated]

### L3-4d9513537d  (L3, 2026-02-04, sha 4d9513537d00, PR #33293)
TITLE: [CI][torch.compile] Reduce e2e fusion test time (#33293)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+16/-29); .buildkite/test-pipeline.yaml (+11/-70); .buildkite/test_areas/compile.yaml (+166/-26); .buildkite/test_areas/distributed.yaml (+2/-57); .buildkite/test_areas/pytorch.yaml (+2/-5); tests/compile/distributed/test_fusions_e2e.py (+0/-321); tests/compile/fusion_test_utils.py (+0/-208); tests/compile/fusions_e2e/__init__.py (+0/-0); tests/compile/fusions_e2e/common.py (+102/-0); tests/compile/fusions_e2e/conftest.py (+158/-0); (+7 more)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ Fusion E2E tests are out of control: they have poor coverage but also take a long time in CI.  ⏎  ⏎ This PR simultaneously improves coverage, splits up the tests, and cuts running times by reducing `n_hidden_layers` and using dummy weights. The old E2E tests are removed completely in favor of a new `fusions_e2e` directory. We add utilities to make it easier to add models and fusions in the future. ⏎  ⏎ In CI, the E2E fusion tests are now spli …[truncated]

### L3-e3bf79ffa0  (L3, 2026-02-04, sha e3bf79ffa080, PR #33841)
TITLE: Revert "[Attention][FA3] Update FA3 to include new swizzle optimization" (#33841)
SOURCES: path_core, subject_keyword, dependency_pin, corpus:confirmed-reverts
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fork_build, L3.mla.flashattn
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/v1/attention/backends/flash_attn.py (+1/-6); vllm/v1/attention/backends/mla/flashattn_mla.py (+1/-6)
LABELS: ready, ci/build, v1
ISSUES: #33802 [CI Failure]: Distributed 2xH100 tests
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 23465 reason=hardware_specific_breakage
BODY: Reverts vllm-project/vllm#23465 ⏎  ⏎ As described in #33802, #23465 broke the Distributed Tests 2 GPUs (H100). ⏎ - CI run for #23465: https://buildkite.com/vllm/ci/builds/49808/steps/canvas?sid=019c244e-f4b2-46e4-bdea-4e8c0c0d8bb9&tab=output ⏎ - CI run for #31034 (previous commit): https://buildkite.com/vllm/ci/builds/49807/steps/canvas?sid=019c244e-cc62-4d6c-a7d6-9c2922ed648f&tab=output ⏎  ⏎ Note that since the tests have been slightly refactored so the fai …[truncated]

### L3-6e98f6d8b6  (L3, 2026-02-04, sha 6e98f6d8b649, PR #33732)
TITLE: Implement zero-copy GQA for multimodal and CPU (#33732)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/mm_encoder_attention.py (+4/-12); vllm/v1/attention/backends/cpu_attn.py (+1/-4); vllm/v1/attention/ops/vit_attn_wrappers.py (+12/-4); vllm/model_executor/models/molmo2.py (+1/-12)
LABELS: ready, v1, cpu
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎   - zero-copy GQA for sdpa for multimodal and cpu backend. ⏎   - existing verison copies heads to match num_heads ⏎   - This patch removes the copies and use 'enable_gqa' option ⏎    ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-bbe0574d8e  (L3, 2026-02-05, sha bbe0574d8e51, PR #33192)
TITLE: [Bugfix] Disable TRTLLM attention when KV transfer is enabled (#33192)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+17/-0)
LABELS: bug, ready, v1, nvidia
DEEP_STUDY: deep-study: this PR was reverted by PR 34832 (confirmed_revert, reason=correctness_or_accuracy)
BODY: ## Summary ⏎  ⏎ - **Problem**: On Blackwell GPUs, vLLM crashes with `AssertionError: is_strictly_contiguous(kv_cache_permute)` when using NixlConnector for P/D disaggregation. ⏎ - **Cause**: FlashInfer auto-enables TRTLLM attention on Blackwell, which requires contiguous KV cache tensors. NixlConnector creates non-contiguous KV cache views. ⏎ - **Fix**: Auto-disable TRTLLM attention when `kv_transfer_config` is set. ⏎  ⏎ ## Purpose ⏎  ⏎ On Blackwell GPUs, FlashI …[truncated]

### L3-a7be77beef  (L3, 2026-02-05, sha a7be77beef5f, PR #33637)
TITLE: [Bugfix] fix DeepSeek R1 with CUTLASS MLA Broken on B200 (#33637)
SOURCES: path_core, subject_keyword, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+1/-4)
LABELS: bug, ready, v1, deepseek, nvidia
ISSUES: #33627 [Bug]: DeepSeek R1 with CUTLASS MLA Broken on B200
DEEP_STUDY: deep-study correctness case vllm:a7be77beef: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎ FIX https://github.com/vllm-project/vllm/issues/33627 ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-59a5cb387a  (L3, 2026-02-05, sha 59a5cb387ae4, PR #31171)
TITLE: [perf] Integrate flashinfer concat_mla_k (#31171)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+17/-3); vllm/utils/flashinfer.py (+47/-0)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Integrate Flashinfer concat_mla_k kernel ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎ ``` ⏎ local-completions (base_url=http://0.0.0.0:8087/v1/completions,pretrained=/ds-models/DeepSeek-R1-0528-FP4-v2,model=/ds-models/DeepSeek-R1-0528-FP4-v2,add_bos_token=True,tokenized_requests=False,tokenizer_backend=None,num_concurrent=1024,timeout=60000,max_retries=5,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: auto ⏎ |Task …[truncated]

### L3-d2f4a71cd5  (L3, 2026-02-05, sha d2f4a71cd544, PR #33858)
TITLE: [Bugfix] Kimi-K2 grouped_topk usage for Flashinfer monolithic kernels. (#33858)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/deepseek_v2.py (+3/-11)
LABELS: bug, ready, deepseek
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:207c3a0c20 (fix PR 33919)
BODY: ## Purpose ⏎ This PR fixes a bug introduced in  PR #33174 that sets the values for n_group and topk_group to None when they are (1, 1) respectively. This while it fixes Kimi-K2 may introduce an error with Mistral. @dbari Please confirm if this fix is good or if the values need to be passed differently ⏎  ⏎ The marlin path works because it doesn't have monolithic kernel for routing + MOE unlike the INT4 TRTLLM MOE Kernels.  ⏎  ⏎ ## Test Plan ⏎ GSM8k before an …[truncated]

### L3-af3162d3aa  (L3, 2026-02-05, sha af3162d3aaa5, PR #32887)
TITLE: [Spec Decode] Unified Parallel Drafting (#32887)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+6/-1); vllm/v1/attention/backends/flashinfer.py (+30/-6); vllm/v1/attention/backends/utils.py (+0/-32); examples/offline_inference/spec_decode.py (+3/-0); tests/v1/e2e/test_spec_decode.py (+32/-63); tests/v1/spec_decode/test_eagle.py (+408/-6); tests/v1/spec_decode/test_mtp.py (+1/-1); vllm/config/speculative.py (+7/-0); vllm/config/vllm.py (+18/-10); vllm/model_executor/models/llama_eagle3.py (+39/-3); (+4 more)
LABELS: documentation, performance, speculative-decoding, ready, v1, llama, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ This PR implements a single input preparation kernel for draft model support, and parallel drafting both with and without hidden states from the target model. As such we now have support for AMD's PARD, which proposes parallel drafting for fine-tuned external draft models, and AWS' P-EAGLE which implements parallel prediction for EAGLE3. Both of these are benchmarked as part of this PR effort. ⏎  ⏎ ## Testing ⏎  ⏎ E2E tests for parallel draft …[truncated]

### L3-4145e50d85  (L3, 2026-02-05, sha 4145e50d854e, PR #33932)
TITLE: [Bugfix] Fix DSV3.2 NVFP4 (#33932)
SOURCES: path_core, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+4/-2)
LABELS: bug, ready
ISSUES: #33859 [Bug]: DeepSeek V3.2-NVFP4 with flashinfer moe reports `q must have dtype torch::kBFloat16`
DEEP_STUDY: deep-study correctness case vllm:4145e50d85: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎ Fix https://github.com/vllm-project/vllm/issues/33859 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-79028d4388  (L3, 2026-02-05, sha 79028d438859, PR #33568)
TITLE: [Perf] Disable clean_logits in deepgemm fp8_mqa_logits kernel (#33568)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_deepgemm_attention.py (+13/-7); tests/kernels/test_top_k_per_row.py (+38/-18); vllm/model_executor/layers/sparse_attn_indexer.py (+2/-0); vllm/utils/deep_gemm.py (+8/-2)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ I noticed in DeepSeek V3.2 sparse_attn_indexer, `deep_gemm::smxx_clean_logits` kernel was launched following `deep_gemm::sm90_fp8_mqa_logits`, because `clean_logits` is set to true in https://github.com/vllm-project/vllm/blob/v0.16.0rc0/vllm/utils/deep_gemm.py#L331. ⏎  ⏎ The purpose of `deep_gemm::smxx_clean_logits` kernel is to fill the padding value of MQA logits with -inf, see https://github.com/deepseek-ai/DeepGEMM/blob/v2.1.1/README. …[truncated]

### L3-d5c4800112  (L3, 2026-02-05, sha d5c4800112c1, PR #33527)
TITLE: Adds padding and perf improvements to wvSplitK_fp8 (#33527)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/rocm/skinny_gemms.cu (+126/-174); tests/kernels/quantization/test_rocm_skinny_gemms.py (+40/-52); vllm/model_executor/layers/quantization/kernels/scaled_mm/rocm.py (+3/-3)
LABELS: rocm, ready
BODY: Adds activation padding support to wvSplitKQ. Additionally improves bias and dpp reduce perf. Expands test scenarios. ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-20d7454c9b  (L3, 2026-02-06, sha 20d7454c9bb0, PR #33511)
TITLE: fix(ROCm): Make flash_attn import optional in MLA attention (#33511)
SOURCES: path_core, subject_keyword, symbol_pickaxe, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+19/-3)
LABELS: rocm, ready
DEEP_STUDY: deep-study correctness case vllm:20d7454c9b: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎  ⏎ On ROCm, models that don't use MLA were failing to load because attention/__init__.py eagerly imported MLAAttention, which in turn tried to import flash_attn unconditionally. ⏎  ⏎ - Makes flash_attn import optional in mla_attention.py with try/except ⏎ - Adds a clear error message when MLA is used without flash_attn ⏎  ⏎ This allows non-MLA models to work on ROCm without needing flash_attn installed. ⏎  ⏎ ## Test Plan ⏎ Tested locally. ⏎  ⏎ ## Test Resul …[truncated]

### L3-ac04dd374f  (L3, 2026-02-06, sha ac04dd374f99, PR #33788)
TITLE: [CPU] Add BF16 Kernel type for s390x (#33788)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: csrc/cpu/mla_decode.cpp (+9/-0)
LABELS: ready, cpu
BODY: ## Purpose ⏎ Add BF16 Kernel types for s390x in `mla_decode.cpp` ⏎ ## Test Plan ⏎ 1. Build the image ⏎ 2. Run inference ⏎  ⏎ ## Test Result ⏎ ``` ⏎ [root@b314lp81 vllm]# docker run --rm  -p 8000:8000   local:test ibm-granite/granite-4.0-micro --port=8000 ⏎ INFO 02-04 11:03:05 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available. ⏎ (APIServer pid=1) INFO 02-04 11:03:10 [utils.py:314]  ⏎ (APIServer pid=1) INFO 02- …[truncated]

### L3-207c3a0c20  (L3, 2026-02-06, sha 207c3a0c20c0, PR #33919)
TITLE: Fix RoutingMethodType logic (#33919)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); docker/Dockerfile.nightly_torch (+2/-2); docker/versions.json (+1/-1); requirements/cuda.txt (+1/-1); vllm/model_executor/layers/fused_moe/config.py (+17/-0); vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py (+9/-4); vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py (+8/-5); vllm/model_executor/layers/fused_moe/router/fused_topk_router.py (+8/-5); vllm/model_executor/layers/fused_moe/router/router_factory.py (+14/-1)
LABELS: ready, ci/build, nvidia, ready-run-all-tests
DEEP_STUDY: deep-study correctness case vllm:207c3a0c20: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=#33858
BODY: ## Purpose ⏎  ⏎ This PR contains two fixes for #33792: ⏎ - Fix the selection logic for `RoutingMethodType` in `fused_topk_bias_router.py` and `fused_topk_router.py` ⏎ - When `use_grouped_topk=True`, the `GroupedTopKRouter` did not find any valid routing methods and there is only one group, fall back to the non-grouped routers ⏎  ⏎ The latter point covers Mistral Large 3, which has `n_group=1` and `topk_group=1` but uses `Renormalize` instead of `DeepSeekV3`  …[truncated]

### L3-1363e3d6d5  (L3, 2026-02-06, sha 1363e3d6d565, PR #32263)
TITLE: [cpu][performance] CPU Paged Attention NEON BFMMLA BF16 Implementation (#32263)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+3/-1); csrc/cpu/cpu_attn_impl.hpp (+2/-1); csrc/cpu/cpu_attn_neon.hpp (+17/-2); csrc/cpu/cpu_attn_neon_bfmmla.hpp (+682/-0)
LABELS: ready, v1, cpu
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ CPU Paged Attention NEON BFMMLA BF16 Implementation ⏎  ⏎ ## Test Results ⏎  ⏎ Using: https://github.com/vllm-project/vllm/pull/31720 Benchmark Suite, ⏎ Against Current NEON Implementation: ⏎ **Prefill: 2.32x ⏎ Decode: 2.07x** ⏎  ⏎ cc. @aditew01 @fadara01

### L3-350ca72c04  (L3, 2026-02-06, sha 350ca72c0423, PR #33749)
TITLE: [ROCm][AITER] Fix AITER import regression for explicit backend selection (#33749)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fa_utils, L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/fa_utils.py (+45/-1); vllm/v1/attention/backends/rocm_aiter_fa.py (+9/-8); tests/kernels/attention/test_aiter_flash_attn.py (+87/-20); vllm/_aiter_ops.py (+114/-34); vllm/v1/spec_decode/eagle.py (+7/-3)
LABELS: rocm, speculative-decoding, ready, v1
BODY: A regression was [introduced](https://github.com/vllm-project/vllm/pull/32902) that broke explicit AITER backend selection on ROCm when `VLLM_ROCM_USE_AITER=0` (or unset). Users could not explicitly select the AITER backend via `attention_config={"backend": "ROCM_AITER_FA"}` even though the backend was available. ⏎  ⏎ **Error observed:** ⏎ ```python ⏎ AttributeError: 'builtin_function_or_method' object has no attribute 'flash_attn_varlen_func' ⏎ ``` ⏎  ⏎ This  …[truncated]

### L3-bc32444b23  (L3, 2026-02-06, sha bc32444b238d, PR #33517)
TITLE: [Kernel] Add enable_sm120_or_later for SM121 (DGX Spark) CUTLASS support (#33517)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/cutlass_extensions/common.hpp (+11/-0); csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh (+2/-1)
LABELS: ready, nvidia
ISSUES: #28589 [Bug]: V1 Engine fails on Blackwell GB10 (SM 12.1): "sink setting not supported" by all compatible attention backends
BODY: ## Summary ⏎  ⏎ Add `enable_sm120_or_later` kernel wrapper to support SM121 (DGX Spark GB10) in addition to SM120 (RTX 5090) for Blackwell CUTLASS kernels. ⏎  ⏎ ## Problem ⏎  ⏎ DGX Spark GB10 (SM121) cannot use CUTLASS kernels because `enable_sm120_only` uses exact architecture match. ⏎  ⏎ ## Root Cause ⏎  ⏎ The existing `enable_sm120_only` wrapper uses: ⏎ ```cpp ⏎ #if defined __CUDA_ARCH__ && __CUDA_ARCH__ == 1200 ⏎ ``` ⏎  ⏎ This excludes SM121 (arch 1210) which has identica …[truncated]

### L3-15a0b9e570  (L3, 2026-02-06, sha 15a0b9e570dc, PR #33978)
TITLE: Fix spelling errors (#33978)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_v1_adapter
FILES: vllm/v1/attention/ops/flashmla.py (+4/-4); tests/kernels/moe/test_cutedsl_moe.py (+1/-1); tests/kernels/moe/test_moe_align_block_size.py (+1/-1); tests/reasoning/test_hunyuan_reasoning_parser.py (+3/-3); vllm/model_executor/layers/fused_moe/oracle/fp8.py (+1/-1)
LABELS: ready, v1
BODY: ## Summary ⏎ - Fix `Intialize` → `Initialize` in test files ⏎ - Fix `is_availble` → `is_available` in flashmla.py   ⏎ - Fix `AVAILBLE_BACKENDS` → `AVAILABLE_BACKENDS` in fp8.py ⏎ - Fix `NO_REASONING_QUICK_THROUGHT` → `NO_REASONING_QUICK_THOUGHT` in test file ⏎  ⏎ ## Test plan ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-906077181b  (L3, 2026-02-07, sha 906077181b21, PR #33967)
TITLE: [Bugfix] Fix QK Norm+RoPE fusion pattern matching on B200+FP8 (#33967)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/fusions_e2e/models.py (+1/-3); tests/compile/passes/test_qk_norm_rope_fusion.py (+19/-3); tests/compile/passes/test_split_coalescing.py (+62/-0); vllm/compilation/passes/pass_manager.py (+2/-0); vllm/compilation/passes/utility/split_coalescing.py (+70/-0)
LABELS: bug, ready
ISSUES: #33295 [Bug]: QKNorm+RoPE fusion broken for qwen3-fp8 on B200
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Purpose ⏎  ⏎ Fixes #33295 ⏎  ⏎ On B200 with FP8-quantized models (e.g., `Qwen/Qwen3-30B-A3B-FP8`), PyTorch Inductor's CSE fails to merge identical `split_with_sizes` calls. The QK Norm+RoPE pattern matcher expects one split node with 3 `getitem` users (q/k/v), but the graph contains 3 separate split nodes with 1 user each, so fusion fails silently. ⏎  ⏎ This PR adds `coalesce_equivalent_splits()` to merge duplicate splits before matching. It groups splits …[truncated]

### L3-dd6a6e1190  (L3, 2026-02-07, sha dd6a6e119062, PR #34006)
TITLE: [Kernel] Add KernelConfig flag to enable/disable FlashInfer autotune (#34006)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:production-kernel-provenance, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/__init__.py (+3/-0); vllm/config/kernel.py (+44/-0); vllm/config/vllm.py (+20/-0); vllm/engine/arg_utils.py (+31/-0); vllm/model_executor/warmup/kernel_warmup.py (+6/-1)
LABELS: ready
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Purpose ⏎  ⏎ Support enabling/disabling FlashInfer autotune. ⏎  ⏎ cc @ProExpertProg  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-de3869bb4d  (L3, 2026-02-07, sha de3869bb4db7, PR #33943)
TITLE: move checks out of `unified_kv_cache_update` custom op (#33943)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_unified
FILES: vllm/model_executor/layers/attention/attention.py (+14/-6); vllm/model_executor/layers/attention/cross_attention.py (+3/-0); vllm/v1/attention/backends/flash_attn.py (+0/-10); vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+11/-20); vllm/v1/attention/backends/rocm_attn.py (+32/-42); vllm/v1/attention/backends/triton_attn.py (+17/-23); vllm/model_executor/models/whisper_causal.py (+3/-0)
LABELS: rocm, ready, v1, ready-run-all-tests
BODY: ## Purpose ⏎ Move checks for k, v, and kv_sharing_target_layer_name up from the `unified_kv_cache_update` into `Attention.forward`. ⏎  ⏎ Note that the other PRs in #32335 should follow this pattern as well; I think after all those PRs are merged, we can safely remove the `kv_sharing_target_layer_name` arg from the `AttentionImpl` class entirely. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-ed17f54c8b  (L3, 2026-02-07, sha ed17f54c8bcc, PR #33493)
TITLE: Perf tuning and expansion of cases covered for wvSplitKrc (#33493)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/rocm/skinny_gemms.cu (+143/-184); tests/kernels/quantization/test_rocm_skinny_gemms.py (+52/-31); vllm/model_executor/layers/utils.py (+19/-8)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: mi355 measurements before and after changes: ⏎ m,   n,  K   , bfor(us), aftr (us) ⏎ 128, 16, 2880, 4.55, 4.56 ⏎ 640, 16, 2880, 4.80, 4.83 ⏎ 128, 32, 2880, 3.91, **3.21** ⏎ 640, 32, 2880, 4.13, 4.05 ⏎ 128, 64, 2880, 4.42, **3.23** ⏎ 640, 64, 2880, 4.88, **4.43** ⏎ 128, 128, 2880, 4.51, **3.98** ⏎ 640, 128, 2880, 5.89, 5.92 ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-179ae7da8f  (L3, 2026-02-08, sha 179ae7da8f48, PR #33771)
TITLE: [Revert] Fix performance regression for GLM-4.7-GPTQ decode and MTP acceptance rate (#33771)
SOURCES: path_core, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+1/-3)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study revert record: partial_revert of PR(s) 31773 reason=performance_regression || deep-study correctness case vllm:179ae7da8f: class=perf_regression_as_correctness; symptom=performance_or_availability; introducing=#31773 || deep-study performance PR (perf_regression_fix)
BODY: ## Purpose ⏎  ⏎   Fix performance regressions introduced by two recent commits: ⏎   - `e0327c9db` (#31773) - Decode performance regression ⏎   ~~- `654a71fc3` (#32805) - MTP (Multi-Token Prediction) acceptance rate regression~~ ⏎  ⏎   **Affected Model:** GLM-4.7-GPTQ-INT4-INT8MIX ⏎   - Model body: int4/int8 mixed quantization ⏎   - MTP module (layer 92): manually configured as unquantized BF16 to increase acceptance rate ⏎  ⏎ ## Test Plan ⏎  ⏎   ### Performance Impact ⏎  ⏎  …[truncated]

### L3-084aa19f02  (L3, 2026-02-08, sha 084aa19f02b0, PR #33786)
TITLE: Add support for ModelOpt MXFP8 dense models (#33786)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/features/quantization/modelopt.md (+1/-0); vllm/config/model.py (+1/-0); vllm/model_executor/layers/fused_moe/config.py (+2/-0); vllm/model_executor/layers/quantization/__init__.py (+3/-1); vllm/model_executor/layers/quantization/modelopt.py (+247/-2); vllm/model_executor/layers/quantization/utils/mxfp8_utils.py (+121/-11)
LABELS: documentation, ready, nvidia
BODY: ## Purpose ⏎  ⏎ Add support for **ModelOpt MXFP8** dense models. ⏎  ⏎ **No support for MoE yet.** ⏎  ⏎ ### Related PRs ⏎ https://github.com/NVIDIA/Model-Optimizer/pull/736 ⏎  ⏎ ## Test Plan ⏎  ⏎ Use this LLM model (BF16): ⏎ https://huggingface.co/nvidia/OpenMath2-Llama3.1-8B ⏎  ⏎ Convert the model to MXFP8 using **ModelOpt**: ⏎ ``` ⏎ export MODEL_PATH=/my_home/hf_models/nvidia/OpenMath2-Llama3.1-8B ⏎ export OUTPUT_PATH=/my_home/hf_models/nvidia/OpenMath2-Llama3.1-8B-MXFP8 ⏎  ⏎ rm -rv …[truncated]

### L3-1ecfabe525  (L3, 2026-02-08, sha 1ecfabe5254f, PR #32958)
TITLE: glm 4.6 fused tuned inference config for B200 (#32958)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/configs/E=160,N=384,device_name=NVIDIA_B200,dtype=fp8_w8a8.json (+163/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: This PR adds a tuned fused MoE kernel configuration for the GLM-4.6 MoE architecture on NVIDIA B200 GPUs using FP8 quantization. ⏎  ⏎ Specifically, it targets the configuration: ⏎  ⏎ Experts (E): 160 ⏎ Sharded size N=384 for TP=4 ⏎ Device: NVIDIA B200 ⏎ Dtype: fp8_w8a8 ⏎ Previously, vLLM lacked a static configuration for these shapes on B200, causing it to fallback to heuristics or require JIT tuning during startup. This config improves startup time and ensures  …[truncated]

### L3-5a5c43511a  (L3, 2026-02-09, sha 5a5c43511ac9, PR #34052)
TITLE: fix(cpu): fix mla_decode compilation on x86 without AVX512 (#34052)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: csrc/cpu/mla_decode.cpp (+1/-10)
LABELS: ready, cpu
ISSUES: #33991 [Installation]: building docker cpu image with VLLM_CPU_DISABLE_AVX512=true (or on any x86_64 CPU without AVX512) fails to compile mla_decode.cpp because BFloat16 has no AVX2 fallback
BODY: ## Purpose ⏎  ⏎ Fixes #33991. ⏎  ⏎ This PR addresses a compilation failure in `mla_decode.cpp` when building on x86_64 CPUs without AVX512 support (or when explicitly disabling it via `VLLM_CPU_DISABLE_AVX512=true`). ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ * **Before:** The build failed during the compilation of `mla_decode.cpp.o` with `error: incomplete type ‘qk_vec_type’`. ⏎ * **After:** The build completes successfully.

### L3-d0d97e2974  (L3, 2026-02-09, sha d0d97e297425, PR #33810)
TITLE: [Misc] Fix up attention benchmarks (#33810)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/benchmarks.yaml (+11/-0); benchmarks/attention_benchmarks/batch_spec.py (+37/-0); benchmarks/attention_benchmarks/common.py (+10/-3); benchmarks/attention_benchmarks/configs/standard_attention.yaml (+10/-2); benchmarks/attention_benchmarks/runner.py (+151/-90)
LABELS: performance, ready, ci/build
BODY: The attention benchmarks rotted on blackwell, various fix ups.  ⏎  ⏎ Test plan: used for FA4 benchmarking https://github.com/vllm-project/vllm/pull/32974

### L3-285bab4752  (L3, 2026-02-09, sha 285bab47526c, PR #32846)
TITLE: [Kernel] use flashinfer for gdn prefill (#32846)
SOURCES: subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_next.py (+115/-2)
LABELS: performance, ready, qwen
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ flashinfer introduced prefill gdn kernel in https://github.com/flashinfer-ai/flashinfer/pull/2276 ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ vllm bench serve --model Qwen/Qwen3-Next-80B-A3B-Instruct --dataset-name random \ ⏎                                                                --num-prompts 512 \ ⏎                                                                --random-input-len 4096 \ ⏎                                                                --rand …[truncated]

### L3-d4f123cc48  (L3, 2026-02-09, sha d4f123cc48c3, PR #33985)
TITLE: [Kernel] FlashInfer: switch allreduce fusion to unified API (#33985)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_fused_collective.py (+54/-69); tests/compile/passes/distributed/test_fusion_all_reduce.py (+3/-2); vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+23/-43)
LABELS: performance, ready
BODY: ## Purpose ⏎  ⏎ - Migrate vLLM’s FlashInfer allreduce fusion to the unified `flashinfer.comm.allreduce_fusion` API and workspace creation. ⏎ - Update the fused collective benchmark and fusion test to the new API and workspace lifecycle. ⏎  ⏎ > **Dependency note:** Requires `flashinfer-python >= 0.6.3` (unified allreduce API). ⏎   ⏎ ## Test Plan ⏎  ⏎ `pytest tests/compile/distributed/test_fusion_all_reduce.py` ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ ================================== …[truncated]

### L3-995bbf38f1  (L3, 2026-02-09, sha 995bbf38f114, PR #34087)
TITLE: [Bugfix] Fix shared expert input for latent MoE in EP+DP (Nemotron-H) (#34087)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py (+1/-1); vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py (+1/-0); vllm/model_executor/layers/fused_moe/modular_kernel.py (+22/-2); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+3/-0); vllm/model_executor/layers/quantization/fp8.py (+1/-0); vllm/model_executor/layers/quantization/modelopt.py (+2/-0)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎  ⏎ Fix incorrect shared expert input when running Nemotron-H (latent MoE) with Expert Parallelism (EP) + Data Parallelism (DP). ⏎  ⏎ Models uses **latent MoE**, where routed experts operate on a latent-projected input (`[S, moe_latent_size]`) while shared experts must receive the original hidden states (`[S, hidden_size]`). When EP+DP is enabled, the `FusedMoEModularKernel` handles shared expert overlap internally via `_finalize()`, but it w …[truncated]

### L3-5e75a14a66  (L3, 2026-02-09, sha 5e75a14a667d, PR #33936)
TITLE: [Doc] Add DCP support to attention backend doc (#33936)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: docs/design/attention_backends.md (+26/-25); tools/pre_commit/generate_attention_backend_docs.py (+743/-619)
LABELS: documentation, ready
BODY: ## Purpose ⏎  ⏎ Also refactor table rendering to not be duplicated, extract FA/FI variant expansion, and compute capability parsing ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-e1060a71a1  (L3, 2026-02-09, sha e1060a71a1bb, PR #32975)
TITLE: [Perf] Optimize detokenizer python logic (#32975)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/engine/detokenizer.py (+8/-4); vllm/v1/engine/output_processor.py (+2/-2)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ 1. adding a `num_output_tokens` function, to avoid `len(self.output_token_ids)` might do a slice in SlowIncrementalDetokenizer ⏎  ⏎ ```py ⏎ class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer): ⏎     @property ⏎     def output_token_ids(self) -> list[int]: ⏎         return ( ⏎             self.token_ids ⏎             if not self.prompt_len ⏎             else (self.token_ids[self.prompt_len :]) ⏎         ) ⏎ ``` ⏎  ⏎ ~2. accumulate pieces and join once  …[truncated]

### L3-81e217fe6b  (L3, 2026-02-10, sha 81e217fe6b5a, PR #34187)
TITLE: [Bugfix] Fix DP Attention Padding in Dummy Run (#34187)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_model_runner.py (+1/-0)
LABELS: bug, ready, v1
ISSUES: #32626 [Bug]: TRTLLM Attention Failure with DP/EP | #33450 [Bug]: Attention Assertion
BODY: Mirror of #34009 by @benchislett — "Maintainers are allowed to edit this pull request." was not enabled on the original PR, so pushing review fixes was not possible. ⏎  ⏎ All credit to @benchislett for the original fix. ⏎  ⏎ ## Purpose ⏎  ⏎ FIX #32626  ⏎ FIX #33450 ⏎  ⏎ Problem: TRTLLM attention requires that num_decode_tokens be divisible by num_requests. However, during DP we sometimes do a dummy run on one of the workers so they don't get out of sync: it such c …[truncated]

### L3-afdce12c89  (L3, 2026-02-10, sha afdce12c8955, PR #33680)
TITLE: [Perf][Kernel] Add faster topKperRow decode kernel for DeepSeek-V3.2 sparse attention (#33680)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: vllm/v1/attention/backends/mla/indexer.py (+19/-0); CMakeLists.txt (+1/-0); csrc/ops.h (+4/-0); csrc/sampler.cu (+1/-1); csrc/topk.cu (+373/-0); csrc/torch_bindings.cpp (+6/-0); tests/kernels/test_top_k_per_row.py (+111/-0); vllm/model_executor/layers/sparse_attn_indexer.py (+39/-11)
LABELS: ready, ci/build, v1, deepseek
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎ This PR adds an optimized top-k per-row decode kernel for DeepSeek-V3.2 sparse attention (K = 2048). The new kernel replaces vLLM’s native `top_k_per_row_decode` in the decode path for long-context inference. It uses a 5-pass radix selection algorithm optimized for large-K workloads and long sequence lengths. The implementation is adapted from TileLang / SGLang (PR by DarkSharpness: https://github.com/sgl-project/sglang/pull/11194). ⏎  ⏎ # …[truncated]

### L3-578977bb5e  (L3, 2026-02-10, sha 578977bb5ed2, PR #31195)
TITLE: [SM100] Resubmit FMHA FP8 prefill for MLA (#31195)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/config/attention.py (+3/-0); vllm/model_executor/layers/attention/mla_attention.py (+138/-20); tests/v1/attention/test_mla_backends.py (+4/-3)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎ Resubmit FP8 FMHA path for MLA Prefill. Clean up how kernel backends opt into FP8 Prefill.  ⏎ Use `--attention-config.use_prefill_query_quantization=true` to enable FP8 Prefill. This is currently guarded because it shows slightly lower perf than BF16 Prefill due to extra casts and quantizations although the kernel level performance is about ~1.5x better. ⏎  ⏎ ## Test Plan ⏎ CI tests and running DeepSeekR1-FP4 manually.  ⏎  ⏎ ## Test Result ⏎ Compari …[truncated]

### L3-066c6da6a0  (L3, 2026-02-10, sha 066c6da6a049, PR #33738)
TITLE: [WideEP] Fix nvfp4 DeepEP High Throughput All2All backend (#33738)
SOURCES: corpus:kernel-correctness-cases
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+6/-2)
LABELS: ready, nvidia
DEEP_STUDY: deep-study correctness case vllm:066c6da6a0: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎  ⏎ On current main, `nvidia/DeepSeek-R1-NVFP4` crashes when used with the DeepEP high throughput all2all backend. ⏎  ⏎ ``` ⏎   NotImplementedError ⏎     File "vllm/distributed/device_communicators/all2all.py", line 355, in dispatch_router_logits ⏎       raise NotImplementedError ⏎ ``` ⏎  ⏎ This PR fixes the issue by rejecting the FLASHINFER_TRTLLM backend in this case. ⏎  ⏎ Details: `FLASHINFER_TRTLLM` is a monolithic kernel that requires `dispatch_router_l …[truncated]

### L3-5ee5c86eeb  (L3, 2026-02-10, sha 5ee5c86eeb00, PR #33884)
TITLE: [Bugfix][DeepSeek-V3.2] fix fp8 kvcache type cast (#33884)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+16/-4)
LABELS: bug, ready, deepseek
ISSUES: #33883 [Bug]: DeepSeek-V3.2 NVFP4 with fp8 kvcache reports `src_cache must be uint8`
BODY: ## Purpose ⏎  ⏎ Fix https://github.com/vllm-project/vllm/issues/33883 ⏎  ⏎ cc @LucasWilkinson ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ vllm serve --port=35496 --gpu-memory-utilization=0.85 --kv-cache-dtype=fp8 --tokenizer-mode=deepseek_v32 --no-enable-expert-parallel --enable-sleep-mode --model /root/workspaces/models/DeepSeek-V3.2-NVFP4 --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cuda"}' -tp 2 -dp 2 -dpl 2 -dpa 10.0.8.12 -dpr …[truncated]

### L3-cb9574eb85  (L3, 2026-02-11, sha cb9574eb8528, PR #34111)
TITLE: [XPU][9/N] clean up existing ipex code/doc (#34111)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.flash_attn.upstream_pip, L3.flash_attn.fa_utils
FILES: vllm/v1/attention/backends/fa_utils.py (+4/-5); vllm/v1/attention/ops/paged_attn.py (+1/-1); docker/Dockerfile.cpu (+0/-1); docs/getting_started/installation/gpu.xpu.inc.md (+5/-4); tests/quantization/test_cpu_wna16.py (+1/-1); tests/quantization/test_ipex_quant.py (+0/-32); vllm/_xpu_ops.py (+3/-3); vllm/model_executor/layers/quantization/mxfp4.py (+1/-1); vllm/model_executor/layers/sparse_attn_indexer.py (+1/-1); vllm/platforms/cpu.py (+0/-1)
LABELS: documentation, ready, ci/build, v1, cpu
BODY: ## Purpose ⏎ part of  https://github.com/vllm-project/vllm/issues/33214 ⏎ clean up ipex, use xpu_ops instead.  ⏎ also update xpu documents. ⏎  ⏎ ## Test Plan ⏎ CI ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-c7914d30f9  (L3, 2026-02-11, sha c7914d30f90b, PR #34043)
TITLE: Reapply [Attention][FA3] Update FA3 to include new swizzle optimization (#34043)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, corpus:confirmed-reverts, corpus:performance-pr-population
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fork_build, L3.mla.flashattn
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/v1/attention/backends/flash_attn.py (+11/-2); vllm/v1/attention/backends/mla/flashattn_mla.py (+11/-1); tests/v1/cudagraph/test_cudagraph_dispatch.py (+13/-9); vllm/forward_context.py (+3/-15); vllm/v1/cudagraph_dispatcher.py (+21/-16)
LABELS: performance, ready, ci/build, v1, nvidia
DEEP_STUDY: deep-study revert record: reland of PR(s) 23465 reason=other || deep-study performance PR (kernel_optimization)
BODY: Reapply https://github.com/vllm-project/vllm/pull/23465 after revert in https://github.com/vllm-project/vllm/pull/33841 but with correct metadata sizes

### L3-fd618871b4  (L3, 2026-02-11, sha fd618871b41c, PR #33948)
TITLE: [Bugfix]: Fix ROCm fusion attn test; use AttentionBackend utils to create kv cache (#33948)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/passes/test_fusion_attn.py (+27/-52)
LABELS: bug, rocm, ready
BODY: Reuse `AttentionBackend` utils to initialize the dummy KV cache with the required shape and stride. Also, turns out this UT is being skipped on ROCm because of a typo/missing rename, so `BACKENDS` -> `BACKENDS_FP8`. ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-275e0d2a99  (L3, 2026-02-11, sha 275e0d2a993b, PR #33715)
TITLE: [NVIDIA][test] Tests for flashinfer TRTLLM BF16 MoE (#33715)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-BF16-fi-cutlass.yaml (+2/-0); tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml (+1/-0); tests/kernels/moe/test_flashinfer.py (+41/-0); tests/kernels/moe/test_moe.py (+100/-0); tests/kernels/moe/test_unquantized_backend_selection.py (+132/-0); tests/quantization/test_blackwell_moe.py (+8/-0); vllm/model_executor/layers/fused_moe/oracle/unquantized.py (+12/-1)
LABELS: ready, nvidia
BODY: ## Purpose ⏎ Adding tests for trtllm bf16 moe backend added in PR [[NVIDIA] [feat] Integrate flashinfer Trtllmgen bf16 moe #32954](https://github.com/vllm-project/vllm/pull/32954) ⏎  ⏎ ## Test Plan ⏎ - unit and integration test for the new moe backend ⏎ - unit tests for utility functions ⏎ - Changing E2E tests to use fi cutlass because E2E tests triggered an intermittent issue with flashinfer ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-5aff2699bd  (L3, 2026-02-11, sha 5aff2699bdce, PR #34316)
TITLE: Fix CI failure - Flashinfer Kernel tests (#34316)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer.py (+1/-0); tests/kernels/moe/test_flashinfer_moe.py (+1/-0); tests/kernels/moe/test_pplx_cutlass_moe.py (+1/-0)
LABELS: ready, nvidia
ISSUES: #34315 [CI Failure]: Flashinfer Kernel tests missing argument: 'num_logical_experts'
BODY: ## Purpose ⏎ Fix #34315 ⏎  ⏎ ## Test Plan ⏎ Tested locally with ⏎ ``` ⏎ pytest -v -s tests/kernels/moe/test_flashinfer.py ⏎ ``` ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-d1b837f0ae  (L3, 2026-02-11, sha d1b837f0ae6a, PR #34116)
TITLE: [CPU] Enable FP16 (Half dtype) support for s390x (#34116)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: csrc/cpu/mla_decode.cpp (+2/-2); csrc/cpu/cpu_attn_impl.hpp (+1/-1); csrc/cpu/cpu_types_vxe.hpp (+241/-6)
LABELS: ready, cpu
BODY: ## Purpose ⏎ Adds FP16 model inference support for s390x (IBM Z) architecture using vectorized Bit-manipulation. FP16 was previously disabled on s390x, limiting users to BF16 or FP32. ⏎  ⏎ ## Test Plan ⏎ 1.  Build docker image and run the server with `dtype=half` ⏎ 2. Send an inference request to the server ⏎ ## Test Result ⏎  ⏎ ``` ⏎ [root@b314lp81 vllm]# docker run --rm -p 8000:8000 local:test   ibm-granite/granite-4.0-micro   --port=8000   --dtype=half  ⏎ INFO 02 …[truncated]

### L3-64f570ab56  (L3, 2026-02-11, sha 64f570ab56ca, PR #33681)
TITLE: [ROCm] [aiter] Split KV cache update for AiterFlashAttention (#33681)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+68/-40)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎ Supporting #32335, this PR extracts KV cache update from the attention forward pass in AiterFlashAttention. ⏎  ⏎ ROCM_AITER_FA supports both flash and shuffled KV cache layouts. This PR covers both of them and uses the same flag to control the respective KV cache layouts. ⏎  ⏎ ## Test Plan ⏎ Accuracy test with lm_eval ⏎  ⏎ Server command ⏎ ``` ⏎ VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT={0,1} \ ⏎ VLLM_ROCM_USE_AITER=1 \ ⏎ vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 …[truncated]

### L3-fa7e0bfacf  (L3, 2026-02-11, sha fa7e0bfacfb4, PR #32458)
TITLE: [CI][BugFix] Fix silent failure in shellcheck hook and baseline exist… (#32458)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tools/pre_commit/shellcheck.baseline (+89/-0); tools/pre_commit/shellcheck.sh (+37/-2)
LABELS: bug, ready
BODY: related issue: https://github.com/vllm-project/vllm/issues/32391 ⏎  ⏎ ## Purpose ⏎ CI was not reliably running shellcheck due to invalid find invocation, so issues could slip through unnoticed. ⏎ Initially started as a small one-line fix but after correcting the find command, CI began failing due to pre-existing shellcheck errors in the affected scripts. This PR restores a working signal by fixing the CI invocation and introducing a baseline, so CI only  …[truncated]

### L3-f120bd42d3  (L3, 2026-02-12, sha f120bd42d3da, PR #33506)
TITLE: [Kernel] Support Flashinfer trtllm fused MoE non gated FP8 & NVFP4 (#33506)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer.py (+46/-12); vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py (+8/-6); vllm/model_executor/layers/quantization/modelopt.py (+4/-3); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+51/-19); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+88/-5)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Add support for Flashinfer trtllm fused MoE non-gated activation for FP8 and for NVFP4. ⏎  ⏎ Changes: ⏎ - Pass `activation_type` argument to FlashInfer trtllm fused MoE FP8 and NVFP4. ⏎ - Add DeepSeek routing to supported routing list of Flashinfer trtllm fused MoE FP8 ⏎ - Add support to non-gated flow in Flashinfer trtllm fused MoE NVFP4 ⏎ - Use `min_alignment=128` (padding) for non-gated activation in Flashinfer trtllm fused MoE ⏎ - Fix `tests/ke …[truncated]

### L3-9ea1f598ce  (L3, 2026-02-12, sha 9ea1f598ce48, PR #34378)
TITLE: Use paged_attention_v1 for sliding window decode in rocm_aiter_fa (#34378)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+2/-29)
LABELS: rocm, ready, v1, meta-exported, fb-exported
BODY: Summary: Replace unified_attention (Triton) with paged_attention_v1 for the sliding window decode path in AiterFlashAttentionImpl. paged_attention_v1 already supports sliding window natively via its sliding_window parameter, so this unifies the NHD decode path for both sliding window and non-sliding window cases. The sliding window value is recovered from the flash-attn convention (self.sliding_window[0] + 1), which yields 0 (disabled) when no sl …[truncated]

### L3-f2c47886fd  (L3, 2026-02-12, sha f2c47886fdba, PR #33451)
TITLE: [Attention] Add FlashInfer Sparse MLA backend (#33451)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_sparse, L3.mla.flashinfer_sparse, L3.dispatch.selector, L3.dispatch.registry, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/model_executor/layers/attention/mla_attention.py (+1/-0); vllm/platforms/cpu.py (+1/-0); vllm/platforms/cuda.py (+43/-8); vllm/platforms/interface.py (+1/-0); vllm/platforms/rocm.py (+1/-0); vllm/platforms/xpu.py (+1/-0); vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+353/-0); vllm/v1/attention/backends/mla/flashmla_sparse.py (+3/-161); vllm/v1/attention/backends/mla/sparse_utils.py (+191/-0); vllm/v1/attention/backends/registry.py (+4/-0); (+14 more)
LABELS: documentation, performance, rocm, ready, ci/build, v1, cpu, nvidia
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ FlashInfer supports sparse MLA as of https://github.com/flashinfer-ai/flashinfer/pull/2138. This PR integrates support for this kernel in vLLM, and enables it by default when head count <= 16 (TP8 or higher) ⏎  ⏎ ## Test Plan ⏎ ### Performance ⏎ ``` ⏎ python benchmarks/attention_benchmarks/benchmark.py --config benchmarks/attention_benchmarks/configs/mla_prefill.yaml ⏎ ``` ⏎  ⏎ ### Correctness ⏎ ``` ⏎ vllm serve deepseek-ai/DeepSeek-V3.2 -tp 8 -ep ⏎ lm_eval …[truncated]

### L3-b86bf4417e  (L3, 2026-02-12, sha b86bf4417e31, PR #33907)
TITLE: [Bugfix] Fix Random Dataset Prefix Length Inaccuracy (#33907)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/benchmarks/datasets.py (+29/-10)
LABELS: bug, performance, ready
BODY: ## Purpose ⏎ This PR fixes the `RandomDataset` prefix generation so the shared prefix token length is adjusted via decode-encode tokenization, similar to the way the input prompt is generated. ⏎  ⏎ ## Test Plan ⏎ Observe prefix cache hit rate before and after. For testing, a prefix length of 4096 with an input length of 1024 is used. The expected prefix cache hit rate should thus be around 80% `(4096/(4096+1024) = 0.8)`.  ⏎  ⏎ ## Test Result ⏎ Before: ⏎  ⏎ ``` ⏎ CUD …[truncated]

### L3-0916e7960b  (L3, 2026-02-13, sha 0916e7960bdd, PR #34498)
TITLE: [GDN] Use CPU tensors to build GDN metadata (#34498)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+2/-2); vllm/v1/attention/backends/gdn_attn.py (+10/-7)
LABELS: v1
BODY: Currently, the GDN metadata builder causes CPU-GPU sync by using `gpu_tensor.item()`. This PR avoids this by using CPU tensors instead of GPU tensors.

### L3-87789c8364  (L3, 2026-02-13, sha 87789c836422, PR #34523)
TITLE: [Misc] vLLM's --enforce-eager should turn off compile and cudagraphs only (#34523)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+7/-7)
LABELS: ready, nvidia
BODY: ## Purpose ⏎  ⏎ Previously, --enforce-eager sets -O0. -O0 turns off compile and cudagraphs but also disables misc things that are present in the default -O2, like the flashinfer autotuning settings. ⏎  ⏎ --enforce-eager is primarily used as a debugging tool. If a model doesn't support compile, then it will individually turn off compile (via the lack of a support_torch_compile decorator). As a debugging tool, it should just be "turn off these compile-mode …[truncated]

### L3-3d2a026fd0  (L3, 2026-02-13, sha 3d2a026fd031, PR #33368)
TITLE: [Feature] Pipeline Parallel Async send/recv, 2.9% E2E throughput improvement (#33368)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/distributed/test_comm_ops.py (+107/-0); vllm/distributed/parallel_state.py (+137/-76); vllm/v1/worker/gpu_worker.py (+54/-5)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Part of the https://github.com/vllm-project/vllm/issues/33356 ⏎  ⏎ Enable async `send/recv` for better performance ⏎  ⏎ ## Test ⏎  ⏎ `export MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"` ⏎  ⏎ `vllm serve $MODEL -pp 4 --port 9256 --max-num-seqs 128  --async-scheduling` ⏎  ⏎ ### Acc ⏎  ⏎ `lm_eval --model local-completions --model_args "base_url=http://127.0.0.1:9256/v1/completions,model=$MODEL,num_concurrent=1024" --tasks gsm8k` ⏎  ⏎ ```bash ⏎ |Tasks|Version|     Fi …[truncated]

### L3-de42abb366  (L3, 2026-02-13, sha de42abb36603, PR #34294)
TITLE: [CI] Heavy refactoring of Voxtral multimodal audio model tests (#34294)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+7/-2); requirements/rocm-test.txt (+2/-0); tests/conftest.py (+0/-2); tests/models/multimodal/generation/test_voxtral.py (+138/-43); tests/models/multimodal/generation/test_voxtral_realtime.py (+6/-2); tests/models/multimodal/generation/vlm_utils/model_utils.py (+88/-0); tests/models/multimodal/processing/test_common.py (+34/-14); vllm/model_executor/models/voxtral.py (+28/-0); vllm/model_executor/models/whisper_causal.py (+44/-4); vllm/reasoning/mistral_reasoning_parser.py (+2/-2); (+1 more)
LABELS: rocm, ready, ci/build, v1, multi-modality
ISSUES: #34283 [CI Failure]: Multi Modal Models (Extended) 1
BODY: FIX: https://github.com/vllm-project/vllm/issues/34283 ⏎  ⏎ Adds three-layer accuracy tests for the Voxtral multimodal audio model (`mistralai/Voxtral-Mini-3B-2507`): ⏎  ⏎ 1. **Offline vLLM greedy inference** --- validates mistral-format weight loading and audio preprocessing. ⏎ 2. **HF Transformers greedy inference** --- independent ground truth via `AutoProcessor` + `VoxtralForConditionalGeneration`. ⏎ 3. **Online OpenAI-compatible API serving** --- valida …[truncated]

### L3-b3c14229b0  (L3, 2026-02-14, sha b3c14229b032, PR #34538)
TITLE: [ROCm][CI] Guard sparse MLA backend imports for ROCm compatibility in tests (#34538)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/attention/test_sparse_mla_backends.py (+11/-0)
LABELS: rocm, ready, v1
BODY: `tests/v1/attention/test_sparse_mla_backends.py` fails to collect on ROCm because it unconditionally imports `FlashInferMLASparseBackend` (which depends on `flashinfer`). This causes an `ImportError` at module level, blocking the entire test file. ⏎  ⏎ ## Solution ⏎  ⏎ - Wrap imports of `FlashInferMLASparseBackend` in `try`/`except` block, falling back to `None` when unavailable. ⏎ - Build the `_SPARSE_BACKENDS` parametrize list dynamically so only availab …[truncated]

### L3-79f3fab05a  (L3, 2026-02-14, sha 79f3fab05a2d, PR #34494)
TITLE: [Bugfix] Handle num_expert_group=None in flashinfer block-scale FP8 MoE (#34494)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer.py (+77/-0); vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py (+1/-0)
LABELS: bug, ready, nvidia
ISSUES: #34477 [Bug]: TVM_FFI_ICHECK(args->n_group != 0) << "n_group should not be zero for DeepSeekV3 routing"
BODY: ## Purpose ⏎  ⏎ Fixes #34477 ⏎  ⏎ - Add missing `None→0` guard for `num_expert_group` in `flashinfer_fused_moe_blockscale_fp8`, matching the pattern already used for `topk_group` on the next line and in the sister function `fi_trtllm_fp8_per_tensor_moe` ⏎ - Add regression test covering `num_expert_group=None` with `DeepSeekV3` routing on the block-scale FP8 path ⏎  ⏎ **Root cause**: MiniMax-M2.1 uses `scoring_func="sigmoid"` with `e_score_correction_bias` but  …[truncated]

### L3-71cd89264f  (L3, 2026-02-15, sha 71cd89264f6c, PR #32183)
TITLE: [MM Encoder] Add Triton ViT attention backend (#32183)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/model_executor/layers/attention/mm_encoder_attention.py (+38/-0); vllm/platforms/cuda.py (+19/-7); vllm/platforms/rocm.py (+1/-0); vllm/v1/attention/ops/vit_attn_wrappers.py (+77/-0); tests/kernels/attention/test_mha_attn.py (+16/-2); vllm/model_executor/models/dots_ocr.py (+5/-4); vllm/model_executor/models/ernie45_vl.py (+5/-4); vllm/model_executor/models/glm4_1v.py (+5/-4); vllm/model_executor/models/paddleocr_vl.py (+2/-8); vllm/model_executor/models/qwen2_5_vl.py (+1/-9); (+4 more)
LABELS: performance, rocm, ready, v1, qwen, nvidia
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ - Currently, if we use FP32 for ViT, TORCH SDPA might be inefficent. This PR adds Triton Attention backend for `MMEncoderAttention` for a better performance. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest -s -v tests/kernels/attention/test_mha_attn.py ⏎ ``` ⏎  ⏎ ## Test Result ⏎ - Test should pass. ⏎  ⏎ ## Benchmark results ⏎ ``` ⏎ vllm bench serve   --backend openai-chat   --model /home/mozf/LLM/Qwen3-VL-4B-Instruct/   --endpoint /v1/chat/completions   --dataset-name rand …[truncated]

### L3-bb85929aa6  (L3, 2026-02-15, sha bb85929aa6f3, PR #34548)
TITLE: [BugFix] Fix Python 3.13 FlashMLA import error (#34548)
SOURCES: path_core, subject_keyword, dependency_pin, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_build
FILES: cmake/external_projects/flashmla.cmake (+1/-1)
LABELS: bug, ready, ci/build
ISSUES: #34504 [Bug]: GLM-5-FP8 Crash - Engine core initialization failed
BODY: FIX https://github.com/vllm-project/vllm/issues/34504 (Potentially) ⏎  ⏎ Pickup https://github.com/vllm-project/FlashMLA/commit/692917b1cda61b93ac9ee2d846ec54e75afe87b1

### L3-dc5fa77a4e  (L3, 2026-02-17, sha dc5fa77a4eb6, PR #34457)
TITLE: [Bugfix][MTP][Sparse MLA] Allow sparse MLA with MTP to run with FULL cudagraphs (#34457)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+16/-6); docs/design/cuda_graphs.md (+1/-0)
LABELS: bug, documentation, performance, ready, v1, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ `DeepseekV32IndexerMetadataBuilder` currently only reports support for `UNIFORM_SINGLE_TOKEN_DECODE`. Therefore, when running a sparse MLA model with MTP, FULL cudagraphs are never captured. ⏎  ⏎ In reality, the deepGEMM kernel `fp8_paged_mqa_logits` supports MTP with `num_speculative_tokens=1` (i.e. `next_n = 2`). ⏎  ⏎ This PR changes the reported support to `UNIFORM_BATCH` and adds an explicit error for `num_speculative_tokens > 1`, rather t …[truncated]

### L3-7743152957  (L3, 2026-02-17, sha 774315295723, PR #33600)
TITLE: [Attention] Refactor `check_and_update_config` (#33600)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.dispatch.abstract_interface, L3.platform.cuda_selection
FILES: vllm/v1/attention/backend.py (+12/-7); vllm/config/cache.py (+4/-7); vllm/engine/arg_utils.py (+1/-2); vllm/platforms/cuda.py (+253/-156)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study: this PR was reverted by PR 34979 (confirmed_revert, reason=correctness_or_accuracy) || deep-study: this PR was reverted by PR 35122 (reland, reason=other)
BODY: ## Purpose ⏎ `check_and_update_config` unnecessarily duplicates much of the logic from the attention selector in order to set an approproate block size. This PR refactors `check_and_update_config` to use the selector, which will be simpler to maintain going forward. ⏎  ⏎ * If the user specifies `--block-size` and `--attention-backend` and the backend doesn't support the block size, we raise an error rather than overriding a user selection ⏎ * If the user …[truncated]

### L3-4a00a511bb  (L3, 2026-02-17, sha 4a00a511bbf7, PR #34653)
TITLE: [BugFix] [Build] fix string literals comparison in indexer_k_quant_and_cache calling site (#34653)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+2/-1)
LABELS: bug, ready
ISSUES: #34132 [Bug]: Compilation Error with Clang 22+ (ROCm) due to String Literal Comparison in  quant_utils.cuh
BODY: ## Purpose ⏎ Fix https://github.com/vllm-project/vllm/issues/34132 caused by two pointer compariton. ⏎  ⏎ Related to string literals comparison using == with clang20+  ⏎  ⏎ ``` ⏎    DISPATCH_BY_KV_CACHE_DTYPE(k.dtype(), "fp8_e4m3", ⏎    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ⏎ csrc/quantization/w8a8/fp8/amd/quant_utils.cuh:643:18: note: expanded from macro 'DISPATCH_BY_KV_CACHE_DTYPE' ⏎     if (KV_DTYPE == "auto") { ⏎         ~~~~~~~~ ^  ~~~~~~ ⏎ ``` ⏎  ⏎ Inste …[truncated]

### L3-cef65f0715  (L3, 2026-02-18, sha cef65f071592, PR #34753)
TITLE: [ROCm][CI] Removed hard-coded attn backend requirement for Qwen VL (#34753)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/generation/test_common.py (+0/-7)
LABELS: rocm, ready, multi-modality, qwen
BODY: Falling back to default attention backend for `pytest -s -v tests/models/multimodal/generation/test_common.py::test_video_models[qwen3_vl-test_case20]`. AITER FA was not matching logprobs on MI355. That is because AITER FA is sacrificing exact logprob match for more performance.

### L3-e24663c5a9  (L3, 2026-02-18, sha e24663c5a958, PR #34228)
TITLE: Add unit tests for fp8 output fusion of triton_attn (#34228)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_triton_unified_attention.py (+127/-1)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Adding unit tests to cover fused quantization inside the `triton_unified_attention` kernels (and help debugging issues like #31785 ).  ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ pytest tests/kernels/attention/test_triton_unified_attention.py ⏎ ``` ⏎  ⏎ ## Test Result ⏎  ⏎ All 1536 tests pass on H100 & B200.  ⏎  ⏎ --- ⏎ [details omitted]

### L3-6874638bc4  (L3, 2026-02-18, sha 6874638bc443, PR #34758)
TITLE: [Model Bash] DeepSeek R1 BF16 Min Latency QKV A GEMM (0.5% E2E Speedup) (#34758)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: vllm/model_executor/layers/mla.py (+1/-0); CMakeLists.txt (+19/-0); csrc/dsv3_fused_a_gemm.cu (+747/-0); csrc/ops.h (+5/-0); csrc/torch_bindings.cpp (+5/-0); vllm/_custom_ops.py (+18/-0); vllm/model_executor/models/deepseek_v2.py (+60/-3)
LABELS: ready, ci/build, deepseek
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ - add min latency bf16 qkv_a_gemm ⏎ - adapted from sgl:  * https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/gemm/dsv3_fused_a_gemm.cu, which adapted from trtllm ⏎ - disappointing E2E win, but sets up using PDL to overlap with AR in future PR ⏎  ⏎ ## Test Plan ⏎ - lm eval ⏎ ```bash ⏎ local-completions ({'model': 'nvidia/DeepSeek-R1-NVFP4', 'base_url': 'http://localhost:7000/v1/completions', 'num_concurrent': 1000, 'tokenized_requests': …[truncated]

### L3-847a57cd12  (L3, 2026-02-18, sha 847a57cd1217, PR #34673)
TITLE: [Bugfix][MoE Kernel] Fix incorrect routing selection for models without expert groups (e.g., MiniMax-M2.1) (#34673)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer.py (+0/-77); vllm/model_executor/layers/fused_moe/config.py (+17/-10); vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py (+2/-0); vllm/model_executor/layers/fused_moe/router/fused_topk_router.py (+2/-0); vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py (+11/-9)
LABELS: bug, ready, nvidia
BODY: Previously, `get_routing_method_type()` returned `DeepSeekV3` for any sigmoid-scored model with top_k > 1, regardless of whether the model actually uses grouped expert routing. This caused crashes in the flashinfer TRTLLM kernel for models like MiniMax-M2.1 that has no expert groups (`num_expert_group=None`).  ⏎  ⏎ This PR fixes the error by assigning it to `RoutingMethodType.Unspecified` when there is no `num_expert_group` ⏎  ⏎ Related to: https://githu …[truncated]

### L3-4fb8beefaa  (L3, 2026-02-19, sha 4fb8beefaa8b, PR #34914)
TITLE: [Bugfix] Fix cutlass fp8 kernel on hopper for Qwen3.5 (#34914)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+11/-0)
LABELS: bug, ready, qwen, nvidia
BODY: ## Purpose ⏎  The FlashInfer CUTLASS MoE kernel on Hopper GPUs produces NaN when FP8 block scales are extremely small (~1e-23). `Qwen3.5-397B-A17B-FP8` model has 18 out of 512 "dead" experts with near-zero block scales. When tokens get routed to these experts, the kernel outputs `NaN` instead of near-zero values. ⏎  ⏎ ## Test Plan ⏎ Prior to this PR, the model is broken on main and will produce gibberish by default, and can be "fixed" by specfiying `VLLM …[truncated]

### L3-662205d34e  (L3, 2026-02-19, sha 662205d34eb1, PR #34818)
TITLE: [Bugfix] Fix Basic Models Test (#34818)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.mla.common_v1, L3.platform.cuda_selection
FILES: vllm/model_executor/layers/attention/chunked_local_attention.py (+6/-5); vllm/model_executor/layers/attention/mla_attention.py (+12/-5); tests/models/multimodal/processing/test_tensor_schema.py (+4/-1); tests/models/utils.py (+7/-2); tests/v1/spec_decode/test_eagle.py (+1/-1); vllm/config/cache.py (+2/-2); vllm/config/vllm.py (+51/-46); vllm/platforms/cuda.py (+62/-155); vllm/platforms/interface.py (+7/-0); vllm/v1/engine/core.py (+8/-1); (+4 more)
LABELS: bug, speculative-decoding, ready, v1, multi-modality, ci-failure, nvidia
ISSUES: #34806 [CI Failure]: models/test_initialization.py::test_can_initialize_large_subset[EagleMiniCPMForCausalLM]
DEEP_STUDY: deep-study: this PR was reverted by PR 34979 (confirmed_revert, reason=correctness_or_accuracy)
BODY: ## Purpose ⏎ Fixes https://github.com/vllm-project/vllm/issues/34806, https://github.com/vllm-project/vllm/issues/34810, https://github.com/vllm-project/vllm/issues/34814, and https://github.com/vllm-project/vllm/issues/34819 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-81bfc21a6a  (L3, 2026-02-19, sha 81bfc21a6ad0, PR #34260)
TITLE: [Model Bash]: Improve FP8 Oracle for Config Specific Kernel Selection (#34260)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/oracle/fp8.py (+46/-13)
LABELS: ready
ISSUES: #34249 [Bug]: [Fp8] [MoE] 'FLASHINFER_CUTLASS' is auto-selected as MoE backend instead of 'DEEPGEMM' on hopper
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: On Hopper (SM90), FLASHINFER_CUTLASS was auto-selected as the FP8 MoE backend over DEEPGEMM due to having higher priority in the selection order. FlashInfer kernels are primarily optimized for Blackwell and do not work well with features like chunked prefill on Hopper. ⏎  ⏎ This change makes the backend priority order architecture-aware: ⏎ - Blackwell (SM100+): FlashInfer backends preferred over DeepGEMM ⏎ - Hopper and older: DeepGEMM preferred over Flas …[truncated]

### L3-059779231f  (L3, 2026-02-19, sha 059779231f15, PR #34916)
TITLE: [Minor] Add logging when using MXFP4 MXFP8 TRTLLM backend (#34916)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/mxfp4.py (+3/-0)
LABELS: ready
BODY: ## Purpose ⏎ When setting `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1`, it was noticed that there was no logging that MXFP4 MXFP8 TRTLLM was the selected backend. Other backends such as `SM100_FI_MXFP4_BF16` and `SM100_FI_MXFP4_MXFP8_CUTLASS` do perform that logging: ⏎ ``` ⏎ VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=1 vllm serve openai/gpt-oss-20b ⏎ ... ⏎ (EngineCore_DP0 pid=1494838) INFO 02-19 20:06:21 [gpu_model_runner.py:4128] Starting to load model openai …[truncated]

### L3-8de7c636cc  (L3, 2026-02-19, sha 8de7c636cc02, PR #32877)
TITLE: [Bugfix][Hardware][AMD] Fix ROCM_AITER_FA speculative decoding support (#32877)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+35/-2)
LABELS: bug, rocm, ready, v1
ISSUES: #31625 [Feature][ROCm][AITER]: Speculative Decoding Accuracy Issue with VLLM_ATTENTION_BACKEND=ROCM_AITER_FA
BODY: ## Summary ⏎ - Fix ROCM_AITER_FA attention backend to support speculative decoding (multi-token decode) ⏎ - The decode path was hardcoding `max_seqlen_q=1`, causing incorrect results with speculative decoding ⏎  ⏎ ## Changes ⏎ - Extract actual `max_query_len` from decode metadata ⏎ - Route multi-token decode queries to `unified_attention` instead of `paged_attention_v1` ⏎ - Use actual `max_seqlen_q` instead of hardcoded `1` ⏎ - Update assertion message to reflec …[truncated]

### L3-1fe462168c  (L3, 2026-02-20, sha 1fe462168c38, PR #34870)
TITLE: [perf] Avoid dtype promotion sync in mamba_get_block_table_tensor (#34870)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+6/-2)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ From the perf profiling on a model using linear attention, one thing draws the attention is `aten::to` from somewhere in vllm gpu_model_runner preprocess ⏎  ⏎ <img width="1386" height="372" alt="image" src="https://github.com/user-attachments/assets/8b0a8714-edab-4ece-add9-6d719ced13e5" /> ⏎  ⏎ For this `aten::to`, ⏎ <img width="432" height="550" alt="image" src="https://github.com/user-attachments/assets/d0e04a74-313b-468d-8753-7b49d751bb01" / …[truncated]

### L3-aaefc58ee0  (L3, 2026-02-20, sha aaefc58ee0f0, PR #34979)
TITLE: [CI] Revert PRs 34818 and 33600 (#34979)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.mla.common_v1, L3.dispatch.abstract_interface, L3.platform.cuda_selection
FILES: vllm/model_executor/layers/attention/chunked_local_attention.py (+5/-6); vllm/model_executor/layers/attention/mla_attention.py (+5/-12); vllm/v1/attention/backend.py (+7/-12); tests/models/multimodal/processing/test_tensor_schema.py (+1/-4); tests/models/utils.py (+2/-7); tests/v1/spec_decode/test_eagle.py (+1/-1); vllm/config/cache.py (+7/-4); vllm/config/vllm.py (+46/-51); vllm/engine/arg_utils.py (+2/-1); vllm/platforms/cuda.py (+163/-167); (+6 more)
LABELS: speculative-decoding, ready, v1, multi-modality, nvidia
ISSUES: #34969 [CI Failure]: LM Eval Small Models (B200) - DeepSeek and Qwen3 Next
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 33600;34818 reason=correctness_or_accuracy
BODY: https://github.com/vllm-project/vllm/pull/33600 broke the basic models test, https://github.com/vllm-project/vllm/pull/34818 tried to fix it but caused accuracy issues for ⏎ ``` ⏎ FAILED evals/gsm8k/test_gsm8k_correctness.py::test_gsm8k_correctness[DeepSeek-V2-Lite-Instruct-FP8] - AssertionError: GSM8K metric too low: 0.0114 < 0.7200 - 0.0800 = 0.6400 ⏎ assert np.float64(0.011372251705837756) >= (0.72 - 0.08) ⏎ FAILED evals/gsm8k/test_gsm8k_correctness.p …[truncated]

### L3-ea5f903f80  (L3, 2026-02-20, sha ea5f903f80fe, PR #34899)
TITLE: Bump Flashinfer Version and Re-enable DeepSeek NVFP4 AR+Norm Fusion (#34899)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); docker/Dockerfile.nightly_torch (+2/-2); docker/versions.json (+1/-1); requirements/cuda.txt (+1/-1); vllm/model_executor/models/config.py (+1/-24)
LABELS: ready, ci/build, deepseek, nvidia, ready-run-all-tests
ISSUES: #34395 [Bug]: AR+rms+fp4 fusion results in total accuracy collapse for DSV3-fp4
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ [The patch](https://github.com/flashinfer-ai/flashinfer/pull/2557) for fixing the Deepseek V3 accuracy issue with AR+rms+fp4 fusion (#34395) is included in flashinfer 0.6.4. This PR bumps flashinfer version and re-enables the fusion pass by default. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ serve nvidia/DeepSeek-V3.1-NVFP4 -tp=4 -cc.pass_config.fuse_allreduce_rms=True ⏎ ``` ⏎  ⏎ ## Test Result ⏎ ``` ⏎ lm_eval --model local-completions --model_args "base_url=http://0.0. …[truncated]

### L3-f24b2de3d3  (L3, 2026-02-20, sha f24b2de3d381, PR #34473)
TITLE: [Test] Add FP8 KV Cache Testing for MLA Backends (#34473)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/attention/test_mla_backends.py (+68/-27)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, ready, ci/build, v1, multi-modality
BODY: ## Purpose ⏎ This PR improves the MLA backend test coverage to include fp8 kv cache testing. ⏎  ⏎ ## Test Plan ⏎ Tested `tests/v1/attention/` ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-ded333fb9b  (L3, 2026-02-20, sha ded333fb9b90, PR #34636)
TITLE: [ROCm][Bugfix]: Only save unpadded sizes for shared_experts in MoERunner to fix rmsnorm pad fusion (#34636)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py (+6/-3)
LABELS: bug, rocm, ready
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Purpose ⏎ #32344 introduced a silent regression for gpt-oss on ROCm by disabling the pattern matching for the RMSNorm+padding fusion introduced in #30976. ⏎  ⏎ Specifically, passing the `original_hidden_states` into the `moe_forward` custom op breaks [pattern matching](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py#L430) for the `AddAiterRMSNormPadPattern`, since there is an additional user `auto_f …[truncated]

### L3-89358f0d35  (L3, 2026-02-20, sha 89358f0d35e7, PR #34567)
TITLE: [CI] Fix ColBERT HF comparison tests on AMD CI + refactor (#34567)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/language/pooling/test_colbert.py (+107/-149)
LABELS: rocm, ready
BODY: This PR fixes the `test_colbert_hf_comparison_modernbert` crash on AMD CI and refactors the ColBERT HF comparison tests for maintainability and cross-platform robustness. ⏎  ⏎ `test_colbert_hf_comparison_modernbert` crashes on AMD CI nightly because the HF reference model runs on CPU while ModernBERT defaults to Triton-based flash attention: ⏎  ⏎ [details omitted] ⏎  ⏎ Multiple ColBERT tests emit `UserWarning: To copy construct from a tensor` due to `torch.t …[truncated]

### L3-cf93c1a128  (L3, 2026-02-20, sha cf93c1a12849, PR #34570)
TITLE: [ROCm][AITER] Fix aiter paged_attention_v1 decode for sliding window and head_size < 64 (#34570)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+44/-1)
LABELS: rocm, ready, v1
BODY: Fixes a regression introduced by #34378 where sliding window models (e.g. Mixtral) and models with `head_size < 64` produce completely wrong decode outputs on the ROCm AITER backend. ⏎  ⏎ ## Root Cause ⏎  ⏎ PR #34378 removed the `unified_attention` triton kernel path for sliding window decode and routed all decode through `paged_attention_v1`'s ll4mi kernel. This kernel computes: ⏎  ⏎ ``` ⏎ VHELOOP = HEAD_SIZE / 16 / NWARPS ⏎ ``` ⏎  ⏎ On ROCm, `NWARPS = 4`, so for ` …[truncated]

### L3-54254f7a61  (L3, 2026-02-20, sha 54254f7a6155, PR #34599)
TITLE: [ROCm][CI] Fix spec decode logprobs flakiness and parametrize tree attention backends (#34599)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/sample/test_logprobs.py (+158/-134); tests/v1/spec_decode/test_tree_attention.py (+198/-17)
LABELS: rocm, speculative-decoding, ready, v1
BODY: This PR fixes `V1 Test others` test flakiness on ROCm. ⏎  ⏎ **`test_logprobs.py`** ⏎  ⏎ `test_spec_decode_logprobs` was intermittently failing on ROCm due to logprob differences between the base and speculative decode LLM that were misattributed to spec decode itself, due to ROCm skinny GEMM non-determinism --- the `wvSplitK` kernels in `gemm_kernels.cu` use persistent workgroup scheduling and wave-level shuffle reductions that produce different results  …[truncated]

### L3-2aab2bb543  (L3, 2026-02-20, sha 2aab2bb54366, PR #34541)
TITLE: [ROCM] Optimize ROCM_AITER_FA spec decode eagle performance (#34541)
SOURCES: path_core, subject_keyword, symbol_pickaxe, corpus:performance-pr-population
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+50/-2)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ This PR has dependency on #32877, which fixes accuracy issue of rocm aiter fa.  ⏎  ⏎ **Changes:** ⏎ - Change cudagraph_support to UNIFORM_BATCH to support decode cudagraph when num_speculative_tokens >0 ⏎ - Override build_for_drafting and remove cpu-gpu sync, so cudagraph can also work for draft model. ⏎  ⏎ **Trace** ⏎ Before ⏎ sync point at attention metadata build time ⏎ <img width="1019" height="749" alt="Screenshot 2026-01-28 at 11 41 14" src="https …[truncated]

### L3-272b535ab3  (L3, 2026-02-21, sha 272b535ab331, PR #34791)
TITLE: [Bugfix] Gate 256-bit instructions to CUDA 12.9+ (#34791)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: csrc/activation_kernels.cu (+4/-2)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎  ⏎ https://github.com/vllm-project/vllm/pull/33022 recently enables 256-bit instructions on vLLM activation kernels, but this is only available in CUDA 12.9+. With this change, building vLLM on CUDA 12.8 is now failing with the following error: ⏎  ⏎ ``` ⏎ FAILED: [code=255] CMakeFiles/_C.dir/csrc/activation_kernels.cu.o ⏎ /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1 -DPy_LIMITED_API=3 -DQU …[truncated]

### L3-b71fbd06e2  (L3, 2026-02-21, sha b71fbd06e215, PR #35036)
TITLE: [Model Runner V2] Support attention group (#35036)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/attn_utils.py (+55/-30); vllm/v1/worker/gpu/cudagraph_utils.py (+17/-7); vllm/v1/worker/gpu/model_runner.py (+5/-8); vllm/v1/worker/gpu/spec_decode/eagle/cudagraph.py (+5/-5); vllm/v1/worker/gpu/spec_decode/eagle/speculator.py (+5/-5)
LABELS: v1, nvidia
BODY: 

### L3-74d90b1ce4  (L3, 2026-02-21, sha 74d90b1ce49e, PR #34900)
TITLE: [Model Bash][DSR1] Add selective dynamic shape marking for CustomOp (#34900)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+4/-1); vllm/model_executor/custom_op.py (+42/-6)
LABELS: performance, ready, v1, deepseek, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ `CustomOp.maybe_compile` uses `dynamic=True`, making ALL tensor dimensions symbolic. For `_DecodeConcatQuantFP8`, model-constant dimensions (num_heads, kv_lora_rank, qk_rope_head_dim) become runtime kernel arguments with expensive integer divisions (~25 cycles each) instead of constexpr strength-reduced ops (~4 cycles). ⏎  ⏎ - Add optional `dynamic_arg_dims` to `CustomOp.register` to specify which argument dimensions are truly dynamic ⏎ - I …[truncated]

### L3-b7892a3bef  (L3, 2026-02-22, sha b7892a3beff0, PR #34478)
TITLE: [Model] Add NVFP4 quantization support for Step3.5-Flash (#34478)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_nvfp4_moe.py (+126/-0); vllm/model_executor/layers/fused_moe/cutlass_moe.py (+4/-0); vllm/model_executor/layers/fused_moe/fused_marlin_moe.py (+3/-0); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+0/-3); vllm/model_executor/models/step3p5.py (+71/-1)
LABELS: ready, nvidia, quantization
BODY: ## Summary ⏎  ⏎ Enable NVFP4 (FP4) quantized MoE inference for [stepfun-ai/Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash). This model uses a `swiglustep` activation (clipped SwiGLU with `limit=7.0`) on MoE layers 43-44, which was previously unsupported by all NVFP4 MoE backends. ⏎  ⏎ A working NVFP4 quantized checkpoint is available at [tacos4me/Step-3.5-Flash-NVFP4](https://huggingface.co/tacos4me/Step-3.5-Flash-NVFP4), quantized usin …[truncated]

### L3-b1b5e045df  (L3, 2026-02-23, sha b1b5e045dfdc, PR #35010)
TITLE: [XPU] allow TORCH_SDPA/TRITON_ATTN as XPU vit Backend (#35010)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/mm_encoder_attention.py (+11/-4); vllm/platforms/xpu.py (+1/-0)
LABELS: ready
BODY: ## Purpose ⏎ XPU should can use `TORCH_SDPA` as vit attn backend, controlled by `mm_encoder_attn_backend`. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result

### L3-a2ba6a5244  (L3, 2026-02-23, sha a2ba6a52443f, PR #34874)
TITLE: [Bugfix] Fix prefix caching for Mamba 'all' mode (Nemotron models) (#34874)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/attention/test_mamba_update_block_table.py (+145/-0); vllm/v1/attention/backends/mamba_attn.py (+21/-0)
LABELS: bug, ready, v1
ISSUES: #34865 [Bug]: Prefix caching failing for Nemotron models
BODY: ## Purpose ⏎  ⏎ Fixes #34865 ⏎  ⏎ Fix prefix caching producing NaN logprobs and garbage tokens for Nemotron hybrid models (`mamba_cache_mode="all"`) when prefix caching is enabled. ⏎  ⏎ ### Root Cause ⏎  ⏎ Nemotron hybrid models have multiple KV cache groups with identical `MambaSpec` (~12 Mamba groups). The metadata caching optimization in `_build_attn_group_metadata()` caches metadata by `(kv_cache_spec, type(builder))` key — the first Mamba group calls `build …[truncated]

### L3-a4bd661fb3  (L3, 2026-02-23, sha a4bd661fb33b, PR #34924)
TITLE: [Perf] Enable FlashInfer DeepGEMM swapAB on SM90 by default (#34924)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/compile/fusions_e2e/test_tp1_quant.py (+5/-0); vllm/envs.py (+2/-2)
LABELS: performance, ready, ready-run-all-tests
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ For some reason in https://github.com/vllm-project/vllm/pull/29213 this pathway wasn't enabled by default when it seems like a +10% improvement at low batch and even with DeepGEMM at large batch. ⏎  ⏎ Let's see if we break anything putting this on by default. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ Commands on 4xH100: ⏎ ``` ⏎ chg run -g=4 -- vllm serve MiniMaxAI/MiniMax-M2.5 -tp=4 --load-format=dummy --trust-remote-code --max-model-len auto ⏎ vllm bench …[truncated]

### L3-2ff4e51152  (L3, 2026-02-23, sha 2ff4e51152d8, PR #33443)
TITLE: [ROCm] AITER fused RoPE+KVCache (#33443)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.rocm.aiter_unified, L3.mla.common_v1, L3.dispatch.abstract_interface
FILES: vllm/model_executor/layers/attention/attention.py (+13/-15); vllm/model_executor/layers/attention/kv_transfer_utils.py (+2/-2); vllm/model_executor/layers/attention/mla_attention.py (+2/-2); vllm/v1/attention/backend.py (+27/-0); vllm/v1/attention/backends/rocm_aiter_fa.py (+37/-48); vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+40/-0); vllm/v1/attention/backends/rocm_attn.py (+44/-0); vllm/v1/attention/backends/triton_attn.py (+40/-0); tests/compile/passes/test_functionalization.py (+84/-12); tests/compile/passes/test_rope_kvcache_fusion.py (+325/-0); (+9 more)
LABELS: rocm, ready, torch.compile, v1, gpt-oss
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ Follow-up to #25954; adda a `fuse_rope_kvcache` Inductor pass to fuse QK RoPE and KVCache ops into the AITER fused kernel. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-14561fabfd  (L3, 2026-02-24, sha 14561fabfd39, PR #35127)
TITLE: [Perf] Optimize pooling model redundant copy, 1.8% throughput improvement (#35127)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_model_runner.py (+51/-15)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ There is a redundant full copy of pooling model output, then pick up the items that are finished. ⏎  ⏎ We can actually only copy the items that are finished to optimize ⏎  ⏎ ## Test ⏎  ⏎ ### Acc ⏎  ⏎ Covered in unit test ⏎  ⏎ ```bash ⏎ pytest -q \ ⏎   tests/v1/e2e/test_pooling_chunked_prefill.py::test_pooling_chunked_prefill \ ⏎   tests/v1/e2e/test_pooling_chunked_prefill.py::test_pooling_prefix_cache ⏎  ⏎ pytest -q \ ⏎   tests/entrypoints/pooling/basic/test_encode. …[truncated]

### L3-34ce0ffd1f  (L3, 2026-02-24, sha 34ce0ffd1f3c, PR #34434)
TITLE: [CPU][Perf] Accelerate Attention head for s390x using vector intrinsics (#34434)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+4/-1); csrc/cpu/cpu_attn.cpp (+4/-0); csrc/cpu/cpu_attn_impl.hpp (+1/-1); csrc/cpu/cpu_attn_vxe.hpp (+386/-0); csrc/cpu/generate_cpu_attn_dispatch.py (+26/-2); vllm/engine/arg_utils.py (+3/-4)
LABELS: ready, v1, cpu
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Accelerate paged attention GEMMs (QK, PV) on s390x with vector intrinsics ⏎ This PR accelerates `cpu_attention_with_kv_cache `on s390x by introducing VXE (Vector Extension Facility) optimized GEMM kernels for both QK and PV attention phases. The vectorized implementation significantly improves token generation throughput, enabling s390x to effectively utilize chunked prefill and prefix caching features. ⏎ ## Test Plan ⏎ 1. Run vllm bench wit …[truncated]

### L3-a0c7081695  (L3, 2026-02-24, sha a0c708169562, PR #35088)
TITLE: Fix fallback to default tactic (flashinfer autotuner) with trtllm_fp4_block_scale_moe (#35088)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+2/-2)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Purpose ⏎  ⏎ The **flashinfer autotuner** expects the first dimension of the MoE tensors to be num_tokens. ⏎  ⏎ The relevant code can be found in **flashinfer** function `get_trtllm_moe_sm100_module`: ⏎ ``` ⏎         # their first dimension is num_tokens which will be tuned ⏎         tuning_config_with_hidden_states_scales = TuningConfig( ⏎             dynamic_tensor_specs=( ⏎                 DynamicTensorSpec( ⏎                     (0, 1, 2, 3, 4, 5), ⏎            …[truncated]

### L3-9609b1f18d  (L3, 2026-02-24, sha 9609b1f18def, PR #35053)
TITLE: Integrate flashinfer mm_mxfp8 in ModelOpt MXFP8 (#35053)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+77/-0); vllm/model_executor/layers/quantization/modelopt.py (+50/-10); vllm/model_executor/layers/quantization/utils/mxfp8_utils.py (+103/-1)
LABELS: ready, nvidia, quantization
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ Follow up to PR: ⏎ https://github.com/vllm-project/vllm/pull/33786 ⏎  ⏎ Flashinfer version was recently updated in vLLM (to v0.6.4). ⏎  ⏎ A new MXFP8 GEMM (CUTLASS) is available - `mm_mxfp8`: ⏎ https://github.com/flashinfer-ai/flashinfer/pull/2464 ⏎  ⏎ This PR integrates this GEMM into vLLM (for ModelOpt MXFP8). ⏎  ⏎ ## Test Plan ⏎  ⏎ Use the following model for testing (used in other related PRs): ⏎ https://huggingface.co/nvidia/OpenMath2-Llama3.1-8B ⏎  ⏎ Compare …[truncated]

### L3-a87cc50859  (L3, 2026-02-24, sha a87cc508599d, PR #34281)
TITLE: [Attn,KV-cache] Use per-head scales in the attention selector (#34281)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.selector, L3.dispatch.abstract_interface
FILES: vllm/model_executor/layers/attention/attention.py (+9/-1); vllm/v1/attention/backend.py (+7/-1); vllm/v1/attention/backends/flash_attn.py (+5/-5); vllm/v1/attention/selector.py (+4/-0); tests/kernels/attention/test_attention_selector.py (+54/-0); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py (+1/-8)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1
BODY: As requested by @MatthewBonanni and @LucasWilkinson in https://github.com/vllm-project/vllm/pull/30141 attention backends should be filtered during backend selection based on whether they support per-head attention quantization scales. ⏎  ⏎ This enables early failure when a user attempts to load a model that requires per-head scales but no compatible attention backend is available.

### L3-f5972a872f  (L3, 2026-02-24, sha f5972a872fa3, PR #33726)
TITLE: [Model][Spec Decode] Nemotron-H MTP and Mamba Speculative Decoding Support (#33726)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/registry.py (+5/-0); tests/v1/attention/test_mamba_update_block_table.py (+7/-1); vllm/config/speculative.py (+17/-2); vllm/config/vllm.py (+9/-0); vllm/model_executor/layers/mamba/abstract.py (+0/-8); vllm/model_executor/layers/mamba/mamba_mixer.py (+2/-18); vllm/model_executor/layers/mamba/mamba_mixer2.py (+27/-19); vllm/model_executor/layers/mamba/mamba_utils.py (+2/-1); vllm/model_executor/layers/mamba/ops/causal_conv1d.py (+3/-1); vllm/model_executor/layers/mamba/short_conv.py (+2/-8); (+9 more)
LABELS: new-model, ready, v1
BODY: ## Purpose ⏎  ⏎ This PR adds support for MTP for the Nemotron-H model family, which [will be introduced with Nemotron V3 Super](https://arxiv.org/pdf/2512.20856). ⏎  ⏎ To facilitate this, we also implement speculative decoding support for the Mamba attention backends. Previously, mamba-style speculative decoding support was limited to Qwen3-Next. This PR attempts to implement the attention metadata in a simple and unified manner that does not introduce t …[truncated]

### L3-9fa5b25a23  (L3, 2026-02-24, sha 9fa5b25a238c, PR #35075)
TITLE: [Bug][DSV3.2] Always prepare metadata for DeepGEMM Sparse Attention (#35075)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+4/-2)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ When `VLLM_USE_DEEP_GEMM=0` is set when using `FLASHMLA_SPARSE`, `is_deep_gemm_supported` will be False but we will still call `fp8_paged_mqa_logits`. This causes a crash when using DeepSeek V3.2 since it will read `is_deep_gemm_supported` as False and the uninitialized/stale schedule metadata will be used by `fp8_paged_mqa_logits`. ⏎  ⏎ ## Testing ⏎  ⏎ ``` ⏎ VLLM_USE_DEEP_GEMM=0 vllm serve nvidia/DeepSeek-V3.2-NVFP4 -tp 4 ⏎ ``` ⏎  ⏎ previously crash …[truncated]

### L3-8ad54a991b  (L3, 2026-02-24, sha 8ad54a991b9f, PR #35042)
TITLE: [Platform] Add current_platform.num_compute_units interface (#35042)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.aiter_fa, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.flashmla_sparse, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/v1/attention/backends/mla/cutlass_mla.py (+2/-2); vllm/v1/attention/backends/mla/flashmla.py (+2/-2); vllm/v1/attention/backends/mla/flashmla_sparse.py (+2/-2); vllm/v1/attention/backends/mla/indexer.py (+2/-2); vllm/v1/attention/backends/rocm_aiter_fa.py (+2/-2); tests/kernels/attention/test_cutlass_mla_decode.py (+2/-2); tests/kernels/quantization/test_allspark_gemm.py (+2/-1); tests/kernels/quantization/test_rocm_skinny_gemms.py (+6/-6); vllm/model_executor/kernels/linear/mixed_precision/allspark.py (+2/-1); vllm/model_executor/kernels/linear/scaled_mm/rocm.py (+2/-2); (+14 more)
LABELS: performance, rocm, ready, v1, nvidia
BODY: ## Purpose ⏎ there are some `torch.cuda.get_device_properties().multi_processor_count` across vllm code base. we can unify it into `current_platform.num_compute_units`  interface to make it clean and extensible for non-cuda hardware like xpu and npu. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-35d44b4557  (L3, 2026-02-24, sha 35d44b455703, PR #34482)
TITLE: [XPU]Support CUDAGraph on XPU Platform (#34482)
SOURCES: symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/platforms/xpu.py (+33/-4); vllm/utils/torch_utils.py (+5/-0); vllm/v1/worker/xpu_model_runner.py (+7/-0)
LABELS: ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ Enable CUDAGraph functionality support on the XPU platform leveraging features available in the nightly PyTorch XPU build. ⏎ - Validated with torch==2.11.0.dev20260216+xpu from nightly channel and torch==2.11.0+xpu from test channel ⏎ - distributed is not supported ⏎ - For FLASH_ATTN backend, only PIECEWISE mode is supported ⏎ - For TRITON_ATTN backend, all modes are supported ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-cd43673668  (L3, 2026-02-24, sha cd4367366814, PR #34424)
TITLE:     [Perf] Optimize FP8 gemm of sm120. (#34424)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh (+133/-1)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Optimize FP8 gemm of sm120 at any input shape. ⏎  ⏎ ## Test Plan ⏎ benchmarks/kernels# python3 bench_fp8_gemm.py ⏎ run bench_fp8_gemm.py and compare the results with the patch or not. ⏎  ⏎ ## Test Result ⏎ **the current restut without the patch** ⏎ ``` ⏎ root@de-22309-vllm-0-15-1-5090-0211172246-79ccd9bbf9-6jfzz:/vllm/benchmarks/kernels# python3 bench_fp8_gemm.py ⏎ meta-llama/Llama-3.1-8B-Instruct, N=6144 K=4096, BF16 vs FP8 GEMMs TFLOP/s: ⏎ BF16 vs FP8 GEM …[truncated]

### L3-2465071510  (L3, 2026-02-24, sha 24650715105a, PR #31828)
TITLE: [Perf] Add opt-in SM100 Oink RMSNorm custom-op path (#31828)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/model_executor/test_oink_integration.py (+74/-0); vllm/_oink_ops.py (+96/-0); vllm/envs.py (+6/-0); vllm/model_executor/layers/layernorm.py (+155/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎   Add an opt-in integration path for external SM100 (Blackwell) RMSNorm kernels (CuTeDSL / Oink) without introducing a hard dependency in vLLM. ⏎  ⏎   When enabled, vLLM routes eligible CUDA RMSNorm calls to externally-registered torch.ops.oink.* kernels (and otherwise falls back to the existing vLLM CUDA implementation). This is primarily intended for torch.compile + CUDA graph execution. ⏎  ⏎   ## What This PR Changes ⏎  ⏎   - Adds env flag VLL …[truncated]

### L3-9571e99945  (L3, 2026-02-25, sha 9571e999451a, PR #35265)
TITLE: [ROCm][CI] Extending attention backend coverage for Eagle spec decode tests (#35265)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/engine.yaml (+1/-1); tests/utils.py (+51/-0); tests/v1/e2e/test_async_scheduling.py (+4/-0); tests/v1/e2e/test_spec_decode.py (+258/-149)
LABELS: rocm, ready, ci/build, v1
BODY: This PR fixes the attention backend handling in `test_eagle_correctness` for ROCm platforms: ⏎  ⏎ Previously, the test would skip entirely when running Llama-4-Scout with `FLASH_ATTN` on ROCm. Instead of skipping, we now fall back to the `FLEX_ATTENTION` backend explicitly, allowing the test to actually run on ROCm. ⏎  ⏎ Also previously, the test skipped DeepSeek models when using `ROCM_AITER_FA` on ROCm. Now we enable the AITER path (`VLLM_ROCM_USE_AITE …[truncated]

### L3-6831650c40  (L3, 2026-02-25, sha 6831650c40ac, PR #29941)
TITLE: [offloader] v2: Hide weight onloading latency via prefetching (#29941)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/scripts/scheduled_integration_test/deepseek_v2_lite_prefetch_offload.sh (+57/-0); .buildkite/test_areas/e2e_integration.yaml (+9/-0); tests/basic_correctness/test_prefetch_offload.py (+33/-0); vllm/compilation/cuda_graph.py (+14/-0); vllm/config/__init__.py (+11/-0); vllm/config/cache.py (+7/-9); vllm/config/offload.py (+153/-0); vllm/config/vllm.py (+7/-0); vllm/engine/arg_utils.py (+57/-8); vllm/entrypoints/llm.py (+21/-0); (+10 more)
LABELS: frontend, ready, ci/build, v1, deepseek, nvidia
BODY: ## Purpose ⏎  ⏎ This PR adds CPU weight offloader that hides weight onloading latency by prefetching weights.  This saves the performance cost of zero-copy UVA access. This technique was first developed in SGLang for GB200: https://lmsys.org/blog/2025-09-25-gb200-part-2/, and now adapted to support torch.compile and CUDA graph within vLLM in this PR. ⏎  ⏎ Also refactors the offloading to be extensible.  ⏎  ⏎ Demonstrated in the trace: ⏎ * H2D is for prefetchin …[truncated]

### L3-71dfce6aa6  (L3, 2026-02-26, sha 71dfce6aa6cc, PR #34109)
TITLE: [Kernel] Refactor FlashInfer allreduce for mnnvl backend (#34109)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: benchmarks/kernels/benchmark_device_communicators.py (+88/-25); benchmarks/kernels/benchmark_fused_collective.py (+114/-96); tests/compile/passes/distributed/test_fusion_all_reduce.py (+9/-1); vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+88/-56); vllm/distributed/device_communicators/cuda_communicator.py (+27/-1); vllm/distributed/device_communicators/flashinfer_all_reduce.py (+252/-0); vllm/envs.py (+14/-0)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ Flashinfer has a new allreduce API: https://github.com/flashinfer-ai/flashinfer/pull/2130 ⏎ and a `mnnvl` backend for allreduce optimized for multi-node NVLink cases (https://github.com/flashinfer-ai/flashinfer/pull/1213). ⏎ The `mnnvl` backend performs [same/better](https://github.com/flashinfer-ai/flashinfer/blob/d5eaa429b1c2c3cc51fe078028551fef10ca9cc9/flashinfer/comm/allreduce.py#L261-L267) than the old `trtllm` backend that vLLM curre …[truncated]

### L3-56a6371706  (L3, 2026-02-26, sha 56a6371706bc, PR #34687)
TITLE: [Update] Use FlashInfer fast_decode_plan directly instead of replication (#34687)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+83/-131); tests/kernels/attention/test_flashinfer.py (+203/-0)
LABELS: ready, v1, nvidia
BODY: Certain versions of FlashInfer (eg 0.6.0) change plan API which breaks fast_plan_decode every time. The main culprit in plan function in FI 0.6.0 was that it started accepting an extra `o_data_type` argument. Since current backend was mostly using positional arguments, those were causing errors in case of new ones being added. ⏎  ⏎ This PR updates FlashInfer to the latest version and fixes API breaking by: ⏎ - Changes API to call FI fast_decode_plan di …[truncated]

### L3-01923eec70  (L3, 2026-02-26, sha 01923eec7092, PR #30357)
TITLE: [ROCm][Quantization] GPT OSS Upstream MoE wmxfp4_afp8 with static scales (#30357)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py (+134/-3); vllm/model_executor/layers/fused_moe/layer.py (+8/-5); vllm/model_executor/layers/quantization/quark/quark_moe.py (+173/-29)
LABELS: rocm, ready, gpt-oss
BODY: # gpt-oss120b-w-mxfp4-a-fp8 ⏎  ⏎ ### server: ⏎ > HIP_VISIBLE_DEVICES=1 VLLM_DISABLE_COMPILE_CACHE=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_ROCM_USE_AITER_MHA=0 vllm serve /data/models/gpt-oss120b-w-mxfp4-a-fp8 --port 8000 --swap-space 64 --tensor-parallel-size 1 --no-enable-prefix-caching --seed 42    --enforce-eager ⏎  ⏎ ### client: ⏎ > curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ ⏎     "model …[truncated]

### L3-38c498b8e3  (L3, 2026-02-26, sha 38c498b8e3aa, PR #35121)
TITLE: [Performance] Cublas Bf16 Gate with Fp32 Output (#35121)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+2/-1); csrc/moe/moe_ops.h (+4/-0); csrc/moe/router_gemm.cu (+52/-0); csrc/moe/torch_bindings.cpp (+4/-0); vllm/_custom_ops.py (+17/-0); vllm/model_executor/layers/fused_moe/__init__.py (+2/-0); vllm/model_executor/layers/fused_moe/router/gate_linear.py (+117/-0); vllm/model_executor/models/deepseek_v2.py (+2/-70); vllm/model_executor/models/nemotron_h.py (+6/-9)
LABELS: performance, ready, ci/build, deepseek, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Introduces `GateLinear`, a specialized MoE gate linear layer with three-tier GEMM dispatch for router logits: ⏎  ⏎ 1. **Tier 1 — DSV3 specialized kernel** (SM90+, batch ≤ 16, supported dims): highest throughput for small batches ⏎ 2. **Tier 2 — cuBLAS bf16×bf16→fp32** (SM90+, bf16 weights, fp32 out_dtype): fp32-accumulate GEMM via `cublasGemmEx` with `CUBLAS_COMPUTE_32F` ⏎ 3. **Tier 3 — F.linear** via `ReplicatedLinear`: ultimate fallback ⏎  ⏎ Al …[truncated]

### L3-99c7892c5b  (L3, 2026-02-26, sha 99c7892c5bf2, PR #35330)
TITLE: [Perf] Optimize maxsim scores computation for pooling models, 13.9% E2E throughput improvement (#35330)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/entrypoints/pooling/score/test_utils.py (+39/-1); vllm/entrypoints/pooling/score/serving.py (+7/-9); vllm/entrypoints/pooling/score/utils.py (+77/-1)
LABELS: frontend, ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Optimize maxsim scores computation for pooling models ⏎  ⏎ Originally it is calculated in CPU, now we calculate it in GPU and using the batched version, so we get a lot of performance improvement ⏎  ⏎ ## Test ⏎  ⏎ ### Acc ⏎  ⏎ Covered in unit tests ⏎  ⏎ ```bash ⏎ tests/entrypoints/pooling/score/test_online_colbert.py::TestColBERTOnline::test_score ⏎ tests/entrypoints/pooling/score/test_online_colbert.py::TestColBERTOnline::test_rerank ⏎ tests/entrypoints/pooli …[truncated]

### L3-4fec53cfcb  (L3, 2026-02-26, sha 4fec53cfcb31, PR #34274)
TITLE: [CI] Actually run tests/kernels/quantization/test_block_fp8.py in CI (#34274)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+1/-1); .buildkite/test_areas/kernels.yaml (+1/-1); tests/kernels/quantization/test_block_fp8.py (+5/-7)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ We were only running the deepgemm tests in `tests/kernels/quantization/test_block_fp8.py` since we had not H100 runner running the rest of the file. I also fixed some outdated test cases after kernel updates like https://github.com/vllm-project/vllm/pull/28431 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-98217b09f9  (L3, 2026-02-26, sha 98217b09f9ce, PR #35422)
TITLE: [Performance] Extract KV cache update op from flashinfer forward (#35422)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+37/-25)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Extract KV cache update op from flashinfer similar to https://github.com/vllm-project/vllm/pull/25954 ⏎ This PR is a part of https://github.com/vllm-project/vllm/issues/32335 ⏎  ⏎ #### Eval (`Qwen/Qwen3-30B-A3B-FP8`): ⏎ main: ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8370|±  |0.0102| ⏎ |     |    …[truncated]

### L3-6283021142  (L3, 2026-02-26, sha 6283021142bb, PR #35430)
TITLE: [Bugfix] Fix KV Scale loading for MLA Models (#35430)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/modelopt.py (+2/-2)
LABELS: bug, ready
BODY: ## Purpose ⏎ Fixes KV Scale loading for MLA Models after `MLAAttention` was moved into its own base class. ⏎  ⏎ ## Test Plan ⏎ Manually run the Deepseek R1 FP4 model and ensure that kv scales are loaded by ensuring no warning is thrown.  ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-487e5c51f7  (L3, 2026-02-27, sha 487e5c51f727, PR #35424)
TITLE: [Bugfix] disable allreduce_rms_fusion by default when pp size > 1 (#35424)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+3/-0)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ When using tensor parallelism (TP) combined with pipeline parallelism (PP), flashinfer's trtllm_create_ipc_workspace_for_all_reduce_fusion function uses tp_rank as the CUDA device index (torch.device("cuda", tp_rank).index). However, in PP scenarios, tp_rank is relative to the TP group within the current PP stage (ranging from 0 to tp_size-1), rather than the actual GPU device index. ⏎  ⏎ For example, with TP=2, PP=2, and GPUs 4,5,6,7: ⏎  ⏎ P …[truncated]

### L3-07bdabef03  (L3, 2026-02-27, sha 07bdabef03c7, PR #33088)
TITLE: [Bugfix] Use 'sum' reduction instead of 'avg' in Async TP reduce-scatter (#33088)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/compilation/passes/fusion/collective_fusion.py (+3/-3)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ For Row Parallel Linear layers, the partial results from different TP ranks should be summed, not averaged. This PR changes the reduce_op to "sum" to ensure the output magnitude is mathematically correct. ⏎  ⏎ ## Test Plan && Test Result ⏎  ⏎ **Test Environment:** H800, TP=4 ⏎  ⏎ We tested the accuracy on MMLU and GSM8K. Strangely, on GSM8K, using "avg" outperformed "sum" and the default config (Async TP off). However, we should change "avg" to " …[truncated]

### L3-9c3fe9936b  (L3, 2026-02-27, sha 9c3fe9936b92, PR #34580)
TITLE: Flashinfer cuDNN backend for Qwen3 VL ViT attention (#34580)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/model_executor/layers/attention/mm_encoder_attention.py (+170/-2); vllm/platforms/cuda.py (+1/-0); vllm/v1/attention/ops/vit_attn_wrappers.py (+88/-0); tests/kernels/attention/test_mha_attn.py (+110/-0); vllm/model_executor/models/qwen2_5_vl.py (+3/-0); vllm/model_executor/models/qwen3_vl.py (+33/-19)
LABELS: performance, ready, ci/build, v1, multi-modality, qwen, nvidia
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ Enable by `--mm-encoder-attn-backend=FLASHINFER` ⏎  ⏎ ### Details ⏎ - Add `FLASHINFER` as a backend for `MMEncoderAttention` ⏎ - Add computation for `cu_seqlens` and `sequence_lengths` before calling vision blocks. This is due to cuDNN backend's requirement to pass batch offsets and actual sequence lengths ⏎ - Pad `cu_seqlens`, `sequence_lengths` and `max_seqlen` to avoid cuDNN frontend graph recompilation ⏎ - Only support Qwen3 VL ViT, Qwen2.5 VL …[truncated]

### L3-1f3dbd95fd  (L3, 2026-02-27, sha 1f3dbd95fd13, PR #35404)
TITLE: [Bugfix][Model] Fix gpt-oss batch invariance (#35404)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/linear.py (+1/-6); vllm/model_executor/models/gpt_oss.py (+13/-2)
LABELS: bug, ready, gpt-oss
BODY: ## Purpose ⏎ GPT-OSS is listed as verified in the [batch invariance doc](https://docs.vllm.ai/en/latest/features/batch_invariance/), but rerunning the provided tests on an H100 suggests it does not in fact work in all the claimed supported configurations: ⏎  ⏎ ``` ⏎ # VLLM_TEST_MODEL="openai/gpt-oss-20b" uv run pytest tests/v1/determinism/test_batch_invariance.py -k "bs1_vs_bsN and not MLA" ⏎ ... ⏎  ⏎ =========================================================== …[truncated]

### L3-9fa6c68fa6  (L3, 2026-02-27, sha 9fa6c68fa627, PR #35334)
TITLE: [ROCm] Enabling encoder and encoder-decoder on ROCm and AITER unified backends (#35334)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn, L3.rocm.aiter_unified
FILES: vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+31/-0); vllm/v1/attention/backends/rocm_attn.py (+73/-5); docs/design/attention_backends.md (+2/-2)
LABELS: documentation, rocm, ready, v1
BODY: Another prerequisite to #33271 ⏎ With this unit tests that use encoder attention such as test_run_batch.py and encoder-decoder models such as `examples/offline_inference/audio_language.py -m whisper` now work with ROCM_ATTN and ROCM_AITER_UNIFIED_ATTN

### L3-6d4f9d3ad5  (L3, 2026-02-27, sha 6d4f9d3ad5aa, PR #35082)
TITLE: [Bugfix] Fix DCP + FA3 crash due to missing num_splits in _forward_with_dcp (#35082)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+2/-0)
LABELS: bug, ready, v1
ISSUES: #35057 [Bug]: Qwen3.5 `scheduler_metadata must have shape (metadata_size)` with Decode Context Parallel (DCP)
BODY: ## Purpose ⏎  ⏎ Fix #35057. Qwen3.5-397B-A17B (and similar models) crash with `RuntimeError: scheduler_metadata must have shape (metadata_size)` when using `--decode-context-parallel-size 2` on H200 GPUs with FlashAttention 3. ⏎  ⏎ **Root cause**: `_forward_with_dcp()` calls `flash_attn_varlen_func` without the `num_splits` parameter (defaulting to 0), while `get_scheduler_metadata()` was called with `num_splits=32` during CUDA graph capture. FA3 compute …[truncated]

### L3-1e69c04887  (L3, 2026-02-28, sha 1e69c0488773, PR #35571)
TITLE: [ROCm][CI] Parametrize vision score tests across attention backends with per-backend tolerances (#35571)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/entrypoints/pooling/score/test_online_score_vision.py (+153/-40)
LABELS: rocm, ready
BODY: Parametrizes `test_online_score_vision.py` across all supported ROCm attention backends (`ROCM_ATTN`, `ROCM_AITER_FA`, `TRITON_ATTN`, `FLEX_ATTENTION`) using a module-scoped fixture with `--attention-config`. On non-ROCm platforms the backend list is empty, so existing behaviour is unchanged. This is in preparation for: ⏎ - https://github.com/vllm-project/vllm/pull/33271 ⏎  ⏎ ## Changes ⏎  ⏎ - **Backend parametrization**: the `server` fixture now accepts ` …[truncated]

### L3-7e08c22b8c  (L3, 2026-02-28, sha 7e08c22b8cb6, PR #35271)
TITLE: [Feat] Add CUDA torch fallbacks for fp8_mqa_logits/fp8_paged_mqa_logits_torch function (#35271)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+5/-2); vllm/model_executor/layers/sparse_attn_indexer.py (+50/-26); vllm/utils/deep_gemm.py (+121/-0)
LABELS: ready, v1, nvidia
ISSUES: #35021 [Bug]: GLM-5（Sparse MLA / DSA 模型）无法在 sm80 GPU（A100/A800）上运行 — DeepGemm 硬依赖无 fallback
DEEP_STUDY: deep-study: this PR was reverted by PR 37968 (confirmed_revert, reason=other)
BODY: ## Purpose ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/35021 ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ # pip uninstall deep_gemm ⏎ # vllm serve /mnt/data4/models/deepseek-ai/DeepSeek-V3___2 -tp=8  --tokenizer-mode deepseek_v32 --enable-auto-tool-choice --tool-call-parser deepseek_v32 --reasoning-parser deepseek_v3 --enforce-eager ⏎ ... ⏎ ... ⏎ (APIServer pid=1017794) INFO 02-25 16:22:09 [launcher.py:47] Route: /v1/messages, Methods: POST ⏎ (APIServer pid=1017794) INFO 02-25  …[truncated]

### L3-0edf101d2b  (L3, 2026-02-28, sha 0edf101d2b50, PR #35527)
TITLE: [ROCm] Add `stablelm` Head Size 80 To Supported Head Sizes For ROCM_ATTN (#35527)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn
FILES: vllm/v1/attention/backends/rocm_attn.py (+1/-1); docs/design/attention_backends.md (+1/-1)
LABELS: documentation, rocm, ready, v1
BODY: This is a prerequisite to https://github.com/vllm-project/vllm/pull/33271. ⏎  ⏎ Two tests fail when using the ROCM_ATTN backend instead of the current default TRITON_ATTN: ⏎ ``` ⏎ pytest -v -s models/language/generation/test_common.py::test_models[True-False-5-32-stabilityai/stablelm-3b-4e1t] ⏎ pytest -v -s models/quantization/test_gguf.py::test_models[1-5-32-bfloat16-model4] ⏎ ``` ⏎  ⏎ They fail with `ValueError: Head size 80 is not supported by RocmAttention`. …[truncated]

### L3-87d319c52f  (L3, 2026-03-01, sha 87d319c52f22, PR #34931)
TITLE: [AMD][CI] Support Triton attention with ExampleConnector (#34931)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_example_connector.py (+11/-7); tests/v1/kv_connector/unit/test_multi_connector.py (+0/-8); vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py (+17/-2)
LABELS: rocm, ready, v1, kv-connector
BODY: ## Purpose ⏎ To support Triton attention with `ExampleConnector`. ROCm uses [Triton attention](https://github.com/vllm-project/vllm/blob/f72061a19ae7fbb7f193c31f0abea355fab41892/vllm/v1/attention/backends/triton_attn.py#L295), which has a different kv cache layout than [Flash attention](https://github.com/vllm-project/vllm/blob/f72061a19ae7fbb7f193c31f0abea355fab41892/vllm/v1/attention/backends/flash_attn.py#L116), the Cuda default. [ExampleConnect …[truncated]

### L3-bbf81f9a92  (L3, 2026-03-01, sha bbf81f9a9284, PR #34798)
TITLE: [Mamba1] - Kernel Level Chunk Alignment for Prefix Caching (#34798)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/mamba/mamba_ssm/selective_scan.h (+3/-1); csrc/mamba/mamba_ssm/selective_scan_fwd.cu (+68/-35); csrc/ops.h (+3/-1); csrc/torch_bindings.cpp (+3/-1); tests/kernels/mamba/test_mamba_ssm.py (+4/-0); vllm/_custom_ops.py (+4/-0); vllm/model_executor/layers/mamba/mamba_mixer.py (+4/-0); vllm/model_executor/layers/mamba/ops/mamba_ssm.py (+4/-0); vllm/v1/attention/backends/mamba1_attn.py (+31/-2); vllm/v1/attention/backends/mamba2_attn.py (+6/-106); (+1 more)
LABELS: ready, v1
BODY: ## Purpose ⏎ The `selective_scan_fn` kernel processed tokens in fixed-size chunks `kChunkSize = kNThreads * kNItems` typically 2048, regardless of where the sequence started within a block. When chunked prefill split a request across scheduler iterations, the kernel would write state at positions that didn't align with block boundaries. ⏎  ⏎ Example of the bug: ⏎ Block size: 2048, seqlen: 3966 ⏎ Iteration 1: Process 1866 tokens → state written at position  …[truncated]

### L3-57a96e26c9  (L3, 2026-03-01, sha 57a96e26c913, PR #34832)
TITLE: Revert "[Bugfix] Disable TRTLLM attention with KV transfer enabled (#33192)" (#34832)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+0/-17)
LABELS: bug, ready, v1, nvidia
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 33192 reason=correctness_or_accuracy
BODY: ## Summary ⏎  ⏎ Reverts PR #33192 which introduced a query dtype mismatch regression in P/D disaggregation on Blackwell GPUs, while the bug it aimed to fix was already resolved. ⏎  ⏎ This reverts commit bbe0574d8e51c1c5935aeff9e92040c61d1d59c5. ⏎  ⏎ @NickLucche @robertgshaw2-redhat  ⏎  ⏎ ## Purpose ⏎ PR #33192 globally disabled TRTLLM attention when KV transfer is enabled, intending to fix an `is_strictly_contiguous(kv_cache_permute)` assertion failure. However: ⏎  ⏎  …[truncated]

### L3-cb21972a97  (L3, 2026-03-01, sha cb21972a976b, PR #34448)
TITLE: [Kernel] Integrate SM100 MXFP8 blockscaled grouped MM and quant kernels (#34448)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+27/-0); csrc/moe/mxfp8_moe/cutlass_mxfp8_grouped_mm.cu (+60/-0); csrc/moe/mxfp8_moe/cutlass_mxfp8_grouped_mm_functor.cuh (+141/-0); csrc/moe/mxfp8_moe/cutlass_mxfp8_grouped_mm_launcher.cuh (+179/-0); csrc/moe/mxfp8_moe/cutlass_mxfp8_grouped_mm_traits.cuh (+127/-0); csrc/moe/mxfp8_moe/mxfp8_experts_quant.cu (+60/-0); csrc/moe/mxfp8_moe/mxfp8_experts_quant.cuh (+414/-0); csrc/torch_bindings.cpp (+16/-0); tests/kernels/moe/test_cutlass_mxfp8_grouped_mm.py (+237/-0); vllm/_custom_ops.py (+70/-0)
LABELS: ready, ci/build, nvidia
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎ To enable serving MXFP8 MoE models, this PR integrates SGLang’s SM100 expert-specialization MXFP8 blockscaled grouped kernels into vLLM so they are built, registered, importable, and test-covered in the vLLM codebase.   ⏎ Source PR for adopted kernels: [sgl-project/sglang#13731](https://github.com/sgl-project/sglang/pull/13731). ⏎  ⏎ This PR: ⏎ - Adds the renamed kernels `cutlass_mxfp8_grouped_mm` and `mxfp8_experts_quant` to vLLM’s `_C` build …[truncated]

### L3-8b5014d3dd  (L3, 2026-03-01, sha 8b5014d3dd34, PR #32974)
TITLE: [Attention] FA4 integration (#32974)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.upstream_pip, L3.flash_attn.fork_build, L3.flash_attn.fa4_cutedsl, L3.flash_attn.fa_utils, L3.mla.common_v1
FILES: cmake/external_projects/vllm_flash_attn.cmake (+55/-30); requirements/cuda.txt (+4/-0); setup.py (+5/-0); vllm/config/attention.py (+2/-2); vllm/model_executor/layers/attention/mla_attention.py (+3/-1); vllm/model_executor/layers/attention/mm_encoder_attention.py (+3/-1); vllm/v1/attention/backends/fa_utils.py (+41/-7); vllm/v1/attention/backends/flash_attn.py (+9/-1); vllm/vllm_flash_attn/__init__.py (+24/-0); vllm/vllm_flash_attn/flash_attn_interface.py (+567/-0); (+5 more)
LABELS: documentation, ready, ci/build, v1, nvidia
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: Integrate upstream FA4; currently only faster for prefill and spec-decode. Follow up PRs will try to use this prefill for MLA and/or used in a composite backend (flashinfer decode, flash attn prefill) ⏎  ⏎ ``` ⏎ Results: ⏎                                              Attention Benchmark Results                                              ⏎ ┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓ …[truncated]

### L3-510bc9e1df  (L3, 2026-03-02, sha 510bc9e1df08, PR #35715)
TITLE: [Misc] Cleanup useless `current_platform` import (#35715)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.fa_utils, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/fa_utils.py (+0/-3); vllm/v1/attention/backends/flashinfer.py (+0/-2); vllm/compilation/passes/fusion/sequence_parallelism.py (+0/-4); vllm/config/model.py (+0/-6); vllm/distributed/parallel_state.py (+0/-2)
LABELS: ready, v1, nvidia
BODY: ## Purpose ⏎ Cleanup useless `current_platform` import ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-d9c7730877  (L3, 2026-03-02, sha d9c77308776b, PR #34627)
TITLE: [Performance] Extract kv update ops from MLA attention backends (#34627)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.dispatch.abstract_interface
FILES: vllm/config/compilation.py (+1/-0); vllm/model_executor/layers/attention/mla_attention.py (+83/-11); vllm/v1/attention/backend.py (+44/-0)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Extract KV cache update from MLA attention backends similar to https://github.com/vllm-project/vllm/pull/25954 ⏎  ⏎ This PR adapts some elements of https://github.com/vllm-project/vllm/pull/33658 ⏎  ⏎ ``` ⏎ lm-eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite --tasks gsm8k --batch_size auto ⏎  ⏎ this PR: ⏎  ⏎ deepgemm, CUDA: ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|------- …[truncated]

### L3-9433acb8df  (L3, 2026-03-02, sha 9433acb8dfda, PR #33736)
TITLE: [Spec Decode] Add hidden states extraction system (#33736)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: examples/offline_inference/extract_hidden_states.py (+58/-0); tests/models/registry.py (+5/-1); tests/v1/kv_connector/extract_hidden_states_integration/__init__.py (+0/-0); tests/v1/kv_connector/extract_hidden_states_integration/predictable_llama.py (+120/-0); tests/v1/kv_connector/extract_hidden_states_integration/test_extraction.py (+155/-0); tests/v1/spec_decode/test_extract_hidden_states.py (+346/-0); vllm/config/speculative.py (+80/-26); vllm/distributed/kv_events.py (+4/-0); vllm/distributed/kv_transfer/kv_connector/factory.py (+6/-0); vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py (+354/-0); (+6 more)
LABELS: documentation, new-model, speculative-decoding, ready, v1, llama, kv-connector
BODY: ## Purpose ⏎ In-tree implementation of hidden states extraction system described in #33118. ⏎  ⏎ FIX #33318 ⏎  ⏎ ## Components ⏎ ### Configs ⏎ - `vllm/config/speculative.py`: Add handling for `extract_hidden_states` spec method ⏎ - `vllm/transformers_utils/configs/extract_hidden_states.py`: ExtractHiddenStatesConfig def ⏎  ⏎ ### ExampleHiddenStatesConnector ⏎ - `vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py`: Connector definition ⏎ - `v …[truncated]

### L3-96fc09503a  (L3, 2026-03-02, sha 96fc09503a2d, PR #35793)
TITLE: [All Reduce] Change default backend of Flashinfer All Reduce to trtllm (#35793)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+5/-2)
LABELS: ready
BODY: https://github.com/vllm-project/vllm/pull/34109 introduced mnnvl backend and it is the default choice for 'auto' backend. ⏎ But it seems to have issues like https://github.com/vllm-project/vllm/issues/35772. ⏎ Forcing default backend to 'trtllm' for stability.

### L3-fa6a6be519  (L3, 2026-03-02, sha fa6a6be51978, PR #35741)
TITLE: [Bugfix] Fix missing sequence_lengths in qwen3_omni_moe_thinker (#35741)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_omni_moe_thinker.py (+17/-0)
LABELS: bug, ready, qwen
BODY: ## Purpose ⏎  ⏎ PR #34580 added a `sequence_lengths` parameter to ⏎ `Qwen2_5_VisionAttention.forward()` for the FlashInfer cuDNN backend and updated callers in `qwen3_vl.py` and `qwen2_5_vl.py`, but missed updating `qwen3_omni_moe_thinker.py`. This causes a TypeError crash when loading any Qwen3-Omni model: ⏎  ⏎ ``` ⏎ TypeError: Qwen2_5_VisionAttention.forward() missing 1 required ⏎ positional argument: 'sequence_lengths' ⏎ ``` ⏎  ⏎ Fix: ⏎ - Add `sequence_lengths` par …[truncated]

### L3-f44d1ddc8c  (L3, 2026-03-02, sha f44d1ddc8cf7, PR #35773)
TITLE: [BugFix] Fix cmake based incremental install (wrong vllm install dir) (#35773)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+8/-12)
LABELS: bug, ready, ci/build
BODY: alternative https://github.com/vllm-project/vllm/pull/35768, (broke by: https://github.com/vllm-project/vllm/commit/8b5014d3dd343736ccf3e26cd44a0bb7700d205c) when using  ⏎  ⏎ ``` ⏎ cmake --build --preset release --target install ⏎ ``` ⏎  ⏎ all components are installed at once, unlike setup.py based methods like `python setup.py build_ext --inplace` or `uv pip install -e .` that install components one-by-one. This led to double appending of `vllm` to the path

### L3-ada4f4fadd  (L3, 2026-03-02, sha ada4f4fadd20, PR #34119)
TITLE: [Fix Bug]`num_active_loras` always equals to zero  (#34119)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/lora/test_fused_moe_lora_kernel.py (+4/-2); tests/lora/test_gptoss_tp.py (+6/-1); vllm/lora/ops/triton_ops/fused_moe_lora_op.py (+8/-8); vllm/lora/ops/triton_ops/lora_expand_op.py (+3/-3); vllm/lora/ops/triton_ops/lora_kernel_metadata.py (+30/-12); vllm/lora/ops/triton_ops/lora_shrink_op.py (+6/-3); vllm/v1/worker/gpu_model_runner.py (+1/-0)
LABELS: bug, ready, v1, gpt-oss
BODY: ## Purpose ⏎ Before the fix, `num_active_loras` was a Python int. Tracing happens during the profile run (first forward pass to determine shapes). At that point, `num_active_loras = 0`. Dynamo sees a Python int and bakes it. Every subsequent call to the compiled function passes 0 to `num_active_loras`, regardless of what `self.num_active_loras` actually is. After the fix, `num_active_loras_cpu` is a torch.Tensor. Each call to the compiled function  …[truncated]

### L3-28ef9ba399  (L3, 2026-03-03, sha 28ef9ba39934, PR #34552)
TITLE: [BugFix] Add support for MTP num_speculative_tokens > 1 with sparse MLA (#34552)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+108/-32); vllm/v1/worker/gpu_model_runner.py (+13/-9); vllm/v1/worker/utils.py (+1/-1); tests/v1/spec_decode/test_eagle.py (+24/-29); tests/v1/spec_decode/test_mtp.py (+7/-3); vllm/model_executor/layers/sparse_attn_indexer.py (+6/-0); vllm/v1/spec_decode/eagle.py (+99/-121)
LABELS: bug, speculative-decoding, ready, v1
ISSUES: #34380 [Bug]: GLM-5 MTP crashes with num_speculative_tokens > 1
BODY: FIX: https://github.com/vllm-project/vllm/issues/34380 ⏎  ⏎ ## Tests ⏎  ⏎ ``` ⏎ vllm serve zai-org/GLM-5-FP8 \ ⏎     -tp 8 -ep \ ⏎     --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \ ⏎     --no-enable-prefix-caching ⏎ ``` ⏎ with ⏎ ``` ⏎ vllm bench serve \ ⏎     --dataset-name spec_bench \ ⏎     --dataset-path question.jsonl \ ⏎     --spec-bench-output-len 1024 \ ⏎     --seed 42 \ ⏎     --ignore-eos \ ⏎     --temperature 0 \ ⏎     --num-prompts 1000 \ ⏎     --req …[truncated]

### L3-9dd656f0ea  (L3, 2026-03-03, sha 9dd656f0ea06, PR #35270)
TITLE: [XPU][NIXL] Add GPUDirect RDMA support for XPU (#35270)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.xpu (+51/-3); vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py (+5/-2); vllm/platforms/xpu.py (+6/-0)
LABELS: ready, ci/build, kv-connector
BODY: ## Purpose ⏎  ⏎ Add GPUDirect RDMA support for XPU in NIXL connector. ⏎  ⏎ **Requirements**： ⏎ - UCX must include the fix from https://github.com/openucx/ucx/pull/11187. ⏎  ⏎ **Limitations**: ⏎ - Must be set UCX_NET_DEVICES manually for better performance until https://github.com/openucx/ucx/pull/11180 is merged. ⏎ - Currently ze-ipc is not supported. https://github.com/openucx/ucx/pull/11218 ⏎  ⏎ ## Test Plan ⏎  ⏎ Performance data of Llama3.3-70B int4 model with fp8 kvca …[truncated]

### L3-97995f6376  (L3, 2026-03-03, sha 97995f6376fd, PR #32564)
TITLE: [MoE Refactor] Create MK for TRTLLM Kernels (#32564)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: .buildkite/test_areas/kernels.yaml (+2/-1); benchmarks/kernels/benchmark_cutlass_moe_fp8.py (+17/-11); benchmarks/kernels/benchmark_cutlass_moe_nvfp4.py (+26/-9); benchmarks/kernels/benchmark_grouped_gemm_cutlass.py (+31/-19); benchmarks/kernels/benchmark_moe.py (+37/-17); docs/design/dbo.md (+1/-1); docs/design/fused_moe_modular_kernel.md (+52/-52); docs/design/moe_kernel_features.md (+7/-9); tests/evals/gsm8k/configs/moe-refactor/config-h100.txt (+0/-3); tests/kernels/moe/modular_kernel_tools/cli_args.py (+2/-2); (+68 more)
LABELS: documentation, performance, rocm, intel-gpu, ci/build, llama, gpt-oss, nvidia, ready-run-all-tests
BODY: ## Purpose ⏎ * convert TRTLLM Kernels into the modular kernel framework ⏎ * introduce the concept of a monolithic kernel to the mk framework ⏎ * remove HACK for the nvfp4 quant pre-AG ⏎  ⏎ MoE refactor CI ⏎  ⏎ --- ⏎ [details omitted]

### L3-e05cb3b93e  (L3, 2026-03-03, sha e05cb3b93e5d, PR #34986)
TITLE: TRTLLM gen-full attn Test Coverage (#34986)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_use_trtllm_attention.py (+196/-0); tests/v1/attention/test_trtllm_attention_integration.py (+360/-0)
LABELS: ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ Add test coverage for the TRTLLM gen-full attention pipeline in vLLM. The existing kernel-level tests (`test_flashinfer_trtllm_attention.py`) call the FlashInfer C++ kernels directly, bypassing the vLLM integration layer entirely. This leaves the decision logic, metadata construction, and forward dispatch at 0% coverage. ⏎  ⏎ This PR adds two new test files: ⏎  ⏎ 1. **Unit tests for attention decision functions** (`tests/kernels/attention/tes …[truncated]

### L3-3a8eef5869  (L3, 2026-03-03, sha 3a8eef5869b8, PR #35601)
TITLE: [ROCm][Bugfix]: Disable AITER Triton ROPE by default (#35601)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.platform.rocm_selection
FILES: vllm/envs.py (+3/-3); vllm/platforms/rocm.py (+2/-3)
LABELS: bug, rocm, ready
BODY: Partially reverts #35180 by disabling the AITER RoPE implementation by default. Experiments show that for e.g. Llama 3.1 8B on large batch size (3000) scenarios, the AITER RoPE impl is up to 25% slower on gfx942. ⏎  ⏎ However, we keep the `+rotary_embedding` custom op enabled by default on ROCm, since the vllm native custom op performs on par with the unfused sequence of torch native ops for RoPE. This also facilitates turning the RoPE+KVCache fusion …[truncated]

### L3-289fc48ab7  (L3, 2026-03-04, sha 289fc48ab73f, PR #35653)
TITLE: Use MMEncoderAttention (=use FlashAttention) instead of torch.sdpa in radio.py (#35653)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/radio.py (+36/-44)
LABELS: ready
BODY: Use attention masking mechanism via MMEncoderAttention (=use FlashAttention when possible) instead torch.sdpa in radio.py ⏎  ⏎ DocVQA_VAL and InfoVQA_VAL equivalent before and after.

### L3-f7da9cdffc  (L3, 2026-03-04, sha f7da9cdffca2, PR #35710)
TITLE: [ROCm][CI] Support async weight transfer example with platform-aware determinism (#35710)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+9/-3); examples/offline_inference/new_weight_syncing/rlhf_async_new_apis.py (+82/-30)
LABELS: documentation, rocm, ready, ci/build
BODY: Enables the async RL weight transfer example on ROCm by applying platform-specific settings to get as close to deterministic generation as possible. ⏎  ⏎ **Key changes:** ⏎  ⏎ - Select `TRITON_ATTN` on ROCm and `FLASH_ATTN` on NVIDIA ⏎ - Disable `VLLM_BATCH_INVARIANT` on ROCm, as enabling it causes an error due to incomplete backend support ⏎ - Apply conservative ROCm settings to minimize non-determinism: fixed seed, prefix caching disabled, sequential reque …[truncated]

### L3-6cb901093f  (L3, 2026-03-04, sha 6cb901093f3d, PR #34883)
TITLE: [Core] Add All-to-All communication backend for DCP  (#34883)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+33/-8); vllm/v1/attention/backends/flash_attn.py (+17/-3); vllm/v1/attention/backends/flashinfer.py (+30/-6); vllm/v1/attention/ops/dcp_alltoall.py (+363/-0); tests/distributed/test_dcp_a2a.py (+192/-0); vllm/config/parallel.py (+14/-0); vllm/config/vllm.py (+2/-0); vllm/engine/arg_utils.py (+7/-0)
LABELS: ready, v1, nvidia
ISSUES: #34018 [RFC]: Helix (Context + Tensor) Parallelism for Efficient Long-Context Decoding
BODY: ## Purpose ⏎  ⏎ Add All-to-All (A2A) as an alternative communication backend for Decode Context Parallel (DCP). ⏎  ⏎ The existing DCP implementation uses **AllGather + ReduceScatter (AG+RS)**, which requires 3 NCCL calls per attention layer (AllGather Q, AllGather K metadata, ReduceScatter output). This PR adds an **All-to-All** backend that exchanges partial attention outputs and their LSE values across ranks, then combines them locally with a Triton ke …[truncated]

### L3-7eca859110  (L3, 2026-03-04, sha 7eca85911072, PR #35240)
TITLE: Add PyTorch profiler schedule support with warmup/active iterations (#35240)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/profiler.py (+25/-2); vllm/profiler/wrapper.py (+61/-1)
LABELS: ready, meta-exported, fb-exported
BODY: Summary: ⏎ Add support for PyTorch profiler schedule-based profiling with warmup and active iterations. ⏎ This allows users to configure the profiler to discard initial warmup iterations (which may ⏎ contain JIT compilation noise) and only collect data during specific active iterations. ⏎  ⏎ Changes: ⏎ - Added `warmup_iterations` and `active_iterations` config fields to ProfilerConfig ⏎ - Added `_profiler_step()` hook in WorkerProfiler base class for schedule- …[truncated]

### L3-c8c3935b70  (L3, 2026-03-04, sha c8c3935b7013, PR #35656)
TITLE: [Bugfix][Model] Fix FP8 k_scale/v_scale not loaded for Qwen3-MoE (#35656)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/model_executor/test_weight_utils.py (+116/-0); vllm/model_executor/models/qwen3_moe.py (+6/-18); vllm/model_executor/models/qwen3_vl_moe.py (+7/-18)
LABELS: bug, ready, qwen
BODY: FP8 KV cache scales from llm-compressor checkpoints (e.g. `qkv_proj.k_scale`) were silently dropped during weight loading in Qwen3MoeModel, causing fallback to scale=1.0 and accuracy degradation. ⏎  ⏎ Root cause: scale names were caught by `ignore_suffixes` before `maybe_remap_kv_scale_name` could remap them to `attn.k_scale`. ⏎  ⏎ Fix: add early scale remapping (matching Llama's approach) between the `get_cache_scale` check and the stacked-params loop,  …[truncated]

### L3-18e01a0a10  (L3, 2026-03-04, sha 18e01a0a10e3, PR #35738)
TITLE: [Misc] Add `--attention-backend auto` option (#35738)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/attention.py (+8/-2); vllm/engine/arg_utils.py (+4/-7); tests/kernels/attention/test_attention_selector.py (+42/-0)
LABELS: ready
BODY: Simple UX change to support "auto" option as *alias* to not providing the `--attention-backend` flag at all. ⏎ Example: ⏎ ``` ⏎ vllm serve openai/whisper-large-v3-turbo --attention-backend auto ⏎ ``` ⏎  ⏎ I believe this should bring a bit more consistency across how similar flags are used in vLLM, eg: ⏎  - `--moe-backend` ⏎  - `--max-model-len`  ⏎  ⏎ I also find it personally useful in just scripts where I have `--attention-backend {{BACKEND}}` hardcoded and quickly …[truncated]

### L3-a8f66cbde8  (L3, 2026-03-04, sha a8f66cbde878, PR #35984)
TITLE: [XPU] bump vllm-xpu-kernels to v0.1.3 (#35984)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/scripts/hardware_ci/run-xpu-test.sh (+1/-1); requirements/xpu.txt (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ bump vllm-xpu-kernels dependency to v0.1.3, major feature: decode attention have some performance improvement, ⏎ see https://github.com/vllm-project/vllm-xpu-kernels/releases/tag/v0.1.3 ⏎  ⏎ known issue: ⏎ attention kernel support block_size 64/128 only. so ignore some ut use block_size=16 ⏎  ⏎ ## Test Plan ⏎ CI ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-17dc9c7fc9  (L3, 2026-03-04, sha 17dc9c7fc945, PR #34950)
TITLE: [CI] Bump `mypy` version (#34950)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+2/-0); .pre-commit-config.yaml (+1/-1); tests/kernels/core/test_pos_encoding.py (+2/-5); tests/kernels/core/test_rotary_embedding.py (+2/-2); tests/kernels/mamba/test_mamba_ssm.py (+7/-7); tests/kernels/quantization/test_fp8_quant.py (+3/-3); vllm/config/parallel.py (+13/-3); vllm/distributed/elastic_ep/elastic_state.py (+25/-11); vllm/distributed/kv_transfer/kv_connector/utils.py (+1/-0); vllm/v1/attention/backends/gdn_attn.py (+12/-11); (+3 more)
LABELS: ready, v1, kv-connector, nvidia
BODY: - Bump `mypy` from 1.11.1 (Jul 30, 2024) to 1.15.0 (Feb 5, 2025) to get better checking and performance ⏎ - 1.13.0 added a `faster-cache` optional extra to improve caching performance. We add this in case it helps ⏎ - We don't bump all the way because newer versions catch things that older versions missed. So we step a few versions at a time to keep these PRs smaller and more manageable

### L3-f600d5192e  (L3, 2026-03-04, sha f600d5192e28, PR #35849)
TITLE: [Bugfix] Fix score layer quantization for sequence classification models  - Qwen3 (VL) Reranker (#35849)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/adapters.py (+29/-7)
LABELS: bug, ready, qwen
BODY: ## Purpose ⏎  ⏎ Fix FP8/NVFP4 quantization bug for sequence classification models (e.g., Qwen3 Reranker). ⏎  ⏎ The `score` layer created by `as_seq_cls_model()` is a dynamic classification head (`output_dim=1`) with no checkpoint weights. Passing `quant_config` to this layer causes: ⏎ - **FP8**: all scores return 0.0 (Marlin tile alignment violation — `output_dim=1` not divisible by `tile_size=64`) ⏎ - **NVFP4**: model load crash (only `weight_packed` regist …[truncated]

### L3-d7adcadb9b  (L3, 2026-03-04, sha d7adcadb9bf4, PR #36017)
TITLE: [Bugfix] Fix passing of activation_type to trtllm fused MoE NVFP4 and FP8 (#36017)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+1/-2); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+1/-0)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎ Fix passing activation_type of type `int` to: ⏎ * `flashinfer.fused_moe.trtllm_fp8_per_tensor_scale_moe` ⏎ * `flashinfer.fused_moe.trtllm_fp4_block_scale_moe` ⏎  ⏎ Without this fix, the NVFP4 flow doesn't pass the activation, so the default of `Swiglu` is always used. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-86483ca774  (L3, 2026-03-05, sha 86483ca7749b, PR #36146)
TITLE: [Bugfix] Disable FlashInfer TRTLLM BF16 path for non-gated MoE (#36146)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py (+3/-3)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎ PR #32564 split the FlashInfer TRTLLM MoE code into precision-specific files (`trtllm_fp4_moe.py`, `trtllm_fp8_moe.py`), each with their own copies of `_supports_no_act_and_mul()` and `_supports_activation()`. The original `flashinfer_trtllm_moe.py` was left handling only the BF16 path, but its helper values were not updated to reflect this narrower scope. ⏎  ⏎ PR #33506 added non-gated MoE support (for FP8/NVFP4) by modifying `_supports_n …[truncated]

### L3-8c760b6ab6  (L3, 2026-03-05, sha 8c760b6ab699, PR #35246)
TITLE: [ROCm] Refactor ROCm attention backend selection logic (#35246)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa, L3.mla.rocm_aiter_sparse, L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+136/-104); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+14/-4); vllm/v1/attention/backends/mla/triton_mla.py (+5/-0); vllm/v1/attention/backends/rocm_aiter_fa.py (+10/-0); docs/design/attention_backends.md (+1/-1); tests/kernels/attention/test_attention_selector.py (+4/-5)
LABELS: documentation, rocm, ready, v1
BODY: ## Purpose ⏎ This PR refactors the attention backend selection logic for ROCm to follow the same style as CUDA. There shouldn't be any changes in behavior, but this set's the stage for us to add more sophisticated attention backend selection logic in the future. ⏎  ⏎ ## Test Plan ⏎ This PR doesn't change any behavior so the existing CI should be sufficient. ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-a57c877f18  (L3, 2026-03-05, sha a57c877f1818, PR #36059)
TITLE: [BugFix] Fallback from FA4->FA2 for Batch Invariance (#36059)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fa_utils
FILES: vllm/v1/attention/backends/fa_utils.py (+11/-0)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎ FA4 is now the default FA version for Blackwell. However, it is not batch invariant. Thus, when batch invariance is enabled, fallback to FA2. ⏎  ⏎ ## Test Plan ⏎ Run batch invariant tests before and after the changes in this PR on a B200: ⏎  ⏎ ``` ⏎ VLLM_LOGGING_LEVEL=INFO CUDA_VISIBLE_DEVICES=4 VLLM_TEST_SEED=42 pytest tests/v1/determinism/test_batch_invariance.py::test_logprobs_bitwise_batch_invariance_bs1_vs_bsN[FLASH_ATTN] -s ⏎ ``` ⏎  ⏎ ## Test Resu …[truncated]

### L3-6a895197fa  (L3, 2026-03-05, sha 6a895197fafa, PR #34934)
TITLE: [Bugfix][CI] fix typos (#34934)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.aiter_fa, L3.mla.common_v1, L3.mla.flashmla_sparse
FILES: .buildkite/scripts/upload-nightly-wheels.sh (+1/-1); .buildkite/test-amd.yaml (+2/-2); .pre-commit-config.yaml (+1/-1); benchmarks/attention_benchmarks/common.py (+1/-1); benchmarks/kernels/benchmark_2d_silu_mul_fp8_quant.py (+1/-1); csrc/cpu/cpu_attn_amx.hpp (+1/-1); csrc/cpu/torch_bindings.cpp (+1/-1); csrc/moe/moe_align_sum_kernels.cu (+2/-2); csrc/quantization/activation_kernels.cu (+1/-1); csrc/rocm/skinny_gemms.cu (+1/-1); (+88 more)
LABELS: bug, documentation, performance, rocm, structured-output, frontend, ready, ci/build, v1, multi-modality
BODY: ## Purpose ⏎ Fix CI tool `typos` ⏎  ⏎ https://github.com/vllm-project/vllm/blob/76df6072ff4829980ad71764191fc970a873275a/pyproject.toml#L147-L149 ⏎  ⏎ The regular expression `".*[UE4M3|ue4m3].*"` match in the original code is incorrect. `[]` should be changed to `()`. The original implementation would lead to many spelling errors being missed. At the same time, I updated the version of the typos tool used in the vllm project and found even more spelling er …[truncated]

### L3-d8839ef7d9  (L3, 2026-03-05, sha d8839ef7d964, PR #36078)
TITLE: [XPU] Enable ModelRunnerV2 on XPU (#36078)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/xpu_model_runner.py (+18/-0); vllm/v1/worker/xpu_worker.py (+3/-2)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ To enable ModelRunnerV2 on XPU. It depends on torch==2.11.0+xpu and triton-xpu==3.7.0 ⏎  ⏎ can install via `pip install --force-reinstall torch torchvision torchaudio triton-xpu --index-url https://download.pytorch.org/whl/test/xpu` ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-c5362c739f  (L3, 2026-03-05, sha c5362c739fb3, PR #36185)
TITLE: Reenable features for ROCm attention backends (#36185)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.rocm.aiter_unified, L3.mla.rocm_aiter, L3.mla.rocm_aiter_sparse, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+1/-1); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+10/-0); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+11/-10); vllm/v1/attention/backends/mla/triton_mla.py (+0/-5); vllm/v1/attention/backends/rocm_aiter_fa.py (+8/-0); vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+17/-1); vllm/v1/attention/backends/rocm_attn.py (+14/-11); docs/design/attention_backends.md (+5/-5)
LABELS: documentation, rocm, ready, v1
BODY: ## Purpose ⏎ #35246 refactored the ROCm attention backend selection logic to validate the backends, but did not specify the feature support e.g. fp8 kvcache on multiple backends. This PR fixes multiple related errors e.g. `Selected backend AttentionBackendEnum.ROCM_AITER_MLA is not valid for this configuration. Reason: ['kv_cache_dtype not supported']`. ⏎  ⏎ cc @SageMoore @gshtras @tjtanaa  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]
