### L3-0191354827  (L3, 2026-05-18, sha 019135482756, PR #42885)
TITLE: [Perf][MLA] Enable FULL cudagraph capture for TRITON_MLA decode (#42885)
SOURCES: path_core, subject_keyword, release_notes, corpus:kernel-correctness-cases(introducing), corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/triton_mla.py (+10/-0)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:2c17d33f42 (fix PR 47144) || deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ `MLACommonMetadataBuilder` defaults `_cudagraph_support` to `NEVER`, which forces `PIECEWISE` mode and leaves `unified_mla_attention_with_output` outside the captured FULL graph — incurring per-layer Python dispatch on every decode step. ⏎  ⏎ This PR adds `TritonMLAMetadataBuilder` advertising `AttentionCGSupport.UNIFORM_BATCH`, mirroring `FlashInferMLAMetadataBuilder` and `FlashAttnMLAMetadataBuilder`. Capture uses worst-case `max_se …[truncated]

### L3-737bfa3a43  (L3, 2026-05-18, sha 737bfa3a43ce, PR #41233)
TITLE: [Bugfix][Hybrid][NemotronH] Fix mamba_cache_mode=all + speculative decoding crash (#41233)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+4/-2); tests/v1/attention/test_mamba_update_block_table.py (+271/-4); tests/v1/e2e/general/test_mamba_prefix_cache.py (+14/-6); vllm/model_executor/layers/mamba/mamba_mixer2.py (+38/-12); vllm/model_executor/models/config.py (+9/-20); vllm/v1/attention/backends/mamba2_attn.py (+3/-1); vllm/v1/attention/backends/mamba_attn.py (+77/-9); vllm/v1/kv_cache_interface.py (+3/-1); vllm/v1/worker/gpu_model_runner.py (+41/-11); vllm/v1/worker/mamba_utils.py (+108/-51)
LABELS: bug, ready, v1
DEEP_STUDY: deep-study correctness case vllm:737bfa3a43: class=shape_alignment_edge; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎  ⏎ Fixes the chain of three bugs reported in #39809 that crash NemotronH (and other hybrid Mamba2) models when prefix caching `all` mode is combined with speculative decoding. ⏎  ⏎ This PR also reverts the temporary workaround merged in #40454. ⏎  ⏎ ### Evaluation ⏎  ⏎ Verified end-to-end with `tests/evals/gsm8k/gsm8k_eval.py` against NVIDIA Nemotron 3 Super using a shared few-shot prefix to exercise the cache. The full `(PC-off, PC-align, P …[truncated]

### L3-47829b1159  (L3, 2026-05-18, sha 47829b115933, PR #42430)
TITLE: [Bugfix] mamba: run single-token extends as decodes (#42430)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/attention/test_mamba_update_block_table.py (+33/-20); tests/v1/attention/utils.py (+40/-1); tests/v1/kv_connector/unit/test_nixl_connector_hma.py (+25/-0); vllm/v1/attention/backends/mamba_attn.py (+22/-0)
LABELS: bug, ready, v1, kv-connector
BODY: NIXL Mamba disagg has the P-side compute prompt tokens `0..N-1`; the D-side computes token `N` from `h(N-1)` to advance Mamba state correctly. ⏎  ⏎ That D-side 1-token row is still marked `is_prefilling` by `num_computed_tokens < num_prompt_tokens`, but there is no reason why it cannot run as a decode. ⏎ In *uniform* 1-token batches, FULL decode CG is selected, while *Mamba still treats the row as prefill*.  ⏎ Mamba prefill is not compatible with FUL …[truncated]

### L3-37ece593c1  (L3, 2026-05-18, sha 37ece593c105, PR #42774)
TITLE: [Perf] Padded nvfp4 quant kernel to remove additional copy, 2.4%~5.7% e2e performance improvement (#42774)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu (+30/-24); tests/kernels/quantization/test_nvfp4_quant.py (+59/-0); vllm/_custom_ops.py (+24/-5); vllm/model_executor/kernels/linear/nvfp4/cutlass.py (+2/-5); vllm/model_executor/kernels/linear/nvfp4/flashinfer.py (+4/-9)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Padded nvfp4 quant kernel to remove additional copy, 15~40% kernel performance improvement, around 2.4%~5.7% e2e improvement ⏎  ⏎ ## Test ⏎  ⏎ Correctness covered by unit tests ⏎  ⏎ ### E2E perf ⏎  ⏎ ```bash ⏎ CUDA_VISIBLE_DEVICES=0,1,2,3 \ ⏎ vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \ ⏎   --quantization modelopt_fp4 \ ⏎   --tensor-parallel-size 4 \ ⏎   --linear-backend flashinfer_cutlass \ ⏎   --host 127.0.0.1 \ ⏎   --port 8000 \ ⏎    …[truncated]

### L3-f85c76d701  (L3, 2026-05-18, sha f85c76d701fc, PR #42991)
TITLE: [CI/Build] Bump nvidia-cutlass-dsl to 4.5.1 (#42991)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: requirements/cuda.txt (+1/-1)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ `cute-dsl 4.5.1` fixes FI Blackwell GDN failures. With the previous pin `==4.5.0` (introduced in #42342), FlashInfer's Blackwell GDN prefill kernel hits a JIT compile ICE during warmup. Bumping the pin to `==4.5.1` unblocks GDN without any source change on the FlashInfer side. ⏎  ⏎ ## Test Result ⏎  ⏎ Hardware: 1xGB200 ⏎  ⏎ ### Functional ⏎  ⏎ **FlashInfer GDN unit tests**: ⏎  ⏎ ``` ⏎ pytest -q tests/gdn/test_prefill_delta_rule.py ⏎ ``` ⏎  ⏎ | `n …[truncated]

### L3-67f58ce23f  (L3, 2026-05-18, sha 67f58ce23f46, PR #42930)
TITLE: [Bugfix] Fix DSV4 MTP after ROCm mHC integration (#42930)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/deepseek_v4.py (+12/-8); vllm/model_executor/models/deepseek_v4_mtp.py (+5/-4)
LABELS: bug, rocm, ready, deepseek
BODY: ## Purpose ⏎  ⏎ This fixes the DSV4 MTP path (again) after #41946 split the decoder layer into CUDA/ROCm helpers plus a dispatching `forward()`. ⏎  ⏎ I previously attempted to fix this in #42320, but that PR was written against the older decoder-layer shape. After it was rebased, the HC-state defaults landed on `_forward_cuda()` but not on the public `DeepseekV4DecoderLayer.forward()` method that `torch.compile` binds to, so DSV4 MTP still failed wit …[truncated]

### L3-965d076148  (L3, 2026-05-18, sha 965d07614832, PR #42740)
TITLE: [CPU] Specify required KV cache layout for CPU attention backend (#42740)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+5/-0)
LABELS: ready, v1, cpu, verified
BODY: Summary ⏎ Add get_required_kv_cache_layout() to CPUAttentionBackend to explicitly declare the required KV cache layout as "HND" (Head, N-tokens/block_size, Dim). ⏎  ⏎ Motivation ⏎ The CPU attention backend's get_kv_cache_shape already returns the shape (2, num_blocks, num_kv_heads, block_size, head_size), which corresponds to the HND layout. However, it did not override get_required_kv_cache_layout() from the base class, which returns None. When the  …[truncated]

### L3-287471b994  (L3, 2026-05-18, sha 287471b99442, PR #43004)
TITLE: [Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N]  (#43004)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/test_deepseek_v4_mega_moe.py (+1/-1); vllm/model_executor/layers/quantization/__init__.py (+1/-1); vllm/model_executor/models/registry.py (+26/-5); vllm/models/__init__.py (+2/-0); vllm/models/deepseek_v4/__init__.py (+30/-0); vllm/models/deepseek_v4/amd/__init__.py (+2/-0); vllm/models/deepseek_v4/amd/deepseek_v4.py (+1/-0); vllm/models/deepseek_v4/amd/deepseek_v4_mtp.py (+1/-0); vllm/models/deepseek_v4/nvidia/__init__.py (+2/-0); vllm/models/deepseek_v4/nvidia/deepseek_v4.py (+11/-114); (+2 more)
LABELS: new-model, ready, deepseek
BODY: This PR moves `vllm/model_executor/models/deepseek_v4.py` and its MTP to `vllm/models/deepseek_v4/nvidia` and `vllm/models/deepseek_v4/amd`. For now, the amd directory just uses symlinks to the nvidia directory, to keep the change minimal. ⏎  ⏎ It's the first step to implement #42770

### L3-da03e549b3  (L3, 2026-05-18, sha da03e549b346, PR #42537)
TITLE: [UX] Add a persistent cache for FlashInfer autotuning (#42537)
SOURCES: path_integration+keyword, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+6/-0); docs/usage/security.md (+1/-0); tests/model_executor/test_flashinfer_autotune_cache.py (+60/-0); vllm/model_executor/warmup/kernel_warmup.py (+36/-16)
LABELS: documentation, ready, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ This PR persists FlashInfer autotune results to per-vLLM-config JSON files so later startups can reuse previously tuned FlashInfer GEMM/MoE tactics instead of profiling the same shapes again during warmup. ⏎  ⏎ By default, cache files are stored under `$VLLM_CACHE_ROOT/flashinfer_autotune_cache/<flashinfer-version>/<arch>/<cache-hash>/autotune_configs.json`. Users can override the cache directory with `VLLM_FLASHINFER_AUTOTUNE_CACHE_D …[truncated]

### L3-fba010dd74  (L3, 2026-05-18, sha fba010dd74e2, PR #42766)
TITLE: [Bugfix][MRV2] Fix KVCache tensor explicit `kernel_block_size` dim (#42766)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/test_config.py (+0/-7); vllm/config/vllm.py (+0/-4); vllm/v1/worker/gpu/attn_utils.py (+49/-12); vllm/v1/worker/gpu/block_table.py (+12/-2); vllm/v1/worker/gpu/model_runner.py (+6/-4); vllm/v1/worker/gpu/spec_decode/eagle/speculator.py (+1/-1)
LABELS: bug, ready, v1
ISSUES: #42846 [Bug][CI] NIXL + FlashInfer fails with Qwen3 MRV2 and --block-size 128
BODY: With MRv2 being on by default for dense Qwen models (ie `Qwen3-0.6B`), we found that there's some discrepancy in how KV cache tensor are exposed to connectors through the `register_kv_caches` API when akernel and logical block_size "don't match" eg: ⏎ ``` ⏎ num_blocks, 2, 128, ...      <==MRV2 ⏎ num_blocks*2, 2, 64, ...    <==MRV1 ⏎ ``` ⏎ assuming block_size=128 and kernel_block_size=64 on FI backend ⏎  ⏎ see https://buildkite.com/vllm/ci/builds/66356/c …[truncated]

### L3-87b08c5f64  (L3, 2026-05-18, sha 87b08c5f6460, PR #43039)
TITLE: [Model Refactoring] Move DeepSeek V4 layers to `models/deepseek_v4/` [2/N] (#43039)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py (+1/-3); .github/CODEOWNERS (+1/-2); vllm/models/deepseek_v4/attention.py (+1/-1); vllm/models/deepseek_v4/compressor.py (+0/-0); vllm/models/deepseek_v4/nvidia/deepseek_v4.py (+5/-5)
LABELS: rocm, ci/build, v1, deepseek
BODY: This PR simply moves the DeepSeek V4-related files in `vllm/model_executor/layers/` to `vllm/models/deepseek_v4/`, so that they are more consolidated

### L3-3ca8db2ef8  (L3, 2026-05-18, sha 3ca8db2ef88e, PR #42899)
TITLE: add cutedsl dsv4 indexer fp8 kernel (#42899)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py (+33/-0); vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py (+37/-20); vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py (+311/-37); tests/kernels/test_fused_indexer_q_rope_quant.py (+30/-3)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎ - Adds a CuTe DSL implementation of the FP8 `fused_indexer_q_rope_quant` op. A similar change was made in https://github.com/vllm-project/vllm/pull/41428, but only improved the fp4 path. ⏎ - Logic shared between the fp8 and fp4 paths is added to a `IndexerQRopeQuantKernel` base class ⏎  ⏎ ## Kernel Benchmarks ⏎ ``` ⏎ num_tokens             triton (us)             cute   (us)   speedup ⏎          1                   8.832                   8 …[truncated]

### L3-b14be81c1f  (L3, 2026-05-19, sha b14be81c1f63, PR #43073)
TITLE: [Model Refactoring] Move deepseek_v4_ops to models/deepseek_v4 [3/N] (#43073)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py (+1/-1); .github/CODEOWNERS (+0/-1); tests/kernels/core/test_fused_q_kv_rmsnorm.py (+1/-1); tests/kernels/test_compressor_kv_cache.py (+2/-2); tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py (+1/-1); tests/kernels/test_fused_indexer_q_rope_quant.py (+2/-4); tests/kernels/test_fused_inv_rope_fp8_quant.py (+1/-1); vllm/models/deepseek_v4/attention.py (+3/-3); vllm/models/deepseek_v4/common/__init__.py (+2/-0); vllm/models/deepseek_v4/common/ops/__init__.py (+0/-0); (+10 more)
LABELS: rocm, ci/build, v1, deepseek
BODY: Move `attention/ops/deepseek_v4_ops/` to `vllm/models/deepseek_v4/common/ops` or `nvidia/ops`.

### L3-07beaed842  (L3, 2026-05-19, sha 07beaed8422d, PR #43077)
TITLE: [Model Refactoring] Rename deepseek_v4.py to model.py [4/N] (#43077)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/test_deepseek_v4_mega_moe.py (+1/-1); vllm/models/deepseek_v4/__init__.py (+4/-4); vllm/models/deepseek_v4/amd/deepseek_v4.py (+0/-1); vllm/models/deepseek_v4/amd/deepseek_v4_mtp.py (+0/-1); vllm/models/deepseek_v4/amd/model.py (+1/-0); vllm/models/deepseek_v4/amd/mtp.py (+1/-0); vllm/models/deepseek_v4/nvidia/model.py (+0/-0); vllm/models/deepseek_v4/nvidia/mtp.py (+1/-1)
LABELS: deepseek
BODY: Rename `deepseek_v4.py` to `model.py` and `deepseek_v4_mtp.py` to `mtp.py`.

### L3-d247a931cc  (L3, 2026-05-19, sha d247a931cc25, PR #42080)
TITLE: [feat] Add FP8 per-tensor Q scale support to Triton attention backend (#42080)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.triton.unified_attention, L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+17/-8); vllm/v1/attention/ops/triton_unified_attention.py (+18/-6); tests/kernels/attention/test_triton_unified_attention.py (+131/-8)
LABELS: bug, ready, v1
BODY: ## Summary ⏎  ⏎ The Triton attention backend incorrectly allowed FP8 query quantization for ⏎ per-tensor KV cache mode — queries were quantized but the kernel ignored the scales, ⏎ producing incorrect results. ⏎  ⏎ This PR fixes the issue by adding proper Q scale support to the kernel. When ⏎ Q is FP8 and the KV cache uses FP8 per-tensor mode, the kernel now folds ⏎ `q_scale * k_scale` into the attention score scale and applies `v_scale` to ⏎ the output a …[truncated]

### L3-b82e908b4c  (L3, 2026-05-19, sha b82e908b4c65, PR #42347)
TITLE: [Perf][4/n] Eliminate various GPU<->CPU syncs (#42347)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/logits_processors/test_correctness.py (+1/-1); vllm/lora/ops/triton_ops/utils.py (+21/-9); vllm/lora/punica_wrapper/utils.py (+4/-2); vllm/model_executor/models/bert.py (+3/-6); vllm/model_executor/models/gemma3_mm.py (+1/-1); vllm/model_executor/models/glm4_1v.py (+1/-0); vllm/model_executor/models/granite_speech.py (+7/-7); vllm/model_executor/models/idefics3.py (+1/-1); vllm/model_executor/models/internvl.py (+5/-3); vllm/model_executor/models/llava_next.py (+1/-1); (+13 more)
LABELS: ready, v1, multi-modality, qwen
DEEP_STUDY: deep-study performance PR ()
BODY: Last batch of "low-hanging" GPU<->CPU sync eliminations, found via https://github.com/vllm-project/vllm/pull/40561.

### L3-36dcaf25d8  (L3, 2026-05-19, sha 36dcaf25d8e0, PR #37844)
TITLE: [XPU] add gptq(int4) support (#37844)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/intel_jobs/test-intel.yaml (+3/-1); vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py (+2/-2); vllm/model_executor/kernels/linear/mixed_precision/xpu.py (+47/-10); vllm/model_executor/layers/quantization/utils/marlin_utils.py (+3/-0)
LABELS: intel-gpu, ready, ci/build
BODY: ## Purpose ⏎ support gptq model on xpu by reusing `XPUwNa16LinearKernel`. ⏎   ⏎  ⏎ ## Test Plan ⏎  ⏎ # gptq model: ⏎ python3 examples/basic/offline_inference/generate.py --model Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4 ⏎ #compressed tensor model: ⏎ python3 examples/basic/offline_inference/generate.py --model superjob/Qwen3-4B-Instruct-2507-GPTQ-Int4 ⏎  ⏎ ## Test Result ⏎ output make sense. ⏎ --- ⏎ [details omitted]

### L3-9aaf83ef50  (L3, 2026-05-19, sha 9aaf83ef502f, PR #43119)
TITLE: [CI failure] Temporarily disable using persistent cache for flashinfer autotune (#43119)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/warmup/kernel_warmup.py (+16/-0)
LABELS: bug, ready, ci-failure, nvidia
ISSUES: #43086 [CI Failure]: [Kernels (B200)
BODY: ## Purpose ⏎ It appears that flashinfer autotune file cache is inadequate to discriminate kernel parameters like `use_8x4_sf_layout` in its file cache keys, resulting in invalid tactics being chosen with colliding calls: ⏎  ⏎ ``` ⏎  ⏎ (EngineCore pid=9381) ERROR 05-19 11:25:35 [core.py:1159]   File "/usr/local/lib/python3.12/dist-packages/flashinfer/gemm/gemm_base.py", line 6627, in forward ⏎ -- ⏎   | (EngineCore pid=9381) ERROR 05-19 11:25:35 [core.py: …[truncated]

### L3-f1e3f0e6d6  (L3, 2026-05-19, sha f1e3f0e6d685, PR #41354)
TITLE: [XPU] Use custom op collective behavior  (#41354)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/platforms/xpu.py (+4/-0)
LABELS: intel-gpu, ready, v1
BODY: ## Purpose ⏎ Use custom op collective so that fusion patterns can be enabled subsequently. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ### Accuracy  ⏎  ⏎ **meta-llama/Llama-2-13b-chat-hf** ⏎  ⏎ | Case | Exact match (strict) | StdErr (strict) | Exact match (flexible) | StdErr (flexible) | ⏎ | --- | ---: | ---: | ---: | ---: | ⏎ | eager(Main) | 0.356 | 0.0303 | 0.356 | 0.0303 | ⏎ | eager(PR） | 0.368 | 0.0306 | 0.368 | 0.0306 | ⏎ | compile(Main) | 0.356 | 0.0303 |  …[truncated]

### L3-f54721bcc3  (L3, 2026-05-19, sha f54721bcc3e0, PR #42976)
TITLE: [Bugfix][MoE] FlashInfer one-sided: workspace union across heterogeneous layers (#42976)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/distributed/test_mnnvl_alltoall.py (+84/-0); vllm/distributed/device_communicators/all2all.py (+58/-27)
LABELS: bug, ready
BODY: ## Purpose ⏎ `FlashInferNVLinkOneSidedManager.initialize()` was first-call-wins: it sized the shared MNNVL workspace from the first MoE layer and short-circuited the rest. Models with heterogeneous MoE blocks payloads (e.g. quantized base MoE + unquantized MTP head) overran the workspace on a later layer's `combine`, tripping FlashInfer's `combinePayloadOffset` assert in `csrc/trtllm_moe_alltoall.cu`. ⏎  ⏎ **Fix**: grow the workspace to the union of …[truncated]

### L3-d740e2c029  (L3, 2026-05-19, sha d740e2c02919, PR #43043)
TITLE: [XPU] update xpu graph usage (#43043)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+1/-1); vllm/distributed/device_communicators/xpu_communicator.py (+1/-0); vllm/distributed/parallel_state.py (+7/-1); vllm/platforms/xpu.py (+0/-17)
LABELS: intel-gpu, ready, v1
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-07aeaf9d4d  (L3, 2026-05-20, sha 07aeaf9d4df8, PR #42663)
TITLE: [6/n] Migrate activation kernels, gptq, gguf, non cutlass w8a8 to libtorch stable ABI (continued) (#42663)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake, L3.platform.rocm_selection
FILES: csrc/attention/dtype_fp8.cuh (+2/-1); CMakeLists.txt (+41/-22); csrc/cuda_vec_utils.cuh (+2/-0); csrc/cutlass_extensions/torch_utils.hpp (+2/-4); csrc/libtorch_stable/activation_kernels.cu (+115/-105); csrc/libtorch_stable/ops.h (+83/-0); csrc/libtorch_stable/quantization/gguf/gguf_kernel.cu (+180/-162); csrc/libtorch_stable/quantization/gguf/moe.cuh (+0/-0); csrc/libtorch_stable/quantization/gguf/moe_vec.cuh (+0/-0); csrc/libtorch_stable/quantization/gptq/compat.cuh (+0/-0); (+18 more)
LABELS: documentation, rocm, ready, ci/build, nvidia
BODY: This is a continuation of the PR #38757 ⏎  ⏎ cc @janeyx99 ⏎  ⏎ ____________________________________________________________________________________ ⏎  ⏎  ⏎ ~~Stacked on https://github.com/vllm-project/vllm/pull/38671, only the top 11 commits are relevant.~~ Commits to review https://github.com/vllm-project/vllm/pull/38757/changes/2c13410412de95b648e9cd8562431dbe9481f9ee..deea6618c38afb4735b442c61e2697c273654292 ⏎  ⏎ Note: some declarations are not deleted …[truncated]

### L3-1cb224430b  (L3, 2026-05-20, sha 1cb224430bea, PR #40717)
TITLE: [GDN] Enable FI Blackwell GDN prefill kernel (#40717)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/model_executor/layers/mamba/gdn_linear_attn.py (+52/-27); vllm/platforms/cuda.py (+6/-0)
LABELS: ready, ci/build, nvidia
BODY: ~## IMPORTANT!!!~ ⏎ ~This PR MUST be merged after this change https://github.com/flashinfer-ai/flashinfer/pull/3155 is merged in Flashinfer and vLLM starts to use this FI version. There is a bug in GDN implementation in FI.~ ⏎  ⏎ ## Purpose ⏎  ⏎ Enable FlashInfer's new Blackwell (SM100) CuTe-DSL GDN prefill kernel ([flashinfer-ai/flashinfer#3001](https://github.com/flashinfer-ai/flashinfer/pull/3001)) by default in vLLM. ⏎ The same PR [Add FlashInfer p …[truncated]

### L3-9c78c99995  (L3, 2026-05-20, sha 9c78c99995b7, PR #42993)
TITLE: [MISC] Fix symm_mem cap-equal gate; log AR backend selection (#42993)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/distributed/device_communicators/cuda_communicator.py (+67/-0); vllm/distributed/device_communicators/symm_mem.py (+1/-1)
LABELS: ready, nvidia
BODY: ## Summary ⏎  ⏎ Two small changes in `vllm/distributed/device_communicators/`: ⏎  ⏎ - **Fix misprint in `SymmMemCommunicator.should_use_symm_mem`** — the size gate ⏎   used a strict less-than (`inp_size < self.max_size`), which silently rejected ⏎   tensors whose byte size exactly equals the per-capability symm_mem cap and ⏎   fell back to PyNCCL even though the buffer was sized to fit them. Changed to ⏎   `<=` so cap-equal tensors take the symm_mem path …[truncated]

### L3-cb600d1cdb  (L3, 2026-05-20, sha cb600d1cdbb0, PR #42330)
TITLE: [Frontend] Forward X-data-parallel-rank header on /inference/v1/generate (#42330)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/entrypoints/serve/disagg/serving.py (+4/-0)
LABELS: frontend, ready
BODY: ## Summary ⏎  ⏎ `/v1/chat/completions` (`vllm/entrypoints/openai/chat_completion/serving.py:265-266`, `:345`) and `/v1/completions` (`vllm/entrypoints/openai/completion/serving.py:141-142`, `:192`) already read the `X-data-parallel-rank` header and pass `data_parallel_rank` through to `engine_client.generate`. The disagg `/inference/v1/generate` endpoint did not — even though: ⏎  ⏎ - The helper `OpenAIServing._get_data_parallel_rank` is already inherited …[truncated]

### L3-53ff50fcd3  (L3, 2026-05-20, sha 53ff50fcd3d2, PR #42651)
TITLE: [Perf] Optimize `CutlassFP8ScaledMMLinearKernel` when padding needed by pre-weight processing, 13.5% TTFT improvement (#42651)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/scaled_mm/cutlass.py (+47/-26)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ We quantize the weight each time we do a forward, this could be optimized through pre-weight processing ⏎  ⏎ ## Test ⏎  ⏎ `VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel,FlashInferFP8ScaledMMLinearKernel VLLM_LOGGING_LEVEL=INFO vllm serve RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8 --tensor-parallel-size 8 --host 127.0.0.1 --port 9256` ⏎  ⏎ ### Acc ⏎  ⏎ `lm_eval --model local-completions --model_args "base_url=http://127.0.0.1:9256/v1 …[truncated]

### L3-2a43b407c5  (L3, 2026-05-20, sha 2a43b407c509, PR #43237)
TITLE: [Bugfix][CI] Add missing import of pad_nvfp4_activation_for_cutlass in flashinfer (#43237)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/nvfp4/flashinfer.py (+1/-0)
LABELS: bug, ready, ci-failure, nvidia
BODY: ## Purpose ⏎ - PR #40082 introduced `FlashInferB12xNvFp4LinearKernel` which calls `pad_nvfp4_activation_for_cutlass` but forgot to add it to the import block, causing a mypy `[name-defined]` error in CI. ⏎ - Adds the missing import from `nvfp4_utils`. ⏎  ⏎ ## Test Plan ⏎ CI mypy check passes on `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py`

### L3-644b2a28e7  (L3, 2026-05-20, sha 644b2a28e7eb, PR #41215)
TITLE: [Bugfix] Use enable_sm120_family for per-tensor FP8 CUTLASS kernels on SM12.1 (#41215)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: 
LABELS: bug, ready, nvidia
ISSUES: #40758 [CI Failure]: `Qwen3.6-35B-A3B-FP8` fails on `NVIDIA GB10` with `cutlass_scaled_mm` / `cutlass_gemm_caller Error Internal` under vLLM nightly + CUDA 13.0
BODY: ## Purpose ⏎  ⏎ Fix per-tensor FP8 CUTLASS kernels failing on SM 12.1 devices (e.g., NVIDIA GB10).  ⏎  ⏎ The `cutlass_3x_gemm_sm120` and `cutlass_3x_gemm_sm120_custom` structs use `enable_sm120_only` which guards on `__CUDA_ARCH == 1200`. On SM 12.1 (`__CUDA_ARCH__ == 1210`), this causes a kernel trap and surfaces as `cutlass_gemm_caller Error Internal`.  ⏎  ⏎ The fix replaces `enable_sm120_only` with `enable_sm120_family` (which checks `>= 1200 && <13 …[truncated]

### L3-5774aad9c5  (L3, 2026-05-20, sha 5774aad9c5b6, PR #43135)
TITLE: [Perf][gpt-oss] Downgrade triton_kernels to v3.5.1 (#43135)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: cmake/external_projects/triton_kernels.cmake (+1/-1); tests/kernels/quantization/test_mxfp4_triton_ep.py (+15/-24); vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py (+57/-32)
LABELS: performance, ready, ci/build, gpt-oss
DEEP_STUDY: deep-study performance PR ()
BODY: Recovers a decode-time regression on gpt-oss models after the triton_kernels v3.5.1 → v3.6.0 bump in #30525 and the routing rewrite in #38504. ⏎  ⏎ - Pin `triton_kernels` back to v3.5.1. ⏎ - Activate the legacy routing path whenever `SparseMatrix` is missing (previously gated to ROCm only). ⏎ - When no expert map is set, call the fused `routing()` kernel directly instead of running `topk_fn` + `pack_bitmatrix` + `routing_from_bitmatrix` as three separate …[truncated]

### L3-6dc0a71843  (L3, 2026-05-20, sha 6dc0a7184387, PR #43230)
TITLE: [Misc] downgrade nvidia-cutlass-dsl to 4.5.0 (#43230)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: requirements/cuda.txt (+1/-1)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎ 4.5.1 breaks the flashinfer GDN kernel on gb200. ⏎  ⏎ How to reproduce: ⏎ ``` ⏎ pytest -s -v tests/gdn/test_prefill_delta_rule.py::test_prefill_kernel_basic[float16-64-seq_lens0-1-1-1-128-1.0-False-True] ⏎ ``` ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ # SPDX-License-Identifier: Apache-2.0 ⏎ # SPDX-FileCopyrightText: Copyright contributors to the vLLM project ⏎  ⏎ from vllm import LLM, SamplingParams ⏎  ⏎ # Sample prompts. ⏎ prompts = [ ⏎     "Hello, my name is", ⏎    …[truncated]

### L3-363fc84407  (L3, 2026-05-20, sha 363fc84407f8, PR #40082)
TITLE: Integrate flashinfer b12x MoE and FP4 GEMM kernels for SM120/121 (#40082)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+39/-0); tests/kernels/moe/test_flashinfer_b12x_moe.py (+229/-0); tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py (+7/-3); vllm/config/kernel.py (+3/-0); vllm/envs.py (+1/-0); vllm/model_executor/kernels/linear/__init__.py (+6/-0); vllm/model_executor/kernels/linear/nvfp4/flashinfer.py (+72/-1); vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py (+223/-0); vllm/model_executor/layers/fused_moe/oracle/nvfp4.py (+13/-0); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+2/-0)
LABELS: ready, nvidia, verified
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Summary ⏎  ⏎ Adds two new FlashInfer b12x backends targeting SM120/SM121 GPUs (DGX Spark GB10, RTX Pro 6000 Blackwell): ⏎  ⏎ **1. b12x fused MoE backend** () ⏎  ⏎ Uses FlashInfer's `b12x_fused_moe` kernel (flashinfer-ai/flashinfer#3066, flashinfer-ai/flashinfer#3080) to accelerate NVFP4 MoE on SM120/SM121. The kernel fuses token dispatch, W1 GEMM, SwiGLU, and W2 GEMM into a single call; BF16 hidden states are passed directly as activation quantization is  …[truncated]

### L3-bde560ed6e  (L3, 2026-05-20, sha bde560ed6e1d, PR #41675)
TITLE: [ROCm] Add QuickReduce min-size override and codec threshold (#41675)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/distributed/test_quick_all_reduce.py (+178/-0); vllm/distributed/device_communicators/quick_all_reduce.py (+65/-8); vllm/envs.py (+15/-0)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ - Add `VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB` to override the ROCm QuickReduce minimum tensor-size threshold. ⏎ - Add `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB` to control when QuickReduce uses the configured codec vs FP QuickReduce. ⏎ - Preserve existing behavior when either env var is unset. ⏎  ⏎ ROCm QuickReduce currently relies on a static minimum-size threshold table to decide when a tensor is large enough to use the cust …[truncated]

### L3-edafea3555  (L3, 2026-05-21, sha edafea35550f, PR #43223)
TITLE: Fix FlashInfer TRTLLM NvFP4 monolithic MoE routing (#43223)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/config.py (+1/-1)
LABELS: bug, ready, nvidia
BODY: ## Summary ⏎  ⏎ This MR fixes the FlashInfer TRTLLM NvFP4 monolithic MoE path for Qwen3.5 MoE NVFP4 models. ⏎  ⏎ The monolithic TRTLLM NvFP4 kernel was receiving vLLM routing metadata with subtly different semantics from FlashInfer TRTLLM. For Qwen3.5 MoE NVFP4 checkpoints, this caused the fused monolithic path to select the wrong routing behavior and quickly diverge into low-quality or repetitive generations. The fix keeps the monolithic implementat …[truncated]

### L3-c68c55d43e  (L3, 2026-05-21, sha c68c55d43e50, PR #42943)
TITLE: [CPU][RISC-V] Add VLEN=256 support to RVV attention kernels (#42943)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+12/-15); csrc/cpu/cpu_attn_rvv.hpp (+68/-101); csrc/cpu/cpu_types_riscv_defs.hpp (+4/-0); csrc/cpu/generate_cpu_attn_dispatch.py (+7/-12)
LABELS: ready, v1, cpu
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎ Following #40119, extend the RVV attention kernel from VLEN=128-only to support both VLEN=128 and VLEN=256(via #39478). ⏎  ⏎ > -march=rv64gcv_zvfh_zvl256b -mrvv-vector-bits=zvl -DRISCV_FP16_SUPPORT (VLEN=256) — the RVV kernel is correctly omitted and dispatch falls back to VEC. ⏎  ⏎ #40119 currently falls back VLEN=256bit to VEC. ⏎  ⏎ ## Test Plan ⏎ Compile with -DVLLM_RVV_VLEN=256 on Spacemit X100 (VLEN=256)  ⏎  ⏎ ## Test Result ⏎ Benchmarked  …[truncated]

### L3-b730c46352  (L3, 2026-05-21, sha b730c4635288, PR #40172)
TITLE: [Perf] [Hybrid] Fused Triton kernel for GPU-side Mamba state postprocessing (#40172)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/e2e/general/test_mamba_prefix_cache.py (+4/-48); tests/v1/worker/test_mamba_utils.py (+2071/-6); vllm/v1/worker/gpu_model_runner.py (+65/-37); vllm/v1/worker/mamba_utils.py (+646/-76)
LABELS: ready, v1, ready-run-all-tests, verified
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Follow up to PR https://github.com/vllm-project/vllm/pull/35442 ⏎  ⏎ PR #35442 eliminated the CPU-GPU synchronization caused by the blocking copy of `num_accepted_tokens` in `_update_states_after_model_execute` for hybrid (Mamba) models when prefix caching is disabled.  ⏎  ⏎ This PR tackles the harder case: removing the CPU bubble caused by `postprocess_mamba`, which requires `num_accepted_tokens` and other per-request metadata on the C …[truncated]

### L3-f2ace1d57d  (L3, 2026-05-21, sha f2ace1d57d28, PR #40848)
TITLE: [Frontend][RFC] Rust front-end integration (#40848)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen
FILES: docker/Dockerfile (+62/-0); docker/Dockerfile.cpu (+50/-0); docker/Dockerfile.nightly_torch (+45/-0); docker/Dockerfile.rocm (+51/-0); docker/Dockerfile.xpu (+45/-0); requirements/build/cpu.txt (+1/-0); requirements/build/cuda.txt (+1/-0); requirements/rocm.txt (+1/-0); setup.py (+115/-43); .buildkite/image_build/image_build.sh (+7/-0); (+25 more)
LABELS: documentation, rocm, frontend, intel-gpu, ready, ci/build, v1, cpu, nvidia
BODY: See corresponding RFC for introducing a rust-based alternative front-end process in vLLM https://github.com/vllm-project/vllm/issues/40846. ⏎  ⏎ For now we have staged the poc implementation in https://github.com/Inferact/vllm-frontend-rs. This PR contains the logic to integrate it into vLLM. ⏎  ⏎ You can try it out by building this branch: ⏎  ⏎ ```shell ⏎ # Full install ⏎ uv pip install . ⏎  ⏎ # Or use pre-compiled wheel ⏎ VLLM_USE_PRECOMPILED=1 uv pip ins …[truncated]

### L3-e26e1f0928  (L3, 2026-05-21, sha e26e1f09280b, PR #42968)
TITLE: [Feature] Add `--cpu-distributed-timeout-seconds` CLI Option for CPU Process Group Timeout (#42968)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/parallel.py (+4/-0); vllm/distributed/parallel_state.py (+7/-1); vllm/distributed/utils.py (+14/-0); vllm/engine/arg_utils.py (+8/-0)
LABELS: ready, verified
BODY: ## Purpose ⏎  ⏎ - Add a new CLI argument `--cpu-distributed-timeout-seconds` to independently configure the timeout for CPU (gloo) process groups, decoupled from the existing device timeout option `--distributed-timeout-seconds`. ⏎ - Improve fault tolerance behavior in future elastic / failure-recovery deployments, where the default Gloo timeout (1800s) can lead to long blocking periods (up to 30 minutes) when a rank fails. ⏎ - Maintain full backward …[truncated]

### L3-b29cbf0652  (L3, 2026-05-21, sha b29cbf065252, PR #42988)
TITLE: [Perf] `zeros` -> `empty` to remove additional fill (#42988)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/turboquant_attn.py (+1/-1); vllm/_custom_ops.py (+2/-2); vllm/model_executor/layers/mamba/gdn_linear_attn.py (+0/-1); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+1/-1)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ `zeros` -> `empty` to remove additional fill: ⏎  ⏎ - switch FP4 quant output-scale allocation from `torch.zeros` to `torch.empty` in `vllm/_custom_ops.py` ⏎ - switch packed FP8 scale allocation from `torch.zeros` to `torch.empty` in `vllm/model_executor/layers/quantization/utils/fp8_utils.py` ⏎ - switch TurboQuant mixed-batch `attn_out` allocation from `torch.zeros` to `torch.empty` ⏎ - remove a redundant `z_out.zero_()` in GDN attention …[truncated]

### L3-65b7a812a2  (L3, 2026-05-22, sha 65b7a812a2da, PR #43225)
TITLE: [CPU] Experimentally enable Triton and MRV2 (#43225)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: .buildkite/hardware_tests/cpu.yaml (+14/-0); docker/Dockerfile.cpu (+5/-6); setup.py (+5/-0); vllm/platforms/cpu.py (+20/-3); vllm/triton_utils/importing.py (+12/-0); vllm/utils/platform_utils.py (+3/-1); vllm/v1/worker/cpu/__init__.py (+0/-0); vllm/v1/worker/cpu/buffer_utils.py (+16/-0); vllm/v1/worker/cpu/model_runner.py (+16/-0); vllm/v1/worker/cpu/shm.py (+62/-0); (+1 more)
LABELS: ready, ci/build, v1, cpu
BODY: ## Purpose ⏎  ⏎ - Refine Triton checks on CPU ⏎ - Enable MRV2 on CPU by patching GPU impl ⏎ - Add a non-block test ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-86ccef7d44  (L3, 2026-05-22, sha 86ccef7d4400, PR #41753)
TITLE: [ROCm] Add XGMI backend for MoRI Connector (#41753)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py (+8/-0); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py (+6/-1); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py (+20/-11)
LABELS: rocm, ready, kv-connector
BODY: ## Purpose ⏎  ⏎ Part of https://github.com/vllm-project/router/issues/142 ⏎  ⏎ This PR adds the option to run MoRI with the XGMI backend using `"kv_connector_extra_config": {"backend": "xgmi"}`. This allows for running PD disagg with MoRI in CI on a single node, without having RDMA drivers installed. ⏎  ⏎ Long term we aim for also adding multi-node tests using the RDMA backend. However this requires https://github.com/vllm-project/vllm/pull/40453 and p …[truncated]

### L3-fb21d8b4f9  (L3, 2026-05-22, sha fb21d8b4f902, PR #42209)
TITLE: Add NVFP4 MOE support for Deepseek V4. (#42209)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/kernels.yaml (+7/-3); tests/kernels/moe/test_trtllm_nvfp4_moe.py (+76/-7); vllm/model_executor/layers/fused_moe/config.py (+7/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+32/-4); vllm/model_executor/layers/fused_moe/layer.py (+1/-0); vllm/model_executor/layers/fused_moe/oracle/nvfp4.py (+39/-2); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py (+1/-0); vllm/model_executor/layers/quantization/modelopt.py (+1/-0); vllm/models/deepseek_v4/quant_config.py (+53/-1)
LABELS: ready, ci/build, deepseek, nvidia
BODY: ## Purpose ⏎ Add NVFP4 MOE support for Deepseek V4 ⏎ Add support for swiglu limit in NVFP4 MOE. ⏎ ## Test Plan ⏎ Added unittest. ⏎ Run NVFP4 MOE DSV4 checkpoint. ⏎ ## Test Result ⏎ passed. ⏎ --- ⏎ [details omitted]

### L3-843715739b  (L3, 2026-05-22, sha 843715739b7b, PR #43149)
TITLE: [Refactor] Extract DeepSeek V4 sparse MLA impl into model folder (#43149)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_sparse
FILES: vllm/models/deepseek_v4/nvidia/flashmla.py (+402/-0); vllm/v1/attention/backends/mla/flashmla_sparse.py (+5/-27); vllm/v1/attention/backends/mla/sparse_swa.py (+1/-1); docs/design/attention_backends.md (+1/-1); tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+2/-2); tests/kernels/test_fused_inv_rope_fp8_quant.py (+3/-2); vllm/models/deepseek_v4/amd/rocm.py (+47/-59); vllm/models/deepseek_v4/nvidia/model.py (+1/-1); vllm/models/deepseek_v4/nvidia/ops/attention.py (+23/-309)
LABELS: documentation, rocm, ready, v1, deepseek, DSv4
BODY: ## Purpose ⏎ Move the V4 sparse-MLA forward path out of vllm.v1.attention.backends and into vllm/models/deepseek_v4/attention/. DeepseekV4MLAAttention now picks its CUDA or ROCm impl once at __init__ via a single platform check inside _select_v4_sparse_impl(); forward() and get_attn_backend() are platform-agnostic. ⏎  ⏎ - New abstract DeepseekV4SparseMLAAttentionImpl with two sibling subclasses: DeepseekV4FlashMLASparseImpl (CUDA) and DeepseekV4ROCM …[truncated]

### L3-d3d1cf6972  (L3, 2026-05-22, sha d3d1cf697260, PR #42951)
TITLE: [XPU]feat: add XPU fallback for MoE topk routing and MXFP4 backend (#42951)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/oracle/mxfp4.py (+13/-1); vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py (+61/-0)
LABELS: intel-gpu, ready, verified
BODY: ## Summary ⏎ Add pure PyTorch fallback for MoE `sqrtsoftplus` scoring on XPU (no Triton dependency), and register XPU in MXFP4 backend priority list. ⏎  ⏎ ## Changes ⏎ - **fused_topk_bias_router.py**: add `_topk_softplus_sqrt_torch()` pure PyTorch fallback for `sqrtsoftplus` scoring on non-CUDA platforms ⏎ - **mxfp4.py**: add XPU to priority backend list with no-op weight conversion ⏎  ⏎ ## Testing ⏎ Tested with DeepSeek-V4 MoE routing on Intel XPU. ⏎  ⏎ # …[truncated]

### L3-8c8b1825eb  (L3, 2026-05-22, sha 8c8b1825eb26, PR #37888)
TITLE: [XPU] Enable multiple key kernels for sparse attention (#37888)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/_custom_ops.py (+44/-0); vllm/_xpu_ops.py (+82/-245); vllm/model_executor/layers/sparse_attn_indexer.py (+74/-65)
LABELS: intel-gpu, ready, deepseek, verified
BODY: ## Purpose ⏎  ⏎ This PR migrates sparse indexer helper ops from XPU-local fallbacks to shared custom op wrappers, and aligns the XPU sparse-attention path with vllm-xpu-kernels usage. ⏎  ⏎ Key changes: ⏎ - Add shared wrappers in [vllm/_custom_ops.py](vllm/_custom_ops.py) for: ⏎   - `top_k_per_row_prefill` ⏎   - `top_k_per_row_decode` ⏎ - Remove Python fallback implementations from [vllm/_xpu_ops.py](vllm/_xpu_ops.py) for: ⏎   - `indexer_k_quant_and_cache` …[truncated]

### L3-a377631d21  (L3, 2026-05-22, sha a377631d21cc, PR #43329)
TITLE: [CI] Fix AMD docker build tests (#43329)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+6/-0); cmake/utils.cmake (+40/-13)
LABELS: bug, rocm, ready, ci/build
BODY: ## Purpose ⏎  ⏎ Fix an intermittent ROCm wheel-build [race in CI (AMD: :docker: build image)](https://buildkite.com/vllm/ci/builds/67384/canvas?sid=019e4a21-b199-4ba9-94ec-2f29928d4bd2&tab=output) introduced by #42663. Four parallel `hipify${NAME}` cmake targets share `${CMAKE_CURRENT_BINARY_DIR}/csrc`; when their `shutil.copytree` + `matched_files_iter` calls overlap, one process reads a `.cu` mid-copy, reports `[skipped, no changes]`, and the mat …[truncated]

### L3-c7624bea5e  (L3, 2026-05-22, sha c7624bea5ebb, PR #42650)
TITLE: [Bugfix] Source num_qo_heads from Attention layers in Flashinfer/Triton metadata builders (#42650)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/flashinfer.py (+5/-3); vllm/v1/attention/backends/triton_attn.py (+8/-4); vllm/v1/attention/backends/utils.py (+26/-0)
LABELS: bug, ready, v1, nvidia
ISSUES: #41651 [Bug]: FlashInfer attention + FP8 KV cache + CUDA graphs produces random output on RTX 6000 Pro Blackwell (sm_120); TRITON_ATTN backend works
BODY: ## Summary ⏎  ⏎ Fixes #41651. And also fixes the illegal memory access error when serving that model with certain configs. ⏎  ⏎ The FlashInfer and Triton attention metadata builders source `num_qo_heads` / `num_heads_q` from `model_config.get_num_attention_heads(parallel_config)`, which returns the model-wide value. Models that override the head count on a subset of layers (Laguna XS.2 sets `num_attention_heads_per_layer = [48, 64, 64, 64, ...]`) end …[truncated]

### L3-7e1b45a092  (L3, 2026-05-22, sha 7e1b45a09252, PR #41126)
TITLE: [Attention] Mamba attention module refactor (#41126)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/compilation.py (+1/-1); vllm/model_executor/layers/mamba/gdn/__init__.py (+0/-0); vllm/model_executor/layers/mamba/gdn/base.py (+58/-0); vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py (+26/-45); vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py (+634/-0); vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (+19/-52); vllm/model_executor/models/kimi_linear.py (+13/-27); vllm/model_executor/models/olmo_hybrid.py (+6/-645); vllm/model_executor/models/qwen3_5.py (+4/-2); vllm/model_executor/models/qwen3_next.py (+4/-2)
LABELS: ready, qwen
BODY: ## Purpose ⏎ following https://github.com/vllm-project/vllm/pull/23831/ ⏎  ⏎ Now mamba attention in vLLM has some problem: ⏎ 1.  the implementation locate here and there,  the code is hard for maintaining ⏎ 2. some implementation is pluggable, some are not. There is no clear principle  ⏎ 3. Lots of duplicate code  can be clean up. ⏎  ⏎ |Model| mamba type| pluggable|location| Used by| ⏎ |-|-|-|-|-| ⏎ |OlmoHybridGatedDeltaNet|gdn_attention|**No**|**model_exe …[truncated]

### L3-23f7b11bf4  (L3, 2026-05-22, sha 23f7b11bf4b7, PR #43427)
TITLE: [Bugfix] Detect wrong libcute_dsl_runtime.so variant in FlashInfer GDN (#43427)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (+84/-2)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ Detect a packaging conflict in `nvidia-cutlass-dsl[cu13]` that silently breaks FlashInfer's Blackwell GDN prefill kernel — and, by extension, any other cuTe-DSL-based kernel. ⏎  ⏎ `nvidia-cutlass-dsl-libs-base` and `nvidia-cutlass-dsl-libs-cu13` both ship into the shared `nvidia_cutlass_dsl/` namespace and write many of the same on-disk paths (the runtime `.so`, MLIR Python bindings, cuTe-DSL Python sources) with different content. Wh …[truncated]

### L3-552bbe6f4e  (L3, 2026-05-22, sha 552bbe6f4e6d, PR #38822)
TITLE: [Attention] Add head_dim=512 support for FlashInfer trtllm attention backend (#38822)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+11/-10); docs/assets/contributing/dockerfile-stages-dependency.png (+0/-0); docs/design/attention_backends.md (+2/-2)
LABELS: documentation, ready, v1, nvidia
BODY: Add `512` to the FlashInfer backend's supported head sizes, enabling models with `head_dim=512` attention layers to use the FlashInfer trtllm attention kernels on Blackwell GPUs. ⏎  ⏎ This companion PR enables the head_dim=512 cubin support in FlashInfer: https://github.com/flashinfer-ai/flashinfer/pull/2959

### L3-8de5cabeb7  (L3, 2026-05-23, sha 8de5cabeb70d, PR #42950)
TITLE: [XPU]fix: add XPU platform guards to DeepSeek-V4 ops (#42950)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/sparse_swa.py (+5/-1); vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+8/-7); vllm/model_executor/layers/activation.py (+3/-3); vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py (+5/-1); vllm/models/deepseek_v4/compressor.py (+5/-1); vllm/models/deepseek_v4/nvidia/model.py (+5/-5)
LABELS: rocm, intel-gpu, ready, v1, deepseek, verified
BODY: ## Summary ⏎ Add `is_xpu()` guards to skip CUDA-specific features on XPU, enabling DeepSeek-V4 ops to run on Intel XPU without functional changes for CUDA/ROCm. ⏎  ⏎ ## Changes ⏎ - **activation.py**: route `SiluAndMulWithClamp` to `forward_native` on XPU ⏎ - **deepseek_v4.py**: disable `aux_streams`, reuse ROCm forward path on XPU ⏎ - **deepseek_compressor.py**: skip `launch_pdl` on XPU ⏎ - **fused_inv_rope_fp8_quant.py**: skip `launch_pdl` on XPU ⏎ - **sparse_s …[truncated]

### L3-a5bbd81e2e  (L3, 2026-05-23, sha a5bbd81e2e87, PR #42952)
TITLE: [XPU]feat: enable FP8 block-scaled quantization on XPU (#42952)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/__init__.py (+3/-0); vllm/model_executor/kernels/linear/scaled_mm/triton.py (+2/-2); vllm/model_executor/layers/quantization/input_quant_fp8.py (+10/-1); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+8/-2)
LABELS: intel-gpu, ready, verified
BODY: ## Summary ⏎ Enable the Triton-based FP8 block-scaled GEMM kernel on XPU, and add necessary scale handling for the XPU quantization path. ⏎  ⏎ ## Changes ⏎ - **scaled_mm/triton.py**: allow XPU in `TritonFp8BlockScaledMMKernel.is_supported()` ⏎ - **linear/__init__.py**: register `TritonFp8BlockScaledMMKernel` for `PlatformEnum.XPU` ⏎ - **input_quant_fp8.py**: route group quant to custom op on XPU (same logic as ROCm) ⏎ - **fp8_utils.py**: upcast E8M0 scales on  …[truncated]

### L3-54d153637b  (L3, 2026-05-23, sha 54d153637b5e, PR #42915)
TITLE: [XPU] reudce host overhead of XPU MOE (#42915)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/xpu_moe.py (+23/-18)
LABELS: intel-gpu, ready, verified
BODY: Need bump up vllm-xpu-kernels.

### L3-a7be0f342d  (L3, 2026-05-23, sha a7be0f342dd3, PR #43209)
TITLE: [7/n] Migrate pos_encoding and norm kernels to libtorch stable ABI (continued) (#43209)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+6/-6); csrc/libtorch_stable/dispatch_utils.h (+82/-0); csrc/libtorch_stable/fused_qknorm_rope_kernel.cu (+80/-77); csrc/libtorch_stable/layernorm_kernels.cu (+55/-49); csrc/libtorch_stable/layernorm_quant_kernels.cu (+46/-39); csrc/libtorch_stable/ops.h (+53/-0); csrc/libtorch_stable/pos_encoding_kernels.cu (+52/-46); csrc/libtorch_stable/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu (+118/-99); csrc/libtorch_stable/quantization/fused_kernels/layernorm_utils.cuh (+2/-2); csrc/libtorch_stable/quantization/fused_kernels/quant_conversions.cuh (+1/-1); (+5 more)
LABELS: ready, ci/build
BODY: This is a continuation of the PR #38783 ⏎  ⏎ cc @janeyx99 ⏎  ⏎ ------------------------------------------------------------------------------------------------------------------- ⏎  ⏎  ⏎  ⏎  ⏎ ## Purpose ⏎ ~~Stacked on https://github.com/vllm-project/vllm/pull/38757, commits to review https://github.com/vllm-project/vllm/pull/38783/changes/deea6618c38afb4735b442c61e2697c273654292..8754a4250584115db08113e0889313c939d85eb6~~ ⏎  ⏎ Note: some declarations are no …[truncated]

### L3-a0be71ee47  (L3, 2026-05-23, sha a0be71ee47d3, PR #42787)
TITLE: [MM] Enable FlashInfer metadata support for Qwen2.5-VL vision attention (#42787)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen2_5_vl.py (+72/-13)
LABELS: ready, qwen
BODY: ## Purpose ⏎  ⏎ Enable FlashInfer metadata support for Qwen2.5-VL vision attention. Without this fix, using Flash Infer for MM encoder attn backend will trigger the following error: ⏎  ⏎ ```log ⏎ (EngineCore pid=206604) ERROR 05-15 11:45:14 [core.py:1159]   File "/home/scratch.huah_gpu/vllm-github/vllm/v1/attention/ops/vit_attn_wrappers.py", line 301, in flashinfer_wrapper ⏎ (EngineCore pid=206604) ERROR 05-15 11:45:14 [core.py:1159]     assert sequenc …[truncated]

### L3-1806d1adfc  (L3, 2026-05-24, sha 1806d1adfc9b, PR #43385)
TITLE: [ROCm] [DSv4] [Perf] Support DeepSeek v4 MTP (#43385)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+46/-16); vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py (+22/-11); vllm/models/deepseek_v4/amd/model.py (+0/-1); vllm/models/deepseek_v4/amd/model.py (+1612/-0); vllm/models/deepseek_v4/amd/mtp.py (+0/-1); vllm/models/deepseek_v4/amd/mtp.py (+520/-0); vllm/models/deepseek_v4/amd/rocm.py (+134/-23); vllm/v1/spec_decode/llm_base_proposer.py (+6/-0)
LABELS: rocm, speculative-decoding, ready, v1, deepseek, gpt-oss
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Enable MTP feature for DeepSeek V4 on ROCm. ⏎  ⏎ Side task: The first PR to implement the hardware specific modeling file. ⏎  ⏎ ## Test Plan ⏎  ⏎ Run all 1319 GSM8K requests at concurrency 256 with `--max-num-seqs 256` numshot 8, for both without MTP and with MTP. High concurrency is to ensure that the sparse indexer is logic is working correctly. In past experience while trying to fix the accuracy of DeepSeekV4, if the sparse indexer is  …[truncated]

### L3-aa2b56ffb0  (L3, 2026-05-25, sha aa2b56ffb0c1, PR #43632)
TITLE: [DeepSeek V4] Move MegaMoE input prep kernel to nvidia/ops (#43632)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/test_deepseek_v4_mega_moe.py (+2/-2); vllm/models/deepseek_v4/nvidia/model.py (+2/-163); vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py (+173/-0)
LABELS: deepseek, nvidia
BODY: Move the MegaMoE input prep kernel from `model.py` to `ops/prepare_megamoe.py`.

### L3-681d7dd38b  (L3, 2026-05-26, sha 681d7dd38b59, PR #43303)
TITLE: [Misc][Refactor][ROCm] Convert MoRI-related envvars to extra config args (#43303)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/v1/kv_connector/unit/test_moriio_connector.py (+4/-12); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py (+47/-4); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py (+12/-4); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py (+14/-9); vllm/envs.py (+0/-18)
LABELS: rocm, ready, v1, kv-connector
BODY: ## Purpose ⏎  ⏎ This PR converts MoRI-related envvars to `kv_connector_extra_config` args.  ⏎  ⏎ No behavior change for users using default settings. Warning is logged to users about porting their non-default envvars to the extra config. ⏎  ⏎ | Old env var | New `kv_connector_extra_config` key | Default | ⏎ |---|---|---| ⏎ | `VLLM_MORIIO_CONNECTOR_READ_MODE` | `read_mode` | `false` | ⏎ | `VLLM_MORIIO_QP_PER_TRANSFER` | `qp_per_transfer` | `1` | ⏎ | `VLLM_M …[truncated]

### L3-6ab6ffb428  (L3, 2026-05-26, sha 6ab6ffb428be, PR #43162)
TITLE: [Feat][DSV4] Fuse q pad into deepseek v4 fused kernel (#43162)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/nvidia/flashmla.py (+23/-1); .buildkite/test_areas/kernels.yaml (+22/-0); csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu (+184/-83); csrc/ops.h (+4/-3); csrc/torch_bindings.cpp (+2/-2); tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py (+72/-17); tests/kernels/test_fused_inv_rope_fp8_quant.py (+2/-2); vllm/models/deepseek_v4/amd/model.py (+1/-1); vllm/models/deepseek_v4/amd/rocm.py (+5/-1); vllm/models/deepseek_v4/attention.py (+23/-40); (+1 more)
LABELS: documentation, rocm, ready, ci/build, v1, deepseek
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ Based on https://github.com/vllm-project/vllm/pull/43149 ⏎ Eliminate F.pad if number of head is smaller than kernel can support. Especially in TP cases. Fuse padding into the qk_rope_norm kernel.  ⏎  ⏎ ## Test Plan ⏎ unit test + gsm8k ⏎  ⏎ ## Test Result ⏎ Both passed ⏎  ⏎ --- ⏎ [details omitted]

### L3-d56612c621  (L3, 2026-05-26, sha d56612c62106, PR #43273)
TITLE: [GDN] GDN Prefill kernel for SM100 (#43273)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/mamba/test_gdn_prefill_cutedsl.py (+199/-0); vllm/cute_utils/__init__.py (+134/-0); vllm/cute_utils/_tcgen05.py (+219/-0); vllm/cute_utils/cvt.py (+33/-66); vllm/engine/arg_utils.py (+2/-2); vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (+152/-49); vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/__init__.py (+251/-0); vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/kernel_h.py (+753/-0); vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/kernel_kkt_inv_uw.py (+832/-0); vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/kernel_o.py (+630/-0); (+3 more)
LABELS: ready, v1, verified
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ This PR adds an in-tree GDN prefill kernel for SM100. The kernel was originally developed by team CUDA_ERROR_UNKNOWN (@simveit, @yue-zhang-2025, @mayankagarwals, @zcnrex) for  MLSys 2026 FlashInfer Kernel competition in CUDA C++. I ported the kernel to CuteDSL to enjoy JIT benefits like adaptation to various head configurations and ease of development. ⏎  ⏎ This kernel requires explicit opt-in. The default backend is still FlashInfer. …[truncated]

### L3-d8eebe6d97  (L3, 2026-05-26, sha d8eebe6d9750, PR #43677)
TITLE: [Perf] Optimize Fp8BlockScaledMMLinearKernel input_scale tensor using new_empty() (#43677)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py (+1/-1)
LABELS: performance, ready, deepseek
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ During profiling DeepSeek V4 I noticed `at::native::vectorized_elementwise_kernel` was launched before `tensorrt_llm::kernels::fp8_blockscale_gemm::scale_1x128_kernel`, and found it's because `input_scale` tensor is created using `new_ones()` when `apply_input_quant=False` in https://github.com/vllm-project/vllm/blob/v0.21.1rc0/vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py#L129. ⏎  ⏎ Since when `apply_input …[truncated]

### L3-7e33081cee  (L3, 2026-05-26, sha 7e33081cee7b, PR #42095)
TITLE: [Attention] Make FlexAttention and FlashAttention use num-blocks first layouts (#42095)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.triton.v1_backend, L3.platform.cuda_selection, L3.platform.rocm_selection, L3.flex_attention
FILES: vllm/platforms/cuda.py (+4/-4); vllm/platforms/rocm.py (+4/-4); vllm/platforms/xpu.py (+4/-4); vllm/v1/attention/backends/flash_attn.py (+6/-6); vllm/v1/attention/backends/flex_attention.py (+13/-5); vllm/v1/attention/backends/triton_attn.py (+2/-2); vllm/v1/worker/gpu/attn_utils.py (+2/-3); tests/v1/attention/test_attention_backends.py (+30/-19); tests/v1/attention/test_kv_head_stride_canonicalization.py (+1/-1); tests/v1/kv_connector/unit/offloading_connector/test_worker.py (+17/-77); (+10 more)
LABELS: rocm, intel-gpu, speculative-decoding, v1, kv-connector, nvidia, ready-run-all-tests
BODY: FIX https://github.com/vllm-project/vllm/pull/41657 (context: https://github.com/vllm-project/vllm/pull/41657#issuecomment-4400641394) ⏎  ⏎ and progress towards: https://github.com/vllm-project/vllm/issues/42082 ⏎  ⏎ Also updates Triton HND+layers permutation

### L3-aa6138169f  (L3, 2026-05-26, sha aa6138169f80, PR #43325)
TITLE: [MLA][Attention] Add OOT MLA prefill backend registration mechanism (#43325)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/prefill/__init__.py (+5/-1); vllm/v1/attention/backends/mla/prefill/registry.py (+85/-4); tests/v1/attention/test_mla_prefill_registry.py (+137/-0)
LABELS: ready, v1
BODY: ## Purpose ⏎ Adds a mechanism for registering OOT MLA prefill backends, similar to the one for standard attention backends. ⏎  ⏎ ## Test Plan ⏎ `tests/v1/attention/test_mla_prefill_registry.py` (Part of `V1 Attention (H100)` and `V1 Attention (B200)`) ⏎  ⏎ ## Test Result ⏎ Should pass in CI ⏎  ⏎ --- ⏎ [details omitted]

### L3-adaa5e455a  (L3, 2026-05-26, sha adaa5e455ad8, PR #43710)
TITLE: [DSv4] Refactor compressor & Fix ROCm compatibility (#43710)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/common/ops/__init__.py (+2/-0); vllm/models/deepseek_v4/common/ops/cache_utils.py (+1/-1); vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py (+78/-34); vllm/models/deepseek_v4/common/ops/fused_indexer_q.py (+2/-2); vllm/models/deepseek_v4/common/ops/save_partial_states.py (+101/-0); vllm/models/deepseek_v4/compressor.py (+68/-198); vllm/models/deepseek_v4/nvidia/model.py (+1/-1); vllm/models/deepseek_v4/nvidia/ops/__init__.py (+16/-0); vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py (+95/-3)
LABELS: bug, rocm, ready
BODY: 1. Moves `_save_partial_states_kernel` to `common/ops` ⏎ 2. Moves `sparse_attn_compress_cutedsl.py` to `nvidia/ops` ⏎ 3. Refactor dispatching in DeepseekCompressor ⏎ 4. Fix broken ROCm path

### L3-1fc2cee50a  (L3, 2026-05-26, sha 1fc2cee50a09, PR #42694)
TITLE: [KVConnector][Mooncake] Wire reset_cache cascade end-to-end (#42694)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_scheduler.py (+21/-0); tests/v1/kv_connector/unit/test_mooncake_store_connector.py (+330/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py (+17/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/protocol.py (+36/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py (+27/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py (+64/-10); vllm/v1/core/sched/scheduler.py (+10/-2); vllm/v1/engine/core.py (+12/-2)
LABELS: ready, v1, kv-connector
BODY: ## Summary ⏎  ⏎ `KVConnectorBase_V1.reset_cache` (added in #27170) is currently a no-op for `MooncakeStoreConnector`. A caller hitting `Scheduler.reset_prefix_cache(reset_connector=True)` (or `POST /reset_prefix_cache?reset_external=true`) on a Mooncake engine gets the **local** prefix cache cleared but the **external Mooncake master** keeps all KV blocks computed against the previous weights. For RL post-training and other weight-update workflows th …[truncated]

### L3-dede691c95  (L3, 2026-05-27, sha dede691c9536, PR #43543)
TITLE: [Bugfix] Split attention groups by num_heads_q for spec-decode drafts (#43543)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/spec_decode.yaml (+1/-0); vllm/v1/worker/gpu_model_runner.py (+25/-5)
LABELS: bug, ready, ci/build, v1
BODY: ## Purpose ⏎  ⏎ Fix a hard crash at engine init for any spec-decode setup whose draft layers have a different per-rank `num_heads_q` than the target model. Surfaced today by Gemma4 MTP (target = 8 Q-heads, assistant draft = 4 Q-heads, both 1 KV-head): ⏎  ⏎ ``` ⏎ AssertionError: All layers in one attention group must share num_heads; ⏎ got {8, 4} for ['language_model.model.layers.3.self_attn.attn', ..., ⏎                 'draft_model.layers.0.self_attn.a …[truncated]

### L3-03d9cc2fe2  (L3, 2026-05-27, sha 03d9cc2fe296, PR #43745)
TITLE: [misc] Bump cutedsl version to 4.5.2 (#43745)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/cuda.txt (+1/-1)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-284e6f543d  (L3, 2026-05-27, sha 284e6f543d46, PR #43361)
TITLE: [8/n] Migrate merge_attn_states, mamba, sampler to torch stable ABI (continued) (#43361)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.merge.cuda_lse
FILES: CMakeLists.txt (+5/-5); csrc/libtorch_stable/attention/merge_attn_states.cu (+56/-49); csrc/ops.h (+0/-43); csrc/torch_bindings.cpp (+0/-59); pyproject.toml (+1/-0); csrc/libtorch_stable/mamba/selective_scan.h (+7/-4); csrc/libtorch_stable/mamba/selective_scan_fwd.cu (+102/-111); csrc/libtorch_stable/mamba/static_switch.h (+0/-0); csrc/libtorch_stable/ops.h (+53/-0); csrc/libtorch_stable/sampler.cu (+59/-56); (+3 more)
LABELS: ready, ci/build
BODY: Continuation of PR #38841 ⏎  ⏎ @janeyx99  ⏎  ⏎ Stacked on #43209, commits to review https://github.com/vllm-project/vllm/pull/43361/changes/2292383dce4e96a468df7d8aac22315c114641a6..dab4bae2e67327dfe418ada22a20f54cdf871e6b ⏎  ⏎ ------------------------------------------------------------------------------------------- ⏎ Commits to review ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest tests/kernels/attention/test_merge_attn_states.py ⏎ pytest tests/kernels/test_top_k_per_r …[truncated]

### L3-158289e0fc  (L3, 2026-05-27, sha 158289e0fce9, PR #43697)
TITLE: [Docs] Fix MLA prefill backend default docs (#43697)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: docs/design/attention_backends.md (+3/-2); tools/pre_commit/generate_attention_backend_docs.py (+4/-3)
LABELS: documentation, ready
BODY: ## Summary ⏎  ⏎ It seems the MLA prefill backend docs were not aligned with the current auto-selection behavior, so this updates them to say FlashAttention is tried first, with Blackwell falling back to TRT-LLM Ragged, FlashInfer, then TokenSpeed MLA. ⏎  ⏎ https://github.com/vllm-project/vllm/blob/3aea37d28e44f0b8389e2cfa9876c3faa62543f4/vllm/v1/attention/backends/mla/prefill/selector.py#L65-L75

### L3-5963c19478  (L3, 2026-05-27, sha 5963c194787d, PR #43617)
TITLE: Fix Qwen3-VL and Qwen3-omni-thinker accuracy degradation from deepstack inputs under torch.compile (#43617)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_omni_moe_thinker.py (+14/-11); vllm/model_executor/models/qwen3_vl.py (+14/-11)
LABELS: ready, qwen
BODY: ## Summary ⏎  ⏎ This PR fixes an accuracy regression for Qwen3-VL models when running with vLLM's default `torch.compile` / CUDA graph path. ⏎  ⏎ Qwen3-VL uses `deepstack_input_embeds` to inject visual features into the decoder. Before this change, `_get_deepstack_input_embeds()` returned `None` when `deepstack_input_embeds_num_tokens == 0`. **During compile warmup / profiling, this can make the decoder graph specialize to the no-deepstack path**. La …[truncated]

### L3-2d2c660104  (L3, 2026-05-27, sha 2d2c660104ee, PR #43727)
TITLE: [MoE] Remove inplace fused experts mechanism (#43727)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/modular_kernel_tools/common.py (+0/-1); tests/kernels/moe/test_batched_deepgemm.py (+0/-2); tests/kernels/moe/test_block_fp8.py (+0/-1); tests/kernels/moe/test_cutlass_moe.py (+0/-2); tests/kernels/moe/test_deepep_deepgemm_moe.py (+0/-3); tests/kernels/moe/test_deepep_moe.py (+0/-1); tests/kernels/moe/test_deepgemm.py (+0/-3); tests/kernels/moe/test_flashinfer.py (+0/-3); tests/kernels/moe/test_flashinfer_b12x_moe.py (+0/-1); tests/kernels/moe/test_flashinfer_moe.py (+0/-1); (+34 more)
LABELS: ready, nvidia
BODY: ## Summary ⏎  ⏎ The `inplace=True` path on the fused-MoE call chain is dead code in mainline vLLM: ⏎ - `requirements/cuda.txt` pins `torch==2.11.0` ⏎ - `disable_inplace()` returns `True` for any `torch>=2.9` (added in #26497 as a workaround for #26378, where the `inplace_fused_experts` torch custom op produced garbled tokens under Inductor with `custom_ops` disabled — torch custom ops can't soundly represent outputs aliasing inputs) ⏎ - every runtime call  …[truncated]

### L3-e54eff769d  (L3, 2026-05-27, sha e54eff769dea, PR #43769)
TITLE: [Bugfix] Pass `routed_scaling_factor` to FlashInfer TRTLLM BF16 MoE (#43769)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_moe.py (+5/-2); vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py (+1/-0)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎  ⏎ `routed_scaling_factor` was not passed to FlashInfer TRTLLM BF16 MoE, leading to serious correctness issues. This affects all BF16 MoE models with `routed_scaling_factor` and `apply_routed_scale_to_output=False` on Blackwell. ⏎  ⏎ (looks like no one runs BF16 MoE Blackwell so far 🙃) ⏎  ⏎ ## Test Plan ⏎  ⏎ Updated `test_unquantized_bf16_flashinfer_trtllm_backend` in `tests/kernels/moe/test_moe.py` ⏎  ⏎ ## Test Result ⏎  ⏎ All results were obta …[truncated]

### L3-7fb9c0197a  (L3, 2026-05-27, sha 7fb9c0197a31, PR #43733)
TITLE: [Bugfix][DFlash]allocate the proper number of lookahead slots (#43733)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/core/sched/scheduler.py (+7/-1)
LABELS: bug, ready, v1, dflash
BODY: ## Purpose ⏎  ⏎ Fixes various crashes when running DFlash, due to lookahead slots not being properly allocated. ⏎  ⏎ - DFlash needs an extra lookahead slot ⏎ - [This PR](https://github.com/vllm-project/vllm/pull/22317) seems to override the lookahead slots when scheduling prefills, despite that drafters will still expect those lookahead slots when drafting after the target model prefill. Only prefill workers in P/D disagg don't worry about the lookahe …[truncated]

### L3-413ac5c070  (L3, 2026-05-27, sha 413ac5c0701a, PR #43664)
TITLE: [Misc][Rocm] Remove redundant `AiterUnifiedAttentionBackend` block size log (#43664)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_unified
FILES: vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+0/-3)
LABELS: rocm, ready, v1
BODY: Minor PR to remove a redundant log on rocm when using AiterUnifiedAttentionBackend. ⏎  ⏎ ``` ⏎  vllm serve Qwen/Qwen3-0.6B --attention-backend ROCM_AITER_UNIFIED_ATTN ⏎  ⏎ . . . ⏎  ⏎ # This log is redundant and also should not be a warning (the user can't really take action on this)  ⏎ (EngineCore pid=1481) WARNING 05-26 10:04:55 [rocm_aiter_unified_attn.py:33] [ROCM_AITER_UNIFIED_ATTN]: Setting kv cache block size to 64. ⏎  ⏎ # This is taken care of by th …[truncated]

### L3-05c50c721e  (L3, 2026-05-28, sha 05c50c721e07, PR #41751)
TITLE: [ROCm] mori: add InterNodeV1LL inter-node kernel selection via VLLM_MORI_INTERNODE_KERNEL (#41751)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/modular_kernel_tools/common.py (+1/-1); tests/kernels/moe/modular_kernel_tools/mk_objects.py (+1/-1); tests/kernels/moe/test_moe_layer.py (+5/-3); vllm/config/parallel.py (+6/-3); vllm/distributed/device_communicators/all2all.py (+12/-3); vllm/distributed/device_communicators/cuda_communicator.py (+7/-2); vllm/model_executor/layers/fused_moe/config.py (+4/-1)
LABELS: rocm, ready, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ This PR adds support for selecting the `InterNodeV1LL` inter-node kernel in the MoRI EP all2all backend via a new environment variable `VLLM_MORI_INTERNODE_KERNEL`. ⏎  ⏎ ## Background ⏎  ⏎ MoRI exposes five kernel types for EP dispatch/combine via `EpDispatchCombineKernelType`: ⏎  ⏎ | Kernel | Value | Characteristic | ⏎ |---|---|---| ⏎ | `IntraNode` | 0 | Single-node XGMI P2P | ⏎ | `InterNode` | 1 | Multi-node baseline | ⏎ | `InterNodeV1` | 2 | Multi-nod …[truncated]

### L3-381edde1b9  (L3, 2026-05-28, sha 381edde1b9bf, PR #43599)
TITLE: [Bugfix][Kernel] TRTLLM NVFP4 MoE chunking (#43599)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py (+0/-3); vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+0/-3); vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py (+0/-3); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+74/-12)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎  ⏎ To mitigate crashes in TRTLLM fused MoE NVFP4 kernel that occurs with a very large number of tokens (>=305k), that can be reached with EP & high DP and/or with high `max_num_batched_tokens`. There are actually two distinct crashes: ⏎ 1. TRTLLM fused MoE kernel may exceed the CUDA grid.Y 64k limit. See [getMaxNumCtasInBatchDim flashinfer function](https://github.com/flashinfer-ai/flashinfer/blob/719ee23fd82cb220d51ad118ca60198718f6c9d …[truncated]

### L3-1223732dda  (L3, 2026-05-28, sha 1223732dda9d, PR #38831)
TITLE: [ModelRunnerV2][Hybrid model] Support kernel block size in hybrid model (#38831)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+6/-2); vllm/v1/worker/gpu/attn_utils.py (+70/-63); vllm/v1/worker/gpu/block_table.py (+2/-5); vllm/v1/worker/gpu/model_runner.py (+3/-4); vllm/v1/worker/gpu/spec_decode/eagle/speculator.py (+1/-1)
LABELS: ready, v1, mrv2
BODY: ## Purpose ⏎ ~~This pr add support for hybrid model in modelrunner v2. Currently add the basic support and validate on Qwen3-Next with A100.~~ ⏎  ⏎ This pr is a follow-up pr of #35520, resolving https://github.com/vllm-project/vllm/pull/35520#discussion_r3050244828 ⏎  ⏎ This pr supports kernel block size in hybrid model, which allows more attention backends to work with hybrid model. ⏎  ⏎ **Mainly changes**: ⏎ 1. refactor the `init_attn_backend` in v2, s …[truncated]

### L3-a583c84e2b  (L3, 2026-05-28, sha a583c84e2be9, PR #43781)
TITLE: [Bugfix][ROCm] Fix Accuracy Drop in Sparse Indexer on gfx950 (#43781)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+5/-3); vllm/model_executor/models/deepseek_v2.py (+9/-1)
LABELS: bug, rocm, ready, v1, deepseek
BODY: ## Purpose ⏎ ROCm's sparse indexer path now supports DeepSeek-V4. However, currently its configuration degrades the accuracy for older models like DeepSeek-V3.2 and GLM-5 on the gfx950 path. ⏎  ⏎ This PR fixes the accuracy drop. Specifically two issues are addressed. ⏎ - The indexer cache layout is hard-coded to `SHUFFLE` and can cause erroneous cache reads when indexer cache `block_size=1`. This PR dispatches the shuffle layout based on the block_si …[truncated]

### L3-61288b5458  (L3, 2026-05-28, sha 61288b5458b9, PR #43860)
TITLE: [Bugfix] Fix HyperCLOVAX CI failure after upstream removed remote code (#43860)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/registry.py (+1/-1); vllm/transformers_utils/config.py (+1/-0)
LABELS: bug, ready, ci/build
BODY: ## Summary ⏎  ⏎ - The upstream HuggingFace repo `naver-hyperclovax/HyperCLOVAX-SEED-Think-14B` [removed](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-14B/commit/9b74e35d4c7e4ffec489f4171273caca8948a2b9) `configuration_hyperclovax.py` and `modeling_hyperclovax.py` (now provided natively by transformers >= 5.9.0), but their `config.json` still has `auto_map` pointing to those deleted files ⏎ - This causes `test_can_initialize_large_sub …[truncated]

### L3-19af4e6dd4  (L3, 2026-05-28, sha 19af4e6dd49b, PR #43846)
TITLE: Fix `OlmoHybridForCausalLM` not initialising (#43846)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/transformers_utils/config.py (+2/-2)
LABELS: ready
BODY: The checkpoint for this model was recently changed from having `rope_parameters = None` to `rope_parameters = {"rope_type": None}`. ⏎  ⏎ This was not compatible with vLLM's `patch_legacy_rope_type` method but could still be valid if the modelling code accounts for it. As you can see from  ⏎  ⏎ https://github.com/vllm-project/vllm/blob/a04afd76aa91106f3e59206f984fa68cfacd5d9f/vllm/model_executor/models/olmo_hybrid.py#L144-L156 ⏎  ⏎ this model does handl …[truncated]

### L3-64e1218673  (L3, 2026-05-28, sha 64e121867382, PR #43014)
TITLE: [Perf] Optimize moe permute by pre-allocate buffer, 9~14% kernel performance improvement (#43014)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_moe_permute_unpermute.py (+21/-0); csrc/moe/moe_ops.h (+3/-0); csrc/moe/moe_permute_unpermute_op.cu (+107/-24); csrc/moe/torch_bindings.cpp (+13/-0); tests/kernels/moe/test_moe_permute_unpermute.py (+77/-0); vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py (+32/-0); vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py (+17/-0); vllm/model_executor/layers/fused_moe/moe_permute_unpermute.py (+176/-34)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Optimize moe permute by pre-allocate buffer, very good for small batch size, no effect for large batch size ⏎  ⏎ ## Test ⏎  ⏎ Covered in unit tests ⏎  ⏎ Perf ⏎  ⏎ `python benchmarks/kernels/benchmark_moe_permute_unpermute.py` for two times ⏎  ⏎ | Batch Size | Now (us) | Main (us) | Improvement | ⏎ |---:|---:|---:|---:| ⏎ | 1 | 6.28 | 7.34 | 14.44%🚀 | ⏎ | 2 | 6.53 | 7.34 | 11.04%🚀 | ⏎ | 4 | 6.69 | 7.40 | 9.59%🚀 | ⏎ | 8 | 6.83 | 7.60 | 10.13%🚀 | ⏎ |  …[truncated]

### L3-0ba46d4b11  (L3, 2026-05-28, sha 0ba46d4b11d2, PR #43679)
TITLE: [ROCm][DSV4] Enable Tilelang MHC replacing torch/triton mhc (#43679)
SOURCES: release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: requirements/build/rocm.txt (+1/-0); requirements/rocm.txt (+1/-0); requirements/test/rocm.in (+1/-0); requirements/test/rocm.txt (+21/-2); tests/kernels/test_mhc_kernels.py (+166/-2); vllm/_tilelang_ops.py (+224/-40); vllm/model_executor/kernels/mhc/tilelang.py (+145/-27); vllm/model_executor/layers/mhc.py (+126/-20); vllm/models/deepseek_v4/amd/model.py (+11/-7); vllm/models/deepseek_v4/amd/mtp.py (+3/-1); (+3 more)
LABELS: rocm, ready, ci/build, nvidia
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ In recent tilelang PR they support Vendor free compilation among CUDA and ROCM wheels in https://github.com/tile-ai/tilelang/pull/2195 . So on ROCm we are pip installing the tilelang wheel from pypi directly. ⏎  ⏎ This PR follows the way in sglang https://github.com/sgl-project/sglang/blob/c47f0e7cdde48ddc718e3c6ee8bc87bebee2e8ff/python/sglang/srt/layers/mhc.py#L88 to add a ENABLE_PDL control so that we can set it to False on unsuppor …[truncated]

### L3-1b5437cec8  (L3, 2026-05-28, sha 1b5437cec810, PR #43136)
TITLE: [ROCm] Bump ROCm to 7.2.3 (#43136)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: .buildkite/release-pipeline.yaml (+1/-1); docker/Dockerfile.rocm_base (+2/-25)
LABELS: rocm, ready, ci/build
BODY: Bump ROCm from 7.2.2 -> 7.2.3 ⏎  ⏎ The profiler hotfix for 7.2.2 is included in 7.2.3, so we no longer need to rebuild the CLR.

### L3-9006204e90  (L3, 2026-05-28, sha 9006204e90d3, PR #42796)
TITLE: [MM][CG] Avoid over-padding Qwen2.5-VL encoder cudagraph window metadata (#42796)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen2_5_vl.py (+82/-11); vllm/v1/worker/encoder_cudagraph.py (+15/-4); vllm/v1/worker/encoder_cudagraph_defs.py (+11/-1)
LABELS: ready, v1, qwen, nvidia
BODY: ## Purpose ⏎  ⏎ This PR follows #40830 .  ⏎  ⏎ On B200, the default MM encoder attention kernel is FLASH_ATTN. Enabling MM encoder CUDA graph capture leads to significant performance degrade. On Hopper, we did not observe such performance degrade. ⏎  ⏎ For B200 and FLASH_ATTN backend, we introduce fixes in this PR to address the performance degrade issue. The original `max_window_seqs_per_batch` upper-bound estimation might be too conservative, and we  …[truncated]

### L3-5b115bb8a3  (L3, 2026-05-28, sha 5b115bb8a33d, PR #43660)
TITLE: [Attention][AMD] Standardize kv layout to blocks first for AMD (#43660)
SOURCES: path_core, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.rocm.aiter_unified, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.platform.rocm_selection
FILES: vllm/v1/attention/backend.py (+7/-0); vllm/v1/attention/backends/rocm_aiter_fa.py (+23/-35); vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+5/-5); vllm/v1/attention/backends/rocm_attn.py (+6/-0); vllm/v1/attention/selector.py (+9/-1); vllm/platforms/rocm.py (+7/-3)
LABELS: rocm, ready, v1
BODY: Specular change for Rocm backend of this PR https://github.com/vllm-project/vllm/pull/42095. ⏎ Standardizing layout (https://github.com/vllm-project/vllm/issues/42082) for attention backends allows to simplify and streamline connector code eg ⏎ https://github.com/vllm-project/vllm/blob/97e4022c6ccb7b2cf1a1fc0a13a17a2a06d74f0d/vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py#L946-L955 ⏎  ⏎ https://github.com/vllm-project/vllm/blob/97e4022c6 …[truncated]

### L3-ed7fe831da  (L3, 2026-05-28, sha ed7fe831da79, PR #43331)
TITLE: [ROCm] Enable the aiter top-k/top-p sampler by default (#43331)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/v1/sample/ops/topk_topp_sampler.py (+0/-4)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ `TopKTopPSampler.forward_hip` (ROCm) was hard-disabled in #32413 (`DISABLE_AITER_SAMPLER = True`) to work around a top-p sampling accuracy issue observed on DeepSeek. **That accuracy issue was fixed in [ROCm/aiter#2035](https://github.com/ROCm/aiter/pull/2035)** (a prefix-sum lane bug in `DeterministicInclusiveSum` plus philox RNG seed/offset handling), so the work-around is obsolete and the aiter sampler can never be reached today. …[truncated]

### L3-f3b2a819f7  (L3, 2026-05-28, sha f3b2a819f7f4, PR #43667)
TITLE: [Perf][KDA] Fuse gate softplus, chunk-local cumsum, and RCP_LN2 scaling (#43667)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_kda.py (+70/-1); vllm/model_executor/layers/fla/ops/kda.py (+282/-18); vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py (+14/-7)
LABELS: ready, verified
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ This PR follows the Kimi Linear KDA prefill optimization direction validated by ⏎ SGLang in https://github.com/sgl-project/sglang/pull/23038. That PR showed that ⏎ the KDA chunk prefill path benefits from moving the raw gate producer closer to ⏎ the chunk consumer and fusing gate activation with chunk-local cumulative gate ⏎ construction. ⏎  ⏎ For vLLM, this PR fuses raw gate activation, chunk-local cumsum, and natural-log ⏎ to log2 scalin …[truncated]

### L3-69b8956dcd  (L3, 2026-05-28, sha 69b8956dcd5a, PR #43891)
TITLE: [Model Refactoring] Remove unncessary torch op registration for DSv4 (#43891)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/attention.py (+9/-59); vllm/models/deepseek_v4/nvidia/model.py (+1/-51)
LABELS: ready
BODY: Simple cleanup: Now that DSv4 does not rely on torch compile, we don't need to wrap the deepgemm kernel calls with the torch op registration.

### L3-9957e4d240  (L3, 2026-05-28, sha 9957e4d240aa, PR #43746)
TITLE: [Model Refactoring] Remove torch compile dependency in DSv4 (#43746)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+17/-0); vllm/models/deepseek_v4/amd/model.py (+0/-2); vllm/models/deepseek_v4/amd/mtp.py (+21/-10); vllm/models/deepseek_v4/common/ops/__init__.py (+3/-0); vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py (+203/-0); vllm/models/deepseek_v4/nvidia/model.py (+0/-2); vllm/models/deepseek_v4/nvidia/mtp.py (+21/-10); vllm/v1/worker/gpu_model_runner.py (+5/-0)
LABELS: ready, v1
BODY: This PR removes the torch compile dependency in the DS V4 model. It uses the breakable CUDA graph #42304 to support PW CUDA graph without torch compile. ⏎  ⏎ | Config | strict-match | flexible-extract | Stderr | ⏎ |---|---:|---:|---:| ⏎ | With MTP (k=3) | 0.9522 | 0.9522 | ±0.0059 | ⏎ | Without MTP | 0.9553 | 0.9545 | ±0.0057 | ⏎  ⏎ To ensure no performance regression, the PR also introduced manual fusions for MTP. With that, I've checked that the perfo …[truncated]

### L3-552eb81918  (L3, 2026-05-28, sha 552eb819184c, PR #40344)
TITLE: [Bugfix][ROCm] Resolve MoRI connector hangs at high concurrency (#40344)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py (+25/-0); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py (+165/-17); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py (+80/-15)
LABELS: bug, rocm, ready, v1, kv-connector
ISSUES: #40340 [Bug]: MoRI Connector hangs at >=128 concurrency
BODY: ## Purpose ⏎  ⏎ Fixes #40340. ⏎  ⏎ There are a few parts of the MoRI-IO connector code that can cause indefinite hangs of the connector. This PR resolves them so we can run at least 512 concurrency requests, both for READ and WRITE modes. ⏎  ⏎ Co-developed with @ichbinblau @chunfangamd ⏎  ⏎ ### Implementation details ⏎  ⏎ [details omitted] ⏎  ⏎ ## Test Plan ⏎  ⏎ 1. Build an image in this branch, or if on MI300X w/ Thor2 NICs you can pull the image I built: ⏎ `` …[truncated]

### L3-c08ebebf30  (L3, 2026-05-28, sha c08ebebf30cd, PR #43803)
TITLE: [Perf] Add do_not_specialize to Mamba SSD chunk kernels (#43803)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mamba/ops/causal_conv1d.py (+0/-2); vllm/model_executor/layers/mamba/ops/ssd_bmm.py (+0/-2); vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py (+0/-2); vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py (+0/-4)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ `seqlen` is a dead parameter in five Mamba prefill kernels (`_chunk_cumsum_fwd_kernel`, `_chunk_state_fwd_kernel`, `_chunk_scan_fwd_kernel`, `_bmm_chunk_fwd_kernel`, `_causal_conv1d_fwd_kernel`). It's either never read or overwritten on entry. Triton specializes plain `int` args by default, so every prefill with a new total-token-count was triggering a fresh JIT compile. ⏎  ⏎ Removing the dead arg eliminates these recompiles. On Nemot …[truncated]

### L3-a9ec46d4b7  (L3, 2026-05-28, sha a9ec46d4b7db, PR #40687)
TITLE: [ROCm][Perf] Support N=5 in wvSplitK skinny GEMM kernels for speculative decoding (#40687)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/rocm/skinny_gemms.cu (+6/-0); vllm/model_executor/layers/utils.py (+1/-1)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR ()
BODY: With speculative decoding (e.g. eagle3 with num_speculative_tokens=4), the target model verification pass has batch size 5 (1 original + 4 speculative tokens). The skinny GEMM wvSplitK kernel only supported N<=4, causing fallback to torch.nn.functional.linear (hipBLAS) for the verification step. ⏎  ⏎ Add case 5 to the wvSplitK dispatch switch and raise the dispatch threshold from N<=4 to N<=5 so the fast HIP kernel path is used during speculative v …[truncated]

### L3-3207e7680e  (L3, 2026-05-28, sha 3207e7680e52, PR #41426)
TITLE: [XPU][MoE] Add WNA16 oracle backend for GPTQ sym-int4 (xpu_fused_moe) (#41426)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/xpu_moe.py (+48/-0); vllm/model_executor/layers/fused_moe/oracle/int_wna16.py (+131/-7)
LABELS: intel-gpu, ready, verified
BODY: ## Summary ⏎  ⏎ Adds a working W4A16 (INT4 weights, FP16 activations) MoE path for Intel XPU, routing through `vllm-xpu-kernels`'s `fused_moe_interface.xpu_fused_moe(is_int4=True)`. Part of [RFC #33214](https://github.com/vllm-project/vllm/issues/33214) — the XPU off-IPEX migration to `vllm-xpu-kernels`. This is the W4A16 MoE milestone; pairs with the existing `INCXPULinearMethod` to give end-to-end attention + MoE on Intel GPUs without IPEX. ⏎  ⏎ AI ass …[truncated]

### L3-53a2088675  (L3, 2026-05-28, sha 53a20886752b, PR #43330)
TITLE: Allow native KV cache dtype in Triton cache update (#43330)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/triton_reshape_and_cache_flash.py (+10/-2)
LABELS: documentation, ready, v1, verified
BODY: # Allow native KV cache dtype in Triton cache update ⏎  ⏎   ## Purpose ⏎  ⏎   Some models advertise quantized KV cache, but explicit unquantized KV cache should still be accepted when the user selects it. The bug is broader than one model: this Triton cache update path rejected explicit native KV dtype strings before reaching the valid non-FP8 store path. ⏎  ⏎   ## Test Plan ⏎  ⏎   Run Gemma4 26B NVFP4 on A100 with explicit BF16 KV cache. ⏎  ⏎   ```bash ⏎   …[truncated]

### L3-9202ea6fda  (L3, 2026-05-28, sha 9202ea6fda05, PR #43445)
TITLE: [Spec Decode] Allow causal DFlash (#43445)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/spec_decode/dflash.py (+17/-15)
LABELS: speculative-decoding, ready, v1
BODY: ## Purpose ⏎  ⏎ This PR is a first step to supporting causal DFlash models. As a starting point, I allow dflash checkpoints to enable full causal or non-causality. This will help enable sliding-window DFlash models for which non-causal attention kernels do not exist.  ⏎  ⏎ The goal of this PR is not to fully support SWA or hybrid causal-non-causal models, but just to make configurable whether we require DFlash to be causal or not. The default is unch …[truncated]

### L3-212deff2ec  (L3, 2026-05-29, sha 212deff2ec77, PR #43575)
TITLE: [feat] add GlmgaProcessor specific logits in `glm4_1v.py` (#43575)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/registry.py (+4/-1); vllm/model_executor/models/glm4_1v.py (+241/-32); vllm/multimodal/video.py (+101/-0)
LABELS: new-model, ready, multi-modality
BODY: Add serving support for the GLMGA multimodal model and improve GLM-4.6V-Flash compatibility within the existing Glm4vForConditionalGeneration framework.                                                     ⏎                                                                          ⏎ # Key changes:                                                                                           ⏎  ⏎ - GLMGA processor integration (glm4_1v.py): GLMGA reuses Glm46 …[truncated]

### L3-60a7a2214f  (L3, 2026-05-29, sha 60a7a2214fab, PR #37622)
TITLE: [Bugfix] Fix Step3 pipeline parallel KeyError for residual tensor (#37622)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/step3_text.py (+1/-1)
LABELS: bug, ready
ISSUES: #37543 [Bug]: 推理vllm，出现如下报错，KeyError：residual
BODY: ## Summary ⏎  ⏎ This PR fixes a `KeyError: 'residual'` that occurs when running Step3 models (e.g., Qwen3.5) with pipeline parallel enabled. ⏎ FIX https://github.com/vllm-project/vllm/issues/37543 ⏎  ⏎ ## Root Cause ⏎  ⏎ The `Step3TextModel.make_empty_intermediate_tensors_factory()` was only creating tensors for `["hidden_states"]`, but the `forward()` method expects both `hidden_states` and `residual` in `intermediate_tensors` for non-first pipeline ra …[truncated]

### L3-0b56815a24  (L3, 2026-05-29, sha 0b56815a24f4, PR #42982)
TITLE: [ROCm][Perf] DSv3.2 MI355X TP4 decode-step orchestration cleanup (3 micro-opts) (#42982)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+57/-25); vllm/model_executor/models/deepseek_v2.py (+2/-4)
LABELS: rocm, ready, v1, deepseek
DEEP_STUDY: deep-study performance PR ()
BODY: ## TL;DR ⏎  ⏎ Three independent CPU-side micro-optimizations on the DeepSeek-V3.2 AITER decode hot path. All are dispatch-side changes — no GPU kernel modifications. ⏎  ⏎ | Workload | Pre-patch | This PR | Δ | ⏎ |---|---:|---:|---:| ⏎ | 5k/500/64 (956-step Kineto aggregate) | 41.522 ms/step | 40.234 ms/step | −1.288 ms (−3.10%) | ⏎ | GSM8K full set, flexible-extract (n=5) | — | **0.9378 ± 0.0067** | — (consistent with baseline) | ⏎  ⏎ ## Changes ⏎  ⏎ ### (1 …[truncated]

### L3-22a58640b4  (L3, 2026-05-29, sha 22a58640b456, PR #43717)
TITLE: [9/n] Migrate attention and cache kernels to torch stable ABI (continued)  (#43717)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.flash_attn.fork_inline_cmake, L3.merge.cuda_lse
FILES: CMakeLists.txt (+9/-11); csrc/libtorch_stable/attention/merge_attn_states.cu (+1/-1); csrc/libtorch_stable/attention/paged_attention_v1.cu (+41/-37); csrc/libtorch_stable/attention/paged_attention_v2.cu (+47/-41); csrc/ops.h (+0/-23); csrc/torch_bindings.cpp (+4/-138); csrc/libtorch_stable/activation_kernels.cu (+1/-1); csrc/libtorch_stable/attention/attention_kernels.cuh (+4/-7); csrc/libtorch_stable/attention/attention_utils.cuh (+2/-2); csrc/libtorch_stable/cache_kernels.cu (+360/-311); (+13 more)
LABELS: ready, ci/build, nvidia
BODY: Continuation of PR #38871 ⏎  ⏎ @janeyx99 ⏎  ⏎  ⏎ ## Purpose ⏎ Migrating vLLM to libtorch stable ABI ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest tests/kernels/attention/test_attention.py ⏎ pytest tests/kernels/attention/test_cache.py ⏎ ``` ⏎  ⏎ ## Test Result ⏎ <img width="1378" height="123" alt="image" src="https://github.com/user-attachments/assets/f653e6eb-93b6-4550-891e-5b4420b43e41" /> ⏎  ⏎ <img width="1096" height="275" alt="image" src="https://github.com/user-attachme …[truncated]

### L3-d2889722ff  (L3, 2026-05-29, sha d2889722ff3a, PR #43961)
TITLE: [Bugfix] Corrupted MLA + linear attention (#43961)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/core/single_type_kv_cache_manager.py (+10/-2)
LABELS: bug, ready, v1
BODY: ## Problem ⏎  ⏎ I noticed that for Kimi Linear, once the KV cache is filled up, the model starts outputting garbage. After a lot of debugging with Codex, we found out that it's related to #35219, which fixed the problem for Qwen. Kimi Linear uses MLA, which is somehow not covered by that PR. I'm not too familiar with KV cache management code, but the fix in this PR (done by Codex) seems to fix the issue. ⏎  ⏎ This problem was not surfaced before prob …[truncated]

### L3-bf18d7e0b4  (L3, 2026-05-29, sha bf18d7e0b453, PR #43270)
TITLE: [Misc][NUMA] Auto-bind to PCT priority cores on DGX B300 + widen EngineCore across shard NUMA nodes (#43270)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/utils_/test_numa_utils.py (+338/-4); vllm/config/parallel.py (+6/-1); vllm/utils/numa_utils.py (+278/-28)
LABELS: ready
BODY: ## Summary ⏎  ⏎ Two zero-config refinements to the existing `--numa-bind` path. Both only change behaviour when `--numa-bind` is on, the user did not pass `--numa-bind-cpus`, and (for the PCT half) the host is a DGX B300. ⏎  ⏎ ## 1. Widen EngineCore across the whole DP shard ⏎  ⏎ EngineCore is the parent of every TP/PP worker spawn, and on a 2-socket box those workers can live on either NUMA node (e.g. TP=8 with `numa_bind_nodes = [0, 0, 0, 0, 1, 1, 1, …[truncated]

### L3-0cff0741ff  (L3, 2026-05-29, sha 0cff0741ff88, PR #41394)
TITLE: [Kernel][ROCm] Native W4A16 kernel for AMD RDNA3 (gfx1100) — fp16 + bf16 (#41394)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.platform.rocm_selection
FILES: CMakeLists.txt (+12/-0); csrc/rocm/ops.h (+9/-0); csrc/rocm/q_gemm_rdna3.cu (+780/-0); csrc/rocm/q_gemm_rdna3_wmma.cu (+2165/-0); csrc/rocm/qdq_4_rdna3.cuh (+239/-0); csrc/rocm/torch_bindings.cpp (+13/-0); tests/kernels/quantization/test_rdna3_w4a16.py (+278/-0); tests/kernels/quantization/test_rdna3_w4a16_selection.py (+89/-0); vllm/_custom_ops.py (+45/-0); vllm/model_executor/kernels/linear/__init__.py (+4/-0); (+3 more)
LABELS: rocm, ready, ci/build, v1
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: <html><head></head><body><h2>Motivation</h2> ⏎ <p>This work was driven by a concrete production problem: <strong>on RDNA3 the only fast W4A16 path was ExLlama, which is fp16-only</strong>, and casting bf16-trained models down to fp16 introduces numerical instability. Modern model families (Qwen2/Qwen3, Qwen3-Coder, Llama 3.x, Mistral, etc.) are trained and released in bf16; their weight ranges and activation magnitudes routinely exceed fp16's ±655 …[truncated]

### L3-6aabe221a5  (L3, 2026-05-29, sha 6aabe221a560, PR #43971)
TITLE: [CI] Make Model Executor test hangs fail fast with a traceback (#43971)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/model_executor.yaml (+9/-2)
LABELS: ready, ci/build
BODY: ## Summary ⏎  ⏎ In build [68772](https://buildkite.com/vllm/ci/builds/68772) (scheduled *Full CI run - daily*), the **Model Executor** step hung for **~10 hours** before being cancelled, blocking the nightly. ⏎  ⏎ Root cause (from the logs): a single test — ⏎ `tests/model_executor/model_loader/fastsafetensors_loader/test_fastsafetensors_loader.py::test_model_loader_download_files` — ⏎ wedged during engine/CUDA init inside the EngineCore subprocess. The last  …[truncated]

### L3-84b2a8a7e7  (L3, 2026-05-29, sha 84b2a8a7e7ec, PR #42553)
TITLE: [MoE Refactor] WNA16 MoE backend selection into oracle module (#42553)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/config.py (+16/-51); vllm/model_executor/layers/fused_moe/experts/marlin_moe.py (+6/-1); vllm/model_executor/layers/fused_moe/experts/trtllm_mxint4_moe.py (+159/-0); vllm/model_executor/layers/fused_moe/oracle/int_wna16.py (+220/-75); vllm/model_executor/layers/quantization/auto_gptq.py (+2/-1); vllm/model_executor/layers/quantization/awq_marlin.py (+5/-7); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py (+134/-241); vllm/model_executor/layers/quantization/utils/quant_utils.py (+3/-0)
LABELS: ready, nvidia
BODY: ## Purpose ⏎  ⏎ Derived from: #39190 ⏎  ⏎ This PR ⏎ 1. refactors `CompressedTensorsWNA16MarlinMoEMethod` to use the `int_wna16` oracle. ⏎ 2. Enables int_w8a16 support for `MarlinExperts`. ⏎ 3. Adds the `TrtLlmMxint4ExpertsMonolithic` experts class for int_w4a16 + group size 32. ⏎  ⏎ cc @bedeks ⏎  ⏎ ## Test Plan ⏎  ⏎ Existing tests should pass as this is a refactoring that maintains the same functionality: ⏎  ⏎ - Flashinfer MxInt4 MoE path (monolithic kernel) ⏎ - …[truncated]

### L3-ab7521d77c  (L3, 2026-05-29, sha ab7521d77cf2, PR #43898)
TITLE: [ROCm][DSv4] Remove device pipeline stall in sparse attention (#43898)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+4/-3)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎ On ROCm's DSv4 paths, there are visible GPU bubbles in each `build_ragged_indices_from_dense` invocation caused by two interacting factors: ⏎ - The host overhead of `indptr[0] = 0` ⏎ - The D2H copy of `indptr[-1].item()` needing synchronization ⏎  ⏎ This PR addresses these by using a single zeros call for tensor creation, and using known parameters known at host-side for buffer creation so that subsequent kernel calls can be promptly enqu …[truncated]

### L3-ff990d0d32  (L3, 2026-05-29, sha ff990d0d322b, PR #43945)
TITLE: [ROCm][CI] Fix AITER unified attention for encoder-decoder cross-attention (#43945)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_unified
FILES: vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+27/-3)
LABELS: rocm, ready, v1
BODY: Fixes ROCm AITER unified attention for encoder-decoder cross-attention when it shares KV cache storage with ROCM_ATTN decoder self-attention. It addresses the regression was introduced by [vllm-project/vllm#43660](https://github.com/vllm-project/vllm/pull/43660). That PR changed `RocmAiterUnifiedAttentionBackend.get_kv_cache_shape` from K/V-first: ⏎  ⏎ ```python ⏎ (2, num_blocks, block_size, num_kv_heads, head_size) ⏎ ``` ⏎  ⏎ to blocks-first: ⏎  ⏎ ```py …[truncated]

### L3-b7fb747d8d  (L3, 2026-05-29, sha b7fb747d8dea, PR #43703)
TITLE: [CI][ROCm] Don't skip MoRI-IO Connector tests (#43703)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_moriio_connector.py (+4/-16)
LABELS: rocm, ready, v1, kv-connector
BODY: ## Purpose ⏎  ⏎ Two MoRI connector unit tests were skipped on AMD CI runners despite the host having RNICs, due to a `ibv_devinfo` command guard which isn't installed in the vLLM ROCm image. We switch to the xGMI MoRI backend to resolve this. ⏎  ⏎ **Rationale:** The unit tests exercise connector functionality and not the RDMA transfer, so we can use any MoRI backend for the unit tests.  ⏎  ⏎ _Note: Future **e2e test** for MoRI **should** use the RDMA b …[truncated]

### L3-e8b5199973  (L3, 2026-05-29, sha e8b51999731e, PR #43565)
TITLE: [XPU] support MTP of gdn attention (#43565)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/_xpu_ops.py (+40/-8)
LABELS: intel-gpu, ready
BODY: Need update kernels: https://github.com/vllm-project/vllm-xpu-kernels/pull/368 ⏎ Tested with Qwen/Qwen3-Next-80B-A3B-Instruct and triton flash attention. ⏎  ⏎ lm_eval results of "num_speculative_tokens": 2: ⏎ <img width="524" height="83" alt="image" src="https://github.com/user-attachments/assets/4850cdbb-d28c-4913-a212-2fee2908e3ac" /> ⏎  ⏎ Throughputs on B60, i/o 1024/512: ⏎ <img width="376" height="561" alt="image" src="https://github.com/user-attach …[truncated]

### L3-559d6710bf  (L3, 2026-05-29, sha 559d6710bf45, PR #38445)
TITLE: [PERF]MiniMax-M2 gate kernel (#38445)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+24/-10); benchmarks/kernels/benchmark_router_gemm.py (+154/-0); cmake/utils.cmake (+10/-0); csrc/libtorch_stable/fp32_router_gemm.cu (+223/-0); csrc/libtorch_stable/fp32_router_gemm_entry.cu (+127/-0); csrc/libtorch_stable/torch_bindings.cpp (+4/-0); tests/kernels/test_fp32_router_gemm.py (+78/-0); vllm/_custom_ops.py (+25/-0); vllm/model_executor/layers/fused_moe/router/gate_linear.py (+67/-9); vllm/model_executor/models/minimax_m2.py (+4/-4)
LABELS: performance, ready, ci/build
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ The MoE gate of MiniMax-M2 needs to perform GEMM in FP32 and requires converting the input to FP32, see [router_logits](https://github.com/vllm-project/vllm/blob/v0.18.1rc0/vllm/model_executor/models/minimax_m2.py#L132). Therefore, in low-concurrency scenarios, up to 3 kernels may be launched in the worst case: ⏎ -  the first one is BF16-to-FP32 conversion ⏎ - and the second and third are the two kernels of `split-K` GEMM.  ⏎  ⏎  ⏎ ![img …[truncated]

### L3-4ff865c38e  (L3, 2026-05-29, sha 4ff865c38eac, PR #43616)
TITLE: [Bugfix] Disable allreduce_rms_fusion when pipeline_parallel_size > 1 (#43616)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+8/-1)
LABELS: bug, ready
BODY: ## Summary ⏎  ⏎ Re-gate the FlashInfer allreduce+RMSNorm fusion to `pipeline_parallel_size == 1`. Verified hang on GB200 with `meta-llama/Llama-3.1-70B-Instruct`, PP=2 TP=2, FlashInfer 0.6.11.post2 — disabling the fusion makes startup complete and inference correct. ⏎  ⏎ ## Why this isn't a duplicate ⏎  ⏎ The same gate originally landed in #35424, was removed in #41458 (which claimed FlashInfer 0.6.7's [#2662](https://github.com/flashinfer-ai/flashinfe …[truncated]

### L3-3becc5db40  (L3, 2026-05-30, sha 3becc5db4034, PR #43817)
TITLE: [ROCm] Add attention sink support to AITer flash attention backend (#43817)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+22/-4); docs/design/attention_backends.md (+1/-1); vllm/_aiter_ops.py (+2/-0)
LABELS: documentation, performance, new-model, rocm, frontend, intel-gpu, speculative-decoding, ready, ci/build, v1
BODY: ## Summary ⏎   - Add `sink_ptr` parameter to `flash_attn_varlen_func` in `_aiter_ops.py` ⏎   - Add `supports_sink` classmethod to `AiterFlashAttentionBackend` ⏎   - Plumb sinks through prefill, extend, and decode paths in `rocm_aiter_fa.py` ⏎   - Use `unified_attention` instead of `pa_fwd_asm`/`paged_attention_v1` ⏎     when sinks are present, as ASM paged attention kernels don't support sinks ⏎  ⏎   ## Test plan ⏎   - Verified with existing ROCm AITer f …[truncated]

### L3-8b8546da1c  (L3, 2026-05-31, sha 8b8546da1c3b, PR #44118)
TITLE: docs: fix MLA attention docstring examples (#44118)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+7/-7)
LABELS: ready, verified
ISSUES: #43309 [Doc]: Fix misleading MLA attention docstring examples
BODY: Fixes #43309. ⏎  ⏎ ## Purpose ⏎  ⏎ Fix misleading examples in the MLA attention module docstring: ⏎ - use `q_nope` in the `ql_nope` example instead of undefined `q` ⏎ - remove the incorrect `@ self.num_heads` from the documented return expression ⏎ - clarify the `Sq / Skv` ratio wording for prefill vs. decode ⏎  ⏎ ## Test Plan ⏎  ⏎ Run ruff on the updated file. ⏎  ⏎ ## Test Result ⏎  ⏎ ```console ⏎ $ python3 -m ruff check vllm/model_executor/layers/attention/mla_attention.py ⏎ Al …[truncated]

### L3-29d69332aa  (L3, 2026-05-31, sha 29d69332aa65, PR #44035)
TITLE: [BugFix] Fix `_has_module` to verify native deps via trial import (#44035)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/utils_/test_import_utils.py (+94/-1); vllm/utils/import_utils.py (+16/-4)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ Original PR: https://github.com/vllm-project/vllm/pull/39873. ⏎  ⏎ - `_has_module` relied solely on `importlib.util.find_spec()`, which only confirms a module's spec exists on disk — not that it can actually be imported. ⏎ - For native-extension packages with missing shared libraries (e.g. `nixl_ep` without `libcudart.so.12`), `find_spec` returns a valid spec while the real import fails with `ImportError: libcudart.so.12: cannot open s …[truncated]

### L3-8796838910  (L3, 2026-06-01, sha 8796838910a0, PR #43770)
TITLE: [Bugfix] fix wrong partial_rotary_factor calculation for bailing_moe model. (#43770)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/bailing_moe.py (+6/-1)
LABELS: bug, ready, verified
BODY: [Ling-flash-2.0](https://www.modelscope.cn/models/inclusionAI/Ling-flash-2.0/file/view/master/config.json?status=1) and  [AntAngelMed](https://www.modelscope.cn/models/MedAIBase/AntAngelMed/file/view/master/config.json?status=1)'s `partial_rotary_factor `is `0.5` & no `rotary_dim`. ⏎  ⏎ `vllm/model_executor/models/bailing_moe.py` old logic: ⏎ ``` ⏎ rotary_dim = getattr(config, "rotary_dim", self.head_dim)  ⏎ # rotary_dim = self.head_dim ⏎ config.rope_p …[truncated]

### L3-985c97a6a8  (L3, 2026-06-01, sha 985c97a6a884, PR #43706)
TITLE: [Perf] Optimize cutlass fp8 scaled mm bypassing padding, 20% kernel performance improvement (#43706)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/scaled_mm/cutlass.py (+57/-9)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Optimize cutlass fp8 scaled mm bypassing padding ⏎  ⏎ ## Test ⏎  ⏎ Using this small script ⏎  ⏎ ```bash ⏎ import time ⏎ import statistics ⏎  ⏎ import torch ⏎ from vllm import _custom_ops as ops ⏎ from vllm.model_executor.kernels.linear.scaled_mm.cutlass import ( ⏎     CutlassFp8BlockScaledMMKernel, ⏎ ) ⏎  ⏎ torch.cuda.set_device(0) ⏎ assert torch.cuda.is_available() ⏎  ⏎ kernel = object.__new__(CutlassFp8BlockScaledMMKernel) ⏎ kernel.config = type("Con …[truncated]

### L3-fd9e91d7e4  (L3, 2026-06-01, sha fd9e91d7e411, PR #41294)
TITLE: [ROCm][CI] Fix and stabilize EAGLE3 acceptance tests (#41294)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/spec_decode/test_acceptance_length.py (+30/-14)
LABELS: rocm, speculative-decoding, ready, v1
BODY: This updates the EAGLE3 acceptance length regression test to use ROCm backend auto-selection instead of hard-selecting attention backends. It adds ROCm-specific expected per-position acceptance values for gpt-oss without widening the default tolerance. The Qwen3-VL FP8 MoE case is limited to ROCm TP4 and enables expert parallelism to avoid the unsupported block-FP8 TP shard shape. The per-position check now treats lower-than-baseline acceptance a …[truncated]

### L3-1f6048abe5  (L3, 2026-06-01, sha 1f6048abe575, PR #42944)
TITLE: fix: glm5.1 pp model loading (#42944)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/deepseek_mtp.py (+8/-2); vllm/model_executor/models/deepseek_v2.py (+17/-3)
LABELS: ready, deepseek
BODY: Hi from [novita.ai](https://novita.ai/) team 👋 ⏎  ⏎ ## Purpose ⏎  ⏎ Fix GLM-5.1 FP8 model loading with pipeline parallelism. ⏎  ⏎ When serving `zai-org/GLM-5.1-FP8` with `--pipeline-parallel-size 8`, loading failed with: ⏎  ⏎ ```text ⏎ KeyError: 'layers.0.self_attn.indexer.wk_weights_proj.weight' ⏎ ``` ⏎  ⏎ ## Test Plan ⏎ Run the server on 8xH200 server and validate on gsm8k. ⏎  ⏎ Start vLLM with pp=8: ⏎ ``` ⏎ vllm serve zai-org/GLM-5.1-FP8 \ ⏎   --host=0.0.0.0 \ …[truncated]

### L3-d68f0b220e  (L3, 2026-06-01, sha d68f0b220efc, PR #43742)
TITLE: [Bugfix][Mooncake] Release GPU pin on failed store in MooncakeStoreConnector (#43742)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_mooncake_store_worker.py (+29/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py (+164/-161)
LABELS: bug, ready, v1, kv-connector
BODY: ## Summary ⏎  ⏎ `KVCacheStoreSendingThread._handle_request` in `MooncakeStoreConnector` decremented `stored_requests[req_id]` only on the happy path. If `batch_is_exist` re-raised (it always does on a Mooncake error — see `worker.py:573`), or any other exception fired between `add_stored_request` and the trailing `dec_stored_request`, the counter stayed `> 0` forever: ⏎  ⏎ - `_get_and_clear_finished_sending` would never mark the request as `done_sending` …[truncated]

### L3-279d25f5cb  (L3, 2026-06-01, sha 279d25f5cbc8, PR #38053)
TITLE: [BugFix] Fix TypeError in MiniCPM-O audio feature unpadding (#38053)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/minicpmo.py (+71/-4)
LABELS: bug, ready
ISSUES: #37981 [Bug]: v0.18.0 fails to run MiniCPM-o-4.5
BODY: ## Summary ⏎ Fixes #37981 ⏎  ⏎ When processing audio inputs for MiniCPM-o-4.5, `process_audios` in `minicpmo.py` raises: ⏎ ``` ⏎ TypeError: only integer tensors of a single element can be converted to an index ⏎ ``` ⏎  ⏎ The root cause is that `audio_feature_lens` from the HF processor is a list of 1D tensors (one per audio, each containing per-chunk frame lengths), but the code used these tensors directly as slice indices in `feat[:, :feature_len]`. PyTorch can …[truncated]

### L3-035733515f  (L3, 2026-06-02, sha 035733515f25, PR #44161)
TITLE: [Kernel][DSv4] Optimize sparse FP8 compressor kernels (#44161)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py (+139/-91)
LABELS: ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎ - optimize the DeepSeek V4 CuteDSL C4A FlashMLA sparse FP8-with-scale packed KV compressor layout ⏎ - optimize the C128A split sparse compressor by matching the 128-row tile with fewer warps and shorter final reductions ⏎ - keep the change scoped to the legacy FlashMLA packed FP8 + UE8M0 scale + BF16 RoPE layout; no FP8/BF16 full-cache kernels are added ⏎  ⏎ ## Why ⏎ Mixed decode/prefill batches with tokens from different sequences regressed ver …[truncated]

### L3-517e74a964  (L3, 2026-06-02, sha 517e74a9644f, PR #44262)
TITLE: [DSV4] Refactor RoPE initialization (#44262)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/amd/model.py (+7/-20); vllm/models/deepseek_v4/common/rope.py (+36/-0); vllm/models/deepseek_v4/nvidia/model.py (+7/-20)
LABELS: ready
BODY: Factor out the repeated initialization logic into `build_deepseek_v4_rope`.

### L3-2588ec4f0a  (L3, 2026-06-02, sha 2588ec4f0a1b, PR #44265)
TITLE: [ROCm] Upgrade AITER to v0.1.13.post1 (#44265)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm_base (+1/-1)
LABELS: rocm, ready, ci/build
BODY: Upgrade AITER to v0.1.13.post1 for kimi-k2.5 perf ⏎  ⏎ CI build with this base candidate: https://buildkite.com/vllm/amd-ci/builds/9015 ⏎  ⏎ @kenroche

### L3-f69ede495b  (L3, 2026-06-02, sha f69ede495b3f, PR #43421)
TITLE: [XPU][Mamba] Triton-based selective scan forward op for XPU (#43421)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/_xpu_ops.py (+474/-0); vllm/model_executor/layers/mamba/ops/mamba_ssm.py (+49/-22)
LABELS: intel-gpu, ready, verified
BODY: ## Purpose ⏎ Adds a Triton implementation of the Mamba selective scan forward pass (selective_scan_fwd) to enable Mamba1 prefill on Intel XPU devices. ⏎  ⏎  ⏎ ## Test Result ⏎ tiiuae/falcon-mamba-7b ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.5208|±  |0.0138| ⏎ |     |       |strict-match    …[truncated]

### L3-8a9eb40808  (L3, 2026-06-02, sha 8a9eb40808dd, PR #43990)
TITLE: [Model Runner V2] Support zeroing freshly allocated KV blocks for hybrid + fp8 KVCache (#43990)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/model_runner.py (+26/-3); vllm/v1/worker/gpu_model_runner.py (+3/-3); vllm/v1/worker/utils.py (+16/-14)
LABELS: ready, v1, mrv2
BODY: ## Purpose ⏎  ⏎ The V2 GPU model runner never zeroed freshly allocated KV cache blocks, unlike the V1 runner. For hybrid gdn+attention models (e.g. Qwen3.5), if use flashinfer(trtllm) attn + fp8 KV cache, FlashInfer quantizes the query and dispatches to the TRTLLM attention kernel, which reads the whole KV page (past the written tokens).  ⏎  ⏎ A newly allocated block still held uninitialized memory, so the kernel produced NaN attention → corrupt logi …[truncated]

### L3-1edfd09ffd  (L3, 2026-06-02, sha 1edfd09ffd1f, PR #43991)
TITLE: [Model Runner V2] Use actual batch max_seq_len for attn metadata (#43991)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/model_states/default.py (+1/-1); vllm/v1/worker/gpu/model_states/mamba_hybrid.py (+7/-1); vllm/v1/worker/gpu/spec_decode/eagle/speculator.py (+6/-1)
LABELS: ready, v1, mrv2
BODY: ## Purpose ⏎  ⏎ PR #40654 introduced using the actual batch `max_seq_len` (instead of `max_model_len`) for attention metadata in `DefaultModelState`. This is a follow-up that applies the same handling to the two V2 paths it missed: `MambaHybridModelState.prepare_attn` and the eagle/MTP draft `_build_draft_attn_metadata`, which still passed `max_seq_len=max_model_len`. ⏎  ⏎ Handing `max_model_len` to FlashInfer makes the TRTLLM attention walk past the …[truncated]

### L3-ea0d045a05  (L3, 2026-06-02, sha ea0d045a05a5, PR #44065)
TITLE: [FlashAttention] Sync FA with upstream (#44065)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+2/-2)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ Corresponding PR: https://github.com/vllm-project/flash-attention/pull/141  ⏎  ⏎ ## Test Plan ⏎ CI ⏎  ⏎ ## Test Result ⏎ TBD ⏎  ⏎ --- ⏎ [details omitted]

### L3-4d93bc35c9  (L3, 2026-06-02, sha 4d93bc35c9f4, PR #44013)
TITLE: Migrate header files to torch stable abi (#44013)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .pre-commit-config.yaml (+1/-1); csrc/libtorch_stable/async_util.cuh (+0/-0); csrc/libtorch_stable/cutlass_extensions/epilogue/broadcast_load_epilogue_c2x.hpp (+0/-0); csrc/libtorch_stable/cutlass_extensions/epilogue/scaled_mm_epilogues_c2x.hpp (+1/-1); csrc/libtorch_stable/fused_qknorm_rope_kernel.cu (+1/-1); csrc/libtorch_stable/launch_bounds_utils.h (+0/-0); csrc/libtorch_stable/persistent_topk.cuh (+0/-0); csrc/libtorch_stable/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu (+1/-1); csrc/libtorch_stable/quantization/fp4/mxfp4_experts_quant.cu (+1/-1); csrc/libtorch_stable/quantization/fp4/nvfp4_experts_quant.cu (+1/-1); (+8 more)
LABELS: ready, nvidia
BODY: ## Purpose ⏎ While moving kernels to use the libtorch stable abi and moving them into the csrc/libtorch_stable directory, we noticed a lot of the header files were being left behind. This PR is to fix that by moving header that are 'stable' (~~i.e. no dependency on libtorch at all or only use stable headers~~ i.e. header files only used by stable kernels) into the libtorch_stable directory ~~and changing the references to these headers in csrc/ ex …[truncated]

### L3-fe32e7830b  (L3, 2026-06-02, sha fe32e7830b2f, PR #43669)
TITLE: [Bugfix] flashinfer: fail fast when --kv-cache-dtype nvfp4 used on unsupported arch (#43669)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+7/-0)
LABELS: bug, ready, v1, nvidia
ISSUES: #43562 [Bug]: --kv-cache-dtype nvfp4 crashes at first request on SM120 instead of failing fast at init
BODY: ## Problem ⏎  ⏎ `--kv-cache-dtype nvfp4` is silently accepted on architectures without ⏎ a trtllm-gen FP4 FMHA kernel. The engine starts cleanly, captures ⏎ graphs, then dies on the first request with either: ⏎  ⏎ - `AttributeError: module 'torch' has no attribute 'nvfp4'` (flashinfer ⏎   dtype resolution), or ⏎ - `RuntimeError: Unsupported architecture` deep in trtllm-gen FMHA ⏎  ⏎ The server appears healthy until the first token, making this harder to ⏎ d …[truncated]

### L3-2427094152  (L3, 2026-06-02, sha 242709415287, PR #43339)
TITLE: [Feature] Support EPLB for DeepSeek v4 Mega Moe (#43339)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/eplb/eplb_utils.py (+16/-7); vllm/models/deepseek_v4/nvidia/model.py (+211/-38); vllm/utils/deep_gemm.py (+1/-0); vllm/v1/worker/gpu_worker.py (+4/-1)
LABELS: ready, ci/build, v1, deepseek
BODY: ## Purpose ⏎ Support EPLB for DeepSeek v4 Mega Moe ⏎  ⏎ ## Test Plan ⏎ - Test gsm8k and GPQA eval on deepseekv4 mega moe with EPLB ⏎  ⏎ ## Test Result ⏎ ### Eval results ⏎ ``` ⏎ VLLM_ENGINE_READY_TIMEOUT_S=3600 \ ⏎ vllm serve deepseek-ai/DeepSeek-V4-Pro \ ⏎   --trust-remote-code \ ⏎   --kv-cache-dtype fp8 \ ⏎   --block-size 256 \ ⏎   --no-enable-prefix-caching \ ⏎   --data-parallel-size 8 \ ⏎   --enable-expert-parallel \ ⏎   --moe-backend deep_gemm_mega_moe \ ⏎    …[truncated]

### L3-0eeba5eec1  (L3, 2026-06-02, sha 0eeba5eec17e, PR #42971)
TITLE: Fix DFlash prefix cache corruption due to missing lookahead block (#42971)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/spec_decode/test_dflash_lookahead.py (+154/-0); vllm/v1/worker/gpu_model_runner.py (+17/-3)
LABELS: bug, speculative-decoding, ready, v1, dflash
BODY: ## Summary ⏎  ⏎ Fixes a KV cache corruption bug in DFlash that causes persistent MGL degradation under prefix caching with concurrency. ⏎  ⏎ ## How we found this ⏎  ⏎ While trying out DFlash, under high concurrency with prefix caching enabled, we observed MGL degradation on requests that were otherwise running correctly. Their KV cache was being corrupted by other newly-arriving prefix-hit requests. We saw a steady decrease in MGL. Specifically, those  …[truncated]

### L3-e9e08c49b9  (L3, 2026-06-02, sha e9e08c49b966, PR #44082)
TITLE: [Bugfix] Cache the EAGLE/MTP lookahead block in the SWA prefix-cache mask (#44082)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_prefix_caching.py (+196/-1); tests/v1/core/test_single_type_kv_cache_manager.py (+2/-2); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py (+5/-5); vllm/v1/core/kv_cache_coordinator.py (+61/-38); vllm/v1/core/single_type_kv_cache_manager.py (+74/-36)
LABELS: bug, ready, v1, kv-connector
BODY: ## Summary ⏎  ⏎ PR #42258 added `SlidingWindowManager._cache_block_mask()` to skip caching SWA blocks that can never serve a prefix-cache hit. When EAGLE/MTP speculative decoding is active **and** the cache-hit alignment (the LCM of per-group block sizes) is larger than the SWA window, that mask is too aggressive: EAGLE's lookup needs `tail + 1` contiguous cached blocks, and the extra `+1` block lives at the *first* position past each aligned segme …[truncated]

### L3-0cbc48c4f9  (L3, 2026-06-02, sha 0cbc48c4f988, PR #42958)
TITLE: Support ModelOpt MXFP8 non-gated MoE (#42958)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+14/-5)
LABELS: ready, nvidia, quantization
BODY: Summary: ⏎ - FlashInfer now supports TRTLLM-GEN MXFP8 MoE kernels. ⏎ - Allow ModelOpt MXFP8 TRTLLM MoE to run non-gated Relu2 activations through that backend. ⏎ - Forward FlashInfer activation_type into the MXFP8 TRTLLM MoE call.

### L3-8b3b71ee9d  (L3, 2026-06-02, sha 8b3b71ee9db1, PR #44036)
TITLE: [CI/Build] Bump flashinfer to v0.6.12 (#44036)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); docker/Dockerfile.nightly_torch (+2/-2); docker/versions.json (+1/-1); requirements/cuda.txt (+2/-2)
LABELS: ci/build, nvidia, ready-run-all-tests
BODY: Bump flashinfer to v0.6.12

### L3-a4ac746405  (L3, 2026-06-02, sha a4ac746405f4, PR #43332)
TITLE: [MoE/b12x] Accept W4A16 (kNvfp4Static, None) in FlashInferB12xExperts supports check (#43332)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py (+27/-4)
LABELS: ready, nvidia, verified
BODY: ## Purpose ⏎  ⏎ `FlashInferB12xExperts._supports_quant_scheme` (introduced by PR #40082) currently requires the activation key to be `kNvfp4Dynamic`, which makes the dispatcher reject every **W4A16 NVFP4** checkpoint (`activation_key == None`) — e.g. `nvidia/Qwen3.6-35B-A3B-2.06GB-per-token`. This forces such checkpoints onto Marlin, even though the b12x kernel itself is W4A16-compatible. ⏎  ⏎ PR #42566 ("W4A16 NVFP4 fused MoE + mixed-precision dispa …[truncated]

### L3-dcdfe66bfa  (L3, 2026-06-02, sha dcdfe66bfacf, PR #44220)
TITLE: [Perf] use triton moe backend on hopper by default (#44220)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/oracle/unquantized.py (+6/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ vLLM uses flashinfer moe backend by default now, it's slower than triton backend ⏎  ⏎ Tested on H200 ⏎ ``` ⏎ vllm bench throughput --model Qwen/Qwen3-30B-A3B --dataset-name random --moe-backend triton -tp 2 ⏎ Throughput: 50.19 requests/s, 57821.92 total tokens/s, 6424.66 output tokens/s ⏎ ``` ⏎  ⏎ ``` ⏎ vllm bench throughput --model Qwen/Qwen3-30B-A3B --dataset-name random --moe-backend flashinfer_cutlass -tp 2 ⏎ Throughput: 48.35 requests/s, 5 …[truncated]

### L3-3f3e2702c2  (L3, 2026-06-02, sha 3f3e2702c2a6, PR #43963)
TITLE: [XPU] Enable rms_norm/act quant fusions (#43963)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/compilation/passes/fusion/act_quant_fusion.py (+5/-1); vllm/compilation/passes/fusion/rms_quant_fusion.py (+12/-4); vllm/compilation/passes/pass_manager.py (+4/-0); vllm/platforms/xpu.py (+10/-9)
LABELS: intel-gpu, ready
BODY: ## Purpose ⏎ - Enable norm/act quant fusions ⏎ - disable warning w/o compile mode ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-c91a87f01a  (L3, 2026-06-02, sha c91a87f01a2b, PR #43978)
TITLE: [BugFix] [GDN] Read linear_key_head_dim from hf_text_config for multimodal models (#43978)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (+2/-2)
LABELS: bug, ready
BODY: ## Summary ⏎  ⏎ For multimodal Qwen3.5 models (e.g. [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json)), `linear_key_head_dim` (128) lives on `hf_text_config`, not `hf_config`. GDN prefill backend selection only read `hf_config`, so `head_k_dim` was `None` and CuteDSL/FlashInfer on Blackwell (SM100) was never enabled: ⏎  ⏎ `INFO 05-29 16:47:49 [qwen_gdn_linear_attn.py:228] Using Triton/FLA GDN prefill kernel (req …[truncated]

### L3-969aec4bc8  (L3, 2026-06-02, sha 969aec4bc845, PR #44356)
TITLE: [Bugfix] Fix Deepseek v4 non-mega-moe model init error (#44356)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/nvidia/model.py (+8/-0)
LABELS: bug, ready, deepseek, nvidia
BODY: ## Purpose ⏎ Fix the following init error, introduced by https://github.com/vllm-project/vllm/pull/43339, which only updated `_init_mega_moe_experts` but not `_init_fused_moe_experts`. ⏎ ``` ⏎  ERROR 06-02 13:02:27 [multiproc_executor.py:868] Traceback (most recent call last): ⏎  ERROR 06-02 13:02:27 [multiproc_executor.py:868]   File "/vllm/vllm/v1/executor/multiproc_executor.py", line 835, in worker_main ⏎  ERROR 06-02 13:02:27 [multiproc_executor.p …[truncated]

### L3-27a93cd426  (L3, 2026-06-02, sha 27a93cd42661, PR #44366)
TITLE: [docker] Stop using extra-index-url for flashinfer-jit-cache (#44366)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1)
LABELS: ready, ci/build
BODY: `flashinfer-jit-cache` is currently quarantined on PyPI ⏎  ⏎ Credit to @jstawinsky

### L3-9af53a3c13  (L3, 2026-06-02, sha 9af53a3c1316, PR #44251)
TITLE: [Perf] Add tuned selective_state_update configs for H200 and RTX PRO … (#44251)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mamba/ops/configs/selective_state_update/headdim=64,dstate=128,device_name=NVIDIA_H200,cache_dtype=float16.json (+87/-0); vllm/model_executor/layers/mamba/ops/configs/selective_state_update/headdim=64,dstate=128,device_name=NVIDIA_H200,cache_dtype=float32.json (+87/-0); vllm/model_executor/layers/mamba/ops/configs/selective_state_update/headdim=64,dstate=128,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,cache_dtype=float16.json (+87/-0); vllm/model_executor/layers/mamba/ops/configs/selective_state_update/headdim=64,dstate=128,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,cache_dtype=float32.json (+87/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Follow up to PR #43083 ⏎  ⏎ Add tuned `selective_state_update` configs for two additional GPUs not yet ⏎ covered: ⏎  ⏎ - **NVIDIA H200** (SM 9.0) ⏎ - **NVIDIA RTX PRO 6000 Blackwell Server Edition** (SM 12.0) ⏎  ⏎ The merged configs in #43083 cover B200, GB200, and H100_80GB_HBM3. On the ⏎ two devices above the loader falls back to the Triton built-in heuristic and ⏎ leaves measurable performance on the table. ⏎  ⏎ ## Test Plan ⏎  ⏎ Generate conf …[truncated]

### L3-ace95c9cf8  (L3, 2026-06-03, sha ace95c9cf830, PR #44347)
TITLE: [Bugfix] Update TrtLLM MoE routing methods (#44347)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/config.py (+17/-12); vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+3/-6); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+1/-6); vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py (+1/-0); vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py (+1/-0); vllm/model_executor/layers/fused_moe/router/zero_expert_router.py (+1/-0)
LABELS: bug, ready, nvidia
DEEP_STUDY: deep-study correctness case vllm:ace95c9cf8: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=#43859
BODY: ## Purpose ⏎ The PR introduces various fixes related to Trtllm MoE routing methods: ⏎ - Revert `_supports_router_logits_dtype` change from https://github.com/vllm-project/vllm/pull/43859, which causes regression in `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`, see [CI failure](https://buildkite.com/vllm/ci/builds/69423/list?sid=019e86ec-54d7-4f57-a959-a4abbc09696d&tab=output) ⏎ ``` ⏎ ValueError: FP8 MoE backend FLASHINFER_TRTLLM does not support the d …[truncated]

### L3-02564b4de0  (L3, 2026-06-03, sha 02564b4de069, PR #43759)
TITLE: [XPU]fallback to TRITON_ATTN for vit attn on xpu when use float32 dtype (#43759)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/platforms/xpu.py (+7/-0)
LABELS: intel-gpu, ready, multi-modality
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎ ` pytest -s -v tests/models/multimodal/generation/test_whisper.py::test_models[True-5-float-openai/whisper-large-v3-turbo]` ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-91945b6e4a  (L3, 2026-06-03, sha 91945b6e4ade, PR #44253)
TITLE: [Bug Fix][Model Runner V2][Spec Decode] Warmup & capture with different attention states for speculator prefill (#44253)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/cudagraph_utils.py (+46/-25); vllm/v1/worker/gpu/model_runner.py (+2/-2); vllm/v1/worker/gpu/spec_decode/eagle/cudagraph.py (+9/-5); vllm/v1/worker/gpu/spec_decode/eagle/speculator.py (+2/-2)
LABELS: bug, ready, v1, nvidia
BODY: # Context ⏎ I noticed an issue previously for MRV2 DSV4 where the model would output gibberish for simple prompts. The root cause was that the model was running both the warmup and capture forward passes using the same attention metadata. This was problematic for attention backends (e.g. FlashMLA) that lazily initialize metadata state. The warmup pass (run eagerly) triggers the lazy init and flips an `have_initialized` flag on the metadata, so the …[truncated]

### L3-128adabfe0  (L3, 2026-06-03, sha 128adabfe0fe, PR #43982)
TITLE: [Bugfix] Fix Gemma4 MTP block_table batch_size mismatch under concurrent load (#43982)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/spec_decode/gemma4.py (+6/-1)
LABELS: bug, speculative-decoding, ready, v1
BODY: ## Purpose ⏎  ⏎ Fix `RuntimeError: batch_size must be equal to batch_size_k` that occurs with **Gemma4 + MTP + FlashAttention** under concurrent load when the batch is partially occupied. ⏎  ⏎ `Gemma4Proposer.set_per_group_block_table()` captures block tables with shape `(num_reqs_padded, max_blocks)` during `_prepare_inputs`. Later, `spec_decode_common_attn_metadata` is unpadded to `num_reqs` via `.unpadded()`, but the per-group block tables stored  …[truncated]

### L3-823d271c0d  (L3, 2026-06-03, sha 823d271c0dc7, PR #44393)
TITLE: [Attention][CPU] Standardize kv layout to blocks first (#44393)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+8/-4); tests/kernels/attention/test_cpu_attn.py (+6/-3)
LABELS: ready, v1, cpu
BODY: ## Purpose ⏎  ⏎ For #42082 ⏎  ⏎ Make the CPU attention backend KV cache follows standard logical shape ⏎  ⏎ ## Test Plan ⏎  ⏎ unit tests ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-e5232679a3  (L3, 2026-06-03, sha e5232679a349, PR #39968)
TITLE: [XPU] Add XPU block-scaled W8A8 fp8 path (#39968)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/__init__.py (+2/-0); vllm/model_executor/kernels/linear/scaled_mm/__init__.py (+4/-0); vllm/model_executor/kernels/linear/scaled_mm/triton.py (+1/-1); vllm/model_executor/kernels/linear/scaled_mm/xpu.py (+38/-4)
LABELS: intel-gpu, ready, verified
BODY: ## Purpose ⏎  ⏎ This PR adds the XPU block-scaled W8A8 FP8 path and updates FP8 block kernel selection so XPU can fall back to Triton when the native XPU FP8 block kernel is unavailable. ⏎  ⏎ Changes included in this update: ⏎ - Enable `TritonFp8BlockScaledMMKernel.is_supported()` on XPU (in addition to CUDA-like). ⏎ - Add `TritonFp8BlockScaledMMKernel` to the XPU FP8 block kernel candidate list as fallback. ⏎ - Add unit tests. ⏎  ⏎ Dependency: ⏎  ⏎  ⏎ ## Tes …[truncated]

### L3-b4b4aaa70e  (L3, 2026-06-04, sha b4b4aaa70ee9, PR #42129)
TITLE: [Inductor] Fast-path Inductor fallback for vllm::*/vllm_aiter::* custom ops (#42129)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/test_inductor_fallback_allow_list_patch.py (+250/-0); vllm/env_override.py (+98/-0)
LABELS: ready
BODY: ## Summary ⏎ When Inductor encounters a custom op without a registered lowering or decomposition (e.g. `vllm::all_reduce`, `vllm_aiter::fused_add_rms_norm`) it correctly creates an implicit fallback that calls into the eager Python impl. However, unless the op's `base_name` (e.g. `vllm::all_reduce`) is a member of `torch._inductor.lowering.FALLBACK_ALLOW_LIST`, `GraphLowering.call_function` (`torch/_inductor/graph.py`, around the `Creating implici …[truncated]

### L3-59d0236193  (L3, 2026-06-04, sha 59d0236193a1, PR #44365)
TITLE: [10b/n] Migrate custom all-reduce, DeepSeek V4 fused MLA, MiniMax reduce-RMS, and MXFP8 MoE to libtorch stable ABI (#44365)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+13/-12); csrc/ops.h (+0/-36); csrc/torch_bindings.cpp (+17/-75); csrc/libtorch_stable/custom_all_reduce.cu (+65/-52); csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu (+70/-56); csrc/libtorch_stable/minimax_reduce_rms_kernel.cu (+71/-58); csrc/libtorch_stable/moe/mxfp8_moe/cutlass_mxfp8_grouped_mm.cu (+69/-0); csrc/libtorch_stable/moe/mxfp8_moe/cutlass_mxfp8_grouped_mm_functor.cuh (+0/-0); csrc/libtorch_stable/moe/mxfp8_moe/cutlass_mxfp8_grouped_mm_launcher.cuh (+73/-54); csrc/libtorch_stable/moe/mxfp8_moe/cutlass_mxfp8_grouped_mm_traits.cuh (+0/-0); (+8 more)
LABELS: ready, ci/build, deepseek, nvidia
BODY: ## Purpose ⏎ Continues the libtorch stable ABI migration by moving several kernels out of legacy _C and into _C_stable_libtorch.  ⏎  ⏎ This PR migrates custom all-reduce, DeepSeek V4 fused MLA, MiniMax reduce-RMS, and MXFP8 MoE kernels from legacy _C to _C_stable_libtorch, converting host code to stable torch APIs and moving CMake/bindings accordingly. QuickReduce stays on legacy _C for ROCm-only builds ⏎  ⏎ cc @janeyx99 @Harry-Chen  ⏎  ⏎ See https://gi …[truncated]

### L3-a6183563b6  (L3, 2026-06-04, sha a6183563b6f6, PR #43447)
TITLE: [Prefix Caching] DeepSeekv4 - Support selective prefix-cache retention for sliding-window KV cache (#43447)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/v1/core/test_kv_cache_utils.py (+37/-0); tests/v1/core/test_prefix_caching.py (+557/-0); vllm/envs.py (+12/-0); vllm/v1/core/block_pool.py (+12/-4); vllm/v1/core/kv_cache_coordinator.py (+52/-2); vllm/v1/core/kv_cache_utils.py (+21/-0); vllm/v1/core/single_type_kv_cache_manager.py (+101/-39)
LABELS: ready, v1, deepseek
BODY: Co-author: @ivanium  ⏎  ⏎ ## Purpose ⏎ DeepSeek v4 now exhibits very low effective prefix cache capacity. For example, on TP8 with 8xB300, the reported KV cache capacity is ~14.5x concurrency. However, a microbenchmark that sends 1M-context requests sequentially shows that after the second request is sent, replaying the first request already begins to miss the prefix cache. This means the practical prefix-cache retention capacity is much lower than  …[truncated]

### L3-ceb0111a90  (L3, 2026-06-04, sha ceb0111a90ac, PR #43241)
TITLE: [Model Runner V2][Spec Decode] Add Gemma4 MTP support (#43241)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/flashinfer.py (+3/-5); vllm/v1/attention/backends/triton_attn.py (+4/-8); vllm/v1/attention/backends/utils.py (+0/-26); vllm/v1/worker/gpu/model_runner.py (+3/-2); vllm/v1/worker/gpu/spec_decode/__init__.py (+16/-3); vllm/v1/worker/gpu/spec_decode/autoregressive/__init__.py (+2/-0); vllm/v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py (+4/-5); vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py (+795/-0); vllm/v1/worker/gpu/spec_decode/eagle/speculator.py (+8/-893); vllm/v1/worker/gpu/spec_decode/gemma4/__init__.py (+2/-0); (+4 more)
LABELS: ready, v1, nvidia, mrv2
BODY: # Context ⏎ Gemma4 MTP is currently not supported in MRV2, but was added to MRV1 in [this PR](https://github.com/vllm-project/vllm/pull/41745). The minimal changes needed for MRV2 include: ⏎ - Constant positions across draft steps ⏎ - Wiring up draft layers to reuse the KV cache of the final target model layer of the same attention group. ⏎ - Returning tuple of tensors: (draft_hidden_states, backbone_hidden_states) from draft model forward, where the …[truncated]

### L3-d0975a4b50  (L3, 2026-06-04, sha d0975a4b5014, PR #42646)
TITLE: [perf] Add gemma RMS AR fusion (#42646)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/passes/distributed/test_fusion_all_reduce.py (+59/-1); vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+162/-1); vllm/model_executor/layers/layernorm.py (+4/-14)
LABELS: intel-gpu, ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ integrate flashinfer gemma RMS AR fusion https://github.com/flashinfer-ai/flashinfer/pull/3322 which are used in gemma and Qwen3-next and Qwen3.5 ⏎  ⏎ Perf: ⏎ Model: `Qwen/Qwen3.5-397B-A17B-FP8`  ISL/OSL = 1024/1024  TP=4 ⏎  ⏎ | conc | Output tok/s (Base) | Mean TTFT ms (Base) | Mean TPOT ms (Base) | Output tok/s (Fusion) | Mean TTFT ms (Fusion) | Mean TPOT ms (Fusion) | ⏎ |------|------:|------:|------:|------:|------:|------:| ⏎ | 1 | 141. …[truncated]

### L3-0c96dd64fb  (L3, 2026-06-04, sha 0c96dd64fb6a, PR #43625)
TITLE: [ROCm] Bump fastsafetensors to v0.3.2 from PyPI, remove git source build (#43625)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: requirements/cuda.txt (+1/-1); requirements/rocm.txt (+4/-1); requirements/test/cuda.in (+1/-1); requirements/test/cuda.txt (+1/-1); requirements/test/nightly-torch.txt (+1/-1); requirements/test/rocm.in (+1/-1); requirements/test/rocm.txt (+4/-2); setup.py (+1/-1)
LABELS: rocm, ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ Bumps `fastsafetensors` from a pinned git commit source build to the PyPI release `v0.3.2` across all requirements files. ⏎  ⏎ Previously, ROCm required installing fastsafetensors directly from git (`git+https://...@<commit>`) because earlier PyPI releases only shipped CUDA wheels. `v0.3.2` ships a universal wheel with CUDA/ROCm runtime detection (foundation-model-stack/fastsafetensors#78), so the source build workaround is no longer need …[truncated]

### L3-b5235fca2e  (L3, 2026-06-04, sha b5235fca2eb7, PR #43827)
TITLE: [DSv4] Adding TRTLLM gen attention kernel (#43827)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.registry
FILES: csrc/torch_bindings.cpp (+2/-1); vllm/models/deepseek_v4/nvidia/flashmla.py (+10/-1); vllm/utils/flashinfer.py (+17/-0); vllm/v1/attention/backends/mla/sparse_swa.py (+7/-2); vllm/v1/attention/backends/registry.py (+11/-0); csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu (+444/-0); csrc/libtorch_stable/ops.h (+17/-0); csrc/libtorch_stable/torch_bindings.cpp (+20/-0); docs/design/attention_backends.md (+14/-0); tests/kernels/test_compressor_kv_cache.py (+149/-0); (+10 more)
LABELS: documentation, ready, v1, deepseek, nvidia
BODY: ## Summary ⏎  ⏎ Rebase of @PerkzZheng's #42316 onto current `main`, plus a few materially ⏎ new pieces: ⏎  ⏎ - **Once-per-step C128A metadata caching** — adds `FlashInferMLASparseMetadata` ⏎   + `FlashInferMLASparseMetadataBuilder` so that for `compress_ratio == 128` ⏎   layers the mixed-sparse-index Triton kernel runs **once per step** instead ⏎   of once per layer. The SWA-baked combine is materialized lazily on first ⏎   access by `ensure_sparse_indice …[truncated]

### L3-1bdc60ed53  (L3, 2026-06-04, sha 1bdc60ed53ad, PR #44493)
TITLE: Fix Kimi-K2.5 FlashInfer ViT metadata (#44493)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/kimi_k25.py (+1/-1); vllm/model_executor/models/kimi_k25_vit.py (+108/-27)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ Fix Kimi-K2.5 ViT metadata handling when using FlashInfer attention. ⏎ ```  bash ⏎ --mm-encoder-attn-backend FLASHINFER ⏎ ``` ⏎ Previously, it would raise an error like below. ⏎ <img width="2816" height="1280" alt="img_v3_0212a_697b1d7e-de15-496f-88ee-14b9a207229g" src="https://github.com/user-attachments/assets/9126ba23-41b9-428f-9aa4-35c92853e16e" /> ⏎  ⏎  ⏎ BTW, I've also removed an unexpected device synchronization by keeping `grid_thws …[truncated]

### L3-3dbb4e0ace  (L3, 2026-06-04, sha 3dbb4e0acef5, PR #44509)
TITLE: [Bugfix] MiniCPM-V-4.6 video inference crash: placeholder count mismatches visual embedding count (#44509)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/minicpmv4_6.py (+52/-1); vllm/multimodal/parse.py (+8/-1)
LABELS: bug, ready, multi-modality
BODY: ## Summary ⏎  ⏎ Sending a video request to `openbmb/MiniCPM-V-4_6` causes `EngineDeadError` — the engine core crashes because the number of `<|video_pad|>` placeholder tokens does not match the number of visual embeddings produced by the vision tower. Image inference works correctly; only video triggers this. ⏎  ⏎ ## Root Cause ⏎  ⏎ Three issues conspire: ⏎  ⏎ 1. **`VideoProcessorItems.get_frame_size` (`parse.py`)** hardcodes `(C, H, W)` shape unpacking. When `_ …[truncated]

### L3-4b87b3e845  (L3, 2026-06-04, sha 4b87b3e845fc, PR #44205)
TITLE: [Bugfix] fix EVS for qwen3-vl (#44205)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_vl.py (+4/-4)
LABELS: bug, ready, qwen
BODY: ## Purpose ⏎ Fix EVS for Qwen3-VL, by reverting the changes to qwen3_vl.py by PR #34246. See Issue #44204 for detailed descriptions. ⏎  ⏎ ## Test Plan ⏎ Launch a service with EVS on (--video_pruning_rate 0.5) and send in a request with video. ⏎ *Other tests should not be necessary, as we are just reverting the change. ⏎  ⏎ ## Test Result ⏎ Fix works on vllm 0.20.2. ⏎  ⏎ ## Duplicate check ⏎ Searched for "evs", did not find any duplicate PR.

### L3-38fd2405f3  (L3, 2026-06-04, sha 38fd2405f35e, PR #41980)
TITLE: use split_group for pytorch process group creation (#41980)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/distributed/test_dcp_a2a.py (+8/-1); tests/distributed/test_pynccl.py (+23/-10); tests/distributed/test_quick_all_reduce.py (+22/-7); tests/distributed/test_split_group.py (+233/-0); tests/distributed/test_torchrun_example.py (+15/-2); tests/distributed/test_torchrun_example_moe.py (+15/-2); tests/kernels/moe/modular_kernel_tools/parallel_utils.py (+10/-1); tests/kernels/moe/test_deepep_deepgemm_moe.py (+8/-1); tests/kernels/moe/test_deepep_moe.py (+8/-1); vllm/distributed/parallel_state.py (+177/-25); (+1 more)
LABELS: rocm, ready
BODY: Summary: ⏎ # Use `torch.distributed.split_group` for process-group creation ⏎  ⏎ ## Summary ⏎  ⏎ This PR replaces `torch.distributed.new_group` with ⏎ `torch.distributed.split_group` for cpu/device subgroup creation in ⏎ `GroupCoordinator`. ⏎  ⏎ `split_group` is required by ⏎ both the deprecation of lazy NCCL initialization and the planned ⏎ migration to torchcomms. ⏎  ⏎ --- ⏎  ⏎ ## Motivation ⏎  ⏎ ### 1. Lazy init is going away — eager init is now the recommended path ⏎  ⏎ PyTorch i …[truncated]

### L3-06f94633e7  (L3, 2026-06-04, sha 06f94633e786, PR #44436)
TITLE: [ROCm][CI] Add test for Aiter unified attn kernel (#44436)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_rocm_aiter_unified_attn.py (+339/-0)
LABELS: rocm, ready
BODY: This PR introduces a test to compare the output of rocm `aiter_unified_attn` kernel with a reference implementation (from existing `test_triton_unified_attention`). ⏎ - Decode, Prefill and Mixed-Prefill Sequences ⏎ - FP16, BF16 and FP8 (kv-cache, query and kv-cache+query) ⏎ - Requires Aiter ⏎  ⏎  ⏎ `pytest -s -v tests/kernels/attention/test_rocm_aiter_unified_attn.py`

### L3-90619351e3  (L3, 2026-06-04, sha 90619351e33f, PR #43556)
TITLE: [Attention] Mamba attention module refactor - LINEAR (#43556)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/attention/test_attention_backends_selection.py (+10/-9); vllm/model_executor/layers/mamba/linear/__init__.py (+0/-0); vllm/model_executor/layers/mamba/linear/bailing_linear_attn.py (+384/-0); vllm/model_executor/layers/mamba/linear/base.py (+66/-0); vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py (+18/-68); vllm/model_executor/models/bailing_moe_linear.py (+13/-439); vllm/model_executor/models/minimax_text_01.py (+14/-35)
LABELS: ready, v1
BODY: ## Purpose ⏎ following https://github.com/vllm-project/vllm/pull/41126/ ⏎  ⏎ This is the 2nd PR for mamba attention module refactor. ⏎  ⏎ This PR merge `BailingMoELinearAttention` and `MiniMaxText01LinearAttention` into `model_executor/layers/mamba/linear`. ⏎  ⏎ After this PR: ⏎ |Model| mamba type| pluggable|location| Used by| ⏎ |-|-|-|-|-| ⏎ |BailingMoELinearAttention|linaer_attention|Yes|model_executor/layers/mamba/gdn/bailing_linear_attn.py|BailingMoeV2 …[truncated]

### L3-56aff0dd15  (L3, 2026-06-04, sha 56aff0dd15c0, PR #44334)
TITLE: [10/n] Migrate cuda_view and silu_and_mul_per_block_quant kernels to torch stale ABI. (#44334)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+7/-11); csrc/cuda_view.cu (+0/-59); csrc/libtorch_stable/cuda_utils_kernels.cu (+0/-0); csrc/libtorch_stable/cuda_view.cu (+76/-0); csrc/libtorch_stable/cutlass_extensions/common.cpp (+1/-1); csrc/libtorch_stable/cutlass_extensions/common.hpp (+0/-0); csrc/libtorch_stable/ops.h (+11/-0); csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_grouped_mm_entry.cu (+1/-1); csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_mm_entry.cu (+1/-1); csrc/libtorch_stable/quantization/fp4/mxfp4_blockwise_moe_kernel.cu (+1/-1); (+15 more)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎ Continues the libtorch stable ABI migration by moving several kernels out of legacy _C and into _C_stable_libtorch.  ⏎  ⏎ **Ops migrated** ⏎ - `get_cuda_view_from_cpu_tensor` — CPU pinned/UVA tensor → CUDA view; uses a version-guarded `deleter` supported for 2.11 and 2.10 fallback copies to device. ⏎ - `silu_and_mul_per_block_quant` — fused SiLU+Mul + per-block FP8/INT8 quant;  ⏎ - Removed cuda_utils_kernels.cu and cutlass_extensions/commo …[truncated]

### L3-4efd6ffde0  (L3, 2026-06-04, sha 4efd6ffde094, PR #44569)
TITLE: [DSV4] Refactor DeepseekV4Attention (#44569)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.mla.flashmla_sparse
FILES: vllm/models/deepseek_v4/nvidia/flashmla.py (+72/-118); vllm/v1/attention/backends/mla/flashmla_sparse.py (+1/-1); vllm/models/deepseek_v4/amd/model.py (+3/-161); vllm/models/deepseek_v4/amd/rocm.py (+65/-68); vllm/models/deepseek_v4/attention.py (+224/-345); vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py (+71/-62); vllm/models/deepseek_v4/nvidia/model.py (+17/-163); vllm/models/deepseek_v4/nvidia/ops/o_proj.py (+68/-0)
LABELS: ready, v1, deepseek, nvidia
BODY: 1. Merge `DeepseekV4MLA` and `DeepseekV4MLAAttention` into `DeepseekV4Attention` as the two classes were essentially no-ops. ⏎ 2. Better code sharing between ROCm and NVIDIA. The dispatching logic is implemented cleanly through class inheritance. This also reduces imports in the shared code. ⏎  ⏎ This PR is pure code re-organization and does not change anything logically. Therefore, the performance and accuracy should remain exactly the same.

### L3-a55fccfc7c  (L3, 2026-06-04, sha a55fccfc7cef, PR #44539)
TITLE: [mamba] unify KDA conv states into one cache to match 2-state SSM layout (#44539)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py (+6/-6); vllm/model_executor/layers/mamba/mamba_utils.py (+7/-19); vllm/model_executor/models/kimi_linear.py (+3/-5)
LABELS: ready
BODY: ## Purpose ⏎ align with GDN for future PD support, see https://github.com/vllm-project/vllm/pull/44064 ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct --trust-remote-code --port 8002 [-tp 2] ⏎ lm_eval --model local-completions --model_args "model=moonshotai/Kimi-Linear-48B-A3B-Instruct,base_url=http://0.0.0.0:8002/v1/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=256,timeout=5000,ma ⏎ x_length=409 …[truncated]

### L3-165b7864d0  (L3, 2026-06-04, sha 165b7864d0e3, PR #40426)
TITLE: [ROCM] [FEAT] Integrate Aiter hipBLASLt GEMM online tuning (#40426)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.platform.rocm_selection
FILES: tests/rocm/aiter/test_aiter_hipb_mm_linear_kernel.py (+406/-0); vllm/_aiter_ops.py (+73/-0); vllm/envs.py (+4/-0); vllm/model_executor/kernels/linear/__init__.py (+3/-0); vllm/model_executor/kernels/linear/scaled_mm/aiter.py (+93/-0); vllm/platforms/rocm.py (+12/-0)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: Enable ROCm AITER hipBLASLt online tuning via a vLLM env var, and add ROCm tests covering the online-tuning flow and kernel gating behavior. ⏎  ⏎ ## Purpose                                                                                                                                                           ⏎   This PR adds support for enabling hipBLASLt online tuning in vLLM through `VLLM_ROCM_USE_AITER_LINEAR_HIPBMM`, which forwards to `HIP_ONLI …[truncated]

### L3-b4a6f26c90  (L3, 2026-06-04, sha b4a6f26c904c, PR #41002)
TITLE: [ROCm][perf] Use workspace manager for sparse indexer allocations (#41002)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+55/-24)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Replace dynamic per-call allocations in the ROCm sparse attention indexer with workspace manager allocations. ⏎  ⏎ The sparse attention indexer uses temporary buffers in both prefill and decode: `k_fp8` / `k_scale` in prefill and decode logits buffers in `rocm_fp8_paged_mqa_logits`. These buffers are runtime scratch space, and their sizes vary with request shape, batch size, and speculative decoding state. They are a good fit for `Wor …[truncated]

### L3-b7c5baf63d  (L3, 2026-06-05, sha b7c5baf63d5f, PR #43926)
TITLE: fix: keep DeepSeek V4 RoPE cache on inv_freq device (#43926)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py (+1/-1)
LABELS: ready, deepseek
BODY: #### Summary ⏎  ⏎ `DeepseekV4ScalingRotaryEmbedding._compute_cos_sin_cache()` currently creates ⏎ `inv_freq` on the active/default torch device, but creates the position arange ⏎ with `device=current_platform.device_type`. ⏎  ⏎ That can mix meta tensors with CPU/CUDA tensors during meta-device model ⏎ construction: ⏎  ⏎ - `inv_freq`: follows `torch.device("meta")` ⏎ - `t`: forced to `current_platform.device_type` ⏎  ⏎ The constructor then fails in `torch.einsum` before  …[truncated]

### L3-063ce98fb7  (L3, 2026-06-05, sha 063ce98fb710, PR #42139)
TITLE: [XPU][MoE] support block_fp8_moe on xpu (#42139)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/intel_jobs/test-intel.yaml (+2/-1); vllm/model_executor/layers/fused_moe/experts/xpu_moe.py (+31/-0); vllm/model_executor/layers/fused_moe/oracle/fp8.py (+2/-1)
LABELS: intel-gpu, ready, ci/build
BODY: add a new XPUExpertsBlockFp8 class for Block FP8 moe model. ⏎  ⏎ Tested with: Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 ⏎ <img width="2470" height="399" alt="image" src="https://github.com/user-attachments/assets/510924b8-03c0-410b-bfba-ab6b03a593c5" />

### L3-91e17d4315  (L3, 2026-06-05, sha 91e17d43152b, PR #38804)
TITLE: Fix sarvam forward compatibility with transformers v5 (#38804)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/transformers_utils/config.py (+34/-1)
LABELS: bug, ready, verified
ISSUES: #38734 [Transformers v5] SarvamMLAForCausalLM
BODY: This PR fixes issue #38734.  ⏎  ⏎ ### Root cause  ⏎ sarvam_mla has a custom [PretrainedConfig](https://huggingface.co/sarvamai/sarvam-105b/blob/main/configuration_sarvam_moe.py) that is incompatible with transformers v5 as the ignore_keys parameter which was supported in previous versions of transformers was removed from the validate_rope() method. This was replaced with a class var in PretrainedConfig `ignore_keys_at_rope_validation`. ⏎ https://gith …[truncated]

### L3-c66b19800b  (L3, 2026-06-05, sha c66b19800bb3, PR #44649)
TITLE: [CI] Bump mistral-common (#44649)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/common.txt (+1/-1); requirements/test/cuda.in (+1/-1); requirements/test/cuda.txt (+1/-1); requirements/test/nightly-torch.txt (+1/-1); requirements/test/rocm.in (+1/-1); requirements/test/rocm.txt (+1/-1); requirements/test/xpu.txt (+1/-1)
LABELS: ready, ci/build, nvidia, mistral
BODY: https://github.com/vllm-project/vllm/pull/44622 updated a mistral tokenizer test to match the behaviour of `mistral-common==1.11.3` without updating `mistral-common` in CI causing failures on `main`. ⏎  ⏎ This PR updates `mistral-common` in CI.

### L3-a80af24356  (L3, 2026-06-05, sha a80af243560e, PR #44635)
TITLE: Speed up docs build (#44635)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+6/-3); AGENTS.md (+2/-0); docs/features/speculative_decoding/README.md (+1/-1); mkdocs.yaml (+9/-9); vllm/_custom_ops.py (+10/-6); vllm/compilation/passes/inductor_pass.py (+9/-6); vllm/compilation/passes/utility/fix_functionalization.py (+10/-7); vllm/compilation/passes/utility/noop_elimination.py (+7/-3); vllm/device_allocator/cumem.py (+10/-7); vllm/distributed/kv_events.py (+10/-5); (+22 more)
LABELS: documentation, ready, v1, multi-modality, kv-connector
BODY: In my local the combination of these changes reduces the build time from 376s to 275s. This is a ~27% decrease which will noticeably improve build and queue times in CI. ⏎  ⏎ The performance affecting changes are: ⏎  ⏎ - Exclude vendored HF processor & config classes from API reference - these model specific classes are not important to include in the API reference ⏎ - Removing `separate_signature` and `show_signature_annotations` - this is more similar to …[truncated]

### L3-7f003a1285  (L3, 2026-06-05, sha 7f003a1285af, PR #44609)
TITLE: Support MiniCPMV batched preprocessing (#44609)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/transformers_utils/processors/minicpmv.py (+63/-56)
LABELS: ready
BODY: ## Purpose ⏎ This PR refines the vendored [MiniCPMVProcessor](https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/processors/minicpmv.py) to support batched text+image preprocessing, referring the upstream MiniCPM-V [processor](https://huggingface.co/openbmb/MiniCPM-V-4_5/blob/main/processing_minicpmv.py) code. It fixes below error when run `python3 ./examples/generate/multimodal/vision_language_offline.py -m minicpmv`: ⏎  ⏎ ``` ⏎  …[truncated]

### L3-aa6fb8a329  (L3, 2026-06-05, sha aa6fb8a329bf, PR #44648)
TITLE: [Bugfix] [ROCm] [Critical] fallback to regular abi for ROCm (#44648)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+16/-8); csrc/cuda_view.cu (+60/-0); csrc/libtorch_stable/ops.h (+4/-3); csrc/libtorch_stable/torch_bindings.cpp (+20/-14); csrc/ops.h (+3/-0); csrc/torch_bindings.cpp (+24/-0)
LABELS: bug, rocm, ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ https://github.com/vllm-project/vllm/pull/44334 PR requires the feature from Torch 2.11 as described in issue https://github.com/vllm-project/vllm/issues/44641 , however ROCm  currently official supports torch 2.10 which is incompatible, this causes the build failure. ⏎  ⏎ Build error message ⏎  ⏎ [details omitted] ⏎  ⏎ ## Test Plan ⏎  ⏎ 1. Build successfully ⏎ 2. Pass `test_uva.py` ⏎  ⏎ ## Test Result ⏎  ⏎ 1. Built successfully ⏎ 2. `pytest -svv …[truncated]
