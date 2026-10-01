### L3-71bcaf99e2  (L3, 2024-02-27, sha 71bcaf99e2cb, PR #3007)
TITLE: Enable GQA support in the prefix prefill kernels (#3007)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L3.triton.prefix_prefill; Prefix-prefill Triton kernel gained GQA support.
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+18/-16); vllm/model_executor/layers/triton_kernel/prefix_prefill.py (+27/-12); tests/kernels/test_prefix_prefill.py (+42/-19)
BODY: 

### L3-2daf23ab0c  (L3, 2024-03-07, sha 2daf23ab0cf0, PR #3005)
TITLE: Separate attention backends (#3005)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
STAGE1: introduce; artifacts=L3.paged.python_wrapper,L3.xformers.v0_backend,L3.flash_attn.v0_backend,L3.flash_attn.upstream_pip,L3.triton.prefix_prefill; Separated attention backends and added wrappers for FlashAttention, xFormers, paged cache ops, and prefix prefill.
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.upstream_pip, L3.triton.prefix_prefill
FILES: setup.py (+45/-3); vllm/model_executor/layers/attention/__init__.py (+5/-0); vllm/model_executor/layers/attention/attention.py (+59/-0); vllm/model_executor/layers/attention/backends/__init__.py (+0/-0); vllm/model_executor/layers/attention/backends/flash_attn.py (+124/-0); vllm/model_executor/layers/attention/backends/xformers.py (+61/-155); vllm/model_executor/layers/attention/ops/__init__.py (+0/-0); vllm/model_executor/layers/attention/ops/paged_attn.py (+138/-0); vllm/model_executor/layers/attention/ops/prefix_prefill.py (+0/-0); vllm/model_executor/models/deepseek.py (+5/-5); vllm/model_executor/models/gemma.py (+5/-5); vllm/model_executor/models/llama.py (+6/-6); .gitignore (+3/-0); tests/kernels/test_prefix_prefill.py (+1/-1); vllm/__init__.py (+23/-7); vllm/model_executor/models/baichuan.py (+6/-7); vllm/model_executor/models/bloom.py (+5/-5); vllm/model_executor/models/chatglm.py (+2/-2); vllm/model_executor/models/falcon.py (+14/-14); vllm/model_executor/models/gpt2.py (+2/-4); vllm/model_executor/models/gpt_bigcode.py (+5/-5); vllm/model_executor/models/gpt_j.py (+2/-2); vllm/model_executor/models/gpt_neox.py (+2/-2); vllm/model_executor/models/internlm2.py (+5/-5); vllm/model_executor/models/mixtral.py (+2/-2); vllm/model_executor/models/mixtral_quant.py (+2/-2); vllm/model_executor/models/mpt.py (+6/-6); vllm/model_executor/models/olmo.py (+4/-4); vllm/model_executor/models/opt.py (+4/-4); vllm/model_executor/models/orion.py (+5/-5); vllm/model_executor/models/phi.py (+2/-2); vllm/model_executor/models/qwen.py (+2/-2); vllm/model_executor/models/qwen2.py (+6/-6); vllm/model_executor/models/stablelm.py (+5/-5); vllm/model_executor/models/starcoder2.py (+2/-2)
BODY: This PR refactors the attention layer. Specifically, it separates the code paths for Ampere or more recent NVIDIA GPUs (which can directly use FlashAttention) and other GPUs, so that the code for the former becomes much simpler. This PR will also bring some performance improvements for ALiBi models, since we now directly call FlashAttention instead of using xformers in the middle.

### L3-1cb0cc2975  (L3, 2024-03-08, sha 1cb0cc2975d1, PR #3269)
TITLE: [FIX] Make `flash_attn` optional (#3269)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes
STAGE1: change_default; artifacts=L3.flash_attn.v0_backend,L3.flash_attn.upstream_pip,L3.xformers.v0_backend; FlashAttention dependency became optional with fallback to xFormers when unavailable.
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.upstream_pip
FILES: setup.py (+3/-45); vllm/model_executor/layers/attention/attention.py (+31/-6); vllm/model_executor/layers/attention/backends/flash_attn.py (+0/-1); .gitignore (+0/-3); vllm/__init__.py (+7/-23)
BODY: The FlashAttention backend introduced in #3005 causes build errors in some environments and increase the package size significantly (44 MB -> 160 MB) as the vLLM package now includes the `flash-attn` (116 MB) package. This PR addresses this by removing `flash_attn` from vLLM's dependency and enables falling back to xFormers when `flash_attn` is not found.

### L3-f48c6791b7  (L3, 2024-03-08, sha f48c6791b7bf, PR #3286)
TITLE: [FIX] Fix prefix test error on main (#3286)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.flash_attn.v0_backend; Fixed main-branch prefix test error by adjusting FlashAttention backend.
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/model_executor/layers/attention/backends/flash_attn.py (+0/-2)
BODY: 

### L3-e4a28e5316  (L3, 2024-03-10, sha e4a28e531659, PR #3262)
TITLE: [ROCM] Fix blockReduceSum to use correct warp counts for ROCm and CUDA (#3262)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.paged.cuda.v1; blockReduceSum warp count was corrected for ROCm/CUDA attention reductions.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+0/-8); csrc/cuda_compat.h (+10/-0); csrc/reduction_utils.cuh (+3/-3)
BODY: blockReduceSum was defaulting to 32 for warp size regardless of the architecture. ⏎  ⏎ Bonus, refactor cuda_compat.h to hold WARP_SIZE define instead of the attention_kernels.cuh

### L3-06ec486794  (L3, 2024-03-14, sha 06ec486794f4, PR #3396)
TITLE: Install `flash_attn` in Docker image (#3396)
SOURCES: subject_keyword, dependency_pin, release_notes
STAGE1: integrate; artifacts=L3.flash_attn.upstream_pip; Docker image began installing flash_attn so FlashAttentionBackend is available in containers.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile (+24/-0)
PERF_LINES: This PR leaves the wheel unchanged, but installs `flash_attn` independently within the Docker build (for both the test as well as the runtime image). This will 
BODY: Recent fix #3269 removed `flash_attn` as an explicit dependency since it was breaking builds in a bunch of environments and made the wheel size larger.  ⏎  ⏎ This PR leaves the wheel unchanged, but installs `flash_attn` independently within the Docker build (for both the test as well as the runtime image). This will allow us to use the `FlashAttentionBackend` in containerized environments without affecting those using the package in other ways.

### L3-6e435de766  (L3, 2024-03-20, sha 6e435de766c7, PR #3236)
TITLE: [1/n][Chunked Prefill] Refactor input query shapes (#3236)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L3.paged.python_wrapper,L3.xformers.v0_backend,L3.flash_attn.v0_backend; Chunked prefill changed query shape assumptions across attention wrappers and backends.
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend
FILES: vllm/model_executor/layers/attention/attention.py (+2/-1); vllm/model_executor/layers/attention/backends/flash_attn.py (+32/-14); vllm/model_executor/layers/attention/backends/xformers.py (+147/-85); vllm/model_executor/layers/attention/ops/paged_attn.py (+5/-4); .buildkite/test-pipeline.yaml (+2/-2); tests/basic_correctness/test_basic_correctness.py (+3/-1); tests/core/test_scheduler.py (+9/-9); tests/lora/test_worker.py (+1/-1); tests/spec_decode/test_multi_step_worker.py (+2/-2); tests/worker/test_model_runner.py (+153/-8); vllm/config.py (+0/-3); vllm/core/scheduler.py (+3/-10); vllm/engine/arg_utils.py (+1/-7); vllm/engine/llm_engine.py (+0/-1); vllm/model_executor/input_metadata.py (+69/-13); vllm/model_executor/layers/activation.py (+2/-2); vllm/model_executor/layers/sampler.py (+0/-1); vllm/worker/model_runner.py (+144/-95)
BODY: It is the first PR to address https://github.com/vllm-project/vllm/issues/3130 ⏎  ⏎ The current query format is not suitable for chunked prefill because after it is enabled, chunked prefill (e.g., size of 764) and decoding requests will be batched together. If we use 2D query (batch_size, seq_len), we should either use hacky solution (treating the last batch as a batch of decoding requests) or have inefficient # of paddings. ⏎  ⏎ To get around this, we should use 1D query, which is more efficient. With 1D query. This PR refactors existing code to support 1D query and adjust padding configuration to support cuda graph. ⏎  ⏎ The first part of https://github.com/vllm-project/vllm/issues/3130

### L3-925f3332ca  (L3, 2024-03-25, sha 925f3332cac4, PR #3462)
TITLE: [Core] Refactor Attention Take 2 (#3462)
SOURCES: path_core, symbol_pickaxe
STAGE1: introduce; artifacts=L3.dispatch.selector,L3.dispatch.abstract_interface,L3.paged.python_wrapper; Attention refactor introduced selector and abstract backend protocol around paged attention wrappers.
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.triton.prefix_prefill, L3.dispatch.selector, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/__init__.py (+0/-0); vllm/attention/backends/abstract.py (+85/-0); vllm/attention/backends/flash_attn.py (+238/-0); vllm/attention/backends/xformers.py (+177/-55); vllm/attention/layer.py (+46/-0); vllm/attention/ops/__init__.py (+0/-0); vllm/attention/ops/paged_attn.py (+217/-0); vllm/attention/ops/prefix_prefill.py (+0/-0); vllm/attention/selector.py (+44/-0); vllm/model_executor/layers/attention/__init__.py (+0/-5); vllm/model_executor/layers/attention/attention.py (+0/-85); vllm/model_executor/layers/attention/backends/flash_attn.py (+0/-139); vllm/model_executor/layers/attention/ops/paged_attn.py (+0/-139); tests/kernels/test_prefix_prefill.py (+1/-2); tests/samplers/test_beam_search.py (+7/-0); tests/worker/test_model_runner.py (+30/-30); vllm/attention/__init__.py (+10/-0); vllm/model_executor/__init__.py (+0/-2); vllm/model_executor/input_metadata.py (+0/-99); vllm/model_executor/models/baichuan.py (+13/-17); vllm/model_executor/models/bloom.py (+14/-18); vllm/model_executor/models/chatglm.py (+18/-23); vllm/model_executor/models/deepseek.py (+14/-18); vllm/model_executor/models/falcon.py (+14/-17); vllm/model_executor/models/gemma.py (+13/-17); vllm/model_executor/models/gpt2.py (+14/-19); vllm/model_executor/models/gpt_bigcode.py (+14/-19); vllm/model_executor/models/gpt_j.py (+14/-18); vllm/model_executor/models/gpt_neox.py (+14/-18); vllm/model_executor/models/internlm2.py (+13/-17); vllm/model_executor/models/jais.py (+15/-20); vllm/model_executor/models/llama.py (+13/-17); vllm/model_executor/models/mixtral.py (+14/-18); vllm/model_executor/models/mixtral_quant.py (+14/-18); vllm/model_executor/models/mpt.py (+14/-18); vllm/model_executor/models/olmo.py (+13/-17); vllm/model_executor/models/opt.py (+17/-22); vllm/model_executor/models/orion.py (+13/-17); vllm/model_executor/models/phi.py (+14/-18); vllm/model_executor/models/qwen.py (+13/-18); (+7 more)
BODY: This PR is the second attempt to modularize the attention backends. The main goal of this PR is to hide any backend-specific attention implementation details from the main logic. This refactoring will greatly help introduce new backends, particularly the FlashInfer backend, which requires a different KV cache layout and input data structures from our current attention backends. **NOTE: Since this PR just re-organizes the code, it shouldn't affect the functionality or performance of vLLM.** ⏎  ⏎ This PR defines three main classes for each attention backend: `AttentionBackend`, `AttentionMetadata`, and `AttentionImpl`. `AttentionBackend` (which can be queried by `get_attn_backend`) is a static class that defines the KV cache layout, swapping ops, and also works as a dispatcher for `AttentionMetadata` and `AttentionImpl`. `AttentionMetadata` is same as the current `InputMetadata`, but it can be different for other backends added in the future. Finally, `AttentionImpl` is the actual implementation of the attention operator. ⏎  ⏎ While ultimately I'd like to move part of ModelRunner's `prepare_inptus` to `AttentionBackend`, I didn't do it to reduce the size of the PR.

### L3-9765b5c406  (L3, 2024-03-29, sha 9765b5c4061b, PR #3699)
TITLE: [ROCm][Bugfix] Fixed several bugs related to rccl path and attention selector logic (#3699)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.xformers.v0_backend,L3.dispatch.selector; ROCm bugfix corrected attention selector logic and xFormers backend checks.
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.upstream_pip
FILES: requirements-rocm.txt (+1/-1); vllm/attention/backends/xformers.py (+2/-2); Dockerfile.rocm (+1/-1); vllm/model_executor/parallel_utils/pynccl.py (+1/-1)
LABELS: rocm
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ This pull request fixes several bugs introduced in previous commits, for example: https://github.com/vllm-project/vllm/pull/3661, https://github.com/vllm-project/vllm/pull/3625 , and previous refactoring in attention backend. ⏎  ⏎ (1) Fixed the librccl.so file name, it should be something like: ⏎ /opt/rocm/lib/librccl.so.1 ⏎  ⏎ (2) a bug related to check whether to use ref-attention resulted from previous refactoring: ⏎  ⏎ Before: even flash-attn is available, it uses naive attention, which is quite slow for our users and is not intended. ⏎  ⏎ ``` ⏎ WARNING 03-28 18:26:49 xformers.py:410] flash_attn is not installed. Using naive attention. This will take significantly more GPU memory. ⏎ ``` ⏎ Now: ⏎ ``` ⏎ INFO 03-28 18:30:12 selector.py:29] Cannot use FlashAttention backend for AMD GPUs. ⏎ Using XFormers backend. ⏎ ``` ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-2ff767b513  (L3, 2024-04-03, sha 2ff767b51301, PR #3290)
TITLE: Enable scaled FP8 (e4m3fn) KV cache on ROCm (AMD GPU) (#3290)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L3.paged.cuda.v1,L3.cache.cuda_reshape,L3.flash_attn.v0_backend,L3.xformers.v0_backend; Added scaled FP8 E4M3 KV-cache support through kernels and backends.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.fork_inline_cmake, L3.dispatch.abstract_interface
FILES: csrc/attention/attention_dtypes.h (+1/-1); csrc/attention/attention_kernels.cu (+79/-44); csrc/attention/dtype_fp8.cuh (+1/-1); csrc/cache_kernels.cu (+42/-23); vllm/attention/backends/abstract.py (+1/-0); vllm/attention/backends/flash_attn.py (+7/-1); vllm/attention/backends/xformers.py (+7/-1); vllm/attention/layer.py (+3/-1); vllm/attention/ops/paged_attn.py (+5/-0); .gitignore (+1/-0); CMakeLists.txt (+1/-1); benchmarks/benchmark_latency.py (+16/-2); benchmarks/benchmark_throughput.py (+19/-3); benchmarks/kernels/benchmark_paged_attention.py (+10/-3); cmake/utils.cmake (+1/-0); csrc/cache.h (+3/-2); csrc/ops.h (+4/-2); csrc/pybind.cpp (+3/-3); csrc/quantization/fp8/amd_detail/hip_float8.h (+167/-0); csrc/quantization/fp8/amd_detail/hip_float8_impl.h (+316/-0); csrc/quantization/fp8/amd_detail/quant_utils.cuh (+517/-0); docs/source/index.rst (+2/-1); docs/source/quantization/fp8_e4m3_kvcache.rst (+49/-0); docs/source/quantization/fp8_e5m2_kvcache.rst (+5/-2); examples/fp8/README.md (+96/-0); examples/fp8/extract_scales.py (+367/-0); examples/fp8/quantizer/README.md (+32/-0); examples/fp8/quantizer/quantize.py (+369/-0); pyproject.toml (+4/-0); tests/fp8_kv/llama2-70b-fp8-kv/kv_cache_scales.json (+90/-0); tests/fp8_kv/llama2-7b-fp8-kv/kv_cache_scales.json (+42/-0); tests/kernels/test_attention.py (+11/-5); tests/kernels/test_cache.py (+86/-12); vllm/config.py (+20/-14); vllm/engine/arg_utils.py (+18/-5); vllm/engine/llm_engine.py (+1/-0); vllm/model_executor/layers/quantization/schema.py (+84/-0); vllm/model_executor/models/llama.py (+39/-3); vllm/model_executor/weight_utils.py (+42/-1); vllm/utils.py (+10/-10); (+1 more)
LABELS: rocm
PERF_LINES: - Enabled on AMD MI3xx GPUs, MI300x (192GB HBM) in particular (less performant on older silicons without FP8 HW) | We observed 20~30% performance increases from FP16 baseline by just turning KV cache to FP8 (e4m3fn), even on the 70B model served by a single MI300X. | ***WizardCoder-34b score, dataset: HumanEval-Python-EN on 1-GPU MI300X*** | |     FP16       (T=0.8)       |        30.63%       |  
BODY: As part of a series of FP8 development in vLLM, we address an [OCP](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1) format (nVIDIA compatible) FP8 KV cache in this pull request. We elaborated upon previous [#2279](https://github.com/vllm-project/vllm/pull/2279), but made following change, enhancement and extensions: ⏎ - Using OCP FP8 data type, E4M3 recommended for inference (as Float8E4M3FN in MLIR, float8_e4m3fn in PyTorch) ⏎ - Using scaled FP8 KV cache, to mitigate quantization loss (scaling factors are aquired from AMD quantizer, AMMO, etc.) ⏎ - Enabled on AMD MI3xx GPUs, MI300x (192GB HBM) in particular (less performant on older silicons without FP8 HW) ⏎  ⏎ ### Design reference: ⏎ - RFC: FP8 Quantization Schema in vLLM [#3218](https://github.com/vllm-project/vllm/discussions/3218) ⏎ - RFC: FP8 in vLLM [#2461](https://github.com/vllm-project/vllm/discussions/2461) ⏎  ⏎ ### Scope: ⏎ - Used in conjunction with Quantizer's output: KV cache scaling factors. For this phase, not include the `activation` and `weights` sections from the JSON schema proposed in [#2461](https://github.com/vllm-project/vllm/discussions/2461). Quantizer's output may need to be formatted to that schema based JSON file for vLLM code to consider, an utility script (`3rdparty/quantizer/extract_scales.py`) is provided for JSON generation from AMMO's output. ⏎ - Quantizer supported: AMD Quantizer, nVIDIA AMMO. an utility script (`3rdparty/quantizer/quantize.py`) is provided for using AMMO to quantize HF model to FP8 with FP8 KV cache s.t. KV cache scaling factors will be generated (over a calibartion dataset, which you can change to your domain of interests), details in `3rdparty/README.md`. ⏎ - Only the common OCP format used for FP8 inference and model forward/eval `e4m3fn` is enabled, this comes with HW support (so performant) on AMD MI3xx GPUs. Same design is still functional but less performant on earlier AMD GPUs, current design does not cover CUDA device. ⏎ - Model: Llama first, others will be added later after approval. ⏎ - FP8 KV cache only, with scaling, FP8 compute coming next. ⏎  ⏎ ### Scaling semantics: ⏎ - In concept and this design, we have following definition: ⏎   ``` ⏎   scaling_factor = AbsMax(input_tensor_fp16_or_bfloat16_or_fp32) / (OCP_E4M3_MAXNORM = 448.0)  ⏎   ``` ⏎ - This semantics is used by AMD quantizer, and AMMO upon observation. ⏎ - `scaled_to_fp8_quant:   fp8_tensor = fp8_quant(higher_precision_tensor / scaling_factor)` ⏎ - `scaled_fr_fp8_dequant: higher_precision_tensor = fp8_dequant(fp8_tensor) * scaling facto …[truncated]

### L3-6c0b04515f  (L3, 2024-04-09, sha 6c0b04515fee, PR #3643)
TITLE: [ROCm][Hardware][AMD] Use Triton Kernel for default FA on ROCm (#3643)
SOURCES: path_core, symbol_pickaxe
STAGE1: introduce; artifacts=L3.triton.flash_attention_rocm,L3.rocm.rocm_flash_attn_v0,L3.dispatch.selector; Added ROCm Triton FlashAttention backend and made it selectable/default on ROCm.
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.upstream_pip, L3.triton.flash_attention_rocm, L3.rocm.rocm_flash_attn_v0, L3.dispatch.selector
FILES: vllm/attention/backends/rocm_flash_attn.py (+348/-0); vllm/attention/backends/xformers.py (+2/-76); vllm/attention/ops/triton_flash_attention.py (+809/-0); vllm/attention/selector.py (+40/-17); Dockerfile.rocm (+14/-0)
LABELS: rocm
BODY: This PR creates and makes default new triton exclusive backend for attention. Additionally removes some unsupported arguments from AMD's version of the `flash_attn_varlen_func` function. ⏎  ⏎ - Changed selector to allow picking between multiple backends rather than just between two. ⏎ - Added a new triton FA backend available to ROCm.  ⏎ - Added new `VLLM_USE_FLASH_ATTN_TRITON` option to be able to swap between Triton and Default FA ⏎ - Removed unsupported attributes from AMD's `flash_attn_varlen_func` ⏎ - Added latest build of AMD's triton to Dockerfile.rocm ⏎  ⏎ Appreciate any and all feedback, and apologies for the long PR ⏎  ⏎  ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-8b317c6dd0  (L3, 2024-04-10, sha 8b317c6dd09c, PR #3972)
TITLE: [Model][AMD] ROCm support for 256 head dims for Gemma (#3972)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.triton.flash_attention_rocm; ROCm Triton FlashAttention gained 256-head-dimension support for Gemma.
ARTIFACT_HINTS: L3.triton.flash_attention_rocm
FILES: vllm/attention/ops/triton_flash_attention.py (+2/-3)
LABELS: rocm
ISSUES: #3073 Serving for Google Gemma model failing on AMD MI 300X GPUs
BODY: Thanks to @jpvillam-amd's contributions on https://github.com/vllm-project/vllm/pull/3643, Gemma should have ROCm support ⏎  ⏎ In my testing, I had to make these additional but trivial tweaks to actually use google/gemma-2b-it ⏎  ⏎ This was tested on an MI100 ⏎  ⏎ FIX #3073 (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-67b4221a61  (L3, 2024-04-10, sha 67b4221a61ac, PR #3884)
TITLE: [Core][5/N] Fully working chunked prefill e2e (#3884)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L3.dispatch.abstract_interface,L3.flash_attn.v0_backend,L3.xformers.v0_backend,L3.rocm.rocm_flash_attn_v0; Chunked prefill unified attention metadata across prefill/decode backends.
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+39/-3); vllm/attention/backends/flash_attn.py (+55/-30); vllm/attention/backends/rocm_flash_attn.py (+63/-34); vllm/attention/backends/torch_sdpa.py (+42/-25); vllm/attention/backends/xformers.py (+79/-59); vllm/attention/layer.py (+3/-2); vllm/attention/ops/paged_attn.py (+0/-6); .buildkite/test-pipeline.yaml (+2/-0); benchmarks/benchmark_latency.py (+1/-2); benchmarks/benchmark_throughput.py (+38/-24); tests/basic_correctness/test_chunked_prefill.py (+70/-0); tests/core/test_chunked_prefill_scheduler.py (+8/-8); tests/distributed/test_basic_distributed_correctness.py (+6/-1); tests/distributed/test_chunked_prefill_distributed.py (+66/-0); tests/entrypoints/test_openai_server.py (+1/-1); tests/models/test_models.py (+1/-1); tests/worker/test_model_runner.py (+170/-19); vllm/attention/__init__.py (+3/-1); vllm/config.py (+10/-3); vllm/core/scheduler.py (+9/-6); vllm/distributed/communication_op.py (+9/-1); vllm/engine/arg_utils.py (+2/-3); vllm/engine/llm_engine.py (+4/-1); vllm/lora/layers.py (+3/-2); vllm/sequence.py (+2/-1); vllm/worker/model_runner.py (+241/-82)
BODY: This PR is a part of the RFC https://github.com/vllm-project/vllm/issues/3130. ⏎  ⏎ This PR enables chunked prefill e2e. Note that chunked prefill is an experimental feature now (though it is actively used within Anyscale), and I will start serious benchmark with this PR.  ⏎  ⏎ The feature can be enabled by using `enable_chunked_prefill`. The chunking is done based on `max_num_batched_tokens` (a.k.a token budget). This is the same way as described in SARATHI paper https://arxiv.org/abs/2308.16369. It also means we can have up to maximum 2 chunked prefill at any given time. ⏎  ⏎ This PR ⏎ - Allow to put 2 attention metadata, one for prefill and one for decode. ⏎     - it is done by broadcasting metadata twice. It can be more optimized by coelescing tensors better, but I made this way for simplicity. The normal path should just use 1 broadcast as usual.  ⏎ - Allow to run attention backend when there are prefill and decode mixed up. ⏎ - Ignore the generated token if chunked prefill is enabled. ⏎ - Add a chunked prefill test with various chunk size, cuda graph, and tp settings. ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-e9da5a40c6  (L3, 2024-04-10, sha e9da5a40c63c, PR #3913)
TITLE: [Misc] Add indirection layer for custom ops  (#3913)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L3.paged.python_wrapper; Custom-op indirection changed how paged attention/cache wrappers invoke compiled ops.
ARTIFACT_HINTS: L3.paged.python_wrapper
FILES: vllm/attention/ops/paged_attn.py (+5/-5); benchmarks/kernels/benchmark_paged_attention.py (+1/-1); tests/kernels/test_attention.py (+3/-3); tests/kernels/test_cache.py (+12/-13); vllm/_custom_ops.py (+193/-0); vllm/model_executor/layers/activation.py (+1/-1); vllm/model_executor/layers/fused_moe/fused_moe.py (+1/-1); vllm/model_executor/layers/layernorm.py (+1/-1); vllm/model_executor/layers/quantization/awq.py (+1/-1); vllm/model_executor/layers/quantization/gptq.py (+1/-1); vllm/model_executor/layers/quantization/marlin.py (+1/-1); vllm/model_executor/layers/quantization/squeezellm.py (+1/-1); vllm/model_executor/layers/rotary_embedding.py (+1/-1); vllm/utils.py (+2/-2)
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ Refactor `ops` and `cache_ops` layer. Add an abstraction ops layer and will use `vllm._C.ops` by default. ⏎ This would be easier to add/extend other third party high performance ops/kernels implementation if necessary. ⏎  ⏎  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-d04973ad54  (L3, 2024-04-12, sha d04973ad5446, PR #3984)
TITLE: Fix triton compilation issue (#3984)
SOURCES: path_core
STAGE1: repair_build_dependency; artifacts=L3.triton.flash_attention_rocm; Fixed Triton compilation error in ROCm FlashAttention op.
ARTIFACT_HINTS: L3.triton.flash_attention_rocm
FILES: vllm/attention/ops/triton_flash_attention.py (+5/-1)
LABELS: rocm
BODY: Solves the following compilation error with triton flash attention backend enabled. ⏎  ⏎ .../vllm/attention/ops/triton_flash_attention.py: ⏎  ⏎ ... ⏎ triton.compiler.errors.UnsupportedLanguageConstruct: at 120:14:            #          + offs_m ⏎             # We store inf to LSE, not -inf because in the bwd pass, ⏎             # we subtract this ⏎             # from qk which makes it -inf, such that exp(qk - inf) = 0 ⏎             # for these masked blocks. ⏎             # l = tl.full([BLOCK_M], value=float("inf"), dtype=tl.float32) ⏎             # tl.store(l_ptrs, l) ⏎             # TODO: Should dropout and return encoded softmax be handled here? ⏎             return ⏎     is_mqa = hq != hk ⏎     off_h_k = off_h_q % hk if is_mqa else off_h_q ⏎               ^ ⏎ Triton does not support `if` expressions (ternary operators) with dynamic conditions, use `if` statements instead ⏎ ... ⏎  ⏎ FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-e8cc7967ff  (L3, 2024-04-18, sha e8cc7967ff8a, PR #4128)
TITLE: [Bugfix][Kernel] allow non-power-of-two head sizes in prefix prefill (#4128)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L3.triton.prefix_prefill; Prefix-prefill kernel padded non-power-of-two head sizes to support more shapes.
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/prefix_prefill.py (+27/-17); tests/kernels/test_prefix_prefill.py (+1/-1)
ISSUES: #4127 [Bug][Chunked prefill]: head size has to be power of two
BODY: The existing prefix prefill kernel only supports head dimension that is a power of two. This due to Triton only supporting power of two block sizes. This PR enlarges the Q,K,V tensors to the next power of two and pads them with zeros when reading (and writing). ⏎  ⏎ It doesn't seem to affect performance of the non-padded case. ⏎  ⏎ CC @rkooo567  ⏎  ⏎ FIX #4127  ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-95e5b087cf  (L3, 2024-04-21, sha 95e5b087cfed, PR #4129)
TITLE: [AMD][Hardware][Misc][Bugfix] xformer cleanup and light navi logic and CI fixes and refactoring (#4129)
SOURCES: path_core, symbol_pickaxe, dependency_pin
STAGE1: change_default; artifacts=L3.rocm.rocm_flash_attn_v0,L3.triton.flash_attention_rocm; ROCm backend logic added Navi routing and removed xFormers patches/selection.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.rocm.rocm_flash_attn_v0
FILES: Dockerfile.rocm (+1/-4); vllm/attention/backends/rocm_flash_attn.py (+18/-13); .buildkite/test-pipeline.yaml (+0/-2); patch_xformers.rocm.sh (+0/-33); rocm_patch/commonpy_xformers-0.0.23.rocm.patch (+0/-13); rocm_patch/flashpy_xformers-0.0.23.rocm.patch (+0/-152)
LABELS: rocm
PERF_LINES: (2) to facilitate gfx1100/navi3x to use triton flash-attn.
BODY: This PR is  ⏎ (1) to remove xformer package and patches since right now raw flash attention api is called instead of using xformers wrapper. ⏎ (2) to facilitate gfx1100/navi3x to use triton flash-attn. ⏎ (3) still make it possible for other gfx target to use flash-attn and updated the flash-attention branch. ⏎  ⏎ (4) fixed the CI failure for "basic correctness" issue on cuda env ⏎  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-18d23f642a  (L3, 2024-04-26, sha 18d23f642af9, PR #4406)
TITLE: [ROCm][Hardware][AMD] Enable group query attention for triton FA (#4406)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L3.triton.flash_attention_rocm,L3.rocm.rocm_flash_attn_v0; ROCm Triton FlashAttention backend gained grouped-query attention support.
ARTIFACT_HINTS: L3.triton.flash_attention_rocm, L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+25/-28); vllm/attention/ops/triton_flash_attention.py (+11/-13)
LABELS: rocm
BODY: Enable group-query-attention for Triton flash attention ⏎  ⏎ cc Vinayak Gokhale @vgokhale , @Alexei-V-Ivanov-AMD  ⏎  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-32881f3f31  (L3, 2024-05-02, sha 32881f3f3106, PR #4405)
TITLE: [kernel] fix sliding window in prefix prefill Triton kernel (#4405)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.triton.prefix_prefill; Prefix-prefill sliding-window masking was fixed to avoid NaNs and support sliding window.
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.triton.prefix_prefill, L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/flash_attn.py (+1/-0); vllm/attention/backends/rocm_flash_attn.py (+1/-0); vllm/attention/backends/xformers.py (+1/-0); vllm/attention/ops/paged_attn.py (+2/-0); vllm/attention/ops/prefix_prefill.py (+56/-19); tests/kernels/test_prefix_prefill.py (+30/-4)
ISSUES: #4057 [Feature][Chunked prefill]: Make sliding window work
BODY: This adds support for the sliding window in prefix prefill kernel. ⏎  ⏎ I had to use a large negative value instead of -inf for masking, since otherwise in some situations we get '-inf - -inf' in softmax which leads to NaNs. ⏎  ⏎ Added tests comparing with xformers. ⏎  ⏎ Also added a bunch of comments with tensor shapes etc. ⏎  ⏎ FIX #4057 ⏎  ⏎ CC @rkooo567 @cadedaniel @simon-mo  ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-3521ba4f25  (L3, 2024-05-03, sha 3521ba4f2554, PR #4518)
TITLE: [Core][Model runner refactoring 1/N] Refactor attn metadata term (#4518)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.dispatch.abstract_interface; Model-runner refactor renamed/restructured attention metadata terms consumed by backends.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.rocm.rocm_flash_attn_v0
FILES: csrc/attention/attention_kernels.cu (+38/-38); csrc/cpu/attention.cpp (+46/-46); vllm/attention/backends/flash_attn.py (+22/-22); vllm/attention/backends/rocm_flash_attn.py (+30/-30); vllm/attention/backends/torch_sdpa.py (+18/-18); vllm/attention/backends/xformers.py (+32/-33); vllm/attention/ops/paged_attn.py (+17/-18); benchmarks/kernels/benchmark_paged_attention.py (+12/-13); csrc/ops.h (+4/-4); tests/kernels/test_attention.py (+17/-18); tests/kernels/test_prefix_prefill.py (+8/-8); tests/samplers/test_sampler.py (+17/-17); tests/spec_decode/e2e/conftest.py (+2/-2); tests/spec_decode/test_multi_step_worker.py (+14/-10); tests/spec_decode/test_ngram_worker.py (+15/-9); tests/spec_decode/utils.py (+4/-4); tests/test_logits_processor.py (+4/-4); tests/worker/test_model_runner.py (+48/-51); vllm/_custom_ops.py (+9/-9); vllm/config.py (+16/-7); vllm/engine/arg_utils.py (+12/-2); vllm/entrypoints/llm.py (+6/-1); vllm/model_executor/layers/sampler.py (+3/-3); vllm/model_executor/sampling_metadata.py (+36/-27); vllm/worker/cpu_model_runner.py (+29/-29); vllm/worker/model_runner.py (+80/-87); vllm/worker/neuron_model_runner.py (+15/-15)
BODY: RFC: https://docs.google.com/document/d/1rg8CoOnrtz1LT-hCK86ZsHuhoTDtqSEGs8KrN4wbITo/edit#heading=h.uwasieoo42mu ⏎  ⏎ <img width="671" alt="Screenshot 2024-05-01 at 5 17 02 PM" src="https://github.com/vllm-project/vllm/assets/18510752/7f69a5a4-4133-4f7a-9aa0-92badafdd9f9"> ⏎  ⏎ - prompt: prompt tokens for seq group. ⏎ - subquery_len -> query_len: The new tokens to compute. (1 token for decode, new chunked tokens for chunked prefill)/. ⏎ - computed_len -> context_len: The # of tokens computed in the past iteration ⏎ - context_len -> seqlen: # of computed tokens + new query tokens ⏎  ⏎ This also unifies the definition of context_len in decode/prefill. (i.e., decode's context_len now becomes seqlen) ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-43c413ec57  (L3, 2024-05-03, sha 43c413ec570e, PR #4353)
TITLE: [Kernel] Use flashinfer for decoding (#4353)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: integrate; artifacts=L3.flashinfer.v0_backend; Added FlashInfer decode backend for attention.
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flashinfer.v0_backend, L3.dispatch.selector, L3.dispatch.abstract_interface
FILES: csrc/cache_kernels.cu (+80/-0); vllm/_custom_ops.py (+12/-0); vllm/attention/backends/abstract.py (+9/-4); vllm/attention/backends/flashinfer.py (+220/-0); vllm/attention/selector.py (+6/-0); vllm/config.py (+5/-0); vllm/utils.py (+52/-15); vllm/worker/model_runner.py (+97/-26); csrc/cache.h (+8/-0); csrc/pybind.cpp (+4/-0); tests/basic_correctness/test_basic_correctness.py (+11/-1); tests/distributed/test_basic_distributed_correctness.py (+9/-5); tests/kernels/conftest.py (+7/-1); tests/kernels/test_cache.py (+77/-0); vllm/sequence.py (+3/-1)
BODY: This PR is a first attempt to integrate [flashinfer](https://flashinfer.ai/) for the decoding phase. The PR still uses flash attention for the prefill phase for now. ⏎  ⏎ Updated after discussion with @yzh119 ⏎ Things need to be fixed: ⏎  ⏎  ⏎ Next step:

### L3-0f9a6e3d22  (L3, 2024-05-08, sha 0f9a6e3d229c, PR #4573)
TITLE: [Bugfix][Kernel] allow non-power-of-2 for prefix prefill with alibi  (#4573)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.triton.prefix_prefill; Prefix-prefill with ALiBi gained non-power-of-two support fix.
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/prefix_prefill.py (+25/-16); tests/kernels/test_prefix_prefill.py (+242/-1)
ISSUES: #4171 [Bug]: Server crash for bloom-3b while use prefix_caching, `AssertionError assert Lk in {16, 32, 64, 128}`
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/4171 ⏎  ⏎ allow non-power-of-two head sizes in prefix prefill with alibi, this is a small fix based on https://github.com/vllm-project/vllm/pull/4128. ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-5510cf0e8a  (L3, 2024-05-08, sha 5510cf0e8a6a, PR #4685)
TITLE: [Misc] Add `get_name` method to attention backends (#4685)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: adapt_framework; artifacts=L3.dispatch.abstract_interface; Attention backends gained get_name API method.
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+5/-0); vllm/attention/backends/flash_attn.py (+4/-0); vllm/attention/backends/flashinfer.py (+7/-9); vllm/attention/backends/rocm_flash_attn.py (+4/-0); vllm/attention/backends/torch_sdpa.py (+4/-0); vllm/attention/backends/xformers.py (+4/-0); vllm/worker/model_runner.py (+2/-3)
BODY: This PR adds the name strs to the attention backends so that we can use the names to identify them (without importing the actual class).

### L3-89579a201f  (L3, 2024-05-08, sha 89579a201f2c, PR #4686)
TITLE: [Misc] Use vllm-flash-attn instead of flash-attn (#4686)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
STAGE1: replace; artifacts=L3.flash_attn.fork_pip,L3.flash_attn.upstream_pip; CUDA dependency switched from upstream flash-attn to vllm-flash-attn wheel.
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.upstream_pip, L3.flash_attn.fork_pip, L3.flashinfer.v0_backend, L3.dispatch.selector
FILES: Dockerfile (+0/-21); requirements-cuda.txt (+1/-0); setup.py (+9/-5); vllm/attention/backends/flash_attn.py (+1/-1); vllm/attention/backends/flashinfer.py (+1/-1); vllm/attention/selector.py (+4/-3)
BODY: This PR is to use the pre-built `vllm-flash-attn` wheel instead of the original `flash-attn`.

### L3-0ee535b294  (L3, 2024-05-09, sha 0ee535b2945d, PR #4705)
TITLE: [Misc] Set block size at initialization & Fix test_model_runner (#4705)
SOURCES: symbol_pickaxe
STAGE1: adapt_framework; artifacts=L3.paged.python_wrapper; Block size was moved to initialization, changing attention wrapper/model-runner contract.
ARTIFACT_HINTS: -
FILES: tests/worker/test_model_runner.py (+32/-58); vllm/worker/cpu_model_runner.py (+9/-12); vllm/worker/cpu_worker.py (+1/-0); vllm/worker/model_runner.py (+21/-33); vllm/worker/worker.py (+1/-1)
BODY: This PR sets the block size to `ModelRunner` at its initialization time, instead of setting it lazily. Currently, we don't have any reason to set this value lazily, since it is configured by the user argument. ⏎ Also, the PR fixes the abuse of `ModelRunner` in `test_model_runner` where basically `ModelRunner` was initialized with `None` arguments.

### L3-c833101740  (L3, 2024-05-09, sha c83310174055, PR #4535)
TITLE: [Kernel] Refactor FP8 kv-cache with NVIDIA float8_e4m3 support (#4535)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L3.cache.cuda_reshape,L3.paged.cuda.v1; FP8 KV-cache support was refactored for NVIDIA float8_e4m3.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.flash_attn.fork_inline_cmake
FILES: csrc/attention/attention_kernels.cu (+110/-176); csrc/attention/dtype_fp8.cuh (+11/-5); csrc/cache_kernels.cu (+70/-73); .buildkite/check-wheel-size.py (+1/-1); CMakeLists.txt (+1/-1); cmake/utils.cmake (+2/-2); csrc/cache.h (+3/-1); csrc/quantization/fp8/amd/hip_float8.h (+0/-0); csrc/quantization/fp8/amd/hip_float8_impl.h (+0/-0); csrc/quantization/fp8/amd/quant_utils.cuh (+58/-1); csrc/quantization/fp8/common.cu (+0/-0); csrc/quantization/fp8/nvidia/quant_utils.cuh (+568/-0); csrc/quantization/fp8_e5m2_kvcache/quant_utils.cuh (+0/-277); tests/kernels/test_attention.py (+2/-2); tests/kernels/test_cache.py (+11/-16); vllm/_custom_ops.py (+5/-2); vllm/utils.py (+1/-1)
BODY: The first PR for #4532. ⏎  ⏎ Task list: ⏎  ⏎  ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-0fca3cdcf2  (L3, 2024-05-13, sha 0fca3cdcf265, PR #4751)
TITLE: [Misc] Enhance attention selector (#4751)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: change_default; artifacts=L3.dispatch.selector; Attention selector was enhanced with new backend selection/fallback logic.
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.selector, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+2/-3); vllm/attention/backends/flash_attn.py (+7/-6); vllm/attention/backends/flashinfer.py (+23/-10); vllm/attention/backends/rocm_flash_attn.py (+9/-7); vllm/attention/backends/torch_sdpa.py (+17/-11); vllm/attention/backends/xformers.py (+6/-6); vllm/attention/layer.py (+17/-2); vllm/attention/selector.py (+23/-5); vllm/model_executor/models/deepseek.py (+13/-3); vllm/model_executor/models/gemma.py (+10/-4); vllm/model_executor/models/llama.py (+13/-4); tests/worker/test_model_runner.py (+0/-1); vllm/attention/__init__.py (+2/-2); vllm/model_executor/model_loader/__init__.py (+11/-8); vllm/model_executor/model_loader/loader.py (+39/-22); vllm/model_executor/models/arctic.py (+13/-3); vllm/model_executor/models/baichuan.py (+22/-7); vllm/model_executor/models/bloom.py (+11/-4); vllm/model_executor/models/chatglm.py (+14/-6); vllm/model_executor/models/commandr.py (+11/-3); vllm/model_executor/models/dbrx.py (+13/-4); vllm/model_executor/models/decilm.py (+3/-1); vllm/model_executor/models/falcon.py (+11/-4); vllm/model_executor/models/gpt2.py (+12/-4); vllm/model_executor/models/gpt_bigcode.py (+10/-4); vllm/model_executor/models/gpt_j.py (+15/-5); vllm/model_executor/models/gpt_neox.py (+12/-4); vllm/model_executor/models/internlm2.py (+10/-3); vllm/model_executor/models/jais.py (+9/-3); vllm/model_executor/models/llava.py (+4/-2); vllm/model_executor/models/minicpm.py (+10/-3); vllm/model_executor/models/mixtral.py (+11/-2); vllm/model_executor/models/mixtral_quant.py (+21/-10); vllm/model_executor/models/mpt.py (+13/-5); vllm/model_executor/models/olmo.py (+10/-4); vllm/model_executor/models/opt.py (+12/-4); vllm/model_executor/models/orion.py (+10/-3); vllm/model_executor/models/phi.py (+12/-4); vllm/model_executor/models/qwen.py (+12/-3); vllm/model_executor/models/qwen2.py (+10/-4); (+9 more)
BODY: This PR is to provide more information (such as block size and kv cache dtype) to attention backend selector so that it can be used to find the appropriate attention backend. Also, the PR moves `kv_cache_dtype` from `AttentionMetadata` to `Attention`. ⏎  ⏎ This PR is a prerequisite for #3648

### L3-1356df53bd  (L3, 2024-05-13, sha 1356df53bd5d, PR #3648)
TITLE: [Kernel] Use flash-attn for decoding (#3648)
SOURCES: path_core, symbol_pickaxe
STAGE1: integrate; artifacts=L3.flash_attn.v0_backend; FlashAttention backend added decode path using flash-attn over paged KV cache.
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.dispatch.selector
FILES: vllm/attention/backends/flash_attn.py (+73/-55); vllm/attention/selector.py (+14/-0); tests/kernels/test_flash_attn.py (+209/-0); tests/models/test_big_models.py (+1/-1); tests/models/test_fp8.py (+5/-5); vllm/worker/model_runner.py (+11/-4)
BODY: Vendors flash-attention from https://github.com/Dao-AILab/flash-attention/pull/824, prunes out the backward pass operator for faster compile times, adds reshape and cache kernel for flash attention kv cache layout, adds logic for selecting kv cache manager / attention backend based on temporary environment variable VLLM_TEMP_USE_FLASH_DECODE. Tested for single GPU on opt-125m, llama-7b

### L3-8a7cc254a0  (L3, 2024-05-15, sha 8a7cc254a064, PR #4820)
TITLE: Revert "[Kernel] Use flash-attn for decoding (#3648)" (#4820)
SOURCES: path_core, symbol_pickaxe
STAGE1: revert; artifacts=L3.flash_attn.v0_backend; Reverted flash-attn decode path after illegal-memory-access failures.
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.dispatch.selector
FILES: vllm/attention/backends/flash_attn.py (+55/-73); vllm/attention/selector.py (+0/-14); tests/kernels/test_flash_attn.py (+0/-209); tests/models/test_big_models.py (+1/-1); tests/models/test_fp8.py (+5/-5); vllm/worker/model_runner.py (+4/-11)
BODY: Lora 3 & 4 test seems to have illegal memory access failure after this commit; ⏎  ⏎ ``` ⏎ [2024-05-14 23:51:18,182 E 22 22] logging.cc:101: Unhandled exception: N3c105ErrorE. what(): CUDA error: an illegal memory access was encountered ⏎ <br class="Apple-interchange-newline"> ⏎ ``` ⏎  ⏎ Exmaple: https://buildkite.com/vllm/ci/builds/7382#018f793d-1527-4e1c-ab59-c3a34ec55241 ⏎  ⏎ This reverts commit 1356df53bd5d6877358aff3d2bbd95f28f8009a4. ⏎  ⏎ FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-65bf2ac165  (L3, 2024-05-15, sha 65bf2ac16573, PR #4681)
TITLE: [Core][2/N] Model runner refactoring part 2. Combine prepare prefill / decode to a single API (#4681)
SOURCES: path_core, body_keyword, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L3.dispatch.abstract_interface,L3.flash_attn.v0_backend,L3.flashinfer.v0_backend; Attention backend API combined prepare_prefill and prepare_decode metadata handling.
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+32/-36); vllm/attention/backends/flash_attn.py (+79/-16); vllm/attention/backends/flashinfer.py (+28/-10); vllm/attention/backends/rocm_flash_attn.py (+80/-18); vllm/attention/backends/torch_sdpa.py (+22/-6); vllm/attention/backends/xformers.py (+77/-15); vllm/attention/layer.py (+2/-3); vllm/attention/ops/paged_attn.py (+5/-5); tests/worker/test_model_runner.py (+84/-39); vllm/attention/__init__.py (+2/-3); vllm/engine/arg_utils.py (+5/-0); vllm/model_executor/layers/rejection_sampler.py (+1/-0); vllm/sequence.py (+2/-1); vllm/spec_decode/batch_expansion.py (+16/-7); vllm/spec_decode/multi_step_worker.py (+1/-0); vllm/worker/cpu_model_runner.py (+3/-7); vllm/worker/embedding_model_runner.py (+15/-115); vllm/worker/model_runner.py (+323/-449)
BODY: This PR combines prepare_prompt and prepare_decode into a single API. This PR also coelsce the attn metadata for prefill/decode to a single class and allow to slice them when running attn backend.  ⏎  ⏎ It also refactors subquery_start_loc which was not refactored in the previous PR ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-b5853f9963  (L3, 2024-05-16, sha b5853f99639a, PR #4845)
TITLE: [ROCm][AMD][Bugfix] adding a missing triton autotune config (#4845)
SOURCES: path_core
STAGE1: retune; artifacts=L3.triton.flash_attention_rocm; Added missing Triton autotune config for ROCm FlashAttention.
ARTIFACT_HINTS: L3.triton.flash_attention_rocm
FILES: vllm/attention/ops/triton_flash_attention.py (+10/-0)
LABELS: rocm
BODY: Vinayak Gokhale @vgokhale found that there is a missing triton autotune config in vllm, but was added to triton repo at some point of time.  ⏎ This small change however causes a series of cascading events which results in autotune picking a different config - the one that didn't exist in vllm. Without it, vllm's autotune picks a much worse config. This is to fix that. ⏎  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-9a31a817a8  (L3, 2024-05-16, sha 9a31a817a85a, PR #4869)
TITLE: [Bugfix] Fix FP8 KV cache support (#4869)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.flash_attn.v0_backend,L3.flashinfer.v0_backend,L3.rocm.rocm_flash_attn_v0; Fixed kv_cache_dtype argument propagation into attention backends.
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/flash_attn.py (+5/-5); vllm/attention/backends/flashinfer.py (+5/-5); vllm/attention/backends/rocm_flash_attn.py (+5/-5); vllm/attention/backends/torch_sdpa.py (+5/-5); vllm/attention/backends/xformers.py (+5/-5); vllm/attention/layer.py (+1/-1)
BODY: This PR fixes a bug introduced in #4751 that the `kv_cache_dtype` arg was not correctly passed to the `__init__` methods of the attention backends.

### L3-c0724fc915  (L3, 2024-05-18, sha c0724fc91503, PR #4658)
TITLE: [ROCm][Hardware][AMD] Adding Navi21 to fallback to naive attention if Triton is not used (#4658)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: change_default; artifacts=L3.rocm.rocm_flash_attn_v0; ROCm backend added Navi21 fallback to naive attention when Triton is unavailable.
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+3/-2)
LABELS: rocm
PERF_LINES: Navi3X - have major HW version 11, Navi21/Navi10 have HW version 10, MI series - HW version 9.
BODY: Navi3X - have major HW version 11, Navi21/Navi10 have HW version 10, MI series - HW version 9. ⏎  ⏎ https://github.com/ROCm/FasterTransformer-Internal/issues/247

### L3-b57e6c5949  (L3, 2024-05-19, sha b57e6c59491e, PR #4907)
TITLE: [Kernel] Add flash-attn back (#4907)
SOURCES: path_core, symbol_pickaxe, dependency_pin
STAGE1: reland; artifacts=L3.flash_attn.v0_backend,L3.flash_attn.fork_pip; Re-landed flash-attn decode after vllm-flash-attn fixed paged-cache index overflow.
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.fork_pip, L3.dispatch.selector
FILES: requirements-cuda.txt (+1/-1); vllm/attention/backends/flash_attn.py (+75/-54); vllm/attention/selector.py (+14/-0); tests/kernels/test_flash_attn.py (+208/-0); tests/models/test_big_models.py (+1/-1); tests/models/test_fp8.py (+5/-5)
BODY: This PR reverts #4820 by adding back `flash-attn`. Previously, using `flash-attn` for decoding caused errors when using small models (like the Llama 68M model in `lora/test_layer_variation.py`). This was because the index calculation for paged KV cache was done in `int` instead of `int64_t`, leading to integer overflow when `num_blocks` is large. In `vllm-flash-attn==2.5.8.post2`, the overflow bug was fixed.

### L3-99eff67ba9  (L3, 2024-05-21, sha 99eff67ba915, PR #4944)
TITLE: [Bugfix][Kernel] Add head size check for attention backend selection (#4944)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L3.flash_attn.v0_backend,L3.dispatch.selector; Selector added head-size guard so unsupported FlashAttention shapes fall back.
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.dispatch.selector
FILES: vllm/attention/backends/flash_attn.py (+8/-4); vllm/attention/selector.py (+13/-3)
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ Previous #4886 PR cause the lora-test failing due to `Phi-2` with LoRA will make attention head_size to 80 while Flash Attention doesn't support it. ⏎  ⏎ - This PR fix the issues by adding a head size check when selecting attention backend. ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-a3a73ab069  (L3, 2024-05-22, sha a3a73ab0696b, PR #4893)
TITLE: [Misc] Load FP8 kv-cache scaling factors from checkpoints (#4893)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.cache.cuda_reshape,L3.paged.python_wrapper; Attention/KV-cache path gained loading of FP8 KV-cache scaling factors from checkpoints.
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+25/-2); benchmarks/benchmark_latency.py (+6/-8); benchmarks/benchmark_throughput.py (+5/-7); benchmarks/kernels/benchmark_paged_attention.py (+4/-6); tests/models/test_fp8.py (+52/-28); vllm/config.py (+3/-5); vllm/engine/arg_utils.py (+3/-4); vllm/model_executor/layers/quantization/fp8.py (+45/-2); vllm/model_executor/models/arctic.py (+2/-1); vllm/model_executor/models/baichuan.py (+4/-2); vllm/model_executor/models/bloom.py (+2/-1); vllm/model_executor/models/chatglm.py (+6/-7); vllm/model_executor/models/commandr.py (+6/-7); vllm/model_executor/models/dbrx.py (+6/-7); vllm/model_executor/models/deepseek.py (+2/-1); vllm/model_executor/models/falcon.py (+6/-3); vllm/model_executor/models/gemma.py (+2/-1); vllm/model_executor/models/gpt2.py (+2/-1); vllm/model_executor/models/gpt_bigcode.py (+2/-1); vllm/model_executor/models/gpt_j.py (+2/-1); vllm/model_executor/models/gpt_neox.py (+2/-1); vllm/model_executor/models/internlm2.py (+2/-1); vllm/model_executor/models/jais.py (+6/-7); vllm/model_executor/models/llama.py (+18/-14); vllm/model_executor/models/minicpm.py (+2/-1); vllm/model_executor/models/mixtral.py (+21/-8); vllm/model_executor/models/mixtral_quant.py (+7/-8); vllm/model_executor/models/mpt.py (+2/-1); vllm/model_executor/models/olmo.py (+2/-1); vllm/model_executor/models/opt.py (+2/-1); vllm/model_executor/models/orion.py (+2/-1); vllm/model_executor/models/phi.py (+2/-1); vllm/model_executor/models/qwen.py (+2/-1); vllm/model_executor/models/qwen2.py (+2/-1); vllm/model_executor/models/qwen2_moe.py (+2/-1); vllm/model_executor/models/stablelm.py (+2/-1); vllm/model_executor/models/starcoder2.py (+7/-8); vllm/model_executor/models/xverse.py (+2/-1); vllm/utils.py (+2/-0); vllm/worker/model_runner.py (+12/-5)
PERF_LINES: Model Dtype | kv-cache Dtype | GPU blocks | TTFT (ms) | ITL (ms) | e2e (s)
BODY: The 2nd PR for #4532. ⏎  ⏎ This PR supports loading FP8 kv-cache scaling factors from a FP8 checkpoint (with `.kv_scale` parameter). ⏎ Specifically, ⏎ 1. We now support `--kv-cache-dtype {auto, fp8, fp8_e4m3, fp8_e5m2}`. `auto=fp16 or bf16` and `fp8=fp8_e4m3`. ⏎ 2. If the checkpoint is in FP16, then kv-cache scaling factors can only be loaded via `--quantization-param-path`; otherwise kv-scale is always 1 regardless `fp8_e4m3` or `fp8_e5m2`. ⏎ 3. If the checkpoint is in FP8 (`e4m3`) AND `--kv-cache-dtype {fp8, fp8_e4m3}`, kv_scale will be loaded from the checkpoint (if the field presents). ⏎  ⏎ Here is a simple benchmark on a single NVIDIA L4 GPU: ⏎ - FP16 model: meta-llama/Meta-Llama-3-8B-Instruct  ⏎ - FP8 model: nm-testing/Meta-Llama-3-8B-Instruct-FP8 (no kv-scale so use 1.0 for now). ⏎ - QPS: 1 ⏎ - Total requests: 30 ⏎ - Average prompt length: 512 ⏎ - Average decoding length: 256 (`ignore_eos` is not enabled so the actual length may be differ). ⏎ - Temperature: 0 ⏎  ⏎ Model Dtype | kv-cache Dtype | GPU blocks | TTFT (ms) | ITL (ms) | e2e (s) ⏎ --|--|--|--|--|-- ⏎ FP16 | FP16 | 1470 | 219.0 | 90.5 | 15.7 ⏎ FP16 | FP8_e5m2 | 2940 | 219.3 | 88.2 | 14.9 ⏎ FP8_e4m3 | FP16 | 4784 | 148.2 | 51.3 | 9.3 ⏎ FP8_e4m3 | FP8_e4m3 | 9568 | 147.4 | 50.1 | 8.1 ⏎  ⏎ @robertgshaw2-neuralmagic @tlrmchlsmth can you help update nm-testing/Meta-Llama-3-8B-Instruct-FP8 to include kv-cache scaling factors so that we could test it? Thanks! ⏎  ⏎ TODO ⏎  ⏎  ⏎ Also cc @pcmoritz @Yard1  ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-ee3eea0a1b  (L3, 2024-05-23, sha ee3eea0a1b2c, PR #4960)
TITLE: [Misc] Take user preference in attention selector (#4960)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: change_default; artifacts=L3.dispatch.selector,L3.flashinfer.v0_backend; Attention selector began honoring user backend preference before automatic fallback.
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.dispatch.selector
FILES: vllm/attention/backends/flashinfer.py (+1/-0); vllm/attention/selector.py (+84/-61); tests/kernels/test_attention_selector.py (+84/-0)
BODY: The current selection logic on NVIDIA GPUs is a bit weird on NVIDIA GPUs. Specially, it checks whether the model can use FlashAttn, and directly falls back to xFormers if not, without considering consider user preference (i.e., environment variable `VLLM_ATTENTION_BACKEND`). User specified backend can only be used  when FlashAttn is valid for the model and installed in the environment. ⏎  ⏎ This PR refactors the logic to consider user preference first. ⏎  ⏎ In addition, this PR also fixes the issue that eager mode is not enforced when FlashInfer is used. This was introduced by #4353 but somehow removed by recent refactoring. Since the model runner refactoring is still ongoing, this PR simply refines the WARNING message to ask users manually specify `--enforce-eager`. ⏎  ⏎ cc @LiuXiaoxuanPKU @rkooo567 @WoosukKwon  ⏎  ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-8e192ff967  (L3, 2024-05-24, sha 8e192ff967b4, PR #4799)
TITLE: [Kernel][Backend][Model] Blocksparse flash attention kernel and Phi-3-Small model (#4799)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: introduce; artifacts=L3.blocksparse.v0; Introduced block-sparse attention backend/kernel and modified paged attention for hybrid sparsity.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: csrc/attention/attention_kernels.cu (+148/-37); csrc/cpu/attention.cpp (+21/-16); csrc/ops.h (+18/-17); vllm/_custom_ops.py (+21/-9); vllm/attention/backends/abstract.py (+1/-0); vllm/attention/backends/blocksparse_attn.py (+410/-0); vllm/attention/backends/flash_attn.py (+4/-1); vllm/attention/backends/rocm_flash_attn.py (+4/-1); vllm/attention/backends/torch_sdpa.py (+4/-1); vllm/attention/backends/xformers.py (+4/-1); vllm/attention/layer.py (+7/-3); vllm/attention/ops/blocksparse_attention/__init__.py (+0/-0); vllm/attention/ops/blocksparse_attention/blocksparse_attention_kernel.py (+423/-0); vllm/attention/ops/blocksparse_attention/interface.py (+238/-0); vllm/attention/ops/blocksparse_attention/utils.py (+216/-0); vllm/attention/ops/paged_attn.py (+24/-1); vllm/attention/selector.py (+7/-0); docs/source/models/supported_models.rst (+4/-0); tests/kernels/test_blocksparse_attention.py (+442/-0); vllm/entrypoints/openai/serving_engine.py (+1/-0); vllm/model_executor/models/__init__.py (+1/-0); vllm/model_executor/models/phi3_small.py (+447/-0); vllm/transformers_utils/config.py (+1/-1)
BODY: - Supports of Microsoft Phi-3-Small-8K and Phi-3-Small-128K models, which use blocksparse flash attention ⏎ - Prefilling Triton kernel for block-sparse attn ⏎ - Modified paged attention CUDA with the block-sparse attention, which allows hybrid sparsity pattern for each attention head. ⏎ - Use torch SPDA in prefilling phase for V100 or older GPUs, as well as CPU ⏎  ⏎ This is joint work between Microsoft GenAI @linxihui, @beagleski, and vLLM @zhuohan123, @simon-mo @youkaichao.

### L3-d4f3985907  (L3, 2024-05-28, sha d4f398590786, PR #4545)
TITLE: [Core] Sliding window for block manager v2 (#4545)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.triton.prefix_prefill; Block manager v2 sliding-window support updated prefix-prefill attention behavior.
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/prefix_prefill.py (+5/-1); tests/core/block/e2e/conftest.py (+26/-0); tests/core/block/e2e/test_correctness.py (+2/-9); tests/core/block/e2e/test_correctness_sliding_window.py (+168/-0); tests/core/block/test_block_manager_v2.py (+69/-0); vllm/core/block/block_table.py (+32/-2); vllm/core/block/cpu_gpu_block_allocator.py (+74/-0); vllm/core/block/interfaces.py (+9/-0); vllm/core/block_manager_v2.py (+17/-7); vllm/engine/arg_utils.py (+2/-1); vllm/worker/cache_engine.py (+4/-1); vllm/worker/model_runner.py (+49/-24)
ISSUES: #3665 [Misc]: Implement SlidingWindowBlockTable in BlockManagerV2 | #4057 [Feature][Chunked prefill]: Make sliding window work
BODY: This implements sliding window in v2 block manager. ⏎  ⏎ First commit comes from #3967 by @ruthe98, but the actual change was somewhat more complex including the concept of a null block. ⏎  ⏎ It passes correctness tests with starcoder3b (the smallest model with sliding window I could find). The test does a bunch of assignments "x1 = 10; x2 = 33; ..." and then asks for value of one of them (which is outside the sliding window). If we tell it upfront which we are going to be looking for, then it answers correctly. ⏎  ⏎ When using chunked prefill all the blocks for prompt are allocated immediately, while we could only allocate enough blocks for the chunk, and free any blocks that are no longer needed. After processing the prompt however, it does free the beginning of prompt at the first generation step. ⏎  ⏎ This can be fixed later. The main problem with fixing this, is that if we're generating more than one sequence, they are all forked in BlockSpaceManagerV2.allocate(), but they really should only be forked after the prompt is fully computed. (see [aborted attempt at fixing this](https://github.com/vllm-project/vllm/pull/4545/commits/9661776b755d7dfce5946bd8830d37a4a30b55e7)) ⏎  ⏎ CC @cadedaniel @ruthe98 ⏎  ⏎ FIX #3665 ⏎ FIX #4057  ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-a22dea54d3  (L3, 2024-05-30, sha a22dea54d3e8, PR #5081)
TITLE: [Model] Support MAP-NEO model (#5081)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.paged.cuda.v1; Paged attention gained support for MAP-NEO attention-head geometry.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.paged.python_wrapper
FILES: csrc/attention/attention_kernels.cu (+6/-0); csrc/cpu/attention.cpp (+6/-0); vllm/attention/ops/paged_attn.py (+1/-1); benchmarks/kernels/benchmark_paged_attention.py (+1/-1); benchmarks/kernels/benchmark_rope.py (+1/-1); tests/kernels/test_attention.py (+1/-1); tests/kernels/test_cache.py (+1/-1); tests/kernels/test_pos_encoding.py (+1/-1)
BODY: This PR support the [MAP-NEO](https://github.com/multimodal-art-projection/MAP-NEO) model which is number of attention head is 192  ⏎  ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-0ab278ca31  (L3, 2024-06-03, sha 0ab278ca3102, PR #5138)
TITLE: [Core] Remove unnecessary copies in flash attn backend (#5138)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
STAGE1: optimize; artifacts=L3.flash_attn.v0_backend,L3.flash_attn.fork_pip; FlashAttention backend removed unnecessary copies using vllm-flash-attn out parameter.
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.fork_pip
FILES: requirements-cuda.txt (+1/-1); vllm/attention/backends/flash_attn.py (+7/-6)
BODY: With vllm-flash-attn == 2.5.8.post3, we can remove the unnecessary copies in flash attn backend by using the `out` kwarg directly. ⏎ --- ⏎  ⏎ [details omitted]

### L3-c96fc06747  (L3, 2024-06-07, sha c96fc0674794, PR #4965)
TITLE: [ROCm][AMD] Use pytorch sdpa math backend to do naive attention (#4965)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: optimize; artifacts=L3.rocm.rocm_flash_attn_v0; ROCm backend replaced naive attention with PyTorch SDPA math backend for better latency.
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+29/-33)
LABELS: rocm
PERF_LINES: This pull request uses pytorch sdpa math backend to replace the existing naive attention in ROCm, as it shows latency improvement (e.g. prefill latency) over th
BODY: This pull request uses pytorch sdpa math backend to replace the existing naive attention in ROCm, as it shows latency improvement (e.g. prefill latency) over the current naive attention based on latency benchmarking. ⏎  ⏎ The naive attention is desirable (like in GPUs where support for ck flash-attention or triton flash-attention is not available).  ⏎  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-5467ac3196  (L3, 2024-06-09, sha 5467ac319636, PR #5047)
TITLE: [Kernel][Misc] Use TORCH_LIBRARY instead of PYBIND11_MODULE for custom ops (#5047)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L3.paged.cuda.v1,L3.cache.cuda_reshape; Custom attention/cache ops moved from PYBIND11_MODULE to TORCH_LIBRARY stable API.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.flash_attn.v0_backend, L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: csrc/attention/attention_kernels.cu (+18/-16); csrc/cache_kernels.cu (+8/-5); csrc/cpu/attention.cpp (+14/-12); CMakeLists.txt (+6/-16); Dockerfile.rocm (+3/-3); cmake/cpu_extension.cmake (+6/-6); cmake/utils.cmake (+8/-3); csrc/activation_kernels.cu (+1/-1); csrc/cache.h (+9/-5); csrc/cpu/cache.cpp (+8/-5); csrc/cpu/cpu_types.hpp (+1/-1); csrc/cpu/layernorm.cpp (+2/-2); csrc/cpu/pos_encoding.cpp (+1/-1); csrc/cpu/pybind.cpp (+0/-43); csrc/cpu/torch_bindings.cpp (+106/-0); csrc/cuda_utils.h (+2/-4); csrc/cuda_utils_kernels.cu (+3/-3); csrc/custom_all_reduce.cu (+14/-8); csrc/dispatch_utils.h (+1/-1); csrc/layernorm_kernels.cu (+3/-3); csrc/moe/moe_ops.cpp (+0/-8); csrc/moe/moe_ops.h (+1/-1); csrc/moe/topk_softmax_kernels.cu (+1/-1); csrc/moe/torch_bindings.cpp (+12/-0); csrc/moe_align_block_size_kernels.cu (+3/-3); csrc/ops.h (+35/-33); csrc/pos_encoding_kernels.cu (+6/-6); csrc/punica/punica_ops.cu (+3/-3); csrc/punica/punica_ops.h (+3/-3); csrc/punica/punica_pybind.cpp (+0/-13); csrc/punica/torch_bindings.cpp (+18/-0); csrc/pybind.cpp (+0/-114); csrc/quantization/aqlm/gemm_kernels.cu (+1/-1); csrc/quantization/awq/gemm_kernels.cu (+4/-4); csrc/quantization/compressed_tensors/int8_quant_kernels.cu (+1/-1); csrc/quantization/cutlass_w8a8/scaled_mm_dq_c2x.cu (+1/-1); csrc/quantization/cutlass_w8a8/scaled_mm_dq_c3x.cu (+1/-1); csrc/quantization/cutlass_w8a8/scaled_mm_dq_entry.cu (+1/-1); csrc/quantization/fp8/common.cu (+1/-1); csrc/quantization/gptq/q_gemm.cu (+3/-3); (+15 more)
BODY: ### This PR makes the following changes ⏎ - replaces the uses of `PYBIND11_MODULE` with `TORCH_LIBRARY` and adds schemas and meta functions (where) needed for all custom (C++/CUDA) kernels. ⏎ - The remainder of the custom operators are wrapped in _custom_ops.py, i.e. cuda_utils, cache_ops, moe, custom_ar and punica. ⏎ - The code + libraries are made to use the python stable api.  See #4694  ⏎  ⏎ ### Motivation ⏎ - Using `TORCH_LIBRARY` is the more official way of defining custom pytorch ops and will also help support use of torch.compile on vllm models.   See: https://github.com/neuralmagic/nm-vllm/issues/133 ⏎ - Removing the use of pybind11 enables building and linking with python's stable api.  This means that only one version of each .so can be used for any version of python that supports the stable api. ⏎ - Using pytorch dispatching would allow multiple implementations to be registered side by side, e.g. cpu/cuda ⏎  ⏎ ### Details ⏎ - The signatures of a number of kernels needed to be modified to adhere to pytorch registration, e.g. int -> int64_t, float -> double ⏎ - Explicit function schemas have been added for kernels that modify their inputs, e.g. `Tensor!` indicates modification of an input argument.  I did my best to get these correct but it is possible I missed a few, so these could use careful review.  I don't think this will be much of an issue until `torch.compile` starts to be used. ⏎ - Any new kernels will need to be registered properly.  Generally, kernels that modify their inputs will require a hand written schema (they can't be properly inferred just by the C++ signature).  And kernels that return Tensors will require a meta function.  The meta functions can simply return an empty Tensor that has the proper shape, type and device.  If a meta function is not provided then the operation will cause a graph break when torch compiling. ⏎ - I've wrapped all the kernels in `_custom_ops.py` and added a function to check for support of a particular kernel based on name (`is_custom_op_supported`) so optional extensions can be checked at runtime. ⏎ - I've kept the ability to `import` the extension shared libs rather than use `torch.ops.load_library` since that requires a platform and abi specific filename. ⏎ - The implementation and signature of `get_graph_buffer_ipc_meta` needed to be changed so that it could be registered via `TORCH_LIBRARY`.  The handles were changed to a `torch::Tensor` rather than a `std::vector<uint8_t>`.  I'm not quite sure of the device registration for the `custom_ar` ops.  I don't know if they should be CPU or CUDA (or both).

### L3-6b0511a57b  (L3, 2024-06-13, sha 6b0511a57bdb, PR #5478)
TITLE: Revert "[Core] Remove unnecessary copies in flash attn backend" (#5478)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: revert; artifacts=L3.flash_attn.v0_backend; Reverted the FlashAttention backend copy-removal optimization.
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+6/-7)
BODY: Reverts vllm-project/vllm#5138 ⏎  ⏎ We seem to still have some cases where vllm-flash-attn will raise a bogus out shape error. Reverting for now pending proper fix.

### L3-7041de4384  (L3, 2024-06-28, sha 7041de43849f, PR #4628)
TITLE: [Kernel] Flashinfer for prefill & decode, with Cudagraph support for decode (#4628)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: extend_support; artifacts=L3.flashinfer.v0_backend,L3.dispatch.selector; FlashInfer backend adds prefill/prefix-cache path and decode CUDA-graph support, with selector changes.
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.dispatch.selector
FILES: requirements-test.txt (+1/-1); vllm/attention/backends/flashinfer.py (+58/-25); vllm/attention/selector.py (+3/-2); vllm/worker/model_runner.py (+248/-78); .buildkite/test-pipeline.yaml (+3/-0); tests/basic_correctness/test_basic_correctness.py (+0/-6); tests/distributed/test_basic_distributed_correctness.py (+0/-5)
BODY: This is the second PR to integrate flashinfer. The goal is to use flashinfer for the prefill phase (including prefix caching). Hopefully, we can get rid of the [context_attention_fwd](https://github.com/vllm-project/vllm/blob/323f27b9048713cdbab31995265975842a937167/vllm/attention/ops/prefix_prefill.py#L663) triton kernel and use flashinfer [append](https://docs.flashinfer.ai/api/python/prefill.html#batch-prefill-append-attention) kernel for better performance.  ⏎ The PR also tries to add the cuda graph support for the deocde phase. ⏎ For flashinfer, currently the alibi slope is missing, will coordinate with the flashinfer side and support in the following PRs.

### L3-c4059ea54f  (L3, 2024-07-01, sha c4059ea54ff3, PR #6044)
TITLE: [Bugfix] Add explicit `end_forward` calls to flashinfer (#6044)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.flashinfer.v0_backend; FlashInfer backend adds explicit end_forward to avoid stale intermediate buffers affecting results.
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+2/-0)
BODY: Without and explicit `end_forward` call, the intermediate buffers may have leftover garbage data, affecting results. ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-482045ee77  (L3, 2024-07-02, sha 482045ee77a4, PR #6080)
TITLE: [hardware][misc] introduce platform abstraction (#6080)
SOURCES: path_core, corpus:production-kernel-provenance
STAGE1: introduce; artifacts=L3.platform.cuda_selection,L3.platform.rocm_selection; Platform abstraction introduces CUDA/ROCm default-attention selection artifacts.
ARTIFACT_HINTS: L3.triton.prefix_prefill, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/attention/ops/blocksparse_attention/interface.py (+3/-2); vllm/attention/ops/prefix_prefill.py (+2/-2); tests/kernels/test_cutlass.py (+2/-2); tests/quantization/utils.py (+2/-2); vllm/lora/punica.py (+2/-2); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py (+2/-2); vllm/model_executor/layers/quantization/fp8.py (+3/-2); vllm/model_executor/layers/quantization/gptq_marlin.py (+2/-2); vllm/model_executor/layers/quantization/utils/marlin_utils.py (+2/-2); vllm/model_executor/model_loader/loader.py (+3/-2); vllm/platforms/__init__.py (+18/-0); vllm/platforms/cuda.py (+34/-0); vllm/platforms/interface.py (+21/-0); vllm/platforms/rocm.py (+15/-0); vllm/utils.py (+0/-7); vllm/worker/worker.py (+2/-2)
ISSUES: #6059 [Bug][CI/Build]: Missing attribute 'nvmlDeviceGetHandleByIndex' in AMD tests
BODY: fixes https://github.com/vllm-project/vllm/issues/6059 ⏎  ⏎ the idea is to progressively absorb common `if-else` patterns into the `vllm/platforms` directory, so that we don't have scattered `if-else` clause for platform identification everywhere. ⏎  ⏎ ideally, the platform identification is done in `from vllm.platforms import current_platform` , just once. ⏎  ⏎ we can progressively move towards the ultimate goal, with this being the first step.

### L3-56b325e977  (L3, 2024-07-03, sha 56b325e97743, PR #6043)
TITLE: [ROCm][AMD][Model]Adding alibi slopes support in ROCm triton flash attention and naive flash attention (#6043)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L3.rocm.rocm_flash_attn_v0; ROCm FlashAttention backend adds ALiBi slopes support for Jais-style models.
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+51/-2)
LABELS: rocm
BODY: Adding alibi slopes support in ROCm triton flash attention and naive flash attention. ⏎ This fixes models such as [Jais](https://huggingface.co/core42/jais-13b) that rely on this functionality. ⏎  ⏎ With this change, the perplexity score for core42/jais13b by the https://github.com/Alexei-V-Ivanov-AMD/vllm/blob/pplv2_test/examples/measure_ppl2_llama2_MC.py script with an adjusted tokenizer improves from 14.5 to 5.2 ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-69ec3ca14c  (L3, 2024-07-04, sha 69ec3ca14cf3, PR #6051)
TITLE: [Kernel][Model] logits_soft_cap for Gemma2 with flashinfer (#6051)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: extend_support; artifacts=L3.flashinfer.v0_backend,L3.dispatch.selector; FlashInfer backend and selector add logits_soft_cap support for Gemma2.
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.dispatch.selector
FILES: vllm/attention/backends/flashinfer.py (+8/-4); vllm/attention/selector.py (+3/-3); vllm/model_executor/models/gemma2.py (+0/-7); vllm/worker/model_runner.py (+15/-4); .buildkite/test-pipeline.yaml (+5/-2); tests/kernels/test_flashinfer.py (+248/-0)
BODY: Add logits_soft_cap for flashinfer, which is needed by Gemma2 model, also add a simple gemma2 test.

### L3-4f0e0ea131  (L3, 2024-07-08, sha 4f0e0ea131ef, PR #6172)
TITLE: Add FlashInfer to default Dockerfile (#6172)
SOURCES: subject_keyword, dependency_pin, release_notes
STAGE1: repair_build_dependency; artifacts=L3.flashinfer.v0_backend; Default Dockerfile installs FlashInfer so the selectable FlashInfer backend works in release images.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile (+3/-0)
ISSUES: #6169 [Bug]: TypeError: 'NoneType' object is not callable when loading Gemma 2 9B with new 0.5.1 version
BODY: Closes #6169  ⏎  ⏎ Testing with  ⏎ ``` ⏎ docker run --gpus all -p 8000:8000 -e HF_TOKEN --ipc=host --env "VLLM_ATTENTION_BACKEND=FLASHINFER" -v /data/xmo/hub:/root/.cache/huggingface vllm/vllm-openai --model google/gemma-2-9b-it ⏎ ``` ⏎  ⏎ ``` ⏎ $ curl http://localhost:8000/v1/completions  -H "Content-Type: application/json"      -d '{ ⏎ "model": "google/gemma-2-9b-it", ⏎ "prompt":"Who won the world series in 2020?", ⏎ "max_tokens": 100, ⏎ "ignore_eos": true ⏎ }' ⏎ {"id":"cmpl-8ce64ceae52449e2b04988b08f3f42f9","object":"text_completion","created":1720253967,"model":"google/gemma-2-9b-it","choices":[{"index":0,"text":"\n\nThe **Los Angeles Dodgers** won the World Series in 2020. \n\\\\\n  \\\\\n\n\n\\\\\n\\\\\n\\\n\n\\\\\n\\\\\n\n\n\n\n.\n\n\n'.\n\n\n\n\n\n **\n\n\n\n\n。","logprobs":null,"finish_reason":"length","stop_reason":null}],"usage":{"prompt_tokens":13,"total_tokens":113,"completion_tokens":100}}(miniconda3) (base) [xmo@flow-matic:/data/xmo/vllm]$  ⏎ ``` ⏎  ⏎ The docker image for `vllm/vllm-openai:latest` and `vllm/vllm-openai:v0.5.1` has been built and updated. This doesn't effect the wheel build.
