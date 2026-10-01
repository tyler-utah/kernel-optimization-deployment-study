### L3-0794e7446e  (L3, 2025-01-15, sha 0794e7446efc, PR #10467)
TITLE: [Misc] Add multipstep chunked-prefill support for FlashInfer (#10467)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L3.flashinfer.v0_backend; Dossier shows L3.flashinfer.v0_backend behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+24/-5); vllm/worker/model_runner.py (+118/-102); vllm/worker/multi_step_model_runner.py (+1/-1); csrc/prepare_inputs/advance_step.cu (+10/-0); tests/multi_step/test_correctness_llm.py (+16/-1)
LABELS: ready
BODY: Support multi-step scheduling for chunked-prefill on FlashInfer, where prefill tokens are turned into decode tokens after the first single step.  ⏎  ⏎ cc @comaniac @yzh199 @WoosukKwon @youkaichao

### L3-69d765f5a5  (L3, 2025-01-17, sha 69d765f5a5bb, PR #11960)
TITLE: [V1] Move more control of kv cache initialization from model_executor to EngineCore (#11960)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.dispatch.abstract_interface; Dossier shows L3.dispatch.abstract_interface behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+2/-0); tests/v1/test_utils.py (+62/-0); vllm/v1/core/kv_cache_utils.py (+124/-0); vllm/v1/engine/core.py (+18/-13); vllm/v1/executor/abstract.py (+8/-3); vllm/v1/executor/multiproc_executor.py (+15/-10); vllm/v1/executor/ray_executor.py (+21/-19); vllm/v1/executor/uniproc_executor.py (+14/-11); vllm/v1/kv_cache_interface.py (+111/-0); vllm/v1/utils.py (+54/-2); vllm/v1/worker/gpu_model_runner.py (+72/-12); vllm/v1/worker/gpu_worker.py (+14/-34)
LABELS: ready
BODY: This pr changes the workflow of `EngineCore._initialize_kv_caches` to enable more flexible control of kv cache format in the future. ⏎ It is splitted from https://github.com/vllm-project/vllm/pull/11938 and is a preparation for https://github.com/vllm-project/vllm/issues/11382 ⏎ Original workflow: ⏎ ```python ⏎ num_gpu_blocks, _ = self.model_executor.determine_num_available_blocks() ⏎ self.model_executor.initialize(num_gpu_blocks) ⏎ ``` ⏎ New workflow: ⏎ ```python ⏎ # Get all kv cache tensor needed by the model ⏎ kv_cache_spec = self.model_executor.get_kv_cache_spec() ⏎  ⏎ # Profiles the peak memory usage of the model to determine how much ⏎ # memory can be allocated for kv cache. ⏎ availble_gpu_memory = self.model_executor.get_available_memory() ⏎  ⏎ # Get the kv cache tensor size ⏎ kv_cache_config, num_gpu_blocks = get_kv_cache_config( ⏎     vllm_config, kv_cache_spec, availble_gpu_memory) ⏎  ⏎ # Initialize kv cache and warmup the execution ⏎ self.model_executor.initialize(kv_cache_config) ⏎ ``` ⏎  ⏎ This pr introduces 2 new concepts: ⏎ 1. `KVCacheSpec`, a data structure to represent the kv cache needed by each attention layer, which is constructed by asking the model runner to analyze all Attention modules. Will add more types of Spec in the future, e.g., `SlidingWindowSpec`, `MLASpec` ⏎ 2. `KVCacheConfig`, a class to represent how to allocate the kv cache Tensor. It is quite simple now, i.e., tensors with the same size. But it may be extended to the following cases: ⏎     1. tensors with different sizes, to support MLA & spec decode ⏎     2. allocate a global buffer, and make the kv_cache tensors to point to different offsets, to support multiple types of layer sharing the same memory pool.

### L3-86bfb6dba7  (L3, 2025-01-20, sha 86bfb6dba7c6, PR #12218)
TITLE: [Misc] Pass `attention` to impl backend (#12218)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: adapt_framework; artifacts=L3.xformers.v0_backend,L3.flash_attn.v0_backend,L3.flash_attn.v1_backend,L3.flashinfer.v0_backend,L3.rocm.rocm_flash_attn_v0,L3.dispatch.abstract_interface,L3.blocksparse.v0; Dossier shows L3.xformers.v0_backend behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: vllm/attention/backends/abstract.py (+19/-4); vllm/attention/backends/blocksparse_attn.py (+6/-6); vllm/attention/backends/flash_attn.py (+5/-5); vllm/attention/backends/flashinfer.py (+8/-8); vllm/attention/backends/hpu_attn.py (+2/-2); vllm/attention/backends/ipex_attn.py (+9/-9); vllm/attention/backends/pallas.py (+3/-3); vllm/attention/backends/rocm_flash_attn.py (+10/-10); vllm/attention/backends/torch_sdpa.py (+8/-10); vllm/attention/backends/xformers.py (+9/-11); vllm/attention/layer.py (+3/-5); vllm/v1/attention/backends/flash_attn.py (+4/-5)
LABELS: ready
BODY: With https://github.com/vllm-project/vllm/pull/11969, a quantization method can be implemented and register to vLLM out-of-tree now. But there is no way to use the registered parameter for attention layer in custom attention backend impl. ⏎  ⏎ This PR pass the attention object to attention backend, so that the backend can use the parameters registered by quantization attention method directly to keep the same with other kind of quantization method(Linear, MoE)

### L3-66818e5b63  (L3, 2025-01-22, sha 66818e5b6381, PR #12253)
TITLE: [core] separate builder init and builder prepare for each batch (#12253)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.flash_attn.v0_backend,L3.flashinfer.v0_backend,L3.dispatch.abstract_interface; Dossier shows L3.flash_attn.v0_backend behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+6/-5); vllm/attention/backends/flash_attn.py (+6/-5); vllm/attention/backends/flashinfer.py (+8/-6); vllm/attention/backends/placeholder_attn.py (+5/-3); vllm/attention/backends/torch_sdpa.py (+4/-1); vllm/attention/backends/utils.py (+7/-6); vllm/worker/cpu_model_runner.py (+17/-7); vllm/worker/model_runner.py (+24/-12); vllm/worker/model_runner_base.py (+5/-0); vllm/worker/xpu_model_runner.py (+8/-2)
LABELS: ready
BODY: Right now we create the builder instance for every batch, and hence create the attention metadata builder instance for every batch, but we don't have some global information when we create the input for every batch. ⏎  ⏎ When upgrading to flashinfer 0.2, see https://github.com/vllm-project/vllm/pull/11194 , the attention metadata builder needs to access some global information such as sliding window, for the `plan` function before the flashinfer wrapper can be used. ⏎  ⏎ This PR fixes the problem, by separating the builder init and builder prepare for each batch: ⏎ - we can remember global config when we create the builder. ⏎ - the builder will call `prepare` (or we can rename it to `reset`) to prepare for the current batch, and now it can access the global config stored during construction.

### L3-978b45f399  (L3, 2025-01-23, sha 978b45f39970, PR #12093)
TITLE: [Kernel] Flash Attention 3 Support (#12093)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
STAGE1: integrate; artifacts=L3.flash_attn.v0_backend,L3.flash_attn.v1_backend,L3.flash_attn.upstream_pip,L3.flash_attn.fork_inline_cmake,L3.flashinfer.trtllm_gen; Dossier shows L3.flash_attn.v0_backend behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake, L3.flashinfer.trtllm_gen
FILES: CMakeLists.txt (+20/-25); setup.py (+8/-4); vllm/attention/backends/flash_attn.py (+24/-3); vllm/envs.py (+12/-0); vllm/v1/attention/backends/flash_attn.py (+33/-11); vllm/v1/worker/gpu_model_runner.py (+21/-25); tests/kernels/test_cascade_flash_attn.py (+14/-10); tests/kernels/test_flash_attn.py (+18/-4)
LABELS: ready, ci/build
PERF_LINES: This currently mainly improves V1 performance, some throughput numbers: | env VLLM_FLASH_ATTN_VERSION=2 VLLM_USE_V1=1 python benchmarks/benchmark_throughput.py --model meta-llama/Llama-3.1-8B-Instruct --input-len 2048 --output-len 128 | Throughput: 12.81 requests/s, 27872.21 total tokens/s, 1639.54 output tokens/s | env VLLM_FLASH_ATTN_VERSION=3 VLLM_USE_V1=1 python benchmarks/benchmark_throughput
BODY: Most of the changes for FA3 are in vllm-flash-attn, but there was some changes required on vLLM side. Namely FA3 doesn't support `cu_seqlens_k` when using a paged kv cache, instead it uses `seqused_k` which is the kv seqlens. FA2 also supports `seqused_k` so we switch to using this for both FA3 and FA2 when dealing with a paged kv-cache in-order to maintain a common interface. ⏎  ⏎ This currently mainly improves V1 performance, some throughput numbers: ⏎ ``` ⏎ env VLLM_FLASH_ATTN_VERSION=2 VLLM_USE_V1=1 python benchmarks/benchmark_throughput.py --model meta-llama/Llama-3.1-8B-Instruct --input-len 2048 --output-len 128 --max-num-batched-tokens 16384  ⏎ Throughput: 12.81 requests/s, 27872.21 total tokens/s, 1639.54 output tokens/s ⏎  ⏎ env VLLM_FLASH_ATTN_VERSION=3 VLLM_USE_V1=1 python benchmarks/benchmark_throughput.py --model meta-llama/Llama-3.1-8B-Instruct --input-len 2048 --output-len 128 --max-num-batched-tokens 16384  ⏎ Throughput: 14.24 requests/s, 30986.15 total tokens/s, 1822.71 output tokens/s ⏎  ⏎ env VLLM_FLASH_ATTN_VERSION=2 VLLM_USE_V1=0 python benchmarks/benchmark_throughput.py --model meta-llama/Llama-3.1-8B-Instruct --input-len 2048 --output-len 128 --max-num-batched-tokens 16384  ⏎ Throughput: 12.40 requests/s, 26991.51 total tokens/s, 1587.74 output tokens/s ⏎  ⏎ env VLLM_FLASH_ATTN_VERSION=3 VLLM_USE_V1=0 python benchmarks/benchmark_throughput.py --model meta-llama/Llama-3.1-8B-Instruct --input-len 2048 --output-len 128 --max-num-batched-tokens 16384  ⏎ Throughput: 12.74 requests/s, 27714.72 total tokens/s, 1630.28 output tokens/s ⏎  ⏎ env VLLM_FLASH_ATTN_VERSION=2 VLLM_USE_V1=0 python benchmarks/benchmark_throughput.py --model meta-llama/Llama-3.1-8B-Instruct --input-len 2048 --output-len 128 --max-num-batched-tokens 16384 --num-scheduler-steps 4 ⏎ Throughput: 13.20 requests/s, 28726.44 total tokens/s, 1689.79 output tokens/s ⏎  ⏎ env VLLM_FLASH_ATTN_VERSION=3 VLLM_USE_V1=0 python benchmarks/benchmark_throughput.py --model meta-llama/Llama-3.1-8B-Instruct --input-len 2048 --output-len 128 --max-num-batched-tokens 16384 --num-scheduler-steps 4 ⏎ Throughput: 13.34 requests/s, 29029.89 total tokens/s, 1707.64 output tokens/s ⏎ ```

### L3-e97f802b2d  (L3, 2025-01-23, sha e97f802b2d74, PR #11906)
TITLE: [FP8][Kernel] Dynamic kv cache scaling factors computation (#11906)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.paged.cuda.v1,L3.paged.cuda.v2_splitkv,L3.cache.cuda_reshape,L3.paged.python_wrapper,L3.xformers.v0_backend,L3.flash_attn.v0_backend,L3.flash_attn.v1_backend,L3.flashinfer.v0_backend,L3.triton.prefix_prefill,L3.rocm.custom_paged,L3.rocm.rocm_flash_attn_v0,L3.dispatch.abstract_interface,L3.blocksparse.v0; Dossier shows L3.paged.cuda.v1 behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flashinfer.v0_backend, L3.triton.prefix_prefill, L3.rocm.custom_paged, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: csrc/attention/attention_kernels.cuh (+5/-5); csrc/attention/paged_attention_v1.cu (+10/-7); csrc/attention/paged_attention_v2.cu (+10/-7); csrc/cache_kernels.cu (+17/-13); csrc/cpu/attention.cpp (+6/-6); csrc/rocm/attention.cu (+10/-7); vllm/attention/backends/abstract.py (+8/-2); vllm/attention/backends/blocksparse_attn.py (+2/-0); vllm/attention/backends/flash_attn.py (+4/-1); vllm/attention/backends/flashinfer.py (+6/-4); vllm/attention/backends/ipex_attn.py (+1/-1); vllm/attention/backends/pallas.py (+1/-1); vllm/attention/backends/placeholder_attn.py (+3/-0); vllm/attention/backends/rocm_flash_attn.py (+2/-0); vllm/attention/backends/torch_sdpa.py (+1/-1); vllm/attention/backends/utils.py (+2/-0); vllm/attention/backends/xformers.py (+2/-0); benchmarks/kernels/benchmark_paged_attention.py (+3/-1); csrc/cache.h (+3/-3); csrc/cpu/cache.cpp (+2/-4); csrc/cpu/torch_bindings.cpp (+3/-3); csrc/ops.h (+6/-4); csrc/rocm/ops.h (+2/-2); csrc/rocm/torch_bindings.cpp (+1/-1); csrc/torch_bindings.cpp (+4/-4); docs/source/features/quantization/quantized_kvcache.md (+6/-4); examples/other/fp8/README.md (+0/-96); examples/other/fp8/extract_scales.py (+0/-367); examples/other/fp8/quantizer/README.md (+0/-32); examples/other/fp8/quantizer/quantize.py (+0/-367); tests/fp8_kv/llama2-70b-fp8-kv/kv_cache_scales.json (+0/-90); tests/fp8_kv/llama2-7b-fp8-kv/kv_cache_scales.json (+0/-42); tests/kernels/test_attention.py (+1/-1); tests/kernels/test_blocksparse_attention.py (+1/-1); tests/kernels/test_cache.py (+5/-5); tests/kernels/test_prefix_prefill.py (+10/-0); tests/kernels/utils.py (+2/-0); tests/models/decoder_only/language/test_fp8.py (+4/-11); tests/worker/test_model_input.py (+3/-0); vllm/_custom_ops.py (+10/-10); (+20 more)
LABELS: documentation, rocm, ready
BODY: This PR deprecates loading kv cache scales from json in favor of adding the option to dynamically compute them based on the first real input to the attention layer. ⏎ Our tests showed that the dynamic range computed based on the first input to each layer is representative of the entire model, and the accuracy is comparable with scaling factors computed using Quark quantizer (such as in HF amd/*-FP8-KV models) ⏎  ⏎ Accuracy measured using the [P3L ](https://github.com/ROCm/vllm/blob/main/benchmarks/P3L.py) benchmark that allows measuring accuracy on decode steps, using the data in the kv cache ⏎  ⏎ K and V scale parameters are made on-device tensors in order to allow changing their values after the graph has been captured. This also lays the foundation to using per-channel quantization with tensor-like scales. ⏎  ⏎ The effect is most visible on models with dynamic value ranges outside of the scope of fp8e4m3, such as Quen2 7B: ⏎ Using dynamic calculation reduces the PPL score from 34.84 to 22.62 ⏎  ⏎ On LLama based models the improvement is much smaller, due to the fact that identity scales work just as well, but still can be in single digit percents, on par with using the scales from a quantized model

### L3-ab5bbf5ae3  (L3, 2025-01-24, sha ab5bbf5ae32b, PR #12375)
TITLE: [Bugfix][Kernel] Fix CUDA 11.8 being broken by FA3 build (#12375)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, corpus:kernel-correctness-cases
STAGE1: repair_build_dependency; artifacts=L3.flash_attn.v0_backend,L3.flash_attn.v1_backend,L3.flash_attn.upstream_pip,L3.flash_attn.fork_inline_cmake; Dossier shows L3.flash_attn.v0_backend behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1); setup.py (+4/-1); vllm/attention/backends/flash_attn.py (+11/-3); vllm/v1/attention/backends/flash_attn.py (+10/-1); tests/kernels/test_cascade_flash_attn.py (+6/-5); tests/kernels/test_flash_attn.py (+10/-11)
LABELS: documentation, ready, ci/build
DEEP_STUDY: deep-study correctness case vllm:ab5bbf5ae3: class=hardware_compiler_specific; symptom=compile_or_build_failure; introducing=unknown
BODY: Can build FA3 with CUDA version < 12.0 due to use of `__viaddmin_s32` and compute capability 9.0a, disable FA3 for 11.8 builds

### L3-3132a933b6  (L3, 2025-01-24, sha 3132a933b65d, PR #12405)
TITLE: [Bugfix][Kernel] FA3 Fix - RuntimeError: This flash attention build only supports pack_gqa (for build size reasons). (#12405)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
STAGE1: repair_correctness; artifacts=L3.flash_attn.fork_inline_cmake; Dossier shows L3.flash_attn.fork_inline_cmake behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1)
LABELS: ready, ci/build
BODY: Based off of https://github.com/vllm-project/vllm/pull/12375 that should be merged first ⏎  ⏎ We thought we could get away with only packed-gqa (is the default for varlen gqa) for build time and size reasons, but fails for true MHA, this turns back on the non packed-gqa kernels. ⏎  ⏎ Tested with `facebook/opt-125m`

### L3-68f11149d8  (L3, 2025-01-26, sha 68f11149d845, PR #12434)
TITLE: [Bugfix][Kernel] Fix perf regression caused by PR #12405 (#12434)
SOURCES: dependency_pin
STAGE1: repair_performance; artifacts=L3.flash_attn.fork_inline_cmake; Dossier shows L3.flash_attn.fork_inline_cmake behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1)
LABELS: ready, ci/build
BODY: Fix perf regression caused by https://github.com/vllm-project/vllm/pull/12405 (resulted in falling back on FA2) ⏎  ⏎ Tested using: ⏎ ``` ⏎ VLLM_FLASH_ATTN_VERSION=3 python -m vllm.scripts serve meta-llama/Meta-Llama-3-8B ⏎ ```

### L3-372bf0890b  (L3, 2025-01-27, sha 372bf0890b19, PR #12464)
TITLE: [Bugfix] Fix missing seq_start_loc in xformers prefill metadata (#12464)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.xformers.v0_backend; Dossier shows L3.xformers.v0_backend behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.xformers.v0_backend
FILES: vllm/attention/backends/xformers.py (+3/-0)
LABELS: ready
BODY: Phi-3-small with blocksparse attention is not working with xformers attention prefill metadata because of missing `seq_start_loc`:  ⏎ [details omitted] ⏎  ⏎ - This PR fixes the missing `seq_start_loc` in xformers prefill metadata.

### L3-2bc3fbba0c  (L3, 2025-01-27, sha 2bc3fbba0cf5, PR #11194)
TITLE: [FlashInfer] Upgrade to 0.2.0 (#11194)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
STAGE1: adapt_framework; artifacts=L3.flash_attn.upstream_pip,L3.flashinfer.v0_backend; Dossier shows L3.flash_attn.upstream_pip behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.v0_backend
FILES: Dockerfile (+21/-2); vllm/attention/backends/flashinfer.py (+162/-21); vllm/config.py (+6/-4); vllm/worker/worker_base.py (+13/-4); .buildkite/test-pipeline.yaml (+10/-1); tests/basic_correctness/test_basic_correctness.py (+3/-2); tests/compile/test_basic_correctness.py (+1/-1); tests/kernels/test_flashinfer.py (+37/-37); vllm/model_executor/model_loader/loader.py (+2/-2); vllm/model_executor/model_loader/tensorizer.py (+2/-1)
LABELS: documentation, frontend, ready, ci/build
BODY: This PR upgrades the [FlashInfer](https://github.com/flashinfer-ai/flashinfer) attention backend to v0.2.0.

### L3-cabaf4eff3  (L3, 2025-01-30, sha cabaf4eff3c7, PR #12528)
TITLE: [Attention] MLA decode optimizations (#12528)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: introduce; artifacts=L3.cache.cuda_reshape,L3.flashinfer.trtllm_gen,L3.triton.decode_attention,L3.mla.triton_v0,L3.dispatch.selector,L3.dispatch.abstract_interface,L3.platform.cuda_selection,L3.platform.rocm_selection; Dossier shows L3.cache.cuda_reshape behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flashinfer.trtllm_gen, L3.triton.decode_attention, L3.mla.triton_v0, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: csrc/cache_kernels.cu (+95/-0); csrc/torch_bindings.cpp (+9/-0); vllm/_custom_ops.py (+13/-0); vllm/attention/backends/abstract.py (+16/-0); vllm/attention/backends/mla/__init__.py (+0/-0); vllm/attention/backends/mla/utils.py (+365/-0); vllm/attention/backends/triton_mla.py (+749/-0); vllm/attention/backends/utils.py (+4/-0); vllm/attention/layer.py (+15/-4); vllm/attention/ops/triton_decode_attention.py (+667/-0); vllm/attention/selector.py (+5/-1); vllm/config.py (+26/-9); vllm/engine/arg_utils.py (+0/-1); vllm/envs.py (+14/-0); vllm/model_executor/models/deepseek_v2.py (+152/-2); vllm/platforms/cpu.py (+2/-1); vllm/platforms/cuda.py (+7/-2); vllm/platforms/hpu.py (+2/-1); vllm/platforms/interface.py (+3/-1); vllm/platforms/openvino.py (+2/-1); vllm/platforms/rocm.py (+2/-1); vllm/platforms/tpu.py (+2/-1); vllm/platforms/xpu.py (+2/-1); vllm/worker/cache_engine.py (+2/-1); vllm/worker/model_runner.py (+3/-1); csrc/cache.h (+5/-0); tests/kernels/test_triton_decode_attention.py (+89/-0); tests/plugins/vllm_add_dummy_platform/vllm_add_dummy_platform/dummy_platform.py (+1/-1); tests/weight_loading/models.txt (+1/-1); tests/weight_loading/run_model_weight_loading_test.sh (+7/-2); vllm/model_executor/model_loader/loader.py (+6/-0)
LABELS: ready, ci/build
BODY: Implements MLA decode optimizations, i.e. computing MQA using latent vectors instead of MHA ⏎  ⏎ Shout-out to @simon-mo for the initial PR: https://github.com/vllm-project/vllm/pull/10927 ⏎ Shout-out to @tsu-bin for the handy reference: https://github.com/flashinfer-ai/flashinfer/pull/551 ⏎ Shout-out to [sglang](https://github.com/sgl-project/sglang) for the triton decode attention kernel

### L3-a1fc18c030  (L3, 2025-01-31, sha a1fc18c030e4, PR #12421)
TITLE: [ROCm][AMD][Model] llama 3.2 support upstreaming (#12421)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.rocm.rocm_flash_attn_v0; Dossier shows L3.rocm.rocm_flash_attn_v0 behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+291/-85); vllm/model_executor/models/mllama.py (+13/-3)
LABELS: rocm, ready
BODY: PR to propagate multimodal llama3.2 support into upstream for rocm arch

### L3-baeded2569  (L3, 2025-01-31, sha baeded25699f, PR #12601)
TITLE: [Attention] Deepseek v3 MLA support with FP8 compute (#12601)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L3.flashinfer.trtllm_gen,L3.mla.triton_v0; Dossier shows L3.flashinfer.trtllm_gen behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.triton_v0
FILES: vllm/attention/backends/mla/utils.py (+184/-36); vllm/attention/backends/triton_mla.py (+7/-11); vllm/attention/layer.py (+2/-2); vllm/config.py (+33/-6); vllm/envs.py (+11/-1); vllm/model_executor/models/deepseek_v3.py (+152/-2); vllm/worker/cache_engine.py (+3/-1); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+52/-22); vllm/model_executor/layers/quantization/utils/quant_utils.py (+115/-1); vllm/model_executor/model_loader/loader.py (+21/-3)
LABELS: ready
BODY: Based off of: https://github.com/vllm-project/vllm/pull/12528 that needs to land first

### L3-4f4d427ac2  (L3, 2025-01-31, sha 4f4d427ac2ce, PR #12642)
TITLE: Disable chunked prefill and/or prefix caching when MLA is enabled  (#12642)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: change_default; artifacts=NEW:attention_adapter; Dossier shows NEW:attention_adapter behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: -
FILES: vllm/config.py (+10/-0)
LABELS: ready
BODY: From @mgoin in https://github.com/vllm-project/vllm/pull/12638 ⏎  ⏎ I cannot push to that branch, therefore a new PR to unblock release.

### L3-c36ac98d01  (L3, 2025-02-04, sha c36ac98d0118, PR #12662)
TITLE: [AMD][ROCm] Enable DeepSeek model on ROCm (#12662)
SOURCES: path_core, symbol_pickaxe
STAGE1: change_default; artifacts=L3.platform.rocm_selection; Dossier shows L3.platform.rocm_selection behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/attention/backends/mla/utils.py (+5/-1); tests/kernels/test_rocm_attention_selector.py (+31/-0); tests/worker/test_model_runner.py (+9/-0); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+10/-0); vllm/platforms/rocm.py (+3/-0)
LABELS: rocm, ready
PERF_LINES: - Tested on DeepSeek V2  (DeepSeek-Coder-V2-Lite-Instruct) model on MI300x and MI210. | - Tested on DeepSeek V3 dummy weights on MI300x.
BODY: Thanks for great work from vLLM community to enable DeepSeek model. ⏎ This PR is to use the Triton MLA backend on ROCm to enable DeepSeek model on AMD GPUs. ⏎  ⏎ Tests: ⏎  ⏎ - Tested on DeepSeek V2  (DeepSeek-Coder-V2-Lite-Instruct) model on MI300x and MI210. ⏎ - Tested on DeepSeek V3 dummy weights on MI300x. ⏎ - Added two unit tests. ⏎ -  ⏎  ⏎ Note: to use DeepSeek model, we should ensure to have the "trust_remote_code" flag on.

### L3-75e94309e8  (L3, 2025-02-04, sha 75e94309e8d8, PR #12676)
TITLE: [Perf] Mem align KV caches for CUDA devices (MLA perf improvement) (#12676)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: optimize; artifacts=L3.cache.cuda_reshape,L3.flashinfer.trtllm_gen,L3.triton.decode_attention,L3.mla.triton_v0; Dossier shows L3.cache.cuda_reshape behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flashinfer.trtllm_gen, L3.triton.decode_attention, L3.mla.triton_v0
FILES: csrc/cache_kernels.cu (+70/-12); csrc/torch_bindings.cpp (+4/-0); vllm/_custom_ops.py (+5/-0); vllm/attention/backends/triton_mla.py (+2/-3); vllm/attention/ops/triton_decode_attention.py (+8/-8); vllm/envs.py (+10/-0); vllm/utils.py (+10/-0); vllm/worker/cache_engine.py (+55/-11); csrc/cache.h (+3/-0); tests/kernels/test_cache.py (+262/-0)
LABELS: ready
PERF_LINES: This means for MLA with a head dim of 576 (like DeepSeek V2/V3) and a fp16/bf16 cache, we allocate 640 elements per cache entry in instead of 576 (1280 bytes in | VLLM_CUDA_MEM_ALIGN_KV_CACHE=0  python3 benchmarks/benchmark_throughput.py --model /data/nm/models/DeepSeek-R1 --trust-remote-code --tensor-parallel-size 8 --ma | Throughput: 0.76 requests/s, 2289.10 total tokens/s, 763.03 output tokens/
BODY: Generally Nvidia hardware likes 256 byte alignment (reasons is foggy due to the blackbox nature of Nvidia hardware), but memory allocated via the CUDA Runtime ensure 256 byte alignment (see https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#a-sequential-but-misaligned-access-pattern). ⏎  ⏎ This PR aligns KV cache entries to start 256 byte boundaries, this mainly targets MLA since for "normal attention" with normal head dims (say 64 or 128) the entries are naturally 256 byte aligned. ⏎  ⏎ This means for MLA with a head dim of 576 (like DeepSeek V2/V3) and a fp16/bf16 cache, we allocate 640 elements per cache entry in instead of 576 (1280 bytes instead of 1152). This increases the size of the cache by ~11% (wasted), but leads to a worthwhile performance gain. ⏎  ⏎ Results DeepSeek-R1 on 8xH200 ⏎  ⏎ ``` ⏎ VLLM_CUDA_MEM_ALIGN_KV_CACHE=0  python3 benchmarks/benchmark_throughput.py --model /data/nm/models/DeepSeek-R1 --trust-remote-code --tensor-parallel-size 8 --max-model-len 8000 --enable-chunked-prefill False --input-len 2000 --output-len 1000  --num-prompts 100 ⏎ ... ⏎ Throughput: 0.76 requests/s, 2289.10 total tokens/s, 763.03 output tokens/s ⏎ ``` ⏎  ⏎ ``` ⏎ VLLM_CUDA_MEM_ALIGN_KV_CACHE=1  python3 benchmarks/benchmark_throughput.py --model /data/nm/models/DeepSeek-R1 --trust-remote-code --tensor-parallel-size 8 --max-model-len 8000 --enable-chunked-prefill False --input-len 2000 --output-len 1000  --num-prompts 100 ⏎ ... ⏎ Throughput: 1.10 requests/s, 3287.09 total tokens/s, 1095.70 output tokens/s ⏎ ``` ⏎  ⏎ Accuracy: ⏎ ``` ⏎ VLLM_MLA_DISABLE=1 lm_eval --model vllm --model_args pretrained=/data/nm/models/DeepSeek-R1,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384,enforce_eager=False --task gsm8k --num_fewshot=5 --limit 100 ⏎ ... ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.94|±  |0.0239| ⏎ |     |       |strict-match    |     5|exact_match|↑  | 0.94|±  |0.0239| ⏎  ⏎ VLLM_MLA_DISABLE=0 VLLM_CUDA_MEM_ALIGN_KV_CACHE=0 lm_eval --model vllm --model_args pretrained=/data/nm/models/DeepSeek-R1,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384,enforce_eager=False --task gsm8k --num_fewshot=5 --limit 100 ⏎ ... ⏎ INFO 02-03 14:26:12 executor_base.py:110] # CUDA blocks: 30218, # CPU blocks: 3819 ⏎ INFO 02-03 14:26:12 executor_base.py:115] Maximum concurrency for 16384 tokens per request: 29.51x ⏎ ... ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |  …[truncated]

### L3-98fd089fc9  (L3, 2025-02-04, sha 98fd089fc974, PR #12729)
TITLE: [VLM] Add MLA with pure RoPE support for deepseek-vl2 models (#12729)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: extend_support; artifacts=NEW:attention_adapter; Dossier shows NEW:attention_adapter behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/mla/utils.py (+26/-4); vllm/model_executor/models/deepseek_v2.py (+2/-1); vllm/model_executor/models/deepseek_v3.py (+2/-1)
LABELS: ready
BODY: FIX https://github.com/vllm-project/vllm/pull/11578#issuecomment-2632920024 ⏎ - Deepseek-VL2 use pure rotary embedding, and current MLA implementation only support yarn rope

### L3-64862d106e  (L3, 2025-02-05, sha 64862d106efa, PR #12713)
TITLE: [ROCM][AMD][TRITON] Halving warps number for fw_prefill to reduce spilling (#12713)
SOURCES: path_core
STAGE1: retune; artifacts=L3.triton.prefix_prefill; Dossier shows L3.triton.prefix_prefill behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/prefix_prefill.py (+1/-1)
LABELS: rocm, ready
PERF_LINES: in general this gives 2-3x perf gain over default and about 1.5x slower than H100 | HIP_VISIBLE_DEVICES=1 python /root/workspace/vllm/benchmarks/profiling/benchmark_throughput.py --model /data/models/Llama-3.1-70B-Instruct --input-len 1024 --ou | Throughput: 0.65 requests/s, 1327.15 total tokens/s, 663.57 output tokens/s | Throughput: 0.64 requests/s, 1318.79 total tokens/s, 659.40 output tokens/s
BODY: in general this gives 2-3x perf gain over default and about 1.5x slower than H100 ⏎  ⏎ for model run llama3.1 70B fp16 ⏎  ⏎ HIP_VISIBLE_DEVICES=1 python /root/workspace/vllm/benchmarks/profiling/benchmark_throughput.py --model /data/models/Llama-3.1-70B-Instruct --input-len 1024 --output-len 1024 --num-prompts 256 --max-model-len 32768 --disable-log-stats --enable-chunked-prefill ⏎  ⏎ 8 warps ⏎ Throughput: 0.65 requests/s, 1327.15 total tokens/s, 663.57 output tokens/s ⏎ Throughput: 0.64 requests/s, 1318.79 total tokens/s, 659.40 output tokens/s ⏎ Throughput: 0.65 requests/s, 1323.83 total tokens/s, 661.91 output tokens/s ⏎  ⏎ 4 warps ⏎ Throughput: 0.68 requests/s, 1396.40 total tokens/s, 698.20 output tokens/s ⏎ Throughput: 0.68 requests/s, 1394.23 total tokens/s, 697.12 output tokens/s ⏎ Throughput: 0.68 requests/s, 1393.89 total tokens/s, 696.94 output tokens/s ⏎  ⏎ kernel time reduced from 5% to 3% of total execution time

### L3-4c3aac51e1  (L3, 2025-02-05, sha 4c3aac51e142, PR #)
TITLE: Merging PR #12536
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L3.dispatch.abstract_interface; Dossier shows L3.dispatch.abstract_interface behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L3-85ac82d228  (L3, 2025-02-06, sha 85ac82d228ef, PR #12777)
TITLE: [Kernel] Make rotary_embedding ops more flexible with input shape (#12777)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=NEW:attention_adapter; Dossier shows NEW:attention_adapter behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/mla/utils.py (+3/-22); csrc/pos_encoding_kernels.cu (+89/-14); tests/kernels/test_pos_encoding.py (+22/-9); vllm/model_executor/models/deepseek_v2.py (+1/-12)
LABELS: ready
BODY: FIX https://github.com/vllm-project/vllm/pull/12729#discussion_r1942203266

### L3-8108ac841d  (L3, 2025-02-06, sha 8108ac841d66, PR #12828)
TITLE: [Bugfix] Fix unsupported FA version check for Turing GPU (#12828)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.flash_attn.fa_utils; Dossier shows L3.flash_attn.fa_utils behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/utils.py (+1/-1)
LABELS: ready
BODY: #12807 broke xformers fallback because `AssertionError` from `assert is_fa_version_supported(fa_version)` is not included in the exception: ⏎ https://github.com/vllm-project/vllm/blob/c786e757fae4519256e4ef88a7d4f56c3339d14d/vllm/attention/backends/utils.py#L611-L616 ⏎ ``` ⏎ ERROR 02-06 12:53:27 utils.py:608] Cannot use FA version 2 is not supported due to FA3 is only supported on devices with compute capability >= 8 excluding 8.6 and 8.9 ⏎ Traceback (most recent call last): ⏎   File "/kaggle/working/vllm/examples/offline_inference/basic.py", line 16, in <module> ⏎     llm = LLM(model="facebook/opt-125m") ⏎           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎   ... ⏎  ⏎   File "/kaggle/working/vllm/vllm/worker/worker.py", line 29, in <module> ⏎     from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner ⏎   File "/kaggle/working/vllm/vllm/worker/enc_dec_model_runner.py", line 12, in <module> ⏎     from vllm.attention.backends.utils import PAD_SLOT_ID ⏎   File "/kaggle/working/vllm/vllm/attention/backends/utils.py", line 614, in <module> ⏎     VLLM_FLASH_ATTN_VERSION = flash_attn_version() ⏎                               ^^^^^^^^^^^^^^^^^^^^ ⏎   File "/kaggle/working/vllm/vllm/attention/backends/utils.py", line 611, in flash_attn_version ⏎     assert is_fa_version_supported(fa_version) ⏎            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ AssertionError ⏎ ``` ⏎  ⏎ cc @LucasWilkinson

### L3-c786e757fa  (L3, 2025-02-06, sha c786e757fae4, PR #12807)
TITLE: [Attention] Use FA3 for MLA on Hopper (#12807)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases(introducing)
STAGE1: change_default; artifacts=L3.flash_attn.v0_backend,L3.flash_attn.v1_backend; Dossier shows L3.flash_attn.v0_backend behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend
FILES: vllm/attention/backends/flash_attn.py (+11/-33); vllm/attention/backends/mla/utils.py (+2/-0); vllm/attention/backends/utils.py (+34/-0); vllm/v1/attention/backends/flash_attn.py (+4/-26)
LABELS: ready, v1
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:97a3d6d995 (fix PR 13310)
BODY: Make sure we are using FA3 for MLA prefill on Hopper

### L3-ef533d25fb  (L3, 2025-02-06, sha ef533d25fba4, PR #12848)
TITLE: [Bugfix] FA2 illegal memory access (#12848)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
STAGE1: repair_correctness; artifacts=L3.flash_attn.fork_inline_cmake; Dossier shows L3.flash_attn.fork_inline_cmake behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1)
LABELS: ready, ci/build
ISSUES: #10389 [Bug]: v0.6.4.post1 crashed：Error in model execution: CUDA error: an illegal memory access was encountered | #11340 [Bug]: CUDA illegal memory access in flash attention only for specific values of --max-num-seqs (with AWQ model )
BODY: PR in the flash-attention repo: https://github.com/vllm-project/flash-attention/pull/42 ⏎  ⏎ should hopefully resolve: ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/10389 ⏎ FIX https://github.com/vllm-project/vllm/issues/11340

### L3-fe743b798d  (L3, 2025-02-09, sha fe743b798dfa, PR #12959)
TITLE: [bugfix] fix early import of flash attention (#12959)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_build_dependency; artifacts=L3.flash_attn.v0_backend,L3.flash_attn.v1_backend; Dossier shows L3.flash_attn.v0_backend behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend
FILES: vllm/attention/backends/flash_attn.py (+7/-6); vllm/attention/backends/mla/utils.py (+3/-2); vllm/attention/backends/utils.py (+6/-8); vllm/v1/attention/backends/flash_attn.py (+4/-3)
LABELS: ready, v1
BODY: the rlhf test example has been broken by https://github.com/vllm-project/vllm/pull/12807 , because it imports vllm flash attention too early.

### L3-f0b2da72a8  (L3, 2025-02-13, sha f0b2da72a84a, PR #13181)
TITLE: Expand MLA to support most types of quantization (#13181)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: extend_support; artifacts=NEW:attention_adapter; Dossier shows NEW:attention_adapter behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/mla/utils.py (+26/-45); vllm/config.py (+1/-31); vllm/model_executor/model_loader/loader.py (+34/-56)
LABELS: ready
BODY: For MLA, aside from the FP8 case, we need to have access to the unquantized weights for the decode kernel.  ⏎ This PR implements a general dequantization for quantization methods in vLLM by creating the identity matrix as input to `layer.quant_method.apply()` in order to get the transposed, unquantized weights as the output. ⏎  ⏎ Thanks to @LucasWilkinson and @tlrmchlsmth for iterating and coming up with this idea. ⏎  ⏎ Manual testing: ⏎  ⏎ #### DeepSeekV2 Lite ⏎  ⏎ ``` ⏎ vllm (pretrained=TechxGenus/DeepSeek-Coder-V2-Lite-Instruct-AWQ,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7119|±  |0.0125| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.6823|±  |0.0128| ⏎ ``` ⏎  ⏎ ``` ⏎ vllm (pretrained=TechxGenus/DeepSeek-Coder-V2-Lite-Instruct-AWQ,quantization=moe_wna16,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7142|±  |0.0124| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.6831|±  |0.0128| ⏎ ``` ⏎  ⏎      ⏎ TP=2 ⏎ ``` ⏎ vllm (pretrained=neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8,tensor_parallel_size=2,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7657|±  |0.0117| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.7528|±  |0.0119| ⏎ ```` ⏎  ⏎ TP=1 ⏎ ``` ⏎ vllm (pretrained=nm-testing/DeepSeek-Coder-V2-Lite-Instruct-FP8,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7665|±  |0.0117| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.7422|±  |0.0120| ⏎ ``` ⏎ TP=2 ⏎ ``` ⏎ vllm (pretrained=nm-testing/DeepSeek-Coder-V2-Lite-Instruct-FP8,tensor_parallel_size=2,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Val …[truncated]

### L3-ba59b78a9c  (L3, 2025-02-13, sha ba59b78a9c5a, PR #12790)
TITLE: [ROCm][V1] Add intial ROCm support to V1 (#12790)
SOURCES: path_core, symbol_pickaxe
STAGE1: introduce; artifacts=L3.flash_attn.v1_backend,L3.triton.prefix_prefill,L3.rocm.v1_rocm_attn,L3.platform.rocm_selection; Dossier shows L3.flash_attn.v1_backend behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.triton.prefix_prefill, L3.rocm.v1_rocm_attn, L3.platform.rocm_selection
FILES: vllm/attention/ops/prefix_prefill.py (+4/-2); vllm/v1/attention/backends/flash_attn.py (+4/-1); vllm/v1/attention/backends/rocm_attn.py (+182/-0); requirements-rocm-build.txt (+16/-0); vllm/platforms/rocm.py (+30/-15)
LABELS: rocm, ready, ci/build, v1
BODY: This Pr is adds initial support for V1 on AMD systems. It uses the `vllm/attention/ops/prefix_prefill.py` kernel instead of flash-attn.  ⏎  ⏎ Current install instructions if you want to try it out. You should start with a new virtual environment. All of the below commands assume you are in the vllm source directory ⏎ `pip install -r requirements-rocm-build.txt` ⏎ `python setup.py develop` ⏎  ⏎ And here's an example command to run ⏎ `VLLM_USE_V1=1 python examples/offline_inference/basic.py`

### L3-97a3d6d995  (L3, 2025-02-14, sha 97a3d6d99522, PR #13310)
TITLE: [Bugfix] Massage MLA's usage of flash attn for RoCM (#13310)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases
STAGE1: repair_correctness; artifacts=NEW:attention_adapter; Dossier shows NEW:attention_adapter behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/mla/utils.py (+11/-2)
LABELS: bug, rocm, ready
DEEP_STUDY: deep-study correctness case vllm:97a3d6d995: class=hardware_compiler_specific; symptom=crash_or_exception; introducing=#12807
BODY: This PR massages some `vllm_flash_attn` vs `flash_attn` interface differences that appeared between a couple of PRs: ⏎  ⏎ https://github.com/vllm-project/vllm/pull/12662 added a this fallback to `flash_attn` in order to support MLA on RoCM: ⏎ https://github.com/vllm-project/vllm/blob/5e5c8e091eacc16672a0a8265eb5cb0ece85d24b/vllm/attention/backends/mla/utils.py#L33-L36 ⏎  ⏎ Subsequently https://github.com/vllm-project/vllm/pull/12807 updated to use the `vllm_flash_attn` specific arguments that control whether we use FA2 or FA3.

### L3-54ed913f34  (L3, 2025-02-15, sha 54ed913f3437, PR #13323)
TITLE: [ci/build] update flashinfer (#13323)
SOURCES: subject_keyword, dependency_pin, release_notes
STAGE1: repair_build_dependency; artifacts=L3.flashinfer.v0_backend; Dossier shows L3.flashinfer.v0_backend behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile (+5/-2)
LABELS: ready, ci/build
BODY: we have work together with flashinfer to make the wheel python version agnostic, see https://github.com/flashinfer-ai/flashinfer/pull/823 . And they have released a new version with better MLA kernel support, we can use their official wheel right now.

### L3-0023cd2b9d  (L3, 2025-02-19, sha 0023cd2b9dfe, PR #13560)
TITLE: [ROCm] MI300A compile targets deprecation (#13560)
SOURCES: path_core
STAGE1: deprecate; artifacts=L3.flash_attn.fork_inline_cmake,L3.rocm.custom_paged,L3.rocm.rocm_flash_attn_v0; Dossier shows L3.flash_attn.fork_inline_cmake behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.rocm.custom_paged, L3.rocm.rocm_flash_attn_v0
FILES: csrc/rocm/attention.cu (+1/-2); vllm/attention/backends/rocm_flash_attn.py (+1/-2); CMakeLists.txt (+1/-1); csrc/quantization/fp8/amd/hip_float8_impl.h (+1/-2)
LABELS: rocm, ready, ci/build
PERF_LINES: Removing gfx940 and gfx941 targets. These have been deprecated in favor of gfx942 for MI300X
BODY: Removing gfx940 and gfx941 targets. These have been deprecated in favor of gfx942 for MI300X

### L3-71face8540  (L3, 2025-02-20, sha 71face854004, PR #13620)
TITLE: [Bugfix] Fix max_num_batched_tokens for MLA (#13620)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=NEW:attention_adapter; Dossier shows NEW:attention_adapter behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: -
FILES: vllm/config.py (+14/-6)
LABELS: bug, ready
BODY: Without this change we are stuck with a very short allowed prompt length by default with deepseek models ⏎  ⏎ Before this PR: ⏎ ``` ⏎ vllm serve /home/vllm-dev/DeepSeek-R1 --trust-remote-code --tensor-parallel-size 8 ⏎ ... ⏎ WARNING 02-20 17:08:38 scheduler.py:949] Input prompt (100005 tokens) is too long and exceeds limit of 2048 ⏎ ``` ⏎  ⏎ You can get past this manually by setting max_num_batched_tokens, but we should take care of this automatically ⏎ ``` ⏎ vllm serve /home/vllm-dev/DeepSeek-R1 --trust-remote-code --tensor-parallel-size 8 --max-num-batched-tokens 163840 ⏎ ``` ⏎  ⏎ Client: ⏎ ```python ⏎ from openai import OpenAI ⏎  ⏎ openai_api_key = "EMPTY" ⏎ openai_api_base = "http://localhost:8000/v1" ⏎  ⏎ client = OpenAI(api_key=openai_api_key, base_url=openai_api_base) ⏎ model_id = client.models.list().data[0].id ⏎  ⏎ # Query a prompt of roughly 100k tokens by repeating a simple token. ⏎ prompt = ("A " * 100000).strip() ⏎ chat_response = client.chat.completions.create( ⏎     model=model_id, ⏎     messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}], ⏎ ) ⏎ print("Text Chat completion output:", chat_response.choices[0].message.content) ⏎ ```

### L3-288cc6c234  (L3, 2025-02-21, sha 288cc6c234d0, PR #12639)
TITLE: [Attention] MLA with chunked prefill (#12639)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: introduce; artifacts=L3.cache.cuda_reshape,L3.flash_attn.v1_backend,L3.merge.triton_lse,L3.mla.triton_v0; Dossier shows L3.cache.cuda_reshape behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flash_attn.v1_backend, L3.merge.triton_lse, L3.mla.triton_v0
FILES: csrc/cache_kernels.cu (+159/-0); csrc/torch_bindings.cpp (+6/-0); vllm/_custom_ops.py (+10/-0); vllm/attention/backends/mla/common.py (+1503/-0); vllm/attention/backends/mla/utils.py (+0/-515); vllm/attention/backends/triton_mla.py (+14/-650); vllm/attention/ops/triton_merge_attn_states.py (+84/-0); vllm/config.py (+0/-13); vllm/engine/arg_utils.py (+3/-4); vllm/utils.py (+4/-0); vllm/v1/attention/backends/flash_attn.py (+2/-69); csrc/cache.h (+7/-0); csrc/core/math.hpp (+0/-5); csrc/cuda_utils.h (+18/-4); csrc/quantization/cutlass_w8a8/scaled_mm_c3x.cu (+3/-2); tests/kernels/test_cache.py (+73/-2); vllm/attention/__init__.py (+4/-8); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+20/-3)
LABELS: ready, v1
BODY: Need to do more benchmarking to see if this makes sense to be on by default in V0, but lays the groundwork for a V1 implementation. (https://github.com/vllm-project/vllm/pull/13111 may help performance) ⏎  ⏎ ``` ⏎ lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384,enable_chunked_prefill=False --task gsm8k --num_fewshot=5 --limit 100 ⏎  ⏎ vllm (pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384,enable_chunked_prefill=False), gen_kwargs: (None), limit: 100.0, num_fewshot: 5, batch_size: 1 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.66|±  |0.0476| ⏎ |     |       |strict-match    |     5|exact_match|↑  | 0.66|±  |0.0476| ⏎  ⏎ lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384,enable_chunked_prefill=True --task gsm8k --num_fewshot=5 --limit 100 ⏎  ⏎ vllm (pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384,enable_chunked_prefill=True), gen_kwargs: (None), limit: 100.0, num_fewshot: 5, batch_size: 1 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.66|±  |0.0476| ⏎ |     |       |strict-match    |     5|exact_match|↑  | 0.66|±  |0.0476| ⏎ ``` ⏎  ⏎ Shout-out to @pathorn for assisting with hardening this PR  ⏎  ⏎ Future work:

### L3-68d630a0c7  (L3, 2025-02-21, sha 68d630a0c727, PR #13650)
TITLE: [ROCM] fix native attention function call (#13650)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.rocm.rocm_flash_attn_v0; Dossier shows L3.rocm.rocm_flash_attn_v0 behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+0/-1)
LABELS: ready
ISSUES: #13648 [Bug]: [ROCM] _sdpa_attention() takes from 8 to 9 positional arguments but 10 were given
BODY: fix native attention function call for navi cards ⏎  ⏎ FIX #13648

### L3-558db8083c  (L3, 2025-02-22, sha 558db8083cfd, PR #13095)
TITLE: [V1][Kernel] Refactor the prefix_prefill kernel so that the caller no longer has to pass in the context lengths (#13095)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: adapt_framework; artifacts=L3.paged.python_wrapper,L3.xformers.v0_backend,L3.triton.prefix_prefill,L3.rocm.rocm_flash_attn_v0,L3.rocm.v1_rocm_attn; Dossier shows L3.paged.python_wrapper behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.triton.prefix_prefill, L3.rocm.rocm_flash_attn_v0, L3.rocm.v1_rocm_attn
FILES: vllm/attention/backends/rocm_flash_attn.py (+0/-1); vllm/attention/backends/xformers.py (+0/-1); vllm/attention/ops/paged_attn.py (+1/-3); vllm/attention/ops/prefix_prefill.py (+9/-8); vllm/v1/attention/backends/rocm_attn.py (+0/-12); tests/kernels/test_prefix_prefill.py (+2/-6)
LABELS: ready, v1
BODY: This patch changes the prefix_prefill kernel so that it will calculate the context length using the query length and the sequence length, both of which are already passed in. This makes the kernel a bit more usable on V1 where we don't keep track of the context lengths tensor in the attention meta data.

### L3-444b0f0f62  (L3, 2025-02-24, sha 444b0f0f6298, PR #12513)
TITLE: [Misc][Docs] Raise error when flashinfer is not installed and `VLLM_ATTENTION_BACKEND` is set (#12513)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L3.dispatch.selector,L3.flashinfer.v0_backend; Dossier shows L3.dispatch.selector behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: -
FILES: vllm/config.py (+9/-0); docs/source/getting_started/quickstart.md (+10/-0)
LABELS: documentation, ready
BODY: FlashAttn is currently not resolved when installing through a package manager, as it still resorting to manual installation (see https://github.com/vllm-project/vllm/blob/main/Dockerfile#L210). ⏎ If the user manually sets the backend to `flashinfer` and `flashinfer` is not installed in the system, the resulting error message is not super informative: ⏎ ``` ⏎ Traceback (most recent call last): ⏎   File "/opt/dlami/nvme/vllm/vllm/worker/model_runner_base.py", line 116, in _wrapper ⏎     return func(*args, **kwargs) ⏎   File "/opt/dlami/nvme/vllm/vllm/worker/model_runner.py", line 1670, in execute_model ⏎     self.attn_state.begin_forward(model_input) ⏎   File "/opt/dlami/nvme/vllm/vllm/attention/backends/flashinfer.py", line 269, in begin_forward ⏎     model_input.attn_metadata.prefill_wrapper = state._get_prefill_wrapper( ⏎   File "/opt/dlami/nvme/vllm/vllm/attention/backends/flashinfer.py", line 121, in _get_prefill_wrapper ⏎     self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper( ⏎ TypeError: 'NoneType' object is not callable ⏎ ``` ⏎  ⏎ We could instead simply check whether the backend is *specifically* set to `flashinfer` and verify the package can be found, raising an error if that's not the case. ⏎  ⏎ When VLLM_ATTENTION_BACKEND is not set, I think it's fine to default to one of the other backends. However, if the environment variable is set, I believe it is best practice not to overwrite it.

### L3-cdc1fa12eb  (L3, 2025-02-24, sha cdc1fa12eb1b, PR #13555)
TITLE: Remove unused kwargs from model definitions (#13555)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=NEW:attention_adapter; Dossier shows NEW:attention_adapter behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+7/-12); docs/source/contributing/model/basic.md (+0/-2); docs/source/contributing/model/multimodal.md (+0/-2); tests/kernels/test_encoder_decoder_attn.py (+3/-11); vllm/model_executor/layers/mamba/mamba_mixer.py (+3/-2); vllm/model_executor/layers/mamba/mamba_mixer2.py (+2/-2); vllm/model_executor/models/adapters.py (+1/-5); vllm/model_executor/models/arctic.py (+5/-19); vllm/model_executor/models/aria.py (+0/-5); vllm/model_executor/models/baichuan.py (+5/-19); vllm/model_executor/models/bamba.py (+7/-22); vllm/model_executor/models/bart.py (+20/-73); vllm/model_executor/models/bert.py (+12/-32); vllm/model_executor/models/blip2.py (+1/-6); vllm/model_executor/models/bloom.py (+7/-24); vllm/model_executor/models/chameleon.py (+4/-23); vllm/model_executor/models/chatglm.py (+8/-34); vllm/model_executor/models/commandr.py (+5/-19); vllm/model_executor/models/dbrx.py (+7/-28); vllm/model_executor/models/deepseek.py (+6/-20); vllm/model_executor/models/deepseek_mtp.py (+4/-15); vllm/model_executor/models/deepseek_v2.py (+7/-24); vllm/model_executor/models/deepseek_vl2.py (+0/-5); vllm/model_executor/models/eagle.py (+1/-6); vllm/model_executor/models/exaone.py (+6/-24); vllm/model_executor/models/falcon.py (+7/-24); vllm/model_executor/models/florence2.py (+6/-28); vllm/model_executor/models/fuyu.py (+0/-5); vllm/model_executor/models/gemma.py (+5/-19); vllm/model_executor/models/gemma2.py (+5/-19); vllm/model_executor/models/glm4v.py (+3/-7); vllm/model_executor/models/gpt2.py (+8/-24); vllm/model_executor/models/gpt_bigcode.py (+8/-24); vllm/model_executor/models/gpt_j.py (+7/-24); vllm/model_executor/models/gpt_neox.py (+7/-24); vllm/model_executor/models/granite.py (+6/-23); vllm/model_executor/models/granitemoe.py (+6/-20); vllm/model_executor/models/gritlm.py (+3/-6); vllm/model_executor/models/idefics3.py (+0/-9); vllm/model_executor/models/interfaces_base.py (+3/-6); (+64 more)
LABELS: documentation, speculative-decoding, ready, v1
BODY: Follow up for #11967 which removes `kv_cache` and `attn_metadata` from all model definitions. ⏎  ⏎ Summary of changes: ⏎ - Remove `kv_cache` and `attn_metadata` from `Attention.forward()` args ⏎ - Use forward context instead of `attn_metadata` arg in `MambaMixer` and `MambaMixer2` ⏎ - Remove `kv_caches`, `kv_cache` and `attn_metadata` from `forward()` args of all model modules ⏎ - Remove `kv_caches` and `attn_metadata` from new model docs ⏎ - Leave `kv_caches` arg (but try not to use it) in all child classes of `ModelRunnerBase.execute_model()` to avoid further complication

### L3-75e9d49796  (L3, 2025-02-25, sha 75e9d4979658, PR #13468)
TITLE: [Bugfix] Initialize attention bias on the same device as Query/Key/Value (#13468)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.xformers.v0_backend; Dossier shows L3.xformers.v0_backend behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.xformers.v0_backend
FILES: vllm/attention/backends/xformers.py (+6/-4)
LABELS: ready
BODY: The attention bias in vLLM's xformers backend is currently initialized on the default device, rather than the device of the Q/K/V tensors: ⏎  ⏎ https://github.com/vllm-project/vllm/blob/b53d79983c273b2775456d99c0e0890aea073512/vllm/attention/backends/xformers.py#L676-L677 ⏎  ⏎ And here is how xformers decide which device to use: ⏎  ⏎ <https://github.com/facebookresearch/xformers/blob/8d91ce05a2f6a5ae059593922a631b9ff325b134/xformers/ops/fmha/attn_bias.py#L742>: ⏎  ⏎ ```python ⏎ class BlockDiagonalMask(AttentionBias): ⏎     ... ⏎     def from_seqlens( ⏎         cls, ⏎         q_seqlen: Sequence[int], ⏎         kv_seqlen: Optional[Sequence[int]] = None, ⏎         *, ⏎         device: Optional[torch.device] = None, ⏎     ) -> "BlockDiagonalMask": ⏎         ... ⏎         device = _get_default_bias_device(device) ⏎ ``` ⏎  ⏎ https://github.com/facebookresearch/xformers/blob/8d91ce05a2f6a5ae059593922a631b9ff325b134/xformers/ops/fmha/attn_bias.py#L90 ⏎  ⏎ ```python ⏎ def _get_default_bias_device(device: Optional[torch.device] = None) -> torch.device: ⏎     if device is None: ⏎         if torch.cuda.is_available(): ⏎             return torch.device("cuda") ⏎         return torch.device("cpu") ⏎     return device ⏎ ``` ⏎  ⏎ This becomes problematic when vLLM is used in conjunction with libraries like `trl` for GRPO training.  In such cases, vLLM might be assigned to run on a specific GPU (e.g., the next available GPU after those used for training, which is the default behaviour of `trl`). ⏎  ⏎ For example, if I have 8 GPUs and use `cuda:0` to `cuda:6` for GRPO training, vLLM will then be assigned to `cuda:7`. However, the current attention bias initialization will place the bias on `cuda:0`, leading to the following error: ⏎  ⏎ ```console ⏎ [rank0]: ValueError: Attention bias and Query/Key/Value should be on the same device ⏎ [rank0]:   query.device: cuda:7 ⏎ [rank0]:   attn_bias   : cuda:0 ⏎ ``` ⏎  ⏎ This PR will probably solve this issue.

### L3-18e505930d  (L3, 2025-02-25, sha 18e505930d78, PR #13725)
TITLE: [Bugfix] Support MLA for CompressedTensorsWNA16 (#13725)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L3.mla.triton_v0; Dossier shows L3.mla.triton_v0 behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.mla.triton_v0
FILES: vllm/attention/backends/mla/common.py (+7/-7)
LABELS: bug, ready
BODY: Without this change, models that use CompressedTensorsWNA16  will fail with MLA enabled due to their usage of `weight_packed`  ⏎  ⏎ Reference: https://github.com/vllm-project/vllm/blob/db986c19ea35d7f3522a45d5205bf5d3ffab14e4/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py#L137 ⏎  ⏎ Error: ⏎ ``` ⏎ [rank0]:   File "/home/mgoin/code/vllm/vllm/attention/backends/mla/common.py", line 1155, in process_weights_after_loading ⏎ [rank0]:     weight_dtype = get_layer_weight(self.kv_b_proj).dtype ⏎ [rank0]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ [rank0]:   File "/home/mgoin/code/vllm/vllm/attention/backends/mla/common.py", line 1138, in get_layer_weight ⏎ [rank0]:     raise AttributeError( ⏎ [rank0]: AttributeError: Layer 'ColumnParallelLinear(in_features=512, output_features=4096, bias=False, tp_size=1, gather_output=False)' has neither weight nor qweight ⏎ ```

### L3-1d35662e6d  (L3, 2025-02-26, sha 1d35662e6dc1, PR #13844)
TITLE: [ROCm] Disable chunked prefill/prefix caching when running MLA on non-cuda platforms (#13844)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: change_default; artifacts=L3.mla.triton_v0; Dossier shows L3.mla.triton_v0 behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.mla.triton_v0
FILES: vllm/attention/backends/mla/common.py (+30/-12); vllm/config.py (+14/-0)
LABELS: rocm, ready
BODY: `flash_attn_varlen_func` in upstream `flash-attn` does not support the `return_softmax_lse` argument. This PR works around that issue by explicitly disabling chunked prefill and prefix caching on non-cuda platforms. ⏎  ⏎ Here are the results from running llmeval with `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` on an AMD system: ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7657|±  |0.0117| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.7453|±  |0.0120| ⏎ ``` ⏎  ⏎ Here are the results from the same model on an H100 system without this PR ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7642|±  |0.0117| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.7468|±  |0.0120| ⏎ ```

### L3-378b3ef6f8  (L3, 2025-02-26, sha 378b3ef6f834, PR #13922)
TITLE: [ROCm][V1] Update reshape_and_cache to properly work with CUDA graph padding (#13922)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.cache.cuda_reshape; Dossier shows L3.cache.cuda_reshape behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+1/-1)
LABELS: ready
ISSUES: #13418 [Bug][v1][rocm] cuda graph gets stuck in case padding is used to meet a captured input size
BODY: This patch updates reshape_and_cache to account for the case where slot_mapping.size(0) is not equal to key.size(0). This occurs when we use CUDA graph padding on V1. ⏎  ⏎ Fixes: https://github.com/vllm-project/vllm/issues/13418

### L3-f95903909f  (L3, 2025-02-27, sha f95903909f07, PR #13747)
TITLE: [Kernel] FlashMLA integration (#13747)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, corpus:production-kernel-provenance
STAGE1: integrate; artifacts=L3.flash_attn.upstream_pip,L3.flash_attn.fork_inline_cmake,L3.flash_attn.fork_build,L3.mla.triton_v0,L3.mla.flashmla_v0_adapter,L3.mla.flashmla_build,L3.platform.cuda_selection; Dossier shows L3.flash_attn.upstream_pip behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake, L3.flash_attn.fork_build, L3.mla.triton_v0, L3.mla.flashmla_v0_adapter, L3.mla.flashmla_build, L3.platform.cuda_selection
FILES: CMakeLists.txt (+4/-73); cmake/external_projects/flashmla.cmake (+66/-0); cmake/external_projects/vllm_flash_attn.cmake (+67/-0); setup.py (+6/-0); vllm/_custom_ops.py (+64/-0); vllm/attention/backends/flashmla.py (+239/-0); vllm/attention/backends/mla/common.py (+15/-13); vllm/attention/ops/flashmla.py (+115/-0); vllm/platforms/cuda.py (+24/-0); vllm/platforms/interface.py (+1/-0); tests/kernels/test_flashmla.py (+132/-0)
LABELS: ready, ci/build
ISSUES: #13735 [Feature]: Support for FlashMLA
BODY: Integrate: https://github.com/deepseek-ai/FlashMLA ⏎  ⏎ currently requires ⏎  ⏎ ``` ⏎ export VLLM_ATTENTION_BACKEND=FLASHMLA ⏎ ``` ⏎ and ⏎ ``` ⏎ block_size=64 ⏎ ``` ⏎  ⏎ TODO: ⏎  ⏎ - ~cuda-graphs are broken~ ⏎ - future PR: enforce block_size 64 gracefully  ⏎  ⏎ Closes #13735

### L3-58d1b2aa77  (L3, 2025-02-27, sha 58d1b2aa772d, PR #13789)
TITLE: [Attention] MLA support for V1 (#13789)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: introduce; artifacts=L3.flash_attn.v1_backend,L3.mla.common_v1,L3.platform.cuda_selection; Dossier shows L3.flash_attn.v1_backend behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.mla.common_v1, L3.platform.cuda_selection
FILES: vllm/attention/layer.py (+24/-11); vllm/model_executor/models/deepseek_v2.py (+11/-2); vllm/platforms/cuda.py (+7/-2); vllm/platforms/interface.py (+1/-0); vllm/v1/attention/backends/flash_attn.py (+67/-2); vllm/v1/attention/backends/mla/__init__.py (+0/-0); vllm/v1/attention/backends/mla/common.py (+1022/-0); vllm/v1/attention/backends/triton_mla.py (+110/-0); vllm/v1/worker/gpu_input_batch.py (+63/-1); vllm/v1/worker/gpu_model_runner.py (+35/-41)
LABELS: ready, v1
BODY: This PR is co-authored with Lucas Wilkinson. ⏎  ⏎ ``` ⏎ VLLM_USE_V1="1" lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384,enforce_eager=True --task gsm8k --num_fewshot=5 --limit 100 ⏎ ... ⏎ vllm (pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384,enforce_eager=True), gen_kwargs: (None), limit: 100.0, num_fewshot: 5, batch_size: 1 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.66|±  |0.0476| ⏎ |     |       |strict-match    |     5|exact_match|↑  | 0.66|±  |0.0476| ⏎ ```

### L3-8294773e48  (L3, 2025-02-27, sha 8294773e48a2, PR #13718)
TITLE: [core] Perf improvement for DSv3 on AMD GPUs (#13718)
SOURCES: path_core, symbol_pickaxe
STAGE1: optimize; artifacts=L3.triton.decode_attention,L3.mla.triton_v0; Dossier shows L3.triton.decode_attention behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.triton.decode_attention, L3.mla.triton_v0
FILES: vllm/attention/backends/mla/common.py (+72/-20); vllm/attention/ops/triton_decode_attention.py (+10/-5); vllm/model_executor/layers/fused_moe/configs/E=256,N=256,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json (+128/-0)
LABELS: rocm, ready
PERF_LINES: 1. Add GPUs tweaks for AMD; 2. Allow triton-fa for prefill stage in MLA path; 3. Add MoE config for MI300X
BODY: 1. Add GPUs tweaks for AMD; 2. Allow triton-fa for prefill stage in MLA path; 3. Add MoE config for MI300X

### L3-2e94b9cfbb  (L3, 2025-02-27, sha 2e94b9cfbb41, PR #13867)
TITLE: [Attention] Flash MLA for V1 (#13867)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: integrate; artifacts=L3.mla.common_v1,L3.mla.flashmla_v1_adapter,L3.platform.cuda_selection; Dossier shows L3.mla.common_v1 behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+22/-13); vllm/platforms/interface.py (+2/-3); vllm/v1/attention/backends/mla/common.py (+7/-4); vllm/v1/attention/backends/mla/flashmla.py (+139/-0); vllm/v1/attention/backends/mla/triton_mla.py (+0/-0)
LABELS: ready, ci/build, v1
BODY: use via: `VLLM_ATTENTION_BACKEND=FLASHMLA VLLM_USE_V1=1` ⏎  ⏎ Results: ⏎  ⏎ https://docs.google.com/spreadsheets/d/1toxQVaA7UPhmY57kv08Wdq0xtIcU8oBbqb9RE3Wy2_E/edit?usp=sharing

### L3-b28246f6ff  (L3, 2025-03-01, sha b28246f6ff16, PR #14065)
TITLE: [ROCm][V1][Bugfix] Add get_builder_cls method to the ROCmAttentionBackend class (#14065)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.rocm.v1_rocm_attn; Dossier shows L3.rocm.v1_rocm_attn behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn
FILES: vllm/v1/attention/backends/rocm_attn.py (+6/-1)
LABELS: ready, v1
BODY: The ROCmAttentionBackend currently just uses the FlashAttentionMetadata so it can use the FlashAttentionMetadataBuilder. This PR is required now that the V1 GPUModelRunner supports other meta data structures.

### L3-848a6438ae  (L3, 2025-03-03, sha 848a6438aed2, PR #12348)
TITLE: [ROCm] Faster Custom Paged Attention kernels (#12348)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: optimize; artifacts=L3.rocm.custom_paged,L3.rocm.rocm_flash_attn_v0; Dossier shows L3.rocm.custom_paged behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.rocm.custom_paged, L3.rocm.rocm_flash_attn_v0
FILES: csrc/rocm/attention.cu (+1084/-422); requirements-rocm.txt (+1/-1); vllm/attention/backends/rocm_flash_attn.py (+2/-2); .buildkite/run-amd-test.sh (+0/-1); benchmarks/kernels/benchmark_paged_attention.py (+51/-20); tests/kernels/test_attention.py (+7/-1)
LABELS: documentation, rocm, frontend, speculative-decoding, ready, ci/build
PERF_LINES: GPU: MI300X | | CPA Version | Input length | Output length | KV-cache-dtype | Quantization | Prompt numbers | Req/s | Total Tokens/s | Output Tokens/s |
BODY: ## Description ⏎ This PR implements a faster Custom Paged Attention (CPA) kernel based on `mfma16x16x16` instructions. ⏎ This feature is from ROCm/vllm (https://github.com/ROCm/vllm/pull/372). ⏎  ⏎ ## End-to-End Performance gain ⏎ Model: Llama-3.1-70B-Instruct ⏎ Tensor Parallelism: 1 ⏎ GPU: MI300X ⏎  ⏎ | CPA Version | Input length | Output length | KV-cache-dtype | Quantization | Prompt numbers | Req/s | Total Tokens/s | Output Tokens/s | ⏎ |-------------|--------------|---------------|----------------|--------------|----------------|-------|----------------|-----------------| ⏎ | before changes | 128          | 128           | fp8_e4m3       | fp8          | 200            | 13.05 | 3340.6         | 1670.3          | ⏎ | before changes | 128          | 256           | fp8_e4m3       | fp8          | 200            | 7.56  | 2901.31        | 1934.21         | ⏎ | before changes | 128          | 2048          | fp8_e4m3       | fp8          | 200            | 0.78  | 1698.35        | 1598.45         | ⏎ | before changes | 512          | 128           | fp8_e4m3       | fp8          | 200            | 6.44  | 4122.57        | 824.51          | ⏎ | before changes | 512          | 256           | fp8_e4m3       | fp8          | 200            | 4.48  | 3443.46        | 1147.82         | ⏎ | before changes | 512          | 2048          | fp8_e4m3       | fp8          | 200            | 0.66  | 1696.64        | 1357.31         | ⏎ | before changes | ShareGPT |               | fp8_e4m3       | fp8          | 1000           | 6.22  | 2574.19        | 1234.64         | ⏎ | optimized   | 128          | 128           | fp8_e4m3       | fp8          | 200            | 15.11 | 3867.75        | 1933.87         | ⏎ | optimized   | 128          | 256           | fp8_e4m3       | fp8          | 200            | 9.01  | 3459.98        | 2306.65         | ⏎ | optimized   | 128          | 2048          | fp8_e4m3       | fp8          | 200            | 1.2   | 2609.04        | 2455.57         | ⏎ | optimized   | 512          | 128           | fp8_e4m3       | fp8          | 200            | 7.33  | 4694.05        | 938.81          | ⏎ | optimized   | 512          | 256           | fp8_e4m3       | fp8          | 200            | 5.5   | 4223.29        | 1407.76         | ⏎ | optimized   | 512          | 2048          | fp8_e4m3       | fp8          | 200            | 1.03  | 2648.55        | 2118.84         | ⏎ | optimized   | ShareGPT |               | fp8_e4m3       | fp8          | 1000           | 7.45  | 3081.14        | 1477.79         |

### L3-72c62eae5f  (L3, 2025-03-04, sha 72c62eae5f01, PR #13931)
TITLE: [V1] EP/TP MoE + DP Attention (#13931)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.dispatch.abstract_interface,L3.platform.cuda_selection; Dossier shows L3.dispatch.abstract_interface behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/attention/layer.py (+2/-2); examples/offline_inference/data_parallel.py (+10/-7); tests/kernels/test_moe.py (+1/-0); vllm/compilation/backends.py (+3/-2); vllm/forward_context.py (+15/-7); vllm/model_executor/layers/fused_moe/layer.py (+155/-32); vllm/model_executor/models/aria.py (+11/-7); vllm/model_executor/models/dbrx.py (+6/-2); vllm/model_executor/models/jamba.py (+15/-6); vllm/model_executor/models/mixtral.py (+2/-0); vllm/model_executor/models/olmoe.py (+3/-1); vllm/model_executor/models/phimoe.py (+4/-1); vllm/model_executor/models/qwen2_moe.py (+5/-2); vllm/platforms/cuda.py (+9/-0); vllm/utils.py (+2/-2); vllm/v1/engine/core.py (+0/-1); vllm/v1/worker/gpu_model_runner.py (+7/-3)
LABELS: documentation, ready, v1
BODY: Based on https://github.com/vllm-project/vllm/pull/13591 ⏎  ⏎ DP+EP implemented via collective ops in the fused_moe layer's forward pass.

### L3-4dacaa4a83  (L3, 2025-03-05, sha 4dacaa4a8346, PR #14255)
TITLE: [BugFix] Fix prefix caching V0 MLA (#14255)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.mla.triton_v0; Dossier shows L3.mla.triton_v0 behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.mla.triton_v0
FILES: vllm/attention/backends/mla/common.py (+24/-20)
LABELS: ready
BODY: Fix for:  ⏎  ⏎ https://github.com/vllm-project/vllm/issues/14069 ⏎ https://github.com/vllm-project/vllm/issues/14009 ⏎  ⏎ based on: ⏎  ⏎ https://github.com/vllm-project/vllm/issues/14069#issuecomment-2696104480

### L3-f6bb18fd9a  (L3, 2025-03-05, sha f6bb18fd9a19, PR #14253)
TITLE: [BugFix] MLA + V1, illegal memory access and accuracy issues (#14253)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.flash_attn.v1_backend,L3.mla.common_v1,L3.mla.flashmla_v1_adapter; Dossier shows L3.flash_attn.v1_backend behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.mla.common_v1, L3.mla.flashmla_v1_adapter
FILES: vllm/v1/attention/backends/flash_attn.py (+2/-2); vllm/v1/attention/backends/mla/common.py (+178/-127); vllm/v1/attention/backends/mla/flashmla.py (+33/-25); vllm/v1/attention/backends/mla/triton_mla.py (+5/-2); vllm/v1/worker/gpu_input_batch.py (+23/-2); vllm/v1/worker/gpu_model_runner.py (+4/-2); tests/v1/worker/test_gpu_input_batch.py (+89/-1)
LABELS: ready, v1
BODY: Fix the decode / prefill split in V1 with MLA not being handled correctly leading to inaccurate results when `max_num_batched_tokens` is low and illegal memory accesses. Also refactored the code to precompute the tensor slicing for prefill and decode metadata. ⏎  ⏎ Main: ⏎  ⏎ ``` ⏎ ** max_num_batched_tokens=1024 ⏎  ⏎ VLLM_USE_V1=1 lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384,enforce_eager=True,max_num_batched_tokens=1024 --task gsm8k --num_fewshot=5 --limit 10 ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |  0.5|±  |0.1667| ⏎ |     |       |strict-match    |     5|exact_match|↑  |  0.5|±  |0.1667| ⏎  ⏎ ** max_num_batched_tokens=Default ⏎  ⏎ VLLM_USE_V1=1 lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9, ⏎ trust_remote_code=True,max_model_len=16384,enforce_eager=True --task gsm8k --num_fewshot=5 --limit 10 ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |  0.8|±  |0.1333| ⏎ |     |       |strict-match    |     5|exact_match|↑  |  0.8|±  |0.1333| ⏎ ``` ⏎  ⏎ This PR: ⏎  ⏎ ``` ⏎ ** max_num_batched_tokens=1024 ⏎  ⏎ VLLM_USE_V1=1 lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384,enforce_eager=True,max_num_batched_tokens=1024 --task gsm8k --num_fewshot=5 --limit 10 ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |  0.8|±  |0.1333| ⏎ |     |       |strict-match    |     5|exact_match|↑  |  0.8|±  |0.1333| ⏎  ⏎ VLLM_USE_V1=1 VLLM_ATTENTION_BACKEND=FLASHMLA  lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384,enforce_eager=True,max_num_batched_tokens=1024 --task gsm8k --num_fewshot=5 --limit 10 ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5| …[truncated]

### L3-5ee10e990d  (L3, 2025-03-05, sha 5ee10e990deb, PR #11301)
TITLE: [Bugfix][CI] ALiBi test case in xformers multi_query_kv_attention (#11301)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.xformers.v0_backend; Dossier shows L3.xformers.v0_backend behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.xformers.v0_backend
FILES: vllm/attention/backends/xformers.py (+0/-2); tests/kernels/test_attention.py (+78/-17); tests/kernels/test_prefix_prefill.py (+5/-3)
LABELS: ready
DEEP_STUDY: deep-study correctness case vllm:5ee10e990d: class=shape_alignment_edge; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: I've been meaning to add a test case to address this TODO https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_attention.py#L367 but I then realized the test would break due to a dim expansion of the attention bias happening here https://github.com/vllm-project/vllm/blob/main/vllm/attention/backends/xformers.py#L783. ⏎  ⏎ Therefore, this PR adds a test case for xformers multi query+alibi bias and fixes the previously untested scenario.
