### L3-c27df94e1f  (L3, 2024-11-25, sha c27df94e1ff9, PR #9850)
TITLE: [Bugfix] Fix chunked prefill with model dtype float32 on Turing Devices (#9850)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/prefix_prefill.py (+28/-13); pyproject.toml (+1/-0); tests/conftest.py (+19/-0); tests/kernels/test_prefix_prefill.py (+63/-0); vllm/config.py (+10/-0); vllm/engine/arg_utils.py (+1/-0)
LABELS: ready
ISSUES: #9096 [Bug]: vllm serve Exception in ASGI application
BODY: The issue is a bug on Triton with Turing devices, there are already opened issues there:  ⏎  ⏎ https://github.com/triton-lang/triton/issues/3011 ⏎ https://github.com/triton-lang/triton/issues/3787 ⏎  ⏎ In a nutshell, it is a limitation of Turing devices to perform matmul with float32. The issue #9096 shows this behavior by casting the model to dtype=float32. A way to fix/workaround is to set input_precision on `tl.dot` to use 'ieee'.  ⏎  ⏎ I also update …[truncated]

### L3-a6760f6456  (L3, 2024-11-25, sha a6760f6456b7, PR #9228)
TITLE: [Feature] vLLM ARM Enablement for AARCH64 CPUs (#9228)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: csrc/cpu/attention.cpp (+17/-1); Dockerfile.arm (+62/-0); cmake/cpu_extension.cmake (+24/-9); csrc/cpu/cpu_types.hpp (+4/-2); csrc/cpu/cpu_types_arm.hpp (+515/-0); docs/source/getting_started/arm-installation.rst (+50/-0); docs/source/index.rst (+1/-0); examples/offline_inference.py (+1/-1); requirements-cpu.txt (+4/-3)
LABELS: documentation, ready, ci/build
BODY: **Description** ⏎ This PR enables support of vLLM for AARCH64 architecture. Motivated by the requirements from (#176, #5741, etc), I implemented this PR which enables the ARM path for CPU inference.   ⏎  ⏎ **ARM Compatibility:** Modified the build scripts and configuration files to ensure compatibility with ARM processors. It currently supports float32, fp16 and bfloat16 datatypes. ⏎  ⏎ **Motivation** ⏎ Enabling vLLM on ARM architecture broadens its usabilit …[truncated]

### L3-9a88f89799  (L3, 2024-11-25, sha 9a88f897993a, PR #10121)
TITLE: custom allreduce + torch.compile (#10121)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/source/getting_started/debugging.rst (+0/-1); tests/distributed/test_pynccl.py (+6/-9); tests/distributed/test_utils.py (+0/-2); vllm/distributed/device_communicators/pynccl.py (+13/-13); vllm/distributed/parallel_state.py (+36/-74); vllm/v1/worker/gpu_model_runner.py (+4/-2)
LABELS: documentation, ready
BODY: This Pr changes pynccl all reduce to be out of place and removes support for torch distributed's all reduce.

### L3-e85250b1d1  (L3, 2024-11-26, sha e85250b1d164, PR #10667)
TITLE: [Hardware][Gaudi]add get_name method for HPUAttentionBackend (#10667)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/hpu_attn.py (+4/-0)
LABELS: ready
BODY: `HPUAttentionBackend` lack of `get_name` method. will throw error when choose backend. ⏎  ⏎ cc @kzawora-intel

### L3-c411def234  (L3, 2024-11-27, sha c411def234b0, PR #10722)
TITLE: [torch.compile] fix shape specialization (#10722)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config.py (+3/-4)
BODY: example running command: ⏎  ⏎ `vllm serve meta-llama/Llama-3.1-8B-Instruct -O '{"level": 3, "inductor_specialize_for_cudagraph_no_more_than": 16}'`

### L3-9a8bff0285  (L3, 2024-11-28, sha 9a8bff028595, PR #10736)
TITLE: [Kernel] Update vllm-flash-attn version (#10736)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1)
LABELS: ready, ci/build
BODY: This PR updates the vllm-flash-attn version to take advantage of https://github.com/vllm-project/flash-attention/pull/28 This PR should give a small speedup for `flash_attn_varlen_func`, because we now skip two redundant copy operations.

### L3-8c1e77fb58  (L3, 2024-11-28, sha 8c1e77fb585c, PR #10742)
TITLE: [Kernel] Update vllm-flash-attn version to reduce CPU overheads (#10742)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1)
LABELS: ready, ci/build
BODY: Upgrades to https://github.com/vllm-project/flash-attention/pull/30, which will help reduce CPU overheads in launching the kernels.

### L3-98f47f2a40  (L3, 2024-11-28, sha 98f47f2a4032, PR #10733)
TITLE: [V1] Optimize the CPU overheads in FlashAttention custom op (#10733)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+9/-8)
LABELS: ready
BODY: With piece-wise CUDA graphs, we have to make sure that the attention custom op causes minimal CPU overheads. This PR made a few changes to optimize the CPU overheads in the FlashAttention custom op: ⏎ 1. ~~We directly use `torch.ops.vllm_flash_attn_c.varlen_fwd` rather than `flash_attn_varlen_func`, since `FlashAttnFunc` which inherits `torch.autograd.Function` causes unnecessary overheads.~~ ⏎ 2. We move the reshapes and shape check logics to outsid …[truncated]

### L3-073a4bd1c0  (L3, 2024-12-01, sha 073a4bd1c041, PR #10811)
TITLE: [Kernel] Use `out` arg in flash_attn_varlen_func (#10811)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1); vllm/v1/attention/backends/flash_attn.py (+3/-3); tests/kernels/test_flash_attn.py (+17/-3)
LABELS: ready, ci/build
BODY: This PR uses the in-place `out` argument of `flash_attn_varlen_func` to avoid redundant copy. This is possible by the change in https://github.com/vllm-project/flash-attention/pull/32

### L3-e25810ae29  (L3, 2024-12-02, sha e25810ae2905, PR #10799)
TITLE: Fill TorchSDPAAttentionMetadata seq_lens_field for prefill (#10799)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/torch_sdpa.py (+5/-1)
LABELS: ready
BODY: Fix attempt for https://github.com/vllm-project/vllm/issues/10738 ⏎  ⏎ In the TorchSDPAAttentionMetadata the seq_lens_tensor is empty for prefill. It looks like this is the case because it was assumed that this field is not necessary for prefill since it's not used for `_run_sdpa_forward`. But in Roberta models we need to know how long each sequence is during prefill to fix the position_ids tensor. ⏎  ⏎ cc: @DarkLight1337 , @bigPYJ1151

### L3-a4c4daf364  (L3, 2024-12-02, sha a4c4daf3642a, PR #10822)
TITLE: [misc] use out argument for flash attention (#10822)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: vllm/attention/backends/abstract.py (+1/-0); vllm/attention/backends/blocksparse_attn.py (+2/-0); vllm/attention/backends/flash_attn.py (+19/-36); vllm/attention/backends/flashinfer.py (+4/-0); vllm/attention/backends/hpu_attn.py (+1/-0); vllm/attention/backends/ipex_attn.py (+1/-0); vllm/attention/backends/pallas.py (+1/-0); vllm/attention/backends/rocm_flash_attn.py (+1/-0); vllm/attention/backends/torch_sdpa.py (+1/-0); vllm/attention/backends/xformers.py (+1/-0); (+3 more)
LABELS: ready
BODY: replace https://github.com/vllm-project/vllm/pull/9740 since that is very old. ⏎  ⏎ To make mypy happy, all attention forward signature needs to match. so i add `output: Optional[torch.Tensor] = None,` for all attention.

### L3-dc5ce861bf  (L3, 2024-12-03, sha dc5ce861bf0e, PR #10838)
TITLE: [torch.compile] remove compilation_context and simplify code (#10838)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/piecewise/test_simple.py (+4/-5); tests/compile/piecewise/test_toy_llama.py (+17/-16); tests/models/decoder_only/language/test_jamba.py (+3/-2); tests/models/decoder_only/language/test_mamba.py (+3/-2); tests/worker/test_encoder_decoder_model_runner.py (+2/-2); tests/worker/test_model_runner.py (+3/-2); vllm/compilation/backends.py (+0/-4); vllm/compilation/compile_context.py (+0/-23); vllm/config.py (+75/-8); vllm/model_executor/models/jamba.py (+2/-4); (+4 more)
LABELS: ready
BODY: remove the confusing compilation context, figure out cudagraph batchsizes during initialization of the config.

### L3-10398b4706  (L3, 2024-12-04, sha 10398b4706ee, PR #10893)
TITLE: [Model] Consolidate ViTs attention implementation without mask (#10893)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+63/-0); vllm/model_executor/models/blip.py (+4/-41); vllm/model_executor/models/clip.py (+4/-42); vllm/model_executor/models/glm4_vision_encoder.py (+6/-16); vllm/model_executor/models/idefics2_vision_model.py (+4/-21); vllm/model_executor/models/intern_vit.py (+4/-24); vllm/model_executor/models/internvl.py (+11/-12); vllm/model_executor/models/molmo.py (+9/-29); vllm/model_executor/models/siglip.py (+4/-41)
LABELS: ready
BODY: - This PR adds a `MultiHeadAttention` layer to consolidate existing duplicate attention forward implementation in ViTs. ⏎ - ViTs that use attention mask are not modified: Qwen2-VL, Pixtral, Mllama.

### L3-e4c34c23de  (L3, 2024-12-04, sha e4c34c23de2a, PR #9621)
TITLE: [CI/Build] improve python-only dev setup (#9621)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen
FILES: setup.py (+79/-4); docs/source/getting_started/installation.rst (+12/-29); python_only_dev.py (+9/-87); vllm/envs.py (+2/-1)
LABELS: documentation, ready, ci/build
BODY: The current process for python-only development and/or builds is kinda painful. ⏎  ⏎ With this PR I'm trying to resurrect the `VLLM_USE_PRECOMPILED` env var to skip extension compilation and make the overall process easier. ⏎  ⏎ The overall idea is to include the compiled libraries from the nightly wheel in the built package (when the env var is set), so that it can be installed as a regular package (e.g with `pip install -e .`). ⏎ I'm including the p …[truncated]

### L3-aa39a8e175  (L3, 2024-12-05, sha aa39a8e17537, PR #10827)
TITLE: [Doc] Create a new "Usage" section (#10827)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+1/-1); docs/source/design/multimodal/multimodal_index.rst (+1/-4); docs/source/index.rst (+15/-10); docs/source/models/enabling_multimodal_inputs.rst (+1/-1); docs/source/models/supported_models.rst (+17/-2); docs/source/serving/openai_compatible_server.md (+2/-2); docs/source/usage/compatibility_matrix.rst (+0/-0); docs/source/usage/engine_args.rst (+0/-0); docs/source/usage/env_vars.rst (+0/-0); docs/source/usage/faq.rst (+2/-0); (+15 more)
LABELS: documentation, ready
BODY: Some of the pages under "Serving" and "Models" aren't really related to their parent section. This PR creates a new "Usage" section to accommodate these pages. ⏎  ⏎ ~~@simon-mo can you help set up redirects as requested in #10428?~~ Done

### L3-f13cf9ad50  (L3, 2024-12-07, sha f13cf9ad5049, PR #10060)
TITLE: [Build] Fix for the Wswitch-bool clang warning (#10060)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/paged_attention_v1.cu (+4/-7); csrc/attention/paged_attention_v2.cu (+4/-7)
LABELS: ready
BODY: Refactor switch on boolean into if to address the Wswitch-bool compiler warning. ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-d1c2e15eb3  (L3, 2024-12-08, sha d1c2e15eb31e, PR #11005)
TITLE: [torch.compile] add dynamo time tracking (#11005)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/compilation/backends.py (+6/-0); vllm/compilation/decorators.py (+3/-3); vllm/compilation/monitor.py (+7/-2)
BODY: ```bash ⏎ $ vllm serve meta-llama/Meta-Llama-3-8B ⏎ Graph capturing finished in 10 secs, took 0.32 GiB ⏎ init engine (profile, create kv cache, warmup model) took 14.18 seconds ⏎  ⏎ $ vllm serve meta-llama/Meta-Llama-3-8B -O3 ⏎ Dynamo bytecode transform time: 4.60 s ⏎ Compiling a graph for general shape takes 14.77 s ⏎ torch.compile takes 19.37 s in total ⏎ Graph capturing finished in 15 secs, took 0.33 GiB ⏎ init engine (profile, create kv cache, warmup model) took …[truncated]

### L3-3b61cb450d  (L3, 2024-12-09, sha 3b61cb450d89, PR #10989)
TITLE: [V1] Further reduce CPU overheads in flash-attn (#10989)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flash_attn.v1_backend
FILES: csrc/cache_kernels.cu (+12/-2); vllm/v1/attention/backends/flash_attn.py (+16/-5)
LABELS: ready
BODY: This PR reduces the CPU ops in V1 flash-attn: two slice ops for `key` and `value` by slightly modifying the `reshape_and_cache_flash` op. ⏎ Also, it uses `kv_cache.unbind(0)` instead of `kv_cache[0]` and `kv_cache[1]`, to reduce the number of ops.

### L3-cbcbdb1ceb  (L3, 2024-12-09, sha cbcbdb1ceb9c, PR #11028)
TITLE: [Bugfix][Hardware][Gaudi] Bump vllm_hpu_extension version (#11028)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/hpu_attn.py (+11/-0); requirements-hpu.txt (+1/-1)
LABELS: ci/build
BODY: vllm_hpu_extension had `vllm.utils.get_vllm_instance_id` dependency, which was removed in https://github.com/vllm-project/vllm/pull/10976, causing HPU backend to crash on vllm_hpu_extension import. Extension was updated to remove that dependency (https://github.com/HabanaAI/vllm-hpu-extension/pull/52), and this PR contains that fix.  ⏎ Small changes were needed to hpu_attn.py, as vllm_hpu_extension had a minor paged attention API change, and the ex …[truncated]

### L3-75f89dc44c  (L3, 2024-12-10, sha 75f89dc44c6e, PR #11059)
TITLE: [torch.compile] add a flag to track batchsize statistics (#11059)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.trtllm_gen
FILES: vllm/v1/attention/backends/flash_attn.py (+1/-0); vllm/envs.py (+3/-0); vllm/forward_context.py (+31/-1); vllm/v1/worker/gpu_model_runner.py (+2/-0)
LABELS: ready
BODY: I find that https://github.com/vllm-project/vllm/pull/11031 only records the statistics from the engine level, which is not enough. ⏎  ⏎ we need to record the forward batchsize for every model forward, including using cudagraph or not. ⏎  ⏎ then we can have accurate statistics of batchsizes, and can optimize the shapes we want to compile.

### L3-9a93973708  (L3, 2024-12-11, sha 9a93973708d7, PR #11071)
TITLE: [Bugfix] Fix Mamba multistep (#11071)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/placeholder_attn.py (+63/-1); vllm/worker/multi_step_model_runner.py (+3/-1)
LABELS: ready
BODY: Implements `advance_step` for `PlaceholderAttention` so that Mamba and other attention-free models function with multistep scheduling. ⏎  ⏎ Fixes the following error in buildkite ⏎ ``` ⏎ ValueError: Multi-Step not supported for attention backend: NO_ATTENTION. Set VLLM_ATTENTION_BACKEND to a value from [‘FLASH_ATTN’, ‘ROCM_FLASH’, ‘FLASHINFER’]. ⏎ ```

### L3-66aaa7722d  (L3, 2024-12-11, sha 66aaa7722df3, PR #11110)
TITLE: [torch.compile] remove graph logging in ci (#11110)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/compilation/backends.py (+5/-3)
BODY: our ci runs with debug level logging by default, and these two lines produce too many lines of logging. ⏎  ⏎ now that we switched to a dedicated system for logging `torch.compile` related information in https://github.com/vllm-project/vllm/pull/10972 , these two lines are not necessary anymore.

### L3-cad5c0a6ed  (L3, 2024-12-11, sha cad5c0a6eda0, PR #11093)
TITLE: [Doc] Update docs to refer to pooling models (#11093)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/placeholder_attn.py (+1/-1); docs/source/usage/faq.rst (+6/-1); vllm/config.py (+4/-4); vllm/core/placeholder_block_space_manager.py (+1/-1); vllm/engine/arg_utils.py (+2/-2); vllm/engine/async_llm_engine.py (+1/-1); vllm/engine/multiprocessing/client.py (+1/-1); vllm/engine/protocol.py (+1/-1); vllm/entrypoints/openai/serving_score.py (+1/-1); vllm/sequence.py (+3/-3); (+4 more)
LABELS: documentation, frontend, ready
BODY: A follow-up to #10820 that updates more documentation to refer to the more general "pooling models" instead of "embedding models". ⏎  ⏎ For more information on pooling models, please read [this page](https://docs.vllm.ai/en/latest/models/pooling_models.html) and [this RFC to automatically convert text generation models into pooling models](https://github.com/vllm-project/vllm/pull/10674).

### L3-30870b4f66  (L3, 2024-12-13, sha 30870b4f6641, PR #10906)
TITLE: [torch.compile] Dynamic fp8 + rms_norm fusion (#10906)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+2/-1); benchmarks/fused_kernels/layernorm_rms_benchmarks.py (+173/-0); csrc/dispatch_utils.h (+14/-0); csrc/ops.h (+8/-0); csrc/quantization/fp8/common.cuh (+7/-19); csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu (+160/-0); csrc/quantization/fused_kernels/layernorm_utils.cuh (+327/-0); csrc/quantization/fused_kernels/quant_conversions.cuh (+81/-0); csrc/quantization/vectorization.cuh (+33/-0); csrc/torch_bindings.cpp (+8/-0); (+10 more)
LABELS: ready, ci/build
BODY: This PR adds support for RMSNorm + (fp8) quant fusion. It also refactors the fusion pass to make it easier to add patterns. That includes support for multiple values of epsilon.

### L3-0a56bcc03d  (L3, 2024-12-13, sha 0a56bcc03de0, PR #11169)
TITLE: [Bugfix][Hardware][CPU] Enable Gemma2 with SDPA on CPU backend (#11169)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/torch_sdpa.py (+4/-3)
LABELS: ready
BODY: Make sure is_causal is only True when mask is None as required by SDPA otherwise there is an error raised ⏎  ⏎ `RuntimeError: _scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True` ⏎  ⏎ With models such as Gemma2, that have alternating layers of sliding window attention and full attention, while self.need_mask is correctly set for each layer, the mask itself is set once via attn_metadata.set_attn_bias() and is ⏎ not None e …[truncated]

### L3-6d917d0eeb  (L3, 2024-12-14, sha 6d917d0eebd0, PR #11105)
TITLE: Enable mypy checking on V1 code (#11105)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+2/-0); tools/mypy.sh (+1/-0); vllm/v1/core/kv_cache_manager.py (+5/-5); vllm/v1/core/kv_cache_utils.py (+9/-8); vllm/v1/core/scheduler.py (+1/-0); vllm/v1/engine/__init__.py (+14/-9); vllm/v1/engine/async_llm.py (+6/-5); vllm/v1/engine/core.py (+10/-10); vllm/v1/engine/core_client.py (+24/-19); vllm/v1/engine/detokenizer.py (+2/-2); (+11 more)
LABELS: ready
BODY: As per #10959 ⏎  ⏎ Some of these definitely deserve closer scrutiny e.g.: ⏎  ⏎ - ~~commit 7974f1d - was there a bug with the `hash_block_tokens()` call?~~ ⏎ - ~~commit 226fd552089ea8f0aeb49c8c2a4cc62f2b32742a - feels we'll be fighting an uphill battle with mypy and this args/kwargs pattern~~ ⏎ - ~~commit 8a6d88c42df22d9ef8d1deb444a9335117b91dc6 - `MultiprocExecutor.workers` and `UniprocExecutor.worker` can be set to `None`, so we have to litter the code with …[truncated]

### L3-a1c02058ba  (L3, 2024-12-14, sha a1c02058baf4, PR #11081)
TITLE: [torch.compile] allow tracking forward time (#11081)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/forward_context.py (+42/-19)
LABELS: ready
BODY: when benchmarking `torch.compile` performance, I always get this question: if the performance gain is not satisfactory, is it because `torch.compile` does not optimize the model well, or is it because of the scheduling overhead? ⏎  ⏎ this pr adds the tracking for forward time, so that we can directly test the perf of `torch.compile` for certain sizes. ⏎  ⏎ e.g. ⏎  ⏎ ```bash ⏎ $ VLLM_LOG_BATCHSIZE_INTERVAL=1.0 python3 benchmarks/benchmark_latency.py --model met …[truncated]

### L3-b3b1526f03  (L3, 2024-12-16, sha b3b1526f0390, PR #11212)
TITLE: WIP: [CI/Build] simplify Dockerfile build for ARM64 / GH200 (#11212)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_pip
FILES: Dockerfile (+32/-8); requirements-build.txt (+1/-1); requirements-cuda-arm64.txt (+3/-0); requirements-cuda.txt (+2/-2); docs/source/serving/deploying_with_docker.rst (+26/-0)
LABELS: documentation, ready, ci/build
BODY: From PR: [10499](https://github.com/vllm-project/vllm/pull/10499) ⏎ Fix Issue: [2021](https://github.com/vllm-project/vllm/issues/2021) ⏎  ⏎ This contribution focuses on simplifying the Dockerfile build process for ARM64 systems. Unnecessary build from source has been removed and requirements handling has been optimized to ensure the correct installation of torch and bitsandbytes for ARM64+CUDA compatibility. The changes have been tested on the Nvidia  …[truncated]

### L3-88a412ed3d  (L3, 2024-12-16, sha 88a412ed3d96, PR #11108)
TITLE: [torch.compile] fast inductor (#11108)
SOURCES: symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/compilation/backends.py (+210/-3); vllm/config.py (+411/-4); vllm/envs.py (+3/-0)
LABELS: ready
BODY: directly bypass aot-autograd and inductor, and load from cache

### L3-f9ecbb18bf  (L3, 2024-12-17, sha f9ecbb18bf03, PR #11252)
TITLE: [Misc] Allow passing logits_soft_cap for xformers backend (#11252)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend
FILES: vllm/attention/backends/xformers.py (+3/-5)
LABELS: ready
BODY: - Similar to #11169, allow passing `logits_soft_cap` to enable Gemma2 inference for xformers backend. ⏎  ⏎ "The Gemma 2 team observed very minor differences when soft-capping is removed during inference." (https://huggingface.co/blog/gemma2#soft-capping-and-attention-implementations)

### L3-60508ffda9  (L3, 2024-12-18, sha 60508ffda91c, PR #10995)
TITLE: [Kernel]: Cutlass 2:4 Sparsity + FP8/Int8 Quant Support (#10995)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+16/-10); benchmarks/cutlass_benchmarks/sparse_benchmarks.py (+384/-0); benchmarks/cutlass_benchmarks/utils.py (+96/-0); benchmarks/cutlass_benchmarks/w8a8_benchmarks.py (+2/-26); benchmarks/cutlass_benchmarks/weight_shapes.py (+1/-1); csrc/core/math.hpp (+7/-0); csrc/cutlass_extensions/common.cpp (+11/-0); csrc/cutlass_extensions/common.hpp (+35/-0); csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp (+2/-2); csrc/ops.h (+9/-0); (+20 more)
LABELS: ready, ci/build
BODY: # Summary ⏎ - Add sparse quantized and unquantized kernels for CUTLASS 3.x. ⏎ - Add compressed tensors support for 2of4 Sparse Only, 2of4 Sparse + INT8/FP8 Quantized Models ⏎  ⏎ From Neural Magic

### L3-32aa2059ad  (L3, 2024-12-23, sha 32aa2059addd, PR #11145)
TITLE: [Docs] Convert rST to MyST (Markdown) (#11145)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.rocm.rocm_flash_attn_v0
FILES: .gitignore (+2/-0); Dockerfile (+1/-1); docs/requirements-docs.txt (+1/-1); docs/source/automatic_prefix_caching/apc.md (+102/-0); docs/source/automatic_prefix_caching/apc.rst (+0/-110); docs/source/community/meetups.md (+15/-0); docs/source/community/meetups.rst (+0/-16); docs/source/conf.py (+1/-1); docs/source/contributing/dockerfile/dockerfile.md (+50/-0); docs/source/contributing/dockerfile/dockerfile.rst (+0/-50); (+157 more)
LABELS: documentation, frontend, ready, ci/build
ISSUES: #10427 [Doc]: Migrate to Markdown
BODY: This PR migrates the existing documentation from reStructuredText to [MyST Markdown](https://myst-parser.readthedocs.io/en/latest/index.html), a flavour of Markdown extended for use with Sphinx. ⏎  ⏎ - Add `.md` template to gitignore ⏎ - Upgrade `myst-parser` to `3.0.1` ⏎ - Converts `.rst` docs into `.md`, preserving content and formatting ⏎ - Update filename references to docs from `.rst` to `.md` ⏎ - In front-end code, update class/function definition refe …[truncated]

### L3-5c7963249d  (L3, 2024-12-24, sha 5c7963249daf, PR #11463)
TITLE: [attn][tiny fix] fix attn backend in MultiHeadAttention (#11463)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+1/-0)
LABELS: ready
BODY: The original logic will never step into `xformer` backend branch, because `attn_backend` is the class object of a certain attention backend.  ⏎ This pr fix that.

### L3-970d6d0776  (L3, 2024-12-30, sha 970d6d077607, PR #11607)
TITLE: [Build][Kernel] Update CUTLASS to v3.6.0 (#11607)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+2/-2); csrc/cutlass_extensions/vllm_cutlass_library_extension.py (+9/-9); csrc/quantization/machete/generate.py (+4/-4); csrc/quantization/machete/machete_collective_builder.cuh (+4/-6); csrc/quantization/machete/machete_mainloop.cuh (+4/-7); csrc/quantization/machete/machete_prepacked_layout.cuh (+2/-3)
LABELS: ready, ci/build
BODY: Picks up the [official CUTLASS 3.6.0 release](https://github.com/NVIDIA/cutlass/releases/tag/v3.6.0) which lets us turn GIT_SHALLOW back on. ⏎  ⏎ As of https://github.com/NVIDIA/cutlass/pull/1972 the `MixedInput` kernel schedule tags no longer exist, so Machete kernels are updated accordingly. See https://github.com/NVIDIA/cutlass/discussions/1956.

### L3-73001445fb  (L3, 2025-01-01, sha 73001445fbfc, PR #11635)
TITLE: [V1] Implement Cascade Attention (#11635)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1); vllm/v1/attention/backends/flash_attn.py (+254/-13); vllm/v1/worker/gpu_model_runner.py (+95/-1); tests/conftest.py (+7/-0); tests/kernels/test_cascade_flash_attn.py (+182/-0); tests/system_messages/sonnet3.5_nov2024.txt (+71/-0); tests/v1/e2e/__init__.py (+0/-0); tests/v1/e2e/test_cascade_attention.py (+22/-0); vllm/v1/core/kv_cache_manager.py (+51/-1); vllm/v1/core/scheduler.py (+10/-0)
LABELS: ready, ci/build
BODY: This PR implements a simple version of [Cascade Attention](https://flashinfer.ai/2024/02/02/cascade-inference.html). Cascade attention can save the HBM bandwidth for reading KV cache when requests share the same prefix. ⏎  ⏎ NOTE: For simplicity, this PR only uses cascade attention when every running request shares the same KV cache. If one or more requests do not share the KV cache, cascade attention is not used.

### L3-4068f4b5b5  (L3, 2025-01-05, sha 4068f4b5b5dc, PR #11730)
TITLE: [MISC] Replace c10::optional with std::optional (#11730)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.rocm.custom_paged
FILES: csrc/attention/paged_attention_v1.cu (+2/-2); csrc/attention/paged_attention_v2.cu (+2/-2); csrc/cpu/attention.cpp (+4/-4); csrc/rocm/attention.cu (+2/-2); csrc/cpu/quant.cpp (+5/-5); csrc/cpu/torch_bindings.cpp (+3/-3); csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c2x.hpp (+3/-3); csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp (+3/-3); csrc/cutlass_extensions/torch_utils.hpp (+1/-1); csrc/mamba/causal_conv1d/causal_conv1d.cu (+12/-12); (+14 more)
BODY: In PyTorch, c10::optional is just an alias of std::optional for a while, and we would like to completely eliminate all the usage of c10::optional. ⏎  ⏎ Context: in PyTorch repo, c10::optional's usage is (almost) gone, see https://github.com/search?q=repo%3Apytorch%2Fpytorch%20c10%3A%3Aoptional&type=code. std::optional is recommended.

### L3-ee77fdb5de  (L3, 2025-01-06, sha ee77fdb5de42, PR #11755)
TITLE: [Doc][2/N] Reorganize Models and Usage sections (#11755)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+1/-1); .github/ISSUE_TEMPLATE/600-new-model.yml (+1/-1); docs/source/assets/features/disagg_prefill/abstraction.jpg (+0/-0); docs/source/assets/features/disagg_prefill/overview.jpg (+0/-0); docs/source/contributing/model/basic.md (+102/-0); docs/source/contributing/model/index.md (+26/-0); docs/source/contributing/model/multimodal.md (+2/-6); docs/source/contributing/model/registration.md (+56/-0); docs/source/design/automatic_prefix_caching.md (+4/-2); docs/source/design/kernel/paged_attention.md (+2/-0); (+35 more)
LABELS: documentation, ready, ci/build
BODY: - Renamed Usage section to Features. ⏎   - Moved Performance and Tuning page to Performance section, and renamed it to Optimization and Tuning. ⏎   - Moved Engine Arguments, Environment Variables, Usage Stats Collection pages to Serving section. ⏎   - To conserve space, Quantization and Automatic Prefix Caching are now part of Features section. ⏎ - Renamed API Documentation section to API Reference. ⏎ - Renamed Design section to Design Documents. ⏎ - Renamed …[truncated]

### L3-e20c92bb61  (L3, 2025-01-07, sha e20c92bb6183, PR #11690)
TITLE: [Kernel] Move attn_type to Attention.__init__() (#11690)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: vllm/attention/backends/abstract.py (+1/-1); vllm/attention/backends/blocksparse_attn.py (+7/-7); vllm/attention/backends/flash_attn.py (+3/-1); vllm/attention/backends/flashinfer.py (+7/-8); vllm/attention/backends/hpu_attn.py (+7/-6); vllm/attention/backends/ipex_attn.py (+6/-6); vllm/attention/backends/pallas.py (+7/-6); vllm/attention/backends/rocm_flash_attn.py (+7/-7); vllm/attention/backends/torch_sdpa.py (+3/-1); vllm/attention/backends/xformers.py (+4/-2); (+8 more)
LABELS: ready
BODY: Move attn_type to Attention.__init__(), so that we can get attn_type from the Attention class. ⏎ It is needed by ⏎ 1. https://github.com/vllm-project/vllm/issues/11382 which needs to know whether the kv_cache is a decoder/encoder_decoder/encoder by analyzing the Attention class ⏎ 2. https://github.com/vllm-project/vllm/pull/11677 to bind kv_cache to the correct Attention module ⏎  ⏎ It also moves the following check from model execution to model initializa …[truncated]

### L3-d848800e88  (L3, 2025-01-09, sha d848800e884f, PR #11298)
TITLE: [Misc] Move `print_*_once` from utils to logger (#11298)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend
FILES: vllm/attention/backends/torch_sdpa.py (+6/-3); vllm/attention/backends/xformers.py (+5/-3); .github/workflows/lint-and-deploy.yaml (+1/-0); vllm/config.py (+4/-5); vllm/entrypoints/chat_utils.py (+3/-4); vllm/inputs/preprocess.py (+11/-9); vllm/inputs/registry.py (+2/-2); vllm/logger.py (+52/-5); vllm/lora/peft_helper.py (+4/-2); vllm/lora/punica_wrapper/punica_selector.py (+5/-3); (+11 more)
LABELS: frontend, ready, ci/build
BODY: Clean up the `vllm.utils` file a bit by moving the `print_*_once` functions to the logger itself. ⏎  ⏎ I'm planning further changes to the `vllm.utils` file. It has become quite disorganized over time.

### L3-405eb8e396  (L3, 2025-01-09, sha 405eb8e3967e, PR #11609)
TITLE: [platform] Allow platform specify attention backend (#11609)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.dispatch.selector, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/attention/selector.py (+12/-127); vllm/platforms/cpu.py (+5/-2); vllm/platforms/cuda.py (+76/-1); vllm/platforms/hpu.py (+5/-2); vllm/platforms/interface.py (+5/-3); vllm/platforms/openvino.py (+5/-2); vllm/platforms/rocm.py (+4/-2); vllm/platforms/tpu.py (+5/-2); vllm/platforms/xpu.py (+5/-2); tests/kernels/test_attention_selector.py (+42/-32)
LABELS: ready
BODY: Part of https://github.com/vllm-project/vllm/issues/11162 ⏎  ⏎ This PR add the ability for platform to implement its own attention backend out-of-tree.

### L3-cf5f000d21  (L3, 2025-01-10, sha cf5f000d218f, PR #11677)
TITLE: [torch.compile] Hide KV cache behind torch.compile boundary (#11677)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+17/-12); tests/kernels/test_encoder_decoder_attn.py (+12/-6); tests/test_utils.py (+83/-2); tests/v1/engine/test_engine_core.py (+3/-0); tests/v1/engine/test_engine_core_client.py (+3/-0); vllm/config.py (+0/-1); vllm/forward_context.py (+20/-13); vllm/utils.py (+35/-0); vllm/v1/worker/gpu_model_runner.py (+5/-1); vllm/worker/cpu_enc_dec_model_runner.py (+2/-1); (+8 more)
LABELS: ready, v1
BODY: Put kv_cache in forward_context, so that we do not need to pass kv_cache to `unified_attention` and `unified_attention_with_output`.  ⏎  ⏎ We need this pr to support https://github.com/vllm-project/vllm/issues/9098 (hide continuous batching complexity through forward context) and https://github.com/vllm-project/vllm/issues/11382 (hybrid memory allocator, kv_cache type will be more complex than `List[torch.Tensor]`) ⏎  ⏎ I've tested this pr with `vllm ser …[truncated]

### L3-9dd02d85ca  (L3, 2025-01-13, sha 9dd02d85ca80, PR #11979)
TITLE: [Bug] Fix usage of `.transpose()` and `.view()` consecutively. (#11979)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+1/-1); vllm/model_executor/models/intern_vit.py (+1/-1)
LABELS: ready
ISSUES: #8630 [Bug]: OpenGVLab/InternVL2-Llama3-76B: view size is not compatible with input tensor's size and stride | #11978 [Bug]: The usage of .transpose() and .view() consecutively is not recommended.
BODY: Cooperate with @Fryezsh-edu ⏎  ⏎ FIX #8630 ⏎ FIX #11978  ⏎  ⏎ As I have said in the issue, this is a logic bug, which only occurs when `USE_XFORMERS_OPS = False`, so it is a little difficult to reproduce on common devices, but this bug really exists.

### L3-0f8cafe2d1  (L3, 2025-01-13, sha 0f8cafe2d155, PR #11967)
TITLE: [Kernel] unified_attention for Attention.forward (#11967)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+14/-12); vllm/utils.py (+0/-1); vllm/worker/hpu_model_runner.py (+11/-2); vllm/worker/hpu_worker.py (+3/-0); vllm/worker/neuron_model_runner.py (+10/-7); vllm/worker/openvino_model_runner.py (+3/-1); vllm/worker/openvino_worker.py (+11/-2); vllm/worker/tpu_model_runner.py (+18/-10); vllm/worker/tpu_worker.py (+5/-1); vllm/worker/xpu_model_runner.py (+12/-9)
LABELS: ready
DEEP_STUDY: deep-study: this PR was reverted by PR 12038 (partial_revert, reason=api_or_compat_break)
BODY: Get `kv_cache` and `attn_metadata` from forward context instead of `Attention.forward`'s argument by using `unified_attention` ⏎ This pr helps to achieve https://github.com/vllm-project/vllm/issues/9098 ⏎  ⏎ We can remove kv_cache and attn_metadata from all models after this pr! ⏎  ⏎ CC @youkaichao

### L3-2e0e017610  (L3, 2025-01-14, sha 2e0e01761049, PR #11981)
TITLE: [Platform] Add output for Attention Backend (#11981)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+4/-0); vllm/attention/backends/flash_attn.py (+2/-0); vllm/attention/layer.py (+1/-5); vllm/v1/attention/backends/flash_attn.py (+2/-0)
LABELS: ready
BODY: Part for https://github.com/vllm-project/vllm/issues/11162 ⏎  ⏎ This PR make `use_output` check to attention backend. So that each backend can decide its own way.

### L3-a2d2acb4c8  (L3, 2025-01-14, sha a2d2acb4c8d2, PR #12040)
TITLE: [Bugfix][Kernel] Give unique name to BlockSparseFlashAttention (#12040)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.blocksparse.v0
FILES: vllm/attention/backends/blocksparse_attn.py (+1/-2); vllm/platforms/interface.py (+1/-0)
LABELS: ready
BODY: As `BlockSparseFlashAttention` has less feature support than flash attention, e.g., not support `use_output`, which causes the crash of the following simplest script, I suggest create a unique name for it instead of reuse FLASH_ATTN. ⏎  ⏎ CC @Isotr0py Is there any historical reason for implementing `get_name` as `FLASH_ATTN`? ⏎  ⏎ The script crashed before this pr: ⏎ ``` ⏎ from vllm import LLM, SamplingParams ⏎  ⏎ # Sample prompts. ⏎ prompts = [ ⏎     "Hello, my nam …[truncated]

### L3-1f18adb245  (L3, 2025-01-14, sha 1f18adb2451e, PR #12038)
TITLE: [Kernel] Revert the API change of Attention.forward (#12038)
SOURCES: path_core, corpus:confirmed-reverts
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+2/-2)
DEEP_STUDY: deep-study revert record: partial_revert of PR(s) 11967 reason=api_or_compat_break
BODY: The changing of `kv_cache` -> `_kv_cache` and `attn_metadata` -> `_attn_metadata` in `Attention.forward` by https://github.com/vllm-project/vllm/pull/11967 breaks models that pass these two arguments with kwargs, e.g., phi3_small: ⏎ ``` ⏎ attn_output = self.attn(q, k, v, kv_cache, attn_metadata=attn_metadata) ⏎ ``` ⏎ And this script is crashed ⏎ ``` ⏎ from vllm import LLM, SamplingParams ⏎  ⏎ # Sample prompts. ⏎ prompts = [ ⏎     "Hello, my name is", ⏎     "The presid …[truncated]

### L3-3adf0ffda8  (L3, 2025-01-15, sha 3adf0ffda8de, PR #12023)
TITLE: [Platform] Do not raise error if _Backend is not found (#12023)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/layer.py (+4/-4); vllm/attention/selector.py (+11/-9); tests/kernels/test_attention_selector.py (+8/-3); tests/plugins/vllm_add_dummy_platform/vllm_add_dummy_platform/dummy_attention_backend.py (+8/-0); tests/plugins/vllm_add_dummy_platform/vllm_add_dummy_platform/dummy_platform.py (+4/-0); tests/plugins_tests/test_platform_plugins.py (+14/-0)
LABELS: ready
BODY: Part of https://github.com/vllm-project/vllm/issues/11162 ⏎  ⏎ The out-of-tree platform may have a different _Backend for attention backend. Ignore the error if the input _Backend is not the one in-tree.

### L3-0794e7446e  (L3, 2025-01-15, sha 0794e7446efc, PR #10467)
TITLE: [Misc] Add multipstep chunked-prefill support for FlashInfer (#10467)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+24/-5); vllm/worker/model_runner.py (+118/-102); vllm/worker/multi_step_model_runner.py (+1/-1); csrc/prepare_inputs/advance_step.cu (+10/-0); tests/multi_step/test_correctness_llm.py (+16/-1)
LABELS: ready
BODY: Support multi-step scheduling for chunked-prefill on FlashInfer, where prefill tokens are turned into decode tokens after the first single step.  ⏎  ⏎ cc @comaniac @yzh199 @WoosukKwon @youkaichao

### L3-69d765f5a5  (L3, 2025-01-17, sha 69d765f5a5bb, PR #11960)
TITLE: [V1] Move more control of kv cache initialization from model_executor to EngineCore (#11960)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+2/-0); tests/v1/test_utils.py (+62/-0); vllm/v1/core/kv_cache_utils.py (+124/-0); vllm/v1/engine/core.py (+18/-13); vllm/v1/executor/abstract.py (+8/-3); vllm/v1/executor/multiproc_executor.py (+15/-10); vllm/v1/executor/ray_executor.py (+21/-19); vllm/v1/executor/uniproc_executor.py (+14/-11); vllm/v1/kv_cache_interface.py (+111/-0); vllm/v1/utils.py (+54/-2); (+2 more)
LABELS: ready
BODY: This pr changes the workflow of `EngineCore._initialize_kv_caches` to enable more flexible control of kv cache format in the future. ⏎ It is splitted from https://github.com/vllm-project/vllm/pull/11938 and is a preparation for https://github.com/vllm-project/vllm/issues/11382 ⏎ Original workflow: ⏎ ```python ⏎ num_gpu_blocks, _ = self.model_executor.determine_num_available_blocks() ⏎ self.model_executor.initialize(num_gpu_blocks) ⏎ ``` ⏎ New workflow: ⏎ ```pyth …[truncated]

### L3-e66faf4809  (L3, 2025-01-19, sha e66faf4809ce, PR #12182)
TITLE: [torch.compile] store inductor compiled Python file (#12182)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/compilation/backends.py (+58/-22); vllm/config.py (+2/-11)
LABELS: ready
BODY: https://github.com/vllm-project/vllm/pull/11108 enabled the compilation cache, but it only stores a hash string, which is not human-readable. ⏎  ⏎ this PR further stores the path of the python file compiled by inductor, so that we can manually check the result. ⏎  ⏎ As a result, I can confirm that `torch.compile` does not do anything w.r.t. kv cache, thanks to https://github.com/vllm-project/vllm/pull/11677 . Therefore, we don't need to split the graph i …[truncated]

### L3-86bfb6dba7  (L3, 2025-01-20, sha 86bfb6dba7c6, PR #12218)
TITLE: [Misc] Pass `attention` to impl backend (#12218)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: vllm/attention/backends/abstract.py (+19/-4); vllm/attention/backends/blocksparse_attn.py (+6/-6); vllm/attention/backends/flash_attn.py (+5/-5); vllm/attention/backends/flashinfer.py (+8/-8); vllm/attention/backends/hpu_attn.py (+2/-2); vllm/attention/backends/ipex_attn.py (+9/-9); vllm/attention/backends/pallas.py (+3/-3); vllm/attention/backends/rocm_flash_attn.py (+10/-10); vllm/attention/backends/torch_sdpa.py (+8/-10); vllm/attention/backends/xformers.py (+9/-11); (+2 more)
LABELS: ready
BODY: With https://github.com/vllm-project/vllm/pull/11969, a quantization method can be implemented and register to vLLM out-of-tree now. But there is no way to use the registered parameter for attention layer in custom attention backend impl. ⏎  ⏎ This PR pass the attention object to attention backend, so that the backend can use the parameters registered by quantization attention method directly to keep the same with other kind of quantization method(Li …[truncated]

### L3-fa9ee08121  (L3, 2025-01-21, sha fa9ee08121d1, PR #12235)
TITLE: [Misc] Set default backend to SDPA for get_vit_attn_backend (#12235)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/vision.py (+16/-14)
LABELS: ready
BODY: Xformers works only on GPU, while torch SDPA  is more generic. This PR set the default attention backend to SDPA for qwen-vl, so that most non-GPU platform(especially out-of-tree ) can work by default with SDPA.

### L3-d4b62d4641  (L3, 2025-01-21, sha d4b62d464137, PR #11777)
TITLE: [AMD][Build] Porting dockerfiles from the ROCm/vllm fork (#11777)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile.rocm (+101/-157); Dockerfile.rocm_base (+158/-0); docs/source/getting_started/installation/gpu/rocm.inc.md (+6/-7); vllm/model_executor/layers/fused_moe/configs/E=8,N=14336,device_name=AMD_Instinct_MI300X.json (+18/-18); vllm/model_executor/layers/fused_moe/configs/E=8,N=1792,device_name=AMD_Instinct_MI300X.json (+18/-18); vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=AMD_Instinct_MI300X.json (+18/-18); vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=AMD_Instinct_MI300X.json (+18/-18)
LABELS: documentation, rocm, ready, ci/build
BODY: An attempt to unify the build process with how it is done in [ROCm/vllm](https://github.com/ROCm/vllm.git) ⏎ Split the build process into: ⏎ - Building the required dependency libraries using Dockerfile.rocm_base. Not needed for the end user, done by AMD. ⏎ - Building the vLLM itself on top of it. ⏎  ⏎ In addition to not building the libraries each time, the new base image is now much smaller (7GB on docker hub vs 18GB previously), which allows to make the …[truncated]

### L3-66818e5b63  (L3, 2025-01-22, sha 66818e5b6381, PR #12253)
TITLE: [core] separate builder init and builder prepare for each batch (#12253)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+6/-5); vllm/attention/backends/flash_attn.py (+6/-5); vllm/attention/backends/flashinfer.py (+8/-6); vllm/attention/backends/placeholder_attn.py (+5/-3); vllm/attention/backends/torch_sdpa.py (+4/-1); vllm/attention/backends/utils.py (+7/-6); vllm/worker/cpu_model_runner.py (+17/-7); vllm/worker/model_runner.py (+24/-12); vllm/worker/model_runner_base.py (+5/-0); vllm/worker/xpu_model_runner.py (+8/-2)
LABELS: ready
BODY: Right now we create the builder instance for every batch, and hence create the attention metadata builder instance for every batch, but we don't have some global information when we create the input for every batch. ⏎  ⏎ When upgrading to flashinfer 0.2, see https://github.com/vllm-project/vllm/pull/11194 , the attention metadata builder needs to access some global information such as sliding window, for the `plan` function before the flashinfer wrap …[truncated]

### L3-978b45f399  (L3, 2025-01-23, sha 978b45f39970, PR #12093)
TITLE: [Kernel] Flash Attention 3 Support (#12093)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake, L3.flashinfer.trtllm_gen
FILES: CMakeLists.txt (+20/-25); setup.py (+8/-4); vllm/attention/backends/flash_attn.py (+24/-3); vllm/envs.py (+12/-0); vllm/v1/attention/backends/flash_attn.py (+33/-11); vllm/v1/worker/gpu_model_runner.py (+21/-25); tests/kernels/test_cascade_flash_attn.py (+14/-10); tests/kernels/test_flash_attn.py (+18/-4)
LABELS: ready, ci/build
BODY: Most of the changes for FA3 are in vllm-flash-attn, but there was some changes required on vLLM side. Namely FA3 doesn't support `cu_seqlens_k` when using a paged kv cache, instead it uses `seqused_k` which is the kv seqlens. FA2 also supports `seqused_k` so we switch to using this for both FA3 and FA2 when dealing with a paged kv-cache in-order to maintain a common interface. ⏎  ⏎ This currently mainly improves V1 performance, some throughput number …[truncated]

### L3-e97f802b2d  (L3, 2025-01-23, sha e97f802b2d74, PR #11906)
TITLE: [FP8][Kernel] Dynamic kv cache scaling factors computation (#11906)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flashinfer.v0_backend, L3.triton.prefix_prefill, L3.rocm.custom_paged, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: csrc/attention/attention_kernels.cuh (+5/-5); csrc/attention/paged_attention_v1.cu (+10/-7); csrc/attention/paged_attention_v2.cu (+10/-7); csrc/cache_kernels.cu (+17/-13); csrc/cpu/attention.cpp (+6/-6); csrc/rocm/attention.cu (+10/-7); vllm/attention/backends/abstract.py (+8/-2); vllm/attention/backends/blocksparse_attn.py (+2/-0); vllm/attention/backends/flash_attn.py (+4/-1); vllm/attention/backends/flashinfer.py (+6/-4); (+50 more)
LABELS: documentation, rocm, ready
BODY: This PR deprecates loading kv cache scales from json in favor of adding the option to dynamically compute them based on the first real input to the attention layer. ⏎ Our tests showed that the dynamic range computed based on the first input to each layer is representative of the entire model, and the accuracy is comparable with scaling factors computed using Quark quantizer (such as in HF amd/*-FP8-KV models) ⏎  ⏎ Accuracy measured using the [P3L ](htt …[truncated]

### L3-ab5bbf5ae3  (L3, 2025-01-24, sha ab5bbf5ae32b, PR #12375)
TITLE: [Bugfix][Kernel] Fix CUDA 11.8 being broken by FA3 build (#12375)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1); setup.py (+4/-1); vllm/attention/backends/flash_attn.py (+11/-3); vllm/v1/attention/backends/flash_attn.py (+10/-1); tests/kernels/test_cascade_flash_attn.py (+6/-5); tests/kernels/test_flash_attn.py (+10/-11)
LABELS: documentation, ready, ci/build
DEEP_STUDY: deep-study correctness case vllm:ab5bbf5ae3: class=hardware_compiler_specific; symptom=compile_or_build_failure; introducing=unknown
BODY: Can build FA3 with CUDA version < 12.0 due to use of `__viaddmin_s32` and compute capability 9.0a, disable FA3 for 11.8 builds

### L3-3132a933b6  (L3, 2025-01-24, sha 3132a933b65d, PR #12405)
TITLE: [Bugfix][Kernel] FA3 Fix - RuntimeError: This flash attention build only supports pack_gqa (for build size reasons). (#12405)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1)
LABELS: ready, ci/build
BODY: Based off of https://github.com/vllm-project/vllm/pull/12375 that should be merged first ⏎  ⏎ We thought we could get away with only packed-gqa (is the default for varlen gqa) for build time and size reasons, but fails for true MHA, this turns back on the non packed-gqa kernels. ⏎  ⏎ Tested with `facebook/opt-125m`

### L3-f1fc0510df  (L3, 2025-01-25, sha f1fc0510dfbb, PR #12355)
TITLE: [Misc] Add FA2 support to ViT MHA layer (#12355)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+20/-5); tests/kernels/test_mha_attn.py (+126/-0)
LABELS: ready
DEEP_STUDY: deep-study: this PR was reverted by PR 12445 (confirmed_revert, reason=crash_or_hang)
BODY: - Add FA2 support to ViT MHA layer. ⏎ - I leave Qwen2-VL's ViT MHA layer consolidation to be done in a separate PR, because it has `head_size=80` and doesn't work with `vllm_flash_attn` (`head_size` should be a multiple of 32).

### L3-a5255270c3  (L3, 2025-01-26, sha a5255270c3ad, PR #12445)
TITLE: [Misc] Revert FA on ViT #12355 and #12435 (#12445)
SOURCES: path_core, symbol_pickaxe, corpus:confirmed-reverts
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+4/-37)
LABELS: ready
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 12355;12435 reason=crash_or_hang
BODY: Unfortunately FA3 support on ViT isn't at a reliable state to be used so revert the related PRs to always use xformers or SDPA. ⏎  ⏎ Related error logs when running llava-one-vision ⏎ ``` ⏎ ERROR 01-26 09:57:39 core.py:208]   File "/home/jovyan/vllm/vllm/model_executor/models/siglip.py", line 329, in forward ⏎ ERROR 01-26 09:57:39 core.py:208]     out = self.attn(query_states, key_states, value_states) ⏎ ERROR 01-26 09:57:39 core.py:208]   File "/usr/local/l …[truncated]

### L3-2a0309a646  (L3, 2025-01-26, sha 2a0309a646b1, PR #12435)
TITLE: [Misc][Bugfix] FA3 support to ViT MHA layer (#12435)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+22/-3)
LABELS: ready
DEEP_STUDY: deep-study: this PR was reverted by PR 12445 (confirmed_revert, reason=crash_or_hang)
BODY: This PR modifies the changes from #12355 to use `flash_attn_varlen_func` instead of deprecated `flash_attn_func`.

### L3-72f4880425  (L3, 2025-01-26, sha 72f4880425ed, PR #12450)
TITLE: [Bugfix/CI] Fix broken kernels/test_mha.py (#12450)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+8/-0); tests/kernels/test_mha_attn.py (+2/-2)
LABELS: ready
BODY: Fix for broken tests in `kernels/test_mha_attn.py` as https://github.com/vllm-project/vllm/commit/a5255270c3ad492b5def19fe38beb9b2df30e74f didn't revert the test changes from https://github.com/vllm-project/vllm/commit/f1fc0510dfbb11c98f41d02a44e092785c626314

### L3-68f11149d8  (L3, 2025-01-26, sha 68f11149d845, PR #12434)
TITLE: [Bugfix][Kernel] Fix perf regression caused by PR #12405 (#12434)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1)
LABELS: ready, ci/build
BODY: Fix perf regression caused by https://github.com/vllm-project/vllm/pull/12405 (resulted in falling back on FA2) ⏎  ⏎ Tested using: ⏎ ``` ⏎ VLLM_FLASH_ATTN_VERSION=3 python -m vllm.scripts serve meta-llama/Meta-Llama-3-8B ⏎ ```

### L3-372bf0890b  (L3, 2025-01-27, sha 372bf0890b19, PR #12464)
TITLE: [Bugfix] Fix missing seq_start_loc in xformers prefill metadata (#12464)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend
FILES: vllm/attention/backends/xformers.py (+3/-0)
LABELS: ready
BODY: Phi-3-small with blocksparse attention is not working with xformers attention prefill metadata because of missing `seq_start_loc`:  ⏎ [details omitted] ⏎  ⏎ - This PR fixes the missing `seq_start_loc` in xformers prefill metadata.

### L3-823ab79633  (L3, 2025-01-27, sha 823ab7963308, PR #12475)
TITLE: Update `pre-commit` hooks (#12475)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.triton.prefix_prefill, L3.triton.flash_attention_rocm, L3.rocm.custom_paged, L3.dispatch.selector
FILES: csrc/rocm/attention.cu (+3/-1); vllm/attention/ops/prefix_prefill.py (+14/-14); vllm/attention/ops/triton_flash_attention.py (+2/-2); vllm/attention/selector.py (+2/-2); .pre-commit-config.yaml (+5/-5); benchmarks/benchmark_serving.py (+2/-2); csrc/custom_all_reduce.cuh (+6/-2); csrc/moe/marlin_kernels/marlin_moe_kernel.h (+4/-4); csrc/quantization/gptq_marlin/gptq_marlin.cu (+8/-8); csrc/quantization/marlin/dense/marlin_cuda_kernel.cu (+2/-2); (+54 more)
LABELS: frontend, ready, ci/build
BODY: The linting packages haven't been updated in a while, so I ran `pre-commit autoupdate` and fixed everything that does not comply with the latest versions of: ⏎ - `yapf` ⏎ - `ruff` ⏎ - `codespell` ⏎ - `clang-format` ⏎ - `pymarkdown`

### L3-ddee88d0ff  (L3, 2025-01-27, sha ddee88d0ff27, PR #11277)
TITLE: [Neuron][Kernel] NKI-based flash-attention kernel with paged KV cache (#11277)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/nki_flash_attn.py (+669/-0); .buildkite/run-neuron-test.sh (+1/-1); tests/neuron/test_prefix_prefill.py (+456/-0)
LABELS: ready, ci/build
ISSUES: #11152 [RFC][Exploratory]: vLLM Neuron Backend with V1 Architecture
BODY: ### Summary ⏎  ⏎ FIX #11152  ⏎  ⏎ This PR introduce a NKI-based kernel that brings the support for chunked-prefill with flash-attention.

### L3-2bc3fbba0c  (L3, 2025-01-27, sha 2bc3fbba0cf5, PR #11194)
TITLE: [FlashInfer] Upgrade to 0.2.0 (#11194)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.v0_backend
FILES: Dockerfile (+21/-2); vllm/attention/backends/flashinfer.py (+162/-21); vllm/config.py (+6/-4); vllm/worker/worker_base.py (+13/-4); .buildkite/test-pipeline.yaml (+10/-1); tests/basic_correctness/test_basic_correctness.py (+3/-2); tests/compile/test_basic_correctness.py (+1/-1); tests/kernels/test_flashinfer.py (+37/-37); vllm/model_executor/model_loader/loader.py (+2/-2); vllm/model_executor/model_loader/tensorizer.py (+2/-1)
LABELS: documentation, frontend, ready, ci/build
BODY: This PR upgrades the [FlashInfer](https://github.com/flashinfer-ai/flashinfer) attention backend to v0.2.0.

### L3-dd66fd2b01  (L3, 2025-01-28, sha dd66fd2b01e1, PR #12494)
TITLE: [CI] fix pre-commit error (#12494)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/nki_flash_attn.py (+25/-12); vllm/spec_decode/spec_decode_worker.py (+4/-4)
LABELS: ready
BODY: fix pre-commit error

### L3-80fcc3ed1c  (L3, 2025-01-28, sha 80fcc3ed1c94, PR #12482)
TITLE: [Kernel] Pipe attn_logits_soft_cap through paged attention TPU kernels (#12482)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/pallas.py (+16/-26); .buildkite/run-tpu-test.sh (+0/-0)
LABELS: tpu, ready, ci/build
BODY: Pipe attn_logits_soft_cap through paged_attention, this will unblock some of our models' adoption. ⏎  ⏎ Note that the changed code currently doesn't have unit tests. I will add one later. ⏎  ⏎ This is the same as #12294. We created a new PR as I messed up the sign off and previous commit chain. ⏎  ⏎ Thanks,

### L3-9798b2fb00  (L3, 2025-01-30, sha 9798b2fb0052, PR #11868)
TITLE: [Kernel] Update `cutlass_scaled_mm` to support 2d group (blockwise) scaling (#11868)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+7/-2); benchmarks/cutlass_benchmarks/w8a8_benchmarks.py (+142/-148); csrc/core/math.hpp (+8/-1); csrc/cutlass_extensions/common.hpp (+17/-0); csrc/cutlass_extensions/gemm/collective/collective_builder.hpp (+123/-0); csrc/cutlass_extensions/gemm/collective/fp8_accumulation.hpp (+183/-0); csrc/cutlass_extensions/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp (+730/-0); csrc/cutlass_extensions/gemm/dispatch_policy.hpp (+39/-0); csrc/cutlass_extensions/vllm_collective_builder.cuh (+1/-1); csrc/quantization/cutlass_w8a8/c3x/cutlass_gemm_caller.cuh (+93/-0); (+15 more)
LABELS: ready, ci/build
BODY: Currently only supports scale_a block shapes of 1x128 and scale_b block shapes of 128x128 (for deepseek v3) ⏎  ⏎ Shout-out to @manishucsd and @soundOfDestiny for the kernel, kernel adapted from: https://github.com/soundOfDestiny/cutlass/tree/f8_blockwise_scaling_pr_branch ⏎  ⏎ This PR also splits up `scaled_mm_c3x.cu` to help parallelize the building of the kernels ⏎  ⏎ TODO: ⏎  ⏎ Benchmarking (Scroll horizontally to see `cutlass_fp8_fp8_fp16_scaled_mm_blockwise …[truncated]

### L3-cabaf4eff3  (L3, 2025-01-30, sha cabaf4eff3c7, PR #12528)
TITLE: [Attention] MLA decode optimizations (#12528)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flashinfer.trtllm_gen, L3.triton.decode_attention, L3.mla.triton_v0, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: csrc/cache_kernels.cu (+95/-0); csrc/torch_bindings.cpp (+9/-0); vllm/_custom_ops.py (+13/-0); vllm/attention/backends/abstract.py (+16/-0); vllm/attention/backends/mla/__init__.py (+0/-0); vllm/attention/backends/mla/utils.py (+365/-0); vllm/attention/backends/triton_mla.py (+749/-0); vllm/attention/backends/utils.py (+4/-0); vllm/attention/layer.py (+15/-4); vllm/attention/ops/triton_decode_attention.py (+667/-0); (+21 more)
LABELS: ready, ci/build
BODY: Implements MLA decode optimizations, i.e. computing MQA using latent vectors instead of MHA ⏎  ⏎ Shout-out to @simon-mo for the initial PR: https://github.com/vllm-project/vllm/pull/10927 ⏎ Shout-out to @tsu-bin for the handy reference: https://github.com/flashinfer-ai/flashinfer/pull/551 ⏎ Shout-out to [sglang](https://github.com/sgl-project/sglang) for the triton decode attention kernel

### L3-a1fc18c030  (L3, 2025-01-31, sha a1fc18c030e4, PR #12421)
TITLE: [ROCm][AMD][Model] llama 3.2 support upstreaming (#12421)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+291/-85); vllm/model_executor/models/mllama.py (+13/-3)
LABELS: rocm, ready
BODY: PR to propagate multimodal llama3.2 support into upstream for rocm arch

### L3-baeded2569  (L3, 2025-01-31, sha baeded25699f, PR #12601)
TITLE: [Attention] Deepseek v3 MLA support with FP8 compute (#12601)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.triton_v0
FILES: vllm/attention/backends/mla/utils.py (+184/-36); vllm/attention/backends/triton_mla.py (+7/-11); vllm/attention/layer.py (+2/-2); vllm/config.py (+33/-6); vllm/envs.py (+11/-1); vllm/model_executor/models/deepseek_v3.py (+152/-2); vllm/worker/cache_engine.py (+3/-1); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+52/-22); vllm/model_executor/layers/quantization/utils/quant_utils.py (+115/-1); vllm/model_executor/model_loader/loader.py (+21/-3)
LABELS: ready
BODY: Based off of: https://github.com/vllm-project/vllm/pull/12528 that needs to land first

### L3-4f4d427ac2  (L3, 2025-01-31, sha 4f4d427ac2ce, PR #12642)
TITLE: Disable chunked prefill and/or prefix caching when MLA is enabled  (#12642)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/config.py (+10/-0)
LABELS: ready
BODY: From @mgoin in https://github.com/vllm-project/vllm/pull/12638 ⏎  ⏎ I cannot push to that branch, therefore a new PR to unblock release.

### L3-e489ad7a21  (L3, 2025-02-02, sha e489ad7a210f, PR #12628)
TITLE: [Misc] Add SPDX-License-Identifier headers to python source files (#12628)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flashinfer.v0_backend, L3.triton.prefix_prefill, L3.triton.flash_attention_rocm, L3.triton.decode_attention, L3.rocm.rocm_flash_attn_v0, L3.mla.triton_v0, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: .buildkite/check-wheel-size.py (+2/-0); .buildkite/generate_index.py (+2/-0); .buildkite/lm-eval-harness/test_lm_eval_correctness.py (+1/-0); .buildkite/nightly-benchmarks/scripts/convert-results-json-to-markdown.py (+2/-0); .buildkite/nightly-benchmarks/scripts/download-tokenizer.py (+2/-0); .buildkite/nightly-benchmarks/scripts/generate-nightly-markdown.py (+2/-0); .buildkite/nightly-benchmarks/scripts/get-lmdeploy-modelname.py (+2/-0); .buildkite/nightly-benchmarks/scripts/summary-nightly-results.py (+2/-0); .pre-commit-config.yaml (+5/-1); benchmarks/backend_request_func.py (+2/-0); (+1002 more)
LABELS: documentation, structured-output, frontend, speculative-decoding, ci/build, v1
BODY: - **Add SPDX license headers to python source files** ⏎ - **Check for SPDX headers using pre-commit** ⏎  ⏎ commit 9d7ef44c3cfb72ca4c32e1c677d99259d10d4745 ⏎ Author: Russell Bryant <rbryant@redhat.com> ⏎ Date:   Fri Jan 31 14:18:24 2025 -0500 ⏎  ⏎     Add SPDX license headers to python source files ⏎      ⏎     This commit adds SPDX license headers to python source files as recommended to ⏎     the project by the Linux Foundation. These headers provide a concise way  …[truncated]

### L3-33e0602e59  (L3, 2025-02-03, sha 33e0602e59cf, PR #12694)
TITLE: [Misc] Fix improper placement of SPDX header in scripts (#12694)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.flash_attention_rocm
FILES: vllm/attention/ops/triton_flash_attention.py (+1/-2); cmake/hipify.py (+1/-2); tests/models/test_transformers.py (+1/-0); tools/check_spdx_header.py (+12/-5); tools/report_build_time_ninja.py (+1/-1); vllm/model_executor/models/transformers.py (+1/-0)
LABELS: ready, ci/build
BODY: When a file starts with '#!', the SPDX header needs to go after that. ⏎  ⏎ When looking for the SPDX header, find it on any line, not just the ⏎ first. ⏎  ⏎ Finally, don't add an extra newline when injecting the header into a ⏎ file.

### L3-6dd5e52823  (L3, 2025-02-03, sha 6dd5e52823cc, PR #12704)
TITLE: Squelch MLA warning for Compressed-Tensors Models (#12704)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/config.py (+4/-2)
LABELS: ready
BODY: ## Purpose ## ⏎ * Squelch MLA warning when loading compressed-tensors models ⏎  ⏎ ``` ⏎ WARNING 02-03 14:14:32 config.py:1007] compressed-tensors MLA support requires fp8 activations and weights in group 'group ⏎ _0', but got activations type 'None' and weights type 'int'. ⏎ WARNING 02-03 14:14:32 config.py:1007]  Full config: {'config_groups': {'group_0': {'input_activations': None, 'output_act ⏎ ivations': None, 'targets': ['Linear'], 'weights': {'actorder': …[truncated]

### L3-c36ac98d01  (L3, 2025-02-04, sha c36ac98d0118, PR #12662)
TITLE: [AMD][ROCm] Enable DeepSeek model on ROCm (#12662)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/attention/backends/mla/utils.py (+5/-1); tests/kernels/test_rocm_attention_selector.py (+31/-0); tests/worker/test_model_runner.py (+9/-0); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+10/-0); vllm/platforms/rocm.py (+3/-0)
LABELS: rocm, ready
BODY: Thanks for great work from vLLM community to enable DeepSeek model. ⏎ This PR is to use the Triton MLA backend on ROCm to enable DeepSeek model on AMD GPUs. ⏎  ⏎ Tests: ⏎  ⏎ - Tested on DeepSeek V2  (DeepSeek-Coder-V2-Lite-Instruct) model on MI300x and MI210. ⏎ - Tested on DeepSeek V3 dummy weights on MI300x. ⏎ - Added two unit tests. ⏎ -  ⏎  ⏎ Note: to use DeepSeek model, we should ensure to have the "trust_remote_code" flag on.

### L3-75e94309e8  (L3, 2025-02-04, sha 75e94309e8d8, PR #12676)
TITLE: [Perf] Mem align KV caches for CUDA devices (MLA perf improvement) (#12676)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flashinfer.trtllm_gen, L3.triton.decode_attention, L3.mla.triton_v0
FILES: csrc/cache_kernels.cu (+70/-12); csrc/torch_bindings.cpp (+4/-0); vllm/_custom_ops.py (+5/-0); vllm/attention/backends/triton_mla.py (+2/-3); vllm/attention/ops/triton_decode_attention.py (+8/-8); vllm/envs.py (+10/-0); vllm/utils.py (+10/-0); vllm/worker/cache_engine.py (+55/-11); csrc/cache.h (+3/-0); tests/kernels/test_cache.py (+262/-0)
LABELS: ready
BODY: Generally Nvidia hardware likes 256 byte alignment (reasons is foggy due to the blackbox nature of Nvidia hardware), but memory allocated via the CUDA Runtime ensure 256 byte alignment (see https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#a-sequential-but-misaligned-access-pattern). ⏎  ⏎ This PR aligns KV cache entries to start 256 byte boundaries, this mainly targets MLA since for "normal attention" with normal head dims (say 64 or 128) the …[truncated]

### L3-98fd089fc9  (L3, 2025-02-04, sha 98fd089fc974, PR #12729)
TITLE: [VLM] Add MLA with pure RoPE support for deepseek-vl2 models (#12729)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/mla/utils.py (+26/-4); vllm/model_executor/models/deepseek_v2.py (+2/-1); vllm/model_executor/models/deepseek_v3.py (+2/-1)
LABELS: ready
BODY: FIX https://github.com/vllm-project/vllm/pull/11578#issuecomment-2632920024 ⏎ - Deepseek-VL2 use pure rotary embedding, and current MLA implementation only support yarn rope

### L3-fcf2e3d7fc  (L3, 2025-02-04, sha fcf2e3d7fcc9, PR #12750)
TITLE: [Bugfix] Fix OpenVINO model runner (#12750)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/openvino.py (+4/-0); vllm/model_executor/model_loader/openvino.py (+5/-6); vllm/worker/openvino_model_runner.py (+3/-6)
LABELS: ready
ISSUES: #12350 [Bug]: Cannot serve Qwen2.5 in OpenVINO | #12687 [Bug]: Cannot run vLLM server with OpenVINO backend
BODY: Fixes #12350 ⏎ Fixes #12687

### L3-64862d106e  (L3, 2025-02-05, sha 64862d106efa, PR #12713)
TITLE: [ROCM][AMD][TRITON] Halving warps number for fw_prefill to reduce spilling (#12713)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/prefix_prefill.py (+1/-1)
LABELS: rocm, ready
BODY: in general this gives 2-3x perf gain over default and about 1.5x slower than H100 ⏎  ⏎ for model run llama3.1 70B fp16 ⏎  ⏎ HIP_VISIBLE_DEVICES=1 python /root/workspace/vllm/benchmarks/profiling/benchmark_throughput.py --model /data/models/Llama-3.1-70B-Instruct --input-len 1024 --output-len 1024 --num-prompts 256 --max-model-len 32768 --disable-log-stats --enable-chunked-prefill ⏎  ⏎ 8 warps ⏎ Throughput: 0.65 requests/s, 1327.15 total tokens/s, 663.57 output …[truncated]

### L3-4c3aac51e1  (L3, 2025-02-05, sha 4c3aac51e142, PR #)
TITLE: Merging PR #12536
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L3-af8486de49  (L3, 2025-02-05, sha af8486de49a2, PR #)
TITLE: [Hardware][Intel-Gaudi] Enable FusedSDPA support for Intel Gaudi (HPU)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/hpu_attn.py
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L3-85ac82d228  (L3, 2025-02-06, sha 85ac82d228ef, PR #12777)
TITLE: [Kernel] Make rotary_embedding ops more flexible with input shape (#12777)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/mla/utils.py (+3/-22); csrc/pos_encoding_kernels.cu (+89/-14); tests/kernels/test_pos_encoding.py (+22/-9); vllm/model_executor/models/deepseek_v2.py (+1/-12)
LABELS: ready
BODY: FIX https://github.com/vllm-project/vllm/pull/12729#discussion_r1942203266

### L3-8108ac841d  (L3, 2025-02-06, sha 8108ac841d66, PR #12828)
TITLE: [Bugfix] Fix unsupported FA version check for Turing GPU (#12828)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/utils.py (+1/-1)
LABELS: ready
BODY: #12807 broke xformers fallback because `AssertionError` from `assert is_fa_version_supported(fa_version)` is not included in the exception: ⏎ https://github.com/vllm-project/vllm/blob/c786e757fae4519256e4ef88a7d4f56c3339d14d/vllm/attention/backends/utils.py#L611-L616 ⏎ ``` ⏎ ERROR 02-06 12:53:27 utils.py:608] Cannot use FA version 2 is not supported due to FA3 is only supported on devices with compute capability >= 8 excluding 8.6 and 8.9 ⏎ Traceback (mo …[truncated]

### L3-c786e757fa  (L3, 2025-02-06, sha c786e757fae4, PR #12807)
TITLE: [Attention] Use FA3 for MLA on Hopper (#12807)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases(introducing)
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend
FILES: vllm/attention/backends/flash_attn.py (+11/-33); vllm/attention/backends/mla/utils.py (+2/-0); vllm/attention/backends/utils.py (+34/-0); vllm/v1/attention/backends/flash_attn.py (+4/-26)
LABELS: ready, v1
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:97a3d6d995 (fix PR 13310)
BODY: Make sure we are using FA3 for MLA prefill on Hopper

### L3-aff404571b  (L3, 2025-02-06, sha aff404571b0d, PR #10909)
TITLE: Add Bamba Model (#10909)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/placeholder_attn.py (+62/-78); tests/kernels/test_mamba_mixer2.py (+125/-0); tests/kernels/test_mamba_ssm_ssd.py (+304/-0); tests/models/decoder_only/language/test_hybrid.py (+23/-12); tests/models/registry.py (+1/-0); vllm/model_executor/layers/mamba/mamba_mixer2.py (+534/-0); vllm/model_executor/layers/mamba/ops/mamba_ssm.py (+1/-1); vllm/model_executor/layers/mamba/ops/ssd_bmm.py (+261/-0); vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py (+615/-0); vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py (+750/-0); (+7 more)
LABELS: ready, ci/build
BODY: This is the companion PR to an [huggingface PR](https://github.com/huggingface/transformers/pull/34982) for adding `Bamba`, which is a hybrid [mamba2](https://github.com/state-spaces/mamba) architecture with SwiGLU. The checkpoints are jointly trained by IBM, Princeton, and UIUC. ⏎  ⏎ In this PR we have: ⏎  ⏎ Currently only `FlashAttention` backend is supported, as we check fields like `context_lens_tensor`.  Have not yet investigated other backends. ⏎  ⏎ We …[truncated]

### L3-ef533d25fb  (L3, 2025-02-06, sha ef533d25fba4, PR #12848)
TITLE: [Bugfix] FA2 illegal memory access (#12848)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1)
LABELS: ready, ci/build
ISSUES: #10389 [Bug]: v0.6.4.post1 crashed：Error in model execution: CUDA error: an illegal memory access was encountered | #11340 [Bug]: CUDA illegal memory access in flash attention only for specific values of --max-num-seqs (with AWQ model )
BODY: PR in the flash-attention repo: https://github.com/vllm-project/flash-attention/pull/42 ⏎  ⏎ should hopefully resolve: ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/10389 ⏎ FIX https://github.com/vllm-project/vllm/issues/11340

### L3-4ea48fb35c  (L3, 2025-02-08, sha 4ea48fb35cf6, PR #12943)
TITLE: [V1][Minor] Move cascade attn logic outside _prepare_inputs (#12943)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_model_runner.py (+89/-61)
LABELS: ready, v1
BODY: Small code re-organization to make the `_prepare_inputs` shorter.

### L3-fe743b798d  (L3, 2025-02-09, sha fe743b798dfa, PR #12959)
TITLE: [bugfix] fix early import of flash attention (#12959)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend
FILES: vllm/attention/backends/flash_attn.py (+7/-6); vllm/attention/backends/mla/utils.py (+3/-2); vllm/attention/backends/utils.py (+6/-8); vllm/v1/attention/backends/flash_attn.py (+4/-3)
LABELS: ready, v1
BODY: the rlhf test example has been broken by https://github.com/vllm-project/vllm/pull/12807 , because it imports vllm flash attention too early.

### L3-44607e07d3  (L3, 2025-02-10, sha 44607e07d3ba, PR #12975)
TITLE: Check if selected backend is None in get_attn_backend_cls() (#12975)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: vllm/platforms/cpu.py (+1/-1)
LABELS: ready
BODY: This avoids the misleading message "cannot use None backend" as seen below: ⏎  ⏎ ``` ⏎ INFO 02-09 03:13:01 __init__.py:190] Automatically detected platform cpu. ⏎ INFO 02-09 03:13:05 cpu.py:39] Cannot use None backend on CPU. ⏎ INFO 02-09 03:13:05 cpu.py:40] Using Torch SDPA backend. ⏎ ```

### L3-974dfd4971  (L3, 2025-02-11, sha 974dfd497149, PR #12830)
TITLE: [Model] IBM/NASA Prithvi Geospatial model  (#12830)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/placeholder_attn.py (+8/-3); examples/offline_inference/prithvi_geospatial_mae.py (+530/-0); tests/models/registry.py (+4/-0); vllm/inputs/preprocess.py (+17/-7); vllm/model_executor/models/prithvi_geospatial_mae.py (+238/-0); vllm/model_executor/models/registry.py (+4/-0); vllm/worker/pooling_model_runner.py (+10/-1)
LABELS: ready
BODY: This pull request proposes a first integration of Prithvi, a non-language model to vLLM. This is related to the discussion we had in a previous RFC https://github.com/vllm-project/vllm/issues/11065. ⏎  ⏎ Prithvi is a non-autoregressive model ingesting torch tensors in input and generating on (or a list of) torch tensors. As suggested in the discussion linked to the RFC, this can be achieved in vLLM by integrating the model as a Pooling model that all …[truncated]

### L3-e92694b6fe  (L3, 2025-02-11, sha e92694b6fe26, PR #12921)
TITLE: [Neuron][Kernel] Support Longer Sequences in NKI-based Flash PagedAttention and Improve Efficiency (#12921)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/nki_flash_attn.py (+87/-129); tests/neuron/test_prefix_prefill.py (+67/-51)
BODY: # Summary ⏎  ⏎ This PR is a follow-up of #11277 . It improves code quality and efficiency, and enables kernel to process larger inputs. ⏎  ⏎ Following things are done in this PR: ⏎ - fix 2 tiling issues triggered when `seqlen_q` (i.e. chunk size in chunked-prefill) is larger than 128 and 512 ⏎ - get rid of the limit of how many KV cache block each tile can access, currently `tile_size / block_size <= 128` ⏎ - unit tests with larger inputs (e.g. `seqlen_q > 128 …[truncated]

### L3-f1042e86f0  (L3, 2025-02-12, sha f1042e86f05c, PR #12923)
TITLE: [Misc] AMD Build Improvements (#12923)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.custom_paged
FILES: csrc/rocm/attention.cu (+1/-1); csrc/moe/moe_align_sum_kernels.cu (+1/-1); vllm/model_executor/models/registry.py (+11/-4); vllm/transformers_utils/configs/__init__.py (+1/-1)
LABELS: rocm, ready
BODY: Some changes to support Meta internal AMD build. Internally we can't directly use sys.executable, so this PR makes it's possible to modify the subprocess args.

### L3-f0b2da72a8  (L3, 2025-02-13, sha f0b2da72a84a, PR #13181)
TITLE: Expand MLA to support most types of quantization (#13181)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/mla/utils.py (+26/-45); vllm/config.py (+1/-31); vllm/model_executor/model_loader/loader.py (+34/-56)
LABELS: ready
BODY: For MLA, aside from the FP8 case, we need to have access to the unquantized weights for the decode kernel.  ⏎ This PR implements a general dequantization for quantization methods in vLLM by creating the identity matrix as input to `layer.quant_method.apply()` in order to get the transposed, unquantized weights as the output. ⏎  ⏎ Thanks to @LucasWilkinson and @tlrmchlsmth for iterating and coming up with this idea. ⏎  ⏎ Manual testing: ⏎  ⏎ #### DeepSeekV2 Lit …[truncated]

### L3-ba59b78a9c  (L3, 2025-02-13, sha ba59b78a9c5a, PR #12790)
TITLE: [ROCm][V1] Add intial ROCm support to V1 (#12790)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.triton.prefix_prefill, L3.rocm.v1_rocm_attn, L3.platform.rocm_selection
FILES: vllm/attention/ops/prefix_prefill.py (+4/-2); vllm/v1/attention/backends/flash_attn.py (+4/-1); vllm/v1/attention/backends/rocm_attn.py (+182/-0); requirements-rocm-build.txt (+16/-0); vllm/platforms/rocm.py (+30/-15)
LABELS: rocm, ready, ci/build, v1
BODY: This Pr is adds initial support for V1 on AMD systems. It uses the `vllm/attention/ops/prefix_prefill.py` kernel instead of flash-attn.  ⏎  ⏎ Current install instructions if you want to try it out. You should start with a new virtual environment. All of the below commands assume you are in the vllm source directory ⏎ `pip install -r requirements-rocm-build.txt` ⏎ `python setup.py develop` ⏎  ⏎ And here's an example command to run ⏎ `VLLM_USE_V1=1 python exampl …[truncated]

### L3-45f90bcbba  (L3, 2025-02-14, sha 45f90bcbba76, PR #13049)
TITLE: [WIP] TPU V1 Support Refactored (#13049)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+353/-0); tests/entrypoints/llm/test_accuracy.py (+15/-5); tests/entrypoints/openai/correctness/test_lmeval.py (+11/-4); vllm/platforms/interface.py (+1/-0); vllm/platforms/tpu.py (+38/-16); vllm/v1/worker/block_table.py (+8/-0); vllm/v1/worker/tpu_model_runner.py (+1109/-0); vllm/v1/worker/tpu_worker.py (+203/-0)
LABELS: ready, ci/build, v1
BODY: This PR is a rebase and modification of @robertgshaw2-redhat original PR for TPU support in vLLM V1 from 2 months ago https://github.com/vllm-project/vllm/pull/10241 ⏎  ⏎ Currently, TPU attention kernel has no support for mixing prefills and decodes in the same scheduler iteration. As a result, this PR separates the requests to (1) prefills and (2) decodes, and executes each one of them separately. Google guys are working on a new TPU attention kerne …[truncated]

### L3-97a3d6d995  (L3, 2025-02-14, sha 97a3d6d99522, PR #13310)
TITLE: [Bugfix] Massage MLA's usage of flash attn for RoCM (#13310)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/mla/utils.py (+11/-2)
LABELS: bug, rocm, ready
DEEP_STUDY: deep-study correctness case vllm:97a3d6d995: class=hardware_compiler_specific; symptom=crash_or_exception; introducing=#12807
BODY: This PR massages some `vllm_flash_attn` vs `flash_attn` interface differences that appeared between a couple of PRs: ⏎  ⏎ https://github.com/vllm-project/vllm/pull/12662 added a this fallback to `flash_attn` in order to support MLA on RoCM: ⏎ https://github.com/vllm-project/vllm/blob/5e5c8e091eacc16672a0a8265eb5cb0ece85d24b/vllm/attention/backends/mla/utils.py#L33-L36 ⏎  ⏎ Subsequently https://github.com/vllm-project/vllm/pull/12807 updated to use the `vll …[truncated]

### L3-54ed913f34  (L3, 2025-02-15, sha 54ed913f3437, PR #13323)
TITLE: [ci/build] update flashinfer (#13323)
SOURCES: subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile (+5/-2)
LABELS: ready, ci/build
BODY: we have work together with flashinfer to make the wheel python version agnostic, see https://github.com/flashinfer-ai/flashinfer/pull/823 . And they have released a new version with better MLA kernel support, we can use their official wheel right now.

### L3-124776ebd5  (L3, 2025-02-16, sha 124776ebd5db, PR #13352)
TITLE: [ci] skip failed tests for flashinfer (#13352)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_flashinfer.py (+2/-0)
BODY: as observed in https://buildkite.com/vllm/ci/builds/13529#01950982-09cb-4aef-8f57-c8228d304aad , new flashinfer version fails the fp8 accuracy test. before it is fixed, let's skip the test for now. ⏎  ⏎ NOTE: we still need flashinfer 0.2.1.post1 for MLA integration. so i'm not reverting https://github.com/vllm-project/vllm/pull/13323

### L3-d0a7a2769d  (L3, 2025-02-18, sha d0a7a2769d92, PR #12139)
TITLE: [Hardware][Gaudi][Feature] Support Contiguous Cache Fetch  (#12139)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/attention/backends/hpu_attn.py (+1/-5); vllm/attention/ops/hpu_paged_attn.py (+1/-0); vllm/envs.py (+8/-0); vllm/worker/hpu_model_runner.py (+71/-43)
LABELS: ready
BODY: Contiguous cache fetching to avoid using costly gather operation on Gaudi3. Requires changes in vllm-hpu-extension (https://github.com/HabanaAI/vllm-hpu-extension/pull/17) to work. ⏎  ⏎ Introduces redundant calculations in decoding phase. Feature improves the performance of all tested workloads over the entire benchmark (5-12%) on Gaudi3. [commit](https://github.com/zhouyu5/vllm-fork/commit/25dcf18e1d50a218643adf3bd6384351d9da005d) further improves t …[truncated]

### L3-0023cd2b9d  (L3, 2025-02-19, sha 0023cd2b9dfe, PR #13560)
TITLE: [ROCm] MI300A compile targets deprecation (#13560)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.rocm.custom_paged, L3.rocm.rocm_flash_attn_v0
FILES: csrc/rocm/attention.cu (+1/-2); vllm/attention/backends/rocm_flash_attn.py (+1/-2); CMakeLists.txt (+1/-1); csrc/quantization/fp8/amd/hip_float8_impl.h (+1/-2)
LABELS: rocm, ready, ci/build
BODY: Removing gfx940 and gfx941 targets. These have been deprecated in favor of gfx942 for MI300X

### L3-497bc83124  (L3, 2025-02-19, sha 497bc8312412, PR #13566)
TITLE: [CI/Build] Use uv in the Dockerfile (#13566)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile (+19/-13)
LABELS: ready, ci/build
BODY: Improvement from [33min](https://buildkite.com/vllm/fastcheck/builds/14507#01951b93-f157-4b14-bdab-af7e97cea943) -> [28.5min](https://buildkite.com/vllm/fastcheck/builds/14593#0195200d-8891-46d2-afe8-1c65fe086657)

### L3-71face8540  (L3, 2025-02-20, sha 71face854004, PR #13620)
TITLE: [Bugfix] Fix max_num_batched_tokens for MLA (#13620)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/config.py (+14/-6)
LABELS: bug, ready
BODY: Without this change we are stuck with a very short allowed prompt length by default with deepseek models ⏎  ⏎ Before this PR: ⏎ ``` ⏎ vllm serve /home/vllm-dev/DeepSeek-R1 --trust-remote-code --tensor-parallel-size 8 ⏎ ... ⏎ WARNING 02-20 17:08:38 scheduler.py:949] Input prompt (100005 tokens) is too long and exceeds limit of 2048 ⏎ ``` ⏎  ⏎ You can get past this manually by setting max_num_batched_tokens, but we should take care of this automatically ⏎ ``` ⏎ vllm ser …[truncated]

### L3-33170081f1  (L3, 2025-02-20, sha 33170081f13b, PR #13245)
TITLE: [Neuron][Kernel] Vectorize KV cache load in FlashPagedAttention to maximize DMA bandwidth (#13245)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/nki_flash_attn.py (+428/-199); tests/neuron/test_block_table.py (+153/-0); tests/neuron/test_prefix_prefill.py (+183/-149)
BODY: Previous version of NKI flash attention kernel did not vectorize KV cache loading to fully utilize HBM bandwidth. As a result, the kernel is bottlenecked by fetching paged KV cache from HBM. ⏎  ⏎ We apply vectorization in this PR to fully saturate DMA bandwidth. ⏎  ⏎ @liangfu

### L3-288cc6c234  (L3, 2025-02-21, sha 288cc6c234d0, PR #12639)
TITLE: [Attention] MLA with chunked prefill (#12639)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flash_attn.v1_backend, L3.merge.triton_lse, L3.mla.triton_v0
FILES: csrc/cache_kernels.cu (+159/-0); csrc/torch_bindings.cpp (+6/-0); vllm/_custom_ops.py (+10/-0); vllm/attention/backends/mla/common.py (+1503/-0); vllm/attention/backends/mla/utils.py (+0/-515); vllm/attention/backends/triton_mla.py (+14/-650); vllm/attention/ops/triton_merge_attn_states.py (+84/-0); vllm/config.py (+0/-13); vllm/engine/arg_utils.py (+3/-4); vllm/utils.py (+4/-0); (+8 more)
LABELS: ready, v1
BODY: Need to do more benchmarking to see if this makes sense to be on by default in V0, but lays the groundwork for a V1 implementation. (https://github.com/vllm-project/vllm/pull/13111 may help performance) ⏎  ⏎ ``` ⏎ lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384,enable_chunked_prefill=False --task gsm8k --num_fewshot=5 - …[truncated]

### L3-68d630a0c7  (L3, 2025-02-21, sha 68d630a0c727, PR #13650)
TITLE: [ROCM] fix native attention function call (#13650)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+0/-1)
LABELS: ready
ISSUES: #13648 [Bug]: [ROCM] _sdpa_attention() takes from 8 to 9 positional arguments but 10 were given
BODY: fix native attention function call for navi cards ⏎  ⏎ FIX #13648

### L3-558db8083c  (L3, 2025-02-22, sha 558db8083cfd, PR #13095)
TITLE: [V1][Kernel] Refactor the prefix_prefill kernel so that the caller no longer has to pass in the context lengths (#13095)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.triton.prefix_prefill, L3.rocm.rocm_flash_attn_v0, L3.rocm.v1_rocm_attn
FILES: vllm/attention/backends/rocm_flash_attn.py (+0/-1); vllm/attention/backends/xformers.py (+0/-1); vllm/attention/ops/paged_attn.py (+1/-3); vllm/attention/ops/prefix_prefill.py (+9/-8); vllm/v1/attention/backends/rocm_attn.py (+0/-12); tests/kernels/test_prefix_prefill.py (+2/-6)
LABELS: ready, v1
BODY: This patch changes the prefix_prefill kernel so that it will calculate the context length using the query length and the sequence length, both of which are already passed in. This makes the kernel a bit more usable on V1 where we don't keep track of the context lengths tensor in the attention meta data.

### L3-322d2a27d6  (L3, 2025-02-22, sha 322d2a27d66e, PR #13706)
TITLE: [BugFix] Minor: logger import in attention backend (#13706)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/utils.py (+2/-2)
BODY: `vllm/attention/backends/utils.py` is not using the logger from `vllm.logger.init_logger`

### L3-444b0f0f62  (L3, 2025-02-24, sha 444b0f0f6298, PR #12513)
TITLE: [Misc][Docs] Raise error when flashinfer is not installed and `VLLM_ATTENTION_BACKEND` is set (#12513)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: vllm/config.py (+9/-0); docs/source/getting_started/quickstart.md (+10/-0)
LABELS: documentation, ready
BODY: FlashAttn is currently not resolved when installing through a package manager, as it still resorting to manual installation (see https://github.com/vllm-project/vllm/blob/main/Dockerfile#L210). ⏎ If the user manually sets the backend to `flashinfer` and `flashinfer` is not installed in the system, the resulting error message is not super informative: ⏎ ``` ⏎ Traceback (most recent call last): ⏎   File "/opt/dlami/nvme/vllm/vllm/worker/model_runner_base.p …[truncated]

### L3-cdc1fa12eb  (L3, 2025-02-24, sha cdc1fa12eb1b, PR #13555)
TITLE: Remove unused kwargs from model definitions (#13555)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+7/-12); docs/source/contributing/model/basic.md (+0/-2); docs/source/contributing/model/multimodal.md (+0/-2); tests/kernels/test_encoder_decoder_attn.py (+3/-11); vllm/model_executor/layers/mamba/mamba_mixer.py (+3/-2); vllm/model_executor/layers/mamba/mamba_mixer2.py (+2/-2); vllm/model_executor/models/adapters.py (+1/-5); vllm/model_executor/models/arctic.py (+5/-19); vllm/model_executor/models/aria.py (+0/-5); vllm/model_executor/models/baichuan.py (+5/-19); (+94 more)
LABELS: documentation, speculative-decoding, ready, v1
BODY: Follow up for #11967 which removes `kv_cache` and `attn_metadata` from all model definitions. ⏎  ⏎ Summary of changes: ⏎ - Remove `kv_cache` and `attn_metadata` from `Attention.forward()` args ⏎ - Use forward context instead of `attn_metadata` arg in `MambaMixer` and `MambaMixer2` ⏎ - Remove `kv_caches`, `kv_cache` and `attn_metadata` from `forward()` args of all model modules ⏎ - Remove `kv_caches` and `attn_metadata` from new model docs ⏎ - Leave `kv_caches` …[truncated]

### L3-75e9d49796  (L3, 2025-02-25, sha 75e9d4979658, PR #13468)
TITLE: [Bugfix] Initialize attention bias on the same device as Query/Key/Value (#13468)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend
FILES: vllm/attention/backends/xformers.py (+6/-4)
LABELS: ready
BODY: The attention bias in vLLM's xformers backend is currently initialized on the default device, rather than the device of the Q/K/V tensors: ⏎  ⏎ https://github.com/vllm-project/vllm/blob/b53d79983c273b2775456d99c0e0890aea073512/vllm/attention/backends/xformers.py#L676-L677 ⏎  ⏎ And here is how xformers decide which device to use: ⏎  ⏎ <https://github.com/facebookresearch/xformers/blob/8d91ce05a2f6a5ae059593922a631b9ff325b134/xformers/ops/fmha/attn_bias.py#L74 …[truncated]

### L3-ab1091d5f2  (L3, 2025-02-25, sha ab1091d5f2fc, PR #13733)
TITLE: [Misc][Attention][Quantization] init property earlier (#13733)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+5/-4)
LABELS: ready
BODY: When loading weights using an custom quantization method, the `num_heads`, `head_size`, `num_kv_heads` may be used. This pr move these properties before quantization initialization so that they can be used by the quantization method directly.

### L3-18e505930d  (L3, 2025-02-25, sha 18e505930d78, PR #13725)
TITLE: [Bugfix] Support MLA for CompressedTensorsWNA16 (#13725)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.triton_v0
FILES: vllm/attention/backends/mla/common.py (+7/-7)
LABELS: bug, ready
BODY: Without this change, models that use CompressedTensorsWNA16  will fail with MLA enabled due to their usage of `weight_packed`  ⏎  ⏎ Reference: https://github.com/vllm-project/vllm/blob/db986c19ea35d7f3522a45d5205bf5d3ffab14e4/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py#L137 ⏎  ⏎ Error: ⏎ ``` ⏎ [rank0]:   File "/home/mgoin/code/vllm/vllm/attention/backends/mla/common.py", line 1155, in process_weights_after_l …[truncated]

### L3-51010a1807  (L3, 2025-02-25, sha 51010a1807e1, PR #13771)
TITLE: [Misc] set single whitespace between log sentences (#13771)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.mla.triton_v0, L3.platform.cuda_selection
FILES: vllm/attention/backends/flashinfer.py (+1/-1); vllm/attention/backends/mla/common.py (+1/-1); vllm/attention/backends/rocm_flash_attn.py (+2/-2); vllm/config.py (+6/-6); vllm/distributed/device_communicators/pynccl_wrapper.py (+2/-2); vllm/distributed/kv_transfer/kv_pipe/mooncake_pipe.py (+1/-1); vllm/entrypoints/chat_utils.py (+1/-1); vllm/entrypoints/llm.py (+1/-1); vllm/entrypoints/openai/api_server.py (+1/-1); vllm/executor/ray_distributed_executor.py (+1/-1); (+26 more)
LABELS: frontend, speculative-decoding, ready, v1
BODY: Set single whitespace between log / error sentences. ⏎  ⏎ I just grepped `logger.*\(` and `Error(` and filtered by counting whitespaces between lines, so the list in this commit is not inclusive.

### L3-094b7d9496  (L3, 2025-02-25, sha 094b7d9496cc, PR #13797)
TITLE: [Kernel][Build/CI] Bump CUTLASS to 3.8 and add initializers for cutlass epilogues (#13797)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+4/-4); csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c2x.hpp (+14/-12); csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp (+15/-13)
LABELS: ready, ci/build
BODY: The original code was added in https://github.com/vllm-project/vllm/pull/9855. I am seeing errors like ⏎ ``` ⏎ cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp:149:194: error: missing initializer for member ‘cutlass::epilogue::fusion::detail::Sm90VisitorImplBase<cutlass::epilogue::fusion::Sm90RowOrScalarBroadcast<0, cute::tuple<cute::C<128>, cute::C<128>, cute::C<128> >, float, cute::tuple<cute::C<0>, cute::C<1>, cute::C<0> >, 4>, cutlass::epi …[truncated]

### L3-1d35662e6d  (L3, 2025-02-26, sha 1d35662e6dc1, PR #13844)
TITLE: [ROCm] Disable chunked prefill/prefix caching when running MLA on non-cuda platforms (#13844)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.mla.triton_v0
FILES: vllm/attention/backends/mla/common.py (+30/-12); vllm/config.py (+14/-0)
LABELS: rocm, ready
BODY: `flash_attn_varlen_func` in upstream `flash-attn` does not support the `return_softmax_lse` argument. This PR works around that issue by explicitly disabling chunked prefill and prefix caching on non-cuda platforms. ⏎  ⏎ Here are the results from running llmeval with `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` on an AMD system: ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|----- …[truncated]

### L3-0ecdd98031  (L3, 2025-02-26, sha 0ecdd98031f9, PR #13887)
TITLE: Add comments on accessing `kv_cache` and `attn_metadata` (#13887)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+13/-0)
BODY: As discussed in https://vllm-dev.slack.com/archives/C07R5Q1Q2BB/p1740557156558859

### L3-ca377cf1b9  (L3, 2025-02-26, sha ca377cf1b993, PR #12098)
TITLE: Use CUDA 12.4 as default for release and nightly wheels (#12098)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: setup.py (+3/-4); .buildkite/release-pipeline.yaml (+12/-1); .buildkite/upload-wheels.sh (+8/-2); docs/source/getting_started/installation/gpu/cuda.inc.md (+2/-2)
LABELS: documentation, ready, ci/build
BODY: I think it is time to match pytorch's default cuda version. We should still keep wheels built with 12.1 around.

### L3-378b3ef6f8  (L3, 2025-02-26, sha 378b3ef6f834, PR #13922)
TITLE: [ROCm][V1] Update reshape_and_cache to properly work with CUDA graph padding (#13922)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+1/-1)
LABELS: ready
ISSUES: #13418 [Bug][v1][rocm] cuda graph gets stuck in case padding is used to meet a captured input size
BODY: This patch updates reshape_and_cache to account for the case where slot_mapping.size(0) is not equal to key.size(0). This occurs when we use CUDA graph padding on V1. ⏎  ⏎ Fixes: https://github.com/vllm-project/vllm/issues/13418

### L3-f95903909f  (L3, 2025-02-27, sha f95903909f07, PR #13747)
TITLE: [Kernel] FlashMLA integration (#13747)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, corpus:production-kernel-provenance
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake, L3.flash_attn.fork_build, L3.mla.triton_v0, L3.mla.flashmla_v0_adapter, L3.mla.flashmla_build, L3.platform.cuda_selection
FILES: CMakeLists.txt (+4/-73); cmake/external_projects/flashmla.cmake (+66/-0); cmake/external_projects/vllm_flash_attn.cmake (+67/-0); setup.py (+6/-0); vllm/_custom_ops.py (+64/-0); vllm/attention/backends/flashmla.py (+239/-0); vllm/attention/backends/mla/common.py (+15/-13); vllm/attention/ops/flashmla.py (+115/-0); vllm/platforms/cuda.py (+24/-0); vllm/platforms/interface.py (+1/-0); (+1 more)
LABELS: ready, ci/build
ISSUES: #13735 [Feature]: Support for FlashMLA
BODY: Integrate: https://github.com/deepseek-ai/FlashMLA ⏎  ⏎ currently requires ⏎  ⏎ ``` ⏎ export VLLM_ATTENTION_BACKEND=FLASHMLA ⏎ ``` ⏎ and ⏎ ``` ⏎ block_size=64 ⏎ ``` ⏎  ⏎ TODO: ⏎  ⏎ - ~cuda-graphs are broken~ ⏎ - future PR: enforce block_size 64 gracefully  ⏎  ⏎ Closes #13735

### L3-58d1b2aa77  (L3, 2025-02-27, sha 58d1b2aa772d, PR #13789)
TITLE: [Attention] MLA support for V1 (#13789)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.mla.common_v1, L3.platform.cuda_selection
FILES: vllm/attention/layer.py (+24/-11); vllm/model_executor/models/deepseek_v2.py (+11/-2); vllm/platforms/cuda.py (+7/-2); vllm/platforms/interface.py (+1/-0); vllm/v1/attention/backends/flash_attn.py (+67/-2); vllm/v1/attention/backends/mla/__init__.py (+0/-0); vllm/v1/attention/backends/mla/common.py (+1022/-0); vllm/v1/attention/backends/triton_mla.py (+110/-0); vllm/v1/worker/gpu_input_batch.py (+63/-1); vllm/v1/worker/gpu_model_runner.py (+35/-41)
LABELS: ready, v1
BODY: This PR is co-authored with Lucas Wilkinson. ⏎  ⏎ ``` ⏎ VLLM_USE_V1="1" lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384,enforce_eager=True --task gsm8k --num_fewshot=5 --limit 100 ⏎ ... ⏎ vllm (pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,ma …[truncated]

### L3-8294773e48  (L3, 2025-02-27, sha 8294773e48a2, PR #13718)
TITLE: [core] Perf improvement for DSv3 on AMD GPUs (#13718)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.triton.decode_attention, L3.mla.triton_v0
FILES: vllm/attention/backends/mla/common.py (+72/-20); vllm/attention/ops/triton_decode_attention.py (+10/-5); vllm/model_executor/layers/fused_moe/configs/E=256,N=256,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json (+128/-0)
LABELS: rocm, ready
BODY: 1. Add GPUs tweaks for AMD; 2. Allow triton-fa for prefill stage in MLA path; 3. Add MoE config for MI300X

### L3-2e94b9cfbb  (L3, 2025-02-27, sha 2e94b9cfbb41, PR #13867)
TITLE: [Attention] Flash MLA for V1 (#13867)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+22/-13); vllm/platforms/interface.py (+2/-3); vllm/v1/attention/backends/mla/common.py (+7/-4); vllm/v1/attention/backends/mla/flashmla.py (+139/-0); vllm/v1/attention/backends/mla/triton_mla.py (+0/-0)
LABELS: ready, ci/build, v1
BODY: use via: `VLLM_ATTENTION_BACKEND=FLASHMLA VLLM_USE_V1=1` ⏎  ⏎ Results: ⏎  ⏎ https://docs.google.com/spreadsheets/d/1toxQVaA7UPhmY57kv08Wdq0xtIcU8oBbqb9RE3Wy2_E/edit?usp=sharing

### L3-c3b6559a10  (L3, 2025-02-28, sha c3b6559a1019, PR #13379)
TITLE: [V1][TPU] Integrate the new ragged paged attention kernel with vLLM v1 on TPU (#13379)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: requirements-tpu.txt (+5/-6); vllm/v1/attention/backends/pallas.py (+62/-218); vllm/v1/worker/gpu_model_runner.py (+2/-2); vllm/v1/worker/tpu_model_runner.py (+281/-674); vllm/v1/worker/tpu_worker.py (+2/-4); vllm/v1/outputs.py (+1/-1)
LABELS: documentation, tpu, ready, ci/build, v1
BODY: This PR integrates the new ragged paged attention kernel with vLLM v1 on TPU. In particular, this PR ⏎  ⏎ - Update torch_xla pin to the latest ⏎ - Update pallas.py in v1 to use the new ragged paged attention kernel instead of the 3 separate kernels in v0. ⏎ - Combine prompt and decode steps into one single step in tpu_model_runner.py, similar to what GPU does today. ⏎  ⏎ Test plan: ⏎ - $ VLLM_USE_V1=1 python vllm/examples/offline_inference/basic.py 2>&1 | tee  …[truncated]

### L3-3b5567a209  (L3, 2025-03-01, sha 3b5567a2099e, PR #13985)
TITLE: [V1][Minor] Do not print attn backend twice (#13985)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+4/-3)
LABELS: ready
BODY: After #13789, the log (`INFO 02-27 10:34:40 [cuda.py:169] Using Flash Attention backend on V1 engine.`) is printed twice. This PR fixes it.

### L3-b28246f6ff  (L3, 2025-03-01, sha b28246f6ff16, PR #14065)
TITLE: [ROCm][V1][Bugfix] Add get_builder_cls method to the ROCmAttentionBackend class (#14065)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn
FILES: vllm/v1/attention/backends/rocm_attn.py (+6/-1)
LABELS: ready, v1
BODY: The ROCmAttentionBackend currently just uses the FlashAttentionMetadata so it can use the FlashAttentionMetadataBuilder. This PR is required now that the V1 GPUModelRunner supports other meta data structures.

### L3-cf069aa8aa  (L3, 2025-03-02, sha cf069aa8aa38, PR #13971)
TITLE: Update deprecated Python 3.8 typing (#13971)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.rocm.v1_rocm_attn, L3.mla.common_v1, L3.mla.flashmla_v1_adapter
FILES: benchmarks/backend_request_func.py (+3/-3); benchmarks/benchmark_guided.py (+8/-9); benchmarks/benchmark_latency.py (+3/-3); benchmarks/benchmark_prefix_caching.py (+8/-8); benchmarks/benchmark_prioritization.py (+4/-4); benchmarks/benchmark_serving.py (+39/-38); benchmarks/benchmark_serving_guided.py (+29/-28); benchmarks/benchmark_throughput.py (+19/-19); benchmarks/benchmark_utils.py (+4/-4); benchmarks/cutlass_benchmarks/sparse_benchmarks.py (+5/-4); (+290 more)
LABELS: documentation, structured-output, frontend, speculative-decoding, ready, ci/build, v1
BODY: The main change is in `pyproject.toml`, the rest is just fixing the pre-commit errors that creates. ⏎  ⏎ The following two rules have been enabled: ⏎ - https://docs.astral.sh/ruff/rules/non-pep585-annotation/ ⏎ - https://docs.astral.sh/ruff/rules/deprecated-import/ ⏎  ⏎ They have been enabled: ⏎ - Outside of the `vllm/` directory ⏎ - Inside `vllm/v1` and `vllm/entrypoints` ⏎ - For the direct children `vllm/*.py`

### L3-848a6438ae  (L3, 2025-03-03, sha 848a6438aed2, PR #12348)
TITLE: [ROCm] Faster Custom Paged Attention kernels (#12348)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.custom_paged, L3.rocm.rocm_flash_attn_v0
FILES: csrc/rocm/attention.cu (+1084/-422); requirements-rocm.txt (+1/-1); vllm/attention/backends/rocm_flash_attn.py (+2/-2); .buildkite/run-amd-test.sh (+0/-1); benchmarks/kernels/benchmark_paged_attention.py (+51/-20); tests/kernels/test_attention.py (+7/-1)
LABELS: documentation, rocm, frontend, speculative-decoding, ready, ci/build
BODY: ## Description ⏎ This PR implements a faster Custom Paged Attention (CPA) kernel based on `mfma16x16x16` instructions. ⏎ This feature is from ROCm/vllm (https://github.com/ROCm/vllm/pull/372). ⏎  ⏎ ## End-to-End Performance gain ⏎ Model: Llama-3.1-70B-Instruct ⏎ Tensor Parallelism: 1 ⏎ GPU: MI300X ⏎  ⏎ | CPA Version | Input length | Output length | KV-cache-dtype | Quantization | Prompt numbers | Req/s | Total Tokens/s | Output Tokens/s | ⏎ |-------------|---------- …[truncated]

### L3-79e4937c65  (L3, 2025-03-03, sha 79e4937c65d5, PR #14155)
TITLE: [v1] Add comments to the new ragged paged attention Pallas kernel (#14155)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+5/-1)
LABELS: ready, v1
BODY: This PR intends to resolve remaining comments in my previous PR https://github.com/vllm-project/vllm/pull/13379.

### L3-72c62eae5f  (L3, 2025-03-04, sha 72c62eae5f01, PR #13931)
TITLE: [V1] EP/TP MoE + DP Attention (#13931)
SOURCES: path_core
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/attention/layer.py (+2/-2); examples/offline_inference/data_parallel.py (+10/-7); tests/kernels/test_moe.py (+1/-0); vllm/compilation/backends.py (+3/-2); vllm/forward_context.py (+15/-7); vllm/model_executor/layers/fused_moe/layer.py (+155/-32); vllm/model_executor/models/aria.py (+11/-7); vllm/model_executor/models/dbrx.py (+6/-2); vllm/model_executor/models/jamba.py (+15/-6); vllm/model_executor/models/mixtral.py (+2/-0); (+7 more)
LABELS: documentation, ready, v1
BODY: Based on https://github.com/vllm-project/vllm/pull/13591 ⏎  ⏎ DP+EP implemented via collective ops in the fused_moe layer's forward pass.

### L3-4dacaa4a83  (L3, 2025-03-05, sha 4dacaa4a8346, PR #14255)
TITLE: [BugFix] Fix prefix caching V0 MLA (#14255)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.triton_v0
FILES: vllm/attention/backends/mla/common.py (+24/-20)
LABELS: ready
BODY: Fix for:  ⏎  ⏎ https://github.com/vllm-project/vllm/issues/14069 ⏎ https://github.com/vllm-project/vllm/issues/14009 ⏎  ⏎ based on: ⏎  ⏎ https://github.com/vllm-project/vllm/issues/14069#issuecomment-2696104480

### L3-f6bb18fd9a  (L3, 2025-03-05, sha f6bb18fd9a19, PR #14253)
TITLE: [BugFix] MLA + V1, illegal memory access and accuracy issues (#14253)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.mla.common_v1, L3.mla.flashmla_v1_adapter
FILES: vllm/v1/attention/backends/flash_attn.py (+2/-2); vllm/v1/attention/backends/mla/common.py (+178/-127); vllm/v1/attention/backends/mla/flashmla.py (+33/-25); vllm/v1/attention/backends/mla/triton_mla.py (+5/-2); vllm/v1/worker/gpu_input_batch.py (+23/-2); vllm/v1/worker/gpu_model_runner.py (+4/-2); tests/v1/worker/test_gpu_input_batch.py (+89/-1)
LABELS: ready, v1
BODY: Fix the decode / prefill split in V1 with MLA not being handled correctly leading to inaccurate results when `max_num_batched_tokens` is low and illegal memory accesses. Also refactored the code to precompute the tensor slicing for prefill and decode metadata. ⏎  ⏎ Main: ⏎  ⏎ ``` ⏎ ** max_num_batched_tokens=1024 ⏎  ⏎ VLLM_USE_V1=1 lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utiliza …[truncated]

### L3-5ee10e990d  (L3, 2025-03-05, sha 5ee10e990deb, PR #11301)
TITLE: [Bugfix][CI] ALiBi test case in xformers multi_query_kv_attention (#11301)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend
FILES: vllm/attention/backends/xformers.py (+0/-2); tests/kernels/test_attention.py (+78/-17); tests/kernels/test_prefix_prefill.py (+5/-3)
LABELS: ready
DEEP_STUDY: deep-study correctness case vllm:5ee10e990d: class=shape_alignment_edge; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: I've been meaning to add a test case to address this TODO https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_attention.py#L367 but I then realized the test would break due to a dim expansion of the attention bias happening here https://github.com/vllm-project/vllm/blob/main/vllm/attention/backends/xformers.py#L783. ⏎  ⏎ Therefore, this PR adds a test case for xformers multi query+alibi bias and fixes the previously untested scenario.

### L3-1b7624bf5c  (L3, 2025-03-05, sha 1b7624bf5cfc, PR #14267)
TITLE: [misc] Add FlashMLA as a new option of VLLM_ATTENTION_BACKEND env (#14267)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+1/-0)
LABELS: documentation, ready
BODY: Add FlashMLA as a new option in the comment of `VLLM_ATTENTION_BACKEND` env variable. This helps new users know that FlashMLA is available as an attention backend choice. ⏎  ⏎ Please help to review~ Thanks! @LucasWilkinson

### L3-ed6ea06577  (L3, 2025-03-05, sha ed6ea06577ec, PR #14244)
TITLE: [Hardware] Update the flash attn tag to support Blackwell (#14244)
SOURCES: path_core, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+2/-2); vllm/attention/backends/utils.py (+6/-1)
LABELS: ready, ci/build
BODY: Update vllm's flash attention tag to update automatically for blackwell platform.

### L3-6bd1dd9d26  (L3, 2025-03-06, sha 6bd1dd9d2625, PR #14152)
TITLE: [Kernel] [V1] Improved performance for V1 Triton (ROCm) backend  (#14152)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.triton.prefix_prefill, L3.triton.chunked_prefill_paged_decode, L3.rocm.v1_rocm_attn
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+289/-0); vllm/attention/ops/prefix_prefill.py (+13/-1); vllm/v1/attention/backends/rocm_attn.py (+20/-17); tests/kernels/test_prefix_prefill.py (+76/-59)
LABELS: rocm, ready, v1
BODY: In this PR we target performance improvement for the V1 `ROCmAttentionBackend` (which should be renamed `TritonAttentionBackend` once [this](https://github.com/vllm-project/vllm/pull/14071) PR is merged).  ⏎  ⏎ cc @SageMoore  ⏎  ⏎ ### Performance  ⏎  ⏎ All of the below results are for `meta-llama/Llama-3.1-8B-Instruct` on an NVIDIA H100 GPU. ⏎  ⏎ The `ROCmAttention` backend on `main` (we have to hack it a bit to make this happen on NVIDIA) currently produces the …[truncated]

### L3-ada19210a3  (L3, 2025-03-06, sha ada19210a356, PR #12613)
TITLE: Adding cpu inference with VXE ISA for s390x architecture (#12613)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: csrc/cpu/attention.cpp (+2/-2); Dockerfile.s390x (+152/-0); cmake/cpu_extension.cmake (+10/-1); csrc/cpu/cpu_types.hpp (+3/-0); csrc/cpu/cpu_types_vxe.hpp (+480/-0); csrc/cpu/quant.cpp (+1/-1); requirements-cpu.txt (+5/-4)
LABELS: ready, ci/build
BODY: This PR provides CPU support for the s390x architecture. Additional PRs in future will optimize the performance and will enable build and test through buildkite. ⏎  ⏎ To build the image for the s390x architecture: ⏎ `docker build --format docker -t vllm:v0.7.0-cpu -f Dockerfile.s390x .` ⏎  ⏎ Example command to run on s390x: ⏎ `docker run -it -p 8000:8000 vllm:v0.7.0-cpu --dtype float --swap_space 1` ⏎  ⏎ Tested models on s390x: ⏎ ``` ⏎ TinyLlama/TinyLlama-1.1B-Chat- …[truncated]

### L3-e642ec962c  (L3, 2025-03-06, sha e642ec962cf2, PR #14371)
TITLE: Add authors to license header. (#14371)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.chunked_prefill_paged_decode
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+5/-0)
LABELS: ready
BODY: I'm making this PR since the co-authors got stripped out when merging https://github.com/vllm-project/vllm/pull/14152.  ⏎  ⏎ cc @robertgshaw2-redhat

### L3-9f1710f1ac  (L3, 2025-03-06, sha 9f1710f1ace3, PR #13897)
TITLE: Fix mla prefill context performance (#13897)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.triton_v0, L3.mla.common_v1
FILES: vllm/attention/backends/mla/common.py (+1/-1); vllm/v1/attention/backends/mla/common.py (+1/-1)
LABELS: ready, v1
BODY: `kv_c_normed` unsqeezed leads to the following `kv_b_proj` slowed down.

### L3-6b2ef5cd17  (L3, 2025-03-06, sha 6b2ef5cd17c5, PR #14313)
TITLE: [Bug] Fix Attention when ignored in by quant_method (#14313)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+3/-1)
LABELS: bug
BODY: There was a bug in the Attention module when a quantized model has the attention module ignored. Many `quant_config.get_quant_method` calls will return `UnquantizedLinearMethod` rather than `None` if the layer is ignored. Maybe we should be returning `UnquantizedAttentionMethod` to be more clear, but that isn't functionally important ⏎  ⏎ On main we hit `assert isinstance(quant_method, BaseKVCacheMethod)`

### L3-6832707e90  (L3, 2025-03-06, sha 6832707e90d4, PR #14221)
TITLE: [V1][Bugfix] Standardize quantized kv cache rejection for attention backends (#14221)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.mla.triton_v0, L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+4/-0); vllm/attention/backends/flash_attn.py (+8/-1); vllm/attention/backends/flashmla.py (+6/-3); vllm/attention/backends/hpu_attn.py (+6/-1); vllm/attention/backends/ipex_attn.py (+3/-2); vllm/attention/backends/pallas.py (+3/-2); vllm/attention/backends/torch_sdpa.py (+6/-2); vllm/attention/backends/triton_mla.py (+6/-3); vllm/v1/attention/backends/flash_attn.py (+5/-1); vllm/v1/attention/backends/mla/flashmla.py (+6/-4); (+1 more)
LABELS: bug, ready, v1
ISSUES: #11329 [Bug]: FP8 kvcache causes RuntimeError in v1 engine | #13133 [Bug]: [V1] wrong output when using kv cache fp8
BODY: Many users have been reporting FP8 KV cache "hasn't been working right" in V1, but in reality we have just been ignoring the parameter since the only (non-MLA) attention backend in V1 is FlashAttention, which doesn't support it. This PR audits existing backends to raise an exception quickly if quantized kv cache is not supported ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/13133  ⏎ FIX https://github.com/vllm-project/vllm/issues/11329 ⏎  ⏎ Before ( …[truncated]

### L3-d9292786e1  (L3, 2025-03-06, sha d9292786e106, PR #13569)
TITLE: [CI/Build] Use uv python for docker rather than ppa:deadsnakes/ppa (#13569)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile (+38/-44)
LABELS: ready, ci/build
DEEP_STUDY: deep-study: this PR was reverted by PR 15377 (confirmed_revert, reason=build_or_dependency)
BODY: I picked this back up since some of the Python apt commands seemed to be timing out on my PRs. The goal of this PR is to use uv instead of ppa:deadsnakes/ppa to install Python

### L3-dae6896977  (L3, 2025-03-06, sha dae68969774e, PR #14384)
TITLE: [Perf] Reduce MLA CPU overheads in V1 (#14384)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+11/-4); vllm/model_executor/layers/rotary_embedding.py (+7/-2)
LABELS: performance, ready, v1
DEEP_STUDY: deep-study: this PR was reverted by PR 14471 (confirmed_revert, reason=correctness_or_accuracy)
BODY: Some temporary hacks to reduce CPU overheads in MLA caused by rotary embeddings (not in torch.compile, or a cuda-graph) ⏎  ⏎ Main ⏎  ⏎ <img width="794" alt="Screenshot 2025-03-06 at 3 14 37 PM" src="https://github.com/user-attachments/assets/f7f46f74-86dd-47b3-8642-b34d25712d1d" /> ⏎  ⏎ This PR ⏎  ⏎ <img width="408" alt="image" src="https://github.com/user-attachments/assets/da3b88e9-10a3-431f-9b9d-e65974cb9e9b" />

### L3-0578e5a462  (L3, 2025-03-06, sha 0578e5a462df, PR #14310)
TITLE: [Hardware][TPU]Enable ragged paged attention kernel and resolve recompilation issue (#14310)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: requirements-tpu.txt (+6/-6); vllm/v1/attention/backends/pallas.py (+23/-16); vllm/v1/worker/tpu_model_runner.py (+29/-44)
LABELS: ready, ci/build, v1
BODY: - enable ragged paged attention kernel ⏎ - resolve recompilation issue ⏎ - change the layout of kv cache from (num_kv_heads, num_blocks, block_size, head_size) to (num_blocks, block_size, num_kv_heads, head_size) ⏎ - update torch xla nightly wheel

### L3-e1744502c2  (L3, 2025-03-07, sha e1744502c21f, PR #14390)
TITLE: [FP8] Refactor apply_fp8_linear and apply_fp8_linear_generic into an object (#14390)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.triton_v0, L3.mla.common_v1
FILES: vllm/attention/backends/mla/common.py (+4/-3); vllm/v1/attention/backends/mla/common.py (+4/-3); tests/compile/test_fusion.py (+7/-13); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py (+8/-11); vllm/model_executor/layers/quantization/fbgemm_fp8.py (+9/-13); vllm/model_executor/layers/quantization/fp8.py (+10/-11); vllm/model_executor/layers/quantization/modelopt.py (+7/-9); vllm/model_executor/layers/quantization/ptpc_fp8.py (+9/-9); vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py (+7/-11); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+54/-38); (+1 more)
LABELS: ready, v1
BODY: This PR replaces `apply_fp8_linear` and `apply_fp8_linear_generic` with objects so that VllmConfig can be accessed in their `__init__` method as opposed to the `forward` method.

### L3-1e3598edeb  (L3, 2025-03-07, sha 1e3598edeb69, PR #14329)
TITLE: Use the optimized block sizes after tuning the kernel. (#14329)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+2/-2)
LABELS: ready, v1
BODY: Use the optimized block sizes after tuning the kernel.

### L3-333681408f  (L3, 2025-03-07, sha 333681408fea, PR #14462)
TITLE: [Bugfix][V1] Handle MLA in kv_cache_interface (#14462)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_model_runner.py (+3/-2); vllm/v1/worker/tpu_model_runner.py (+4/-3); vllm/v1/kv_cache_interface.py (+8/-5)
LABELS: ready, v1
BODY: main:  ⏎ ```INFO 03-07 22:50:53 [kv_cache_utils.py:537] GPU KV cache size: 249,584 tokens``` ⏎ this PR:  ⏎ ```INFO 03-07 22:54:52 [kv_cache_utils.py:537] GPU KV cache size: 499,184 tokens```

### L3-ca7a2d5f28  (L3, 2025-03-07, sha ca7a2d5f28ea, PR #14471)
TITLE: Revert "[Perf] Reduce MLA CPU overheads in V1 (#14384)" (#14471)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+4/-11); vllm/model_executor/layers/rotary_embedding.py (+2/-7)
LABELS: v1
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 14384 reason=correctness_or_accuracy
BODY: Running ⏎ ``` ⏎ VLLM_USE_V1=1 vllm serve deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --tensor_parallel_size=2 --port 8192 --trust-remote-code ⏎ ``` ⏎ and then ⏎ ``` ⏎ lm_eval --model local-completions --tasks gsm8k --model_args model=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct,base_url=http://127.0.0.1:8192/v1/completions,num_concurrent=5,max_retries=3,tokenized_requests=False --limit 100 ⏎ ``` ⏎  ⏎ On current main we see: ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-sho …[truncated]

### L3-206e2577fa  (L3, 2025-03-08, sha 206e2577fa9c, PR #12547)
TITLE: Move requirements into their own directory (#12547)
SOURCES: corpus:production-kernel-provenance
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: .buildkite/nightly-benchmarks/scripts/run-nightly-benchmarks.sh (+1/-1); .buildkite/run-cpu-test.sh (+1/-1); .buildkite/test-pipeline.yaml (+1/-1); .github/workflows/publish.yml (+1/-1); .github/workflows/scripts/build.sh (+1/-1); .pre-commit-config.yaml (+2/-2); .readthedocs.yaml (+1/-1); Dockerfile (+12/-12); Dockerfile.arm (+5/-5); Dockerfile.cpu (+5/-5); (+40 more)
LABELS: documentation, ready, ci/build
BODY: Moves `requirements-x.txt` to `requirements/x.txt` and updates all references to requirements files.

### L3-db84f5eb3b  (L3, 2025-03-08, sha db84f5eb3bd2, PR #14476)
TITLE: [Bugfix] DeepSeek Accuracy (#14476)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+7/-5)
LABELS: ready, v1
BODY: Fix accuracy issue introduced by https://github.com/vllm-project/vllm/pull/14384 , on main `forward_native` is called (due to `enabled()` be false in `CustomOp(nn.Module)` on main) the PR updated this cuda, changing the behavior slightly. We should probably investigate further to figure out why `forward_native` differs form `forward_cuda` (on an Nvidia GPUs). This is potentially due to `q_pe` and `k_pe` being different shapes in MLA.  ⏎  ⏎ Main ⏎ ``` ⏎ l …[truncated]

### L3-b0d541947a  (L3, 2025-03-08, sha b0d541947ab5, PR #14451)
TITLE: [Attention] Default to FlashMLA backend for MLA (#14451)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+24/-16)
LABELS: ready
BODY: Numbers from @simon-mo , seems like a consistent enough win to have on by default ⏎  ⏎ ![image](https://github.com/user-attachments/assets/2a2c83e7-33e7-41a3-b58e-31a9ca33130c)

### L3-10f7552789  (L3, 2025-03-08, sha 10f755278953, PR #14467)
TITLE: [V1][TPU] Remove unnecessary padding for running on TPU. (#14467)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+2/-2); vllm/v1/worker/tpu_model_runner.py (+4/-16)
LABELS: tpu, ready, v1
BODY: Remove unnecessary padding for running on TPU. Also update the tunable block size (NUM_QUERIES_PER_BLOCK, NUM_KV_PAGES_PER_BLOCK) for the ragged kernel. ⏎  ⏎ Test plan: ⏎ 1. $ VLLM_USE_V1=1 pytest -s -v vllm/tests/entrypoints/llm/test_accuracy.py::test_lm_eval_accuracy_v1_engine 2>&1 | tee out.txt ⏎ 2.  ⏎ ``` ⏎ VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct --disable-log-requests --port 8003 --gpu-memory-utilization 0.95 --max-num-batched-tokens 8 …[truncated]

### L3-fb0acb6c72  (L3, 2025-03-10, sha fb0acb6c7287, PR #14540)
TITLE: [Perf] Improve MLA on V1 (#14540)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+41/-27)
LABELS: ready, v1
BODY: This PR helps V1 to _mostly match_ and exceed (in most cases) V0's performance for MLA. Mostly by two things ⏎ 1. Fix @LucasWilkinson's `rotary_emb` specialization (#14384, #14471, #14476) to reduce CPU overhead. ⏎   * Identified that the cause of 0 GSM8K score comes from the cuda kernel needs the input to be continuous. ⏎   * Fixed it by make the input contiguous if possible. A better fix will be to change the kernel (help wanted).  ⏎ 2. Reordered some  …[truncated]

### L3-3b352a2f92  (L3, 2025-03-10, sha 3b352a2f92bc, PR #14562)
TITLE: Correct capitalisation: `VLLM` -> `vLLM` (#14562)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.dispatch.selector, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/attention/selector.py (+1/-1); benchmarks/kernels/benchmark_rmsnorm.py (+1/-1); docs/source/contributing/vulnerability_management.md (+1/-1); docs/source/design/v1/metrics.md (+1/-1); examples/offline_inference/disaggregated_prefill_lmcache.py (+1/-1); tests/tpu/test_quantization_accuracy.py (+1/-1); vllm/compilation/backends.py (+1/-1); vllm/compilation/compiler_interface.py (+1/-1); vllm/config.py (+4/-4); vllm/entrypoints/openai/protocol.py (+1/-1); (+8 more)
LABELS: documentation, frontend, ready, v1
BODY: 

### L3-c91b64f749  (L3, 2025-03-10, sha c91b64f749d3, PR #14391)
TITLE: [neuron] add reshape_and_cache (#14391)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/nki_flash_attn.py (+43/-0); tests/neuron/test_cache.py (+83/-0)
BODY: Add `reshape_and_cache` function for Neuron KV cache updates ⏎  ⏎ Implements a helper function to write key-value pairs into block-based KV cache tensors. Handles the layout mismatch between: ⏎ - Input tensors: (num_tokens, n_kv_head, d_head)   ⏎ - Cache tensors: (num_blocks, n_kv_head, block_size, d_head) ⏎  ⏎ Uses block index calculations and torch.index_put_ to efficiently map and write the inputs into the correct cache positions. Optimized for Neuron's m …[truncated]

### L3-53056731fd  (L3, 2025-03-11, sha 53056731fdf8, PR #14627)
TITLE: fix some typos : supported_head_sizes (#14627)
SOURCES: path_core
ARTIFACT_HINTS: L3.blocksparse.v0
FILES: vllm/attention/backends/blocksparse_attn.py (+3/-3)
BODY: 

### L3-a1c8f3796c  (L3, 2025-03-11, sha a1c8f3796c89, PR #14245)
TITLE: dynamic distpatch of fp8 kernels (#14245)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.triton_v0, L3.mla.common_v1, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/attention/backends/mla/common.py (+3/-3); vllm/v1/attention/backends/mla/common.py (+4/-3); benchmarks/kernels/benchmark_moe.py (+1/-2); csrc/dispatch_utils.h (+26/-6); csrc/layernorm_quant_kernels.cu (+33/-25); csrc/quantization/fp8/amd/quant_utils.cuh (+22/-0); csrc/quantization/fp8/common.cu (+43/-25); csrc/quantization/fp8/common.cuh (+59/-27); csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu (+3/-0); csrc/quantization/fused_kernels/quant_conversions.cuh (+11/-8); (+15 more)
LABELS: rocm, ready, v1
BODY: This mostly affects ROCm with hardware that can support one or the other FP8 type. All fp8 kernels are now templated on fp8_type instead of assuming a single fp8_type via `using FP8_TYPE = `. CUDA is largely unaffected.  For CUDA, fp8 kernels only instantiate the OCP type; no binary bloat. For ROCm, two kernel templates are instantiated, one for each fp8 type. ⏎  ⏎ - FP8_TYPE is removed. ⏎ - All fp8 kernels have additional fp8_type template param. ⏎ - Th …[truncated]

### L3-863d315c86  (L3, 2025-03-11, sha 863d315c867e, PR #14597)
TITLE: [V1][TPU] Pad the block_table.shape[1] so the ragged paged attention can handle correctly (#14597)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/tpu_model_runner.py (+5/-2)
LABELS: bug, tpu, ready, v1
BODY: We uncovered an ragged paged attention constraint that will cause trouble if it is not handle properly. The kernel assumes that block_table.shape[1]%NUM_KV_PAGES_PER_BLOCK==0. If this constraint is violated, there may be out-of-bound issue and the kernel will crash. ⏎  ⏎ While we are fixing the kernel internally, we want to fix the vLLM in the interim so that our vLLM users won't see the error. Once the kernel is fixed, we can safely revert this PR. ⏎  …[truncated]

### L3-916836bbfb  (L3, 2025-03-12, sha 916836bbfb7e, PR #14664)
TITLE: [FEAT] [ROCm] [Embedding] Add encoder-only model support into ROCm Flash Attention to enable embedding models. (#14664)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.rocm.rocm_flash_attn_v0
FILES: CMakeLists.txt (+4/-0); vllm/attention/backends/rocm_flash_attn.py (+68/-44); csrc/moe/torch_bindings.cpp (+1/-0); tests/models/embedding/language/test_cls_models.py (+15/-2); tests/models/embedding/language/test_embedding.py (+11/-2); tests/models/embedding/language/test_gritlm.py (+2/-2); tests/models/embedding/vision_language/test_llava_next.py (+17/-0)
LABELS: rocm, ready, ci/build
ISSUES: #14062 [Bug]: Alibaba-NLP/gte-Qwen2-7B-instruct on AMD MI300X | #14583 [Bug]: ERROR 03-11 07:47:00 [engine.py:141] AttributeError: Invalid attention type encoder-only
BODY: # Description ⏎ This PR add the logic to enable ENCODER_ONLY model support to ROCm Flash Attention. Thus, enabling language embedding models and some vision language embedding models. ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/14062 ⏎ FIX #14583 ⏎  ⏎ # File changes: ⏎ * `vllm/attention/backends/rocm_flash_attn.py`: Add ENCODER_ONLY code path ⏎ * `tests/models/embedding/language/test_embedding.py`: Fix the code logic to enable unit tests for ROCm suppor …[truncated]

### L3-ff47aab056  (L3, 2025-03-12, sha ff47aab05640, PR #13381)
TITLE: [CPU] Upgrade CPU backend to torch-2.6 (#13381)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: vllm/attention/ops/ipex_attn.py (+1/-1); .buildkite/run-cpu-test.sh (+5/-3); Dockerfile.cpu (+1/-1); cmake/cpu_extension.cmake (+1/-1); requirements/cpu.txt (+1/-1); tests/lora/test_qwen2vl.py (+1/-1); vllm/executor/multiproc_worker_utils.py (+5/-4); vllm/model_executor/layers/fused_moe/layer.py (+5/-1); vllm/platforms/cpu.py (+3/-0)
LABELS: ready, ci/build
BODY: Upgrade CPU backend torch to 2.6.0, all tests are verified on local. ⏎  ⏎ ~~Waiting for #12721~~

### L3-d9f83d6206  (L3, 2025-03-12, sha d9f83d62068b, PR #14316)
TITLE: [ROCm] Enable chunked prefill/paged attention in MLA on ROCm (#14316)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.triton_v0
FILES: vllm/attention/backends/mla/common.py (+2/-16); vllm/config.py (+2/-2)
LABELS: rocm, ready
BODY: This PR is largely just removing the guards in config.py to allow chunked prefill and paged attention in MLA. The LSE computation in the triton kernel doesn't work so we always fall back to flash attention in this case. ⏎  ⏎ I ran `lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct,trust_remote_code=True,enable_chunked_prefill=True --tasks gsm8k --num_fewshot 5 --batch_size auto` and got  ⏎  ⏎ ``` ⏎ vllm (pretrained=de …[truncated]

### L3-fb4c7f8ef0  (L3, 2025-03-13, sha fb4c7f8ef016, PR #14431)
TITLE: [Kernel] [V1] Further optimizations to ROCm (Triton) Backend to better handle GQA. (#14431)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.chunked_prefill_paged_decode
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+63/-40)
LABELS: ready
BODY: **TLDR:** This PR adds some further optimizations to `chunked_prefill_paged_decode` op to better handle models with GQA. Serving benchmarks using V1 indicate that with these changes, we see a **25% improvement in throughput** for `llama3.1-8b` on an H100 vs. the current Triton implementation. With these changes, the throughput of the Triton implementation is only **8% worse than the V1 CUDA backend** (FlashAttention). ⏎  ⏎ Using `FlashAttentionBacken …[truncated]

### L3-d3d4956261  (L3, 2025-03-13, sha d3d4956261e8, PR #14712)
TITLE: [Neuron] flatten test parameterization for neuron attention kernels (#14712)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/run-neuron-test.sh (+1/-1); tests/neuron/1_core/test_activation.py (+0/-0); tests/neuron/1_core/test_block_table.py (+0/-0); tests/neuron/1_core/test_cache.py (+0/-0); tests/neuron/1_core/test_layernorm.py (+0/-0); tests/neuron/1_core/test_logits_processor.py (+0/-0); tests/neuron/1_core/test_prefix_prefill.py (+25/-21); tests/neuron/1_core/test_rotary_embedding.py (+0/-0); tests/neuron/2_core/test_comm_ops.py (+0/-0)
LABELS: ci/build
BODY: This PR include two changes: ⏎  ⏎ - Split the test scripts into 1_core and 2-core test folders ⏎ - Refactored test_contexted_kv_attention to flatten the nested parameterization (3 loops → 1 list) while expanding coverage variety. The change reduces test combinations from 64 to 10 carefully selected cases, achieving ~80% CI time reduction. The new test suite covers 2x more parameter ranges - from minimal (1-8 heads) through common (16-32 heads) to large …[truncated]

### L3-d47807ba08  (L3, 2025-03-13, sha d47807ba0806, PR #14769)
TITLE: [Attention] Remove slow setattr in MLA (#14769)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/rotary_embedding.py (+7/-2)
LABELS: ready
BODY: <img width="516" alt="image" src="https://github.com/user-attachments/assets/bc36adfb-fd5e-4dc7-9d34-f50eea1bf8d1" /> ⏎  ⏎ Partial undo of https://github.com/vllm-project/vllm/pull/14471 which was a revert of https://github.com/vllm-project/vllm/pull/14384

### L3-9532c49836  (L3, 2025-03-13, sha 9532c49836ad, PR #14770)
TITLE: [Attention] MLA get rid of materialization (#14770)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.triton_v0, L3.mla.common_v1
FILES: vllm/attention/backends/mla/common.py (+57/-210); vllm/envs.py (+0/-19); vllm/v1/attention/backends/mla/common.py (+58/-213); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+2/-57)
LABELS: ready, v1
BODY: Based on these calculations: ⏎  ⏎ https://docs.google.com/spreadsheets/d/17eoqEbhblvtNsRRlFSjCQnEXZiBxtLgZGKD4IgZUz38/edit?usp=sharing ⏎  ⏎ It's actually better to just not materialize the absorbed `W_Q_UK` and `W_UV_O` as it reduces memory usage (and total flops) and instead compute using sequential matmuls. One issue is that we do not have an FP8 bmm (which is needed if not materializing the absorbed matrix, materializing  absorbing allowed us to bypas …[truncated]

### L3-14f301b541  (L3, 2025-03-14, sha 14f301b541ff, PR #12721)
TITLE: Update to torch==2.6.0 (#12721)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+2/-2); Dockerfile (+1/-1); requirements/build.txt (+1/-1); requirements/cuda.txt (+5/-5); requirements/test.in (+4/-3); requirements/test.txt (+10/-8); pyproject.toml (+1/-1); tests/compile/backend.py (+4/-2); vllm/config.py (+15/-0)
LABELS: ready, ci/build
ISSUES: #12719 [Installation]: Supporting PyTorch 2.6?
BODY: Only updates for CUDA. Successfully built locally on H100 CUDA 12.5 system and tested with `vllm serve meta-llama/Llama-3.1-8B-Instruct` ⏎  ⏎ We should upgrade other hardware backends separately. For instance, CPU is blocked by IPEX in the Dockerfile.cpu ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/12719

### L3-d4d93db2c5  (L3, 2025-03-14, sha d4d93db2c54a, PR #13726)
TITLE: [V1] V1 Enablement Oracle  (#13726)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: .buildkite/lm-eval-harness/configs/Minitron-4B-Base-FP8.yaml (+2/-2); .buildkite/lm-eval-harness/test_lm_eval_correctness.py (+5/-0); .buildkite/test-pipeline.yaml (+18/-16); tests/async_engine/conftest.py (+11/-0); tests/async_engine/test_api_server.py (+5/-1); tests/async_engine/test_async_llm_engine.py (+9/-0); tests/basic_correctness/test_chunked_prefill.py (+9/-0); tests/basic_correctness/test_cpu_offload.py (+7/-0); tests/basic_correctness/test_preemption.py (+9/-0); tests/compile/conftest.py (+14/-0); (+86 more)
LABELS: documentation, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality
BODY: SUMMARY: ⏎ * adds oracle for V1 feature enablement ⏎ * removes usage of `envs.VLLM_USE_V1` from code base (now only used in tests + in oracle) ⏎ * enables V1 by default ⏎  ⏎ BEHAVIOR ⏎ ``` ⏎ # * If VLLM_USE_V1 is unset, we enable V1 for "supported features" ⏎ #   and fall back to V0 for experimental or unsupported features. ⏎ # * If VLLM_USE_V1=1, we enable V1 for supported + experimental ⏎ #   features and raise error for unsupported features. ⏎ # * If VLLM_USE_V1=0, …[truncated]

### L3-a2ae496589  (L3, 2025-03-14, sha a2ae49658901, PR #14741)
TITLE: [CPU] Support FP8 KV cache (#14741)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/torch_sdpa.py (+5/-4); csrc/cpu/cache.cpp (+21/-17); csrc/cpu/cpu_types_x86.hpp (+9/-0); docs/source/getting_started/installation/cpu.md (+1/-1); tests/basic_correctness/test_chunked_prefill.py (+2/-2); tests/models/decoder_only/language/test_fp8.py (+61/-0); vllm/platforms/cpu.py (+19/-11); vllm/worker/cpu_worker.py (+4/-1)
LABELS: documentation, ready
BODY: This PR added FP8-E5M2 KV cache support for the CPU backend, and also updated tests for chunked-prefill with ```Half``` dtype.
