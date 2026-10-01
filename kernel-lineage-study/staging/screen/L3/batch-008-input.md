### L3-ed6ae1e36a  (L3, 2025-11-20, sha ed6ae1e36a03, PR #29124)
TITLE: [AITER] [ROCm] Fix crash when loading llama4 model with old aiter version installed, fallback to forward_native implementation (#29124)
SOURCES: subject_keyword, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/sample/ops/topk_topp_sampler.py (+13/-6)
LABELS: rocm, ready, v1, llama
DEEP_STUDY: deep-study correctness case vllm:ed6ae1e36a: class=hardware_compiler_specific; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎  ⏎ Catch importing aiter.ops.sampling failure and fallback to forward_native implementation. So that running vllm without aiter sampling on AMD works. ⏎  ⏎ ## Test Plan ⏎  ⏎ In my local test env, aiter old version is installed, started the server with llama4 model ⏎  ⏎ ## Test Result ⏎  ⏎ In the server start up log, will get "aiter.ops.sampling is not available on ROCm..." warning log instead of crash: ⏎  ⏎ ``` ⏎ INFO 11-20 14:02:11 [parallel_state.py:1425] r …[truncated]

### L3-647464719b  (L3, 2025-11-20, sha 647464719b13, PR #27743)
TITLE: [KVConnector][Core] Support cross-layer KV blocks (#27743)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+28/-1); vllm/v1/attention/backends/flash_attn.py (+10/-2); vllm/v1/attention/backends/flashinfer.py (+10/-2); vllm/v1/attention/backends/mla/common.py (+9/-0); vllm/v1/attention/backends/mla/indexer.py (+5/-1); tests/v1/kv_connector/unit/test_offloading_connector.py (+6/-2); tests/v1/kv_offload/test_cpu_offloading.py (+93/-52); tests/v1/worker/test_gpu_model_runner.py (+4/-1); vllm/distributed/kv_transfer/kv_connector/v1/base.py (+31/-2); vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py (+38/-5); (+5 more)
LABELS: ready, v1, kv-connector, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Core of RFC #27742. ⏎ Following this PR, connectors can turn-on and adapt to the new layout. ⏎  ⏎ This PR enables the GPU model runner to allocate the KV cache tensors, so that the KV data for all layers will be contiguous per block. This can yield a significant speed up the transfer time of KV transfers (e.g. X4), such in the case of using NixlConnector or OffloadingConnector. Currently, this new layout is disabled by default, and will only be enabled …[truncated]

### L3-11857a00b0  (L3, 2025-11-20, sha 11857a00b0a5, PR #29103)
TITLE: [Attention] Add ROCM_AITER_MLA_SPARSE to attention backend registry (#29103)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.dispatch.registry, L3.platform.rocm_selection
FILES: vllm/attention/backends/registry.py (+3/-0); vllm/platforms/rocm.py (+1/-4)
LABELS: rocm, ready
BODY: ## Purpose ⏎ Adds the new `ROCM_AITER_MLA_SPARSE` backend introduced by #26670 to the attention backend registry. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-3d84ef9054  (L3, 2025-11-20, sha 3d84ef9054af, PR #29043)
TITLE: [CI/Build][AMD] Skip if flash_attn_varlen_func not available in test_aiter_flash_attn.py (#29043)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_aiter_flash_attn.py (+3/-0)
LABELS: rocm, ready
BODY: This PR skips the tests in `test_aiter_flash_attn.py` if `flash_attn_varlen_func `is not available.  The function isn't currently available, but could be in the future, or the test might get updated.   ⏎  ⏎ This is similar to how `test_flash_attn.py` works.

### L3-5e5a7eb16f  (L3, 2025-11-20, sha 5e5a7eb16f12, PR #29064)
TITLE: [CI/Build] Make test_attention_selector.py run tests on correct platform (#29064)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_attention_selector.py (+4/-1)
LABELS: ready
BODY: This PR fixes test_attention_selector.py to make sure that tests run only correct platform, so if `hip `or `cuda` is selected, the test will only run on the corresponding platform, since the `get_attn_backend`  needs to be on the correct platform to work correctly.

### L3-fc9f821d20  (L3, 2025-11-21, sha fc9f821d2062, PR #28346)
TITLE: fix cross attention (#28346)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+9/-8)
LABELS: ready, v1
ISSUES: #27442 [CI Failure][AMD] Encoder-Decoder Models Fail on AMD CI
BODY: ## Purpose ⏎ Fix https://github.com/vllm-project/vllm/issues/27442 ⏎ ## Test Plan ⏎ pytest entrypoints/openai/test_transcription_validation.py::test_basic_audio[openai/whisper-large-v3-turbo] --maxfail=1 ⏎ ## Test Result ⏎ ``` ⏎ =============================================================================== test session starts =============================================================================== ⏎ platform linux -- Python 3.12.11, pytest-8.4.1, plugg …[truncated]

### L3-30b44a1598  (L3, 2025-11-21, sha 30b44a1598ea, PR #25266)
TITLE: GPU Model Runner V2 (#25266)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+3/-0); .github/CODEOWNERS (+3/-0); vllm/envs.py (+5/-0); vllm/v1/core/sched/output.py (+24/-0); vllm/v1/core/sched/scheduler.py (+22/-6); vllm/v1/worker/gpu/README.md (+4/-0); vllm/v1/worker/gpu/__init__.py (+0/-0); vllm/v1/worker/gpu/async_utils.py (+89/-0); vllm/v1/worker/gpu/attn_utils.py (+187/-0); vllm/v1/worker/gpu/block_table.py (+315/-0); (+8 more)
LABELS: documentation, ready, ci/build, v1, nvidia
ISSUES: #23446 [RFC]: Redesigning Persistent Batch in vLLM
BODY: # Key Changes ⏎  ⏎ * Remove persistent batch ⏎   * No “reordering” & complex bookkeeping ⏎   * Almost all CPU states are Numpy arrays → We can vectorize most of the Python loops in pre-/post-processing ⏎   * Simpler handling for requests resumed from preemption ⏎ * GPU-persistent block tables ⏎   * The CPU does not have the block tables at all. GPU maintains the persistent block tables. ⏎   * In every step, we only send the “diff”s to the GPU, and use a Triton k …[truncated]

### L3-b4c8fbaae2  (L3, 2025-11-21, sha b4c8fbaae259, PR #28892)
TITLE: Add TRTLLM MoE NVFP4 kernel to CompressedTensorsW4A4MoeMethod (#28892)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+122/-20); vllm/model_executor/layers/quantization/modelopt.py (+15/-190); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+221/-0)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ This PR is created for two purposes: ⏎ 1. Add trtllm fp4 kernel to be used in compressedtensorw4a4 method ⏎ 2. Create a mutually used function for modelopt and compressedtensorw4a4 path to call flashinfer trtllm moe fp4 kernel ⏎  ⏎ ## Test Plan ⏎ ### Accuracy test ⏎ Tested all code paths before and after the code change, including: ⏎ 1. modelopt + trtllm fp4 moe ⏎ 2. modelopt + flashinfer cutlass moe ⏎ 3. compressedtensor + cutlass moe ⏎ 4. compressedtens …[truncated]

### L3-b7f1f490a6  (L3, 2025-11-21, sha b7f1f490a61c, PR #28888)
TITLE: Upstream triton fp4 weight preshuffle (#28888)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/_aiter_ops.py (+25/-0); vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py (+52/-15)
LABELS: rocm, ready
BODY: ## Server command: ⏎ ``` ⏎ HIP_VISIBLE_DEVICES=7 \ ⏎ VLLM_DISABLE_COMPILE_CACHE=1 \ ⏎ USE_FASTSAFETENSOR=1 \ ⏎ SAFETENSORS_FAST_GPU=1 \ ⏎ VLLM_USE_V1=1 \ ⏎ AMDGCN_USE_BUFFER_OPS=1 \ ⏎ TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 \ ⏎ TRITON_HIP_USE_ASYNC_COPY=1 \ ⏎ TRITON_HIP_USE_BLOCK_PINGPONG=1 \ ⏎ TRITON_HIP_ASYNC_FAST_SWIZZLE=1 \ ⏎ VLLM_ROCM_USE_AITER=1 \ ⏎ VLLM_ROCM_USE_AITER_MHA=0 \ ⏎ VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 \ ⏎ VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 \ ⏎ VLLM_ROC …[truncated]

### L3-c68c7b403d  (L3, 2025-11-21, sha c68c7b403dce, PR #29107)
TITLE: [BugFix] Fix missing symbol triggering FA2 fallback on Hopper (#29107)
SOURCES: path_core, subject_keyword, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: bug, ready, ci/build
BODY: vLLM side of https://github.com/vllm-project/flash-attention/pull/111 (land that first)

### L3-066209a045  (L3, 2025-11-22, sha 066209a04521, PR #29084)
TITLE: [Attention] Refactor FA `block_size` limitations to hybrid models only  (#29084)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v1_backend, L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.aiter_fa, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.flashmla_sparse, L3.mla.rocm_aiter, L3.dispatch.abstract_interface, L3.tree_attention
FILES: vllm/attention/backends/abstract.py (+7/-3); vllm/v1/attention/backends/flash_attn.py (+21/-6); vllm/v1/attention/backends/flashinfer.py (+6/-6); vllm/v1/attention/backends/mla/cutlass_mla.py (+4/-1); vllm/v1/attention/backends/mla/flashattn_mla.py (+4/-1); vllm/v1/attention/backends/mla/flashinfer_mla.py (+4/-1); vllm/v1/attention/backends/mla/flashmla.py (+4/-1); vllm/v1/attention/backends/mla/flashmla_sparse.py (+4/-1); vllm/v1/attention/backends/mla/indexer.py (+3/-3); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+3/-1); (+7 more)
LABELS: rocm, ready, v1, nvidia
BODY: This PR limits the blocks-size changes introduced in https://github.com/vllm-project/vllm/pull/27753/ to hybrid-models only. ⏎ As non hybrid-models are not affected by the issues reported in the PR above, not allowing a supported physical block_size can hinder performance of kv cache transfers. ⏎  ⏎ Eg on the likes of PD disaggregation with NIXL, we're not limited by bw so reducing the size of blocks reduces optimal saturation of medium. ⏎  ⏎ To do so I ha …[truncated]

### L3-730bd35378  (L3, 2025-11-22, sha 730bd35378bf, PR #29193)
TITLE: [perf][cpu] Accelerate paged attention GEMMs (QK, PV) on Arm CPUs with NEON (#29193)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/engine/arg_utils.py (+1/-2); vllm/v1/attention/backends/cpu_attn.py (+5/-2); csrc/cpu/cpu_attn.cpp (+17/-0); csrc/cpu/cpu_attn_impl.hpp (+7/-1); csrc/cpu/cpu_attn_neon.hpp (+386/-0)
LABELS: performance, ready, v1
ISSUES: #28981 [Bug]: chunked prefill disabled & max batched tokens not compatible with max model length on non-X86 CPU Backend
DEEP_STUDY: deep-study performance PR ()
BODY: [perf][cpu] Accelerate paged attention GEMMs (QK, PV) on Arm CPUs with NEON ⏎  ⏎ NEON is an Arm SIMD instruction set extension, compulsory since `armv8-a` ⏎  ⏎ ## Purpose ⏎  ⏎ Fixes #28981 for Arm CPUs ⏎ Related to #23934 (since it enables attention sinks for Arm) ⏎  ⏎ PR #27954 added `cpu_attention_with_kv_cache` which supports chucked prefill, prefix caching, SWA, alibi, softcap and sinks. ⏎  ⏎ However, it's currently disabled for the prefill phase on Arm CPUs becau …[truncated]

### L3-eb5352a770  (L3, 2025-11-22, sha eb5352a7707d, PR #26966)
TITLE: [CI/build] Removes source compilation from runtime image (#26966)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+43/-27); docs/assets/contributing/dockerfile-stages-dependency.png (+0/-0); tools/ep_kernels/install_python_libraries.sh (+86/-74); tools/install_deepgemm.sh (+30/-14)
LABELS: documentation, ready, ci/build
BODY: ## Purpose ⏎  ⏎ This PR removes the need to build NVSHMEM from source which should build times of the Docker container by around 3-5 minutes ⏎ Additionally this moves any other source builds DEEPGEMM etc from the final runtime container to the build container and copies over the resulting artifacts to the final runtime image. This allows for removing any build level tools (compilers, git, etc) from the actual final image to reduce its size. ⏎ This additi …[truncated]

### L3-8e22da1d7f  (L3, 2025-11-22, sha 8e22da1d7fcd, PR #29229)
TITLE: [CI/Build Don't add FLASHINFER backend in test_cpu_offloading.py (#29229)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_offload/test_cpu_offloading.py (+5/-1)
LABELS: ready, v1
BODY: This fixes a test failure where` tests/v1/kv_offload/test_cpu_offloading.py` adds the `FLASHINFER `backend to the test, but ROCm platform does not support `flashinfer` library.  Doing this allows the test to be successful in AMD CI.  The test runs to completion and the result is: ⏎  ⏎  1 passed, 3 warnings

### L3-5f96c00c55  (L3, 2025-11-23, sha 5f96c00c557f, PR #29144)
TITLE: [Fix] Add SM check to flashinfer MOE backend (#29144)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+10/-0)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎ Add SM version check to flashinfer MOE backend because Flashinfer TRTLLM MOE only support SM100 and later. ⏎ ## Test Plan ⏎ Not functional change, just a check and fallback so to test plan ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-5253f4276f  (L3, 2025-11-24, sha 5253f4276f33, PR #28376)
TITLE: [ROCm] Support for Whisper v1 with Aiter Unified Attention and Aiter Flash Attention (#28376)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.rocm.aiter_unified
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+14/-8); vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+12/-2); vllm/v1/attention/backends/rocm_attn.py (+2/-5)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎ This PR enables broader attention backend support for Whisper v1 on ROCm platform. ⏎ Building on the existing Triton backend PR #28346 , it introduces: ⏎  ⏎ Aiter Unified Attention ⏎ Aiter Flash Attention ⏎  ⏎ This change depends on modifications from the Triton backend PR. Since both PRs modify the same file (`vllm/v1/worker/utils.py`). ⏎  ⏎ ## Test Plan ⏎ Whisper v1 on ROCm with Aiter backends requires the latest Aiter version, tested with commit 7639 …[truncated]

### L3-0ff70821c9  (L3, 2025-11-24, sha 0ff70821c9b0, PR #29262)
TITLE: [Core] Deprecate `xformers` (#29262)
SOURCES: path_core, symbol_pickaxe, dependency_pin, body_keyword
ARTIFACT_HINTS: L3.xformers.v1_backend, L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen, L3.dispatch.selector, L3.dispatch.registry, L3.platform.cuda_selection
FILES: docker/Dockerfile.nightly_torch (+1/-34); requirements/cuda.txt (+0/-1); vllm/attention/backends/registry.py (+0/-1); vllm/attention/layer.py (+0/-38); vllm/attention/ops/vit_attn_wrappers.py (+1/-37); vllm/attention/selector.py (+8/-1); vllm/v1/attention/backends/xformers.py (+0/-420); docs/contributing/ci/update_pytorch_version.md (+0/-15); docs/getting_started/quickstart.md (+1/-1); examples/online_serving/openai_embedding_long_text/service.sh (+0/-1); (+21 more)
LABELS: documentation, ready, ci/build, v1, qwen, nvidia
BODY: ## Purpose ⏎ Reopened from #28287 ⏎  ⏎ This PR completely removes the dependency of `xformers` library and should be only merged after v0.11.1 release. The rationale behind removing `xformers` is that:  ⏎   1. `xformers` is used for multimodal attention (MHA) but we can have alternative attention backends to replace it ⏎   2. We have `xformers` attention backend for decoder LM, but it's no longer used for anything ⏎   3. Having another external dependency pu …[truncated]

### L3-4d6afcaddc  (L3, 2025-11-24, sha 4d6afcaddcca, PR #29270)
TITLE: [CI/Build] Moves to cuda-base runtime image while retaining minimal JIT dependencies (#29270)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+14/-2); docs/assets/contributing/dockerfile-stages-dependency.png (+0/-0)
LABELS: documentation, ready, ci/build, nvidia, ready-run-all-tests
BODY: ## Purpose ⏎  ⏎ Now that https://github.com/vllm-project/vllm/pull/26966 is merged, more runtime image dependencies can be culled. The `devel` base image isn't needed anymore and switched to `base`. While flashinfer doesn't require source compilation, DeepGEMM and DeepEP still do, so some JIT dependencies are still needed.  ⏎  ⏎ ``` ⏎ vllm:old                                           92e48d400646       27.7GB            0B    U    ⏎ vllm:new                 …[truncated]

### L3-e48b2e6848  (L3, 2025-11-24, sha e48b2e6848ac, PR #26980)
TITLE: [Bugfix] [ROCm] [UX] Reorganize ROCm Backend Selection Logic (#26980)
SOURCES: symbol_pickaxe, corpus:confirmed-reverts(reverted), body_keyword
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: tests/v1/attention/test_rocm_attention_backends_selection.py (+337/-0); vllm/platforms/rocm.py (+57/-23)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study: this PR was reverted by PR 29371 (partial_revert, reason=premature_or_process)
BODY: ## Purpose ⏎  ⏎ This is to simplify the user experiences so that users do not need to keep track of the environment variables to select the backend. Users can directly specify the backend through `VLLM_ATTENTION_BACKEND`. ⏎  ⏎ ### Fix Attention Backend Configuration on ROCm ⏎  ⏎ This PR fixes several issues with attention backend configuration on AMD ROCm platforms to ensure proper backend selection based on environment variables and flags.  ⏎  ⏎ ### Issues Fixe …[truncated]

### L3-699bca76c0  (L3, 2025-11-24, sha 699bca76c00b, PR #29348)
TITLE: [UX] Raise error for attn backend of batch invariant (#29348)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/batch_invariant.py (+7/-7)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Raise error instead of setting implicitly.

### L3-77e10c9cab  (L3, 2025-11-24, sha 77e10c9cab75, PR #28029)
TITLE: [Perf][Deepseek] optimize gather_and_maybe_dequant_cache kernel's perf for extremely long sequence (#28029)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.mla.common_v1
FILES: csrc/cache_kernels.cu (+86/-92); vllm/v1/attention/backends/mla/common.py (+25/-3); csrc/cache.h (+6/-5); csrc/torch_bindings.cpp (+2/-1); tests/kernels/attention/test_cache.py (+9/-3); vllm/_custom_ops.py (+4/-2)
LABELS: performance, ready, v1, deepseek
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ We found the `gather_and_maybe_dequant_cache` kernel can be the performance bottle neck on the extremely long sequence case, In our private test branch, this kernel takes 2.12ms in a single layer when operating on a 57k prefix caching case, and that count to almost 30% time of this layer's time consumption.  ⏎  ⏎ In this PR, we rewrite this kernel in more hardware friendly way to accelerate this kernel include ⏎ - parallel on `num_tokens` fo …[truncated]

### L3-2d9ee28cab  (L3, 2025-11-24, sha 2d9ee28cab20, PR #29338)
TITLE: [CI/Test Fix] Fix CP tests on Blackwell (#29338)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/common.py (+0/-1)
LABELS: bug, ready, ci-failure
BODY: https://github.com/vllm-project/vllm/pull/28718 inadvertently undid part of https://github.com/vllm-project/vllm/pull/28404/

### L3-3cfa63ad99  (L3, 2025-11-24, sha 3cfa63ad9916, PR #29309)
TITLE: [XPU]fix Kimi-VL-A3B-thinking on xpu (#29309)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/moonvit.py (+14/-6)
LABELS: ready
BODY: ## Purpose ⏎ enable Kimi-VL-A3B-thinking text/image support on xpu. For image processing, we route to `flash_attn` backend and use `varlen_attention`. `torch.SDPA` path can't work due to OOM issue. ⏎ ## Test Plan ⏎ `VLLM_MLA_DISABLE=1 python examples/offline_inference/vision_language.py  --model-type kimi_vl` ⏎ ## Test Result ⏎ ``` ⏎ -------------------------------------------------- ⏎ The image is a photograph that falls under the category of architecture. It …[truncated]

### L3-dbc3d9991a  (L3, 2025-11-25, sha dbc3d9991ab0, PR #29337)
TITLE: [UX] Put CUDA attention backend selection log into one line (#29337)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+2/-4)
LABELS: ready, nvidia
BODY: ## Purpose ⏎  ⏎ Before: ⏎ ``` ⏎ (EngineCore_DP0 pid=3645964) INFO 11-24 20:03:41 [cuda.py:410] Valid backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'] ⏎ (EngineCore_DP0 pid=3645964) INFO 11-24 20:03:41 [cuda.py:419] Using FLASH_ATTN backend. ⏎ ``` ⏎  ⏎ After: ⏎ ``` ⏎ (EngineCore_DP0 pid=3642945) INFO 11-24 20:03:06 [cuda.py:416] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTEN …[truncated]

### L3-64deead719  (L3, 2025-11-25, sha 64deead719cc, PR #29371)
TITLE: [Bugfix] [ROCm] [UX]: revert Flex attention backend (#29371)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:confirmed-reverts, body_keyword
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+4/-0); tests/v1/attention/test_rocm_attention_backends_selection.py (+6/-0)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study revert record: partial_revert of PR(s) 26980 reason=premature_or_process
BODY: Description This PR restores the FLEX_ATTENTION backend selection logic that was accidentally removed in PR #26980.  ⏎  ⏎ Changes ⏎  ⏎ Re-added explicit check for AttentionBackendEnum.FLEX_ATTENTION in vllm/platforms/rocm.py. ⏎  ⏎ Added a corresponding unit test case in tests/v1/attention/test_rocm_attention_backends_selection.py to ensure it is correctly selected.

### L3-ef1f7030f0  (L3, 2025-11-25, sha ef1f7030f016, PR #29367)
TITLE: [ROCm][CI] Fix test_cudagraph_mode failure in AMD CI (#29367)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: tests/v1/attention/utils.py (+7/-0); tests/v1/cudagraph/test_cudagraph_mode.py (+42/-20); vllm/platforms/rocm.py (+2/-2)
LABELS: rocm, ready, v1, nvidia
BODY: We are seeing failures in the `tests/v1/cudagraph/test_cudagraph_mode.py` test in AMD CI after https://github.com/vllm-project/vllm/pull/26980 was merged. It fails because it reaches the error "V0 attention backends have been removed. Set VLLM_USE_V1=1 to select a supported backend" since the test tries to use the FlashAttn backend. I updated the test to test ROCm attention backends if current_platform.is_rocm(). ⏎  ⏎ After this PR, we see: ⏎ ``` ⏎ pytes …[truncated]

### L3-cb7214d8ea  (L3, 2025-11-25, sha cb7214d8eaa2, PR #28032)
TITLE: [ROCm][MLA] enable fp8 MLA decode on ROCm (#28032)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+11/-1); vllm/_aiter_ops.py (+10/-0)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎ AITER MLA decode supports fp8 inputs now, so add the related scale arguments accordingly. ⏎  ⏎ ## Test Plan ⏎ The accuracy and performance of Deepseek-v3 are verified on MI355. ⏎  ⏎ Server cmd: ⏎ ```bash ⏎ #!/bin/bash ⏎ export VLLM_USE_V1=1 ⏎ export SAFETENSORS_FAST_GPU=1 ⏎ export VLLM_ROCM_USE_AITER=1 ⏎ export VLLM_ROCM_USE_AITER_MOE=1 ⏎ export VLLM_USE_TRITON_FLASH_ATTN=0 ⏎ export NCCL_DEBUG=WARN ⏎ export VLLM_RPC_TIMEOUT=1800000 ⏎ export VLLM_ROCM_USE_AITER_MHA= …[truncated]

### L3-798e87db5c  (L3, 2025-11-25, sha 798e87db5c21, PR #29268)
TITLE: [Core] Generalize Encoder-Decoder `seq_lens` computation to avoid Whisper hardcoded logic   (#29268)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/attention/layers/cross_attention.py (+22/-22); vllm/v1/attention/backends/utils.py (+2/-1); vllm/v1/worker/gpu_model_runner.py (+27/-11)
LABELS: ready, v1
BODY: ### Overview ⏎ This PR generalizes current encoder-decoder `attn_metadata.seq_lens` computation in order to fix some hard-coded assumptions that are only valid for Whisper and based on static config values, rather than scheduler output. ⏎  ⏎ Currently, `GPUModelRunner._get_encoder_seq_lens` returns the number of scheduled encoder tokens, which is used to compute the slot_mapping needed when caching the cross attn kv cache.  ⏎ When no encoder tokens are s …[truncated]

### L3-794029f012  (L3, 2025-11-25, sha 794029f01206, PR #29137)
TITLE: [Feature]: Improve GGUF loading from HuggingFace user experience like repo_id:quant_type (#29137)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/test_gguf_download.py (+240/-0); tests/transformers_utils/test_utils.py (+146/-0); vllm/config/model.py (+12/-3); vllm/engine/arg_utils.py (+3/-3); vllm/model_executor/model_loader/gguf_loader.py (+22/-9); vllm/model_executor/model_loader/weight_utils.py (+46/-0); vllm/transformers_utils/config.py (+40/-11); vllm/transformers_utils/processor.py (+5/-5); vllm/transformers_utils/tokenizer.py (+12/-5); vllm/transformers_utils/utils.py (+53/-0)
LABELS: ready
ISSUES: #25182 [Feature]: improve GGUF loading from HuggingFace user experience
BODY: ## Purpose ⏎ Fixes #25182 ⏎ Improve GGUF loading from HuggingFace user experience like repo_id:quant_type ⏎  ⏎ ``` ⏎ vllm serve unsloth/Qwen3-0.6B-GGUF:IQ1_S --tokenizer Qwen/Qwen3-0.6B ⏎ ``` ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ vllm serve unsloth/Qwen3-0.6B-GGUF:IQ1_S --tokenizer Qwen/Qwen3-0.6B ⏎ pytest tests/models/test_gguf_download.py ⏎ pytest tests/transformers_utils/test_utils.py ⏎ ``` ⏎  ⏎ ## Test Result ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  …[truncated]

### L3-8d6a89dffd  (L3, 2025-11-25, sha 8d6a89dffd9e, PR #29250)
TITLE: [UX] Suppress gloo log spam (#29250)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/distributed/parallel_state.py (+3/-1); vllm/distributed/utils.py (+28/-26); vllm/utils/system_utils.py (+33/-0)
LABELS: ready, startup-ux
BODY: ## Purpose ⏎  ⏎ We basically always see a bunch of lines of `- [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0` when starting up vLLM for many weeks now. I think we should suppress this for now since it is often many lines of log space. We will not suppress the output if debug logs are enabled i.e. `VLLM_LOGGING_LEVEL=DEBUG` ⏎  ⏎ Thank you to @AlpinDale for letting me know the fix in `aphrodite-engine`: https:// …[truncated]

### L3-430dd4d9eb  (L3, 2025-11-26, sha 430dd4d9eb7e, PR #29342)
TITLE: [Attention] Remove imports from `vllm/attention/__init__.py` (#29342)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.abstract_interface, L3.flex_attention
FILES: vllm/attention/backends/abstract.py (+1/-1); vllm/attention/layer.py (+5/-2); docs/contributing/model/basic.md (+1/-1); tests/compile/test_fusion_attn.py (+2/-1); tests/compile/test_qk_norm_rope_fusion.py (+2/-1); tests/kernels/utils.py (+1/-1); tests/v1/worker/test_gpu_model_runner.py (+1/-1); tests/v1/worker/test_utils.py (+2/-2); vllm/attention/__init__.py (+0/-19); vllm/compilation/fusion_attn.py (+1/-1); (+86 more)
LABELS: documentation, tpu, ready, v1, llama, qwen, deepseek, gpt-oss, kv-connector, nvidia
BODY: ## Purpose ⏎ The `vllm/attention` module is coupled to the rest of the codebase, so the imports in `vllm/attention/__init__.py` frequently cause circular imports, leading to the use of `TYPE_CHECKING`. ⏎  ⏎ This issue is an obstacle to #26315 , which cannot use `TYPE_CHECKING` imports because `AttentionConfig` is a dataclass. ⏎  ⏎ This PR empties `__init__.py` and updates the rest of the codebase to use full-path imports. ⏎  ⏎ ## Test Plan ⏎ CI should suffice. R …[truncated]

### L3-d9d342d214  (L3, 2025-11-26, sha d9d342d214b8, PR #27457)
TITLE: [Performance][MLA][ROCm] Remove redundant D2D copy in deepseek (#27457)
SOURCES: path_core, path_integration+keyword, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L3.merge.triton_lse, L3.merge.cuda_lse, L3.mla.common_v1
FILES: csrc/attention/merge_attn_states.cu (+12/-15); csrc/ops.h (+1/-2); csrc/torch_bindings.cpp (+1/-2); vllm/attention/ops/triton_merge_attn_states.py (+17/-6); vllm/v1/attention/backends/mla/common.py (+18/-16)
LABELS: rocm, ready, v1, deepseek
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ We found there are two redundant D2D copy in deepseek, which can be removed by in-place write inside the kernel. ⏎  ⏎ Before ⏎ <img width="1451" height="51" alt="image" src="https://github.com/user-attachments/assets/9d6a14e1-2e20-47ab-9111-1967644894f8" /> ⏎  ⏎ <img width="1463" height="62" alt="image" src="https://github.com/user-attachments/assets/4521c8d0-3385-4cef-b3cb-426b3ad54927" /> ⏎  ⏎ After ⏎ <img width="1563" height="76" alt="image" src="h …[truncated]

### L3-56539cddac  (L3, 2025-11-26, sha 56539cddac9e, PR #28579)
TITLE: [Core] Refactor padding logic and pad for CUDA graphs before attention metadata building  (#28579)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/flashinfer.py (+1/-20); vllm/v1/attention/backends/utils.py (+4/-1); docs/design/cuda_graphs.md (+5/-3); tests/v1/cudagraph/test_cudagraph_dispatch.py (+31/-12); vllm/forward_context.py (+11/-7); vllm/v1/attention/backends/mamba_attn.py (+2/-0); vllm/v1/cudagraph_dispatcher.py (+66/-31); vllm/v1/worker/dp_utils.py (+16/-1); vllm/v1/worker/gpu_model_runner.py (+230/-202); vllm/v1/worker/gpu_worker.py (+35/-6)
LABELS: documentation, ready, v1, nvidia
ISSUES: #23789 [Attention]: Pad for cudagraphs before constructing attention metadata
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:be493e0b3c (fix PR 29578)
BODY: FIX https://github.com/vllm-project/vllm/issues/23789 ⏎  ⏎ The goal of this PR is to: ⏎  ⏎ 1) Pad for cudagraphs before building attention metadata; this will allow us to  ⏎                  - update to the latest FA3 (https://github.com/vllm-project/flash-attention/pull/82) ⏎                  - remove hacks like: https://github.com/vllm-project/FlashMLA/pull/3 ⏎                  - remove `pad_for_cudagraphs` from attention backends; this is done for FlashInfe …[truncated]

### L3-77740191de  (L3, 2025-11-26, sha 77740191de96, PR #29449)
TITLE: [Attention][Async] Eliminate `seq_lens_cpu` in FlashAttention metadata building with DCP > 1 (#29449)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/flash_attn.py (+14/-13); vllm/v1/attention/backends/utils.py (+4/-2)
LABELS: ready, v1
BODY: ## Purpose ⏎ Currently, when DCP > 1, FlashAttention uses the host-side `seq_lens_cpu` to compute the DCP KV context lens. This requires that the host and device be synchronized, which interferes with asynchronous speculative decoding. This PR modifies the logic to use only device-side tensors, and employs a safe upper bound for `max_seq_len`. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ vllm serve deepseek-ai/DeepSeek-R1 \ ⏎   --speculative-config '{"method": "mtp", "num_spec …[truncated]

### L3-43c5792592  (L3, 2025-11-27, sha 43c5792592d9, PR #29548)
TITLE: [ROCm][CI] Fix test_cpu_offloading for ROCm (#29548)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_offload/test_cpu_offloading.py (+2/-0)
LABELS: rocm, ready, v1
BODY: This PR fixes the test_cpu_offloading.py test. With the fix applied, we see: ⏎ ``` ⏎ pytest -v -s v1/kv_offload ⏎ ========================================================================================= warnings summary ========================================================================================= ⏎ <frozen importlib._bootstrap>:488 ⏎   <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute ⏎  …[truncated]

### L3-fc1d8be3dc  (L3, 2025-11-27, sha fc1d8be3dc97, PR #29540)
TITLE: [Attention] Update attention imports (#29540)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.abstract_interface, L3.platform.cuda_selection, L3.platform.rocm_selection, L3.flex_attention
FILES: vllm/attention/backends/abstract.py (+4/-7); vllm/attention/layers/chunked_local_attention.py (+1/-2); vllm/model_executor/layers/attention_layer_base.py (+2/-5); vllm/v1/attention/backends/cpu_attn.py (+0/-2); vllm/v1/attention/backends/flash_attn.py (+0/-2); vllm/v1/attention/backends/flex_attention.py (+0/-2); vllm/v1/attention/backends/utils.py (+5/-2); tests/v1/attention/test_rocm_attention_backends_selection.py (+3/-6); tests/v1/kv_connector/unit/test_backwards_compatibility.py (+3/-3); vllm/config/model.py (+1/-2); (+28 more)
LABELS: rocm, tpu, speculative-decoding, ready, v1, kv-connector, nvidia
BODY: ## Purpose ⏎ Now that #29342 has landed, many of the `TYPE_CHECKING` and lazy imports are no longer necessary. This PR updates them throughout the codebase ⏎  ⏎ ## Test Plan ⏎ Recommend to run all CI tests (`ready-run-all-tests`) ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-a5345bf49d  (L3, 2025-11-27, sha a5345bf49df7, PR #29426)
TITLE: [BugFix] Fix `plan` API Mismatch when using latest FlashInfer (#29426)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: docker/Dockerfile (+2/-2); requirements/cuda.txt (+1/-1); vllm/v1/attention/backends/flashinfer.py (+2/-1)
LABELS: ready, ci/build, v1, nvidia
BODY: Latest FlashInfer (`0.5.3`) changes `plan` API which breaks `fast_plan_decode`. ⏎ This PR updates FlashInfer to the latest version and fixes API breaking by adding another argument to `self._cached_module.plan` call, corresponding to `num_colocated_ctas=0`.

### L3-be493e0b3c  (L3, 2025-11-27, sha be493e0b3cfb, PR #29578)
TITLE: [BugFix] Fix new nightly failures (#29578)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+26/-0); vllm/v1/worker/gpu_model_runner.py (+11/-1)
LABELS: ready, v1, ready-run-all-tests
DEEP_STUDY: deep-study correctness case vllm:be493e0b3c: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=#28579
BODY: Fix nightly failures: ⏎ ``` ⏎ tests/models/language/pooling_mteb_test/test_jina.py::test_embed_models_mteb[model_info0] ⏎ tests/models/language/pooling/test_token_classification.py::test_modernbert_models[float-disham993/electrical-ner-ModernBERT-base] ⏎ tests/models/language/pooling_mteb_test/test_st_projector.py::test_embed_models_mteb[model_info1] ⏎ tests/models/multimodal/pooling/test_siglip.py::test_models_text[float-google/siglip-base-patch16-224] ⏎ te …[truncated]

### L3-0840abdd24  (L3, 2025-11-27, sha 0840abdd242b, PR #29582)
TITLE: [BugFix] Optional tokenizer argument when loading GGUF models (#29582)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/model.py (+8/-7); vllm/transformers_utils/gguf_utils.py (+42/-0); vllm/transformers_utils/tokenizer.py (+9/-1)
LABELS: ready
ISSUES: #29563 [Feature]: Improve tokenizer loading when loading GGUF models
BODY: ## Purpose ⏎ fixes: #29563 ⏎  ⏎ Fix correct to use optional `--tokenizer` argument when using GGUF model (local and remote). ⏎  ⏎ As-Is: `vllm serve <gguf_model> --tokenizer <tokenizer>` ⏎ To-Be: `vllm serve <gguf_model> (--tokenizer optional)` ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ vllm serve unsloth/Qwen3-0.6B-GGUF:IQ1_S ⏎ ``` ⏎  ⏎ ## Test Result ⏎ [details omitted] ⏎  ⏎ --- ⏎ [details omitted]

### L3-38658ec6f3  (L3, 2025-11-27, sha 38658ec6f3b3, PR #29614)
TITLE: [Bugfix][MM encoder] Fix ViT attention backend resolving for Turing GPU (#29614)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+9/-8)
LABELS: ready, nvidia
ISSUES: #29598 [Bug]: vllm serve HunYuanOCR error
BODY: ## Purpose ⏎ - Fix #29598 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-3cb32e5d6e  (L3, 2025-11-28, sha 3cb32e5d6e22, PR #28985)
TITLE: [Rocm] Set VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS default is disabled (#28985)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+3/-3)
LABELS: rocm, ready
BODY: ## Purpose ⏎ The  VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS feature is incompatible with many exciting features. e.g Kimi v2 thinking model , which shared experts dtype is bf16 but moe experts dtype is INT4, DeepSeek FP4 lacks corresponding adaptations will cause precision issue, incompatible with EPLB. Shared experts multi-stream overlap is potentially more optimal than fused shared experts. Enabled by default, this feature significantly increases …[truncated]

### L3-33b06a6f24  (L3, 2025-11-28, sha 33b06a6f24be, PR #29650)
TITLE: [Misc] Remove redundant attention var constants (#29650)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/selector.py (+4/-5); tests/kernels/attention/test_attention_selector.py (+9/-10); tests/kernels/attention/test_rocm_attention_selector.py (+5/-6); tests/kernels/utils.py (+0/-20); tests/models/quantization/test_fp8.py (+1/-2); vllm/model_executor/models/deepseek_eagle.py (+0/-3); vllm/utils/__init__.py (+0/-17)
LABELS: rocm, speculative-decoding, ready, deepseek
BODY: ## Purpose ⏎  ⏎ Remove unnecessary abstractions to simplify the code. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-f8151b66fa  (L3, 2025-11-28, sha f8151b66fa23, PR #29335)
TITLE: Revert "Supress verbose logs from model_hosting_container_standards (… (#29335)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: requirements/common.txt (+1/-1); vllm/entrypoints/openai/api_server.py (+0/-4)
LABELS: frontend, ready, ci/build
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 28949 reason=other
BODY: This reverts commit 67745d189fd981ee824bde35666a3737a962c031. (Supress verbose logs from model_hosting_container_standards #28949) ⏎  ⏎ ## Purpose ⏎  ⏎ we have already implement this log suppression in our plugin ⏎  ⏎ ## Test Plan ⏎  ⏎ rebuild the docker image and check the log ⏎  ⏎ ## Test Result ⏎ ``` ⏎ ubuntu@ip-172-31-28-149:/opt/dlami/nvme/vllm$ docker run   --runtime nvidia --gpus all  -v ~/.cache/huggingface:/root/.cache ⏎ /huggingface --env "HUGGING_FACE_HUB_TOKEN …[truncated]

### L3-460d8bbf2d  (L3, 2025-11-28, sha 460d8bbf2d19, PR #29471)
TITLE: Remove upstream fa checks (#29471)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.fa_utils
FILES: vllm/attention/layer.py (+7/-50); vllm/attention/ops/vit_attn_wrappers.py (+2/-8); vllm/attention/utils/fa_utils.py (+8/-0); vllm/model_executor/models/dots_ocr.py (+0/-8); vllm/model_executor/models/ernie45_vl.py (+0/-9); vllm/model_executor/models/glm4_1v.py (+1/-11); vllm/model_executor/models/keye.py (+0/-1); vllm/model_executor/models/paddleocr_vl.py (+0/-15); vllm/model_executor/models/qwen2_5_vl.py (+0/-18); vllm/model_executor/models/qwen2_vl.py (+0/-8); (+3 more)
LABELS: ready, qwen, nvidia
BODY: ## Purpose ⏎ According to https://github.com/vllm-project/vllm/pull/28763, the vllm flash attention has supported all headsize for vit module, thus removing the upstream flash-attn checks as they are no longer necessary. ⏎  ⏎ ## Test Plan ⏎ Use one of impacted model qwen3-vl-235B as example to start the server  ⏎ `vllm serve RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4 -tp 4 -dp 1 --mm-encoder-tp-mode data --enable-expert-parallel --async-scheduling --max-nu …[truncated]

### L3-9eec282cb5  (L3, 2025-11-28, sha 9eec282cb5a6, PR #29415)
TITLE: Guard FlashInfer sampler using the same check as FlashInfer attention backend (#29415)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/sample/ops/topk_topp_sampler.py (+10/-0)
LABELS: ready, v1
BODY: FlashInfer sampler is disabled by default and opt-in using `VLLM_USE_FLASHINFER_SAMPLER`. ⏎  ⏎ This PR adds a clear error when users on unsupported hardware try and enable this feature. ⏎  ⏎ Prior to this PR the error occured in FlashInfer and was obscure (see #27729). ⏎  ⏎ Supersedes #28379

### L3-7c1ed45848  (L3, 2025-11-28, sha 7c1ed4584889, PR #29241)
TITLE: [CI/Build]: make it possible to build with a free-threaded interpreter (#29241)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: cmake/utils.cmake (+7/-1); setup.py (+6/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ For Python 3.13/3.14, free-threaded Python does not support using the Limited C API and the Stable ABI. The purpose of this PR is to ensure that vLLM can be built from source under a free-threaded interpreter. It doesn't change anything for default (with-GIL) Python interpreters. ⏎  ⏎ See gh-28762 for more context on getting vLLM to work with free-threaded Python. ⏎  ⏎ Note that this same change is needed in [vllm_flash_attn](https://github.c …[truncated]

### L3-6f9d81d03b  (L3, 2025-11-28, sha 6f9d81d03b3b, PR #28043)
TITLE: [V0 deprecation] Clean up legacy paged attention helper functions (#28043)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.paged.python_wrapper
FILES: vllm/attention/ops/paged_attn.py (+0/-211); vllm/attention/ops/rocm_aiter_paged_attn.py (+0/-123)
LABELS: rocm, ready
BODY: ## Purpose ⏎ - After v0 deprecation, most of helper functions in `PagedAttention` are no longer used, so we can safely remove it. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-762a4a6ca9  (L3, 2025-11-28, sha 762a4a6ca902, PR #29706)
TITLE: [Frontend] Perform offline path replacement to `tokenizer` (#29706)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/entrypoints/offline_mode/test_offline_mode.py (+10/-0); vllm/engine/arg_utils.py (+17/-6)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ If `tokenizer` is a Hugging Face model, vLLM attempts to access Hugging Face even if the tokenizer is already available offline. ⏎ It prevents specifying a Hugging Face model as `tokenizer` on the offline mode (i.e. `HF_HUB_OFFLINE` is true). ⏎  ⏎ It also tweaks when the offline mode path replacement log is emitted in a separate commit: only when model/tokenizer value changes.  This is because it's not helpful to log the replacement when th …[truncated]

### L3-e23f665d83  (L3, 2025-11-28, sha e23f665d835a, PR #29698)
TITLE: [BugFix] Fix DBO failing with TypeError: 'NoneType' object is not iterable (#29698)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+1/-3); tests/v1/distributed/test_dbo.py (+0/-1); vllm/v1/worker/dp_utils.py (+6/-3)
LABELS: ready, v1
BODY: https://github.com/vllm-project/vllm/pull/28579 broke DBO but was hidden by existing test failures on the nightly (https://github.com/vllm-project/vllm/pull/28893, with those fixed this now showed up) ⏎  ⏎ Test Plan: ⏎  ⏎ ``` ⏎ tests/v1/distributed/test_dbo.py ⏎ ``` ⏎  ⏎ Fails on Main, Passes on this PR ⏎  ⏎ NOTE: using `padded_second_ubatch_slice` instead of `ubatch_slices[1].request_slice` for requests was a bug that pre-dated https://github.com/vllm-project/vllm/ …[truncated]

### L3-9726e64530  (L3, 2025-11-29, sha 9726e64530e3, PR #28840)
TITLE: bugfix: correct attn output with base 2 or e (#28840)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1
FILES: vllm/attention/ops/common.py (+32/-10); vllm/v1/attention/backends/flashinfer.py (+9/-2); vllm/v1/attention/backends/mla/common.py (+6/-1)
LABELS: ready, v1, nvidia
BODY: flashinfer attention use 2 as base of lse instead of e, see https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/mla.cuh#L400 ⏎  ⏎ ## Purpose ⏎  ⏎ correct attn output with proper factor when using context parallel.  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-e1464c3a08  (L3, 2025-11-30, sha e1464c3a0861, PR #29732)
TITLE: [Quantization] Enable compressed-tensors AWQ for Turing GPU (#29732)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py (+2/-2)
LABELS: ready
ISSUES: #29707 [Usage]: Workaround to run model on GPUs with Compute Capability < 8.0?
BODY: ## Purpose ⏎ - Fix #29707 ⏎ - Compressed-tensors AWQ quantization should be able to run on Turing GPU, verified on Tesla T4 GPU ⏎  ⏎ ## Test Plan ⏎ Tested with `cpatonn/Qwen3-VL-32B-Instruct-AWQ-4bit` on Tesla T4 GPU: ⏎ ``` ⏎ python examples/offline_inference/vision_language.py -m qwen3_vl ⏎ ``` ⏎  ⏎ ## Test Result ⏎ ``` ⏎ (EngineCore_DP0 pid=9615) INFO 11-29 16:05:38 [core.py:93] Initializing a V1 LLM engine (v0.11.2.dev393+g39e63dec7) with config: model='cpatonn/Qwen3 …[truncated]

### L3-82c795d6f2  (L3, 2025-11-30, sha 82c795d6f28e, PR #29734)
TITLE: Fix AttributeError about _use_fi_prefill (#29734)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+1/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ After https://github.com/vllm-project/vllm/pull/28840 , the nightly [Distributed Tests (H200)](https://buildkite.com/vllm/ci/builds/41171/steps/table?jid=019ace6a-57d7-4bc7-a3aa-c99174395dbd) and [Distributed Tests (B200)](https://buildkite.com/vllm/ci/builds/41171/steps/table?jid=019ace6a-57db-4e0b-9528-e04a0af07b6a) are failing due to  ⏎ ``` ⏎ AttributeError: 'FlashAttnMLAImpl' object has no attribute '_use_fi_prefill' ⏎ AttributeError: ' …[truncated]

### L3-39d28108f4  (L3, 2025-11-30, sha 39d28108f46a, PR #29004)
TITLE: [Feat] Support non-gated activations in NVFP4 modelopt path (#29004)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer_moe.py (+17/-7); tests/kernels/moe/utils.py (+10/-1); tests/kernels/utils.py (+7/-1); vllm/model_executor/layers/fused_moe/layer.py (+9/-3); vllm/model_executor/layers/quantization/modelopt.py (+55/-10)
LABELS: ready, nvidia
BODY: ## Purpose ⏎ For upcoming Nvidia releases, we require support for running non-gated activations for NVFP4 modelopt checkpoints. ⏎ This PR enables this by: ⏎ 1. Modifying the sizes of the modelopt nvfp4 MoE weights according to the value of `self.moe.is_act_and_mul` ⏎ 2. Calling the flashinfer cutlass fused moe kernel, which supports this feature. Since this is only supported in this path, running such models requires providing the following environment v …[truncated]

### L3-8c363ed666  (L3, 2025-11-30, sha 8c363ed6663f, PR #29234)
TITLE: [ROCm][Attention] Sliding window support for `AiterFlashAttentionBackend` (#29234)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+224/-49)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎ This PR add the support for sliding windows to `AiterFlashAttentionBackend` ⏎ ## Test Plan ⏎ gsm8k on c4ai-command-r7b ⏎ ## Test Result ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7726|±  |0.0115| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.7672|±  |0.0116| ⏎ ``` ⏎ --- ⏎ [ …[truncated]

### L3-66b5840287  (L3, 2025-11-30, sha 66b584028779, PR #28783)
TITLE: [Bugfix][sleepmode][fp8 kv cache]: Fix FP8 KV cache + sleep(level=2) gibberish output (#28783)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/basic_correctness/test_cumem.py (+32/-1); tests/utils.py (+7/-0); vllm/v1/worker/gpu_model_runner.py (+45/-1); vllm/v1/worker/gpu_worker.py (+10/-0)
LABELS: ready, v1
ISSUES: #25800 [Bug]: FP8 KV cache + sleep(level=2) leads to gibberish output, but level=1 is fine
BODY: This PR fixes #25800 where waking from sleep(level=2) with kv_cache_dtype="fp8" results in gibberish output. ⏎  ⏎ ### Bug Cause ⏎  ⏎ When the engine enters level 2 sleep (llm.sleep(level=2)), all VRAM is discarded, including the FP8 scaling factors for the KV cache tensors. ⏎ The existing wake_up(tags=["kv_cache"]) logic successfully re-allocates the VRAM for the KV cache, but it failed to re-initialize the FP8 scaling factors (._scale). This left the scal …[truncated]

### L3-b95db244ee  (L3, 2025-12-01, sha b95db244ee2d, PR #26015)
TITLE: [v1] Add real sliding window calculation to FlexAttention direct BlockMask building (#26015)
SOURCES: path_core
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+27/-6); tests/v1/attention/test_attention_backends.py (+11/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎ - Following PR for https://github.com/vllm-project/vllm/pull/24089#pullrequestreview-3249376146 ⏎ - Support real sliding window when using direct build. ⏎ - To achieve real sliding window calculation, we excluded out-of-window page kv block from BlockMask building. ⏎  ⏎ For `sliding_window=2047` and `block_size=128`, the `kv_num_blocks` used to construct BlockMask reduced from 128 to 68 (batch_size=8, prefill_tokens=4096). ⏎  ⏎ - Benchmark with ni …[truncated]

### L3-e2fbfc955e  (L3, 2025-12-02, sha e2fbfc955e1b, PR #29827)
TITLE: [CI][AMD] spec_decode:eagle skip FLASH_ATTN for deepseek on ROCm (#29827)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/e2e/test_spec_decode.py (+4/-1)
LABELS: rocm, ready, v1, deepseek
BODY: This PR skips FLASH_ATTN for deepseek model as it is not supported on ROCm.  ⏎ [CI][ROCM] pytest:   ⏎ `pytest -s -v test_spec_decode.py::test_eagle_correctness -k deepseek_eagle`

### L3-d8c6210eea  (L3, 2025-12-02, sha d8c6210eeaa7, PR #29757)
TITLE: Add Mistral Large 3 and Ministral 3 (#29757)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mla.py (+4/-0); docs/models/supported_models.md (+3/-2); tests/models/registry.py (+14/-0); tests/tokenizers_/test_mistral.py (+151/-7); vllm/config/speculative.py (+4/-0); vllm/entrypoints/openai/tool_parsers/mistral_tool_parser.py (+1/-1); vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8.json (+146/-0); vllm/model_executor/layers/rotary_embedding/__init__.py (+1/-1); vllm/model_executor/models/deepseek_v2.py (+59/-7); vllm/model_executor/models/mistral_large_3.py (+63/-0); (+6 more)
LABELS: documentation, new-model, frontend, speculative-decoding, ready, v1, tool-calling, deepseek
BODY: ## Purpose ⏎  ⏎ This PR adds support to Mistral-Large-3 and Ministral-3. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-1d93f11675  (L3, 2025-12-02, sha 1d93f116754f, PR #29352)
TITLE: [Attention][CUDAGraph] Remove CG padding from attention backends (#29352)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mamba/mamba_mixer.py (+8/-9); vllm/v1/attention/backends/gdn_attn.py (+5/-17); vllm/v1/attention/backends/mamba1_attn.py (+3/-9); vllm/v1/attention/backends/mamba2_attn.py (+3/-9); vllm/v1/attention/backends/short_conv_attn.py (+1/-2)
LABELS: ready, ci/build, v1, nvidia
BODY: ## Purpose ⏎ #28579 refactors the logic for cudagraph padding so that it happens prior to attention metadata building. This is a follow-up which removes remaining unnecessary padding calls from miscellaneous backends. ⏎  ⏎ ## Test Plan ⏎ `pytest tests/models/language/generation -m hybrid_model` ⏎  ⏎ ## Test Result ⏎ Should pass in CI: Language Models Tests (Hybrid) ⏎  ⏎ --- ⏎ [details omitted]

### L3-c719c40540  (L3, 2025-12-03, sha c719c40540a8, PR #29631)
TITLE: [Bugfix] Defunctionalize TRTLLM AR+Norm op for avoiding extra clone kernel before it (#29631)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/compilation/fix_functionalization.py (+12/-0); vllm/compilation/fx_utils.py (+2/-2)
LABELS: ready
ISSUES: #29181 [Bug]: Got additional triton kernel when using native rms_norm op matching with all_reduce+rms_norm fusion
BODY: ## Purpose ⏎  ⏎ Resolved #29181. ⏎ cc @ProExpertProg @nvpohanh  ⏎  ⏎ ## Test Plan && Test Result ⏎  ⏎ PR: ⏎ ``` ⏎ fmhaSm100fKernel_QkvE4m3OBfloat16H64PagedKvDenseP16MultiCtasKvCgaVarSeqQ8Kv128     7.776 μs ⏎ nvjet_tst_24x64_64x16_4x1_v_bz_bias_TNN                                            6.911 μs ⏎ void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport  6.304 μs ⏎ ``` ⏎  ⏎ main: ⏎ ``` ⏎ fmhaSm100fKernel_QkvE4m3OBfloat16H64PagedKvDenseP16MultiCtasKvC …[truncated]

### L3-f5d3d93c40  (L3, 2025-12-03, sha f5d3d93c4041, PR #29452)
TITLE: [docker] Build CUDA kernels in separate Docker stage for faster rebuilds (#29452)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen
FILES: docker/Dockerfile (+58/-8); setup.py (+11/-3); docs/assets/contributing/dockerfile-stages-dependency.png (+0/-0); vllm/envs.py (+5/-0)
LABELS: documentation, ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ Saves ~30 minutes on Docker rebuilds when changing Python code by building CUDA kernels in a separate cacheable stage. ⏎  ⏎ **Context:** 93% of commits in the last 30 days (678/920) modified Python files while only 5.5% (51/920) changed kernel code. Yet every Python change currently forces a 30-minute kernel recompilation in ⏎   Docker builds. ⏎  ⏎ ## Solution ⏎ Splits the Dockerfile into two stages: ⏎ 1. **csrc-build** - Compiles CUDA kernels (slo …[truncated]

### L3-afe9eb408e  (L3, 2025-12-03, sha afe9eb408ee1, PR #29960)
TITLE: [Bugfix] Fix flashinfer ar+norm kernel not available issue (#29960)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/compilation/fix_functionalization.py (+2/-1)
LABELS: ready
BODY: ## Purpose ⏎ Fix #29631 ⏎  ⏎ ``` ⏎ (EngineCore_DP0 pid=12345)   File "/src/vllm/vllm/compilation/fix_functionalization.py", line 108, in __call__ ⏎ (EngineCore_DP0 pid=12345)     == torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm.default ⏎ (EngineCore_DP0 pid=12345)        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ (EngineCore_DP0 pid=12345)   File "/home/user/.venvs/vllm-rocm/lib/python3.13/site-packages/torch/_ops.py", line 1347, in __geta …[truncated]

### L3-3f1b03739a  (L3, 2025-12-04, sha 3f1b03739ae1, PR #29974)
TITLE: [ROCm] [Bugfix] `compute_attn_mask_seqlen` for qwen3 omni (#29974)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_omni_moe_thinker.py (+4/-1)
LABELS: rocm, ready, qwen
BODY: ## Purpose ⏎  ⏎ This is a bugfix for qwen3-omni model when using AITER Flash Attention ⏎  ⏎ ## Test Plan ⏎  ⏎ Evaluate the qwen3-omni chartqa ⏎  ⏎ ## Test Result ⏎ ``` ⏎ For detailed information on this command, run: ⏎   run.py eval_vllm --model_name Qwen/Qwen3-Omni-30B-A3B-Instruct --url http://0.0.0.0:7001 --output_dir ./chartqa --eval_name chartqa - --help ⏎ ================================================================================ ⏎ Metrics: ⏎ { ⏎     "explicit_prom …[truncated]

### L3-b8a6ae4158  (L3, 2025-12-04, sha b8a6ae415859, PR #30005)
TITLE: [ROCm] add fallback for aiter fp8 decode mla (#30005)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/_aiter_ops.py (+33/-4)
LABELS: rocm, ready
BODY: ## Purpose ⏎ Our AITER version is a bit behind. Adding fallback for https://github.com/vllm-project/vllm/pull/28032. ⏎ ``` ⏎ [0;36m(Worker_TP0 pid=4049)[0;0m ERROR 12-03 02:27:03 [multiproc_executor.py:822]     unified_mla_attention_with_output = torch.ops.vllm.unified_mla_attention_with_output(q, out_2, key_rot_1, output_3, 'model.layers.0.self_attn.attn');  q = out_2 = key_rot_1 = output_3 = unified_mla_attention_with_output = None ⏎ [0;36m(Worker_TP2  …[truncated]

### L3-f2f4cea6cc  (L3, 2025-12-04, sha f2f4cea6ccaa, PR #29995)
TITLE: [CI/Build][AMD] Skip test on test_hybrid_attention_mamba_tensor_shapes on ROCm, requires FLASHINFER (#29995)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_model_runner.py (+4/-0)
LABELS: rocm, ready, v1
BODY: This PR skips `test_hybrid_attention_mamba_tensor_shapes`  in `test_gpu_model_runner.py` since it uses `FLASHINFER `backend which is not supported on ROCm.

### L3-e96a6a6dca  (L3, 2025-12-04, sha e96a6a6dca93, PR #30013)
TITLE: [ROCm][CI][Bugfix] Fixing the `Multi-Modal Models Test (Extended) 1` group (#30013)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+13/-1); .buildkite/test-amd.yaml (+4/-2); tests/models/multimodal/generation/conftest.py (+16/-0); tests/models/multimodal/generation/test_common.py (+10/-2); tests/models/multimodal/generation/test_granite_speech.py (+13/-2); tests/models/multimodal/generation/test_pixtral.py (+10/-0); tests/models/multimodal/generation/vlm_utils/custom_inputs.py (+1/-1); tests/models/multimodal/generation/vlm_utils/model_utils.py (+44/-1); tests/models/multimodal/pooling/conftest.py (+24/-0); tests/models/registry.py (+4/-0)
LABELS: rocm, ready, ci/build, v1, multi-modality
BODY: This PR is connected to: https://github.com/vllm-project/vllm/pull/29984 ⏎ Still, it addresses a different test group. Adds ROCm-specific configurations for multimodal model tests to improve compatibility and accuracy on ROCm: ⏎  ⏎ **Changes:** ⏎  ⏎ 1. **`tests/models/multimodal/pooling/conftest.py`** - New pytest configuration that sets `VLLM_ATTENTION_BACKEND=FLEX_ATTENTION` for SigLIP tests on ROCm, addressing encoder self-attention compatibility. ⏎  ⏎ 2. * …[truncated]

### L3-d698bb382d  (L3, 2025-12-05, sha d698bb382db9, PR #29487)
TITLE: [Bugfix] Correct num_q_heads on DCP for Flashinfer backends  (#29487)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+2/-3)
LABELS: ready, v1, nvidia
BODY: ## Purpose ⏎ Fix num_q_heads on DCP for Flashinfer backends ⏎ In current DCP code for Flashinfer, the `num_qo_heads` is repeatedly multiplied by `dcp_world_size`. ⏎  ⏎ ## Test Result ⏎ Although it does not affect the precision of inference, sometimes it causes share memory issues. After fixing this bug, the issue can be mitigated.

### L3-66e674cdd5  (L3, 2025-12-05, sha 66e674cdd549, PR #26315)
TITLE: [Attention][UX][1/N] Add AttentionConfig and change attention env vars to CLI arguments (#26315)
SOURCES: path_core, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fa_utils, L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.mla.common_v1, L3.mla.flashattn, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.platform.cuda_selection
FILES: vllm/attention/backends/abstract.py (+10/-16); vllm/attention/layer.py (+2/-2); vllm/attention/selector.py (+9/-128); vllm/attention/utils/fa_utils.py (+6/-5); vllm/utils/flashinfer.py (+14/-23); vllm/v1/attention/backends/flash_attn.py (+5/-4); vllm/v1/attention/backends/flashinfer.py (+14/-12); vllm/v1/attention/backends/mla/common.py (+14/-5); vllm/v1/attention/backends/mla/flashattn_mla.py (+3/-2); vllm/v1/attention/backends/rocm_attn.py (+1/-1); (+12 more)
LABELS: rocm, ready, v1, kv-connector, nvidia, ready-run-all-tests
BODY: ## Purpose ⏎ As part of the effort to reduce env variables (#25700), this PR introduces `AttentionConfig` and the `--attention-config` CLI argument group. The following environment variables are converted to CLI arguments: ⏎  ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND                        -> --attention-config.backend ⏎ VLLM_FLASH_ATTN_VERSION                       -> --attention-config.flash_attn_version ⏎ VLLM_V1_USE_PREFILL_DECODE_ATTENTION          -> --attention- …[truncated]

### L3-3628bcaaf2  (L3, 2025-12-05, sha 3628bcaaf229, PR #29775)
TITLE: [ROCm][MXFP4] Infer w4a4 quant method in rocm aiter fused moe (#29775)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/config.py (+4/-0); vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py (+2/-2)
LABELS: rocm, ready
BODY: ## Purpose ⏎ The quant recipe that models use is with bf16/fp16 activation dynamic quantized + fp4 weight static quantized (aka w4a16). This PR make this recipe as the default path and we successfully verify it on DeepSeek-R1 with MXFP4, while the code w/o this modification would trigger quant kernel jitting failure. ⏎  ⏎ ## Test Plan ⏎  ⏎ Server launching command ⏎  ⏎ ```bash ⏎ export VLLM_USE_V1=1 ⏎ # export VLLM_LOGGING_LEVEL=DEBUG ⏎ export VLLM_RPC_TIMEOUT=18000 …[truncated]

### L3-962d703818  (L3, 2025-12-05, sha 962d703818c0, PR #29926)
TITLE: [Bugfix][llama4_eagle] Fix missing 'lm_head' attribute (#29926)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/e2e/test_spec_decode.py (+5/-1); vllm/model_executor/models/llama4_eagle.py (+11/-2)
LABELS: rocm, speculative-decoding, ready, v1, llama
BODY: This PR: ⏎ 1. Fixes llama4_eagle model by adding the missing `lm_head` attribute to the model class. ⏎ 2. Fixes `test_spec_decode.py::test_eagle_correctness[FLASH_ATTN-llama4_eagle]` for ROCm CI ⏎  ⏎ ### Explanation for 1. ⏎  ⏎ The PR resolves the following error: `AttributeError: 'EagleLlama4ForCausalLM' object has no attribute 'lm_head' ` which occurs on both H100 and MI300. ⏎ #### Run cmd: ⏎ `vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct -tp 4 --specul …[truncated]

### L3-e3fbb6f152  (L3, 2025-12-05, sha e3fbb6f152fe, PR #30093)
TITLE: fix#30092 Kimi-Linear model loading failure with missing indexer_rotary_emb (#30093)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mla.py (+1/-1)
LABELS: ready
ISSUES: #30092 [Bug]: Kimi-Linear-48B-A3B-Instruct fails to load due to 88f5b19f0bc681c016eaaa17502d3bb4e2b59b51
BODY: ## Purpose ⏎ fix #30092  ⏎ ## Test Plan ⏎ vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \ ⏎   --port 8000 \ ⏎   --tensor-parallel-size 8 \ ⏎   --max-model-len 1048576 \ ⏎   --trust-remote-code \ ⏎   --gpu-memory-utilization 0.95   ⏎ ## Test Result ⏎ <img width="1363" height="319" alt="image" src="https://github.com/user-attachments/assets/d4b8e5a8-ab70-45a4-a537-906f02e748d7" /> ⏎  ⏎ --- ⏎ [details omitted]

### L3-b12f4a9830  (L3, 2025-12-05, sha b12f4a983077, PR #29985)
TITLE: [CI/Build][AMD] Use ROCM_ATTN instead of FLASH_ATTN test for test_register_kv_caches for ROCm and update test for TRITON_ATTN (#29985)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_nixl_connector.py (+37/-11)
LABELS: rocm, ready, v1, kv-connector
BODY: This test skips `FLASH_ATTN ` for `test_register_kv_caches `since it is not supported on ROCm. ⏎  ⏎ This also  updates test_register_kv_caches for TRITON_ATTN which was failing with the following error: ⏎  ⏎ ``` ⏎             # Verify get_reg_descs was called with caches_data ⏎             assert mock_wrapper_instance.get_reg_descs.called ⏎             caches_data, _ = mock_wrapper_instance.get_reg_descs.call_args[0] ⏎ >           assert len(caches_data) == 4 ⏎ E  …[truncated]

### L3-d6aeaddf4a  (L3, 2025-12-06, sha d6aeaddf4a62, PR #30051)
TITLE: [bugfix] fix type[AttentionBackend] bug in kv_connector_base_v1 (#30051)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/base.py (+1/-1)
LABELS: ready, kv-connector
BODY: ## Purpose ⏎ fix the bug in KVConnectorBase_v1, change type[AttentionBackend] to  type["AttentionBackend"]

### L3-0044c4038c  (L3, 2025-12-07, sha 0044c4038c57, PR #30195)
TITLE: [BugFix][DeepSeek-V3.2] Fix backend selection logic for Blackwell (#30195)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+2/-2)
LABELS: ready, deepseek, nvidia
BODY: Fix ⏎ ``` ⏎ ValueError: Selected backend AttentionBackendEnum.CUTLASS_MLA is not valid for this configuration. Reason: ['kv_cache_dtype not supported', 'sparse not supported'] ⏎ ``` ⏎  ⏎ When running ⏎ ``` ⏎ vllm serve deepseek-ai/DeepSeek-V3.2  ⏎ ``` ⏎  ⏎ on 8xB200

### L3-b952f4d3c3  (L3, 2025-12-07, sha b952f4d3c310, PR #27938)
TITLE: [v1] Add PrefixLM support to FlexAttention backend (#27938)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.dispatch.selector, L3.dispatch.abstract_interface, L3.platform.cuda_selection, L3.platform.rocm_selection, L3.flex_attention
FILES: vllm/attention/backends/abstract.py (+9/-0); vllm/attention/layer.py (+5/-0); vllm/attention/selector.py (+5/-0); vllm/config/model.py (+14/-0); vllm/platforms/cpu.py (+1/-0); vllm/platforms/cuda.py (+19/-0); vllm/platforms/interface.py (+1/-0); vllm/platforms/rocm.py (+1/-0); vllm/platforms/tpu.py (+3/-2); vllm/platforms/xpu.py (+2/-1); (+6 more)
LABELS: documentation, rocm, tpu, ready, v1, multi-modality, nvidia
BODY: ## Purpose ⏎ - Currently, there is no attention backend supports image-bidirectional attention in vLLM, so Gemma3 and paligemma can't generate correct outputs. And models like moondream are blocked due to missing attention backend support. ⏎ - This PR adds image-bidirectional support to FlexAttention backend to fill the void. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest -s -v tests/models/multimodal/generation/test_common.py -k gemma3 ⏎ ``` ⏎  ⏎ ## Test Result ⏎ vllm (both nati …[truncated]

### L3-c6df05ebb4  (L3, 2025-12-08, sha c6df05ebb499, PR #29773)
TITLE: [ROCm] [Fused Moe EP] Use binary expert mask for aiter fused moe kernel (#29773)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/layer.py (+4/-0); vllm/model_executor/layers/quantization/quark/quark_moe.py (+1/-0)
LABELS: rocm, ready
BODY: ## Purpose ⏎  ⏎ Quark Fused Moe meets acc issue in DeepSeek-R1 Expert Parallelism. This PR fixes it through setting the right expert mask with format required by aiter kernel. The aiter kernel requires the mask to be a bitmask. 1 means the expert is on current rank, while 0 is not. ⏎  ⏎ ## Test Plan ⏎  ⏎ Server Launch ⏎  ⏎ ```bash ⏎ export VLLM_USE_V1=1 ⏎ # export VLLM_LOGGING_LEVEL=DEBUG ⏎ export VLLM_RPC_TIMEOUT=1800000 ⏎ export VLLM_USE_TRITON_FLASH_ATTN=1 ⏎ export VLL …[truncated]

### L3-bcb6f5947f  (L3, 2025-12-08, sha bcb6f5947f8a, PR #30232)
TITLE: [Perf] Remove sync point in vit torch sdpa attn backend (#30232)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/vit_attn_wrappers.py (+6/-6); vllm/model_executor/models/ernie45_vl.py (+6/-6); vllm/model_executor/models/glm4_1v.py (+6/-6); vllm/model_executor/models/qwen2_vl.py (+6/-6)
LABELS: ready, qwen
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ When we use torch sdpa vit aten backend,  the way to get q_i, k_i, v_i introduce sync point inside the loop。 ⏎ Here is the profiler. ⏎ <img width="1666" height="681" alt="image" src="https://github.com/user-attachments/assets/0af395d6-9b16-494a-8916-c2ebb35a7b81" /> ⏎ [sdpa_before.pt.trace.json.gz](https://github.com/user-attachments/files/24023457/sdpa_before.pt.trace.json.gz) ⏎  ⏎ So I remove the q_i, k_i, o_i outside of the loop, atfer that c …[truncated]

### L3-1fb632fdb6  (L3, 2025-12-08, sha 1fb632fdb6b6, PR #29795)
TITLE: [Perf] Improve fp8 quant in mla; replace ReduceSum with ReduceScatterSum (#29795)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+21/-12); vllm/distributed/device_communicators/cuda_communicator.py (+1/-1)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ - Choose `ReduceScatterSum` over `ReduceSum` when sizes are the same ⏎    ⏎   <img width="1022" height="210" alt="image" src="https://github.com/user-attachments/assets/4f25ca64-0321-4e38-8dd7-a1b6736652bd" /> ⏎  ⏎   <img width="1066" height="169" alt="image" src="https://github.com/user-attachments/assets/d6983667-bf05-4418-8711-e079d8590603" /> ⏎  ⏎ - Avoid calling fp8 quant kernel twice when using fp8 attention by creating a stage buffer for ` …[truncated]

### L3-6af70e11a0  (L3, 2025-12-08, sha 6af70e11a0a3, PR #29916)
TITLE: [ROCm][CI] Fix test_max_len.py for Rocm (#29916)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/basic_correctness/test_basic_correctness.py (+4/-1); tests/utils.py (+2/-2); tests/v1/e2e/test_spec_decode.py (+2/-2); tests/v1/spec_decode/test_eagle.py (+6/-2); tests/v1/spec_decode/test_max_len.py (+1/-1)
LABELS: rocm, speculative-decoding, v1
BODY: Use ROCM_AITER_FA backend for flash attn on rocm.

### L3-d9417096d1  (L3, 2025-12-08, sha d9417096d134, PR #29125)
TITLE: [Feature] Batch invariant: Enable `TRITON_MLA` without prefix-caching (#29125)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+36/-0); tests/v1/determinism/test_batch_invariance.py (+1/-5); tests/v1/determinism/test_online_batch_invariance.py (+4/-1); tests/v1/determinism/utils.py (+1/-0); vllm/model_executor/layers/batch_invariant.py (+1/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Enable `TRITON_MLA` without prefix-caching ⏎  ⏎ It involves very large change if we want to support the kernel with prefix-caching, I will have a follow up issue for this. ⏎  ⏎ ## Test ⏎  ⏎ ### Other unit tests ⏎  ⏎ ```bash ⏎ (wentao) wentao@dgxB200-09:~/vllm-source/tests/v1/determinism$ pytest ⏎ ==================================== test session starts ==================================== ⏎ platform linux -- Python 3.12.3, pytest-9.0.1, pluggy-1.6.0 ⏎ rootdi …[truncated]

### L3-9d6235ca9a  (L3, 2025-12-09, sha 9d6235ca9a36, PR #29936)
TITLE: [moe] Allow disabling DP chunking (#29936)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+4/-0); vllm/model_executor/layers/fused_moe/layer.py (+1/-1)
LABELS: ready
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ We found that disabling dp chunking when using flashinfer cutlass backend provides better throughput performance on GB200 with nvlink. ⏎ This PR adds an env var to force disabling it. VLLM_ENABLE_MOE_DP_CHUNK is true by default, not changing current behavior. ⏎  ⏎ ## Test Plan ⏎  ⏎ use VLLM_ENABLE_MOE_DP_CHUNK=0 in DPEP test ⏎  ⏎ ## Test Result ⏎  ⏎ * accuracy not impacted ⏎ * throughput performance is better ⏎  ⏎ --- ⏎ [details omitted]

### L3-aed846917f  (L3, 2025-12-09, sha aed846917fee, PR #29644)
TITLE: [Attention] Make `split_decodes_and_prefills(..., require_uniform=True)` support padding (#29644)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+7/-3); tests/v1/attention/test_attention_splitting.py (+24/-1)
LABELS: ready, v1
BODY: with https://github.com/vllm-project/vllm/pull/28579 we pad attention metadata before building; in the case of a uniform decode request we want to make sure that num_decodes matches the cudagraph size so that attention schedulers receive the same batch size as the graph was captured with. Currently we have work around this in all the existing attention backends (e.g. https://github.com/vllm-project/FlashMLA/pull/3) since we used to pad for attent …[truncated]

### L3-aeb82b1930  (L3, 2025-12-09, sha aeb82b193045, PR #30306)
TITLE: [CI] Fix Flaky test_eagle_max_len Test (#30306)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/spec_decode/test_max_len.py (+1/-1)
LABELS: rocm, speculative-decoding, ready, v1
BODY: After https://github.com/vllm-project/vllm/pull/29916 was merged, which fixed the original failure in `tests/v1/spec_decode/test_max_len.py`, we are now noticing that the test is flaky on ROCm. Roughly 15% of the time, I am seeing this when running `pytest -v -s v1/spec_decode/test_max_len.py::test_eagle_max_len[ROCM_AITER_FA-3]`: ⏎  ⏎ ``` ⏎ AssertionError: This test is only meaningful if the output is longer than the eagle max length ⏎ ... ⏎ ============= …[truncated]

### L3-67475a6e81  (L3, 2025-12-09, sha 67475a6e81ab, PR #30309)
TITLE: [DCP][Bugfix][CI] Fix accuracy issue of DCP when using FLASH_ATTN_MLA (#30309)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.flashattn
FILES: vllm/v1/attention/backends/mla/flashattn_mla.py (+2/-1); tests/distributed/test_context_parallel.py (+4/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎ https://github.com/vllm-project/vllm/pull/25049 add MTP support for DCP with FA3, but this only works when cp_kv_cache_interleave_size=1. ⏎  ⏎ For FA3 backend, we should check `cp_kv_cache_interleave_size` and set `supports_dcp_with_varlen` accordingly. ⏎  ⏎ ## Test Plan ⏎ ```shell ⏎ vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat/ --gpu-memory-utilization 0.9 --tensor-parallel-size 4 --decode-context-parallel-size 4 --cp-kv-cache-interleave-size 64 ⏎  …[truncated]

### L3-5dcd593baf  (L3, 2025-12-09, sha 5dcd593baf9c, PR #30018)
TITLE: [Feature] Batch-Invariant Support for FA2 and LoRA (#30018)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/determinism/test_batch_invariance.py (+10/-0); tests/v1/determinism/utils.py (+8/-2); vllm/model_executor/layers/batch_invariant.py (+5/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎ This PR introduces batch-invariant operations with support for both FlashAttention-2 (FA2) and LoRA, ensuring deterministic inference under batch reordering, concurrent access, and multi-tenant scenarios. This PR follows  #27433. ⏎ ## Test Plan ⏎ 1. Install FlashAttention-2 with batch-invariant support ⏎ Use FA2 PR https://github.com/vllm-project/flash-attention/pull/110 ⏎ Install by: ⏎     - Installing directly from that PR, or ⏎     - Building a …[truncated]

### L3-ee14644ba9  (L3, 2025-12-09, sha ee14644ba9a3, PR #25552)
TITLE: [ROCm] Aiter Quant Kernels (#25552)
SOURCES: release_notes
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/_aiter_ops.py (+87/-0); vllm/model_executor/layers/quantization/input_quant_fp8.py (+31/-0); vllm/platforms/rocm.py (+5/-2)
LABELS: rocm, ready, ci/build
BODY: ## Purpose ⏎ Integrate Aiter's fp8 quant kernels for improved perf. the following quant strategies are supported: ⏎  ⏎ - static per tensor quantization ⏎ - dynamic per tensor quantization ⏎ - dynamic per token quantization ⏎  ⏎ Perf gain is summarised in the following tables: ⏎  ⏎ **amd/Llama-3.1-70B-Instruct-FP8-KV (static per tensor quantization)** ⏎ Metric | Before | After ⏎ -- | -- | -- ⏎ Successful requests | 2000 | 2000 ⏎ Benchmark duration (s) | 1343.38 | 1303.08 ⏎ T …[truncated]

### L3-abe93bce59  (L3, 2025-12-09, sha abe93bce5952, PR #29624)
TITLE: [Attention] Make seq_lens_cpu optional in CommonAttentionMetadata to enable true async spec-decode (#29624)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/attention/layers/cross_attention.py (+1/-1); vllm/v1/attention/backends/utils.py (+49/-17); tests/v1/attention/utils.py (+2/-2); tests/v1/e2e/test_async_spec_decode.py (+131/-0); tests/v1/spec_decode/test_tree_attention.py (+2/-2); vllm/v1/attention/backends/gdn_attn.py (+1/-1); vllm/v1/spec_decode/eagle.py (+10/-10); vllm/v1/worker/gpu/attn_utils.py (+2/-2); vllm/v1/worker/gpu_model_runner.py (+2/-2)
LABELS: speculative-decoding, ready, v1, ready-run-all-tests
BODY: Implement: https://github.com/vllm-project/vllm/issues/29134

### L3-7618dc973d  (L3, 2025-12-09, sha 7618dc973dd1, PR #29145)
TITLE: [CI/Build] Make test_mha_attn.py run on correct platform only and check for flash_attn_varlen_func in layer.py (#29145)
SOURCES: path_core, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+4/-1); tests/kernels/attention/test_mha_attn.py (+9/-2)
LABELS: rocm, ready, ci/build
BODY: This PR changes `test_mha_attn.py` so it will only run on the correct platform.  This also fixes an issue in `layer.py` where `flash_attn_varlen_func` might not be available, so returns `None `in this case.  All unit tests in `test_mha_attn.py` pass now on AMD hardware: ⏎  ⏎ 18 passed, 3 warnings in 27.36s

### L3-3c680f4a17  (L3, 2025-12-09, sha 3c680f4a1705, PR #25693)
TITLE: [Rocm][torch.compile] Adding layernorm + fp8 block quant and silu + fp8 block quant for Aiter (#25693)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/test_fusion.py (+93/-5); tests/compile/test_silu_mul_quant_fusion.py (+57/-5); vllm/_aiter_ops.py (+170/-44); vllm/compilation/pass_manager.py (+11/-0); vllm/compilation/rocm_aiter_fusion.py (+242/-0); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+37/-6)
LABELS: rocm, ready
BODY: This PR adds a few fusion passes for Aiter to fusion layernorm + fp8 block quant and silu + fp8 block quant.

### L3-cebda2a4af  (L3, 2025-12-10, sha cebda2a4afa9, PR #30062)
TITLE: [CPU] Support for Whisper (#30062)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+19/-19); .buildkite/scripts/hardware_ci/run-cpu-test-arm.sh (+5/-0); csrc/cpu/cpu_attn.cpp (+0/-1); tests/models/multimodal/generation/test_whisper.py (+19/-2); vllm/v1/worker/utils.py (+6/-2)
LABELS: ready, ci/build, v1, multi-modality
BODY: ## Purpose ⏎ Enable support for Whisper for CPU backend  ⏎  ⏎ ## Test Plan ⏎  ⏎ * Added tests to `tests/models/multimodal/generation/test_whisper.py` -> should be enabled in [run-cpu-test.sh](https://github.com/vllm-project/vllm/blob/1b7c7f5159484063af28cb47809d79e83d3301ec/.buildkite/scripts/hardware_ci/run-cpu-test.sh#L66) ⏎ * `python examples/offline_inference/audio_language.py -m whisper` ⏎  ⏎ * ## Test Result ⏎  ⏎ ``` ⏎ python examples/offline_inference/audio_lan …[truncated]

### L3-ed7af3178a  (L3, 2025-12-10, sha ed7af3178aa2, PR #29358)
TITLE: [ROCm][CI] Attempt to fix the failures under a subgroup of the e2e the test group (#29358)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: requirements/rocm-test.txt (+1/-1); tests/multimodal/test_utils.py (+8/-2); tests/v1/e2e/test_async_scheduling.py (+76/-10)
LABELS: rocm, ready, ci/build, v1, multi-modality
BODY: This PR ensures that during `test_async_scheduling`, we utilize the `TRITON_ATTN` backend which is the default attention backend for ROCm. ⏎  ⏎ Test used to verify functionality on ROCm: `pytest -v -s tests/v1/e2e/test_async_scheduling.py`

### L3-434ac76a7c  (L3, 2025-12-10, sha 434ac76a7c2f, PR #30347)
TITLE: [cpu][ci] Add CPU Attention Tests for Neon Backend (#30347)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_cpu_attn.py (+63/-10)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Add CPU Attention Tests for Neon Backend ⏎ Should really have been part of #29193 but I missed it ⏎  ⏎ ## Test Plan ⏎  ⏎ Arm CI which includes CPU attention tests. ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-eea41804a4  (L3, 2025-12-10, sha eea41804a4b4, PR #30241)
TITLE: [bug] Fix "Current vLLM config is not set." warnings when FlashInfer attention is used (#30241)
SOURCES: path_core, path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+4/-1); vllm/v1/attention/backends/flashinfer.py (+2/-0)
LABELS: bug, ready, v1, nvidia
ISSUES: #30240 [Bug]: Lots of "Current vLLM config is not set." warnings when FlashInfer attention is used
BODY: ## Purpose ⏎  ⏎ VLLM config is set only during initialization stage, not during runtime stage. Therefore, we should not call get_current_vllm_config() during dunrime stage. Instead, cache the config we want during initialization stage and reuse it during runtime stage. ⏎  ⏎ This was caused by https://github.com/vllm-project/vllm/pull/26315 by @MatthewBonanni . ⏎  ⏎ This fixes https://github.com/vllm-project/vllm/issues/30240 ⏎  ⏎ ## Test Plan ⏎  ⏎ On H200: ⏎  ⏎ ``` ⏎ pyth …[truncated]

### L3-5a87d8b9b1  (L3, 2025-12-10, sha 5a87d8b9b1f3, PR #30396)
TITLE: [Deprecation] Remove deprecated plugin and compilation fields for v0.13 release (#30396)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.dispatch.selector, L3.dispatch.registry
FILES: vllm/attention/backends/registry.py (+0/-32); vllm/attention/selector.py (+12/-34); docs/design/plugin_system.md (+2/-2); tests/compile/test_config.py (+1/-62); tests/kernels/moe/test_ocp_mx_moe.py (+2/-2); tests/quantization/test_quark.py (+2/-2); tests/test_config.py (+1/-1); vllm/config/compilation.py (+1/-80); vllm/config/vllm.py (+1/-1); vllm/engine/arg_utils.py (+0/-22)
LABELS: documentation, ready
BODY: ## Purpose ⏎  ⏎ Remove items scheduled for removal in the upcoming release. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-b51255f369  (L3, 2025-12-11, sha b51255f369cf, PR #30432)
TITLE: [ROCm] Fix broken import in platform attention backend dispatching (#30432)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+15/-1)
LABELS: rocm, ready
BODY: ## Summary ⏎ Removes broken dependency on `get_env_variable_attn_backend` from `vllm.attention.selector` in ROCm platform configuration. ⏎  ⏎ ## Problem ⏎ The import `from vllm.attention.selector import get_env_variable_attn_backend` was causing failures on ROCm. This was used to check for `ROCM_AITER_UNIFIED_ATTN` backend selection when setting KV cache block size. ⏎  ⏎ ## Fix ⏎ Deprecate the `get_env_variable_attn_backend` check and rely on environment varia …[truncated]

### L3-a11f4a81e0  (L3, 2025-12-11, sha a11f4a81e027, PR #30050)
TITLE: [Misc][PCP&DCP] relocate PCP feature check (#30050)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+6/-0); vllm/config/parallel.py (+0/-5); vllm/config/vllm.py (+0/-5); vllm/engine/arg_utils.py (+0/-10); vllm/v1/worker/cp_utils.py (+42/-0); vllm/v1/worker/gpu_model_runner.py (+4/-14)
LABELS: ready, v1
BODY: Recently, vllm-ascend has been advancing the main-to-main adaptation work with the vllm repository. We have now fully adapted PCP and other features on vllm-ascend, so we believe it is necessary to add essential feature adaptation flags for the attention backend and modify the logic of the PCP feature check code to ensure that the functional switches for GPU and NPU do not interfere with each other. ⏎  ⏎ Modifications:​ ⏎ - Consolidate all PCP-related  …[truncated]

### L3-36c9ce2554  (L3, 2025-12-11, sha 36c9ce25543b, PR #30285)
TITLE: Ensure minimum frames for GLM 4.6V compatibility (#30285)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/glm4_1v.py (+1/-0)
LABELS: ready
BODY: Fix for startup video generation of 1 frame that causes issue with GLM 4.6V-FP8: ` (EngineCore_DP0 pid=1037) INFO 12-08 23:30:53 [core.py:93] Initializing a V1 LLM engine (v0.12.0) with config: model='zai-org/GLM-4.6V-FP8', speculative_config=None, tokenizer='zai-org/GLM-4.6V-FP8', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=N …[truncated]

### L3-fba8906930  (L3, 2025-12-11, sha fba89069302e, PR #29710)
TITLE: [perf] Use direct copy (broadcast) instead of cat for k_nope/k_pe in MLA prefill (#29710)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+30/-3); benchmarks/kernels/benchmark_mla_k_concat.py (+150/-0)
LABELS: performance, ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Reduce k tensor concatenation latency, from **3.16ms to 1.61ms** for batch size 32768 (i.e., `k.shape=torch.Size([32768, 128, 192]), k_nope.shape=torch.Size([32768, 128, 128]), k_pe.shape=torch.Size([32768, 1, 64])`. ⏎  ⏎ <img width="2565" height="1415" alt="image" src="https://github.com/user-attachments/assets/6398017c-e2af-4aa8-ae54-a51b9d6f82d8" /> ⏎  ⏎ ## Test Plan ⏎  ⏎ * verified perf improvement with trace ⏎ * verified accuracy ⏎ * microbenchm …[truncated]

### L3-72aaac5b66  (L3, 2025-12-11, sha 72aaac5b66f9, PR #30430)
TITLE: [ROCm][Bugfix] Add MLACommonMetadata to allowed attention types for speculative decoding (#30430)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/spec_decode/eagle.py (+6/-0)
LABELS: rocm, speculative-decoding, ready, v1
BODY: ## Summary ⏎ Adds `MLACommonMetadata` to the allowed attention types for ROCm in the EAGLE speculative decoding proposer. ⏎  ⏎ ## Problem ⏎ When using MLA on ROCm (e.g., DeepSeek) and speculative decoding with `num_speculative_tokens > 1`, the following error occurs: ⏎ ``` ⏎ ValueError: Unsupported attention metadata type for speculative decoding with num_speculative_tokens > 1: <class 'vllm.v1.attention.backends.mla.common.MLACommonMetadata'>. Supported typ …[truncated]

### L3-3e41992fec  (L3, 2025-12-12, sha 3e41992fecdc, PR #27532)
TITLE: [Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2 (#27532)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flashinfer.trtllm_gen, L3.mla.flashmla_sparse, L3.dispatch.abstract_interface
FILES: csrc/cache_kernels.cu (+130/-1); csrc/torch_bindings.cpp (+7/-0); vllm/_custom_ops.py (+23/-0); vllm/envs.py (+4/-0); vllm/model_executor/models/deepseek_v2.py (+18/-19); vllm/v1/attention/backends/mla/flashmla_sparse.py (+567/-98); vllm/v1/attention/backends/mla/indexer.py (+12/-36); vllm/v1/attention/backends/utils.py (+27/-0); vllm/v1/worker/gpu_model_runner.py (+6/-0); vllm/v1/worker/gpu_worker.py (+5/-0); (+20 more)
LABELS: ready, v1, deepseek, gpt-oss, nvidia, ready-run-all-tests
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: When doing prefill up-convert the kv-cache from fp8 to bf16 and call the bf16 prefill kernel instead of the decode kernel. This PR introduce global workspace management to have the bf16 workspace overlap with the MoE workspace buffers. ⏎  ⏎ ## GSM8K Accuracy (DeepSeek-V3.2, FP8 KV-cache, TP=8) ⏎  ⏎ | Branch | Accuracy | ⏎ |--------|----------| ⏎ | Main | 95.30% ± 0.58% | ⏎ | PR (Hybrid) | **95.53% ± 0.57%** | ⏎  ⏎ ## Benchmark Results (4096 in, 512 out, 100 prompt …[truncated]

### L3-09ad3b76b3  (L3, 2025-12-12, sha 09ad3b76b320, PR #30534)
TITLE: [Bug] Fix attention_backend arg string parsing (#30534)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/engine/arg_utils.py (+7/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ The issue was that `attention_backend` (a string from CLI parsing) is directly assigned to `attention_config.backend`, but the code expects an `AttentionBackendEnum` object  ⏎  ⏎ `vllm serve mgoin/Qwen3-0.6B-NVFP4 --attention-backend flashinfer` ⏎  ⏎ Before ⏎ ``` ⏎ (EngineCore_DP0 pid=36719)   File "/home/mgoin/code/vllm/vllm/attention/selector.py", line 46, in get_attn_backend ⏎ (EngineCore_DP0 pid=36719)     return _cached_get_attn_backend( ⏎ (Engi …[truncated]

### L3-cd7740ac5c  (L3, 2025-12-12, sha cd7740ac5c39, PR #26668)
TITLE: [ROCm] Enable Triton ScaledMM fallback + kernel selection fix (#26668)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test-pipeline.yaml (+1/-1); tests/kernels/quantization/test_scaled_mm_kernel_selection.py (+91/-0); vllm/model_executor/layers/quantization/kernels/scaled_mm/ScaledMMLinearKernel.py (+4/-1); vllm/model_executor/layers/quantization/kernels/scaled_mm/__init__.py (+12/-28); vllm/model_executor/layers/quantization/kernels/scaled_mm/aiter.py (+15/-7); vllm/model_executor/layers/quantization/kernels/scaled_mm/cpu.py (+6/-5); vllm/model_executor/layers/quantization/kernels/scaled_mm/cutlass.py (+12/-5); vllm/model_executor/layers/quantization/kernels/scaled_mm/triton.py (+46/-17); vllm/model_executor/layers/quantization/kernels/scaled_mm/xla.py (+6/-5)
LABELS: rocm, ready, ci/build, nvidia
ISSUES: #14397 [Bug]: `triton_scaled_mm` never used on ROCm
BODY: <h3><strong>Purpose</strong></h3> ⏎ <p>Fixes #14397 — <code inline="">triton_scaled_mm</code> was never used on ROCm due to missing dispatch and checks.<br> ⏎ This PR:</p> ⏎ <ul> ⏎ <li> ⏎ <p>Enables <strong>Triton fallback</strong> for ROCm when AITriton is unavailable</p> ⏎ </li> ⏎ <li> ⏎ <p>Adds Triton fallback after CUTLASS on CUDA</p> ⏎ </li> ⏎ <li> ⏎ <p>Implements <code inline="">is_supported()</code> checks for kernel selection</p> ⏎ </li> ⏎ <li> ⏎ <p>Adds a lightweig …[truncated]

### L3-9c0ee995a8  (L3, 2025-12-12, sha 9c0ee995a81f, PR #28306)
TITLE: [Kernel] Support CUDA Graphs in 3D Triton Attention Kernel (#28306)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.triton.unified_attention, L3.triton.v1_backend
FILES: vllm/attention/ops/triton_unified_attention.py (+30/-39); vllm/v1/attention/backends/triton_attn.py (+83/-1); tests/kernels/attention/test_triton_unified_attention.py (+27/-0)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ ~~This pull request depends on PR #27993.~~ ⏎  ⏎ This pull request adapts the 3D Triton attention kernel, which is used exclusively for decode operations, to support full CUDA Graphs. The key changes include: ⏎  ⏎ - The allocation of the intermediate data structures used for the tiled softmax implementation has been moved to the attention metadata builder class. ⏎  ⏎ - The dynamic selection between the 2D and 3D attention kernels during decode is  …[truncated]

### L3-4fa7ce46f3  (L3, 2025-12-12, sha 4fa7ce46f31c, PR #30484)
TITLE: [Feature] Add SM103 (Blackwell Ultra) Support to vLLM (#30484)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.mla.flashmla_sparse, L3.platform.cuda_selection
FILES: vllm/utils/flashinfer.py (+3/-1); vllm/v1/attention/backends/flashinfer.py (+1/-1); vllm/v1/attention/backends/mla/common.py (+3/-3); vllm/v1/attention/backends/mla/flashmla_sparse.py (+2/-2); tests/compile/distributed/test_fusions_e2e.py (+1/-1); tests/kernels/attention/test_cutlass_mla_decode.py (+2/-2); tests/kernels/attention/test_flashinfer_trtllm_attention.py (+2/-2); tests/kernels/moe/test_ocp_mx_moe.py (+2/-2); tests/quantization/test_blackwell_moe.py (+2/-2); vllm/model_executor/layers/batch_invariant.py (+1/-1); (+11 more)
LABELS: ready, v1, nvidia
BODY: ## Summary ⏎  ⏎ This PR introduces initial **SM103 support** needed to run vLLM on NVIDIA’s (G)B300 GPUs. The main focus so far has been validating correct functionality and ensuring that key kernels, quantization paths, and MoE components work reliably on Blackwell Ultra GPUs. ⏎  ⏎ - All tests in `tests/quantization/test_blackwell_moe.py` **pass**. ⏎ - Resolved Triton issues related to SM103 support (details below). ⏎ - Verified that `nvidia/DeepSeek-V3.1-N …[truncated]

### L3-13618626df  (L3, 2025-12-12, sha 13618626dff7, PR #29748)
TITLE: [MoE-FP8-modelopt] Add FlashInfer alignment padding for intermediate dimensions (#29748)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/modelopt.py (+48/-0)
LABELS: ready
BODY: This PR adds support for FlashInfer MoE FP8 kernels that require the intermediate (gated) dimension to be aligned. ⏎ Some models break when the intermediate size isn’t divisible by 16, especially under different TP. ⏎  ⏎ This PR introduces _maybe_pad_intermediate_for_flashinfer, which pads w13 and w2 along the intermediate dim so the weights meet FlashInfer’s alignment constraints. ⏎  ⏎ If padding is needed, we zero-pad the up/gate and down projection weig …[truncated]

### L3-86a3261525  (L3, 2025-12-13, sha 86a326152585, PR #30575)
TITLE: [Bugfix] Pass FA version in `MultiHeadAttention` (#30575)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+10/-0)
LABELS: ready
BODY: ## Purpose ⏎ The call to `flash_attn_varlen_func` in `MultiHeadAttention` does not pass `fa_version`. This causes multimodal models to use a mix of FA2 (default) and FA3. This PR fixes this. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ vllm serve openai/whisper-large-v3 ⏎ ``` ⏎ ``` ⏎ curl -X POST http://localhost:8000/v1/audio/transcriptions \ ⏎     -H "Content-Type: multipart/form-data" \ ⏎     -F "file=@mary_had_lamb.ogg" \ ⏎     -F "model=openai/whisper-large-v3" ⏎ ``` ⏎  ⏎ ## Test Result ⏎  …[truncated]

### L3-f5dfbbd8e9  (L3, 2025-12-13, sha f5dfbbd8e9f3, PR #30564)
TITLE: [Docs] Remove references to `VLLM_ATTENTION_BACKEND` (#30564)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: docs/getting_started/quickstart.md (+16/-6)
LABELS: documentation, ready
BODY: ## Purpose ⏎ PR #26315 deprecated the environment variable `VLLM_ATTENTION_BACKEND`. This PR updates the documentation accordingly. ⏎  ⏎ ## Test Plan ⏎ N/A ⏎  ⏎ ## Test Result ⏎ N/A ⏎  ⏎ --- ⏎ [details omitted]

### L3-87b4d1557d  (L3, 2025-12-15, sha 87b4d1557dc8, PR #30125)
TITLE: [CustomOp][MM] Extract MMEncoderAttention as CustomOp and replace the backend of QwenVisionAttention with it. (#30125)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:kernel-correctness-cases(introducing), body_keyword
ARTIFACT_HINTS: L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/attention/layer.py (+5/-68); vllm/attention/layers/mm_encoder_attention.py (+284/-0); vllm/attention/ops/vit_attn_wrappers.py (+3/-8); vllm/platforms/cuda.py (+36/-18); vllm/platforms/interface.py (+38/-7); vllm/platforms/rocm.py (+38/-19); vllm/platforms/tpu.py (+27/-1); vllm/platforms/xpu.py (+29/-7); tests/models/multimodal/generation/test_vit_backend_functionality.py (+434/-0); vllm/model_executor/models/dots_ocr.py (+46/-83); (+14 more)
LABELS: rocm, tpu, ready, multi-modality, qwen, nvidia
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:2410132bb1 (fix PR 30789)
BODY: ## Purpose ⏎  ⏎ To avoid maintaining a variety of modeling files in vllm-ascend, we propose to remove all files in `models` dir in vllm-ascend. After this, the only thing a vllm plugin need to do is just registering their custom device-specific OOT ops to vllm when adding a new model. To achieve this, there are some refactors need to be done both in vllm and vllm-ascend, such as extracting some general layers as CustomOp, find more details at https:/ …[truncated]

### L3-511e81e7c9  (L3, 2025-12-15, sha 511e81e7c9a8, PR #30705)
TITLE: [BUILD] use sm_100f when compiling flashmla to fix support on sm103 (#30705)
SOURCES: path_core, subject_keyword, dependency_pin, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_build
FILES: cmake/external_projects/flashmla.cmake (+11/-5)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ Now flashmla is compiled with CUDA arch `9.0a` and `10.0a`. This will trigger errors on B300 (`sm103a`). ⏎  ⏎ This PR uses the new family specifier `10.0f` introduced in CUDA 12.9 when possible, which will support all possible 10.x computing capabilities. ⏎  ⏎ Besides, the original version CUDA checking was wrong. CUDA 12.3 has introduced `sm_90a`, 12.8 has `sm_100a`, and 12.9 has `sm_100f`. This PR also fixes that. ⏎  ⏎ ## Test Plan ⏎  ⏎ Run unit-te …[truncated]

### L3-1adeb3b84c  (L3, 2025-12-15, sha 1adeb3b84c2d, PR #28439)
TITLE: [New Model] BAGEL support (AR only) (#28439)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/models/supported_models.md (+1/-0); examples/offline_inference/vision_language.py (+27/-0); tests/models/registry.py (+1/-0); vllm/model_executor/models/bagel.py (+584/-0); vllm/model_executor/models/qwen2.py (+32/-0); vllm/model_executor/models/registry.py (+1/-0); vllm/transformers_utils/config.py (+1/-0); vllm/transformers_utils/configs/__init__.py (+2/-0); vllm/transformers_utils/configs/bagel.py (+53/-0); vllm/transformers_utils/processors/__init__.py (+2/-0); (+1 more)
LABELS: documentation, new-model, ready, qwen
BODY: ## Purpose ⏎ Implement Bagel Model [#18793](https://github.com/vllm-project/vllm/issues/18793). Since vLLM currently does not support Diffusion models, Bagel skips the VAE-related weights for now. In the future, once support for these models is implemented, I will complete the full Bagel model, including the remaining Img2Img and Text2Img functionalities. ⏎ ## Test Plan ⏎  ⏎ ### First paste the config.json to bagel model folder ⏎ ``` ⏎ { ⏎   "architectures": [ …[truncated]

### L3-60dbf7d8f1  (L3, 2025-12-15, sha 60dbf7d8f136, PR #30704)
TITLE: Update batch invariant to use attention config (#30704)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/batch_invariant.py (+22/-17); vllm/v1/worker/gpu_worker.py (+2/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎ The `VLLM_ATTENTION_BACKEND` environment variable has been deprecated by #26315. This PR updates the batch invariant initialization accordingly. ⏎  ⏎ ## Test Plan ⏎ `pytest tests/v1/determinism/test_batch_invariance.py::test_decode_logprobs_match_prefill_logprobs[FLASH_ATTN]` ⏎  ⏎ ## Test Result ⏎ Should pass in CI ⏎  ⏎ --- ⏎ [details omitted]

### L3-ec154c36ee  (L3, 2025-12-15, sha ec154c36ee74, PR #30212)
TITLE: [Platform] Refactor Platform attention backend selection to avoid breakpoint for OOT platform (#30212)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L3.dispatch.selector, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/attention/selector.py (+36/-23); vllm/platforms/cpu.py (+4/-11); vllm/platforms/cuda.py (+15/-59); vllm/platforms/interface.py (+2/-10); vllm/platforms/rocm.py (+9/-13); vllm/platforms/tpu.py (+3/-10); vllm/platforms/xpu.py (+4/-11)
LABELS: rocm, tpu, ready, nvidia
BODY: ## Purpose ⏎ - Currently, there are too many arguments used by platform's `get_attn_backend_cls`, while not all platforms use all of them. And it will also easily break OOT platform when introduce attention feature with new argument like `use_mla` and `use_sink`. ⏎ - To avoid this kind of mess and breakage, this PR wrap platform's attention selection arguments into hashable `AttentionSelectorConfig`, so that platform can use these arguments on demand …[truncated]

### L3-ff21a0fc85  (L3, 2025-12-15, sha ff21a0fc8593, PR #30626)
TITLE: [docker] Restructure Dockerfile for more efficient and cache-friendly builds (#30626)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+157/-115); docs/assets/contributing/dockerfile-stages-dependency.png (+0/-0)
LABELS: documentation, ready, ci/build
BODY: ## Approach: ⏎ - Pre-install PyTorch, FlashInfer, and other slow-changing dependencies in vllm-base before installing the vLLM wheel for better layer caching ⏎ - Add parallel extensions-build stage for DeepGEMM and EP kernels ⏎ - Move stable packages (accelerate, bitsandbytes, etc.) earlier in build ⏎  ⏎ This allows incremental builds with Python-only changes to skip the expensive dependency installation layers. ⏎  ⏎ ## Performance: ⏎ Incremental builds with Pyt …[truncated]

### L3-3bd9c49158  (L3, 2025-12-15, sha 3bd9c491583d, PR #29873)
TITLE: [CustomOp] Extract ApplyRotaryEmb as CustomOp and unify the dispatch logic (#29873)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/core/test_apply_rotary_emb.py (+203/-0); vllm/model_executor/layers/rotary_embedding/base.py (+17/-3); vllm/model_executor/layers/rotary_embedding/common.py (+153/-71); vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py (+10/-3); vllm/model_executor/layers/rotary_embedding/mrope.py (+20/-5); vllm/model_executor/layers/rotary_embedding/xdrope.py (+62/-4); vllm/model_executor/models/dots_ocr.py (+13/-27); vllm/model_executor/models/ernie45_vl.py (+14/-49); vllm/model_executor/models/glm4_1v.py (+10/-3); vllm/model_executor/models/keye.py (+9/-13); (+4 more)
LABELS: documentation, rocm, ready, qwen
BODY: ## Purpose ⏎  ⏎ 1. In some modeling files, there are direct calling of `apply_rotary_emb` function by using pre-computed cos/sin cache, like: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_5_vl.py#L383-L385. This is just a function, not an operation, and cannot be overwritten by some plugins (e.g., vllm-ascend). By extracting it as an CustomOp, we can just extend this class and implement our `forward_oot()` function,  …[truncated]

### L3-51e5b3e3c4  (L3, 2025-12-15, sha 51e5b3e3c422, PR #30703)
TITLE: [Bugfix] Fix ViT with FlashAttention on ROCm (#30703)
SOURCES: path_core, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+4/-1)
LABELS: rocm, ready
BODY: ## Purpose ⏎ #30575 introduced a bug on AMD because AMD's `flash_attn_varlen_func` doesn't take the `fa_version` argument. This led to CI failures: https://buildkite.com/vllm/amd-ci/builds/1680/steps/canvas?sid=019b222d-cb50-495f-b5e2-3ba219617298 ⏎  ⏎ This PR is a fix for that failure. ⏎  ⏎ ## Test Plan ⏎ AMD CI: `mi325_1: V1 Test entrypoints`, specifically `v1/entrypoints/openai/test_completion_with_image_embeds.py::test_completions_with_image_embeds[dtype …[truncated]

### L3-b9ff4f2a8d  (L3, 2025-12-16, sha b9ff4f2a8dff, PR #30120)
TITLE: [feature] extend DBO to XBO (#30120)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+9/-5); tests/v1/attention/test_attention_splitting.py (+1/-0); vllm/config/parallel.py (+10/-0); vllm/config/vllm.py (+5/-2); vllm/engine/arg_utils.py (+6/-0); vllm/v1/worker/dp_utils.py (+4/-4); vllm/v1/worker/gpu_model_runner.py (+26/-7); vllm/v1/worker/gpu_ubatch_wrapper.py (+19/-16); vllm/v1/worker/ubatch_utils.py (+37/-34); vllm/v1/worker/ubatching.py (+16/-5)
LABELS: documentation, ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ As mentioned in #30105  ⏎ This PR implements the extension of DBO  to XBO in the codebase. I have added a command-line argument --num-of-microbatches to be used in conjunction with --enable-dbo. ⏎ At present, the core code modification for the DBO→XBO extension has been completed. The test results with 3 microbatches are still in progress and will be supplemented and attached to this PR as soon as they are available. ⏎ The main purpose of su …[truncated]

### L3-e94384bbad  (L3, 2025-12-16, sha e94384bbadba, PR #30731)
TITLE: [Bugfix] Fix broken ViT attention selection for Blackwell device (#30731)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/vision.py (+2/-8)
LABELS: ready, nvidia
BODY: ## Purpose ⏎ - Fix broken CI: https://buildkite.com/vllm/ci/builds/43544/steps/canvas?jid=019b2000-cb09-48be-b1e6-4394d847f73e#019b2000-cb09-48be-b1e6-4394d847f73e ⏎ - On blackwell device, default FlashInfer backend is not a valid ones for ViT, we need to fall back to valid backends. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-254a7f8fd6  (L3, 2025-12-16, sha 254a7f8fd613, PR #30014)
TITLE: [Perf] Do FP4 quant before All gather on flashinfer trtllmgen MOE  (#30014)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+17/-0); vllm/distributed/device_communicators/all2all.py (+24/-5); vllm/distributed/device_communicators/base_device_communicator.py (+6/-1); vllm/distributed/device_communicators/cuda_communicator.py (+11/-5); vllm/distributed/parallel_state.py (+10/-3); vllm/model_executor/layers/fused_moe/fused_moe_method_base.py (+12/-0); vllm/model_executor/layers/fused_moe/layer.py (+39/-2); vllm/model_executor/layers/quantization/modelopt.py (+24/-1); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+22/-14)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Move the FP4 quant before All Gather when Flashinfer TRTLLMGEN MOE is used on DP + EP enabled. This reduce the message size in All gather thus speed up All gather kernel time. ⏎ Blocked by https://github.com/vllm-project/vllm/pull/29804, will add same opt to flashinfer_trtllm_fp4_routed_moe after merged. ⏎ Original size ⏎ ``` ⏎ hidden_states (num_tokens, hidden_size), dtype bf16 ⏎ routing_logits (num_tokens, num_experts), dtype fp32 ⏎ ``` ⏎ After ch …[truncated]

### L3-9dbbc59b15  (L3, 2025-12-16, sha 9dbbc59b1511, PR #28624)
TITLE: [ROCm][MTP] Support MTP for AITER MLA backend (#28624)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+15/-9)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ This PR support the MTP inference for deepseek. ⏎ ## Test Plan ⏎ gsm8k ⏎ ## Test Result ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|    20|exact_match|↑  |0.9386|±  |0.0066| ⏎ |     |       |strict-match    |    20|exact_match|↑  |0.9371|±  |0.0067| ⏎  ⏎ ``` ⏎ INFO 12-15 05:36:45 [metrics.py:100] SpecDecoding metri …[truncated]

### L3-2410132bb1  (L3, 2025-12-16, sha 2410132bb1f9, PR #30789)
TITLE: [ROCm] [Bugfix] Fix torch sdpa hallucination (#30789)
SOURCES: path_core, corpus:kernel-correctness-cases
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/vit_attn_wrappers.py (+8/-0)
LABELS: rocm, ready
DEEP_STUDY: deep-study correctness case vllm:2410132bb1: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=#30125
BODY: ## Purpose ⏎  ⏎ PR https://github.com/vllm-project/vllm/pull/30125 removed the necessary steps for ROCm that was addressed in https://github.com/vllm-project/vllm/pull/27744 ⏎  ⏎ ## Test Plan ⏎  ⏎ Use the `examples/offline_inference/vision_language.py` for quick validation if hallucination has been fixed. ⏎  ⏎ ## Test Result ⏎  ⏎ For the same tokyo tree image in the `examples/offline_inference/vision_language.py` ⏎ When it hallucinates ⏎  ⏎ Before fix qwen3_vl ⏎ ``` ⏎ ------- …[truncated]

### L3-9fec0e13d5  (L3, 2025-12-16, sha 9fec0e13d512, PR #29627)
TITLE: [Attention] Cache attention metadata builds across hybrid KV-cache groups (#29627)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.trtllm_gen, L3.dispatch.abstract_interface
FILES: vllm/attention/layers/chunked_local_attention.py (+12/-4); vllm/v1/attention/backends/flash_attn.py (+13/-0); vllm/v1/attention/backends/utils.py (+27/-5); tests/v1/attention/test_chunked_local_attention.py (+1/-1); vllm/envs.py (+2/-2); vllm/v1/attention/backends/mamba2_attn.py (+27/-0); vllm/v1/worker/gpu_model_runner.py (+23/-1)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Scaled back version of: https://github.com/vllm-project/vllm/pull/22788 and generalization of: https://github.com/vllm-project/vllm/pull/29444 ⏎  ⏎ Depends on https://github.com/vllm-project/vllm/pull/29628 land that first ⏎  ⏎ ## Test Plan ⏎  ⏎ Decode-heavy workload with APC enabled => **latency reduction through this PR: 13.7%** ⏎  ⏎ ``` ⏎ vllm bench latency --model ibm-granite/granite-4.0-tiny-preview --input-len 128  --output-len 2048 \ ⏎     --batch-size 8 --en …[truncated]

### L3-d4d2751732  (L3, 2025-12-16, sha d4d2751732c3, PR #30711)
TITLE: Update note comment for flashinfer attention warmup (#30711)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/warmup/kernel_warmup.py (+3/-4)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-e80455ca8b  (L3, 2025-12-16, sha e80455ca8b69, PR #30817)
TITLE: Replace deprecated enable_fusion with fuse_norm_quant in test_rms_group_quant (#30817)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/distributed/test_fusions_e2e.py (+1/-1)
LABELS: ready, ci-failure
BODY: ## Purpose ⏎  ⏎ Landing strict config matching in https://github.com/vllm-project/vllm/pull/30708 caused this test to start to fail ⏎ https://buildkite.com/vllm/ci/builds/43658/steps/canvas?sid=019b2408-04fa-4f90-be13-3d338355f821#019b2408-0652-4d81-aa1d-60475172b711/86-5524 ⏎ ``` ⏎ FAILED tests/compile/distributed/test_fusions_e2e.py::test_rms_group_quant[True-Qwen/Qwen3-30B-A3B-FP8-model_kwargs0-AttentionBackendEnum.TRITON_ATTN-matches0-+quant_fp8,+rms_n …[truncated]

### L3-a100152288  (L3, 2025-12-17, sha a100152288c8, PR #30842)
TITLE: [Kernels][FI] Skip trtllm attention when num_kv_heads=1 (#30842)
SOURCES: path_core, corpus:confirmed-reverts(reverted), body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+21/-1); tests/kernels/attention/test_flashinfer_trtllm_attention.py (+35/-0)
LABELS: ready, nvidia
DEEP_STUDY: deep-study: this PR was reverted by PR 31617 (confirmed_revert, reason=crash_or_hang)
BODY: ## Purpose ⏎ We got the following error when running a small model on blackwell  ⏎ ``` ⏎ [WORKER]:  File "/redacted/path/executor/abstract.py", line 116, in initialize_from_config ⏎ [WORKER]:    self.collective_rpc("compile_or_warm_up_model") ⏎ [WORKER]:  File "/redacted/path/executor/uniproc_executor.py", line 75, in collective_rpc ⏎ [WORKER]:    result = run_method(self.driver_worker, method, args, kwargs) ⏎ [WORKER]:  File "/redacted/path/serial_utils.py",  …[truncated]

### L3-e087fbc393  (L3, 2025-12-17, sha e087fbc39305, PR #30756)
TITLE: [MM] Pass FA version in ViT Attn (#30756)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/layers/mm_encoder_attention.py (+6/-0); vllm/attention/ops/vit_attn_wrappers.py (+8/-1)
LABELS: ready
BODY: Same rationale as https://github.com/vllm-project/vllm/pull/30575/ but for non-causal ViT attn. ⏎  ⏎ Test with `pytest -v -s tests/models/multimodal/generation/test_vit_backend_functionality.py`

### L3-a9e15c21ef  (L3, 2025-12-17, sha a9e15c21efbb, PR #30712)
TITLE: [Mamba] Removed disable cascade attn in MambaModelConfig (#30712)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/config.py (+0/-6)
LABELS: ready
BODY: ## Purpose ⏎ Since this is already protected by `MambaManager` there is no reason to disable cascade attention from here. ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-7eb6cb6c18  (L3, 2025-12-17, sha 7eb6cb6c18a9, PR #30563)
TITLE: [Attention] Update tests to remove deprecated env vars (#30563)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn
FILES: vllm/v1/attention/backends/rocm_attn.py (+1/-1); .buildkite/scripts/hardware_ci/run-xpu-test.sh (+1/-1); tests/basic_correctness/test_basic_correctness.py (+41/-48); tests/compile/distributed/test_fusions_e2e.py (+6/-3); tests/compile/fullgraph/test_basic_correctness.py (+40/-42); tests/compile/fullgraph/test_full_cudagraph.py (+3/-10); tests/compile/fullgraph/test_full_graph.py (+3/-4); tests/distributed/test_context_parallel.py (+1/-3); tests/distributed/test_pp_cudagraph.py (+12/-14); tests/engine/test_arg_utils.py (+134/-1); (+24 more)
LABELS: rocm, speculative-decoding, ready, ci/build, v1, multi-modality, kv-connector, nvidia
BODY: ## Purpose ⏎ Attention-related environment variables have been deprecated by #26315. This PR updates the tests to remove all usage of these environment variables. ⏎  ⏎ ## Test Plan ⏎ CI (use `ready-run-all-tests`) ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-74a1ac38b0  (L3, 2025-12-17, sha 74a1ac38b00a, PR #30386)
TITLE: [v1] Add PrefixLM support to TritonAttention backend (#30386)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.triton.unified_attention, L3.triton.v1_backend
FILES: vllm/attention/ops/triton_unified_attention.py (+143/-21); vllm/model_executor/models/gemma3.py (+0/-69); vllm/v1/attention/backends/triton_attn.py (+39/-0); tests/models/multimodal/generation/test_multimodal_gguf.py (+99/-34)
LABELS: rocm, ready, v1, multi-modality
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ - There are some feedbacks from model vendors reported that FlexAttention's PrefixLM implementation (image BiDi attetion) is a bit slow because of block mask recomputation. ⏎ - This PR also adds image-BiDi attention support to exisition Triton attention backend, which should be faster than FlexAttention. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎ ``` ⏎ pytest -s -v tests/models/multimodal/generation/test_common.py -k gemma3-test ⏎ ``` ⏎ ``` ⏎ tests/models/mu …[truncated]

### L3-6482e3895b  (L3, 2025-12-17, sha 6482e3895baa, PR #30688)
TITLE: chores: adjust the attn register param order (#30688)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.registry
FILES: vllm/attention/backends/registry.py (+1/-1)
LABELS: ready
BODY: ## Purpose ⏎ This is a follow-up to [#26487](https://github.com/vllm-project/vllm/pull/26487). ⏎  ⏎ It accidently change the order of attention registry default parameter, which caused the failure in normal registry and crashed all the workflow of our plugins tests (the vllm-metax). ⏎  ⏎ For not adding `class_path =` everywhere in the registry code in non-mamba cases ( and I think that's the most common situation) , it is needed to exchange the order of th …[truncated]

### L3-4a8412f773  (L3, 2025-12-17, sha 4a8412f773c6, PR #30903)
TITLE: [UX] Reduce DeepGEMM warmup log output to single progress bar (#30903)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/warmup/deep_gemm_warmup.py (+99/-42)
LABELS: ready, startup-ux
BODY: ## Purpose ⏎ Showing a progress bar for each shape during DeepGEMM warmup is unnecessary. In the interest of reducing log clutter, this PR reduces the output to a single progress bar, only shown on rank 0. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ vllm serve deepseek-ai/DeepSeek-R1 -dp 8 --enable-expert-parallel ⏎ ``` ⏎  ⏎ ## Test Result ⏎ Main: ⏎ ``` ⏎ DeepGemm(fp8_gemm_nt) warmup (W=torch.Size([24576, 1536])) [relaxed]:   0%|                                     | 0/297 [00:00<?, ?i …[truncated]

### L3-11a89cf95c  (L3, 2025-12-18, sha 11a89cf95caa, PR #30915)
TITLE: [Fix][FlexAttention] return max logical block index to handle reused blocks (#30915)
SOURCES: path_core
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+12/-3); tests/kernels/test_flex_attention.py (+30/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ For FlexAttention, we need to build `physical_to_logical_mapping` to reversely map physical block ids to logical block ids. This process previously assumes logical block ids are always unique, which is not true for some attention types such as sliding window attention, where some blocks may be released and reused later, causing the same physical block id to appear multiple times in a row of `block_table` at different logical block ind …[truncated]

### L3-6628758233  (L3, 2025-12-18, sha 66287582339d, PR #30907)
TITLE: [Bug] Fix batch invariant in torch 2.10 (#30907)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/batch_invariant.py (+20/-24)
LABELS: ready
ISSUES: #170490 vLLM Batch Invariance tests failing PyTorch 2.10
BODY: ## Purpose ⏎  ⏎ Fixes https://github.com/pytorch/pytorch/issues/170490 ⏎  ⏎ ## Test ⏎  ⏎ Originally: ⏎  ⏎ ```bash ⏎ ... ⏎  ⏎ == Not the same as == ⏎  ⏎ 19-year-old girl named Sarah. She was a very talented artist and had a passion for painting. She loved to paint landscapes, portraits, and even abstract art. Sarah had a small studio in her home where she spent most of her time creating beautiful paintings. ⏎  ⏎ One day, Sarah received an invitation to participate in a prestig …[truncated]

### L3-fd8afdf38d  (L3, 2025-12-18, sha fd8afdf38dad, PR #30811)
TITLE: [ROCm][CI] Reduce Flakiness For test_async_scheduling Using ROCM_ATTN With FP32 (#30811)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn
FILES: vllm/v1/attention/backends/rocm_attn.py (+5/-1); tests/v1/e2e/test_async_scheduling.py (+2/-10)
LABELS: rocm, ready, v1
BODY: We've seen some flakiness in `v1/e2e/test_async_scheduling.py::test_without_spec_decoding` ever since the test was initially fixed in https://github.com/vllm-project/vllm/pull/29358. Ultimately we believe it was due to switching to fp16 for the test. Before, the test failed about 10%-20% of the time with the following error: ⏎ ``` ⏎ =========================== short test summary info ============================ ⏎ FAILED v1/e2e/test_async_scheduling.py …[truncated]

### L3-53ad423f26  (L3, 2025-12-18, sha 53ad423f2638, PR #30729)
TITLE: [Perf] enable flashinfer rotary_embedding custom ops in DeepSeek rotary (#30729)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/rotary_embedding/base.py (+4/-1); vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py (+20/-1)
LABELS: ready, deepseek
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ - Enable flashinfer rotary_embedding custom ops in DeepSeek ⏎ **To use the custom op, add `--compilation_config.custom_ops+=+rotary_embedding` in vllm config** ⏎ ## Test Plan ⏎ ``` ⏎ VLLM_USE_FLASHINFER_MOE_FP4=1 python3 -m vllm.entrypoints.openai.api_server --model nvidia/DeepSeek-R1-0528-FP4-v2 --tokenizer nvidia/DeepSeek-R1-0528-FP4-v2 --dtype auto --kv-cache-dtype fp8 --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size …[truncated]

### L3-d6b3d39b6d  (L3, 2025-12-18, sha d6b3d39b6d87, PR #29128)
TITLE: [Cleanup] Refactor FlashInferMetadataBuilder (#29128)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+403/-269)
LABELS: ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ Reorganize the FlashInferMetadata into clear `prefill` and `decode` sections that either belong to FlashInfer or TRTLLM execution pathways. This separation is desirable because it allows us to make explicit which metadata needs to be prepared for each backend, and therefore which computations can be omitted when a certain backend is not used. ⏎  ⏎ As such, I refactor (and make skip-able) some computations related to the paged kv indices,  …[truncated]

### L3-8da6ae49c3  (L3, 2025-12-18, sha 8da6ae49c3d9, PR #30909)
TITLE: [ROCm][Bugfix] Fix `fa_version` argument error in `flash_attn_maxseqlen_wrapper` for ROCm without aiter (#30909)
SOURCES: path_core, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/vit_attn_wrappers.py (+5/-4)
LABELS: rocm, ready
BODY: ### Problem ⏎ On ROCm platforms with `AITER` either uninstalled or disabled, `flash_attn_varlen_func` fails with: ⏎ ``` ⏎ TypeError: flash_attn_varlen_func() got an unexpected keyword argument 'fa_version' ⏎ ``` ⏎  ⏎ This occurs because `is_rocm_aiter` is `False` when `aiter` is disabled, causing the code to fall through to the else branch which unconditionally passes `fa_version` to `flash_attn_varlen_func`. However, the ROCm version of Flash Attention (via …[truncated]

### L3-d2dc5dfc6e  (L3, 2025-12-18, sha d2dc5dfc6eca, PR #30973)
TITLE: [Bugfix] Remove `tile_size=64` for mm_prefix triton attention (#30973)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+0/-7)
LABELS: ready
BODY: ## Purpose ⏎ - Fix https://github.com/vllm-project/vllm/pull/30386#discussion_r2631541204 ⏎  ⏎ cc @lucianommartins  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-700a5ad6c6  (L3, 2025-12-19, sha 700a5ad6c616, PR #30684)
TITLE: [MM Encoder]: Migrate legacy ViT `MultiHeadAttention` to new `MMEncoderAttention` interface (#30684)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+0/-132); vllm/attention/layers/mm_encoder_attention.py (+27/-63); vllm/attention/ops/vit_attn_wrappers.py (+39/-14); tests/kernels/attention/test_attention.py (+3/-2); tests/kernels/attention/test_mha_attn.py (+67/-11); tests/v1/tpu/test_mha_attn.py (+3/-3); vllm/model_executor/models/aimv2.py (+2/-2); vllm/model_executor/models/blip.py (+2/-2); vllm/model_executor/models/clip.py (+6/-5); vllm/model_executor/models/deepencoder.py (+2/-2); (+10 more)
LABELS: tpu, ready, v1, llama
BODY: ## Purpose ⏎ - Following PR for #30125 ⏎ - Migrate `MultiHeadAttention` usage to new `MMEncoderAttention` ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest - s-v tests/kernels/attention/test_attention.py ⏎ ``` ⏎ ``` ⏎ pytest -s -v tests/kernels/attention/test_mha_attn.py ⏎ ``` ⏎  ⏎ ## Test Result ⏎ Test should pass ⏎  ⏎ --- ⏎ [details omitted]

### L3-7b43db210c  (L3, 2025-12-19, sha 7b43db210c34, PR #30270)
TITLE: [ROCm][CI][Bugfix] Multi-Modal Model Support Fixes and Attention Backend Improvements (#30270)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+28/-14); .buildkite/test-amd.yaml (+16/-5); tests/models/multimodal/conftest.py (+3/-6); tests/models/multimodal/generation/test_common.py (+30/-2); tests/models/multimodal/generation/test_granite_speech.py (+1/-1); tests/models/multimodal/pooling/conftest.py (+0/-18); vllm/model_executor/models/transformers/multimodal.py (+31/-1)
LABELS: rocm, ready, ci/build, multi-modality, qwen
BODY: This PR addresses several ROCm-specific issues with multi-modal/vision-language models and improves attention backend dispatching for encoder-only self-attention models. It renders green the following test groups on ROCm: ⏎ - `Multi-Modal Models Test (Standard)` ⏎ - `Multi-Modal Models Test (Extended) 1` ⏎ - `Multi-Modal Models Test (Extended) 2` ⏎ - `Multi-Modal Models Test (Extended) 3` ⏎  ⏎ #### Key Changes ⏎  ⏎ **Attention Backend Selection (`vllm/platforms/ …[truncated]

### L3-83a317f650  (L3, 2025-12-19, sha 83a317f650f2, PR #30990)
TITLE: [MoE Refactor][3/N] Deprecate cutlass block quant fp8 (b200) (#30990)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+0/-18); csrc/quantization/w8a8/cutlass/moe/blockwise_scaled_group_mm_sm100.cu (+0/-373); csrc/torch_bindings.cpp (+0/-7); tests/kernels/moe/test_cutlass_grouped_gemm.py (+0/-92); vllm/_custom_ops.py (+0/-14); vllm/model_executor/layers/fused_moe/cutlass_moe.py (+1/-157); vllm/model_executor/layers/fused_moe/fused_moe.py (+0/-23); vllm/model_executor/layers/quantization/fp8.py (+2/-20)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎ * per https://github.com/vllm-project/vllm/issues/30989, CUTLASS Block quant FP8 does not run on main. When modifying vLLM to force it to run, it get an IMA immediately ⏎ * per discussion with NVIDIA, FlashInfer kernels are better for TP DSR1 anyways ⏎ * rather than fixing the IMA, we will just remove this kernel ⏎ * remove to simplify code ⏎  ⏎ ## Test Plan ⏎ * ci to ensure nothing broke ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-ac1c934276  (L3, 2025-12-19, sha ac1c93427616, PR #30974)
TITLE: [Bugfix] Fix incorrect tiles creation for mm prefix triton attention (#30974)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+10/-4)
LABELS: ready
BODY: ## Purpose ⏎ - Actually, #30386's implementation is still incorrect, because the tiles are still selected casually, so the hidden_states is not fully converged. ⏎ - This PR fixes this issue to make sure correct all valid tiles are selected for computaion ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ python examples/offline_inference/vision_language.py -m gemma3 --num-prompts 1 ⏎ ``` ⏎  ⏎ ## Test Result ⏎ **Main** ⏎ ``` ⏎ (EngineCore_DP0 pid=350376) qkv in 0.0172119140625 -0.044921875 0.092 …[truncated]

### L3-b5545d9d5c  (L3, 2025-12-19, sha b5545d9d5cab, PR #30887)
TITLE: [Bugfix] [Kernel] Triton attention kernels: mask out V blocks that fall outside sliding window (#30887)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+12/-0)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ There is currently a bug in the Triton attention kernels where we don't correctly mask out V blocks that fall out of the sliding window. On main, we can be reading garbage blocks (that may even contain NaN values) which will corrupt the output. This PR resolves it ⏎  ⏎ Potentially fix: ⏎ - https://github.com/vllm-project/vllm/issues/29998 ⏎ - https://github.com/vllm-project/vllm/issues/26480 ⏎  ⏎ ## Test Plan ⏎  ⏎ Server: ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=T …[truncated]

### L3-d52c5096d7  (L3, 2025-12-20, sha d52c5096d730, PR #30869)
TITLE: [Bugfix] fix the alias bug of AttentionBackendEnum when register CUSTOM attention backend to vllm (#30869)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.dispatch.registry
FILES: vllm/attention/backends/registry.py (+4/-2); tests/test_attention_backend_registry.py (+169/-0)
LABELS: rocm, ready
BODY: The bug is the TORCH_SDPA is the alias of the CUSTOM backend in AttentionBackendEnum because both of them don't have the value. When you register your attention backend as below, it will be unexpectedly registered to TORCH_SDPA: ⏎ ``` ⏎ import torch ⏎ from vllm.attention.backends.registry import register_backend, AttentionBackendEnum ⏎ from vllm.attention.backends.abstract import AttentionType ⏎ from vllm.attention.selector import get_attn_backend ⏎ from vll …[truncated]

### L3-06d490282f  (L3, 2025-12-21, sha 06d490282f2b, PR #30897)
TITLE: [NVFP4][Perf] Tune NVFP4 input quant kernel for small batch size (#30897)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/bench_nvfp4_quant.py (+177/-0); csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu (+4/-1); csrc/quantization/fp4/nvfp4_experts_quant.cu (+14/-17); csrc/quantization/fp4/nvfp4_quant_kernels.cu (+20/-42); csrc/quantization/fp4/nvfp4_utils.cuh (+28/-37)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ We discovered that the nvfp4 input quant operation was severely bottlenecking performance for dense models at small batch sizes. While investigating the flashinfer impl, I noticed that there is logic to [increase the grid size for small M](https://github.com/flashinfer-ai/flashinfer/blob/f0592411a3478d0548d62f7c877046499caf8d95/csrc/nv_internal/cpp/kernels/quantization.cu), based on the swizzled layout to help out with the padding ops …[truncated]

### L3-42b42824ae  (L3, 2025-12-21, sha 42b42824ae82, PR #31115)
TITLE: [Misc] Fix grammar errors in comments and messages (#31115)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/merge_attn_states.py (+1/-1); tests/quantization/test_compressed_tensors.py (+3/-3)
BODY: ## Summary ⏎ Fix grammar errors where "is not support" should be "is not supported" or "does not support": ⏎  ⏎ - `vllm/attention/ops/merge_attn_states.py`: "is not support for FP8" → "does not support FP8" ⏎ - `tests/quantization/test_compressed_tensors.py`: "is not support on ROCm" → "is not supported on ROCm" (3 occurrences) ⏎  ⏎ ## Test plan ⏎ - No functional changes, only grammar corrections in comments and skip messages ⏎ - Verified files parse correctly w …[truncated]

### L3-19cc9468fd  (L3, 2025-12-21, sha 19cc9468fd0f, PR #30957)
TITLE: [Feature]: Support NVIDIA ModelOpt HF FP8 variants FP8_PER_CHANNEL_PER_TOKEN and FP8_PB_WO  in vLLM (#30957)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/features/quantization/modelopt.md (+31/-0); tests/quantization/test_modelopt.py (+141/-0); vllm/config/model.py (+12/-6); vllm/model_executor/layers/linear.py (+2/-0); vllm/model_executor/layers/quantization/modelopt.py (+251/-9)
LABELS: documentation, frontend, ready, nvidia
BODY: ## Purpose ⏎  ⏎   This PR adds support for two NVIDIA ModelOpt-exported FP8 HuggingFace checkpoint variants that currently cannot be loaded by vLLM: ⏎  ⏎   - quant_algo=FP8_PER_CHANNEL_PER_TOKEN (aka hf_fp8_pc_pt) ⏎   - quant_algo=FP8_PB_WO (ModelOpt may emit fp8_pb_wo, handled case-insensitively) ⏎  ⏎   It keeps strict quant_algo matching (only known ModelOpt algos map to quantization="modelopt") to avoid ⏎   conflating other FP8 implementations. ⏎  ⏎   Fixes/Addre …[truncated]

### L3-a5bc77c253  (L3, 2025-12-22, sha a5bc77c253a6, PR #31040)
TITLE: [AMD][CI] Add "V1 Test e2e + engine" to mi325_8 Agent Pool (#31040)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+3/-1)
LABELS: rocm, ready, ci/build
BODY: We have seen two failures in the past week for the`V1 Test e2e + engine` test group in AMD CI for which there is no error message, and the test group passes after a retry. The only commonality between both occurrences of this issue is that it failed during `v1/e2e/test_spec_decode.py::test_eagle_correctness` in different test cases involving `llama4_eagle_mm`. Buildkite shows the failures with an "agent lost" error.  Here is one example from yest …[truncated]

### L3-de71747655  (L3, 2025-12-22, sha de7174765566, PR #29845)
TITLE: [SpecDecode] Simplified alternative padded-speculation acceptance rate fix (#29845)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.flashattn, L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/common.py (+15/-9); vllm/v1/attention/backends/mla/flashattn_mla.py (+2/-3); vllm/v1/attention/backends/mla/flashmla.py (+1/-1); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+1/-1); tests/v1/spec_decode/test_eagle.py (+8/-2); vllm/v1/spec_decode/eagle.py (+23/-3); vllm/v1/spec_decode/utils.py (+2/-0); vllm/v1/worker/gpu_model_runner.py (+10/-6)
LABELS: rocm, speculative-decoding, ready, v1
BODY: Alternative fix to: https://github.com/vllm-project/vllm/pull/26498 ⏎  ⏎ ## Benchmark Results (H100 + FA3) ⏎  ⏎ Benchmarked on SpecBench dataset with EAGLE3 + Llama-3.1-8B-Instruct using `examples/offline_inference/spec_decode.py`. ⏎  ⏎ ### Commands ⏎  ⏎ ```bash ⏎ # Benchmark script (bench_spec_decode.sh) ⏎ python3 examples/offline_inference/spec_decode.py \ ⏎   --method eagle3 --tp 1 --num-spec-tokens 3 \ ⏎   --dataset-name spec_bench --dataset-path datasets/specbench. …[truncated]

### L3-5312a7284e  (L3, 2025-12-22, sha 5312a7284e58, PR #31173)
TITLE: [Bug] Fix `'CutlassMLAImpl' object has no attribute '_workspace_buffer'` (#31173)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+7/-3)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study correctness case vllm:5312a7284e: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎  ⏎ `export MODEL="deepseek-ai/DeepSeek-V3.1"` ⏎ `export VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1` ⏎ `vllm serve $MODEL -tp 8   --port 9256 --enable-expert-parallel --enforce_eager` ⏎ will raise error ⏎  ⏎ ```bash ⏎  ⏎ (Worker_TP1_EP1 pid=1533438) ERROR 12-22 10:52:18 [multiproc_executor.py:824]   File "/home/wentao/vllm-source/vllm/v1/attention/backends/mla/common.py", line 1498, in _run_prefill_new_tokens_trtllm_ragged ⏎ (Worker_TP1_EP1 pid=1533438) E …[truncated]

### L3-3e10262356  (L3, 2025-12-22, sha 3e1026235665, PR #31197)
TITLE: Revert "[SM100] Enable fp8 compute for prefill MLA (#30746)" (#31197)
SOURCES: path_core, subject_keyword, corpus:confirmed-reverts
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+14/-112); tests/v1/attention/test_mla_backends.py (+3/-4); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+1/-0)
LABELS: v1, nvidia
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 30746 reason=premature_or_process
BODY: This reverts commit b10f41c894bc55c8dd8234a40edd1971be762805. ⏎  ⏎ ## Purpose ⏎ Revert #30746 untill #30993 is merged.  ⏎  ⏎ ## Test Plan ⏎ Regular CI  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-b10f41c894  (L3, 2025-12-22, sha b10f41c894bc, PR #30746)
TITLE: [SM100] Enable fp8 compute for prefill MLA (#30746)
SOURCES: path_core, subject_keyword, corpus:confirmed-reverts(reverted), corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+113/-14); tests/v1/attention/test_mla_backends.py (+4/-3); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+0/-1)
LABELS: documentation, rocm, ready, ci/build, v1, multi-modality, tool-calling, deepseek, nvidia
DEEP_STUDY: deep-study: this PR was reverted by PR 31197 (confirmed_revert, reason=premature_or_process) || deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ Adds a hook to run FP8 Prefill for Flashinfer Cutlass FMHA and TRTLLM Ragged Prefill Kernels for the Latent Attention models that use variable sequence length Prefill Kernels for SM100+. ⏎ This is triggered when `--kv-cache-dtype fp8` is enabled, similar to the decode path.  ⏎  ⏎ Depends on https://github.com/flashinfer-ai/flashinfer/pull/2047/ ⏎  ⏎ This is step 0. To avoid the additional fp8 quant ops for q,k,v tensors, I plan on adding a rope …[truncated]

### L3-8cef137689  (L3, 2025-12-22, sha 8cef13768917, PR #31153)
TITLE: [Chore] Update more locations to use `attention_config.backend` (#31153)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: benchmarks/benchmark_batch_invariance.py (+1/-1); tests/compile/distributed/test_fusions_e2e.py (+2/-1)
LABELS: performance, ready
BODY: ## Purpose ⏎  ⏎ Missed some places from #26315 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-f32cfd7d97  (L3, 2025-12-23, sha f32cfd7d9797, PR #26575)
TITLE: [ROCm][FEAT] Support AITER RMSNorm quantization fusion pass  (#26575)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/test_fusion.py (+189/-127); vllm/_aiter_ops.py (+144/-13); vllm/compilation/matcher_utils.py (+103/-11); vllm/compilation/pass_manager.py (+4/-2); vllm/compilation/rocm_aiter_fusion.py (+223/-66)
LABELS: rocm, ready, ci/build
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ This PR supports fusion pass for ROCM AITER by fusing `+rms_norm`, aiter rmsnorm ops, and `+quant_fp8`, vllm quantization custom ops. ⏎  ⏎ **Benchmark Result** ⏎  ⏎ | Metric                              | Without Fusion Pass | With Fusion Pass   | ⏎ |--------------------------------------|---------------------|-------------------| ⏎ | Successful requests                  | 500                 | 500               | ⏎ | Benchmark duration (s)         …[truncated]

### L3-3faa8bee57  (L3, 2025-12-23, sha 3faa8bee5798, PR #31095)
TITLE: adapt voxtral (#31095)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+9/-0); tests/models/multimodal/generation/test_voxtral.py (+1/-0); tests/models/registry.py (+5/-0); vllm/config/model.py (+4/-0); vllm/model_executor/models/interfaces.py (+10/-0); vllm/model_executor/models/registry.py (+4/-0); vllm/model_executor/models/voxtral.py (+29/-11); vllm/model_executor/models/voxtral_streaming.py (+243/-0); vllm/model_executor/models/whisper.py (+91/-81); vllm/model_executor/models/whisper_utils.py (+299/-0); (+2 more)
LABELS: new-model, ready, v1, multi-modality
BODY: adapts voxtral ⏎  ⏎ `python3 -m pytest tests/models/multimodal/generation/test_voxtral.py` are all still passing. ⏎ Needs: https://github.com/mistralai/mistral-common/pull/172

### L3-a37328fc5c  (L3, 2025-12-23, sha a37328fc5c6c, PR #30097)
TITLE: [Feature] Batch invariant: Lora (#30097)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/lora/ops/triton_ops/utils.py (+8/-2)
LABELS: ready
BODY: ## Purpose ⏎ This PR extracts the LoRA-related changes from the  PR #30018 and submits them separately. ⏎ The  #30018 will now focus only on the FA2 implementation. ⏎  ⏎ ## Test Plan ⏎ The testing procedure remains the same as described in the #30018. ⏎  ⏎ ## Test Result ⏎ All test results can also be found in the #30018. ⏎  ⏎ --- ⏎ [details omitted]

### L3-4ed11105d7  (L3, 2025-12-23, sha 4ed11105d7b8, PR #30967)
TITLE: [Misc] Remove unused custom ops `copy_blocks` and `copy_blocks_mla` (#30967)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+0/-88); csrc/cache.h (+0/-10); csrc/torch_bindings.cpp (+0/-10); tests/kernels/attention/test_cache.py (+0/-154); vllm/_custom_ops.py (+0/-12); vllm/_ipex_ops.py (+0/-12)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Found that some defined ops are no longer in use, so need remote it. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-369f47aa0f  (L3, 2025-12-23, sha 369f47aa0f1c, PR #31047)
TITLE: [DeepSeek v3.2] Remove unnecessary syncwarps (#31047)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+1/-6)
LABELS: ready, deepseek
BODY: ## Purpose ⏎ `indexer_k_quant_and_cache_kernel` has two unnecessary `__syncwarp` calls (the synchronization is already guaranteed by `__shfl_xor_sync`). This PR removes them. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ vllm serve deepseek-ai/DeepSeek-V3.2 -tp 8 --enable-expert-parallel ⏎ ``` ⏎ with ⏎ ``` ⏎ lm_eval --model local-completions --model_args base_url=http://localhost:8000/v1/completions,model=deepseek-ai/DeepSeek-V3.2,num_concurrent=128,tokenized_requests=False --tasks g …[truncated]

### L3-bfa2c0bbb9  (L3, 2025-12-23, sha bfa2c0bbb9b4, PR #31203)
TITLE: [ROCm][Bugfix] Fix RuntimeError in MMEncoderAttention by replacing .view() with .reshape() (#31203)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/attention/layers/mm_encoder_attention.py (+2/-2); tests/models/multimodal/conftest.py (+1/-1)
LABELS: rocm, ready, multi-modality
BODY: Fixes a `RuntimeError` in `MMEncoderAttention` when the output tensor from attention operations is non-contiguous. ⏎  ⏎ ## Problem ⏎  ⏎ The `_forward_sdpa` and `_forward_fa` methods use `.view()` to reshape the output tensor, which fails when the tensor is not contiguous in memory: ⏎ ``` ⏎ RuntimeError: view size is not compatible with input tensor's size and stride  ⏎ (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead. ⏎ ` …[truncated]

### L3-0247a91e00  (L3, 2025-12-23, sha 0247a91e00d6, PR #28979)
TITLE: [ROCm][CI] Fix entrypoints tests and Python-only installation test on ROCm (#28979)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: setup.py (+126/-26); tests/entrypoints/openai/conftest.py (+24/-0); tests/entrypoints/openai/test_chat.py (+5/-5); tests/entrypoints/openai/test_optional_middleware.py (+1/-0); tests/entrypoints/openai/test_response_api_with_harmony.py (+6/-1); tests/entrypoints/openai/test_serving_tokens.py (+25/-4); tests/entrypoints/openai/test_shutdown.py (+23/-11); tests/entrypoints/openai/test_transcription_validation.py (+16/-4); tests/entrypoints/openai/test_translation_validation.py (+25/-6); tests/entrypoints/openai/test_video.py (+16/-1); (+16 more)
LABELS: rocm, frontend, ready, ci/build, v1, gpt-oss
BODY: This PR fixes ROCm CI failures by: ⏎  ⏎ 1. **Setting correct attention backends for entrypoints tests on ROCm** ⏎    - Encoder-only models (embeddings/pooling) require `FLEX_ATTENTION` backend as it's the only attention backend that supports encoder-only self-attention on ROCm ⏎    - Audio transcription/translation tests require `ROCM_AITER_FA` backend ⏎  ⏎ 2. **Fixing Python-only installation test on ROCm** ⏎  ⏎ ## Test Plan ⏎  ⏎ ROCm CI tests should pass after thi …[truncated]

### L3-e42894f5b5  (L3, 2025-12-24, sha e42894f5b530, PR #31235)
TITLE: [ROCm][CI][Bugfix] Fix Siglip2 rotary embedding dispatch and InternVL video test tolerance (#31235)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/generation/test_common.py (+1/-0); vllm/model_executor/models/siglip2navit.py (+3/-1)
LABELS: rocm, ready, multi-modality
BODY: This PR fixes two ROCm-related issues: ⏎  ⏎ 1. **Siglip2 rotary embedding dispatch bug**: The `apply_rotary_pos_emb` function in `siglip2.py` had inverted logic for selecting the rotary embedding implementation. It was calling `forward_cuda` when `not current_platform.is_cuda()`, which is incorrect. ⏎  ⏎ 2. **InternVL video test flakiness on ROCm**: The `intern_vl-video` test was failing due to minor numerical precision differences between HF and vLLM ou …[truncated]

### L3-254f6b9867  (L3, 2025-12-25, sha 254f6b986720, PR #31241)
TITLE: [Bugfix] Fix eagle dp tests on A100 (#31241)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/distributed/test_eagle_dp.py (+7/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎ `TP_SIZE=1 DP_SIZE=2 pytest -v -s tests/v1/distributed/test_eagle_dp.py` fails on A100 for me before this PR. ⏎  ⏎ Here's what I think is happening: ⏎ - the test is checking that the tokens produced by a model with eagle is identical to a model without eagle ⏎ - the model with eagle uses a draft model to produce draft tokens ⏎ - the target model takes all of the draft tokens and then does a forward pass to see how many of the tokens to accept/re …[truncated]

### L3-c79dbfa9ad  (L3, 2025-12-26, sha c79dbfa9ad2f, PR #31324)
TITLE: [CI] Fix flaky vision beam search test with flexible semantic validation (#31324)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/entrypoints/openai/test_vision.py (+50/-48)
LABELS: ready
BODY: Replaces brittle exact string matching in `test_single_chat_session_image_base64encoded_beamsearch` with flexible semantic term validation, fixing intermittent test failures and rendering the test platform-agnostic and compatible with other attention backends as well. ⏎  ⏎ ## Changes ⏎  ⏎ - Remove platform-specific expected results (`EXPECTED_MM_BEAM_SEARCH_RES`, `EXPECTED_MM_BEAM_SEARCH_RES_ROCM`) ⏎ - Add `REQUIRED_BEAM_SEARCH_TERMS` with AND/OR logic for …[truncated]

### L3-727c41f3fd  (L3, 2025-12-27, sha 727c41f3fd7c, PR #31169)
TITLE: [MoE Refactor][10/N] Cleanup Fp8 Process Weights After Loading (#31169)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/fp8.py (+150/-194)
LABELS: rocm, ready, nvidia
BODY: ## Purpose ⏎ SUMMARY: ⏎ * Refactoring fp8.py ⏎ * Clean up `process_weights_after_loading` by removing branches for block vs non-block ⏎ * make methods into helpers that can be called from online too ⏎  ⏎ FOLLOW UP: ⏎ * update FlashInfer FP8 to be able to be called from fp8.py [there are no llama models that exist) ⏎ * investigate failures of FlashInfer Fp8 Per-tensor (also does not work on main) ⏎ * update the marlin reshuffle and flashinfer register scales to be  …[truncated]

### L3-573dd0e6f0  (L3, 2025-12-28, sha 573dd0e6f089, PR #31327)
TITLE: [ROCm] Migrate xgrammar to upstream release (#31327)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/common.txt (+1/-1); requirements/rocm-test.txt (+0/-4)
LABELS: rocm, ci/build, ready-run-all-tests
BODY: This PR updates the xgrammar dependency in `requirements/rocm-test.txt` from a custom fork to the upstream PyPI release. ⏎  ⏎ ### Changes ⏎ - Replaced `xgrammar @ git+https://github.com/divakar-amd/xgrammar@3272f7c...` with `xgrammar==0.1.29` ⏎  ⏎ ### Motivation ⏎ It's better practice to use upstream repositories instead of maintaining custom forks when possible. This is a follow-up to a comment from #29702 to eventually migrate to the upstream xgrammar repo …[truncated]

### L3-62def07d67  (L3, 2025-12-28, sha 62def07d6786, PR #31395)
TITLE: [BugFix] register quant scale tensors as buffer (#31395)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+82/-16)
LABELS: ready
ISSUES: #31377 [Bug]: torch.ops._C.static_scaled_fp8_quant IMA error
BODY: Prior to this pr, `pytest -sv "tests/compile/test_fusion_attn.py::test_attention_quant_pattern[AttentionBackendEnum.TRITON_ATTN-nvidia/Llama-4-Scout-17B-16E-Instruct-FP8-TestAttentionFp8StaticQuantPatternModel-+quant_fp8-dtype1-256-128-40-8]"` leads to illegal memory access error. I found that the IMA comes from `torch.ops._C.static_scaled_fp8_quant` as described in #31377. ⏎  ⏎ @lengrongfu pointed out the issue is that scale tensor is not on GPU so  …[truncated]

### L3-d63b969675  (L3, 2025-12-29, sha d63b969675f1, PR #31187)
TITLE: [CI/ROCm] Fixing "V1 Test attention (H100)" test group. (#31187)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/attention/test_attention_backends.py (+29/-9); tests/v1/attention/test_rocm_attention_backends_selection.py (+39/-10); tests/v1/attention/test_sparse_mla_backends.py (+4/-0)
LABELS: rocm, ready, v1
BODY: [CI/ROCm] Fixing "V1 Test attention (H100)" test group.

### L3-3f52fa5aa2  (L3, 2025-12-30, sha 3f52fa5aa26d, PR #28775)
TITLE: [Model] Add support for openPangu moe model (#28775)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.dispatch.registry
FILES: vllm/attention/backends/registry.py (+3/-0); vllm/attention/layer.py (+11/-4); vllm/attention/layers/static_sink_attention.py (+254/-0); vllm/attention/ops/triton_reshape_and_cache_flash.py (+171/-0); vllm/v1/attention/backends/flash_attn_diffkv.py (+269/-0); docs/models/supported_models.md (+1/-0); tests/models/registry.py (+5/-0); vllm/model_executor/models/openpangu.py (+329/-3); vllm/model_executor/models/registry.py (+1/-0); vllm/v1/core/single_type_kv_cache_manager.py (+26/-0); (+1 more)
LABELS: documentation, new-model, ready, v1
BODY: ## Purpose ⏎ This PR adds support for openpangu moe model, which characterizes by its different kv head size and sink kv in attention. ⏎  ⏎ The model has two new features that have not been supported: ⏎ 1. Different key head size and value head size. ⏎ Although flash_attn kernel can handle the different kv head size, current vllm framwork does not make it optional choice. In the implemented `FlashSinkAttentionBackend`, the kv_cache_shape is defined to be o …[truncated]

### L3-357d435c54  (L3, 2025-12-30, sha 357d435c549d, PR #31390)
TITLE: [Bug] Fix log issue with `\n` (#31390)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_sparse
FILES: vllm/v1/attention/backends/mla/flashmla_sparse.py (+2/-2)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ `(Worker_TP5_EP5 pid=1048443) WARNING 12-26 15:57:52 [flashmla_sparse.py:932] padding num_heads to 64                     due to sparse attn kernel requirement` ⏎  ⏎ Now should be fixed with  ⏎  ⏎ `(Worker_TP5_EP5 pid=1048443) WARNING 12-26 15:57:52 [flashmla_sparse.py:932] padding num_heads to 64 due to sparse attn kernel requirement`

### L3-cf16342d43  (L3, 2025-12-31, sha cf16342d435f, PR #31551)
TITLE: [ROCm][CI] Update MiniCPM model test: MiniCPM3-4B to MiniCPM4.1-8B and simplify attention backend testing (#31551)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/language/generation/test_common.py (+16/-7); tests/models/registry.py (+3/-0)
LABELS: rocm, ready
BODY: This PR updates the language model generation tests for ROCm CI by upgrading the MiniCPM model from v3 to v4.1, removing explicit AITER kernel testing in favor of default backend dispatching, and adding embedding scaling support for MiniCPM models. ⏎  ⏎ ### Changes ⏎  ⏎ #### 1. Model Upgrade: MiniCPM3-4B to MiniCPM4.1-8B ⏎  ⏎ The previous model `openbmb/MiniCPM3-4B` is incompatible with recent versions of `transformers` due to an outdated `DynamicCache` API  …[truncated]

### L3-825c2dc133  (L3, 2026-01-01, sha 825c2dc133d4, PR #31282)
TITLE: [Bugfix][Hardware][AMD] Fix last_page_len calculation in AITER MLA decode (#31282)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+12/-9)
LABELS: rocm, ready, v1
BODY: ## Summary ⏎  ⏎ Fixes incorrect `paged_kv_last_page_len` calculation in the ROCm AITER MLA decode path. ⏎  ⏎ ### The Bug ⏎  ⏎ The AITER MLA kernel uses a block size of 1 (each page contains exactly 1 token). However, the `paged_kv_last_page_len` was incorrectly set to the full sequence length: ⏎  ⏎ ```python ⏎ # BEFORE (buggy): ⏎ paged_kv_last_page_len = torch.where(seq_lens_device == 0, 1, seq_lens_device) ⏎ ``` ⏎  ⏎ For a sequence of 127 tokens, this would set `last_pag …[truncated]
