### L3-de92d916fe  (L3, 2025-10-15, sha de92d916fe8a, PR #26107)
TITLE: [NVIDIA] Add support for cudnn fp4 gemm via flashinfer (#26107)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+11/-6); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py (+25/-15); vllm/model_executor/layers/quantization/modelopt.py (+21/-17)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Add support for the cuDNN FP4 GEMM through FlashInfer. ⏎ This operator is primarily used in the shared expert and output projection layers (currently available only in nvidia/DeepSeek-R1-FP4-v2). ⏎ Preliminary latency benchmarks indicate an ~1% end-to-end performance improvement. ⏎  ⏎ To enable this feature, users need to install the required dependencies and set the environment variable: ⏎ ``` ⏎ pip install nvidia-cudnn-cu12 ⏎ pip install nvidia-c …[truncated]

### L3-0a9ef0cfce  (L3, 2025-10-15, sha 0a9ef0cfce13, PR #26534)
TITLE: Move query quantization to attention layer for Flashinfer & Triton. (#26534)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+16/-8); vllm/attention/layer.py (+6/-3); vllm/v1/attention/backends/flash_attn.py (+3/-1); vllm/v1/attention/backends/flashinfer.py (+12/-10); vllm/v1/attention/backends/triton_attn.py (+3/-15); tests/compile/test_fusion_attn.py (+3/-1)
LABELS: ready, v1
BODY: ### Purpose ⏎ Implements refactor of quantization to the attention layer for triton and flashinfer, resolves feature request [#25584](https://github.com/vllm-project/vllm/issues/25584) ⏎  ⏎ ### Test Plan ⏎ Spin up server: ⏎  ⏎ Flashinfer: ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=FLASHINFER  vllm serve meta-llama/Llama-3.1-8B-Instruct \ ⏎   --kv-cache-dtype fp8 \ ⏎   --compilation-config '{"compile_sizes": [1,2,4,8], "cudagraph_capture_sizes": [1,2,4,8], "cudagraph_mode": "FUL …[truncated]

### L3-7d8975de84  (L3, 2025-10-15, sha 7d8975de84da, PR #26609)
TITLE: Deepseek-v3 Batch Invariant on 8xH100 (#26609)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.flashattn
FILES: vllm/model_executor/layers/mla.py (+1/-0); vllm/v1/attention/backends/flash_attn.py (+12/-1); vllm/v1/attention/backends/mla/common.py (+10/-1); vllm/v1/attention/backends/mla/flashattn_mla.py (+11/-1); vllm/v1/attention/backends/mla/flashmla.py (+36/-2); vllm/v1/attention/backends/mla/triton_mla.py (+6/-1); tests/v1/generation/test_batch_invariance.py (+769/-73); tests/v1/generation/test_rms_norm_batch_invariant.py (+346/-0); vllm/compilation/caching.py (+3/-1); vllm/config/model.py (+7/-0); (+11 more)
LABELS: ready, ci/build, v1, deepseek, gpt-oss
BODY: This PR replaces https://github.com/vllm-project/vllm/pull/26136 with a far more rigorous set of tests and implementation choices.  It is, somewhat unfortunately, quite big.  I will be trying to make this smaller over the weekend but would appreciate some initial eyes!  (also I'll do a writeup because this was quite a journey to get working and I think folks would benefit from that) ⏎  ⏎ Adds support for FLASH_ATTN, rms_norm, batched matmul, linear,  …[truncated]

### L3-314fa8abbf  (L3, 2025-10-16, sha 314fa8abbf9d, PR #26846)
TITLE: [Attention] Tune CUTLASS MLA num_splits (#26846)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.cutlass_sm100
FILES: csrc/attention/mla/cutlass_sm100_mla/device/sm100_mla.hpp (+21/-16)
LABELS: ready
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Purpose ⏎ Tune the num_splits heuristic for CUTLASS_MLA to achieve some speedup now that #26026 has fixed the hang. Based on experiments performed using the tools introduced in #26835, this is the optimal num_splits policy: ⏎  ⏎ <img width="3331" height="2363" alt="numsplits_heatmap" src="https://github.com/user-attachments/assets/c2898b4c-2ff2-4965-a760-9bc71446e945" /> ⏎  ⏎ Following the optimal policy would yield this speedup: ⏎  ⏎ <img width="3331" heig …[truncated]

### L3-5afd3276df  (L3, 2025-10-16, sha 5afd3276dfd7, PR #26870)
TITLE: [Feature] Add process_weights_after_loading to AttentionImpl (#26870)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+3/-0); vllm/attention/layer.py (+1/-10); vllm/v1/attention/backends/flashinfer.py (+5/-0)
LABELS: ready, v1
ISSUES: #26817 [Feature]: Add process_weights_after_loading to AttentionImpl
BODY: ## Purpose ⏎  ⏎ FIX: https://github.com/vllm-project/vllm/issues/26817 ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ $ VLLM_ATTENTION_BACKEND=FLASHINFER python3-m vllm.entrypoints.cli.main serve Qwen/Qwen3-0.6B ⏎ ``` ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-785d8b6410  (L3, 2025-10-16, sha 785d8b6410c3, PR #26437)
TITLE: [PERF] Qwen3-next MTP speedup (change bool mask indexing to index_select / index_copy to reduce d2h) (#26437)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fla/ops/utils.py (+1/-1); vllm/model_executor/models/qwen3_next.py (+12/-12); vllm/v1/attention/backends/gdn_attn.py (+43/-23)
LABELS: ready, v1, qwen
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ Qwen3-next MTP suffered from d<->h memory transfers on prefill phase. That makes MTP slower than STM (at least for some inputs).  ⏎  ⏎ This PR fixes the following ⏎ 1. Remove boolean indexing from GDN. Change it on direct indexing. (`tensor[bool_mask]` cause d2h transfer of number of `True` element needed for creation resulting tensor). ⏎ 2. Increase cache size in `@tensor_cache`. There are 2 calls for every GDN attn, they are grouped by 3. We …[truncated]

### L3-41d3071918  (L3, 2025-10-16, sha 41d3071918bd, PR #26714)
TITLE: [NVIDIA] [Perf] Update to leverage flashinfer trtllm FP4 MOE throughput kernel (#26714)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+2/-2); docker/Dockerfile.nightly_torch (+2/-2); requirements/cuda.txt (+1/-1); tests/kernels/moe/test_ocp_mx_moe.py (+7/-7); vllm/model_executor/layers/fused_moe/trtllm_moe.py (+1/-28); vllm/model_executor/layers/quantization/modelopt.py (+4/-17); vllm/model_executor/layers/quantization/mxfp4.py (+8/-39)
LABELS: ready, ci/build
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ - Update flashinfer version to 0.4.1 to leverage latest flashinfer trtllm throughput moe kernel.  ⏎ - Change use of falshinfer `trtllm_fp4_block_scale_moe`. Remove the tile_tokens_dim, pass None so flashinfer can calculate itself ⏎  ⏎ ## Test Plan ⏎ `lm_eval --model local-completions --tasks gsm8k --model_args model=openai/gpt-oss-120b,base_url=http://0.0.0.0:30000/v1/completions,max_retries=3,tokenized_requests=False,timeout=1200,max_gen_tok …[truncated]

### L3-b2f78cbad4  (L3, 2025-10-16, sha b2f78cbad4e8, PR #26855)
TITLE: [small][batch invariance] Rename the env and internal flags to simplify usage (#26855)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.flashattn, L3.flex_attention
FILES: vllm/v1/attention/backends/flash_attn.py (+5/-5); vllm/v1/attention/backends/flashinfer.py (+3/-3); vllm/v1/attention/backends/flex_attention.py (+2/-2); vllm/v1/attention/backends/mla/common.py (+2/-2); vllm/v1/attention/backends/mla/flashattn_mla.py (+3/-3); vllm/v1/attention/backends/mla/flashmla.py (+2/-2); vllm/v1/attention/backends/mla/triton_mla.py (+2/-2); csrc/core/batch_invariant.hpp (+4/-4); csrc/layernorm_kernels.cu (+2/-2); csrc/layernorm_quant_kernels.cu (+1/-1); (+10 more)
LABELS: ready, v1, gpt-oss
BODY: Environment variable VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT  -> VLLM_BATCH_INVARIANT ⏎ Internal flag: vllm_kernel_override_batch_invariant() -> vllm_is_batch_invariant() ⏎  ⏎ ## Purpose ⏎  ⏎ Simplify usage across the stack. ⏎  ⏎ ## Test Plan ⏎  ⏎ Rebuild for the C++ component: ⏎ ``` ⏎ CCACHE_NOHASHDIR="true"  uv pip install -e . --no-build-isolation -v ⏎ ``` ⏎  ⏎ `pytest -s -v tests/v1/generation/test_batch_invariance.py -k test_logprobs_bitwise_batch_invariance_bs1_vs_bsN` ⏎  ⏎  …[truncated]

### L3-4d4d6bad19  (L3, 2025-10-17, sha 4d4d6bad1981, PR #27022)
TITLE: [Chore] Separate out `vllm.utils.importlib` (#27022)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.selector, L3.dispatch.registry
FILES: vllm/attention/backends/registry.py (+1/-1); vllm/attention/selector.py (+2/-1); tests/model_executor/model_loader/tensorizer_loader/test_tensorizer.py (+1/-1); tests/utils_/test_import_utils.py (+46/-0); tests/utils_/test_utils.py (+0/-41); tests/v1/attention/utils.py (+1/-1); vllm/assets/audio.py (+1/-1); vllm/assets/video.py (+1/-1); vllm/benchmarks/datasets.py (+1/-1); vllm/compilation/backends.py (+2/-1); (+31 more)
LABELS: performance, structured-output, frontend, ready, v1, multi-modality, tool-calling
BODY: ## Purpose ⏎  ⏎ Part of https://github.com/vllm-project/vllm/issues/26900 ⏎  ⏎ - `vllm.utils.import_from_path -> vllm.utils.import_utils.import_from_path` ⏎ - `vllm.utils.resolve_obj_by_qualname -> vllm.utils.import_utils.resolve_obj_by_qualname` ⏎ - `vllm.utils.get_vllm_optional_dependencies -> vllm.utils.import_utils.get_vllm_optional_dependencies` ⏎ - `vllm.utils.PlaceholderModule -> vllm.utils.import_utils.PlaceholderModule` ⏎ - `vllm.utils.LazyLoader -> vll …[truncated]

### L3-6c9fdbf725  (L3, 2025-10-17, sha 6c9fdbf72581, PR #27091)
TITLE: [Docs] Replace `rst` style double-backtick with `md` single-backtick (#27091)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.platform.rocm_selection
FILES: vllm/utils/flashinfer.py (+10/-10); benchmarks/multi_turn/benchmark_serving_multi_turn.py (+1/-1); docs/models/extensions/fastsafetensor.md (+1/-1); tests/models/registry.py (+3/-3); tests/models/utils.py (+1/-1); tools/check_init_lazy_imports.py (+1/-1); vllm/assets/base.py (+1/-1); vllm/benchmarks/serve.py (+1/-1); vllm/compilation/decorators.py (+2/-2); vllm/config/pooler.py (+3/-3); (+21 more)
LABELS: documentation, performance, rocm, frontend, tpu, v1, multi-modality, gpt-oss
BODY: Only affects comments/docstrings/docs so can be force merged if docs build passes

### L3-fec2b341ad  (L3, 2025-10-17, sha fec2b341ad76, PR #26977)
TITLE: [Kernel] Lazy import FlashInfer (#26977)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/sample/test_topk_topp_sampler.py (+9/-8); vllm/v1/sample/ops/topk_topp_sampler.py (+16/-30)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ - `Importing flashinfer` adds considerable startup overhead. Since we have already disabled  fashinfer sampler by default(see: https://github.com/vllm-project/vllm/pull/26859), we can  use lazy import to avoid the startup overhead caused by importing flashinfer ⏎ - Additionally, considering that flashinfer (currently 0.4.0) is already a default dependency(see: https://github.com/vllm-project/vllm/pull/26443), when users explicitly set `V …[truncated]

### L3-f50cc221ea  (L3, 2025-10-17, sha f50cc221ea78, PR #27054)
TITLE: [Test] Make `test_failure` more stable for batch invariance (#27054)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/generation/test_batch_invariance.py (+20/-5)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Some of the models may fail in `test_failure` ⏎  ⏎ This PR make it more stable for the test by making the prompt more different. ⏎  ⏎ ## Test ⏎  ⏎ Tested on H100, with `export VLLM_USE_DEEP_GEMM=0` ⏎  ⏎ Origin ⏎  ⏎ ```bash ⏎ VLLM_TEST_MODEL=Qwen/Qwen3-30B-A3B-FP8 pytest test_batch_invariance.py ⏎  ⏎ FAILED test_batch_invariance.py::test_logprobs_WITHOUT_batch_invariance_should_FAIL[FLASH_ATTN] - Failed: ✗ UNEXPECTED: All 32 prompts matched between BS=1 and BS= …[truncated]

### L3-950cf9e58e  (L3, 2025-10-17, sha 950cf9e58eef, PR #27114)
TITLE: [Bugfix] Use PIECEWISE cudagraphs on Blackwell if max_model_len > 131072 (#27114)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+37/-15)
LABELS: bug, ready
ISSUES: #27057 [Bug]: Qwen3-VL broken on Blackwell with `PIECEWISE_AND_FULL`
BODY: ## Purpose ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/27057 ⏎  ⏎ The original issue was found because Qwen3-VL models completely lost accuracy (1% vs 86% on GSM8K) on B200 GPUs when using the default FULL_AND_PIECEWISE cudagraph_mode. The issue did not occur on Hopper at all, with PIECEWISE mode only, FlashAttention backend, or when explicitly disabling TRTLLM attention. ⏎  ⏎ Because TRTLLM attention is selected dynamically based on runtime conditi …[truncated]

### L3-d29483b58a  (L3, 2025-10-17, sha d29483b58a03, PR #27115)
TITLE: [Minor] Remove unnecessary error message (#27115)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+8/-27); vllm/model_executor/layers/linear.py (+11/-28)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Revert https://github.com/vllm-project/vllm/pull/20321 ⏎  ⏎ This is an unnecessary error message that complicates the model code. If there is an OOM error, a user can know which tensor is causing the OOM by pytorch's original error message. Also, OOM on a specific tensor doesn't have any actual meaning: it can be a previous tensor being very big, and then this tensor happens to break the GPU size limit. Error message here can even be conf …[truncated]

### L3-c312320764  (L3, 2025-10-17, sha c31232076419, PR #26663)
TITLE: [CI/Build] tests(v1): feed Triton attention the (num_blocks, 2, …) KV cache layout in backend-correctness tests (#26663)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/attention/test_attention_backends.py (+3/-2)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ ~~This PR makes K/V cache unbinding robust across cache layouts by detecting the axis of size 2 at runtime instead of assuming it sits at `dim=1`. This fixes unpacking errors seen when `kv_cache` is shaped with the K/V dimension elsewhere (e.g., `dim=0`).~~ ⏎  ⏎ When running tests `tests/v1/attention/test_attention_backends.py` on H100, the following line ⏎ ``` ⏎ key_cache, value_cache = kv_cache.unbind(1) ⏎ ``` ⏎ failed with `ValueError: too man …[truncated]

### L3-5c2acb270a  (L3, 2025-10-18, sha 5c2acb270aad, PR #27106)
TITLE: [Models][QwenVL] Remove unnecessary `.contiguous()` calls (#27106)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen2_5_vl.py (+1/-1); vllm/model_executor/models/qwen2_vl.py (+1/-1)
LABELS: qwen
BODY: ## Purpose ⏎  ⏎ This PR removes calls to `.contiguous()` in the `QwenVisionAttention` module. Tested with both xformers and flash attention backend. ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ vllm bench serve Qwen/Qwen2.5-VL-3B-Instruct ⏎  ⏎ vllm bench serve --backend openai-chat --model Qwen/Qwen2.5-VL-3B-Instruct --endpoint /v1/chat/completions --dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat --hf-split train --num-prompts 1000 ⏎ ``` ⏎  ⏎ ## Test Result ⏎  ⏎ ### Before ⏎  ⏎ <im …[truncated]

### L3-6ac5e06f7c  (L3, 2025-10-18, sha 6ac5e06f7c5d, PR #26908)
TITLE: [Chore] Clean up pytorch helper functions in `vllm.utils` (#26908)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.aiter_fa, L3.flex_attention
FILES: benchmarks/kernels/bench_per_token_quant_fp8.py (+2/-1); benchmarks/kernels/benchmark_activation.py (+2/-1); benchmarks/kernels/benchmark_layernorm.py (+2/-1); benchmarks/kernels/benchmark_paged_attention.py (+2/-2); benchmarks/kernels/benchmark_quant.py (+2/-1); benchmarks/kernels/benchmark_reshape_and_cache.py (+2/-2); benchmarks/kernels/benchmark_reshape_and_cache_flash.py (+2/-2); tests/compile/piecewise/test_full_cudagraph.py (+1/-1); tests/compile/piecewise/test_multiple_graphs.py (+1/-1); tests/compile/piecewise/test_simple.py (+1/-1); (+109 more)
LABELS: performance, rocm, tpu, ready, v1, multi-modality, llama, qwen, deepseek, kv-connector
BODY: ## Purpose ⏎ - Part fo #26900 ⏎ - Split pytorch related helper functions into single file  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-9f020f4f31  (L3, 2025-10-18, sha 9f020f4f3109, PR #27111)
TITLE: [BugFix] Fix failing gemma-3-1b-it test: `test_lm_eval_accuracy_v1_engine[google/gemma-3-1b-it]` (#27111)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build
BODY: vLLM side of: https://github.com/vllm-project/flash-attention/pull/102 ⏎  ⏎ Fix `pytest tests/entrypoints/llm/test_accuracy.py::test_lm_eval_accuracy_v1_engine[google/gemma-3-1b-it]` ⏎  ⏎ Now passes

### L3-ab4be40fc5  (L3, 2025-10-18, sha ab4be40fc5bd, PR #27035)
TITLE: [fix][cpu] fix prefill attention in CPU attention backend (#27035)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/engine/arg_utils.py (+9/-1); vllm/v1/attention/backends/cpu_attn.py (+7/-3)
LABELS: ready, v1
ISSUES: #27034 [Bug]: Incorrect outputs with batch size > 1 on AArch64 CPU
BODY: [fix][cpu] fix prefill attention in CPU attention backend ⏎  ⏎ - Disables prefix caching because prefill attention can't handle paged KV cache ⏎ - Fixes Q/K/V used during prefill on mixed prefill/decode requests ⏎  ⏎ ## Purpose ⏎ Fixes #27034 ⏎  ⏎ ## Test Plan ⏎ test script attached to #27034 ⏎ ## Test Result ⏎ Output of test script attached to #27034 is same when prompts are batched and when prompts are ran one at a time  ⏎  ⏎ --- ⏎ [details omitted]

### L3-b26b70bec4  (L3, 2025-10-18, sha b26b70bec4fa, PR #26587)
TITLE: [Misc] Refactor `get_kv_cache_spec` into `AttentionLayerBase` (#26587)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+53/-5); vllm/attention/layers/chunked_local_attention.py (+13/-0); vllm/attention/layers/cross_attention.py (+9/-1); vllm/attention/layers/encoder_only_attention.py (+6/-0); vllm/model_executor/layers/attention_layer_base.py (+11/-0); vllm/model_executor/layers/mamba/abstract.py (+29/-0); vllm/model_executor/models/deepseek_v2.py (+1/-1); vllm/utils/__init__.py (+9/-0); vllm/v1/spec_decode/eagle.py (+1/-1); vllm/v1/worker/gpu_model_runner.py (+19/-110)
LABELS: speculative-decoding, ready, v1, deepseek
BODY: This PR modifies the `AttentionLayerBase` interface to add a new `get_kv_cache_spec` method. ⏎ This allows different attention layers to define their own KV Cache spec, by making the spec entirely transparent to the Model Runner. ⏎  ⏎ As a consequence, the runner can now limit itself to collect the specs without having to handle different attention types and/or model-specific hacks such as the one for DSv32 Indexer.  ⏎ It also makes the code much simpler …[truncated]

### L3-c3a2c6ac5f  (L3, 2025-10-21, sha c3a2c6ac5f9a, PR #27061)
TITLE: [MM][Core] Decouple ViT backend from LM backend (#27061)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+10/-1); tests/config/test_multimodal_config.py (+25/-0); vllm/config/model.py (+5/-0); vllm/config/multimodal.py (+38/-4); vllm/engine/arg_utils.py (+9/-0); vllm/model_executor/models/dots_ocr.py (+17/-2); vllm/model_executor/models/ernie45_vl.py (+15/-1); vllm/model_executor/models/glm4_1v.py (+15/-1); vllm/model_executor/models/keye.py (+18/-1); vllm/model_executor/models/ovis2_5.py (+12/-0); (+6 more)
LABELS: ready, qwen
BODY: ## Purpose ⏎ Multimdal encoder attention backend for a selection of models have been adopting the same backend as the language model backbone (via env var). This has exposed challenges and inflexibilities since some backends may work for the LM but not for the multimodal encoder. ⏎  ⏎ This PR refactors the ViT backend selection for models that use `get_vit_attn_backend` so that users can specify `--mm-encoder-attn-backend` as an override. ⏎  ⏎ Model-specif …[truncated]

### L3-5ff5d94e77  (L3, 2025-10-21, sha 5ff5d94e7785, PR #26729)
TITLE: [Bugfix] Fix gpt-oss w4a8 DP/EP on B200 (#26729)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/quantization/test_blackwell_moe.py (+20/-0); vllm/model_executor/layers/fused_moe/config.py (+20/-0); vllm/model_executor/layers/quantization/mxfp4.py (+18/-0); vllm/model_executor/layers/quantization/utils/mxfp8_utils.py (+4/-1); vllm/model_executor/warmup/kernel_warmup.py (+20/-1)
LABELS: ready, gpt-oss
BODY: ## Purpose ⏎ Running  ⏎ ``` ⏎ VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1  VLLM_ALL2ALL_BACKEND="deepep_high_throughput"  vllm serve openai/gpt-oss-20b --data-parallel-size 2 --tensor-parallel-size 1 --enable-expert-parallel   --no-enable-prefix-caching  ⏎ ``` ⏎ From `main` runs into a few issues,  ⏎ 1. Requirement for `amd-quark`:  Note that despite explicitly asking for `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8`, i.e. MXFP4 weights and MXFP8 activations. The code pat …[truncated]

### L3-86ed77022d  (L3, 2025-10-21, sha 86ed77022db2, PR #27229)
TITLE: [Feature] Batch Invariant for R1 TP 8 on Blackwell (#27229)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/batch_invariant.py (+1/-1)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Batch Invariant for R1 TP 8 on Blackwell ⏎  ⏎ Backend: FlashinferMLA ⏎  ⏎ ## Test ⏎  ⏎ ```bash ⏎ VLLM_ATTENTION_BACKEND=FLASHINFER_MLA VLLM_TEST_TP_SIZE=8 VLLM_TEST_MODEL="deepseek-ai/DeepSeek-R1" pytest -s -v tests/v1/generation/test_batch_invariance.py -k test_logprobs_bitwise_batch_invariance_bs1_vs_bsN[FLASHINFER] ⏎  ⏎ PASSED ⏎  ⏎ == 1 passed, 6 deselected, 4 warnings in 120.24s (0:02:00) === ⏎ ```

### L3-bd66b8529b  (L3, 2025-10-21, sha bd66b8529bb0, PR #27262)
TITLE: [CI] Install pre-release version of `apache-tvm-ffi` for `flashinfer` (#27262)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+6/-2)
LABELS: ready, ci/build
BODY: Hotfix for CI ⏎  ⏎ --- ⏎  ⏎ Issue reported in https://github.com/flashinfer-ai/flashinfer/issues/1962 ⏎ Fix in https://github.com/flashinfer-ai/flashinfer/issues/1960

### L3-becb7de40b  (L3, 2025-10-21, sha becb7de40b29, PR #24994)
TITLE: Update PyTorch to 2.9.0+cu129 (#24994)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+2/-2); docker/Dockerfile (+12/-2); docker/Dockerfile.cpu (+4/-0); requirements/build.txt (+1/-1); requirements/cuda.txt (+5/-5); requirements/rocm-build.txt (+5/-5); requirements/test.in (+4/-4); requirements/test.txt (+19/-18); .buildkite/test-pipeline.yaml (+7/-2); .pre-commit-config.yaml (+1/-1); (+6 more)
LABELS: documentation, rocm, ready, ci/build, v1
ISSUES: #27208 [Feature]: Upgrade CUDA version to 12.9.1 in docker images
BODY: ## Purpose ⏎  ⏎ Upgrade PyTorch to 2.9.0+cu129 ⏎  ⏎ ## Test Plan ⏎  ⏎ CI ⏎  ⏎ --- ⏎ <details> ⏎ <summary> Essential Elements of an Effective PR Description Checklist </summary>

### L3-250fb1b8ea  (L3, 2025-10-21, sha 250fb1b8ea83, PR #27144)
TITLE: [Bugfix] fixes the decoding metadata of dense mla's fp8 kvcache. (#27144)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.mla.flashmla_build
FILES: cmake/external_projects/flashmla.cmake (+2/-1); vllm/attention/ops/flashmla.py (+6/-0); vllm/v1/attention/backends/mla/flashmla.py (+2/-0)
LABELS: ready, ci/build, v1
BODY: Require the flashmla patch https://github.com/vllm-project/FlashMLA/pull/7 to be landed first.

### L3-344a0017c0  (L3, 2025-10-21, sha 344a0017c06e, PR #26440)
TITLE: [Performance] Dual stream execution of "shared_experts" and "selected_experts" inside FusedMoE (#26440)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+5/-0); vllm/model_executor/layers/fused_moe/layer.py (+89/-15); vllm/model_executor/layers/fused_moe/shared_fused_moe.py (+16/-1); vllm/model_executor/models/deepseek_v2.py (+12/-6)
LABELS: documentation, ready, deepseek
DEEP_STUDY: deep-study performance PR ()
BODY: This PR executes the shared_experts part of the FusedMoE on a separate GPU stream, so that the execution is parallelized with the "selected_experts" part. This is possible since the outputs of both are independent and are later combined. Thanks @wenscarl for pointing this out. ⏎  ⏎ For DeepSeekR1 FP8 with Flashinfer latency kernels (trtllm-gen) on 8xB200s batch size 32, the TPOT improves from 23.35ms to 22.09ms (with latest FlashInfer codebase), so a …[truncated]

### L3-ecc3c0940a  (L3, 2025-10-21, sha ecc3c0940a09, PR #27213)
TITLE: Add @pavanimajety to .github/codeowners for Flashinfer, ModelOpt related code (#27213)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .github/CODEOWNERS (+5/-4)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ I have signed up for Flashinfer/ModelOpt related work. Thank you for the opportunity! 🙇🏽  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-f6027b2855  (L3, 2025-10-22, sha f6027b28553d, PR #26982)
TITLE: [1/N][Platform] Cleanup useless function (#26982)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: tests/models/quantization/test_fp8.py (+5/-2); tests/quantization/test_compressed_tensors.py (+0/-4); vllm/platforms/cuda.py (+1/-44); vllm/platforms/interface.py (+14/-27); vllm/platforms/rocm.py (+1/-7); vllm/platforms/tpu.py (+0/-6); vllm/platforms/xpu.py (+0/-16)
LABELS: rocm, tpu, ready
BODY: ## Purpose ⏎  ⏎ Platform interface becomes more complex. There are some issues we found: some interfaces are out of date, some are meaningless. I plan to submit some PRs to make it more clean. ⏎  ⏎ This is the first PR to remove the useless `is_kv_cache_dtype_supported` interface. This PR also moved some useless import lib to `TYPE_CHECKING` as well. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-1a0f4defb7  (L3, 2025-10-22, sha 1a0f4defb76c, PR #27282)
TITLE: [Log] Add Warning for `LLM(data_parallel_size=k)` single-process DP Usage (#27282)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/entrypoints/llm.py (+10/-0)
LABELS: frontend, ready
BODY: ## Purpose ⏎  ⏎ Currently, offline DP is not well supported, we may meet hang issue as https://github.com/vllm-project/vllm/issues/27269 describes. ⏎  ⏎ This PR adds a warning for users who use the param in this way. ⏎  ⏎ Followup: we may still need to fix this issue throughly and make the usage of `LLM(dp)` later. ⏎  ⏎ Appendix: the minimal code that works for offline dp ⏎  ⏎ ```bash ⏎ import os ⏎ from time import sleep ⏎ from multiprocessing import Process ⏎  ⏎ from vllm im …[truncated]

### L3-b4fda58a2d  (L3, 2025-10-22, sha b4fda58a2d0e, PR #27354)
TITLE: [MLA] Bump FlashMLA (#27354)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_build
FILES: cmake/external_projects/flashmla.cmake (+1/-1)
LABELS: ready, ci/build
ISSUES: #27043 [Bug]: FlashMLA: invalid configuration argument
BODY: ## Purpose ⏎ @starwang1024 implemented a fix for `flash_mla_combine_kernel` in https://github.com/vllm-project/FlashMLA/pull/10. This PR bumps the FlashMLA version to grab that change ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/27043 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-084a9dae80  (L3, 2025-10-22, sha 084a9dae801c, PR #27344)
TITLE: [Bugfix] Disable FlexAttention direct block mask building for encoder-only models (#27344)
SOURCES: path_core
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+4/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎ - The FP32 Mteb tests are failing after pytorch2.9 update, because `_build_block_mask_direct` return incorrect block mask shape. (https://buildkite.com/vllm/ci/builds/35829/steps/canvas?sid=019a0a13-ef28-4260-87f8-b6f4d685791a) ⏎ ``` ⏎ (EngineCore_DP0 pid=41725)   Developer debug context: raised exception ValueError([ConstantVariable(str: "block_mask was created for block_mask.shape=(1, 1, 7, 16) but got q_len=7 and kv_len=7. As the block  …[truncated]

### L3-5beacce2ea  (L3, 2025-10-22, sha 5beacce2eae2, PR #27128)
TITLE: [BugFix] bugfix for Flash Attention MLA with full cuda graph IMA following pr-25490 (#27128)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashattn
FILES: vllm/v1/attention/backends/mla/flashattn_mla.py (+20/-13)
LABELS: ready, v1
BODY: Bugfix for Flash Attention MLA with full cuda graph IMA following pr-25490 ⏎  ⏎ Run into illegal memory access error when testing some prompts with prefix caching enabled on Flash Attention MLA backend ⏎  ⏎ Log below is generated with CUDA_LAUNCH_BLOCKING=1 which indicating it's flash attn mla. ⏎  ⏎ ``` ⏎ INFO:/scripts/vllm_scripts/utils.py:CUDA error (../../.deps/vllm-flash-attn-src/hopper/flash_fwd_combine_launch_template.h:60): an illegal memory access was  …[truncated]

### L3-6644796bf4  (L3, 2025-10-22, sha 6644796bf445, PR #26060)
TITLE: [V1][spec decode] return logprobs for spec decoding (#26060)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/sample/test_logprobs.py (+94/-0); tests/v1/sample/test_rejection_sampler.py (+85/-79); vllm/v1/engine/logprobs.py (+1/-1); vllm/v1/outputs.py (+24/-9); vllm/v1/sample/rejection_sampler.py (+121/-39); vllm/v1/sample/sampler.py (+20/-11); vllm/v1/spec_decode/metadata.py (+8/-0); vllm/v1/worker/gpu_model_runner.py (+40/-48)
LABELS: speculative-decoding, ready, v1
BODY: # Purpose ⏎ Add support for returning logprobs for v1 spec decoding. ⏎  ⏎ # Test Plan ⏎  ⏎ ## Automated Tests ⏎ Added automated testing for logprobs which compares spec decode LLM per-token output logprobs with those of a reference model. The comparison is done for: `raw_logits`, `raw_logprobs`, `processed_logits`, and `processed_logprobs`. ⏎ ``` ⏎ (py312conda) bash-5.1$ pytest tests/v1/sample/test_logprobs.py -k test_spec_decode_logprobs ⏎ ======================= …[truncated]

### L3-50b788a17a  (L3, 2025-10-23, sha 50b788a17a8a, PR #27388)
TITLE: [CI/Build] Fix AMD CI: test_cpu_gpu.py (#27388)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_offload/test_cpu_gpu.py (+13/-4)
LABELS: rocm, ready, ci/build, v1, ci-failure
BODY: ## Purpose ⏎ `test_cpu_gpu.py` is added in https://github.com/vllm-project/vllm/pull/21448, while it add different attention backends to test, some backends might not be compatible with platforms - like FlashInferBackend is not supported in ROCM at the current moment. ⏎  ⏎ This PR refactors the test to conditional import backends, like what we did in [test_attention_backends.py](https://github.com/vllm-project/vllm/blob/7e0941055fdf89bae93045683dd80542 …[truncated]

### L3-dbfbf9f324  (L3, 2025-10-23, sha dbfbf9f32445, PR #27368)
TITLE: [Attention] Fix FlashMLA metadata builder arguments for q_len > 1 (#27368)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_v1_adapter
FILES: vllm/v1/attention/backends/mla/flashmla.py (+5/-1)
LABELS: bug, ready, v1, deepseek
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Purpose ⏎ As of #26541, FlashMLA now supports `q_len > 1` in the decode pipeline. The `get_mla_metadata` call was not updated, however, leading to poor performance (and potentially, crashes) in these cases. This PR is a simple bug fix achieving a substantial speedup, especially at small batch sizes. ⏎  ⏎ Note: uses the benchmarks in #26835 (not yet merged) ⏎  ⏎ cc @LucasWilkinson  ⏎  ⏎ ## Test Plan ⏎ `python benchmarks/attention_benchmarks/benchmark.py --conf …[truncated]

### L3-ca76486a16  (L3, 2025-10-23, sha ca76486a16fb, PR #27374)
TITLE: [Chore] Separate out `vllm.utils.platform_utils.py` (#27374)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+2/-1); tests/kernels/core/test_uva.py (+1/-1); tests/v1/logits_processors/test_correctness.py (+1/-1); tests/v1/sample/test_sampler.py (+1/-1); tests/v1/worker/test_gpu_input_batch.py (+1/-1); vllm/device_allocator/cumem.py (+1/-1); vllm/lora/lora_weights.py (+1/-1); vllm/lora/models.py (+1/-1); vllm/model_executor/model_loader/utils.py (+1/-1); vllm/model_executor/models/qwen2_5_vl.py (+1/-1); (+11 more)
LABELS: tpu, speculative-decoding, ready, v1, qwen
BODY: ## Purpose ⏎ Part of #26900  ⏎  ⏎  - `vllm.utils.cuda_is_initialized ⇒ vllm.utils.hardware_utils.cuda_is_initialized` ⏎  - `vllm.utils.xpu_is_initialized ⇒ vllm.utils.hardware_utils.xpu_is_initialized` ⏎  - `vllm.utils.cuda_get_device_properties ⇒ vllm.utils.hardware_utils.cuda_get_device_properties` ⏎  - `vllm.utils.is_pin_memory_available ⇒ vllm.utils.hardware_utils.is_pin_memory_available` ⏎  - `vllm.utils.is_uva_available ⇒ vllm.utils.hardware_utils.is_uva …[truncated]

### L3-570c3e1cd4  (L3, 2025-10-23, sha 570c3e1cd4b3, PR #27124)
TITLE: [Bugfix] Honor --mm_encoder_attn_backend when used (#27124)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+5/-1); vllm/model_executor/models/dots_ocr.py (+1/-0); vllm/model_executor/models/ernie45_vl.py (+1/-0); vllm/model_executor/models/glm4_1v.py (+1/-0); vllm/model_executor/models/qwen2_vl.py (+1/-0); vllm/model_executor/models/siglip2navit.py (+1/-0)
LABELS: rocm, ready, ci/build, ci-failure, qwen
BODY: Summary: ⏎ In https://github.com/vllm-project/vllm/pull/26104, some changes were made in layer.py that resulted in always trying to switch to FA backend for ViT, even when  `VLLM_ATTENTION_BACKEND` is set. ⏎  ⏎ This broke Meta's internal AMD pipelines as it is not desired nor expected behavior. With this change, the models that were changed in the offending PR can explicitly opt-in to this behavior. ⏎  ⏎ Differential Revision: D84946967

### L3-8dbe0c527f  (L3, 2025-10-23, sha 8dbe0c527fa7, PR #27423)
TITLE: [Misc] Add TPU usage report when using tpu_inference. (#27423)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/usage/usage_lib.py (+30/-10)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Report usage for users who are using tpu_inference, aka vllm-tpu 2.0 ⏎  ⏎ The associated PR in the tpu_inference repo is: https://github.com/vllm-project/tpu-inference/pull/925 ⏎  ⏎ ## Test Plan ⏎  ⏎ Tested manually. ⏎  ⏎ ## Test Result ⏎  ⏎ Create the usage_stats file: $HOME/.config/vllm/usage_stats.json and start vllm server, a report such as this was generated: ⏎  ⏎ ``` ⏎ {"uuid": "dd2ebebc-fa7c-4b5c-88f9-c6136bad1a2e", "provider": "GCP", "num_cpu": 180, "c …[truncated]

### L3-284cc92275  (L3, 2025-10-24, sha 284cc922757b, PR #26016)
TITLE: [MISC] `cudagraph_capture_sizes`  related improvements (#26016)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.flashattn
FILES: vllm/v1/attention/backends/flash_attn.py (+1/-1); vllm/v1/attention/backends/flashinfer.py (+1/-1); vllm/v1/attention/backends/mla/flashattn_mla.py (+1/-1); tests/compile/test_config.py (+73/-0); vllm/config/compilation.py (+40/-38); vllm/config/scheduler.py (+0/-15); vllm/config/vllm.py (+107/-26); vllm/engine/arg_utils.py (+62/-5); vllm/model_executor/layers/quantization/mxfp4.py (+1/-1); vllm/model_executor/models/config.py (+11/-13); (+4 more)
LABELS: speculative-decoding, ready, v1
ISSUES: #20283 [RFC][UX][torch.compile][CUDAGraph]: Overhaul `CompilationConfig` and improve CLI `-O<n>`
BODY: ## Purpose ⏎  ⏎ Part of the CompilationConfig improvements ([#20283](https://github.com/vllm-project/vllm/issues/20283)), asked by @ProExpertProg and @WoosukKwon: ⏎  ⏎ - rename max_capture_size -> max_cudagraph_capture_size: used to specify a single size, fills in the rest ⏎ - cudagraph_capture_sizes -> stays the same, used only to specify a full list ⏎ - SchedulerConfig.cuda_graph_sizes: -> remove ⏎ - Also stop reversing the sizes, always store them in ascend …[truncated]

### L3-0f67d4d962  (L3, 2025-10-24, sha 0f67d4d96287, PR #26397)
TITLE: [Attention] Add MLA prefill backend: trtllm_ragged_attention_deepseek (#26397)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.common_v1
FILES: vllm/envs.py (+6/-0); vllm/v1/attention/backends/mla/common.py (+107/-2)
LABELS: ready, v1, deepseek
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Add MLA prefill backend: trtllm_ragged_attention_deepseek ⏎ * controlled by `VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1` ⏎  ⏎ * [x] Need to fix accuracy issue ⏎   * 0.54 -> 0.59 fixed by zeroing output tensor for context chunk prefill ⏎   * 0.59 -> 0.61 (0.22 -> 0.64 w/ prefix caching) fixed by transposing lse ⏎   * 0.61 -> 0.65 fixed by zeroing workspace ⏎ * [x] performance benchmark ⏎  ⏎ ## Test Plan ⏎  ⏎ * [ ] microbenchmark ⏎   * in follow-up work after r …[truncated]

### L3-d95d0f4b98  (L3, 2025-10-24, sha d95d0f4b985f, PR #27328)
TITLE: [Distributed] Basic set of configuration for large EP deployment on GB200 (#27328)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/distributed/device_communicators/all2all.py (+3/-1); vllm/envs.py (+22/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ This PR adds basic configuration options for running large Expert Parallelism (EP) DeepSeek deployments on GB200 NVL72 systems. ⏎  ⏎ GB200 NVL72 architecture differs significantly from traditional large EP deployments. To maximize performance, we need to leverage MNNVL (Multi-Node NVLink) for both low latency and high throughput kernels, even in inter-node scenarios. This PR introduces three new environment variables to enable these optimi …[truncated]

### L3-a99564ac5b  (L3, 2025-10-25, sha a99564ac5b2b, PR #27490)
TITLE: [Attention] Add missing kv cache scale setup (#27490)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+72/-59)
LABELS: bug, ready, deepseek
BODY: ## Purpose ⏎ #25103 missed the kv cache scale setup when breaking out `MLAAttention`. This PR adds this to `MLAAttention.__init__` ⏎  ⏎ cc @pavanimajety  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-71b1c8b667  (L3, 2025-10-26, sha 71b1c8b66758, PR #27188)
TITLE: [Chore]:Extract math and argparse utilities to separate modules (#27188)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.mla.flashmla_sparse, L3.mla.rocm_aiter, L3.dispatch.abstract_interface, L3.flex_attention
FILES: benchmarks/benchmark_block_pool.py (+1/-1); benchmarks/benchmark_long_document_qa_throughput.py (+1/-1); benchmarks/benchmark_ngram_proposer.py (+1/-1); benchmarks/benchmark_prefix_caching.py (+1/-1); benchmarks/benchmark_prioritization.py (+1/-1); benchmarks/benchmark_serving_structured_output.py (+1/-1); benchmarks/cutlass_benchmarks/sparse_benchmarks.py (+1/-1); benchmarks/cutlass_benchmarks/w8a8_benchmarks.py (+2/-1); benchmarks/kernels/bench_per_token_quant_fp8.py (+1/-1); benchmarks/kernels/benchmark_activation.py (+1/-1); (+115 more)
LABELS: documentation, performance, rocm, structured-output, frontend, tpu, speculative-decoding, ready, v1, qwen
BODY: ## Purpose ⏎   - Extract math utilities (cdiv, round_up, round_down, etc.) to math_utils.py ⏎   - Extract argument parsing classes (FlexibleArgumentParser, etc.) to argparse_utils.py ⏎   - Reduce __init__.py from 1,293 to 819 lines (37% reduction) ⏎   - Maintain full backward compatibility via re-exports ⏎  ⏎   Contributes to #26900 ⏎  ⏎ ## Test Plan ⏎  Verify that the refactoring of math and argparse utilities into separate modules (vllm/utils/math_utils.py and v …[truncated]

### L3-65d2cf9511  (L3, 2025-10-26, sha 65d2cf95114a, PR #27190)
TITLE: [BUGFIX][ROCM] ViT FlashAttention on ROCm (no GFX9) and contiguous on qwen3vl ROCm TORCH_SDPA (#27190)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/attention/layer.py (+29/-11); vllm/platforms/rocm.py (+5/-1); vllm/model_executor/models/qwen2_5_vl.py (+6/-0); vllm/model_executor/models/qwen2_vl.py (+6/-0)
LABELS: rocm, ready, qwen
BODY: The refactor introduced in the following PR: ⏎ https://github.com/vllm-project/vllm/pull/26104 ⏎ improved the flash-attn selection, but broke the loading of models like Qwen/Qwen3-VL-30B-A3B-Instruct in RocM RDNA3, as it doesn't support flash-attn for VL. In the PR, I use the same backend selection for ROCM, which is used in: https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py ⏎  ⏎ in the method: ⏎  ⏎ get_vit_attn_backend ⏎  ⏎ I'm quoting @tjta …[truncated]

### L3-141e6a0505  (L3, 2025-10-28, sha 141e6a050596, PR #27367)
TITLE: [Misc] Make reorder batch also separate extends (#27367)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+53/-45); tests/v1/attention/test_batch_reordering.py (+111/-0)
LABELS: ready, v1
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:b5d70751d8 (fix PR 27739)
BODY: Pre-requisite for optimizations that want to seperate a batch into `[decode, extend, prefill]`, e.g. https://github.com/vllm-project/vllm/pull/25763 (future optimizations that may use this include skipping 0-seqlen in MLA when batch contains both extends and prefills; DeepSeek-V3.2-Exp skip up-converting fp8-kv-cache for pure prefills)

### L3-a8c02fb5bf  (L3, 2025-10-28, sha a8c02fb5bf2e, PR #26597)
TITLE: [Bugfix][CI] Fix v1 attention backend tests and add CI coverage (#26597)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+34/-11); .buildkite/test-pipeline.yaml (+9/-0); tests/v1/attention/test_attention_backends.py (+22/-14); tests/v1/attention/test_mla_backends.py (+2/-1)
LABELS: ready, ci/build, v1
ISSUES: #26537 [Bug]: V1 attention tests are broken
BODY: ## Purpose ⏎  ⏎ Fixes #26537 by correcting KV-cache layout handling and MLA config setup in the harness, disabling gradients on mock projection weights to avoid autograd asserts, and ensuring these attention suites now run in CI by adding them to the Buildkite “V1 Test others” step. ⏎  ⏎ ## Test Plan ⏎  ⏎ `pytest tests/v1/attention` ⏎  ⏎ ## Test Result ⏎  ⏎ `78 passed, 2 warnings in 79.24s` ⏎  ⏎ --- ⏎ [details omitted]

### L3-f257544709  (L3, 2025-10-28, sha f257544709a8, PR #27598)
TITLE: Install pre-built xformers-0.0.32.post2 built with pt-2.9.0 (#27598)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+0/-7); requirements/cuda.txt (+1/-1)
LABELS: ready, ci/build
DEEP_STUDY: deep-study: this PR was reverted by PR 27714 (confirmed_revert, reason=build_or_dependency) || deep-study: this PR was reverted by PR 27768 (reland, reason=build_or_dependency)
BODY: ## Purpose ⏎  ⏎ Instead of waiting for xformers to release a new version for PyTorch 2.9.0, I have built `0.0.32.post2` locally and made the wheel available. ⏎  ⏎ For more context, we don’t want to wait for `xformers` package for 2.9 to become available.  So, I opt to build it from source.  This works for CI, but has several issues like (1) increasing build time and (2) not listed as a dependency in `cuda.txt`.  So, installing a pre-built wheel would hel …[truncated]

### L3-9007bf57e6  (L3, 2025-10-28, sha 9007bf57e6a2, PR #27714)
TITLE: Revert "Install pre-built xformers-0.0.32.post2 built with pt-2.9.0" (#27714)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+7/-0); requirements/cuda.txt (+1/-1)
LABELS: ci/build
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 27598 reason=build_or_dependency
BODY: Reverts vllm-project/vllm#27598 ⏎  ⏎ Broke CUDA 13 build. https://buildkite.com/vllm/release/builds/9637/steps/canvas?sid=019a2dfc-911c-4783-b421-9d3acc153e1b

### L3-94666612a9  (L3, 2025-10-28, sha 94666612a938, PR #23207)
TITLE: [Misc][qwen2_5_vl][torch.compile] Enable `supports_torch_compile` on generic nn.Module and demonstrate speedup on Qwen Vision model (#23207)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/vit_attn_wrappers.py (+125/-0); tests/compile/test_multimodal_compile.py (+36/-0); vllm/compilation/decorators.py (+56/-4); vllm/config/compilation.py (+2/-0); vllm/model_executor/models/qwen2_5_omni_thinker.py (+6/-2); vllm/model_executor/models/qwen2_5_vl.py (+110/-92)
LABELS: documentation, ready, llama, qwen
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:b13a447546 (fix PR 27748)
BODY: We enable `@supports_torch_compile` on generic nn.Modules (as opposed to only top level architecture) and demonstrate the application in qwen_2_5_vl ⏎  ⏎ ## Purpose ⏎ This PR is a first step towards supporting torch compile for multimodal encoders (such as vision or audio).  Since these modality specific components do not have vLLM config, we begin by taking a step to allow these modules to be compiled ⏎  ⏎ ## Test Plan ⏎ ### Unit Test ⏎ ``` ⏎ with-proxy pytest  …[truncated]

### L3-5b0448104f  (L3, 2025-10-29, sha 5b0448104fa1, PR #27424)
TITLE: [Bug] Raise error explicitly if using incompatible backend (#27424)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+15/-0)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ If the users choose the wrong backend, will meet unexpected error, eg: ⏎  ⏎ ``` ⏎ (EngineCore_DP7 pid=4191233)   File "/home/wentao/vllm-source/vllm/model_executor/models/utils.py", line 642, in make_layers ⏎ (EngineCore_DP7 pid=4191233)     maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}")) ⏎ (EngineCore_DP7 pid=4191233)                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ (EngineCore_DP7 pid=4191233)   File "/home/wentao/vllm-sou …[truncated]

### L3-b5d70751d8  (L3, 2025-10-29, sha b5d70751d82c, PR #27739)
TITLE: [BugFix] Reordering extend logic fix (#27739)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+5/-5); tests/v1/attention/test_batch_reordering.py (+18/-3)
LABELS: ready, v1
DEEP_STUDY: deep-study correctness case vllm:b5d70751d8: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=#27367
BODY: Fix incorrect definition of extends in https://github.com/vllm-project/vllm/pull/27367, more extensive unit tests, fix reordering bug when multiple swaps are involved + light refactor. ⏎  ⏎ Credit to @ganyi1996ppo for finding the issue

### L3-e806178d2a  (L3, 2025-10-30, sha e806178d2a9b, PR #27790)
TITLE: [BugFix][VL] Fix FA selection on Qwen2.5-VL (#27790)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+1/-1); vllm/model_executor/models/qwen2_5_vl.py (+19/-11)
LABELS: rocm, ready, ci/build, qwen
BODY: ## Purpose ⏎ https://github.com/vllm-project/vllm/pull/27190 breaks AMD CI (and also qwen2.5 vl): [tests/v1/entrypoints/openai/responses/test_image.py ](https://github.com/vllm-project/vllm/blob/main/tests/v1/entrypoints/openai/responses/test_image.py): with _Backend.FLASH_ATTN it did NOT set  use_upstream_fa = True([code](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_5_vl.py#L702-L703)), so we got ImportError: can …[truncated]

### L3-e5e076cad7  (L3, 2025-10-30, sha e5e076cad7c1, PR #27762)
TITLE: [BugFix] Stopgap - Flashinfer Autotuner + GPT-OSS + DP/TP (#27762)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/warmup/kernel_warmup.py (+13/-7)
LABELS: bug, ready, gpt-oss
BODY: ## Purpose ⏎ Running the Flashinfer autotuner when, ⏎  - using data-parallel or tensor-parallel, and ⏎  - using a flashinfer mxfp4 backend, and ⏎  - eager-mode ⏎  ⏎ Causes the engine startup to fail with,  ⏎ ``` ⏎ (EngineCore_DP1 pid=3074289)   File "/home/varun-sundar-rabindranath/code/vllm/vllm-test/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl ⏎ (EngineCore_DP1 pid=3074289)     return forward_call(*args, **kwargs) ⏎ (EngineCor …[truncated]

### L3-af826e0820  (L3, 2025-10-30, sha af826e082045, PR #27784)
TITLE: [V0 deprecation] Remove VLLM_USE_V1 usage in config module (#27784)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/config/lora.py (+0/-5); vllm/config/model.py (+2/-23); vllm/config/speculative.py (+0/-7); vllm/config/vllm.py (+7/-27)
LABELS: ready
BODY: ## Purpose ⏎ V0 code is removed already. `VLLM_USE_V1` env is useless now. Let's remove `VLLM_USE_V1` related code. This PR just cleanup it from config module to make the change small for easier review.  More PR is on the way. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-ba33e8830d  (L3, 2025-10-30, sha ba33e8830dce, PR #27768)
TITLE: Reapply "Install pre-built xformers-0.0.32.post2 built with pt-2.9.0" (#27768)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+0/-7); requirements/cuda.txt (+2/-2)
LABELS: ready, ci/build
DEEP_STUDY: deep-study revert record: reland of PR(s) 27598 reason=build_or_dependency
BODY: ## Purpose ⏎  ⏎ This relands https://github.com/vllm-project/vllm/pull/27598 exactly as it is.  For the CUDA 13.0 build failure in https://buildkite.com/vllm/release/builds/9637#019a3162-f87b-4fd4-820a-5612913b590e, I have added the missing xformers wheel for cu130 at https://download.pytorch.org/whl/cu130/xformers-0.0.33%2B5d4b92a5.d20251029-cp39-abi3-linux_x86_64.whl ⏎  ⏎ ## Test Plan ⏎  ⏎ https://buildkite.com/vllm/release/builds/9672 should all be green ⏎  …[truncated]

### L3-e7acb20076  (L3, 2025-10-30, sha e7acb200766a, PR #27660)
TITLE: [Feature] Batch invariant torch.compile (#27660)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/config/model.py (+0/-7); vllm/envs.py (+7/-1); vllm/model_executor/layers/batch_invariant.py (+71/-0); vllm/model_executor/layers/quantization/fp8.py (+4/-1)
LABELS: ready
BODY: ## Purpose ⏎ Building off of https://github.com/vllm-project/vllm/pull/27660 to enable batch invariant numerics in vllm, this PR allows batch invariant numerics in conjunction with `torch.compile`. Namely, we just disable cublas level optimizations that cause different numerics based on batch size, for example split-k. This PR is mainly just using work that @ngimel has done in PyTorch for manipulating cuBLAS to be batch invariant. ⏎  ⏎ This change allo …[truncated]

### L3-4e68cc9b6a  (L3, 2025-10-30, sha 4e68cc9b6aa2, PR #27809)
TITLE: [Model] Introduce Kimi Linear to vLLM (#27809)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mla.py (+4/-3); docs/models/supported_models.md (+1/-0); tests/models/registry.py (+3/-0); vllm/config/compilation.py (+1/-0); vllm/config/model.py (+1/-0); vllm/model_executor/layers/fla/ops/kda.py (+1/-1); vllm/model_executor/layers/kda.py (+426/-0); vllm/model_executor/layers/mamba/mamba_utils.py (+41/-0); vllm/model_executor/models/config.py (+25/-26); vllm/model_executor/models/kimi_linear.py (+663/-0); (+5 more)
LABELS: documentation, new-model, v1
BODY: ## Purpose ⏎ Introducing Kimi Linear, an advanced hybrid attention model that combines the efficiency of [Kimi Delta Attention (KDA)](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda), a refined version of Gated DeltaNet, with reduced memory requirements and superior performance across short, long, and reinforcement learning contexts. By introducing an optimized gating mechanism, Kimi Linear significantly cuts down the need f …[truncated]

### L3-f29aeb5a25  (L3, 2025-10-31, sha f29aeb5a25da, PR #27663)
TITLE: Add FLASHINFER_MLA to test_mla_backends and add B200 CI run (#27663)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer
FILES: vllm/v1/attention/backends/mla/flashinfer_mla.py (+5/-1); .buildkite/test-pipeline.yaml (+10/-0); tests/v1/attention/test_mla_backends.py (+182/-62); tests/v1/attention/utils.py (+11/-1)
LABELS: ready, ci/build, v1
BODY: ## Purpose ⏎ Adds `FLASHINFER_MLA` to `test_mla_backends.py`. While #26597 added this to CI, it was only run on H100. This PR adds a B200 runner so that this and other SM100 backends get tested as well. ⏎ ## Test Plan ⏎ `pytest tests/v1/attention/test_mla_backends.py` ⏎  ⏎ ## Test Result ⏎ Passes (as long as FlashInfer prefill is disabled) ⏎  ⏎ --- ⏎ [details omitted]

### L3-933cdea440  (L3, 2025-10-31, sha 933cdea44061, PR #27861)
TITLE: [BugFix] Don’t compute reorder threshold when there are no attention groups (#27861)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_model_runner.py (+5/-0)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ This PR fixes a startup crash in the v1 runtime for attention‑free models (e.g., Terratorch) introduced after #27809. The engine unconditionally computed the batch reorder threshold even when no attention backends were created, leading to: ⏎ ``` ⏎ TypeError: reduce() of empty iterable with no initial value ⏎ ``` ⏎ from the nightly run (https://buildkite.com/vllm/ci/builds/37041/steps/canvas?sid=019a386d-1b25-4c07-9a9b-085c1e07ea05, https://bu …[truncated]

### L3-3933f18a5e  (L3, 2025-10-31, sha 3933f18a5e7b, PR #27853)
TITLE: [Bugfix] Avoid too small block m/n for FlexAttention kernel option (#27853)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+5/-0)
LABELS: ready, v1
ISSUES: #27724 [CI Failure]: torch._inductor.exc.InductorError in Nightly build to run all tests
BODY: ## Purpose ⏎ - Fix #27724 ⏎ - Current `get_kernel_options` can cause too small BLOCK_M/BLOCK_N (BLOCK_M=8), which broke some encoder-only models.  ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest -s -v tests/models/language/pooling_mteb_test/test_jina.py -k test_embed_models_mteb[model_info0] ⏎ ``` ⏎  ⏎ ## Test Result ⏎ Failed test should pass now. ⏎  ⏎ --- ⏎ [details omitted]

### L3-7e2729b57e  (L3, 2025-11-01, sha 7e2729b57e5c, PR #27525)
TITLE: [Multimodal][XPU]Enable vision attn backend for xpu platform (#27525)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+18/-17); vllm/attention/ops/vit_attn_wrappers.py (+1/-1); vllm/platforms/xpu.py (+6/-0); vllm/_ipex_ops.py (+59/-25); vllm/model_executor/models/qwen2_5_vl.py (+3/-4); vllm/model_executor/models/qwen2_vl.py (+1/-4)
LABELS: ready, qwen
BODY: ## Purpose ⏎  ⏎ This PR uses `FLASH_ATTN` as vision attention backend for xpu platform and actually calls `varlen_attention` kernel in IPEX by dispatching in `flash_attn_varlen_func`. ⏎  ⏎ ## Test Plan ⏎  ⏎ `python examples/offline_inference/vision_language.py -m glm-4v` and `python examples/offline_inference/vision_language.py -m qwen2_5_vl` ⏎  ⏎ ## Test Result ⏎  ⏎ `python examples/offline_inference/vision_language.py -m qwen2_5_vl`: ⏎ ``` ⏎ Processed prompts: 100%|██ …[truncated]

### L3-30a14b034f  (L3, 2025-11-01, sha 30a14b034fa3, PR #27798)
TITLE: [V0 deprecation] Remove VLLM_USE_V1 usage in platform and v1 module (#27798)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/platforms/cuda.py (+84/-100); vllm/platforms/interface.py (+1/-8); vllm/platforms/rocm.py (+32/-52); vllm/platforms/tpu.py (+0/-4); vllm/platforms/xpu.py (+3/-6); vllm/v1/engine/async_llm.py (+0/-16); vllm/v1/engine/llm_engine.py (+1/-10); vllm/v1/executor/uniproc_executor.py (+4/-5)
LABELS: rocm, tpu, ready, v1
BODY: ## Purpose ⏎ Following #27784, this PR remove `VLLM_USE_V1` usage in platform and v1 module ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-18961c5ea6  (L3, 2025-11-03, sha 18961c5ea629, PR #27753)
TITLE: [Hybrid] Pass kernel block size to builders (#27753)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+5/-1); vllm/v1/kv_cache_interface.py (+7/-1); vllm/v1/worker/gpu_model_runner.py (+25/-6); vllm/v1/worker/utils.py (+25/-19)
LABELS: ready, v1
ISSUES: #26936 [Bug]: Hybrid Attention models broken after switching to flashinfer 0.4 (tested on Granite 4.0 H, Qwen3-Next, Jamba-3B, Nemotron-H-8b) | #27264 [Bug]: Cache malformation in hybrid models with SSM cache dtype float32 and block allocation wrap around
BODY: ## Purpose ⏎  ⏎ Solves https://github.com/vllm-project/vllm/issues/27264 and https://github.com/vllm-project/vllm/issues/26936 ⏎  ⏎ This PR makes two changes: ⏎ - The GPU model runner will now pass the kernel block size to the metadata builders, fixing a pretty bad bug that exists on main. ⏎ - I also restrict the kernel block sizes for the FlashAttention backend based on the discussion in https://github.com/vllm-project/vllm/issues/27264. This is required si …[truncated]

### L3-55011aef24  (L3, 2025-11-03, sha 55011aef24c2, PR #27764)
TITLE: [Bugfix][Qwen][Multimodal] Move Qwen2_5_vl sdpa to custom op and reenable compile (#27764)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/vit_attn_wrappers.py (+53/-0); vllm/model_executor/models/qwen2_5_vl.py (+16/-28)
LABELS: ready, qwen
BODY: ## Purpose ⏎ See title - this PR fixes an error caused by https://github.com/vllm-project/vllm/pull/23207 where the SDPA backend could not be compiled. ⏎  ⏎ This was because a fix for tracing SliceVariables containing single element tensors is not yet landed (but should be in Pytorch2.10) ⏎  ⏎ Until then, we shall use a custom op to prevent graph break ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ python examples/offline_inference/vision_language.py -m qwen2_5_vl ⏎ ``` ⏎ With edit to the …[truncated]

### L3-145c00a4d3  (L3, 2025-11-03, sha 145c00a4d32b, PR #27777)
TITLE: [Bugfix] change FlashMLA reorder_batch_threshold (#27777)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_v1_adapter
FILES: vllm/v1/attention/backends/mla/flashmla.py (+1/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎ Reduce FlashMLA `reorder_batch_threshold` from 512 to 128. This fixes LM Eval Large Models when applied to 82af928 (the commit that merged #26541) but doesn't seem sufficient to fix it on TOT. There might be multiple commits involved in that failure ⏎  ⏎ ## Test Plan ⏎ `pytest -s -v test_lm_eval_correctness.py::test_lm_eval_correctness_param[config_filename4] --config-list-file=configs/models-large.txt --tp-size=4` ⏎  ⏎ ## Test Result ⏎ Passes (on …[truncated]

### L3-b13a447546  (L3, 2025-11-03, sha b13a44754674, PR #27748)
TITLE: [Bugfix][ROCm] Fix ViT rotary embeddings for torch.compile compatibility on ROCm (#27748)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/rotary_embedding/common.py (+7/-4); vllm/model_executor/models/glm4_1v.py (+1/-1)
LABELS: rocm, ready, qwen
DEEP_STUDY: deep-study correctness case vllm:b13a447546: class=hardware_compiler_specific; symptom=crash_or_exception; introducing=#23207
BODY: ## Purpose ⏎  ⏎ This pull request updates the selection of rotary embedding function previously selected for the ROCm platform from `flash_attn.ops.triton.rotary` because it is not compatible with `torch.compile`. Since [this PR](https://github.com/vllm-project/vllm/pull/23207) enabled `torch.compile` support for Qwen-VL models, it became necessary to address this incompatibility. After registering the rotary embedding operation using `direct_registe …[truncated]

### L3-40b69e33e7  (L3, 2025-11-03, sha 40b69e33e796, PR #27758)
TITLE: [Model] Add PaddleOCR-VL Model Support  (#27758)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/models/supported_models.md (+1/-0); examples/offline_inference/vision_language.py (+27/-0); examples/offline_inference/vision_language_multi_image.py (+22/-0); tests/models/registry.py (+4/-0); vllm/model_executor/models/ernie45.py (+10/-0); vllm/model_executor/models/paddleocr_vl.py (+1407/-0); vllm/model_executor/models/registry.py (+4/-0)
LABELS: documentation, new-model, ready
ISSUES: #27167 [New Model]: support paddleocr-vl-0.9B | #27804 [New Model]: Support for PaddleOCR-VL
BODY: # Purpose ⏎  ⏎ Support Baidu PaddleOCR-VL model for vllm ⏎  ⏎ FIX #27167 ⏎ FIX #27804 ⏎  ⏎ # Start Command (v0.10.2) ⏎  ⏎ ``` ⏎ vllm serve PaddlePaddle/PaddleOCR-VL \ ⏎     --trust-remote-code \ ⏎     --max-model-len 16384 \ ⏎     --max-num-batched-tokens 16384 ⏎ ``` ⏎  ⏎ # Note ⏎  ⏎ 1. Based on the releases/v0.10.2 branch ⏎ 2. We do not have an AMD GPU machine to verify the feasibility of `ROCM_AITER_FA`, and we are wondering whether it is required to add support for `ROCM_ALTER_FA`

### L3-5fd8f02ea9  (L3, 2025-11-04, sha 5fd8f02ea9f2, PR #27512)
TITLE: [PERF] Decouple projections from GDN custom op (#27512)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/compilation.py (+1/-1); vllm/model_executor/layers/layernorm.py (+102/-0); vllm/model_executor/models/qwen3_next.py (+101/-52)
LABELS: ready, qwen
DEEP_STUDY: deep-study: this PR was reverted by PR 28080 (confirmed_revert, reason=unstated) || deep-study performance PR ()
BODY: ## Purpose ⏎ This PR is refactoring of GDN.  ⏎  ⏎ The main goal is to allow wider using of `torch.compile`. ⏎  ⏎ 1. Separated forward pass of GDN attention into three distinct pieces: Input Projection, Core Attention, Output Projection. Before projections was in the GDN custom op and were not covered by `torch.compile`. ⏎ 2. Added `RMSNormGated` class that implements torch native gated rmsnorm and use it for GDN. `torch.compile` creates a good code for `RMSN …[truncated]

### L3-7e4be74104  (L3, 2025-11-04, sha 7e4be741044b, PR #27884)
TITLE: [Bug] Batch invariant: Fix flash attn MLA `RuntimeError: scheduler_metadata must have shape (metadata_size)` (#27884)
SOURCES: path_core, subject_keyword, release_notes, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L3.mla.flashattn
FILES: vllm/v1/attention/backends/mla/flashattn_mla.py (+3/-3); vllm/model_executor/layers/batch_invariant.py (+2/-0)
LABELS: ready, v1
DEEP_STUDY: deep-study correctness case vllm:7e4be74104: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎  ⏎ ```bash ⏎ export VLLM_BATCH_INVARIANT=1 ⏎ vllm serve deepseek-ai/DeepSeek-V3 -tp 8  --enable-expert-parallel --port 9256 ⏎ vllm bench serve --model deepseek-ai/DeepSeek-V3  --dataset-name random --host 127.0.0.1 --port 9256 --random-input-len 4 --random-output-len 64 --request-rate inf --num-prompts 256 ⏎ ``` ⏎  ⏎ will trigger error: ⏎  ⏎ ```bash ⏎ ^[[1;36m(Worker_TP4_EP4 pid=4005888)^[[0;0m ERROR 10-31 00:28:52 [multiproc_executor.py:703]     return s …[truncated]

### L3-4022a9d279  (L3, 2025-11-04, sha 4022a9d279d0, PR #27904)
TITLE: [BugFix][Performance] Restore flashinfer autotuning for all scenarios (#27904)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/quantization/test_blackwell_moe.py (+2/-14); vllm/model_executor/layers/fused_moe/trtllm_moe.py (+9/-2); vllm/model_executor/layers/quantization/mxfp4.py (+2/-2); vllm/model_executor/warmup/kernel_warmup.py (+1/-26)
LABELS: ready, nvidia
ISSUES: #27751 [Bug]: Issue with  Flashinfer Autotune + DP or TP + Eager-Mode
BODY: ## Purpose ⏎  ⏎ Bug:  ⏎ on `main` + B200 : `vllm serve openai/gpt-oss-20b --enforce-eager` fails. ⏎ on `main` + H100 : `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1 vllm serve openai/gpt-oss-20b --enforce-eager` fails. ⏎  ⏎ Both failures are asserts in the flashinfer code base,  ⏎ ``` ⏎ (EngineCore_DP0 pid=3490083)   File "/home/varun/code/vllm/vllm/model_executor/layers/quantization/mxfp4.py", line 1109, in apply ⏎ (EngineCore_DP0 pid=3490083)     _ = flashinfer_cutlass_ …[truncated]

### L3-2d977a7a9e  (L3, 2025-11-04, sha 2d977a7a9ead, PR #26969)
TITLE: [ROCm] gemm_a16w16 upstreaming (#26969)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/utils.py (+34/-8); vllm/model_executor/models/gpt_oss.py (+9/-1)
LABELS: rocm, frontend, ready, tool-calling, gpt-oss
BODY: GPT OSS, m and n to check: https://github.com/ROCm/vllm/commit/bcc4e6920bf0a51594216ba0d34d0dc3f360397c ⏎ ``` ⏎ HIP_VISIBLE_DEVICES=7 \ ⏎ HSA_NO_SCRATCH_RECLAIM=1 \ ⏎ NCCL_MIN_NCHANNELS=112 \ ⏎ USE_FASTSAFETENSOR=1 \ ⏎ SAFETENSORS_FAST_GPU=1 \ ⏎ VLLM_DISABLE_COMPILE_CACHE=1 \ ⏎ VLLM_ROCM_USE_AITER=1 \ ⏎ VLLM_USE_AITER_UNIFIED_ATTENTION=1 \ ⏎ VLLM_ROCM_USE_AITER_MHA=0 \ ⏎ vllm serve /data/models/openai/gpt-oss-120b \ ⏎     --host localhost \ ⏎     --port 30000 \ ⏎     --tens …[truncated]

### L3-dc937175d4  (L3, 2025-11-04, sha dc937175d496, PR #25763)
TITLE: [ROCm][Perf] New design on ROCm AITER MHA backend Implementation (#25763)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+526/-275); vllm/v1/attention/backends/utils.py (+67/-0)
LABELS: rocm, ready, ci/build, v1
BODY: ## Purpose ⏎ The current `AiterFlashAttentionImpl` fetches K/V every run, which creates unnecessary memory pressure and non-trivial latency—especially with long prompts. This PR: ⏎ - Removes redundant KV fetches from the attention backend ⏎ - Introduces a phase-aware execution (decode, pure prefill, chunk prefill) and reorders inputs to [decode:chunk_prefill:pure_prefill] for token-contiguous memory access. ⏎ - Rewrites the “fetch KV” Triton kernel for b …[truncated]

### L3-428bc7bf1c  (L3, 2025-11-04, sha 428bc7bf1c54, PR #27955)
TITLE: [V0 deprecation] Remove VLLM_USE_V1 usage in most modules (#27955)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/layers/chunked_local_attention.py (+6/-12); vllm/attention/layers/cross_attention.py (+4/-10); vllm/attention/layers/encoder_only_attention.py (+4/-11); vllm/attention/selector.py (+1/-7); docs/usage/v1_guide.md (+0/-2); tests/conftest.py (+0/-20); tests/v1/engine/test_async_llm.py (+2/-5); tests/v1/entrypoints/llm/test_struct_output_generate.py (+0/-3); tests/v1/sample/test_logprobs.py (+59/-62); vllm/distributed/kv_transfer/kv_connector/factory.py (+0/-7); (+9 more)
LABELS: documentation, structured-output, frontend, ready, v1, multi-modality, kv-connector
BODY: ## Purpose ⏎ This PR remove `VLLM_USE_V1` usage in most place. After this PR, there are only 3 work items left: ⏎  ⏎ 1. cleanup `_is_v1_supported_oracle` check. There is already a PR, I'll take it it if it's still WIP. https://github.com/vllm-project/vllm/pull/25673 ⏎ 2. Deprecate  `use_v1` parameter for `get_attn_backend_cls` platform interface ⏎ 3. remove `VLLM_USE_V1` env var totally. ⏎  ⏎ I'll do it  one by one in the next 3 PRs. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Res …[truncated]

### L3-0ff05e3770  (L3, 2025-11-04, sha 0ff05e3770ad, PR #28021)
TITLE: [Bugfix] Fix encoder-only model support for transformers backend (#28021)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/registry.py (+6/-6); tests/models/test_transformers.py (+1/-1); vllm/model_executor/models/transformers/base.py (+8/-2); vllm/model_executor/models/transformers/moe.py (+1/-1)
LABELS: ready
BODY: ## Purpose ⏎ - Fix broken encoder-only model support on transformers backend after #26587 ⏎ ``` ⏎ (EngineCore_DP0 pid=5260) INFO 11-04 03:34:52 [base.py:120] Using Transformers backend. ⏎ (EngineCore_DP0 pid=5260) INFO 11-04 03:34:52 [cuda.py:414] Using FlexAttention backend. ⏎ (EngineCore_DP0 pid=5260) [2025-11-04 03:34:54] INFO _client.py:1025: HTTP Request: GET https://huggingface.co/api/models/papluca/xlm-roberta-base-language-detection "HTTP/1.1 200 O …[truncated]

### L3-18b39828d9  (L3, 2025-11-05, sha 18b39828d904, PR #27786)
TITLE: [XPU] Add gpt-oss model support for Intel GPU (#27786)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fa_utils
FILES: vllm/attention/utils/fa_utils.py (+7/-0); vllm/v1/attention/backends/flash_attn.py (+2/-1); vllm/model_executor/layers/quantization/mxfp4.py (+92/-2); vllm/model_executor/models/gpt_oss.py (+0/-3)
LABELS: ready, v1, gpt-oss
BODY: ## Purpose ⏎ this PR introduce a new `IpexFp4MoeMethod` for xpu, which support Wfp4A16 moe_gemm, implemented in ipex library.  With that we can run openai/gpt-oss-20b, openai/gpt-oss-120b on Intel B60 GPU. ⏎  ⏎ gpt-oss-20b accuracy: ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     0|exact_match|?  |0.9333|±  |0.0069| ⏎ | …[truncated]

### L3-c765f0b443  (L3, 2025-11-05, sha c765f0b443c2, PR #27994)
TITLE: [FlashInfer] Avoid FlashInfer block_size 16 + head_size 256 on blackwell (#27994)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts(reverted), body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+9/-0); vllm/model_executor/models/config.py (+12/-0)
LABELS: ready, v1
DEEP_STUDY: deep-study: this PR was reverted by PR 36987 (confirmed_revert, reason=build_or_dependency)
BODY: ## Purpose ⏎ Avoid this combination as https://github.com/flashinfer-ai/flashinfer/issues/1993 reports this combination is not correct. ⏎  ⏎ For most models with head_size 256, users now need --block_size 32 / --block_size 64 ⏎  ⏎ For hybrid mamba like qwen3-next, the block_size can be resolved automatically  ⏎  ⏎ Thanks @vadiklyutiy for the exploration on this problem https://github.com/vllm-project/vllm/pull/27704 ⏎  ⏎ ## Test Plan ⏎  ⏎ Test non-hybrid case and hybr …[truncated]

### L3-faedbb4d4f  (L3, 2025-11-05, sha faedbb4d4fe4, PR #27856)
TITLE: [Feature] Extend batch invariant torch.compile to B200 (#27856)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+6/-0); tests/v1/generation/test_batch_invariance.py (+0/-2); vllm/model_executor/layers/batch_invariant.py (+24/-15)
LABELS: ready, v1
BODY: ## Purpose ⏎ This PR resolves issues with torch.compile + cudagraphs batch invariance issues on B200. Namely, `trtllm_attention` on B200 + cudagraphs causes issues. ⏎  ⏎ We extend all the unit tests as well to use torch.compile for evaluating batch invariance, and also disable GEMM custom operator overriding on PyTorch 2.10+, as it now contains the batch invariant cuda overrides, such as https://github.com/pytorch/pytorch/pull/166735. For PyTorch 2.9,  …[truncated]

### L3-6cae1e5332  (L3, 2025-11-05, sha 6cae1e53326a, PR #27224)
TITLE: [ROCm][MLA] Support block-size > 1 for AITER MLA backend  (#27224)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter, L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+3/-10); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+31/-7); tests/kernels/attention/test_attention_selector.py (+0/-7)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎ The `AITERMLABackend` now only support `block-size=1` scenario for inference. This constrain may lead to some serious host overhead when we are about to allocate or free cache blocks for long context requests cause there might exist large amount of blocks to operate. Thanks to the insights of @gyu-amd . ⏎  ⏎ In this PR, we remapping the `block_table` to 1 block size case every step in `AITERMLAMetadataBuilder` to alleviate the host overhea …[truncated]

### L3-d43ad5a757  (L3, 2025-11-05, sha d43ad5a75790, PR #28100)
TITLE: [BugFix] Fix DCP Assert (AssertionError: DCP not support reorder_batch_threshold > 1 now.) (#28100)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashattn, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/mla/common.py (+2/-1); vllm/v1/attention/backends/mla/flashattn_mla.py (+6/-1); vllm/v1/attention/backends/utils.py (+10/-1)
LABELS: v1
BODY: Fix `AssertionError: DCP not support reorder_batch_threshold > 1 now.` when running with DCP on hopper

### L3-b6a248bdd7  (L3, 2025-11-05, sha b6a248bdd7bf, PR #28083)
TITLE: [PERF] Decouple projections from GDN custom op. Attempt 2 (#28083)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/compilation.py (+1/-1); vllm/model_executor/layers/layernorm.py (+103/-0); vllm/model_executor/models/qwen3_next.py (+101/-52)
LABELS: ready, qwen
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ The second attempt for #27512 ⏎  ⏎ This PR is refactoring of GDN.  ⏎  ⏎ The main goal is to allow wider using of `torch.compile`. ⏎  ⏎ 1. Separated forward pass of GDN attention into three distinct pieces: Input Projection, Core Attention, Output Projection. Before projections was in the GDN custom op and were not covered by `torch.compile`. ⏎ 2. Added `RMSNormGated` class that implements torch native gated rmsnorm and use it for GDN. `torch.compile …[truncated]

### L3-16b37f3119  (L3, 2025-11-05, sha 16b37f311991, PR #27518)
TITLE: [bugfix] fix wrong `dcp_local_seq_lens` calc (#27518)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+1/-1)
LABELS: v1
BODY: When the `seq_lens` is exactly divisible by the `dcp_world_size`, it causes the `dcp_local_seq_lens` on all ranks to be incremented by one. Fix this calculation logic. ⏎  ⏎ CC @youzhedian @minosfuture @youkaichao

### L3-e31946f86e  (L3, 2025-11-06, sha e31946f86eb7, PR #28166)
TITLE: [flashinfer] fix FI all2all with FI cutlass moe (#28166)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py (+3/-1)
LABELS: ready
BODY: Summary: ⏎ Running FI Cutlass moe with FI a2av backend runs into error: ⏎  ⏎ ``` ⏎ ^[[1;36m(EngineCore_DP7 pid=104761)^[[0;0m ERROR 11-05 14:09:51 [core.py:843]     ) = self.prepare_finalize.prepare( ⏎ ^[[1;36m(EngineCore_DP7 pid=104761)^[[0;0m ERROR 11-05 14:09:51 [core.py:843]   File "/data/users/mxz/fbsource/buck-out/v2/gen/fbcode/c9838acc51201940/smart/inference_platform_sp/llm_predictor_gpu/__service__/service#link-tree/vllm/model_executor/layers/fuse …[truncated]

### L3-3755c14532  (L3, 2025-11-06, sha 3755c14532ae, PR #28130)
TITLE: [CPU] Enable torch profiling (#28130)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/cpu_worker.py (+36/-0)
LABELS: ready, v1
BODY: ## Purpose ⏎ The PR  enables profiling for vllm models using `torch.profile` on CPU  ⏎  ⏎ ## Usage ⏎  ⏎ export VLLM_TORCH_PROFILER_DIR=example_directory ⏎  ⏎ ## Example ⏎  ⏎ ``` ⏎ VLLM_TORCH_PROFILER_DIR=vllm_profile vllm bench throughput --num-prompts 1 --seed 0   --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --input_len 128 --load-format  dummy   --profile ⏎ ``` ⏎  ⏎ Example output for reference:  ⏎ ``` ⏎ (EngineCore_DP0 pid=44565) INFO 11-05 13:22:49 [cpu_worker.py:210] ----- …[truncated]

### L3-c3ee80a01a  (L3, 2025-11-06, sha c3ee80a01ae8, PR #28116)
TITLE: [V0 deprecation]clean up is_v1_supported_oracle (#28116)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: tests/v1/test_oracle.py (+6/-44); vllm/engine/arg_utils.py (+15/-90)
LABELS: ready, v1
BODY: ## Purpose ⏎ Clean up _is_v1_supported_oracle. Let' raise `NotImplementedError` for unsupported feature directly.  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-981cadb35c  (L3, 2025-11-06, sha 981cadb35c19, PR #28181)
TITLE: [Bugfix][Kernel] fix merge attn states when both prefix and suffix are empty (#28181)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.merge.cuda_lse
FILES: csrc/attention/merge_attn_states.cu (+26/-0)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-72b1c2ae2c  (L3, 2025-11-07, sha 72b1c2ae2c2d, PR #27439)
TITLE: [Bugfix] Use latency MOE backend as default for Flashinfer and other misc fixes (#27439)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/_custom_ops.py (+1/-1); vllm/envs.py (+3/-3); csrc/quantization/fp4/nvfp4_quant_kernels.cu (+20/-2); tests/kernels/quantization/test_nvfp4_quant.py (+0/-2); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py (+3/-0); vllm/model_executor/layers/quantization/fp8.py (+7/-0); vllm/model_executor/layers/quantization/modelopt.py (+13/-4)
LABELS: bug, ready
ISSUES: #26070 [Bug]: DSR1 FP4 + DEP8 on B200 fails with TensorRT-LLM throughput kernels
BODY: ## Purpose ⏎ This PR switches the default MOE backend to use Flashinfer TRTLLM MOE kernels which are optimized for the latency scenarios.  ⏎  ⏎ Additionally, I address a few more issues - ⏎ 1. Move the zero initialization for fp4 quantization in the padded scenarios to kernel to avoid extra kernel call for the whole tensor (introduced in https://github.com/vllm-project/vllm/pull/25947.) ⏎ 2. [x] Fix the k_scale and v_scale loading again! ⏎  ⏎ Fixes: #26070 ⏎ ##  …[truncated]

### L3-a736e5ff77  (L3, 2025-11-07, sha a736e5ff770b, PR #28074)
TITLE: [CI] Reduce Blackwell Fusion test runtime by filtering tests and only run all tests in nightly (#28074)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-pipeline.yaml (+26/-1); tests/compile/test_fusions_e2e.py (+5/-7)
LABELS: ready, ci/build
BODY: ## Summary ⏎  ⏎ This PR reduces Blackwell Fusion test runtime in CI by: ⏎ 1. Adding a new optional "Blackwell Fusion E2E Tests" CI group that runs all test combinations (for nightly builds) ⏎ 2. Filtering the non-optional "Blackwell Fusion Tests" to run only a focused subset of tests ⏎ 3. Filtering the "PyTorch Fullgraph Test" to exclude expensive custom ops variants ⏎ 4. Expanding test coverage by adding both custom ops variants (`+` and `-`) for comprehens …[truncated]

### L3-811df41ee9  (L3, 2025-11-07, sha 811df41ee901, PR #27952)
TITLE: Update Flashinfer from `v0.4.1` to `v0.5.2` (#27952)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+4/-8); docker/Dockerfile.nightly_torch (+2/-2); requirements/cuda.txt (+1/-1); tests/kernels/attention/test_flashinfer_trtllm_attention.py (+4/-2)
LABELS: ready, ci/build, nvidia
ISSUES: #27476 [Installation]: FlashInfer Dependency issue due to pre-release apache-tvm-ffi
BODY: - Bump Flashinfer to `v0.5.2` ⏎ - Remove workaround from docker build ⏎ - Allows us to stop needing `--pre` when installing from source (fixes #27476) ⏎  ⏎ _N.B. xformers is also causing `--pre` to be required at the moment_

### L3-4a36681f85  (L3, 2025-11-07, sha 4a36681f8548, PR #27990)
TITLE: [flashinfer][fix] do not check nvcc availability when using pre-downloaded cubins (#27990)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+6/-2)
LABELS: ready, nvidia
BODY: Summary: https://github.com/vllm-project/vllm/pull/26443 adds checking of availability of nvcc as a condition to enable flashinfer moe. In our deployment env, there is no nvcc, so flashinfer moe is disabled ⏎ Differential Revision: D86104899

### L3-608bb14462  (L3, 2025-11-07, sha 608bb1446285, PR #27840)
TITLE: [Attention] Remove max cudagraph size limit of 992 (#27840)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.mla.flashattn
FILES: vllm/v1/attention/backends/flash_attn.py (+0/-7); vllm/v1/attention/backends/mla/flashattn_mla.py (+0/-7)
LABELS: ready, v1
BODY: This is to support cuda graph capturing beyond 992. Tested working for larger size

### L3-d15afc1fd0  (L3, 2025-11-08, sha d15afc1fd05b, PR #28026)
TITLE: Refactor CPU/GPU extension targets for CMake build (#28026)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.mla.flashmla_build
FILES: cmake/external_projects/flashmla.cmake (+2/-2); CMakeLists.txt (+4/-4); cmake/cpu_extension.cmake (+2/-2); cmake/utils.cmake (+34/-37)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ Currently as described in #9129 `define_gpu_extension_target` is used for CPU extensions as well which leads to confusion. ⏎ This PR properly renames that target to `define_extension_target` in `utils.camke` and adds the appropriate implementation for the new target. ⏎  ⏎ ## Test Plan ⏎ No brand new test is needed. I used the instructions to build CPU package as described here: ⏎ https://docs.vllm.ai/en/latest/getting_started/installation/cpu.ht …[truncated]

### L3-2108a571d7  (L3, 2025-11-09, sha 2108a571d7ee, PR #26696)
TITLE: [DCP] Support dcp kv_cache interleave size > 1 (#26696)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.mla.common_v1, L3.dispatch.abstract_interface
FILES: vllm/attention/ops/common.py (+1/-0); vllm/v1/attention/backends/flash_attn.py (+11/-2); vllm/v1/attention/backends/mla/common.py (+77/-75); vllm/v1/attention/backends/utils.py (+38/-0); tests/distributed/test_context_parallel.py (+7/-0); tests/v1/worker/test_gpu_model_runner.py (+2/-0); vllm/config/parallel.py (+11/-0); vllm/config/vllm.py (+17/-0); vllm/engine/arg_utils.py (+6/-0); vllm/v1/worker/block_table.py (+16/-2); (+2 more)
LABELS: ready, v1
BODY: ## Purpose ⏎ ### 1. cp_kv_cache_interleave_size support ⏎ In dcp scenario, kv_cache is split across dcp ranks, current implementation ([#23734](https://github.com/vllm-project/vllm/pull/23734)) split kv_cache with a token-level interleave style: token_idx i is stored on GPU whose dcp_rank == i % dcp_world_size. ⏎  ⏎ For the convenience of pd disaggregate support, we add the cp_kv_cache_interleave_size argument to control the interleave size of kv_cache s …[truncated]

### L3-f080a83511  (L3, 2025-11-10, sha f080a8351151, PR #24490)
TITLE: [RFC][ROCm][AITER] Keep all AITER kernels in `_aiter_ops` class like `_custom_ops` and `_ipex_ops` (#24490)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.common_v1, L3.mla.rocm_aiter, L3.platform.rocm_selection
FILES: vllm/attention/ops/rocm_aiter_mla.py (+0/-105); vllm/v1/attention/backends/mla/common.py (+21/-34); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+2/-7); docs/design/moe_kernel_features.md (+1/-1); tests/kernels/moe/test_moe.py (+6/-5); tests/model_executor/test_enabled_custom_ops.py (+14/-27); vllm/_aiter_ops.py (+941/-0); vllm/envs.py (+4/-4); vllm/model_executor/layers/fused_moe/fused_moe.py (+7/-8); vllm/model_executor/layers/fused_moe/layer.py (+43/-40); (+15 more)
LABELS: documentation, rocm, ready, v1, deepseek
BODY: ## Purpose ⏎  ⏎ This PR introduces `_aiter_ops.py` as proposed in the [RFC here](https://github.com/vllm-project/vllm/issues/21504). The `aiter_ops` namespace provides several key benefits: ⏎  ⏎ - Centralized kernel registration: Ensures that kernels from the aiter package are properly registered ⏎  ⏎ - Environment availability checks: Encapsulates aiter support detection and environment compatibility validation ⏎  ⏎ - Reduced code duplication: Eliminates the ne …[truncated]

### L3-9c84ca8293  (L3, 2025-11-10, sha 9c84ca829303, PR #27889)
TITLE: [FA/Chore] Bump FA version for FP8 two-level accumulation  (#27889)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build
ISSUES: #23813 [Bug]: Accuracy under FA3 in FP8 changes drastically with TP size on very long context length
BODY: vLLM side of: https://github.com/vllm-project/flash-attention/pull/104 ⏎  ⏎ FIX: https://github.com/vllm-project/vllm/issues/23813

### L3-34553b9d27  (L3, 2025-11-10, sha 34553b9d2702, PR #27492)
TITLE: [Performance] Support FP8 flashinfer TRTLLM MOE on Qwen3 and Qwen-3next (#27492)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/config.py (+21/-0); vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py (+12/-14); vllm/model_executor/layers/fused_moe/layer.py (+20/-0); vllm/model_executor/layers/quantization/fp8.py (+7/-7); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+14/-9); vllm/model_executor/models/qwen3_moe.py (+2/-0); vllm/model_executor/models/qwen3_next.py (+2/-0)
LABELS: ready, ci/build, qwen, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ - Integrate multiple routing methods for FP8 flashinfer trtllm MOE, currently only DS and Llama4 ⏎ - Add  FP8 flashinfer trtllm MOE support on Qwen3 and Qwen3-next  ⏎  ⏎ ## Test Plan ⏎ **Qwen3-Next-80B-A3B-Instruct-FP8 on 2xB200 TP2** ⏎ ``` ⏎ VLLM_USE_FLASHINFER_MOE_FP8=1 VLLM_FLASHINFER_MOE_BACKEND=latency VLLM_USE_DEEP_GEMM=0 VLLM_USE_TRTLLM_ATTENTION=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \ ⏎     --max …[truncated]

### L3-e8697faf03  (L3, 2025-11-10, sha e8697faf037d, PR #28370)
TITLE: [V0 deprecation] Remove no longer used `get_metadata_cls` (#28370)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v1_backend, L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.rocm.aiter_unified, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.flashattn, L3.mla.flashmla_sparse, L3.mla.rocm_aiter, L3.dispatch.abstract_interface, L3.flex_attention, L3.tree_attention
FILES: vllm/attention/backends/abstract.py (+0/-9); vllm/v1/attention/backends/cpu_attn.py (+0/-4); vllm/v1/attention/backends/flash_attn.py (+0/-5); vllm/v1/attention/backends/flashinfer.py (+0/-4); vllm/v1/attention/backends/flex_attention.py (+0/-5); vllm/v1/attention/backends/mla/common.py (+0/-5); vllm/v1/attention/backends/mla/flashattn_mla.py (+0/-4); vllm/v1/attention/backends/mla/flashmla.py (+0/-4); vllm/v1/attention/backends/mla/flashmla_sparse.py (+0/-5); vllm/v1/attention/backends/mla/indexer.py (+0/-5); (+10 more)
LABELS: rocm, tpu, ready, v1
BODY: `get_metadata_cls` was a V0 specific thing; remove it  ⏎  ⏎ `make_test_metadata` uses it but is in-turn no longer used by anything

### L3-57201a6a4c  (L3, 2025-11-10, sha 57201a6a4c53, PR #28323)
TITLE: Fix rotary embedding benchmark script (#28323)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_rope.py (+64/-90)
LABELS: performance, ready
BODY: ## Purpose ⏎  ⏎ Currently running the `benchmark_rope.py` script failed with the error (error pasted below). This PR fixes the benchmark script: ⏎  ⏎ * Fixed error ⏎ * Remove batched rope benchmark, because `batched_rotary_embedding` kernel was removed in https://github.com/vllm-project/vllm/pull/24789 ⏎ * Compare kernel differences between pytorch, flashinfer and vllm custom op ⏎  ⏎ Error: ⏎  ⏎ ``` ⏎ Traceback (most recent call last): ⏎   File "/root/workspace/vllm-pro …[truncated]

### L3-cc079763c5  (L3, 2025-11-10, sha cc079763c59a, PR #28253)
TITLE: [BugFix] Avoid calling KV connector layer APIs when metadata is unset (#28253)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+4/-0); vllm/distributed/kv_transfer/kv_connector/v1/base.py (+8/-1); vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py (+6/-0)
LABELS: ready, kv-connector
ISSUES: #26675 [Bug]:AssertionError: assert self._connector_metadata is not None when using graph mode
BODY: ## Purpose ⏎ Fixes #26675, following the approach suggested by [@markmc](https://github.com/vllm-project/vllm/pull/27026#issuecomment-3456079661) in #27026. ⏎ This change temporarily disables graph capture for the KV connector path by skipping calls to KV connector layer APIs when the connector metadata is unset. ⏎  ⏎ ## Test Plan ⏎ Reproduce the scenario described in [#26675](https://github.com/vllm-project/vllm/issues/26675) (full-graph capture with KV c …[truncated]

### L3-a5a790eea6  (L3, 2025-11-10, sha a5a790eea603, PR #27232)
TITLE: [Bugfix] Ensure calculated KV scales are applied in attention. (#27232)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+7/-22); .buildkite/test-pipeline.yaml (+5/-2); tests/compile/test_full_graph.py (+8/-2); vllm/v1/worker/gpu_model_runner.py (+9/-10)
LABELS: ready, ci/build, v1
BODY: ### Purpose: ⏎  ⏎ Resolves bug https://github.com/vllm-project/vllm/issues/27102 . ⏎  ⏎ ### Test Plan: ⏎ **Throughput** ⏎ ``` ⏎ vllm bench throughput   --model Qwen/Qwen3-8B   --quantization fp8   --kv-cache-dtype fp8_e4m3   --tensor-parallel-size 2   --dataset-name random --input-len 1024 --output-len 256 ⏎ ``` ⏎  ⏎ **E2E Correctness** ⏎ Server with KV scale calculation ON (remove --calculate-kv-scales flag for OFF case): ⏎  ⏎ ``` ⏎ vllm serve Qwen/Qwen3-8B \ ⏎     --tensor- …[truncated]

### L3-0bf29fadf5  (L3, 2025-11-10, sha 0bf29fadf5f8, PR #28420)
TITLE: [Test] Remove old non-varlen FA2 test (#28420)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_flash_attn.py (+0/-119)
LABELS: ready
BODY: ## Purpose ⏎ This is testing a function that is no longer used anywhere in vLLM, so this PR removes it. This helps eliminate dependence on FA2. ⏎  ⏎ --- ⏎ [details omitted]

### L3-39029d5192  (L3, 2025-11-11, sha 39029d519276, PR #28404)
TITLE: [CI/Test Fix] Fix CP tests on Blackwell (#28404)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/common.py (+0/-1); tests/distributed/test_context_parallel.py (+12/-0)
LABELS: ready
ISSUES: #28400 [CI Failure]: Nightly Failure B200 (Context Parallel)
BODY: FIX: https://github.com/vllm-project/vllm/issues/28400

### L3-2e78150d24  (L3, 2025-11-11, sha 2e78150d24e3, PR #28417)
TITLE: [CI] Add mergify rules for `nvidia` label (#28417)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/mergify.yml (+17/-0)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ Essentially it should match with any files or paths that contain cuda, cutlass, flashinfer, or trtllm ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-b30dfa03c5  (L3, 2025-11-11, sha b30dfa03c564, PR #24794)
TITLE: [Attention] Refactor CUDA attention backend selection logic (#24794)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases(introducing)
ARTIFACT_HINTS: L3.xformers.v1_backend, L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.flashmla_sparse, L3.dispatch.selector, L3.dispatch.registry, L3.dispatch.abstract_interface, L3.platform.cuda_selection, L3.platform.rocm_selection, L3.flex_attention, L3.tree_attention
FILES: vllm/attention/backends/abstract.py (+139/-10); vllm/attention/backends/registry.py (+168/-84); vllm/attention/layer.py (+39/-29); vllm/attention/selector.py (+46/-78); vllm/config/cache.py (+9/-1); vllm/config/model.py (+4/-4); vllm/config/multimodal.py (+11/-21); vllm/engine/arg_utils.py (+2/-2); vllm/envs.py (+3/-3); vllm/platforms/cpu.py (+6/-6); (+51 more)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, tpu, speculative-decoding, ready, ci/build
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:07a606aa7e (fix PR 28534)
BODY: ## Purpose ⏎ `CudaPlatformBase.get_attention_backend_cls` has gotten complex and messy over time. This PR cleans up the logic (without changing the behavior) and standardizes the interface. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-8c32c6e4b4  (L3, 2025-11-11, sha 8c32c6e4b485, PR #28389)
TITLE: [Misc] fix typo in DCP comment (#28389)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+1/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ 1. fix typo ⏎  ⏎ ## Test Plan ⏎  ⏎ NA ⏎  ⏎ ## Test Result ⏎  ⏎ NA ⏎  ⏎ --- ⏎ [details omitted]

### L3-a7ef3eb0cd  (L3, 2025-11-11, sha a7ef3eb0cd03, PR #28282)
TITLE: [NIXL] Generalize block-first backend layouts (FlashInfer-like) (#28282)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_nixl_connector.py (+15/-2); vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py (+37/-10)
LABELS: ready, v1, kv-connector
BODY: This PR generalizes the handling of attention backends that make use of a "swapped" layout wrt the historically default one (ie FlashAttn). ⏎ To clarify: ⏎ ``` ⏎ # "Default" layout (FlashAttn) ⏎ 2, num_blocks, N, H, D ⏎ # "Swapped" layout (FlashInfer) ⏎ num_blocks, 2, N, H, D ⏎ ``` ⏎ Mind that in the latter case we're able to register num_layers physical memory regions and handle the K/V splitting only logically. ⏎  ⏎ This PR should hence enable the use of other Fla …[truncated]

### L3-684f254585  (L3, 2025-11-11, sha 684f2545851e, PR #27363)
TITLE: Prefer FlashAttention MLA as default over FlashMLA (#27363)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+2/-2)
LABELS: rocm, speculative-decoding, ready, v1, kv-connector, nvidia
BODY: ## Purpose ⏎ Based on benchmarking in #26835, FlashAttention MLA is faster than FlashMLA across head counts, batch sizes, and query lengths on Hopper. This PR sets it as the preferred backend. ⏎  ⏎ Based on #24794, merge that first ⏎  ⏎ Note: Results shown below apply the fix of #27368, FlashMLA performance is worse prior to the bugfix ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎ ``` ⏎ Model Parameter Sweep Results: ⏎  ⏎ num_q_heads = 16 ⏎               Attention Benchmark Resul …[truncated]

### L3-6c3c0f8235  (L3, 2025-11-11, sha 6c3c0f8235ca, PR #27931)
TITLE: [Kernel] Optimize rms_norm kernel (#27931)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: csrc/dispatch_utils.h (+29/-0); csrc/layernorm_kernels.cu (+28/-11); csrc/layernorm_quant_kernels.cu (+29/-14)
LABELS: ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Optimize rms_norm kernel: ⏎ * Do vectorized load and store using `vec_n_t<scalar_t, VEC_SIZE>`. ⏎ * Since input, weight, output tensors are the same dtype, and all of their last dimension are hidden_size, we can get the largest safe vec_size by calculating the number that can divide hidden_size and fits in 16 bytes, to avoid misalignment. ⏎ * For large num_tokens (>=256), use smaller blocks to increase SM concurrency. ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ pyt …[truncated]

### L3-76e4dcf225  (L3, 2025-11-11, sha 76e4dcf225e4, PR #26971)
TITLE: [Misc] Remove unused attention prefix prefill ops functions (#26971)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/prefix_prefill.py (+0/-210); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+0/-3)
LABELS: frontend, ready
BODY: This removes a couple of functions that don't seem to be used anywhere and are reported as `reportUnusedFunction` by `basedpyright`.

### L3-9f0247cfa4  (L3, 2025-11-11, sha 9f0247cfa40a, PR #27611)
TITLE: `VLLM_USE_TRITON_FLASH_ATTN` V0 variable deprecation (#27611)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.triton.flash_attention_rocm, L3.platform.rocm_selection
FILES: vllm/attention/ops/triton_flash_attention.py (+0/-932); vllm/envs.py (+0/-6); vllm/platforms/rocm.py (+2/-19); vllm/v1/attention/backends/mla/triton_mla.py (+7/-48); .buildkite/scripts/hardware_ci/run-amd-test.sh (+2/-6); tests/kernels/test_triton_flash_attention.py (+0/-516); tests/models/language/pooling/test_classification.py (+0/-6); tests/models/language/pooling/test_embedding.py (+0/-7); tests/models/language/pooling/test_mm_classifier_conversion.py (+0/-13); tests/models/language/pooling/test_reward.py (+0/-6); (+5 more)
LABELS: documentation, rocm, ready, ci/build, v1, multi-modality
BODY: Deprecated the `VLLM_USE_TRITON_FLASH_ATTN` environment variable. This variable was used for ROCm platform and was V0 specific. Therefore, we deprecated it.

### L3-412e153df5  (L3, 2025-11-11, sha 412e153df557, PR #28269)
TITLE: [Feature] Allow configuring FlashInfer workspace size (#28269)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1
FILES: vllm/envs.py (+6/-0); vllm/v1/attention/backends/flashinfer.py (+3/-3); vllm/v1/attention/backends/mla/common.py (+7/-9)
LABELS: ready, v1, nvidia
BODY: Resubmission of https://github.com/vllm-project/vllm/pull/25344 ⏎  ⏎ Add `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` environment variable to configure FlashInfer workspace size

### L3-d23539549a  (L3, 2025-11-12, sha d23539549a6d, PR #28491)
TITLE: Use FLASHINFER MLA backend when testing fp8_kv_scale_compile (#28491)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/test_full_graph.py (+16/-4)
LABELS: ready
ISSUES: #28468 [CI Failure]: Blackwell Fusion E2E Tests
BODY: Fixes #28468 ⏎  ⏎ Current Blackwell Fusion E2E Tests uses CUTLASS MLA backend causing IMA.

### L3-e553424919  (L3, 2025-11-12, sha e55342491968, PR #28424)
TITLE: [CI/Build] Refactor Attention backend for test_prefix_prefill from xformers to SDPA (#28424)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_prefix_prefill.py (+194/-116)
LABELS: ready
BODY: ## Purpose ⏎ 1) This PR will help https://github.com/vllm-project/vllm/pull/28287#discussion_r2507610717 by changing the ground truth for attention backend from xformers to pytorch SDPA ⏎ 2) Fixes some incompatibilities on AMD:  ⏎ a. The ROCm paged attention kernel expects 32-bit `int` tensors, but the test passes 64-bit `torch.long` tensors. ⏎ b. ROCm paged attention kernel only supports `auto`, `fp8`, and `fp8_e4m3` KV cache dtypes. ⏎  ⏎ ## Test Plan ⏎ H100( …[truncated]

### L3-b9ce9a3013  (L3, 2025-11-12, sha b9ce9a301341, PR #28447)
TITLE: [BugFix] Add fallback path in `apply_rotary_pos_emb_flashattn` for non-cuda platforms (#28447)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/keye.py (+7/-0)
LABELS: ready, nvidia
BODY: ## Purpose ⏎ This PR enables `Kwai-Keye/Keye-VL-8B-Preview` and `Kwai-Keye/Keye-VL-1_5-8B` on XPU by adding a fallback path in the `apply_rotary_pos_emb_flashattn` function. Otherwise, the code would throw an error that `apply_rotary_emb` is not available. ⏎  ⏎ ## Test Plan ⏎ ```bash ⏎ python3 examples/offline_inference/vision_language_multi_image.py -m keye_vl --seed 42 -n 1 ⏎ ``` ⏎  ⏎ ## Test Result ⏎ ```bash ⏎ <analysis>This question asks for the content of the i …[truncated]

### L3-1761dea1a8  (L3, 2025-11-12, sha 1761dea1a856, PR #27733)
TITLE: [BugFix]: --enable-lora with model granite-4.0-micro crash (#27733)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/granitemoehybrid.py (+3/-0)
LABELS: ready
ISSUES: #27620 [Bug]: Setting --enable-lora with model granite-4.0-micro crashes server
BODY: ## Purpose ⏎  ⏎ Fix: #27620 ⏎  ⏎ note: must set `--enforce-eager` otherwise vllm will crash when use vscode debug mode ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ start up log: ⏎ ``` ⏎ INFO 10-29 09:17:34 [__init__.py:224] Automatically detected platform cuda. ⏎ (APIServer pid=6796) INFO 10-29 09:17:44 [api_server.py:1882] vLLM API server version 0.11.0rc2.dev330+ge3938e2e9.d20251009 ⏎ (APIServer pid=6796) INFO 10-29 09:17:44 [utils.py:239] non-default args: {'model': '/home …[truncated]

### L3-7f829be7d3  (L3, 2025-11-12, sha 7f829be7d3d7, PR #27954)
TITLE: [CPU] Refactor CPU attention backend (#27954)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.dispatch.registry, L3.dispatch.abstract_interface
FILES: csrc/cpu/attention.cpp (+0/-798); docker/Dockerfile.cpu (+4/-0); vllm/_custom_ops.py (+82/-0); vllm/attention/backends/registry.py (+2/-1); vllm/engine/arg_utils.py (+0/-3); vllm/platforms/cpu.py (+11/-26); vllm/utils/__init__.py (+0/-1); vllm/v1/attention/backends/cpu_attn.py (+264/-717); vllm/v1/attention/backends/utils.py (+1/-1); vllm/v1/worker/cpu_model_runner.py (+1/-13); (+24 more)
LABELS: documentation, ready, ci/build, v1
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ This PR refactors CPU attention backend, includes: ⏎ - clean up unused code and simplifiy metadata ⏎ - a unified kernel supports chunked prefill, sliding window, alibi, softcap, and sink ⏎ - renamed ```TorchSDPABackend``` to ```CPUAttentionBackend``` for less misunderstandings. For now the ```TORCH_SDPA``` tag is only used for ViT attention ⏎ - better performance on both of prefill and decode ⏎ - enable more related unit tests ⏎  ⏎ cc @fadara01 @Ak …[truncated]

### L3-a4730c1b4f  (L3, 2025-11-12, sha a4730c1b4fa2, PR #28520)
TITLE: [XPU]Fix crash due to removed VLLM_USE_V1 attribute (#28520)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/platforms/xpu.py (+3/-2)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ 1. Fix crash due to removed VLLM_USE_V1 attribute  ⏎ 2.  Fix AttentionBackendEnum.FLASH_ATTN fails because **AttentionBackendEnum** is None. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-10138c92a5  (L3, 2025-11-12, sha 10138c92a5c7, PR #28112)
TITLE: [V0 deprecation] Deprecate use_v1 parameter (#28112)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.dispatch.selector, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/attention/selector.py (+30/-11); tests/plugins/vllm_add_dummy_platform/vllm_add_dummy_platform/dummy_platform.py (+0/-1); vllm/platforms/cpu.py (+0/-3); vllm/platforms/cuda.py (+0/-7); vllm/platforms/interface.py (+0/-1); vllm/platforms/rocm.py (+0/-7); vllm/platforms/tpu.py (+0/-3); vllm/platforms/xpu.py (+1/-2)
LABELS: rocm, tpu, ready, nvidia
BODY: ## Purpose ⏎ `use_v1` for `get_attn_backend_cls` is useless now. Since this is an interface impact change, let's deprecate it  and remove it in the next release.  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-bc5bd45c7d  (L3, 2025-11-12, sha bc5bd45c7d1a, PR #28271)
TITLE: [Refactor] Remove redundant TP gather/split in split_qkv in QwenVL (#28271)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen2_5_vl.py (+0/-30); vllm/model_executor/models/qwen2_vl.py (+1/-12)
LABELS: ready, qwen
DEEP_STUDY: deep-study: this PR was reverted by PR 30542 (partial_revert, reason=correctness_or_accuracy)
BODY: ## Purpose ⏎  ⏎ This code path uses head-parallel attention, where each rank holds full Q/K/V vectors for its own subset of heads. All attention backends operate per-rank on local heads. Therefore, the previous `tp_size > 1` communication logic (`all_gather_interleave` and re-slicing) is unnecessary and introduces redundant synchronization overhead. Removing it preserves correctness and improves performance. ⏎  ⏎ ## Test Plan ⏎  ⏎ It has been tested in [vllm …[truncated]

### L3-728a9eb70e  (L3, 2025-11-12, sha 728a9eb70ee3, PR #27816)
TITLE: [Misc] Refactor Attention kv transfer methods into decorator (#27816)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+39/-76); vllm/attention/utils/kv_transfer_utils.py (+60/-0)
LABELS: ready
BODY: Small quality of life improvement by removing some of the kv transfer-specifc code in `layer.py` and refactoring that into a decorator. I believe the on entry-on exit pattern (wait_read-wait_write) here is very suitable for that. ⏎ The result is simply that there's less non-attention related code in the file. Behavior should be unchanged. ⏎  ⏎ Also, I found that after grouping some common boilerplate code for both `maybe_save_kv_layer_to_connector` and …[truncated]

### L3-a543e678b4  (L3, 2025-11-12, sha a543e678b45a, PR #28561)
TITLE: [Bugfix] Fix SM100 gpt-oss regression due to faulty attn sink support (#28561)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+21/-10); vllm/v1/attention/backends/flashinfer.py (+15/-0)
LABELS: ready, v1, gpt-oss, nvidia
BODY: ## Purpose ⏎  ⏎ Fix regression introduced by https://github.com/vllm-project/vllm/pull/24794 where gpt-oss on Blackwell shows triton as the only valid attention backend ⏎  ⏎ ``` ⏎ vllm serve openai/gpt-oss-20b --load-format dummy ⏎ ... ⏎ (EngineCore_DP0 pid=210162) INFO 11-12 10:18:02 [cuda.py:408] Valid backends: ['TRITON_ATTN'] ⏎ (EngineCore_DP0 pid=210162) INFO 11-12 10:18:02 [cuda.py:417] Using TRITON_ATTN backend. ⏎ ``` ⏎  ⏎ FlashInfer can support attention sinks …[truncated]

### L3-ca00b1bfc6  (L3, 2025-11-12, sha ca00b1bfc69e, PR #28383)
TITLE: [ROCm][BugFix] Remove the usage of `device_info` from aiter (#28383)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+5/-6)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎ Many user report the following error encountered after this PR merged  https://github.com/vllm-project/vllm/pull/25763 ⏎  ⏎ ``` ⏎ from aiter.ops.triton.utils.device_info import get_num_sms ⏎ (EngineCore_DP0 pid=230496) ModuleNotFoundError: No module named 'aiter.ops.triton.utils.device_info' ⏎ ``` ⏎ And we notice many user's aiter doesn't have this module, This PR remove its usage to maintain the backward compatibility to aiter ⏎  ⏎ ## Test Plan ⏎ gsm8k …[truncated]

### L3-c33b87e777  (L3, 2025-11-12, sha c33b87e7778d, PR #28600)
TITLE: Use official xformers-0.0.33 built for PT 2.9 (#28600)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: requirements/cuda.txt (+1/-2)
LABELS: ready, ci/build, nvidia
ISSUES: #27880 [Installation]: [HELP]How to install the latest main version of vllm
BODY: ## Purpose ⏎  ⏎ `xformers-0.0.33` has been released today for PyTorch 2.9 https://pypi.org/project/xformers/#history, and we can finally switch to that version instead of the custom one we build on PyTorch ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/27880 https://github.com/vllm-project/vllm/issues/27851 ⏎ ## Test Plan ⏎  ⏎ CI ⏎  ⏎ cc @simon-mo @ywang96 @ProExpertProg

### L3-304419576a  (L3, 2025-11-13, sha 304419576ae9, PR #28479)
TITLE: [Perf] Refactor cudagraph_support to enable full CUDA graphs for spec decoding with FlashInfer (#28479)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.flashmla_sparse, L3.mla.rocm_aiter, L3.dispatch.abstract_interface
FILES: vllm/attention/layers/chunked_local_attention.py (+1/-1); vllm/v1/attention/backends/flash_attn.py (+1/-1); vllm/v1/attention/backends/flashinfer.py (+23/-15); vllm/v1/attention/backends/mla/cutlass_mla.py (+1/-1); vllm/v1/attention/backends/mla/flashattn_mla.py (+1/-1); vllm/v1/attention/backends/mla/flashinfer_mla.py (+1/-1); vllm/v1/attention/backends/mla/flashmla.py (+1/-1); vllm/v1/attention/backends/mla/flashmla_sparse.py (+1/-1); vllm/v1/attention/backends/mla/indexer.py (+1/-1); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+1/-1); (+8 more)
LABELS: documentation, rocm, ready, v1, nvidia
ISSUES: #26856 [Performance]: FalshInfer attn backend. Use dynamic AttentionCGSupport
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Revised implementation of https://github.com/vllm-project/vllm/pull/26937 ⏎  ⏎ This PR makes `_cudagraph_support` a private member and uses `get_cudagraph_support(vllm_config, kv_cache_spec)`. Also updates `_check_and_update_cudagraph_mode` to consider support per-backend, per-kv-group.  ⏎  ⏎ TRTLLM-gen kernels support full cuda graphs, but are only used with FlashInfer on Blackwell under certain conditions. ⏎ It might not be safe to change Fla …[truncated]

### L3-8832fff972  (L3, 2025-11-13, sha 8832fff972b2, PR #28599)
TITLE: [BugFix] Fix `mm_encoder_attn_backend` arg type checking (#28599)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/multimodal.py (+3/-0); .buildkite/test-pipeline.yaml (+3/-1)
LABELS: bug, ready, ci/build
BODY: Currently, use of `--mm-encoder-attn-backend` fails with the following error: ⏎  ⏎ ``` ⏎ self = typing.Any, obj = 'FLASH_ATTN' ⏎  ⏎     def __instancecheck__(self, obj): ⏎         if self is Any: ⏎ >           raise TypeError("typing.Any cannot be used with isinstance()") ⏎ E           TypeError: typing.Any cannot be used with isinstance() ⏎  ⏎ ../../../.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/typing.py:530: TypeError ⏎ ``` ⏎  ⏎ There is a [t …[truncated]

### L3-07a606aa7e  (L3, 2025-11-13, sha 07a606aa7eb3, PR #28534)
TITLE: [CI Failure] Fix backend selection for encoder-only models (#28534)
SOURCES: path_core, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.mla.flashmla_sparse, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.platform.cuda_selection, L3.platform.rocm_selection, L3.flex_attention
FILES: vllm/attention/backends/abstract.py (+14/-0); vllm/attention/layer.py (+1/-0); vllm/attention/layers/encoder_only_attention.py (+5/-1); vllm/attention/selector.py (+5/-0); vllm/v1/attention/backends/cpu_attn.py (+11/-0); vllm/v1/attention/backends/flash_attn.py (+12/-0); vllm/v1/attention/backends/flex_attention.py (+7/-0); vllm/v1/attention/backends/mla/flashmla_sparse.py (+5/-5); vllm/platforms/cpu.py (+1/-0); vllm/platforms/cuda.py (+10/-0); (+4 more)
LABELS: rocm, tpu, ready, v1, nvidia
DEEP_STUDY: deep-study correctness case vllm:07a606aa7e: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=#24794
BODY: ## Purpose ⏎  ⏎ After #24794, encoder-only models (e.g., BERT) fail to initialize because the TRITON_ATTN backend is selected by default, but it doesn't support encoder self-attention, causing: ⏎ ``` ⏎ NotImplementedError: Encoder self-attention and encoder/decoder cross-attention are not implemented for TritonAttentionImpl ⏎ ``` ⏎  ⏎ This PR implemented an opt-in approach for attention type support: ⏎  ⏎   1. Added `supports_attn_type()` method to `AttentionBacke …[truncated]

### L3-968060c15a  (L3, 2025-11-13, sha 968060c15adc, PR #28526)
TITLE: [bugfix] correct local_chunk_len for DCP in reorg_kvcache with long context (#28526)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+25/-4)
LABELS: ready, v1
BODY: Fix the issues https://github.com/vllm-project/vllm/issues/28476 https://github.com/vllm-project/vllm/issues/28411 ⏎ The previous DCP implementation did not account for cases where long contexts were split to multi-chunks. This has now been addressed by updating the correct chunked local_chunk_len based on the chunk_size and local_context_len. ⏎  ⏎ CC @LucasWilkinson @cjackal @Nemo-G

### L3-f9f3b596f3  (L3, 2025-11-13, sha f9f3b596f374, PR #28660)
TITLE: [Attention][Bugfix] Fix FA sink support (#28660)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+6/-0)
LABELS: bug, ready, v1, gpt-oss
BODY: ## Purpose ⏎ #24794 didn't mark FlashAttention as supporting sink, so GPT-OSS ends up selecting TRITON_ATTN, causing failure. This PR fixes the issue. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ python examples/offline_inference/spec_decode.py \ ⏎   --method "eagle3" \ ⏎   --tp 1 \ ⏎   --print-output \ ⏎   --model-dir openai/gpt-oss-20b" \ ⏎   --eagle-dir "RedHatAI/gpt-oss-20b-speculator.eagle3" \ ⏎   --dataset_name "hf" \ ⏎   --dataset_path "philschmid/mt-bench" \ ⏎   --num-spec-tokens 3 ⏎  …[truncated]

### L3-86d15bfd8d  (L3, 2025-11-13, sha 86d15bfd8d68, PR #28535)
TITLE: [Hardware][PowerPC] Fix fp16 compilation error for Power in cpu attention backend and bump oneDNN version (#28535)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: cmake/cpu_extension.cmake (+2/-2); csrc/cpu/cpu_attn_impl.hpp (+2/-0)
LABELS: ready, ci/build
BODY: # Purpose ⏎  ⏎ Fix Power compilation failure: PR [#27954](https://github.com/vllm-project/vllm/pull/27954) ⏎  refactored the attention backend for CPU, but the build fails on Power due to the use of fp16, which is not supported on this architecture. This PR addresses that issue (also mentioned in my review on the original PR). ⏎  ⏎ Update oneDNN to v3.10: The oneDNN PR [uxlfoundation/oneDNN#4002](https://github.com/uxlfoundation/oneDNN/pull/4002) ⏎  — includ …[truncated]

### L3-8da2f28f53  (L3, 2025-11-13, sha 8da2f28f53c1, PR #28618)
TITLE: [ROCm][BugFix]Fix `get_cu_count` in rocm_aiter_fa.py (#28618)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+2/-1)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎ Sorry for missing this PR https://github.com/vllm-project/vllm/pull/28383 when fix `get_cu_count` issue, please take a look again ⏎ @tjtanaa @DarkLight1337  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-0aecd9138f  (L3, 2025-11-13, sha 0aecd9138f45, PR #28678)
TITLE: [Misc] Update xformers to 0.33.0.post1 (#28678)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: requirements/cuda.txt (+1/-1)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-15ae8e0784  (L3, 2025-11-13, sha 15ae8e0784d3, PR #28432)
TITLE: [Bugfix][CI/Test][Spec Decode] Fix illegal memory access in offline_inference/spec_decode.py (Issue  27619) (#28432)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/triton_reshape_and_cache_flash.py (+4/-2)
LABELS: ready, v1
ISSUES: #27619 [Bug][CI Failure]: EAGLE Spec Decode failing with Triton Attention Backend
BODY: This PR fixes an illegal memory access that occurs when running ⏎  ⏎ `python3 offline_inference/spec_decode.py --test --method eagle --num_spec_tokens 3 --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0 --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill --max-model-len 2048 ⏎ ` ⏎  ⏎ The number of tokens, `num_tokens`, is used to launch the grid in `triton_reshape_and_cache_flash` which may be slightly larger than the `slot_ …[truncated]

### L3-3f8a874065  (L3, 2025-11-14, sha 3f8a8740656f, PR #27134)
TITLE: [Kernels] Enable FlashInfer FP8 Blockscale on SM90 (for TEP DSR1) (#27134)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py (+22/-1); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py (+97/-50); vllm/model_executor/layers/quantization/fp8.py (+36/-12); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+24/-5)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ **DeepSeek FP8 block-scale path** ⏎  - Introduces and propagates use_deepseek_fp8_block_scale through prepare/finalize, GEMM selection, and the FlashInfer kernel call. ⏎  - In this mode, activations are not quantized in prepare (a1q = a1, a1q_scale = None); the kernel consumes per-block weight scales instead. ⏎  - Expert GEMM receives quant_scales=[w1_scale, w2_scale] and the flag use_deepseek_fp8_block_scale=True to activate the block-scal …[truncated]

### L3-8cc40f8992  (L3, 2025-11-14, sha 8cc40f89926f, PR #28429)
TITLE: [Attention] Bump FA for removed method (#28429)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ vLLM-side PR for https://github.com/vllm-project/flash-attention/pull/107. Removes unused `flash_attn_with_kvcache` and sets flags for build speedup + size reduction. ⏎  ⏎ It looks like we see a 20% reduction in wheel size (481MB --> 380MB) ⏎  ⏎ [Recent commit on main](https://buildkite.com/vllm/ci/builds/39043/steps/canvas?jid=019a831f-f663-40fc-82a9-65b2e8b13e95#019a831f-f663-40fc-82a9-65b2e8b13e95/7-3851): ⏎ ``` ⏎ [2025-11-14T16:34:18Z] #30 0.8 …[truncated]

### L3-622e6106a9  (L3, 2025-11-14, sha 622e6106a9e3, PR #28681)
TITLE: [CPU][Bugfix] Fix Apple Silicon M1 compilation failure (#28681)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: csrc/cpu/cpu_attn_impl.hpp (+28/-0)
LABELS: bug, ready, cpu
DEEP_STUDY: deep-study correctness case vllm:622e6106a9: class=hardware_compiler_specific; symptom=compile_or_build_failure; introducing=unknown
BODY: ## Purpose ⏎  ⏎ The CPU attention backend refactor introduced unconditional use of BF16Vec16, which is only available on ARM platforms with native BF16 support (ARMv8.6-A extension). M1 chips and other older ARM processors lack this hardware capability. ⏎  ⏎ Additionally, macOS doesn't support `_SC_LEVEL2_CACHE_SIZE` sysconf, causing L2 cache detection to fail. ⏎  ⏎ Errors found: ⏎ - `error: no type named 'BF16Vec16' in namespace 'vec_op'` in `cpu_attn_impl.hp …[truncated]

### L3-4516d44b7f  (L3, 2025-11-14, sha 4516d44b7f99, PR #25438)
TITLE: [DCP] Support Decode Context Parallel (DCP) for GQA with Flashinfer (#25438)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/config/model.py (+8/-0); vllm/utils/flashinfer.py (+9/-0); vllm/v1/attention/backends/flashinfer.py (+294/-49); tests/distributed/test_context_parallel.py (+15/-2); vllm/v1/executor/multiproc_executor.py (+5/-0)
LABELS: ready, ci/build, v1, qwen, nvidia
BODY: ## Purpose ⏎ This PR adds Decode Context Parallel (DCP) support for GQA follwing PR [#23734](https://github.com/vllm-project/vllm/pull/23734) and PR [#24864](https://github.com/vllm-project/vllm/pull/24864). Current implementation based on FlashInfer Attention. ⏎  ⏎ FlashInfer inserts the current query KV into the cache before computation. Each query then attends to both its own KV and the context KV on the local device, with LSE applied to correct the …[truncated]

### L3-9324e10275  (L3, 2025-11-14, sha 9324e10275cc, PR #28537)
TITLE: Fix KV sharing fast prefill with cudagraph enabled (#28537)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+2/-13); tests/v1/e2e/test_kv_sharing_fast_prefill.py (+14/-43); vllm/v1/worker/gpu_model_runner.py (+1/-1)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Purpose ⏎  ⏎ `--kv-sharing-fast-prefill` works if using `--enforce-eager` but not without: ⏎  ⏎ ``` ⏎ File "/vllm/vllm/v1/attention/backends/utils.py", line 1005, in __init__ ⏎    common_attn_metadata.logits_indices_padded is not None ⏎ AssertionError ⏎ ``` ⏎  ⏎ This is because we now build attention metadata for cudagraph capture during dummy runs but this doesn't have logits information.  ⏎  ⏎ As a fix, we can just skip this assertion which will set `logits_indices …[truncated]

### L3-db56a59970  (L3, 2025-11-14, sha db56a59970a8, PR #28702)
TITLE: [BugFix] Fix FA3 IMA with FULL_AND_PIECEWISE and cascade attention (default) (#28702)
SOURCES: path_core, subject_keyword, release_notes, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+4/-2); tests/kernels/attention/test_cascade_flash_attn.py (+1/-0)
LABELS: bug, ready, v1
DEEP_STUDY: deep-study correctness case vllm:db56a59970: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: Fix a subtle bug where the `get_scheduler_metadata` doesn't match the `flash_attn_varlen_func` in cascade attention; when full cudagraphs is enabled we force `num_splits` to `VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH` to make sure the `get_scheduler_metadata` always matches what the cudagraph was captured with. In the case of cascade attention this information is not piped into `cascade_attention`; this PR makes sure they match.

### L3-e5c78956c0  (L3, 2025-11-14, sha e5c78956c0c5, PR #28740)
TITLE: [Bugfix] Fix incorrect use of hidden_states for shared_experts due to do_naive_dispatch_combine (#28740)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/layer.py (+4/-2)
LABELS: bug, ready, deepseek
DEEP_STUDY: deep-study correctness case vllm:e5c78956c0: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=#28406
BODY: This PR fixes an issue created by https://github.com/vllm-project/vllm/pull/28406 when running: ⏎  ⏎ ``` ⏎ export VLLM_ATTENTION_BACKEND=FLASHINFER_MLA ⏎ export VLLM_FLASHINFER_MOE_BACKEND=latency ⏎ export VLLM_USE_FLASHINFER_MOE_FP8=1 ⏎ export VLLM_USE_FLASHINFER_MOE_FP4=1 ⏎ vllm serve --model nvidia/DeepSeek-R1-0528-FP4 --kv-cache-dtype fp8 --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel ⏎ ``` ⏎  ⏎ Thanks @hjjq f …[truncated]

### L3-bf3ffb61e6  (L3, 2025-11-14, sha bf3ffb61e615, PR #28739)
TITLE: [Bugfix] Fix ChunkedLocalAttention CUDA Graph setting (#28739)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/attention/layers/chunked_local_attention.py (+16/-3)
LABELS: bug, ready, llama, nvidia
ISSUES: #28604 [Bug]: Llama4 on B200 flashinfer produces garbage
BODY: ## Purpose ⏎  ⏎ Bugfix for incorrect outputs on llama4 caused by the refactor in #28479: ChunkedLocalAttentionBuilder incorrectly inherits FlashInfer's `get_cudagraph_support` method, causing the `_cudagraph_support` to be ignored and CUDA graphs to always get used. ⏎  ⏎ ## Test Plan ⏎  ⏎ Breaking command: ⏎  ⏎ ``` ⏎ python examples/offline_inference/basic/generate.py --model=nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 --kv-cache-dtype=fp8 --max-model-len=1024 ⏎ ``` ⏎  ⏎  …[truncated]

### L3-f36292dbee  (L3, 2025-11-15, sha f36292dbee27, PR #27126)
TITLE: [compile] Enable sequence parallelism matching w/o custom ops enabled  (#27126)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-pipeline.yaml (+8/-6); tests/compile/test_fusions_e2e.py (+190/-38); tests/compile/test_sequence_parallelism.py (+123/-139); tests/distributed/test_sequence_parallel.py (+14/-1); vllm/compilation/sequence_parallelism.py (+111/-258); vllm/config/vllm.py (+26/-2)
LABELS: ready, torch.compile, ci/build
BODY: ## Purpose ⏎  ⏎ Based on https://github.com/vllm-project/vllm/pull/24604, modified sequence-parallelism pass to do custom op matching w/o needing to enable the custom op ⏎  ⏎ ## Test Plan ⏎  ⏎ `pytest -sv tests/compile/test_sequence_parallelism.py` ⏎  ⏎ ## Performance numbers ⏎  ⏎ I did some benchmarking with the command on H100 w/o flashinfer ⏎  ⏎ ``` ⏎ VLLM_DISABLE_COMPILE_CACHE=1 VLLM_USE_STANDALONE_COMPILE=1 VLLM_LOGGING_LEVEL=DEBUG vllm bench latency --model=nvidia/L …[truncated]

### L3-173b356abf  (L3, 2025-11-15, sha 173b356abff3, PR #28755)
TITLE: [PERF] Remove TRTLLM Gen attn kernel limitation `max_seq_len <=131072` (#28755)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/config/vllm.py (+0/-15); vllm/utils/flashinfer.py (+2/-4)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Right now we have limitation to use TRTLLM-Gen full attn kernel only if longest sequence length in batch less or equal than 131072.  ⏎ From one point of view it is a bit unclear where it came from. TRTLLM-Gen full attn kernel functionally works well for any `max_seq_len`. Performance data also don't show sufficient difference from `max_seq_len=64K` and `max_seq_len=128K` - behavior  for bigger `max_seq_len` is the same as for `max_seq_le …[truncated]

### L3-be263f7645  (L3, 2025-11-15, sha be263f76451a, PR #28751)
TITLE: [BugFix] Fix `AssertionError: DCP not support reorder_batch_threshold > 1 now.`  (#28751)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_model_runner.py (+0/-10)
LABELS: ready, v1
BODY: Fix `AssertionError: DCP not support reorder_batch_threshold > 1 now.` caused by https://github.com/vllm-project/vllm/pull/27363 ⏎  ⏎ Simply removing the assert; this assert has resulted in more false positives than true positives causing unnecessary thrash. The GPU model runner should not be responsible for tracking support of attention backends and ensuring they are advertising correct reorder batch thresholds; this would be better enforced via som …[truncated]

### L3-f849ee739c  (L3, 2025-11-16, sha f849ee739cdb, PR #28161)
TITLE: Adding a benchmark for batch invariance (#28161)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: benchmarks/benchmark_batch_invariance.py (+380/-0)
LABELS: performance, ready
BODY: ## Purpose ⏎  ⏎   Add benchmark to measure VLLM_BATCH_INVARIANT mode performance overhead. ⏎  ⏎ [details omitted] ⏎ [details omitted] ⏎ [details omitted] ⏎  ⏎   ## Test Plan ⏎  ⏎   ```bash ⏎   # Default (Qwen3-1.7B) ⏎   python benchmarks/benchmark_batch_invariance.py ⏎  ⏎   # DeepSeek-V3 with TP=8 ⏎   VLLM_BENCH_MODEL="deepseek-ai/DeepSeek-V3" VLLM_BENCH_TP_SIZE=8 VLLM_BENCH_GPU_MEMORY_UTILIZATION=0.9 python benchmarks/benchmark_batch_invariance.py ⏎  ⏎   Test Result ⏎  ⏎   Benchmark …[truncated]

### L3-03ee48111d  (L3, 2025-11-16, sha 03ee48111de7, PR #27261)
TITLE: Feature: Support Relu2 in FusedMoE fp8 cutlass path (#27261)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer.py (+13/-5); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py (+9/-2); vllm/model_executor/layers/quantization/modelopt.py (+20/-13)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎ This PR enables FusedMoE FP8 cutlass path for models using non-gated relu2 models. This new path gives a performance gain of around 20% on output token throughput over the triton path. ⏎ This PR requires flashinfer `0.5.0` . ⏎  ⏎ ## Tests ⏎ New parameterization in `test_flashinfer.py::test_flashinfer_cutlass_moe_fp8_no_graph` to verify the new non-gated activation. ⏎  ⏎ Performance tests: ⏎ Ran on a single H100. Started the server (once with `VLLM_U …[truncated]

### L3-561253b37f  (L3, 2025-11-16, sha 561253b37faa, PR #28569)
TITLE: [Performance][Fix] update nvfp4 code to support renorm routing (#28569)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/modelopt.py (+11/-7); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+4/-1)
LABELS: performance, frontend, ready, nvidia
ISSUES: #28007 [Bug]: Can't run Flashinfer MoE TRTLLM backend FP4 for Qwen3 235B
BODY: ## Purpose ⏎ Fixes https://github.com/vllm-project/vllm/pull/28007 ⏎ - Add multi routing method to flashinfer fp4 trtllm moe to support models like Qwen3 ⏎ - Add flashinfer trtllm moe into global_sf list which was missed ⏎ ## Test Plan ⏎ ``` ⏎ VLLM_USE_FLASHINFER_MOE_FP4=1 VLLM_FLASHINFER_MOE_BACKEND=latency vllm serve nvidia/Qwen3-235B-A22B-FP4   --max-num-batched-tokens 8192     --max-model-len 16384     --no-enable-prefix-caching     --cuda_graph_sizes 10 …[truncated]

### L3-60e089f0b9  (L3, 2025-11-16, sha 60e089f0b90b, PR #28670)
TITLE: [ROCm][Qwen3-32B] Fix AITER MHA accuracy issue cause by #25763 (#28670)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+2/-2)
LABELS: rocm, ready, v1, qwen
BODY: ## Purpose ⏎ This PR aim to fix Qwen3-32B accuracy issue w/ AITER MHA reported by https://github.com/vllm-project/vllm/issues/28598 by passing correct `min_seqlen_q` value in `aiter.flash_attn_varlen_func` in pure prefill and extend phase.  ⏎  ⏎ After further investigation, I found https://github.com/vllm-project/vllm/pull/25763 introduced this issue during refactor of AITER MHA backend w/ default AITER version `9716b1b8`: https://github.com/vllm-proje …[truncated]

### L3-64e39d667c  (L3, 2025-11-17, sha 64e39d667cb5, PR #28315)
TITLE: [BugFix] Temporary fix for IMA with MTP = 2 and full-cg (#28315)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/compilation.py (+64/-13); vllm/v1/worker/gpu_model_runner.py (+16/-0)
LABELS: bug, ready, v1
BODY: Temporary fix for https://github.com/vllm-project/vllm/issues/28207 ⏎  ⏎ For now just make sure when spec-decode is enabled that the cudagraph shapes are evenly divisible by `1 + num_speculative_tokens`; see https://github.com/vllm-project/vllm/issues/28207 for more details ⏎  ⏎  ### Test 1 ⏎ Tested with  ⏎ ``` ⏎ MODEL=deepseek-ai/DeepSeek-R1 ⏎ VLLM_ATTENTION_BACKEND="FLASHINFER_MLA" \ ⏎ VLLM_USE_V1=1 \ ⏎ vllm serve $MODEL \ ⏎ --tensor-parallel-size 8 \ ⏎ --disable-log- …[truncated]

### L3-a289cc1dde  (L3, 2025-11-17, sha a289cc1dde4a, PR #27421)
TITLE: [Test] Batch Invariant: Rename and organize tests (#27421)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/determinism/conftest.py (+11/-0); tests/v1/determinism/test_batch_invariance.py (+1/-74); tests/v1/determinism/test_online_batch_invariance.py (+161/-0); tests/v1/determinism/test_rms_norm_batch_invariant.py (+1/-6); tests/v1/determinism/utils.py (+74/-0)
LABELS: frontend, ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ Refactor the unit test folder to reuse the utils. ⏎  ⏎ Add online test ⏎  ⏎ ## Test ⏎  ⏎ Online test example ⏎  ⏎ - `export VLLM_ATTENTION_BACKEND="FLASHINFER_MLA"` ⏎ - `export VLLM_BATCH_INVARIANT=1` ⏎ - `vllm serve deepseek-ai/DeepSeek-R1 -tp 8  --enable-expert-parallel --port 9256` ⏎ - `pytest tests/v1/determinism/test_online_batch_invariance.py` ⏎  ⏎ ```bash ⏎ (wentao) wentao@dgxB200-09:~/vllm-source$ pytest tests/v1/generation/test_online_batch_invariance.p …[truncated]

### L3-285eaa4285  (L3, 2025-11-18, sha 285eaa42857b, PR #28846)
TITLE: [Bugfix] Safeguard against missing backend in AttentionBackendEnum (#28846)
SOURCES: path_core, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+2/-1)
LABELS: ready
BODY: ## Purpose ⏎ fix issue https://github.com/vllm-project/vllm-ascend/issues/4197 ,set backend to None when attention not in AttentionBackendEnum ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-9912b8ccb8  (L3, 2025-11-18, sha 9912b8ccb861, PR #28788)
TITLE: [Build] Add OpenAI triton_kernels (#28788)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+5/-0); cmake/external_projects/triton_kernels.cmake (+53/-0); setup.py (+17/-0); .gitignore (+3/-0); vllm/model_executor/layers/quantization/utils/mxfp4_utils.py (+2/-0); vllm/utils/import_utils.py (+39/-1)
LABELS: ready, ci/build, gpt-oss
ISSUES: #27672 [Feature]: Adding `triton_kernels` from Triton repo as a dependency
BODY: ## Purpose ⏎ This PR adds https://github.com/triton-lang/triton/tree/main/python/triton_kernels to vLLM.  ⏎ We can't install this package via pip. Please take a look at https://github.com/vllm-project/vllm/pull/27659 . As a result, this PR, injects the `triton_kernels` package directly into vLLM during build time, similar to the approach we take with `vllm_flash_attn`. Concretely, we just copy the entire `triton_kernels` folder in `<triton-root>/pyth …[truncated]

### L3-4c23690f43  (L3, 2025-11-18, sha 4c23690f43e5, PR #28763)
TITLE: [Attention] FlashAttention ViT support, make default backend (#28763)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fork_build, L3.platform.cuda_selection
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/platforms/cuda.py (+9/-12); vllm/v1/attention/backends/flash_attn.py (+2/-2); tests/kernels/attention/test_flash_attn.py (+2/-2); tests/kernels/attention/test_mha_attn.py (+1/-29)
LABELS: ready, ci/build, v1, nvidia
BODY: ## Purpose ⏎ This PR is paired with https://github.com/vllm-project/flash-attention/pull/109 (merge that first after CI passes, then I'll update the git tag), which enables FA2 to support the head sizes required for vision transformers (40, 72, and 80) (FA3 supports these by default). This PR also updates the selector to make FlashAttention the default backend over xFormers. ⏎  ⏎ ## Test Plan ⏎ `pytest tests/kernels/attention/test_flash_attn.py` (updated …[truncated]

### L3-1ffe934c8a  (L3, 2025-11-19, sha 1ffe934c8ae9, PR #26468)
TITLE: [torch.compile] caching of config fields should be opt-out by default (#26468)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/config/test_config_utils.py (+166/-0); vllm/compilation/backends.py (+83/-22); vllm/compilation/pass_manager.py (+1/-1); vllm/config/cache.py (+23/-8); vllm/config/compilation.py (+22/-18); vllm/config/model.py (+45/-47); vllm/config/parallel.py (+35/-14); vllm/config/utils.py (+118/-1); vllm/envs.py (+87/-82); vllm/logging_utils/__init__.py (+2/-0); (+1 more)
LABELS: ready, torch.compile, ready-run-all-tests
ISSUES: #16501 [RFC]: vLLM x torch.compile caching should be opt-out by default
BODY: Referring: https://github.com/vllm-project/vllm/issues/23107 ⏎  ⏎ Implements opt-out system of hashing, suggested by https://github.com/vllm-project/vllm/issues/16501  ⏎  ⏎ ``` ⏎ def compute_hash(self): ⏎    factors = list(self.__dict__.values()) ⏎    factors.remove("enforce_eager") ⏎    factors.remove("tokenizer_config") ⏎    ... ⏎ ```

### L3-48fc8b1e59  (L3, 2025-11-19, sha 48fc8b1e5957, PR #28990)
TITLE: [BugFix] Fix async-scheduling + FlashAttn MLA (#28990)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashattn, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/mla/common.py (+9/-6); vllm/v1/attention/backends/mla/flashattn_mla.py (+1/-1); vllm/v1/attention/backends/utils.py (+1/-0); vllm/v1/worker/gpu_model_runner.py (+7/-3)
LABELS: ready, v1
BODY: Fix Host<>GPU sync caused by `seq_len_device.max().item()` in FlashAttn MLA ⏎  ⏎ Main: ⏎  ⏎ <img width="1302" height="459" alt="image" src="https://github.com/user-attachments/assets/e3f9bc8c-838d-4a78-9101-9ed72984863e" /> ⏎  ⏎ PR: ⏎  ⏎ <img width="754" height="493" alt="image" src="https://github.com/user-attachments/assets/4365b241-6baa-41a8-814d-a6816905ee67" /> ⏎  ⏎ # Tests: ⏎  ⏎ ``` ⏎ pytest -v tests/distributed/test_context_parallel.py::test_cp_generation -k "DeepS …[truncated]

### L3-613abb50d5  (L3, 2025-11-19, sha 613abb50d571, PR #25990)
TITLE: [MoE] Nvfp4 Masked Gemm: Add flashinfer grouped_gemm_nt_masked (#25990)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+42/-0); .buildkite/test-pipeline.yaml (+1/-0); tests/kernels/moe/test_cutedsl_moe.py (+582/-0); vllm/envs.py (+6/-2); vllm/model_executor/layers/fused_moe/deepep_ll_prepare_finalize.py (+14/-2); vllm/model_executor/layers/fused_moe/flashinfer_cutedsl_moe.py (+346/-0); vllm/model_executor/layers/quantization/modelopt.py (+21/-9); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+33/-10); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+12/-9); vllm/model_executor/layers/quantization/utils/nvfp4_moe_support.py (+5/-1)
LABELS: ready, ci/build, nvidia
BODY: Add [grouped_gemm_nt_masked](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/cute_dsl/blockscaled_gemm.py#L2708) from flashinfer to support nvfp4 MoE.  ⏎  ⏎ depends on [silu_and_mul nvfp4 quanization fusion rework](https://github.com/flashinfer-ai/flashinfer/pull/1927) ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ VLLM_WORKER_MULTIPROC_METHOD="spawn" \ ⏎ VLLM_ALL2ALL_BACKEND="masked_gemm" \ ⏎ VLLM_USE_STANDALONE_COMPILE=0 \ ⏎ VLLM_USE_FLASHINFER_MOE_FP4=1  …[truncated]

### L3-61728cd1df  (L3, 2025-11-19, sha 61728cd1dfb0, PR #28966)
TITLE: Re-enable FlashInfer for Llama4 on Blackwell in e2e fusion tests (#28966)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-pipeline.yaml (+2/-0); tests/compile/distributed/test_fusions_e2e.py (+4/-8)
LABELS: ready, torch.compile, ci/build, llama
BODY: ## Purpose ⏎  ⏎ Issue #28604 has been fixed (resolved by #28739) - re-enable FlashInfer as the attention backend for Llama4 on Blackwell platforms in e2e fusion tests. ⏎  ⏎ ## Test Plan ⏎  ⏎ Existing e2e fusion tests in `tests/compile/distributed/test_fusions_e2e.py` will validate the change: ⏎ - `test_attn_quant` - Tests attention+quant fusion with FlashInfer on Blackwell ⏎ - `test_tp2_attn_quant_allreduce_rmsnorm` - Tests multi-GPU fusion patterns ⏎ - `test_tp2_ …[truncated]

### L3-da2f6800e0  (L3, 2025-11-19, sha da2f6800e0d6, PR #28449)
TITLE: [Feat][Perf] Enable deepep-low-latency with round-robin expert placement. (#28449)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/all2all_utils.py (+11/-0); vllm/model_executor/layers/fused_moe/deepep_ll_prepare_finalize.py (+28/-2); vllm/model_executor/layers/fused_moe/fused_moe_method_base.py (+7/-2); vllm/model_executor/layers/fused_moe/layer.py (+135/-22); vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py (+5/-2); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+10/-4); vllm/model_executor/layers/quantization/fp8.py (+5/-2); vllm/model_executor/layers/quantization/modelopt.py (+7/-3)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Enable deepep low latency all2all backend with round-robin expert plamement strategy, which produces significant performance improvement. ⏎  ⏎ ## Performance ⏎ Test Platform: CUDA 12.8, drivier 550.144.03 ⏎ Model: DeepSeek-R1-671B ⏎ GPU: H20 * 8 * 2 nodes ⏎ Vllm config: dp=16, tp=1, enable_expert_parallel=1, all2all_backend=deepep_low_latency, use_deep_gemm=1 ⏎  ⏎ The current functionality has been fully implemented with correct accuracy and significa …[truncated]

### L3-2fd893b4ce  (L3, 2025-11-19, sha 2fd893b4cec0, PR #28718)
TITLE: [Feature] Prefill Context Parallel (PCP) basic support (#28718)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.mla.common_v1, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+17/-0); vllm/attention/ops/common.py (+37/-3); vllm/v1/attention/backends/flash_attn.py (+3/-3); vllm/v1/attention/backends/mla/common.py (+3/-3); vllm/v1/attention/backends/utils.py (+9/-9); tests/distributed/test_context_parallel.py (+6/-6); tests/kernels/moe/modular_kernel_tools/common.py (+6/-1); tests/v1/worker/test_gpu_model_runner.py (+2/-2); vllm/config/parallel.py (+31/-9); vllm/config/vllm.py (+26/-6); (+17 more)
LABELS: ready, v1, gpt-oss
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: # Purpose ⏎ This PR, splited from full PR #26864, adds the basic supports for the Prefill Context Parallelism (PCP) feature, which corresponds to DCP. For specific implementation details, please refer to the RFC #25749. ⏎  ⏎ TL;DR: PCP enhances long-sequence inference capabilities by partitioning the sequence dimension during the prefill stage. ⏎  ⏎ The current implementation primarily includes the following changes: ⏎  ⏎ - Modified files such as `block_tables …[truncated]

### L3-68d7231991  (L3, 2025-11-19, sha 68d7231991cc, PR #28905)
TITLE: [CI/Build] Fix test_prefix_prefill for AMD (#28905)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_prefix_prefill.py (+6/-6)
LABELS: rocm, ready
BODY: Resolves issue #28490. ⏎  ⏎ ## Purpose ⏎ This PR moves the typecast after torch.cumsum, to prevent unwanted promotion to int64. ⏎  ⏎ ## Test Plan ⏎ `pytest -s -v 'tests/kernels/attention/test_prefix_prefill.py'` ⏎  ⏎ ## Test Result ⏎ > 160 passed, 224 skipped, 3 warnings in 178.15s ⏎  ⏎ --- ⏎ [details omitted]

### L3-d44e9df7d4  (L3, 2025-11-19, sha d44e9df7d49a, PR #26487)
TITLE: [Model][Mamba] Add selector for mamba attention backend and make it pluggable for other device (#26487)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.dispatch.selector, L3.dispatch.registry
FILES: vllm/attention/backends/registry.py (+100/-14); vllm/attention/selector.py (+32/-1); docs/contributing/model/basic.md (+1/-0); vllm/attention/__init__.py (+2/-1); vllm/model_executor/layers/kda.py (+1/-7); vllm/model_executor/layers/mamba/abstract.py (+5/-5); vllm/model_executor/layers/mamba/linear_attn.py (+0/-14); vllm/model_executor/layers/mamba/mamba_mixer.py (+1/-9); vllm/model_executor/layers/mamba/mamba_mixer2.py (+0/-9); vllm/model_executor/layers/mamba/short_conv.py (+0/-9); (+2 more)
LABELS: documentation, new-model, ready, qwen
BODY: ## Purpose ⏎  ⏎ **Motivation:** ⏎  ⏎ Like attention and MLA attention modules, we want to use some device-specific kernels for mamba layers and customize the proccessing of mamba attn backend, i.e., this is significant for running some mamba-like models (e.g., Qwen3-Next) on Ascend platform. ⏎  ⏎ **Main changes:** ⏎  ⏎ - Add `get_mamba_attn_backend()` to attention selector, and force all mamba layers to get their attention backend by calling this method. ⏎ - Add ` …[truncated]

### L3-88f5b19f0b  (L3, 2025-11-19, sha 88f5b19f0bc6, PR #28968)
TITLE: [DeepSeek] Fix DeepSeek V3.2 Rope Embedding (#28968)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mla.py (+5/-1); vllm/model_executor/models/deepseek_v2.py (+12/-2)
LABELS: bug, ready, deepseek
BODY: ## Purpose ⏎ Deepseek recently find error in their official [implementation](https://x.com/deepseek_ai/status/1990616182161068094?s=20) that ROPE in indexer shouldn't be interleaved. ⏎  ⏎ ## Test Plan ⏎ gsm8k 20-shots ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|    20|exact_match|↑  |0.9568|±  |0.005 …[truncated]

### L3-ac10fd3c69  (L3, 2025-11-19, sha ac10fd3c6900, PR #28701)
TITLE: Upstreaming aiter triton attention backend as a new backend (#28701)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.aiter_triton, L3.dispatch.registry, L3.platform.rocm_selection
FILES: vllm/attention/backends/registry.py (+3/-0); vllm/platforms/rocm.py (+3/-1); vllm/v1/attention/backends/mla/aiter_triton_mla.py (+74/-0)
LABELS: rocm, ready, v1
BODY: Adding new Triton MLA to handle prefills in DS ⏎  ⏎ ## Server command: ⏎ ``` ⏎ VLLM_DISABLE_COMPILE_CACHE=1 \ ⏎ AMDGCN_USE_BUFFER_OPS=1 \ ⏎ VLLM_ROCM_USE_AITER=1 \ ⏎ VLLM_ATTENTION_BACKEND=ROCM_AITER_TRITON_MLA \ ⏎ VLLM_ROCM_USE_AITER_MLA=1 \ ⏎ VLLM_ROCM_USE_TRITON_ROPE=1 \ ⏎ vllm serve /data/models/deepseek-ai/DeepSeek-R1-0528 \ ⏎     --host localhost \ ⏎     --port 8000 \ ⏎     --swap-space 64 \ ⏎     --disable-log-requests \ ⏎     --dtype auto \ ⏎     --max-model-len 8192 \ ⏎  …[truncated]

### L3-1607e664f0  (L3, 2025-11-19, sha 1607e664f0de, PR #28967)
TITLE: [Bug] Fix Batch Invariant MLA test (#28967)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/determinism/test_batch_invariance.py (+32/-9); vllm/model_executor/layers/batch_invariant.py (+1/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Fix Batch Invariant MLA test ⏎  ⏎ TritonMLA will be removed in this PR: https://github.com/vllm-project/vllm/pull/28832 ⏎  ⏎ After Flashinfer's update, we found that batch invariant support of FlashinferMLA is broken, having issue https://github.com/flashinfer-ai/flashinfer/issues/2107 here, we don't want to use a for loop which will be very slow, so just ban the FlashinferMLA for a while. Update: even if using a for loop, still may cause som …[truncated]

### L3-cb0a7b4bea  (L3, 2025-11-19, sha cb0a7b4bea26, PR #29018)
TITLE: [Bugfix] Move flashinfer kernel check into ```__init__``` function of ```FusedMoE``` (#29018)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/layer.py (+4/-1)
LABELS: bug, ready, qwen, nvidia
ISSUES: #29015 [Bug]: Qwen3-VL-235B-A22B-Instruct-NVFP4 fused moe failed to be traced by torch dynamo
DEEP_STUDY: deep-study correctness case vllm:cb0a7b4bea: class=integration_backend_cudagraph; symptom=compile_or_build_failure; introducing=unknown
BODY: ## Purpose ⏎ Fix https://github.com/vllm-project/vllm/issues/29015. ⏎  ⏎ Move flashinfer kernel check which includes a check for flashinfer lib availability which can't be traced by ```torch.dynamo``` into ```__init__``` function of ```FusedMoE``` class ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ VLLM_USE_FLASHINFER_MOE_FP4=1 VLLM_FLASHINFER_MOE_BACKEND=throughput vllm serve RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4 --host 0.0.0.0 --port 53693 -dp 2 --mm-encoder-tp-mode data  …[truncated]

### L3-05c2dee7e9  (L3, 2025-11-20, sha 05c2dee7e9f4, PR #29039)
TITLE: [DeepSeek + LMCache Multiprocess] handle MLA for deepseek model + LMCache Multiprocess connector (#29039)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py (+39/-8)
LABELS: ready, deepseek, kv-connector
BODY: ## Purpose ⏎  ⏎ This PR handles MLA for deepseek models ⏎  ⏎ ## Test Plan ⏎  ⏎ Run  ⏎ ``` ⏎ MODEL="deepseek-ai/DeepSeek-V2-Lite" ⏎ PORT=8000 ⏎ NUM_CONCURRENT=50 ⏎  ⏎ lm_eval --model local-completions --tasks gsm8k \ ⏎     --model_args model=${MODEL},base_url=http://127.0.0.1:${PORT}/v1/completions,num_concurrent=${NUM_CONCURRENT},max_retries=3,tokenized_requests=False \ ⏎     --limit 300 \ ⏎     --seed 0 \ ⏎     --gen_kwargs '{"temperature": 0.0}' ⏎ ``` ⏎  ⏎ ## Test Result ⏎  ⏎ ### Basel …[truncated]

### L3-3fb0d90999  (L3, 2025-11-20, sha 3fb0d9099988, PR #27715)
TITLE: [AMD] Use Decoupled Kernel Block Size to Support AITER MLA block_size=1 (#27715)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+6/-8); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+7/-38)
LABELS: rocm, ready, v1, nvidia
BODY: PR #24486 implemented this mechanism but did not fully utilize it.  ⏎ The block_size restriction during startup (when using operators such as FlashMla and AiterMLA) has now been lifted. ⏎  ⏎ Compared to #27224, this PR avoids remapping before each operator execution, instead making use of the functionality provided by the framework (by #24486), and also provides support for scenarios other than AMD aiter.

### L3-fb8851f254  (L3, 2025-11-20, sha fb8851f25485, PR #28760)
TITLE: [Bugfix][cache_kernels]: Fix OOB in cache_kernels.cu (#28760)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+12/-7); tests/kernels/test_cache_kernels.py (+65/-0)
LABELS: ready, gpt-oss
DEEP_STUDY: deep-study correctness case vllm:fb8851f254: class=memory_safety_oob; symptom=illegal_memory_access; introducing=unknown
BODY: ## Purpose ⏎ This PR fixes a potential out-of-bounds (OOB) memory access in the gather_and_maybe_dequant_cache CUDA kernel, as originally reported in Issue #27909. ⏎  ⏎ The bug was identified by static analysis. The root cause was that the offset (calculated from seq_starts[bid] / block_size) was not validated against the block_table_stride (the bound of the block table) before being used to access the batch_block_table pointer. ⏎  ⏎ 1.  this fix calculate …[truncated]

### L3-06c20c9904  (L3, 2025-11-20, sha 06c20c990464, PR #26670)
TITLE: [ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA (#26670)
SOURCES: path_core, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.mla.flashmla_sparse, L3.mla.rocm_aiter_sparse, L3.platform.rocm_selection
FILES: csrc/cache_kernels.cu (+4/-0); vllm/attention/ops/rocm_aiter_mla_sparse.py (+210/-0); vllm/v1/attention/backends/mla/flashmla_sparse.py (+1/-1); vllm/v1/attention/backends/mla/indexer.py (+9/-6); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+325/-0); vllm/model_executor/models/deepseek_v2.py (+18/-4); vllm/platforms/rocm.py (+12/-1); vllm/utils/deep_gemm.py (+3/-2); vllm/v1/worker/utils.py (+1/-1)
LABELS: documentation, performance, rocm, speculative-decoding, ready, ci/build, v1, qwen, deepseek
BODY: ## Purpose ⏎ The PR add Deepseek v3.2 support on ROCm platforms. The main change in this PR include:  ⏎ - Replace all hardcode float8_e4m3fn to platform supported fp8 dtype, and add FP8_E4M3FNUZ enum to cpp kernels. ⏎ - Add torch impl to deepgemm  ⏎ - Add rocm_aiter_mla_sparse backend and dispatch it in rocm platform ⏎  ⏎ ## Test Plan ⏎ Verify its accuracy on gsm8k and wikitext ⏎  ⏎ ## Test Result ⏎ ``` ⏎ # wikitext ⏎ | Tasks  |Version|Filter|n-shot|    Metric     |   | …[truncated]

### L3-371b1d4c61  (L3, 2025-11-20, sha 371b1d4c6133, PR #28037)
TITLE: [RL] Add Pause and Resume Generation for Asynchronous RL Training (#28037)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/engine/protocol.py (+27/-0); vllm/entrypoints/openai/api_server.py (+78/-0); vllm/v1/engine/async_llm.py (+64/-0); vllm/v1/engine/output_processor.py (+13/-0)
LABELS: documentation, frontend, speculative-decoding, ready, v1, rl
BODY: ## Purpose ⏎  ⏎ Integrate Pause/Resume generation to support asynchronous RL workflows.  ⏎  ⏎ Reference: https://github.com/sgl-project/sglang/pull/7419 ⏎  ⏎ <div align="center"> ⏎ <img width="90%"  alt="Async RL flow" src="https://github.com/user-attachments/assets/f37e190b-984a-402a-97a1-86f8c48a4263" /> ⏎ <p> Pause and resume generation for parameter sync. in Async RL training  </p> ⏎ </div> ⏎  ⏎ As illustrated in the above  figure from [VeRL Fully Async Policy](ht …[truncated]

### L3-a2e9ebe9e2  (L3, 2025-11-20, sha a2e9ebe9e242, PR #29082)
TITLE: [BugFix] Fix flash_attn import in `siglip2navit.py` (#29082)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/siglip2navit.py (+1/-1)
LABELS: ready
BODY: ## Purpose ⏎ Fixes the following error when running models using Siglip2NavitModel (e.g., Ovis2.5): ⏎  ⏎ ```bash ⏎ (EngineCore_DP0 pid=265387)   File "/mnt/disk4/fanlilin/upstream/vllm/vllm/model_executor/models/siglip2navit.py", line 298, in forward ⏎ (EngineCore_DP0 pid=265387)     queries, keys = apply_rotary_pos_emb( ⏎ (EngineCore_DP0 pid=265387)                     ^^^^^^^^^^^^^^^^^^^^^ ⏎ (EngineCore_DP0 pid=265387)   File "/mnt/disk4/fanlilin/upstream/vl …[truncated]
