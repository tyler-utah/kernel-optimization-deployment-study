### L3-609b533cb6  (L3, 2025-08-06, sha 609b533cb6f2, PR #22314)
TITLE: [Bugfix] Add proper comparison for package versions (#22314)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.decode_attention
FILES: vllm/attention/ops/triton_decode_attention.py (+3/-1); benchmarks/kernels/benchmark_bitblas.py (+3/-1); docs/design/arch_overview.md (+2/-1); vllm/model_executor/layers/quantization/bitblas.py (+3/-1); vllm/model_executor/layers/quantization/bitsandbytes.py (+5/-2); vllm/model_executor/layers/quantization/deepspeedfp.py (+2/-1); vllm/model_executor/layers/quantization/gptq_bitblas.py (+3/-1); vllm/model_executor/layers/quantization/ipex_quant.py (+5/-2); vllm/model_executor/layers/quantization/kernels/mixed_precision/bitblas.py (+3/-1); vllm/model_executor/layers/quantization/utils/bitblas_utils.py (+3/-1); (+3 more)
LABELS: documentation, performance, ready, v1
ISSUES: #22297 [Bug]: Incorrect version judgment method for flashinfer
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ To fix issue #22297 for flashinfer and more generally for all cases. Replaced occurrences of direct string comparison with version.parse() from the packaging libary.  ⏎  ⏎ Under string comparison, something like '2.2.10' < '2.2.3' returns True. ⏎  ⏎ Using version.parse(), we get correct comparisons. ⏎  ⏎ Resolves #22297. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-a3b9c17b56  (L3, 2025-08-07, sha a3b9c17b56d0, PR #21331)
TITLE: Support Tensorrt-LLM MoE fp4 for low-latency (#21331)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+4/-0); vllm/envs.py (+15/-0); vllm/model_executor/layers/fused_moe/config.py (+2/-1); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+4/-5); vllm/model_executor/layers/quantization/modelopt.py (+252/-32); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+9/-3); vllm/model_executor/layers/quantization/utils/nvfp4_moe_support.py (+2/-2)
LABELS: ready, llama, deepseek
BODY: co-authored by: @xinli-git ⏎ patched partially https://github.com/vllm-project/vllm/pull/22073 to fix DSR1 weight loading. ⏎ ``` ⏎ export VLLM_USE_FLASHINFER_MOE_FP4=1 ⏎ export VLLM_FLASHINFER_MOE_BACKEND="latency" ⏎ lm_eval --model vllm --model_args pretrained=nvidia/Llama-4-Scout-17B-16E-Instruct-FP4,tensor_parallel_size=4,max_model_len=2048,enforce_eager=True,kv_cache_dtype=auto --gen_kwargs temperature=0.0 --limit 500 --trust_remote_code --tasks gsm8k  …[truncated]

### L3-af473f0a85  (L3, 2025-08-07, sha af473f0a8573, PR #22426)
TITLE: [bugfix] Fix Llama3/4 issues caused by FlashInfer 0.2.10 (#22426)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+2/-1); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+16/-8)
LABELS: ready, v1, llama
BODY: Changes: ⏎  ⏎ 1. Set FlashInfer trtllm MoE tile size back to 8 since kernels with larger tile sizes are missing in FlashInfer 0.2.10. ⏎ 2. Disable FlashInfer trtllm prefill attn backend when kv-cache dtype if FP8 because here is no BF16-Q FP8-KV prefill kernels in FlashInfer 0.2.10. ⏎  ⏎ # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Fix Llama3/4 issues caused by FlashInfer 0.2.10 ⏎  ⏎ ## Test Plan ⏎  ⏎ Llama4 Scout FP8 on B200 ⏎  ⏎ ``` ⏎ exp …[truncated]

### L3-cd9b9de1fb  (L3, 2025-08-08, sha cd9b9de1fb00, PR #21691)
TITLE: [BugFix] Fix IMA FlashMLA full cuda-graph and DP + Update FlashMLA (#21691)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.mla.flashmla_build
FILES: cmake/external_projects/flashmla.cmake (+4/-4); vllm/attention/ops/flashmla.py (+0/-1); vllm/v1/attention/backends/mla/flashmla.py (+38/-22)
LABELS: bug, ready, ci/build, v1, deepseek
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Merge: https://github.com/vllm-project/FlashMLA/pull/3 first ⏎  ⏎ Fix an IMA that occurs when using FlashMLA with full-cudagraphs and wide-ep ⏎  ⏎ Also updates FlashMLA (i.e. https://github.com/vllm-project/vllm/pull/17027) since the FlashMLA changes were made on top of that. https://github.com/vllm-project/vllm/pull/17027 was back-burnered since it shows a slight slowdown in the …[truncated]

### L3-bd875d2eb7  (L3, 2025-08-08, sha bd875d2eb71b, PR #22546)
TITLE: [Bugfix] Update FA commit hash (#22546)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ci/build
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ vLLM side of: https://github.com/vllm-project/flash-attention/pull/79 ⏎  ⏎ Tests that compare against V0 (e.g., hybrid model tests) are currently failing with FA3 because we are not passing the s_aux parameter through the `flash_attn_with_kvcache` interface. This has been fixed in vLLM flash attention and this PR just updates the git commit. ⏎  ⏎ This is the error we are seeing wi …[truncated]

### L3-e3edc0a7a8  (L3, 2025-08-08, sha e3edc0a7a8f0, PR #22524)
TITLE: Extract `CompilationConfig` from `config.py` (#22524)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: tests/engine/test_arg_utils.py (+0/-33); vllm/config/__init__.py (+7/-442); vllm/config/compilation.py (+428/-0); vllm/config/utils.py (+29/-0); vllm/engine/arg_utils.py (+3/-5)
LABELS: ready
BODY: Part of https://github.com/vllm-project/vllm/issues/18953 ⏎  ⏎ Notable changes: ⏎ - removal of `CompilationConfig.from_cli`. It no longer does anything since the `-O` special case is handled in the `FlexibleArgumentParser` ⏎ - handling of `from_cli` in `get_kwargs` has been removed now that nothing uses it. It was only added as a convenience and it's better to not use custom `from_cli` methods for the CLI ⏎ - Majority of `vllm/config.py` has been move to ` …[truncated]

### L3-6ade99eafa  (L3, 2025-08-08, sha 6ade99eafa37, PR #22151)
TITLE: [V1] [Hybrid] Support Minimax-Text-01 in V1  (#22151)
SOURCES: corpus:production-kernel-provenance
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/lightning_attn.py (+1/-1); vllm/model_executor/layers/mamba/mamba_utils.py (+11/-0); vllm/model_executor/models/minimax_text_01.py (+152/-40); vllm/v1/attention/backends/linear_attn.py (+67/-0); vllm/v1/attention/backends/mamba_selectors.py (+3/-1)
LABELS: ready, v1
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ This PR enables Minimax-Text-01 in vLLM V1. Currently only eager mode is supported.  ⏎  ⏎ I've tried to keep the changes minimal. I hope this PR could serve as a reference/template to how to enable other non-mamba2 based hybrid models in V1.  ⏎  ⏎ ## Test Plan ⏎  ⏎ Deploy using V0 from main: ⏎ ``` ⏎ vllm serve MiniMaxAI/MiniMax-Text-01 \ ⏎ 	--tensor-parallel-size 8 \ ⏎ 	--trust-remote-code \ ⏎  …[truncated]

### L3-b7c0942b65  (L3, 2025-08-08, sha b7c0942b6538, PR #22097)
TITLE: [ROCm][Misc] Rename the context_len to seq_len in ROCm custom paged attention kernel (#22097)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.custom_paged
FILES: csrc/rocm/attention.cu (+87/-92); csrc/rocm/ops.h (+2/-2); csrc/rocm/torch_bindings.cpp (+2/-2)
LABELS: rocm, ready
BODY: I suppose we are using `seq_len` to represent the size of whole input sequence, including query. But in the ROCm paged attention kernel, `context_len` is being used. This PR removes the inconsistence.

### L3-42172ad18f  (L3, 2025-08-09, sha 42172ad18fbc, PR #22375)
TITLE: [FEAT] [Performance] Add triton mrope to replace the torch code path (#22375)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_mrope.py (+328/-0); tests/kernels/test_mrope.py (+207/-0); vllm/model_executor/layers/rotary_embedding/mrope.py (+231/-0)
LABELS: performance, ready
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ This is to optimize the mrope forward pass using a triton kernel adapted from https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/qwen2vl_mrope.py to supports flatten input tensors from vLLM and and supports cos and sin cache with shape (3, num_tokens, head_dim // 2) ⏎  ⏎ Related to https://github.com/vllm-project/vllm/issues/22293 ⏎  ⏎ ## Test Plan ⏎  ⏎ - mrope pa …[truncated]

### L3-2a84fb422f  (L3, 2025-08-09, sha 2a84fb422fc6, PR #22394)
TITLE: [TPU] kv cache update kernel doesn't need to be padded slices to multiple of num_slices_per_block (#22394)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/pallas_kv_cache_update.py (+10/-6); tests/v1/tpu/test_kv_cache_update_kernel.py (+0/-5); vllm/v1/worker/tpu_model_runner.py (+9/-10)
LABELS: tpu, ready, v1
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ - [ x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)". ⏎ - [ x] The test plan, such as providing test command. ⏎ - [x ] The test results, such as pasting the results comparison before and after, or e2e results ⏎  ⏎ ## Purpose ⏎  ⏎ kv cache update kernel doesn't need to be padded slices to multiple of num_slices_per_block ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest -s -v  …[truncated]

### L3-c49848396d  (L3, 2025-08-09, sha c49848396d34, PR #21927)
TITLE: Refactor sliding window configuration to Transformers best practice (#21927)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs/contributing/model/basic.md (+1/-1); tests/test_config.py (+0/-22); vllm/config/__init__.py (+31/-80); vllm/engine/arg_utils.py (+9/-1); vllm/model_executor/models/commandr.py (+7/-15); vllm/model_executor/models/exaone4.py (+4/-17); vllm/model_executor/models/gemma2.py (+3/-6); vllm/model_executor/models/gemma3.py (+4/-10); vllm/model_executor/models/gemma3_mm.py (+2/-4); vllm/model_executor/models/gemma3n.py (+6/-7); (+6 more)
LABELS: documentation, ready, llama, qwen
ISSUES: #15752 [Bug]: Gemma-3-12B-it model getting stuck in repetitive output loops | #17689 [Bug]: gemma3 shows degraded accuracy in vLLM v0.8.4 | #20341 [Bug]: No output / Repeated outputs when using Gemma 3  on vLLM | #22270 AttributeError: 'Gemma3TextConfig' object has no attribute 'interleaved_sliding_window' | #22475 [Bug]: Major issues with transformers version causing rubbish generations with Gemma3 family using vllm
BODY: - Uses `layer_types` instead of `sliding_window_pattern` ⏎ - Removes custom `interleaved_sliding_window` as it should not be necessary ⏎  ⏎ Unlike the last time I tried to remove `sliding_window: list[int]` (https://github.com/vllm-project/vllm/pull/18494#discussion_r2106505755), this PR correctly maps Mistral format sliding window configuration to the new HF style sliding window configuration. ⏎  ⏎ --- ⏎  ⏎ This PR should also fix the issues that everyone has …[truncated]

### L3-00976db0c3  (L3, 2025-08-10, sha 00976db0c311, PR #22588)
TITLE: [Docs] Fix warnings in docs build (#22588)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layers/__init__.py (+0/-0); docs/api/summary.md (+0/-2); docs/configuration/tpu.md (+1/-1); docs/contributing/model/multimodal.md (+5/-3); docs/models/generative_models.md (+2/-2); docs/models/pooling_models.md (+1/-1); docs/models/supported_models.md (+1/-1); vllm/inputs/__init__.py (+6/-4); vllm/model_executor/warmup/__init__.py (+0/-0); vllm/sampling_params.py (+64/-76)
LABELS: documentation
BODY: It would be nice to enable `--strict` mode in the docs build so that it can catch any broken links before we merge docs PRs. ⏎  ⏎ This is currently not possible because the docs build contains too many warnings. ⏎  ⏎ This PR begins the process of fixing them. ⏎  ⏎ ---  ⏎  ⏎ Warnings fixed: ⏎  ⏎ - WARNING -  api-autonav: Skipping implicit namespace package (without an __init__.py file) ⏎ - WARNING -  Doc file 'configuration/tpu.md' contains a link '../../benchmarks/aut …[truncated]

### L3-d1af8b7be9  (L3, 2025-08-10, sha d1af8b7be9c5, PR #22106)
TITLE: enable Docker-aware precompiled wheel setup (#22106)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen
FILES: docker/Dockerfile (+5/-10); setup.py (+102/-83); vllm/envs.py (+9/-2)
LABELS: ready, ci/build
BODY: - Enables `VLLM_USE_PRECOMPILED` in Docker builds by exposing it as a build arg ⏎ - Adds `VLLM_DOCKER_BUILD_CONTEXT` env + Dockerfile arg for safe wheel fallback ⏎ - Replaces old `repackage_wheel` class with `precompiled_build_ext` to avoid triggering `build_ext` during `pip install -e .` with precompiled binaries ⏎ - Refactors wheel unpacking to run before `setup()` and patch `package_data` ⏎ - Fixes truthiness of `VLLM_USE_PRECOMPILED` and new `VLLM_DO …[truncated]

### L3-dc5e4a653c  (L3, 2025-08-11, sha dc5e4a653c85, PR #22613)
TITLE: Upgrade FlashInfer to v0.2.11 (#22613)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); setup.py (+1/-1)
LABELS: ready, ci/build
BODY: This includes the fix needed for FlashInfer AOT compilation failures. See: https://github.com/flashinfer-ai/flashinfer/pull/1403 and https://github.com/flashinfer-ai/flashinfer/pull/1410 ⏎  ⏎ # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ Llama 4 Scout FP8/FP4 accuracy ⏎  ⏎ ## Test Result ⏎  ⏎ Llama 4 Scout FP8 accuracy ⏎  ⏎ ``` ⏎ local-completions (base_url=http://0.0.0.0:8080/v1/completions,model=nvidia/Llama-4-Scout-17B …[truncated]

### L3-6d729c43fb  (L3, 2025-08-12, sha 6d729c43fbaf, PR #22637)
TITLE: [Bugfix] Fix ModernBert load & Enable sliding window attention for bidirectional attention. (#22637)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+2/-0); tests/models/language/pooling/test_gte.py (+18/-3); vllm/model_executor/models/modernbert.py (+15/-16); vllm/v1/worker/gpu_model_runner.py (+66/-40)
LABELS: ready, v1
ISSUES: #22620 [Bug]: Cant classify with ModernBERT
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Fix #22620 ⏎  ⏎ ## Test Plan ⏎  ⏎ pytest -s -vvv tests/models/language/pooling/test_gte.py::test_rerank_models_mteb ⏎  ⏎ ## Test Result ⏎  ⏎ pass ⏎  ⏎ ## (Optional) Documentation Update ⏎  ⏎ ## Known Issues ⏎  ⏎ Does not support fp32 ⏎  ⏎ NotImplementedError: FlexAttention does not support sliding window yet.

### L3-007dd90859  (L3, 2025-08-12, sha 007dd90859cc, PR #22714)
TITLE: [gpt-oss] Enable gpt-oss on ampere (#22714)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.selector, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/attention/layer.py (+3/-1); vllm/attention/selector.py (+4/-1); tests/plugins/vllm_add_dummy_platform/vllm_add_dummy_platform/dummy_platform.py (+3/-2); vllm/model_executor/layers/quantization/mxfp4.py (+1/-1); vllm/platforms/cpu.py (+2/-2); vllm/platforms/cuda.py (+5/-2); vllm/platforms/interface.py (+2/-2); vllm/platforms/rocm.py (+2/-2); vllm/platforms/tpu.py (+2/-2); vllm/platforms/xpu.py (+2/-2)
LABELS: rocm, tpu, ready, gpt-oss
BODY: - Hardcode to use triton attention on hopper when attention sink is not used.  ⏎ - change mxfp4 to support ampere ⏎  ⏎ gpt-oss-20b output ⏎ ``` ⏎ Generated Outputs: ⏎ ------------------------------------------------------------ ⏎ Prompt:    'How are you?' ⏎ Output:    "'\n    },\n    {\n        'id': 2,\n        'sender': 'Bob',\n        'timestamp': '2023-08-01 10:05:00',\n        'content': 'I am fine, thanks!'\n    },\n    {\n        'id': 3,\n        'sender …[truncated]

### L3-6bd8ebf026  (L3, 2025-08-12, sha 6bd8ebf02660, PR #22683)
TITLE: [Kernel][AMD] Avoid D2H copy and cumsum kernel (#22683)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+20/-12)
LABELS: performance, rocm, ready, v1
BODY: Current AITER attention for prefill incurs one D2H copy (which causes synchonrization between CPU and GPU) and one cumsum kernel. ⏎ <img width="1299" height="477" alt="Screenshot 2025-08-11 at 3 48 28 PM" src="https://github.com/user-attachments/assets/ec199155-fef9-4f7d-b20e-778dae27ee9b" /> ⏎  ⏎ The D2H copy is used to get the actual KV tokens on the host side for intermediate buffer allocation; this value should not change during the inference, so w …[truncated]

### L3-53c730286c  (L3, 2025-08-12, sha 53c730286c5a, PR #22641)
TITLE: [Misc] parametrize 'dtype' in test_flash_mla (#22641)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_flashmla.py (+3/-4)
LABELS: ready
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ complete the TODO list in `tests/kernels/attention/test_flashmla.py` to make `test_flash_mla` test more flexible by parametrizing `dtype`  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-71683ca6f6  (L3, 2025-08-12, sha 71683ca6f676, PR #22138)
TITLE: [V0 Deprecation] Remove multi-step scheduling (#22138)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: .buildkite/nightly-benchmarks/tests/genai-perf-tests.json (+0/-1); .buildkite/nightly-benchmarks/tests/nightly-tests.json (+0/-6); .buildkite/test-pipeline.yaml (+0/-22); .github/CODEOWNERS (+0/-1); tests/async_engine/test_async_llm_engine.py (+0/-409); tests/config/test_config.yaml (+0/-1); tests/config/test_config_with_model.yaml (+0/-1); tests/core/test_chunked_prefill_scheduler.py (+3/-7); tests/core/test_num_computed_tokens_update.py (+4/-20); tests/engine/test_multi_step_output_processor.py (+0/-274); (+27 more)
LABELS: performance, rocm, tpu, ready, ci/build, v1
BODY: ## Summary ⏎ - drop multi-step scheduling logic and workers ⏎ - remove multi-step CLI flags and config options ⏎ - delete multi-step tests ⏎  ⏎ ## Testing ⏎ - `pre-commit run --files tests/config/test_config.yaml tests/config/test_config_with_model.yaml tests/core/test_chunked_prefill_scheduler.py tests/core/test_num_computed_tokens_update.py tests/entrypoints/openai/correctness/test_lmeval.py tests/metrics/test_metrics.py tests/models/language/generation/te …[truncated]

### L3-b1361c7273  (L3, 2025-08-12, sha b1361c7273f6, PR #22738)
TITLE: [Bugfix] Fix default enable for CUTLASS MLA on SM100 (#22738)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+6/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ We found that triton mla ending up being chosen on B200, even though cutlass mla was being "forced" ⏎  ⏎ ``` ⏎ vllm serve deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --load-format dummy ⏎ ... ⏎ (APIServer pid=183965) INFO 08-12 10:27:33 [cuda.py:184] Forcing kv cache block size to 128 for CUTLASS_MLA backend. ⏎ ... ⏎ (EngineCore_0 pid=184744) INFO 08-12 10:27:40 [cuda.py:240] Using Triton MLA backend on V1 engine. ⏎ ``` ⏎  ⏎ This is because `vllm/attenti …[truncated]

### L3-c6b928798e  (L3, 2025-08-12, sha c6b928798e96, PR #22678)
TITLE: Force TRTLLM attention for gpt-oss on SM100 (#22678)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.abstract_interface
FILES: vllm/utils/flashinfer.py (+8/-0); vllm/v1/attention/backends/flashinfer.py (+7/-4); vllm/v1/attention/backends/utils.py (+4/-1); vllm/model_executor/models/gpt_oss.py (+1/-4)
LABELS: ready, v1, gpt-oss
BODY: If attention sinks are being used, then use_trtllm_attention should return True since we only support sinks with that backend. This PR also moves the float32 sinks cast that the TRTLLM backend needs out of the model def and into the backend ⏎  ⏎ This removes the need to specify `VLLM_USE_TRTLLM_ATTENTION=1` ⏎  ⏎ Validated manually with gpt-oss eval on B200

### L3-4082338a25  (L3, 2025-08-12, sha 4082338a2585, PR #22765)
TITLE: Remove unneeded ROCm platform import when using CUDA (#22765)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.chunked_prefill_paged_decode, L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+1/-1); vllm/attention/ops/chunked_prefill_paged_decode.py (+1/-1)
LABELS: bug, rocm, ready
BODY: Without this move, we were importing and warning about rocm libraries not available on cuda ⏎  ⏎ ``` ⏎ vllm serve ⏎ ... ⏎ (EngineCore_0 pid=1696779) WARNING 08-12 17:06:03 [rocm.py:29] Failed to import from amdsmi with ModuleNotFoundError("No module named 'amdsmi'") ⏎ (EngineCore_0 pid=1696779) WARNING 08-12 17:06:03 [rocm.py:40] Failed to import from vllm._rocm_C with ModuleNotFoundError("No module named 'vllm._rocm_C'") ⏎ ``` ⏎  ⏎ Here is the full stacktrace to  …[truncated]

### L3-dab4f9f764  (L3, 2025-08-13, sha dab4f9f76411, PR #22741)
TITLE: [Chore] Update CODEOWNERS to include @yewentao256 for CUDA kernels, attention backends, quantization, and related tests (#22741)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .github/CODEOWNERS (+5/-5)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ Thank so much to Kaichao (@youkaichao ) for the invitation to serve as a vLLM committer. I’m honored and excited to help maintain kernel performance optimizations and related components. ⏎  ⏎ I would also like to thank @mgoin , @robertgshaw2-redhat ,@taneem-ibrahim, @ProExpertProg, @varun-sundar-rabindranath, @tlrmchlsmth , @LucasWilkinson , @SageMoore , @terrytangyuan , @houseroad,  and the vLLM community for their guidance, reviews, and …[truncated]

### L3-d94e3026de  (L3, 2025-08-13, sha d94e3026ded8, PR #22705)
TITLE: [V1] Add tree drafting tests for eagle spec decoding (#22705)
SOURCES: path_core
ARTIFACT_HINTS: L3.tree_attention
FILES: vllm/v1/attention/backends/tree_attn.py (+3/-3); tests/v1/spec_decode/test_eagle.py (+157/-3); tests/v1/spec_decode/test_max_len.py (+0/-6); vllm/v1/spec_decode/eagle.py (+18/-43)
LABELS: speculative-decoding, ready, v1
BODY: ## Purpose ⏎ The goal of this PR is to add tests for eagle tree drafting. Tree attention and drafting was originally added in this PR: https://github.com/vllm-project/vllm/pull/20401. However, it is missing sufficient testing to ensure it does not break in the future. ⏎  ⏎ In addition, tree attention was causing a new test (added here: https://github.com/vllm-project/vllm/pull/22664) to fail. I have fixed the tree draft proposer code so that this test  …[truncated]

### L3-279a5f31b3  (L3, 2025-08-14, sha 279a5f31b3fa, PR #22346)
TITLE: [Kernel] Add nvfp4 gemm flashinfer backends (#22346)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/envs.py (+7/-0); vllm/utils/flashinfer.py (+71/-0); vllm/v1/worker/gpu_worker.py (+2/-2); .buildkite/test-pipeline.yaml (+1/-0); tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py (+139/-0); tests/kernels/quantization/test_nvfp4_scaled_mm.py (+3/-0); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py (+49/-15); vllm/model_executor/layers/quantization/modelopt.py (+61/-23); vllm/model_executor/warmup/kernel_warmup.py (+38/-1)
LABELS: ready, ci/build, v1
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Add FP4 gemms from FlashInfer as backends. ⏎  ⏎ If FlashInfer is available, defaults to FlashInfer cutlass kernels because testing on Llama3 ISL=OSL=1024 max_num_batched_tokens=8192 concurrency=128 shows that it is always faster than alternatives. ⏎  ⏎ Time of original vLLM cutlass kernels vs FlashInfer cutlass kernels in gen-only phase in μs: ⏎ fc_qkv: 21.3 vs 14.8 ⏎ fc_out: 15.1 vs 1 …[truncated]

### L3-39cd09dc86  (L3, 2025-08-14, sha 39cd09dc86ca, PR #22933)
TITLE: [Bugfix] use flash attn on sm90 (#22933)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+1/-1)
LABELS: ready
BODY: Earlier forgot to add `if` to see if the current platform is `hopper`.

### L3-5c3fbfe46b  (L3, 2025-08-15, sha 5c3fbfe46bf6, PR #22763)
TITLE: [Feature] Full Cuda Graph Support for Cutlass MLA and 6% E2E Throughput Improvement (#22763)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.cutlass_v1_backend
FILES: vllm/v1/attention/backends/mla/cutlass_mla.py (+14/-2); tests/compile/piecewise/test_full_cudagraph.py (+74/-0)
LABELS: performance, ready, v1
BODY: ## Purpose ⏎  ⏎ Support full cuda graph for cutlass MLA (SM100) and 6% E2E Throughput Improvement ⏎  ⏎ Thanks for previous work of enabling cutlass MLA on SM100! ⏎  ⏎ ## Test ⏎  ⏎ `vllm serve deepseek-ai/DeepSeek-V2-Lite --port 10256 --enable-expert-parallel --data-parallel-size 2 --trust_remote_code  -O '{"full_cuda_graph": true}' --cuda-graph-sizes 16 32 64 128 256 512` ⏎  ⏎ ### Acc ⏎  ⏎ ```bash ⏎ lm_eval --model local-completions --model_args "base_url=http://127.0.0.1 …[truncated]

### L3-1c859a1387  (L3, 2025-08-15, sha 1c859a138728, PR #22969)
TITLE: [V0 Deprecation] Remove advance_step (#22969)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.fork_inline_cmake, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.mla.triton_v0, L3.mla.flashmla_v0_adapter, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+0/-5); vllm/attention/backends/differential_flash_attn.py (+1/-75); vllm/attention/backends/flash_attn.py (+1/-75); vllm/attention/backends/flashinfer.py (+2/-63); vllm/attention/backends/flashmla.py (+1/-14); vllm/attention/backends/mla/common.py (+1/-86); vllm/attention/backends/placeholder_attn.py (+1/-61); vllm/attention/backends/rocm_aiter_mla.py (+0/-21); vllm/attention/backends/rocm_flash_attn.py (+1/-67); CMakeLists.txt (+0/-1); (+6 more)
LABELS: rocm, ready, ci/build
BODY: `advance_step` is only used for multi-step scheduling, which was already deprecated. Therefore, it can be safely removed now.

### L3-74f441f4b5  (L3, 2025-08-15, sha 74f441f4b517, PR #20059)
TITLE: [Core] Allow full cudagraph with separate attention routines and orthogonal to compilation, add support for FA2 and FlashInfer (#20059)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.aiter_fa, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.rocm_aiter, L3.dispatch.abstract_interface, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/compilation/backends.py (+30/-12); vllm/compilation/base_piecewise_backend.py (+0/-72); vllm/compilation/base_static_graph.py (+54/-0); vllm/compilation/cuda_graph.py (+193/-0); vllm/compilation/cuda_piecewise_backend.py (+16/-117); vllm/compilation/monitor.py (+18/-0); vllm/compilation/wrapper.py (+4/-3); vllm/config/__init__.py (+40/-12); vllm/config/compilation.py (+162/-26); vllm/platforms/cuda.py (+8/-5); (+24 more)
LABELS: rocm, tpu, ready, ci/build, v1, llama
BODY: ## What's changed in this PR:   ⏎  ⏎ 1. We allow cudagraph logic to be orthogonal to VLLM compilation in v1. In other words, full cudagraph without compilation is supported in v1 now;   ⏎ 2. Full cudagraph support for Flash Attention v2 and FlashInfer (moved to [#21367](https://github.com/vllm-project/vllm/pull/21367) with further optimizations); ⏎ 3. New CLI flag `cudagraph_mode` is introduced in CompilationConfig, supporting the following five modes: ` …[truncated]

### L3-3e2f7985a2  (L3, 2025-08-15, sha 3e2f7985a2fc, PR #22672)
TITLE: Support multiple attention groups for KV sharing (#22672)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/test_kv_sharing.py (+189/-0); vllm/v1/worker/utils.py (+24/-16)
LABELS: ready, v1
BODY: Summary: ⏎ https://github.com/vllm-project/vllm/pull/21588 added support for multiple attention metadata builders per kv-cache spec.  ⏎  ⏎ As part of this change, each KV cache group now maps to one or more `AttentionGroup`, with one attention group being created for each type of attention backend used.  ⏎  ⏎ However, if we want to enable KV sharing when we have more than one attention group, we run into the following assertion: ⏎ ``` ⏎             assert len( …[truncated]

### L3-177e55e3bd  (L3, 2025-08-15, sha 177e55e3bd3d, PR #22478)
TITLE: [Attention] FA3 Attention Sinks Perf Boost (#22478)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ vLLM side of https://github.com/vllm-project/flash-attention/pull/78 (merge that first) ⏎  ⏎ Shout-out to @jayhshah (the performance wizard 🪄) for the implementation  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ vllm bench serve --dataset-name random --random-input-len=1000 --random-output-len=100 --num-prompts 1000 --port 3333 --model openai/gpt-oss-20b --request-rate 100 ⏎  ⏎ ## PR ⏎  ⏎ ============ Serving Benchmark Result ============ ⏎ Successful reques …[truncated]

### L3-79899b63f6  (L3, 2025-08-15, sha 79899b63f6d4, PR #22449)
TITLE: [Bugfix] Added more env vars to hash (#22449)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+36/-10)
LABELS: ready
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Environment variables that affect the computation graph must be included in the hash in order for the cache to be correct. ⏎ Previously, it's missing a lot of them. ⏎ I added everything that looks like something that will affect the computation graph, but it's a best effort only. ⏎ This is a short-term solution and it's extremely brittle. We should look at rewriting how caching l …[truncated]

### L3-1723ef1aae  (L3, 2025-08-15, sha 1723ef1aae74, PR #22603)
TITLE: minor: zero workspace buffer init for flashinfer trtllm-gen attn (#22603)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/attention/backends/flashinfer.py (+1/-1); vllm/v1/attention/backends/flashinfer.py (+1/-1); tests/kernels/attention/test_flashinfer_trtllm_attention.py (+2/-2)
LABELS: ready, v1
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ flashinfer v0.2.11.post3 updates: https://github.com/flashinfer-ai/flashinfer/pull/1463 ⏎  ⏎ cc @elvischenv ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-070da660c1  (L3, 2025-08-16, sha 070da660c1bf, PR #22735)
TITLE: [Kernel] Simplify `get_kv_cache_layout` and cache `use_trtllm_attention` env-dependent bit (#22735)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.abstract_interface
FILES: vllm/utils/flashinfer.py (+31/-15); vllm/v1/attention/backends/utils.py (+11/-7)
LABELS: ready, v1
BODY: Let's try to keep `get_kv_cache_layout` as simple as possible, if we start adding if/else checks on backends this is going to very quickly become a mess. ⏎  ⏎ We have two ways to set the kv cache layout right now: ⏎  - `envs.VLLM_KV_CACHE_LAYOUT`, which *should* allow the user to specify their preferred layout ⏎  - `_KV_CACHE_LAYOUT_OVERRIDE` for all other runtime use-cases that require force-setting a layout with priority in code. ⏎  ⏎ I would argue we coul …[truncated]

### L3-000cceca8c  (L3, 2025-08-16, sha 000cceca8c32, PR #23016)
TITLE: [Bugfix gpt-oss] Fix float32 convert for flashinfer sink support (#23016)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/attention/layer.py (+9/-0); vllm/v1/attention/backends/flashinfer.py (+0/-3)
LABELS: bug, ready, v1, gpt-oss
ISSUES: #23013 [Bug]: tracking, gpt-oss broken on main on blackwell
BODY: ## Purpose ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/23013 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1 lm_eval --model vllm --model_args pretrained=openai/gpt-oss-20b --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto ⏎ vllm (pretrained=openai/gpt-oss-20b,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   …[truncated]

### L3-292084e72a  (L3, 2025-08-17, sha 292084e72ac5, PR #22967)
TITLE: [BugFix] Fix for IMA in FA3 varlen combine (#22967)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ vLLM side of https://github.com/vllm-project/flash-attention/pull/80 ⏎  ⏎ ## Test Plan + Result ⏎  ⏎ @sarckk confirms this fixes his issue; may help with some of the H100 related issues in https://github.com/vllm-project/vllm/issues/19483 (not the A100 related ones; this issue appears to have multiple independent IMAs being discussed) ⏎  ⏎ ## (Optional) Documentation Update ⏎  ⏎ --- ⏎ [details omitted]

### L3-6d25e3fd6e  (L3, 2025-08-18, sha 6d25e3fd6ea9, PR #23008)
TITLE: Use Blackwell FlashInfer MXFP4 MoE by default if available  (#23008)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/layer.py (+5/-5); vllm/model_executor/layers/quantization/mxfp4.py (+46/-14)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Essentially we should act like `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1` if on SM100 and flashinfer is installed ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ Main branch: ⏎ ``` ⏎ lm_eval --model vllm --model_args pretrained=openai/gpt-oss-20b --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto ⏎ (EngineCore_0 pid=4179585)   File "/home/mgoin/code/vllm/vllm/model_executor/layers/quantization/mxfp4.py", line 356, in process_weights_after_loadi …[truncated]

### L3-14006840ea  (L3, 2025-08-18, sha 14006840eacf, PR #22776)
TITLE: [V0 Deprecation] Remove V0 FlashInfer attention backend (#22776)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.platform.cuda_selection
FILES: vllm/attention/backends/flashinfer.py (+0/-1098); vllm/platforms/cuda.py (+1/-15); tests/basic_correctness/test_basic_correctness.py (+1/-8); tests/compile/test_basic_correctness.py (+1/-1); tests/core/block/e2e/test_correctness_sliding_window.py (+2/-6); tests/distributed/test_pp_cudagraph.py (+0/-1); tests/kernels/attention/test_attention_selector.py (+3/-0); tests/models/quantization/test_fp8.py (+1/-4)
LABELS: ready, needs-rebase
BODY: ## Summary ⏎ - drop legacy FlashInfer attention backend implementation ⏎ - adjust CUDA backend selection and tests to remove FlashInfer support on V0 engine ⏎  ⏎ ## Testing ⏎ - `pre-commit run --files vllm/platforms/cuda.py tests/basic_correctness/test_basic_correctness.py tests/basic_correctness/test_chunked_prefill.py tests/compile/test_basic_correctness.py tests/core/block/e2e/test_correctness_sliding_window.py tests/distributed/test_pp_cudagraph.py tes …[truncated]

### L3-90bbe0a5ad  (L3, 2025-08-18, sha 90bbe0a5adc0, PR #23137)
TITLE: [Log] Warning Once for Cutlass MLA  (#23137)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.cutlass_v1_backend
FILES: vllm/v1/attention/backends/mla/cutlass_mla.py (+3/-3)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Warning Once for Cutlass MLA  ⏎  ⏎ To avoid  ⏎  ⏎ ```bash ⏎ (EngineCore_2 pid=2626618) WARNING 08-18 16:51:26 [cutlass_mla.py:126] Forcing num_kv_splits to 1 ⏎ (EngineCore_2 pid=2626618) WARNING 08-18 16:51:26 [cutlass_mla.py:126] Forcing num_kv_splits to 1 ⏎ (EngineCore_2 pid=2626618) WARNING 08-18 16:51:26 [cutlass_mla.py:126] Forcing num_kv_splits to 1 ⏎ (EngineCore_2 pid=2626618) WARNING 08-18 16:51:26 [cutlass_mla.py:126] Forcing num_kv_splits t …[truncated]

### L3-78dba404ad  (L3, 2025-08-19, sha 78dba404adaa, PR #22725)
TITLE: [Hardware][IBM Z]Enable v1 for s390x and s390x dockerfile fixes (#22725)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.s390x (+79/-8); requirements/common.txt (+2/-1); requirements/cpu.txt (+2/-2); vllm/engine/arg_utils.py (+4/-3); vllm/platforms/cpu.py (+3/-2); vllm/platforms/interface.py (+3/-0); vllm/v1/worker/cpu_worker.py (+3/-2)
LABELS: ready, ci/build, v1
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ This PR enables V1 Engine for IBMZ, as V0 Engine would be deprecated (https://github.com/vllm-project/vllm/pull/20437, https://github.com/vllm-project/vllm/pull/20412). ⏎ Same as this POWER arch PR (https://github.com/vllm-project/vllm/pull/20554) ⏎  ⏎ ## Test Plan ⏎ Inferencing is working as expected, tested locally. ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ (APIServer pid=1) INFO 08-11 19:17:08 [laun …[truncated]

### L3-03752dba8f  (L3, 2025-08-19, sha 03752dba8f1c, PR #21716)
TITLE: [NVIDIA] Support Flashinfer TRTLLM FP8-q/kv/out Attention Kernel (#21716)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/attention/layer.py (+9/-2); vllm/compilation/fusion_attn.py (+43/-34); vllm/utils/flashinfer.py (+16/-6); vllm/v1/attention/backends/flashinfer.py (+80/-29); .buildkite/test-pipeline.yaml (+2/-0); benchmarks/kernels/benchmark_trtllm_decode_attention.py (+149/-134); benchmarks/kernels/benchmark_trtllm_prefill_attention.py (+130/-99); tests/compile/test_fusion_attn.py (+248/-1); tests/kernels/attention/test_flashinfer_trtllm_attention.py (+172/-128)
LABELS: performance, rocm, ready, ci/build, v1, llama
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ - Refactor the `AttentionStaticQuantPattern` in fusion_attn pass, which will **fuse the attn+fp8_quant pattern** for using the TRTLLM FP8-in FP8-out kernel. ⏎ - Align with other backends, Flashinfer will insert FP8 quant op for query if kv cache dtype is set to FP8. ⏎  ⏎ ## Test Plan && Test Result ⏎ Functional: ⏎ - FP8 TRTLLM Prefill/Decode kernel unit test: `tests/kernels/attenti …[truncated]

### L3-95e3095136  (L3, 2025-08-19, sha 95e30951365d, PR #23122)
TITLE: [Misc] Add @tdoublep as a maintainer of hybrid model and Triton-attention related code (#23122)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .github/CODEOWNERS (+9/-0)
LABELS: ci/build
BODY: ## Purpose ⏎  ⏎ I would like to take ownership of: ⏎ - Mamba layers (and other non-attention layers like linear attention) ⏎ - The Hybrid models test  ⏎ - V1 Triton Attention ⏎  ⏎ I am happy to maintain these components of vLLM, including reviewing and shepherding related PRs.  ⏎  ⏎ ## Test Plan ⏎  ⏎ n/a ⏎  ⏎ ## Test Result ⏎  ⏎ n/a ⏎  ⏎ ## (Optional) Documentation Update ⏎  ⏎ --- ⏎ [details omitted]

### L3-5b5f350d67  (L3, 2025-08-19, sha 5b5f350d67a1, PR #23193)
TITLE: [Misc] Enable yapf for FlashInfer backend (#23193)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+24/-13)
LABELS: v1
BODY: 

### L3-e9d6a3db69  (L3, 2025-08-19, sha e9d6a3db69c5, PR #23081)
TITLE: [TPU] make ptxla not imported when using tpu_commons (#23081)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+51/-46); vllm/distributed/device_communicators/tpu_communicator.py (+13/-14); vllm/model_executor/layers/fused_moe/moe_pallas.py (+1/-1); vllm/model_executor/model_loader/default_loader.py (+13/-8); vllm/platforms/tpu.py (+3/-0); vllm/v1/worker/tpu_worker.py (+13/-9)
LABELS: tpu, ready, v1
BODY: ## Purpose ⏎  ⏎ make ptxla not imported when using tpu_commons ⏎  ⏎ ## Test Plan ⏎  ⏎ All the current CI tests ⏎  ⏎ ## Test Result ⏎ Passed ⏎  ⏎ ## (Optional) Documentation Update ⏎  ⏎ --- ⏎ [details omitted]

### L3-e61bac87ee  (L3, 2025-08-19, sha e61bac87eefd, PR #23147)
TITLE: [Misc] Minor refactoring for FlashInfer backend (#23147)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+65/-91)
LABELS: ready, v1
BODY: Moves static attributes (e.g., `num_qo_heads`, `q_data_type`) from `FlashInferMetadata` to `FlashInferMetadataBuilder`. ⏎  ⏎ Plus, some minor code style changes. ⏎  ⏎ Also, this PR includes minor perf optimizations by skipping `get_seq_lens` when `use_cuda_graph and use_tensor_cores`.

### L3-a38b8af4c3  (L3, 2025-08-19, sha a38b8af4c364, PR #22357)
TITLE: [NVIDIA] Add SM100 Flashinfer Cutlass MoE fp8 backend (#22357)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test-pipeline.yaml (+2/-0); tests/kernels/moe/test_flashinfer.py (+248/-0); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py (+33/-22); vllm/model_executor/layers/quantization/fp8.py (+134/-81); vllm/model_executor/layers/quantization/modelopt.py (+83/-35); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+112/-0)
LABELS: performance, ready, ci/build
BODY: ## Purpose ⏎ This PR will add support for flashinfer-cutlass fused MoE kernels for llama4. ⏎  ⏎ ## Test Plan ⏎ Ran `lm_eval`: ⏎ ``` ⏎ VLLM_FLASHINFER_MOE_BACKEND=throughput VLLM_USE_FLASHINFER_MOE_FP8=1 CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 VLLM_ATTENTION_BACKEND=FLASHINFER lm_eval --model vllm --model_args pretrained=<ckpt_path>,tensor_parallel_size=1,max_model_len=2048,kv_cache_dtype=auto --gen_kwargs temperature=0.0 --limit 500 --trust_remote_code --tasks  …[truncated]

### L3-0167efe20d  (L3, 2025-08-19, sha 0167efe20d3d, PR #21917)
TITLE: [Core] Optimize scheduler request removal for single completions (#21917)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/core/sched/scheduler.py (+6/-8); vllm/v1/core/sched/utils.py (+33/-0)
LABELS: ready, v1
BODY: - Add fast path for single request removal - Maintain original logic for multiple requests ⏎   - Achieve 18.7% scheduler performance improvement ⏎   - Minimal code changes with maximum impact ⏎  ⏎   Performance results: ⏎   - 2.6-3.4x speedup for single request removal ⏎   - 64% improvement in request removal operations ⏎   - Production impact: 560ms saved per hour at 200 req/s ⏎  ⏎   Signed-off-by: cliu_whu@yeah.net ⏎  ⏎ # Essential Elements of an Effective PR Descri …[truncated]

### L3-14e2b0730b  (L3, 2025-08-19, sha 14e2b0730bf9, PR #23200)
TITLE: [BugFix] fix CUTLASS MLA full cudagraph  (#23200)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.cutlass_v1_backend
FILES: vllm/v1/attention/backends/mla/cutlass_mla.py (+1/-1)
LABELS: ready, v1
BODY: Fix:  ⏎ ``` ⏎ EngineCore_0 pid=1248800) ValueError: CUDAGraphMode.FULL is not supported with CutlassMLAMetadataBuilder backend (support: AttentionCGSupport.NEVER); please try cudagraph_mode=PIECEWISE, and make sure compilation level is piecewise ⏎ ``` ⏎  ⏎ caused by CUTLASS MLA getting missed in https://github.com/vllm-project/vllm/pull/20059 ⏎  ⏎ thanks @tjtanaa for the find!

### L3-a634733f67  (L3, 2025-08-20, sha a634733f67b3, PR #23185)
TITLE: [Attention] Optimize make_local_attention_virtual_batches for Flash Attention (#23185)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+7/-7)
LABELS: ready, v1
BODY: ## Purpose ⏎ This PR improves efficiency of `cu_seqlens_q_local` and `batch_indices` in `make_local_attention_virtual_batches`. ⏎  ⏎ For `cu_seqlens_q_local`, original code creates multiple intermediate arrays (`np.cumsum`, `np.pad`, `.astype(np.int32)`); also `np.pad` is not necessary. In the new code, we pre-allocate output array and let `np.cumsum` directly write to the pre-allocated array. ⏎  ⏎ For `batch_indices`, used the following operations to avoi …[truncated]

### L3-d983769c41  (L3, 2025-08-20, sha d983769c41db, PR #22721)
TITLE: fix cuda graph (#22721)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+4/-3)
LABELS: ready, v1
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-50df09fe13  (L3, 2025-08-20, sha 50df09fe13c9, PR #23129)
TITLE: Update to flashinfer-python==0.2.12 and disable AOT compile for non-release image (#23129)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+33/-19); setup.py (+1/-1); .buildkite/release-pipeline.yaml (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.2.12 ⏎  ⏎ Also include small update to download cubins ahead of time from flashinfer. This affects docker build time, so I moved AOT compilation to not happen by default. You can enable AOT compile by setting `--build-arg FLASHINFER_AOT_COMPILE=true` during docker build ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update ⏎  ⏎ --- ⏎ [details omitted]

### L3-d6d13bd49e  (L3, 2025-08-20, sha d6d13bd49ed7, PR #23216)
TITLE: [Misc] Add max_seq_len to CommonAttentionMetadata  (#23216)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v1_backend, L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.aiter_fa, L3.dispatch.abstract_interface, L3.flex_attention, L3.tree_attention
FILES: vllm/v1/attention/backends/flash_attn.py (+1/-1); vllm/v1/attention/backends/flashinfer.py (+1/-1); vllm/v1/attention/backends/flex_attention.py (+1/-1); vllm/v1/attention/backends/rocm_aiter_fa.py (+1/-1); vllm/v1/attention/backends/tree_attn.py (+1/-1); vllm/v1/attention/backends/triton_attn.py (+1/-1); vllm/v1/attention/backends/utils.py (+6/-0); vllm/v1/attention/backends/xformers.py (+1/-1); tests/v1/attention/utils.py (+2/-0); tests/v1/spec_decode/test_tree_attention.py (+2/-0); (+2 more)
LABELS: speculative-decoding, ready, v1
BODY: `max_seq_len` is commonly used so I think it's worthwhile to add it into `CommonAttentionMetadata`.

### L3-3b11b26b50  (L3, 2025-08-20, sha 3b11b26b5069, PR #22795)
TITLE: [FIXBUG ] Allow disabling rocm_aiter_fa backend for ROCm GPUs not compatible with AITER (#22795)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/spec_decode/eagle.py (+45/-35)
LABELS: rocm, speculative-decoding, ready, v1
BODY: This PR fixes an issue where VLLM failed to start on ROCm GPUs that are not compatible with the rocm_aiter_fa attention backend. An example of such a GPU is the AMD Radeon RX 7900 XTX, which uses the RDNA 3 architecture. ⏎  ⏎ The bug was introduced in commit 1ee5ead5f8f1c3c77b73effcb230ee02952fbe1f, which hardcoded the loading of the vllm.v1.attention.backends.rocm_aiter_fa module in vllm/v1/spec_decode/eagle.py. This forced VLLM to fail on startup b …[truncated]

### L3-bf7c99dfc4  (L3, 2025-08-20, sha bf7c99dfc40b, PR #20413)
TITLE: [Perf] Speed up function `_convert_tokens_to_string_with_added_encoders` by 13.7x (#20413)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/transformers_utils/detokenizer_utils.py (+15/-10)
LABELS: ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Incremental decoding can be slow, according to an original comment on this code section. I tried optimizing it with Codeflash (the automated performance optimization tool I'm building) and found this optimization. I verified it with the below benchmark, and generated many tests to ensure correctness. ⏎  ⏎ ⏱️ Runtime :   **`0.28 seconds`**  **→** **`0.019 seconds`** (per 1000 l …[truncated]

### L3-a4fbb32fab  (L3, 2025-08-20, sha a4fbb32fab3d, PR #23183)
TITLE: Remove chunked_prefill_enabled flag in V1 MLA (#23183)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+23/-27)
LABELS: ready, v1
BODY: ## Purpose ⏎ Chunked prefill is required in V1 MLA. This PR removes the `chunked_prefill_enabled` flag, left over from V0.

### L3-10cc12ba66  (L3, 2025-08-20, sha 10cc12ba6683, PR #23195)
TITLE: Feature/mla tests (#23195)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+8/-8); tests/v1/attention/test_attention_backends.py (+11/-15); tests/v1/attention/test_mla_backends.py (+522/-0); tests/v1/attention/utils.py (+10/-1)
LABELS: rocm, ready, ci/build, v1
BODY: ## Purpose ⏎ MLA backends are currently missing unit tests. This PR adds `test_mla_backends.py`, borrowing from the existing `test_attention_backends.py`. ⏎  ⏎ ## Test Plan ⏎ `pytest tests/v1/attention/test_attention_backends.py` ⏎  ⏎ ## Test Result ⏎ Tests pass ⏎  ⏎ --- ⏎ [details omitted]

### L3-f94bf9b924  (L3, 2025-08-21, sha f94bf9b924af, PR #23287)
TITLE: [Compile] Fix Compile Warning SM100 Cutlass MLA (#23287)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: csrc/attention/mla/sm100_cutlass_mla_kernel.cu (+2/-2)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Fix compile warning ⏎  ⏎ ```bash ⏎ /home/wentao/vllm-source/csrc/attention/mla/sm100_cutlass_mla_kernel.cu:149:400: warning: narrowing conversion of ‘num_kv_splits’ from ‘int64_t’ {aka ‘long int’} to ‘int’ [-Wnarrowing] ⏎ /home/wentao/vllm-source/csrc/attention/mla/sm100_cutlass_mla_kernel.cu: In instantiation of ‘typename T::Fmha::Arguments args_from_options(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at …[truncated]

### L3-1d353b6352  (L3, 2025-08-21, sha 1d353b6352da, PR #23214)
TITLE: [Core] Always use tensor cores for Flashinfer Decode Wrapper (#23214)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/envs.py (+0/-7); vllm/v1/attention/backends/flashinfer.py (+28/-50); benchmarks/kernels/benchmark_trtllm_decode_attention.py (+1/-1); tests/kernels/attention/test_flashinfer.py (+2/-4); tests/kernels/attention/test_flashinfer_trtllm_attention.py (+1/-3)
LABELS: performance, ready, v1
BODY: ## Purpose ⏎ Always enable tensor cores for the flashinfer `BatchDecodeWithPagedKVCacheWrapper`. Previously we enabled tensor cores only for head group ratio > 4 because not using tensor cores for those sizes caused a dramatic perf drop. However, for head group ratio 1-4, we can still enable tensor cores for computation because they are always as good as cuda cores for MMAs.  ⏎  ⏎ ## Test Plan ⏎ Existing flashinfer tests should cover the tests ⏎ ## Test Re …[truncated]

### L3-394591e343  (L3, 2025-08-21, sha 394591e34371, PR #23351)
TITLE: [Feature] Enable DeepGEMM Linear on B200; 1.5% E2E throughput improvement (#23351)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/utils/fp8_utils.py (+6/-16); vllm/utils/deep_gemm.py (+7/-0)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Enable DeepGEMM Linear on B200 ⏎  ⏎ This should also fix some cutlass linear acc error since the weight is quantized to e8m0 ⏎  ⏎ ## Test ⏎  ⏎ ```bash ⏎ VLLM_USE_DEEP_GEMM=1 lm_eval   --model vllm   --model_args "pretrained=Qwen/Qwen3-30B-A3B-FP8,max_model_len=32768,enforce_eager=True"   --trust_remote_code   --tasks gsm8k   --num_fewshot 5   --batch_size auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:| …[truncated]

### L3-19fe1a0510  (L3, 2025-08-22, sha 19fe1a051085, PR #22668)
TITLE: [Kernel] Add FP8 support with FlashMLA backend (#22668)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.mla.triton_v0, L3.mla.common_v1, L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.mla.flashmla_build, L3.mla.cutlass_v1_backend, L3.mla.rocm_aiter, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: cmake/external_projects/flashmla.cmake (+5/-4); csrc/cache_kernels.cu (+29/-28); csrc/torch_bindings.cpp (+9/-4); vllm/_custom_ops.py (+12/-8); vllm/attention/backends/mla/common.py (+9/-5); vllm/attention/ops/flashmla.py (+6/-0); vllm/engine/arg_utils.py (+1/-2); vllm/platforms/cuda.py (+33/-8); vllm/platforms/interface.py (+2/-1); vllm/platforms/rocm.py (+2/-1); (+9 more)
LABELS: rocm, tpu, ready, ci/build, v1
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Enable FP8 KV cache with MLA ⏎  ⏎ ## Test Plan ⏎  ⏎ ### Correctness ⏎ ``` ⏎ pytest tests/kernels/attention/test_flashmla.py ⏎ pytest tests/kernels/attention/test_cache.py::test_gather_and_maybe_dequant_cache_mla ⏎ ``` ⏎ ### Accuracy ⏎  ⏎ With `kv_cache_type = "auto"`: ⏎ `VLLM_ATTENTION_BACKEND=FLASHMLA lm_eval --model vllm --model_args '{"pretrained": "deepseek-ai/DeepSeek-V2-Lite-Chat", "trust_r …[truncated]

### L3-17373dcd93  (L3, 2025-08-22, sha 17373dcd93ca, PR #23154)
TITLE: [Attention] Refactor AttentionMetadata Preparation for Encoder-only Models (#23154)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/attention/layers/chunked_local_attention.py (+16/-13); vllm/attention/layers/encoder_only_attention.py (+86/-0); vllm/v1/attention/backends/utils.py (+1/-31); tests/v1/worker/test_gpu_model_runner.py (+7/-4); vllm/model_executor/models/bert.py (+8/-9); vllm/model_executor/models/bert_with_rope.py (+8/-9); vllm/model_executor/models/llama.py (+5/-1); vllm/model_executor/models/modernbert.py (+7/-7); vllm/model_executor/models/qwen2.py (+4/-1); vllm/v1/kv_cache_interface.py (+8/-0); (+2 more)
LABELS: ready, v1, llama, qwen
BODY: ## Purpose ⏎ Clean up attention metadata preparation of encoder-only models. Prepare cleaner code base for encoder-decoder. ⏎  ⏎ ## Test Plan ⏎ Test an attention free model by ⏎ ``` ⏎ pytest -vs test_gte.py::test_rerank_models_mteb ⏎ ``` ⏎  ⏎ ## Test Result ⏎  ⏎ Can pass ⏎  ⏎ ## (Optional) Documentation Update ⏎  ⏎ --- ⏎ [details omitted]

### L3-285178b3b8  (L3, 2025-08-22, sha 285178b3b824, PR #23418)
TITLE: [V0 Deprecation] Remove V0 LoRA test (#23418)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/lora/conftest.py (+4/-27); tests/lora/test_add_lora.py (+4/-7); tests/lora/test_llama_tp.py (+1/-4); tests/lora/test_lora_manager.py (+68/-62); tests/lora/test_mixtral.py (+0/-1); tests/lora/test_worker.py (+5/-15); tests/lora/utils.py (+76/-0)
LABELS: ready, llama
BODY: ## Purpose ⏎ Remove all LoRA tests that depend on V0 in support of https://github.com/vllm-project/vllm/pull/22804 ⏎ ## Test Plan ⏎ CI LoRA tests ⏎ ## Test Result ⏎ All LoRA tests should pass correctly ⏎ ## (Optional) Documentation Update ⏎  ⏎ --- ⏎ [details omitted]

### L3-281710ef9a  (L3, 2025-08-22, sha 281710ef9a2a, PR #23297)
TITLE: [Attention] Allow V1 flash_attn to support cross-attention (#23297)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+7/-10)
LABELS: ready, v1
BODY: This is a piece of #21088 split out for more focused review. ⏎  ⏎ These are the only changes to v1's flash_attn backend to support ⏎ cross-attention. ⏎  ⏎ 1. Remove the block on ENCODER_DECODER attention. ⏎  ⏎ 2. Handle the fact that `key` and `value` may be `None`. These are set ⏎    the first time the decoder runs, but for subsequent runs, they come ⏎    from KV cache only.

### L3-da65bec309  (L3, 2025-08-22, sha da65bec3096b, PR #22675)
TITLE: add an env var for path to pre-downloaded flashinfer cubin files (#22675)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/envs.py (+6/-0); vllm/utils/flashinfer.py (+5/-0)
LABELS: performance, ready
BODY: Summary: ⏎ Previously https://github.com/vllm-project/vllm/pull/21893 added a check for network access to NV cubin artifact endpoint. And if the check failed, we would opt out from trtllm attn backend. ⏎  ⏎ This PR added a env var `FLASHINFER_CUBIN_DIR` to allow people specify a cubin dir which contains pre-downloaded cubin files. When `FLASHINFER_CUBIN_DIR` is specified, we will directly return True in `has_nvidia_artifactory()`. ⏎  ⏎ Differential Revisio …[truncated]

### L3-24d0c9e6ed  (L3, 2025-08-22, sha 24d0c9e6edc4, PR #22703)
TITLE: [NVIDIA][torch.compile] Support Flashinfer TRTLLM FP8-q/kv NVFP4-out Attention Kernel (#22703)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.xformers.v1_backend, L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.rocm_flash_attn_v0, L3.rocm.aiter_fa, L3.mla.triton_v0, L3.mla.common_v1, L3.dispatch.abstract_interface, L3.flex_attention, L3.tree_attention
FILES: vllm/attention/backends/abstract.py (+5/-8); vllm/attention/backends/differential_flash_attn.py (+6/-0); vllm/attention/backends/dual_chunk_flash_attn.py (+2/-1); vllm/attention/backends/flash_attn.py (+2/-1); vllm/attention/backends/mla/common.py (+2/-1); vllm/attention/backends/rocm_flash_attn.py (+9/-5); vllm/attention/backends/xformers.py (+2/-1); vllm/attention/layer.py (+5/-2); vllm/compilation/fusion.py (+18/-48); vllm/compilation/fusion_attn.py (+151/-34); (+17 more)
LABELS: performance, rocm, tpu, ready, ci/build, v1, llama
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ This PR based on the previous attn + FP8-quant fusion #21716, adding another attn + NVFP4-quant fusion for supporting TRTLLM-gen attn kernel. ⏎  ⏎ ## Test Plan && Test Result ⏎ Functional: ⏎ - NVFP4 TRTLLM Prefill/Decode kernel unit test: tests/kernels/attention/test_flashinfer_trtllm_attention.py ⏎ ``` ⏎ ======= 44 passed, 4 skipped, 4 warnings in 64.13s (0:01:04) ====== ⏎ ``` ⏎ - Attenti …[truncated]

### L3-65197a5fb3  (L3, 2025-08-23, sha 65197a5fb37e, PR #23459)
TITLE: [Misc] Modify CacheConfig import (#23459)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layers/encoder_only_attention.py (+1/-1)
LABELS: ready
BODY: ## Purpose ⏎ When using the latest main branch of `transformers`, encounter the following import error ⏎ ```text ⏎ ERROR 08-23 02:59:11 [registry.py:430]   File "/root/Code/vllm_dev/vllm/vllm/attention/layers/encoder_only_attention.py", line 8, in <module> ⏎ ERROR 08-23 02:59:11 [registry.py:430]     from transformers import CacheConfig ⏎ ERROR 08-23 02:59:11 [registry.py:430] ImportError: cannot import name 'CacheConfig' from 'transformers' (/root/Code/vl …[truncated]

### L3-e76e233540  (L3, 2025-08-24, sha e76e23354033, PR #23198)
TITLE: [kernel] Support W4A8 on Hopper (#23198)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+27/-0); benchmarks/kernels/benchmark_machete.py (+33/-0); benchmarks/kernels/weight_shapes.py (+6/-0); csrc/quantization/cutlass_w4a8/w4a8_mm_entry.cu (+418/-0); csrc/torch_bindings.cpp (+20/-0); tests/kernels/quantization/test_cutlass_w4a8.py (+259/-0); vllm/_custom_ops.py (+48/-0); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py (+37/-6); vllm/model_executor/layers/quantization/compressed_tensors/schemes/__init__.py (+3/-1); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py (+160/-0); (+2 more)
LABELS: performance, ready, ci/build
BODY: ## Purpose ⏎ Add support in vLLM for CUTLASS-based W4A8 kernel on Hopper, see [example 55](https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm.cu) which uses LUT trick to bypass int4 -> bf16 -> fp8 conversion in the GEMM mainloop. This improves the compute-bound performance and allows W4A8 to approach peak FP8 throughput while still maintaining the fast decoding speed of W4A16. ⏎  ⏎ The kernel perform …[truncated]

### L3-5c4b6e66fe  (L3, 2025-08-25, sha 5c4b6e66fec0, PR #23171)
TITLE: [Attention] Unify mamba and attention backend selection (#23171)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+2/-1); vllm/model_executor/layers/attention_layer_base.py (+23/-0); vllm/v1/worker/gpu_model_runner.py (+8/-18); tests/v1/attention/test_attention_backends_selection.py (+104/-0); tests/v1/attention/test_mamba_selectors.py (+0/-25); vllm/model_executor/layers/mamba/abstract.py (+13/-2); vllm/model_executor/layers/mamba/mamba_mixer.py (+9/-1); vllm/model_executor/layers/mamba/mamba_mixer2.py (+9/-1); vllm/model_executor/layers/mamba/short_conv.py (+9/-1); vllm/model_executor/models/minimax_text_01.py (+9/-1); (+1 more)
LABELS: ready, v1
ISSUES: #23073 [Refactor]: Get rid of `get_mamba_attn_backend`
BODY: ## Purpose ⏎  ⏎ Fixes #23073. This PR unifies mamba and attention backend selection logic by removing the separate `get_mamba_attn_backend()` function and implementing the standard `Layer.get_attn_backend()` interface for all layer types. ⏎  ⏎ **Fixes the issue where mamba models used duplicate backend selection logic instead of the unified approach.** ⏎  ⏎ ### Changes Made: ⏎ - **Deleted** `vllm/v1/attention/backends/mamba_selectors.py` to eliminate duplicate …[truncated]

### L3-e0329ed4b4  (L3, 2025-08-25, sha e0329ed4b426, PR #21416)
TITLE: Updates to Flex + VLLm integration (#21416)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+334/-68); tests/kernels/test_flex_attention.py (+87/-23); tests/v1/attention/test_attention_backends.py (+18/-12)
LABELS: ready, v1
BODY: ## Purpose ⏎ Improve flex attention performance by adding a custom blockmask metadata builder for common case. ⏎ Also updates to newer metadata passing APIs. ⏎ Co-authored by Horace ⏎  ⏎ ## Test Plan ⏎  pytest tests/kernels/test_flex_attention.py      ⏎  ⏎   ⏎  Here is my perf numbers on a sweep of vllm, using this script: ⏎  https://gist.github.com/drisspg/c983e853ba8e9d999ae429783cde3c2f ⏎  ⏎ Batch | Total Tokens | Avg Tokens/Prompt | Token Range | Flex Input (tok/s)  …[truncated]

### L3-8a3cd90af5  (L3, 2025-08-25, sha 8a3cd90af534, PR #23274)
TITLE: [Kernel] Add fused grouped_topk kernel for MoE (#23274)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.flashinfer.trtllm_gen
FILES: CMakeLists.txt (+3/-1); csrc/moe/grouped_topk_kernels.cu (+757/-0); csrc/moe/moe_ops.h (+5/-0); csrc/moe/torch_bindings.cpp (+6/-0); tests/kernels/moe/test_grouped_topk.py (+76/-0); vllm/_custom_ops.py (+11/-0); vllm/envs.py (+6/-0); vllm/model_executor/layers/fused_moe/fused_moe.py (+45/-1)
LABELS: rocm, ready, ci/build
BODY: ## Purpose ⏎  ⏎ This PR add fused grouped_topk kernel for MoE. ⏎  ⏎ * `grouped_topk_kernels.cu`: grouped topk kernel ⏎   * Added `renormalize` arg, instead of always renormalize ⏎ * Introduce `routed_scaling_factor` parameter to both `grouped_topk` and `fused_grouped_topk`, which is needed for grouped topk compute. ⏎  ⏎ ## Test Plan ⏎  ⏎ Added unit tests. ⏎  ⏎ ``` ⏎ pytest -s -v tests/kernels/moe/test_grouped_topk.py ⏎ ``` ⏎ ## Test Result ⏎  ⏎ Unit tests passed. ⏎  ⏎ ## Accuracy Tes …[truncated]

### L3-efc88cf64a  (L3, 2025-08-25, sha efc88cf64a39, PR #23585)
TITLE: [Misc] Simplify FlashInfer attention metadata (#23585)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+114/-163)
LABELS: ready, v1
BODY: Currently, the `build` method of the FlashInfer backend passes arguments to `self._plan()` by adding them to `FlashInferMetadata`, while most of them are not actually used in the forward pass. This is an unnecessary complexity and can be simply avoided by integrating `self._plan()` into `self.build()`.

### L3-ae067888d6  (L3, 2025-08-25, sha ae067888d680, PR #23537)
TITLE: Update Flashinfer to  0.2.14.post1 (#23537)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); setup.py (+1/-1); vllm/compilation/collective_fusion.py (+2/-1); vllm/v1/worker/gpu_worker.py (+4/-3); vllm/model_executor/layers/quantization/mxfp4.py (+6/-1)
LABELS: ready, ci/build, v1
BODY: ## Purpose ⏎ Update flashinfer to ⏎ - fix allreduce fusion kernel suboptimal perf issue. ⏎ - Add GPT-OSS cutlass MoE backend. ⏎ - Include https://github.com/vllm-project/vllm/pull/23209 for flashinfer autotune @IwakuraRein  ⏎ - Include  https://github.com/vllm-project/vllm/pull/23311 for flashinfer enum API change. ⏎  ⏎ ## Test Plan ⏎ lm_eval on llama3 ⏎  ⏎ gpt-oss/eval test on gpt-oss ⏎ - `python3 -m gpt_oss.evals --sampler chat_completions --model gpt-oss-120b --rea …[truncated]

### L3-b395b3b0a3  (L3, 2025-08-25, sha b395b3b0a316, PR #22760)
TITLE: [Disagg][Perf] Use CUDA event sync instead of blocking `tolist` to avoid unintentional copy ops blocking across different CUDA streams, improving disagg TTIT/TTFT (#22760)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_model_runner.py (+23/-2)
LABELS: ready, v1
BODY: ## Current Issue ⏎ Mitigation to avoid blocking copy operations across different CUDA streams. Details could be found in: https://github.com/vllm-project/vllm/issues/22754 ⏎  ⏎ ## Change ⏎ When we copy the sampled valid token ids from device to host, avoid using `tolist` which would trigger a CUDA wise stream sync if the source is on device. We change it to use non-blocking copy followed by an explicit CUDA event sync. ⏎  ⏎ ## Test for Non Disagg ⏎ Bring up vL …[truncated]

### L3-f66673a39d  (L3, 2025-08-26, sha f66673a39d9f, PR #22895)
TITLE: [Kernel] Added flashinfer fp8 per-tensor gemms (#22895)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+61/-0); .buildkite/test-pipeline.yaml (+1/-0); tests/compile/test_fusion.py (+7/-8); tests/compile/test_sequence_parallelism.py (+1/-2); tests/compile/test_silu_mul_quant_fusion.py (+6/-7); tests/kernels/quantization/test_flashinfer_scaled_mm.py (+73/-0); vllm/model_executor/layers/quantization/fp8.py (+3/-2); vllm/model_executor/layers/quantization/ptpc_fp8.py (+2/-2); vllm/model_executor/layers/quantization/utils/w8a8_utils.py (+44/-15)
LABELS: ready, ci/build, v1
BODY: ## Purpose ⏎ Added fp8 gemms from flashinfer. ⏎ The added gemms have better or same perf as the original gemms so we use it as the default. ⏎ For gemm sizes with small M, the added gemms are marginally faster. ⏎ For gemm sizes with large M, the added gemms are much faster. ⏎ These are the results for llama3 ISL=OSL=1024 concurrency=128 max_num_batched_tokens=8192 TP1. ⏎ As expected, TPOT is roughly the same but TTFT improved by ~13%. ⏎  ⏎ ~Requires flashinfer au …[truncated]

### L3-2b4fc9bd9b  (L3, 2025-08-26, sha 2b4fc9bd9b83, PR #23299)
TITLE: Support FlashAttention Backend for Hybrid SSM Models (#23299)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_model_runner.py (+17/-24); tests/models/language/generation/test_hybrid.py (+0/-3)
LABELS: ready, v1
BODY: ## Purpose ⏎ As mentioned in https://github.com/vllm-project/vllm/pull/20016, v1 hybrid ssm requires the layout of attention to be (num_blocks, 2, hidden_size) so that the blocks can be shared between attention layers and mamba layers. This PR supports (2, num_blocks, hidden_size) layout by changing the tensor stride of (2, num_blocks) dimensions to (hidden_size, 2*hidden_size) ⏎ ## Test Plan ⏎ Run unit tests. Should pass ⏎ ## Test Result ⏎ Let's wait for  …[truncated]

### L3-730d0ac8b9  (L3, 2025-08-26, sha 730d0ac8b967, PR #23649)
TITLE: [Docs] Fix warnings in `mkdocs build` (#23649)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/differential_flash_attn.py (+9/-5); vllm/attention/backends/flash_attn.py (+3/-2); vllm/attention/backends/rocm_flash_attn.py (+6/-5); vllm/attention/backends/utils.py (+1/-1); vllm/attention/backends/xformers.py (+6/-6); vllm/core/block_manager.py (+3/-5); vllm/engine/async_llm_engine.py (+2/-2); vllm/engine/llm_engine.py (+4/-4); vllm/entrypoints/llm.py (+5/-5); vllm/entrypoints/openai/tool_parsers/minimax_tool_parser.py (+2/-1); (+4 more)
LABELS: rocm, frontend, ready, tool-calling
BODY: As discussed with @hmellor on [Slack](https://vllm-dev.slack.com/archives/C07R5Q1Q2BB/p1756198808485369), this PR addresses the warnings generated when running mkdocs build. (#22588) ⏎  ⏎ Please let me know if you have any feedback! ⏎  ⏎ --- ⏎  ⏎ > [!NOTE] ⏎ > Warnings fixed: ⏎  ⏎ ``` ⏎ WARNING -  griffe: vllm/model_executor/layers/lightning_attn.py:461: No type or annotation for parameter 'q' ⏎ WARNING -  griffe: vllm/model_executor/layers/lightning_attn.py:462: No t …[truncated]

### L3-227e231b55  (L3, 2025-08-26, sha 227e231b5590, PR #23665)
TITLE: [Docs] [V1] [Hybrid] Update docs to remove FlashInfer constraint for hybrid models (#23665)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docs/usage/v1_guide.md (+2/-3)
LABELS: documentation, ready
BODY: ## Purpose ⏎  ⏎ Update docs to reflect new status since #23299 was merged.  ⏎  ⏎ cc @heheda12345  ⏎  ⏎ ## Test Plan ⏎  ⏎ n/a ⏎  ⏎ ## Test Result ⏎  ⏎ n/a ⏎  ⏎ --- ⏎ [details omitted]

### L3-c3b0fd1ee6  (L3, 2025-08-26, sha c3b0fd1ee670, PR #23536)
TITLE: [V1][P/D]P2pNcclConnector supports flashinfer (#23536)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py (+78/-80)
LABELS: ready
BODY: - Currently, the P2pNcclConnector only supports KV-cache layouts similar to those used in Flash-Attention, e.g., [2, num_pages, page_size, xxx], and those used in MLA, e.g., [num_pages, page_size, xxx]. ⏎  ⏎ - In contrast, FlashInfer’s KV-cache layout is [num_pages, 2, page_size, xxx].

### L3-6578e87365  (L3, 2025-08-27, sha 6578e8736558, PR #23174)
TITLE: Optimize input preparation for FlashInfer [2/N] (#23174)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+54/-26)
LABELS: speculative-decoding, ready, v1
BODY: Should be merged after #23147

### L3-11eddf02f0  (L3, 2025-08-27, sha 11eddf02f023, PR #23732)
TITLE: [FlashInfer] Cache hyper params in metadata builder (#23732)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+15/-15)
LABELS: v1
BODY: Minor code simplification

### L3-fce10dbed5  (L3, 2025-08-27, sha fce10dbed544, PR #22609)
TITLE: [XPU] Add xpu torch.compile support (#22609)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/attention/layer.py (+1/-2); .buildkite/scripts/hardware_ci/run-xpu-test.sh (+1/-0); vllm/compilation/fix_functionalization.py (+8/-0); vllm/platforms/cpu.py (+4/-0); vllm/platforms/cuda.py (+4/-0); vllm/platforms/interface.py (+8/-0); vllm/platforms/rocm.py (+4/-0); vllm/platforms/xpu.py (+6/-9)
LABELS: rocm, ready, ci/build
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ This PR enable torch compile on XPU platform. user can enable with `-O3` option ⏎ limitations: ⏎ - due to xpu still use ipex kernels for now, custom ops are not register. all custom ops (except attention) are not enabled, will use torch native impl. we will improve this with vllm-xpu-kernels. meanwhile, almost all custom passes are not supported.   ⏎ - xpu not support graph mode  …[truncated]

### L3-3af47c3cc6  (L3, 2025-08-27, sha 3af47c3cc693, PR #23666)
TITLE: [Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt (#23666)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/kernels/moe/test_block_fp8.py (+2/-3); tests/kernels/moe/test_deepep_deepgemm_moe.py (+3/-4); vllm/envs.py (+7/-1); vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py (+2/-2); vllm/model_executor/layers/fused_moe/fused_moe.py (+3/-4); vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py (+3/-3); vllm/model_executor/layers/quantization/fp8.py (+4/-5); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+2/-2); vllm/transformers_utils/config.py (+18/-0); vllm/utils/deep_gemm.py (+24/-29)
LABELS: ready, deepseek
BODY: ## Purpose ⏎  ⏎ Recently DeepGEMM has supported E8M0 scale on Hopper, and this is also required by DeepSeekV3.1 ⏎  ⏎ This PR adds the support for it ⏎  ⏎ ## Test ⏎  ⏎ ### Unit Test ⏎  ⏎ ```bash ⏎ (wentao) wentao@H100-GPU17:~/vllm/tests/kernels/moe$ pytest -x test_deepgemm.py  ⏎ ======================================= test session starts ======================================== ⏎ platform linux -- Python 3.12.11, pytest-8.4.1, pluggy-1.6.0 ⏎ rootdir: /home/wentao/vllm ⏎ config …[truncated]

### L3-4e4d017b6f  (L3, 2025-08-27, sha 4e4d017b6f70, PR #23743)
TITLE: [Docs] Fix warnings in `mkdocs build` (continued) (#23743)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v1_backend, L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.aiter_fa, L3.flex_attention, L3.tree_attention
FILES: vllm/v1/attention/backends/cpu_attn.py (+2/-1); vllm/v1/attention/backends/flash_attn.py (+2/-1); vllm/v1/attention/backends/flashinfer.py (+3/-5); vllm/v1/attention/backends/flex_attention.py (+2/-1); vllm/v1/attention/backends/pallas.py (+3/-2); vllm/v1/attention/backends/rocm_aiter_fa.py (+2/-1); vllm/v1/attention/backends/tree_attn.py (+2/-1); vllm/v1/attention/backends/triton_attn.py (+2/-1); vllm/v1/attention/backends/xformers.py (+2/-1); vllm/core/block/naive_block.py (+1/-1); (+16 more)
LABELS: structured-output, tpu, ready, v1
BODY: Continued #23649 ⏎  ⏎ --- ⏎  ⏎ > [!NOTE] ⏎ > Warnings fixed: ⏎  ⏎ ```bash ⏎ WARNING -  griffe: vllm/core/block/naive_block.py:210: Failed to get 'name: description' pair from 'in whole allocator.' ⏎ WARNING -  griffe: vllm/core/block/prefix_caching_block.py:64: Parameter 'block_ids(Optional[Iterable[int]],' does not appear in the function signature ⏎  ⏎ WARNING -  griffe: vllm/core/scheduler.py:660: Failed to get 'name: description' pair from 'that are currently runni …[truncated]

### L3-082cc07ef8  (L3, 2025-08-27, sha 082cc07ef8f8, PR #23608)
TITLE: DP/EP Support for gpt-oss with deepep-ht comm kernel on SM100 (#23608)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/device_communicators/base_device_communicator.py (+1/-1); vllm/model_executor/layers/fused_moe/config.py (+6/-0); vllm/model_executor/layers/fused_moe/layer.py (+4/-2); vllm/model_executor/layers/fused_moe/trtllm_moe.py (+197/-0); vllm/model_executor/layers/fused_moe/utils.py (+16/-0); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+4/-4); vllm/model_executor/layers/quantization/fp8.py (+1/-0); vllm/model_executor/layers/quantization/modelopt.py (+2/-0); vllm/model_executor/layers/quantization/mxfp4.py (+110/-0); vllm/model_executor/layers/quantization/utils/mxfp4_utils.py (+4/-5); (+1 more)
LABELS: ready, gpt-oss
BODY: Rebased and cleaned up version for #22907 since @varun-sundar-rabindranath is out.  ⏎  ⏎ Model: 120B ⏎  ⏎ | Reasoning Effort | GPQA | AIME25 | ⏎ | :---- | :---- | :---- | ⏎ | Low  |  0.6540404 |  0.5458 | ⏎ | Mid  |  0.71906 | 0.7791 | ⏎ | High  | 0.79356  |  | ⏎  ⏎ Benchmark on Random Datset for DEP 4 ⏎ ```bash ⏎ python benchmark_serving.py --model "openai/gpt-oss-120b" --dataset-name random --ignore-eos --num-prompts 2048 --random-input-len 1000 --random-output-len 10 …[truncated]

### L3-95089607fa  (L3, 2025-08-28, sha 95089607fa30, PR #23819)
TITLE: [Model][gpt-oss] Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE (#23819)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/config.py (+8/-7); vllm/model_executor/layers/fused_moe/layer.py (+4/-4); vllm/model_executor/layers/quantization/mxfp4.py (+2/-4)
LABELS: ready, gpt-oss
BODY: Changes: ⏎  ⏎ - Enable EP for GPT-OSS with FlashInfer trtllm-gen MoE ⏎ - Fix an issue that VLLM_USE_FLASHINFER_MOE_FP4 is checked even when the quant dtype is not nvfp4. ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ Run GPT-OSS-120b with DP+EP on B200x2 ⏎  ⏎ Server command: ⏎  ⏎ ``` ⏎ export VLLM_SKIP_P2P_CHECK=1 ⏎ export VLLM_USE_FLASHINFER_MOE_FP8=1 ⏎ export VLLM_USE_FLASHINFER_MOE_FP4=1 ⏎ export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 ⏎ export VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_ …[truncated]

### L3-a781e84ec2  (L3, 2025-08-28, sha a781e84ec25b, PR #23748)
TITLE: [Perf] Tune configs for triton block fp8 gemm H100/H200 (#23748)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/bench_block_fp8_gemm.py (+113/-0); vllm/model_executor/layers/quantization/utils/configs/N=12288,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json (+146/-0); vllm/model_executor/layers/quantization/utils/configs/N=12288,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json (+146/-0); vllm/model_executor/layers/quantization/utils/configs/N=24576,K=1536,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json (+146/-0); vllm/model_executor/layers/quantization/utils/configs/N=24576,K=1536,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json (+146/-0); vllm/model_executor/layers/quantization/utils/configs/N=24576,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json (+33/-33); vllm/model_executor/layers/quantization/utils/configs/N=24576,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json (+35/-35); vllm/model_executor/layers/quantization/utils/configs/N=32768,K=512,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json (+22/-22); vllm/model_executor/layers/quantization/utils/configs/N=32768,K=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json (+24/-24); vllm/model_executor/layers/quantization/utils/configs/N=36864,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json (+32/-32); (+11 more)
LABELS: performance, ready
BODY: ## Purpose ⏎  ⏎ Retune the triton fp8 block dense gemm configs for modern triton. Also adds a simple benchmark script that doesn't tune. ⏎  ⏎ Mostly improves performance for smaller M, but crucially gives improvement for `N=576,K=7168` ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ### H100 ⏎ <img width="899" height="883" alt="Screenshot 2025-08-27 at 5 30 16 PM" src="https://github.com/user-attachments/assets/a406a911-92bb-465b-aee3-0ac8bd722e73" /> ⏎ <img width="899" heig …[truncated]

### L3-7ffbf27239  (L3, 2025-08-28, sha 7ffbf27239c3, PR #23737)
TITLE: [BugFix][FlashInfer] Fix potential race condition for paged_kv_indptr_cpu (#23737)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+10/-2)
LABELS: v1
BODY: 

### L3-cb293f6a79  (L3, 2025-08-28, sha cb293f6a790d, PR #22628)
TITLE: [V1] Enable prefill optimization for Gemma3n (#22628)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+126/-13); tests/v1/e2e/test_kv_sharing_fast_prefill.py (+0/-57); vllm/config/cache.py (+7/-5); vllm/model_executor/models/gemma3n.py (+354/-65); vllm/model_executor/models/gemma3n_mm.py (+1/-1); vllm/v1/engine/async_llm.py (+7/-0); vllm/v1/worker/gpu_model_runner.py (+53/-43); vllm/v1/worker/tpu_model_runner.py (+28/-11); vllm/v1/worker/utils.py (+7/-33)
LABELS: tpu, speculative-decoding, ready, v1
DEEP_STUDY: deep-study: this PR was reverted by PR 23897 (partial_revert, reason=ci_or_test_failure)
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ This PR adds an option to enable prefill optimization for Gemma3n model with `--kv-sharing-fast-prefill`.  ⏎  ⏎ ### Background ⏎ In You Only Cache Once (https://arxiv.org/abs/2405.05254), self-decoder layers generate KV caches while cross-decoder layers use cross-attention and reuse the shared KV cache. As only self-decoder layers generate KV caches, cross-decoder layers don't n …[truncated]

### L3-186aced5ff  (L3, 2025-08-28, sha 186aced5ffb6, PR #23791)
TITLE: [Kernel] cuda kernels for upcoming decode context parallel feature (#23791)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+247/-0); csrc/cache.h (+16/-1); csrc/torch_bindings.cpp (+15/-0); tests/kernels/attention/test_cache.py (+72/-0); vllm/_custom_ops.py (+24/-0)
BODY: Pre-PR for [#1367](https://github.com/vllm-project/vllm/pull/23734) ⏎  ⏎ Suggestions from @youkaichao : to accelerate the review and merge (especially ci testing), maybe we can split the kernel side changes to a separate PR and get it merged first.

### L3-b668055a11  (L3, 2025-08-28, sha b668055a1140, PR #23862)
TITLE: [V0 Deprecation] Remove V0 Samplers test (#23862)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/samplers/test_sampler.py (+0/-769); tests/samplers/test_seeded_generate.py (+0/-86)
LABELS: ready
BODY: 

### L3-04d1dd7f4a  (L3, 2025-08-28, sha 04d1dd7f4a44, PR #23264)
TITLE: [ROCm][Aiter] Add triton fp8 bmm kernel for mla (#23264)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.common_v1
FILES: vllm/envs.py (+8/-0); vllm/v1/attention/backends/mla/common.py (+96/-12)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎ Replace `torch.bmm` with aiter `batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant` kernel for MLA ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎ [details omitted] ⏎  ⏎ ### Performance test on DeepSeek-R1 with full-cudagraph capture mode ⏎ ``` ⏎ REQUEST_RATES=(1 5 7 9) ⏎ TOTAL_SECONDS=20  ⏎ TP=8 ⏎ OUTPUT_LEN=128 ⏎  ⏎ DATASET_PATH="ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json" ⏎ PYTHON_BENCH_SCRIPT="benchmarks/benchmark_serving.p …[truncated]

### L3-006477e60b  (L3, 2025-08-28, sha 006477e60b49, PR #23847)
TITLE: [ROCm][Fix] Fix rocm build caused by #23791 (#23847)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+0/-1)
LABELS: rocm, ready
BODY: The compiler is rejecting an unused var.

### L3-2554b27baa  (L3, 2025-08-29, sha 2554b27baa58, PR #23434)
TITLE: [V0 Deprecation] Remove pooling model support in V0  (#23434)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/distributed/test_pipeline_parallel.py (+6/-2); tests/entrypoints/llm/test_classify.py (+0/-8); tests/entrypoints/llm/test_encode.py (+0/-8); tests/entrypoints/llm/test_reward.py (+0/-8); tests/entrypoints/llm/test_score.py (+0/-8); tests/entrypoints/offline_mode/test_offline_mode.py (+10/-9); tests/entrypoints/openai/test_embedding.py (+0/-8); tests/entrypoints/openai/test_rerank.py (+0/-8); tests/entrypoints/openai/test_score.py (+0/-9); tests/models/language/pooling/test_embedding.py (+3/-17); (+28 more)
LABELS: frontend, ready, multi-modality
BODY: Continuation of PR https://github.com/vllm-project/vllm/pull/23302 ⏎  ⏎ ## Summary ⏎  ⏎ - drop pooling model runner and related code paths in V0 worker ⏎ - simplify V0 engine to handle only sampling requests ⏎ - add stubs that raise for pooling model entry points ⏎  ⏎ @DarkLight1337 , @noooop , do you remember anything else to remove?

### L3-235c9db8a7  (L3, 2025-08-29, sha 235c9db8a755, PR #22887)
TITLE: [XPU] support data parallel for MoE models on XPU (#22887)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/device_communicators/xpu_communicator.py (+11/-0); vllm/model_executor/layers/fused_moe/layer.py (+2/-0)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ VLLM_WORKER_MULTIPROC_METHOD=spawn   python examples/offline_inference/data_parallel.py --enforce-eager             --model="ibm-research/PowerMoE-3b"             --dp-size=2             --tp-size=2 ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update ⏎  ⏎ --- ⏎ [details omitted]

### L3-b7adf94c4a  (L3, 2025-08-29, sha b7adf94c4a6c, PR #23939)
TITLE: Tuned H100/H200 triton fp8 block configs for fused_qkv_a_proj (#23939)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/bench_block_fp8_gemm.py (+1/-0); benchmarks/kernels/benchmark_w8a8_block_fp8.py (+1/-0); vllm/model_executor/layers/quantization/utils/configs/N=2112,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json (+146/-0); vllm/model_executor/layers/quantization/utils/configs/N=2112,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json (+123/-3)
LABELS: performance
BODY: ## Purpose ⏎  ⏎ ``` ⏎ # Before ⏎ Benchmarking DeepSeek-V3, N=2112 K=7168 ⏎ TFLOP/s comparison (block_size=(128, 128)): ⏎ WARNING 08-29 12:07:57 [fp8_utils.py:581] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at /home/mgoin/code/vllm/vllm/model_executor/layers/quantization/utils/configs/N=2112,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json ⏎ BF16 vs W8A8 Block FP8 GEMMs: ⏎  …[truncated]

### L3-67c14906aa  (L3, 2025-08-29, sha 67c14906aaa4, PR #20358)
TITLE: Update PyTorch to 2.8.0 (#20358)
SOURCES: path_core, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.flex_attention
FILES: CMakeLists.txt (+2/-2); requirements/build.txt (+2/-1); requirements/cpu.txt (+4/-5); requirements/cuda.txt (+5/-5); requirements/rocm-build.txt (+4/-4); requirements/test.in (+3/-3); requirements/test.txt (+18/-18); vllm/v1/attention/backends/flex_attention.py (+3/-2); .buildkite/test-pipeline.yaml (+2/-2); pyproject.toml (+1/-1); (+2 more)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Update vLLM to PyTorch 2.8.0 now that it has been release ⏎  ⏎ ## Test Plan ⏎  ⏎ CI ⏎  ⏎ ## Test Result ⏎  ⏎ There are some failures, I'm trying to evaluate each one to confirm that they are existing failures from main. ⏎  ⏎ * [x] [basic-models-test](https://buildkite.com/vllm/ci/builds/26147#0198806a-1e55-40d7-91ab-562419a94a16): existing failures = https://buildkite.com/vllm/ci/builds/26180 …[truncated]

### L3-fb4983e112  (L3, 2025-08-30, sha fb4983e112a8, PR #23798)
TITLE: [Misc] add reorder_batch AttentionMetadataBuilder (#23798)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+17/-0)
LABELS: ready, v1
BODY: ## Purpose ⏎ Add `reorder_batch` to `AttentionMetadataBuilder`. As `AttentionGroup.metadata_builder` is `AttentionMetadataBuilder`  type.  ⏎ https://github.com/vllm-project/vllm/blob/c8851a47235f5dfd3da3abf6c89453b3bdb41ad1/vllm/v1/worker/utils.py#L129-L133 ⏎  ⏎ Code in cpu_model_runner `_may_reorder_batch` will just use assert to check the `metadata_builder` is a subclass `TorchSDPAMetadataBuilderV1` of `AttentionMetadataBuilder`. `AttentionMetadataBuil …[truncated]

### L3-7be0cb8e9e  (L3, 2025-09-02, sha 7be0cb8e9e48, PR #23148)
TITLE: [XPU][Feature] fp8 online quantization support for XPU (#23148)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/_ipex_ops.py (+55/-1); vllm/model_executor/layers/quantization/fp8.py (+25/-0); vllm/model_executor/layers/quantization/ipex_quant.py (+158/-1); vllm/platforms/xpu.py (+4/-0)
LABELS: ready
BODY: ## Purpose ⏎ This PR enable online fp8 quantization on XPU platform with corresponding `dynamic_scaled_fp8_quant` and grouped gemm kernels. Note that need specify `dtype` with `float16` since `bfloat16` has poor perf in current used ipex-2.8 whl used in upstream. ⏎  ⏎ ## Test Plan ⏎ Verified on several models with `--quantization fp8` added. ⏎ ``` ⏎ VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python3 examples/offline_inference/basic/ge …[truncated]

### L3-1bd007f234  (L3, 2025-09-02, sha 1bd007f23476, PR #24071)
TITLE: fix some typos (#24071)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flash_attn.py (+1/-1); vllm/v1/attention/backends/flashinfer.py (+1/-1); benchmarks/benchmark_block_pool.py (+1/-1); benchmarks/benchmark_ngram_proposer.py (+1/-1); csrc/quantization/cutlass_w4a8/w4a8_mm_entry.cu (+1/-1); docs/configuration/optimization.md (+2/-2); docs/design/io_processor_plugins.md (+1/-1); examples/offline_inference/prithvi_geospatial_mae_io_processor.py (+1/-1); examples/online_serving/prithvi_geospatial_mae.py (+1/-1); tests/compile/piecewise/test_multiple_graphs.py (+1/-1); (+22 more)
LABELS: documentation, performance, frontend, ready, v1, multi-modality, llama
BODY: ## Purpose ⏎ fix some typos ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-d7e1e59972  (L3, 2025-09-02, sha d7e1e599724e, PR #24093)
TITLE: [Doc]: fix typos in Python comments (#24093)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+1/-1); tests/core/test_scheduler.py (+1/-1); tests/entrypoints/openai/correctness/test_transcription_api_correctness.py (+1/-1); tests/entrypoints/openai/test_return_token_ids.py (+1/-1); tests/entrypoints/openai/test_serving_chat.py (+1/-1); tests/kernels/utils.py (+1/-1); tests/multimodal/test_utils.py (+2/-2); tests/v1/e2e/test_spec_decode.py (+1/-1); tests/v1/kv_connector/unit/test_remote_decode_lifecycle.py (+2/-2); tests/v1/spec_decode/test_tree_attention.py (+2/-2); (+5 more)
LABELS: structured-output, tpu, speculative-decoding, ready, v1, multi-modality
BODY: ## Purpose ⏎  ⏎ Improve quality of documentation for Python by eliminating typos: see commit diffs for details. ⏎  ⏎ ## Test Plan ⏎  ⏎ N/A ⏎  ⏎ ## Test Result ⏎  ⏎ N/A

### L3-457e471971  (L3, 2025-09-02, sha 457e4719710e, PR #23692)
TITLE: [AMD][Kernel][Bugfix] Cast offsets tensor bn to tl.int64 to avoid GPU segfault (#23692)
SOURCES: path_core, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/prefix_prefill.py (+3/-3)
LABELS: rocm, ready
DEEP_STUDY: deep-study correctness case vllm:457e471971: class=memory_safety_oob; symptom=illegal_memory_access; introducing=unknown
BODY: The tensor loaded into `bn ` is multiplied by `stride_k_cache_bs `in the `_fwd_kernel `in `prefix_prefill.py` and produces an integer overflow resulting in negative offsets which result in a GPU segfault.  Changing `stride_k_cache_bs  `to be `tl.int64` in the function signature did **not** work.  Casting the `bn` tensor to `tl.int64` fixes the problem.  I added some additional casts into `_fwd_kernel_flash_attn_v2 `and `_fwd_kernel_alibi` as well …[truncated]

### L3-e81d4e69c1  (L3, 2025-09-03, sha e81d4e69c16c, PR #24070)
TITLE: [Misc] Add check for dual_chunk_attention (#24070)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/config/__init__.py (+6/-1)
LABELS: ready
ISSUES: #24048 [Bug]: TypeError: FlashAttentionImpl.__init__() got an unexpected keyword argument 'layer_idx' in Qwen/Qwen2.5-14B-Instruct-1M
BODY: ## Purpose ⏎ FIX https://github.com/vllm-project/vllm/issues/24048 ⏎ Add an early validation check for the `DUAL_CHUNK_FLASH_ATTN` environment variable when using the `Qwen2.5-14B-Instruct-1M model`. This preemptive check should trigger a clear error on startup rather than allowing the code to fail later and obscurely within the attention module. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-a742322092  (L3, 2025-09-03, sha a74232209237, PR #23289)
TITLE: [Attention] Blackwell FP8 MLA support with CUTLASS_MLA backend (#23289)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.cutlass_v1_backend, L3.platform.cuda_selection
FILES: csrc/attention/mla/sm100_cutlass_mla_kernel.cu (+8/-8); vllm/platforms/cuda.py (+2/-2); vllm/v1/attention/backends/mla/cutlass_mla.py (+9/-14); tests/kernels/test_cutlass_mla_decode.py (+167/-83)
LABELS: ready, v1, deepseek
BODY: ## Purpose ⏎ Enable FP8 KV cache support on Blackwell in the CUTLASS_MLA backend. ⏎  ⏎ ## Test Plan ⏎  ⏎ ### Correctness ⏎ `VLLM_ATTENTION_BACKEND=CUTLASS_MLA lm_eval --model vllm --model_args '{"pretrained": "deepseek-ai/DeepSeek-V2-Lite-Chat", "trust_remote_code": true, "kv_cache_dtype": "fp8"}' --tasks gsm8k --batch_size auto` ⏎  ⏎ ### Performance ⏎ V2 Lite ⏎ `VLLM_ATTENTION_BACKEND=CUTLASS_MLA vllm bench throughput --model=deepseek-ai/DeepSeek-V2-Lite-Chat --dat …[truncated]

### L3-6d80ae83e1  (L3, 2025-09-03, sha 6d80ae83e145, PR #23424)
TITLE: [Bugfix] Fixing division by zero in triton_attn if query_heads/kv_heads > 16  (#23424)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+2/-1)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Currently, the Triton unified attention creates a division by zero for models that have more than 16 query heads per kv-head (e.g. MQA models). This one-line PR fixes it.  ⏎  ⏎ CC: @jvlunteren @SageMoore @tdoublep  ⏎  ⏎ ## Test Plan ⏎  ⏎ Comparing correctness for Triton attention backend for an affected model with the correctness using flash attention backend.  ⏎  ⏎ ## Test Result ⏎  ⏎ baseline, on an H100 ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=FLASH_ATTN_VLLM_V1 lm …[truncated]

### L3-f0c503f66e  (L3, 2025-09-03, sha f0c503f66e2f, PR #20189)
TITLE: [Nixl] Heterogeneous TP support FlashInfer (#20189)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py (+53/-9)
LABELS: ready
BODY: This PR enables the use of FlashInfer in a heterogeneous TP setting when using NixlConnector, particularly important for Blackwell systems since they will default to FlashInfer. ⏎  ⏎ The main difference from FA is that the cache layout goes from `[2, num_blocks, HND]`  to `[num_blocks, 2, HND]` where 2 is K/V.  ⏎ With homogeneous TP, this layout change has no particular implication: quite the contrary, we can actually read both K and V in a single mess …[truncated]

### L3-3efb9f4d95  (L3, 2025-09-04, sha 3efb9f4d95bf, PR #23332)
TITLE: [Attention][Platform] Refactor MLA to support Custom Op (#23332)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mla.py (+158/-0); vllm/model_executor/models/deepseek_v2.py (+28/-58)
LABELS: ready, deepseek
BODY: ## Purpose ⏎ MLA (Multi-Head Latent Attention) is a complex layer with significant optimization potential, and different hardware backends may require distinct optimization approaches—such as operator fusion, multi-streaming, etc. ⏎  ⏎ The current abstraction in vLLM struggles to accommodate scenarios that require integrating large-grained fused operators from different platforms. For example, on the Ascend platform, we need to integrate a fused operat …[truncated]

### L3-402759d472  (L3, 2025-09-04, sha 402759d4727d, PR #14258)
TITLE: [Attention] FlashAttn MLA (#14258)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L3.xformers.v1_backend, L3.flash_attn.upstream_pip, L3.flash_attn.fork_build, L3.flash_attn.fa_utils, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.flashattn, L3.mla.rocm_aiter, L3.platform.cuda_selection
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); docker/Dockerfile (+1/-1); vllm/attention/utils/fa_utils.py (+13/-0); vllm/engine/arg_utils.py (+2/-0); vllm/envs.py (+1/-0); vllm/platforms/cuda.py (+40/-28); vllm/platforms/interface.py (+3/-2); vllm/v1/attention/backends/flashinfer.py (+2/-1); vllm/v1/attention/backends/mla/common.py (+12/-5); vllm/v1/attention/backends/mla/flashattn_mla.py (+189/-0); (+12 more)
LABELS: rocm, ready, ci/build, v1
BODY: Use latest FlashAttention code to compute decode MQA in MLA ⏎  ⏎ ``` ⏎ VLLM_USE_V1=0 lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384 --task gsm8k --num_fewshot 5  --batch_size auto ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|--- …[truncated]

### L3-83609ca91d  (L3, 2025-09-04, sha 83609ca91d42, PR #24173)
TITLE: [Doc]: fix typos in Python comments (#24173)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.triton_v0, L3.mla.common_v1
FILES: vllm/attention/backends/mla/common.py (+1/-1); vllm/v1/attention/backends/mla/common.py (+2/-2); benchmarks/benchmark_dataset.py (+1/-1); benchmarks/kernels/benchmark_lora.py (+1/-1); benchmarks/multi_turn/benchmark_serving_multi_turn.py (+1/-1); examples/offline_inference/audio_language.py (+1/-1); tests/models/multimodal/generation/vlm_utils/builders.py (+1/-1); tests/models/multimodal/generation/vlm_utils/case_filtering.py (+1/-1); vllm/engine/async_llm_engine.py (+1/-1); vllm/engine/multiprocessing/client.py (+1/-1); (+2 more)
LABELS: documentation, performance, ready, v1, multi-modality
BODY: ## Purpose ⏎  ⏎ Improve project doc by fixing typos ⏎  ⏎ ## Test Plan ⏎  ⏎ N/A ⏎  ⏎ ## Test Result ⏎  ⏎ N/A

### L3-78336a0c3e  (L3, 2025-09-04, sha 78336a0c3ee4, PR #24086)
TITLE: Upgrade FlashInfer to v0.3.0 (#24086)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); setup.py (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ Mainly to get the GPT-OSS MXFP4 trtllm-gen MoE autotuning and the bug fix in: https://github.com/flashinfer-ai/flashinfer/pull/1573 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-16ded21eeb  (L3, 2025-09-04, sha 16ded21eeb57, PR #24149)
TITLE: [XPU] support Triton Attention backend on Intel GPU (#24149)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.triton.v1_backend
FILES: vllm/attention/ops/paged_attn.py (+6/-1); vllm/platforms/xpu.py (+26/-2); vllm/v1/attention/backends/triton_attn.py (+10/-5); .buildkite/scripts/hardware_ci/run-xpu-test.sh (+5/-4); vllm/_ipex_ops.py (+2/-3)
LABELS: ready, ci/build, v1
BODY: ## Purpose ⏎ this PR enable Triton attention backend on Intel GPU. Meanwhile, add XPU fp8 kv cache support in triton attention. ⏎ Please be aware that triton kernel performance need further tuning on Intel GPU. ⏎  ⏎ ## Test Plan ⏎ add a test in CI ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-cee182b297  (L3, 2025-09-05, sha cee182b2970b, PR #23569)
TITLE: [Perf][V1] Fully overlap model execution (#23569)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/executor/multiproc_executor.py (+45/-5); vllm/v1/outputs.py (+15/-0); vllm/v1/worker/gpu_input_batch.py (+5/-0); vllm/v1/worker/gpu_model_runner.py (+182/-21); vllm/v1/worker/gpu_worker.py (+5/-5)
LABELS: performance, ready, v1
BODY: ## Purpose ⏎  ⏎ This PR allows the model runner to function asynchronously when using async scheduling. This allows full overlap of the cpu operations (including prepare_inputs) and the model forward pass. This diff is functional and does not support speculative decoding, PP, or guided decoding.  ⏎  ⏎ Expected speedup is 5-10% over the current async scheduling. ⏎  ⏎ This PR is pending some light refactoring (see inline comments) and testing but is otherwise  …[truncated]

### L3-35bf193864  (L3, 2025-09-05, sha 35bf19386489, PR #24294)
TITLE: [Doc]: fix typos in Python comments (#24294)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+1/-1); csrc/quantization/machete/generate.py (+1/-1); docs/getting_started/installation/cpu.md (+1/-1); tests/models/multimodal/generation/vlm_utils/core.py (+1/-1); vllm/distributed/device_communicators/custom_all_reduce.py (+2/-2); vllm/entrypoints/openai/tool_parsers/internlm2_tool_parser.py (+3/-3); vllm/envs.py (+1/-1); vllm/model_executor/layers/fused_moe/fused_moe.py (+1/-1); vllm/model_executor/layers/fused_moe/layer.py (+2/-2); vllm/model_executor/layers/quantization/gptq_marlin.py (+2/-2); (+2 more)
LABELS: documentation, frontend, ready, v1, multi-modality, tool-calling
BODY: ## Purpose ⏎  ⏎ Improve code documentation quality by fixing typos in comments: see commit diffs for details. ⏎  ⏎ ## Test Plan ⏎  ⏎ N/A ⏎  ⏎ ## Test Result ⏎  ⏎ N/A

### L3-ac201a0eaf  (L3, 2025-09-06, sha ac201a0eaf2a, PR #23734)
TITLE: [Feature] Support Decode Context Parallel (DCP) for MLA (#23734)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.mla.common_v1, L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.flashattn, L3.mla.rocm_aiter
FILES: csrc/cache_kernels.cu (+0/-103); csrc/torch_bindings.cpp (+0/-10); vllm/_custom_ops.py (+0/-14); vllm/attention/ops/common.py (+139/-0); vllm/attention/ops/flashmla.py (+3/-1); vllm/config/parallel.py (+5/-0); vllm/engine/arg_utils.py (+17/-0); vllm/v1/attention/backends/mla/common.py (+309/-25); vllm/v1/attention/backends/mla/cutlass_mla.py (+11/-7); vllm/v1/attention/backends/mla/flashattn_mla.py (+9/-4); (+17 more)
LABELS: rocm, ready, ci/build, v1
BODY: This PR adds Decode Context Parallel (DCP) support for MLA inference, fully compatible with chunked prefill and APC.  ⏎  ⏎ **You can enable DCP with `--decode-context-parallel-size/-dcp xxx` (only support flashmla backend now), and tp_size needs to be divisible by dcp_size**, because the world size does not change by dcp, it simply reuse the GPUs of TP group, and split one TP group into tp_size//dcp_size DCP groups. e.g. ⏎ ``` ⏎ with -tp 8 -dcp 8 , we us …[truncated]

### L3-4172235ab7  (L3, 2025-09-06, sha 4172235ab78b, PR #21159)
TITLE: [V0 deprecation] Deprecate V0 Neuron backend (#21159)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen
FILES: vllm/attention/ops/nki_flash_attn.py (+0/-903); .buildkite/release-pipeline.yaml (+0/-16); .buildkite/scripts/hardware_ci/run-neuron-test.sh (+0/-64); MANIFEST.in (+0/-1); docker/Dockerfile.neuron (+0/-56); examples/offline_inference/neuron.py (+0/-49); examples/offline_inference/neuron_eagle.py (+0/-61); examples/offline_inference/neuron_int8_quantization.py (+0/-63); examples/offline_inference/neuron_multimodal.py (+0/-110); examples/offline_inference/neuron_speculation.py (+0/-64); (+36 more)
LABELS: documentation, speculative-decoding, ready, ci/build
BODY: In vLLM V1, Neuron will be supported as a plugin. ⏎  ⏎ See https://github.com/vllm-project/vllm/issues/21082

### L3-558f0907dc  (L3, 2025-09-07, sha 558f0907dc67, PR #24372)
TITLE: [attention][DCP] use AttentionImpl.need_to_return_lse_for_decode (#24372)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+26/-0); vllm/v1/attention/backends/mla/common.py (+0/-4); vllm/v1/attention/backends/mla/flashmla.py (+2/-0); vllm/v1/worker/gpu_model_runner.py (+10/-5)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Expose a new attribute to avoid hardcoded conditions. ⏎  ⏎ Might be cleaner than https://github.com/vllm-project/vllm/pull/24369 ? ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-81c53ef55c  (L3, 2025-09-07, sha 81c53ef55c01, PR #24378)
TITLE: [Misc] collect flashinfer version in collect_env.py (#24378)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/collect_env.py (+2/-0)
LABELS: ready
BODY: ## Purpose ⏎ While working on repro in https://github.com/yeqcharlotte/vllm/pull/3, notice flashinfer version is not collected 😂 ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ python vllm/collect_env.py  ⏎ ``` ⏎  ⏎ ## Test Result ⏎ https://gist.github.com/yeqcharlotte/edbbf13b07a3e26e05ef14519641ca6e ⏎ ``` ⏎ ... ⏎ ============================== ⏎ Versions of relevant libraries ⏎ ============================== ⏎ [pip3] efficientnet_pytorch==0.7.1 ⏎ [pip3] flashinfer-python==0.3.1 ⏎ ``` ⏎  ⏎ --- ⏎ [details o …[truncated]

### L3-f4962a6d55  (L3, 2025-09-08, sha f4962a6d55a3, PR #24417)
TITLE: [Doc]: fix typos in Python comments (#24417)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.triton_v0
FILES: vllm/attention/backends/mla/common.py (+1/-1); examples/offline_inference/chat_with_tools.py (+1/-1); vllm/config/__init__.py (+1/-1); vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py (+1/-1); vllm/engine/arg_utils.py (+1/-1); vllm/engine/multiprocessing/client.py (+1/-1); vllm/entrypoints/openai/cli_args.py (+1/-1); vllm/entrypoints/openai/tool_parsers/llama4_pythonic_tool_parser.py (+1/-1); vllm/entrypoints/openai/tool_parsers/mistral_tool_parser.py (+1/-1); vllm/model_executor/layers/fused_moe/modular_kernel.py (+1/-1); (+2 more)
LABELS: documentation, frontend, v1, tool-calling, llama
BODY: ## Purpose ⏎  ⏎ Improve quality by removing typos: see commit diffs for exact details ⏎  ⏎ ## Test Plan ⏎  ⏎ N/A ⏎  ⏎ ## Test Result ⏎  ⏎ N/A ⏎  ⏎ --- ⏎ [details omitted]

### L3-86173ad593  (L3, 2025-09-08, sha 86173ad593ff, PR #24385)
TITLE: [Kernel] Support decode context parallelism on Blackwell with CUTLASS MLA (#24385)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.mla.cutlass_v1_backend
FILES: csrc/attention/mla/sm100_cutlass_mla_kernel.cu (+12/-5); csrc/torch_bindings.cpp (+4/-4); vllm/_custom_ops.py (+3/-3); vllm/v1/attention/backends/mla/cutlass_mla.py (+22/-10); tests/kernels/test_cutlass_mla_decode.py (+22/-10)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ This PR supports decode context parallelism with CUTLASS MLA kernels on GB200 ⏎  ⏎ credits to https://github.com/vllm-project/vllm/pull/22789 from @LucasWilkinson, and @youkaichao  ⏎  ⏎ ## Test Plan ⏎  ⏎ pytest -v -s tests/distributed/test_context_parallel.py ⏎  ⏎ note: on GB200, needs to modify this UT to use only two GPUs. This can be added later or in a follow-up PR. ⏎  ⏎ ## Test Result ⏎  ⏎ both `-tp 2 -dcp 2` and `-tp 2` work ⏎  ⏎ --- ⏎ [details omitted]

### L3-43d9ad03ba  (L3, 2025-09-08, sha 43d9ad03ba85, PR #23928)
TITLE: [Model loader]: support multi-thread model weight loading (#23928)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/model_loader/default_loader.py (+41/-12); vllm/model_executor/model_loader/weight_utils.py (+64/-0)
LABELS: ready
BODY: ## Purpose ⏎ This pull request introduces support for multi-threaded model weight loading in the `DefaultModelLoader`, allowing faster initialization of models by parallelizing the loading of safetensor and pt/bin weight files. It also adds configuration options to control threading behavior and validates extra config keys. ⏎  ⏎ The code was modified from the [Sglang's PR].(https://github.com/sgl-project/sglang/pull/7277)  ⏎  ⏎ ### Multi-threaded weight lo …[truncated]

### L3-bba1042c6f  (L3, 2025-09-08, sha bba1042c6fcb, PR #23647)
TITLE: [Flashinfer] Support Flashinfer TRTLLM FP8-qkv BF16/FP16-out Attention Kernel (#23647)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/compilation/fusion_attn.py (+4/-2); vllm/v1/attention/backends/flashinfer.py (+4/-9); benchmarks/kernels/benchmark_trtllm_decode_attention.py (+1/-0); benchmarks/kernels/benchmark_trtllm_prefill_attention.py (+1/-0); tests/kernels/attention/test_flashinfer_trtllm_attention.py (+12/-0)
LABELS: performance, ready, v1
BODY: ## Purpose ⏎ Support Flashinfer TRTLLM FP8-qkv BF16/FP16-out Attention Kernel. ⏎ After this PR, `Flashinfer + kv_cache_dtype=fp8` will always quantize query to fp8 and use TRTLLM attn kernel(support FP8-qkv BF16/FP16/FP8/NVFP4-out). ⏎ Note: This requires Flashinfer 0.3.0 to land ⏎  ⏎ ## Test Plan && Test Result ⏎ **Kernel functional**: ⏎ `tests/kernels/attention/test_flashinfer_trtllm_attention.py` ⏎ ``` ⏎ ===== 112 passed, 8 skipped in 15.66s ===== ⏎ ``` ⏎  ⏎ **Kernel  …[truncated]

### L3-620db1fc58  (L3, 2025-09-08, sha 620db1fc5879, PR #23958)
TITLE: [Attention] FlashAttention MLA cudagraph support (#23958)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.flashattn, L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/common.py (+14/-8); vllm/v1/attention/backends/mla/flashattn_mla.py (+71/-6); vllm/v1/attention/backends/mla/flashmla.py (+6/-5); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+6/-4); tests/compile/piecewise/test_full_cudagraph.py (+11/-1); tests/v1/attention/test_mla_backends.py (+0/-5); tests/v1/cudagraph/test_cudagraph_mode.py (+10/-0)
LABELS: rocm, ready, ci/build, v1
BODY: ## Purpose ⏎  ⏎ Add full cudagraph support in FlashAttention MLA backend. ⏎  ⏎ ## Test Plan ⏎ `VLLM_ATTENTION_BACKEND=FLASH_ATTN_MLA vllm bench throughput --model=deepseek-ai/DeepSeek-V2-Lite-Chat --dataset-name=random --input-len=128 --output-len=128 --num-prompts=10 --kv-cache-dtype=auto --compilation-config='{"cudagraph_mode": "full_decode_only"}'` ⏎  ⏎ ## Test Result ⏎ Functional ⏎  ⏎ --- ⏎ [details omitted]

### L3-1c63a16b65  (L3, 2025-09-09, sha 1c63a16b653d, PR #24128)
TITLE: [Core] Run garbage collector after CUDA graph capture to fix throughput regression (#24128)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_model_runner.py (+1/-0)
LABELS: ready, v1
BODY: ## Summary ⏎  ⏎ We identified a regression in throughput caused by #21146 in some tests (most noticeably when using small input & output lengths). Specifically, on 8xMI300X: ⏎ ``` ⏎ VLLM_USE_V1=1 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 python3 vllm/benchmarks/benchmark_throughput.py --model /models/Llama-3.1-70B-Instruct-FP8-KV -tp 8 --num-prompts 1024 --kv-cache-dtype "fp8" --input-len 128 --output-len 128 --dtype float16  --max-num-seqs 1024 --gpu-memor …[truncated]

### L3-b9a1c4c8a2  (L3, 2025-09-09, sha b9a1c4c8a243, PR #24279)
TITLE: [ROCm][CI/Build] Sync ROCm dockerfiles with the ROCm fork (#24279)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm (+3/-1); docker/Dockerfile.rocm_base (+30/-32)
LABELS: rocm, ready, ci/build
BODY: Bringing dockerfiles in sync with the ROCm fork, to match what is used to build rocm/vllm-dev:base; rocm/vllm-dev:nightly and rocm/vllm images

### L3-15de5ff9ea  (L3, 2025-09-09, sha 15de5ff9ea30, PR #24521)
TITLE: [Feature] Disallow FlashMLA on Blackwell (#24521)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter
FILES: vllm/attention/backends/flashmla.py (+11/-0); vllm/v1/attention/backends/mla/flashmla.py (+11/-0)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Fixing https://github.com/vllm-project/vllm/issues/24513 ⏎  ⏎ FlashMLA is poorly supported on Blackwell, see ⏎ https://github.com/deepseek-ai/FlashMLA/issues/83 ⏎ for more details.

### L3-dc625ea6b8  (L3, 2025-09-09, sha dc625ea6b8f8, PR #24474)
TITLE: [Perf] Convert np array to torch tensor to index into block table for attn chunking (#24474)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+8/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ upstream pytorch change https://github.com/pytorch/pytorch/pull/160256 causes perf regression when expanding torch.tensor with numpy arrays. This affects perf of `make_local_attention_virtual_batches ` which is used to create virtual batches for chunked local attn (llama4). ⏎  ⏎ On local benchmarks I see the below line take 10+ms, from <1ms before: ⏎  ⏎ ``` ⏎ block_table_local = block_table[batch_indices, block_indices]\ ⏎         .view(virtual_b …[truncated]

### L3-7e7db04310  (L3, 2025-09-09, sha 7e7db0431092, PR #24536)
TITLE: [CI] Retry flaky fp8 cutlass mla tests (#24536)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_cutlass_mla_decode.py (+7/-1)
LABELS: ready, ci/build
BODY: Such as https://buildkite.com/vllm/ci/builds/29987#01992f55-5a98-42e8-9589-751e26e35165.

### L3-73e688cb79  (L3, 2025-09-09, sha 73e688cb7978, PR #24275)
TITLE: [ROCm][Feature] Enable Pipeline Parallelism with Ray Compiled Graph on ROCm (#24275)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm (+1/-0); requirements/rocm.txt (+1/-1); vllm/utils/__init__.py (+14/-2)
LABELS: rocm, ready, ci/build
BODY: To enable pipeline parallelism with Ray Compiled Graph. ⏎  ⏎ - Add RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 in the docker build file to avoid set_device error. ⏎ - set the new stream created for rocm to torch's current stream. ⏎ - bump Ray version to 2.48.0

### L3-1aa427fdc1  (L3, 2025-09-10, sha 1aa427fdc1d4, PR #24518)
TITLE: [Kernels] Add Flash Linear Attention Kernels (#24518)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tools/mypy.sh (+1/-1); vllm/model_executor/layers/fla/__init__.py (+8/-0); vllm/model_executor/layers/fla/ops/__init__.py (+17/-0); vllm/model_executor/layers/fla/ops/chunk.py (+225/-0); vllm/model_executor/layers/fla/ops/chunk_delta_h.py (+289/-0); vllm/model_executor/layers/fla/ops/chunk_o.py (+176/-0); vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py (+138/-0); vllm/model_executor/layers/fla/ops/cumsum.py (+226/-0); vllm/model_executor/layers/fla/ops/fused_recurrent.py (+366/-0); vllm/model_executor/layers/fla/ops/index.py (+39/-0); (+7 more)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ These kernels are adapted from https://github.com/fla-org/flash-linear-attention and customized for some incoming models. ⏎  ⏎ ## Test Plan ⏎  ⏎ Linters pass and these kernels can work in cuda environments. It will not error when the platform does not support triton. ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-3c2156b3af  (L3, 2025-09-10, sha 3c2156b3af0e, PR #24129)
TITLE: [Hardware][Apple-CPU] Enable native bfloat16 on Apple Silicon (M2 and later) (#24129)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/platforms/cpu.py (+6/-6)
LABELS: ready
BODY: ### Purpose ⏎ This PR adds bfloat16 (bf16) support for Apple M-series CPUs, based on a sysctl capability check. ⏎ According to the [LLVM commit](https://github.com/llvm/llvm-project/commit/677da09d0259d7530d32e85cb561bee15f0066e2?utm_source=chatgpt.com#diff-9c6b94ec4c0b662b700d94091204d00cea9ad766ef596ca67c0f5b84d957aff8), bf16 instructions are supported on Apple Silicon CPUs starting with the M2 generation. ⏎ ### Details ⏎ The necessary `cpu_extension.c …[truncated]

### L3-6cbd41909e  (L3, 2025-09-10, sha 6cbd41909eda, PR #23978)
TITLE: Feature/vit attention unification# 23880 (#23978)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+19/-6); tests/kernels/attention/test_mha_attn.py (+17/-5); vllm/model_executor/models/idefics2_vision_model.py (+3/-0); vllm/model_executor/models/intern_vit.py (+6/-5); vllm/model_executor/models/interns1_vit.py (+7/-10); vllm/model_executor/models/mllama.py (+9/-15); vllm/model_executor/models/step3_vl.py (+7/-16); vllm/model_executor/models/vision.py (+1/-1); vllm/platforms/interface.py (+1/-0)
LABELS: performance, ready, llama
BODY: Purpose ⏎  ⏎ This PR implements a unified VisionAttention interface that automatically selects the optimal attention backend based on hardware capabilities, compute requirements, and model configuration. This addresses GitHub issue #23880 by providing a simple, consistent API for Vision Transformer attention computation, eliminating the need for developers to manually implement complex attention logic for each model. ⏎  ⏎ Key Features: ⏎ Automatic backend  …[truncated]

### L3-a0933c3bd6  (L3, 2025-09-10, sha a0933c3bd67c, PR #24577)
TITLE: [Bugfix] Enable FP8 KV cache for FlashInfer and Triton backend on non-sm100 GPUs (#24577)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+4/-0); vllm/v1/attention/backends/flashinfer.py (+5/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ FlashInfer supports FP8 KV-cache on GPUs that use FA2 backend e.g. sm80, sm89, sm120. Currently I only have access to sm120 GPU so I can only confirm that it works. ⏎  ⏎ Triton attention backend also support FP8. ⏎  ⏎ Changes ⏎ - When attention backend is FlashInfer or Triton, `is_kv_cache_dtype_supported()` returns True ⏎ - In FlashInfer attention, only enable FP8 query when TRT-LLM attention is supported on the current GPU - see https://github. …[truncated]

### L3-37e8182bfe  (L3, 2025-09-10, sha 37e8182bfe15, PR #21088)
TITLE: [v1] Add Whisper model support (encoder-decoder) (#21088)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.xformers.v1_backend, L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.aiter_fa, L3.dispatch.abstract_interface, L3.flex_attention, L3.tree_attention
FILES: vllm/attention/layers/cross_attention.py (+160/-0); vllm/v1/attention/backends/cpu_attn.py (+2/-2); vllm/v1/attention/backends/flash_attn.py (+1/-2); vllm/v1/attention/backends/flashinfer.py (+1/-3); vllm/v1/attention/backends/flex_attention.py (+2/-1); vllm/v1/attention/backends/rocm_aiter_fa.py (+2/-3); vllm/v1/attention/backends/tree_attn.py (+2/-1); vllm/v1/attention/backends/triton_attn.py (+2/-2); vllm/v1/attention/backends/utils.py (+6/-0); vllm/v1/attention/backends/xformers.py (+2/-1); (+21 more)
LABELS: documentation, speculative-decoding, ready, ci/build, v1, multi-modality
ISSUES: #12761 [RFC]: Initial support for multi-model models using cross attention in V1
BODY: v1: Add Whisper encoder-decoder model support ⏎  ⏎ Implements Whisper mdoel support in the V1 engine. Key changes include: ⏎  ⏎ - Add encoder-decoder architecture support with cross-attention KV cache management ⏎ - Add CrossAttentionManager and CrossAttentionSpec for encoder-decoder KV cache ⏎ - Update scheduler to handle cross-attention block allocation and disable prefix caching ⏎ - Modify GPU model runner for encoder input processing and attention metadata …[truncated]

### L3-9a161307f5  (L3, 2025-09-10, sha 9a161307f5f0, PR #19767)
TITLE: [torch.compile][ROCm][V1] Enable attention output FP8 fusion for V1 attention backends (#19767)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.prefix_prefill, L3.triton.chunked_prefill_paged_decode, L3.triton.unified_attention, L3.triton.v1_backend
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+16/-2); vllm/attention/ops/prefix_prefill.py (+13/-1); vllm/attention/ops/triton_unified_attention.py (+81/-58); vllm/compilation/backends.py (+2/-1); vllm/compilation/fusion_attn.py (+22/-9); vllm/v1/attention/backends/triton_attn.py (+9/-3); tests/compile/test_fusion_attn.py (+87/-42); vllm/model_executor/layers/quantization/utils/w8a8_utils.py (+19/-19)
LABELS: rocm, ready, torch.compile, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ An extension of https://github.com/vllm-project/vllm/pull/16756 for V1 unified attention (and its fallback split attention) backend. ⏎ Requires https://github.com/vllm-project/vllm/pull/19158 (full graph capture for this backend) to actually perform the fusion. ⏎  ⏎ Fixes the fusion path to support torch.zeros initialized output tensor (used to be torch.empty before https://gith …[truncated]

### L3-b5e383cd8b  (L3, 2025-09-10, sha b5e383cd8b62, PR #24482)
TITLE: [gpt-oss] raise error for flashinfer backend without trtllm (#24482)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+10/-2)
LABELS: ready, v1, gpt-oss
BODY: ## Purpose ⏎ The attention sink is not integrated into flashinfer backend yet. Raise an error. ⏎ ## Test Plan ⏎ ` VLLM_ATTENTION_BACKEND=FLASHINFER vllm serve openai/gpt-oss-20b` on hopper ⏎ ## Test Result ⏎ raise the newly added error. ⏎ --- ⏎ [details omitted]

### L3-fba7856581  (L3, 2025-09-10, sha fba785658170, PR #23439)
TITLE: [Perf] Warmup FlashInfer attention during startup (#23439)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+0/-16); vllm/v1/worker/gpu_model_runner.py (+28/-3); vllm/model_executor/warmup/kernel_warmup.py (+27/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Make sure to run a dummy run with mixed batch attention included when flashinfer is used to trigger JIT/download/load from disk before real requests. Without this, flashinfer attention kernels are loaded on the first real inference which is unexpected for users. ⏎  ⏎ This required adding a flag to make a mixed decode/prefill batch to dummy run. I tried to keep it minimal, producing decode tokens out of the first half of `num_tokens` and o …[truncated]

### L3-dcb28a332b  (L3, 2025-09-10, sha dcb28a332bfb, PR #21078)
TITLE: [Kernel] Flashinfer MLA (trtllm-gen) decode kernel integration (#21078)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashinfer, L3.platform.cuda_selection
FILES: vllm/engine/arg_utils.py (+1/-0); vllm/platforms/cuda.py (+15/-0); vllm/platforms/interface.py (+1/-0); vllm/v1/attention/backends/mla/common.py (+3/-0); vllm/v1/attention/backends/mla/flashinfer_mla.py (+110/-0); .buildkite/test-pipeline.yaml (+2/-1); tests/kernels/attention/test_cutlass_mla_decode.py (+0/-0); tests/kernels/attention/test_flashinfer_mla_decode.py (+123/-0)
LABELS: ready, ci/build, v1
BODY: E2E Benchmark: ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=FLASHINFER_MLA vllm serve deepseek-ai/DeepSeek-R1 --block-size 32 --tensor-parallel-size 8 --max-num-seqs 512 ⏎  ⏎ python benchmarks/benchmark_serving.py  --model deepseek-ai/DeepSeek-R1 --dataset-name random --ignore-eos --num-prompts 1024 --max-concurrency 512 --random-input-len 1024 --random-output-len 2048 ⏎ ``` ⏎  ⏎ TRTLLM-gen (via Flashinfer) MLA: ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Success …[truncated]

### L3-0ae43dbf8c  (L3, 2025-09-10, sha 0ae43dbf8cb2, PR #24453)
TITLE: [Attention] add DCP support for FLASH_ATTN_MLA backend (#24453)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.mla.flashattn
FILES: vllm/v1/attention/backends/mla/flashattn_mla.py (+16/-2); vllm/v1/worker/gpu_model_runner.py (+3/-0)
LABELS: v1
BODY: ## Purpose ⏎  ⏎ Fix FA MLA return and add [DCP support](https://github.com/vllm-project/vllm/pull/23734) for FA MLA  ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=FLASH_ATTN_MLA chg run -g 4 -- pytest tests/distributed/test_context_parallel.py -s ⏎ ``` ⏎ lm_eval results ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=FLASH_ATTN_MLA chg run -g 2 --  vllm serve --model="deepseek-ai/DeepSeek-V2-Lite-Chat" --trust-remote-code -tp 2 -dcp 2 --port 3331 ⏎  ⏎ lm_eval --mode …[truncated]

### L3-29799ddacc  (L3, 2025-09-10, sha 29799ddacc92, PR #24623)
TITLE: [Bugfix] Add missing VIT backend dispatch on CPU (#24623)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+2/-1)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ - Add missing VIT backend dispatch on CPU to  fix failed tests ⏎  ⏎ ## Test Plan ⏎  ⏎ CI tests ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-41329a0ff9  (L3, 2025-09-10, sha 41329a0ff95e, PR #24469)
TITLE: [Core] feat: Add --safetensors-load-strategy flag for faster safetensors loading from Lustre (#24469)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/load.py (+9/-0); vllm/engine/arg_utils.py (+5/-0); vllm/model_executor/model_loader/default_loader.py (+1/-0); vllm/model_executor/model_loader/weight_utils.py (+16/-6)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ The default method for loading safetensors relies on lazy loading (mmap). While this is highly efficient for local storage (like SSDs), it causes severe performance degradation on network or distributed file systems (like Lustre or NFS). These file systems handle the small random reads required by mmap very inefficiently, leading to extremely long model load times. ⏎  ⏎ This issue is documented in the safetensors repository (see: https:// …[truncated]

### L3-e2b1f863aa  (L3, 2025-09-10, sha e2b1f863aa0c, PR #24635)
TITLE: [Doc]: fixing doc typos (#24635)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+1/-1); vllm/config/__init__.py (+1/-1); vllm/entrypoints/openai/tool_parsers/internlm2_tool_parser.py (+1/-1); vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py (+1/-1); vllm/model_executor/models/arcee.py (+1/-1); vllm/model_executor/models/llava_onevision.py (+1/-1); vllm/model_executor/models/phi4_multimodal.py (+1/-1); vllm/model_executor/models/phi4mm_audio.py (+1/-1); vllm/model_executor/models/qwen2_5_omni_thinker.py (+2/-2)
LABELS: frontend, v1, tool-calling, qwen
BODY: ## Purpose ⏎  ⏎ Fixing typos to improve quality: see commit diffs for details ⏎  ⏎ ## Test Plan ⏎  ⏎ N/A ⏎  ⏎ ## Test Result ⏎  ⏎ N/A

### L3-9bd831f501  (L3, 2025-09-10, sha 9bd831f5012b, PR #23414)
TITLE: [Model] New model support for Motif-1-Tiny (#23414)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/differential_flash_attn.py (+3/-0); benchmarks/kernels/benchmark_polynorm.py (+155/-0); csrc/layernorm_kernels.cu (+251/-0); csrc/ops.h (+4/-1); csrc/torch_bindings.cpp (+6/-0); docs/models/supported_models.md (+1/-0); tests/kernels/core/test_layernorm.py (+32/-1); tests/models/registry.py (+3/-0); tests/models/test_initialization.py (+3/-2); vllm/_custom_ops.py (+8/-0); (+3 more)
LABELS: documentation, performance, new-model, ready
BODY: ## Purpose ⏎ New model for :  ⏎ https://huggingface.co/Motif-Technologies/Motif-2.6B ⏎ https://huggingface.co/Motif-Technologies/Motif-2.6b-v1.1-LC ⏎ Tech report (https://arxiv.org/pdf/2508.09148) ⏎  ⏎ Implemented the Polynorm custom kernel based on https://arxiv.org/abs/2411.03884 ⏎ co-author : @WyldeCat  ⏎ ## Test Plan ⏎ benchmark for Polynorm kernel (benchmarks/kernels/benchmark_polynorm.py) ⏎ ## Test Result ⏎ Benchmark Results ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted] …[truncated]

### L3-6aeb1dab4a  (L3, 2025-09-11, sha 6aeb1dab4a35, PR #24631)
TITLE: [Bugfix] Fix incorrect import of CacheConfig (#24631)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layers/cross_attention.py (+1/-2)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ #21088 incorrectly imports the deprecated `CacheConfig` from transformers library instead of vLLM, which breaks Whisper model (and any models that use Whisper vLLM impl as encoder) when using latest transformers version. ⏎  ⏎ FIX https://buildkite.com/vllm/ci/builds/30279/steps/canvas?sid=019936ef-4094-4ebf-85c3-047098b7a6ee ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-1fdd5c42d7  (L3, 2025-09-11, sha 1fdd5c42d7c5, PR #24111)
TITLE: [Kernels] Enable Torch Symmetric Memory All-Reduce By Default (#24111)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: benchmarks/kernels/benchmark_device_communicators.py (+486/-0); csrc/custom_all_reduce.cuh (+40/-13); vllm/distributed/device_communicators/all_reduce_utils.py (+2/-2); vllm/distributed/device_communicators/cuda_communicator.py (+8/-5); vllm/distributed/device_communicators/custom_all_reduce.py (+3/-2); vllm/distributed/device_communicators/symm_mem.py (+31/-6); vllm/envs.py (+2/-2)
LABELS: performance, ready
BODY: Enable torch symm memory for TP allreduce by default. ⏎ Add an testing option to custom allreduce to choose between one shot and two shot algos ⏎ Add a benchmark to compare nccl, custom allreduce and torch symm mem allreduce ⏎ The dispatching of the algorithms is done based on input size for cuda Hopper and Blackwell devices reaching the best performance out of the existing allreduce algorithms. ⏎  ⏎ E2E results are presented in the original PR #20759 ⏎ (Up  …[truncated]

### L3-e26fef8397  (L3, 2025-09-11, sha e26fef83977d, PR #24616)
TITLE: fix some typos (#24616)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+1/-1); tests/compile/test_basic_correctness.py (+1/-1); tests/kernels/mamba/test_mamba_ssm_ssd.py (+1/-1); vllm/model_executor/layers/fla/ops/cumsum.py (+1/-1); vllm/v1/worker/block_table.py (+2/-2)
LABELS: ready, v1
BODY: ## Purpose ⏎ fix some typos ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-e42af78b18  (L3, 2025-09-11, sha e42af78b18c5, PR #24197)
TITLE: [flashinfer] [kernel] support for fp8 kv cache for trtllm prefill attention (#24197)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/envs.py (+6/-0); vllm/utils/flashinfer.py (+6/-5); vllm/v1/attention/backends/flashinfer.py (+109/-9)
LABELS: ready, v1
BODY: Summary: ⏎ trtllm prefill attention does not support bf16 Q + fp8 kv. To enable fp8 kv using trtllm, we can add a dequant kernel before pretill attention, which creates a mock kv cache and mock block table, which include only tokens needed by prefills. ⏎  ⏎ Since for prefill, the involved tokens are small compared to what KV cache can hold overall, this dequant kernel brings relatively small overhead and allows us to enable fp8 kv. ⏎  ⏎ Test Plan: ⏎ AIME ⏎  ⏎ Lo …[truncated]

### L3-e93f4cc9e3  (L3, 2025-09-11, sha e93f4cc9e374, PR #24526)
TITLE: Add the support for the qwen3 next model (a hybrid attention model). (#24526)
SOURCES: corpus:production-kernel-provenance
ARTIFACT_HINTS: -
FILES: .yapfignore (+1/-0); docs/models/supported_models.md (+1/-0); pyproject.toml (+1/-0); tests/models/registry.py (+5/-1); vllm/config/__init__.py (+46/-10); vllm/config/compilation.py (+1/-0); vllm/model_executor/layers/fla/ops/chunk_delta_h.py (+3/-2); vllm/model_executor/layers/fla/ops/chunk_o.py (+5/-4); vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py (+6/-4); vllm/model_executor/layers/fla/ops/fused_recurrent.py (+2/-2); (+19 more)
LABELS: documentation, new-model, speculative-decoding, ready, v1, qwen
BODY: 

### L3-074854b24f  (L3, 2025-09-11, sha 074854b24f6e, PR #23696)
TITLE: [Kernel][B200] `mxfp4` fused cutlass moe (#23696)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/kernels/moe/test_mxfp4_moe.py (+319/-0); vllm/envs.py (+12/-1); vllm/model_executor/layers/fused_moe/layer.py (+10/-3); vllm/model_executor/layers/quantization/mxfp4.py (+283/-58); vllm/model_executor/warmup/kernel_warmup.py (+2/-2)
LABELS: documentation, performance, ready
BODY: # Purpose ⏎  ⏎ This PR adds the cutlass fused moe bf16xmxfp4 for gpt-oss on hopper and mxfp8xmxfp4 for gpt-oss on blackwell. ⏎  ⏎ For Hopper there is a performance regression compared to triton (you can see in the results section). This is something we are actively working on fixing and will push to flashinfer once it is resolved. Please let me know how you want to proceed with backend selection, and I can update accordingly.  ⏎  ⏎ For Blackwell the mxfp8 x  …[truncated]

### L3-d4fd2768ef  (L3, 2025-09-11, sha d4fd2768ef1c, PR #24692)
TITLE: [Bugfix][Attention] Fix FlashInfer MLA block size logic (#24692)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+11/-2)
LABELS: ready
BODY: ## Purpose ⏎ Before, specifying the `FLASHINFER_MLA` backend without a specified block size would lead to error. Block size would default to 16, and the backend only supports 32 or 64. This PR fixes it by overriding in a manner similar to the `CUTLASS_MLA` backend ⏎  ⏎ ## Test Plan ⏎ VLLM_ATTENTION_BACKEND=FLASHINFER_MLA vllm bench throughput --model=deepseek-ai/DeepSeek-V2-Lite-Chat --dataset-name=random --input-len=128 --output-len=128 --num-prompts=10 …[truncated]

### L3-bcb06d7baf  (L3, 2025-09-12, sha bcb06d7baf22, PR #24726)
TITLE: [Doc]: fix typos in various files (#24726)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+1/-1); benchmarks/kernels/benchmark_w8a8_block_fp8.py (+1/-1); csrc/cpu/cpu_types_vxe.hpp (+1/-1); csrc/cpu/sgl-kernels/moe.cpp (+1/-1); docs/design/multiprocessing.md (+1/-1); vllm/benchmarks/datasets.py (+1/-1); vllm/entrypoints/openai/protocol.py (+1/-1); vllm/model_executor/layers/mamba/mamba_mixer2.py (+1/-1); vllm/model_executor/models/minicpmv.py (+1/-1); vllm/v1/worker/gpu_model_runner.py (+1/-1); (+1 more)
LABELS: documentation, performance, frontend, tpu, v1
BODY: ## Purpose ⏎  ⏎ Improve quality by removing typos: see commit diffs for details. ⏎  ⏎ ## Test Plan ⏎  ⏎ N/A ⏎  ⏎ ## Test Result ⏎  ⏎ N/A

### L3-7ba32aa60b  (L3, 2025-09-12, sha 7ba32aa60b7b, PR #24705)
TITLE: [Attention][FlashInfer] Enable FP8 FlashInfer (TRTLLM) MLA decode (#24705)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.common_v1, L3.mla.flashinfer, L3.dispatch.abstract_interface, L3.platform.cuda_selection
FILES: vllm/attention/backends/abstract.py (+1/-0); vllm/envs.py (+2/-0); vllm/platforms/cuda.py (+4/-1); vllm/v1/attention/backends/mla/common.py (+0/-2); vllm/v1/attention/backends/mla/flashinfer_mla.py (+11/-7); tests/v1/attention/test_attention_backends.py (+1/-0); tests/v1/tpu/test_pallas.py (+2/-0); vllm/model_executor/layers/quantization/kv_cache.py (+2/-0)
LABELS: tpu, ready, v1, deepseek
BODY: ## Purpose ⏎ Enable FP8 kv cache for `FLASHINFER_MLA` backend. ⏎  ⏎ ## Test Plan ⏎  ⏎ ### Correctness ⏎ `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA lm_eval --model vllm --model_args '{"pretrained": "deepseek-ai/DeepSeek-V2-Lite-Chat", "trust_remote_code": true, "kv_cache_dtype": <dtype>}' --tasks gsm8k --batch_size auto` ⏎  ⏎ ### Performance ⏎ `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA vllm bench throughput --model=deepseek-ai/DeepSeek-V2-Lite-Chat --dataset-name=random -- …[truncated]

### L3-7a1c4025f1  (L3, 2025-09-12, sha 7a1c4025f1e2, PR #24701)
TITLE: [Kernel] [CPU] refactor `cpu_attn.py:_run_sdpa_forward` for better memory access (#24701)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+4/-4)
LABELS: ready, v1
BODY: ## Refactor movement ops in _run_sdpa_forward ⏎  ⏎ This pr moves the `repeat_interleave` call to occur after `movedim`, aligning the key/value tensors with the [B, H, L, D] layout before materializing extra heads. This avoids creating large strided views, which previously forced expensive copies. By repeating along the already head-major axis, we get a contiguous memory layout, producing a friendlier memory layout. ⏎  ⏎ In local benchmarks this yields mo …[truncated]

### L3-72fc8aa412  (L3, 2025-09-12, sha 72fc8aa41255, PR #24347)
TITLE: [Multi Modal] Add FA3 in VIT (#24347)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/attention/layer.py (+67/-8); vllm/platforms/cuda.py (+17/-11); vllm/platforms/interface.py (+2/-1); vllm/platforms/rocm.py (+9/-9); tests/entrypoints/openai/test_vision.py (+2/-2); tests/kernels/attention/test_mha_attn.py (+35/-14); vllm/model_executor/models/ernie45_vl.py (+20/-3); vllm/model_executor/models/glm4_1v.py (+19/-3); vllm/model_executor/models/keye.py (+15/-2); vllm/model_executor/models/qwen2_5_vl.py (+21/-3); (+3 more)
LABELS: rocm, ready, multi-modality, qwen
BODY: ## Purpose ⏎  ⏎ 1. Enable FA3 in VIT. The original FA3 is not supported because head_dim has to be a multiple of 32 (in vllm's build). However, in the upstream FA, it could theoratically support any head_dim which is a multiple of 8. (https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/flash_api.cpp#L605).  ⏎  ⏎ vLLM's native FA does not support it because it forbids `FLASHATTENTION_DISABLE_UNEVEN_K` to padding. (http://github.com/vl …[truncated]

### L3-5fe643fc26  (L3, 2025-09-12, sha 5fe643fc2656, PR #24753)
TITLE: Add FLASHINFER_MLA to backend selector test (#24753)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_attention_selector.py (+41/-19); tests/v1/attention/utils.py (+2/-0)
LABELS: ready, v1
BODY: ## Purpose ⏎ Adds the recently-added FlashInfer MLA backend to the attention selector test. This test would have caught the bug fixed by #24692  ⏎  ⏎ ## Test Plan ⏎ `pytest tests/kernels/attention/test_attention_selector.py` ⏎  ⏎ ## Test Result ⏎ passes ⏎  ⏎ --- ⏎ [details omitted]

### L3-dbeee3844c  (L3, 2025-09-13, sha dbeee3844cf9, PR #24757)
TITLE: [Perf] Use NVIDIA hardware-accelerated instruction for float to fp8_e4m3 quantization (#24757)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/quantization/fp8/common.cuh (+6/-2); csrc/quantization/fp8/nvidia/quant_utils.cuh (+16/-3)
LABELS: performance, ready
BODY: ## Purpose ⏎ Use NVIDIA hardware-accelerated instruction for float to fp8_e4m3 quantization. ⏎  ⏎ ## Test Plan && Test Result ⏎ Unit test: ⏎ `tests/kernels/quantization/test_fp8_quant.py` ⏎ ``` ⏎ ===== 109 passed, 1 warning in 15.71s ===== ⏎ ``` ⏎  ⏎ E2E accuracy test: ⏎ ``` ⏎ vllm ({'pretrained': 'nvidia/Llama-3.3-70B-Instruct-FP8', 'kv_cache_dtype': 'fp8', 'tensor_parallel_size': 1, 'max_model_len': 2048, 'trust_remote_code': True}), gen_kwargs: (temperature=0.0), lim …[truncated]

### L3-59d7ffc17f  (L3, 2025-09-13, sha 59d7ffc17f4c, PR #24750)
TITLE: [CI Failure] Fix test_flashinfer_cutlass_mxfp4_mxfp8_fused_moe (#24750)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: csrc/attention/mla/sm100_cutlass_mla_kernel.cu (+1/-0); tests/kernels/moe/test_mxfp4_moe.py (+2/-2)
LABELS: ready, ci-failure
BODY: ## Purpose ⏎  ⏎ Also possibly found the culprit for the blackwell cutlass mla failing test https://buildkite.com/vllm/ci/builds/30554/steps/canvas?jid=01993edf-720e-4749-81eb-da58099b7c78 ⏎ ``` ⏎ E       RuntimeError: _C::sm100_cutlass_mla_decode() expected at most 9 argument(s) but received 10 argument(s). Declaration: _C::sm100_cutlass_mla_decode(Tensor($0! -> ) out, Tensor q_nope, Tensor q_pe, Tensor kv_c_and_k_pe_cache, Tensor seq_lens, Tensor page_t …[truncated]

### L3-cc3173ae98  (L3, 2025-09-14, sha cc3173ae982c, PR #24511)
TITLE: [Multi Modal][Performance] Fused Q,K's apply_rope into one (#24511)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen2_5_vl.py (+10/-6)
LABELS: ready, qwen
BODY: 1. Reduce Q,K's apply_rope into one in Qwen2.5-VL -> reduce time cost from 731 &micro;s to 378 &micro;s, also reduce GPU time from 485 &micro;s to 255 &micro;s ⏎ 2. Merge multiple rearrange together in the code path of flash_attn -> further reduce 12 &micro;s when flash_attn is used ⏎  ⏎ Overall ~11% improvement in TTFT. I will work on other models if the change makes sense to reviewers.  ⏎  ⏎ Related: https://github.com/vllm-project/vllm/issues/23880, htt …[truncated]

### L3-01413e0cf5  (L3, 2025-09-15, sha 01413e0cf5a0, PR #22222)
TITLE: Fp8 paged attention update (#22222)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.rocm.custom_paged
FILES: csrc/rocm/attention.cu (+247/-78); vllm/_custom_ops.py (+2/-1); vllm/envs.py (+6/-0); csrc/rocm/ops.h (+2/-1); csrc/rocm/torch_bindings.cpp (+2/-1)
LABELS: rocm, ready
BODY: ## Purpose ⏎ Support fp8 mfma instruction with per wrap dynamic quantization for Query to improve performance, which reduces fp8 to fp16 data type conversion cost and improve mfma throughput for MI300x or later accelerators.  ⏎  ⏎ ## Test Plan ⏎ * Unit testing: ⏎    export VLLM_ROCM_FP8_MFMA_PAGE_ATTN=1 ⏎    pytest -s tests/kernels/attention/test_attention.py ⏎ * Benchmark: lm-eval-harness  ⏎ * LLM model: Meta-Llama-3.1-8B-Instruct ⏎ * Dataset: wikitext and gsm8k ⏎  …[truncated]

### L3-94b03f88dd  (L3, 2025-09-15, sha 94b03f88dd4d, PR #24868)
TITLE: Bump Flashinfer to 0.3.1 (#24868)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); setup.py (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ Bumps flashinfer to v0.3.1. Thes version comes with some fixes around certain code paths not being AOT'd.  ⏎  ⏎ ## Test Plan ⏎ Run CI checks ⏎ ## Test Result ⏎ CI checks passed ⏎ --- ⏎ [details omitted]

### L3-b834b4cbf1  (L3, 2025-09-15, sha b834b4cbf1d5, PR #20321)
TITLE: [USAGE] Improve error handling for weight initialization in Unquantized… (#20321)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+21/-4); vllm/model_executor/layers/linear.py (+22/-4)
LABELS: documentation, ready, v1
BODY: …LinearMethod and Attention class. ⏎  ⏎ When initializing vLLM the `torch.OutOfMemoryError: CUDA out of memory.` is opaque and does not indicate which step of the loading model it failed. Also it is not clear where the pre-allocated memory has been used. ⏎  ⏎ Errors like: ⏎ ``` ⏎ torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 2 has a total capacity of 23.57 GiB of which 235.88 MiB is free. Including non-PyTorch memory, this pro …[truncated]

### L3-2e41f5abca  (L3, 2025-09-15, sha 2e41f5abca79, PR #24745)
TITLE: [XPU] Set consistent default KV cache layout (#24745)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+10/-4); vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py (+9/-6); vllm/platforms/xpu.py (+4/-6)
LABELS: frontend, ready, v1
BODY: Small quality of life change to keep the way we interact with KV cache layout consistent as I explain a bit here https://github.com/vllm-project/vllm/pull/22735. ⏎ cc @zhenwei-intel to keep me true on the XPU-related change. PS that limit will prevent XPU from being compatible with heteroTP as of now.  ⏎  ⏎ cc @LucasWilkinson as this is another kv layout constraint which is good to keep in mind.

### L3-b42566f440  (L3, 2025-09-15, sha b42566f44039, PR #24774)
TITLE: [Bug] Fix `is_flashmla_supported` Check Error (#24774)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter
FILES: vllm/attention/backends/flashmla.py (+2/-13); vllm/v1/attention/backends/mla/flashmla.py (+2/-13)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Fixes https://github.com/vllm-project/vllm/pull/24521#issuecomment-3283794766 ⏎  ⏎ While developing, just found that we have `is_flashmla_supported` and it has a bug with (False, "message"), so this PR fix that

### L3-aae725af7c  (L3, 2025-09-15, sha aae725af7cee, PR #24891)
TITLE: [Performance] Remove redundant clone() calls in cutlass_mla (#24891)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.cutlass_v1_backend
FILES: vllm/v1/attention/backends/mla/cutlass_mla.py (+8/-8)
LABELS: ready, v1
BODY: This PR removes 2 redundant clone() calls in pre-attn cutlass MLA python code (that we found in the profiling work). This PR is step 1 (_"Remove unnecessary copies from Cutlass MLA"_) from this meta fusion issue (https://github.com/vllm-project/vllm/issues/24629):  For DeekSeekR1 on 8xB200 GPUs batch size 32, this improves decode perf by 2.4% from 19.15ms TPO to 18.7ms. ⏎  ⏎ Verified that correctness is preserved via manual check and also lm_eval on  …[truncated]

### L3-759ef49b15  (L3, 2025-09-15, sha 759ef49b1514, PR #24907)
TITLE: Remove V0 Encoder-Decoder Support (#24907)
SOURCES: symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/scripts/hardware_ci/run-cpu-test.sh (+0/-1); .buildkite/test-pipeline.yaml (+0/-9); docs/contributing/model/multimodal.md (+0/-1); docs/models/supported_models.md (+0/-8); docs/usage/v1_guide.md (+1/-1); examples/offline_inference/dolphin.py (+0/-311); examples/offline_inference/encoder_decoder.py (+0/-195); examples/offline_inference/encoder_decoder_multimodal.py (+1/-113); examples/offline_inference/vision_language.py (+0/-62); examples/offline_inference/vision_language_multi_image.py (+0/-21); (+37 more)
LABELS: documentation, new-model, ready, ci/build, v1, multi-modality, llama
BODY: Remove V0 encoder decoder model runner. ⏎ Also, this PR deletes the deprecated models such as BART. After this PR, Whisper will be the only encoder-decoder model that are supported by vLLM.

### L3-e95084308b  (L3, 2025-09-16, sha e95084308b20, PR #24906)
TITLE: Updated CODEOWNERS for flashinfer, mla, fused_moe (#24906)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .github/CODEOWNERS (+7/-2)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-0af3ce1355  (L3, 2025-09-16, sha 0af3ce135580, PR #24470)
TITLE: Upgrade flashinfer to 0.3.1 (#24470)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.nightly_torch (+2/-2)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ More fixes release from FlashInfer, so bump up to 0.3.1. ⏎  ⏎ ## Test Plan ⏎ CI ⏎  ⏎ ## Test Result ⏎ TODO ⏎  ⏎ [details omitted]

### L3-17871983a2  (L3, 2025-09-16, sha 17871983a2ef, PR #24021)
TITLE: [Bugfix] Fix sequence parallelism bug when enable pipeline parallelism (#24021)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/distributed/test_sequence_parallel.py (+2/-1); vllm/distributed/parallel_state.py (+46/-5); vllm/v1/worker/cpu_worker.py (+15/-3); vllm/v1/worker/gpu_model_runner.py (+33/-30); vllm/v1/worker/gpu_worker.py (+13/-2); vllm/v1/worker/utils.py (+26/-1)
LABELS: ready, v1
BODY: This PR fixes incorrect output when Tensor Parallelism + Sequence Parallelism + Pipeline Parallelism are enabled. ⏎ The issue was due to residuals differing across Tensor Parallelism GPUs after reduce-scatter, which makes using all-gather incorrect when sending/receiving residual tensors for Pipeline Parallelism. ⏎ It also fixes a unit test bug.

### L3-567939953b  (L3, 2025-09-16, sha 567939953b7a, PR #23693)
TITLE: [Core/DBO][1/N] Add Dual-Batch Overlap mechanism to VLLM (#23693)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+9/-9); examples/offline_inference/data_parallel.py (+8/-0); tests/v1/attention/test_attention_splitting.py (+5/-5); tests/v1/spec_decode/test_eagle.py (+6/-2); vllm/config/__init__.py (+8/-0); vllm/config/parallel.py (+8/-0); vllm/distributed/device_communicators/all2all.py (+0/-5); vllm/engine/arg_utils.py (+10/-0); vllm/forward_context.py (+101/-20); vllm/model_executor/layers/fused_moe/deepep_ht_prepare_finalize.py (+14/-12); (+12 more)
LABELS: documentation, speculative-decoding, ready, v1
BODY: ## Purpose ⏎ This PR adds support for Dual-Batch Overlap in VLLM. In it's current state it will only be abled when a user provides the --enable-microbatching flag. Furthermore, it will only be used when all DP groups are running full-decode batches. This PR supports running DBO with full cudagraphs, which is essential for minimizing the CPU overhead and getting performance from this feature.  ⏎  ⏎ To implement Dual-Batch Overlap (DBO), at a high level, …[truncated]

### L3-cd1f885bcf  (L3, 2025-09-16, sha cd1f885bcfe3, PR #24866)
TITLE: Directly get max encoder len from VLLM config in V1 (#24866)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layers/cross_attention.py (+7/-4)
LABELS: ready
BODY: Improves performance by getting the max encoder length directly from the initialized `vllm_config.scheduler_config`. This avoids the expensive lookup and re-computation previously done by `MULTIMODAL_REGISTRY.get_encdec_max_encoder_len`. ⏎  ⏎ **Test Results:** ⏎ - **Environment:** H20 GPU ⏎ - **Data:** 10s audio ⏎ - **Before:** Average latency was 1300ms. ⏎ - **After:** Average latency is now 305ms. ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omi …[truncated]

### L3-cef32104b4  (L3, 2025-09-16, sha cef32104b4b4, PR #24342)
TITLE: [FP8] Extend per-token-group quantization support to QuantFP8 (#24342)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/bench_per_token_quant_fp8.py (+216/-47); tests/kernels/quantization/test_fp8_quant_group.py (+150/-0); vllm/model_executor/layers/fused_moe/fused_moe.py (+3/-1); vllm/model_executor/layers/quantization/input_quant_fp8.py (+66/-13); vllm/model_executor/layers/quantization/utils/quant_utils.py (+9/-0)
LABELS: performance, ready
BODY: ## Purpose ⏎  ⏎ Extends `QuantFP8` to support per-token-group quantization and adds a torch implementation of the QuantFP8 group quantization. ⏎  ⏎ Addresses #24185 ⏎  ⏎ ## Changes ⏎  ⏎ - Add `is_per_tensor()`, `is_per_token()`, `is_per_group()` helper methods to `GroupShape` ⏎ - Extend `QuantFP8` to support arbitrary group sizes like `GroupShape(1, 128)` ⏎ - Added `test_fp8_quant_group.py` and `benchmark_quantfp8_group.py` ⏎  ⏎ ## Test Plan ⏎  ⏎ Tested with existing test s …[truncated]

### L3-dd83a157f1  (L3, 2025-09-16, sha dd83a157f12c, PR #24761)
TITLE: [UX] Enforce valid choices for envs like VLLM_ATTENTION_BACKEND, etc (#24761)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+77/-24)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ For environment variables that are looking for known string values, we have no standard for checking and guiding the user if they gave an invalid value. This PR introduces an `env_with_choices` helper function that can take a list of choices to compare against the user's request, raising an error if it isn't valid. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]
