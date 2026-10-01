### L3-79c92c7c8a  (L3, 2024-06-27, sha 79c92c7c8aa6, PR #5908)
TITLE: [Model] Add Gemma 2 (#5908)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/source/models/supported_models.rst (+4/-0); requirements-common.txt (+1/-1); vllm/config.py (+23/-7); vllm/lora/layers.py (+4/-0); vllm/model_executor/layers/layernorm.py (+46/-0); vllm/model_executor/layers/logits_processor.py (+9/-1); vllm/model_executor/layers/rotary_embedding.py (+10/-0); vllm/model_executor/models/__init__.py (+1/-0); vllm/model_executor/models/gemma2.py (+401/-0)
LABELS: new-model
ISSUES: #5912 [Feature]: Support for google/gemma-2-9b-it / gemma-2-27b-it
BODY: This PR adds Gemma 2, a new family of open LLMs from Google. ⏎  ⏎ Two major issues to note: ⏎ 1. Attention logit soft-capping: Gemma 2 models soft-cap the attention logits. This requires changes to all the attention kernels vLLM is using, so **this PR removes soft-capping as a temporary workaround**. While this makes the model different from the original implementation, it does not significantly affect the model's generation capability. ⏎ 2. Sliding  …[truncated]

### L3-f136da15e1  (L3, 2024-06-27, sha f136da15e154, PR #5878)
TITLE: [Hardware][TPU] Optimize KV cache swapping (#5878)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/pallas.py (+4/-12); vllm/worker/tpu_worker.py (+32/-10)
LABELS: tpu
BODY: Currently, the swapping operation inadvertently copies the entire CPU KV cache into GPU, which is very slow. This PR optimizes this.

### L3-be0b3af9e0  (L3, 2024-06-28, sha be0b3af9e068, PR #4650)
TITLE: Support Deepseek-V2 (#4650)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config.py (+6/-0); vllm/model_executor/layers/fused_moe/__init__.py (+2/-1); vllm/model_executor/layers/fused_moe/fused_moe.py (+31/-0); vllm/model_executor/layers/rotary_embedding.py (+126/-0); vllm/model_executor/models/__init__.py (+1/-0); vllm/model_executor/models/deepseek_v2.py (+534/-0)
LABELS: new-model, deepseek
ISSUES: #5763 DeepSeekCoderV2
BODY: ## Description: ⏎  ⏎ This PR introduces support for the recently released DeepSeek-V2 model by DeepSeek-AI.  ⏎  ⏎ ### Key Updates: ⏎  ⏎ - **Model Integration**: Successfully integrated the DeepSeek-V2 model, developed by the DeepSeek-AI team, aiming to provide advanced natural language processing capabilities. ⏎  ⏎ ### Related Resources: ⏎  ⏎ - **Model Repository**: [DeepSeek-V2 Model Repository](https://github.com/deepseek-ai/DeepSeek-V2) ⏎ - **Technical R …[truncated]

### L3-57f09a419c  (L3, 2024-06-28, sha 57f09a419c04, PR #5379)
TITLE: [Hardware][Intel] OpenVINO vLLM backend (#5379)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen, L3.dispatch.selector
FILES: vllm/attention/backends/openvino.py (+101/-0); vllm/attention/selector.py (+11/-1); .buildkite/run-openvino-test.sh (+14/-0); Dockerfile.openvino (+26/-0); benchmarks/benchmark_latency.py (+4/-3); benchmarks/benchmark_throughput.py (+4/-3); docs/source/getting_started/openvino-installation.rst (+95/-0); docs/source/index.rst (+1/-0); requirements-openvino.txt (+9/-0); setup.py (+10/-1); (+12 more)
BODY: Adds OpenVINO vLLM backend, described here https://github.com/vllm-project/vllm/issues/5377 ⏎  ⏎ Below and performance measurements: ⏎ - VLLM - vLLM CPU backend is taken from commit https://github.com/vllm-project/vllm/commit/388596c91437a51d428a447594e9faec340c29b2 ⏎ - OV (-Q) - OpenVINO vLLM backend w/o KV cache / weights compression, default vLLM scheduling mode ⏎ - OV (+Q) - OpenVINO vLLM backend w/ KV cache / weights compression, chunked prefill  …[truncated]

### L3-7041de4384  (L3, 2024-06-28, sha 7041de43849f, PR #4628)
TITLE: [Kernel] Flashinfer for prefill & decode, with Cudagraph support for decode (#4628)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.dispatch.selector
FILES: requirements-test.txt (+1/-1); vllm/attention/backends/flashinfer.py (+58/-25); vllm/attention/selector.py (+3/-2); vllm/worker/model_runner.py (+248/-78); .buildkite/test-pipeline.yaml (+3/-0); tests/basic_correctness/test_basic_correctness.py (+0/-6); tests/distributed/test_basic_distributed_correctness.py (+0/-5)
BODY: This is the second PR to integrate flashinfer. The goal is to use flashinfer for the prefill phase (including prefix caching). Hopefully, we can get rid of the [context_attention_fwd](https://github.com/vllm-project/vllm/blob/323f27b9048713cdbab31995265975842a937167/vllm/attention/ops/prefix_prefill.py#L663) triton kernel and use flashinfer [append](https://docs.flashinfer.ai/api/python/prefill.html#batch-prefill-append-attention) kernel for bett …[truncated]

### L3-4bf35ed9ae  (L3, 2024-06-28, sha 4bf35ed9ae3c, PR #5936)
TITLE: [Bugfix] Only add `Attention.kv_scale` if kv cache quantization is enabled (#5936)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+14/-9)
BODY: This PR addresses an issue where `Attention.kv_scale` parameters were being created unnecessarily for models without FP8 KV cache quantization.  ⏎  ⏎ Now we will only create and attempt to load `kv_scale` parameters when FP8 KV cache quantization is explicitly enabled with `kv_cache_dtype`. This change prevents errors when loading checkpoints for models that don't use FP8 quantization, while maintaining compatibility with FP8 quantized models. This …[truncated]

### L3-614aa51203  (L3, 2024-06-30, sha 614aa5120303, PR #6007)
TITLE: [misc][cuda] use nvml to avoid accidentally cuda initialization (#6007)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/blocksparse_attention/interface.py (+3/-3); vllm/attention/ops/prefix_prefill.py (+3/-1); tests/kernels/test_cutlass.py (+2/-1); tests/quantization/utils.py (+2/-1); vllm/distributed/device_communicators/custom_all_reduce.py (+5/-53); vllm/lora/punica.py (+2/-1); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py (+2/-1); vllm/model_executor/layers/quantization/fp8.py (+2/-2); vllm/model_executor/layers/quantization/gptq_marlin.py (+2/-1); vllm/model_executor/layers/quantization/utils/marlin_utils.py (+2/-1); (+3 more)
ISSUES: #6004 [Bug]: `distributed_executor_backend=mp` does not work with GPTQ tp>1
BODY: fixes https://github.com/vllm-project/vllm/issues/6004

### L3-c4059ea54f  (L3, 2024-07-01, sha c4059ea54ff3, PR #6044)
TITLE: [Bugfix] Add explicit `end_forward` calls to flashinfer (#6044)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+2/-0)
BODY: Without and explicit `end_forward` call, the intermediate buffers may have leftover garbage data, affecting results. ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-c5832d2ae9  (L3, 2024-07-02, sha c5832d2ae943, PR #4412)
TITLE: [Core] Pipeline Parallel Support (#4412)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: .buildkite/test-pipeline.yaml (+10/-0); tests/async_engine/test_async_llm_engine.py (+13/-1); tests/async_engine/test_openapi_server_ray.py (+2/-2); tests/basic_correctness/test_preemption.py (+12/-12); tests/distributed/test_comm_ops.py (+17/-3); tests/distributed/test_pipeline_parallel.py (+149/-0); tests/engine/output_processor/test_multi_step.py (+4/-4); tests/entrypoints/openai/test_chat.py (+2/-2); tests/entrypoints/openai/test_completion.py (+2/-2); tests/entrypoints/openai/test_embedding.py (+2/-2); (+72 more)
ISSUES: #4461 [RFC]: Initial support for Pipeline Paralleism
BODY: Adds initial pipeline parallelism support to vLLM.  ⏎  ⏎ ToDo: ⏎  ⏎ Milestone 1: POC Prototype ⏎  ⏎  ⏎ Milestone 2: Mergeable ⏎  ⏎  ⏎ FIX #4461  ⏎  ⏎ Goals for this PR: ⏎ - Functional eager-mode PP ⏎ - Support AsyncLLMEngine ⏎ - Support RayGPUExecutor ⏎ - Support LLaMa/GPT2 ⏎ - Support chunked prefill ⏎  ⏎ Non-goals for this PR (To be covered in future PRs) ⏎ - Be fully optimized ⏎ - Support LLMEngine (this may be removed in the future) ⏎ - Support any other distributed …[truncated]

### L3-482045ee77  (L3, 2024-07-02, sha 482045ee77a4, PR #6080)
TITLE: [hardware][misc] introduce platform abstraction (#6080)
SOURCES: path_core, corpus:production-kernel-provenance
ARTIFACT_HINTS: L3.triton.prefix_prefill, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/attention/ops/blocksparse_attention/interface.py (+3/-2); vllm/attention/ops/prefix_prefill.py (+2/-2); tests/kernels/test_cutlass.py (+2/-2); tests/quantization/utils.py (+2/-2); vllm/lora/punica.py (+2/-2); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py (+2/-2); vllm/model_executor/layers/quantization/fp8.py (+3/-2); vllm/model_executor/layers/quantization/gptq_marlin.py (+2/-2); vllm/model_executor/layers/quantization/utils/marlin_utils.py (+2/-2); vllm/model_executor/model_loader/loader.py (+3/-2); (+6 more)
ISSUES: #6059 [Bug][CI/Build]: Missing attribute 'nvmlDeviceGetHandleByIndex' in AMD tests
BODY: fixes https://github.com/vllm-project/vllm/issues/6059 ⏎  ⏎ the idea is to progressively absorb common `if-else` patterns into the `vllm/platforms` directory, so that we don't have scattered `if-else` clause for platform identification everywhere. ⏎  ⏎ ideally, the platform identification is done in `from vllm.platforms import current_platform` , just once. ⏎  ⏎ we can progressively move towards the ultimate goal, with this being the first step.

### L3-56b325e977  (L3, 2024-07-03, sha 56b325e97743, PR #6043)
TITLE: [ROCm][AMD][Model]Adding alibi slopes support in ROCm triton flash attention and naive flash attention (#6043)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+51/-2)
LABELS: rocm
BODY: Adding alibi slopes support in ROCm triton flash attention and naive flash attention. ⏎ This fixes models such as [Jais](https://huggingface.co/core42/jais-13b) that rely on this functionality. ⏎  ⏎ With this change, the perplexity score for core42/jais13b by the https://github.com/Alexei-V-Ivanov-AMD/vllm/blob/pplv2_test/examples/measure_ppl2_llama2_MC.py script with an adjusted tokenizer improves from 14.5 to 5.2 ⏎  ⏎ **BEFORE SUBMITTING, PLEASE REA …[truncated]

### L3-69ec3ca14c  (L3, 2024-07-04, sha 69ec3ca14cf3, PR #6051)
TITLE: [Kernel][Model] logits_soft_cap for Gemma2 with flashinfer (#6051)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.dispatch.selector
FILES: vllm/attention/backends/flashinfer.py (+8/-4); vllm/attention/selector.py (+3/-3); vllm/model_executor/models/gemma2.py (+0/-7); vllm/worker/model_runner.py (+15/-4); .buildkite/test-pipeline.yaml (+5/-2); tests/kernels/test_flashinfer.py (+248/-0)
BODY: Add logits_soft_cap for flashinfer, which is needed by Gemma2 model, also add a simple gemma2 test.

### L3-e58294ddf2  (L3, 2024-07-05, sha e58294ddf231, PR #5695)
TITLE: [Bugfix] Add verbose error if scipy is missing for blocksparse attention (#5695)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/blocksparse_attention/utils.py (+13/-6)
BODY: Scipy is required for https://github.com/vllm-project/vllm/blob/4a30d7e3ccae6e977d728e2157aaa11ac0fed549/vllm/attention/ops/blocksparse_attention/utils.py#L9 ⏎ but is no longer installed with the dependencies. ⏎  ⏎ --- ⏎ This pr fixes this by adding scipy to `requirements-common.txt`.

### L3-4f0e0ea131  (L3, 2024-07-08, sha 4f0e0ea131ef, PR #6172)
TITLE: Add FlashInfer to default Dockerfile (#6172)
SOURCES: subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile (+3/-0)
ISSUES: #6169 [Bug]: TypeError: 'NoneType' object is not callable when loading Gemma 2 9B with new 0.5.1 version
BODY: Closes #6169  ⏎  ⏎ Testing with  ⏎ ``` ⏎ docker run --gpus all -p 8000:8000 -e HF_TOKEN --ipc=host --env "VLLM_ATTENTION_BACKEND=FLASHINFER" -v /data/xmo/hub:/root/.cache/huggingface vllm/vllm-openai --model google/gemma-2-9b-it ⏎ ``` ⏎  ⏎ ``` ⏎ $ curl http://localhost:8000/v1/completions  -H "Content-Type: application/json"      -d '{ ⏎ "model": "google/gemma-2-9b-it", ⏎ "prompt":"Who won the world series in 2020?", ⏎ "max_tokens": 100, ⏎ "ignore_eos": true …[truncated]

### L3-543aa48573  (L3, 2024-07-08, sha 543aa4857362, PR #4888)
TITLE: [Kernel] Correctly invoke prefill & decode kernels for cross-attention (towards eventual encoder/decoder model support) (#4888)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: vllm/attention/backends/abstract.py (+8/-0); vllm/attention/backends/blocksparse_attn.py (+8/-1); vllm/attention/backends/flash_attn.py (+8/-1); vllm/attention/backends/flashinfer.py (+7/-1); vllm/attention/backends/ipex_attn.py (+7/-1); vllm/attention/backends/pallas.py (+7/-1); vllm/attention/backends/rocm_flash_attn.py (+8/-1); vllm/attention/backends/torch_sdpa.py (+7/-1); vllm/attention/backends/utils.py (+7/-0); vllm/attention/backends/xformers.py (+394/-78); (+4 more)
BODY: This PR is a step towards encoder/decoder model support. This PR modifies the xFormers backend* such that (1) the attention impl can implement cross-attention, and (2) the attention metadata data structure can represent the necessary metadata for invoking cross-attention.  ⏎  ⏎ \* FlashAttention backend support for encoder/decoder models is left as future work ⏎  ⏎ ---------- ⏎  ⏎ A quick overview of the plan for supporting encoder/decoder models in vL …[truncated]

### L3-d6ab528997  (L3, 2024-07-12, sha d6ab5289976f, PR #6351)
TITLE: [Misc] Remove flashinfer warning, add flashinfer tests to CI (#6351)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/selector.py (+0/-3); .buildkite/test-pipeline.yaml (+5/-3); tests/basic_correctness/test_basic_correctness.py (+5/-0)
BODY: Add flashinfer basic correctness tests to CI, remove the llama-7b warning since it is fixed. ⏎ CI will fail for now, we need flashinfer's new release to pass the CI, will update CI flashinfer version accordingly.

### L3-d59eb98489  (L3, 2024-07-12, sha d59eb9848910, PR #6343)
TITLE: [Model][Phi3-Small] Remove scipy from blocksparse_attention (#6343)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/blocksparse_attention/utils.py (+27/-8)
ISSUES: #6334 [Bug]: Unable to run phi-3-small in latest release
BODY: Scipy is currently an optional dependency since we only use it once for Phi3-Small blocksparse attention. This PR proposes to remove scipy completely by re-implementing the CSR matrix calculation in a minimal and naive way. This is useful because users are confused by the need to install scipy themselves. The performance of this implementation shouldn't matter much since its result is cached. ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/633 …[truncated]

### L3-aa48e502fb  (L3, 2024-07-12, sha aa48e502fba0, PR #5327)
TITLE: [MISC] Upgrade dependency to PyTorch 2.3.1 (#5327)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1); requirements-build.txt (+1/-1); requirements-cuda.txt (+4/-4); .github/workflows/publish.yml (+1/-1); pyproject.toml (+1/-1)
ISSUES: #4509 [Bug] FP8 MoE performance regression since updating to torch 2.3.0 | #5535 [Bug]: Performance : very slow inference for Mixtral 8x7B Instruct FP8 on H100 with 0.5.0 and 0.5.0.post1 | #5579 [Bug]: Mixtral8x7B very high spikes for Inter Token Latency (ITL) | #5705 [Feature]: support torch 2.3.1
BODY: PyTorch 2.3.1 just released, and this PR upgrades the dependency to 2.3.1. ⏎ The most important reason of this upgrade is PyTorch strictly depends on a particular triton version, and PyTorch 2.3.0 depends on triton 2.3.0. However, @pcmoritz pointed out a performance bug in triton 2.3.0 and it has been fixed in triton 2.3.1, so in order to achieve the best Mixtral FP8 performance, we have to use triton 2.3.1, which results in version conflict if we …[truncated]

### L3-f8f9ff57ee  (L3, 2024-07-12, sha f8f9ff57ee36, PR #6397)
TITLE: [Bugfix][TPU] Fix megacore setting for v5e-litepod (#6397)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/pallas.py (+1/-1)
LABELS: tpu
BODY: The current code does not correctly set the megacore for TPU-v5e-litepod. This PR fixes the bug.

### L3-e1684a766a  (L3, 2024-07-12, sha e1684a766ad3, PR #6373)
TITLE: [Bugfix] Fix hard-coded value of x in context_attention_fwd (#6373)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/prefix_prefill.py (+2/-2)
ISSUES: #6332 [Bug]: `tests/basic_correctness/test_chunked_prefill.py` is failing on main in fp32 
BODY: Fixes #6332  ⏎  ⏎ The prefix prefill code assumes that `x=8` but this is only the case for fp16 (e.g., see this [line](https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/paged_attn.py#L51)).  In certain scenarios it is useful to be able to use the prefix prefill code with fp32 (e.g., to compare logprobs generated by chunked prefill against HF generate in CI tests). The correct value of `x` is very easy to extract from the `key_cache` …[truncated]

### L3-44874a0bf9  (L3, 2024-07-14, sha 44874a0bf970, PR #6437)
TITLE: [Doc] add env docs for flashinfer backend (#6437)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+1/-0)
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ add env docs for flashinfer backend ⏎  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-a63a4c6341  (L3, 2024-07-15, sha a63a4c634174, PR #6447)
TITLE: [Misc] Use 0.0.9 version for flashinfer (#6447)
SOURCES: subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile (+1/-1)
LABELS: ready
BODY: This is a fix for the segfault issue: https://github.com/vllm-project/vllm/issues/6252

### L3-4ef95b0f06  (L3, 2024-07-15, sha 4ef95b0f0677, PR #6409)
TITLE: [Bugfix] use float32 precision in samplers/test_logprobs.py for comparing with HF  (#6409)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/prefix_prefill.py (+6/-0); tests/samplers/test_logprobs.py (+2/-1)
LABELS: ready
ISSUES: #6408 [Bug]: `samplers/test_logprobs.py` fail on H100
BODY: Fixes #6408  ⏎  ⏎ This PR changes the precision in `tests/samplers/test_logprobs.py` from `half` to `float`.  ⏎  ⏎ This is needed because the test is comparing the actual values of the logprobs against the equivalent outputs from HF. There is precedent established for doing this in other tests (see e.g., [here](https://github.com/vllm-project/vllm/blob/d80aef37764feda51e21065de9c669785fe1d94a/tests/models/test_models.py#L27) or [here](https://github. …[truncated]

### L3-978aed5300  (L3, 2024-07-16, sha 978aed53004b, PR #6081)
TITLE: [Kernel][Attention] Separate `Attention.kv_scale` into `k_scale` and `v_scale` (#6081)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: csrc/attention/attention_kernels.cu (+27/-26); csrc/cache_kernels.cu (+11/-10); csrc/cpu/attention.cpp (+6/-6); vllm/attention/backends/abstract.py (+2/-1); vllm/attention/backends/blocksparse_attn.py (+6/-3); vllm/attention/backends/flash_attn.py (+4/-2); vllm/attention/backends/flashinfer.py (+4/-2); vllm/attention/backends/ipex_attn.py (+9/-5); vllm/attention/backends/pallas.py (+3/-2); vllm/attention/backends/rocm_flash_attn.py (+6/-3); (+23 more)
LABELS: ready
BODY: Since we already quantize `key_cache` and `value_cache` separately in PagedAttention, there is "free accuracy on the table" for FP8 KV Cache quantization as we could use separate per-tensor scales for each.  ⏎  ⏎ The FlashInfer FP8 attention kernel also uses separate `k_scale` and `v_scale` values, so this PR is in preparation to enable that usage. Source: https://github.com/flashinfer-ai/flashinfer/blob/dc2c76f8577d8695112b61d1fd43ef88569272ef/pyt …[truncated]

### L3-2fa4623d9e  (L3, 2024-07-17, sha 2fa4623d9e67, PR #6164)
TITLE: [Core] Refactor _prepare_model_input_tensors - take 2 (#6164)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: vllm/attention/backends/abstract.py (+43/-2); vllm/attention/backends/blocksparse_attn.py (+11/-0); vllm/attention/backends/flash_attn.py (+181/-2); vllm/attention/backends/flashinfer.py (+236/-2); vllm/attention/backends/rocm_flash_attn.py (+11/-0); vllm/attention/backends/utils.py (+233/-1); vllm/attention/backends/xformers.py (+10/-0); vllm/attention/selector.py (+3/-2); tests/worker/test_model_input.py (+5/-1); vllm/attention/__init__.py (+3/-1); (+2 more)
LABELS: ready
BODY: This PR refactors `_prepare_model_input_tensors`. Specifically, we introduce `ModelRunnerInputBuilder` mainly for logic isolation and modularization. Specifically, `ModelRunnerInputBuilder` manages all processed input data, including token IDs, positions, sequence length, etc, in one place, and isolates the following logic: ⏎ The logic of inserting a new sequence group to input data, considering prefix caching, chunked prefill, sliding windows, et …[truncated]

### L3-c8a7d51c49  (L3, 2024-07-18, sha c8a7d51c4982, PR #6501)
TITLE: [Bugfix] Update flashinfer.py with PagedAttention forwards - Fixes Gemma2 OpenAI Server Crash (#6501)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+3/-2)
LABELS: ready
ISSUES: #6477 [Bug]: Gemma 27B crashes on GCP A100
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ FIX #6477 (*link existing issues this PR will resolve*) ⏎  ⏎ https://github.com/vllm-project/vllm/issues/6477 ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-9042d68362  (L3, 2024-07-20, sha 9042d683620a, PR #6541)
TITLE: [Misc] Consolidate and optimize logic for building padded tensors (#6541)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+0/-3); vllm/attention/backends/flashinfer.py (+0/-3); vllm/attention/backends/utils.py (+0/-3); tests/conftest.py (+3/-9); vllm/model_executor/sampling_metadata.py (+19/-37); vllm/utils.py (+51/-10); vllm/worker/cpu_model_runner.py (+0/-3); vllm/worker/neuron_model_runner.py (+4/-4); vllm/worker/xpu_model_runner.py (+0/-3)
LABELS: ready
BODY: Following #6442 , this PR introduces a small refactor to clean up the code. ⏎  ⏎ cc @peng1999

### L3-683e3cb9c4  (L3, 2024-07-20, sha 683e3cb9c4b2, PR #6559)
TITLE: [ Misc ] `fbgemm` checkpoints (#6559)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+2/-1); .buildkite/lm-eval-harness/configs/Meta-Llama-3-8B-Instruct-Channelwise-compressed-tensors.yaml (+2/-2); .buildkite/lm-eval-harness/configs/Meta-Llama-3-8B-Instruct-FBGEMM-nonuniform.yaml (+11/-0); .buildkite/lm-eval-harness/run-lm-eval-gsm-vllm-baseline.sh (+1/-1); vllm/_custom_ops.py (+2/-0); vllm/config.py (+1/-1); vllm/model_executor/layers/fused_moe/layer.py (+1/-1); vllm/model_executor/layers/linear.py (+16/-10); vllm/model_executor/layers/quantization/__init__.py (+2/-0); vllm/model_executor/layers/quantization/aqlm.py (+2/-2); (+14 more)
LABELS: ready
BODY: SUMMARY: ⏎ * support `fbgemm` checkpoints from https://github.com/huggingface/transformers/pull/32047/files using our existing cutlass kernels ⏎ * note: updated with latest state that removes `activation_scale_ub` from the state dict and instead uses the config.json ⏎  ⏎ ```python ⏎ from vllm import LLM ⏎ model = LLM("nm-testing/Meta-Llama-3-8B-Instruct-FBGEMM-nonuniform")  ⏎ ``` ⏎  ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE  …[truncated]

### L3-06d6c5fe9f  (L3, 2024-07-20, sha 06d6c5fe9fad, PR #6543)
TITLE: [Bugfix][CI/Build][Hardware][AMD] Fix AMD tests, add HF cache, update CK FA, add partially supported model notes (#6543)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake, L3.rocm.rocm_flash_attn_v0
FILES: CMakeLists.txt (+2/-2); Dockerfile.rocm (+35/-25); requirements-rocm.txt (+4/-0); vllm/attention/backends/rocm_flash_attn.py (+8/-0); .buildkite/run-amd-test.sh (+7/-0); .buildkite/test-pipeline.yaml (+2/-1); docs/source/getting_started/amd-installation.rst (+4/-3); tests/basic_correctness/test_cpu_offload.py (+7/-2); tests/models/test_paligemma.py (+17/-1); tests/models/test_phi3v.py (+8/-1); (+2 more)
LABELS: rocm, ready
BODY: This PR attempts to fix existing failures on AMD tests. ⏎  ⏎ - Metrics tests are currently failing because of https://github.com/vllm-project/vllm/pull/6338, which accidentally added a hard requirement for `vllm_flash_attn` in `vllm/spec_decode/draft_model_runner.py`. This is not installed on ROCm, so the correct ROCm FA component is used instead. ⏎ - Basic correctness tests are currently failing because the test harness unconditionally installs `fl …[truncated]

### L3-e0c15758b8  (L3, 2024-07-23, sha e0c15758b858, PR #6596)
TITLE: [Core] Modulize prepare input and attention metadata builder (#6596)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+3/-17); vllm/attention/backends/flash_attn.py (+22/-21); vllm/attention/backends/flashinfer.py (+31/-29); vllm/attention/backends/utils.py (+25/-24); vllm/utils.py (+5/-0); vllm/worker/model_runner.py (+323/-207)
LABELS: ready
BODY: This PR further refactors model input builder and attention metadata builder to be more modulized and maintainable. Specifically: ⏎ - Introduce an inner data class to encapsulate intermediate data. ⏎ - Put the logic of processing a particular feature (e.g., prefix caching, sliding windows, lora, mm, etc) to separate functions. ⏎ - Use pre-defined lists to apply these functions in order. ⏎ - Make `attention_matadata_builder._add_seq_group` private, an …[truncated]

### L3-9e0b558a09  (L3, 2024-07-23, sha 9e0b558a0923, PR #6528)
TITLE: [Misc] Support FP8 kv cache scales from compressed-tensors (#6528)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+11/-12); tests/quantization/test_compressed_tensors.py (+7/-0); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py (+58/-5); vllm/model_executor/layers/quantization/compressed_tensors/utils.py (+17/-0); vllm/model_executor/layers/quantization/fp8.py (+5/-58); vllm/model_executor/layers/quantization/kv_cache.py (+78/-0); vllm/model_executor/models/llama.py (+10/-0)
LABELS: ready
BODY: Adding the logic for loading models with quantized kv cache scales, generated using `compressed-tensors` framework. ⏎  ⏎ - `CompressedTensorsConfig` now has an optional `kv_cache_scheme` argument. As of the next `compressed-tensors` release, the key contains the information about the properties of the quantized kv cache.  ⏎ - Added an interface `BaseKVCacheMethod` that both `Fp8KVCacheMethod` and `CompressedTensorsKVCacheMethod` inherits from, to pr …[truncated]

### L3-5e8ca973eb  (L3, 2024-07-24, sha 5e8ca973ebd5, PR #6708)
TITLE: [Bugfix] fix flashinfer cudagraph capture for PP (#6708)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/worker/model_runner.py (+7/-7); tests/distributed/test_pipeline_parallel.py (+24/-0)
LABELS: ready
BODY: The previous cudagraph capture assumes only a single pass through the batch sizes and clobbers the pre-allocated max_batch_size `indptr` and `last_page_len` tensors. However with PP, we need to capture the graphs for each VE, thus the clobbered tensors results in a size mismatch in the flashinfer wrapper. ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-5448f67635  (L3, 2024-07-24, sha 5448f6763557, PR #6712)
TITLE: [Core] Tweaks to model runner/input builder developer APIs (#6712)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+19/-16); vllm/worker/embedding_model_runner.py (+3/-1); vllm/worker/model_runner.py (+87/-47)
LABELS: ready
BODY: Fixes some pain points with extensibility for new APIs. ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-0e63494cf3  (L3, 2024-07-24, sha 0e63494cf334, PR #6667)
TITLE: Add fp8 support to `reshape_and_cache_flash` (#6667)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend
FILES: csrc/cache_kernels.cu (+45/-30); vllm/attention/backends/flash_attn.py (+2/-0); vllm/attention/backends/flashinfer.py (+2/-0); csrc/cache.h (+2/-1); csrc/torch_bindings.cpp (+2/-1); tests/kernels/test_cache.py (+34/-8); vllm/_custom_ops.py (+4/-1); vllm/utils.py (+7/-2)
LABELS: ready
BODY: Preparation for fp8 support in flash attn-based backends. ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-309aaef825  (L3, 2024-07-24, sha 309aaef8255f, PR #6757)
TITLE: [Bugfix] Fix decode tokens w. CUDA graph (#6757)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+10/-2); vllm/attention/backends/flashinfer.py (+10/-1); vllm/attention/backends/utils.py (+10/-1); tests/worker/test_model_runner.py (+1/-0)
LABELS: ready
ISSUES: #6703 [Bug]: Flash-attn on-GPU advance step optimization bug with spec decode on LLama 405B
BODY: Fixes #6703. ⏎ This is not related to speculative decoding particularly but not sure why no unit test failed. Will try to add a unit test in this or a follow-up PR. ⏎  ⏎ cc @cadedaniel @alexm-neuralmagic

### L3-14dbd5a767  (L3, 2024-07-26, sha 14dbd5a7674e, PR #6451)
TITLE: [Model] H2O Danube3-4b (#6451)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.paged.python_wrapper
FILES: csrc/attention/attention_kernels.cu (+6/-0); vllm/attention/ops/paged_attn.py (+1/-1); .buildkite/run-cpu-test.sh (+1/-1); benchmarks/kernels/benchmark_paged_attention.py (+1/-1); benchmarks/kernels/benchmark_rope.py (+1/-1); tests/kernels/test_attention.py (+3/-1); tests/kernels/test_cache.py (+7/-1); tests/kernels/test_pos_encoding.py (+1/-1); tests/models/test_danube3_4b.py (+52/-0); vllm/utils.py (+6/-0)
LABELS: ready
BODY: This PR mainly focuses on adding a head size of 120 for GPU inference, to support [h2oai/h2o-danube3-4b-base](https://huggingface.co/h2oai/h2o-danube3-4b-base). ⏎  ⏎ Head sizes that are not a multiple of 16 aren't compatible with `fp8` kv cache, so those tests are skipped. ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-52f07e3dec  (L3, 2024-07-26, sha 52f07e3dec2b, PR #5871)
TITLE: [Hardware][TPU] Implement tensor parallelism with Ray (#5871)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/pallas.py (+2/-2); requirements-tpu.txt (+1/-0); vllm/engine/llm_engine.py (+8/-2); vllm/executor/ray_tpu_executor.py (+313/-0); vllm/worker/tpu_model_runner.py (+31/-11); vllm/worker/tpu_worker.py (+10/-6)
LABELS: tpu, ready
BODY: This PR implements Ray TPU executor for distributed inference support on TPU.

### L3-fad5576c58  (L3, 2024-07-27, sha fad5576c5886, PR #6856)
TITLE: [TPU] Reduce compilation time & Upgrade PyTorch XLA version  (#6856)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: vllm/attention/backends/pallas.py (+0/-1); Dockerfile.tpu (+1/-1); docs/source/getting_started/tpu-installation.rst (+8/-1); vllm/distributed/device_communicators/tpu_communicator.py (+2/-1); vllm/worker/tpu_model_runner.py (+13/-2); vllm/worker/tpu_worker.py (+0/-1)
LABELS: tpu
BODY: This PR bumps up the PyTorch XLA version to 0726 and utilizes the new dynamic shape support to reduce the compilation time. When the XLA graphs are already cached in the disk, this reduces the compilation time from 30 mins to 5 mins.

### L3-9a7e2d0534  (L3, 2024-07-29, sha 9a7e2d053405, PR #6786)
TITLE: [Bugfix] Allow vllm to still work if triton is not installed. (#6786)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.paged.python_wrapper
FILES: vllm/attention/ops/paged_attn.py (+4/-1); requirements-cpu.txt (+0/-1); requirements-openvino.txt (+0/-2); requirements-tpu.txt (+0/-1); tests/kernels/test_sampler.py (+4/-3); vllm/model_executor/layers/fused_moe/__init__.py (+15/-7); vllm/model_executor/layers/ops/sample.py (+1/-13); vllm/model_executor/layers/quantization/fp8.py (+2/-2); vllm/model_executor/layers/sampler.py (+5/-1); vllm/model_executor/sampling_metadata.py (+1/-1); (+3 more)
LABELS: ready
BODY: We are currently needing to add `triton` as a dependency to all of the non-CUDA backends. This is because importing triton is still performed in various places throughout the library regardless of the backend.  ⏎  ⏎ This PR adds a function `maybe_import_triton` which will check to see if Triton is available in the environment. If it is not, it will replace Triton with a mocked up version that allows all the vLLM code to be imported.  ⏎  ⏎ An alternat …[truncated]

### L3-cbbc904470  (L3, 2024-07-30, sha cbbc90447066, PR #6914)
TITLE: [Kernel] Squash a few more warnings (#6914)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+2/-2); csrc/quantization/aqlm/gemm_kernels.cu (+0/-2); csrc/quantization/fp8/amd/quant_utils.cuh (+2/-0); csrc/quantization/fp8/nvidia/quant_utils.cuh (+2/-0); csrc/quantization/squeezellm/quant_cuda_kernel.cu (+2/-1)
LABELS: ready
BODY: Squash a few minor warnings across a couple of kernels.  ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-c32ab8be1a  (L3, 2024-07-31, sha c32ab8be1ada, PR #6964)
TITLE: [Speculative decoding] Add serving benchmark for llama3 70b + speculative decoding (#6964)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/nightly-benchmarks/tests/serving-tests.json (+22/-1)
LABELS: ready
BODY: * Use `turboderp/Qwama-0.5B-Instruct` open model with QPS=2

### L3-7e0861bd0b  (L3, 2024-08-01, sha 7e0861bd0bb2, PR #6951)
TITLE: [CI/Build] Update PyTorch to 2.4.0 (#6951)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1); Dockerfile (+1/-1); requirements-build.txt (+1/-1); requirements-cuda.txt (+4/-4); .buildkite/test-pipeline.yaml (+3/-3); .github/workflows/publish.yml (+1/-1); pyproject.toml (+1/-1); vllm/model_executor/layers/ops/sample.py (+1/-1)
LABELS: ready
ISSUES: #6944 [Feature]: when will support torch2.4 use it out of box?
BODY: This PR updates pytorch to version 2.4. All dependencies that depend on pytorch have also been updated.

### L3-805a8a75f2  (L3, 2024-08-01, sha 805a8a75f2f1, PR #7022)
TITLE: [Misc] Support attention logits soft-capping with flash-attn (#7022)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.fork_pip, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: requirements-cuda.txt (+1/-1); vllm/attention/backends/abstract.py (+1/-0); vllm/attention/backends/blocksparse_attn.py (+3/-0); vllm/attention/backends/flash_attn.py (+10/-11); vllm/attention/backends/flashinfer.py (+5/-9); vllm/attention/backends/ipex_attn.py (+6/-2); vllm/attention/backends/pallas.py (+4/-0); vllm/attention/backends/rocm_flash_attn.py (+8/-2); vllm/attention/backends/torch_sdpa.py (+6/-2); vllm/attention/backends/utils.py (+0/-9); (+4 more)
LABELS: ready
BODY: This PR adds support for attention logits soft-capping in the FlashAttention backend. ⏎  ⏎ This is done by ⏎ 1. Moving `logits_soft_cap` from `AttentionMetadata` to `AttentionImpl`. ⏎ 2. Using `vllm-flash-attn == 2.6.1` which added support for soft-capping. ⏎  ⏎ Note that vllm-flash-attn v2.6.1 uses PyTorch 2.4.0, so the PR must be merged after #6951 ⏎  ⏎ This will hopefully resolve many of the issues in running Gemma2 models.

### L3-954f7305a1  (L3, 2024-08-01, sha 954f7305a106, PR #7008)
TITLE: [Kernel] Fix input for flashinfer prefill wrapper. (#7008)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+9/-2)
LABELS: ready
ISSUES: #362 RuntimeError: Out of workspace memory in AlignedAlloactor when there is a lot of GPU memory left
BODY: fix https://github.com/flashinfer-ai/flashinfer/issues/362

### L3-fb2c1c86c1  (L3, 2024-08-02, sha fb2c1c86c196, PR #7018)
TITLE: [Bugfix] Fix block table for seqs that have prefix cache hits (#7018)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+9/-3); tests/prefix_caching/test_prefix_caching.py (+56/-0)
LABELS: ready
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ When the prefix caching is enabled, it is possible that a some requests have cache hit but some don't. In this case, the flash attention will use the kv cache as the input. However, the block tables is not propagated correctly in this situation. For example, if we two sequences the block table is generated like [[1, 2, 3, 4], [0, 0, 0, 0]]. The non-cached sequence is always 0, and cause the flash attention gener …[truncated]

### L3-e9630458c7  (L3, 2024-08-05, sha e9630458c7b1, PR #6926)
TITLE: [SpecDecode] Support FlashInfer in DraftModelRunner (#6926)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/spec_decode/draft_model_runner.py (+47/-0)
LABELS: ready
ISSUES: #6885 [Bug]: Speculative Decoding + FlashInfer + benchmark_serving.py TransferEncodingError ISSUE
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ FIX #6885 ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ This PR resolves the `benchmark_serving.py` error for Speculative Decoding when using FlashInfer Backend. ⏎  ⏎ After fixing, we can see that the benchmark results are correctly displayed in both FlashAttn Backend and FlashInfer Backend. ⏎  ⏎ ### FLASH_ATTN backend ⏎ ``` ⏎ ============ Serving Benchmark Result ===== …[truncated]

### L3-82a1b1a82b  (L3, 2024-08-05, sha 82a1b1a82b1f, PR #6963)
TITLE: [Speculative decoding] Add periodic log with time spent in proposal/scoring/verification (#6963)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/spec_decode/test_spec_decode_worker.py (+48/-20); vllm/config.py (+7/-1); vllm/engine/arg_utils.py (+1/-0); vllm/spec_decode/spec_decode_worker.py (+54/-14); vllm/spec_decode/util.py (+15/-0)
LABELS: ready
BODY: * Log values whenever rejection sampler produces metrics ⏎ * Measures average time spent proposing each token, time spent scoring, and time spent verifiying.

### L3-8571ac4672  (L3, 2024-08-05, sha 8571ac4672c8, PR #7085)
TITLE: [Kernel] Update CUTLASS to 3.5.1 (#7085)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+3/-3); csrc/quantization/cutlass_w8a8/broadcast_load_epilogue_c3x.hpp (+111/-81); csrc/quantization/cutlass_w8a8/scaled_mm_c2x.cuh (+4/-4); csrc/quantization/cutlass_w8a8/scaled_mm_c3x.cu (+11/-19)
LABELS: ready
BODY: There is no git tag for CUTLASS 3.5.1 available yet, but talking to folks at Nvidia, it should be safe to update to current main. ⏎  ⏎ Notably, 3.5.1 contains a fix to the upstream `RowBroadcast`, which this PR replicates in `RowOrScalarBroadcast`. This replaces the fix we added in https://github.com/vllm-project/vllm/pull/6852.  ⏎  ⏎ <del>Heads up that I need to re-verify llama3.1 405b FP8 before landing (see https://github.com/vllm-project/vllm/iss …[truncated]

### L3-6e4852ce28  (L3, 2024-08-05, sha 6e4852ce28ad, PR #7001)
TITLE: [CI/Build] Suppress divide-by-zero and missing return statement warnings (#7001)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: csrc/attention/dtype_bfloat16.cuh (+8/-0); csrc/quantization/awq/dequantize.cuh (+1/-0); csrc/quantization/fp8/nvidia/quant_utils.cuh (+3/-2); csrc/quantization/gptq_marlin/gptq_marlin.cu (+12/-6)
LABELS: ready
BODY: See also #6994 ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-ef527be06c  (L3, 2024-08-05, sha ef527be06c40, PR #7172)
TITLE: [MISC] Use non-blocking transfer in prepare_input (#7172)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+12/-15); vllm/attention/backends/flashinfer.py (+11/-12); vllm/attention/backends/utils.py (+12/-15); vllm/worker/model_runner.py (+8/-7)
LABELS: ready
BODY: This PR uses non-blocking data transfer in prepare_input. This is beneficial because we transfer several tensors to GPU in prepare_input. Here are some benchmark results using Llama-3.1-8B-Instruct on 1xH100: ⏎  ⏎ Batching ⏎  ⏎ Command: ⏎ ``` ⏎ python3 benchmark_throughput.py \ ⏎     --model meta-llama/Meta-Llama-3.1-8B-Instruct \ ⏎     --backend vllm \ ⏎     --input-len 292 \ ⏎     --output-len 579 \ ⏎     --num-prompts 1000 ⏎ ``` ⏎ Result (I observed some v …[truncated]

### L3-fd95e026e0  (L3, 2024-08-06, sha fd95e026e0f9, PR #4942)
TITLE: [Core] Subclass ModelRunner to support cross-attention & encoder sequences (towards eventual encoder/decoder model support) (#4942)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/layer.py (+1/-1); vllm/attention/selector.py (+111/-12); .buildkite/test-pipeline.yaml (+3/-1); examples/offline_inference_encoder_decoder.py (+99/-0); tests/conftest.py (+185/-37); tests/core/test_scheduler.py (+4/-26); tests/core/test_scheduler_encoder_decoder.py (+99/-0); tests/core/utils.py (+64/-35); tests/distributed/test_basic_distributed_correctness_enc_dec.py (+101/-0); tests/kernels/test_attention_selector.py (+2/-2); (+23 more)
LABELS: ready
BODY: This PR is a step towards encoder/decoder model support. This PR creates a specialized ModelRunner subclass for encoder/decoder models; it differs from the base ModelRunner class primarily in that it (1) expects each SequenceGroup to have an encoder sequence, and (2) it properly constructs the AttentionMetadata structure to support both self- and cross-attention. ⏎  ⏎ ---------- ⏎  ⏎ A quick overview of the plan for supporting encoder/decoder models  …[truncated]

### L3-e53dfd3eaf  (L3, 2024-08-07, sha e53dfd3eafd9, PR #7284)
TITLE: [Kernel] Fix Flashinfer Correctness (#7284)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+7/-3)
LABELS: ready
ISSUES: #7176 [Bug]: FlashInfer backend generate bad output 
BODY: FIX #7176

### L3-e02ac55617  (L3, 2024-08-08, sha e02ac5561748, PR #7162)
TITLE: [Performance] Optimize e2e overheads: Reduce python allocations (#7162)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+5/-1); vllm/attention/backends/utils.py (+10/-2); vllm/block.py (+45/-3); vllm/core/block_manager_v1.py (+15/-12); vllm/core/scheduler.py (+127/-44); vllm/model_executor/__init__.py (+3/-1); vllm/model_executor/sampling_metadata.py (+71/-10); vllm/outputs.py (+1/-1); vllm/sequence.py (+24/-4); vllm/utils.py (+38/-0); (+1 more)
LABELS: ready
BODY: This PR introduces a bunch of end-to-end overhead optimizations to reduce python object allocations/deallocations over scheduler iterations. In particular: ⏎ 1. Avoid python object allocations for "InterDataForSeqGroup" objects: These objects have lot of fields, and most of the them were allocated dynamically. In this PR, these objects are pre-allocated and reused between runs. The pre-allocation is done per "number of sequences per group" (to sup …[truncated]

### L3-999ef0b917  (L3, 2024-08-09, sha 999ef0b917aa, PR #7377)
TITLE: [Misc] Add numpy implementation of `compute_slot_mapping` (#7377)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/utils.py (+41/-12)
LABELS: ready
BODY: For prefill, the python implementation of `compute_slot_mapping` is inefficient as we loop over large lists. This PR adds a numpy implementation of the same operation, letting us leverage numpy vectorized instructions for up to 3x speedup with large lists. If the number of slots is small, we still use the python implementation as the overheads from creating numpy arrays are too great and actually cause a slowdown. ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-386087970a  (L3, 2024-08-11, sha 386087970ac2, PR #4773)
TITLE: [CI/Build] build on empty device for better dev experience (#4773)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_pip
FILES: requirements-cuda.txt (+2/-2); setup.py (+19/-5)
BODY: This PR enables build of a platform-agnostic wheel which is installable also on macos. The idea is to improve the dev-experience for creating projects that import and use vLLM. ⏎ **Important:** This wheel does *not* enable running of vllm on mac, but does allow to import it.  ⏎  ⏎ The PR doesn't entirely fix issues #212, #695, #1397, #1921, but it's a step forward.

### L3-ec2affa8ae  (L3, 2024-08-12, sha ec2affa8ae26, PR #7319)
TITLE: [Kernel] Flashinfer correctness fix for v0.1.3 (#7319)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.v0_backend
FILES: Dockerfile (+1/-1); vllm/attention/backends/flashinfer.py (+19/-18); .buildkite/test-pipeline.yaml (+0/-5)
LABELS: ready
BODY: Reported by @felixzhu555 , `is_profile_run` is buggy in flashinfer backend, which will fail flashinfer v0.1.3. This PR fixes this, and update CI flashinfer version to v1.0.3.

### L3-cfba4def5d  (L3, 2024-08-12, sha cfba4def5d42, PR #7425)
TITLE: [Bugfix] Fix logit soft cap in flash-attn backend (#7425)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+1/-0)
ISSUES: #7419 [Bug]: `gemma2` does not work with flash attention
BODY: Fixes #7419 ⏎  ⏎ After this PR, the GSM8K result looks correct: ⏎  ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.868|±  |0.0215| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.860|±  |0.0220| ⏎ ```

### L3-a046f86397  (L3, 2024-08-12, sha a046f86397c0, PR #7208)
TITLE: [Core/Bugfix] Add FP8 K/V Scale and dtype conversion for prefix/prefill Triton Kernel (#7208)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.triton.prefix_prefill, L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+3/-0); vllm/attention/backends/xformers.py (+3/-0); vllm/attention/ops/ipex_attn.py (+1/-0); vllm/attention/ops/paged_attn.py (+6/-0); vllm/attention/ops/prefix_prefill.py (+73/-24); docs/source/quantization/fp8_e4m3_kvcache.rst (+0/-2); docs/source/quantization/fp8_e5m2_kvcache.rst (+0/-2); tests/basic_correctness/test_chunked_prefill.py (+96/-8); tests/kernels/test_prefix_prefill.py (+26/-7); vllm/config.py (+0/-4)
LABELS: ready
ISSUES: #4381 [Bug]: Chunked prefill doesn't seem to work when --kv-cache-dtype fp8
BODY: Fix the FP8 Triton kernel issue. Should enable FP8 KV Cache to be used with: ⏎ 1. chunked prefill ⏎ 2. prefix caching ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/4381 https://github.com/vllm-project/vllm/issues/3880 https://github.com/vllm-project/vllm/issues/3156 https://github.com/vllm-project/vllm/issues/3880 ⏎  ⏎ TODO: ⏎  ⏎  ⏎ Notes: ⏎ 1. @comaniac mentions upcoming flashinfer support, but I think supporting it in Triton as fallback is good for …[truncated]

### L3-4d2dc5072b  (L3, 2024-08-13, sha 4d2dc5072b8c, PR #7102)
TITLE: [hardware] unify usage of is_tpu to current_platform.is_tpu() (#7102)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/selector.py (+2/-3); vllm/config.py (+4/-3); vllm/executor/ray_utils.py (+3/-2); vllm/model_executor/custom_op.py (+3/-2); vllm/model_executor/layers/rotary_embedding.py (+2/-2); vllm/model_executor/model_loader/loader.py (+3/-3); vllm/platforms/__init__.py (+12/-9); vllm/utils.py (+0/-9)
LABELS: tpu
BODY: 

### L3-54bd9a03c4  (L3, 2024-08-15, sha 54bd9a03c4b2, PR #7536)
TITLE: register custom op for flash attn and use from torch.ops (#7536)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+129/-26); vllm/attention/backends/flashinfer.py (+3/-3); .buildkite/test-pipeline.yaml (+7/-0); tests/compile/test_full_graph.py (+20/-0); tests/kernels/test_flash_attn.py (+61/-12)
BODY: 

### L3-f366f6339b  (L3, 2024-08-16, sha f366f6339b1d, PR #7571)
TITLE: [spec decode] [4/N] Move update_flash_attn_metadata to attn backend (#7571)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+3/-0); vllm/attention/backends/flash_attn.py (+45/-0); vllm/spec_decode/draft_model_runner.py (+1/-33)
LABELS: ready
BODY: for #7000  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-f710fb5265  (L3, 2024-08-19, sha f710fb5265ab, PR #7137)
TITLE: [Core] Use flashinfer sampling kernel when available (#7137)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen
FILES: Dockerfile (+1/-1); .buildkite/test-pipeline.yaml (+3/-1); tests/samplers/test_sampler.py (+36/-1); vllm/envs.py (+5/-0); vllm/model_executor/layers/sampler.py (+85/-25)
LABELS: ready
BODY: Flashinfer contains a combined kernel `top_k_top_p_sampling_from_probs`, and it is way faster than the sorting kernels used currently. This will eliminate the timely `_apply_top_k_top_p` function and reduce the GPU time of sampler. ⏎  ⏎ main: ⏎ <img width="1140" alt="图片" src="https://github.com/user-attachments/assets/53ba9fac-9b77-41d6-8c71-0547883a155e"> ⏎  ⏎ this PR: ⏎ <img width="1140" alt="图片" src="https://github.com/user-attachments/assets/16fc17 …[truncated]

### L3-47b65a5508  (L3, 2024-08-19, sha 47b65a550866, PR #7000)
TITLE: [core] Multi Step Scheduling (#7000)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: .buildkite/test-pipeline.yaml (+9/-0); tests/multi_step/__init__.py (+0/-0); tests/multi_step/test_correctness.py (+85/-0); tests/worker/test_model_input.py (+77/-0); vllm/engine/arg_utils.py (+6/-1); vllm/engine/async_llm_engine.py (+127/-8); vllm/executor/gpu_executor.py (+10/-4); vllm/executor/ray_gpu_executor.py (+3/-0); vllm/sequence.py (+5/-5); vllm/worker/model_runner_base.py (+34/-12); (+3 more)
LABELS: ready
BODY: Adds initial multi step scheduling support to vLLM. ⏎ RFC: https://github.com/vllm-project/vllm/issues/6854 ⏎  ⏎ **Current Status**: ⏎  ⏎ **8/16: Initial support for chunked prefill thanks to @varun-sundar-rabindranath** ⏎   ⏎ 8/14: Ready for another round of reviews! ~~please review https://github.com/vllm-project/vllm/pull/7452~~ ⏎ 8/8: multi-node working ⏎ 8/6: PP+TP working; PP+ray fixed; ~~a few single GPU perf regressions (easy fix)~~ ⏎ 8/2 PP works  …[truncated]

### L3-3b682179dd  (L3, 2024-08-20, sha 3b682179dd48, PR #7663)
TITLE: [Core] Add `AttentionState` abstraction (#7663)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: vllm/attention/backends/abstract.py (+50/-1); vllm/attention/backends/blocksparse_attn.py (+6/-1); vllm/attention/backends/flash_attn.py (+6/-1); vllm/attention/backends/flashinfer.py (+164/-1); vllm/attention/backends/ipex_attn.py (+5/-0); vllm/attention/backends/openvino.py (+6/-1); vllm/attention/backends/pallas.py (+5/-0); vllm/attention/backends/rocm_flash_attn.py (+6/-1); vllm/attention/backends/torch_sdpa.py (+5/-0); vllm/attention/backends/utils.py (+73/-2); (+6 more)
LABELS: ready
BODY: Adds an `AttentionState` abstraction intended to hold attention backend-specific objects reused for the lifetime of the model runner. This allows us to remove all of the special casing for flashinfer in model runner. ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-53328d7536  (L3, 2024-08-21, sha 53328d7536b3, PR #7509)
TITLE: [BUG] fix crash on flashinfer backend with cudagraph disabled, when attention group_size not in [1,2,4,8] (#7509)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+4/-2); tests/kernels/test_flashinfer.py (+5/-2)
LABELS: ready
BODY: when I use flashinfer backend and disable cuda graph, load a model with attention group_size=6, vllm crashs and shows the following log: ⏎ ![screenshot-20240814-161315](https://github.com/user-attachments/assets/39224bf6-821b-487e-877c-10ed92df4139) ⏎  ⏎ This error consistently occurs under the following conditions： ⏎ 1. user use flashinfer attention backend explicitly, (set env VLLM_ATTENTION_BACKEND="FLASHINFER") ⏎ 2. the model use GQA, and group si …[truncated]

### L3-9984605412  (L3, 2024-08-21, sha 9984605412de, PR #7477)
TITLE: [AMD][CI/Build] Disambiguation of the function call for ROCm 6.2 headers compatibility (#7477)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: csrc/attention/attention_utils.cuh (+1/-1)
LABELS: rocm, ready
BODY: This is to fix build issues with ROCm 6.2 where there is an ambiguous function call which causes an error due to arguments mismatch ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-666ad0aa16  (L3, 2024-08-22, sha 666ad0aa16f0, PR #7705)
TITLE: [ci] Cleanup & refactor Dockerfile to pass different Python versions and sccache bucket via build args (#7705)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile (+26/-35)
LABELS: ready
BODY: - Clean up unused commands ⏎ - Pack dependencies installation into as few commands as possible so Docker can cache it better ⏎ - Install flashinfer based on Python version passed in instead of fixing it to `3.10` ⏎ - Make `SCCACHE_BUCKET` and `SCCACHE_REGION` build arg (with default set to vllm's bucket and region)

### L3-9606c7197d  (L3, 2024-08-27, sha 9606c7197df0, PR #7887)
TITLE: Revert #7509 (#7887)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+2/-4)
LABELS: ready
BODY: Revert #7509 ⏎  ⏎ This should also cover group size 6.  ⏎  ⏎ cc @elfiegg @learninmou @yzh119

### L3-b98cc28f91  (L3, 2024-08-28, sha b98cc28f91aa, PR #7798)
TITLE: [Core][Kernels] Use FlashInfer backend for FP8 KV Cache when available. (#7798)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.dispatch.selector
FILES: vllm/attention/backends/flashinfer.py (+23/-6); vllm/attention/selector.py (+4/-0); tests/kernels/test_flashinfer.py (+222/-6)
LABELS: ready
BODY: This PR enables FlashInfer when `--kv-cache-dtype=fp8` is set.  ⏎  ⏎  ⏎ Reason- on H200 we see 20% perf improvement compared to Xformers backend with `paged_attention_v1` kernel that uses fp8 datatype.  ⏎  ⏎ This PR makes use of the fp8 support added by flashinfer in [v0.1.4]. (https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.1.4) ⏎  ⏎  ⏎ Benchmarks with `benchmark_throughput.py` for `neuralmagic/Meta-Llama-3-70B-Instruct-FP8-KV` ⏎  ⏎ ```markdow …[truncated]

### L3-ef99a78760  (L3, 2024-08-28, sha ef99a7876089, PR #7982)
TITLE: Revert "[Core][Kernels] Use FlashInfer backend for FP8 KV Cache when available." (#7982)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.dispatch.selector
FILES: vllm/attention/backends/flashinfer.py (+6/-23); vllm/attention/selector.py (+0/-4); tests/kernels/test_flashinfer.py (+6/-222)
BODY: Reverts vllm-project/vllm#7798

### L3-6b3421567d  (L3, 2024-08-29, sha 6b3421567d7a, PR #7985)
TITLE: [Core][Kernels] Enable FP8 KV Cache with Flashinfer backend.  + BugFix for kv_cache_dtype=auto (#7985)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.dispatch.selector
FILES: vllm/attention/backends/flashinfer.py (+24/-6); vllm/attention/selector.py (+4/-0); tests/kernels/test_flashinfer.py (+222/-6)
LABELS: ready
BODY: Previous PR with dicussions: https://github.com/vllm-project/vllm/pull/7798 ⏎  ⏎ Revert PR due to bug with `--kv-cache-dtype=auto` : https://github.com/vllm-project/vllm/pull/7982 ⏎  ⏎  ⏎ `--kv-cache-dtype=fp8` ⏎ ``` ⏎ model_str="neuralmagic/Meta-Llama-3-8B-Instruct-FP8" ⏎ model = LLM(model=model_str, quantization="fp8",kv_cache_dtype="fp8") ⏎ prompts = [ ⏎     "Hello, my name is", ⏎     "The president of the United States is", ⏎     "The capital of France i …[truncated]

### L3-2148441fd3  (L3, 2024-08-30, sha 2148441fd371, PR #7613)
TITLE: [TPU] Support single and multi-host TPUs on GKE (#7613)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/pallas.py (+4/-1); requirements-tpu.txt (+1/-1); vllm/distributed/device_communicators/tpu_communicator.py (+25/-2); vllm/executor/ray_tpu_executor.py (+15/-0); vllm/executor/ray_utils.py (+29/-0)
LABELS: tpu
BODY: Fixes a few issues with vLLM on TPUs using GKE and RayServe: ⏎ - Installs ray dashboard and serve libraries in Docker image ⏎ - Calculates the number of nodes based on current placement group instead of the entire Ray cluster (which may include non-TPU nodes) ⏎ - Ensures TPU-specific environment variables are set correctly for each Ray worker

### L3-2684efc467  (L3, 2024-08-30, sha 2684efc4678e, PR #8035)
TITLE: [TPU][Bugfix] Fix tpu type api (#8035)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/pallas.py (+4/-1)
LABELS: tpu
BODY: 

### L3-622f8abff8  (L3, 2024-08-30, sha 622f8abff8e1, PR #8013)
TITLE: [Bugfix] bugfix and add model test for flashinfer fp8 kv cache. (#8013)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+13/-5); tests/models/test_fp8kv_flashinfer.py (+96/-0)
LABELS: ready
BODY: This addresses the bug described in #8009. Fix is to consistently replace the fp8 type right before we call into Flashinfer API.  ⏎  ⏎ Please note that this test is disabled because FP8 numerics are resulting in slightly different strings each run. But can be evaluated that the strings are essentially the same in English. ⏎  ⏎  ⏎ --- ⏎ [details omitted]

### L3-e6a26ed037  (L3, 2024-09-01, sha e6a26ed0376f, PR #7244)
TITLE: [SpecDecode][Kernel] Flashinfer Rejection Sampling (#7244)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen
FILES: Dockerfile (+1/-1); tests/samplers/test_rejection_sampler.py (+97/-19); tests/samplers/test_typical_acceptance_sampler.py (+32/-18); tests/spec_decode/test_spec_decode_worker.py (+2/-3); vllm/envs.py (+1/-0); vllm/model_executor/layers/rejection_sampler.py (+141/-43); vllm/model_executor/layers/spec_decode_base_sampler.py (+25/-18); vllm/model_executor/layers/typical_acceptance_sampler.py (+4/-3); vllm/spec_decode/spec_decode_worker.py (+3/-4)
LABELS: ready
BODY: End to end Speculative Decoding Performance (request latency): ⏎ `Draft: LLama-160M, Target: Vicuna-7B, batch size=8, input_len=256, output_len=512` ⏎  ⏎ Before this PR: ⏎ ``` ⏎ Avg latency: 5.9652480507269505 seconds ⏎ 10% percentile latency: 5.729408229794354 seconds ⏎ 25% percentile latency: 5.794497653492726 seconds ⏎ 50% percentile latency: 5.954964595614001 seconds ⏎ 75% percentile latency: 6.124162045423873 seconds ⏎ 90% percentile latency: 6.175716 …[truncated]

### L3-e39ebf5cf5  (L3, 2024-09-05, sha e39ebf5cf5ec, PR #8173)
TITLE: [Core/Bugfix] Add query dtype as per FlashInfer API requirements. (#8173)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+8/-1); tests/kernels/test_flashinfer.py (+2/-1)
LABELS: ready
BODY: As per [FlashInfer doc](https://github.com/flashinfer-ai/flashinfer/blob/45eac04f9420b2372737d16d51f4d07bf928d293/python/flashinfer/decode.py#L458), `q_data_type` will be set to KV cache's `data_type` if not explicitly set, which results in RuntimeError: `BatchPrefillWithPagedKVCachePyTorchWrapper::BeginForward(at::Tensor, at::Tensor, at::Tensor, at::Tensor, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, at::Tensor)::<lambd …[truncated]

### L3-5faedf1b62  (L3, 2024-09-10, sha 5faedf1b6224, PR #8224)
TITLE: [Spec Decode] Move ops.advance_step to flash attn advance_step (#8224)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+15/-6); vllm/worker/multi_step_model_runner.py (+5/-14); vllm/spec_decode/draft_model_runner.py (+3/-13)
LABELS: ready
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ Addresses TODO in #7571   ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-22f3a4bc6c  (L3, 2024-09-10, sha 22f3a4bc6c68, PR #8340)
TITLE: [Bugfix] lookahead block table with cuda graph max capture (#8340)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+11/-1)
LABELS: ready
ISSUES: #8068 [Bug]: ValueError: could not broadcast input array from shape (513,) into shape (512,)
BODY: This PR fixes the issue described here: https://github.com/vllm-project/vllm/issues/8068. ⏎  ⏎ The problem is that multistep lookahead block allocation may allocate more blocks than what max capture is using for the cuda graphs. The solution here is to simply ignore these extra additional blocks, since the model won't execute more than the max capture size / max_model_len anyway.

### L3-1d5e397aa4  (L3, 2024-09-10, sha 1d5e397aa4d9, PR #8172)
TITLE: [Core/Bugfix] pass VLLM_ATTENTION_BACKEND to ray workers (#8172)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/executor/ray_gpu_executor.py (+3/-0)
LABELS: ready
BODY: Currently vllm will crash if using TP>2 with flashinfer backend and ray workers as the env var is not passed to workers. ⏎ Related PR https://github.com/vllm-project/vllm/pull/7928 ⏎ cc @WoosukKwon @comaniac  ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-94144e726c  (L3, 2024-09-10, sha 94144e726cfe, PR #8043)
TITLE: [CI/Build][Kernel] Update CUTLASS to 3.5.1 tag (#8043)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+13/-2)
LABELS: ready
BODY: Now that CUTLASS 3.5.1 is officially out, use the `v3.5.1` tag instead of the commit that we were on. ⏎  ⏎ This has the benefit of letting us use `GIT_SHALLOW=TRUE` again, which should speed up fetching CUTLASS. ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-7de49aa86c  (L3, 2024-09-12, sha 7de49aa86c7f, PR #8384)
TITLE: [torch.compile] hide slicing under custom op for inductor (#8384)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+71/-34); tests/compile/test_full_graph.py (+3/-1)
BODY: see https://github.com/pytorch/pytorch/issues/131192  ⏎  ⏎ when inductor sees a view being mutated, it will copy the tensor. ⏎  ⏎ hiding the slicing operation under custom op solves the issue for inductor.

### L3-a6c0f3658d  (L3, 2024-09-12, sha a6c0f3658da4, PR #7928)
TITLE: [multi-step] add flashinfer backend (#7928)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.dispatch.abstract_interface
FILES: csrc/ops.h (+15/-4); csrc/torch_bindings.cpp (+13/-2); vllm/_custom_ops.py (+29/-9); vllm/attention/backends/abstract.py (+3/-1); vllm/attention/backends/flash_attn.py (+9/-9); vllm/attention/backends/flashinfer.py (+77/-10); vllm/worker/multi_step_model_runner.py (+16/-21); csrc/prepare_inputs/advance_step.cu (+200/-25); tests/multi_step/test_correctness_async_llm.py (+9/-3)
LABELS: ready
ISSUES: #8194 [Bug]: vllm.engine.async_llm_engine.AsyncEngineDeadError
BODY: @WoosukKwon  ⏎ FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ FIX #8194 (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-019877253b  (L3, 2024-09-12, sha 019877253be4, PR #8427)
TITLE: [Bugfix] multi-step + flashinfer: ensure cuda graph compatible  (#8427)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+11/-1)
LABELS: ready
BODY: This PR ports multi-step cuda graph block table fix from the flash_attn backend to flashinfer backend

### L3-851725202a  (L3, 2024-09-13, sha 851725202af3, PR #8365)
TITLE: [Hardware][intel GPU] bump up ipex version to 2.3 (#8365)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: vllm/attention/backends/ipex_attn.py (+6/-2); Dockerfile.xpu (+10/-2); requirements-xpu.txt (+5/-4); vllm/_ipex_ops.py (+29/-69); vllm/model_executor/layers/activation.py (+9/-6); vllm/model_executor/layers/layernorm.py (+1/-4)
LABELS: intel-gpu
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ ipex-xpu [release ](https://github.com/intel/intel-extension-for-pytorch/releases/tag/v2.3.110%2Bxpu)2.3 version recently, this pr update ipex-xpu dependency to latest. Also fix below issues: ⏎ 1. support Arc graphic grad. ⏎ 2. support GQA model like llama3-8B ⏎ 3. support some new kernels like gelu_quick ⏎ 4. update oneAPI version to 2024.2.1 ⏎  ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE …[truncated]

### L3-1ef0d2efd0  (L3, 2024-09-13, sha 1ef0d2efd07f, PR #8310)
TITLE: [Kernel][Hardware][Amd]Custom paged attention kernel for rocm (#8310)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:production-kernel-provenance
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake, L3.rocm.custom_paged, L3.rocm.rocm_flash_attn_v0
FILES: CMakeLists.txt (+23/-0); csrc/rocm/attention.cu (+1038/-0); setup.py (+3/-0); vllm/_custom_ops.py (+27/-0); vllm/attention/backends/rocm_flash_attn.py (+70/-14); csrc/rocm/ops.h (+13/-0); csrc/rocm/torch_bindings.cpp (+33/-0); tests/kernels/test_attention.py (+164/-2)
LABELS: rocm
BODY: This PR adds custom paged attention kernel support for rocm. ⏎  ⏎ - add a separate folder for rocm kernel, which enables more specialized kernels for rocm in the future. ⏎ - add custom paged attention for rocm ⏎ - enable rocm custom paged attention kernel for specific size and dtype. ⏎ - add unit test ⏎  ⏎ Todo: ⏎ - add fp8 kv cache support for rocm custom paged attention. ⏎ - add navi support for rocm custom paged attention. ⏎  ⏎ Performance improvement (l …[truncated]

### L3-1009e93c5d  (L3, 2024-09-17, sha 1009e93c5d63, PR #7631)
TITLE: [Encoder decoder] Add cuda graph support during decoding for encoder-decoder models (#7631)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+13/-4); vllm/attention/backends/flashinfer.py (+9/-3); vllm/attention/backends/utils.py (+107/-6); .buildkite/test-pipeline.yaml (+7/-0); tests/encoder_decoder/__init__.py (+0/-0); tests/encoder_decoder/test_e2e_correctness.py (+98/-0); tests/worker/test_encoder_decoder_model_runner.py (+160/-22); vllm/config.py (+8/-33); vllm/engine/arg_utils.py (+4/-1); vllm/entrypoints/llm.py (+4/-4); (+5 more)
LABELS: ready
BODY: In this PR we are adding support for CUDA Graph capture & replay during the decoding phase for encoder-decoder models. Currently this support is missing for encoder-decoder models. To that end this PR makes the following changes ⏎  ⏎ 1. Updates the CUDA capture process in model_runner.py and CUDAGraphRunner to capture the additional inputs/attention metadata needed for encoder-decoder models. We also had to make changes to CommonAttentionState in v …[truncated]

### L3-6ffa3f314c  (L3, 2024-09-18, sha 6ffa3f314c59, PR #8534)
TITLE: [CI/Build] Avoid CUDA initialization (#8534)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.triton.prefix_prefill, L3.rocm.rocm_flash_attn_v0, L3.dispatch.selector
FILES: vllm/attention/backends/rocm_flash_attn.py (+2/-1); vllm/attention/ops/blocksparse_attention/interface.py (+2/-3); vllm/attention/ops/prefix_prefill.py (+1/-2); vllm/attention/selector.py (+2/-2); benchmarks/kernels/benchmark_layernorm.py (+3/-6); benchmarks/kernels/benchmark_moe.py (+3/-3); benchmarks/kernels/benchmark_paged_attention.py (+2/-5); benchmarks/kernels/benchmark_quant.py (+3/-6); benchmarks/kernels/benchmark_rope.py (+2/-4); tests/kernels/test_activation.py (+3/-6); (+45 more)
LABELS: ready
BODY: This PR removes the vast majority of `torch.cuda.is_available` and `torch.cuda.get_device_capability` calls, thus avoiding CUDA re-initialization errors when vLLM is run multiple times in tests. The only remaining ones are in the files that don't import vLLM.

### L3-9d104b5beb  (L3, 2024-09-18, sha 9d104b5beb7b, PR #8469)
TITLE: [CI/Build] Update Ruff version (#8469)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/utils.py (+2/-4); .github/workflows/ruff.yml (+2/-2); benchmarks/kernels/graph_machete_bench.py (+1/-3); format.sh (+2/-2); pyproject.toml (+2/-0); requirements-lint.txt (+1/-1); tests/conftest.py (+1/-4); tests/lora/conftest.py (+1/-4); tests/multimodal/test_base.py (+1/-1); tests/test_cache_block_hashing.py (+1/-4); (+17 more)
LABELS: ready
BODY: Updated ruff to 0.6.5, possible to also run `ruff check` ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-9cc373f390  (L3, 2024-09-19, sha 9cc373f39036, PR #8577)
TITLE: [Kernel][Amd] Add fp8 kv cache support for rocm custom paged attention (#8577)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.custom_paged, L3.rocm.rocm_flash_attn_v0
FILES: csrc/rocm/attention.cu (+161/-79); vllm/_custom_ops.py (+3/-1); vllm/attention/backends/rocm_flash_attn.py (+15/-13); csrc/rocm/ops.h (+2/-1); csrc/rocm/torch_bindings.cpp (+2/-1); tests/kernels/test_attention.py (+63/-188)
LABELS: ready
BODY: This is a follow-up PR of #8310 that adds the support of fp8 kv cache for rocm's custom paged attention. ⏎  ⏎ - two more parameter `k_scale` and `v_scale` added to the kernel signature. ⏎ - remove the requirement of `kv_cache_dtype` need to be `auto` ⏎ - unit test added. ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-9e5ec35b1f  (L3, 2024-09-19, sha 9e5ec35b1f82, PR #8474)
TITLE: [bugfix] [AMD] add multi-step advance_step to ROCmFlashAttentionMetadata (#8474)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+58/-1); vllm/worker/multi_step_model_runner.py (+1/-1)
LABELS: ready
ISSUES: #8472 [Bug]: AMD with multi-step enabled crashes
BODY: I don't have AMD GPUs and cannot test locally. We can also considering moving the `advance_step` inside flash_attn.py and rocm_flash_attn.py to `AttentionMetadata` as a default implementation since the code is identical. ⏎  ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/8472

### L3-71c60491f2  (L3, 2024-09-20, sha 71c60491f287, PR #8245)
TITLE: [Kernel] Build flash-attn from source (#8245)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.upstream_pip, L3.flash_attn.fork_pip, L3.flash_attn.fork_inline_cmake, L3.dispatch.selector
FILES: CMakeLists.txt (+73/-25); Dockerfile (+3/-0); requirements-cuda.txt (+0/-1); setup.py (+30/-8); vllm/attention/backends/flash_attn.py (+7/-2); vllm/attention/selector.py (+4/-4); .github/workflows/scripts/build.sh (+1/-0); .gitignore (+5/-0); cmake/utils.cmake (+1/-1)
LABELS: ready
ISSUES: #8002 [RFC]: Build `vllm-flash-attn` from source
BODY: This PR resolves #8002 and builds vllm-flash-attn from source. This is required for using torch nightly. ⏎  ⏎ This PR relies on the new CMake-based build system in vllm-flash-attn.  ⏎  ⏎ To make installation smoother, this process uses an additional CMake install step for putting the .so targets in the correct place instead of using `CMAKE_LIBRARY_OUTPUT_DIRECTORY` during build. ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE  …[truncated]

### L3-0e40ac9b7b  (L3, 2024-09-21, sha 0e40ac9b7b5d, PR #8699)
TITLE: [ci][build] fix vllm-flash-attn (#8699)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+3/-0); setup.py (+15/-0); vllm/vllm_flash_attn/.gitkeep (+0/-0)
BODY: 

### L3-530821d00c  (L3, 2024-09-23, sha 530821d00cb2, PR #8674)
TITLE: [Hardware][AMD] ROCm6.2 upgrade (#8674)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile.rocm (+19/-37); docs/source/getting_started/amd-installation.rst (+42/-23)
LABELS: rocm
BODY: This PR is to upgrade Dockerfile.rocm to support ROCm 6.2 (with torch 2.6) for AMD GPUs. ⏎  ⏎ - base image upgrade ⏎ - torch upgrade ⏎ - triton flash-attention branch upgrade ⏎ - ck flash-attention branch upgrade ⏎ - documentation update ⏎ - cleaned up 6.1 related steps ⏎  ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-e551ca1555  (L3, 2024-09-23, sha e551ca1555b6, PR #8729)
TITLE: [Hardware][CPU] Refactor CPU model runner (#8729)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/worker/cpu_model_runner.py (+193/-109)
LABELS: ready
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ - This PR refactors `cpu_model_runner.py` to use `ModelInputWithSamplingMetadata` and `ModelInputBuilder` like GPU and XPU model runner. ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-8df2dc3c88  (L3, 2024-09-27, sha 8df2dc3c8812, PR #8871)
TITLE: [TPU] Update pallas.py to support trillium (#8871)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/pallas.py (+1/-1)
LABELS: tpu
BODY: Description: These changes are needed to support trillium in vllm. ⏎  ⏎ Why it's needed: trillium does not have "lite" in the accelerator_type but it also does not use megacore. ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-a9b15c606f  (L3, 2024-09-27, sha a9b15c606fea, PR #8875)
TITLE: [torch.compile] use empty tensor instead of None for profiling (#8875)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.blocksparse.v0
FILES: vllm/attention/backends/blocksparse_attn.py (+4/-2); vllm/attention/backends/flash_attn.py (+4/-2); vllm/attention/backends/flashinfer.py (+3/-3); vllm/attention/backends/ipex_attn.py (+6/-3); vllm/attention/backends/pallas.py (+7/-5); vllm/attention/backends/rocm_flash_attn.py (+4/-2); vllm/attention/backends/torch_sdpa.py (+6/-3); vllm/attention/backends/xformers.py (+5/-3); tests/kernels/test_encoder_decoder_attn.py (+6/-2); vllm/worker/embedding_model_runner.py (+7/-1); (+5 more)
LABELS: ready
BODY: one step of https://github.com/vllm-project/vllm/issues/8821

### L3-c2ec430ab5  (L3, 2024-09-27, sha c2ec430ab571, PR #8378)
TITLE: [Core] Multi-Step + Single Step Prefills via Chunked Prefill code path (#8378)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+27/-5); vllm/attention/backends/flashinfer.py (+12/-8); csrc/prepare_inputs/advance_step.cu (+1/-1); tests/multi_step/test_correctness_async_llm.py (+9/-0); tests/multi_step/test_correctness_llm.py (+4/-0); vllm/config.py (+10/-3); vllm/core/block/block_table.py (+9/-4); vllm/core/block_manager_v1.py (+6/-1); vllm/core/block_manager_v2.py (+4/-1); vllm/core/embedding_model_block_manager.py (+3/-1); (+9 more)
LABELS: ready
BODY: Adds support for scheduling prompts with decodes in Multi-Step. This PR uses the Chunked-Prefill code path to this effect. ⏎  ⏎ Adding chunked-prompts to multi-step, in realization of a full chunked-prefill support, is a future-work and needs some investigation as it is not clear that it will have a positive performance impact.  ⏎  ⏎ With this PR decode sequences can run in every step and will not have to be interrupted by prefills. This has an effec …[truncated]

### L3-260024a374  (L3, 2024-09-27, sha 260024a3749f, PR #7824)
TITLE: [Bugfix][Intel] Fix XPU Dockerfile Build (#7824)
SOURCES: corpus:production-kernel-provenance
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: .buildkite/run-xpu-test.sh (+1/-1); .dockerignore (+3/-1); Dockerfile.xpu (+38/-9); requirements-common.txt (+1/-1); requirements-xpu.txt (+6/-2); setup.py (+2/-0); vllm/platforms/__init__.py (+12/-0); vllm/platforms/interface.py (+4/-0); vllm/platforms/xpu.py (+20/-0)
LABELS: intel-gpu, ready
BODY: The XPU Dockerfile consistently wasn't building, I've worked out some of the issues and assigned the correct IPEX XPU versions to be installed in the image. ⏎  ⏎ I also added support for the openai server version of xpu. I validated on a Max 1100. ⏎  ⏎ Because Ubuntu 20.04 isn't supported by the Triton XPU Backend anymore, 22.04 needs to be utilized for the newer python version. ⏎  ⏎ [details omitted]

### L3-bce324487a  (L3, 2024-10-01, sha bce324487a8e, PR #8975)
TITLE: [CI][SpecDecode] Fix spec decode tests, use flash attention backend for spec decode CI tests. (#8975)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test-pipeline.yaml (+0/-2); tests/spec_decode/test_multi_step_worker.py (+4/-1)
LABELS: ready
ISSUES: #5152 [Bug] [spec decode] [flash_attn]: CUDA illegal memory access when calling flash_attn_cuda.fwd_kvcache
BODY: fix https://github.com/vllm-project/vllm/issues/5152

### L3-1570203864  (L3, 2024-10-01, sha 157020386421, PR #8839)
TITLE: [Spec Decode] (1/2) Remove batch expansion (#8839)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.blocksparse.v0
FILES: vllm/attention/backends/blocksparse_attn.py (+6/-0); vllm/attention/backends/flash_attn.py (+29/-7); vllm/attention/backends/flashinfer.py (+0/-2); vllm/attention/backends/rocm_flash_attn.py (+8/-0); vllm/attention/backends/utils.py (+2/-1); vllm/attention/backends/xformers.py (+6/-0); .buildkite/test-pipeline.yaml (+1/-1); tests/samplers/test_sampler.py (+1/-1); tests/spec_decode/e2e/test_integration.py (+44/-0); tests/spec_decode/e2e/test_medusa_correctness.py (+49/-0); (+19 more)
LABELS: ready
BODY: This is a clean up PR for https://github.com/vllm-project/vllm/pull/5691 because that PR is stale. [RFC](https://docs.google.com/document/d/1I6H8bGYfwM8YappcP8HuoBhpS_pMlN0UIH2e_w9hv_M/edit) ⏎ Some constraints: ⏎ 1. This PR current does not support cuda graph. It will be addressed in the follow up PR. ⏎ 2. MQA scorer is only enabled for flash attention backend.  ⏎ 3. It does not support ngram because MQA scorer does not handle mixed batches.

### L3-f58d4fccc9  (L3, 2024-10-02, sha f58d4fccc9b2, PR #8192)
TITLE: [OpenVINO] Enable GPU support for OpenVINO vLLM backend (#8192)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/attention/backends/openvino.py (+32/-8); docs/source/getting_started/openvino-installation.rst (+29/-6); requirements-openvino.txt (+3/-2); vllm/envs.py (+6/-0); vllm/executor/openvino_executor.py (+53/-20); vllm/model_executor/model_loader/openvino.py (+10/-16); vllm/worker/openvino_model_runner.py (+6/-5); vllm/worker/openvino_worker.py (+307/-50)
LABELS: intel-gpu, ready
BODY: These changes add GPU device support for OpenVINO vLLM backend ⏎  ⏎ - Added `VLLM_OPENVINO_DEVICE` environment variable for OpenVINO device selection ⏎ - Updated GPU-related components in OpenVINO backend (KV cache shapes, swap capability, model profiling run etc) ⏎ - Updated OpenVINO version to 2024.4 RC1 in dependencies ⏎ - Updated installation instructions ⏎  ⏎ Some performance measurements obtained for Intel ARC A770 (16GB) for 1000 prompts from Sha …[truncated]

### L3-afb050b29d  (L3, 2024-10-02, sha afb050b29d0c, PR #8645)
TITLE: [Core] CUDA Graphs for Multi-Step + Chunked-Prefill (#8645)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+28/-20); csrc/prepare_inputs/advance_step.cu (+11/-0); vllm/worker/model_runner.py (+58/-14)
LABELS: ready
BODY: Multi-Step + Chunked-Prefill on `main` uses cuda-graphs sparsely. When a scheduler-step schedules prefills with the decode, all the steps in the multi-step runs in eager mode. But in fact, only the first step in the multi-step will have both prefills and decodes and the rest of the steps are fully decode-only. This PR leverages that fact and pads the necessary data structures so to use CUDA graphs in all steps expect the first-step. This improves …[truncated]

### L3-9aaf14c62e  (L3, 2024-10-03, sha 9aaf14c62e16, PR #9029)
TITLE: [misc] add forward context for attention (#9029)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+178/-251); vllm/attention/backends/flashinfer.py (+2/-2); tests/kernels/test_flash_attn.py (+7/-49); vllm/forward_context.py (+22/-0); vllm/spec_decode/draft_model_runner.py (+12/-10); vllm/worker/embedding_model_runner.py (+3/-1); vllm/worker/enc_dec_model_runner.py (+13/-11); vllm/worker/model_runner.py (+13/-10)
LABELS: ready
BODY: one step of https://github.com/vllm-project/vllm/issues/8821

### L3-22482e495e  (L3, 2024-10-04, sha 22482e495e00, PR #9062)
TITLE: [Bugfix] Flash attention arches not getting set properly (#9062)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+11/-0)
LABELS: ready
ISSUES: #9060 [Bug]: FATAL: FlashAttention requires building with sm version sm80-sm90, but
BODY: After https://github.com/vllm-project/vllm/pull/8845/ the flash attention arches were not getting set correctly since in the flash attention sub-repo the CMakeList.txt depends on VLLM_GPU_ARCHES (not something readily obvious). The https://github.com/vllm-project/vllm/pull/8845 PR stopped uses VLLM_GPU_ARCHES for CUDA in favor of setting the gencodes PR file. This is a quick fix (potential for a cleaner fix in a future PR) for restoring the VLLM_ …[truncated]

### L3-f4dd830e09  (L3, 2024-10-05, sha f4dd830e0945, PR #9097)
TITLE: [core] use forward context for flash infer (#9097)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+127/-67)
LABELS: ready
BODY: continue of https://github.com/vllm-project/vllm/pull/9029 ⏎  ⏎ to make flashinfer compatible with torch.compile single-graph capture.

### L3-cb3b2b9ba4  (L3, 2024-10-06, sha cb3b2b9ba4a9, PR #9038)
TITLE: [Bugfix] Fix incorrect updates to num_computed_tokens in multi-step scheduling (#9038)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+12/-2); tests/core/test_num_computed_tokens_update.py (+81/-0); tests/core/utils.py (+5/-1); vllm/engine/llm_engine.py (+66/-90); vllm/engine/output_processor/interfaces.py (+2/-6); vllm/engine/output_processor/multi_step.py (+13/-11)
LABELS: ready
BODY: `num_computed_tokens` tracks the number of tokens in a sequence that has a computed KV cache slot. This semantics is  not respected in `main` when Multi-Step scheduling is used.  ⏎  ⏎ Thanks @LiuXiaoxuanPKU for identifying this bug. ⏎  ⏎ Added unit tests to test updates to `num_computed_tokens` ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-23fea8714a  (L3, 2024-10-06, sha 23fea8714a1e, PR #9101)
TITLE: [Bugfix] Fix try-catch conditions to import correct Flash Attention Backend in Draft Model (#9101)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/spec_decode/draft_model_runner.py (+10/-5)
LABELS: ready
ISSUES: #9100 [Bug]: Try-catch conditions are incorrect to import correct  ROCm Flash Attention Backend in Draft Model
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ FIX #9100 (*link existing issues this PR will resolve*) ⏎  ⏎ Updated the try-catch block to import correct flash attention backend in `vllm/spec_decode/draft_model_runner.py` when running on non-CUDA platform e.g. ROCm. ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-4f95ffee6f  (L3, 2024-10-07, sha 4f95ffee6f40, PR #9089)
TITLE: [Hardware][CPU] Cross-attention and Encoder-Decoder models support on CPU backend (#9089)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/torch_sdpa.py (+299/-61); vllm/worker/cpu_enc_dec_model_runner.py (+311/-0); vllm/worker/cpu_model_runner.py (+3/-7); vllm/worker/cpu_worker.py (+9/-2); .buildkite/run-cpu-test.sh (+1/-0); tests/models/encoder_decoder/language/test_bart.py (+211/-217)
LABELS: ready
ISSUES: #9114 [Usage]: How to run llama 3.2 with CPU only version
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ FIX #9114 ⏎ - Add cross-attention support for SDPA backend. ⏎ - Add Encoder-Decoder models support for CPU backend. ⏎  ⏎ **TODO** ⏎  ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-8baf85e4e9  (L3, 2024-10-11, sha 8baf85e4e935, PR #8512)
TITLE: [Doc] Compatibility matrix for mutual exclusive features (#8512)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+2/-0); docs/source/index.rst (+1/-0); docs/source/models/performance.rst (+2/-0); docs/source/serving/compatibility_matrix.rst (+427/-0); vllm/config.py (+10/-0); vllm/engine/arg_utils.py (+2/-0); vllm/engine/output_processor/multi_step.py (+2/-0); vllm/executor/cpu_executor.py (+8/-0); vllm/inputs/preprocess.py (+2/-0); vllm/spec_decode/spec_decode_worker.py (+2/-0); (+3 more)
LABELS: ready
BODY: Greetings, ⏎  ⏎ We did a study of mutual exclusive features on vLLM and consolidated in a compatibility matrix. ⏎  ⏎ We propose to add the compatibility matrix to the documentation pages to help users to quick consult to plan their implementation or study. ⏎  ⏎ Following the table in markdown for quick checking and help reviewers.  ⏎  ⏎ CC @njhill  @maxdebayser  ⏎  ⏎ | | | | | | | | | | | | | ⏎ |-|-|-|-|-|-|-|-|-|-|-|-| ⏎ |Unnamed: 0|Chunked Prefill|APC|LoRa …[truncated]

### L3-7342a7d7f8  (L3, 2024-10-11, sha 7342a7d7f87e, PR #6484)
TITLE: [Model] Support Mamba (#6484)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/backends/placeholder_attn.py (+324/-0); vllm/attention/layer.py (+5/-3); vllm/attention/selector.py (+14/-7); .buildkite/run-cpu-test-ppc64le.sh (+7/-1); .buildkite/run-cpu-test.sh (+1/-0); docs/source/models/supported_models.rst (+5/-0); tests/kernels/test_attention_selector.py (+21/-16); tests/models/decoder_only/language/test_mamba.py (+295/-0); vllm/config.py (+28/-22); vllm/core/interfaces.py (+4/-4); (+19 more)
LABELS: ready
BODY: This is closely based on vLLM's Jamba implementation. It also has several changes and fixes to deal with the fact that there is no KV cache at all.  ⏎  ⏎ #### Changes in this PR ⏎ * Added the Mamba model definition and integration tests. ⏎ * Factored the Mamba cache management used by both Mamba and Jamba into a `mamba_cache.py`  ⏎ * Added a new "placeholder" attention backend with many noop methods, as Mamba's state needs to be managed differently. A …[truncated]

### L3-89feb4c84d  (L3, 2024-10-12, sha 89feb4c84dc8, PR #9298)
TITLE: [SpecDec] Remove Batch Expansion (2/3) (#9298)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.blocksparse.v0
FILES: vllm/attention/backends/blocksparse_attn.py (+2/-5); vllm/attention/backends/flash_attn.py (+42/-27); vllm/attention/backends/rocm_flash_attn.py (+2/-5); vllm/attention/backends/utils.py (+1/-1); vllm/attention/backends/xformers.py (+2/-5); tests/spec_decode/test_scorer.py (+39/-13); vllm/spec_decode/mqa_scorer.py (+34/-8); vllm/spec_decode/spec_decode_worker.py (+0/-6)
LABELS: ready
BODY: Follow up of #8839. ⏎  ⏎ We will use `flash_attn_varlen_func` for MQA scorer. Therefore, we can support different propose lengths for different requests in the batch, which is essential for [ngram](https://github.com/vllm-project/vllm/blob/main/vllm/spec_decode/ngram_worker.py) and [dynamic speculative decoding](https://github.com/vllm-project/vllm/issues/4565). ⏎  ⏎ The following are some preliminary benchmark numbers with MQA scorer compared with b …[truncated]

### L3-00298e092c  (L3, 2024-10-12, sha 00298e092c38, PR #9026)
TITLE: [Bugfix] Fix bug of xformer prefill for encoder-decoder (#9026)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend
FILES: vllm/attention/backends/xformers.py (+18/-11)
LABELS: ready
DEEP_STUDY: deep-study correctness case vllm:00298e092c: class=shape_alignment_edge; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ The K and V should not be cut off by the prompt length for encoder-decoder model, because they are calculated from the vision encoder output. For a encoder-decoder model, the actual length of K and V is the number of image tokens, which is usually larger than the number of prompt tokens. ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-f519902c52  (L3, 2024-10-13, sha f519902c52cf, PR #9317)
TITLE: [CI] Fix merge conflict (#9317)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/placeholder_attn.py (+7/-10)
LABELS: ready
BODY: There is a conflict between #6484 and #9298, which causes the CI to fail as shown [here](https://buildkite.com/vllm/ci-aws/builds/9832#01927f88-1ddf-40be-b4fd-446ae7a10005/180-4930). Fix the merge conflict in field `max_decode_query_len`.

### L3-473e7b3606  (L3, 2024-10-14, sha 473e7b3606e9, PR #9350)
TITLE: [TPU] Fix TPU SMEM OOM by Pallas paged attention kernel (#9350)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/pallas.py (+80/-21); vllm/worker/tpu_model_runner.py (+9/-0)
LABELS: tpu
BODY: This PR fixes the SMEM OOM caused by the Pallas paged attention kernel when the `max_model_len` is large. To reduce the peak SMEM usage, this PR split the batch dimension into sub-batches and invoke the paged attention kernel for each sub-batch, rather than invoking the kernel only once for the entire batch. While this degrades the performance and increases the compilation time, it works as a temporary workaround to the SMEM OOM issue.

### L3-776dbd74f1  (L3, 2024-10-16, sha 776dbd74f1d6, PR #9267)
TITLE: [CI/Build] mypy: Resolve some errors from checking vllm/engine (#9267)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+1/-1); tools/mypy.sh (+1/-11); vllm/compilation/backends.py (+2/-2); vllm/compilation/decorators.py (+5/-3); vllm/compilation/wrapper.py (+1/-1); vllm/config.py (+6/-4); vllm/core/scheduler.py (+4/-3); vllm/engine/arg_utils.py (+7/-5); vllm/engine/llm_engine.py (+12/-8); vllm/engine/metrics.py (+9/-5); (+10 more)
LABELS: ready
BODY: This PR is a chunk of mypy checking progress. ⏎  ⏎ 1. I got to where I knew everything would pass when running `mypy` with ⏎    `--follow-imports skip`.  (though some new issues have crept back in since I passed this point) ⏎  ⏎ 2. I then started working on expanding the coverage for what can be checked ⏎    using `--follow-imports silent`. Ongoing progress will come in chunks to make it ⏎    easier to review. I started with 100 errors and it's down to  …[truncated]

### L3-81ede99ca4  (L3, 2024-10-17, sha 81ede99ca44a, PR #8704)
TITLE: [Core] Deprecating block manager v1 and make block manager v2 default (#8704)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+3/-5); vllm/attention/backends/flashinfer.py (+3/-5); vllm/attention/backends/utils.py (+4/-12); .buildkite/test-pipeline.yaml (+6/-12); benchmarks/benchmark_latency.py (+0/-4); benchmarks/benchmark_prefix_caching.py (+0/-6); benchmarks/benchmark_throughput.py (+1/-10); benchmarks/overheads/benchmark_hashing.py (+0/-4); docs/source/models/spec_decode.rst (+0/-3); examples/offline_inference_mlpspeculator.py (+0/-2); (+35 more)
LABELS: ready
BODY: This PR deprecates block manager v1 and makes block manager v2 the default to simplify the code path. ⏎  ⏎ This is supported by this [benchmark](https://docs.google.com/document/d/1XxYUFai07ta5rE7OdtCVhLJ5J0oAxEqrGgarFdjv0Zc/edit?usp=sharing), where block manager v2 is <2% slower than block manager v1 on Llama 8B when no prefix hit, and has significant speedup upon full prefix hit. ⏎  ⏎ Summary of changes: ⏎ - Leave `--use-v2-block-manager` in the Eng …[truncated]

### L3-343f8e0905  (L3, 2024-10-17, sha 343f8e09055b, PR #9056)
TITLE: Support `BERTModel` (first `encoder-only` embedding model) (#9056)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+5/-2); vllm/attention/backends/xformers.py (+50/-9); tests/models/embedding/language/test_embedding.py (+12/-2); vllm/model_executor/layers/pooler.py (+10/-2); vllm/model_executor/models/bert.py (+419/-0); vllm/model_executor/models/registry.py (+1/-0)
LABELS: ready
ISSUES: #5179 [Feature]: BERT models for embeddings
BODY: SUMMARY: ⏎ * built on top of https://github.com/vllm-project/vllm/pull/5447 (co-author: [laishzh](https://github.com/laishzh)) ⏎ * introduces Encoder-Only models (BERT - tested with https://huggingface.co/BAAI/bge-base-en-v1.5 to match sentence-transformers implementation) ⏎ * introduces Encoder-Only attention for XFORMERS backend.  ⏎  ⏎ Note: this PR requires setting the `VLLM_ATTENTION_BACKEND=XFORMERS` variable to run. We throw a loud error if it i …[truncated]

### L3-0c9a5258f9  (L3, 2024-10-18, sha 0c9a5258f905, PR #9497)
TITLE: [Kernel] Add env variable to force flashinfer backend to enable tensor cores (#9497)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.flashinfer.trtllm_gen
FILES: vllm/attention/backends/flashinfer.py (+5/-2); vllm/envs.py (+6/-0)
LABELS: ready
BODY: Relates to #9471  ⏎  ⏎ We find that the heuristic used to decide when to enable tensor cores in Flashinfer is not working well for llama3.1-8b. While we try to figure out a better one, we propose adding this environment variable to override the logic.  ⏎  ⏎ ![image](https://github.com/user-attachments/assets/9c41b363-e5bd-4413-b4eb-5e85710b6461)

### L3-4fa3e33349  (L3, 2024-10-20, sha 4fa3e3334978, PR #9403)
TITLE: [Kernel] Support sliding window in flash attention backend (#9403)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.dispatch.selector
FILES: vllm/attention/backends/flash_attn.py (+5/-8); vllm/attention/layer.py (+3/-4); vllm/attention/selector.py (+2/-8); vllm/worker/cache_engine.py (+0/-1); vllm/worker/cpu_model_runner.py (+0/-1); vllm/worker/cpu_worker.py (+0/-1); vllm/worker/model_runner.py (+0/-1); vllm/worker/openvino_model_runner.py (+0/-1); vllm/worker/openvino_worker.py (+0/-1); vllm/worker/tpu_model_runner.py (+0/-1); (+3 more)
LABELS: ready
BODY: Flash attention backend provides sliding window support now. We can use it instead of fall back to xformer. ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-496e991da8  (L3, 2024-10-21, sha 496e991da824, PR #9498)
TITLE: [Doc] Consistent naming of attention backends (#9498)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/flash_attn.py (+1/-1); vllm/attention/backends/flashinfer.py (+1/-1); vllm/attention/backends/ipex_attn.py (+1/-1); vllm/attention/backends/openvino.py (+1/-1); vllm/attention/backends/pallas.py (+4/-0); vllm/attention/backends/placeholder_attn.py (+1/-1); vllm/attention/backends/rocm_flash_attn.py (+1/-1); vllm/attention/backends/torch_sdpa.py (+1/-1); vllm/attention/backends/utils.py (+6/-6); vllm/attention/backends/xformers.py (+1/-1); (+4 more)
LABELS: ready
BODY: Right now if you try to enable an unsupported feature (e.g., multi-step with xformers) you get a message like: ⏎ ``` ⏎ ValueError: Multi-Step not supported for attention backend: xformers. Set VLLM_ATTENTION_BACKEND to a value from ['flash-attn', 'rocm-flash-attn', 'flashinfer'] ⏎ ``` ⏎ I find this confusing because the names given do not match how the environment variable needs to be set in order to enable the corresponding feature (e.g. `rocm-flash …[truncated]

### L3-3ddbe25502  (L3, 2024-10-22, sha 3ddbe25502fb, PR #9536)
TITLE: [Hardware][CPU] using current_platform.is_cpu (#9536)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/backends/torch_sdpa.py (+4/-4); vllm/attention/ops/blocksparse_attention/interface.py (+10/-10); vllm/attention/selector.py (+3/-3); tests/conftest.py (+4/-2); tests/encoder_decoder/test_e2e_correctness.py (+3/-3); tests/kernels/test_attention_selector.py (+2/-1); tests/models/decoder_only/language/test_phimoe.py (+2/-2); tests/models/decoder_only/vision_language/test_fuyu.py (+3/-3); tests/models/decoder_only/vision_language/test_internvl.py (+3/-3); tests/models/decoder_only/vision_language/test_phi3v.py (+3/-2); (+7 more)
LABELS: ready
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ Part of the RPC: https://github.com/vllm-project/vllm/issues/9268 ⏎  ⏎ ### Changes: ⏎ `is_cpu -> current_platform.is_cpu` ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-6c5af09b39  (L3, 2024-10-22, sha 6c5af09b3969, PR #9289)
TITLE: [V1] Implement vLLM V1 [1/N] (#9289)
SOURCES: path_core, symbol_pickaxe, corpus:production-kernel-provenance
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.trtllm_gen, L3.dispatch.selector
FILES: vllm/attention/selector.py (+8/-0); vllm/v1/attention/__init__.py (+0/-0); vllm/v1/attention/backends/__init__.py (+0/-0); vllm/v1/attention/backends/flash_attn.py (+241/-0); vllm/engine/multiprocessing/engine.py (+17/-10); vllm/entrypoints/llm.py (+6/-1); vllm/envs.py (+5/-0); vllm/model_executor/layers/logits_processor.py (+6/-4); vllm/transformers_utils/detokenizer.py (+3/-165); vllm/transformers_utils/detokenizer_utils.py (+167/-0); (+17 more)
BODY: 

### L3-2394962d70  (L3, 2024-10-23, sha 2394962d7083, PR #9605)
TITLE: [Hardware][XPU] using current_platform.is_xpu (#9605)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/selector.py (+3/-3); vllm/config.py (+2/-2); vllm/executor/ray_utils.py (+2/-2); vllm/model_executor/custom_op.py (+2/-2); vllm/utils.py (+3/-26); vllm/worker/xpu_worker.py (+4/-3)
LABELS: ready
BODY: Part of https://github.com/vllm-project/vllm/issues/9268 ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-51c24c9736  (L3, 2024-10-23, sha 51c24c9736b1, PR #9596)
TITLE: [Build] Fix `FetchContent` multiple build issue (#9596)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+6/-4); setup.py (+8/-0)
LABELS: ready
BODY: Currently, all `FetchContent` dependencies live in `.deps`, and that includes their binary directories. During development with multiple builds, those different builds all place their binary dir for the dependency in `.deps`, leading to conflict. ⏎  ⏎ This PR fixes this by overriding the dependency's `BINARY_DIR`. ⏎ It also adds a mechanism to override the `FETCHCONTENT_BASE_DIR` if that's desired by the user. ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE …[truncated]

### L3-9645b9f646  (L3, 2024-10-24, sha 9645b9f64602, PR #9679)
TITLE: [V1] Support sliding window attention (#9679)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+4/-8)
BODY: This PR ports the change in #9403 to support sliding window attention with `vllm-flash-attn` on V1.

### L3-5cbdccd151  (L3, 2024-10-26, sha 5cbdccd151ef, PR #9716)
TITLE: [Hardware][openvino] is_openvino --> current_platform.is_openvino (#9716)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/selector.py (+2/-2); tests/kernels/test_attention_selector.py (+2/-1); vllm/config.py (+2/-2); vllm/executor/openvino_executor.py (+7/-13); vllm/model_executor/model_loader/openvino.py (+2/-2); vllm/platforms/__init__.py (+10/-0); vllm/platforms/interface.py (+4/-0); vllm/platforms/openvino.py (+31/-0); vllm/utils.py (+1/-10); vllm/worker/openvino_worker.py (+8/-8)
LABELS: ready
BODY: part of #9268 ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-55137e8ee3  (L3, 2024-10-26, sha 55137e8ee325, PR #9560)
TITLE: Fix: MI100 Support By Bypassing Custom Paged Attention (#9560)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+6/-2)
LABELS: ready
BODY: FIX #[8963](https://github.com/vllm-project/vllm/issues/8963)  ⏎  ⏎ ### Why is it needed? ⏎  ⏎ * Currently, Custom Paged Attention for ROCm on vLLM is only supported for MI250 and MI300. This limitation is enforced by the _use_rocm_custom_paged_attention function, which, when run on other GPUs like the MI100, causes the program to hit an UNREACHABLE_CODE path and fail. ⏎ --------------------- ⏎ * By the proposed changes this function can return False i …[truncated]

### L3-3cb07a36a2  (L3, 2024-10-27, sha 3cb07a36a20f, PR #9588)
TITLE: [Misc] Upgrade to pytorch 2.5 (#9588)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_pip, L3.flash_attn.fork_inline_cmake, L3.platform.cuda_selection
FILES: CMakeLists.txt (+2/-2); requirements-build.txt (+1/-1); requirements-cuda.txt (+3/-3); requirements-openvino.txt (+1/-1); cmake/utils.cmake (+1/-5); pyproject.toml (+1/-1); tests/models/decoder_only/language/test_big_models.py (+34/-12); vllm/platforms/cuda.py (+5/-0)
LABELS: ready
BODY: Upgrade to pytorch 2.5 ⏎  ⏎ Requires changes to flash attn: https://github.com/vllm-project/flash-attention/pull/23 ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-4e2d95e372  (L3, 2024-10-28, sha 4e2d95e372ad, PR #9642)
TITLE: [Hardware][ROCM] using current_platform.is_rocm (#9642)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/ops/blocksparse_attention/interface.py (+3/-3); vllm/attention/selector.py (+2/-2); tests/basic_correctness/test_basic_correctness.py (+2/-2); tests/compile/utils.py (+2/-2); tests/kernels/quant_utils.py (+11/-6); tests/kernels/test_attention.py (+13/-10); tests/kernels/test_attention_selector.py (+2/-1); tests/kernels/test_blocksparse_attention.py (+4/-3); tests/kernels/test_encoder_decoder_attn.py (+40/-36); tests/kernels/test_moe.py (+4/-3); (+22 more)
LABELS: ready
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ Part of the RPC: https://github.com/vllm-project/vllm/issues/9268 ⏎  ⏎ **Changes:** ⏎ `is_hip` -> `current_platform.is_rocm` ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-9ff4511e43  (L3, 2024-10-30, sha 9ff4511e43bb, PR #9781)
TITLE: [Misc] Add chunked-prefill support on FlashInfer. (#9781)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+60/-28); tests/basic_correctness/test_chunked_prefill.py (+12/-0)
LABELS: ready
BODY: Add chunked-prefill support on the current FlashInfer implementation as per discussion it will remain for a while until cascade inference is stable. Once cascade inference integration is done, we'll add support on that.  ⏎ Tested via `pytest tests/basic_correctness/test_chunked_prefill.py` with `vLLM_ATTENTION_BACKEND=FLASHINFER`. ⏎  ⏎ @comaniac @yzh199 @WoosukKwon @youkaichao

### L3-55650c83a0  (L3, 2024-10-31, sha 55650c83a0c3, PR #9532)
TITLE: [Bugfix] Fix `illegal memory access` error with chunked prefill, prefix caching, block manager v2 and xformers enabled together (#9532)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/utils.py (+6/-3); tests/prefix_caching/test_prefix_caching.py (+28/-0)
LABELS: ready, ci/build
BODY: [details omitted] ⏎  ⏎ For certain sequences of requests ~~(you can find one in `temp.py` to reproduce manually, it works using an OpenAI-compatible API)~~, vLLMs with ⏎  ⏎ 1. Chunked prefill ⏎ 2. Prefix caching ⏎ 3. Block manager V2 ⏎ 4. XFormers ⏎  ⏎ crashes with `CUDA error: an illegal memory access was encountered` somewhere. (sometimes, e.g., in the prefix caching kernel). ⏎  ⏎ ~~I believe this is not a Triton for Pascal (or my hardware in general) pro …[truncated]

### L3-96e0c9cbbd  (L3, 2024-10-31, sha 96e0c9cbbd65, PR #9896)
TITLE: [torch.compile] directly register custom op (#9896)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+11/-5); vllm/attention/backends/flashinfer.py (+11/-6); vllm/v1/attention/backends/flash_attn.py (+10/-4); tests/compile/piecewise/test_simple.py (+16/-4); tests/compile/piecewise/test_toy_llama.py (+16/-4); vllm/distributed/parallel_state.py (+23/-11); vllm/model_executor/layers/fused_moe/fused_marlin_moe.py (+19/-6); vllm/model_executor/layers/fused_moe/fused_moe.py (+41/-27); vllm/utils.py (+45/-0)
LABELS: ready
BODY: see https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5 for the discussion on the perf implification of registering custom ops

### L3-598b6d7b07  (L3, 2024-11-01, sha 598b6d7b0701, PR #9861)
TITLE: [Bugfix/Core] Flashinfer k_scale and v_scale (#9861)
SOURCES: path_core, subject_keyword, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+6/-3); tests/kernels/test_cache.py (+14/-7); vllm/model_executor/layers/quantization/modelopt.py (+5/-2)
LABELS: ready
ISSUES: #8641 [Bug]: Using FlashInfer with FP8 model with FP8 KV cache produces an error
DEEP_STUDY: deep-study correctness case vllm:598b6d7b07: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: Removes the assertion to allow models with k_scale and v_scale like `neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV` ⏎  ⏎ fixes: #8641 ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-6c0b7f548d  (L3, 2024-11-01, sha 6c0b7f548d80, PR #8346)
TITLE: [Core][VLM] Add precise multi-modal placeholder tracking (#8346)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: vllm/attention/backends/abstract.py (+11/-0); vllm/attention/backends/blocksparse_attn.py (+3/-0); vllm/attention/backends/flash_attn.py (+20/-0); vllm/attention/backends/flashinfer.py (+18/-0); vllm/attention/backends/placeholder_attn.py (+21/-1); vllm/attention/backends/rocm_flash_attn.py (+3/-0); vllm/attention/backends/utils.py (+18/-0); vllm/attention/backends/xformers.py (+3/-0); examples/offline_inference_audio_language.py (+1/-5); tests/kernels/utils.py (+2/-0); (+43 more)
LABELS: ready
BODY: Currently, multi-modal prompt placeholders are related to the multi-modal embeddings exclusively by index, i.e. the first placeholder token in the prompt must correspond to the first MM embedding vector, etc. This adds a mechanism for tracking multi-modal placeholder ranges precisely which allows multi-modal models to be used with chunked prefill and is a prerequisite for allowing multi-modal models to be used with prefix caching enabled (see #83 …[truncated]

### L3-a78dd3303e  (L3, 2024-11-01, sha a78dd3303efa, PR #9559)
TITLE: [Encoder Decoder] Add flash_attn kernel support for encoder-decoder models (#9559)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.dispatch.selector
FILES: vllm/attention/backends/flash_attn.py (+278/-86); vllm/attention/backends/utils.py (+143/-16); vllm/attention/backends/xformers.py (+26/-105); vllm/attention/selector.py (+1/-1); vllm/utils.py (+2/-2); vllm/worker/enc_dec_model_runner.py (+25/-10); tests/encoder_decoder/test_e2e_correctness.py (+51/-37); tests/kernels/test_encoder_decoder_attn.py (+115/-41); tests/kernels/utils.py (+74/-16); tests/models/encoder_decoder/vision_language/test_florence2.py (+1/-1); (+1 more)
LABELS: ready
BODY: This PR adds support for flash attention kernel for encoder decoder models.  For encoder-decoder models with dtype=bfloat16 the default backend choice is now FlashAttention instead of XFormers. However for llama-3.2-11b-vision-instruct we still use the Xformers backend even with dtype=bfloat16 because the model implementation (models/mllama.py) has dependency on PagedAttention.  ⏎  ⏎ For adding this support, we make the following changes in this pr …[truncated]

### L3-4dbcbbeb09  (L3, 2024-11-04, sha 4dbcbbeb0962, PR #9447)
TITLE: [Misc] Compute query_start_loc/seq_start_loc on CPU (#9447)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+10/-18); vllm/attention/backends/utils.py (+10/-18)
LABELS: ready
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ Avoid creating intermediate tensor query_lens_tensor and compute query_start_loc on CPU and allow h2d async. ⏎  ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-d93478b399  (L3, 2024-11-04, sha d93478b39953, PR #10001)
TITLE: [Bugfix] Upgrade to pytorch 2.5.1 (#10001)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+2/-2); requirements-build.txt (+1/-1); requirements-cuda.txt (+3/-3); requirements-openvino.txt (+1/-1); requirements-test.in (+1/-1); requirements-test.txt (+2/-2); pyproject.toml (+1/-1)
LABELS: ready, ci/build
BODY: Upgrade to pytorch 2.5.1 ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-4089985552  (L3, 2024-11-05, sha 40899855520e, PR #10058)
TITLE: [V1] Integrate Piecewise CUDA graphs (#10058)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+21/-14); vllm/compilation/backends.py (+5/-2); vllm/v1/worker/gpu_model_runner.py (+107/-20)
BODY: This PR integrates the piecewise CUDA graphs into the V1 model runner. ⏎  ⏎ * Set `VLLM_TORCH_COMPILE_LEVEL=3` to enable the feature. ⏎ * The compilation + capturing time takes at most ~17 secs. ⏎ * Currently, piecewise CUDA graphs are not compatible with custom ops. Therefore, we rely on Torch Inductor to optimize the model. ⏎ * Consequently, FP8 or other quantizations are not supported with piecewise CUDA graphs. ⏎  ⏎ ## Benchmarks (`benchmark_latency …[truncated]

### L3-a02a50e6e5  (L3, 2024-11-06, sha a02a50e6e5bb, PR #6143)
TITLE: [Hardware][Intel-Gaudi] Add Intel Gaudi (HPU) inference backend (#6143)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.dispatch.selector
FILES: vllm/attention/backends/hpu_attn.py (+264/-0); vllm/attention/ops/hpu_paged_attn.py (+103/-0); vllm/attention/selector.py (+8/-0); Dockerfile.hpu (+16/-0); docs/source/getting_started/gaudi-installation.rst (+402/-0); docs/source/index.rst (+2/-1); requirements-hpu.txt (+11/-0); setup.py (+45/-2); vllm/_custom_ops.py (+1/-1); vllm/config.py (+19/-3); (+21 more)
LABELS: documentation, ci/build
BODY: This PR adds initial support for Intel Gaudi backend to vLLM.  ⏎  ⏎  ⏎ Requirements ⏎ ------------ ⏎  ⏎ -   OS: Ubuntu 22.04 LTS ⏎ -   Python: 3.10 ⏎ -   Intel Gaudi accelerator ⏎ -   Intel Gaudi software version 1.18.0 ⏎  ⏎  ⏎ Supported Features ⏎ ================== ⏎  ⏎ -   [Offline batched inference](https://github.com/HabanaAI/vllm-fork/blob/habana_main/docs/source/getting_started/quickstart.rst#offline-batched-inference) ⏎ -   Online inference via [OpenAI-C …[truncated]

### L3-ffc0f2b47a  (L3, 2024-11-06, sha ffc0f2b47add, PR #10045)
TITLE: [Model][OpenVINO] Fix regressions from #8346 (#10045)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/openvino.py (+11/-1); .buildkite/run-openvino-test.sh (+1/-1); vllm/model_executor/models/molmo.py (+3/-3)
LABELS: ready, ci/build
ISSUES: #10025 [Bug]: Error while trying to run vLLM microsoft/Phi-3-mini-4k-instruct with OpenVINO backend | #10042 [Bug]: AttributeError: 'tuple' object has no attribute 'seq_data' when loading Molmo7B on two Nvidia L4
BODY: Fix #10025 by adding placeholder index maps to `OpenVINOAttentionMetadata`. ⏎ Fix #10042 by returning `DummyData` instead of a tuple. ⏎  ⏎ (As per the comment that I copied from `AttentionMetadata`, this doesn't really belong here and should probably be on `ModelInputs` instead, but I didn't want to diverge the OpenVINO/non-OpenVINO paths further.)

### L3-21063c11c7  (L3, 2024-11-06, sha 21063c11c7d3, PR #8464)
TITLE: [CI/Build] drop support for  Python 3.8 EOL (#8464)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: vllm/attention/ops/blocksparse_attention/interface.py (+2/-4); .buildkite/nightly-benchmarks/scripts/convert-results-json-to-markdown.py (+5/-5); .buildkite/nightly-benchmarks/scripts/generate-nightly-markdown.py (+2/-2); .buildkite/nightly-benchmarks/scripts/summary-nightly-results.py (+2/-2); .github/workflows/mypy.yaml (+1/-1); .github/workflows/publish.yml (+1/-1); .github/workflows/ruff.yml (+16/-16); .github/workflows/yapf.yml (+13/-13); .readthedocs.yaml (+5/-6); CMakeLists.txt (+18/-18); (+105 more)
LABELS: documentation, frontend, ready, ci/build
BODY: As Python 3.8 reaches [EOL](https://devguide.python.org/versions/), this PR would be a starting point to remove potential code branch on 3.8 only. ⏎  ⏎ I'm still working through all of the code, but this would be a starting point for CI.  ⏎  ⏎ Depends on #8469, and wait til PyTorch drop 3.8 support ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-d58268c56a  (L3, 2024-11-06, sha d58268c56a8e, PR #9888)
TITLE: [V1] Make v1 more testable (#9888)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.upstream_pip, L3.dispatch.selector
FILES: vllm/attention/selector.py (+33/-10); Dockerfile (+3/-0); pyproject.toml (+1/-0); tests/conftest.py (+18/-0); tests/entrypoints/llm/test_prompt_validation.py (+9/-0); tests/kernels/test_attention_selector.py (+2/-0); tests/kernels/test_encoder_decoder_attn.py (+2/-2); vllm/engine/multiprocessing/engine.py (+10/-8); vllm/entrypoints/llm.py (+17/-9); vllm/model_executor/layers/sampler.py (+9/-0); (+65 more)
LABELS: frontend, ready, ci/build
BODY: This PR fixes some small issues with running the v1 engine, and makes it easily testable. ⏎  ⏎ - The v1 detokenizer process wasn't being terminated ⏎ - The `Sampler` class was being patched over by `unittest.mock.patch` in the v1 gpu executor, which is prone to import order bugs and persists as a side effect once the patch context is exited ⏎ - The choice of engine class was being determined at import-time rather than at runtime, which can also lead  …[truncated]

### L3-098f94de42  (L3, 2024-11-06, sha 098f94de4285, PR #10038)
TITLE: [CI/Build] Drop Python 3.8 support (#10038)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1); setup.py (+4/-8); .readthedocs.yaml (+1/-1); docs/source/getting_started/amd-installation.rst (+0/-2); docs/source/getting_started/installation.rst (+1/-1); docs/source/getting_started/neuron-installation.rst (+1/-1); docs/source/getting_started/quickstart.rst (+1/-1); vllm/distributed/parallel_state.py (+2/-3)
LABELS: documentation, ready, ci/build
BODY: 706e3609 [CI/Build] Drop Python 3.8 support ⏎ bbdf33a7 [Core] Make use of str.removeprefix() ⏎  ⏎ commit 706e3609221eccc8ea03baf5b90e6e5a7ea4b758 ⏎ Author: Russell Bryant <rbryant@redhat.com> ⏎ Date:   Tue Nov 5 09:23:59 2024 -0500 ⏎  ⏎     [CI/Build] Drop Python 3.8 support ⏎      ⏎     Pytorch 3.5 dropped support for Python 3.8, so it's time we do the ⏎     same. ⏎      ⏎     Signed-off-by: Russell Bryant <rbryant@redhat.com> ⏎  ⏎ commit bbdf33a7a425640f1846 …[truncated]

### L3-d3859f1891  (L3, 2024-11-06, sha d3859f18915a, PR #9823)
TITLE: [Misc][XPU] Upgrade to Pytorch 2.5 for xpu backend (#9823)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: vllm/attention/backends/ipex_attn.py (+20/-16); Dockerfile.xpu (+11/-1); requirements-xpu.txt (+4/-4); vllm/_ipex_ops.py (+8/-25)
LABELS: intel-gpu, ci/build
BODY: Upgrade to Pytorch 2.5 for xpu backend ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-a4b3e0c1e9  (L3, 2024-11-07, sha a4b3e0c1e999, PR #9911)
TITLE: [Hardware][CPU] Update torch 2.5 (#9911)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: csrc/cpu/attention.cpp (+10/-0); .buildkite/run-cpu-test.sh (+1/-1); Dockerfile.cpu (+1/-1); cmake/cpu_extension.cmake (+1/-0); csrc/cpu/cpu_types_x86.hpp (+45/-33); csrc/cpu/dnnl_helper.hpp (+6/-0); csrc/cpu/quant.cpp (+7/-0); docs/source/getting_started/cpu-installation.rst (+2/-4); requirements-cpu.txt (+1/-1); tests/models/decoder_only/language/test_models.py (+1/-2); (+2 more)
LABELS: documentation, frontend, ready, ci/build
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:a6f332d0d9 (fix PR 10108)
BODY: - Update torch 2.5 ⏎ - Enable FP16 data type ⏎  ⏎ [details omitted]

### L3-9d43afcc53  (L3, 2024-11-07, sha 9d43afcc5386, PR #9291)
TITLE: [Feature] [Spec decode]: Combine chunked prefill with speculative decoding (#9291)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/flash_attn.py (+7/-3); vllm/attention/backends/rocm_flash_attn.py (+6/-0); vllm/attention/backends/xformers.py (+7/-0); tests/spec_decode/e2e/test_compatibility.py (+0/-34); tests/spec_decode/e2e/test_multistep_correctness.py (+102/-3); tests/spec_decode/e2e/test_ngram_correctness.py (+35/-1); tests/spec_decode/test_ngram_worker.py (+7/-2); tests/spec_decode/test_scorer.py (+27/-4); tests/spec_decode/test_spec_decode_worker.py (+82/-0); tests/spec_decode/utils.py (+60/-11); (+7 more)
LABELS: documentation, frontend, ready, ci/build
BODY: Hey, this PR implements https://github.com/vllm-project/vllm/issues/5016. ⏎  ⏎ The main idea is to make use of the current Speculative Decoder workflow and integrate it with mixed prefill-decode batches. ⏎ In particular, we can run the batched prefills and decodes together through the scorer (with the usual prefill|decode layout supported by backend), while the proposer can sync its KV cache on prefills only. ⏎  ⏎ ![image](https://github.com/user-atta …[truncated]

### L3-58170d6503  (L3, 2024-11-11, sha 58170d65034f, PR #10193)
TITLE: [Hardware][CPU] Add embedding models support for CPU backend (#10193)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/torch_sdpa.py (+10/-4); .buildkite/run-cpu-test-ppc64le.sh (+1/-2); .buildkite/run-cpu-test.sh (+1/-2); tests/models/embedding/language/test_embedding.py (+4/-3); vllm/model_executor/models/bert.py (+0/-6); vllm/worker/cpu_embedding_model_runner.py (+122/-0); vllm/worker/cpu_enc_dec_model_runner.py (+5/-6); vllm/worker/cpu_model_runner.py (+35/-22); vllm/worker/cpu_worker.py (+7/-7)
LABELS: ready, ci/build
ISSUES: #9379 [Bug]: Unable to embed any text using the vLLM CPU server
BODY: FIX #9379  ⏎ - Implement CPU embedding model runner ⏎ - Add encoder-only attention support to SDPA backend

### L3-812c981fa0  (L3, 2024-11-11, sha 812c981fa00a, PR #10091)
TITLE: Splitting attention kernel file (#10091)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+3/-2); csrc/attention/attention_kernels.cuh (+0/-326); csrc/attention/paged_attention_v1.cu (+193/-0); csrc/attention/paged_attention_v2.cu (+203/-0)
LABELS: ready, ci/build
BODY: Split paged attention kernel into two files for v1 and v2 to speed up compilation when template instantiation explodes. ⏎  ⏎ side effect: support Navi32

### L3-b41fb9d3b1  (L3, 2024-11-12, sha b41fb9d3b10d, PR #9982)
TITLE: [Encoder Decoder] Update Mllama to run with both FlashAttention and XFormers (#9982)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: vllm/worker/enc_dec_model_runner.py (+6/-28); tests/encoder_decoder/test_e2e_correctness.py (+8/-1); tests/models/encoder_decoder/vision_language/test_mllama.py (+63/-37); tests/test_config.py (+2/-0); vllm/model_executor/models/mllama.py (+38/-14)
LABELS: ready
BODY: In this pr we update Mllama to run on both xFormers and Flash Attention backend. Currently it runs only with the xFormers backend. This pr makes the following changes ⏎  ⏎ 1. Updates mllama.py to run with both xFormers and FlashAttention. There were 2 changes needed for this (a) update attention_with_mask to use appropriately update the cache depending on the backend being used (b) update the shape of the query when computing the attention. Current …[truncated]

### L3-b6dde33019  (L3, 2024-11-13, sha b6dde3301988, PR #10282)
TITLE: [Core] Flashinfer - Remove advance step size restriction (#10282)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: csrc/prepare_inputs/advance_step.cu (+38/-28)
LABELS: ready
BODY: This PR addresses the error when flashinfer block_tables can't be mapped to available GPU threads, leading to an error(example):  ⏎  ⏎ ``` ⏎ RuntimeError: multi-step: not enough threads to map block_table toFlashInfer's paged_kv_indices on GPU. Try reducing the number of seqs, increasing the block size or take smaller steps. num_queries = 512 block_tables.stride(0) = 512 blocks = 132 max_threads = 1024 ⏎ ``` ⏎  ⏎ With this fix, individual threads can b …[truncated]

### L3-4a18fd14ba  (L3, 2024-11-14, sha 4a18fd14ba4a, PR #9387)
TITLE: Support Roberta embedding models (#9387)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.paged.python_wrapper
FILES: csrc/attention/paged_attention_v1.cu (+3/-0); csrc/attention/paged_attention_v2.cu (+3/-0); csrc/cpu/attention.cpp (+6/-0); vllm/attention/ops/ipex_attn.py (+1/-1); vllm/attention/ops/paged_attn.py (+1/-1); tests/model_executor/test_model_load_with_params.py (+44/-0); tests/models/embedding/language/test_embedding.py (+2/-0); vllm/model_executor/models/bert.py (+23/-12); vllm/model_executor/models/registry.py (+2/-0); vllm/model_executor/models/roberta.py (+117/-0)
LABELS: ready
ISSUES: #9847 [New Model]: BAAI/bge-m3
BODY: This PR adds support for Roberta embedding models. It's mostly the same as the Bert architecture, the only thing that changes is the padding token in the Embedding layer so this PR tries to reuse Bert modeling classes as much as possible. For some of the models we also need head size 32, so this size is added to the kernels here. ⏎  ⏎ cc: @robertgshaw2-neuralmagic , @DarkLight1337  ⏎  ⏎ FIX #9847

### L3-4fd9375028  (L3, 2024-11-16, sha 4fd937502827, PR #10383)
TITLE: [2/N][torch.compile] make compilation cfg part of vllm cfg (#10383)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/compile/piecewise/test_simple.py (+5/-3); tests/compile/piecewise/test_toy_llama.py (+12/-10); tests/compile/test_basic_correctness.py (+1/-1); tests/compile/test_full_graph.py (+1/-1); tests/compile/test_fusion.py (+1/-1); tests/compile/test_wrapper.py (+3/-1); tests/compile/utils.py (+1/-1); tests/model_executor/test_enabled_custom_ops.py (+26/-26); tests/tpu/test_compilation.py (+1/-1); tests/tpu/test_custom_dispatcher.py (+1/-1); (+17 more)
LABELS: ready
BODY: continue of https://github.com/vllm-project/vllm/pull/10237 ⏎  ⏎ move `vllm.compilation.config` into `vllm.config` ⏎  ⏎ it is still controlled by the env var, but in the core code, no one should read the env var. ⏎  ⏎ TODO: ⏎  ⏎ 1. move the env var to cli arg. ⏎ 2. remove compilation context, initialize all config fields during init, rather than during model forward time

### L3-c2170a5b39  (L3, 2024-11-18, sha c2170a5b395a, PR #9014)
TITLE: [Kernel] Explicitly specify other value in tl.load calls (#9014)
SOURCES: path_core
ARTIFACT_HINTS: L3.blocksparse.v0
FILES: vllm/attention/ops/blocksparse_attention/blocksparse_attention_kernel.py (+10/-3); vllm/lora/ops/bgmv_expand.py (+3/-1); vllm/lora/ops/bgmv_expand_slice.py (+7/-1); vllm/lora/ops/sgmv_expand.py (+4/-1); vllm/lora/ops/sgmv_expand_slice.py (+4/-1); vllm/model_executor/layers/quantization/awq_triton.py (+7/-7)
LABELS: ready
BODY: According to [Triton's user documentation](https://triton-lang.org/main/python-api/generated/triton.language.load.html), using a mask with `tl.load` will not load the data specified by the mask. If `other` is not specified, there is no guarantee that the values affected by the mask will be valid (could be nan in theory). It would be a good idea to explicitly specify the `other` value in all masked `tl.load` calls to avoid potential propagation of …[truncated]

### L3-a03ea40792  (L3, 2024-11-18, sha a03ea4079220, PR #10399)
TITLE: [3/N][torch.compile] consolidate custom op logging (#10399)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config.py (+10/-2); vllm/model_executor/custom_op.py (+6/-3); vllm/plugins/__init__.py (+4/-0)
BODY: previously the custom op logging is too verbose. ⏎  ⏎ after https://github.com/vllm-project/vllm/pull/10383 , we now have global control and aggregation of the custom op statistics.

### L3-7851b45196  (L3, 2024-11-18, sha 7851b45196af, PR #10406)
TITLE: [5/N][torch.compile] torch.jit.script --> torch.compile (#10406)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/rejection_sampler.py (+1/-1); vllm/model_executor/layers/vocab_parallel_embedding.py (+2/-2); vllm/model_executor/models/phi3_small.py (+2/-2); vllm/worker/model_runner.py (+1/-1)
LABELS: ready
ISSUES: #8536 [Bug]: Model load on 2 or 4-gpu A100 setup may cause default text encoding to be ascii, unless enforce_eager=True
BODY: fixes https://github.com/vllm-project/vllm/issues/8536 ⏎  ⏎ and I also observe performance gain: ⏎  ⏎ ```shell ⏎ python benchmarks/benchmark_throughput.py --input-len 1024 --output-len 256 --model meta-llama/Llama-3.1-8B -tp 2 --load-format dummy ⏎  ⏎ main branch:  ⏎ Throughput: 17.91 requests/s, 22928.01 total tokens/s, 4585.60 output tokens/s ⏎  ⏎ this pr: ⏎ Throughput: 18.69 requests/s, 23918.95 total tokens/s, 4783.79 output tokens/s ⏎ ```

### L3-1ea291a417  (L3, 2024-11-19, sha 1ea291a4173a, PR #10421)
TITLE: Fix: Build error seen on Power Architecture (#10421)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: csrc/cpu/attention.cpp (+10/-2); cmake/cpu_extension.cmake (+10/-4); csrc/cpu/quant.cpp (+6/-0)
LABELS: ready, ci/build
DEEP_STUDY: deep-study correctness case vllm:1ea291a417: class=hardware_compiler_specific; symptom=compile_or_build_failure; introducing=unknown
BODY: The issue in the vLLM library involves the introduction of the -mf16c flag and FP16 vector types, which are specific to x86 architecture. These changes cause build failures on Power architecture, though the library builds and functions correctly on x86. To resolve this, architecture-specific checks will be implemented. The -mf16c flag will not be applied on the Power architecture systems, while FP32Vec types will be used as a fallback for unsuppo …[truncated]

### L3-803f37eaaa  (L3, 2024-11-19, sha 803f37eaaa11, PR #10437)
TITLE: [6/N] torch.compile rollout to users (#10437)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/compile/piecewise/piecewise_compilation_config.json (+0/-5); tests/compile/piecewise/test_simple.py (+7/-11); tests/compile/piecewise/test_toy_llama.py (+17/-28); tests/compile/test_basic_correctness.py (+9/-4); tests/compile/utils.py (+2/-2); tests/model_executor/test_enabled_custom_ops.py (+1/-3); tests/tpu/test_compilation.py (+35/-12); tests/tpu/test_custom_dispatcher.py (+6/-4); vllm/config.py (+20/-23); vllm/engine/arg_utils.py (+23/-6); (+5 more)
LABELS: ready
BODY: the user interface: ⏎  ⏎ command line: ⏎  ⏎ short: `-O 0/1/2/3` ⏎ long: `--compilation-config json_string` ⏎  ⏎ `vllm.LLM`: ⏎  ⏎ users can construct `CompilationConfig` object directly, and pass `compilation_config=obj` . ⏎  ⏎ the first roll out will only roll out 3 levels, more fine-grained control will come later as we stabilize them. ⏎  ⏎ documentation on how to use and the design doc will come later.

### L3-8c1fb50705  (L3, 2024-11-19, sha 8c1fb507052d, PR #10358)
TITLE: [Platform][Refactor] Extract func `get_default_attn_backend` to `Platform` (#10358)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.dispatch.selector, L3.platform.rocm_selection
FILES: vllm/attention/selector.py (+6/-50); vllm/platforms/__init__.py (+1/-0); vllm/platforms/cpu.py (+9/-1); vllm/platforms/hpu.py (+5/-1); vllm/platforms/interface.py (+19/-0); vllm/platforms/openvino.py (+7/-1); vllm/platforms/rocm.py (+13/-1); vllm/platforms/tpu.py (+11/-1); vllm/platforms/xpu.py (+11/-1); vllm/worker/enc_dec_model_runner.py (+2/-1); (+4 more)
LABELS: ready
BODY: Extrct func `get_default_attn_backend` to `Platform` and implement it on different backends. ⏎  ⏎ Part of #9268 ⏎  ⏎  ⏎ [details omitted]

### L3-63f1fde277  (L3, 2024-11-20, sha 63f1fde277d0, PR #10355)
TITLE: [Hardware][CPU] Support chunked-prefill and prefix-caching on CPU (#10355)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/torch_sdpa.py (+152/-37); vllm/attention/ops/ipex_attn.py (+109/-41); .buildkite/run-cpu-test.sh (+8/-1); docs/source/getting_started/cpu-installation.rst (+5/-5); docs/source/serving/compatibility_matrix.rst (+2/-2); tests/basic_correctness/test_chunked_prefill.py (+62/-1); vllm/platforms/cpu.py (+6/-9); vllm/worker/cpu_model_runner.py (+215/-273)
LABELS: documentation, ready, ci/build
BODY: This PR provides CPU chunked-prefill and prefix-caching support. For now only FP32 and BF16 are supported, ipex-2.6 will provide the FP16 support.  ⏎  ⏎ [details omitted]

### L3-0cd3d9717e  (L3, 2024-11-20, sha 0cd3d9717e38, PR #10460)
TITLE: [7/N] torch.compile, reduce compilation time (#10460)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/piecewise/test_simple.py (+1/-1); tests/compile/piecewise/test_toy_llama.py (+2/-2); vllm/compilation/backends.py (+1/-1); vllm/config.py (+10/-7); vllm/worker/worker.py (+13/-5)
LABELS: ready
BODY: Normal run: ⏎  ⏎ ```shell ⏎ $ vllm serve meta-llama/Meta-Llama-3-8B ⏎ Memory profiling took 1.16 seconds ⏎ # GPU blocks: 27846, # CPU blocks: 2048 ⏎ Graph capturing finished in 15 secs, took 0.32 GiB ⏎ ``` ⏎  ⏎ Run with inductor compiling the full graph: ⏎  ⏎ ```shell ⏎ $ vllm serve meta-llama/Meta-Llama-3-8B -O 3 ⏎ Memory profiling took 28.21 seconds ⏎ # GPU blocks: 27825, # CPU blocks: 2048 ⏎ Graph capturing finished in 32 secs, took 0.33 GiB ⏎ ``` ⏎  ⏎ Run with inductor compilin …[truncated]

### L3-2f77b6cfec  (L3, 2024-11-20, sha 2f77b6cfec32, PR #10307)
TITLE: [TPU] Implement prefix caching for TPUs (#10307)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/pallas.py (+43/-23); requirements-tpu.txt (+3/-3); vllm/worker/tpu_model_runner.py (+134/-77); vllm/worker/tpu_worker.py (+2/-2)
LABELS: tpu, ci/build
BODY: This PR implements the prefix caching support for the TPU backend.

### L3-6c1208d083  (L3, 2024-11-20, sha 6c1208d083fb, PR #10462)
TITLE: [Core] Add Sliding Window Support with Flashinfer (#10462)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+8/-5); tests/core/block/e2e/test_correctness_sliding_window.py (+10/-2)
LABELS: ready
ISSUES: #9854 [help wanted]: add sliding window support for flashinfer 
BODY: Fix #9854 ⏎  ⏎ Tests following.

### L3-33e0a2540a  (L3, 2024-11-21, sha 33e0a2540a6b, PR #10552)
TITLE: [9/N] torch.compile LLM usage (#10552)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/tpu/test_compilation.py (+2/-3); vllm/entrypoints/llm.py (+14/-1)
LABELS: frontend
BODY: 

### L3-eebad39f26  (L3, 2024-11-22, sha eebad39f2656, PR #10558)
TITLE: [torch.compile] support all attention backends (#10558)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface, L3.platform.cuda_selection, L3.platform.rocm_selection, L3.blocksparse.v0
FILES: vllm/attention/backends/abstract.py (+14/-9); vllm/attention/backends/blocksparse_attn.py (+1/-1); vllm/attention/backends/flash_attn.py (+172/-240); vllm/attention/backends/flashinfer.py (+111/-169); vllm/attention/backends/hpu_attn.py (+1/-1); vllm/attention/backends/ipex_attn.py (+1/-1); vllm/attention/backends/pallas.py (+1/-1); vllm/attention/backends/rocm_flash_attn.py (+1/-1); vllm/attention/backends/torch_sdpa.py (+6/-6); vllm/attention/backends/utils.py (+2/-2); (+67 more)
LABELS: ready
BODY: previously we register attention ops separately, e.g. flashinfer, flash attention. ⏎  ⏎ this pr changes the registration to be the unified attention interface, so that we don't need to register these attention backends one by one. ⏎  ⏎ how it works: ⏎  ⏎ 1. when we create an attention class, we register it in the per-model static forward context, identified by its layer name ⏎ 2. when we call the attention implementation, we pass in the layer name through pyto …[truncated]

### L3-4aba6e3d1a  (L3, 2024-11-22, sha 4aba6e3d1a0c, PR #10584)
TITLE: [core] gemma2 full context length support (#10584)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+10/-2); tests/basic_correctness/test_basic_correctness.py (+18/-7); vllm/config.py (+20/-9); vllm/model_executor/models/gemma2.py (+7/-6)
LABELS: ready
ISSUES: #6220 [Bug]: Gemma2 supports 8192 context with sliding window, but vllm only does 4196 or fails if try 8192 | #8580 [Bug]: Wrong Response with Gemma2 with 8k context length
BODY: the scheduler treats it as a model without sliding window, and sliding window is only used for computation. ⏎  ⏎ FIX #6220 ⏎ FIX #8580

### L3-04668ebe7a  (L3, 2024-11-23, sha 04668ebe7a35, PR #10593)
TITLE: [Bugfix] Avoid import AttentionMetadata explicitly in Mllama (#10593)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.blocksparse.v0
FILES: vllm/attention/backends/blocksparse_attn.py (+5/-0); vllm/attention/layer.py (+2/-1); vllm/v1/attention/backends/flash_attn.py (+1/-1); vllm/model_executor/models/mllama.py (+7/-7); vllm/platforms/openvino.py (+6/-2)
LABELS: ready
BODY: - Since mllama is also supported on CPU backend, import `FlashAttentionMetadata` and `XFormersMetadata` on CPU backend will cause import error due to missing packages. ⏎ - This PR modifies masked attention implementation selection to use `_Backend` instead of `attn_metadata` ⏎ - Also fix broken kernel test due to `import openvino`

### L3-7c2134beda  (L3, 2024-11-24, sha 7c2134beda9a, PR #10620)
TITLE: [torch.compile] force inductor threads (#10620)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/plugins/__init__.py (+4/-1)
ISSUES: #10619 [Usage]: torch.compile still generates multiple subprocesses
BODY: FIX #10619 ⏎ ping @youkaichao

### L3-65813781a2  (L3, 2024-11-24, sha 65813781a2e2, PR #10622)
TITLE: [torch.compile] add warning for unsupported models (#10622)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/compilation/counter.py (+1/-0); vllm/compilation/decorators.py (+2/-0); vllm/plugins/__init__.py (+15/-0)
LABELS: ready
BODY: 

### L3-571841b7fc  (L3, 2024-11-25, sha 571841b7fcc6, PR #10613)
TITLE: [torch.compile] support encoder based models (#10613)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/test_basic_correctness.py (+10/-0); vllm/model_executor/models/bert.py (+7/-10)
LABELS: ready
BODY: after https://github.com/vllm-project/vllm/pull/10558 , it should be pretty easy to support `torch.compile` for encoder based models now.

### L3-05d1f8c9c6  (L3, 2024-11-25, sha 05d1f8c9c64b, PR #10624)
TITLE: [misc] move functions to config.py (#10624)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+1/-2); tests/compile/piecewise/test_simple.py (+2/-2); tests/compile/piecewise/test_toy_llama.py (+2/-2); tests/kernels/test_encoder_decoder_attn.py (+1/-2); tests/model_executor/test_enabled_custom_ops.py (+1/-2); vllm/compilation/wrapper.py (+1/-2); vllm/config.py (+51/-0); vllm/model_executor/custom_op.py (+1/-1); vllm/model_executor/model_loader/loader.py (+1/-2); vllm/model_executor/model_loader/tensorizer.py (+1/-2); (+1 more)
LABELS: ready
BODY: address https://github.com/vllm-project/vllm/pull/10622#discussion_r1855967082
