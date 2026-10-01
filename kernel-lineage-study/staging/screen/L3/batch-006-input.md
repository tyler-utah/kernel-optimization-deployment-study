### L3-3053a22b33  (L3, 2025-09-16, sha 3053a22b330c, PR #22758)
TITLE: fp8 kv cache support fix for torch.compile (#22758)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+1/-1); vllm/model_executor/layers/quantization/kv_cache.py (+3/-1)
LABELS: ready, v1
BODY: Torch compile was erroring in attention layer on assert for q_scale to be equals 1.0. The error came originally from HIP saying that operation is not allowed during cuda graph capture. Thus implementing a copy of q_scale - q_scale_float (similar to k_scale_float and v_scale float). ⏎  ⏎ PS q_scale needs to be one because upscalling doesn't happen on AMD from predeceasing GEMMs and scales are only applied to k and v if those are in fp8.

### L3-bfe9380161  (L3, 2025-09-17, sha bfe93801614b, PR #24599)
TITLE: Apply fixes for CUDA 13 (#24599)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+10/-0); csrc/cub_helpers.h (+17/-0); csrc/layernorm_kernels.cu (+4/-9); csrc/layernorm_quant_kernels.cu (+4/-9); csrc/moe/topk_softmax_kernels.cu (+3/-13); csrc/quantization/compressed_tensors/int8_quant_kernels.cu (+2/-9); csrc/quantization/fp8/common.cu (+2/-7); csrc/quantization/fused_kernels/layernorm_utils.cuh (+5/-9)
LABELS: ready, ci/build
ISSUES: #24464 [Bug]: Building vLLM with CUDA 13.0
BODY: ## Purpose ⏎ Fixes https://github.com/vllm-project/vllm/issues/24464 "Building vLLM with CUDA 13.0". ⏎  ⏎ ## Test Plan ⏎ No plans. Just build. ⏎  ⏎ ## Test Result ⏎ Test results must be the same as on CUDA 12.x. ⏎  ⏎ --- ⏎ [details omitted]

### L3-8f3616f422  (L3, 2025-09-17, sha 8f3616f422e3, PR #23961)
TITLE: Remove old cutlass mla (#23961)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.mla.cutlass_kernels, L3.mla.cutlass_v1_backend
FILES: CMakeLists.txt (+0/-2); csrc/attention/mla/cutlass_mla_entry.cu (+0/-38); csrc/attention/mla/cutlass_mla_kernels.cu (+0/-225); csrc/torch_bindings.cpp (+0/-7); vllm/_custom_ops.py (+0/-9); vllm/v1/attention/backends/mla/cutlass_mla.py (+10/-64)
LABELS: ready, ci/build, v1
BODY: ## Purpose ⏎ Remove old CUTLASS MLA kernel, no longer used. ⏎  ⏎ ## Test Plan ⏎ `pytest tests/kernels/test_cutlass_mla_decode.py` ⏎  ⏎ ## Test Result ⏎ Test passes ⏎  ⏎ --- ⏎ [details omitted]

### L3-e67a79db03  (L3, 2025-09-17, sha e67a79db0375, PR #24600)
TITLE: [Bugfix] Refactor Flashinfer TRTLLM attention kernel selection logic (#24600)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/envs.py (+5/-2); vllm/utils/flashinfer.py (+48/-22); vllm/v1/attention/backends/flashinfer.py (+12/-5)
LABELS: bug, ready, v1
ISSUES: #24689 [Bug]: Llama 3.3 70B hangs with full cuda graph for decode-only
BODY: ## Purpose ⏎ #23647 always quantizes query if kv cache type is set to FP8, and will use TRTLLM attention kernel. However, there are lots of cases that do not support TRTLLM attention. ⏎  ⏎ - For the fix, we still try quantizing query first and see if the TRTLLM attention is finally used or not. If it is not used, then we set the query dtype back to the model dtype. ⏎   - This fix still need #24577 to support FP8 KV cache for flashinfer generally ⏎ - Also,  …[truncated]

### L3-fedb75fa27  (L3, 2025-09-17, sha fedb75fa2790, PR #24966)
TITLE: [Bugfix][B200] Fix `cutlass_mla` hang (#24966)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.cutlass_sm100
FILES: csrc/attention/mla/cutlass_sm100_mla/device/sm100_mla.hpp (+8/-0)
LABELS: bug, ready, deepseek
BODY: This PR fixes the hang issue with cutlass_mla when batch size is sufficiently large and kv_splits is high.  ⏎ The solution is to limit the max kv_splits to 2 when batch size >= 1. We avoid limiting batch_size == 1, since larger kv_splits improve low-latency performance.

### L3-1a456c7c90  (L3, 2025-09-17, sha 1a456c7c90af, PR #24991)
TITLE: Aiter mha fp8 fix (#24991)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/attention/ops/rocm_aiter_paged_attn.py (+2/-2); vllm/v1/attention/backends/rocm_aiter_fa.py (+2/-2)
LABELS: rocm, ready, v1
BODY: Fixes accuracy when using VLLM_ROCM_USE_AITER_MHA=1 on gfx architectures that use torch.float8_e4m3fn datatypes ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-3bc18127ff  (L3, 2025-09-18, sha 3bc18127ff1c, PR #25123)
TITLE: [XPU] Whisper model support on XPU Platform (#25123)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+2/-2); vllm/v1/worker/utils.py (+1/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎ Add Whisper model support on XPU ⏎  ⏎ ## Test Plan ⏎  ⏎ VLLM_USE_V1=1 XPU_CCL_BACKEND=xccl CCL_ATL_SHM=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python3 -m vllm.entrypoints.openai.api_server --model openai/whisper-large-v3 --dtype=float16 --enforce-eager --port 8000 --trust-remote-code --max_num_batched_tokens 32768 --gpu-memory-util 0.85 ⏎ ## Test Result ⏎ with this PR:  ⏎  ⏎ server started ⏎  ⏎ Without this pr: ⏎ 1. raise error  …[truncated]

### L3-dc34059360  (L3, 2025-09-18, sha dc3405936090, PR #25178)
TITLE: [ROCm][CI/Build] Use ROCm7.0 as the base (#25178)
SOURCES: dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm (+4/-1); docker/Dockerfile.rocm_base (+12/-49)
LABELS: rocm, ci/build
BODY: Switching the ROCm base image to ROCm7 release with the matching torch 2.8 and triton 3.4 versions ⏎  ⏎ In the final dockerfile also pulling the tags from the upstream repo if it is built from a different one to have the right version based on the upstream tags ⏎  ⏎ The resulting image is pushed to `rocm/vllm-dev:base`

### L3-01a583fea4  (L3, 2025-09-18, sha 01a583fea405, PR #21197)
TITLE: [Kernel] Decouple Tile Size from Block Size in Triton Unified Attention Kernel (#21197)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+70/-52); tests/kernels/attention/test_triton_unified_attention.py (+0/-3)
LABELS: ready, v1
BODY: This PR introduces modifications and extensions to the Triton unified attention kernel (https://github.com/vllm-project/vllm/pull/16828 and https://github.com/vllm-project/vllm/pull/19152) that enable support for hybrid models such Granite 4.0 which combines Mamba-2 and Transformer layers.  ⏎  ⏎ Key changes: ⏎  ⏎ - ~~adopt FlashInfer-style KV cache layout `(num_blocks, 2, block_size, num_kv_heads, head_size)`~~ ⏎  ⏎ - ~~implement `reorder_batch()` function~~ …[truncated]

### L3-bbdc0f2366  (L3, 2025-09-18, sha bbdc0f236699, PR #25104)
TITLE: [ROCm][AITER][Bugfix] Switch AITER to use PIECEWISE_AND_FULL compilation (#25104)
SOURCES: path_core, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+1/-1)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study correctness case vllm:bbdc0f2366: class=integration_backend_cudagraph; symptom=performance_or_availability; introducing=unknown
BODY: ## Purpose ⏎ AiterFlashAttentionBackend (enabled through VLLM_ROCM_USE_AITER_MHA=1) does not work correctly with cudagraph_mode = FULL, since it calls two AITER kernels for prefill (aiter.flash_attn_varlen_func) and decode (aiter.paged_attention_v1). The correct AttentionCGSupport and cudagraph_mode to use here are `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE` and `FULL_AND_PIECEWISE` respectively. ⏎  ⏎ ## Test Plan ⏎ For e.g. Llama 3.3 70B FP8 TP1 ma …[truncated]

### L3-75fb112d80  (L3, 2025-09-18, sha 75fb112d80f6, PR #25106)
TITLE: [Bug] Fix `returned_lse` not Defined issue (#25106)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.cutlass_v1_backend
FILES: vllm/v1/attention/backends/mla/cutlass_mla.py (+3/-4)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Fixes https://github.com/vllm-project/vllm/pull/25106 ⏎  ⏎ ```bash ⏎ (EngineCore_DP1 pid=3176030)   File "/home/wentao/.venv/lib/python3.12/site-packages/torch/_ops.py", line 1158, in __call__ ⏎ (EngineCore_DP1 pid=3176030)     return self._op(*args, **(kwargs or {})) ⏎ (EngineCore_DP1 pid=3176030)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ (EngineCore_DP1 pid=3176030)   File "/home/wentao/vllm-source/vllm/attention/layer.py", line 597, in un …[truncated]

### L3-5f696c33b1  (L3, 2025-09-18, sha 5f696c33b1fb, PR #24872)
TITLE: [New Model] Support BertForTokenClassification / Named Entity Recognition (NER) task (#24872)
SOURCES: path_core
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+11/-1); docs/models/supported_models.md (+11/-0); examples/offline_inference/pooling/README.md (+7/-1); examples/offline_inference/pooling/ner.py (+54/-0); examples/online_serving/pooling/README.md (+6/-0); examples/online_serving/pooling/ner.py (+71/-0); tests/models/language/pooling/test_token_classification.py (+39/-0); tests/models/registry.py (+1/-0); vllm/entrypoints/llm.py (+4/-0); vllm/model_executor/models/bert.py (+52/-0); (+1 more)
LABELS: documentation, new-model, frontend, ready, v1
ISSUES: #24752 [Bug]: Some Bert models no longer compatible for BertForSequenceClassification | #25060 [Bug]: Does not run embedding model sergeyzh/rubert-tiny-turbo
BODY: ## Purpose ⏎  ⏎ Support BertForTokenClassification / Named Entity Recognition (NER) task ⏎  ⏎ fix bert + Flex Attention + torch.comple by @Isotr0py  ⏎ (see: https://github.com/vllm-project/vllm/pull/24872#discussion_r2353252290 ⏎  ⏎ Fix #24752 ⏎ Fix #25060  ⏎  ⏎ ## Test Plan ⏎  ⏎ tests/models/language/pooling/test_token_classification.py ⏎  ⏎ ## Test Result ⏎  ⏎ pass ⏎  ⏎ --- ⏎ [details omitted]

### L3-a684c0124c  (L3, 2025-09-19, sha a684c0124cb8, PR #25146)
TITLE: [bugfix] fix MHA for models like OpenGVLab/InternVL3_5-38B (#25146)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+5/-3)
LABELS: ready
BODY: ## Purpose ⏎ The query of models like OpenGVLab/InternVL3_5-38B is 4 dimension so will pop error: ⏎ ``` ⏎ ^[[1;36m(Worker_TP2 pid=946)^[[0;0m ERROR 09-15 13:21:14 [multiproc_executor.py:654]   File "/workspace/vllm/vllm/attention/layer.py", line 383, in forward ⏎  ⏎ ^[[1;36m(Worker_TP2 pid=946)^[[0;0m ERROR 09-15 13:21:14 [multiproc_executor.py:654]     bsz, q_len, _ = query.size() ⏎  ⏎ ^[[1;36m(Worker_TP2 pid=946)^[[0;0m ERROR 09-15 13:21:14 [multiproc_execut …[truncated]

### L3-a3d087adec  (L3, 2025-09-19, sha a3d087adecad, PR #22188)
TITLE: [P/D][Nixl] Introduce `KVTransferMetrics` and aggregation strategy (#22188)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_nixl_connector.py (+210/-1); vllm/distributed/kv_transfer/kv_connector/utils.py (+18/-3); vllm/distributed/kv_transfer/kv_connector/v1/base.py (+21/-1); vllm/distributed/kv_transfer/kv_connector/v1/metrics.py (+100/-0); vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py (+65/-3); vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py (+65/-3); vllm/v1/core/sched/scheduler.py (+17/-10); vllm/v1/metrics/loggers.py (+7/-1); vllm/v1/metrics/stats.py (+2/-1); vllm/v1/outputs.py (+10/-1); (+1 more)
LABELS: ready, v1, kv-connector
BODY: This PR provides support for a general `KVTransferMetrics` object while laying the ground for recording stats, but without tracking actual metrics just yet, waiting for NIXL to expose actual telemetry.  ⏎ Adding those should be a matter of expanding `record_transfer` and providing `aggregate`+`reduce` operations. ⏎  ⏎ Therefore, this PR is focusing on interfaces and plumbing required for aggregating and reducing KVStats across multiple different kvconn …[truncated]

### L3-5089fd749c  (L3, 2025-09-19, sha 5089fd749cbe, PR #25242)
TITLE: [V0 Deprecation] Remove V0 logic from `get_input_embeddings` interface (#25242)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/hyperclovax_vision.py (+17/-28); vllm/model_executor/models/interfaces.py (+0/-24); vllm/model_executor/models/ultravox.py (+4/-15); vllm/model_executor/models/utils.py (+1/-17)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Remove V0-related code from `get_input_embeddings` in order to facilitate #16229 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-a2a5f79e09  (L3, 2025-09-19, sha a2a5f79e0907, PR #24390)
TITLE: Optimize triton unified attention performance for sliding window attention (#24390)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+24/-2); tests/kernels/attention/test_triton_unified_attention.py (+1/-1)
LABELS: performance, ready
BODY: Summary: ⏎ Optimize kernel_unified_attention_2d attention performance for sliding window attention by pruning unused tiles   ⏎  ⏎ Before this change, GPT-OSS + triton attention backend was spending equal amount of time between global and local attention due to attn score still calculated by masked out sequences ⏎ <img width="1556" height="303" alt="gpt-oss-before" src="https://github.com/user-attachments/assets/951c0cc6-3cd3-4431-97a8-5d8c5a83c044" /> ⏎  ⏎ A …[truncated]

### L3-48ecb4438b  (L3, 2025-09-19, sha 48ecb4438b28, PR #21126)
TITLE: [Perf] Use FlashInfer RoPE for RotaryEmbedding.forward_cuda when available (#21126)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/rotary_embedding/base.py (+28/-10); vllm/model_executor/layers/rotary_embedding/common.py (+46/-0); vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py (+1/-3); vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py (+1/-1); vllm/model_executor/layers/rotary_embedding/mrope.py (+2/-0)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Enables FlashInfer RoPE for the cuda forward impl when custom op is set.  ⏎ Microbenchmarks show ~2x speedup vs our custom op, but haven't seen noticeable end-to-end results yet. ⏎  ⏎ ## Test Plan ⏎  ⏎ lm-eval with llama 8b ⏎  ⏎ ## Test Result ⏎  ⏎ I found no noticeable performance difference so leaving this not enabled by default. Looking at the profiles, it seems like we can possibly introduce more  If we want to enable it by default, this is the cha …[truncated]

### L3-aed16879a9  (L3, 2025-09-19, sha aed16879a919, PR #25252)
TITLE: Move `ModelConfig` from `config/__init__.py` to `config/model.py` (#25252)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: tests/conftest.py (+2/-1); tests/distributed/test_pipeline_parallel.py (+1/-1); tests/models/test_initialization.py (+2/-3); tests/v1/sample/test_logprobs.py (+4/-5); vllm/config/__init__.py (+14/-2094); vllm/config/model.py (+2006/-0); vllm/config/scheduler.py (+2/-6); vllm/config/utils.py (+99/-1); vllm/engine/arg_utils.py (+6/-9); vllm/model_executor/model_loader/utils.py (+3/-4); (+3 more)
LABELS: new-model, ready, v1
BODY: Part of #18953 ⏎  ⏎ Notable changes: ⏎  ⏎ - `RunnerType` belongs to the scheduler now (that's where the field actually lives) ⏎ - `ModelImpl` and `LogprobsMode` both converted to `Literal` for easier handling (i.e. `ModelImpl.VLLM` or `ModelImplVLLM.value`? Is not a question we need to cate about anymore) ⏎ - A few fuctions that are not directly related to any config have been moved to `utils.py`

### L3-12aed7e453  (L3, 2025-09-19, sha 12aed7e453ae, PR #25174)
TITLE: Encoder model support for the Transformers backend (#25174)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+4/-4); docs/models/supported_models.md (+25/-12); tests/models/test_transformers.py (+35/-1); vllm/model_executor/models/transformers.py (+47/-7)
LABELS: documentation, ready
BODY: Adds support for encoder models to the Transformers backend. ⏎  ⏎ ## Depends on ⏎  ⏎ I have conditioned this feature on the Transformers version, so as soon as the following dependency is merged we can merge this PR and users installing vLLM and Transformers from main can use this feature: ⏎  ⏎ ## Changes ⏎  ⏎ - Use `EncoderOnlyAttention` if an encoder model is detected ⏎ - Skip `position_ids` buffers if they're found in the checkpoint because vLLM always passes ` …[truncated]

### L3-b1a63d1b3b  (L3, 2025-09-19, sha b1a63d1b3be9, PR #25040)
TITLE: [BugFix] Make FlashInferMetadataBuilder non-blocking (#25040)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+3/-2)
LABELS: ready, v1
BODY: ## Purpose ⏎ The blocking H2D memcpys breaks overlap scheduler #23569, setting them to non-blocking fixes it. ⏎ The correctness is ensured by vllm/v1/worker/gpu_model_runner.py:2112 ⏎ ```python3 ⏎             if self.prepare_inputs_event is not None: ⏎                 # Ensure prior step has finished with reused CPU tensors. ⏎                 self.prepare_inputs_event.synchronize() ⏎             try: ⏎                 # Prepare the decoder inputs. ⏎                …[truncated]

### L3-ee7a66dd9a  (L3, 2025-09-19, sha ee7a66dd9a5e, PR #25276)
TITLE: allow disable flashinfer prefill (#25276)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.common_v1
FILES: vllm/envs.py (+3/-0); vllm/v1/attention/backends/mla/common.py (+2/-1)
LABELS: ready, v1
BODY: Summary: GB200 FlashInfer Prefill is not compatible with CutlassMLA FP8, allowing disable it for now. ⏎  ⏎ Differential Revision: D81994905

### L3-9607d5eb44  (L3, 2025-09-19, sha 9607d5eb4497, PR #25101)
TITLE: [Hybrid Allocator] Support full attention with different hidden size  (#25101)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_kv_cache_utils.py (+103/-15); vllm/v1/core/kv_cache_utils.py (+108/-38); vllm/v1/engine/core.py (+6/-10); vllm/v1/kv_cache_interface.py (+70/-0); vllm/v1/worker/gpu_model_runner.py (+36/-29); vllm/v1/worker/utils.py (+2/-1)
LABELS: ready, v1
ISSUES: #22432 [Feature]: Support Eagle Draft Model with different number of KV heads
BODY: ## Purpose ⏎  ⏎ Hybrid allocator requires all layers have the same kv_hidden_size. But some model breaks this assumption, e.g.,  ⏎ 1. spec decode, the MTP layer and the main model, like https://github.com/vllm-project/vllm/issues/22432 ⏎ 2. recent native sparse attention like minicpm 4.1. It needs to run attention.0 on a smaller KV cache to select tokens, and then do attention.1 only on the selected tokens. Both attention.0 and attention.1 can be regarde …[truncated]

### L3-8945b001db  (L3, 2025-09-20, sha 8945b001db32, PR #24281)
TITLE: [torch.compile] CUDAGraph Inductor partition integration (#24281)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+2/-0); tests/compile/piecewise/test_simple.py (+60/-11); tests/compile/silly_attention.py (+1/-0); tests/compile/test_full_graph.py (+58/-1); tests/compile/test_fusion_attn.py (+15/-1); vllm/compilation/backends.py (+8/-2); vllm/compilation/decorators.py (+55/-2); vllm/config/compilation.py (+73/-11); vllm/v1/cudagraph_dispatcher.py (+8/-4)
LABELS: documentation, performance, new-model, rocm, frontend, ready, torch.compile, v1, multi-modality, llama
ISSUES: #23261 [RFC]: Address piecewise graph splitting and attention fusion incompatibility
BODY: Depends on pytorch/pytorch#162207 [landed and available in PyTorch 2.9]. ⏎  ⏎ Command: ⏎ ``` ⏎ vllm bench latency -O.cudagraph_mode=PIECEWISE -O.use_inductor_graph_partition=true ⏎ ``` ⏎  ⏎ ## Tests ⏎  ⏎ #### Test1 ⏎ `test_simple_inductor_graph_partition` checks that, when `use_inductor_graph_partition=True`, we have 1 num_piecewise_graphs_seen and 1 num_backend_compilations. By contrast, `test_simple_piecewise_compile` checks that, when `use_inductor_graph_partitio …[truncated]

### L3-6c5f82e5aa  (L3, 2025-09-20, sha 6c5f82e5aa87, PR #25298)
TITLE: [BUG FIX][NON-CUDA]quick fix to avoid call cudagraph_unsafe in attention (#25298)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+6/-2)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Fix failing for non cuda device when calling torch._C.Tag.cudagraph_unsafe ⏎ from https://github.com/vllm-project/vllm/pull/24281 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-e08a3a3fdb  (L3, 2025-09-20, sha e08a3a3fdbdb, PR #25299)
TITLE: [CI Failure] Disable FlashInfer RoPE to unblock CI (#25299)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/rotary_embedding/base.py (+7/-7)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ https://github.com/vllm-project/vllm/pull/21126 seems to have broken V1 Test entrypoints, like due to numerical differences. Disabling for now ⏎  ⏎ <img width="911" height="961" alt="image" src="https://github.com/user-attachments/assets/f237cf84-c0e0-4e22-9f2e-edc67716e5c9" /> ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-bf8b26cad1  (L3, 2025-09-20, sha bf8b26cad17f, PR #23558)
TITLE:  Generate _ModelInfo properties file when loading to improve loading speed (#23558)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/logging_utils/__init__.py (+2/-0); vllm/logging_utils/log_time.py (+32/-0); vllm/model_executor/model_loader/weight_utils.py (+44/-0); vllm/model_executor/models/registry.py (+89/-3)
LABELS: documentation, new-model, rocm, ready
ISSUES: #19317 [Performance]: Option for disabling model info collection in subprocess
BODY: Save changed _ModelInfo properties for a model in json file under `$VLLM_CACHE_ROOT/modelinfos` during load. ⏎  ⏎ Next time the same model loads, the _ModelInfo class will be created from the saved properties, improving loading times. ⏎  ⏎ CLOSE #19317 ⏎  ⏎ ## Purpose ⏎ Improve loading times during lazy creation of a _ModelInfo class ⏎ The solution avoids a OS process creation with the imports and CUDA initialization overhead. ⏎  ⏎ ## Test Plan ⏎ Start vLLM server wit …[truncated]

### L3-86647d1cd0  (L3, 2025-09-20, sha 86647d1cd0f3, PR #25320)
TITLE: [V0 Deprecation] Remove V0 Output Processor (#25320)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/build_cython.py (+0/-39); vllm/engine/output_processor/__init__.py (+0/-0); vllm/engine/output_processor/interfaces.py (+0/-59); vllm/engine/output_processor/single_step.py (+0/-145); vllm/engine/output_processor/stop_checker.py (+0/-139); vllm/v1/engine/detokenizer.py (+40/-2)
LABELS: documentation, frontend, ready, ci/build, v1
BODY: 

### L3-c99db8c8dd  (L3, 2025-09-20, sha c99db8c8ddd5, PR #25321)
TITLE: [V0 Deprecation] Remove V0 core (#25321)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.mla.triton_v0
FILES: vllm/attention/backends/differential_flash_attn.py (+4/-8); vllm/attention/backends/dual_chunk_flash_attn.py (+3/-7); vllm/attention/backends/flash_attn.py (+4/-8); vllm/attention/backends/mla/common.py (+4/-9); vllm/attention/backends/placeholder_attn.py (+3/-8); vllm/attention/backends/rocm_aiter_mla.py (+2/-5); vllm/attention/backends/utils.py (+2/-7); .buildkite/test-pipeline.yaml (+0/-3); .github/CODEOWNERS (+0/-2); pyproject.toml (+0/-2); (+19 more)
LABELS: documentation, rocm, frontend, ready, ci/build, v1
BODY: 

### L3-1cd885bd54  (L3, 2025-09-20, sha 1cd885bd540c, PR #25328)
TITLE: [V0 Deprecation] Remove V0 model runner base & simplify worker base (#25328)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+5/-10); vllm/attention/backends/utils.py (+2/-6); vllm/worker/model_runner_base.py (+0/-307); vllm/worker/worker_base.py (+4/-383)
BODY: 

### L3-12dbd834cf  (L3, 2025-09-20, sha 12dbd834cf33, PR #25330)
TITLE: [V0 Deprecation] Remove from_seq_group methods (#25330)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/multimodal/base.py (+1/-121); vllm/outputs.py (+1/-194)
LABELS: multi-modality
BODY: 

### L3-035fd2bd2c  (L3, 2025-09-21, sha 035fd2bd2cd2, PR #25005)
TITLE: [Multi Modal][Performance] Fused Q,K's apply_rope in more models (#25005)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/ernie45_vl.py (+9/-6); vllm/model_executor/models/glm4_1v.py (+10/-7); vllm/model_executor/models/qwen2_vl.py (+10/-6)
LABELS: ready, qwen
BODY: Follow up of https://github.com/vllm-project/vllm/pull/24511. Apply the combined Q,K's apply_rope and combined rearrange in more models  ⏎  ⏎ Related: https://github.com/vllm-project/vllm/issues/23880, https://github.com/vllm-project/vllm/issues/23884 ⏎  ⏎ cc: @ywang96 @DarkLight1337 @Isotr0py

### L3-cf56cf78b4  (L3, 2025-09-21, sha cf56cf78b47e, PR #24089)
TITLE: [V1] Add sliding window support to Flex Attention backend (#24089)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+72/-18); tests/v1/attention/test_attention_backends.py (+157/-51)
LABELS: ready, v1
ISSUES: #24358 [Bug]:  FlexAttention does not support sliding window yet.
BODY: ## Purpose ⏎ - Fix #24358 ⏎ - Add sliding windows support to flex attention backend ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest -s -v tests/v1/attention/test_attention_backends.py ⏎ ``` ⏎  ⏎ ## Test Result ⏎ Test should still pass. ⏎  ⏎ --- ⏎ [details omitted]

### L3-26e673fe93  (L3, 2025-09-21, sha 26e673fe9303, PR #25332)
TITLE: [V0 Deprecation] Remove V0 Sequence class & Sampler (#25332)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/conftest.py (+1/-1); tests/models/multimodal/generation/test_granite_speech.py (+1/-1); tests/models/multimodal/generation/test_phi4mm.py (+1/-1); tests/models/multimodal/generation/test_pixtral.py (+1/-1); tests/models/multimodal/generation/vlm_utils/model_utils.py (+1/-1); tests/models/multimodal/generation/vlm_utils/types.py (+1/-1); tests/models/utils.py (+1/-1); tests/tokenization/test_detokenize.py (+1/-139); tests/tool_use/test_jamba_tool_parser.py (+1/-1); tests/tool_use/test_qwen3coder_tool_parser.py (+1/-1); (+17 more)
LABELS: speculative-decoding, ready, v1, multi-modality, tool-calling, qwen
BODY: 

### L3-0ff8ebb2d7  (L3, 2025-09-21, sha 0ff8ebb2d700, PR #25334)
TITLE: [V0 Deprecation] Remove async_output_proc, preemption mode, delay factor (#25334)
SOURCES: release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: tests/detokenizer/test_stop_strings.py (+5/-41); tests/v1/engine/test_processor_multi_modal_uuids.py (+0/-10); tests/v1/test_oracle.py (+0/-18); vllm/config/__init__.py (+0/-4); vllm/config/model.py (+6/-42); vllm/config/scheduler.py (+1/-14); vllm/engine/arg_utils.py (+0/-34); vllm/entrypoints/llm.py (+0/-4); vllm/executor/uniproc_executor.py (+0/-4); vllm/platforms/cpu.py (+0/-4); (+5 more)
LABELS: rocm, frontend, tpu, ready, v1
BODY: 

### L3-1c3ffdbecc  (L3, 2025-09-21, sha 1c3ffdbeccaf, PR #25345)
TITLE: [V0 Deprecation] Remove V0 sampling metadata (#25345)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_llava.py (+3/-5); tests/plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_opt.py (+3/-5); vllm/model_executor/__init__.py (+0/-2); vllm/model_executor/layers/logits_processor.py (+0/-2); vllm/model_executor/models/apertus.py (+1/-4); vllm/model_executor/models/arcee.py (+3/-4); vllm/model_executor/models/arctic.py (+1/-4); vllm/model_executor/models/aria.py (+2/-5); vllm/model_executor/models/aya_vision.py (+1/-4); vllm/model_executor/models/baichuan.py (+1/-4); (+131 more)
LABELS: tpu, speculative-decoding, ready, v1, llama, qwen, deepseek, gpt-oss
BODY: 

### L3-bc6e542d9f  (L3, 2025-09-21, sha bc6e542d9fc4, PR #25351)
TITLE: Remove V0 attention backends (#25351)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.trtllm_gen, L3.rocm.rocm_flash_attn_v0, L3.mla.triton_v0, L3.mla.flashmla_v0_adapter, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/attention/backends/differential_flash_attn.py (+0/-931); vllm/attention/backends/dual_chunk_flash_attn.py (+0/-1495); vllm/attention/backends/flash_attn.py (+0/-929); vllm/attention/backends/flashmla.py (+0/-227); vllm/attention/backends/mla/__init__.py (+0/-0); vllm/attention/backends/mla/common.py (+0/-1305); vllm/attention/backends/rocm_aiter_mla.py (+0/-407); vllm/attention/backends/rocm_flash_attn.py (+0/-953); vllm/attention/backends/triton_mla.py (+0/-111); vllm/attention/backends/utils.py (+6/-8); (+18 more)
LABELS: documentation, rocm, ready, qwen, deepseek, kv-connector
BODY: ## Summary ⏎ - restrict CUDA and ROCm attention backend selection to the V1 engine and raise errors when a removed V0 backend is requested ⏎ - update runtime helpers and tests to consume the V1 attention metadata/backends and add local ALiBi utilities for kernel tests ⏎ - skip V0-only model initialization coverage and drop the legacy attention selector tests that depended on V0 backends ⏎  ⏎ ## Testing ⏎ - pytest tests/kernels/attention/test_prefix_prefill.p …[truncated]

### L3-f92d952632  (L3, 2025-09-22, sha f92d95263254, PR #25366)
TITLE: [V0 Deprecation] Remove `MultiModalPlaceholderMap` (#25366)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+0/-10); vllm/attention/backends/placeholder_attn.py (+1/-22); vllm/attention/backends/utils.py (+0/-18); vllm/v1/attention/backends/cpu_attn.py (+0/-1); tests/kernels/utils.py (+0/-2); vllm/multimodal/__init__.py (+0/-2); vllm/multimodal/base.py (+1/-73)
LABELS: ready, v1, multi-modality
BODY: ## Purpose ⏎  ⏎ Now that V0 model runner is gone, we can remove this. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-cfbee3d0e7  (L3, 2025-09-22, sha cfbee3d0e725, PR #25274)
TITLE: [CLI env var] Add VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH in env variables (#25274)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.trtllm_gen, L3.mla.flashattn
FILES: vllm/envs.py (+8/-0); vllm/v1/attention/backends/flash_attn.py (+3/-4); vllm/v1/attention/backends/mla/flashattn_mla.py (+3/-5); tests/compile/piecewise/test_full_cudagraph.py (+9/-2); tests/v1/cudagraph/test_cudagraph_mode.py (+9/-2)
LABELS: documentation, performance, structured-output, frontend, ready, ci/build, v1, multi-modality, qwen, gpt-oss
BODY: ## Purpose ⏎  ⏎ Add `VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH` in env variables so users have control over cuda graph max_num_splits in cli level. ⏎  ⏎ When applying #23958, realized the [_DEFAULT_MAX_NUM_SPLITS_FOR_CUDA_GRAPH](https://github.com/MatthewBonanni/vllm/blob/235670f424885751ba47080b173e4e0c33444a58/vllm/v1/attention/backends/mla/flashattn_mla.py#L28) value is copied from Flash_attn ([code ref](https://github.com/MatthewBonanni/vllm/blob …[truncated]

### L3-417a164af6  (L3, 2025-09-22, sha 417a164af66a, PR #25374)
TITLE: [Misc] Remove unused encoder-decoder error strings (#25374)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/utils.py (+0/-5); vllm/utils/__init__.py (+0/-58)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ After removing encoder-decoder support, these strings aren't used anymore. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-6d0b827cbd  (L3, 2025-09-22, sha 6d0b827cbd05, PR #25362)
TITLE: [V0 Deprecation] Remove V0-only methods in multi-modal registry (#25362)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/generation/test_qwen2_vl.py (+0/-1); vllm/multimodal/registry.py (+1/-31)
LABELS: ready, multi-modality, qwen
BODY: ## Purpose ⏎  ⏎ Remove dead code ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-175811e3b5  (L3, 2025-09-22, sha 175811e3b53f, PR #24648)
TITLE: [V1][Attention] Split triton_attn in triton-only and rocm specific backends  (#24648)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.platform.rocm_selection
FILES: vllm/engine/arg_utils.py (+1/-0); vllm/platforms/interface.py (+1/-0); vllm/platforms/rocm.py (+10/-0); vllm/v1/attention/backends/rocm_attn.py (+426/-0); vllm/v1/attention/backends/triton_attn.py (+45/-124)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎  ⏎ This PR splits the `triton_attn` backend of V1 into two backends: One triton-only and platform-independent `triton_attn`  and one rocm-specific `rocm_attn`, including aiter kernels.  ⏎ This facilitates easier maintenance of both backends. Also, adaptations to the vllm-internal triton backend then don't need to ensure full compatibility with the external library aiter. For example, the standardization of kv-cache layouts in #21624 was bl …[truncated]

### L3-0b7bed9c38  (L3, 2025-09-22, sha 0b7bed9c386d, PR #25184)
TITLE: [Performance] Remove input pads in cutlass_mla and optimize v_proj output handling (#25184)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.cutlass_v1_backend
FILES: vllm/v1/attention/backends/mla/common.py (+39/-6); vllm/v1/attention/backends/mla/cutlass_mla.py (+16/-14)
LABELS: documentation, ready, v1
BODY: This PR removes the need to pad cutlass mla inputs (q_nope and q_pe) to max_heads==128 by pre-padding the buffers that are used by previous operations. Also, the PR improves the way v_proj handles the output by reusing the output buffer earlier inside torch.bmm. For DeepSeekR1 on 8xB200 batch_size==32, decode iteration TPOT performance imporves from 18.87 to 18.25ms, about 3.3%. ⏎  ⏎ Verified correctness with: lm_eval --model vllm --model_args pretra …[truncated]

### L3-78237e43bf  (L3, 2025-09-22, sha 78237e43bf3a, PR #25414)
TITLE: [Bugfix] Remove contiguous output req for context parallel MLA (#25414)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/common.py (+0/-1)
LABELS: bug, ready, ci-failure
BODY: ## Purpose ⏎  ⏎ "Distributed Tests (B200)" has been failing due to this assert on the output being contiguous. This doesn't seem to be required and if I remove the assert the test is green  ⏎  ⏎ https://buildkite.com/vllm/ci/builds/31718/steps/canvas?sid=01996a6e-b6da-4752-9ec9-dfb3968b9a2e ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ lm_eval --model local-completions --model_args model=deepseek-ai/DeepSeek-V2-Lite-Chat,base_url=http://0.0.0.0:8000/v1/completions, …[truncated]

### L3-4741239db7  (L3, 2025-09-22, sha 4741239db7b7, PR #25290)
TITLE: [Bug] Fix Long Context OOM Issue (#25290)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+1/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Context from @smarterclayton  ⏎  ⏎ >Trying to test a very long context response - 128k input, 1 output token.  I used DeepSeek-V3.1 and changed --max-model-len (although deepseek v3.1 is 128k automatically). ⏎ >When I try to run a single request I'm getting an OOM: ⏎  ⏎ ```bash ⏎ (EngineCore_DP0 pid=948) ERROR 09-16 14:10:49 [v1/engine/core.py:720] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.12 GiB. GPU 0 has a total capacity  …[truncated]

### L3-c625f9043c  (L3, 2025-09-23, sha c625f9043c5b, PR #25409)
TITLE: [V0 deprecation] Remove `_set_default_args_v0` function (#25409)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/engine/arg_utils.py (+11/-72)
LABELS: ready
BODY: ## Purpose ⏎ - Remove `_set_default_args_v0` since v0 engine has been removed ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-a903669e10  (L3, 2025-09-23, sha a903669e10cc, PR #25400)
TITLE: [V1] Remove V0 code paths for Hybrid models (#25400)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+51/-0); tests/models/language/generation/test_hybrid.py (+19/-36); tests/models/registry.py (+6/-7); vllm/model_executor/layers/mamba/abstract.py (+1/-4); vllm/model_executor/layers/mamba/linear_attn.py (+40/-67); vllm/model_executor/layers/mamba/mamba2_metadata.py (+0/-177); vllm/model_executor/layers/mamba/mamba_mixer.py (+50/-99); vllm/model_executor/layers/mamba/mamba_mixer2.py (+56/-116); vllm/model_executor/layers/mamba/mamba_utils.py (+2/-18); vllm/model_executor/layers/mamba/ops/causal_conv1d.py (+0/-3); (+21 more)
LABELS: new-model, ready, v1, qwen
BODY: ## Purpose ⏎  ⏎ Remove V0 code path from the following models: ⏎  ⏎ Please now that the model `phi4flash` has not been ported to V1 yet (#23996) and is non-trival to do so (requires porting differential attention backend). This PR removes the modeling code for that model. It can be re-added once the porting is done. The code will be useless until then anyway since V0 has been removed.  ⏎  ⏎ ## Test Plan ⏎  ⏎ Hybrid model test should verify nearly all of the abov …[truncated]

### L3-a8ffc4f0f2  (L3, 2025-09-23, sha a8ffc4f0f2d0, PR #25508)
TITLE: [Bugfix] Lower gpt-oss max cudagraph size to 992 to be compatible with FA3 (#25508)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/config.py (+5/-5)
LABELS: bug, gpt-oss
DEEP_STUDY: deep-study: this PR was reverted by PR 28345 (explicit_rollback, reason=other)
BODY: ## Purpose ⏎  ⏎ Due to compiling full cudagraphs by default now in https://github.com/vllm-project/vllm/pull/25444, gpt-oss crashes on Hopper. ⏎  ⏎ ``` ⏎ vllm serve openai/gpt-oss-20b ⏎ ... ⏎ (EngineCore_DP0 pid=2061997)   File "/home/mgoin/code/vllm/vllm/v1/worker/gpu_model_runner.py", line 3503, in create_attn_groups ⏎ (EngineCore_DP0 pid=2061997)     attn_metadata_builders.append(attn_backend.get_builder_cls()( ⏎ (EngineCore_DP0 pid=2061997)                     …[truncated]

### L3-100b630a60  (L3, 2025-09-23, sha 100b630a604a, PR #24503)
TITLE: [V1][Kernel] Add triton implementation for `reshape_and_cache_flash` (#24503)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.v1_backend
FILES: vllm/attention/ops/triton_reshape_and_cache_flash.py (+176/-0); vllm/v1/attention/backends/triton_attn.py (+12/-3); benchmarks/kernels/benchmark_reshape_and_cache_flash.py (+67/-11); tests/kernels/attention/test_cache.py (+21/-6)
LABELS: performance, ready, v1
BODY: ## Purpose ⏎  ⏎ This PR adds a triton implementation of the `reshape_and_cache` kernel, this helps reducing the dependency on non-pytorch/triton kernels for the `triton_attn` kernel.  ⏎  ⏎ The kernel itself has the same (or slightly better) performance on H100 and MI300 as the CUDA kernel.  ⏎ <img width="2013" height="798" alt="image" src="https://github.com/user-attachments/assets/cc0d1b1f-1a09-46b3-90e0-f7895baf5df5" /> ⏎ <img width="2013" height="798" alt …[truncated]

### L3-b6a136b58c  (L3, 2025-09-23, sha b6a136b58c10, PR #25471)
TITLE: [CI/Build] Fix disabled v1 attention backend selection test (#25471)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_attention_selector.py (+1/-2)
LABELS: ready
BODY: ## Purpose ⏎ - Related to #25364 ⏎ - Fix disabled `test_env` test in `tests/kernels/attention/test_attention_selector.py` ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest -s -v tests/kernels/attention/test_attention_selector.py -k test_env ⏎ ``` ⏎  ⏎ ## Test Result ⏎ All tests should pass now: ⏎ ``` ⏎ tests/kernels/attention/test_attention_selector.py::test_env[cuda_TRITON_MLA_mla_T_blks16] INFO 09-23 19:32:46 [cuda.py:278] Using Triton MLA backend on V1 engine. ⏎ W0923 19:32:46.830000 8 …[truncated]

### L3-4f2954f724  (L3, 2025-09-23, sha 4f2954f7240b, PR #25522)
TITLE: Fix triton_reshape_and_cache_flash.py triton import (#25522)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/triton_reshape_and_cache_flash.py (+1/-2)
LABELS: bug
BODY: ## Purpose ⏎  ⏎ Currently failing pre-commit ⏎  ⏎ ``` ⏎ Forbid direct 'import triton'.......................................................................Failed ⏎ - hook id: forbid-direct-triton-import ⏎ - exit code: 1 ⏎  ⏎ ❌ Forbidden direct `import triton` detected. ➤ Use `from vllm.triton_utils import triton` instead. ⏎  ⏎ ❌ vllm/attention/ops/triton_reshape_and_cache_flash.py:5: import triton ⏎ ❌ vllm/attention/ops/triton_reshape_and_cache_flash.py:6: import trito …[truncated]

### L3-cc1dc7ed6d  (L3, 2025-09-23, sha cc1dc7ed6d2d, PR #24845)
TITLE: [Core/DBO][2/N] Dual-Batch Overlap add DeepEP High Throughput support and Prefill support (#24845)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+41/-3); tests/v1/attention/test_attention_splitting.py (+82/-1); tests/v1/spec_decode/test_eagle.py (+2/-4); vllm/config/__init__.py (+7/-5); vllm/config/parallel.py (+10/-4); vllm/distributed/device_communicators/all2all.py (+19/-9); vllm/distributed/device_communicators/base_device_communicator.py (+6/-0); vllm/engine/arg_utils.py (+6/-0); vllm/envs.py (+11/-0); vllm/model_executor/layers/fused_moe/deepep_ht_prepare_finalize.py (+51/-25); (+9 more)
LABELS: documentation, speculative-decoding, ready, v1
BODY: ## Purpose ⏎  ⏎ - Add support for having prefill requests inside ubatches; this is done by allowing a request to be split across ubatches with the second batch effectively chunked-prefill continuing on from the the first (since it always runs after) ⏎ - Add support for the DeepEP high-throughput kernels with SM control for DeepGEMM and DeepEP (shout-out to @yewentao256 for the HT support) ⏎ - General cleanups ⏎  ⏎ ## Test Plan ⏎  ⏎ lm_eval  ⏎  ⏎ ## Test Result ⏎  ⏎ `exp …[truncated]

### L3-e0b24ea030  (L3, 2025-09-23, sha e0b24ea0305e, PR #25495)
TITLE: [Perf] Increase default max splits for FA3 full cudagraphs (#25495)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+2/-2)
LABELS: performance, ready
BODY: https://github.com/vllm-project/vllm/pull/25274 provides evidence that 32 would be a much better default ⏎  ⏎ due to full-CG potential becoming default https://github.com/vllm-project/vllm/pull/25444 seems like a good time to improve this

### L3-1210e4d95b  (L3, 2025-09-23, sha 1210e4d95b51, PR #25509)
TITLE: [Bugfix] [B200] cutlass_mla - ensure kv_split == 1 for batch size > 1 (#25509)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.cutlass_sm100
FILES: csrc/attention/mla/cutlass_sm100_mla/device/sm100_mla.hpp (+2/-2)
LABELS: bug, ready
BODY: Tests showed that limiting kv_split to 2 is not enough and it needs to be limited to 1 to ensure no-hangs (1 disables the sm100 cutlass reduction kernel)

### L3-9df8da548e  (L3, 2025-09-23, sha 9df8da548e92, PR #25478)
TITLE: [BugFix] Fix MLA assert with CUTLASS MLA (#25478)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+46/-18)
LABELS: ready, v1
BODY: Cutlass MLA has a block_size of 128 so following https://github.com/vllm-project/vllm/pull/25290 this would assert since default `max_num_seqs` is 1024 ⏎  ⏎ also make sure we capture the size of the up-projection in the profile run (this was in v0 backend but failed to make it to v0)

### L3-969b4da3a6  (L3, 2025-09-23, sha 969b4da3a6ab, PR #25510)
TITLE: [V0 Deprecation] Remove placeholder attn (#25510)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/backends/placeholder_attn.py (+0/-314); vllm/attention/layer.py (+0/-3); vllm/attention/selector.py (+0/-9); tests/kernels/attention/test_attention_selector.py (+10/-27); vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py (+0/-1)
LABELS: ready, kv-connector
BODY: ## Purpose ⏎  ⏎ Remove placeholder attention backend. It is no longer needed for Mamba models in V1, since each mamba/linear attention layer has its own "real" attention backend. ⏎  ⏎ ## Test Plan ⏎  ⏎ Let's see if CI passes ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-c30b405b8f  (L3, 2025-09-23, sha c30b405b8f02, PR #25196)
TITLE: [Spec Decode] Enable FlashInfer Spec Decoding (#25196)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.xformers.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.mla.flashattn, L3.dispatch.abstract_interface
FILES: vllm/utils/flashinfer.py (+15/-0); vllm/v1/attention/backends/flashinfer.py (+56/-13); vllm/v1/attention/backends/mla/common.py (+2/-2); vllm/v1/attention/backends/mla/flashattn_mla.py (+2/-2); vllm/v1/attention/backends/utils.py (+66/-6); vllm/v1/attention/backends/xformers.py (+2/-2); tests/v1/attention/test_attention_splitting.py (+108/-1); vllm/v1/attention/backends/gdn_attn.py (+3/-3); vllm/v1/attention/backends/linear_attn.py (+1/-2); vllm/v1/attention/backends/mamba_attn.py (+1/-1); (+2 more)
LABELS: speculative-decoding, ready, v1
BODY: ## Purpose ⏎  ⏎ This PR enables FlashInfer for speculative decoding. When possible, the `trtllm-gen` decode-optimized kernel is used for speculative decoding. The fallback case is the prefill kernel, which can handle arbitrary query lengths but is not as performant.  ⏎  ⏎ This PR depends on #25183 for the refactor of the batch reordering threshold variable. ⏎  ⏎ Here's an example launch command for EAGLE3 on 1xB200: ⏎ ```bash ⏎ vllm serve meta-llama/Llama-3.1-8B …[truncated]

### L3-7361ab379f  (L3, 2025-09-23, sha 7361ab379f81, PR #25512)
TITLE: Remove redundant mutates_args and dispatch_key for direct_register_custom_op (#25512)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+0/-3); vllm/compilation/collective_fusion.py (+0/-1); vllm/distributed/device_communicators/pynccl.py (+0/-1); vllm/distributed/parallel_state.py (+0/-7); vllm/lora/ops/triton_ops/lora_expand_op.py (+0/-2); vllm/lora/ops/triton_ops/lora_shrink_op.py (+0/-2); vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py (+0/-1); vllm/model_executor/layers/fused_moe/fused_marlin_moe.py (+0/-1); vllm/model_executor/layers/fused_moe/fused_moe.py (+0/-1); vllm/model_executor/layers/fused_moe/layer.py (+0/-2); (+18 more)
LABELS: rocm, ready, qwen, deepseek
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-1983609239  (L3, 2025-09-24, sha 1983609239ca, PR #25520)
TITLE: [Bugfix] Use a separate FlashInfer workspace buffer for trtllm-gen (#25520)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+12/-2)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ I've been getting some rare illegal memory accesses when developing using trtllm-gen flashinfer kernels. ⏎  ⏎ I believe the main issue comes down to the fact that the trtllm-gen and non-trtllm-gen kernels need separate workspaces. Here is a FlashInfer PR (merged) that updates the tests to avoid this issue. ⏎  ⏎ https://github.com/flashinfer-ai/flashinfer/pull/1643 ⏎  ⏎ ### Detailed summary ⏎  ⏎ Flashinfer's wrapper-based kernels (both prefill and dec …[truncated]

### L3-bf68fd76a9  (L3, 2025-09-24, sha bf68fd76a91b, PR #25518)
TITLE: [Compile] Fix AMD Compile Error (#25518)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.custom_paged
FILES: csrc/rocm/attention.cu (+6/-0); csrc/quantization/activation_kernels.cu (+6/-1)
LABELS: rocm, ready
BODY: ## Purpose ⏎  ⏎ Fix AMD Compile Error in MI300 ⏎  ⏎ Originally ⏎  ⏎ ```bash ⏎ /mnt/nvme5n1p1/wentao/vllm/cmake-build-release/csrc/rocm/attention.hip:131:33: error: use of undeclared identifier '__hip_fp8_e4m3' ⏎   131 |   if constexpr (std::is_same<T, __hip_fp8_e4m3>::value) { ⏎       |                                 ^ ⏎ /mnt/nvme5n1p1/wentao/vllm/cmake-build-release/csrc/rocm/attention.hip:134:40: error: use of undeclared identifier '__hip_fp8_e5m2' ⏎   134 |   } el …[truncated]

### L3-2e19a848d4  (L3, 2025-09-24, sha 2e19a848d42d, PR #25543)
TITLE: [V0 Deprecation] Remove max_seq_len_to_capture (#25543)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/utils.py (+2/-2); tests/tpu/lora/test_lora.py (+0/-1); vllm/config/model.py (+0/-18); vllm/config/speculative.py (+0/-2); vllm/engine/arg_utils.py (+0/-4); vllm/entrypoints/llm.py (+0/-7); vllm/model_executor/models/config.py (+0/-14)
LABELS: frontend, tpu, ready
BODY: 

### L3-2338daffd3  (L3, 2025-09-24, sha 2338daffd3ec, PR #25490)
TITLE: [BugFix] Potential Fix for FA3 full-cudagraph IMA  (#25490)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+11/-11)
LABELS: ready, v1
BODY: @WoosukKwon reported an IMA with FA3 full-CG that was fixed by doing https://github.com/vllm-project/vllm/compare/woosuk/fa3-ima?expand=1 ⏎  ⏎ the theory here is that `get_scheduler_metadata` was being called with a different `max_num_splits` than what was being passed to `FlashAttentionMetadata` ⏎  ⏎ this is an alternative solution that doesn't lose the logic to use `max_num_splits=0` (i.e. use the heuristic) for batches larger then `max_cudagraph_size` …[truncated]

### L3-302eb941f3  (L3, 2025-09-24, sha 302eb941f360, PR #25415)
TITLE: [ROCm][Build][Bugfix] Fix ROCm base docker whls installation order (#25415)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm_base (+12/-12)
LABELS: rocm, ready, ci/build
BODY: Make all whls install at the same command to fix the missing dependency issue when triton_kernels would pull CUDA dependencies from pypi ⏎ Split FA into its own build step to run in parallel

### L3-e6750d0b18  (L3, 2025-09-24, sha e6750d0b18e0, PR #25541)
TITLE: [V0 Deprecation] Remove unused classes in attention (#25541)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+3/-142); vllm/attention/backends/utils.py (+1/-544); vllm/v1/attention/backends/cpu_attn.py (+0/-15); vllm/v1/attention/backends/pallas.py (+0/-5); vllm/attention/__init__.py (+1/-5); vllm/v1/spec_decode/eagle.py (+6/-5)
LABELS: tpu, speculative-decoding, ready, v1
BODY: 

### L3-54e42b72db  (L3, 2025-09-24, sha 54e42b72dbf7, PR #21003)
TITLE: Support mnnvl all2allv from Flashinfer (#21003)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/envs.py (+5/-2); vllm/utils/flashinfer.py (+30/-0); tests/kernels/moe/modular_kernel_tools/mk_objects.py (+3/-2); vllm/distributed/device_communicators/all2all.py (+110/-15); vllm/distributed/device_communicators/cuda_communicator.py (+5/-0); vllm/distributed/device_communicators/mnnvl_compat.py (+28/-0); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py (+3/-4); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py (+220/-13); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+4/-2); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+2/-2)
LABELS: ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ Needs https://github.com/flashinfer-ai/flashinfer/pull/1245 ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ VLLM_ALL2ALL_BACKEND="flashinfer_all2allv" \ ⏎ CUDA_VISIBLE_DEVICES=0,1,2,3 \ ⏎ VLLM_USE_FLASHINFER_MOE_FP4=1 \ ⏎ VLLM_FLASHINFER_MOE_BACKEND="throughput" \ ⏎   /home/shuw/.local/bin/vllm serve nvidia/DeepSeek-R1-FP4 \ ⏎     --quantization="modelopt_fp4" \ ⏎     --trust-remote-code \ ⏎     --max-model-len=20 …[truncated]

### L3-05c19485a5  (L3, 2025-09-24, sha 05c19485a529, PR #25132)
TITLE: [Kernel] Support DCP for Triton backend  (#25132)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.triton.decode_attention
FILES: vllm/attention/ops/triton_decode_attention.py (+17/-2); vllm/v1/attention/backends/mla/triton_mla.py (+7/-5); tests/kernels/attention/test_triton_decode_attention.py (+5/-0); vllm/model_executor/models/deepseek_v2.py (+1/-1)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1
BODY: ## Purpose ⏎ As a follow up for https://github.com/vllm-project/vllm/pull/23734, this PR made some changes to support triton backend for DCP.  ⏎ Specifically, 1) return the LSE from triton kernel 2) fix a bug in deepseekV2 which could potentially modify the `residual` variable. ⏎  ⏎ ## Test Plan ⏎ export CUDA_VISIBLE_DEVICES=4,5,6,7 ⏎ export VLLM_USE_V1=1 ⏎ export VLLM_ATTENTION_BACKEND=TRITON_MLA ⏎ export VLLM_LOG_LEVEL=DEBUG ⏎ pytest tests/distributed/test_conte …[truncated]

### L3-6160ba4151  (L3, 2025-09-24, sha 6160ba415108, PR #25503)
TITLE: feat: BF16 FlashInfer Fused Cutlass MOE for Hopper and Blackwell Expert Parallel (#25503)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+6/-0); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py (+60/-5); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py (+2/-1); vllm/model_executor/layers/fused_moe/layer.py (+51/-0); vllm/model_executor/layers/fused_moe/modular_kernel.py (+2/-0)
LABELS: performance, ready
BODY: ## Purpose ⏎  ⏎ Enable BF16 FlashInfer Cutlass Fused MOE for Hopper and Blackwell, specifically testing expert parallel Qwen3-Next ⏎  ⏎ ## Test Plan ⏎  ⏎ ### Accuracy ⏎ ```bash ⏎ VLLM_USE_FLASHINFER_MOE_FP16=1 lm_eval --model vllm --model_args pretrained=Qwen3-Next-80B-A3B-Instruct,tensor_parallel_size=2,enable_expert_parallel=true,gpu_memory_utilization=0.80 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto ⏎ ``` ⏎  ⏎ ### Performance ⏎  ⏎ Triton and Fla …[truncated]

### L3-8c853050e7  (L3, 2025-09-24, sha 8c853050e7da, PR #25580)
TITLE: [Docs] Enable `fail_on_warning` for the docs build in CI (#25580)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/common.py (+19/-17); .readthedocs.yaml (+1/-0); docs/features/nixl_connector_usage.md (+4/-4); docs/mkdocs/hooks/generate_argparse.py (+3/-2); docs/models/generative_models.md (+1/-1); docs/models/supported_models.md (+1/-1); docs/usage/README.md (+1/-1); examples/online_serving/dashboards/grafana/README.md (+2/-2); examples/online_serving/dashboards/perses/README.md (+2/-2); vllm/inputs/data.py (+2/-2); (+10 more)
LABELS: documentation, ready, v1, qwen
ISSUES: #25020 [Docs]: Fix errors and warnings about griffe spotted by `mkdocs build`
BODY: Enables `fail_on_warning` so that docs issues such as broken links can be caught in CI. ⏎  ⏎ Closes https://github.com/vllm-project/vllm/issues/25020

### L3-845adb3ec6  (L3, 2025-09-24, sha 845adb3ec6d7, PR #23991)
TITLE: [Model] Add LongCat-Flash  (#23991)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mla.py (+0/-1); csrc/moe/moe_align_sum_kernels.cu (+8/-2); docs/models/supported_models.md (+1/-0); tests/kernels/moe/test_flashinfer.py (+2/-2); tests/models/registry.py (+6/-0); tests/models/utils.py (+9/-3); tests/test_routing_simulator.py (+1/-1); vllm/config/model.py (+5/-1); vllm/config/speculative.py (+19/-2); vllm/model_executor/layers/fused_moe/fused_moe.py (+89/-0); (+21 more)
LABELS: documentation, new-model, speculative-decoding, ready, v1, deepseek
BODY: This PR implements support for the newly released LongCat-Flash model by Meituan. ⏎ The core implementation includes: ⏎ - Model architecture in longcat_flash.py ⏎ - MTP model architecture in longcat_flash_mtp.py

### L3-69a8c8e99a  (L3, 2025-09-25, sha 69a8c8e99ab9, PR #24914)
TITLE: [torch.compile] Make Query Quantization Fusable (#24914)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+8/-0); vllm/attention/layer.py (+22/-1); vllm/v1/attention/backends/flash_attn.py (+2/-7)
LABELS: ready, torch.compile, v1
BODY: ## Purpose ⏎ When running attention in FP8, quantizing the queries causes overhead, since this is done via a custom operation. And because it is a custom op it cannot be fused by torch compile. Profiling shows that this is a (context-lenght-independent) overhead.  ⏎  ⏎ This PR moves the query quantization out of the attention backend into `attention/layer.py` and uses a simple torch implementation. Then torch compile is able to fuse it and reduce the q …[truncated]

### L3-d2af67441d  (L3, 2025-09-25, sha d2af67441ddf, PR #25643)
TITLE: [XPU][Triton]add xpu config in triton_reshape_and_cache_flash (#25643)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/triton_reshape_and_cache_flash.py (+1/-1)
LABELS: ready
BODY: ## Purpose ⏎ #24503 add `triton_reshape_and_cache_flash` kernel, this break XPU by accident. this PR add xpu config to avoid go to cuda path and throw error. ⏎  ⏎ ## Test Plan ⏎ CI ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-eb32335e35  (L3, 2025-09-25, sha eb32335e355f, PR #25652)
TITLE: [CPU] update torch 2.8 and fix missing fields in TorchSDPAMetadata (#25652)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: vllm/v1/attention/backends/cpu_attn.py (+13/-1); .buildkite/scripts/hardware_ci/run-cpu-test.sh (+2/-5); docker/Dockerfile.cpu (+0/-3); requirements/cpu-build.txt (+1/-3); requirements/cpu.txt (+2/-2); vllm/triton_utils/importing.py (+1/-0); vllm/v1/sample/ops/topk_topp_sampler.py (+41/-0); vllm/v1/worker/cpu_worker.py (+0/-39)
LABELS: ready, ci/build, v1
BODY: ## Purpose ⏎  ⏎ - Use torch compile for torch native sample on CPU to avoid perf regression started from torch 2.6 ⏎ - Update torch 2.8 to resolve some failed torch version checks ⏎ - Fix missing fields in ```TorchSDPAMetadata``` ⏎ - Fix ```tl.tensor``` import on non-triton platforms ⏎  ⏎ ## Test Plan ⏎  ⏎ CI tests ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-e71b8e210d  (L3, 2025-09-25, sha e71b8e210db4, PR #24986)
TITLE: [Spec Decode] Add Batch Parallel Ngram. Upto 8x lower overhead. (#24986)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/benchmark_ngram_proposer.py (+104/-3); tests/v1/spec_decode/test_ngram.py (+114/-28); vllm/v1/sample/rejection_sampler.py (+1/-1); vllm/v1/spec_decode/ngram_proposer.py (+157/-38); vllm/v1/worker/gpu_model_runner.py (+5/-37)
LABELS: performance, speculative-decoding, ready, v1
BODY: ## Purpose ⏎ Ngram overhead increases with batch size and seq len. This is CPU overhead and adds to the critical path of inference and becomes more imp if seq len or bs increases. This PR parallelizes the CPU compute along batch dimension. The overhead reduces upto 8x. The threads are capped at 8 since there are other processes like frontend (tokenizaton and req handling) and structured output that need multithreading. ⏎  ⏎ This PR also cleans the prop …[truncated]

### L3-71b25b0d48  (L3, 2025-09-25, sha 71b25b0d482e, PR #25675)
TITLE: [V0 deprecation] Clean up V0 fallback in compilation config (#25675)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/__init__.py (+20/-70); vllm/config/compilation.py (+2/-3)
LABELS: ready
BODY: ## Purpose ⏎ - Clean up `torch.compile` related V0 fallback code ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-3468f17ebe  (L3, 2025-09-25, sha 3468f17ebe4d, PR #25489)
TITLE: [V0 deprecation] Remove _VLLM_V1 suffixes from attention backend names (#25489)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.xformers.v1_backend, L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.rocm_aiter, L3.dispatch.selector, L3.platform.cuda_selection, L3.platform.rocm_selection, L3.tree_attention
FILES: vllm/attention/layer.py (+6/-15); vllm/attention/selector.py (+8/-0); vllm/engine/arg_utils.py (+4/-8); vllm/platforms/cuda.py (+10/-11); vllm/platforms/interface.py (+2/-10); vllm/platforms/rocm.py (+2/-3); vllm/platforms/tpu.py (+1/-2); vllm/platforms/xpu.py (+6/-6); vllm/v1/attention/backends/cpu_attn.py (+1/-1); vllm/v1/attention/backends/flash_attn.py (+1/-1); (+32 more)
LABELS: rocm, tpu, speculative-decoding, ready, ci/build, v1, kv-connector
BODY: ## Purpose ⏎ Remove `_VLLM_V1` suffixes as these are no longer needed. If a user includes that suffix in their `VLLM_ATTENTION_BACKEND` environment variable, it is stripped and a warning is printed. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-0fa673af4c  (L3, 2025-09-25, sha 0fa673af4c2a, PR #25686)
TITLE: [V0 deprecation] Clean up LoRA  (#25686)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/lora/punica_wrapper/punica_gpu.py (+1/-8)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-081b5594a2  (L3, 2025-09-25, sha 081b5594a2b1, PR #25711)
TITLE: Fix routing_bias dtype  (#25711)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/modelopt.py (+4/-1)
LABELS: ready
BODY: Fix the routing_bias dtype to bf16 for flashinfer.fused_moe.trtllm_fp4_block_scale_moe ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-d48f4d6daf  (L3, 2025-09-26, sha d48f4d6daf7c, PR #25739)
TITLE: perf: Avoid copying inputs_embeds tensors to GPU unless prompt_embeds is enabled (#25739)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_model_runner.py (+15/-11)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ @njhill pointed out that #24278 introduced a performance regression in the case when prompt embeds is disabled, where inputs_embeds tensors are copied to GPU, even though those tensors are either empty or filled with garbage when `--enable-prompt-embeds` is not on. We can guard those copies with `self.enable_prompt_embeds` to avoid these copies unless prompt embeds is enabled. ⏎  ⏎ ## Test Plan ⏎  ⏎ No new tests are needed. I have some local  …[truncated]

### L3-dd70437a4f  (L3, 2025-09-26, sha dd70437a4f36, PR #25555)
TITLE: Remove cuda hard-code in compute_causal_conv1d_metadata (#25555)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+3/-2)
LABELS: ready, v1
BODY: ## Purpose ⏎ Remove cuda hard-code in compute_causal_conv1d_metadata ⏎  ⏎ ## Test Plan ⏎ Test through the exsiting tests ⏎  ⏎ --- ⏎ [details omitted]

### L3-cf89202855  (L3, 2025-09-26, sha cf89202855a4, PR #25730)
TITLE: [CI] Fix FlashInfer AOT in release docker image (#25730)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+3/-0); .buildkite/release-pipeline.yaml (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ FlashInfer has shifted to using FLASHINFER_CUDA_ARCH_LIST in recent releases https://docs.flashinfer.ai/installation.html#python-package ⏎  ⏎ This has resulted in the 0.10.2 release not having prebuilt flashinfer binaries ⏎  ⏎ See buildkite log https://buildkite.com/organizations/vllm/pipelines/release/builds/8688/jobs/0199833c-09e7-46fe-a38a-6dd9493547e5/log#117-11445 ⏎ ``` ⏎ [2025-09-25T23:59:38Z] #33 8.381 RuntimeError: Please explicitly set e …[truncated]

### L3-f075693da7  (L3, 2025-09-26, sha f075693da767, PR #23046)
TITLE: [V1] address post issues related to #20059 (part 1) (#23046)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/piecewise/test_full_cudagraph.py (+1/-85); tests/compile/test_config.py (+65/-2); tests/v1/attention/utils.py (+86/-1); tests/v1/cudagraph/test_cudagraph_dispatch.py (+26/-35); tests/v1/cudagraph/test_cudagraph_mode.py (+9/-75); vllm/compilation/backends.py (+3/-3); vllm/compilation/decorators.py (+2/-2); vllm/compilation/piecewise_backend.py (+0/-0); vllm/config/__init__.py (+25/-12); vllm/config/compilation.py (+76/-37); (+3 more)
LABELS: ready, v1
BODY: ## Purpose ⏎ This PR addressed several issues after #20059 was landed. ⏎  ⏎ CC list: @ProExpertProg @LucasWilkinson  ⏎  ⏎ More issues affecting spec-decode (part 2), please see #23679. ⏎  ⏎ ## Test Plan ⏎ Simply test if the dispatcher can dispatch to NONE or PIECEWISE runtime mode for cascade attention. ⏎ No benchmark or correctness test is provided.  ⏎  ⏎ ## Test Result ⏎  ⏎ It passed. ⏎  ⏎ ## (Optional) Documentation Update ⏎  ⏎ --- ⏎ [details omitted]

### L3-dc48ba0c75  (L3, 2025-09-26, sha dc48ba0c750e, PR #25603)
TITLE: Kernel-override Determinism [1/n] (#25603)
SOURCES: path_core
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+7/-0); csrc/core/batch_invariant.hpp (+16/-0); csrc/layernorm_kernels.cu (+6/-2); csrc/layernorm_quant_kernels.cu (+4/-1); csrc/moe/topk_softmax_kernels.cu (+3/-1); tests/v1/generation/test_batch_invariance.py (+290/-0); vllm/model_executor/layers/batch_invariant.py (+561/-0); vllm/v1/worker/gpu_model_runner.py (+3/-0)
LABELS: ready, v1
BODY: Adds first plumbing and initial working test for issue #25404. ⏎  ⏎ ## Purpose ⏎  ⏎ Landing this will help make additional determinism work highly parallelizable.  This PR adds three hooks for determinism that can be followed in future changes. ⏎  ⏎ 1. C++ hook by using `bool deterministic_launch = vllm_kernel_override_determinism_all();` anywhere in csrc, requires `#include`ing core/determinism.hpp ⏎     If a specific env flag is needed, use `VLLM_REGISTER_KE …[truncated]

### L3-92da847cf5  (L3, 2025-09-26, sha 92da847cf5f4, PR #25782)
TITLE: Add flashinfer-build.sh and register precompiled cu128 wheel in Dockerfile (#25782)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+20/-10); tools/flashinfer-build.sh (+63/-0)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-a3ae45a38c  (L3, 2025-09-29, sha a3ae45a38cb0, PR #25825)
TITLE: [Misc] fix tests failure by using current_platform (#25825)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/triton_reshape_and_cache_flash.py (+1/-1)
LABELS: ready
BODY: Fix the local unit test error by using standard platform detection method in vllm ⏎  ⏎ ## Purpose ⏎ Fix the failing UTs.  Error msg: ⏎ ``` ⏎         TILE_SIZE = min(2048, triton.next_power_of_2(n)) ⏎ >       if torch.version.hip or torch.version.xpu: ⏎ E       AttributeError: module 'torch.version' has no attribute 'xpu' ⏎ ``` ⏎  ⏎ ## Test Plan ⏎ Re-run UT ⏎  ⏎ ## Test Result ⏎  ⏎ tests/kernels/attention/test_cache.py::test_reshape_and_cache_flash[triton-NHD-auto-cuda:1-0-dty …[truncated]

### L3-65ecb4f134  (L3, 2025-09-29, sha 65ecb4f134a2, PR #25851)
TITLE: [Bugfix] Fallback ViT attn backend to SDPA for blackwell (#25851)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+6/-0); vllm/model_executor/models/qwen3_vl.py (+1/-9)
LABELS: ready, qwen
BODY: ## Purpose ⏎ https://github.com/vllm-project/vllm/pull/25788 Fixed the issue for Qwen3-VL - while we don't know if xformers is going to work with other head sizes, it's not officially supported yet according to https://github.com/facebookresearch/xformers/issues/1317#issuecomment-3199392579. Therefore it's probably safer for us to force ViT backend to SDPA for all models on blackwell for now. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-c42ff4f4fd  (L3, 2025-09-29, sha c42ff4f4fdc4, PR #25513)
TITLE: [BugFix][torch.compile] KV scale calculation issues with FP8 quantization (#25513)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+40/-3); tests/compile/test_full_graph.py (+15/-0); vllm/v1/worker/gpu_model_runner.py (+9/-0)
LABELS: ready, v1
ISSUES: #21640 [Bug]: calculate_kv_scales leads to dynamo compilation issue; enforce_eager=True leads to another issue
BODY: ## Purpose ⏎ Fix KV scale calculation incompatibility with torch.compile and enforce_eager mode (#21640) ⏎  ⏎ The conditional check `if attn_metadata.enable_kv_scales_calculation`: at line 274 in `vllm/attention/layer.py` caused two failures: ⏎  ⏎ - _Dynamo compilation error (enforce_eager=False): Data-dependent branching that torch.compile cannot trace through ⏎ - _AttributeError (enforce_eager=True): attn_metadata could be None, causing crashes ⏎  ⏎ This PR mo …[truncated]

### L3-61aedb5ffe  (L3, 2025-09-29, sha 61aedb5ffe05, PR #25271)
TITLE: Move`VllmConfig` from `config/__init__.py` to `config/vllm.py` (#25271)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+1/-2); vllm/attention/layers/chunked_local_attention.py (+2/-1); vllm/config/__init__.py (+79/-826); vllm/config/utils.py (+36/-6); vllm/config/vllm.py (+789/-0); vllm/model_executor/layers/mamba/linear_attn.py (+1/-2); vllm/model_executor/layers/quantization/auto_round.py (+2/-3); vllm/model_executor/layers/quantization/bitblas.py (+2/-3); vllm/model_executor/layers/quantization/bitsandbytes.py (+2/-3); vllm/model_executor/layers/quantization/deepspeedfp.py (+2/-3); (+26 more)
LABELS: tpu, speculative-decoding, ready, llama
ISSUES: #18953 [RFC]: vLLM configuration  refactoring and modularization
BODY: Closes #18953 ⏎  ⏎ Notable changes: ⏎  ⏎ - Several places were importing `QuantizationConfg` from `config.__init__` which is not where it is defined. Even worse, at runtime, `QuantizationConfig` from `config.__init__.py` is `Any`. ⏎ - I've gone though every import of `QuantizationConfig` making sure it's imported from the right place. ⏎ - This required a lot of type checking changes to `gptq_utils`.

### L3-ef283548f7  (L3, 2025-09-30, sha ef283548f751, PR #25895)
TITLE: [Bugfix] Fix accuracy issue of TRTLLM FP8 MOE and improve logging (#25895)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/fp8.py (+23/-16); vllm/utils/deep_gemm.py (+7/-2)
LABELS: bug, ready
ISSUES: #25189 [Bug]: B200 FlashInfer FP8 MoE low-latency - incorrect results
BODY: ## Purpose ⏎ Fixes #25189: accuracy issues for TRTLLM DSR1 Latency kernels  ⏎ Bugs introduced in #23991 and #23640  ⏎  ⏎ Also fixes incorrect logging prints for which kernels are used. When Flashinfer MOE is enabled, Deep Gemm is automatically disabled for SM100. ⏎  ⏎ ``` ⏎ (EngineCore_DP0 pid=101213) (Worker_TP3 pid=101225) INFO 09-29 10:14:46 [fp8.py:454] Detected Blackwell GPUs, using FlashInfer TensorRT-LLM kernels for FP8 MOE. ⏎ (EngineCore_DP0 pid=101213)  …[truncated]

### L3-80608ba5af  (L3, 2025-09-30, sha 80608ba5afe6, PR #25902)
TITLE: [NIXL] Add support for MLA caches with different latent dim (#25902)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_nixl_connector.py (+7/-6); vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py (+59/-36); vllm/v1/core/kv_cache_utils.py (+5/-2)
LABELS: ready, v1, kv-connector
BODY: This PR enables the transfer of KV caches with different shapes (last dim only here)  for MLA models, and in particular for the new DeepseekV3-2, allowing to send/rcv its Indexer cache as well in a disaggregated setup. ⏎  ⏎ It does so by extending `block_len` (N*H*D in a regular KV cache) to `block_len_per_layer`, allowing each layer to define its own "stride".     ⏎ This approach has the potential of being re-used for dense models too, although for no …[truncated]

### L3-bb6d43047e  (L3, 2025-09-30, sha bb6d43047e24, PR #25816)
TITLE: [Fix] Improve CPU backend compatibility for RISC-V (#25816)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/engine/arg_utils.py (+5/-4)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1
ISSUES: #25737 [Bug]: RISC-V and non-Intel CPU architectures fail due to widespread unconditional IPEX dependencies
BODY: ## Purpose ⏎ Fixes #25737 ⏎ This PR aims to fix crashes and improve the compatibility of vLLM's CPU backend when running on the RISC-V architecture. It addresses two specific issues: ⏎  ⏎ 1.  **IPEX Dependency Crash:** The `chunked_prefill` feature in the CPU attention backend unconditionally imports `intel_extension_for_pytorch`, causing a `ModuleNotFoundError` on non-x86 platforms. This PR fixes this by guarding the import with the existing `_use_ipex` …[truncated]

### L3-fa7e254a7f  (L3, 2025-09-30, sha fa7e254a7f3e, PR #25896)
TITLE: [New Model] DeepSeek-V3.2 (Rebased to Main) (#25896)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.paged.python_wrapper, L3.xformers.v1_backend, L3.flash_attn.v1_backend, L3.flash_attn.upstream_pip, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.aiter_fa, L3.mla.common_v1, L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.mla.flashmla_build, L3.mla.flashmla_sparse, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.flex_attention, L3.tree_attention
FILES: cmake/external_projects/flashmla.cmake (+76/-11); csrc/cache_kernels.cu (+252/-6); setup.py (+4/-0); vllm/attention/backends/abstract.py (+1/-0); vllm/attention/layer.py (+4/-1); vllm/attention/ops/common.py (+205/-0); vllm/attention/ops/flashmla.py (+117/-48); vllm/attention/ops/paged_attn.py (+1/-0); vllm/attention/selector.py (+4/-1); vllm/model_executor/layers/mla.py (+16/-0); (+61 more)
LABELS: documentation, new-model, rocm, tpu, speculative-decoding, ready, ci/build, v1, deepseek
BODY: Rebased dsv32, based on #25869 ⏎  ⏎ Run command ⏎ ``` ⏎ vllm serve deepseek-ai/DeepSeek-V3.2-Exp  --max_model_len=20000 --gpu_memory_utilization=0.9 -tp 8 --max_num_seqs=256 ⏎ ``` ⏎  ⏎ gsm8k ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9568|±  |0.0056| ⏎ |     |       |strict-match    |     5|exact_ma …[truncated]

### L3-e952eee698  (L3, 2025-09-30, sha e952eee698fb, PR #25996)
TITLE: [Bugfix] Fix `__syncwarp` on ROCM (#25996)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+5/-1)
LABELS: rocm, ready, deepseek
BODY: ## Purpose ⏎ We are seeing failure on AMD due to __syncwarp is exclusively on CUDA but not on HIP ⏎ ``` ⏎ __vllm_cpp_lib_hipify_gen__/out/csrc/cache_kernels.hip:541:3: error: use of undeclared identifier '__syncwarp'; did you mean '__sync_swap'? ⏎   541 |   __syncwarp(); ⏎       |   ^~~~~~~~~~ ⏎       |   __sync_swap ⏎ ``` ⏎ ## Test Plan ⏎ CI

### L3-a2e6fa7e03  (L3, 2025-10-01, sha a2e6fa7e035f, PR #25956)
TITLE: [bugfix][deepseek] fix flashmla kernel selection (#25956)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_v0_adapter
FILES: vllm/attention/ops/flashmla.py (+1/-1)
LABELS: ready, deepseek
BODY: ## Purpose ⏎  ⏎ It seems we always pass in `descale_q` as tensors in flashmla backend, so it selects the wrong kernel implementation. ⏎  ⏎ Fixes https://github.com/vllm-project/vllm/pull/25896#issuecomment-3350484670 and potentially https://github.com/vllm-project/vllm/pull/25896#issuecomment-3351342583  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-001e50c92c  (L3, 2025-10-01, sha 001e50c92c15, PR #25982)
TITLE: [Model] MTP fallback to eager for DeepSeek v32 (#25982)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+1/-1); tests/v1/spec_decode/test_eagle.py (+10/-1); tests/v1/spec_decode/test_mtp.py (+7/-1); vllm/config/speculative.py (+7/-1); vllm/v1/spec_decode/eagle.py (+7/-1)
LABELS: speculative-decoding, ready, v1, deepseek
BODY: ## Purpose ⏎  ⏎ 1. Cudagraph + MTP V32 is still under progress, enable eager  for now on MTP part by default which verified acceptance and eval and piecewise used with MTP enabled ⏎ 2. Fix Eagle tests as well MTP tests (@njhill ) ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ VLLM_SKIP_DEEP_GEMM_WARMUP=1 vllm serve "deepseek-ai/DeepSeek-V3.2-Exp" --max_model_len=20000 --gpu_memory_utilization=0.9 --tensor_parallel_size 8 --max_num_seqs=256 --speculative_config '{"num_speculative_t …[truncated]

### L3-2518230d3e  (L3, 2025-10-01, sha 2518230d3eda, PR #25829)
TITLE: [MISC] Fix misleading batch_size_capture_list when cuda_graph_sizes < 4 (#25829)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+6/-3)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Previously, the logic for generating `batch_size_capture_list` always included [1, 2, 4] by default. Refer to https://github.com/vllm-project/vllm/blob/f4e4088c99020c9711e824096f013f89da54bb36/vllm/config/__init__.py#L619-L622 ⏎  ⏎ This was misleading when cuda_graph_sizes < 4 because the list contained batch sizes that exceeded the actual maximum. ⏎  ⏎ ## How to fix in this patch ⏎ Filtered out [1, 2, 4] values that are larger than cuda_graph_ …[truncated]

### L3-1726e93ef1  (L3, 2025-10-01, sha 1726e93ef1c8, PR #26026)
TITLE: [BugFix][DP/EP] Fix CUTLASS MLA hang under load (#26026)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: csrc/attention/mla/cutlass_sm100_mla/kernel/sm100_fmha_mla_tma_warpspecialized.hpp (+32/-32)
LABELS: bug, ready, deepseek
BODY: The early return in `compute(` calls arrive: ⏎  ⏎ ``` ⏎       cutlass::arch::NamedBarrier( ⏎           (kNumComputeWarps + kNumLoadWarps) * NumThreadsPerWarp, ⏎           kNamedBarrierEpilogue ⏎       ).arrive(); ⏎ ``` ⏎  ⏎ but didn't have any barrier before looping around and calling it again causing a deadlock when the load warps waits on: ⏎  ⏎ ``` ⏎ cutlass::arch::NamedBarrier((kNumComputeWarps + kNumLoadWarps) * NumThreadsPerWarp, kNamedBarrierEpilogue).arrive_and_w …[truncated]

### L3-4134312b35  (L3, 2025-10-01, sha 4134312b3546, PR #26034)
TITLE: [BugFix] ChunkedLocalAttention is currently not CG compatible (#26034)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layers/chunked_local_attention.py (+5/-3)
LABELS: bug, ready, llama
BODY: potential fix for: https://github.com/vllm-project/vllm/issues/25960

### L3-0b018d8baf  (L3, 2025-10-01, sha 0b018d8baf1b, PR #26029)
TITLE: [ROCm][Bugfix] Add missing parameter to ROCm backend (#26029)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn
FILES: vllm/v1/attention/backends/rocm_attn.py (+1/-0)
LABELS: rocm, ready, v1
BODY: Follow up to #25896 that skipped ROCm attention backend when adding the new parameter to get_kv_cache_shape

### L3-aac622e0cd  (L3, 2025-10-01, sha aac622e0cd00, PR #25908)
TITLE: [ROCm][Build] Add support for AMD Ryzen AI MAX / AI 300 Series (#25908)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake, L3.rocm.custom_paged
FILES: csrc/rocm/attention.cu (+2/-1); CMakeLists.txt (+1/-1); docker/Dockerfile.rocm_base (+2/-2)
LABELS: rocm, ready, ci/build
ISSUES: #25634 [Feature]: When vllm plan to support AMD APU - AMD Ryzen AI Max 395 - gfx1151 | gfx 1150
BODY: ## Purpose ⏎ Added support for AMD Ryzen AI MAX / AI 300 Series ⏎ Updated 'CMakeLists.txt' and 'Dockerfile.rocm_base' to include gfx1150 and gfx1151 ⏎  ⏎ CLOSE #25634 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-5e4a8223c6  (L3, 2025-10-02, sha 5e4a8223c644, PR #24642)
TITLE: [Qwen][ROCm] Flash Attention Rotary Embeddings (#24642)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/rotary_embedding/common.py (+23/-0); vllm/model_executor/models/qwen2_vl.py (+5/-5)
LABELS: rocm, ready, qwen
BODY: ## Purpose ⏎ Qwen VL models previously relied on a basic PyTorch implementation for applying rotary positional embeddings on ROCm architectures. This PR adds a ROCm specialisation to use flash_attn module's Triton-based operation which makes it much faster as seen in the benchmarking results below: ⏎  ⏎ benchmark_serving script ⏎  ⏎ ```bash ⏎ python3 vllm/benchmarks/benchmark_serving.py  \ ⏎ --backend openai-chat   \ ⏎ --model Qwen/Qwen2-VL-7B-Instruct     \ ⏎ --e …[truncated]

### L3-1e50f1be70  (L3, 2025-10-02, sha 1e50f1be7058, PR #25999)
TITLE: [Deepseek v3.2] Support indexer prefill chunking (#25999)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+90/-41); tests/v1/attention/test_sparse_mla_backends.py (+22/-0); vllm/model_executor/models/deepseek_v2.py (+37/-38)
LABELS: ready, v1, deepseek
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ Split the prefill to multiple steps, with each step contains a subset of prefill requests. With this approach, we can avoid the large output caused by gather kv cache. ⏎  ⏎ ## Test Plan ⏎ 20 shot gsm 8k ⏎ ``` ⏎ vllm serve deepseek-ai/DeepSeek-V3.2-Exp -tp 8 --max_model_len 32768 ⏎ ``` ⏎ ``` ⏎ lm-eval --model local-completions --tasks gsm8k   --model_args model=deepseek-ai/DeepSeek-V3.2-Exp,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=1 …[truncated]

### L3-418d111f8c  (L3, 2025-10-02, sha 418d111f8c8e, PR #25537)
TITLE: [FA/Chore] Bump vllm-flash-attention (#25537)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build
BODY: Bump FA to pickup ⏎  ⏎ https://github.com/vllm-project/flash-attention/pull/94 ⏎ https://github.com/vllm-project/flash-attention/pull/91 ⏎ https://github.com/vllm-project/flash-attention/pull/87

### L3-f1fc2107a3  (L3, 2025-10-02, sha f1fc2107a314, PR #26130)
TITLE: [Bugfix] Disable cascade attention with FlashInfer (#26130)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+3/-2)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Using cascade attention with FlashInfer on Blackwell seems to break when prefix caching is hit. ⏎  ⏎ This eval hangs on the first batch of requests ⏎ ``` ⏎ vllm serve Qwen/Qwen3-0.6B ⏎ python tests/evals/gsm8k/gsm8k_eval.py ⏎ (APIServer pid=2318591) INFO 10-02 12:38:13 [loggers.py:127] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 100 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.5%, Prefix cach …[truncated]

### L3-decf7f794b  (L3, 2025-10-02, sha decf7f794bff, PR #26063)
TITLE: [BugFix] Fix FI accuracy issue when used for MLA prefill (#26063)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+9/-2)
LABELS: bug, ready, v1
BODY: # PR ⏎  ⏎ ``` ⏎ VLLM_LOGGING_LEVEL=DEBUG vllm serve deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct ⏎ ... ⏎ (EngineCore_DP0 pid=3662669) DEBUG 10-01 17:41:11 [v1/attention/.../mla/common.py:980] Using FlashInfer prefill for MLA ⏎ ... ⏎ ======================================================================== ⏎ (vllm) lwilkinson@dgxB200-09:~/code/vllm$ python tests/evals/gsm8k/gsm8k_eval.py ⏎ Running GSM8K evaluation: 1319 questions, 5-shot ⏎ Evaluating: 100%|████████████ …[truncated]

### L3-2aaa423842  (L3, 2025-10-02, sha 2aaa42384280, PR #25893)
TITLE: [Attention] Move Backend enum into registry (#25893)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.dispatch.selector, L3.dispatch.registry, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/attention/backends/registry.py (+27/-0); vllm/attention/layer.py (+2/-1); vllm/attention/selector.py (+2/-1); vllm/envs.py (+3/-2); vllm/platforms/__init__.py (+0/-1); vllm/platforms/cpu.py (+5/-2); vllm/platforms/cuda.py (+7/-2); vllm/platforms/interface.py (+5/-26); vllm/platforms/rocm.py (+7/-2); vllm/platforms/tpu.py (+5/-2); (+21 more)
LABELS: rocm, tpu, speculative-decoding, ready, v1, qwen, kv-connector
BODY: ## Purpose ⏎ Creates `vllm/attention/backends/registry.py`, containing the `_Backend` enum. This PR follows #25489 and is a component of the larger attention backend refactor (now being split up), #24794. This file will contain more utility functions in the future. ⏎  ⏎ ## Test Plan ⏎ CI should be sufficient ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-47b9339546  (L3, 2025-10-02, sha 47b93395463d, PR #26132)
TITLE: [DeepSeek] Improve performance of DS MLA cache kernel (#26132)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+62/-68)
LABELS: ready, deepseek
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎ Improve the performance of the `concat_and_cache_ds_mla_kernel` by having each thread handle either 8 NoPE elements or 2 RoPE elements. Roughly 60% best-case speedup. ⏎  ⏎ ## Correctness ⏎ `pytest tests/kernels/attention/test_cache.py::test_concat_and_cache_ds_mla` (Passes) ⏎  ⏎ ## Performance ⏎  ⏎ ### Before ⏎  ⏎ ``` ⏎ concat_and_cache_ds_mla Kernel Benchmark ⏎ ================================================== ⏎ PyTorch version: 2.8.0+cu128 ⏎ CUDA version: 12 …[truncated]

### L3-9c5ee91b2a  (L3, 2025-10-02, sha 9c5ee91b2af8, PR #26104)
TITLE: [ROCm] [VL] [Bugfix] Fix vit flash attn dispatcher logic for ROCm (#26104)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/attention/layer.py (+49/-23); vllm/platforms/rocm.py (+0/-2); vllm/model_executor/models/dots_ocr.py (+19/-22); vllm/model_executor/models/ernie45_vl.py (+23/-26); vllm/model_executor/models/glm4_1v.py (+17/-14); vllm/model_executor/models/qwen2_5_vl.py (+17/-17); vllm/model_executor/models/qwen2_vl.py (+18/-22); vllm/model_executor/models/qwen3_vl.py (+3/-1); vllm/model_executor/models/siglip2navit.py (+8/-14)
LABELS: rocm, ready, qwen
BODY: ## Purpose ⏎  ⏎ The refactoring of code has causes the vit flash attn dispatcher logic to enter the wrong code path to import ⏎ `from vllm.vllm_flash_attn import flash_attn_varlen_func` on ROCm platform. ⏎  ⏎ Fix incorrect usage of `aiter.flash_attn_varlen_func` in `MultiHeadAttention` class introduced in https://github.com/vllm-project/vllm/pull/23978 ⏎  ⏎ ## Test Plan ⏎  ⏎ Evaluate accuracy of all of the models that uses this vit flash attn dispatcher logic on c …[truncated]

### L3-eb0fa43868  (L3, 2025-10-03, sha eb0fa43868cf, PR #25955)
TITLE: [Perf] Optimize `reshape_and_cache` CUDA Kernel (#25955)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+51/-45); benchmarks/kernels/benchmark_reshape_and_cache.py (+174/-0)
LABELS: performance, ready
ISSUES: #25705 [Feature]: [Perf] Optimize `reshape_and_cache` CUDA Kernel
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ FIX https://github.com/vllm-project/vllm/issues/25705 ⏎ Optimize `reshape_and_cache` CUDA Kernel. ⏎ Separate key/value loops - Allows specialized indexing for each ⏎  ⏎ ## Test ⏎ ``` ⏎ pytest -s tests/kernels/attention/test_cache.py::test_reshape_and_cache ⏎  ⏎ passed ⏎ ``` ⏎ gsm8k ⏎ ``` ⏎ vllm (pretrained=/data/datasets/models-hf/Qwen3-4B-Instruct-2507-FP8/,max_model_len=32768,enforce_eager=True,trust_remote_code=True), gen_kwargs: (None), limit: None, num_f …[truncated]

### L3-5f2cacdb1e  (L3, 2025-10-03, sha 5f2cacdb1e62, PR #25983)
TITLE: Quick fix for IMA with the Prefix Prefill kernel during graph capture (#25983)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn
FILES: vllm/v1/attention/backends/rocm_attn.py (+8/-0)
LABELS: ready, v1
BODY: ## Purpose ⏎ This PR is a band aid fix for the `context_attn_fwd` kernel inside of `prefix_prefill.py`. This kernel triggers an IMA since we added proper handling of query_start_locs in the _dummy_run in #24845. The fix is just to revert to the old behavior when we detect that we are capturing with this kernel. ⏎  ⏎ ## Test Plan ⏎ lm_eval ⏎ ## Test Result ⏎ `VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 vllm serve meta-llama/Llama-3.1-8B-Instruct --max-num-seqs 256 …[truncated]

### L3-c1ffcb55da  (L3, 2025-10-03, sha c1ffcb55da6a, PR #26044)
TITLE: [Refactor] Optimize FP8 MOE Backend Choice and Log (#26044)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/fp8.py (+71/-46)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ A follow up PR for https://github.com/vllm-project/vllm/pull/25709 ⏎  ⏎ Here we similarly use the structure from `/vllm/model_executor/layers/quantization/mxfp4.py`, choose the related backend and print clear log for user ⏎  ⏎ ## Test ⏎  ⏎ With DeepGEMM: ⏎  ⏎ `[fp8.py:114] Using DeepGEMM backend for FP8 MoE` ⏎  ⏎ Without DeepGEMM: ⏎  ⏎ `[fp8.py:122] Using Cutlass BlockScaled GroupedGemm backend for FP8 MoE` ⏎  ⏎ For Flashinfer:  ⏎  ⏎ `[fp8.py:98] Using FlashInfer FP …[truncated]

### L3-300a59c4c3  (L3, 2025-10-03, sha 300a59c4c313, PR #26174)
TITLE: Avoid division by zero in cache DS MLA kernel (#26174)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+2/-1)
LABELS: ready
BODY: ## Purpose ⏎ Division by zero fix got missed in #26132, this is a tiny PR to add it back. ⏎  ⏎ ## Test Plan ⏎ `pytest tests/kernels/attention/test_cache.py::test_concat_and_cache_ds_mla` ⏎  ⏎ ## Test Result ⏎ Passes ⏎  ⏎ --- ⏎ [details omitted]

### L3-cd9e5b8340  (L3, 2025-10-03, sha cd9e5b8340b4, PR #26148)
TITLE: Fix V1 engine serialization error with Ray distributed executor (#26148)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/executor/ray_utils.py (+6/-0)
LABELS: ready
BODY: ## Purpose ⏎ **Problem**: When using V1 engine with Ray distributed executor, inference crashes with `TypeError: cannot pickle Event object` during the first request after initialization. ⏎  ⏎ **Fixes** ⏎ - [Ray #57039](https://github.com/ray-project/ray/issues/57039) ⏎ - [vLLM #25861](https://github.com/vllm-project/vllm/issues/25861) ⏎  ⏎ **Root Cause:** `AsyncGPUModelRunnerOutput` contains `torch.cuda.Event` objects that cannot be serialized when Ray's comp …[truncated]

### L3-a26917332f  (L3, 2025-10-03, sha a26917332fab, PR #25968)
TITLE: [Quantization/NVFP4] Speed up TRTLLM NVFP4 MOE weight loading and fix K/V scale loading for MLA Attn (#25968)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/kv_cache.py (+4/-4); vllm/model_executor/layers/quantization/modelopt.py (+66/-50); vllm/model_executor/model_loader/weight_utils.py (+7/-1)
LABELS: ready
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ Speeds up weight loading for NVFP4 TRTLLM MoE kernels by 10x ⏎  ⏎ Before PR: ⏎ ``` ⏎ (Worker_TP2 pid=479) INFO 10-03 09:57:04 [default_loader.py:294] Loading weights took 44.17 seconds ⏎ (Worker_TP3 pid=480) WARNING 10-03 09:57:04 [weight_utils.py:1032] Found k_scale in the checkpoint (e.g. model.layers.5.self_attn.k_proj.k_scale), but not found the expected name in the model (e.g. model.layers.5.self_attn.attn.k_scale). k_scale is not loaded. ⏎ ( …[truncated]

### L3-2f7dbc9b42  (L3, 2025-10-03, sha 2f7dbc9b42c5, PR #25769)
TITLE: Add batch invariant kernel override for FlashInfer backend [2/n] (#25769)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts(reverted), body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+32/-5); tests/v1/generation/test_batch_invariance.py (+40/-23); vllm/model_executor/layers/batch_invariant.py (+12/-1)
LABELS: ready, v1
DEEP_STUDY: deep-study: this PR was reverted by PR 26220 (confirmed_revert, reason=ci_or_test_failure)
BODY: Continuing from https://github.com/vllm-project/vllm/pull/25603, this patch extends to the much faster flashinfer backend ⏎ (This might look like a big change, but I am going to rebase onto #25603 and most of it will go away, mostly just look at the flashinfer.py file) ⏎  ⏎ ## Purpose ⏎  ⏎ Add optional determinism to flashinfer backend. ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=FLASHINFER VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT=1 pytest -s -v tests/v1/generat …[truncated]

### L3-1838cd4860  (L3, 2025-10-04, sha 1838cd4860cf, PR #26220)
TITLE: Revert "Add batch invariant kernel override for FlashInfer backend [2/n]" (#26220)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+5/-32); tests/v1/generation/test_batch_invariance.py (+23/-40); vllm/model_executor/layers/batch_invariant.py (+1/-12)
LABELS: ready, v1
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 25769 reason=ci_or_test_failure
BODY: Reverts vllm-project/vllm#25769 because it failed PyTorch Fullgraph Smoke Test

### L3-4570535ec4  (L3, 2025-10-04, sha 4570535ec41e, PR #26010)
TITLE: [Model] CLIP Embedding Support (#26010)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+5/-1); docs/models/supported_models.md (+1/-0); examples/offline_inference/vision_language_pooling.py (+31/-3); examples/online_serving/pooling/openai_chat_embedding_client_for_multimodal.py (+64/-8); tests/models/multimodal/pooling/test_clip.py (+138/-0); tests/models/registry.py (+8/-2); vllm/model_executor/models/bert.py (+1/-1); vllm/model_executor/models/clip.py (+596/-61); vllm/model_executor/models/registry.py (+1/-0); vllm/model_executor/models/vision.py (+4/-2); (+1 more)
LABELS: documentation, new-model, ready, multi-modality
BODY: ## Purpose ⏎  ⏎ Support CLIP text and image embedding in the same model. ⏎ - For text inputs, we only apply `token_embedding` when calling `get_input_embeddings`. The rest of the text embedding and the encoder logic are applied when calling `forward` on the model. ⏎ - For image inputs, we apply vision embeddings when calling `get_input_embeddings`. Since the model doesn't have a decoder, we directly return the embeddings inside the `forward` method. ⏎ - In …[truncated]

### L3-5c057e068f  (L3, 2025-10-04, sha 5c057e068fc6, PR #26096)
TITLE: [CPU] Refine batch reorder of CPU attention backend (#26096)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+42/-87); vllm/v1/worker/cpu_model_runner.py (+2/-41)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Refine batch reorder of CPU attention backend to put decode requests in front of prefill requests likes other attention backends.   ⏎  ⏎ ## Test Plan ⏎  ⏎ CI tests ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-d6953beb91  (L3, 2025-10-05, sha d6953beb91da, PR #26247)
TITLE: Convert formatting to use `ruff` instead of `yapf` + `isort` (#26247)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v1_backend, L3.flash_attn.v1_backend, L3.flash_attn.upstream_pip, L3.flash_attn.fa_utils, L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.prefix_prefill, L3.triton.flash_attention_rocm, L3.triton.decode_attention, L3.triton.chunked_prefill_paged_decode, L3.triton.unified_attention, L3.triton.v1_backend, L3.merge.triton_lse, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.mla.common_v1, L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.flashmla_sparse, L3.mla.rocm_aiter, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.flex_attention, L3.tree_attention
FILES: setup.py (+151/-104); .buildkite/pyproject.toml (+0/-46); .pre-commit-config.yaml (+0/-12); benchmarks/benchmark_block_pool.py (+1/-1); benchmarks/benchmark_ngram_proposer.py (+1/-1); benchmarks/benchmark_serving_structured_output.py (+2/-3); benchmarks/pyproject.toml (+0/-49); cmake/hipify.py (+24/-19); csrc/cutlass_extensions/vllm_cutlass_library_extension.py (+13/-15); csrc/moe/marlin_moe_wna16/generate_kernels.py (+24/-18); (+1498 more)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, tpu, speculative-decoding, ready, ci/build
ISSUES: #17657 Migrating from `yapf` to `ruff format`
BODY: Closes #17657 ⏎  ⏎ This is a massive change and would be impossible to merge during the week. Any PR's merged while this was waiting for CI would cause merge conflicts which would be time consuming to solve in this PR. ⏎  ⏎ Therefore, the plan to getting this PR merged quickly is as follows: ⏎  ⏎ - First commit: ⏎   - Make the changes to the pre-commit ⏎   - Run `pre-commit run -a` and note all the things `ruff` couldn't automatically fix ⏎   - Add these as `pre-f …[truncated]

### L3-4e256cadc2  (L3, 2025-10-05, sha 4e256cadc217, PR #26251)
TITLE: Remove all references to `yapf` as it's no longer used (#26251)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.rocm_aiter
FILES: csrc/quantization/machete/generate.py (+0/-5); examples/others/tensorize_vllm_model.py (+95/-73); tests/compile/test_silu_mul_quant_fusion.py (+0/-5); tests/distributed/test_expert_parallel.py (+22/-20); tests/distributed/test_pipeline_parallel.py (+3/-3); tests/engine/test_arg_utils.py (+10/-25); tests/entrypoints/test_chat_utils.py (+179/-209); tests/lora/test_layers.py (+0/-5); tests/model_executor/model_loader/tensorizer_loader/test_tensorizer.py (+0/-4); tests/models/multimodal/generation/test_common.py (+191/-169); (+68 more)
LABELS: documentation, new-model, rocm, frontend, tpu, ready, v1, multi-modality, qwen, deepseek
BODY: Now that we don't have `yapf` fighting `isort`, we can remove any annotations that disabled `yapf`. ⏎  ⏎ NOTE: There were a couple of files which had `yapf` globally disabled, I think this caused ruff to ignore them too. That's why the lines changed count is so high.

### L3-6b6e98775f  (L3, 2025-10-05, sha 6b6e98775f24, PR #25998)
TITLE: [NVIDIA] flashinfer TRTLLM attention prefill token limit (#25998)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+12/-5)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Change the heuristic so that the flashinfer TRTLLM attention gets used more for prefill. Previously it was only used for `<= 256` tokens despite being faster (benchmark below) for all cases tested.  ⏎  ⏎ ## Test Plan ⏎  ⏎ benchmark using `benchmarks/kernels/benchmark_trtllm_prefill_attention.py` , datapoints causing OOM were left out ⏎  ⏎ ## Test Result ⏎ Benchmark results below. `speedup_% > 0` always meaning TRTLLM attention is always faster for  …[truncated]

### L3-1c0c68202c  (L3, 2025-10-05, sha 1c0c68202cc1, PR #26254)
TITLE: Fix per file ruff ignores related to typing (#26254)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.flashmla_v0_adapter, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+6/-6); vllm/attention/layer.py (+3/-3); vllm/attention/layers/chunked_local_attention.py (+2/-2); vllm/attention/ops/flashmla.py (+5/-5); vllm/attention/ops/paged_attn.py (+5/-5); vllm/utils/flashinfer.py (+4/-4); vllm/v1/attention/backends/flashinfer.py (+25/-25); pyproject.toml (+1/-17); tests/compile/test_full_graph.py (+2/-2); tests/entrypoints/openai/test_serving_chat.py (+3/-3); (+22 more)
LABELS: structured-output, frontend, ready, v1, multi-modality, gpt-oss
BODY: Forward fixes some of the issues skipped by https://github.com/vllm-project/vllm/pull/26247

### L3-432e1cbc23  (L3, 2025-10-05, sha 432e1cbc2324, PR #25933)
TITLE: [Bugfix]: Assertion error when using FlashInfer backend (#25933)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/fp8.py (+2/-2)
LABELS: ready
ISSUES: #25928 [Bug]: Assertion error when using FlashInfer backend
BODY: ## Purpose ⏎  ⏎ Fixes #25928. ⏎  ⏎ ## Test Plan ⏎  ⏎ The following does not err any more: ⏎  ⏎ ```bash ⏎ OMP_NUM_THREADS=8 VLLM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_ATTENTION_BACKEND=FLASHINFER VLLM_USE_FLASHINFER_MOE_FP8=1 vllm serve --async-scheduling --gpu-memory-utilization 0.8 --enable-auto-tool-choice --tool-call-parser hermes --model=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 ⏎ ``` ⏎ and  ⏎ with Qwen/Qwen3-4B-Instruct-2507-FP8. ⏎  ⏎ ## Test Result ⏎  ⏎ Runs successfully without  …[truncated]

### L3-b893d661b1  (L3, 2025-10-05, sha b893d661b1b9, PR #26259)
TITLE: Fix per file ruff ignores related to simplification (#26259)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/triton_reshape_and_cache_flash.py (+3/-6); pyproject.toml (+0/-34); tests/distributed/test_expert_placement.py (+1/-4); tests/kernels/attention/test_cutlass_mla_decode.py (+1/-4); tests/kernels/attention/test_flashmla.py (+1/-4); tests/kernels/attention/test_lightning_attn.py (+1/-4); tests/kernels/moe/test_pplx_moe.py (+1/-4); tests/kernels/quantization/test_cutlass_scaled_mm.py (+2/-8); tests/kernels/test_onednn.py (+1/-4); tests/kernels/utils.py (+3/-7); (+22 more)
LABELS: rocm, frontend, tpu, ready, v1, multi-modality
BODY: Forward fixes some of the issues skipped by #26247

### L3-6c04638214  (L3, 2025-10-06, sha 6c04638214d4, PR #26262)
TITLE: Fix per file ruff ignores related to line length (#26262)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: benchmarks/benchmark_ngram_proposer.py (+1/-1); benchmarks/benchmark_serving_structured_output.py (+2/-2); csrc/cutlass_extensions/vllm_cutlass_library_extension.py (+3/-3); examples/offline_inference/vision_language_pooling.py (+1/-1); examples/online_serving/disaggregated_serving/disagg_proxy_demo.py (+2/-2); pyproject.toml (+0/-46); tests/compile/piecewise/test_simple.py (+17/-11); tests/compile/piecewise/test_toy_llama.py (+9/-5); tests/compile/test_functionalization.py (+1/-1); tests/compile/test_fusion_attn.py (+1/-1); (+55 more)
LABELS: documentation, performance, structured-output, frontend, tpu, speculative-decoding, ready, v1, multi-modality, tool-calling
BODY: Forward fixes the last of the issues skipped by #26247

### L3-4727a8afa7  (L3, 2025-10-06, sha 4727a8afa795, PR #24463)
TITLE: [Attention] Remove unused reorder_batch method (#24463)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.abstract_interface, L3.flex_attention, L3.tree_attention
FILES: vllm/v1/attention/backends/flashinfer.py (+2/-4); vllm/v1/attention/backends/flex_attention.py (+1/-10); vllm/v1/attention/backends/tree_attn.py (+3/-14); vllm/v1/attention/backends/utils.py (+0/-22); vllm/v1/attention/backends/xformers.py (+1/-13); tests/v1/logits_processors/test_correctness.py (+1/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎ The `AttentionMetadataBuilder` class method `reorder_batch` is no longer used and has been replaced by `reorder_batch_to_split_decodes_and_prefills` from `vllm.v1.attention.backends.utils`. This PR removes the old unused code. ⏎  ⏎ ## Test Plan ⏎ `VLLM_ATTENTION_BACKEND=FLASH_ATTN_MLA chg run --gpus 1 -- lm_eval --model vllm --model_args '{"pretrained": "deepseek-ai/DeepSeek-V2-Lite-Chat", "trust_remote_code": true, "kv_cache_dtype": "auto"} …[truncated]

### L3-20db99cc69  (L3, 2025-10-06, sha 20db99cc692a, PR #26188)
TITLE: [CI Bugfix] Make sure TRTLLM attention is available in test_blackwell_moe (#26188)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/quantization/test_blackwell_moe.py (+9/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ Sometimes the Blackwell Quantized MoE Test CI fails due to connectivity problems with NVIDIA's artifactory. In the case of GPT-OSS, we need TRTLLM attention in order to run, so force availability ⏎  ⏎ ``` ⏎ (EngineCore_DP0 pid=3728) WARNING 10-03 10:13:32 [flashinfer.py:180] Failed to connect to NVIDIA artifactory: HTTPSConnectionPool(host='edge.urm.nvidia.com', port=443): Read timed out. (read timeout=5) ⏎ ``` ⏎ https://buildkite.com/vllm/ci/b …[truncated]

### L3-f231e5bc21  (L3, 2025-10-06, sha f231e5bc21d5, PR #25507)
TITLE: [ROCm] Split AITER unified attention into its own backend (#25507)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.rocm.v1_rocm_attn, L3.rocm.aiter_unified, L3.dispatch.selector, L3.dispatch.registry, L3.platform.rocm_selection
FILES: vllm/attention/backends/registry.py (+1/-0); vllm/attention/selector.py (+1/-0); vllm/engine/arg_utils.py (+1/-0); vllm/envs.py (+7/-6); vllm/platforms/rocm.py (+18/-10); vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+203/-0); vllm/v1/attention/backends/rocm_attn.py (+40/-121); tests/compile/test_fusion_attn.py (+54/-164)
LABELS: rocm, ready, v1
BODY: Follow up on #24648 ⏎ Splitting AITER Unified attention into its own backend class ⏎ The non-MLA selection logic on ROCm is now as follows: ⏎ - The default is `TritonAttentionBackend` ⏎ - If AITER is enabled together with VLLM_ROCM_USE_AITER_MHA, or ROCM_AITER_FA backend specified, use `AiterFlashAttentionBackend` ⏎ - if AITER is enabled together with VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION, or ROCM_AITER_UNIFIED_ATTN backend specified, use `RocmAiterUnified …[truncated]

### L3-f77df94647  (L3, 2025-10-06, sha f77df94647ca, PR #26313)
TITLE: [Perf] Add decode full-graph support to FlashInfer-MLA backend (#26313)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer
FILES: vllm/v1/attention/backends/mla/flashinfer_mla.py (+12/-1)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ The annotation was missing from FlashInfer-MLA while the implementation has support. ⏎  ⏎ Running **DSR1-FP4 on 4xB200** gets me **97 TPS**: ⏎  ⏎ ```bash ⏎ VLLM_FLASHINFER_MOE_BACKEND=latency VLLM_USE_FLASHINFER_MOE_FP4=1 VLLM_ATTENTION_BACKEND=FLASHINFER_MLA vllm serve nvidia/DeepSeek-R1-FP4 -tp 4 --max-model-len 32768 --max-num-seqs 128 --no-enable-prefix-caching --async-scheduling --port 8049 ⏎ ``` ⏎  ⏎ I also tested on a local development branch  …[truncated]

### L3-41f1cf38f2  (L3, 2025-10-07, sha 41f1cf38f2e1, PR #21166)
TITLE: [Feature][OCP MX] Support mxfp6 and mixed mxfp6-mxfp4 (#21166)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+2/-2); docs/features/quantization/quark.md (+8/-4); tests/kernels/moe/test_ocp_mx_moe.py (+23/-17); tests/quantization/test_quark.py (+74/-19); vllm/model_executor/layers/fused_moe/config.py (+56/-17); vllm/model_executor/layers/fused_moe/fused_batched_moe.py (+2/-2); vllm/model_executor/layers/fused_moe/fused_moe.py (+69/-21); vllm/model_executor/layers/fused_moe/utils.py (+60/-13); vllm/model_executor/layers/quantization/mxfp4.py (+3/-2); vllm/model_executor/layers/quantization/quark/quark.py (+20/-18); (+8 more)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1
BODY: As per title. ⏎  ⏎ Support for MXFP4 in vllm was added in https://github.com/vllm-project/vllm/pull/16943 and https://github.com/vllm-project/vllm/pull/17888. ⏎  ⏎ However, some hardware as AMD Instinct MI350/MI355 support as well math in mxfp6 or mixed fp4 / fp6 (see e.g. https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf, with switches for the input dtype …[truncated]

### L3-6f59beaf0b  (L3, 2025-10-07, sha 6f59beaf0b1f, PR #26340)
TITLE: [Model] Add support for ModernBertForTokenClassification (#26340)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/models/supported_models.md (+1/-0); tests/models/language/pooling/test_token_classification.py (+32/-1); tests/models/registry.py (+3/-0); vllm/model_executor/models/modernbert.py (+72/-1); vllm/model_executor/models/registry.py (+4/-0)
LABELS: documentation, new-model, ready
BODY: ## Purpose ⏎ Add support for ModernBertForTokenClassification. ⏎ I got inspired from https://github.com/vllm-project/vllm/pull/24872 , and adapted to ModernBert architecture. I did not touch to flex attention backend or anything, though, because I expected the changes already done in that previous PR to make BERT work for NER, would work for ModernBert as well. ⏎  ⏎ ## Test Plan ⏎ Added a single test case in `tests/models/language/pooling/test_token_classi …[truncated]

### L3-3d1f67616d  (L3, 2025-10-07, sha 3d1f67616da8, PR #25984)
TITLE: [Spec Decode] Enable efficient speculative decoding with FlashInfer-MLA (#25984)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashinfer, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/mla/common.py (+24/-2); vllm/v1/attention/backends/mla/flashinfer_mla.py (+15/-1); vllm/v1/attention/backends/utils.py (+3/-2)
LABELS: speculative-decoding, ready, v1
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ This PR refactors the `MLACommonMetadataBuilder` to easily support spec decode kernel optimization in MLA implementations. This is used to enable FlashInfer-MLA support using the trtllm-gen kernels which have explicit support for spec-as-decode. ⏎  ⏎ ## Test Plan ⏎  ⏎ I ran a suite of evals over `nvidia/DeepSeek-R1-FP4` and `deepseek-ai/DeepSeek-R1-0528` on 4xB200 and 8xB200 respectively, using `Cutlass-MLA` and `FlashInfer-MLA` backends. Run …[truncated]

### L3-eb577e4655  (L3, 2025-10-07, sha eb577e465589, PR #26325)
TITLE: [Bugfix] Add missing sink tensor into flash attn cascade attn implementation (#26325)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+5/-0)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ This PR fixes the missing sink tensor used in the GPT OSS model within the flash attention cascade attention implementation. ⏎  ⏎ It is to address the issue https://github.com/vllm-project/vllm/issues/26203 ⏎  ⏎ ## Test Plan ⏎  ⏎ I can't find a good way to add a unit test. The following is the test plan for the integration test. ⏎ First, set up the vLLM environment, and apply the code change to force cascade attention. ⏎ Then, send the provided query …[truncated]

### L3-caf8b1c084  (L3, 2025-10-07, sha caf8b1c0840b, PR #26361)
TITLE: [Bugfix] Fix MTP+FlashInfer crash when trtllm kernels are available but disabled (#26361)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+2/-0)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ The `reorder_batch_threshold` is set based on `can_use_trtllm_attention`, assuming that `use_trtllm_attention` will return True if spec is enabled and it can be used. ⏎  ⏎ There is a missing edge-case here for the force disable trtllm attention flag. In this case `can_use_trtllm_attention` says True, but `use_trtllm_attention`  says False, and the mismatch causes incorrect padding leading to the crash observed in #26312

### L3-335b28f7d1  (L3, 2025-10-07, sha 335b28f7d102, PR #26279)
TITLE: [TPU] Rename tpu_commons to tpu_inference (#26279)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+1/-1); vllm/distributed/device_communicators/tpu_communicator.py (+7/-7); vllm/model_executor/model_loader/default_loader.py (+2/-2); vllm/platforms/__init__.py (+1/-1); vllm/platforms/tpu.py (+5/-5); vllm/v1/worker/tpu_worker.py (+6/-6)
LABELS: tpu, ready, v1
BODY: ## Purpose ⏎  ⏎ We are renaming tpu_commons package to tpu_inference. This is being done for public release. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-cd9890544b  (L3, 2025-10-08, sha cd9890544b97, PR #23485)
TITLE: fix(v1/kv_cache): resolve async KV transfer bug in cascade attention (#23485)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/core/kv_cache_coordinator.py (+9/-16); vllm/v1/core/kv_cache_manager.py (+18/-28); vllm/v1/core/sched/scheduler.py (+1/-1); vllm/v1/core/single_type_kv_cache_manager.py (+13/-27)
LABELS: ready, v1
BODY: ## Purpose ⏎ Solves #23130. This change fixes a critical bug in vLLM's cascade attention optimization in the V1 arch. The bug is in `get_num_common_prefix_blocks()`, which determines how many KV cache blocks are shared among all currently running requests to enable cascade attention optimizations. ⏎  ⏎ ## Changes made ⏎  ⏎ * Replace ref_cnt-based common prefix detection with running request tracking ⏎ * Update get_num_common_prefix_blocks() to accept running …[truncated]

### L3-127c8b782a  (L3, 2025-10-08, sha 127c8b782a63, PR #25931)
TITLE: Add gather_indexer_k_quant_cache kernel (#25931)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+120/-0); csrc/cache.h (+8/-0); csrc/torch_bindings.cpp (+6/-0); vllm/_custom_ops.py (+12/-0)
LABELS: ready
BODY: This PR added `cp_gather_indexer_k_quant_cache` for getting quantized k/k_scale from indexer k cache.

### L3-f80e7866c0  (L3, 2025-10-08, sha f80e7866c096, PR #26125)
TITLE: [Misc] Clean up cruft from previous FlashMLA sparse implementation (#26125)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.mla.flashmla_sparse, L3.platform.cuda_selection
FILES: vllm/attention/ops/flashmla.py (+35/-9); vllm/platforms/cuda.py (+4/-4); vllm/v1/attention/backends/mla/flashmla.py (+2/-2); vllm/v1/attention/backends/mla/flashmla_sparse.py (+0/-37); tests/kernels/attention/test_attention_selector.py (+2/-2); tests/kernels/attention/test_flashmla.py (+6/-4); tests/kernels/attention/test_flashmla_sparse.py (+9/-16); tests/v1/attention/test_sparse_mla_backends.py (+17/-66); tests/v1/attention/utils.py (+4/-0)
LABELS: ready, v1
BODY: Clean up cruft from previous FlashMLA sparse implementation and get tests passing again after final DeepseekV32 changes

### L3-241b4cfe66  (L3, 2025-10-08, sha 241b4cfe6609, PR #25293)
TITLE: [Refactor] Refactor FP8 & INT8 Quant Folder inside `w8a8` (#25293)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.flash_attn.fork_inline_cmake, L3.rocm.custom_paged
FILES: csrc/attention/attention_kernels.cuh (+2/-2); csrc/cache_kernels.cu (+2/-2); CMakeLists.txt (+21/-20); csrc/cub_helpers.h (+3/-2); csrc/layernorm_quant_kernels.cu (+1/-1); csrc/quantization/activation_kernels.cu (+1/-1); csrc/quantization/fused_kernels/quant_conversions.cuh (+1/-1); csrc/quantization/w8a8/cutlass/Epilogues.md (+0/-0); csrc/quantization/w8a8/cutlass/c3x/cutlass_gemm_caller.cuh (+0/-0); csrc/quantization/w8a8/cutlass/c3x/scaled_mm.cuh (+0/-0); (+44 more)
LABELS: rocm, ready, ci/build, qwen
BODY: ## Purpose ⏎  ⏎ Refactor FP8 & INT8 & Cutlass Quant Folder inside `w8a8` for better code maintainability ⏎  ⏎ ## Test ⏎  ⏎ ``` ⏎ lm_eval   --model vllm   --model_args "pretrained=Qwen/Qwen3-30B-A3B-FP8,max_model_len=32768,enforce_eager=True"   --trust_remote_code   --tasks gsm8k   --num_fewshot 5   --batch_size auto ⏎ vllm (pretrained=Qwen/Qwen3-30B-A3B-FP8,max_model_len=32768,enforce_eager=True,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewsh …[truncated]

### L3-76879cc160  (L3, 2025-10-08, sha 76879cc16042, PR #25900)
TITLE: [Attention] Implement universal BACKEND_MAP (#25900)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.dispatch.selector, L3.dispatch.registry
FILES: vllm/attention/backends/registry.py (+83/-2); vllm/attention/layer.py (+2/-2); vllm/attention/selector.py (+1/-14); tests/kernels/attention/test_attention_selector.py (+2/-2); tests/kernels/attention/test_rocm_attention_selector.py (+1/-1); tests/v1/attention/test_attention_backends.py (+2/-2); tests/v1/attention/test_mla_backends.py (+3/-3); tests/v1/attention/utils.py (+14/-38); tests/v1/spec_decode/test_eagle.py (+5/-5); tests/v1/spec_decode/test_mtp.py (+2/-2); (+2 more)
LABELS: rocm, speculative-decoding, ready, v1, qwen, kv-connector
BODY: ## Purpose ⏎ This PR continues the attention backend selection refactor #24794, which is now being split up. Here, we allow backends to register themselves in a `BACKEND_MAP` which will be used later in the refactor. This also enables out-of-tree backends for other hardware to plug in neatly. ⏎  ⏎ ## Test Plan ⏎ CI should be sufficient ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-e614ab7806  (L3, 2025-10-08, sha e614ab780645, PR #25103)
TITLE: Separate MLAAttention class from Attention (#25103)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+26/-0); vllm/attention/layer.py (+318/-22); vllm/model_executor/layers/mla.py (+11/-21); vllm/v1/attention/backends/utils.py (+2/-2); vllm/config/compilation.py (+2/-0); vllm/model_executor/model_loader/utils.py (+4/-4); vllm/model_executor/models/deepseek_v2.py (+2/-2); vllm/v1/spec_decode/eagle.py (+3/-3); vllm/v1/worker/gpu_model_runner.py (+72/-69); vllm/v1/worker/tpu_model_runner.py (+62/-40)
LABELS: tpu, speculative-decoding, ready, v1, deepseek
ISSUES: #24620 [Refactor]: Make an common MLAAttention Layer and custom OP
BODY: ## Purpose ⏎ This PR implements the first step of #24620 by separating Multi-Head Latent Attention into its own dedicated `AttentionLayerBase` subclass.  ⏎  ⏎ --- ⏎ [details omitted]

### L3-b82f4307c9  (L3, 2025-10-08, sha b82f4307c9b9, PR #25924)
TITLE: [Bugfix][Flashinfer] fix VLLM_USE_TRTLLM_ATTENTION issue for models with diff hyperparameters (#25924)
SOURCES: path_core, subject_keyword, release_notes, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/flashinfer.py (+27/-18); vllm/v1/attention/backends/utils.py (+13/-18)
LABELS: ready, v1
DEEP_STUDY: deep-study correctness case vllm:b82f4307c9: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎  ⏎ - Issue: for the models with diff hyperparameters(`window_left`, `logits_soft_cap`, `sm_scale`) to use flashinfer backend, we always need to set `VLLM_USE_TRTLLM_ATTENTION=1` to bypass hyperparameters check in `infer_global_hyperparameters()` ⏎   - `VLLM_USE_TRTLLM_ATTENTION` should not be used directly to determine TRTLLM attention is used or not. Should use `prefill_use_trtllm` and `decode_use_trtllm` instead. ⏎ - This PR delayed the ch …[truncated]

### L3-2a03f93de9  (L3, 2025-10-08, sha 2a03f93de9d1, PR #26441)
TITLE: [Attention] Register FLASHMLA_SPARSE (#26441)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_sparse, L3.dispatch.registry
FILES: vllm/attention/backends/registry.py (+2/-0); vllm/v1/attention/backends/mla/flashmla_sparse.py (+1/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎ Add `FLASHMLA_SPARSE` to the backend registry ⏎  ⏎ ## Test Plan ⏎ CI should suffice ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-d24cf322e1  (L3, 2025-10-08, sha d24cf322e19a, PR #24486)
TITLE: [Hybrid]: Decouple Kernel Block Size from KV Page Size (#24486)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.xformers.v1_backend, L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.aiter_fa, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.dispatch.abstract_interface, L3.platform.cuda_selection, L3.tree_attention
FILES: vllm/attention/backends/abstract.py (+17/-1); vllm/v1/attention/backends/flash_attn.py (+6/-1); vllm/v1/attention/backends/flashinfer.py (+8/-0); vllm/v1/attention/backends/mla/cutlass_mla.py (+5/-0); vllm/v1/attention/backends/mla/flashmla.py (+5/-1); vllm/v1/attention/backends/rocm_aiter_fa.py (+6/-1); vllm/v1/attention/backends/tree_attn.py (+6/-1); vllm/v1/attention/backends/triton_attn.py (+6/-1); vllm/v1/attention/backends/xformers.py (+6/-1); tests/v1/worker/test_gpu_input_batch.py (+3/-0); (+8 more)
LABELS: documentation, performance, rocm, tpu, speculative-decoding, ready, ci/build, v1, qwen, deepseek
BODY: ## Purpose ⏎  This PR introduces a hybrid cache architecture that separates logical kernel block size from ⏎   physical page size, enabling more flexible memory management. Key changes include: ⏎  ⏎   - Added kernel_block_size field to CacheConfig for logical block sizing ⏎   - Enhanced platform-specific configurations for CUDA and ROCm to support hybrid blocks ⏎   - Implemented block table conversion logic between physical and logical representations ⏎   - Ad …[truncated]

### L3-5e49c3e777  (L3, 2025-10-08, sha 5e49c3e777b5, PR #26326)
TITLE: Bump Flashinfer to v0.4.0 (#26326)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: docker/Dockerfile (+3/-3); docker/Dockerfile.nightly_torch (+2/-2); setup.py (+1/-1); vllm/v1/attention/backends/flashinfer.py (+4/-1); tests/kernels/attention/test_flashinfer_trtllm_attention.py (+9/-12); tests/kernels/quantization/nvfp4_utils.py (+5/-3); tests/quantization/test_blackwell_moe.py (+1/-1)
LABELS: ready, ci/build, v1
BODY: ## Purpose ⏎ Bump Flashinfer to v0.4.0. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-de253d63b7  (L3, 2025-10-09, sha de253d63b7a4, PR #26439)
TITLE: [Hardware][AMD] Enable FlexAttention backend on ROCm (#26439)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+3/-0)
LABELS: rocm, ready
BODY: ## Purpose ⏎ Enable FlexAttention on ROCm if it is specified e.g. via `VLLM_ATTENTION_BACKEND=FLEX_ATTENTION`. This makes progress towards batch invariant inference on AMD. ⏎  ⏎ ## Test Plan ⏎ Comparing correctness of default attention backend vs FlexAttention ⏎ `lm_eval --model local-completions --model_args model=meta-llama/Llama-3.1-8B,base_url=http://0.0.0.0:8000/v1/completions,num_concurrent=128,max_retries=5 --tasks gsm8k` ⏎  ⏎ ## Test Result ⏎ Default: ⏎ `` …[truncated]

### L3-3b736e1c38  (L3, 2025-10-09, sha 3b736e1c3869, PR #25049)
TITLE: [Attention][DCP] Support DCP with query length > 1 (MTP) with FA3 (#25049)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_build, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.flashattn, L3.mla.rocm_aiter, L3.dispatch.abstract_interface
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/v1/attention/backends/mla/common.py (+14/-2); vllm/v1/attention/backends/mla/flashattn_mla.py (+7/-9); vllm/v1/attention/backends/mla/flashmla.py (+2/-0); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+2/-0); vllm/v1/attention/backends/utils.py (+3/-0); vllm/v1/worker/gpu_model_runner.py (+14/-1); vllm/v1/spec_decode/eagle.py (+2/-0)
LABELS: rocm, speculative-decoding, ready, ci/build, v1
BODY: ## Purpose ⏎  ⏎ Combined with https://github.com/vllm-project/flash-attention/pull/93, this is to enable MTP (multi-token prediction) with DCP (decode context parallelism). It also allows prefill/decode to be mixed in a batch. ⏎  ⏎ See https://github.com/vllm-project/flash-attention/pull/93 for the implementation and solution details. Here we just need to pass the cp world size and cp rank. ⏎  ⏎ ## Test Plan ⏎  ⏎ 1. Benchmark output token throughput ⏎ 2. Verify LM …[truncated]

### L3-ec10fd0abc  (L3, 2025-10-09, sha ec10fd0abcc7, PR #16601)
TITLE: [Bugfix] Move current_platform import to avoid python import cache. (#16601)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/selector.py (+2/-1); tests/kernels/attention/test_attention_selector.py (+6/-6)
LABELS: ready
BODY: We are adding a new platform to vLLM using plugin mechanism. To test our platform, we compare it with the CPU platform within a single file.  ⏎ ``` ⏎ from vllm import LLM, SamplingParams ⏎  ⏎ import os ⏎ from vllm.platforms import builtin_platform_plugins ⏎ from vllm.utils import resolve_obj_by_qualname ⏎ from vllm import platforms ⏎  ⏎ # Sample prompts. ⏎ prompts = [ ⏎     "Hello, my name is", ⏎ ] ⏎ # Create a sampling params object. ⏎ sampling_params = SamplingParams(temp …[truncated]

### L3-4069db3f2e  (L3, 2025-10-09, sha 4069db3f2e3f, PR #25947)
TITLE: [Bugfix] Enable padded FP4 quantization (#25947)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+0/-2); vllm/_custom_ops.py (+1/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎ This PR fixes an issue with padded FP4 quantization, where previously initialized values in the allocated tensors aren't overwritten properly when quantizing to FP a tensor which requires padding. ⏎  ⏎ This issue prevents running NVIDIA's new Nemotron-Nano-9B-v2 with TP2, as some of the tensor sizes in such cases do not divide evenly by 128. ⏎  ⏎ Note that this specific model doesn't work with TP>2, and this PR doesn't solve issues that come u …[truncated]

### L3-47e66c24e2  (L3, 2025-10-09, sha 47e66c24e277, PR #26145)
TITLE: [Model] Apply shared experts overlap optimization to all models with shared experts (#26145)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/__init__.py (+2/-0); vllm/model_executor/layers/fused_moe/shared_fused_moe.py (+23/-12); vllm/model_executor/layers/quantization/fp8.py (+2/-0); vllm/model_executor/layers/shared_fused_moe/__init__.py (+0/-5); vllm/model_executor/models/aria.py (+15/-13); vllm/model_executor/models/bailing_moe.py (+26/-21); vllm/model_executor/models/deepseek_v2.py (+24/-45); vllm/model_executor/models/dots1.py (+22/-18); vllm/model_executor/models/ernie45_moe.py (+20/-20); vllm/model_executor/models/ernie45_vl_moe.py (+34/-23); (+5 more)
LABELS: ready, llama, qwen, deepseek
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ - Use `SharedFusedMoE` in all models that use shared experts.  This will enable the shared experts/communication overlap  optimization for all the changed models. ⏎ - Move `SharedFusedMoE` class to `fused_moe` directory. ⏎ - Update `SharedFusedMoE` to behave like `FusedMoE` when the `shared_experts` are `None` ⏎ - Disable shared expert overlap if EP is disabled or we are not using flashinfer + DP since there is nothing to be gained in this c …[truncated]

### L3-c9d33c60dc  (L3, 2025-10-09, sha c9d33c60dcdc, PR #26443)
TITLE: [UX] Add FlashInfer as default CUDA dependency (#26443)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: docker/Dockerfile (+8/-69); requirements/cuda.txt (+2/-0); setup.py (+1/-2); vllm/utils/flashinfer.py (+9/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ It seems that FlashInfer does not require `nvcc` to installed from source anymore since `flashinfer-python>=0.2.9`, so we can move it to be a default dependency! ⏎  ⏎ Obviously to have it JIT compile kernels it needs to have `nvcc` available, so we will currently add that condition to `has_flashinfer()`. This is more conservative than it needs to be, as if we have an AOT compiled wheel installed there is likely no need for `nvcc`, but we  …[truncated]

### L3-44f633dba1  (L3, 2025-10-09, sha 44f633dba17f, PR #25674)
TITLE: [Flashinfer][gpt-oss] Support FP8-qkv Flashinfer TRTLLM Sinks Attention (#25674)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+0/-5); tests/kernels/attention/test_flashinfer_trtllm_attention.py (+76/-41)
LABELS: ready, ci/build, v1, gpt-oss
BODY: ## Purpose ⏎ Support FP8-qkv Flashinfer TRTLLM sinks attention. ⏎ Note: require flashinfer v0.4.0rc4(updating in #26326) ⏎ https://github.com/flashinfer-ai/flashinfer/pull/1758 ⏎  ⏎ ## Test Plan && Test Result ⏎ #### Kernel unit test: ⏎ `tests/kernels/attention/test_flashinfer_trtllm_attention.py` ⏎ ``` ⏎ ===== 224 passed, 16 skipped in 30.94s ==== ⏎ ``` ⏎ #### E2E accuracy: ⏎ kv_cache_dtype=fp8 ⏎ ``` ⏎ [{'eval_name': 'gpqa', 'model_name': 'gpt-oss-120b-high_temp1.0_2025092 …[truncated]

### L3-6e783bc54b  (L3, 2025-10-09, sha 6e783bc54b03, PR #26499)
TITLE: [Bugfix] Fix CUDA graph selection bug in FlashInfer at high concurrency (#26499)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+9/-2)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ Previously, FlashInfer would choose to use decode CUDA graphs if `num_decodes < threshold`, even if spec decoding was enabled and the number of tokens in the batch was much higher. This can cause issues when `num_decode_tokens > max_cudagraph_size`. ⏎  ⏎ I fixed the check to correctly use `num_decode_tokens` instead, and also updated the constructor to update the threshold to a higher value when speculative decoding is used.

### L3-96ad65b7fe  (L3, 2025-10-10, sha 96ad65b7fe51, PR #24440)
TITLE: [Transform] [Quantization] Add QuTLASS support to vLLM (#24440)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-0); cmake/external_projects/qutlass.cmake (+97/-0); .buildkite/test-pipeline.yaml (+2/-0); benchmarks/kernels/bench_mxfp4_qutlass.py (+191/-0); benchmarks/kernels/bench_nvfp4_qutlass.py (+207/-0); tests/kernels/quantization/test_mxfp4_qutlass.py (+303/-0); tests/kernels/quantization/test_nvfp4_qutlass.py (+268/-0); tests/quantization/fp_quant.py (+32/-0); vllm/_custom_ops.py (+139/-1); vllm/model_executor/layers/quantization/__init__.py (+3/-0); (+2 more)
LABELS: performance, ready, ci/build
BODY: ## Purpose ⏎  ⏎ This pull request brings in the QuTLASS library: https://github.com/iST-DASLab/qutlass ⏎  ⏎ QuTLASS is a high-performance library designed for low-precision kernel support in deep learning quantization, built on top of [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass). ⏎  ⏎ QuTLASS v0.1.0 introduces 4-bit microscaling routines tailored for Large Language Model (LLM) inference on NVIDIA Blackwell GPUs. ⏎  ⏎ * Online rotations: ⏎    - Fused transfo …[truncated]

### L3-213b64452a  (L3, 2025-10-10, sha 213b64452a4a, PR #26535)
TITLE: [Bugfix] Convert untraceable GroupShape to list for AMD impl (#26535)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/utils/fp8_utils.py (+2/-2)
LABELS: rocm, ready
BODY: ## Purpose ⏎ https://github.com/vllm-project/vllm/pull/25696 converts GroupShape to list where possible for cutlass custom ops; however, this was not done for the aiter or triton impl, which causes the code to fail in dynamo tracing ⏎  ⏎ ## Test Plan ⏎ Run DeepseekR1-0528 on AMD hardware with FP8 kernels ⏎ ``` ⏎  FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE VLLM_USE_V1=1 VLLM_MLA_DISABLE=0 VLLM_FP8_PADDING=1 VLLM_USE_TRITON_FLASH_ATTN=1 VLLM_USE_ROCM_FP8_FLASH_ATT …[truncated]

### L3-e94cfd51da  (L3, 2025-10-10, sha e94cfd51da5f, PR #26564)
TITLE: [BUG] Qwen3-next MTP. Fix attn metadata build bug (#26564)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/spec_decode/eagle.py (+6/-7)
LABELS: speculative-decoding, ready, v1, qwen
BODY: ## Purpose ⏎ After fixing #24486 Qwen3-next with FlashInfer full attn start working without MTP.  ⏎ But with MTP it fails.  ⏎  ⏎ The reason we choose incorrect attn metadata type for draft model (choose GDN instead of full attn).  ⏎  ⏎ Fix it.  ⏎  ⏎ ## Test Result ⏎ Qwen3-next with MTP works now.

### L3-cddce79fda  (L3, 2025-10-10, sha cddce79fdae2, PR #25845)
TITLE: [torch.compile] Make inductor partition rules respect splitting_ops #25691 (#25845)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/piecewise/test_multiple_graphs.py (+2/-2); tests/compile/piecewise/test_simple.py (+2/-2); tests/compile/piecewise/test_toy_llama.py (+2/-2); tests/compile/test_config.py (+53/-29); tests/compile/test_decorator.py (+3/-3); vllm/compilation/backends.py (+44/-8); vllm/compilation/compiler_interface.py (+12/-16); vllm/compilation/partition_rules.py (+95/-0); vllm/config/compilation.py (+54/-50)
LABELS: performance, ready, torch.compile, llama
ISSUES: #25691 [Feature]: Inductor partitioning should decide what ops to partition on dynamically
BODY: ### Key Changes ⏎  ⏎ - **Preserve user-specified `splitting_ops`**: When `use_inductor_graph_partition=True`, user-provided `splitting_ops` are now preserved in `_user_specified_splitting_ops` instead of being cleared ⏎ - **Dynamic partition rules**: Implement `_setup_dynamic_partition_rules()` using PyTorch 2.9+'s `register_should_partition_rule` API to register custom partition points ⏎ - **Robust fallback mechanism**: If dynamic rule registration fail …[truncated]

### L3-6f0f570c43  (L3, 2025-10-10, sha 6f0f570c436c, PR #26559)
TITLE: [deepseek] kernel block size for UniformTypeKVCacheSpecs (#26559)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+10/-2); vllm/v1/worker/gpu_model_runner.py (+9/-4)
LABELS: ready, v1, deepseek
ISSUES: #26524 [Bug]: prepare_kernel_block_sizes doesn't parse UniformTypeKVCacheSpecs
BODY: ## Purpose ⏎ After https://github.com/vllm-project/vllm/pull/24486 , deepseek 3.2 will throw this error: ⏎ ``` ⏎  NotImplementedError: unknown kv cache spec UniformTypeKVCacheSpecs ⏎  ``` ⏎ This PR fix it. ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/26524 ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ python3 examples/offline_inference/basic/generate.py --model deepseek-ai/DeepSeek-V3.2-Exp --gpu_memory_utilization 0.8 -tp 8 ⏎ ``` ⏎ ## Test Result ⏎  ⏎ ``` ⏎ ------------------------------ …[truncated]

### L3-0cd103e7cb  (L3, 2025-10-11, sha 0cd103e7cbf0, PR #26509)
TITLE: CP: make correct_attn_out robust to 4‑D views and fix Triton arg binding (#26509)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/common.py (+46/-8)
LABELS: ready
BODY: ## Purpose ⏎ The issue is found from @simon-mo where we want to bring up H100 resources for CI https://github.com/vllm-project/vllm/pull/26396 , and there is a failure https://buildkite.com/vllm/ci/builds/33954/steps/canvas?sid=0199c256-fc49-4e2d-afd6-c54961f0ffb0 with the error msg ⏎  ⏎ ``` ⏎ (Worker_TP0 pid=3232) ERROR 10-07 22:59:10 [multiproc_executor.py:706] TypeError: dynamic_func() got multiple values for argument 'HEAD_DIM' ⏎ ``` ⏎ After some investi …[truncated]

### L3-8fcaaf6a16  (L3, 2025-10-12, sha 8fcaaf6a165e, PR #26633)
TITLE: Update `Optional[x]` -> `x | None` and `Union[x, y]` to `x | y` (#26633)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v1_backend, L3.flash_attn.v1_backend, L3.flash_attn.fa_utils, L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.merge.triton_lse, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.rocm.aiter_unified, L3.mla.common_v1, L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.flashmla_sparse, L3.mla.rocm_aiter, L3.dispatch.selector, L3.dispatch.registry, L3.dispatch.abstract_interface, L3.flex_attention, L3.tree_attention
FILES: benchmarks/backend_request_func.py (+13/-14); benchmarks/benchmark_prefix_caching.py (+2/-3); benchmarks/benchmark_prioritization.py (+1/-2); benchmarks/benchmark_serving_structured_output.py (+3/-4); benchmarks/benchmark_utils.py (+8/-8); benchmarks/cutlass_benchmarks/sparse_benchmarks.py (+1/-2); benchmarks/cutlass_benchmarks/w8a8_benchmarks.py (+5/-6); benchmarks/fused_kernels/layernorm_rms_benchmarks.py (+4/-5); benchmarks/kernels/bench_per_token_quant_fp8.py (+1/-1); benchmarks/kernels/benchmark_device_communicators.py (+3/-3); (+934 more)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, tpu, speculative-decoding, ready, v1
BODY: Removes all `UP` rule ignores from `pyproject.toml` now that we no longer support Python 3.9 https://github.com/vllm-project/vllm/pull/26247. ⏎  ⏎ This will be the last repo-wide formatting PR for a long time. ⏎  ⏎ Thank you everyone for persevering with these changes. They will be good for the long-term health of the codebase.

### L3-18ed7746ea  (L3, 2025-10-12, sha 18ed7746eacb, PR #26339)
TITLE: [Feature] Add support for naver/splade-v3 (BERT-based sparse embedding model) (#26339)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/language/pooling/test_splade_sparse_pooler.py (+122/-0); tests/models/registry.py (+3/-0); vllm/model_executor/models/bert.py (+214/-0); vllm/model_executor/models/registry.py (+1/-0)
LABELS: new-model, ready
BODY: <h2>Purpose</h2> ⏎ <p>This PR adds <strong>official support for the <code inline="">naver/splade-v3</code> model</strong>, a BERT-based sparse retrieval model utilizing the <strong>SPLADE pooling mechanism</strong>.<br> ⏎ The implementation introduces the <code inline="">BertSpladeSparseEmbeddingModel</code> class, extending <code inline="">BertEmbeddingModel</code> to generate <strong>sparse lexical embeddings</strong> from the MLM head output (<cod …[truncated]

### L3-3263799056  (L3, 2025-10-13, sha 3263799056f4, PR #26373)
TITLE: [unrevert] Add batch invariant kernel override for FlashInfer backend [2/n] (#26373)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+29/-3); csrc/moe/topk_softmax_kernels.cu (+1/-3); tests/v1/generation/test_batch_invariance.py (+37/-28); vllm/model_executor/layers/batch_invariant.py (+14/-1)
LABELS: ready, ci/build, v1
BODY: This change reinstates already approved + landed #25769 based on the latest bump to flashinfer: ⏎  ⏎ https://github.com/vllm-project/vllm/pull/26326 ⏎  ⏎ It should *not* land before #26326

### L3-e251e457c5  (L3, 2025-10-14, sha e251e457c584, PR #26601)
TITLE: [Log] Optimize Startup Log (#26601)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.cutlass_v1_backend
FILES: vllm/v1/attention/backends/mla/cutlass_mla.py (+1/-1); vllm/distributed/device_communicators/cuda_communicator.py (+7/-6); vllm/distributed/device_communicators/pynccl.py (+1/-2); vllm/utils/__init__.py (+1/-1); vllm/v1/engine/core.py (+7/-5); vllm/v1/worker/gpu_model_runner.py (+0/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ 1. remove duplicate logs when using multi-cards ⏎ 2. debug or delete some useless logs

### L3-ea97940d6c  (L3, 2025-10-14, sha ea97940d6c2a, PR #24864)
TITLE: [DCP] Support Decode Context Parallel (DCP) for GQA with FlashAttention (#24864)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/ops/common.py (+9/-1); vllm/config/model.py (+17/-0); vllm/v1/attention/backends/flash_attn.py (+172/-30); vllm/v1/attention/backends/utils.py (+1/-0); vllm/v1/worker/gpu_model_runner.py (+1/-0); tests/distributed/test_context_parallel.py (+5/-1); tests/models/registry.py (+4/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎ This PR adds Decode Context Parallel (DCP) support for GQA following PR https://github.com/vllm-project/vllm/pull/23734. Current implementation based on FlashAttention.  ⏎  ⏎ Unlike MLA inference,  GQA (with FlashAttention) does not distinguish between prefill and decode during forward pass. To support DCP, this PR separately computes the attention scores for the context and query KV within a sequence and then merges the results. ⏎ ```text   …[truncated]

### L3-82af928c41  (L3, 2025-10-14, sha 82af928c4188, PR #26541)
TITLE: [Attention][Spec Decode] FlashMLA spec decode support (#26541)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.flashattn, L3.mla.flashinfer
FILES: vllm/v1/attention/backends/mla/common.py (+37/-12); vllm/v1/attention/backends/mla/flashattn_mla.py (+3/-2); vllm/v1/attention/backends/mla/flashinfer_mla.py (+2/-4); vllm/v1/attention/backends/mla/flashmla.py (+16/-2); tests/v1/attention/test_mla_backends.py (+156/-71)
LABELS: ready, v1
BODY: ## Purpose ⏎ This PR implements speculative decoding support for the FlashMLA backend. ⏎  ⏎ **NOTE**: the comment about the intermittent test failure was true prior to this PR, I just made a note of it. ⏎  ⏎ cc @LucasWilkinson  ⏎  ⏎ ## Test Plan ⏎ `pytest tests/v1/attention/test_mla_backends.py` ⏎  ⏎ ## Test Result ⏎ Passes ⏎  ⏎ --- ⏎ [details omitted]

### L3-2dcd12d357  (L3, 2025-10-14, sha 2dcd12d3571b, PR #26116)
TITLE: [torch.compile] Fix tests for torch==2.9 inductor partition (#26116)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+0/-6); tests/compile/piecewise/test_full_cudagraph.py (+21/-8); tests/compile/piecewise/test_multiple_graphs.py (+27/-11); tests/compile/piecewise/test_toy_llama.py (+74/-43); tests/compile/silly_attention.py (+0/-1); tests/compile/test_decorator.py (+3/-0); vllm/compilation/partition_rules.py (+11/-2); vllm/config/compilation.py (+2/-1)
LABELS: documentation, rocm, ready, torch.compile, ci/build, llama
BODY: ## Purpose ⏎ Fix compilation tests for Inductor graph partitioning. Also remove a noisy warning and remove `cudagraph_unsafe` tags from attention ops to allow proper behavior with empty splitting ops. ⏎  ⏎ To work this still requires #26735 and a spawn workaround. But for current main this should not change any behavior. ⏎  ⏎ Remaining issues: ⏎ - nested fused_moe compilation ⏎ - test_full_cudagraph.py fails for one test case? ⏎ - test_multiple_graphs.py fails ⏎ - …[truncated]

### L3-a86b4c58e8  (L3, 2025-10-14, sha a86b4c58e8f7, PR #26680)
TITLE: remove attn output view kernel (#26680)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v1_backend, L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.rocm.aiter_unified, L3.flex_attention, L3.tree_attention
FILES: vllm/attention/layer.py (+3/-3); vllm/v1/attention/backends/flash_attn.py (+1/-1); vllm/v1/attention/backends/flashinfer.py (+1/-1); vllm/v1/attention/backends/flex_attention.py (+1/-1); vllm/v1/attention/backends/rocm_aiter_fa.py (+1/-1); vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+1/-1); vllm/v1/attention/backends/rocm_attn.py (+1/-1); vllm/v1/attention/backends/tree_attn.py (+1/-1); vllm/v1/attention/backends/triton_attn.py (+1/-1); vllm/v1/attention/backends/xformers.py (+1/-1)
LABELS: ready, v1
ISSUES: #22293 [Feature]: Optimize RoPE
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: Before this PR, attention output is allocated and initialized with 0 (due to `torch.zeros`), and the view into a shape, before the output tensor is used by any other ops. This becomes a triton kernel of ~1 us latency, which is on-par with a rope/layer norm (~1.6us) latency. ⏎  ⏎ This PR changes to allocate with `torch.empty` which only allocates the tensor and does not initialize it. This allocation will be removed by cudagraph so it is free. ⏎  ⏎ As a r …[truncated]

### L3-579d2e5458  (L3, 2025-10-14, sha 579d2e5458b1, PR #26836)
TITLE: [WideEP][P/D] Add usage stats for DP+EP and KV Connector (#26836)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/utils.py (+14/-1)
LABELS: ready, v1, kv-connector
BODY: ## Purpose ⏎  ⏎ Add usage stats for disaggregated serving + more distributed serving parameters. Lets us see how vLLM is parallelized, what All2All implementation is being used, and what KV connectors are in use. ⏎  ⏎ Example output from running: ⏎ ``` ⏎ vllm serve Qwen/Qwen3-30B-A3B-FP8 --port 8192 -dp 4 --enable-expert-parallel \ ⏎         --kv-transfer-config.kv_connector OffloadingConnector \ ⏎         --kv-transfer-config.kv_role kv_both \ ⏎         --kv-tran …[truncated]

### L3-302ef403a2  (L3, 2025-10-15, sha 302ef403a230, PR #26656)
TITLE: [DSA][MLA] Tiny refactor on DeepSeek to make it reusable for different backends (#26656)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+2/-0); vllm/model_executor/models/deepseek_mtp.py (+8/-2); vllm/model_executor/models/deepseek_v2.py (+2/-1)
LABELS: ready, v1, deepseek
BODY: ## Purpose ⏎  ⏎ Tiny refactor on DeepSeek to make it reusable for different backends ⏎   * allow MLAAttention to accept variable length parameter lists ⏎   ~~* add `enable_dsa_topk_indices_buffer` in `AttentionBackend` to flexibly determine whether to create topk indices buffer~~ ⏎   * remove cuda hard code ⏎   ⏎ ## Test Plan ⏎ Test pass with DeepSeek-V3.2-Exp ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-e66d787bce  (L3, 2025-10-15, sha e66d787bce22, PR #26859)
TITLE: Disable FlashInfer sampler by default (#26859)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/sample/ops/topk_topp_sampler.py (+6/-14)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ There have been increasing reports of correctness issues or IMA with FlashInfer's top-p & top-k sampling kernel (see https://github.com/vllm-project/vllm/issues/26480#issuecomment-3390003242). For instance, it seems it can generates the same output even when the temperature is quite high (even though the seed is not set). vLLM generates different results (expectedly) once the kernel is disabled. ⏎  ⏎ Since flashinfer-python is a default d …[truncated]

### L3-e471d7ca7e  (L3, 2025-10-15, sha e471d7ca7ee8, PR #26773)
TITLE: [CI/Build][Bugfix] fix qutlass cmake error when set QUTLASS_SRC_DIR (#26773)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: cmake/external_projects/qutlass.cmake (+2/-2)
LABELS: bug, ready, ci/build
BODY: ## Purpose ⏎  ⏎ When building from source with QUTLASS_SRC_DIR specified, the current code encounters the following error: ⏎  ⏎ ``` ⏎   CMake Error at cmake/external_projects/qutlass.cmake:30 (message): ⏎     [QUTLASS] source directory could not be resolved. ⏎ ``` ⏎  ⏎ The cause is that when specifying QUTLASS_SRC_DIR, only FetchContent_Declare is executed, but FetchContent_Populate is omitted. This results in qutlass_SOURCE_DIR remaining undefined.  ⏎  ⏎ This PR fixe …[truncated]

### L3-5210dc3940  (L3, 2025-10-15, sha 5210dc3940b0, PR #26853)
TITLE: [Misc] Update TritonLanguagePlaceholder to have attributes that are used by Flash Linear Attention ops. (#26853)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/triton_utils/importing.py (+3/-0)
LABELS: ready
BODY: Summary: ⏎ Update TritonLanguagePlaceholder to have attributes that are used by Flash Linear Attention ops. ⏎  ⏎ Differential Revision: D84470467
