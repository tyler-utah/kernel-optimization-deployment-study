### L3-ec10fd0abc  (L3, 2025-10-09, sha ec10fd0abcc7, PR #16601)
TITLE: [Bugfix] Move current_platform import to avoid python import cache. (#16601)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.dispatch.selector; Attention selector imports current_platform lazily to avoid plugin cache errors.
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/selector.py (+2/-1); tests/kernels/attention/test_attention_selector.py (+6/-6)
LABELS: ready
BODY: We are adding a new platform to vLLM using plugin mechanism. To test our platform, we compare it with the CPU platform within a single file.  ⏎ ``` ⏎ from vllm import LLM, SamplingParams ⏎  ⏎ import os ⏎ from vllm.platforms import builtin_platform_plugins ⏎ from vllm.utils import resolve_obj_by_qualname ⏎ from vllm import platforms ⏎  ⏎ # Sample prompts. ⏎ prompts = [ ⏎     "Hello, my name is", ⏎ ] ⏎ # Create a sampling params object. ⏎ sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1) ⏎  ⏎ # Run on cpu platform ⏎ platforms._current_platform = resolve_obj_by_qualname( ⏎     "vllm.platforms.cpu.CpuPlatform" ⏎ )() ⏎ os.environ["VLLM_USE_V1"] = "0" ⏎ llm = LLM(model="facebook/opt-125m") ⏎ outputs = llm.generate(prompts, sampling_params) ⏎ print("\nGenerated Outputs:\n" + "-" * 60) ⏎  ⏎ # Run on our platform ⏎ platforms._current_platform = resolve_obj_by_qualname( ⏎     "vllm.platforms.Ourplatform" ⏎ )() ⏎  ⏎ os.environ["VLLM_USE_V1"] = "1" ⏎ llm = LLM(model="facebook/opt-125m") ⏎ outputs = llm.generate(prompts, sampling_params) ⏎  ⏎ ``` ⏎ However, due to Python’s import caching mechanism, the runner of our platform does not re-import current_platform when getting the attention backend for our platform. ⏎ https://github.com/vllm-project/vllm/blob/1dd23386ecab7b7c50ea61b8ff37ca14d2dbc0f7/vllm/worker/cpu_model_runner.py#L466 ⏎ https://github.com/vllm-project/vllm/blob/1dd23386ecab7b7c50ea61b8ff37ca14d2dbc0f7/vllm/attention/selector.py#L13 ⏎ So the solution is to move the import of `current_platform` inside the `_cached_get_attn_backend` function.

### L3-c9d33c60dc  (L3, 2025-10-09, sha c9d33c60dcdc, PR #26443)
TITLE: [UX] Add FlashInfer as default CUDA dependency (#26443)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
STAGE1: integrate; artifacts=L3.flashinfer.utils_dependency,L3.flashinfer.v1_backend; FlashInfer becomes a default CUDA dependency and availability checks update.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: docker/Dockerfile (+8/-69); requirements/cuda.txt (+2/-0); setup.py (+1/-2); vllm/utils/flashinfer.py (+9/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ It seems that FlashInfer does not require `nvcc` to installed from source anymore since `flashinfer-python>=0.2.9`, so we can move it to be a default dependency! ⏎  ⏎ Obviously to have it JIT compile kernels it needs to have `nvcc` available, so we will currently add that condition to `has_flashinfer()`. This is more conservative than it needs to be, as if we have an AOT compiled wheel installed there is likely no need for `nvcc`, but we can improve this later. ⏎  ⏎ Also with flashinfer-python==0.4.0, we can now move to using flashinfer-cubin and flashinfer-jit-cache wheels instead of pre-compiling ourselves in the docker! ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ I used a uv python image to simulate installing flashinfer-python without any cuda toolkit available ⏎  ⏎ ``` ⏎ docker run -it --rm astral/uv:bookworm-slim bash ⏎ root@bbbb3b887d99:/# uv venv --python 3.12 ⏎ root@bbbb3b887d99:/# source .venv/bin/activate ⏎ (.venv) root@bbbb3b887d99:/# uv pip install flashinfer-python==0.2.8 ⏎ ... ⏎       FileNotFoundError: [Errno 2] No such file or directory: 'nvcc' ⏎  ⏎ (.venv) root@bbbb3b887d99:/# uv pip install flashinfer-python==0.2.9 ⏎ Installed 41 packages in 451ms ⏎  ⏎ (.venv) root@bbbb3b887d99:/# uv pip install flashinfer-python==0.3.1 ⏎ Installed 5 packages in 103ms ⏎ ``` ⏎  ⏎ --- ⏎ [details omitted]

### L3-44f633dba1  (L3, 2025-10-09, sha 44f633dba17f, PR #25674)
TITLE: [Flashinfer][gpt-oss] Support FP8-qkv Flashinfer TRTLLM Sinks Attention (#25674)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L3.flashinfer.trtllm_gen; FlashInfer TRTLLM attention gains FP8-QKV sinks support.
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+0/-5); tests/kernels/attention/test_flashinfer_trtllm_attention.py (+76/-41)
LABELS: ready, ci/build, v1, gpt-oss
PERF_LINES: kv_cache_dtype=fp8, 6.9% perf gain | Output token throughput (tok/s):         22427.23 | Output token throughput (tok/s):         20977.07
BODY: ## Purpose ⏎ Support FP8-qkv Flashinfer TRTLLM sinks attention. ⏎ Note: require flashinfer v0.4.0rc4(updating in #26326) ⏎ https://github.com/flashinfer-ai/flashinfer/pull/1758 ⏎  ⏎ ## Test Plan && Test Result ⏎ #### Kernel unit test: ⏎ `tests/kernels/attention/test_flashinfer_trtllm_attention.py` ⏎ ``` ⏎ ===== 224 passed, 16 skipped in 30.94s ==== ⏎ ``` ⏎ #### E2E accuracy: ⏎ kv_cache_dtype=fp8 ⏎ ``` ⏎ [{'eval_name': 'gpqa', 'model_name': 'gpt-oss-120b-high_temp1.0_20250925_081936', 'metric': 0.7904040404040404}] ⏎ [{'eval_name': 'aime25', 'model_name': 'gpt-oss-120b-high_temp1.0_20250925_084243', 'metric': 0.925}] ⏎ ``` ⏎ kv_cache_dtype=auto ⏎ ``` ⏎ [{'eval_name': 'gpqa', 'model_name': 'gpt-oss-120b-high_temp1.0_20250925_090839', 'metric': 0.7910353535353535}] ⏎ [{'eval_name': 'aime25', 'model_name': 'gpt-oss-120b-high_temp1.0_20250925_093344', 'metric': 0.9125}] ⏎ ``` ⏎ #### E2E perf: ⏎ kv_cache_dtype=fp8, 6.9% perf gain ⏎ ``` ⏎ Output token throughput (tok/s):         22427.23 ⏎ ``` ⏎ kv_cache_dtype=auto ⏎ ``` ⏎ Output token throughput (tok/s):         20977.07 ⏎ ``` ⏎ --- ⏎ [details omitted]

### L3-6e783bc54b  (L3, 2025-10-09, sha 6e783bc54b03, PR #26499)
TITLE: [Bugfix] Fix CUDA graph selection bug in FlashInfer at high concurrency (#26499)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
STAGE1: repair_correctness; artifacts=L3.flashinfer.v1_backend; FlashInfer CUDA graph selection uses decode token count under spec decode.
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+9/-2)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ Previously, FlashInfer would choose to use decode CUDA graphs if `num_decodes < threshold`, even if spec decoding was enabled and the number of tokens in the batch was much higher. This can cause issues when `num_decode_tokens > max_cudagraph_size`. ⏎  ⏎ I fixed the check to correctly use `num_decode_tokens` instead, and also updated the constructor to update the threshold to a higher value when speculative decoding is used.

### L3-e94cfd51da  (L3, 2025-10-10, sha e94cfd51da5f, PR #26564)
TITLE: [BUG] Qwen3-next MTP. Fix attn metadata build bug (#26564)
SOURCES: body_keyword
STAGE1: repair_correctness; artifacts=L3.dispatch.abstract_interface; Qwen3-Next MTP metadata builder selects full-attention metadata instead of GDN.
ARTIFACT_HINTS: -
FILES: vllm/v1/spec_decode/eagle.py (+6/-7)
LABELS: speculative-decoding, ready, v1, qwen
BODY: ## Purpose ⏎ After fixing #24486 Qwen3-next with FlashInfer full attn start working without MTP.  ⏎ But with MTP it fails.  ⏎  ⏎ The reason we choose incorrect attn metadata type for draft model (choose GDN instead of full attn).  ⏎  ⏎ Fix it.  ⏎  ⏎ ## Test Result ⏎ Qwen3-next with MTP works now.

### L3-6f0f570c43  (L3, 2025-10-10, sha 6f0f570c436c, PR #26559)
TITLE: [deepseek] kernel block size for UniformTypeKVCacheSpecs (#26559)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=NEW:mla_indexer; MLA indexer handles UniformTypeKVCacheSpecs kernel block size.
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+10/-2); vllm/v1/worker/gpu_model_runner.py (+9/-4)
LABELS: ready, v1, deepseek
ISSUES: #26524 [Bug]: prepare_kernel_block_sizes doesn't parse UniformTypeKVCacheSpecs
BODY: ## Purpose ⏎ After https://github.com/vllm-project/vllm/pull/24486 , deepseek 3.2 will throw this error: ⏎ ``` ⏎  NotImplementedError: unknown kv cache spec UniformTypeKVCacheSpecs ⏎  ``` ⏎ This PR fix it. ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/26524 ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ python3 examples/offline_inference/basic/generate.py --model deepseek-ai/DeepSeek-V3.2-Exp --gpu_memory_utilization 0.8 -tp 8 ⏎ ``` ⏎ ## Test Result ⏎  ⏎ ``` ⏎ -------------------------------------------------- ⏎ Prompt: 'Hello, my name is' ⏎ Generated text: ' Christian Munoz and\nthis is my final project for my summer\n2020' ⏎ -------------------------------------------------- ⏎ Prompt: 'The president of the United States is' ⏎ Generated text: ' the head of state and head of government of the United States, indirectly elected to' ⏎ -------------------------------------------------- ⏎ Prompt: 'The capital of France is' ⏎ Generated text: ' Paris, and the capital of Spain is Madrid.\n\n**Question:** The capital of' ⏎ -------------------------------------------------- ⏎ Prompt: 'The future of AI is' ⏎ Generated text: ' the future of work\n\nThe future of AI is the future of work\n\nThe' ⏎ -------------------------------------------------- ⏎ ``` ⏎  ⏎ --- ⏎ [details omitted]

### L3-0cd103e7cb  (L3, 2025-10-11, sha 0cd103e7cbf0, PR #26509)
TITLE: CP: make correct_attn_out robust to 4‑D views and fix Triton arg binding (#26509)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=NEW:context_parallel_attention_common; Common context-parallel attention output correction supports 4-D views.
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/common.py (+46/-8)
LABELS: ready
BODY: ## Purpose ⏎ The issue is found from @simon-mo where we want to bring up H100 resources for CI https://github.com/vllm-project/vllm/pull/26396 , and there is a failure https://buildkite.com/vllm/ci/builds/33954/steps/canvas?sid=0199c256-fc49-4e2d-afd6-c54961f0ffb0 with the error msg ⏎  ⏎ ``` ⏎ (Worker_TP0 pid=3232) ERROR 10-07 22:59:10 [multiproc_executor.py:706] TypeError: dynamic_func() got multiple values for argument 'HEAD_DIM' ⏎ ``` ⏎ After some investigation on both H100 and B200, I find this error is only showing up on H100 where `out` and `lses` can arrive as 4‑D views: ⏎ * out.stride() = (8192, 8192, 512, 1) ⇒ shape like [B, 1, H, D] ⏎ * lses.stride() = (128, 16, 1, 1) ⇒ shape like [N, B, H, 1] ⏎  ⏎ so the original impl unpacks `*out.stride()` and `*lses.stride()` blindly, which in this case yields 4 + 4 stride values instead of the 3 + 3 the kernel signature expects. Triton then binds one of those “extra” stride values positionally into the `HEAD_DIM` slot, and we also pass `HEAD_DIM` via `**const_args`, hence: ⏎ ``` ⏎ TypeError: dynamic_func() got multiple values for argument 'HEAD_DIM' ⏎ ``` ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest tests/distributed/test_context_parallel.py ⏎ export VLLM_ATTENTION_BACKEND=TRITON_MLA pytest tests/distributed/test_context_parallel.py ⏎ ``` ⏎ on both H100 and B200 ⏎  ⏎ ## Test Result ⏎ all 4 tests passed ⏎  ⏎ --- ⏎ [details omitted]

### L3-3263799056  (L3, 2025-10-13, sha 3263799056f4, PR #26373)
TITLE: [unrevert] Add batch invariant kernel override for FlashInfer backend [2/n] (#26373)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
STAGE1: reland; artifacts=L3.flashinfer.v1_backend; Re-lands the FlashInfer batch-invariant override after dependency bump.
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+29/-3); csrc/moe/topk_softmax_kernels.cu (+1/-3); tests/v1/generation/test_batch_invariance.py (+37/-28); vllm/model_executor/layers/batch_invariant.py (+14/-1)
LABELS: ready, ci/build, v1
BODY: This change reinstates already approved + landed #25769 based on the latest bump to flashinfer: ⏎  ⏎ https://github.com/vllm-project/vllm/pull/26326 ⏎  ⏎ It should *not* land before #26326

### L3-ea97940d6c  (L3, 2025-10-14, sha ea97940d6c2a, PR #24864)
TITLE: [DCP] Support Decode Context Parallel (DCP) for GQA with FlashAttention (#24864)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: extend_support; artifacts=L3.flash_attn.v1_backend,L3.dispatch.abstract_interface; FlashAttention backend adds DCP support for GQA.
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/ops/common.py (+9/-1); vllm/config/model.py (+17/-0); vllm/v1/attention/backends/flash_attn.py (+172/-30); vllm/v1/attention/backends/utils.py (+1/-0); vllm/v1/worker/gpu_model_runner.py (+1/-0); tests/distributed/test_context_parallel.py (+5/-1); tests/models/registry.py (+4/-1)
LABELS: ready, v1
PERF_LINES: [kv_cache_utils.py:868] Maximum concurrency for 262,144 tokens per request: 6.57x | Additionally, I think we can reduce latency by overlapping query computation with context communication.
BODY: ## Purpose ⏎ This PR adds Decode Context Parallel (DCP) support for GQA following PR https://github.com/vllm-project/vllm/pull/23734. Current implementation based on FlashAttention.  ⏎  ⏎ Unlike MLA inference,  GQA (with FlashAttention) does not distinguish between prefill and decode during forward pass. To support DCP, this PR separately computes the attention scores for the context and query KV within a sequence and then merges the results. ⏎ ```text   ⏎   # |- tokenA -|......................|-- newTokens ---| ⏎   # |---------- context_len ----------|-- query_len ---| ⏎ ``` ⏎ - For the query, no collective communication is required among the DCP group.  ⏎ - For context, the KV is distributed across different DCP ranks. This PR follows the DCP decode approach from MLA, i.e., all-gathering Q and lse, then correcting the attn out before performing reduce-scatter. ⏎  ⏎ ## Test Plan ⏎ Qwen/Qwen3-235B-A22B-Instruct-2507 ⏎ ```shell ⏎ vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507 --gpu-memory-utilization 0.9 --tensor-parallel-size 8 --decode-context-parallel-size 2 ⏎ ``` ⏎  ⏎ ## Test Result ⏎ - **KV Cache Size** ⏎ ```shell ⏎ [kv_cache_utils.py:859] Multiplying the GPU KV cache size by the dcp_world_size 2. ⏎ [kv_cache_utils.py:864] GPU KV cache size: 1,723,584 tokens ⏎ [kv_cache_utils.py:868] Maximum concurrency for 262,144 tokens per request: 6.57x ⏎ ``` ⏎ - **gsm8k eval** ⏎ ```text ⏎ TP8 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8836|±  |0.0062| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.8628|±  |0.0067| ⏎  ⏎ TP8DCP2 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8988|±  |0.0059| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.8870|±  |0.0062| ⏎ ``` ⏎  ⏎ Additionally, I think we can reduce latency by overlapping query computation with context communication. ⏎  ⏎ --- ⏎ [details omitted]

### L3-82af928c41  (L3, 2025-10-14, sha 82af928c4188, PR #26541)
TITLE: [Attention][Spec Decode] FlashMLA spec decode support (#26541)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
STAGE1: extend_support; artifacts=L3.mla.flashmla_v1_adapter,L3.mla.common_v1; FlashMLA backend and MLA common metadata add speculative decoding support.
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.flashattn, L3.mla.flashinfer
FILES: vllm/v1/attention/backends/mla/common.py (+37/-12); vllm/v1/attention/backends/mla/flashattn_mla.py (+3/-2); vllm/v1/attention/backends/mla/flashinfer_mla.py (+2/-4); vllm/v1/attention/backends/mla/flashmla.py (+16/-2); tests/v1/attention/test_mla_backends.py (+156/-71)
LABELS: ready, v1
BODY: ## Purpose ⏎ This PR implements speculative decoding support for the FlashMLA backend. ⏎  ⏎ **NOTE**: the comment about the intermittent test failure was true prior to this PR, I just made a note of it. ⏎  ⏎ cc @LucasWilkinson  ⏎  ⏎ ## Test Plan ⏎ `pytest tests/v1/attention/test_mla_backends.py` ⏎  ⏎ ## Test Result ⏎ Passes ⏎  ⏎ --- ⏎ [details omitted]

### L3-a86b4c58e8  (L3, 2025-10-14, sha a86b4c58e8f7, PR #26680)
TITLE: remove attn output view kernel (#26680)
SOURCES: path_core
STAGE1: optimize; artifacts=L3.dispatch.abstract_interface,L3.flash_attn.v1_backend,L3.flashinfer.v1_backend,L3.triton.v1_backend,L3.flex_attention; Attention output allocation switches from zeros plus view to empty.
ARTIFACT_HINTS: L3.xformers.v1_backend, L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.rocm.aiter_unified, L3.flex_attention, L3.tree_attention
FILES: vllm/attention/layer.py (+3/-3); vllm/v1/attention/backends/flash_attn.py (+1/-1); vllm/v1/attention/backends/flashinfer.py (+1/-1); vllm/v1/attention/backends/flex_attention.py (+1/-1); vllm/v1/attention/backends/rocm_aiter_fa.py (+1/-1); vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+1/-1); vllm/v1/attention/backends/rocm_attn.py (+1/-1); vllm/v1/attention/backends/tree_attn.py (+1/-1); vllm/v1/attention/backends/triton_attn.py (+1/-1); vllm/v1/attention/backends/xformers.py (+1/-1)
LABELS: ready, v1
ISSUES: #22293 [Feature]: Optimize RoPE
DEEP_STUDY: deep-study performance PR (kernel_optimization)
PERF_LINES: Before this PR, attention output is allocated and initialized with 0 (due to `torch.zeros`), and the view into a shape, before the output tensor is used by any 
BODY: Before this PR, attention output is allocated and initialized with 0 (due to `torch.zeros`), and the view into a shape, before the output tensor is used by any other ops. This becomes a triton kernel of ~1 us latency, which is on-par with a rope/layer norm (~1.6us) latency. ⏎  ⏎ This PR changes to allocate with `torch.empty` which only allocates the tensor and does not initialize it. This allocation will be removed by cudagraph so it is free. ⏎  ⏎ As a result, this PR removes the attn_out_view kernel at the end of this qwen3-0.6b trace. ⏎ <img width="1740" height="188" alt="image" src="https://github.com/user-attachments/assets/729cae3a-69e7-4788-a093-76f5de64d745" /> ⏎  ⏎ See https://github.com/vllm-project/vllm/pull/26682#issue-3508480559 for perf win.

### L3-302ef403a2  (L3, 2025-10-15, sha 302ef403a230, PR #26656)
TITLE: [DSA][MLA] Tiny refactor on DeepSeek to make it reusable for different backends (#26656)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: adapt_framework; artifacts=L3.dispatch.abstract_interface,L3.mla.common_v1; DeepSeek refactor lets MLAAttention accept variable-length parameters.
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+2/-0); vllm/model_executor/models/deepseek_mtp.py (+8/-2); vllm/model_executor/models/deepseek_v2.py (+2/-1)
LABELS: ready, v1, deepseek
BODY: ## Purpose ⏎  ⏎ Tiny refactor on DeepSeek to make it reusable for different backends ⏎   * allow MLAAttention to accept variable length parameter lists ⏎   ~~* add `enable_dsa_topk_indices_buffer` in `AttentionBackend` to flexibly determine whether to create topk indices buffer~~ ⏎   * remove cuda hard code ⏎   ⏎ ## Test Plan ⏎ Test pass with DeepSeek-V3.2-Exp ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-0a9ef0cfce  (L3, 2025-10-15, sha 0a9ef0cfce13, PR #26534)
TITLE: Move query quantization to attention layer for Flashinfer & Triton. (#26534)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
STAGE1: adapt_framework; artifacts=L3.flashinfer.v1_backend,L3.triton.v1_backend,L3.dispatch.abstract_interface; Moves query quantization into attention layer and updates FlashInfer/Triton backend interfaces.
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+16/-8); vllm/attention/layer.py (+6/-3); vllm/v1/attention/backends/flash_attn.py (+3/-1); vllm/v1/attention/backends/flashinfer.py (+12/-10); vllm/v1/attention/backends/triton_attn.py (+3/-15); tests/compile/test_fusion_attn.py (+3/-1)
LABELS: ready, v1
PERF_LINES: Request throughput (req/s):              1.19 | Output token throughput (tok/s):         237.97 | Peak output token throughput (tok/s):    240.00 | Total Token throughput (tok/s):          1323.60 | Mean TTFT (ms):                          20.13 | Median TTFT (ms):                        19.97 | P99 TTFT (ms):                           23.20 | Mean TPOT (ms):                          4.12 | Median
BODY: ### Purpose ⏎ Implements refactor of quantization to the attention layer for triton and flashinfer, resolves feature request [#25584](https://github.com/vllm-project/vllm/issues/25584) ⏎  ⏎ ### Test Plan ⏎ Spin up server: ⏎  ⏎ Flashinfer: ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=FLASHINFER  vllm serve meta-llama/Llama-3.1-8B-Instruct \ ⏎   --kv-cache-dtype fp8 \ ⏎   --compilation-config '{"compile_sizes": [1,2,4,8], "cudagraph_capture_sizes": [1,2,4,8], "cudagraph_mode": "FULL_AND_PIECEWISE"}' \ ⏎   --no-enable-prefix-caching ⏎ ``` ⏎    ⏎ Triton: ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm serve meta-llama/Llama-3.1-8B-Instruct \ ⏎   --kv-cache-dtype fp8 \ ⏎   --compilation-config '{"compile_sizes": [1,2,4,8], "cudagraph_capture_sizes": [1,2,4,8], "cudagraph_mode": "FULL_AND_PIECEWISE"}' \ ⏎   --no-enable-prefix-caching ⏎ ``` ⏎  ⏎ Benchmark: ⏎ ``` ⏎ vllm bench serve \ ⏎     --backend vllm \ ⏎     --model meta-llama/Llama-3.1-8B-Instruct  \ ⏎     --dataset-name sonnet \ ⏎     --dataset-path vllm/benchmarks/sonnet.txt \ ⏎     --sonnet-input-len 1000 \ ⏎     --sonnet-output-len 200 \ ⏎     --port 8000 \ ⏎     --num-prompts 20 \ ⏎     --max-concurrency 1 ⏎ ``` ⏎ ### Accuracy ⏎ To ensure there is no accidental accuracy degradation we also run the following for Flashinfer & Triton with kv_cache_dtype in {auto,fp8} both on this PR and on mainline. We also run without enforce_eager=True for the FP8 variants ⏎ ``` ⏎ lm_eval \ ⏎   --model vllm \ ⏎   --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,kv_cache_dtype=auto,tensor_parallel_size=1,enforce_eager=True \ ⏎   --tasks gsm8k \ ⏎   --batch_size  ⏎ ``` ⏎  ⏎ ### Test Results ⏎ ``` ⏎ PR + FlashInfer ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     20         ⏎ Maximum request concurrency:             1          ⏎ Benchmark duration (s):                  16.81      ⏎ Total input tokens:                      18248      ⏎ Total generated tokens:                  4000       ⏎ Request throughput (req/s):              1.19       ⏎ Output token throughput (tok/s):         237.97     ⏎ Peak output token throughput (tok/s):    240.00     ⏎ Peak concurrent requests:                3.00       ⏎ Total Token throughput (tok/s):          1323.60    ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):                          20.13      ⏎ Median TTFT (ms):                        19.97      ⏎ P99 TTFT (ms):                           23.20      ⏎ -----Time per Output Token (excl. 1st token)------ ⏎ Mean TPOT (ms):                          4.12       ⏎ Median TPOT (ms):                        4.13       ⏎ P99 TPOT (ms):                           4.15       ⏎ ---------------Inter-token Latency------ …[truncated]

### L3-7d8975de84  (L3, 2025-10-15, sha 7d8975de84da, PR #26609)
TITLE: Deepseek-v3 Batch Invariant on 8xH100 (#26609)
SOURCES: path_core, body_keyword
STAGE1: extend_support; artifacts=L3.flash_attn.v1_backend,L3.mla.common_v1,L3.mla.flashmla_v1_adapter,L3.mla.flashattn; Adds batch-invariant execution support across FlashAttention and MLA attention backends.
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.flashattn
FILES: vllm/model_executor/layers/mla.py (+1/-0); vllm/v1/attention/backends/flash_attn.py (+12/-1); vllm/v1/attention/backends/mla/common.py (+10/-1); vllm/v1/attention/backends/mla/flashattn_mla.py (+11/-1); vllm/v1/attention/backends/mla/flashmla.py (+36/-2); vllm/v1/attention/backends/mla/triton_mla.py (+6/-1); tests/v1/generation/test_batch_invariance.py (+769/-73); tests/v1/generation/test_rms_norm_batch_invariant.py (+346/-0); vllm/compilation/caching.py (+3/-1); vllm/config/model.py (+7/-0); vllm/config/parallel.py (+7/-1); vllm/distributed/device_communicators/all_reduce_utils.py (+6/-0); vllm/distributed/device_communicators/symm_mem.py (+5/-0); vllm/engine/arg_utils.py (+1/-1); vllm/model_executor/layers/batch_invariant.py (+240/-13); vllm/model_executor/layers/fused_moe/fused_moe.py (+28/-4); vllm/model_executor/layers/layernorm.py (+10/-0); vllm/model_executor/layers/quantization/fp8.py (+65/-0); vllm/model_executor/models/gpt_oss.py (+1/-0); vllm/v1/worker/gpu_model_runner.py (+0/-3); vllm/v1/worker/gpu_worker.py (+3/-0)
LABELS: ready, ci/build, v1, deepseek, gpt-oss
BODY: This PR replaces https://github.com/vllm-project/vllm/pull/26136 with a far more rigorous set of tests and implementation choices.  It is, somewhat unfortunately, quite big.  I will be trying to make this smaller over the weekend but would appreciate some initial eyes!  (also I'll do a writeup because this was quite a journey to get working and I think folks would benefit from that) ⏎  ⏎ Adds support for FLASH_ATTN, rms_norm, batched matmul, linear, fused_moe (the actual triton impl, not the native one), FLASH_ATTN_MLA, TRITON_MLA, allreduce (on NCCL, not the custom all reduce). ⏎  ⏎ It also attemps to configure *all* relevant flags across the stack (including env variables) so users don't need to specify things like "disable_custom_ar" and "enforce_eager" ⏎  ⏎ ## Purpose ⏎  ⏎ Fully support Deepseek-v3 Batch Invariance on 8xH100s.  This has large impact on mainstream models (including things like full multi-gpu support for Qwen30b-3a). ⏎  ⏎ ## Test Plan ⏎  ⏎ The biggest test is this: ⏎  ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=FLASH_ATTN_MLA VLLM_TEST_TP_SIZE=8 VLLM_TEST_MODEL="deepseek-ai/DeepSeek-V3" VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 pytest -s -v tests/v1/generation/test_batch_invariance.py -k test_logprobs_bitwise_batch_invariance_bs1_vs_bsN[FLASH_ATTN] ⏎ ``` ⏎ Which runs hundreds of queries individually and in batched form, achieving exact bitwise alignment across every generated token (including sampling with temp=0.6) ⏎  ⏎ ## Test Result ⏎  ⏎ Pass. ⏎  ⏎ --- ⏎ [details omitted]

### L3-314fa8abbf  (L3, 2025-10-16, sha 314fa8abbf9d, PR #26846)
TITLE: [Attention] Tune CUTLASS MLA num_splits (#26846)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
STAGE1: retune; artifacts=L3.mla.cutlass_sm100; Retunes CUTLASS MLA num_splits heuristic in SM100 kernel header for speed.
ARTIFACT_HINTS: L3.mla.cutlass_sm100
FILES: csrc/attention/mla/cutlass_sm100_mla/device/sm100_mla.hpp (+21/-16)
LABELS: ready
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
PERF_LINES: Tune the num_splits heuristic for CUTLASS_MLA to achieve some speedup now that #26026 has fixed the hang. Based on experiments performed using the tools introdu | Following the optimal policy would yield this speedup: | <img width="3331" height="2363" alt="numsplits_speedup" src="https://github.com/user-attachments/assets/06f76c49-d4cf-4817-9bc0-964f919ab0e7" /> | This results in the following spe
BODY: ## Purpose ⏎ Tune the num_splits heuristic for CUTLASS_MLA to achieve some speedup now that #26026 has fixed the hang. Based on experiments performed using the tools introduced in #26835, this is the optimal num_splits policy: ⏎  ⏎ <img width="3331" height="2363" alt="numsplits_heatmap" src="https://github.com/user-attachments/assets/c2898b4c-2ff2-4965-a760-9bc71446e945" /> ⏎  ⏎ Following the optimal policy would yield this speedup: ⏎  ⏎ <img width="3331" height="2363" alt="numsplits_speedup" src="https://github.com/user-attachments/assets/06f76c49-d4cf-4817-9bc0-964f919ab0e7" /> ⏎  ⏎ As a simpler alternative, we implement a heuristic yielding the following policy: ⏎  ⏎ <img width="3335" height="2364" alt="numsplits_policy_batch-based_improved" src="https://github.com/user-attachments/assets/f5834e32-d0b8-48ca-afd2-f06ff508e694" /> ⏎  ⏎ This results in the following speedup: ⏎  ⏎ <img width="3331" height="2364" alt="numsplits_speedup_batch-based_improved" src="https://github.com/user-attachments/assets/71af1582-09ac-4f45-baeb-799b7002050c" /> ⏎  ⏎ --- ⏎ [details omitted]

### L3-5afd3276df  (L3, 2025-10-16, sha 5afd3276dfd7, PR #26870)
TITLE: [Feature] Add process_weights_after_loading to AttentionImpl (#26870)
SOURCES: path_core, symbol_pickaxe, body_keyword
STAGE1: adapt_framework; artifacts=L3.flashinfer.v1_backend,L3.dispatch.abstract_interface; Adds process_weights_after_loading hook to AttentionImpl and FlashInfer implementation contract.
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+3/-0); vllm/attention/layer.py (+1/-10); vllm/v1/attention/backends/flashinfer.py (+5/-0)
LABELS: ready, v1
ISSUES: #26817 [Feature]: Add process_weights_after_loading to AttentionImpl
BODY: ## Purpose ⏎  ⏎ FIX: https://github.com/vllm-project/vllm/issues/26817 ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ $ VLLM_ATTENTION_BACKEND=FLASHINFER python3-m vllm.entrypoints.cli.main serve Qwen/Qwen3-0.6B ⏎ ``` ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-b2f78cbad4  (L3, 2025-10-16, sha b2f78cbad4e8, PR #26855)
TITLE: [small][batch invariance] Rename the env and internal flags to simplify usage (#26855)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.flash_attn.v1_backend,L3.flashinfer.v1_backend,L3.mla.common_v1,L3.mla.flashattn,L3.flex_attention; Renames batch-invariant environment and internal flags consumed by attention backends.
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.flashattn, L3.flex_attention
FILES: vllm/v1/attention/backends/flash_attn.py (+5/-5); vllm/v1/attention/backends/flashinfer.py (+3/-3); vllm/v1/attention/backends/flex_attention.py (+2/-2); vllm/v1/attention/backends/mla/common.py (+2/-2); vllm/v1/attention/backends/mla/flashattn_mla.py (+3/-3); vllm/v1/attention/backends/mla/flashmla.py (+2/-2); vllm/v1/attention/backends/mla/triton_mla.py (+2/-2); csrc/core/batch_invariant.hpp (+4/-4); csrc/layernorm_kernels.cu (+2/-2); csrc/layernorm_quant_kernels.cu (+1/-1); tests/v1/e2e/test_async_sched_and_preempt.py (+1/-1); tests/v1/generation/test_batch_invariance.py (+12/-12); vllm/config/model.py (+2/-2); vllm/config/parallel.py (+2/-2); vllm/distributed/device_communicators/all_reduce_utils.py (+2/-2); vllm/distributed/device_communicators/symm_mem.py (+2/-2); vllm/model_executor/layers/batch_invariant.py (+3/-3); vllm/model_executor/layers/fused_moe/fused_moe.py (+5/-5); vllm/model_executor/layers/layernorm.py (+3/-3); vllm/model_executor/layers/quantization/fp8.py (+3/-3)
LABELS: ready, v1, gpt-oss
BODY: Environment variable VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT  -> VLLM_BATCH_INVARIANT ⏎ Internal flag: vllm_kernel_override_batch_invariant() -> vllm_is_batch_invariant() ⏎  ⏎ ## Purpose ⏎  ⏎ Simplify usage across the stack. ⏎  ⏎ ## Test Plan ⏎  ⏎ Rebuild for the C++ component: ⏎ ``` ⏎ CCACHE_NOHASHDIR="true"  uv pip install -e . --no-build-isolation -v ⏎ ``` ⏎  ⏎ `pytest -s -v tests/v1/generation/test_batch_invariance.py -k test_logprobs_bitwise_batch_invariance_bs1_vs_bsN` ⏎  ⏎ ## Test Result ⏎  ⏎ Pass. ⏎  ⏎ --- ⏎ [details omitted]

### L3-950cf9e58e  (L3, 2025-10-17, sha 950cf9e58eef, PR #27114)
TITLE: [Bugfix] Use PIECEWISE cudagraphs on Blackwell if max_model_len > 131072 (#27114)
SOURCES: body_keyword
STAGE1: repair_correctness; artifacts=L3.flashinfer.trtllm_gen; Changes CUDA graph mode on Blackwell long contexts to avoid TRTLLM attention accuracy failures.
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+37/-15)
LABELS: bug, ready
ISSUES: #27057 [Bug]: Qwen3-VL broken on Blackwell with `PIECEWISE_AND_FULL`
PERF_LINES: The original issue was found because Qwen3-VL models completely lost accuracy (1% vs 86% on GSM8K) on B200 GPUs when using the default FULL_AND_PIECEWISE cudagr | Evaluating: 100%|████████████████████████████████████████| 1319/1319 [01:04<00:00, 20.56it/s] | Total latency: 64.173 s | Evaluating: 100%|███████████████████████████████████████| 1319/1319 [00:11<00:00, 118.01it/s] | Total latency: 11.1
BODY: ## Purpose ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/27057 ⏎  ⏎ The original issue was found because Qwen3-VL models completely lost accuracy (1% vs 86% on GSM8K) on B200 GPUs when using the default FULL_AND_PIECEWISE cudagraph_mode. The issue did not occur on Hopper at all, with PIECEWISE mode only, FlashAttention backend, or when explicitly disabling TRTLLM attention. ⏎  ⏎ Because TRTLLM attention is selected dynamically based on runtime conditions `(num_tokens, max_seq_len, kv_cache_dtype)`. During FULL CG capture, the `max_seq_len` is used which when greater than 128K results in FlashInfer being selected, but during actual inference without using full context length, the same conditions triggered TRTLLM selection. This created a graph/runtime mismatch where captured graphs referenced FlashInfer kernels but runtime attempted to execute TRTLLM kernels, producing incorrect results. **I was able to see this behavior on any model with default `max_model_len>128K`** ⏎  ⏎ By enforcing PIECEWISE mode in this PR to disable cuda graph capture of attention, we can avoid this issue of dynamism. In the future we should see if we can made TRTLLM support larger context lengths to support FULL graphs ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ Reproduction on B200 on `main`: ⏎  ⏎ ``` ⏎ vllm serve gradientai/Llama-3-8B-Instruct-Gradient-1048k ⏎ python tests/evals/gsm8k/gsm8k_eval.py ⏎  ⏎ Running GSM8K evaluation: 1319 questions, 5-shot ⏎ Evaluating: 100%|████████████████████████████████████████| 1319/1319 [01:04<00:00, 20.56it/s] ⏎  ⏎ Results: ⏎ Accuracy: 0.006 ⏎ Invalid responses: 0.775 ⏎ Total latency: 64.173 s ⏎ Questions per second: 20.554 ⏎ ``` ⏎  ⏎ ``` ⏎ vllm serve gradientai/Llama-3-8B-Instruct-Gradient-1048k --max-model-len=100K ⏎ python tests/evals/gsm8k/gsm8k_eval.py ⏎  ⏎ Running GSM8K evaluation: 1319 questions, 5-shot ⏎ Evaluating: 100%|███████████████████████████████████████| 1319/1319 [00:11<00:00, 118.01it/s] ⏎  ⏎ Results: ⏎ Accuracy: 0.579 ⏎ Invalid responses: 0.004 ⏎ Total latency: 11.190 s ⏎ Questions per second: 117.872 ⏎ ``` ⏎  ⏎ Running on this PR: ⏎  ⏎ ``` ⏎ vllm serve gradientai/Llama-3-8B-Instruct-Gradient-1048k ⏎ (APIServer pid=2588191) WARNING 10-17 13:50:11 [vllm.py:385] NVIDIA Blackwell TRTLLM attention cannot support max_model_len >= 131072 (found 1048576), causing dynamic dispatching that breaks full cudagraphs. Overriding cudagraph_mode to PIECEWISE. ⏎ python tests/evals/gsm8k/gsm8k_eval.py ⏎  ⏎ Running GSM8K evaluation: 1319 questions, 5-shot ⏎ Evaluating: 100%|███████████████████████████████████████| 1319/1319 [00:12<00:00, 107.95it/s] ⏎  ⏎ Results: ⏎ Accuracy: 0.578 ⏎ Invalid responses: 0.004 ⏎ Total latency: 12.233 s ⏎ Questions per sec …[truncated]

### L3-9f020f4f31  (L3, 2025-10-18, sha 9f020f4f3109, PR #27111)
TITLE: [BugFix] Fix failing gemma-3-1b-it test: `test_lm_eval_accuracy_v1_engine[google/gemma-3-1b-it]` (#27111)
SOURCES: path_core, dependency_pin
STAGE1: repair_correctness; artifacts=L3.flash_attn.fork_build; Bumps vllm-flash-attn tag to pick up accuracy fix for Gemma attention.
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build
BODY: vLLM side of: https://github.com/vllm-project/flash-attention/pull/102 ⏎  ⏎ Fix `pytest tests/entrypoints/llm/test_accuracy.py::test_lm_eval_accuracy_v1_engine[google/gemma-3-1b-it]` ⏎  ⏎ Now passes

### L3-b26b70bec4  (L3, 2025-10-18, sha b26b70bec4fa, PR #26587)
TITLE: [Misc] Refactor `get_kv_cache_spec` into `AttentionLayerBase` (#26587)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.dispatch.abstract_interface; Moves get_kv_cache_spec into attention-layer interface, changing cache-spec contract.
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+53/-5); vllm/attention/layers/chunked_local_attention.py (+13/-0); vllm/attention/layers/cross_attention.py (+9/-1); vllm/attention/layers/encoder_only_attention.py (+6/-0); vllm/model_executor/layers/attention_layer_base.py (+11/-0); vllm/model_executor/layers/mamba/abstract.py (+29/-0); vllm/model_executor/models/deepseek_v2.py (+1/-1); vllm/utils/__init__.py (+9/-0); vllm/v1/spec_decode/eagle.py (+1/-1); vllm/v1/worker/gpu_model_runner.py (+19/-110)
LABELS: speculative-decoding, ready, v1, deepseek
BODY: This PR modifies the `AttentionLayerBase` interface to add a new `get_kv_cache_spec` method. ⏎ This allows different attention layers to define their own KV Cache spec, by making the spec entirely transparent to the Model Runner. ⏎  ⏎ As a consequence, the runner can now limit itself to collect the specs without having to handle different attention types and/or model-specific hacks such as the one for DSv32 Indexer.  ⏎ It also makes the code much simpler as all ENCODER,ENCODER_ONLY and ENCODER_DECODER type management is moved to a method dispatch system. ⏎  ⏎ cc @heheda12345 who clearly defined the task ⏎   ⏎ PS this used to be a TODO in code from @LucasWilkinson https://github.com/vllm-project/vllm/blob/releases/v0.11.0/vllm/v1/worker/gpu_model_runner.py#L4065

### L3-250fb1b8ea  (L3, 2025-10-21, sha 250fb1b8ea83, PR #27144)
TITLE: [Bugfix] fixes the decoding metadata of dense mla's fp8 kvcache. (#27144)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, body_keyword
STAGE1: repair_correctness; artifacts=L3.mla.flashmla_build,L3.mla.flashmla_v1_adapter; Bumps FlashMLA and updates dense MLA fp8 KV-cache metadata setup.
ARTIFACT_HINTS: L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.mla.flashmla_build
FILES: cmake/external_projects/flashmla.cmake (+2/-1); vllm/attention/ops/flashmla.py (+6/-0); vllm/v1/attention/backends/mla/flashmla.py (+2/-0)
LABELS: ready, ci/build, v1
BODY: Require the flashmla patch https://github.com/vllm-project/FlashMLA/pull/7 to be landed first.

### L3-b4fda58a2d  (L3, 2025-10-22, sha b4fda58a2d0e, PR #27354)
TITLE: [MLA] Bump FlashMLA (#27354)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, body_keyword
STAGE1: repair_correctness; artifacts=L3.mla.flashmla_build; Bumps FlashMLA to include combine-kernel fix for invalid configuration.
ARTIFACT_HINTS: L3.mla.flashmla_build
FILES: cmake/external_projects/flashmla.cmake (+1/-1)
LABELS: ready, ci/build
ISSUES: #27043 [Bug]: FlashMLA: invalid configuration argument
BODY: ## Purpose ⏎ @starwang1024 implemented a fix for `flash_mla_combine_kernel` in https://github.com/vllm-project/FlashMLA/pull/10. This PR bumps the FlashMLA version to grab that change ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/27043 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-084a9dae80  (L3, 2025-10-22, sha 084a9dae801c, PR #27344)
TITLE: [Bugfix] Disable FlexAttention direct block mask building for encoder-only models (#27344)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.flex_attention; Disables faulty direct block-mask building in FlexAttention for encoder-only models.
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+4/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎ - The FP32 Mteb tests are failing after pytorch2.9 update, because `_build_block_mask_direct` return incorrect block mask shape. (https://buildkite.com/vllm/ci/builds/35829/steps/canvas?sid=019a0a13-ef28-4260-87f8-b6f4d685791a) ⏎ ``` ⏎ (EngineCore_DP0 pid=41725)   Developer debug context: raised exception ValueError([ConstantVariable(str: "block_mask was created for block_mask.shape=(1, 1, 7, 16) but got q_len=7 and kv_len=7. As the block mask was created for a larger length than you're using it for, you can either 1. create a new block mask with the correct length, or 2. 'adjust' the existing block mask to the correct length by calling block_mask._adjust(q_len, kv_len). This essentially 'crops' the block mask to the upper left corner, which does not work for all mask_mods!")]) ⏎ ``` ⏎ - Let's disable it to unblock #27329 first, then fix the issue in a follow-up PR. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest -s -v tests/models/language/pooling_mteb_test/test_st_projector.py -k test_embed_models_mteb[model_info1] ⏎ ``` ⏎  ⏎ ## Test Result ⏎ Test can still pass with `create_block_mask` code path ⏎  ⏎ --- ⏎ [details omitted]

### L3-5beacce2ea  (L3, 2025-10-22, sha 5beacce2eae2, PR #27128)
TITLE: [BugFix] bugfix for Flash Attention MLA with full cuda graph IMA following pr-25490 (#27128)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
STAGE1: repair_correctness; artifacts=L3.mla.flashattn; Fixes FlashAttention MLA CUDA-graph illegal memory access with prefix caching.
ARTIFACT_HINTS: L3.mla.flashattn
FILES: vllm/v1/attention/backends/mla/flashattn_mla.py (+20/-13)
LABELS: ready, v1
BODY: Bugfix for Flash Attention MLA with full cuda graph IMA following pr-25490 ⏎  ⏎ Run into illegal memory access error when testing some prompts with prefix caching enabled on Flash Attention MLA backend ⏎  ⏎ Log below is generated with CUDA_LAUNCH_BLOCKING=1 which indicating it's flash attn mla. ⏎  ⏎ ``` ⏎ INFO:/scripts/vllm_scripts/utils.py:CUDA error (../../.deps/vllm-flash-attn-src/hopper/flash_fwd_combine_launch_template.h:60): an illegal memory access was encountered ⏎ INFO:/scripts/vllm_scripts/utils.py:CUDA error (../../.deps/vllm-flash-attn-src/hopper/flash_fwd_combine_launch_template.h:60): an illegal memory access was encountered ⏎ INFO:/scripts/vllm_scripts/utils.py:CUDA error (../../.deps/vllm-flash-attn-src/hopper/flash_fwd_combine_launch_template.h:60): an illegal memory access was encountered ⏎ INFO:/scripts/vllm_scripts/utils.py:CUDA error (../../.deps/vllm-flash-attn-src/hopper/flash_fwd_combine_launch_template.h:60): an illegal memory access was encountered ⏎ INFO:/scripts/vllm_scripts/utils.py:CUDA error (../../.deps/vllm-flash-attn-src/hopper/flash_fwd_combine_launch_template.h:60): an illegal memory access was encountered ⏎ INFO:/scripts/vllm_scripts/utils.py:CUDA error (../../.deps/vllm-flash-attn-src/hopper/flash_fwd_combine_launch_template.h:60): an illegal memory access was encountered ⏎ INFO:/scripts/vllm_scripts/utils.py:CUDA error (../../.deps/vllm-flash-attn-src/hopper/flash_fwd_combine_launch_template.h:60): an illegal memory access was encountered ⏎ INFO:/scripts/vllm_scripts/utils.py:CUDA error (../../.deps/vllm-flash-attn-src/hopper/flash_fwd_combine_launch_template.h:60): an illegal memory access was encountered ⏎ INFO:/scripts/vllm_scripts/utils.py:[1;36m(EngineCore_0 pid=481)[0;0m ERROR 10-13 10:51:40 [multiproc_executor.py:146] Worker proc VllmWorker-5 died unexpectedly, shutting down executor. ⏎ ... ⏎ ``` ⏎  ⏎ And realized it's the same root cause as https://github.com/vllm-project/vllm/pull/25490 where `get_scheduler_metadata` was being called with a different `max_num_splits` than what was being passed to `FlashAttnMLAMetadata`.

### L3-dbfbf9f324  (L3, 2025-10-23, sha dbfbf9f32445, PR #27368)
TITLE: [Attention] Fix FlashMLA metadata builder arguments for q_len > 1 (#27368)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
STAGE1: repair_performance; artifacts=L3.mla.flashmla_v1_adapter; Updates FlashMLA metadata-builder q_len arguments, fixing poor decode performance.
ARTIFACT_HINTS: L3.mla.flashmla_v1_adapter
FILES: vllm/v1/attention/backends/mla/flashmla.py (+5/-1)
LABELS: bug, ready, v1, deepseek
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
PERF_LINES: As of #26541, FlashMLA now supports `q_len > 1` in the decode pipeline. The `get_mla_metadata` call was not updated, however, leading to poor performance (and p | Query Len  | Before (s)   | After (s)    | Speedup | 1          |     0.000051 |     0.000050 |    1.01x | 2          |     0.000051 |     0.000050 |    1.02x | 4          |     0.000052 |     0.000048 |    1.10x | 8          |     0.000
BODY: ## Purpose ⏎ As of #26541, FlashMLA now supports `q_len > 1` in the decode pipeline. The `get_mla_metadata` call was not updated, however, leading to poor performance (and potentially, crashes) in these cases. This PR is a simple bug fix achieving a substantial speedup, especially at small batch sizes. ⏎  ⏎ Note: uses the benchmarks in #26835 (not yet merged) ⏎  ⏎ cc @LucasWilkinson  ⏎  ⏎ ## Test Plan ⏎ `python benchmarks/attention_benchmarks/benchmark.py --config benchmarks/attention_benchmarks/configs/flashmla_bugfix_demo.yaml` ⏎  ⏎ ## Test Result ⏎ ``` ⏎ Batch Size = 1 ⏎ Query Len  | Before (s)   | After (s)    | Speedup  ⏎ ------------------------------------------------------------ ⏎ 1          |     0.000051 |     0.000050 |    1.01x ⏎ 2          |     0.000051 |     0.000050 |    1.02x ⏎ 4          |     0.000052 |     0.000048 |    1.10x ⏎ 8          |     0.000098 |     0.000053 |    1.87x ⏎ 16         |     0.000192 |     0.000057 |    3.39x ⏎ 32         |     0.000359 |     0.000067 |    5.35x ⏎ 64         |     0.000702 |     0.000067 |   10.52x ⏎ 128        |     0.001350 |     0.000131 |   10.27x ⏎ 256        |     0.002630 |     0.000257 |   10.23x ⏎ 512        |     0.005094 |     0.000471 |   10.82x ⏎  ⏎ Batch Size = 2 ⏎ Query Len  | Before (s)   | After (s)    | Speedup  ⏎ ------------------------------------------------------------ ⏎ 1          |     0.000050 |     0.000050 |    1.01x ⏎ 2          |     0.000057 |     0.000047 |    1.23x ⏎ 4          |     0.000101 |     0.000052 |    1.94x ⏎ 8          |     0.000190 |     0.000056 |    3.38x ⏎ 16         |     0.000348 |     0.000091 |    3.81x ⏎ 32         |     0.000680 |     0.000065 |   10.39x ⏎ 64         |     0.001325 |     0.000128 |   10.36x ⏎ 128        |     0.002601 |     0.000256 |   10.17x ⏎ 256        |     0.005099 |     0.000487 |   10.48x ⏎ 512        |     0.009949 |     0.000895 |   11.12x ⏎  ⏎ Batch Size = 4 ⏎ Query Len  | Before (s)   | After (s)    | Speedup  ⏎ ------------------------------------------------------------ ⏎ 1          |     0.000047 |     0.000046 |    1.02x ⏎ 2          |     0.000053 |     0.000051 |    1.04x ⏎ 4          |     0.000098 |     0.000054 |    1.81x ⏎ 8          |     0.000185 |     0.000091 |    2.03x ⏎ 16         |     0.000360 |     0.000065 |    5.54x ⏎ 32         |     0.000702 |     0.000126 |    5.56x ⏎ 64         |     0.001369 |     0.000248 |    5.51x ⏎ 128        |     0.002692 |     0.000498 |    5.41x ⏎ 256        |     0.005233 |     0.000955 |    5.48x ⏎ 512        |     0.010128 |     0.001747 |    5.80x ⏎  ⏎ Batch Size = 8 ⏎ Query Len  | Before (s)   | After (s)    | Speedup  ⏎ ---------------------------------------------- …[truncated]

### L3-284cc92275  (L3, 2025-10-24, sha 284cc922757b, PR #26016)
TITLE: [MISC] `cudagraph_capture_sizes`  related improvements (#26016)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.flash_attn.v1_backend,L3.flashinfer.v1_backend,L3.mla.flashattn; Renames cudagraph capture config fields consumed by attention backends.
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.flashattn
FILES: vllm/v1/attention/backends/flash_attn.py (+1/-1); vllm/v1/attention/backends/flashinfer.py (+1/-1); vllm/v1/attention/backends/mla/flashattn_mla.py (+1/-1); tests/compile/test_config.py (+73/-0); vllm/config/compilation.py (+40/-38); vllm/config/scheduler.py (+0/-15); vllm/config/vllm.py (+107/-26); vllm/engine/arg_utils.py (+62/-5); vllm/model_executor/layers/quantization/mxfp4.py (+1/-1); vllm/model_executor/models/config.py (+11/-13); vllm/v1/attention/backends/gdn_attn.py (+1/-1); vllm/v1/attention/backends/mamba_attn.py (+1/-1); vllm/v1/spec_decode/eagle.py (+1/-1); vllm/v1/worker/gpu_model_runner.py (+3/-6)
LABELS: speculative-decoding, ready, v1
ISSUES: #20283 [RFC][UX][torch.compile][CUDAGraph]: Overhaul `CompilationConfig` and improve CLI `-O<n>`
BODY: ## Purpose ⏎  ⏎ Part of the CompilationConfig improvements ([#20283](https://github.com/vllm-project/vllm/issues/20283)), asked by @ProExpertProg and @WoosukKwon: ⏎  ⏎ - rename max_capture_size -> max_cudagraph_capture_size: used to specify a single size, fills in the rest ⏎ - cudagraph_capture_sizes -> stays the same, used only to specify a full list ⏎ - SchedulerConfig.cuda_graph_sizes: -> remove ⏎ - Also stop reversing the sizes, always store them in ascending order (already on the issue), and always sort when using so we don't rely on them being sorted ⏎  ⏎ Updated:   ⏎ - The assignment of  cudagraph_capture_size changed, from a uniformly stepped size 8 to a 2-level step size:  step_size =16 for size >= 256, and step_size =8 for size <256. (Asked by @ProExpertProg ) https://github.com/vllm-project/vllm/pull/26016#discussion_r2394921792 ⏎ -  new CLIs `--cudagraph-capture-sizes ` and `--max-cudagraph-capture-size`, for corresponding settings in compilation_config respectively.  ⏎ - CLI `--cuda-graph-sizes` is deprecated and will be removed in 0.13.0 or 1.0.0, and is equivalent to `--cudagraph-capture-sizes` now.  ⏎ --- ⏎ [details omitted]

### L3-0f67d4d962  (L3, 2025-10-24, sha 0f67d4d96287, PR #26397)
TITLE: [Attention] Add MLA prefill backend: trtllm_ragged_attention_deepseek (#26397)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
STAGE1: integrate; artifacts=L3.mla.common_v1,NEW:trtllm_ragged_mla_prefill; Adds optional TRTLLM ragged DeepSeek MLA prefill path and environment flag.
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.common_v1
FILES: vllm/envs.py (+6/-0); vllm/v1/attention/backends/mla/common.py (+107/-2)
LABELS: ready, v1, deepseek
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
PERF_LINES: 12.7% qps improvement | -10.2% TTFT improvement | Request throughput (req/s):              10.22 | Output token throughput (tok/s):         10463.54 | Peak output token throughput (tok/s):    16094.00 | Total Token throughput (tok/s):          31380.41 | Mean TTFT (ms):                          64511.82 | Median TTFT (ms):                        55968.36 | P99 TTFT (ms):                           
BODY: ## Purpose ⏎  ⏎ Add MLA prefill backend: trtllm_ragged_attention_deepseek ⏎ * controlled by `VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1` ⏎  ⏎ * [x] Need to fix accuracy issue ⏎   * 0.54 -> 0.59 fixed by zeroing output tensor for context chunk prefill ⏎   * 0.59 -> 0.61 (0.22 -> 0.64 w/ prefix caching) fixed by transposing lse ⏎   * 0.61 -> 0.65 fixed by zeroing workspace ⏎ * [x] performance benchmark ⏎  ⏎ ## Test Plan ⏎  ⏎ * [ ] microbenchmark ⏎   * in follow-up work after refactoring ⏎ * [x] vllm bench serve ⏎ * [x] lm_eval gsm8k ⏎  ⏎ ## Test Result ⏎  ⏎ ### bench serve ⏎  ⏎ 12.7% qps improvement ⏎ -10.2% TTFT improvement ⏎  ⏎ #### Baseline ⏎ ``` ⏎ CUDA_VISIBLE_DEVICES=0 VLLM_LOGGING_LEVEL=DEBUG VLLM_ATTENTION_BACKEND=FLASHINFER_MLA vllm serve --model="deepseek-ai/DeepSeek-V2-Lite-Chat" 2>&1 | tee /tmp/fi.log ⏎ ``` ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     2048 ⏎ Maximum request concurrency:             2048 ⏎ Benchmark duration (s):                  200.42 ⏎ Total input tokens:                      4192256 ⏎ Total generated tokens:                  2097152 ⏎ Request throughput (req/s):              10.22 ⏎ Output token throughput (tok/s):         10463.54 ⏎ Peak output token throughput (tok/s):    16094.00 ⏎ Peak concurrent requests:                2048.00 ⏎ Total Token throughput (tok/s):          31380.41 ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):                          64511.82 ⏎ Median TTFT (ms):                        55968.36 ⏎ P99 TTFT (ms):                           132592.32 ⏎ -----Time per Output Token (excl. 1st token)------ ⏎ Mean TPOT (ms):                          86.54 ⏎ Median TPOT (ms):                        89.57 ⏎ P99 TPOT (ms):                           93.98 ⏎ ---------------Inter-token Latency---------------- ⏎ Mean ITL (ms):                           86.54 ⏎ Median ITL (ms):                         70.26 ⏎ P99 ITL (ms):                            208.33 ⏎ ================================================== ⏎ ``` ⏎ #### This PR ⏎ ``` ⏎ CUDA_VISIBLE_DEVICES=0 VLLM_LOGGING_LEVEL=DEBUG VLLM_ATTENTION_BACKEND=FLASHINFER_MLA VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1 vllm serve --model="deepseek-ai/DeepSeek-V2-Lite-Chat" 2>&1 | tee /tmp/ragged.log ⏎ ``` ⏎  ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     2048 ⏎ Maximum request concurrency:             2048 ⏎ Benchmark duration (s):                  177.74 ⏎ Total input tokens:                      4192256 ⏎ Total generated tokens:                  2097152 ⏎ Request throughput (req/s):              11.52 ⏎ Output token throughput (tok/s):         11799.21 ⏎ Peak output token throughp …[truncated]

### L3-a99564ac5b  (L3, 2025-10-25, sha a99564ac5b2b, PR #27490)
TITLE: [Attention] Add missing kv cache scale setup (#27490)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.mla.common_v1; Restores KV-cache scale setup in MLAAttention initialization.
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+72/-59)
LABELS: bug, ready, deepseek
BODY: ## Purpose ⏎ #25103 missed the kv cache scale setup when breaking out `MLAAttention`. This PR adds this to `MLAAttention.__init__` ⏎  ⏎ cc @pavanimajety  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-141e6a0505  (L3, 2025-10-28, sha 141e6a050596, PR #27367)
TITLE: [Misc] Make reorder batch also separate extends (#27367)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.dispatch.abstract_interface; Changes attention batch-reordering utility to distinguish decode, extend, and prefill.
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+53/-45); tests/v1/attention/test_batch_reordering.py (+111/-0)
LABELS: ready, v1
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:b5d70751d8 (fix PR 27739)
BODY: Pre-requisite for optimizations that want to seperate a batch into `[decode, extend, prefill]`, e.g. https://github.com/vllm-project/vllm/pull/25763 (future optimizations that may use this include skipping 0-seqlen in MLA when batch contains both extends and prefills; DeepSeek-V3.2-Exp skip up-converting fp8-kv-cache for pure prefills)

### L3-a8c02fb5bf  (L3, 2025-10-28, sha a8c02fb5bf2e, PR #26597)
TITLE: [Bugfix][CI] Fix v1 attention backend tests and add CI coverage (#26597)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.flex_attention; Fixes FlexAttention KV-cache layout handling and MLA test setup in backend tests.
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+34/-11); .buildkite/test-pipeline.yaml (+9/-0); tests/v1/attention/test_attention_backends.py (+22/-14); tests/v1/attention/test_mla_backends.py (+2/-1)
LABELS: ready, ci/build, v1
ISSUES: #26537 [Bug]: V1 attention tests are broken
BODY: ## Purpose ⏎  ⏎ Fixes #26537 by correcting KV-cache layout handling and MLA config setup in the harness, disabling gradients on mock projection weights to avoid autograd asserts, and ensuring these attention suites now run in CI by adding them to the Buildkite “V1 Test others” step. ⏎  ⏎ ## Test Plan ⏎  ⏎ `pytest tests/v1/attention` ⏎  ⏎ ## Test Result ⏎  ⏎ `78 passed, 2 warnings in 79.24s` ⏎  ⏎ --- ⏎ [details omitted]

### L3-f257544709  (L3, 2025-10-28, sha f257544709a8, PR #27598)
TITLE: Install pre-built xformers-0.0.32.post2 built with pt-2.9.0 (#27598)
SOURCES: dependency_pin
STAGE1: repair_build_dependency; artifacts=L3.xformers.v1_backend; Installs prebuilt xFormers wheel for PyTorch 2.9 CUDA builds.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+0/-7); requirements/cuda.txt (+1/-1)
LABELS: ready, ci/build
DEEP_STUDY: deep-study: this PR was reverted by PR 27714 (confirmed_revert, reason=build_or_dependency) || deep-study: this PR was reverted by PR 27768 (reland, reason=build_or_dependency)
BODY: ## Purpose ⏎  ⏎ Instead of waiting for xformers to release a new version for PyTorch 2.9.0, I have built `0.0.32.post2` locally and made the wheel available. ⏎  ⏎ For more context, we don’t want to wait for `xformers` package for 2.9 to become available.  So, I opt to build it from source.  This works for CI, but has several issues like (1) increasing build time and (2) not listed as a dependency in `cuda.txt`.  So, installing a pre-built wheel would help in the meantime until there is a new xformers version ⏎  ⏎ @ywang96

### L3-9007bf57e6  (L3, 2025-10-28, sha 9007bf57e6a2, PR #27714)
TITLE: Revert "Install pre-built xformers-0.0.32.post2 built with pt-2.9.0" (#27714)
SOURCES: dependency_pin
STAGE1: revert; artifacts=L3.xformers.v1_backend; Reverts prebuilt xFormers wheel change after CUDA 13 build failure.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+7/-0); requirements/cuda.txt (+1/-1)
LABELS: ci/build
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 27598 reason=build_or_dependency
BODY: Reverts vllm-project/vllm#27598 ⏎  ⏎ Broke CUDA 13 build. https://buildkite.com/vllm/release/builds/9637/steps/canvas?sid=019a2dfc-911c-4783-b421-9d3acc153e1b

### L3-5b0448104f  (L3, 2025-10-29, sha 5b0448104fa1, PR #27424)
TITLE: [Bug] Raise error explicitly if using incompatible backend (#27424)
SOURCES: symbol_pickaxe, body_keyword
STAGE1: repair_correctness; artifacts=L3.platform.cuda_selection; Adds CUDA backend compatibility guard for user-selected attention backends.
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+15/-0)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ If the users choose the wrong backend, will meet unexpected error, eg: ⏎  ⏎ ``` ⏎ (EngineCore_DP7 pid=4191233)   File "/home/wentao/vllm-source/vllm/model_executor/models/utils.py", line 642, in make_layers ⏎ (EngineCore_DP7 pid=4191233)     maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}")) ⏎ (EngineCore_DP7 pid=4191233)                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ (EngineCore_DP7 pid=4191233)   File "/home/wentao/vllm-source/vllm/model_executor/models/deepseek_v2.py", line 1128, in <lambda> ⏎ (EngineCore_DP7 pid=4191233)     lambda prefix: DeepseekV2DecoderLayer( ⏎ (EngineCore_DP7 pid=4191233)                    ^^^^^^^^^^^^^^^^^^^^^^^ ⏎ (EngineCore_DP7 pid=4191233)   File "/home/wentao/vllm-source/vllm/model_executor/models/deepseek_v2.py", line 1005, in __init__ ⏎ (EngineCore_DP7 pid=4191233)     self.self_attn = attn_cls( ⏎ (EngineCore_DP7 pid=4191233)                      ^^^^^^^^^ ⏎ (EngineCore_DP7 pid=4191233)   File "/home/wentao/vllm-source/vllm/model_executor/models/deepseek_v2.py", line 953, in __init__ ⏎ (EngineCore_DP7 pid=4191233)     self.mla_attn = MultiHeadLatentAttentionWrapper( ⏎ (EngineCore_DP7 pid=4191233)                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ (EngineCore_DP7 pid=4191233)   File "/home/wentao/vllm-source/vllm/model_executor/layers/mla.py", line 90, in __init__ ⏎ (EngineCore_DP7 pid=4191233)     self.mla_attn = MLAAttention( ⏎ (EngineCore_DP7 pid=4191233)                     ^^^^^^^^^^^^^ ⏎ (EngineCore_DP7 pid=4191233)   File "/home/wentao/vllm-source/vllm/attention/layer.py", line 651, in __init__ ⏎ (EngineCore_DP7 pid=4191233)     self.impl = impl_cls( ⏎ (EngineCore_DP7 pid=4191233)                 ^^^^^^^^^ ⏎ (EngineCore_DP7 pid=4191233) TypeError: FlashInferImpl.__init__() got an unexpected keyword argument 'q_lora_rank' ⏎ ``` ⏎  ⏎ This PR raise error explicitly and let users know how to solve the issue ⏎  ⏎ ``` ⏎ ValueError: Attention backend _Backend.FLASHINFER is incompatible with MLA models. Please use one of the MLA backends: FLASHINFER_MLA, CUTLASS_MLA, FLASHMLA, FLASH_ATTN_MLA, or TRITON_MLA. Alternatively, set VLLM_MLA_DISABLE=1 to disable MLA for this model. ⏎ ```

### L3-b5d70751d8  (L3, 2025-10-29, sha b5d70751d82c, PR #27739)
TITLE: [BugFix] Reordering extend logic fix (#27739)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.dispatch.abstract_interface; Fixes extend classification in attention batch-reordering utility.
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+5/-5); tests/v1/attention/test_batch_reordering.py (+18/-3)
LABELS: ready, v1
DEEP_STUDY: deep-study correctness case vllm:b5d70751d8: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=#27367
BODY: Fix incorrect definition of extends in https://github.com/vllm-project/vllm/pull/27367, more extensive unit tests, fix reordering bug when multiple swaps are involved + light refactor. ⏎  ⏎ Credit to @ganyi1996ppo for finding the issue

### L3-ba33e8830d  (L3, 2025-10-30, sha ba33e8830dce, PR #27768)
TITLE: Reapply "Install pre-built xformers-0.0.32.post2 built with pt-2.9.0" (#27768)
SOURCES: dependency_pin
STAGE1: reland; artifacts=L3.xformers.v1_backend; Relands prebuilt xFormers dependency change with CUDA 13 wheel fix.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+0/-7); requirements/cuda.txt (+2/-2)
LABELS: ready, ci/build
DEEP_STUDY: deep-study revert record: reland of PR(s) 27598 reason=build_or_dependency
PERF_LINES: This relands https://github.com/vllm-project/vllm/pull/27598 exactly as it is.  For the CUDA 13.0 build failure in https://buildkite.com/vllm/release/builds/963
BODY: ## Purpose ⏎  ⏎ This relands https://github.com/vllm-project/vllm/pull/27598 exactly as it is.  For the CUDA 13.0 build failure in https://buildkite.com/vllm/release/builds/9637#019a3162-f87b-4fd4-820a-5612913b590e, I have added the missing xformers wheel for cu130 at https://download.pytorch.org/whl/cu130/xformers-0.0.33%2B5d4b92a5.d20251029-cp39-abi3-linux_x86_64.whl ⏎  ⏎ ## Test Plan ⏎  ⏎ https://buildkite.com/vllm/release/builds/9672 should all be green ⏎  ⏎ @simon-mo

### L3-3933f18a5e  (L3, 2025-10-31, sha 3933f18a5e7b, PR #27853)
TITLE: [Bugfix] Avoid too small block m/n for FlexAttention kernel option (#27853)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.flex_attention; Adjusts FlexAttention kernel options to avoid too-small BLOCK_M/BLOCK_N failures.
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+5/-0)
LABELS: ready, v1
ISSUES: #27724 [CI Failure]: torch._inductor.exc.InductorError in Nightly build to run all tests
BODY: ## Purpose ⏎ - Fix #27724 ⏎ - Current `get_kernel_options` can cause too small BLOCK_M/BLOCK_N (BLOCK_M=8), which broke some encoder-only models.  ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest -s -v tests/models/language/pooling_mteb_test/test_jina.py -k test_embed_models_mteb[model_info0] ⏎ ``` ⏎  ⏎ ## Test Result ⏎ Failed test should pass now. ⏎  ⏎ --- ⏎ [details omitted]

### L3-18961c5ea6  (L3, 2025-11-03, sha 18961c5ea629, PR #27753)
TITLE: [Hybrid] Pass kernel block size to builders (#27753)
SOURCES: path_core, body_keyword
STAGE1: repair_correctness; artifacts=L3.flash_attn.v1_backend; Passes kernel block size to FlashAttention metadata builders and restricts bad hybrid sizes.
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+5/-1); vllm/v1/kv_cache_interface.py (+7/-1); vllm/v1/worker/gpu_model_runner.py (+25/-6); vllm/v1/worker/utils.py (+25/-19)
LABELS: ready, v1
ISSUES: #26936 [Bug]: Hybrid Attention models broken after switching to flashinfer 0.4 (tested on Granite 4.0 H, Qwen3-Next, Jamba-3B, Nemotron-H-8b) | #27264 [Bug]: Cache malformation in hybrid models with SSM cache dtype float32 and block allocation wrap around
BODY: ## Purpose ⏎  ⏎ Solves https://github.com/vllm-project/vllm/issues/27264 and https://github.com/vllm-project/vllm/issues/26936 ⏎  ⏎ This PR makes two changes: ⏎ - The GPU model runner will now pass the kernel block size to the metadata builders, fixing a pretty bad bug that exists on main. ⏎ - I also restrict the kernel block sizes for the FlashAttention backend based on the discussion in https://github.com/vllm-project/vllm/issues/27264. This is required since for block_size >= 128, FA will read partial block data that may contain NaN's if mamba state is in fp32, which leads to NaN's appearing in output.  ⏎  ⏎ ## Test Plan ⏎  ⏎ I'm using the following test script: ⏎ ```python ⏎ from vllm import LLM, SamplingParams ⏎ import os ⏎  ⏎ os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN' ⏎ #os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER' ⏎  ⏎ prompts = ["Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nConsider the paths of length $16$ that follow the lines from the lower left corner to the upper right corner on an $8\\times 8$ grid. Find the number of such paths that change direction exactly four times, as in the examples shown below.\n\nRemember to put your answer on its own line after \"Answer:\"."] ⏎ sampling_params = SamplingParams(temperature=0.8, top_p=0.95) ⏎  ⏎ llm = LLM( ⏎     model="nvidia/NVIDIA-Nemotron-Nano-9B-v2", ⏎     trust_remote_code=True, ⏎     num_gpu_blocks_override=10, ⏎     compilation_config={'cudagraph_capture_sizes': [1]}, ⏎     mamba_ssm_cache_dtype="float32", ⏎     max_num_seqs=1, ⏎     enable_prefix_caching=True, ⏎ ) ⏎  ⏎ outputs = [] ⏎ for _ in range(10): ⏎     outputs.append(llm.generate(prompts, sampling_params)[0]) ⏎  ⏎ # Print the outputs. ⏎ print(f"Prompt: {prompts[0]!r}") ⏎ for output in outputs: ⏎     prompt = output.prompt ⏎     generated_text = output.outputs[0].text ⏎     print(f"Generated text: {generated_text!r}") ⏎     print(f"Generated token IDs: {output.outputs[0].token_ids!r}") ⏎ ``` ⏎ Which on main using `FLASH_ATTN` produces: ⏎ ``` ⏎ Generated text: '</think>\nThe problem requires finding the number of paths of length 16' ⏎ Generated token IDs: [1885, 74045, 1561, 1784, 4127, 10867, 13170, 1278, 2782, 1307, 22344, 1307, 5592, 1032, 1049, 1054] ⏎ Generated text: '' ⏎ Generated token IDs: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ⏎ Generated text: '' ⏎ Generated token IDs: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ⏎ Generated text: '' ⏎ Generated token IDs: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ⏎ Generated text: '' ⏎ Generated token IDs: [0, 0, 0, 0, 0, 0, 0, 0, …[truncated]

### L3-145c00a4d3  (L3, 2025-11-03, sha 145c00a4d32b, PR #27777)
TITLE: [Bugfix] change FlashMLA reorder_batch_threshold (#27777)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
STAGE1: retune; artifacts=L3.mla.flashmla_v1_adapter; Retunes FlashMLA reorder_batch_threshold from 512 to 128 to avoid failures.
ARTIFACT_HINTS: L3.mla.flashmla_v1_adapter
FILES: vllm/v1/attention/backends/mla/flashmla.py (+1/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎ Reduce FlashMLA `reorder_batch_threshold` from 512 to 128. This fixes LM Eval Large Models when applied to 82af928 (the commit that merged #26541) but doesn't seem sufficient to fix it on TOT. There might be multiple commits involved in that failure ⏎  ⏎ ## Test Plan ⏎ `pytest -s -v test_lm_eval_correctness.py::test_lm_eval_correctness_param[config_filename4] --config-list-file=configs/models-large.txt --tp-size=4` ⏎  ⏎ ## Test Result ⏎ Passes (on 82af928) ⏎  ⏎ --- ⏎ [details omitted]

### L3-7e4be74104  (L3, 2025-11-04, sha 7e4be741044b, PR #27884)
TITLE: [Bug] Batch invariant: Fix flash attn MLA `RuntimeError: scheduler_metadata must have shape (metadata_size)` (#27884)
SOURCES: path_core, subject_keyword, release_notes, corpus:kernel-correctness-cases, body_keyword
STAGE1: repair_correctness; artifacts=L3.mla.flashattn; Fixes FlashAttention MLA scheduler_metadata shape under batch-invariant mode.
ARTIFACT_HINTS: L3.mla.flashattn
FILES: vllm/v1/attention/backends/mla/flashattn_mla.py (+3/-3); vllm/model_executor/layers/batch_invariant.py (+2/-0)
LABELS: ready, v1
DEEP_STUDY: deep-study correctness case vllm:7e4be74104: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎  ⏎ ```bash ⏎ export VLLM_BATCH_INVARIANT=1 ⏎ vllm serve deepseek-ai/DeepSeek-V3 -tp 8  --enable-expert-parallel --port 9256 ⏎ vllm bench serve --model deepseek-ai/DeepSeek-V3  --dataset-name random --host 127.0.0.1 --port 9256 --random-input-len 4 --random-output-len 64 --request-rate inf --num-prompts 256 ⏎ ``` ⏎  ⏎ will trigger error: ⏎  ⏎ ```bash ⏎ ^[[1;36m(Worker_TP4_EP4 pid=4005888)^[[0;0m ERROR 10-31 00:28:52 [multiproc_executor.py:703]     return self._op(*args, **kwargs) ⏎ ^[[1;36m(Worker_TP4_EP4 pid=4005888)^[[0;0m ERROR 10-31 00:28:52 [multiproc_executor.py:703]            ^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ ^[[1;36m(Worker_TP4_EP4 pid=4005888)^[[0;0m ERROR 10-31 00:28:52 [multiproc_executor.py:703]   File "/home/yewentao256/vllm-source/vllm/attention/layer.py", line 1055, in unified_mla_attention_with_output ⏎ ^[[1;36m(Worker_TP4_EP4 pid=4005888)^[[0;0m ERROR 10-31 00:28:52 [multiproc_executor.py:703]     self.impl.forward( ⏎ ^[[1;36m(Worker_TP4_EP4 pid=4005888)^[[0;0m ERROR 10-31 00:28:52 [multiproc_executor.py:703]   File "/home/yewentao256/vllm-source/vllm/v1/attention/backends/mla/common.py", line 2024, in forward ⏎ ^[[1;36m(Worker_TP4_EP4 pid=4005888)^[[0;0m ERROR 10-31 00:28:52 [multiproc_executor.py:703]     attn_out, lse = self._forward_decode( ⏎ ^[[1;36m(Worker_TP4_EP4 pid=4005888)^[[0;0m ERROR 10-31 00:28:52 [multiproc_executor.py:703]                     ^^^^^^^^^^^^^^^^^^^^^ ⏎ ^[[1;36m(Worker_TP4_EP4 pid=4005888)^[[0;0m ERROR 10-31 00:28:52 [multiproc_executor.py:703]   File "/home/yewentao256/vllm-source/vllm/v1/attention/backends/mla/flashattn_mla.py", line 289, in _forward_decode ⏎ ^[[1;36m(Worker_TP4_EP4 pid=4005888)^[[0;0m ERROR 10-31 00:28:52 [multiproc_executor.py:703]     attn_out = flash_attn_varlen_func( ⏎ ^[[1;36m(Worker_TP4_EP4 pid=4005888)^[[0;0m ERROR 10-31 00:28:52 [multiproc_executor.py:703]                ^^^^^^^^^^^^^^^^^^^^^^^ ⏎ ^[[1;36m(Worker_TP4_EP4 pid=4005888)^[[0;0m ERROR 10-31 00:28:52 [multiproc_executor.py:703]   File "/home/yewentao256/vllm-source/vllm/vllm_flash_attn/flash_attn_interface.py", line 261, in flash_attn_varlen_func ⏎ ^[[1;36m(Worker_TP4_EP4 pid=4005888)^[[0;0m ERROR 10-31 00:28:52 [multiproc_executor.py:703]     out, softmax_lse, _, _ = torch.ops._vllm_fa3_C.fwd( ⏎ ^[[1;36m(Worker_TP4_EP4 pid=4005888)^[[0;0m ERROR 10-31 00:28:52 [multiproc_executor.py:703]                              ^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ ^[[1;36m(Worker_TP4_EP4 pid=4005888)^[[0;0m ERROR 10-31 00:28:52 [multiproc_executor.py:703]   File "/home/yewentao256/.venv/lib64/python3.12/site-packages/torch/_ops.py", line 1255, in __call__ ⏎ ^[[1;36m(Worker_TP4_EP4 pid …[truncated]

### L3-dc937175d4  (L3, 2025-11-04, sha dc937175d496, PR #25763)
TITLE: [ROCm][Perf] New design on ROCm AITER MHA backend Implementation (#25763)
SOURCES: path_core, symbol_pickaxe, body_keyword
STAGE1: optimize; artifacts=L3.rocm.aiter_fa; Redesigns ROCm AITER MHA backend to avoid redundant KV fetches and reorder phases.
ARTIFACT_HINTS: L3.rocm.aiter_fa, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+526/-275); vllm/v1/attention/backends/utils.py (+67/-0)
LABELS: rocm, ready, ci/build, v1
PERF_LINES: The current `AiterFlashAttentionImpl` fetches K/V every run, which creates unnecessary memory pressure and non-trivial latency—especially with long prompts. Thi | Long prompt, short output (`2k prompt, 16 output`): ~4.x throughput improvement. | Short prompt, long output (`128 prompt, 1k output`): ~2.x throughput improvement. | Extremely long prompt (`192k prompt, 2k output`): ~5.x throughput impr
BODY: ## Purpose ⏎ The current `AiterFlashAttentionImpl` fetches K/V every run, which creates unnecessary memory pressure and non-trivial latency—especially with long prompts. This PR: ⏎ - Removes redundant KV fetches from the attention backend ⏎ - Introduces a phase-aware execution (decode, pure prefill, chunk prefill) and reorders inputs to [decode:chunk_prefill:pure_prefill] for token-contiguous memory access. ⏎ - Rewrites the “fetch KV” Triton kernel for better occupancy in chunked prefill and similar scenarios. ⏎  ⏎ ## Design and implementation ⏎  ⏎ <img width="1475" height="2031" alt="image" src="https://github.com/user-attachments/assets/b49519dd-ccd1-4f85-b4c2-4bf271ea86f4" /> ⏎  ⏎ Phase-aware path: ⏎ - decode ⏎ - chunk prefill (cp) ⏎ - pure prefill (pp) ⏎  ⏎ Input reordering to [decode:cp:pp] ensures tokens are contiguous in memory, improving kernel locality and occupancy. The reorder occurs in both Scheduler's scheduling phase and ModelRunner's state updating phase. We add this `split_prefill_from_chunk` to the `SchedulerConfig` to control this behavior, which will be turned on if both `VLLM_ROCM_USE_AITER`  and `VLLM_ROCM_USE_AITER_MHA` are set. ⏎  ⏎ Compared with the old one, this solution is more memory efficient and fast, especially on the long prompt scenario. Here is the Performance Measured on Qwen3, Mi308: ⏎  ⏎ Long prompt, short output (`2k prompt, 16 output`): ~4.x throughput improvement. ⏎ Short prompt, long output (`128 prompt, 1k output`): ~2.x throughput improvement. ⏎ Extremely long prompt (`192k prompt, 2k output`): ~5.x throughput improvement.  ⏎  ⏎ ## Test Plan ⏎  ⏎ __acc__  : lm_eval test for accuracy verification  ⏎ __perf__ : vllm bench test ⏎ ## Test Result ⏎ ### 2k prompt 16 output case: ⏎ old impl ⏎ <img width="775" height="723" alt="image" src="https://github.com/user-attachments/assets/2266b6aa-8dc7-46e5-bd1d-c191dfb79c9e" /> ⏎  ⏎ new impl ⏎ <img width="752" height="713" alt="image" src="https://github.com/user-attachments/assets/54cf96ce-7c0c-4596-8cf8-a0ca29df0fad" /> ⏎  ⏎ ### 128 prompt 1k output case: ⏎ old impl ⏎ <img width="728" height="718" alt="image" src="https://github.com/user-attachments/assets/64be93c5-3538-4413-b7b4-625cd26ceb56" /> ⏎  ⏎ new impl ⏎ <img width="722" height="722" alt="image" src="https://github.com/user-attachments/assets/bb874940-5f0c-4fcb-974c-bbefaf072029" /> ⏎  ⏎ ### acc verification ⏎ We test this PR on Qwen3-30B-A3B-FP8 on gsm8k with lm_eval, and here is the result: ⏎  ⏎ ``` ⏎ # previous implementation ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract| …[truncated]

### L3-c765f0b443  (L3, 2025-11-05, sha c765f0b443c2, PR #27994)
TITLE: [FlashInfer] Avoid FlashInfer block_size 16 + head_size 256 on blackwell (#27994)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts(reverted), body_keyword
STAGE1: repair_correctness; artifacts=L3.flashinfer.v1_backend; Guards FlashInfer Blackwell head_size 256 with block_size 16 due incorrect results.
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+9/-0); vllm/model_executor/models/config.py (+12/-0)
LABELS: ready, v1
DEEP_STUDY: deep-study: this PR was reverted by PR 36987 (confirmed_revert, reason=build_or_dependency)
BODY: ## Purpose ⏎ Avoid this combination as https://github.com/flashinfer-ai/flashinfer/issues/1993 reports this combination is not correct. ⏎  ⏎ For most models with head_size 256, users now need --block_size 32 / --block_size 64 ⏎  ⏎ For hybrid mamba like qwen3-next, the block_size can be resolved automatically  ⏎  ⏎ Thanks @vadiklyutiy for the exploration on this problem https://github.com/vllm-project/vllm/pull/27704 ⏎  ⏎ ## Test Plan ⏎  ⏎ Test non-hybrid case and hybrid case ⏎  ⏎ ## Test Result ⏎ All tests are on B200 ⏎  ⏎ ### Normal model ⏎  ⏎ I don't know a model with head_size 256. I've changed the 256 in this PR to 64 and test opt-125m ⏎ ``` ⏎ vllm serve facebook/opt-125m ⏎ ``` ⏎ will result in ⏎ ``` ⏎ AssertionError: There is a bug in FlashInfer block_size 16 head size 256 support. Please avoid this combination by passing --block-size 32 or --block-size 64. ⏎ ``` ⏎  ⏎ ``` ⏎ vllm serve facebook/opt-125m --block-size 64 ⏎ ``` ⏎ works ⏎  ⏎ ### Hybrid ⏎ ``` ⏎ vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct -tp 4 ⏎ lm_eval --model local-completions --model_args model=Qwen/Qwen3-Next-80B-A3B-Instruct,base_url=http://localhost:8000/v1/completions -t gsm8k --num_fewshot 5 --batch_size 250 ⏎ ``` ⏎ before this PR ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.2123|±  |0.0113| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.1933|±  |0.0109| ⏎ ``` ⏎ after this PR ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8491|±  |0.0099| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.8120|±  |0.0108| ⏎ ``` ⏎  ⏎ --- ⏎ [details omitted]

### L3-faedbb4d4f  (L3, 2025-11-05, sha faedbb4d4fe4, PR #27856)
TITLE: [Feature] Extend batch invariant torch.compile to B200 (#27856)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.flashinfer.trtllm_gen,L3.flashinfer.utils_dependency; Disables problematic TRTLLM attention in B200 batch-invariant torch.compile/cudagraph cases.
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+6/-0); tests/v1/generation/test_batch_invariance.py (+0/-2); vllm/model_executor/layers/batch_invariant.py (+24/-15)
LABELS: ready, v1
PERF_LINES: B200 torch 2.9 with batch invariant overrides, 15% throughput gains | Request throughput (req/s):              18.41 | Output token throughput (tok/s):         1165.62 | Peak output token throughput (tok/s):    1276.00 | Total Token throughput (tok/s):          1220.85 | Mean TTFT (ms):                          679.74 | Median TTFT (ms):                        687.39 | P99 TTFT (ms):              
BODY: ## Purpose ⏎ This PR resolves issues with torch.compile + cudagraphs batch invariance issues on B200. Namely, `trtllm_attention` on B200 + cudagraphs causes issues. ⏎  ⏎ We extend all the unit tests as well to use torch.compile for evaluating batch invariance, and also disable GEMM custom operator overriding on PyTorch 2.10+, as it now contains the batch invariant cuda overrides, such as https://github.com/pytorch/pytorch/pull/166735. For PyTorch 2.9, B200 still requires the custom Triton GEMM kernels. ⏎ ## Test Plan ⏎ The following tests are run on both H100 and B200. ⏎  ⏎ `pytest tests/v1/generation/test_batch_invariance.py` ⏎ `VLLM_TEST_MODEL="deepseek-ai/DeepSeek-V3" VLLM_TEST_TP_SIZE=8 VLLM_ATTENTION_BACKEND="FLASH_ATTN_MLA" pytest tests/v1/generation/test_batch_invariance.py -k "test_logprobs_bitwise_batch_invariance_bs1_vs_bsN[FLASH_ATTN_MLA]"` ⏎  ⏎ ## Test Result ⏎  ⏎ ## Performance ⏎ `VLLM_BATCH_INVARIANT=1 VLLM_ATTENTION_BACKEND="TRITON_MLA" vllm serve deepseek-ai/DeepSeek-R1 -tp 8  --enable-expert-parallel --port 9256 --gpu_memory_utilization 0.95 --max_model_len 40960` ⏎  ⏎ `vllm bench serve --model deepseek-ai/DeepSeek-R1  --dataset-name random --host 127.0.0.1 --port 9256 --random-input-len 4 --random-output-len 64 --request-rate inf --num-prompts 256` ⏎  ⏎ B200 torch 2.9 with batch invariant overrides, 15% throughput gains ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     256        ⏎ Failed requests:                         0          ⏎ Benchmark duration (s):                  13.91      ⏎ Total input tokens:                      768        ⏎ Total generated tokens:                  16210      ⏎ Request throughput (req/s):              18.41      ⏎ Output token throughput (tok/s):         1165.62    ⏎ Peak output token throughput (tok/s):    1276.00    ⏎ Peak concurrent requests:                256.00     ⏎ Total Token throughput (tok/s):          1220.85    ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):                          679.74     ⏎ Median TTFT (ms):                        687.39     ⏎ P99 TTFT (ms):                           698.39     ⏎ -----Time per Output Token (excl. 1st token)------ ⏎ Mean TPOT (ms):                          209.57     ⏎ Median TPOT (ms):                        209.51     ⏎ P99 TPOT (ms):                           212.54     ⏎ ---------------Inter-token Latency---------------- ⏎ Mean ITL (ms):                           209.56     ⏎ Median ITL (ms):                         209.37     ⏎ P99 ITL (ms):                            216.25     ⏎ ================================================== ⏎ ``` ⏎  ⏎ B200 torch 2.9 eager ⏎ ` …[truncated]

### L3-6cae1e5332  (L3, 2025-11-05, sha 6cae1e53326a, PR #27224)
TITLE: [ROCm][MLA] Support block-size > 1 for AITER MLA backend  (#27224)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
STAGE1: extend_support; artifacts=L3.mla.rocm_aiter,L3.platform.rocm_selection; Extends ROCm AITER MLA backend to support block-size greater than one.
ARTIFACT_HINTS: L3.mla.rocm_aiter, L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+3/-10); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+31/-7); tests/kernels/attention/test_attention_selector.py (+0/-7)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎ The `AITERMLABackend` now only support `block-size=1` scenario for inference. This constrain may lead to some serious host overhead when we are about to allocate or free cache blocks for long context requests cause there might exist large amount of blocks to operate. Thanks to the insights of @gyu-amd . ⏎  ⏎ In this PR, we remapping the `block_table` to 1 block size case every step in `AITERMLAMetadataBuilder` to alleviate the host overhead during allocate and deallocate blocks This change also helps to support a wider range of block size for `AITERMLABackend`, makes the `AITERMLABackend` on ROCm platform aligns with the vllm's official usgae and more flexible . ⏎  ⏎ ## Test Plan ⏎ Verified on gsm8k for accuracy, performance improvement will also be attached later ⏎ test script: ⏎ ``` ⏎  ⏎ export VLLM_USE_V1=1 ⏎ export SAFETENSORS_FAST_GPU=1 ⏎ export VLLM_ROCM_USE_AITER=1 ⏎ export VLLM_ROCM_USE_AITER_MOE=1 ⏎ export VLLM_USE_TRITON_FLASH_ATTN=0 ⏎ export NCCL_DEBUG=WARN ⏎ export VLLM_RPC_TIMEOUT=1800000 ⏎ export VLLM_ROCM_USE_AITER_ASMMOE=1 ⏎ export VLLM_ROCM_USE_AITER_MHA=0 ⏎ export VLLM_ROCM_USE_TRITON_ROPE=1 ⏎  ⏎ model_path="deepseek-r1-FP8-Dynamic" ⏎ vllm serve $model_path \ ⏎   --tensor-parallel-size 8 \ ⏎   --max-num-batched-tokens 32768 \ ⏎   --trust-remote-code \ ⏎   --no-enable-prefix-caching \ ⏎   --disable-log-requests \ ⏎   --gpu_memory_utilization 0.9 \ ⏎   --block-size 128 \ ⏎   --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' ⏎ ``` ⏎ ## Test Result ⏎  ⏎ ``` ⏎ # gsm8k test ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9507|±  |0.0060| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.9484|±  |0.0061| ⏎ ``` ⏎  ⏎ --- ⏎ [details omitted]

### L3-d43ad5a757  (L3, 2025-11-05, sha d43ad5a75790, PR #28100)
TITLE: [BugFix] Fix DCP Assert (AssertionError: DCP not support reorder_batch_threshold > 1 now.) (#28100)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.mla.common_v1,L3.mla.flashattn,L3.dispatch.abstract_interface; Fixes DCP reorder_batch_threshold assertions in MLA metadata and utility code.
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashattn, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/mla/common.py (+2/-1); vllm/v1/attention/backends/mla/flashattn_mla.py (+6/-1); vllm/v1/attention/backends/utils.py (+10/-1)
LABELS: v1
BODY: Fix `AssertionError: DCP not support reorder_batch_threshold > 1 now.` when running with DCP on hopper

### L3-16b37f3119  (L3, 2025-11-05, sha 16b37f311991, PR #27518)
TITLE: [bugfix] fix wrong `dcp_local_seq_lens` calc (#27518)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.mla.common_v1; Corrects dcp_local_seq_lens calculation in MLA common backend.
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+1/-1)
LABELS: v1
BODY: When the `seq_lens` is exactly divisible by the `dcp_world_size`, it causes the `dcp_local_seq_lens` on all ranks to be incremented by one. Fix this calculation logic. ⏎  ⏎ CC @youzhedian @minosfuture @youkaichao

### L3-981cadb35c  (L3, 2025-11-06, sha 981cadb35c19, PR #28181)
TITLE: [Bugfix][Kernel] fix merge attn states when both prefix and suffix are empty (#28181)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.merge.cuda_lse; Fixes CUDA merge_attn_states when both prefix and suffix are empty.
ARTIFACT_HINTS: L3.merge.cuda_lse
FILES: csrc/attention/merge_attn_states.cu (+26/-0)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-4a36681f85  (L3, 2025-11-07, sha 4a36681f8548, PR #27990)
TITLE: [flashinfer][fix] do not check nvcc availability when using pre-downloaded cubins (#27990)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
STAGE1: repair_build_dependency; artifacts=L3.flashinfer.utils_dependency; Allows FlashInfer pre-downloaded cubins without nvcc availability check.
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+6/-2)
LABELS: ready, nvidia
BODY: Summary: https://github.com/vllm-project/vllm/pull/26443 adds checking of availability of nvcc as a condition to enable flashinfer moe. In our deployment env, there is no nvcc, so flashinfer moe is disabled ⏎ Differential Revision: D86104899

### L3-608bb14462  (L3, 2025-11-07, sha 608bb1446285, PR #27840)
TITLE: [Attention] Remove max cudagraph size limit of 992 (#27840)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.flash_attn.v1_backend,L3.mla.flashattn; Removes FlashAttention and FlashAttention-MLA CUDA graph capture-size limit.
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.mla.flashattn
FILES: vllm/v1/attention/backends/flash_attn.py (+0/-7); vllm/v1/attention/backends/mla/flashattn_mla.py (+0/-7)
LABELS: ready, v1
BODY: This is to support cuda graph capturing beyond 992. Tested working for larger size

### L3-2108a571d7  (L3, 2025-11-09, sha 2108a571d7ee, PR #26696)
TITLE: [DCP] Support dcp kv_cache interleave size > 1 (#26696)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.flash_attn.v1_backend,L3.mla.common_v1,L3.dispatch.abstract_interface; Adds DCP KV-cache interleave-size support across FlashAttention/MLA metadata path.
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.mla.common_v1, L3.dispatch.abstract_interface
FILES: vllm/attention/ops/common.py (+1/-0); vllm/v1/attention/backends/flash_attn.py (+11/-2); vllm/v1/attention/backends/mla/common.py (+77/-75); vllm/v1/attention/backends/utils.py (+38/-0); tests/distributed/test_context_parallel.py (+7/-0); tests/v1/worker/test_gpu_model_runner.py (+2/-0); vllm/config/parallel.py (+11/-0); vllm/config/vllm.py (+17/-0); vllm/engine/arg_utils.py (+6/-0); vllm/v1/worker/block_table.py (+16/-2); vllm/v1/worker/gpu_input_batch.py (+2/-0); vllm/v1/worker/gpu_model_runner.py (+14/-0)
LABELS: ready, v1
BODY: ## Purpose ⏎ ### 1. cp_kv_cache_interleave_size support ⏎ In dcp scenario, kv_cache is split across dcp ranks, current implementation ([#23734](https://github.com/vllm-project/vllm/pull/23734)) split kv_cache with a token-level interleave style: token_idx i is stored on GPU whose dcp_rank == i % dcp_world_size. ⏎  ⏎ For the convenience of pd disaggregate support, we add the cp_kv_cache_interleave_size argument to control the interleave size of kv_cache split size: store interleave_size tokens on dcp i, then store next interleave_size tokens on dcp i+1. The default value of cp_kv_cache_interleave_size is 1, which is same as original token-level interleave implementation. By setting cp_kv_cache_interleave_size to block_size, we can split kv_cache with a block-level interleave style, and makes it easy to support pd disaggregate with dcp > 1: D nodes only need to pull the corresponding kv_cache blocks, without need to rearange tokens in blocks. ⏎  ⏎ Only dcp with cp_kv_cache_interleave_size is supported now, but the case of (p)cp is also considered and is easy to extend in the future. ⏎  ⏎ ### 2. Move dcp_local_seq_lens computation to utils ⏎ Move dcp_local_seq_lens computation to utils and pass it by metadata, so other attn backends can reuse it. ⏎  ⏎ ## Test Plan ⏎ Model: DeepSeek-V2-Lite-Chat ⏎ Dataset: gsm8k ⏎ ```python ⏎ vllm serve DeepSeek-V2-Lite-Chat --gpu-memory-utilization 0.9 --tensor-parallel-size 2 --decode-context-parallel-size 2 --cp-kv-cache-interleave-size 64 ⏎ ``` ⏎  ⏎ ## Test Result ⏎ tp2 dcp2, original code ⏎ | dataset | version | metric | mode | vllm-api-stream-chat | ⏎ |----- | ----- | ----- | ----- | -----| ⏎ | gsm8k | 7cd45e | accuracy | gen | 67.85 | ⏎  ⏎ tp2 dcp2, interleave_size = 1 ⏎ | dataset | version | metric | mode | vllm-api-stream-chat | ⏎ |----- | ----- | ----- | ----- | -----| ⏎ | gsm8k | 7cd45e | accuracy | gen | 67.85 | ⏎  ⏎ tp2 dcp2, interleave_size = 64 ⏎ | dataset | version | metric | mode | vllm-api-stream-chat | ⏎ |----- | ----- | ----- | ----- | -----| ⏎ | gsm8k | 7cd45e | accuracy | gen | 67.55 | ⏎  ⏎ --- ⏎ [details omitted]

### L3-f080a83511  (L3, 2025-11-10, sha f080a8351151, PR #24490)
TITLE: [RFC][ROCm][AITER] Keep all AITER kernels in `_aiter_ops` class like `_custom_ops` and `_ipex_ops` (#24490)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.mla.rocm_aiter,L3.platform.rocm_selection; Centralizes AITER kernel registration and availability through _aiter_ops for MLA.
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.common_v1, L3.mla.rocm_aiter, L3.platform.rocm_selection
FILES: vllm/attention/ops/rocm_aiter_mla.py (+0/-105); vllm/v1/attention/backends/mla/common.py (+21/-34); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+2/-7); docs/design/moe_kernel_features.md (+1/-1); tests/kernels/moe/test_moe.py (+6/-5); tests/model_executor/test_enabled_custom_ops.py (+14/-27); vllm/_aiter_ops.py (+941/-0); vllm/envs.py (+4/-4); vllm/model_executor/layers/fused_moe/fused_moe.py (+7/-8); vllm/model_executor/layers/fused_moe/layer.py (+43/-40); vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py (+17/-312); vllm/model_executor/layers/layernorm.py (+21/-69); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+3/-9); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py (+2/-2); vllm/model_executor/layers/quantization/fp8.py (+6/-10); vllm/model_executor/layers/quantization/kernels/scaled_mm/aiter.py (+3/-45); vllm/model_executor/layers/quantization/quark/quark_moe.py (+15/-30); vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py (+7/-0); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+35/-89); vllm/model_executor/layers/quantization/utils/w8a8_utils.py (+1/-1); vllm/model_executor/layers/rotary_embedding/base.py (+5/-8); vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py (+9/-0); vllm/model_executor/layers/rotary_embedding/rocm_aiter_rope_ops.py (+0/-94); vllm/model_executor/models/deepseek_v2.py (+12/-15); vllm/platforms/rocm.py (+18/-9)
LABELS: documentation, rocm, ready, v1, deepseek
BODY: ## Purpose ⏎  ⏎ This PR introduces `_aiter_ops.py` as proposed in the [RFC here](https://github.com/vllm-project/vllm/issues/21504). The `aiter_ops` namespace provides several key benefits: ⏎  ⏎ - Centralized kernel registration: Ensures that kernels from the aiter package are properly registered ⏎  ⏎ - Environment availability checks: Encapsulates aiter support detection and environment compatibility validation ⏎  ⏎ - Reduced code duplication: Eliminates the need for duplicate helper functions, namely checking device compability and environment varible enability checks across different vLLM modules. ⏎  ⏎ This implementation establishes the foundation for future refactoring efforts, where existing kernels throughout the vLLM repository will be migrated to use this unified approach for better maintainability and consistency. ⏎  ⏎ This PR uses [5ee37dce](https://github.com/ROCm/aiter/commit/5ee37dced6f1bde0229b2c77ce079433549aa25f) commit from `aiter` repo. ⏎  ⏎ ## Test Plan ⏎  ⏎ Test models that are afftected by this change, using lm_eval on gsm8k dataset. ⏎  ⏎ **environment setting** ⏎  ⏎ Step 1: run vllm serve ⏎  ⏎ ` ⏎ VLLM_USE_V1=1 \ ⏎ VLLM_ROCM_USE_AITER=1 \ ⏎ SAFETENSORS_FAST_GPU=1 \ ⏎ VLLM_DISABLE_COMPILE_CACHE=1 \ ⏎ vllm serve $MODEL_NAME --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1,2,4,8,16,24,32]}'  --trust-remote-code --swap-space 16 --distributed-executor-backend mp` ⏎  ⏎ Step 2: run lm_eval ⏎  ⏎ `lm_eval --model local-completions ⏎ --tasks gsm8k ⏎ --model_args model=$MODEL_NAME,base_url=http://localhost:8000/v1/completions ⏎ --trust_remote_code ⏎ --num_fewshot 5 ⏎ --batch_size 256` ⏎  ⏎ ## Test Results ⏎  ⏎ ### deepseek-ai/DeepSeek-V3 -tp 8 --block-size 1  --max-model-len 32768 --max_seq_len_to_capture 32768  ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9500|±  | 0.006| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.9507|±  | 0.006| ⏎  ⏎ ### meta-llama/Llama-4-Scout-17B-16E-Instruct -tp 8 --max-model-len 8192  ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9174|±  |0.0076| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.8999|±  |0.0083| ⏎  ⏎ ### meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 -tp 8 --max-model-len 8192  ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---| …[truncated]
