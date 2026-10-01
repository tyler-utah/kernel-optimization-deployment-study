### L3-467bef18a3  (L3, 2025-06-10, sha 467bef18a353, PR #19134)
TITLE: [BugFix][FlashInfer] Fix attention backend interface mismatch with unexpected keyword `use_irope` (#19134)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+5/-0)
LABELS: bug, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Fix the interface mismatch for FlashInfer attention backend for `use_irope`. ⏎  ⏎ Error: ⏎  ⏎ ``` ⏎ (VllmWorker rank=5 pid=2407942) ERROR 06-03 14:44:28 [multiproc_executor.py:486]   File "/home/guorachel/venv/vllm/vllm/model_executor/models/llama4.py", line 262, in __init__ ⏎ (VllmWorker rank=5 pid=2407942) ERROR 06-03 14:44:28 [multiproc_executor.py:486]     self.self_attn = Llama4A …[truncated]

### L3-c7ea0b56cd  (L3, 2025-06-11, sha c7ea0b56cd9a, PR #17331)
TITLE: [AMD] [Quantization] Add override flag for attention dtype instead of using kv_cache_dtype trigger (#17331)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+9/-1); vllm/config.py (+8/-0); vllm/engine/arg_utils.py (+4/-0)
LABELS: rocm, ready
BODY: This adds a flag to use override dtype  in VllmConfig instead of using the kv_cache_dtype flag so any FP8 model will work instead of just those with fp8 kv cache

### L3-2f1c19b245  (L3, 2025-06-11, sha 2f1c19b2456d, PR #18711)
TITLE: [CI] change spell checker from codespell to typos (#18711)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: csrc/cpu/attention.cpp (+3/-3); vllm/attention/backends/utils.py (+2/-2); .gitignore (+1/-1); .pre-commit-config.yaml (+3/-5); csrc/cpu/cpu_types_x86.hpp (+5/-5); csrc/moe/moe_permute_unpermute_op.cu (+8/-8); csrc/moe/topk_softmax_kernels.cu (+3/-3); csrc/moe/torch_bindings.cpp (+1/-1); csrc/quantization/machete/machete_mainloop.cuh (+3/-3); csrc/rocm/skinny_gemms.cu (+7/-7); (+48 more)
LABELS: documentation, rocm, frontend, tpu, speculative-decoding, ready, v1, multi-modality, tool-calling, llama
BODY: Currently, codespell can not help in finding all possible typos. According to [this comparison](https://github.com/crate-ci/typos/blob/master/docs/comparison.md), it seems typos has more good performance and correctness. typos also supports both pre-commit and Github actions.

### L3-97a9465bbc  (L3, 2025-06-11, sha 97a9465bbca1, PR #19501)
TITLE: [UX] Add Feedback During CUDAGraph Capture (#19501)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_model_runner.py (+4/-1)
LABELS: ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Add feedback during one of the longer parts of CUDAGraph capture to show progress during a long startup phase ⏎  ⏎ ## Test Plan ⏎  ⏎ - Run a model  ⏎  ⏎ ```bash ⏎ vllm serve $MODEL ⏎ ``` ⏎  ⏎ ## Test Result ⏎  ⏎ - View output. We now see this during cuda graph capture: ⏎  ⏎ ```bash ⏎ Capturing CUDA graphs: 100%|███████████████████████████████████████████████████████████████████████████████████████████ …[truncated]

### L3-497a91e9f7  (L3, 2025-06-11, sha 497a91e9f77b, PR #19297)
TITLE: [CI] Update FlashInfer to 0.2.6.post1 (#19297)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+17/-15)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ Update to the latest stable release of [FlashInfer](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.2.6). This is the first stable release with Blackwell support, so fairly important to solidify on. However there are not pre-built wheels yet. We can wait to see if wheels will be published, or build our own. @huydhn could you help me with this? ⏎  ⏎ I updated the instructions in the dockerfile to match the new method for buildi …[truncated]

### L3-04a55612dd  (L3, 2025-06-12, sha 04a55612dd6f, PR #19486)
TITLE: [Misc] Fix  misleading ROCm warning (#19486)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.flash_attention_rocm
FILES: vllm/attention/ops/triton_flash_attention.py (+6/-1)
LABELS: rocm, ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Importing [rocm](https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/attention/ops/triton_flash_attention.py#L28 ) causes the following log output, and this PR is to avoid this. ⏎ ```shell ⏎ WARNING 06-11 08:09:14 [rocm.py:28] Failed to import from amdsmi with ModuleNotFoundError("No module named 'amdsmi'") ⏎ WARNING 06-11 08:09:14 [rocm.py:39] Failed to import from vllm._rocm …[truncated]

### L3-f98548b9da  (L3, 2025-06-12, sha f98548b9da39, PR #16756)
TITLE: [torch.compile][ROCm] Fuse quantization onto attention using a torch.compile pass (#16756)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.rocm_flash_attn_v0, L3.mla.triton_v0, L3.mla.common_v1, L3.dispatch.abstract_interface, L3.flex_attention, L3.blocksparse.v0
FILES: vllm/_custom_ops.py (+7/-1); vllm/attention/backends/abstract.py (+17/-0); vllm/attention/backends/blocksparse_attn.py (+6/-0); vllm/attention/backends/dual_chunk_flash_attn.py (+9/-0); vllm/attention/backends/flash_attn.py (+6/-0); vllm/attention/backends/flashinfer.py (+6/-0); vllm/attention/backends/hpu_attn.py (+6/-0); vllm/attention/backends/ipex_attn.py (+6/-0); vllm/attention/backends/mla/common.py (+6/-0); vllm/attention/backends/pallas.py (+6/-0); (+23 more)
LABELS: rocm, tpu, ready, ci/build, v1
BODY: This PR implements the fusion of fp8 quantization onto attention, described in #16220. It performs this fusion using a new `AttnFusionPass`, which uses the pattern matcher and only performs the fusion if the backend supports it. It is currently off by default, pending more robust V1 support and performance measurement. ⏎  ⏎ This PR also makes the following changes: ⏎ - `output_scale` added as a parameter to `unified_attention_with_output`. During the ` …[truncated]

### L3-af09b3f0a0  (L3, 2025-06-12, sha af09b3f0a054, PR #19492)
TITLE: [Bugfix][V1] Allow manual FlashAttention for Blackwell (#19492)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+13/-4)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ In the previous PR to use FlashInfer by default (https://github.com/vllm-project/vllm/pull/19118), this inadventantly prevents FlashAttention from being used if FlashInfer is installed since we don't have an explicit case to check for the selected_backend to be FA. ⏎  ⏎ ## Test Plan ⏎  ⏎ Test locally on a B200 ⏎  ⏎ ## Test Result ⏎  ⏎ Before (main): ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm serve meta-llama/Llama-3.1-8B-Instruct ⏎ ... ⏎ INFO 06-11 10:42 …[truncated]

### L3-3597b06a4f  (L3, 2025-06-13, sha 3597b06a4faa, PR #18581)
TITLE: [CUDA] Enable full cudagraph for FlashMLA (#18581)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.rocm_aiter, L3.dispatch.abstract_interface, L3.flex_attention
FILES: vllm/compilation/cuda_piecewise_backend.py (+5/-1); vllm/v1/attention/backends/cpu_attn.py (+8/-4); vllm/v1/attention/backends/flash_attn.py (+18/-11); vllm/v1/attention/backends/flashinfer.py (+7/-4); vllm/v1/attention/backends/flex_attention.py (+9/-13); vllm/v1/attention/backends/mla/common.py (+36/-8); vllm/v1/attention/backends/mla/flashmla.py (+31/-3); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+1/-1); vllm/v1/attention/backends/utils.py (+71/-2); vllm/v1/worker/gpu_model_runner.py (+93/-67); (+7 more)
LABELS: rocm, structured-output, frontend, ready, v1, llama
BODY: Enable fullgraph CUDAGraph capture for the FlashMLA decode case. ⏎  ⏎ Hacks: ⏎ - building the capture metadata ⏎ - prefill batch bypasses compiled code and manually calls eager code ⏎  ⏎ Tested with: ⏎ ``` ⏎ python examples/offline_inference/basic/generate.py --model deepseek-ai/DeepSeek-V2-Lite --trust-remote-code -O {"full_cuda_graph":true} ⏎ ```

### L3-055915e6ce  (L3, 2025-06-15, sha 055915e6ce0b, PR #19617)
TITLE: Enable prefix caching with full cuda graphs (#19617)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config.py (+0/-1)
LABELS: documentation, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1, llama
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Currently, vLLM silently disables prefix caching when using full CUDA graphs. However, prefix caching should be supported with full cuda graphs already. This PR enables it back. ⏎  ⏎ ## Test Plan ⏎  ⏎ Locally tested `tests/compile/piecewise/test_full_cudagraph.py` and it passed the test. ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-0b73736a0d  (L3, 2025-06-15, sha 0b73736a0d86, PR #19339)
TITLE: [Kernel] Raise verbose error and consolidate `num_heads/num_kv_heads` divisibility check (#19339)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.rocm_flash_attn_v0, L3.flex_attention, L3.blocksparse.v0
FILES: vllm/attention/backends/blocksparse_attn.py (+1/-3); vllm/attention/backends/dual_chunk_flash_attn.py (+0/-1); vllm/attention/backends/flash_attn.py (+0/-1); vllm/attention/backends/flashinfer.py (+0/-1); vllm/attention/backends/hpu_attn.py (+0/-1); vllm/attention/backends/ipex_attn.py (+0/-1); vllm/attention/backends/pallas.py (+1/-2); vllm/attention/backends/rocm_flash_attn.py (+0/-1); vllm/attention/backends/torch_sdpa.py (+0/-1); vllm/attention/backends/xformers.py (+0/-1); (+7 more)
LABELS: rocm, tpu, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Encountered this divisibility assertion error during debugging but error message was not verbose. This PR raises an error specifying the values of `num_heads` and `num_kv_heads`. Also taking this chance to consolidate the check at a central place. ⏎  ⏎ ## Test Plan ⏎ ```bash ⏎ pytest tests/kernels/attention/test_attention.py -k 'test_num_heads_not_divisble_by_num_kv_heads' ⏎ ``` ⏎  ⏎ ## …[truncated]

### L3-c6703d1e0d  (L3, 2025-06-15, sha c6703d1e0d48, PR #19609)
TITLE: [MISC] Remove unused variableds in C++ (#19609)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.rocm.custom_paged
FILES: csrc/attention/paged_attention_v1.cu (+1/-4); csrc/attention/paged_attention_v2.cu (+1/-4); csrc/rocm/attention.cu (+0/-20); csrc/prepare_inputs/advance_step.cu (+0/-1); csrc/quantization/fp8/amd/quant_utils.cuh (+0/-2); csrc/quantization/gptq/q_gemm.cu (+0/-8)
LABELS: rocm, ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Some C++ code contain unused definition / var. We used [[maybe_unused]] to mark them. Now clean up ⏎  ⏎ ## Test Plan ⏎ AMD: build from scratch ⏎ Nvidia: build from scratch ⏎ Also CI ⏎  ⏎ ## Test Result ⏎ AMD built successfully ⏎ Nvidia built successfully ⏎ CI built successfully. ⏎  ⏎ ## (Optional) Documentation Update ⏎ N/A

### L3-a77aea59fd  (L3, 2025-06-16, sha a77aea59fd2f, PR #19620)
TITLE: [TPU] support attention head dim smaller than 128 (#19620)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+28/-7); tests/v1/tpu/test_basic.py (+37/-0)
LABELS: tpu, ready, v1
BODY: ## Purpose ⏎ To support models whose head dim is smaller than 128 on TPU. ⏎ Note: for head_dim which is a multiply of 128, we will support it in a separate PR. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest -s -v tests/v1/tpu/test_basic.py::test_phi3 ⏎ ``` ⏎  ⏎ ## Test Result ⏎ Passed.

### L3-1173804dca  (L3, 2025-06-16, sha 1173804dca83, PR #19657)
TITLE: [Bugfix] Fix TP inference for Flex attention backend (#19657)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+7/-1); vllm/v1/worker/gpu_worker.py (+5/-0); vllm/v1/worker/tpu_worker.py (+5/-0); tests/v1/engine/test_engine_core.py (+35/-1); vllm/v1/engine/core.py (+2/-0)
LABELS: tpu, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ - Flex Attention doesn't work with tensor parallel currently because `num_gpu_blocks` is not updated in `cache_config` properly: ⏎ ``` ⏎ (VllmWorker rank=1 pid=2540) ERROR 06-15 05:58:54 [multiproc_executor.py:527]   File "/kaggle/working/vllm/vllm/v1/worker/gpu_model_runner.py", line 1211, in execute_model ⏎ (VllmWorker rank=1 pid=2540) ERROR 06-15 05:58:54 [multiproc_executor. …[truncated]

### L3-ddfed314f9  (L3, 2025-06-17, sha ddfed314f9c3, PR #19712)
TITLE: Fixes IMA for TP w/ flex-attention (#19712)
SOURCES: path_core
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+2/-8); tests/kernels/test_flex_attention.py (+0/-2)
LABELS: ready, v1
BODY: # FlexAttention Backend Fix ⏎  ⏎ So I have been using this file for testing: https://gist.github.com/drisspg/3050c61f587030f09b96d86e14b10711 ⏎ I am on the latest PyTorch Nightly, and I found that it it is working even before this fix: ⏎  ⏎ So I am not sure, that being said people have ran into the create-block mask problem before w/ compile so this was a mistake on my end ⏎ ```Shell ⏎ INFO 06-16 13:44:24 [kv_cache_utils.py:720] Maximum concurrency for 32,768  …[truncated]

### L3-4c8f64faa7  (L3, 2025-06-17, sha 4c8f64faa741, PR #19280)
TITLE: [V1][Kernel] Flashinfer HND KV cache layout (#19280)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/flashinfer.py (+1/-3); vllm/envs.py (+11/-0); vllm/v1/attention/backends/flash_attn.py (+5/-7); vllm/v1/attention/backends/flashinfer.py (+21/-6); vllm/v1/attention/backends/utils.py (+21/-0); vllm/distributed/kv_transfer/kv_connector/utils.py (+5/-4)
LABELS: ready, v1
BODY: Follow up PR to https://github.com/vllm-project/vllm/pull/18775, again porting over functionality from V0 (ref https://github.com/vllm-project/vllm/pull/16605). ⏎  ⏎ This PR will enable the use of FlashInfer with a HND cache layout in V1.  ⏎ Among the most immediate benefits, this PR is a prerequisite to enabling heterogeneous TP support for disaggregated prefill-decode setup, optimizing the layout for xfers. ⏎  ⏎ Test with: ⏎ ``` ⏎ FLASHINFER_KV_CACHE_LAYOUT= …[truncated]

### L3-ccd7c05089  (L3, 2025-06-17, sha ccd7c050898c, PR #19152)
TITLE: [Kernel] Add Split-KV Support to Unified Triton Attention Kernel (#19152)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+456/-52)
LABELS: ready
BODY: In this PR, we introduce performance enhancements to the triton unified attention kernel #16828 by adding a split-KV variant that also parallelizes across the sequence (context) dimension. This approach provides a clear advantage over the current upstream implementation in scenarios involving small batch sizes and long sequences. ⏎  ⏎ This initial version utilizes a simple heuristic to dynamically select between the original and the split-KV kernel v …[truncated]

### L3-07334959d8  (L3, 2025-06-17, sha 07334959d810, PR #19336)
TITLE: [Wheel Size] Only build FA2 8.0+PTX (#19336)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ Reduce wheel size by only building FA2 8.0+PTX instead of 8.0,9.0,10.0 etc. ⏎  ⏎ 376.35 MB -> 339.32 MB ⏎  ⏎ This does cause a slowdown in the initial runs on a machine while the PTX is JIT compiled ⏎  ⏎ ``` ⏎ (vllm) lwilkinson@gpu66:~/code/vllm$ vllm bench throughput --model RedHatAI/Meta-Llama-3.1-8B-FP8 --load-format dummy --input-len 10000 --output-len 200 --num-prompts 100 ⏎ ... ⏎ Throughput: 1.27 requests/s, 12954.78 total tokens/s, 254.02 output …[truncated]

### L3-a44b1c951d  (L3, 2025-06-17, sha a44b1c951df9, PR #19158)
TITLE: [Feature][ROCm] Add full graph capture support for TritonAttentionBackend (#19158)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.triton.v1_backend, L3.dispatch.abstract_interface, L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+3/-2); vllm/v1/attention/backends/flash_attn.py (+3/-169); vllm/v1/attention/backends/triton_attn.py (+159/-7); vllm/v1/attention/backends/utils.py (+168/-0); tests/compile/piecewise/test_full_cudagraph.py (+1/-0)
LABELS: rocm, ready, v1
BODY: This PR adds full graph capture for TritonAttentionBackend. ⏎  ⏎ - add exemption for TritonAttentionBackend in model runner. ⏎ - Avoid requirement for aot_scheduling in metadata build function by overwirte the build function and __init__ function.

### L3-c53711bd63  (L3, 2025-06-17, sha c53711bd63db, PR #19696)
TITLE: [MISC] correct copy_blocks src_to_dists param type (#19696)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/ipex_attn.py (+2/-2)
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Type fix for `copy_blocks` parameter `src_to_dists`. ⏎ ## Test Plan ⏎ NA ⏎ ## Test Result ⏎ NA ⏎ ## (Optional) Documentation Update

### L3-dac8cc49f4  (L3, 2025-06-17, sha dac8cc49f43f, PR #19706)
TITLE: [TPU] Update torch version to include paged attention kernel change (#19706)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: requirements/tpu.txt (+5/-5)
LABELS: ready, ci/build
BODY: 

### L3-f04d604567  (L3, 2025-06-18, sha f04d60456792, PR #19784)
TITLE: [Minor] Zero-initialize attn output buffer (#19784)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+1/-1)
LABELS: ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Currently, we use `torch.empty` for initializing the attention output buffer. This could cause a numerical issue in the initial memory profiling run, because all the subsequent operators get uninitialized inputs that could contain NaNs. This PR fixes this by using `torch.zeros` instead. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-8b6e1d639c  (L3, 2025-06-18, sha 8b6e1d639c66, PR #18596)
TITLE: [Hardware][AMD] integrate aiter chunked prefill into vllm (#18596)
SOURCES: path_core, symbol_pickaxe, release_notes, corpus:production-kernel-provenance
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.rocm.aiter_fa, L3.platform.rocm_selection
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+585/-0); vllm/envs.py (+8/-0); vllm/platforms/rocm.py (+9/-3)
LABELS: documentation, rocm, ready, ci/build, v1
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:2de12be428 (fix PR 18990)
BODY: CMD: VLLM_TORCH_PROFILER_DIR=/mnt/raid0/sixifang/vllm/vllm_profile HIP_VISIBLE_DEVICES=4,5,6,7 VLLM_ROCM_USE_AITER=1 VLLM_USE_V1=1 vllm serve /models/models--amd--Meta-Llama-3.1-8B-Instruct-FP8-KV/snapshots/fa42f9a9105c545755fea25cf69f49ac8c8b40e1/ --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --trust-remote-code --disable-log-requests --block-size 16 --max-model-len 32768 --dtype float16 --quantization fp8 --no-enable-prefix-caching --ma …[truncated]

### L3-a89209b78d  (L3, 2025-06-18, sha a89209b78de0, PR #19327)
TITLE: [v1] Support mamba2 (#19327)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: tests/models/language/generation/test_hybrid.py (+42/-11); tests/v1/test_oracle.py (+1/-1); vllm/engine/arg_utils.py (+6/-1); vllm/model_executor/layers/mamba/mamba_mixer2.py (+175/-60); vllm/model_executor/models/mamba2.py (+33/-21); vllm/v1/attention/backends/mamba_attn.py (+192/-0); vllm/v1/core/single_type_kv_cache_manager.py (+42/-1); vllm/v1/kv_cache_interface.py (+24/-0); vllm/v1/worker/gpu_model_runner.py (+68/-26)
LABELS: ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ This PR adds the initial support for mamba2 in v1. Difference with v0: ⏎ 1. Don't need a separate MambaCacheManager. Instead, we reuse the KVCacheManager and implemen necessary customizations by a new SingleTypeKVCacheManager. ⏎ 2. Wrap all input preparation logic into a new attention backend. ⏎ 3. Put decode prompts before prefill prompts as v1 persistent batch prefers decode  …[truncated]

### L3-04fefe7c9a  (L3, 2025-06-18, sha 04fefe7c9a79, PR #19813)
TITLE: [TPU] Update torch-xla version to include paged attention tuned block change (#19813)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: requirements/tpu.txt (+5/-5)
LABELS: ready, ci/build, qwen
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Update torch-xla version to include paged attention tuned block change ⏎  ⏎ ## Test Plan ⏎ Run vLLM benchmark with new package and compare with old ⏎  ⏎ comparing with hourly run. ⏎  ⏎ ## Test Result ⏎  ⏎ | MODEL                                                              | Device | 20250618_100001 | with new package| ⏎ |--------------------------------------------------------------------|-- …[truncated]

### L3-36239f79dd  (L3, 2025-06-19, sha 36239f79dd35, PR #19781)
TITLE: Fix FA2 fallback for Blackwell V1 (#19781)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+1/-1)
LABELS: bug, ready
BODY: `elif` was not the right choice, as we want FA2 to be a constant fallback on V1 so we can use it for Blackwell when FlashInfer isn't installed

### L3-aa20d10a91  (L3, 2025-06-19, sha aa20d10a9182, PR #19803)
TITLE: [Misc] [ROCm] Prevent surplus tensor reshape (#19803)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+1/-1)
LABELS: rocm, ready, v1
BODY: In case of ROCm, there is no need to reshape.

### L3-71d1219545  (L3, 2025-06-20, sha 71d1219545b5, PR #19745)
TITLE: [Kernel] correct cpu worker function parameter type (#19745)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/ipex_attn.py (+1/-1); vllm/worker/cpu_worker.py (+4/-4)
LABELS: ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ The type hints for `src_to_dst` and `src_to_dsts` should be `torch.Tensor`. ⏎ ## Test Plan ⏎ NA ⏎ ## Test Result ⏎ NA ⏎ ## (Optional) Documentation Update

### L3-e3a3e4db46  (L3, 2025-06-20, sha e3a3e4db463d, PR #19822)
TITLE: [Bugfix] Enable PP with AITER+V1 (#19822)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+3/-10); vllm/model_executor/layers/layernorm.py (+0/-1); vllm/model_executor/models/qwen2_5_omni_thinker.py (+5/-5); vllm/model_executor/models/qwen2_5_vl.py (+5/-5); vllm/model_executor/models/qwen2_vl.py (+5/-5)
LABELS: rocm, ready, v1, qwen
BODY: ## Purpose ⏎ Enable Pipeline Parallelism with AITER + V1.  ⏎ 1. fixed an AITER MLA setting error; ⏎ 2. enabled AITER rmsnorm for V1 (reverted because the current version doesn't work with some models so we will add some extra changes from a separate PR) ⏎  ⏎ ## Problem resolved ⏎ this command:  ⏎ VLLM_ROCM_USE_AITER=1  VLLM_ROCM_USE_AITER_RMSNORM=0  VLLM_USE_V1=1 vllm serve /models/DeepSeek-R1/ -pp 8 -tp 1 --block-size 1 --max-model-len 32768 --disable-log-req …[truncated]

### L3-71baf85ae1  (L3, 2025-06-20, sha 71baf85ae11b, PR #19749)
TITLE: [Kernel] mark TorchSDPABackend swap_blocks NotImplementedError (#19749)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/torch_sdpa.py (+1/-1)
LABELS: ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ `TorchSDPABackend` is only used with cpu device and for now and cpu does not support cache swap operations as stated in: ⏎ https://github.com/vllm-project/vllm/blob/5a1c2e15d847e30d9c72d60ba1e28dc4b89df23d/vllm/worker/cpu_worker.py#L91-L95 ⏎  ⏎ So, `TorchSDPABackend` can raise `NotImplementedError`. ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-e6327c9b3e  (L3, 2025-06-23, sha e6327c9b3eb2, PR #19181)
TITLE: [Feature] Support sequence parallelism for static fp8 quantization (#19181)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/test_sequence_parallelism.py (+144/-17); tests/distributed/test_sequence_parallel.py (+52/-56); tests/models/registry.py (+2/-1); vllm/compilation/fusion.py (+2/-2); vllm/compilation/pass_manager.py (+4/-4); vllm/compilation/sequence_parallelism.py (+328/-114); vllm/config.py (+2/-4)
LABELS: ready
BODY: Add support sequence parallelism for static fp8 quantization in this PR. ⏎ It requires below config to enable it ⏎ ``` ⏎ config = CompilationConfig(level=3, ⏎                            splitting_ops=[], ⏎                            compile_sizes=[4], ⏎                            custom_ops=["+rms_norm"]) ⏎  ⏎ # enable_noop is required to be True for correct sp pattern match  ⏎ config.pass_config.enable_noop = True ⏎ config.pass_config.enable_sequence_parallelism =  …[truncated]

### L3-a045b7e89a  (L3, 2025-06-24, sha a045b7e89a24, PR #19463)
TITLE: [Perf] Improve/Fix-regression for FA3 in High QPS regimes (#19463)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); test-qwen (+1/-0)
LABELS: ready, ci/build, qwen
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ This PR is meant to address spending excessive time in the combine phase of FA3; findings from https://github.com/vllm-project/vllm/issues/18619 . The associated vllm-flash-attn PR is: https://github.com/vllm-project/flash-attention/pull/70 see that for more details (that PR must also land first). ⏎  ⏎ ### Perf Results ⏎  ⏎ Setup ⏎ ``` ⏎ vllm serve <model> --disable-log-requests --ma …[truncated]

### L3-0d06b533a0  (L3, 2025-06-24, sha 0d06b533a0fc, PR #20032)
TITLE: cmake: Update vllm_flash_attn for vllm_kernels (#20032)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Some stepping stones for https://github.com/vllm-project/vllm/issues/17419, ⏎  ⏎ This resolves an import issue that I ran into while testing the package refactor. ⏎  ⏎ Updates vllm_flash_attn to latest main, which includes 2 commits: ⏎ * vllm_flash_attn: Setup for vllm_kernels package ⏎ * varlen combine scheduler ⏎  ⏎ Full diff: ⏎ https://github.com/vllm-project/flash-attention/compare/763 …[truncated]

### L3-879f69bed3  (L3, 2025-06-25, sha 879f69bed375, PR #20023)
TITLE: [Refactor] Remove duplicate `ceil_div` (#20023)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/nki_flash_attn.py (+6/-9); benchmarks/cutlass_benchmarks/w8a8_benchmarks.py (+3/-8); tests/kernels/attention/test_mla_decode_cpu.py (+1/-4); tests/kernels/attention/test_triton_decode_attention.py (+1/-4); tests/neuron/1_core/test_prefix_prefill.py (+4/-5); vllm/model_executor/layers/fused_moe/moe_align_block_size.py (+2/-6); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+3/-6)
LABELS: performance, ready
BODY: ## Purpose ⏎  ⏎ Remove duplicate `ceil_div`

### L3-0f9e7354f5  (L3, 2025-06-25, sha 0f9e7354f508, PR #20057)
TITLE: [BugFix] Fix full-cuda-graph illegal memory access in FA3 (#20057)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+7/-18)
LABELS: ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Fix a sporadic illegal memory access discovered by @WoosukKwon   ⏎  ⏎ ## Test Plan ⏎  ⏎ lm-eval + @WoosukKwon repro attempt ⏎  ⏎ ## Test Result ⏎  ⏎ after 1.5h @WoosukKwon has yet to repo ⏎  ⏎ ``` ⏎ lm_eval   --model vllm   --model_args '{"pretrained":"Qwen/Qwen2.5-VL-72B-Instruct","tensor_parallel_size":8,"compilation_config":{"full_cuda_graph":true}}'   --tasks gsm8k   --batch_size auto ⏎ ... ⏎  …[truncated]

### L3-2d7620c3eb  (L3, 2025-06-25, sha 2d7620c3ebb3, PR #19919)
TITLE: [TPU] Add TPU specific var VLLM_TPU_MOST_MODEL_LEN (#19919)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/v1/attention/backends/pallas.py (+5/-0); tests/v1/tpu/worker/test_tpu_model_runner.py (+14/-0); vllm/envs.py (+3/-0); vllm/platforms/tpu.py (+0/-10); vllm/v1/worker/tpu_model_runner.py (+163/-67)
LABELS: tpu, ready, v1
BODY: ## Purpose ⏎  ⏎ Add TPU specific environment variable: VLLM_TPU_MOST_MODEL_LEN. This is used to pass in the request length for most of requests, comparing with the existing variable`max_model_len` for the length of longest requests. This will benefit when `most_model_len` is much longer than `most_model_len`, and a large portion of requests fall inside `most_model_len`, such as 1% requests are 32k for `most_model_len` and 99% requests are 2k for `mos …[truncated]

### L3-2cc2069970  (L3, 2025-06-25, sha 2cc206997012, PR #20048)
TITLE: [TPU][Bugfix] fix kv cache padding (#20048)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+1/-7); vllm/v1/worker/tpu_worker.py (+13/-2)
LABELS: tpu, ready, v1
DEEP_STUDY: deep-study correctness case vllm:2cc2069970: class=shape_alignment_edge; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎  ⏎ Fix the bug when padding head_size for TPU. Previously block_num is only changed by `get_kv_cache_shape`. It's not enough since we still need to modify the value in `KVCacheConfig` as well. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ vllm serve microsoft/Phi-3-mini-128k-instruct --seed 42 --disable-log-requests --gpu-memory-utilization 0.95 --max-num-batched-tokens 2048 --max-num-seqs 128 --tensor-parallel-size 1 --max-model-len 2048 --no-enable-prefix-caching …[truncated]

### L3-b69781f107  (L3, 2025-06-26, sha b69781f107b7, PR #19560)
TITLE: [Hardware][Intel GPU] Add v1 Intel GPU support with Flash attention backend. (#19560)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.upstream_pip, L3.flash_attn.fa_utils
FILES: docker/Dockerfile.xpu (+1/-0); requirements/xpu.txt (+1/-0); vllm/attention/utils/fa_utils.py (+14/-1); vllm/platforms/xpu.py (+70/-34); vllm/v1/attention/backends/flash_attn.py (+5/-7); vllm/v1/worker/xpu_model_runner.py (+32/-0); vllm/v1/worker/xpu_worker.py (+164/-0); .buildkite/scripts/hardware_ci/run-xpu-test.sh (+1/-0); vllm/_ipex_ops.py (+105/-0); vllm/executor/ray_distributed_executor.py (+1/-1)
LABELS: ready, ci/build, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ previous PR is #14612  ⏎ this PR add Intel GPU V1 engine support. This PR includes ⏎ a. integrates `flash_attn_varlen_func` kernel(V2) on xpu which is implemented in ipex, ⏎ b. Refine some xpu related configs ⏎ c. Support vLLM V1 by adding xpu_worker/xpu_model_runner to handle xpu code path ⏎ d. Docker file/ dependency updates. ⏎ e. Add a V1 test in CI. ⏎  ⏎ ## Test Plan ⏎ Add a V1 test in  …[truncated]

### L3-04e1642e32  (L3, 2025-06-26, sha 04e1642e3251, PR #19928)
TITLE: [TPU] add kv cache update kernel (#19928)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/pallas_kv_cache_update.py (+117/-0); vllm/v1/attention/backends/pallas.py (+50/-5); .buildkite/scripts/hardware_ci/run-tpu-v1-test.sh (+2/-0); tests/v1/tpu/test_kv_cache_update_kernel.py (+71/-0); tests/v1/tpu/test_pallas.py (+2/-1); vllm/v1/worker/tpu_model_runner.py (+100/-32)
LABELS: tpu, ready, ci/build, v1
BODY: ## Purpose ⏎  ⏎ TPU is not good at scatter-update. Here consecutive new kv status will be updated together with the help of the kv cache update kernel. ⏎  ⏎ ## Test Plan ⏎  ⏎ Kernel test: pytest -s -v tests/v1/tpu/test_kv_cache_update_kernel.py ⏎ Accuracy test: pytest -s -v tests/entrypoints/llm/test_accuracy.py::test_lm_eval_accuracy_v1_engine ⏎  ⏎ ## Test Result ⏎  ⏎ Passed.

### L3-27c065df50  (L3, 2025-06-26, sha 27c065df5040, PR #19904)
TITLE: [Bugfix][V1][ROCm] Fix AITER Flash Attention Backend (Fix API Break and Local Attention Logic: affecting Llama4) (#19904)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/attention/layer.py (+9/-5); vllm/v1/attention/backends/rocm_aiter_fa.py (+37/-18)
LABELS: rocm, ready, v1, llama
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ This is to fix this issue https://github.com/vllm-project/vllm/issues/19867 ⏎ PR https://github.com/vllm-project/vllm/pull/18212 introduced cross-layer kvcache ⏎ This PR https://github.com/vllm-project/vllm/pull/18212 (which  introduced cross-layer kvcache) has introduced a new argument to `AttentionImpl` init function (`kv_sharing_target_layer_name: Optional[str] = None,`) ⏎  ⏎ # …[truncated]

### L3-0740e29b66  (L3, 2025-06-26, sha 0740e29b66ca, PR #19744)
TITLE: [Feature] add quick all reduce (#19744)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.flashinfer.trtllm_gen
FILES: CMakeLists.txt (+8/-0); csrc/custom_quickreduce.cu (+114/-0); csrc/ops.h (+11/-0); csrc/quickreduce/base.h (+338/-0); csrc/quickreduce/quick_reduce.h (+196/-0); csrc/quickreduce/quick_reduce_impl.cuh (+698/-0); csrc/torch_bindings.cpp (+18/-0); tests/distributed/test_quick_all_reduce.py (+138/-0); vllm/_custom_ops.py (+32/-0); vllm/distributed/device_communicators/cuda_communicator.py (+20/-2); (+2 more)
LABELS: rocm, ready, ci/build, qwen
BODY: Just For ROCM ⏎ 1.Add [quickreduce](https://github.com/mk1-project/quickreduce/) alternative to custom allreduce and rccl. (In case of large amount of data, custom quick reduce is used instead of custom allreduce and rccl, you can refer to the results of kernel tests.) ⏎  ⏎ 2.The collective is only enabled on **AMD, MI300**, for fp16/bf16 inputs and when custom allreduce is enabled. The kernels support full precision and quantized int8, int6, int4 (sym …[truncated]

### L3-dec197e3e5  (L3, 2025-06-27, sha dec197e3e5d1, PR #20143)
TITLE: Quick Fix by adding conditional import for flash_attn_varlen_func in flash_attn (#20143)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fa_utils
FILES: vllm/attention/utils/fa_utils.py (+4/-0); vllm/v1/attention/backends/flash_attn.py (+7/-3)
LABELS: ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎ - [x ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)". ⏎  ⏎ ## Purpose ⏎  ⏎ Fix comments: https://github.com/vllm-project/vllm/pull/19560#discussion_r2169597705 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-94a55c7681  (L3, 2025-06-27, sha 94a55c76813f, PR #19891)
TITLE: [Fix][ROCm] Remove unused variables to fix build error on GFX11/12 (#19891)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.custom_paged
FILES: csrc/rocm/attention.cu (+0/-4)
LABELS: rocm, ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ After this PR: https://github.com/vllm-project/vllm/pull/19796 build is failing on AMD Radeon GPU (gfx11/12) ⏎ Removed unused variables from custom paged attention kernel to fix build error ⏎  ⏎ ## Test Plan ⏎ python setup.py develop ⏎  ⏎ ## Test Result ⏎ Build succeeded after removing unused variables. ⏎ No functional changes. ⏎  ⏎ ## (Optional) Documentation Update

### L3-e8c3bd2cd1  (L3, 2025-06-27, sha e8c3bd2cd164, PR #20141)
TITLE: [Bugfix] Fix some narrowing conversion warnings (#20141)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.cutlass_kernels
FILES: csrc/attention/mla/cutlass_mla_kernels.cu (+1/-1); csrc/mamba/causal_conv1d/causal_conv1d.cu (+2/-6); csrc/mamba/mamba_ssm/selective_scan_fwd.cu (+1/-3); csrc/quantization/fp4/nvfp4_experts_quant.cu (+2/-2); csrc/quantization/fp4/nvfp4_quant_kernels.cu (+1/-1); csrc/quantization/fp4/nvfp4_scaled_mm_kernels.cu (+1/-1)
LABELS: ready
BODY: ### Purpose ⏎ Fix some warnings of the form: ⏎ ``` ⏎ warning #2361-D: invalid narrowing conversion from "char" to "signed char" ⏎         at::cuda::CUDAGuard device_guard{(char)x.get_device()}; ⏎ ``` ⏎  ⏎ From `rg device_guard csrc`, the standard way of doing this is: ⏎ ``` ⏎ const at::cuda::OptionalCUDAGuard device_guard(device_of(x)) ⏎ ``` ⏎ so this PR updates these spots to make them consistent. ⏎  ⏎ ### Test Plan ⏎ Compile vLLM kernels, look at logs ⏎  ⏎ ### Test Result

### L3-3c545c0c3b  (L3, 2025-06-27, sha 3c545c0c3b98, PR #18064)
TITLE: [CI/Build] Allow hermetic builds (#18064)
SOURCES: dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+158/-30)
LABELS: documentation, ready, ci/build
BODY: The current Dockerfile assumes that many build artifacts are available from public repositories and downloads them from these repositories, making it more difficult for downstream distributions to perform hermetic builds of vLLM container images. ⏎  ⏎ This pull request introduces changes that should be backward compatible, while improving the situation for hermetic builds. Below is the list of changes: ⏎  ⏎ - Build argument for the base images. Currently …[truncated]

### L3-8acb4badee  (L3, 2025-07-01, sha 8acb4badee64, PR #20301)
TITLE: [CUDA graphs] Enable full cuda graphs with FA3 AoT scheduling (#20301)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/v1/attention/backends/flash_attn.py (+53/-6)
LABELS: ready, ci/build, v1
BODY: This PR enables the full cuda graph with FA3 AoT scheduling. ⏎ Previously, AoT scheduling caused illegal memory access when the run-time split factor is larger than `num_splits` set by internal heuristics at capture-time. ⏎ This case can be prevented by explicitly setting `num_splits` (the upper bound) at both capture and run time.

### L3-96453cfa83  (L3, 2025-07-01, sha 96453cfa8313, PR #19067)
TITLE: [BugFix][V1][ROCm] Triton MLA uses V0 backend on V1 engine (#19067)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.mla.common_v1, L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+8/-2); vllm/v1/attention/backends/mla/common.py (+7/-2); vllm/v1/attention/backends/mla/triton_mla.py (+57/-0); tests/kernels/attention/test_attention_selector.py (+2/-4); tests/kernels/attention/test_rocm_attention_selector.py (+4/-2)
LABELS: rocm, ready, v1
BODY: This PR aims to fix the issue that Triton MLA still uses the V0 version backend even on vLLM V1 engine. ⏎  ⏎ Also port ROCm-specific code from vllm/vllm/attention/backends/mla/common.py to vllm/vllm/v1/attention/backends/mla/common.py to resolve the following error: ⏎ > RuntimeError: Worker failed with error 'flash_attn_varlen_func() got an unexpected keyword argument 'return_softmax_lse'' ⏎  ⏎ Local verification with `lm_eval`: ⏎ ``` ⏎ VLLM_USE_V1=1 VLLM_USE_ …[truncated]

### L3-27b8017636  (L3, 2025-07-01, sha 27b8017636c5, PR #20348)
TITLE: [FIX][Intel GPU]fix ipex flash_attn_varlen_func api missing parameter (#20348)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/_ipex_ops.py (+1/-0)
LABELS: ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Fix intel gpu path attention kernel parameter mismatch issue. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-7da296be04  (L3, 2025-07-02, sha 7da296be0493, PR #20235)
TITLE: [TPU] kv cache update kernel supports dynamic grid (#20235)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/pallas_kv_cache_update.py (+6/-3); vllm/v1/attention/backends/pallas.py (+22/-12); tests/v1/tpu/test_kv_cache_update_kernel.py (+6/-2); vllm/v1/worker/tpu_model_runner.py (+8/-0)
LABELS: tpu, ready, v1
BODY: ## Purpose ⏎  ⏎ In the `kv_cache_update` TPU kernel, we pad the input tensor `slices` to prevent recompilation, but this resulted in numerous trivial DMA copies. We now use the dynamic grid features of a Pallas kernel to eliminate most of them. It can provide more than 1% e2e throughput gain usually. ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ pytest -s -v tests/v1/tpu/test_kv_cache_update_kernel.py ⏎ ``` ⏎  ⏎ ## Test Result ⏎  ⏎ Passed

### L3-a0389e0554  (L3, 2025-07-02, sha a0389e055472, PR #20169)
TITLE: [UT][intel GPU] use current_platform instead of device hardcode in v1 tests (#20169)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.mla.common_v1, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/v1/attention/backends/mla/common.py (+2/-1); tests/conftest.py (+2/-2); tests/v1/sample/test_rejection_sampler.py (+5/-4); tests/v1/sample/test_topk_topp_sampler.py (+6/-5); tests/v1/spec_decode/test_eagle.py (+13/-10); tests/v1/worker/test_gpu_input_batch.py (+3/-1); tests/v1/worker/test_gpu_model_runner.py (+2/-1); vllm/platforms/cuda.py (+5/-1); vllm/platforms/rocm.py (+5/-0); vllm/platforms/xpu.py (+1/-1)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎ We went through all the tests under v1/ with some local modification on xpu device and we found that to reuse v1 tests, the biggest gap is device hardcode. So in this PR we use current_platform class attributes and methods to take place them. After this one we will keep contributing some changes to let xpu(intel GPU) users get guaranteed by vllm tests. ⏎  ⏎ ## Test Plan ⏎ To reuse tests on intel gpu. ⏎  ⏎ ## Test Result ⏎ passed on cuda/xpu.

### L3-a1aafc827a  (L3, 2025-07-02, sha a1aafc827a2a, PR #20254)
TITLE: [ROCm][FEAT] Enable Full Graph Mode in AITER MLA V1 Attn Backend (Decode Phase only) (#20254)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+59/-31)
LABELS: rocm, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ # Purpose ⏎  ⏎ This PR adds full graph mode support in AITER MLA backend for decode only. ⏎  ⏎ # Test Plan ⏎ running lm_eval: ⏎  ⏎ enabling full graph: ⏎ `VLLM_ROCM_USE_AITER=1 vllm serve deepseek-ai/DeepSeek-V3 -tp 8 --trust-remote-code --gpu_memory_utilization 0.95  --block-size 1 -O '{"full_cuda_graph":true}'` ⏎  ⏎ piecewise: ⏎ `VLLM_ROCM_USE_AITER=1 vllm serve deepseek-ai/DeepSeek-V3 -tp 8 --trust-rem …[truncated]

### L3-bdb84e26b0  (L3, 2025-07-02, sha bdb84e26b06c, PR #20136)
TITLE: [Bugfix] Fixes for FlashInfer's TORCH_CUDA_ARCH_LIST (#20136)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+38/-17)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ CUDA 11.8 build is broken on main because we build for the same set of CUDA arches unconditionally. ⏎  ⏎ This PR updates the TORCH_CUDA_ARCH_LIST to one supported by FLASHINFER. ⏎  ⏎ Future work would be to respect the TORCH_CUDA_ARCH_LIST passed in by the user, stripping out the unsupported arches. ⏎  ⏎ ## Test Plan ⏎ build image jobs for 11.8

### L3-8d775dd30a  (L3, 2025-07-03, sha 8d775dd30a14, PR #20400)
TITLE: [Misc] Fix `Unable to detect current VLLM config. Defaulting to NHD kv cache layout` warning (#20400)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+1/-1); vllm/distributed/kv_transfer/kv_connector/utils.py (+2/-2)
LABELS: ready, v1
BODY: This warning was popping up in scenarios where it wasn't meant to be shown, as the original purpose ⏎ was to notify of the KV layout change in disagg PD setups. ⏎  ⏎ ``` ⏎ INFO 07-02 17:57:24 [default_loader.py:272] Loading weights took 0.31 seconds ⏎ INFO 07-02 17:57:25 [gpu_model_runner.py:1782] Model loading took 1.1201 GiB and 0.721210 seconds ⏎ INFO 07-02 17:57:30 [backends.py:508] Using cache directory: /home/mgoin/.cache/vllm/torch_compile_cache/a9f88 …[truncated]

### L3-1caca5a589  (L3, 2025-07-04, sha 1caca5a5899a, PR #20428)
TITLE: [Misc] Add SPDX-FileCopyrightText (#20428)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.aiter_fa, L3.mla.cutlass_v1_backend, L3.flex_attention
FILES: benchmarks/kernels/bench_fp8_gemm.py (+1/-0); examples/offline_inference/spec_decode.py (+1/-0); examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py (+1/-0); examples/online_serving/multi_instance_data_parallel.py (+1/-0); examples/online_serving/openai_chat_completion_client_with_tools_xlam.py (+1/-0); examples/online_serving/openai_chat_completion_client_with_tools_xlam_streaming.py (+1/-0); tests/compile/test_fusion_attn.py (+1/-0); tests/kernels/moe/parallel_utils.py (+1/-0); tests/kernels/moe/test_deepep_deepgemm_moe.py (+1/-0); tests/kernels/moe/test_deepep_moe.py (+1/-0); (+48 more)
LABELS: documentation, performance, frontend, tpu, speculative-decoding, ready, v1, tool-calling
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-2f35a022e6  (L3, 2025-07-04, sha 2f35a022e648, PR #20016)
TITLE: Enable V1 for Hybrid SSM/Attention Models (#20016)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: tests/models/language/generation/test_hybrid.py (+60/-10); tests/models/registry.py (+1/-1); tests/v1/test_oracle.py (+0/-1); vllm/model_executor/layers/mamba/ops/mamba_ssm.py (+1/-1); vllm/model_executor/models/bamba.py (+30/-15); vllm/model_executor/models/falcon_h1.py (+40/-17); vllm/model_executor/models/granitemoehybrid.py (+32/-17); vllm/model_executor/models/nemotron_h.py (+30/-15); vllm/model_executor/models/zamba2.py (+64/-37); vllm/v1/core/kv_cache_coordinator.py (+8/-3); (+4 more)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ This PR enables V1 for models that use both SSM and attention layers.  ⏎  ⏎ Related RFCs: #18571 #11382  ⏎  ⏎ cc @heheda12345 @tlrmchlsmth  ⏎  ⏎ ## Implementation  ⏎  ⏎ The current hybrid cache allocator implementation assumes that the [page size is the same across all KV cache groups ](https://github.com/vllm-project/vllm/blob/9a3b88328f7e434cac35b90ee463de6689f9a833/vllm/v1/core/kv_cache_utils.py#L770-L772). Therefore, we need to ensure that the pa …[truncated]

### L3-32c9be2200  (L3, 2025-07-05, sha 32c9be2200a2, PR #19754)
TITLE: [v1] Re-add fp32 support to v1 engine through FlexAttention (#19754)
SOURCES: path_core
ARTIFACT_HINTS: L3.platform.cuda_selection, L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+11/-1); .github/workflows/lint-and-deploy.yaml (+1/-1); tests/kernels/attention/test_attention_selector.py (+28/-0); tests/v1/worker/test_gpu_model_runner.py (+5/-0); vllm/engine/arg_utils.py (+0/-7); vllm/model_executor/model_loader/tensorizer_loader.py (+6/-2); vllm/platforms/cuda.py (+4/-0); vllm/v1/sample/ops/topk_topp_sampler.py (+4/-1)
LABELS: ready, ci/build, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ - We reverted [v1 fp32 support](https://github.com/vllm-project/vllm/pull/19319) to unblock v0.9.1 release before due to following failing tests: ⏎ https://buildkite.com/vllm/ci/builds/21699#01975508-6c54-47d6-a951-82873b896a11 ⏎ https://buildkite.com/vllm/ci/builds/21699#01975508-6c52-41ed-ba7a-cc1a464cec91 ⏎ - This PR re-adds v1 fp32 support with corresponding fix. ⏎  ⏎ ## Test Pl …[truncated]

### L3-4548c03c50  (L3, 2025-07-05, sha 4548c03c50d8, PR #20339)
TITLE: [TPU][Bugfix] fix the MoE OOM issue (#20339)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/layer.py (+7/-2)
LABELS: ready
DEEP_STUDY: deep-study correctness case vllm:4548c03c50: class=hardware_compiler_specific; symptom=performance_or_availability; introducing=unknown
BODY: ## Purpose ⏎  ⏎ The XLA backend for TPUs handles its own functionalization, so we don't need to wrap it as a custom operation to benefit from torch.compile's auto-functionalization. Additionally, using a custom operation would cause HBM OOM errors on TPU. ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 --seed 42 --disable-log-requests --gpu-memory-utilization 0.95  --max-num-batched-tokens 4096 --max-num-seqs 256 --tensor-parallel- …[truncated]

### L3-e202dd2736  (L3, 2025-07-06, sha e202dd2736bc, PR #20412)
TITLE: [V0 deprecation] Remove V0 CPU/XPU/TPU backends (#20412)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/cpu_mla.py (+0/-307); vllm/attention/backends/ipex_attn.py (+0/-403); vllm/attention/backends/pallas.py (+0/-356); vllm/attention/backends/torch_sdpa.py (+3/-164); .buildkite/scripts/hardware_ci/run-cpu-test.sh (+4/-4); .buildkite/scripts/hardware_ci/run-xpu-test.sh (+0/-2); examples/online_serving/chart-helm/values.yaml (+1/-1); tests/kernels/attention/test_attention_selector.py (+10/-7); vllm/platforms/cpu.py (+6/-20); vllm/platforms/tpu.py (+16/-35); (+10 more)
LABELS: documentation, performance, structured-output, frontend, tpu, speculative-decoding, ready, ci/build, v1, multi-modality
BODY: This PR is part of V0 deprecation; To begin with, we will delete the hardware backends in vLLM V0 that has already been migrated to vLLM V1 (i.e., CPU, XPU, and TPU).

### L3-cede942b87  (L3, 2025-07-06, sha cede942b87b5, PR #20516)
TITLE: [Benchmark] Add support for multiple batch size benchmark through CLI in `benchmark_moe.py` (#20516)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_moe.py (+2/-2); vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=NVIDIA_B200,dtype=fp8_w8a8.json (+147/-0)
LABELS: performance, ready
BODY: ## Purpose ⏎  ⏎ Add support for specifying multiple batch sizes via `--batch-size` in `benchmark_moe.py`.  ⏎  ⏎ ## Test Plan ⏎  ⏎ * Modified argparse to accept `nargs="+"` for batch size. ⏎ * Validated with CLI input: `--batch-size 128 256 512`. ⏎ * Verified correct tuning and config saving ⏎  ⏎ Also add tuned fused MoE for `RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic` for B200 ⏎  ⏎ ## Test Result ⏎  ⏎ Confirmed successful tuning for multiple batch sizes in a single …[truncated]

### L3-9fb52e523a  (L3, 2025-07-06, sha 9fb52e523abf, PR #20467)
TITLE: [V1] Support any head size for FlexAttention backend (#20467)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.aiter_fa, L3.mla.common_v1, L3.dispatch.selector, L3.platform.cuda_selection, L3.flex_attention
FILES: vllm/attention/layer.py (+2/-1); vllm/attention/selector.py (+30/-3); vllm/config.py (+1/-1); vllm/platforms/cuda.py (+28/-16); vllm/v1/attention/backends/cpu_attn.py (+18/-2); vllm/v1/attention/backends/flash_attn.py (+14/-8); vllm/v1/attention/backends/flashinfer.py (+16/-10); vllm/v1/attention/backends/flex_attention.py (+20/-19); vllm/v1/attention/backends/mla/common.py (+15/-8); vllm/v1/attention/backends/rocm_aiter_fa.py (+14/-10); (+10 more)
LABELS: documentation, speculative-decoding, ready, ci/build, v1, multi-modality
ISSUES: #14524 [Bug]: [V1] Support Fallback For Unsupported Head Dim on FA
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Currently, models with `head_size in [80, 112, 120]` are supported in V0 but not in V1. The reason is that the FlexAttention backend is marked as only supporting `head_size in [32, 64, 80, 96, 112, 120, 128, 192, 256]` when it should actually support all head sizes. ⏎  ⏎ This PR updates the FlexAttention backend to allow any head size, and updates the unsupported head size me …[truncated]

### L3-3112271f6e  (L3, 2025-07-07, sha 3112271f6e5d, PR #20553)
TITLE: [XPU] log clean up for XPU platform (#20553)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/_custom_ops.py (+2/-1); vllm/platforms/xpu.py (+3/-4)
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-a37d75bbec  (L3, 2025-07-07, sha a37d75bbec7e, PR #19334)
TITLE: [Front-end] microbatch tokenization (#19334)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/entrypoints/openai/test_serving_chat.py (+23/-16); vllm/entrypoints/openai/serving_engine.py (+73/-48); vllm/utils/__init__.py (+193/-1)
LABELS: frontend, ready, qwen
ISSUES: #19012 [Feature]: Microbatch Tokenization
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Fix https://github.com/vllm-project/vllm/issues/19012 ⏎  ⏎ ## Test Plan ⏎ On RTX3090 ⏎ ``` ⏎ vllm serve Qwen/Qwen1.5-1.8B --tensor-parallel-size 1 --enforce-eager --trust-remote-code ⏎ ``` ⏎ ``` ⏎ python - <<'PY' ⏎ import asyncio, httpx, json ⏎ async def main(): ⏎     url="http://localhost:8000/v1/completions" ⏎     payload={"prompt":"hello "*10000,"max_tokens":1} ⏎     async with httpx.AsyncClient …[truncated]

### L3-22dd9c2730  (L3, 2025-07-07, sha 22dd9c2730dc, PR #20308)
TITLE: [Kernel] Optimize Prefill Attention in Unified Triton Attention Kernel (#20308)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+13/-1)
LABELS: ready
BODY: This PR introduces an optimization to the unified triton attention kernel (#16828 and #19152) that enhances prefill attention performance. The key improvement involves reducing the number of tiles processed during the prefill phase by leveraging the causal mask to skip unnecessary computations. This results in more efficient execution, particularly for long prompts. ⏎  ⏎ ## Performance ⏎ The following results were obtained for `meta-llama/Llama-3.1-8B- …[truncated]

### L3-7721ef1786  (L3, 2025-07-07, sha 7721ef1786c4, PR #20560)
TITLE: [CI/Build][CPU] Fix CPU CI and remove all CPU V0 files (#20560)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/torch_sdpa.py (+0/-546); vllm/attention/ops/ipex_attn.py (+0/-195); vllm/v1/attention/backends/cpu_attn.py (+749/-13); .buildkite/scripts/hardware_ci/run-cpu-test.sh (+12/-12); tests/basic_correctness/test_chunked_prefill.py (+0/-58); tests/models/language/generation/test_common.py (+6/-2); tests/models/language/pooling/test_embedding.py (+11/-12); tests/models/language/pooling/test_reward.py (+5/-0); tests/quantization/test_compressed_tensors.py (+2/-1)
LABELS: ci/build, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ During the merging of #20412 , there were some conflicts and dropped the commit of #20437 accidently.  ⏎  ⏎ This PR resolves the conflicts and re-commits the commit to fix CPU CI tests. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-e34d130c16  (L3, 2025-07-08, sha e34d130c1613, PR #20278)
TITLE: [TPU] Temporary fix vmem oom for long model len by reducing page size (#20278)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+6/-0)
LABELS: tpu, ready, v1
DEEP_STUDY: deep-study correctness case vllm:e34d130c16: class=hardware_compiler_specific; symptom=performance_or_availability; introducing=unknown
BODY: This pr temporarily fixes vmem oom before a permanent fix to reduce VREG spill.  When max-model-len is very long (for example, Llama70b, 32k tokens), we sometimes experience VMEM OOM because VREG spill take too much memory.

### L3-6db31e7a27  (L3, 2025-07-08, sha 6db31e7a2735, PR #20554)
TITLE: [Hardware][PPC64LE] Enable V1 for ppc64le and ARM (#20554)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+4/-2); vllm/engine/arg_utils.py (+9/-6); vllm/platforms/cpu.py (+3/-2); vllm/v1/worker/cpu_worker.py (+61/-3)
LABELS: v1
ISSUES: #20622 [Bug]: Torch SDPA path broken on AArch64 due to default chunked_prefill in vLLM Engine V1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ FIX #20622 ⏎  ⏎ ## Purpose ⏎ This PR enables V1 Engine for IBM POWER, as V0 Engine would be deprecated (https://github.com/vllm-project/vllm/pull/20437, https://github.com/vllm-project/vllm/pull/20412).  ⏎  ⏎ As chunked prefill is not supported on ppc64le, this PR turns off chunked prefill in case of ppc64le.  ⏎  ⏎ Also, the CPU binding logic is not optimized for POWER (The issue and its impact ar …[truncated]

### L3-b6e7e3d58f  (L3, 2025-07-09, sha b6e7e3d58f57, PR #20659)
TITLE: [Intel GPU] support ray as distributed executor backend for XPU. (#20659)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/scripts/hardware_ci/run-xpu-test.sh (+2/-0); vllm/executor/ray_distributed_executor.py (+1/-1)
LABELS: ready, ci/build
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ add back Ray as distributed executor backend. ⏎  ⏎ ## Test Plan ⏎ Add tp=2 test in CI for both `mp` and `ray` ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-97abeb1daa  (L3, 2025-07-09, sha 97abeb1daac6, PR #20640)
TITLE: [feat] enable SM100 CUTLASS block scaled group gemm for smaller batch sizes (#20640)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/cutlass_moe.py (+4/-6); vllm/model_executor/layers/fused_moe/fused_moe.py (+1/-1)
LABELS: performance, ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Enable CUTLASS block scaled grouped GEMM for small batch sizes. ⏎  ⏎ Performance for the smaller sizes is roughly 1.2x-1.4x better than standard triton. ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ lm_eval --model vllm --model_args pretrained=/scratch/models/DeepSeek-R1,tensor_parallel_size=4,max_model_len=2048,gpu_memory_utilization=0.9,max_num_seqs=32 --trust_remote_code --tasks gsm8k --num_fewshot  …[truncated]

### L3-47043eb678  (L3, 2025-07-09, sha 47043eb6787b, PR #18218)
TITLE: [Kernel] Triton implementation of causal-conv1d for Mamba-based models (#18218)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+0/-1); csrc/mamba/causal_conv1d/causal_conv1d.cu (+0/-656); csrc/mamba/causal_conv1d/causal_conv1d.h (+0/-159); csrc/mamba/causal_conv1d/static_switch.h (+0/-28); csrc/ops.h (+0/-16); csrc/torch_bindings.cpp (+0/-22); tests/kernels/mamba/test_causal_conv1d.py (+44/-114); tests/kernels/mamba/test_mamba_ssm_ssd.py (+2/-2); vllm/_custom_ops.py (+5/-29); vllm/model_executor/layers/mamba/mamba2_metadata.py (+103/-42); (+5 more)
LABELS: ready, ci/build, v1
BODY: This PR adds Triton-based causal-conv1d, making Mamba-based models in vLLM ⏎ 1.  fully Triton-only backend. ⏎ 2. one step closer to be compatible with vLLM v1 design, i.e. without splitting the batch into prefill-only and decode-only for CUDA-split processing. ⏎  ⏎ **There are two kernels implemented** ⏎ * causal_conv1d_update_triton: which outperforms the corresponding CUDA kernel in handling decode-only requests ⏎ <img width="380" alt="image" src="https:// …[truncated]

### L3-b7d9e9416f  (L3, 2025-07-09, sha b7d9e9416f4e, PR #20651)
TITLE: [CI/Build] Fix FlashInfer double build in Dockerfile (#20651)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+12/-16)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ It seems there was a bad merge between https://github.com/vllm-project/vllm/pull/18064 and https://github.com/vllm-project/vllm/pull/20136 that resulted in FlashInfer being built "both ways" with both sets of TORCH_CUDA_ARCH_LIST. I'm not sure how the CUDA 11.8 build was working with this in place, but this may have resulted in some of the JIT performance weirdness observed in the last release. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result

### L3-4b9a9435bb  (L3, 2025-07-10, sha 4b9a9435bb6b, PR #20718)
TITLE: Update Dockerfile FlashInfer to v0.2.8rc1 (#20718)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+5/-2)
LABELS: ready, ci/build
BODY: Also add USE_FLASHINFER_PREBUILT_WHEEL to the dockerfile to prevent downloading the flashinfer wheel while we don't have an up-to-date one available

### L3-c7753a9809  (L3, 2025-07-10, sha c7753a980934, PR #14129)
TITLE: [Hardware][CPU] Vllm int8 quantization enablement for ARM CPU (#14129)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: cmake/cpu_extension.cmake (+24/-4); csrc/cpu/cpu_types_arm.hpp (+264/-3); csrc/cpu/dnnl_helper.hpp (+45/-13); csrc/cpu/quant.cpp (+12/-9); csrc/cpu/torch_bindings.cpp (+2/-1)
LABELS: ci/build, cpu
BODY: **Description** ⏎ This PR enables support of vLLM INT8 quantized model for AARCH64 architecture. Enabled ARM path for CPU inference of INT8 quantized models. ⏎  ⏎ **ARM Compatibility:** ⏎ Modified the build scripts, and configuration files to ensure compatibility with ARM processors.  ⏎  ⏎ **Checklist** ⏎  ⏎ Code changes have been tested on ARM devices (Graviton3). ⏎  ⏎ **Modifications** ⏎ 1. Modifications have been made to dnnl_helper file, the memory tag check has b …[truncated]

### L3-5b032352cc  (L3, 2025-07-10, sha 5b032352cc72, PR #20034)
TITLE: [Attention] MLA - Flashinfer Ragged Prefill (#20034)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.mla.cutlass_v1_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/layer.py (+1/-1); vllm/v1/attention/backends/flashinfer.py (+5/-68); vllm/v1/attention/backends/mla/common.py (+222/-40); vllm/v1/attention/backends/mla/cutlass_mla.py (+1/-0); vllm/v1/attention/backends/utils.py (+69/-34); tests/v1/kv_connector/__init__.py (+0/-0); tests/v1/kv_connector/unit/test_multi_connector.py (+15/-72); tests/v1/kv_connector/unit/utils.py (+62/-0); vllm/attention/utils/kv_sharing_utils.py (+33/-0); vllm/logger.py (+14/-0)
LABELS: documentation, performance, frontend, ready, v1, deepseek
BODY: This PR adds MLA FlashInfer ragged prefill support on B200 GPUs - it is dependent on this FlashInfer FIX from NVIDIA: https://github.com/flashinfer-ai/flashinfer/pull/1198  ⏎  ⏎ Here are performance results for deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct on a single B200 with 10000/100 prompt/output for various batch sizes. We can see 20-25% improvement for TTFT and 15-20% improvement for TPOT.  ⏎  ⏎ ![image](https://github.com/user-attachments/assets/2c …[truncated]

### L3-e2de455c34  (L3, 2025-07-10, sha e2de455c349d, PR #20087)
TITLE:  [Feature] Integrate SM100 DeepGEMM support (#20087)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_moe.py (+3/-0); tests/kernels/moe/test_block_fp8.py (+8/-8); tests/kernels/moe/test_deepep_deepgemm_moe.py (+5/-0); tests/kernels/moe/test_deepgemm.py (+8/-47); tests/kernels/quantization/test_block_fp8.py (+11/-16); vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py (+9/-12); vllm/model_executor/layers/fused_moe/deep_gemm_moe.py (+11/-11); vllm/model_executor/layers/fused_moe/fused_moe.py (+9/-3); vllm/model_executor/layers/fused_moe/prepare_finalize.py (+0/-1); vllm/model_executor/layers/fused_moe/triton_deep_gemm_moe.py (+5/-2); (+6 more)
LABELS: performance, frontend, ready, deepseek
BODY: ## Purpose ⏎  ⏎ DeepGemm is updating to v2.0, which includes a new implementation for SM100 that expects block FP8 scales in E8M0 format https://github.com/deepseek-ai/DeepGEMM/pull/112 ⏎  ⏎ Previous context: https://github.com/vllm-project/vllm/pull/19820 ⏎  ⏎ We add a wrapper to support both Hopper (1.x) and Blackwell (2.x) interfaces ⏎  ⏎ ## Test ⏎  ⏎ ### Unit Test ⏎  ⏎ ![image](https://github.com/user-attachments/assets/307f7716-ec62-4c58-ac6c-e78b93013a96) ⏎  ⏎ ### Acc …[truncated]

### L3-cf75cd2098  (L3, 2025-07-11, sha cf75cd2098f6, PR #20772)
TITLE: [CI Bugfix] Specify same TORCH_CUDA_ARCH_LIST for flashinfer aot and install (#20772)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+2/-1)
LABELS: bug, ready, ci/build
BODY: Fix a small issue where TORCH_CUDA_ARCH_LIST was being overriden for the flashinfer install but not the aot compile. This would give issues on CUDA <12.8

### L3-31d5c1797f  (L3, 2025-07-11, sha 31d5c1797f32, PR #19830)
TITLE: [Perf][fp8] Use CustomOp abstraction for fp8 quant for better perf (#19830)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+4/-2); vllm/attention/backends/rocm_flash_attn.py (+4/-2); benchmarks/kernels/bench_per_token_quant_fp8.py (+98/-0); tests/compile/test_fusion.py (+7/-4); tests/compile/test_fusion_attn.py (+2/-0); tests/compile/test_silu_mul_quant_fusion.py (+30/-7); vllm/compilation/fusion.py (+3/-22); vllm/model_executor/layers/fused_moe/utils.py (+2/-0); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_24.py (+16/-8); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py (+7/-1); (+8 more)
LABELS: performance, rocm, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ This PR refactors FP8 quantization kernels to use the `CustomOp` abstraction, allowing Inductor to generate fast(er) Triton kernels and automatically perform fusion with `RMSNorm` and `SiluMul` (already implemented as `CustomOp`s). This gives significant speedups, demonstrated below. ⏎  ⏎ All forward pass code for dense linear layers instantiates `QuantFP8` inside layer's `__ …[truncated]

### L3-7bd4c37ae7  (L3, 2025-07-11, sha 7bd4c37ae7c6, PR #19825)
TITLE: [Core] Add Flashinfer TRTLLM Backend for Flashinfer decode path (SM100).  (#19825)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.abstract_interface, L3.platform.cuda_selection
FILES: vllm/attention/backends/flashinfer.py (+107/-16); vllm/engine/arg_utils.py (+2/-0); vllm/envs.py (+5/-1); vllm/platforms/cuda.py (+17/-2); vllm/v1/attention/backends/flashinfer.py (+147/-36); vllm/v1/attention/backends/utils.py (+9/-1); benchmarks/kernels/benchmark_trtllm_attention.py (+240/-0); tests/kernels/attention/test_flashinfer_trtllm_decode_attention.py (+140/-0)
LABELS: performance, ready, v1
BODY: Co-authored by @wenscarl  ⏎  ⏎ ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ - [N/A] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model. ⏎  ⏎ ## Purpose ⏎ Adds decode kernels for Paged GQA for kv-cache-dtype="auto". A follow up PR would include FA3 style of Q=FP8 and KV=FP8 support  ⏎  ⏎ ## Test Plan ⏎ 1. Check the baseline perf ⏎ 2. Check the perf with the integration ⏎ 3. Check the …[truncated]

### L3-2c11a738b3  (L3, 2025-07-12, sha 2c11a738b35e, PR #20702)
TITLE: [Model] New model support for microsoft/Phi-4-mini-flash-reasoning (#20702)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.platform.cuda_selection, L3.blocksparse.v0
FILES: vllm/attention/backends/blocksparse_attn.py (+2/-1); vllm/attention/backends/differential_flash_attn.py (+1000/-0); vllm/attention/backends/dual_chunk_flash_attn.py (+2/-1); vllm/attention/backends/flash_attn.py (+2/-1); vllm/attention/backends/flashinfer.py (+2/-1); vllm/attention/backends/hpu_attn.py (+2/-1); vllm/attention/backends/rocm_flash_attn.py (+2/-1); vllm/attention/backends/xformers.py (+2/-1); vllm/attention/layer.py (+0/-4); csrc/mamba/mamba_ssm/selective_scan_fwd.cu (+25/-24); (+12 more)
LABELS: documentation, new-model, rocm, ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ New Model for https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning ⏎  ⏎ co-author: @aatkinson and @renll ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-c1acd6d7d4  (L3, 2025-07-12, sha c1acd6d7d485, PR #20774)
TITLE: [Refactor] Change the way of import triton (#20774)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+1/-2); tests/kernels/moe/test_batched_moe.py (+1/-1); vllm/lora/ops/triton_ops/lora_expand_op.py (+1/-2); vllm/lora/ops/triton_ops/lora_shrink_op.py (+1/-2); vllm/model_executor/layers/fused_moe/fused_batched_moe.py (+1/-2)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ As shown in `vllm/tools/check_triton_import.py`, only  ⏎  ⏎ ```py ⏎ ALLOWED_LINES = { ⏎     "from vllm.triton_utils import triton", ⏎     "from vllm.triton_utils import tl", ⏎     "from vllm.triton_utils import tl, triton", ⏎ } ⏎ ``` ⏎  ⏎ are allowed, this pr fixes the possible pre-commit error

### L3-e8cc53af5e  (L3, 2025-07-14, sha e8cc53af5e17, PR #20699)
TITLE: [Misc] Log the reason for falling back to FlexAttention (#20699)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.aiter_fa, L3.mla.common_v1, L3.dispatch.selector, L3.platform.cuda_selection, L3.flex_attention
FILES: vllm/attention/selector.py (+40/-9); vllm/v1/attention/backends/cpu_attn.py (+4/-0); vllm/v1/attention/backends/flash_attn.py (+4/-0); vllm/v1/attention/backends/flashinfer.py (+4/-0); vllm/v1/attention/backends/flex_attention.py (+4/-0); vllm/v1/attention/backends/mla/common.py (+4/-0); vllm/v1/attention/backends/rocm_aiter_fa.py (+4/-0); vllm/v1/attention/backends/triton_attn.py (+4/-0); vllm/platforms/cuda.py (+35/-22); vllm/reasoning/hunyuan_a13b_reasoning_parser.py (+1/-1)
LABELS: ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ #20467 accidentally disabled the warning message for failing to import FlashInfer for SM 10.0 devices. This PR fixes the issue and also consolidates the logic for falling back to FlexAttention based on head_size and dtype. ⏎  ⏎ Notable changes: ⏎  ⏎ - Added `get_supported_dtypes` to V1 attention backends. ⏎ - Renamed `supports_head_size` to a more general `is_attn_backend_supported …[truncated]

### L3-8cdc371217  (L3, 2025-07-15, sha 8cdc37121722, PR #20769)
TITLE: SM100 Cutlass MLA decode with unrestricted num_heads (< 128) for DeepSeek TP (#20769)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.mla.common_v1, L3.mla.cutlass_sm100, L3.mla.cutlass_v1_backend, L3.platform.cuda_selection
FILES: CMakeLists.txt (+2/-1); csrc/attention/mla/cutlass_sm100_mla/device/sm100_mla.hpp (+372/-0); csrc/attention/mla/cutlass_sm100_mla/kernel/sm100_fmha_mla_reduction.hpp (+203/-0); csrc/attention/mla/cutlass_sm100_mla/kernel/sm100_fmha_mla_tma_warpspecialized.hpp (+2023/-0); csrc/attention/mla/cutlass_sm100_mla/kernel/sm100_mla_tile_scheduler.hpp (+165/-0); csrc/attention/mla/sm100_cutlass_mla_kernel.cu (+273/-0); csrc/ops.h (+13/-0); csrc/torch_bindings.cpp (+17/-0); vllm/_custom_ops.py (+20/-0); vllm/platforms/cuda.py (+7/-0); (+2 more)
LABELS: documentation, performance, ready, ci/build, v1, deepseek
BODY: This PR ports SGLANG changes to remove num_heads==128 restriction in cutlass mla decode kernel to vllm.  ⏎  ⏎ Here are performance results on deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct with a single B200 (this model has < 128 num_heads so could not run before this PR). 65% TPOT improvement for b1 and around 8-10% for larger batch sizes. ⏎  ⏎ <img width="2590" height="642" alt="image" src="https://github.com/user-attachments/assets/2ace4259-0929-428b-8d2 …[truncated]

### L3-9ad0a4588b  (L3, 2025-07-15, sha 9ad0a4588ba4, PR #20934)
TITLE: [Bugfix] Switch bailout logic for kv-cache-dtype with SM100 Flashinfer (#20934)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/engine/arg_utils.py (+4/-3)
LABELS: bug, ready
BODY: This fix first checks if the current device is SM100 to validate whether kv-cache-dtype=fp8 is supported.  ⏎  ⏎ ## Essential Elements of an Effective PR Description Checklist ⏎ - [N/A] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)". ⏎  ⏎ ## Purpose ⏎ Removes the necessity for specifying `VLLM_ATTENTION_BACKEND=FLASHINFER` when `kv-cache-dtype=auto` is used on SM100 GPUs ⏎ ## Test Plan ⏎ The accuracy itself is not affe …[truncated]

### L3-c586b55667  (L3, 2025-07-15, sha c586b55667fa, PR #20415)
TITLE: [TPU] Optimize kv cache update kernel (#20415)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+6/-0); vllm/utils/__init__.py (+8/-1); vllm/v1/worker/tpu_model_runner.py (+50/-16)
LABELS: tpu, ready, v1
BODY: ## Purpose ⏎  ⏎ This PR improves the throughput of the kv cache update kernel from 28.15 GB/s to 91.65 GB/s on v6e. ⏎  ⏎ This PR adds to optimizations on top of https://github.com/vllm-project/vllm/pull/19928. It picks the optimal number of slices to copy per kernel program instance based on results from microbenchmarks. ⏎  ⏎ Reference data demonstrating improvements: https://github.com/tengyifei/playground/blob/master/pallas/better-index-copy.ipynb ⏎  ⏎ An earl …[truncated]

### L3-ed10f3cea1  (L3, 2025-07-15, sha ed10f3cea199, PR #20330)
TITLE: [ROCm] warpSize is being made non constexpr in ROCm 7.0 (#20330)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cuh (+1/-7); csrc/attention/paged_attention_v1.cu (+1/-7); csrc/attention/paged_attention_v2.cu (+1/-7); csrc/cuda_compat.h (+3/-3)
LABELS: rocm, ready
BODY: ROCm7.0 hip compiler adds a breaking change of making warpSize a non constexpr value ⏎  ⏎ Using an alternative method of deducing the right warp size to use ⏎  ⏎ References: ⏎ https://rocm.blogs.amd.com/ecosystems-and-partners/transition-to-hip-7.0:-guidance-on-upcoming-compatibility-changes/README.html#warpsize-change ⏎ https://rocm.docs.amd.com/en/latest/about/release-notes.html#amdgpu-wavefront-size-compiler-macro-deprecation

### L3-30800b01c2  (L3, 2025-07-15, sha 30800b01c23d, PR #20411)
TITLE: [Nvidia] Integrate SM100 cudnn prefill API to MLA prefill (#20411)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.common_v1
FILES: vllm/envs.py (+5/-0); vllm/v1/attention/backends/mla/common.py (+108/-5)
LABELS: performance, ready, v1, deepseek
BODY: ## Purpose ⏎ Integrate cudnn prefill API.  ⏎  ⏎ ## Dependency  ⏎ FlashInfer dependency: https://github.com/flashinfer-ai/flashinfer/commit/9157d0514fc251178e98877b4595c701ee8fb482  ⏎ FlashInfer build command: `TORCH_CUDA_ARCH_LIST="10.0a" MAX_JOBS=160 pip install --no-build-isolation --verbose --editable .` ⏎  ⏎ ## Repo command: ⏎ To repo, run offline: ⏎ ``` ⏎ VLLM_USE_CUDNN_PREFILL=1 python3 benchmarks/benchmark_throughput.py --model=deepseek-ai/DeepSeek-Coder-V2-L …[truncated]

### L3-1eb2b9c102  (L3, 2025-07-15, sha 1eb2b9c10205, PR #20919)
TITLE: [CI] update typos config for CI pre-commit and fix some spells (#20919)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.mla.common_v1
FILES: vllm/attention/backends/differential_flash_attn.py (+1/-1); vllm/v1/attention/backends/mla/common.py (+1/-1); .pre-commit-config.yaml (+1/-1); csrc/cpu/sgl-kernels/common.h (+1/-1); csrc/cpu/sgl-kernels/gemm.h (+1/-1); csrc/cpu/sgl-kernels/gemm_int8.cpp (+1/-1); csrc/cpu/sgl-kernels/vec.h (+1/-1); docker/Dockerfile (+1/-1); docs/usage/v1_guide.md (+1/-1); pyproject.toml (+183/-0); (+9 more)
LABELS: documentation, performance, frontend, tpu, ready, ci/build, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ - [] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model. ⏎  ⏎ ## Purpose ⏎  ⏎ Add `codespell` check in pre-commit. which find more problems then `pre-commit run typos` ⏎  ⏎ ## Test Plan ⏎  ⏎ all test passed. ⏎ ``` ⏎  pre-commit run codespell --all-files ⏎ codespell................................................................Passed ⏎ pre-co …[truncated]

### L3-d31a647124  (L3, 2025-07-15, sha d31a64712489, PR #21020)
TITLE: [BugFix] Fix import error on non-blackwell machines (#21020)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: csrc/attention/mla/sm100_cutlass_mla_kernel.cu (+10/-0); csrc/ops.h (+0/-13); csrc/torch_bindings.cpp (+2/-3)
LABELS: bug, ready
BODY: FIX https://github.com/vllm-project/vllm/pull/20769#issuecomment-3074250008 ⏎  ⏎ Checked `vllm serve` runs when built on hopper ⏎ Checked can run `VLLM_ATTENTION_BACKEND=CUTLASS_MLA_VLLM_V1 lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,trust_remote_code=true --tasks gsm ⏎ 8k --batch_size auto`

### L3-85431bd9ad  (L3, 2025-07-16, sha 85431bd9ad16, PR #21007)
TITLE: [TPU] fix kv_cache_update kernel block size choosing logic (#21007)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+48/-1); vllm/v1/worker/tpu_model_runner.py (+3/-2)
LABELS: tpu, ready, v1
DEEP_STUDY: deep-study correctness case vllm:85431bd9ad: class=hardware_compiler_specific; symptom=performance_or_availability; introducing=unknown
BODY: ## Purpose ⏎  ⏎ Fix TPU CI test. ⏎  ⏎ ## Test Plan ⏎  ⏎ pytest -s -v tests/v1/tpu/test_basic.py ⏎  ⏎ ## Test Result ⏎  ⏎ Passed. ⏎  ⏎ ## (Optional) Documentation Update

### L3-a50d918225  (L3, 2025-07-16, sha a50d918225f8, PR #21013)
TITLE: [Docker] Allow FlashInfer to be built in the ARM CUDA Dockerfile (#21013)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+27/-41)
LABELS: ready, ci/build
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-72ad273582  (L3, 2025-07-17, sha 72ad2735823e, PR #21065)
TITLE: Remove torch_xla.tpu.version() from pallas.py. (#21065)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+0/-4)
LABELS: tpu, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Because of this [issue](https://github.com/pytorch/xla/issues/9449), some TPU test randomly fails.  ⏎ We used to use the tpu_version() to decide some setting but now it is just a safety check.  There is better place to put it.  Since there is no one actually using version < 4 on vLLM today and we focuses on v6e TPU, it is safe to remove it.   ⏎  ⏎ ## Test Plan ⏎ Run the test on CI …[truncated]

### L3-4e7dfbe7b4  (L3, 2025-07-17, sha 4e7dfbe7b49a, PR #21011)
TITLE: Update PyTorch to `torch==2.7.1` for CUDA (#21011)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1); requirements/build.txt (+1/-1); requirements/cuda.txt (+5/-5); requirements/test.in (+3/-3); requirements/test.txt (+4/-4); pyproject.toml (+1/-1); tests/entrypoints/openai/test_vision.py (+2/-2)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ Update CUDA to `torch==2.7.1`. Separate work will need to be done to update ROCm, CPU, etc. ⏎  ⏎ ## Test Plan ⏎  ⏎ CI ⏎  ⏎ As a smoke test, I quickly verified that I can install vLLM from source (using precompiled wheels just for convience) ⏎ ``` ⏎ VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto ⏎ Using Python 3.12.11 environment at: /home/mgoin/venvs/test ⏎ Resolved 134 packages in 7.31s ⏎       Built vllm @ file:///home/mgoin/code/vllm …[truncated]

### L3-76b494444f  (L3, 2025-07-17, sha 76b494444fd8, PR #20466)
TITLE: [Attention] Refactor attention metadata builder interface (#20466)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.aiter_fa, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.rocm_aiter, L3.dispatch.abstract_interface, L3.flex_attention
FILES: vllm/v1/attention/backends/cpu_attn.py (+34/-31); vllm/v1/attention/backends/flash_attn.py (+48/-53); vllm/v1/attention/backends/flashinfer.py (+51/-106); vllm/v1/attention/backends/flex_attention.py (+26/-33); vllm/v1/attention/backends/mla/common.py (+64/-119); vllm/v1/attention/backends/mla/flashmla.py (+8/-7); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+19/-16); vllm/v1/attention/backends/rocm_aiter_fa.py (+39/-50); vllm/v1/attention/backends/triton_attn.py (+34/-39); vllm/v1/attention/backends/utils.py (+138/-2); (+8 more)
LABELS: performance, rocm, speculative-decoding, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Refactor the attention metadata to try to break the dependency on `runner`. This has a few distinct advantages: ⏎  ⏎ 1) Easier to unit-test (don't need to mock an entire GPU runner) ⏎ 2) Easier to benchmark (don't need to mock an entire GPU runner) ⏎ 3) Some attention schemes are fairly generic and can be implemented by just manipulating a sufficiently descriptive CommonAttention …[truncated]

### L3-4de7146351  (L3, 2025-07-17, sha 4de7146351d6, PR #21131)
TITLE: [V0 deprecation] Remove V0 HPU backend (#21131)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen
FILES: vllm/attention/backends/hpu_attn.py (+0/-319); vllm/attention/ops/hpu_paged_attn.py (+0/-88); docker/Dockerfile.hpu (+0/-21); requirements/hpu.txt (+0/-12); setup.py (+2/-34); vllm/_custom_ops.py (+1/-2); vllm/config.py (+1/-1); vllm/core/block/cpu_gpu_block_allocator.py (+1/-3); vllm/distributed/device_communicators/hpu_communicator.py (+0/-46); vllm/engine/arg_utils.py (+2/-3); (+17 more)
LABELS: ready, ci/build
BODY: In vLLM V1, HPU is supported as a plugin.

### L3-c7d8724e78  (L3, 2025-07-17, sha c7d8724e7865, PR #20037)
TITLE: [Core] FlashInfer CUTLASS fused MoE backend (NVFP4) (#20037)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+107/-0); vllm/_custom_ops.py (+9/-13); vllm/envs.py (+5/-0); vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py (+13/-23); vllm/model_executor/layers/fused_moe/batched_triton_or_deep_gemm_moe.py (+4/-3); vllm/model_executor/layers/fused_moe/config.py (+16/-0); vllm/model_executor/layers/fused_moe/cutlass_moe.py (+238/-46); vllm/model_executor/layers/fused_moe/deep_gemm_moe.py (+2/-1); vllm/model_executor/layers/fused_moe/deepep_ht_prepare_finalize.py (+8/-11); vllm/model_executor/layers/fused_moe/deepep_ll_prepare_finalize.py (+8/-11); (+12 more)
LABELS: performance, ready, ci/build, v1, deepseek
BODY: This PR covers: ⏎ 1. Integrates Flashinfer NVFP4 cutlass MoE kernel from https://github.com/flashinfer-ai/flashinfer/pull/1113, which can be enabled with env var `VLLM_USE_FLASHINFER_MOE`=1. It supports ⏎  ⏎ - DP + EP  ⏎ - TP + EP ⏎ - DP + TP + EP ⏎  ⏎ 2. Refactor cutlass_moe_fp4 to modular kernel structure. `cutlass_moe_fp4` backend supports DP or TP or DP + TP. EP is not supported yet. ⏎  ⏎ ## Example usage ⏎ Use Data Parallel [script](https://github.com/vllm-proj …[truncated]

### L3-89cab4d01f  (L3, 2025-07-18, sha 89cab4d01f83, PR #21093)
TITLE: [Attention] Make local attention backend agnostic (#21093)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.aiter_fa, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/flash_attn.py (+10/-74); vllm/v1/attention/backends/flashinfer.py (+1/-4); vllm/v1/attention/backends/rocm_aiter_fa.py (+7/-90); vllm/v1/attention/backends/triton_attn.py (+7/-61); vllm/v1/attention/backends/utils.py (+24/-6); vllm/v1/worker/gpu_model_runner.py (+23/-4); vllm/v1/core/single_type_kv_cache_manager.py (+7/-3); vllm/v1/kv_cache_interface.py (+15/-0)
LABELS: ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Make local attention backend agnostic now that https://github.com/vllm-project/vllm/pull/20466 has landed so we can turn on llama4 iRoPE for FlashInfer on Blackwell ⏎  ⏎ ## Test Plan ⏎  ⏎ Ruler eval ⏎  ⏎ ## Test Result ⏎  ⏎ ### This PR ⏎  ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=FLASHINFER python -m lm_eval --model vllm --model_args pretrained=/home/lwilkinson/local_models/meta-llama--Llama-4-Scout-17B- …[truncated]

### L3-8dfb45ca33  (L3, 2025-07-18, sha 8dfb45ca3379, PR #21133)
TITLE: [Bugfix] Fix the tensor non-contiguous issue for Flashinfer TRT-LLM backend attention kernel (#21133)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+23/-11)
LABELS: bug, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ - Fix the non-contiguous tensor `decode_query` used in Flashinfer TRT-LLM attention kernel ⏎ - Fix the function arguments passing of `FlashInferBackend.use_trtllm_decode_attention` ⏎   - `self.cache_config.cache_dtype` instead of `attn_metadata.kv_data_type` ⏎  ⏎ ## Test Plan ⏎ Check the accuracy with lm_eval. ⏎  ⏎ ## Test Result ⏎ Before: ⏎ ``` ⏎ vllm (pretrained=nvidia/Llama-4-Scout-17B-16E …[truncated]

### L3-b2eb2b5ad7  (L3, 2025-07-18, sha b2eb2b5ad709, PR #19346)
TITLE: [Kernel] Apply torch.Tag.needs_fixed_stride_order only for torch==2.6.0 (#19346)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/rocm_aiter_mla.py (+6/-2); csrc/torch_bindings.cpp (+8/-4); vllm/model_executor/layers/fused_moe/fused_moe.py (+5/-3)
LABELS: rocm, ready
BODY: Summary: ⏎ In torch 2.6.0, torch accidentally changed the default for custom operators to be "requires_contiguous". As a workaround, vLLM added needs_fixed_stride_order to a large number of custom operators. ⏎  ⏎ vLLM is currently on torch 2.7.0 which has reverted the default for custom operators back to needs_fixed_stride_order. This PR cleans up the kernel logic by flipping the default back. ⏎  ⏎ The other reason why I want to flip the default back is th …[truncated]

### L3-5782581acf  (L3, 2025-07-18, sha 5782581acfa4, PR #21077)
TITLE: [Bugfix] Voxtral on Blackwell GPUs (RTX 50 series) (#21077)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+33/-0)
LABELS: ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Voxtral uses `MultiHeadAttention` class, which uses xformers for acceleration    ⏎ But unfortunately, xformers does not support Blackwell GPUs    ⏎ This PR allows the code to fallback onto torch_sdpa backend for compatibility    ⏎  ⏎ Similar PR: #20998  ⏎ Similar Issues: #20025, #20193 ⏎  ⏎ ## Test Plan ⏎  ⏎ Use the command from Mistral AI documentation to start a server ⏎ ```bash ⏎ uv run vllm …[truncated]

### L3-9a9fda1423  (L3, 2025-07-18, sha 9a9fda1423c9, PR #19351)
TITLE: [Core] Support Local Chunked Attention for Hybrid KV Cache (#19351)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/layer.py (+1/-0); vllm/v1/attention/backends/flash_attn.py (+2/-1); vllm/v1/attention/backends/utils.py (+1/-0); tests/v1/core/test_specialized_manager.py (+155/-2); vllm/config.py (+7/-0); vllm/v1/core/kv_cache_utils.py (+16/-3); vllm/v1/core/single_type_kv_cache_manager.py (+124/-1); vllm/v1/kv_cache_interface.py (+37/-12); vllm/v1/worker/gpu_model_runner.py (+8/-0)
LABELS: frontend, ready, v1, tool-calling
BODY: ## Purpose ⏎  ⏎ This PR follows  #17996 to add Hybrid KV Cache support for local chunked attention for supporting models like llama4 maverick and scout. ⏎  ⏎ ## Test Plan ⏎  ⏎ ### unit test: ⏎  ⏎ ``` ⏎ pytest tests/v1/core/test_specialized_manager.py -k "chunked_local" ⏎ ``` ⏎  ⏎ ### eval: ⏎ 1. run mmlupro before and after on llama4 scout model ⏎  ⏎ ``` ⏎ lm_eval --model vllm --tasks mmlu_pro --model_args pretrained=$HUB/Llama-4-Scout-17B-16E-Instruct,max_model_len=131072,tenso …[truncated]

### L3-59f935300c  (L3, 2025-07-19, sha 59f935300c48, PR #21196)
TITLE: [BugFix] Fix potential cuda-graph IMA (#21196)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+0/-5); vllm/v1/worker/gpu_model_runner.py (+6/-1)
LABELS: bug, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Cuda-graph padding happens after prepare inputs; it's safer to -1 fill here (and closer behavior to pre https://github.com/vllm-project/vllm/pull/20466 ). No errors reported yet this is just preventative ⏎  ⏎ ## Test Plan ⏎  ⏎ lm_eval ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1  lm_eval   --model vllm   --model_args '{ ⏎     "pretrained": "meta-llama/Meta-Llama- …[truncated]

### L3-6d0734c562  (L3, 2025-07-19, sha 6d0734c562e7, PR #20645)
TITLE: [NVIDIA] Add SM100 Flashinfer MoE blockscale fp8 backend for low latency (#20645)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+12/-2); vllm/envs.py (+8/-3); vllm/model_executor/layers/fused_moe/config.py (+1/-1); vllm/model_executor/layers/fused_moe/fused_moe.py (+99/-1); vllm/model_executor/layers/quantization/fp8.py (+63/-19); vllm/model_executor/layers/quantization/modelopt.py (+4/-5)
LABELS: performance, ready
BODY: For [this PR](https://github.com/flashinfer-ai/flashinfer/pull/1212), Flashinfer introduces a new backend for block-wise scaled FP8. ⏎ This PR adds support for that backend. ⏎  ⏎ cc. @kushanam @wenscarl @pavanimajety

### L3-752c6ade2e  (L3, 2025-07-19, sha 752c6ade2e0f, PR #21217)
TITLE: [V0 Deprecation] Deprecate BlockSparse Attention & Phi3-Small (#21217)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.rocm_flash_attn_v0, L3.rocm.aiter_fa, L3.mla.triton_v0, L3.mla.common_v1, L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.rocm_aiter, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.flex_attention, L3.blocksparse.v0
FILES: vllm/attention/backends/abstract.py (+0/-1); vllm/attention/backends/blocksparse_attn.py (+0/-466); vllm/attention/backends/differential_flash_attn.py (+0/-4); vllm/attention/backends/dual_chunk_flash_attn.py (+0/-1); vllm/attention/backends/flash_attn.py (+1/-5); vllm/attention/backends/flashinfer.py (+0/-1); vllm/attention/backends/flashmla.py (+4/-8); vllm/attention/backends/mla/common.py (+0/-1); vllm/attention/backends/rocm_aiter_mla.py (+4/-8); vllm/attention/backends/rocm_flash_attn.py (+1/-5); (+28 more)
LABELS: documentation, new-model, rocm, tpu, ready, ci/build, v1
BODY: This PR removes the block sparse attention and the support for phi3-small which uses the attention.

### L3-3a1d8940ae  (L3, 2025-07-20, sha 3a1d8940aea5, PR #19292)
TITLE: [TPU] support fp8 kv cache quantization (#19292)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+50/-8); tests/entrypoints/llm/test_accuracy.py (+30/-10); tests/v1/tpu/test_pallas.py (+2/-0); vllm/engine/arg_utils.py (+4/-4); vllm/platforms/tpu.py (+3/-1); vllm/v1/worker/tpu_model_runner.py (+6/-5)
LABELS: tpu, ready, v1
BODY: ## Purpose ⏎ To support fp8 kv cache quantization on TPU. ⏎  ⏎ ## Test Plan ⏎  ⏎ `chengjiyao/Llama-3.1-8B-Instruct-FP8-KV` was created based on https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache.html ⏎  ⏎ Test 0: ⏎ ```python ⏎ # SPDX-License-Identifier: Apache-2.0 ⏎ # SPDX-FileCopyrightText: Copyright contributors to the vLLM project ⏎  ⏎ import argparse ⏎ import os ⏎  ⏎ from vllm import LLM, SamplingParams ⏎  ⏎ prompts = [ ⏎     "A robot may not injure a human  …[truncated]

### L3-6dda13c86b  (L3, 2025-07-21, sha 6dda13c86ba1, PR #21282)
TITLE: [Misc] Add sliding window to flashinfer test (#21282)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_flashinfer.py (+31/-18)
LABELS: ready
BODY: 

### L3-a15a50fc17  (L3, 2025-07-21, sha a15a50fc17f9, PR #21289)
TITLE: [CPU] Enable shared-memory based pipeline parallel for CPU backend (#21289)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: .buildkite/scripts/hardware_ci/run-cpu-test.sh (+9/-9); csrc/cpu/shm.cpp (+48/-21); docs/getting_started/installation/cpu.md (+14/-0); vllm/distributed/device_communicators/cpu_communicator.py (+58/-2); vllm/distributed/parallel_state.py (+12/-0); vllm/engine/arg_utils.py (+5/-4); vllm/envs.py (+4/-3); vllm/platforms/cpu.py (+15/-20)
LABELS: documentation, ready, ci/build
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ - Enable shared-memory based pipeline parallel for CPU backend. ⏎ - Refine default batch-size settings of CPU backend. ⏎ - Refactor ```get_device_total_memory``` of ```CpuPlatform```. ⏎ - Enable a E2E test for TP&PP in the CPU tests script. ⏎ - Update related doc and add more tunning guidance. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-304dce7ec0  (L3, 2025-07-21, sha 304dce7ec027, PR #21188)
TITLE: [Attention] Clean up iRoPE in V1 (#21188)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.aiter_fa
FILES: vllm/attention/layer.py (+7/-0); vllm/v1/attention/backends/cpu_attn.py (+0/-5); vllm/v1/attention/backends/flash_attn.py (+0/-2); vllm/v1/attention/backends/flashinfer.py (+0/-2); vllm/v1/attention/backends/pallas.py (+0/-5); vllm/v1/attention/backends/rocm_aiter_fa.py (+0/-2); vllm/v1/attention/backends/triton_attn.py (+0/-6); vllm/v1/worker/gpu_model_runner.py (+3/-4); vllm/v1/worker/tpu_model_runner.py (+4/-0)
LABELS: tpu, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ With https://github.com/vllm-project/vllm/commit/89cab4d01f83f8def180e723cee30c7ef8c53e86 we can actually entirely remove the concept of iRoPE from V1 backends. ⏎  ⏎ ## Test Plan ⏎  ⏎ lm eval checks; maybe someone with access to a TPU could help me test that? otherwise the change seems simple enough ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=FLASH_ATTN python -m lm_eval --model  …[truncated]

### L3-c17231e827  (L3, 2025-07-21, sha c17231e82799, PR #21302)
TITLE: Fix kv_cache_dtype handling for out-of-tree HPU plugin (#21302)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/engine/arg_utils.py (+2/-16); vllm/platforms/cuda.py (+13/-0); vllm/platforms/interface.py (+7/-0); vllm/platforms/rocm.py (+4/-0); vllm/platforms/tpu.py (+4/-0)
LABELS: rocm, tpu, ready
BODY: PR https://github.com/vllm-project/vllm/pull/21131 removed HPU checks for `--kv-cache-dtype` flag, and now HPU out-of-tree plugin (https://github.com/vllm-project/vllm-gaudi) is unable to use KV cache quantization with INC due to `NotImplementedError: VLLM_USE_V1=1 is not supported with --kv-cache-dtype.`. This PR adds a check for out-of-tree HPU plugin and allows it to use fp8_inc KV cache quantization.

### L3-9e23ad9655  (L3, 2025-07-21, sha 9e23ad9655c8, PR #21327)
TITLE: Update fp4 quantize API (#21327)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+4/-4); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py (+5/-5); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py (+2/-2)
LABELS: ready
BODY: Update fp4 quantize API ⏎  ⏎ cc. @alexm-redhat  ⏎ ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-4fb56914c5  (L3, 2025-07-22, sha 4fb56914c5f2, PR #21116)
TITLE: [perf] Add fused MLA QKV + strided layernorm (#21116)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/deepseek_v2.py (+39/-18); csrc/layernorm_kernels.cu (+40/-23); csrc/layernorm_quant_kernels.cu (+25/-14); csrc/quantization/fp8/common.cu (+4/-0); tests/kernels/core/test_layernorm.py (+19/-7); vllm/model_executor/layers/linear.py (+77/-1); vllm/model_executor/layers/quantization/fp8.py (+10/-3)
LABELS: ready, deepseek
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ For MLA models that have a q_lora_rank: fuse q_lora and kv_lora into the same matrix (avoids some traffic + one less kernel call). ⏎  ⏎ Also adds a implementation for layernorm to operate on strided input, this avoids memory copy. ⏎  ⏎ ## Test Plan ⏎  ⏎ Units tests added for strided layernorm. E2E testing & benchamrks results in this PR ⏎  ⏎ ## Test Result ⏎  ⏎ ### Accuracy ⏎  ⏎ main (20149d84d9 …[truncated]

### L3-2c8db17cfd  (L3, 2025-07-22, sha 2c8db17cfd6f, PR #20447)
TITLE: [feat]: add SM100 support for cutlass FP8 groupGEMM (#20447)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+21/-1); csrc/quantization/cutlass_w8a8/moe/grouped_mm_c3x.cuh (+9/-4); csrc/quantization/cutlass_w8a8/moe/grouped_mm_c3x_sm100.cu (+140/-0); csrc/quantization/cutlass_w8a8/moe/grouped_mm_c3x_sm90.cu (+17/-13); csrc/quantization/cutlass_w8a8/moe/moe_data.cu (+1/-1); csrc/quantization/cutlass_w8a8/scaled_mm_entry.cu (+34/-11); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py (+6/-0); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+27/-2)
LABELS: ready, ci/build
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Add GroupGEMM support for SM100 ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ python -m pytest tests/kernels/moe/test_cutlass_moe.py ⏎  ⏎ lm_eval --model vllm --model_args pretrained=/scratch/models/Llama-4-Maverick-17B-128E-Instruct-FP8,tensor_parallel_size=4,max_model_len=2048,gpu_memory_utilization=0.9,max_num_seqs=32 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto ⏎  ⏎ lm_eval --mode …[truncated]

### L3-3ec7170ff1  (L3, 2025-07-22, sha 3ec7170ff191, PR #21393)
TITLE: [Bugfix][ROCm][Build] Fix build regression on ROCm (#21393)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+2/-2); csrc/ops.h (+5/-5); csrc/torch_bindings.cpp (+9/-9)
LABELS: rocm, ready, ci/build
BODY: Moving CUDA specific file to the ifdef CUDA section ⏎ Fix regression from #21083

### L3-2dec7c1a5d  (L3, 2025-07-22, sha 2dec7c1a5df9, PR #21420)
TITLE: [Bugfix][CUDA] fixes CUDA FP8 kv cache dtype supported (#21420)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+13/-13)
LABELS: ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Previous PR(#21302) defined a class method `is_kv_cache_dtype_supported()` under `NonNvmlCudaPlatform` class, but `NvmlCudaPlatform` also needs this method. ⏎  ⏎ So this PR moved the method to the parent class `CudaPlatformBase`. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-7c734ee09b  (L3, 2025-07-23, sha 7c734ee09b0a, PR #21364)
TITLE: [Bugfix][Qwen][DCA] fixes bug in dual-chunk-flash-attn backend for qwen 1m models. (#21364)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/dual_chunk_flash_attn.py (+0/-8)
LABELS: ready, qwen
BODY: …en 1m models. ⏎  ⏎ ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Fixes a bug in the DCA backend. The error was introduced during rebasing previous PR: https://github.com/vllm-project/vllm/pull/11844

### L3-78c13e30e1  (L3, 2025-07-23, sha 78c13e30e164, PR #21419)
TITLE: [V1] Fix local chunked attention always disabled (#21419)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+2/-1)
LABELS: ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ [#21188](https://github.com/vllm-project/vllm/pull/21188) and [#19351](https://github.com/vllm-project/vllm/pull/19351) made similar and conflicting changes around `self.use_irope` in Attention layer, causing `self.use_irope` to always be `False` in V1: ⏎  ⏎ ``` ⏎ self.use_irope = extra_impl_args.pop("use_irope", False) ⏎ ... ⏎ self.use_irope = extra_impl_args.get("use_irope", False …[truncated]

### L3-90eeea8f85  (L3, 2025-07-24, sha 90eeea8f8501, PR #21205)
TITLE: [Bugfix][ROCm] Fix for warp_size uses on host (#21205)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.rocm.custom_paged
FILES: csrc/attention/attention_kernels.cuh (+1/-1); csrc/attention/paged_attention_v1.cu (+2/-3); csrc/attention/paged_attention_v2.cu (+2/-3); csrc/rocm/attention.cu (+1/-1); csrc/cuda_compat.h (+29/-2); csrc/moe/topk_softmax_kernels.cu (+29/-18); csrc/quantization/activation_kernels.cu (+1/-1); csrc/quantization/gguf/gguf_kernel.cu (+1/-1); csrc/rocm/skinny_gemms.cu (+1/-1)
LABELS: rocm, ready
BODY: Fix for the regression added in #20330  ⏎ Before that change certain configs relied on the undefined behavior on ROCm (using warpSize compiler builtin on host), and after they started crashing due to the value mismatch. ⏎  ⏎ On ROCm the same compiled image can be used on different platforms with different warp sizes (Instinct, Radeon), therefore the WARP_SIZE that is used in the host code can't be made constexpr, but rather needs to be queried from the …[truncated]

### L3-526078a96c  (L3, 2025-07-24, sha 526078a96c52, PR #21385)
TITLE: bump `flashinfer` to `v0.2.8` (#21385)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1)
LABELS: ready, ci/build
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Bump flashinfer version in official vllm docker image to `v0.2.8`, the latest stable release. ⏎  ⏎ `flashinfer==0.2.8` includes [a QoL update for nonstandard NVSHMEM location support](https://github.com/flashinfer-ai/flashinfer/pull/1253), which is especially useful for vllm users experimenting with expert parallelism by stacking EP libraries on top of vllm docker image. ⏎  ⏎ ##  …[truncated]

### L3-61b8cea3b4  (L3, 2025-07-24, sha 61b8cea3b42f, PR #21137)
TITLE: [Attention] Optimize FlashInfer MetadataBuilder Build call (#21137)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+83/-74); tests/v1/attention/test_attention_backends.py (+10/-3); tests/v1/attention/utils.py (+1/-1)
LABELS: rocm, speculative-decoding, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Flash infer prefers host side CPU buffers in many cases, example: https://github.com/flashinfer-ai/flashinfer/blob/3c40456effae8b9c5b1a11c0d1e0594295b1a312/flashinfer/prefill.py#L1430-L1436 ⏎  ⏎ So we pass host side buffers (since https://github.com/vllm-project/vllm/pull/20466 we now have access to these) to reduce D2H transfers. ⏎  ⏎ Trace from main showing D2H transfers in `pl …[truncated]

### L3-1b25f1fe75  (L3, 2025-07-24, sha 1b25f1fe757b, PR #21408)
TITLE: Update flashinfer CUTLASS MoE Kernel (#21408)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+4/-4); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py (+2/-2); vllm/model_executor/layers/quantization/modelopt.py (+2/-2)
LABELS: ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ Before the change: ⏎ INFO:lm_eval.loggers.evaluation_tracker:Output path not provided, skipping saving results aggregated ⏎ vllm (pretrained=nvidia/DeepSeek-R1-FP4,quantization=modelopt_fp4,tensor_parallel_size=4,enforce_eager=True,max_model_len=2048,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric  …[truncated]

### L3-2dd72d23d9  (L3, 2025-07-24, sha 2dd72d23d965, PR #21485)
TITLE: update flashinfer to v0.2.9rc1 (#21485)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: docker/Dockerfile (+1/-1); vllm/attention/backends/flashinfer.py (+3/-7); vllm/v1/attention/backends/flashinfer.py (+2/-7)
LABELS: ready, ci/build, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ updata flashinfer and modify it's trtllm-gen call to use latest API. ⏎ ## Test Plan ⏎ test llama4 with lm_eval ⏎ ## Test Result ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.934|±  |0.0111| ⏎ |     |       |strict-match    …[truncated]

### L3-6066284914  (L3, 2025-07-24, sha 6066284914c7, PR #18293)
TITLE: [P/D] Support CPU Transfer in NixlConnector (#18293)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/tpu.txt (+1/-0); tests/v1/kv_connector/nixl_integration/run_tpu_disagg_accuracy_test.sh (+162/-0); tests/v1/kv_connector/nixl_integration/run_tpu_edge_case_test.sh (+128/-0); tests/v1/kv_connector/nixl_integration/test_disagg_accuracy.py (+162/-0); tests/v1/kv_connector/nixl_integration/test_edge_cases.py (+6/-3); tests/v1/kv_connector/nixl_integration/toy_proxy_server.py (+3/-3); vllm/distributed/kv_transfer/kv_connector/v1/base.py (+14/-1); vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py (+229/-43); vllm/v1/worker/gpu_model_runner.py (+6/-52); vllm/v1/worker/kv_connector_model_runner_mixin.py (+70/-0); (+2 more)
LABELS: tpu, ready, ci/build, v1
BODY: This PR adds TPU support in `NixlConnector` (#17751) for P/D disaggregated serving. ⏎ The high-level idea is to use a buffer in host memory as the kv transfer buffer. The kv transfer buffer is registered under nixl agent (as the type of "DRAM"). The computed KV cache (full blocks) at the prefill instance will be saved to the transfer buffer. One the decode side, the remote KV data will be read into the transfer buffer and then load into the device  …[truncated]

### L3-b3caeb82e7  (L3, 2025-07-25, sha b3caeb82e740, PR #20295)
TITLE: [ROCm][AITER] Enable fp8 kv cache on rocm aiter backend. (#20295)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+129/-96); tests/kernels/attention/test_aiter_flash_attn.py (+191/-0)
LABELS: documentation, performance, rocm, speculative-decoding, ready, ci/build, v1, llama
BODY: Rocm aiter backend could support fp8 kv cache with latest aiter ⏎ CMD: ⏎ HIP_VISIBLE_DEVICES=3,4 VLLM_ROCM_USE_AITER=1 VLLM_USE_V1=1 vllm serve /models/models--amd--Meta-Llama-3.1-8B-Instruct-FP8-KV/snapshots/fa42f9a9105c545755fea25cf69f49ac8c8b40e1/ --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --trust-remote-code --disable-log-requests --block-size 16 --max-model-len 32768 --dtype float16 --quantization fp8 --no-enable-prefix-caching --max- …[truncated]

### L3-136d750f5f  (L3, 2025-07-25, sha 136d750f5f42, PR #21556)
TITLE: [Kernel] Improve machete memory bound perf (#21556)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/quantization/machete/machete_prepacked_layout.cuh (+6/-2)
LABELS: ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Thanks @LucasWilkinson for tips and guidance on this! ⏎  ⏎ TL;DR - Improve the memory-bound performance of Machete kernel by fixing TMA load memory alignment (see test result for performance comparison before/after this change) ⏎  ⏎ Machete kernel uses TMA load instruction to copy matrices from global memory to shared memory ([ref](https://github.com/vllm-project/vllm/blob/main/cs …[truncated]

### L3-c215f5c877  (L3, 2025-07-26, sha c215f5c877e0, PR #21634)
TITLE: [Bug] Fix `has_flashinfer_moe` Import Error when it is not installed (#21634)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+2/-1)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Originally, if user doesn't install flashinfer, this will raise an error: ⏎  ⏎ ```bash ⏎ [rank0]:     return importlib.util.find_spec("flashinfer.fused_moe") is not None ⏎ [rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ [rank0]:   File "<frozen importlib.util>", line 92, in find_spec ⏎ [rank0]: ModuleNotFoundError: No module named 'flashinfer' ⏎ ``` ⏎  ⏎ Because `find_spec("flashinfer.fused_moe")` is different with `find_spec("fl …[truncated]

### L3-1cd6eaba54  (L3, 2025-07-26, sha 1cd6eaba54a2, PR #21270)
TITLE: Support encoder-only models without KV-Cache (#21270)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/flash_attn.py (+84/-7); vllm/v1/attention/backends/utils.py (+3/-0); examples/offline_inference/prithvi_geospatial_mae.py (+1/-1); tests/conftest.py (+11/-2); tests/model_executor/test_model_load_with_params.py (+9/-3); tests/models/language/pooling/test_embedding.py (+3/-11); tests/models/language/pooling/test_jina.py (+8/-0); tests/v1/attention/utils.py (+1/-0); tests/v1/test_oracle.py (+0/-1); tests/v1/test_utils.py (+1/-2); (+7 more)
LABELS: documentation, speculative-decoding, ready, v1
ISSUES: #18052 [RFC]: Support pooling in V1
BODY: Add support for encoder models such as BERT which don't support a KV cache due to the non-causal attention. Since the KV Cache Spec is used to build the attention metadata for decoder models, this PR initializes the attention metadata builds for encoder-only models directly from the layers and adds a function to build the attention metadata. ⏎  ⏎ This PR combines elements of PRs ⏎ https://github.com/vllm-project/vllm/pull/21088 ⏎ and https://github.com/v …[truncated]

### L3-8f605ee309  (L3, 2025-07-27, sha 8f605ee30912, PR #21626)
TITLE: [Attention] Make CutlassMLA the default backend for SM100 (blackwell) (#21626)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+22/-7)
LABELS: ready, v1, deepseek
BODY: This PR makes CutlassMLA the default backend for Blackwell GPUs. Extensive benchmarking showed that CutlassMLA is much faster than TritonMLA for decode operations for prompt-heavy workloads on B200 GPUs for both small and large DeepSeek models, for both FP8 and FP4, while not imposing penalty for prompt-average workloads.  Here is an example result for DeepSeek FP4 on 4xB200 GPUs that compares TritonMLA vs CutlassMLA - we can see improvement arou …[truncated]

### L3-e626d286f5  (L3, 2025-07-28, sha e626d286f5ac, PR #21242)
TITLE: [FEAT] [ROCm] [AITER]: Add AITER HIP block quant kernel (#21242)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/utils/fp8_utils.py (+13/-2)
LABELS: rocm, ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Introduce AITER HIP Block Scale Quantization kernel. ⏎ Verified on AITER Commit: `916bf3c` ⏎  ⏎ ## Test Plan ⏎ - Accuracy: Run lm_eval on gsm8k dataset ⏎ - Perf gain: Compare before and after of DeepSeek-R1 ⏎   - ISL: 1000  ⏎   - OSL: 1000 ⏎   - Dataset: Random  ⏎ ## Test Result ⏎  ⏎ #### Accuracy ⏎ vllm (pretrained=deepseek-ai/DeepSeek-R1,tensor_parallel_size=8,max_model_len=32768,block_size=1,t …[truncated]

### L3-b361f14e39  (L3, 2025-07-28, sha b361f14e3948, PR #21350)
TITLE: [AMD][BugFix] Fix omission  of wvSplitK kernel for small batch sizes (1-4) due to torch.compile (#21350)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/utils.py (+28/-4)
LABELS: rocm, ready
BODY: It turns out that torch.compile was omitting calls to wvSplitK, our skinny gemm, since it was never being called after torch.compile compilation.   ⏎  ⏎ Several attempts were made to fix this, such as lifting the conditional logic in rocm_unquantized_gemm into its caller, converting wvSplitK into a custom op, and twiddling with the logic in rocm_unquantized_gemm.    ⏎  ⏎ Converting rocm_unquantized_gemm into a custom op via direct_register_custom_op with …[truncated]

### L3-01c753ed98  (L3, 2025-07-28, sha 01c753ed98c7, PR #21701)
TITLE: update flashinfer to v0.2.9rc2 (#21701)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1)
LABELS: ready, ci/build
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ We need https://github.com/flashinfer-ai/flashinfer/pull/1332 to make building work on GH200 and GB200. ⏎  ⏎ ## Test Plan ⏎  ⏎ lm_eval test llama4 with flashinfer backend ⏎  ⏎ ## Test Result ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_matc …[truncated]

### L3-58b11b24a6  (L3, 2025-07-29, sha 58b11b24a69f, PR #21525)
TITLE: [Bugfix] Fix workspace buffer None issue for Flashinfer TRTLLM Backend (#21525)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/attention/backends/flashinfer.py (+11/-4); vllm/v1/attention/backends/flashinfer.py (+15/-15); benchmarks/kernels/benchmark_trtllm_attention.py (+28/-14); tests/kernels/attention/test_flashinfer_trtllm_decode_attention.py (+7/-9)
LABELS: performance, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ There is an always valid `workspace_buffer` initialized in `attn_metadata.decode_wrapper._float_workspace_buffer`, so no need to store the extra one `attn_metadata.workspace_buffer`. They should be the same. ⏎  ⏎ Original PR(#19825) initialize `attn_metadata` via the following way, `self._workspace_buffer` will be None at the beginning, and will be allocated after ONE round of …[truncated]

### L3-a33ea28b1b  (L3, 2025-07-29, sha a33ea28b1be6, PR #21389)
TITLE: Add `flashinfer_python` to CUDA wheel requirements (#21389)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+3/-1); requirements/cuda.txt (+2/-0)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ We have installed flashinfer by default in the docker image for a long time, but now as we use flashinfer for many critical kernels for NVIDIA Blackwell we should consider adding it to the default CUDA dependencies. ⏎  ⏎ The `flashinfer-python` wheel by default does not include pre-compiled kernels, so users will JIT at runtime. ⏎  ⏎ ## Test Plan ⏎  ⏎ See if there are any conflicts in CI with the Dockerfile's manual AOT build ⏎  ⏎ ## Test Result

### L3-a1873db23d  (L3, 2025-07-29, sha a1873db23dd5, PR #21127)
TITLE: docker: docker-aware precompiled wheel support (#21127)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen
FILES: docker/Dockerfile (+16/-10); setup.py (+43/-15); vllm/envs.py (+9/-2)
LABELS: ci/build
BODY: Main goal is in the context of CI, in order to not build wheels when unnecessary, and speed up CI builds overall. ⏎  ⏎ - added VLLM_DOCKER_BUILD_CONTEXT to keep precompiled wheel logic in  setup.py but add parameterization for use during a docker build. ⏎ - normalized VLLM_USE_PRECOMPILED, treat only "1" or "true" as true (makes it more complex to force unset in CI context) ⏎ - setup.py now copies contextually-named precompiled wheel into dist/ during do …[truncated]

### L3-555e7225bc  (L3, 2025-07-30, sha 555e7225bcb9, PR #21412)
TITLE: [v1][attention] Support Hybrid Allocator + FlashInfer (#21412)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.aiter_fa, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.rocm_aiter, L3.dispatch.abstract_interface, L3.flex_attention
FILES: vllm/config.py (+24/-8); vllm/v1/attention/backends/cpu_attn.py (+2/-2); vllm/v1/attention/backends/flash_attn.py (+2/-2); vllm/v1/attention/backends/flashinfer.py (+7/-11); vllm/v1/attention/backends/flex_attention.py (+2/-2); vllm/v1/attention/backends/mla/common.py (+3/-1); vllm/v1/attention/backends/mla/flashmla.py (+4/-3); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+4/-3); vllm/v1/attention/backends/rocm_aiter_fa.py (+2/-2); vllm/v1/attention/backends/triton_attn.py (+2/-2); (+6 more)
LABELS: documentation, rocm, speculative-decoding, ready, ci/build, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Support hybrid allocator + flashinfer backend. Achieved by letting the attention backend know the set of layers that use this backend, and only performs plan for these layers. ⏎  ⏎ Limitation: ⏎ For a model with both sliding window attention and full attention, when hybrid allocator is disabled, both the two types of layer use the same attention metadata builder, and flashinfer  …[truncated]

### L3-b876860c62  (L3, 2025-07-30, sha b876860c6214, PR #21848)
TITLE: [Hardware][CPU] Build fix for ARM without BF16 (#21848)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/cpu/quant.cpp (+2/-0)
LABELS: ready
BODY: So we can build on older ARM chips like M1 that don't have bf16.

### L3-fcfd1eb9c5  (L3, 2025-07-30, sha fcfd1eb9c556, PR #21910)
TITLE: [Doc] Remove vLLM prefix and add citation for PagedAttention (#21910)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docs/assets/design/paged_attention/k_vecs.png (+0/-0); docs/assets/design/paged_attention/key.png (+0/-0); docs/assets/design/paged_attention/logits_vec.png (+0/-0); docs/assets/design/paged_attention/q_vecs.png (+0/-0); docs/assets/design/paged_attention/query.png (+0/-0); docs/assets/design/paged_attention/v_vec.png (+0/-0); docs/assets/design/paged_attention/value.png (+0/-0); docs/design/paged_attention.md (+20/-9); docs/design/plugin_system.md (+1/-1); docs/design/torch_compile.md (+1/-1)
LABELS: documentation, ready
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ - Remove redundant "vLLM" prefix from design doc titles to keep pages in alphabetical order ⏎ - **Add paper link and BibTeX citation to PagedAttention docs** ⏎ - Also fix links to the figures ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-ad510309ee  (L3, 2025-07-30, sha ad510309ee10, PR #21590)
TITLE: Override attention metadata for fast prefill in some KV sharing setups (#21590)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+33/-2); tests/v1/e2e/test_kv_sharing_fast_prefill.py (+143/-0); vllm/config.py (+15/-0); vllm/engine/arg_utils.py (+6/-0); vllm/model_executor/models/gemma3n.py (+1/-0); vllm/v1/worker/gpu_model_runner.py (+89/-24)
LABELS: frontend, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ To support YOCO prefill skip for cross-decoder layers in #19719, we need to propagate `logits_indices` that has been padded to align with cudagraph captured batch sizes as well as size of the original `logits_indices`.  ⏎  ⏎ To do this, we can identify which layers are eligible, store these layer names in `truncated_prefill_eligible_layers` and dynamically build a subclass of  …[truncated]

### L3-287f527f54  (L3, 2025-07-30, sha 287f527f5403, PR #20155)
TITLE: [Feature] Add async tensor parallelism for scaled mm (#20155)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/test_async_tp.py (+138/-5); vllm/compilation/collective_fusion.py (+242/-2); vllm/compilation/sequence_parallelism.py (+1/-1)
LABELS: ready
BODY: This PR adds [torch async tp](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487) using compilation pass for scaled mm.  ⏎ It builds upon previous work to extend async tensor parallelism support to quantized models. ⏎  ⏎ It requires below config to run ⏎ ``` ⏎ config = CompilationConfig( ⏎     level=3, ⏎     compile_sizes=[4, 8, 16], ⏎     splitting_ops=[], ⏎ ) ⏎ config.pass_config.enable_noop = True ⏎ config. …[truncated]

### L3-61445453df  (L3, 2025-07-30, sha 61445453df8e, PR #21966)
TITLE: [UX] Rename CUTLASS_MLA_VLLM_V1 to CUTLASS_MLA (#21966)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.mla.cutlass_v1_backend, L3.platform.cuda_selection
FILES: vllm/engine/arg_utils.py (+1/-1); vllm/platforms/cuda.py (+5/-5); vllm/platforms/interface.py (+1/-1); vllm/v1/attention/backends/mla/cutlass_mla.py (+1/-1)
LABELS: ready, v1
BODY: We should probably remove most of the "VLLM_V1" attention names and just use their base name + `use_v1`

### L3-207b750e19  (L3, 2025-07-31, sha 207b750e1948, PR #21458)
TITLE: [NVIDIA] Add SM100 Flashinfer MoE per tensor scale fp8 backend (#21458)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+2/-0); vllm/model_executor/layers/fused_moe/fused_moe.py (+95/-18); vllm/model_executor/layers/quantization/fp8.py (+44/-31); vllm/model_executor/layers/quantization/modelopt.py (+28/-0); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+100/-0)
LABELS: ready
BODY: ## Purpose ⏎ This PR introduces a new backend for per-tensor scaled MoE from flashinfer. This backend gives a perf improvement as described below. ⏎  ⏎ ## Accuracy tests ⏎ Ran manual `lm_eval gsm8k`, using the following command: ⏎ ``` ⏎ VLLM_USE_FLASHINFER_MOE_FP8=1 CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 VLLM_ATTENTION_BACKEND=FLASHINFER \ ⏎ lm_eval --model vllm --model_args pretrained=<Llama4 Scout ckpts path>,\ ⏎ tensor_parallel_size=1,max_model_len=2048,kv_cach …[truncated]

### L3-58bb902186  (L3, 2025-07-31, sha 58bb902186a8, PR #22025)
TITLE: fix(setup): improve precompiled wheel setup for Docker builds (#22025)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-0); requirements/test.txt (+16/-8); setup.py (+87/-116)
LABELS: ready, ci/build
BODY: This refactors the logic for handling `VLLM_USE_PRECOMPILED` to ensure that Docker builds extract only the required .so files and properly modify the package_data before setup() ⏎  ⏎ - Removes errant precompiled wheel copy that was copying old code ⏎   - e.g. not code from the current checkout. ⏎ - Moves precompiled wheel extraction logic into a utility class ⏎ - Applies package_data patch before calling setup() ⏎ - Now skips build_ext when precompiled is en …[truncated]

### L3-d2aab336ad  (L3, 2025-07-31, sha d2aab336ad78, PR #21599)
TITLE: [CI/Build] get rid of unused VLLM_FA_CMAKE_GPU_ARCHES (#21599)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+0/-3); docker/Dockerfile.nightly_torch (+0/-3); .buildkite/scripts/hardware_ci/run-gh200-test.sh (+1/-2); .github/workflows/scripts/build.sh (+0/-1); docs/deployment/docker.md (+1/-2)
LABELS: documentation, ready, ci/build
BODY: This variable is currently unused and is confusing. ⏎  ⏎ Related: https://github.com/vllm-project/flash-attention/pull/74

### L3-0bd409cf01  (L3, 2025-07-31, sha 0bd409cf01c3, PR #21959)
TITLE: Move flashinfer-python to optional extra `vllm[flashinfer]` (#21959)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: requirements/cuda.txt (+1/-3); setup.py (+3/-1)
LABELS: ready, ci/build
BODY: Adding flashinfer-python by default to the CUDA requirements introduced a new hard requirement for vLLM to have `nvcc` installed, which we don't want to enforce for all users. See issue https://github.com/vllm-project/vllm/issues/21960 ⏎  ⏎ For now we will move the dependency as an extras i.e. `uv pip install vllm[flashinfer]`

### L3-e1a7fe4af5  (L3, 2025-08-01, sha e1a7fe4af5e9, PR #19750)
TITLE: [BugFix] fix: aot passes kvcache dtype information (#19750)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+21/-4)
LABELS: ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ AOT doesn't pass all the information to vLLM's metadata function. That results in inconsistent runs, potentially cause data corruption. ⏎  ⏎ ## Test Plan ⏎  ⏎ Not sure what I should add here. Maybe @LucasWilkinson? ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-da31f6ad3d  (L3, 2025-08-01, sha da31f6ad3dac, PR #22055)
TITLE: Revert precompile wheel changes (#22055)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen
FILES: docker/Dockerfile (+10/-17); requirements/test.txt (+8/-16); setup.py (+87/-95); vllm/envs.py (+2/-9)
LABELS: ready, ci/build
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s)  reason=build_or_dependency
BODY: Reverting the 3 commits touching setup.py that has caused some issues. We will rework them to ensure it doesn't break any workload for both local dev and CI testing. cc @dougbtv @DarkLight1337  ⏎  ⏎ 58bb902186a87007deeeef2d2af02ed2b13bb182 ⏎ b9b753e7a7d95311186bbfc2b30b643a2f9e6ca1 ⏎ a1873db23dd597930a7e4731a53314ace92baf49

### L3-f81c1bb055  (L3, 2025-08-01, sha f81c1bb05504, PR #21893)
TITLE: [Bugfix] Check NVIDIA artifactory is accessible before using flashinfer cubin kernels (#21893)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1
FILES: vllm/attention/backends/flashinfer.py (+2/-44); vllm/utils/flashinfer.py (+80/-1); vllm/v1/attention/backends/flashinfer.py (+3/-46); vllm/v1/attention/backends/mla/common.py (+8/-8)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Due to limitations in FlashInfer (https://github.com/flashinfer-ai/flashinfer/issues/1352), the AOT wheel generation does not download kernels ahead of time that are cubin-based. This causes issues for users that rely on our docker image to run in network-isolated environment with everything included. ⏎  ⏎ To support this usage immediately, I believe we should gate default usage of kernels that require downloading with an initial check to …[truncated]

### L3-eefbf4a68b  (L3, 2025-08-01, sha eefbf4a68b7b, PR #22036)
TITLE: [Perf] Optimize `reshape_and_cache_flash` CUDA Kernel (#22036)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+69/-23); benchmarks/kernels/benchmark_reshape_and_cache_flash.py (+156/-0)
LABELS: performance, ready
BODY: ## Purpose ⏎  ⏎ Using vectorization utils to  `reshape_and_cache_flash` and get performance improvement ⏎  ⏎ ## Test ⏎  ⏎ ### Acc ⏎  ⏎ ```bash ⏎ lm_eval   --model vllm   --model_args "pretrained=Qwen/Qwen3-30B-A3B-FP8,max_model_len=32768,enforce_eager=True"   --trust_remote_code   --tasks gsm8k   --num_fewshot 5   --batch_size auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----- …[truncated]

### L3-0edaf752d7  (L3, 2025-08-01, sha 0edaf752d748, PR #21153)
TITLE: [Attention][DBO] Add support for "splitting" the CommonAttentionMetadata (#21153)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+83/-0); tests/v1/attention/test_attention_splitting.py (+157/-0)
LABELS: ready, v1
BODY: ## Purpose ⏎ Note: the infrastructure in this PR is currently unused outside of unit tests. ⏎  ⏎ The purpose of this PR is to add a "splitting" mechanism to the `CommonAttentionMetadata` class. This PR is a prerequisite for Dual Batch Overlap (#20448 ) support in vllm. Splitting, in this context, means slicing a batch of requests along some mid point and generating two new `CommonAttentionMetadata` instances. One for each "micro" batch. ⏎  ⏎ ## Test Plan ⏎ I …[truncated]

### L3-23322431c8  (L3, 2025-08-01, sha 23322431c802, PR #21367)
TITLE: [V1][CUDA] Full cudagraph support for FlashInfer (#21367)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.mla.flashmla_v1_adapter, L3.mla.rocm_aiter, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/flash_attn.py (+5/-2); vllm/v1/attention/backends/flashinfer.py (+323/-34); vllm/v1/attention/backends/mla/flashmla.py (+3/-1); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+3/-1); vllm/v1/attention/backends/triton_attn.py (+4/-2); vllm/v1/attention/backends/utils.py (+17/-1); vllm/v1/worker/gpu_model_runner.py (+17/-7); vllm/v1/worker/gpu_worker.py (+5/-0)
LABELS: rocm, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ This PR is split from origin #20059 to support full cudagraph for FlashInfer (pure decode only), which runs pure decode batches at full cudagraph, and falls back to no cudagraph at mix prefill-decode batches. Hope to land this first before #20059. ⏎  ⏎ Details include: ⏎ - Using the persistent buffer trick. ⏎ - Create many decode_warpers, one for a cudagraph batch size, as this is …[truncated]

### L3-d3a6f2120b  (L3, 2025-08-01, sha d3a6f2120bb6, PR #22069)
TITLE: [FEAT][ROCm] Enable running Flash Attention as ViT attn backend for Qwen-VL models on ROCm platform. (#22069)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/platforms/cuda.py (+14/-0); vllm/platforms/interface.py (+5/-0); vllm/platforms/rocm.py (+12/-0); vllm/model_executor/models/qwen2_5_vl.py (+13/-5); vllm/model_executor/models/qwen2_vl.py (+13/-5); vllm/model_executor/models/vision.py (+7/-29)
LABELS: rocm, ready, qwen
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose  ⏎ This PR enables the ViT backend selection in the Qwen2 and Qwen2.5 vision language models to support the `FLASH_ATTN` backend on the ROCm platform. It uses the flash_attn_varlen_func from vLLM and additionally provides support for the AITER flash attention implementation (flash_attn_varlen_func from AITER). ⏎  ⏎ ## Test Plan and Results ⏎  ⏎ Use [mistral-evals](https://github.com/ …[truncated]

### L3-4abfd8796f  (L3, 2025-08-02, sha 4abfd8796f37, PR #21557)
TITLE: [V1] [Hybrid] Validate compatibility of attention backend batch reordering at init time (#21557)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.rocm.aiter_fa, L3.mla.common_v1, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/flashinfer.py (+12/-16); vllm/v1/attention/backends/mla/common.py (+7/-15); vllm/v1/attention/backends/rocm_aiter_fa.py (+0/-3); vllm/v1/attention/backends/utils.py (+4/-8); vllm/v1/worker/cpu_model_runner.py (+33/-1); vllm/v1/worker/gpu_model_runner.py (+34/-15); vllm/v1/attention/backends/mamba_attn.py (+6/-14)
LABELS: ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Right now on main we have a check at runtime to see if the different attention backends reorder the requests in the same way. This has a number of problems: ⏎ 1. Sometimes problems only get caught after running a long experiment  ⏎ 2. The check calls reorder_batch for the first backend, and then calls reorder_batch for all subsequent backends. If any of the subsequent backend …[truncated]

### L3-e27d25a0dc  (L3, 2025-08-03, sha e27d25a0dcbb, PR #22154)
TITLE: [fix] fix correct assertion syntax error in attention utils. (#22154)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+3/-1)
LABELS: v1
BODY: ## Purpose ⏎  ⏎   Fix syntax error in assertion statement in vllm/v1/attention/backends/utils.py where ⏎   parentheses were incorrectly placed. The assertion len(query_start_loc >= 2) should be ⏎    len(query_start_loc) >= 2 to properly validate that query_start_loc has at least 2 ⏎   elements. ⏎  ⏎   ## Test Plan ⏎  ⏎   - Run existing unit tests to ensure no regressions ⏎   - Verify the assertion now correctly validates tensor length ⏎   - Test with malformed input t …[truncated]

### L3-aa7012eb6d  (L3, 2025-08-03, sha aa7012eb6db6, PR #20401)
TITLE: Add tree attention backend for v1 (part 1) (#20401)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.triton.unified_attention, L3.dispatch.abstract_interface, L3.platform.cuda_selection, L3.tree_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+48/-0); vllm/config.py (+13/-0); vllm/engine/arg_utils.py (+1/-1); vllm/platforms/cuda.py (+4/-0); vllm/platforms/interface.py (+1/-0); vllm/v1/attention/backends/tree_attn.py (+452/-0); vllm/v1/attention/backends/utils.py (+20/-0); tests/v1/attention/test_attention_backends.py (+1/-1); tests/v1/attention/utils.py (+4/-2); tests/v1/spec_decode/test_eagle.py (+4/-3); (+2 more)
LABELS: speculative-decoding, ready, v1, llama
BODY: # Purpose ⏎ Add support for tree attention v1 backend. Tree attention is used in EAGLE speculative decoding by the target model to validate a set of draft tokens. Draft tokens only attend to ancestor tokens, and so attention bias must be used to omit attention between non-descendant tokens. To suppor that, I added a new parameter to triton `unified_attention` called `qq_bias`. This parameter enables applying query-on-query attention bias using a 2D …[truncated]

### L3-cdfd6871a5  (L3, 2025-08-04, sha cdfd6871a5c4, PR #22226)
TITLE: [Bugfix] Misaligned params in TreeAttentionImpl (#22226)
SOURCES: path_core
ARTIFACT_HINTS: L3.tree_attention
FILES: vllm/v1/attention/backends/tree_attn.py (+1/-5)
LABELS: ready, v1
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Tree attention is still broken because it conflicts with #21217 causing misaligned args to be passed to the model (notably `attn_type` becomes `None`). This PR fixes it. ⏎  ⏎ cc @TheEpicDolphin ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-e79a12fc3a  (L3, 2025-08-04, sha e79a12fc3afb, PR #22217)
TITLE: [UX] Fail if an invalid attention backend is specified (#22217)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/selector.py (+4/-0); tests/kernels/attention/test_attention_selector.py (+5/-15)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ As the title states, I think if an invalid attention backend is manually specified like VLLM_ATTENTION_BACKEND=INVALID it should fail rather than fallback to some default ⏎  ⏎ Rob found that this PR https://github.com/vllm-project/vllm/pull/21966 causes fallback to V0 if that env variable is set incorrectly ⏎ ``` ⏎ WARNING 08-04 15:00:03 [arg_utils.py:1771] VLLM_ATTENTION_BACKEND=CUTLASS_MLA_VLLM_V1 is not supported by the V1 Engine. Falling  …[truncated]

### L3-83156c7b89  (L3, 2025-08-05, sha 83156c7b89fb, PR #22095)
TITLE: [NVIDIA] Support Flashinfer TRT-LLM Prefill Attention Kernel (#22095)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/attention/backends/flashinfer.py (+2/-2); vllm/envs.py (+3/-3); vllm/utils/flashinfer.py (+7/-10); vllm/v1/attention/backends/flashinfer.py (+145/-80); .buildkite/test-pipeline.yaml (+1/-1); benchmarks/kernels/benchmark_trtllm_decode_attention.py (+0/-1); benchmarks/kernels/benchmark_trtllm_prefill_attention.py (+250/-0); tests/kernels/attention/test_flashinfer_trtllm_attention.py (+293/-0); tests/kernels/attention/test_flashinfer_trtllm_decode_attention.py (+0/-138)
LABELS: performance, ready, ci/build, v1
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Previously #19825 supported TRTLLM attn kernel for decode code path, this PR is aiming to support the prefill path. ⏎ - Unified the decode and prefill code path, use only one env `VLLM_USE_TRTLLM_ATTENTION` to control ⏎ - Currently the TRTLLM prefill kernel only does not support Q-BF16 KV-FP8 O-BF16 config, will directly use Q-FP8 KV-FP8 O-FP8/O-NVFP4 after the attn+quant fusio …[truncated]

### L3-a7cb6101ca  (L3, 2025-08-05, sha a7cb6101ca7b, PR #22233)
TITLE: [CI/Build] Update flashinfer to 0.2.9 (#22233)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); setup.py (+1/-1)
LABELS: ready, ci/build
BODY: Critical to make the docker build on CPU actually build the AOT kernels https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.2.9

### L3-469b3ffaaa  (L3, 2025-08-05, sha 469b3ffaaadb, PR #21342)
TITLE: [V1] port xformers backend to v1 (#21342)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.xformers.v1_backend, L3.platform.cuda_selection, L3.tree_attention
FILES: vllm/v1/attention/backends/tree_attn.py (+1/-6); vllm/v1/attention/backends/xformers.py (+430/-0); tests/v1/attention/utils.py (+2/-0); vllm/engine/arg_utils.py (+1/-0); vllm/platforms/cuda.py (+4/-0); vllm/platforms/interface.py (+1/-0)
LABELS: ready, v1
BODY: # Purpose ⏎ Port over the xformers backend to the v1 engine. There are several benefits to using XFormers, including: ⏎ 1. Built-in heursitic which determines which attention implementation is best suited for the given inputs. ⏎ 2. AMD kernel support ⏎ 3. Well suited for certain Meta models. ⏎  ⏎ # Test Plan ⏎ Added test case to `test_attention_backends` which verifies correctness of the xformers v1 backend attention output. ⏎ ``` ⏎ (py312conda) bash-5.1$ pytest t …[truncated]

### L3-e3c876dca3  (L3, 2025-08-05, sha e3c876dca357, PR #22313)
TITLE: Upgrade FA3 for attention sink (#22313)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ci/build
BODY: 

### L3-6e20924350  (L3, 2025-08-05, sha 6e20924350e3, PR #22320)
TITLE: Add attention sink in attention backends (#22320)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.trtllm_gen, L3.triton.prefix_prefill, L3.triton.chunked_prefill_paged_decode, L3.triton.unified_attention, L3.triton.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+27/-6); vllm/attention/ops/prefix_prefill.py (+15/-3); vllm/attention/ops/triton_unified_attention.py (+28/-2); vllm/envs.py (+16/-3); vllm/v1/attention/backends/flash_attn.py (+10/-0); vllm/v1/attention/backends/triton_attn.py (+56/-19); vllm/v1/attention/backends/utils.py (+24/-12)
LABELS: v1
BODY: 

### L3-98a3a81024  (L3, 2025-08-05, sha 98a3a8102464, PR #22329)
TITLE: [ROCm] Add attention sink to use_rocm_custom_paged_attention (#22329)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+6/-5)
LABELS: rocm
BODY: 

### L3-90ec006937  (L3, 2025-08-05, sha 90ec006937c4, PR #22330)
TITLE: [gpt-oss] flashinfer attention sink init (#22330)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+10/-0)
LABELS: v1
BODY: 

### L3-35509fc5be  (L3, 2025-08-06, sha 35509fc5be5d, PR #22286)
TITLE: [Bugfix] Remove faulty test for oot attention backend (#22286)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/plugins_tests/test_platform_plugins.py (+0/-10)
LABELS: ready
ISSUES: #22285 [CI Failure]: Plugin Tests (2 GPUs) - plugins_tests/test_platform_plugins.py::test_oot_attention_backend
BODY: FIX https://github.com/vllm-project/vllm/issues/22285 ⏎  ⏎ This test isn't valid anymore since we changed the attention selection behavior to error with invalid backends

### L3-2cb6ef8996  (L3, 2025-08-06, sha 2cb6ef899632, PR #22365)
TITLE: [BugFix] Fix FA2 RuntimeError when sinks is provided (#22365)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ci/build
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ fix  ⏎ ``` ⏎ RuntimeError: _vllm_fa2_C::varlen_fwd() expected at most 21 argument(s) but received 22 argument(s). Declaration: _vllm_fa2_C::varlen_fwd(Tensor($0! -> ) q, Tensor k, Tensor v, Tensor($1! -> )? out, Tensor cu_seqlens_q, Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? leftpad_k, Tensor? block_table, Tensor? alibi_slopes, int max_seqlen_q, int max_seqlen_k, float p_ …[truncated]

### L3-4a6b72c2ab  (L3, 2025-08-06, sha 4a6b72c2ab98, PR #22368)
TITLE: [BugFix] Fix triton compile error in `kernel_unified_attention_2/3d` caused by attention sinks (#22368)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+15/-8)
LABELS: bug, ready
ISSUES: #22363 [Bug]: AttributeError("'NoneType' object has no attribute 'type'") from `sink_ptr` of `unified_attention` when running spec dec with Triton Attention Backend V1
BODY: FIX: https://github.com/vllm-project/vllm/issues/22363 ⏎  ⏎ Tested with: ⏎  ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 vllm serve ⏎ ``` ⏎  ⏎ and  ⏎  ⏎ ``` ⏎ curl -X POST http://127.0.0.1:8000/v1/chat/completions      -H "Content-Type: application/json"      -d '{ ⏎            "model": "Qwen/Qwen3-0.6B", ⏎            "messages": [ ⏎              { "role": "user", "content": "Hello, vLLM!" } ⏎            ], ⏎            "max_tokens": 50, ⏎            "temperature": 0.7 ⏎ }' ⏎ ` …[truncated]

### L3-2435ea7ed5  (L3, 2025-08-06, sha 2435ea7ed5c3, PR #22370)
TITLE: [Bugfix] Make condition in triton kernel constexpr (#22370)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.prefix_prefill, L3.triton.chunked_prefill_paged_decode
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+3/-1); vllm/attention/ops/prefix_prefill.py (+3/-1)
DEEP_STUDY: deep-study correctness case vllm:2435ea7ed5: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: Followup to #22320  ⏎ The error to be fixed is: ⏎ ``` ⏎ (EngineCore_0 pid=17748) ERROR 08-06 16:03:14 [core.py:683]     if sink_ptr is None or segm_idx != 0: ⏎ (EngineCore_0 pid=17748) ERROR 08-06 16:03:14 [core.py:683]         M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32) ⏎ (EngineCore_0 pid=17748) ERROR 08-06 16:03:14 [core.py:683]     else: ⏎ (EngineCore_0 pid=17748) ERROR 08-06 16:03:14 [core.py:683]         M = tl.load( ⏎ (EngineCore_0 pid=17748 …[truncated]

### L3-31f5dc5b2a  (L3, 2025-08-06, sha 31f5dc5b2a5d, PR #22335)
TITLE: [gpt-oss] Enhance error msg on attention sink init (#22335)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+9/-5)
LABELS: v1
BODY: @WoosukKwon gemini fix

### L3-9a3835aaa9  (L3, 2025-08-06, sha 9a3835aaa900, PR #22378)
TITLE: Fix trtllm-gen attention env and add attention sink (#22378)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.abstract_interface
FILES: vllm/envs.py (+4/-9); vllm/utils/flashinfer.py (+4/-4); vllm/v1/attention/backends/flashinfer.py (+9/-8); vllm/v1/attention/backends/utils.py (+2/-4); vllm/model_executor/models/gpt_oss.py (+2/-3)
LABELS: v1
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-e8961e963a  (L3, 2025-08-06, sha e8961e963a76, PR #22389)
TITLE: Update `flashinfer-python==0.2.10` (#22389)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); setup.py (+1/-1)
LABELS: ready, ci/build
BODY: Needed for GPT-OSS Support: Add Blackwell MoE mxfp4 implementation from TRTLLM and Attention Sink in https://github.com/flashinfer-ai/flashinfer/pull/1389

### L3-f825c6bd22  (L3, 2025-08-06, sha f825c6bd2213, PR #22273)
TITLE: Support encoder_only attention for FlexAttention (#22273)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+70/-27); tests/kernels/test_flex_attention.py (+68/-20)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ This PR adds support for encoder-only attention, which is bidirectional and doesn't use KV cache. This type of attention is used in embedding and classifier models such as the Bert and Roberta architecture models. They are already supported with flash attention after PR https://github.com/vllm-project/vllm/pull/21270 but flash attention only supports float16 and bfloat16. To be able to run embedding benchmarks such as MTEB at the high …[truncated]

### L3-1dc8a70b6d  (L3, 2025-08-06, sha 1dc8a70b6d4e, PR #21588)
TITLE: [Attention] Support multiple attention metadata builders per kv_cache_spec  + proper local attention no hybrid kv cache fix (#21588)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.dispatch.selector, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+4/-0); vllm/attention/layer.py (+18/-18); vllm/attention/layers/chunked_local_attention.py (+88/-0); vllm/attention/selector.py (+1/-1); vllm/v1/attention/backends/utils.py (+46/-2); tests/v1/spec_decode/test_eagle.py (+2/-1); tests/v1/worker/test_gpu_model_runner.py (+3/-3); vllm/model_executor/models/llama4.py (+6/-4); vllm/v1/spec_decode/eagle.py (+5/-4); vllm/v1/worker/cpu_model_runner.py (+4/-4); (+3 more)
LABELS: tpu, speculative-decoding, ready, v1, llama
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Support multiple attention metadata builders per kv-cache spec so we can undo the hacky fix in https://github.com/vllm-project/vllm/pull/21707 ⏎  ⏎ ## Test Plan ⏎  ⏎ Use the same ruler task as in: https://github.com/vllm-project/vllm/pull/21707 ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ python -m lm_eval --model vllm --model_args pretrained=meta-llama/Llama-4-Scout-17B-16E-Instruct,tensor_parallel_siz …[truncated]

### L3-6b47ef24de  (L3, 2025-08-06, sha 6b47ef24de3d, PR #22350)
TITLE: [XPU]Fix `flash_attn_varlen_func` interface on xpu (#22350)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/_ipex_ops.py (+1/-0)
LABELS: ready
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ `flash_attention_varlen_func` add s_aux as parameter, make it unblock for xpu path for now. ⏎ ## Test Plan ⏎ UT. ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update
