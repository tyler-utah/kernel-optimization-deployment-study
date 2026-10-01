### L3-9e96f56efb  (L3, 2025-04-25, sha 9e96f56efb5b, PR #16605)
TITLE: Allocate kv_cache with stride order (#16605)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.cache.cuda_reshape,L3.flashinfer.v0_backend,L3.dispatch.abstract_interface; Adds stride-order KV-cache allocation contract exposed by FlashInfer backend.
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flashinfer.v0_backend, L3.dispatch.abstract_interface
FILES: csrc/cache_kernels.cu (+21/-18); vllm/attention/backends/abstract.py (+4/-0); vllm/attention/backends/flashinfer.py (+25/-5); tests/kernels/attention/test_cache.py (+41/-19); vllm/utils.py (+10/-3); vllm/worker/cache_engine.py (+18/-5)
LABELS: ready
BODY: Allow KV cache manager to support an stride order to the allocation which the attention backend could provide. Mainly affect Flashinfer backend. Ref. #8200  ⏎ @tlrmchlsmth @LucasWilkinson

### L3-65e262b93b  (L3, 2025-04-26, sha 65e262b93bef, PR #17159)
TITLE: Fix Python packaging edge cases (#17159)
SOURCES: path_core
STAGE1: repair_build_dependency; artifacts=L3.flash_attn.fork_build; Fixes packaging so vllm_flash_attn subpackage is included in wheels.
ARTIFACT_HINTS: -
FILES: vllm/vllm_flash_attn/__init__.py (+0/-0); .gitignore (+1/-0); pyproject.toml (+1/-2); vllm/benchmarks/__init__.py (+0/-0)
LABELS: ready
ISSUES: #15812 [Bug]: run on cpu:  ModuleNotFoundError: No module named 'vllm.benchmarks'
BODY: Address some edge cases with packaging of Python subpackages in vLLM. The `vllm/benchmark` and `vllm/vllm_flash_attn` directories were missing `__init__.py` files. A subdirectory without an `__init__.py` is considered a namespace package. Because vLLM's `pyproject.toml` file excludes namespace packages, setuptools' package finder excludes the subpackages `vllm.benchmark` and `vllm.vllm_flash_attn` from wheel distributions. ⏎  ⏎ vLLM's wheels still shipped both subpackages, because vLLM also uses `setuptools-scm`. The build dependency hooks into setuptools and modifies its package finder. The override only works if-and-only-if a source checkout has a `.git` VCS directory. ⏎  ⏎ This change turns `vllm.benchmark` and `vllm.vllm_flash_attn` into regular importable packages and removes `namespaces = false` from setuptools' package finder configuration. It also changes the finder excludes into finder includes for `vllm*`, which matches `vllm` root package and all `vllm` subpackages. ⏎  ⏎ FIX #15812

### L3-838cedade7  (L3, 2025-04-27, sha 838cedade77a, PR #17222)
TITLE: [Bugfix] Get a specific type of layer from forward context (#17222)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.flashinfer.v0_backend,L3.flashinfer.v1_backend; FlashInfer backend now fetches only Attention layers from forward context.
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/attention/backends/flashinfer.py (+2/-4); vllm/v1/attention/backends/flashinfer.py (+3/-4); vllm/config.py (+15/-1); vllm/v1/worker/gpu_model_runner.py (+5/-10); vllm/v1/worker/tpu_model_runner.py (+3/-4)
LABELS: tpu, ready, v1
DEEP_STUDY: deep-study correctness case vllm:838cedade7: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: Forward context saves both `Attention` and `FusedMOE` layers. Add an interface to only get one type of layer from forward context. ⏎  ⏎ Fix bug: ⏎ `VLLM_ATTENTION_BACKEND=FLASHINFER python3 examples/offline_inference/data_parallel.py --model="Qwen/Qwen1.5-MoE-A2.7B-Chat" --dp-size=2 --tp-size=1` ⏎ ``` ⏎ (EngineCore_0 pid=3081205)   File “***/vllm/v1/attention/backends/flashinfer.py", line 88, in get_per_layer_parameters ⏎ (EngineCore_0 pid=3081205)     assert isinstance(layer, Attention) ⏎ ```

### L3-ed7a29d9f8  (L3, 2025-04-27, sha ed7a29d9f8b4, PR #16032)
TITLE: [NVIDIA] Support Cutlass MLA for Blackwell GPUs (#16032)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
STAGE1: introduce; artifacts=L3.mla.cutlass_kernels; Adds CUTLASS MLA decode kernels and custom op for Blackwell.
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.mla.cutlass_kernels
FILES: CMakeLists.txt (+24/-4); csrc/attention/mla/cutlass_mla_entry.cu (+38/-0); csrc/attention/mla/cutlass_mla_kernels.cu (+225/-0); csrc/ops.h (+6/-0); csrc/torch_bindings.cpp (+7/-0); vllm/_custom_ops.py (+9/-0); csrc/quantization/fp4/nvfp4_scaled_mm_kernels.cu (+1/-1); tests/kernels/test_cutlass_mla_decode.py (+93/-0)
LABELS: ready, ci/build
BODY: The latest cutlass supports MLA for the blackwell GPUs. Examples can be found [here](https://github.com/NVIDIA/cutlass/blob/main/examples/77_blackwell_fmha/77_blackwell_mla.cu). It should be available in the next release (v3.9). ⏎  ⏎ This PR integrates this kernel as `ops.cutlass_mla_decode`. ⏎  ⏎ cc. @kushanam

### L3-d8bccde686  (L3, 2025-04-27, sha d8bccde68634, PR #17267)
TITLE: [BugFix] Fix vllm_flash_attn install issues (#17267)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
STAGE1: repair_build_dependency; artifacts=L3.flash_attn.fa_utils,L3.flash_attn.fork_build; Fixes vllm_flash_attn installation and introduces fa_utils import wrapper.
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flash_attn.upstream_pip, L3.flash_attn.fa4_cutedsl, L3.flash_attn.fa_utils, L3.mla.triton_v0, L3.mla.common_v1
FILES: setup.py (+19/-7); vllm/attention/backends/flash_attn.py (+3/-3); vllm/attention/backends/mla/common.py (+1/-1); vllm/attention/utils/fa_utils.py (+0/-0); vllm/engine/arg_utils.py (+1/-1); vllm/v1/attention/backends/flash_attn.py (+2/-2); vllm/v1/attention/backends/mla/common.py (+1/-1); vllm/vllm_flash_attn/__init__.py (+0/-22); vllm/vllm_flash_attn/flash_attn_interface.pyi (+0/-245); .github/CODEOWNERS (+1/-0); .gitignore (+0/-2)
LABELS: ready, ci/build, v1
ISSUES: #17263 [Bug]: nightly version: ModuleNotFoundError: No module named 'vllm.vllm_flash_attn.layers'
PERF_LINES: 1) https://github.com/vllm-project/vllm/pull/17159 introduced an `__init__.py` into the `vllm_flash_attn`, during the install process we copy over the [`__init_
BODY: This PR is a collection of fixes for `vllm_flash_attn` install issues, unfortunately the `vllm_flash_attn` install is fairly hacky/sensitive/complex right now. This will hopefully be fixed in the future if we move to a separate kernel library. Apologies for missing https://github.com/vllm-project/vllm/pull/17159 . ⏎  ⏎ 1) https://github.com/vllm-project/vllm/pull/17159 introduced an `__init__.py` into the `vllm_flash_attn`, during the install process we copy over the [`__init__.py`](https://github.com/vllm-project/flash-attention/blob/main/vllm_flash_attn/__init__.py) file from the vllm_flash_attn repo https://github.com/vllm-project/flash-attention. This creates a bad diff since `vllm/vllm_flash_attn/__init__.py` was now being tracked by git. PRs https://github.com/vllm-project/vllm/pull/17228 and https://github.com/vllm-project/vllm/pull/17260 attempted to fix this by not coping the `__init__.py` from vllm_flash_attn repo. This is error prone because it requires us to keep the stub files in-sync between the repos.  I am assuming https://github.com/vllm-project/vllm/pull/17159 introduced an `__init__.py` to make sure `vllm/vllm_flash_attn/fa_utils.py` got packaged correctly since this is the only python file that would exist in that folder if  `__init__.py` from the vllm_flash_attn repo was not copied over. As an alternative in this PR I moved that file to `vllm/attention/utils/fa_utils.py`, this way it always gets packaged regardless if vllm_flash_attn is available, meaning we can remove the `__init__.py` from the folder and rely on the one that gets copied from vllm_flash_attn repo (if its not present we dont care about packaging this folder anymore). (NOTE this PR reverts https://github.com/vllm-project/vllm/pull/17228) ⏎  ⏎ 2) https://github.com/vllm-project/vllm/pull/16457 and https://github.com/vllm-project/flash-attention/pull/64 introduced new python files to `vllm_flash_attn` but these were not getting properly picked up by setup.py, this PR now recursively globs for `.py` files in `vllm_flash_attn` making sure no files are inadvertently missed (FIX https://github.com/vllm-project/vllm/issues/17263), Note  this PR supersedes: https://github.com/vllm-project/vllm/pull/17247 which achieves the same result albeit in more lines of code (shout-out to @jeejeelee and @aarnphm for highlighting the issue).

### L3-cc5befbced  (L3, 2025-04-28, sha cc5befbced22, PR #17283)
TITLE: [BugFix] Fix cascade attention - RuntimeError: scheduler_metadata must have shape (metadata_size) (#17283)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.flash_attn.v1_backend; Fixes cascade attention scheduler_metadata shape runtime error.
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+1/-1)
LABELS: ready, v1
ISSUES: #17276 [Bug]: nightly version:  EngineCore encountered a fatal error.
BODY: FIX https://github.com/vllm-project/vllm/issues/17276

### L3-17eb306fcc  (L3, 2025-04-28, sha 17eb306fcc70, PR #17091)
TITLE: [Bugfix] Add contiguous call inside rope kernel wrapper (#17091)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.mla.common_v1; Makes MLA RoPE wrapper inputs contiguous to avoid incorrect output.
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+3/-4); vllm/_custom_ops.py (+14/-3)
LABELS: ready, v1
ISSUES: #16658 [Bug]: V0 engines gives incorrect output for Moonlight model
BODY: This PR fixes #16658.  ⏎  ⏎ Following @LucasWilkinson's suggestion, I followed the second way: add `.contiguous()` call in the kernel wrappers. Although making the query contiguous is sufficient for RoPE in the MLA backend, I found that a non-contiguous key is also likely to lead to the same issue because the stride along the first dimension is not considered in the kernel. Therefore, I make both query and key contiguous in the wrappers.  ⏎  ⏎ I also added a TODO so that we can remove this in the future once the kernel is modified to support tensor slices.  ⏎  ⏎ The fix is tested on vLLM v0.8.4.

### L3-24e6ad3f16  (L3, 2025-04-29, sha 24e6ad3f16d5, PR #17193)
TITLE: [V1] Remove num_input_tokens from attn_metadata (#17193)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.dispatch.abstract_interface,L3.flash_attn.v1_backend,L3.flashinfer.v1_backend,L3.mla.common_v1; Removes num_input_tokens from AttentionMetadata protocol.
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1
FILES: vllm/v1/attention/backends/flash_attn.py (+0/-3); vllm/v1/attention/backends/flashinfer.py (+0/-3); vllm/v1/attention/backends/mla/common.py (+0/-3); vllm/forward_context.py (+7/-9); vllm/v1/worker/gpu_model_runner.py (+3/-2); vllm/v1/worker/tpu_model_runner.py (+4/-1)
LABELS: tpu, ready, v1
BODY: `num_input_tokens` is not related to attention and only used in `set_forward_context`, so I prefer to remove it from attention_metadata and pass it to set_forward_context explicitly.

### L3-06ffc7e1d3  (L3, 2025-04-29, sha 06ffc7e1d35b, PR #17289)
TITLE: [Misc][ROCm] Exclude `cutlass_mla_decode` for ROCm build (#17289)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_build_dependency; artifacts=L3.mla.cutlass_kernels; Excludes cutlass_mla_decode binding from ROCm builds to avoid import errors.
ARTIFACT_HINTS: -
FILES: csrc/torch_bindings.cpp (+7/-7)
LABELS: ready
BODY: A small fix for https://github.com/vllm-project/vllm/pull/16032. Do not include the op `cutlass_mla_decode` for ROCm build; otherwise it will cause some import error.

### L3-28566d73b3  (L3, 2025-05-01, sha 28566d73b3c7, PR #17536)
TITLE: [ROCm] remove unsupported archs from rocm triton flash-attention supported list (#17536)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: remove; artifacts=L3.triton.flash_attention_rocm; Removes unsupported gfx940/gfx941 from ROCm Triton FlashAttention supported architectures.
ARTIFACT_HINTS: L3.triton.flash_attention_rocm
FILES: vllm/attention/ops/triton_flash_attention.py (+1/-1)
LABELS: rocm, ready
BODY: gfx940 and gfx941 are no longer supported since ROCm 6.3, and also they are not in the PyTorch arch list (env PYTORCH_ROCM_ARCH) . This simple clean up PR is to remove them from the code (here the triton flash-attention supported list).

### L3-811a6c0972  (L3, 2025-05-01, sha 811a6c0972da, PR #16034)
TITLE: [ROCM] Add gfx950 to the custom attention archs (#16034)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L3.rocm.custom_paged,L3.platform.rocm_selection; Adds gfx950 to ROCm custom paged-attention supported architectures.
ARTIFACT_HINTS: L3.rocm.custom_paged, L3.platform.rocm_selection
FILES: csrc/rocm/attention.cu (+6/-5); vllm/platforms/rocm.py (+6/-3)
LABELS: ready
BODY: Adding gfx950 to the supported architectures and small renaming of the variable

### L3-afcb3f8863  (L3, 2025-05-02, sha afcb3f8863ee, PR #17484)
TITLE: [Attention] MLA move o_proj q_proj into cuda-graph region (#17484)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: adapt_framework; artifacts=L3.mla.common_v1; Moves MLA projection work into CUDA-graph region for lower overhead.
ARTIFACT_HINTS: L3.mla.triton_v0, L3.mla.common_v1, L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter
FILES: vllm/attention/backends/cpu_mla.py (+2/-3); vllm/attention/backends/flashmla.py (+1/-1); vllm/attention/backends/mla/common.py (+19/-35); vllm/attention/backends/rocm_aiter_mla.py (+1/-1); vllm/attention/backends/triton_mla.py (+1/-1); vllm/model_executor/models/deepseek_v2.py (+12/-9); vllm/v1/attention/backends/mla/common.py (+17/-33); vllm/v1/attention/backends/mla/flashmla.py (+1/-1); vllm/v1/attention/backends/mla/triton_mla.py (+1/-1)
LABELS: ready, v1
BODY: With: https://github.com/vllm-project/vllm/pull/14770 we are no longer materializing the absorbed matrices so we can move q_proj and o_proj into the cuda-graph region lowering CPU overhead: ⏎  ⏎ # Accuracy ⏎  ⏎ ``` ⏎ VLLM_USE_V1=0 lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384 --task gsm8k --num_fewshot 5  --batch_size auto ⏎  ⏎ vllm (pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6679|±  |0.0130| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.6581|±  |0.0131| ⏎ ``` ⏎  ⏎ ``` ⏎ VLLM_USE_V1=1 lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=1638 ⏎ 4 --task gsm8k --num_fewshot 5  --batch_size auto ⏎  ⏎ vllm (pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6657|±  |0.0130| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.6573|±  |0.0131| ⏎ ``` ⏎  ⏎ CPU implementation confirmed working with: https://github.com/vllm-project/vllm/pull/17494 ⏎  ⏎ # Benchmark Results (DeepSeek-R1, 8xH200) ⏎  ⏎ ## This PR  ⏎ ``` ⏎   input_tokens output_tokens  vllm_output_toks/s ⏎ 0         1000          2000         1183.526109 ⏎ 1         5000          1000          917.148781 ⏎ 2        10000           500          475.100371 ⏎ 3        30000           100           36.287921 ⏎ 4     sharegpt      sharegpt         1415.561950 ⏎ ``` ⏎  ⏎ ## Main ⏎  ⏎ ``` ⏎   input_tokens output_tokens  vllm_output_toks/s ⏎ 0         1000          2000         1113.702931 ⏎ 1         5000          1000          857.432545 ⏎ 2        10000           500          463.061531 ⏎ 3        30000           100           35.905324 ⏎ 4     sharegpt      sharegpt         1379.961186 ⏎ ```

### L3-0f87d8f7b2  (L3, 2025-05-02, sha 0f87d8f7b26d, PR #17574)
TITLE: [BugFix][Attention] Fix sliding window attention in V1 giving incorrect results (#17574)
SOURCES: path_core, symbol_pickaxe, corpus:kernel-correctness-cases
STAGE1: repair_correctness; artifacts=L3.flash_attn.v1_backend; Fixes V1 sliding-window attention incorrect results.
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+35/-1)
LABELS: ready, v1
ISSUES: #17476 [Bug]: Flash attention with sliding window
DEEP_STUDY: deep-study correctness case vllm:0f87d8f7b2: class=shape_alignment_edge; symptom=wrong_output_or_accuracy; introducing=#13111
BODY: The FA3 update to use a AOT scheduler (https://github.com/vllm-project/vllm/pull/13111) did not properly handle sliding window attention. FIX https://github.com/vllm-project/vllm/issues/17476 ⏎  ⏎ Tested using: ⏎ ``` ⏎ python -m pytest -vs tests/v1/e2e/test_correctness_sliding_window.py ⏎ ```

### L3-4c33d67321  (L3, 2025-05-02, sha 4c33d6732148, PR #17438)
TITLE: [Bugfix] fix tmp_out and exp_sums dimensions (#17438)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.triton.chunked_prefill_paged_decode; Fixes tmp_out/exp_sums dimensions in chunked-prefill paged decode.
ARTIFACT_HINTS: L3.triton.chunked_prefill_paged_decode
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+1/-1)
LABELS: bug, ready
BODY: The first dimension of tmp_out and exp_sums is inferred from block_tables.size(0), which may be different from query.shape(0). The later can be much larger than block_tables.size(0), which may cause OOM. ⏎  ⏎ This PR fix the total_num_seq and the comments.

### L3-cba31c47c4  (L3, 2025-05-06, sha cba31c47c481, PR #17394)
TITLE: [v1] AttentionMetadata for each layer (#17394)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.dispatch.abstract_interface,L3.flash_attn.v1_backend,L3.flashinfer.v1_backend,L3.mla.common_v1; Changes ForwardContext to per-layer AttentionMetadata objects for hybrid allocators.
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.dispatch.abstract_interface
FILES: vllm/attention/layer.py (+12/-3); vllm/v1/attention/backends/flash_attn.py (+5/-6); vllm/v1/attention/backends/flashinfer.py (+5/-5); vllm/v1/attention/backends/mla/common.py (+5/-5); vllm/v1/attention/backends/utils.py (+18/-0); vllm/forward_context.py (+8/-3); vllm/v1/spec_decode/eagle.py (+10/-1); vllm/v1/worker/gpu_model_runner.py (+47/-21); vllm/v1/worker/tpu_model_runner.py (+16/-2)
LABELS: tpu, ready, v1
BODY: Should be merge after https://github.com/vllm-project/vllm/pull/17193 ⏎  ⏎ This PR changes ForwardContext.attn_metadata from a global one to dict[layer_name, AttentionMetadata] to prepare for hybrid allocator which allocate different block table to sliding window layers and full attention layers. We only need to build one attention metadata for each kv cache group and let all layers inside that kv cache group point to that attention metadata object.

### L3-f9bc5a0693  (L3, 2025-05-06, sha f9bc5a0693d8, PR #17446)
TITLE: [Bugfix] Fix triton import with local TritonPlaceholder (#17446)
SOURCES: path_core
STAGE1: repair_build_dependency; artifacts=L3.triton.prefix_prefill,L3.triton.decode_attention,L3.triton.chunked_prefill_paged_decode,L3.merge.triton_lse; Fixes TritonPlaceholder imports across Triton attention kernels.
ARTIFACT_HINTS: L3.triton.prefix_prefill, L3.triton.flash_attention_rocm, L3.triton.decode_attention, L3.triton.chunked_prefill_paged_decode, L3.merge.triton_lse, L3.blocksparse.v0
FILES: vllm/attention/ops/blocksparse_attention/blocksparse_attention_kernel.py (+2/-2); vllm/attention/ops/blocksparse_attention/utils.py (+2/-1); vllm/attention/ops/chunked_prefill_paged_decode.py (+1/-2); vllm/attention/ops/prefix_prefill.py (+1/-2); vllm/attention/ops/triton_decode_attention.py (+1/-3); vllm/attention/ops/triton_flash_attention.py (+1/-2); vllm/attention/ops/triton_merge_attn_states.py (+2/-2); benchmarks/kernels/benchmark_moe.py (+1/-1); benchmarks/kernels/benchmark_rmsnorm.py (+1/-1); benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py (+1/-1); tests/kernels/attention/test_flashmla.py (+1/-1); tests/test_triton_utils.py (+92/-0); vllm/lora/ops/triton_ops/kernel_utils.py (+1/-2); vllm/model_executor/layers/fused_moe/fused_moe.py (+1/-2); vllm/model_executor/layers/fused_moe/moe_align_block_size.py (+1/-2); vllm/model_executor/layers/lightning_attn.py (+2/-2); vllm/model_executor/layers/mamba/ops/mamba_ssm.py (+1/-3); vllm/model_executor/layers/mamba/ops/ssd_bmm.py (+2/-2); vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py (+2/-2); vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py (+2/-2); vllm/model_executor/layers/mamba/ops/ssd_combined.py (+2/-1); vllm/model_executor/layers/mamba/ops/ssd_state_passing.py (+2/-2); vllm/model_executor/layers/quantization/awq_triton.py (+2/-2); vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py (+2/-2); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+1/-2); vllm/model_executor/layers/quantization/utils/int8_utils.py (+1/-2); vllm/triton_utils/__init__.py (+10/-2); vllm/triton_utils/importing.py (+31/-29); vllm/v1/sample/rejection_sampler.py (+1/-2); vllm/v1/spec_decode/eagle.py (+1/-2)
LABELS: ready, v1
BODY: Fix triton import error in non-triton platforms with the local `TritonPlaceholder`

### L3-2f925e5777  (L3, 2025-05-06, sha 2f925e5777cc, PR #16828)
TITLE: [Kernel] Unified Triton kernel that doesn't distinguish between prefill + decode (#16828)
SOURCES: path_core, symbol_pickaxe
STAGE1: introduce; artifacts=L3.triton.unified_attention,L3.triton.v1_backend; Introduces unified Triton attention kernel and wires V1 Triton backend to it.
ARTIFACT_HINTS: L3.triton.unified_attention, L3.triton.v1_backend
FILES: vllm/attention/ops/triton_unified_attention.py (+333/-0); vllm/v1/attention/backends/triton_attn.py (+44/-27); tests/kernels/test_triton_unified_attention.py (+189/-0)
LABELS: ready, v1
PERF_LINES: I've run the following scenario across both H100 and MI300x using different backends from main, as well as using the changes from this PR: | - The changes from this PR significantly improve the performance on MI300x at high QPS | - For some reason, using compile seems to make the results worse on MI300x (something that @SageMoore had mentioned to me previously). Do we have an issue to tr
BODY: In this PR we add: ⏎ - A new Triton kernel (`triton_unified_attention`) that works like `flash_attn_varlen_func` and can handle arbitrary query length. The kernel does GQA "packing" along the query dimension to ensure the Tensor cores are well used. ⏎ - Added a new unit test that is based on the unit tests for `flash_attn_varlen_func`  ⏎ - Updated the V1 Triton attention backend to use this kernel. Note that the memory layout for the key cache also changes to match what the Flash attention backend is doing.  ⏎  ⏎ Best performance is obtained when using the jit cache decorator from `triton_dejavu`. In this code I'm using the jit cache decorator from `triton_dejavu` package but if https://github.com/vllm-project/vllm/pull/16606 is merged we could use it directly from vLLM. ⏎  ⏎ Note that the unit tests don't currently pass when I enable the jit cache, but they all pass if it disabled. This is because we are testing different combinations of numbers of heads etc, which we assume to be constant in the decorator. We probably need to think of a good testing strategy for kernels with this decorator (cc @bringlein).  ⏎  ⏎ **Initial benchmarking** ⏎  ⏎ Here are some initial benchmarking results on H100 for `llama3.1-8b` using: ⏎ ``` ⏎ python benchmark_serving.py \ ⏎     --model meta-llama/Llama-3.1-8B-Instruct  \ ⏎     --dataset-name sharegpt \ ⏎     --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json ⏎  ⏎ ``` ⏎ <img width="543" alt="image" src="https://github.com/user-attachments/assets/b4e8bcdb-44d4-4552-a322-4dc5d18d76e4" /> ⏎  ⏎ Note that with these changes, the Triton backend significantly outperforms FlashAttention backend on an H100 GPU for this workloads. ⏎  ⏎ **Correctness** ⏎ ``` ⏎ $ VLLM_USE_V1=1 lm_eval --model vllm --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct --tasks gsm8k --num_fewshot 5 --batch_size auto --limit 500 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.800|±  |0.0179| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.784|±  |0.0184| ⏎ ``` ⏎  ⏎ @bringlein the correctness check only looks good if I use branch `tpa-grid-copy` from triton-dejavu, if I use main it fails. you should be able to reproduce on H100.  ⏎  ⏎ **Further benchmarking** ⏎  ⏎ I've run the following scenario across both H100 and MI300x using different backends from main, as well as using the changes from this PR: ⏎ ```bash ⏎ MODEL=mistralai/Mistral-Small-24B-Instruct-2501 ⏎ REQUEST_RATES=(1 5 7 9) ⏎ TOTAL_SECONDS=120 ⏎  ⏎ for REQUEST_RATE in "${REQUEST_RATES[@]}"; ⏎ do ⏎      …[truncated]

### L3-32aa74c09c  (L3, 2025-05-07, sha 32aa74c09c82, PR #17139)
TITLE: [ROCm][FP8][Kernel] FP8 quantization fused into Custom Paged Attention (#17139)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L3.rocm.custom_paged; Adds FP8 output scaling to ROCm custom paged attention kernel.
ARTIFACT_HINTS: L3.rocm.custom_paged
FILES: csrc/rocm/attention.cu (+61/-31); vllm/_custom_ops.py (+2/-1); csrc/rocm/ops.h (+9/-11); csrc/rocm/torch_bindings.cpp (+2/-1)
LABELS: rocm, ready
BODY: An option to apply fp8 output scale in ROCm custom paged attention and output FP8 tensor ⏎ In case a non-None scale tensor is passed to the kernel, the output tensor is expected to be in the current_platform.fp8_dtype() type (float8_fnuz or float8_fn), and the scale is applied to it before storing into an 8-bit type

### L3-7ea2adb802  (L3, 2025-05-07, sha 7ea2adb8026e, PR #16072)
TITLE: [Core] Support full cuda graph in v1 (#16072)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L3.flash_attn.v1_backend; Makes V1 FlashAttention path compatible with full CUDA graph capture.
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+10/-3); docs/source/design/v1/torch_compile.md (+6/-0); tests/compile/piecewise/test_full_cudagraph.py (+97/-0); vllm/config.py (+17/-2); vllm/v1/worker/gpu_model_runner.py (+60/-8)
LABELS: documentation, ready, ci/build, v1
PERF_LINES: This reduces median TPOT by 7% for small models like Qwen 2.5 1.5B. | Request throughput (req/s):              0.97 | Output token throughput (tok/s):         96.95 | Total Token throughput (tok/s):          1066.46 | Mean TTFT (ms):                          29.08 | Median TTFT (ms):                        28.89 | P99 TTFT (ms):                           36.17 | Mean TPOT (ms):                    
BODY: ## Summary ⏎ Support capturing a single CUDA graph for the entire model's forward pass, instead of piecewise graphs. This requires creating persistent buffers to make attention graphable. Credit to @tlrmchlsmth for the original implementation. ⏎  ⏎ Limitations: ⏎   1. This only works with V1 + FA3, since FA2 currently is not graphable due to an optimization for GQA. ⏎   2. This doesn't work with Cascade Attention. ⏎  ⏎ Work in progress: ⏎   1. Investigating changes needed to make this work with Llama4 / local attention ⏎  ⏎ This reduces median TPOT by 7% for small models like Qwen 2.5 1.5B. ⏎  ⏎ ## Before ⏎ With piecewise, there are multiple kernel launches per layer, with more gaps between the kernel execution (13ms time to decide one token in profiling mode): ⏎ <img width="1643" alt="Screenshot 2025-04-04 at 12 04 24 PM" src="https://github.com/user-attachments/assets/b76b39a2-44ca-4fc3-8a91-a5d50bcd0657" /> ⏎  ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     100        ⏎ Benchmark duration (s):                  103.15     ⏎ Total input tokens:                      100000     ⏎ Total generated tokens:                  10000      ⏎ Request throughput (req/s):              0.97       ⏎ Output token throughput (tok/s):         96.95      ⏎ Total Token throughput (tok/s):          1066.46    ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):                          29.08      ⏎ Median TTFT (ms):                        28.89      ⏎ P99 TTFT (ms):                           36.17      ⏎ -----Time per Output Token (excl. 1st token)------ ⏎ Mean TPOT (ms):                          5.75       ⏎ Median TPOT (ms):                        5.75       ⏎ P99 TPOT (ms):                           6.00       ⏎ ---------------Inter-token Latency---------------- ⏎ Mean ITL (ms):                           5.75       ⏎ Median ITL (ms):                         5.70       ⏎ P99 ITL (ms):                            6.58       ⏎ ================================================== ⏎ ``` ⏎  ⏎ ## After ⏎ There is now a single kernel launch, with almost no gaps between kernel execution (6ms time to decode one token in profiling mode):  ⏎ <img width="1645" alt="Screenshot 2025-04-04 at 12 05 54 PM" src="https://github.com/user-attachments/assets/7819b6d8-ecb5-44a6-a344-41b7a480ea9b" /> ⏎  ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     100        ⏎ Benchmark duration (s):                  103.10     ⏎ Total input tokens:                      100000     ⏎ Total generated tokens:                  10000      ⏎ Request throughput (req/s):              0.97       …[truncated]

### L3-3c9396a64f  (L3, 2025-05-09, sha 3c9396a64fbf, PR #17523)
TITLE: [FEAT][ROCm]: Support AITER MLA on V1 Engine (#17523)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: introduce; artifacts=L3.mla.rocm_aiter; Introduces ROCm AITER MLA backend for the V1 engine.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.mla.common_v1, L3.mla.rocm_aiter, L3.platform.rocm_selection
FILES: docker/Dockerfile.rocm_base (+1/-1); vllm/attention/ops/rocm_aiter_mla.py (+46/-0); vllm/engine/arg_utils.py (+1/-0); vllm/platforms/interface.py (+2/-1); vllm/platforms/rocm.py (+8/-3); vllm/v1/attention/backends/mla/common.py (+6/-5); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+196/-0); tests/kernels/attention/test_attention_selector.py (+4/-1); tests/kernels/attention/test_rocm_attention_selector.py (+4/-2); vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py (+1/-1)
LABELS: rocm, ready, ci/build, v1
PERF_LINES: | Request throughput (req/s)                   | 5.56                                | 5.53              | | | Output token throughput (tok/s)              | 220.64                              | 214.12            | | | Total Token throughput (tok/s)               | 5916.41                             | 5873.93           | | | Mean TTFT (ms)                               | 85618.95                
BODY: ## AITER MLA Support for V1 Engine ⏎  ⏎ This PR implements AITER MLA attention backend support for the V1 engine. The implementation mirrors the V0 engine's established approach from [PR #15893](https://github.com/vllm-project/vllm/pull/15893). ⏎  ⏎ This PR also introduces a new environment variable, `VLLM_ROCM_EXECUTE_MODEL_TIMEOUT`, which specifies the model execution timeout in seconds. This allows for flexible adjustment of execution time, which is helpful since a timeout error was encountered during graph building when enabling AITER MLA ops on the v1 engine. ⏎  ⏎ ### Accuracy Validation ⏎ using the command below: ⏎ `VLLM_ATTENTION_BACKEND=ROCM_AITER_MLA VLLM_USE_V1=1 lm_eval \ ⏎   --model vllm \ ⏎   --model_args pretrained=deepseek-ai/DeepSeek-V3,tensor_parallel_size=8,trust_remote_code=True,max_model_len=32768,block_size=1,enforce_eager=False \ ⏎   --tasks gsm8k  --num_fewshot 5 --batch_size auto ` ⏎  ⏎ Results: ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9492|±  |0.0060| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.9477|±  |0.0061| ⏎  ⏎ ### Performance: ⏎ The results of `benchmarks/benchmark_serving.py` ⏎  ⏎ using the commands below: ⏎ v0 engine = `VLLM_ATTENTION_BACKEND=ROCM_AITER_MLA VLLM_USE_V1=0 python benchmarks/benchmark_serving.py --model deepseek-ai/DeepSeek-V3  --trust-remote-code --dataset-name random` ⏎ v1 engine = `VLLM_ATTENTION_BACKEND=ROCM_AITER_MLA VLLM_USE_V1=1 python benchmarks/benchmark_serving.py --model deepseek-ai/DeepSeek-V3  --trust-remote-code --dataset-name random` ⏎  ⏎ | Metric                                       | ROCm AITER MLA V1                   | ROCm AITER MLA V0 | ⏎ |----------------------------------------------|-------------------------------------|-------------------| ⏎ | Successful requests                          | 1000                                | 1000              | ⏎ | Benchmark duration (s)                       | 179.78                              | 180.92            | ⏎ | Total input tokens                           | 1024000                             | 1024000           | ⏎ | Total generated tokens                       | 39667                               | 38739             | ⏎ | Request throughput (req/s)                   | 5.56                                | 5.53              | ⏎ | Output token throughput (tok/s)              | 220.64                              | 214.12            | ⏎ | Total Token throughput (tok/s)               | 5916.41                             | …[truncated]

### L3-5e6f939484  (L3, 2025-05-09, sha 5e6f93948449, PR #17668)
TITLE: [Attention] MLA move rotary embedding to cuda-graph region (#17668)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: adapt_framework; artifacts=L3.mla.common_v1,L3.mla.flashmla_v1_adapter; Moves MLA rotary embedding into CUDA-graph region and updates adapters.
ARTIFACT_HINTS: L3.mla.triton_v0, L3.mla.common_v1, L3.mla.flashmla_v1_adapter
FILES: vllm/attention/backends/mla/common.py (+11/-60); vllm/attention/backends/rocm_aiter_mla.py (+2/-4); vllm/model_executor/models/deepseek_v2.py (+7/-1); vllm/v1/attention/backends/mla/common.py (+11/-51); vllm/v1/attention/backends/mla/flashmla.py (+1/-3); vllm/model_executor/layers/rotary_embedding.py (+3/-2)
LABELS: ready, v1
BODY: Following on from https://github.com/vllm-project/vllm/pull/17484, with: https://github.com/vllm-project/vllm/pull/14770 we are no longer materializing the absorbed matrices so we can move the rotary embeddings into the cuda-graph region lowering CPU overhead: ⏎  ⏎ # Perf ⏎  ⏎ ## Main ⏎  ⏎ CPU time for `unified_attention` is 216us ⏎ <img width="1393" alt="image" src="https://github.com/user-attachments/assets/d81d15a3-17a6-4dc3-9c02-03a3d2392daf" /> ⏎  ⏎ ## PR ⏎  ⏎ CPU time for `unified_attention` is 149us ⏎ <img width="1289" alt="image" src="https://github.com/user-attachments/assets/e813d452-f4ef-43e8-894c-f22ab8e49b74" /> ⏎  ⏎ # Accuracy ⏎  ⏎ ``` ⏎ VLLM_USE_V1=1 lm-eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16 ⏎ 384 --task gsm8k --num_fewshot 5  --batch_size auto ⏎ ... ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6649|±  |0.0130| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.6558|±  |0.0131| ⏎ ``` ⏎  ⏎ ``` ⏎ VLLM_USE_V1=0 lm-eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16 ⏎ 384 --task gsm8k --num_fewshot 5  --batch_size auto ⏎ ... ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6649|±  |0.0130| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.6550|±  |0.0131| ⏎ ```

### L3-950751a987  (L3, 2025-05-10, sha 950751a9870f, PR #17483)
TITLE: [v1] Pass BlockTable and KVCacheSpec to AttentionMetadataBuilders (#17483)
SOURCES: path_core, release_notes
STAGE1: adapt_framework; artifacts=L3.dispatch.abstract_interface,L3.flash_attn.v1_backend,L3.flashinfer.v1_backend,L3.mla.common_v1; Passes BlockTable/KVCacheSpec directly to AttentionMetadataBuilders.
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/flash_attn.py (+30/-17); vllm/v1/attention/backends/flashinfer.py (+20/-15); vllm/v1/attention/backends/mla/common.py (+15/-8); vllm/v1/attention/backends/mla/flashmla.py (+7/-4); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+5/-2); tests/v1/worker/test_gpu_input_batch.py (+3/-0); tests/v1/worker/test_gpu_model_runner.py (+20/-1); vllm/v1/worker/block_table.py (+11/-0); vllm/v1/worker/gpu_input_batch.py (+3/-0); vllm/v1/worker/gpu_model_runner.py (+9/-12); vllm/v1/worker/tpu_model_runner.py (+9/-9)
LABELS: tpu, ready, v1
BODY: Should merge after https://github.com/vllm-project/vllm/pull/17394 ⏎  ⏎ Hybrid allocator will need to build attention metadata for each kv cache group because different kv cache groups may have different attention type and block_table. To achieve that, we will introduce one AttentionMetadataBuilder and one BlockTable for each group. ⏎  ⏎ To prepare for this, this PR makes AttentionMetadataBuilder to access its block_table and KVCacheSpec, instead of reading from model_runner. ⏎  ⏎ And as slot_mapping will also be different for different kv cache groups, this pr moves the slot_mapping_cpu tensor from runner to BlockTable. ⏎  ⏎ Splitted from https://github.com/vllm-project/vllm/pull/16101

### L3-eea22a56ab  (L3, 2025-05-11, sha eea22a56ab08, PR #17871)
TITLE: fix amd triton mla path (#17871)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.mla.triton_v0; Fixes AMD Triton MLA branch selection path.
ARTIFACT_HINTS: L3.mla.triton_v0
FILES: vllm/attention/backends/mla/common.py (+1/-1)
LABELS: ready
BODY: Summary: Should be `elif` rather than a new if branch. ⏎  ⏎ Differential Revision: D74436575

### L3-7de18d541b  (L3, 2025-05-11, sha 7de18d541b0d, PR #17961)
TITLE: [BUG] [ROCm] [MLA] Fix variable name bug due to change in variable name in PR #17483 (#17961)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.mla.rocm_aiter; Fixes AITER MLA variable rename breakage after metadata-builder change.
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+3/-3)
LABELS: ready, v1
BODY: This is to fix the error due to change of name introduced by PR https://github.com/vllm-project/vllm/pull/17483 ⏎  ⏎ the rename of the `block_table` to `block_table_tensor` in `_build_decode` function had broken `AiterMLAMetadataBuilder._build_decode`

### L3-06c0922a69  (L3, 2025-05-11, sha 06c0922a69c1, PR #17870)
TITLE: [FP8][ROCm][Attention] Enable FP8 KV cache on ROCm for V1 (#17870)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L3.triton.chunked_prefill_paged_decode,L3.triton.v1_backend; Enables FP8 KV-cache support in ROCm V1 Triton attention fallback.
ARTIFACT_HINTS: L3.triton.chunked_prefill_paged_decode, L3.triton.v1_backend
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+2/-1); vllm/engine/arg_utils.py (+3/-1); vllm/v1/attention/backends/triton_attn.py (+12/-6)
LABELS: ready, v1
BODY: Enable FP8 KV cache support in the unified triton kernel for ROCm ⏎ Also fix the chunked_prefill_paged_decode in case it falls back from custom_paged_attention to the triton kernel ⏎  ⏎ cc @tdoublep

### L3-60f7624334  (L3, 2025-05-12, sha 60f76243344d, PR #11844)
TITLE: Implements dual-chunk-flash-attn backend for dual chunk attention with sparse attention support (#11844)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: introduce; artifacts=NEW:dual_chunk_flash_attn_backend; Introduces dual-chunk FlashAttention backend with sparse-attention support.
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.platform.cuda_selection
FILES: CMakeLists.txt (+1/-0); csrc/attention/vertical_slash_index.cu (+401/-0); csrc/ops.h (+25/-0); csrc/torch_bindings.cpp (+23/-0); vllm/_custom_ops.py (+95/-0); vllm/attention/backends/dual_chunk_flash_attn.py (+1494/-0); vllm/config.py (+19/-0); vllm/engine/arg_utils.py (+13/-2); vllm/platforms/cuda.py (+4/-0); vllm/platforms/interface.py (+1/-0); vllm/utils.py (+1/-0); vllm/worker/model_runner.py (+12/-0); examples/offline_inference/qwen_1m.py (+66/-0); vllm/model_executor/layers/rotary_embedding.py (+202/-2); vllm/model_executor/model_loader/weight_utils.py (+33/-0); vllm/model_executor/models/qwen2.py (+35/-21); vllm/model_executor/models/qwen2_moe.py (+19/-7)
LABELS: documentation, ready, ci/build
ISSUES: #12452 [Feature]: Support Qwen/Qwen2.5-14B-Instruct-1M
BODY: This PR implements the [dual-chunk flash attention](https://arxiv.org/pdf/2402.17463.pdf), a training-free method to extend model context length (see also #6139), with sparse attention (https://github.com/microsoft/MInference) support. ⏎  ⏎ This PR requires the [sparse attention kernel](https://github.com/vllm-project/flash-attention/pull/33) from vllm-flash-attention. Qwen models with 1m context length support will be open-sourced in the next one or two weeks, and unit tests will be added later. ⏎  ⏎ FIX #12452

### L3-176a95c670  (L3, 2025-05-13, sha 176a95c670f6, PR #18104)
TITLE: [Fix] Support CUDAGraph capture for encoder-decoder on ROCm (#18104)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.rocm.rocm_flash_attn_v0; Allows ROCm encoder-decoder attention backend to be CUDA-graph captured.
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/utils.py (+8/-8)
LABELS: bug, rocm, ready
BODY: On `main`, encoder-decoder models are supported on ROCm with `--enforce-eager`, but when capturing CUDA Graphs, a stale assert is triggered, saying the `ROCM_FLASH` attention backend isn't supported (which it is as it runs successfully in eager mode). This PR removes that assert. ⏎  ⏎ Tested with: ⏎ ``` ⏎ vllm serve openai/whisper-large-v3  ⏎ ```

### L3-12e6c0b41c  (L3, 2025-05-13, sha 12e6c0b41c19, PR #18086)
TITLE: [Bugfix][V1] Fix FlashInfer V1 backend using the wrong VllmConfig (#18086)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.flashinfer.v1_backend; Reads FlashInfer V1 config from runner instead of stale global VLLM config.
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+2/-3)
LABELS: bug, ready, v1
BODY: FIX https://github.com/vllm-project/vllm/pull/17483#issuecomment-2875609788 ⏎  ⏎ Thanks to @chenyang78 for reporting and @heheda12345 for the fix. ⏎  ⏎ The global vllm config may not be set by `set_current_vllm_config`, so we should read it from runner directly so `self.vllm_config = runner.vllm_config`. This is already what FlashInfer V0 does so this was likely just an oversight https://github.com/vllm-project/vllm/blob/19324d660c61a63c6ea3dfbb18995d255c05ee6d/vllm/attention/backends/flashinfer.py#L200 ⏎  ⏎ Before: ⏎ ``` ⏎ VLLM_USE_V1=1 VLLM_ATTENTION_BACKEND=FLASHINFER lm_eval --model vllm --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto  ⏎  ⏎ assert len(per_layer_params) > 0, "No attention layers found in the model." ⏎ ``` ⏎  ⏎ After: ⏎ ``` ⏎ VLLM_USE_V1=1 VLLM_ATTENTION_BACKEND=FLASHINFER lm_eval --model vllm --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto  ⏎  ⏎ vllm (pretrained=meta-llama/Llama-3.1-8B-Instruct,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7892|±  |0.0112| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.7635|±  |0.0117| ⏎ ```

### L3-4f8b373225  (L3, 2025-05-13, sha 4f8b37322561, PR #17912)
TITLE: [BugFix][AMD] Compatible patch for AITER lib after 04/20 (#17912)
SOURCES: path_core
STAGE1: repair_build_dependency; artifacts=L3.mla.rocm_aiter; Adapts ROCm AITER MLA wrapper to upstream AITER API changes.
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/rocm_aiter_mla.py (+37/-12); vllm/attention/ops/rocm_aiter_mla.py (+12/-1); vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py (+5/-4)
LABELS: ready
BODY: 1. Changes to adapt new AITER MLA API (MTP is not enabled) ⏎ 2. Some other bug fixes. ⏎  ⏎ This PR is for AITER API changes introduced by commit 939f741fc37f46694e48c32c7164f49eae2584c4 (merged on 04/20/2025). AITER versions after this commit require this patch to work.

### L3-01c22335ba  (L3, 2025-05-15, sha 01c22335baa0, PR #18161)
TITLE: [Kernel] [V1] Fix performance regression for triton unified attention (#18161)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_performance; artifacts=L3.triton.unified_attention,L3.triton.v1_backend; Fixes ~40% Triton 3.3 performance regression in unified attention.
ARTIFACT_HINTS: L3.triton.unified_attention, L3.triton.v1_backend
FILES: vllm/attention/ops/triton_unified_attention.py (+2/-2); vllm/v1/attention/backends/triton_attn.py (+16/-3)
LABELS: ready, v1
PERF_LINES: We have observed a pretty severe (~40%) performance regression for the `triton_unified_attention` kernel when moving from triton 3.2 to triton 3.3. | Request throughput (req/s):              37.62 | Output token throughput (tok/s):         4785.69 | Total Token throughput (tok/s):          12798.78 | Mean TTFT (ms):                          4889.40 | Median TTFT (ms):                        4844.8
BODY: We have observed a pretty severe (~40%) performance regression for the `triton_unified_attention` kernel when moving from triton 3.2 to triton 3.3.  ⏎  ⏎ After a lot of investigation, I was able to figure out that it came from [this](https://github.com/triton-lang/triton/pull/5512) commit to Triton. This change reworked the way that Triton determines what kernel arguments are constant. It seems that before this PR, Triton was (correctly) detecting that `stride_k_cache_3` and `stride_v_cache_3` are constant and leveraging this to obtain a faster kernel. For whatever reason, the new logic doesn't do this and we need to explicitly tell the compiler to interpret these strides as constant. ⏎  ⏎ This minor change recovers the performance.  ⏎  ⏎ There is one other small change: the TritonBackend no longer works on main because of [this](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py#L287) assert. Since the TritonBackend re-uses the `FlashAttentionMetadata` and `FlashAttentionMetadataBuilder` directly (this is intentional to reduce duplicate code), this assert doesn't really make sense unless `TritonAttentionImpl` inherits from `FlashAttentionImpl`. Happy to consider other ways to solve that, but would like to avoid create entirely new `TritonAttentionMetadata` etc that are just duplicates.  ⏎  ⏎ cc @SageMoore @bringlein  ⏎    ⏎ Main branch: ⏎ ``` ⏎ ++++++++++++++++++ Repetition 0 ++++++++++++++++++ ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     985        ⏎ Benchmark duration (s):                  26.18      ⏎ Total input tokens:                      209782     ⏎ Total generated tokens:                  125289     ⏎ Request throughput (req/s):              37.62      ⏎ Output token throughput (tok/s):         4785.69    ⏎ Total Token throughput (tok/s):          12798.78   ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):                          4889.40    ⏎ Median TTFT (ms):                        4844.86    ⏎ P99 TTFT (ms):                           8946.27    ⏎ -----Time per Output Token (excl. 1st token)------ ⏎ Mean TPOT (ms):                          110.97     ⏎ Median TPOT (ms):                        53.41      ⏎ P99 TPOT (ms):                           390.86     ⏎ ---------------Inter-token Latency---------------- ⏎ Mean ITL (ms):                           46.36      ⏎ Median ITL (ms):                         28.78      ⏎ P99 ITL (ms):                            324.39     ⏎ ================================================== ⏎  ⏎ =++++++++++++++++++ Repetition 1 ++++++++++++++++++ ⏎ ============ Serving Benchma …[truncated]

### L3-e6b8e65d2d  (L3, 2025-05-15, sha e6b8e65d2d68, PR #18013)
TITLE: [Bugfix] Fix fp8 tests for triton_unified_attention for Triton 3.3 (#18013)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.triton.unified_attention; Fixes Triton 3.3 FP8 unified-attention test/runtime issue.
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+4/-0); tests/kernels/attention/test_triton_unified_attention.py (+3/-0)
LABELS: ready
BODY: The FP8 unit tests for triton_unified_attention don't pass for Triton 3.3. ⏎  ⏎ We didn't catch this through CI when upgrading Triton because the corresponding file hadn't been moved into the new "kernels" subfolder. ⏎  ⏎ This PR resolves both issues. ⏎  ⏎ cc @LucasWilkinson

### L3-ee659e3b60  (L3, 2025-05-15, sha ee659e3b601e, PR #18093)
TITLE: [Bugfix][ROCm] Use `chunked_prefill_paged_decode` as fallback for V1 attention on ROCm (#18093)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L3.triton.v1_backend,L3.triton.chunked_prefill_paged_decode; Falls back to chunked_prefill_paged_decode when ROCm unified attention cannot handle head ratio.
ARTIFACT_HINTS: L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+77/-32)
LABELS: rocm, ready, v1
BODY: On ROCm, vLLM’s V1 engine uses the unified attention kernel as its sole attention backend. However, at the moment this kernel fails when running models where the number of query heads over the number of key-value heads is not a power of two. This makes models like Llama-4-Scout, whose `num_queries_per_kv` evaluates to an odd number, fail to run and yield the following error on ROCm: ⏎  ⏎ ``` ⏎ offs_m = tl.arange(0, BLOCK_Q * num_queries_per_kv) ⏎          ^ ⏎ ValueError: arange's range must be a power of 2 ⏎ ``` ⏎  ⏎ This PR addresses this issue by adding back the `chunked_prefill_paged_decode` kernel as a fallback for cases where the input tensor shapes are incompatible with the unified attention kernel.

### L3-dcfe95234c  (L3, 2025-05-17, sha dcfe95234c11, PR #18095)
TITLE: Update Dockerfile to build for Blackwell (#18095)
SOURCES: dependency_pin
STAGE1: repair_build_dependency; artifacts=L3.flashinfer.v1_backend; Docker includes latest FlashInfer build for Blackwell attention support.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+3/-3)
LABELS: ready, ci/build
ISSUES: #17325 [Feature]: Integrate FlashInfer Blackwell kernels
BODY: Updates the docker to build wheels for blackwell (SM 10.0) and include the latest flashinfer for performance blackwell attention support (FIX https://github.com/vllm-project/vllm/issues/17325). We didn't include SM 12.0 for now because of wheel size concerns. ⏎  ⏎ Updates to latest flashinfer main as of 5/15 since there isn't a release yet: https://github.com/flashinfer-ai/flashinfer/commit/e00e8cedbfcb220f328fd36aa8f529f869b01e6b

### L3-dd5fa7e04f  (L3, 2025-05-21, sha dd5fa7e04f75, PR #17004)
TITLE: [ROCm][Kernel][V1] Enable AMD Radeon GPU Custom Paged Attention on v1 (#17004)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L3.rocm.custom_paged,L3.platform.rocm_selection; Adds AMD gfx11/gfx12 custom paged-attention kernels and selection support.
ARTIFACT_HINTS: L3.triton.chunked_prefill_paged_decode, L3.rocm.custom_paged, L3.rocm.rocm_flash_attn_v0, L3.platform.rocm_selection
FILES: csrc/rocm/attention.cu (+1880/-171); vllm/attention/backends/rocm_flash_attn.py (+2/-1); vllm/attention/ops/chunked_prefill_paged_decode.py (+2/-1); vllm/platforms/rocm.py (+34/-14); benchmarks/kernels/benchmark_paged_attention.py (+5/-1); tests/kernels/attention/test_attention.py (+7/-1)
LABELS: ready
PERF_LINES: Request throughput (req/s) | 7.71 | 8.98 | 4.36 | 5.19 | Output token throughput (tok/s) | 1520.75 | 1768.76 | 859.67 | 1023.06 | Total Token throughput (tok/s) | 3179.21 | 3701.28 | 1797.83 | 2139.86 | Mean TTFT (ms) | 38565.45 | 34609.46 | 69690.52 | 61926.23 | Median TTFT (ms) | 34804.15 | 31791.33 | 62976.49 | 57000.47 | P99 TTFT (ms) | 91181.45 | 79898.55 | 165778.94 | 144393.74 | Mean TPOT (
BODY: Add additional custom paged attention kernels for AMD gfx11/gfx12 GPU support. Based on PRs:  https://github.com/vllm-project/vllm/pull/12348 https://github.com/vllm-project/vllm/pull/15720 https://github.com/vllm-project/vllm/pull/13843 ⏎  ⏎ Due to the differences in architecture from MI, specific instructions and detailed logic have changed (mfma16 -> wmma16/wmma16_gfx12), so new kernels for each architecture has been added. ⏎  ⏎ - Supports cases where head_size == 128 and block_size == 16. ⏎ - It does not support alibi_slopes and kv_cache_dtype == fp8. ⏎ - It supports gqa_ratio up to 16, and shows performance gains over the existing kernel when gqa_ratio is 3 or higher. Therefore, it is enabled for gqa_ratio values between 3 and 16. ⏎  ⏎ **Performance** ⏎ python ./benchmarks/benchmark_serving.py  --model /home/user/workspace/models/Llama-3.1-8B-Instruct --dataset-name sharegpt --dataset-path /home/user/workspace/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json --port 8000 ⏎   | gfx12 |   | gfx11 |   ⏎ -- | -- | -- | -- | -- ⏎   | v1 (original) | v1+CPA (this PR) | v1 (original) | v1+CPA (this PR) ⏎ Successful requests | 1000 | 1000 | 1000 | 1000 ⏎ Benchmark duration (s) | 129.76 | 111.36 | 229.38 | 192.69 ⏎ Total input tokens | 215196 | 215196 | 215196 | 215196 ⏎ Total generated tokens | 197328 | 196961 | 197190 | 197134 ⏎ Request throughput (req/s) | 7.71 | 8.98 | 4.36 | 5.19 ⏎ Output token throughput (tok/s) | 1520.75 | 1768.76 | 859.67 | 1023.06 ⏎ Total Token throughput (tok/s) | 3179.21 | 3701.28 | 1797.83 | 2139.86 ⏎ Mean TTFT (ms) | 38565.45 | 34609.46 | 69690.52 | 61926.23 ⏎ Median TTFT (ms) | 34804.15 | 31791.33 | 62976.49 | 57000.47 ⏎ P99 TTFT (ms) | 91181.45 | 79898.55 | 165778.94 | 144393.74 ⏎ Mean TPOT (ms) | 151.15 | 133.5 | 275.79 | 241.8 ⏎ Median TPOT (ms) | 137.91 | 119.24 | 253.68 | 216.16 ⏎ P99 TPOT (ms) | 379.85 | 370.63 | 703.72 | 689.53 ⏎ Mean ITL (ms) | 127.7 | 110.06 | 231.54 | 196.25 ⏎ Median ITL (ms) | 95.57 | 80.11 | 178.23 | 144.42 ⏎ P99 ITL (ms) | 385.62 | 368.52 | 713.23 | 692.44 ⏎  ⏎ **Correctness** ⏎ vllm (pretrained=/home/user/workspace/models/Llama-3.1-8B-Instruct,max_model_len=4096), gen_kwargs: (None), limit: 500.0, num_fewshot: 5, batch_size: auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.798|±  |0.0180| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.780|±  |0.0185|

### L3-94d8ec8d2b  (L3, 2025-05-21, sha 94d8ec8d2bcb, PR #18338)
TITLE: [FEAT][ROCm] Upgrade AITER MLA v1 backend (#18338)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_build_dependency; artifacts=L3.mla.rocm_aiter; Upgrades AITER MLA V1 backend for new upstream AITER API constraints.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.mla.rocm_aiter
FILES: docker/Dockerfile.rocm_base (+1/-1); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+30/-6)
LABELS: ready, ci/build, v1
PERF_LINES: For example, on MI300X AMD devices, the following configurations are required:
BODY: This PR upgrades the AITER MLA attention backend on the v1 engine, as this backend was previously upgraded on the v0 engine in [this PR](https://github.com/vllm-project/vllm/pull/17912). The upgrade uses AITER commit [c1debd8](https://github.com/ROCm/aiter/commit/c1debd87ce0391aa27438d9e07e76e4fea7c4b70), which introduces the relevant API changes. ⏎  ⏎ It's important to note that the `mla_decode_fwd` kernel in the AITER package imposes new constraints on supported attention heads, requiring either 128 or 16 heads. To load DeepSeek models correctly, you must configure the `tensor_parallel_size` parameter so that each rank processes 16 heads.  ⏎  ⏎ For example, on MI300X AMD devices, the following configurations are required: ⏎  ⏎ - DeepSeek-v3/DeepSeek-v2 (128 total heads): Use tensor_parallel_size=8 (128 ÷ 16 = 8 ranks) ⏎  ⏎  - DeepSeek-v2-Lite (16 total heads): Use tensor_parallel_size=1 (16 ÷ 16 = 1 rank) ⏎  ⏎ ## lm_eval Results ⏎  ⏎ ### deepseek-ai/DeepSeek-V3 on v1 engine ⏎ `VLLM_USE_V1=1 ⏎ VLLM_ATTENTION_BACKEND="ROCM_AITER_MLA" ⏎ SAFETENSORS_FAST_GPU=1 ⏎ lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V3,tensor_parallel_size=8,max_model_len=32768,block_size=1 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto` ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9439|±  |0.0063| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.9439|±  |0.0063|

### L3-6550114c9c  (L3, 2025-05-23, sha 6550114c9cd2, PR #18593)
TITLE: [v1] Redo "Support multiple KV cache groups in GPU model runner (#17945)" (#18593)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L3.dispatch.abstract_interface,L3.mla.rocm_aiter; Relands multiple KV-cache group support and per-group attention metadata.
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+2/-2); tests/v1/core/test_kv_cache_utils.py (+68/-3); tests/v1/core/test_prefix_caching.py (+18/-18); tests/v1/worker/test_gpu_input_batch.py (+33/-6); tests/v1/worker/test_gpu_model_runner.py (+41/-16); vllm/distributed/kv_transfer/kv_connector/v1/shared_storage_connector.py (+3/-3); vllm/v1/core/kv_cache_manager.py (+21/-13); vllm/v1/core/kv_cache_utils.py (+6/-7); vllm/v1/core/sched/output.py (+6/-6); vllm/v1/core/sched/scheduler.py (+10/-6); vllm/v1/kv_cache_interface.py (+42/-0); vllm/v1/worker/block_table.py (+41/-0); vllm/v1/worker/gpu_input_batch.py (+6/-6); vllm/v1/worker/gpu_model_runner.py (+152/-101); vllm/v1/worker/tpu_model_runner.py (+20/-16)
LABELS: tpu, ready, v1
BODY: Redo #17945 that reverted by https://github.com/vllm-project/vllm/pull/18459 to make CI green. ⏎  ⏎ This PR finishes the hybrid allocator support on worker side. It does the following things: ⏎ 1. change `block_ids` in SchedulerOutput to `list[list[int]]`, where the outer list is for multiple kv cache groups and inner list is for blocks in one group. ⏎ 2. Create `BlockTable` class for each kv cache group. ⏎ 3. Build different attention metadata for each kv cache group. ⏎ 4. TPU backend still only supports one KVCacheGroup after this PR. ⏎  ⏎ Splitted from https://github.com/vllm-project/vllm/pull/16101 ⏎  ⏎ NOTE: I still don't know why #17945 fails on `quantization/test_cpu_offload.py::test_cpu_offload_gptq`. Compared with #17945, the only change of this PR is to initialize `self.input_batch` in `GPUModelRunner.__init__` instead of `GPUModelRunner.initialize_kv_cache`. In future hybrid allocator PRs, I plan to add a hack to re-initialize `self.input_batch` in `initializ_kv_cache` if there are more than one KV cache group, and enable hybrid allocator only when weight offloading is not enabled.

### L3-a3896c7f02  (L3, 2025-05-27, sha a3896c7f0216, PR #18570)
TITLE: [Build] Fixes for CMake install (#18570)
SOURCES: path_core, symbol_pickaxe, dependency_pin
STAGE1: repair_build_dependency; artifacts=L3.flash_attn.fork_build; Fixes CMake install logic for vllm_flash_attn external project.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake, L3.flash_attn.fork_build
FILES: CMakeLists.txt (+5/-0); cmake/external_projects/vllm_flash_attn.cmake (+18/-2); setup.py (+1/-4); cmake/utils.cmake (+1/-1)
LABELS: ready, ci/build
BODY: This fixes the CMake install logic such that using `cmake --install` directly is supported. Necessary to fix CMake-based user workflows. ⏎  ⏎ It also fixes the error on ROCm if CMake is used outside the venv (common for IDE-run CMake).

### L3-ce75efeecb  (L3, 2025-05-28, sha ce75efeecb57, PR #18807)
TITLE: [BugFix] FA2 MLA Accuracy Issue (#18807)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.merge.cuda_lse,L3.mla.common_v1; Fixes FA2 MLA accuracy by avoiding sliced merge_attn_states inputs.
ARTIFACT_HINTS: L3.merge.cuda_lse, L3.mla.triton_v0, L3.mla.common_v1
FILES: csrc/attention/merge_attn_states.cu (+8/-0); vllm/attention/backends/mla/common.py (+4/-4); vllm/v1/attention/backends/mla/common.py (+4/-4)
LABELS: ready, v1
ISSUES: #18561 [Bug]: MLA correctness issues when using FA2 | #18766 [CI Failure]: LM Eval Large Models - test_lm_eval_correctness.py
BODY: FIX https://github.com/vllm-project/vllm/issues/18561 ⏎ FIX #18766 ⏎  ⏎ `merge_attn_states` doesnt support tensor slices, so avoid slicing till after all the attn states are merged ⏎  ⏎ After this PR (tested on an A100): ⏎  ⏎ ``` ⏎ lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto ⏎ ... ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6672|±  |0.0130| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.6588|±  |0.0131| ⏎ ``` ⏎  ⏎ ``` ⏎ VLLM_USE_V1=0 lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto ⏎ ... ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6702|±  |0.0129| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.6649|±  |0.0130| ⏎ ```

### L3-269d901734  (L3, 2025-05-29, sha 269d90173432, PR #18100)
TITLE: [Bugfix][ROCm] fix the power of 2 exception from triton_unified_attention.py when running llama4 models and unit test fix (#18100)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.triton.unified_attention; Fixes ROCm unified attention power-of-two exception for Llama4 models.
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+51/-55); tests/kernels/attention/test_triton_unified_attention.py (+3/-1)
LABELS: ready
PERF_LINES: test_triton_unified_attention.py .................................................................................................................. [  9%] | ................................................................................................................................................... [ 22%] | ......................................................................................
BODY: FIX  *https://github.com/vllm-project/vllm/issues/18088* ⏎  ⏎ As detailed in the above issue, when running V1 on llama4 issues, we saw the exception that requires the parameter is a power of 2. However, when running on llama4 128E FP8 models, the following expression in (https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py#L97) is not a power of 2. ⏎ ``` ⏎  offs_m = tl.arange(0, BLOCK_Q * num_queries_per_kv) ⏎ ``` ⏎  ⏎ Debugging found that those values are: ⏎ print("BLOCK_Q:", BLOCK_Q) -> 3 ⏎ print("num_queries_per_kv:", num_queries_per_kv) -> 5 ⏎ print("Product:", BLOCK_Q * num_queries_per_kv) -> 15 ⏎  ⏎ Noticed if we pass the BLOCK_M as the parameter which is hard-coded to 16 now, we can prevent this power of two issue, and also simplify the code without needing the padding. ⏎ ``` ⏎ BLOCK_M = 16 ⏎ BLOCK_Q = BLOCK_M // num_queries_per_kv ⏎ ``` ⏎ So, the `BLOCK_Q * num_queries_per_kv` essentially = `BLOCK_M` ⏎  ⏎ (2) It also uses a tl.constexpr BLOCK_M to replace many places to avoid multiple re-calculations of the same expression later on. ⏎  ⏎ (3) This PR also fixed the the test_triton_unified_attention.py so that it can run successfully on ROCm. ⏎  ⏎ Tests: ⏎  ⏎ Initially, the `test_triton_unified_attention.py` would abort on ROCm platform. After the unit test fix, it is able to run the full test suite without issue.  ⏎  ⏎ (1) Passed the test_triton_unified_attention.py ⏎ ``` ⏎ root@tx:/vllm/tests/kernels/attention# pytest test_triton_unified_attention.py ⏎ /usr/local/lib/python3.12/dist-packages/pytest_asyncio/plugin.py:208: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset. ⏎ The event loop scope for asynchronous fixtures will default to the fixture caching scope. Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to function scope. Set the default fixture loop scope explicitly in order to avoid unexpected behavior in the future. Valid fixture loop scopes are: "function", "class", "module", "package", "session" ⏎  ⏎   warnings.warn(PytestDeprecationWarning(_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET)) ⏎ =================================================================== test session starts =================================================================== ⏎ platform linux -- Python 3.12.10, pytest-8.3.5, pluggy-1.6.0 ⏎ rootdir: /dockerx/vllm ⏎ configfile: pyproject.toml ⏎ plugins: anyio-4.9.0, asyncio-1.0.0 ⏎ asyncio: mode=Mode.STRICT, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function ⏎ collected 1152 items                                                                              …[truncated]

### L3-da4b69d0b4  (L3, 2025-05-29, sha da4b69d0b435, PR #18275)
TITLE: [Attention][V1] Toggle for v1 attention backend (#18275)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: change_default; artifacts=L3.triton.v1_backend,L3.triton.chunked_prefill_paged_decode; Adds toggle to force V1 fallback from unified to two-stage Triton attention.
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.triton.prefix_prefill, L3.triton.chunked_prefill_paged_decode, L3.triton.v1_backend
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+2/-2); vllm/attention/ops/prefix_prefill.py (+2/-2); vllm/envs.py (+10/-2); vllm/v1/attention/backends/triton_attn.py (+6/-3)
LABELS: ready, v1
BODY: Expanding on https://github.com/vllm-project/vllm/pull/18093 ⏎ Adding a toggle to force fallback to the 2 stage attention kernel in V1 ⏎  ⏎ Including a small fix for the FP8 kv cache on ROCm in the 2 stage kernel approach

### L3-1b7cfd5a36  (L3, 2025-05-29, sha 1b7cfd5a367b, PR #18226)
TITLE: [ROCm][V0][Attention] Revert to the previous FA triton kernel (#18226)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:confirmed-reverts
STAGE1: revert; artifacts=L3.triton.flash_attention_rocm,L3.rocm.rocm_flash_attn_v0; Reverts ROCm Triton FA kernel to previous implementation after perf/FP8 regressions.
ARTIFACT_HINTS: L3.triton.flash_attention_rocm, L3.rocm.rocm_flash_attn_v0, L3.platform.rocm_selection
FILES: vllm/attention/backends/rocm_flash_attn.py (+3/-2); vllm/attention/ops/triton_flash_attention.py (+685/-1081); vllm/platforms/rocm.py (+6/-0)
LABELS: rocm, ready
DEEP_STUDY: deep-study revert record: explicit_rollback of PR(s) 12591 reason=performance_regression
BODY: Revert to the previous version of the triton attention kernel, modified to support FP8 computation. ⏎ The kernel brought in in https://github.com/vllm-project/vllm/pull/12591 turned out to have performance issues, and broken support for FP8 quantized models. ⏎ Until that is resolved we want to replace it from the performant version from the ROCm fork

### L3-77b6e74fe2  (L3, 2025-05-29, sha 77b6e74fe2b6, PR #18938)
TITLE: [ROCm] Remove unnecessary assertion of max_model_len in ROCM_AITER_MLA attention backend. (#18938)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L3.mla.rocm_aiter; Removes max_model_len assertion in ROCm AITER MLA backend.
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/attention/backends/rocm_aiter_mla.py (+0/-2); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+0/-3)
LABELS: v1
BODY: This PR removes the unnecessary constraint and assertion of a specific `max_model_len value` from the `ROCM_AITER_MLA` attention backend on both VLLM v1 and v0 engines. ⏎  ⏎ ### lm_eval results on DeepSeek-V2-Lite-Chat with default engine args on both VLLM v0 and v1 engines. ⏎  ⏎ `VLLM_USE_V1=1 ⏎ VLLM_ROCM_USE_AITER=1 ⏎ VLLM_ROCM_USE_AITER_MOE=0 ⏎ VLLM_ROCM_USE_AITER_RMSNORM=0 ⏎ VLLM_ROCM_USE_AITER_LINEAR=0 ⏎ lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=1,block_size=1 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto` ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6657|±  |0.0130| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.6543|±  |0.0131| ⏎  ⏎ `VLLM_USE_V1=0 ⏎ VLLM_ROCM_USE_AITER=1 ⏎ VLLM_ROCM_USE_AITER_MOE=0 ⏎ VLLM_ROCM_USE_AITER_RMSNORM=0 ⏎ VLLM_ROCM_USE_AITER_LINEAR=0 ⏎ lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=1,block_size=1 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto` ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.5974|±  |0.0135| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.5876|±  |0.0136|

### L3-bdf13965ab  (L3, 2025-06-03, sha bdf13965ab4a, PR #18212)
TITLE: [V1] Support cross-layer KV sharing (#18212)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.dispatch.abstract_interface,L3.flash_attn.v1_backend,L3.flashinfer.v1_backend,L3.mla.common_v1; Adds cross-layer KV sharing support across attention backend protocols.
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.rocm_flash_attn_v0, L3.mla.triton_v0, L3.mla.common_v1, L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.mla.rocm_aiter, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: vllm/attention/backends/abstract.py (+1/-0); vllm/attention/backends/blocksparse_attn.py (+3/-0); vllm/attention/backends/cpu_mla.py (+2/-1); vllm/attention/backends/dual_chunk_flash_attn.py (+3/-0); vllm/attention/backends/flash_attn.py (+3/-0); vllm/attention/backends/flashinfer.py (+3/-0); vllm/attention/backends/flashmla.py (+2/-1); vllm/attention/backends/hpu_attn.py (+3/-0); vllm/attention/backends/ipex_attn.py (+3/-0); vllm/attention/backends/mla/common.py (+3/-0); vllm/attention/backends/pallas.py (+3/-0); vllm/attention/backends/rocm_aiter_mla.py (+2/-1); vllm/attention/backends/rocm_flash_attn.py (+3/-0); vllm/attention/backends/torch_sdpa.py (+3/-0); vllm/attention/backends/triton_mla.py (+2/-1); vllm/attention/backends/xformers.py (+3/-0); vllm/attention/layer.py (+16/-1); vllm/v1/attention/backends/flash_attn.py (+21/-15); vllm/v1/attention/backends/flashinfer.py (+21/-15); vllm/v1/attention/backends/mla/common.py (+4/-0); vllm/v1/attention/backends/mla/flashmla.py (+2/-1); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+2/-1); vllm/v1/attention/backends/mla/triton_mla.py (+2/-1); vllm/v1/attention/backends/pallas.py (+5/-1); vllm/v1/attention/backends/triton_attn.py (+28/-23); vllm/v1/attention/backends/utils.py (+33/-0); tests/v1/tpu/worker/test_tpu_model_runner.py (+226/-1); tests/v1/worker/test_gpu_model_runner.py (+237/-7); vllm/v1/worker/gpu_model_runner.py (+29/-2); vllm/v1/worker/tpu_model_runner.py (+29/-1); vllm/v1/worker/utils.py (+36/-0)
LABELS: tpu, ready, v1
BODY: ## Motivation ⏎  ⏎ Some models like Tencent-Hunyuan-Large (#10043) and Hymba-1.5B-Base (#10783) use cross-layer KV sharing (e.g. [Cross-Layer Attention](https://arxiv.org/abs/2405.12981)). This PR adds the ability for KV caches to be shared between attention layers. ⏎  ⏎ ## Design ⏎ This PR adds a new argument `kv_sharing_target_layer_name: Optional[str] = None` to the `Attention` layer class. This is only supported in V1. To have an Attention layer not allocate its own KV cache and instead share the KV cache with another layer (referred to as the _target layer_), you can pass in the fully-qualified name of the `Attention` layer in the target layer e.g. `model.layers.0.attn`. The arg `kv_sharing_target_layer_name` is only valid if a) it refers to an `Attention` layer, b) it has the same attn type (e.g. decoder) as the current layer, and c) it comes before the current layer. It is referred to the as the _target layer_ because during attention, the current layer will use its queries and perform the attention op with the keys and values tensor from the KV cache of the target layer.  ⏎  ⏎ If an `Attention` layer has a valid `kv_sharing_target_layer_name` defined, then we skip creating a KVCacheSpec for it, while recording the mapping in `self.shared_kv_cache_layers`: ⏎  ⏎ https://github.com/vllm-project/vllm/blob/89450fc323e9eee05cbba76fb5b9a0d29f7038d8/vllm/v1/worker/gpu_model_runner.py#L2142-L2152 ⏎  ⏎ During KV cache initialization, KV cache management logic will continue as if this layer did not exist and will not allocate a KV cache for the layer. The KV cache for these layers will instead be a reference to the allocated KV caches of the matching target layers, which enables the memory savings of cross-layers KV sharing. ⏎  ⏎ https://github.com/vllm-project/vllm/blob/89450fc323e9eee05cbba76fb5b9a0d29f7038d8/vllm/v1/worker/utils.py#L107-L110 ⏎  ⏎ We also add these layers to the list of layer names kept by each KV cache group, as this ensures that each layer is assigned its own attention metadata. From the perspective of the `Attention` layer, it does not know where the key and value caches are coming from. ⏎  ⏎ The memory savings of cross-layer KV sharing allows a given amount of memory to accommodate longer context lengths or enable more request to be processed in parallel.  ⏎  ⏎ --- ⏎  ⏎ ## Testing  ⏎  ⏎ ### Sanity Check ⏎ As a sanity check that the implementation is working, I made all layers after the 18th layer in Qwen/Qwen3-8B (36 layers total) and printed out the id() of the kv cache used in attention forward: ⏎  ⏎ ``` ⏎ model.layers.0.self_attn.attn => 139678446053136 ⏎ model.layers.1.self_attn.attn = …[truncated]

### L3-41aa578428  (L3, 2025-06-03, sha 41aa5784287f, PR #17625)
TITLE: [NVIDIA] Add Cutlass MLA backend (#17625)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:production-kernel-provenance
STAGE1: introduce; artifacts=L3.mla.cutlass_v1_backend,L3.mla.cutlass_kernels; Introduces V1 CUTLASS MLA backend selecting cutlass_mla_decode on Blackwell.
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.cutlass_kernels, L3.mla.cutlass_v1_backend, L3.platform.cuda_selection
FILES: csrc/attention/mla/cutlass_mla_kernels.cu (+1/-1); vllm/engine/arg_utils.py (+1/-0); vllm/platforms/cuda.py (+8/-0); vllm/platforms/interface.py (+1/-0); vllm/v1/attention/backends/mla/common.py (+1/-1); vllm/v1/attention/backends/mla/cutlass_mla.py (+96/-0); tests/kernels/test_cutlass_mla_decode.py (+3/-1)
LABELS: ready, v1
PERF_LINES: It also includes performance results using DeepSeek-V3 on 8×B200 GPUs under DP+EP parallelism settings, which delivers ~17% improved throughput. | Request throughput (req/s):              2.86 | Output token throughput (tok/s):         2857.52 | Total Token throughput (tok/s):          5715.04 | Mean TTFT (ms):                          200716.51 | Median TTFT (ms):                        199463.35
BODY: This PR introduces the `CUTLASS_MLA_VLLM_V1` backend, enabling support for `ops.cutlass_mla_decode()` on NVIDIA Blackwell GPUs. ⏎  ⏎ It also includes performance results using DeepSeek-V3 on 8×B200 GPUs under DP+EP parallelism settings, which delivers ~17% improved throughput. ⏎  ⏎ ``` ⏎ # With default triton backend: ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     2989 ⏎ Benchmark duration (s):                  1046.01 ⏎ Total input tokens:                      2989000 ⏎ Total generated tokens:                  2989000 ⏎ Request throughput (req/s):              2.86 ⏎ Output token throughput (tok/s):         2857.52 ⏎ Total Token throughput (tok/s):          5715.04 ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):                          200716.51 ⏎ Median TTFT (ms):                        199463.35 ⏎ P99 TTFT (ms):                           395239.25 ⏎ -----Time per Output Token (excl. 1st token)------ ⏎ Mean TPOT (ms):                          826.04 ⏎ Median TPOT (ms):                        826.20 ⏎ P99 TPOT (ms):                           1001.39 ⏎ ---------------Inter-token Latency---------------- ⏎ Mean ITL (ms):                           826.04 ⏎ Median ITL (ms):                         648.89 ⏎ P99 ITL (ms):                            8337.69 ⏎ ================================================== ⏎  ⏎ With cutlass_mla backend: ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     2989 ⏎ Benchmark duration (s):                  881.52 ⏎ Total input tokens:                      2989000 ⏎ Total generated tokens:                  2989000 ⏎ Request throughput (req/s):              3.39 ⏎ Output token throughput (tok/s):         3390.73 ⏎ Total Token throughput (tok/s):          6781.46 ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):                          190244.11 ⏎ Median TTFT (ms):                        189563.96 ⏎ P99 TTFT (ms):                           372713.07 ⏎ -----Time per Output Token (excl. 1st token)------ ⏎ Mean TPOT (ms):                          685.60 ⏎ Median TPOT (ms):                        686.96 ⏎ P99 TPOT (ms):                           858.01 ⏎ ---------------Inter-token Latency---------------- ⏎ Mean ITL (ms):                           685.60 ⏎ Median ITL (ms):                         518.56 ⏎ P99 ITL (ms):                            7738.23 ⏎ ================================================== ⏎ ``` ⏎  ⏎ To repro the results: ⏎ ```bash ⏎ # Server side with triton backend (Plz use VLLM_ATTENTION_BACKEND=CUTLASS_MLA_VLLM_V1 for cutlass backend): ⏎ VLLM_LOGGING_LEVEL=DEBUG \ ⏎ VLLM_WORKER_MULTIPROC_MET …[truncated]

### L3-b124e1085b  (L3, 2025-06-03, sha b124e1085b1b, PR #19106)
TITLE: [Bugfix] Fix FA3 full cuda graph correctness (#19106)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases
STAGE1: repair_correctness; artifacts=L3.flash_attn.v1_backend; Fixes FA3 full CUDA graph correctness in V1 FlashAttention.
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+21/-8); vllm/v1/worker/gpu_model_runner.py (+5/-0); .buildkite/test-pipeline.yaml (+1/-0); tests/compile/piecewise/test_full_cudagraph.py (+5/-2)
LABELS: ready, ci/build, v1
DEEP_STUDY: deep-study correctness case vllm:b124e1085b: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: Fixes the correctness issue when using the full cuda graph, and adds `test_full_cudagraph.py` into the CI.

### L3-b2fac67130  (L3, 2025-06-04, sha b2fac67130b1, PR #18833)
TITLE: [P/D] Heterogeneous TP (#18833)
SOURCES: path_core, release_notes
STAGE1: adapt_framework; artifacts=L3.flash_attn.v1_backend; Adds FlashAttention metadata support for heterogeneous TP/P-D KV transfer.
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+16/-0); tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh (+8/-3); tests/v1/kv_connector/nixl_integration/test_accuracy.py (+1/-0); vllm/distributed/kv_transfer/kv_connector/utils.py (+17/-1); vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py (+243/-95); vllm/worker/worker_base.py (+2/-1)
LABELS: ready, v1, kv-connector
BODY: Yet another take at https://github.com/vllm-project/vllm/pull/18079. It builds on the same commits so the rationale of the PR is the same. ⏎  ⏎ The issue with the previous approach is that it appears using a higher number of descriptors per read -`block_size` as many, each region smaller by  `block_size` times, so total number of bytes moved is unchanged- causes significant slowdowns. Mind that this is not happening for homogenous TP, where memory regions are seemingly merged by nixl prior to transfer.  ⏎  ⏎ Changing NIXL+UCX versions has a noticeable effect on performance, so we could in principle tackle the above directly at the transport layer. ⏎ Instead, here we take a different approach to factor out the transport layer altogether, and instantiate the kv cache with a memory layout  `[2, num_blocks, kv_heads, block_size, head_dim]` . We then permute back to the original NHD to provide a view that guarantees correctness in the rest of the codebase.  ⏎  ⏎ This enables the splitting to be carried out on dim2, leading to much better performance as we maintain the same number of descriptors as well as bytes per-read (minus a factor of tp_ratio). The code is also somewhat easier to read with one less nested dim to account for. ⏎  ⏎ This PR requires this https://github.com/vllm-project/vllm/pull/18775 to be merged first, as we need the proper scaffolding code for enforcing a different KV cache layout.  ⏎  ⏎ Here's some numbers: ⏎ ![image](https://github.com/user-attachments/assets/c5c517c0-7316-43ec-9121-8edd954919c1) ⏎  ⏎ This has been tested on NIXL 0.2.1 (`4f37f07`) and UCX 1.18.0. ⏎  ⏎ --- ⏎ In the MLA case, most of the splitting complexity above is not needed as kv caches are replicated and they can just be copied over in their entirety just like homogenous TP.  ⏎ Codewise the changes are minimal as we're using the same logic for discovery as well as rank assignment so we can conveniently support both.

### L3-87360308b7  (L3, 2025-06-05, sha 87360308b7ee, PR #19118)
TITLE: [V1] Use FlashInfer by default on Blackwell GPUs (#19118)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: change_default; artifacts=L3.platform.cuda_selection,L3.flashinfer.v1_backend; Makes FlashInfer the default V1 attention backend on Blackwell when installed.
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+15/-0); vllm/platforms/interface.py (+24/-0)
LABELS: performance, ready, v1
PERF_LINES: uv pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
BODY: ## Purpose ⏎  ⏎ FlashInfer has a specific backend for NVIDIA Blackwell so it is much more performant than FlashAttention2 default in V1 (FA3 is unsupported). If a user has it installed, I think we should choose it by default, which this PR achieves. See this comment for benchmarks https://github.com/vllm-project/vllm/pull/18095#issuecomment-2877849390 ⏎  ⏎ This PR also adds `is_device_capability` to the platform interface for exact cc checking since we only want this for SM 10.0 ⏎  ⏎ ## Test Plan ⏎  ⏎ Test locally on a B200 that FlashInfer is enabled by default ⏎  ⏎ ## Test Result ⏎  ⏎ On B200 without flashinfer installed: ⏎ ``` ⏎ vllm serve facebook/opt-125m --enforce-eager ⏎ ... ⏎ INFO 06-04 13:16:58 [cuda.py:234] FlashInfer failed to import for V1 engine on Blackwell (SM 10.0) GPUs; it is recommended to install FlashInfer for better performance. ⏎ INFO 06-04 13:16:58 [cuda.py:240] Using Flash Attention backend on V1 engine. ⏎ ``` ⏎  ⏎ On B200 with flashinfer installed: ⏎ ``` ⏎ uv pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl ⏎ vllm serve facebook/opt-125m --enforce-eager ⏎ ... ⏎ INFO 06-04 13:15:34 [cuda.py:228] Using FlashInfer backend on V1 engine by default for Blackwell (SM 10.0) GPUs. ⏎ ```

### L3-18093084be  (L3, 2025-06-05, sha 18093084be93, PR #19138)
TITLE: [Misc] Remove unnecessary fallback to prefill-decode attention (#19138)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: change_default; artifacts=L3.triton.v1_backend; Removes unnecessary Triton backend fallback to prefill-decode attention.
ARTIFACT_HINTS: L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+1/-4)
LABELS: ready, v1
BODY: ## Purpose ⏎ This PR removes an unnecessary fallback to prefill-decode attention in `vllm/v1/attention/backends/triton_attn.py`. The code currently falls back to prefill-decode attention if the number of queries per key and value is of power 2. This was introduced in #18093 to overcome an issue with unified attention. However,  after the fix in #18100, this condition is not needed anymore. ⏎ ## Test Plan ⏎ Run lm_eval with and without prefill-decode attention using the following commands: ⏎ 1. With prefill-decode attention ⏎ ```bash ⏎ VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 \ ⏎ VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1\ ⏎ vllm serve Qwen/Qwen2-7B \ ⏎ --port 19999 \ ⏎ --host 0.0.0.0 \ ⏎ --gpu-memory-utilization 0.90 \ ⏎ --max-model-len 32768 \ ⏎ ``` ⏎ 2. Without prefill-decode attention ⏎ ```bash ⏎ VLLM_V1_USE_PREFILL_DECODE_ATTENTION=0 \ ⏎ VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1\ ⏎ vllm serve Qwen/Qwen2-7B \ ⏎ --port 19999 \ ⏎ --host 0.0.0.0 \ ⏎ --gpu-memory-utilization 0.90 \ ⏎ --max-model-len 32768 \ ⏎ ``` ⏎  ⏎ lm_eval ⏎  ⏎ ```bash ⏎ lm_eval   \ ⏎ --model local-completions \ ⏎ --model_args model=Qwen/Qwen2-7B,base_url=http://localhost:19999/v1/completions \ ⏎ --batch_size auto \ ⏎ --tasks gsm8k \ ⏎ --num_fewshot 5 \ ⏎ --batch_size auto ⏎ ``` ⏎  ⏎ ## Test Result ⏎ Lm_eval with prefill-decode attention: ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7832|±  |0.0114| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.7801|±  |0.0114| ⏎  ⏎ lm_eval without prefill-decode attention: ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7786|±  |0.0114| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.7771|±  |0.0115|

### L3-9ef9173cfa  (L3, 2025-06-05, sha 9ef9173cfa33, PR #19090)
TITLE: [P/D][NixlConnector] Enable FlashInfer backend (#19090)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L3.flashinfer.v1_backend; Enables FlashInfer attention backend in Nixl disaggregated-prefill setups.
ARTIFACT_HINTS: -
FILES: vllm/platforms/interface.py (+1/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py (+50/-15)
LABELS: ready
BODY: This PR enables the use of `VLLM_ATTENTION_BACKEND=FLASHINFER` in disaggregated prefill setups leveraging NixlConnector (which is currently allowed but broken on main). ⏎  ⏎ The main difference wrt default FA backend is that FlashInfer swaps the cache first two dims (`K/V` and `num_blocks`) resulting in `[num_blocks, KV(2), N,H,D]`.  The easiest approach here is to maintain the layout and just transfer the whole region (for each layer) instead of trying to split the K/V dim.  ⏎  ⏎ As a result, the message size should be twice as big when running FlashInfer, resulting in an interesting trade-off that we should monitor to ensure optimal transfer size. ⏎  ⏎ As a side note, this will also enable the `TRITON_MLA_VLLM_V1` backend when an MLA model is detected.  ⏎ Note that for MLA model the behavior should be unchanged as the kv shape is not backend-dependent.   ⏎  ⏎ Test with ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=FLASHINFER NUM_DECODE_INSTANCES=1 bash tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh  ⏎ ```

### L3-cf02f9b283  (L3, 2025-06-06, sha cf02f9b283a6, PR #16078)
TITLE: Add FlexAttention to V1 (#16078)
SOURCES: path_core, symbol_pickaxe
STAGE1: introduce; artifacts=L3.flex_attention,L3.platform.cuda_selection; Introduces V1 FlexAttention backend and CUDA platform registration.
ARTIFACT_HINTS: L3.platform.cuda_selection, L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+477/-0); tests/kernels/test_flex_attention.py (+93/-0); vllm/engine/arg_utils.py (+1/-0); vllm/platforms/cuda.py (+3/-0); vllm/platforms/interface.py (+1/-0)
LABELS: documentation, ready, ci/build, v1
BODY: # Summary ⏎ This PR adds FlexAttention as a new unified_attention backend for the V1 engine.  ⏎  ⏎ This requires torch > 2.7 since we fixed a number of dynamic shapes issues that show up by default here. ⏎  ⏎ ### Design ⏎  ⏎ FlexAttention is broken up into two distinct phases, block mask creation and the call to forward. For most Transformers they N attention layers share a common attention pattern and thus we can amortize the cost of block mask creation over the N attention layers.  This lends itself pretty nicley to the Metadata Builder for the UA OP.  ⏎  ⏎ Majority of the work here is to build the correct BlockMask. ⏎  ⏎ The current BlockTable is of the form:  ⏎ ![image](https://github.com/user-attachments/assets/30e0e432-9fc2-430d-8623-b4e98fa373c2) ⏎  ⏎ This block table is a map from logical KV pages to Physical KV pages in the paged KV cache. It has a size of  ⏎ MAX_REQS x (MAX_SEQ_LEN//PAGE_SIZE).  ⏎  ⏎ FlexAttention has no notion of a PageTable and we have to build out inverse mapping from Physical Pages (the full paged KV Cache is input to kernel) to Logical Indices. We then use these logical indices to determine if we should compute attention for a query x KV pair. ⏎ ``` Shell ⏎  ⏎     Logical to Physical (Original block_table): ⏎     ┌───────────────────────────────────────────┐ ⏎     │ Request 0:                                │ ⏎     │                                           │ ⏎     │ Logical Blocks:  0  1  2  3  4  5  6  7   │ ⏎     │                  │  │  │  │  │  │  │  │   │ ⏎     │                  v  v  v  v  v  v  v  v   │ ⏎     │ Physical Blocks: 3  5  1  7  4  2  0  6   │ ⏎     └───────────────────────────────────────────┘ ⏎  ⏎     This function creates the inverse mapping: ⏎  ⏎     Physical to Logical (Inverse mapping): ⏎     ┌───────────────────────────────────────────┐ ⏎     │ Request 0:                                │ ⏎     │                                           │ ⏎     │ Physical Blocks: 0  1  2  3  4  5  6  7   │ ⏎     │                  │  │  │  │  │  │  │  │   │ ⏎     │                  v  v  v  v  v  v  v  v   │ ⏎     │ Logical Blocks:  6  2  5  0  4  1  7  3   │ ⏎     └───────────────────────────────────────────┘ ⏎  ⏎ ``` ⏎  ⏎ - Uses more memory than the page table ⏎ - Required memory: MAX_REQS × NUM_PAGES ⏎ - For smaller models: ⏎   - Number of pages can be up to 178,375 ⏎   - Calculation: (total tokens) / default_page_size = 2,854,000/16 ⏎   - Typical max_seq_len of 2048 = 128 Pages ⏎  ⏎ ### Setting up a Generic Physical to Logical re-writer ⏎ Once we have this Physical to Logical Map we can abstract this away from different logical mask_mods. We do this w/ this function: https://github.com/vllm-project/vllm/pull/1 …[truncated]
