### L3-c413c41cda  (L3, 2023-02-18, sha c413c41cda0f, PR #)
TITLE: Add reshape_and_cache op
SOURCES: path_core
STAGE1: introduce; artifacts=L3.cache.cuda_reshape; Added reshape_and_cache CUDA op and binding for paged KV-cache writes.
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L3-0deacbce6e  (L3, 2023-03-01, sha 0deacbce6e96, PR #3)
TITLE: Implement `single_query_cached_kv_attention` kernel (#3)
SOURCES: path_core, path_integration+keyword, subject_keyword
STAGE1: introduce; artifacts=L3.paged.cuda.v1; Added single_query_cached_kv_attention CUDA PagedAttention kernel and wrapper.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.cache.cuda_reshape, L3.flash_attn.upstream_pip
FILES: csrc/attention.cpp (+19/-0); csrc/attention_kernels.cu (+400/-0); csrc/attention_utils.h (+204/-0); csrc/cache_kernels.cu (+5/-5); setup.py (+9/-2); cacheflow/master/block_manager.py (+3/-1); cacheflow/models/attention.py (+19/-35); cacheflow/worker/cache_engine.py (+12/-8); cacheflow/worker/worker.py (+1/-1); csrc/cuda_primitives.h (+1318/-0); tests/kernels/attention.py (+142/-0); tests/kernels/cache.py (+8/-8)
BODY: This PR adds the `single_query_cached_kv_attention` kernel. ⏎  ⏎ Supported data types: ⏎ * `half` ⏎ * `float` ⏎  ⏎ Tested models: ⏎ * OPT-125M ⏎ * OPT-350M ⏎ * OPT-1.3B ⏎ * OPT-2.7B ⏎ * OPT-6.7B ⏎ * OPT-13B ⏎  ⏎ Tested GPUs: ⏎ * A100

### L3-3e9f991d6a  (L3, 2023-03-01, sha 3e9f991d6acd, PR #4)
TITLE: Use FlashAttention for `multi_query_kv_attention` (#4)
SOURCES: subject_keyword
STAGE1: integrate; artifacts=NEW:early_flashattention_prompt_adapter; Prompt multi-query attention switched to upstream FlashAttention kernels in model attention code.
ARTIFACT_HINTS: -
FILES: README.md (+1/-0); cacheflow/models/attention.py (+37/-30); server.py (+5/-2); tests/kernels/attention.py (+65/-2)
BODY: This PR is to use [FlashAttention](https://github.com/HazyResearch/flash-attention) kernels for `multi_query_kv_attention`, which performs masked attention for the prompt inputs. ⏎  ⏎ ### Pros ⏎ - FlashAttention is fast and memory-efficient. ⏎ - FlashAttention supports 1D inputs and only invokes a single kernel to handle multiple sequences with variable lengths. ⏎  ⏎ ### Cons ⏎ - FlashAttention **does NOT support FP32**. ⏎ - FlashAttention does not support head_size > 128. (This is fine for all models except GPT-J). ⏎   - Ref: https://github.com/HazyResearch/flash-attention/issues/67 ⏎ - FlashAttention does not support attention bias (GPT-J, BLOOM, LLaMA). ⏎  ⏎ Besides, note that FlashAttention does not support cached KV, which is required for interactive generation. ⏎  ⏎ Tested models: ⏎ - OPT-125M ⏎ - OPT-350M ⏎ - OPT-1.3B ⏎ - OPT-2.7B ⏎ - OPT-6.7B ⏎ - OPT-13B ⏎  ⏎ Tested GPUs: ⏎ - A100

### L3-1a7eb7da61  (L3, 2023-03-10, sha 1a7eb7da6157, PR #7)
TITLE: Support beam search & parallel generation (#7)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.cache.cuda_reshape; Cache kernel changed to support beam search and parallel generation cache layout needs.
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+31/-1); cacheflow/block.py (+4/-0); cacheflow/master/frontend.py (+28/-5); cacheflow/master/scheduler.py (+55/-36); cacheflow/models/__init__.py (+3/-1); cacheflow/models/input_metadata.py (+11/-7); cacheflow/models/model_utils.py (+10/-0); cacheflow/models/opt.py (+2/-1); cacheflow/models/sample.py (+258/-17); cacheflow/sampling_params.py (+54/-19); cacheflow/sequence.py (+72/-6); cacheflow/worker/cache_engine.py (+27/-8); cacheflow/worker/controller.py (+7/-10); cacheflow/worker/worker.py (+76/-43); csrc/cache.cpp (+14/-2); server.py (+10/-7)
BODY: This PR adds support for beam search and parallel generation (i.e., `n` > 1). ⏎  ⏎ **NOTE**: The correctness is only checked for beam search, but not for random sampling methods. ⏎  ⏎ Tested models: ⏎ - OPT-125M ⏎ - OPT-350M ⏎ - OPT-1.3B ⏎ - OPT-2.7B ⏎ - OPT-6.7B ⏎ - OPT-13B ⏎  ⏎ Tested GPUs: ⏎ - A100

### L3-897cb2ae28  (L3, 2023-04-02, sha 897cb2ae28e9, PR #20)
TITLE: Optimize data movement (#20)
SOURCES: path_core
STAGE1: optimize; artifacts=L3.paged.cuda.v1,L3.cache.cuda_reshape; Attention and cache kernels were changed to accept non-contiguous tensors and remove redundant data movement.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.cache.cuda_reshape, L3.flash_attn.upstream_pip
FILES: csrc/attention_kernels.cu (+13/-9); csrc/cache_kernels.cu (+17/-8); cacheflow/models/activation.py (+20/-0); cacheflow/models/attention.py (+46/-46); cacheflow/models/input_metadata.py (+5/-0); cacheflow/models/llama.py (+7/-12); cacheflow/models/opt.py (+2/-5); cacheflow/worker/worker.py (+8/-0); csrc/activation.cpp (+12/-0); csrc/activation_kernels.cu (+46/-0); csrc/pos_encoding.cpp (+0/-2); csrc/pos_encoding_kernels.cu (+22/-27); setup.py (+7/-0); tests/kernels/activation.py (+30/-0); tests/kernels/attention.py (+31/-15); tests/kernels/cache.py (+5/-5); tests/kernels/pos_encoding.py (+4/-6)
BODY: Should be merged after #15 . ⏎  ⏎ The changes in this PR eliminate the need for redundant data movements such as `torch.cat`, `torch.stack`, and `torch.contiguous`, which were previously used to align input and output shapes. The PR modifies existing kernels and adds new kernels to accommodate non-contiguous tensors, making these data movement operators unnecessary.

### L3-21b3671bbc  (L3, 2023-04-04, sha 21b3671bbc50, PR #24)
TITLE: Basic attention kernel that supports cached KV + (multi-)prompts (#24)
SOURCES: path_core, subject_keyword
STAGE1: introduce; artifacts=L3.paged.cuda.v1; Added basic cached-KV multi-prompt attention CUDA kernel.
ARTIFACT_HINTS: L3.paged.cuda.v1
FILES: csrc/attention.cpp (+16/-0); csrc/attention_kernels.cu (+463/-0); tests/kernels/attention.py (+143/-0)
BODY: This PR implements a basic and not highly optimized kernel to support cached KV with multiple import prompts.

### L3-0f40557af6  (L3, 2023-04-07, sha 0f40557af614, PR #32)
TITLE: Implement block copy kernel to optimize beam search (#32)
SOURCES: path_core
STAGE1: introduce; artifacts=NEW:cache_block_copy; Introduced cache block-copy CUDA kernel to reduce beam-search cache copy launches.
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+82/-22); benchmark/benchmark_latency.py (+6/-3); cacheflow/models/sample.py (+3/-2); cacheflow/worker/cache_engine.py (+4/-20); csrc/cache.cpp (+2/-2); tests/kernels/cache.py (+58/-0)
BODY: This PR implements a block copy kernel. By using this kernel, we can reduce the number of kernel invocations from `2 * num_layers * num_copying_blocks` to 1.

### L3-c267b1a02c  (L3, 2023-04-08, sha c267b1a02c95, PR #27)
TITLE: Add query stride to multi_query_cached_kv_attention & Add kernel benchmark script (#27)
SOURCES: path_core, subject_keyword
STAGE1: extend_support; artifacts=L3.paged.cuda.v1; Paged attention kernel gained query stride support for non-contiguous query tensors.
ARTIFACT_HINTS: L3.paged.cuda.v1
FILES: csrc/attention_kernels.cu (+11/-5); benchmark/benchmark_attention.py (+165/-0); tests/kernels/attention.py (+5/-3)
BODY: This PR adds query stride to the `multi_query_cached_kv_attention` kernel so that it can support non-contiguous query tensors in our OPT and LLaMA models (after #20). ⏎  ⏎ This PR also adds a benchmark script comparing our `multi_query_cached_kv_attention` and the optimized Flash attention implementation.

### L3-b9926f7f66  (L3, 2023-04-09, sha b9926f7f663b, PR #35)
TITLE: Support block size 32 (#35)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.paged.cuda.v1; Paged attention added block-size-32 support through kernel/build instantiation.
ARTIFACT_HINTS: L3.paged.cuda.v1
FILES: csrc/attention_kernels.cu (+44/-0); cacheflow/master/block_manager.py (+2/-2); cacheflow/master/server.py (+1/-1); tests/kernels/attention.py (+2/-2)
BODY: This PR adds support for block size 32. It turns out that no modification to our attention kernel is required for this support.

### L3-e3cec88aa5  (L3, 2023-04-10, sha e3cec88aa5b7, PR #29)
TITLE: Memcpy kernel for flash attention (#29)
SOURCES: path_core, subject_keyword
STAGE1: introduce; artifacts=NEW:gather_cached_kv; Added gather_cached_kv memcpy kernel for FlashAttention over paged KV cache.
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+157/-0); benchmark/benchmark_cache.py (+81/-0); csrc/cache.cpp (+11/-0); tests/kernels/cache.py (+44/-0)
PERF_LINES: [Latency] gather_cached_kv: 0.008 ms | [Throughput] gather_cached_kv: 156.479 GB/s | [Latency] gather_cached_kv: 0.011 ms | [Throughput] gather_cached_kv: 216.171 GB/s | [Latency] gather_cached_kv: 0.032 ms | [Throughput] gather_cached_kv: 152.631 GB/s | [Latency] gather_cached_kv: 0.057 ms | [Throughput] gather_cached_kv: 172.325 GB/s | [Latency] gather_cached_kv: 0.104 ms | [Throughput] gather_c
BODY: Memcpy kernel for flash attention ⏎  ⏎ ``` ⏎ num_tokens: 64, num_heads: 40, head_size: 128, block_size: 8, num_blocks: 1024, dtype: torch.float16 ⏎ [Latency] gather_cached_kv: 0.008 ms ⏎ [Throughput] gather_cached_kv: 156.479 GB/s ⏎ num_tokens: 128, num_heads: 40, head_size: 128, block_size: 8, num_blocks: 1024, dtype: torch.float16 ⏎ [Latency] gather_cached_kv: 0.011 ms ⏎ [Throughput] gather_cached_kv: 216.171 GB/s ⏎ num_tokens: 256, num_heads: 40, head_size: 128, block_size: 8, num_blocks: 1024, dtype: torch.float16 ⏎ [Latency] gather_cached_kv: 0.032 ms ⏎ [Throughput] gather_cached_kv: 152.631 GB/s ⏎ num_tokens: 512, num_heads: 40, head_size: 128, block_size: 8, num_blocks: 1024, dtype: torch.float16 ⏎ [Latency] gather_cached_kv: 0.057 ms ⏎ [Throughput] gather_cached_kv: 172.325 GB/s ⏎ num_tokens: 1024, num_heads: 40, head_size: 128, block_size: 8, num_blocks: 1024, dtype: torch.float16 ⏎ [Latency] gather_cached_kv: 0.104 ms ⏎ [Throughput] gather_cached_kv: 187.537 GB/s ⏎ num_tokens: 2048, num_heads: 40, head_size: 128, block_size: 8, num_blocks: 1024, dtype: torch.float16 ⏎ [Latency] gather_cached_kv: 0.204 ms ⏎ [Throughput] gather_cached_kv: 191.603 GB/s ⏎ ``` ⏎  ⏎ The performance is pretty good (theoretical optimal throughput is 1.6TB/s for A100-40GB), considering the memory layout is not ideal. ⏎  ⏎ result for unoptimized kernel: ⏎  ⏎ ``` ⏎ num_tokens: 64, num_heads: 40, head_size: 128, block_size: 8, num_blocks: 1024, dtype: torch.float16 ⏎ [Latency] gather_cached_kv: 0.010 ms ⏎ [Throughput] gather_cached_kv: 125.891 GB/s ⏎ num_tokens: 128, num_heads: 40, head_size: 128, block_size: 8, num_blocks: 1024, dtype: torch.float16 ⏎ [Latency] gather_cached_kv: 0.015 ms ⏎ [Throughput] gather_cached_kv: 160.678 GB/s ⏎ num_tokens: 256, num_heads: 40, head_size: 128, block_size: 8, num_blocks: 1024, dtype: torch.float16 ⏎ [Latency] gather_cached_kv: 0.032 ms ⏎ [Throughput] gather_cached_kv: 150.732 GB/s ⏎ num_tokens: 512, num_heads: 40, head_size: 128, block_size: 8, num_blocks: 1024, dtype: torch.float16 ⏎ [Latency] gather_cached_kv: 0.060 ms ⏎ [Throughput] gather_cached_kv: 162.482 GB/s ⏎ num_tokens: 1024, num_heads: 40, head_size: 128, block_size: 8, num_blocks: 1024, dtype: torch.float16 ⏎ [Latency] gather_cached_kv: 0.108 ms ⏎ [Throughput] gather_cached_kv: 180.763 GB/s ⏎ num_tokens: 2048, num_heads: 40, head_size: 128, block_size: 8, num_blocks: 1024, dtype: torch.float16 ⏎ [Latency] gather_cached_kv: 0.206 ms ⏎ [Throughput] gather_cached_kv: 189.757 GB/s ⏎ ``` ⏎  ⏎ the optimized kernel works much better for smaller number of tokens (+20% speedup)

### L3-0f4b32199e  (L3, 2023-04-15, sha 0f4b32199ec6, PR #38)
TITLE: Support various block sizes & Change default block size to 16 (#38)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.paged.cuda.v1; Paged attention was generalized to various block sizes and default block size changed.
ARTIFACT_HINTS: L3.paged.cuda.v1
FILES: csrc/attention.cpp (+0/-16); csrc/attention_kernels.cu (+557/-579); benchmark/benchmark_text_completion.py (+1/-0); cacheflow/master/block_manager.py (+0/-3); cacheflow/master/scheduler.py (+2/-1); cacheflow/master/server.py (+2/-2); csrc/cuda_primitives.h (+40/-18)
BODY: 

### L3-436e523bf1  (L3, 2023-05-03, sha 436e523bf161, PR #53)
TITLE: Refactor attention kernels (#53)
SOURCES: path_core, path_integration+keyword, subject_keyword
STAGE1: optimize; artifacts=L3.paged.cuda.v1; Attention kernel refactor reduced computation overhead by using reduced precision for logits-times-V.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.flash_attn.upstream_pip
FILES: csrc/attention/attention_dtypes.cuh (+5/-0); csrc/attention/attention_generic.cuh (+47/-0); csrc/attention/attention_kernels.cu (+451/-0); csrc/attention/attention_utils.cuh (+38/-0); csrc/attention/dtype_float16.cuh (+426/-0); csrc/attention/dtype_float32.cuh (+250/-0); csrc/attention_kernels.cu (+0/-896); csrc/attention_utils.h (+0/-165); setup.py (+1/-1); csrc/cuda_primitives.h (+0/-1340); csrc/layernorm_kernels.cu (+1/-1); csrc/reduction_utils.cuh (+34/-0); csrc/reduction_utils.h (+0/-76); tests/kernels/attention.py (+0/-90)
BODY: This PR refactors attention kernels, making the helper functions more modular and pruning unused code. This PR will make it easier to add support for a new data type such as bfloat16. ⏎  ⏎ In addition, this PR reduces the computation overhead of the attention kernel, by using the reduced precision (i.e., fp16) for `logits * V` instead of the full precision. This is compatible with the [FasterTransformer's implementation](https://github.com/NVIDIA/FasterTransformer/blob/c6e8f60ec40da218804a60e6aa986903e7fa8594/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp#LL39C24-L39C24).

### L3-e070829ae8  (L3, 2023-05-03, sha e070829ae81d, PR #54)
TITLE: Support bfloat16 data type (#54)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.paged.cuda.v1,L3.cache.cuda_reshape; Added bfloat16 dtype support in attention and cache kernels.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.flash_attn.upstream_pip
FILES: csrc/attention/attention_dtypes.h (+4/-0); csrc/attention/attention_kernels.cu (+6/-2); csrc/attention/attention_utils.cuh (+1/-1); csrc/attention/dtype_bfloat16.cuh (+361/-0); csrc/attention/dtype_float32.cuh (+1/-1); csrc/cache_kernels.cu (+55/-43); cacheflow/master/server.py (+2/-2); cacheflow/models/utils.py (+1/-0); csrc/activation_kernels.cu (+3/-1); csrc/layernorm_kernels.cu (+3/-1); csrc/pos_encoding_kernels.cu (+3/-1); setup.py (+15/-1)
BODY: Should be merged after #53 ⏎  ⏎ This PR adds support for the bfloat16 data type, which is used for some LLMs including Dolly V2.

### L3-130d5fd8c7  (L3, 2023-05-04, sha 130d5fd8c790, PR #68)
TITLE: Fix a bug in attention kernel (#68)
SOURCES: path_core, subject_keyword
STAGE1: repair_correctness; artifacts=L3.paged.cuda.v1; Fixed attention-kernel bug introduced by reduced-precision computation refactor.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+1/-1)
ISSUES: #66 A critical bug in attention kernel after refactoring
BODY: Fixes #66  ⏎  ⏎ This PR fixes a bug in our attention kernel. The bug was introduced in #53 when changing the precision of computations in the attention kernel. Now the kernel unit tests are passed normally.

### L3-c9d5b6d4a8  (L3, 2023-05-05, sha c9d5b6d4a8b3, PR #70)
TITLE: Replace FlashAttention with xformers (#70)
SOURCES: subject_keyword
STAGE1: replace; artifacts=NEW:early_xformers_prompt_adapter; Prompt attention path replaced direct FlashAttention use with xFormers for broader feature compatibility.
ARTIFACT_HINTS: -
FILES: README.md (+1/-5); cacheflow/master/server.py (+1/-1); cacheflow/models/attention.py (+16/-38); cacheflow/models/input_metadata.py (+9/-12); cacheflow/models/llama.py (+2/-2); cacheflow/models/memory_analyzer.py (+6/-6); cacheflow/models/opt.py (+2/-2); cacheflow/worker/worker.py (+0/-8); tests/kernels/activation.py (+1/-1); tests/kernels/attention.py (+35/-44); tests/kernels/cache.py (+10/-9); tests/kernels/layernorm.py (+3/-2); tests/kernels/pos_encoding.py (+1/-1)
BODY: This PR replaces FlashAttention with [xformers](https://github.com/facebookresearch/xformers). ⏎  ⏎ Pros: ⏎ - Richer features & higher compatibility. xformers supports attention bias, FP32, head size 256, and old GPUs (such as V100) while FlashAttention does not. ⏎ - xformers provides pre-compiled python wheels, while FlashAttention compiles the entire CUDA code during installation. ⏎ - Future-proof, as the repository is maintained by many developers from Meta. ⏎  ⏎ Cons: ⏎ - xformers can be slower than FlashAttention for small inputs, because it incurs higher CPU overheads. ⏎ - xformers internally creates a new tensor for the attention output. In our case, this leads to an extra copy overhead, because we concatenate the outputs of the two attention ops.

### L3-d721168449  (L3, 2023-05-27, sha d72116844928, PR #130)
TITLE: Improve setup script & Add a guard for bfloat16 kernels (#130)
SOURCES: path_core
STAGE1: repair_build_dependency; artifacts=L3.paged.cuda.v1; Improved setup and bfloat16 kernel guards to avoid build failures for attention kernels.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.flash_attn.upstream_pip
FILES: csrc/attention/attention_dtypes.h (+0/-3); csrc/attention/attention_kernels.cu (+0/-2); csrc/attention/dtype_bfloat16.cuh (+44/-0); setup.py (+46/-11)
BODY: This PR fixes `setup.py` so that the compiled modules can include the binaries for multiple types of GPUs. And the PR improves how we avoid build errors on bfloat16 kernels.

### L3-e38074b1e6  (L3, 2023-06-07, sha e38074b1e6ad, PR #141)
TITLE: Support FP32 (#141)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.paged.cuda.v1; Paged attention kernel/build support changed to enable FP32.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.flash_attn.upstream_pip
FILES: cacheflow/model_executor/layers/attention.py (+3/-5); csrc/attention/attention_kernels.cu (+37/-32); cacheflow/config.py (+3/-4); cacheflow/entrypoints/llm.py (+5/-4); cacheflow/server/arg_utils.py (+4/-4); docs/source/getting_started/installation.rst (+3/-0); setup.py (+5/-0); tests/kernels/test_attention.py (+5/-5)
ISSUES: #72 Support FP32
PERF_LINES: ~~This PR removes the support for some head and block sizes to enhance the compilation speed. The removed sizes are not used for the models we currently support
BODY: Closes #72  ⏎  ⏎ ~~This PR removes the support for some head and block sizes to enhance the compilation speed. The removed sizes are not used for the models we currently support. In addition, removing them allows us to support FP32. On my machine, the compilation time got reduced **from 7.5 mins to 1.5 mins** even though FP32 is now added.~~ ⏎  ⏎ This PR removes the support for some head and block sizes to enable FP32. Besides, the PR changes the `default` dtype option to `auto`, to make its meaning clearer.

### L3-e41f06702c  (L3, 2023-07-03, sha e41f06702cb6, PR #331)
TITLE: Add support for BLOOM (#331)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.paged.cuda.v1; Attention kernel added ALiBi slope handling needed by BLOOM.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention.cpp (+3/-1); csrc/attention/attention_kernels.cu (+19/-6); vllm/model_executor/layers/attention.py (+128/-5); README.md (+1/-0); docs/source/models/supported_models.rst (+3/-0); tests/kernels/test_attention.py (+1/-0); vllm/model_executor/input_metadata.py (+4/-2); vllm/model_executor/model_loader.py (+2/-3); vllm/model_executor/models/__init__.py (+2/-0); vllm/model_executor/models/bloom.py (+316/-0); vllm/model_executor/models/gpt_neox.py (+0/-1)
ISSUES: #61 Support BLOOM
BODY: Closes #61  ⏎  ⏎ This PR adds the BLOOM model and modifies the paged attention kernel to support ALiBi bias.

### L3-404422f42e  (L3, 2023-07-03, sha 404422f42ed9, PR #334)
TITLE: [Model] Add support for MPT (#334)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.paged.cuda.v1; Paged attention added head-size 112 support for MPT.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+3/-0); vllm/model_executor/layers/attention.py (+1/-1); README.md (+1/-0); docs/source/models/supported_models.rst (+3/-0); vllm/config.py (+3/-2); vllm/model_executor/model_loader.py (+2/-1); vllm/model_executor/models/__init__.py (+2/-0); vllm/model_executor/models/mpt.py (+279/-0); vllm/transformers_utils/config.py (+15/-0); vllm/transformers_utils/configs/__init__.py (+5/-0); vllm/transformers_utils/configs/mpt.py (+74/-0)
ISSUES: #218 Support for MPT-7B and MPT-30B | #332 feature request: support mpt-30b
BODY: Closes #218 and #332 ⏎  ⏎ Should be merged after #61

### L3-c894836108  (L3, 2023-07-08, sha c89483610873, PR #226)
TITLE: [Model] Add support for GPT-J (#226)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.paged.cuda.v1; Paged attention enabled head-size 256 support for GPT-J.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+4/-4); vllm/model_executor/layers/attention.py (+1/-1); README.md (+1/-0); docs/source/models/supported_models.rst (+3/-0); tests/kernels/test_attention.py (+2/-2); vllm/model_executor/layers/sampler.py (+3/-0); vllm/model_executor/model_loader.py (+1/-0); vllm/model_executor/models/__init__.py (+2/-0); vllm/model_executor/models/gpt_j.py (+251/-0); vllm/model_executor/models/mpt.py (+1/-0)
BODY: reference to issue https://github.com/vllm-project/vllm/issues/198

### L3-96853af5a8  (L3, 2023-07-14, sha 96853af5a830, PR #452)
TITLE: Optimize MQA Kernel (#452)
SOURCES: path_core
STAGE1: optimize; artifacts=L3.paged.cuda.v1; MQA paged-attention kernel changed implementation for faster multi-query attention.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention.cpp (+1/-0); csrc/attention/attention_kernels.cu (+22/-9); vllm/model_executor/layers/attention.py (+32/-11); vllm/config.py (+7/-0); vllm/model_executor/models/gpt_bigcode.py (+22/-52)
ISSUES: #393 [StarCoder] TypeError: Got unsupported ScalarType BFloat16 | #462 Starcoder is 5-10x slower on vllm than HF's TGI when passing in a continuous batch of requests
BODY: This PR implements the MQA paged attention kernel and modifies the GPT Bigcode model to utilize the optimized MQA kernel. ⏎  ⏎ TODO: Check performance gain.

### L3-bda41c70dd  (L3, 2023-07-18, sha bda41c70ddb1, PR #496)
TITLE: hotfix attn alibi wo head mapping (#496)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=NEW:early_paged_attention_layer; Attention layer hotfix corrected ALiBi handling without changing kernel body.
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+1/-0); tests/kernels/test_attention.py (+2/-0)
BODY: 

### L3-79af7e96a0  (L3, 2023-08-04, sha 79af7e96a0e2, PR #420)
TITLE: [OPTIMIZATION] Optimizes the single_query_cached_kv_attention kernel (#420)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: optimize; artifacts=L3.paged.cuda.v1; single_query_cached_kv_attention now cooperatively loads query head, reducing repeated memory reads.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+7/-4)
PERF_LINES: Instead of having each thread group fetch the query head (which causes 64x memory to be read), we have all threads in the block share the task of loading the qu
BODY: Instead of having each thread group fetch the query head (which causes 64x memory to be read), we have all threads in the block share the task of loading the query head. On the benchmark of running 1000 sequences through LLaMA13B on an A100 (80GB), this improves the throughput by 1.10x.

### L3-2a4ec90854  (L3, 2023-08-23, sha 2a4ec90854ae, PR #834)
TITLE: Fix for breaking changes in xformers 0.0.21 (#834)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=NEW:early_xformers_adapter; Attention adapter was updated for xFormers 0.0.21 attention-bias API changes.
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+3/-2); requirements.txt (+1/-1)
ISSUES: #407 RuntimeError: attn_bias is not correctly aligned | #468 attn_bias not aligned & some questions regarding float16 | #795 "attn_bias is not correctly aligned" on A100 for MPT-30B | #832 Issue while loading MPT-7B-8K-instruct
BODY: Fixes #832  ⏎  ⏎ This PR fixes the breaking changes made in the latest release of xformers. Specifically, now the xformers attention bias should have the batch dimension, which is 1 in the current vLLM implementation.

### L3-75471386de  (L3, 2023-08-29, sha 75471386de62, PR #877)
TITLE: use flash-attn via xformers (#877)
SOURCES: path_core
STAGE1: change_default; artifacts=NEW:early_xformers_flashattn_selection; Attention path changed to use FlashAttention through xFormers.
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+0/-3); tests/kernels/test_attention.py (+0/-2)
ISSUES: #485 Flash Attention V2
BODY: fixes https://github.com/vllm-project/vllm/issues/485#issuecomment-1693821853

### L3-8ce9c50d40  (L3, 2023-09-02, sha 8ce9c50d4034, PR #933)
TITLE: Avoid compiling kernels for double data type (#933)
SOURCES: path_core
STAGE1: repair_build_dependency; artifacts=L3.cache.cuda_reshape; Cache kernel dispatch stopped instantiating unused double dtype to reduce build/binary overhead.
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+5/-9); csrc/activation_kernels.cu (+4/-6); csrc/dispatch_utils.h (+14/-0); csrc/layernorm_kernels.cu (+2/-3); csrc/pos_encoding_kernels.cu (+3/-3)
BODY: This PR fixes a dispatch logic for our custom CUDA kernels. Currently, vLLM uses `AT_DISPATCH_FLOATING_TYPES_AND2` which actually [includes the double data type](https://github.com/pytorch/pytorch/blob/e9ebda29d87ce0916ab08c06ab26fd3766a870e5/aten/src/ATen/Dispatch.h#L245) that is never used. This leads to unnecessary increase in compilation time and binary size. The PR solves this issue by limiting the data types to `float`, `half` and `bfloat16`.

### L3-bf87484efa  (L3, 2023-09-04, sha bf87484efac9, PR #936)
TITLE: [BugFix] Fix NaN errors in paged attention kernel (#936)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.paged.cuda.v1; Paged attention now zeros masked values to avoid NaNs from 0-times-NaN.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+12/-0); csrc/attention/dtype_bfloat16.cuh (+10/-0); csrc/attention/dtype_float16.cuh (+5/-5); csrc/attention/dtype_float32.cuh (+5/-0)
ISSUES: #641 RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
BODY: Fixes #641  ⏎ This PR fixes the paged attention kernel. Currently, the kernel computes `attn_weight * value` for all tokens in a value block, even if some of them are not included in the context. It is generally acceptable since the `attn_weight` for those tokens is 0, but this causes errors when the tokens contain NaNs (since 0 * NaN is NaN). The PR solves this by explicitly setting the values of those tokens as 0.

### L3-db09d4ad83  (L3, 2023-09-07, sha db09d4ad833b, PR #945)
TITLE: [FIX] Fix Alibi implementation in PagedAttention kernel (#945)
SOURCES: path_core, subject_keyword, body_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.paged.cuda.v1; Fixed ALiBi bias computation inside PagedAttention kernel.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+1/-1); tests/kernels/test_attention.py (+3/-2)
BODY: cc @Oliver-ss  ⏎  ⏎ Will verify and test the correctness.

### L3-a62de9ecfd  (L3, 2023-09-09, sha a62de9ecfdc6, PR #996)
TITLE: Fix wrong dtype in PagedAttentionWithALiBi bias (#996)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=NEW:early_paged_attention_layer; PagedAttentionWithALiBi layer fixed bias dtype passed around the attention path.
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+11/-4)
ISSUES: #995 PagedAttentionWithALiBi uses wrong dtype for bias
BODY: Fixes dtype not being set correctly as it was derived from the default dtype before (which will now always be float32). ⏎  ⏎ Closes https://github.com/vllm-project/vllm/issues/995 ⏎  ⏎ cc @WoosukKwon @chu-tianxiang

### L3-cf5cb1e33e  (L3, 2023-09-26, sha cf5cb1e33eed, PR #1154)
TITLE: Allocate more shared memory to attention kernel (#1154)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L3.paged.cuda.v1; PagedAttention kernel uses additional shared memory on capable GPUs to support longer contexts.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.flash_attn.upstream_pip
FILES: csrc/attention/attention_kernels.cu (+5/-0); setup.py (+11/-0); vllm/utils.py (+12/-1); vllm/worker/worker.py (+25/-1); csrc/cuda_utils.cpp (+13/-0); csrc/cuda_utils_kernels.cu (+14/-0); tests/kernels/test_attention.py (+7/-1)
ISSUES: #905 vLLM doesn't support context length exceeding about 13k
BODY: Makes use of additional shared memory present on compute capability >=7.0 cards to support longer context length in the attention kernel. ⏎  ⏎ See https://stackoverflow.com/questions/63757245/using-maximum-shared-memory-in-cuda for details. ⏎  ⏎ As pointed out by @WoosukKwon offline, ideally we would also store logits inside the kernel in float16 instead of float32 as the accuracy loss should be minimal. This will enable even longer context lengths. ⏎  ⏎ Note that the buffer of 512 * sizeof(float32) may be too conservative, but this is still going to result in more supported tokens than ~11k previously. The attention test has been ran on A10 and A100 successfully. ⏎  ⏎ With this PR, the supported context lengths with current kernel (float32 logits) will be: ⏎  ⏎ - CC 7.5 (Turing): 64KiB shared memory -> 16328 tokens ⏎ - CC 7.0 (Volta): 96KiB shared memory -> 24984 tokens ⏎ - CC 8.6 (Ampere A10): 100KiB shared memory -> 25128 tokens ⏎ - CC 8.0 (Ampere A100): 160KiB shared memory -> 39936 tokens ⏎ - CC 9.0 (Hopper): 227KiB shared memory -> 57600 tokens ⏎  ⏎ Closes https://github.com/vllm-project/vllm/issues/905

### L3-bb1ba58f06  (L3, 2023-09-28, sha bb1ba58f0647, PR #1196)
TITLE: [Mistral] Mistral-7B-v0.1 support (#1196)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.paged.python_wrapper; Mistral/sliding-window support changed attention/cache caller contract for supported models.
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+22/-5); requirements.txt (+1/-1); vllm/config.py (+2/-0); vllm/core/block_manager.py (+30/-11); vllm/core/scheduler.py (+1/-1); vllm/engine/arg_utils.py (+3/-3); vllm/engine/llm_engine.py (+2/-0); vllm/model_executor/input_metadata.py (+20/-1); vllm/model_executor/model_loader.py (+1/-0); vllm/model_executor/models/__init__.py (+2/-0); vllm/model_executor/models/mistral.py (+404/-0); vllm/transformers_utils/configs/mistral.py (+66/-0); vllm/worker/worker.py (+17/-3)
ISSUES: #1199 Support for Mistral 7B
BODY: 

### L3-ebe4d1db3a  (L3, 2023-10-01, sha ebe4d1db3a42, PR #1241)
TITLE: Fix boundary check in paged attention kernel (#1241)
SOURCES: path_core, subject_keyword
STAGE1: repair_correctness; artifacts=L3.paged.cuda.v1; Fixed boundary check in single_query_cached_kv_attention CUDA kernel.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+1/-1)
BODY: This PR fixes a vulnerable memory modification to gpu shared memory in CUDA kernel function `single_query_cached_kv_attention_kernel`.

### L3-928de46888  (L3, 2023-10-16, sha 928de46888b9, PR #1348)
TITLE: Implement PagedAttention V2 (#1348)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: introduce; artifacts=L3.paged.cuda.v2_splitkv; Introduced PagedAttention V2 with sequence-level parallelism and split-KV reductions.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention.cpp (+24/-4); csrc/attention/attention_kernels.cu (+413/-71); csrc/attention/dtype_bfloat16.cuh (+5/-0); vllm/model_executor/layers/attention.py (+71/-47); benchmarks/kernels/benchmark_paged_attention.py (+197/-0); tests/kernels/test_attention.py (+54/-17)
PERF_LINES: This PR implements the first part of the PagedAttention V2 kernel, which uses sequence-level parallelism for better work partitioning. Compared to V1, the V2 ke
BODY: This PR implements the first part of the PagedAttention V2 kernel, which uses sequence-level parallelism for better work partitioning. Compared to V1, the V2 kernel achieves huge speedup when the batch size is small (e.g., <= 8). We will further optimize the kernel henceforth.

### L3-c1376e0f82  (L3, 2023-10-16, sha c1376e0f825e, PR #1381)
TITLE: Change scheduler & input tensor shape (#1381)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.cache.cuda_reshape; Scheduler/model contract switched to 2D tensors and cache kernel signature was updated.
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+7/-2); vllm/model_executor/layers/attention.py (+71/-95); csrc/activation_kernels.cu (+14/-14); csrc/layernorm_kernels.cu (+6/-6); csrc/pos_encoding_kernels.cu (+11/-11); vllm/config.py (+3/-0); vllm/core/scheduler.py (+11/-4); vllm/engine/arg_utils.py (+7/-1); vllm/model_executor/input_metadata.py (+5/-6); vllm/model_executor/layers/activation.py (+8/-12); vllm/model_executor/layers/quantized_linear/awq.py (+2/-2); vllm/model_executor/layers/sampler.py (+2/-1); vllm/worker/worker.py (+34/-25)
BODY: This PR updates the scheduler and model code to use 2D tensors instead of 1D tensors. The change will enable using a wider range of libraries and hardware, and facilitate future optimizations like CUDA graph.

### L3-0ce8647dc5  (L3, 2023-10-31, sha 0ce8647dc5bc, PR #1514)
TITLE: Fix integer overflows in attention & cache ops (#1514)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.paged.cuda.v1,L3.cache.cuda_reshape; Fixed integer overflows in paged attention and cache ops for large block counts.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape
FILES: csrc/attention/attention_kernels.cu (+8/-2); csrc/cache_kernels.cu (+36/-36); tests/kernels/test_attention.py (+1/-1); tests/kernels/test_cache.py (+7/-7); vllm/worker/worker.py (+1/-1)
ISSUES: #1486 Array Index Overflow for Large #Blocks
BODY: Fixes #1486  ⏎  ⏎ This PR fixes the overflows in paged attention & cache ops when the number of blocks is huge.

### L3-9738b84a08  (L3, 2023-11-01, sha 9738b84a0895, PR #1510)
TITLE: Force paged attention v2 for long contexts (#1510)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: change_default; artifacts=L3.paged.cuda.v2_splitkv; Long contexts were routed to PagedAttention V2 when V1 shared-memory limits would fail.
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+3/-1); vllm/worker/worker.py (+1/-28)
BODY: Removes the hard limit on context length that was tied to paged attention v1 limitations and instead forces v2 to be used if the context cannot fit in shared memory.

### L3-e0c6f556e8  (L3, 2023-11-23, sha e0c6f556e850, PR #1624)
TITLE: [Build] Avoid building too many extensions (#1624)
SOURCES: path_core
STAGE1: repair_build_dependency; artifacts=L3.paged.cuda.v1,L3.cache.cuda_reshape; Custom ops build was consolidated to one extension, changing delivery of attention/cache kernels.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: csrc/attention.cpp (+0/-42); vllm/model_executor/layers/attention.py (+4/-4); benchmarks/kernels/benchmark_paged_attention.py (+3/-3); csrc/activation.cpp (+0/-28); csrc/cache.h (+0/-19); csrc/cuda_utils.cpp (+0/-13); csrc/cuda_utils.h (+5/-0); csrc/layernorm.cpp (+0/-24); csrc/ops.h (+75/-0); csrc/pos_encoding.cpp (+0/-16); csrc/pybind.cpp (+80/-0); csrc/quantization.cpp (+0/-19); setup.py (+10/-72); tests/kernels/test_activation.py (+4/-4); tests/kernels/test_attention.py (+3/-3); tests/kernels/test_cache.py (+1/-1); tests/kernels/test_layernorm.py (+2/-2); tests/kernels/test_pos_encoding.py (+2/-2); vllm/model_executor/layers/activation.py (+4/-4); vllm/model_executor/layers/layernorm.py (+3/-3); vllm/model_executor/layers/quantization/awq.py (+2/-3); vllm/model_executor/layers/quantization/squeezellm.py (+2/-3); vllm/model_executor/layers/rotary_embedding.py (+4/-5); vllm/utils.py (+1/-1); vllm/worker/cache_engine.py (+1/-1)
PERF_LINES: This PR lets vllm only build one extension and reduces the build time by 75% (from ~5.5min to ~1.5min) based on my test.
BODY: This PR lets vllm only build one extension and reduces the build time by 75% (from ~5.5min to ~1.5min) based on my test.

### L3-27feead2f8  (L3, 2023-11-29, sha 27feead2f80f, PR #1843)
TITLE: Refactor Worker & InputMetadata (#1843)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.paged.python_wrapper; Worker/InputMetadata refactor changed metadata consumed by attention execution.
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+3/-11); vllm/config.py (+6/-0); vllm/engine/arg_utils.py (+4/-3); vllm/engine/llm_engine.py (+0/-2); vllm/model_executor/__init__.py (+2/-0); vllm/model_executor/input_metadata.py (+16/-65); vllm/model_executor/layers/sampler.py (+60/-55); vllm/model_executor/models/aquila.py (+10/-2); vllm/model_executor/models/baichuan.py (+10/-2); vllm/model_executor/models/bloom.py (+10/-2); vllm/model_executor/models/chatglm.py (+10/-2); vllm/model_executor/models/falcon.py (+10/-3); vllm/model_executor/models/gpt2.py (+10/-2); vllm/model_executor/models/gpt_bigcode.py (+10/-2); vllm/model_executor/models/gpt_j.py (+10/-2); vllm/model_executor/models/gpt_neox.py (+10/-2); vllm/model_executor/models/internlm.py (+10/-2); vllm/model_executor/models/llama.py (+10/-2); vllm/model_executor/models/mistral.py (+10/-2); vllm/model_executor/models/mpt.py (+10/-2); vllm/model_executor/models/opt.py (+10/-2); vllm/model_executor/models/phi_1_5.py (+27/-26); vllm/model_executor/models/qwen.py (+10/-2); vllm/model_executor/models/yi.py (+10/-2); vllm/model_executor/sampling_metadata.py (+43/-0); vllm/worker/model_runner.py (+334/-0); vllm/worker/worker.py (+13/-248)
BODY: Should be merged after #1840  ⏎  ⏎ This PR refactors worker & input metadata, making the following changes: ⏎  ⏎ 1. `InputMetadata` is split into `InputMetadata` and `SamplingMetadata`. ⏎ 2. Each forward pass consists of two stages: `hidden_states = model.forward(...)` and `next_tokens = model.sample(hidden_states, sampling_metadata)`. This will be useful because we will only handle the former in using CUDA graph or `torch.compile`. ⏎ 3. `_prepare_inputs` is split into `_prepare_prompts`, `_prepare_decode`, and `_prepare_sample`. ⏎ 4. Minor optimizations in caching keys and values with sliding window

### L3-6ccc0bfffb  (L3, 2023-12-07, sha 6ccc0bfffbcf, PR #1836)
TITLE: Merge EmbeddedLLM/vllm-rocm into vLLM main (#1836)
SOURCES: path_core, symbol_pickaxe, dependency_pin
STAGE1: port; artifacts=L3.paged.cuda.v1,L3.cache.cuda_reshape; Merged ROCm support porting existing vLLM kernels, including attention/cache paths, to AMD.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.flash_attn.upstream_pip
FILES: Dockerfile.rocm (+62/-0); csrc/attention/attention_kernels.cu (+21/-13); csrc/attention/attention_utils.cuh (+2/-1); csrc/attention/dtype_bfloat16.cuh (+16/-3); csrc/attention/dtype_float16.cuh (+63/-5); csrc/cache_kernels.cu (+7/-6); requirements-rocm.txt (+17/-0); setup.py (+159/-73); vllm/model_executor/layers/attention.py (+3/-0); .gitignore (+4/-0); README.md (+2/-0); csrc/activation_kernels.cu (+4/-3); csrc/cuda_compat.h (+28/-0); csrc/cuda_utils_kernels.cu (+3/-0); csrc/ops.h (+2/-0); csrc/pos_encoding_kernels.cu (+5/-4); csrc/pybind.cpp (+4/-0); csrc/quantization/squeezellm/quant_cuda_kernel.cu (+74/-0); csrc/reduction_utils.cuh (+3/-1); docs/source/getting_started/amd-installation.rst (+143/-0); docs/source/index.rst (+2/-0); patch_xformers-0.0.22.post7.rocm.sh (+22/-0); rocm_patch/commonpy_xformers-0.0.22.post7.rocm.patch (+13/-0); rocm_patch/flashpy_xformers-0.0.22.post7.rocm.patch (+134/-0); vllm/config.py (+35/-4); vllm/engine/ray_utils.py (+7/-1); vllm/model_executor/layers/quantization/squeezellm.py (+9/-3); vllm/model_executor/model_loader.py (+24/-0); vllm/utils.py (+5/-1)
LABELS: rocm
BODY: **Add ROCm- Support** ⏎  ⏎  ⏎ As there are too many changes has been made after https://github.com/vllm-project/vllm/pull/1749 , ⏎ the previous PR https://github.com/vllm-project/vllm/pull/1749 is closed as and continued here. ⏎  ⏎  ⏎ PR Authors: ⏎ @kliuae ⏎ @iAmir97 ⏎ @tjtanaa  ⏎ @tanpinsiang ⏎  ⏎ Contributer: ⏎ @pcmortiz ⏎  ⏎ This pull request also incorporates the work from [Port most vLLM kernels to ROCm #1313](https://github.com/vllm-project/vllm/pull/1313 ) by @pcmoritz, which was not merged. We appreciate @pcmoritz's contribution.

### L3-dacaf5a400  (L3, 2023-12-10, sha dacaf5a40056, PR #1997)
TITLE: Replace head_mapping params with num_kv_heads to attention kernel. (#1997)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: optimize; artifacts=L3.paged.cuda.v1; Paged attention replaced head_mapping parameter with num_kv_heads to avoid global-memory head mapping loads.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+15/-16); csrc/ops.h (+2/-2); vllm/model_executor/layers/attention.py (+5/-8); benchmarks/kernels/benchmark_paged_attention.py (+2/-6); tests/kernels/test_attention.py (+2/-5)
ISSUES: #1928 Why we need head_mapping as param pass to paged_attention kernel?
BODY: Replace head_mapping params with num_kv_heads to attention kernel. ⏎ Base on this issue: ⏎ [https://github.com/vllm-project/vllm/issues/1928](url) ⏎ To avoid the head_mapping load from global memory.

### L3-37ca558103  (L3, 2023-12-16, sha 37ca55810392, PR #1926)
TITLE: Optimize model execution with CUDA graph (#1926)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.paged.python_wrapper; CUDA graph execution changed attention-layer interaction with captured model execution.
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+18/-22); benchmarks/benchmark_latency.py (+4/-0); benchmarks/benchmark_throughput.py (+7/-2); requirements.txt (+1/-0); vllm/config.py (+17/-0); vllm/engine/arg_utils.py (+15/-1); vllm/engine/llm_engine.py (+6/-1); vllm/engine/ray_utils.py (+1/-8); vllm/entrypoints/llm.py (+10/-0); vllm/model_executor/input_metadata.py (+4/-1); vllm/model_executor/models/aquila.py (+2/-10); vllm/model_executor/models/baichuan.py (+2/-10); vllm/model_executor/models/bloom.py (+2/-10); vllm/model_executor/models/chatglm.py (+2/-13); vllm/model_executor/models/falcon.py (+2/-12); vllm/model_executor/models/gpt2.py (+3/-10); vllm/model_executor/models/gpt_bigcode.py (+3/-10); vllm/model_executor/models/gpt_j.py (+2/-10); vllm/model_executor/models/gpt_neox.py (+2/-10); vllm/model_executor/models/internlm.py (+2/-10); vllm/model_executor/models/llama.py (+2/-10); vllm/model_executor/models/mistral.py (+2/-10); vllm/model_executor/models/mixtral.py (+3/-10); vllm/model_executor/models/mpt.py (+2/-10); vllm/model_executor/models/opt.py (+5/-14); vllm/model_executor/models/phi_1_5.py (+2/-10); vllm/model_executor/models/qwen.py (+2/-10); vllm/model_executor/models/yi.py (+2/-10); vllm/model_executor/parallel_utils/communication_op.py (+8/-2); vllm/model_executor/parallel_utils/cupy_utils.py (+115/-0); vllm/model_executor/parallel_utils/parallel_state.py (+37/-0); vllm/utils.py (+7/-0); vllm/worker/model_runner.py (+225/-16); vllm/worker/worker.py (+40/-12)
BODY: This PR uses CUDA graph to optimize the CPU overheads in model execution. This is particularly effective for small models and when using tensor parallelism. ⏎  ⏎ Many thanks to @scv119 and @Yard1 for addressing the NCCL-related issues.

### L3-77af974b40  (L3, 2024-01-02, sha 77af974b406f, PR #1959)
TITLE: [FIX] Support non-zero CUDA devices in custom kernels (#1959)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.paged.cuda.v1,L3.cache.cuda_reshape; Added device guards so attention/cache custom kernels work on non-zero CUDA devices.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape
FILES: csrc/attention/attention_kernels.cu (+3/-0); csrc/cache_kernels.cu (+5/-0); csrc/activation_kernels.cu (+4/-1); csrc/layernorm_kernels.cu (+3/-0); csrc/pos_encoding_kernels.cu (+2/-0); csrc/quantization/squeezellm/quant_cuda_kernel.cu (+2/-1); tests/kernels/conftest.py (+3/-2); tests/kernels/test_activation.py (+13/-3); tests/kernels/test_attention.py (+15/-10); tests/kernels/test_cache.py (+11/-6); tests/kernels/test_layernorm.py (+6/-3); tests/kernels/test_pos_encoding.py (+7/-4)
BODY: While assessing the effectiveness of the RMSNorm operator, I observed that executing this operator on non-zero GPU resulted in a 'RuntimeError: CUDA error: an illegal memory access was encountered.' ⏎  Upon further investigation through debugging, I  found that cause as the absence of [device guards](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/awq/gemm_kernels.cu#L514), most cuda kernels have the same issues . ⏎ I have addressed the issue by incorporating device guards for all kernels. Additionally, I have augmented the kernel tests by including device id,such as in the [test_activation](https://github.com/jeejeeli/vllm/blob/fix-kernel-bug/tests/kernels/test_activation.py#L10-L17)provided

### L3-d10f8e1d43  (L3, 2024-01-17, sha d10f8e1d43bf, PR #1669)
TITLE: [Experimental] Prefix Caching Support (#1669)
SOURCES: path_core
STAGE1: introduce; artifacts=L3.triton.prefix_prefill; Prefix caching added Triton prefix_prefill attention kernel.
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+55/-33); vllm/model_executor/layers/triton_kernel/prefix_prefill.py (+728/-0); .buildkite/test-pipeline.yaml (+4/-0); examples/offline_inference_with_prefix.py (+51/-0); tests/kernels/test_prefix_prefill.py (+168/-0); tests/prefix_caching/test_prefix_caching.py (+41/-0); tests/samplers/test_sampler.py (+12/-6); tests/worker/test_model_runner.py (+3/-2); vllm/block.py (+4/-0); vllm/core/block_manager.py (+42/-5); vllm/core/scheduler.py (+5/-0); vllm/engine/async_llm_engine.py (+13/-3); vllm/engine/llm_engine.py (+17/-1); vllm/entrypoints/api_server.py (+5/-1); vllm/entrypoints/llm.py (+14/-3); vllm/model_executor/input_metadata.py (+6/-0); vllm/model_executor/layers/triton_kernel/__init__.py (+0/-0); vllm/prefix.py (+87/-0); vllm/sequence.py (+6/-1); vllm/worker/model_runner.py (+95/-16)
BODY: add prefix caching support ⏎  ⏎ Section 1 (Basic Functionality): ⏎  ⏎  ⏎ Todo: ⏎ Automatic Prefix Caching Support -- [SGLang RadixAttention](https://github.com/sgl-project/sglang)

### L3-5265631d15  (L3, 2024-01-25, sha 5265631d15d5, PR #2583)
TITLE: use a correct device when creating OptionalCUDAGuard (#2583)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.cache.cuda_reshape; Cache kernel device guard was corrected for OptionalCUDAGuard.
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+1/-1)
BODY: fix of https://github.com/vllm-project/vllm/issues/2350

### L3-9090bf02e7  (L3, 2024-01-28, sha 9090bf02e743, PR #2279)
TITLE: Support FP8-E5M2 KV Cache (#2279)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L3.paged.cuda.v1,L3.cache.cuda_reshape; Attention and cache kernels added FP8 E5M2 KV-cache support.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.flash_attn.upstream_pip
FILES: csrc/attention/attention_dtypes.h (+1/-0); csrc/attention/attention_kernels.cu (+164/-95); csrc/attention/dtype_fp8_e5m2.cuh (+35/-0); csrc/cache_kernels.cu (+108/-27); vllm/model_executor/layers/attention.py (+3/-0); benchmarks/benchmark_latency.py (+8/-0); benchmarks/benchmark_throughput.py (+11/-1); benchmarks/kernels/benchmark_paged_attention.py (+18/-15); csrc/cache.h (+7/-1); csrc/dispatch_utils.h (+10/-0); csrc/ops.h (+4/-2); csrc/pybind.cpp (+4/-0); csrc/quantization/fp8_e5m2_kvcache/quant_utils.cuh (+278/-0); docs/source/quantization/fp8_e5m2_kv_cache.rst (+32/-0); setup.py (+3/-0); tests/kernels/conftest.py (+2/-39); tests/kernels/test_attention.py (+36/-5); tests/kernels/test_cache.py (+7/-3); vllm/config.py (+28/-1); vllm/engine/arg_utils.py (+10/-1); vllm/engine/llm_engine.py (+6/-0); vllm/model_executor/input_metadata.py (+5/-1); vllm/utils.py (+109/-1); vllm/worker/cache_engine.py (+12/-3); vllm/worker/model_runner.py (+7/-0); vllm/worker/worker.py (+4/-1)
PERF_LINES: Quantize KV Cache to fp8 can reduce memory usage of kv cache and then could boost throughput. The impl uses fp8 data type for kv cache and has been tested on A1 | HumanEval-Python-EN | 68.293% | 65.854% (↓ 2.439%) | 67.683% (↓ 0.61%) | HumanEval-Python-CN | 59.146% | 59.146% (=) | 59.756% (↑ 0.61%) | LLaMA-7B | Baseline(KV Cache FP16) | KV Cache FP8 E5M2 | Speedup | Offline throughput (tokens/sec)
BODY: Quantize KV Cache to fp8 can reduce memory usage of kv cache and then could boost throughput. The impl uses fp8 data type for kv cache and has been tested on A100. ⏎  ⏎ The following test is under WarzardCoder-34B. ⏎  ⏎ Dataset | Baseline(KV Cache FP16) | KV Cache FP8 E5M2 | KV Cache FP8 E4M3 ⏎ -- | -- | -- | -- ⏎ HumanEval-Python-EN | 68.293% | 65.854% (↓ 2.439%) | 67.683% (↓ 0.61%) ⏎ HumanEval-Python-CN | 59.146% | 59.146% (=) | 59.756% (↑ 0.61%) ⏎  ⏎ LLaMA-7B | Baseline(KV Cache FP16) | KV Cache FP8 E5M2 | Speedup ⏎ --|--|--|-- ⏎ Offline throughput (tokens/sec) | 1514.35 | 2265.89 | 1.49x ⏎  ⏎ Usage: ⏎ ```python ⏎     from vllm import LLM, SamplingParams ⏎     # Sample prompts. ⏎     prompts = [ ⏎         "Hello, my name is", ⏎         "The president of the United States is", ⏎         "The capital of France is", ⏎         "The future of AI is", ⏎     ] ⏎     # Create a sampling params object. ⏎     sampling_params = SamplingParams(temperature=0.8, top_p=0.95) ⏎     # Create an LLM. ⏎     llm = LLM(model="facebook/opt-125m", kv_cache_dtype="fp8") ⏎     # Generate texts from the prompts. The output is a list of RequestOutput objects ⏎     # that contain the prompt, generated text, and other information. ⏎     outputs = llm.generate(prompts, sampling_params) ⏎     # Print the outputs. ⏎     for output in outputs: ⏎         prompt = output.prompt ⏎         generated_text = output.outputs[0].text ⏎         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}") ⏎ ``` ⏎  ⏎ - **Throughput**: It will increase offline throughput as the memory for kv cache is doubled. If the on line requests are enough, it will also boost the online throughput. ⏎ - **Latency**: It may increase the paged attention kernel as there are quantize/dequantize for cache, especially using fp8e4m3. So we use fp8e5m2 as default.  ⏎ - **Accuracy**: We use HumanEval to evaluate the impact of fp8 and found that both e5m2 and e4m3 could be acceptable. In general, please use e4m3 if you want higher accuracy, but be aware that e4m3 will also make latency high as e4m3 may cost more cycles than e5m2 when casting from fp16/bf16/float.

### L3-923797fea4  (L3, 2024-02-01, sha 923797fea4d8, PR #2648)
TITLE: Fix compile error when using rocm (#2648)
SOURCES: path_core
STAGE1: repair_build_dependency; artifacts=L3.paged.cuda.v1,L3.cache.cuda_reshape; Fixed ROCm compile errors in attention/cache kernel sources.
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape
FILES: csrc/attention/attention_kernels.cu (+2/-0); csrc/cache_kernels.cu (+7/-0); csrc/quantization/fp8_e5m2_kvcache/quant_utils.cuh (+0/-1)
LABELS: rocm
ISSUES: #2646 [BUG] Compile source code error for ROCM6.0
BODY: fix #2646

### L3-0580aab02f  (L3, 2024-02-10, sha 0580aab02ffe, PR #2768)
TITLE: [ROCm] support Radeon™ 7900 series (gfx1100) without using flash-attention (#2768)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes
STAGE1: change_default; artifacts=NEW:early_rocm_attention_selection; Radeon gfx1100 was routed away from unsupported FlashAttention to reference attention.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile.rocm (+12/-3); setup.py (+1/-1); vllm/model_executor/layers/attention.py (+45/-0); docs/source/getting_started/amd-installation.rst (+2/-1)
LABELS: rocm
BODY: This pull request adds vllm support for AMD Radeon™ 7900 series GPU (gfx1100) without using flash-attention. ⏎ Currently, flash-attention does not fully support gfx1100. So, we used vllm reference implementation instead. ⏎  ⏎ Note: ⏎ When building the docker image, pass `--build-arg BUILD_FA="0"` to the `docker build` command.

### L3-5255d99dc5  (L3, 2024-02-15, sha 5255d99dc595, PR #2885)
TITLE: [ROCm] Dockerfile fix for flash-attention build (#2885)
SOURCES: subject_keyword, dependency_pin, release_notes
STAGE1: repair_build_dependency; artifacts=L3.flash_attn.upstream_pip; Fixed Dockerfile path for building FlashAttention on ROCm.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile.rocm (+3/-3)
BODY: This pull request fixes Dockerfile for flash-attention build on ROCm.

### L3-64da65b322  (L3, 2024-02-16, sha 64da65b3225b, PR #2517)
TITLE: Prefix Caching- fix t4 triton error (#2517)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.triton.prefix_prefill; Prefix-prefill Triton kernel used a smaller block size to fix T4 failures.
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/triton_kernel/prefix_prefill.py (+3/-1)
ISSUES: #2513 prefix caching error with baichuan model
BODY: Fix #2513, need a smaller block size for Turing GPUs

### L3-d6e4a130b0  (L3, 2024-02-26, sha d6e4a130b028, PR #3043)
TITLE: [Minor] Remove gather_cached_kv kernel (#3043)
SOURCES: path_core
STAGE1: remove; artifacts=NEW:gather_cached_kv; Removed unused gather_cached_kv cache kernel.
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+0/-161); csrc/cache.h (+0/-7); csrc/pybind.cpp (+0/-4)
BODY: The `gather_cached_kv` kernel was originally implemented for prefix sharing and is not currently used. I believe we can remove the kernel.
