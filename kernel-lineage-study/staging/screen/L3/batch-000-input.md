### L3-ffad4e1e03  (L3, 2023-02-16, sha ffad4e1e031a, PR #)
TITLE: cache_kernel -> cache_kernels
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L3-c413c41cda  (L3, 2023-02-18, sha c413c41cda0f, PR #)
TITLE: Add reshape_and_cache op
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L3-0deacbce6e  (L3, 2023-03-01, sha 0deacbce6e96, PR #3)
TITLE: Implement `single_query_cached_kv_attention` kernel (#3)
SOURCES: path_core, path_integration+keyword, subject_keyword
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.cache.cuda_reshape, L3.flash_attn.upstream_pip
FILES: csrc/attention.cpp (+19/-0); csrc/attention_kernels.cu (+400/-0); csrc/attention_utils.h (+204/-0); csrc/cache_kernels.cu (+5/-5); setup.py (+9/-2); cacheflow/master/block_manager.py (+3/-1); cacheflow/models/attention.py (+19/-35); cacheflow/worker/cache_engine.py (+12/-8); cacheflow/worker/worker.py (+1/-1); csrc/cuda_primitives.h (+1318/-0); (+2 more)
BODY: This PR adds the `single_query_cached_kv_attention` kernel. ⏎  ⏎ Supported data types: ⏎ * `half` ⏎ * `float` ⏎  ⏎ Tested models: ⏎ * OPT-125M ⏎ * OPT-350M ⏎ * OPT-1.3B ⏎ * OPT-2.7B ⏎ * OPT-6.7B ⏎ * OPT-13B ⏎  ⏎ Tested GPUs: ⏎ * A100

### L3-3e9f991d6a  (L3, 2023-03-01, sha 3e9f991d6acd, PR #4)
TITLE: Use FlashAttention for `multi_query_kv_attention` (#4)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: README.md (+1/-0); cacheflow/models/attention.py (+37/-30); server.py (+5/-2); tests/kernels/attention.py (+65/-2)
BODY: This PR is to use [FlashAttention](https://github.com/HazyResearch/flash-attention) kernels for `multi_query_kv_attention`, which performs masked attention for the prompt inputs. ⏎  ⏎ ### Pros ⏎ - FlashAttention is fast and memory-efficient. ⏎ - FlashAttention supports 1D inputs and only invokes a single kernel to handle multiple sequences with variable lengths. ⏎  ⏎ ### Cons ⏎ - FlashAttention **does NOT support FP32**. ⏎ - FlashAttention does not suppo …[truncated]

### L3-1a7eb7da61  (L3, 2023-03-10, sha 1a7eb7da6157, PR #7)
TITLE: Support beam search & parallel generation (#7)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+31/-1); cacheflow/block.py (+4/-0); cacheflow/master/frontend.py (+28/-5); cacheflow/master/scheduler.py (+55/-36); cacheflow/models/__init__.py (+3/-1); cacheflow/models/input_metadata.py (+11/-7); cacheflow/models/model_utils.py (+10/-0); cacheflow/models/opt.py (+2/-1); cacheflow/models/sample.py (+258/-17); cacheflow/sampling_params.py (+54/-19); (+6 more)
BODY: This PR adds support for beam search and parallel generation (i.e., `n` > 1). ⏎  ⏎ **NOTE**: The correctness is only checked for beam search, but not for random sampling methods. ⏎  ⏎ Tested models: ⏎ - OPT-125M ⏎ - OPT-350M ⏎ - OPT-1.3B ⏎ - OPT-2.7B ⏎ - OPT-6.7B ⏎ - OPT-13B ⏎  ⏎ Tested GPUs: ⏎ - A100

### L3-cfae35b861  (L3, 2023-03-13, sha cfae35b861c5, PR #8)
TITLE: Add miscellaneous updates (#8)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+5/-2); cacheflow/master/scheduler.py (+8/-7); cacheflow/models/attention.py (+5/-6); cacheflow/models/memory_analyzer.py (+14/-4); cacheflow/models/sample.py (+1/-1); cacheflow/worker/worker.py (+7/-0); server.py (+4/-2)
BODY: This PR contains several miscellaneous updates to the system, with two notable changes: ⏎  ⏎ 1. The size of the CPU KV cache is now calculated based on the `swap_space` size provided by the user (defaulting to 20 GiB). ⏎ 2. The default value for max_num_batched_tokens has been increased from 2048 to 2560.

### L3-a1b3de86cd  (L3, 2023-03-29, sha a1b3de86cd6f, PR #13)
TITLE: Refactor the test code for attention kernels (#13)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention.py (+53/-19)
BODY: 

### L3-88c0268a18  (L3, 2023-03-30, sha 88c0268a18f1, PR #14)
TITLE: Implement custom kernel for LLaMA rotary embedding (#14)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flash_attn.upstream_pip
FILES: csrc/cache_kernels.cu (+3/-3); cacheflow/models/attention.py (+71/-2); cacheflow/models/llama.py (+5/-59); cacheflow/models/memory_analyzer.py (+1/-2); cacheflow/models/opt.py (+1/-2); cacheflow/models/sample.py (+1/-1); csrc/pos_encoding.cpp (+16/-0); csrc/pos_encoding_kernels.cu (+83/-0); setup.py (+8/-0); tests/kernels/pos_encoding.py (+129/-0)
BODY: This PR implements a custom CUDA kernel for rotary embedding, which is used in LLaMA. The kernel is responsible for the entire process of applying rotary embedding to query and key, and is thus much more efficient than the PyTorch implementation. ⏎  ⏎ Tested models: ⏎ - LLaMA-7B ⏎ - LLaMA-13B ⏎  ⏎ Tested GPUs: ⏎ - A100

### L3-09e9245478  (L3, 2023-04-01, sha 09e9245478a4, PR #16)
TITLE: Add custom kernel for RMS normalization (#16)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.flash_attn.upstream_pip
FILES: csrc/attention_kernels.cu (+1/-0); csrc/attention_utils.h (+0/-39); cacheflow/models/layernorm.py (+26/-0); cacheflow/models/llama.py (+4/-19); csrc/layernorm.cpp (+14/-0); csrc/layernorm_kernels.cu (+61/-0); csrc/reduction_utils.h (+76/-0); setup.py (+8/-0); tests/kernels/layernorm.py (+53/-0)
BODY: This PR adds a custom CUDA kernel for RMS normalization, which is used in LLaMA models. The kernel removes the inefficient data movement in the current PyTorch implementation. ⏎  ⏎ Performance (fp16): ⏎ ``` ⏎ * num_tokens=7, hidden_size=1024 ⏎ Kernel: 9 us ⏎ PyTorch: 93 us ⏎  ⏎ * num_tokens=128, hidden_size=1024 ⏎ Kernel: 5 us ⏎ PyTorch: 84 us ⏎  ⏎ * num_tokens=2048, hidden_size=5120 ⏎ Kernel: 60 us ⏎ PyTorch: 353 us ⏎ ``` ⏎  ⏎ Tested models: ⏎ - LLaMA-7B ⏎ - LLaMA …[truncated]

### L3-897cb2ae28  (L3, 2023-04-02, sha 897cb2ae28e9, PR #20)
TITLE: Optimize data movement (#20)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.cache.cuda_reshape, L3.flash_attn.upstream_pip
FILES: csrc/attention_kernels.cu (+13/-9); csrc/cache_kernels.cu (+17/-8); cacheflow/models/activation.py (+20/-0); cacheflow/models/attention.py (+46/-46); cacheflow/models/input_metadata.py (+5/-0); cacheflow/models/llama.py (+7/-12); cacheflow/models/opt.py (+2/-5); cacheflow/worker/worker.py (+8/-0); csrc/activation.cpp (+12/-0); csrc/activation_kernels.cu (+46/-0); (+7 more)
BODY: Should be merged after #15 . ⏎  ⏎ The changes in this PR eliminate the need for redundant data movements such as `torch.cat`, `torch.stack`, and `torch.contiguous`, which were previously used to align input and output shapes. The PR modifies existing kernels and adds new kernels to accommodate non-contiguous tensors, making these data movement operators unnecessary.

### L3-21b3671bbc  (L3, 2023-04-04, sha 21b3671bbc50, PR #24)
TITLE: Basic attention kernel that supports cached KV + (multi-)prompts (#24)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.paged.cuda.v1
FILES: csrc/attention.cpp (+16/-0); csrc/attention_kernels.cu (+463/-0); tests/kernels/attention.py (+143/-0)
BODY: This PR implements a basic and not highly optimized kernel to support cached KV with multiple import prompts.

### L3-0f40557af6  (L3, 2023-04-07, sha 0f40557af614, PR #32)
TITLE: Implement block copy kernel to optimize beam search (#32)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+82/-22); benchmark/benchmark_latency.py (+6/-3); cacheflow/models/sample.py (+3/-2); cacheflow/worker/cache_engine.py (+4/-20); csrc/cache.cpp (+2/-2); tests/kernels/cache.py (+58/-0)
BODY: This PR implements a block copy kernel. By using this kernel, we can reduce the number of kernel invocations from `2 * num_layers * num_copying_blocks` to 1.

### L3-c267b1a02c  (L3, 2023-04-08, sha c267b1a02c95, PR #27)
TITLE: Add query stride to multi_query_cached_kv_attention & Add kernel benchmark script (#27)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.paged.cuda.v1
FILES: csrc/attention_kernels.cu (+11/-5); benchmark/benchmark_attention.py (+165/-0); tests/kernels/attention.py (+5/-3)
BODY: This PR adds query stride to the `multi_query_cached_kv_attention` kernel so that it can support non-contiguous query tensors in our OPT and LLaMA models (after #20). ⏎  ⏎ This PR also adds a benchmark script comparing our `multi_query_cached_kv_attention` and the optimized Flash attention implementation.

### L3-b9926f7f66  (L3, 2023-04-09, sha b9926f7f663b, PR #35)
TITLE: Support block size 32 (#35)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1
FILES: csrc/attention_kernels.cu (+44/-0); cacheflow/master/block_manager.py (+2/-2); cacheflow/master/server.py (+1/-1); tests/kernels/attention.py (+2/-2)
BODY: This PR adds support for block size 32. It turns out that no modification to our attention kernel is required for this support.

### L3-e3cec88aa5  (L3, 2023-04-10, sha e3cec88aa5b7, PR #29)
TITLE: Memcpy kernel for flash attention (#29)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+157/-0); benchmark/benchmark_cache.py (+81/-0); csrc/cache.cpp (+11/-0); tests/kernels/cache.py (+44/-0)
BODY: Memcpy kernel for flash attention ⏎  ⏎ ``` ⏎ num_tokens: 64, num_heads: 40, head_size: 128, block_size: 8, num_blocks: 1024, dtype: torch.float16 ⏎ [Latency] gather_cached_kv: 0.008 ms ⏎ [Throughput] gather_cached_kv: 156.479 GB/s ⏎ num_tokens: 128, num_heads: 40, head_size: 128, block_size: 8, num_blocks: 1024, dtype: torch.float16 ⏎ [Latency] gather_cached_kv: 0.011 ms ⏎ [Throughput] gather_cached_kv: 216.171 GB/s ⏎ num_tokens: 256, num_heads: 40, head_ …[truncated]

### L3-0f4b32199e  (L3, 2023-04-15, sha 0f4b32199ec6, PR #38)
TITLE: Support various block sizes & Change default block size to 16 (#38)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1
FILES: csrc/attention.cpp (+0/-16); csrc/attention_kernels.cu (+557/-579); benchmark/benchmark_text_completion.py (+1/-0); cacheflow/master/block_manager.py (+0/-3); cacheflow/master/scheduler.py (+2/-1); cacheflow/master/server.py (+2/-2); csrc/cuda_primitives.h (+40/-18)
BODY: 

### L3-436e523bf1  (L3, 2023-05-03, sha 436e523bf161, PR #53)
TITLE: Refactor attention kernels (#53)
SOURCES: path_core, path_integration+keyword, subject_keyword
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.flash_attn.upstream_pip
FILES: csrc/attention/attention_dtypes.cuh (+5/-0); csrc/attention/attention_generic.cuh (+47/-0); csrc/attention/attention_kernels.cu (+451/-0); csrc/attention/attention_utils.cuh (+38/-0); csrc/attention/dtype_float16.cuh (+426/-0); csrc/attention/dtype_float32.cuh (+250/-0); csrc/attention_kernels.cu (+0/-896); csrc/attention_utils.h (+0/-165); setup.py (+1/-1); csrc/cuda_primitives.h (+0/-1340); (+4 more)
BODY: This PR refactors attention kernels, making the helper functions more modular and pruning unused code. This PR will make it easier to add support for a new data type such as bfloat16. ⏎  ⏎ In addition, this PR reduces the computation overhead of the attention kernel, by using the reduced precision (i.e., fp16) for `logits * V` instead of the full precision. This is compatible with the [FasterTransformer's implementation](https://github.com/NVIDIA/F …[truncated]

### L3-e070829ae8  (L3, 2023-05-03, sha e070829ae81d, PR #54)
TITLE: Support bfloat16 data type (#54)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.flash_attn.upstream_pip
FILES: csrc/attention/attention_dtypes.h (+4/-0); csrc/attention/attention_kernels.cu (+6/-2); csrc/attention/attention_utils.cuh (+1/-1); csrc/attention/dtype_bfloat16.cuh (+361/-0); csrc/attention/dtype_float32.cuh (+1/-1); csrc/cache_kernels.cu (+55/-43); cacheflow/master/server.py (+2/-2); cacheflow/models/utils.py (+1/-0); csrc/activation_kernels.cu (+3/-1); csrc/layernorm_kernels.cu (+3/-1); (+2 more)
BODY: Should be merged after #53 ⏎  ⏎ This PR adds support for the bfloat16 data type, which is used for some LLMs including Dolly V2.

### L3-130d5fd8c7  (L3, 2023-05-04, sha 130d5fd8c790, PR #68)
TITLE: Fix a bug in attention kernel (#68)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+1/-1)
ISSUES: #66 A critical bug in attention kernel after refactoring
BODY: Fixes #66  ⏎  ⏎ This PR fixes a bug in our attention kernel. The bug was introduced in #53 when changing the precision of computations in the attention kernel. Now the kernel unit tests are passed normally.

### L3-c9d5b6d4a8  (L3, 2023-05-05, sha c9d5b6d4a8b3, PR #70)
TITLE: Replace FlashAttention with xformers (#70)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: README.md (+1/-5); cacheflow/master/server.py (+1/-1); cacheflow/models/attention.py (+16/-38); cacheflow/models/input_metadata.py (+9/-12); cacheflow/models/llama.py (+2/-2); cacheflow/models/memory_analyzer.py (+6/-6); cacheflow/models/opt.py (+2/-2); cacheflow/worker/worker.py (+0/-8); tests/kernels/activation.py (+1/-1); tests/kernels/attention.py (+35/-44); (+3 more)
BODY: This PR replaces FlashAttention with [xformers](https://github.com/facebookresearch/xformers). ⏎  ⏎ Pros: ⏎ - Richer features & higher compatibility. xformers supports attention bias, FP32, head size 256, and old GPUs (such as V100) while FlashAttention does not. ⏎ - xformers provides pre-compiled python wheels, while FlashAttention compiles the entire CUDA code during installation. ⏎ - Future-proof, as the repository is maintained by many developers  …[truncated]

### L3-7c041ab578  (L3, 2023-05-09, sha 7c041ab57847, PR #82)
TITLE: Refactor system architecture (#82)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: cacheflow/model_executor/layers/attention.py (+1/-1); cacheflow/core/block_manager.py (+0/-0); cacheflow/core/policy.py (+0/-0); cacheflow/core/scheduler.py (+2/-2); cacheflow/core/server.py (+5/-4); cacheflow/frontend/fastapi_frontend.py (+7/-7); cacheflow/frontend/simple_frontend.py (+0/-0); cacheflow/model_executor/__init__.py (+11/-0); cacheflow/model_executor/input_metadata.py (+0/-0); cacheflow/model_executor/layers/activation.py (+0/-0); (+30 more)
BODY: This PR includes extensive refactoring of the system. ⏎  ⏎ Major changes are: ⏎ - Moved `parallel_utils` into `model_executor` ⏎ - Moved `simple_frontend` to `frontend` ⏎ - Moved `gradio_webserver` and `test_cli_client` to the root ⏎ - Removed `plot`

### L3-667ba3995c  (L3, 2023-05-14, sha 667ba3995c01, PR #104)
TITLE: Add copyright headers to source files adapted from FT (#104)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_generic.cuh (+17/-0); csrc/attention/attention_kernels.cu (+17/-0); csrc/attention/attention_utils.cuh (+17/-0); csrc/attention/dtype_bfloat16.cuh (+18/-0); csrc/attention/dtype_float16.cuh (+18/-0); csrc/attention/dtype_float32.cuh (+18/-0); cacheflow/model_executor/models/gpt2.py (+1/-2); cacheflow/model_executor/models/gpt_neox.py (+1/-2); cacheflow/model_executor/models/llama.py (+1/-2); cacheflow/model_executor/models/opt.py (+1/-2); (+1 more)
BODY: 

### L3-b322fd1607  (L3, 2023-05-14, sha b322fd160759, PR #100)
TITLE: Add docstrings to some modules and classes (#100)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: cacheflow/model_executor/layers/attention.py (+28/-1); cacheflow/block.py (+9/-2); cacheflow/core/block_manager.py (+20/-15); cacheflow/core/server.py (+2/-2); cacheflow/model_executor/layers/activation.py (+5/-0); cacheflow/model_executor/layers/layernorm.py (+6/-0); cacheflow/model_executor/layers/sampler.py (+14/-0); cacheflow/model_executor/model_loader.py (+2/-2); cacheflow/model_executor/models/gpt2.py (+5/-1); cacheflow/model_executor/models/gpt_neox.py (+6/-1); (+7 more)
BODY: 

### L3-f756799b84  (L3, 2023-05-19, sha f756799b84f5, PR #81)
TITLE: Use runtime profiling to replace manual memory analyzers (#81)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: cacheflow/model_executor/layers/attention.py (+29/-21); cacheflow/core/server.py (+30/-22); cacheflow/frontend/fastapi_frontend.py (+7/-3); cacheflow/model_executor/__init__.py (+4/-3); cacheflow/model_executor/layers/sampler.py (+1/-1); cacheflow/model_executor/memory_analyzer.py (+0/-370); cacheflow/model_executor/model_loader.py (+0/-37); cacheflow/model_executor/models/gpt2.py (+2/-1); cacheflow/model_executor/models/gpt_neox.py (+2/-1); cacheflow/model_executor/models/llama.py (+2/-1); (+4 more)
ISSUES: #59 Profile memory usage
BODY: Fix #59. ⏎  ⏎ Previously we used a manual memory profiler, which must be implemented separately for each model. This PR replaces it with a general memory profiler.

### L3-c3442c1f6f  (L3, 2023-05-20, sha c3442c1f6fab, PR #109)
TITLE: Refactor system architecture (#109)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: cacheflow/model_executor/layers/attention.py (+1/-1); README.md (+8/-5); cacheflow/__init__.py (+19/-0); cacheflow/config.py (+165/-0); cacheflow/core/scheduler.py (+80/-84); cacheflow/core/server.py (+0/-302); cacheflow/entrypoints/fastapi_server.py (+128/-0); cacheflow/frontend/fastapi_frontend.py (+0/-201); cacheflow/frontend/simple_frontend.py (+0/-72); cacheflow/model_executor/__init__.py (+1/-3); (+14 more)
BODY: Should be merged after #81

### L3-a283ec2eec  (L3, 2023-05-23, sha a283ec2eece5, PR #122)
TITLE: Add contributing guideline and mypy config (#122)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: cacheflow/model_executor/layers/attention.py (+2/-2); CONTRIBUTING.md (+74/-0); cacheflow/core/scheduler.py (+1/-1); cacheflow/model_executor/layers/sampler.py (+1/-1); cacheflow/model_executor/model_loader.py (+3/-1); cacheflow/model_executor/models/gpt2.py (+4/-4); cacheflow/model_executor/models/gpt_neox.py (+6/-6); cacheflow/model_executor/models/llama.py (+6/-6); cacheflow/model_executor/models/opt.py (+7/-7); cacheflow/outputs.py (+1/-1); (+6 more)
ISSUES: #73 Use mypy
BODY: Fixes #73  ⏎  ⏎ This PR adds `CONTRIGUING.md` and mypy configs. While the repository currently does not completely pass the mypy tests, it helps me identify some type bugs in our codebase. Those bugs are also fixed in this PR.

### L3-d721168449  (L3, 2023-05-27, sha d72116844928, PR #130)
TITLE: Improve setup script & Add a guard for bfloat16 kernels (#130)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.flash_attn.upstream_pip
FILES: csrc/attention/attention_dtypes.h (+0/-3); csrc/attention/attention_kernels.cu (+0/-2); csrc/attention/dtype_bfloat16.cuh (+44/-0); setup.py (+46/-11)
BODY: This PR fixes `setup.py` so that the compiled modules can include the binaries for multiple types of GPUs. And the PR improves how we avoid build errors on bfloat16 kernels.

### L3-e38074b1e6  (L3, 2023-06-07, sha e38074b1e6ad, PR #141)
TITLE: Support FP32 (#141)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.flash_attn.upstream_pip
FILES: cacheflow/model_executor/layers/attention.py (+3/-5); csrc/attention/attention_kernels.cu (+37/-32); cacheflow/config.py (+3/-4); cacheflow/entrypoints/llm.py (+5/-4); cacheflow/server/arg_utils.py (+4/-4); docs/source/getting_started/installation.rst (+3/-0); setup.py (+5/-0); tests/kernels/test_attention.py (+5/-5)
ISSUES: #72 Support FP32
BODY: Closes #72  ⏎  ⏎ ~~This PR removes the support for some head and block sizes to enhance the compilation speed. The removed sizes are not used for the models we currently support. In addition, removing them allows us to support FP32. On my machine, the compilation time got reduced **from 7.5 mins to 1.5 mins** even though FP32 is now added.~~ ⏎  ⏎ This PR removes the support for some head and block sizes to enable FP32. Besides, the PR changes the `de …[truncated]

### L3-0b98ba15c7  (L3, 2023-06-17, sha 0b98ba15c744, PR #150)
TITLE: Change the name to vLLM (#150)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.flash_attn.upstream_pip
FILES: csrc/attention/attention_generic.cuh (+3/-3); csrc/attention/attention_kernels.cu (+4/-4); csrc/attention/attention_utils.cuh (+3/-3); csrc/attention/dtype_bfloat16.cuh (+3/-3); csrc/attention/dtype_float16.cuh (+3/-3); csrc/attention/dtype_float32.cuh (+3/-3); csrc/cache_kernels.cu (+9/-9); CONTRIBUTING.md (+6/-6); README.md (+5/-5); benchmarks/README.md (+1/-1); (+80 more)
BODY: The current plan is ⏎ 1. Merge this PR ⏎ 2. Change the repo name to `vllm` ⏎ 3. Move the repo to an organization ⏎ 4. Fix readthedocs URL

### L3-d6fa1be3a8  (L3, 2023-07-03, sha d6fa1be3a8ef, PR #326)
TITLE: [Quality] Add code formatter and linter (#326)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+81/-32); .pylintrc (+434/-0); CONTRIBUTING.md (+4/-1); examples/api_client.py (+5/-2); examples/gradio_webserver.py (+16/-9); examples/llm_engine_example.py (+7/-2); examples/offline_inference.py (+0/-1); examples/openai_client.py (+7/-2); format.sh (+108/-0); requirements-dev.txt (+11/-1); (+37 more)
ISSUES: #57 Add code formatting script & Add CI to check code format
BODY: Partially fix #57. Adding formatter and linter. ⏎  ⏎ TODO: Add formatter into CI.

### L3-e41f06702c  (L3, 2023-07-03, sha e41f06702cb6, PR #331)
TITLE: Add support for BLOOM (#331)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention.cpp (+3/-1); csrc/attention/attention_kernels.cu (+19/-6); vllm/model_executor/layers/attention.py (+128/-5); README.md (+1/-0); docs/source/models/supported_models.rst (+3/-0); tests/kernels/test_attention.py (+1/-0); vllm/model_executor/input_metadata.py (+4/-2); vllm/model_executor/model_loader.py (+2/-3); vllm/model_executor/models/__init__.py (+2/-0); vllm/model_executor/models/bloom.py (+316/-0); (+1 more)
ISSUES: #61 Support BLOOM
BODY: Closes #61  ⏎  ⏎ This PR adds the BLOOM model and modifies the paged attention kernel to support ALiBi bias.

### L3-404422f42e  (L3, 2023-07-03, sha 404422f42ed9, PR #334)
TITLE: [Model] Add support for MPT (#334)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+3/-0); vllm/model_executor/layers/attention.py (+1/-1); README.md (+1/-0); docs/source/models/supported_models.rst (+3/-0); vllm/config.py (+3/-2); vllm/model_executor/model_loader.py (+2/-1); vllm/model_executor/models/__init__.py (+2/-0); vllm/model_executor/models/mpt.py (+279/-0); vllm/transformers_utils/config.py (+15/-0); vllm/transformers_utils/configs/__init__.py (+5/-0); (+1 more)
ISSUES: #218 Support for MPT-7B and MPT-30B | #332 feature request: support mpt-30b
BODY: Closes #218 and #332 ⏎  ⏎ Should be merged after #61

### L3-c894836108  (L3, 2023-07-08, sha c89483610873, PR #226)
TITLE: [Model] Add support for GPT-J (#226)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+4/-4); vllm/model_executor/layers/attention.py (+1/-1); README.md (+1/-0); docs/source/models/supported_models.rst (+3/-0); tests/kernels/test_attention.py (+2/-2); vllm/model_executor/layers/sampler.py (+3/-0); vllm/model_executor/model_loader.py (+1/-0); vllm/model_executor/models/__init__.py (+2/-0); vllm/model_executor/models/gpt_j.py (+251/-0); vllm/model_executor/models/mpt.py (+1/-0)
BODY: reference to issue https://github.com/vllm-project/vllm/issues/198

### L3-96853af5a8  (L3, 2023-07-14, sha 96853af5a830, PR #452)
TITLE: Optimize MQA Kernel (#452)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention.cpp (+1/-0); csrc/attention/attention_kernels.cu (+22/-9); vllm/model_executor/layers/attention.py (+32/-11); vllm/config.py (+7/-0); vllm/model_executor/models/gpt_bigcode.py (+22/-52)
ISSUES: #393 [StarCoder] TypeError: Got unsupported ScalarType BFloat16 | #462 Starcoder is 5-10x slower on vllm than HF's TGI when passing in a continuous batch of requests
BODY: This PR implements the MQA paged attention kernel and modifies the GPT Bigcode model to utilize the optimized MQA kernel. ⏎  ⏎ TODO: Check performance gain.

### L3-bda41c70dd  (L3, 2023-07-18, sha bda41c70ddb1, PR #496)
TITLE: hotfix attn alibi wo head mapping (#496)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+1/-0); tests/kernels/test_attention.py (+2/-0)
BODY: 

### L3-6fc2a38b11  (L3, 2023-07-20, sha 6fc2a38b110f, PR #505)
TITLE: Add support for LLaMA-2 (#505)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+9/-6); README.md (+2/-1); csrc/pos_encoding_kernels.cu (+13/-8); docs/source/models/supported_models.rst (+2/-2); requirements.txt (+1/-1); vllm/config.py (+6/-1); vllm/model_executor/models/llama.py (+34/-19)
ISSUES: #501 Support LLaMA-2
BODY: Fix #501  ⏎  ⏎ ~Update: this PR has some correctness issues on 70B models. Will look into it.~ ⏎  ⏎ @WoosukKwon This PR is ready to go. Please review and let's merge it!

### L3-d7a1c6d614  (L3, 2023-07-24, sha d7a1c6d61475, PR #495)
TITLE: Fix paged attention testing. (#495)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_attention.py (+10/-0)
BODY: 

### L3-1b0bd0fe8a  (L3, 2023-08-02, sha 1b0bd0fe8a4a, PR #592)
TITLE: Add Falcon support (new) (#592)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+24/-12); README.md (+1/-0); csrc/pos_encoding_kernels.cu (+29/-13); docs/source/models/supported_models.rst (+3/-0); examples/llm_engine_example.py (+2/-1); vllm/config.py (+7/-2); vllm/model_executor/model_loader.py (+2/-0); vllm/model_executor/models/__init__.py (+2/-0); vllm/model_executor/models/falcon.py (+496/-0); vllm/model_executor/parallel_utils/parallel_state.py (+1/-72); (+6 more)
ISSUES: #195 Support for Falcon-7B / 40B models
BODY: Close #195 #197 #356 ⏎  ⏎ This PR replaces PR #321  ⏎  ⏎ Also, revert an early all-reduce optimization that stores all all-reduce results in a shared buffer. This can lead to wrong results in distributed settings for models with two parallel all-reduce branches. ⏎  ⏎ Correctness check:

### L3-55fe8a81ec  (L3, 2023-08-02, sha 55fe8a81ec5d, PR #658)
TITLE: Refactor scheduler (#658)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+17/-5); examples/llm_engine_example.py (+1/-1); vllm/core/scheduler.py (+96/-129); vllm/engine/llm_engine.py (+91/-9)
BODY: This PR refactors the vLLM's scheduler. For a major change, vLLM now uses a TGI-style continuous batching mechanism. For minor changes, this PR makes the scheduler code more concise by moving the logging part to `LLMEngine` and moving some data structures to `SchedulerOutputs`.

### L3-79af7e96a0  (L3, 2023-08-04, sha 79af7e96a0e2, PR #420)
TITLE: [OPTIMIZATION] Optimizes the single_query_cached_kv_attention kernel (#420)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+7/-4)
BODY: Instead of having each thread group fetch the query head (which causes 64x memory to be read), we have all threads in the block share the task of loading the query head. On the benchmark of running 1000 sequences through LLaMA13B on an A100 (80GB), this improves the throughput by 1.10x.

### L3-2a4ec90854  (L3, 2023-08-23, sha 2a4ec90854ae, PR #834)
TITLE: Fix for breaking changes in xformers 0.0.21 (#834)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+3/-2); requirements.txt (+1/-1)
ISSUES: #407 RuntimeError: attn_bias is not correctly aligned | #468 attn_bias not aligned & some questions regarding float16 | #795 "attn_bias is not correctly aligned" on A100 for MPT-30B | #832 Issue while loading MPT-7B-8K-instruct
BODY: Fixes #832  ⏎  ⏎ This PR fixes the breaking changes made in the latest release of xformers. Specifically, now the xformers attention bias should have the batch dimension, which is 1 in the current vLLM implementation.

### L3-75471386de  (L3, 2023-08-29, sha 75471386de62, PR #877)
TITLE: use flash-attn via xformers (#877)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+0/-3); tests/kernels/test_attention.py (+0/-2)
ISSUES: #485 Flash Attention V2
BODY: fixes https://github.com/vllm-project/vllm/issues/485#issuecomment-1693821853

### L3-8ce9c50d40  (L3, 2023-09-02, sha 8ce9c50d4034, PR #933)
TITLE: Avoid compiling kernels for double data type (#933)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+5/-9); csrc/activation_kernels.cu (+4/-6); csrc/dispatch_utils.h (+14/-0); csrc/layernorm_kernels.cu (+2/-3); csrc/pos_encoding_kernels.cu (+3/-3)
BODY: This PR fixes a dispatch logic for our custom CUDA kernels. Currently, vLLM uses `AT_DISPATCH_FLOATING_TYPES_AND2` which actually [includes the double data type](https://github.com/pytorch/pytorch/blob/e9ebda29d87ce0916ab08c06ab26fd3766a870e5/aten/src/ATen/Dispatch.h#L245) that is never used. This leads to unnecessary increase in compilation time and binary size. The PR solves this issue by limiting the data types to `float`, `half` and `bfloat16 …[truncated]

### L3-bf87484efa  (L3, 2023-09-04, sha bf87484efac9, PR #936)
TITLE: [BugFix] Fix NaN errors in paged attention kernel (#936)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+12/-0); csrc/attention/dtype_bfloat16.cuh (+10/-0); csrc/attention/dtype_float16.cuh (+5/-5); csrc/attention/dtype_float32.cuh (+5/-0)
ISSUES: #641 RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
BODY: Fixes #641  ⏎ This PR fixes the paged attention kernel. Currently, the kernel computes `attn_weight * value` for all tokens in a value block, even if some of them are not included in the context. It is generally acceptable since the `attn_weight` for those tokens is 0, but this causes errors when the tokens contain NaNs (since 0 * NaN is NaN). The PR solves this by explicitly setting the values of those tokens as 0.

### L3-320a622ec4  (L3, 2023-09-06, sha 320a622ec4d0, PR #941)
TITLE: [BugFix] Implement RoPE for GPT-J (#941)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+5/-2); csrc/pos_encoding.cpp (+6/-5); csrc/pos_encoding_kernels.cu (+68/-45); tests/kernels/test_pos_encoding.py (+38/-18); vllm/model_executor/models/gpt_j.py (+5/-2)
ISSUES: #590 GPTJ output not consistent with that of transformers | #747 Maybe Wrong implementation of AttentionWithRoPE for GPTJ and GPT-NeoX?
BODY: Fixes #747 and fixes #590 ⏎  ⏎ This PR fixes a bug in the GPT-J model implementation. The GPT-J model uses a rotary embedding that is slightly different from the GPT-NeoX style rotary embedding (which is commonly used for LLaMA and recent models). The difference isn't considered in the vLLM's current implementation. The PR resolves this by adding a RoPE kernel for GPT-J. After this fix, I've checked that the outputs of GPT-J when using FP32 and arg …[truncated]

### L3-db09d4ad83  (L3, 2023-09-07, sha db09d4ad833b, PR #945)
TITLE: [FIX] Fix Alibi implementation in PagedAttention kernel (#945)
SOURCES: path_core, subject_keyword, body_keyword, release_notes
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+1/-1); tests/kernels/test_attention.py (+3/-2)
BODY: cc @Oliver-ss  ⏎  ⏎ Will verify and test the correctness.

### L3-4b5bcf8906  (L3, 2023-09-08, sha 4b5bcf89065e, PR #982)
TITLE: faster startup of vLLM  (#982)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+3/-2)
BODY: ## issue ## ⏎  ⏎ In production + using autoscaling the startup time of servers and vllm is very important to performance. Currently a lot of CPU work is being done on startup which can be done blazingly fast if done on the GPU instead ⏎  ⏎ ## metrics ## ⏎  ⏎ 2 CPU environment ⏎ defaultdict(<class 'int'>, {'rope': 143.8646891117096s}) ⏎  ⏎ 4-8 CPU environment ⏎ ~ 20-40s ⏎  ⏎ 20 CPU environment ⏎ ~10s (not exact number) ⏎  ⏎ on device ⏎ defaultdict(<class 'int'>,  …[truncated]

### L3-a62de9ecfd  (L3, 2023-09-09, sha a62de9ecfdc6, PR #996)
TITLE: Fix wrong dtype in PagedAttentionWithALiBi bias (#996)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+11/-4)
ISSUES: #995 PagedAttentionWithALiBi uses wrong dtype for bias
BODY: Fixes dtype not being set correctly as it was derived from the default dtype before (which will now always be float32). ⏎  ⏎ Closes https://github.com/vllm-project/vllm/issues/995 ⏎  ⏎ cc @WoosukKwon @chu-tianxiang

### L3-e67b4f2c2a  (L3, 2023-09-11, sha e67b4f2c2a21, PR #1004)
TITLE: Use FP32 in RoPE initialization (#1004)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+4/-4); tests/kernels/test_pos_encoding.py (+3/-2)
ISSUES: #883 Code llama output gibberish code.
BODY: Fixes #883 ⏎  ⏎ This PR uses FP32 for initializing the cos and sin cache in RoPE. This resolves the precision issues when using a large `base` like 1000000 with FP16. ⏎  ⏎ NOTE: This change is aligned with the [RoPE definition](https://github.com/huggingface/transformers/blob/95b374952dc27d8511541d6f5a4e22c9ec11fb24/src/transformers/models/llama/modeling_llama.py#L99) in HuggingFace. However, as mentioned in #863, this may not align with the original …[truncated]

### L3-03ffd0a022  (L3, 2023-09-26, sha 03ffd0a02251, PR #1176)
TITLE: Add comments on RoPE initialization (#1176)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+9/-1)
BODY: Added more comments that can be useful for understanding the differences from HF.

### L3-cf5cb1e33e  (L3, 2023-09-26, sha cf5cb1e33eed, PR #1154)
TITLE: Allocate more shared memory to attention kernel (#1154)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.flash_attn.upstream_pip
FILES: csrc/attention/attention_kernels.cu (+5/-0); setup.py (+11/-0); vllm/utils.py (+12/-1); vllm/worker/worker.py (+25/-1); csrc/cuda_utils.cpp (+13/-0); csrc/cuda_utils_kernels.cu (+14/-0); tests/kernels/test_attention.py (+7/-1)
ISSUES: #905 vLLM doesn't support context length exceeding about 13k
BODY: Makes use of additional shared memory present on compute capability >=7.0 cards to support longer context length in the attention kernel. ⏎  ⏎ See https://stackoverflow.com/questions/63757245/using-maximum-shared-memory-in-cuda for details. ⏎  ⏎ As pointed out by @WoosukKwon offline, ideally we would also store logits inside the kernel in float16 instead of float32 as the accuracy loss should be minimal. This will enable even longer context lengths. …[truncated]

### L3-21877b0d75  (L3, 2023-09-27, sha 21877b0d7523, PR #555)
TITLE: Support Longchat and RoPE scaling (#555)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+25/-38); vllm/config.py (+11/-0); vllm/model_executor/layers/rotary_embedding.py (+169/-0); vllm/model_executor/models/llama.py (+6/-2)
LABELS: new-model
ISSUES: #333 Support for Condensed RotaryEmbeddings
BODY: Add LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding. Attempt to fix #333, #464, #479

### L3-bb1ba58f06  (L3, 2023-09-28, sha bb1ba58f0647, PR #1196)
TITLE: [Mistral] Mistral-7B-v0.1 support (#1196)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+22/-5); requirements.txt (+1/-1); vllm/config.py (+2/-0); vllm/core/block_manager.py (+30/-11); vllm/core/scheduler.py (+1/-1); vllm/engine/arg_utils.py (+3/-3); vllm/engine/llm_engine.py (+2/-0); vllm/model_executor/input_metadata.py (+20/-1); vllm/model_executor/model_loader.py (+1/-0); vllm/model_executor/models/__init__.py (+2/-0); (+3 more)
ISSUES: #1199 Support for Mistral 7B
BODY: 

### L3-6f88f762bf  (L3, 2023-09-28, sha 6f88f762bf90, PR #1223)
TITLE: Fix OOM in attention kernel test (#1223)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_attention.py (+5/-2)
BODY: Currently, we use a very large sequence length (e.g., 40K) to test prompt attention. This causes OOM in reference implementation which does not use flash attention. As `xformers` is already tested with its own tests, I believe it's ok to use shorter sequences in our tests.

### L3-ebe4d1db3a  (L3, 2023-10-01, sha ebe4d1db3a42, PR #1241)
TITLE: Fix boundary check in paged attention kernel (#1241)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+1/-1)
BODY: This PR fixes a vulnerable memory modification to gpu shared memory in CUDA kernel function `single_query_cached_kv_attention_kernel`.

### L3-928de46888  (L3, 2023-10-16, sha 928de46888b9, PR #1348)
TITLE: Implement PagedAttention V2 (#1348)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention.cpp (+24/-4); csrc/attention/attention_kernels.cu (+413/-71); csrc/attention/dtype_bfloat16.cuh (+5/-0); vllm/model_executor/layers/attention.py (+71/-47); benchmarks/kernels/benchmark_paged_attention.py (+197/-0); tests/kernels/test_attention.py (+54/-17)
BODY: This PR implements the first part of the PagedAttention V2 kernel, which uses sequence-level parallelism for better work partitioning. Compared to V1, the V2 kernel achieves huge speedup when the batch size is small (e.g., <= 8). We will further optimize the kernel henceforth.

### L3-9d9072a069  (L3, 2023-10-16, sha 9d9072a06920, PR #1328)
TITLE: Implement prompt logprobs & Batched topk for computing logprobs (#1328)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+1/-1); examples/llm_engine_example.py (+1/-1); tests/async_engine/test_request_tracker.py (+1/-1); tests/conftest.py (+33/-0); tests/samplers/test_logprobs.py (+55/-0); vllm/config.py (+1/-1); vllm/engine/llm_engine.py (+13/-7); vllm/model_executor/layers/sampler.py (+203/-107); vllm/model_executor/parallel_utils/communication_op.py (+1/-1); vllm/model_executor/parallel_utils/layers.py (+1/-1); (+4 more)
BODY: This PR: ⏎  ⏎ - Added `prompt_logprobs` to `SamplingParams` and `RequestOutput`. This makes vLLM to support returning the log probabilities of prompt tokens, which is required to support `echo` in OpenAI server. ⏎ - Refactor the logprobs logic so that the query to `topk` logits is done in a batched fashion. ⏎  ⏎ This PR will have merge conflicts with #1337. I think a good plan is to perform the optimization in #1337 along with the refactoring of `Inpu …[truncated]

### L3-c1376e0f82  (L3, 2023-10-16, sha c1376e0f825e, PR #1381)
TITLE: Change scheduler & input tensor shape (#1381)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+7/-2); vllm/model_executor/layers/attention.py (+71/-95); csrc/activation_kernels.cu (+14/-14); csrc/layernorm_kernels.cu (+6/-6); csrc/pos_encoding_kernels.cu (+11/-11); vllm/config.py (+3/-0); vllm/core/scheduler.py (+11/-4); vllm/engine/arg_utils.py (+7/-1); vllm/model_executor/input_metadata.py (+5/-6); vllm/model_executor/layers/activation.py (+8/-12); (+3 more)
BODY: This PR updates the scheduler and model code to use 2D tensors instead of 1D tensors. The change will enable using a wider range of libraries and hardware, and facilitate future optimizations like CUDA graph.

### L3-0ce8647dc5  (L3, 2023-10-31, sha 0ce8647dc5bc, PR #1514)
TITLE: Fix integer overflows in attention & cache ops (#1514)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape
FILES: csrc/attention/attention_kernels.cu (+8/-2); csrc/cache_kernels.cu (+36/-36); tests/kernels/test_attention.py (+1/-1); tests/kernels/test_cache.py (+7/-7); vllm/worker/worker.py (+1/-1)
ISSUES: #1486 Array Index Overflow for Large #Blocks
BODY: Fixes #1486  ⏎  ⏎ This PR fixes the overflows in paged attention & cache ops when the number of blocks is huge.

### L3-9738b84a08  (L3, 2023-11-01, sha 9738b84a0895, PR #1510)
TITLE: Force paged attention v2 for long contexts (#1510)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+3/-1); vllm/worker/worker.py (+1/-28)
BODY: Removes the hard limit on context length that was tied to paged attention v1 limitations and instead forces v2 to be used if the context cannot fit in shared memory.

### L3-9f669a9a7c  (L3, 2023-11-03, sha 9f669a9a7c2b, PR #1264)
TITLE: Support YaRN models (#1264)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+14/-1); csrc/activation_kernels.cu (+6/-6); csrc/pos_encoding_kernels.cu (+1/-1); vllm/config.py (+3/-0); vllm/model_executor/layers/rotary_embedding.py (+104/-0)
LABELS: new-model
ISSUES: #980 Support YaRN models (RoFormer implementation in rotary_embedding kernel)
BODY: Supersedes https://github.com/vllm-project/vllm/pull/1161, thank you @viktor-ferenczi for laying down the groundwork :) ⏎  ⏎ This PR implements support for YaRN models. ⏎  ⏎ YaRN paper: https://arxiv.org/abs/2309.00071 ⏎ YaRN repository: https://github.com/jquesnelle/yarn ⏎ Smallest model to test with: https://huggingface.co/NousResearch/Yarn-Llama-2-7b-64k ⏎  ⏎ Closes https://github.com/vllm-project/vllm/issues/980

### L3-054072bee5  (L3, 2023-11-12, sha 054072bee534, PR #1633)
TITLE: [Minor] Move RoPE selection logic to `get_rope` (#1633)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+3/-33); vllm/model_executor/layers/rotary_embedding.py (+44/-1)
BODY: 

### L3-5ffc0d13a2  (L3, 2023-11-20, sha 5ffc0d13a2d3, PR #1665)
TITLE: Migrate linter from `pylint` to `ruff` (#1665)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: vllm/model_executor/layers/attention.py (+0/-1); .github/workflows/ruff.yml (+5/-5); .pylintrc (+0/-434); benchmarks/benchmark_throughput.py (+2/-3); format.sh (+8/-8); pyproject.toml (+24/-0); requirements-dev.txt (+1/-1); setup.py (+8/-6); tests/async_engine/api_server_async_engine.py (+0/-1); tests/async_engine/test_api_server.py (+2/-5); (+35 more)
BODY: The ruleset is roughly the same. `ruff` actually implement at lot more other rules that's super useful. ⏎  ⏎ `ruff` yields 300x speedup for linting the files.  ⏎  ⏎ ``` ⏎ (base) xmo@simon-dev-l4x2:~/vllm$ time pylint vllm ⏎ real    0m10.963s ⏎ user    0m40.860s ⏎ sys     0m0.864s ⏎ (base) xmo@simon-dev-l4x2:~/vllm$ time ruff . ⏎  ⏎ real    0m0.036s ⏎ user    0m0.007s ⏎ sys     0m0.052s ⏎ ``` ⏎  ⏎ Currently I have disabled import sorting rules as the changes are  …[truncated]

### L3-819b18e7ba  (L3, 2023-11-20, sha 819b18e7ba7f, PR #1599)
TITLE: Rewrite torch.repeat_interleave to remove cpu synchronization (#1599)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+20/-10)
BODY: 

### L3-e0c6f556e8  (L3, 2023-11-23, sha e0c6f556e850, PR #1624)
TITLE: [Build] Avoid building too many extensions (#1624)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: csrc/attention.cpp (+0/-42); vllm/model_executor/layers/attention.py (+4/-4); benchmarks/kernels/benchmark_paged_attention.py (+3/-3); csrc/activation.cpp (+0/-28); csrc/cache.h (+0/-19); csrc/cuda_utils.cpp (+0/-13); csrc/cuda_utils.h (+5/-0); csrc/layernorm.cpp (+0/-24); csrc/ops.h (+75/-0); csrc/pos_encoding.cpp (+0/-16); (+15 more)
BODY: This PR lets vllm only build one extension and reduces the build time by 75% (from ~5.5min to ~1.5min) based on my test.

### L3-a9e4574261  (L3, 2023-11-29, sha a9e4574261a2, PR #1840)
TITLE: Refactor Attention (#1840)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+191/-361); vllm/model_executor/layers/rotary_embedding.py (+2/-2); vllm/model_executor/models/aquila.py (+14/-10); vllm/model_executor/models/baichuan.py (+16/-16); vllm/model_executor/models/bloom.py (+5/-3); vllm/model_executor/models/chatglm.py (+10/-9); vllm/model_executor/models/falcon.py (+19/-20); vllm/model_executor/models/gpt_j.py (+11/-10); vllm/model_executor/models/gpt_neox.py (+10/-8); vllm/model_executor/models/internlm.py (+10/-8); (+6 more)
BODY: This PR refactors the `PageAttention` module, making the following changes: ⏎  ⏎ 1. The `forward` method is simplified. ⏎ 2. `PagedAttentionWithRoPE` is removed. Now RoPE is explicitly executed before `PagedAttention` in the model code. ⏎ 3.  `PagedAttentionWithALiBi` is removed. ALiBi bias is used if `PagedAttention` is initialized with `alibi_slopes`. ⏎  ⏎ Tested (single-GPU):

### L3-27feead2f8  (L3, 2023-11-29, sha 27feead2f80f, PR #1843)
TITLE: Refactor Worker & InputMetadata (#1843)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+3/-11); vllm/config.py (+6/-0); vllm/engine/arg_utils.py (+4/-3); vllm/engine/llm_engine.py (+0/-2); vllm/model_executor/__init__.py (+2/-0); vllm/model_executor/input_metadata.py (+16/-65); vllm/model_executor/layers/sampler.py (+60/-55); vllm/model_executor/models/aquila.py (+10/-2); vllm/model_executor/models/baichuan.py (+10/-2); vllm/model_executor/models/bloom.py (+10/-2); (+17 more)
BODY: Should be merged after #1840  ⏎  ⏎ This PR refactors worker & input metadata, making the following changes: ⏎  ⏎ 1. `InputMetadata` is split into `InputMetadata` and `SamplingMetadata`. ⏎ 2. Each forward pass consists of two stages: `hidden_states = model.forward(...)` and `next_tokens = model.sample(hidden_states, sampling_metadata)`. This will be useful because we will only handle the former in using CUDA graph or `torch.compile`. ⏎ 3. `_prepare_inpu …[truncated]

### L3-6ccc0bfffb  (L3, 2023-12-07, sha 6ccc0bfffbcf, PR #1836)
TITLE: Merge EmbeddedLLM/vllm-rocm into vLLM main (#1836)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.flash_attn.upstream_pip
FILES: Dockerfile.rocm (+62/-0); csrc/attention/attention_kernels.cu (+21/-13); csrc/attention/attention_utils.cuh (+2/-1); csrc/attention/dtype_bfloat16.cuh (+16/-3); csrc/attention/dtype_float16.cuh (+63/-5); csrc/cache_kernels.cu (+7/-6); requirements-rocm.txt (+17/-0); setup.py (+159/-73); vllm/model_executor/layers/attention.py (+3/-0); .gitignore (+4/-0); (+19 more)
LABELS: rocm
BODY: **Add ROCm- Support** ⏎  ⏎  ⏎ As there are too many changes has been made after https://github.com/vllm-project/vllm/pull/1749 , ⏎ the previous PR https://github.com/vllm-project/vllm/pull/1749 is closed as and continued here. ⏎  ⏎  ⏎ PR Authors: ⏎ @kliuae ⏎ @iAmir97 ⏎ @tjtanaa  ⏎ @tanpinsiang ⏎  ⏎ Contributer: ⏎ @pcmortiz ⏎  ⏎ This pull request also incorporates the work from [Port most vLLM kernels to ROCm #1313](https://github.com/vllm-project/vllm/pull/1313 ) …[truncated]

### L3-dacaf5a400  (L3, 2023-12-10, sha dacaf5a40056, PR #1997)
TITLE: Replace head_mapping params with num_kv_heads to attention kernel. (#1997)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+15/-16); csrc/ops.h (+2/-2); vllm/model_executor/layers/attention.py (+5/-8); benchmarks/kernels/benchmark_paged_attention.py (+2/-6); tests/kernels/test_attention.py (+2/-5)
ISSUES: #1928 Why we need head_mapping as param pass to paged_attention kernel?
BODY: Replace head_mapping params with num_kv_heads to attention kernel. ⏎ Base on this issue: ⏎ [https://github.com/vllm-project/vllm/issues/1928](url) ⏎ To avoid the head_mapping load from global memory.

### L3-6428f1d051  (L3, 2023-12-12, sha 6428f1d051d7, PR #1938)
TITLE: Support MPT with GQA (#1938)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+8/-4); vllm/model_executor/models/mpt.py (+20/-2)
BODY: mpt.py modified to support GQA in MPTAttention

### L3-f375ec8440  (L3, 2023-12-13, sha f375ec844036, PR #2079)
TITLE: [ROCm] Upgrade xformers version for ROCm & update doc (#2079)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile.rocm (+2/-2); requirements-rocm.txt (+0/-1); docs/source/getting_started/amd-installation.rst (+8/-8); patch_xformers.rocm.sh (+17/-6); rocm_patch/commonpy_xformers-0.0.23.rocm.patch (+0/-0); rocm_patch/flashpy_xformers-0.0.23.rocm.patch (+57/-39)
LABELS: rocm
BODY: 

### L3-05bdf4eaf3  (L3, 2023-12-14, sha 05bdf4eaf3bd, PR #2101)
TITLE: Fix Dockerfile.rocm (#2101)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile.rocm (+1/-1)
BODY: Fix Dockerfile.rocm

### L3-37ca558103  (L3, 2023-12-16, sha 37ca55810392, PR #1926)
TITLE: Optimize model execution with CUDA graph (#1926)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+18/-22); benchmarks/benchmark_latency.py (+4/-0); benchmarks/benchmark_throughput.py (+7/-2); requirements.txt (+1/-0); vllm/config.py (+17/-0); vllm/engine/arg_utils.py (+15/-1); vllm/engine/llm_engine.py (+6/-1); vllm/engine/ray_utils.py (+1/-8); vllm/entrypoints/llm.py (+10/-0); vllm/model_executor/input_metadata.py (+4/-1); (+24 more)
BODY: This PR uses CUDA graph to optimize the CPU overheads in model execution. This is particularly effective for small models and when using tensor parallelism. ⏎  ⏎ Many thanks to @scv119 and @Yard1 for addressing the NCCL-related issues.

### L3-77af974b40  (L3, 2024-01-02, sha 77af974b406f, PR #1959)
TITLE: [FIX] Support non-zero CUDA devices in custom kernels (#1959)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape
FILES: csrc/attention/attention_kernels.cu (+3/-0); csrc/cache_kernels.cu (+5/-0); csrc/activation_kernels.cu (+4/-1); csrc/layernorm_kernels.cu (+3/-0); csrc/pos_encoding_kernels.cu (+2/-0); csrc/quantization/squeezellm/quant_cuda_kernel.cu (+2/-1); tests/kernels/conftest.py (+3/-2); tests/kernels/test_activation.py (+13/-3); tests/kernels/test_attention.py (+15/-10); tests/kernels/test_cache.py (+11/-6); (+2 more)
BODY: While assessing the effectiveness of the RMSNorm operator, I observed that executing this operator on non-zero GPU resulted in a 'RuntimeError: CUDA error: an illegal memory access was encountered.' ⏎  Upon further investigation through debugging, I  found that cause as the absence of [device guards](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/awq/gemm_kernels.cu#L514), most cuda kernels have the same issues . ⏎ I have addresse …[truncated]

### L3-28c3f12104  (L3, 2024-01-08, sha 28c3f121040d, PR #2384)
TITLE: [Minor] Remove unused code in attention (#2384)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+9/-14)
BODY: 

### L3-d10f8e1d43  (L3, 2024-01-17, sha d10f8e1d43bf, PR #1669)
TITLE: [Experimental] Prefix Caching Support (#1669)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+55/-33); vllm/model_executor/layers/triton_kernel/prefix_prefill.py (+728/-0); .buildkite/test-pipeline.yaml (+4/-0); examples/offline_inference_with_prefix.py (+51/-0); tests/kernels/test_prefix_prefill.py (+168/-0); tests/prefix_caching/test_prefix_caching.py (+41/-0); tests/samplers/test_sampler.py (+12/-6); tests/worker/test_model_runner.py (+3/-2); vllm/block.py (+4/-0); vllm/core/block_manager.py (+42/-5); (+10 more)
BODY: add prefix caching support ⏎  ⏎ Section 1 (Basic Functionality): ⏎  ⏎  ⏎ Todo: ⏎ Automatic Prefix Caching Support -- [SGLang RadixAttention](https://github.com/sgl-project/sglang)

### L3-7a0b011dd5  (L3, 2024-01-22, sha 7a0b011dd51e, PR #2553)
TITLE: Add a 1-line docstring to explain why calling context_attention_fwd twice in test_prefix_prefill.py (#2553)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_prefix_prefill.py (+1/-0)
BODY: Initially it was confusing to me why we call it twice repeatedly, later found out it was for warming up the triton kernel, just add 1-linear doc string: ⏎  ⏎ Calling it once (warmup): ⏎  ⏎ triton Time: 15.10 ms ⏎ xformers Time: 0.61 ms ⏎  ⏎ Calling it twice (after warmup): ⏎ triton Time: 1.95 ms ⏎ xformers Time: 0.62 ms

### L3-5265631d15  (L3, 2024-01-25, sha 5265631d15d5, PR #2583)
TITLE: use a correct device when creating OptionalCUDAGuard (#2583)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+1/-1)
BODY: fix of https://github.com/vllm-project/vllm/issues/2350

### L3-6b7de1a030  (L3, 2024-01-26, sha 6b7de1a030e5, PR #2274)
TITLE: [ROCm] add support to ROCm 6.0 and MI300 (#2274)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile.rocm (+31/-5); setup.py (+2/-0); README.md (+2/-1); csrc/cuda_utils.h (+3/-0); csrc/cuda_utils_kernels.cu (+18/-0); csrc/pybind.cpp (+6/-0); docs/source/getting_started/amd-installation.rst (+30/-3); vllm/utils.py (+4/-4)
LABELS: rocm
BODY: This diff adds support to ROCm 6.0 and MI300

### L3-9090bf02e7  (L3, 2024-01-28, sha 9090bf02e743, PR #2279)
TITLE: Support FP8-E5M2 KV Cache (#2279)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.flash_attn.upstream_pip
FILES: csrc/attention/attention_dtypes.h (+1/-0); csrc/attention/attention_kernels.cu (+164/-95); csrc/attention/dtype_fp8_e5m2.cuh (+35/-0); csrc/cache_kernels.cu (+108/-27); vllm/model_executor/layers/attention.py (+3/-0); benchmarks/benchmark_latency.py (+8/-0); benchmarks/benchmark_throughput.py (+11/-1); benchmarks/kernels/benchmark_paged_attention.py (+18/-15); csrc/cache.h (+7/-1); csrc/dispatch_utils.h (+10/-0); (+16 more)
BODY: Quantize KV Cache to fp8 can reduce memory usage of kv cache and then could boost throughput. The impl uses fp8 data type for kv cache and has been tested on A100. ⏎  ⏎ The following test is under WarzardCoder-34B. ⏎  ⏎ Dataset | Baseline(KV Cache FP16) | KV Cache FP8 E5M2 | KV Cache FP8 E4M3 ⏎ -- | -- | -- | -- ⏎ HumanEval-Python-EN | 68.293% | 65.854% (↓ 2.439%) | 67.683% (↓ 0.61%) ⏎ HumanEval-Python-CN | 59.146% | 59.146% (=) | 59.756% (↑ 0.61%) ⏎  ⏎ LLaMA-7B | …[truncated]

### L3-923797fea4  (L3, 2024-02-01, sha 923797fea4d8, PR #2648)
TITLE: Fix compile error when using rocm (#2648)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape
FILES: csrc/attention/attention_kernels.cu (+2/-0); csrc/cache_kernels.cu (+7/-0); csrc/quantization/fp8_e5m2_kvcache/quant_utils.cuh (+0/-1)
LABELS: rocm
ISSUES: #2646 [BUG] Compile source code error for ROCM6.0
BODY: fix #2646

### L3-96b6f475dd  (L3, 2024-02-01, sha 96b6f475dda4, PR #2503)
TITLE: Remove hardcoded `device="cuda" ` to support more devices (#2503)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+1/-1); benchmarks/benchmark_latency.py (+7/-0); benchmarks/benchmark_throughput.py (+9/-1); benchmarks/kernels/benchmark_paged_attention.py (+18/-9); tests/kernels/test_activation.py (+21/-16); tests/kernels/test_attention.py (+23/-28); tests/kernels/test_cache.py (+21/-21); tests/kernels/test_layernorm.py (+10/-7); tests/kernels/test_pos_encoding.py (+11/-11); tests/kernels/test_prefix_prefill.py (+20/-37); (+22 more)
BODY: Refer to #1948 , there are a lot of code use `cuda` as device, especially in tensor creation, which is not friendly to add other device support. This PR aims to refactor the code to leave some interface for better and easily add new device like `cpu` or `xpu`

### L3-0580aab02f  (L3, 2024-02-10, sha 0580aab02ffe, PR #2768)
TITLE: [ROCm] support Radeon™ 7900 series (gfx1100) without using flash-attention (#2768)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile.rocm (+12/-3); setup.py (+1/-1); vllm/model_executor/layers/attention.py (+45/-0); docs/source/getting_started/amd-installation.rst (+2/-1)
LABELS: rocm
BODY: This pull request adds vllm support for AMD Radeon™ 7900 series GPU (gfx1100) without using flash-attention. ⏎ Currently, flash-attention does not fully support gfx1100. So, we used vllm reference implementation instead. ⏎  ⏎ Note: ⏎ When building the docker image, pass `--build-arg BUILD_FA="0"` to the `docker build` command.

### L3-5255d99dc5  (L3, 2024-02-15, sha 5255d99dc595, PR #2885)
TITLE: [ROCm] Dockerfile fix for flash-attention build (#2885)
SOURCES: subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile.rocm (+3/-3)
BODY: This pull request fixes Dockerfile for flash-attention build on ROCm.

### L3-64da65b322  (L3, 2024-02-16, sha 64da65b3225b, PR #2517)
TITLE: Prefix Caching- fix t4 triton error (#2517)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/triton_kernel/prefix_prefill.py (+3/-1)
ISSUES: #2513 prefix caching error with baichuan model
BODY: Fix #2513, need a smaller block size for Turing GPUs

### L3-93dc5a2870  (L3, 2024-02-21, sha 93dc5a287086, PR #2820)
TITLE: chore(vllm): codespell for spell checking  (#2820)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/triton_kernel/prefix_prefill.py (+1/-1); .github/workflows/ruff.yml (+4/-1); benchmarks/benchmark_serving.py (+1/-1); format.sh (+48/-3); mypy.ini (+0/-8); pyproject.toml (+18/-0); requirements-dev.txt (+2/-0); tests/lora/test_layers.py (+1/-1); tests/lora/test_llama.py (+2/-2); vllm/core/block_manager.py (+1/-1); (+6 more)
BODY: 

### L3-6f32cddf1c  (L3, 2024-02-22, sha 6f32cddf1c79, PR #2982)
TITLE: Remove Flash Attention in test env (#2982)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: requirements-dev.txt (+1/-2)
BODY: To my understanding, we don't need `flash_attn` for our testing. Removing it from dependencies will resolve the occasional test failures due to `flash_attn`.

### L3-d6e4a130b0  (L3, 2024-02-26, sha d6e4a130b028, PR #3043)
TITLE: [Minor] Remove gather_cached_kv kernel (#3043)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+0/-161); csrc/cache.h (+0/-7); csrc/pybind.cpp (+0/-4)
BODY: The `gather_cached_kv` kernel was originally implemented for prefix sharing and is not currently used. I believe we can remove the kernel.

### L3-71bcaf99e2  (L3, 2024-02-27, sha 71bcaf99e2cb, PR #3007)
TITLE: Enable GQA support in the prefix prefill kernels (#3007)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention.py (+18/-16); vllm/model_executor/layers/triton_kernel/prefix_prefill.py (+27/-12); tests/kernels/test_prefix_prefill.py (+42/-19)
BODY: 

### L3-27a7b070db  (L3, 2024-03-04, sha 27a7b070db52, PR #2978)
TITLE: Add document for vllm paged attention kernel. (#2978)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docs/source/assets/kernel/k_vecs.png (+0/-0); docs/source/assets/kernel/key.png (+0/-0); docs/source/assets/kernel/logits_vec.png (+0/-0); docs/source/assets/kernel/q_vecs.png (+0/-0); docs/source/assets/kernel/query.png (+0/-0); docs/source/assets/kernel/v_vec.png (+0/-0); docs/source/assets/kernel/value.png (+0/-0); docs/source/dev/kernel/paged_attention.rst (+525/-0); docs/source/index.rst (+1/-0)
LABELS: documentation
BODY: Hello, I am currently studying the vLLM paged attention kernel, and I've found that the implementation can be quite complex for newcomers. After thoroughly reviewing the primary implementation of the kernel in `csrc/attention/attention_kernels.cu`, I have composed this document to provide a high-level understanding of the paged attention kernel. The document covers explanations on memory layout, read patterns, and step-by-step calculations, accom …[truncated]

### L3-05af6da8d9  (L3, 2024-03-04, sha 05af6da8d927, PR #3123)
TITLE: [ROCm] enable cupy in order to enable  cudagraph mode for AMD GPUs (#3123)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile.rocm (+25/-5); vllm/worker/worker.py (+1/-3)
LABELS: rocm
BODY: This pull request enables cupy for ROCm backend in order to run throughput benchmarking script in cudagraph (hipgraph) mode successfully. ⏎  ⏎ **[Reason]**: ⏎ vllm currently needs cupy in order to run cudagraph mode successfully, as mentioned in this [comment](https://github.com/vllm-project/vllm/blob/22de45235c6dd14e901e089971635ec655d5fbe0/vllm/model_executor/parallel_utils/cupy_utils.py#L3)  ⏎  ⏎ ``` ⏎ We use CuPy all-reduce instead of torch.distrib …[truncated]

### L3-2daf23ab0c  (L3, 2024-03-07, sha 2daf23ab0cf0, PR #3005)
TITLE: Separate attention backends (#3005)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.upstream_pip, L3.triton.prefix_prefill
FILES: setup.py (+45/-3); vllm/model_executor/layers/attention/__init__.py (+5/-0); vllm/model_executor/layers/attention/attention.py (+59/-0); vllm/model_executor/layers/attention/backends/__init__.py (+0/-0); vllm/model_executor/layers/attention/backends/flash_attn.py (+124/-0); vllm/model_executor/layers/attention/backends/xformers.py (+61/-155); vllm/model_executor/layers/attention/ops/__init__.py (+0/-0); vllm/model_executor/layers/attention/ops/paged_attn.py (+138/-0); vllm/model_executor/layers/attention/ops/prefix_prefill.py (+0/-0); vllm/model_executor/models/deepseek.py (+5/-5); (+25 more)
BODY: This PR refactors the attention layer. Specifically, it separates the code paths for Ampere or more recent NVIDIA GPUs (which can directly use FlashAttention) and other GPUs, so that the code for the former becomes much simpler. This PR will also bring some performance improvements for ALiBi models, since we now directly call FlashAttention instead of using xformers in the middle.

### L3-1cb0cc2975  (L3, 2024-03-08, sha 1cb0cc2975d1, PR #3269)
TITLE: [FIX] Make `flash_attn` optional (#3269)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.upstream_pip
FILES: setup.py (+3/-45); vllm/model_executor/layers/attention/attention.py (+31/-6); vllm/model_executor/layers/attention/backends/flash_attn.py (+0/-1); .gitignore (+0/-3); vllm/__init__.py (+7/-23)
BODY: The FlashAttention backend introduced in #3005 causes build errors in some environments and increase the package size significantly (44 MB -> 160 MB) as the vLLM package now includes the `flash-attn` (116 MB) package. This PR addresses this by removing `flash_attn` from vLLM's dependency and enables falling back to xFormers when `flash_attn` is not found.

### L3-f48c6791b7  (L3, 2024-03-08, sha f48c6791b7bf, PR #3286)
TITLE: [FIX] Fix prefix test error on main (#3286)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/model_executor/layers/attention/backends/flash_attn.py (+0/-2)
BODY: 

### L3-e4a28e5316  (L3, 2024-03-10, sha e4a28e531659, PR #3262)
TITLE: [ROCM] Fix blockReduceSum to use correct warp counts for ROCm and CUDA (#3262)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cu (+0/-8); csrc/cuda_compat.h (+10/-0); csrc/reduction_utils.cuh (+3/-3)
BODY: blockReduceSum was defaulting to 32 for warp size regardless of the architecture. ⏎  ⏎ Bonus, refactor cuda_compat.h to hold WARP_SIZE define instead of the attention_kernels.cuh

### L3-2f8844ba08  (L3, 2024-03-10, sha 2f8844ba08d7, PR #3305)
TITLE: Re-enable the 80 char line width limit (#3305)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: vllm/model_executor/layers/attention/attention.py (+2/-2); pyproject.toml (+4/-2); setup.py (+2/-2); tests/async_engine/test_chat_template.py (+4/-2); tests/core/test_block_manager.py (+2/-1); tests/entrypoints/test_guided_processors.py (+2/-2); tests/entrypoints/test_openai_server.py (+23/-13); tests/kernels/test_moe.py (+2/-1); tests/kernels/test_prefix_prefill.py (+2/-1); tests/lora/test_layer_variation.py (+3/-3); (+57 more)
BODY: The current linter does not check for line width. Let's add this check back again.

### L3-06ec486794  (L3, 2024-03-14, sha 06ec486794f4, PR #3396)
TITLE: Install `flash_attn` in Docker image (#3396)
SOURCES: subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile (+24/-0)
BODY: Recent fix #3269 removed `flash_attn` as an explicit dependency since it was breaking builds in a bunch of environments and made the wheel size larger.  ⏎  ⏎ This PR leaves the wheel unchanged, but installs `flash_attn` independently within the Docker build (for both the test as well as the runtime image). This will allow us to use the `FlashAttentionBackend` in containerized environments without affecting those using the package in other ways.

### L3-6e435de766  (L3, 2024-03-20, sha 6e435de766c7, PR #3236)
TITLE: [1/n][Chunked Prefill] Refactor input query shapes (#3236)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend
FILES: vllm/model_executor/layers/attention/attention.py (+2/-1); vllm/model_executor/layers/attention/backends/flash_attn.py (+32/-14); vllm/model_executor/layers/attention/backends/xformers.py (+147/-85); vllm/model_executor/layers/attention/ops/paged_attn.py (+5/-4); .buildkite/test-pipeline.yaml (+2/-2); tests/basic_correctness/test_basic_correctness.py (+3/-1); tests/core/test_scheduler.py (+9/-9); tests/lora/test_worker.py (+1/-1); tests/spec_decode/test_multi_step_worker.py (+2/-2); tests/worker/test_model_runner.py (+153/-8); (+8 more)
BODY: It is the first PR to address https://github.com/vllm-project/vllm/issues/3130 ⏎  ⏎ The current query format is not suitable for chunked prefill because after it is enabled, chunked prefill (e.g., size of 764) and decoding requests will be batched together. If we use 2D query (batch_size, seq_len), we should either use hacky solution (treating the last batch as a batch of decoding requests) or have inefficient # of paddings. ⏎  ⏎ To get around this,  …[truncated]

### L3-925f3332ca  (L3, 2024-03-25, sha 925f3332cac4, PR #3462)
TITLE: [Core] Refactor Attention Take 2 (#3462)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.triton.prefix_prefill, L3.dispatch.selector, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/__init__.py (+0/-0); vllm/attention/backends/abstract.py (+85/-0); vllm/attention/backends/flash_attn.py (+238/-0); vllm/attention/backends/xformers.py (+177/-55); vllm/attention/layer.py (+46/-0); vllm/attention/ops/__init__.py (+0/-0); vllm/attention/ops/paged_attn.py (+217/-0); vllm/attention/ops/prefix_prefill.py (+0/-0); vllm/attention/selector.py (+44/-0); vllm/model_executor/layers/attention/__init__.py (+0/-5); (+37 more)
BODY: This PR is the second attempt to modularize the attention backends. The main goal of this PR is to hide any backend-specific attention implementation details from the main logic. This refactoring will greatly help introduce new backends, particularly the FlashInfer backend, which requires a different KV cache layout and input data structures from our current attention backends. **NOTE: Since this PR just re-organizes the code, it shouldn't affect …[truncated]

### L3-01bfb22b41  (L3, 2024-03-25, sha 01bfb22b4112, PR #3495)
TITLE: [CI] Try introducing isort.  (#3495)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.upstream_pip, L3.dispatch.selector
FILES: .github/workflows/ruff.yml (+5/-2); benchmarks/benchmark_prefix_caching.py (+1/-2); benchmarks/benchmark_serving.py (+3/-6); benchmarks/benchmark_throughput.py (+1/-1); benchmarks/kernels/benchmark_mixtral_moe.py (+3/-1); benchmarks/kernels/benchmark_paged_attention.py (+2/-2); benchmarks/kernels/benchmark_rope.py (+4/-3); cmake/hipify.py (+1/-1); collect_env.py (+1/-1); docs/source/conf.py (+2/-1); (+134 more)
BODY: It is WIP, and Idk if committers agree on introducing this feature yet.  ⏎  ⏎ Seems like vllm oss has sorting rule that's implicit. This PR tries introducing isort to `format.sh`.  ⏎  ⏎ This PR only touches small # of files to avoid huge merge conflict.  ⏎  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-395aa823ea  (L3, 2024-03-28, sha 395aa823ea45, PR #3716)
TITLE: [Misc] Minor type annotation fix (#3716)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/selector.py (+2/-1)
BODY: 

### L3-4716a32dd4  (L3, 2024-03-28, sha 4716a32dd4b4, PR #3701)
TITLE: fix logging msg for block manager (#3701)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/selector.py (+3/-1); vllm/core/block_manager_v1.py (+1/-2); vllm/model_executor/parallel_utils/pynccl_utils.py (+1/-1)
BODY: Showing  ⏎  ⏎ ``` ⏎ INFO 03-28 21:39:58 block_manager_v1.py:239] disable automatic prefix caching ⏎ ``` ⏎  ⏎ Is not good UX. the message for feature flags should be informative.

### L3-9765b5c406  (L3, 2024-03-29, sha 9765b5c4061b, PR #3699)
TITLE: [ROCm][Bugfix] Fixed several bugs related to rccl path and attention selector logic (#3699)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.upstream_pip
FILES: requirements-rocm.txt (+1/-1); vllm/attention/backends/xformers.py (+2/-2); Dockerfile.rocm (+1/-1); vllm/model_executor/parallel_utils/pynccl.py (+1/-1)
LABELS: rocm
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ This pull request fixes several bugs introduced in previous commits, for example: https://github.com/vllm-project/vllm/pull/3661, https://github.com/vllm-project/vllm/pull/3625 , and previous refactoring in attention backend. ⏎  ⏎ (1) Fixed the librccl.so file name, it should be something like: ⏎ /opt/rocm/lib/librccl.so.1 ⏎  ⏎ (2) a bug related to check whet …[truncated]

### L3-0e3f06fe9c  (L3, 2024-04-01, sha 0e3f06fe9ccf, PR #3634)
TITLE: [Hardware][Intel] Add CPU inference backend (#3634)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake, L3.dispatch.selector
FILES: csrc/cpu/attention.cpp (+744/-0); vllm/attention/backends/torch_sdpa.py (+253/-0); vllm/attention/selector.py (+7/-1); .buildkite/run-cpu-test.sh (+14/-0); .buildkite/test-template.j2 (+3/-0); CMakeLists.txt (+16/-0); Dockerfile.cpu (+20/-0); cmake/cpu_extension.cmake (+90/-0); csrc/cpu/activation.cpp (+148/-0); csrc/cpu/cache.cpp (+139/-0); (+14 more)
BODY: This PR adds a new CPU backend to vLLM and supports the basic model inference feature, with BF16 and FP32 dtype. FP16 support and TP support will be added in the future. ⏎  ⏎ Changes to vLLM: ⏎  ⏎ - Added ```VLLM_TARGET_DEVICE``` ENV to specify backend explicitily. ⏎ - Added ```CPUExecutor``` to isolate CPU backend with others. ⏎ - Added ```TorchSDPABackend``` to support MHA on CPU. ⏎ - Added ```_C``` related kernels on CPU. ⏎ - Forwarded ```DeviceConfig …[truncated]

### L3-2ff767b513  (L3, 2024-04-03, sha 2ff767b51301, PR #3290)
TITLE: Enable scaled FP8 (e4m3fn) KV cache on ROCm (AMD GPU) (#3290)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.fork_inline_cmake, L3.dispatch.abstract_interface
FILES: csrc/attention/attention_dtypes.h (+1/-1); csrc/attention/attention_kernels.cu (+79/-44); csrc/attention/dtype_fp8.cuh (+1/-1); csrc/cache_kernels.cu (+42/-23); vllm/attention/backends/abstract.py (+1/-0); vllm/attention/backends/flash_attn.py (+7/-1); vllm/attention/backends/xformers.py (+7/-1); vllm/attention/layer.py (+3/-1); vllm/attention/ops/paged_attn.py (+5/-0); .gitignore (+1/-0); (+31 more)
LABELS: rocm
BODY: As part of a series of FP8 development in vLLM, we address an [OCP](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1) format (nVIDIA compatible) FP8 KV cache in this pull request. We elaborated upon previous [#2279](https://github.com/vllm-project/vllm/pull/2279), but made following change, enhancement and extensions: ⏎ - Using OCP FP8 data type, E4M3 recommended for inference (as Floa …[truncated]

### L3-498eb5cfa3  (L3, 2024-04-04, sha 498eb5cfa3b4, PR #3840)
TITLE: [Bugfix] Add kv_scale input parameter to CPU backend (#3840)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.python_wrapper
FILES: csrc/cpu/attention.cpp (+4/-2); vllm/attention/backends/torch_sdpa.py (+4/-1); vllm/attention/ops/paged_attn.py (+1/-1); csrc/cpu/cache.cpp (+3/-1)
BODY: This PR fixes the broken CPU CI by adding the `kv_scale` parameter to the attention-related kernels for CPUs.

### L3-d03d64fd2e  (L3, 2024-04-04, sha d03d64fd2e22, PR #)
TITLE: [CI/Build] refactor dockerfile & fix pip cache
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L3-cfaf49a167  (L3, 2024-04-05, sha cfaf49a1673c, PR #3841)
TITLE: [Misc] Define common requirements (#3841)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_pip
FILES: Dockerfile (+5/-3); requirements-common.txt (+3/-9); requirements-cpu.txt (+6/-15); requirements-cuda.txt (+10/-0); requirements-neuron.txt (+4/-9); requirements-rocm.txt (+4/-17); setup.py (+26/-20); .github/workflows/publish.yml (+1/-1); .github/workflows/scripts/build.sh (+1/-1); CONTRIBUTING.md (+0/-1); (+1 more)
BODY: This PR factors out the common dependencies to `requirements-common.txt` to better manage the dependencies across different backends.

### L3-6c0b04515f  (L3, 2024-04-09, sha 6c0b04515fee, PR #3643)
TITLE: [ROCm][Hardware][AMD] Use Triton Kernel for default FA on ROCm (#3643)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.upstream_pip, L3.triton.flash_attention_rocm, L3.rocm.rocm_flash_attn_v0, L3.dispatch.selector
FILES: vllm/attention/backends/rocm_flash_attn.py (+348/-0); vllm/attention/backends/xformers.py (+2/-76); vllm/attention/ops/triton_flash_attention.py (+809/-0); vllm/attention/selector.py (+40/-17); Dockerfile.rocm (+14/-0)
LABELS: rocm
BODY: This PR creates and makes default new triton exclusive backend for attention. Additionally removes some unsupported arguments from AMD's version of the `flash_attn_varlen_func` function. ⏎  ⏎ - Changed selector to allow picking between multiple backends rather than just between two. ⏎ - Added a new triton FA backend available to ROCm.  ⏎ - Added new `VLLM_USE_FLASH_ATTN_TRITON` option to be able to swap between Triton and Default FA ⏎ - Removed unsupp …[truncated]

### L3-8b317c6dd0  (L3, 2024-04-10, sha 8b317c6dd09c, PR #3972)
TITLE: [Model][AMD] ROCm support for 256 head dims for Gemma (#3972)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.flash_attention_rocm
FILES: vllm/attention/ops/triton_flash_attention.py (+2/-3)
LABELS: rocm
ISSUES: #3073 Serving for Google Gemma model failing on AMD MI 300X GPUs
BODY: Thanks to @jpvillam-amd's contributions on https://github.com/vllm-project/vllm/pull/3643, Gemma should have ROCm support ⏎  ⏎ In my testing, I had to make these additional but trivial tweaks to actually use google/gemma-2b-it ⏎  ⏎ This was tested on an MI100 ⏎  ⏎ FIX #3073 (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-67b4221a61  (L3, 2024-04-10, sha 67b4221a61ac, PR #3884)
TITLE: [Core][5/N] Fully working chunked prefill e2e (#3884)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+39/-3); vllm/attention/backends/flash_attn.py (+55/-30); vllm/attention/backends/rocm_flash_attn.py (+63/-34); vllm/attention/backends/torch_sdpa.py (+42/-25); vllm/attention/backends/xformers.py (+79/-59); vllm/attention/layer.py (+3/-2); vllm/attention/ops/paged_attn.py (+0/-6); .buildkite/test-pipeline.yaml (+2/-0); benchmarks/benchmark_latency.py (+1/-2); benchmarks/benchmark_throughput.py (+38/-24); (+16 more)
BODY: This PR is a part of the RFC https://github.com/vllm-project/vllm/issues/3130. ⏎  ⏎ This PR enables chunked prefill e2e. Note that chunked prefill is an experimental feature now (though it is actively used within Anyscale), and I will start serious benchmark with this PR.  ⏎  ⏎ The feature can be enabled by using `enable_chunked_prefill`. The chunking is done based on `max_num_batched_tokens` (a.k.a token budget). This is the same way as described in …[truncated]

### L3-e9da5a40c6  (L3, 2024-04-10, sha e9da5a40c63c, PR #3913)
TITLE: [Misc] Add indirection layer for custom ops  (#3913)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.paged.python_wrapper
FILES: vllm/attention/ops/paged_attn.py (+5/-5); benchmarks/kernels/benchmark_paged_attention.py (+1/-1); tests/kernels/test_attention.py (+3/-3); tests/kernels/test_cache.py (+12/-13); vllm/_custom_ops.py (+193/-0); vllm/model_executor/layers/activation.py (+1/-1); vllm/model_executor/layers/fused_moe/fused_moe.py (+1/-1); vllm/model_executor/layers/layernorm.py (+1/-1); vllm/model_executor/layers/quantization/awq.py (+1/-1); vllm/model_executor/layers/quantization/gptq.py (+1/-1); (+4 more)
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ Refactor `ops` and `cache_ops` layer. Add an abstraction ops layer and will use `vllm._C.ops` by default. ⏎ This would be easier to add/extend other third party high performance ops/kernels implementation if necessary. ⏎  ⏎  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-e42df7227d  (L3, 2024-04-11, sha e42df7227d18, PR #3961)
TITLE: [Test] Add xformer and flash attn tests (#3961)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/selector.py (+9/-0); tests/basic_correctness/test_basic_correctness.py (+6/-0)
BODY: Currently, we only test flash attn implementation in the master. We should make sure to test xformer attention works e2e. ⏎  ⏎ Ideally, we should run model-only test, but there's no way to do this now, so I instead added e2e tests with basic_correctness.  ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-8afca50889  (L3, 2024-04-11, sha 8afca50889ba, PR #3824)
TITLE: [Hardware][Intel] Isolate CPUModelRunner and ModelRunner for better maintenance (#3824)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/torch_sdpa.py (+24/-48); vllm/executor/cpu_executor.py (+10/-0); vllm/utils.py (+0/-1); vllm/worker/cpu_model_runner.py (+408/-0); vllm/worker/cpu_worker.py (+1/-12)
ISSUES: #3776 [Misc]: Isolate CPUModelRunner and ModelRunner for better maintenance
BODY: Fix #3776  ⏎  ⏎ This PR created a new ```CPUModelRunner``` from ```ModelRunner``` for better maintenance. ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-36729bac13  (L3, 2024-04-12, sha 36729bac1303, PR #4023)
TITLE: [Test] Test multiple attn backend for chunked prefill.  (#4023)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+6/-12); .buildkite/test-pipeline.yaml (+7/-1); tests/basic_correctness/test_basic_correctness.py (+0/-6); tests/basic_correctness/test_chunked_prefill.py (+0/-4)
BODY: Add multi attn backend test for chunked prefill. ⏎  ⏎ I also found the previous approach didin't work because of lru_cache usage https://github.com/vllm-project/vllm/blob/c2b4a1bce9a7707179cdfab2fb498c20b2b221e6/vllm/attention/selector.py#L24 ⏎  ⏎ So I instead setting the env var from pipeline.yaml ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-d04973ad54  (L3, 2024-04-12, sha d04973ad5446, PR #3984)
TITLE: Fix triton compilation issue (#3984)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.flash_attention_rocm
FILES: vllm/attention/ops/triton_flash_attention.py (+5/-1)
LABELS: rocm
BODY: Solves the following compilation error with triton flash attention backend enabled. ⏎  ⏎ .../vllm/attention/ops/triton_flash_attention.py: ⏎  ⏎ ... ⏎ triton.compiler.errors.UnsupportedLanguageConstruct: at 120:14:            #          + offs_m ⏎             # We store inf to LSE, not -inf because in the bwd pass, ⏎             # we subtract this ⏎             # from qk which makes it -inf, such that exp(qk - inf) = 0 ⏎             # for these masked bloc …[truncated]

### L3-e8cc7967ff  (L3, 2024-04-18, sha e8cc7967ff8a, PR #4128)
TITLE: [Bugfix][Kernel] allow non-power-of-two head sizes in prefix prefill (#4128)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/prefix_prefill.py (+27/-17); tests/kernels/test_prefix_prefill.py (+1/-1)
ISSUES: #4127 [Bug][Chunked prefill]: head size has to be power of two
BODY: The existing prefix prefill kernel only supports head dimension that is a power of two. This due to Triton only supporting power of two block sizes. This PR enlarges the Q,K,V tensors to the next power of two and pads them with zeros when reading (and writing). ⏎  ⏎ It doesn't seem to affect performance of the non-padded case. ⏎  ⏎ CC @rkooo567  ⏎  ⏎ FIX #4127  ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-95e5b087cf  (L3, 2024-04-21, sha 95e5b087cfed, PR #4129)
TITLE: [AMD][Hardware][Misc][Bugfix] xformer cleanup and light navi logic and CI fixes and refactoring (#4129)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.rocm.rocm_flash_attn_v0
FILES: Dockerfile.rocm (+1/-4); vllm/attention/backends/rocm_flash_attn.py (+18/-13); .buildkite/test-pipeline.yaml (+0/-2); patch_xformers.rocm.sh (+0/-33); rocm_patch/commonpy_xformers-0.0.23.rocm.patch (+0/-13); rocm_patch/flashpy_xformers-0.0.23.rocm.patch (+0/-152)
LABELS: rocm
BODY: This PR is  ⏎ (1) to remove xformer package and patches since right now raw flash attention api is called instead of using xformers wrapper. ⏎ (2) to facilitate gfx1100/navi3x to use triton flash-attn. ⏎ (3) still make it possible for other gfx target to use flash-attn and updated the flash-attention branch. ⏎  ⏎ (4) fixed the CI failure for "basic correctness" issue on cuda env ⏎  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SU …[truncated]

### L3-0ae11f78ab  (L3, 2024-04-22, sha 0ae11f78ab89, PR #4161)
TITLE: [Mypy] Part 3 fix typing for nested directories for most of directory (#4161)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+1/-1); vllm/attention/backends/rocm_flash_attn.py (+1/-0); vllm/attention/backends/torch_sdpa.py (+2/-1); vllm/attention/backends/xformers.py (+1/-0); .github/workflows/mypy.yaml (+15/-14); format.sh (+12/-14); pyproject.toml (+4/-2); vllm/core/block/block_table.py (+1/-0); vllm/core/block/common.py (+4/-2); vllm/core/block/interfaces.py (+2/-4); (+19 more)
BODY: This fixes typing for most of the nested directories.  ⏎  ⏎ The remaining one has a lot of changes required for each of them.  ⏎  ⏎ Also move --follow-imports to the pyproject.toml ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-b6dcb4d442  (L3, 2024-04-25, sha b6dcb4d44281, PR #4368)
TITLE: [Misc] Fix flash attention backend log  (#4368)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/selector.py (+5/-5)
ISSUES: #4246 [Feature]: Cannot use FlashAttention backend for Volta and Turing GPUs. (but FlashAttention v1.0.9 supports Turing GPU.)
BODY: It’s better to indicate the FlashAttention version to use. As `FlashAttentionBackend` especially uses version 2, and `XFormerBackend` will auto select best backend(including `FlashAttention v1`). ⏎  ⏎ FIX #4246  ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-18d23f642a  (L3, 2024-04-26, sha 18d23f642af9, PR #4406)
TITLE: [ROCm][Hardware][AMD] Enable group query attention for triton FA (#4406)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.flash_attention_rocm, L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+25/-28); vllm/attention/ops/triton_flash_attention.py (+11/-13)
LABELS: rocm
BODY: Enable group-query-attention for Triton flash attention ⏎  ⏎ cc Vinayak Gokhale @vgokhale , @Alexei-V-Ivanov-AMD  ⏎  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-d627a3d837  (L3, 2024-04-29, sha d627a3d83797, PR #4454)
TITLE: [Misc] Upgrade to `torch==2.3.0` (#4454)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1); Dockerfile (+1/-1); requirements-build.txt (+1/-1); requirements-cpu.txt (+1/-1); requirements-cuda.txt (+2/-2); .github/workflows/publish.yml (+1/-1); pyproject.toml (+1/-1)
BODY: Now that there is an [xformers release](https://github.com/facebookresearch/xformers/releases/tag/v0.0.26.post1) and [flash-attn release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.8) with PyTorch 2.3.0 support, we can consider upgrading to PyTorch 2.3.0. ⏎  ⏎ PyTorch 2.3.0 is an useful version because it allows using `torch._scaled_mm` on Ada Lovelace GPUs (CUDA compute capability 8.9) for running native FP8 matmuls. (We will e …[truncated]

### L3-d6f4bd7cdd  (L3, 2024-04-30, sha d6f4bd7cddc9, PR #4132)
TITLE: [Misc]Add customized information for models (#4132)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+7/-0); tests/models/test_big_models.py (+15/-0); tests/models/test_models.py (+15/-0); vllm/model_executor/layers/activation.py (+3/-0); vllm/model_executor/layers/layernorm.py (+5/-0); vllm/model_executor/layers/linear.py (+22/-0); vllm/model_executor/layers/logits_processor.py (+6/-0); vllm/model_executor/layers/rotary_embedding.py (+6/-0); vllm/model_executor/layers/vocab_parallel_embedding.py (+8/-0)
BODY: When I debug the VLLM, the model's print output always bothers me because it lacks details, as shown below: ⏎ ```python ⏎ LlavaForConditionalGeneration( ⏎   (vision_tower): CLIPVisionModel( ⏎     (vision_model): CLIPVisionTransformer( ⏎       (embeddings): CLIPVisionEmbeddings( ⏎         (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False) ⏎         (position_embedding): Embedding(577, 1024) ⏎       ) ⏎       (pre_layrnorm …[truncated]

### L3-5b8a7c1cb0  (L3, 2024-05-02, sha 5b8a7c1cb0f1, PR #4548)
TITLE: [Misc] centralize all usage of environment variables (#4548)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.rocm.rocm_flash_attn_v0, L3.dispatch.selector
FILES: vllm/attention/backends/rocm_flash_attn.py (+2/-3); vllm/attention/selector.py (+2/-4); vllm/config.py (+0/-5); vllm/distributed/device_communicators/custom_all_reduce.py (+4/-4); vllm/distributed/parallel_state.py (+2/-2); vllm/distributed/utils.py (+5/-2); vllm/engine/async_llm_engine.py (+2/-3); vllm/entrypoints/openai/api_server.py (+2/-2); vllm/envs.py (+160/-0); vllm/executor/cpu_executor.py (+2/-3); (+8 more)
BODY: An implementation of RFC https://github.com/vllm-project/vllm/issues/4407 . ⏎  ⏎ Exceptions are: ⏎  ⏎ - Test code related environment variables, leave as they are ⏎ - Modify/write to environment variables,  leave as they are ⏎ - environment variables used in `setup.py`, when we cannot import `vllm/envs.py` . leave as they are

### L3-32881f3f31  (L3, 2024-05-02, sha 32881f3f3106, PR #4405)
TITLE: [kernel] fix sliding window in prefix prefill Triton kernel (#4405)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.triton.prefix_prefill, L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/flash_attn.py (+1/-0); vllm/attention/backends/rocm_flash_attn.py (+1/-0); vllm/attention/backends/xformers.py (+1/-0); vllm/attention/ops/paged_attn.py (+2/-0); vllm/attention/ops/prefix_prefill.py (+56/-19); tests/kernels/test_prefix_prefill.py (+30/-4)
ISSUES: #4057 [Feature][Chunked prefill]: Make sliding window work
BODY: This adds support for the sliding window in prefix prefill kernel. ⏎  ⏎ I had to use a large negative value instead of -inf for masking, since otherwise in some situations we get '-inf - -inf' in softmax which leads to NaNs. ⏎  ⏎ Added tests comparing with xformers. ⏎  ⏎ Also added a bunch of comments with tensor shapes etc. ⏎  ⏎ FIX #4057 ⏎  ⏎ CC @rkooo567 @cadedaniel @simon-mo  ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCR …[truncated]

### L3-3521ba4f25  (L3, 2024-05-03, sha 3521ba4f2554, PR #4518)
TITLE: [Core][Model runner refactoring 1/N] Refactor attn metadata term (#4518)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.rocm.rocm_flash_attn_v0
FILES: csrc/attention/attention_kernels.cu (+38/-38); csrc/cpu/attention.cpp (+46/-46); vllm/attention/backends/flash_attn.py (+22/-22); vllm/attention/backends/rocm_flash_attn.py (+30/-30); vllm/attention/backends/torch_sdpa.py (+18/-18); vllm/attention/backends/xformers.py (+32/-33); vllm/attention/ops/paged_attn.py (+17/-18); benchmarks/kernels/benchmark_paged_attention.py (+12/-13); csrc/ops.h (+4/-4); tests/kernels/test_attention.py (+17/-18); (+17 more)
BODY: RFC: https://docs.google.com/document/d/1rg8CoOnrtz1LT-hCK86ZsHuhoTDtqSEGs8KrN4wbITo/edit#heading=h.uwasieoo42mu ⏎  ⏎ <img width="671" alt="Screenshot 2024-05-01 at 5 17 02 PM" src="https://github.com/vllm-project/vllm/assets/18510752/7f69a5a4-4133-4f7a-9aa0-92badafdd9f9"> ⏎  ⏎ - prompt: prompt tokens for seq group. ⏎ - subquery_len -> query_len: The new tokens to compute. (1 token for decode, new chunked tokens for chunked prefill)/. ⏎ - computed_len  …[truncated]

### L3-43c413ec57  (L3, 2024-05-03, sha 43c413ec570e, PR #4353)
TITLE: [Kernel] Use flashinfer for decoding (#4353)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flashinfer.v0_backend, L3.dispatch.selector, L3.dispatch.abstract_interface
FILES: csrc/cache_kernels.cu (+80/-0); vllm/_custom_ops.py (+12/-0); vllm/attention/backends/abstract.py (+9/-4); vllm/attention/backends/flashinfer.py (+220/-0); vllm/attention/selector.py (+6/-0); vllm/config.py (+5/-0); vllm/utils.py (+52/-15); vllm/worker/model_runner.py (+97/-26); csrc/cache.h (+8/-0); csrc/pybind.cpp (+4/-0); (+5 more)
BODY: This PR is a first attempt to integrate [flashinfer](https://flashinfer.ai/) for the decoding phase. The PR still uses flash attention for the prefill phase for now. ⏎  ⏎ Updated after discussion with @yzh119 ⏎ Things need to be fixed: ⏎  ⏎  ⏎ Next step:

### L3-63575bc2e1  (L3, 2024-05-06, sha 63575bc2e197, PR #4607)
TITLE: [Core][Optimization] change python dict to pytorch tensor (#4607)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface
FILES: csrc/cache_kernels.cu (+5/-15); vllm/attention/backends/abstract.py (+1/-1); vllm/attention/backends/flash_attn.py (+1/-1); vllm/attention/backends/flashinfer.py (+1/-1); vllm/attention/backends/rocm_flash_attn.py (+1/-1); vllm/attention/backends/torch_sdpa.py (+1/-1); vllm/attention/backends/xformers.py (+1/-1); vllm/attention/ops/paged_attn.py (+1/-1); csrc/cache.h (+1/-1); csrc/cpu/cache.cpp (+6/-14); (+9 more)
BODY: Prior to this PR, `blocks_to_copy` uses `Dict[int, List[int]]` type in Python, and convert to `std::map<int64_t, std::vector<int64_t>>&` in c++ . Although the type contains a reference `&` , it is actually a copy. Inside the cpp code, we convert the data into pytorch array. ⏎  ⏎ In distributed setting, the cost is even larger: `blocks_to_copy` will be serialized, and broadcasted, deserialized, passed to c++, and finally convert to pytorch tensor. ⏎  …[truncated]

### L3-0f9a6e3d22  (L3, 2024-05-08, sha 0f9a6e3d229c, PR #4573)
TITLE: [Bugfix][Kernel] allow non-power-of-2 for prefix prefill with alibi  (#4573)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/prefix_prefill.py (+25/-16); tests/kernels/test_prefix_prefill.py (+242/-1)
ISSUES: #4171 [Bug]: Server crash for bloom-3b while use prefix_caching, `AssertionError assert Lk in {16, 32, 64, 128}`
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/4171 ⏎  ⏎ allow non-power-of-two head sizes in prefix prefill with alibi, this is a small fix based on https://github.com/vllm-project/vllm/pull/4128. ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-5510cf0e8a  (L3, 2024-05-08, sha 5510cf0e8a6a, PR #4685)
TITLE: [Misc] Add `get_name` method to attention backends (#4685)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+5/-0); vllm/attention/backends/flash_attn.py (+4/-0); vllm/attention/backends/flashinfer.py (+7/-9); vllm/attention/backends/rocm_flash_attn.py (+4/-0); vllm/attention/backends/torch_sdpa.py (+4/-0); vllm/attention/backends/xformers.py (+4/-0); vllm/worker/model_runner.py (+2/-3)
BODY: This PR adds the name strs to the attention backends so that we can use the names to identify them (without importing the actual class).

### L3-20cfcdec99  (L3, 2024-05-08, sha 20cfcdec998b, PR #4659)
TITLE: [Core][Optimization] change python dict to pytorch tensor for blocks to swap (#4659)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.paged.python_wrapper, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface
FILES: csrc/cache_kernels.cu (+11/-5); vllm/attention/backends/abstract.py (+1/-1); vllm/attention/backends/flash_attn.py (+2/-2); vllm/attention/backends/flashinfer.py (+1/-1); vllm/attention/backends/rocm_flash_attn.py (+2/-2); vllm/attention/backends/torch_sdpa.py (+2/-2); vllm/attention/ops/paged_attn.py (+2/-2); csrc/cache.h (+2/-2); csrc/cpu/cache.cpp (+2/-2); tests/core/test_block_manager.py (+2/-2); (+11 more)
BODY: Continue after https://github.com/vllm-project/vllm/pull/4607 .

### L3-89579a201f  (L3, 2024-05-08, sha 89579a201f2c, PR #4686)
TITLE: [Misc] Use vllm-flash-attn instead of flash-attn (#4686)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.upstream_pip, L3.flash_attn.fork_pip, L3.flashinfer.v0_backend, L3.dispatch.selector
FILES: Dockerfile (+0/-21); requirements-cuda.txt (+1/-0); setup.py (+9/-5); vllm/attention/backends/flash_attn.py (+1/-1); vllm/attention/backends/flashinfer.py (+1/-1); vllm/attention/selector.py (+4/-3)
BODY: This PR is to use the pre-built `vllm-flash-attn` wheel instead of the original `flash-attn`.

### L3-0ee535b294  (L3, 2024-05-09, sha 0ee535b2945d, PR #4705)
TITLE: [Misc] Set block size at initialization & Fix test_model_runner (#4705)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: tests/worker/test_model_runner.py (+32/-58); vllm/worker/cpu_model_runner.py (+9/-12); vllm/worker/cpu_worker.py (+1/-0); vllm/worker/model_runner.py (+21/-33); vllm/worker/worker.py (+1/-1)
BODY: This PR sets the block size to `ModelRunner` at its initialization time, instead of setting it lazily. Currently, we don't have any reason to set this value lazily, since it is configured by the user argument. ⏎ Also, the PR fixes the abuse of `ModelRunner` in `test_model_runner` where basically `ModelRunner` was initialized with `None` arguments.

### L3-c833101740  (L3, 2024-05-09, sha c83310174055, PR #4535)
TITLE: [Kernel] Refactor FP8 kv-cache with NVIDIA float8_e4m3 support (#4535)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.flash_attn.fork_inline_cmake
FILES: csrc/attention/attention_kernels.cu (+110/-176); csrc/attention/dtype_fp8.cuh (+11/-5); csrc/cache_kernels.cu (+70/-73); .buildkite/check-wheel-size.py (+1/-1); CMakeLists.txt (+1/-1); cmake/utils.cmake (+2/-2); csrc/cache.h (+3/-1); csrc/quantization/fp8/amd/hip_float8.h (+0/-0); csrc/quantization/fp8/amd/hip_float8_impl.h (+0/-0); csrc/quantization/fp8/amd/quant_utils.cuh (+58/-1); (+7 more)
BODY: The first PR for #4532. ⏎  ⏎ Task list: ⏎  ⏎  ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-0fca3cdcf2  (L3, 2024-05-13, sha 0fca3cdcf265, PR #4751)
TITLE: [Misc] Enhance attention selector (#4751)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.selector, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+2/-3); vllm/attention/backends/flash_attn.py (+7/-6); vllm/attention/backends/flashinfer.py (+23/-10); vllm/attention/backends/rocm_flash_attn.py (+9/-7); vllm/attention/backends/torch_sdpa.py (+17/-11); vllm/attention/backends/xformers.py (+6/-6); vllm/attention/layer.py (+17/-2); vllm/attention/selector.py (+23/-5); vllm/model_executor/models/deepseek.py (+13/-3); vllm/model_executor/models/gemma.py (+10/-4); (+39 more)
BODY: This PR is to provide more information (such as block size and kv cache dtype) to attention backend selector so that it can be used to find the appropriate attention backend. Also, the PR moves `kv_cache_dtype` from `AttentionMetadata` to `Attention`. ⏎  ⏎ This PR is a prerequisite for #3648

### L3-1356df53bd  (L3, 2024-05-13, sha 1356df53bd5d, PR #3648)
TITLE: [Kernel] Use flash-attn for decoding (#3648)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.dispatch.selector
FILES: vllm/attention/backends/flash_attn.py (+73/-55); vllm/attention/selector.py (+14/-0); tests/kernels/test_flash_attn.py (+209/-0); tests/models/test_big_models.py (+1/-1); tests/models/test_fp8.py (+5/-5); vllm/worker/model_runner.py (+11/-4)
BODY: Vendors flash-attention from https://github.com/Dao-AILab/flash-attention/pull/824, prunes out the backward pass operator for faster compile times, adds reshape and cache kernel for flash attention kv cache layout, adds logic for selecting kv cache manager / attention backend based on temporary environment variable VLLM_TEMP_USE_FLASH_DECODE. Tested for single GPU on opt-125m, llama-7b

### L3-8a7cc254a0  (L3, 2024-05-15, sha 8a7cc254a064, PR #4820)
TITLE: Revert "[Kernel] Use flash-attn for decoding (#3648)" (#4820)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.dispatch.selector
FILES: vllm/attention/backends/flash_attn.py (+55/-73); vllm/attention/selector.py (+0/-14); tests/kernels/test_flash_attn.py (+0/-209); tests/models/test_big_models.py (+1/-1); tests/models/test_fp8.py (+5/-5); vllm/worker/model_runner.py (+4/-11)
BODY: Lora 3 & 4 test seems to have illegal memory access failure after this commit; ⏎  ⏎ ``` ⏎ [2024-05-14 23:51:18,182 E 22 22] logging.cc:101: Unhandled exception: N3c105ErrorE. what(): CUDA error: an illegal memory access was encountered ⏎ <br class="Apple-interchange-newline"> ⏎ ``` ⏎  ⏎ Exmaple: https://buildkite.com/vllm/ci/builds/7382#018f793d-1527-4e1c-ab59-c3a34ec55241 ⏎  ⏎ This reverts commit 1356df53bd5d6877358aff3d2bbd95f28f8009a4. ⏎  ⏎ FILL IN THE P …[truncated]

### L3-65bf2ac165  (L3, 2024-05-15, sha 65bf2ac16573, PR #4681)
TITLE: [Core][2/N] Model runner refactoring part 2. Combine prepare prefill / decode to a single API (#4681)
SOURCES: path_core, body_keyword, symbol_pickaxe
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+32/-36); vllm/attention/backends/flash_attn.py (+79/-16); vllm/attention/backends/flashinfer.py (+28/-10); vllm/attention/backends/rocm_flash_attn.py (+80/-18); vllm/attention/backends/torch_sdpa.py (+22/-6); vllm/attention/backends/xformers.py (+77/-15); vllm/attention/layer.py (+2/-3); vllm/attention/ops/paged_attn.py (+5/-5); tests/worker/test_model_runner.py (+84/-39); vllm/attention/__init__.py (+2/-3); (+8 more)
BODY: This PR combines prepare_prompt and prepare_decode into a single API. This PR also coelsce the attn metadata for prefill/decode to a single class and allow to slice them when running attn backend.  ⏎  ⏎ It also refactors subquery_start_loc which was not refactored in the previous PR ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-b5853f9963  (L3, 2024-05-16, sha b5853f99639a, PR #4845)
TITLE: [ROCm][AMD][Bugfix] adding a missing triton autotune config (#4845)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.flash_attention_rocm
FILES: vllm/attention/ops/triton_flash_attention.py (+10/-0)
LABELS: rocm
BODY: Vinayak Gokhale @vgokhale found that there is a missing triton autotune config in vllm, but was added to triton repo at some point of time.  ⏎ This small change however causes a series of cascading events which results in autotune picking a different config - the one that didn't exist in vllm. Without it, vllm's autotune picks a much worse config. This is to fix that. ⏎  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SUBMITTIN …[truncated]

### L3-2060e93659  (L3, 2024-05-16, sha 2060e93659f1, PR #4749)
TITLE: [Kernel] Add w8a8 CUTLASS kernels (#4749)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+26/-1); csrc/ops.h (+8/-0); csrc/pybind.cpp (+1/-0); csrc/quantization/cutlass_w8a8/common.hpp (+12/-0); csrc/quantization/cutlass_w8a8/cutlass_visitor_2x_broadcast_epilogue.hpp (+340/-0); csrc/quantization/cutlass_w8a8/scaled_mm_dq_c2x.cu (+296/-0); csrc/quantization/cutlass_w8a8/scaled_mm_dq_c3x.cu (+240/-0); csrc/quantization/cutlass_w8a8/scaled_mm_dq_entry.cu (+65/-0); tests/kernels/test_cutlass.py (+192/-0); vllm/_custom_ops.py (+17/-1)
BODY: This PR adds fp8_e4m3fn and int8 GEMM kernels, using NVIDIA CUTLASS and unit tests for them. The kernels are not used in this present PR, but are planned to be used in https://github.com/vllm-project/vllm/pull/4525. ⏎  ⏎ The main contributions of this PR is the function `cutlass_scaled_mm_dq`: ⏎ * Supports symmetric quantized activations and weights ⏎ * The activations may be either per-tensor or per-token ⏎ * The weights may be either per-tensor or p …[truncated]

### L3-9a31a817a8  (L3, 2024-05-16, sha 9a31a817a85a, PR #4869)
TITLE: [Bugfix] Fix FP8 KV cache support (#4869)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/flash_attn.py (+5/-5); vllm/attention/backends/flashinfer.py (+5/-5); vllm/attention/backends/rocm_flash_attn.py (+5/-5); vllm/attention/backends/torch_sdpa.py (+5/-5); vllm/attention/backends/xformers.py (+5/-5); vllm/attention/layer.py (+1/-1)
BODY: This PR fixes a bug introduced in #4751 that the `kv_cache_dtype` arg was not correctly passed to the `__init__` methods of the attention backends.

### L3-c0724fc915  (L3, 2024-05-18, sha c0724fc91503, PR #4658)
TITLE: [ROCm][Hardware][AMD] Adding Navi21 to fallback to naive attention if Triton is not used (#4658)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+3/-2)
LABELS: rocm
BODY: Navi3X - have major HW version 11, Navi21/Navi10 have HW version 10, MI series - HW version 9. ⏎  ⏎ https://github.com/ROCm/FasterTransformer-Internal/issues/247

### L3-b57e6c5949  (L3, 2024-05-19, sha b57e6c59491e, PR #4907)
TITLE: [Kernel] Add flash-attn back (#4907)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.fork_pip, L3.dispatch.selector
FILES: requirements-cuda.txt (+1/-1); vllm/attention/backends/flash_attn.py (+75/-54); vllm/attention/selector.py (+14/-0); tests/kernels/test_flash_attn.py (+208/-0); tests/models/test_big_models.py (+1/-1); tests/models/test_fp8.py (+5/-5)
BODY: This PR reverts #4820 by adding back `flash-attn`. Previously, using `flash-attn` for decoding caused errors when using small models (like the Llama 68M model in `lora/test_layer_variation.py`). This was because the index calculation for paged KV cache was done in `int` instead of `int64_t`, leading to integer overflow when `num_blocks` is large. In `vllm-flash-attn==2.5.8.post2`, the overflow bug was fixed.

### L3-99eff67ba9  (L3, 2024-05-21, sha 99eff67ba915, PR #4944)
TITLE: [Bugfix][Kernel] Add head size check for attention backend selection (#4944)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.dispatch.selector
FILES: vllm/attention/backends/flash_attn.py (+8/-4); vllm/attention/selector.py (+13/-3)
BODY: FILL IN THE PR DESCRIPTION HERE ⏎  ⏎ Previous #4886 PR cause the lora-test failing due to `Phi-2` with LoRA will make attention head_size to 80 while Flash Attention doesn't support it. ⏎  ⏎ - This PR fix the issues by adding a head size check when selecting attention backend. ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE** ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-5f6d10c14c  (L3, 2024-05-22, sha 5f6d10c14c17, PR #4722)
TITLE: [CI/Build] Enforce style for C++ and CUDA code with `clang-format` (#4722)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape
FILES: csrc/attention/attention_generic.cuh (+10/-9); csrc/attention/attention_kernels.cu (+299/-337); csrc/attention/attention_utils.cuh (+6/-5); csrc/attention/dtype_bfloat16.cuh (+39/-35); csrc/attention/dtype_float16.cuh (+47/-45); csrc/attention/dtype_float32.cuh (+33/-55); csrc/attention/dtype_fp8.cuh (+16/-16); csrc/cache_kernels.cu (+131/-157); csrc/cpu/attention.cpp (+205/-206); .clang-format (+26/-0); (+54 more)
BODY: We have made great strides in enforcing code quality and consistency within our Python codebase, particularly with the recent [MyPy integration](https://github.com/vllm-project/vllm/issues/3680). To maintain a high standard of code quality throughout the project, we should apply similar formatting and style guidelines to our C++ and CUDA code. ⏎  ⏎ This PR integrates `clang-format` into our `format.sh` code quality workflow to automatically enforce …[truncated]

### L3-a3a73ab069  (L3, 2024-05-22, sha a3a73ab0696b, PR #4893)
TITLE: [Misc] Load FP8 kv-cache scaling factors from checkpoints (#4893)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+25/-2); benchmarks/benchmark_latency.py (+6/-8); benchmarks/benchmark_throughput.py (+5/-7); benchmarks/kernels/benchmark_paged_attention.py (+4/-6); tests/models/test_fp8.py (+52/-28); vllm/config.py (+3/-5); vllm/engine/arg_utils.py (+3/-4); vllm/model_executor/layers/quantization/fp8.py (+45/-2); vllm/model_executor/models/arctic.py (+2/-1); vllm/model_executor/models/baichuan.py (+4/-2); (+30 more)
BODY: The 2nd PR for #4532. ⏎  ⏎ This PR supports loading FP8 kv-cache scaling factors from a FP8 checkpoint (with `.kv_scale` parameter). ⏎ Specifically, ⏎ 1. We now support `--kv-cache-dtype {auto, fp8, fp8_e4m3, fp8_e5m2}`. `auto=fp16 or bf16` and `fp8=fp8_e4m3`. ⏎ 2. If the checkpoint is in FP16, then kv-cache scaling factors can only be loaded via `--quantization-param-path`; otherwise kv-scale is always 1 regardless `fp8_e4m3` or `fp8_e5m2`. ⏎ 3. If th …[truncated]

### L3-ee3eea0a1b  (L3, 2024-05-23, sha ee3eea0a1b2c, PR #4960)
TITLE: [Misc] Take user preference in attention selector (#4960)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.dispatch.selector
FILES: vllm/attention/backends/flashinfer.py (+1/-0); vllm/attention/selector.py (+84/-61); tests/kernels/test_attention_selector.py (+84/-0)
BODY: The current selection logic on NVIDIA GPUs is a bit weird on NVIDIA GPUs. Specially, it checks whether the model can use FlashAttn, and directly falls back to xFormers if not, without considering consider user preference (i.e., environment variable `VLLM_ATTENTION_BACKEND`). User specified backend can only be used  when FlashAttn is valid for the model and installed in the environment. ⏎  ⏎ This PR refactors the logic to consider user preference fi …[truncated]

### L3-8e192ff967  (L3, 2024-05-24, sha 8e192ff967b4, PR #4799)
TITLE: [Kernel][Backend][Model] Blocksparse flash attention kernel and Phi-3-Small model (#4799)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: csrc/attention/attention_kernels.cu (+148/-37); csrc/cpu/attention.cpp (+21/-16); csrc/ops.h (+18/-17); vllm/_custom_ops.py (+21/-9); vllm/attention/backends/abstract.py (+1/-0); vllm/attention/backends/blocksparse_attn.py (+410/-0); vllm/attention/backends/flash_attn.py (+4/-1); vllm/attention/backends/rocm_flash_attn.py (+4/-1); vllm/attention/backends/torch_sdpa.py (+4/-1); vllm/attention/backends/xformers.py (+4/-1); (+13 more)
BODY: - Supports of Microsoft Phi-3-Small-8K and Phi-3-Small-128K models, which use blocksparse flash attention ⏎ - Prefilling Triton kernel for block-sparse attn ⏎ - Modified paged attention CUDA with the block-sparse attention, which allows hybrid sparsity pattern for each attention head. ⏎ - Use torch SPDA in prefilling phase for V100 or older GPUs, as well as CPU ⏎  ⏎ This is joint work between Microsoft GenAI @linxihui, @beagleski, and vLLM @zhuohan123 …[truncated]

### L3-1102bef219  (L3, 2024-05-27, sha 1102bef2195a, PR #4846)
TITLE: [Bugfix / Core] Prefix Caching Guards (merged with main) (#4846)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+2/-1); tests/prefix_caching/test_disable_sliding_window.py (+44/-0); tests/test_config.py (+24/-0); vllm/config.py (+64/-3); vllm/engine/arg_utils.py (+9/-3); vllm/model_executor/models/llama.py (+0/-4); vllm/model_executor/models/mixtral.py (+10/-12); vllm/model_executor/models/mixtral_quant.py (+0/-4); vllm/model_executor/models/qwen2.py (+14/-11); vllm/model_executor/models/starcoder2.py (+0/-2); (+1 more)
BODY: Updated version of #3903 ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-d4f3985907  (L3, 2024-05-28, sha d4f398590786, PR #4545)
TITLE: [Core] Sliding window for block manager v2 (#4545)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/prefix_prefill.py (+5/-1); tests/core/block/e2e/conftest.py (+26/-0); tests/core/block/e2e/test_correctness.py (+2/-9); tests/core/block/e2e/test_correctness_sliding_window.py (+168/-0); tests/core/block/test_block_manager_v2.py (+69/-0); vllm/core/block/block_table.py (+32/-2); vllm/core/block/cpu_gpu_block_allocator.py (+74/-0); vllm/core/block/interfaces.py (+9/-0); vllm/core/block_manager_v2.py (+17/-7); vllm/engine/arg_utils.py (+2/-1); (+2 more)
ISSUES: #3665 [Misc]: Implement SlidingWindowBlockTable in BlockManagerV2 | #4057 [Feature][Chunked prefill]: Make sliding window work
BODY: This implements sliding window in v2 block manager. ⏎  ⏎ First commit comes from #3967 by @ruthe98, but the actual change was somewhat more complex including the concept of a null block. ⏎  ⏎ It passes correctness tests with starcoder3b (the smallest model with sliding window I could find). The test does a bunch of assignments "x1 = 10; x2 = 33; ..." and then asks for value of one of them (which is outside the sliding window). If we tell it upfront w …[truncated]

### L3-5bd3c65072  (L3, 2024-05-29, sha 5bd3c650721c, PR #5091)
TITLE: [Core][Optimization] remove vllm-nccl (#5091)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_pip
FILES: requirements-cuda.txt (+0/-1); setup.py (+2/-5); .buildkite/test-pipeline.yaml (+0/-1); tests/distributed/test_pynccl_library.py (+0/-43); vllm/distributed/device_communicators/pynccl_wrapper.py (+7/-13); vllm/utils.py (+8/-35); vllm/worker/worker_base.py (+4/-2)
BODY: After months of investigation, I finally find the root cause is that NCCL 2.19 will turn on virtual memory by default, which costs memory during cudagraph capture. ⏎  ⏎ Per the [documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-cumem-enable): ⏎  ⏎ > NCCL_CUMEM_ENABLE ⏎ > (since 2.18) ⏎ >  ⏎ > Use CUDA cuMem* functions to allocate memory in NCCL. ⏎ >  ⏎ > Values accepted ⏎ > 0 or 1. Default is 0 in 2.18 (disabled); since …[truncated]

### L3-a22dea54d3  (L3, 2024-05-30, sha a22dea54d3e8, PR #5081)
TITLE: [Model] Support MAP-NEO model (#5081)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.paged.python_wrapper
FILES: csrc/attention/attention_kernels.cu (+6/-0); csrc/cpu/attention.cpp (+6/-0); vllm/attention/ops/paged_attn.py (+1/-1); benchmarks/kernels/benchmark_paged_attention.py (+1/-1); benchmarks/kernels/benchmark_rope.py (+1/-1); tests/kernels/test_attention.py (+1/-1); tests/kernels/test_cache.py (+1/-1); tests/kernels/test_pos_encoding.py (+1/-1)
BODY: This PR support the [MAP-NEO](https://github.com/multimodal-art-projection/MAP-NEO) model which is number of attention head is 192  ⏎  ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-0ab278ca31  (L3, 2024-06-03, sha 0ab278ca3102, PR #5138)
TITLE: [Core] Remove unnecessary copies in flash attn backend (#5138)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.fork_pip
FILES: requirements-cuda.txt (+1/-1); vllm/attention/backends/flash_attn.py (+7/-6)
BODY: With vllm-flash-attn == 2.5.8.post3, we can remove the unnecessary copies in flash attn backend by using the `out` kwarg directly. ⏎ --- ⏎  ⏎ [details omitted]

### L3-c65146e75e  (L3, 2024-06-05, sha c65146e75e71, PR #5271)
TITLE: [Misc] Fix docstring of get_attn_backend (#5271)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/selector.py (+2/-3)
BODY: 

### L3-3a6ae1d33c  (L3, 2024-06-05, sha 3a6ae1d33c7a, PR #5286)
TITLE: [CI] Disable flash_attn backend for spec decode (#5286)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test-pipeline.yaml (+5/-2)
BODY: See #5152

### L3-c96fc06747  (L3, 2024-06-07, sha c96fc0674794, PR #4965)
TITLE: [ROCm][AMD] Use pytorch sdpa math backend to do naive attention (#4965)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+29/-33)
LABELS: rocm
BODY: This pull request uses pytorch sdpa math backend to replace the existing naive attention in ROCm, as it shows latency improvement (e.g. prefill latency) over the current naive attention based on latency benchmarking. ⏎  ⏎ The naive attention is desirable (like in GPUs where support for ck flash-attention or triton flash-attention is not available).  ⏎  ⏎ FIX #xxxx (*link existing issues this PR will resolve*) ⏎  ⏎ **BEFORE SUBMITTING, PLEASE READ THE C …[truncated]

### L3-5467ac3196  (L3, 2024-06-09, sha 5467ac319636, PR #5047)
TITLE: [Kernel][Misc] Use TORCH_LIBRARY instead of PYBIND11_MODULE for custom ops (#5047)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.cache.cuda_reshape, L3.flash_attn.v0_backend, L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: csrc/attention/attention_kernels.cu (+18/-16); csrc/cache_kernels.cu (+8/-5); csrc/cpu/attention.cpp (+14/-12); CMakeLists.txt (+6/-16); Dockerfile.rocm (+3/-3); cmake/cpu_extension.cmake (+6/-6); cmake/utils.cmake (+8/-3); csrc/activation_kernels.cu (+1/-1); csrc/cache.h (+9/-5); csrc/cpu/cache.cpp (+8/-5); (+45 more)
BODY: ### This PR makes the following changes ⏎ - replaces the uses of `PYBIND11_MODULE` with `TORCH_LIBRARY` and adds schemas and meta functions (where) needed for all custom (C++/CUDA) kernels. ⏎ - The remainder of the custom operators are wrapped in _custom_ops.py, i.e. cuda_utils, cache_ops, moe, custom_ar and punica. ⏎ - The code + libraries are made to use the python stable api.  See #4694  ⏎  ⏎ ### Motivation ⏎ - Using `TORCH_LIBRARY` is the more offi …[truncated]

### L3-1a8bfd92d5  (L3, 2024-06-12, sha 1a8bfd92d5f3, PR #5292)
TITLE: [Hardware] Initial TPU integration (#5292)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen, L3.dispatch.selector
FILES: vllm/attention/backends/pallas.py (+232/-0); vllm/attention/selector.py (+11/-2); Dockerfile.tpu (+19/-0); benchmarks/benchmark_latency.py (+1/-1); benchmarks/benchmark_throughput.py (+1/-1); docs/source/getting_started/tpu-installation.rst (+75/-0); docs/source/index.rst (+2/-1); requirements-tpu.txt (+7/-0); setup.py (+17/-5); vllm/config.py (+5/-1); (+12 more)
LABELS: tpu
BODY: This PR implements the initial integration of the Google TPU backend. It uses PyTorch XLA for maximal reuse of the existing code base. ⏎  ⏎ The PR features: ⏎ * Seamless support for popular HF models such as Llama, Mistral, Gemma, etc. The model's head size must be either 128 or 256. ⏎ * Basic functionalities of vLLM, including continuous batching ⏎ * Optimized pallas kernels for FlashAttention and PagedAttention ⏎  ⏎ TODOs (next steps):

### L3-c3c2903e72  (L3, 2024-06-12, sha c3c2903e72c6, PR #5402)
TITLE: [Bugfix] Add device assertion to TorchSDPA (#5402)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/selector.py (+3/-0)
ISSUES: #5351 [Bug]: TorchSDPAMetadata is out of date
BODY: Fix #5351  ⏎  ⏎ Added a device assertion to avoid using TorchSDPA with NVIDIA GPUs. ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-ea3890a5f0  (L3, 2024-06-12, sha ea3890a5f031, PR #5293)
TITLE: [Core][Distributed] code deduplication in tp&pp with coordinator(#5293)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/pallas.py (+1/-1); tests/conftest.py (+3/-1); tests/distributed/test_custom_all_reduce.py (+3/-3); tests/distributed/test_pynccl.py (+8/-4); tests/lora/conftest.py (+13/-10); tests/worker/test_model_runner.py (+3/-1); vllm/distributed/communication_op.py (+13/-298); vllm/distributed/device_communicators/custom_all_reduce.py (+4/-9); vllm/distributed/device_communicators/custom_all_reduce_utils.py (+3/-4); vllm/distributed/device_communicators/pynccl.py (+3/-8); (+2 more)
BODY: This PR adds coordinator, the last piece in https://github.com/vllm-project/vllm/issues/3587 . The main benefit is that all distributed stuff (e.g. complicated selection logic in communication ops, prev/next rank logic, initialization&destruction logic) can be shared for all tp/pp code. ⏎  ⏎ After this PR, we explicitly require the code writer specify which group it wants to operate on: ⏎ - world group, by `get_world()` ⏎ - tp group, by `get_tp()` ⏎ - …[truncated]

### L3-80aa7e91fc  (L3, 2024-06-13, sha 80aa7e91fcd5, PR #4971)
TITLE: [Hardware][Intel] Optimize CPU backend and add more performance tips (#4971)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: vllm/attention/backends/torch_sdpa.py (+16/-7); vllm/attention/ops/ipex_attn.py (+120/-0); Dockerfile.cpu (+6/-2); README.md (+1/-1); docs/source/getting_started/cpu-installation.rst (+21/-2); requirements-cpu.txt (+1/-1)
BODY: This PR optimized CPU backend performance and added more performance tips. ⏎  ⏎ - Optimized input shape of torch_sdpa to use fast code path for better TTFT (~40% reduction). ⏎ - Added tip and example to use TCMalloc, it will significantly improve the performance. ⏎ - Initially integrated Paged attention from Intel Extension for PyTorch. ⏎ - Updated related doc. ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-6b0511a57b  (L3, 2024-06-13, sha 6b0511a57bdb, PR #5478)
TITLE: Revert "[Core] Remove unnecessary copies in flash attn backend" (#5478)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+6/-7)
BODY: Reverts vllm-project/vllm#5138 ⏎  ⏎ We seem to still have some cases where vllm-flash-attn will raise a bogus out shape error. Reverting for now pending proper fix.

### L3-28c145eb57  (L3, 2024-06-14, sha 28c145eb5755, PR #5558)
TITLE: [Bugfix] Fix typo in Pallas backend (#5558)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/pallas.py (+1/-1)
LABELS: tpu
BODY: Fixes a bug introduced in #5293

### L3-0e9164b40a  (L3, 2024-06-15, sha 0e9164b40abd, PR #5017)
TITLE: [mypy] Enable type checking for test directory (#5017)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend
FILES: .github/workflows/mypy.yaml (+1/-1); benchmarks/benchmark_serving.py (+9/-9); benchmarks/benchmark_throughput.py (+2/-2); benchmarks/kernels/benchmark_aqlm.py (+5/-5); benchmarks/kernels/benchmark_marlin.py (+5/-3); benchmarks/kernels/benchmark_moe.py (+18/-8); benchmarks/kernels/benchmark_paged_attention.py (+7/-4); benchmarks/kernels/benchmark_rope.py (+4/-3); examples/fp8/extract_scales.py (+6/-6); examples/offline_inference_distributed.py (+4/-4); (+82 more)
BODY: Improve type annotations across various files, especially under the `tests/` directory. This enables the `tests/` directory to be type checked in CI. ⏎  ⏎ @rkooo567 might be interested in this.

### L3-728c4c8a06  (L3, 2024-06-17, sha 728c4c8a063c, PR #3814)
TITLE: [Hardware][Intel GPU] Add Intel GPU(XPU) inference backend (#3814)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.dispatch.selector
FILES: vllm/attention/backends/ipex_attn.py (+355/-0); vllm/attention/selector.py (+13/-2); .buildkite/run-xpu-test.sh (+14/-0); .buildkite/test-template.j2 (+5/-0); Dockerfile.xpu (+22/-0); benchmarks/benchmark_latency.py (+1/-1); benchmarks/benchmark_throughput.py (+1/-1); docs/source/getting_started/xpu-installation.rst (+61/-0); docs/source/index.rst (+1/-0); requirements-xpu.txt (+11/-0); (+21 more)
LABELS: intel-gpu
BODY: **This PR is first PR for RFC #3725** ⏎  ⏎ _Intel is contributing both Intel CPU and Intel GPU support for vLLM. Initial PR #3634 for CPU already got merged._ ⏎  ⏎ This PR adds a new Intel GPU(also named `XPU` in pytorch context) backend to vLLM and supports the basic model inference feature, with FP16 type support.  ⏎  ⏎ Changes to vLLM: ⏎  ⏎ Added VLLM_TARGET_DEVICE ENV to specify backend explicitly. ⏎ Added `XPUExecutor`, `XPUWorker`, `XPUModelRunner`  …[truncated]

### L3-9e74d9d003  (L3, 2024-06-17, sha 9e74d9d003d5, PR #5592)
TITLE: Correct alignment in the seq_len diagram. (#5592)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+1/-1)
BODY: This PR corrects a minor formatting issue in the documentation for context_len, query_len, and seq_len. There is seemingly an extra dash in the `seq_len` line, causing a slight misalignment in the diagram. This misalignment could potentially confuse users trying to understand the definitions.

### L3-dd793d1de5  (L3, 2024-06-25, sha dd793d1de59b, PR #5422)
TITLE: [Hardware][AMD][CI/Build][Doc] Upgrade to ROCm 6.1, Dockerfile improvements, test fixes (#5422)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+6/-14); Dockerfile.rocm (+145/-64); cmake/utils.cmake (+12/-8); docs/source/getting_started/amd-installation.rst (+3/-3); tests/async_engine/test_openapi_server_ray.py (+2/-2); tests/distributed/test_utils.py (+11/-6); tests/entrypoints/test_openai_embedding.py (+2/-2); tests/entrypoints/test_openai_server.py (+2/-2); tests/entrypoints/test_openai_vision.py (+2/-2); tests/utils.py (+32/-6); (+5 more)
LABELS: rocm
BODY: This PR does the following: ⏎  ⏎ 1. Upgrade the default base image to ROCm 6.1.2 with the correct fixes applied, and the default PyTorch version to 2.4.0, bringing it in line with CUDA's update to 2.3.0. ⏎ 2. Refactor to enable multi-stage builds in Dockerfile.rocm: flash-attn and Triton are now built in separate stages as wheels which are then installed inside the final image. vLLM will eventually also be built in a similar manner once AMD CI is co …[truncated]

### L3-dda4811591  (L3, 2024-06-25, sha dda4811591fd, PR #5408)
TITLE: [Core] Refactor Worker and ModelRunner to consolidate control plane communication (#5408)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: vllm/attention/backends/abstract.py (+5/-1); vllm/attention/backends/blocksparse_attn.py (+2/-2); vllm/attention/backends/flash_attn.py (+2/-2); vllm/attention/backends/flashinfer.py (+2/-2); vllm/attention/backends/ipex_attn.py (+2/-2); vllm/attention/backends/pallas.py (+2/-2); vllm/attention/backends/rocm_flash_attn.py (+2/-2); vllm/attention/backends/torch_sdpa.py (+2/-2); vllm/attention/backends/xformers.py (+2/-2); tests/worker/test_model_input.py (+152/-0); (+19 more)
BODY: Currently the ModelRunner class contains both model execution code and multi-GPU control plane communication code, i.e.`broadcast_tensor_dict` calls. This makes it difficult to swap out the control plane mechanism, e.g., using NCCL vs CPU-based serialization to move the inputs from the LLMEngine to the Workers. It also makes it difficult to improve performance for multi-GPU settings, because we need to reason about all of the possible broadcast c …[truncated]

### L3-cbc53b6b8d  (L3, 2024-06-26, sha cbc53b6b8d87, PR #5855)
TITLE: [Hardware][TPU] Support parallel sampling & Swapping (#5855)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/pallas.py (+22/-8); vllm/worker/tpu_model_runner.py (+50/-26); vllm/worker/tpu_worker.py (+75/-22)
LABELS: tpu
BODY: This PR adds parallel sampling and swapping support for the TPU backend. ⏎  ⏎ ~~**NOTE:** Swapping is not implemented for the TPU backend yet. Therefore, vLLM will raise an error if OOM happens during parallel sampling.~~

### L3-f5c8628fdc  (L3, 2024-06-26, sha f5c8628fdc78, PR #5869)
TITLE: [Bugfix][TPU] Fix CPU cache allocation (#5869)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/pallas.py (+2/-3); vllm/worker/tpu_worker.py (+6/-2)
LABELS: tpu
BODY: This PR fixes the CPU cache allocation for the TPU backend. ~~Also, it avoids using dynamo for the swapping op.~~
