### L3-5952d8ab61  (L3, 2025-03-15, sha 5952d8ab61a3, PR #14842)
TITLE: [Attention] Get rid of mla cache alignment (#14842)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+0/-10); vllm/utils.py (+0/-6); vllm/worker/cache_engine.py (+3/-39); tests/kernels/test_cache.py (+11/-28)
LABELS: ready
BODY: With FlashMLA now being the default on Nvidia GPU and the fact that it [seemingly doesnt help the Triton backend anymore/ever-did](https://github.com/vllm-project/vllm/pull/12676#issuecomment-2652494624) (bit of a mystery). I think we can go ahead and rip this out reclaiming the memory lost to padding: ⏎  ⏎ ``` ⏎ VLLM_MLA_CUDA_MEM_ALIGN_KV_CACHE=1 VLLM_USE_V1=0 VLLM_USE_FLASHINFER_SAMPLER=1 vllm serve /home/vllm-dev/DeepSeek-R1 --tensor-parallel-size 8 …[truncated]

### L3-8c0d15d5c5  (L3, 2025-03-15, sha 8c0d15d5c565, PR #14798)
TITLE: [Misc][Easy] Annotate unused vars in the csrc files (#14798)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.custom_paged
FILES: csrc/rocm/attention.cu (+4/-3); csrc/prepare_inputs/advance_step.cu (+1/-1); csrc/quantization/fp8/amd/quant_utils.cuh (+1/-1); csrc/quantization/gptq/q_gemm.cu (+8/-8)
LABELS: ready
BODY: We turned on -Wno-unused-variable, and need some annotation to get the compiler happy.

### L3-b4ad56c1bd  (L3, 2025-03-17, sha b4ad56c1bd2f, PR #14846)
TITLE: [V1][TPU] Apply the ragged paged attention kernel fix and remove the padding. (#14846)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: requirements/tpu.txt (+6/-6); vllm/v1/worker/tpu_model_runner.py (+2/-5)
LABELS: tpu, ready, ci/build, v1
BODY: In https://github.com/vllm-project/vllm/pull/14597, we added the padding due to an ragged paged attention kernel issue. Since we have fixed the issue in the kernel, we can remove the padding in vLLM. This will make the kernel run more efficiently. ⏎  ⏎ Test plan: ⏎ 1. [Repro script](https://gist.github.com/vanbasten23/f00868169d6e4e68edb34d99ee7f287e) for https://github.com/vllm-project/vllm/pull/14597 ⏎ 2. VLLM_USE_V1=1 pytest -s -v vllm/tests/entrypoin …[truncated]

### L3-1e799b7ec1  (L3, 2025-03-17, sha 1e799b7ec1b1, PR #14910)
TITLE: [BugFix] Fix MLA + V1 + TP==1 causing reinitialization of cuda context (#14910)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+1/-1)
LABELS: bug, ready
BODY: on main: ⏎  ⏎ ``` ⏎ # SPDX-License-Identifier: Apache-2.0 ⏎  ⏎ from vllm import LLM, SamplingParams ⏎  ⏎ # Sample prompts. ⏎ prompts = [ ⏎     "Hello, my name is", ⏎     "The president of the United States is", ⏎     "The capital of France is", ⏎     "The future of AI is", ⏎ ] ⏎ # Create a sampling params object. ⏎ sampling_params = SamplingParams(temperature=0.8, top_p=0.95) ⏎  ⏎ # Create an LLM. ⏎ llm = LLM( ⏎     model="deepseek-ai/DeepSeek-V2-Lite", ⏎     trust_remote_code=True, ⏎ ) ⏎  …[truncated]

### L3-89fca671fb  (L3, 2025-03-17, sha 89fca671fbb5, PR #14921)
TITLE: [V1] Default MLA to V1 (#14921)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/engine/arg_utils.py (+1/-5)
LABELS: ready
BODY: 1. This is stable enough to recommend it. ⏎ 2. V0 is currently broken when serving a local method with trust remote code ⏎  ⏎ ``` ⏎ INFO 03-17 04:22:35 [llm_engine.py:241] Initializing a V0 LLM engine (v0.7.4.dev497+ga73e183e) with config: model='/home/vllm-dev/DeepSeek-R1', speculative_config=None, tokenizer='/home/vllm-dev/DeepSeek-R1', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None,  …[truncated]

### L3-cd0cd85102  (L3, 2025-03-17, sha cd0cd85102e4, PR #14926)
TITLE: [MISC] More AMD unused var clean up (#14926)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.custom_paged
FILES: csrc/rocm/attention.cu (+4/-4)
LABELS: ready
BODY: Will force the -Wno-unused-variable for AMD build in the following PR. ⏎  ⏎ Maybe just delete them in the future build. This is non-intrusive change to be safe. ⏎  ⏎ cc: @hongxiayang

### L3-64fc2193dc  (L3, 2025-03-18, sha 64fc2193dc77, PR #14347)
TITLE: [Misc][Docs] fix the comments of KV_T and CACHE_T in CALL_RESHAPE_AND_CACHE_XX macros (#14347)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+6/-6)
LABELS: ready
BODY: Perhaps I misunderstood something, but I currently think that the comments of KV_T and CACHE_T in the CALL_RESHAPE_AND_CACHE_XX macros are swapped. ⏎ If my understanding is correct, this PR will fix them. ⏎  ⏎ Please help to review~ Thanks! cc @comaniac @Yard1 @LucasWilkinson

### L3-53a0cf8b95  (L3, 2025-03-18, sha 53a0cf8b95be, PR #14988)
TITLE: [Neuron] trim attention kernel tests to fit trn1.2x instance (#14988)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/neuron/1_core/test_prefix_prefill.py (+1/-1)
LABELS: ready
BODY: We are trying to scale up neuron backend test infra, where trn1.2xl instance would be used for testing 1_core and 2_core unit test scripts. ⏎  ⏎ One special test we are trying to handle [here](https://buildkite.com/vllm/ci/builds/15537#0195a5df-b719-4d41-9f04-740d3e1803c1) can run out-of-memory on trn1.2x instance. The temporary solution here is to trim the test case down to fit CPU memory on the instance. ⏎  ⏎ cc @lingfanyu

### L3-b0e96aaebb  (L3, 2025-03-19, sha b0e96aaebbfb, PR #15145)
TITLE: [V1][TPU] Change kv cache shape. (#15145)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+7/-10); requirements/tpu.txt (+6/-6)
LABELS: ready, ci/build, v1
BODY: This PR changes the kv cache shape from `[num_blocks, block_size, num_kv_heads, head_size]` to `[num_blocks, block_size, num_kv_heads * head_size]`, in accordance with the ragged paged attention kernel change https://github.com/pytorch/xla/pull/8851, in order to unblock the multi-chip scenario: ⏎ - before the change, the ragged paged attention kernel will fail on some certain scenario ( eg if num_kv_head == 1 and dtype=bfloat16) because in this cas …[truncated]

### L3-a597a57595  (L3, 2025-03-20, sha a597a57595b5, PR #14570)
TITLE: [Attention] Flash Attention 3 - fp8 (#14570)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flash_attn.fork_build, L3.flashinfer.trtllm_gen, L3.mla.triton_v0, L3.dispatch.abstract_interface, L3.platform.cuda_selection
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/attention/backends/abstract.py (+1/-0); vllm/attention/backends/flash_attn.py (+68/-11); vllm/attention/backends/mla/common.py (+1/-1); vllm/attention/backends/utils.py (+0/-34); vllm/attention/layer.py (+7/-2); vllm/envs.py (+6/-1); vllm/fa_utils.py (+42/-0); vllm/platforms/cuda.py (+12/-9); vllm/v1/attention/backends/flash_attn.py (+40/-1); (+5 more)
LABELS: ready, ci/build, v1
BODY: This PR add support for FP8 KV cache with FlashAttention3 (related PR in flash-attn [here](https://github.com/vllm-project/flash-attention/pull/50)) cc @LucasWilkinson Please do not merge this PR as long as it's not referencing vllm-project/flash-attention yet. ⏎  ⏎ FlashAttention (contrary to FlashInfer) does attention with all Q, K and V in FP8. ⏎ The performance is usually better than FlashInfer FP8 KV and FlashAttention 3 with bf16. ⏎  ⏎ I added suppor …[truncated]

### L3-2b22290ce0  (L3, 2025-03-20, sha 2b22290ce01b, PR #15243)
TITLE: [V1] Add flag to disable cascade attention (#15243)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/config.py (+2/-0); vllm/engine/arg_utils.py (+12/-0); vllm/v1/worker/gpu_model_runner.py (+9/-5)
LABELS: ready, v1
BODY: This PR adds a flag to disable cascade attention in V1. This could be useful when potential numerical issues are concerned.

### L3-0c6f5023c3  (L3, 2025-03-20, sha 0c6f5023c390, PR #15250)
TITLE: [V1] Scheduler Refactoring [1/N] - Add Scheduler Interface (#15250)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.mla.common_v1
FILES: vllm/v1/attention/backends/flash_attn.py (+1/-1); vllm/v1/attention/backends/mla/common.py (+1/-1); tests/plugins_tests/test_scheduler_plugins.py (+1/-1); tests/v1/core/test_scheduler.py (+2/-1); tests/v1/worker/test_gpu_model_runner.py (+2/-2); vllm/engine/arg_utils.py (+1/-1); vllm/executor/ray_utils.py (+1/-1); vllm/v1/core/sched/__init__.py (+0/-0); vllm/v1/core/sched/interface.py (+139/-0); vllm/v1/core/sched/output.py (+0/-0); (+7 more)
LABELS: ready, v1
BODY: This PR is the first step to refactor the V1 scheduler. Specifically, it introduces the `v1/core/sched` directory with the `SchedulerInterface` class. No functionality or performance change is expected. ⏎  ⏎ It's a subset of #14731

### L3-0032903a5b  (L3, 2025-03-20, sha 0032903a5bb7, PR #15231)
TITLE: [Bugfix] detect alibi and revert to FA2 (#15231)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+2/-1); vllm/fa_utils.py (+9/-3)
LABELS: ready
ISSUES: #13810 [Usage]: vllm v0.7.2 can not support baichuan2 model
DEEP_STUDY: deep-study correctness case vllm:0032903a5b: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: On Hopper, Flash Attention 3 is enabled by default (since the v0.7.0 release), but FA3 does not support ALiBi positional encodings. This PR will revert to FA2 if alibi_slopes are passed in to `FlashAttentionBackend`. ⏎  ⏎ In the v0.8.1 release, the error when attempting to use a model with FA3 with ALiBi (such as [bigscience/bloom-1b1](https://huggingface.co/bigscience/bloom-1b1)) is: ⏎ ``` ⏎ RuntimeError: If cu_seqlens_k is passed in, then page table is …[truncated]

### L3-f8a08cb90d  (L3, 2025-03-21, sha f8a08cb90dc0, PR #14071)
TITLE: [V1] Enable Triton(ROCm) Attention backend for Nvidia GPUs (#14071)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:production-kernel-provenance
ARTIFACT_HINTS: L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/engine/arg_utils.py (+1/-1); vllm/platforms/cuda.py (+8/-3); vllm/platforms/interface.py (+1/-0); vllm/platforms/rocm.py (+3/-2); vllm/v1/attention/backends/triton_attn.py (+10/-10)
LABELS: rocm, ready, v1
BODY: Related issue: #12724  ⏎ - Rename v1 `ROCmAttention` to `TritonAttention` and allow user to use it on Nvidia GPUs through `VLLM_ATTENTION_BACKEND=triton_attn_vllm_v1` ⏎ - Since v1 ROCm attn backend is implemented with Triton, it can be used on Nvidia GPUs too for triton kernel development. ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-91ca929dc7  (L3, 2025-03-21, sha 91ca929dc7aa, PR #15280)
TITLE: [V1] Fix wrong import path of get_flash_attn_version (#15280)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+1/-1)
LABELS: ready, v1
ISSUES: #15265 [Bug]: V1 with MLA enable throw error `cannot import name 'get_flash_attn_version' from 'vllm.attention.backends.utils'`
DEEP_STUDY: deep-study correctness case vllm:91ca929dc7: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: FIX #15265

### L3-cfbb8c930f  (L3, 2025-03-21, sha cfbb8c930fcd, PR #15288)
TITLE: [TPU][V1] MHA Pallas backend (#15288)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+8/-2); tests/v1/tpu/test_mha_attn.py (+109/-0)
LABELS: tpu, ready, v1, multi-modality
BODY: The torch sdpa F.scaled_dot_product_attention impl is particularly inefficient on TPU.  ⏎ It's both slow and also causes issues when XLA compiling as reported here https://github.com/vllm-project/vllm/pull/15051, which is a major obstacle in multimodal adoption on TPU. ⏎  ⏎ This PR adds a Pallas branch to the vanilla MHA module used in most ViTs. ⏎  ⏎ Some numbers: ⏎  ⏎ Pre PR (TPUv6): ⏎ ``` ⏎ VLLM_USE_V1=1 vllm serve llava-hf/llava-1.5-7b-hf --max-model-len 2512  …[truncated]

### L3-b877031d80  (L3, 2025-03-22, sha b877031d806e, PR #15339)
TITLE: Remove openvino support in favor of external plugin (#15339)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen
FILES: vllm/attention/backends/openvino.py (+0/-146); .buildkite/run-openvino-test.sh (+0/-16); Dockerfile.openvino (+0/-29); docs/source/getting_started/installation.md (+0/-1); docs/source/getting_started/installation/ai_accelerator.md (+0/-77); docs/source/getting_started/installation/ai_accelerator/openvino.inc.md (+0/-110); requirements/openvino.txt (+0/-8); setup.py (+1/-9); tests/conftest.py (+1/-2); tests/kernels/test_attention_selector.py (+3/-11); (+10 more)
LABELS: documentation, ready, ci/build
ISSUES: #14374 [RFC]: Drop Support for OpenVINO
BODY: This PR implements the proposal from issue #14374 to remove openvino ⏎ support from main and instead move it to an external plugin repo: ⏎  ⏎ https://github.com/vllm-project/vllm-openvino ⏎  ⏎ Further details on the justification can be found in the issue.

### L3-dccf535f8e  (L3, 2025-03-23, sha dccf535f8edb, PR #15191)
TITLE: [V1] Enable V1 Fp8 cache for FA3 in the oracle (#15191)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.mla.triton_v0, L3.mla.common_v1, L3.platform.cuda_selection
FILES: vllm/attention/backends/flash_attn.py (+12/-4); vllm/attention/backends/mla/common.py (+1/-1); vllm/config.py (+0/-4); vllm/engine/arg_utils.py (+14/-3); vllm/platforms/cuda.py (+3/-5); vllm/v1/attention/backends/flash_attn.py (+6/-4); vllm/v1/attention/backends/mla/common.py (+1/-1); vllm/vllm_flash_attn/fa_utils.py (+6/-0); .gitignore (+2/-1)
LABELS: ready, v1
BODY: Now that FA3 Fp8 support has landed (https://github.com/vllm-project/vllm/pull/14570) we can enable Fp8 KV-caches for Hopper devices in V1 ⏎  ⏎ Update the oracle to allow this plus basic refactors ⏎  ⏎ Test script ⏎ ``` ⏎ # SPDX-License-Identifier: Apache-2.0 ⏎  ⏎ from vllm import LLM, SamplingParams ⏎  ⏎ # Sample prompts. ⏎ prompts = [ ⏎     "Hello, my name is", ⏎     "The president of the United States is", ⏎     "The capital of France is", ⏎     "The future of AI is", ⏎ ] ⏎ #  …[truncated]

### L3-7ffcccfa5c  (L3, 2025-03-24, sha 7ffcccfa5ca3, PR #15377)
TITLE: Revert "[CI/Build] Use uv python for docker rather than ppa:deadsnakess/ppa (#13569)" (#15377)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: Dockerfile (+51/-38)
LABELS: ready, ci/build
ISSUES: #14991 [Bug]: Docker image in trunk cannot find libpython.so | #15088 [Installation]: python is missing inside the v0.8.0 docker | #15174 [Misc]: missing python inside the container v0.8.1 | #15359 [Bug]: Can't create non-root user using vllm/vllm-openai:v0.8.1 as a base image
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 13569 reason=build_or_dependency
BODY: This reverts commit d9292786e106f62c7315832d26d2ffb4eb28f023. ⏎  ⏎ Closes #15174 ⏎ Closes #15088 ⏎ Closes #15359 ⏎ Closes #14991

### L3-4f044b1d67  (L3, 2025-03-25, sha 4f044b1d6796, PR #14744)
TITLE: [Kernel][CPU] CPU MLA (#14744)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:production-kernel-provenance
ARTIFACT_HINTS: L3.mla.triton_v0
FILES: csrc/cpu/mla_decode.cpp (+393/-0); vllm/_custom_ops.py (+12/-0); vllm/attention/backends/cpu_mla.py (+303/-0); vllm/attention/backends/mla/common.py (+13/-5); vllm/platforms/cpu.py (+3/-3); vllm/worker/cpu_model_runner.py (+1/-0); vllm/worker/cpu_worker.py (+1/-0); .buildkite/run-cpu-test.sh (+2/-0); cmake/cpu_extension.cmake (+1/-0); csrc/cpu/cache.cpp (+74/-0); (+5 more)
LABELS: ready, ci/build
BODY: In this PR, I add preliminary support for MLA on CPU. I'm opening this PR to get feedback and comments from the maintainers on the high level design. The MLA kernel itself is not optimized and I plan to optimize it further (either in this PR or leave it in a future PR). ⏎  ⏎ The main changes can be summarized as follows ⏎ - Add `concat_and_cache_mla` CPU kernel ⏎ - Add `mla_decode_kvcache_cpu` kernel. This currently does not follow any existing API. See  …[truncated]

### L3-051da7efe3  (L3, 2025-03-25, sha 051da7efe395, PR #15160)
TITLE: Fix CUDA kernel index data type in vllm/csrc/quantization/gptq_marlin/awq_marlin_repack.cu +10 (#15160)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.custom_paged
FILES: csrc/rocm/attention.cu (+26/-26); csrc/quantization/gptq_marlin/awq_marlin_repack.cu (+6/-6); csrc/quantization/gptq_marlin/gptq_marlin.cu (+14/-14); csrc/quantization/gptq_marlin/gptq_marlin_repack.cu (+8/-8); csrc/quantization/marlin/dense/marlin_cuda_kernel.cu (+5/-5); csrc/quantization/marlin/qqq/marlin_qqq_gemm_kernel.cu (+7/-7); csrc/quantization/marlin/sparse/marlin_24_cuda_kernel.cu (+7/-7)
LABELS: ready
BODY: Summary: ⏎ CUDA kernel variables matching the type `(thread|block|grid).(Idx|Dim).(x|y|z)` [have the data type `uint`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#built-in-variables). ⏎  ⏎ Many programmers mistakenly use implicit casts to turn these data types into `int`. In fact, the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) it self is inconsistent and incorrect in its use of data types in programming  …[truncated]

### L3-33437bc6e7  (L3, 2025-03-25, sha 33437bc6e7af, PR #15492)
TITLE: [BugFix] Fix nightly MLA failure (FA2 + MLA chunked prefill, i.e. V1, producing bad results) (#15492)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.merge.triton_lse
FILES: vllm/attention/ops/triton_merge_attn_states.py (+9/-0)
LABELS: ready
BODY: For MLA chunked prefill we use `merge_attn_states` to compute the context for a chunked-prefill, this sometimes leads to 0 length contexts when regular prefills and chunked-prefills are mixed in the same batch. FA2 and FA3 have differences in what the return when the sum-exp is 0, (i.e. what to return for log(0) which is technically undefined), FA3 returns -inf while FA2 returns inf. The code was written for FA3, i.e. -inf (I think this is also j …[truncated]

### L3-1aa162e030  (L3, 2025-03-26, sha 1aa162e030d6, PR #15532)
TITLE: Apply torchfix (#15532)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+2/-3); vllm/lora/models.py (+3/-1); vllm/model_executor/models/nemotron.py (+3/-3); vllm/model_executor/models/phi4mm_utils.py (+6/-3); vllm/multimodal/image.py (+1/-1)
LABELS: ready, multi-modality
BODY: Fix all issues reported by torchfix.

### L3-ecff8309a3  (L3, 2025-03-26, sha ecff8309a3ca, PR #15557)
TITLE: [ROCm] Env variable to trigger custom PA (#15557)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+2/-1); vllm/envs.py (+6/-0)
LABELS: ready
BODY: Returning the trigger env for the custom PA that got lost during upstreaming of this kernel ⏎ This would allow to force disable custom PA kernel and fall back to the default implementation

### L3-8958217ad5  (L3, 2025-03-27, sha 8958217ad5a6, PR #15211)
TITLE: [Bugfix] Fix use_cascade_attention handling for Alibi-based models on vllm/v1 (#15211)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/utils.py (+13/-1); vllm/v1/worker/gpu_model_runner.py (+5/-2)
LABELS: ready, v1
BODY: When using Alibi-based models like MPT, the following assertion error causes. ⏎ ```bash ⏎ AssertionError: Cascade attention does not support ALiBi. ⏎ ``` ⏎  ⏎ This is because that the determination logic for `use_cascade` in `gpu_model_runner.py` incorrectly uses a hard-coded `use_alibi=False`. Therefore, `cascade_attention` is wrongly enabled. ⏎  ⏎ This PR allows setting `use_alibi` based on the alibi configuration specified in the `config.json` for MPT model …[truncated]

### L3-7c1f760024  (L3, 2025-03-28, sha 7c1f7600248a, PR #15659)
TITLE: [Kernel][TPU][ragged-paged-attn] vLLM code change for PR#8896 (#15659)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+22/-21); requirements/tpu.txt (+6/-6); vllm/v1/worker/tpu_model_runner.py (+5/-6); vllm/v1/worker/tpu_worker.py (+4/-4)
LABELS: tpu, ready, ci/build, v1
BODY: Updated for APIs for the most recent kernel

### L3-e6e3c55ef2  (L3, 2025-03-31, sha e6e3c55ef28f, PR #14549)
TITLE: Move dockerfiles into their own directory (#14549)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1); docker/Dockerfile (+0/-0); docker/Dockerfile.arm (+0/-0); docker/Dockerfile.cpu (+0/-0); docker/Dockerfile.hpu (+0/-0); docker/Dockerfile.neuron (+0/-0); docker/Dockerfile.ppc64le (+0/-0); docker/Dockerfile.rocm (+0/-0); docker/Dockerfile.rocm_base (+0/-0); docker/Dockerfile.s390x (+0/-0); (+24 more)
LABELS: documentation, ci/build
BODY: Follow up to #12547 to move the dockerfiles to their own directory too.

### L3-a57a3044aa  (L3, 2025-04-01, sha a57a3044aa57, PR #15820)
TITLE: [ROCm][Build][Bugfix] Bring the base dockerfile in sync with the ROCm fork (#15820)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm_base (+30/-22); requirements/rocm-build.txt (+1/-1)
LABELS: ci/build
BODY: Bringing the base dockerfile in sync with the ROCm fork, where the `rocm/vllm-dev:base` image is built from ⏎ Also including changes from https://github.com/vllm-project/vllm/pull/15709 to the requirements file, since this PR would come in conflict with just the cmake change in the dockerfile

### L3-2edc87b161  (L3, 2025-04-02, sha 2edc87b161a1, PR #15848)
TITLE: [Bugfix] Fix cache block size calculation for CPU MLA (#15848)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/worker/cpu_worker.py (+1/-1)
LABELS: ready
BODY: Calculate correct block size (in bytes) for CPU MLA

### L3-e73ff24e31  (L3, 2025-04-02, sha e73ff24e31d2, PR #15720)
TITLE: [ROCM][KERNEL] Paged attention for V1 (#15720)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.triton.prefix_prefill, L3.triton.chunked_prefill_paged_decode, L3.triton.v1_backend, L3.rocm.custom_paged, L3.rocm.rocm_flash_attn_v0, L3.platform.rocm_selection
FILES: csrc/rocm/attention.cu (+73/-32); vllm/_custom_ops.py (+4/-2); vllm/attention/backends/rocm_flash_attn.py (+6/-18); vllm/attention/ops/chunked_prefill_paged_decode.py (+103/-54); vllm/attention/ops/paged_attn.py (+2/-0); vllm/attention/ops/prefix_prefill.py (+1/-0); vllm/platforms/rocm.py (+20/-1); vllm/v1/attention/backends/triton_attn.py (+1/-0); csrc/rocm/ops.h (+3/-2); csrc/rocm/torch_bindings.cpp (+3/-1); (+3 more)
LABELS: ready, ci/build, v1
BODY: Adopting ROCM Paged Attention to be use in V1 FA as alternative to Triton kernel. Perf I see: ⏎  ⏎ Baseline (VLLM_ROCM_CUSTOM_PAGED_ATTN=0 and --no-enable-prefix-caching): ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     1000 ⏎ Benchmark duration (s):                  27.51 ⏎ Total input tokens:                      215196 ⏎ Total generated tokens:                  197090 ⏎ Request throughput (req/s):            …[truncated]

### L3-d2b58ca203  (L3, 2025-04-03, sha d2b58ca203fc, PR #15911)
TITLE: [Neuron][kernel] Fuse kv cache into a single tensor (#15911)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/ops/nki_flash_attn.py (+38/-47); tests/neuron/1_core/test_cache.py (+3/-1); tests/neuron/1_core/test_prefix_prefill.py (+5/-8)
BODY: Fusing KV cache into a single tensor can help eliminate unnecessary slice operator on K/V cache tensor. ⏎  ⏎ ``` ⏎ %p11.224 = bf16[2,17487,4,32,64]{4,3,2,1,0} parameter(11), frontend_attributes={neff_input_names="input11"} ⏎ %slice.225 = bf16[1,17487,4,32,64]{4,3,2,1,0} slice(bf16[2,17487,4,32,64]{4,3,2,1,0} %p11.224), slice={[0:1], [0:17487], [0:4], [0:32], [0:64]} ⏎ %reshape.226 = bf16[17487,4,32,64]{3,2,1,0} reshape(bf16[1,17487,4,32,64]{4,3,2,1,0} %sli …[truncated]

### L3-b6be6f8d1e  (L3, 2025-04-03, sha b6be6f8d1e49, PR #15732)
TITLE: [TPU] Support sliding window and logit soft capping in the paged attention kernel for TPU. (#15732)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+6/-6); .buildkite/run-tpu-v1-test.sh (+4/-2); tests/entrypoints/llm/test_accuracy.py (+20/-10); tests/v1/tpu/test_pallas.py (+98/-0)
LABELS: tpu, ready, ci/build, v1
BODY: After I added the sliding window and logit soft-capping support to the paged attention Pallas kernel, this PR is intended to added the sliding window and logit soft-capping at the vLLM level, so that we can support gemma model. ⏎  ⏎ Test plans: ⏎ - pytest -s -v /workspace/vllm/tests/v1/tpu/test_pallas.py ⏎ - pytest -v -s tests/entrypoints/llm/test_accuracy.py::test_lm_eval_accuracy_v1_engine 2>&1 | tee out.txt ⏎  ⏎ The Gemma model that we care about are [goo …[truncated]

### L3-fadc59c0e6  (L3, 2025-04-04, sha fadc59c0e6b3, PR #16041)
TITLE: [TPU][V1] Remove ragged attention kernel parameter hard coding (#16041)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+6/-14); vllm/v1/worker/tpu_model_runner.py (+2/-6)
LABELS: tpu, ready, v1
BODY: 

### L3-40a36ccfeb  (L3, 2025-04-04, sha 40a36ccfeb49, PR #15717)
TITLE: [ROCm][Bugfix] Use platform specific FP8 dtype (#15717)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/prefix_prefill.py (+1/-1)
LABELS: ready
BODY: 

### L3-620fc2d09e  (L3, 2025-04-05, sha 620fc2d09ed8, PR #16112)
TITLE: [Model] fix model testing for TeleChat2ForCausalLM and V0 llama4 (#16112)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+5/-0); vllm/model_executor/models/telechat2.py (+6/-2)
BODY: 1. `TeleChat2ForCausalLM` inherits `LlamaForCausalLM` which requires a change to `_init_model` signature ⏎ 2. irope is not supported for V0 attention, we need to add the parameter and warning.

### L3-55dcce91df  (L3, 2025-04-07, sha 55dcce91df15, PR #16113)
TITLE: Upstream Llama4 Support to Main (#16113)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.triton.v1_backend
FILES: .buildkite/test-pipeline.yaml (+2/-1); benchmarks/kernels/benchmark_moe.py (+3/-0); docs/source/models/supported_models.md (+9/-2); examples/offline_inference/audio_language.py (+1/-1); examples/offline_inference/vision_language.py (+37/-0); examples/offline_inference/vision_language_multi_image.py (+38/-0); requirements/common.txt (+1/-1); requirements/test.in (+1/-1); requirements/test.txt (+1/-1); tests/models/decoder_only/audio_language/test_ultravox.py (+13/-1); (+33 more)
LABELS: documentation, frontend, ready, ci/build, v1, multi-modality
ISSUES: #16177 [Bug]: AttributeError: 'Llama4ForConditionalGeneration' object has no attribute 'sampler' with prompt_logprobs
BODY: As a follow up of https://github.com/vllm-project/vllm/pull/16104, we upstream the Llama4 support to the main branch. ⏎  ⏎ The goal of this PR: ⏎  ⏎ - Support the llama4 ⏎ - Clean up some hacks ⏎  ⏎ More enhancement will be tracked in https://github.com/vllm-project/vllm/issues/16114. ⏎  ⏎ Fixes from v0.8.3: ⏎ - Fix missing sampler ⏎ - Fix failing CI on `transformers==4.51.0` ⏎  ⏎ FIX #16177

### L3-05a015d6a5  (L3, 2025-04-08, sha 05a015d6a52e, PR #16212)
TITLE: Add warning for Attention backends that do not support irope yet (#16212)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flashinfer.v0_backend, L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/flashinfer.py (+8/-0); vllm/attention/backends/hpu_attn.py (+5/-0); vllm/attention/backends/ipex_attn.py (+8/-0); vllm/attention/backends/pallas.py (+8/-0); vllm/attention/backends/rocm_flash_attn.py (+5/-0); vllm/attention/backends/torch_sdpa.py (+5/-0); vllm/attention/backends/xformers.py (+5/-0); vllm/v1/attention/backends/pallas.py (+8/-0)
LABELS: tpu, ready, v1
ISSUES: #16189 [Bug]: use_irope error on run using vllm docker
BODY: Same fix as #16112 to add use_irope flag to attention backends that might be used in Llama4. Didn't make changes for MLA backends which are not used in Llama4 ⏎  ⏎ Fixes #16189  ⏎  ⏎ Tested linter happy with `pre-commit run --show-diff-on-failure --color=always --hook-stage manual --all-files`

### L3-e1a2c699dd  (L3, 2025-04-08, sha e1a2c699dda8, PR #16209)
TITLE: [BugFix] Fix Llama4 - Index Error When Single Request Near Max Context (#16209)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+1/-1)
LABELS: ready, v1
BODY: Fix for: https://github.com/vllm-project/vllm/issues/16157

### L3-2976dc27e9  (L3, 2025-04-08, sha 2976dc27e9dc, PR #16198)
TITLE: [Bug] [ROCm] Fix Llama 4 Enablement Bug on ROCm: V0 ROCmFlashAttentionImpl and Triton Fused MoE bugs (#16198)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+5/-1); vllm/utils.py (+9/-8); vllm/model_executor/layers/fused_moe/fused_moe.py (+2/-0)
LABELS: ready
BODY: # Description ⏎ This PR fixes two bugs: ⏎  ⏎ 1. `TypeError: ROCmFlashAttentionImpl.__init__() got an unexpected keyword argument 'use_irope'` ⏎ 2. Fix the `topk_weights` in `invoke_fused_moe_kernel` being not contiguous under V1 + ROCm + torch.compile + Dynamo + hipgraph mode.

### L3-819d548e8a  (L3, 2025-04-09, sha 819d548e8a4e, PR #16312)
TITLE: [BugFix] logger is not callable (#16312)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/hpu_attn.py (+2/-2)
LABELS: ready
BODY: As title this is maybe by mistake

### L3-04149cce27  (L3, 2025-04-09, sha 04149cce2775, PR #16314)
TITLE: [BugFix] fix some typos found by typos. (#16314)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.mla.triton_v0, L3.mla.common_v1
FILES: vllm/attention/backends/flash_attn.py (+1/-1); vllm/attention/backends/hpu_attn.py (+3/-3); vllm/attention/backends/mla/common.py (+3/-3); vllm/attention/backends/xformers.py (+3/-3); vllm/attention/ops/nki_flash_attn.py (+1/-1); vllm/v1/attention/backends/mla/common.py (+2/-2); benchmarks/benchmark_serving.py (+2/-2); benchmarks/benchmark_serving_structured_output.py (+2/-2); csrc/mamba/causal_conv1d/causal_conv1d.cu (+1/-1); vllm/benchmarks/serve.py (+2/-2); (+11 more)
LABELS: frontend, tpu, ready, v1
BODY: This patch fix some typos found by typos https://github.com/crate-ci/typos ⏎  ⏎ there are quite a few typos left but maybe make break change so I did not touch them. ⏎  ⏎ ```toml ⏎ [default.extend-words] ⏎ # Random strings. ⏎ "Dum" = "Dum" ⏎ "Hel" = "Hel" ⏎ "ba" = "ba" ⏎ "hellow" = "hellow" ⏎ # Showed up in examples. ⏎ "thw" = "thw" ⏎ # Showed up in test. ⏎ "hsa" = "hsa" ⏎ # Tech words ⏎ "WRONLY" = "WRONLY" ⏎ "typ" = "typ" ⏎ "arange" = "arange" ⏎ "ot" = "ot" ⏎ "Ded" = "Ded" ⏎ "indicies"  …[truncated]

### L3-e9528f6dc6  (L3, 2025-04-11, sha e9528f6dc614, PR #16173)
TITLE: [Kernel] support merge_attn_states CUDA kernel, 3x speedup (#16173)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fork_inline_cmake, L3.merge.cuda_lse, L3.mla.triton_v0, L3.mla.common_v1
FILES: CMakeLists.txt (+1/-0); csrc/attention/merge_attn_states.cu (+173/-0); csrc/ops.h (+9/-0); csrc/torch_bindings.cpp (+15/-0); vllm/_custom_ops.py (+11/-0); vllm/attention/backends/mla/common.py (+1/-2); vllm/attention/ops/merge_attn_states.py (+42/-0); vllm/v1/attention/backends/flash_attn.py (+1/-1); vllm/v1/attention/backends/mla/common.py (+1/-1); tests/kernels/test_merge_attn_states.py (+265/-0)
LABELS: ready, ci/build, v1
BODY: base on [vllm/attention/ops/triton_merge_attn_states.py](https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_merge_attn_states.py) ⏎  ⏎ Use CUDA kernel instead of Triton to minimize CPU overhead. Compared to the Triton kernel, the CUDA kernel implemented in this PR can achieve a maximum speedup of over `3x`. @WoosukKwon, End2End performance improved for R1 with PP=3 + TP=8 on L20,  4K IN:1K OUT (TTFT 5687.80 ms -> 5654.02 ms). Th …[truncated]

### L3-ce4ddd2d1a  (L3, 2025-04-14, sha ce4ddd2d1a96, PR #16553)
TITLE: [Misc] remove warning if triton>=3.2.0 (#16553)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.decode_attention
FILES: vllm/attention/ops/triton_decode_attention.py (+6/-5)
LABELS: ready
BODY: Remove warning if triton>=3.2.0, since now triton are already >= 3.2.0

### L3-280d62b8a2  (L3, 2025-04-15, sha 280d62b8a2cf, PR #16123)
TITLE: [Kernel] Remove redundant Exp calculations (#16123)
SOURCES: path_core
ARTIFACT_HINTS: L3.merge.triton_lse
FILES: vllm/attention/ops/triton_merge_attn_states.py (+6/-3)
LABELS: ready
BODY: Remove redundant Exp calculations

### L3-e82ee40de3  (L3, 2025-04-16, sha e82ee40de336, PR #16693)
TITLE: [Bugfix][Kernel] fix potential cuda graph broken for merge_attn_states kernel (#16693)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.merge.cuda_lse
FILES: csrc/attention/merge_attn_states.cu (+15/-10)
LABELS: ready
BODY: Fix potential CUDA graph broken for the merge_attn_states kernel. A CUDA graph error related to merge_state was observed in sglang (https://github.com/sgl-project/sglang/issues/5404) and fixed in https://github.com/sgl-project/sglang/pull/5419. Since the merge_attn_states kernel is often active as a fundamental kernel in many scenarios, it would be better to bind the merge_attn_states kernel to the CUDA stream, as required by the CUDA graph. This …[truncated]

### L3-3408e47159  (L3, 2025-04-17, sha 3408e471597e, PR #15960)
TITLE: [P/D][V1] KV Connector API V1 (#15960)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+44/-1); examples/offline_inference/disaggregated-prefill-v1/decode_example.py (+36/-0); examples/offline_inference/disaggregated-prefill-v1/prefill_example.py (+43/-0); examples/offline_inference/disaggregated-prefill-v1/run.sh (+5/-0); tests/v1/core/test_scheduler.py (+406/-9); vllm/distributed/kv_transfer/__init__.py (+11/-0); vllm/distributed/kv_transfer/kv_connector/base.py (+4/-0); vllm/distributed/kv_transfer/kv_connector/factory.py (+47/-5); vllm/distributed/kv_transfer/kv_connector/v1/__init__.py (+8/-0); vllm/distributed/kv_transfer/kv_connector/v1/base.py (+209/-0); (+14 more)
LABELS: documentation, ready, ci/build, v1
BODY: # APIS ARE SUBJECT TO CHANGE IN FOLLOW UPS ⏎  ⏎ ### **TL;DR:** ⏎ This PR opens the KV connector API in v1 to support disaggregated prefill. It also includes a minimal functional implementation as an example of how to use the connector API. ⏎  ⏎ Detailed design doc: https://docs.google.com/document/d/1uPGdbEXksKXeN4Q9nUm9hzotqEjQhYmnpAhidLuAsjk ⏎  ⏎ _This PR is co-authored by:_ ⏎ - _KuntaiDu <kuntai@uchicago.edu>_ ⏎ - _YaoJiayi <1200040070@link.cuhk.edu.cn>_ ⏎  ⏎ ## TO …[truncated]

### L3-183dad7a85  (L3, 2025-04-17, sha 183dad7a8548, PR #13111)
TITLE: [Attention] Update to lastest FA3 code (#13111)
SOURCES: path_core, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, corpus:kernel-correctness-cases(introducing)
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fork_build, L3.mla.triton_v0, L3.mla.common_v1
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/attention/backends/mla/common.py (+92/-90); vllm/attention/backends/utils.py (+25/-1); vllm/v1/attention/backends/flash_attn.py (+58/-1); vllm/v1/attention/backends/mla/common.py (+65/-25)
LABELS: ready, ci/build, v1
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:d0da99fb70 (fix PR 16998) || deep-study: introduced the defect fixed in case vllm:0f87d8f7b2 (fix PR 17574)
BODY: NOTE: Tested MLA on AMD V0 is working, V1 is broken but is also broken on main ⏎  ⏎ Perf: https://docs.google.com/spreadsheets/d/1U5lsoCKuWq99Cz1QbWkc0dBn1bij1Ifb3tphE2UXJj0/edit?usp=sharing ⏎   ⏎ # Main: ⏎  ⏎ ``` ⏎ -------------------------------------- ⏎ Full Command: ⏎ VLLM_USE_V1=0 lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=1638 …[truncated]

### L3-0377b8310b  (L3, 2025-04-17, sha 0377b8310b28, PR #16673)
TITLE: [MLA] Simplification to batch P/D reordering (#16673)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+5/-7); vllm/v1/worker/gpu_model_runner.py (+7/-9)
LABELS: ready, v1
BODY: I noticed that we're unnecessarily re-creating the sampling metadata twice when reordering the batch requests into prefill and decode groups for MLA. ⏎  ⏎ This moves the reorder op from the start of the `_prepare_inputs()` method to the end of the `_update_stats()` method (which is called right before).

### L3-aaec845f8e  (L3, 2025-04-18, sha aaec845f8ed7, PR #16431)
TITLE: [ROCm] [Attention] Cleanup ROCm output passing (#16431)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+18/-23)
LABELS: rocm, ready
BODY: This PR cleans up the many copies & reallocations of outputs in the ROCm attention backend. This is needed for the attention+quant fusion described in #16220.

### L3-471fe65630  (L3, 2025-04-21, sha 471fe65630e0, PR #16871)
TITLE: [TPU][V1] Implicitly adjust page size when there's SMEM OOM (#16871)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+15/-0); tests/v1/tpu/test_basic.py (+5/-2); vllm/platforms/tpu.py (+14/-0)
LABELS: tpu, ready, v1
BODY: 

### L3-fe3462c774  (L3, 2025-04-22, sha fe3462c77492, PR #15591)
TITLE: [XPU][Bugfix] minor fix for XPU (#15591)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/ipex_attn.py (+6/-6); docs/source/getting_started/installation/gpu/xpu.inc.md (+2/-0)
LABELS: documentation
BODY: 

### L3-986537f1c3  (L3, 2025-04-22, sha 986537f1c3c8, PR #16684)
TITLE: [V1] V1 FlashInfer Attention (#16684)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:production-kernel-provenance
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.platform.cuda_selection
FILES: vllm/engine/arg_utils.py (+10/-3); vllm/platforms/cuda.py (+3/-0); vllm/v1/attention/backends/flash_attn.py (+3/-4); vllm/v1/attention/backends/flashinfer.py (+639/-0); vllm/v1/attention/backends/mla/common.py (+3/-4); vllm/v1/worker/gpu_model_runner.py (+1/-1); tests/v1/e2e/test_cascade_attention.py (+9/-1)
LABELS: ready, v1
BODY: Carrying on @aurickq work from here https://github.com/vllm-project/vllm/pull/14061. Thanks to @LucasWilkinson for helping debug qo_indptr issues. ⏎  ⏎ There are some performance issues in the original PR due to using `BatchPrefillWithPagedKVCacheWrapper` for all prefill and decode tokens. This PR separates prefill and decode tokens in V1 using the `reorder_batch()` functionality added for MLA, where the requests in the `input_batch` is reshuffled su …[truncated]

### L3-0e237f0035  (L3, 2025-04-22, sha 0e237f00357c, PR #15001)
TITLE: [FEAT][ROCm] Integrate Paged Attention Kernel from AITER (#15001)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen, L3.rocm.rocm_flash_attn_v0, L3.platform.rocm_selection
FILES: docker/Dockerfile.rocm_base (+1/-1); vllm/attention/backends/rocm_flash_attn.py (+84/-25); vllm/attention/ops/rocm_aiter_paged_attn.py (+101/-0); vllm/envs.py (+7/-0); vllm/platforms/rocm.py (+2/-1)
LABELS: ready, ci/build
DEEP_STUDY: deep-study: this PR was reverted by PR 17022 (partial_revert, reason=api_or_compat_break)
BODY: # This PR integrates Paged Attention Kernel from AITER (AI Tensor Engine for ROCm) ⏎  ⏎ The `pa_fwd_asm` kernel from AITER is integrated as a new paged attention op in `/vllm/attention/ops/rocm_aiter_paged_attn.py` and implemented into the ROCM attention backend in `/vllm/attention/backends/rocm_flash_attn.py`. ⏎  ⏎ This feature is disabled by default, even when the parent switch (`VLLM_ROCM_USE_AITER=1`) is enabled. To use this kernel, both the parent s …[truncated]

### L3-f961d7f6ef  (L3, 2025-04-22, sha f961d7f6ef14, PR #16973)
TITLE: [BugFix] Pass in correct VLLM config in FlashInfer backend (#13207) (#16973)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+3/-3)
LABELS: ready
ISSUES: #13207 [Bug]: VLLM config not set when using Flash Infer backend.
BODY: This PR fixes the issue of the flashinfer backend complaining "Current VLLM config is not set" during CUDA graph capturing and closes #13207. The fix follows the proposed method in that issue.

### L3-30bc3e0f66  (L3, 2025-04-22, sha 30bc3e0f665e, PR #15893)
TITLE: [FEAT][ROCm]: Support AITER MLA (#15893)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.triton_v0, L3.platform.rocm_selection
FILES: vllm/attention/backends/mla/common.py (+18/-3); vllm/attention/backends/rocm_aiter_mla.py (+412/-0); vllm/attention/ops/rocm_aiter_mla.py (+42/-0); vllm/config.py (+1/-1); vllm/envs.py (+6/-0); vllm/platforms/interface.py (+1/-0); vllm/platforms/rocm.py (+31/-3); tests/kernels/test_attention_selector.py (+128/-21); tests/kernels/test_rocm_attention_selector.py (+28/-1)
LABELS: ready, ci/build
BODY: # Description ⏎ This PR integrates the AITER ops to improve the MLA functionality from [AITER flash_attn_varlen_func](https://github.com/ROCm/aiter/blob/86256916eee13a94c2213be2b7fde27a145d7103/aiter/ops/mha.py#L1155) and [AITER mla_decode_fwd](https://github.com/ROCm/aiter/blob/86256916eee13a94c2213be2b7fde27a145d7103/aiter/mla.py#L74) into vLLM, and will allow any up-coming optimizations in AITER kernel to be directly used and evaluated within th …[truncated]

### L3-f67e9e9f22  (L3, 2025-04-22, sha f67e9e9f221e, PR #16936)
TITLE: add Dockerfile build vllm against torch nightly (#16936)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.nightly_torch (+307/-0); requirements/nightly_torch_test.txt (+28/-0); .buildkite/test-pipeline.yaml (+3/-0)
LABELS: ci/build
BODY: Add Dockerfile that build vllm against nightly torch, we only install test packages that covers necessary tests in this pr: ⏎ - entrypoint test ⏎ - correctness test  ⏎  ⏎ Addtional install: ⏎ 1. install xformer from source (otherwise it will override torch version and not compatiable with torch 2.8.0) ⏎ 2. install flashinfer from source  ⏎  ⏎ related pr in CI-Infra to introduce the dockerfile: ⏎ https://github.com/vllm-project/ci-infra/pull/87/files ⏎  ⏎ Testing Built …[truncated]

### L3-bc7c4d206b  (L3, 2025-04-22, sha bc7c4d206bbf, PR #13305)
TITLE: [Kernel][ROCM] Upstream prefix prefill speed up for vLLM V1 (#13305)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/prefix_prefill.py (+821/-813); tests/core/block/e2e/test_correctness.py (+3/-3)
LABELS: rocm, ready, ci/build, v1
BODY: Speed up prefix prefill with vLLM V1 on AMG GPUs ⏎  ⏎ Improvements: ⏎ 1. Vectorization in the context loop (most complex one as k cache shape is very specific) ⏎ 2. Refactoring for online softmax computation ⏎ 3. Refactoring to the kernel so autotune might select the best configs per shape ⏎ 4. Plus adding new spectrum of unrolling/staging in autotuner ⏎  ⏎ More details on triton kernel tunning: https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-opti …[truncated]

### L3-7e081ba7ca  (L3, 2025-04-22, sha 7e081ba7cad2, PR #17022)
TITLE: [BugFix] Revert ROCm Custom Paged Attention Env Flag Check (#17022)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+1/-0)
DEEP_STUDY: deep-study revert record: partial_revert of PR(s) 15001 reason=api_or_compat_break
BODY: #15001 Removes `envs.VLLM_ROCM_CUSTOM_PAGED_ATTN` from the `use_rocm_custom_paged_attention` check. This might cause vllm to use rocm custom paged attention even when the flag is not set. This PR reverts the check to maintain correctness.

### L3-047797ef90  (L3, 2025-04-22, sha 047797ef904f, PR #16902)
TITLE: [Bugfix] Triton FA function takes no keyword arguments (#16902)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.triton_v0
FILES: vllm/attention/backends/mla/common.py (+8/-1)
LABELS: rocm, ready
BODY: This PR resolves an existing bug in serving Deepseek model using MLA backend with triton flash attention when running the command below: ⏎  ⏎ `VLLM_MLA_DISABLE=0 VLLM_ATTENTION_BACKEND=TRITON_MLA VLLM_USE_TRITON_FLASH_ATTN=1 vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --swap-space 16 --disable-log-requests -tp 8` ⏎  ⏎ Throws the error below: ⏎  ⏎ ""ERROR 04-21 04:42:55 [engine.py:448]   File "/app/vllm/vllm/attention/backends/mla/common.py", line  …[truncated]

### L3-d0da99fb70  (L3, 2025-04-22, sha d0da99fb70ba, PR #16998)
TITLE: [BugFix] llama4 fa3 fix - RuntimeError: scheduler_metadata must have shape (metadata_size) (#16998)
SOURCES: path_core, subject_keyword, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+48/-28)
LABELS: ready, v1
ISSUES: #16948 [Bug]: Run Llama4 Scout 16E w/ 10000 input length trigger vllm crashing, but run fine if use FA2. | #16997 [Bug]: Qwen2.5-VL-72B Inference
DEEP_STUDY: deep-study correctness case vllm:d0da99fb70: class=shape_alignment_edge; symptom=crash_or_exception; introducing=#13111
BODY: FIX https://github.com/vllm-project/vllm/issues/16948 ⏎ FIX #16997 ⏎  ⏎ Cause by: https://github.com/vllm-project/vllm/pull/13111 ⏎  ⏎ `+` fix for numheads (did not affect accuracy)

### L3-47bdee409c  (L3, 2025-04-24, sha 47bdee409c92, PR #17026)
TITLE: Molmo Requirements (#17026)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: requirements/molmo.txt (+20/-0); docs/source/models/supported_models.md (+4/-0)
LABELS: documentation, ready, ci/build
BODY: Add Molmo-specific requirements file for stable accuracy on different architectures. ⏎ Without these requirements user installs latest libs which doesnt provide same accuracy as it would on required libs.

### L3-41ca7eb491  (L3, 2025-04-24, sha 41ca7eb49192, PR #16864)
TITLE: [Attention] FA3 decode perf improvement - single mma warp group support for head dim 128 (#16864)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build, v1
BODY: vLLM side of https://github.com/vllm-project/flash-attention/pull/63 ⏎  ⏎ Perf Results: ⏎  ⏎ 1000 in / 100 out ⏎  ⏎ https://docs.google.com/spreadsheets/d/1r7Hdgy1OGK7tU9DD4QWVt8IlA-688h0mHlYFeoD6Jtk/edit?usp=sharing ⏎  ⏎ 4xH100 ⏎ ![meta-llama_Llama-4-Scout-17B-16E](https://github.com/user-attachments/assets/d4bf3170-8f4e-4d15-b92c-37c1abe1c031) ⏎  ⏎ 1xH100 ⏎ ![mistralai_Mistral-Small-24B-Instruct-2501](https://github.com/user-attachments/assets/ffea227a-a3da-4aa7-a79b- …[truncated]

### L3-a41351f363  (L3, 2025-04-25, sha a41351f363f3, PR #15734)
TITLE: [Quantization][FP8] Add support for FP8 models with input_scale for output projection and QK quantization (#15734)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+1/-0); vllm/attention/backends/rocm_flash_attn.py (+7/-0); vllm/attention/layer.py (+1/-0); vllm/config.py (+11/-0); vllm/engine/arg_utils.py (+17/-0); vllm/model_executor/layers/quantization/fp8.py (+5/-0); vllm/model_executor/layers/quantization/kv_cache.py (+36/-0); vllm/model_executor/layers/quantization/quark/quark.py (+27/-20)
LABELS: ready
BODY: This PR adds fp8 quantization support for: ⏎  ⏎ 1) Preserving FP8 quantization after FA output, so that output of FA into the next layer will be FP8 ⏎      - this uses the model parameter self_attn.q/k/v_proj.input_scale and passes it as _out_scale to the FA kernel ⏎ 2) During execution of FA loop, the quantity softmax(QK^T) can also be quantized as FP8 ⏎     - this uses the model parameter self_attn.prob_output_scale ⏎  ⏎ These are using Quark quantized model …[truncated]

### L3-b22980a1dc  (L3, 2025-04-25, sha b22980a1dc89, PR #16457)
TITLE: [Perf]Optimize rotary_emb implementation to use Triton operator for improved inference performance (#16457)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/model_executor/layers/rotary_embedding.py (+23/-11)
LABELS: ready, ci/build
BODY: Optimize Rotary Positional Embeddings with Triton Kernel in VLLM ⏎  ⏎ This PR enhances rotary positional embedding computation by leveraging Triton-optimized kernels from flash_attn, addressing a significant performance bottleneck observed in models like Qwen2-VL. ⏎  ⏎ Background ⏎ The original PyTorch-native rotary embedding implementation (rotary_emb) consumed 40-60% of total inference latency for Qwen2-VL, particularly scaling with output token count. P …[truncated]

### L3-9d98ab5ec6  (L3, 2025-04-25, sha 9d98ab5ec60a, PR #17190)
TITLE: [Misc] Inline Molmo requirements (#17190)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: requirements/molmo.txt (+0/-20); docs/source/models/supported_models.md (+27/-1)
LABELS: documentation, ready, ci/build
BODY: This avoids false-positive security flags due to having old libraries in requirements directory.

### L3-9e96f56efb  (L3, 2025-04-25, sha 9e96f56efb5b, PR #16605)
TITLE: Allocate kv_cache with stride order (#16605)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flashinfer.v0_backend, L3.dispatch.abstract_interface
FILES: csrc/cache_kernels.cu (+21/-18); vllm/attention/backends/abstract.py (+4/-0); vllm/attention/backends/flashinfer.py (+25/-5); tests/kernels/attention/test_cache.py (+41/-19); vllm/utils.py (+10/-3); vllm/worker/cache_engine.py (+18/-5)
LABELS: ready
BODY: Allow KV cache manager to support an stride order to the allocation which the attention backend could provide. Mainly affect Flashinfer backend. Ref. #8200  ⏎ @tlrmchlsmth @LucasWilkinson

### L3-c48334d405  (L3, 2025-04-26, sha c48334d405d6, PR #17186)
TITLE: [Hardware][Intel-Gaudi] Update hpu-extension and update bucketing system for HPU device (#17186)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/hpu_attn.py (+54/-52); vllm/attention/ops/hpu_paged_attn.py (+0/-1); requirements/hpu.txt (+1/-1); vllm/model_executor/layers/layernorm.py (+2/-1); vllm/worker/hpu_model_runner.py (+70/-280); vllm/worker/hpu_worker.py (+1/-0)
LABELS: ci/build
BODY: Many bucketing mechanisms were moved to external vllm-hpu-extension repository. This PR updates sha for vllm-hpu-extension and resolves code mismatches.

### L3-65e262b93b  (L3, 2025-04-26, sha 65e262b93bef, PR #17159)
TITLE: Fix Python packaging edge cases (#17159)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/vllm_flash_attn/__init__.py (+0/-0); .gitignore (+1/-0); pyproject.toml (+1/-2); vllm/benchmarks/__init__.py (+0/-0)
LABELS: ready
ISSUES: #15812 [Bug]: run on cpu:  ModuleNotFoundError: No module named 'vllm.benchmarks'
BODY: Address some edge cases with packaging of Python subpackages in vLLM. The `vllm/benchmark` and `vllm/vllm_flash_attn` directories were missing `__init__.py` files. A subdirectory without an `__init__.py` is considered a namespace package. Because vLLM's `pyproject.toml` file excludes namespace packages, setuptools' package finder excludes the subpackages `vllm.benchmark` and `vllm.vllm_flash_attn` from wheel distributions. ⏎  ⏎ vLLM's wheels still sh …[truncated]

### L3-e782e0a170  (L3, 2025-04-26, sha e782e0a170a6, PR #17228)
TITLE: [Chore] added stubs for `vllm_flash_attn` during development mode (#17228)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fa4_cutedsl
FILES: pyproject.toml (+2/-1); setup.py (+0/-1); vllm/vllm_flash_attn/__init__.py (+22/-0); vllm/vllm_flash_attn/flash_attn_interface.pyi (+245/-0)
LABELS: ready, ci/build
BODY: With #17159, whenever vLLM is installed with VLLM_USE_PRECOMPILED, ⏎ it will inadvertently copy the `__init__.py` from `vllm_flash_attn`, which will generate bad diff ⏎  ⏎ This PR removes the `__init__.py` copy in `setup.py`, and add a function stubs for all functions within the `flash_attn_interface.py`

### L3-9869453c42  (L3, 2025-04-26, sha 9869453c42b8, PR #17102)
TITLE: Update test_flash_attn.py (#17102)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_flash_attn.py (+1/-1)
LABELS: ready
BODY: fix flash_attn_fp8 test

### L3-8e4b351a0c  (L3, 2025-04-27, sha 8e4b351a0c9e, PR #12591)
TITLE: [Kernel][Triton][FP8] Adding fp8 and variable length sequence support to Triton FAv2 kernel (#12591)
SOURCES: path_core, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L3.triton.flash_attention_rocm
FILES: vllm/attention/ops/triton_flash_attention.py (+1158/-604); tests/kernels/test_triton_flash_attention.py (+499/-0)
LABELS: ready
DEEP_STUDY: deep-study: this PR was reverted by PR 18226 (explicit_rollback, reason=performance_regression)
BODY: This PR adds fp8 and variable length sequence support to Triton FAv2 kernel. ⏎  ⏎ This kernel supports 8-bit KV cache, and also the following (forward only): ⏎  ⏎ 1) Fwd with causal masking ⏎ 2) Arbitrary Q and KV sequence lengths ⏎ 3) Arbitrary head sizes ⏎ 4) Multi and grouped query attention ⏎ 5) Variable sequence lengths ⏎ 6) ALiBi and matrix bias ⏎ 7) Supports fp8 for models, currently for Llama-3.1-8B-Instruct-FP8-QKV-Prob ⏎  ⏎ This kernel is slightly faster than  …[truncated]

### L3-838cedade7  (L3, 2025-04-27, sha 838cedade77a, PR #17222)
TITLE: [Bugfix] Get a specific type of layer from forward context (#17222)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/attention/backends/flashinfer.py (+2/-4); vllm/v1/attention/backends/flashinfer.py (+3/-4); vllm/config.py (+15/-1); vllm/v1/worker/gpu_model_runner.py (+5/-10); vllm/v1/worker/tpu_model_runner.py (+3/-4)
LABELS: tpu, ready, v1
DEEP_STUDY: deep-study correctness case vllm:838cedade7: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: Forward context saves both `Attention` and `FusedMOE` layers. Add an interface to only get one type of layer from forward context. ⏎  ⏎ Fix bug: ⏎ `VLLM_ATTENTION_BACKEND=FLASHINFER python3 examples/offline_inference/data_parallel.py --model="Qwen/Qwen1.5-MoE-A2.7B-Chat" --dp-size=2 --tp-size=1` ⏎ ``` ⏎ (EngineCore_0 pid=3081205)   File “***/vllm/v1/attention/backends/flashinfer.py", line 88, in get_per_layer_parameters ⏎ (EngineCore_0 pid=3081205)     asser …[truncated]

### L3-ed7a29d9f8  (L3, 2025-04-27, sha ed7a29d9f8b4, PR #16032)
TITLE: [NVIDIA] Support Cutlass MLA for Blackwell GPUs (#16032)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.mla.cutlass_kernels
FILES: CMakeLists.txt (+24/-4); csrc/attention/mla/cutlass_mla_entry.cu (+38/-0); csrc/attention/mla/cutlass_mla_kernels.cu (+225/-0); csrc/ops.h (+6/-0); csrc/torch_bindings.cpp (+7/-0); vllm/_custom_ops.py (+9/-0); csrc/quantization/fp4/nvfp4_scaled_mm_kernels.cu (+1/-1); tests/kernels/test_cutlass_mla_decode.py (+93/-0)
LABELS: ready, ci/build
BODY: The latest cutlass supports MLA for the blackwell GPUs. Examples can be found [here](https://github.com/NVIDIA/cutlass/blob/main/examples/77_blackwell_fmha/77_blackwell_mla.cu). It should be available in the next release (v3.9). ⏎  ⏎ This PR integrates this kernel as `ops.cutlass_mla_decode`. ⏎  ⏎ cc. @kushanam

### L3-690fe019f0  (L3, 2025-04-27, sha 690fe019f046, PR #16155)
TITLE: [Feature] support sequence parallelism using compilation pass (#16155)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test-pipeline.yaml (+3/-0); tests/compile/test_functionalization.py (+8/-6); tests/compile/test_fusion.py (+5/-4); tests/compile/test_pass_manager.py (+5/-4); tests/compile/test_sequence_parallelism.py (+190/-0); tests/distributed/test_comm_ops.py (+30/-1); tests/distributed/test_sequence_parallel.py (+296/-0); vllm/compilation/backends.py (+1/-1); vllm/compilation/compiler_interface.py (+8/-5); vllm/compilation/fusion.py (+4/-4); (+11 more)
LABELS: ready, ci/build, v1
BODY: This PR support sequence parallelism using below compilation config ⏎ ``` ⏎ config = CompilationConfig( ⏎     level=3, ⏎     custom_ops=["+rms_norm"], ⏎     compile_sizes=[4, 8, 16], ⏎     splitting_ops=[], ⏎ ) ⏎ config.pass_config.enable_sequence_parallelism= True ⏎  ⏎ llm = LLM(model="llama/Llama-3.2-1B-Instruct", ⏎           enforce_eager=False, ⏎           tensor_parallel_size=2, ⏎           dtype=torch.float16, ⏎           max_num_batched_tokens=2048, ⏎           compila …[truncated]

### L3-d8bccde686  (L3, 2025-04-27, sha d8bccde68634, PR #17267)
TITLE: [BugFix] Fix vllm_flash_attn install issues (#17267)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flash_attn.upstream_pip, L3.flash_attn.fa4_cutedsl, L3.flash_attn.fa_utils, L3.mla.triton_v0, L3.mla.common_v1
FILES: setup.py (+19/-7); vllm/attention/backends/flash_attn.py (+3/-3); vllm/attention/backends/mla/common.py (+1/-1); vllm/attention/utils/fa_utils.py (+0/-0); vllm/engine/arg_utils.py (+1/-1); vllm/v1/attention/backends/flash_attn.py (+2/-2); vllm/v1/attention/backends/mla/common.py (+1/-1); vllm/vllm_flash_attn/__init__.py (+0/-22); vllm/vllm_flash_attn/flash_attn_interface.pyi (+0/-245); .github/CODEOWNERS (+1/-0); (+1 more)
LABELS: ready, ci/build, v1
ISSUES: #17263 [Bug]: nightly version: ModuleNotFoundError: No module named 'vllm.vllm_flash_attn.layers'
BODY: This PR is a collection of fixes for `vllm_flash_attn` install issues, unfortunately the `vllm_flash_attn` install is fairly hacky/sensitive/complex right now. This will hopefully be fixed in the future if we move to a separate kernel library. Apologies for missing https://github.com/vllm-project/vllm/pull/17159 . ⏎  ⏎ 1) https://github.com/vllm-project/vllm/pull/17159 introduced an `__init__.py` into the `vllm_flash_attn`, during the install process …[truncated]

### L3-cc5befbced  (L3, 2025-04-28, sha cc5befbced22, PR #17283)
TITLE: [BugFix] Fix cascade attention - RuntimeError: scheduler_metadata must have shape (metadata_size) (#17283)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+1/-1)
LABELS: ready, v1
ISSUES: #17276 [Bug]: nightly version:  EngineCore encountered a fatal error.
BODY: FIX https://github.com/vllm-project/vllm/issues/17276

### L3-17eb306fcc  (L3, 2025-04-28, sha 17eb306fcc70, PR #17091)
TITLE: [Bugfix] Add contiguous call inside rope kernel wrapper (#17091)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+3/-4); vllm/_custom_ops.py (+14/-3)
LABELS: ready, v1
ISSUES: #16658 [Bug]: V0 engines gives incorrect output for Moonlight model
BODY: This PR fixes #16658.  ⏎  ⏎ Following @LucasWilkinson's suggestion, I followed the second way: add `.contiguous()` call in the kernel wrappers. Although making the query contiguous is sufficient for RoPE in the MLA backend, I found that a non-contiguous key is also likely to lead to the same issue because the stride along the first dimension is not considered in the kernel. Therefore, I make both query and key contiguous in the wrappers.  ⏎  ⏎ I also add …[truncated]

### L3-24e6ad3f16  (L3, 2025-04-29, sha 24e6ad3f16d5, PR #17193)
TITLE: [V1] Remove num_input_tokens from attn_metadata (#17193)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1
FILES: vllm/v1/attention/backends/flash_attn.py (+0/-3); vllm/v1/attention/backends/flashinfer.py (+0/-3); vllm/v1/attention/backends/mla/common.py (+0/-3); vllm/forward_context.py (+7/-9); vllm/v1/worker/gpu_model_runner.py (+3/-2); vllm/v1/worker/tpu_model_runner.py (+4/-1)
LABELS: tpu, ready, v1
BODY: `num_input_tokens` is not related to attention and only used in `set_forward_context`, so I prefer to remove it from attention_metadata and pass it to set_forward_context explicitly.

### L3-06ffc7e1d3  (L3, 2025-04-29, sha 06ffc7e1d35b, PR #17289)
TITLE: [Misc][ROCm] Exclude `cutlass_mla_decode` for ROCm build (#17289)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: csrc/torch_bindings.cpp (+7/-7)
LABELS: ready
BODY: A small fix for https://github.com/vllm-project/vllm/pull/16032. Do not include the op `cutlass_mla_decode` for ROCm build; otherwise it will cause some import error.

### L3-2c4f59afc3  (L3, 2025-04-29, sha 2c4f59afc3d5, PR #16859)
TITLE: Update PyTorch to 2.7.0 (#16859)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+2/-2); docker/Dockerfile (+32/-14); requirements/build.txt (+1/-1); requirements/cpu.txt (+6/-5); requirements/cuda.txt (+5/-4); requirements/rocm-build.txt (+3/-3); requirements/test.in (+3/-3); requirements/test.txt (+24/-20); setup.py (+1/-1); vllm/attention/ops/ipex_attn.py (+2/-1); (+8 more)
LABELS: documentation, ci/build
BODY: Notable changes: ⏎  ⏎ * PyTorch 2.7.0 has dropped CUDA 12.4, so the remaining options are 12.6 and 12.8  ⏎     * CUDA 12.6 has build issue https://github.com/vllm-project/vllm/issues/15435#issuecomment-2775924628, so only 12.8 remains ⏎ * We need a new ~~xformers~~ (0.0.30 is ready now), flashinfer, and mamba-ssm packages, so let build them from source for now.  They can be installed from pypi once they are built upstream with 2.7.0 ⏎ * Leave XPU for later …[truncated]

### L3-ed6cfb90c8  (L3, 2025-04-30, sha ed6cfb90c8ad, PR #17444)
TITLE: [Hardware][Intel GPU] Upgrade to torch 2.7 (#17444)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: vllm/attention/backends/ipex_attn.py (+6/-8); docker/Dockerfile.xpu (+0/-6); docs/source/getting_started/installation/gpu/xpu.inc.md (+0/-9); requirements/xpu.txt (+3/-3); vllm/_ipex_ops.py (+9/-9)
LABELS: documentation, ci/build
BODY: 

### L3-90d0a54c4d  (L3, 2025-04-30, sha 90d0a54c4dae, PR #17229)
TITLE: [ROCm] Effort to reduce the number of environment variables in command line (#17229)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm (+9/-0)
LABELS: rocm, ready, ci/build
BODY: This is to set two environment variables in the Docker file so that users can reduce the number of environment variables when running scripts. ⏎  ⏎ ENV that can improve safe tensor loading, and end-to-end time ⏎ ``` ⏎ ENV SAFETENSORS_FAST_GPU=1 ⏎ ``` ⏎ ENV that needed for multi-process on cuda-like platform ⏎ ``` ⏎ ENV VLLM_WORKER_MULTIPROC_METHOD=spawn ⏎ ``` ⏎  ⏎ Test: ⏎ 1. build the docker image ⏎ ``` ⏎ DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t vllm-rocm …[truncated]

### L3-2007d4d54f  (L3, 2025-05-01, sha 2007d4d54f8f, PR #17530)
TITLE: [FEAT] [ROCm]: Add Qwen/Qwen3-30B-A3B-FP8 fused moe config for MI300X (#17530)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json (+164/-0)
BODY: This PR add a tuned fused moe config for Qwen/Qwen3-30B-A3B-FP8 TP1 ⏎  ⏎ Benchmark command:  ⏎ `VLLM_ROCM_USE_AITER=0 VLLM_USE_TRITON_FLASH_ATTN=0 python3 benchmarks/benchmark_throughput.py --input-len $_in_len --output-len $_out_len --trust-remote-code --num-prompts 200 --model Qwen/Qwen3-30B-A3B-FP8 --max-model-len 32768 --gpu_memory_utilization 0.95 --tensor-parallel-size 1 --max_seq_len_to_capture 32768 --quantization fp8 --kv-cache-dtype fp8` ⏎  ⏎ | I …[truncated]

### L3-f5a3c655b2  (L3, 2025-05-01, sha f5a3c655b2e0, PR #17535)
TITLE: [FEAT] [ROCm]: Add Qwen/Qwen3-235B-A22B-FP8 TP4 triton fused moe config (#17535)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/configs/E=128,N=384,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json (+164/-0)
BODY: Add Qwen/Qwen3-235B-A22B-FP8 TP4 triton fused moe config ⏎  ⏎ `VLLM_ROCM_USE_AITER=0 VLLM_USE_TRITON_FLASH_ATTN=0 python3 benchmarks/benchmark_throughput.py --input-len $_in_len --output-len $_out_len --trust-remote-code --num-prompts 200 --model Qwen/Qwen3-235B-A22B-FP8 --max-model-len 32768 --gpu_memory_utilization 0.95 --tensor-parallel-size 4 --max_seq_len_to_capture 32768 --quantization fp8 --kv-cache-dtype fp8` ⏎  ⏎ | Input Tokens | Output Tokens | …[truncated]

### L3-28566d73b3  (L3, 2025-05-01, sha 28566d73b3c7, PR #17536)
TITLE: [ROCm] remove unsupported archs from rocm triton flash-attention supported list (#17536)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.flash_attention_rocm
FILES: vllm/attention/ops/triton_flash_attention.py (+1/-1)
LABELS: rocm, ready
BODY: gfx940 and gfx941 are no longer supported since ROCm 6.3, and also they are not in the PyTorch arch list (env PYTORCH_ROCM_ARCH) . This simple clean up PR is to remove them from the code (here the triton flash-attention supported list).

### L3-811a6c0972  (L3, 2025-05-01, sha 811a6c0972da, PR #16034)
TITLE: [ROCM] Add gfx950 to the custom attention archs (#16034)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.custom_paged, L3.platform.rocm_selection
FILES: csrc/rocm/attention.cu (+6/-5); vllm/platforms/rocm.py (+6/-3)
LABELS: ready
BODY: Adding gfx950 to the supported architectures and small renaming of the variable

### L3-3c3d767201  (L3, 2025-05-01, sha 3c3d76720156, PR #17494)
TITLE: [BugFix] Fix mla cpu - missing 3 required positional arguments (#17494)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/cpu_mla.py (+3/-0); vllm/_ipex_ops.py (+3/-1)
LABELS: bug, ready
BODY: Fix  ⏎ ``` ⏎ [rank0]: TypeError: ipex_ops.varlen_attention() missing 3 required positional arguments: 'alibi_slopes', 'window_size_left', and 'window_size_right' ⏎ ``` ⏎ caused by  ⏎ https://github.com/vllm-project/vllm/pull/17444 ⏎  ⏎ cc @gau-nernst @jikunshang ⏎  ⏎ ``` ⏎ (vllm-cpu) lwilkinson@beaker:~/code/vllm$ python3 examples/offline_inference/basic/generate.py --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code --max-model-len 1024 --block-size 16 ⏎ [W …[truncated]

### L3-cc2a77d7f1  (L3, 2025-05-02, sha cc2a77d7f1df, PR #15428)
TITLE: [Core] [Bugfix] Add Input Embeddings (#15428)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+11/-3); tests/conftest.py (+10/-8); tests/core/test_scheduler.py (+73/-1); tests/core/utils.py (+9/-2); tests/models/language/generation/test_common.py (+26/-0); tests/worker/test_model_runner.py (+109/-34); vllm/core/scheduler.py (+32/-0); vllm/engine/async_llm_engine.py (+8/-0); vllm/engine/llm_engine.py (+13/-3); vllm/engine/output_processor/multi_step.py (+4/-2); (+12 more)
LABELS: frontend, speculative-decoding, ready
ISSUES: #416 [Feature Request] Support input embedding in `LLM.generate()` | #8323 Do vLLM support `input_embeds` as input while using LLama? | #14621 [Usage]:  how to use embeddings as input rather than token_ids
BODY: > [!NOTE] ⏎ > This PR is just #11684, but rebased onto main and then with pre-commit errors fixed, since it has been some time since @Bryce1010 has updated that PR. ⏎  ⏎ Adds support for passing prompt_embeds to LLM.generate as ⏎  ⏎ ``` ⏎ llm.generate({"prompt_embeds": input_embeds}, sampling_params) ⏎ ``` ⏎ or ⏎ ``` ⏎ llm.generate( ⏎     [{"prompt_embeds": input_embeds} for input_embeds in inputs_embeds], sampling_params ⏎ ) ⏎ ``` ⏎ this enables use cases when only the emb …[truncated]

### L3-afcb3f8863  (L3, 2025-05-02, sha afcb3f8863ee, PR #17484)
TITLE: [Attention] MLA move o_proj q_proj into cuda-graph region (#17484)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.triton_v0, L3.mla.common_v1, L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter
FILES: vllm/attention/backends/cpu_mla.py (+2/-3); vllm/attention/backends/flashmla.py (+1/-1); vllm/attention/backends/mla/common.py (+19/-35); vllm/attention/backends/rocm_aiter_mla.py (+1/-1); vllm/attention/backends/triton_mla.py (+1/-1); vllm/model_executor/models/deepseek_v2.py (+12/-9); vllm/v1/attention/backends/mla/common.py (+17/-33); vllm/v1/attention/backends/mla/flashmla.py (+1/-1); vllm/v1/attention/backends/mla/triton_mla.py (+1/-1)
LABELS: ready, v1
BODY: With: https://github.com/vllm-project/vllm/pull/14770 we are no longer materializing the absorbed matrices so we can move q_proj and o_proj into the cuda-graph region lowering CPU overhead: ⏎  ⏎ # Accuracy ⏎  ⏎ ``` ⏎ VLLM_USE_V1=0 lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384 --task gsm8k --num_fewshot 5  --batch_size aut …[truncated]

### L3-0f87d8f7b2  (L3, 2025-05-02, sha 0f87d8f7b26d, PR #17574)
TITLE: [BugFix][Attention] Fix sliding window attention in V1 giving incorrect results (#17574)
SOURCES: path_core, symbol_pickaxe, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+35/-1)
LABELS: ready, v1
ISSUES: #17476 [Bug]: Flash attention with sliding window
DEEP_STUDY: deep-study correctness case vllm:0f87d8f7b2: class=shape_alignment_edge; symptom=wrong_output_or_accuracy; introducing=#13111
BODY: The FA3 update to use a AOT scheduler (https://github.com/vllm-project/vllm/pull/13111) did not properly handle sliding window attention. FIX https://github.com/vllm-project/vllm/issues/17476 ⏎  ⏎ Tested using: ⏎ ``` ⏎ python -m pytest -vs tests/v1/e2e/test_correctness_sliding_window.py ⏎ ```

### L3-3e887d2e0c  (L3, 2025-05-02, sha 3e887d2e0c1f, PR #14568)
TITLE: permute/unpermute kernel for moe optimization (#14568)
SOURCES: corpus:kernel-correctness-cases(introducing)
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+13/-1); benchmarks/kernels/benchmark_grouped_gemm_cutlass.py (+2/-1); benchmarks/kernels/benchmark_moe.py (+2/-2); benchmarks/kernels/benchmark_moe_permute_unpermute.py (+349/-0); csrc/moe/moe_permute_unpermute_op.cu (+133/-0); csrc/moe/permute_unpermute_kernels/dispatch.h (+53/-0); csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu (+229/-0); csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.h (+95/-0); csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.inl (+211/-0); csrc/moe/torch_bindings.cpp (+22/-0); (+9 more)
LABELS: ready, ci/build
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:6e588da0f4 (fix PR 17679)
BODY: `moe_permute` kernel expands and oreders token in activation to gather uncontinuous tokens for each expert. And then call grouped-gemm for moe speedup. ⏎ `moe_unpermute` kernel reduces  expanded grouped-gemm output and scales with `topk_weight`. ⏎ <img width="646" alt="image" src="https://github.com/user-attachments/assets/4b6b7691-8772-474a-b803-ad3a34c6417b" /> ⏎  ⏎ This implementation refers to  moe kernel in  tensorrt-llm in archive https://github.co …[truncated]

### L3-4c33d67321  (L3, 2025-05-02, sha 4c33d6732148, PR #17438)
TITLE: [Bugfix] fix tmp_out and exp_sums dimensions (#17438)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.chunked_prefill_paged_decode
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+1/-1)
LABELS: bug, ready
BODY: The first dimension of tmp_out and exp_sums is inferred from block_tables.size(0), which may be different from query.shape(0). The later can be much larger than block_tables.size(0), which may cause OOM. ⏎  ⏎ This PR fix the total_num_seq and the comments.

### L3-c8386fa61d  (L3, 2025-05-02, sha c8386fa61d97, PR #17602)
TITLE: [Build/CI] Upgrade CUTLASS to 3.9.1 (#17602)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+3/-4)
LABELS: ready, ci/build
BODY: From the [release notes](https://github.com/NVIDIA/cutlass/releases/tag/v3.9.1): ⏎  ⏎ > * Fixed Group Gemm hang issue in CUTLASS 3.x ⏎ > * Improved Hopper [Blockwise](https://github.com/NVIDIA/cutlass/blob/v3.9.1/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu) and [Groupwise](https://github.com/NVIDIA/cutlass/blob/v3.9.1/examples/67_hopper_fp8_warp_specialized_gemm_with_ …[truncated]

### L3-d6484ef3c3  (L3, 2025-05-03, sha d6484ef3c3a0, PR #17485)
TITLE: Add full API docs and improve the UX of navigating them (#17485)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.triton_v0, L3.mla.common_v1
FILES: vllm/attention/backends/mla/common.py (+2/-0); vllm/attention/backends/utils.py (+1/-1); .buildkite/test-pipeline.yaml (+1/-1); .gitignore (+1/-0); docs/Makefile (+1/-0); docs/source/api/engine/async_llm_engine.md (+0/-7); docs/source/api/engine/index.md (+0/-17); docs/source/api/engine/llm_engine.md (+0/-7); docs/source/api/inference_params.md (+0/-21); docs/source/api/model/adapters.md (+0/-9); (+91 more)
LABELS: documentation, frontend, speculative-decoding, ready, ci/build, v1, multi-modality
BODY: Changes: ⏎ - Ported from `sphinx.ext.autodoc` and `sphinx.ext.autosummary` to `autodoc2`: ⏎   - `autodoc2` supports `myst` (markdown) so we can stop writing `rst` formatted docstrings  ⏎   - The _entire_ codebase is documented in API Reference ⏎   - All of these docs can be cross-referenced throughout the docs ⏎ - Updated the API Reference index to include all the important classes/functions we already singled out on the individual pages we had before (red …[truncated]

### L3-cba31c47c4  (L3, 2025-05-06, sha cba31c47c481, PR #17394)
TITLE: [v1] AttentionMetadata for each layer (#17394)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.dispatch.abstract_interface
FILES: vllm/attention/layer.py (+12/-3); vllm/v1/attention/backends/flash_attn.py (+5/-6); vllm/v1/attention/backends/flashinfer.py (+5/-5); vllm/v1/attention/backends/mla/common.py (+5/-5); vllm/v1/attention/backends/utils.py (+18/-0); vllm/forward_context.py (+8/-3); vllm/v1/spec_decode/eagle.py (+10/-1); vllm/v1/worker/gpu_model_runner.py (+47/-21); vllm/v1/worker/tpu_model_runner.py (+16/-2)
LABELS: tpu, ready, v1
BODY: Should be merge after https://github.com/vllm-project/vllm/pull/17193 ⏎  ⏎ This PR changes ForwardContext.attn_metadata from a global one to dict[layer_name, AttentionMetadata] to prepare for hybrid allocator which allocate different block table to sliding window layers and full attention layers. We only need to build one attention metadata for each kv cache group and let all layers inside that kv cache group point to that attention metadata object.

### L3-621ca2c0ab  (L3, 2025-05-06, sha 621ca2c0aba8, PR #16458)
TITLE: [TPU] Increase block size and reset block shapes (#16458)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+15/-1); examples/offline_inference/tpu.py (+2/-1); requirements/tpu.txt (+5/-5); vllm/platforms/tpu.py (+6/-4); vllm/utils.py (+7/-0)
LABELS: documentation, tpu, ready, ci/build, v1
BODY: Increase kv cache block size and reset kernel block shapes based on autotuned results from kernel.  ⏎ But still need to retune the kernel block shapes in kernel. ⏎  ⏎ > Note: we should wait for https://github.com/pytorch/xla/pull/9041 to be checkin and update new torch_xla version in requirements.txt ⏎  ⏎ Benchmarked without cache: ⏎  ⏎ ## v6e-1 (single chip): 7.87  -> 8.37 req / sec  ⏎  ⏎ Benchmarking script: ⏎  ⏎ ``` ⏎ VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B- …[truncated]

### L3-f9bc5a0693  (L3, 2025-05-06, sha f9bc5a0693d8, PR #17446)
TITLE: [Bugfix] Fix triton import with local TritonPlaceholder (#17446)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.prefix_prefill, L3.triton.flash_attention_rocm, L3.triton.decode_attention, L3.triton.chunked_prefill_paged_decode, L3.merge.triton_lse, L3.blocksparse.v0
FILES: vllm/attention/ops/blocksparse_attention/blocksparse_attention_kernel.py (+2/-2); vllm/attention/ops/blocksparse_attention/utils.py (+2/-1); vllm/attention/ops/chunked_prefill_paged_decode.py (+1/-2); vllm/attention/ops/prefix_prefill.py (+1/-2); vllm/attention/ops/triton_decode_attention.py (+1/-3); vllm/attention/ops/triton_flash_attention.py (+1/-2); vllm/attention/ops/triton_merge_attn_states.py (+2/-2); benchmarks/kernels/benchmark_moe.py (+1/-1); benchmarks/kernels/benchmark_rmsnorm.py (+1/-1); benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py (+1/-1); (+20 more)
LABELS: ready, v1
BODY: Fix triton import error in non-triton platforms with the local `TritonPlaceholder`

### L3-e50a1f1a9c  (L3, 2025-05-06, sha e50a1f1a9cc5, PR #17496)
TITLE: [TPU] Add kernel test for moe_pallas (#17496)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/pallas.py (+2/-1); .buildkite/scripts/hardware_ci/run-tpu-v1-test.sh (+3/-1); tests/tpu/test_moe_pallas.py (+87/-0); vllm/model_executor/layers/fused_moe/moe_pallas.py (+4/-1)
LABELS: tpu, ready, ci/build
BODY: First step in re-enabling the moe_pallas kernel so we can replace moe_torch_iterative for TPU

### L3-2f925e5777  (L3, 2025-05-06, sha 2f925e5777cc, PR #16828)
TITLE: [Kernel] Unified Triton kernel that doesn't distinguish between prefill + decode (#16828)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.triton.unified_attention, L3.triton.v1_backend
FILES: vllm/attention/ops/triton_unified_attention.py (+333/-0); vllm/v1/attention/backends/triton_attn.py (+44/-27); tests/kernels/test_triton_unified_attention.py (+189/-0)
LABELS: ready, v1
BODY: In this PR we add: ⏎ - A new Triton kernel (`triton_unified_attention`) that works like `flash_attn_varlen_func` and can handle arbitrary query length. The kernel does GQA "packing" along the query dimension to ensure the Tensor cores are well used. ⏎ - Added a new unit test that is based on the unit tests for `flash_attn_varlen_func`  ⏎ - Updated the V1 Triton attention backend to use this kernel. Note that the memory layout for the key cache also cha …[truncated]

### L3-6de3e13413  (L3, 2025-05-07, sha 6de3e13413a5, PR #17669)
TITLE: Add logging for torch nightly version (#17669)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.nightly_torch (+3/-1); requirements/nightly_torch_test.txt (+9/-1)
LABELS: ready, ci/build
BODY: print out pip freeze | grep to confirm the torch version ⏎  ⏎ testing build: https://buildkite.com/vllm/ci/builds/19304 ⏎ https://buildkite.com/vllm/ci/builds/19304#0196a14e-0a7d-4da6-a997-4b37dbb01008 ⏎  ⏎ working: ⏎ <img width="1095" alt="image" src="https://github.com/user-attachments/assets/f052eaba-90ea-43d3-b660-29fcd09f00e0" /> ⏎  ⏎ Also add basic model test dependency: ⏎ https://buildkite.com/vllm/ci/builds/19437#0196a741-8743-4539-aaba-acdd2d980381

### L3-c3e9d5060e  (L3, 2025-05-07, sha c3e9d5060e89, PR #17726)
TITLE: [Misc] Use `apply_rotary_emb` from vllm_flash_attn for Qwen2-VL vision RoPE (#17726)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen2_5_vl.py (+2/-7); vllm/model_executor/models/qwen2_vl.py (+4/-5)
LABELS: ready
BODY: - Since we have ported FA's RoPE kernel in `vllm_flash_attn`, there is no need to use original FA's `apply_rotary_emb` anymore ⏎ - Replace original FA's `apply_rotary_emb` with `vllm_flash_attn`'s

### L3-32aa74c09c  (L3, 2025-05-07, sha 32aa74c09c82, PR #17139)
TITLE: [ROCm][FP8][Kernel] FP8 quantization fused into Custom Paged Attention (#17139)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.rocm.custom_paged
FILES: csrc/rocm/attention.cu (+61/-31); vllm/_custom_ops.py (+2/-1); csrc/rocm/ops.h (+9/-11); csrc/rocm/torch_bindings.cpp (+2/-1)
LABELS: rocm, ready
BODY: An option to apply fp8 output scale in ROCm custom paged attention and output FP8 tensor ⏎ In case a non-None scale tensor is passed to the kernel, the output tensor is expected to be in the current_platform.fp8_dtype() type (float8_fnuz or float8_fn), and the scale is applied to it before storing into an 8-bit type

### L3-7ea2adb802  (L3, 2025-05-07, sha 7ea2adb8026e, PR #16072)
TITLE: [Core] Support full cuda graph in v1 (#16072)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+10/-3); docs/source/design/v1/torch_compile.md (+6/-0); tests/compile/piecewise/test_full_cudagraph.py (+97/-0); vllm/config.py (+17/-2); vllm/v1/worker/gpu_model_runner.py (+60/-8)
LABELS: documentation, ready, ci/build, v1
BODY: ## Summary ⏎ Support capturing a single CUDA graph for the entire model's forward pass, instead of piecewise graphs. This requires creating persistent buffers to make attention graphable. Credit to @tlrmchlsmth for the original implementation. ⏎  ⏎ Limitations: ⏎   1. This only works with V1 + FA3, since FA2 currently is not graphable due to an optimization for GQA. ⏎   2. This doesn't work with Cascade Attention. ⏎  ⏎ Work in progress: ⏎   1. Investigating chan …[truncated]

### L3-843b222723  (L3, 2025-05-07, sha 843b222723b6, PR #17648)
TITLE: [Hardware][Intel-Gaudi] Support Automatic Prefix Caching on HPU (#17648)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/hpu_attn.py (+20/-12); vllm/attention/ops/hpu_paged_attn.py (+9/-27); vllm/worker/hpu_model_runner.py (+117/-17)
LABELS: ready
BODY: Added support for Automatic Prefix Caching for HPU/Gaudi. Disabled by default, might be turned on with enable_prefix_caching=True (in code) / --enable-prefix-caching (server spawning)

### L3-f50dcb7c21  (L3, 2025-05-08, sha f50dcb7c215b, PR #17819)
TITLE: [Easy] Eliminate c10::optional usage in vllm/csrc (#17819)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.custom_paged
FILES: csrc/rocm/attention.cu (+2/-2); csrc/quantization/gptq_allspark/allspark_qgemm_w8a16.cu (+2/-2); csrc/quantization/gptq_allspark/allspark_repack.cu (+2/-2); csrc/rocm/ops.h (+1/-1)
LABELS: ready
BODY: Summary: c10::optional will fail internal build. So replace with std::optional ⏎  ⏎ Differential Revision: D74356723

### L3-a463555dee  (L3, 2025-05-08, sha a463555dee7c, PR #17820)
TITLE: [TPU] Fix the test_sampler (#17820)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/pallas.py (+1/-1); tests/v1/tpu/test_sampler.py (+1/-1)
LABELS: tpu, ready, ci/build, v1
BODY: TLDR: this PR is temporary quick fix for broken CI for tests/v1/tpu/test_sampler.py. Kernel fix is WIP.  ⏎  ⏎ After https://github.com/vllm-project/vllm/pull/16458, for the `test_sampler_different` in `tests/v1/tpu/test_sampler.py`, we will use (32, 32) block shape instead of (128, 32) block shape. However, this triggers some correctness issue. The root cause is kernel assumes kv_cache has no NAN value. We should fix that in kernel! Meanwhile, just t …[truncated]

### L3-217db4baa6  (L3, 2025-05-09, sha 217db4baa646, PR #17880)
TITLE: [Bugfix][ROCm] Fix AITER MLA V1 (#17880)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+1/-3)
LABELS: ready, v1
BODY: This PR updates the AITER MLA V1 based on the changes in this [PR](https://github.com/vllm-project/vllm/pull/17668), which resolves the pre-commit errors [here](https://github.com/vllm-project/vllm/actions/runs/14920620116).

### L3-85b72cb7b1  (L3, 2025-05-09, sha 85b72cb7b12c, PR #17910)
TITLE: Revert "[BugFix][AMD] Compatible patch for latest AITER(05/07/2025)" (#17910)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.triton_v0
FILES: vllm/attention/backends/mla/common.py (+6/-6); vllm/attention/backends/rocm_aiter_mla.py (+12/-37); vllm/attention/ops/rocm_aiter_mla.py (+1/-6); vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py (+4/-5)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 17864 reason=unstated
BODY: Reverts vllm-project/vllm#17864

### L3-9f64e93415  (L3, 2025-05-09, sha 9f64e93415c4, PR #17864)
TITLE: [BugFix][AMD] Compatible patch for latest AITER(05/07/2025) (#17864)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.triton_v0
FILES: vllm/attention/backends/mla/common.py (+6/-6); vllm/attention/backends/rocm_aiter_mla.py (+37/-12); vllm/attention/ops/rocm_aiter_mla.py (+6/-1); vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py (+5/-4)
LABELS: rocm, ready
DEEP_STUDY: deep-study: this PR was reverted by PR 17910 (confirmed_revert, reason=unstated)
BODY: 1. Changes to adapt new AITER MoE API; ⏎ 2. Changes to adapt new AITER MLA API (MTP is not enabled) ⏎ 3. Some other bug fixes; ⏎ This PR is for AITER API changes introduced by commit 939f741fc37f46694e48c32c7164f49eae2584c4 (merged on 04/20/2025). AITER versions after this require this patch to work.

### L3-3c9396a64f  (L3, 2025-05-09, sha 3c9396a64fbf, PR #17523)
TITLE: [FEAT][ROCm]: Support AITER MLA on V1 Engine (#17523)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.mla.common_v1, L3.mla.rocm_aiter, L3.platform.rocm_selection
FILES: docker/Dockerfile.rocm_base (+1/-1); vllm/attention/ops/rocm_aiter_mla.py (+46/-0); vllm/engine/arg_utils.py (+1/-0); vllm/platforms/interface.py (+2/-1); vllm/platforms/rocm.py (+8/-3); vllm/v1/attention/backends/mla/common.py (+6/-5); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+196/-0); tests/kernels/attention/test_attention_selector.py (+4/-1); tests/kernels/attention/test_rocm_attention_selector.py (+4/-2); vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py (+1/-1)
LABELS: rocm, ready, ci/build, v1
BODY: ## AITER MLA Support for V1 Engine ⏎  ⏎ This PR implements AITER MLA attention backend support for the V1 engine. The implementation mirrors the V0 engine's established approach from [PR #15893](https://github.com/vllm-project/vllm/pull/15893). ⏎  ⏎ This PR also introduces a new environment variable, `VLLM_ROCM_EXECUTE_MODEL_TIMEOUT`, which specifies the model execution timeout in seconds. This allows for flexible adjustment of execution time, which is h …[truncated]

### L3-5e6f939484  (L3, 2025-05-09, sha 5e6f93948449, PR #17668)
TITLE: [Attention] MLA move rotary embedding to cuda-graph region (#17668)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.triton_v0, L3.mla.common_v1, L3.mla.flashmla_v1_adapter
FILES: vllm/attention/backends/mla/common.py (+11/-60); vllm/attention/backends/rocm_aiter_mla.py (+2/-4); vllm/model_executor/models/deepseek_v2.py (+7/-1); vllm/v1/attention/backends/mla/common.py (+11/-51); vllm/v1/attention/backends/mla/flashmla.py (+1/-3); vllm/model_executor/layers/rotary_embedding.py (+3/-2)
LABELS: ready, v1
BODY: Following on from https://github.com/vllm-project/vllm/pull/17484, with: https://github.com/vllm-project/vllm/pull/14770 we are no longer materializing the absorbed matrices so we can move the rotary embeddings into the cuda-graph region lowering CPU overhead: ⏎  ⏎ # Perf ⏎  ⏎ ## Main ⏎  ⏎ CPU time for `unified_attention` is 216us ⏎ <img width="1393" alt="image" src="https://github.com/user-attachments/assets/d81d15a3-17a6-4dc3-9c02-03a3d2392daf" /> ⏎  ⏎ ## PR ⏎  ⏎ CP …[truncated]

### L3-246e3e0a36  (L3, 2025-05-10, sha 246e3e0a36fd, PR #17873)
TITLE: fix broken test vllm:test_kernels - test_attention_selector.py::test_flash_attn (#17873)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_attention_selector.py (+3/-2)
LABELS: ready
BODY: Summary: xformers calls torch.cuda.get_device_capability("cuda"). Our monkey patching version of get_device_capability didn't accept string argument. ⏎  ⏎ Differential Revision: D74440549

### L3-950751a987  (L3, 2025-05-10, sha 950751a9870f, PR #17483)
TITLE: [v1] Pass BlockTable and KVCacheSpec to AttentionMetadataBuilders (#17483)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/flash_attn.py (+30/-17); vllm/v1/attention/backends/flashinfer.py (+20/-15); vllm/v1/attention/backends/mla/common.py (+15/-8); vllm/v1/attention/backends/mla/flashmla.py (+7/-4); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+5/-2); tests/v1/worker/test_gpu_input_batch.py (+3/-0); tests/v1/worker/test_gpu_model_runner.py (+20/-1); vllm/v1/worker/block_table.py (+11/-0); vllm/v1/worker/gpu_input_batch.py (+3/-0); vllm/v1/worker/gpu_model_runner.py (+9/-12); (+1 more)
LABELS: tpu, ready, v1
BODY: Should merge after https://github.com/vllm-project/vllm/pull/17394 ⏎  ⏎ Hybrid allocator will need to build attention metadata for each kv cache group because different kv cache groups may have different attention type and block_table. To achieve that, we will introduce one AttentionMetadataBuilder and one BlockTable for each group. ⏎  ⏎ To prepare for this, this PR makes AttentionMetadataBuilder to access its block_table and KVCacheSpec, instead of read …[truncated]

### L3-eea22a56ab  (L3, 2025-05-11, sha eea22a56ab08, PR #17871)
TITLE: fix amd triton mla path (#17871)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.triton_v0
FILES: vllm/attention/backends/mla/common.py (+1/-1)
LABELS: ready
BODY: Summary: Should be `elif` rather than a new if branch. ⏎  ⏎ Differential Revision: D74436575

### L3-7de18d541b  (L3, 2025-05-11, sha 7de18d541b0d, PR #17961)
TITLE: [BUG] [ROCm] [MLA] Fix variable name bug due to change in variable name in PR #17483 (#17961)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+3/-3)
LABELS: ready, v1
BODY: This is to fix the error due to change of name introduced by PR https://github.com/vllm-project/vllm/pull/17483 ⏎  ⏎ the rename of the `block_table` to `block_table_tensor` in `_build_decode` function had broken `AiterMLAMetadataBuilder._build_decode`

### L3-06c0922a69  (L3, 2025-05-11, sha 06c0922a69c1, PR #17870)
TITLE: [FP8][ROCm][Attention] Enable FP8 KV cache on ROCm for V1 (#17870)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.chunked_prefill_paged_decode, L3.triton.v1_backend
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+2/-1); vllm/engine/arg_utils.py (+3/-1); vllm/v1/attention/backends/triton_attn.py (+12/-6)
LABELS: ready, v1
BODY: Enable FP8 KV cache support in the unified triton kernel for ROCm ⏎ Also fix the chunked_prefill_paged_decode in case it falls back from custom_paged_attention to the triton kernel ⏎  ⏎ cc @tdoublep

### L3-60f7624334  (L3, 2025-05-12, sha 60f76243344d, PR #11844)
TITLE: Implements dual-chunk-flash-attn backend for dual chunk attention with sparse attention support (#11844)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.platform.cuda_selection
FILES: CMakeLists.txt (+1/-0); csrc/attention/vertical_slash_index.cu (+401/-0); csrc/ops.h (+25/-0); csrc/torch_bindings.cpp (+23/-0); vllm/_custom_ops.py (+95/-0); vllm/attention/backends/dual_chunk_flash_attn.py (+1494/-0); vllm/config.py (+19/-0); vllm/engine/arg_utils.py (+13/-2); vllm/platforms/cuda.py (+4/-0); vllm/platforms/interface.py (+1/-0); (+7 more)
LABELS: documentation, ready, ci/build
ISSUES: #12452 [Feature]: Support Qwen/Qwen2.5-14B-Instruct-1M
BODY: This PR implements the [dual-chunk flash attention](https://arxiv.org/pdf/2402.17463.pdf), a training-free method to extend model context length (see also #6139), with sparse attention (https://github.com/microsoft/MInference) support. ⏎  ⏎ This PR requires the [sparse attention kernel](https://github.com/vllm-project/flash-attention/pull/33) from vllm-flash-attention. Qwen models with 1m context length support will be open-sourced in the next one or …[truncated]

### L3-40de1ef455  (L3, 2025-05-13, sha 40de1ef455f2, PR #14968)
TITLE: [FEAT] [ROCm]: Add AITER Block-Scaled GEMM Feature (#14968)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/model_executor/test_enabled_custom_ops.py (+31/-0); vllm/model_executor/layers/quantization/fp8.py (+8/-0); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+98/-32)
LABELS: ready, ci/build
BODY: # Description ⏎ This PR integrates the Block-Scaled GEMM functionality from [AITER](https://github.com/ROCm/aiter/blob/82d128add808b994046273e6b834f52c4e21c435/aiter/ops/gemm_op_a8w8.py#L42) into vLLM, and will allow any up-coming optimizations in AITER kernel to be directly used and evaluated within the vLLM framework. ⏎  ⏎ ## Implementation ⏎ The gemm_a8w8_blockscale kernel from AITER has been added to `/vllm/model_executor/layers/quantization/utils/fp …[truncated]

### L3-176a95c670  (L3, 2025-05-13, sha 176a95c670f6, PR #18104)
TITLE: [Fix] Support CUDAGraph capture for encoder-decoder on ROCm (#18104)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/utils.py (+8/-8)
LABELS: bug, rocm, ready
BODY: On `main`, encoder-decoder models are supported on ROCm with `--enforce-eager`, but when capturing CUDA Graphs, a stale assert is triggered, saying the `ROCM_FLASH` attention backend isn't supported (which it is as it runs successfully in eager mode). This PR removes that assert. ⏎  ⏎ Tested with: ⏎ ``` ⏎ vllm serve openai/whisper-large-v3  ⏎ ```

### L3-12e6c0b41c  (L3, 2025-05-13, sha 12e6c0b41c19, PR #18086)
TITLE: [Bugfix][V1] Fix FlashInfer V1 backend using the wrong VllmConfig (#18086)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+2/-3)
LABELS: bug, ready, v1
BODY: FIX https://github.com/vllm-project/vllm/pull/17483#issuecomment-2875609788 ⏎  ⏎ Thanks to @chenyang78 for reporting and @heheda12345 for the fix. ⏎  ⏎ The global vllm config may not be set by `set_current_vllm_config`, so we should read it from runner directly so `self.vllm_config = runner.vllm_config`. This is already what FlashInfer V0 does so this was likely just an oversight https://github.com/vllm-project/vllm/blob/19324d660c61a63c6ea3dfbb18995d255 …[truncated]

### L3-2d912fb66f  (L3, 2025-05-13, sha 2d912fb66fed, PR #17955)
TITLE: [FEAT] [ROCm] [V1]: Add AITER biased group topk for DeepSeekV3 (#17955)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_rocm_aiter_topk.py (+122/-0); vllm/model_executor/layers/fused_moe/layer.py (+8/-2); vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py (+71/-0)
LABELS: rocm, ready
BODY: This PR add the use of `biased_group_topk` kernel for DeepSeekV3 ⏎  ⏎ ## Performance Comparison Summary ⏎  ⏎ | Metric | V0 Before | V0 After | Change (%) | V1 Before | V1 After | Change (%) | ⏎ |--------|-----------|----------|------------|-----------|----------|------------| ⏎ | **Benchmark Duration (s)** | 168.90 | 163.00 | -3.5% | 144.65 | 137.63 | -4.9% | ⏎ | **Request Throughput (req/s)** | 2.96 | 3.07 | +3.7% | 3.46 | 3.63 | +4.9% | ⏎ | **Output Token Thro …[truncated]

### L3-4f8b373225  (L3, 2025-05-13, sha 4f8b37322561, PR #17912)
TITLE: [BugFix][AMD] Compatible patch for AITER lib after 04/20 (#17912)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/rocm_aiter_mla.py (+37/-12); vllm/attention/ops/rocm_aiter_mla.py (+12/-1); vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py (+5/-4)
LABELS: ready
BODY: 1. Changes to adapt new AITER MLA API (MTP is not enabled) ⏎ 2. Some other bug fixes. ⏎  ⏎ This PR is for AITER API changes introduced by commit 939f741fc37f46694e48c32c7164f49eae2584c4 (merged on 04/20/2025). AITER versions after this commit require this patch to work.

### L3-d62a076e84  (L3, 2025-05-14, sha d62a076e8467, PR #18109)
TITLE: [Model] GritLM supports other attention backends (#18109)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/llama.py (+28/-14); tests/models/language/pooling/test_gritlm.py (+31/-46); vllm/model_executor/models/gritlm.py (+12/-34); vllm/model_executor/models/qwen2.py (+13/-13)
LABELS: ready
BODY: This PR works around the issue where the xformers attention backend fails to be selected in GritLM tests, by simply supporting other attention backends for this model as well.

### L3-c8ea982d9b  (L3, 2025-05-14, sha c8ea982d9b86, PR #18129)
TITLE: Update deprecated type hinting in `platform`, `plugins`, `triton_utils`, `vllm_flash_attn` (#18129)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: pyproject.toml (+0/-5); vllm/platforms/cuda.py (+6/-7); vllm/platforms/interface.py (+3/-3); vllm/platforms/rocm.py (+5/-5); vllm/platforms/tpu.py (+2/-2); vllm/plugins/__init__.py (+2/-2)
LABELS: tpu, ready
BODY: Removes deprecated Python 3.8 syntax. ⏎  ⏎ N.B. also removes `transformers_utils` from the skip list in `pyproject.toml`, it must have been missed in the PR which updated the typing in that directory.

### L3-f9c069c85e  (L3, 2025-05-14, sha f9c069c85e02, PR #15956)
TITLE: Modularize fused experts and integrate PPLX kernels (#15956)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1, L3.platform.cuda_selection
FILES: vllm/v1/attention/backends/mla/common.py (+4/-2); csrc/activation_kernels.cu (+3/-0); csrc/dispatch_utils.h (+14/-0); csrc/moe/moe_align_sum_kernels.cu (+4/-4); csrc/moe/topk_softmax_kernels.cu (+45/-18); examples/offline_inference/data_parallel.py (+16/-6); tests/kernels/moe/test_batched_moe.py (+114/-0); tests/kernels/moe/test_cutlass_moe.py (+21/-25); tests/kernels/moe/test_moe.py (+51/-42); tests/kernels/moe/test_pplx_moe.py (+691/-0); (+32 more)
LABELS: documentation, tpu, ready, ci/build, v1
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:92540529c0 (fix PR 18205)
BODY: This PR defines a set of base classes used to make MoE kernels more modular. The goal is to be able to utilize different communication mechanisms with any fused MoE kernel without needing to have combinatoric implementations.                                                                                                      ⏎  ⏎ The fused moe kernels are broken down into the following components: ⏎ ``` ⏎ [Router] → [Quantize-Dispatch] → [Permute-Experts …[truncated]

### L3-e60f550b38  (L3, 2025-05-14, sha e60f550b3825, PR #17945)
TITLE: [v1] Support multiple KV cache groups in GPU model runner (#17945)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+2/-2); tests/v1/core/test_kv_cache_utils.py (+68/-3); tests/v1/core/test_prefix_caching.py (+18/-18); tests/v1/worker/test_gpu_input_batch.py (+33/-6); tests/v1/worker/test_gpu_model_runner.py (+41/-16); tests/weight_loading/models.txt (+1/-1); vllm/distributed/kv_transfer/kv_connector/v1/shared_storage_connector.py (+3/-3); vllm/v1/attention/backends/flashinfer.py (+2/-3); vllm/v1/core/kv_cache_manager.py (+21/-13); vllm/v1/core/kv_cache_utils.py (+6/-7); (+7 more)
LABELS: tpu, ready, v1
DEEP_STUDY: deep-study: this PR was reverted by PR 18459 (confirmed_revert, reason=ci_or_test_failure)
BODY: Should be merged after https://github.com/vllm-project/vllm/pull/17483 ⏎  ⏎ This PR finishes the hybrid allocator support on worker side. It does the following things: ⏎ 1. change `block_ids` in SchedulerOutput to `list[list[int]]`, where the outer list is for multiple kv cache groups and inner list is for blocks in one group. ⏎ 2. Create `BlockTable` class for each kv cache group. ⏎ 3. Build different attention metadata for each kv cache group. ⏎ 4. TPU bac …[truncated]

### L3-a9944aabfa  (L3, 2025-05-15, sha a9944aabfa0e, PR #18151)
TITLE: fix: typos (#18151)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv
FILES: csrc/attention/attention_kernels.cuh (+2/-2); examples/offline_inference/chat_with_tools.py (+2/-2); tests/lora/test_lora_huggingface.py (+1/-1); tests/model_executor/weight_utils.py (+3/-3); vllm/config.py (+1/-1); vllm/lora/ops/triton_ops/lora_expand_op.py (+1/-1); vllm/model_executor/layers/mamba/mamba_mixer2.py (+1/-1); vllm/model_executor/models/granite_speech.py (+2/-2); vllm/model_executor/models/phi4mm_audio.py (+4/-4); vllm/v1/request.py (+1/-1)
LABELS: documentation, ready, v1, tool-calling
BODY: fix: typos

### L3-01c22335ba  (L3, 2025-05-15, sha 01c22335baa0, PR #18161)
TITLE: [Kernel] [V1] Fix performance regression for triton unified attention (#18161)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.unified_attention, L3.triton.v1_backend
FILES: vllm/attention/ops/triton_unified_attention.py (+2/-2); vllm/v1/attention/backends/triton_attn.py (+16/-3)
LABELS: ready, v1
BODY: We have observed a pretty severe (~40%) performance regression for the `triton_unified_attention` kernel when moving from triton 3.2 to triton 3.3.  ⏎  ⏎ After a lot of investigation, I was able to figure out that it came from [this](https://github.com/triton-lang/triton/pull/5512) commit to Triton. This change reworked the way that Triton determines what kernel arguments are constant. It seems that before this PR, Triton was (correctly) detecting th …[truncated]

### L3-e6b8e65d2d  (L3, 2025-05-15, sha e6b8e65d2d68, PR #18013)
TITLE: [Bugfix] Fix fp8 tests for triton_unified_attention for Triton 3.3 (#18013)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+4/-0); tests/kernels/attention/test_triton_unified_attention.py (+3/-0)
LABELS: ready
BODY: The FP8 unit tests for triton_unified_attention don't pass for Triton 3.3. ⏎  ⏎ We didn't catch this through CI when upgrading Triton because the corresponding file hadn't been moved into the new "kernels" subfolder. ⏎  ⏎ This PR resolves both issues. ⏎  ⏎ cc @LucasWilkinson

### L3-ee659e3b60  (L3, 2025-05-15, sha ee659e3b601e, PR #18093)
TITLE: [Bugfix][ROCm] Use `chunked_prefill_paged_decode` as fallback for V1 attention on ROCm (#18093)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+77/-32)
LABELS: rocm, ready, v1
BODY: On ROCm, vLLM’s V1 engine uses the unified attention kernel as its sole attention backend. However, at the moment this kernel fails when running models where the number of query heads over the number of key-value heads is not a power of two. This makes models like Llama-4-Scout, whose `num_queries_per_kv` evaluates to an odd number, fail to run and yield the following error on ROCm: ⏎  ⏎ ``` ⏎ offs_m = tl.arange(0, BLOCK_Q * num_queries_per_kv) ⏎         …[truncated]

### L3-7fdfa01530  (L3, 2025-05-16, sha 7fdfa015304e, PR #15777)
TITLE: [Sampler] Adapt to FlashInfer 0.2.3 sampler API (#15777)
SOURCES: dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+2/-1); tests/samplers/test_rejection_sampler.py (+12/-2); tests/samplers/test_sampler.py (+2/-0); tests/v1/sample/test_topk_topp_sampler.py (+71/-1); vllm/model_executor/layers/rejection_sampler.py (+7/-6); vllm/model_executor/layers/sampler.py (+13/-39); vllm/v1/sample/ops/topk_topp_sampler.py (+16/-40)
LABELS: ready, ci/build, v1
ISSUES: #15666 [Feature]: update to flashinfer 0.2.3
BODY: FlashInfer [0.2.3](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.2.3) introduced some breaking changes to its sampler API, this PR updates the calling sites in vLLM to adapt to the update. ⏎  ⏎ FIX #14815 ⏎ FIX #15666

### L3-dcfe95234c  (L3, 2025-05-17, sha dcfe95234c11, PR #18095)
TITLE: Update Dockerfile to build for Blackwell (#18095)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+3/-3)
LABELS: ready, ci/build
ISSUES: #17325 [Feature]: Integrate FlashInfer Blackwell kernels
BODY: Updates the docker to build wheels for blackwell (SM 10.0) and include the latest flashinfer for performance blackwell attention support (FIX https://github.com/vllm-project/vllm/issues/17325). We didn't include SM 12.0 for now because of wheel size concerns. ⏎  ⏎ Updates to latest flashinfer main as of 5/15 since there isn't a release yet: https://github.com/flashinfer-ai/flashinfer/commit/e00e8cedbfcb220f328fd36aa8f529f869b01e6b

### L3-9ab2c02ff8  (L3, 2025-05-17, sha 9ab2c02ff8ef, PR #18243)
TITLE: Support sequence parallelism combined with pipeline parallelism (#18243)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/distributed/test_sequence_parallel.py (+35/-2); vllm/config.py (+0/-12); vllm/v1/worker/gpu_model_runner.py (+39/-13)
LABELS: ready, v1
BODY: this PR adds support for sp+pp scenario. ⏎ ``` ⏎ vllm serve meta-llama/Llama-3.2-1B-Instruct --tensor-parallel-size 2 --pipeline-parallel-size 2 --distributed-executor-backend mp -O '{"level": 3, "compile_sizes": [4, 8, 16], "splitting_ops": [], "pass_config": {"enable_sequence_parallelism" : true}}' ⏎ ```

### L3-47fda6d089  (L3, 2025-05-18, sha 47fda6d089ff, PR #18316)
TITLE: [Build] Supports CUDA 12.6 and 11.8 after Blackwell Update (#18316)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+7/-2); .buildkite/release-pipeline.yaml (+2/-2)
LABELS: ci/build
BODY: #18095 broke CUDA 12.6 and 11.8 wheel build

### L3-980a172474  (L3, 2025-05-20, sha 980a172474fa, PR #18099)
TITLE: [Kernel] update comment for KV shape in unified triton attn (#18099)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+2/-2)
BODY: Unified Triton attention uses a new layout of KV, but the comment in the code has not been updated. ⏎  ⏎ ``` ⏎ shape of K cache: torch.Size([11784, 16, 8, 128])  # blk num, blk size, head num, head size ⏎ shape of V cache: torch.Size([11784, 16, 8, 128]) ⏎ ```

### L3-dd5fa7e04f  (L3, 2025-05-21, sha dd5fa7e04f75, PR #17004)
TITLE: [ROCm][Kernel][V1] Enable AMD Radeon GPU Custom Paged Attention on v1 (#17004)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.chunked_prefill_paged_decode, L3.rocm.custom_paged, L3.rocm.rocm_flash_attn_v0, L3.platform.rocm_selection
FILES: csrc/rocm/attention.cu (+1880/-171); vllm/attention/backends/rocm_flash_attn.py (+2/-1); vllm/attention/ops/chunked_prefill_paged_decode.py (+2/-1); vllm/platforms/rocm.py (+34/-14); benchmarks/kernels/benchmark_paged_attention.py (+5/-1); tests/kernels/attention/test_attention.py (+7/-1)
LABELS: ready
BODY: Add additional custom paged attention kernels for AMD gfx11/gfx12 GPU support. Based on PRs:  https://github.com/vllm-project/vllm/pull/12348 https://github.com/vllm-project/vllm/pull/15720 https://github.com/vllm-project/vllm/pull/13843 ⏎  ⏎ Due to the differences in architecture from MI, specific instructions and detailed logic have changed (mfma16 -> wmma16/wmma16_gfx12), so new kernels for each architecture has been added. ⏎  ⏎ - Supports cases where …[truncated]

### L3-bb0a311213  (L3, 2025-05-21, sha bb0a3112130a, PR #18459)
TITLE: Revert "[v1] Support multiple KV cache groups in GPU model runner (#17945) (#18459)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+2/-2); tests/v1/core/test_kv_cache_utils.py (+3/-68); tests/v1/core/test_prefix_caching.py (+18/-18); tests/v1/worker/test_gpu_input_batch.py (+6/-33); tests/v1/worker/test_gpu_model_runner.py (+16/-41); vllm/distributed/kv_transfer/kv_connector/v1/shared_storage_connector.py (+3/-3); vllm/v1/core/kv_cache_manager.py (+13/-21); vllm/v1/core/kv_cache_utils.py (+7/-6); vllm/v1/core/sched/output.py (+6/-6); vllm/v1/core/sched/scheduler.py (+6/-10); (+5 more)
LABELS: tpu, ready, v1
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 17945 reason=ci_or_test_failure
BODY: This reverts commit e60f550b3825cbce2d3c7e882b029e2c1d914d8d. ⏎  ⏎ To see if #18425, #18418, #18245, and #18416 can be reproduced without this PR

### L3-94d8ec8d2b  (L3, 2025-05-21, sha 94d8ec8d2bcb, PR #18338)
TITLE: [FEAT][ROCm] Upgrade AITER MLA v1 backend (#18338)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.mla.rocm_aiter
FILES: docker/Dockerfile.rocm_base (+1/-1); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+30/-6)
LABELS: ready, ci/build, v1
BODY: This PR upgrades the AITER MLA attention backend on the v1 engine, as this backend was previously upgraded on the v0 engine in [this PR](https://github.com/vllm-project/vllm/pull/17912). The upgrade uses AITER commit [c1debd8](https://github.com/ROCm/aiter/commit/c1debd87ce0391aa27438d9e07e76e4fea7c4b70), which introduces the relevant API changes. ⏎  ⏎ It's important to note that the `mla_decode_fwd` kernel in the AITER package imposes new constraint …[truncated]

### L3-6e588da0f4  (L3, 2025-05-22, sha 6e588da0f4b9, PR #17679)
TITLE: [Build/CI] Fix CUDA 11.8 build (#17679)
SOURCES: corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+5/-1); csrc/moe/moe_ops.h (+3/-1); csrc/moe/moe_permute_unpermute_op.cu (+42/-1); csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu (+6/-4); csrc/moe/torch_bindings.cpp (+3/-1); csrc/quantization/cutlass_w8a8/scaled_mm_entry.cu (+1/-1); docker/Dockerfile (+11/-5); tests/kernels/moe/test_moe_permute_unpermute.py (+3/-1); vllm/model_executor/layers/fused_moe/moe_permute_unpermute.py (+4/-0)
LABELS: ready, ci/build
DEEP_STUDY: deep-study correctness case vllm:6e588da0f4: class=hardware_compiler_specific; symptom=compile_or_build_failure; introducing=#14568
BODY: The CUDA 11.8 build is failing for a couple of reasons. ⏎  ⏎ This PR: ⏎  ⏎ Skips building `moe_permute_unpermute` kernels on CUDA < 12.0 in order to fix the CUDA 11.8 build. Also add a function to report whether the `moe_permute_unpermute` kernels are available. (Was broken by #14568). ⏎  ⏎ Also disables `FLASHINFER_ENABLE_AOT=0` on CUDA 11.8

### L3-71ea614d4a  (L3, 2025-05-23, sha 71ea614d4ab2, PR #17882)
TITLE: [Feature]Add async tensor parallelism using compilation pass (#17882)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test-pipeline.yaml (+1/-0); tests/compile/backend.py (+18/-0); tests/compile/test_async_tp.py (+248/-0); tests/compile/test_fusion.py (+17/-19); tests/compile/test_sequence_parallelism.py (+18/-29); vllm/compilation/collective_fusion.py (+126/-0); vllm/compilation/pass_manager.py (+3/-0); vllm/compilation/sequence_parallelism.py (+5/-4); vllm/compilation/vllm_inductor_pass.py (+2/-1); vllm/config.py (+10/-1); (+1 more)
LABELS: ready, ci/build
BODY: This PR adds [torch async tp](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487) using compilation pass.  ⏎ It requires below config to run ⏎ ``` ⏎ config = CompilationConfig( ⏎     level=3, ⏎     compile_sizes=[4, 8, 16], ⏎     splitting_ops=[], ⏎ ) ⏎ config.pass_config.enable_async_tp= True ⏎  ⏎ llm = LLM(model="llama/Llama-3.2-1B-Instruct", ⏎           enforce_eager=False, ⏎           tensor_parallel_size=2, …[truncated]

### L3-6550114c9c  (L3, 2025-05-23, sha 6550114c9cd2, PR #18593)
TITLE: [v1] Redo "Support multiple KV cache groups in GPU model runner (#17945)" (#18593)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+2/-2); tests/v1/core/test_kv_cache_utils.py (+68/-3); tests/v1/core/test_prefix_caching.py (+18/-18); tests/v1/worker/test_gpu_input_batch.py (+33/-6); tests/v1/worker/test_gpu_model_runner.py (+41/-16); vllm/distributed/kv_transfer/kv_connector/v1/shared_storage_connector.py (+3/-3); vllm/v1/core/kv_cache_manager.py (+21/-13); vllm/v1/core/kv_cache_utils.py (+6/-7); vllm/v1/core/sched/output.py (+6/-6); vllm/v1/core/sched/scheduler.py (+10/-6); (+5 more)
LABELS: tpu, ready, v1
BODY: Redo #17945 that reverted by https://github.com/vllm-project/vllm/pull/18459 to make CI green. ⏎  ⏎ This PR finishes the hybrid allocator support on worker side. It does the following things: ⏎ 1. change `block_ids` in SchedulerOutput to `list[list[int]]`, where the outer list is for multiple kv cache groups and inner list is for blocks in one group. ⏎ 2. Create `BlockTable` class for each kv cache group. ⏎ 3. Build different attention metadata for each kv …[truncated]

### L3-4b0da7b60e  (L3, 2025-05-23, sha 4b0da7b60e35, PR #18494)
TITLE: Enable hybrid attention models for Transformers backend (#18494)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/config.py (+11/-8); docs/source/contributing/model/basic.md (+1/-1); tests/models/test_transformers.py (+42/-14); vllm/model_executor/models/transformers.py (+51/-6)
LABELS: documentation, ready
BODY: Tested with Gemma 2 by asking it to summarise >4k tokens of the Wikipedia page on frogs. ⏎  ⏎ vLLM reference: ⏎  ⏎ ``` ⏎ Generated text: '\n\nFrogs are a diverse group of amphibians with a wide range of adaptations for survival. They are' ⏎ ``` ⏎  ⏎ Transformers backend before this PR: ⏎  ⏎ ``` ⏎ Generated text: '\n\nFrogs are a diverse group of amphibians with a wide range of habitats. They are a diverse' ⏎ ``` ⏎  ⏎ Transformers backend after this PR: ⏎  ⏎ ``` ⏎ Generated text:  …[truncated]

### L3-1645b60196  (L3, 2025-05-23, sha 1645b601960a, PR #18537)
TITLE: Use prebuilt FlashInfer x86_64 PyTorch 2.7 CUDA 12.8 wheel for CI (#18537)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+8/-9)
LABELS: ready, ci/build
BODY: I chose to build FlashInfer from source to unblock PyTorch 2.7 update.  However, this has increased CI build time by close to half an hour.  With https://github.com/flashinfer-ai/flashinfer/pull/1063 merged and while we are waiting for the official wheel, let's just use a prebuilt x86_64 wheel for PyTorch 2.7 CUDA 12.8.  The wheel has been uploaded to download.pytorch.org, but let me know if that's ok. ⏎  ⏎ 12.6 and 11.8 wheel would still be built fr …[truncated]

### L3-ef1dd6870f  (L3, 2025-05-24, sha ef1dd6870f84, PR #18659)
TITLE: [Doc] Fix indentation problems in V0 Paged Attention docs (#18659)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docs/deployment/k8s.md (+1/-0); docs/design/kernel/paged_attention.md (+371/-373)
LABELS: documentation, ready
BODY: Fix indentation problems in paged attention docs by removing unnecessary bullet points

### L3-794ae1f551  (L3, 2025-05-27, sha 794ae1f551e4, PR #18764)
TITLE: [rocm] Fix wrong attention log (#18764)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+3/-2)
BODY: As per title, the log `INFO 05-27 14:35:30 [rocm.py:208] None is not supported in AMD GPUs.` may get printed for non-deepseek model (see https://github.com/vllm-project/vllm/blob/6b6d4961147220fb80f9cc7dcb74db478f9c9a23/vllm/config.py#L1363-L1365), as `selected_backend` may be `None` in case it is not set with an env var or through a global https://github.com/vllm-project/vllm/blob/6b6d4961147220fb80f9cc7dcb74db478f9c9a23/vllm/attention/selector. …[truncated]

### L3-aaa4ac1c95  (L3, 2025-05-27, sha aaa4ac1c95aa, PR #18639)
TITLE: Disable prefix cache by default for benchmark (#18639)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/benchmark_latency.py (+3/-0); vllm/benchmarks/latency.py (+3/-0)
LABELS: ready
BODY: v1 has prefix cache enabled by default to improve performance.  ⏎ However, this creates misleading latency numbers when running benchmarks without explicitly disabling it. ⏎ This PR disables prefix cache in the benchmark script to improve usability for new users.

### L3-a3896c7f02  (L3, 2025-05-27, sha a3896c7f0216, PR #18570)
TITLE: [Build] Fixes for CMake install (#18570)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake, L3.flash_attn.fork_build
FILES: CMakeLists.txt (+5/-0); cmake/external_projects/vllm_flash_attn.cmake (+18/-2); setup.py (+1/-4); cmake/utils.cmake (+1/-1)
LABELS: ready, ci/build
BODY: This fixes the CMake install logic such that using `cmake --install` directly is supported. Necessary to fix CMake-based user workflows. ⏎  ⏎ It also fixes the error on ROCm if CMake is used outside the venv (common for IDE-run CMake).

### L3-51e98e4ffd  (L3, 2025-05-28, sha 51e98e4ffd69, PR #18771)
TITLE: [Bugfix] Disable prefix caching by default for benchmark (#18771)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/benchmarks/latency.py (+1/-1)
LABELS: ready
BODY: bugfix - Set to False instead of True

### L3-ce75efeecb  (L3, 2025-05-28, sha ce75efeecb57, PR #18807)
TITLE: [BugFix] FA2 MLA Accuracy Issue (#18807)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.merge.cuda_lse, L3.mla.triton_v0, L3.mla.common_v1
FILES: csrc/attention/merge_attn_states.cu (+8/-0); vllm/attention/backends/mla/common.py (+4/-4); vllm/v1/attention/backends/mla/common.py (+4/-4)
LABELS: ready, v1
ISSUES: #18561 [Bug]: MLA correctness issues when using FA2 | #18766 [CI Failure]: LM Eval Large Models - test_lm_eval_correctness.py
BODY: FIX https://github.com/vllm-project/vllm/issues/18561 ⏎ FIX #18766 ⏎  ⏎ `merge_attn_states` doesnt support tensor slices, so avoid slicing till after all the attn states are merged ⏎  ⏎ After this PR (tested on an A100): ⏎  ⏎ ``` ⏎ lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto ⏎ ... ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|-- …[truncated]

### L3-7951d78738  (L3, 2025-05-28, sha 7951d7873858, PR #18724)
TITLE: [Core] Enable CUDA graphs for DP + All2All kernels  (#18724)
SOURCES: release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/forward_context.py (+41/-22); vllm/model_executor/layers/fused_moe/layer.py (+39/-3); vllm/platforms/cuda.py (+0/-11); vllm/v1/worker/gpu_model_runner.py (+20/-1)
LABELS: ready, v1
BODY: Enable CUDA Graphs for DP + All2All kernels. ⏎  ⏎ Fixes: ⏎  1. The input buffers to the quant_method aren't captured properly when using CUDAGraphs + torch.compile. This PR introduces a staging area where the hidden_states and router_logits are copied into and it is this tensor that gets passed into quant_method. ⏎  2. It is important that all DP ranks invoke the same number of dispatch and combine kernels. The kernels need to synchronize between DP rank …[truncated]

### L3-269d901734  (L3, 2025-05-29, sha 269d90173432, PR #18100)
TITLE: [Bugfix][ROCm] fix the power of 2 exception from triton_unified_attention.py when running llama4 models and unit test fix (#18100)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+51/-55); tests/kernels/attention/test_triton_unified_attention.py (+3/-1)
LABELS: ready
BODY: FIX  *https://github.com/vllm-project/vllm/issues/18088* ⏎  ⏎ As detailed in the above issue, when running V1 on llama4 issues, we saw the exception that requires the parameter is a power of 2. However, when running on llama4 128E FP8 models, the following expression in (https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py#L97) is not a power of 2. ⏎ ``` ⏎  offs_m = tl.arange(0, BLOCK_Q * num_queries_per_kv) ⏎ ``` ⏎  ⏎  …[truncated]

### L3-da4b69d0b4  (L3, 2025-05-29, sha da4b69d0b435, PR #18275)
TITLE: [Attention][V1] Toggle for v1 attention backend (#18275)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.triton.prefix_prefill, L3.triton.chunked_prefill_paged_decode, L3.triton.v1_backend
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+2/-2); vllm/attention/ops/prefix_prefill.py (+2/-2); vllm/envs.py (+10/-2); vllm/v1/attention/backends/triton_attn.py (+6/-3)
LABELS: ready, v1
BODY: Expanding on https://github.com/vllm-project/vllm/pull/18093 ⏎ Adding a toggle to force fallback to the 2 stage attention kernel in V1 ⏎  ⏎ Including a small fix for the FP8 kv cache on ROCm in the 2 stage kernel approach

### L3-1b7cfd5a36  (L3, 2025-05-29, sha 1b7cfd5a367b, PR #18226)
TITLE: [ROCm][V0][Attention] Revert to the previous FA triton kernel (#18226)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L3.triton.flash_attention_rocm, L3.rocm.rocm_flash_attn_v0, L3.platform.rocm_selection
FILES: vllm/attention/backends/rocm_flash_attn.py (+3/-2); vllm/attention/ops/triton_flash_attention.py (+685/-1081); vllm/platforms/rocm.py (+6/-0)
LABELS: rocm, ready
DEEP_STUDY: deep-study revert record: explicit_rollback of PR(s) 12591 reason=performance_regression
BODY: Revert to the previous version of the triton attention kernel, modified to support FP8 computation. ⏎ The kernel brought in in https://github.com/vllm-project/vllm/pull/12591 turned out to have performance issues, and broken support for FP8 quantized models. ⏎ Until that is resolved we want to replace it from the performant version from the ROCm fork

### L3-77b6e74fe2  (L3, 2025-05-29, sha 77b6e74fe2b6, PR #18938)
TITLE: [ROCm] Remove unnecessary assertion of max_model_len in ROCM_AITER_MLA attention backend. (#18938)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/attention/backends/rocm_aiter_mla.py (+0/-2); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+0/-3)
LABELS: v1
BODY: This PR removes the unnecessary constraint and assertion of a specific `max_model_len value` from the `ROCM_AITER_MLA` attention backend on both VLLM v1 and v0 engines. ⏎  ⏎ ### lm_eval results on DeepSeek-V2-Lite-Chat with default engine args on both VLLM v0 and v1 engines. ⏎  ⏎ `VLLM_USE_V1=1 ⏎ VLLM_ROCM_USE_AITER=1 ⏎ VLLM_ROCM_USE_AITER_MOE=0 ⏎ VLLM_ROCM_USE_AITER_RMSNORM=0 ⏎ VLLM_ROCM_USE_AITER_LINEAR=0 ⏎ lm_eval --model vllm --model_args pretrained=deepseek-a …[truncated]

### L3-02f0c7b220  (L3, 2025-06-03, sha 02f0c7b22042, PR #19100)
TITLE: [Misc] Add SPDX-FileCopyrightText  (#19100)
SOURCES: path_core
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flash_attn.fa_utils, L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.prefix_prefill, L3.triton.flash_attention_rocm, L3.triton.decode_attention, L3.triton.chunked_prefill_paged_decode, L3.triton.unified_attention, L3.triton.v1_backend, L3.merge.triton_lse, L3.rocm.rocm_flash_attn_v0, L3.mla.triton_v0, L3.mla.common_v1, L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.mla.rocm_aiter, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: .buildkite/check-wheel-size.py (+1/-0); .buildkite/generate_index.py (+1/-0); .buildkite/lm-eval-harness/conftest.py (+1/-0); .buildkite/lm-eval-harness/test_lm_eval_correctness.py (+1/-0); .buildkite/nightly-benchmarks/scripts/convert-results-json-to-markdown.py (+1/-0); .buildkite/nightly-benchmarks/scripts/download-tokenizer.py (+1/-0); .buildkite/nightly-benchmarks/scripts/generate-nightly-markdown.py (+1/-0); .buildkite/nightly-benchmarks/scripts/get-lmdeploy-modelname.py (+1/-0); .buildkite/nightly-benchmarks/scripts/summary-nightly-results.py (+1/-0); benchmarks/backend_request_func.py (+1/-0); (+1422 more)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, tpu, speculative-decoding, ci/build, v1
BODY: Add a copyright clause to our codebase to serve as a catchall copyright clause. This is done after consultation with Linux Foundation.

### L3-fa98d77773  (L3, 2025-06-03, sha fa98d77773c6, PR #18434)
TITLE: [Kernel] DeepEP dispatch-combine kernel integration (#18434)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.platform.cuda_selection
FILES: csrc/moe/topk_softmax_kernels.cu (+14/-2); tests/kernels/moe/__init__.py (+0/-0); tests/kernels/moe/deepep_utils.py (+188/-0); tests/kernels/moe/test_deepep_deepgemm_moe.py (+371/-0); tests/kernels/moe/test_deepep_moe.py (+459/-0); vllm/config.py (+2/-0); vllm/distributed/device_communicators/all2all.py (+145/-1); vllm/distributed/device_communicators/cuda_communicator.py (+8/-0); vllm/envs.py (+2/-0); vllm/model_executor/layers/fused_moe/deep_gemm_moe.py (+18/-14); (+13 more)
LABELS: ready, v1
BODY: Integrate DeepEP dispatch-combine kernels ⏎  - Integrated DeepEP high-throughput and low-latency kernels  ⏎  - Integrate DeepEP high-throughput kernel with the corresponding DeepGemm kernel  ⏎   ⏎ Correctness: ⏎  - Tested correctness using lm_eval on H100 for, ⏎     Models: `deepseek-ai/DeepSeek-V2-Lite` `RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8` `Qwen/Qwen3-30B-A3B-FP8` ⏎     ALL2ALL Backend: `deepep_high_throughput` ⏎     Cases: for DP=2 TP=1 case. ⏎  - Test …[truncated]

### L3-e31446b6c8  (L3, 2025-06-03, sha e31446b6c8d8, PR #18844)
TITLE: [Perf] Tune `scaled_fp8_quant` by increasing vectorization (#18844)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/quantization/fp8/common.cu (+19/-16); csrc/quantization/fp8/common.cuh (+35/-33); csrc/quantization/fused_kernels/layernorm_utils.cuh (+50/-49); csrc/quantization/vectorization.cuh (+11/-12)
LABELS: performance, ready
BODY: Increase the vectorization from 4 to 16 elements and decrease the block_size to launch for kernel for the per-tensor and per-token fp8 quantization CUDA kernels. The improvements are visible on H100 and obvious on B200. ⏎  ⏎ ### Evaluations ⏎  ⏎ ``` ⏎ # Command for static per-tensor quantization ⏎ lm_eval --model vllm --model_args pretrained=RedHatAI/Meta-Llama-3.1-8B-FP8 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto ⏎  ⏎ # 0.9.0 ⏎ vllm (pret …[truncated]

### L3-4555143ea7  (L3, 2025-06-03, sha 4555143ea7fd, PR #16441)
TITLE: [CPU] V1 support for the CPU backend (#16441)
SOURCES: path_core, symbol_pickaxe, corpus:production-kernel-provenance
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/cpu_mla.py (+3/-3); vllm/attention/backends/torch_sdpa.py (+12/-4); vllm/v1/attention/backends/cpu_attn.py (+163/-0); .buildkite/scripts/hardware_ci/run-cpu-test.sh (+5/-8); docs/usage/v1_guide.md (+2/-0); requirements/cpu.txt (+3/-0); tests/kernels/attention/test_attention_selector.py (+4/-1); tests/models/language/generation/test_common.py (+0/-1); vllm/compilation/wrapper.py (+6/-1); vllm/engine/arg_utils.py (+3/-1); (+5 more)
LABELS: documentation, ready, ci/build, v1
ISSUES: #16056 [Usage]: v1 engine on CPU
BODY: resolve #16056  ⏎  ⏎ Support all features listed in the CPU doc excepts FP8 KV cache. ⏎  ⏎ ### Changes ⏎  ⏎ - Add V1 ```CPUWorker``` and ```CPUModelRunner```, derived from ```Worker``` and ```GPUModelRunner``` to reduce code duplication.  ⏎ - Add V1 ```TorchSDPABackend``` with compatible interfaces for ```GPUModelRunner```, such as ```reorder_batch``` and ```build```. ⏎ - Additional changes in ```GPUModelRunner``` to avoid importing flash-attn explicitly and usi …[truncated]

### L3-bdf13965ab  (L3, 2025-06-03, sha bdf13965ab4a, PR #18212)
TITLE: [V1] Support cross-layer KV sharing (#18212)
SOURCES: path_core
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.rocm_flash_attn_v0, L3.mla.triton_v0, L3.mla.common_v1, L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.mla.rocm_aiter, L3.dispatch.abstract_interface, L3.blocksparse.v0
FILES: vllm/attention/backends/abstract.py (+1/-0); vllm/attention/backends/blocksparse_attn.py (+3/-0); vllm/attention/backends/cpu_mla.py (+2/-1); vllm/attention/backends/dual_chunk_flash_attn.py (+3/-0); vllm/attention/backends/flash_attn.py (+3/-0); vllm/attention/backends/flashinfer.py (+3/-0); vllm/attention/backends/flashmla.py (+2/-1); vllm/attention/backends/hpu_attn.py (+3/-0); vllm/attention/backends/ipex_attn.py (+3/-0); vllm/attention/backends/mla/common.py (+3/-0); (+21 more)
LABELS: tpu, ready, v1
BODY: ## Motivation ⏎  ⏎ Some models like Tencent-Hunyuan-Large (#10043) and Hymba-1.5B-Base (#10783) use cross-layer KV sharing (e.g. [Cross-Layer Attention](https://arxiv.org/abs/2405.12981)). This PR adds the ability for KV caches to be shared between attention layers. ⏎  ⏎ ## Design ⏎ This PR adds a new argument `kv_sharing_target_layer_name: Optional[str] = None` to the `Attention` layer class. This is only supported in V1. To have an Attention layer not al …[truncated]

### L3-41aa578428  (L3, 2025-06-03, sha 41aa5784287f, PR #17625)
TITLE: [NVIDIA] Add Cutlass MLA backend (#17625)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:production-kernel-provenance
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.cutlass_kernels, L3.mla.cutlass_v1_backend, L3.platform.cuda_selection
FILES: csrc/attention/mla/cutlass_mla_kernels.cu (+1/-1); vllm/engine/arg_utils.py (+1/-0); vllm/platforms/cuda.py (+8/-0); vllm/platforms/interface.py (+1/-0); vllm/v1/attention/backends/mla/common.py (+1/-1); vllm/v1/attention/backends/mla/cutlass_mla.py (+96/-0); tests/kernels/test_cutlass_mla_decode.py (+3/-1)
LABELS: ready, v1
BODY: This PR introduces the `CUTLASS_MLA_VLLM_V1` backend, enabling support for `ops.cutlass_mla_decode()` on NVIDIA Blackwell GPUs. ⏎  ⏎ It also includes performance results using DeepSeek-V3 on 8×B200 GPUs under DP+EP parallelism settings, which delivers ~17% improved throughput. ⏎  ⏎ ``` ⏎ # With default triton backend: ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     2989 ⏎ Benchmark duration (s):                  10 …[truncated]

### L3-b124e1085b  (L3, 2025-06-03, sha b124e1085b1b, PR #19106)
TITLE: [Bugfix] Fix FA3 full cuda graph correctness (#19106)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+21/-8); vllm/v1/worker/gpu_model_runner.py (+5/-0); .buildkite/test-pipeline.yaml (+1/-0); tests/compile/piecewise/test_full_cudagraph.py (+5/-2)
LABELS: ready, ci/build, v1
DEEP_STUDY: deep-study correctness case vllm:b124e1085b: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: Fixes the correctness issue when using the full cuda graph, and adds `test_full_cudagraph.py` into the CI.

### L3-53a5a0ce30  (L3, 2025-06-04, sha 53a5a0ce30dd, PR #18778)
TITLE: [Perf] Tunings for SM100 FP8 CUTLASS kernel (#18778)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm100_fp8_dispatch.cuh (+51/-2)
LABELS: performance, ready
BODY: I noticed that the FP8 CUTLASS kernel for Blackwell only had one default set of configs. This PR adds new configs for small M < 128. ⏎  ⏎ For Llama 8B on B200, these tunings offer a GEMM improvement of: ⏎ * 1.7 to 2.5x speedup at M<64 ⏎ * 1.1 to 1.3x speedup at 64<=M<128 ⏎  ⏎ Accuracy eval: ⏎ ``` ⏎ lm_eval --model vllm --model_args pretrained=RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto ⏎ vllm  …[truncated]

### L3-c3fd4d669a  (L3, 2025-06-04, sha c3fd4d669a44, PR #19111)
TITLE: [Kernel] Integrate batched/masked deepgemm kernel (#19111)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/deepep_utils.py (+4/-1); tests/kernels/moe/test_deepep_deepgemm_moe.py (+166/-24); vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py (+124/-0); vllm/model_executor/layers/fused_moe/batched_triton_or_deep_gemm_moe.py (+116/-0); vllm/model_executor/layers/fused_moe/deepep_ll_prepare_finalize.py (+55/-21); vllm/model_executor/layers/quantization/fp8.py (+7/-5)
LABELS: ready
BODY: ## Purpose ⏎   Integration of batched deepgemm kernels. These kernels are Gemm kernels used in the fused MOE operation for block-quantized matmuls.  ⏎  ⏎ ## Test  ⏎  - Added unit tests. Verified that the tests pass locally on an H100. ⏎  - Verified correctness with lm_eval on the `Qwen/Qwen3-30B-A3B-FP8` model for DP=2, TP=1, Expert-Parallel case ⏎  ⏎ ## Test Result ⏎ sever command: `VLLM_ALL2ALL_BACKEND="deepep_low_latency" VLLM_USE_DEEP_GEMM=1 vllm serve Qwen/ …[truncated]

### L3-b2fac67130  (L3, 2025-06-04, sha b2fac67130b1, PR #18833)
TITLE: [P/D] Heterogeneous TP (#18833)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+16/-0); tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh (+8/-3); tests/v1/kv_connector/nixl_integration/test_accuracy.py (+1/-0); vllm/distributed/kv_transfer/kv_connector/utils.py (+17/-1); vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py (+243/-95); vllm/worker/worker_base.py (+2/-1)
LABELS: ready, v1, kv-connector
BODY: Yet another take at https://github.com/vllm-project/vllm/pull/18079. It builds on the same commits so the rationale of the PR is the same. ⏎  ⏎ The issue with the previous approach is that it appears using a higher number of descriptors per read -`block_size` as many, each region smaller by  `block_size` times, so total number of bytes moved is unchanged- causes significant slowdowns. Mind that this is not happening for homogenous TP, where memory re …[truncated]

### L3-87360308b7  (L3, 2025-06-05, sha 87360308b7ee, PR #19118)
TITLE: [V1] Use FlashInfer by default on Blackwell GPUs (#19118)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+15/-0); vllm/platforms/interface.py (+24/-0)
LABELS: performance, ready, v1
BODY: ## Purpose ⏎  ⏎ FlashInfer has a specific backend for NVIDIA Blackwell so it is much more performant than FlashAttention2 default in V1 (FA3 is unsupported). If a user has it installed, I think we should choose it by default, which this PR achieves. See this comment for benchmarks https://github.com/vllm-project/vllm/pull/18095#issuecomment-2877849390 ⏎  ⏎ This PR also adds `is_device_capability` to the platform interface for exact cc checking since we o …[truncated]

### L3-18093084be  (L3, 2025-06-05, sha 18093084be93, PR #19138)
TITLE: [Misc] Remove unnecessary fallback to prefill-decode attention (#19138)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+1/-4)
LABELS: ready, v1
BODY: ## Purpose ⏎ This PR removes an unnecessary fallback to prefill-decode attention in `vllm/v1/attention/backends/triton_attn.py`. The code currently falls back to prefill-decode attention if the number of queries per key and value is of power 2. This was introduced in #18093 to overcome an issue with unified attention. However,  after the fix in #18100, this condition is not needed anymore. ⏎ ## Test Plan ⏎ Run lm_eval with and without prefill-decode at …[truncated]

### L3-9ef9173cfa  (L3, 2025-06-05, sha 9ef9173cfa33, PR #19090)
TITLE: [P/D][NixlConnector] Enable FlashInfer backend (#19090)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/platforms/interface.py (+1/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py (+50/-15)
LABELS: ready
BODY: This PR enables the use of `VLLM_ATTENTION_BACKEND=FLASHINFER` in disaggregated prefill setups leveraging NixlConnector (which is currently allowed but broken on main). ⏎  ⏎ The main difference wrt default FA backend is that FlashInfer swaps the cache first two dims (`K/V` and `num_blocks`) resulting in `[num_blocks, KV(2), N,H,D]`.  The easiest approach here is to maintain the layout and just transfer the whole region (for each layer) instead of try …[truncated]

### L3-84166fee97  (L3, 2025-06-06, sha 84166fee9770, PR #18762)
TITLE: [Kernel] Integrate CUTLASS MoE kernel with PPLX (#18762)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+2/-2); benchmarks/kernels/benchmark_grouped_gemm_cutlass.py (+11/-50); csrc/ops.h (+10/-1); csrc/quantization/cutlass_w8a8/moe/grouped_mm_c3x.cu (+19/-10); csrc/quantization/cutlass_w8a8/moe/grouped_mm_c3x.cuh (+2/-4); csrc/quantization/cutlass_w8a8/moe/moe_data.cu (+38/-4); csrc/quantization/cutlass_w8a8/scaled_mm_entry.cu (+36/-3); csrc/torch_bindings.cpp (+18/-1); tests/kernels/moe/test_cutlass_moe.py (+2/-6); tests/kernels/moe/test_pplx_cutlass_moe.py (+287/-0); (+16 more)
LABELS: ready, ci/build
BODY: Integrate CUTLASS MoE fp8 kernels with PPLX. ⏎  ⏎ Unit tests: ⏎ ``` ⏎ tests/kernels/moe/test_pplx_cutlass_moe.py ⏎ ``` ⏎  ⏎ E2E testing: ⏎ ``` ⏎ export MASTER_ADDR=127.0.0.1 ⏎ export MASTER_PORT=29500 ⏎ export VLLM_ALL2ALL_BACKEND=pplx ⏎ python3 examples/offline_inference/data_parallel.py \ ⏎         --model="nm-testing/DeepSeek-Coder-V2-Lite-Instruct-FP8" \ ⏎         --dp-size=2 \ ⏎         --tp-size=1 \ ⏎         --trust-remote-code ⏎ ```

### L3-cf02f9b283  (L3, 2025-06-06, sha cf02f9b283a6, PR #16078)
TITLE: Add FlexAttention to V1 (#16078)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.platform.cuda_selection, L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+477/-0); tests/kernels/test_flex_attention.py (+93/-0); vllm/engine/arg_utils.py (+1/-0); vllm/platforms/cuda.py (+3/-0); vllm/platforms/interface.py (+1/-0)
LABELS: documentation, ready, ci/build, v1
BODY: # Summary ⏎ This PR adds FlexAttention as a new unified_attention backend for the V1 engine.  ⏎  ⏎ This requires torch > 2.7 since we fixed a number of dynamic shapes issues that show up by default here. ⏎  ⏎ ### Design ⏎  ⏎ FlexAttention is broken up into two distinct phases, block mask creation and the call to forward. For most Transformers they N attention layers share a common attention pattern and thus we can amortize the cost of block mask creation over  …[truncated]

### L3-8058c91108  (L3, 2025-06-09, sha 8058c91108a3, PR #19374)
TITLE: [HOT-FIX] Add `kv_sharing_target_layer_name` argument to cutlass_mla backend (#19374)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.cutlass_v1_backend
FILES: vllm/v1/attention/backends/mla/cutlass_mla.py (+2/-1)
LABELS: bug, ready, v1
BODY: Add `kv_sharing_target_layer_name` argument ⏎ ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update
