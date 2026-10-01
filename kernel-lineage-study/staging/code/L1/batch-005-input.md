### L1-0242bb9c74  (L1, 2025-08-03, sha 0242bb9c7437, PR #8732)
TITLE: Fix triton kernels topk with keyword arguments (#8732)
SOURCES: path_core, symbol_pickaxe
STAGE1: repair_correctness; artifacts=L1.routing.topk_py; Fixes topk.py call compatibility with triton_kernels keyword arguments.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+22/-3)
BODY: ## Motivation ⏎  ⏎ For the triton kernels topk routing function, the meaning of `sm_first` is not exactly the same as `renormalize`. Use keyword arguments to avoid confusion and unknown bugs. ⏎ - If `sm_first` = True, the softmax will before the topk, so the result can be unnormalized before sort. `renormalize` is not supported in triton kernels rounting funciton. ⏎ - If `sm_first` = False, the softmax will after the topk, so the result is already normalized, so we don't need to renormalize.

### L1-e67276ecb3  (L1, 2025-08-03, sha e67276ecb305, PR #8678)
TITLE: feat: support cutlass_moe_fp8 kernel for fusedmoe in sm90 (#8678)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L1.cutlass.adapters; Adds SM90 cutlass_moe_fp8 kernel support for FusedMoE path.
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+20/-6); python/sglang/srt/layers/quantization/fp8.py (+3/-3); python/sglang/srt/layers/utils.py (+9/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ### reproduce ⏎ #### Qwen3-MOE, H20, TP4 ⏎ * server ⏎ ``` ⏎ SGL_ENABLE_JIT_DEEPGEMM=1 SGLANG_CUTLASS_MOE=1 python3 -m sglang.launch_server --model-path /data/models/Qwen3-235B-A22B-FP8 --tp 4 --trust-remote-code --host 0.0.0.0 --port 8080 --mem-fraction-static 0.7  --max-running-requests 128 --context-length 4096 --chunked-prefill-size 4096 ⏎ ``` ⏎ * gsm8k ⏎ ``` ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 320 --parallel 80 --port 8080 ⏎ ``` ⏎ <img width="1476" height="172" alt="image" src="https://github.com/user-attachments/assets/dc0ed1ad-3030-4f15-8ce5-4ce4b37f6fb5" /> ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9f47d686e5  (L1, 2025-08-03, sha 9f47d686e521, PR #8709)
TITLE: Fix fused MoE when `routed_scaling_factor is None` (#8709)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L1.ep.layer; Fixes EPMoE when routed_scaling_factor is None.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-1)
BODY: Reproduction: ⏎  ⏎ ```bash ⏎ python3 -m sglang.compile_deep_gemm --host=0.0.0.0 --port=8000 --model=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 --tp=8 --trust-remote-code --enable-ep-moe --tool-call-parser=qwen25 --reasoning-parser=qwen3 --context-length=131072 ⏎ ``` ⏎  ⏎ Issue: ⏎  ⏎ ```bash ⏎   File "/root/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py", line 283, in forward_deepgemm ⏎     return output * self.routed_scaling_factor ⏎ TypeError: unsupported operand type(s) for *: 'Tensor' and 'NoneType' ⏎ ```

### L1-b102353f8f  (L1, 2025-08-03, sha b102353f8f2d, PR #8735)
TITLE: [MoE] Enable `renormalize=False` in Triton kernels (#8735)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: extend_support; artifacts=L1.routing.topk_py; Enables renormalize=False routing behavior for Triton kernels through topk.py.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+4/-22)
BODY: ## Motivation ⏎  ⏎  ⏎ Fix incorrect fix in #8732. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-fee0ab0fba  (L1, 2025-08-03, sha fee0ab0fba12, PR #8294)
TITLE: [CI] Ascend NPU CI enhancement (#8294)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L1.hardware.cpu_npu_musa,L1.routing.topk_py; Changes top-k sorted argument expression to satisfy NPU compiler limitation.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+10/-2); .github/workflows/pr-test-npu.yml (+67/-3); scripts/npu_ci_install_dependency.sh (+36/-24); test/srt/run_suite.py (+8/-2); test/srt/test_ascend_attention_backend.py (+0/-62); test/srt/test_ascend_mla_backend.py (+0/-96); test/srt/test_ascend_mla_w8a8int8.py (+100/-0); test/srt/test_ascend_tp1_bf16.py (+96/-0); test/srt/test_ascend_tp2_bf16.py (+98/-0)
LABELS: ready-to-merge, npu
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ To enhance ascend npu ci by covering all the features we have already supported on sglang. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Add a new test case covering Ascend-specific W8A8-int8 quant method ⏎ - Add a new test case covering tp=2 scenario ⏎ - Merge attention backend test case into tp=1 bf16 scenario ⏎ - Merge attention mla backend test case into W8A8-int8 scenario with DeepSeek-V2-Lite model ⏎ - **_Fix a functional issue on Ascend NPU that can't support directly evaluating a comparison in torch compile_** ⏎  ⏎ ## Checklist

### L1-915140fd18  (L1, 2025-08-04, sha 915140fd18c9, PR #8552)
TITLE: [NVIDIA] Add Low Latency NVFP4 decode kernels from Flashinfer (#8552)
SOURCES: path_core, symbol_pickaxe, corpus:kernel-correctness-cases(introducing)
STAGE1: integrate; artifacts=L1.upstream.flashinfer_moe,L1.ep.layer; Adds FlashInfer low-latency NVFP4 MoE decode path in EP/fused MoE layers.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+18/-7); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+173/-16); python/sglang/srt/layers/moe/utils.py (+16/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+260/-63); python/sglang/srt/managers/schedule_batch.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+25/-24); python/sglang/srt/models/glm4_moe.py (+2/-4); python/sglang/srt/server_args.py (+7/-0)
LABELS: high priority
DEEP_STUDY: deep-study: introduced the defect fixed in case sglang:b01eeb80f8 (fix PR 8779) || deep-study: introduced the defect fixed in case sglang:6d0646da11 (fix PR 8773)
PERF_LINES: Bring best low latency NVFP4 kernels for Blackwell MoE. Currently enabling DSR1.
BODY: ## Motivation ⏎  ⏎ Bring best low latency NVFP4 kernels for Blackwell MoE. Currently enabling DSR1.  ⏎  ⏎ ## Modifications ⏎  ⏎ Changing some weight preprocessing logic as well as exposing these kernels. Plus various piping to make it work.    ⏎  ⏎ ## Accuracy Test ⏎  ⏎ Ran accuracy tests, find description and repro below.  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎ Added below with repros.  ⏎  ⏎ ## Checklist

### L1-9bd4872a34  (L1, 2025-08-04, sha 9bd4872a343e, PR #8768)
TITLE: [bugfix] Fix typo in modelopt quant: 'FusedMoE' object has no attribute 'local_num_experts' (#8768)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=NEW:modelopt_moe_adapter; Fixes ModelOpt quant adapter typo accessing FusedMoE local_num_experts.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ After https://github.com/sgl-project/sglang/pull/8552 was merged, I get this error `AttributeError: 'FusedMoE' object has no attribute 'local_num_experts'`. It should be `num_local_experts` instead.

### L1-fc8c8e5041  (L1, 2025-08-04, sha fc8c8e504156, PR #8762)
TITLE: Integrate triton_kernels in sgl-kernel (#8762)
SOURCES: dependency_pin
STAGE1: integrate; artifacts=L1.upstream.openai_triton_kernels; Adds CMake integration for triton_kernels inside sgl-kernel.
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+16/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-6d0646da11  (L1, 2025-08-04, sha 6d0646da1145, PR #8773)
TITLE: [NVIDIA] Fix breakage of using trtllm-gen fp8 moe (#8773)
SOURCES: path_core, symbol_pickaxe, corpus:kernel-correctness-cases
STAGE1: repair_correctness; artifacts=L1.upstream.flashinfer_moe,L1.ep.layer; Fixes trtllm-gen FP8 MoE breakage caused by prior FlashInfer NVFP4 changes.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+4/-62); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+14/-1)
LABELS: bug, high priority
DEEP_STUDY: deep-study correctness case sglang:6d0646da11: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=#8552
BODY: This PR fixes a breakage caused by [this PR](https://github.com/sgl-project/sglang/pull/8552). ⏎  ⏎ cc. @kushanam

### L1-08f8f49016  (L1, 2025-08-04, sha 08f8f4901650, PR #8212)
TITLE: [CPU][sgl-kernel] biased_grouped_topk: fix correction_bias dtype to float32 (#8212)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L1.hardware.cpu_npu_musa; Fixes CPU biased_grouped_topk correction_bias dtype to float32.
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: sgl-kernel/csrc/cpu/topk.cpp (+28/-25); sgl-kernel/csrc/cpu/common.h (+39/-0); sgl-kernel/csrc/cpu/vec.h (+19/-0); test/srt/cpu/test_topk.py (+8/-3)
LABELS: sgl-kernel, ready-to-merge, intel, cpu
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ https://github.com/sgl-project/sglang/pull/7825 has fixed the dtype of `e_score_correction_bias` to float32 instead of bfloat16. We need to fix the support in the topk kernel. ⏎ UT has been updated accordingly.

### L1-b01eeb80f8  (L1, 2025-08-04, sha b01eeb80f840, PR #8779)
TITLE: [NVIDIA]Fix local_num_experts for EP (#8779)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; Fixes local_num_experts handling for EP in FusedMoE path.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+2/-1)
LABELS: bug, high priority
DEEP_STUDY: deep-study: this PR was reverted by PR 8797 (confirmed_revert, reason=unstated) || deep-study correctness case sglang:b01eeb80f8: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=#8552
BODY: ## Motivation ⏎  ⏎ This PR fixes a bug in https://github.com/sgl-project/sglang/pull/8552. At create_weight, the num_experts and num_loca_experts should both be passed in for EP case. ⏎  ⏎ @kaixih @kushanam  ⏎  ⏎ cc. @kushanam ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5e91fed1c5  (L1, 2025-08-04, sha 5e91fed1c593, PR #8797)
TITLE: Revert "[NVIDIA]Fix local_num_experts for EP (#8779)" (#8797)
SOURCES: path_core
STAGE1: revert; artifacts=L1.triton.fused_moe; Reverts prior local_num_experts EP fix.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-2); python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-2)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 8779 reason=unstated
BODY: This reverts commit b01eeb80f8406cba569af5deb40f394293f9950d. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-c1d2061f97  (L1, 2025-08-05, sha c1d2061f97ae, PR #8824)
TITLE: Add initial support for gpt-oss (#8824)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L1.runner.openai_triton_kernels,L1.triton.fused_moe; Adds GPT-OSS support in FusedMoE and triton_kernels_moe adapter.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.openai_triton_kernels, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+134/-8); python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+178/-3); python/sglang/srt/layers/attention/triton_backend.py (+85/-14); python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+17/-0); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+36/-8); python/sglang/srt/layers/linear.py (+0/-5); python/sglang/srt/layers/quantization/fp8_utils.py (+29/-0); python/sglang/srt/layers/quantization/mxfp4_tensor.py (+133/-0); python/sglang/srt/layers/quantization/unquant.py (+52/-7); python/sglang/srt/managers/schedule_batch.py (+4/-2); python/sglang/srt/models/gpt_oss.py (+923/-0); python/sglang/srt/server_args.py (+4/-0)
LABELS: high priority
BODY: Future progress will be tracked here: https://github.com/sgl-project/sglang/issues/8833 ⏎  ⏎ **This PR only works for FP8/BF16 ckpt. The FP8/BF16 ckpt has been uploaded to:** ⏎ `lmsys/gpt-oss-20b-bf16` and `lmsys/gpt-oss-120b-bf16` ⏎  ⏎ Install SGLang: ⏎ 1. local build: `pip install -e "python[all]"` ⏎ 2. normal install should also be fine (you might see version conflict complaints, but should be fine) ⏎  ⏎ Additional install for gpt-oss: ⏎ ``` ⏎ pip3 install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu126 ⏎ pip3 install sgl-kernel==0.3.2 ⏎ ``` ⏎  ⏎ Launch server examples: ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path lmsys/gpt-oss-20b-bf16 ⏎ python3 -m sglang.launch_server --model-path lmsys/gpt-oss-120b-bf16 --tp 4 ⏎ ```

### L1-168033d5fb  (L1, 2025-08-06, sha 168033d5fb1e, PR #8843)
TITLE: Support mxfp4 for GPT-OSS (#8843)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L1.runner.openai_triton_kernels,L1.triton.fused_moe; Adds GPT-OSS MXFP4 support through FusedMoE and triton_kernels_moe changes.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.openai_triton_kernels, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+58/-6); python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+25/-15); python/sglang/srt/layers/quantization/__init__.py (+12/-2); python/sglang/srt/layers/quantization/fp4.py (+28/-293); python/sglang/srt/layers/quantization/mxfp4.py (+443/-0); python/sglang/srt/layers/quantization/unquant.py (+2/-0); python/sglang/srt/models/gpt_oss.py (+209/-9); python/sglang/srt/server_args.py (+10/-0); python/sglang/srt/utils.py (+4/-0)
BODY: 

### L1-288ae41f7a  (L1, 2025-08-06, sha 288ae41f7ae9, PR #8811)
TITLE: [NVIDIA] Fix num_experts in modelopt_quant (#8811)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; Fixes num_experts handling in ModelOpt/FusedMoE EP path.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+2/-4)
DEEP_STUDY: deep-study correctness case sglang:288ae41f7a: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
PERF_LINES: Latency: 697.093 s | Output throughput: 139.871 token/s | Latency: 636.827 s | Output throughput: 152.779 token/s | Latency: 633.068 s | Output throughput: 154.200 token/s
BODY: A hotfix for https://github.com/sgl-project/sglang/pull/8779.  ⏎ The trtllm_fp4_block_scale_moe API checks for the routing_logits dim to be consistent with num_experts(global). But previously at create_weight, the num_experts is overwritten by num_local_experts.  ⏎ cc. @kushanam  @zhyncs  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎ With EP enabled: ⏎ Accuracy: 0.963 ⏎ Invalid: 0.000 ⏎ Latency: 697.093 s ⏎ Output throughput: 139.871 token/s ⏎  ⏎ With TP enabled ⏎ Accuracy: 0.956 ⏎ Invalid: 0.000 ⏎ Latency: 636.827 s ⏎ Output throughput: 152.779 token/s ⏎ ## Benchmark & Profiling ⏎  ⏎ With shared expert fusion ⏎ Accuracy: 0.963 ⏎ Invalid: 0.000 ⏎ Latency: 633.068 s ⏎ Output throughput: 154.200 token/s ⏎  ⏎  ⏎ ## Checklist

### L1-4373df5525  (L1, 2025-08-06, sha 4373df55258e, PR #8847)
TITLE: add flashinfer mxfp4 (#8847)
SOURCES: path_core, symbol_pickaxe
STAGE1: integrate; artifacts=L1.runner.flashinfer_mxfp4,L1.upstream.flashinfer_moe; Adds FlashInfer MXFP4 MoE integration and server flag wiring.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+20/-2); python/sglang/srt/layers/quantization/mxfp4.py (+195/-19); python/sglang/srt/server_args.py (+15/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-47824c1488  (L1, 2025-08-07, sha 47824c14881b, PR #8898)
TITLE: [Perf] Auto enable best flashinfer mxfp4  kernel in b200 (#8898)
SOURCES: path_core, symbol_pickaxe
STAGE1: change_default; artifacts=L1.runner.flashinfer_mxfp4,L1.upstream.flashinfer_moe; Auto-enables the best FlashInfer MXFP4 MoE kernel on B200 configurations.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-4); python/sglang/srt/layers/quantization/mxfp4.py (+24/-27); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/models/gpt_oss.py (+9/-5); python/sglang/srt/server_args.py (+10/-12)
PERF_LINES: |    |   max_concurrency |   input_throughput |   output_throughput |   mean_ttft_ms |   median_ttft_ms |   p99_ttft_ms |   mean_tpot_ms |   median_tpot_ms |    | |    |   max_concurrency |   input_throughput |   output_throughput |   mean_ttft_ms |   median_ttft_ms |   p99_ttft_ms |   mean_tpot_ms |   median_tpot_ms |    | |    |   max_concurrency |   input_throughput |   output_throughput |   me
BODY: ### GPT-OSS-20B ⏎  ⏎ - SGLANG_USE_FLASHINFER_MXFP4_BF16_MOE=1 ⏎  ⏎ ```markdown ⏎ CUDA_VISIBLE_DEVICES=1 SGLANG_USE_FLASHINFER_MXFP4_BF16_MOE=1 python3 -m sglang.launch_server --model-path openai/gpt-oss-20b --tp-size 1 --port 30001 ⏎  ⏎ curl http://127.0.0.1:30001/flush_cache ⏎ python3 -m sglang.bench_serving --backend sglang-oai  --dataset-name random --random-input-len 512 --random-output-len 1024 --random-range-ratio 1 --num-prompts 20 --max-concurrency 1 --output-file res.jsonl --port 30001 --warmup-requests 10 ⏎ curl http://127.0.0.1:30001/flush_cache ⏎ python3 -m sglang.bench_serving --backend sglang-oai  --dataset-name random --random-input-len 512 --random-output-len 1024 --random-range-ratio 1 --num-prompts 200 --max-concurrency 32 --output-file res.jsonl --port 30001 --warmup-requests 10 ⏎  ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |    |   max_concurrency |   input_throughput |   output_throughput |   mean_ttft_ms |   median_ttft_ms |   p99_ttft_ms |   mean_tpot_ms |   median_tpot_ms |   p99_tpot_ms |   per_user_throughput | ⏎ +====+===================+====================+=====================+================+==================+===============+================+==================+===============+=======================+ ⏎ |  0 |             1.000 |            114.520 |             229.040 |         72.553 |           55.199 |       194.158 |          4.298 |            3.946 |         6.897 |               229.040 | ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |  1 |            32.000 |           2301.694 |            4603.388 |        304.275 |          245.623 |       760.550 |          6.043 |            6.123 |         6.377 |               143.856 | ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ ``` ⏎  ⏎ - SGLANG_USE_FLASHINFER_MXFP4_MOE=1 ⏎  ⏎ ```markdown ⏎ CUDA_VISIBLE_DEVICES=1 SGLANG_USE_FLASHINFER_MXFP4_MOE=1 python3 -m sglang.launch_server --model-path openai/gpt-oss-20b --tp-size 1 --port 30001 ⏎  ⏎ curl http://127.0.0.1:30001/flush_cache ⏎ python3 -m sglang.bench_serving --backend sglang-oai  --dataset-name random --random-input-len 512 --random-output-len 1024 --random-range-ratio 1 --num-pro …[truncated]

### L1-39fd178831  (L1, 2025-08-07, sha 39fd1788311c, PR #8720)
TITLE: refactor: Move scalar_types.py to sgl-kernel to avoid circular import (#8720)
SOURCES: path_core
STAGE1: repair_build_dependency; artifacts=L1.runner.marlin; Moves scalar_types into sgl-kernel to break circular import used by fused_marlin_moe.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: sgl-kernel/python/sgl_kernel/fused_moe.py (+1/-1); sgl-kernel/python/sgl_kernel/scalar_type.py (+352/-0); sgl-kernel/tests/test_marlin_repack.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ This PR addresses a Python circular import issue encountered in  https://github.com/sgl-project/sglang/pull/8112.  ⏎  ⏎ We believe that placing the `scalar_types`  inside the sgl-kernel will be better.  ⏎  ⏎ Please merge this PR first to ensure the dependent PR works as expected. ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-76915d68a8  (L1, 2025-08-07, sha 76915d68a8f8, PR #8950)
TITLE: Fix enable flashinfer mxfp4 moe  bf16 check (#8950)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: change_default; artifacts=L1.runner.flashinfer_mxfp4; Fixes guard so BF16 GPT-OSS bypasses MXFP4 fused_moe path.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+9/-8)
BODY: ## Motivation ⏎  ⏎ Fix running lmsys/gpt-oss-20b-bf16, which should bypass fused_moe with mxfp4 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1d24db8348  (L1, 2025-08-08, sha 1d24db834803, PR #8944)
TITLE: Expert Parallelism for GPT-OSS (#8944)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: extend_support; artifacts=L1.ep.layer,L1.triton.fused_moe; Adds expert parallelism support for GPT-OSS in EP and FusedMoE code.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+6/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+101/-12); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-2); python/sglang/srt/layers/quantization/mxfp4.py (+80/-52); python/sglang/srt/layers/quantization/unquant.py (+9/-2); python/sglang/srt/models/gpt_oss.py (+54/-47); python/sglang/srt/server_args.py (+10/-4); python/sglang/srt/utils.py (+5/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ - What's in this PR: ⏎   - Enable GPT-OSS launch without triton-kernels ⏎   - Support expert parallelism for GPT-OSS ⏎  ⏎ Example: ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model openai/gpt-oss-120b --tp 4 --ep 2 ⏎ ``` ⏎  ⏎ - TODO: ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b4c9f38a76  (L1, 2025-08-08, sha b4c9f38a76bd, PR #8955)
TITLE: [NVIDIA] Fix missing `get_col_major_tma_aligned_tensor` for Blackwell deepgemm in EpMoE (#8955)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.ep.layer,L1.runner.deep_gemm; Fixes Blackwell DeepGEMM EPMoE failure by supplying TMA-aligned scale tensors.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+49/-4)
BODY: EpMoE on Blackwell with deepgemm currently fails with the following error due to the absence of get_col_major_tma_aligned_tensor: ⏎  ⏎ ```  ⏎  File "/sgl-workspace/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py", line 204, in forward_deepgemm ⏎     deep_gemm_wrapper.get_col_major_tma_aligned_tensor(gateup_input_scale), ⏎     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ AttributeError: module 'sglang.srt.layers.quantization.deep_gemm_wrapper' has no attribute 'get_col_major_tma_aligned_tensor' ⏎ ``` ⏎  ⏎ This PR addresses the issue by adding the missing function or ensuring compatibility with Blackwell deepgemm. ⏎  ⏎ cc. @kushanam @zhyncs

### L1-591c232f7c  (L1, 2025-08-08, sha 591c232f7c9e, PR #8770)
TITLE: [1/2][resubmit] sgl-kernel: Fuse routed scaling factor into moe_fused_gate (select_experts) (#8770)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts(reverted)
STAGE1: reland; artifacts=L1.routing.fused_gate,L1.routing.topk_py; Relands routed_scaling_factor fusion in moe_fused_gate with topk.py test support.
ARTIFACT_HINTS: L1.routing.topk_py, L1.routing.fused_gate
FILES: python/sglang/srt/layers/moe/topk.py (+24/-0); sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/csrc/moe/moe_fused_gate.cu (+20/-7); sgl-kernel/include/sgl_kernel_ops.h (+2/-1); sgl-kernel/python/sgl_kernel/moe.py (+9/-2); sgl-kernel/tests/test_moe_fused_gate.py (+6/-1)
DEEP_STUDY: deep-study: this PR was reverted by PR 9035 (confirmed_revert, reason=unstated)
PERF_LINES: 10.46% speedup at BS 1 | 1.86% speedup at BS 128 | 1.22% speedup at BS1 | 0.26% speedup at BS128
BODY: ## Motivation ⏎  ⏎ Resubmit of https://github.com/sgl-project/sglang/pull/8364 which was missing changes required for the unit test. ⏎  ⏎ https://github.com/sgl-project/sglang/pull/8690 will enable using this fusion for deepseek. ⏎  ⏎ Prefill: ⏎ 10.46% speedup at BS 1 ⏎ 1.86% speedup at BS 128 ⏎ Decode: ⏎ 1.22% speedup at BS1 ⏎ 0.26% speedup at BS128

### L1-3f2e315f6e  (L1, 2025-08-09, sha 3f2e315f6e5b, PR #8962)
TITLE: optimize: reduce shulffle and quantization overhead in cutlass_moe sm90 (#8962)
SOURCES: path_core
STAGE1: optimize; artifacts=L1.cutlass.adapters; Optimizes cutlass_moe SM90 by quantizing before shuffle and transposing scales.
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+11/-16); python/sglang/srt/layers/quantization/fp8_kernel.py (+59/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ In https://github.com/sgl-project/sglang/pull/8678, `per_token_group_quant_fp8_hopper_moe_mn_major` is called after `shuffle_rows`. This approach shuffles hidden_states in bf16 and increases the tokens to quantize by top-k fold,  which is inefficient. We optimize this workflow by first quantizing the hidden_states prior to shuffling, and then utilizing the `per_group_transpose` Triton kernel to handle scale transposition. ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ * gsm8k: 95.3 ⏎ <img width="2090" height="142" alt="image" src="https://github.com/user-attachments/assets/ae3e4e1f-91f2-456f-9338-3c67ff2e0a8f" /> ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎ * main ⏎ <img width="2514" height="886" alt="image" src="https://github.com/user-attachments/assets/2490aaf1-149a-4d4a-9071-f105d53b92b0" /> ⏎  ⏎ * this PR ⏎ <img width="3024" height="1646" alt="image" src="https://github.com/user-attachments/assets/f63e69e5-dea0-4fa8-b901-8ba15e602d10" /> ⏎  ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-137e75daa1  (L1, 2025-08-09, sha 137e75daa1d3, PR #8355)
TITLE: [Feature] Optimize DeepSeek's DeepEP on Ascend NPU (#8355)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: optimize; artifacts=L1.hardware.cpu_npu_musa,L1.ep.deepep_dispatcher; Adds Ascend/NPU DeepEP-compatible fusion kernels and dispatcher changes.
ARTIFACT_HINTS: L1.routing.topk_py, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+60/-2); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+61/-24); python/sglang/srt/layers/moe/topk.py (+2/-1); python/sglang/srt/layers/quantization/w8a8_int8.py (+39/-31); python/sglang/srt/distributed/parallel_state.py (+4/-2); python/sglang/srt/layers/attention/ascend_backend.py (+3/-0); python/sglang/srt/layers/rotary_embedding.py (+41/-1)
LABELS: high priority, ready-to-merge
PERF_LINES: Following our roadmap #8004, this is the first pr to support deepep expert parallelism and speed up with some fusion kernels. Now it is possible to activate `--
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Following our roadmap #8004, this is the first pr to support deepep expert parallelism and speed up with some fusion kernels. Now it is possible to activate `--enable-deepep-moe` with `--deepep-mode low_latency` only under pd disaggregation scenario on decode nodes. However, we only support W8A8 int8 quant method currently. ⏎  ⏎ More info about our DeepEP-compatible kernels, check out our roadmap in [sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu/issues/6) ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - a new DeepEPMoE impl w/ gemm and swiglu fusion kernels ⏎ - gating_topk kernel using fp32 precision to improve accuracy ⏎ - introducing fused rms_norm kernel and add_rms_norm kernel ⏎ - introducing fused rope kernel under certain condition, waiting to be generalized ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Accuracy & Performance ⏎  ⏎ Accuracy with `python3 -m sglang.test.few_shot_gsm8k --num-questions 1318` ⏎  ⏎ <img width="1088" height="90" alt="image" src="https://github.com/user-attachments/assets/75fc026c-056e-4dba-97a3-f8eb333e7ccb" /> ⏎  ⏎ Performance on 2xA3 PD disaggregation w/o DP-ATTN ⏎  ⏎ <img width="330" height="186" alt="image" src="https://github.com/user-attachments/assets/be0c97e0-916d-4fea-bc28-f2173c09dddb" /> ⏎  ⏎ ## Code Format ⏎  ⏎ <img width="656" height="354" alt="image" src="https://github.com/user-attachments/assets/9f492c9c-ebda-4671-9f4b-d50691482069" />

### L1-ef48d5547e  (L1, 2025-08-09, sha ef48d5547ec9, PR #9013)
TITLE: Fix CI (#9013)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L1.hardware.cpu_npu_musa,L1.routing.topk_py; Adds CPU topk signature parameter for routed-scaling protocol, asserting unsupported output scaling.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-0); .github/workflows/cancel-all-pending-pr-test-runs.yml (+25/-11); .github/workflows/execute-notebook.yml (+1/-1); .github/workflows/pr-test-xeon.yml (+1/-1); .github/workflows/pr-test.yml (+7/-3); test/srt/run_suite.py (+64/-48); test/srt/test_bench_serving.py (+1/-1); test/srt/test_gpt_oss_1gpu.py (+6/-6); test/srt/test_gpt_oss_common.py (+13/-4)
BODY: - Fix the threshold in CI ⏎ - Reorganize the CI pipeline ⏎ - Fix the CPU CI

### L1-dd949ace23  (L1, 2025-08-10, sha dd949ace23d6, PR #9035)
TITLE: Revert "[1/2][resubmit] sgl-kernel: Fuse routed scaling factor into m… (#9035)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts
STAGE1: revert; artifacts=L1.routing.fused_gate,L1.routing.topk_py; Reverts relanded routed_scaling_factor fusion in moe_fused_gate/select_experts.
ARTIFACT_HINTS: L1.routing.topk_py, L1.routing.fused_gate
FILES: python/sglang/srt/layers/moe/topk.py (+0/-24); sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/csrc/moe/moe_fused_gate.cu (+7/-20); sgl-kernel/include/sgl_kernel_ops.h (+1/-2); sgl-kernel/python/sgl_kernel/moe.py (+2/-9); sgl-kernel/tests/test_moe_fused_gate.py (+1/-6)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 8770 reason=unstated
BODY: …oe_fused_gate (select_experts) (#8770)" ⏎  ⏎ This reverts commit 591c232f7c9ef959c8620567789fe9d918a58bb4. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-89f1d4f536  (L1, 2025-08-11, sha 89f1d4f5367e, PR #9066)
TITLE: update deepep commit to support qwen3-coder (#9066)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L1.upstream.deepep; DeepEP install commit was updated to support Qwen3-Coder MoE dispatch.
ARTIFACT_HINTS: -
FILES: scripts/ci/ci_install_deepep.sh (+1/-1); docker/Dockerfile (+1/-1)
LABELS: high priority
PERF_LINES: Latency: 37.781 s | Output throughput: 709.087 token/s | |    |   max_concurrency |   input_throughput |   output_throughput |   mean_ttft_ms |   median_ttft_ms |   p99_ttft_ms |   mean_tpot_ms |   median_tpot_ms |    | |    |   max_concurrency |   input_throughput |   output_throughput |   mean_ttft_ms |   median_ttft_ms |   p99_ttft_ms |   mean_tpot_ms |   median_tpot_ms |   
BODY: ## Motivation ⏎  ⏎  ⏎ update deepep version to support `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` , refer https://github.com/deepseek-ai/DeepEP/pull/329, related issue: https://github.com/sgl-project/sglang/issues/9038 ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 --host 0.0.0.0 --port 40000 --trust-remote-code  --tp-size 8 --moe-a2a-backend deepep --mem-fraction-static 0.8 ⏎ python3 benchmark/gsm8k/bench_sglang.py --port 40000  ⏎ Accuracy: 0.955 ⏎ Invalid: 0.000 ⏎ Latency: 37.781 s ⏎ Output throughput: 709.087 token/s ⏎ ``` ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path Qwen/Qwen3-235B-A22B-FP8  --host 0.0.0.0 --port 40000 --trust-remote-code  --tp-size 8 --dp 8 --moe-dense-tp-size 1 --enable-dp-lm-head --moe-a2a-backend deepep --mem-fraction-static 0.8 --enable-dp-attention --max-running-requests 1024 --deepep-mode auto --cuda-graph-max-bs 128 ⏎  ⏎ # before deepep version ⏎  ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |    |   max_concurrency |   input_throughput |   output_throughput |   mean_ttft_ms |   median_ttft_ms |   p99_ttft_ms |   mean_tpot_ms |   median_tpot_ms |   p99_tpot_ms |   per_user_throughput | ⏎ +====+===================+====================+=====================+================+==================+===============+================+==================+===============+=======================+ ⏎ |  0 |             1.000 |             63.256 |              63.256 |        256.467 |          254.349 |       263.485 |         15.563 |           15.576 |        15.589 |                63.256 | ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |  1 |             4.000 |            218.609 |             218.609 |        573.219 |          573.862 |       715.690 |         17.733 |           17.668 |        18.273 |                54.652 | ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |  2 |            16.000 |            723.531 |             723.531 |        708.242 |          707.572 |       917.711 |         21.418 |           21.516 |        22.234 |                45.221 | …[truncated]

### L1-9f24dfefd1  (L1, 2025-08-11, sha 9f24dfefd156, PR #9079)
TITLE: chore(gb200): remove ToT flashinfer installation  (#9079)
SOURCES: dependency_pin
STAGE1: repair_build_dependency; artifacts=L1.upstream.flashinfer_moe; GB200 Docker stopped installing ToT FlashInfer after packaged version carried FP4 MoE quantization fix.
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+0/-8)
BODY: SGL now uses newer flashinfer that contains fix for fp4 quantization. We can remove ToT installation

### L1-90f44b74e6  (L1, 2025-08-11, sha 90f44b74e6c2, PR #8752)
TITLE: fix: w4afp8 accuracy problem and rebase (#8752)
SOURCES: path_core, symbol_pickaxe
STAGE1: repair_correctness; artifacts=L1.cutlass.adapters,L1.ep.layer; W4AFP8 accuracy fix changed Cutlass adapter and EP MoE kernel handling.
ARTIFACT_HINTS: L1.ep.layer, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+4/-5); python/sglang/srt/layers/moe/ep_moe/kernels.py (+43/-0); python/sglang/srt/layers/quantization/w4afp8.py (+20/-11)
BODY: co-author: @ayrnb ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5190ba7f42  (L1, 2025-08-12, sha 5190ba7f4216, PR #9005)
TITLE: Fuse two kernels of hidden states padding into quantization kernel (#9005)
SOURCES: path_core
STAGE1: optimize; artifacts=L1.triton.fused_moe; Fused-MoE layer stopped separate hidden-state padding because quantization kernel absorbed it.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-8); python/sglang/srt/layers/quantization/mxfp4.py (+4/-1)
BODY: ## Motivation ⏎  ⏎ Flashinfer side is ready (https://github.com/flashinfer-ai/flashinfer/pull/1445). ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-13c48dcf88  (L1, 2025-08-12, sha 13c48dcf8800, PR #9088)
TITLE: [1/2][resubmit again] sgl-kernel: Fuse routed scaling factor into moe_fused_gate  (#9088)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: optimize; artifacts=L1.routing.fused_gate; moe_fused_gate CUDA kernel fused routed-scaling multiply into select_experts.
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/csrc/moe/moe_fused_gate.cu (+20/-7); sgl-kernel/include/sgl_kernel_ops.h (+2/-1); sgl-kernel/python/sgl_kernel/moe.py (+9/-2)
BODY: ## Motivation ⏎  ⏎ 2nd resubmit of https://github.com/sgl-project/sglang/pull/8364 - see this for perf ⏎  ⏎ This PR contains sgl-kernel changes to fused routed scaling multiply into select_experts. https://github.com/sgl-project/sglang/pull/8690 will enable using this fusion for deepseek. ⏎  ⏎ Removed unit test for now because it would fail until sgl-kernel is updated. Will reenable in #8690

### L1-b3363cc1aa  (L1, 2025-08-13, sha b3363cc1aaf1, PR #9171)
TITLE: Fix docker container DeepEP error on Blackwell (#9171)
SOURCES: subject_keyword, release_notes
STAGE1: repair_build_dependency; artifacts=L1.upstream.deepep; Docker change fixes DeepEP use on Blackwell containers.
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1)
BODY: ## Motivation ⏎  ⏎ test: not run the full image, but only run that single command on an existing container and see the error disappears ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1bc183c6de  (L1, 2025-08-13, sha 1bc183c6de95, PR #9162)
TITLE: Faster weight processing (trtllm-gen moe nvfp4) (#9162)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: optimize; artifacts=L1.runner.flashinfer_trtllm; FlashInfer TRTLLM-gen MoE weight processing was cached to reduce startup time.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/modelopt_quant.py (+52/-30)
LABELS: high priority
PERF_LINES: Latency: 176.088 s | Output throughput: 515.702 token/s | **✅ Weights processing is improved 12x**
BODY: ## Motivation ⏎  ⏎ Reduce server start-up time in weights processing for trtllm-gen MoE ⏎  ⏎ ## Modifications ⏎  ⏎ Speeding up the weights processing by caching. The utility function is integrated to FI https://github.com/flashinfer-ai/flashinfer/pull/1412 . Now utilizing this inside SGL. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ $ python3 benchmark/gsm8k/bench_sglang.py \ ⏎   --num-questions 900 \ ⏎   --parallel 32 \ ⏎   --num-shots 8 ⏎  ⏎ Accuracy: 0.961 ⏎ Invalid: 0.000 ⏎ Latency: 176.088 s ⏎ Output throughput: 515.702 token/s ⏎ ``` ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎ **✅ Weights processing is improved 12x** ⏎  ⏎ ``` ⏎ (Before) ⏎ [2025-08-13 14:27:06 TP0] Load weight begin. avail mem=175.34 GB ⏎ [2025-08-13 14:27:52 TP2] Applied flashinfer weight processing for both w13 and w2 ⏎ [2025-08-13 14:27:52 TP3] Applied flashinfer weight processing for both w13 and w2 ⏎ [2025-08-13 14:27:52 TP0] Applied flashinfer weight processing for both w13 and w2 ⏎ [2025-08-13 14:27:52 TP1] Applied flashinfer weight processing for both w13 and w2 ⏎ [2025-08-13 14:34:24 TP0] Load weight end. type=DeepseekV3ForCausalLM, dtype=torch.bfloat16, avail mem=80.84 GB, mem usage=94.50 GB. ⏎  ⏎ (After) ⏎ [2025-08-13 14:47:38 TP0] Load weight begin. avail mem=175.34 GB ⏎ [2025-08-13 14:48:10 TP3] Applied flashinfer weight processing for both w13 and w2 ⏎ [2025-08-13 14:48:12 TP1] Applied flashinfer weight processing for both w13 and w2 ⏎ [2025-08-13 14:48:13 TP0] Applied flashinfer weight processing for both w13 and w2 ⏎ [2025-08-13 14:48:14 TP2] Applied flashinfer weight processing for both w13 and w2 ⏎ [2025-08-13 14:48:15 TP0] Load weight end. type=DeepseekV3ForCausalLM, dtype=torch.bfloat16, avail mem=80.83 GB, mem usage=94.51 GB. ⏎ ``` ⏎  ⏎ Forward pass is unaffected (but see further perf testing results in comments) ⏎  ⏎ thanks to @azhurkevich for development/testing instructions ⏎  ⏎ followed the same benchmark https://github.com/sgl-project/sglang/pull/8552#issuecomment-3148720680 ⏎  ⏎ ## Checklist

### L1-5aa1ebd242  (L1, 2025-08-14, sha 5aa1ebd24289, PR #8112)
TITLE: [2/n]decouple quantization implementation from vLLM dependency (#8112)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L1.runner.marlin; Marlin MoE kernel code was adjusted while decoupling quantization from vLLM dependency.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.runner.marlin
FILES: sgl-kernel/csrc/moe/marlin_moe_wna16/core/registration.h (+0/-25); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel.h (+2/-2); sgl-kernel/csrc/moe/marlin_moe_wna16/marlin_template.h (+2/-3); sgl-kernel/csrc/moe/marlin_moe_wna16/ops.cu (+1/-3); sgl-kernel/python/sgl_kernel/fused_moe.py (+2/-1); python/sglang/srt/layers/quantization/utils.py (+24/-0); sgl-kernel/CMakeLists.txt (+4/-2); sgl-kernel/csrc/common_extension.cc (+22/-6); sgl-kernel/csrc/gemm/gptq/compat.cuh (+62/-0); sgl-kernel/csrc/gemm/gptq/gptq_kernel.cu (+1950/-0); sgl-kernel/csrc/gemm/gptq/matrix_view.cuh (+269/-0); sgl-kernel/csrc/gemm/gptq/qdq_2.cuh (+74/-0); sgl-kernel/csrc/gemm/gptq/qdq_3.cuh (+146/-0); sgl-kernel/csrc/gemm/gptq/qdq_4.cuh (+114/-0); sgl-kernel/csrc/gemm/gptq/qdq_8.cuh (+30/-0); sgl-kernel/csrc/gemm/gptq/qdq_util.cuh (+53/-0); sgl-kernel/csrc/gemm/marlin/awq_marlin_repack.cu (+31/-33); sgl-kernel/csrc/gemm/marlin/dequant.h (+459/-0); sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu (+1120/-0); sgl-kernel/csrc/gemm/marlin/gptq_marlin_repack.cu (+43/-47); sgl-kernel/csrc/gemm/marlin/kernel.h (+36/-0); sgl-kernel/csrc/gemm/marlin/marlin.cuh (+2/-2); sgl-kernel/csrc/gemm/marlin/marlin_dtypes.cuh (+1/-2); sgl-kernel/csrc/gemm/marlin/marlin_template.h (+1629/-0); sgl-kernel/include/scalar_type.hpp (+2/-0); sgl-kernel/include/sgl_kernel_ops.h (+34/-9); sgl-kernel/python/sgl_kernel/__init__.py (+3/-0); sgl-kernel/python/sgl_kernel/gemm.py (+61/-1); sgl-kernel/python/sgl_kernel/marlin.py (+2/-2); sgl-kernel/tests/test_gptq_kernel.py (+131/-0); sgl-kernel/tests/test_marlin_gemm.py (+121/-0); sgl-kernel/tests/test_marlin_repack.py (+78/-66)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ The primary goal of this change is to enhance the consistency and stability of SGLang's quantization features. By decoupling the quantization implementation from its vLLM dependency, we aim to make the module easier to maintain and more portable. ⏎ Full realization of this goal will involve several subsequent PRs; this particular PR addresses the marlin kernel issues. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-392de007cb  (L1, 2025-08-14, sha 392de007cb6f, PR #9205)
TITLE: Minor fix docker container DeepEP on multi platforms (#9205)
SOURCES: subject_keyword, release_notes
STAGE1: repair_build_dependency; artifacts=L1.upstream.deepep; Docker container dependency fix was specifically for DeepEP multi-platform use.
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+12/-1)
BODY: ## Motivation ⏎  ⏎ temp modify the ci yaml to let it run once ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-295895120d  (L1, 2025-08-14, sha 295895120df4, PR #8849)
TITLE: [6/N] MoE Refactor: Cleanup MoE-related configs (#8849)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L1.runner.framework,L1.routing.topk_py,L1.ep.deepep_dispatcher; MoE refactor changed runner/backend configuration objects and dispatch integration flags.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py, L1.runner.framework, L1.runner.openai_triton_kernels, L1.runner.marlin, L1.ep.layer, L1.ep.deepep_dispatcher, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/__init__.py (+29/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+31/-30); python/sglang/srt/layers/moe/fused_moe_native.py (+14/-25); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+51/-69); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+46/-112); python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+20/-18); python/sglang/srt/layers/moe/moe_runner/__init__.py (+3/-0); python/sglang/srt/layers/moe/moe_runner/base.py (+13/-0); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+6/-0); python/sglang/srt/layers/moe/token_dispatcher/base_dispatcher.py (+55/-14); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+11/-21); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+1/-1); python/sglang/srt/layers/moe/topk.py (+128/-75); python/sglang/srt/layers/moe/utils.py (+136/-18); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+13/-5); docs/advanced_features/server_arguments.md (+3/-5); python/sglang/bench_one_batch.py (+0/-6); python/sglang/srt/eplb/expert_distribution.py (+2/-3); python/sglang/srt/layers/communicator.py (+3/-2); python/sglang/srt/layers/quantization/awq.py (+7/-7); python/sglang/srt/layers/quantization/base_config.py (+2/-6); python/sglang/srt/layers/quantization/blockwise_int8.py (+4/-12); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+7/-14); python/sglang/srt/layers/quantization/fp4.py (+13/-30); python/sglang/srt/layers/quantization/fp8.py (+24/-24); python/sglang/srt/layers/quantization/fp8_utils.py (+1/-0); python/sglang/srt/layers/quantization/gptq.py (+5/-4); python/sglang/srt/layers/quantization/marlin_utils.py (+4/-3); python/sglang/srt/layers/quantization/modelopt_quant.py (+23/-34); python/sglang/srt/layers/quantization/moe_wna16.py (+10/-15); python/sglang/srt/layers/quantization/mxfp4.py (+9/-25); python/sglang/srt/layers/quantization/unquant.py (+27/-69); python/sglang/srt/layers/quantization/w4afp8.py (+7/-8); python/sglang/srt/layers/quantization/w8a8_fp8.py (+5/-13); python/sglang/srt/layers/quantization/w8a8_int8.py (+5/-13); python/sglang/srt/managers/schedule_batch.py (+1/-9); python/sglang/srt/managers/scheduler.py (+11/-14); python/sglang/srt/model_executor/model_runner.py (+0/-3); python/sglang/srt/models/dbrx.py (+12/-6); python/sglang/srt/models/deepseek.py (+2/-1); (+29 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ - Adding `--moe-runner-backend` and deprecating `--enable-triton-kernel-moe`, `--enable-flashinfer-cutlass-moe`, and `--enable-flashinfer-trtllm-moe`. ⏎ - Adding `TopKOutputChecker` and `DispatchOutputChecker` to make pylint happy. ⏎ - Adding some util functions to avoid calling `global_server_args` in moe-related logics. ⏎ - Adding `MoeRunnerConfig` to wrap up moe runner configs. ⏎ - Some minor cleanup ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4fc09e0df0  (L1, 2025-08-15, sha 4fc09e0df0f0, PR #8777)
TITLE: Fp4 MOE quant kernel optimization (#8777)
SOURCES: subject_keyword, release_notes
STAGE1: optimize; artifacts=NEW:nvfp4_expert_quant; FP4 expert quantization CUDA kernel was optimized for MoE weight processing.
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/gemm/nvfp4_expert_quant.cu (+222/-41)
LABELS: high priority
PERF_LINES: | Version | Avg Time(sec) | Speedup| | | Optimized | 59.17 | ~+13%|
BODY: ## Motivation ⏎ Port vLLM's [FP4 MoE kernel optimization (PR #19500)](https://github.com/vllm-project/vllm/pull/19500) to SGLang, improving performance of expert-based FP4 quantization on NVIDIA Blackwell GPUs. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ This PR introduces several optimizations for FP4 expert quantization: ⏎ - Switched to a grid-stride loop layout to replace per-block row processing, enabling better thread-level parallelism. ⏎ - Added launch configuration tuning: if grid size is smaller than the number of SMs and block size is large, we double the grid size and halve the block size to improve occupancy. ⏎ - For small problem sizes where blocks are not reused, expert offsets are read into registers for low-overhead lookup. ⏎ - For large problem sizes where blocks are reused, expert offsets are first loaded into shared memory and then accessed via binary search for efficiency. ⏎  ⏎  ⏎  ⏎  ## Accuracy Test  ⏎ We verified correctness through lm_eval, dataset gsm8k : ⏎ python -m lm_eval   --model sglang   --model_args pretrained=/models/DeepSeek-R1-FP4,tp_size=4,ep_size=4   --tasks gsm8k   --num_fewshot 5   --device cuda   --batch_size auto   --output_path ./results.json ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9568|±  |0.0056| ⏎  |     |       |strict-match    |     5|exact_match|↑  |0.9560|±  |0.0056| ⏎  ⏎ Verification environment ⏎ - Hardware: NVIDIA B200 (Blackwell GPU) ⏎ - SGLang version: v0.4.9.post6 ⏎ - CUDA version: 12.8 ⏎ - lm_eval version:  v0.4.9.1 ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎ We perform benchmark and profiling through ⏎ pytest -s -v test_fp4_moe.py ⏎  ⏎ | Version | Avg Time(sec) | Speedup| ⏎ |---|---:|---| ⏎ | Baseline | 68.22 | - | ⏎ | Optimized | 59.17 | ~+13%| ⏎  ⏎ Benchmark environment ⏎ - Hardware: NVIDIA B200 (Blackwell GPU) ⏎ - SGLang version: v0.4.9.post6 ⏎ - CUDA version: 12.8 ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-eff4eb3fdd  (L1, 2025-08-15, sha eff4eb3fdd81, PR #7667)
TITLE: Add fp4 quantize before all-gather for Flashinfer cutlass MoE DP (max throughput) (#7667)
SOURCES: path_core, symbol_pickaxe
STAGE1: optimize; artifacts=L1.runner.flashinfer_cutlass,L1.routing.topk_py; FlashInfer Cutlass MoE DP path added FP4 quantization before all-gather.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/__init__.py (+2/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-3); python/sglang/srt/layers/moe/topk.py (+7/-0); python/sglang/srt/layers/moe/utils.py (+23/-0); python/sglang/srt/distributed/device_communicators/pynccl.py (+68/-18); python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py (+52/-0); python/sglang/srt/distributed/parallel_state.py (+81/-0); python/sglang/srt/layers/communicator.py (+9/-2); python/sglang/srt/layers/dp_attention.py (+22/-3); python/sglang/srt/layers/logits_processor.py (+5/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+39/-8); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/model_executor/forward_batch_info.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+36/-15); python/sglang/srt/operations.py (+6/-1); python/sglang/srt/server_args.py (+6/-0)
PERF_LINES: Latency: 23.484 s | Output throughput: 6228.185 token/s | End to end speedup: 9.38% | Request throughput (req/s):              13.56 | Input token throughput (tok/s):          13881.54 | Output token throughput (tok/s):         13881.54 | Total token throughput (tok/s):          27763.09 | ----------------End-to-End Latency---------------- | Mean E2E Latency (ms):                   75318.71 | Medi
BODY: ## Motivation ⏎  ⏎ The goal of this PR is to optimize communications for DP with FlashInfer Cutlass MoE. ⏎  ⏎ ## Modifications ⏎  ⏎ Improvements include: ⏎ 1. Add Allgatherv collective. This is a pynccl implementation of TRT-LLM's allgather which supports varying sizes per rank and a list of tensors as inputs ⏎ 2. Add reducescatterv collective. This is a pynccl implementation of TRT-LLM's reducescatter which supports varying sizes per rank ⏎ 3. For Flashinfer MoE with DP, use allgatherv to dispatch tokens. We also move the fp4 quantize before the allgather so the communication is smaller. Finally, we use reducescatterv to combine the results instead of all_reduce. ⏎  ⏎ ## Usage ⏎  ⏎ Enabled automatically when applicable: `--enable-flashinfer-cutlass-moe`, `--enable-dp-attention`, and `dp_size == ep_size` must all be true. ⏎ Can be disabled with `--disable-flashinfer-cutlass-moe-fp4-allgather`. ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path nvidia/DeepSeek-R1-0528-FP4 --trust-remote-code --quantization modelopt_fp4 --tp 8 --enable-flashinfer-cutlass-moe --ep-size 8 --dp 8 --enable-dp-attention ⏎ ``` ⏎  ⏎ ## Results ⏎  ⏎ ### Accuracy ⏎  ⏎ ``` ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319 --port=30000 ⏎ Accuracy: 0.958 ⏎ Invalid: 0.000 ⏎ Latency: 23.484 s ⏎ Output throughput: 6228.185 token/s ⏎ ``` ⏎  ⏎ ### Benchmark ⏎  ⏎ End to end speedup: 9.38% ⏎  ⏎ ``` ⏎ python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 1024 --random-input 1024 --random-output 1024 --random-range-ratio 1 --max-concurrency 1024 ⏎ ``` ⏎  ⏎ BEFORE ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Backend:                                 sglang ⏎ Traffic request rate:                    inf ⏎ Max request concurrency:                 1024 ⏎ Successful requests:                     1024 ⏎ Benchmark duration (s):                  75.54 ⏎ Total input tokens:                      1048576 ⏎ Total generated tokens:                  1048576 ⏎ Total generated tokens (retokenized):    1045749 ⏎ Request throughput (req/s):              13.56 ⏎ Input token throughput (tok/s):          13881.54 ⏎ Output token throughput (tok/s):         13881.54 ⏎ Total token throughput (tok/s):          27763.09 ⏎ Concurrency:                             1021.04 ⏎ ----------------End-to-End Latency---------------- ⏎ Mean E2E Latency (ms):                   75318.71 ⏎ Median E2E Latency (ms):                 75293.46 ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):                          11651.64 ⏎ Median TTFT (ms):                        11573.21 ⏎ P99 TTFT …[truncated]

### L1-1c1f8a118e  (L1, 2025-08-16, sha 1c1f8a118ef8, PR #9049)
TITLE: Combine fp4.py and mxfp4.py into one file and support dynamic mxfp4 quantization in mxfp4.py (#9049)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L1.triton.fused_moe; FP4/MXFP4 quantization consolidation changed fused-MoE layer integration.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-1); python/sglang/srt/layers/quantization/__init__.py (+8/-9); python/sglang/srt/layers/quantization/fp4.py (+0/-540); python/sglang/srt/layers/quantization/mxfp4.py (+156/-5); python/sglang/srt/layers/quantization/quark/quark.py (+390/-0); python/sglang/srt/layers/quantization/quark/quark_moe.py (+197/-0); python/sglang/srt/server_args.py (+3/-2)
LABELS: high priority
PERF_LINES: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ | Latency: 83.977 s` | 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ | Latency: 110.876 s | Output throughput: 3927.407 toke
BODY: ## Motivation ⏎  ⏎ There are many duplicated features in the fp4.py and mxfp4.py. ⏎  ⏎ In order to simply them, combine these two files into one.  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ **[Dynamic Quant Grok-1]** ⏎ `~/sglang# python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 ⏎ 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [01:23<00:00, 15.76it/s] ⏎ Accuracy: 0.822 ⏎ Invalid: 0.001 ⏎ Latency: 83.977 s` ⏎  ⏎ **[Static Quant GPT-OSS-120b]** ⏎ `~/sglang# python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 ⏎ 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [01:50<00:00, 11.91it/s] ⏎ Accuracy: 0.832 ⏎ Invalid: 0.010 ⏎ Latency: 110.876 s ⏎ Output throughput: 3927.407 token/s` ⏎  ⏎ ## Checklist

### L1-0fc54b971e  (L1, 2025-08-17, sha 0fc54b971e14, PR #9272)
TITLE: [fix]:  fix cutlass moe ut and and Opt H20 cutlass groupGemm performance (#9272)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: optimize; artifacts=L1.cutlass.fp8_blockwise; FP8 blockwise CUTLASS MoE kernel added optimized H20 grouped-GEMM instance.
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+109/-35); python/sglang/test/test_cutlass_moe.py (+4/-6); sgl-kernel/include/utils.h (+19/-0)
PERF_LINES: | Batch Size | origin_h20 Cutlass fused_experts Time (ms) | opt_h20 Cutlass fused_experts Time (ms) |
BODY: ## Motivation ⏎ 1.  fix cutlass moe UT ⏎ 2. add optimized cutlass groupGemm instance for H20(SGL_TUNE_DEVICE_KERNEL=1  to open opt) ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎ Deepseek-tp8 on H20 ⏎ | Batch Size | origin_h20 Cutlass fused_experts Time (ms) | opt_h20 Cutlass fused_experts Time (ms) | ⏎ |------------|-------------------------------------------|-------------------------------------------| ⏎ | 1 | 0.110 | 0.099 | ⏎ | 4 | 0.279 | 0.171 | ⏎ | 8 | 0.457 | 0.279 | ⏎ | 16 | 0.709 | 0.408 | ⏎ | 32 | 1.078 | 0.601 | ⏎ | 64 | 1.434 | 0.779 | ⏎ | 128 | 1.619 | 0.857 | ⏎ | 256 | 1.684 | 0.920 | ⏎ | 512 | 1.720 | 0.951 | ⏎ | 1024 | 1.798 | 1.022 | ⏎  ⏎ ## Checklist

### L1-c6c379ab31  (L1, 2025-08-18, sha c6c379ab3161, PR #9320)
TITLE: [AMD] Reorganize hip-related header files in sgl-kernel (#9320)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L1.align.cuda_aot; AMD header reorganization changed moe_align kernel include/warp definitions for ROCm builds.
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/csrc/moe/moe_align_kernel.cu (+0/-2); .github/workflows/pr-test-amd.yml (+1/-0); sgl-kernel/csrc/allreduce/mscclpp_allreduce.cuh (+1/-1); sgl-kernel/csrc/elementwise/activation.cu (+1/-1); sgl-kernel/csrc/gemm/per_tensor_quant_fp8.cu (+2/-2); sgl-kernel/csrc/gemm/per_token_quant_fp8.cu (+2/-2); sgl-kernel/include/hip/hip_act_and_mul.cuh (+0/-0); sgl-kernel/include/hip/hip_math_def.h (+1/-1); sgl-kernel/include/hip/hip_vec_dtypes.h (+0/-0); sgl-kernel/include/hip/impl/hip_vec_bf16_impl.h (+0/-0); sgl-kernel/include/hip/impl/hip_vec_fp32_impl.h (+0/-0); sgl-kernel/include/hip/impl/hip_vec_half_impl.h (+0/-0); sgl-kernel/include/utils.h (+6/-7); sgl-kernel/setup_rocm.py (+4/-1)
BODY: ## Motivation ⏎  ⏎ Reorganize the header files for AMD GPUs in `sgl-kernel` ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ - Reorganized the header files in `sgl-kernel` for AMD GPUs ⏎ - Changed `__HIP_PLATFORM_AMD__` to `USE_ROCM` ⏎ - Changed `WARP_SIZE=32` to `WARP_SIZE=C10_WARP_SIZE` defined in `include/utils.h` on AMD GPUs. I have tested it locally for the correctness and also confirmed that there is no performance regression using `benchmark/bench_moe_align_block_size.py` ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ CC: @HaiShaw @merrymercy

### L1-3c2c9f6c9e  (L1, 2025-08-18, sha 3c2c9f6c9e7b, PR #9317)
TITLE: [Bug] Fix input arguments of flashinfer_trtllm_moe (#9317)
SOURCES: path_core, symbol_pickaxe
STAGE1: repair_correctness; artifacts=L1.runner.flashinfer_trtllm,L1.routing.topk_py; FlashInfer TRTLLM MoE arguments and TopK attributes were fixed after MoE refactor.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-2); python/sglang/srt/layers/moe/topk.py (+14/-14); python/sglang/srt/layers/quantization/fp8.py (+14/-5)
LABELS: bug, high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ flashinfer_trtllm fp8 moe kernel has several bugs in attributes, which are changed in moe refactoring. and add assertion to avoid using the fp8 kernel if n_group and topk_group are None ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-01d47a27b6  (L1, 2025-08-19, sha 01d47a27b6f6, PR #9327)
TITLE: [Bugfix] fix kv buffer register & dp attention & deepepmoe (#9327)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.ep.layer; EP MoE layer one-line fix repaired behavior after MoE-layer refactor.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-1); python/sglang/srt/disaggregation/ascend/conn.py (+1/-3); python/sglang/srt/layers/dp_attention.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Fix kv register when kv alloc is separated on Ascend NPU ⏎ - Fix an issue in dp attention ⏎ - Fix an issue after refectoring of moe layer ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a91e90d9a3  (L1, 2025-08-20, sha a91e90d9a360, PR #8690)
TITLE: [2/2] Fuse routed scaling factor into select_experts (#8690)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: optimize; artifacts=L1.routing.topk_py,L1.routing.fused_gate; select_experts consumes fused routed-scaling output from moe_fused_gate.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+7/-0); python/sglang/srt/layers/moe/topk.py (+20/-0); python/sglang/srt/layers/quantization/fp8.py (+8/-9); python/sglang/srt/layers/quantization/modelopt_quant.py (+2/-4); python/sglang/srt/models/deepseek_v2.py (+12/-11); sgl-kernel/tests/test_moe_fused_gate.py (+6/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ See https://github.com/sgl-project/sglang/pull/8364 ⏎  ⏎ Requires https://github.com/sgl-project/sglang/pull/8770 [1/2][resubmit] sgl-kernel: Fuse routed scaling factor into moe_fused_gate (select_experts) ⏎  ⏎ ## Modifications ⏎  ⏎ See https://github.com/sgl-project/sglang/pull/8364 ⏎  ⏎ ## Accuracy Test ⏎  ⏎ See https://github.com/sgl-project/sglang/pull/8364 ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎ See https://github.com/sgl-project/sglang/pull/8364 ⏎  ⏎ ## Checklist

### L1-c674bf9c6b  (L1, 2025-08-20, sha c674bf9c6b0a, PR #9420)
TITLE: Fix biased_grouped_topk_cpu (#9420)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L1.routing.topk_py,L1.hardware.cpu_npu_musa; CPU biased_grouped_topk path gained a correctness fix in topk.py.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-0)
LABELS: intel
BODY: Fix `biased_grouped_topk_cpu` by adding the argument `apply_routed_scaling_factor_on_output`.

### L1-7cd2ee06d7  (L1, 2025-08-20, sha 7cd2ee06d741, PR #9251)
TITLE: feat: Add Triton fallback option and SM120 MoE configs for FP8 models (#9251)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: change_default; artifacts=L1.triton.fused_moe; FP8 utilities gained a Triton fallback option for MoE backend selection.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=352,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=384,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/quantization/fp8_utils.py (+20/-8)
LABELS: high priority
BODY: ## Summary ⏎ This PR adds support for running FP8 quantized models on SM120 (Blackwell) GPUs by: ⏎ 1. Adding `USE_TRITON_W8A8_FP8_KERNEL` environment variable to force Triton fallback ⏎ 2. Including optimized MoE configs for RTX 6000 Blackwell GPUs ⏎  ⏎ ## Problem ⏎ Currently, FP8 quantized models fail on SM120/Blackwell GPUs (RTX 5090/6000) with two issues: ⏎ 1. **CUTLASS kernel incompatibility**: The fp8_scaled_mm kernel doesn't support SM120, causing crashing in fp8_scaled_mm ⏎ 2. **Shared memory overflow**: Triton MoE kernels exceed SM120's 101KB shared memory limit ⏎  ⏎ ## Solution ⏎  ⏎ ### 1. Triton Fallback Option ⏎ Added `USE_TRITON_W8A8_FP8_KERNEL` environment variable that forces the use of Triton kernel instead of CUTLASS fp8_scaled_mm. This provides a universal fallback that works on all GPU architectures. ⏎  ⏎ Usage: ⏎ ```bash ⏎ USE_TRITON_W8A8_FP8_KERNEL=1 python -m sglang.launch_server --model-path <fp8-model> ... ⏎ ``` ⏎  ⏎ ### 2. SM120 MoE Configurations   ⏎ Added optimized Triton configs for MoE models with reduced memory requirements: ⏎ - E=129 (for GLM-4.5-Air-FP8) ⏎ - E=161 (for GLM-4.5-FP8) ⏎  ⏎ ## Testing ⏎ Tested successfully with: ⏎ - GLM-4.5-Air-FP8 on RTX 6000 Blackwell Q-max ⏎ - GLM-4.5-FP8 on RTX 6000 Blackwell Q-max ⏎  ⏎ Both models now run without errors when using the Triton fallback flag. ⏎  ⏎ ## Impact ⏎ - **No breaking changes**: Existing functionality remains unchanged when flag is not set ⏎ - **Enables new hardware**: Allows FP8 GLM models to run on RTX 5090 and RTX 6000  ⏎  ⏎ ## Notes ⏎ - Config file names must exactly match GPU name from `torch.cuda.get_device_name()` ⏎ - Users with other RTX 5090 and 6000 variants may need to create matching config files

### L1-18da2c96ec  (L1, 2025-08-21, sha 18da2c96ec09, PR #9384)
TITLE: [NVIDIA] Fix trtllm fp4 moe backend when used in MTP (#9384)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:kernel-correctness-cases
STAGE1: repair_correctness; artifacts=L1.runner.flashinfer_trtllm,L1.routing.topk_py; TRTLLM FP4 MoE backend metadata/top-k handling was fixed for MTP.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+5/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-0); python/sglang/srt/layers/moe/topk.py (+3/-1); python/sglang/srt/models/deepseek_v2.py (+2/-1)
LABELS: high priority
DEEP_STUDY: deep-study correctness case sglang:18da2c96ec: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: This PR addresses the issue mentioned [here](https://github.com/sgl-project/sglang/pull/9238#issuecomment-3199279509). ⏎  ⏎ The root cause is that some MoE layers may use an unquantized method. For example: ⏎ ``` ⏎ model.layers.3.mlp.experts → quantized   ⏎ model.decoder.mlp.experts  → unquantized ⏎ ``` ⏎  ⏎ This may be triggered by the MTP settings. ⏎  ⏎ `FusedMoE` works with both quantized and unquantized methods, which explains why the `flashinfer_cutlass` fp4 backend works fine. ⏎  ⏎ `FlashInferFP4MoE` (i.e. the `flashinfer_trtllm` backend), however, only supports the modelopt_fp4 quantized method and fails with unquantized ones. ⏎  ⏎ This PR fixes the issue by checking the quantization config. If no quantization config is found, we fall back to `FusedMoE`. ⏎  ⏎ cc. @pranavm-nvidia @zhyncs @kushanam

### L1-de4990a5b2  (L1, 2025-08-21, sha de4990a5b2d1, PR #9392)
TITLE: [Bug] Fix w4afp8 moe kernel (#9392)
SOURCES: subject_keyword, release_notes, corpus:kernel-correctness-cases
STAGE1: repair_correctness; artifacts=L1.cutlass.w4a8; W4AFP8 MoE CUTLASS kernel support header was fixed.
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp (+4/-0)
DEEP_STUDY: deep-study correctness case sglang:de4990a5b2: class=nondeterminism_race_sync; symptom=nondeterministic_output; introducing=#7772
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR fixes an issue introduced in [PR7772](https://github.com/sgl-project/sglang/pull/7772). ⏎ When running `test_int4_fp8_grouped_gemm_multi_experts` in `sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py` with `k = 512, n = 1024`, the test may occasionally produce incorrect results. (It happens very rarely, but you can reproduce it more easily by setting `batch_size = 512` and `num_experts = 256`.) ⏎  ⏎ This bug also affects the E2E results of DeepSeek-R1 in w4afp8 TP mode ([PR8118](https://github.com/sgl-project/sglang/pull/8118)), where you may observe inconsistent outputs even with the same random seed and prompt. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ This fix is adapted from [NVIDIA CUTLASS v4.0.0](https://github.com/NVIDIA/cutlass/pull/2294/files#diff-33619fb73da11116cba19dfe6c4ca78e6510ea4b28c915e461d496152af72bbf), and has also been merged into [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/pull/7072). ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5fd311d33e  (L1, 2025-08-21, sha 5fd311d33e62, PR #9333)
TITLE: [code  clean] add H20 cutlass groupGemm default  config (#9333)
SOURCES: path_core
STAGE1: retune; artifacts=L1.cutlass.fp8_blockwise; CUTLASS FP8 blockwise MoE kernel default H20 grouped-GEMM config was changed.
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+15/-35)
PERF_LINES: cutlass: 968.0480003356934 us | cutlass: 1867.8911209106445 us | cutlass: 1040.2175903320312 us | cutlass: 1863.4271621704102 us | cutlass: 960.6847763061523 us | cutlass: 515.7408237457275 us | cutlass: 1866.2431716918945 us | cutlass: 1865.2576446533203 us | cutlass: 959.654426574707 us | cutlass: 514.9983882904053 us | cutlass: 3160.8959197998047 us | cutlass: 1621.9488143920898 us | cutlass: 4
BODY: ## Motivation ⏎  ⏎ add default blockwise groupgemm config for  H20  ⏎ related to ： https://github.com/sgl-project/sglang/pull/9272 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ <img width="1725" height="221" alt="image" src="https://github.com/user-attachments/assets/13b37235-d2f5-4b5e-917a-7a892f293474" /> ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎ sgl-kernel/benchmark/bench_fp8_blockwise_group_gemm.py ⏎ **before:** ⏎ Benchmark: expected_m_per_group=128, n=512, k=7168, num_groups=256 ⏎ cutlass: 968.0480003356934 us ⏎  ⏎ Benchmark: expected_m_per_group=256, n=512, k=7168, num_groups=256 ⏎ cutlass: 1867.8911209106445 us ⏎  ⏎ Benchmark: expected_m_per_group=256, n=256, k=7168, num_groups=256 ⏎ cutlass: 1040.2175903320312 us ⏎  ⏎ Benchmark: expected_m_per_group=512, n=256, k=7168, num_groups=256 ⏎ cutlass: 1863.4271621704102 us ⏎  ⏎ Benchmark: expected_m_per_group=1, n=512, k=7168, num_groups=256 ⏎ cutlass: 960.6847763061523 us ⏎  ⏎ Benchmark: expected_m_per_group=2, n=256, k=7168, num_groups=256 ⏎ cutlass: 515.7408237457275 us ⏎  ⏎ Benchmark: expected_m_per_group=256, n=4096, k=7168, num_groups=32 ⏎ cutlass: 1866.2431716918945 us ⏎  ⏎ Benchmark: expected_m_per_group=512, n=4096, k=7168, num_groups=16 ⏎ cutlass: 1865.2576446533203 us ⏎  ⏎ Benchmark: expected_m_per_group=4, n=4096, k=7168, num_groups=32 ⏎ cutlass: 959.654426574707 us ⏎  ⏎ Benchmark: expected_m_per_group=8, n=4096, k=7168, num_groups=16 ⏎ cutlass: 514.9983882904053 us ⏎  ⏎ Benchmark: expected_m_per_group=1024, n=768, k=4096, num_groups=128 ⏎ cutlass: 3160.8959197998047 us ⏎  ⏎ Benchmark: expected_m_per_group=1024, n=4096, k=384, num_groups=128 ⏎ cutlass: 1621.9488143920898 us ⏎  ⏎ Benchmark: expected_m_per_group=16, n=768, k=4096, num_groups=128 ⏎ cutlass: 427.3087978363037 us ⏎  ⏎ Benchmark: expected_m_per_group=16, n=4096, k=384, num_groups=128 ⏎ cutlass: 263.9264106750488 us ⏎  ⏎ **after:** ⏎  ⏎ Benchmark: expected_m_per_group=128, n=512, k=7168, num_groups=256 ⏎ cutlass: 971.9296455383301 us ⏎  ⏎ Benchmark: expected_m_per_group=256, n=512, k=7168, num_groups=256 ⏎ cutlass: 1868.0927276611328 us ⏎  ⏎ Benchmark: expected_m_per_group=256, n=256, k=7168, num_groups=256 ⏎ cutlass: 965.8687591552734 us ⏎  ⏎ Benchmark: expected_m_per_group=512, n=256, k=7168, num_groups=256 ⏎ cutlass: 1866.1407470703125 us ⏎  ⏎ Benchmark: expected_m_per_group=1, n=512, k=7168, num_groups=256 ⏎ cutlass: 516.9856071472168 us ⏎  ⏎ Benchmark: expected_m_per_group=2, n=256, k=7168, num_groups=256 ⏎ cutlass: 281.1072111129761 us ⏎  ⏎ Benchmark: expected_m_per_group=256, n=4096, k=7168, num_groups=32 ⏎ cutlass: 1868.7807083129883 us ⏎  ⏎ Benchmark: expected_m_per_group=512, n=4096, k=7168, nu …[truncated]

### L1-f445a1d9a3  (L1, 2025-08-22, sha f445a1d9a3a3, PR #7699)
TITLE: [AMD] Fix Llama 4 FP8 accuracy issues on MI300X (#7699)
SOURCES: path_core, symbol_pickaxe
STAGE1: repair_correctness; artifacts=L1.ep.layer; ROCm FP8 MoE accuracy fix changed EP layer/ROCm MoE utilities.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+0/-1); python/sglang/srt/layers/moe/rocm_moe_utils.py (+141/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+66/-15); python/sglang/srt/layers/quantization/fp8.py (+1/-0); python/sglang/srt/server_args.py (+4/-1)
PERF_LINES: - To fix Llama 4 FP8 accuracy issue on MI300X (https://github.com/sgl-project/sglang/issues/5362).
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ - To fix Llama 4 FP8 accuracy issue on MI300X (https://github.com/sgl-project/sglang/issues/5362). ⏎ - This PR also includes the initial effort to streamline/modularize aiter's fused_moe implementations for various code paths. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ **Command to launch server** ⏎ ``` ⏎ SGLANG_USE_AITER=1 python -m sglang.launch_server --model-path /data/huggingface/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 --tp 8 --attention-backend aiter ⏎ ``` ⏎  ⏎ **GSM8K benchmark** ⏎ ``` ⏎ lm_eval --model local-chat-completions --model_args model=/data/huggingface/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8/,base_url=http://localhost:30000/v1/chat/completions,num_concurrent=64,max_gen_toks=2048 --tasks gsm8k --apply_chat_template --num_fewshot 0 ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     0|exact_match|↑  |0.9636|±  |0.0052| ⏎ |     |       |strict-match    |     0|exact_match|↑  |0.0000|±  |0.0000| ⏎  ⏎ ``` ⏎ **MMLU-Pro benchmark** ⏎ ``` ⏎ lm_eval --model local-chat-completions --model_args model=/data/huggingface/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8/,base_url=http://localhost:30000/v1/chat/completions,num_concurrent=64,timeout=999999,max_gen_toks=2048 --tasks mmlu_pro --batch_size 64 --apply_chat_template --num_fewshot 0 ⏎ ``` ⏎ ![image](https://github.com/user-attachments/assets/294dacda-82aa-4755-9835-773de83f4fc9) ⏎  ⏎  ⏎ **Toy prompt** ⏎ ``` ⏎ #!/bin/bash ⏎ curl -X POST "http://localhost:30000/v1/chat/completions" \ ⏎ 	-H "Content-Type: application/json" \ ⏎ 	-d '{ ⏎ 	"model": "/data/huggingface/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", ⏎ 	"messages": [ ⏎ 	{ ⏎ 		"role": "user", ⏎ 		"content": "What is Deep Learning" ⏎ 	} ⏎ 	], ⏎ 	"max_tokens": 512 ⏎ }' ⏎ ``` ⏎ ![image](https://github.com/user-attachments/assets/cebec615-d3e7-4465-bdd4-1bc8598cd2c1) ⏎  ⏎  ⏎  ⏎ CC: @HaiShaw @kkHuang-amd
