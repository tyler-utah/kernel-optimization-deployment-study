### L1-89f1d4f536  (L1, 2025-08-11, sha 89f1d4f5367e, PR #9066)
TITLE: update deepep commit to support qwen3-coder (#9066)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: scripts/ci/ci_install_deepep.sh (+1/-1); docker/Dockerfile (+1/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ update deepep version to support `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` , refer https://github.com/deepseek-ai/DeepEP/pull/329, related issue: https://github.com/sgl-project/sglang/issues/9038 ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 --host 0.0.0.0 --port 40000 --trust-remote-code  --tp-size 8 --moe-a2a-backend deepep --mem-fract …[truncated]

### L1-9f24dfefd1  (L1, 2025-08-11, sha 9f24dfefd156, PR #9079)
TITLE: chore(gb200): remove ToT flashinfer installation  (#9079)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+0/-8)
BODY: SGL now uses newer flashinfer that contains fix for fp4 quantization. We can remove ToT installation

### L1-90f44b74e6  (L1, 2025-08-11, sha 90f44b74e6c2, PR #8752)
TITLE: fix: w4afp8 accuracy problem and rebase (#8752)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.ep.layer, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+4/-5); python/sglang/srt/layers/moe/ep_moe/kernels.py (+43/-0); python/sglang/srt/layers/quantization/w4afp8.py (+20/-11)
BODY: co-author: @ayrnb ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-44e86480e8  (L1, 2025-08-11, sha 44e86480e876, PR #8731)
TITLE: fuse allreduce and residual_rmsnorm (#8731)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-1); python/sglang/srt/layers/communicator.py (+1/-1); python/sglang/srt/layers/flashinfer_comm_fusion.py (+3/-3); python/sglang/srt/layers/linear.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+48/-33); python/sglang/srt/models/glm4_moe.py (+9/-9); python/sglang/srt/models/gpt_oss.py (+66/-10); python/sglang/srt/server_args.py (+1/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ ## Acc ⏎  ⏎ ```shell ⏎ python3 -m sglang.launch_server --model-path /dev/shm/DeepSeek-R1-0528-FP4 --trust-remote-code --quantization modelopt_fp4 --tp 8 --enable-flashinfer-allreduce-fusion --attention-backend cutlass_mla ⏎  ⏎ ➜  sglang git:(cache_fuse_allreduce_residual_rmsnorm_judge) ✗ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8 ⏎ 100%|███████████████████████████████████████████████████ …[truncated]

### L1-c46c75f8c0  (L1, 2025-08-11, sha c46c75f8c0f0, PR #9087)
TITLE: feat: add fused moe config for Qwen3-30B-A3B on B200 (#9087)
SOURCES: path_config_only, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=768,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
DEEP_STUDY: deep-study: this PR was reverted by PR 10185 (confirmed_revert, reason=correctness_or_accuracy)
BODY: ## Motivation ⏎ Add fused MoE config for Qwen3-30B-A3B on NVIDIA B200 ⏎ Port from: https://github.com/vllm-project/vllm/pull/19455 ⏎  ⏎ ## Benchmark ⏎  ⏎ With config:  ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Backend:                                 sglang ⏎ Traffic request rate:                    inf ⏎ Max request concurrency:                 10 ⏎ Successful requests:                     100 ⏎ Benchmark duration (s):                  30. …[truncated]

### L1-5190ba7f42  (L1, 2025-08-12, sha 5190ba7f4216, PR #9005)
TITLE: Fuse two kernels of hidden states padding into quantization kernel (#9005)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-8); python/sglang/srt/layers/quantization/mxfp4.py (+4/-1)
BODY: ## Motivation ⏎  ⏎ Flashinfer side is ready (https://github.com/flashinfer-ai/flashinfer/pull/1445). ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3a9afe2a42  (L1, 2025-08-12, sha 3a9afe2a42eb, PR #9103)
TITLE: chore: bump sgl-kernel v0.3.4 (#9103)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1f9ec65374  (L1, 2025-08-12, sha 1f9ec65374f8, PR #9118)
TITLE: fix(docker): update sgl_kernel version to 0.3.4 in Dockerfile.gb200 (#9118)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+1/-1)
BODY: 

### L1-c9ee738515  (L1, 2025-08-12, sha c9ee73851540, PR #9014)
TITLE: Fuse writing KV buffer into rope kernel (part 2: srt) (#9014)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); .github/workflows/pr-test-pd-router.yml (+1/-1); docker/Dockerfile.gb200 (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/rotary_embedding.py (+10/-0); python/sglang/srt/models/gpt_oss.py (+51/-2)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ Fuse  set_kv_buffer to sgl-kernel rope function, only for trtllm_mha attention ⏎  ⏎ ----- ⏎  ⏎ (below is from @fzyzcjy) ⏎  ⏎ speed may be suboptimal (I have not done any ncu profile or thorough optimization), but anyway it is faster than non-fused ⏎  ⏎ ``` ⏎ tests/test_rotary_embedding.py ..................                                                                  [100%] ⏎  ⏎ =================================================== 18 p …[truncated]

### L1-305b27c124  (L1, 2025-08-12, sha 305b27c12476, PR #9125)
TITLE: fix: update Dockerfile (#9125)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2ecbd8b8bf  (L1, 2025-08-12, sha 2ecbd8b8bf98, PR #8700)
TITLE: [feat] add ascend readme and docker release (#8700)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.npu (+81/-0); .github/workflows/pr-test-npu.yml (+21/-3); .github/workflows/release-docker-npu-nightly.yaml (+76/-0); .github/workflows/release-docker-npu.yaml (+77/-0); docs/basic_usage/deepseek.md (+2/-0); docs/platforms/ascend_npu.md (+203/-4); scripts/ci/npu_ci_install_dependency.sh (+7/-11)
LABELS: ready-to-merge, npu
BODY: ## Motivation ⏎  ⏎ In the past, we only had images for GPU and AMD, but this PR would try to build and push docker image for NPU hardware ⏎  ⏎ ## Modifications ⏎  ⏎ Add two new workflow and NPU related Dockerfile, both docker images will be published to [offical registry](https://hub.docker.com/r/lmsysorg/sglang): ⏎ 1. daily dev image for user to try and nightly test case, named as `sglang:main-cann8.2.rc1.alpha003-a3 ` ⏎ 2. release image when new tag ad …[truncated]

### L1-c81daf838d  (L1, 2025-08-12, sha c81daf838da5, PR #9129)
TITLE: fix: update Dockerfile (#9129)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-924827c3de  (L1, 2025-08-12, sha 924827c3ded3, PR #9130)
TITLE: chore: use cp310 (#9130)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-13c48dcf88  (L1, 2025-08-12, sha 13c48dcf8800, PR #9088)
TITLE: [1/2][resubmit again] sgl-kernel: Fuse routed scaling factor into moe_fused_gate  (#9088)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/csrc/moe/moe_fused_gate.cu (+20/-7); sgl-kernel/include/sgl_kernel_ops.h (+2/-1); sgl-kernel/python/sgl_kernel/moe.py (+9/-2)
BODY: ## Motivation ⏎  ⏎ 2nd resubmit of https://github.com/sgl-project/sglang/pull/8364 - see this for perf ⏎  ⏎ This PR contains sgl-kernel changes to fused routed scaling multiply into select_experts. https://github.com/sgl-project/sglang/pull/8690 will enable using this fusion for deepseek. ⏎  ⏎ Removed unit test for now because it would fail until sgl-kernel is updated. Will reenable in #8690

### L1-0ff6d1fce1  (L1, 2025-08-13, sha 0ff6d1fce122, PR #9028)
TITLE: Support FA3 backend for gpt-oss (#9028)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/attention/flashattention_backend.py (+18/-0); python/sglang/srt/models/gpt_oss.py (+1/-1); python/sglang/srt/server_args.py (+4/-4)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Apply changes of https://github.com/sgl-project/sgl-attn/pull/4. ⏎  ⏎ ## Accuracy Test ⏎ `openai/gpt-oss-20b` mmlu 4k: ⏎ ``` ⏎ | model_name                                              |   ('metric', 'mmlu') | ⏎ |:--------------------------------------------------------|---------------------:| ⏎ | o4-mini-with-chat-completion-and-4k-gen_20250810_151719 |                0.835 | ⏎ ``` ⏎  ⏎ ## Benchmark & Profiling ⏎ ### `openai/gpt-oss-20b` T …[truncated]

### L1-71fb8c9527  (L1, 2025-08-13, sha 71fb8c9527cb, PR #9126)
TITLE: feat: update fa3 (#9126)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b3363cc1aa  (L1, 2025-08-13, sha b3363cc1aaf1, PR #9171)
TITLE: Fix docker container DeepEP error on Blackwell (#9171)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1)
BODY: ## Motivation ⏎  ⏎ test: not run the full image, but only run that single command on an existing container and see the error disappears ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1bc183c6de  (L1, 2025-08-13, sha 1bc183c6de95, PR #9162)
TITLE: Faster weight processing (trtllm-gen moe nvfp4) (#9162)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/modelopt_quant.py (+52/-30)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Reduce server start-up time in weights processing for trtllm-gen MoE ⏎  ⏎ ## Modifications ⏎  ⏎ Speeding up the weights processing by caching. The utility function is integrated to FI https://github.com/flashinfer-ai/flashinfer/pull/1412 . Now utilizing this inside SGL. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ $ python3 benchmark/gsm8k/bench_sglang.py \ ⏎   --num-questions 900 \ ⏎   --parallel 32 \ ⏎   --num-shots 8 ⏎  ⏎ Accuracy: 0.961 ⏎ Invalid: 0.0 …[truncated]

### L1-d6451c3f65  (L1, 2025-08-13, sha d6451c3f65ec, PR #8808)
TITLE: Add A800 fused MoE kernel tuning configs for GLM4.5 and GLM4.5-Air (#8808)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_2_0/E=129,N=352,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_2_0/E=161,N=192,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8.json (+146/-0)
BODY: ## Motivation ⏎ Add A800 fused MoE kernel tuning configs for GLM4.5 and GLM4.5-Air on channel-wise INT8. ⏎  ⏎ ## Modifications ⏎ Add fused MoE kernel tuning config file. ⏎  ⏎ ## Accuracy Test ⏎ main: ⏎ ``` ⏎ Accuracy: 0.970 ⏎ Invalid: 0.000 ⏎ Latency: 20.199 s ⏎ Output throughput: 1069.858 token/s ⏎ ``` ⏎  ⏎ pr: ⏎ ``` ⏎ Accuracy: 0.970 ⏎ Invalid: 0.000 ⏎ Latency: 19.836 s ⏎ Output throughput: 1087.133 token/s ⏎ ``` ⏎  ⏎ ## Benchmark & Profiling ⏎ Run command: ⏎ ``` ⏎ pyth …[truncated]

### L1-ac15bdc194  (L1, 2025-08-13, sha ac15bdc19491, PR #8852)
TITLE: Add H200 fused MoE kernel tuning configs for Qwen3-Coder-480B-A35B-Instruct (#8852)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=160,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎ Add H200 fused MoE kernel tuning configs for Qwen3-Coder-480B-A35B-Instruct. ⏎  ⏎ ## Modifications ⏎  ⏎ Add H200 fused MoE kernel tuning configs for Qwen3-Coder-480B-A35B-Instruct. ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎ `python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 500 --random-input 4000 --random-output 200 --random-range-ratio 0.5 --port 8090 --flush-cache --max-concurrency 16` ⏎  ⏎ Without moe confi …[truncated]

### L1-2871eacc05  (L1, 2025-08-13, sha 2871eacc0574, PR #7004)
TITLE: Add Triton Fused MoE kernel config for E=16 on B200 (#7004)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_0/E=16,N=1024,device_name=NVIDIA_B200.json (+146/-0)
LABELS: ready-to-merge, blackwell
BODY: ## Motivation ⏎  ⏎ Improve the E=16 performance on B200 (`meta-llama/Llama-4-Scout-17B-16E`). I'll follow up with other quants (fp4 first) ⏎  ⏎ The triton version is `triton==3.3.0` (this is what the Blackwell image uses). So I had to add a new folder to match this version ⏎  ⏎ @BBuf Let me know if this is ok 👍  ⏎  ⏎ Here's a table summarizing the performance improvements: ⏎  ⏎ | Metric                      | Previous Value | New Value | Improvement | ⏎ |-- …[truncated]

### L1-83feef5b2c  (L1, 2025-08-13, sha 83feef5b2c32, PR #7631)
TITLE: Add H20 fused MoE kernel configs for Dpsk & Qwen3 (#7631)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=128,N=384,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=128,N=768,device_name=NVIDIA_H20.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=257,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎ New fused MoE configs for triton 3.3 ⏎  ⏎ ## Modifications ⏎  ⏎ * `E=128,N=768,device_name=NVIDIA_H20`: Qwen3-30B-A3B, TP1 ⏎ * `E=128,N=384,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]`: Qwen3-235B-A22B-FP8, TP4 ⏎ * `E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]`: DeepSeek V3, TP8 ⏎ * `E=257,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]`: DeepSeek V3, TP16 ⏎  ⏎ ### DeepSeek-V …[truncated]

### L1-4063234c1a  (L1, 2025-08-13, sha 4063234c1a0e, PR #7687)
TITLE: Add H200 fused MoE kernel configs for DeepSeek-V3 in triton 3.3.1 (#7687)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=257,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎ After SGLang 0.4.8 we have triton 3.3.1 and different fused MoE config for different triton versions. I tuned several missing configs and get one with significant performance difference (DeepSeek V3, H200 TP16) ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ Tune: ⏎  ⏎ `python3 /sgl-workspace/sglang/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py --model /path-to-model --dtype fp8_w8a8 --tune --tp 16` ⏎  ⏎ (also tried `--tp 8`) ⏎  ⏎ Bench: ⏎  ⏎ ` …[truncated]

### L1-3d6be1fbce  (L1, 2025-08-13, sha 3d6be1fbce6c, PR #8018)
TITLE: add w8a8-fp8-block-wise H20-3e triton config (#8018)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4dbf43601d  (L1, 2025-08-14, sha 4dbf43601d49, PR #9065)
TITLE: fix: zero_init buffer (#9065)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=128,N=384,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=128,N=768,device_name=NVIDIA_H20.json (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=257,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+1/-0); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+1/-0); python/sglang/srt/layers/attention/trtllm_mha_backend.py (+8/-6); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+10/-3)
BODY: ## Motivation ⏎  ⏎ Flashinfer **v0.2.11.post3** feature: enable PDL, trtllm-gen attention zero_init buffer ⏎  ⏎ Fix crash at running gptoss introduced by this flashinfer update: ⏎ https://github.com/flashinfer-ai/flashinfer/pull/1463 ⏎  ⏎ How to re-produce the crash: ⏎ ``` ⏎ python3 -m sglang.launch_server --model openai/gpt-oss-120b --tp 8 --port 40000 --attention-backend trtllm_mha ⏎ OPENAI_API_KEY=“” python3 -m gpt_oss.evals --model openai/gpt-oss-120b  …[truncated]

### L1-5aa1ebd242  (L1, 2025-08-14, sha 5aa1ebd24289, PR #8112)
TITLE: [2/n]decouple quantization implementation from vLLM dependency (#8112)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.runner.marlin
FILES: sgl-kernel/csrc/moe/marlin_moe_wna16/core/registration.h (+0/-25); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel.h (+2/-2); sgl-kernel/csrc/moe/marlin_moe_wna16/marlin_template.h (+2/-3); sgl-kernel/csrc/moe/marlin_moe_wna16/ops.cu (+1/-3); sgl-kernel/python/sgl_kernel/fused_moe.py (+2/-1); python/sglang/srt/layers/quantization/utils.py (+24/-0); sgl-kernel/CMakeLists.txt (+4/-2); sgl-kernel/csrc/common_extension.cc (+22/-6); sgl-kernel/csrc/gemm/gptq/compat.cuh (+62/-0); sgl-kernel/csrc/gemm/gptq/gptq_kernel.cu (+1950/-0); (+22 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ The primary goal of this change is to enhance the consistency and stability of SGLang's quantization features. By decoupling the quantization implementation from its vLLM dependency, we aim to make the module easier to maintain and more portable. ⏎ Full realization of this goal will involve several subsequent PRs; this particular PR addresses the marlin kernel issues. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1fea998a45  (L1, 2025-08-14, sha 1fea998a452f, PR #9185)
TITLE: chore: bump sgl-kernel v0.3.5 (#9185)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2cc9eeab01  (L1, 2025-08-14, sha 2cc9eeab015d, PR #9191)
TITLE: [4/n]decouple quantization implementation from vLLM dependency (#9191)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.runner.marlin, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); .github/workflows/vllm-dependency-test.yml (+1/-6); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/quantization/__init__.py (+5/-32); python/sglang/srt/layers/quantization/awq.py (+10/-14); python/sglang/srt/layers/quantization/gptq.py (+10/-16); python/sglang/srt/layers/quantization/marlin_utils.py (+8/-3); test/srt/run_suite.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ follow https://github.com/sgl-project/sglang/pull/8112 ⏎ remove vllm dependency for AWQ and GPTQ quantization. ⏎  ⏎ co-author:@AniZpZ ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d2fbf2de0c  (L1, 2025-08-14, sha d2fbf2de0c1e, PR #9204)
TITLE: feat: add fused moe config for Qwen3-235B-A22B-FP8 on B200 (#9204)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=384,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎ Add fused MoE config for Qwen3-235B-A22B-FP8 on NVIDIA B200 ⏎  ⏎  ⏎ ## Benchmarking  ⏎  ⏎ With config: ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Backend:                                 sglang ⏎ Traffic request rate:                    inf ⏎ Max request concurrency:                 10 ⏎ Successful requests:                     100 ⏎ Benchmark duration (s):                  65.62 ⏎ Total input tokens:                      33 …[truncated]

### L1-392de007cb  (L1, 2025-08-14, sha 392de007cb6f, PR #9205)
TITLE: Minor fix docker container DeepEP on multi platforms (#9205)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+12/-1)
BODY: ## Motivation ⏎  ⏎ temp modify the ci yaml to let it run once ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-295895120d  (L1, 2025-08-14, sha 295895120df4, PR #8849)
TITLE: [6/N] MoE Refactor: Cleanup MoE-related configs (#8849)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py, L1.runner.framework, L1.runner.openai_triton_kernels, L1.runner.marlin, L1.ep.layer, L1.ep.deepep_dispatcher, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/__init__.py (+29/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+31/-30); python/sglang/srt/layers/moe/fused_moe_native.py (+14/-25); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+51/-69); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+46/-112); python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+20/-18); python/sglang/srt/layers/moe/moe_runner/__init__.py (+3/-0); python/sglang/srt/layers/moe/moe_runner/base.py (+13/-0); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+6/-0); python/sglang/srt/layers/moe/token_dispatcher/base_dispatcher.py (+55/-14); (+59 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ - Adding `--moe-runner-backend` and deprecating `--enable-triton-kernel-moe`, `--enable-flashinfer-cutlass-moe`, and `--enable-flashinfer-trtllm-moe`. ⏎ - Adding `TopKOutputChecker` and `DispatchOutputChecker` to make pylint happy. ⏎ - Adding some util functions to avoid calling `global_server_args` in moe-related logics. ⏎ - Adding `MoeRunnerConfig` to wrap up moe runner configs. ⏎ - Some minor cleanup ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  …[truncated]

### L1-e3e75a786a  (L1, 2025-08-14, sha e3e75a786a5b, PR #9214)
TITLE: Fix the deprecation warning for enable_flashinfer_mxfp4_moe (#9214)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+11/-5)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a3d99d6dcd  (L1, 2025-08-15, sha a3d99d6dcddd, PR #8790)
TITLE: [Misc] feat: Deepgemm update for sgl-kernel (#8790)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+46/-20)
LABELS: high priority
DEEP_STUDY: deep-study: this PR was reverted by PR 9260 (confirmed_revert, reason=ci_or_test_failure)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Update deepgemm sgl-kernel for nvrtc ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4fc09e0df0  (L1, 2025-08-15, sha 4fc09e0df0f0, PR #8777)
TITLE: Fp4 MOE quant kernel optimization (#8777)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/gemm/nvfp4_expert_quant.cu (+222/-41)
LABELS: high priority
BODY: ## Motivation ⏎ Port vLLM's [FP4 MoE kernel optimization (PR #19500)](https://github.com/vllm-project/vllm/pull/19500) to SGLang, improving performance of expert-based FP4 quantization on NVIDIA Blackwell GPUs. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ This PR introduces several optimizations for FP4 expert quantization: ⏎ - Switched to a grid-stride loop layout to replace per-block row processing, enabling better thread-level parallelism. ⏎ - Added launch configura …[truncated]

### L1-c186feed7f  (L1, 2025-08-15, sha c186feed7fb7, PR #9220)
TITLE: chore: bump sgl-kernel v0.3.6 (#9220)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
LABELS: high priority
DEEP_STUDY: deep-study: this PR was reverted by PR 9247 (confirmed_revert, reason=unstated)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-21b8846066  (L1, 2025-08-15, sha 21b884606676, PR #9198)
TITLE: [router] allow more health check configuration (#9198)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+34/-19); sgl-router/py_src/sglang_router/launch_router.py (+84/-19); sgl-router/src/config/types.rs (+55/-21); sgl-router/src/core/mod.rs (+2/-2); sgl-router/src/core/worker.rs (+54/-50); sgl-router/src/lib.rs (+48/-19); sgl-router/src/main.rs (+49/-21); sgl-router/src/routers/factory.rs (+2/-0); sgl-router/src/routers/pd_router.rs (+43/-10); sgl-router/src/routers/router.rs (+18/-4); (+5 more)
LABELS: enhancement, router
BODY: ## Motivation ⏎  ⏎ The current health check implementation in sgl-router immediately marks workers as unhealthy on the first failed health check, which can lead to false positives when workers are temporarily busy processing requests. This PR implements more robust health check configuration with consecutive success/failure thresholds to make health checking more robust and configurable. ⏎  ⏎ Additionally, the router was using the same interval confi …[truncated]

### L1-e52c3866eb  (L1, 2025-08-15, sha e52c3866eb77, PR #9243)
TITLE: chore(docker): update sgl_kernel version to 0.3.6 in Dockerfile.gb200 (#9243)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+1/-1)
DEEP_STUDY: deep-study: this PR was reverted by PR 9246 (confirmed_revert, reason=unstated)
BODY: 

### L1-5121af4627  (L1, 2025-08-15, sha 5121af4627ef, PR #9246)
TITLE: Revert "chore(docker): update sgl_kernel version to 0.3.6 in Dockerfi… (#9246)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+1/-1)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 9243 reason=unstated
BODY: …le.gb200 (#9243)" ⏎  ⏎ This reverts commit e52c3866eb7770076865da38618ce3cccc3af00f. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-87dab54824  (L1, 2025-08-15, sha 87dab548243b, PR #9247)
TITLE: Revert "chore: bump sgl-kernel v0.3.6 (#9220)" (#9247)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 9220 reason=unstated
BODY: This reverts commit c186feed7fb7604db59377e74d48bcc61053832e. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0c8594e67d  (L1, 2025-08-15, sha 0c8594e67d1e, PR #9231)
TITLE: Optional extension for green context (#9231)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+12/-1); sgl-kernel/csrc/common_extension.cc (+0/-6); sgl-kernel/csrc/spatial/greenctx_stream.cu (+3/-7); sgl-kernel/csrc/spatial_extension.cc (+29/-0); sgl-kernel/python/sgl_kernel/__init__.py (+14/-1); sgl-kernel/python/sgl_kernel/spatial.py (+15/-5)
BODY: follow #9021

### L1-eff4eb3fdd  (L1, 2025-08-15, sha eff4eb3fdd81, PR #7667)
TITLE: Add fp4 quantize before all-gather for Flashinfer cutlass MoE DP (max throughput) (#7667)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/__init__.py (+2/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-3); python/sglang/srt/layers/moe/topk.py (+7/-0); python/sglang/srt/layers/moe/utils.py (+23/-0); python/sglang/srt/distributed/device_communicators/pynccl.py (+68/-18); python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py (+52/-0); python/sglang/srt/distributed/parallel_state.py (+81/-0); python/sglang/srt/layers/communicator.py (+9/-2); python/sglang/srt/layers/dp_attention.py (+22/-3); python/sglang/srt/layers/logits_processor.py (+5/-1); (+6 more)
BODY: ## Motivation ⏎  ⏎ The goal of this PR is to optimize communications for DP with FlashInfer Cutlass MoE. ⏎  ⏎ ## Modifications ⏎  ⏎ Improvements include: ⏎ 1. Add Allgatherv collective. This is a pynccl implementation of TRT-LLM's allgather which supports varying sizes per rank and a list of tensors as inputs ⏎ 2. Add reducescatterv collective. This is a pynccl implementation of TRT-LLM's reducescatter which supports varying sizes per rank ⏎ 3. For Flashi …[truncated]

### L1-1c1f8a118e  (L1, 2025-08-16, sha 1c1f8a118ef8, PR #9049)
TITLE: Combine fp4.py and mxfp4.py into one file and support dynamic mxfp4 quantization in mxfp4.py (#9049)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-1); python/sglang/srt/layers/quantization/__init__.py (+8/-9); python/sglang/srt/layers/quantization/fp4.py (+0/-540); python/sglang/srt/layers/quantization/mxfp4.py (+156/-5); python/sglang/srt/layers/quantization/quark/quark.py (+390/-0); python/sglang/srt/layers/quantization/quark/quark_moe.py (+197/-0); python/sglang/srt/server_args.py (+3/-2)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ There are many duplicated features in the fp4.py and mxfp4.py. ⏎  ⏎ In order to simply them, combine these two files into one.  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ **[Dynamic Quant Grok-1]** ⏎ `~/sglang# python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 ⏎ 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ …[truncated]

### L1-0fc54b971e  (L1, 2025-08-17, sha 0fc54b971e14, PR #9272)
TITLE: [fix]:  fix cutlass moe ut and and Opt H20 cutlass groupGemm performance (#9272)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+109/-35); python/sglang/test/test_cutlass_moe.py (+4/-6); sgl-kernel/include/utils.h (+19/-0)
BODY: ## Motivation ⏎ 1.  fix cutlass moe UT ⏎ 2. add optimized cutlass groupGemm instance for H20(SGL_TUNE_DEVICE_KERNEL=1  to open opt) ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎ Deepseek-tp8 on H20 ⏎ | Batch Size | origin_h20 Cutlass fused_experts Time (ms) | opt_h20 Cutlass fused_experts Time (ms) | ⏎ |------------|-------------------------------------------|-------------------------------------------| ⏎ | 1 | 0 …[truncated]

### L1-a1c7f742f9  (L1, 2025-08-17, sha a1c7f742f912, PR #9286)
TITLE: chore: bump sgl-kernel v0.3.6.post1 (#9286)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4d98e48649  (L1, 2025-08-17, sha 4d98e4864998, PR #9260)
TITLE: Revert "[Misc] feat: Deepgemm update for sgl-kernel (#8790)" to fix kernel CI (#9260)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+20/-40)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 8790 reason=ci_or_test_failure
BODY: This reverts commit a3d99d6dcdddcfb6bde7b50c0830f0ad5534d74d. ⏎  ⏎ CI failes https://github.com/sgl-project/sglang/actions/runs/17002005000/job/48209894546 ⏎  ⏎ To reintroduce deepgemm upgrade, it depends on #9167 ⏎  ⏎ @fzyzcjy @FlamingoPg @zhyncs

### L1-968e181826  (L1, 2025-08-18, sha 968e1818261e, PR #9276)
TITLE: Fix triton_fused_moe unit test and benchmark (#9276)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/benchmark_sglang_fused_moe_triton.py (+24/-7); test/srt/test_triton_fused_moe.py (+17/-1)
BODY: ## Motivation ⏎  ⏎ The test_triton_fused_moe and benchmark_sglang_fused_moe_triton have been broken due to recent refactor. This PR is to fix these cases. ⏎  ⏎ **Unit test result:** ⏎ ``` ⏎ $python ./test/srt/test_triton_fused_moe.py ⏎ INFO 08-17 21:26:58 [__init__.py:235] Automatically detected platform cuda. ⏎ All deep_gemm operations loaded successfully! ⏎ WARNING:sglang.srt.layers.quantization.deep_gemm_wrapper.compile_utils:NVCC Compiler not found, u …[truncated]

### L1-c480a3f6ea  (L1, 2025-08-18, sha c480a3f6ea1b, PR #9289)
TITLE: Minor style fixes for sgl-kernel (#9289)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: sgl-kernel/python/sgl_kernel/fused_moe.py (+2/-2); docs/developer_guide/contribution_guide.md (+14/-0); python/pyproject.toml (+1/-1); python/sglang/eval/llama3_eval.py (+0/-1); python/sglang/profiler.py (+0/-1); python/sglang/utils.py (+0/-1); scripts/playground/replay_request_dump.py (+2/-1); sgl-kernel/CMakeLists.txt (+12/-4); sgl-kernel/csrc/common_extension.cc (+45/-33); sgl-kernel/csrc/common_extension_rocm.cc (+1/-0); (+7 more)
BODY: 

### L1-c6c379ab31  (L1, 2025-08-18, sha c6c379ab3161, PR #9320)
TITLE: [AMD] Reorganize hip-related header files in sgl-kernel (#9320)
SOURCES: path_core
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/csrc/moe/moe_align_kernel.cu (+0/-2); .github/workflows/pr-test-amd.yml (+1/-0); sgl-kernel/csrc/allreduce/mscclpp_allreduce.cuh (+1/-1); sgl-kernel/csrc/elementwise/activation.cu (+1/-1); sgl-kernel/csrc/gemm/per_tensor_quant_fp8.cu (+2/-2); sgl-kernel/csrc/gemm/per_token_quant_fp8.cu (+2/-2); sgl-kernel/include/hip/hip_act_and_mul.cuh (+0/-0); sgl-kernel/include/hip/hip_math_def.h (+1/-1); sgl-kernel/include/hip/hip_vec_dtypes.h (+0/-0); sgl-kernel/include/hip/impl/hip_vec_bf16_impl.h (+0/-0); (+4 more)
BODY: ## Motivation ⏎  ⏎ Reorganize the header files for AMD GPUs in `sgl-kernel` ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ - Reorganized the header files in `sgl-kernel` for AMD GPUs ⏎ - Changed `__HIP_PLATFORM_AMD__` to `USE_ROCM` ⏎ - Changed `WARP_SIZE=32` to `WARP_SIZE=C10_WARP_SIZE` defined in `include/utils.h` on AMD GPUs. I have tested it locally for the correctness and also confirmed that there is no performance regression using `benchmark/bench_moe_align_block_siz …[truncated]

### L1-3c2c9f6c9e  (L1, 2025-08-18, sha 3c2c9f6c9e7b, PR #9317)
TITLE: [Bug] Fix input arguments of flashinfer_trtllm_moe (#9317)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-2); python/sglang/srt/layers/moe/topk.py (+14/-14); python/sglang/srt/layers/quantization/fp8.py (+14/-5)
LABELS: bug, high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ flashinfer_trtllm fp8 moe kernel has several bugs in attributes, which are changed in moe refactoring. and add assertion to avoid using the fp8 kernel if n_group and topk_group are None ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-01d47a27b6  (L1, 2025-08-19, sha 01d47a27b6f6, PR #9327)
TITLE: [Bugfix] fix kv buffer register & dp attention & deepepmoe (#9327)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-1); python/sglang/srt/disaggregation/ascend/conn.py (+1/-3); python/sglang/srt/layers/dp_attention.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Fix kv register when kv alloc is separated on Ascend NPU ⏎ - Fix an issue in dp attention ⏎ - Fix an issue after refectoring of moe layer ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a3b810ebdb  (L1, 2025-08-19, sha a3b810ebdba1, PR #6295)
TITLE: fix: enable multi-GPU Triton fused MoE tuning (#6295)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+43/-37)
BODY: ## Motivation ⏎  ⏎ On multi-GPU AMD machines, `sglang/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` uses only a single GPU. Setting `CUDA_VISIBLE_DEVICES` doesn't help. ⏎  ⏎ ## Modifications ⏎  ⏎ Device ID is set explicitly for ROCm platform before benchmarking and tuning. ⏎ Two weeks ago, the same issue was fixed in [vLLM repo](https://github.com/vllm-project/vllm/pull/16263/files). ⏎  ⏎ ## Checklist

### L1-a91e90d9a3  (L1, 2025-08-20, sha a91e90d9a360, PR #8690)
TITLE: [2/2] Fuse routed scaling factor into select_experts (#8690)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+7/-0); python/sglang/srt/layers/moe/topk.py (+20/-0); python/sglang/srt/layers/quantization/fp8.py (+8/-9); python/sglang/srt/layers/quantization/modelopt_quant.py (+2/-4); python/sglang/srt/models/deepseek_v2.py (+12/-11); sgl-kernel/tests/test_moe_fused_gate.py (+6/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ See https://github.com/sgl-project/sglang/pull/8364 ⏎  ⏎ Requires https://github.com/sgl-project/sglang/pull/8770 [1/2][resubmit] sgl-kernel: Fuse routed scaling factor into moe_fused_gate (select_experts) ⏎  ⏎ ## Modifications ⏎  ⏎ See https://github.com/sgl-project/sglang/pull/8364 ⏎  ⏎ ## Accuracy Test ⏎  ⏎ See https://github.com/sgl-project/sglang/pull/8364 ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎ See https://github.com/sgl-project/sglang/pull/8 …[truncated]

### L1-c674bf9c6b  (L1, 2025-08-20, sha c674bf9c6b0a, PR #9420)
TITLE: Fix biased_grouped_topk_cpu (#9420)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-0)
LABELS: intel
BODY: Fix `biased_grouped_topk_cpu` by adding the argument `apply_routed_scaling_factor_on_output`.

### L1-7cd2ee06d7  (L1, 2025-08-20, sha 7cd2ee06d741, PR #9251)
TITLE: feat: Add Triton fallback option and SM120 MoE configs for FP8 models (#9251)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=352,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=384,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/quantization/fp8_utils.py (+20/-8)
LABELS: high priority
BODY: ## Summary ⏎ This PR adds support for running FP8 quantized models on SM120 (Blackwell) GPUs by: ⏎ 1. Adding `USE_TRITON_W8A8_FP8_KERNEL` environment variable to force Triton fallback ⏎ 2. Including optimized MoE configs for RTX 6000 Blackwell GPUs ⏎  ⏎ ## Problem ⏎ Currently, FP8 quantized models fail on SM120/Blackwell GPUs (RTX 5090/6000) with two issues: ⏎ 1. **CUTLASS kernel incompatibility**: The fp8_scaled_mm kernel doesn't support SM120, causing …[truncated]

### L1-18da2c96ec  (L1, 2025-08-21, sha 18da2c96ec09, PR #9384)
TITLE: [NVIDIA] Fix trtllm fp4 moe backend when used in MTP (#9384)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+5/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-0); python/sglang/srt/layers/moe/topk.py (+3/-1); python/sglang/srt/models/deepseek_v2.py (+2/-1)
LABELS: high priority
DEEP_STUDY: deep-study correctness case sglang:18da2c96ec: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: This PR addresses the issue mentioned [here](https://github.com/sgl-project/sglang/pull/9238#issuecomment-3199279509). ⏎  ⏎ The root cause is that some MoE layers may use an unquantized method. For example: ⏎ ``` ⏎ model.layers.3.mlp.experts → quantized   ⏎ model.decoder.mlp.experts  → unquantized ⏎ ``` ⏎  ⏎ This may be triggered by the MTP settings. ⏎  ⏎ `FusedMoE` works with both quantized and unquantized methods, which explains why the `flashinfer_cutla …[truncated]

### L1-de4990a5b2  (L1, 2025-08-21, sha de4990a5b2d1, PR #9392)
TITLE: [Bug] Fix w4afp8 moe kernel (#9392)
SOURCES: subject_keyword, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp (+4/-0)
DEEP_STUDY: deep-study correctness case sglang:de4990a5b2: class=nondeterminism_race_sync; symptom=nondeterministic_output; introducing=#7772
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR fixes an issue introduced in [PR7772](https://github.com/sgl-project/sglang/pull/7772). ⏎ When running `test_int4_fp8_grouped_gemm_multi_experts` in `sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py` with `k = 512, n = 1024`, the test may occasionally produce incorrect results. (It happens very rarely, but you can reproduce it more easily by setting `batch_size = 512` and `num_experts = 256`.) ⏎  ⏎ This bug also affects the  …[truncated]

### L1-275f9df381  (L1, 2025-08-21, sha 275f9df381fb, PR #9463)
TITLE: feat: add fused moe config for GLM-4.5-Air-FP8 on B200 (#9463)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json (+146/-0)
BODY: ## Motivation ⏎  ⏎ Add fused MoE config for GLM-4.5-Air-FP8 on NVIDIA B200 ⏎  ⏎  ⏎ ## Benchmarking ⏎ With config: ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     480 ⏎ Benchmark duration (s):                  159.80 ⏎ Total input tokens:                      1679845 ⏎ Total generated tokens:                  720000 ⏎ Request throughput (req/s):              3.00 ⏎ Request goodput (req/s):                 1. …[truncated]

### L1-5fd311d33e  (L1, 2025-08-21, sha 5fd311d33e62, PR #9333)
TITLE: [code  clean] add H20 cutlass groupGemm default  config (#9333)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+15/-35)
BODY: ## Motivation ⏎  ⏎ add default blockwise groupgemm config for  H20  ⏎ related to ： https://github.com/sgl-project/sglang/pull/9272 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ <img width="1725" height="221" alt="image" src="https://github.com/user-attachments/assets/13b37235-d2f5-4b5e-917a-7a892f293474" /> ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎ sgl-kernel/benchmark/bench_fp8_blockwise_group_gemm.py ⏎ **before:** ⏎ Benchmark: expected_m_per_group=128,  …[truncated]

### L1-b6b2287e4b  (L1, 2025-08-21, sha b6b2287e4b9f, PR #9475)
TITLE: chore: bump sgl-kernel v0.3.6.post2 (#9475)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f445a1d9a3  (L1, 2025-08-22, sha f445a1d9a3a3, PR #7699)
TITLE: [AMD] Fix Llama 4 FP8 accuracy issues on MI300X (#7699)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+0/-1); python/sglang/srt/layers/moe/rocm_moe_utils.py (+141/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+66/-15); python/sglang/srt/layers/quantization/fp8.py (+1/-0); python/sglang/srt/server_args.py (+4/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ - To fix Llama 4 FP8 accuracy issue on MI300X (https://github.com/sgl-project/sglang/issues/5362). ⏎ - This PR also includes the initial effort to streamline/modularize aiter's fused_moe implementations for various code paths. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ **Command to launch server** ⏎ ``` ⏎ SGLANG_USE_AITER=1 python -m sglang.launch_server --model-path /data/huggingface/meta-llama/Llama-4-Maverick-17B-128E-Instru …[truncated]

### L1-c4500233ff  (L1, 2025-08-22, sha c4500233ff20, PR #9456)
TITLE: Add Qwen3-30B-A3B-Thinking-2507 support on AMD GPUs. (#9456)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+18/-7)
BODY: ## Motivation ⏎ Add Qwen3-30B-A3B-Thinking model support on AMD GPUs.  ⏎  ⏎  ⏎ ## Modifications ⏎ Enabling triton backend on AMD GPU. ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ SGLANG_USE_AITER=0 python3 -m sglang.launch_server --model-path Qwen3-30B-A3B-Thinking-2507/ --tp 8 --trust-remote-code --chunked-prefill-size 130172 --max-running-requests 128 --mem-fraction-static 0.85 --attention-backend aiter --enable-torch-compile ⏎  ⏎ python3 benchmark/gsm8k/bench_sglang.py - …[truncated]

### L1-ccd3fb946e  (L1, 2025-08-23, sha ccd3fb946e04, PR #9473)
TITLE: [fix] Fix mxfp4 triton MoE tp bug (#9473)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-6); python/sglang/srt/layers/quantization/mxfp4.py (+7/-0); python/sglang/srt/models/gpt_oss.py (+5/-6)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/pull/9433 broke GPT-OSS on Hopper because the MoE intermediate size was not padded after sharding on Hopper. Adding padding would fix the issue. However, it's only a workaround. The correct fix is to change the way of the weight sharding for block scaling data formats (mxfp4, nvfp4 etc) to the following before creating the weights and then add padding on top of that: ⏎  ⏎ ``` ⏎         intermedi …[truncated]

### L1-86d10d220f  (L1, 2025-08-23, sha 86d10d220f66, PR #9532)
TITLE: Update grok.py and tiktoken tokenizer (#9532)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.router_py
FILES: python/sglang/srt/layers/moe/router.py (+15/-9); python/sglang/srt/constrained/xgrammar_backend.py (+10/-6); python/sglang/srt/hf_transformers_utils.py (+5/-0); python/sglang/srt/layers/attention/triton_backend.py (+16/-2); python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+31/-0); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+18/-0); python/sglang/srt/layers/elementwise.py (+94/-0); python/sglang/srt/layers/radix_attention.py (+6/-0); python/sglang/srt/models/grok.py (+376/-47); python/sglang/srt/tokenizer/tiktoken_tokenizer.py (+161/-0)
BODY: 

### L1-0374304a2c  (L1, 2025-08-23, sha 0374304a2cb6, PR #9004)
TITLE: Add enable_flashinfer_mxfp4_bf16_moe for higher precision and slower moe backend (#9004)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/mxfp4.py (+27/-5); python/sglang/srt/server_args.py (+9/-0); python/sglang/srt/managers/schedule_batch.py (+1/-0)
BODY: ## Motivation ⏎  ⏎ ``` ⏎ CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path openai/gpt-oss-20b --tp-size 1 --port 30000 --enable-flashinfer-mxfp4-moe ⏎  ⏎ CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server --model-path openai/gpt-oss-20b --tp-size 1 --port 31000 --enable-flashinfer-mxfp4-bf16-moe ⏎  ⏎ python3 -m sglang.bench_serving --backend sglang-oai  --dataset-name random --random-input-len 512 --random-output-len 1024 --random- …[truncated]

### L1-bf863e3bbf  (L1, 2025-08-24, sha bf863e3bbfff, PR #9565)
TITLE: fix: use sgl-kernel 0.3.5 (#9565)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-fda4792620  (L1, 2025-08-24, sha fda47926208e, PR #9559)
TITLE: Update CUTLASS 4.2 & Enable K-Major Scale Factor for SM90 FP8 Blockwise Group GEMM (#9559)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+0/-7); sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+49/-40); python/sglang/test/test_cutlass_moe.py (+33/-28); sgl-kernel/CMakeLists.txt (+1/-1); sgl-kernel/tests/test_fp8_blockwise_moe.py (+20/-57)
LABELS: high priority
BODY: ## Motivation ⏎ 1. CUTLASS 4.2 supports scale factors in the K-Major format, allowing us to unify the code path with Blackwell, thus avoiding some format conversion kernel calls(per_group_transpose).  ⏎ 2. In addition, we optimized scenes with smaller M by swapping the A/B matrices. ⏎ 3. Use the ATen interface to determine whether the current device is H20 to avoid performance impact of `cudaGetDeviceProperties`. ⏎  ⏎ TODO: ⏎ 1. Test performance on H20 …[truncated]

### L1-938e986e15  (L1, 2025-08-25, sha 938e986e1584, PR #9578)
TITLE: chore: upgrade flashinfer 0.2.14.post1 (#9578)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f8b757bcac  (L1, 2025-08-25, sha f8b757bcac5d, PR #9587)
TITLE: fix: resolve tuning fused moe issue (#9587)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+3/-3)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-051068673c  (L1, 2025-08-25, sha 051068673c67, PR #9591)
TITLE: chore: update config (#9591)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f92b729d52  (L1, 2025-08-25, sha f92b729d5240, PR #8328)
TITLE: [new feat] ascend backend support fia fusion kernel (#8328)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+1/-1); .github/workflows/pr-test-npu.yml (+3/-3); python/sglang/srt/layers/attention/ascend_backend.py (+218/-111); python/sglang/srt/mem_cache/memory_pool.py (+73/-14); python/sglang/srt/models/deepseek_v2.py (+9/-1); test/srt/ascend/test_ascend_mla_fia_w8a8int8.py (+103/-0); test/srt/ascend/test_ascend_mla_w8a8int8.py (+1/-0); test/srt/ascend/test_ascend_tp2_fia_bf16.py (+101/-0); test/srt/run_suite.py (+2/-0)
LABELS: high priority, ready-to-merge, npu
BODY: ## Motivation ⏎ In this MR, we implemented the NPU fusion kernels npu_fused_infer_attention_score in the Qwen2.5-7b, deepseek-v2-lite and deepseek-v3 models, this fusion kernel is  suitable for the graph mode. One needs to export **ASCEND_USE_FIA=ture** to activate this  fusion kernel. ⏎  ⏎ ## Modifications ⏎ Ascend Backend: support npu_fused_infer_attention_score kernel ⏎ Add unittest: test_ascend_tp_fia_bf16.py and test_ascend_mla_fia_w8a8int8.py ⏎  …[truncated]

### L1-90313fb09a  (L1, 2025-08-26, sha 90313fb09ac8, PR #9656)
TITLE: [router] add token bucket rate limiter (#9656)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+10/-1); sgl-router/py_src/sglang_router/launch_router.py (+42/-0); sgl-router/src/config/types.rs (+21/-0); sgl-router/src/core/mod.rs (+1/-0); sgl-router/src/core/token_bucket.rs (+195/-0); sgl-router/src/lib.rs (+17/-2); sgl-router/src/main.rs (+3/-0); sgl-router/src/middleware.rs (+189/-2); sgl-router/src/server.rs (+30/-4); sgl-router/tests/api_endpoints_test.rs (+12/-0); (+5 more)
LABELS: enhancement, router
BODY: ## Motivation ⏎  ⏎ This PR introduces a proper bucket rate limiting system to the SGLang router. ⏎ The new system provides rate limiting with burst capacity and optional request queuing for better handling of traffic spikes. ⏎  ⏎ ## Modifications ⏎  ⏎ ### Core Implementation ⏎  ⏎ - Token Bucket Algorithm (sgl-router/src/core/token_bucket.rs): ⏎   - Implements a standard token bucket with configurable capacity and refill rate ⏎   - Supports burst capacity fo …[truncated]

### L1-8f7b1c31e8  (L1, 2025-08-26, sha 8f7b1c31e825, PR #9677)
TITLE: Add A100 fused MoE kernel configs for Dpsk (#9677)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=257,N=64,device_name=NVIDIA_A100-SXM4-80GB.json (+146/-0); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ New fused MoE configs for DeepSeek V3/R1/3.1 bf16 model, tp32 on A100-SXM4-80GB (4nodes with 8gpus each) ⏎ First fused moe configs for deepseek bf16 model on A100 ⏎  ⏎ ## Modifications ⏎  ⏎ E=257,N=64,device_name=NVIDIA_A100-SXM4-80GB.json: DeepSeek V3/R1/3.1, TP32 ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎ ### Deploy model: ⏎  ⏎ `if [ -z "$RANK" ]; then ⏎     echo "ERROR: RANK environment variable is not set." ⏎     exit 1 ⏎ fi ⏎  ⏎ if [ "$RANK" - …[truncated]

### L1-79e6a8a6ac  (L1, 2025-08-26, sha 79e6a8a6acd8, PR #9495)
TITLE: support cuda 13.0 and trtllm kernel by Aug 25 2025 (#9495)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_softmax, L1.runner.marlin
FILES: sgl-kernel/CMakeLists.txt (+23/-9); sgl-kernel/csrc/moe/marlin_moe_wna16/generate_kernels.py (+25/-2); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel.h (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4b8.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku8b128.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4b8.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku8b128.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_marlin.cuh (+10/-0); (+3 more)
BODY: ## Motivation ⏎  ⏎ #9490  ⏎  ⏎ - Support cuda130 with custom flashinfer and trtllm kernel [Aug 25 2025](https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/80c7f8677a063859c07760dc98fe235db0ea22a8/fmha/) ⏎ - Support sm_110 and sm_121 on cuda 130 ⏎ - Support --compress-mode=size on cuda 130 ⏎ - Keep sm_101  support on cuda 128/129 ⏎  ⏎  ⏎ ### Test ⏎ - Step 1, use nvidia pytorch 25.08 image  ⏎ ```bash ⏎ docker pull nvcr.io/nvi …[truncated]

### L1-fd71b11b1d  (L1, 2025-08-27, sha fd71b11b1d96, PR #9679)
TITLE: move is_sm90_supported/is_sm100_supported to python/sglang/srt/utils.py (#9679)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+0/-8); python/sglang/srt/layers/attention/flashinfer_backend.py (+5/-2); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+5/-2); python/sglang/srt/layers/communicator.py (+1/-2); python/sglang/srt/layers/quantization/fp8.py (+2/-1); python/sglang/srt/layers/quantization/fp8_utils.py (+1/-1); python/sglang/srt/layers/quantization/mxfp4.py (+1/-2); python/sglang/srt/layers/utils.py (+0/-14); python/sglang/srt/model_executor/model_runner.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+3/-2); (+3 more)
BODY: 

### L1-aa3eba8eb4  (L1, 2025-08-27, sha aa3eba8eb42c, PR #9340)
TITLE: [sgl-kernel] misc: update deepgemm version for sgl-kernel (#9340)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_softmax, L1.runner.marlin, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-7); sgl-kernel/CMakeLists.txt (+48/-44); sgl-kernel/csrc/moe/marlin_moe_wna16/generate_kernels.py (+2/-25); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel.h (+0/-1); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4.cu (+0/-1); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4b8.cu (+0/-1); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku8b128.cu (+0/-1); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4.cu (+0/-1); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4b8.cu (+0/-1); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku8b128.cu (+0/-1); (+15 more)
LABELS: bug, enhancement, high priority, dependencies
BODY: ## Motivation ⏎  ⏎ DeepGEMM updated for unify cuda version. ⏎ So we need upd for TORCH LIBRARY. ⏎  ⏎ It depends on: https://github.com/sgl-project/sglang/pull/9167 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b962a296ed  (L1, 2025-08-27, sha b962a296edbe, PR #9708)
TITLE: chore: upgrade sgl-kernel 0.3.7 (#9708)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); .github/workflows/vllm-dependency-test.yml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-bc80dc4ce0  (L1, 2025-08-27, sha bc80dc4ce0ae, PR #9716)
TITLE: chore: bump v0.5.1.post3 (#9716)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+2/-2); python/pyproject.toml (+1/-1); benchmark/deepseek_v3/README.md (+1/-1); docs/get_started/install.md (+2/-2); docs/platforms/amd_gpu.md (+1/-1); docs/platforms/ascend_npu.md (+1/-1); python/sglang/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-28684f909d  (L1, 2025-08-27, sha 28684f909dce, PR #9720)
TITLE: [router] upgrade kernel version in pd ci (#9720)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-pd-router.yml (+2/-2)
LABELS: router, ci
BODY: update sgl-kernel and genai-bench version in pd-ci ⏎  ⏎ ## Checklist

### L1-6b39f9cf8c  (L1, 2025-08-28, sha 6b39f9cf8c51, PR #9721)
TITLE: Support compile sgl-kernel on cuda 13.0 (#9721)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_softmax, L1.runner.marlin
FILES: sgl-kernel/csrc/moe/marlin_moe_wna16/generate_kernels.py (+25/-2); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel.h (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4b8.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku8b128.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4b8.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku8b128.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_marlin.cuh (+10/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/marlin_template.h (+2/-0); (+3 more)
BODY: ## Motivation ⏎  ⏎  ⏎ #9490  ⏎ [PR 9495](https://github.com/sgl-project/sglang/pull/9495) ⏎  ⏎ - Support cuda130 with custom flashinfer and trtllm kernel [Aug 25 2025](https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/80c7f8677a063859c07760dc98fe235db0ea22a8/fmha/) ⏎ - Support sm_110 and sm_121 on cuda 130 ⏎ - Support --compress-mode=size on cuda 130 ⏎ - Keep sm_101  support on cuda 128/129 ⏎  ⏎  ⏎ ### Test ⏎ - Step 1, use …[truncated]

### L1-dc20c22f76  (L1, 2025-08-28, sha dc20c22f764c, PR #9770)
TITLE: feat: add tuned fused moe config for GLM-4.5-Air-FP8 tp = 4 on B200 (#9770)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=129,N=352,device_name=NVIDIA_B200,dtype=fp8_w8a8.json (+146/-0)
BODY: ## Motivation ⏎  ⏎ Add fused MoE config for GLM-4.5-Air-FP8 on NVIDIA B200, tp = 4 ⏎  ⏎ ## Benchmarking ⏎  ⏎ With config: ⏎  ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     516 ⏎ Maximum request concurrency:             256 ⏎ Request rate configured (RPS):           10.00 ⏎ Benchmark duration (s):                  129.15 ⏎ Total input tokens:                      1805842 ⏎ Total generated tokens:             …[truncated]

### L1-74dd4249ac  (L1, 2025-08-28, sha 74dd4249ac60, PR #9355)
TITLE: [Feature] Support NPUGraph for DeepSeek on Ascend NPU (#9355)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+12/-6); python/sglang/srt/layers/moe/topk.py (+12/-2); python/sglang/srt/disaggregation/ascend/conn.py (+75/-0); python/sglang/srt/layers/attention/ascend_backend.py (+183/-88); python/sglang/srt/layers/quantization/w8a8_int8.py (+7/-3); python/sglang/srt/mem_cache/memory_pool.py (+4/-0); python/sglang/srt/models/deepseek_v2.py (+14/-6)
LABELS: high priority, ready-to-merge, npu
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ this pr improves deepseek model (mla to be exact) performance with npugraph support ⏎ check initial npugraph support on mha model here #9399 and #8030 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - support concurrent kvcache transfer following mooncake transfer engine ⏎ - add mla graph support in ascend attention backend ⏎ - some performance and accuracy improvements ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ <img width="925" height="72" alt="87DF5544-5E77-4E8C …[truncated]

### L1-5ad296bda1  (L1, 2025-08-28, sha 5ad296bda140, PR #8750)
TITLE: Optimize prefill performance on cpu backend (#8750)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/cpu/moe.cpp (+39/-69); sgl-kernel/csrc/cpu/moe_fp8.cpp (+51/-46); sgl-kernel/csrc/cpu/moe_int8.cpp (+334/-96); sgl-kernel/csrc/cpu/common.h (+118/-1); sgl-kernel/csrc/cpu/gemm.cpp (+30/-14); sgl-kernel/csrc/cpu/gemm.h (+4/-3); sgl-kernel/csrc/cpu/gemm_fp8.cpp (+34/-31); sgl-kernel/csrc/cpu/gemm_int8.cpp (+70/-12); sgl-kernel/csrc/cpu/qkv_proj.cpp (+1/-2)
LABELS: high priority, sgl-kernel, ready-to-merge, intel, cpu
BODY: ## Motivation ⏎  ⏎ This PR aims at improving prefill performance for CPU backend, it will improve `bfloat16`, `int8_w8a8` and `fp8_w8a8` paths. This one has no effective on decoding performance. ⏎  ⏎ ## Modifications ⏎  ⏎ * **amx-int8**: enable **amx-int8** for GEMM and MoE kernels. As SGLang pops up with PyTorch 2.7, we are free to use amx-int8 on SGLang CPU backend. ⏎ * **thread blocking**: the original code base utilizes a simple blocking scheme for  …[truncated]

### L1-7a16db9bd9  (L1, 2025-08-28, sha 7a16db9bd9ab, PR #9789)
TITLE: Make sm100 fp8 kernels available on sm103 (#9789)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+6/-2); sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu (+5/-1); sgl-kernel/csrc/gemm/fp8_gemm_kernel.cu (+5/-1)
BODY: ## Motivation ⏎  ⏎ The tests failed on B300 without the fix. ⏎  ⏎ ## Accuracy Tests ⏎ On B300 ⏎ <img width="1718" height="618" alt="image" src="https://github.com/user-attachments/assets/c1dd80eb-c0ca-432f-8e27-4244d727115d" /> ⏎  ⏎  ⏎ ## Checklist

### L1-3d8fc43400  (L1, 2025-08-29, sha 3d8fc43400ba, PR #9793)
TITLE: chore: upgrade flashinfer 0.3.0rc1 (#9793)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-42f34437ab  (L1, 2025-08-29, sha 42f34437abeb, PR #9670)
TITLE: Adds initialize_moe_config to bench_one_batch so MOE backend is respected (#9670)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/bench_one_batch.py (+3/-0)
BODY: ## Motivation ⏎  ⏎ Currently, the `bench_one_batch` script does not respect the MOE backend option and will always use the default backend. In other entrypoints (e.g. `launch_server`) this is not an issue because they most likely use the scheduler, which calls `initialize_moe_config` and therefore uses the correct backend.  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ Updated `bench_one_batch` to call `initialize_moe_config` so that the correct MOE backend is used.  ⏎  …[truncated]

### L1-9c99949ef3  (L1, 2025-08-30, sha 9c99949ef3d9, PR #9820)
TITLE: chore: update Dockerfile (#9820)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9970e3bf32  (L1, 2025-08-30, sha 9970e3bf328a, PR #9822)
TITLE: chore: upgrade sgl-kernel 0.3.7.post1 with deepgemm fix (#9822)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+2/-2); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5e194b2143  (L1, 2025-08-30, sha 5e194b21437f, PR #9824)
TITLE: [Model] Support Meituan LongCat-Flash && LongCat-Flash-MTP (#9824)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+74/-0); python/sglang/srt/layers/moe/topk.py (+23/-10); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/longcat_flash.py (+104/-0); python/sglang/srt/configs/model_config.py (+12/-0); python/sglang/srt/hf_transformers_utils.py (+2/-0); python/sglang/srt/layers/quantization/utils.py (+13/-0); python/sglang/srt/model_executor/model_runner.py (+4/-1); python/sglang/srt/models/longcat_flash.py (+1015/-0); python/sglang/srt/models/longcat_flash_nextn.py (+691/-0)
LABELS: enhancement, high priority
BODY: ## Motivation ⏎  ⏎ Support Meituan LongCat-Flash && LongCat-Flash-MTP ⏎  ⏎ ## Modifications ⏎  ⏎ [python/sglang/srt/models/longcat_flash.py](https://github.com/sgl-project/sglang/pull/9824/files/be711b71a60b8dd894c14b45ed2108ece803d6d8#diff-37eb91f71e6f2207973471f07977fb97e6ba216d834f8c490b689733d3950e7b) ⏎ [python/sglang/srt/models/longcat_flash_nextn.py](https://github.com/sgl-project/sglang/pull/9824/files/be711b71a60b8dd894c14b45ed2108ece803d6d8#dif …[truncated]

### L1-349b491c63  (L1, 2025-09-01, sha 349b491c635a, PR #9864)
TITLE: chore: upgrade flashinfer 0.3.0 (#9864)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9a0cac1be0  (L1, 2025-09-01, sha 9a0cac1be0e5, PR #9893)
TITLE: [router] add grpc pd and regular router init (#9893)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+6/-0); sgl-router/py_src/sglang_router/launch_router.py (+20/-0); sgl-router/py_test/test_launch_router.py (+2/-0); sgl-router/src/config/types.rs (+32/-0); sgl-router/src/config/validation.rs (+71/-2); sgl-router/src/lib.rs (+59/-0); sgl-router/src/main.rs (+57/-2); sgl-router/src/routers/factory.rs (+131/-36); sgl-router/src/routers/grpc/pd_router.rs (+226/-7); sgl-router/src/routers/grpc/router.rs (+154/-7); (+4 more)
LABELS: feature, router
BODY: ## Summary ⏎  ⏎ This PR implements gRPC router constructors for SGL Router. The router automatically detects the connection mode based on worker URLs and initializes the appropriate router implementation. ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ 1.  Automatic Connection Mode Detection ⏎ 2. gRPC Router Constructor Implementation (`src/routers/grpc/router.rs`) ⏎ 3. gRPC PD Router Constructor Implementation (`src/routers/grpc/pd_router.rs`) ⏎ 4. Route …[truncated]

### L1-d4a938417d  (L1, 2025-09-01, sha d4a938417d2c, PR #8118)
TITLE: [feat] Support tp mode for DeepSeek-R1-W4AFP8 (#8118)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.cutlass.w4a8, L1.ep.layer, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+1/-9); python/sglang/srt/layers/moe/ep_moe/layer.py (+0/-3); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-2); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_get_group_starts.cuh (+1/-1); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu (+206/-60); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh (+7/-6); python/sglang/srt/configs/model_config.py (+2/-1); python/sglang/srt/layers/quantization/w4afp8.py (+30/-25); python/sglang/srt/models/deepseek_v2.py (+5/-0); python/sglang/test/test_cutlass_w4a8_moe.py (+24/-9); (+1 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Support tp mode for DeepSeek w4a8 model, which has a better performace than ep mode. ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Add W4AFp8MoEMethod and associated `create_weights`, `process_weights_after_loading` function and `apply` function. In the apply function, we use the same cutlass_w4a8_moe kernel as ep moe uses. ⏎ 2. Add some tile shape and cluster shape config for tp moe in `cutlass_w4a8_moe` kernel. ⏎ 3. Add a router logic in w4afp8 quan …[truncated]

### L1-b9eb0d9c2b  (L1, 2025-09-02, sha b9eb0d9c2bac, PR #9844)
TITLE: Change tensor alignment method to mn major (#9844)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-4)
BODY: ## Modifications ⏎  ⏎ Change tensor alignment method to mn major (from changes on #9340).

### L1-b7361cc444  (L1, 2025-09-02, sha b7361cc4441d, PR #9916)
TITLE: [Fix] fix the issue encountered when inference LongCat-Flash/MTP EP MoE on b200 (#9916)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/longcat_flash.py (+26/-15); python/sglang/srt/models/longcat_flash_nextn.py (+23/-15)
LABELS: high priority
BODY: …en weight_requant_ue8m0 is required ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  fix the issue encountered when inference LongCat-Flash/MTP EP MoE on b200 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1db649ac02  (L1, 2025-09-02, sha 1db649ac0204, PR #9879)
TITLE: [feat] apply deep_gemm compile_mode to skip launch (#9879)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+2/-2); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py (+8/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ do not merged until next version of sgl-kernel bumped ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b5245064f6  (L1, 2025-09-02, sha b5245064f66e, PR #9878)
TITLE: [code style] restruct fused_moe to avoid very long single file (#9878)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/__init__.py (+5/-3); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+5/-1048); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py (+212/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+796/-0); python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py (+87/-0)
BODY: - unit-test ⏎  ⏎ <img width="967" height="406" alt="图片" src="https://github.com/user-attachments/assets/2626a405-77a2-4faa-b21b-c5847f8a97f6" /> ⏎  ⏎ - MoE model acc test ⏎  ⏎ ```shell ⏎ python3 -m sglang.launch_server --model-path Qwen/Qwen2-57B-A14B-Instruct  --trust-remote-code  --tp 4 ⏎ ``` ⏎  ⏎ main: ⏎  ⏎ ```shell ⏎ ➜  sglang git:(main) ✗  python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8 ⏎ 100%|███████████████████ …[truncated]

### L1-60e37f8028  (L1, 2025-09-02, sha 60e37f8028e7, PR #9912)
TITLE: Move parsers under a single folder (#9912)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+0/-1); docs/advanced_features/separate_reasoning.ipynb (+1/-1); docs/advanced_features/vlm_query.ipynb (+2/-2); examples/runtime/engine/offline_batch_inference_vlm.py (+1/-1); python/sglang/lang/interpreter.py (+1/-1); python/sglang/srt/entrypoints/http_server.py (+1/-1); python/sglang/srt/entrypoints/openai/serving_chat.py (+3/-3); python/sglang/srt/entrypoints/openai/serving_completions.py (+3/-1); python/sglang/srt/entrypoints/openai/serving_embedding.py (+1/-1); python/sglang/srt/entrypoints/openai/serving_responses.py (+1/-1); (+18 more)
BODY: 

### L1-d631290e32  (L1, 2025-09-02, sha d631290e32b8, PR #9905)
TITLE: Remove annoying warnings in sgl kernel build (#9905)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+34/-32); sgl-kernel/Makefile (+1/-2); sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+2/-2); sgl-kernel/csrc/gemm/dsv3_fused_a_gemm.cu (+1/-0); sgl-kernel/csrc/gemm/nvfp4_expert_quant.cu (+5/-0)
BODY: remove most nvcc warnings by either fixing the code or suppressing them

### L1-8cbf71dc2d  (L1, 2025-09-03, sha 8cbf71dc2d74, PR #9978)
TITLE: Triton 3.4.0 MoE config for Deepseek TP16 H100 (#9978)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=257,N=128,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎ There was no config for Deepseek H100 and triton 3.4.0

### L1-f78b7fd16d  (L1, 2025-09-03, sha f78b7fd16dbf, PR #9953)
TITLE: [1/N][Bug] Fix w4afp8 MoE NaN issue (sgl-kernel) (#9953)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L1.cutlass.w4a8
FILES: sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh (+2/-2)
DEEP_STUDY: deep-study: this PR was reverted by PR 10097 (confirmed_revert, reason=ci_or_test_failure)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR provides the sgl-kernel changes for [PR#9918](https://github.com/sgl-project/sglang/pull/9918). ⏎  ⏎ Update: This PR lacks unit test. The updated version is here: [PR#10108](https://github.com/sgl-project/sglang/pull/10108). ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Update the w4afp8 grouped gemm kernel to use bfloat16 as the output type. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ See [PR#9918](https://github.com/sgl-project/sglang/pull/9918). ⏎  ⏎ ## …[truncated]

### L1-0e9387a95d  (L1, 2025-09-04, sha 0e9387a95ddc, PR #10052)
TITLE: fix: update gb200 dep (#10052)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-ec15c8360e  (L1, 2025-09-04, sha ec15c8360e73, PR #9973)
TITLE: Optimize Qwen3-moe model by using flashinfer fused allreduce  (#9973)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen2_moe.py (+4/-1); python/sglang/srt/models/qwen3_moe.py (+39/-8); python/sglang/srt/layers/communicator.py (+9/-3)
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is to make qwen moe model leverage FlashInfer fused_allreduce to fuse allreduce+rmsnorm+residual_add.  ⏎ The E2E input throughput improved 2.2%. Currently use a small MoE model so the speedup is slightly, larger 235B model can gain better performance. Will try and update the report. ⏎  ⏎ Before fusion: ⏎ The AllReduce 13.26% + FusedNormAdd 6.45% == 19.71% ⏎ <img width="2994" height="1456" alt="image" src="https://github.com/ …[truncated]

### L1-6e95f5e5bd  (L1, 2025-09-05, sha 6e95f5e5bd24, PR #9964)
TITLE: Simplify `Router` arguments passing and build it in docker image (#9964)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+41/-123); docker/Dockerfile (+14/-1); docs/advanced_features/pd_disaggregation.md (+3/-3); docs/advanced_features/router.md (+15/-15); docs/references/multi_node_deployment/lws_pd/lws-examples/lb.yaml (+2/-1); docs/references/multi_node_deployment/lws_pd/lws_pd_deploy.md (+2/-1); python/sglang/srt/disaggregation/launch_lb.py (+0/-118); python/sglang/srt/disaggregation/mini_lb.py (+6/-445); python/sglang/srt/disaggregation/utils.py (+2/-49); python/sglang/srt/entrypoints/http_server.py (+1/-13); (+14 more)
LABELS: documentation, high priority, router
BODY: This PR does: ⏎ - Simplify the Python code of the router, i.e., remove the argument passing duplication. ⏎ - Move mini_lb into the router folder, and make mini_lb use the same argument space as router. ⏎ - Build router in dockerfile by default.

### L1-2985090084  (L1, 2025-09-05, sha 298509008451, PR #10087)
TITLE: Update flashinfer to 0.3.1 for B300 support (#10087)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ Update flashinfer to 0.3.1 for initial B300 support ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ CI ⏎  ⏎ ## Checklist

### L1-adf73175d6  (L1, 2025-09-05, sha adf73175d617, PR #9567)
TITLE: Forbid DeepEP racing condition when too many tokens (#9567)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+3/-0)
BODY: ## Motivation ⏎  ⏎ todo: pass CI ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0e78c63c0e  (L1, 2025-09-05, sha 0e78c63c0ec9, PR #10097)
TITLE: Revert "[1/N][Bug] Fix w4afp8 MoE NaN issue (sgl-kernel) (#9953)" (#10097)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L1.cutlass.w4a8
FILES: sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh (+2/-2)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 9953 reason=ci_or_test_failure
BODY: This reverts commit f78b7fd16dbfe32c2ee73c1f3fef49fc1257b27f. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3fa62da78c  (L1, 2025-09-05, sha 3fa62da78c12, PR #9269)
TITLE: [7/N] MoE Refactor: the implementation of new framework (#9269)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.runner.framework, L1.runner.triton, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/__init__.py (+2/-1); python/sglang/srt/layers/moe/fused_moe_native.py (+5/-3); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+5/-2); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+48/-29); python/sglang/srt/layers/moe/moe_runner/__init__.py (+2/-1); python/sglang/srt/layers/moe/moe_runner/base.py (+284/-1); python/sglang/srt/layers/moe/moe_runner/runner.py (+84/-0); python/sglang/srt/layers/moe/moe_runner/triton.py (+442/-0); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+16/-2); python/sglang/srt/layers/moe/token_dispatcher/base.py (+68/-7); (+24 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR implements a new MoE framework introduced in #8715 to streamline the integration of new all-to-all backends or grouped-gemm backends.  ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ One MoE forward function can be decoupled into three parts: dispatch -> pre-permute -> core runner -> post-permute -> combine. With this PR, a developer can implement dispatch/combine in the FusedMoE module and define their customized moe runner. The de …[truncated]

### L1-9a719b7afc  (L1, 2025-09-05, sha 9a719b7afcc2, PR #9764)
TITLE: [NVIDIA] Remove unused `get_fused_moe_impl_class` function (#9764)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-13); python/sglang/srt/layers/quantization/fp8.py (+1/-5)
BODY: While checking the MTP MoE, I noticed this function is never used in the repo. We should remove `get_fused_moe_impl_class` to avoid confusion with the correct function: `get_moe_impl_class`.

### L1-a5a03209e9  (L1, 2025-09-06, sha a5a03209e959, PR #10107)
TITLE: Fix circular import (#10107)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.framework, L1.runner.triton, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/moe_runner/base.py (+8/-18); python/sglang/srt/layers/moe/moe_runner/runner.py (+1/-5); python/sglang/srt/layers/moe/moe_runner/triton.py (+10/-4); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+5/-5); python/sglang/srt/layers/quantization/w4afp8.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5f1eb20484  (L1, 2025-09-06, sha 5f1eb2048427, PR #9956)
TITLE: [chore] Remove unused ep_moe cuda kernels (#9956)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.reorder_aot
FILES: sgl-kernel/CMakeLists.txt (+0/-2); sgl-kernel/csrc/common_extension.cc (+0/-12); sgl-kernel/csrc/moe/ep_moe_reorder_kernel.cu (+0/-181); sgl-kernel/csrc/moe/ep_moe_silu_and_mul_kernel.cu (+0/-115); sgl-kernel/include/sgl_kernel_ops.h (+0/-29); sgl-kernel/python/sgl_kernel/__init__.py (+0/-3); sgl-kernel/python/sgl_kernel/moe.py (+0/-64); sgl-kernel/benchmark/bench_moe_ep_post_reorder.py (+4/-22); sgl-kernel/benchmark/bench_moe_ep_pre_reorder.py (+0/-103); sgl-kernel/benchmark/bench_moe_silu_and_mul.py (+0/-92); (+3 more)
BODY: ## Motivation ⏎  ⏎ Clean up some unused kernels causing accuracy issues in https://github.com/sgl-project/sglang/issues/9944, benchmarks scripts, and unit tests (some of which fail on B200/B300 due to larger atol needed). ⏎  ⏎ ## Checklist

### L1-cb3918a091  (L1, 2025-09-07, sha cb3918a09127, PR #9477)
TITLE: Optimize moe_sum_reduce_kernel (#9477)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+23/-20); benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py (+24/-21)
BODY: ## Motivation ⏎  ⏎ During the Prefill stage of TP MoE, the moe_sum_reduce_kernel operator accounts for a relatively large proportion. ⏎ <img width="2996" height="1606" alt="image" src="https://github.com/user-attachments/assets/c49f154e-92dc-41a5-a0e5-dbf5fc645b87" /> ⏎  ⏎ This PR is to optimize moe_sum_reduce_kernel and get speedup up to 17.6%. ⏎ DeepSeek-V3 E2E throughput speedup 2-5%. ⏎  ⏎ ============Main============ ⏎ TritonKernel column is the conce …[truncated]

### L1-5a7e10fe4c  (L1, 2025-09-07, sha 5a7e10fe4c1e, PR #10144)
TITLE: [MoE] fix: incorrect weight initialization for  cutlass_fused_experts_fp8 (#10144)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8.py (+1/-1)
ISSUES: #10138 [Bug] Qwen3 w8a8 generate garbage output with all MoE backend
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ fix #10138  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-7577f0e40f  (L1, 2025-09-07, sha 7577f0e40f56, PR #7843)
TITLE: Add graph runner support with torch compile on CPU (#7843)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-xeon.yml (+1/-1); docs/platforms/cpu_server.md (+6/-1); python/sglang/srt/distributed/parallel_state.py (+4/-3); python/sglang/srt/layers/attention/intel_amx_backend.py (+3/-0); python/sglang/srt/layers/quantization/fp8.py (+3/-0); python/sglang/srt/layers/quantization/w8a8_int8.py (+5/-7); python/sglang/srt/managers/scheduler.py (+4/-5); python/sglang/srt/managers/scheduler_metrics_mixin.py (+1/-1); python/sglang/srt/model_executor/cpu_graph_runner.py (+640/-0); python/sglang/srt/model_executor/forward_batch_info.py (+3/-0); (+6 more)
LABELS: high priority, intel, cpu
BODY: ## Motivation ⏎  ⏎ Inspired by https://github.com/mingfeima/sglang/pull/73. We add CPU graph runner with torch compile to reduce python overhead to speed up decoding on CPU. ⏎  ⏎ Profiling with disabling torch compile: ⏎ <img width="1042" height="325" alt="image" src="https://github.com/user-attachments/assets/567b3d68-a53c-4f7e-b5ba-0b39276e68c7" /> ⏎  ⏎ Profiling with enabling torch compile: ⏎ <img width="812" height="278" alt="image" src="https://gith …[truncated]

### L1-ee0b3c5bad  (L1, 2025-09-07, sha ee0b3c5bad6c, PR #10108)
TITLE: [1/N][Bug] Fix w4afp8 MoE NaN issue (sgl-kernel, fixed) (#10108)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.cutlass.w4a8
FILES: sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh (+2/-2); sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py (+5/-6)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR provides the sgl-kernel changes for [PR#9918](https://github.com/sgl-project/sglang/pull/9918). It fixes [PR#9953](https://github.com/sgl-project/sglang/pull/9953). ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Update the w4afp8 grouped gemm kernel to use bfloat16 as the output type. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ See [PR#9918](https://github.com/sgl-project/sglang/pull/9918). ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ See [PR#9918](https://g …[truncated]

### L1-c8295d2353  (L1, 2025-09-07, sha c8295d235386, PR #6226)
TITLE: enable auto-round quantization model (#6226)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+9/-0); docs/advanced_features/quantization.md (+88/-0); python/sglang/srt/configs/model_config.py (+1/-0); python/sglang/srt/layers/quantization/__init__.py (+2/-0); python/sglang/srt/layers/quantization/auto_round.py (+360/-0); python/sglang/srt/server_args.py (+1/-0); python/sglang/test/test_utils.py (+5/-0); test/srt/quant/test_autoround.py (+62/-0)
LABELS: high priority, intel
DEEP_STUDY: deep-study: this PR was reverted by PR 10148 (confirmed_revert, reason=ci_or_test_failure)
BODY: This pr is to support models quantized by AutoRound [github](https://github.com/intel/auto-round) [paper](https://arxiv.org/abs/2309.05516), ⏎  ⏎ AutoRound delivers significantly higher accuracy at extremely low bit-widths (e.g., 2-bit) and offers broader compatibility across models (LLMs and VLMs), quantization formats, and configurations. You can check out our github/paper or this [blog post](https://huggingface.co/blog/autoround). ⏎  ⏎ AutoRound h …[truncated]

### L1-b7d1f17b8d  (L1, 2025-09-07, sha b7d1f17b8da9, PR #10148)
TITLE: Revert "enable auto-round quantization model (#6226)" (#10148)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-9); docs/advanced_features/quantization.md (+0/-88); python/sglang/srt/configs/model_config.py (+0/-1); python/sglang/srt/layers/quantization/__init__.py (+0/-2); python/sglang/srt/layers/quantization/auto_round.py (+0/-360); python/sglang/srt/server_args.py (+0/-1); python/sglang/test/test_utils.py (+0/-5); test/srt/quant/test_autoround.py (+0/-62)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 6226 reason=ci_or_test_failure
BODY: This reverts commit c8295d235386db8b5c4c3571b6d963716d0bdce1. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-91f0fd95a4  (L1, 2025-09-08, sha 91f0fd95a476, PR #10166)
TITLE: pref: Add H20 fp8 fused MoE kernel configs for Qwen3 (#10166)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=128,N=768,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎ Add H20 fp8 fused MoE kernel tuning configs for Qwen3-MoE-A3B. ⏎  ⏎ ## Modifications ⏎  ⏎ Add H20 fp8 fused MoE kernel tuning configs for Qwen3-MoE-A3B. ⏎  ⏎  ⏎ ## Checklist

### L1-148022fc36  (L1, 2025-09-08, sha 148022fc36f0, PR #9522)
TITLE: gb200: update dockerfile to latest kernel (#9522)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+6/-9)
BODY: fp8 disagg working

### L1-df5407fb53  (L1, 2025-09-08, sha df5407fb53b8, PR #10185)
TITLE: Revert "feat: add fused moe config for Qwen3-30B-A3B on B200" (#10185)
SOURCES: path_config_only, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=768,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+0/-146)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 9087 reason=correctness_or_accuracy
BODY: This reverts commit 49d71f90ed90f3834356bd93c66c3cb381ffd831. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ With the triton configuration, Qwen3 return incorrect response when using triton moe backend. Let's revert the commit and retune the result. ⏎  ⏎ ## Modifications ⏎  ⏎ 1. revert the qwen3 fp8 triton tuning config. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ - With the config the Qwen3 30B FP8 model return incorrect response ⏎  ⏎ <img width="1662" height="574" alt="Screenshot 2025-09-08 a …[truncated]

### L1-94fb4e9e54  (L1, 2025-09-09, sha 94fb4e9e54ef, PR #10205)
TITLE: feat: support fa cute in sgl-kernel (#10205)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-0); sgl-kernel/CMakeLists.txt (+19/-0); sgl-kernel/python/sgl_kernel/_fa4_interface.py (+376/-0); sgl-kernel/python/sgl_kernel/flash_attn.py (+42/-0); sgl-kernel/tests/test_flash_attention_4.py (+877/-0)
BODY: ## Motivation ⏎  ⏎ - extract fa cute sgl-kernel part from https://github.com/sgl-project/sglang/pull/9928, authored by @cicirori  ⏎ - also unify interface for https://github.com/sgl-project/sglang/pull/9428 ⏎  ⏎ After this pull request, a new sgl-kernel release will be updated. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d3ee70985f  (L1, 2025-09-09, sha d3ee70985f3a, PR #10220)
TITLE: chore: upgrade v0.3.9 sgl-kernel (#10220)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+2/-5); docker/Dockerfile.gb200 (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); scripts/ci/ci_install_dependency.sh (+1/-1)
DEEP_STUDY: deep-study: this PR was reverted by PR 10245 (confirmed_revert, reason=ci_or_test_failure)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d352c29aa0  (L1, 2025-09-09, sha d352c29aa09f, PR #10210)
TITLE: Revert the changes on NCCL symmetric memory (#10210)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-4); python/sglang/srt/distributed/parallel_state.py (+0/-11); python/sglang/srt/layers/linear.py (+1/-7); python/sglang/srt/layers/vocab_parallel_embedding.py (+3/-7); python/sglang/srt/models/deepseek_v2.py (+3/-14)
DEEP_STUDY: deep-study revert record: partial_revert of PR(s) 8238 reason=premature_or_process
BODY: Revert the changes of model forward code in https://github.com/sgl-project/sglang/pull/8238. ⏎ We will do a clean version in https://github.com/sgl-project/sglang/pull/9358

### L1-4582931ac3  (L1, 2025-09-09, sha 4582931ac3f7, PR #10238)
TITLE: Revert "Revert the changes on NCCL symmetric memory" (#10238)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-0); python/sglang/srt/distributed/parallel_state.py (+11/-0); python/sglang/srt/layers/linear.py (+7/-1); python/sglang/srt/layers/vocab_parallel_embedding.py (+7/-3); python/sglang/srt/models/deepseek_v2.py (+14/-3)
DEEP_STUDY: deep-study revert record: reland of PR(s) 8238 reason=unstated
BODY: Reverts sgl-project/sglang#10210

### L1-bcf1955f7e  (L1, 2025-09-09, sha bcf1955f7e42, PR #10245)
TITLE: Revert "chore: upgrade v0.3.9 sgl-kernel" (#10245)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+5/-2); docker/Dockerfile.gb200 (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); scripts/ci/ci_install_dependency.sh (+1/-1)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 10220 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#10220 ⏎  ⏎ because of the failed test case https://github.com/sgl-project/sglang/actions/runs/17579411785/job/49948124294#step:4:2936

### L1-2286e85e77  (L1, 2025-09-10, sha 2286e85e7758, PR #10241)
TITLE: pass a_scale from fp8 quant result instead of hard code to 1.0f  (#10241)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.w4a8, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+3/-3); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh (+1/-1); sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py (+30/-25)
BODY: coauthor with @yicwang  @ayrnb  ⏎  ⏎ ## Motivation ⏎  ⏎ Issue: #10215  ⏎  ⏎ Fix the w4afp8 test case failures on hopper ⏎  ⏎ ## Modifications ⏎  ⏎ - Fix the w4afp8 kernel pass the quantization a_scale instead of hard coding to 1.0f ⏎ - Fix the failed test cases ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ - Unit tests ⏎  ⏎ ```bash ⏎  python test_cutlass_w4a8_moe_mm.py ⏎ ======================================================================= test session starts ===================== …[truncated]

### L1-5b64f006ec  (L1, 2025-09-10, sha 5b64f006ec2e, PR #9881)
TITLE: [Feature] Support DeepEP normal & Redundant Experts on NPU (#9881)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.routing.topk_py, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/eplb/eplb_manager.py (+2/-2); python/sglang/srt/eplb/expert_distribution.py (+12/-4); python/sglang/srt/eplb/expert_location_updater.py (+1/-1); python/sglang/srt/layers/moe/ep_moe/layer.py (+108/-48); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+0/-2); python/sglang/srt/layers/moe/token_dispatcher/base.py (+0/-11); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+8/-35); python/sglang/srt/layers/moe/topk.py (+8/-0); .github/workflows/pr-test-npu.yml (+36/-0); .github/workflows/release-docker-npu-nightly.yml (+1/-0); (+5 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR adds support on Ascend NPU for: ⏎  ⏎ 1. DeepEP normal mode ⏎ 2. Redundant Experts ⏎  ⏎ Along with previously merged #8355, we are now allowing both prefill and decode to run with expert parallelism on Altas 800I A3. This also means running large-scale moe models without PD disaggregation is also possible if HBM capacity allows. ⏎  ⏎ Checkout our roadmap [here](https://github.com/sgl-project/sgl-kernel-npu/issues/47) about De …[truncated]

### L1-37367da639  (L1, 2025-09-10, sha 37367da6390f, PR #10299)
TITLE: [fix CI] Fix logical condition in fused MoE layer for compressed tensor quantization (#10299)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-2)
LABELS: high priority
BODY: ### Background ⏎  ⏎ The bug is caused by https://github.com/sgl-project/sglang/pull/8118 cc @chenxijun1029  ⏎  ⏎ The current code in `fused_moe_triton/layer.py` has a logical operator precedence issue in the condition check for input scales validation (lines 615-620). The problematic condition was: ⏎  ⏎ ```python ⏎ if ( ⏎     "compressed" in self.quant_method.__class__.__name__.lower() ⏎     or "w4afp8" in self.quant_config.get_name() ⏎     and (param.data …[truncated]

### L1-bfe01a5eef  (L1, 2025-09-11, sha bfe01a5eef40, PR #10297)
TITLE: chore: upgrade v0.3.9.post2 sgl-kernel (#10297)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+2/-5); docker/Dockerfile.gb200 (+1/-1); python/pyproject.toml (+2/-2); .github/workflows/pr-test-pd-router.yml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); scripts/ci/ci_install_dependency.sh (+3/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-30c6e1f569  (L1, 2025-09-11, sha 30c6e1f56967, PR #10233)
TITLE: Qwen3-Next support (#10233)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=512,N=128,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=512,N=64,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/model_config.py (+3/-0); python/sglang/srt/configs/qwen3_next.py (+326/-0); python/sglang/srt/hf_transformers_utils.py (+2/-0); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+581/-0); python/sglang/srt/layers/attention/mamba/causal_conv1d.py (+128/-0); python/sglang/srt/layers/attention/mamba/mamba.py (+64/-0); python/sglang/srt/managers/schedule_batch.py (+8/-5); (+9 more)
LABELS: high priority
BODY: ## Motivation ⏎ ref #10306 ⏎ support qwen3-next/qwen3-next-mtp ⏎  ⏎ ## Modifications ⏎ 1. add `MambaPool` / `HybridReqTokenPool` to allocate mamba cache ⏎ 2. add `HybridLinearKVPool` to avoid kv cache allocation in linear layers  ⏎ 3. add hybrid linear attention backend ⏎ 4. support qwen3-next basic model ⏎ 5. support qwen3-next mtp / use `MambaStateUpdateCudaGraphRunner` to accelerate update mamba/conv state in verify stage ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ `` …[truncated]

### L1-ab795ae840  (L1, 2025-09-11, sha ab795ae84089, PR #10264)
TITLE: add h20 qwen3 next config (#10264)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=512,N=128,device_name=NVIDIA_H20-3e.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=512,N=256,device_name=NVIDIA_H20-3e.json (+146/-0)
BODY: ## Motivation ⏎  ⏎  ⏎ need more configs ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-c5d2b01cea  (L1, 2025-09-11, sha c5d2b01cea6c, PR #10303)
TITLE: [LongCat] Optimize zero_experts_compute_triton by changing mask (#10303)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+1/-1)
BODY: update moe expert mask @Orchard-DT . Sincerely invite you to help check. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ Regarding the zero expert proposed by longcat_flash, we found that there may be an optimization when masking the expert. The original masking of the ID to 0 may cause confusion with the expert whose ID was originally 0. We modified this and found that the throughput of our model in the prefill phase increased by 10%. ⏎  ⏎ ## Modifications ⏎  ⏎ modify the  …[truncated]

### L1-3df05f4d6a  (L1, 2025-09-11, sha 3df05f4d6ab8, PR #9199)
TITLE: [NVIDIA] [3/N] Nvfp4 Masked Gemm: Add flashinfer grouped_gemm_nt_masked  (#9199)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.runner.flashinfer_cutedsl, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+18/-0); python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py (+156/-0); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+2/-1); python/sglang/srt/layers/moe/utils.py (+4/-0); docs/references/environment_variables.md (+5/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+41/-1); python/sglang/srt/models/deepseek_v2.py (+6/-2); python/sglang/srt/server_args.py (+12/-0); python/sglang/test/test_fp4_moe.py (+370/-1); python/sglang/test/test_utils.py (+3/-0); (+1 more)
BODY: @kaixih @kushanam @fzyzcjy ⏎  ⏎ ## Motivation ⏎  ⏎ Add  [grouped_gemm_nt_masked](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/cute_dsl/blockscaled_gemm.py#L2708) from flashinfer to support nvfp4 MoE. This PR exposes 2 APIs: `flashinfer_cutedsl_grouped_gemm_nt_masked` and `flashinfer_cutedsl_grouped_gemm_nt_masked`. ⏎ Depends on [9200](https://github.com/sgl-project/sglang/pull/9200/files) ⏎ The next step is to integrate into EpMoE.  …[truncated]

### L1-c7e85f5378  (L1, 2025-09-11, sha c7e85f537870, PR #10296)
TITLE: fix: flashinfer_cutlass_moe: Use max of global expert scales instead of local for input scale (#10296)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+7/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+2/-2)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ To fix accuracy issues. ⏎  ⏎ ## Modifications ⏎  ⏎ This matches trt-llm usage ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ SGL_ENABLE_JIT_DEEPGEMM=0 python -m sglang.launch_server   --max-running-requests 1024   --disable-radix-cache   --disable-shared-experts-fusion   --tp-size 8   --dp-size 8   --ep-size 8   --enable-dp-attention   --chunked-prefill-size $((4096 * 8))   --moe-dense-tp-size 1   --enable-dp-lm-head   --model-path nvidia/DeepSeek-R1- …[truncated]

### L1-4aa39d72c4  (L1, 2025-09-11, sha 4aa39d72c422, PR #10356)
TITLE: fix the break in FlashInferFusedMoE (#10356)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-2)
BODY: This PR fixed the break due to the recent changes to API apply_with_router_logits() in https://github.com/sgl-project/sglang/pull/9269.

### L1-fac07c9b08  (L1, 2025-09-11, sha fac07c9b08fd, PR #10359)
TITLE: Support LingV2 model (#10359)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_H20.json (+146/-0); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+11/-2); python/sglang/srt/configs/model_config.py (+5/-0); python/sglang/srt/layers/linear.py (+32/-0); python/sglang/srt/models/bailing_moe.py (+795/-218); python/sglang/srt/models/bailing_moe_nextn.py (+168/-0); python/sglang/srt/server_args.py (+8/-1)
BODY: ## Motivation ⏎  ⏎  ⏎ This is used for support Bailing models, Ling-2.0 and Ring-2.0, https://huggingface.co/inclusionAI/Ling-mini-2.0 . ⏎ More information can be found [here](https://github.com/inclusionAI/Ling) ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3a77c80b26  (L1, 2025-09-12, sha 3a77c80b26db, PR #10368)
TITLE: Fix FA4 import cause moe_fused_gate output be illegal memory (#10368)
SOURCES: subject_keyword, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: -
FILES: sgl-kernel/python/sgl_kernel/flash_attn.py (+2/-8)
DEEP_STUDY: deep-study: this PR was reverted by PR 10432 (confirmed_revert, reason=premature_or_process)
BODY: ## Motivation ⏎  ⏎ before fix: error, cannot use sglang in my scenario ⏎  ⏎ [details omitted] ⏎  ⏎ after fix: no error ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-cef11e9a55  (L1, 2025-09-12, sha cef11e9a552c, PR #10343)
TITLE: fix: resolve gb200 image link (#10343)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+10/-1)
LABELS: bug
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2f173ea074  (L1, 2025-09-12, sha 2f173ea0744d, PR #10244)
TITLE: [router] allow one router to support different model families and serving mode (#10244)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+3/-0); sgl-router/py_src/sglang_router/router_args.py (+6/-0); sgl-router/py_test/e2e/conftest.py (+3/-0); sgl-router/py_test/integration/test_retries.py (+1/-1); sgl-router/src/core/mod.rs (+2/-0); sgl-router/src/core/worker.rs (+105/-5); sgl-router/src/core/worker_registry.rs (+526/-0); sgl-router/src/policies/cache_aware.rs (+185/-57); sgl-router/src/policies/mod.rs (+12/-8); sgl-router/src/policies/power_of_two.rs (+8/-8); (+18 more)
LABELS: enhancement, high priority, feature, router
BODY: ## Summary ⏎ This PR implements multi-model routing support for SGLang Router, enabling efficient management of multiple models through centralized worker and policy registries. The implementation allows different models to use different load balancing policies dynamically, with automatic lifecycle management and cleanup. ⏎  ⏎ ## High-Level Design and Goals ⏎  ⏎ ### Primary Goals ⏎ 1. **Multi-Model Support**: Enable routing to different models with mod …[truncated]

### L1-16cd550c85  (L1, 2025-09-12, sha 16cd550c8554, PR #10379)
TITLE: Support Qwen3-Next on Ascend NPU (#10379)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.npu (+8/-3); .github/workflows/release-docker-npu-nightly.yml (+1/-1); .github/workflows/release-docker-npu.yml (+1/-1); python/sglang/srt/layers/attention/fla/layernorm_gated.py (+1/-1); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+22/-4); python/sglang/srt/mem_cache/memory_pool.py (+12/-3); python/sglang/srt/model_executor/model_runner.py (+16/-5); python/sglang/srt/models/qwen3_next.py (+6/-3); python/sglang/srt/server_args.py (+2/-1); scripts/ci/npu_ci_install_dependency.sh (+10/-4)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b3c977622f  (L1, 2025-09-13, sha b3c977622fa1, PR #10404)
TITLE: Add h200 fused moe config for Qwen3-Next (#10404)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=512,N=128,device_name=NVIDIA_H200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=512,N=256,device_name=NVIDIA_H200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=512,N=64,device_name=NVIDIA_H200.json (+146/-0)
BODY: ## Motivation ⏎ H200 TP 2/4/8 for Qwen3-Next  fused moe config. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎ **Cmd** ⏎ ``` ⏎ python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 100 --max-concurrency 8 --dataset-name random --random-input-len 1000 --random-output-len 2000 --random-range-ratio 1 --warmup-requests 5 ⏎ ``` ⏎ **Result** ⏎ TP Size|Input-Output Length | Before Output Token Throughput (tok …[truncated]

### L1-2df532ef20  (L1, 2025-09-14, sha 2df532ef20cb, PR #10369)
TITLE: Fix the global scale fix does not support EPLB and improve enabling condition (#10369)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-10); python/sglang/srt/layers/quantization/modelopt_quant.py (+2/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-fa46e2bd40  (L1, 2025-09-14, sha fa46e2bd4003, PR #9948)
TITLE: Support offloading in fp8 (#9948)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+52/-10); python/sglang/srt/layers/quantization/fp8_utils.py (+7/-2); python/sglang/srt/models/deepseek_v2.py (+9/-2); python/sglang/srt/offloader.py (+27/-3)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ (multi code change to be extracted) ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2f8ba6fe82  (L1, 2025-09-14, sha 2f8ba6fe82be, PR #10429)
TITLE: [Fix] MoE: fix w8a8_fp8 MoE and add tests to cover this code path (#10429)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/w8a8_fp8.py (+1/-1); test/srt/quant/test_w8a8_quantization.py (+43/-7)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR fixes the issue mentioned in [this comment](https://github.com/sgl-project/sglang/pull/9269#discussion_r2342544472). It also adds two cases to cover E2E w8a8_fp8 test. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-258d02c86d  (L1, 2025-09-14, sha 258d02c86d93, PR #10426)
TITLE: Fix correction bias undefined behavior for nvfp4 models (#10426)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: sgl-kernel/csrc/moe/moe_fused_gate.cu (+2/-0); python/sglang/srt/models/deepseek_v2.py (+3/-1)
BODY: ## Motivation ⏎  ⏎ TODO tomorrow: (1) add assertions at kernel level (2) check shall we make it bf16 etc (3) test e2e ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-ca63f075b7  (L1, 2025-09-14, sha ca63f075b7d8, PR #10432)
TITLE: Revert "Fix FA4 import cause moe_fused_gate output be illegal memory" (#10432)
SOURCES: subject_keyword, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: -
FILES: sgl-kernel/python/sgl_kernel/flash_attn.py (+8/-2)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 10368 reason=premature_or_process
BODY: Reverts sgl-project/sglang#10368 ⏎  ⏎ DO NOT MERGE now, wait for double check from https://github.com/sgl-project/sglang/pull/10426

### L1-4844fac91d  (L1, 2025-09-14, sha 4844fac91d07, PR #9338)
TITLE: Refactor TopK to ensure readability and extensibility (#9338)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+4/-4); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-10); python/sglang/srt/layers/moe/topk.py (+30/-9); python/sglang/srt/managers/schedule_batch.py (+0/-1); python/sglang/srt/models/bailing_moe.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+7/-12); python/sglang/srt/models/ernie4.py (+1/-1); python/sglang/srt/models/glm4_moe.py (+1/-1); python/sglang/srt/models/gpt_oss.py (+1/-1); python/sglang/srt/models/longcat_flash.py (+2/-2); (+4 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Some recent fixes and optimizations are hardcoded in `deepseek_v2.py`. This PR slightly adjust the code structure and rename some variables to ensure readability and extensibility. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-010181388c  (L1, 2025-09-14, sha 010181388cd1, PR #10437)
TITLE: Tiny fix wrong naming (#10437)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+2/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-76becc1dbc  (L1, 2025-09-15, sha 76becc1dbc9f, PR #10439)
TITLE: Add rtx5880 moe triton (#10439)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_RTX_5880_Ada_Generation,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py (+1/-1); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+2/-2)
LABELS: ready-to-merge
BODY: ## Motivation ⏎  ⏎ This PR continues the work from #10167 ⏎ Changes: ⏎ - run `pre-commit run --all-files` ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-059c13de5c  (L1, 2025-09-15, sha 059c13de5cba, PR #10440)
TITLE: Fix trtllm_moe wrong correction bias (#10440)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+0/-8); python/sglang/srt/models/deepseek_v2.py (+14/-2)
LABELS: run-ci
DEEP_STUDY: deep-study correctness case sglang:059c13de5c: class=memory_safety_oob; symptom=nan_inf; introducing=unknown
BODY: ## Motivation ⏎  ⏎ before fix: comes from torch.empty, can be zero or even nan ⏎  ⏎ <img width="1740" height="410" alt="image" src="https://github.com/user-attachments/assets/94d11287-0447-4ef1-8b71-88a8257ddafd" /> ⏎  ⏎ after fix: ⏎  ⏎ <img width="1143" height="305" alt="image" src="https://github.com/user-attachments/assets/10c57390-0f09-4ece-a02a-6cd8064a0e98" /> ⏎  ⏎ (trtllm_gen requires bf16 dtype, though we know it is good to be fp32 since the value  …[truncated]

### L1-5afd036533  (L1, 2025-09-15, sha 5afd0365334c, PR #10465)
TITLE: feat: support pip install sglang (#10465)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile.npu (+1/-1); docker/Dockerfile.rocm (+1/-0); docker/Dockerfile.xeon (+1/-0); python/pyproject.toml (+88/-132); .github/workflows/pr-test-xeon.yml (+1/-0); python/pyproject_other.toml (+174/-0); scripts/ci/amd_ci_install_dependency.sh (+2/-0); scripts/ci/npu_ci_install_dependency.sh (+1/-0)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎ - We should support `pip install sglang` on NVIDIA GPU. ⏎ - If we don't back up `pyproject_other.toml`, other platforms will encounter errors during installation. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-57234d0c9c  (L1, 2025-09-15, sha 57234d0c9c32, PR #10471)
TITLE: [bugfix] fix typo (#10471)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+1/-1)
BODY: 

### L1-c0c6f543e4  (L1, 2025-09-16, sha c0c6f543e472, PR #10500)
TITLE: chore: upgrade sgl-kernel 0.3.10 (#10500)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); docker/Dockerfile.gb200 (+1/-1); python/pyproject.toml (+4/-4); python/pyproject_other.toml (+3/-3); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9b876889b7  (L1, 2025-09-16, sha 9b876889b7d1, PR #10491)
TITLE: Update CUTLASS. Refine KernelSchedule for fp8 (grouped) gemm. (#10491)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+3/-3); sgl-kernel/CMakeLists.txt (+1/-1); sgl-kernel/csrc/cutlass_extensions/gemm/fp8_blockwise_gemm_sm90_dispatch.cuh (+1/-1)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎ Updated CUTLASS to tag version 4.2. Aligned the fp8 (grouped) gemm kernel schedule to CUTLASS Example: ⏎  ⏎ - https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling ⏎ - https://github.com/NVIDIA/cutlass/tree/main/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - sgl-kernel/CMakeLists.txt ⏎ - sgl-kernel/csrc/cutlass_extensions …[truncated]

### L1-e1d45bc280  (L1, 2025-09-16, sha e1d45bc280e4, PR #10529)
TITLE: Fix decord dependency for aarch64 docker build (#10529)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile.gb200 (+16/-4); python/pyproject.toml (+2/-1); .github/workflows/release-docker-gb200.yml (+1/-1)
BODY: ## Motivation ⏎  ⏎ `pyproject.toml` was modified to include `decord` as a dependency. ⏎ However, `decord` does not have an ARM package, hence it's breaking `Dockerfile.gb200` build. ⏎  ⏎ ## Modifications ⏎  ⏎ This PR creates a new `blackwell_aarch64` build target, and excludes `decord` dependency for that build target. It also adds support for BRANCH_TYPE argument in `Dockerfile.gb200. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ This is only a CI change. No impact on accur …[truncated]

### L1-e07b21ceaf  (L1, 2025-09-18, sha e07b21ceaf1b, PR #10624)
TITLE: update deepep version for qwen3-next deepep moe (#10624)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: scripts/ci/ci_install_deepep.sh (+1/-1); docker/Dockerfile (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Support qwen3-next expert number = 512 since https://github.com/deepseek-ai/DeepEP/pull/403 ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1344ebc833  (L1, 2025-09-18, sha 1344ebc8333d, PR #10622)
TITLE: support qwen3-next-fp8 deepep (#10622)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen2_moe.py (+64/-1); python/sglang/srt/models/qwen3_next.py (+29/-8)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ need merge https://github.com/sgl-project/sglang/pull/10624 first ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ ``` bash ⏎ python3 -m sglang.launch_server --model Qwen/Qwen-Next-80B-A3B-Instruct-FP8/  --tp 4 --dp 2 --enable-dp-attention --enable-deepep-moe --cuda-graph-max-bs 128 ⏎  ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 1000  ⏎ Accuracy: 0.942 ⏎ Invalid: 0.000 ⏎ Latency: 176.143 s ⏎ Output throughput: 943.616 token/s …[truncated]

### L1-388c05d544  (L1, 2025-09-18, sha 388c05d54435, PR #10579)
TITLE: Fix bias handling in TritonMoeQuantInfo within quantization/mxfp4.py (#10579)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/mxfp4.py (+2/-2)
LABELS: ready-to-merge, run-ci
BODY: ## Motivation ⏎  ⏎ This PR fixes a field mismatch when constructing `TritonMoeQuantInfo` in `quantization/mxfp4.py`. ⏎ - Issue: The call site passed `w13_weight_bias` and `w2_weight_bias`, but `TritonMoeQuantInfo` defines these as optional fields `b13` and `b2`. This could raise a TypeError and/or leave biases unwired. ⏎ - Fix: Pass biases via `b13` and `b2`, sourcing them from the layer with `getattr(..., None)` to safely handle layers without bias  …[truncated]

### L1-8c52de6fab  (L1, 2025-09-18, sha 8c52de6faba2, PR #10631)
TITLE: feat: add fused moe config for Qwen3-Next-80B-A3B-Instruct on B200 (#10631)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=512,N=256,device_name=NVIDIA_B200.json (+146/-0)
BODY: ## Motivation ⏎  ⏎ Add fused MoE config for Qwen3-Next-80B-A3B-Instruct on NVIDIA B200 ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ With config: ⏎ <img width="1274" height="156" alt="Screenshot 2025-09-18 at 4 54 21 PM" src="https://github.com/user-attachments/assets/5b891240-00fe-42b6-a382-54ff4d95f7c5" /> ⏎ Without config: ⏎ <img width="1280" height="153" alt="Screenshot 2025-09-18 at 4 58 09 PM" src="https://github.com/user-attachments/assets/1094eeb1-9fc4-45f4-93cc- …[truncated]

### L1-4039c626e2  (L1, 2025-09-19, sha 4039c626e27a, PR #8274)
TITLE: fix deepep assert when PD disaggregation == null (#8274)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+11/-2)
BODY: when `--pd-disaggregation` == `null` deepep will assert:  ⏎  ⏎ https://github.com/deepseek-ai/DeepEP/blob/bdd119f8b249953cab366f4d737ad39d4246fd7e/csrc/kernels/internode.cu#L386 ⏎  ⏎ deepep PR: https://github.com/deepseek-ai/DeepEP/pull/181 ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3fa3c22ae2  (L1, 2025-09-19, sha 3fa3c22ae2c4, PR #10634)
TITLE: Fix fast decode plan for flashinfer v0.4.0rc1 and upgrade sgl-kernel 0.3.11 (#10634)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/pyproject_other.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+2/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+3/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ In #10587, updating flashinfer to v0.4.0rc1 will cause some bugs, which is caused by some extra arguments added in https://github.com/flashinfer-ai/flashinfer/pull/1661 and  https://github.com/flashinfer-ai/flashinfer/pull/1675. ⏎  ⏎ To solve this, we just need to pass the default values of these extra arguments in `fast_decode_plan` of flashinfer backend. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and  …[truncated]

### L1-616a3e20df  (L1, 2025-09-19, sha 616a3e20df5f, PR #10321)
TITLE: [sgl-kernel] Support moe_sum_reduce cuda kernel (#10321)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.reduce.moe_sum_reduce
FILES: sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/csrc/common_extension.cc (+2/-0); sgl-kernel/csrc/moe/moe_sum_reduce.cu (+303/-0); sgl-kernel/include/sgl_kernel_ops.h (+2/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/python/sgl_kernel/moe.py (+12/-0); benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py (+25/-10)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ moe_sum_reduce kernel in MoE is a Triton kernel (combine with TorchCompile version for small batch). This PR is to rewrite it in CUDA for better performance. ⏎  ⏎ Currently the CudaKernel outperforms both TorchCompile and TritonKernel in small num_tokens. In large token_num the performs has not drop. ⏎ ``` ⏎ root@blackwell-research:/sgl-workspace/sglang_dev# python ./benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py ⏎ Runnin …[truncated]

### L1-6f993e8b9e  (L1, 2025-09-19, sha 6f993e8b9e6c, PR #10671)
TITLE: chore: cleanup docker image (#10671)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+4/-3); .github/workflows/release-docker-dev.yml (+1/-7); .github/workflows/release-docker.yml (+2/-10)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ `lmsysorg/sglang:dev` -> cu129, all ⏎ `lmsysorg/sglang:v0.5.3` -> release version, all ⏎ `lmsysorg/sglang:v0.5.3-cu126` -> hopper version, all (It should also works on hopper cu128) ⏎  ⏎ We should remove `lmsysorg/sglang:v0.5.3-cu126` in the near future. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9c53dad809  (L1, 2025-09-22, sha 9c53dad80993, PR #10758)
TITLE: Fix MTP MoE weight loading with NVFP4 target model. (#10758)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-1)
LABELS: bug, high priority, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-16adf3dcab  (L1, 2025-09-22, sha 16adf3dcabaf, PR #10774)
TITLE: [router] fix logger type mismatch (#10774)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+1/-1); sgl-router/py_src/sglang_router/router_args.py (+1/-1)
LABELS: bug, router, run-ci
BODY: fix logger from warning to warn, since rust expects `warn` instead of `warning` ⏎  ⏎ ## Checklist

### L1-1c82d9db28  (L1, 2025-09-22, sha 1c82d9db2846, PR #10705)
TITLE: feat: unify dockerfiles (#10705)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+38/-24); .github/workflows/release-docker-dev.yml (+110/-4); .github/workflows/release-docker-gb200.yml (+0/-5); .github/workflows/release-docker.yml (+61/-28); .github/workflows/release-whl-kernel.yml (+7/-1)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎ SGLang should be the best across all platforms!! This PR unifies the current Dockerfile and Dockerfile.gb200 to stem from the same file. This simplifies maintenance for version bumps and makes CI/release easier. ⏎  ⏎ In a follow up PR we should ⏎ 1. test the sgl-kernel release -> pypi directly ⏎ 2. update the dockerfile to pip install it vs installing from the repo ⏎ 3. delete the gb200 dockerfile ⏎  ⏎  ⏎ # Tests

### L1-8c1ef0f914  (L1, 2025-09-23, sha 8c1ef0f91444, PR #10782)
TITLE: chore: upgrade sgl-kernel 0.3.12 (#10782)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/pyproject_other.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b06db198ba  (L1, 2025-09-23, sha b06db198ba9e, PR #10783)
TITLE: followup: clean up dockerfiles and release yamls  (#10783)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+0/-369); .github/workflows/release-docker-gb200.yml (+0/-31)
LABELS: run-ci
BODY: 

### L1-ea338676b5  (L1, 2025-09-23, sha ea338676b502, PR #10770)
TITLE: Clean up server args (#10770)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+190/-238)
LABELS: run-ci
BODY: - Do not use "Step 1", "Step 2", ... , as it will need lots of changes if we insert something in the middle ⏎ - rename `model_specific_adjustments` -> `_handle_model_specific_adjustments` ⏎ - Sort the names and functions better ⏎ - Deprecate old arguments `enable_ep_moe`, `enable_deepep_moe`, ..

### L1-984730b732  (L1, 2025-09-23, sha 984730b732aa, PR #10794)
TITLE: add tunning files for QWEN-3-NEXT (#10794)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=256,device_name=NVIDIA_H800,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=512,N=128,device_name=NVIDIA_H800,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+5/-1)
BODY: ## Motivation ⏎  ⏎  ⏎ Tunning files are missing for H800 ... ⏎  ⏎ See the log: ⏎ [serv_qwen3-next.log](https://github.com/user-attachments/files/22490492/serv_qwen3-next.log) ⏎  ⏎ After apply the patch, you should see when serving QWEN-3-NEXT: ⏎  ⏎ > [2025-09-23 02:24:02 TP0] Using MoE kernel config from /home/yiakwy/workspace/Github/sglang-gold/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=512,N=128,device_name=NVIDIA_H800,dtype=fp8 …[truncated]

### L1-adba172fd1  (L1, 2025-09-24, sha adba172fd155, PR #10786)
TITLE: ci: free space on workers for build (#10786)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); .github/workflows/release-docker-dev.yml (+19/-58); .github/workflows/release-docker.yml (+2/-2)
LABELS: run-ci
BODY: WIP work with @zhyncs to wire up x86 runner ⏎  ⏎ After swapping to custom aarch64 runner - builds are fast and complete (https://github.com/sgl-project/sglang/actions/runs/17965314447/job/51096758220)

### L1-592ddf374f  (L1, 2025-09-26, sha 592ddf374fe4, PR #10944)
TITLE: Add simple docker file for B300 (#10944)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.b300 (+55/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Add simple docker file for B300. It uses the ngc pytorch container as the base because the pytorch release doesn't have B300 support yet. ⏎ DeepEP, DeepGemm etc will be enabled in the future. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Tested sgl-kernel tests. I'll do more e2e tests.  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist
